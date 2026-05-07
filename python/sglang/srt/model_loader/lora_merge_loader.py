import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.weight_utils import (
    get_actual_shard_size,
    narrow_padded_param_and_loaded_weight,
)
from sglang.srt.model_loader.lora_merge.options import (
    prestage_request_id,
    resolve_lora_merge_options,
)
from sglang.srt.model_loader.lora_merge.flashinfer_trtllm import (
    finalize_flashinfer_moe_layer_after_merge as _finalize_flashinfer_moe_layer_after_merge,
    get_flashinfer_moe_layer_for_postprocess as _get_flashinfer_moe_layer_for_postprocess,
    try_apply_flashinfer_trtllm_moe_lora_op_cuda_bucketed as _try_apply_flashinfer_trtllm_moe_lora_op_cuda_bucketed,
)
from sglang.srt.utils import is_cpu

logger = logging.getLogger(__name__)

_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_EXPLICIT_EXPERT_RE = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<expert_id>\d+)\.(?P<target>[^.]+)$"
)


@dataclass(frozen=True)
class _AliasRule:
    source_suffix: str
    dest_suffix: str
    loader_args: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class _DenseLoraOp:
    param_name: str
    lora_a: torch.Tensor
    lora_b: torch.Tensor
    loader_args: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class _AggregateExpertLoraOp:
    experts_prefix: str
    target: str
    lora_a: torch.Tensor
    lora_b: torch.Tensor


@dataclass(frozen=True)
class _LoraPair:
    base_name: str
    lora_a: torch.Tensor
    lora_b: torch.Tensor
    prestaged: bool = False


@dataclass(frozen=True)
class _DenseLocalApplySpec:
    dst_view: torch.Tensor
    lora_b_start: int
    lora_b_rows: int
    lora_a_start: int
    lora_a_cols: int


_ALIAS_RULES = (
    _AliasRule(".q_proj", ".qkv_proj", ("q",)),
    _AliasRule(".k_proj", ".qkv_proj", ("k",)),
    _AliasRule(".v_proj", ".qkv_proj", ("v",)),
    _AliasRule(".gate_proj", ".gate_up_proj", (0,)),
    _AliasRule(".up_proj", ".gate_up_proj", (1,)),
    _AliasRule(".linear_attn.in_proj_q", ".linear_attn.in_proj_qkvz", (0,)),
    _AliasRule(".linear_attn.in_proj_k", ".linear_attn.in_proj_qkvz", (1,)),
    _AliasRule(".linear_attn.in_proj_v", ".linear_attn.in_proj_qkvz", (2,)),
    _AliasRule(".linear_attn.in_proj_z", ".linear_attn.in_proj_qkvz", (3,)),
    _AliasRule(".linear_attn.in_proj_qkv", ".linear_attn.in_proj_qkvz", ((0, 1, 2),)),
    _AliasRule(".linear_attn.in_proj_b", ".linear_attn.in_proj_ba", (0,)),
    _AliasRule(".linear_attn.in_proj_a", ".linear_attn.in_proj_ba", (1,)),
)

_EXPERT_TARGET_ALIASES = {
    "w1": "w1",
    "gate_proj": "w1",
    "w3": "w3",
    "up_proj": "w3",
    "w2": "w2",
    "down_proj": "w2",
    "gate_up_proj": "gate_up_proj",
}

_PRESTAGED_LORA_CACHE_ATTR = "_sglang_lora_merge_prestaged_cache"


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _require_cuda_model_device(model: torch.nn.Module) -> torch.device:
    device = _model_device(model)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError(
            "Merged LoRA update requires a CUDA model. "
            f"Got model device {device}."
        )
    return device


def _collect_lora_pairs(
    named_tensors: List[Tuple[str, torch.Tensor]], strict: bool
) -> List[_LoraPair]:
    pending_pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    pairs: List[_LoraPair] = []
    for name, tensor in named_tensors:
        base_name, kind = _split_lora_tensor_name(name)
        if kind not in ("A", "B"):
            raise ValueError(f"Unsupported LoRA tensor name for merged update: {name}")

        base_name = _canonicalize_lora_base_name(base_name)
        pair = pending_pairs.setdefault(base_name, {})
        if kind in pair:
            raise ValueError(f"Duplicate LoRA tensor for {base_name}: {kind}")
        pair[kind] = tensor

        if "A" not in pair or "B" not in pair:
            continue

        pairs.append(_LoraPair(base_name, pair["A"], pair["B"], prestaged=False))
        del pending_pairs[base_name]

    if strict and pending_pairs:
        incomplete = ", ".join(sorted(pending_pairs.keys()))
        raise ValueError(f"Incomplete LoRA pairs for merged update: {incomplete}")
    return pairs


def _tensor_fp32_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * 4)


def _lora_pair_fp32_bytes(pair: _LoraPair) -> int:
    return _tensor_fp32_bytes(pair.lora_a) + _tensor_fp32_bytes(pair.lora_b)


def _stage_lora_pair_cuda_fp32(pair: _LoraPair, device: torch.device) -> _LoraPair:
    staged_a = pair.lora_a.to(device=device, dtype=torch.float32, non_blocking=True)
    staged_b = pair.lora_b.to(device=device, dtype=torch.float32, non_blocking=True)
    return _LoraPair(pair.base_name, staged_a, staged_b, prestaged=True)


def _estimate_lora_pair_apply_temp_bytes(
    pair: _LoraPair,
    params: Dict[str, torch.nn.Parameter],
    model: torch.nn.Module,
    strict: bool,
    bucket_bytes: int,
) -> int:
    try:
        ops = _resolve_delta_ops(
            base_name=pair.base_name,
            lora_a=pair.lora_a,
            lora_b=pair.lora_b,
            params=params,
            model=model,
        )
    except ValueError:
        if strict:
            raise
        return 0

    max_temp_bytes = 0
    for op in ops:
        if isinstance(op, _AggregateExpertLoraOp):
            temp_bytes = _estimate_aggregate_expert_lora_op_temp_bytes(
                op=op,
                params=params,
                model=model,
                strict=strict,
                bucket_bytes=bucket_bytes,
            )
        else:
            temp_bytes = _estimate_dense_lora_op_temp_bytes(
                op=op,
                params=params,
                model=model,
                strict=strict,
                bucket_bytes=bucket_bytes,
            )
        max_temp_bytes = max(max_temp_bytes, temp_bytes)
    return int(max_temp_bytes)


def _estimate_dense_lora_op_temp_bytes(
    op: _DenseLoraOp,
    params: Dict[str, torch.nn.Parameter],
    model: torch.nn.Module,
    strict: bool,
    bucket_bytes: int,
) -> int:
    param = params.get(op.param_name)
    if param is None:
        if strict:
            raise ValueError(
                f"Target parameter not found for merged LoRA update: {op.param_name}"
            )
        return 0
    owner_module = _resolve_param_owner_module(model, op.param_name, param)
    expert_temp_bytes = _estimate_explicit_expert_lora_op_temp_bytes(
        op=op,
        param=param,
        owner_module=owner_module,
        bucket_bytes=bucket_bytes,
    )
    if expert_temp_bytes is not None:
        return expert_temp_bytes
    shard_size = _vocab_lora_shard_size(op, param, owner_module)
    if shard_size is not None:
        if shard_size <= 0:
            return 0
        rank = int(op.lora_a.shape[0])
        cols = int(op.lora_a.shape[1])
        a_bytes = 4 * rank * cols
        bytes_per_row = 4 * (rank + 2 * cols)
        available_for_rows = max(0, bucket_bytes - a_bytes)
        rows_per_bucket = max(
            1, min(shard_size, available_for_rows // max(bytes_per_row, 1))
        )
        return _estimate_dense_temp_bytes(op.lora_a, op.lora_b, rows_per_bucket)
    specs = _resolve_dense_local_apply_specs(op, param, owner_module)
    if specs is not None:
        max_temp_bytes = 0
        rank = int(op.lora_a.shape[0])
        for spec in specs:
            if spec.lora_b_rows <= 0:
                continue
            rows_per_bucket = _dense_rows_per_bucket(
                rank=rank,
                cols=spec.lora_a_cols,
                total_rows=spec.lora_b_rows,
                bucket_bytes=bucket_bytes,
            )
            max_temp_bytes = max(
                max_temp_bytes,
                _estimate_dense_temp_bytes(
                    op.lora_a,
                    op.lora_b,
                    output_rows=rows_per_bucket,
                    input_cols=spec.lora_a_cols,
                ),
            )
        return int(max_temp_bytes)
    raise _unsupported_dense_bucketed_update(op, param, owner_module)


def _estimate_explicit_expert_lora_op_temp_bytes(
    op: _DenseLoraOp,
    param: torch.nn.Parameter,
    owner_module: Optional[torch.nn.Module],
    bucket_bytes: int,
) -> Optional[int]:
    if (
        owner_module is None
        or len(op.loader_args) != 3
        or not hasattr(owner_module, "w13_weight")
        or not hasattr(owner_module, "w2_weight")
        or op.lora_a.dim() != 2
        or op.lora_b.dim() != 2
    ):
        return None

    _, shard_id, _ = op.loader_args
    shard_id = str(shard_id)
    if shard_id in ("w1", "w3"):
        rows = int(op.lora_b.shape[0])
        cols = int(op.lora_a.shape[1])
    elif shard_id == "w2":
        rows = int(op.lora_b.shape[0])
        cols = int(op.lora_a.shape[1])
    else:
        raise ValueError(
            f"Unsupported expert shard id for merged update on {op.param_name}: {shard_id}"
        )

    if rows <= 0 or cols <= 0:
        return 0
    rank = int(op.lora_a.shape[0])
    rows_per_bucket = _dense_rows_per_bucket(
        rank=rank,
        cols=cols,
        total_rows=rows,
        bucket_bytes=bucket_bytes,
    )
    return _estimate_dense_temp_bytes(
        op.lora_a,
        op.lora_b,
        output_rows=rows_per_bucket,
        input_cols=cols,
    )


def _vocab_lora_shard_size(
    op: _DenseLoraOp,
    param: torch.nn.Parameter,
    owner_module: Optional[torch.nn.Module],
) -> Optional[int]:
    if op.loader_args:
        return None
    if not isinstance(owner_module, (ParallelLMHead, VocabParallelEmbedding)):
        return None
    if op.lora_a.dim() != 2 or op.lora_b.dim() != 2:
        return None

    output_dim = getattr(param, "output_dim", None)
    if output_dim not in (None, 0):
        return None

    if output_dim is None:
        source_start = 0
        shard_size = param.data.shape[0]
    elif owner_module.use_presharded_weights:
        source_start = 0
        shard_size = param.data.shape[0]
    else:
        source_start = owner_module.shard_indices.org_vocab_start_index
        shard_size = owner_module.shard_indices.org_vocab_end_index - source_start

    if source_start + shard_size > op.lora_b.shape[0]:
        shard_size = max(op.lora_b.shape[0] - source_start, 0)
    return int(shard_size)


def _estimate_aggregate_expert_lora_op_temp_bytes(
    op: _AggregateExpertLoraOp,
    params: Dict[str, torch.nn.Parameter],
    model: torch.nn.Module,
    strict: bool,
    bucket_bytes: int,
) -> int:
    param_name = _aggregate_expert_param_name(op.experts_prefix, op.target)
    param = params.get(param_name)
    if param is None:
        if strict:
            raise ValueError(
                f"Target parameter not found for merged LoRA update: {param_name}"
            )
        return 0
    owner_module = _resolve_param_owner_module(model, param_name, param)
    if owner_module is None:
        if strict:
            raise ValueError(f"Cannot resolve owner module for expert target {param_name}")
        return 0

    expert_count = max(_expert_axis_size(op.lora_a), _expert_axis_size(op.lora_b))
    quant_method = getattr(owner_module, "quant_method", None)
    use_unquantized_fused = isinstance(quant_method, UnquantizedFusedMoEMethod)
    if use_unquantized_fused:
        has_local_expert = any(
            _resolve_moe_expert_ids(owner_module, param, expert_id)
            for expert_id in range(expert_count)
        )
        if not has_local_expert:
            return 0
    elif expert_count == 0:
        return 0

    out_dim = int(op.lora_b.shape[-2])
    rank = int(op.lora_a.shape[-2])
    in_dim = int(op.lora_a.shape[-1])
    rows_per_bucket = _dense_rows_per_bucket(
        rank=rank,
        cols=in_dim,
        total_rows=out_dim,
        bucket_bytes=bucket_bytes,
    )
    return _estimate_dense_temp_bytes(
        op.lora_a,
        op.lora_b,
        output_rows=rows_per_bucket,
        input_cols=in_dim,
    )


def prepare_lora_tensors_for_merge(
    model: torch.nn.Module,
    named_tensors: List[Tuple[str, torch.Tensor]],
    load_context: Optional[Dict[str, Any]] = None,
) -> None:
    started_at = time.monotonic()
    load_context = load_context or {}
    manifest = load_context.get("manifest") or {}
    trace = load_context.get("trace")
    model_device = _model_device(model)
    if model_device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("LoRA merge prestage requires a CUDA model.")
    options = resolve_lora_merge_options(
        manifest,
        device=model_device,
        include_scaling=False,
    )
    request_id = options.prestage_request_id or prestage_request_id(options.manifest)
    strict = options.strict

    pairs = _collect_lora_pairs(named_tensors, strict=strict)
    params = dict(model.named_parameters(remove_duplicate=False))
    memory_budget = options.memory_budget
    prestage_bucket_bytes = options.apply_bucket_bytes
    if trace is not None:
        trace["lora_loader_prestage_pair_count"] = len(pairs)
        trace["lora_loader_prestage_peak_device_budget_bytes"] = int(
            memory_budget.peak_bytes
        )
        trace["lora_loader_prestage_gpu_bucket_bytes"] = int(prestage_bucket_bytes)
    max_apply_temp_bytes = 0
    for pair in pairs:
        max_apply_temp_bytes = max(
            max_apply_temp_bytes,
            _estimate_lora_pair_apply_temp_bytes(
                pair=pair,
                params=params,
                model=model,
                strict=strict,
                bucket_bytes=prestage_bucket_bytes,
            ),
        )

    prestage_capacity_bytes = max(0, memory_budget.peak_bytes - max_apply_temp_bytes)
    cached_pairs: List[_LoraPair] = []
    staged_bytes = 0
    unstaged_bytes = 0
    staged_count = 0
    unstaged_count = 0
    for pair in pairs:
        pair_staged_bytes = _lora_pair_fp32_bytes(pair)
        if staged_bytes + pair_staged_bytes <= prestage_capacity_bytes:
            staged_pair = _stage_lora_pair_cuda_fp32(pair, model_device)
            cached_pairs.append(staged_pair)
            staged_bytes += pair_staged_bytes
            staged_count += 1
        else:
            cached_pairs.append(pair)
            unstaged_bytes += pair_staged_bytes
            unstaged_count += 1

    cache = getattr(model, _PRESTAGED_LORA_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(model, _PRESTAGED_LORA_CACHE_ATTR, cache)
    cache[request_id] = {
        "pairs": cached_pairs,
        "gpu_bucket_bytes": int(prestage_bucket_bytes),
    }
    if trace is not None:
        trace["lora_loader_prestage_staged_pair_count"] = staged_count
        trace["lora_loader_prestage_unstaged_pair_count"] = unstaged_count
        trace["lora_loader_prestage_staged_bytes"] = int(staged_bytes)
        trace["lora_loader_prestage_unstaged_bytes"] = int(unstaged_bytes)
        trace["lora_loader_prestage_max_apply_temp_bytes"] = int(max_apply_temp_bytes)
        trace["lora_loader_prestage_capacity_bytes"] = int(prestage_capacity_bytes)
        trace["lora_loader_prestage_complete"] = True
        trace["lora_loader_prestage_total_ms"] = round(
            (time.monotonic() - started_at) * 1000, 3
        )


def discard_lora_tensors_for_merge(
    model: torch.nn.Module,
    named_tensors: List[Tuple[str, torch.Tensor]],
    load_context: Optional[Dict[str, Any]] = None,
) -> None:
    load_context = load_context or {}
    manifest = load_context.get("manifest") or {}
    request_id = prestage_request_id(manifest)
    cache = getattr(model, _PRESTAGED_LORA_CACHE_ATTR, None)
    if cache is not None:
        cache.pop(request_id, None)


def merge_lora_tensors_inplace(
    model: torch.nn.Module,
    named_tensors: List[Tuple[str, torch.Tensor]],
    load_context: Optional[Dict[str, Any]] = None,
) -> None:
    merge_started_at = time.monotonic()
    load_context = load_context or {}
    manifest = load_context.get("manifest") or {}
    trace = load_context.get("trace")
    model_device = _require_cuda_model_device(model)
    options = resolve_lora_merge_options(
        manifest,
        device=model_device,
        include_scaling=True,
    )
    scaling = options.scaling
    strict = options.strict
    if options.added_tokens_config:
        raise ValueError("Merged LoRA update does not support added tokens yet.")

    params = dict(model.named_parameters(remove_duplicate=False))
    layers_needing_postprocess: Dict[int, torch.nn.Module] = {}
    lora_pairs: List[Optional[_LoraPair]]
    if options.consume_prestaged:
        request_id = options.prestage_request_id or prestage_request_id(
            options.manifest
        )
        cache = getattr(model, _PRESTAGED_LORA_CACHE_ATTR, {})
        prestage_cache_entry = cache.pop(request_id, None)
        if prestage_cache_entry is None:
            raise ValueError(
                f"No prestaged LoRA merge tensors found for request {request_id}."
            )
        lora_pairs = list(prestage_cache_entry["pairs"])
        gpu_bucket_bytes = int(prestage_cache_entry["gpu_bucket_bytes"])
        if trace is not None:
            staged_count = sum(1 for pair in lora_pairs if pair is not None and pair.prestaged)
            trace["lora_loader_prestage_consumed"] = True
            trace["lora_loader_prestage_hit_count"] = staged_count
            trace["lora_loader_prestage_miss_count"] = len(lora_pairs) - staged_count
    else:
        gpu_bucket_bytes = options.apply_bucket_bytes
        lora_pairs = list(_collect_lora_pairs(named_tensors, strict=strict))
        if trace is not None:
            trace["lora_loader_prestage_consumed"] = False

    if trace is not None:
        trace["lora_loader_merge_impl"] = "bucketed_cuda"
        trace["lora_loader_pair_count"] = sum(pair is not None for pair in lora_pairs)
        trace["lora_loader_peak_device_budget_bytes"] = int(
            options.memory_budget.peak_bytes
        )
        trace["lora_loader_gpu_bucket_bytes"] = int(gpu_bucket_bytes)

    def apply_pair(
        base_name: str,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
    ) -> None:
        touched_layers = _apply_lora_pair_cuda_bucketed(
            base_name=base_name,
            lora_a=lora_a,
            lora_b=lora_b,
            scaling=scaling,
            params=params,
            model=model,
            strict=strict,
            bucket_bytes=int(gpu_bucket_bytes),
        )
        for layer in touched_layers:
            layers_needing_postprocess[id(layer)] = layer

    try:
        pair_traces: List[Dict[str, Any]] = []
        pair_apply_total_ms = 0.0
        pair_apply_max_ms = 0.0
        for pair_index, pair in enumerate(lora_pairs):
            if pair is None:
                continue
            lora_a = pair.lora_a
            lora_b = pair.lora_b
            pair_started_at = time.monotonic()
            apply_pair(
                pair.base_name,
                lora_a,
                lora_b,
            )
            pair_ms = round((time.monotonic() - pair_started_at) * 1000, 3)
            pair_apply_total_ms += pair_ms
            pair_apply_max_ms = max(pair_apply_max_ms, pair_ms)
            if trace is not None:
                pair_traces.append(
                    {
                        "index": pair_index,
                        "base_name": pair.base_name,
                        "prestaged": pair.prestaged,
                        "pair_ms": pair_ms,
                    }
                )
            lora_pairs[pair_index] = None
            pair = None
            lora_a = None
            lora_b = None
        if trace is not None:
            trace["lora_loader_pair_apply_total_ms"] = round(pair_apply_total_ms, 3)
            trace["lora_loader_pair_apply_max_ms"] = round(pair_apply_max_ms, 3)
            trace["lora_loader_first_pairs"] = pair_traces[:8]
            trace["lora_loader_top_pairs"] = sorted(
                pair_traces, key=lambda item: item["pair_ms"], reverse=True
            )[:8]
    finally:
        lora_pairs = []
        lora_a = None
        lora_b = None
        finalize_started_at = time.monotonic()
        for layer in layers_needing_postprocess.values():
            _finalize_flashinfer_moe_layer_after_merge(layer)
        if trace is not None:
            trace["lora_loader_finalize_flashinfer_ms"] = round(
                (time.monotonic() - finalize_started_at) * 1000, 3
            )
        if options.empty_cache_after_merge:
            empty_cache_started_at = time.monotonic()
            torch.cuda.empty_cache()
            if trace is not None:
                trace["lora_loader_empty_cache_ms"] = round(
                    (time.monotonic() - empty_cache_started_at) * 1000, 3
                )
        if trace is not None:
            trace["lora_loader_merge_total_ms"] = round(
                (time.monotonic() - merge_started_at) * 1000, 3
            )


def _split_lora_tensor_name(name: str) -> Tuple[str, str]:
    suffixes = {
        ".lora_A.weight": "A",
        ".lora_B.weight": "B",
        ".lora_embedding_A": "embedding_A",
        ".lora_embedding_B": "embedding_B",
    }
    for suffix, kind in suffixes.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], kind
    raise ValueError(f"Unsupported LoRA tensor name: {name}")


def _canonicalize_lora_base_name(name: str) -> str:
    while name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]

    if "model.language_model." in name:
        name = name.replace("model.language_model.", "model.")
    return name


def _base_name_variants(base_name: str) -> List[str]:
    variants = [base_name]
    if ".self_attn." in base_name:
        variants.append(base_name.replace(".self_attn.", "."))

    deduped: List[str] = []
    for variant in variants:
        if variant not in deduped:
            deduped.append(variant)
    return deduped


def _resolve_delta_ops(
    base_name: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    params: Dict[str, torch.nn.Parameter],
    model: torch.nn.Module,
) -> List[Any]:
    if _is_unembed_target(base_name):
        return _resolve_unembed_delta_ops(
            base_name=base_name,
            lora_a=lora_a,
            lora_b=lora_b,
            params=params,
            model=model,
        )

    explicit_match = _EXPLICIT_EXPERT_RE.match(base_name)
    if explicit_match:
        return _resolve_explicit_expert_ops(
            experts_prefix=explicit_match.group("prefix"),
            expert_id=int(explicit_match.group("expert_id")),
            target=explicit_match.group("target"),
            lora_a=lora_a,
            lora_b=lora_b,
            explicit_name=base_name,
        )

    aggregate_match = _match_aggregate_expert_target(base_name)
    if aggregate_match is not None:
        experts_prefix, target = aggregate_match
        return _resolve_aggregate_expert_ops(
            experts_prefix=experts_prefix,
            target=target,
            lora_a=lora_a,
            lora_b=lora_b,
            explicit_name=base_name,
        )

    direct_param_name = _resolve_direct_param_name(base_name, params)
    if direct_param_name is not None:
        return [_DenseLoraOp(direct_param_name, lora_a, lora_b)]

    for variant in _base_name_variants(base_name):
        for rule in _ALIAS_RULES:
            if not variant.endswith(rule.source_suffix):
                continue

            dest_base_name = _replace_suffix(
                variant, rule.source_suffix, rule.dest_suffix
            )
            dest_param_name = f"{dest_base_name}.weight"
            if dest_param_name not in params:
                continue

            return [
                _DenseLoraOp(
                    dest_param_name,
                    lora_a,
                    lora_b,
                    loader_args=rule.loader_args,
                )
            ]

    raise ValueError(f"Unsupported LoRA target for merged update: {base_name}")


def _resolve_unembed_delta_ops(
    base_name: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    params: Dict[str, torch.nn.Parameter],
    model: torch.nn.Module,
) -> List[_DenseLoraOp]:
    config = getattr(model, "config", None)
    tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))

    lm_head_name = _resolve_lm_head_param_name(base_name, params)
    embed_name = _resolve_embed_param_name(base_name, params)

    target_names: List[str] = []
    if tie_word_embeddings:
        if embed_name is not None:
            target_names.append(embed_name)
        if lm_head_name is not None:
            target_names.append(lm_head_name)
    else:
        if lm_head_name is not None:
            target_names.append(lm_head_name)
        elif embed_name is not None:
            target_names.append(embed_name)

    ops: List[_DenseLoraOp] = []
    seen_param_ids: Set[int] = set()
    for target_name in target_names:
        param = params.get(target_name)
        if param is None:
            continue
        param_id = id(param)
        if param_id in seen_param_ids:
            continue
        seen_param_ids.add(param_id)
        ops.append(_DenseLoraOp(target_name, lora_a, lora_b))

    if ops:
        return ops
    raise ValueError(f"Unsupported LoRA target for merged update: {base_name}")


def _resolve_aggregate_expert_ops(
    experts_prefix: str,
    target: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    explicit_name: str,
) -> List[Any]:
    canonical_target = _EXPERT_TARGET_ALIASES.get(target)
    if canonical_target is None:
        raise ValueError(
            f"Unsupported expert LoRA target for merged update: {explicit_name}"
        )

    if canonical_target in ("w1", "w2", "w3"):
        return [
            _AggregateExpertLoraOp(
                experts_prefix=experts_prefix,
                target=canonical_target,
                lora_a=lora_a,
                lora_b=lora_b,
            )
        ]

    if lora_b.shape[-2] % 2 != 0:
        raise ValueError(
            f"Cannot split packed expert LoRA B for {explicit_name}: "
            f"output dimension {lora_b.shape[-2]} is not even."
        )
    half = lora_b.shape[-2] // 2
    return [
        _AggregateExpertLoraOp(
            experts_prefix=experts_prefix,
            target="w1",
            lora_a=lora_a,
            lora_b=lora_b.narrow(-2, 0, half),
        ),
        _AggregateExpertLoraOp(
            experts_prefix=experts_prefix,
            target="w3",
            lora_a=lora_a,
            lora_b=lora_b.narrow(-2, half, half),
        ),
    ]


def _resolve_explicit_expert_ops(
    experts_prefix: str,
    expert_id: int,
    target: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    explicit_name: str,
) -> List[_DenseLoraOp]:
    canonical_target = _EXPERT_TARGET_ALIASES.get(target)
    if canonical_target is None:
        raise ValueError(
            f"Unsupported expert LoRA target for merged update: {explicit_name}"
        )

    if canonical_target == "w1":
        return [
            _DenseLoraOp(
                f"{experts_prefix}.w13_weight",
                lora_a,
                lora_b,
                loader_args=(f"{experts_prefix}.w13_weight", "w1", expert_id),
            )
        ]
    if canonical_target == "w3":
        return [
            _DenseLoraOp(
                f"{experts_prefix}.w13_weight",
                lora_a,
                lora_b,
                loader_args=(f"{experts_prefix}.w13_weight", "w3", expert_id),
            )
        ]
    if canonical_target == "w2":
        return [
            _DenseLoraOp(
                f"{experts_prefix}.w2_weight",
                lora_a,
                lora_b,
                loader_args=(f"{experts_prefix}.w2_weight", "w2", expert_id),
            )
        ]

    if lora_b.shape[-2] % 2 != 0:
        raise ValueError(
            f"Cannot split packed expert LoRA B for {explicit_name}: "
            f"output dimension {lora_b.shape[-2]} is not even."
        )
    half = lora_b.shape[-2] // 2
    return [
        _DenseLoraOp(
            f"{experts_prefix}.w13_weight",
            lora_a,
            lora_b.narrow(-2, 0, half),
            loader_args=(f"{experts_prefix}.w13_weight", "w1", expert_id),
        ),
        _DenseLoraOp(
            f"{experts_prefix}.w13_weight",
            lora_a,
            lora_b.narrow(-2, half, half),
            loader_args=(f"{experts_prefix}.w13_weight", "w3", expert_id),
        ),
    ]


def _apply_lora_pair_cuda_bucketed(
    base_name: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    params: Dict[str, torch.nn.Parameter],
    model: torch.nn.Module,
    strict: bool,
    bucket_bytes: int,
) -> Set[torch.nn.Module]:
    ops = _resolve_delta_ops(
        base_name=base_name,
        lora_a=lora_a,
        lora_b=lora_b,
        params=params,
        model=model,
    )

    touched_layers: Set[torch.nn.Module] = set()
    for op in ops:
        if isinstance(op, _AggregateExpertLoraOp):
            touched_layers.update(
                _apply_aggregate_expert_lora_op_cuda_bucketed(
                    op=op,
                    params=params,
                    model=model,
                    strict=strict,
                    scaling=scaling,
                    bucket_bytes=bucket_bytes,
                )
            )
        else:
            layer = _apply_dense_lora_op_cuda_bucketed(
                op=op,
                params=params,
                model=model,
                strict=strict,
                scaling=scaling,
                bucket_bytes=bucket_bytes,
            )
            if layer is not None:
                touched_layers.add(layer)
    return touched_layers


def _aggregate_expert_param_name(experts_prefix: str, target: str) -> str:
    if target in ("w1", "w3"):
        return f"{experts_prefix}.w13_weight"
    if target == "w2":
        return f"{experts_prefix}.w2_weight"
    raise ValueError(f"Unsupported aggregate expert target: {target}")


def _stage_cuda_fp32(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device == device and tensor.dtype == torch.float32:
        return tensor
    return tensor.to(device=device, dtype=torch.float32, non_blocking=True)


def _estimate_dense_temp_bytes(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    output_rows: Optional[int] = None,
    input_cols: Optional[int] = None,
) -> int:
    rows = int(output_rows if output_rows is not None else lora_b.shape[-2])
    rank = int(lora_a.shape[-2])
    cols = int(input_cols if input_cols is not None else lora_a.shape[-1])
    # A + B + delta + fp32 destination scratch.
    return 4 * (rank * cols + rows * rank + 2 * rows * cols)


def _dense_rows_per_bucket(
    *, rank: int, cols: int, total_rows: int, bucket_bytes: int
) -> int:
    a_bytes = 4 * rank * cols
    bytes_per_row = 4 * (rank + 2 * cols)
    available_for_rows = max(0, bucket_bytes - a_bytes)
    return max(
        1,
        min(total_rows, available_for_rows // max(bytes_per_row, 1)),
    )

def _unsupported_dense_bucketed_update(
    op: _DenseLoraOp,
    param: torch.nn.Parameter,
    owner_module: Optional[torch.nn.Module],
) -> ValueError:
    owner_name = type(owner_module).__name__ if owner_module is not None else None
    return ValueError(
        "Bucketed LoRA merge could not resolve a local dense destination for "
        f"{op.param_name}: param_shape={tuple(param.data.shape)} "
        f"lora_A_shape={tuple(op.lora_a.shape)} lora_B_shape={tuple(op.lora_b.shape)} "
        f"loader_args={op.loader_args} owner_module={owner_name}."
    )


def _make_dense_local_apply_spec(
    *,
    param_name: str,
    dst_view: torch.Tensor,
    lora_b_start: int,
    lora_b_rows: int,
    lora_a_start: int,
    lora_a_cols: int,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> _DenseLocalApplySpec:
    if dst_view.dim() != 2:
        raise ValueError(
            f"Merged LoRA dense bucketing requires a 2D destination for {param_name}; "
            f"got shape={tuple(dst_view.shape)}."
        )
    if lora_b_start < 0 or lora_b_start + lora_b_rows > lora_b.shape[0]:
        raise ValueError(
            f"LoRA B row slice is out of bounds for {param_name}: "
            f"start={lora_b_start} rows={lora_b_rows} shape={tuple(lora_b.shape)}."
        )
    if lora_a_start < 0 or lora_a_start + lora_a_cols > lora_a.shape[1]:
        raise ValueError(
            f"LoRA A column slice is out of bounds for {param_name}: "
            f"start={lora_a_start} cols={lora_a_cols} shape={tuple(lora_a.shape)}."
        )
    if tuple(dst_view.shape) != (lora_b_rows, lora_a_cols):
        raise ValueError(
            f"Shape mismatch for bucketed dense LoRA target {param_name}: "
            f"dst={tuple(dst_view.shape)} lora_B_rows={lora_b_rows} "
            f"lora_A_cols={lora_a_cols}."
        )
    return _DenseLocalApplySpec(
        dst_view=dst_view,
        lora_b_start=int(lora_b_start),
        lora_b_rows=int(lora_b_rows),
        lora_a_start=int(lora_a_start),
        lora_a_cols=int(lora_a_cols),
    )


def _direct_dense_local_apply_spec(
    op: _DenseLoraOp,
    param: torch.nn.Parameter,
) -> Optional[List[_DenseLocalApplySpec]]:
    if param.data.dim() != 2:
        return None
    return [
        _make_dense_local_apply_spec(
            param_name=op.param_name,
            dst_view=param.data,
            lora_b_start=0,
            lora_b_rows=param.data.shape[0],
            lora_a_start=0,
            lora_a_cols=param.data.shape[1],
            lora_a=op.lora_a,
            lora_b=op.lora_b,
        )
    ]


def _resolve_generic_packed_dense_local_apply_specs(
    op: _DenseLoraOp,
    param: torch.nn.Parameter,
    owner_module: torch.nn.Module,
) -> Optional[List[_DenseLocalApplySpec]]:
    if param.data.dim() != 2 or len(op.loader_args) > 1:
        return None

    output_sizes = list(getattr(owner_module, "output_sizes"))
    loaded_shard_id = op.loader_args[0] if op.loader_args else None

    def make_shard_spec(
        shard_id: Any, source_offset: int
    ) -> Tuple[_DenseLocalApplySpec, int]:
        shard_idx = _normalize_generic_shard_id(shard_id)
        shard_size = output_sizes[shard_idx]
        dest_start = sum(output_sizes[:shard_idx])
        return (
            _make_dense_local_apply_spec(
                param_name=op.param_name,
                dst_view=param.data.narrow(0, dest_start, shard_size),
                lora_b_start=source_offset,
                lora_b_rows=shard_size,
                lora_a_start=0,
                lora_a_cols=param.data.shape[1],
                lora_a=op.lora_a,
                lora_b=op.lora_b,
            ),
            source_offset + shard_size,
        )

    if isinstance(loaded_shard_id, tuple):
        specs = []
        source_offset = 0
        for shard_id in loaded_shard_id:
            spec, source_offset = make_shard_spec(shard_id, source_offset)
            specs.append(spec)
        return specs

    if loaded_shard_id is None:
        if tuple(param.data.shape) == (op.lora_b.shape[0], op.lora_a.shape[1]):
            return _direct_dense_local_apply_spec(op, param)

        specs = []
        source_offset = 0
        for shard_idx in range(len(output_sizes)):
            spec, source_offset = make_shard_spec(shard_idx, source_offset)
            specs.append(spec)
        return specs

    spec, _ = make_shard_spec(loaded_shard_id, 0)
    return [spec]


def _resolve_column_parallel_dense_local_apply_specs(
    op: _DenseLoraOp,
    param: torch.nn.Parameter,
    owner_module: ColumnParallelLinear,
) -> Optional[List[_DenseLocalApplySpec]]:
    if op.loader_args or param.data.dim() != 2:
        return None
    output_dim = getattr(param, "output_dim", None)
    if output_dim not in (None, 0):
        return None
    _assert_no_special_loader_features(param, param_name=op.param_name)

    lora_b_start = 0
    if output_dim is not None and not owner_module.use_presharded_weights:
        lora_b_start = owner_module.tp_rank * param.data.shape[0]

    return [
        _make_dense_local_apply_spec(
            param_name=op.param_name,
            dst_view=param.data,
            lora_b_start=lora_b_start,
            lora_b_rows=param.data.shape[0],
            lora_a_start=0,
            lora_a_cols=param.data.shape[1],
            lora_a=op.lora_a,
            lora_b=op.lora_b,
        )
    ]


def _resolve_merged_column_dense_local_apply_specs(
    op: _DenseLoraOp,
    param: torch.nn.Parameter,
    owner_module: MergedColumnParallelLinear,
) -> Optional[List[_DenseLocalApplySpec]]:
    if param.data.dim() != 2 or len(op.loader_args) > 1:
        return None
    output_dim = getattr(param, "output_dim", None)
    if output_dim != 0:
        return None
    _assert_no_special_loader_features(param, param_name=op.param_name)

    loaded_shard_id = op.loader_args[0] if op.loader_args else None

    def make_shard_spec(
        shard_id: int, source_offset: int
    ) -> Tuple[_DenseLocalApplySpec, int]:
        full_shard_size = owner_module.output_sizes[shard_id]
        local_shard_size = full_shard_size // owner_module.tp_size
        dest_start = sum(owner_module.output_sizes[:shard_id]) // owner_module.tp_size
        lora_b_start = source_offset
        next_source_offset = source_offset + local_shard_size
        if not owner_module.use_presharded_weights:
            lora_b_start += owner_module.tp_rank * local_shard_size
            next_source_offset = source_offset + full_shard_size

        return (
            _make_dense_local_apply_spec(
                param_name=op.param_name,
                dst_view=param.data.narrow(0, dest_start, local_shard_size),
                lora_b_start=lora_b_start,
                lora_b_rows=local_shard_size,
                lora_a_start=0,
                lora_a_cols=param.data.shape[1],
                lora_a=op.lora_a,
                lora_b=op.lora_b,
            ),
            next_source_offset,
        )

    if isinstance(loaded_shard_id, tuple):
        specs = []
        source_offset = 0
        for shard_id in loaded_shard_id:
            spec, source_offset = make_shard_spec(int(shard_id), source_offset)
            specs.append(spec)
        return specs

    if loaded_shard_id is None:
        if owner_module.use_presharded_weights and tuple(param.data.shape) == (
            op.lora_b.shape[0],
            op.lora_a.shape[1],
        ):
            return _direct_dense_local_apply_spec(op, param)

        specs = []
        source_offset = 0
        for shard_id in range(len(owner_module.output_sizes)):
            spec, source_offset = make_shard_spec(shard_id, source_offset)
            specs.append(spec)
        return specs

    spec, _ = make_shard_spec(int(loaded_shard_id), 0)
    return [spec]


def _resolve_qkv_dense_local_apply_specs(
    op: _DenseLoraOp,
    param: torch.nn.Parameter,
    owner_module: QKVParallelLinear,
) -> Optional[List[_DenseLocalApplySpec]]:
    if param.data.dim() != 2 or len(op.loader_args) > 1:
        return None
    output_dim = getattr(param, "output_dim", None)
    if output_dim != 0:
        return None
    _assert_no_special_loader_features(param, param_name=op.param_name)

    loaded_shard_id = op.loader_args[0] if op.loader_args else None
    shard_specs = {
        "q": (
            0,
            owner_module.num_heads * owner_module.head_size,
            owner_module.tp_rank,
            0,
            owner_module.total_num_heads * owner_module.head_size,
        ),
        "k": (
            owner_module.num_heads * owner_module.head_size,
            owner_module.num_kv_heads * owner_module.head_size,
            owner_module.tp_rank // owner_module.num_kv_head_replicas,
            owner_module.total_num_heads * owner_module.head_size,
            owner_module.total_num_kv_heads * owner_module.head_size,
        ),
        "v": (
            (owner_module.num_heads + owner_module.num_kv_heads)
            * owner_module.head_size,
            owner_module.num_kv_heads * owner_module.v_head_size,
            owner_module.tp_rank // owner_module.num_kv_head_replicas,
            (owner_module.total_num_heads + owner_module.total_num_kv_heads)
            * owner_module.head_size,
            owner_module.total_num_kv_heads * owner_module.v_head_size,
        ),
    }

    def make_qkv_spec(shard_id: str, source_offset: int) -> _DenseLocalApplySpec:
        shard_offset, shard_size, tp_shard_id, _, _ = shard_specs[shard_id]
        lora_b_start = source_offset
        if not owner_module.use_presharded_weights:
            lora_b_start += tp_shard_id * shard_size
        return _make_dense_local_apply_spec(
            param_name=op.param_name,
            dst_view=param.data.narrow(0, shard_offset, shard_size),
            lora_b_start=lora_b_start,
            lora_b_rows=shard_size,
            lora_a_start=0,
            lora_a_cols=param.data.shape[1],
            lora_a=op.lora_a,
            lora_b=op.lora_b,
        )

    if loaded_shard_id is None:
        if owner_module.use_presharded_weights and tuple(param.data.shape) == (
            op.lora_b.shape[0],
            op.lora_a.shape[1],
        ):
            return _direct_dense_local_apply_spec(op, param)
        return [
            make_qkv_spec(shard_id, source_global_offset)
            for shard_id, (_, _, _, source_global_offset, _) in shard_specs.items()
        ]

    if loaded_shard_id not in shard_specs:
        raise ValueError(
            f"Unsupported qkv shard id for merged update on {op.param_name}: {loaded_shard_id}"
        )
    return [make_qkv_spec(loaded_shard_id, 0)]


def _resolve_row_parallel_dense_local_apply_specs(
    op: _DenseLoraOp,
    param: torch.nn.Parameter,
    owner_module: RowParallelLinear,
) -> Optional[List[_DenseLocalApplySpec]]:
    if op.loader_args or param.data.dim() != 2:
        return None
    input_dim = getattr(param, "input_dim", None)
    if input_dim not in (None, 1):
        return None
    _assert_no_special_loader_features(param, param_name=op.param_name)

    lora_a_start = 0
    if input_dim is not None and not owner_module.use_presharded_weights:
        lora_a_start = owner_module.tp_rank * param.data.shape[1]

    return [
        _make_dense_local_apply_spec(
            param_name=op.param_name,
            dst_view=param.data,
            lora_b_start=0,
            lora_b_rows=param.data.shape[0],
            lora_a_start=lora_a_start,
            lora_a_cols=param.data.shape[1],
            lora_a=op.lora_a,
            lora_b=op.lora_b,
        )
    ]


def _resolve_dense_local_apply_specs(
    op: _DenseLoraOp,
    param: torch.nn.Parameter,
    owner_module: Optional[torch.nn.Module],
) -> Optional[List[_DenseLocalApplySpec]]:
    if op.lora_a.dim() != 2 or op.lora_b.dim() != 2:
        return None
    if (
        owner_module is not None
        and len(op.loader_args) == 3
        and hasattr(owner_module, "w13_weight")
        and hasattr(owner_module, "w2_weight")
    ):
        _, shard_id, expert_id = op.loader_args
        quant_method = getattr(owner_module, "quant_method", None)
        use_unquantized_fused = isinstance(quant_method, UnquantizedFusedMoEMethod)
        local_expert_ids = (
            _resolve_moe_expert_ids(owner_module, param, int(expert_id))
            if use_unquantized_fused
            else [int(expert_id)]
        )
        return [
            _resolve_expert_dense_local_apply_spec(
                param_name=op.param_name,
                param=param,
                owner_module=owner_module,
                shard_id=str(shard_id),
                local_expert_id=local_expert_id,
                lora_a=op.lora_a,
                lora_b=op.lora_b,
                use_unquantized_fused=use_unquantized_fused,
            )
            for local_expert_id in local_expert_ids
        ]

    if isinstance(owner_module, QKVParallelLinear):
        return _resolve_qkv_dense_local_apply_specs(op, param, owner_module)
    if isinstance(owner_module, MergedColumnParallelLinear):
        return _resolve_merged_column_dense_local_apply_specs(op, param, owner_module)
    if isinstance(owner_module, RowParallelLinear):
        return _resolve_row_parallel_dense_local_apply_specs(op, param, owner_module)
    if isinstance(owner_module, (ParallelLMHead, VocabParallelEmbedding)):
        return None
    if isinstance(owner_module, ColumnParallelLinear):
        return _resolve_column_parallel_dense_local_apply_specs(op, param, owner_module)
    if owner_module is not None and hasattr(owner_module, "output_sizes"):
        return _resolve_generic_packed_dense_local_apply_specs(op, param, owner_module)
    if op.loader_args:
        return None
    return _direct_dense_local_apply_spec(op, param)


def _apply_dense_local_apply_specs_cuda_bucketed(
    op: _DenseLoraOp,
    specs: List[_DenseLocalApplySpec],
    scaling: float,
    bucket_bytes: int,
) -> None:
    if not specs:
        return

    rank = int(op.lora_a.shape[0])
    for spec in specs:
        if spec.lora_b_rows <= 0:
            continue
        device = spec.dst_view.device
        a_source = op.lora_a.narrow(1, spec.lora_a_start, spec.lora_a_cols)
        a = _stage_cuda_fp32(a_source, device)
        rows_per_bucket = _dense_rows_per_bucket(
            rank=rank,
            cols=spec.lora_a_cols,
            total_rows=spec.lora_b_rows,
            bucket_bytes=bucket_bytes,
        )

        row = 0
        while row < spec.lora_b_rows:
            rows = min(rows_per_bucket, spec.lora_b_rows - row)
            b_source = op.lora_b.narrow(0, spec.lora_b_start + row, rows)
            b = _stage_cuda_fp32(b_source, device)
            delta = torch.matmul(b, a).mul_(scaling)
            _add_local_delta_fp32_(spec.dst_view.narrow(0, row, rows), delta)
            row += rows


def _apply_dense_lora_op_cuda_bucketed(
    op: _DenseLoraOp,
    params: Dict[str, torch.nn.Parameter],
    model: torch.nn.Module,
    strict: bool,
    scaling: float,
    bucket_bytes: int,
) -> Optional[torch.nn.Module]:
    if op.param_name not in params:
        if strict:
            raise ValueError(
                f"Target parameter not found for merged LoRA update: {op.param_name}"
            )
        logger.warning("Skipping unknown merged LoRA target: %s", op.param_name)
        return None

    param = params[op.param_name]
    _ensure_supported_param(param, op.param_name)
    owner_module = _resolve_param_owner_module(model, op.param_name, param)
    quant_method = getattr(owner_module, "quant_method", None)
    if (
        isinstance(quant_method, UnquantizedFusedMoEMethod)
        and len(op.loader_args) == 3
    ):
        weight_name, shard_id, expert_id = op.loader_args
        if _try_apply_flashinfer_trtllm_moe_lora_op_cuda_bucketed(
            param_name=op.param_name,
            param=param,
            owner_module=owner_module,
            shard_id=str(shard_id),
            local_expert_id=int(expert_id),
            lora_a=op.lora_a,
            lora_b=op.lora_b,
            scaling=scaling,
            bucket_bytes=bucket_bytes,
        ):
            return None
        quant_method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
            layer=owner_module,
            param=param,
            weight_name=str(weight_name),
        )


    if _try_apply_vocab_lora_op_cuda_bucketed(
        op=op,
        param=param,
        owner_module=owner_module,
        scaling=scaling,
        bucket_bytes=bucket_bytes,
    ):
        return _get_flashinfer_moe_layer_for_postprocess(param, owner_module=owner_module)

    specs = _resolve_dense_local_apply_specs(op, param, owner_module)
    if specs is None:
        raise _unsupported_dense_bucketed_update(op, param, owner_module)

    _apply_dense_local_apply_specs_cuda_bucketed(
        op=op,
        specs=specs,
        scaling=scaling,
        bucket_bytes=bucket_bytes,
    )
    return _get_flashinfer_moe_layer_for_postprocess(param, owner_module=owner_module)


def _try_apply_vocab_lora_op_cuda_bucketed(
    op: _DenseLoraOp,
    param: torch.nn.Parameter,
    owner_module: Optional[torch.nn.Module],
    scaling: float,
    bucket_bytes: int,
) -> bool:
    if op.loader_args:
        return False
    if not isinstance(owner_module, (ParallelLMHead, VocabParallelEmbedding)):
        return False
    if op.lora_a.dim() != 2 or op.lora_b.dim() != 2:
        return False

    _assert_no_special_loader_features(param, param_name=op.param_name)
    output_dim = getattr(param, "output_dim", None)
    if output_dim not in (None, 0):
        return False

    if output_dim is None:
        source_start = 0
        shard_size = param.data.shape[0]
    elif owner_module.use_presharded_weights:
        source_start = 0
        shard_size = param.data.shape[0]
    else:
        source_start = owner_module.shard_indices.org_vocab_start_index
        shard_size = owner_module.shard_indices.org_vocab_end_index - source_start

    if source_start + shard_size > op.lora_b.shape[0]:
        shard_size = max(op.lora_b.shape[0] - source_start, 0)
    if shard_size <= 0:
        return True

    dst_base = param[:shard_size].data
    device = param.device
    a = _stage_cuda_fp32(op.lora_a, device)
    rank = int(op.lora_a.shape[0])
    cols = int(op.lora_a.shape[1])
    a_bytes = 4 * rank * cols
    bytes_per_row = 4 * (rank + 2 * cols)
    available_for_rows = max(0, bucket_bytes - a_bytes)
    rows_per_bucket = max(
        1, min(shard_size, available_for_rows // max(bytes_per_row, 1))
    )

    row = 0
    while row < shard_size:
        rows = min(rows_per_bucket, shard_size - row)
        b = _stage_cuda_fp32(op.lora_b.narrow(0, source_start + row, rows), device)
        delta = torch.matmul(b, a).mul_(scaling)
        _add_local_delta_fp32_(dst_base.narrow(0, row, rows), delta)
        row += rows

    return True


def _select_expert_lora_2d(
    tensor: torch.Tensor,
    expert_id: int,
    expert_count: int,
    device: torch.device,
) -> torch.Tensor:
    selected = _select_expert_chunk_cuda_fp32(
        tensor, [expert_id], expert_count, device
    )[0]
    if selected.dim() != 2:
        raise ValueError(f"Unsupported MoE LoRA tensor rank: {selected.dim()}")
    return selected


def _resolve_simple_expert_dense_local_apply_spec(
    *,
    param_name: str,
    param: torch.nn.Parameter,
    shard_id: str,
    expert_id: int,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> _DenseLocalApplySpec:
    if shard_id in ("w1", "w3"):
        half = param.shape[1] // 2
        start = 0 if shard_id == "w1" else half
        dst_view = param.data[expert_id].narrow(0, start, half)
    elif shard_id == "w2":
        dst_view = param.data[expert_id]
    else:
        raise ValueError(
            f"Unsupported expert shard id for merged update on {param_name}: {shard_id}"
        )

    return _make_dense_local_apply_spec(
        param_name=param_name,
        dst_view=dst_view,
        lora_b_start=0,
        lora_b_rows=dst_view.shape[0],
        lora_a_start=0,
        lora_a_cols=dst_view.shape[1],
        lora_a=lora_a,
        lora_b=lora_b,
    )


def _resolve_unquantized_fused_moe_dense_local_apply_spec(
    *,
    param_name: str,
    param: torch.nn.Parameter,
    owner_module: torch.nn.Module,
    shard_id: str,
    local_expert_id: int,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> _DenseLocalApplySpec:
    quant_method = getattr(owner_module, "quant_method", None)
    if not isinstance(quant_method, UnquantizedFusedMoEMethod):
        raise ValueError(
            f"Merged LoRA update only supports unquantized fused MoE weights for {param_name}."
        )

    use_flashinfer = getattr(quant_method, "use_flashinfer_trtllm_moe", False)
    use_triton_kernels = getattr(owner_module, "use_triton_kernels", False)
    if is_cpu():
        raise ValueError(
            f"Bucketed LoRA merge requires CUDA for unquantized fused MoE {param_name}."
        )

    tp_rank = getattr(owner_module, "moe_tp_rank", 0)
    use_presharded_weights = getattr(owner_module, "use_presharded_weights", False)
    expert_data = param.data[local_expert_id]

    if shard_id in ("w1", "w3"):
        shard_dim = 1 if use_triton_kernels else 0
        shard_size = expert_data.shape[shard_dim] // 2
        weight_start = shard_size * tp_rank
        lora_b_start = 0 if use_presharded_weights else weight_start
        lora_b_rows = shard_size
        if use_flashinfer:
            lora_b_rows = get_actual_shard_size(
                shard_size,
                lora_b_start,
                lora_b.shape[0],
            )
        switch_w13 = getattr(quant_method, "load_up_proj_weight_first", False)
        is_gated = getattr(
            getattr(owner_module, "moe_runner_config", None), "is_gated", False
        )
        start = (
            shard_size
            if (
                (switch_w13 and shard_id == "w1")
                or (not switch_w13 and shard_id == "w3")
            )
            and is_gated
            else 0
        )
        dst_view = expert_data.narrow(shard_dim, start, lora_b_rows)
        if use_triton_kernels:
            dst_view = dst_view.transpose(0, 1)
        return _make_dense_local_apply_spec(
            param_name=param_name,
            dst_view=dst_view,
            lora_b_start=lora_b_start,
            lora_b_rows=lora_b_rows,
            lora_a_start=0,
            lora_a_cols=dst_view.shape[1],
            lora_a=lora_a,
            lora_b=lora_b,
        )

    if shard_id == "w2":
        shard_dim = 0 if use_triton_kernels else 1
        shard_size = expert_data.shape[shard_dim]
        weight_start = shard_size * tp_rank
        lora_a_start = 0 if use_presharded_weights else weight_start
        lora_a_cols = shard_size
        if use_flashinfer:
            lora_a_cols = get_actual_shard_size(
                shard_size,
                lora_a_start,
                lora_a.shape[1],
            )
        dst_view = expert_data.narrow(shard_dim, 0, lora_a_cols)
        if use_triton_kernels:
            dst_view = dst_view.transpose(0, 1)
        return _make_dense_local_apply_spec(
            param_name=param_name,
            dst_view=dst_view,
            lora_b_start=0,
            lora_b_rows=dst_view.shape[0],
            lora_a_start=lora_a_start,
            lora_a_cols=lora_a_cols,
            lora_a=lora_a,
            lora_b=lora_b,
        )

    raise ValueError(
        f"Unsupported MoE shard id for merged update on {param_name}: {shard_id}"
    )


def _resolve_expert_dense_local_apply_spec(
    *,
    param_name: str,
    param: torch.nn.Parameter,
    owner_module: torch.nn.Module,
    shard_id: str,
    local_expert_id: int,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    use_unquantized_fused: bool,
) -> _DenseLocalApplySpec:
    if use_unquantized_fused:
        return _resolve_unquantized_fused_moe_dense_local_apply_spec(
            param_name=param_name,
            param=param,
            owner_module=owner_module,
            shard_id=shard_id,
            local_expert_id=local_expert_id,
            lora_a=lora_a,
            lora_b=lora_b,
        )
    return _resolve_simple_expert_dense_local_apply_spec(
        param_name=param_name,
        param=param,
        shard_id=shard_id,
        expert_id=local_expert_id,
        lora_a=lora_a,
        lora_b=lora_b,
    )

def _apply_aggregate_expert_lora_op_cuda_bucketed(
    op: _AggregateExpertLoraOp,
    params: Dict[str, torch.nn.Parameter],
    model: torch.nn.Module,
    strict: bool,
    scaling: float,
    bucket_bytes: int,
) -> Set[torch.nn.Module]:
    param_name = _aggregate_expert_param_name(op.experts_prefix, op.target)
    if param_name not in params:
        if strict:
            raise ValueError(
                f"Target parameter not found for merged LoRA update: {param_name}"
            )
        logger.warning("Skipping unknown merged LoRA target: %s", param_name)
        return set()

    param = params[param_name]
    _ensure_supported_param(param, param_name)
    owner_module = _resolve_param_owner_module(model, param_name, param)
    touched_layer = _get_flashinfer_moe_layer_for_postprocess(
        param, owner_module=owner_module
    )
    if owner_module is None:
        raise ValueError(f"Cannot resolve owner module for expert target {param_name}")

    quant_method = getattr(owner_module, "quant_method", None)
    use_unquantized_fused = isinstance(quant_method, UnquantizedFusedMoEMethod)
    use_flashinfer_blocked_direct = (
        use_unquantized_fused
        and getattr(quant_method, "use_flashinfer_trtllm_moe", False)
        and param.data.dim() == 4
    )
    if use_unquantized_fused and not use_flashinfer_blocked_direct:
        quant_method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
            layer=owner_module,
            param=param,
            weight_name=param_name,
        )

    expert_count = max(_expert_axis_size(op.lora_a), _expert_axis_size(op.lora_b))
    expert_entries: List[Tuple[int, int]] = []
    for expert_id in range(expert_count):
        if use_unquantized_fused:
            local_expert_ids = _resolve_moe_expert_ids(owner_module, param, expert_id)
        else:
            local_expert_ids = [expert_id]
        for local_expert_id in local_expert_ids:
            expert_entries.append((expert_id, local_expert_id))

    if not expert_entries:
        if use_flashinfer_blocked_direct:
            return set()
        return {touched_layer} if touched_layer is not None else set()

    device = param.device

    if use_flashinfer_blocked_direct:
        for source_expert_id, local_expert_id in expert_entries:
            lora_a = _select_expert_lora_2d(
                op.lora_a, source_expert_id, expert_count, device
            )
            lora_b = _select_expert_lora_2d(
                op.lora_b, source_expert_id, expert_count, device
            )
            applied = _try_apply_flashinfer_trtllm_moe_lora_op_cuda_bucketed(
                param_name=param_name,
                param=param,
                owner_module=owner_module,
                shard_id=op.target,
                local_expert_id=local_expert_id,
                lora_a=lora_a,
                lora_b=lora_b,
                scaling=scaling,
                bucket_bytes=bucket_bytes,
            )
            if not applied:
                raise RuntimeError(
                    "Expected direct FlashInfer TRT-LLM MoE LoRA update to apply "
                    f"for {param_name}."
                )
        return set()

    for source_expert_id, local_expert_id in expert_entries:
        lora_a = _select_expert_lora_2d(
            op.lora_a, source_expert_id, expert_count, device
        )
        lora_b = _select_expert_lora_2d(
            op.lora_b, source_expert_id, expert_count, device
        )
        expert_op = _DenseLoraOp(param_name=param_name, lora_a=lora_a, lora_b=lora_b)
        spec = _resolve_expert_dense_local_apply_spec(
            param_name=param_name,
            param=param,
            owner_module=owner_module,
            shard_id=op.target,
            local_expert_id=local_expert_id,
            lora_a=lora_a,
            lora_b=lora_b,
            use_unquantized_fused=use_unquantized_fused,
        )
        _apply_dense_local_apply_specs_cuda_bucketed(
            expert_op, [spec], scaling, bucket_bytes
        )

    return {touched_layer} if touched_layer is not None else set()


def _select_expert_chunk_cuda_fp32(
    tensor: torch.Tensor,
    expert_ids: List[int],
    expert_count: int,
    device: torch.device,
) -> torch.Tensor:
    if tensor.dim() <= 2:
        selected = tensor.unsqueeze(0).expand(len(expert_ids), *tensor.shape)
    elif tensor.dim() != 3:
        raise ValueError(f"Unsupported MoE LoRA tensor rank: {tensor.dim()}")
    elif tensor.shape[0] == 1:
        selected = tensor[0].unsqueeze(0).expand(len(expert_ids), *tensor.shape[1:])
    elif tensor.shape[0] == expert_count:
        index = torch.tensor(expert_ids, dtype=torch.long, device=tensor.device)
        selected = tensor.index_select(0, index)
    else:
        raise ValueError(
            "Mismatched expert dimensions in MoE LoRA tensor: "
            f"got {tensor.shape[0]}, expected {expert_count}"
        )
    if selected.device == device and selected.dtype == torch.float32:
        return selected
    return selected.to(device=device, dtype=torch.float32, non_blocking=True)


def _is_unembed_target(base_name: str) -> bool:
    return base_name == "unembed_tokens" or base_name.endswith(".unembed_tokens")


def _resolve_lm_head_param_name(
    base_name: str, params: Dict[str, torch.nn.Parameter]
) -> Optional[str]:
    candidates: List[str] = []
    if base_name in ("unembed_tokens", "model.unembed_tokens"):
        candidates.append("lm_head.weight")
    if base_name.endswith(".unembed_tokens"):
        candidates.append(_replace_suffix(base_name, ".unembed_tokens", ".lm_head.weight"))
    candidates.append("lm_head.weight")
    return _first_existing_param_name(candidates, params)


def _resolve_embed_param_name(
    base_name: str, params: Dict[str, torch.nn.Parameter]
) -> Optional[str]:
    candidates: List[str] = []
    if base_name == "unembed_tokens":
        candidates.append("model.embed_tokens.weight")
    if base_name.endswith(".unembed_tokens"):
        candidates.append(
            _replace_suffix(base_name, ".unembed_tokens", ".embed_tokens.weight")
        )
    candidates.append("model.embed_tokens.weight")
    return _first_existing_param_name(candidates, params)


def _first_existing_param_name(
    candidates: Iterable[str], params: Dict[str, torch.nn.Parameter]
) -> Optional[str]:
    for candidate in candidates:
        if candidate in params:
            return candidate
    return None


def _resolve_direct_param_name(
    base_name: str, params: Dict[str, torch.nn.Parameter]
) -> Optional[str]:
    for variant in _base_name_variants(base_name):
        candidate = f"{variant}.weight"
        if candidate in params:
            return candidate
    return None


def _replace_suffix(value: str, source_suffix: str, dest_suffix: str) -> str:
    return value[: -len(source_suffix)] + dest_suffix


def _match_aggregate_expert_target(base_name: str) -> Optional[Tuple[str, str]]:
    suffixes = (
        ".experts.w1",
        ".experts.w2",
        ".experts.w3",
        ".experts.gate_proj",
        ".experts.up_proj",
        ".experts.down_proj",
        ".experts.gate_up_proj",
    )
    for suffix in suffixes:
        if base_name.endswith(suffix):
            experts_prefix = base_name[: -len(suffix)] + ".experts"
            target = suffix[len(".experts.") :]
            return experts_prefix, target
    return None


def _expert_axis_size(tensor: torch.Tensor) -> int:
    if tensor.dim() <= 2:
        return 1
    if tensor.dim() != 3:
        raise ValueError(f"Unsupported MoE LoRA tensor rank: {tensor.dim()}")
    return tensor.shape[0]


def _ensure_supported_param(param: torch.nn.Parameter, param_name: str) -> None:
    if param.dtype not in _FLOAT_DTYPES:
        raise ValueError(
            f"Merged LoRA update currently supports only fp32/fp16/bf16 weights. "
            f"Parameter {param_name} has dtype {param.dtype}."
        )


def _resolve_param_owner_module(
    model: torch.nn.Module,
    param_name: str,
    param: torch.nn.Parameter,
) -> Optional[torch.nn.Module]:
    module_name = param_name.rsplit(".", 1)[0]
    if module_name:
        try:
            return model.get_submodule(module_name)
        except AttributeError:
            pass

    weight_loader = getattr(param, "weight_loader", None)
    return getattr(weight_loader, "__self__", None)


def _add_local_delta_fp32_(dst_view: torch.Tensor, delta: torch.Tensor) -> None:
    delta_fp32 = delta.to(device=dst_view.device, dtype=torch.float32)
    updated = dst_view.to(device=dst_view.device, dtype=torch.float32)
    updated.add_(delta_fp32)
    dst_view.copy_(updated)


def _assert_no_special_loader_features(
    param: torch.nn.Parameter,
    *,
    param_name: str,
    allow_packed_dim: bool = False,
) -> None:
    if getattr(param, "use_bitsandbytes_4bit", False):
        raise ValueError(
            f"Merged LoRA update does not support bitsandbytes weights for {param_name}."
        )
    if getattr(param, "is_gguf_weight", False) or getattr(
        param, "is_gguf_weight_type", False
    ):
        raise ValueError(
            f"Merged LoRA update does not support GGUF weights for {param_name}."
        )
    if getattr(param, "is_metadata", False):
        raise ValueError(
            f"Merged LoRA update does not support metadata-backed weights for {param_name}."
        )
    if getattr(param, "needs_scalar_to_array", False):
        raise ValueError(
            f"Merged LoRA update does not support scalar-array fused weights for {param_name}."
        )

    packed_dim = getattr(param, "packed_dim", None)
    if packed_dim is not None and not allow_packed_dim:
        raise ValueError(
            f"Merged LoRA update does not support packed weights for {param_name}."
        )


def _normalize_generic_shard_id(loaded_shard_id: Any) -> int:
    shard_map = {"q": 0, "k": 1, "v": 2}
    if loaded_shard_id in shard_map:
        return shard_map[loaded_shard_id]
    if isinstance(loaded_shard_id, int):
        return loaded_shard_id
    raise ValueError(f"Unsupported packed shard id for merged update: {loaded_shard_id}")


def _resolve_moe_expert_ids(
    owner_module: torch.nn.Module,
    param: torch.nn.Parameter,
    expert_id: int,
) -> List[int]:
    require_global_experts = getattr(param, "_sglang_require_global_experts", False)
    global_expert_location_metadata = get_global_expert_location_metadata()
    if global_expert_location_metadata is None:
        if not require_global_experts and hasattr(
            owner_module, "_map_global_expert_id_to_local_expert_id"
        ):
            local_expert_id = owner_module._map_global_expert_id_to_local_expert_id(
                expert_id
            )
            if local_expert_id == -1:
                return []
            return [local_expert_id]
        return [expert_id]

    if expert_id >= (
        owner_module.num_experts - getattr(owner_module, "num_fused_shared_experts", 0)
    ):
        physical_expert_ids = [expert_id]
    else:
        physical_expert_ids = global_expert_location_metadata.logical_to_all_physical(
            owner_module.layer_id,
            expert_id,
            require_global_experts,
        )

    resolved: List[int] = []
    for physical_expert_id in physical_expert_ids:
        if not require_global_experts and hasattr(
            owner_module, "_map_global_expert_id_to_local_expert_id"
        ):
            local_expert_id = owner_module._map_global_expert_id_to_local_expert_id(
                physical_expert_id
            )
            if local_expert_id < 0 or local_expert_id >= owner_module.num_local_experts:
                continue
            resolved.append(local_expert_id)
        else:
            resolved.append(physical_expert_id)
    return resolved



merge_lora_tensors_inplace.sglang_supports_host_tensors = True
merge_lora_tensors_inplace.sglang_requires_cuda_graph_recapture = False
merge_lora_tensors_inplace.sglang_prepare_tensors = prepare_lora_tensors_for_merge
merge_lora_tensors_inplace.sglang_discard_prepared_tensors = (
    discard_lora_tensors_for_merge
)
