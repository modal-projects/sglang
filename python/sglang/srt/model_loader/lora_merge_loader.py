import logging
import os
import re
import time
from contextvars import ContextVar
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
from sglang.srt.layers.utils.common import pad_or_narrow_weight
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.weight_utils import narrow_padded_param_and_loaded_weight
from sglang.srt.utils import is_cpu

logger = logging.getLogger(__name__)
_PAIR_TRACE: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "_PAIR_TRACE", default=None
)

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
class _MergeMemoryBudget:
    peak_bytes: int
    bucket_bytes: int
    source: str
    headroom_gb: float
    free_bytes: Optional[int]


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


def _manifest_bool(manifest: Dict[str, Any], key: str) -> bool:
    value = manifest.get(key)
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def _prestage_request_id(
    manifest: Dict[str, Any], trace: Optional[Dict[str, Any]]
) -> str:
    request_id = (
        manifest.get("lora_merge_prestage_request_id")
        or manifest.get("prestage_request_id")
        or (trace or {}).get("request_id")
    )
    if request_id is None:
        raise ValueError("LoRA merge prestage requires a request_id.")
    return str(request_id)


def _trace_add_ms(trace: Optional[Dict[str, Any]], key: str, elapsed_ms: float) -> None:
    if trace is None:
        return
    trace[key] = round(float(trace.get(key, 0.0)) + elapsed_ms, 3)


def _trace_max_ms(trace: Optional[Dict[str, Any]], key: str, elapsed_ms: float) -> None:
    if trace is None:
        return
    trace[key] = round(max(float(trace.get(key, 0.0)), elapsed_ms), 3)


def _trace_inc(trace: Optional[Dict[str, Any]], key: str, amount: int = 1) -> None:
    if trace is None:
        return
    trace[key] = int(trace.get(key, 0)) + amount


def _trace_bool_env(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def _trace_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _trace_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _parse_bytes_value(raw: Any) -> int:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    value = str(raw).strip().lower()
    multipliers = {
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
    }
    for suffix, multiplier in multipliers.items():
        if value.endswith(suffix):
            return int(float(value[: -len(suffix)]) * multiplier)
    return int(float(value))


def _parse_bytes_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return _parse_bytes_value(raw)


def _current_pair_trace() -> Optional[Dict[str, Any]]:
    return _PAIR_TRACE.get()


def _pair_trace_add_ms(key: str, elapsed_ms: float) -> None:
    pair_trace = _current_pair_trace()
    if pair_trace is None:
        return
    pair_trace[key] = round(float(pair_trace.get(key, 0.0)) + elapsed_ms, 3)


def _pair_trace_inc(key: str, amount: int = 1) -> None:
    pair_trace = _current_pair_trace()
    if pair_trace is None:
        return
    pair_trace[key] = int(pair_trace.get(key, 0)) + amount


def _pair_trace_max_int(key: str, value: int) -> None:
    pair_trace = _current_pair_trace()
    if pair_trace is None:
        return
    pair_trace[key] = max(int(pair_trace.get(key, 0)), int(value))


def _pair_trace_copy_to_merge_trace(trace: Optional[Dict[str, Any]], pair_trace: Dict[str, Any]) -> None:
    if trace is None:
        return
    for key in (
        "gpu_stage_lora_ms",
        "gpu_select_lora_ms",
        "gpu_matmul_ms",
        "gpu_apply_ms",
        "gpu_bucket_apply_ms",
        "gpu_vocab_tile_ms",
        "gpu_dense_full_ms",
    ):
        if key in pair_trace:
            _trace_add_ms(trace, f"lora_loader_{key}", float(pair_trace[key]))
    for key in ("gpu_bucket_count", "gpu_matmul_count"):
        if key in pair_trace:
            _trace_inc(trace, f"lora_loader_{key}", int(pair_trace[key]))
    if "gpu_bucket_bytes_estimate_max" in pair_trace:
        current = int(trace.get("lora_loader_gpu_bucket_bytes_estimate_max", 0))
        trace["lora_loader_gpu_bucket_bytes_estimate_max"] = max(
            current, int(pair_trace["gpu_bucket_bytes_estimate_max"])
        )


def _tensor_shape(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _maybe_sync_tensor_device(tensor: torch.Tensor) -> None:
    if not _trace_bool_env("SGLANG_LORA_MERGE_TRACE_SYNC"):
        return
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def _memory_snapshot_for_tensor(tensor: torch.Tensor) -> Optional[Dict[str, float]]:
    if tensor.device.type != "cuda" or not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info(tensor.device)
    return {
        "free_gb": round(free_bytes / (1024**3), 3),
        "total_gb": round(total_bytes / (1024**3), 3),
        "allocated_gb": round(torch.cuda.memory_allocated(tensor.device) / (1024**3), 3),
        "reserved_gb": round(torch.cuda.memory_reserved(tensor.device) / (1024**3), 3),
    }


def _memory_snapshot_for_model(model: torch.nn.Module) -> Optional[Dict[str, float]]:
    try:
        param = next(model.parameters())
    except StopIteration:
        return None
    return _memory_snapshot_for_tensor(param)


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


def _manifest_first(manifest: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[Any, str]:
    for key in keys:
        if key in manifest:
            return manifest[key], f"manifest:{key}"
    return None, ""


def _resolve_merge_memory_budget(
    manifest: Dict[str, Any], device: torch.device
) -> _MergeMemoryBudget:
    default = 512 * 1024**2
    raw_budget, source = _manifest_first(
        manifest,
        (
            "peak_device_bytes",
            "lora_merge_peak_device_bytes",
            "gpu_peak_device_bytes",
            "lora_merge_gpu_peak_device_bytes",
            "gpu_bucket_bytes",
            "lora_merge_gpu_bucket_bytes",
        ),
    )
    if raw_budget is not None:
        budget = _parse_bytes_value(raw_budget)
    elif os.environ.get("SGLANG_LORA_MERGE_PEAK_DEVICE_BYTES") is not None:
        budget = _parse_bytes_env("SGLANG_LORA_MERGE_PEAK_DEVICE_BYTES", default)
        source = "env:SGLANG_LORA_MERGE_PEAK_DEVICE_BYTES"
    elif os.environ.get("SGLANG_LORA_MERGE_GPU_BUCKET_BYTES") is not None:
        budget = _parse_bytes_env("SGLANG_LORA_MERGE_GPU_BUCKET_BYTES", default)
        source = "env:SGLANG_LORA_MERGE_GPU_BUCKET_BYTES"
    else:
        budget = default
        source = "default"

    free_bytes = None
    headroom_gb = _trace_float_env("SGLANG_LORA_MERGE_VRAM_HEADROOM_GB", 8.0)
    if "vram_headroom_gb" in manifest:
        headroom_gb = float(manifest["vram_headroom_gb"])
    elif "lora_merge_vram_headroom_gb" in manifest:
        headroom_gb = float(manifest["lora_merge_vram_headroom_gb"])

    if device.type == "cuda" and torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info(device)
        available = int(free_bytes - headroom_gb * 1024**3)
        if available > 0:
            budget = min(budget, available)

    explicit_budget = source != "default"
    min_budget = 1024**2 if explicit_budget else 64 * 1024**2
    peak_bytes = max(min_budget, int(budget))
    return _MergeMemoryBudget(
        peak_bytes=peak_bytes,
        bucket_bytes=peak_bytes,
        source=source,
        headroom_gb=headroom_gb,
        free_bytes=free_bytes,
    )


def _resolve_prestage_bucket_bytes(
    manifest: Dict[str, Any], budget: _MergeMemoryBudget
) -> Tuple[int, float]:
    raw_bucket, _ = _manifest_first(
        manifest, ("apply_bucket_bytes", "lora_merge_apply_bucket_bytes")
    )
    if raw_bucket is not None:
        return max(1, min(budget.peak_bytes, _parse_bytes_value(raw_bucket))), 0.0

    if os.environ.get("SGLANG_LORA_MERGE_APPLY_BUCKET_BYTES") is not None:
        return (
            max(
                1,
                min(
                    budget.peak_bytes,
                    _parse_bytes_env(
                        "SGLANG_LORA_MERGE_APPLY_BUCKET_BYTES", budget.peak_bytes
                    ),
                ),
            ),
            0.0,
        )

    fraction = _trace_float_env("SGLANG_LORA_MERGE_APPLY_BUDGET_FRACTION", 0.5)
    if "apply_budget_fraction" in manifest:
        fraction = float(manifest["apply_budget_fraction"])
    elif "lora_merge_apply_budget_fraction" in manifest:
        fraction = float(manifest["lora_merge_apply_budget_fraction"])
    fraction = min(1.0, max(0.0, fraction))
    return max(1, int(budget.peak_bytes * fraction)), fraction


def _trace_memory_budget(
    trace: Optional[Dict[str, Any]], prefix: str, budget: _MergeMemoryBudget
) -> None:
    if trace is None:
        return
    trace[f"{prefix}_peak_device_budget_bytes"] = int(budget.peak_bytes)
    trace[f"{prefix}_gpu_bucket_bytes"] = int(budget.bucket_bytes)
    trace[f"{prefix}_memory_budget_source"] = budget.source
    trace[f"{prefix}_vram_headroom_gb"] = round(float(budget.headroom_gb), 3)
    if budget.free_bytes is not None:
        trace[f"{prefix}_budget_free_bytes"] = int(budget.free_bytes)


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


def _stage_lora_pair_cuda_fp32(
    pair: _LoraPair, device: torch.device, trace: Optional[Dict[str, Any]]
) -> _LoraPair:
    stage_started_at = time.monotonic()
    staged_a = pair.lora_a.to(device=device, dtype=torch.float32, non_blocking=True)
    staged_b = pair.lora_b.to(device=device, dtype=torch.float32, non_blocking=True)
    _maybe_sync_tensor_device(staged_a)
    _maybe_sync_tensor_device(staged_b)
    _trace_add_ms(
        trace,
        "lora_prestage_stage_lora_ms",
        (time.monotonic() - stage_started_at) * 1000,
    )
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
    return _estimate_dense_temp_bytes(op.lora_a, op.lora_b)


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
    expert_entry_count = 0
    quant_method = getattr(owner_module, "quant_method", None)
    use_unquantized_fused = isinstance(quant_method, UnquantizedFusedMoEMethod)
    for expert_id in range(expert_count):
        if use_unquantized_fused:
            expert_entry_count += len(
                _resolve_moe_expert_ids(owner_module, param, expert_id)
            )
        else:
            expert_entry_count += 1

    if expert_entry_count == 0:
        return 0

    out_dim = int(op.lora_b.shape[-2])
    rank = int(op.lora_a.shape[-2])
    in_dim = int(op.lora_a.shape[-1])
    bytes_per_expert = 4 * (rank * in_dim + out_dim * rank + 2 * out_dim * in_dim)
    experts_per_bucket = max(
        1, min(expert_entry_count, bucket_bytes // max(bytes_per_expert, 1))
    )
    return int(bytes_per_expert * experts_per_bucket)


def prepare_lora_tensors_for_merge(
    model: torch.nn.Module,
    named_tensors: List[Tuple[str, torch.Tensor]],
    load_context: Optional[Dict[str, Any]] = None,
) -> None:
    load_context = load_context or {}
    trace = load_context.get("trace")
    manifest = load_context.get("manifest") or {}
    request_id = _prestage_request_id(manifest, trace)
    strict = manifest.get("strict", True)
    model_device = _model_device(model)
    if model_device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("LoRA merge prestage requires a CUDA model.")

    started_at = time.monotonic()
    if trace is not None:
        trace["lora_prestage_request_id"] = request_id
        trace["lora_prestage_tensor_count"] = len(named_tensors)
        trace["lora_prestage_memory_start"] = _memory_snapshot_for_model(model)

    pairs = _collect_lora_pairs(named_tensors, strict=strict)
    params = dict(model.named_parameters(remove_duplicate=False))
    memory_budget = _resolve_merge_memory_budget(manifest, model_device)
    prestage_bucket_bytes, apply_budget_fraction = _resolve_prestage_bucket_bytes(
        manifest, memory_budget
    )
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
    input_bytes = 0
    for pair in pairs:
        input_bytes += (
            pair.lora_a.numel() * pair.lora_a.element_size()
            + pair.lora_b.numel() * pair.lora_b.element_size()
        )
        pair_staged_bytes = _lora_pair_fp32_bytes(pair)
        if staged_bytes + pair_staged_bytes <= prestage_capacity_bytes:
            staged_pair = _stage_lora_pair_cuda_fp32(pair, model_device, trace)
            cached_pairs.append(staged_pair)
            staged_bytes += pair_staged_bytes
        else:
            cached_pairs.append(pair)
            unstaged_bytes += pair_staged_bytes

    cache = getattr(model, _PRESTAGED_LORA_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(model, _PRESTAGED_LORA_CACHE_ATTR, cache)
    cache[request_id] = {
        "pairs": cached_pairs,
        "staged_bytes": int(staged_bytes),
        "unstaged_bytes": int(unstaged_bytes),
        "max_apply_temp_bytes": int(max_apply_temp_bytes),
        "capacity_bytes": int(prestage_capacity_bytes),
        "peak_device_budget_bytes": int(memory_budget.peak_bytes),
        "gpu_bucket_bytes": int(prestage_bucket_bytes),
        "apply_budget_fraction": float(apply_budget_fraction),
        "memory_budget_source": memory_budget.source,
        "complete": unstaged_bytes == 0,
        "created_monotonic": time.monotonic(),
    }

    if trace is not None:
        _trace_memory_budget(trace, "lora_prestage", memory_budget)
        trace["lora_prestage_gpu_bucket_bytes"] = int(prestage_bucket_bytes)
        trace["lora_prestage_apply_budget_fraction"] = round(
            float(apply_budget_fraction), 3
        )
        trace["lora_prestage_pair_count"] = len(pairs)
        trace["lora_prestage_input_bytes"] = int(input_bytes)
        trace["lora_prestage_staged_bytes"] = int(staged_bytes)
        trace["lora_prestage_unstaged_bytes"] = int(unstaged_bytes)
        trace["lora_prestage_staged_pair_count"] = sum(
            1 for pair in cached_pairs if pair.prestaged
        )
        trace["lora_prestage_unstaged_pair_count"] = sum(
            1 for pair in cached_pairs if not pair.prestaged
        )
        trace["lora_prestage_max_apply_temp_bytes"] = int(max_apply_temp_bytes)
        trace["lora_prestage_capacity_bytes"] = int(prestage_capacity_bytes)
        trace["lora_prestage_complete"] = bool(unstaged_bytes == 0)
        trace["lora_prestage_memory_end"] = _memory_snapshot_for_model(model)
        trace["lora_prestage_total_ms"] = round(
            (time.monotonic() - started_at) * 1000, 3
        )


def _record_pair_trace(trace: Optional[Dict[str, Any]], pair_trace: Dict[str, Any]) -> None:
    if trace is None:
        return

    first_limit = max(0, _trace_int_env("SGLANG_LORA_MERGE_TRACE_FIRST_N", 8))
    first_pairs = trace.setdefault("lora_loader_first_pairs", [])
    if len(first_pairs) < first_limit:
        first_pairs.append(dict(pair_trace))

    top_k = max(0, _trace_int_env("SGLANG_LORA_MERGE_TRACE_TOPK", 8))
    if top_k == 0:
        return

    top_pairs = trace.setdefault("lora_loader_top_pairs", [])
    top_pairs.append(dict(pair_trace))
    top_pairs.sort(key=lambda item: item.get("pair_ms", 0.0), reverse=True)
    del top_pairs[top_k:]


def merge_lora_tensors_inplace(
    model: torch.nn.Module,
    named_tensors: List[Tuple[str, torch.Tensor]],
    load_context: Optional[Dict[str, Any]] = None,
) -> None:
    load_context = load_context or {}
    trace = load_context.get("trace")
    merge_started_at = time.monotonic()
    if trace is not None:
        trace["lora_loader_tensor_count"] = len(named_tensors)
        trace["lora_loader_trace_sync"] = _trace_bool_env(
            "SGLANG_LORA_MERGE_TRACE_SYNC"
        )
        trace["lora_loader_memory_start"] = _memory_snapshot_for_model(model)
    manifest = load_context.get("manifest") or {}
    scaling = _resolve_scaling(manifest)
    strict = manifest.get("strict", True)
    model_device = _require_cuda_model_device(model)
    if manifest.get("added_tokens_config"):
        raise ValueError("Merged LoRA update does not support added tokens yet.")

    params = dict(model.named_parameters(remove_duplicate=False))
    layers_needing_postprocess: Dict[int, torch.nn.Module] = {}
    lora_pairs: List[Optional[_LoraPair]]
    prestage_cache_entry: Optional[Dict[str, Any]] = None
    lora_a = None
    lora_b = None
    memory_budget = _resolve_merge_memory_budget(manifest, model_device)
    gpu_bucket_bytes = memory_budget.bucket_bytes
    using_prestage_cache = False
    if _manifest_bool(manifest, "lora_merge_consume_prestaged"):
        prestage_request_id = _prestage_request_id(manifest, trace)
        cache = getattr(model, _PRESTAGED_LORA_CACHE_ATTR, {})
        prestage_cache_entry = cache.pop(prestage_request_id, None)
        if prestage_cache_entry is None:
            raise ValueError(
                f"No prestaged LoRA merge tensors found for request {prestage_request_id}."
            )
        lora_pairs = list(prestage_cache_entry["pairs"])
        gpu_bucket_bytes = int(
            prestage_cache_entry.get("gpu_bucket_bytes", memory_budget.bucket_bytes)
        )
        using_prestage_cache = True
        if trace is not None:
            trace["lora_loader_prestage_consumed"] = True
            trace["lora_loader_prestage_request_id"] = prestage_request_id
            trace["lora_loader_prestage_pair_count"] = len(lora_pairs)
            trace["lora_loader_prestage_hit_count"] = 0
            trace["lora_loader_prestage_miss_count"] = 0
            trace["lora_loader_prestage_staged_pair_count"] = sum(
                1 for pair in lora_pairs if pair is not None and pair.prestaged
            )
            trace["lora_loader_prestage_unstaged_pair_count"] = sum(
                1 for pair in lora_pairs if pair is not None and not pair.prestaged
            )
            for key in (
                "staged_bytes",
                "unstaged_bytes",
                "max_apply_temp_bytes",
                "capacity_bytes",
                "peak_device_budget_bytes",
                "apply_budget_fraction",
                "memory_budget_source",
                "complete",
            ):
                if key in prestage_cache_entry:
                    trace[f"lora_loader_prestage_{key}"] = prestage_cache_entry[key]
    else:
        lora_pairs = list(_collect_lora_pairs(named_tensors, strict=strict))

    if trace is not None:
        trace["lora_loader_merge_impl"] = "cuda_bucketed"
        _trace_memory_budget(trace, "lora_loader", memory_budget)
        if using_prestage_cache:
            if "peak_device_budget_bytes" in prestage_cache_entry:
                trace["lora_loader_peak_device_budget_bytes"] = int(
                    prestage_cache_entry["peak_device_budget_bytes"]
                )
            if "memory_budget_source" in prestage_cache_entry:
                trace["lora_loader_memory_budget_source"] = prestage_cache_entry[
                    "memory_budget_source"
                ]
            if "max_apply_temp_bytes" in prestage_cache_entry:
                trace["lora_loader_max_apply_temp_bytes"] = int(
                    prestage_cache_entry["max_apply_temp_bytes"]
                )
        trace["lora_loader_gpu_bucket_bytes"] = int(gpu_bucket_bytes)
        trace["lora_loader_pair_input_count"] = len(lora_pairs)

    def apply_pair(
        base_name: str,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        *,
        prestaged: bool,
    ) -> None:
        pair_trace = {
            "index": int(trace.get("lora_loader_pair_count", 0))
            if trace is not None
            else 0,
            "base_name": base_name,
            "prestaged": prestaged,
            "lora_a_shape": _tensor_shape(lora_a),
            "lora_b_shape": _tensor_shape(lora_b),
            "lora_a_dtype": str(lora_a.dtype),
            "lora_b_dtype": str(lora_b.dtype),
            "lora_a_device": str(lora_a.device),
            "lora_b_device": str(lora_b.device),
            "memory_before": _memory_snapshot_for_model(model),
        }
        pair_started_at = time.monotonic()
        token = _PAIR_TRACE.set(pair_trace)
        try:
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
        finally:
            _PAIR_TRACE.reset(token)
        pair_ms = (time.monotonic() - pair_started_at) * 1000
        pair_trace["pair_ms"] = round(pair_ms, 3)
        pair_trace["touched_flashinfer_layer_count"] = len(touched_layers)
        pair_trace["memory_after"] = _memory_snapshot_for_model(model)
        _trace_inc(trace, "lora_loader_pair_count")
        if prestaged:
            _trace_inc(trace, "lora_loader_prestage_hit_count")
        elif using_prestage_cache:
            _trace_inc(trace, "lora_loader_prestage_miss_count")
        _trace_add_ms(trace, "lora_loader_pair_apply_total_ms", pair_ms)
        _trace_max_ms(trace, "lora_loader_pair_apply_max_ms", pair_ms)
        _pair_trace_copy_to_merge_trace(trace, pair_trace)
        _record_pair_trace(trace, pair_trace)
        for layer in touched_layers:
            layers_needing_postprocess[id(layer)] = layer

    try:
        try:
            for pair_index, pair in enumerate(lora_pairs):
                if pair is None:
                    continue
                lora_a = pair.lora_a
                lora_b = pair.lora_b
                apply_pair(
                    pair.base_name,
                    lora_a,
                    lora_b,
                    prestaged=pair.prestaged,
                )
                lora_pairs[pair_index] = None
                pair = None
                lora_a = None
                lora_b = None
        finally:
            lora_pairs = []
            prestage_cache_entry = None
            lora_a = None
            lora_b = None
            finalize_started_at = time.monotonic()
            for layer in layers_needing_postprocess.values():
                _finalize_flashinfer_moe_layer_after_merge(layer)
            finalize_ms = (time.monotonic() - finalize_started_at) * 1000
            _trace_add_ms(trace, "lora_loader_finalize_flashinfer_ms", finalize_ms)
            if trace is not None:
                trace["lora_loader_finalize_layer_count"] = len(
                    layers_needing_postprocess
                )
            if (
                os.environ.get(
                    "SGLANG_LORA_MERGE_EMPTY_CACHE", "1"
                ).lower()
                not in ("0", "false", "no", "off")
            ):
                empty_cache_started_at = time.monotonic()
                torch.cuda.empty_cache()
                _trace_add_ms(
                    trace,
                    "lora_loader_empty_cache_ms",
                    (time.monotonic() - empty_cache_started_at) * 1000,
                )
    finally:
        if trace is not None:
            trace["lora_loader_memory_end"] = _memory_snapshot_for_model(model)
            trace["lora_loader_total_ms"] = round(
                (time.monotonic() - merge_started_at) * 1000, 3
            )


def _resolve_scaling(manifest: Dict[str, Any]) -> float:
    if "scaling" in manifest:
        return float(manifest["scaling"])

    config = manifest.get("config_dict") or manifest.get("adapter_config")
    if config is None:
        raise ValueError(
            "Merged LoRA update requires manifest['config_dict'] or manifest['scaling']."
        )

    lora_alpha = float(config["lora_alpha"])
    rank = int(config["r"])
    if rank <= 0:
        raise ValueError(f"Invalid LoRA rank: {rank}")
    return lora_alpha / rank


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
    resolve_started_at = time.monotonic()
    ops = _resolve_delta_ops(
        base_name=base_name,
        lora_a=lora_a,
        lora_b=lora_b,
        params=params,
        model=model,
    )
    _pair_trace_add_ms(
        "resolve_delta_specs_ms", (time.monotonic() - resolve_started_at) * 1000
    )
    pair_trace = _current_pair_trace()
    if pair_trace is not None:
        pair_trace["spec_count"] = len(ops)
        pair_trace["spec_param_names"] = [
            _op_param_name(op) for op in ops[:8]
        ]

    touched_layers: Set[torch.nn.Module] = set()
    for op in ops:
        apply_started_at = time.monotonic()
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
        _pair_trace_add_ms(
            "apply_loaded_delta_ms", (time.monotonic() - apply_started_at) * 1000
        )
    return touched_layers


def _op_param_name(op: Any) -> str:
    if isinstance(op, _AggregateExpertLoraOp):
        return _aggregate_expert_param_name(op.experts_prefix, op.target)
    return op.param_name


def _aggregate_expert_param_name(experts_prefix: str, target: str) -> str:
    if target in ("w1", "w3"):
        return f"{experts_prefix}.w13_weight"
    if target == "w2":
        return f"{experts_prefix}.w2_weight"
    raise ValueError(f"Unsupported aggregate expert target: {target}")


def _stage_cuda_fp32(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device == device and tensor.dtype == torch.float32:
        return tensor
    started_at = time.monotonic()
    staged = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
    _maybe_sync_tensor_device(staged)
    _pair_trace_add_ms("gpu_stage_lora_ms", (time.monotonic() - started_at) * 1000)
    return staged


def _estimate_dense_temp_bytes(
    lora_a: torch.Tensor, lora_b: torch.Tensor, output_rows: Optional[int] = None
) -> int:
    rows = int(output_rows if output_rows is not None else lora_b.shape[-2])
    rank = int(lora_a.shape[-2])
    cols = int(lora_a.shape[-1])
    # A + B + delta + fp32 destination scratch.
    return 4 * (rank * cols + rows * rank + 2 * rows * cols)


def _record_bucket_estimate(temp_bytes: int) -> None:
    _pair_trace_inc("gpu_bucket_count")
    _pair_trace_max_int("gpu_bucket_bytes_estimate_max", int(temp_bytes))


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

    if _try_apply_vocab_lora_op_cuda_bucketed(
        op=op,
        param=param,
        owner_module=owner_module,
        scaling=scaling,
        bucket_bytes=bucket_bytes,
    ):
        return _get_flashinfer_moe_layer_for_postprocess(param, owner_module=owner_module)

    device = param.device
    temp_bytes = _estimate_dense_temp_bytes(op.lora_a, op.lora_b)
    _record_bucket_estimate(temp_bytes)
    a = _stage_cuda_fp32(op.lora_a, device)
    b = _stage_cuda_fp32(op.lora_b, device)
    matmul_started_at = time.monotonic()
    delta = torch.matmul(b, a).mul_(scaling)
    _maybe_sync_tensor_device(delta)
    _pair_trace_add_ms("gpu_matmul_ms", (time.monotonic() - matmul_started_at) * 1000)
    _pair_trace_inc("gpu_matmul_count")
    pair_trace = _current_pair_trace()
    if pair_trace is not None:
        pair_trace["dense_delta_shape"] = _tensor_shape(delta)
        pair_trace["dense_delta_dtype"] = str(delta.dtype)

    apply_started_at = time.monotonic()
    _apply_loaded_delta(
        param_name=op.param_name,
        param=param,
        owner_module=owner_module,
        loaded_weight=delta,
        loader_args=op.loader_args,
    )
    _pair_trace_add_ms("gpu_apply_ms", (time.monotonic() - apply_started_at) * 1000)
    _pair_trace_add_ms("gpu_dense_full_ms", (time.monotonic() - matmul_started_at) * 1000)
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

    pair_trace = _current_pair_trace()
    if pair_trace is not None:
        pair_trace["dense_delta_shape"] = (int(shard_size), cols)
        pair_trace["dense_delta_dtype"] = "torch.float32"

    row = 0
    while row < shard_size:
        rows = min(rows_per_bucket, shard_size - row)
        temp_bytes = _estimate_dense_temp_bytes(op.lora_a, op.lora_b, rows)
        _record_bucket_estimate(temp_bytes)
        bucket_started_at = time.monotonic()
        b = _stage_cuda_fp32(op.lora_b.narrow(0, source_start + row, rows), device)
        matmul_started_at = time.monotonic()
        delta = torch.matmul(b, a).mul_(scaling)
        _maybe_sync_tensor_device(delta)
        _pair_trace_add_ms(
            "gpu_matmul_ms", (time.monotonic() - matmul_started_at) * 1000
        )
        _pair_trace_inc("gpu_matmul_count")
        apply_started_at = time.monotonic()
        _add_local_delta_fp32_(dst_base.narrow(0, row, rows), delta)
        _pair_trace_add_ms("gpu_apply_ms", (time.monotonic() - apply_started_at) * 1000)
        _pair_trace_add_ms(
            "gpu_vocab_tile_ms", (time.monotonic() - bucket_started_at) * 1000
        )
        row += rows

    return True


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
    if use_unquantized_fused:
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
        return {touched_layer} if touched_layer is not None else set()

    out_dim = int(op.lora_b.shape[-2])
    rank = int(op.lora_a.shape[-2])
    in_dim = int(op.lora_a.shape[-1])
    bytes_per_expert = 4 * (rank * in_dim + out_dim * rank + 2 * out_dim * in_dim)
    experts_per_bucket = max(
        1, min(len(expert_entries), bucket_bytes // max(bytes_per_expert, 1))
    )
    device = param.device

    pair_trace = _current_pair_trace()
    if pair_trace is not None:
        pair_trace["dense_delta_shape"] = (out_dim, in_dim)
        pair_trace["dense_delta_dtype"] = "torch.float32"

    for start in range(0, len(expert_entries), experts_per_bucket):
        chunk = expert_entries[start : start + experts_per_bucket]
        source_expert_ids = [entry[0] for entry in chunk]
        temp_bytes = bytes_per_expert * len(chunk)
        _record_bucket_estimate(temp_bytes)
        bucket_started_at = time.monotonic()
        a = _select_expert_chunk_cuda_fp32(
            op.lora_a, source_expert_ids, expert_count, device
        )
        b = _select_expert_chunk_cuda_fp32(
            op.lora_b, source_expert_ids, expert_count, device
        )
        matmul_started_at = time.monotonic()
        deltas = torch.bmm(b, a).mul_(scaling)
        _maybe_sync_tensor_device(deltas)
        _pair_trace_add_ms("gpu_matmul_ms", (time.monotonic() - matmul_started_at) * 1000)
        _pair_trace_inc("gpu_matmul_count")

        apply_started_at = time.monotonic()
        for delta_idx, (_, local_expert_id) in enumerate(chunk):
            loaded_weight = deltas[delta_idx]
            if use_unquantized_fused:
                _apply_unquantized_fused_moe_delta_to_local_expert(
                    param_name=param_name,
                    param=param,
                    owner_module=owner_module,
                    loaded_weight=loaded_weight,
                    shard_id=op.target,
                    local_expert_id=local_expert_id,
                )
            else:
                _apply_simple_expert_delta(
                    param_name=param_name,
                    param=param,
                    loaded_weight=loaded_weight,
                    shard_id=op.target,
                    expert_id=local_expert_id,
                )
        _pair_trace_add_ms("gpu_apply_ms", (time.monotonic() - apply_started_at) * 1000)
        _pair_trace_add_ms(
            "gpu_bucket_apply_ms", (time.monotonic() - bucket_started_at) * 1000
        )

    return {touched_layer} if touched_layer is not None else set()


def _select_expert_chunk_cuda_fp32(
    tensor: torch.Tensor,
    expert_ids: List[int],
    expert_count: int,
    device: torch.device,
) -> torch.Tensor:
    started_at = time.monotonic()
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
            f"Mismatched expert dimensions in MoE LoRA tensor: got {tensor.shape[0]}, expected {expert_count}"
        )
    if selected.device == device and selected.dtype == torch.float32:
        staged = selected
        trace_key = "gpu_select_lora_ms"
    else:
        staged = selected.to(device=device, dtype=torch.float32, non_blocking=True)
        trace_key = "gpu_stage_lora_ms"
    _maybe_sync_tensor_device(staged)
    _pair_trace_add_ms(trace_key, (time.monotonic() - started_at) * 1000)
    return staged


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
    _pair_trace_inc("gpu_add_calls")
    pair_trace = _current_pair_trace()
    if pair_trace is not None:
        pair_trace["gpu_add_bytes"] = int(pair_trace.get("gpu_add_bytes", 0)) + int(
            delta.numel() * 4
        )

    _maybe_sync_tensor_device(dst_view)
    h2d_started_at = time.monotonic()
    delta_fp32 = delta.to(device=dst_view.device, dtype=torch.float32)
    _maybe_sync_tensor_device(dst_view)
    _pair_trace_add_ms("gpu_delta_to_device_ms", (time.monotonic() - h2d_started_at) * 1000)

    dst_to_fp32_started_at = time.monotonic()
    updated = dst_view.to(device=dst_view.device, dtype=torch.float32)
    _maybe_sync_tensor_device(dst_view)
    _pair_trace_add_ms(
        "gpu_dst_to_fp32_ms", (time.monotonic() - dst_to_fp32_started_at) * 1000
    )

    add_started_at = time.monotonic()
    updated.add_(delta_fp32)
    _maybe_sync_tensor_device(dst_view)
    _pair_trace_add_ms("gpu_add_ms", (time.monotonic() - add_started_at) * 1000)

    copy_started_at = time.monotonic()
    dst_view.copy_(updated)
    _maybe_sync_tensor_device(dst_view)
    _pair_trace_add_ms("gpu_copy_back_ms", (time.monotonic() - copy_started_at) * 1000)


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


def _narrow_loaded_weight(
    loaded_weight: torch.Tensor,
    dim: int,
    start_idx: int,
    shard_size: int,
    *,
    pad_if_needed: bool,
) -> torch.Tensor:
    end_idx = start_idx + shard_size
    if pad_if_needed and end_idx > loaded_weight.shape[dim]:
        return pad_or_narrow_weight(loaded_weight, dim, start_idx, shard_size)
    return loaded_weight.narrow(dim, start_idx, shard_size)


def _normalize_generic_shard_id(loaded_shard_id: Any) -> int:
    shard_map = {"q": 0, "k": 1, "v": 2}
    if loaded_shard_id in shard_map:
        return shard_map[loaded_shard_id]
    if isinstance(loaded_shard_id, int):
        return loaded_shard_id
    raise ValueError(f"Unsupported packed shard id for merged update: {loaded_shard_id}")


def _apply_direct_delta(
    param_name: str,
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
) -> None:
    if tuple(param.data.shape) != tuple(loaded_weight.shape):
        raise ValueError(
            f"Shape mismatch for direct delta add on {param_name}: "
            f"param={tuple(param.data.shape)} loaded={tuple(loaded_weight.shape)}"
        )
    _add_local_delta_fp32_(param.data, loaded_weight)


def _apply_generic_packed_delta(
    param_name: str,
    param: torch.nn.Parameter,
    owner_module: torch.nn.Module,
    loaded_weight: torch.Tensor,
    loaded_shard_id: Any,
) -> None:
    output_sizes = list(getattr(owner_module, "output_sizes"))
    if isinstance(loaded_shard_id, tuple):
        offset = 0
        for shard_id in loaded_shard_id:
            shard_idx = _normalize_generic_shard_id(shard_id)
            shard_size = output_sizes[shard_idx]
            shard = loaded_weight.narrow(0, offset, shard_size)
            _apply_generic_packed_delta(
                param_name, param, owner_module, shard, shard_id
            )
            offset += shard_size
        return

    if loaded_shard_id is None:
        if tuple(param.data.shape) == tuple(loaded_weight.shape):
            _add_local_delta_fp32_(param.data, loaded_weight)
            return

        offset = 0
        for shard_idx, shard_size in enumerate(output_sizes):
            shard = loaded_weight.narrow(0, offset, shard_size)
            _apply_generic_packed_delta(
                param_name, param, owner_module, shard, shard_idx
            )
            offset += shard_size
        return

    shard_idx = _normalize_generic_shard_id(loaded_shard_id)
    start = sum(output_sizes[:shard_idx])
    shard_size = output_sizes[shard_idx]
    dst_view = param.data.narrow(0, start, shard_size)
    if tuple(dst_view.shape) != tuple(loaded_weight.shape):
        raise ValueError(
            f"Shape mismatch for packed delta add on {param_name}: "
            f"dst={tuple(dst_view.shape)} loaded={tuple(loaded_weight.shape)}"
        )
    _add_local_delta_fp32_(dst_view, loaded_weight)


def _apply_column_parallel_delta(
    param_name: str,
    param: torch.nn.Parameter,
    owner_module: ColumnParallelLinear,
    loaded_weight: torch.Tensor,
) -> None:
    _assert_no_special_loader_features(param, param_name=param_name)
    param_data = param.data
    output_dim = getattr(param, "output_dim", None)

    if output_dim is not None and not owner_module.use_presharded_weights:
        shard_size = param_data.shape[output_dim]
        start_idx = owner_module.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

    if len(loaded_weight.shape) == 0:
        loaded_weight = loaded_weight.reshape(1)

    if tuple(param_data.shape) != tuple(loaded_weight.shape):
        raise ValueError(
            f"Shape mismatch for column-parallel delta add on {param_name}: "
            f"param={tuple(param_data.shape)} loaded={tuple(loaded_weight.shape)}"
        )
    _add_local_delta_fp32_(param_data, loaded_weight)


def _apply_merged_column_delta(
    param_name: str,
    param: torch.nn.Parameter,
    owner_module: MergedColumnParallelLinear,
    loaded_weight: torch.Tensor,
    loaded_shard_id: Any,
) -> None:
    _assert_no_special_loader_features(param, param_name=param_name)
    output_dim = getattr(param, "output_dim", None)
    if output_dim is None:
        raise ValueError(
            f"Merged LoRA update requires output_dim for merged-column param {param_name}."
        )

    if isinstance(loaded_shard_id, tuple):
        offset = 0
        for shard_id in loaded_shard_id:
            shard_size = owner_module.output_sizes[shard_id]
            shard = loaded_weight.narrow(output_dim, offset, shard_size)
            _apply_merged_column_delta(
                param_name, param, owner_module, shard, shard_id
            )
            offset += shard_size
        return

    if loaded_shard_id is None:
        if owner_module.use_presharded_weights and tuple(param.data.shape) == tuple(
            loaded_weight.shape
        ):
            _add_local_delta_fp32_(param.data, loaded_weight)
            return

        current_shard_offset = 0
        for shard_id, output_size in enumerate(owner_module.output_sizes):
            effective_size = (
                output_size // owner_module.tp_size
                if owner_module.use_presharded_weights
                else output_size
            )
            shard = loaded_weight.narrow(output_dim, current_shard_offset, effective_size)
            _apply_merged_column_delta(
                param_name, param, owner_module, shard, shard_id
            )
            current_shard_offset += effective_size
        return

    shard_offset = sum(owner_module.output_sizes[:loaded_shard_id]) // owner_module.tp_size
    shard_size = owner_module.output_sizes[loaded_shard_id] // owner_module.tp_size
    dst_view = param.data.narrow(output_dim, shard_offset, shard_size)

    if not owner_module.use_presharded_weights:
        start_idx = owner_module.tp_rank * shard_size
        loaded_weight = _narrow_loaded_weight(
            loaded_weight,
            output_dim,
            start_idx,
            shard_size,
            pad_if_needed=True,
        )

    if tuple(dst_view.shape) != tuple(loaded_weight.shape):
        raise ValueError(
            f"Shape mismatch for merged-column delta add on {param_name}: "
            f"dst={tuple(dst_view.shape)} loaded={tuple(loaded_weight.shape)}"
        )
    _add_local_delta_fp32_(dst_view, loaded_weight)


def _apply_qkv_parallel_delta(
    param_name: str,
    param: torch.nn.Parameter,
    owner_module: QKVParallelLinear,
    loaded_weight: torch.Tensor,
    loaded_shard_id: Optional[str],
) -> None:
    _assert_no_special_loader_features(param, param_name=param_name)
    output_dim = getattr(param, "output_dim", None)
    if output_dim is None:
        raise ValueError(
            f"Merged LoRA update requires output_dim for qkv param {param_name}."
        )

    if loaded_shard_id is None:
        if owner_module.use_presharded_weights and tuple(param.data.shape) == tuple(
            loaded_weight.shape
        ):
            _add_local_delta_fp32_(param.data, loaded_weight)
            return

        shard_offsets = [
            ("q", 0, owner_module.total_num_heads * owner_module.head_size),
            (
                "k",
                owner_module.total_num_heads * owner_module.head_size,
                owner_module.total_num_kv_heads * owner_module.head_size,
            ),
            (
                "v",
                (owner_module.total_num_heads + owner_module.total_num_kv_heads)
                * owner_module.head_size,
                owner_module.total_num_kv_heads * owner_module.v_head_size,
            ),
        ]
        for shard_id, shard_offset, shard_size in shard_offsets:
            shard = loaded_weight.narrow(output_dim, shard_offset, shard_size)
            _apply_qkv_parallel_delta(param_name, param, owner_module, shard, shard_id)
        return

    if loaded_shard_id not in ("q", "k", "v"):
        raise ValueError(
            f"Unsupported qkv shard id for merged update on {param_name}: {loaded_shard_id}"
        )

    if loaded_shard_id == "q":
        shard_offset = 0
        shard_size = owner_module.num_heads * owner_module.head_size
        tp_shard_id = owner_module.tp_rank
    elif loaded_shard_id == "k":
        shard_offset = owner_module.num_heads * owner_module.head_size
        shard_size = owner_module.num_kv_heads * owner_module.head_size
        tp_shard_id = owner_module.tp_rank // owner_module.num_kv_head_replicas
    else:
        shard_offset = (
            owner_module.num_heads + owner_module.num_kv_heads
        ) * owner_module.head_size
        shard_size = owner_module.num_kv_heads * owner_module.v_head_size
        tp_shard_id = owner_module.tp_rank // owner_module.num_kv_head_replicas

    dst_view = param.data.narrow(output_dim, shard_offset, shard_size)
    if not owner_module.use_presharded_weights:
        start_idx = tp_shard_id * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

    if tuple(dst_view.shape) != tuple(loaded_weight.shape):
        raise ValueError(
            f"Shape mismatch for qkv delta add on {param_name}: "
            f"dst={tuple(dst_view.shape)} loaded={tuple(loaded_weight.shape)}"
        )
    _add_local_delta_fp32_(dst_view, loaded_weight)


def _apply_row_parallel_delta(
    param_name: str,
    param: torch.nn.Parameter,
    owner_module: RowParallelLinear,
    loaded_weight: torch.Tensor,
) -> None:
    _assert_no_special_loader_features(param, param_name=param_name)
    param_data = param.data
    input_dim = getattr(param, "input_dim", None)

    if input_dim is not None and not owner_module.use_presharded_weights:
        shard_size = param_data.shape[input_dim]
        start_idx = owner_module.tp_rank * shard_size
        loaded_weight = _narrow_loaded_weight(
            loaded_weight,
            input_dim,
            start_idx,
            shard_size,
            pad_if_needed=True,
        )

    if len(loaded_weight.shape) == 0:
        loaded_weight = loaded_weight.reshape(1)

    if tuple(param_data.shape) != tuple(loaded_weight.shape):
        raise ValueError(
            f"Shape mismatch for row-parallel delta add on {param_name}: "
            f"param={tuple(param_data.shape)} loaded={tuple(loaded_weight.shape)}"
        )
    _add_local_delta_fp32_(param_data, loaded_weight)


def _apply_vocab_parallel_delta(
    param_name: str,
    param: torch.nn.Parameter,
    owner_module: VocabParallelEmbedding,
    loaded_weight: torch.Tensor,
) -> None:
    _assert_no_special_loader_features(param, param_name=param_name)
    output_dim = getattr(param, "output_dim", None)
    if output_dim is None:
        _apply_direct_delta(param_name, param, loaded_weight)
        return

    start_idx = owner_module.shard_indices.org_vocab_start_index
    shard_size = owner_module.shard_indices.org_vocab_end_index - start_idx
    if not owner_module.use_presharded_weights:
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

    dst_rows = loaded_weight.shape[0]
    dst_view = param[:dst_rows].data
    if tuple(dst_view.shape) != tuple(loaded_weight.shape):
        raise ValueError(
            f"Shape mismatch for vocab-parallel delta add on {param_name}: "
            f"dst={tuple(dst_view.shape)} loaded={tuple(loaded_weight.shape)}"
        )
    _add_local_delta_fp32_(dst_view, loaded_weight)


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


def _apply_simple_expert_delta(
    param_name: str,
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str,
    expert_id: int,
) -> None:
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

    if tuple(dst_view.shape) != tuple(loaded_weight.shape):
        raise ValueError(
            f"Shape mismatch for expert delta add on {param_name}: "
            f"dst={tuple(dst_view.shape)} loaded={tuple(loaded_weight.shape)}"
        )
    _add_local_delta_fp32_(dst_view, loaded_weight)


def _apply_unquantized_fused_moe_delta(
    param_name: str,
    param: torch.nn.Parameter,
    owner_module: torch.nn.Module,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
) -> None:
    quant_method = getattr(owner_module, "quant_method", None)
    if not isinstance(quant_method, UnquantizedFusedMoEMethod):
        raise ValueError(
            f"Merged LoRA update only supports unquantized fused MoE weights for {param_name}."
        )

    quant_method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
        layer=owner_module,
        param=param,
        weight_name=weight_name,
    )

    use_flashinfer = getattr(quant_method, "use_flashinfer_trtllm_moe", False)
    # Hot-merge restores FlashInfer TRT-LLM BF16 weights back into true
    # checkpoint/canonical load layout before applying the additive delta.
    # Unlike the regular weight_loader path, we should therefore keep the
    # caller's logical shard id unchanged here.

    local_expert_ids = _resolve_moe_expert_ids(owner_module, param, expert_id)
    if not local_expert_ids:
        return

    tp_rank = getattr(owner_module, "moe_tp_rank", 0)
    use_presharded_weights = getattr(owner_module, "use_presharded_weights", False)
    use_triton_kernels = getattr(owner_module, "use_triton_kernels", False)
    if use_triton_kernels:
        loaded_weight = loaded_weight.transpose(-2, -1).contiguous()

    for local_expert_id in local_expert_ids:
        expert_data = param.data[local_expert_id]
        shard_dim = {"w1": 0, "w2": 1, "w3": 0}[shard_id]
        if use_triton_kernels:
            shard_dim = int(not shard_dim)

        if shard_id in ("w1", "w3"):
            shard_size = expert_data.shape[shard_dim] // 2
            switch_w13 = getattr(
                quant_method, "load_up_proj_weight_first", False
            )
            start = (
                shard_size
                if (
                    (switch_w13 and shard_id == "w1")
                    or (not switch_w13 and shard_id == "w3")
                )
                and getattr(owner_module.moe_runner_config, "is_gated", False)
                else 0
            )
            use_padded_loading = is_cpu() or use_flashinfer
            if use_padded_loading:
                dst_view, local_loaded_weight = narrow_padded_param_and_loaded_weight(
                    expert_data,
                    loaded_weight,
                    start,
                    shard_size * tp_rank,
                    shard_dim,
                    shard_size,
                    not use_presharded_weights,
                )
            else:
                dst_view = expert_data.narrow(shard_dim, start, shard_size)
                local_loaded_weight = loaded_weight
                if not use_presharded_weights:
                    local_loaded_weight = local_loaded_weight.narrow(
                        shard_dim, shard_size * tp_rank, shard_size
                    )
        elif shard_id == "w2":
            shard_size = expert_data.shape[shard_dim]
            use_padded_loading = is_cpu() or use_flashinfer
            if use_padded_loading:
                dst_view, local_loaded_weight = narrow_padded_param_and_loaded_weight(
                    expert_data,
                    loaded_weight,
                    0,
                    shard_size * tp_rank,
                    shard_dim,
                    shard_size,
                    not use_presharded_weights,
                )
            else:
                dst_view = expert_data
                local_loaded_weight = loaded_weight
                if not use_presharded_weights:
                    local_loaded_weight = local_loaded_weight.narrow(
                        shard_dim, shard_size * tp_rank, shard_size
                    )
        else:
            raise ValueError(
                f"Unsupported MoE shard id for merged update on {param_name}: {shard_id}"
            )

        if tuple(dst_view.shape) != tuple(local_loaded_weight.shape):
            raise ValueError(
                f"Shape mismatch for fused MoE delta add on {param_name}: "
                f"dst={tuple(dst_view.shape)} loaded={tuple(local_loaded_weight.shape)}"
            )
        _add_local_delta_fp32_(dst_view, local_loaded_weight)


def _apply_unquantized_fused_moe_delta_to_local_expert(
    param_name: str,
    param: torch.nn.Parameter,
    owner_module: torch.nn.Module,
    loaded_weight: torch.Tensor,
    shard_id: str,
    local_expert_id: int,
) -> None:
    quant_method = getattr(owner_module, "quant_method", None)
    if not isinstance(quant_method, UnquantizedFusedMoEMethod):
        raise ValueError(
            f"Merged LoRA update only supports unquantized fused MoE weights for {param_name}."
        )

    use_flashinfer = getattr(quant_method, "use_flashinfer_trtllm_moe", False)
    tp_rank = getattr(owner_module, "moe_tp_rank", 0)
    use_presharded_weights = getattr(owner_module, "use_presharded_weights", False)
    use_triton_kernels = getattr(owner_module, "use_triton_kernels", False)
    if use_triton_kernels:
        loaded_weight = loaded_weight.transpose(-2, -1).contiguous()

    expert_data = param.data[local_expert_id]
    shard_dim = {"w1": 0, "w2": 1, "w3": 0}[shard_id]
    if use_triton_kernels:
        shard_dim = int(not shard_dim)

    if shard_id in ("w1", "w3"):
        shard_size = expert_data.shape[shard_dim] // 2
        switch_w13 = getattr(quant_method, "load_up_proj_weight_first", False)
        start = (
            shard_size
            if (
                (switch_w13 and shard_id == "w1")
                or (not switch_w13 and shard_id == "w3")
            )
            and getattr(owner_module.moe_runner_config, "is_gated", False)
            else 0
        )
        use_padded_loading = is_cpu() or use_flashinfer
        if use_padded_loading:
            dst_view, local_loaded_weight = narrow_padded_param_and_loaded_weight(
                expert_data,
                loaded_weight,
                start,
                shard_size * tp_rank,
                shard_dim,
                shard_size,
                not use_presharded_weights,
            )
        else:
            dst_view = expert_data.narrow(shard_dim, start, shard_size)
            local_loaded_weight = loaded_weight
            if not use_presharded_weights:
                local_loaded_weight = local_loaded_weight.narrow(
                    shard_dim, shard_size * tp_rank, shard_size
                )
    elif shard_id == "w2":
        shard_size = expert_data.shape[shard_dim]
        use_padded_loading = is_cpu() or use_flashinfer
        if use_padded_loading:
            dst_view, local_loaded_weight = narrow_padded_param_and_loaded_weight(
                expert_data,
                loaded_weight,
                0,
                shard_size * tp_rank,
                shard_dim,
                shard_size,
                not use_presharded_weights,
            )
        else:
            dst_view = expert_data
            local_loaded_weight = loaded_weight
            if not use_presharded_weights:
                local_loaded_weight = local_loaded_weight.narrow(
                    shard_dim, shard_size * tp_rank, shard_size
                )
    else:
        raise ValueError(
            f"Unsupported MoE shard id for merged update on {param_name}: {shard_id}"
        )

    if tuple(dst_view.shape) != tuple(local_loaded_weight.shape):
        raise ValueError(
            f"Shape mismatch for fused MoE delta add on {param_name}: "
            f"dst={tuple(dst_view.shape)} loaded={tuple(local_loaded_weight.shape)}"
        )
    _add_local_delta_fp32_(dst_view, local_loaded_weight)


def _apply_loaded_delta(
    param_name: str,
    param: torch.nn.Parameter,
    owner_module: Optional[torch.nn.Module],
    loaded_weight: torch.Tensor,
    loader_args: Tuple[Any, ...],
) -> None:
    if (
        owner_module is not None
        and len(loader_args) == 3
        and hasattr(owner_module, "w13_weight")
        and hasattr(owner_module, "w2_weight")
    ):
        weight_name, shard_id, expert_id = loader_args
        quant_method = getattr(owner_module, "quant_method", None)
        if isinstance(quant_method, UnquantizedFusedMoEMethod):
            _apply_unquantized_fused_moe_delta(
                param_name=param_name,
                param=param,
                owner_module=owner_module,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            return
        _apply_simple_expert_delta(
            param_name=param_name,
            param=param,
            loaded_weight=loaded_weight,
            shard_id=shard_id,
            expert_id=expert_id,
        )
        return

    if isinstance(owner_module, QKVParallelLinear):
        loaded_shard_id = loader_args[0] if loader_args else None
        _apply_qkv_parallel_delta(
            param_name, param, owner_module, loaded_weight, loaded_shard_id
        )
        return

    if isinstance(owner_module, MergedColumnParallelLinear):
        loaded_shard_id = loader_args[0] if loader_args else None
        _apply_merged_column_delta(
            param_name, param, owner_module, loaded_weight, loaded_shard_id
        )
        return

    if isinstance(owner_module, RowParallelLinear):
        if loader_args:
            raise ValueError(
                f"Unexpected loader args for row-parallel merged update on {param_name}: {loader_args}"
            )
        _apply_row_parallel_delta(param_name, param, owner_module, loaded_weight)
        return

    if isinstance(owner_module, (ParallelLMHead, VocabParallelEmbedding)):
        if loader_args:
            raise ValueError(
                f"Unexpected loader args for vocab-parallel merged update on {param_name}: {loader_args}"
            )
        _apply_vocab_parallel_delta(param_name, param, owner_module, loaded_weight)
        return

    if isinstance(owner_module, ColumnParallelLinear):
        if loader_args:
            raise ValueError(
                f"Unexpected loader args for column-parallel merged update on {param_name}: {loader_args}"
            )
        _apply_column_parallel_delta(param_name, param, owner_module, loaded_weight)
        return

    if owner_module is not None and hasattr(owner_module, "output_sizes"):
        loaded_shard_id = loader_args[0] if loader_args else None
        _apply_generic_packed_delta(
            param_name, param, owner_module, loaded_weight, loaded_shard_id
        )
        return

    if loader_args:
        raise ValueError(
            f"Unsupported loader-backed merged update target for {param_name}: loader_args={loader_args}"
        )
    _apply_direct_delta(param_name, param, loaded_weight)


def _get_flashinfer_moe_layer_for_postprocess(
    param: torch.nn.Parameter,
    owner_module: Optional[torch.nn.Module] = None,
) -> Optional[torch.nn.Module]:
    layer = owner_module
    if layer is None:
        weight_loader = getattr(param, "weight_loader", None)
        layer = getattr(weight_loader, "__self__", None)
    if layer is None:
        return None

    quant_method = getattr(layer, "quant_method", None)
    if not isinstance(quant_method, UnquantizedFusedMoEMethod):
        return None
    if not getattr(quant_method, "use_flashinfer_trtllm_moe", False):
        return None
    if not hasattr(layer, "w13_weight") or not hasattr(layer, "w2_weight"):
        return None
    return layer


def _finalize_flashinfer_moe_layer_after_merge(layer: torch.nn.Module) -> None:
    quant_method = getattr(layer, "quant_method", None)
    if not isinstance(quant_method, UnquantizedFusedMoEMethod):
        return
    if getattr(quant_method, "use_flashinfer_trtllm_moe", False):
        # Partial routed-expert merges only restore the weight tensor that the
        # active loader touched. The untouched sibling can still be in blocked
        # FlashInfer TRT-LLM layout, so restore both canonical load shapes
        # before re-running the shared postprocess.
        for weight_name in ("w13_weight", "w2_weight"):
            param = getattr(layer, weight_name, None)
            if param is None:
                continue
            quant_method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
                layer,
                param,
                f"merged_update.experts.{weight_name}",
            )
    quant_method.process_weights_after_loading(layer)


merge_lora_tensors_inplace.sglang_supports_host_tensors = True
merge_lora_tensors_inplace.sglang_requires_cuda_graph_recapture = True
merge_lora_tensors_inplace.sglang_prepare_tensors = prepare_lora_tensors_for_merge
