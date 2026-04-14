import logging
import re
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

_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_EXPLICIT_EXPERT_RE = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<expert_id>\d+)\.(?P<target>[^.]+)$"
)


@dataclass(frozen=True)
class _DeltaSpec:
    param_name: str
    loaded_weight: torch.Tensor
    loader_args: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class _AliasRule:
    source_suffix: str
    dest_suffix: str
    loader_args: Tuple[Any, ...] = ()


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


def merge_lora_tensors_inplace(
    model: torch.nn.Module,
    named_tensors: List[Tuple[str, torch.Tensor]],
    load_context: Optional[Dict[str, Any]] = None,
) -> None:
    manifest = (load_context or {}).get("manifest") or {}
    scaling = _resolve_scaling(manifest)
    strict = manifest.get("strict", True)
    if manifest.get("added_tokens_config"):
        raise ValueError("Merged LoRA update does not support added tokens yet.")

    params = dict(model.named_parameters(remove_duplicate=False))
    pending_pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    layers_needing_postprocess: Dict[int, torch.nn.Module] = {}

    try:
        for name, tensor in named_tensors:
            base_name, kind = _split_lora_tensor_name(name)
            if kind not in ("A", "B"):
                raise ValueError(
                    f"Unsupported LoRA tensor name for merged update: {name}"
                )

            base_name = _canonicalize_lora_base_name(base_name)
            pair = pending_pairs.setdefault(base_name, {})
            if kind in pair:
                raise ValueError(f"Duplicate LoRA tensor for {base_name}: {kind}")
            pair[kind] = tensor

            if "A" not in pair or "B" not in pair:
                continue

            touched_layers = _apply_lora_pair(
                base_name=base_name,
                lora_a=pair["A"],
                lora_b=pair["B"],
                scaling=scaling,
                params=params,
                model=model,
                strict=strict,
            )
            for layer in touched_layers:
                layers_needing_postprocess[id(layer)] = layer
            del pending_pairs[base_name]

        if strict and pending_pairs:
            incomplete = ", ".join(sorted(pending_pairs.keys()))
            raise ValueError(f"Incomplete LoRA pairs for merged update: {incomplete}")
    finally:
        for layer in layers_needing_postprocess.values():
            _finalize_flashinfer_moe_layer_after_merge(layer)


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


def _resolve_delta_specs(
    base_name: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    params: Dict[str, torch.nn.Parameter],
    model: torch.nn.Module,
) -> List[_DeltaSpec]:
    if _is_unembed_target(base_name):
        return _resolve_unembed_delta_specs(
            base_name=base_name,
            lora_a=lora_a,
            lora_b=lora_b,
            scaling=scaling,
            params=params,
            model=model,
        )

    explicit_match = _EXPLICIT_EXPERT_RE.match(base_name)
    if explicit_match:
        return _resolve_explicit_expert_specs(
            experts_prefix=explicit_match.group("prefix"),
            expert_id=int(explicit_match.group("expert_id")),
            target=explicit_match.group("target"),
            lora_a=lora_a,
            lora_b=lora_b,
            scaling=scaling,
            explicit_name=base_name,
        )

    aggregate_match = _match_aggregate_expert_target(base_name)
    if aggregate_match is not None:
        experts_prefix, target = aggregate_match
        return _resolve_aggregate_expert_specs(
            experts_prefix=experts_prefix,
            target=target,
            lora_a=lora_a,
            lora_b=lora_b,
            scaling=scaling,
        )

    direct_param_name = _resolve_direct_param_name(base_name, params)
    if direct_param_name is not None:
        return [
            _DeltaSpec(
                param_name=direct_param_name,
                loaded_weight=_dense_delta(lora_a, lora_b, scaling),
            )
        ]

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
                _DeltaSpec(
                    param_name=dest_param_name,
                    loaded_weight=_dense_delta(lora_a, lora_b, scaling),
                    loader_args=rule.loader_args,
                )
            ]

    raise ValueError(f"Unsupported LoRA target for merged update: {base_name}")


def _apply_lora_pair(
    base_name: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    params: Dict[str, torch.nn.Parameter],
    model: torch.nn.Module,
    strict: bool,
) -> Set[torch.nn.Module]:
    specs = _resolve_delta_specs(
        base_name=base_name,
        lora_a=lora_a,
        lora_b=lora_b,
        scaling=scaling,
        params=params,
        model=model,
    )
    touched_layers: Set[torch.nn.Module] = set()

    for spec in specs:
        if spec.param_name not in params:
            if strict:
                raise ValueError(
                    f"Target parameter not found for merged LoRA update: {spec.param_name}"
                )
            logger.warning("Skipping unknown merged LoRA target: %s", spec.param_name)
            continue

        param = params[spec.param_name]
        _ensure_supported_param(param, spec.param_name)
        owner_module = _resolve_param_owner_module(model, spec.param_name, param)
        _apply_loaded_delta(
            param_name=spec.param_name,
            param=param,
            owner_module=owner_module,
            loaded_weight=spec.loaded_weight,
            loader_args=spec.loader_args,
        )
        layer = _get_flashinfer_moe_layer_for_postprocess(
            param, owner_module=owner_module
        )
        if layer is not None:
            touched_layers.add(layer)

    return touched_layers


def _is_unembed_target(base_name: str) -> bool:
    return base_name == "unembed_tokens" or base_name.endswith(".unembed_tokens")


def _resolve_unembed_delta_specs(
    base_name: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    params: Dict[str, torch.nn.Parameter],
    model: torch.nn.Module,
) -> List[_DeltaSpec]:
    delta = _dense_delta(lora_a, lora_b, scaling)
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

    deduped_specs: List[_DeltaSpec] = []
    seen_param_ids: Set[int] = set()
    for target_name in target_names:
        param = params.get(target_name)
        if param is None:
            continue
        param_id = id(param)
        if param_id in seen_param_ids:
            continue
        seen_param_ids.add(param_id)
        deduped_specs.append(
            _DeltaSpec(param_name=target_name, loaded_weight=delta)
        )

    if deduped_specs:
        return deduped_specs

    raise ValueError(f"Unsupported LoRA target for merged update: {base_name}")


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


def _resolve_aggregate_expert_specs(
    experts_prefix: str,
    target: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
) -> List[_DeltaSpec]:
    expert_count = max(_expert_axis_size(lora_a), _expert_axis_size(lora_b))
    specs: List[_DeltaSpec] = []
    for expert_id in range(expert_count):
        specs.extend(
            _resolve_explicit_expert_specs(
                experts_prefix=experts_prefix,
                expert_id=expert_id,
                target=target,
                lora_a=_select_expert_slice(lora_a, expert_id, expert_count),
                lora_b=_select_expert_slice(lora_b, expert_id, expert_count),
                scaling=scaling,
                explicit_name=f"{experts_prefix}.{expert_id}.{target}",
            )
        )
    return specs


def _resolve_explicit_expert_specs(
    experts_prefix: str,
    expert_id: int,
    target: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    explicit_name: str,
) -> List[_DeltaSpec]:
    canonical_target = _EXPERT_TARGET_ALIASES.get(target)
    if canonical_target is None:
        raise ValueError(
            f"Unsupported expert LoRA target for merged update: {explicit_name}"
        )

    if canonical_target == "w1":
        return [
            _DeltaSpec(
                param_name=f"{experts_prefix}.w13_weight",
                loaded_weight=_dense_delta(lora_a, lora_b, scaling),
                loader_args=(f"{experts_prefix}.w13_weight", "w1", expert_id),
            )
        ]
    if canonical_target == "w3":
        return [
            _DeltaSpec(
                param_name=f"{experts_prefix}.w13_weight",
                loaded_weight=_dense_delta(lora_a, lora_b, scaling),
                loader_args=(f"{experts_prefix}.w13_weight", "w3", expert_id),
            )
        ]
    if canonical_target == "w2":
        return [
            _DeltaSpec(
                param_name=f"{experts_prefix}.w2_weight",
                loaded_weight=_dense_delta(lora_a, lora_b, scaling),
                loader_args=(f"{experts_prefix}.w2_weight", "w2", expert_id),
            )
        ]

    delta = _dense_delta(lora_a, lora_b, scaling)
    gate_delta, up_delta = _split_packed_expert_delta(delta, explicit_name)
    return [
        _DeltaSpec(
            param_name=f"{experts_prefix}.w13_weight",
            loaded_weight=gate_delta,
            loader_args=(f"{experts_prefix}.w13_weight", "w1", expert_id),
        ),
        _DeltaSpec(
            param_name=f"{experts_prefix}.w13_weight",
            loaded_weight=up_delta,
            loader_args=(f"{experts_prefix}.w13_weight", "w3", expert_id),
        ),
    ]


def _expert_axis_size(tensor: torch.Tensor) -> int:
    if tensor.dim() <= 2:
        return 1
    if tensor.dim() != 3:
        raise ValueError(f"Unsupported MoE LoRA tensor rank: {tensor.dim()}")
    return tensor.shape[0]


def _select_expert_slice(
    tensor: torch.Tensor, expert_id: int, expert_count: int
) -> torch.Tensor:
    if tensor.dim() <= 2:
        return tensor
    if tensor.dim() != 3:
        raise ValueError(f"Unsupported MoE LoRA tensor rank: {tensor.dim()}")
    if tensor.shape[0] == 1:
        return tensor[0]
    if tensor.shape[0] != expert_count:
        raise ValueError(
            f"Mismatched expert dimensions in MoE LoRA tensor: got {tensor.shape[0]}, expected {expert_count}"
        )
    return tensor[expert_id]


def _dense_delta(
    lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float
) -> torch.Tensor:
    if lora_a.dim() != 2 or lora_b.dim() != 2:
        raise ValueError(
            f"Expected 2D LoRA tensors for dense merge, got A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}"
        )
    return torch.matmul(lora_b.float(), lora_a.float()).mul_(scaling)


def _split_packed_expert_delta(
    delta: torch.Tensor, explicit_name: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    if delta.shape[0] % 2 != 0:
        raise ValueError(
            f"Cannot split packed expert delta for {explicit_name}: output dimension {delta.shape[0]} is not even."
        )
    half = delta.shape[0] // 2
    return delta[:half], delta[half:]


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
