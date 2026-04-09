import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

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
        _apply_loaded_delta(param, spec.loaded_weight, spec.loader_args)
        layer = _get_flashinfer_moe_layer_for_postprocess(param)
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


def _accumulate_fp32(
    target: torch.Tensor, delta: torch.Tensor, *, non_blocking: bool = False
) -> torch.Tensor:
    updated = target.to(device=target.device, dtype=torch.float32)
    updated.add_(
        delta.to(device=target.device, dtype=torch.float32, non_blocking=non_blocking),
        alpha=1,
    )
    return updated.to(dtype=target.dtype, non_blocking=non_blocking)


@contextmanager
def _additive_loader_context():
    original_copy = torch.Tensor.copy_
    original_fill = torch.Tensor.fill_

    def additive_copy_(self, src, non_blocking=False):
        original_copy(
            self,
            _accumulate_fp32(self, src, non_blocking=non_blocking),
            non_blocking=non_blocking,
        )
        return self

    def additive_fill_(self, value):
        if value not in (0, 0.0):
            raise ValueError(
                f"Unsupported fill_({value}) during additive merged LoRA update."
            )
        return self

    torch.Tensor.copy_ = additive_copy_
    torch.Tensor.fill_ = additive_fill_
    try:
        yield
    finally:
        torch.Tensor.copy_ = original_copy
        torch.Tensor.fill_ = original_fill


def _apply_loaded_delta(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    loader_args: Tuple[Any, ...],
) -> None:
    weight_loader = getattr(param, "weight_loader", None)
    if weight_loader is None:
        if loader_args:
            raise ValueError(
                f"Parameter does not expose weight_loader but loader args were provided: {loader_args}"
            )
        if param.data.shape != loaded_weight.shape:
            raise ValueError(
                f"Shape mismatch for direct delta add: param={tuple(param.data.shape)} "
                f"loaded={tuple(loaded_weight.shape)}"
            )
        param.data.copy_(
            _accumulate_fp32(param.data, loaded_weight),
            non_blocking=False,
        )
        return

    with _additive_loader_context():
        weight_loader(param, loaded_weight.to(device=param.data.device), *loader_args)


def _get_flashinfer_moe_layer_for_postprocess(
    param: torch.nn.Parameter,
) -> Optional[torch.nn.Module]:
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
