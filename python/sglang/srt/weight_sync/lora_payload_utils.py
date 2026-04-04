from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors.torch import save as save_safetensors

_FACTOR_SUFFIXES = (
    (".lora_A.default.weight", "lora_A"),
    (".lora_B.default.weight", "lora_B"),
    (".lora_A.weight", "lora_A"),
    (".lora_B.weight", "lora_B"),
    (".lora_A", "lora_A"),
    (".lora_B", "lora_B"),
)

_PREFIX_REWRITES = (
    ("base_model.model.", ""),
    ("model.language_model.", "model."),
    ("language_model.model.", "model."),
    ("model.model.", "model."),
)

_ATTN_COMPONENTS = {
    "q_proj": ("qkv_proj.weight", "q", "q"),
    "k_proj": ("qkv_proj.weight", "k", "k"),
    "v_proj": ("qkv_proj.weight", "v", "v"),
    "o_proj": ("o_proj.weight", None, None),
    "qkv_proj": ("qkv_proj.weight", None, None),
}

_MLP_COMPONENTS = {
    "gate_proj": ("gate_up_proj.weight", "gate", 0),
    "up_proj": ("gate_up_proj.weight", "up", 1),
    "down_proj": ("down_proj.weight", None, None),
    "gate_up_proj": ("gate_up_proj.weight", None, None),
}

_EXPERT_COMPONENTS = {
    "gate_proj": ("w13_weight", "w1", "w1"),
    "up_proj": ("w13_weight", "w3", "w3"),
    "down_proj": ("w2_weight", "w2", "w2"),
    "w1": ("w13_weight", "w1", "w1"),
    "w2": ("w2_weight", "w2", "w2"),
    "w3": ("w13_weight", "w3", "w3"),
}

_EXPERT_DIRECT_COMPONENTS = {
    "w1": ("w13_weight", "w1", "w1"),
    "w2": ("w2_weight", "w2", "w2"),
    "w3": ("w13_weight", "w3", "w3"),
    "down_proj": ("w2_weight", "w2", "w2"),
}


@dataclass
class PreparedLoRAWeightSyncPayload:
    named_tensors: List[Tuple[str, torch.Tensor]]
    loader_metadata: Dict[str, Any]
    skipped_tensor_names: List[str]


def convert_peft_lora_tensors_to_weight_sync_payload(
    adapter_tensors: Dict[str, torch.Tensor],
    *,
    adapter_config: Optional[Dict[str, Any]] = None,
    skip_visual_tensors: bool = True,
) -> PreparedLoRAWeightSyncPayload:
    """Convert PEFT-style LoRA tensors into the live weight-sync payload format."""

    named_tensors: List[Tuple[str, torch.Tensor]] = []
    target_specs: Dict[str, Dict[str, Any]] = {}
    skipped_tensor_names: List[str] = []
    expert_factor_groups: Dict[Tuple[str, str, str], Dict[Optional[int], torch.Tensor]] = {}

    for raw_name, tensor in sorted(adapter_tensors.items()):
        split_factor = _split_lora_factor_name(raw_name)
        if split_factor is None:
            skipped_tensor_names.append(raw_name)
            continue

        module_name, factor_kind = split_factor
        module_name = _normalize_adapter_module_name(module_name)
        if not module_name:
            skipped_tensor_names.append(raw_name)
            continue
        if skip_visual_tensors and (
            module_name.startswith("visual.")
            or module_name.startswith("model.visual.")
            or ".visual." in module_name
        ):
            skipped_tensor_names.append(raw_name)
            continue

        if _register_dense_target(
            target_specs,
            named_tensors,
            module_name,
            factor_kind,
            tensor,
        ):
            continue

        if _register_dense_expert_target(
            target_specs,
            named_tensors,
            expert_factor_groups,
            module_name,
            factor_kind,
            tensor,
        ):
            continue

        skipped_tensor_names.append(raw_name)

    _finalize_expert_factor_groups(target_specs, named_tensors, expert_factor_groups)

    target_list = []
    for target_name in sorted(target_specs):
        spec = target_specs[target_name]
        components = spec.get("components", [])
        if not components:
            continue
        target_list.append({"target_name": target_name, "components": components})

    loader_metadata: Dict[str, Any] = {"targets": target_list}
    if adapter_config is not None:
        if "lora_alpha" in adapter_config:
            loader_metadata["lora_alpha"] = adapter_config["lora_alpha"]
        rank = adapter_config.get("r", adapter_config.get("rank"))
        if rank is not None:
            loader_metadata["rank"] = rank

    return PreparedLoRAWeightSyncPayload(
        named_tensors=named_tensors,
        loader_metadata=loader_metadata,
        skipped_tensor_names=skipped_tensor_names,
    )


def serialize_weight_sync_payload(named_tensors: List[Tuple[str, torch.Tensor]]) -> bytes:
    return save_safetensors(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in named_tensors
        }
    )


def negate_lora_payload(
    named_tensors: List[Tuple[str, torch.Tensor]],
) -> List[Tuple[str, torch.Tensor]]:
    negated: List[Tuple[str, torch.Tensor]] = []
    for name, tensor in named_tensors:
        if ".lora_B" in name:
            negated.append((name, -tensor))
        else:
            negated.append((name, tensor))
    return negated


def _split_lora_factor_name(name: str) -> Optional[Tuple[str, str]]:
    for suffix, factor_kind in _FACTOR_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], factor_kind
    return None


def _normalize_adapter_module_name(name: str) -> str:
    normalized = name
    for prefix, replacement in _PREFIX_REWRITES:
        if normalized.startswith(prefix):
            normalized = replacement + normalized[len(prefix) :]
    return normalized.rstrip(".")


def _ensure_target_spec(
    target_specs: Dict[str, Dict[str, Any]], target_name: str
) -> Dict[str, Any]:
    return target_specs.setdefault(target_name, {"target_name": target_name, "components": []})


def _make_tensor_name(
    target_name: str,
    factor_kind: str,
    component_label: Optional[str] = None,
) -> str:
    if component_label is None:
        return f"{target_name}.{factor_kind}"
    return f"{target_name}.{component_label}.{factor_kind}"


def _register_component(
    target_specs: Dict[str, Dict[str, Any]],
    named_tensors: List[Tuple[str, torch.Tensor]],
    *,
    target_name: str,
    tensor: torch.Tensor,
    factor_kind: str,
    component_label: Optional[str] = None,
    shard_id: Optional[Any] = None,
    fused_experts: bool = False,
) -> None:
    tensor_name = _make_tensor_name(target_name, factor_kind, component_label)
    named_tensors.append((tensor_name, tensor))
    spec = _ensure_target_spec(target_specs, target_name)
    components = spec["components"]
    if not components or components[-1].get("_component_label") != component_label:
        component: Dict[str, Any] = {"_component_label": component_label}
        if shard_id is not None:
            component["shard_id"] = shard_id
        if fused_experts:
            component["fused_experts"] = True
        components.append(component)
    component = components[-1]
    if factor_kind == "lora_A":
        component["lora_a_name"] = tensor_name
    else:
        component["lora_b_name"] = tensor_name


def _register_dense_target(
    target_specs: Dict[str, Dict[str, Any]],
    named_tensors: List[Tuple[str, torch.Tensor]],
    module_name: str,
    factor_kind: str,
    tensor: torch.Tensor,
) -> bool:
    if module_name == "model.embed_tokens":
        _register_component(
            target_specs,
            named_tensors,
            target_name="model.embed_tokens.weight",
            tensor=tensor,
            factor_kind=factor_kind,
        )
        return True
    if module_name in ("lm_head", "model.unembed_tokens", "unembed_tokens"):
        _register_component(
            target_specs,
            named_tensors,
            target_name="lm_head.weight",
            tensor=tensor,
            factor_kind=factor_kind,
        )
        return True

    attn_prefix, attn_module = _split_suffix(
        module_name,
        ("q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"),
    )
    if attn_prefix is not None:
        target_suffix, component_label, shard_id = _ATTN_COMPONENTS[attn_module]
        _register_component(
            target_specs,
            named_tensors,
            target_name=f"{attn_prefix}.{target_suffix}",
            tensor=tensor,
            factor_kind=factor_kind,
            component_label=component_label,
            shard_id=shard_id,
        )
        return True

    mlp_prefix, mlp_module = _split_suffix(
        module_name,
        ("gate_proj", "up_proj", "down_proj", "gate_up_proj"),
    )
    if mlp_prefix is not None and ".experts" not in mlp_prefix:
        target_suffix, component_label, shard_id = _MLP_COMPONENTS[mlp_module]
        _register_component(
            target_specs,
            named_tensors,
            target_name=f"{mlp_prefix}.{target_suffix}",
            tensor=tensor,
            factor_kind=factor_kind,
            component_label=component_label,
            shard_id=shard_id,
        )
        return True

    return False


def _register_dense_expert_target(
    target_specs: Dict[str, Dict[str, Any]],
    named_tensors: List[Tuple[str, torch.Tensor]],
    expert_factor_groups: Dict[Tuple[str, str, str], Dict[Optional[int], torch.Tensor]],
    module_name: str,
    factor_kind: str,
    tensor: torch.Tensor,
) -> bool:
    expert_result = _split_expert_module_name(module_name)
    if expert_result is not None:
        layer_prefix, expert_id, expert_module = expert_result
        target_suffix, component_label, shard_id = _EXPERT_COMPONENTS[expert_module]
        target_name = f"{layer_prefix}.experts.{target_suffix}"
        group_key = (target_name, component_label, factor_kind)
        expert_factor_groups.setdefault(group_key, {})[expert_id] = tensor
        _ensure_target_spec(target_specs, target_name)
        return True

    expert_prefix, expert_module = _split_suffix(
        module_name,
        ("gate_up_proj", "down_proj", "w1", "w2", "w3"),
    )
    if expert_prefix is None or not expert_prefix.endswith(".mlp.experts"):
        return False

    if expert_module == "gate_up_proj":
        split_w1, split_w3 = _split_stacked_factor_tensor(tensor)
        for shard_id, component_label, split_tensor in (
            ("w1", "w1", split_w1),
            ("w3", "w3", split_w3),
        ):
            _register_component(
                target_specs,
                named_tensors,
                target_name=f"{expert_prefix}.w13_weight",
                tensor=split_tensor,
                factor_kind=factor_kind,
                component_label=component_label,
                shard_id=shard_id,
                fused_experts=True,
            )
        return True

    target_suffix, component_label, shard_id = _EXPERT_DIRECT_COMPONENTS[expert_module]
    _register_component(
        target_specs,
        named_tensors,
        target_name=f"{expert_prefix}.{target_suffix}",
        tensor=tensor,
        factor_kind=factor_kind,
        component_label=component_label,
        shard_id=shard_id,
        fused_experts=True,
    )
    return True


def _finalize_expert_factor_groups(
    target_specs: Dict[str, Dict[str, Any]],
    named_tensors: List[Tuple[str, torch.Tensor]],
    expert_factor_groups: Dict[Tuple[str, str, str], Dict[Optional[int], torch.Tensor]],
) -> None:
    for (target_name, component_label, factor_kind), expert_map in sorted(
        expert_factor_groups.items()
    ):
        shard_id = component_label
        stacked_tensor = _stack_expert_factors(expert_map)
        _register_component(
            target_specs,
            named_tensors,
            target_name=target_name,
            tensor=stacked_tensor,
            factor_kind=factor_kind,
            component_label=component_label,
            shard_id=shard_id,
            fused_experts=True,
        )

    for spec in target_specs.values():
        for component in spec.get("components", []):
            component.pop("_component_label", None)


def _split_suffix(
    name: str,
    suffixes: Tuple[str, ...],
) -> Tuple[Optional[str], Optional[str]]:
    for suffix in suffixes:
        marker = f".{suffix}"
        if name.endswith(marker):
            return name[: -len(marker)], suffix
    return None, None


def _split_expert_module_name(name: str) -> Optional[Tuple[str, int, str]]:
    for module_name in ("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"):
        marker = f".{module_name}"
        if not name.endswith(marker):
            continue
        prefix = name[: -len(marker)]
        expert_marker = ".mlp.experts."
        if expert_marker not in prefix:
            continue
        layer_prefix, expert_id_str = prefix.rsplit(expert_marker, 1)
        if not expert_id_str.isdigit():
            continue
        return f"{layer_prefix}.mlp", int(expert_id_str), module_name
    return None


def _split_stacked_factor_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    split_dim = tensor.ndim - 2
    if tensor.shape[split_dim] % 2 != 0:
        raise ValueError(
            "Stacked gate_up LoRA tensor must have an even size along the split dimension, "
            f"got shape {tuple(tensor.shape)}."
        )
    return torch.chunk(tensor, 2, dim=split_dim)


def _stack_expert_factors(expert_map: Dict[Optional[int], torch.Tensor]) -> torch.Tensor:
    if None in expert_map:
        if len(expert_map) != 1:
            raise ValueError("Cannot mix explicit expert ids with shared expert factors.")
        tensor = expert_map[None]
        return tensor if tensor.ndim == 3 else tensor.unsqueeze(0)

    expert_ids = sorted(expert_map)
    if len(expert_ids) == 1:
        if expert_ids[0] != 0:
            raise ValueError(
                f"Single expert factor must use expert id 0, got {expert_ids[0]}."
            )
        tensor = expert_map[expert_ids[0]]
        return tensor if tensor.ndim == 3 else tensor.unsqueeze(0)

    expected_ids = list(range(expert_ids[0], expert_ids[0] + len(expert_ids)))
    if expert_ids != expected_ids or expert_ids[0] != 0:
        raise ValueError(
            f"Expert ids must be contiguous and 0-based for live weight sync, got {expert_ids}."
        )

    tensors = []
    for expert_id in expert_ids:
        tensor = expert_map[expert_id]
        if tensor.ndim == 3:
            if tensor.shape[0] != 1:
                raise ValueError(
                    "Per-expert factors must be rank-2 or have a singleton expert dimension, "
                    f"got shape {tuple(tensor.shape)}."
                )
            tensor = tensor[0]
        tensors.append(tensor)
    return torch.stack(tensors, dim=0)
