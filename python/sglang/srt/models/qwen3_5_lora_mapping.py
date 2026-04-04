"""Qwen3.5 LoRA target mapping for PEFT adapter ingestion."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.lora.utils import (
    normalize_lora_target_module_name,
    rename_lora_expert_w_to_proj_name,
    rewrite_lora_embedding_aliases_in_weight_name,
)
from sglang.srt.models.qwen3_5_weight_mapping import normalize_qwen3_5_checkpoint_name

LORA_A_SUFFIXES = (".lora_A", ".lora_A.weight")
LORA_B_SUFFIXES = (".lora_B", ".lora_B.weight")

_QWEN3_5_PEFT_PREFIX_REWRITES = (
    ("base_model.model.model.", "model."),
    ("base_model.model.", ""),
)
_QWEN3_5_STACKED_COMPONENT_SPECS = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
    "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
    "in_proj_z": ("in_proj_qkvz", 3),
    "in_proj_b": ("in_proj_ba", 0),
    "in_proj_a": ("in_proj_ba", 1),
}
_QWEN3_5_DIRECT_TARGET_MODULES = frozenset(
    {
        "o_proj",
        "down_proj",
        "out_proj",
        "conv1d",
        "embed_tokens",
        "lm_head",
    }
)
_QWEN3_5_COMPONENT_ORDER = {
    "embed_tokens": 0,
    "q_proj": 1,
    "k_proj": 2,
    "v_proj": 3,
    "o_proj": 4,
    "gate_proj": 5,
    "up_proj": 6,
    "down_proj": 7,
    "in_proj_qkv": 8,
    "in_proj_z": 9,
    "in_proj_b": 10,
    "in_proj_a": 11,
    "out_proj": 12,
    "conv1d": 13,
    "w1": 14,
    "w3": 15,
    "w2": 16,
    "lm_head": 17,
}


def compile_qwen3_5_peft_lora_payload(
    tensor_items: List[tuple[str, torch.Tensor]],
    *,
    loader_metadata: Dict[str, Any],
) -> tuple[List[tuple[str, torch.Tensor]], Dict[str, Any]]:
    payload_tensors: Dict[str, torch.Tensor] = {
        name: tensor for name, tensor in tensor_items
    }
    target_components: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    expert_factor_groups: Dict[tuple[str, str, str], Dict[int, torch.Tensor]] = (
        defaultdict(dict)
    )
    unhandled_names: List[str] = []
    strict = bool(loader_metadata.get("strict", True))

    for tensor_name, _tensor in tensor_items:
        base_name, factor_key = _parse_factor_tensor_name(tensor_name)
        if base_name is None or factor_key is None:
            if strict and ".lora_" in tensor_name:
                unhandled_names.append(tensor_name)
            continue

        runtime_name = _rewrite_qwen3_5_peft_name(base_name)

        dense_component = _build_qwen3_5_dense_component(
            runtime_name,
            tensor_name,
            factor_key,
        )
        if dense_component is not None:
            target_name, component = dense_component
            _merge_component(target_components[target_name], component)
            continue

        expert_component = _build_qwen3_5_routed_expert_component(
            runtime_name,
            tensor_name,
            factor_key,
        )
        if expert_component is not None:
            group_key, expert_id = expert_component
            factor_group = expert_factor_groups[group_key]
            if expert_id in factor_group:
                raise ValueError(
                    f"Duplicate routed-expert factor for {tensor_name!r} (expert_id={expert_id})."
                )
            factor_group[expert_id] = payload_tensors[tensor_name]
            continue

        if strict:
            unhandled_names.append(tensor_name)

    if unhandled_names:
        raise ValueError(
            "Unsupported PEFT LoRA tensor names for the Qwen3.5 compiler: "
            + ", ".join(sorted(unhandled_names))
        )

    _materialize_qwen3_5_routed_expert_groups(
        payload_tensors,
        target_components,
        expert_factor_groups,
    )

    resolved_loader_metadata: Dict[str, Any] = {
        key: value
        for key, value in loader_metadata.items()
        if key not in {"adapter_config", "strict", "targets"}
    }
    resolved_loader_metadata["targets"] = [
        {
            "target_name": target_name,
            "components": sorted(
                components,
                key=lambda component: (
                    _QWEN3_5_COMPONENT_ORDER.get(component["component_id"], 999),
                    component["component_id"],
                ),
            ),
        }
        for target_name, components in sorted(target_components.items())
    ]

    adapter_config = loader_metadata.get("adapter_config") or {}
    rank = (
        loader_metadata.get("rank")
        or loader_metadata.get("r")
        or adapter_config.get("r")
    )
    lora_alpha = loader_metadata.get("lora_alpha") or adapter_config.get("lora_alpha")
    if rank is not None:
        resolved_loader_metadata["rank"] = int(rank)
    if lora_alpha is not None:
        resolved_loader_metadata["lora_alpha"] = float(lora_alpha)

    return list(payload_tensors.items()), resolved_loader_metadata


def _build_qwen3_5_dense_component(
    runtime_name: str,
    tensor_name: str,
    factor_key: str,
) -> Optional[tuple[str, Dict[str, Any]]]:
    if ".mlp.experts." in runtime_name:
        return None

    module_name = runtime_name.rsplit(".", 1)[-1]
    stacked_spec = _QWEN3_5_STACKED_COMPONENT_SPECS.get(module_name)
    if stacked_spec is not None:
        target_module, shard_id = stacked_spec
        target_name = _as_weight_target_name(
            _rewrite_module_suffix(runtime_name, module_name, target_module)
        )
        component: Dict[str, Any] = {
            "component_id": module_name,
            factor_key: tensor_name,
        }
        if shard_id is not None:
            component["shard_id"] = shard_id
        return target_name, component

    normalized_module_name = normalize_lora_target_module_name(module_name)
    if normalized_module_name in _QWEN3_5_DIRECT_TARGET_MODULES:
        target_runtime_name = runtime_name
        if normalized_module_name != module_name:
            target_runtime_name = _rewrite_module_suffix(
                runtime_name,
                module_name,
                normalized_module_name,
            )
        return _as_weight_target_name(target_runtime_name), {
            "component_id": normalized_module_name,
            factor_key: tensor_name,
        }

    return None


def _build_qwen3_5_routed_expert_component(
    runtime_name: str,
    tensor_name: str,
    factor_key: str,
) -> Optional[tuple[tuple[str, str, str], int]]:
    parts = runtime_name.rsplit(".", 2)
    if len(parts) != 3:
        return None
    prefix, expert_id_text, module_name = parts
    if not prefix.endswith(".mlp.experts"):
        return None
    if module_name not in {"gate_proj", "up_proj", "down_proj"}:
        return None
    if not expert_id_text.isdigit():
        return None

    expert_id = int(expert_id_text)
    if module_name == "gate_proj":
        return (f"{prefix}.w13_weight", "w1", factor_key), expert_id
    if module_name == "up_proj":
        return (f"{prefix}.w13_weight", "w3", factor_key), expert_id
    return (f"{prefix}.w2_weight", "w2", factor_key), expert_id


def _materialize_qwen3_5_routed_expert_groups(
    payload_tensors: Dict[str, torch.Tensor],
    target_components: Dict[str, List[Dict[str, Any]]],
    expert_factor_groups: Dict[tuple[str, str, str], Dict[int, torch.Tensor]],
) -> None:
    grouped_components: Dict[tuple[str, str], Dict[str, Dict[int, torch.Tensor]]] = (
        defaultdict(dict)
    )
    for (target_name, component_id, factor_key), expert_factors in (
        expert_factor_groups.items()
    ):
        grouped_components[(target_name, component_id)][factor_key] = expert_factors

    for (target_name, component_id), factors in sorted(grouped_components.items()):
        if "lora_a_name" not in factors or "lora_b_name" not in factors:
            raise ValueError(
                f"Missing routed-expert LoRA factor pair for {target_name!r} ({component_id})."
            )

        expert_ids_by_factor = {
            factor_key: set(expert_factors)
            for factor_key, expert_factors in factors.items()
        }
        if expert_ids_by_factor["lora_a_name"] != expert_ids_by_factor["lora_b_name"]:
            raise ValueError(
                "Routed-expert LoRA factors must cover identical expert ids for "
                f"{target_name!r} ({component_id})."
            )

        num_experts = max(expert_ids_by_factor["lora_a_name"]) + 1
        factor_names: Dict[str, str] = {}
        for factor_key, expert_factors in factors.items():
            packed_name = (
                f"__sglang_live_lora_packed_experts.{target_name}.{component_id}."
                f"{_factor_key_to_label(factor_key)}.weight"
            )
            payload_tensors[packed_name] = _stack_expert_factors(
                expert_factors,
                num_experts=num_experts,
            )
            factor_names[factor_key] = packed_name

        target_components[target_name].append(
            {
                "component_id": component_id,
                "lora_a_name": factor_names["lora_a_name"],
                "lora_b_name": factor_names["lora_b_name"],
                "shard_id": component_id,
                "fused_experts": True,
            }
        )


def _parse_factor_tensor_name(tensor_name: str) -> tuple[Optional[str], Optional[str]]:
    for suffix in LORA_A_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)], "lora_a_name"
    for suffix in LORA_B_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)], "lora_b_name"
    return None, None


def _rewrite_qwen3_5_peft_name(name: str) -> str:
    for prefix, replacement in _QWEN3_5_PEFT_PREFIX_REWRITES:
        if name.startswith(prefix):
            name = replacement + name[len(prefix) :]
            break

    name = rewrite_lora_embedding_aliases_in_weight_name(name)
    name = rename_lora_expert_w_to_proj_name(name)
    return normalize_qwen3_5_checkpoint_name(name)


def _rewrite_module_suffix(
    runtime_name: str,
    source_module: str,
    target_module: str,
) -> str:
    if runtime_name == source_module:
        return target_module
    suffix = f".{source_module}"
    if not runtime_name.endswith(suffix):
        raise ValueError(
            f"{runtime_name!r} does not end with the module suffix {source_module!r}."
        )
    return f"{runtime_name[: -len(source_module)]}{target_module}"


def _as_weight_target_name(runtime_name: str) -> str:
    return runtime_name if runtime_name.endswith(".weight") else f"{runtime_name}.weight"


def _merge_component(components: List[Dict[str, Any]], incoming: Dict[str, Any]) -> None:
    for component in components:
        if component.get("component_id") != incoming.get("component_id"):
            continue
        if component.get("shard_id") != incoming.get("shard_id"):
            continue
        overlap = {"lora_a_name", "lora_b_name"} & component.keys() & incoming.keys()
        if overlap:
            raise ValueError(
                "Duplicate LoRA factor assignment for component "
                f"{incoming.get('component_id')!r}."
            )
        component.update(incoming)
        return
    components.append(dict(incoming))


def _factor_key_to_label(factor_key: str) -> str:
    return "lora_a" if factor_key == "lora_a_name" else "lora_b"


def _stack_expert_factors(
    expert_factors: Dict[int, torch.Tensor],
    *,
    num_experts: int,
) -> torch.Tensor:
    if not expert_factors:
        raise ValueError("Cannot stack an empty routed-expert factor set.")

    example = next(iter(expert_factors.values()))
    stacked = example.new_zeros((num_experts, *example.shape))
    for expert_id, tensor in expert_factors.items():
        if tensor.shape != example.shape:
            raise ValueError(
                "Routed-expert LoRA factors must share the same shape, got "
                f"{tuple(example.shape)} and {tuple(tensor.shape)}."
            )
        stacked[expert_id].copy_(tensor)
    return stacked
