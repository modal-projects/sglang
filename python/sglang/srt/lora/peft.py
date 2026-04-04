"""Compile PEFT LoRA adapters into explicit live-merge target metadata."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

import torch

from sglang.srt.models.qwen3_5_lora_mapping import compile_qwen3_5_peft_lora_payload

_Compiler = Callable[..., tuple[List[tuple[str, torch.Tensor]], Dict[str, Any]]]

_PEFT_TARGET_COMPILERS: dict[str, _Compiler] = {
    "qwen3_5": compile_qwen3_5_peft_lora_payload,
}


def compile_peft_lora_payload(
    named_tensors: List[tuple[str, torch.Tensor]] | Mapping[str, torch.Tensor],
    *,
    target_resolver: str,
    loader_metadata: Optional[Dict[str, Any]] = None,
) -> tuple[List[tuple[str, torch.Tensor]], Dict[str, Any]]:
    """Compile PEFT adapter tensors into explicit live-merge target metadata."""
    loader_metadata = dict(loader_metadata or {})
    tensor_items = (
        list(named_tensors.items())
        if isinstance(named_tensors, Mapping)
        else list(named_tensors)
    )
    if loader_metadata.get("targets") is not None:
        return tensor_items, loader_metadata

    compiler = _PEFT_TARGET_COMPILERS.get(target_resolver)
    if compiler is None:
        raise NotImplementedError(
            f"Unsupported PEFT target resolver {target_resolver!r}."
        )

    return compiler(
        tensor_items,
        loader_metadata=loader_metadata,
    )
