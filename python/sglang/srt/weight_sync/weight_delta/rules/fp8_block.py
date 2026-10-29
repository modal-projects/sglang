"""Block-FP8 MoE (Fp8MoEMethod) whose post-load keeps the loaded layout, as on Hopper with the
Triton or DeepGEMM runner: per expert w13 = [gate; up] and w2, each with FP32 block scales
laid out the same way."""

from __future__ import annotations

import torch

from sglang.srt.weight_sync.weight_delta.map import Copy, Place


def handles(layer) -> bool:
    method = layer.quant_method
    if type(method).__name__ != "Fp8MoEMethod" or not method.block_quant:
        return False
    # Any post-load rewrite of the loaded bytes (UE8M0 requantization, aiter shuffles) is
    # a layout this rule does not describe.
    rewritten = (
        layer.w13_weight_scale_inv.dtype != torch.float32
        or getattr(layer.w13_weight_scale_inv, "format_ue8m0", False)
        or getattr(layer.w13_weight, "is_shuffled", False)
    )
    if rewritten:
        raise ValueError("block-FP8 experts rewritten after load are not mapped")
    return True


def expert_place(layer, slot: int, projection: str, kind: str, tensor) -> Place:
    if kind == "weight":
        w13, w2 = layer.w13_weight[slot], layer.w2_weight[slot]
    elif kind == "weight_scale_inv":
        w13, w2 = layer.w13_weight_scale_inv[slot], layer.w2_weight_scale_inv[slot]
    else:
        raise ValueError(f"block-FP8 experts have no {kind}")
    whole = (slice(None),) * len(tensor.shape)
    if projection == "down":
        return Place(whole, w2)
    rows = w13.shape[0] // 2
    return Place(whole, w13[:rows] if projection == "gate" else w13[rows:])


def copies(layer) -> list[Copy]:
    return []
