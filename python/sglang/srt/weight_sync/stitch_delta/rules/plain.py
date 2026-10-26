"""Unquantized MoE whose post-load leaves the loaded layout (e.g. the BF16 Triton draft)."""

from __future__ import annotations

from sglang.srt.weight_sync.stitch_delta.map import Copy, Place


def handles(layer) -> bool:
    return type(layer.quant_method).__name__ == "UnquantizedFusedMoEMethod"


def expert_place(layer, slot: int, projection: str, kind: str, tensor) -> Place:
    if kind != "weight":
        raise ValueError(f"unquantized experts have no {kind}")
    whole = (slice(None),) * len(tensor.shape)
    if projection == "down":
        return Place(whole, layer.w2_weight[slot])
    w13 = layer.w13_weight[slot]
    rows = w13.shape[0] // 2
    return Place(whole, w13[:rows] if projection == "gate" else w13[rows:])


def copies(layer) -> list[Copy]:
    return []
