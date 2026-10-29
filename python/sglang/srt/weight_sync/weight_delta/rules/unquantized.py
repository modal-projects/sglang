"""Unquantized MoE on the Triton runner, whose post-load leaves the loaded layout (e.g. the BF16
draft). Other runners rearrange the experts after loading (TRT-LLM swaps gate and up, permutes
rows and converts to its block layout; the fused-SwiGLU option interleaves gate and up rows), so
no rule claims them and the map refuses them at startup."""

from __future__ import annotations

from sglang.srt.weight_sync.weight_delta.map import Copy, Place


def handles(layer) -> bool:
    method = layer.quant_method
    if type(method).__name__ != "UnquantizedFusedMoEMethod":
        return False
    moe_runner = getattr(method, "runner", None)
    return (
        moe_runner is not None
        and moe_runner.runner_backend.is_triton()
        and not getattr(method, "w13_swiglu_interleaved", False)
    )


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
