"""FlashInfer CuTe DSL NVFP4 MoE (standard path), W4A16 or W4A4."""

from __future__ import annotations

from sglang.srt.weight_sync.weight_delta.map import Copy, Place

GROUP = 64  # interleave_w13_halves: [Up, Gate] in 64-row groups


def handles(layer) -> bool:
    method = layer.quant_method
    return type(method).__name__ == "ModelOptNvFp4FusedMoEMethod" and getattr(
        method, "_is_cutedsl_v2_standard", False
    )


def _half(projection: str) -> int:
    # CuTe DSL loads up first, so up is the first half of every 128-row pair.
    return 0 if projection == "up" else 1


def _swizzled(scale):
    """swizzle_blockscale's layout seen in canonical (row, column) order.

    Physical order is (row // 128, col // 4, row % 32, row % 128 // 32, col % 4); the
    returned view has axes (row // 128, row % 128 // 32, row % 32, col // 4, col % 4).
    """
    rows, cols = scale.shape
    if rows % 128 or cols % 4:
        raise ValueError("padded NVFP4 block scales are not supported yet")
    return scale.view(rows // 128, cols // 4, 32, 4, 4).permute(0, 3, 2, 1, 4)


def expert_place(layer, slot: int, projection: str, kind: str, tensor) -> Place:
    whole = (slice(None),) * len(tensor.shape)
    if kind == "weight_scale_2":
        if projection == "down":
            return Place(whole, layer.w2_weight_scale_2[slot])
        # Per-tensor scales stay gate-first: [:, 0] gate, [:, 1] up.
        return Place(
            whole, layer.w13_weight_scale_2[slot, 0 if projection == "gate" else 1]
        )
    if kind == "weight":
        if projection == "down":
            return Place(whole, layer.w2_weight[slot])
        w13 = layer.w13_weight[slot]
        groups = w13.view(w13.shape[0] // (2 * GROUP), 2, GROUP, w13.shape[1])
        return Place(whole, groups[:, _half(projection)])
    if kind == "weight_scale":
        if projection == "down":
            return Place(whole, _swizzled(layer.w2_weight_scale[slot]))
        # After the 64-row interleave, a 128-row tile holds 64 up rows then 64 gate
        # rows, i.e. 32-row blocks 0-1 are up and 2-3 are gate.
        half = _half(projection)
        return Place(
            whole, _swizzled(layer.w13_weight_scale[slot])[:, 2 * half : 2 * half + 2]
        )
    raise ValueError(f"unknown NVFP4 expert tensor kind {kind}")


def copies(layer) -> list[Copy]:
    """CuTe DSL's GEMM alphas: weight_scale_2 times activation scales that are 1.0."""
    if layer._cutedsl_wrapper is None:
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
            ensure_cutedsl_wrapper,
        )

        ensure_cutedsl_wrapper(layer)
    gate, up = layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]
    down = layer.w2_weight_scale_2
    w1_alpha, _, w2_alpha = layer._cutedsl_scales
    return [
        Copy(gate, layer.g1_alphas),
        Copy(up, layer.g1_alphas_up),
        Copy(down, layer.g2_alphas),
        Copy(gate, w1_alpha),
        Copy(down, w2_alpha),
    ]
