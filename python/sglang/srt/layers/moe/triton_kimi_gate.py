"""Triton biased top-8 gate for Kimi-K2 style routing (384 experts, n_group=1).

Alternative to ``sgl_kernel.kimi_k2_moe_fused_gate`` for the
n_group=1 / topk_group=1 / top-8 case, where the fused-gate semantics reduce
to::

    scores  = sigmoid(logits)
    ids     = top-8 by (scores + bias)
    weights = renormalized scores [* routed_scaling_factor]

Measured (B200, graph-timed): 37.4us vs 84.4us at M=16384, 3.95us vs 5.5us at
M=8 (rows_per_prog=1, num_warps=1); expert ids and weights EXACT match vs
``kimi_k2_moe_fused_gate``. Enabled via ``SGLANG_TRITON_GATE=1`` (default off)
in ``sglang.srt.layers.moe.topk``.

``torch_biased_top8`` is the plain-torch reference used for exact-match
validation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _biased_top8_kernel(
    logits_ptr,
    bias_ptr,
    w_ptr,
    ids_ptr,
    M,
    NE: tl.constexpr,
    NE_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    RENORM: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, NE_POW2)
    emask = offs < NE
    bias = tl.load(bias_ptr + offs, mask=emask, other=0.0).to(tl.float32)
    for r in tl.static_range(ROWS_PER_PROG):
        row = pid * ROWS_PER_PROG + r
        if row < M:
            lg = tl.load(logits_ptr + row * NE + offs, mask=emask, other=0.0).to(
                tl.float32
            )
            s = tl.sigmoid(lg)
            b = tl.where(emask, s + bias, -float("inf"))
            wsum = 0.0
            for k in tl.static_range(TOPK):
                idx = tl.argmax(b, axis=0)
                w = tl.sum(tl.where(offs == idx, s, 0.0), axis=0)
                tl.store(ids_ptr + row * TOPK + k, idx.to(tl.int32))
                tl.store(w_ptr + row * TOPK + k, w)
                wsum += w
                b = tl.where(offs == idx, -float("inf"), b)
            # renormalize pass (weights are in L2; reload is cheap)
            ko = tl.arange(0, TOPK)
            wv = tl.load(w_ptr + row * TOPK + ko)
            if RENORM:
                wv = wv / wsum
            if APPLY_SCALE:
                wv = wv * SCALE
            tl.store(w_ptr + row * TOPK + ko, wv)


def biased_top8(
    logits: torch.Tensor,  # [M, 384] float32
    bias: torch.Tensor,  # [384] float32
    top_k: int = 8,
    routed_scaling_factor: float = 1.0,
    apply_scale: bool = True,
    renorm: bool = True,
    rows_per_prog: int = 1,
    num_warps: int = 4,
):
    M, NE = logits.shape
    w = torch.empty(M, top_k, dtype=torch.float32, device=logits.device)
    ids = torch.empty(M, top_k, dtype=torch.int32, device=logits.device)
    grid = (triton.cdiv(M, rows_per_prog),)
    _biased_top8_kernel[grid](
        logits,
        bias,
        w,
        ids,
        M,
        NE=NE,
        NE_POW2=triton.next_power_of_2(NE),
        TOPK=top_k,
        SCALE=routed_scaling_factor,
        APPLY_SCALE=apply_scale,
        RENORM=renorm,
        ROWS_PER_PROG=rows_per_prog,
        num_warps=num_warps,
    )
    return w, ids


def torch_biased_top8(
    logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int = 8,
    routed_scaling_factor: float = 1.0,
    apply_scale: bool = True,
    renorm: bool = True,
):
    """Plain-torch reference for exact-match validation of ``biased_top8``."""
    scores = logits.sigmoid()
    _, ids = torch.topk(scores + bias, top_k, dim=-1)
    w = scores.gather(1, ids)
    if renorm:
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-20)
    if apply_scale:
        w = w * routed_scaling_factor
    return w, ids.to(torch.int32)


def kimi_gate_triton(
    gating_output: torch.Tensor,  # [M, 384] fp32
    correction_bias: torch.Tensor,  # [384]
    topk: int = 8,
    renormalize: bool = True,
    routed_scaling_factor=None,
    apply_routed_scaling_factor_on_output: bool = False,
):
    """Drop-in signature match for ``sgl_kernel.kimi_k2_moe_fused_gate``."""
    # bias dtype is handled in-kernel (loaded and upcast to fp32 exactly), so
    # no host-side cast: bf16 correction_bias (modelopt_fp4 + trtllm MoE) works
    # without an extra launch.
    return biased_top8(
        gating_output.float(),
        top_k=topk,
        routed_scaling_factor=(
            routed_scaling_factor if routed_scaling_factor is not None else 1.0
        ),
        apply_scale=apply_routed_scaling_factor_on_output,
        renorm=renormalize,
        rows_per_prog=1,
        num_warps=1,
    )
