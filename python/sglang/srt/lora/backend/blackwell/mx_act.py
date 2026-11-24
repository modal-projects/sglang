"""Clamped SwiGLU between the two MXFP8 projections of DeepSeek-V4.1's TP shared expert.

The merged gate/up projection (blackwell_backend._fused_split) writes each slice padded to a
128-row multiple: yp [rows, 2 * SP] = [gate (IS) | 0 | up (IS) | 0]. One launch turns it into
both inputs of the down projection, whose K (IS) is padded to KP: the BF16 activation
act [rows, KP] (zero past IS; its LoRA-A shrink) and the activation's MXFP8 quant with
FlashInfer's 128x4 scale layout (the base GEMM). It replaces the slice gather, the clamped
SwiGLU, the K padding and the quantize launches of the unfused path; the activation is
SGLang's silu_mul_clamp (branches clamped in BF16, product rounded to BF16) and the quant is
mxfp8_quantize_swizzled's, applied to that BF16 value.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.mxfp8_swizzled_triton import (
    _mx_e8m0_from_amax,
    _mx_scale_offsets,
)


@triton.jit(do_not_specialize=["n_rows"])  # one compile for every batch size
def _swiglu_pad_mx_kernel(yp, act, q, sf, n_rows, limit, YS: tl.constexpr, IS: tl.constexpr,
                          SP: tl.constexpr, KP: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    live = r < n_rows
    NCB: tl.constexpr = (KP // 32 + 3) // 4
    for g in tl.static_range(KP // 32):
        c = g * 32 + tl.arange(0, 32)
        m = live[:, None] & (c < IS)[None, :]
        gate = tl.load(yp + r[:, None].to(tl.int64) * YS + c[None, :], mask=m,
                       other=0.0).to(tl.float32)
        up = tl.load(yp + r[:, None].to(tl.int64) * YS + SP + c[None, :], mask=m,
                     other=0.0).to(tl.float32)
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
        a = (gate / (1.0 + tl.exp(-gate)) * up).to(tl.bfloat16)
        tl.store(act + r[:, None].to(tl.int64) * KP + c[None, :], a, mask=live[:, None])
        af = a.to(tl.float32)
        e8, inv = _mx_e8m0_from_amax(tl.max(tl.abs(af), axis=1))
        tl.store(q + r[:, None].to(tl.int64) * KP + c[None, :],
                 (af * inv[:, None]).to(tl.float8e4nv), mask=live[:, None])
        # every row of the 128-row scale groups (rows past n_rows: zero activation)
        tl.store(sf + _mx_scale_offsets(r, g, NCB), e8.to(tl.uint8))


def swiglu_pad_mxfp8(yp, n_rows, inter, slice_pad, k_pad, limit):
    """yp [rows, 2 * slice_pad] BF16 -> (act [rows, k_pad] BF16, q [rows, k_pad] E4M3,
    swizzled UE8M0 scales [ceil(rows / 128) * 128 * k_pad / 32])."""
    assert yp.dtype == torch.bfloat16 and yp.stride(1) == 1 and k_pad % 128 == 0
    assert inter <= slice_pad and inter <= k_pad
    rows_pad = triton.cdiv(n_rows, 128) * 128
    act = torch.empty(n_rows, k_pad, dtype=torch.bfloat16, device=yp.device)
    q = torch.empty(n_rows, k_pad, dtype=torch.float8_e4m3fn, device=yp.device)
    sf = torch.empty(rows_pad * (k_pad // 32), dtype=torch.uint8, device=yp.device)
    BR = 16
    _swiglu_pad_mx_kernel[(rows_pad // BR,)](
        yp, act, q, sf, n_rows, float(limit), YS=yp.stride(0), IS=inter, SP=slice_pad,
        KP=k_pad, BR=BR, num_warps=4)
    return act, q, sf
