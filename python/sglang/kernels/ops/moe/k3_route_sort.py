# SPDX-License-Identifier: Apache-2.0
"""Decode routing kernels for the ROCm aiter MoE runner.

_k3_topk_kernel: sigmoid + correction-bias top-k, one CTA per token (replaces
aiter's topk_reg kernel). _k3_sort_quant_kernel: MoE sorting + stage-1 mxfp8
activation quant + moe_buf zeroing in one launch for M <= block_size, where
every used expert owns exactly one block, so there is no cross-CTA ordering
(replaces moe_sorting_small's distributed kernel at decode sizes).

Output layouts are identical to moe_sorting_small (and therefore aiter):
  sorted_ids[i]        = (topk_slot << 24) | token   (padding: (topk << 24) | M)
  sorted_weights       = pair weight                 (padding: 0)
  sorted_expert_ids[b] = expert of block b
  num_valid_ids        = [num_blocks * block_size, M]
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _score_key(s, idx):
    # order-preserving int64 key: higher score first, lower expert id on ties
    b = s.to(tl.int32, bitcast=True)
    k = tl.where(b >= 0, b, b ^ 0x7FFFFFFF).to(tl.int64)
    return (k << 32) | (0xFFFF - idx).to(tl.int64)


@triton.jit(do_not_specialize=["stride_lg"])
def _k3_topk_kernel(
    logits_ptr,  # [M, E] fp32/bf16, row stride stride_lg
    stride_lg,
    bias_ptr,  # [E] correction bias
    topk_ids_ptr,  # [M, TOPK] i32 out
    topk_weights_ptr,  # [M, TOPK] fp32 out
    routed_scale,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    RENORM: tl.constexpr,
):
    # one CTA per token: top-k of sigmoid(logit) + bias, weights = renormalized sigmoid
    t = tl.program_id(0)
    offs_e = tl.arange(0, E_POW2)
    me = offs_e < E
    lg = tl.load(logits_ptr + t * stride_lg + offs_e, mask=me, other=0.0).to(tl.float32)
    # the bias is rounded to the logits dtype first, as the aiter router does
    bias = tl.load(bias_ptr + offs_e, mask=me, other=0.0).to(logits_ptr.dtype.element_ty).to(tl.float32)
    biased = tl.where(me, tl.sigmoid(lg) + bias, float("-inf"))
    top = tl.topk(_score_key(biased, offs_e), TOPK, dim=0)  # descending
    ids = (0xFFFF - (top & 0xFFFF)).to(tl.int32)
    w = tl.sigmoid(tl.load(logits_ptr + t * stride_lg + ids).to(tl.float32))
    if RENORM:
        w = w / tl.sum(w, axis=0)
    w = w * routed_scale
    offs_k = tl.arange(0, TOPK)
    tl.store(topk_ids_ptr + t * TOPK + offs_k, ids)
    tl.store(topk_weights_ptr + t * TOPK + offs_k, w)


@triton.jit(do_not_specialize=["M", "stride_qx"])
def _k3_sort_quant_kernel(
    topk_ids_ptr,  # [M, TOPK] i32
    topk_weights_ptr,  # [M, TOPK] fp32
    sorted_ids_ptr,
    sorted_weights_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    moe_buf_ptr,  # [M, N_COLS] zeroed here
    qx_ptr,  # [M, N_COLS] routed activations, row stride stride_qx
    stride_qx,
    qout_ptr,  # [M, N_COLS] fp8 out
    qscale_ptr,  # swizzled e8m0, one per (sorted_row, group)
    M,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    P_POW2: tl.constexpr,
    N_COLS: tl.constexpr,
    QCHUNK: tl.constexpr,
    SCALEN_PAD: tl.constexpr,
    MOE_BUF_ZERO: tl.constexpr,
):
    # M <= BLOCK_SIZE: a token picks an expert at most once, so every used
    # expert owns exactly one block, indexed by the number of used experts
    # below it. CTA (t, c) quantizes token t's column chunk c and scatters its
    # e8m0 scales to the token's TOPK sorted rows; the c == 0 CTA also writes
    # the token's sorted entries and, for each expert the token is first to
    # pick, that block's expert id and tail padding. Every byte has one writer.
    pid = tl.program_id(0)
    CHUNKS: tl.constexpr = N_COLS // QCHUNK
    t = pid // CHUNKS
    c0 = (pid % CHUNKS) * QCHUNK
    P = M * TOPK
    offs_p = tl.arange(0, P_POW2)
    mask_p = offs_p < P
    SENT: tl.constexpr = 0x7FFFFFFF
    e = tl.load(topk_ids_ptr + offs_p, mask=mask_p, other=SENT)
    # distinct used experts, ascending (tl.histogram miscompiles on gfx950)
    srt = tl.sort(e)
    prv = tl.gather(srt, tl.maximum(offs_p - 1, 0), axis=0)
    first = ((offs_p == 0) | (srt != prv)) & (srt != SENT)

    offs_k = tl.arange(0, TOPK)
    ek = tl.load(topk_ids_ptr + t * TOPK + offs_k)
    match = e[None, :] == ek[:, None]
    rank_k = tl.sum(tl.where(match & (offs_p[None, :] < t * TOPK), 1, 0), axis=1)
    bb_k = tl.sum(tl.where(first[None, :] & (srt[None, :] < ek[:, None]), 1, 0), axis=1)
    dest_k = bb_k * BLOCK_SIZE + rank_k

    if c0 == 0:
        wk = tl.load(topk_weights_ptr + t * TOPK + offs_k)
        tl.store(sorted_ids_ptr + dest_k, (offs_k << 24) | t)
        tl.store(sorted_weights_ptr + dest_k, wk)
        lead = rank_k == 0
        cnt_k = tl.sum(tl.where(match, 1, 0), axis=1)
        tl.store(sorted_expert_ids_ptr + bb_k, ek, mask=lead)
        offs_b = tl.arange(0, BLOCK_SIZE)
        pm = lead[:, None] & (offs_b[None, :] >= cnt_k[:, None])
        pad_at = bb_k[:, None] * BLOCK_SIZE + offs_b[None, :]
        tl.store(sorted_ids_ptr + pad_at, tl.full((TOPK, BLOCK_SIZE), (TOPK << 24), tl.int32) + M, mask=pm)
        tl.store(sorted_weights_ptr + pad_at, tl.zeros((TOPK, BLOCK_SIZE), tl.float32), mask=pm)
        if t == 0:
            num_valid = tl.sum(first.to(tl.int32), axis=0) * BLOCK_SIZE
            tl.store(num_valid_ids_ptr + tl.arange(0, 2), tl.where(tl.arange(0, 2) == 0, num_valid, M))

    offs_q = tl.arange(0, QCHUNK)
    x = tl.load(qx_ptr + t * stride_qx + c0 + offs_q).to(tl.float32)
    x2 = tl.reshape(x, (QCHUNK // 32, 32))
    amax = tl.maximum(tl.max(tl.abs(x2), axis=1), 1e-10)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    exp = tl.where((bits & 0x7FFFFF) != 0, exp + 1, exp)
    scale = (exp << 23).to(tl.float32, bitcast=True)
    q = tl.clamp(x2 / scale[:, None], -448.0, 448.0)
    tl.store(qout_ptr + t * N_COLS + c0 + offs_q, tl.reshape(q, (QCHUNK,)).to(qout_ptr.dtype.element_ty))
    if MOE_BUF_ZERO:
        tl.store(moe_buf_ptr + t * N_COLS + c0 + offs_q, tl.zeros((QCHUNK,), moe_buf_ptr.dtype.element_ty))
    offs_g = tl.arange(0, QCHUNK // 32)
    y = c0 // 32 + offs_g
    base_sw = (dest_k // 32) * (SCALEN_PAD * 32) + (dest_k % 16) * 4 + (dest_k % 32) // 16
    sw = base_sw[:, None] + ((y // 8) * 256 + (y % 4) * 64 + ((y % 8) // 4) * 2)[None, :]
    tl.store(qscale_ptr + sw, tl.broadcast_to(exp[None, :], (TOPK, QCHUNK // 32)).to(tl.uint8))


def biased_topk_sigmoid(
    gating_output: torch.Tensor,  # [M, E] (row-strided ok)
    correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,  # [M, topk] fp32 out
    topk_ids: torch.Tensor,  # [M, topk] i32 out
    renormalize: bool,
    routed_scale: float = 1.0,
) -> None:
    """top-k of sigmoid(logit) + bias (bias rounded to the logits dtype, as
    aiter's biased_grouped_topk does); weights = renormalized sigmoid."""
    M, E = gating_output.shape
    if M == 0:
        return
    topk = topk_ids.shape[1]
    assert E < 0xFFFF and topk & (topk - 1) == 0 and gating_output.stride(1) == 1
    _k3_topk_kernel[(M,)](
        gating_output,
        gating_output.stride(0),
        correction_bias,
        topk_ids,
        topk_weights,
        float(routed_scale),
        E=E,
        E_POW2=triton.next_power_of_2(E),
        TOPK=topk,
        RENORM=renormalize,
        num_warps=4,
    )


SORT_QCHUNK = 512


def sort_quant_supported(m: int, topk: int, block_size: int, n_cols: int) -> bool:
    return (
        0 < m <= block_size
        and topk & (topk - 1) == 0
        and block_size & (block_size - 1) == 0
        and n_cols % SORT_QCHUNK == 0
    )


def sort_quant(
    topk_ids,
    topk_weights,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
    block_size,
    mx_quant_input,  # [M, N] bf16, row-strided ok
    qout,  # [M, N] fp8 out
    qscale,  # u8 [ceil(max_padded/32)*32, scalen_pad] out
):
    """Sort (one block per used expert, needs M <= block_size) + stage-1 mxfp8
    quant + moe_buf zero in one launch; layouts as moe_sorting_small."""
    m, topk = topk_ids.shape
    n = mx_quant_input.shape[1]
    _k3_sort_quant_kernel[(m * (n // SORT_QCHUNK),)](
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        mx_quant_input,
        mx_quant_input.stride(0),
        qout,
        qscale,
        m,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        P_POW2=triton.next_power_of_2(m * topk),
        N_COLS=n,
        QCHUNK=SORT_QCHUNK,
        SCALEN_PAD=qscale.shape[1],
        MOE_BUF_ZERO=moe_buf.numel() > 0,
        num_warps=8,
    )
