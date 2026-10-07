# SPDX-License-Identifier: Apache-2.0
"""Two-launch MoE routing glue for mid-sized batches (SGLANG_ROCM_K3_ROUTE_SORT_MID).

Replaces, for 16 < M <= SGLANG_ROCM_K3_ROUTE_SORT_MID_MAX_TOKENS (default 128,
kernel limit 512; it loses to aiter from M = 256) on the ROCm aiter a8w4 MoE
path (K3 DFlash verify),
  aiter topk_reg (or any top-k) + opus_moe_sorting P0_v2 + P23
  + fused_mx_quant_moe_sort (M <= 128) / per_1x32_mx_quant + mxfp4_moe_sort (M > 128)
with two launches:

  A  _k3_mid_route_kernel, one CTA per token:
       [sigmoid + bias top-k ->] topk_ids / topk_weights
       atomic per-expert pair counts (int32, counts[E])
       mxfp8 quant of the token's activation row -> fp8 row + e8m0 bytes [M, G]
  B  _k3_mid_sort_kernel, one CTA per token + one CTA per expert chunk:
       every CTA reads counts -> block prefix (blocks of smaller experts)
       token CTA t: stable rank of each of its pairs among earlier tokens' pairs
         of the same expert -> sorted_ids / sorted_weights at dest, its e8m0
         bytes scattered to the dest rows' swizzled scale slots, moe_buf row zero
       expert-chunk CTA: sorted_expert_ids, block-tail padding, num_valid_ids
       the last CTA to finish (atomic ticket, no waiting) re-zeroes counts

Output layout (identical to aiter's opus sort, moe_sorting_small, k3_route_sort):
  sorted_ids[i]        = (topk_slot << 24) | token   (padding: (topk << 24) | M)
  sorted_weights       = pair weight                 (padding: 0)
  sorted_expert_ids[b] = expert of block b
  num_valid_ids        = [num_blocks * block_size, M]
  pairs of one expert in (token, slot) order, experts ascending, one block run each
  a1                   = per-token fp8 rows; scale byte per (sorted_row, group) at
                         aiter's mx_scale_shuffle_idx (RoundUp e8m0, as aiter)

Only the valid region (< num_valid_ids[0]) is defined, as in aiter: padding rows'
scale bytes are left unwritten (aiter's fused M <= 128 quant leaves them too).
A pair with routed weight exactly 0 gets scale byte 0, as aiter's fused quant does
(ZERO_W_SCALE; aiter's M > 128 split path does not do this).

The counts buffer must be all-zero before launch A; kernel B leaves it zeroed.
If A runs and B does not (the MoE runner took another path), call reset().
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


MID_MAX_TOKENS = 512  # kernel limit; dispatch cap is SGLANG_ROCM_K3_ROUTE_SORT_MID_MAX_TOKENS
# aiter fused_dynamic_mx_quant_moe_sort uses its fused (zero-weight aware)
# kernel at stage 1 up to M = 8 * 256 / topk tokens, the split path above
_AITER_FUSED_QUANT_PAIRS = 8 * 256


def _quant_chunk(n_cols: int) -> int | None:
    for chunk in (1024, 512, 256, 128, 64, 32):
        if chunk <= n_cols and n_cols % chunk == 0:
            return chunk
    return None


@triton.jit
def _aiter_sigmoid(x):
    # aiter topk_reg: __builtin_amdgcn_rcpf(1.0f + exp2f(-C_LOG2E * x)), where
    # C_LOG2E is a double literal (the product is formed in fp64)
    a = (x.to(tl.float64) * -1.44269504088896340736).to(tl.float32)
    ex = tl.inline_asm_elementwise("v_exp_f32 $0, $1", "=v,v", [a], dtype=tl.float32, is_pure=True, pack=1)
    return tl.inline_asm_elementwise("v_rcp_f32 $0, $1", "=v,v", [1.0 + ex], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _aiter_tie_rank(idx):
    # aiter legacy_topk_tie_rank (wave64 DPP order): lower wins among equal scores
    vec4 = idx >> 2
    lane = vec4 & 63
    local = ((vec4 >> 6) << 2) | (idx & 3)
    low2 = (lane & 3) ^ tl.where((lane & 4) != 0, 3, 0)
    bit2 = (((lane >> 2) ^ (lane >> 3)) & 1) << 2
    lane_rank = ((~lane) & 0x38) | bit2 | low2
    return (lane_rank << 5) | local


@triton.jit
def _aiter_biased_topk(logits_ptr, bias_ptr, offs_e, me, E_POW2: tl.constexpr, TOPK: tl.constexpr,
                       RENORM: tl.constexpr, routed_scale):
    """Bit-exact aiter biased_grouped_topk (topk_reg_kernel, one group): ids in
    (score desc, legacy tie rank asc) order, weights = pre-bias sigmoid times
    routed_scale / serial-sum."""
    lg = tl.load(logits_ptr + offs_e, mask=me, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs_e, mask=me, other=0.0).to(logits_ptr.dtype.element_ty).to(tl.float32)
    sig = _aiter_sigmoid(lg)
    sc = tl.where(me, sig + bias, float("-inf"))
    b = sc.to(tl.int32, bitcast=True)
    k = tl.where(b >= 0, b, b ^ 0x7FFFFFFF).to(tl.int64)
    key = (k << 32) | ((0xFFFF - _aiter_tie_rank(offs_e)).to(tl.int64) << 16) | offs_e.to(tl.int64)
    top = tl.topk(key, TOPK, dim=0)  # descending
    ids = (top & 0xFFFF).to(tl.int32)
    w = _aiter_sigmoid(tl.load(logits_ptr + ids).to(tl.float32))
    offs_k = tl.arange(0, TOPK)
    if RENORM:
        total = 0.0
        for i in tl.static_range(TOPK):
            total += tl.sum(tl.where(offs_k == i, w, 0.0), axis=0)
        f = tl.where(total > 0.0, routed_scale / total, 0.0)
        w = w * f
    else:
        w = w * routed_scale
    return ids, w


@triton.jit(do_not_specialize=["stride_lg", "stride_qx"])
def _k3_mid_route_kernel(
    logits_ptr,  # [M, E] bf16/fp32, row stride stride_lg (DO_TOPK only)
    stride_lg,
    bias_ptr,  # [E] correction bias (DO_TOPK only)
    topk_ids_ptr,  # [M, TOPK] i32 (out if DO_TOPK, else in)
    topk_weights_ptr,  # [M, TOPK] fp32 (out if DO_TOPK)
    routed_scale,
    counts_ptr,  # [>= E] i32, zero on entry; += pairs per expert
    qx_ptr,  # [M, N_COLS] activations, row stride stride_qx
    stride_qx,
    qout_ptr,  # [M, N_COLS] fp8 out
    exps_ptr,  # [M, G] u8 out (e8m0 per (token, group))
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    RENORM: tl.constexpr,
    DO_TOPK: tl.constexpr,
    N_COLS: tl.constexpr,
    QCHUNK: tl.constexpr,
    COUNT_QUANT: tl.constexpr = True,
):
    t = tl.program_id(0).to(tl.int64)
    offs_k = tl.arange(0, TOPK)
    if DO_TOPK:
        offs_e = tl.arange(0, E_POW2)
        ids, w = _aiter_biased_topk(
            logits_ptr + t * stride_lg, bias_ptr, offs_e, offs_e < E, E_POW2, TOPK, RENORM, routed_scale
        )
        tl.store(topk_ids_ptr + t * TOPK + offs_k, ids)
        tl.store(topk_weights_ptr + t * TOPK + offs_k, w)
    else:
        ids = tl.load(topk_ids_ptr + t * TOPK + offs_k)
    if not COUNT_QUANT:
        return
    tl.atomic_add(counts_ptr + ids, tl.full((TOPK,), 1, tl.int32), sem="relaxed")

    G: tl.constexpr = N_COLS // 32
    offs_q = tl.arange(0, QCHUNK)
    offs_g = tl.arange(0, QCHUNK // 32)
    for cc in tl.static_range(N_COLS // QCHUNK):
        c0 = cc * QCHUNK
        x = tl.load(qx_ptr + t * stride_qx + c0 + offs_q).to(tl.float32)
        x2 = tl.reshape(x, (QCHUNK // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(x2), axis=1), 1e-10)
        bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
        exp = (bits >> 23) & 0xFF
        exp = tl.where((bits & 0x7FFFFF) != 0, exp + 1, exp)
        scale = (exp << 23).to(tl.float32, bitcast=True)
        q = tl.clamp(x2 / scale[:, None], -448.0, 448.0)
        tl.store(qout_ptr + t * N_COLS + c0 + offs_q, tl.reshape(q, (QCHUNK,)).to(qout_ptr.dtype.element_ty))
        tl.store(exps_ptr + t * G + c0 // 32 + offs_g, exp.to(tl.uint8))


@triton.jit(do_not_specialize=["M"])
def _k3_mid_sort_kernel(
    topk_ids_ptr,  # [M, TOPK] i32
    topk_weights_ptr,  # [M, TOPK] fp32
    counts_ptr,  # [E_POW2 + 1] i32: counts[0:E], ticket at [E_POW2]; re-zeroed here
    exps_ptr,  # [M, G] u8
    sorted_ids_ptr,
    sorted_weights_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    qscale_ptr,  # u8, swizzled, one byte per (sorted_row, group)
    moe_buf_ptr,  # [M, BUF_COLS] zeroed here (MOE_BUF_ZERO)
    M,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAXB_POW2: tl.constexpr,  # >= max blocks of one expert (cdiv(M, BLOCK_SIZE))
    XC: tl.constexpr,  # experts per expert-chunk CTA
    G: tl.constexpr,
    G_POW2: tl.constexpr,
    SCALEN_PAD: tl.constexpr,
    RANK_TILE: tl.constexpr,
    ZERO_W_SCALE: tl.constexpr,
    MOE_BUF_ZERO: tl.constexpr,
    BUF_COLS: tl.constexpr,
    BUF_CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs_e = tl.arange(0, E_POW2)
    me = offs_e < E
    cnt = tl.load(counts_ptr + offs_e, mask=me, other=0)
    nb = (cnt + BLOCK_SIZE - 1) // BLOCK_SIZE
    is_tok = pid < M
    xc = tl.maximum(pid - M, 0)
    offs_x = xc * XC + tl.arange(0, XC)
    mx = offs_x < E
    # the chunk's own counts, read before the ticket like everything else
    cnt_x = tl.load(counts_ptr + offs_x, mask=mx & (pid >= M), other=0)

    # ticket: once every CTA has read counts, the last one re-zeroes them
    # (data-dependent increment keeps the counts reads ahead of the atomic)
    # (vector form, one active lane: the result is only waited on at the end)
    dep = tl.sum(cnt, axis=0) + tl.sum(cnt_x, axis=0)
    offs_one = tl.arange(0, 2)
    tick = tl.atomic_add(
        counts_ptr + E_POW2 + offs_one * 0,
        tl.where(dep >= 0, 1, 2) + offs_one * 0,
        mask=offs_one == 0,
        sem="relaxed",
        scope="gpu",
    )

    if is_tok:
        t = pid
        offs_k = tl.arange(0, TOPK)
        ek = tl.load(topk_ids_ptr + t * TOPK + offs_k)
        wk = tl.load(topk_weights_ptr + t * TOPK + offs_k)
        # stable rank: pairs of earlier tokens on the same expert (a token
        # picks an expert at most once)
        rank = tl.zeros((TOPK,), tl.int32)
        n_prev = t * TOPK
        for s in range(0, n_prev, RANK_TILE):
            offs_p = s + tl.arange(0, RANK_TILE)
            ep = tl.load(topk_ids_ptr + offs_p, mask=offs_p < n_prev, other=-1)
            rank += tl.sum((ep[None, :] == ek[:, None]).to(tl.int32), axis=1)
        bb_all = tl.cumsum(nb, axis=0) - nb
        bb_k = tl.gather(bb_all, ek, axis=0)
        dest = bb_k * BLOCK_SIZE + rank
        tl.store(sorted_ids_ptr + dest, (offs_k << 24) | t)
        tl.store(sorted_weights_ptr + dest, wk)

        offs_g = tl.arange(0, G_POW2)
        mg = offs_g < G
        ex = tl.load(exps_ptr + t.to(tl.int64) * G + offs_g, mask=mg, other=0).to(tl.int32)
        val = tl.broadcast_to(ex[None, :], (TOPK, G_POW2))
        if ZERO_W_SCALE:
            val = tl.where(wk[:, None] == 0.0, 0, val)
        base_sw = (dest // 32) * (SCALEN_PAD * 32) + (dest % 16) * 4 + (dest % 32) // 16
        sw = base_sw[:, None] + ((offs_g // 8) * 256 + (offs_g % 4) * 64 + ((offs_g % 8) // 4) * 2)[None, :]
        tl.store(qscale_ptr + sw, val.to(tl.uint8), mask=mg[None, :])

        if MOE_BUF_ZERO:
            offs_c = tl.arange(0, BUF_CHUNK)
            row = moe_buf_ptr + t.to(tl.int64) * BUF_COLS
            for c0 in tl.static_range(0, BUF_COLS, BUF_CHUNK):
                tl.store(
                    row + c0 + offs_c,
                    tl.zeros((BUF_CHUNK,), moe_buf_ptr.dtype.element_ty),
                    mask=c0 + offs_c < BUF_COLS,
                )
    else:
        # expert chunk: expert id per block, block-tail padding, num_valid_ids
        nb_x = (cnt_x + BLOCK_SIZE - 1) // BLOCK_SIZE
        x0 = xc * XC
        base = tl.sum(tl.where(offs_e < x0, nb, 0), axis=0)
        bb_x = base + tl.cumsum(nb_x, axis=0) - nb_x
        offs_j = tl.arange(0, MAXB_POW2)
        tl.store(
            sorted_expert_ids_ptr + bb_x[:, None] + offs_j[None, :],
            tl.broadcast_to(offs_x[:, None], (XC, MAXB_POW2)),
            mask=mx[:, None] & (offs_j[None, :] < nb_x[:, None]),
        )
        offs_b = tl.arange(0, BLOCK_SIZE)
        npad = nb_x * BLOCK_SIZE - cnt_x
        pad_at = bb_x[:, None] * BLOCK_SIZE + cnt_x[:, None] + offs_b[None, :]
        pm = mx[:, None] & (offs_b[None, :] < npad[:, None])
        tl.store(sorted_ids_ptr + pad_at, tl.full((XC, BLOCK_SIZE), (TOPK << 24), tl.int32) + M, mask=pm)
        tl.store(sorted_weights_ptr + pad_at, tl.zeros((XC, BLOCK_SIZE), tl.float32), mask=pm)
        if xc == 0:
            num_valid = tl.sum(nb, axis=0) * BLOCK_SIZE
            two = tl.arange(0, 2)
            tl.store(num_valid_ids_ptr + two, tl.where(two == 0, num_valid, M))

    last = tl.max(tl.where(offs_one == 0, tick, -1), axis=0) == nprog - 1
    if last:
        tl.store(counts_ptr + offs_e, tl.zeros((E_POW2,), tl.int32), mask=me)
        tl.store(counts_ptr + E_POW2, 0)


@triton.jit
def _max_combine(a, b):
    return tl.maximum(a, b)


@triton.jit(do_not_specialize=["M", "stride_qx"])
def _k3_sort_quant1_kernel(
    topk_ids_ptr,  # [M, TOPK] i32
    topk_weights_ptr,  # [M, TOPK] fp32
    sorted_ids_ptr,
    sorted_weights_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    moe_buf_ptr,  # [M, N_COLS] zeroed here (MOE_BUF_ZERO)
    qx_ptr,  # [M, N_COLS] activations, row stride stride_qx
    stride_qx,
    qout_ptr,  # [M, N_COLS] fp8 out
    qscale_ptr,  # swizzled e8m0, one per (sorted_row, group)
    M,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    P_POW2: tl.constexpr,
    MAXB_POW2: tl.constexpr,
    N_COLS: tl.constexpr,
    QCHUNK: tl.constexpr,
    SCALEN_PAD: tl.constexpr,
    ZERO_W_SCALE: tl.constexpr,
    MOE_BUF_ZERO: tl.constexpr,
    CPB: tl.constexpr = 1,  # column chunks per CTA
):
    # Single launch, no cross-CTA traffic: k3_route_sort._k3_sort_quant_kernel
    # generalized to several blocks per expert (M > BLOCK_SIZE). Every CTA
    # sorts all M*TOPK expert ids itself, derives per-expert block counts from
    # the sorted runs, and places its token's pairs; CTA (t, c) quantizes
    # token t's column chunk c and scatters the e8m0 bytes to the TOPK dest
    # rows; c == 0 also writes the token's sorted entries and, for experts it
    # leads (rank 0), the expert's block ids and tail padding.
    pid = tl.program_id(0)
    CTAS: tl.constexpr = N_COLS // QCHUNK // CPB
    t = pid // CTAS
    cbase = (pid % CTAS) * (QCHUNK * CPB)
    P = M * TOPK
    offs_p = tl.arange(0, P_POW2)
    mask_p = offs_p < P
    SENT: tl.constexpr = 0x7FFFFFFF
    e = tl.load(topk_ids_ptr + offs_p, mask=mask_p, other=SENT)
    srt = tl.sort(e)  # tl.histogram miscompiles on gfx950
    nxt = tl.gather(srt, tl.minimum(offs_p + 1, P_POW2 - 1), axis=0)
    prv = tl.gather(srt, tl.maximum(offs_p - 1, 0), axis=0)
    valid = srt != SENT
    first = ((offs_p == 0) | (srt != prv)) & valid
    last = ((offs_p == P_POW2 - 1) | (srt != nxt)) & valid
    start = tl.associative_scan(tl.where(first, offs_p, 0), 0, _max_combine)
    nb_last = tl.where(last, (offs_p - start + BLOCK_SIZE) // BLOCK_SIZE, 0)

    offs_k = tl.arange(0, TOPK)
    ek = tl.load(topk_ids_ptr + t * TOPK + offs_k)
    match = e[None, :] == ek[:, None]
    rank_k = tl.sum(tl.where(match & (offs_p[None, :] < t * TOPK), 1, 0), axis=1)
    bb_k = tl.sum(tl.where(srt[None, :] < ek[:, None], nb_last[None, :], 0), axis=1)
    dest_k = bb_k * BLOCK_SIZE + rank_k
    wk = tl.load(topk_weights_ptr + t * TOPK + offs_k)

    if cbase == 0:
        tl.store(sorted_ids_ptr + dest_k, (offs_k << 24) | t)
        tl.store(sorted_weights_ptr + dest_k, wk)
        lead = rank_k == 0
        cnt_k = tl.sum(tl.where(match, 1, 0), axis=1)
        nb_k = (cnt_k + BLOCK_SIZE - 1) // BLOCK_SIZE
        offs_j = tl.arange(0, MAXB_POW2)
        tl.store(
            sorted_expert_ids_ptr + bb_k[:, None] + offs_j[None, :],
            tl.broadcast_to(ek[:, None], (TOPK, MAXB_POW2)),
            mask=lead[:, None] & (offs_j[None, :] < nb_k[:, None]),
        )
        offs_b = tl.arange(0, BLOCK_SIZE)
        npad = nb_k * BLOCK_SIZE - cnt_k
        pad_at = bb_k[:, None] * BLOCK_SIZE + cnt_k[:, None] + offs_b[None, :]
        pm = lead[:, None] & (offs_b[None, :] < npad[:, None])
        tl.store(sorted_ids_ptr + pad_at, tl.full((TOPK, BLOCK_SIZE), (TOPK << 24), tl.int32) + M, mask=pm)
        tl.store(sorted_weights_ptr + pad_at, tl.zeros((TOPK, BLOCK_SIZE), tl.float32), mask=pm)
        if t == 0:
            num_valid = tl.sum(nb_last, axis=0) * BLOCK_SIZE
            two = tl.arange(0, 2)
            tl.store(num_valid_ids_ptr + two, tl.where(two == 0, num_valid, M))

    base_sw = (dest_k // 32) * (SCALEN_PAD * 32) + (dest_k % 16) * 4 + (dest_k % 32) // 16
    offs_q = tl.arange(0, QCHUNK)
    offs_g = tl.arange(0, QCHUNK // 32)
    for cc in tl.static_range(CPB):
        c0 = cbase + cc * QCHUNK
        x = tl.load(qx_ptr + t.to(tl.int64) * stride_qx + c0 + offs_q).to(tl.float32)
        x2 = tl.reshape(x, (QCHUNK // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(x2), axis=1), 1e-10)
        bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
        exp = (bits >> 23) & 0xFF
        exp = tl.where((bits & 0x7FFFFF) != 0, exp + 1, exp)
        scale = (exp << 23).to(tl.float32, bitcast=True)
        q = tl.clamp(x2 / scale[:, None], -448.0, 448.0)
        tl.store(qout_ptr + t.to(tl.int64) * N_COLS + c0 + offs_q, tl.reshape(q, (QCHUNK,)).to(qout_ptr.dtype.element_ty))
        if MOE_BUF_ZERO:
            tl.store(moe_buf_ptr + t.to(tl.int64) * N_COLS + c0 + offs_q, tl.zeros((QCHUNK,), moe_buf_ptr.dtype.element_ty))
        y = c0 // 32 + offs_g
        val = tl.broadcast_to(exp[None, :], (TOPK, QCHUNK // 32))
        if ZERO_W_SCALE:
            val = tl.where(wk[:, None] == 0.0, 0, val)
        sw = base_sw[:, None] + ((y // 8) * 256 + (y % 4) * 64 + ((y % 8) // 4) * 2)[None, :]
        tl.store(qscale_ptr + sw, val.to(tl.uint8))


# ---------------------------------------------------------------------------
# host side

_COUNTS: dict = {}


def counts_buffer(device: torch.device, key=None, num_experts: int = 896) -> torch.Tensor:
    """Persistent zeroed counts (+ ticket) buffer; one per (device, key).

    Allocated once (zeroed) and kept zero by the sort kernel, so it is safe to
    capture into CUDA graphs. Distinct keys (e.g. per MoE layer) keep
    concurrently-captured streams from sharing one buffer."""
    e_pow2 = triton.next_power_of_2(num_experts)
    k = (str(device), key, e_pow2)
    buf = _COUNTS.get(k)
    if buf is None:
        buf = torch.zeros(e_pow2 + 32, dtype=torch.int32, device=device)
        _COUNTS[k] = buf
    return buf


def max_tokens() -> int:
    from sglang.srt.environ import envs

    return min(MID_MAX_TOKENS, envs.SGLANG_ROCM_K3_ROUTE_SORT_MID_MAX_TOKENS.get())


def supported(m: int, topk: int, num_experts: int, block_size: int, n_cols: int) -> bool:
    return (
        0 < m <= max_tokens()
        and topk & (topk - 1) == 0
        and topk <= 64
        and num_experts < 0xFFFF
        and block_size & (block_size - 1) == 0
        and n_cols % 32 == 0
        and _quant_chunk(n_cols) is not None
    )


@dataclass
class RouteState:
    """Launch-A products handed to the sort launch inside fused_moe."""

    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    qx: torch.Tensor
    qout: torch.Tensor
    exps: torch.Tensor
    counts: torch.Tensor
    consumed: bool = False


_pending: ContextVar[RouteState | None] = ContextVar("k3_route_mid_pending", default=None)


def route_quant(
    qx: torch.Tensor,  # [M, N] bf16 routed activations (row-strided ok)
    counts: torch.Tensor,
    *,
    topk_ids: torch.Tensor,  # [M, topk] i32 (out when logits is given)
    topk_weights: torch.Tensor,  # [M, topk] fp32 (out when logits is given)
    logits: torch.Tensor | None = None,  # [M, E] router logits (row-strided ok)
    correction_bias: torch.Tensor | None = None,
    renormalize: bool = True,
    routed_scale: float = 1.0,
    num_experts: int | None = None,
) -> RouteState:
    """Launch A: [top-k +] per-expert counts + per-token mxfp8 quant."""
    m, topk = topk_ids.shape
    n = qx.shape[1]
    do_topk = logits is not None
    e = logits.shape[1] if do_topk else int(num_experts)
    assert qx.stride(1) == 1 and n % 32 == 0
    if do_topk:
        assert logits.stride(1) == 1 and correction_bias is not None
    qout = torch.empty(m, n, dtype=torch.float8_e4m3fn, device=qx.device)
    exps = torch.empty(m, n // 32, dtype=torch.uint8, device=qx.device)
    _k3_mid_route_kernel[(m,)](
        logits if do_topk else qx,
        logits.stride(0) if do_topk else 0,
        correction_bias if do_topk else qx,
        topk_ids,
        topk_weights,
        float(routed_scale),
        counts,
        qx,
        qx.stride(0),
        qout,
        exps,
        E=e,
        E_POW2=triton.next_power_of_2(e),
        TOPK=topk,
        RENORM=bool(renormalize),
        DO_TOPK=do_topk,
        N_COLS=n,
        QCHUNK=_quant_chunk(n),
        num_warps=4,
    )
    return RouteState(topk_ids, topk_weights, qx, qout, exps, counts)


def biased_topk_aiter_exact(
    logits: torch.Tensor,  # [M, E] bf16, row-strided ok
    correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    renormalize: bool = True,
    routed_scale: float = 1.0,
) -> None:
    """Standalone top-k, bit-identical to aiter.biased_grouped_topk (1 group)."""
    m, e = logits.shape
    topk = topk_ids.shape[1]
    assert logits.stride(1) == 1 and e < 2048
    _k3_mid_route_kernel[(m,)](
        logits, logits.stride(0), correction_bias, topk_ids, topk_weights, float(routed_scale),
        topk_ids, logits, 0, topk_ids, topk_ids,
        E=e, E_POW2=triton.next_power_of_2(e), TOPK=topk, RENORM=bool(renormalize), DO_TOPK=True,
        N_COLS=32, QCHUNK=32, COUNT_QUANT=False, num_warps=4,
    )


def sort_scatter(
    state: RouteState,
    sorted_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch B: sort + scale scatter + moe_buf zero; returns (a1, a1_scale)."""
    m, topk = state.topk_ids.shape
    n = state.qout.shape[1]
    g = n // 32
    scalen_pad = (g + 7) // 8 * 8
    qscale = torch.empty(
        (sorted_ids.shape[0] + 31) // 32 * 32, scalen_pad, dtype=torch.uint8, device=sorted_ids.device
    )
    e_pow2 = triton.next_power_of_2(num_experts)
    xc = 128
    n_xc = triton.cdiv(num_experts, xc)
    buf_zero = moe_buf.numel() > 0
    buf_cols = moe_buf.shape[-1] if buf_zero else 32
    if buf_zero:
        assert moe_buf.is_contiguous() and moe_buf.numel() == m * buf_cols
    _k3_mid_sort_kernel[(m + n_xc,)](
        state.topk_ids,
        state.topk_weights,
        state.counts,
        state.exps,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        qscale,
        moe_buf,
        m,
        E=num_experts,
        E_POW2=e_pow2,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        MAXB_POW2=triton.next_power_of_2(triton.cdiv(m, block_size)),
        XC=xc,
        G=g,
        G_POW2=triton.next_power_of_2(g),
        SCALEN_PAD=scalen_pad,
        RANK_TILE=1024,
        ZERO_W_SCALE=m * topk <= _AITER_FUSED_QUANT_PAIRS,
        MOE_BUF_ZERO=buf_zero,
        BUF_COLS=buf_cols,
        BUF_CHUNK=min(1024, triton.next_power_of_2(buf_cols)),
        num_warps=4,
    )
    state.consumed = True
    return state.qout, qscale.view(torch.float8_e8m0fnu)


# The single-launch variant re-sorts all M*topk ids in every CTA; tl.sort of
# 1024 keys costs ~13 us per CTA on gfx950, so it loses to the two-launch path
# from M = 32 (12.8 vs 10.4 us) and is not dispatched (M <= 16 already has
# k3_route_sort.sort_quant). Kept validated for reference.
ONE_LAUNCH_MAX_PAIRS = 0
SORT1_QCHUNK = 512


def sort_quant_one_launch(
    topk_ids,
    topk_weights,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
    block_size,
    qx,  # [M, N] bf16, row-strided ok
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort + stage-1 mxfp8 quant + moe_buf zero in ONE launch (no counts
    buffer, no atomics) for M * topk <= ONE_LAUNCH_MAX_PAIRS."""
    m, topk = topk_ids.shape
    n = qx.shape[1]
    qchunk = SORT1_QCHUNK if n % SORT1_QCHUNK == 0 else _quant_chunk(n)
    g = n // 32
    scalen_pad = (g + 7) // 8 * 8
    qout = torch.empty(m, n, dtype=torch.float8_e4m3fn, device=qx.device)
    qscale = torch.empty(
        (sorted_ids.shape[0] + 31) // 32 * 32, scalen_pad, dtype=torch.uint8, device=qx.device
    )
    buf_zero = moe_buf.numel() > 0
    if buf_zero:
        assert moe_buf.is_contiguous() and moe_buf.numel() == m * n
    cpb = 1  # more chunks per CTA only adds sort latency per CTA
    _k3_sort_quant1_kernel[(m * (n // qchunk // cpb),)](
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        qx,
        qx.stride(0),
        qout,
        qscale,
        m,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        P_POW2=triton.next_power_of_2(m * topk),
        MAXB_POW2=triton.next_power_of_2(triton.cdiv(m, block_size)),
        N_COLS=n,
        QCHUNK=qchunk,
        SCALEN_PAD=scalen_pad,
        ZERO_W_SCALE=m * topk <= _AITER_FUSED_QUANT_PAIRS,
        MOE_BUF_ZERO=buf_zero,
        CPB=cpb,
        num_warps=8,
    )
    return qout, qscale.view(torch.float8_e8m0fnu)


def reset(state: RouteState) -> None:
    """Re-zero the counts of a launch A whose sort launch never ran."""
    state.counts.zero_()


@contextlib.contextmanager
def pending(state: RouteState | None):
    """Offer launch-A products to the patched aiter sort inside fused_moe."""
    if state is None:
        yield None
        return
    token = _pending.set(state)
    try:
        yield state
    finally:
        _pending.reset(token)
        if not state.consumed:
            reset(state)


def take_pending(topk_ids: torch.Tensor, qx: torch.Tensor | None) -> RouteState | None:
    st = _pending.get()
    if (
        st is None
        or st.consumed
        or topk_ids.data_ptr() != st.topk_ids.data_ptr()
        or topk_ids.shape != st.topk_ids.shape
        or qx is None
        or qx.data_ptr() != st.qx.data_ptr()
        or qx.shape != st.qx.shape
        or qx.stride() != st.qx.stride()
    ):
        return None
    return st
