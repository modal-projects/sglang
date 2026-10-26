"""K3 MLA target verify (QLEN=8, 12 heads, FP8 KV latent 576) on gfx950.

An on-device switch picks per batch between AITER's Gluon MLA kernel (short
context) and the HipKittens kernel k3_mla_verify_hk (long context): the Gluon
prologue computes bs * max(L), publishes the regime and exits in the HK regime,
and the HK stage 1 is a no-op in the Gluon regime. The grid depends only on bs,
so this is CUDA-graph safe.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch
import triton
import triton.language as tl
from aiter.ops.triton.gluon.mla_gluon import _mla_gluon

from sglang.kernels.ops.attention.k3_mla_verify_hk import (
    PART8_MIN_SPLITS,
    hk_reduce,
    hk_stage1,
)

H, QLEN, D, DV = 12, 8, 576, 512
_GLUON_BLOCK_H = 16
_GLUON_BLOCK_N = 128
_HK_BLOCK_N = 64
_MIN_CHUNK = 256  # min KV tokens per HK split
_TARGET_CTAS = 256  # one HK CTA per CU

# HK regime iff bs * max(L) >= threshold (graph-replay crossovers on MI355X).
_THRESH_HK = {1: 27000, 2: 28000, 4: 27000, 8: 29000, 16: 30000}
_THRESH_HK_DEFAULT = 30000


def auto_threshold_hk(bs: int) -> int:
    for k in sorted(_THRESH_HK):
        if bs <= k:
            return _THRESH_HK[k]
    return _THRESH_HK_DEFAULT


def hk_num_splits(bs: int) -> int:
    """Power-of-two splits per request, bs * nsplit ~ _TARGET_CTAS, at least 2."""
    return max(2, min(triton.next_power_of_2(-(-_TARGET_CTAS // bs)), 256))


@triton.jit
def _split_chunk(
    L, NSPLIT: tl.constexpr, BLOCK_N: tl.constexpr, MIN_CHUNK: tl.constexpr
):
    per = tl.cdiv(L, NSPLIT)
    per = tl.cdiv(per, BLOCK_N) * BLOCK_N
    return tl.maximum(per, MIN_CHUNK)


@triton.jit
def _k3_auto_reduce(
    FLAG,
    KV_INDPTR,
    O_PART,
    LSE_PART,
    G_LOGITS,
    G_LSE,
    O,
    stride_o_tok,
    stride_o_h,
    g_sl_b,
    g_sl_qs,
    g_sl_h,
    g_sl_s,
    g_ml_b,
    g_ml_qs,
    g_ml_h,
    g_ml_s,
    H: tl.constexpr,
    QLEN: tl.constexpr,
    D_V: tl.constexpr,
    NSPLIT: tl.constexpr,
    NSPLIT_P2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_CHUNK: tl.constexpr,
    S_BLK: tl.constexpr,
    G_NSPLIT: tl.constexpr,
    G_S_BLK: tl.constexpr,
    G_BLOCK_N: tl.constexpr,
    D_BLK: tl.constexpr,
):
    ROWS: tl.constexpr = H * QLEN
    b = tl.program_id(0)
    r = tl.program_id(1)
    d0 = tl.program_id(2) * D_BLK
    offs_d = d0 + tl.arange(0, D_BLK)
    qpos = r // H
    h = r % H
    o_off = (b * QLEN + qpos).to(tl.int64) * stride_o_tok + h * stride_o_h
    if tl.load(FLAG) != 0:
        # HK splits, base-2 LSE
        L = tl.load(KV_INDPTR + b + 1) - tl.load(KV_INDPTR + b)
        chunk = _split_chunk(L, NSPLIT, BLOCK_N, MIN_CHUNK)
        nact = tl.minimum(tl.cdiv(L, chunk), NSPLIT)
        base = (b * NSPLIT) * ROWS + r
        offs_all = tl.arange(0, NSPLIT_P2)
        lse_all = tl.load(
            LSE_PART + base + offs_all * ROWS, mask=offs_all < nact, other=float("-inf")
        )
        m = tl.max(lse_all, axis=0)
        m = tl.where(m == float("-inf"), 0.0, m)
        wsum = tl.sum(tl.where(offs_all < nact, tl.math.exp2(lse_all - m), 0.0), axis=0)
        offs_s = tl.arange(0, S_BLK)
        acc = tl.zeros([D_BLK], dtype=tl.float32)
        for s0 in range(0, nact, S_BLK):
            sid = s0 + offs_s
            sm = sid < nact
            lse = tl.load(LSE_PART + base + sid * ROWS, mask=sm, other=float("-inf"))
            w = tl.where(sm, tl.math.exp2(lse - m), 0.0)
            po = tl.load(
                O_PART
                + (base + sid * ROWS)[:, None].to(tl.int64) * D_V
                + offs_d[None, :],
                mask=sm[:, None],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w[:, None] * po, axis=0)
        o = acc / tl.where(wsum > 0.0, wsum, 1.0)
        tl.store(O + o_off + offs_d, o.to(O.dtype.element_ty))
    elif G_NSPLIT > 1:
        # Gluon splits, as aiter _mla_softmax_reducev_kernel
        gL = tl.load(KV_INDPTR + b + 1) - tl.load(KV_INDPTR + b)
        gper = tl.maximum(G_BLOCK_N, gL // G_NSPLIT)
        gnact = tl.minimum(tl.cdiv(gL, gper), G_NSPLIT)
        base_l = b * g_sl_b + qpos * g_sl_qs + h * g_sl_h
        base_ml = b * g_ml_b + qpos * g_ml_qs + h * g_ml_h
        goffs_s = tl.arange(0, G_S_BLK)
        e_sum = 0.0
        e_max = -float("inf")
        gacc = tl.zeros([D_BLK], dtype=tl.float32)
        for s0 in range(0, gnact, G_S_BLK):
            gsid = s0 + goffs_s
            gsm = gsid < gnact
            glse = tl.load(
                G_LSE + base_ml + gsid * g_ml_s, mask=gsm, other=-float("inf")
            )
            lg = tl.load(
                G_LOGITS + base_l + gsid[:, None] * g_sl_s + offs_d[None, :],
                mask=gsm[:, None],
                other=0.0,
            ).to(tl.float32)
            tile_max = tl.max(glse, axis=0)
            n_e_max = tl.maximum(e_max, tile_max)
            old_scale = tl.where(e_max == -float("inf"), 0.0, tl.exp(e_max - n_e_max))
            gw = tl.where(glse == -float("inf"), 0.0, tl.exp(glse - n_e_max))
            lg = tl.where(glse[:, None] == -float("inf"), 0.0, lg)
            gacc = gacc * old_scale + tl.sum(gw[:, None] * lg, axis=0)
            e_sum = e_sum * old_scale + tl.sum(gw, axis=0)
            e_max = n_e_max
        go = gacc / tl.where(e_sum > 0.0, e_sum, 1.0)
        tl.store(O + o_off + offs_d, go.to(O.dtype.element_ty))


def k3_mla_verify_auto(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    kv_scale: torch.Tensor,
    kv_scale_float: float,
    thresh: Optional[int] = None,
) -> torch.Tensor:
    """kv_scale: fp32 [1]; kv_scale_float: the same value as a host float (Gluon).
    thresh overrides auto_threshold_hk(bs)."""
    assert kv_buffer.shape[-1] == D
    kv = kv_buffer.view(-1, D)
    q = q.view(-1, H, D)
    ntok = q.shape[0]
    bs = ntok // QLEN
    out = q.new_empty((ntok, H, DV))
    if bs == 0:
        return out
    if thresh is None:
        thresh = auto_threshold_hk(bs)
    dev = q.device
    rows = H * QLEN

    # Gluon (bh16bn128): 256 workgroups; with one split it writes out directly.
    g_ns = max(1, 256 // (bs * QLEN * triton.cdiv(H, _GLUON_BLOCK_H)))
    q4 = q.view(bs, QLEN, H, D)
    q_nope = q4[..., :DV]
    q_pe = q4[..., DV:]
    if g_ns == 1:
        g_logits = out.view(bs, QLEN, H, 1, DV)
        g_lse = None
        g_ml = (0, 0, 0, 0)
    else:
        g_logits = torch.empty((bs, QLEN, H, g_ns, DV), dtype=out.dtype, device=dev)
        g_lse = torch.empty((bs, QLEN, H, g_ns), dtype=torch.float32, device=dev)
        g_ml = g_lse.stride()
    g_sl = g_logits.stride()[:4]
    g_within_2gb = kv.shape[0] * kv.stride(0) * kv.element_size() <= 0x80000000

    nsplit = hk_num_splits(bs)
    # The HIP merge (bs <= 3) also reads fp8 partials; the Triton merge is faster at bs >= 4.
    hip_reduce = nsplit >= 128
    part8_min = PART8_MIN_SPLITS if hip_reduce else 0
    o_part = torch.empty((bs, nsplit, rows, DV), dtype=torch.bfloat16, device=dev)
    lse_part = torch.empty((bs, nsplit, rows), dtype=torch.float32, device=dev)
    sel_p2 = max(256, triton.next_power_of_2(bs + 2))

    # sel: HK-regime kv_indptr (all zero in the Gluon regime) + regime flag.
    sel = torch.empty((bs + 2,), dtype=torch.int32, device=dev)
    flag = sel[bs + 1 :]
    _mla_gluon[(bs, g_ns, triton.cdiv(H, _GLUON_BLOCK_H) * QLEN)](
        q_nope,
        q_pe,
        kv,
        kv,
        kv_indices,
        kv_indptr,
        g_logits,
        _dummy_f32(dev),
        sm_scale,
        kv_scale_float,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        q_pe.stride(0),
        q_pe.stride(1),
        q_pe.stride(2),
        kv.stride(0),
        kv.stride(0),
        0,
        g_sl[0],
        g_sl[1],
        g_sl[2],
        g_sl[3],
        g_lse,
        g_ml[0],
        g_ml[1],
        g_ml[2],
        g_ml[3],
        None,
        0,
        0,
        0,
        BLOCK_H=_GLUON_BLOCK_H,
        BLOCK_N=_GLUON_BLOCK_N,
        NUM_KV_SPLITS=g_ns,
        PAGE_SIZE=1,
        HEAD_DIM_CKV=DV,
        HEAD_DIM_KPE=D - DV,
        KV_PE_OFFSET=DV,
        USE_2D_VIEW=False,
        WITHIN_2GB=g_within_2gb,
        NUM_XCDS=1,
        NHEAD=H,
        REGIME="bh16bn128",
        RETURN_LSE=False,
        QLEN=QLEN,
        HAS_PE=True,
        HAS_ATTN_SINK=False,
        Sel_buf=sel,
        sel_bs=bs,
        sel_thresh=thresh,
        SEL_P2=sel_p2,
        REGIME_SEL=True,
    )
    hk_stage1(
        q,
        kv,
        sel,
        kv_indices,
        kv_scale,
        o_part,
        lse_part,
        nsplit,
        _MIN_CHUNK,
        sm_scale,
        part8_min,
    )
    if hip_reduce:
        hk_reduce(
            o_part,
            lse_part,
            kv_indptr,
            out,
            nsplit,
            _MIN_CHUNK,
            kv_scale,
            part8_min,
            flag,
            g_logits,
            g_lse,
            g_ns,
            _GLUON_BLOCK_N,
        )
        return out
    _k3_auto_reduce[(bs, rows, DV // 128)](
        flag,
        kv_indptr,
        o_part,
        lse_part,
        g_logits,
        g_lse if g_lse is not None else lse_part,
        out,
        out.stride(0),
        out.stride(1),
        g_sl[0],
        g_sl[1],
        g_sl[2],
        g_sl[3],
        g_ml[0],
        g_ml[1],
        g_ml[2],
        g_ml[3],
        H=H,
        QLEN=QLEN,
        D_V=DV,
        NSPLIT=nsplit,
        NSPLIT_P2=triton.next_power_of_2(nsplit),
        BLOCK_N=_HK_BLOCK_N,
        MIN_CHUNK=_MIN_CHUNK,
        S_BLK=min(64, triton.next_power_of_2(nsplit)),
        G_NSPLIT=g_ns,
        G_S_BLK=min(64, triton.next_power_of_2(g_ns)),
        G_BLOCK_N=_GLUON_BLOCK_N,
        D_BLK=128,
        num_warps=4 if nsplit >= 64 else 2,
    )
    return out


@functools.lru_cache(maxsize=8)
def _dummy_f32(device):
    return torch.empty(1, dtype=torch.float32, device=device)
