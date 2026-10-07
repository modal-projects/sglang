"""K3 MLA target-verify (QLEN=8) split-KV attention for gfx950, v3.

Same contract and reduce as k3_mla_verify_v2 (one program = all 8 q_pos x 12
heads of a request x one KV chunk, KV read once), but stage 1 runs 8 warps =
2 waves per SIMD so LDS->MFMA latency of one wave hides behind the other
(v2's single wave per SIMD reached ~30% of FP8 MFMA peak):

  * warps_per_cta = [4, 2]: warp (r, c) owns rows 32r..32r+31; for S it owns
    tokens 32c..32c+31 of the tile, for O it owns half of every 128-column
    chunk -> 128 fp32 accumulator VGPRs per wave.
  * Q (FP8, per-row scale) lives in LDS and is re-read per tile; only q_pe
    stays in VGPRs.
  * KV tiles go global -> VGPR -> LDS one tile ahead (async LDS DMA trips an
    LLVM waitcnt assertion with 8 warps in this Triton build).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton

from sglang.kernels.ops.attention.k3_mla_verify_v2 import (
    _HAS_GLUON,
    _LOG2E,
    _k3_mla_verify_reduce,
    _ones,
)

if _HAS_GLUON:
    from sglang.kernels.ops.attention.k3_mla_verify_v2 import _k3_mla_verify_fwd_gluon

_TARGET_CTAS = 256
_MAX_SPLITS = 256
_MIN_CHUNK = 128
_LAZY_TAU = 2.0
# Default: batches up to this size use the 8-warp (2 waves/SIMD) stage 1;
# larger batches use v2's 4-warp stage 1 (lower LDS traffic, wins at long
# context). Override per call with eight_warps / w8_max_bs.
_W8_MAX_BS = 2

if _HAS_GLUON:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    from sglang.kernels.ops.attention.k3_mla_verify_v2 import (
        _gl_load_idx,
        _gl_split_chunk,
    )

    @gluon.jit
    def _gl_gather(KV, tok, offs_d, KV_STRIDE: gl.constexpr, OFF: gl.constexpr, WITHIN_2GB: gl.constexpr):
        if WITHIN_2GB:
            x = gl.amd.cdna4.buffer_load(KV, tok[:, None] * KV_STRIDE + OFF + offs_d[None, :])
        else:
            x = gl.load(KV + (tok.to(gl.int64) * KV_STRIDE + OFF)[:, None] + offs_d[None, :])
        return x

    @gluon.jit
    def _k3_mla_verify_v3_fwd(
        Q,
        KV,
        KV_INDPTR,
        KV_INDICES,
        KV_SCALE,
        O_PART,
        LSE_PART,
        stride_q_tok,
        stride_q_h,
        sm_scale_log2,
        KV_STRIDE: gl.constexpr,
        H: gl.constexpr,
        QLEN: gl.constexpr,
        BLOCK_N: gl.constexpr,
        NSPLIT: gl.constexpr,
        MIN_CHUNK: gl.constexpr,
        P_SCALE: gl.constexpr,
        LAZY_TAU: gl.constexpr,
        WITHIN_2GB: gl.constexpr,
    ):
        BLOCK_M: gl.constexpr = 128
        D_V: gl.constexpr = 512
        D_PE: gl.constexpr = 64
        DC: gl.constexpr = 128
        NW: gl.constexpr = 8
        gl.static_assert(BLOCK_N == 64)
        ROWS: gl.constexpr = H * QLEN
        split = gl.program_id(0)
        b = gl.program_id(1)

        kv_start = gl.load(KV_INDPTR + b)
        L = gl.load(KV_INDPTR + b + 1) - kv_start
        chunk = _gl_split_chunk(L, NSPLIT, BLOCK_N, MIN_CHUNK)
        start = split * chunk
        if start >= L:
            return
        end = gl.minimum(start + chunk, L)
        num_iter = gl.cdiv(end - start, BLOCK_N)

        # ---------------- layouts
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[32, 32, 64], transposed=True, warps_per_cta=[4, 2])
        dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=16)
        dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=16)
        ld_nope: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [8, 0], [32, 0]],
            lane_bases=[[0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [16, 0]],
            warp_bases=[[1, 0], [2, 0], [4, 0]], block_bases=[], shape=[BLOCK_N, D_V])
        sm_nope: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[1024, 16]],
            offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
                          [16, 0], [1, 0], [2, 0], [4, 0], [8, 0], [32, 0]],
            cga_layout=[], shape=[BLOCK_N, D_V])
        ld_pe: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [NW, 1], [1, 0])
        sm_pe: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[D_PE, 16]], [BLOCK_N, D_PE], [1, 0])
        ld_q: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [NW, 1], [1, 0])  # [128, 128] bf16
        ld_qpe: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [NW, 1], [1, 0])
        sm_q: gl.constexpr = gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=16, order=[1, 0])
        sm_qpe: gl.constexpr = gl.SwizzledSharedLayout(vec=16, per_phase=4, max_phase=4, order=[1, 0])

        bufs_n = gl.allocate_shared_memory(gl.float8e4nv, [2, BLOCK_N, D_V], sm_nope)
        bufs_p = gl.allocate_shared_memory(gl.float8e4nv, [2, BLOCK_N, D_PE], sm_pe)
        sq = gl.allocate_shared_memory(gl.float8e4nv, [BLOCK_M, D_V], sm_q)
        offs_tn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, ld_nope))
        offs_dn = gl.arange(0, D_V, layout=gl.SliceLayout(0, ld_nope))
        offs_tp = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, ld_pe))
        offs_dp = gl.arange(0, D_PE, layout=gl.SliceLayout(0, ld_pe))
        idx_base = KV_INDICES + kv_start

        # ---------------- prologue: Q loads first, then index -> KV chain
        rows_q = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, ld_q))
        dq = gl.arange(0, DC, layout=gl.SliceLayout(0, ld_q))
        q_off = (b * QLEN + rows_q // H) * stride_q_tok + (rows_q % H) * stride_q_h
        qmask = (rows_q < ROWS)[:, None] & (dq < DC)[None, :]
        rows_qp = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, ld_qpe))
        dqp = gl.arange(0, D_PE, layout=gl.SliceLayout(0, ld_qpe))
        qp_off = (b * QLEN + rows_qp // H) * stride_q_tok + (rows_qp % H) * stride_q_h
        qp = gl.amd.cdna4.buffer_load(Q, qp_off[:, None] + D_V + dqp[None, :],
                                      mask=(rows_qp < ROWS)[:, None] & (dqp < D_PE)[None, :], other=0.0)
        qc0 = gl.amd.cdna4.buffer_load(Q, q_off[:, None] + 0 * DC + dq[None, :], mask=qmask, other=0.0)
        qc1 = gl.amd.cdna4.buffer_load(Q, q_off[:, None] + 1 * DC + dq[None, :], mask=qmask, other=0.0)
        qc2 = gl.amd.cdna4.buffer_load(Q, q_off[:, None] + 2 * DC + dq[None, :], mask=qmask, other=0.0)
        qc3 = gl.amd.cdna4.buffer_load(Q, q_off[:, None] + 3 * DC + dq[None, :], mask=qmask, other=0.0)

        tn = _gl_load_idx(idx_base, start, end, offs_tn)
        tp = _gl_load_idx(idx_base, start, end, offs_tp)
        tok_n = _gl_load_idx(idx_base, start + BLOCK_N, end, offs_tn)
        tok_p = _gl_load_idx(idx_base, start + BLOCK_N, end, offs_tp)
        kn = _gl_gather(KV, tn, offs_dn, KV_STRIDE, 0, WITHIN_2GB)
        kp = _gl_gather(KV, tp, offs_dp, KV_STRIDE, D_V, WITHIN_2GB)

        qp = qp.to(gl.float32)
        amax = gl.convert_layout(gl.max(gl.abs(qp), axis=1), gl.SliceLayout(1, ld_q))
        amax = gl.maximum(amax, gl.max(gl.abs(qc0.to(gl.float32)), axis=1))
        amax = gl.maximum(amax, gl.max(gl.abs(qc1.to(gl.float32)), axis=1))
        amax = gl.maximum(amax, gl.max(gl.abs(qc2.to(gl.float32)), axis=1))
        amax = gl.maximum(amax, gl.max(gl.abs(qc3.to(gl.float32)), axis=1))
        amax = gl.maximum(amax, 1e-20)
        q_inv = 448.0 / amax
        sq.slice(0 * DC, DC, dim=1).store((qc0.to(gl.float32) * q_inv[:, None]).to(gl.float8e4nv))
        sq.slice(1 * DC, DC, dim=1).store((qc1.to(gl.float32) * q_inv[:, None]).to(gl.float8e4nv))
        sq.slice(2 * DC, DC, dim=1).store((qc2.to(gl.float32) * q_inv[:, None]).to(gl.float8e4nv))
        sq.slice(3 * DC, DC, dim=1).store((qc3.to(gl.float32) * q_inv[:, None]).to(gl.float8e4nv))
        q_inv_p = gl.convert_layout(q_inv, gl.SliceLayout(1, ld_qpe))
        sqp = gl.allocate_shared_memory(gl.float8e4nv, [BLOCK_M, D_PE], sm_qpe,
                                        (qp * q_inv_p[:, None]).to(gl.float8e4nv))
        qp8 = sqp.load(dot_a)
        bufs_n.index(0).store(kn)
        bufs_p.index(0).store(kp)

        kv_scale = gl.load(KV_SCALE)
        amax_m = gl.convert_layout(amax, gl.SliceLayout(1, mma))
        s_scale = amax_m * ((kv_scale * sm_scale_log2) / 448.0)
        rows_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mma))
        row_lim = L - QLEN + rows_m // H
        causal_from = L - QLEN

        m_i = gl.full([BLOCK_M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, mma))
        l_i = gl.full([BLOCK_M], 0.0, gl.float32, layout=gl.SliceLayout(1, mma))
        acc0 = gl.zeros([BLOCK_M, DC], gl.float32, layout=mma)
        acc1 = gl.zeros([BLOCK_M, DC], gl.float32, layout=mma)
        acc2 = gl.zeros([BLOCK_M, DC], gl.float32, layout=mma)
        acc3 = gl.zeros([BLOCK_M, DC], gl.float32, layout=mma)
        offs_s = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mma))

        for i in range(num_iter):
            slot = i % 2
            n0 = start + i * BLOCK_N
            # prefetch tile i+1 into VGPRs, indices of tile i+2
            tok_n2 = _gl_load_idx(idx_base, n0 + 2 * BLOCK_N, end, offs_tn)
            tok_p2 = _gl_load_idx(idx_base, n0 + 2 * BLOCK_N, end, offs_tp)
            kn = _gl_gather(KV, tok_n, offs_dn, KV_STRIDE, 0, WITHIN_2GB)
            kp = _gl_gather(KV, tok_p, offs_dp, KV_STRIDE, D_V, WITHIN_2GB)
            tok_n = tok_n2
            tok_p = tok_p2

            cur_n = bufs_n.index(slot)
            kpT = bufs_p.index(slot).permute((1, 0)).load(dot_b)
            s = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=mma)
            s = gl.amd.cdna4.mfma_scaled(qp8, None, "e4m3", kpT, None, "e4m3", s)
            for c in gl.static_range(4):
                qa = sq.slice(c * DC, DC, dim=1).load(dot_a)
                kT = cur_n.slice(c * DC, DC, dim=1).permute((1, 0)).load(dot_b)
                s = gl.amd.cdna4.mfma_scaled(qa, None, "e4m3", kT, None, "e4m3", s)
            s = s * s_scale[:, None]
            if n0 + BLOCK_N > causal_from:
                offs_n = n0 + offs_s
                vis = (offs_n[None, :] <= row_lim[:, None]) & (offs_n < end)[None, :]
                s = gl.where(vis, s, float("-inf"))
            m_new = gl.maximum(m_i, gl.max(s, axis=1))
            if gl.max(m_new - m_i, axis=0) > LAZY_TAU:
                alpha = gl.exp2(m_i - m_new)
                l_i = l_i * alpha
                acc0 = acc0 * alpha[:, None]
                acc1 = acc1 * alpha[:, None]
                acc2 = acc2 * alpha[:, None]
                acc3 = acc3 * alpha[:, None]
                m_i = m_new
            p = gl.exp2(s - m_i[:, None])
            l_i = l_i + gl.sum(p, axis=1)
            p8 = gl.convert_layout((p * P_SCALE).to(gl.float8e4nv), dot_a)
            v = cur_n.slice(0 * DC, DC, dim=1).load(dot_b)
            acc0 = gl.amd.cdna4.mfma_scaled(p8, None, "e4m3", v, None, "e4m3", acc0)
            v = cur_n.slice(1 * DC, DC, dim=1).load(dot_b)
            acc1 = gl.amd.cdna4.mfma_scaled(p8, None, "e4m3", v, None, "e4m3", acc1)
            v = cur_n.slice(2 * DC, DC, dim=1).load(dot_b)
            acc2 = gl.amd.cdna4.mfma_scaled(p8, None, "e4m3", v, None, "e4m3", acc2)
            v = cur_n.slice(3 * DC, DC, dim=1).load(dot_b)
            acc3 = gl.amd.cdna4.mfma_scaled(p8, None, "e4m3", v, None, "e4m3", acc3)
            bufs_n.index(1 - slot).store(kn)
            bufs_p.index(1 - slot).store(kp)

        l_safe = gl.where(l_i > 0.0, l_i, 1.0)
        o_mul = kv_scale / (l_safe * P_SCALE)
        rows_l = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mma))
        lse = gl.where(l_i > 0.0, m_i + gl.log2(l_safe), float("-inf"))
        gl.amd.cdna4.buffer_store(lse, LSE_PART, (b * NSPLIT + split) * ROWS + rows_l, mask=rows_l < ROWS)
        st: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [NW, 1], [1, 0])
        rows_o = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, st))
        dv = gl.arange(0, DC, layout=gl.SliceLayout(0, st))
        o_off = (((b * NSPLIT + split) * ROWS + rows_o) * D_V)[:, None] + dv[None, :]
        om = (rows_o < ROWS)[:, None] & (dv < DC)[None, :]
        ety = O_PART.dtype.element_ty
        o_mul2 = o_mul[:, None]
        gl.amd.cdna4.buffer_store(gl.convert_layout((acc0 * o_mul2).to(ety), st), O_PART, o_off + 0 * DC, mask=om)
        gl.amd.cdna4.buffer_store(gl.convert_layout((acc1 * o_mul2).to(ety), st), O_PART, o_off + 1 * DC, mask=om)
        gl.amd.cdna4.buffer_store(gl.convert_layout((acc2 * o_mul2).to(ety), st), O_PART, o_off + 2 * DC, mask=om)
        gl.amd.cdna4.buffer_store(gl.convert_layout((acc3 * o_mul2).to(ety), st), O_PART, o_off + 3 * DC, mask=om)


def default_num_splits(bs: int) -> int:
    n = max(1, -(-_TARGET_CTAS // bs))
    p = 1
    while p < n:
        p *= 2
    return min(p, _MAX_SPLITS)


def k3_mla_verify_v3(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    kv_scale=None,
    *,
    num_heads: int = 12,
    qlen: int = 8,
    v_head_dim: int = 512,
    out: Optional[torch.Tensor] = None,
    num_splits: Optional[int] = None,
    eight_warps: Optional[bool] = None,
    w8_max_bs: int = _W8_MAX_BS,
) -> torch.Tensor:
    """Same contract as k3_mla_verify_v2. Graph-capturable (grid depends on bs only)."""
    assert _HAS_GLUON, "k3_mla_verify_v3 needs triton.experimental.gluon (gfx950)"
    D = kv_buffer.shape[-1]
    assert D == 576 and v_head_dim == 512 and num_heads * qlen <= 128
    kv = kv_buffer.view(-1, D)
    q = q.view(-1, num_heads, D)
    assert q.stride(-1) == 1 and kv.stride(-1) == 1
    ntok = q.shape[0]
    bs = ntok // qlen
    assert bs * qlen == ntok
    if out is None:
        out = q.new_empty((ntok, num_heads, v_head_dim))
    if kv_scale is None:
        kv_scale = _ones(q.device)
    elif not isinstance(kv_scale, torch.Tensor):
        kv_scale = torch.full((1,), float(kv_scale), dtype=torch.float32, device=q.device)
    if bs == 0:
        return out
    nsplit = max(2, num_splits or default_num_splits(bs))
    block_n = 64
    rows = num_heads * qlen
    o_part = torch.empty((bs, nsplit, rows, v_head_dim), dtype=torch.bfloat16, device=q.device)
    lse_part = torch.empty((bs, nsplit, rows), dtype=torch.float32, device=q.device)
    p_scale = 256.0 / (2.0**_LAZY_TAU)
    within_2gb = kv.numel() * kv.element_size() <= 0x7FFFFFFF
    if eight_warps is None:
        eight_warps = bs <= w8_max_bs
    if eight_warps:
        _k3_mla_verify_v3_fwd[(nsplit, bs)](
            q, kv, kv_indptr, kv_indices, kv_scale, o_part, lse_part,
            q.stride(0), q.stride(1), sm_scale * _LOG2E,
            KV_STRIDE=kv.stride(0), H=num_heads, QLEN=qlen, BLOCK_N=block_n, NSPLIT=nsplit,
            MIN_CHUNK=_MIN_CHUNK, P_SCALE=p_scale, LAZY_TAU=_LAZY_TAU, WITHIN_2GB=within_2gb,
            num_warps=8,
        )
    else:
        _k3_mla_verify_fwd_gluon[(nsplit, bs)](
            q, kv, kv_indptr, kv_indices, kv_scale, o_part, lse_part, out,
            q.stride(0), q.stride(1), out.stride(0), out.stride(1),
            sm_scale * _LOG2E,
            KV_STRIDE=kv.stride(0), H=num_heads, QLEN=qlen, BLOCK_M=128,
            BLOCK_N=block_n, D_V=v_head_dim, D_PE=D - v_head_dim, NSPLIT=nsplit, MIN_CHUNK=_MIN_CHUNK,
            P_SCALE=p_scale, WITHIN_2GB=within_2gb, NUM_WARPS=4, NUM_STAGES=3, PAD_PAIRS=((1024, 16),),
            LAZY_TAU=_LAZY_TAU,
            num_warps=4,
        )
    _k3_mla_verify_reduce[(bs, rows, v_head_dim // 128)](
        o_part, lse_part, kv_indptr, out, out.stride(0), out.stride(1),
        H=num_heads, QLEN=qlen, D_V=v_head_dim, NSPLIT=nsplit,
        NSPLIT_P2=triton.next_power_of_2(nsplit), BLOCK_N=block_n, MIN_CHUNK=_MIN_CHUNK,
        S_BLK=min(64, triton.next_power_of_2(nsplit)), D_BLK=128,
        num_warps=4 if nsplit >= 64 else 2,
    )
    return out
