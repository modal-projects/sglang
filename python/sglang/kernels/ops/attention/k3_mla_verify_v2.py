"""K3 MLA target-verify (QLEN=8) split-KV attention for gfx950 (Gluon + Triton).

One program owns ALL 8 query positions x H (=12) heads of one request (96 rows,
padded to BLOCK_M=128) and one contiguous KV chunk, so every KV token is read
from HBM once per layer (the Gluon qlen-8 path re-reads it per q_pos, the
split-4 ASM path twice). Splits per request are derived on the GPU from
kv_indptr, so the grid only depends on the (graph-captured) batch size; empty
splits early-exit.

  stage 1  _k3_mla_verify_fwd_gluon: Q quantized to FP8 per row in-kernel,
           S = Q8 K8^T and O += P8 V8 with FP8 MFMA (32x32x64, unit scales),
           causal mask on the last QLEN keys, online softmax with lazy
           rescaling; writes the per-split normalized partial O (bf16) and
           base-2 LSE.
  stage 2  _k3_mla_verify_reduce: LSE merge of the splits -> bf16 O.

Layout: q [bs*QLEN, H, 576] (row r = q_pos*H + h of request b lives at token
b*QLEN + q_pos), KV pool [N, 576] fp8 e4m3 (V = first 512), kv_indptr [bs+1] /
kv_indices (page size 1, KV length per request INCLUDING the QLEN drafts).
Query position p sees keys [0, L - QLEN + p]. Out-of-range tail tokens of a
tile are fetched from slot 0 and masked out of the softmax.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634


@triton.jit
def _split_chunk(L, NSPLIT: tl.constexpr, BLOCK_N: tl.constexpr, MIN_CHUNK: tl.constexpr):
    per = tl.cdiv(L, NSPLIT)
    per = tl.cdiv(per, BLOCK_N) * BLOCK_N
    return tl.maximum(per, MIN_CHUNK)


@triton.jit
def _k3_mla_verify_reduce(
    O_PART,
    LSE_PART,
    KV_INDPTR,
    O,
    stride_o_tok,
    stride_o_h,
    H: tl.constexpr,
    QLEN: tl.constexpr,
    D_V: tl.constexpr,
    NSPLIT: tl.constexpr,
    NSPLIT_P2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_CHUNK: tl.constexpr,
    S_BLK: tl.constexpr,
    D_BLK: tl.constexpr,
):
    """One program per (request, row, D_BLK columns): base-2 LSE merge."""
    ROWS: tl.constexpr = H * QLEN
    b = tl.program_id(0)
    r = tl.program_id(1)
    d0 = tl.program_id(2) * D_BLK
    L = tl.load(KV_INDPTR + b + 1) - tl.load(KV_INDPTR + b)
    chunk = _split_chunk(L, NSPLIT, BLOCK_N, MIN_CHUNK)
    nact = tl.minimum(tl.cdiv(L, chunk), NSPLIT)
    base = (b * NSPLIT) * ROWS + r
    offs_all = tl.arange(0, NSPLIT_P2)
    lse_all = tl.load(LSE_PART + base + offs_all * ROWS, mask=offs_all < nact, other=float("-inf"))
    m = tl.max(lse_all, axis=0)
    m = tl.where(m == float("-inf"), 0.0, m)
    wsum = tl.sum(tl.where(offs_all < nact, tl.math.exp2(lse_all - m), 0.0), axis=0)
    offs_s = tl.arange(0, S_BLK)
    offs_d = d0 + tl.arange(0, D_BLK)
    acc = tl.zeros([D_BLK], dtype=tl.float32)
    for s0 in range(0, nact, S_BLK):
        sid = s0 + offs_s
        sm = sid < nact
        lse = tl.load(LSE_PART + base + sid * ROWS, mask=sm, other=float("-inf"))
        w = tl.where(sm, tl.math.exp2(lse - m), 0.0)
        po = tl.load(O_PART + (base + sid * ROWS)[:, None].to(tl.int64) * D_V + offs_d[None, :],
                     mask=sm[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(w[:, None] * po, axis=0)
    o = acc / tl.where(wsum > 0.0, wsum, 1.0)
    o_off = (b * QLEN + r // H).to(tl.int64) * stride_o_tok + (r % H) * stride_o_h
    tl.store(O + o_off + offs_d, o.to(O.dtype.element_ty))


# Tuning constants (gfx950, 256 CUs).
_TARGET_CTAS = 256  # bs * NSPLIT ~ one CTA per CU (1 CTA/CU: ~480 VGPRs, 131 KB LDS)
_MAX_SPLITS = 256
_MIN_CHUNK = 256  # min KV tokens per split
_LAZY_TAU = 2.0  # lazy O rescale threshold (log2 units)


def default_num_splits(bs: int) -> int:
    """Power-of-two splits per request so that bs * NSPLIT ~ _TARGET_CTAS."""
    n = max(1, -(-_TARGET_CTAS // bs))
    p = 1
    while p < n:
        p *= 2
    return min(p, _MAX_SPLITS)


def k3_mla_verify_v2(
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
) -> torch.Tensor:
    """MLA target-verify attention, all qlen x num_heads rows of a request per program.

    q: [bs*qlen, H, 576] bf16 (token-major, row q_pos*H + h of request b at token
    b*qlen + q_pos); kv_buffer: fp8 e4m3 pool viewable as [N, 576] (V = first 512);
    kv_indptr [>= bs+1] int32 (KV length per request INCLUDING the qlen drafts),
    kv_indices page-size-1 slots; kv_scale: fp32 [1] tensor (or float / None).
    Returns [bs*qlen, H, v_head_dim] in q.dtype. No host syncs; the grid depends
    only on bs, so it is CUDA-graph capturable."""
    assert _HAS_GLUON, "k3_mla_verify_v2 needs triton.experimental.gluon (gfx950)"
    D = kv_buffer.shape[-1]
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
    # NSPLIT == 1 (direct store) trips an LLVM assertion in this Triton build,
    # so always go through the split + reduce path.
    nsplit = max(2, num_splits or default_num_splits(bs))
    block_n = 64
    min_chunk = _MIN_CHUNK
    rows = num_heads * qlen
    o_part = torch.empty((bs, nsplit, rows, v_head_dim), dtype=torch.bfloat16, device=q.device)
    lse_part = torch.empty((bs, nsplit, rows), dtype=torch.float32, device=q.device)
    # Lazy rescale: the O accumulators are only rescaled when a row max grows
    # by > lazy_tau (log2 units), so p <= 2^lazy_tau and P_SCALE keeps
    # p * P_SCALE <= 256 (< e4m3 max 448).
    lazy_tau = _LAZY_TAU
    p_scale = 256.0 / (2.0**lazy_tau) if lazy_tau > 0 else 256.0
    within_2gb = kv.numel() * kv.element_size() <= 0x7FFFFFFF
    _k3_mla_verify_fwd_gluon[(nsplit, bs)](
        q, kv, kv_indptr, kv_indices, kv_scale, o_part, lse_part, out,
        q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        sm_scale * _LOG2E,
        KV_STRIDE=kv.stride(0), H=num_heads, QLEN=qlen, BLOCK_M=triton.next_power_of_2(rows),
        BLOCK_N=block_n, D_V=v_head_dim, D_PE=D - v_head_dim, NSPLIT=nsplit, MIN_CHUNK=min_chunk,
        P_SCALE=p_scale, WITHIN_2GB=within_2gb, NUM_WARPS=4, NUM_STAGES=3, PAD_PAIRS=((1024, 16),),
        LAZY_TAU=lazy_tau,
        num_warps=4,
    )
    _k3_mla_verify_reduce[(bs, rows, v_head_dim // 128)](
        o_part, lse_part, kv_indptr, out, out.stride(0), out.stride(1),
        H=num_heads, QLEN=qlen, D_V=v_head_dim, NSPLIT=nsplit,
        NSPLIT_P2=triton.next_power_of_2(nsplit), BLOCK_N=block_n, MIN_CHUNK=min_chunk,
        S_BLK=min(64, triton.next_power_of_2(nsplit)), D_BLK=128,
        num_warps=4 if nsplit >= 64 else 2,
    )
    return out


@functools.lru_cache(maxsize=8)
def _ones(device):
    return torch.ones(1, dtype=torch.float32, device=device)




# ============================================================================
# Gluon (gfx950) stage 1 with explicit layouts.
#   * 4 warps, each owns 32 of the BLOCK_M=128 rows (8 q_pos x 12 heads = 96
#     real rows); MFMA 32x32x64 FP8 (mfma_scaled with unit scales) for both
#     S = Q8 K8^T and O += P8 V8.
#   * K_nope tile [BLOCK_N, 512] goes global->LDS with async buffer loads,
#     double buffered. Each warp-instruction writes one contiguous 1 KiB chunk
#     holding tokens (t, t+16) so the QK B-operand reads (16 tokens at the same
#     d) hit 16 distinct bank groups with a 16 B pad per KiB.
#   * K_pe tile [BLOCK_N, 64] goes through registers into a padded LDS tile
#     (80 B pitch, conflict-free).
#   * Q is quantized to FP8 per row in the prologue and stays in VGPRs.
# ============================================================================
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    _HAS_GLUON = True
except ImportError:  # pragma: no cover
    _HAS_GLUON = False


if _HAS_GLUON:

    @gluon.jit
    def _gl_split_chunk(L, NSPLIT: gl.constexpr, BLOCK_N: gl.constexpr, MIN_CHUNK: gl.constexpr):
        per = gl.cdiv(L, NSPLIT)
        per = gl.cdiv(per, BLOCK_N) * BLOCK_N
        return gl.maximum(per, MIN_CHUNK)

    @gluon.jit
    def _gl_issue_nope(dst, KV, tok, offs_d, KV_STRIDE: gl.constexpr, WITHIN_2GB: gl.constexpr):
        # tok: [BLOCK_N] token slots (masked tail -> slot 0, discarded by the
        # softmax mask). No per-lane mask: masked async copies lower to exec
        # branches with full vmcnt drains.
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(dst, KV, tok[:, None] * KV_STRIDE + offs_d[None, :])
        else:
            gl.amd.cdna4.async_copy.global_load_to_shared(
                dst, KV + (tok.to(gl.int64) * KV_STRIDE)[:, None] + offs_d[None, :])

    @gluon.jit
    def _gl_load_pe(KV, tok, offs_d, KV_STRIDE: gl.constexpr, D_V: gl.constexpr, WITHIN_2GB: gl.constexpr):
        if WITHIN_2GB:
            kp = gl.amd.cdna4.buffer_load(KV, tok[:, None] * KV_STRIDE + D_V + offs_d[None, :])
        else:
            kp = gl.load(KV + (tok.to(gl.int64) * KV_STRIDE + D_V)[:, None] + offs_d[None, :])
        return kp

    @gluon.jit
    def _gl_qk(cur_n, cur_p, q0, q1, q2, q3, qp8, mma: gl.constexpr, dot_b: gl.constexpr, DC: gl.constexpr,
               BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
        # operand loads issued one chunk ahead of their MFMAs
        kpT = gl.amd.cdna4.async_copy.load_shared_relaxed(cur_p.permute((1, 0)), dot_b)
        k0 = gl.amd.cdna4.async_copy.load_shared_relaxed(cur_n.slice(0 * DC, DC, dim=1).permute((1, 0)), dot_b)
        s = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=mma)
        s = gl.amd.cdna4.mfma_scaled(qp8, None, "e4m3", kpT, None, "e4m3", s)
        k1 = gl.amd.cdna4.async_copy.load_shared_relaxed(cur_n.slice(1 * DC, DC, dim=1).permute((1, 0)), dot_b)
        s = gl.amd.cdna4.mfma_scaled(q0, None, "e4m3", k0, None, "e4m3", s)
        k2 = gl.amd.cdna4.async_copy.load_shared_relaxed(cur_n.slice(2 * DC, DC, dim=1).permute((1, 0)), dot_b)
        s = gl.amd.cdna4.mfma_scaled(q1, None, "e4m3", k1, None, "e4m3", s)
        k3 = gl.amd.cdna4.async_copy.load_shared_relaxed(cur_n.slice(3 * DC, DC, dim=1).permute((1, 0)), dot_b)
        s = gl.amd.cdna4.mfma_scaled(q2, None, "e4m3", k2, None, "e4m3", s)
        s = gl.amd.cdna4.mfma_scaled(q3, None, "e4m3", k3, None, "e4m3", s)
        return s

    @gluon.jit
    def _gl_load_idx(idx_base, n0, end, offs_t):
        return gl.amd.cdna4.buffer_load(idx_base, n0 + offs_t, mask=(n0 + offs_t) < end, other=0)


    @gluon.jit
    def _k3_mla_verify_fwd_gluon(
        Q,
        KV,
        KV_INDPTR,
        KV_INDICES,
        KV_SCALE,
        O_PART,
        LSE_PART,
        O_FINAL,
        stride_q_tok,
        stride_q_h,
        stride_o_tok,
        stride_o_h,
        sm_scale_log2,
        KV_STRIDE: gl.constexpr,
        H: gl.constexpr,
        QLEN: gl.constexpr,
        BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr,
        D_V: gl.constexpr,
        D_PE: gl.constexpr,
        NSPLIT: gl.constexpr,
        MIN_CHUNK: gl.constexpr,
        P_SCALE: gl.constexpr,
        WITHIN_2GB: gl.constexpr,
        NUM_WARPS: gl.constexpr,
        NUM_STAGES: gl.constexpr = 3,
        LAZY_TAU: gl.constexpr = 0.0,
        PAD_PAIRS: gl.constexpr = ((1024, 16),),
        ASYNC_NOPE: gl.constexpr = True,
    ):
        gl.static_assert((NUM_WARPS == 4) or (NUM_WARPS == 8))
        gl.static_assert(D_V == 512)
        gl.static_assert(D_PE == 64)
        gl.static_assert(BLOCK_M == 128)
        gl.static_assert(BLOCK_N == 64)
        ROWS: gl.constexpr = H * QLEN
        DC: gl.constexpr = D_V // 4
        S: gl.constexpr = NUM_STAGES
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
        # 4 warps: warp w owns rows 32w..32w+31. 8 warps: [4, 2] -> S is split by
        # tokens and O by columns between the two warps sharing a row block.
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[32, 32, 64], transposed=True, warps_per_cta=[4, NUM_WARPS // 4])
        dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=16)
        dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=16)
        # one warp-instruction = 1 KiB contiguous LDS chunk holding tokens (t, t+16)
        if NUM_WARPS == 4:
            ld_nope: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0], [32, 0]],
                lane_bases=[[0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [16, 0]],
                warp_bases=[[1, 0], [2, 0]], block_bases=[], shape=[BLOCK_N, D_V])
        else:
            ld_nope: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [8, 0], [32, 0]],
                lane_bases=[[0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [16, 0]],
                warp_bases=[[1, 0], [2, 0], [4, 0]], block_bases=[], shape=[BLOCK_N, D_V])
        sm_nope: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=PAD_PAIRS,
            offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
                          [16, 0], [1, 0], [2, 0], [4, 0], [8, 0], [32, 0]],
            cga_layout=[], shape=[BLOCK_N, D_V])
        ld_pe: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [NUM_WARPS, 1], [1, 0])
        sm_pe: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[D_PE, 16]], [BLOCK_N, D_PE], [1, 0])
        ld_q: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [NUM_WARPS, 1], [1, 0])  # [128, 128] bf16
        ld_qpe: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [NUM_WARPS, 1], [1, 0])
        sm_q: gl.constexpr = gl.SwizzledSharedLayout(vec=16, per_phase=2, max_phase=8, order=[1, 0])
        sm_qpe: gl.constexpr = gl.SwizzledSharedLayout(vec=16, per_phase=4, max_phase=4, order=[1, 0])

        bufs_n = gl.allocate_shared_memory(gl.float8e4nv, [S, BLOCK_N, D_V], sm_nope)
        bufs_p = gl.allocate_shared_memory(gl.float8e4nv, [S, BLOCK_N, D_PE], sm_pe)
        offs_tn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, ld_nope))
        offs_dn = gl.arange(0, D_V, layout=gl.SliceLayout(0, ld_nope))
        offs_tp = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, ld_pe))
        offs_dp = gl.arange(0, D_PE, layout=gl.SliceLayout(0, ld_pe))
        idx_base = KV_INDICES + kv_start

        # ---------------- prologue. Issue order matters: Q loads first, then the
        # index -> KV chain, so the Q quantization only waits on Q (vmcnt is in order).
        rows_q = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, ld_q))
        rmask_q = rows_q < ROWS
        q_off = (b * QLEN + rows_q // H) * stride_q_tok + (rows_q % H) * stride_q_h
        dq = gl.arange(0, DC, layout=gl.SliceLayout(0, ld_q))
        rows_qp = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, ld_qpe))
        rmask_qp = rows_qp < ROWS
        qp_off = (b * QLEN + rows_qp // H) * stride_q_tok + (rows_qp % H) * stride_q_h
        dqp = gl.arange(0, D_PE, layout=gl.SliceLayout(0, ld_qpe))
        qmask = rmask_q[:, None] & (dq < DC)[None, :]
        qp = gl.amd.cdna4.buffer_load(Q, qp_off[:, None] + D_V + dqp[None, :],
                                      mask=rmask_qp[:, None] & (dqp < D_PE)[None, :], other=0.0)
        qc0 = gl.amd.cdna4.buffer_load(Q, q_off[:, None] + 0 * DC + dq[None, :], mask=qmask, other=0.0)
        qc1 = gl.amd.cdna4.buffer_load(Q, q_off[:, None] + 1 * DC + dq[None, :], mask=qmask, other=0.0)
        qc2 = gl.amd.cdna4.buffer_load(Q, q_off[:, None] + 2 * DC + dq[None, :], mask=qmask, other=0.0)
        qc3 = gl.amd.cdna4.buffer_load(Q, q_off[:, None] + 3 * DC + dq[None, :], mask=qmask, other=0.0)

        # tiles 0..S-2 in flight, indices of tile S-1 in regs
        tn0 = _gl_load_idx(idx_base, start, end, offs_tn)
        tp0 = _gl_load_idx(idx_base, start, end, offs_tp)
        tn1 = _gl_load_idx(idx_base, start + BLOCK_N, end, offs_tn)
        tp1 = _gl_load_idx(idx_base, start + BLOCK_N, end, offs_tp)
        tok_n = _gl_load_idx(idx_base, start + 2 * BLOCK_N, end, offs_tn)
        tok_p = _gl_load_idx(idx_base, start + 2 * BLOCK_N, end, offs_tp)
        kp0 = _gl_load_pe(KV, tp0, offs_dp, KV_STRIDE, D_V, WITHIN_2GB)
        kp1 = _gl_load_pe(KV, tp1, offs_dp, KV_STRIDE, D_V, WITHIN_2GB)
        if ASYNC_NOPE:
            _gl_issue_nope(bufs_n.index(0), KV, tn0, offs_dn, KV_STRIDE, WITHIN_2GB)
            gl.amd.cdna4.async_copy.commit_group()
            _gl_issue_nope(bufs_n.index(1), KV, tn1, offs_dn, KV_STRIDE, WITHIN_2GB)
            gl.amd.cdna4.async_copy.commit_group()
        else:
            bufs_n.index(0).store(_gl_load_pe(KV, tn0, offs_dn, KV_STRIDE, 0, WITHIN_2GB))
            bufs_n.index(1).store(_gl_load_pe(KV, tn1, offs_dn, KV_STRIDE, 0, WITHIN_2GB))

        # ---------------- Q: per-row FP8 quantization (amax over all 576 dims)
        qp = qp.to(gl.float32)
        amax = gl.convert_layout(gl.max(gl.abs(qp), axis=1), gl.SliceLayout(1, ld_q))
        amax = gl.maximum(amax, gl.max(gl.abs(qc0.to(gl.float32)), axis=1))
        amax = gl.maximum(amax, gl.max(gl.abs(qc1.to(gl.float32)), axis=1))
        amax = gl.maximum(amax, gl.max(gl.abs(qc2.to(gl.float32)), axis=1))
        amax = gl.maximum(amax, gl.max(gl.abs(qc3.to(gl.float32)), axis=1))
        amax = gl.maximum(amax, 1e-20)
        q_inv = 448.0 / amax
        sq = gl.allocate_shared_memory(gl.float8e4nv, [BLOCK_M, DC], sm_q)
        sq.store((qc0.to(gl.float32) * q_inv[:, None]).to(gl.float8e4nv))
        q0 = sq.load(dot_a)
        sq.store((qc1.to(gl.float32) * q_inv[:, None]).to(gl.float8e4nv))
        q1 = sq.load(dot_a)
        sq.store((qc2.to(gl.float32) * q_inv[:, None]).to(gl.float8e4nv))
        q2 = sq.load(dot_a)
        sq.store((qc3.to(gl.float32) * q_inv[:, None]).to(gl.float8e4nv))
        q3 = sq.load(dot_a)
        q_inv_p = gl.convert_layout(q_inv, gl.SliceLayout(1, ld_qpe))
        sqp = gl.allocate_shared_memory(gl.float8e4nv, [BLOCK_M, D_PE], sm_qpe,
                                        (qp * q_inv_p[:, None]).to(gl.float8e4nv))
        qp8 = sqp.load(dot_a)
        bufs_p.index(0).store(kp0)
        bufs_p.index(1).store(kp1)

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

        # Software pipeline: iteration i issues QK(i+1) (async MFMAs) before
        # the softmax VALU work of tile i, then PV(i). Slots: tile i (PV),
        # tile i+1 (QK) resident, tile i+2 in flight.
        gl.static_assert(S == 3)
        gl.amd.cdna4.async_copy.wait_group(1)
        s_next = _gl_qk(bufs_n.index(0), bufs_p.index(0), q0, q1, q2, q3, qp8, mma, dot_b, DC, BLOCK_M, BLOCK_N)
        for i in range(num_iter):
            slot = i % 3
            n0 = start + i * BLOCK_N
            gl.amd.cdna4.async_copy.wait_group(0)
            iss_slot = (i + 2) % 3
            tok_n2 = _gl_load_idx(idx_base, n0 + 3 * BLOCK_N, end, offs_tn)
            tok_p2 = _gl_load_idx(idx_base, n0 + 3 * BLOCK_N, end, offs_tp)
            kp = _gl_load_pe(KV, tok_p, offs_dp, KV_STRIDE, D_V, WITHIN_2GB)
            # unconditional (tail tiles read slot 0): a data-dependent group
            # size makes the async wait-count pass fall back to vmcnt(0)
            if ASYNC_NOPE:
                _gl_issue_nope(bufs_n.index(iss_slot), KV, tok_n, offs_dn, KV_STRIDE, WITHIN_2GB)
                gl.amd.cdna4.async_copy.commit_group()
            else:
                kn = _gl_load_pe(KV, tok_n, offs_dn, KV_STRIDE, 0, WITHIN_2GB)
            tok_n = tok_n2
            tok_p = tok_p2

            s = s_next * s_scale[:, None]
            nslot = (i + 1) % 3
            s_next = _gl_qk(bufs_n.index(nslot), bufs_p.index(nslot), q0, q1, q2, q3, qp8, mma, dot_b, DC,
                            BLOCK_M, BLOCK_N)
            if n0 + BLOCK_N > causal_from:
                offs_n = n0 + offs_s
                vis = (offs_n[None, :] <= row_lim[:, None]) & (offs_n < end)[None, :]
                s = gl.where(vis, s, float("-inf"))
            m_new = gl.maximum(m_i, gl.max(s, axis=1))
            if LAZY_TAU > 0.0:
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
            else:
                alpha = gl.exp2(m_i - m_new)
                p = gl.exp2(s - m_new[:, None])
                l_i = l_i * alpha + gl.sum(p, axis=1)
                acc0 = acc0 * alpha[:, None]
                acc1 = acc1 * alpha[:, None]
                acc2 = acc2 * alpha[:, None]
                acc3 = acc3 * alpha[:, None]
                m_i = m_new
            p8 = gl.convert_layout((p * P_SCALE).to(gl.float8e4nv), dot_a)
            cur_n = bufs_n.index(slot)
            v = gl.amd.cdna4.async_copy.load_shared_relaxed(cur_n.slice(0 * DC, DC, dim=1), dot_b)
            acc0 = gl.amd.cdna4.mfma_scaled(p8, None, "e4m3", v, None, "e4m3", acc0)
            v = gl.amd.cdna4.async_copy.load_shared_relaxed(cur_n.slice(1 * DC, DC, dim=1), dot_b)
            acc1 = gl.amd.cdna4.mfma_scaled(p8, None, "e4m3", v, None, "e4m3", acc1)
            v = gl.amd.cdna4.async_copy.load_shared_relaxed(cur_n.slice(2 * DC, DC, dim=1), dot_b)
            acc2 = gl.amd.cdna4.mfma_scaled(p8, None, "e4m3", v, None, "e4m3", acc2)
            v = gl.amd.cdna4.async_copy.load_shared_relaxed(cur_n.slice(3 * DC, DC, dim=1), dot_b)
            acc3 = gl.amd.cdna4.mfma_scaled(p8, None, "e4m3", v, None, "e4m3", acc3)
            bufs_p.index(iss_slot).store(kp)
            if not ASYNC_NOPE:
                bufs_n.index(iss_slot).store(kn)
        gl.amd.cdna4.async_copy.wait_group(0)

        l_safe = gl.where(l_i > 0.0, l_i, 1.0)
        o_mul = kv_scale / (l_safe * P_SCALE)
        # epilogue: row-contiguous 16 B stores (layout change through LDS)
        st: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [NUM_WARPS, 1], [1, 0])
        rows_o = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, st))
        dv = gl.arange(0, DC, layout=gl.SliceLayout(0, st))
        rmask_o = rows_o < ROWS
        if NSPLIT == 1:
            o_ptr = O_FINAL
            o_off = ((b * QLEN + rows_o // H) * stride_o_tok + (rows_o % H) * stride_o_h)[:, None] + dv[None, :]
        else:
            rows_l = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mma))
            lse = gl.where(l_i > 0.0, m_i + gl.log2(l_safe), float("-inf"))
            gl.amd.cdna4.buffer_store(lse, LSE_PART, (b * NSPLIT + split) * ROWS + rows_l, mask=rows_l < ROWS)
            o_ptr = O_PART
            o_off = (((b * NSPLIT + split) * ROWS + rows_o) * D_V)[:, None] + dv[None, :]
        ety = o_ptr.dtype.element_ty
        om = rmask_o[:, None] & (dv < DC)[None, :]
        o_mul2 = o_mul[:, None]
        gl.amd.cdna4.buffer_store(gl.convert_layout((acc0 * o_mul2).to(ety), st), o_ptr, o_off + 0 * DC, mask=om)
        gl.amd.cdna4.buffer_store(gl.convert_layout((acc1 * o_mul2).to(ety), st), o_ptr, o_off + 1 * DC, mask=om)
        gl.amd.cdna4.buffer_store(gl.convert_layout((acc2 * o_mul2).to(ety), st), o_ptr, o_off + 2 * DC, mask=om)
        gl.amd.cdna4.buffer_store(gl.convert_layout((acc3 * o_mul2).to(ety), st), o_ptr, o_off + 3 * DC, mask=om)
