"""K3 MLA target-verify (QLEN=8, 12 heads, FP8 KV latent 576) with an on-device
regime switch between the bf16 Gluon kernel (aiter mla_gluon bh16bn128, best at
short context / small batch) and k3_mla_verify_v2 (read-KV-once FP8, best at
long context or larger batches). Graph capturable: the grid depends only on the
batch size; the regime is decided at replay time from kv_indptr.

Default (fused=True), 2 launches like either native path:
  1. _k3_auto_fwd     1-D grid of max(#gluon programs, #v2 programs). Every
                      program loads kv_indptr[0..bs], computes
                      eff = bs * max_b(L_b) and picks v2 iff eff >= thresh(bs);
                      it then runs _v2_body (v2 stage 1, MIN_CHUNK 128) or
                      _gluon_body (aiter _mla_gluon @ 19316974 with the program
                      ids passed in, same linear program order) or exits.
                      Program 0 writes the regime flag.
  2. _k3_auto_reduce  reads the flag; v2: base-2 LSE merge of the v2 splits
                      (= _k3_mla_verify_reduce); Gluon: natural-log merge of the
                      Gluon splits (= _mla_softmax_reducev_kernel), a no-op when
                      Gluon ran with a single split (direct store, bs >= 32).

fused=False keeps the earlier 3-launch variant (_mla_gluon_sel = Gluon with an
early-exit prologue that publishes an all-zero kv_indptr for the unchanged
v2/v3 stage 1 in the Gluon regime); it costs one extra launch (~2us).

thresh(bs) comes from graph-replayed crossovers of the two bodies (both kernels
split each request into ~256/bs programs, so both costs track bs * max_L).
Override with SGLANG_ROCM_K3_MLA_VERIFY_AUTO_THRESH (one int for all bs,
0 = always v2, -1 = always Gluon) or the thresh= argument.
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.k3_mla_verify_v2 import (
    _HAS_GLUON,
    _LOG2E,
    _MIN_CHUNK as _V2_MIN_CHUNK,
    _LAZY_TAU as _V2_LAZY_TAU,
    _ones,
    _split_chunk,
    default_num_splits,
)

# gfx950-only module, imported lazily by the backend.
from triton.experimental import gluon  # noqa: E402
from triton.experimental.gluon import language as gl  # noqa: E402

if _HAS_GLUON:
    from sglang.kernels.ops.attention.k3_mla_verify_v2 import (  # noqa: F401  (used by _v2_body)
        _gl_issue_nope,
        _gl_load_idx,
        _gl_load_pe,
        _gl_qk,
        _gl_split_chunk,
        _k3_mla_verify_fwd_gluon,
    )
    from sglang.kernels.ops.attention.k3_mla_verify_v3 import (
        _MIN_CHUNK as _V3_MIN_CHUNK,
        _k3_mla_verify_v3_fwd,
    )

_GLUON_BLOCK_H = 16
_GLUON_BLOCK_N = 128
_INT32_MAX = 2**31 - 1

# Crossover of gluon vs v2 in eff = bs * max_L tokens (v2 regime iff eff >= thresh).
# Measured crossovers of the fused kernel's two bodies (graph replay, MI355X,
# uniform batches, bench_auto.py sweep): bs1 ~36k, bs2 ~36k, bs4 ~35k,
# bs8 ~34k, bs16 ~38k tokens of bs * max_L.
_THRESH = {8: 35000, 16: 37000}
_THRESH_DEFAULT = 38000  # bs > 16 (extrapolated)
_W8_MAX_BS = 2  # fused=False only


def auto_threshold(bs: int) -> int:
    env = os.environ.get("SGLANG_ROCM_K3_MLA_VERIFY_AUTO_THRESH")
    if env is not None and env != "":
        v = int(env)
        return _INT32_MAX if v < 0 else v
    for k in sorted(_THRESH):
        if bs <= k:
            return _THRESH[k]
    return _THRESH_DEFAULT


# fmt: off
# Copy of aiter _mla_gluon (aiter/ops/triton/gluon/mla_gluon.py @ 19316974), unchanged
# except for the 'k3 auto' signature tail and prologue block.
@gluon.jit
def _mla_gluon_sel(
    Q_nope,
    Q_pe,
    Kv_c_cache,
    K_pe_cache,
    Req_to_tokens,
    B_seq_len,
    O,
    Attn_sink,
    sm_scale,
    kv_scale,
    stride_q_nope_bs,
    stride_q_nope_s,  # MTP: q_pos (qlen) stride; 0 when QLEN==1
    stride_q_nope_h,
    stride_q_pe_bs,
    stride_q_pe_s,  # MTP: q_pos (qlen) stride; 0 when QLEN==1
    stride_q_pe_h,
    stride_kv_c_bs,
    stride_k_pe_bs,
    stride_req_to_tokens_bs,
    stride_o_b,
    stride_o_s,  # MTP: q_pos (qlen) stride on O/logits; 0 when QLEN==1
    stride_o_h,
    stride_o_split,
    Mid_lse,  # split>1: per-split fp32 lse [B, QLEN, H, NUM_KV_SPLITS] (else None)
    stride_mid_lse_b,
    stride_mid_lse_s,  # MTP: q_pos stride; 0 when QLEN==1
    stride_mid_lse_h,
    stride_mid_lse_split,
    Final_lse,  # RETURN_LSE only: merged fp32 lse [B, QLEN, H] (else None)
    stride_final_lse_b,
    stride_final_lse_s,  # MTP: q_pos stride; 0 when QLEN==1
    stride_final_lse_h,
    BLOCK_H: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    HEAD_DIM_CKV: gl.constexpr,
    HEAD_DIM_KPE: gl.constexpr,
    KV_PE_OFFSET: gl.constexpr,
    USE_2D_VIEW: gl.constexpr,
    WITHIN_2GB: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NHEAD: gl.constexpr,
    REGIME: gl.constexpr,
    RETURN_LSE: gl.constexpr,
    QLEN: gl.constexpr,  # MTP query length; 1 for plain decode
    # --- dsv4-prefill knobs ---
    HAS_PE: gl.constexpr,
    HAS_ATTN_SINK: gl.constexpr,
    # --- k3 auto regime switch (added) ---
    Sel_buf,  # int32 [sel_bs + 2]: v2 kv_indptr (or zeros) + use_v2 flag
    sel_bs,
    sel_thresh,
    SEL_P2: gl.constexpr,
):
    # Grid mapping: bh64 uses 3-D XCD-aware multi-batch; bh16bn64 and bh16bn128
    # use 2-D (batch, split) — for batch_size=1 this is (1, NUM_KV_SPLITS).
    # MTP: an extra q_pos axis carries the query position within QLEN. bh64 packs
    # it into grid axis 1 (after the head-block index); bh16 uses grid axis 2.
    # When QLEN==1, q_pos is always 0 and the layout below is identical to before.
    if REGIME == 'bh64':
        NUM_M_BLOCKS: gl.constexpr = (NHEAD + BLOCK_H - 1) // BLOCK_H
        cur_batch = gl.program_id(0) + (gl.program_id(2) // NUM_KV_SPLITS) * NUM_XCDS
        cur_head_id = gl.program_id(1) % NUM_M_BLOCKS
        q_pos = gl.program_id(1) // NUM_M_BLOCKS
        split_kv_id = gl.program_id(2) % NUM_KV_SPLITS
    else:
        # bh16*: grid axis 2 carries (head_block, q_pos). For nhead <= 16 there is
        # a single head block (NUM_M_BLOCKS==1) so cur_head_id==0 and q_pos==pid(2),
        # identical to the original 2-D+qlen mapping. For nhead > 16 (e.g. 96) the
        # head range is tiled into NUM_M_BLOCKS = cdiv(NHEAD, BLOCK_H) blocks of 16.
        NUM_M_BLOCKS: gl.constexpr = (NHEAD + BLOCK_H - 1) // BLOCK_H
        cur_batch = gl.program_id(0)
        split_kv_id = gl.program_id(1)
        cur_head_id = gl.program_id(2) % NUM_M_BLOCKS
        q_pos = gl.program_id(2) // NUM_M_BLOCKS

    # USE_2D_VIEW=True: fixed len or max padded VarLen
    # Req_to_tokens = block_table[batch, max_seqlen], B_seq_len = cache_seqlens[batch]
    # USE_2D_VIEW=False: flattened VarLen
    # Req_to_tokens = kv_indices[total_kv],           B_seq_len = kv_indptr[batch+1]
    if USE_2D_VIEW:
        batch_page_start = stride_req_to_tokens_bs * cur_batch
        cur_batch_seq_len = gl.load(B_seq_len + cur_batch)
    else:
        batch_page_start = gl.load(B_seq_len + cur_batch)
        cur_batch_seq_len = gl.load(B_seq_len + cur_batch + 1) - batch_page_start

    # k3 auto: regime decision (identical integer formula in every program and
    # in the reduce). Program (0, 0, 0) publishes the v2 indptr (zeros when the
    # Gluon regime is picked, so every v2 program early-exits) and the flag.
    sel_lay: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    offs_sel = gl.arange(0, SEL_P2, layout=sel_lay)
    ip_lo = gl.load(B_seq_len + offs_sel, mask=offs_sel <= sel_bs, other=0)
    ip_hi = gl.load(B_seq_len + offs_sel + 1, mask=offs_sel < sel_bs, other=0)
    sel_max_len = gl.max(gl.where(offs_sel < sel_bs, ip_hi - ip_lo, 0), axis=0)
    use_v2 = sel_max_len * sel_bs >= sel_thresh
    if (gl.program_id(0) == 0) & (gl.program_id(1) == 0) & (gl.program_id(2) == 0):
        flag_v = gl.where(offs_sel == sel_bs + 1, use_v2.to(gl.int32), 0)
        sel_v = gl.where(offs_sel <= sel_bs, gl.where(use_v2, ip_lo, 0), flag_v)
        gl.store(Sel_buf + offs_sel, sel_v, mask=offs_sel < sel_bs + 2)
    if use_v2:
        return

    # NUM_KV_SPLITS is a launch-time budget only. 
    # the partition is derived here from the runtime per-batch KV length.
    # kv_len_per_split = max(BLOCK_N, floor(seq / NUM_KV_SPLITS)):
    #   - the BLOCK_N floor keeps every split at >= 1 full block, so a short seq
    #     is spread over fewer, whole-block splits instead of many partial ones;
    kv_len_per_split = gl.maximum(BLOCK_N, cur_batch_seq_len // NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = gl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    if split_kv_id == NUM_KV_SPLITS - 1:
        split_kv_end = cur_batch_seq_len
    # early return for inactive split
    if split_kv_start >= split_kv_end:
        return
    num_iter = gl.cdiv(split_kv_end - split_kv_start, BLOCK_N)
    start_n = split_kv_start

    # >2GB KV cache (global_load path): widen strides to int64 so kv offsets don't overflow int32.
    if not WITHIN_2GB:
        stride_kv_c_bs = stride_kv_c_bs.to(gl.int64)
        stride_k_pe_bs = stride_k_pe_bs.to(gl.int64)

    # MTP causal tail mask: query position q_pos may attend KV
    # [0, seq_len-QLEN+q_pos] only, so score_end is its per-program valid-score
    # bound. For QLEN==1 this equals split_kv_end, keeping the original code
    # path untouched.
    if QLEN > 1:
        score_end = gl.minimum(split_kv_end, cur_batch_seq_len - QLEN + q_pos + 1)
    else:
        score_end = split_kv_end

    ######### layout setting begin #########
    # Q-side layouts + mfma_layout: switch by BLOCK_H.
    # bh64 has BLOCK_H=64; bh16bn128 and bh16bn64 share BLOCK_H=16 (identical Q layouts + mfma orientation).
    if BLOCK_H == 64:
        # bh64: Q is [64, 512] / [64, 64]; warps tile M.
        blocked_q_nope: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[1, 64],
            warps_per_cta=[4, 1],
            order=[1, 0],
        )
        shared_q_nope: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
            cga_layout=[],
            shape=[64, 512]
        )
        blocked_q_pe: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4), (32, 0)),
            lane_bases=((0, 8), (0, 16), (0, 32), (4, 0), (8, 0), (16, 0)),
            warp_bases=((1, 0), (2, 0)),
            block_bases=[],
            shape=[64, 64],
        )
        shared_q_pe: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [4, 0], [8, 0], [16, 0], [1, 0], [2, 0], [32, 0]],
            cga_layout=[],
            shape=[64, 64]
        )
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=True,
            warps_per_cta=[4, 1],
        )
    else:
        # BLOCK_H == 16: shared by bh16bn128 and bh16bn64. Q is [16, 512] / [16, 64]; warps tile K.
        blocked_q_nope: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[1, 64],
            warps_per_cta=[4, 1],
            order=[1, 0],
        )
        shared_q_nope: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [1, 0], [2, 0], [4, 0], [8, 0]],
            cga_layout=[],
            shape=[16, 512]
        )
        blocked_q_pe: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4)),
            lane_bases=((0, 8), (0, 16), (0, 32), (1, 0), (2, 0), (4, 0)),
            warp_bases=((8, 0), (0, 0)),
            block_bases=[],
            shape=[16, 64],
        )
        shared_q_pe: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=8, order=[1, 0])
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=True,
            warps_per_cta=[1, 4],
        )

    # KV-side layouts: switch by BLOCK_N.
    # bh16bn128 (BLOCK_N=128, fp8 KV) needs distinct K layouts; bh64 and bh16bn64 share BLOCK_N=64 bf16 KV.
    if BLOCK_N == 128:
        # bh16bn128: K is [512, 128]fp8, KPE is [64, 128]fp8.
        blocked_kv: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 4), (0, 32), (0, 64)),
            lane_bases=((16, 0), (32, 0), (64, 0), (128, 0), (256, 0), (0, 16)),
            warp_bases=((0, 1), (0, 2)),
            block_bases=[],
            shape=[512, 128],
        )
        shared_kv: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[1024, 32], [8192, 16]],
            offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0], [0, 16], [0, 1], [0, 2], [0, 8], [0, 4], [0, 32], [0, 64]],
            cga_layout=[],
            shape=[512, 128]
        )
        blocked_kpe: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 2)),
            lane_bases=((16, 0), (32, 0), (0, 4), (0, 8), (0, 16), (0, 32)),
            warp_bases=((0, 64), (0, 1)),
            block_bases=[],
            shape=[64, 128],
        )
        shared_kpe: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[2048, 16]],
            offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 1], [0, 2]],
            cga_layout=[],
            shape=[64, 128]
        )
        blocked_page: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0,),),
            lane_bases=((1,), (2,), (4,), (8,), (16,), (32,)),
            warp_bases=((64,), (0,)),
            block_bases=[],
            shape=[128],
        )
        blocked_kv_slice: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 4), (0, 32)),
            lane_bases=((16, 0), (32, 0), (64, 0), (128, 0), (256, 0), (0, 16)),
            warp_bases=((0, 1), (0, 2)),
            block_bases=[],
            shape=[512, 64],
        )
    else:
        # BLOCK_N == 64: shared by bh64 and bh16bn64 (both bf16 KV).
        # K is [512, 64]bf16, KPE is [64, 64]bf16.
        blocked_kv: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (0, 8), (0, 4), (0, 16), (0, 32)),
            lane_bases=((8, 0), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
            warp_bases=((0, 1), (0, 2)),
            block_bases=[],
            shape=[512, 64],
        )
        shared_kv: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0], [0, 1], [0, 2], [0, 8], [0, 4], [0, 16], [0, 32]],
            cga_layout=[],
            shape=[512, 64]
        )
        blocked_kpe: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (0, 32)),
            lane_bases=((8, 0), (16, 0), (32, 0), (0, 4), (0, 8), (0, 16)),
            warp_bases=((0, 1), (0, 2)),
            block_bases=[],
            shape=[64, 64],
        )
        shared_kpe: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16], [0, 1], [0, 2], [0, 32]],
            cga_layout=[],
            shape=[64, 64]
        )
        blocked_page: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0,),),
            lane_bases=((1,), (2,), (4,), (8,), (16,), (32,)),
            warp_bases=((0,), (0,)),
            block_bases=[],
            shape=[64],
        )
        blocked_kv_slice: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (0, 8), (0, 4), (0, 16)),
            lane_bases=((8, 0), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
            warp_bases=((0, 1), (0, 2)),
            block_bases=[],
            shape=[512, 32],
        )

    # linear_v: each regime has unique warp/reg mapping (bh64 has degenerate warp_bases,
    # bh16bn128 has an extra K reg base for the 128-wide K, bh16bn64 has the bh16 warp layout at 64-wide K).
    if REGIME == 'bh64':
        linear_v: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4), (0, 32), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
            lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
            warp_bases=((0, 0), (0, 0)),
            block_bases=[],
            shape=[512, 64],
        )
    elif REGIME == 'bh16bn128':
        linear_v: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4), (0, 32), (0, 64), (64, 0), (128, 0), (256, 0)),
            lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
            warp_bases=((16, 0), (32, 0)),
            block_bases=[],
            shape=[512, 128],
        )
    else:
        linear_v: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4), (0, 32), (64, 0), (128, 0), (256, 0)),
            lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
            warp_bases=((16, 0), (32, 0)),
            block_bases=[],
            shape=[512, 64],
        )

    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=8)
    dtype = Q_nope.type.element_ty
    kvtype = Kv_c_cache.type.element_ty
    ######### layout setting end #########

    buf_q_nope = gl.allocate_shared_memory(dtype, shape=[BLOCK_H, HEAD_DIM_CKV], layout=shared_q_nope)
    if HAS_PE:
        buf_q_pe = gl.allocate_shared_memory(dtype, shape=[BLOCK_H, HEAD_DIM_KPE], layout=shared_q_pe)

    # load q_nope
    offs_d_ckv = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(0, blocked_q_nope))
    cur_head = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_nope))
    offs_q_nope = cur_batch * stride_q_nope_bs + q_pos * stride_q_nope_s + cur_head[:, None] * stride_q_nope_h + offs_d_ckv[None, :]
    ### For nhead < BLOCK_H, mask OOB heads to zero on Q load and skip OOB O stores; wasted MFMA lanes are free (memory-bound).
    gl.amd.cdna4.async_copy.buffer_load_to_shared(buf_q_nope, Q_nope, offs_q_nope, mask = (cur_head < NHEAD)[:, None] if NHEAD % BLOCK_H != 0 else None)
    gl.amd.cdna4.async_copy.commit_group()

    # load q_pe
    if HAS_PE:
        offs_d_kpe = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(0, blocked_q_pe))
        cur_head_qpe = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_pe))
        offs_q_pe = cur_batch * stride_q_pe_bs + q_pos * stride_q_pe_s + cur_head_qpe[:, None] * stride_q_pe_h + offs_d_kpe[None, :]
        gl.amd.cdna4.async_copy.buffer_load_to_shared(buf_q_pe, Q_pe, offs_q_pe, mask = (cur_head_qpe < NHEAD)[:, None] if NHEAD % BLOCK_H != 0 else None)
        gl.amd.cdna4.async_copy.commit_group()

    e_max = gl.zeros([BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout)) - float("inf")
    e_sum = gl.zeros([BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout))
    acc = gl.zeros([BLOCK_H, HEAD_DIM_CKV], dtype=gl.float32, layout=mfma_layout)

    # Fold KV dequant scale into the QK temperature. For fp8 KV the real
    # logits are (Q @ K_fp8^T) * kv_scale * sm_scale; softmax is shift- but
    # not scale-invariant, so kv_scale must affect qk (not just acc).
    # For bf16 KV the wrapper passes kv_scale=1.0, so this is a no-op.
    qk_scale = sm_scale * kv_scale

    ### bufs of page_number
    shared_page: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
    bufs_page = gl.allocate_shared_memory(gl.int32, shape=[2, BLOCK_N], layout=shared_page)
    gl.static_assert(PAGE_SIZE == 1)

    offs_page_raw = gl.arange(0, BLOCK_N, layout=blocked_page)

    ################ prologue
    #### global load page number
    offs_n_page = start_n + offs_page_raw
    offs_page = batch_page_start + offs_n_page // PAGE_SIZE
    gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_page.index(0), Req_to_tokens, offs_page, offs_n_page < split_kv_end)
    gl.amd.cdna4.async_copy.commit_group()

    start_n += BLOCK_N
    #### global load page number
    offs_n_page = start_n + offs_page_raw
    offs_page = batch_page_start + offs_n_page // PAGE_SIZE
    gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_page.index(1), Req_to_tokens, offs_page, offs_n_page < split_kv_end)
    gl.amd.cdna4.async_copy.commit_group()

    #### local load Q
    gl.amd.cdna4.async_copy.wait_group(2)
    q_nope = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_q_nope, mfma_layout_a)
    if HAS_PE:
        q_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_q_pe, mfma_layout_a)

    #################### move here to work around allocate_shared_memory bug
    bufs_kv = gl.allocate_shared_memory(kvtype, shape=[2, HEAD_DIM_CKV, BLOCK_N], layout=shared_kv)
    if HAS_PE:
        bufs_kpe = gl.allocate_shared_memory(kvtype, shape=[2, HEAD_DIM_KPE, BLOCK_N], layout=shared_kpe)

    #### global load K
    # local load page number
    gl.amd.cdna4.async_copy.wait_group(1)
    if HAS_PE:
        kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page.index(0), gl.SliceLayout(0, blocked_kpe))
        # simplify for page_size 1
        kv_loc_pe = kv_page_number_pe

    # local load page number for slice 0
    bufs_page_0 = bufs_page.index(0).slice(0, BLOCK_N // 2, 0)
    kv_page_number_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_0, gl.SliceLayout(0, blocked_kv_slice))
    kv_loc0 = kv_page_number_0

    # global load K_nope slice 0
    offs_n_nope0 = split_kv_start + gl.arange(0, BLOCK_N // 2, layout=gl.SliceLayout(0, blocked_kv_slice))
    offs_d_ckv_10 = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv_slice))
    offs_k_c0 = kv_loc0[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
    bufs_kv0 = bufs_kv.index(0).slice(0, BLOCK_N // 2, 1)
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv0, Kv_c_cache, offs_k_c0, mask=offs_n_nope0[None, :] < split_kv_end)
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv0, Kv_c_cache + offs_k_c0, mask=offs_n_nope0[None, :] < split_kv_end, other=0.0)
    gl.amd.cdna4.async_copy.commit_group()

    # global load K_pe
    if HAS_PE:
        offs_n_pe0 = split_kv_start + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kpe))
        offs_d_kpe_1 = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(1, blocked_kpe))
        offs_k_pe = kv_loc_pe[None, :] * stride_k_pe_bs + offs_d_kpe_1[:, None] + KV_PE_OFFSET
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kpe.index(0), K_pe_cache, offs_k_pe, mask=offs_n_pe0[None, :] < split_kv_end)
        else:
            gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kpe.index(0), K_pe_cache + offs_k_pe, mask=offs_n_pe0[None, :] < split_kv_end, other=0.0)
        gl.amd.cdna4.async_copy.commit_group()

    # local load page number for slice 1
    bufs_page_1 = bufs_page.index(0).slice(BLOCK_N // 2, BLOCK_N // 2, 0)
    kv_page_number_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_1, gl.SliceLayout(0, blocked_kv_slice))
    kv_loc1 = kv_page_number_1

    # global load K_nope slice 1
    offs_n_nope1 = offs_n_nope0 + BLOCK_N // 2
    bufs_kv1 = bufs_kv.index(0).slice(BLOCK_N // 2, BLOCK_N // 2, 1)
    offs_k_c1 = kv_loc1[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv1, Kv_c_cache, offs_k_c1, mask=offs_n_nope1[None, :] < split_kv_end)
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv1, Kv_c_cache + offs_k_c1, mask=offs_n_nope1[None, :] < split_kv_end, other=0.0)
    gl.amd.cdna4.async_copy.commit_group()

    if REGIME == 'bh64':
        # bh64 guarantees >= 3 iters/split; this constant-folds the
        # `if num_iter >= 2` epilogue-1 guard below so its codegen is unchanged.
        gl.assume(num_iter >= 3)
    buf_idx = 0
    ################ loop
    for i in range(num_iter - 2):
        async_idx = (buf_idx + 1) % 2

        gl.amd.cdna4.async_copy.wait_group(0)
        #### global load page number
        offs_n_page = start_n + BLOCK_N + offs_page_raw
        offs_page = batch_page_start + offs_n_page // PAGE_SIZE
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_page.index(buf_idx), Req_to_tokens, offs_page, offs_n_page < split_kv_end)
        gl.amd.cdna4.async_copy.commit_group()

        #### global load K
        bufs_kv0 = bufs_kv.index(async_idx).slice(0, BLOCK_N // 2, 1)
        bufs_kv1 = bufs_kv.index(async_idx).slice(BLOCK_N // 2, BLOCK_N // 2, 1)
        # local load page number for slice 0
        bufs_page_0 = bufs_page.index(async_idx).slice(0, BLOCK_N // 2, 0)
        kv_page_number_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_0, gl.SliceLayout(0, blocked_kv_slice))
        kv_loc0 = kv_page_number_0
        # global load K_nope slice 0
        offs_n_nope0 = start_n + gl.arange(0, BLOCK_N // 2, layout=gl.SliceLayout(0, blocked_kv_slice))
        offs_d_ckv_10 = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv_slice))
        offs_k_c0 = kv_loc0[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv0, Kv_c_cache, offs_k_c0, mask=offs_n_nope0[None, :] < split_kv_end)
        else:
            # >2GB path needs the same bounds mask + other=0.0 as buffer_load.
            gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv0, Kv_c_cache + offs_k_c0, mask=offs_n_nope0[None, :] < split_kv_end, other=0.0)
        gl.amd.cdna4.async_copy.commit_group()

        # local load page_number_pe
        if HAS_PE:
            kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page.index(async_idx), gl.SliceLayout(0, blocked_kpe))
            kv_loc_pe = kv_page_number_pe
            # global load K_pe
            offs_n_pe = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kpe))
            offs_d_kpe_1 = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(1, blocked_kpe))
            offs_k_pe = kv_loc_pe[None, :] * stride_k_pe_bs + offs_d_kpe_1[:, None] + KV_PE_OFFSET
            if WITHIN_2GB:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kpe.index(async_idx), K_pe_cache, offs_k_pe, mask=offs_n_pe[None, :] < split_kv_end)
            else:
                # >2GB path needs the same bounds mask + other=0.0 as buffer_load.
                gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kpe.index(async_idx), K_pe_cache + offs_k_pe, mask=offs_n_pe[None, :] < split_kv_end, other=0.0)
            gl.amd.cdna4.async_copy.commit_group()

        #### dot, softmax, dot (part0)
        k_c = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_kv.index(buf_idx), mfma_layout_b)
        zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
        qk = gl.amd.cdna4.mfma(q_nope, k_c.to(dtype), zeros)
        if HAS_PE:
            k_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_kpe.index(buf_idx), mfma_layout_b)
            qk = gl.amd.cdna4.mfma(q_pe, k_pe.to(dtype), qk)

        # local load page number for slice 1
        bufs_page_1 = bufs_page.index(async_idx).slice(BLOCK_N // 2, BLOCK_N // 2, 0)
        kv_page_number_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_1, gl.SliceLayout(0, blocked_kv_slice))
        kv_loc1 = kv_page_number_1
        # global load K_nope slice 1
        offs_n1 = offs_n_nope0 + BLOCK_N // 2
        offs_k_c1 = kv_loc1[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv1, Kv_c_cache, offs_k_c1, mask=offs_n1[None, :] < split_kv_end)
        else:
            # >2GB path needs the same bounds mask + other=0.0 as buffer_load.
            gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv1, Kv_c_cache + offs_k_c1, mask=offs_n1[None, :] < split_kv_end, other=0.0)
        gl.amd.cdna4.async_copy.commit_group()

        #### dot, softmax, dot (part1)
        qk *= qk_scale
        offs_n_qk = split_kv_start + i * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
        qk = gl.where(offs_n_qk[None, :] < score_end, qk, float("-inf"))
        n_e_max = gl.maximum(gl.max(qk, 1), e_max)
        LOG2E: gl.constexpr = 1.4426950408889634
        re_scale = gl.exp2((e_max - n_e_max) * LOG2E)
        p = gl.exp2((qk - n_e_max[:, None]) * LOG2E)
        if QLEN > 1:
            # MTP: a leading/whole fully-masked split keeps e_max=n_e_max=-inf,
            # making re_scale/p NaN. Force them to 0 so the split cleanly yields
            # e_sum=0 -> lse=-inf, which stage-2 drops.
            re_scale = gl.where(e_max == float("-inf"), 0.0, re_scale)
            p = gl.where(n_e_max[:, None] == float("-inf"), 0.0, p)
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        e_max = n_e_max
        p = p.to(dtype)
        p = gl.convert_layout(p, mfma_layout_a)
        acc *= re_scale[:, None]
        v_c = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_kv.index(buf_idx), linear_v)
        v_c = v_c.to(dtype)
        v_c = gl.permute(v_c, [1, 0])
        v_c = gl.convert_layout(v_c, mfma_layout_b)
        acc = gl.amd.cdna4.mfma(p, v_c, acc)

        start_n += BLOCK_N
        buf_idx = (buf_idx + 1) % 2

    LOG2E: gl.constexpr = 1.4426950408889634

    ################ epilogue 1
    # Skip when num_iter < 2 (possible for bh16bn64 / bh16bn128 in either mode).
    # bh64 has gl.assume(num_iter >= 3) above so the compiler folds this branch
    # out there; for the bh16 regimes it stays a runtime branch.
    if num_iter >= 2:
        async_idx = (buf_idx + 1) % 2

        #### global load K
        # local load page number
        gl.amd.cdna4.async_copy.wait_group(3 if HAS_PE else 2)
        kv_page_number = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page.index(async_idx), gl.SliceLayout(0, blocked_kv))
        kv_loc = kv_page_number
        if HAS_PE:
            kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page.index(async_idx), gl.SliceLayout(0, blocked_kpe))
            kv_loc_pe = kv_page_number_pe
        # global load K_nope
        offs_n_nope = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kv))
        offs_d_ckv_1 = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv))
        offs_k_c = kv_loc[None, :] * stride_kv_c_bs + offs_d_ckv_1[:, None]
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv.index(async_idx), Kv_c_cache, offs_k_c, mask=offs_n_nope[None, :] < split_kv_end)
        else:
            # >2GB path needs the same bounds mask + other=0.0 as buffer_load.
            gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv.index(async_idx), Kv_c_cache + offs_k_c, mask=offs_n_nope[None, :] < split_kv_end, other=0.0)
        gl.amd.cdna4.async_copy.commit_group()
        # global load K_pe
        if HAS_PE:
            offs_n_pe = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kpe))
            offs_d_kpe_1 = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(1, blocked_kpe))
            offs_k_pe = kv_loc_pe[None, :] * stride_k_pe_bs + offs_d_kpe_1[:, None] + KV_PE_OFFSET
            if WITHIN_2GB:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kpe.index(async_idx), K_pe_cache, offs_k_pe, mask=offs_n_pe[None, :] < split_kv_end)
            else:
                gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kpe.index(async_idx), K_pe_cache + offs_k_pe, mask=offs_n_pe[None, :] < split_kv_end, other=0.0)
            gl.amd.cdna4.async_copy.commit_group()

        # dot, softmax, dot
        gl.amd.cdna4.async_copy.wait_group(2 if HAS_PE else 1)
        k_c = bufs_kv.index(buf_idx).load(layout=mfma_layout_b)
        zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
        qk = gl.amd.cdna4.mfma(q_nope, k_c.to(dtype), zeros)

        if HAS_PE:
            k_pe = bufs_kpe.index(buf_idx).load(layout=mfma_layout_b)
            qk = gl.amd.cdna4.mfma(q_pe, k_pe.to(dtype), qk)
        qk *= qk_scale
        offs_n_qk = split_kv_start + (num_iter - 2) * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
        qk = gl.where(offs_n_qk[None, :] < score_end, qk, float("-inf"))
        n_e_max = gl.maximum(gl.max(qk, 1), e_max)
        re_scale = gl.exp2((e_max - n_e_max) * LOG2E)
        p = gl.exp2((qk - n_e_max[:, None]) * LOG2E)
        if QLEN > 1:
            re_scale = gl.where(e_max == float("-inf"), 0.0, re_scale)
            p = gl.where(n_e_max[:, None] == float("-inf"), 0.0, p)
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        e_max = n_e_max
        p = p.to(dtype)
        p = gl.convert_layout(p, mfma_layout_a)
        acc *= re_scale[:, None]
        v_c = bufs_kv.index(buf_idx).load(layout=linear_v)
        v_c = v_c.to(dtype)
        v_c = gl.permute(v_c, [1, 0])
        v_c = gl.convert_layout(v_c, mfma_layout_b)
        acc = gl.amd.cdna4.mfma(p, v_c, acc)

        start_n += BLOCK_N
        buf_idx = (buf_idx + 1) % 2

    ################ epilogue 2
    #### dot, softmax, dot
    gl.amd.cdna4.async_copy.wait_group(0)
    k_c = bufs_kv.index(buf_idx).load(layout=mfma_layout_b)
    zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
    qk = gl.amd.cdna4.mfma(q_nope, k_c.to(dtype), zeros)

    if HAS_PE:
        k_pe = bufs_kpe.index(buf_idx).load(layout=mfma_layout_b)
        qk = gl.amd.cdna4.mfma(q_pe, k_pe.to(dtype), qk)
    qk *= qk_scale
    offs_n_qk = split_kv_start + (num_iter - 1) * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
    qk = gl.where(offs_n_qk[None, :] < score_end, qk, float("-inf"))
    n_e_max = gl.maximum(gl.max(qk, 1), e_max)
    re_scale = gl.exp2((e_max - n_e_max) * LOG2E)
    p = gl.exp2((qk - n_e_max[:, None]) * LOG2E)
    if QLEN > 1:
        re_scale = gl.where(e_max == float("-inf"), 0.0, re_scale)
        p = gl.where(n_e_max[:, None] == float("-inf"), 0.0, p)
    e_sum = e_sum * re_scale + gl.sum(p, 1)
    e_max = n_e_max
    p = p.to(dtype)
    p = gl.convert_layout(p, mfma_layout_a)
    acc *= re_scale[:, None]
    v_c = bufs_kv.index(buf_idx).load(layout=linear_v)
    v_c = v_c.to(dtype)
    v_c = gl.permute(v_c, [1, 0])
    v_c = gl.convert_layout(v_c, mfma_layout_b)
    acc = gl.amd.cdna4.mfma(p, v_c, acc)

    cur_head_o = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_layout))
    offs_d_ckv_o = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(0, mfma_layout))
    offs_o = cur_batch * stride_o_b + q_pos * stride_o_s + cur_head_o[:, None] * stride_o_h + split_kv_id * stride_o_split + offs_d_ckv_o[None, :]

    if HAS_ATTN_SINK:
        # Fold the optional per-head sink into the softmax denom (no V contribution).
        # e_max/e_sum are natural-log units (the *LOG2E is inside exp2), so is sink.
        if NHEAD % BLOCK_H != 0:
            sink = gl.load(Attn_sink + cur_head_o, mask=cur_head_o < NHEAD, other=float("-inf")).to(gl.float32)
        else:
            sink = gl.load(Attn_sink + cur_head_o).to(gl.float32)
        n_e_max = gl.maximum(e_max, sink)
        re_scale = gl.exp2((e_max - n_e_max) * LOG2E)
        acc *= re_scale[:, None]
        e_sum = e_sum * re_scale + gl.exp2((sink - n_e_max) * LOG2E)
        e_max = n_e_max

    acc *= kv_scale
    rcp = 1.0 / e_sum
    stored_value = (acc * rcp[:, None]).to(dtype)
    if NHEAD % BLOCK_H != 0:
        gl.amd.cdna4.buffer_store(stored_value, ptr=O, offsets=offs_o, mask=(cur_head_o < NHEAD)[:, None])
    else:
        gl.amd.cdna4.buffer_store(stored_value, ptr=O, offsets=offs_o)

    ### store lse
    blocked_lse: gl.constexpr = gl.BlockedLayout(size_per_thread=[1], threads_per_warp=[64], warps_per_cta=[4], order=[0])
    cur_head_lse = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=blocked_lse)
    if RETURN_LSE and NUM_KV_SPLITS == 1:
        # split==1: single split is the whole sequence, so its lse is the final lse.
        offs_final_lse = cur_batch * stride_final_lse_b + q_pos * stride_final_lse_s + cur_head_lse * stride_final_lse_h
        lse = e_max + gl.log(e_sum)
        lse = gl.convert_layout(lse, blocked_lse)
        if NHEAD % BLOCK_H != 0:
            gl.amd.cdna4.buffer_store(lse, ptr=Final_lse, offsets=offs_final_lse, mask=(cur_head_lse < NHEAD))
        else:
            gl.amd.cdna4.buffer_store(lse, ptr=Final_lse, offsets=offs_final_lse)
    elif NUM_KV_SPLITS > 1:
        # per-split lse for stage-2 reduce.
        offs_mid_lse = cur_batch * stride_mid_lse_b + q_pos * stride_mid_lse_s + cur_head_lse * stride_mid_lse_h + split_kv_id * stride_mid_lse_split
        lse = e_max + gl.log(e_sum)
        lse = gl.convert_layout(lse, blocked_lse)
        if NHEAD % BLOCK_H != 0:
            gl.amd.cdna4.buffer_store(lse, ptr=Mid_lse, offsets=offs_mid_lse, mask=(cur_head_lse < NHEAD))
        else:
            gl.amd.cdna4.buffer_store(lse, ptr=Mid_lse, offsets=offs_mid_lse)
# fmt: on


# ============================================================================
# Fused single-launch stage 1: both bodies as inlined gluon functions.
# _gluon_body: aiter _mla_gluon (@ 19316974) with program ids as arguments.
# _v2_body:    k3_mla_verify_v2._k3_mla_verify_fwd_gluon with (split, b) as arguments.
# ============================================================================
# fmt: off
@gluon.jit
def _gluon_body(
    pid0,
    pid1,
    pid2,
    Q_nope,
    Q_pe,
    Kv_c_cache,
    K_pe_cache,
    Req_to_tokens,
    B_seq_len,
    O,
    Attn_sink,
    sm_scale,
    kv_scale,
    stride_q_nope_bs,
    stride_q_nope_s,  # MTP: q_pos (qlen) stride; 0 when QLEN==1
    stride_q_nope_h,
    stride_q_pe_bs,
    stride_q_pe_s,  # MTP: q_pos (qlen) stride; 0 when QLEN==1
    stride_q_pe_h,
    stride_kv_c_bs,
    stride_k_pe_bs,
    stride_req_to_tokens_bs,
    stride_o_b,
    stride_o_s,  # MTP: q_pos (qlen) stride on O/logits; 0 when QLEN==1
    stride_o_h,
    stride_o_split,
    Mid_lse,  # split>1: per-split fp32 lse [B, QLEN, H, NUM_KV_SPLITS] (else None)
    stride_mid_lse_b,
    stride_mid_lse_s,  # MTP: q_pos stride; 0 when QLEN==1
    stride_mid_lse_h,
    stride_mid_lse_split,
    Final_lse,  # RETURN_LSE only: merged fp32 lse [B, QLEN, H] (else None)
    stride_final_lse_b,
    stride_final_lse_s,  # MTP: q_pos stride; 0 when QLEN==1
    stride_final_lse_h,
    BLOCK_H: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    HEAD_DIM_CKV: gl.constexpr,
    HEAD_DIM_KPE: gl.constexpr,
    KV_PE_OFFSET: gl.constexpr,
    USE_2D_VIEW: gl.constexpr,
    WITHIN_2GB: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NHEAD: gl.constexpr,
    REGIME: gl.constexpr,
    RETURN_LSE: gl.constexpr,
    QLEN: gl.constexpr,  # MTP query length; 1 for plain decode
    # --- dsv4-prefill knobs ---
    HAS_PE: gl.constexpr,
    HAS_ATTN_SINK: gl.constexpr,
):
    # Grid mapping: bh64 uses 3-D XCD-aware multi-batch; bh16bn64 and bh16bn128
    # use 2-D (batch, split) — for batch_size=1 this is (1, NUM_KV_SPLITS).
    # MTP: an extra q_pos axis carries the query position within QLEN. bh64 packs
    # it into grid axis 1 (after the head-block index); bh16 uses grid axis 2.
    # When QLEN==1, q_pos is always 0 and the layout below is identical to before.
    if REGIME == 'bh64':
        NUM_M_BLOCKS: gl.constexpr = (NHEAD + BLOCK_H - 1) // BLOCK_H
        cur_batch = pid0 + (pid2 // NUM_KV_SPLITS) * NUM_XCDS
        cur_head_id = pid1 % NUM_M_BLOCKS
        q_pos = pid1 // NUM_M_BLOCKS
        split_kv_id = pid2 % NUM_KV_SPLITS
    else:
        # bh16*: grid axis 2 carries (head_block, q_pos). For nhead <= 16 there is
        # a single head block (NUM_M_BLOCKS==1) so cur_head_id==0 and q_pos==pid(2),
        # identical to the original 2-D+qlen mapping. For nhead > 16 (e.g. 96) the
        # head range is tiled into NUM_M_BLOCKS = cdiv(NHEAD, BLOCK_H) blocks of 16.
        NUM_M_BLOCKS: gl.constexpr = (NHEAD + BLOCK_H - 1) // BLOCK_H
        cur_batch = pid0
        split_kv_id = pid1
        cur_head_id = pid2 % NUM_M_BLOCKS
        q_pos = pid2 // NUM_M_BLOCKS

    # USE_2D_VIEW=True: fixed len or max padded VarLen
    # Req_to_tokens = block_table[batch, max_seqlen], B_seq_len = cache_seqlens[batch]
    # USE_2D_VIEW=False: flattened VarLen
    # Req_to_tokens = kv_indices[total_kv],           B_seq_len = kv_indptr[batch+1]
    if USE_2D_VIEW:
        batch_page_start = stride_req_to_tokens_bs * cur_batch
        cur_batch_seq_len = gl.load(B_seq_len + cur_batch)
    else:
        batch_page_start = gl.load(B_seq_len + cur_batch)
        cur_batch_seq_len = gl.load(B_seq_len + cur_batch + 1) - batch_page_start

    # NUM_KV_SPLITS is a launch-time budget only. 
    # the partition is derived here from the runtime per-batch KV length.
    # kv_len_per_split = max(BLOCK_N, floor(seq / NUM_KV_SPLITS)):
    #   - the BLOCK_N floor keeps every split at >= 1 full block, so a short seq
    #     is spread over fewer, whole-block splits instead of many partial ones;
    kv_len_per_split = gl.maximum(BLOCK_N, cur_batch_seq_len // NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = gl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    if split_kv_id == NUM_KV_SPLITS - 1:
        split_kv_end = cur_batch_seq_len
    # early return for inactive split
    if split_kv_start >= split_kv_end:
        return
    num_iter = gl.cdiv(split_kv_end - split_kv_start, BLOCK_N)
    start_n = split_kv_start

    # >2GB KV cache (global_load path): widen strides to int64 so kv offsets don't overflow int32.
    if not WITHIN_2GB:
        stride_kv_c_bs = stride_kv_c_bs.to(gl.int64)
        stride_k_pe_bs = stride_k_pe_bs.to(gl.int64)

    # MTP causal tail mask: query position q_pos may attend KV
    # [0, seq_len-QLEN+q_pos] only, so score_end is its per-program valid-score
    # bound. For QLEN==1 this equals split_kv_end, keeping the original code
    # path untouched.
    if QLEN > 1:
        score_end = gl.minimum(split_kv_end, cur_batch_seq_len - QLEN + q_pos + 1)
    else:
        score_end = split_kv_end

    ######### layout setting begin #########
    # Q-side layouts + mfma_layout: switch by BLOCK_H.
    # bh64 has BLOCK_H=64; bh16bn128 and bh16bn64 share BLOCK_H=16 (identical Q layouts + mfma orientation).
    if BLOCK_H == 64:
        # bh64: Q is [64, 512] / [64, 64]; warps tile M.
        blocked_q_nope: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[1, 64],
            warps_per_cta=[4, 1],
            order=[1, 0],
        )
        shared_q_nope: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
            cga_layout=[],
            shape=[64, 512]
        )
        blocked_q_pe: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4), (32, 0)),
            lane_bases=((0, 8), (0, 16), (0, 32), (4, 0), (8, 0), (16, 0)),
            warp_bases=((1, 0), (2, 0)),
            block_bases=[],
            shape=[64, 64],
        )
        shared_q_pe: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [4, 0], [8, 0], [16, 0], [1, 0], [2, 0], [32, 0]],
            cga_layout=[],
            shape=[64, 64]
        )
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=True,
            warps_per_cta=[4, 1],
        )
    else:
        # BLOCK_H == 16: shared by bh16bn128 and bh16bn64. Q is [16, 512] / [16, 64]; warps tile K.
        blocked_q_nope: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[1, 64],
            warps_per_cta=[4, 1],
            order=[1, 0],
        )
        shared_q_nope: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [1, 0], [2, 0], [4, 0], [8, 0]],
            cga_layout=[],
            shape=[16, 512]
        )
        blocked_q_pe: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4)),
            lane_bases=((0, 8), (0, 16), (0, 32), (1, 0), (2, 0), (4, 0)),
            warp_bases=((8, 0), (0, 0)),
            block_bases=[],
            shape=[16, 64],
        )
        shared_q_pe: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=8, order=[1, 0])
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=True,
            warps_per_cta=[1, 4],
        )

    # KV-side layouts: switch by BLOCK_N.
    # bh16bn128 (BLOCK_N=128, fp8 KV) needs distinct K layouts; bh64 and bh16bn64 share BLOCK_N=64 bf16 KV.
    if BLOCK_N == 128:
        # bh16bn128: K is [512, 128]fp8, KPE is [64, 128]fp8.
        blocked_kv: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 4), (0, 32), (0, 64)),
            lane_bases=((16, 0), (32, 0), (64, 0), (128, 0), (256, 0), (0, 16)),
            warp_bases=((0, 1), (0, 2)),
            block_bases=[],
            shape=[512, 128],
        )
        shared_kv: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[1024, 32], [8192, 16]],
            offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0], [0, 16], [0, 1], [0, 2], [0, 8], [0, 4], [0, 32], [0, 64]],
            cga_layout=[],
            shape=[512, 128]
        )
        blocked_kpe: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 2)),
            lane_bases=((16, 0), (32, 0), (0, 4), (0, 8), (0, 16), (0, 32)),
            warp_bases=((0, 64), (0, 1)),
            block_bases=[],
            shape=[64, 128],
        )
        shared_kpe: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[2048, 16]],
            offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 1], [0, 2]],
            cga_layout=[],
            shape=[64, 128]
        )
        blocked_page: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0,),),
            lane_bases=((1,), (2,), (4,), (8,), (16,), (32,)),
            warp_bases=((64,), (0,)),
            block_bases=[],
            shape=[128],
        )
        blocked_kv_slice: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 4), (0, 32)),
            lane_bases=((16, 0), (32, 0), (64, 0), (128, 0), (256, 0), (0, 16)),
            warp_bases=((0, 1), (0, 2)),
            block_bases=[],
            shape=[512, 64],
        )
    else:
        # BLOCK_N == 64: shared by bh64 and bh16bn64 (both bf16 KV).
        # K is [512, 64]bf16, KPE is [64, 64]bf16.
        blocked_kv: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (0, 8), (0, 4), (0, 16), (0, 32)),
            lane_bases=((8, 0), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
            warp_bases=((0, 1), (0, 2)),
            block_bases=[],
            shape=[512, 64],
        )
        shared_kv: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0], [0, 1], [0, 2], [0, 8], [0, 4], [0, 16], [0, 32]],
            cga_layout=[],
            shape=[512, 64]
        )
        blocked_kpe: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (0, 32)),
            lane_bases=((8, 0), (16, 0), (32, 0), (0, 4), (0, 8), (0, 16)),
            warp_bases=((0, 1), (0, 2)),
            block_bases=[],
            shape=[64, 64],
        )
        shared_kpe: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16], [0, 1], [0, 2], [0, 32]],
            cga_layout=[],
            shape=[64, 64]
        )
        blocked_page: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0,),),
            lane_bases=((1,), (2,), (4,), (8,), (16,), (32,)),
            warp_bases=((0,), (0,)),
            block_bases=[],
            shape=[64],
        )
        blocked_kv_slice: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (0, 8), (0, 4), (0, 16)),
            lane_bases=((8, 0), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
            warp_bases=((0, 1), (0, 2)),
            block_bases=[],
            shape=[512, 32],
        )

    # linear_v: each regime has unique warp/reg mapping (bh64 has degenerate warp_bases,
    # bh16bn128 has an extra K reg base for the 128-wide K, bh16bn64 has the bh16 warp layout at 64-wide K).
    if REGIME == 'bh64':
        linear_v: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4), (0, 32), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
            lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
            warp_bases=((0, 0), (0, 0)),
            block_bases=[],
            shape=[512, 64],
        )
    elif REGIME == 'bh16bn128':
        linear_v: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4), (0, 32), (0, 64), (64, 0), (128, 0), (256, 0)),
            lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
            warp_bases=((16, 0), (32, 0)),
            block_bases=[],
            shape=[512, 128],
        )
    else:
        linear_v: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4), (0, 32), (64, 0), (128, 0), (256, 0)),
            lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
            warp_bases=((16, 0), (32, 0)),
            block_bases=[],
            shape=[512, 64],
        )

    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=8)
    dtype = Q_nope.type.element_ty
    kvtype = Kv_c_cache.type.element_ty
    ######### layout setting end #########

    buf_q_nope = gl.allocate_shared_memory(dtype, shape=[BLOCK_H, HEAD_DIM_CKV], layout=shared_q_nope)
    if HAS_PE:
        buf_q_pe = gl.allocate_shared_memory(dtype, shape=[BLOCK_H, HEAD_DIM_KPE], layout=shared_q_pe)

    # load q_nope
    offs_d_ckv = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(0, blocked_q_nope))
    cur_head = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_nope))
    offs_q_nope = cur_batch * stride_q_nope_bs + q_pos * stride_q_nope_s + cur_head[:, None] * stride_q_nope_h + offs_d_ckv[None, :]
    ### For nhead < BLOCK_H, mask OOB heads to zero on Q load and skip OOB O stores; wasted MFMA lanes are free (memory-bound).
    gl.amd.cdna4.async_copy.buffer_load_to_shared(buf_q_nope, Q_nope, offs_q_nope, mask = (cur_head < NHEAD)[:, None] if NHEAD % BLOCK_H != 0 else None)
    gl.amd.cdna4.async_copy.commit_group()

    # load q_pe
    if HAS_PE:
        offs_d_kpe = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(0, blocked_q_pe))
        cur_head_qpe = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_pe))
        offs_q_pe = cur_batch * stride_q_pe_bs + q_pos * stride_q_pe_s + cur_head_qpe[:, None] * stride_q_pe_h + offs_d_kpe[None, :]
        gl.amd.cdna4.async_copy.buffer_load_to_shared(buf_q_pe, Q_pe, offs_q_pe, mask = (cur_head_qpe < NHEAD)[:, None] if NHEAD % BLOCK_H != 0 else None)
        gl.amd.cdna4.async_copy.commit_group()

    e_max = gl.zeros([BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout)) - float("inf")
    e_sum = gl.zeros([BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout))
    acc = gl.zeros([BLOCK_H, HEAD_DIM_CKV], dtype=gl.float32, layout=mfma_layout)

    # Fold KV dequant scale into the QK temperature. For fp8 KV the real
    # logits are (Q @ K_fp8^T) * kv_scale * sm_scale; softmax is shift- but
    # not scale-invariant, so kv_scale must affect qk (not just acc).
    # For bf16 KV the wrapper passes kv_scale=1.0, so this is a no-op.
    qk_scale = sm_scale * kv_scale

    ### bufs of page_number
    shared_page: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
    bufs_page = gl.allocate_shared_memory(gl.int32, shape=[2, BLOCK_N], layout=shared_page)
    gl.static_assert(PAGE_SIZE == 1)

    offs_page_raw = gl.arange(0, BLOCK_N, layout=blocked_page)

    ################ prologue
    #### global load page number
    offs_n_page = start_n + offs_page_raw
    offs_page = batch_page_start + offs_n_page // PAGE_SIZE
    gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_page.index(0), Req_to_tokens, offs_page, offs_n_page < split_kv_end)
    gl.amd.cdna4.async_copy.commit_group()

    start_n += BLOCK_N
    #### global load page number
    offs_n_page = start_n + offs_page_raw
    offs_page = batch_page_start + offs_n_page // PAGE_SIZE
    gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_page.index(1), Req_to_tokens, offs_page, offs_n_page < split_kv_end)
    gl.amd.cdna4.async_copy.commit_group()

    #### local load Q
    gl.amd.cdna4.async_copy.wait_group(2)
    q_nope = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_q_nope, mfma_layout_a)
    if HAS_PE:
        q_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_q_pe, mfma_layout_a)

    #################### move here to work around allocate_shared_memory bug
    bufs_kv = gl.allocate_shared_memory(kvtype, shape=[2, HEAD_DIM_CKV, BLOCK_N], layout=shared_kv)
    if HAS_PE:
        bufs_kpe = gl.allocate_shared_memory(kvtype, shape=[2, HEAD_DIM_KPE, BLOCK_N], layout=shared_kpe)

    #### global load K
    # local load page number
    gl.amd.cdna4.async_copy.wait_group(1)
    if HAS_PE:
        kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page.index(0), gl.SliceLayout(0, blocked_kpe))
        # simplify for page_size 1
        kv_loc_pe = kv_page_number_pe

    # local load page number for slice 0
    bufs_page_0 = bufs_page.index(0).slice(0, BLOCK_N // 2, 0)
    kv_page_number_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_0, gl.SliceLayout(0, blocked_kv_slice))
    kv_loc0 = kv_page_number_0

    # global load K_nope slice 0
    offs_n_nope0 = split_kv_start + gl.arange(0, BLOCK_N // 2, layout=gl.SliceLayout(0, blocked_kv_slice))
    offs_d_ckv_10 = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv_slice))
    offs_k_c0 = kv_loc0[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
    bufs_kv0 = bufs_kv.index(0).slice(0, BLOCK_N // 2, 1)
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv0, Kv_c_cache, offs_k_c0, mask=offs_n_nope0[None, :] < split_kv_end)
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv0, Kv_c_cache + offs_k_c0, mask=offs_n_nope0[None, :] < split_kv_end, other=0.0)
    gl.amd.cdna4.async_copy.commit_group()

    # global load K_pe
    if HAS_PE:
        offs_n_pe0 = split_kv_start + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kpe))
        offs_d_kpe_1 = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(1, blocked_kpe))
        offs_k_pe = kv_loc_pe[None, :] * stride_k_pe_bs + offs_d_kpe_1[:, None] + KV_PE_OFFSET
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kpe.index(0), K_pe_cache, offs_k_pe, mask=offs_n_pe0[None, :] < split_kv_end)
        else:
            gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kpe.index(0), K_pe_cache + offs_k_pe, mask=offs_n_pe0[None, :] < split_kv_end, other=0.0)
        gl.amd.cdna4.async_copy.commit_group()

    # local load page number for slice 1
    bufs_page_1 = bufs_page.index(0).slice(BLOCK_N // 2, BLOCK_N // 2, 0)
    kv_page_number_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_1, gl.SliceLayout(0, blocked_kv_slice))
    kv_loc1 = kv_page_number_1

    # global load K_nope slice 1
    offs_n_nope1 = offs_n_nope0 + BLOCK_N // 2
    bufs_kv1 = bufs_kv.index(0).slice(BLOCK_N // 2, BLOCK_N // 2, 1)
    offs_k_c1 = kv_loc1[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv1, Kv_c_cache, offs_k_c1, mask=offs_n_nope1[None, :] < split_kv_end)
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv1, Kv_c_cache + offs_k_c1, mask=offs_n_nope1[None, :] < split_kv_end, other=0.0)
    gl.amd.cdna4.async_copy.commit_group()

    if REGIME == 'bh64':
        # bh64 guarantees >= 3 iters/split; this constant-folds the
        # `if num_iter >= 2` epilogue-1 guard below so its codegen is unchanged.
        gl.assume(num_iter >= 3)
    buf_idx = 0
    ################ loop
    for i in range(num_iter - 2):
        async_idx = (buf_idx + 1) % 2

        gl.amd.cdna4.async_copy.wait_group(0)
        #### global load page number
        offs_n_page = start_n + BLOCK_N + offs_page_raw
        offs_page = batch_page_start + offs_n_page // PAGE_SIZE
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_page.index(buf_idx), Req_to_tokens, offs_page, offs_n_page < split_kv_end)
        gl.amd.cdna4.async_copy.commit_group()

        #### global load K
        bufs_kv0 = bufs_kv.index(async_idx).slice(0, BLOCK_N // 2, 1)
        bufs_kv1 = bufs_kv.index(async_idx).slice(BLOCK_N // 2, BLOCK_N // 2, 1)
        # local load page number for slice 0
        bufs_page_0 = bufs_page.index(async_idx).slice(0, BLOCK_N // 2, 0)
        kv_page_number_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_0, gl.SliceLayout(0, blocked_kv_slice))
        kv_loc0 = kv_page_number_0
        # global load K_nope slice 0
        offs_n_nope0 = start_n + gl.arange(0, BLOCK_N // 2, layout=gl.SliceLayout(0, blocked_kv_slice))
        offs_d_ckv_10 = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv_slice))
        offs_k_c0 = kv_loc0[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv0, Kv_c_cache, offs_k_c0, mask=offs_n_nope0[None, :] < split_kv_end)
        else:
            # >2GB path needs the same bounds mask + other=0.0 as buffer_load.
            gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv0, Kv_c_cache + offs_k_c0, mask=offs_n_nope0[None, :] < split_kv_end, other=0.0)
        gl.amd.cdna4.async_copy.commit_group()

        # local load page_number_pe
        if HAS_PE:
            kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page.index(async_idx), gl.SliceLayout(0, blocked_kpe))
            kv_loc_pe = kv_page_number_pe
            # global load K_pe
            offs_n_pe = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kpe))
            offs_d_kpe_1 = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(1, blocked_kpe))
            offs_k_pe = kv_loc_pe[None, :] * stride_k_pe_bs + offs_d_kpe_1[:, None] + KV_PE_OFFSET
            if WITHIN_2GB:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kpe.index(async_idx), K_pe_cache, offs_k_pe, mask=offs_n_pe[None, :] < split_kv_end)
            else:
                # >2GB path needs the same bounds mask + other=0.0 as buffer_load.
                gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kpe.index(async_idx), K_pe_cache + offs_k_pe, mask=offs_n_pe[None, :] < split_kv_end, other=0.0)
            gl.amd.cdna4.async_copy.commit_group()

        #### dot, softmax, dot (part0)
        k_c = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_kv.index(buf_idx), mfma_layout_b)
        zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
        qk = gl.amd.cdna4.mfma(q_nope, k_c.to(dtype), zeros)
        if HAS_PE:
            k_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_kpe.index(buf_idx), mfma_layout_b)
            qk = gl.amd.cdna4.mfma(q_pe, k_pe.to(dtype), qk)

        # local load page number for slice 1
        bufs_page_1 = bufs_page.index(async_idx).slice(BLOCK_N // 2, BLOCK_N // 2, 0)
        kv_page_number_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_1, gl.SliceLayout(0, blocked_kv_slice))
        kv_loc1 = kv_page_number_1
        # global load K_nope slice 1
        offs_n1 = offs_n_nope0 + BLOCK_N // 2
        offs_k_c1 = kv_loc1[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv1, Kv_c_cache, offs_k_c1, mask=offs_n1[None, :] < split_kv_end)
        else:
            # >2GB path needs the same bounds mask + other=0.0 as buffer_load.
            gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv1, Kv_c_cache + offs_k_c1, mask=offs_n1[None, :] < split_kv_end, other=0.0)
        gl.amd.cdna4.async_copy.commit_group()

        #### dot, softmax, dot (part1)
        qk *= qk_scale
        offs_n_qk = split_kv_start + i * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
        qk = gl.where(offs_n_qk[None, :] < score_end, qk, float("-inf"))
        n_e_max = gl.maximum(gl.max(qk, 1), e_max)
        LOG2E: gl.constexpr = 1.4426950408889634
        re_scale = gl.exp2((e_max - n_e_max) * LOG2E)
        p = gl.exp2((qk - n_e_max[:, None]) * LOG2E)
        if QLEN > 1:
            # MTP: a leading/whole fully-masked split keeps e_max=n_e_max=-inf,
            # making re_scale/p NaN. Force them to 0 so the split cleanly yields
            # e_sum=0 -> lse=-inf, which stage-2 drops.
            re_scale = gl.where(e_max == float("-inf"), 0.0, re_scale)
            p = gl.where(n_e_max[:, None] == float("-inf"), 0.0, p)
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        e_max = n_e_max
        p = p.to(dtype)
        p = gl.convert_layout(p, mfma_layout_a)
        acc *= re_scale[:, None]
        v_c = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_kv.index(buf_idx), linear_v)
        v_c = v_c.to(dtype)
        v_c = gl.permute(v_c, [1, 0])
        v_c = gl.convert_layout(v_c, mfma_layout_b)
        acc = gl.amd.cdna4.mfma(p, v_c, acc)

        start_n += BLOCK_N
        buf_idx = (buf_idx + 1) % 2

    LOG2E: gl.constexpr = 1.4426950408889634

    ################ epilogue 1
    # Skip when num_iter < 2 (possible for bh16bn64 / bh16bn128 in either mode).
    # bh64 has gl.assume(num_iter >= 3) above so the compiler folds this branch
    # out there; for the bh16 regimes it stays a runtime branch.
    if num_iter >= 2:
        async_idx = (buf_idx + 1) % 2

        #### global load K
        # local load page number
        gl.amd.cdna4.async_copy.wait_group(3 if HAS_PE else 2)
        kv_page_number = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page.index(async_idx), gl.SliceLayout(0, blocked_kv))
        kv_loc = kv_page_number
        if HAS_PE:
            kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page.index(async_idx), gl.SliceLayout(0, blocked_kpe))
            kv_loc_pe = kv_page_number_pe
        # global load K_nope
        offs_n_nope = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kv))
        offs_d_ckv_1 = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv))
        offs_k_c = kv_loc[None, :] * stride_kv_c_bs + offs_d_ckv_1[:, None]
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv.index(async_idx), Kv_c_cache, offs_k_c, mask=offs_n_nope[None, :] < split_kv_end)
        else:
            # >2GB path needs the same bounds mask + other=0.0 as buffer_load.
            gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv.index(async_idx), Kv_c_cache + offs_k_c, mask=offs_n_nope[None, :] < split_kv_end, other=0.0)
        gl.amd.cdna4.async_copy.commit_group()
        # global load K_pe
        if HAS_PE:
            offs_n_pe = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kpe))
            offs_d_kpe_1 = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(1, blocked_kpe))
            offs_k_pe = kv_loc_pe[None, :] * stride_k_pe_bs + offs_d_kpe_1[:, None] + KV_PE_OFFSET
            if WITHIN_2GB:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kpe.index(async_idx), K_pe_cache, offs_k_pe, mask=offs_n_pe[None, :] < split_kv_end)
            else:
                gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kpe.index(async_idx), K_pe_cache + offs_k_pe, mask=offs_n_pe[None, :] < split_kv_end, other=0.0)
            gl.amd.cdna4.async_copy.commit_group()

        # dot, softmax, dot
        gl.amd.cdna4.async_copy.wait_group(2 if HAS_PE else 1)
        k_c = bufs_kv.index(buf_idx).load(layout=mfma_layout_b)
        zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
        qk = gl.amd.cdna4.mfma(q_nope, k_c.to(dtype), zeros)

        if HAS_PE:
            k_pe = bufs_kpe.index(buf_idx).load(layout=mfma_layout_b)
            qk = gl.amd.cdna4.mfma(q_pe, k_pe.to(dtype), qk)
        qk *= qk_scale
        offs_n_qk = split_kv_start + (num_iter - 2) * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
        qk = gl.where(offs_n_qk[None, :] < score_end, qk, float("-inf"))
        n_e_max = gl.maximum(gl.max(qk, 1), e_max)
        re_scale = gl.exp2((e_max - n_e_max) * LOG2E)
        p = gl.exp2((qk - n_e_max[:, None]) * LOG2E)
        if QLEN > 1:
            re_scale = gl.where(e_max == float("-inf"), 0.0, re_scale)
            p = gl.where(n_e_max[:, None] == float("-inf"), 0.0, p)
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        e_max = n_e_max
        p = p.to(dtype)
        p = gl.convert_layout(p, mfma_layout_a)
        acc *= re_scale[:, None]
        v_c = bufs_kv.index(buf_idx).load(layout=linear_v)
        v_c = v_c.to(dtype)
        v_c = gl.permute(v_c, [1, 0])
        v_c = gl.convert_layout(v_c, mfma_layout_b)
        acc = gl.amd.cdna4.mfma(p, v_c, acc)

        start_n += BLOCK_N
        buf_idx = (buf_idx + 1) % 2

    ################ epilogue 2
    #### dot, softmax, dot
    gl.amd.cdna4.async_copy.wait_group(0)
    k_c = bufs_kv.index(buf_idx).load(layout=mfma_layout_b)
    zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
    qk = gl.amd.cdna4.mfma(q_nope, k_c.to(dtype), zeros)

    if HAS_PE:
        k_pe = bufs_kpe.index(buf_idx).load(layout=mfma_layout_b)
        qk = gl.amd.cdna4.mfma(q_pe, k_pe.to(dtype), qk)
    qk *= qk_scale
    offs_n_qk = split_kv_start + (num_iter - 1) * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
    qk = gl.where(offs_n_qk[None, :] < score_end, qk, float("-inf"))
    n_e_max = gl.maximum(gl.max(qk, 1), e_max)
    re_scale = gl.exp2((e_max - n_e_max) * LOG2E)
    p = gl.exp2((qk - n_e_max[:, None]) * LOG2E)
    if QLEN > 1:
        re_scale = gl.where(e_max == float("-inf"), 0.0, re_scale)
        p = gl.where(n_e_max[:, None] == float("-inf"), 0.0, p)
    e_sum = e_sum * re_scale + gl.sum(p, 1)
    e_max = n_e_max
    p = p.to(dtype)
    p = gl.convert_layout(p, mfma_layout_a)
    acc *= re_scale[:, None]
    v_c = bufs_kv.index(buf_idx).load(layout=linear_v)
    v_c = v_c.to(dtype)
    v_c = gl.permute(v_c, [1, 0])
    v_c = gl.convert_layout(v_c, mfma_layout_b)
    acc = gl.amd.cdna4.mfma(p, v_c, acc)

    cur_head_o = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_layout))
    offs_d_ckv_o = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(0, mfma_layout))
    offs_o = cur_batch * stride_o_b + q_pos * stride_o_s + cur_head_o[:, None] * stride_o_h + split_kv_id * stride_o_split + offs_d_ckv_o[None, :]

    if HAS_ATTN_SINK:
        # Fold the optional per-head sink into the softmax denom (no V contribution).
        # e_max/e_sum are natural-log units (the *LOG2E is inside exp2), so is sink.
        if NHEAD % BLOCK_H != 0:
            sink = gl.load(Attn_sink + cur_head_o, mask=cur_head_o < NHEAD, other=float("-inf")).to(gl.float32)
        else:
            sink = gl.load(Attn_sink + cur_head_o).to(gl.float32)
        n_e_max = gl.maximum(e_max, sink)
        re_scale = gl.exp2((e_max - n_e_max) * LOG2E)
        acc *= re_scale[:, None]
        e_sum = e_sum * re_scale + gl.exp2((sink - n_e_max) * LOG2E)
        e_max = n_e_max

    acc *= kv_scale
    rcp = 1.0 / e_sum
    stored_value = (acc * rcp[:, None]).to(dtype)
    if NHEAD % BLOCK_H != 0:
        gl.amd.cdna4.buffer_store(stored_value, ptr=O, offsets=offs_o, mask=(cur_head_o < NHEAD)[:, None])
    else:
        gl.amd.cdna4.buffer_store(stored_value, ptr=O, offsets=offs_o)

    ### store lse
    blocked_lse: gl.constexpr = gl.BlockedLayout(size_per_thread=[1], threads_per_warp=[64], warps_per_cta=[4], order=[0])
    cur_head_lse = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=blocked_lse)
    if RETURN_LSE and NUM_KV_SPLITS == 1:
        # split==1: single split is the whole sequence, so its lse is the final lse.
        offs_final_lse = cur_batch * stride_final_lse_b + q_pos * stride_final_lse_s + cur_head_lse * stride_final_lse_h
        lse = e_max + gl.log(e_sum)
        lse = gl.convert_layout(lse, blocked_lse)
        if NHEAD % BLOCK_H != 0:
            gl.amd.cdna4.buffer_store(lse, ptr=Final_lse, offsets=offs_final_lse, mask=(cur_head_lse < NHEAD))
        else:
            gl.amd.cdna4.buffer_store(lse, ptr=Final_lse, offsets=offs_final_lse)
    elif NUM_KV_SPLITS > 1:
        # per-split lse for stage-2 reduce.
        offs_mid_lse = cur_batch * stride_mid_lse_b + q_pos * stride_mid_lse_s + cur_head_lse * stride_mid_lse_h + split_kv_id * stride_mid_lse_split
        lse = e_max + gl.log(e_sum)
        lse = gl.convert_layout(lse, blocked_lse)
        if NHEAD % BLOCK_H != 0:
            gl.amd.cdna4.buffer_store(lse, ptr=Mid_lse, offsets=offs_mid_lse, mask=(cur_head_lse < NHEAD))
        else:
            gl.amd.cdna4.buffer_store(lse, ptr=Mid_lse, offsets=offs_mid_lse)
# fmt: on


@gluon.jit
def _v2_body(
    split,
    b,
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


@gluon.jit
def _k3_auto_fwd(
    # Gluon (bh16bn128) operands
    Q_nope, Q_pe, KV, KV_INDICES, KV_INDPTR, G_O, Attn_sink, sm_scale, kv_scale_f,
    sq_nope_bs, sq_nope_s, sq_nope_h, sq_pe_bs, sq_pe_s, sq_pe_h, stride_kv,
    so_b, so_s, so_h, so_split, Mid_lse, sml_b, sml_s, sml_h, sml_split,
    # v2 operands
    Q, KV_SCALE, O_PART, LSE_PART, O_FINAL, stride_q_tok, stride_q_h, stride_o_tok, stride_o_h,
    sm_scale_log2,
    # regime switch
    Sel_buf, sel_bs, sel_thresh,
    SEL_P2: gl.constexpr,
    H: gl.constexpr,
    QLEN: gl.constexpr,
    KV_STRIDE: gl.constexpr,
    G_NSPLIT: gl.constexpr,
    G_WITHIN_2GB: gl.constexpr,
    V_NSPLIT: gl.constexpr,
    V_MIN_CHUNK: gl.constexpr,
    V_P_SCALE: gl.constexpr,
    V_LAZY_TAU: gl.constexpr,
    V_WITHIN_2GB: gl.constexpr,
):
    """1-D grid of max(gluon programs, v2 programs); every program evaluates
    eff = bs * max_b(L_b) >= thresh and runs the matching body (or exits)."""
    pid = gl.program_id(0)
    sel_lay: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    offs_sel = gl.arange(0, SEL_P2, layout=sel_lay)
    ip_lo = gl.load(KV_INDPTR + offs_sel, mask=offs_sel < sel_bs, other=0)
    ip_hi = gl.load(KV_INDPTR + offs_sel + 1, mask=offs_sel < sel_bs, other=0)
    sel_max_len = gl.max(ip_hi - ip_lo, axis=0)
    use_v2 = sel_max_len * sel_bs >= sel_thresh
    if pid == 0:
        gl.store(Sel_buf + offs_sel, gl.where(offs_sel == 0, use_v2.to(gl.int32), 0), mask=offs_sel == 0)
    if use_v2:
        if pid < V_NSPLIT * sel_bs:
            _v2_body(
                pid % V_NSPLIT, pid // V_NSPLIT,
                Q, KV, KV_INDPTR, KV_INDICES, KV_SCALE, O_PART, LSE_PART, O_FINAL,
                stride_q_tok, stride_q_h, stride_o_tok, stride_o_h, sm_scale_log2,
                KV_STRIDE=KV_STRIDE, H=H, QLEN=QLEN, BLOCK_M=128, BLOCK_N=64, D_V=512, D_PE=64,
                NSPLIT=V_NSPLIT, MIN_CHUNK=V_MIN_CHUNK, P_SCALE=V_P_SCALE, WITHIN_2GB=V_WITHIN_2GB,
                NUM_WARPS=4, NUM_STAGES=3, LAZY_TAU=V_LAZY_TAU,
            )
    else:
        if pid < sel_bs * G_NSPLIT * QLEN:
            _gluon_body(
                pid % sel_bs, (pid // sel_bs) % G_NSPLIT, pid // (sel_bs * G_NSPLIT),
                Q_nope, Q_pe, KV, KV, KV_INDICES, KV_INDPTR, G_O, Attn_sink, sm_scale, kv_scale_f,
                sq_nope_bs, sq_nope_s, sq_nope_h, sq_pe_bs, sq_pe_s, sq_pe_h,
                stride_kv, stride_kv, 0,
                so_b, so_s, so_h, so_split,
                Mid_lse, sml_b, sml_s, sml_h, sml_split,
                None, 0, 0, 0,
                BLOCK_H=16, BLOCK_N=128, NUM_KV_SPLITS=G_NSPLIT, PAGE_SIZE=1, HEAD_DIM_CKV=512,
                HEAD_DIM_KPE=64, KV_PE_OFFSET=512, USE_2D_VIEW=False, WITHIN_2GB=G_WITHIN_2GB,
                NUM_XCDS=1, NHEAD=H, REGIME="bh16bn128", RETURN_LSE=False, QLEN=QLEN, HAS_PE=True,
                HAS_ATTN_SINK=False,
            )


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
    """One program per (request, row = q_pos*H + h, D_BLK columns)."""
    ROWS: tl.constexpr = H * QLEN
    b = tl.program_id(0)
    r = tl.program_id(1)
    d0 = tl.program_id(2) * D_BLK
    offs_d = d0 + tl.arange(0, D_BLK)
    qpos = r // H
    h = r % H
    o_off = (b * QLEN + qpos).to(tl.int64) * stride_o_tok + h * stride_o_h
    use_v2 = tl.load(FLAG)
    if use_v2 != 0:
        # == _k3_mla_verify_reduce (base-2 LSE merge of the v2/v3 splits)
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
        tl.store(O + o_off + offs_d, o.to(O.dtype.element_ty))
    elif G_NSPLIT > 1:
        # == aiter _mla_softmax_reducev_kernel for this row / column block
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
            glse = tl.load(G_LSE + base_ml + gsid * g_ml_s, mask=gsm, other=-float("inf"))
            lg = tl.load(G_LOGITS + base_l + gsid[:, None] * g_sl_s + offs_d[None, :],
                         mask=gsm[:, None], other=0.0).to(tl.float32)
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


def gluon_num_splits(bs: int, qlen: int, num_heads: int) -> int:
    """aiter mla_gluon bh16bn128 split budget (AITER_MLA_GLUON_WG_BUDGET honoured)."""
    wg_budget = int(os.environ.get("AITER_MLA_GLUON_WG_BUDGET", "256"))
    return max(1, wg_budget // (bs * qlen * triton.cdiv(num_heads, _GLUON_BLOCK_H)))


def k3_mla_verify_auto(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    kv_scale=None,
    kv_scale_float: Optional[float] = None,
    *,
    num_heads: int = 12,
    qlen: int = 8,
    v_head_dim: int = 512,
    out: Optional[torch.Tensor] = None,
    thresh: Optional[int] = None,
    w8_max_bs: int = _W8_MAX_BS,
    fused: bool = True,
) -> torch.Tensor:
    """MLA target verify; same contract as k3_mla_verify_v2.

    kv_scale: fp32 [1] tensor for the v2 path (or float / None);
    kv_scale_float: the same scale as a host float for the Gluon path (taken
    from kv_scale when it is not a tensor, else 1.0 if omitted).
    thresh: v2 regime iff bs * max(L) >= thresh (default auto_threshold(bs)).
    fused: one stage-1 launch running either body (2 launches total);
        False = separate Gluon-with-exit + v2/v3 launches (3 launches).
    w8_max_bs: (fused=False only) v3 8-warp stage 1 for bs <= w8_max_bs."""
    assert _HAS_GLUON, "k3_mla_verify_auto needs triton.experimental.gluon (gfx950)"
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
    assert out.is_contiguous()
    if kv_scale_float is None:
        kv_scale_float = 1.0 if isinstance(kv_scale, torch.Tensor) or kv_scale is None else float(kv_scale)
    if kv_scale is None:
        kv_scale = _ones(q.device)
    elif not isinstance(kv_scale, torch.Tensor):
        kv_scale = torch.full((1,), float(kv_scale), dtype=torch.float32, device=q.device)
    if bs == 0:
        return out
    if thresh is None:
        thresh = auto_threshold(bs)
    thresh = int(min(max(thresh, 0), _INT32_MAX))
    dev = q.device
    rows = num_heads * qlen

    # Gluon (bh16bn128) operands / partials
    g_ns = gluon_num_splits(bs, qlen, num_heads)
    q4 = q.view(bs, qlen, num_heads, D)
    q_nope = q4[..., :v_head_dim]
    q_pe = q4[..., v_head_dim:]
    o4 = out.view(bs, qlen, num_heads, v_head_dim)
    if g_ns == 1:  # Gluon stores the final output directly
        g_logits = o4.view(bs, qlen, num_heads, 1, v_head_dim)
        g_lse = None
        g_ml = (0, 0, 0, 0)
    else:
        g_logits = torch.empty((bs, qlen, num_heads, g_ns, v_head_dim), dtype=out.dtype, device=dev)
        g_lse = torch.empty((bs, qlen, num_heads, g_ns), dtype=torch.float32, device=dev)
        g_ml = g_lse.stride()
    g_sl = g_logits.stride()[:4]
    g_within_2gb = kv.shape[0] * kv.stride(0) * kv.element_size() <= 0x80000000

    # v2 operands / partials
    w8 = (not fused) and bs <= w8_max_bs
    min_chunk = _V3_MIN_CHUNK
    nsplit = max(2, default_num_splits(bs))
    block_n = 64
    o_part = torch.empty((bs, nsplit, rows, v_head_dim), dtype=torch.bfloat16, device=dev)
    lse_part = torch.empty((bs, nsplit, rows), dtype=torch.float32, device=dev)
    lazy_tau = _V2_LAZY_TAU
    p_scale = 256.0 / (2.0**lazy_tau)
    v_within_2gb = kv.numel() * kv.element_size() <= 0x7FFFFFFF
    sel_p2 = max(256, triton.next_power_of_2(bs + 2))

    if fused:
        # ---------------- 1. one stage-1 launch, regime picked per program
        flag = torch.empty((1,), dtype=torch.int32, device=dev)
        grid = max(bs * g_ns * qlen, nsplit * bs)
        _k3_auto_fwd[(grid,)](
            q_nope, q_pe, kv, kv_indices, kv_indptr, g_logits, _dummy_f32(dev), sm_scale, kv_scale_float,
            q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
            q_pe.stride(0), q_pe.stride(1), q_pe.stride(2), kv.stride(0),
            g_sl[0], g_sl[1], g_sl[2], g_sl[3],
            g_lse, g_ml[0], g_ml[1], g_ml[2], g_ml[3],
            q, kv_scale, o_part, lse_part, out, q.stride(0), q.stride(1), out.stride(0), out.stride(1),
            sm_scale * _LOG2E,
            flag, bs, thresh,
            SEL_P2=sel_p2, H=num_heads, QLEN=qlen, KV_STRIDE=kv.stride(0),
            G_NSPLIT=g_ns, G_WITHIN_2GB=g_within_2gb,
            V_NSPLIT=nsplit, V_MIN_CHUNK=min_chunk, V_P_SCALE=p_scale, V_LAZY_TAU=lazy_tau,
            V_WITHIN_2GB=v_within_2gb,
            num_warps=4,
        )
    else:
        # ---------------- 1. Gluon stage 1 with the regime prologue
        sel = torch.empty((bs + 2,), dtype=torch.int32, device=dev)
        flag = sel[bs + 1:]
        if not w8:
            min_chunk = _V2_MIN_CHUNK
        _mla_gluon_sel[(bs, g_ns, triton.cdiv(num_heads, _GLUON_BLOCK_H) * qlen)](
            q_nope, q_pe, kv, kv, kv_indices, kv_indptr, g_logits, _dummy_f32(dev),
            sm_scale, kv_scale_float,
            q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
            q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
            kv.stride(0), kv.stride(0), 0,
            g_sl[0], g_sl[1], g_sl[2], g_sl[3],
            g_lse, g_ml[0], g_ml[1], g_ml[2], g_ml[3],
            None, 0, 0, 0,
            BLOCK_H=_GLUON_BLOCK_H, BLOCK_N=_GLUON_BLOCK_N, NUM_KV_SPLITS=g_ns, PAGE_SIZE=1,
            HEAD_DIM_CKV=v_head_dim, HEAD_DIM_KPE=D - v_head_dim, KV_PE_OFFSET=v_head_dim,
            USE_2D_VIEW=False, WITHIN_2GB=g_within_2gb, NUM_XCDS=1, NHEAD=num_heads,
            REGIME="bh16bn128", RETURN_LSE=False, QLEN=qlen, HAS_PE=True, HAS_ATTN_SINK=False,
            Sel_buf=sel, sel_bs=bs, sel_thresh=thresh, SEL_P2=sel_p2,
        )
        # ---------------- 2. v2 / v3 stage 1 on sel (all-empty in the Gluon regime)
        if w8:
            _k3_mla_verify_v3_fwd[(nsplit, bs)](
                q, kv, sel, kv_indices, kv_scale, o_part, lse_part,
                q.stride(0), q.stride(1), sm_scale * _LOG2E,
                KV_STRIDE=kv.stride(0), H=num_heads, QLEN=qlen, BLOCK_N=block_n, NSPLIT=nsplit,
                MIN_CHUNK=min_chunk, P_SCALE=p_scale, LAZY_TAU=lazy_tau, WITHIN_2GB=v_within_2gb,
                num_warps=8,
            )
        else:
            _k3_mla_verify_fwd_gluon[(nsplit, bs)](
                q, kv, sel, kv_indices, kv_scale, o_part, lse_part, out,
                q.stride(0), q.stride(1), out.stride(0), out.stride(1),
                sm_scale * _LOG2E,
                KV_STRIDE=kv.stride(0), H=num_heads, QLEN=qlen, BLOCK_M=triton.next_power_of_2(rows),
                BLOCK_N=block_n, D_V=v_head_dim, D_PE=D - v_head_dim, NSPLIT=nsplit, MIN_CHUNK=min_chunk,
                P_SCALE=p_scale, WITHIN_2GB=v_within_2gb, NUM_WARPS=4, NUM_STAGES=3, PAD_PAIRS=((1024, 16),),
                LAZY_TAU=lazy_tau,
                num_warps=4,
            )

    # ---------------- regime-aware reduce
    g_lse_arg = g_lse if g_lse is not None else lse_part
    _k3_auto_reduce[(bs, rows, v_head_dim // 128)](
        flag, kv_indptr, o_part, lse_part, g_logits, g_lse_arg, out,
        out.stride(0), out.stride(1),
        g_sl[0], g_sl[1], g_sl[2], g_sl[3],
        g_ml[0], g_ml[1], g_ml[2], g_ml[3],
        H=num_heads, QLEN=qlen, D_V=v_head_dim, NSPLIT=nsplit,
        NSPLIT_P2=triton.next_power_of_2(nsplit), BLOCK_N=block_n, MIN_CHUNK=min_chunk,
        S_BLK=min(64, triton.next_power_of_2(nsplit)),
        G_NSPLIT=g_ns, G_S_BLK=min(64, triton.next_power_of_2(g_ns)), G_BLOCK_N=_GLUON_BLOCK_N,
        D_BLK=128,
        num_warps=4 if nsplit >= 64 else 2,
    )
    return out


@functools.lru_cache(maxsize=8)
def _dummy_f32(device):
    return torch.empty(1, dtype=torch.float32, device=device)
