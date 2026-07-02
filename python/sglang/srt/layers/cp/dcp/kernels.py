# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Triton kernels for decode context parallel (DCP).

Consolidated from the two merged DCP implementations:
  - create_triton_kv_indices_for_dcp_triton  (PR #25090, Triton/MHA path)
  - create_dcp_kv_indices / update_kv_lens_and_indices  (PR #14194, MLA path)
  - _correct_attn_cp_out_kernel / correct_attn_out / CPTritonContext  (PR #14194)
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# KV-index build (PR #25090, Triton/MHA): per-rank local KV indices.
# ---------------------------------------------------------------------------
@triton.jit
def create_triton_kv_indices_for_dcp_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    dcp_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    dcp_size: tl.constexpr,
    dcp_rank: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)

    # First absolute token position in this range owned by dcp_rank.
    # Triton follows C-style remainder for negative values, so avoid
    # computing the offset as a negative remainder when kv_start > dcp_rank.
    kv_start_mod = kv_start % dcp_size
    first = kv_start + ((dcp_rank + dcp_size - kv_start_mod) % dcp_size)
    local_len = tl.load(dcp_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(local_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < local_len
        abs_pos = first + offset * dcp_size
        data = tl.load(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + abs_pos,
            mask=mask,
        )
        tl.store(
            kv_indices_ptr + kv_indices_offset + offset, data // dcp_size, mask=mask
        )


# ---------------------------------------------------------------------------
# KV-index build (PR #14194, MLA): global prefix+extend layout for the
# all-gathered dcp_kv_buffer, plus the per-rank shard/compact kernel.
# ---------------------------------------------------------------------------
@triton.jit
def create_dcp_kv_indices(
    kv_indptr,
    extend_lens_ptr,
    extend_cu_lens_ptr,
    extend_prefix_lens_ptr,
    extend_cu_prefix_lens_ptr,
    kv_indices_ptr,
    extend_prefix_lens_sum,
    dcp_world_size: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)
    prefix_len = tl.load(extend_prefix_lens_ptr + pid)
    prefix_start = tl.load(extend_cu_prefix_lens_ptr + pid)
    kv_ind_start = tl.load(kv_indptr + pid)
    num_loop = tl.cdiv(prefix_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < prefix_len
        data = prefix_start + offset
        tl.store(kv_indices_ptr + kv_ind_start + offset, data, mask=mask)
    extend_len = tl.load(extend_lens_ptr + pid)
    extend_start = tl.load(extend_cu_lens_ptr + pid)
    num_loop = tl.cdiv(extend_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < extend_len
        data = extend_prefix_lens_sum + extend_start + offset
        tl.store(
            kv_indices_ptr + kv_ind_start + prefix_len + offset,
            data,
            mask=mask,
        )


@triton.jit
def update_kv_lens_and_indices(
    kv_lens: torch.Tensor,
    kv_lens_cumsum: torch.Tensor,
    kv_indices: torch.Tensor,
    local_kv_lens: torch.Tensor,
    local_kv_lens_cumsum: torch.Tensor,
    local_kv_indices: torch.Tensor,
    dcp_rank: tl.constexpr,
    dcp_world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bs_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    local_kv_len = tl.load(local_kv_lens + bs_idx)
    local_kv_indices_start = tl.load(local_kv_lens_cumsum + bs_idx)
    kv_indices_start = tl.load(kv_lens_cumsum + bs_idx)

    block_start = block_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < local_kv_len

    kv_indice_offsets = offsets * dcp_world_size + dcp_rank + kv_indices_start
    local_kv_indices_offsets = local_kv_indices_start + offsets

    kv_values = tl.load(kv_indices + kv_indice_offsets, mask=mask)
    tl.store(
        local_kv_indices + local_kv_indices_offsets,
        kv_values // dcp_world_size,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Partial-attention LSE correction (PR #14194, MLA path).
# ---------------------------------------------------------------------------
@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    new_outputs_stride_H,
    new_outputs_stride_B,
    new_outputs_stride_D,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
):
    """
    Apply the all-gathered lses to correct each local rank's attention
    output. we still need perform a cross-rank reduction to obtain the
    final attention output.

    Args:
        outputs_ptr (triton.PointerType):
            Pointer to input tensor of shape [ B, H, D ]
        lses_ptr (triton.PointerType):
            Pointer to input tensor of shape [ N, B, H ]
        new_output_ptr (triton.PointerType):
            Pointer to output tensor of shape [ H, B, D ]
        vlse_ptr (triton.PointerType):
            Pointer to output tensor of shape [ B, H ]
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)

    # Use int32 for offsets where possible to reduce register pressure
    b_i32 = batch_idx.to(tl.int32)
    h_i32 = head_idx.to(tl.int32)

    # Vectorized load of LSE values: shape = [N]
    num_n_offsets = tl.arange(0, N_ROUNDED)
    lse_offsets = (
        num_n_offsets * lses_stride_N + b_i32 * lses_stride_B + h_i32 * lses_stride_H
    )

    # Compute final LSE using online softmax algorithm (more numerically stable)
    lse = tl.load(lses_ptr + lse_offsets)

    # Replace NaN and inf with -inf for numerical stability
    neg_inf = float("-inf")
    lse = tl.where((lse != lse) | (lse == float("inf")), neg_inf, lse)

    # Online softmax: find max, subtract, exp, sum, log
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == neg_inf, 0.0, lse_max)
    lse = lse - lse_max
    lse_exp = tl.exp2(lse)
    lse_acc = tl.sum(lse_exp, axis=0)
    final_lse = tl.log2(lse_acc) + lse_max

    # Compute correction factor
    lse_offset = lse_idx * lses_stride_N + b_i32 * lses_stride_B + h_i32 * lses_stride_H
    local_lse = tl.load(lses_ptr + lse_offset)
    lse_diff = local_lse - final_lse
    lse_diff = tl.where(
        (lse_diff != lse_diff) | (lse_diff == float("inf")),
        neg_inf,
        lse_diff,
    )
    factor = tl.exp2(lse_diff)

    # Store final LSE
    tl.store(vlse_ptr + b_i32 * lses_stride_B + h_i32 * lses_stride_H, final_lse)

    # Load output with vectorized access: shape = [D]
    d_offsets = tl.arange(0, HEAD_DIM)
    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )

    new_output_offsets = (
        head_idx * new_outputs_stride_H
        + batch_idx * new_outputs_stride_B
        + d_offsets * new_outputs_stride_D
    )
    # Apply correction and store. A rank whose local KV shard is empty for
    # this token contributes factor == 0, but its kernel output may be NaN
    # (e.g. the tokenspeed non-split-KV epilogue computes 0 * inf for empty
    # rows); NaN * 0 = NaN would poison the reduce-scatter sum, so force the
    # zero-weight contribution to exactly 0 instead of trusting the output.
    output = tl.load(outputs_ptr + output_offsets)
    output = tl.where(factor == 0.0, 0.0, output * factor)
    tl.store(new_output_ptr + new_output_offsets, output)


class CPTritonContext:
    """The CPTritonContext is used to avoid recompilation of the Triton JIT."""

    def __init__(self):
        self.inner_kernel = None

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        if self.inner_kernel is None:
            self.inner_kernel = kernel[grid](*regular_args, **const_args)
        else:
            self.inner_kernel[grid](*regular_args)


def correct_attn_out(
    out: torch.Tensor,
    lses: torch.Tensor,
    cp_rank: int,
    ctx: Optional[CPTritonContext],
    new_output: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct the attention output using the all-gathered lses.

    Args:
        out: Tensor of shape [ B, H, D ]
        lses: Tensor of shape [ N, B, H ]
        cp_rank: Current rank in the context-parallel group
        ctx: Triton context to avoid recompilation

    Returns:
        Tuple of (out, lse) with corrected attention and final log-sum-exp.
    """
    if ctx is None:
        ctx = CPTritonContext()

    # --- Normalize to 3D views ---
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, f"expected out [B,H,D] or [B,1,H,D], got {tuple(out.shape)}"

    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3, (
        f"expected lses [N,B,H] (optionally with a 1-sized extra dim), "
        f"got {tuple(lses.shape)}"
    )

    B, H, D = out.shape
    N = lses.shape[0]

    # Strides after we normalized shapes to 3-D views.  The kernel computes
    # offsets for `vlse_ptr` using lses_stride_B/H, so the output buffer must
    # have the same B/H stride layout as a slice of `lses`.
    o_sB, o_sH, o_sD = out.stride()
    l_sN, l_sB, l_sH = lses.stride()
    no_sH, no_sB, no_sD = new_output.stride()
    # Allocate LSE with the same B/H strides as `lses` so writes land correctly
    # even when `lses` is a non-contiguous view (e.g., 4-D to 3-D squeeze).
    lse = torch.empty_strided(
        (B, H), (l_sB, l_sH), device=lses.device, dtype=lses.dtype
    )

    # Kernel launch config
    grid = (B, H, 1)

    regular_args = (
        out,
        new_output,
        lses,
        lse,
        o_sB,
        o_sH,
        o_sD,
        l_sN,
        l_sB,
        l_sH,
        no_sH,
        no_sB,
        no_sD,
        cp_rank,
    )
    const_args = {"HEAD_DIM": D, "N_ROUNDED": N}

    ctx.call_kernel(_correct_attn_cp_out_kernel, grid, *regular_args, **const_args)
    return new_output, lse


# ---------------------------------------------------------------------------
# DCP speculative-verify: fused residue-class draft-block attention + local
# base-2 LSE merge with the prefix-phase partial (out, lse).
#
# Replaces the ~25 unfused fp32 torch ops previously inlined in
# TRTLLMMLABackend._forward_target_verify_dcp phases (b)+(c). The torch
# reference implementation is kept below (dcp_verify_draft_merge_torch) for
# A/B debugging via SGLANG_DCP_VERIFY_FUSED=0 and for the unit test.
# ---------------------------------------------------------------------------

_LOG2_E = math.log2(math.e)
# torch.finfo(torch.float32).tiny — matches the .clamp(min=tiny) guards of the
# torch reference exactly.
_FP32_TINY = 1.1754943508222875e-38


@triton.jit
def _dcp_verify_draft_merge_kernel(
    q_ptr,  # [bs, draft, H, KV_LORA + ROPE_DIM] (may be fp8/bf16/fp16)
    k_lat_ptr,  # [bs, draft, KV_LORA] (same dtype family as q)
    k_rope_ptr,  # [bs, draft, ROPE_DIM]
    o_a_ptr,  # [bs, draft, H, KV_LORA] phase-(a) output (kernel out dtype)
    lse_a_ptr,  # [bs, draft, H] fp32, base-2, +inf/NaN sentinel on empty rows
    seq_lens_ptr,  # [bs] committed prefix lengths (device tensor)
    out_ptr,  # [bs, draft, H, KV_LORA] merged output (o_a dtype)
    lse_out_ptr,  # [bs, draft, H] fp32 merged base-2 LSE (-inf if empty)
    stride_q_b,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kl_b,
    stride_kl_t,
    stride_kl_d,
    stride_kr_b,
    stride_kr_t,
    stride_kr_d,
    stride_oa_b,
    stride_oa_t,
    stride_oa_h,
    stride_oa_d,
    stride_la_b,
    stride_la_t,
    stride_la_h,
    stride_o_b,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_lo_b,
    stride_lo_t,
    stride_lo_h,
    scale_log2,  # softmax_scale * log2(e): scores land in the base-2 domain
    output_scale,  # fp8-KV dequant scale applied to the block value path
    dcp_rank: tl.constexpr,
    dcp_world_size: tl.constexpr,
    DRAFT: tl.constexpr,
    DRAFT_POW2: tl.constexpr,
    KV_LORA: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per (batch, head). draft x draft score tile, vectorized
    loads over the head dim. All math fp32; q/k/o_a converted at load."""
    NEG_INF = float("-inf")
    POS_INF = float("inf")
    FP32_TINY = 1.1754943508222875e-38

    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)

    t = tl.arange(0, DRAFT_POW2)
    t_mask = t < DRAFT

    # keep[i, j]: q token i attends block token j iff this rank owns block
    # position j ((seq_len + j) % dcp_world_size == dcp_rank) and i >= j.
    seq_len = tl.load(seq_lens_ptr + b).to(tl.int64)
    owner = ((seq_len + t) % dcp_world_size) == dcp_rank
    keep = (
        (t[:, None] >= t[None, :])
        & owner[None, :]
        & t_mask[:, None]
        & t_mask[None, :]
    )

    # ---- scores[i, j] = log2(e) * scale * dot(q_i, concat(k_lat, k_rope)_j)
    scores = tl.zeros([DRAFT_POW2, DRAFT_POW2], dtype=tl.float32)
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    for d0 in tl.static_range(0, KV_LORA, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        qb = tl.load(
            q_base + t[:, None] * stride_q_t + d[None, :] * stride_q_d,
            mask=t_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        kb = tl.load(
            k_lat_ptr
            + b * stride_kl_b
            + t[:, None] * stride_kl_t
            + d[None, :] * stride_kl_d,
            mask=t_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        scores += tl.sum(qb[:, None, :] * kb[None, :, :], axis=2)
    dr = tl.arange(0, ROPE_DIM)
    qr = tl.load(
        q_base + t[:, None] * stride_q_t + (KV_LORA + dr[None, :]) * stride_q_d,
        mask=t_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    kr = tl.load(
        k_rope_ptr
        + b * stride_kr_b
        + t[:, None] * stride_kr_t
        + dr[None, :] * stride_kr_d,
        mask=t_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    scores += tl.sum(qr[:, None, :] * kr[None, :, :], axis=2)

    scores = scores * scale_log2
    scores = tl.where(keep, scores, NEG_INF)

    # ---- base-2 softmax with max subtraction; empty row -> lse_b = -inf.
    m_b = tl.max(scores, axis=1)
    m_b_safe = tl.where(
        (m_b != m_b) | (m_b == POS_INF) | (m_b == NEG_INF), 0.0, m_b
    )
    p = tl.exp2(scores - m_b_safe[:, None])
    p = tl.where(keep, p, 0.0)
    s_b = tl.sum(p, axis=1)
    lse_b = tl.where(
        s_b > 0, m_b_safe + tl.log2(tl.maximum(s_b, FP32_TINY)), NEG_INF
    )

    # ---- sanitize phase-(a) LSE (+inf / NaN empty-row sentinel -> -inf).
    lse_a = tl.load(
        lse_a_ptr + b * stride_la_b + t * stride_la_t + h * stride_la_h,
        mask=t_mask,
        other=NEG_INF,
    ).to(tl.float32)
    lse_a = tl.where(
        (lse_a != lse_a) | (lse_a == POS_INF) | (lse_a == NEG_INF),
        NEG_INF,
        lse_a,
    )

    # ---- base-2 merge weights.
    m_ab = tl.maximum(lse_a, lse_b)
    m_ab_safe = tl.where(
        (m_ab != m_ab) | (m_ab == POS_INF) | (m_ab == NEG_INF), 0.0, m_ab
    )
    w_a = tl.exp2(lse_a - m_ab_safe)
    w_b = tl.exp2(lse_b - m_ab_safe)
    denom = w_a + w_b
    denom_c = tl.maximum(denom, FP32_TINY)
    merged_lse = tl.where(denom > 0, m_ab_safe + tl.log2(denom_c), NEG_INF)
    tl.store(
        lse_out_ptr + b * stride_lo_b + t * stride_lo_t + h * stride_lo_h,
        merged_lse,
        mask=t_mask,
    )

    # Fold the per-row scalars into two coefficients:
    #   out = c_a * o_a + c_b * (sum_j p_j * v_j)
    # with c_a = w_a / denom and c_b = w_b * output_scale / (denom * s_b),
    # zeroed on empty rows — algebraically identical to the torch reference
    # (o_b = p@v * output_scale / s_b; out = (w_a*o_a + w_b*o_b) / denom).
    c_a = tl.where(denom > 0, w_a / denom_c, 0.0)
    c_b = tl.where(
        (denom > 0) & (s_b > 0),
        (w_b * output_scale) / (denom_c * tl.maximum(s_b, FP32_TINY)),
        0.0,
    )

    # ---- value pass (value = the 512-dim latent) + merge + store.
    oa_base = o_a_ptr + b * stride_oa_b + h * stride_oa_h
    out_base = out_ptr + b * stride_o_b + h * stride_o_h
    for d0 in tl.static_range(0, KV_LORA, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        v = tl.load(
            k_lat_ptr
            + b * stride_kl_b
            + t[:, None] * stride_kl_t
            + d[None, :] * stride_kl_d,
            mask=t_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        o_b = tl.sum(p[:, :, None] * v[None, :, :], axis=1)
        oa = tl.load(
            oa_base + t[:, None] * stride_oa_t + d[None, :] * stride_oa_d,
            mask=t_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        # nan_to_num(nan=0, posinf=0, neginf=0) on the phase-(a) output.
        oa = tl.where((oa != oa) | (oa == POS_INF) | (oa == NEG_INF), 0.0, oa)
        o = c_a[:, None] * oa + c_b[:, None] * o_b
        tl.store(
            out_base + t[:, None] * stride_o_t + d[None, :] * stride_o_d,
            o.to(out_ptr.dtype.element_ty),
            mask=t_mask[:, None],
        )


def dcp_verify_draft_merge(
    q: torch.Tensor,  # [bs, draft, H, kv_lora + rope_dim]
    k_latent: torch.Tensor,  # [bs, draft, kv_lora]
    k_rope: torch.Tensor,  # [bs, draft, rope_dim]
    o_a: torch.Tensor,  # [bs, draft, H, kv_lora]
    lse_a: torch.Tensor,  # [bs, draft, H] fp32
    seq_lens: torch.Tensor,  # [bs] device tensor (graph-replay safe)
    softmax_scale: float,
    output_scale: float,
    dcp_rank: int,
    dcp_world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused DCP verify: residue-class draft-block attention + base-2 merge.

    Returns (out [bs, draft, H, kv_lora] in o_a.dtype, lse [bs, draft, H]
    fp32). Static shapes, no host reads — CUDA-graph capturable. q/k may be
    fp8; conversion to fp32 happens inside the kernel at load time.
    """
    bs, draft, num_heads, d_qk = q.shape
    kv_lora = k_latent.shape[-1]
    rope_dim = k_rope.shape[-1]
    assert d_qk == kv_lora + rope_dim, (d_qk, kv_lora, rope_dim)
    BLOCK_D = 128
    assert kv_lora % BLOCK_D == 0, kv_lora
    assert rope_dim & (rope_dim - 1) == 0, rope_dim  # power of 2 for arange

    out = torch.empty(
        (bs, draft, num_heads, kv_lora), dtype=o_a.dtype, device=q.device
    )
    merged_lse = torch.empty(
        (bs, draft, num_heads), dtype=torch.float32, device=q.device
    )
    _dcp_verify_draft_merge_kernel[(bs, num_heads)](
        q,
        k_latent,
        k_rope,
        o_a,
        lse_a,
        seq_lens,
        out,
        merged_lse,
        *q.stride(),
        *k_latent.stride(),
        *k_rope.stride(),
        *o_a.stride(),
        *lse_a.stride(),
        *out.stride(),
        *merged_lse.stride(),
        float(softmax_scale) * _LOG2_E,
        float(output_scale),
        dcp_rank=dcp_rank,
        dcp_world_size=dcp_world_size,
        DRAFT=draft,
        DRAFT_POW2=triton.next_power_of_2(draft),
        KV_LORA=kv_lora,
        ROPE_DIM=rope_dim,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out, merged_lse


def dcp_verify_draft_merge_torch(
    q: torch.Tensor,
    k_latent: torch.Tensor,
    k_rope: torch.Tensor,
    o_a: torch.Tensor,
    lse_a: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    output_scale: float,
    dcp_rank: int,
    dcp_world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference for :func:`dcp_verify_draft_merge` (the original,
    numerically validated unfused implementation from
    TRTLLMMLABackend._forward_target_verify_dcp). Kept for A/B debugging
    (SGLANG_DCP_VERIFY_FUSED=0) and as the unit-test oracle."""
    bs, draft, num_heads, _ = q.shape
    kv_lora = k_latent.shape[-1]
    rope_dim = k_rope.shape[-1]
    neg_inf = float("-inf")
    fp32_tiny = torch.finfo(torch.float32).tiny

    q32 = q.to(torch.float32)
    k_lat32 = k_latent.reshape(bs, draft, kv_lora).to(torch.float32)
    k_pe32 = k_rope.reshape(bs, draft, rope_dim).to(torch.float32)
    k_full32 = torch.cat([k_lat32, k_pe32], dim=-1)  # [bs, draft, D_qk]

    # Scores directly in the base-2 domain: s2 = log2(e) * scale * (q . k)
    scores = torch.einsum("bqhd,bkd->bqhk", q32, k_full32) * (
        softmax_scale * _LOG2_E
    )

    j_idx = torch.arange(draft, device=q.device)
    # owner[b, j]: this rank owns block position j of request b
    owner = (
        seq_lens.to(torch.int64).unsqueeze(1) + j_idx.unsqueeze(0)
    ) % dcp_world_size == dcp_rank
    # causal[i, j]: q token i may attend block token j
    causal = j_idx.unsqueeze(1) >= j_idx.unsqueeze(0)  # [q, k]
    mask = owner[:, None, None, :] & causal[None, :, None, :]  # [bs,q,1,k]

    scores = scores.masked_fill(~mask, neg_inf)
    m_b = scores.amax(dim=-1, keepdim=True)  # [bs, q, H, 1]
    m_b_safe = torch.where(torch.isfinite(m_b), m_b, torch.zeros_like(m_b))
    p = torch.exp2(scores - m_b_safe)
    p = torch.where(mask, p, torch.zeros_like(p))
    s_b = p.sum(dim=-1, keepdim=True)  # [bs, q, H, 1]
    lse_b = torch.where(
        s_b.squeeze(-1) > 0,
        m_b_safe.squeeze(-1) + torch.log2(s_b.squeeze(-1).clamp(min=fp32_tiny)),
        torch.full_like(m_b_safe.squeeze(-1), neg_inf),
    )
    # value = latent (absorbed MLA); dequant with the same output scale
    # the FP8 decode kernel applies.
    o_b = torch.einsum("bqhk,bkd->bqhd", p, k_lat32) * output_scale
    o_b = o_b / s_b.clamp(min=fp32_tiny)
    o_b = torch.where(s_b > 0, o_b, torch.zeros_like(o_b))

    # ---- local base-2 merge of (a) and (b) ----
    lse_a32 = lse_a.to(torch.float32)
    # Normalize the kernel's empty-row sentinel (+inf, or NaN from
    # degenerate rows) to -inf; guard the paired output.
    lse_a32 = torch.where(
        torch.isnan(lse_a32) | torch.isinf(lse_a32),
        torch.full_like(lse_a32, neg_inf),
        lse_a32,
    )
    o_a32 = torch.nan_to_num(o_a.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)

    m_ab = torch.maximum(lse_a32, lse_b)
    m_ab_safe = torch.where(torch.isfinite(m_ab), m_ab, torch.zeros_like(m_ab))
    w_a = torch.exp2(lse_a32 - m_ab_safe)  # 0 when lse_a == -inf
    w_b = torch.exp2(lse_b - m_ab_safe)
    denom = w_a + w_b  # [bs, q, H]
    merged_lse = torch.where(
        denom > 0,
        m_ab_safe + torch.log2(denom.clamp(min=fp32_tiny)),
        torch.full_like(denom, neg_inf),
    )
    o = (
        w_a.unsqueeze(-1) * o_a32 + w_b.unsqueeze(-1) * o_b
    ) / denom.clamp(min=fp32_tiny).unsqueeze(-1)
    o = torch.where(denom.unsqueeze(-1) > 0, o, torch.zeros_like(o))
    return o.to(o_a.dtype), merged_lse
