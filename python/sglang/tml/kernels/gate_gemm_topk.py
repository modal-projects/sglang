"""Fused Inkling gate: GEMM (x @ W.T) + sigmoid+bias top-k + logsigmoid renorm.

One triton launch computes the whole gate for a tile of tokens: the gate
linear on tensor cores (bf16 x [T, 6144] @ bf16 W [264, 6144].T, fp32
accumulate), then the same top-6 + renorm epilogue as
``sigmoid_gate_topk_renorm`` -- without materializing logits in global memory.

This is a reference implementation and is not production-dispatched.
Production uses CUDA GEMV for small batches and cuBLAS otherwise.

The renorm needs sigmoid(raw logit) at the selected experts. Raw logits live
in registers and triton cannot gather from a register tile, so we recover them
from the selection score instead: ``sel = fl(sigmoid(raw) + bias)`` implies
``sel - bias = sigmoid(raw)`` up to one fp32 rounding of the addition (~1e-7
absolute), with no re-gather.
"""

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.kernels.gate_topk import (
    fpval_to_key,
    indx_to_key,
    key_to_fpval,
    key_to_indx,
)


@triton.jit
def _gate_gemm_topk_kernel(
    x_ptr,  # [M, HIDDEN] bf16
    w_ptr,  # W [>=N+S, HIDDEN] bf16, or W.T [HIDDEN, >=N+S] if TRANSPOSED_W
    bias_ptr,  # [N] fp32
    global_scale_ptr,
    routed_w_ptr,
    shared_w_ptr,
    indices_ptr,
    packed_indices_ptr,
    route_scale,
    M,
    stride_xm,
    stride_wn,
    HIDDEN: tl.constexpr,
    N: tl.constexpr,  # num routed experts (top-k sort dim), power of 2
    S: tl.constexpr,  # num shared experts
    K: tl.constexpr,
    A_POW2: tl.constexpr,  # next_pow2(K + S); must equal K + S here
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TRANSPOSED_W: tl.constexpr = False,
    RETURN_PACKED_TOPK: tl.constexpr = False,
    ENABLE_PDL: tl.constexpr = False,
    DEBUG_GEMM_ONLY: tl.constexpr = False,
):
    tl.static_assert(A_POW2 == K + S, "active slots 0..K-1 routed, K..K+S-1 shared")
    tl.static_assert(S == 2, "shared extraction below is unrolled for S == 2")
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, N)
    offs_sw = tl.arange(0, 16)  # shared-rows dot tile (tl.dot needs >= 16)
    mask_sw = offs_sw < S

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    # --- gate linear on tensor cores: fp32 accumulate like the cublas path ---
    acc_r = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    acc_s = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    if TRANSPOSED_W:
        # W.T [HIDDEN, >=N+S]: [BLOCK_K, N] tiles load directly, no register
        # transpose in the MMA loop (stride_wn is W.T's row stride, >= N+S).
        wr_ptrs = w_ptr + offs_k[:, None] * stride_wn + offs_n[None, :]
        ws_ptrs = w_ptr + offs_k[:, None] * stride_wn + (N + offs_sw)[None, :]
        for _i in range(HIDDEN // BLOCK_K):
            a = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
            br = tl.load(wr_ptrs)
            bs = tl.load(ws_ptrs, mask=mask_sw[None, :], other=0.0)
            acc_r = tl.dot(a, br, acc_r)
            acc_s = tl.dot(a, bs, acc_s)
            x_ptrs += BLOCK_K
            wr_ptrs += BLOCK_K * stride_wn
            ws_ptrs += BLOCK_K * stride_wn
    else:
        wr_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
        ws_ptrs = w_ptr + (N + offs_sw)[:, None] * stride_wn + offs_k[None, :]
        for _i in range(HIDDEN // BLOCK_K):
            a = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
            br = tl.load(wr_ptrs)
            bs = tl.load(ws_ptrs, mask=mask_sw[:, None], other=0.0)
            acc_r = tl.dot(a, tl.trans(br), acc_r)
            acc_s = tl.dot(a, tl.trans(bs), acc_s)
            x_ptrs += BLOCK_K
            wr_ptrs += BLOCK_K
            ws_ptrs += BLOCK_K

    if DEBUG_GEMM_ONLY:  # bisect: time the GEMM loop without the gate epilogue
        dummy = tl.sum(acc_r, axis=1) + tl.sum(acc_s, axis=1)
        tl.store(shared_w_ptr + offs_m * S, dummy, mask=mask_m)
        return

    # --- top-k by selection score sigmoid(logit) + bias (ties: smaller id) ---
    sel = tl.sigmoid(acc_r) + tl.load(bias_ptr + offs_n).to(tl.float32)[None, :]
    key = fpval_to_key(sel.to(tl.uint32, bitcast=True))
    packed_key = (key.to(tl.uint64) << 16) | indx_to_key(offs_n, N)[None, :]
    acc = tl.sort(tl.topk(packed_key, A_POW2, dim=1), dim=1, descending=True)

    offs_a = tl.arange(0, A_POW2)
    mask_k = offs_a < K
    y_indices = key_to_indx((acc & 0xFFFF).to(tl.uint32), N)
    sel_val = key_to_fpval((acc >> 16).to(tl.uint32)).to(tl.float32, bitcast=True)

    # --- renorm: sigmoid(raw) recovered as sel - bias; shared from acc_s -----
    gather_idx = tl.where(mask_k[None, :], y_indices.to(tl.int32), 0)
    probs_routed = sel_val - tl.load(bias_ptr + gather_idx).to(tl.float32)
    sig_s = tl.sigmoid(acc_s)
    sh0 = tl.sum(tl.where(offs_sw[None, :] == 0, sig_s, 0.0), axis=1)
    sh1 = tl.sum(tl.where(offs_sw[None, :] == 1, sig_s, 0.0), axis=1)
    active = tl.where(
        mask_k[None, :],
        probs_routed,
        tl.where(offs_a[None, :] == K, sh0[:, None], sh1[:, None]),
    )
    weights = active / tl.sum(active, axis=1, keep_dims=True)
    weights *= (route_scale * tl.load(global_scale_ptr)).to(weights.dtype)

    mask_rk = mask_m[:, None] & mask_k[None, :]
    if RETURN_PACKED_TOPK:
        weights_bits = weights.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
        packed = (y_indices.to(tl.int32) << 16) | weights_bits
        tl.store(
            packed_indices_ptr + offs_m[:, None] * K + offs_a[None, :],
            packed,
            mask=mask_rk,
        )
    else:
        tl.store(
            routed_w_ptr + offs_m[:, None] * K + offs_a[None, :], weights, mask=mask_rk
        )
        tl.store(
            indices_ptr + offs_m[:, None] * K + offs_a[None, :], y_indices, mask=mask_rk
        )
    offs_sh = offs_a - K
    mask_sh = mask_m[:, None] & (offs_sh[None, :] >= 0)
    tl.store(shared_w_ptr + offs_m[:, None] * S + offs_sh[None, :], weights, mask=mask_sh)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def gate_gemm_topk(
    x: torch.Tensor,
    weight: torch.Tensor,  # [>=N+S, hidden], or [hidden, >=N+S] if transposed_w
    bias: torch.Tensor,
    global_scale: torch.Tensor,
    k: int,
    n_shared_experts: int,
    route_scale: float,
    *,
    return_packed_topk: bool = False,
    block_m: int = 32,
    block_k: int = 256,
    num_warps: int = 8,
    num_stages: int = 3,
    debug_gemm_only: bool = False,
    transposed_w: bool = False,
):
    """Fused gate GEMM + top-k + renorm; contract of ``sigmoid_gate_topk_renorm``.

    ``x`` [tokens, hidden] bf16, ``weight`` the row-padded gate weight
    [>= n_routed + n_shared, hidden] bf16 (only the first 258 rows are read).
    Returns ``(routed_w, indices, shared_w, packed)``.
    """
    assert x.ndim == 2 and x.stride(1) == 1, f"{x.shape=} {x.stride()=}"
    assert weight.ndim == 2 and weight.stride(1) == 1
    M, hidden = x.shape
    N = 256  # routed experts: power of 2 required by the in-register topk
    S = n_shared_experts
    if transposed_w:
        assert weight.shape[0] == hidden and weight.shape[1] >= N + S
    else:
        assert weight.shape[1] == hidden and weight.shape[0] >= N + S
    assert hidden % block_k == 0
    assert bias.numel() == N and bias.stride(-1) == 1
    assert k + S == triton.next_power_of_2(k + S), f"need K+S power of 2: {k=} {S=}"
    assert S == 2, f"kernel specialized for 2 shared experts: {S=}"

    shared_w = torch.empty((M, S), dtype=torch.float32, device=x.device)
    if return_packed_topk:
        packed = torch.empty((M, k), dtype=torch.int32, device=x.device)
        routed_w = indices = None
        routed_w_arg = indices_arg = packed_arg = packed
    else:
        routed_w = torch.empty((M, k), dtype=torch.float32, device=x.device)
        indices = torch.empty((M, k), dtype=torch.int32, device=x.device)
        packed = None
        routed_w_arg, indices_arg, packed_arg = routed_w, indices, indices

    if M == 0:
        return routed_w, indices, shared_w, packed

    kwargs = {"num_warps": num_warps, "num_stages": num_stages}
    if is_arch_support_pdl():
        kwargs.update({"ENABLE_PDL": True, "launch_pdl": True})

    grid = (triton.cdiv(M, block_m),)
    _gate_gemm_topk_kernel[grid](
        x,
        weight,
        bias,
        global_scale,
        routed_w_arg,
        shared_w,
        indices_arg,
        packed_arg,
        route_scale,
        M,
        x.stride(0),
        weight.stride(0),
        HIDDEN=hidden,
        N=N,
        S=S,
        K=k,
        A_POW2=triton.next_power_of_2(k + S),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        TRANSPOSED_W=transposed_w,
        RETURN_PACKED_TOPK=return_packed_topk,
        DEBUG_GEMM_ONLY=debug_gemm_only,
        **kwargs,
    )
    return routed_w, indices, shared_w, packed
