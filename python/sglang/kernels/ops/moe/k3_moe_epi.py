# SPDX-License-Identifier: Apache-2.0
"""Fused glue kernels around the Kimi-K3 MoE sub-layer on ROCm (decode /
target-verify, M <= a few hundred tokens).

shared_down_topk: SiTU, then one launch with two independent CTA ranges:
  the shared-expert down GEMM and the sigmoid + correction-bias router top-k
  (same math as aiter biased_grouped_topk with one group).
situ_topk + shared_gemm_sort_quant: the same work regrouped by dependency for
  M * topk <= 256, one launch fewer per layer:
    [SiTU | top-k] -> [shared down GEMM | sort + mxfp8 quant] -> stage 1
  The sort runs inside the aiter MoE runner, so the model hands the pending
  GEMM over in moe_sorting_small.K3SortPlan.shared_gemm.
norm_add3: rstd scale + 3-way add behind the up GEMM (norm weight folded into
  w_up).
All variants share the bodies below, so their outputs are bit-identical.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.moe.k3_route_sort import SORT_QCHUNK
from sglang.kernels.ops.moe.k3_route_sort_mid import _aiter_biased_topk

_SITU_BLOCK = 256


@triton.jit
def _situ_block(
    gu_ptr,
    stride_gu,
    act_ptr,
    m,
    blk,
    beta,
    inv_beta,
    lin_beta,
    inv_lin_beta,
    I: tl.constexpr,
    HAS_LIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # act[m, blk] = situ(gate) * up, gate_up = [M, gate | up]
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mk = offs < I
    g = tl.load(gu_ptr + m * stride_gu + offs, mask=mk, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + m * stride_gu + I + offs, mask=mk, other=0.0).to(tl.float32)
    a = beta * libdevice.tanh(g * inv_beta) * (1.0 / (1.0 + tl.exp(-g)))
    if HAS_LIN:
        u = lin_beta * libdevice.tanh(u * inv_lin_beta)
    tl.store(act_ptr + m * I + offs, (a * u).to(act_ptr.dtype.element_ty), mask=mk)


@triton.jit
def _topk_program(
    t,
    lg_ptr,
    stride_lg,
    bias_ptr,
    topk_ids_ptr,
    topk_w_ptr,
    routed_scale,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    RENORM: tl.constexpr,
):
    offs_e = tl.arange(0, E_POW2)
    ids, wt = _aiter_biased_topk(
        lg_ptr + t * stride_lg,
        bias_ptr,
        offs_e,
        offs_e < E,
        E_POW2,
        TOPK,
        RENORM,
        routed_scale,
    )
    offs_k = tl.arange(0, TOPK)
    tl.store(topk_ids_ptr + t * TOPK + offs_k, ids)
    tl.store(topk_w_ptr + t * TOPK + offs_k, wt)


@triton.jit
def _down_gemm_program(
    pid,
    act_ptr,
    stride_act,
    wd_ptr,
    out_ptr,
    stride_out,
    M,
    N: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # out[M, N] = act[M, I] @ wd[N, I]^T, one (BM, BN) tile
    NT_N: tl.constexpr = N // BN
    offs_m = (pid // NT_N) * BM + tl.arange(0, BM)
    offs_n = (pid % NT_N) * BN + tl.arange(0, BN)
    mm = offs_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, I, BK):
        offs_k = k0 + tl.arange(0, BK)
        w = tl.load(wd_ptr + offs_n[:, None] * I + offs_k[None, :])
        a = tl.load(
            act_ptr + offs_m[:, None] * stride_act + offs_k[None, :],
            mask=mm[:, None],
            other=0.0,
        )
        acc = tl.dot(a, tl.trans(w), acc)
    tl.store(
        out_ptr + offs_m[:, None] * stride_out + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mm[:, None],
    )


@triton.jit(do_not_specialize=["M", "stride_gu"])
def _k3_situ_kernel(
    gu_ptr,
    stride_gu,
    act_ptr,
    M,
    beta,
    inv_beta,
    lin_beta,
    inv_lin_beta,
    I: tl.constexpr,
    HAS_LIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _situ_block(
        gu_ptr,
        stride_gu,
        act_ptr,
        tl.program_id(0),
        tl.program_id(1),
        beta,
        inv_beta,
        lin_beta,
        inv_lin_beta,
        I,
        HAS_LIN,
        BLOCK,
    )


@triton.jit(do_not_specialize=["M", "stride_act", "stride_out", "stride_lg"])
def _k3_shared_topk_kernel(
    act_ptr,
    stride_act,
    wd_ptr,
    out_ptr,
    stride_out,
    lg_ptr,
    stride_lg,
    bias_ptr,
    topk_ids_ptr,
    topk_w_ptr,
    routed_scale,
    M,
    N: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DO_TOPK: tl.constexpr,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    RENORM: tl.constexpr,
):
    pid = tl.program_id(0)
    n_gemm = (N // BN) * tl.cdiv(M, BM)
    if pid < n_gemm:
        _down_gemm_program(
            pid, act_ptr, stride_act, wd_ptr, out_ptr, stride_out, M, N, I, BM, BN, BK
        )
    elif DO_TOPK:
        _topk_program(
            pid - n_gemm,
            lg_ptr,
            stride_lg,
            bias_ptr,
            topk_ids_ptr,
            topk_w_ptr,
            routed_scale,
            E,
            E_POW2,
            TOPK,
            RENORM,
        )


@triton.jit(do_not_specialize=["M", "stride_gu", "stride_lg"])
def _k3_situ_topk_kernel(
    gu_ptr,
    stride_gu,
    act_ptr,
    M,
    beta,
    inv_beta,
    lin_beta,
    inv_lin_beta,
    lg_ptr,
    stride_lg,
    bias_ptr,
    topk_ids_ptr,
    topk_w_ptr,
    routed_scale,
    I: tl.constexpr,
    HAS_LIN: tl.constexpr,
    BLOCK: tl.constexpr,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    RENORM: tl.constexpr,
):
    pid = tl.program_id(0)
    NB: tl.constexpr = (I + BLOCK - 1) // BLOCK
    if pid < M * NB:
        _situ_block(
            gu_ptr,
            stride_gu,
            act_ptr,
            pid // NB,
            pid % NB,
            beta,
            inv_beta,
            lin_beta,
            inv_lin_beta,
            I,
            HAS_LIN,
            BLOCK,
        )
    else:
        _topk_program(
            pid - M * NB,
            lg_ptr,
            stride_lg,
            bias_ptr,
            topk_ids_ptr,
            topk_w_ptr,
            routed_scale,
            E,
            E_POW2,
            TOPK,
            RENORM,
        )


@triton.jit(do_not_specialize=["M", "stride_out", "stride_qx"])
def _k3_shared_sortq_kernel(
    # shared down GEMM (pre-activated): out[M, N] = act[M, I] @ wd[N, I]^T
    act_ptr,
    wd_ptr,
    out_ptr,
    stride_out,
    # sort + mxfp8 quant (k3_route_sort._k3_sort_quant_kernel)
    topk_ids_ptr,
    topk_weights_ptr,
    sorted_ids_ptr,
    sorted_weights_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    moe_buf_ptr,
    qx_ptr,
    stride_qx,
    qout_ptr,
    qscale_ptr,
    M,
    N: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    P_POW2: tl.constexpr,
    N_COLS: tl.constexpr,
    QCHUNK: tl.constexpr,
    SCALEN_PAD: tl.constexpr,
    MOE_BUF_ZERO: tl.constexpr,
):
    # sort/quant programs first: they gate stage 1, the GEMM only the AR
    CHUNKS: tl.constexpr = N_COLS // QCHUNK
    n_sort = M * CHUNKS
    pid = tl.program_id(0)
    NT_N: tl.constexpr = N // BN
    if pid >= n_sort:
        # == _k3_shared_topk_kernel GEMM branch with PRE_ACT
        gpid = pid - n_sort
        pid_n = gpid % NT_N
        pid_m = gpid // NT_N
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        mm = offs_m < M
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in tl.range(0, I, BK):
            offs_k = k0 + tl.arange(0, BK)
            w = tl.load(wd_ptr + offs_n[:, None] * I + offs_k[None, :])
            a = tl.load(
                act_ptr + offs_m[:, None] * I + offs_k[None, :],
                mask=mm[:, None],
                other=0.0,
            )
            acc = tl.dot(a, tl.trans(w), acc)
        tl.store(
            out_ptr + offs_m[:, None] * stride_out + offs_n[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=mm[:, None],
        )
    else:
        # == k3_route_sort._k3_sort_quant_kernel, program pid
        spid = pid
        t = spid // CHUNKS
        c0 = (spid % CHUNKS) * QCHUNK
        P = M * TOPK
        offs_p = tl.arange(0, P_POW2)
        mask_p = offs_p < P
        SENT: tl.constexpr = 0x7FFFFFFF
        e = tl.load(topk_ids_ptr + offs_p, mask=mask_p, other=SENT)
        srt = tl.sort(e)
        prv = tl.gather(srt, tl.maximum(offs_p - 1, 0), axis=0)
        first = ((offs_p == 0) | (srt != prv)) & (srt != SENT)
        offs_k = tl.arange(0, TOPK)
        ek = tl.load(topk_ids_ptr + t * TOPK + offs_k)
        match = e[None, :] == ek[:, None]
        rank_k = tl.sum(tl.where(match & (offs_p[None, :] < t * TOPK), 1, 0), axis=1)
        bb_k = tl.sum(
            tl.where(first[None, :] & (srt[None, :] < ek[:, None]), 1, 0), axis=1
        )
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
            tl.store(
                sorted_ids_ptr + pad_at,
                tl.full((TOPK, BLOCK_SIZE), (TOPK << 24), tl.int32) + M,
                mask=pm,
            )
            tl.store(
                sorted_weights_ptr + pad_at,
                tl.zeros((TOPK, BLOCK_SIZE), tl.float32),
                mask=pm,
            )
            if t == 0:
                num_valid = tl.sum(first.to(tl.int32), axis=0) * BLOCK_SIZE
                tl.store(
                    num_valid_ids_ptr + tl.arange(0, 2),
                    tl.where(tl.arange(0, 2) == 0, num_valid, M),
                )
        offs_q = tl.arange(0, QCHUNK)
        x = tl.load(qx_ptr + t * stride_qx + c0 + offs_q).to(tl.float32)
        x2 = tl.reshape(x, (QCHUNK // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(x2), axis=1), 1e-10)
        bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
        exp = (bits >> 23) & 0xFF
        exp = tl.where((bits & 0x7FFFFF) != 0, exp + 1, exp)
        scale = (exp << 23).to(tl.float32, bitcast=True)
        q = tl.clamp(x2 / scale[:, None], -448.0, 448.0)
        tl.store(
            qout_ptr + t * N_COLS + c0 + offs_q,
            tl.reshape(q, (QCHUNK,)).to(qout_ptr.dtype.element_ty),
        )
        if MOE_BUF_ZERO:
            tl.store(
                moe_buf_ptr + t * N_COLS + c0 + offs_q,
                tl.zeros((QCHUNK,), moe_buf_ptr.dtype.element_ty),
            )
        offs_g = tl.arange(0, QCHUNK // 32)
        y = c0 // 32 + offs_g
        base_sw = (
            (dest_k // 32) * (SCALEN_PAD * 32) + (dest_k % 16) * 4 + (dest_k % 32) // 16
        )
        sw = (
            base_sw[:, None]
            + ((y // 8) * 256 + (y % 4) * 64 + ((y % 8) // 4) * 2)[None, :]
        )
        tl.store(
            qscale_ptr + sw,
            tl.broadcast_to(exp[None, :], (TOPK, QCHUNK // 32)).to(tl.uint8),
        )


def _gemm_cfg(m: int):
    # (BM, BN, BK, num_warps, num_stages); MI355X sweep for N=7168, I=768
    if m <= 16:
        return 16, 32, 256, 4, 3
    if m <= 32:
        return 32, 16, 256, 4, 2
    return 64, 32, 256, 8, 2


def _situ_args(beta: float, linear_beta: Optional[float]):
    lin = 1.0 if linear_beta is None else float(linear_beta)
    return float(beta), 1.0 / float(beta), lin, 1.0 / lin


def _launch_shared_topk(act, w_down, out, topk=None):
    """out = act @ w_down^T, plus the router top-k CTAs when topk =
    (router_logits, correction_bias, topk_ids, topk_weights, renormalize,
    routed_scale) is given."""
    M, I = act.shape
    N = w_down.shape[0]
    assert w_down.shape[1] == I and w_down.is_contiguous() and out.stride(1) == 1
    bm, bn, bk, nw, ns = _gemm_cfg(M)
    assert N % bn == 0 and I % bk == 0
    if topk is None:
        lg, bias, ids, wts, renorm, scale = act, act, act, act, False, 1.0
        E = n_topk = 1
    else:
        lg, bias, ids, wts, renorm, scale = topk
        E, n_topk = lg.shape[1], ids.shape[1]
        assert lg.stride(1) == 1 and E < 0xFFFF and n_topk & (n_topk - 1) == 0
    grid = ((N // bn) * triton.cdiv(M, bm) + (0 if topk is None else M),)
    _k3_shared_topk_kernel[grid](
        act,
        act.stride(0),
        w_down,
        out,
        out.stride(0),
        lg,
        lg.stride(0),
        bias,
        ids,
        wts,
        float(scale),
        M,
        N=N,
        I=I,
        BM=bm,
        BN=bn,
        BK=bk,
        DO_TOPK=topk is not None,
        E=E,
        E_POW2=triton.next_power_of_2(E),
        TOPK=n_topk,
        RENORM=renorm,
        num_warps=nw,
        num_stages=ns,
    )


def shared_down_topk(
    gate_up: torch.Tensor,  # [M, 2I] bf16, row-strided ok
    w_down: torch.Tensor,  # [N, I] bf16
    out: torch.Tensor,  # [M, N] bf16
    beta: float,
    linear_beta: Optional[float],
    router_logits: torch.Tensor,  # [M, E], row-strided ok
    correction_bias: torch.Tensor,
    topk_ids: torch.Tensor,  # [M, TOPK] i32 out
    topk_weights: torch.Tensor,  # [M, TOPK] fp32 out
    renormalize: bool = True,
    routed_scale: float = 1.0,
) -> None:
    M, I = gate_up.shape[0], gate_up.shape[1] // 2
    assert gate_up.stride(1) == 1
    act = torch.empty((M, I), dtype=w_down.dtype, device=gate_up.device)
    _k3_situ_kernel[(M, triton.cdiv(I, _SITU_BLOCK))](
        gate_up,
        gate_up.stride(0),
        act,
        M,
        *_situ_args(beta, linear_beta),
        I=I,
        HAS_LIN=linear_beta is not None,
        BLOCK=_SITU_BLOCK,
        num_warps=2,
    )
    _launch_shared_topk(
        act,
        w_down,
        out,
        (
            router_logits,
            correction_bias,
            topk_ids,
            topk_weights,
            renormalize,
            routed_scale,
        ),
    )


def shared_gemm_preact(act, w_down, out) -> None:
    _launch_shared_topk(act, w_down, out)


def situ_topk(
    gate_up: torch.Tensor,  # [M, 2I]
    act: torch.Tensor,  # [M, I] out
    beta: float,
    linear_beta: Optional[float],
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    renormalize: bool = True,
    routed_scale: float = 1.0,
) -> None:
    M, I = gate_up.shape[0], gate_up.shape[1] // 2
    E, topk = router_logits.shape[1], topk_ids.shape[1]
    assert gate_up.stride(1) == 1 and router_logits.stride(1) == 1
    assert act.is_contiguous() and E < 0xFFFF and topk & (topk - 1) == 0
    _k3_situ_topk_kernel[(M * triton.cdiv(I, _SITU_BLOCK) + M,)](
        gate_up,
        gate_up.stride(0),
        act,
        M,
        *_situ_args(beta, linear_beta),
        router_logits,
        router_logits.stride(0),
        correction_bias,
        topk_ids,
        topk_weights,
        float(routed_scale),
        I=I,
        HAS_LIN=linear_beta is not None,
        BLOCK=_SITU_BLOCK,
        E=E,
        E_POW2=triton.next_power_of_2(E),
        TOPK=topk,
        RENORM=renormalize,
        num_warps=4,
    )


def shared_gemm_sort_quant(
    pending,  # (act, w_down, out) from K3SortPlan.shared_gemm
    topk_ids,
    topk_weights,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
    block_size,
    mx_quant_input,
    qout,
    qscale,
) -> None:
    """The pending shared down GEMM + k3_route_sort.sort_quant in one launch."""
    act, w_down, out = pending
    M, I = act.shape
    N = w_down.shape[0]
    m, topk = topk_ids.shape
    n = mx_quant_input.shape[1]
    assert m == M and act.is_contiguous() and w_down.is_contiguous()
    assert out.stride(1) == 1
    bm, bn, bk, nw, ns = _gemm_cfg(M)
    # wider N tiles next to the sort programs (MI355X M=8: 6.3 vs 6.7 us); the
    # K loop is unchanged, so the output stays bit-identical
    bn = 64 if M <= 16 else bn
    n_gemm = (N // bn) * triton.cdiv(M, bm)
    _k3_shared_sortq_kernel[(n_gemm + m * (n // SORT_QCHUNK),)](
        act,
        w_down,
        out,
        out.stride(0),
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
        M,
        N=N,
        I=I,
        BM=bm,
        BN=bn,
        BK=bk,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        P_POW2=triton.next_power_of_2(m * topk),
        N_COLS=n,
        QCHUNK=SORT_QCHUNK,
        SCALEN_PAD=qscale.shape[1],
        MOE_BUF_ZERO=moe_buf.numel() > 0,
        num_warps=nw,
        num_stages=ns,
    )


@triton.jit(
    do_not_specialize=["stride_up", "stride_x", "stride_sh", "stride_pf", "stride_out"]
)
def _k3_norm_add3_kernel(
    up_ptr,
    stride_up,  # [M, N] = x @ (w_up * norm_w)^T
    x_ptr,
    stride_x,  # [M, K] latent (pre-norm)
    sh_ptr,
    stride_sh,
    pf_ptr,
    stride_pf,
    out_ptr,
    stride_out,
    eps,
    N: tl.constexpr,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_PF: tl.constexpr,
):
    m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mn = offs < N
    up = tl.load(up_ptr + m * stride_up + offs, mask=mn, other=0.0).to(tl.float32)
    s = tl.load(sh_ptr + m * stride_sh + offs, mask=mn, other=0.0).to(tl.float32)
    if HAS_PF:
        p = tl.load(pf_ptr + m * stride_pf + offs, mask=mn, other=0.0).to(tl.float32)
    offs_k = tl.arange(0, K_POW2)
    x = tl.load(x_ptr + m * stride_x + offs_k, mask=offs_k < K, other=0.0)
    x = x.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / K + eps)
    y = (up * rstd).to(out_ptr.dtype.element_ty)
    y = (y.to(tl.float32) + s).to(out_ptr.dtype.element_ty)
    if HAS_PF:
        y = (y.to(tl.float32) + p).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + m * stride_out + offs, y, mask=mn)


def norm_add3(
    up: torch.Tensor,  # [M, N] fp32 (preferred) or bf16
    x: torch.Tensor,  # [M, K] latent the GEMM consumed (pre-norm)
    eps: float,
    shared: torch.Tensor,
    prefix: Optional[torch.Tensor],
    out: Optional[torch.Tensor] = None,
    block: int = 1024,
) -> torch.Tensor:
    """out = bf16(bf16(bf16(rstd(x) * up) + shared) + prefix)"""
    M, N = up.shape
    K = x.shape[1]
    if out is None:
        out = torch.empty((M, N), dtype=shared.dtype, device=up.device)
    pf = shared if prefix is None else prefix
    _k3_norm_add3_kernel[(M, triton.cdiv(N, block))](
        up,
        up.stride(0),
        x,
        x.stride(0),
        shared,
        shared.stride(0),
        pf,
        0 if prefix is None else pf.stride(0),
        out,
        out.stride(0),
        float(eps),
        N=N,
        K=K,
        K_POW2=triton.next_power_of_2(K),
        BLOCK=block,
        HAS_PF=prefix is not None,
        num_warps=4,
    )
    return out
