# SPDX-License-Identifier: Apache-2.0
"""Fused prologue/epilogue kernels for the Kimi-K3 MoE sub-layer on ROCm
(decode / target-verify, M = tokens <= a few hundred).

shared_down_topk: one launch that does two independent jobs of the fused
  front's consumers ("horizontal" fusion; no cross-CTA dependency):
    * CTAs [0, n_gemm): shared-expert down GEMM with the SiTU activation in
      its prologue:  out[M, N] = bf16(situ(gate) * up) @ w_down[N, I]^T
      (replaces sglang::situ_and_mul + the hipBLASLt down GEMM)
    * CTAs [n_gemm, n_gemm + M): sigmoid + correction-bias top-k of the router
      logits, one CTA per token (same math as k3_route_sort._k3_topk_kernel /
      aiter biased_grouped_topk with one group).

up_norm_add3: the latent RMSNorm, the replicated latent->hidden up_proj and the
  tail add in one launch:
    out = bf16(bf16(bf16(rmsnorm(x) @ w_up^T) + shared) + prefix)
  rmsnorm(x) @ w^T = rstd[m] * (x * g) @ w^T, so the GEMM consumes bf16(x * g)
  and rstd is applied in the fp32 epilogue (the per-row sum of squares is
  accumulated from the same x tiles the GEMM streams). Optional split-K: every
  split writes an fp32 partial + its partial sum of squares, and the last
  arriving CTA of an output tile (atomic ticket) reduces them in a fixed order
  and runs the epilogue (deterministic).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# --------------------------------------------------------------------------
# shared-expert down GEMM (SiTU prologue) + router top-k
# --------------------------------------------------------------------------


@triton.jit
def _score_key(s, idx):
    # order-preserving int64 key: higher score first, lower expert id on ties
    b = s.to(tl.int32, bitcast=True)
    k = tl.where(b >= 0, b, b ^ 0x7FFFFFFF).to(tl.int64)
    return (k << 32) | (0xFFFF - idx).to(tl.int64)


@triton.jit(do_not_specialize=["M", "stride_gu", "stride_out", "stride_lg"])
def _k3_shared_topk_kernel(
    gu_ptr,  # [M, 2*I] bf16 (gate | up), row stride stride_gu
    stride_gu,
    wd_ptr,  # [N, I] bf16 contiguous
    out_ptr,  # [M, N] bf16, row stride stride_out
    stride_out,
    lg_ptr,  # [M, E] router logits, row stride stride_lg
    stride_lg,
    bias_ptr,  # [E] correction bias
    topk_ids_ptr,  # [M, TOPK] i32 out
    topk_w_ptr,  # [M, TOPK] fp32 out
    routed_scale,
    M,
    beta,
    inv_beta,
    lin_beta,
    inv_lin_beta,
    N: tl.constexpr,
    I: tl.constexpr,
    HAS_LIN: tl.constexpr,
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
    NT_N: tl.constexpr = N // BN
    n_gemm = NT_N * tl.cdiv(M, BM)
    if pid < n_gemm:
        pid_n = pid % NT_N
        pid_m = pid // NT_N
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        mm = offs_m < M
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in tl.range(0, I, BK):
            offs_k = k0 + tl.arange(0, BK)
            row = gu_ptr + offs_m[:, None] * stride_gu + offs_k[None, :]
            g = tl.load(row, mask=mm[:, None], other=0.0).to(tl.float32)
            u = tl.load(row + I, mask=mm[:, None], other=0.0).to(tl.float32)
            w = tl.load(wd_ptr + offs_n[:, None] * I + offs_k[None, :])
            gate = beta * libdevice.tanh(g * inv_beta) * (1.0 / (1.0 + tl.exp(-g)))
            if HAS_LIN:
                u = lin_beta * libdevice.tanh(u * inv_lin_beta)
            a = (gate * u).to(wd_ptr.dtype.element_ty)
            acc = tl.dot(a, tl.trans(w), acc)
        tl.store(
            out_ptr + offs_m[:, None] * stride_out + offs_n[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=mm[:, None],
        )
    else:
        if DO_TOPK:
            t = pid - n_gemm
            offs_e = tl.arange(0, E_POW2)
            me = offs_e < E
            lg = tl.load(lg_ptr + t * stride_lg + offs_e, mask=me, other=0.0).to(
                tl.float32
            )
            # the bias is rounded to the logits dtype first, as aiter's router does
            bias = (
                tl.load(bias_ptr + offs_e, mask=me, other=0.0)
                .to(lg_ptr.dtype.element_ty)
                .to(tl.float32)
            )
            biased = tl.where(me, tl.sigmoid(lg) + bias, float("-inf"))
            top = tl.topk(_score_key(biased, offs_e), TOPK, dim=0)  # descending
            ids = (0xFFFF - (top & 0xFFFF)).to(tl.int32)
            wt = tl.sigmoid(tl.load(lg_ptr + t * stride_lg + ids).to(tl.float32))
            if RENORM:
                wt = wt / tl.sum(wt, axis=0)
            wt = wt * routed_scale
            offs_k = tl.arange(0, TOPK)
            tl.store(topk_ids_ptr + t * TOPK + offs_k, ids)
            tl.store(topk_w_ptr + t * TOPK + offs_k, wt)


def _shared_cfg(m: int):
    # (BM, BN, BK, num_warps); tuned on MI355X for N=7168, I=768
    if m <= 16:
        return 16, 32, 128, 8, 2
    if m <= 32:
        return 32, 32, 256, 4
    if m <= 64:
        return 64, 32, 128, 4
    return 64, 64, 128, 4


_SHARED_CFG_OVERRIDE: Optional[tuple] = None


def shared_down_topk(
    gate_up: torch.Tensor,  # [M, 2I] bf16, row-strided ok
    w_down: torch.Tensor,  # [N, I] bf16
    out: torch.Tensor,  # [M, N] bf16
    beta: float,
    linear_beta: Optional[float],
    router_logits: Optional[torch.Tensor] = None,  # [M, E], row-strided ok
    correction_bias: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,  # [M, TOPK] i32 out
    topk_weights: Optional[torch.Tensor] = None,  # [M, TOPK] fp32 out
    renormalize: bool = True,
    routed_scale: float = 1.0,
    cfg: Optional[tuple] = None,
) -> None:
    M, two_i = gate_up.shape
    I = two_i // 2
    N = w_down.shape[0]
    assert w_down.shape[1] == I and w_down.is_contiguous()
    assert gate_up.stride(1) == 1 and out.stride(1) == 1
    bm, bn, bk, nw, *rest = cfg or _SHARED_CFG_OVERRIDE or _shared_cfg(M)
    ns = rest[0] if rest else 2
    assert N % bn == 0 and I % bk == 0
    do_topk = router_logits is not None
    if do_topk:
        E = router_logits.shape[1]
        topk = topk_ids.shape[1]
        assert router_logits.stride(1) == 1 and E < 0xFFFF and topk & (topk - 1) == 0
    else:
        E, topk = 1, 1
        router_logits = correction_bias = topk_ids = topk_weights = gate_up
    grid = ((N // bn) * triton.cdiv(M, bm) + (M if do_topk else 0),)
    lin = linear_beta is not None
    _k3_shared_topk_kernel[grid](
        gate_up,
        gate_up.stride(0),
        w_down,
        out,
        out.stride(0),
        router_logits,
        router_logits.stride(0),
        correction_bias,
        topk_ids,
        topk_weights,
        float(routed_scale),
        M,
        float(beta),
        1.0 / float(beta),
        float(linear_beta) if lin else 1.0,
        1.0 / float(linear_beta) if lin else 1.0,
        N=N,
        I=I,
        HAS_LIN=lin,
        BM=bm,
        BN=bn,
        BK=bk,
        DO_TOPK=do_topk,
        E=E,
        E_POW2=triton.next_power_of_2(E),
        TOPK=topk,
        RENORM=renormalize,
        num_warps=nw,
        num_stages=ns,
    )


# --------------------------------------------------------------------------
# latent RMSNorm + up_proj + add3
# --------------------------------------------------------------------------


@triton.jit
def _up_epilogue(
    acc,
    ss,
    offs_m,
    offs_n,
    mm,
    sh_ptr,
    stride_sh,
    pf_ptr,
    stride_pf,
    out_ptr,
    stride_out,
    eps,
    K: tl.constexpr,
    HAS_NORM: tl.constexpr,
    HAS_SH: tl.constexpr,
    HAS_PF: tl.constexpr,
):
    if HAS_NORM:
        rstd = 1.0 / tl.sqrt(ss / K + eps)
        acc = acc * rstd[:, None]
    y = acc.to(out_ptr.dtype.element_ty)
    m2 = mm[:, None]
    if HAS_SH:
        s = tl.load(sh_ptr + offs_m[:, None] * stride_sh + offs_n[None, :], mask=m2, other=0.0)
        y = (y.to(tl.float32) + s.to(tl.float32)).to(out_ptr.dtype.element_ty)
    if HAS_PF:
        p = tl.load(pf_ptr + offs_m[:, None] * stride_pf + offs_n[None, :], mask=m2, other=0.0)
        y = (y.to(tl.float32) + p.to(tl.float32)).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_m[:, None] * stride_out + offs_n[None, :], y, mask=m2)


@triton.jit(do_not_specialize=["M", "stride_x", "stride_sh", "stride_pf", "stride_out"])
def _k3_up_norm_add3_kernel(
    x_ptr,  # [M, K] bf16 latent (pre-norm), row stride stride_x
    stride_x,
    g_ptr,  # [K] norm weight
    w_ptr,  # [N, K] bf16 contiguous
    sh_ptr,  # [M, N] bf16 shared-expert output
    stride_sh,
    pf_ptr,  # [M, N] bf16 delayed prefix sum
    stride_pf,
    out_ptr,  # [M, N] bf16
    stride_out,
    part_ptr,  # fp32 [SPLIT, M_PAD, N] split-K partials
    ssp_ptr,  # fp32 [tiles, SPLIT, BM] partial sums of squares
    cnt_ptr,  # i32 [tiles] arrival counters (0 between launches)
    M,
    eps,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_NORM: tl.constexpr,
    HAS_SH: tl.constexpr,
    HAS_PF: tl.constexpr,
    SPLIT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)
    KS: tl.constexpr = K // SPLIT
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mm = offs_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    ss = tl.zeros((BM,), dtype=tl.float32)
    for kk in tl.range(0, KS, BK):
        offs_k = pid_s * KS + kk + tl.arange(0, BK)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_x + offs_k[None, :],
            mask=mm[:, None],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :])
        if HAS_NORM:
            ss += tl.sum(x * x, axis=1)
            g = tl.load(g_ptr + offs_k).to(tl.float32)
            x = x * g[None, :]
        acc = tl.dot(x.to(w_ptr.dtype.element_ty), tl.trans(w), acc)
    if SPLIT == 1:
        _up_epilogue(acc, ss, offs_m, offs_n, mm, sh_ptr, stride_sh, pf_ptr, stride_pf,
                     out_ptr, stride_out, eps, K, HAS_NORM, HAS_SH, HAS_PF)
    else:
        NT_N: tl.constexpr = N // BN
        tile = pid_m * NT_N + pid_n
        m_pad = tl.num_programs(2) * BM
        # write-through partials + a relaxed ticket (no L2 writeback/invalidate
        # fences: those cost several us per CTA on gfx950)
        tl.store(part_ptr + (pid_s * m_pad + offs_m[:, None]) * N + offs_n[None, :], acc,
                 cache_modifier=".wt")
        if HAS_NORM:
            tl.store(ssp_ptr + (tile * SPLIT + pid_s) * BM + tl.arange(0, BM), ss, cache_modifier=".wt")
        tl.debug_barrier()
        ticket = tl.atomic_add(cnt_ptr + tile, 1, sem="relaxed", scope="gpu")
        if ticket == SPLIT - 1:
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            ss = tl.zeros((BM,), dtype=tl.float32)
            for s in tl.static_range(SPLIT):
                acc += tl.load(
                    part_ptr + (s * m_pad + offs_m[:, None]) * N + offs_n[None, :],
                    cache_modifier=".cv",
                )
                if HAS_NORM:
                    ss += tl.load(ssp_ptr + (tile * SPLIT + s) * BM + tl.arange(0, BM), cache_modifier=".cv")
            _up_epilogue(acc, ss, offs_m, offs_n, mm, sh_ptr, stride_sh, pf_ptr, stride_pf,
                         out_ptr, stride_out, eps, K, HAS_NORM, HAS_SH, HAS_PF)
            tl.atomic_xchg(cnt_ptr + tile, 0, sem="relaxed", scope="gpu")


def _up_cfg(m: int):
    # (BM, BN, BK, SPLIT, num_warps); tuned on MI355X for N=7168, K=3584
    if m <= 16:
        return 16, 32, 256, 1, 4
    if m <= 32:
        return 32, 32, 256, 1, 4
    if m <= 64:
        return 64, 64, 128, 2, 4
    return 64, 64, 128, 2, 4


_UP_CFG_OVERRIDE: Optional[tuple] = None

# split-K scratch, shared by every layer (one stream): allocate before graph capture
_WS: dict = {}
_WS_ROWS = 1024  # split * padded M
_WS_SPLIT_MAX = 8
_WS_TILES = 4096
_WS_SS = 1 << 18  # floats


def _dev_key(device) -> int:
    device = torch.device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


def ensure_workspace(device, n: int = 7168) -> None:
    """Allocate the split-K scratch (call once, outside CUDA-graph capture)."""
    key = (_dev_key(device), n)
    if key in _WS:
        return
    part = torch.empty(_WS_ROWS * n, dtype=torch.float32, device=device)
    ssp = torch.empty(_WS_SS, dtype=torch.float32, device=device)
    cnt = torch.zeros(_WS_TILES, dtype=torch.int32, device=device)
    _WS[key] = (part, ssp, cnt)


def up_norm_add3(
    x: torch.Tensor,  # [M, K] bf16 latent (pre-norm)
    norm_weight: Optional[torch.Tensor],  # [K] or None (no norm)
    eps: float,
    w_up: torch.Tensor,  # [N, K] bf16
    shared: Optional[torch.Tensor],  # [M, N] bf16
    prefix: Optional[torch.Tensor],  # [M, N] bf16
    out: Optional[torch.Tensor] = None,
    cfg: Optional[tuple] = None,
) -> torch.Tensor:
    M, K = x.shape
    N = w_up.shape[0]
    assert w_up.shape[1] == K and w_up.is_contiguous() and x.stride(1) == 1
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    bm, bn, bk, split, nw, *rest = cfg or _UP_CFG_OVERRIDE or _up_cfg(M)
    ns = rest[0] if rest else 2
    m_tiles = triton.cdiv(M, bm)
    if split > 1:
        ws = _WS.get((_dev_key(x.device), N))
        if (
            ws is None
            or split > _WS_SPLIT_MAX
            or split * m_tiles * bm > _WS_ROWS
            or bm > 128
            or m_tiles * (N // bn) > _WS_TILES
            or m_tiles * (N // bn) * split * bm > _WS_SS
        ):
            split = 1
    if split == 1:
        part = ssp = cnt = x
    else:
        part, ssp, cnt = ws
    assert N % bn == 0 and K % (split * bk) == 0
    has_sh, has_pf = shared is not None, prefix is not None
    _k3_up_norm_add3_kernel[(N // bn, split, m_tiles)](
        x,
        x.stride(0),
        norm_weight if norm_weight is not None else x,
        w_up,
        shared if has_sh else x,
        shared.stride(0) if has_sh else 0,
        prefix if has_pf else x,
        prefix.stride(0) if has_pf else 0,
        out,
        out.stride(0),
        part,
        ssp,
        cnt,
        M,
        float(eps),
        N=N,
        K=K,
        HAS_NORM=norm_weight is not None,
        HAS_SH=has_sh,
        HAS_PF=has_pf,
        SPLIT=split,
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=nw,
        num_stages=ns,
    )
    return out


# --------------------------------------------------------------------------
# rstd-scale + add3 tail behind an unfused up GEMM (norm weight folded into w_up)
# --------------------------------------------------------------------------


@triton.jit(do_not_specialize=["stride_up", "stride_x", "stride_sh", "stride_pf", "stride_out"])
def _k3_norm_add3_kernel(
    up_ptr,  # [M, N] fp32/bf16 = x @ (w_up * norm_w)^T
    stride_up,
    x_ptr,  # [M, K] latent (pre-norm)
    stride_x,
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
    x = tl.load(x_ptr + m * stride_x + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
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
    has_pf = prefix is not None
    _k3_norm_add3_kernel[(M, triton.cdiv(N, block))](
        up, up.stride(0), x, x.stride(0), shared, shared.stride(0),
        prefix if has_pf else shared, prefix.stride(0) if has_pf else 0,
        out, out.stride(0), float(eps),
        N=N, K=K, K_POW2=triton.next_power_of_2(K), BLOCK=block, HAS_PF=has_pf, num_warps=4,
    )
    return out
