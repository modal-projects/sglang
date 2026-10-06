"""Split-H attention-residual aggregation for Kimi-K3 on ROCm (small batches).

attn_res_hip._agg_kernel runs one CTA per token, so at decode batch sizes
(T = 1..16 tokens, or 8 per request under DFlash verify) it occupies a handful
of the 256 CUs and is bound by one CU's load latency (~9.4 us per call, 192
calls per K3 decode step). This kernel splits each token's H columns over S
CTAs and needs only one cross-CTA barrier:

  phase 1  per (token, chunk): materialize the prefix row (pending residual add,
           bank snapshot), then reduce the chunk's partial score dots
           d_r = <x_r, cw> and its partial Gram matrix G_rs = <x_r, x_s> over the
           R = nvb + 1 rows (bank rows + prefix row) into a workspace;
  barrier  per token (monotonic counter, wrap-safe, no reset across replays);
  phase 2  every CTA sums the S partials, forms the RMSNorm'd scores and the
           softmax weights w, mixes its own chunk from the rows it still holds in
           registers, and applies the output RMSNorm with ||acc||^2 = w^T G w, so
           no second pass over H is needed.

All T*S CTAs must be co-resident for the barrier, which the caller guarantees
by keeping T*S <= the CU count (see plan_splits). Larger batches use the
one-CTA-per-token kernel, which is efficient there.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_NUM_CUS_DEFAULT = 256
_counters: dict[torch.device, torch.Tensor] = {}
_workspaces: dict[tuple, torch.Tensor] = {}
_MAX_T = 1 << 14


def plan_splits(T: int, H: int, num_cus: int = _NUM_CUS_DEFAULT) -> int:
    """Chunks per token, or 1 when splitting does not pay (or cannot be resident)."""
    s = 16
    while s > 1 and (T * s > num_cus or triton.cdiv(H, s) < 256):
        s //= 2
    return s


@triton.jit
def _agg_split_kernel(
    prefix_ptr, addend_ptr, prefix_out_ptr, bank_ptr, cw_ptr, ow_ptr,
    out_ptr, out8_ptr, ws_ptr, cnt_ptr,
    score_eps, out_eps,
    stride_pm, stride_am, stride_om, stride_bm, stride_bb, stride_o, stride_o8,
    H: tl.constexpr, S: tl.constexpr, BLOCK_C: tl.constexpr,
    NVB: tl.constexpr, RP2: tl.constexpr,
    HAS_ADD: tl.constexpr, WRITE_BANK: tl.constexpr,
    APPLY_OUT_NORM: tl.constexpr, HAS_OUT8: tl.constexpr,
):
    t = tl.program_id(0)
    s = tl.program_id(1)
    offs = s * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs < H

    # ---- phase 1: rows of this chunk (bank rows 0..NVB-1, prefix row NVB) ----
    row = tl.load(prefix_ptr + t * stride_pm + offs, mask=mask, other=0.0)
    if HAS_ADD:
        row = (
            row.to(tl.float32)
            + tl.load(addend_ptr + t * stride_am + offs, mask=mask, other=0.0).to(tl.float32)
        ).to(prefix_out_ptr.dtype.element_ty)
        tl.store(prefix_out_ptr + t * stride_om + offs, row, mask=mask)
    if WRITE_BANK:
        tl.store(bank_ptr + t * stride_bm + NVB * stride_bb + offs, row, mask=mask)
    pv = row.to(tl.float32)

    offs_r = tl.arange(0, RP2)
    is_bank = offs_r < NVB
    x = tl.load(
        bank_ptr + t * stride_bm + offs_r[:, None] * stride_bb + offs[None, :],
        mask=is_bank[:, None] & mask[None, :],
        other=0.0,
    ).to(tl.float32)
    # prefix row goes into row NVB of the [RP2, BLOCK_C] tile
    x = tl.where((offs_r == NVB)[:, None], pv[None, :], x)

    cw = tl.load(cw_ptr + offs, mask=mask, other=0.0)
    d_part = tl.sum(x * cw[None, :], axis=1)  # [RP2]
    g_part = tl.dot(x, tl.trans(x), input_precision="ieee")  # [RP2, RP2]

    ws_t = ws_ptr + (t * S + s) * (RP2 + RP2 * RP2)
    tl.store(ws_t + offs_r, d_part)
    tl.store(ws_t + RP2 + offs_r[:, None] * RP2 + offs_r[None, :], g_part)

    # ---- barrier: all S chunks of token t have published their partials ----
    # The poll must sit in the loop condition: a loop-carried polled value is
    # not re-loaded by the ROCm backend and spins forever (same form as hc_mix).
    old = tl.atomic_add(cnt_ptr + t, 1, sem="acq_rel", scope="gpu")
    target = (old // S + 1) * S
    while tl.atomic_add(cnt_ptr + t, 0, sem="acq_rel", scope="gpu") - target < 0:
        pass

    # ---- phase 2: reduce partials, softmax weights, mix this chunk ----
    d = tl.zeros((RP2,), tl.float32)
    g = tl.zeros((RP2, RP2), tl.float32)
    for j in tl.static_range(S):
        ws_j = ws_ptr + (t * S + j) * (RP2 + RP2 * RP2)
        d += tl.load(ws_j + offs_r, cache_modifier=".cg")
        g += tl.load(ws_j + RP2 + offs_r[:, None] * RP2 + offs_r[None, :], cache_modifier=".cg")
    sumsq = tl.sum(tl.where(offs_r[:, None] == offs_r[None, :], g, 0.0), axis=1)  # diag
    valid = offs_r <= NVB
    score = d / tl.sqrt(sumsq / H + score_eps)
    m = tl.max(tl.where(valid, score, -float("inf")))
    w = tl.where(valid, tl.exp(score - m), 0.0)
    w = w / tl.sum(w)

    acc = tl.sum(w[:, None] * x, axis=0)  # [BLOCK_C]
    if APPLY_OUT_NORM:
        acc_sq = tl.sum(w * tl.sum(g * w[None, :], axis=1))  # w^T G w
        scale = 1.0 / tl.sqrt(acc_sq / H + out_eps)
        ow = tl.load(ow_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc = acc * scale * ow
    ob = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + t * stride_o + offs, ob, mask=mask)
    if HAS_OUT8:
        q = tl.minimum(tl.maximum(ob.to(tl.float32), -448.0), 448.0)
        tl.store(out8_ptr + t * stride_o8 + offs, q.to(out8_ptr.dtype.element_ty), mask=mask)


def attn_res_hip_split(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: Optional[torch.Tensor],
    out: torch.Tensor,
    nvb: int,
    score_eps: float,
    out_eps: float,
    splits: int,
    *,
    addend: Optional[torch.Tensor] = None,
    prefix_out: Optional[torch.Tensor] = None,
    write_prefix: bool = False,
    out_fp8: Optional[torch.Tensor] = None,
) -> None:
    """Same contract as attn_res_hip.attn_res_hip, for splits >= 2 (see plan_splits)."""
    T, H = prefix_sum.shape
    assert 1 <= nvb and nvb + 1 <= 16 and splits >= 2 and T <= _MAX_T
    has_add = addend is not None
    assert not has_add or prefix_out is not None
    dev = prefix_sum.device
    cnt = _counters.get(dev)
    if cnt is None:
        cnt = _counters[dev] = torch.zeros(_MAX_T, dtype=torch.int32, device=dev)
    rp2 = 16 if nvb + 1 > 8 else (8 if nvb + 1 > 4 else 4)
    rp2 = max(rp2, 16)  # tl.dot needs >= 16 on each dim
    key = (dev, splits, rp2)
    ws = _workspaces.get(key)
    if ws is None:
        ws = _workspaces[key] = torch.empty(_MAX_T * splits * (rp2 + rp2 * rp2),
                                            dtype=torch.float32, device=dev)
    block_c = triton.next_power_of_2(triton.cdiv(H, splits))
    addend_arg = addend if has_add else prefix_sum
    prefix_out_arg = prefix_out if has_add else prefix_sum
    ow_arg = ow if ow is not None else cw
    out8_arg = out_fp8 if out_fp8 is not None else out
    _agg_split_kernel[(T, splits)](
        prefix_sum, addend_arg, prefix_out_arg, bank, cw, ow_arg, out, out8_arg, ws, cnt,
        score_eps, out_eps,
        prefix_sum.stride(0), addend_arg.stride(0), prefix_out_arg.stride(0),
        bank.stride(0), bank.stride(1), out.stride(0), out8_arg.stride(0),
        H=H, S=splits, BLOCK_C=block_c, NVB=nvb, RP2=rp2,
        HAS_ADD=has_add, WRITE_BANK=write_prefix,
        APPLY_OUT_NORM=ow is not None, HAS_OUT8=out_fp8 is not None,
        num_warps=4,
    )
