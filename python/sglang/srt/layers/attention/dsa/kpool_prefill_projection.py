"""Fixed reduction order for opt-in pooled-indexer prefill projections."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _projection(
    X, W, P, M,
    N: tl.constexpr, K: tl.constexpr,
    SX: tl.constexpr, SW: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SPLIT: tl.constexpr, FP32: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    ks = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK * SPLIT)):
        k = (block * SPLIT + split) * BK + ks
        x = tl.load(
            X + rows[:, None] * SX + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K), 0,
        )
        w = tl.load(
            W + cols[None, :] * SW + k[:, None],
            (cols[None, :] < N) & (k[:, None] < K), 0,
        )
        if FP32:
            acc = tl.dot(
                x.to(tl.float32), w.to(tl.float32), acc, input_precision="tf32x3"
            )
        else:
            acc = tl.dot(x, w, acc)
    tl.store(
        P + split * M * N + rows[:, None] * N + cols[None, :], acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit(do_not_specialize=["SIZE"])
def _reduce(P, Y, SIZE, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT)
    values = tl.load(
        P + splits[:, None] * SIZE + offsets[None, :], offsets[None, :] < SIZE, 0
    )
    value = tl.sum(values, axis=0)
    tl.store(Y + offsets, value, offsets < SIZE)


def kpool_prefill_linear(x, weight, dtype=None, split=8):
    assert x.dtype == torch.bfloat16 and weight.dtype in (torch.bfloat16, torch.float32)
    assert x.ndim == weight.ndim == 2 and x.shape[1] == weight.shape[1]
    assert x.stride(1) == weight.stride(1) == 1
    m, k = x.shape
    n = weight.shape[0]
    bm, bn, bk = (
        (16, 32, 64) if weight.dtype == torch.float32 else (32, 64, 64)
    )
    y = torch.empty((m, n), device=x.device, dtype=dtype or x.dtype)
    partial = (
        y
        if split == 1
        else torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    )
    _projection[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
        x, weight, partial, m, n, k, x.stride(0), weight.stride(0),
        bm, bn, bk, split, weight.dtype == torch.float32,
        num_warps=4, num_stages=3,
    )
    if split > 1:
        _reduce[(triton.cdiv(m * n, 128),)](
            partial, y, m * n, split, 128, num_warps=4
        )
    return y
