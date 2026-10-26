"""Vocab-parallel greedy argmax with one packed 32-bit key per row.

Key (uint32): ``orderable(bf16 max) << 16 | (0xFFFF - local_argmax)``, with -0
canonicalized to +0, so the max key is the max value with the first index on
ties. ``select_from_gathered`` keeps the lowest rank on value ties, matching
``torch.argmax`` over contiguous ascending vocab shards. Requires uniform,
unpadded local shards of at most 65536 entries.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_row_argmax_kernel(
    logits_ptr,
    stride_row,
    keys_ptr,  # int32 [n_pad]
    n_rows,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        tl.store(keys_ptr + row, 0)
        return
    best = tl.full([BLOCK], -1, tl.int64)
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < V
        x = tl.load(logits_ptr + row * stride_row + offs, mask=m, other=0.0)
        bits = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        bits = tl.where(bits == 0x8000, 0, bits)  # -0 -> +0
        neg = (bits & 0x8000) != 0
        ordv = tl.where(neg, (~bits) & 0xFFFF, bits | 0x8000)
        key = ordv.to(tl.int64) * 65536 + (65535 - offs).to(tl.int64)
        key = tl.where(m, key, -1)
        best = tl.maximum(best, key)
    k = tl.max(best, 0)
    tl.store(keys_ptr + row, (k & 0xFFFFFFFF).to(tl.uint32).to(tl.int32, bitcast=True))


@triton.jit
def _select_kernel(
    gathered_ptr,  # int32 [TP, n_pad]
    out_ptr,  # int64 [n]
    n_rows,
    n_pad,
    shard,
    TP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_rows
    best_val = tl.full([BLOCK], -1, tl.int64)
    best_id = tl.zeros([BLOCK], tl.int64)
    for r in tl.static_range(TP):
        key = tl.load(gathered_ptr + r * n_pad + offs, mask=m, other=0)
        key = key.to(tl.int64) & 0xFFFFFFFF
        val = key >> 16
        loc = 65535 - (key & 0xFFFF)
        take = val > best_val
        best_val = tl.where(take, val, best_val)
        best_id = tl.where(take, r * shard + loc, best_id)
    tl.store(out_ptr + offs, best_id, mask=m)


def pack_row_argmax(logits: torch.Tensor, keys_i32: torch.Tensor) -> None:
    """logits [n, V] bf16 (row-strided); keys_i32 [n_pad] int32, rows >= n zeroed."""
    n, V = logits.shape
    assert logits.dtype == torch.bfloat16 and logits.stride(1) == 1
    assert V <= 65536
    _pack_row_argmax_kernel[(keys_i32.shape[0],)](
        logits, logits.stride(0), keys_i32, n, V=V, BLOCK=2048, num_warps=8
    )


def select_from_gathered(
    gathered_i32: torch.Tensor,
    out: torch.Tensor,
    n: int,
    n_pad: int,
    shard: int,
    tp: int,
) -> None:
    BLOCK = 64
    _select_kernel[(triton.cdiv(max(n, 1), BLOCK),)](
        gathered_i32, out, n, n_pad, shard, TP=tp, BLOCK=BLOCK, num_warps=1
    )
