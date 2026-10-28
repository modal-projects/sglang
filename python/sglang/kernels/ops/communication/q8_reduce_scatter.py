"""Reduce-scatter of a TP-partial tensor over its rows with INT8 payloads.

Each rank quantizes its partial in groups of 32 values with one fp32 scale per group,
an all-to-all moves the rows each rank owns to it, and the owner sums the dequantized
contributions in fp32 in rank order. That moves ~0.56x the bytes of a bf16
reduce-scatter. A group holding a NaN or an infinity arrives as NaN.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

GROUP = 32
# Widths in multiples of 16 groups keep every size argument a multiple of 16, so
# each kernel compiles once.
WIDTH_MULTIPLE = 16 * GROUP
# (groups per program, num_warps) of each kernel, from an MI355X sweep at 4K-16K
# rows x 3584 and 7168 columns.
_PACK_CFG = (32, 1)
_UNPACK_CFG = (32, 2)


@triton.jit
def _pack_kernel(
    x_ptr,
    out_ptr,
    n_groups,
    n_valid_groups,
    groups_per_chunk,
    chunk_bytes,
    scale_offset,
    BLOCK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    g = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = g < n_groups
    cols = tl.arange(0, GROUP_SIZE)
    x = tl.load(
        x_ptr + g.to(tl.int64)[:, None] * GROUP_SIZE + cols[None, :],
        mask=(g < n_valid_groups)[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    v = x * tl.where(amax > 0, 127.0 / amax, 0.0)[:, None]
    # Rounds half away from zero: the int8 cast truncates.
    q = (v + tl.where(v >= 0, 0.5, -0.5)).to(tl.int8)
    # x * 0 is NaN where x is NaN or infinite, which makes the group's scale NaN.
    scale = amax / 127.0 + tl.sum(x * 0.0, axis=1)
    chunk = g // groups_per_chunk
    local = (g - chunk * groups_per_chunk).to(tl.int64)
    base = chunk.to(tl.int64) * chunk_bytes
    tl.store(
        out_ptr + base[:, None] + local[:, None] * GROUP_SIZE + cols[None, :],
        q,
        mask=mask[:, None],
    )
    scale_ptr = (out_ptr + base + scale_offset + local * 4).to(
        tl.pointer_type(tl.float32)
    )
    tl.store(scale_ptr, scale, mask=mask)


@triton.jit
def _unpack_sum_kernel(
    in_ptr,
    out_ptr,
    n_groups,
    chunk_bytes,
    scale_offset,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    g = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = g < n_groups
    cols = tl.arange(0, GROUP_SIZE)
    gl = g.to(tl.int64)
    acc = tl.zeros((BLOCK, GROUP_SIZE), dtype=tl.float32)
    for j in tl.static_range(WORLD):
        base = chunk_bytes.to(tl.int64) * j
        q = tl.load(
            in_ptr + base + gl[:, None] * GROUP_SIZE + cols[None, :],
            mask=mask[:, None],
            other=0,
        ).to(tl.float32)
        scale_ptr = (in_ptr + base + scale_offset + gl * 4).to(
            tl.pointer_type(tl.float32)
        )
        acc += q * tl.load(scale_ptr, mask=mask, other=0.0)[:, None]
    tl.store(
        out_ptr + gl[:, None] * GROUP_SIZE + cols[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask[:, None],
    )


_workspace: dict = {}


def _chunk_bytes(local: int) -> int:
    return local + local // GROUP * 4


def supports(partial: torch.Tensor) -> bool:
    """Whether reduce_scatter_q8 takes this partial."""
    return (
        partial.dtype in (torch.bfloat16, torch.float16)
        and partial.dim() == 2
        and partial.shape[1] % WIDTH_MULTIPLE == 0
    )


def send_bytes(rows: int, width: int, world: int) -> int:
    """The size of one rank's packed send buffer for a [rows, width] partial."""
    return world * _chunk_bytes(rows // world * width)


def _buffers(device: torch.device, nbytes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Send and receive buffers of at least nbytes, kept for later calls."""
    send, recv = _workspace.get(device, (None, None))
    if send is None or send.numel() < nbytes:
        send = torch.empty(nbytes, dtype=torch.int8, device=device)
        recv = torch.empty_like(send)
        _workspace[device] = (send, recv)
    return send[:nbytes], recv[:nbytes]


def pack(partial: torch.Tensor, rows: int, world: int, out: torch.Tensor) -> None:
    """Quantize ``partial`` [valid_rows, width], zero rows to ``rows``, into ``out``
    as ``world`` chunks of [int8 values | fp32 scales]."""
    width = partial.shape[1]
    local = rows // world * width
    chunk_bytes = _chunk_bytes(local)
    n_groups = rows * width // GROUP
    block, warps = _PACK_CFG
    _pack_kernel[(triton.cdiv(n_groups, block),)](
        partial.contiguous(),
        out,
        n_groups,
        partial.shape[0] * width // GROUP,
        local // GROUP,
        chunk_bytes,
        local,
        BLOCK=block,
        GROUP_SIZE=GROUP,
        num_warps=warps,
    )


def unpack_sum(received: torch.Tensor, world: int, out: torch.Tensor) -> None:
    """Sum the ``world`` received chunks into ``out`` [rows / world, width]."""
    local = out.numel()
    chunk_bytes = _chunk_bytes(local)
    block, warps = _UNPACK_CFG
    _unpack_sum_kernel[(triton.cdiv(local // GROUP, block),)](
        received,
        out,
        local // GROUP,
        chunk_bytes,
        local,
        WORLD=world,
        BLOCK=block,
        GROUP_SIZE=GROUP,
        num_warps=warps,
    )


def reduce_scatter_q8(
    partial: torch.Tensor, group, rows: Optional[int] = None
) -> torch.Tensor:
    """This rank's rows of the sum over ``group`` of a TP-partial [valid_rows, width]
    tensor, its rows zero-padded to ``rows`` (a multiple of the group size)."""
    world = group.world_size
    width = partial.shape[1]
    rows = partial.shape[0] if rows is None else rows
    assert supports(partial) and rows % world == 0 and partial.shape[0] <= rows
    send, recv = _buffers(partial.device, send_bytes(rows, width, world))
    pack(partial, rows, world, send)
    group.all_to_all_single(recv, send)
    out = partial.new_empty(rows // world, width)
    unpack_sum(recv, world, out)
    return out
