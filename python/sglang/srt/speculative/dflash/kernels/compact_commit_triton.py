from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dflash.reference.compact_commit import (
    _validate_compact_commit_inputs,
)


def _compute_row_offsets(commit_lens: torch.Tensor) -> tuple[torch.Tensor, int]:
    commit_lens_i32 = commit_lens.to(dtype=torch.int32)
    total_valid = int(commit_lens_i32.sum().item())
    row_offsets = (
        torch.cumsum(commit_lens_i32, dim=0, dtype=torch.int32) - commit_lens_i32
    )
    return row_offsets.contiguous(), total_valid


def _allocate_outputs(
    *,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids_2d: torch.Tensor,
    hidden_out: torch.Tensor | None,
    positions_out: torch.Tensor | None,
    slot_ids_out: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    capacity = int(verify_hidden.shape[0] * verify_hidden.shape[1])
    hidden_size = int(verify_hidden.shape[2])
    if hidden_out is None:
        hidden_out = torch.empty(
            (capacity, hidden_size),
            dtype=verify_hidden.dtype,
            device=verify_hidden.device,
        )
    if positions_out is None:
        positions_out = torch.empty(
            (capacity,), dtype=positions.dtype, device=positions.device
        )
    if slot_ids_out is None:
        slot_ids_out = torch.empty(
            (capacity,), dtype=slot_ids_2d.dtype, device=slot_ids_2d.device
        )
    return hidden_out, positions_out, slot_ids_out


def _pick_block_size(hidden_size: int) -> tuple[int, int]:
    if hidden_size <= 128:
        return 128, 4
    if hidden_size <= 256:
        return 256, 4
    return 256, 8


@triton.jit
def _compact_commit_kernel(
    hidden_ptr,
    positions_ptr,
    slot_ids_ptr,
    commit_lens_ptr,
    row_offsets_ptr,
    hidden_out_ptr,
    positions_out_ptr,
    slot_ids_out_ptr,
    hidden_batch_stride,
    hidden_block_stride,
    hidden_col_stride,
    positions_batch_stride,
    positions_block_stride,
    slot_ids_batch_stride,
    slot_ids_block_stride,
    commit_lens_stride,
    row_offsets_stride,
    hidden_out_row_stride,
    positions_out_stride,
    slot_ids_out_stride,
    block_size,
    hidden_size,
    total_rows,
    BLOCK_H: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_h = tl.program_id(1)
    if row_idx >= total_rows:
        return

    batch_idx = row_idx // block_size
    token_idx = row_idx % block_size
    keep = tl.load(commit_lens_ptr + batch_idx * commit_lens_stride)
    if token_idx >= keep:
        return

    out_row = tl.load(row_offsets_ptr + batch_idx * row_offsets_stride) + token_idx
    offs = block_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < hidden_size

    hidden_in = (
        hidden_ptr
        + batch_idx * hidden_batch_stride
        + token_idx * hidden_block_stride
        + offs * hidden_col_stride
    )
    hidden_out = hidden_out_ptr + out_row * hidden_out_row_stride + offs
    tl.store(hidden_out, tl.load(hidden_in, mask=mask, other=0), mask=mask)

    if block_h == 0:
        position = tl.load(
            positions_ptr
            + batch_idx * positions_batch_stride
            + token_idx * positions_block_stride
        )
        slot_id = tl.load(
            slot_ids_ptr
            + batch_idx * slot_ids_batch_stride
            + token_idx * slot_ids_block_stride
        )
        tl.store(positions_out_ptr + out_row * positions_out_stride, position)
        tl.store(slot_ids_out_ptr + out_row * slot_ids_out_stride, slot_id)


def compact_commit_triton(
    *,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    hidden_out: torch.Tensor | None = None,
    positions_out: torch.Tensor | None = None,
    slot_ids_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    verify_hidden, positions, slot_ids_2d, commit_lens = (
        _validate_compact_commit_inputs(
            verify_hidden=verify_hidden,
            positions=positions,
            slot_ids_2d=slot_ids_2d,
            commit_lens=commit_lens,
        )
    )
    row_offsets, total_valid = _compute_row_offsets(commit_lens)
    hidden_out, positions_out, slot_ids_out = _allocate_outputs(
        verify_hidden=verify_hidden,
        positions=positions,
        slot_ids_2d=slot_ids_2d,
        hidden_out=hidden_out,
        positions_out=positions_out,
        slot_ids_out=slot_ids_out,
    )
    if total_valid == 0:
        return (
            hidden_out.narrow(0, 0, 0),
            positions_out.narrow(0, 0, 0),
            slot_ids_out.narrow(0, 0, 0),
        )
    block_h, num_warps = _pick_block_size(int(verify_hidden.shape[2]))
    total_rows = int(verify_hidden.shape[0] * verify_hidden.shape[1])
    grid = (total_rows, triton.cdiv(int(verify_hidden.shape[2]), block_h))
    _compact_commit_kernel[grid](
        verify_hidden.contiguous(),
        positions.contiguous(),
        slot_ids_2d.contiguous(),
        commit_lens.contiguous(),
        row_offsets,
        hidden_out,
        positions_out,
        slot_ids_out,
        verify_hidden.stride(0),
        verify_hidden.stride(1),
        verify_hidden.stride(2),
        positions.stride(0),
        positions.stride(1),
        slot_ids_2d.stride(0),
        slot_ids_2d.stride(1),
        commit_lens.stride(0),
        row_offsets.stride(0),
        hidden_out.stride(0),
        positions_out.stride(0),
        slot_ids_out.stride(0),
        int(verify_hidden.shape[1]),
        int(verify_hidden.shape[2]),
        total_rows,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )
    return (
        hidden_out.narrow(0, 0, total_valid),
        positions_out.narrow(0, 0, total_valid),
        slot_ids_out.narrow(0, 0, total_valid),
    )
