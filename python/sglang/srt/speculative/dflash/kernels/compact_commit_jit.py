from __future__ import annotations

import torch

from sglang.jit_kernel.dflash_compact_commit import compact_commit_cuda
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


def compact_commit_jit(
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
    compact_commit_cuda(
        verify_hidden.contiguous(),
        positions.contiguous(),
        slot_ids_2d.contiguous(),
        commit_lens.contiguous(),
        row_offsets,
        hidden_out,
        positions_out,
        slot_ids_out,
    )
    return (
        hidden_out.narrow(0, 0, total_valid),
        positions_out.narrow(0, 0, total_valid),
        slot_ids_out.narrow(0, 0, total_valid),
    )
