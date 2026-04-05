from __future__ import annotations

import torch


def _validate_compact_commit_inputs(
    *,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids_2d: torch.Tensor,
    commit_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if verify_hidden.ndim != 3:
        raise ValueError(
            "verify_hidden must be rank-3 [batch, block, hidden]. "
            f"Got {tuple(verify_hidden.shape)}."
        )
    if positions.shape != verify_hidden.shape[:2]:
        raise ValueError(
            "positions must match verify_hidden batch/block shape. "
            f"Got {tuple(positions.shape)} vs {tuple(verify_hidden.shape[:2])}."
        )
    if slot_ids_2d.shape != verify_hidden.shape[:2]:
        raise ValueError(
            "slot_ids_2d must match verify_hidden batch/block shape. "
            f"Got {tuple(slot_ids_2d.shape)} vs {tuple(verify_hidden.shape[:2])}."
        )
    if commit_lens.ndim != 1 or int(commit_lens.numel()) != int(verify_hidden.shape[0]):
        raise ValueError(
            "commit_lens must be rank-1 with batch_size entries. "
            f"Got shape {tuple(commit_lens.shape)} for batch {int(verify_hidden.shape[0])}."
        )
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "positions must have dtype int32 or int64. " f"Got {positions.dtype}."
        )
    if slot_ids_2d.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "slot_ids_2d must have dtype int32 or int64. " f"Got {slot_ids_2d.dtype}."
        )
    if commit_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "commit_lens must have dtype int32 or int64. " f"Got {commit_lens.dtype}."
        )
    return verify_hidden, positions, slot_ids_2d, commit_lens


def compact_commit_reference(
    *,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids_2d: torch.Tensor,
    commit_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    verify_hidden, positions, slot_ids_2d, commit_lens = (
        _validate_compact_commit_inputs(
            verify_hidden=verify_hidden,
            positions=positions,
            slot_ids_2d=slot_ids_2d,
            commit_lens=commit_lens,
        )
    )
    if int(commit_lens.numel()) == 0:
        hidden_size = int(verify_hidden.shape[-1])
        return (
            verify_hidden.new_empty((0, hidden_size)),
            positions.new_empty((0,)),
            slot_ids_2d.new_empty((0,)),
        )
    block_idx = torch.arange(
        int(verify_hidden.shape[1]),
        device=commit_lens.device,
        dtype=commit_lens.dtype,
    )
    valid_mask = block_idx.unsqueeze(0) < commit_lens.unsqueeze(1)
    return (
        verify_hidden[valid_mask],
        positions[valid_mask],
        slot_ids_2d[valid_mask],
    )
