from __future__ import annotations

import torch

from sglang.srt.speculative.dflash.contracts import (
    GPU_STOP_MASK,
    STATUS_ACTIVE,
    DFlashRequestStateTable,
)


def publish_state_control(
    *,
    state: DFlashRequestStateTable,
    req_pool_indices: torch.Tensor,
    req_generation: torch.Tensor,
    commit_lens: torch.Tensor,
    bonus_ids: torch.Tensor,
    gpu_stop_flags: torch.Tensor,
    inplace: bool = False,
) -> DFlashRequestStateTable:
    state.validate()
    updated = state if inplace else state.clone()

    req_idx = req_pool_indices.to(dtype=torch.int64)
    req_generation = req_generation.to(
        device=updated.generation.device,
        dtype=updated.generation.dtype,
    )
    commit_lens = commit_lens.to(
        device=updated.committed_len.device,
        dtype=updated.committed_len.dtype,
    )
    bonus_ids = bonus_ids.to(
        device=updated.next_verified_id.device,
        dtype=updated.next_verified_id.dtype,
    )
    gpu_stop_flags = gpu_stop_flags.to(
        device=updated.status_flags.device,
        dtype=updated.status_flags.dtype,
    )

    valid_mask = updated.generation.index_select(0, req_idx) == req_generation
    valid_mask &= commit_lens > 0
    if not bool(valid_mask.any().item()):
        return updated

    active_req_idx = req_idx[valid_mask]
    new_committed = (
        updated.committed_len.index_select(0, active_req_idx) + commit_lens[valid_mask]
    )
    reserved = updated.reserved_len.index_select(0, active_req_idx)
    if torch.any(new_committed > reserved):
        raise ValueError(
            "publish_state_control would advance committed_len beyond reserved_len."
        )

    updated.committed_len[active_req_idx] = new_committed
    updated.next_verified_id[active_req_idx] = bonus_ids[valid_mask]

    new_status = (
        updated.status_flags.index_select(0, active_req_idx)
        | gpu_stop_flags[valid_mask]
    )
    clear_active = (new_status & GPU_STOP_MASK) != 0
    new_status = torch.where(
        clear_active,
        new_status & ~STATUS_ACTIVE,
        new_status,
    )
    updated.status_flags[active_req_idx] = new_status
    updated.validate()
    return updated
