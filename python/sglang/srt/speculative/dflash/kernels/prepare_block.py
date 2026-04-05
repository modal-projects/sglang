from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.speculative.dflash.contracts import (
    DFlashPrepareBlockResult,
    DFlashRequestStateTable,
    compute_live_row_mask,
)


@dataclass
class DFlashPrepareBlockWorkspace:
    query_input_ids: torch.Tensor
    query_positions: torch.Tensor
    query_slot_ids: torch.Tensor
    emit_ids: torch.Tensor
    sample_indices: torch.Tensor
    active_mask_i32: torch.Tensor
    oob_flags: torch.Tensor
    include_sample_indices: bool


def create_prepare_block_workspace(
    *,
    bucket_bs: int,
    block_size: int,
    state_dtype: torch.dtype,
    token_dtype: torch.dtype,
    device: torch.device | str,
    include_sample_indices: bool = True,
) -> DFlashPrepareBlockWorkspace:
    if bucket_bs < 0:
        raise ValueError(f"bucket_bs must be non-negative, got {bucket_bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    device = torch.device(device)
    sample_width = max(block_size - 1, 0)
    return DFlashPrepareBlockWorkspace(
        query_input_ids=torch.empty(
            (bucket_bs, block_size), dtype=state_dtype, device=device
        ),
        query_positions=torch.empty(
            (bucket_bs, block_size), dtype=torch.int64, device=device
        ),
        query_slot_ids=torch.empty(
            (bucket_bs, block_size), dtype=token_dtype, device=device
        ),
        emit_ids=torch.empty((bucket_bs, block_size), dtype=state_dtype, device=device),
        sample_indices=torch.empty(
            (bucket_bs, sample_width), dtype=torch.int32, device=device
        ),
        active_mask_i32=torch.empty((bucket_bs,), dtype=torch.int32, device=device),
        oob_flags=torch.empty((bucket_bs,), dtype=torch.int32, device=device),
        include_sample_indices=include_sample_indices,
    )


def prepare_block_result_from_workspace(
    workspace: DFlashPrepareBlockWorkspace,
) -> DFlashPrepareBlockResult:
    return DFlashPrepareBlockResult(
        query_input_ids=workspace.query_input_ids,
        query_positions=workspace.query_positions,
        query_slot_ids=workspace.query_slot_ids,
        emit_ids=workspace.emit_ids,
        sample_indices=(
            workspace.sample_indices if workspace.include_sample_indices else None
        ),
        active_mask=workspace.active_mask_i32.to(dtype=torch.bool),
    )


def prepare_block_control(
    *,
    state: DFlashRequestStateTable,
    req_pool_indices: torch.Tensor,
    req_generation: torch.Tensor,
    req_to_token: torch.Tensor,
    block_size: int,
    mask_token_id: int,
    dummy_slot_id: int = -1,
    include_sample_indices: bool = True,
) -> DFlashPrepareBlockResult:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    state.validate()
    device = req_pool_indices.device
    bucket_bs = int(req_pool_indices.numel())
    req_idx = req_pool_indices.to(dtype=torch.int64)
    active_mask = compute_live_row_mask(state, req_pool_indices, req_generation).to(
        device=device
    )

    committed_len = state.committed_len.index_select(0, req_idx).to(
        device=device, dtype=torch.int64
    )
    reserved_len = state.reserved_len.index_select(0, req_idx).to(
        device=device, dtype=torch.int64
    )
    next_verified_id = state.next_verified_id.index_select(0, req_idx).to(device=device)

    offsets = torch.arange(block_size, dtype=torch.int64, device=device)
    query_positions = committed_len[:, None] + offsets[None, :]

    if bool(active_mask.any().item()):
        active_last_pos = query_positions[active_mask, -1]
        if torch.any(active_last_pos >= reserved_len[active_mask]):
            raise ValueError(
                "prepare_block_control would read beyond reserved_len for at least one live row."
            )
        if torch.any(active_last_pos >= int(req_to_token.shape[1])):
            raise ValueError(
                "prepare_block_control would read beyond req_to_token width for at least one live row."
            )

    safe_positions = torch.where(
        active_mask[:, None],
        query_positions,
        torch.zeros_like(query_positions),
    )
    gathered_slot_ids = req_to_token[
        req_idx[:, None],
        safe_positions,
    ]
    query_slot_ids = torch.where(
        active_mask[:, None],
        gathered_slot_ids,
        torch.full_like(gathered_slot_ids, int(dummy_slot_id)),
    )

    query_input_ids = torch.full(
        (bucket_bs, block_size),
        int(mask_token_id),
        dtype=next_verified_id.dtype,
        device=device,
    )
    query_input_ids[:, 0] = torch.where(
        active_mask,
        next_verified_id,
        query_input_ids[:, 0],
    )

    emit_ids = torch.zeros(
        (bucket_bs, block_size),
        dtype=next_verified_id.dtype,
        device=device,
    )
    emit_ids[:, 0] = torch.where(active_mask, next_verified_id, emit_ids[:, 0])

    query_positions = torch.where(
        active_mask[:, None],
        query_positions,
        torch.zeros_like(query_positions),
    )

    sample_indices = None
    if include_sample_indices:
        sample_indices = torch.arange(
            bucket_bs * block_size,
            dtype=torch.int32,
            device=device,
        ).view(bucket_bs, block_size)[:, 1:]

    return DFlashPrepareBlockResult(
        query_input_ids=query_input_ids,
        query_positions=query_positions,
        query_slot_ids=query_slot_ids,
        emit_ids=emit_ids,
        sample_indices=sample_indices,
        active_mask=active_mask,
    )
