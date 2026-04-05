from __future__ import annotations

from collections.abc import Iterable

import torch

from sglang.srt.speculative.dflash.contracts import (
    STATUS_ACTIVE,
    STATUS_EOS_SEEN,
    STATUS_FINISHED,
    STATUS_STOPPED_BY_TOKEN,
    DFlashAcceptBonusResult,
    DFlashAcceptPublishResult,
    DFlashPrepareBlockResult,
    DFlashRequestStateTable,
    compute_live_row_mask,
)


def _to_token_id_set(token_ids: Iterable[int] | torch.Tensor | None) -> set[int]:
    if token_ids is None:
        return set()
    if isinstance(token_ids, torch.Tensor):
        return {int(x) for x in token_ids.view(-1).tolist()}
    return {int(x) for x in token_ids}


def prepare_block_reference(
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
    query_input_ids = torch.full(
        (bucket_bs, block_size),
        int(mask_token_id),
        dtype=state.next_verified_id.dtype,
        device=device,
    )
    query_positions = torch.zeros(
        (bucket_bs, block_size),
        dtype=torch.int64,
        device=device,
    )
    query_slot_ids = torch.full(
        (bucket_bs, block_size),
        int(dummy_slot_id),
        dtype=req_to_token.dtype,
        device=device,
    )
    emit_ids = torch.zeros(
        (bucket_bs, block_size),
        dtype=state.next_verified_id.dtype,
        device=device,
    )
    active_mask = compute_live_row_mask(state, req_pool_indices, req_generation).to(
        device=device
    )

    req_pool_indices_i64 = req_pool_indices.to(dtype=torch.int64)
    for row in range(bucket_bs):
        if not bool(active_mask[row].item()):
            continue

        req_idx = int(req_pool_indices_i64[row].item())
        committed_len = int(state.committed_len[req_idx].item())
        reserved_len = int(state.reserved_len[req_idx].item())
        next_verified_id = state.next_verified_id[req_idx]

        query_input_ids[row, 0] = next_verified_id
        emit_ids[row, 0] = next_verified_id

        for col in range(block_size):
            logical_pos = committed_len + col
            if logical_pos >= reserved_len:
                raise ValueError(
                    "prepare_block_reference tried to read beyond reserved_len. "
                    f"req_idx={req_idx}, logical_pos={logical_pos}, "
                    f"reserved_len={reserved_len}, block_size={block_size}."
                )
            if logical_pos >= int(req_to_token.shape[1]):
                raise ValueError(
                    "prepare_block_reference tried to read beyond req_to_token width. "
                    f"req_idx={req_idx}, logical_pos={logical_pos}, "
                    f"req_to_token_width={int(req_to_token.shape[1])}."
                )
            query_positions[row, col] = logical_pos
            query_slot_ids[row, col] = req_to_token[req_idx, logical_pos]

    sample_indices = None
    if include_sample_indices:
        sample_indices = torch.empty(
            (bucket_bs, max(block_size - 1, 0)),
            dtype=torch.int32,
            device=device,
        )
        for row in range(bucket_bs):
            for col in range(max(block_size - 1, 0)):
                sample_indices[row, col] = row * block_size + (col + 1)

    return DFlashPrepareBlockResult(
        query_input_ids=query_input_ids,
        query_positions=query_positions,
        query_slot_ids=query_slot_ids,
        emit_ids=emit_ids,
        sample_indices=sample_indices,
        active_mask=active_mask,
    )


def accept_bonus_reference(
    *,
    emit_ids: torch.Tensor,
    target_top1: torch.Tensor,
    active_mask: torch.Tensor | None = None,
    eos_token_ids: Iterable[int] | torch.Tensor | None = None,
    stop_token_ids: Iterable[int] | torch.Tensor | None = None,
) -> DFlashAcceptBonusResult:
    if emit_ids.ndim != 2 or target_top1.ndim != 2:
        raise ValueError(
            "emit_ids and target_top1 must both be rank-2 tensors. "
            f"Got {tuple(emit_ids.shape)} and {tuple(target_top1.shape)}."
        )
    if emit_ids.shape != target_top1.shape:
        raise ValueError(
            "emit_ids and target_top1 must have the same shape. "
            f"Got {tuple(emit_ids.shape)} and {tuple(target_top1.shape)}."
        )

    bs, block_size = emit_ids.shape
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    device = emit_ids.device
    if active_mask is None:
        active_mask = torch.ones((bs,), dtype=torch.bool, device=device)
    else:
        active_mask = active_mask.to(device=device, dtype=torch.bool)

    eos_ids = _to_token_id_set(eos_token_ids)
    stop_ids = _to_token_id_set(stop_token_ids)

    accept_lens = torch.zeros((bs,), dtype=torch.int32, device=device)
    commit_lens = torch.zeros((bs,), dtype=torch.int32, device=device)
    bonus_ids = torch.zeros((bs,), dtype=target_top1.dtype, device=device)
    gpu_stop_flags = torch.zeros((bs,), dtype=torch.int32, device=device)

    for row in range(bs):
        if not bool(active_mask[row].item()):
            continue

        accept_len = 0
        for col in range(block_size - 1):
            if int(target_top1[row, col].item()) != int(emit_ids[row, col + 1].item()):
                break
            accept_len += 1

        accept_lens[row] = accept_len
        commit_lens[row] = accept_len + 1
        bonus_ids[row] = target_top1[row, accept_len]

        committed_prefix = emit_ids[row, : accept_len + 1]
        flags = 0
        if eos_ids and any(int(x.item()) in eos_ids for x in committed_prefix):
            flags |= STATUS_EOS_SEEN | STATUS_FINISHED
        if stop_ids and any(int(x.item()) in stop_ids for x in committed_prefix):
            flags |= STATUS_STOPPED_BY_TOKEN | STATUS_FINISHED
        gpu_stop_flags[row] = flags

    return DFlashAcceptBonusResult(
        accept_lens=accept_lens,
        commit_lens=commit_lens,
        bonus_ids=bonus_ids,
        gpu_stop_flags=gpu_stop_flags,
    )


def publish_state_reference(
    *,
    state: DFlashRequestStateTable,
    req_pool_indices: torch.Tensor,
    req_generation: torch.Tensor,
    commit_lens: torch.Tensor,
    bonus_ids: torch.Tensor,
    gpu_stop_flags: torch.Tensor,
) -> DFlashRequestStateTable:
    state.validate()
    updated = state.clone()

    req_pool_indices_i64 = req_pool_indices.to(dtype=torch.int64)
    req_generation = req_generation.to(
        device=state.generation.device,
        dtype=state.generation.dtype,
    )
    commit_lens = commit_lens.to(
        device=state.committed_len.device,
        dtype=state.committed_len.dtype,
    )
    bonus_ids = bonus_ids.to(
        device=state.next_verified_id.device,
        dtype=state.next_verified_id.dtype,
    )
    gpu_stop_flags = gpu_stop_flags.to(
        device=state.status_flags.device,
        dtype=state.status_flags.dtype,
    )

    for row in range(int(req_pool_indices_i64.numel())):
        req_idx = int(req_pool_indices_i64[row].item())
        if int(state.generation[req_idx].item()) != int(req_generation[row].item()):
            continue

        commit_len = int(commit_lens[row].item())
        if commit_len <= 0:
            continue

        new_committed_len = int(updated.committed_len[req_idx].item()) + commit_len
        reserved_len = int(updated.reserved_len[req_idx].item())
        if new_committed_len > reserved_len:
            raise ValueError(
                "publish_state_reference would advance committed_len beyond reserved_len. "
                f"req_idx={req_idx}, new_committed_len={new_committed_len}, "
                f"reserved_len={reserved_len}."
            )

        updated.committed_len[req_idx] = new_committed_len
        updated.next_verified_id[req_idx] = bonus_ids[row]

        status_flags = int(updated.status_flags[req_idx].item()) | int(
            gpu_stop_flags[row].item()
        )
        if status_flags & (STATUS_EOS_SEEN | STATUS_FINISHED | STATUS_STOPPED_BY_TOKEN):
            status_flags &= ~STATUS_ACTIVE
        updated.status_flags[req_idx] = status_flags

    updated.validate()
    return updated


def accept_publish_reference(
    *,
    state: DFlashRequestStateTable,
    req_pool_indices: torch.Tensor,
    req_generation: torch.Tensor,
    emit_ids: torch.Tensor,
    target_top1: torch.Tensor,
    active_mask: torch.Tensor | None = None,
    eos_token_ids: Iterable[int] | torch.Tensor | None = None,
    stop_token_ids: Iterable[int] | torch.Tensor | None = None,
) -> DFlashAcceptPublishResult:
    accept = accept_bonus_reference(
        emit_ids=emit_ids,
        target_top1=target_top1,
        active_mask=active_mask,
        eos_token_ids=eos_token_ids,
        stop_token_ids=stop_token_ids,
    )
    updated = publish_state_reference(
        state=state,
        req_pool_indices=req_pool_indices,
        req_generation=req_generation,
        commit_lens=accept.commit_lens,
        bonus_ids=accept.bonus_ids,
        gpu_stop_flags=accept.gpu_stop_flags,
    )
    return DFlashAcceptPublishResult(accept=accept, state=updated)
