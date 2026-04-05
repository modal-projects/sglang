from __future__ import annotations

import torch

from sglang.jit_kernel.dflash_prepare_block import (
    prepare_block_cuda,
    prepare_block_fused_sample_cuda,
)
from sglang.srt.speculative.dflash.contracts import (
    DFlashPrepareBlockResult,
    DFlashRequestStateTable,
)
from sglang.srt.speculative.dflash.kernels.prepare_block import (
    DFlashPrepareBlockWorkspace,
    create_prepare_block_workspace,
    prepare_block_result_from_workspace,
)
from sglang.srt.speculative.dflash.kernels.prepare_block_triton import (
    _make_sample_indices,
    _validate_prepare_block_inputs,
)


def prepare_block_jit(
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
    req_pool_indices, req_generation = _validate_prepare_block_inputs(
        state=state,
        req_pool_indices=req_pool_indices,
        req_generation=req_generation,
        req_to_token=req_to_token,
        block_size=block_size,
    )
    device = req_to_token.device
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
    active_mask_i32 = torch.zeros((bucket_bs,), dtype=torch.int32, device=device)
    oob_flags = torch.zeros((bucket_bs,), dtype=torch.int32, device=device)

    prepare_block_cuda(
        state.committed_len,
        state.reserved_len,
        state.next_verified_id,
        state.generation,
        state.status_flags,
        req_pool_indices.contiguous(),
        req_generation.to(dtype=state.generation.dtype).contiguous(),
        req_to_token,
        active_mask_i32,
        oob_flags,
        query_positions,
        query_slot_ids,
        query_input_ids,
        emit_ids,
    )

    if bool(oob_flags.any().item()):
        first_bad = int(torch.nonzero(oob_flags, as_tuple=False)[0].item())
        raise ValueError(
            "prepare_block_jit would read beyond reserved_len or req_to_token width. "
            f"row={first_bad}, block_size={block_size}."
        )

    return DFlashPrepareBlockResult(
        query_input_ids=query_input_ids,
        query_positions=query_positions,
        query_slot_ids=query_slot_ids,
        emit_ids=emit_ids,
        sample_indices=_make_sample_indices(
            bucket_bs=bucket_bs,
            block_size=block_size,
            device=device,
            include_sample_indices=include_sample_indices,
        ),
        active_mask=active_mask_i32.to(dtype=torch.bool),
    )


def prepare_block_jit_fast(
    *,
    state: DFlashRequestStateTable,
    req_pool_indices: torch.Tensor,
    req_generation: torch.Tensor,
    req_to_token: torch.Tensor,
    block_size: int,
    mask_token_id: int,
    dummy_slot_id: int = -1,
    include_sample_indices: bool = True,
    workspace: DFlashPrepareBlockWorkspace | None = None,
) -> DFlashPrepareBlockResult:
    req_pool_indices, req_generation = _validate_prepare_block_inputs(
        state=state,
        req_pool_indices=req_pool_indices,
        req_generation=req_generation,
        req_to_token=req_to_token,
        block_size=block_size,
    )
    device = req_to_token.device
    bucket_bs = int(req_pool_indices.numel())
    if workspace is None:
        workspace = create_prepare_block_workspace(
            bucket_bs=bucket_bs,
            block_size=block_size,
            state_dtype=state.next_verified_id.dtype,
            token_dtype=req_to_token.dtype,
            device=device,
            include_sample_indices=include_sample_indices,
        )
    elif workspace.include_sample_indices != include_sample_indices:
        raise ValueError(
            "workspace.include_sample_indices does not match include_sample_indices."
        )
    if tuple(workspace.query_input_ids.shape) != (bucket_bs, block_size):
        raise ValueError(
            "prepare_block workspace shape mismatch. "
            f"Expected {(bucket_bs, block_size)}, got {tuple(workspace.query_input_ids.shape)}."
        )

    sample_out = workspace.sample_indices
    prepare_block_fused_sample_cuda(
        state.committed_len,
        state.reserved_len,
        state.next_verified_id,
        state.generation,
        state.status_flags,
        req_pool_indices.contiguous(),
        req_generation.to(dtype=state.generation.dtype).contiguous(),
        req_to_token,
        workspace.active_mask_i32,
        workspace.oob_flags,
        workspace.query_positions,
        workspace.query_slot_ids,
        workspace.query_input_ids,
        workspace.emit_ids,
        sample_out,
        mask_token_id,
        dummy_slot_id,
    )

    if bool(workspace.oob_flags.any().item()):
        first_bad = int(torch.nonzero(workspace.oob_flags, as_tuple=False)[0].item())
        raise ValueError(
            "prepare_block_jit_fast would read beyond reserved_len or req_to_token width. "
            f"row={first_bad}, block_size={block_size}."
        )

    return prepare_block_result_from_workspace(workspace)
