from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dflash.contracts import (
    DFlashPrepareBlockResult,
    DFlashRequestStateTable,
)
from sglang.srt.speculative.dflash.kernels.prepare_block import (
    DFlashPrepareBlockWorkspace,
    create_prepare_block_workspace,
    prepare_block_result_from_workspace,
)

_STATUS_ACTIVE_I32 = 1 << 0
_GPU_STOP_MASK_I32 = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)


def _validate_prepare_block_inputs(
    *,
    state: DFlashRequestStateTable,
    req_pool_indices: torch.Tensor,
    req_generation: torch.Tensor,
    req_to_token: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    state.validate()
    req_pool_indices = req_pool_indices.view(-1).to(device=req_to_token.device)
    req_generation = req_generation.view(-1).to(device=req_to_token.device)
    if req_pool_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "req_pool_indices must have dtype int32 or int64. "
            f"Got {req_pool_indices.dtype}."
        )
    if req_generation.dtype not in (state.generation.dtype, torch.int32, torch.int64):
        raise ValueError(
            "req_generation must have integer dtype compatible with state.generation. "
            f"Got {req_generation.dtype}."
        )
    if req_to_token.ndim != 2:
        raise ValueError(
            "req_to_token must be rank-2. " f"Got {tuple(req_to_token.shape)}."
        )
    if int(req_pool_indices.numel()) != int(req_generation.numel()):
        raise ValueError(
            "req_pool_indices and req_generation must have the same number of rows. "
            f"Got {int(req_pool_indices.numel())} and {int(req_generation.numel())}."
        )
    if req_to_token.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "req_to_token must have dtype int32 or int64. " f"Got {req_to_token.dtype}."
        )
    return req_pool_indices, req_generation


def _make_sample_indices(
    *,
    bucket_bs: int,
    block_size: int,
    device: torch.device,
    include_sample_indices: bool,
) -> torch.Tensor | None:
    if not include_sample_indices:
        return None
    if block_size <= 1:
        return torch.empty((bucket_bs, 0), dtype=torch.int32, device=device)
    return torch.arange(
        bucket_bs * block_size,
        dtype=torch.int32,
        device=device,
    ).view(
        bucket_bs, block_size
    )[:, 1:]


@triton.jit
def _prepare_block_kernel(
    committed_len_ptr,
    reserved_len_ptr,
    next_verified_id_ptr,
    generation_ptr,
    status_flags_ptr,
    req_pool_indices_ptr,
    req_generation_ptr,
    req_to_token_ptr,
    active_mask_out_ptr,
    oob_flags_out_ptr,
    query_positions_out_ptr,
    query_slot_ids_out_ptr,
    query_input_ids_out_ptr,
    emit_ids_out_ptr,
    state_stride,
    req_pool_stride,
    req_generation_stride,
    req_to_token_row_stride,
    req_to_token_col_stride,
    active_mask_stride,
    oob_flags_stride,
    query_positions_row_stride,
    query_positions_col_stride,
    query_slot_ids_row_stride,
    query_slot_ids_col_stride,
    query_input_ids_row_stride,
    emit_ids_row_stride,
    req_to_token_width,
    block_size,
    bucket_bs,
    STATUS_ACTIVE_I32: tl.constexpr,
    GPU_STOP_MASK_I32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= bucket_bs:
        return

    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_stride)
    status_flags = tl.load(status_flags_ptr + req_idx * state_stride)
    status_i32 = status_flags.to(tl.int32)
    generation = tl.load(generation_ptr + req_idx * state_stride)
    req_generation = tl.load(req_generation_ptr + row * req_generation_stride)
    active = ((status_i32 & STATUS_ACTIVE_I32) != 0) & (
        (status_i32 & GPU_STOP_MASK_I32) == 0
    )
    active = active & (generation == req_generation)
    tl.store(active_mask_out_ptr + row * active_mask_stride, active.to(tl.int32))

    committed_len = tl.load(
        committed_len_ptr + req_idx * state_stride, mask=active, other=0
    )
    reserved_len = tl.load(
        reserved_len_ptr + req_idx * state_stride, mask=active, other=0
    )
    next_verified_id = tl.load(
        next_verified_id_ptr + req_idx * state_stride,
        mask=active,
        other=0,
    )

    last_pos = committed_len + (block_size - 1)
    oob = active & ((last_pos >= reserved_len) | (last_pos >= req_to_token_width))
    tl.store(oob_flags_out_ptr + row * oob_flags_stride, oob.to(tl.int32))
    valid = active & (oob == 0)

    tl.store(
        query_input_ids_out_ptr + row * query_input_ids_row_stride,
        next_verified_id,
        mask=valid,
    )
    tl.store(
        emit_ids_out_ptr + row * emit_ids_row_stride,
        next_verified_id,
        mask=valid,
    )

    cols = tl.arange(0, BLOCK_SIZE)
    mask = valid & (cols < block_size)
    logical_pos = committed_len + cols
    token_offsets = (
        req_idx * req_to_token_row_stride + logical_pos * req_to_token_col_stride
    )
    slot_ids = tl.load(req_to_token_ptr + token_offsets, mask=mask, other=0)
    tl.store(
        query_positions_out_ptr
        + row * query_positions_row_stride
        + cols * query_positions_col_stride,
        logical_pos.to(tl.int64),
        mask=mask,
    )
    tl.store(
        query_slot_ids_out_ptr
        + row * query_slot_ids_row_stride
        + cols * query_slot_ids_col_stride,
        slot_ids,
        mask=mask,
    )


@triton.jit
def _prepare_block_fused_kernel(
    committed_len_ptr,
    reserved_len_ptr,
    next_verified_id_ptr,
    generation_ptr,
    status_flags_ptr,
    req_pool_indices_ptr,
    req_generation_ptr,
    req_to_token_ptr,
    active_mask_out_ptr,
    oob_flags_out_ptr,
    query_positions_out_ptr,
    query_slot_ids_out_ptr,
    query_input_ids_out_ptr,
    emit_ids_out_ptr,
    sample_indices_out_ptr,
    state_stride,
    req_pool_stride,
    req_generation_stride,
    req_to_token_row_stride,
    req_to_token_col_stride,
    active_mask_stride,
    oob_flags_stride,
    query_positions_row_stride,
    query_positions_col_stride,
    query_slot_ids_row_stride,
    query_slot_ids_col_stride,
    query_input_ids_row_stride,
    emit_ids_row_stride,
    sample_indices_row_stride,
    sample_indices_col_stride,
    req_to_token_width,
    block_size,
    bucket_bs,
    mask_token_id,
    dummy_slot_id,
    sample_cols,
    STATUS_ACTIVE_I32: tl.constexpr,
    GPU_STOP_MASK_I32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WRITE_SAMPLE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= bucket_bs:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size
    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_stride)
    status_flags = tl.load(status_flags_ptr + req_idx * state_stride)
    status_i32 = status_flags.to(tl.int32)
    generation = tl.load(generation_ptr + req_idx * state_stride)
    req_generation = tl.load(req_generation_ptr + row * req_generation_stride)
    active = ((status_i32 & STATUS_ACTIVE_I32) != 0) & (
        (status_i32 & GPU_STOP_MASK_I32) == 0
    )
    active = active & (generation == req_generation)

    committed_len = tl.load(
        committed_len_ptr + req_idx * state_stride, mask=active, other=0
    )
    reserved_len = tl.load(
        reserved_len_ptr + req_idx * state_stride, mask=active, other=0
    )
    next_verified_id = tl.load(
        next_verified_id_ptr + req_idx * state_stride,
        mask=active,
        other=0,
    )
    last_pos = committed_len + (block_size - 1)
    oob = active & ((last_pos >= reserved_len) | (last_pos >= req_to_token_width))
    valid = active & (oob == 0)

    tl.store(active_mask_out_ptr + row * active_mask_stride, active.to(tl.int32))
    tl.store(oob_flags_out_ptr + row * oob_flags_stride, oob.to(tl.int32))

    logical_pos = committed_len + cols
    token_offsets = (
        req_idx * req_to_token_row_stride + logical_pos * req_to_token_col_stride
    )
    slot_ids = tl.load(req_to_token_ptr + token_offsets, mask=valid & row_mask, other=0)
    query_positions = tl.where(valid, logical_pos.to(tl.int64), 0)
    query_slot_ids = tl.where(valid, slot_ids, dummy_slot_id)
    is_first_col = cols == 0
    query_input_ids = tl.where(valid & is_first_col, next_verified_id, mask_token_id)
    emit_ids = tl.where(valid & is_first_col, next_verified_id, 0)

    tl.store(
        query_positions_out_ptr
        + row * query_positions_row_stride
        + cols * query_positions_col_stride,
        query_positions,
        mask=row_mask,
    )
    tl.store(
        query_slot_ids_out_ptr
        + row * query_slot_ids_row_stride
        + cols * query_slot_ids_col_stride,
        query_slot_ids,
        mask=row_mask,
    )
    tl.store(
        query_input_ids_out_ptr + row * query_input_ids_row_stride + cols,
        query_input_ids,
        mask=row_mask,
    )
    tl.store(
        emit_ids_out_ptr + row * emit_ids_row_stride + cols,
        emit_ids,
        mask=row_mask,
    )

    if WRITE_SAMPLE:
        sample_mask = cols < sample_cols
        sample_values = row * block_size + (cols + 1)
        tl.store(
            sample_indices_out_ptr
            + row * sample_indices_row_stride
            + cols * sample_indices_col_stride,
            sample_values.to(tl.int32),
            mask=sample_mask,
        )


def prepare_block_triton(
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

    block = min(64, triton.next_power_of_2(block_size))
    _prepare_block_kernel[(bucket_bs,)](
        state.committed_len,
        state.reserved_len,
        state.next_verified_id,
        state.generation,
        state.status_flags,
        req_pool_indices,
        req_generation.to(dtype=state.generation.dtype),
        req_to_token,
        active_mask_i32,
        oob_flags,
        query_positions,
        query_slot_ids,
        query_input_ids,
        emit_ids,
        state.committed_len.stride(0),
        req_pool_indices.stride(0),
        req_generation.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        active_mask_i32.stride(0),
        oob_flags.stride(0),
        query_positions.stride(0),
        query_positions.stride(1),
        query_slot_ids.stride(0),
        query_slot_ids.stride(1),
        query_input_ids.stride(0),
        emit_ids.stride(0),
        int(req_to_token.shape[1]),
        block_size,
        bucket_bs,
        STATUS_ACTIVE_I32=_STATUS_ACTIVE_I32,
        GPU_STOP_MASK_I32=_GPU_STOP_MASK_I32,
        BLOCK_SIZE=block,
        num_warps=1,
    )

    if bool(oob_flags.any().item()):
        first_bad = int(torch.nonzero(oob_flags, as_tuple=False)[0].item())
        raise ValueError(
            "prepare_block_triton would read beyond reserved_len or req_to_token width. "
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


def prepare_block_triton_fast(
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

    block = min(64, triton.next_power_of_2(block_size))
    sample_cols = max(block_size - 1, 0)
    sample_out = (
        workspace.sample_indices if include_sample_indices else workspace.sample_indices
    )
    _prepare_block_fused_kernel[(bucket_bs,)](
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
        state.committed_len.stride(0),
        req_pool_indices.stride(0),
        req_generation.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        workspace.active_mask_i32.stride(0),
        workspace.oob_flags.stride(0),
        workspace.query_positions.stride(0),
        workspace.query_positions.stride(1),
        workspace.query_slot_ids.stride(0),
        workspace.query_slot_ids.stride(1),
        workspace.query_input_ids.stride(0),
        workspace.emit_ids.stride(0),
        sample_out.stride(0) if sample_out.ndim == 2 else 0,
        sample_out.stride(1) if sample_out.ndim == 2 and sample_out.shape[1] > 0 else 1,
        int(req_to_token.shape[1]),
        block_size,
        bucket_bs,
        int(mask_token_id),
        int(dummy_slot_id),
        sample_cols,
        STATUS_ACTIVE_I32=_STATUS_ACTIVE_I32,
        GPU_STOP_MASK_I32=_GPU_STOP_MASK_I32,
        BLOCK_SIZE=block,
        WRITE_SAMPLE=include_sample_indices,
        num_warps=1,
    )

    if bool(workspace.oob_flags.any().item()):
        first_bad = int(torch.nonzero(workspace.oob_flags, as_tuple=False)[0].item())
        raise ValueError(
            "prepare_block_triton_fast would read beyond reserved_len or req_to_token width. "
            f"row={first_bad}, block_size={block_size}."
        )

    return prepare_block_result_from_workspace(workspace)
