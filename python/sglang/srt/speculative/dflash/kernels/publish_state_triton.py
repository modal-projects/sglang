from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dflash.contracts import DFlashRequestStateTable

_STATUS_ACTIVE_I32 = 1 << 0
_GPU_STOP_MASK_I32 = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)


def _validate_publish_state_inputs(
    *,
    state: DFlashRequestStateTable,
    req_pool_indices: torch.Tensor,
    req_generation: torch.Tensor,
    commit_lens: torch.Tensor,
    bonus_ids: torch.Tensor,
    gpu_stop_flags: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state.validate()
    req_pool_indices = req_pool_indices.view(-1).to(device=state.generation.device)
    if req_pool_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "req_pool_indices must have dtype int32 or int64. "
            f"Got {req_pool_indices.dtype}."
        )

    num_rows = int(req_pool_indices.numel())

    def _cast_state_like(name: str, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.view(-1).to(
            device=state.generation.device,
            dtype=state.generation.dtype,
        )
        if int(tensor.numel()) != num_rows:
            raise ValueError(
                f"{name} must have one entry per row. "
                f"Got {int(tensor.numel())} for {num_rows} rows."
            )
        return tensor

    req_generation = _cast_state_like("req_generation", req_generation)
    commit_lens = _cast_state_like("commit_lens", commit_lens)
    bonus_ids = _cast_state_like("bonus_ids", bonus_ids)
    gpu_stop_flags = _cast_state_like("gpu_stop_flags", gpu_stop_flags)
    return req_pool_indices, req_generation, commit_lens, bonus_ids, gpu_stop_flags


@triton.jit
def _publish_state_kernel(
    committed_len_ptr,
    reserved_len_ptr,
    next_verified_id_ptr,
    generation_ptr,
    status_flags_ptr,
    req_pool_indices_ptr,
    req_generation_ptr,
    commit_lens_ptr,
    bonus_ids_ptr,
    gpu_stop_flags_ptr,
    oob_flags_out_ptr,
    state_stride,
    req_pool_stride,
    req_generation_stride,
    commit_lens_stride,
    bonus_ids_stride,
    gpu_stop_flags_stride,
    oob_flags_stride,
    batch_size,
    STATUS_ACTIVE_I32: tl.constexpr,
    GPU_STOP_MASK_I32: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= batch_size:
        return

    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_stride)
    req_generation = tl.load(req_generation_ptr + row * req_generation_stride)
    commit_len = tl.load(commit_lens_ptr + row * commit_lens_stride)
    if commit_len <= 0:
        tl.store(oob_flags_out_ptr + row * oob_flags_stride, 0)
        return

    generation = tl.load(generation_ptr + req_idx * state_stride)
    if generation != req_generation:
        tl.store(oob_flags_out_ptr + row * oob_flags_stride, 0)
        return

    committed_len = tl.load(committed_len_ptr + req_idx * state_stride)
    reserved_len = tl.load(reserved_len_ptr + req_idx * state_stride)
    new_committed_len = committed_len + commit_len
    oob = new_committed_len > reserved_len
    tl.store(oob_flags_out_ptr + row * oob_flags_stride, oob.to(tl.int32))
    if oob:
        return

    tl.store(committed_len_ptr + req_idx * state_stride, new_committed_len)
    tl.store(
        next_verified_id_ptr + req_idx * state_stride,
        tl.load(bonus_ids_ptr + row * bonus_ids_stride),
    )

    new_status = tl.load(status_flags_ptr + req_idx * state_stride).to(tl.int32)
    new_status |= tl.load(gpu_stop_flags_ptr + row * gpu_stop_flags_stride).to(tl.int32)
    clear_active = (new_status & GPU_STOP_MASK_I32) != 0
    new_status = tl.where(
        clear_active,
        new_status & ~STATUS_ACTIVE_I32,
        new_status,
    )
    tl.store(status_flags_ptr + req_idx * state_stride, new_status)


def publish_state_triton(
    *,
    state: DFlashRequestStateTable,
    req_pool_indices: torch.Tensor,
    req_generation: torch.Tensor,
    commit_lens: torch.Tensor,
    bonus_ids: torch.Tensor,
    gpu_stop_flags: torch.Tensor,
    inplace: bool = False,
) -> DFlashRequestStateTable:
    req_pool_indices, req_generation, commit_lens, bonus_ids, gpu_stop_flags = (
        _validate_publish_state_inputs(
            state=state,
            req_pool_indices=req_pool_indices,
            req_generation=req_generation,
            commit_lens=commit_lens,
            bonus_ids=bonus_ids,
            gpu_stop_flags=gpu_stop_flags,
        )
    )
    updated = state if inplace else state.clone()
    batch_size = int(req_pool_indices.numel())
    oob_flags = torch.zeros(
        (batch_size,), dtype=torch.int32, device=updated.generation.device
    )

    _publish_state_kernel[(batch_size,)](
        updated.committed_len,
        updated.reserved_len,
        updated.next_verified_id,
        updated.generation,
        updated.status_flags,
        req_pool_indices,
        req_generation,
        commit_lens,
        bonus_ids,
        gpu_stop_flags,
        oob_flags,
        updated.committed_len.stride(0),
        req_pool_indices.stride(0),
        req_generation.stride(0),
        commit_lens.stride(0),
        bonus_ids.stride(0),
        gpu_stop_flags.stride(0),
        oob_flags.stride(0),
        batch_size,
        STATUS_ACTIVE_I32=_STATUS_ACTIVE_I32,
        GPU_STOP_MASK_I32=_GPU_STOP_MASK_I32,
        num_warps=1,
    )

    if bool(oob_flags.any().item()):
        raise ValueError(
            "publish_state_triton would advance committed_len beyond reserved_len."
        )
    updated.validate()
    return updated
