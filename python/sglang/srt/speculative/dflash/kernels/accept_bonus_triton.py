from __future__ import annotations

from collections.abc import Iterable

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dflash.contracts import DFlashAcceptBonusResult
from sglang.srt.speculative.dflash.kernels.accept_bonus import _normalize_token_ids

_EOS_FINISHED_FLAGS_I32 = (1 << 1) | (1 << 2)
_STOP_FINISHED_FLAGS_I32 = (1 << 4) | (1 << 2)


def _validate_accept_bonus_inputs(
    *,
    emit_ids: torch.Tensor,
    target_top1: torch.Tensor,
    active_mask: torch.Tensor | None,
    eos_token_ids: Iterable[int] | torch.Tensor | None,
    stop_token_ids: Iterable[int] | torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    if emit_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "emit_ids must have dtype int32 or int64. " f"Got {emit_ids.dtype}."
        )
    if target_top1.dtype != emit_ids.dtype:
        raise ValueError(
            "target_top1 must have the same dtype as emit_ids. "
            f"Got {target_top1.dtype} and {emit_ids.dtype}."
        )

    bs, block_size = emit_ids.shape
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    device = emit_ids.device
    if active_mask is None:
        active_mask_i32 = torch.ones((bs,), dtype=torch.int32, device=device)
    else:
        active_mask_i32 = active_mask.to(device=device, dtype=torch.int32).view(-1)
        if int(active_mask_i32.numel()) != bs:
            raise ValueError(
                "active_mask must have one entry per row. "
                f"Got {int(active_mask_i32.numel())} for batch size {bs}."
            )

    eos_ids = _normalize_token_ids(
        eos_token_ids,
        device=device,
        dtype=emit_ids.dtype,
    )
    if eos_ids is None:
        eos_ids = torch.empty((0,), dtype=emit_ids.dtype, device=device)
    else:
        eos_ids = eos_ids.contiguous()

    stop_ids = _normalize_token_ids(
        stop_token_ids,
        device=device,
        dtype=emit_ids.dtype,
    )
    if stop_ids is None:
        stop_ids = torch.empty((0,), dtype=emit_ids.dtype, device=device)
    else:
        stop_ids = stop_ids.contiguous()

    return (
        emit_ids,
        target_top1,
        active_mask_i32,
        eos_ids,
        stop_ids,
    )


@triton.jit
def _accept_bonus_kernel(
    emit_ids_ptr,
    target_top1_ptr,
    active_mask_ptr,
    eos_ids_ptr,
    stop_ids_ptr,
    accept_lens_out_ptr,
    commit_lens_out_ptr,
    bonus_ids_out_ptr,
    gpu_stop_flags_out_ptr,
    emit_row_stride,
    emit_col_stride,
    target_row_stride,
    target_col_stride,
    active_stride,
    eos_stride,
    stop_stride,
    accept_stride,
    commit_stride,
    bonus_stride,
    flags_stride,
    batch_size,
    block_size,
    eos_count,
    stop_count,
    EOS_FINISHED_FLAGS_I32: tl.constexpr,
    STOP_FINISHED_FLAGS_I32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_EOS_IDS: tl.constexpr,
    MAX_STOP_IDS: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= batch_size:
        return

    active = tl.load(active_mask_ptr + row * active_stride) != 0
    if not active:
        tl.store(accept_lens_out_ptr + row * accept_stride, 0)
        tl.store(commit_lens_out_ptr + row * commit_stride, 0)
        tl.store(bonus_ids_out_ptr + row * bonus_stride, 0)
        tl.store(gpu_stop_flags_out_ptr + row * flags_stride, 0)
        return

    accept_len = tl.full((), 0, tl.int32)
    prefix_live = tl.full((), 1, tl.int32)
    for col in range(BLOCK_SIZE - 1):
        in_range = col < (block_size - 1)
        emit_id = tl.load(
            emit_ids_ptr + row * emit_row_stride + (col + 1) * emit_col_stride,
            mask=in_range,
            other=0,
        )
        target_id = tl.load(
            target_top1_ptr + row * target_row_stride + col * target_col_stride,
            mask=in_range,
            other=0,
        )
        match_i32 = (emit_id == target_id).to(tl.int32)
        keep = in_range & (prefix_live != 0) & (match_i32 != 0)
        accept_len += keep.to(tl.int32)
        prefix_live = tl.where(in_range, prefix_live & match_i32, prefix_live)

    commit_len = accept_len + 1
    bonus_id = tl.load(
        target_top1_ptr
        + row * target_row_stride
        + accept_len.to(tl.int64) * target_col_stride
    )

    gpu_stop_flags = tl.full((), 0, tl.int32)
    for col in range(BLOCK_SIZE):
        in_commit = col < commit_len
        token = tl.load(
            emit_ids_ptr + row * emit_row_stride + col * emit_col_stride,
            mask=in_commit,
            other=0,
        )

        eos_hit = tl.full((), 0, tl.int32)
        for i in range(MAX_EOS_IDS):
            eos_valid = i < eos_count
            eos_id = tl.load(
                eos_ids_ptr + i * eos_stride,
                mask=eos_valid,
                other=0,
            )
            eos_hit |= (eos_valid & (token == eos_id)).to(tl.int32)

        stop_hit = tl.full((), 0, tl.int32)
        for i in range(MAX_STOP_IDS):
            stop_valid = i < stop_count
            stop_id = tl.load(
                stop_ids_ptr + i * stop_stride,
                mask=stop_valid,
                other=0,
            )
            stop_hit |= (stop_valid & (token == stop_id)).to(tl.int32)

        gpu_stop_flags = tl.where(
            in_commit & (eos_hit != 0),
            gpu_stop_flags | EOS_FINISHED_FLAGS_I32,
            gpu_stop_flags,
        )
        gpu_stop_flags = tl.where(
            in_commit & (stop_hit != 0),
            gpu_stop_flags | STOP_FINISHED_FLAGS_I32,
            gpu_stop_flags,
        )

    tl.store(accept_lens_out_ptr + row * accept_stride, accept_len)
    tl.store(commit_lens_out_ptr + row * commit_stride, commit_len)
    tl.store(bonus_ids_out_ptr + row * bonus_stride, bonus_id)
    tl.store(gpu_stop_flags_out_ptr + row * flags_stride, gpu_stop_flags)


def accept_bonus_triton(
    *,
    emit_ids: torch.Tensor,
    target_top1: torch.Tensor,
    active_mask: torch.Tensor | None = None,
    eos_token_ids: Iterable[int] | torch.Tensor | None = None,
    stop_token_ids: Iterable[int] | torch.Tensor | None = None,
) -> DFlashAcceptBonusResult:
    emit_ids, target_top1, active_mask_i32, eos_ids, stop_ids = (
        _validate_accept_bonus_inputs(
            emit_ids=emit_ids,
            target_top1=target_top1,
            active_mask=active_mask,
            eos_token_ids=eos_token_ids,
            stop_token_ids=stop_token_ids,
        )
    )
    batch_size, block_size = emit_ids.shape
    device = emit_ids.device

    accept_lens = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    commit_lens = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    bonus_ids = torch.zeros((batch_size,), dtype=target_top1.dtype, device=device)
    gpu_stop_flags = torch.zeros((batch_size,), dtype=torch.int32, device=device)

    block = min(64, triton.next_power_of_2(block_size))
    max_eos_ids = max(1, triton.next_power_of_2(int(eos_ids.numel()) or 1))
    max_stop_ids = max(1, triton.next_power_of_2(int(stop_ids.numel()) or 1))

    _accept_bonus_kernel[(batch_size,)](
        emit_ids,
        target_top1,
        active_mask_i32,
        eos_ids,
        stop_ids,
        accept_lens,
        commit_lens,
        bonus_ids,
        gpu_stop_flags,
        emit_ids.stride(0),
        emit_ids.stride(1),
        target_top1.stride(0),
        target_top1.stride(1),
        active_mask_i32.stride(0),
        eos_ids.stride(0) if eos_ids.numel() > 0 else 1,
        stop_ids.stride(0) if stop_ids.numel() > 0 else 1,
        accept_lens.stride(0),
        commit_lens.stride(0),
        bonus_ids.stride(0),
        gpu_stop_flags.stride(0),
        batch_size,
        block_size,
        int(eos_ids.numel()),
        int(stop_ids.numel()),
        EOS_FINISHED_FLAGS_I32=_EOS_FINISHED_FLAGS_I32,
        STOP_FINISHED_FLAGS_I32=_STOP_FINISHED_FLAGS_I32,
        BLOCK_SIZE=block,
        MAX_EOS_IDS=max_eos_ids,
        MAX_STOP_IDS=max_stop_ids,
        num_warps=1,
    )

    return DFlashAcceptBonusResult(
        accept_lens=accept_lens,
        commit_lens=commit_lens,
        bonus_ids=bonus_ids,
        gpu_stop_flags=gpu_stop_flags,
    )
