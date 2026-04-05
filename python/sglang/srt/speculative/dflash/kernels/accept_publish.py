from __future__ import annotations

from collections.abc import Iterable

import torch

from sglang.srt.speculative.dflash.contracts import (
    DFlashAcceptPublishResult,
    DFlashRequestStateTable,
)
from sglang.srt.speculative.dflash.kernels.accept_bonus import accept_bonus_control
from sglang.srt.speculative.dflash.kernels.accept_bonus_triton import (
    _validate_accept_bonus_inputs,
)
from sglang.srt.speculative.dflash.kernels.publish_state import publish_state_control


def _validate_accept_publish_inputs(
    *,
    state: DFlashRequestStateTable,
    req_pool_indices: torch.Tensor,
    req_generation: torch.Tensor,
    emit_ids: torch.Tensor,
    target_top1: torch.Tensor,
    active_mask: torch.Tensor | None,
    eos_token_ids: Iterable[int] | torch.Tensor | None,
    stop_token_ids: Iterable[int] | torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    emit_ids, target_top1, active_mask_i32, eos_ids, stop_ids = (
        _validate_accept_bonus_inputs(
            emit_ids=emit_ids,
            target_top1=target_top1,
            active_mask=active_mask,
            eos_token_ids=eos_token_ids,
            stop_token_ids=stop_token_ids,
        )
    )
    state.validate()
    if emit_ids.device != state.generation.device:
        raise ValueError(
            "emit_ids and state tensors must live on the same device. "
            f"Got {emit_ids.device} and {state.generation.device}."
        )
    req_pool_indices = req_pool_indices.view(-1).to(device=state.generation.device)
    if req_pool_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "req_pool_indices must have dtype int32 or int64. "
            f"Got {req_pool_indices.dtype}."
        )
    batch_size = int(emit_ids.shape[0])
    if int(req_pool_indices.numel()) != batch_size:
        raise ValueError(
            "req_pool_indices must have one entry per emit row. "
            f"Got {int(req_pool_indices.numel())} for batch size {batch_size}."
        )
    req_generation = req_generation.view(-1).to(
        device=state.generation.device,
        dtype=state.generation.dtype,
    )
    if int(req_generation.numel()) != batch_size:
        raise ValueError(
            "req_generation must have one entry per emit row. "
            f"Got {int(req_generation.numel())} for batch size {batch_size}."
        )
    return (
        emit_ids,
        target_top1,
        active_mask_i32,
        eos_ids,
        stop_ids,
        req_pool_indices,
        req_generation,
    )


def accept_publish_control(
    *,
    state: DFlashRequestStateTable,
    req_pool_indices: torch.Tensor,
    req_generation: torch.Tensor,
    emit_ids: torch.Tensor,
    target_top1: torch.Tensor,
    active_mask: torch.Tensor | None = None,
    eos_token_ids: Iterable[int] | torch.Tensor | None = None,
    stop_token_ids: Iterable[int] | torch.Tensor | None = None,
    inplace: bool = False,
) -> DFlashAcceptPublishResult:
    (
        emit_ids,
        target_top1,
        active_mask_i32,
        eos_ids,
        stop_ids,
        req_pool_indices,
        req_generation,
    ) = _validate_accept_publish_inputs(
        state=state,
        req_pool_indices=req_pool_indices,
        req_generation=req_generation,
        emit_ids=emit_ids,
        target_top1=target_top1,
        active_mask=active_mask,
        eos_token_ids=eos_token_ids,
        stop_token_ids=stop_token_ids,
    )
    accept = accept_bonus_control(
        emit_ids=emit_ids,
        target_top1=target_top1,
        active_mask=active_mask_i32,
        eos_token_ids=eos_ids,
        stop_token_ids=stop_ids,
    )
    updated = publish_state_control(
        state=state,
        req_pool_indices=req_pool_indices,
        req_generation=req_generation,
        commit_lens=accept.commit_lens,
        bonus_ids=accept.bonus_ids,
        gpu_stop_flags=accept.gpu_stop_flags,
        inplace=inplace,
    )
    return DFlashAcceptPublishResult(accept=accept, state=updated)
