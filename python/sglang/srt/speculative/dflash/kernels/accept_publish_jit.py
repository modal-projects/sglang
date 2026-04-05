from __future__ import annotations

from collections.abc import Iterable

import torch

from sglang.jit_kernel.dflash_accept_publish import accept_publish_cuda
from sglang.srt.speculative.dflash.contracts import (
    DFlashAcceptBonusResult,
    DFlashAcceptPublishResult,
    DFlashRequestStateTable,
)
from sglang.srt.speculative.dflash.kernels.accept_publish import (
    _validate_accept_publish_inputs,
)


def accept_publish_jit(
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
    updated = state if inplace else state.clone()
    batch_size = int(emit_ids.shape[0])
    device = emit_ids.device

    accept_lens = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    commit_lens = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    bonus_ids = torch.zeros((batch_size,), dtype=target_top1.dtype, device=device)
    gpu_stop_flags = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    oob_flags = torch.zeros((batch_size,), dtype=torch.int32, device=device)

    accept_publish_cuda(
        updated.committed_len,
        updated.reserved_len,
        updated.next_verified_id,
        updated.generation,
        updated.status_flags,
        req_pool_indices.contiguous(),
        req_generation.contiguous(),
        emit_ids.contiguous(),
        target_top1.contiguous(),
        active_mask_i32.contiguous(),
        eos_ids.contiguous(),
        stop_ids.contiguous(),
        accept_lens,
        commit_lens,
        bonus_ids,
        gpu_stop_flags,
        oob_flags,
    )

    if bool(oob_flags.any().item()):
        raise ValueError(
            "accept_publish_jit would advance committed_len beyond reserved_len."
        )
    updated.validate()
    return DFlashAcceptPublishResult(
        accept=DFlashAcceptBonusResult(
            accept_lens=accept_lens,
            commit_lens=commit_lens,
            bonus_ids=bonus_ids,
            gpu_stop_flags=gpu_stop_flags,
        ),
        state=updated,
    )
