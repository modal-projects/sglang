from __future__ import annotations

from collections.abc import Iterable

import torch

from sglang.jit_kernel.dflash_accept_bonus import accept_bonus_cuda
from sglang.srt.speculative.dflash.contracts import DFlashAcceptBonusResult
from sglang.srt.speculative.dflash.kernels.accept_bonus_triton import (
    _validate_accept_bonus_inputs,
)


def accept_bonus_jit(
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
    batch_size = int(emit_ids.shape[0])
    device = emit_ids.device

    accept_lens = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    commit_lens = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    bonus_ids = torch.zeros((batch_size,), dtype=target_top1.dtype, device=device)
    gpu_stop_flags = torch.zeros((batch_size,), dtype=torch.int32, device=device)

    accept_bonus_cuda(
        emit_ids.contiguous(),
        target_top1.contiguous(),
        active_mask_i32.contiguous(),
        eos_ids,
        stop_ids,
        accept_lens,
        commit_lens,
        bonus_ids,
        gpu_stop_flags,
    )

    return DFlashAcceptBonusResult(
        accept_lens=accept_lens,
        commit_lens=commit_lens,
        bonus_ids=bonus_ids,
        gpu_stop_flags=gpu_stop_flags,
    )
