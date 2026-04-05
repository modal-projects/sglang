from __future__ import annotations

from collections.abc import Iterable

import torch

from sglang.srt.speculative.dflash.contracts import (
    STATUS_EOS_SEEN,
    STATUS_FINISHED,
    STATUS_STOPPED_BY_TOKEN,
    DFlashAcceptBonusResult,
)


def _normalize_token_ids(
    token_ids: Iterable[int] | torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if token_ids is None:
        return None
    if isinstance(token_ids, torch.Tensor):
        return token_ids.to(device=device, dtype=dtype).view(-1)
    token_ids = list(int(x) for x in token_ids)
    if not token_ids:
        return None
    return torch.tensor(token_ids, device=device, dtype=dtype)


def accept_bonus_control(
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

    matches = emit_ids[:, 1:] == target_top1[:, :-1]
    accept_lens = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    commit_lens = torch.where(
        active_mask,
        accept_lens + 1,
        torch.zeros_like(accept_lens),
    )

    safe_accept_idx = accept_lens.clamp(min=0, max=block_size - 1).to(torch.int64)
    bonus_ids = target_top1.gather(1, safe_accept_idx[:, None]).squeeze(1)
    bonus_ids = torch.where(active_mask, bonus_ids, torch.zeros_like(bonus_ids))

    row_offsets = torch.arange(block_size, device=device, dtype=torch.int32)[None, :]
    committed_mask = active_mask[:, None] & (row_offsets < commit_lens[:, None])
    gpu_stop_flags = torch.zeros((bs,), dtype=torch.int32, device=device)

    eos_tensor = _normalize_token_ids(
        eos_token_ids,
        device=device,
        dtype=emit_ids.dtype,
    )
    if eos_tensor is not None and eos_tensor.numel() > 0:
        eos_hits = committed_mask & (emit_ids[:, :, None] == eos_tensor).any(dim=-1)
        eos_rows = eos_hits.any(dim=1)
        gpu_stop_flags |= eos_rows.to(torch.int32) * (STATUS_EOS_SEEN | STATUS_FINISHED)

    stop_tensor = _normalize_token_ids(
        stop_token_ids,
        device=device,
        dtype=emit_ids.dtype,
    )
    if stop_tensor is not None and stop_tensor.numel() > 0:
        stop_hits = committed_mask & (emit_ids[:, :, None] == stop_tensor).any(dim=-1)
        stop_rows = stop_hits.any(dim=1)
        gpu_stop_flags |= stop_rows.to(torch.int32) * (
            STATUS_STOPPED_BY_TOKEN | STATUS_FINISHED
        )

    accept_lens = torch.where(
        active_mask,
        accept_lens,
        torch.zeros_like(accept_lens),
    )
    return DFlashAcceptBonusResult(
        accept_lens=accept_lens,
        commit_lens=commit_lens,
        bonus_ids=bonus_ids,
        gpu_stop_flags=gpu_stop_flags,
    )
