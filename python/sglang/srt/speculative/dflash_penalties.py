from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.sampling.penaltylib.frequency_penalty import BatchedFrequencyPenalizer
from sglang.srt.sampling.penaltylib.min_new_tokens import BatchedMinNewTokensPenalizer
from sglang.srt.sampling.penaltylib.presence_penalty import BatchedPresencePenalizer
from sglang.srt.sampling.penaltylib.repetition_penalty import (
    BatchedRepetitionPenalizer,
    apply_scaling_penalties,
)
from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch


class DFlashBlockPenaltyState(msgspec.Struct, frozen=True):
    """Owned effective output history for ordinary and block verification."""

    # Per-request ownership and settled output boundary for the overlap relay.
    resolved_token_lens: Optional[torch.Tensor]
    request_generations: Optional[torch.Tensor]
    additive_base: torch.Tensor
    frequency_penalties: Optional[torch.Tensor]
    presence_penalties: Optional[torch.Tensor]
    cumulated_presence: Optional[torch.Tensor]
    repetition_penalties: Optional[torch.Tensor]
    scaling_base: Optional[torch.Tensor]
    min_new_tokens: Optional[torch.Tensor]
    len_output_tokens: Optional[torch.Tensor]
    stop_token_penalties: Optional[torch.Tensor]

    @classmethod
    def from_orchestrator(
        cls,
        orchestrator,
        resolved_token_lens: Optional[torch.Tensor] = None,
        request_generations: Optional[torch.Tensor] = None,
    ) -> Optional[DFlashBlockPenaltyState]:
        """Return a snapshot of prepared penalties, or None if unused."""
        prepared = {
            penalizer_type: penalizer
            for penalizer_type, penalizer in orchestrator.penalizers.items()
            if penalizer._is_prepared
        }
        if not prepared:
            return None

        frequency = prepared.get(BatchedFrequencyPenalizer)
        presence = prepared.get(BatchedPresencePenalizer)
        repetition = prepared.get(BatchedRepetitionPenalizer)
        min_new_tokens = prepared.get(BatchedMinNewTokensPenalizer)
        if frequency is not None:
            additive_base = -frequency.cumulated_frequency_penalties
        elif presence is not None:
            additive_base = torch.zeros_like(presence.cumulated_presence_penalties)
        elif repetition is not None:
            additive_base = torch.zeros_like(repetition.cumulated_repetition_penalties)
        else:
            additive_base = torch.zeros_like(min_new_tokens.stop_token_penalties)
        if presence is not None:
            presence.apply(additive_base)

        return cls(
            resolved_token_lens=resolved_token_lens,
            request_generations=request_generations,
            additive_base=additive_base,
            frequency_penalties=(
                frequency.frequency_penalties if frequency is not None else None
            ),
            presence_penalties=(
                presence.presence_penalties if presence is not None else None
            ),
            cumulated_presence=(
                presence.cumulated_presence_penalties.clone()
                if presence is not None
                else None
            ),
            repetition_penalties=(
                repetition.repetition_penalties if repetition is not None else None
            ),
            scaling_base=(
                repetition.cumulated_repetition_penalties.clone()
                if repetition is not None
                else None
            ),
            min_new_tokens=(
                min_new_tokens.min_new_tokens if min_new_tokens is not None else None
            ),
            len_output_tokens=(
                min_new_tokens.len_output_tokens.clone()
                if min_new_tokens is not None
                else None
            ),
            stop_token_penalties=(
                min_new_tokens.stop_token_penalties
                if min_new_tokens is not None
                else None
            ),
        )

    def cumulate_pending(self, tokens: torch.Tensor, num_valid: torch.Tensor) -> None:
        """Fold an owned pending suffix into this temporary forward snapshot."""
        k = tokens.shape[1]
        valid = torch.arange(k, device=tokens.device)[None, :] < num_valid[:, None]
        vocab_size = self.additive_base.shape[1]
        invalid = valid & ((tokens < 0) | (tokens >= vocab_size))
        # CPU result handling retires invalid output rows and discards their
        # next overlap result. Keep those rows out of this temporary snapshot
        # without changing the tokens or the CPU's retained-prefix policy.
        num_valid = torch.where(invalid.any(dim=1), 0, num_valid)
        any_valid = (num_valid > 0)[:, None]
        valid = valid & any_valid
        # Redirect padding to a valid id even when graph outputs use -1 tails.
        ids = torch.where(valid, tokens, tokens[:, :1].clamp(0, vocab_size - 1))
        if self.frequency_penalties is not None:
            self.additive_base.scatter_add_(
                1, ids, -self.frequency_penalties.expand(-1, k) * valid
            )
        if self.presence_penalties is not None:
            previous = self.cumulated_presence.gather(1, ids)
            updated = torch.where(any_valid, self.presence_penalties, previous)
            # Duplicate token ids receive the same replacement, not multiple
            # additions of the presence penalty.
            self.additive_base.scatter_(
                1, ids, self.additive_base.gather(1, ids) - (updated - previous)
            )
            self.cumulated_presence.scatter_(1, ids, updated)
        if self.scaling_base is not None:
            self.scaling_base.scatter_(
                1,
                ids,
                torch.where(
                    any_valid,
                    self.repetition_penalties,
                    self.scaling_base.gather(1, ids),
                ),
            )
        if self.len_output_tokens is not None:
            self.len_output_tokens.add_(num_valid[:, None])

    def apply(self, logits: torch.Tensor) -> None:
        """Apply the effective history to the ordinary sampler's logits."""
        logits.add_(self.additive_base.to(dtype=logits.dtype))
        if self.min_new_tokens is not None:
            logits.add_(
                torch.where(
                    self.len_output_tokens < self.min_new_tokens,
                    self.stop_token_penalties,
                    0.0,
                ).to(dtype=logits.dtype)
            )
        if self.scaling_base is not None:
            apply_scaling_penalties(logits, self.scaling_base)


def _cumulate_block_candidate(
    *,
    state: DFlashBlockPenaltyState,
    tokens: torch.Tensor,
    delta_add: Optional[torch.Tensor],
    seen_presence: Optional[torch.Tensor],
    running_scale: Optional[torch.Tensor],
) -> None:
    if state.frequency_penalties is not None:
        delta_add.scatter_add_(1, tokens, -state.frequency_penalties)
    if state.presence_penalties is not None:
        new_hits = seen_presence.gather(1, tokens).eq(0)
        delta_add.scatter_add_(1, tokens, -state.presence_penalties * new_hits)
        seen_presence.scatter_(1, tokens, state.presence_penalties)
    if running_scale is not None:
        running_scale.scatter_(1, tokens, state.repetition_penalties)


def _apply_block_penalties(
    logits2d: torch.Tensor,
    state: DFlashBlockPenaltyState,
    candidates: torch.Tensor,
    bs: int,
    k: int,
) -> None:
    """Roll the committed penalty state forward one candidate at a time.

    The rolled-forward state lives in [bs, V] working buffers -- the same size
    class as the committed penalty state itself -- and each position of the
    verify block is adjusted through the [bs, V] view of its rows, so no
    adjustment plane the size of the [bs * k, V] verify logits is ever
    materialized.
    """
    vocab_size = logits2d.shape[1]
    c = candidates[:, 1:]
    logits3 = logits2d.view(bs, k, vocab_size)

    # The committed additive state is position-invariant; broadcast it once.
    logits3.add_(state.additive_base.unsqueeze(1).to(dtype=logits2d.dtype))

    delta_add = (
        torch.zeros_like(state.additive_base)
        if state.frequency_penalties is not None or state.presence_penalties is not None
        else None
    )
    seen_presence = (
        state.cumulated_presence.clone()
        if state.presence_penalties is not None
        else None
    )
    running_scale = (
        state.scaling_base.clone() if state.scaling_base is not None else None
    )

    for position in range(k):
        position_logits = logits3[:, position]
        if delta_add is not None:
            position_logits.add_(delta_add.to(dtype=position_logits.dtype))
        if state.min_new_tokens is not None:
            under_min = (state.len_output_tokens + position) < state.min_new_tokens
            position_logits.add_(
                torch.where(
                    under_min,
                    state.stop_token_penalties,
                    0.0,
                ).to(dtype=position_logits.dtype)
            )
        if running_scale is not None:
            apply_scaling_penalties(position_logits, running_scale)
        if position + 1 == k:
            break
        # Fold candidate `position + 1` into the rolled-forward state so later
        # positions are penalized as if it had been committed.
        _cumulate_block_candidate(
            state=state,
            tokens=c[:, position : position + 1],
            delta_add=delta_add,
            seen_presence=seen_presence,
            running_scale=running_scale,
        )


def prepare_dflash_penalty_state(
    *, batch: ScheduleBatch, future_map: Optional[FutureMap] = None
) -> Optional[DFlashBlockPenaltyState]:
    """Snapshot settled and pending outputs after scheduling has finalized rows."""
    orchestrator = batch.sampling_info.penalizer_orchestrator
    if not orchestrator.is_required:
        return None
    batch.cumulate_penalty_output_tokens_since_last()
    metadata = (
        torch.tensor(
            [
                [
                    len(req.origin_input_ids) + req.penalty_cumulated_len,
                    req.penalty_generation,
                ]
                for req in batch.reqs
            ],
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(batch.device),
        )
        .reshape(-1, 2)
        .to(batch.device, non_blocking=True)
    )
    state = DFlashBlockPenaltyState.from_orchestrator(
        orchestrator,
        resolved_token_lens=metadata[:, 0],
        request_generations=metadata[:, 1],
    )
    if future_map is not None:
        pending = future_map.resolve_penalty_outputs(
            indices=batch.req_pool_indices,
            resolved_token_lens=state.resolved_token_lens,
            request_generations=state.request_generations,
        )
        if pending is not None:
            state.cumulate_pending(pending.tokens, pending.num_valid)
    return state
