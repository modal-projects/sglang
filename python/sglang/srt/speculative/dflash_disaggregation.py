from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs


def build_dflash_disagg_draft_input(
    batch: ScheduleBatch,
    server_args: ServerArgs,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> DFlashDraftInputV2:
    """Rebuild DFLASH spec-v2 draft state on the decode worker after a
    disaggregated prefill.

    EAGLE ships per-request ``topk_p`` / ``topk_index`` / last hidden state
    through the metadata buffers and reconstructs an ``EagleDraftInput`` from
    them (see ``eagle_disaggregation.build_eagle_disagg_draft_input``). DFLASH
    is different: it grafts onto the *target* model's hidden states, which the
    prefill worker materializes directly into the draft KV pool
    (``DFlashWorkerV2.forward_batch_generation`` extend branch ->
    ``_append_target_hidden_to_draft_kv_by_loc``). That draft KV pool is
    transferred to the decode worker alongside the target KV at the same slot
    indices — ``prefill.py`` / ``decode.py`` register ``draft_token_to_kv_pool``
    for transfer — so the hidden states arrive for free.

    What's left to rebuild here is only the lightweight per-iteration draft
    state that does *not* live in the KV pool: the verified (bonus) token
    produced by prefill and the committed sequence lengths. ``topk_p`` /
    ``topk_index`` / ``hidden_states`` are unused at decode start and are
    created empty, exactly as ``DFlashWorkerV2._make_next_draft_input_prefill``
    does in the aggregated path.
    """
    bs = len(batch.reqs)
    device = batch.device

    # Mirrors DFlashWorkerV2._make_next_draft_input_prefill: only the bonus
    # token and committed sequence lengths carry real state; there is no
    # in-flight verify event to synchronize on (the draft KV transfer, not a
    # local forward, is what gated this batch becoming schedulable).
    spec_info = DFlashDraftInputV2(
        topk_p=torch.empty((bs, 0), device=device, dtype=torch.float32),
        topk_index=torch.empty((bs, 0), device=device, dtype=torch.int64),
        bonus_tokens=last_tokens_tensor.to(dtype=torch.int64),
        new_seq_lens=batch.seq_lens.to(dtype=torch.int64),
        hidden_states=torch.empty((bs, 0), device=device, dtype=torch.float16),
    )

    # Mirror the per-iteration overlap bookkeeping the scheduler performs after
    # every spec-v2 forward (scheduler.py: `batch.spec_info.future_indices =
    # future_indices`). The running decode batch this prebuilt batch may be
    # merged into already carries `future_indices`, and DFlashDraftInputV2.merge
    # asserts both sides do; populating the FutureMap also makes bonus_tokens /
    # new_seq_lens resolvable once a filter/merge flips `direct_carry_valid` off.
    if batch.enable_overlap:
        spec_info.future_indices = batch.req_pool_indices
        future_map.publish(spec_info.future_indices, batch.seq_lens)
        future_map.stash(spec_info.future_indices, spec_info)

    return spec_info
