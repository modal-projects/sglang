import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import pytest
import torch

from sglang.srt.constrained.base_grammar_backend import GrammarMask
from sglang.srt.sampling.penaltylib.frequency_penalty import (
    BatchedFrequencyPenalizer,
)
from sglang.srt.sampling.penaltylib.min_new_tokens import (
    BatchedMinNewTokensPenalizer,
)
from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
)
from sglang.srt.sampling.penaltylib.presence_penalty import (
    BatchedPresencePenalizer,
)
from sglang.srt.sampling.penaltylib.repetition_penalty import (
    BatchedRepetitionPenalizer,
    apply_scaling_penalties,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams
from sglang.srt.speculative.dflash_utils import (
    DFlashBlockPenaltyState,
    apply_dflash_verify_logits_adjustments,
)
from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.srt.speculative.dspark_components.dspark_planner import (
    VerifyWindow,
    apply_logits_adjustments_strided,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    TargetVerifyExecutor,
    TargetVerifyResult,
    verify_logits_adjustments_are_noop,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _MaskFillGrammar:
    def apply_vocab_mask(self, *, logits, vocab_mask):
        logits.masked_fill_(~vocab_mask, float("-inf"))


def _make_req(
    *,
    repetition_penalty=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    min_new_tokens=0,
):
    return SimpleNamespace(
        sampling_params=SamplingParams(
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            min_new_tokens=min_new_tokens,
        ),
        tokenizer=SimpleNamespace(
            additional_stop_token_ids=None,
            eos_token_id=2,
        ),
        eos_token_ids=None,
        penalty_cumulated_len=0,
    )


def _make_batch(reqs):
    class FakeBatch:
        pass

    batch = FakeBatch()
    batch.reqs = reqs
    batch.device = torch.device("cpu")
    return batch


def _make_orchestrator(reqs, vocab_size=16):
    return BatchedPenalizerOrchestrator(
        vocab_size,
        _make_batch(reqs),
        {
            BatchedFrequencyPenalizer,
            BatchedMinNewTokensPenalizer,
            BatchedPresencePenalizer,
            BatchedRepetitionPenalizer,
        },
    )


def _make_sampling_info(batch_size, vocab_size, **overrides):
    values = dict(
        temperatures=torch.ones(batch_size, 1),
        top_ps=torch.ones(batch_size),
        top_ks=torch.full((batch_size,), TOP_K_ALL, dtype=torch.int32),
        min_ps=torch.zeros(batch_size),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=vocab_size,
        device="cpu",
        penalizer_orchestrator=None,
    )
    values.update(overrides)
    return SamplingBatchInfo(**values)


def _assert_logits_equal(actual, expected):
    torch.testing.assert_close(actual, expected)


def _make_reference(orchestrator, logits, candidates):
    reference = torch.empty_like(logits)
    for position in range(candidates.shape[1]):
        row = logits[:, position].clone()
        orchestrator.accumulate_additive_penalties(row)
        scaling = orchestrator.accumulate_scaling_penalties()
        if scaling is not None:
            apply_scaling_penalties(row, scaling)
        reference[:, position] = row
        if position + 1 < candidates.shape[1]:
            orchestrator.cumulate_output_tokens(candidates[:, position + 1])
    return reference


def _make_block_case():
    reqs = [
        _make_req(
            repetition_penalty=1.5,
            frequency_penalty=0.3,
            presence_penalty=0.4,
            min_new_tokens=3,
        ),
        _make_req(
            repetition_penalty=1.5,
            frequency_penalty=0.3,
            presence_penalty=0.4,
            min_new_tokens=6,
        ),
    ]
    committed = torch.tensor(
        [
            [5, 7],
            [4, 8],
        ],
        dtype=torch.int64,
    )
    candidates = torch.tensor(
        [
            [7, 5, 9, 5],
            [8, 1, 10, 1],
        ],
        dtype=torch.int64,
    )
    return reqs, committed, candidates


def _prepare_orchestrator(reqs, committed):
    orchestrator = _make_orchestrator(reqs)
    for token_column in committed.T:
        orchestrator.cumulate_output_tokens(token_column)
    return orchestrator


class TestDFlashBlockPenalties(CustomTestCase):
    def test_block_matches_token_by_token_all_penalties(self):
        torch.manual_seed(0)
        bs, k, vocab_size = 2, 4, 16
        reqs, committed, candidates = _make_block_case()
        logits = torch.randn(bs, k, vocab_size, dtype=torch.float32)

        overlap_orchestrator = _prepare_orchestrator(reqs, committed)
        state = DFlashBlockPenaltyState.from_orchestrator(overlap_orchestrator)
        additive = torch.zeros(bs, vocab_size)
        overlap_orchestrator.accumulate_additive_penalties(additive)
        sampling_info = _make_sampling_info(
            bs,
            vocab_size,
            acc_additive_penalties=additive,
            acc_scaling_penalties=overlap_orchestrator.accumulate_scaling_penalties(),
            dflash_block_penalty_state=state,
        )
        overlap_logits = logits.clone().view(bs * k, vocab_size)
        apply_dflash_verify_logits_adjustments(
            next_token_logits=overlap_logits,
            sampling_info=sampling_info,
            draft_token_num=k,
            candidates=candidates,
        )
        overlap_reference = _make_reference(
            _prepare_orchestrator(reqs, committed),
            logits,
            candidates,
        )
        _assert_logits_equal(overlap_logits.view(bs, k, vocab_size), overlap_reference)

        non_overlap_orchestrator = _prepare_orchestrator(reqs, committed)
        non_overlap_info = _make_sampling_info(
            bs,
            vocab_size,
            penalizer_orchestrator=non_overlap_orchestrator,
        )
        non_overlap_logits = logits.clone().view(bs * k, vocab_size)
        apply_dflash_verify_logits_adjustments(
            next_token_logits=non_overlap_logits,
            sampling_info=non_overlap_info,
            draft_token_num=k,
            candidates=candidates,
        )
        _assert_logits_equal(
            non_overlap_logits.view(bs, k, vocab_size), overlap_reference
        )

    def test_position_zero_matches_broadcast_path(self):
        torch.manual_seed(1)
        bs, k, vocab_size = 2, 4, 16
        reqs, committed, candidates = _make_block_case()
        orchestrator = _prepare_orchestrator(reqs, committed)
        sampling_info = _make_sampling_info(
            bs,
            vocab_size,
            penalizer_orchestrator=orchestrator,
        )
        logits = torch.randn(bs, k, vocab_size)
        # Legacy (pre-fix) broadcast: position-invariant penalties applied to the
        # real logits. Computed inline because upstream's dense-fallback path in
        # apply_dflash_verify_logits_adjustments applies scaling penalties to a
        # zeros buffer (where they vanish) instead of to the logits.
        broadcast = logits.clone().view(bs * k, vocab_size)
        additive = torch.zeros(bs, vocab_size)
        orchestrator.accumulate_additive_penalties(additive)
        broadcast.view(bs, k, vocab_size).add_(additive[:, None, :])
        scaling = orchestrator.accumulate_scaling_penalties()
        if scaling is not None:
            apply_scaling_penalties(
                broadcast, torch.repeat_interleave(scaling, k, dim=0)
            )
        block = logits.clone().view(bs * k, vocab_size)
        apply_dflash_verify_logits_adjustments(
            next_token_logits=block,
            sampling_info=sampling_info,
            draft_token_num=k,
            candidates=candidates,
        )
        _assert_logits_equal(
            block.view(bs, k, vocab_size)[:, 0], broadcast.view(bs, k, vocab_size)[:, 0]
        )
        # The broadcast (pre-fix) path ignores the preceding candidates, so it must
        # disagree with the token-by-token reference at every later position.
        reference = _make_reference(
            _prepare_orchestrator(reqs, committed), logits, candidates
        )
        for position in range(1, k):
            assert not torch.equal(
                broadcast.view(bs, k, vocab_size)[:, position], reference[:, position]
            )

    def test_anchor_unresolved_fold_matches_prefed_anchor(self):
        torch.manual_seed(0)
        bs, k, vocab_size = 2, 4, 16
        reqs, committed, candidates = _make_block_case()
        logits = torch.randn(bs, k, vocab_size, dtype=torch.float32)
        # candidates[:, 0] is the anchor: the last committed token of each row.
        assert torch.equal(candidates[:, 0], committed[:, -1])

        # Overlap: the anchor is committed but not yet resolved into output_ids,
        # so the cursor never fed it and the snapshot was built without it. The
        # block path must fold it for the rows marked unresolved, matching the
        # token-by-token reference where the anchor was fed up front.
        overlap_orchestrator = _prepare_orchestrator(reqs, committed[:, :-1])
        state = DFlashBlockPenaltyState.from_orchestrator(
            overlap_orchestrator,
            resolved_token_lens=torch.tensor([3, 3]),
        )
        state.cumulate_pending(committed[:, -1:], torch.tensor([1, 1]))
        sampling_info = _make_sampling_info(
            bs,
            vocab_size,
            dflash_block_penalty_state=state,
        )
        overlap_logits = logits.clone().view(bs * k, vocab_size)
        apply_dflash_verify_logits_adjustments(
            next_token_logits=overlap_logits,
            sampling_info=sampling_info,
            draft_token_num=k,
            candidates=candidates,
        )
        reference = _make_reference(
            _prepare_orchestrator(reqs, committed), logits, candidates
        )
        _assert_logits_equal(overlap_logits.view(bs, k, vocab_size), reference)

    def test_anchor_unresolved_fold_masked_rows(self):
        torch.manual_seed(0)
        bs, k, vocab_size = 2, 4, 16
        reqs, committed, candidates = _make_block_case()
        logits = torch.randn(bs, k, vocab_size, dtype=torch.float32)

        # Row 0's anchor is unresolved (never fed); row 1's anchor was already
        # fed through the multi-token path. Only row 0 may be folded.
        mixed_orchestrator = _prepare_orchestrator(reqs, committed[:, :-1])
        mixed_orchestrator.cumulate_output_tokens_multi(
            committed[:, -1:].clone(), torch.tensor([0, 1], dtype=torch.int64)
        )
        state = DFlashBlockPenaltyState.from_orchestrator(
            mixed_orchestrator,
            resolved_token_lens=torch.tensor([3, 4]),
        )
        state.cumulate_pending(committed[:, -1:], torch.tensor([1, 0]))
        sampling_info = _make_sampling_info(
            bs,
            vocab_size,
            dflash_block_penalty_state=state,
        )
        mixed_logits = logits.clone().view(bs * k, vocab_size)
        apply_dflash_verify_logits_adjustments(
            next_token_logits=mixed_logits,
            sampling_info=sampling_info,
            draft_token_num=k,
            candidates=candidates,
        )
        reference = _make_reference(
            _prepare_orchestrator(reqs, committed), logits, candidates
        )
        _assert_logits_equal(mixed_logits.view(bs, k, vocab_size), reference)

    def test_noop_predicate_sees_block_penalty_state(self):
        # copy_for_forward drops the orchestrator but keeps the per-step block
        # state; the DSpark graph-folding predicate must still see that penalties
        # are active, or greedy acceptance would run on raw logits inside the
        # cuda graph.
        bs, vocab_size = 2, 16
        reqs, committed, _ = _make_block_case()
        orchestrator = _prepare_orchestrator(reqs, committed)
        state = DFlashBlockPenaltyState.from_orchestrator(orchestrator)

        forward_copy = _make_sampling_info(
            bs,
            vocab_size,
            penalizer_orchestrator=None,
            dflash_block_penalty_state=state,
        )
        assert not verify_logits_adjustments_are_noop(forward_copy)

        no_penalty_copy = _make_sampling_info(
            bs,
            vocab_size,
            penalizer_orchestrator=None,
            dflash_block_penalty_state=None,
        )
        assert verify_logits_adjustments_are_noop(no_penalty_copy)

    def test_single_token_block_uses_block_path(self):
        torch.manual_seed(0)
        bs, k, vocab_size = 2, 1, 16
        reqs, committed, candidates = _make_block_case()
        candidates = candidates[:, :1]
        logits = torch.randn(bs, k, vocab_size, dtype=torch.float32)

        orchestrator = _prepare_orchestrator(reqs, committed)
        state = DFlashBlockPenaltyState.from_orchestrator(orchestrator)
        sampling_info = _make_sampling_info(
            bs,
            vocab_size,
            penalizer_orchestrator=orchestrator,
            dflash_block_penalty_state=state,
        )
        block_logits = logits.clone().view(bs * k, vocab_size)
        apply_dflash_verify_logits_adjustments(
            next_token_logits=block_logits,
            sampling_info=sampling_info,
            draft_token_num=k,
            candidates=candidates,
        )
        reference = _make_reference(
            _prepare_orchestrator(reqs, committed), logits, candidates
        )
        _assert_logits_equal(block_logits.view(bs, k, vocab_size), reference)
        # Repetition scaling must engage on the real logits; the dense fallback
        # applies scaling to a zeros buffer, where it vanishes.
        for row in range(bs):
            token = committed[row, -1]
            assert block_logits[row, token] != logits[row, 0, token]

    def test_no_penalty_path_is_byte_identical(self):
        bs, k, vocab_size = 2, 4, 16
        vocab_mask = torch.ones(bs, vocab_size, dtype=torch.bool)
        vocab_mask[0, 3] = False
        vocab_mask[1, 7] = False
        logit_bias = torch.randn(bs, vocab_size)

        sampling_info = _make_sampling_info(
            bs,
            vocab_size,
            grammar_mask=GrammarMask(_MaskFillGrammar(), vocab_mask),
            logit_bias=logit_bias,
        )
        candidates = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int64)
        logits = torch.randn(bs * k, vocab_size)
        with_candidates = logits.clone()
        without_candidates = logits.clone()
        apply_dflash_verify_logits_adjustments(
            next_token_logits=with_candidates,
            sampling_info=sampling_info,
            draft_token_num=k,
            candidates=candidates,
        )
        apply_dflash_verify_logits_adjustments(
            next_token_logits=without_candidates,
            sampling_info=sampling_info,
            draft_token_num=k,
        )
        assert torch.equal(with_candidates, without_candidates)

        no_penalty_orchestrator = _make_orchestrator(
            [_make_req(), _make_req()],
            vocab_size,
        )
        assert (
            DFlashBlockPenaltyState.from_orchestrator(no_penalty_orchestrator) is None
        )

    def test_block_path_still_applies_vocab_mask_and_logit_bias(self):
        torch.manual_seed(2)
        bs, k, vocab_size = 2, 4, 16
        reqs, committed, candidates = _make_block_case()
        vocab_mask = torch.ones(bs, vocab_size, dtype=torch.bool)
        vocab_mask[0, 3] = False
        vocab_mask[1, 11] = False
        logit_bias = torch.randn(bs, vocab_size)

        sampling_info = _make_sampling_info(
            bs,
            vocab_size,
            penalizer_orchestrator=_prepare_orchestrator(reqs, committed),
            grammar_mask=GrammarMask(_MaskFillGrammar(), vocab_mask),
            logit_bias=logit_bias,
        )
        logits = torch.randn(bs, k, vocab_size)
        block = logits.clone().view(bs * k, vocab_size)
        apply_dflash_verify_logits_adjustments(
            next_token_logits=block,
            sampling_info=sampling_info,
            draft_token_num=k,
            candidates=candidates,
        )
        reference = _make_reference(
            _prepare_orchestrator(reqs, committed), logits, candidates
        )
        reference.add_(logit_bias[:, None, :])
        reference.masked_fill_(~vocab_mask[:, None, :], float("-inf"))
        _assert_logits_equal(block.view(bs, k, vocab_size), reference)

    def test_strided_wrapper_forwards_candidates_into_block_path(self):
        # The DSpark compact-verify path reaches the adjustments through
        # apply_logits_adjustments_strided; dropping candidates there silently
        # falls back to the position-invariant dense path.
        torch.manual_seed(3)
        bs, k, vocab_size = 2, 4, 16
        reqs, committed, candidates = _make_block_case()
        sampling_info = _make_sampling_info(
            bs,
            vocab_size,
            penalizer_orchestrator=_prepare_orchestrator(reqs, committed),
        )
        logits = torch.randn(bs, k, vocab_size)
        strided = logits.clone().view(bs * k, vocab_size)
        apply_logits_adjustments_strided(
            next_token_logits=strided,
            sampling_info=sampling_info,
            verify_num_draft_tokens=k,
            candidates=candidates,
        )
        reference = _make_reference(
            _prepare_orchestrator(reqs, committed), logits, candidates
        )
        _assert_logits_equal(strided.view(bs, k, vocab_size), reference)

    def test_forward_copy_snapshot_is_isolated(self):
        reqs, committed, _ = _make_block_case()
        orchestrator = _prepare_orchestrator(reqs, committed)
        state = DFlashBlockPenaltyState.from_orchestrator(orchestrator)
        presence = state.cumulated_presence.clone()
        scaling = state.scaling_base.clone()
        lengths = state.len_output_tokens.clone()

        orchestrator.cumulate_output_tokens(torch.tensor([9, 10], dtype=torch.int64))

        assert torch.equal(state.cumulated_presence, presence)
        assert torch.equal(state.scaling_base, scaling)
        assert torch.equal(state.len_output_tokens, lengths)

    def test_candidates_shape_mismatch_raises(self):
        reqs, committed, candidates = _make_block_case()
        orchestrator = _prepare_orchestrator(reqs, committed)
        sampling_info = _make_sampling_info(
            len(reqs),
            16,
            dflash_block_penalty_state=DFlashBlockPenaltyState.from_orchestrator(
                orchestrator
            ),
        )
        logits = torch.randn(2 * 4, 16)
        with pytest.raises(ValueError, match="candidates shape mismatch"):
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits,
                sampling_info=sampling_info,
                draft_token_num=4,
                candidates=candidates[:, :-1],
            )

    def test_penalty_state_rows_mismatch_raises(self):
        reqs, committed, candidates = _make_block_case()
        orchestrator = _prepare_orchestrator(reqs, committed)
        state = DFlashBlockPenaltyState.from_orchestrator(orchestrator)
        sampling_info = _make_sampling_info(
            len(reqs),
            16,
            dflash_block_penalty_state=msgspec.structs.replace(
                state,
                additive_base=state.additive_base[:1],
            ),
        )
        with pytest.raises(ValueError, match="penalty state rows mismatch"):
            apply_dflash_verify_logits_adjustments(
                next_token_logits=torch.randn(2 * 4, 16),
                sampling_info=sampling_info,
                draft_token_num=4,
                candidates=candidates,
            )

    def test_individual_penalties_match_sequential_reference(self):
        cases = [
            dict(frequency_penalty=-0.5),
            dict(presence_penalty=-0.5),
            dict(repetition_penalty=0.5),
            dict(min_new_tokens=4),
            dict(frequency_penalty=0.5, presence_penalty=0.5, repetition_penalty=2.0),
        ]
        for params in cases:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for k in (1, 4):
                    with self.subTest(params=params, dtype=dtype, block_size=k):
                        reqs = [_make_req(**params), _make_req()]
                        committed = torch.tensor([[5, 7], [4, 8]])
                        candidates = torch.tensor([[7, 0, 0, 5], [8, 1, 2, 3]])[:, :k]
                        logits = torch.linspace(-4, 4, 16, dtype=dtype).repeat(2, k, 1)
                        info = _make_sampling_info(
                            2,
                            16,
                            penalizer_orchestrator=_prepare_orchestrator(
                                reqs, committed
                            ),
                        )
                        expected = _make_reference(
                            _prepare_orchestrator(reqs, committed), logits, candidates
                        )
                        actual = logits.reshape(2 * k, 16).clone()
                        apply_dflash_verify_logits_adjustments(
                            next_token_logits=actual,
                            sampling_info=info,
                            draft_token_num=k,
                            candidates=candidates,
                        )
                        _assert_logits_equal(actual.view(2, k, 16), expected)

    def test_dspark_executor_consumers_apply_candidates_and_pending_anchor(self):
        """Both executor routes must adjust the logits used by acceptance."""
        bs, k, vocab_size = 2, 4, 16
        reqs, committed, candidates = _make_block_case()
        logits = torch.linspace(-2, 3, vocab_size).repeat(bs, k, 1)
        expected = _make_reference(
            _prepare_orchestrator(reqs, committed), logits, candidates
        )
        for compact in (False, True):
            for graph_output in (False, True) if compact else (False,):
                with self.subTest(compact=compact, graph_output=graph_output):
                    orchestrator = _prepare_orchestrator(reqs, committed[:, :-1])
                    orchestrator.cumulate_output_tokens_multi(
                        committed[:, -1:], torch.tensor([0, 1])
                    )
                    state = DFlashBlockPenaltyState.from_orchestrator(orchestrator)
                    state.cumulate_pending(committed[:, -1:], torch.tensor([1, 0]))
                    info = _make_sampling_info(
                        bs,
                        vocab_size,
                        dflash_block_penalty_state=state,
                    )
                    prefix_lens = torch.tensor([3, 3])
                    batch = SimpleNamespace(
                        seq_lens=prefix_lens,
                        seq_lens_cpu=prefix_lens.clone(),
                        seq_lens_sum=6,
                    )
                    hidden = torch.arange(bs * k * 3).reshape(bs * k, 3).float()
                    verify_lens = [4, 2]
                    layout = RaggedVerifyLayout.from_verify_lens(
                        verify_lens_cpu=verify_lens,
                        device=torch.device("cpu"),
                        grid=[8],
                    )
                    compact_logits = torch.cat(
                        [logits[0], logits[1, :2], torch.zeros(2, vocab_size)]
                    )
                    compact_hidden = torch.cat(
                        [hidden[:4], hidden[4:6], torch.zeros(2, 3)]
                    )
                    result = TargetVerifyResult(
                        logits_output=SimpleNamespace(
                            next_token_logits=compact_logits
                            if compact
                            else logits.reshape(-1, vocab_size).clone(),
                            hidden_states=compact_hidden if compact else hidden.clone(),
                        ),
                        can_run_cuda_graph=graph_output,
                    )
                    executor = TargetVerifyExecutor(
                        target_worker=None,
                        gamma=k - 1,
                        verify_num_draft_tokens=k,
                        model_runner=None,
                        kv_injector=None,
                        tp_sync=None,
                    )
                    executor._verify_backend_self_adds_seq_lens_cache = True
                    if graph_output:
                        executor.verify_epilogue = SimpleNamespace(
                            begin_step=lambda *args, **kwargs: None,
                            strided_logits=logits.reshape(-1, vocab_size).clone(),
                            strided_hidden=hidden.clone(),
                        )
                    if compact:
                        # Substitute model execution and KV allocation only; the
                        # executor's ragged scatter and penalty consumer run normally.
                        with (
                            patch.object(
                                dspark_verify.BuildRaggedVerifyWindow,
                                "execute",
                                return_value=None,
                            ),
                            patch.object(executor, "_run_ragged", return_value=result),
                        ):
                            actual, _ = executor.run_compact(
                                batch=batch,
                                layout=layout,
                                draft_block_ids=candidates[:, :-1],
                                draft_tokens=candidates[:, 1:],
                                verify_ids_2d=candidates,
                                bs=bs,
                                device="cpu",
                                sampling_info=info,
                            )
                    else:
                        window = VerifyWindow(
                            positions_2d=prefix_lens[:, None] + torch.arange(k),
                            verify_cache_loc=torch.arange(bs * k),
                            verify_cache_loc_2d=torch.arange(bs * k).view(bs, k),
                        )
                        with patch.object(
                            executor, "_forward_prepared_verify", return_value=result
                        ):
                            actual = executor.run_non_compact(
                                batch=batch,
                                draft_input=None,
                                verify_ids_2d=candidates,
                                verify_window=window,
                                sampling_info=info,
                            )
                    adjusted = actual.logits_output.next_token_logits.view(
                        bs, k, vocab_size
                    )
                    for row, count in enumerate(verify_lens if compact else [k, k]):
                        _assert_logits_equal(
                            adjusted[row, :count], expected[row, :count]
                        )


if __name__ == "__main__":
    unittest.main()
