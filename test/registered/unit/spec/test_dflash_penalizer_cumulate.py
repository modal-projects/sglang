"""Committed output accounting and overlap snapshot regressions."""

import unittest
from array import array
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.sampling.penaltylib.frequency_penalty import BatchedFrequencyPenalizer
from sglang.srt.sampling.penaltylib.min_new_tokens import BatchedMinNewTokensPenalizer
from sglang.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator
from sglang.srt.sampling.penaltylib.presence_penalty import BatchedPresencePenalizer
from sglang.srt.sampling.penaltylib.repetition_penalty import BatchedRepetitionPenalizer
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative import spec_utils
from sglang.srt.speculative.dflash_penalties import prepare_dflash_penalty_state
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _PenaltyBatch:
    cumulate_penalty_output_tokens_since_last = (
        ScheduleBatch.cumulate_penalty_output_tokens_since_last
    )

    def __init__(self, reqs, *, overlap=False):
        self.reqs = reqs
        self.device = torch.device("cpu")
        self.enable_overlap = overlap
        self.spec_algorithm = SimpleNamespace(is_dflash_family=lambda: True)
        # Allocation is outside this CPU test's contract.
        self.spec_info = SimpleNamespace(prepare_for_decode=lambda batch: None)
        self.sampling_info = SamplingBatchInfo(
            temperatures=torch.ones(len(reqs), 1),
            top_ps=torch.ones(len(reqs)),
            top_ks=torch.ones(len(reqs), dtype=torch.int32),
            min_ps=torch.zeros(len(reqs)),
            is_all_greedy=True,
            is_any_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=16,
            device="cpu",
            penalizer_orchestrator=BatchedPenalizerOrchestrator(
                16,
                self,
                {
                    BatchedFrequencyPenalizer,
                    BatchedMinNewTokensPenalizer,
                    BatchedPresencePenalizer,
                    BatchedRepetitionPenalizer,
                },
            ),
        )

    @property
    def orchestrator(self):
        return self.sampling_info.penalizer_orchestrator

    def prepare(self):
        with patch.object(
            spec_utils,
            "get_exec",
            return_value=SimpleNamespace(
                mamba=SimpleNamespace(enable_mamba_extra_buffer_lazy=False)
            ),
        ):
            spec_utils.spec_prepare_for_decode(self)
        self.sampling_info.dflash_block_penalty_state = prepare_dflash_penalty_state(
            batch=self
        )


def _make_req(*, outputs=(), enabled=True, input_embeds=None):
    req = Req(
        rid="synthetic",
        origin_input_text="",
        origin_input_ids=array("q", [10, 11]),
        sampling_params=SamplingParams(
            frequency_penalty=0.25 if enabled else 0.0,
            presence_penalty=-0.5 if enabled else 0.0,
            repetition_penalty=1.5 if enabled else 1.0,
            min_new_tokens=4 if enabled else 0,
        ),
        input_embeds=input_embeds,
        vocab_size=16,
    )
    req.tokenizer = SimpleNamespace(additional_stop_token_ids=None, eos_token_id=2)
    req.output_ids.extend(outputs)
    req.kv.kv_committed_len = len(req.origin_input_ids) + max(len(outputs) - 1, 0)
    return req


def _reference_logits(req, history, logits):
    result = logits.clone()
    counts = Counter(history)
    params = req.sampling_params
    for token, count in counts.items():
        result[token] -= params.frequency_penalty * count + params.presence_penalty
    if len(history) < params.min_new_tokens:
        result[2] = float("-inf")
    for token in counts:
        result[token] = (
            result[token] * params.repetition_penalty
            if result[token] < 0
            else result[token] / params.repetition_penalty
        )
    return result


def _assert_state(batch, histories):
    logits = torch.linspace(-2, 3, 16).repeat(len(histories), 1)
    batch.orchestrator.accumulate_additive_penalties(logits)
    scaling = batch.orchestrator.accumulate_scaling_penalties()
    if scaling is not None:
        logits = torch.where(logits < 0, logits * scaling, logits / scaling)
    for row, (req, history) in enumerate(zip(batch.reqs, histories)):
        torch.testing.assert_close(
            logits[row], _reference_logits(req, history, torch.linspace(-2, 3, 16))
        )


class TestDFlashPenalizerCumulate(CustomTestCase):
    def test_ragged_outputs_count_once_without_prompt_or_padding(self):
        reqs = [_make_req(outputs=[5, 7, 7]), _make_req()]
        batch = _PenaltyBatch(reqs)
        batch.prepare()
        _assert_state(batch, [[5, 7, 7], []])
        self.assertEqual([r.penalty_cumulated_len for r in reqs], [3, 0])

        reqs[0].output_ids.extend([0, 5])
        reqs[1].output_ids.append(0)
        batch.prepare()
        _assert_state(batch, [[5, 7, 7, 0, 5], [0]])
        batch.prepare()
        _assert_state(batch, [[5, 7, 7, 0, 5], [0]])
        self.assertEqual([r.penalty_cumulated_len for r in reqs], [5, 1])

    def test_unresolved_output_is_noop_including_repeated_empty_steps(self):
        batch = _PenaltyBatch([_make_req()], overlap=True)
        batch.prepare()
        batch.prepare()
        _assert_state(batch, [[]])
        self.assertEqual(batch.reqs[0].penalty_cumulated_len, 0)

    def test_multi_token_matches_single_token_for_ragged_rows(self):
        histories = [[0, 7, 7, 5], [], [4, 0]]
        batch = _PenaltyBatch([_make_req() for _ in histories])
        batch.orchestrator.cumulate_output_tokens_multi(
            torch.tensor([[0, 7, 7, 5], [0, 0, 0, 0], [4, 0, 0, 0]]),
            torch.tensor([4, 0, 2]),
        )
        _assert_state(batch, histories)
        for row, history in enumerate(histories):
            single = _PenaltyBatch([_make_req()])
            for token in history:
                single.orchestrator.cumulate_output_tokens(torch.tensor([token]))
            for kind, field in (
                (BatchedFrequencyPenalizer, "cumulated_frequency_penalties"),
                (BatchedPresencePenalizer, "cumulated_presence_penalties"),
                (BatchedRepetitionPenalizer, "cumulated_repetition_penalties"),
                (BatchedMinNewTokensPenalizer, "len_output_tokens"),
            ):
                torch.testing.assert_close(
                    getattr(batch.orchestrator.penalizers[kind], field)[row],
                    getattr(single.orchestrator.penalizers[kind], field)[0],
                )

    def test_cursor_clamps_after_output_truncation(self):
        req = _make_req(outputs=[1, 2, 3])
        batch = _PenaltyBatch([req])
        batch.prepare()
        req.output_ids = req.output_ids[:1]
        batch.prepare()
        self.assertEqual(req.penalty_cumulated_len, 1)
        req.output_ids.extend([9, 8])
        batch.prepare()
        self.assertEqual(req.penalty_cumulated_len, 3)
        # Rewind rebuilds the retained history instead of keeping removed tokens.
        _assert_state(batch, [[1, 9, 8]])

    def test_retraction_rebuilds_history_with_a_fresh_orchestrator(self):
        for embeds in (None, [[0.0], [1.0]]):
            with self.subTest(input_embeds=embeds):
                req = _make_req(outputs=[5, 7, 7], input_embeds=embeds)
                batch = _PenaltyBatch([req])
                batch.prepare()
                req.reset_for_retract()
                self.assertEqual(req.penalty_cumulated_len, 0)
                rebuilt = _PenaltyBatch([req])
                rebuilt.prepare()
                _assert_state(rebuilt, [list(req.output_ids)])
                self.assertEqual(list(req.output_ids), [] if embeds else [5, 7, 7])

    def test_filter_merge_preserves_cursors_and_snapshot_values(self):
        reqs = [_make_req(outputs=[5, 5]), _make_req(outputs=[4]), _make_req()]
        batch = _PenaltyBatch(reqs)
        batch.prepare()
        old_snapshot = batch.sampling_info.dflash_block_penalty_state
        old_additive = old_snapshot.additive_base.clone()
        batch.reqs = [reqs[2], reqs[0]]
        batch.orchestrator.filter(torch.tensor([2, 0]))
        added = _PenaltyBatch([_make_req(outputs=[9, 0])])
        added.prepare()
        batch.orchestrator.merge(added.orchestrator)
        batch.reqs = batch.reqs + added.reqs
        batch.reqs[0].output_ids.append(8)
        batch.reqs[1].output_ids.append(0)
        batch.prepare()
        _assert_state(batch, [[8], [5, 5, 0], [9, 0]])
        self.assertEqual([r.penalty_cumulated_len for r in batch.reqs], [1, 3, 2])
        torch.testing.assert_close(old_snapshot.additive_base, old_additive)

    def test_no_penalty_batch_skips_accumulation_and_clears_snapshot(self):
        req = _make_req(outputs=[5], enabled=False)
        batch = _PenaltyBatch([req])
        batch.sampling_info.dflash_block_penalty_state = object()
        batch.prepare()
        self.assertIsNone(batch.sampling_info.dflash_block_penalty_state)
        self.assertEqual(req.penalty_cumulated_len, 0)
        _assert_state(batch, [[]])


if __name__ == "__main__":
    unittest.main()
