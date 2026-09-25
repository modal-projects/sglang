"""Pending output penalties survive overlap, row changes, and buffer reuse."""

import unittest
from array import array
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers import overlap_utils
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling import sampling_batch_info
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative import spec_utils
from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

VOCAB_SIZE = 16


def _req(outputs=(), *, prompt=(10, 11), frequency=0.25, presence=-0.5, enabled=True):
    req = Req(
        rid="reusable-rid",
        origin_input_text="",
        origin_input_ids=array("q", prompt),
        sampling_params=SamplingParams(
            temperature=0,
            top_k=1,
            frequency_penalty=frequency if enabled else 0,
            presence_penalty=presence if enabled else 0,
            repetition_penalty=1.5 if enabled else 1,
            min_new_tokens=4 if enabled else 0,
        ),
        vocab_size=VOCAB_SIZE,
    )
    req.tokenizer = SimpleNamespace(eos_token_id=2, additional_stop_token_ids=None)
    req.output_ids.extend(outputs)
    req.kv.kv_committed_len = len(prompt) + max(len(outputs) - 1, 0)
    return req


def _sampling_info(batch):
    # Runtime configuration is an external construction dependency.
    runtime = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
        features=SimpleNamespace(enable_custom_logit_processor=False),
    )
    with patch.object(sampling_batch_info, "get_exec", return_value=runtime):
        return SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE)


def _batch(reqs, slots, *, algorithm=SpeculativeAlgorithm.DFLASH):
    batch = ScheduleBatch(
        reqs=reqs,
        device="cpu",
        enable_overlap=True,
        spec_algorithm=algorithm,
        req_pool_indices=torch.tensor(slots),
        req_pool_indices_cpu=torch.tensor(slots),
        forward_mode=ForwardMode.DECODE,
    )
    batch.sampling_info = _sampling_info(batch)
    return batch


def _reference(req, history, logits):
    result = logits.clone()
    params = req.sampling_params
    for token, count in Counter(history).items():
        result[token] -= count * params.frequency_penalty + params.presence_penalty
    if len(history) < params.min_new_tokens:
        result[2] = -torch.inf
    for token in set(history):
        result[token] = (
            result[token] * params.repetition_penalty
            if result[token] < 0
            else result[token] / params.repetition_penalty
        )
    return result


class _SchedulerBoundary:
    _forward_isolation = Scheduler._forward_isolation
    _relay_forward_payload = Scheduler._relay_forward_payload
    record_batch_in_overlap = Scheduler.record_batch_in_overlap

    def __init__(self, algorithm=SpeculativeAlgorithm.DFLASH):
        self.spec_algorithm = algorithm
        # Replace only device-stream allocation; all relay methods are real.
        with patch.object(overlap_utils, "_is_cuda", False):
            self.future_map = FutureMap(
                device=torch.device("cpu"),
                spec_algo=algorithm,
                req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(8, 64)),
            )
        self.batch_record_ct = 0
        self.batch_record_buf = [None, None]
        self.chunked_req = None
        self.beam_coordinator = SimpleNamespace(
            maybe_select_and_relay=lambda *a, **k: None
        )

    def publish(self, batch, tokens, lengths, ends):
        draft_input = None
        if lengths is not None:
            draft_input = SimpleNamespace(
                bonus_tokens=tokens.gather(
                    1, (lengths - 1).clamp_min(0)[:, None]
                ).flatten(),
                topk_p=None,
                topk_index=None,
                hidden_states=None,
                dsa_topk_indices=None,
            )
        result = GenerationBatchResult(
            next_token_ids=tokens.reshape(-1),
            accept_lens=lengths,
            new_seq_lens=torch.tensor(ends),
            next_draft_input=draft_input,
        )
        with (
            self._forward_isolation(batch, overlap=True),
            patch.object(
                spec_utils,
                "get_spec",
                return_value=SimpleNamespace(
                    speculative_algorithm=self.spec_algorithm.name
                ),
            ),
        ):
            self._relay_forward_payload(batch, batch.req_pool_indices, result)
        return result

    def snapshot(self, batch, *, overlap=True):
        with self._forward_isolation(batch, overlap=overlap):
            return batch.sampling_info


class TestDFlashPendingPenalties(CustomTestCase):
    def assert_consumers(self, batch, info, histories):
        raw = torch.linspace(-2, 3, VOCAB_SIZE).repeat(len(batch.reqs), 1)
        expected = torch.stack(
            [
                _reference(req, hist, row)
                for req, hist, row in zip(batch.reqs, histories, raw)
            ]
        )
        ordinary = raw.clone()
        info.apply_logits_bias(ordinary)
        torch.testing.assert_close(ordinary, expected)

        candidates = torch.tensor([[9, 0, 7]]).repeat(len(batch.reqs), 1)
        logits = raw[:, None, :].expand(-1, 3, -1).reshape(-1, VOCAB_SIZE).clone()
        verify_expected = torch.stack(
            [
                _reference(req, hist + candidates[row, 1 : pos + 1].tolist(), raw[row])
                for row, (req, hist) in enumerate(zip(batch.reqs, histories))
                for pos in range(3)
            ]
        )
        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits,
            sampling_info=info,
            draft_token_num=3,
            candidates=candidates,
        )
        torch.testing.assert_close(logits, verify_expected)

    def test_multitoken_pending_then_cpu_settlement_counts_once(self):
        """The next launch sees every previous output before CPU settlement."""
        for algorithm in (SpeculativeAlgorithm.DFLASH, SpeculativeAlgorithm.DSPARK):
            with self.subTest(algorithm=algorithm):
                scheduler = _SchedulerBoundary(algorithm)
                req = _req([5])
                batch = _batch([req], [3], algorithm=algorithm)
                scheduler.publish(
                    batch, torch.tensor([[7, 8, 9]]), torch.tensor([3]), [5]
                )
                pending = scheduler.snapshot(batch)
                self.assert_consumers(batch, pending, [[5, 7, 8, 9]])
                self.assertEqual(req.penalty_cumulated_len, 1)
                # CPU result processing follows the next launch.
                req.output_ids.extend([7, 8, 9])
                req.kv.kv_committed_len = 5
                settled = scheduler.snapshot(batch)
                self.assert_consumers(batch, settled, [[5, 7, 8, 9]])
                self.assert_consumers(batch, pending, [[5, 7, 8, 9]])
                self.assertEqual(req.penalty_cumulated_len, 4)
                self.assertTrue(
                    any(
                        pending is value
                        for record in scheduler.batch_record_buf
                        for value in record[1]
                    )
                )

    def test_mixed_extend_ordinary_sampler_uses_pending_history(self):
        """Fresh prefill sampling metadata must retain a decode tail's history."""
        scheduler = _SchedulerBoundary()
        running = _batch([_req([5])], [3])
        scheduler.publish(running, torch.tensor([[7]]), None, [3])
        fresh = _batch([_req()], [1])
        fresh.sampling_info.merge_batch(running.sampling_info)
        fresh.reqs = fresh.reqs + running.reqs
        fresh.req_pool_indices = torch.tensor([1, 3])
        fresh.forward_mode = ForwardMode.EXTEND
        fresh.is_extend_in_batch = True
        fresh.seq_lens = torch.tensor([2, 4])
        fresh.extend_lens = [2, 1]
        fresh.prefix_lens = [0, 3]
        fresh.out_cache_loc = torch.tensor([0, 1, 2])
        raw = torch.linspace(-2, 3, VOCAB_SIZE).repeat(2, 1)
        observed = []

        def target_forward(batch, **kwargs):
            logits = raw.clone()
            batch.sampling_info.apply_logits_bias(logits)
            observed.append(logits)
            return GenerationBatchResult(
                logits_output=SimpleNamespace(hidden_states=torch.ones(2, 2)),
                next_token_ids=torch.tensor([4, 9]),
            )

        worker = SimpleNamespace(
            _validate_phase1_sampling_support=lambda batch: None,
            target_worker=SimpleNamespace(forward_batch_generation=target_forward),
            _tp_sync=SimpleNamespace(sync=lambda *args: None),
            model_runner=SimpleNamespace(prefill_attention_backend_str="torch_native"),
            _append_target_hidden_to_draft_kv_by_loc=lambda **kwargs: None,
            _make_next_draft_input_prefill=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        with scheduler._forward_isolation(fresh, overlap=True):
            DFlashWorkerV2.forward_batch_generation(worker, fresh)
            self.assert_consumers(fresh, fresh.sampling_info, [[], [5, 7]])
        torch.testing.assert_close(
            observed[0][1], _reference(running.reqs[0], [5, 7], raw[1])
        )
        torch.testing.assert_close(
            observed[0][0], _reference(fresh.reqs[0], [], raw[0])
        )

    def test_filter_reorder_merge_and_partial_settlement(self):
        """Final row order and each captured cursor select the right pending suffix."""
        scheduler = _SchedulerBoundary()
        a, b, dropped = _req([5]), _req([4], frequency=-0.25), _req([1])
        batch = _batch([a, b, dropped], [3, 5, 6])
        scheduler.publish(
            batch,
            torch.tensor([[7, 7, 0], [0, 8, -1], [9, -1, -1]]),
            torch.tensor([3, 2, 1]),
            [5, 4, 3],
        )
        a.output_ids.append(7)
        batch.sampling_info.filter_batch([1, 0], torch.tensor([1, 0]))
        batch.reqs = [b, a]
        batch.req_pool_indices = torch.tensor([5, 3])
        fresh = _batch([_req()], [1])
        fresh.sampling_info.merge_batch(batch.sampling_info)
        fresh.reqs = fresh.reqs + batch.reqs
        fresh.req_pool_indices = torch.tensor([1, 5, 3])
        self.assert_consumers(
            fresh, scheduler.snapshot(fresh), [[], [4, 0, 8], [5, 7, 7, 0]]
        )

    def test_chunk_prefill_discards_samples_and_rebuilds_retained_outputs(self):
        """A fresh chunk accumulator rebuilds history and ignores prompt-only samples."""
        scheduler = _SchedulerBoundary()
        req = _req([5, 0], prompt=(10, 11, 12, 13))
        req.reset_for_retract()
        req.is_retracted = False
        batch = _batch([req], [3])
        batch.forward_mode = ForwardMode.EXTEND
        scheduler.publish(batch, torch.tensor([[7]]), None, [3])
        self.assert_consumers(batch, scheduler.snapshot(batch), [[5, 0]])
        batch.sampling_info = _sampling_info(batch)
        self.assert_consumers(batch, scheduler.snapshot(batch), [[5, 0]])
        scheduler.publish(batch, torch.tensor([[9]]), None, [6])
        self.assert_consumers(batch, scheduler.snapshot(batch), [[5, 0, 9]])

    def test_retract_rewind_and_reused_slot_reject_stale_owner(self):
        """Retraction, rewind, and a new request cannot inherit an older run."""
        scheduler = _SchedulerBoundary()
        req = _req([5, 7])
        batch = _batch([req], [3])
        scheduler.publish(batch, torch.tensor([[8, 9]]), torch.tensor([2]), [5])
        generation = req.penalty_generation
        req.reset_for_retract()
        batch.sampling_info = _sampling_info(batch)
        self.assertNotEqual(req.penalty_generation, generation)
        self.assert_consumers(batch, scheduler.snapshot(batch), [[5, 7]])
        scheduler.publish(batch, torch.tensor([[8, 9]]), torch.tensor([2]), [5])
        generation = req.penalty_generation
        req.output_ids = req.output_ids[:1]
        self.assert_consumers(batch, scheduler.snapshot(batch), [[5]])
        self.assertNotEqual(req.penalty_generation, generation)
        replacement = _batch([_req()], [3])
        self.assert_consumers(replacement, scheduler.snapshot(replacement), [[]])

    def test_graph_output_and_relay_overwrite_do_not_mutate_snapshot(self):
        """Replayed static outputs and a later publication cannot rewrite a launch."""
        scheduler = _SchedulerBoundary()
        batch = _batch([_req([5], frequency=-0.5)], [3])
        tokens, lengths = torch.tensor([[0, 0, 7]]), torch.tensor([3])
        scheduler.publish(batch, tokens, lengths, [5])
        tokens.fill_(12)
        lengths.zero_()
        saved = scheduler.snapshot(batch)
        self.assert_consumers(batch, saved, [[5, 0, 0, 7]])
        for value in (8, 9, 10):
            scheduler.publish(batch, torch.tensor([[value]]), None, [3])
        self.assert_consumers(batch, saved, [[5, 0, 0, 7]])

    def test_nonoverlap_ignores_unsettled_relay(self):
        """Synchronous forwards derive penalties solely from settled outputs."""
        scheduler = _SchedulerBoundary()
        batch = _batch([_req([5, 7, 7, 0])], [3])
        scheduler.publish(batch, torch.tensor([[9]]), None, [6])
        self.assert_consumers(
            batch, scheduler.snapshot(batch, overlap=False), [[5, 7, 7, 0]]
        )

    def test_missing_sampling_metadata_preserves_output_relay(self):
        """Optional sampling metadata must not bypass the normal output relay."""
        scheduler = _SchedulerBoundary()
        batch = _batch([_req()], [3])
        batch.sampling_info = None
        scheduler.publish(batch, torch.tensor([[7]]), None, [3])
        torch.testing.assert_close(
            scheduler.future_map.output_tokens_buf[batch.req_pool_indices],
            torch.tensor([7]),
        )
        self.assertIsNone(scheduler.future_map.penalty_output_relay.tokens)

    def test_no_penalties_leave_output_relay_unallocated(self):
        """Ordinary requests do not pay for an unused pending-output matrix."""
        scheduler = _SchedulerBoundary()
        batch = _batch([_req([5], enabled=False)], [3])
        scheduler.publish(batch, torch.tensor([[7]]), None, [3])
        info = scheduler.snapshot(batch)
        self.assertIsNone(info.dflash_block_penalty_state)
        self.assertIsNone(scheduler.future_map.penalty_output_relay.tokens)
        self.assertIsNone(scheduler.future_map.penalty_output_relay.num_valid)
        self.assert_consumers(batch, info, [[5, 7]])


if __name__ == "__main__":
    unittest.main()
