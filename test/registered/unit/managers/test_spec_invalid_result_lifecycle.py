"""CPU regressions for speculative invalid-row settlement and late results.

These tests use the actual request finish logic and decode result loop. Runtime
services (cache, allocator, output transport, and device copies) stay at the
boundary so token, grammar, cursor, and accounting state are observable.
"""

import unittest
from array import array
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from time import perf_counter
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_STR,
    FINISH_MATCHED_TOKEN,
    FINISHED_MATCHED_REGEX,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_PROCESSOR = "sglang.srt.managers.scheduler_components.batch_result_processor"


class _Tokenizer:
    eos_token_id = 99
    additional_stop_token_ids = set()

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


class _Grammar:
    def __init__(self, terminate_after=100, rollback_limit=None):
        self.accepted = []
        self.finished = False
        self.terminate_after = terminate_after
        self.rollback_limit = rollback_limit

    def accept_token(self, token_id):
        self.accepted.append(token_id)

    def rollback(self, count):
        if self.rollback_limit is not None and count > self.rollback_limit:
            raise ValueError("grammar rollback history exceeded")
        if count:
            del self.accepted[-count:]

    def is_terminated(self):
        return len(self.accepted) >= self.terminate_after

    def fork(self):
        return deepcopy(self)


class _Batch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.has_grammar = any(req.grammar is not None for req in reqs)
        self.forward_mode = SimpleNamespace(
            is_decode=lambda: True, is_extend=lambda: False
        )
        self.spec_algorithm = SimpleNamespace(
            is_none=lambda: False, is_dflash=lambda: False
        )
        self.return_logprob = any(req.return_logprob for req in reqs)
        self.mamba_track_mask_cpu = None

    def batch_size(self):
        return len(self.reqs)


class _CacheReceipts:
    """External cache boundary with visible receipt ownership and retirement."""

    def __init__(self):
        self.captured = []

    def capture_verification_attempt(self, req):
        receipt = SimpleNamespace(
            handle=req.cache_request_handle,
            retraction=req.retraction_count,
            valid=True,
            released=False,
        )
        self.captured.append(receipt)
        return receipt

    def invalidate_verification_attempt(self, receipt):
        receipt.valid = False

    def release_verification_attempt(self, receipt):
        if receipt.released:
            raise AssertionError("verification receipt released twice")
        receipt.released = True


def _request(
    rid="request",
    *,
    grammar=None,
    output=(),
    return_logprob=False,
    return_hidden_states=False,
    **sampling,
):
    params = SamplingParams(max_new_tokens=100, temperature=0, **sampling)
    tokenizer = _Tokenizer()
    params.normalize(tokenizer)
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", [1, 2, 3]),
        sampling_params=params,
        require_reasoning=True,
        return_logprob=return_logprob,
        return_hidden_states=return_hidden_states,
    )
    req.tokenizer = tokenizer
    req.grammar = grammar
    req.output_ids.extend(output)
    req.kv.kv_committed_len = 3 + len(output)
    # Set metadata after construction so pre-fix controls reach semantic
    # assertions instead of failing because a constructor keyword is unknown.
    req.cache_invalid = False
    return req


def _result(reqs, rows, invalid, *, accept_lens=None, stride=None):
    stride = stride or max((len(row) for row in rows), default=1)
    result = GenerationBatchResult(
        next_token_ids=torch.tensor(
            [token for row in rows for token in row + [0] * (stride - len(row))],
            dtype=torch.int64,
        ),
        accept_lens=torch.tensor(
            accept_lens if accept_lens is not None else [len(row) for row in rows],
            dtype=torch.int64,
        ),
        speculative_num_draft_tokens=stride,
        logits_output=LogitsProcessorOutput(next_token_logits=None),
    )
    result.first_invalid_rows = torch.tensor(invalid, dtype=torch.int32)
    result.spec_request_attempts = [
        (req.cache_request_handle, req.retraction_count) for req in reqs
    ]
    return result


class TestInvalidSpecResultLifecycle(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.observations = []
        self.streamed = []
        self.released = []
        metrics = Mock()
        metrics.num_generated_tokens = 0
        metrics.forward_ct_decode = 0
        self.processor = SchedulerBatchResultProcessor(
            is_generation=True,
            disaggregation_mode=None,
            enable_overlap=True,
            enable_overlap_mlx=False,
            model_config=SimpleNamespace(think_end_ids=[127]),
            token_to_kv_pool_allocator=Mock(),
            tree_cache=_CacheReceipts(),
            hisparse_coordinator=None,
            req_to_token_pool=None,
            decode_offload_manager=None,
            metrics_collector=Mock(),
            metrics_reporter=metrics,
            draft_worker=None,
            model_worker=SimpleNamespace(
                on_verify_complete_cpu=lambda lengths, batch_size: (
                    self.observations.append((list(lengths), batch_size))
                ),
            ),
            logprob_result_processor=None,
            output_streamer=SimpleNamespace(
                stream_output=lambda reqs, *_: self.streamed.append(
                    [list(req.output_ids_through_stop) for req in reqs]
                )
            ),
            beam_coordinator=None,
            abort_request=lambda *args, **kwargs: None,
        )
        patches = (
            patch(
                f"{_PROCESSOR}.get_exec",
                return_value=SimpleNamespace(
                    mamba=SimpleNamespace(enable_mamba_extra_buffer_lazy=False)
                ),
            ),
            patch(
                f"{_PROCESSOR}.get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_enable_offload_kvcache=False
                ),
            ),
            patch(
                f"{_PROCESSOR}.get_memory",
                return_value=SimpleNamespace(enable_hisparse=False),
            ),
            patch(
                f"{_PROCESSOR}.get_observability",
                return_value=SimpleNamespace(enable_metrics=False),
            ),
            patch(f"{_PROCESSOR}.get_global_indexer_capturer", return_value=None),
            patch(
                f"{_PROCESSOR}.release_kv_cache",
                side_effect=lambda req, *_args, **_kwargs: self.released.append(req),
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _decode(self, reqs, result):
        self.processor.process_batch_result_decode(_Batch(reqs), result)

    @staticmethod
    def _state(req):
        return (
            list(req.output_ids),
            req.finished_reason,
            req.finished_len,
            req.kv.kv_committed_len,
            req.spec_verify_ct,
            req.spec_num_correct_drafts,
            list(req.spec_correct_drafts_histogram),
            req.reasoning_tokens,
            req.send_token_offset,
            req.surr_offset,
            req.read_offset,
            list(req.grammar.accepted) if req.grammar is not None else None,
        )

    def test_reached_invalid_row_zero_aborts_without_output_or_accounting(self):
        req = _request(output=[70])
        result = _result([req], [[65, 66, 67]], [0])

        self._decode([req], result)

        self.assertEqual(list(req.output_ids), [70])
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertTrue(req.cache_invalid)
        self.assertEqual(req.kv.kv_committed_len, 4)
        self.assertEqual(req.spec_verify_ct, 0)
        self.assertEqual(req.spec_num_correct_drafts, 0)
        self.assertEqual(req.reasoning_tokens, 0)
        self.assertEqual(self.processor.metrics_reporter.num_generated_tokens, 0)
        self.assertFalse(any(lengths for lengths, _ in self.observations))
        self.assertEqual(self.streamed, [[[70]]])

    def test_nonterminal_valid_prefix_is_discarded_with_grammar_unchanged(self):
        grammar = _Grammar()
        grammar.accept_token(70)
        req = _request(output=[70], grammar=grammar)
        result = _result([req], [[65, 66, 67, 68]], [2])

        self._decode([req], result)

        self.assertEqual(list(req.output_ids), [70])
        self.assertEqual(grammar.accepted, [70])
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertTrue(req.cache_invalid)
        self.assertEqual(req.kv.kv_committed_len, 4)
        self.assertEqual(req.spec_verify_ct, 0)
        self.assertEqual(req.reasoning_tokens, 0)

    def test_discarded_prefix_longer_than_grammar_rollback_history_is_safe(self):
        grammar = _Grammar(terminate_after=1000, rollback_limit=200)
        grammar.accept_token(70)
        req = _request(output=[70], grammar=grammar)
        req.sampling_params.max_new_tokens = 1000
        result = _result([req], [[65] * 300], [250])

        self._decode([req], result)

        self.assertEqual(list(req.output_ids), [70])
        self.assertEqual(grammar.accepted, [70])
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.kv.kv_committed_len, 4)

    def test_terminal_conditions_strictly_before_invalid_keep_earliest_prefix(self):
        cases = (
            ("eos", {}, [65, 99, 66, 67], FINISH_MATCHED_TOKEN),
            (
                "stop_token",
                {"stop_token_ids": {66}},
                [65, 66, 67, 68],
                FINISH_MATCHED_TOKEN,
            ),
            ("stop_string", {"stop": "AB"}, [65, 66, 67, 68], FINISH_MATCHED_STR),
            (
                "stop_regex",
                {"stop_regex": "A[B-C]"},
                [65, 66, 67, 68],
                FINISHED_MATCHED_REGEX,
            ),
            ("max_length", {}, [65, 66, 67, 68], FINISH_LENGTH),
            ("grammar", {}, [65, 66, 67, 68], FINISH_MATCHED_TOKEN),
        )
        for name, params, tokens, finish_type in cases:
            with self.subTest(terminal=name):
                req = _request(
                    grammar=_Grammar(2) if name == "grammar" else None, **params
                )
                if name == "max_length":
                    req.sampling_params.max_new_tokens = 2
                result = _result([req], [tokens], [3])

                self._decode([req], result)

                self.assertEqual(list(req.output_ids), tokens[:2])
                self.assertEqual(list(req.output_ids_through_stop), tokens[:2])
                self.assertIsInstance(req.finished_reason, finish_type)
                self.assertTrue(req.cache_invalid)
                self.assertEqual(req.kv.kv_committed_len, 5)
                self.assertEqual(req.reasoning_tokens, 2)
                self.assertEqual(result.get_num_generated_tokens(1), 2)
                if req.grammar is not None:
                    self.assertEqual(req.grammar.accepted, tokens[:2])

    def test_terminal_token_at_invalid_boundary_does_not_finish_naturally(self):
        req = _request(stop_token_ids={66})
        result = _result([req], [[65, 66, 67]], [1])

        self._decode([req], result)

        self.assertEqual(list(req.output_ids), [])
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.kv.kv_committed_len, 3)

    def test_earliest_terminal_beats_later_string_and_length(self):
        req = _request(stop="BC", stop_token_ids={65})
        req.sampling_params.max_new_tokens = 3
        result = _result([req], [[65, 66, 67, 68]], [3])

        self._decode([req], result)

        self.assertEqual(list(req.output_ids), [65])
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_reason.matched, 65)
        self.assertEqual(req.reasoning_tokens, 1)

    def test_stop_string_can_span_previous_output_and_valid_prefix(self):
        req = _request(output=[65], stop="AB")
        result = _result([req], [[66, 67, 68]], [2])

        self._decode([req], result)

        self.assertEqual(list(req.output_ids), [65, 66])
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_STR)
        self.assertEqual(req.finished_len, 2)
        self.assertEqual(result.get_num_generated_tokens(1), 1)

    def test_terminal_prefix_bounds_logprobs_hidden_states_and_auxiliary_output(self):
        req = _request(
            stop_token_ids={66}, return_logprob=True, return_hidden_states=True
        )
        result = _result([req], [[65, 66, 67, 68]], [3])
        result.logits_output.next_token_logprobs = torch.tensor(
            [[-1.0, -2.0, -3.0, -4.0]]
        )
        result.logits_output.hidden_states = torch.arange(8).view(4, 2)
        commits = []
        result.auxiliary_host_output = SimpleNamespace(
            consume=lambda _batch, records: commits.extend(records)
        )

        self._decode([req], result)

        self.assertEqual(list(req.output_ids), [65, 66])
        self.assertEqual(req.logprob.output_token_logprobs_idx, [65, 66])
        self.assertEqual(req.logprob.output_token_logprobs_val, [-1.0, -2.0])
        self.assertEqual(req.hidden_states, [[0, 1], [2, 3]])
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0].token_ids, (65, 66))

    def test_grammar_barrier_defers_invalid_row_until_terminal_settlement(self):
        dirty = _request("dirty", grammar=_Grammar(2))
        healthy = _request("healthy", grammar=_Grammar())
        result = _result([dirty, healthy], [[65, 66, 67], [70, 71, 72]], [2, -1])
        batch = _Batch([dirty, healthy])

        self.processor.advance_grammar_fsm(result, batch)

        self.assertEqual(dirty.grammar.accepted, [])
        self.assertEqual(healthy.grammar.accepted, [70, 71, 72])
        self.processor.process_batch_result_decode(batch, result)
        self.assertEqual(dirty.grammar.accepted, [65, 66])
        self.assertEqual(healthy.grammar.accepted, [70, 71, 72])
        self.assertEqual(list(dirty.output_ids), [65, 66])
        self.assertEqual(list(healthy.output_ids), [70, 71, 72])

    def test_finished_late_invalid_result_retires_only_saved_cache_receipt(self):
        req = _request(output=[65], grammar=_Grammar())
        req.finished_reason = FINISH_LENGTH(length=1)
        req.finished_len = 1
        req.send_token_offset = 1
        req.surr_offset = 2
        req.read_offset = 3
        req.spec_verify_ct = 5
        req.spec_num_correct_drafts = 8
        before = self._state(req)
        result = _result([req], [[66, 67, 68]], [0])
        receipt = self.processor.tree_cache.capture_verification_attempt(req)
        result.cache_verification_attempts = [receipt]

        self._decode([req], result)

        self.assertEqual(self._state(req), before)
        self.assertFalse(req.cache_invalid)
        self.assertFalse(receipt.valid)
        self.assertTrue(receipt.released)
        self.assertEqual(result.get_num_generated_tokens(1), 0)
        self.assertFalse(any(lengths for lengths, _ in self.observations))
        self.assertEqual(self.released, [])

    def test_finished_nonoverlap_result_does_not_mutate_request(self):
        self.processor = replace(self.processor, enable_overlap=False)
        req = _request(output=[65], grammar=_Grammar())
        req.finished_reason = FINISH_LENGTH(length=1)
        req.finished_len = 1
        before = self._state(req)

        self._decode([req], _result([req], [[66, 67, 68]], [0]))

        self.assertEqual(self._state(req), before)
        self.assertEqual(self.processor.metrics_reporter.num_generated_tokens, 0)

    def test_retracted_readmission_ignores_old_generation_snapshot(self):
        req = _request(grammar=_Grammar())
        result = _result([req], [[65, 66, 67]], [0])
        req.reset_for_retract()
        req.is_retracted = False
        req.kv.kv_committed_len = 3
        before = self._state(req)

        self._decode([req], result)

        self.assertEqual(self._state(req), before)
        self.assertFalse(req.cache_invalid)
        self.assertEqual(result.get_num_generated_tokens(1), 0)

    def test_same_rid_new_cache_attempt_ignores_old_snapshot(self):
        req = _request("same-rid", grammar=_Grammar())
        result = _result([req], [[65, 66, 67]], [0])
        req.cache_request_handle = CacheRequestHandle(req.rid, 1)
        before = self._state(req)

        self._decode([req], result)

        self.assertEqual(self._state(req), before)
        self.assertFalse(req.cache_invalid)
        self.assertEqual(result.get_num_generated_tokens(1), 0)

    def test_mixed_batch_counts_only_committed_contributing_rows(self):
        healthy = _request("healthy")
        terminal = _request("terminal", stop_token_ids={71})
        discarded = _request("discarded")
        finished = _request("finished", output=[80])
        finished.finished_reason = FINISH_LENGTH(length=1)
        finished.finished_len = 1
        stale = _request("stale")
        reqs = [healthy, terminal, discarded, finished, stale]
        result = _result(
            reqs,
            [[65, 66, 67], [70, 71, 72], [73, 74, 75], [81, 82, 83], [84, 85, 86]],
            [-1, 2, 1, 0, 0],
        )
        result.block_accept_lens = torch.tensor([3, 3, 3, 3, 3])
        result.cap_lens = torch.tensor([2, 3, 3, 3, 3])
        stale.retraction_count += 1

        self._decode(reqs, result)

        self.assertEqual(
            [list(req.output_ids) for req in reqs],
            [[65, 66, 67], [70, 71], [], [80], []],
        )
        self.assertEqual([req.spec_verify_ct for req in reqs], [1, 1, 0, 0, 0])
        self.assertEqual([req.spec_num_correct_drafts for req in reqs], [2, 1, 0, 0, 0])
        self.assertEqual(result.num_correct_drafts, 3)
        self.assertEqual(result.get_num_generated_tokens(5), 5)
        self.assertEqual(self.processor.metrics_reporter.num_generated_tokens, 5)
        self.assertEqual(self.observations, [([2, 1], 2)])
        self.assertEqual(result.num_spec_verify_rows, 2)
        self.assertEqual(result.num_block_accept_tokens, 5)
        self.assertEqual(result.num_cap_tokens, 4)
        self.assertEqual(
            [req.spec_num_block_accept_tokens for req in reqs], [3, 2, 0, 0, 0]
        )
        self.assertEqual([req.spec_num_cap_tokens for req in reqs], [2, 2, 0, 0, 0])

    def test_empty_accepted_row_has_no_negative_drafts_or_bonus_credit(self):
        req = _request()
        result = _result([req], [[]], [-1], stride=4)

        self._decode([req], result)

        self.assertEqual(list(req.output_ids), [])
        self.assertFalse(req.finished())
        self.assertEqual(req.kv.kv_committed_len, 3)
        self.assertEqual(req.spec_verify_ct, 0)
        self.assertEqual(req.spec_num_correct_drafts, 0)
        self.assertEqual(result.num_correct_drafts, 0)
        self.assertEqual(result.get_num_generated_tokens(1), 0)
        self.assertFalse(any(lengths for lengths, _ in self.observations))

    def test_metadata_copy_precedes_event_and_consumption_waits_for_event(self):
        req = _request()
        result = _result([req], [[65, 66]], [0])
        original = result.first_invalid_rows
        copied = []
        event = Mock()

        def copy(tensor):
            host = tensor.clone()
            copied.append((tensor, host))
            return host

        def recorded():
            self.assertTrue(any(source is original for source, _ in copied))
            self.assertIsNot(result.first_invalid_rows, original)

        event.record.side_effect = recorded
        result.copy_done = event
        with patch("sglang.srt.managers.utils._async_d2h", side_effect=copy):
            result.copy_to_cpu(return_logprob=False)

        # The asynchronous device-to-host copy becomes visible only on wait.
        result.first_invalid_rows.fill_(-1)
        event.synchronize.side_effect = lambda: result.first_invalid_rows.fill_(0)
        self._decode([req], result)
        self.assertEqual(list(req.output_ids), [])
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertTrue(req.cache_invalid)

    def test_scheduler_snapshots_before_worker_can_finish_previous_result(self):
        # Import the real scheduler without constructing its runtime services.
        from sglang.srt.managers.scheduler import Scheduler

        for previous_result in ("finish", "retract"):
            with self.subTest(previous_result=previous_result):
                req = _request(grammar=_Grammar())
                original_attempt = (req.cache_request_handle, req.retraction_count)
                result = _result([req], [[65, 66, 67]], [0])
                batch = _Batch([req])
                batch.spec_algorithm.is_dflash = lambda: True
                cache = self.processor.tree_cache

                def forward(_batch):
                    if previous_result == "finish":
                        req.finished_reason = FINISH_LENGTH(length=0)
                        req.finished_len = 0
                    else:
                        req.reset_for_retract()
                        req.is_retracted = False
                        req.kv.kv_committed_len = 3
                    return result

                scheduler = SimpleNamespace(
                    tree_cache=cache,
                    model_worker=SimpleNamespace(forward_batch_generation=forward),
                )
                returned = Scheduler._forward_generation_with_validation(
                    scheduler, batch
                )
                self.assertIs(returned, result)
                self.assertEqual(returned.spec_request_attempts, [original_attempt])
                receipt = returned.cache_verification_attempts[0]
                self.assertEqual((receipt.handle, receipt.retraction), original_attempt)
                self.assertFalse(receipt.released)
                before = self._state(req)

                self._decode([req], returned)

                self.assertEqual(self._state(req), before)
                self.assertFalse(req.cache_invalid)
                self.assertFalse(receipt.valid)
                self.assertTrue(receipt.released)

    def test_scheduler_forward_exception_releases_captured_receipts(self):
        from sglang.srt.managers.scheduler import Scheduler

        reqs = [_request("first"), _request("second")]
        batch = _Batch(reqs)
        batch.spec_algorithm.is_dflash = lambda: True
        cache = self.processor.tree_cache

        def forward(_batch):
            reqs[0].reset_for_retract()
            raise RuntimeError("injected worker failure")

        scheduler = SimpleNamespace(
            tree_cache=cache,
            model_worker=SimpleNamespace(forward_batch_generation=forward),
        )
        with self.assertRaisesRegex(RuntimeError, "injected worker failure"):
            Scheduler._forward_generation_with_validation(scheduler, batch)

        self.assertEqual(len(cache.captured), 2)
        self.assertTrue(all(receipt.released for receipt in cache.captured))
        self.assertEqual([receipt.retraction for receipt in cache.captured], [0, 0])

    def test_scheduler_capture_failure_releases_earlier_captures(self):
        from sglang.srt.managers.scheduler import Scheduler

        reqs = [_request("first"), _request("second")]
        batch = _Batch(reqs)
        batch.spec_algorithm.is_dflash = lambda: True
        cache = self.processor.tree_cache
        original_capture = cache.capture_verification_attempt

        def capture(req):
            if req is reqs[1]:
                raise RuntimeError("injected capture failure")
            return original_capture(req)

        cache.capture_verification_attempt = capture
        scheduler = SimpleNamespace(tree_cache=cache, model_worker=Mock())

        with self.assertRaisesRegex(RuntimeError, "injected capture failure"):
            Scheduler._forward_generation_with_validation(scheduler, batch)

        self.assertEqual(len(cache.captured), 1)
        self.assertTrue(cache.captured[0].released)

    def _scheduler_with_runtime_boundaries(self, req, result, *, overlap):
        from sglang.srt.managers.scheduler import Scheduler

        algorithm = SimpleNamespace(
            is_none=lambda: False,
            is_dflash=lambda: True,
            is_ngram=lambda: False,
            supports_grammar_overlap=lambda: False,
        )
        batch = ScheduleBatch(
            reqs=[req],
            forward_mode=ForwardMode.DECODE,
            spec_algorithm=algorithm,
            enable_overlap=overlap,
            input_ids=torch.tensor([1]),
            req_pool_indices=torch.tensor([0]),
        )
        scheduler = SimpleNamespace(
            scheduler_stage_metrics=None,
            metrics_reporter=Mock(),
            forward_ct=0,
            _sched_idled=False,
            scripted_scheduler_hook=None,
            profiler_manager=Mock(),
            forward_sleep_time=None,
            disaggregation_mode=None,
            is_generation=True,
            enable_overlap=overlap,
            enable_pdmux=False,
            enable_unified_memory=False,
            _confidence_budget_prepare=None,
            future_map=Mock(),
            forward_stream_ctx=nullcontext(),
            forward_stream=Mock(),
            schedule_stream=Mock(),
            batch_record_buf=[None, None],
            batch_record_ct=0,
            tree_cache=self.processor.tree_cache,
            model_worker=SimpleNamespace(
                forward_batch_generation=lambda *a, **k: result
            ),
            device_module=SimpleNamespace(Event=Mock),
            ps=SimpleNamespace(pp_size=1),
            spec_algorithm=algorithm,
            publish_load_snapshot=Mock(
                side_effect=RuntimeError("injected publication failure")
            ),
        )
        for name in (
            "_run_batch",
            "_forward_isolation",
            "record_batch_in_overlap",
            "_forward_generation_with_validation",
            "_relay_forward_payload",
            "update_cache_from_scheduler",
            "_process_batch_result",
        ):
            setattr(scheduler, name, MethodType(getattr(Scheduler, name), scheduler))
        return scheduler, batch

    def test_scheduler_d2h_failure_releases_forward_receipts(self):
        from sglang.srt.managers.scheduler import Scheduler

        req = _request()
        result = _result([req], [[65, 66]], [0])
        scheduler, batch = self._scheduler_with_runtime_boundaries(
            req, result, overlap=False
        )
        with (
            patch(
                "sglang.srt.managers.utils._async_d2h",
                side_effect=RuntimeError("injected D2H failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "injected D2H failure"),
        ):
            Scheduler.run_batch(scheduler, batch)

        self.assertEqual(len(scheduler.tree_cache.captured), 1)
        self.assertTrue(scheduler.tree_cache.captured[0].released)
        self.assertIsNone(result.cache_verification_attempts)
        self.assertEqual(list(req.output_ids), [])

    def test_scheduler_relay_failure_releases_forward_receipts(self):
        from sglang.srt.managers.scheduler import Scheduler

        req = _request()
        result = _result([req], [[65, 66]], [0])
        scheduler, batch = self._scheduler_with_runtime_boundaries(
            req, result, overlap=True
        )
        scheduler.future_map.stash.side_effect = RuntimeError("injected relay failure")
        with self.assertRaisesRegex(RuntimeError, "injected relay failure"):
            Scheduler.run_batch(scheduler, batch)

        self.assertEqual(len(scheduler.tree_cache.captured), 1)
        self.assertTrue(scheduler.tree_cache.captured[0].released)
        self.assertIsNone(result.cache_verification_attempts)
        self.assertEqual(list(req.output_ids), [])

    def test_predispatch_publication_failure_releases_receipt_once(self):
        from sglang.srt.managers.scheduler import Scheduler

        req = _request()
        result = _result([req], [[65, 66]], [0])
        scheduler, batch = self._scheduler_with_runtime_boundaries(
            req, result, overlap=False
        )
        receipt = scheduler.tree_cache.capture_verification_attempt(req)
        result.cache_verification_attempts = [receipt]

        with self.assertRaisesRegex(RuntimeError, "injected publication failure"):
            Scheduler.process_batch_result(scheduler, batch, result)

        self.assertTrue(receipt.released)
        self.assertIsNone(result.cache_verification_attempts)
        self.assertEqual(list(req.output_ids), [])
        result.release_verification_attempts(scheduler.tree_cache)
        self.assertTrue(receipt.released)


class TestInvalidSpecMetricsWindow(CustomTestCase):
    def test_report_decode_stats_accepts_window_with_no_contributing_rows(self):
        # Exercise the production reporting calculation and logging path without
        # starting collectors, device timers, or the scheduler runtime.
        reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
        reporter.scheduler = SimpleNamespace(
            pool_stats_observer=SimpleNamespace(
                get_pool_stats=lambda: SimpleNamespace(
                    get_decode_usage_msg_parts=lambda: []
                )
            ),
            spec_algorithm=SimpleNamespace(
                is_none=lambda: False, is_dspark=lambda: False
            ),
            disaggregation_mode=None,
            waiting_queue=[],
            forward_ct=1,
            kv_events_publisher=Mock(),
        )
        reporter.reset_metrics()
        reporter.current_scheduler_metrics_enabled = False
        reporter.is_stats_logging_rank = True
        reporter.enable_mfu_metrics = False
        reporter.decode_log_interval = 1
        reporter.last_decode_stats_tic = perf_counter() - 1
        reporter._graph_backend_label = "cpu graph"
        reporter.fwd_occupancy = 0.0
        reporter.step_time_dict = defaultdict(list)
        reporter.update_spec_metrics(0, 0, 0)
        batch = SimpleNamespace(reqs=[], forward_iter=1)
        path = "sglang.srt.managers.scheduler_components.metrics_reporter"

        with (
            patch(f"{path}.logger") as logger,
            patch(
                f"{path}.get_disagg", return_value=SimpleNamespace(language_only=False)
            ),
            patch(
                f"{path}.get_spec",
                return_value=SimpleNamespace(
                    speculative_num_draft_tokens=4, speculative_num_steps=3
                ),
            ),
        ):
            reporter.report_decode_stats(
                False, running_batch=batch, num_generated_tokens=0
            )

        self.assertEqual(reporter.last_gen_throughput, 0)
        self.assertEqual(reporter.spec_total_num_forward_ct, 0)
        self.assertEqual(reporter.spec_total_num_accept_tokens, 0)
        self.assertEqual(logger.info.call_count, 1)
        self.assertIn(
            "accept len: 0.00, accept rate: 0.00", logger.info.call_args.args[0]
        )


if __name__ == "__main__":
    unittest.main()
