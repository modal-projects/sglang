"""Generated-token timing follows host output commitment across result paths.

This fixture runs real Req lifecycle and result processing with CPU result
buffers. Cache allocation, model execution, and socket transport are boundaries;
no timing setter or output-commit method is replaced.
"""

import asyncio
import unittest
from array import array
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.beam_search.beam_group import BeamGroup
from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.beam_search.logits_capture import BeamLogitsCapture
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput, SamplingMaskStatus
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sglang.srt.managers.utils import EmbeddingBatchResult, GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    SchedulerReqTimeStats,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_PROCESSOR = "sglang.srt.managers.scheduler_components.batch_result_processor"
_CLOCK = "sglang.srt.observability.req_time_stats.time.perf_counter"


class TestGeneratedTokenTiming(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            enable_metrics=False, optimistic_prefill_attempts=1
        )
        override.install()
        self.addCleanup(override.restore)
        # The fixture has no allocated KV; only these external cache effects
        # need replacing. The request and all token/finish/timing logic are real.
        for target in (
            f"{_PROCESSOR}.maybe_cache_unfinished_req",
            f"{_PROCESSOR}.release_kv_cache",
            "sglang.srt.dllm.mixin.scheduler.release_kv_cache",
            "sglang.srt.disaggregation.prefill.release_kv_cache",
            "sglang.srt.disaggregation.prefill.maybe_cache_unfinished_req",
        ):
            replacement = patch(target)
            replacement.start()
            self.addCleanup(replacement.stop)
        reporter = Mock()
        reporter.num_generated_tokens = 0
        reporter.forward_ct_decode = 0
        self.processor = SchedulerBatchResultProcessor(
            is_generation=True,
            disaggregation_mode=None,
            enable_overlap=True,
            enable_overlap_mlx=False,
            model_config=SimpleNamespace(think_end_ids=None),
            token_to_kv_pool_allocator=Mock(),
            tree_cache=None,
            hisparse_coordinator=None,
            req_to_token_pool=None,
            decode_offload_manager=None,
            metrics_collector=None,
            metrics_reporter=reporter,
            draft_worker=None,
            model_worker=Mock(),
            logprob_result_processor=None,
            output_streamer=Mock(),
            beam_coordinator=SimpleNamespace(commit_decode=lambda batch: set()),
            abort_request=lambda *args, **kwargs: None,
        )

    def req(self, *, max_new_tokens=20):
        sampling = SamplingParams(max_new_tokens=max_new_tokens, temperature=0)
        sampling.normalize(None)
        req = Req(
            rid="timing",
            origin_input_text="",
            origin_input_ids=array("q", [1, 2]),
            sampling_params=sampling,
            metrics_collector=Mock(),
        )
        self.assertIsInstance(req.time_stats, SchedulerReqTimeStats)
        req.time_stats.enable_metrics = True
        return req

    def batch(self, req, *, spec=False):
        return SimpleNamespace(
            reqs=[req],
            decoding_reqs=[],
            return_logprob=False,
            return_hidden_states=False,
            return_hidden_states_mode=CaptureHiddenMode.NULL,
            spec_info=None,
            prefill_stats=None,
            dp_cooperation_info=None,
            mamba_track_mask_cpu=None,
            has_grammar=req.grammar is not None,
            forward_mode=ForwardMode.DECODE,
            spec_algorithm=SpeculativeAlgorithm.EAGLE
            if spec
            else SpeculativeAlgorithm.NONE,
            batch_size=lambda: 1,
        )

    def prefill(self, req, token=7, *, now=100.0, spec=False, logits=None):
        result = GenerationBatchResult(
            logits_output=logits,
            next_token_ids=torch.tensor([token]),
        )
        with patch(_CLOCK, return_value=now):
            self.processor.process_batch_result_prefill(
                self.batch(req, spec=spec), result
            )

    def decode(self, req, tokens=(8,), *, now=200.0, spec=False, logits=None):
        result = GenerationBatchResult(
            logits_output=logits,
            next_token_ids=torch.tensor(tokens, dtype=torch.int64),
            accept_lens=torch.tensor([len(tokens)]) if spec else None,
            speculative_num_draft_tokens=max(len(tokens), 1) if spec else None,
            num_non_draft_tokens_per_req=0 if spec and not tokens else 1,
        )
        with patch(_CLOCK, return_value=now):
            self.processor.process_batch_result_decode(
                self.batch(req, spec=spec), result
            )

    def test_prefill_decode_and_retraction_keep_original_commit_time(self):
        """Later chunks, decode, and re-prefill must not move the first milestone."""
        for spec in (False, True):
            with self.subTest(spec=spec):
                req = self.req()
                req.inflight_middle_chunks = 2
                self.prefill(req, now=10.0, spec=spec)
                self.prefill(req, now=20.0, spec=spec)
                self.assertEqual(list(req.output_ids), [])
                self.assertEqual(req.time_stats.first_token_generated_time, 0.0)
                self.prefill(req, now=30.0, spec=spec)
                self.assertEqual(list(req.output_ids), [7])
                self.assertEqual(req.time_stats.first_token_generated_time, 30.0)
                self.decode(req, (8, 9) if spec else (8,), now=40.0, spec=spec)
                req.reset_for_retract()
                self.prefill(req, token=10, now=50.0, spec=spec)
                req.is_retracted = False
                self.prefill(req, token=11, now=60.0, spec=spec)
                self.assertEqual(
                    list(req.output_ids), [7, 8, 9, 11] if spec else [7, 8, 11]
                )
                self.assertEqual(req.time_stats.first_token_generated_time, 30.0)

    def test_decode_first_commit_and_empty_spec_result(self):
        """Decode-only requests need a milestone; zero accepted tokens do not."""
        for spec in (False, True):
            with self.subTest(spec=spec):
                req = self.req()
                if spec:
                    self.decode(req, (), now=10.0, spec=True)
                    self.assertEqual(list(req.output_ids), [])
                    self.assertEqual(req.time_stats.first_token_generated_time, 0.0)
                self.decode(req, (7, 8) if spec else (7,), now=20.0, spec=spec)
                self.assertEqual(list(req.output_ids), [7, 8] if spec else [7])
                self.assertEqual(req.time_stats.first_token_generated_time, 20.0)

    def test_aborted_or_retracted_results_do_not_create_milestone(self):
        """Discarded overlap results cannot turn an aborted request into a sample."""
        for path in (self.prefill, self.decode):
            for state in ("aborted", "retracted"):
                with self.subTest(path=path.__name__, state=state):
                    req = self.req()
                    if state == "aborted":
                        req.finished_reason = FINISH_ABORT()
                    else:
                        req.reset_for_retract()
                    path(req)
                    self.assertEqual(list(req.output_ids), [])
                    self.assertEqual(req.time_stats.first_token_generated_time, 0.0)

    def test_sampling_abort_preserves_only_existing_milestone(self):
        """A rejected sample creates no timing; an abort after output retains it."""
        for path in (self.prefill, self.decode):
            for has_output in (False, True):
                with self.subTest(path=path.__name__, has_output=has_output):
                    req = self.req()
                    if has_output:
                        self.prefill(req, now=10.0)
                    req.return_sampling_mask = True
                    logits = LogitsProcessorOutput(
                        next_token_logits=None,
                        next_token_sampling_mask_status=[SamplingMaskStatus.INVALID],
                    )
                    path(req, now=20.0, logits=logits)
                    self.assertIsInstance(req.finished_reason, FINISH_ABORT)
                    self.assertEqual(list(req.output_ids), [7] if has_output else [])
                    self.assertEqual(
                        req.time_stats.first_token_generated_time,
                        10.0 if has_output else 0.0,
                    )

    def test_terminal_first_token_is_stamped_but_embedding_dummy_is_not(self):
        """Length-one generation and embedding both finish with an output id."""
        req = self.req(max_new_tokens=1)
        self.prefill(req, now=10.0)
        self.assertTrue(req.finished())
        self.assertEqual(req.time_stats.first_token_generated_time, 10.0)
        embedding_req = self.req(max_new_tokens=1)
        self.processor = replace(self.processor, is_generation=False)
        with patch(_CLOCK, return_value=20.0):
            self.processor.process_batch_result_prefill(
                self.batch(embedding_req),
                EmbeddingBatchResult(embeddings=torch.tensor([[0.5, 0.25]])),
            )
        self.assertTrue(embedding_req.finished())
        self.assertEqual(list(embedding_req.output_ids), [0])
        self.assertEqual(embedding_req.embedding, [0.5, 0.25])
        self.assertEqual(embedding_req.time_stats.first_token_generated_time, 0.0)

    def test_zero_token_limit_has_no_generated_milestone(self):
        """A sampled placeholder excluded by the output cap is not generation."""
        req = self.req(max_new_tokens=0)
        self.prefill(req, now=10.0)
        self.assertTrue(req.finished())
        self.assertEqual(list(req.output_ids_through_stop), [])
        self.assertEqual(req.time_stats.first_token_generated_time, 0.0)

    def test_prebuilt_marks_local_handoff_only_when_token_exists(self):
        """PD decode timestamps local availability, even for a terminal handoff."""
        self.processor = replace(
            self.processor, disaggregation_mode=DisaggregationMode.DECODE
        )
        for tokens, aborted, limit in (
            ([], False, 1),
            ([7], False, 1),
            ([7], True, 1),
            ([7], False, 0),
        ):
            with self.subTest(tokens=tokens, aborted=aborted, limit=limit):
                req = self.req(max_new_tokens=limit)
                req.output_ids.extend(tokens)
                if aborted:
                    req.finished_reason = FINISH_ABORT()
                with patch(_CLOCK, return_value=40.0):
                    self.processor.process_batch_result_prebuilt(self.batch(req))
                self.assertEqual(
                    req.time_stats.first_token_generated_time,
                    40.0 if tokens and not aborted and limit else 0.0,
                )
                if tokens and not aborted:
                    self.assertTrue(req.finished())

    def test_beam_timing_follows_committed_dag_not_placeholder_output_ids(self):
        """Terminal first beams have no placeholder; aborted launches can have one."""
        for path in ("prefill", "decode"):
            for abort in (False, True):
                with self.subTest(path=path, abort=abort):
                    req = self.req()
                    pool = SimpleNamespace(
                        device="cpu",
                        req_to_token=torch.zeros((2, 8), dtype=torch.int64),
                        alloc_rows=Mock(return_value=[1]),
                        free_rows=Mock(),
                    )
                    coordinator = BeamCoordinator(
                        model_config=None,
                        spec_algorithm=SpeculativeAlgorithm.NONE,
                        dllm_enabled=False,
                        max_req_len=8,
                        req_to_token_pool=pool,
                        token_to_kv_pool_allocator=self.processor.token_to_kv_pool_allocator,
                        tree_cache=None,
                        future_map=Mock(),
                    )
                    coordinator._num_live_groups = 1
                    processor = replace(self.processor, beam_coordinator=coordinator)
                    group = BeamGroup(beam_width=2, max_new_tokens=20 if abort else 1)
                    group.leader = req
                    group.prompt_len = 2
                    req.beam_group = group
                    req.kv.req_pool_idx = 0
                    req.kv.kv_allocated_len = req.kv.kv_committed_len = 2
                    logits = LogitsProcessorOutput(
                        next_token_logits=None,
                        beam=BeamLogitsCapture(
                            leader_logits=torch.arange(8, dtype=torch.float32).reshape(
                                1, 8
                            )
                        ),
                    )
                    coordinator.select_leader_prefill(req, 0, logits, tick=1)
                    self.assertEqual(list(req.output_ids), [0] if abort else [])
                    self.assertEqual(group.num_committed, 0)
                    if abort:
                        req.to_finish = FINISH_ABORT()
                    batch = self.batch(req)
                    batch.forward_iter = 1
                    result = GenerationBatchResult(
                        logits_output=logits, next_token_ids=torch.tensor([7])
                    )
                    with patch(_CLOCK, return_value=10.0):
                        getattr(processor, f"process_batch_result_{path}")(
                            batch, result
                        )
                    self.assertTrue(req.finished())
                    self.assertEqual(group.num_committed, 0 if abort else 1)
                    self.assertEqual(
                        req.time_stats.first_token_generated_time,
                        0.0 if abort else 10.0,
                    )
                    if abort:
                        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
                        self.assertEqual(group.final_results, [])
                    else:
                        self.assertEqual(group.final_results[0].tokens, [7])

    def test_disagg_prefill_only_successful_handoff_creates_milestone(self):
        """PD's local one-token budget must not turn zero-output requests into generation."""
        for mode in (
            "success",
            "middle",
            "preabort",
            "retry",
            "sampling",
            "grammar",
            "zero",
            "clipped_zero",
        ):
            with self.subTest(mode=mode):
                req = self.req(max_new_tokens=0 if mode == "zero" else 20)
                if mode == "clipped_zero":
                    Scheduler.init_req_max_new_tokens(
                        SimpleNamespace(
                            max_new_tokens_limit=None,
                            page_size=1,
                            max_req_len=len(req.origin_input_ids) + 1,
                            max_total_num_tokens=32,
                        ),
                        req,
                    )
                    self.assertEqual(req.sampling_params.max_new_tokens, 0)
                bootstrap = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
                bootstrap._process_req(req)
                # Sender recreation must not reinterpret the prefill-only budget.
                bootstrap._process_req(req)
                self.assertEqual(req.sampling_params.max_new_tokens, 1)
                scheduler = SchedulerDisaggregationPrefillMixin()
                scheduler.batch_result_processor = self.processor
                scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
                scheduler.tree_cache = None
                scheduler.disagg_prefill_inflight_queue = []
                scheduler.disagg_prefill_pending_chunk_rids = set()
                scheduler.send_kv_chunk = Mock()
                scheduler.output_streamer = self.processor.output_streamer
                scheduler.metrics_reporter = self.processor.metrics_reporter
                scheduler.metrics_reporter.enable_metrics = False
                scheduler.maybe_send_health_check_signal = Mock()
                scheduler.req_to_metadata_buffer_idx_allocator = Mock()
                scheduler.enable_hicache_storage = False
                scheduler.chunked_req = None
                scheduler.enable_overlap = False
                scheduler._release_aborted_request = Mock()
                scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
                scheduler.waiting_queue = []
                scheduler.processed_tokens_counter = 0
                req.metadata_buffer_index = 0
                if mode == "middle":
                    req.inflight_middle_chunks = 1
                    scheduler.chunked_req = req
                elif mode == "preabort":
                    req.to_finish = FINISH_ABORT()
                elif mode == "retry":
                    req.pending_bootstrap = True
                    req.prefill_attempt_count = 1
                elif mode == "sampling":
                    req.return_sampling_mask = True
                elif mode == "grammar":
                    req.grammar = Mock()
                    req.grammar.accept_token.side_effect = ValueError("rejected token")
                result = GenerationBatchResult(
                    next_token_ids=torch.tensor([7]),
                    logits_output=LogitsProcessorOutput(
                        next_token_logits=None,
                        next_token_sampling_mask_status=[SamplingMaskStatus.INVALID],
                    ),
                )
                with (
                    patch(_CLOCK, return_value=10.0),
                    patch.object(
                        envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB,
                        "get",
                        return_value=1.0 if mode == "retry" else 0.0,
                    ),
                ):
                    scheduler.process_batch_result_disagg_prefill(
                        self.batch(req), result
                    )
                self.assertEqual(
                    req.time_stats.first_token_generated_time,
                    10.0 if mode == "success" else 0.0,
                )
                self.assertEqual(
                    scheduler.disagg_prefill_inflight_queue,
                    [req] if mode in ("success", "zero", "clipped_zero") else [],
                )
                if mode in ("success", "zero", "clipped_zero"):
                    self.assertEqual(list(req.output_ids), [7])
                elif mode != "grammar":
                    self.assertEqual(list(req.output_ids), [])
                if mode == "retry":
                    self.assertEqual(
                        scheduler.disagg_prefill_bootstrap_queue.queue, [req]
                    )
                    self.assertTrue(req.is_retracted)

    def test_delayed_output_preserves_generated_time_through_native_response(self):
        """Forced transport batching cannot substitute receipt time for generation."""
        req = self.req(max_new_tokens=3)
        packets = []
        streamer = SchedulerOutputStreamer(
            send_to_detokenizer=SimpleNamespace(
                send_output=lambda obj: packets.append(msgpack_encode(obj))
            ),
            tree_cache=None,
            ps=SimpleNamespace(dp_rank=0, attn_tp_rank=0),
            server_args=None,
            is_generation=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            enable_hicache_storage=lambda: False,
        )
        self.processor = replace(self.processor, output_streamer=streamer)
        detokenizer = DetokenizerManager.__new__(DetokenizerManager)
        detokenizer.vocab_size = 100
        detokenizer.disable_tokenizer_batch_decode = True
        detokenizer.decode_status = {}
        detokenizer.tokenizer = SimpleNamespace(
            decode=lambda ids, **kwargs: "".join(chr(65 + i) for i in ids)
        )
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.enable_metrics = True
        manager.enable_lora = False
        manager.incremental_streaming_output = False
        manager.skip_tokenizer_init = False
        manager.dump_requests_folder = ""
        manager.crash_dump_folder = ""
        state = ReqState(
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=GenerateReqInput(rid=req.rid, text="", stream=False, log_metrics=False),
            time_stats=APIServerReqTimeStats(created_time=1.0),
        )
        manager.rid_to_state = {req.rid: state}
        with (
            patch("sglang.srt.managers.io_struct._USE_PICKLE_IPC", False),
            patch(
                "sglang.srt.managers.scheduler_components.output_streamer.DEFAULT_FORCE_STREAM_INTERVAL",
                2,
            ),
            patch(
                "sglang.srt.observability.req_time_stats.global_diff_realtime_monotonic",
                1000.0,
            ),
        ):
            self.prefill(req, now=10.0)
            self.assertEqual(packets, [])
            self.decode(req, (8,), now=20.0)
            self.assertEqual(len(packets), 1)
            token_output = msgpack_decode(packets.pop())
            string_output = detokenizer.handle_batch_token_id_out(token_output)
            with patch(_CLOCK, return_value=25.0):
                asyncio.run(
                    manager._handle_batch_output(
                        msgpack_decode(msgpack_encode(string_output))
                    )
                )
            self.assertEqual(state.time_stats.first_token_time, 25.0)
            self.assertEqual(state.out_list, [])
            self.decode(req, (9,), now=30.0)
            self.assertEqual(len(packets), 1)
            token_output = msgpack_decode(packets.pop())
            string_output = detokenizer.handle_batch_token_id_out(token_output)
            with patch(_CLOCK, return_value=35.0):
                asyncio.run(
                    manager._handle_batch_output(
                        msgpack_decode(msgpack_encode(string_output))
                    )
                )
        response = state.out_list[-1]
        self.assertEqual(response["output_ids"], [7, 8, 9])
        self.assertEqual(response["text"], "HIJ")
        self.assertEqual(response["meta_info"]["first_token_generated_time"], 1010.0)
        self.assertEqual(response["meta_info"]["scheduler_completion_time"], 1030.0)
        self.assertEqual(state.time_stats.first_token_time, 25.0)
        self.assertEqual(state.time_stats.finished_time, 35.0)

    def test_dllm_only_resolved_generated_tokens_create_milestone(self):
        """Unresolved blocks and resolved prompt-only blocks are not output."""
        for fdfo in (False, True):
            with self.subTest(fdfo=fdfo):
                req = self.req()
                req.full_untruncated_fill_ids = array("q", [1, 2, 0, 0])
                req.extend_range = SimpleNamespace(end=4)
                scheduler = SimpleNamespace(
                    dllm_config=SimpleNamespace(
                        first_done_first_out_mode=fdfo, block_size=2
                    ),
                    token_to_kv_pool_allocator=self.processor.token_to_kv_pool_allocator,
                    tree_cache=None,
                    metrics_reporter=self.processor.metrics_reporter,
                    output_streamer=self.processor.output_streamer,
                )

                def process(tokens, accepted, now):
                    result = GenerationBatchResult(
                        next_token_ids=[tokens if fdfo else torch.tensor(tokens)],
                        accept_length_per_req_cpu=[accepted] if fdfo else None,
                    )
                    with patch(_CLOCK, return_value=now):
                        SchedulerDllmMixin.process_batch_result_dllm(
                            scheduler, self.batch(req), result
                        )

                process([0, 0] if fdfo else [], 0, 10.0)
                self.assertEqual(list(req.output_ids), [])
                self.assertEqual(req.time_stats.first_token_generated_time, 0.0)
                if fdfo:
                    req.extend_range = SimpleNamespace(end=2)
                    process([1, 2], 2, 20.0)
                    self.assertEqual(list(req.output_ids), [])
                    self.assertEqual(req.time_stats.first_token_generated_time, 0.0)
                    req.extend_range = SimpleNamespace(end=4)
                process([7, 8], 2, 30.0)
                self.assertEqual(list(req.output_ids), [7, 8])
                self.assertEqual(req.time_stats.first_token_generated_time, 30.0)
                req.finished_reason = FINISH_ABORT()
                process([9, 10], 2, 40.0)
                self.assertEqual(req.time_stats.first_token_generated_time, 30.0)
                req = self.req()
                req.full_untruncated_fill_ids = array("q", [1, 2, 0, 0])
                req.extend_range = SimpleNamespace(end=4)
                req.finished_reason = FINISH_ABORT()
                process([7, 8], 2, 50.0)
                self.assertEqual(req.time_stats.first_token_generated_time, 0.0)


if __name__ == "__main__":
    unittest.main()
