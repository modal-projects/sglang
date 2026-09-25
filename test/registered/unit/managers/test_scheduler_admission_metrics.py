"""Exercise admission causes from real scheduler decisions to Prometheus samples."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Summary

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.min_free_slots_delayer import MinFreeSlotsDelayer
from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, IncLockRefResult
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestSchedulerAdmissionMetrics(CustomTestCase):
    def setUp(self):
        self.server_args = ServerArgs(model_path="dummy")
        set_global_server_args_for_scheduler(self.server_args)
        self.parallel = SimpleNamespace(pp_max_micro_batch_size=16)
        parallel = patch(
            "sglang.srt.managers.scheduler.get_parallel", return_value=self.parallel
        )
        parallel.start()
        self.addCleanup(parallel.stop)

    def request(self, rid, *, tokens=4, priority=0, max_new_tokens=8):
        # Request/cache matching is the external dependency here. Budget selection,
        # range selection, scheduler control flow and metrics remain production code.
        req = MagicMock(spec=Req)
        req.rid = rid
        req.cache_request_handle = CacheRequestHandle(rid, 0)
        req.priority = priority
        req.prefix_indices = []
        req.last_node = MagicMock()
        req.full_untruncated_fill_ids = list(range(tokens))
        req.output_ids = []
        req.sampling_params = SimpleNamespace(
            max_new_tokens=max_new_tokens, ignore_eos=False
        )
        req.time_stats = MagicMock(wait_queue_entry_time=0)
        req.retracted_stain = False
        req.host_hit_length = 0
        req.swa_host_hit_length = 0
        req.mamba_cache_miss_tokens = 0
        req.mamba_cache_miss_end = 0
        req.mamba_cache_miss_cause = "unknown"
        req._mamba_cache_miss_reported = False
        req.storage_hit_length = 0
        req.storage_hit_start = None
        req.host_hit_is_storage = False
        req.host_loaded_length = 0
        req.materialized_host_hit_len.return_value = 0
        req.fulfilled_storage_hit_len.return_value = 0
        req.finished.return_value = False
        req.needs_host_load_back.return_value = False
        req.beam_group = None
        req.lora_id = None
        req.kv = SimpleNamespace(holds_mamba=False)
        req.session = None
        req.inflight_middle_chunks = 0
        req.set_extend_range.side_effect = lambda start, end: setattr(
            req, "extend_range", Range(start, end)
        )
        return req

    def collector(self, registry, labels):

        class PrivateCollector(SchedulerMetricsCollector):
            _counter_cls = partial(Counter, registry=registry)
            _gauge_cls = partial(Gauge, registry=registry)
            _histogram_cls = partial(Histogram, registry=registry)
            _summary_cls = partial(Summary, registry=registry)

        # GaugeHistogram predates the DI hooks and imports Gauge directly.
        with patch("prometheus_client.Gauge", partial(Gauge, registry=registry)):
            return PrivateCollector(
                labels={"moe_ep_rank": "1", **labels}, server_args=self.server_args
            )

    def scheduler(self, waiting, *, running=(), slots=16, tokens=1000, enabled=True):
        registry = CollectorRegistry()
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.metrics_collector = self.collector(registry, {"model_name": "test"})
        scheduler.metrics_collector_context = SimpleNamespace(
            current_scheduler_metrics_enabled=enabled
        )
        scheduler.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        scheduler.enable_priority_preemption = False
        scheduler.enable_priority_scheduling = False
        scheduler.is_hybrid_swa = False
        scheduler.waiting_queue = list(waiting)
        scheduler.running_batch = ScheduleBatch(reqs=list(running))
        scheduler.chunked_req = None
        scheduler.min_free_slots_delayer = None
        scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: slots)
        scheduler.beam_coordinator = SimpleNamespace(
            pending_member_rows=lambda batch: 0
        )
        scheduler.policy = SimpleNamespace(calc_priority=lambda *args, **kwargs: None)
        scheduler.processed_tokens_counter = None
        scheduler.chunked_prefill_size = None
        scheduler.dynamic_chunk_sizer = None
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(extend_attention_block_m=64),
                prefill_aware_swa=False,
            )
        )
        scheduler.page_size = 1
        scheduler.tree_cache = MagicMock()
        scheduler.tree_cache.disable = False
        scheduler.tree_cache.supports_mamba.return_value = False
        scheduler.tree_cache.evictable_size.return_value = 0
        scheduler.tree_cache.full_evictable_size.return_value = 0
        scheduler.tree_cache.inc_lock_ref.return_value = IncLockRefResult()
        scheduler.tree_cache.buffer_pipeline = None
        scheduler.tree_cache.storage_prefetch_retries = None
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.token_to_kv_pool_allocator.available_size.return_value = tokens
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        scheduler.max_prefill_tokens = 1000
        scheduler.is_mixed_chunk = False
        scheduler.priority_scheduling_preemption_threshold = 0
        scheduler.max_prefill_bs = 16
        scheduler.max_running_requests = 16
        scheduler.dllm_config = None
        scheduler.enable_lora = False
        scheduler.lora_drainer = None
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.enable_hicache_storage = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_unified_cache_external_linker = False
        scheduler.truncation_align_size = None
        scheduler.model_config = None
        scheduler.enable_overlap = False
        scheduler.spec_algorithm = None
        scheduler.load_inquirer = SimpleNamespace(
            _get_num_pending_tokens=lambda **kw: 0
        )
        scheduler._add_request_to_queue = scheduler.waiting_queue.append
        return scheduler, registry

    def run_pass(self, scheduler):
        # init_new/prepare_for_extend allocate device tensors after admission.
        # Retain admitted identities while replacing only that device boundary.
        def build_batch(reqs, *args, **kwargs):
            batch = ScheduleBatch(reqs=list(reqs))
            batch.prepare_for_extend = lambda: None
            return batch

        with patch.object(ScheduleBatch, "init_new", side_effect=build_batch):
            return scheduler._get_new_batch_prefill_raw(None, scheduler.running_batch)

    def counts(self, registry):
        return {
            (sample.name, sample.labels["reason"]): sample.value
            for metric in registry.collect()
            for sample in metric.samples
            if sample.name.startswith("sglang:admission_blocked_")
            and sample.name.endswith("_total")
        }

    def assert_counts(self, registry, reason, passes, requests):
        self.assertEqual(
            self.counts(registry),
            {
                ("sglang:admission_blocked_passes_total", reason): passes,
                ("sglang:admission_blocked_requests_total", reason): requests,
            },
        )

    def test_reserved_reason_label_fails_before_registration(self):
        registry = CollectorRegistry()
        with self.assertRaisesRegex(ValueError, "reason.*reserved"):
            self.collector(registry, {"model_name": "test", "reason": "user-value"})
        self.assertEqual(list(registry.collect()), [])
        collector = self.collector(registry, {"model_name": "custom-model"})
        collector.record_admission_blocked("input_tokens", 3)
        self.assertEqual(
            registry.get_sample_value(
                "sglang:admission_blocked_requests_total",
                {
                    "moe_ep_rank": "1",
                    "model_name": "custom-model",
                    "reason": "input_tokens",
                },
            ),
            3,
        )

    def test_full_pass_retains_cause_and_empty_queue_does_not_count(self):
        scheduler, registry = self.scheduler(
            [self.request("a"), self.request("b")], tokens=1
        )
        self.assertIsNone(self.run_pass(scheduler)[0])
        self.assertTrue(scheduler.running_batch.batch_is_full)
        self.assert_counts(registry, "token_capacity", 1, 2)
        self.assertIsNone(self.run_pass(scheduler)[0])
        self.assert_counts(registry, "token_capacity", 2, 4)
        scheduler.waiting_queue.clear()
        self.assertIsNone(self.run_pass(scheduler)[0])
        self.assert_counts(registry, "token_capacity", 2, 4)

    def test_min_free_slots_delays_waiters(self):
        scheduler, registry = self.scheduler(
            [self.request("wait")], running=[self.request("run")], slots=1
        )
        scheduler.min_free_slots_delayer = MinFreeSlotsDelayer(2)
        self.assertIsNone(self.run_pass(scheduler)[0])
        self.assert_counts(registry, "min_free_slots", 1, 1)
        self.assertFalse(scheduler.running_batch.batch_is_full)
        self.assertEqual([r.rid for r in scheduler.waiting_queue], ["wait"])

    def test_slot_operand_beam_reservation_and_microbatch_tie(self):
        # The cause comes from the same operands as the admission decision.
        for slots, pending, beam, pp_limit, expected in [
            (3, 2, 2, 8, "request_slots"),
            (2, 0, 2, 1, "microbatch_limit"),
            (2, 0, None, 0, "microbatch_limit"),
        ]:
            with self.subTest(slots=slots, pending=pending, beam=beam, pp=pp_limit):
                candidate = self.request("candidate")
                if beam is not None:
                    candidate.beam_group = SimpleNamespace(beam_width=beam)
                waiting = [candidate]
                if pp_limit == 1:
                    second = self.request("second")
                    second.beam_group = candidate.beam_group
                    waiting.append(second)
                scheduler, registry = self.scheduler(waiting, slots=slots)
                scheduler.beam_coordinator.pending_member_rows = lambda batch: pending
                with patch.object(self.parallel, "pp_max_micro_batch_size", pp_limit):
                    capacity = scheduler.get_num_allocatable_reqs(
                        0,
                        beam,
                        running_batch=scheduler.running_batch,
                        record_admission_cause=True,
                        admitted_requests=1,
                    )
                    self.assertEqual(capacity, 1 if pp_limit == 1 else 0)
                    self.assertEqual(
                        scheduler.running_batch.admission_stop_reason, expected
                    )
                    batch, _ = self.run_pass(scheduler)
                    if capacity == 0:
                        self.assertIsNone(batch)
                    else:
                        self.assertEqual([r.rid for r in batch.reqs], ["candidate"])
                    self.assert_counts(registry, expected, 1, 1)

    def test_final_continuation_is_excluded_after_chunk_pointer_clears(self):
        continuation = self.request("continuation", tokens=4)
        scheduler, registry = self.scheduler([self.request("waiting")], slots=1)
        scheduler.chunked_prefill_size = 16
        scheduler.chunked_req = continuation
        scheduler.min_free_slots_delayer = MinFreeSlotsDelayer(2)
        batch, _ = self.run_pass(scheduler)
        self.assertEqual([r.rid for r in batch.reqs], ["continuation"])
        self.assertIsNone(scheduler.chunked_req)
        self.assertEqual([r.rid for r in scheduler.waiting_queue], ["waiting"])
        self.assert_counts(registry, "request_slots", 1, 1)

    def test_observed_skips_are_excluded_from_blocked_requests(self):
        for skip in ("lora", "prefetch", "restage", "buffer"):
            with self.subTest(skip=skip):
                skipped, blocked, unvisited = [
                    self.request(name) for name in ("skipped", "blocked", "unvisited")
                ]
                scheduler, registry = self.scheduler(
                    [skipped, blocked, unvisited], tokens=1
                )
                if skip == "lora":
                    scheduler.enable_lora = True
                    scheduler.can_schedule_lora_req = lambda req, loras: (
                        req is not skipped
                    )
                else:
                    scheduler.enable_hicache_storage = True
                    scheduler.tree_cache.check_prefetch_progress.side_effect = (
                        lambda handle: skip != "prefetch" or handle.rid != "skipped"
                    )
                    scheduler.tree_cache.pop_prefetch_loaded_span.return_value = (
                        0,
                        None,
                    )
                    scheduler._prefetch_after_device_hit_loss = lambda req: (
                        skip == "restage" and req is skipped
                    )
                    if skip == "buffer":
                        scheduler.tree_cache.buffer_pipeline = SimpleNamespace(
                            prepare_staged_prefetch=lambda req: req is not skipped
                        )
                self.assertIsNone(self.run_pass(scheduler)[0])
                self.assertEqual(scheduler.waiting_queue, [skipped, blocked, unvisited])
                self.assert_counts(registry, "token_capacity", 1, 2)

    def test_preemption_failure_counts_but_success_does_not(self):
        override = get_context().override_server_args(
            schedule_low_priority_values_first=True
        )
        override.install()
        self.addCleanup(override.restore)
        for priority, expected_admitted in [(1, False), (0, True)]:
            with self.subTest(priority=priority):
                victim = self.request("victim", priority=1)
                candidate = self.request("candidate", priority=priority)
                scheduler, registry = self.scheduler(
                    [candidate], running=[victim], slots=0
                )
                scheduler.enable_priority_preemption = True
                # Releasing/filtering request tensors is a device boundary.
                scheduler.running_batch.release_req = lambda *args: None
                scheduler.running_batch.filter_batch = lambda keep_indices: setattr(
                    scheduler.running_batch,
                    "reqs",
                    [scheduler.running_batch.reqs[i] for i in keep_indices],
                )
                # Keep queue insertion bound to the current list after replacement.
                scheduler._add_request_to_queue = lambda req: (
                    scheduler.waiting_queue.append(req)
                )
                batch, _ = self.run_pass(scheduler)
                if expected_admitted:
                    self.assertEqual([r.rid for r in batch.reqs], ["candidate"])
                    self.assertEqual(
                        [r.rid for r in scheduler.waiting_queue], ["victim"]
                    )
                    self.assertEqual(scheduler.running_batch.reqs, [])
                    self.assertEqual(self.counts(registry), {})
                else:
                    self.assertIsNone(batch)
                    self.assertEqual(scheduler.running_batch.reqs, [victim])
                    self.assert_counts(registry, "request_slots", 1, 1)

    def test_prefill_cadence_counts_waiters_without_changing_countdown(self):
        """Cadence deferrals count waiters and resume after the configured interval."""
        for enabled in (False, True):
            for waiting_count in (0, 2):
                with self.subTest(enabled=enabled, waiting_count=waiting_count):
                    waiting = [self.request(str(i)) for i in range(waiting_count)]
                    scheduler, registry = self.scheduler(waiting, enabled=enabled)
                    scheduler.scheduler_stage_metrics = None
                    scheduler.process_pending_chunked_abort = MagicMock()
                    scheduler.enable_fpm = False
                    scheduler.enable_hisparse = False
                    scheduler.require_mlp_sync = False
                    scheduler.prefill_decode_interval = 2
                    scheduler._prefill_decode_interval_remaining = 2
                    scheduler.dp_attn_adapter = SimpleNamespace(
                        maybe_prepare_mlp_sync_batch=lambda batch, **kwargs: batch,
                        maybe_convert_decode_to_extend=lambda batch: batch,
                    )
                    scheduler.ngram_embedding_manager = SimpleNamespace(
                        prepare_for_forward=lambda batch, **kwargs: batch
                    )
                    scheduler.get_new_batch_prefill = MagicMock(
                        return_value=NextBatchPlan(
                            batch_to_run=None, running_batch=scheduler.running_batch
                        )
                    )

                    for remaining in (1, 0):
                        plan = scheduler.get_next_batch_to_run(
                            scheduler.running_batch, None
                        )
                        self.assertIsNone(plan.batch_to_run)
                        self.assertEqual(scheduler.waiting_queue, waiting)
                        self.assertEqual(
                            scheduler._prefill_decode_interval_remaining, remaining
                        )
                        scheduler.get_new_batch_prefill.assert_not_called()

                    scheduler.get_next_batch_to_run(scheduler.running_batch, None)
                    scheduler.get_new_batch_prefill.assert_called_once()
                    if enabled and waiting_count:
                        self.assert_counts(
                            registry, "prefill_cadence", 2, 2 * waiting_count
                        )
                    else:
                        self.assertEqual(self.counts(registry), {})

    def test_metrics_disabled_preserves_scheduling(self):
        outcomes = []
        for enabled in (False, True):
            scheduler, registry = self.scheduler(
                [self.request("first"), self.request("second")], enabled=enabled
            )
            scheduler.max_prefill_tokens = 4
            batch, _ = self.run_pass(scheduler)
            outcomes.append(
                (
                    [r.rid for r in batch.reqs],
                    [r.rid for r in scheduler.waiting_queue],
                    scheduler.running_batch.batch_is_full,
                    batch.prefill_stats.log_input_tokens,
                )
            )
            if enabled:
                self.assert_counts(registry, "input_tokens", 1, 1)
            else:
                self.assertEqual(self.counts(registry), {})
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0], (["first"], ["second"], False, 4))


if __name__ == "__main__":
    unittest.main()
