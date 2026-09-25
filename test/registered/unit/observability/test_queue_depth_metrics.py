"""Pure-CPU unit tests for the queue-depth gauges on the scheduler metrics path.

``sglang:prefill_queue_depth`` / ``sglang:decode_queue_depth`` split the single
non-PD ``waiting_queue`` by ``Req.is_retracted`` (computed inside the existing
``QueueCount.from_reqs`` pass), and ``sglang:num_prefill_inflight_reqs`` reflects
``scheduler.chunked_req``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    PrefillStats,
    SchedulerMetricsReporter,
)
from sglang.srt.observability.metrics_collector import (
    QueueCount,
    SchedulerMetricsCollector,
    SchedulerMetricsCollectorContext,
    SchedulerStats,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler


class _FakeReq:
    def __init__(
        self, is_retracted: bool = False, priority=None, retracted_stain=False
    ):
        self.is_retracted = is_retracted
        self.priority = priority
        self.retracted_stain = retracted_stain
        self.routing_key = None
        self.seqlen = 4


class _BoundRecordingMetric:
    def __init__(self, metric, labels):
        self.metric = metric
        self.labels = labels

    def set(self, value):
        self.metric.values[tuple(sorted(self.labels.items()))] = value

    def inc(self, value=1):
        pass

    def observe(self, value):
        pass


class _RecordingMetric:
    def __init__(self, *args, name=None, labelnames=(), **kwargs):
        self.name = name if name is not None else args[0]
        self.labelnames = tuple(labelnames)
        self.values = {}

    def labels(self, *values, **labels):
        if values:
            labels = dict(zip(self.labelnames, values, strict=True))
        return _BoundRecordingMetric(self, labels)


def _make_reporter(scheduler) -> SchedulerMetricsReporter:
    context = SchedulerMetricsCollectorContext(
        enable_metrics=False,
        is_stats_logging_rank=True,
        current_scheduler_metrics_enabled=False,
        enable_kv_cache_events=False,
        collector=None,
    )
    with patch.object(SchedulerMetricsReporter, "__init__", return_value=None):
        reporter = SchedulerMetricsReporter()
    reporter.scheduler = scheduler
    reporter.metrics_collector_context = context
    reporter.metrics_collector = None
    reporter.stats = SchedulerStats()
    return reporter


class TestQueueCountRetracted(CustomTestCase):
    def test_counts_retracted_in_same_pass(self):
        class OnePassList(list):
            passes = 0

            def __iter__(self):
                self.passes += 1
                if self.passes > 1:
                    raise AssertionError("Queue counting traversed requests twice")
                return super().__iter__()

        reqs = OnePassList([_FakeReq(), _FakeReq(is_retracted=True), _FakeReq()])
        qc = QueueCount.from_reqs(reqs, count_retracted=True)
        self.assertEqual(qc.total, 3)
        self.assertEqual(qc.num_retracted, 1)
        self.assertIsNone(qc.by_priority)

    def test_priority_breakdown_and_retracted_together(self):
        reqs = [
            _FakeReq(priority=0),
            _FakeReq(is_retracted=True, priority=1),
            _FakeReq(priority=1),
        ]
        qc = QueueCount.from_reqs(
            reqs, enable_priority_scheduling=True, count_retracted=True
        )
        self.assertEqual(qc.by_priority, {0: 1, 1: 2})
        self.assertEqual(qc.num_retracted, 1)

    def test_default_does_not_count_retracted(self):
        qc = QueueCount.from_reqs([object()])
        self.assertEqual((qc.total, qc.num_retracted), (1, 0))

    def test_empty_queue(self):
        qc = QueueCount.from_reqs([], count_retracted=True)
        self.assertEqual((qc.total, qc.num_retracted), (0, 0))


class TestQueueDepths(CustomTestCase):
    def _scheduler(self, waiting, chunked_req=None, mode=DisaggregationMode.NULL):
        return types.SimpleNamespace(
            waiting_queue=waiting,
            chunked_req=chunked_req,
            disaggregation_mode=mode,
        )

    def test_non_pd_split_by_retraction(self):
        waiting = [_FakeReq(), _FakeReq(is_retracted=True), _FakeReq(), _FakeReq()]
        reporter = _make_reporter(self._scheduler(waiting))
        reporter.stats.num_queue_reqs = QueueCount.from_reqs(
            waiting, count_retracted=True
        )
        reporter._update_queue_depths()
        self.assertEqual(reporter.stats.prefill_queue_depth, 3)
        self.assertEqual(reporter.stats.decode_queue_depth, 1)
        self.assertEqual(
            reporter.stats.prefill_queue_depth + reporter.stats.decode_queue_depth,
            reporter.stats.num_queue_reqs.total,
        )
        self.assertEqual(reporter.stats.num_prefill_inflight_reqs, 0)

    def test_chunked_prefill_inflight(self):
        reporter = _make_reporter(self._scheduler([], chunked_req=_FakeReq()))
        reporter.stats.num_queue_reqs = QueueCount.from_reqs([], count_retracted=True)
        reporter._update_queue_depths()
        self.assertEqual(reporter.stats.prefill_queue_depth, 0)
        self.assertEqual(reporter.stats.decode_queue_depth, 0)
        self.assertEqual(reporter.stats.num_prefill_inflight_reqs, 1)

    def test_pd_prefill_engine(self):
        scheduler = self._scheduler([_FakeReq()], mode=DisaggregationMode.PREFILL)
        scheduler.disagg_prefill_bootstrap_queue = types.SimpleNamespace(queue=[1, 2])
        # Prefill-completed KV transfers are excluded: already prefilled.
        scheduler.disagg_prefill_inflight_queue = [1]
        reporter = _make_reporter(scheduler)
        reporter._update_queue_depths()
        self.assertEqual(reporter.stats.prefill_queue_depth, 3)
        self.assertEqual(reporter.stats.decode_queue_depth, 0)

    def test_pd_decode_engine(self):
        reboot_waiting = _FakeReq()
        reboot_waiting.pd_rebootstrap_in_progress = True
        both_markers = _FakeReq(retracted_stain=True)
        both_markers.pd_rebootstrap_in_progress = True
        scheduler = self._scheduler(
            [_FakeReq(), _FakeReq(retracted_stain=True), reboot_waiting, both_markers],
            mode=DisaggregationMode.DECODE,
        )
        fresh = types.SimpleNamespace(is_rebootstrap=False)
        reboot = types.SimpleNamespace(is_rebootstrap=True)
        # Retracted, rebootstrap-held, in-flight rebootstraps, and restored
        # retractions still in waiting_queue count (both markers -> once);
        # ordinary first-decode handoffs in prealloc/transfer/waiting do not.
        scheduler.disagg_decode_prealloc_queue = types.SimpleNamespace(
            queue=[fresh, reboot],
            retracted_queue=[1, 2, 3, 4],
            held_rebootstrap_reqs=[1],
        )
        scheduler.disagg_decode_transfer_queue = types.SimpleNamespace(
            queue=[fresh, reboot]
        )
        reporter = _make_reporter(scheduler)
        reporter._update_queue_depths()
        self.assertEqual(reporter.stats.prefill_queue_depth, 0)
        self.assertEqual(reporter.stats.decode_queue_depth, 10)


class _RecordingCollector(SchedulerMetricsCollector):
    _counter_cls = _RecordingMetric
    _gauge_cls = _RecordingMetric
    _histogram_cls = _RecordingMetric
    _summary_cls = _RecordingMetric


class TestCollectorGauges(CustomTestCase):
    LABELS = {
        "model_name": "m",
        "engine_type": "unified",
        "tp_rank": 0,
        "pp_rank": 0,
        "moe_ep_rank": 0,
    }

    # Built once: the collector also registers non-DI'd metrics (GaugeHistogram)
    # on the global prometheus registry, which rejects duplicates.
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Upstream reads prefill-delayer buckets from the runtime context
        # (get_schedule()) and gates the EPLB balancedness summary on
        # exports_expert_balancedness_to_prometheus(); neither is published in
        # a pure unit test, so stub both at the module import site.
        schedule = types.SimpleNamespace(
            max_queued_requests=None,
            prefill_delayer_max_delay_passes=30,
            prefill_delayer_forward_passes_buckets=None,
            prefill_delayer_wait_seconds_buckets=None,
        )
        with (
            patch(
                "sglang.srt.observability.metrics_collector.get_schedule",
                return_value=schedule,
            ),
            patch(
                "sglang.srt.observability.metrics_collector.exports_expert_balancedness_to_prometheus",
                return_value=False,
            ),
        ):
            cls.collector = _RecordingCollector(
                labels=dict(cls.LABELS), server_args=types.SimpleNamespace()
            )

    def _collector(self) -> _RecordingCollector:
        for gauge in (
            self.collector.prefill_queue_depth,
            self.collector.decode_queue_depth,
            self.collector.num_prefill_inflight_reqs,
        ):
            gauge.values.clear()
        return self.collector

    def _value(self, gauge, **extra):
        return gauge.values[tuple(sorted({**self.LABELS, **extra}.items()))]

    def _reporting_reporter(self, collector):
        override = get_context().override_server_args(
            enable_metrics=True,
            enable_mfu_metrics=False,
            enable_forward_pass_metrics=False,
            decode_log_interval=1,
        )
        override.install()
        self.addCleanup(override.restore)
        pool_stats = types.SimpleNamespace(
            get_prefill_usage_msg_parts=lambda: [],
            get_decode_usage_msg_parts=lambda: [],
            update_scheduler_stats=lambda stats: None,
        )
        scheduler = types.SimpleNamespace(
            device="cpu",
            ps=ParallelState.trivial(),
            disaggregation_mode=DisaggregationMode.NULL,
            waiting_queue=[],
            chunked_req=None,
            running_batch=types.SimpleNamespace(
                reqs=[_FakeReq()],
                seq_lens_cpu=None,
                forward_iter=0,
                dp_cooperation_info=None,
            ),
            grammar_manager=[],
            enable_priority_scheduling=True,
            enable_lora=False,
            enable_hierarchical_cache=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            pool_stats_observer=types.SimpleNamespace(
                get_pool_stats=lambda: pool_stats,
                streaming_session_count=lambda: 0,
                session_held_tokens=lambda: 0,
            ),
            kv_events_publisher=types.SimpleNamespace(
                emit_kv_metrics=lambda: None,
                publish_kv_events=lambda: None,
            ),
        )
        context = SchedulerMetricsCollectorContext(
            enable_metrics=True,
            is_stats_logging_rank=False,
            current_scheduler_metrics_enabled=True,
            enable_kv_cache_events=False,
            collector=collector,
        )
        reporter = SchedulerMetricsReporter(
            scheduler=scheduler,
            tp_rank=0,
            pp_rank=0,
            dp_rank=0,
            metrics_collector_context=context,
            metrics_collector=collector,
        )
        return reporter

    def test_periodic_reports_refresh_and_idle_clears_exported_depths(self):
        """Each reporting path must publish current queues, including drain to zero."""
        collector = self._collector()
        reporter = self._reporting_reporter(collector)
        scheduler = reporter.scheduler
        scheduler.waiting_queue = [
            _FakeReq(priority=0),
            _FakeReq(is_retracted=True, priority=1),
        ]
        scheduler.chunked_req = _FakeReq()
        reporter.report_prefill_stats(
            batch=scheduler.running_batch,
            prefill_stats=PrefillStats(4, 0, 1.0, QueueCount(total=1), 1),
            can_run_cuda_graph=False,
        )
        self.assertEqual(self._value(collector.prefill_queue_depth), 1)
        self.assertEqual(self._value(collector.decode_queue_depth), 1)
        self.assertEqual(self._value(collector.num_prefill_inflight_reqs), 1)
        self.assertEqual(self._value(collector.num_queue_reqs, priority="0"), 1)
        self.assertEqual(self._value(collector.num_queue_reqs, priority="1"), 1)

        scheduler.waiting_queue.pop(0)
        scheduler.chunked_req = None
        reporter.report_decode_stats(can_run_cuda_graph=False)
        self.assertEqual(self._value(collector.prefill_queue_depth), 0)
        self.assertEqual(self._value(collector.decode_queue_depth), 1)
        self.assertEqual(self._value(collector.num_prefill_inflight_reqs), 0)
        self.assertEqual(self._value(collector.num_queue_reqs, priority="0"), 0)

        scheduler.waiting_queue.clear()
        scheduler.running_batch.reqs.clear()
        reporter._maybe_log_idle_metrics()
        self.assertEqual(self._value(collector.prefill_queue_depth), 0)
        self.assertEqual(self._value(collector.decode_queue_depth), 0)
        self.assertEqual(self._value(collector.num_prefill_inflight_reqs), 0)
        self.assertEqual(self._value(collector.num_queue_reqs, priority="1"), 0)

    def test_pd_rebootstrap_stays_counted_until_scheduled(self):
        """A retry crossing queue boundaries must never vanish or count twice."""
        collector = self._collector()
        reporter = self._reporting_reporter(collector)
        scheduler = reporter.scheduler
        scheduler.enable_priority_scheduling = False
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.running_batch.reqs.clear()
        fresh = types.SimpleNamespace(is_rebootstrap=False)
        retry = _FakeReq(is_retracted=True, retracted_stain=True)
        wrapper = types.SimpleNamespace(is_rebootstrap=True)
        prealloc = types.SimpleNamespace(
            queue=[fresh], retracted_queue=[retry], held_rebootstrap_reqs=[]
        )
        transfer = types.SimpleNamespace(queue=[fresh])
        scheduler.disagg_decode_prealloc_queue = prealloc
        scheduler.disagg_decode_transfer_queue = transfer
        scheduler.waiting_queue = [_FakeReq()]
        stages = [
            prealloc.retracted_queue,
            prealloc.held_rebootstrap_reqs,
            prealloc.queue,
            transfer.queue,
            scheduler.waiting_queue,
        ]
        for index, stage in enumerate(stages):
            with self.subTest(stage=index):
                collector.last_log_time = 0
                reporter._maybe_log_idle_metrics()
                self.assertEqual(self._value(collector.prefill_queue_depth), 0)
                self.assertEqual(self._value(collector.decode_queue_depth), 1)
                stage.pop()
                if index + 1 < len(stages):
                    retry.is_retracted = False
                    stages[index + 1].append(wrapper if index in (1, 2) else retry)
        collector.last_log_time = 0
        reporter._maybe_log_idle_metrics()
        self.assertEqual(self._value(collector.decode_queue_depth), 0)

    def test_stalled_pd_queues_publish_without_a_forward_at_bounded_cadence(self):
        """Pending handshakes must remain visible when no model batch can run."""
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            with self.subTest(mode=mode):
                collector = self._collector()
                reporter = self._reporting_reporter(collector)
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.__dict__.update(vars(reporter.scheduler))
                reporter.scheduler = scheduler
                scheduler.metrics_reporter = reporter
                scheduler.scheduler_stage_metrics = None
                scheduler.disaggregation_mode = mode
                scheduler.enable_priority_scheduling = False
                scheduler.running_batch.reqs.clear()
                scheduler.running_batch.is_empty = lambda: True
                scheduler.last_batch = None
                scheduler.enable_overlap = False
                scheduler.dllm_manager = types.SimpleNamespace(
                    any_staging_reqs=lambda: False
                )
                scheduler.grammar_manager = MagicMock()
                scheduler.grammar_manager.grammar_queue = []
                scheduler.enable_hisparse = False
                scheduler.decode_offload_manager = None
                scheduler.maybe_send_health_check_signal = lambda: None
                scheduler.publish_load_snapshot = lambda **kwargs: None
                scheduler.load_publisher = MagicMock()
                scheduler.load_inquirer = MagicMock()
                scheduler._last_stall_publish_ts = float("-inf")
                scheduler.disagg_prefill_bootstrap_queue = types.SimpleNamespace(
                    queue=[]
                )
                scheduler.disagg_prefill_inflight_queue = []
                scheduler.disagg_decode_prealloc_queue = types.SimpleNamespace(
                    queue=[], retracted_queue=[], held_rebootstrap_reqs=[]
                )
                scheduler.disagg_decode_transfer_queue = types.SimpleNamespace(queue=[])
                if mode == DisaggregationMode.PREFILL:
                    pending = scheduler.disagg_prefill_bootstrap_queue.queue
                    gauge = collector.prefill_queue_depth
                    request = _FakeReq()
                else:
                    pending = scheduler.disagg_decode_prealloc_queue.retracted_queue
                    gauge = collector.decode_queue_depth
                    request = _FakeReq(is_retracted=True, retracted_stain=True)
                pending.extend([request, request])
                self.assertFalse(scheduler.is_fully_idle())
                collector.log_stats(SchedulerStats())
                collector.last_log_time = 0
                with (
                    patch("time.monotonic", return_value=100.0) as monotonic,
                    patch("time.perf_counter", return_value=100.0) as perf_counter,
                    patch.object(
                        reporter,
                        "_update_queue_depths",
                        wraps=reporter._update_queue_depths,
                    ) as update_depths,
                ):
                    scheduler.on_idle()
                    self.assertEqual(self._value(gauge), 2)
                    pending.append(request)
                    monotonic.return_value = 101.0
                    perf_counter.return_value = 101.0
                    for _ in range(100):
                        scheduler.on_idle()
                    self.assertEqual(self._value(gauge), 2)
                    self.assertEqual(update_depths.call_count, 1)
                    monotonic.return_value = 131.0
                    perf_counter.return_value = 131.0
                    scheduler.on_idle()
                    self.assertEqual(self._value(gauge), 3)
                    self.assertEqual(update_depths.call_count, 2)

    def test_metrics_disabled_does_not_scan_waiting_requests(self):
        """Disabled reporting must not require queue-depth request fields."""
        collector = self._collector()
        reporter = self._reporting_reporter(collector)
        reporter.enable_metrics = False
        reporter.current_scheduler_metrics_enabled = False
        reporter.scheduler.waiting_queue = [object()]
        reporter.report_prefill_stats(
            batch=None,
            prefill_stats=PrefillStats(0, 0, 1.0, QueueCount(), 0),
            can_run_cuda_graph=False,
        )
        reporter.report_decode_stats(can_run_cuda_graph=False)
        reporter._maybe_log_idle_metrics()
        self.assertEqual(collector.prefill_queue_depth.values, {})
        self.assertEqual(collector.decode_queue_depth.values, {})

    def test_gauge_names(self):
        c = self._collector()
        self.assertEqual(c.prefill_queue_depth.name, "sglang:prefill_queue_depth")
        self.assertEqual(c.decode_queue_depth.name, "sglang:decode_queue_depth")
        self.assertEqual(
            c.num_prefill_inflight_reqs.name, "sglang:num_prefill_inflight_reqs"
        )

    def test_log_stats_emits_queue_depths(self):
        c = self._collector()
        stats = SchedulerStats(
            prefill_queue_depth=3, decode_queue_depth=1, num_prefill_inflight_reqs=1
        )
        c.log_stats(stats)
        self.assertEqual(self._value(c.prefill_queue_depth), 3)
        self.assertEqual(self._value(c.decode_queue_depth), 1)
        self.assertEqual(self._value(c.num_prefill_inflight_reqs), 1)


if __name__ == "__main__":
    unittest.main()
