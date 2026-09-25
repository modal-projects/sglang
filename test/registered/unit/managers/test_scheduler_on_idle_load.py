"""on_idle's stalled-path load publish is wall-clock bounded.

A no-batch-but-not-idle stall spins on_idle without sleeping, so the gate must
cap the O(queue) get_loads for both the DP-balancing writer and the load
socket. CPU-only: builds a bare Scheduler with mocked collaborators, like
test_scheduler_flush_cache.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import NextBatchPlan
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.observability.metrics_collector import SchedulerStats

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestOnIdleStallPublish(CustomTestCase):
    def _stalled_scheduler(self) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.scheduler_stage_metrics = None
        s.maybe_send_health_check_signal = MagicMock()
        s.is_fully_idle = MagicMock(return_value=False)  # stalled, not idle
        s.publish_load_snapshot = MagicMock(return_value=None)
        s.load_publisher = MagicMock()
        s.load_inquirer = MagicMock()
        s.metrics_reporter = MagicMock()
        s._last_stall_publish_ts = float("-inf")
        return s

    def test_spinning_stall_publishes_once_within_the_floor(self):
        s = self._stalled_scheduler()
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0):
            for _ in range(100):
                s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 1)
        self.assertEqual(s.load_publisher.publish_load_stat.call_count, 1)

    def test_publishes_again_after_the_floor_elapses(self):
        s = self._stalled_scheduler()
        with patch("sglang.srt.managers.scheduler.time.monotonic") as mono:
            mono.return_value = 100.0
            s.on_idle()
            mono.return_value = 100.10  # > LOAD_STALL_REFRESH_S
            s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 2)


class TestPipelineQueueMetrics(CustomTestCase):
    def _scheduler(self, mode):
        s = Scheduler.__new__(Scheduler)
        s.scheduler_stage_metrics = None
        s.init_pp_loop_state = MagicMock()
        s.pp_loop_size = 2
        s.ps = SimpleNamespace(pp_size=2)
        s.pp_group = SimpleNamespace(is_last_rank=True)
        s.running_mbs = [
            SimpleNamespace(reqs=[], batch_is_full=False) for _ in range(2)
        ]
        s.last_mbs = [None, None]
        s.mbs = [None, None]
        s.send_proxy_work = []
        s.waiting_queue = []
        s.chunked_req = None
        s.grammar_manager = []
        s.enable_priority_scheduling = False
        s.disaggregation_mode = mode
        s.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        s.disagg_prefill_inflight_queue = [object()]
        s.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[], held_rebootstrap_reqs=[]
        )
        s.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        s.pool_stats_observer = SimpleNamespace(
            get_pool_stats=lambda: SimpleNamespace(
                update_scheduler_stats=lambda stats: None
            ),
            streaming_session_count=lambda: 0,
            session_held_tokens=lambda: 0,
        )
        s.on_idle = MagicMock(
            side_effect=AssertionError("Pending queues bypass idle housekeeping")
        )
        s._pp_commit_comm_work = MagicMock()
        s._pp_commit_send_output_work_and_preprocess_output_tensors = MagicMock(
            return_value=(None, None, None)
        )
        s._pp_pd_send_consensus_bootstrapped_ids = MagicMock(return_value=([], []))
        s._pp_pd_send_consensus_release_ids = MagicMock(return_value=([], []))
        s._pp_recv_pyobj_from_prev_stage = MagicMock(return_value=[])
        s._pp_pd_get_bootstrapped_ids = MagicMock(return_value=[])
        s._pp_pd_get_prefill_transferred_ids = MagicMock(return_value=[])
        s._pp_pd_get_retract_ids = MagicMock(return_value=[])
        s._pp_pd_get_prealloc_ids = MagicMock(return_value=[])
        s._pp_pd_get_decode_transferred_ids = MagicMock(return_value=[])
        s.process_bootstrapped_queue = MagicMock(return_value=[])
        s.process_disagg_prefill_inflight_queue = MagicMock()
        s.process_retract_queue = MagicMock(return_value=[])
        s.process_prealloc_queue = MagicMock(return_value=[])
        s.process_decode_transfer_queue = MagicMock(return_value=[])
        s.process_prefill_chunk = MagicMock()
        s._process_hicache_events = MagicMock()
        s.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch: batch
        )

        def no_batch(running_batch):
            return NextBatchPlan(batch_to_run=None, running_batch=running_batch)

        s.get_new_batch_prefill = no_batch
        s.get_next_disagg_decode_batch_to_run = no_batch
        reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
        reporter.scheduler = s
        reporter.current_scheduler_metrics_enabled = True
        reporter.stats = SchedulerStats()
        s.metrics_reporter = reporter
        return s

    def test_pending_pp_queues_publish_without_forward_at_bounded_cadence(self):
        """Pending transfers must not freeze queue metrics or publish on every spin."""
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            with self.subTest(mode=mode):
                s = self._scheduler(mode)
                clock = [0.0]
                snapshots = []

                def log_stats(stats):
                    snapshots.append(
                        (clock[0], stats.prefill_queue_depth, stats.decode_queue_depth)
                    )
                    collector.last_log_time = clock[0]

                collector = SimpleNamespace(last_log_time=0.0, log_stats=log_stats)
                s.metrics_reporter.metrics_collector = collector
                # Each sweep contains two microbatches; the last two sweeps grow
                # the queue before and after the reporter's publication window.
                frames = iter(
                    [(31.0, 1)] * 2
                    + [(31.1, 2)] * 20
                    + [(60.9, 2)] * 2
                    + [(62.0, 3)] * 2
                )

                def ingest_requests():
                    clock[0], depth = next(frames)
                    if mode == DisaggregationMode.PREFILL:
                        s.disagg_prefill_bootstrap_queue.queue = [object()] * depth
                    else:
                        s.disagg_decode_transfer_queue.queue = [
                            SimpleNamespace(is_rebootstrap=True) for _ in range(depth)
                        ] + [SimpleNamespace(is_rebootstrap=False)]
                    return []

                s.ingest_requests = ingest_requests
                with (
                    patch(
                        "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
                        return_value=SimpleNamespace(pp_async_batch_depth=0),
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_pp_mixin.get_disagg",
                        return_value=SimpleNamespace(
                            disaggregation_decode_enable_offload_kvcache=False
                        ),
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_components.metrics_reporter.ENABLE_METRICS_DEVICE_TIMER",
                        False,
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_components.metrics_reporter.time.perf_counter",
                        side_effect=lambda: clock[0],
                    ),
                    self.assertRaises(StopIteration),
                ):
                    if mode == DisaggregationMode.PREFILL:
                        s.event_loop_pp_disagg_prefill()
                    else:
                        s.event_loop_pp_disagg_decode()

                expected = (
                    [(31.0, 1, 0), (62.0, 3, 0)]
                    if mode == DisaggregationMode.PREFILL
                    else [(31.0, 0, 1), (62.0, 0, 3)]
                )
                self.assertEqual(snapshots, expected)


if __name__ == "__main__":
    unittest.main()
