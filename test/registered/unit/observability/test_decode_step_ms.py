"""Pure-CPU unit tests for the decode-log ``step-ms``/``gap-ms`` accounting.

Host wall intervals ending in decode contribute to average ``step-ms``;
idle, prefill and pause intervals contribute to total ``gap-ms``. Neither
measurement isolates device execution.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import logging
import re
import types
import unittest
from array import array
from collections import deque
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    PrefillStats,
    SchedulerMetricsReporter,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollectorContext,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase

_LOGGER_NAME = "sglang.srt.managers.scheduler_components.metrics_reporter"


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def set(self, t):
        self.t = t


def _make_scheduler():
    scheduler = MagicMock()
    scheduler.device = "cuda"
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.waiting_queue = []
    scheduler.forward_ct = 0
    scheduler.spec_algorithm = types.SimpleNamespace(is_none=lambda: True)
    scheduler.pool_stats_observer.get_pool_stats.return_value = types.SimpleNamespace(
        get_decode_usage_msg_parts=lambda: [],
        get_prefill_usage_msg_parts=lambda: [],
    )
    return scheduler


def _bind_result_dispatch(scheduler, scheduler_class):
    # Some scheduler versions extract dispatch from the outer result boundary.
    # Keep that production path active instead of creating a no-op child mock.
    dispatch = getattr(scheduler_class, "_process_batch_result", None)
    if dispatch is not None:
        scheduler._process_batch_result = types.MethodType(dispatch, scheduler)


def _make_reporter() -> SchedulerMetricsReporter:
    context = SchedulerMetricsCollectorContext(
        enable_metrics=False,
        is_stats_logging_rank=True,
        current_scheduler_metrics_enabled=False,
        enable_kv_cache_events=False,
        collector=None,
    )
    return SchedulerMetricsReporter(
        scheduler=_make_scheduler(),
        tp_rank=0,
        pp_rank=0,
        dp_rank=None,
        metrics_collector_context=context,
        metrics_collector=None,
    )


def _make_result_processor(reporter):
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=DisaggregationMode.NULL,
        enable_overlap=False,
        enable_overlap_mlx=False,
        model_config=types.SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=MagicMock(),
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=reporter,
        draft_worker=None,
        model_worker=MagicMock(),
        logprob_result_processor=None,
        output_streamer=MagicMock(),
        beam_coordinator=MagicMock(),
        abort_request=lambda *args, **kwargs: None,
    )


def _fake_batch():
    return types.SimpleNamespace(
        reqs=[object()],
        batch_size=lambda: 1,
        forward_iter=None,
        dp_cooperation_info=None,
    )


def _decode_lines(records):
    return [r.getMessage() for r in records if "Decode batch" in r.getMessage()]


def _metric(line, name):
    m = re.search(rf"{name}: ([0-9.]+)", line)
    return float(m.group(1)) if m else None


class _ReporterTestBase(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(decode_log_interval=1)
        override.install()
        self.addCleanup(override.restore)
        self.clock = _Clock()
        self._patcher = patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter."
            "time.perf_counter",
            new=self.clock,
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        self.reporter = _make_reporter()

    def _decode(self, t):
        self.reporter.forward_ct_decode = (self.reporter.forward_ct_decode + 1) % (
            1 << 30
        )
        self.clock.set(t)
        self.reporter.report_decode_stats(
            can_run_cuda_graph=True, running_batch=_fake_batch()
        )

    def _idle(self, t):
        self.clock.set(t)
        self.reporter.mark_idle()

    def _prefill(self, t):
        self.clock.set(t)
        self.reporter.report_prefill_stats(
            batch=None,
            prefill_stats=PrefillStats(
                log_input_tokens=1,
                log_hit_tokens=0,
                new_token_ratio=1.0,
                num_running_reqs=types.SimpleNamespace(total=0),
                num_new_seqs=1,
            ),
            can_run_cuda_graph=False,
        )


class TestIdleGapExcluded(_ReporterTestBase):
    def test_idle_and_prefill_time_go_to_gap_not_step(self):
        self.clock.set(1.0)
        self._decode(1.0)
        self._idle(2.0)
        self._idle(13.0)
        self._prefill(13.05)
        logger = logging.getLogger(_LOGGER_NAME)
        with self.assertLogs(logger, level="INFO") as cm:
            self._decode(13.07)

        lines = _decode_lines(cm.records)
        self.assertEqual(len(lines), 1)
        line = lines[0]
        m = re.search(r"step-ms: ([0-9.]+)", line)
        self.assertIsNotNone(m)
        self.assertAlmostEqual(float(m.group(1)), 20.0, delta=0.1)
        self.assertAlmostEqual(_metric(line, "gap-ms"), 12050.0, delta=0.1)
        # Field order kept: gen throughput, step-ms, gap-ms, #queue-req.
        idx_throughput = line.index("gen throughput (token/s):")
        idx_step = line.index("step-ms:")
        idx_queue = line.index("#queue-req:")
        self.assertLess(idx_throughput, idx_step)
        self.assertLess(idx_step, idx_queue)


class TestBackToBackDecode(_ReporterTestBase):
    def test_back_to_back_decodes_unchanged(self):
        logger = logging.getLogger(_LOGGER_NAME)
        self._decode(1.000)
        with self.assertLogs(logger, level="INFO") as cm2:
            self._decode(1.013)
        with self.assertLogs(logger, level="INFO") as cm3:
            self._decode(1.026)

        for records in (cm2.records, cm3.records):
            lines = _decode_lines(records)
            self.assertEqual(len(lines), 1)
            self.assertAlmostEqual(_metric(lines[0], "step-ms"), 13.0, delta=0.1)
            self.assertAlmostEqual(_metric(lines[0], "gap-ms"), 0.0, delta=0.1)


class TestDecodeLogInterval(_ReporterTestBase):
    def setUp(self):
        super().setUp()
        self.reporter.decode_log_interval = 4

    def test_interval_averages_steps_and_accumulates_gap(self):
        logger = logging.getLogger(_LOGGER_NAME)
        for t in (0.010, 0.020, 0.030):
            with self.assertNoLogs(logger, level="INFO"):
                self._decode(t)
        with self.assertLogs(logger, level="INFO") as cm:
            self._decode(0.040)
        lines = _decode_lines(cm.records)
        self.assertEqual(len(lines), 1)
        self.assertAlmostEqual(_metric(lines[0], "step-ms"), 10.0, delta=0.1)
        self.assertAlmostEqual(_metric(lines[0], "gap-ms"), 0.0, delta=0.1)

        self._prefill(0.100)
        for t in (0.110, 0.120, 0.130):
            with self.assertNoLogs(logger, level="INFO"):
                self._decode(t)
        with self.assertLogs(logger, level="INFO") as cm2:
            self._decode(0.140)
        lines = _decode_lines(cm2.records)
        self.assertEqual(len(lines), 1)
        self.assertAlmostEqual(_metric(lines[0], "step-ms"), 10.0, delta=0.1)
        self.assertAlmostEqual(_metric(lines[0], "gap-ms"), 60.0, delta=0.1)


class TestConvertedDecodeAccounting(_ReporterTestBase):
    def _check_converted_decode(self, *, following_prefill):
        self._decode(1.0)
        params = SamplingParams(max_new_tokens=10)
        params.normalize(tokenizer=None)
        req = Req(
            rid="converted",
            origin_input_text="",
            origin_input_ids=array("q", [1, 2]),
            sampling_params=params,
        )
        req.output_ids.append(3)
        batch = ScheduleBatch(
            reqs=[req],
            forward_mode=ForwardMode.DECODE,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            seq_lens_cpu=torch.tensor([4]),
        )
        batch.convert_decode_to_extend()
        processor = _make_result_processor(self.reporter)

        from sglang.srt.managers.scheduler import Scheduler

        scheduler = self.reporter.scheduler
        _bind_result_dispatch(scheduler, Scheduler)
        scheduler.metrics_reporter = self.reporter
        scheduler.scheduler_stage_metrics = self.reporter.scheduler_stage_metrics
        scheduler.batch_result_processor = processor
        scheduler.enable_fpm = False
        self.clock.set(3.0)
        Scheduler.process_batch_result(
            scheduler, batch, GenerationBatchResult(next_token_ids=torch.tensor([4]))
        )
        self.assertEqual(list(req.output_ids), [3, 4])
        if following_prefill:
            self._prefill(6.0)
        with self.assertLogs(_LOGGER_NAME, level="INFO") as captured:
            self._decode(7.0 if following_prefill else 4.0)
        line = _decode_lines(captured.records)[-1]
        self.assertAlmostEqual(_metric(line, "step-ms"), 1500, delta=0.1)
        self.assertAlmostEqual(
            _metric(line, "gap-ms"), 3000 if following_prefill else 0, delta=0.1
        )

    def test_converted_decode_contributes_to_average_denominator(self):
        """Two decode intervals need two samples even when one uses extend mode."""
        self._check_converted_decode(following_prefill=False)

    def test_converted_decode_is_not_charged_to_following_prefill(self):
        """A converted decode must close before the next prefill gap starts."""
        self._check_converted_decode(following_prefill=True)


class TestDecodeReportingContinuation(_ReporterTestBase):
    def _run_window(self, interval, *, raises=False, reset=False):
        self.clock.set(0.0)
        self.reporter = _make_reporter()
        self.reporter.decode_log_interval = interval
        for tick in range(1, interval):
            self._decode(float(tick))

        def slow_publish():
            self.clock.set(interval + 6.0)
            if raises:
                raise RuntimeError("publication failed")

        publisher = self.reporter.scheduler.kv_events_publisher
        publisher.publish_kv_events.side_effect = slow_publish
        if raises:
            with self.assertRaisesRegex(RuntimeError, "publication failed"):
                self._decode(float(interval))
        else:
            self._decode(float(interval))
        publisher.publish_kv_events.side_effect = None
        if reset:
            self.reporter.reset_metrics()
        self._prefill(interval + 7.0)
        for tick in range(1, interval):
            self._decode(interval + 7.0 + tick / 10)
        end = interval + 7.0 + interval / 10
        with self.assertLogs(_LOGGER_NAME, level="INFO") as captured:
            self._decode(end)
        line = _decode_lines(captured.records)[-1]
        step = _metric(line, "step-ms")
        gap = _metric(line, "gap-ms")
        carry = 0.0 if reset else 6.0
        self.assertAlmostEqual(step, (carry / interval + 0.1) * 1000, delta=0.1)
        self.assertAlmostEqual(gap, 1000, delta=0.1)
        window_start = interval + 6.0 if reset else float(interval)
        self.assertAlmostEqual(
            step * interval + gap, (end - window_start) * 1000, delta=0.2
        )

    def test_reporting_cost_continues_without_an_extra_sample(self):
        """Post-log publication costs belong to decode time in the next log window."""
        for interval in (1, 3):
            for raises in (False, True):
                with self.subTest(interval=interval, raises=raises):
                    self._run_window(interval, raises=raises)

    def test_realtime_publication_is_in_the_closing_window(self):
        """Work observed before the log snapshot must not be delayed one window."""
        scheduler = _make_scheduler()
        scheduler.enable_priority_scheduling = False
        scheduler.enable_lora = False
        scheduler.enable_hierarchical_cache = False
        scheduler.max_running_requests_under_SLO = None
        scheduler.pool_stats_observer.get_pool_stats.return_value.update_scheduler_stats = (
            lambda stats: None
        )
        collector = MagicMock()
        context = SchedulerMetricsCollectorContext(
            enable_metrics=True,
            is_stats_logging_rank=True,
            current_scheduler_metrics_enabled=True,
            enable_kv_cache_events=False,
            collector=collector,
        )
        reporter = SchedulerMetricsReporter(
            scheduler=scheduler,
            tp_rank=0,
            pp_rank=0,
            dp_rank=None,
            metrics_collector_context=context,
            metrics_collector=collector,
        )
        batch = TestResultHousekeepingAccounting._batch("ordinary")
        self.clock.set(1.0)
        reporter.forward_ct_decode = 1
        reporter.report_decode_stats(False, running_batch=batch)
        collector.increment_realtime_tokens.side_effect = lambda **kwargs: (
            self.clock.set(12.0)
        )
        self.clock.set(2.0)
        reporter.forward_ct_decode = 2
        with self.assertLogs(_LOGGER_NAME, level="INFO") as captured:
            reporter.report_decode_stats(False, running_batch=batch)
        line = _decode_lines(captured.records)[-1]
        self.assertAlmostEqual(_metric(line, "step-ms"), 11000, delta=0.1)
        self.assertAlmostEqual(_metric(line, "gap-ms"), 0, delta=0.1)
        collector.increment_realtime_tokens.side_effect = None
        self.clock.set(13.0)
        reporter.mark_idle()
        self.clock.set(13.02)
        reporter.forward_ct_decode = 3
        with self.assertLogs(_LOGGER_NAME, level="INFO") as captured:
            reporter.report_decode_stats(False, running_batch=batch)
        line = _decode_lines(captured.records)[-1]
        self.assertAlmostEqual(_metric(line, "step-ms"), 20, delta=0.1)
        self.assertAlmostEqual(_metric(line, "gap-ms"), 1000, delta=0.1)

    def test_reset_discards_post_log_decode_carry(self):
        """Reset must clear decode carry even when no new sample has arrived."""
        self._run_window(3, reset=True)


class TestResultHousekeepingAccounting(_ReporterTestBase):
    @staticmethod
    def _batch(mode):
        params = SamplingParams(max_new_tokens=10)
        params.normalize(tokenizer=None)
        req = Req(
            rid="outer-result",
            origin_input_text="",
            origin_input_ids=array("q", [1, 2]),
            sampling_params=params,
        )
        req.output_ids.append(3)
        batch = ScheduleBatch(
            reqs=[] if mode == "idle" else [req],
            forward_mode=ForwardMode.IDLE if mode == "idle" else ForwardMode.DECODE,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            seq_lens_cpu=torch.tensor([4]),
            forward_iter=1,
            launch_ts=1.0,
        )
        if mode not in ("idle", "ordinary"):
            batch.convert_decode_to_extend()
            if mode == "prefill":
                batch.decoding_reqs = []
                batch.prefill_stats = PrefillStats(
                    log_input_tokens=1,
                    log_hit_tokens=0,
                    new_token_ratio=1.0,
                    num_running_reqs=types.SimpleNamespace(total=0),
                    num_new_seqs=1,
                )
        return batch

    def _run_result(
        self, mode, hook, *, raises=False, following_prefill=False, following_idle=False
    ):
        from sglang.srt.managers.scheduler import Scheduler

        self.clock.set(0.0)
        self.reporter = _make_reporter()
        reporter = self.reporter
        scheduler = reporter.scheduler
        _bind_result_dispatch(scheduler, Scheduler)
        scheduler.metrics_reporter = reporter
        scheduler.batch_result_processor = _make_result_processor(reporter)
        scheduler.scheduler_stage_metrics = reporter.scheduler_stage_metrics
        scheduler.enable_fpm = False
        scheduler._prev_step = None
        scheduler._record_step_counters.side_effect = lambda batch, result: (
            Scheduler._record_step_counters(scheduler, batch, result)
        )
        scheduler._maybe_clear_mm_inputs.side_effect = lambda batch: (
            Scheduler._maybe_clear_mm_inputs(scheduler, batch)
        )
        scheduler.return_health_check_ipcs = (
            deque(["synthetic"]) if hook == "health" else deque()
        )
        scheduler.maybe_send_health_check_signal.side_effect = lambda: (
            Scheduler.maybe_send_health_check_signal(scheduler)
        )

        def slow_report_effect(*args, **kwargs):
            self.clock.set(12.0)
            if raises:
                raise RuntimeError("reporting failed")

        slow_report = MagicMock(side_effect=slow_report_effect)
        scheduler.ipc_channels.send_to_tokenizer.send_output.side_effect = slow_report
        reporter._device_timer_window_batch_count = 0
        reporter._device_timer_window_start = 0.0
        reporter._device_timer_window_gpu_time = 0.0
        reporter.forward_pass_device_timer = MagicMock()
        reporter.forward_pass_device_timer._report.side_effect = (
            slow_report if hook == "device" else None
        )
        self._decode(1.0)
        self.clock.set(2.0)
        with (
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.ENABLE_METRICS_DEVICE_TIMER",
                hook == "device",
            ),
            patch(
                "sglang.srt.managers.scheduler_components.batch_result_processor.maybe_cache_unfinished_req"
            ),
        ):
            if raises:
                with self.assertRaisesRegex(RuntimeError, "reporting failed"):
                    Scheduler.process_batch_result(
                        scheduler,
                        self._batch(mode),
                        GenerationBatchResult(next_token_ids=torch.tensor([4])),
                    )
            else:
                Scheduler.process_batch_result(
                    scheduler,
                    self._batch(mode),
                    GenerationBatchResult(next_token_ids=torch.tensor([4])),
                )
        slow_report.assert_called_once()
        if following_prefill or following_idle:
            scheduler.return_health_check_ipcs.clear()
            if following_prefill:
                self._prefill(13.0)
            else:
                self._idle(13.0)
        with self.assertLogs(_LOGGER_NAME, level="INFO") as captured:
            self._decode(13.02 if following_prefill or following_idle else 12.02)
        line = _decode_lines(captured.records)[-1]
        return _metric(line, "step-ms"), _metric(line, "gap-ms")

    def test_non_decode_housekeeping_and_errors_stay_in_gap(self):
        """Prefill and idle results must close after common health/device reporting."""
        for mode, hook, raises in (
            ("prefill", "health", False),
            ("prefill", "device", False),
            ("prefill", "health", True),
            ("idle", "health", False),
            ("idle", "health", True),
            ("idle", "device", False),
        ):
            with self.subTest(mode=mode, hook=hook, raises=raises):
                step, gap = self._run_result(mode, hook, raises=raises)
                self.assertAlmostEqual(step, 20, delta=0.1)
                self.assertAlmostEqual(gap, 11000, delta=0.1)

    def test_ordinary_decode_tail_is_not_charged_to_next_gap(self):
        """Common decode reporting remains decode time across a prefill or idle gap."""
        for hook, raises, idle in (
            ("health", False, False),
            ("device", False, True),
            ("health", True, False),
        ):
            with self.subTest(hook=hook, raises=raises, idle=idle):
                step, gap = self._run_result(
                    "ordinary",
                    hook,
                    raises=raises,
                    following_prefill=not idle,
                    following_idle=idle,
                )
                self.assertAlmostEqual(step, 10020, delta=0.1)
                self.assertAlmostEqual(gap, 1000, delta=0.1)
                self.assertAlmostEqual(step + gap, 11020, delta=0.1)

    def test_converted_decode_housekeeping_is_one_decode_sample(self):
        """Converted decode tails must not leak into a following prefill or add samples."""
        for hook, following_prefill in (
            ("health", False),
            ("device", False),
            ("health", True),
        ):
            with self.subTest(hook=hook, following_prefill=following_prefill):
                step, gap = self._run_result(
                    "converted", hook, following_prefill=following_prefill
                )
                self.assertAlmostEqual(step, 5510, delta=0.1)
                self.assertAlmostEqual(gap, 1000 if following_prefill else 0, delta=0.1)


class TestSchedulerGapBoundaries(_ReporterTestBase):
    def _assert_next_decode(self, t, *, step_ms, gap_ms):
        with self.assertLogs(_LOGGER_NAME, level="INFO") as captured:
            self._decode(t)
        line = _decode_lines(captured.records)[-1]
        self.assertAlmostEqual(_metric(line, "step-ms"), step_ms, delta=0.1)
        self.assertAlmostEqual(_metric(line, "gap-ms"), gap_ms, delta=0.1)

    def test_sleep_is_closed_before_immediate_decode(self):
        """A long blocked idle poll must not inflate the first waking decode."""
        from sglang.srt.managers.scheduler import Scheduler

        self._decode(1.0)
        scheduler = self.reporter.scheduler
        scheduler.metrics_reporter = self.reporter
        scheduler.scheduler_stage_metrics = self.reporter.scheduler_stage_metrics
        scheduler.is_fully_idle.return_value = True
        scheduler.enable_unified_memory = False
        scheduler.enable_hisparse = False
        scheduler.invariant_checker._check_all_pools.return_value = (False, [])
        scheduler.token_to_kv_pool_allocator.verify_byte_accounting.return_value = []
        scheduler.maybe_sleep_on_idle.side_effect = lambda: self.clock.set(12.0)
        self.clock.set(2.0)
        Scheduler.on_idle(scheduler)
        self._assert_next_decode(12.02, step_ms=20, gap_ms=11000)

    def test_stalled_idle_publish_closes_early_return(self):
        """A stalled worker's final load publication must stay in the idle gap."""
        from sglang.srt.managers.scheduler import Scheduler

        self._decode(1.0)
        scheduler = self.reporter.scheduler
        scheduler.metrics_reporter = self.reporter
        scheduler.scheduler_stage_metrics = self.reporter.scheduler_stage_metrics
        scheduler.is_fully_idle.return_value = False
        scheduler._last_stall_publish_ts = float("-inf")
        scheduler.load_publisher.publish_load_stat.side_effect = (
            lambda *args, **kwargs: self.clock.set(12.0)
        )
        self.clock.set(2.0)
        Scheduler.on_idle(scheduler)
        self._assert_next_decode(12.02, step_ms=20, gap_ms=11000)

    def test_pause_metrics_export_stays_in_gap(self):
        """Slow idle/active metric export while paused must not inflate decode."""
        from sglang.srt.managers.scheduler import Scheduler

        for fully_idle in (True, False):
            with self.subTest(fully_idle=fully_idle):
                self.reporter.reset_metrics()
                self._decode(self.clock.t + 1.0)
                start = self.clock.t
                scheduler = self.reporter.scheduler
                scheduler.metrics_reporter = self.reporter
                scheduler.is_fully_idle.return_value = fully_idle
                self.reporter.enable_metrics = True
                self.reporter.metrics_collector = MagicMock()
                self.reporter.metrics_collector.increment_scheduler_process_cpu_seconds.side_effect = (
                    lambda _: self.clock.set(start + 11.0)
                )
                with patch(
                    "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic_ns",
                    side_effect=lambda: int(self.clock.t * 1e9),
                ):
                    self.reporter.start_scheduler_time_accounting()
                    self.clock.set(start + 1.0)
                    Scheduler._record_scheduler_state_for_paused_engine(scheduler)
                self._assert_next_decode(start + 11.02, step_ms=20, gap_ms=11000)

    def test_prefill_reporting_is_closed_after_all_work(self):
        """Slow reporting after a prefill must not inflate the following decode."""
        self._decode(1.0)
        self.reporter.scheduler.kv_events_publisher.publish_kv_events.side_effect = (
            lambda: self.clock.set(12.0)
        )
        self._prefill(2.0)
        self.reporter.scheduler.kv_events_publisher.publish_kv_events.side_effect = None
        self._assert_next_decode(12.02, step_ms=20, gap_ms=11000)

    def test_prefill_early_return_closes_gap(self):
        """A rank skipping prefill logging must still exclude that prefill."""
        self._decode(1.0)
        self.reporter.is_stats_logging_rank = False
        self._prefill(2.0)
        self.reporter.is_stats_logging_rank = True
        self._assert_next_decode(2.02, step_ms=20, gap_ms=1000)

    def test_prefill_exception_closes_gap(self):
        """A failed prefill report must close its elapsed host interval."""
        self._decode(1.0)

        def fail_publish():
            self.clock.set(12.0)
            raise RuntimeError("publisher failed")

        publisher = self.reporter.scheduler.kv_events_publisher
        publisher.publish_kv_events.side_effect = fail_publish
        with self.assertRaisesRegex(RuntimeError, "publisher failed"):
            self._prefill(2.0)
        self.reporter.scheduler.kv_events_publisher.publish_kv_events.side_effect = None
        self._assert_next_decode(12.02, step_ms=20, gap_ms=11000)

    def test_dp_idle_result_includes_copy_and_output(self):
        """An idle DP result excludes both copy wait and output work from decode."""
        self._decode(1.0)
        processor = SchedulerBatchResultProcessor.__new__(SchedulerBatchResultProcessor)
        object.__setattr__(processor, "metrics_reporter", self.reporter)
        output_streamer = MagicMock()
        output_streamer._stream_output_generation.side_effect = lambda *args, **kwargs: (
            self.clock.set(12.0)
        )
        object.__setattr__(processor, "output_streamer", output_streamer)
        copy_done = MagicMock()
        copy_done.synchronize.side_effect = lambda: self.clock.set(6.0)
        self.clock.set(2.0)
        processor.process_batch_result_idle(
            types.SimpleNamespace(reqs=[], return_logprob=False),
            types.SimpleNamespace(copy_done=copy_done),
        )
        self._assert_next_decode(12.02, step_ms=20, gap_ms=11000)

    def test_pause_with_or_without_pending_requests_is_gap(self):
        """Pause accounting must exclude wall time even with queued requests."""
        from sglang.srt.managers.scheduler import Scheduler

        for fully_idle in (True, False):
            with self.subTest(fully_idle=fully_idle):
                self.reporter.reset_metrics()
                self._decode(self.clock.t + 1.0)
                scheduler = self.reporter.scheduler
                scheduler.metrics_reporter = self.reporter
                scheduler.is_fully_idle.return_value = fully_idle
                self.clock.set(self.clock.t + 60.0)
                Scheduler._record_scheduler_state_for_paused_engine(scheduler)
                self._assert_next_decode(self.clock.t + 0.015, step_ms=15, gap_ms=60000)

    def test_reset_discards_old_window_and_moves_origin(self):
        """Resetting metrics must not leak old decode or gap time into a new log."""
        self.reporter.decode_log_interval = 2
        self._decode(1.0)
        self._idle(2.0)
        self.clock.set(9.0)
        self.reporter.reset_metrics()
        self._decode(9.01)
        self._assert_next_decode(9.02, step_ms=10, gap_ms=0)

    def test_prebuilt_admission_uses_worker_wide_idle_state(self):
        """Admission closes a gap only when every PP microbatch is drained."""
        from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
        from sglang.srt.managers.scheduler import Scheduler

        for active in (
            "none",
            "running",
            "running_mbs",
            "mbs",
            "no_prebuilt",
            "filtered",
        ):
            with self.subTest(active=active):
                self.reporter.reset_metrics()
                self._decode(self.clock.t + 1.0)
                start = self.clock.t
                scheduler = self.reporter.scheduler
                scheduler.metrics_reporter = self.reporter
                scheduler.scheduler_stage_metrics = (
                    self.reporter.scheduler_stage_metrics
                )
                scheduler.enable_hisparse = False
                scheduler.chunked_req = None
                scheduler.ps.pp_size = 2
                empty = types.SimpleNamespace(is_empty=lambda: True, reqs=[])
                busy = types.SimpleNamespace(
                    is_empty=lambda: False,
                    reqs=[types.SimpleNamespace(finished=lambda: False)],
                )
                scheduler.running_mbs = [busy if active == "running_mbs" else empty]
                scheduler.mbs = [busy if active == "mbs" else None]
                scheduler._pp_decode_pending_results = [active == "mbs"]
                scheduler._is_decode_worker_idle_for_metrics.side_effect = (
                    lambda batch: (
                        SchedulerDisaggregationDecodeMixin._is_decode_worker_idle_for_metrics(
                            scheduler, batch
                        )
                    )
                )
                scheduler._pp_microbatches_drained.side_effect = lambda: (
                    Scheduler._pp_microbatches_drained(scheduler)
                )
                running = MagicMock()
                running.is_empty.return_value = active != "running"
                running.reqs = busy.reqs if active == "running" else []
                prebuilt = MagicMock()
                prebuilt.is_empty.return_value = active == "filtered"

                def plan_batch(_):
                    self.clock.set(start + 2.0)
                    return None if active == "no_prebuilt" else prebuilt

                scheduler.get_new_prebuilt_batch.side_effect = plan_batch
                scheduler.batch_result_processor.process_batch_result_prebuilt.side_effect = (
                    lambda _: self.clock.set(start + 3.0)
                )
                prebuilt.filter_batch.side_effect = lambda: self.clock.set(start + 4.0)
                scheduler.update_running_batch.side_effect = lambda batch: batch
                scheduler.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = (
                    lambda batch: batch
                )
                plan = SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run(
                    scheduler, running
                )
                if active == "no_prebuilt":
                    self.assertIsNone(plan.batch_to_run)
                    self._assert_next_decode(start + 2.02, step_ms=2020, gap_ms=0)
                elif active == "filtered":
                    self.assertIsNone(plan.batch_to_run)
                    self._assert_next_decode(start + 4.02, step_ms=20, gap_ms=4000)
                else:
                    self.assertIs(
                        plan.running_batch, running if active == "running" else prebuilt
                    )
                    self._assert_next_decode(
                        start + 4.02,
                        step_ms=20 if active == "none" else 4020,
                        gap_ms=4000 if active == "none" else 0,
                    )


class _StopPipelineLoop(Exception):
    pass


class TestPipelineGapAccounting(_ReporterTestBase):
    def _run_pipeline_case(self, mode, async_depth):
        from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        self.clock.set(0.0)
        self.reporter = _make_reporter()
        scheduler = self.reporter.scheduler
        loop_size = 2 + async_depth
        iteration = -1
        old_processed = False
        final_line = None

        def request(rid):
            params = SamplingParams(max_new_tokens=10)
            params.normalize(tokenizer=None)
            return Req(
                rid=rid,
                origin_input_text="",
                origin_input_ids=array("q", [1, 2]),
                sampling_params=params,
            )

        def batch(reqs, forward_mode):
            slots = torch.arange(len(reqs), dtype=torch.int64)
            lengths = torch.full_like(slots, 2)
            return ScheduleBatch(
                reqs=reqs,
                forward_mode=forward_mode,
                device="cpu",
                spec_algorithm=SpeculativeAlgorithm.NONE,
                model_config=types.SimpleNamespace(is_encoder_decoder=False),
                sampling_info=MagicMock(),
                req_pool_indices=slots,
                req_pool_indices_cpu=slots,
                seq_lens=lengths,
                seq_lens_cpu=lengths,
                orig_seq_lens=lengths,
            )

        old_req, new_req = request("old"), request("new")
        old_batch = batch([old_req], ForwardMode.PREBUILT)
        new_batch = batch([new_req], ForwardMode.PREBUILT)
        idle_batch = batch([], ForwardMode.IDLE)
        scheduler.metrics_reporter = self.reporter
        scheduler.scheduler_stage_metrics = self.reporter.scheduler_stage_metrics
        scheduler.ps.pp_size = 2
        scheduler.pp_group.is_last_rank = True
        scheduler.chunked_req = None
        scheduler.enable_hisparse = False
        scheduler.waiting_queue = []
        scheduler.disagg_decode_transfer_queue.queue = []
        scheduler.disagg_decode_prealloc_queue.queue = []
        scheduler._pp_pd_get_retract_ids.return_value = None
        scheduler._pp_pd_get_prealloc_ids.return_value = None
        scheduler._pp_pd_get_decode_transferred_ids.return_value = None
        scheduler._pp_pd_send_consensus_bootstrapped_ids.return_value = ([], None)
        scheduler._pp_pd_send_consensus_release_ids.return_value = ([], None)
        scheduler.init_pp_loop_state.side_effect = lambda: (
            SchedulerPPMixin.init_pp_loop_state(scheduler)
        )
        scheduler._pp_microbatches_drained.side_effect = lambda: (
            Scheduler._pp_microbatches_drained(scheduler)
        )
        scheduler._is_decode_worker_idle_for_metrics.side_effect = lambda running: (
            SchedulerDisaggregationDecodeMixin._is_decode_worker_idle_for_metrics(
                scheduler, running
            )
        )
        scheduler.on_idle.side_effect = AssertionError("unexpected idle housekeeping")

        def ingest():
            nonlocal iteration
            iteration += 1
            self.assertLessEqual(iteration, 2 * loop_size)
            if iteration == 0:
                self.clock.set(1.0)
            if mode.startswith("pending") and iteration == 1:
                self.clock.set(12.0)
                if mode == "pending_finished":
                    old_req.finished_reason = FINISH_LENGTH(1)
            elif iteration == loop_size:
                self.clock.set(12.0)
            return []

        scheduler.ingest_requests.side_effect = ingest
        arrival = 1 if mode.startswith("pending") else loop_size

        def get_prebuilt(running):
            if iteration == 0 and mode != "empty_idle":
                return old_batch
            return new_batch if iteration == arrival else None

        scheduler.get_new_prebuilt_batch.side_effect = get_prebuilt

        def update(running):
            running.filter_batch()
            running.forward_mode = ForwardMode.DECODE
            return running

        scheduler.update_running_batch.side_effect = update
        scheduler.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = (
            lambda running: (
                idle_batch if mode == "empty_idle" and iteration == 0 else running
            )
        )
        scheduler.get_next_disagg_decode_batch_to_run.side_effect = (
            lambda *, running_batch: (
                SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run(
                    scheduler, running_batch
                )
            )
        )
        scheduler._pp_launch_batch.side_effect = lambda *args: (
            GenerationBatchResult(),
            MagicMock(),
        )
        scheduler._pp_commit_send_output_work_and_preprocess_output_tensors.side_effect = (
            lambda *args: (None, GenerationBatchResult(), MagicMock())
        )
        scheduler._pp_process_batch_result.side_effect = lambda current, result: (
            SchedulerPPMixin._pp_process_batch_result(scheduler, current, result)
        )

        def consume(current, result):
            nonlocal old_processed, final_line
            if current.forward_mode.is_idle():
                self.clock.set(2.0)
                processor = SchedulerBatchResultProcessor.__new__(
                    SchedulerBatchResultProcessor
                )
                object.__setattr__(processor, "metrics_reporter", self.reporter)
                object.__setattr__(processor, "output_streamer", MagicMock())
                processor.process_batch_result_idle(current, result)
                old_processed = True
                return
            if (
                old_req in current.reqs
                and not mode.startswith("pending")
                and not old_processed
            ):
                self.clock.set(2.0)
                if mode == "finished_decode":
                    old_req.finished_reason = FINISH_LENGTH(1)
                old_processed = True
                self.reporter.forward_ct_decode += 1
                self.reporter.report_decode_stats(True, running_batch=current)
                return
            self.clock.set(12.02)
            self.reporter.forward_ct_decode += 1
            with self.assertLogs(_LOGGER_NAME, level="INFO") as captured:
                self.reporter.report_decode_stats(True, running_batch=current)
            final_line = _decode_lines(captured.records)[-1]
            raise _StopPipelineLoop()

        scheduler.process_batch_result.side_effect = consume
        self._decode(1.0)
        with (
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
                return_value=types.SimpleNamespace(pp_async_batch_depth=async_depth),
            ),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.get_disagg",
                return_value=types.SimpleNamespace(
                    disaggregation_decode_enable_offload_kvcache=False
                ),
            ),
            self.assertRaises(_StopPipelineLoop),
        ):
            SchedulerPPMixin.event_loop_pp_disagg_decode(scheduler)
        return _metric(final_line, "step-ms"), _metric(final_line, "gap-ms")

    def test_consumed_slots_are_distinct_from_pending_results(self):
        """Retained finished slots are idle; even finished requests can have pending output."""
        for async_depth in (0, 1):
            for mode, expected_step, expected_gap in (
                ("empty_idle", 20, 11000),
                ("finished_decode", 20, 10000),
                ("processed_live", 10020, 0),
                ("pending_live", 11020, 0),
                ("pending_finished", 11020, 0),
            ):
                with self.subTest(mode=mode, async_depth=async_depth):
                    step, gap = self._run_pipeline_case(mode, async_depth)
                    self.assertAlmostEqual(step, expected_step, delta=0.1)
                    self.assertAlmostEqual(gap, expected_gap, delta=0.1)


if __name__ == "__main__":
    unittest.main()
