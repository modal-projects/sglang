"""Unit tests for ReqTimeStats.

ReqTimeStatsBase.__setstate__ rebases perf_counter fields onto the receiving
process's clock anchor. Rebasing a field that was never stamped (0.0) turns
the sentinel into a tiny epsilon (sender_diff - receiver_diff), which defeats
== 0.0 / > 0.0 "was this stamped?" checks downstream. Concretely, a PD decode
server never stamps prefill_finished_time locally; if the sentinel arrives at
the tokenizer as an epsilon, first-token bookkeeping mistakes it for a real
stamp and the TTFT / inter-token-latency histograms record ~node-uptime-sized
garbage samples.

The same "was this stamped?" rule governs the gen_ai.latency.* span attrs: the
ones derived from finished_time are only emitted once it is stamped, and
set_finished_time() closes the trace root span, so they have to be derived
inside that call rather than by the caller.
"""

import asyncio
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sglang.srt.observability.req_time_stats as rts
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.observability.request_metrics_exporter import FileRequestMetricsExporter
from sglang.srt.observability.trace import SpanAttributes
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestSetstatePreservesUnsetTimeSentinels(CustomTestCase):
    def test_two_hop_round_trip(self):
        src = rts.SchedulerReqTimeStats()
        src.enable_metrics = True
        src.wait_queue_entry_time = 123.456
        src.prefill_finished_time = 0.0
        src.set_first_token_generated_time(ts=125.0)
        src.completion_time = 130.0

        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
            blob = pickle.dumps(src)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_005.0):
            hop1 = pickle.loads(blob)
            blob2 = pickle.dumps(hop1)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_009.0):
            hop2 = pickle.loads(blob2)

        self.assertEqual(hop2.prefill_finished_time, 0.0)
        self.assertAlmostEqual(hop2.wait_queue_entry_time, 123.456 - 9.0)
        self.assertEqual(hop2.first_token_generated_time, 116.0)
        self.assertEqual(hop2.completion_time, 121.0)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_009.0):
            meta_info = hop2.convert_to_output_meta_info()
        self.assertEqual(meta_info["first_token_generated_time"], 1_000_125.0)
        self.assertEqual(meta_info["scheduler_completion_time"], 1_000_130.0)

    def test_unset_milestones_stay_absent_after_two_hops(self):
        src = rts.SchedulerReqTimeStats(enable_metrics=True)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 100.0):
            blob = pickle.dumps(src)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 90.0):
            blob = pickle.dumps(pickle.loads(blob))
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 80.0):
            restored = pickle.loads(blob)
            meta_info = restored.convert_to_output_meta_info()
        self.assertEqual(restored.first_token_generated_time, 0.0)
        self.assertEqual(restored.completion_time, 0.0)
        self.assertNotIn("first_token_generated_time", meta_info)
        self.assertNotIn("scheduler_completion_time", meta_info)

    def test_metrics_disabled_does_not_serialize_timing(self):
        src = rts.SchedulerReqTimeStats()
        src.set_first_token_generated_time(ts=5.0)
        src.completion_time = 6.0
        restored = pickle.loads(pickle.dumps(src))
        self.assertNotIn(
            "first_token_generated_time", restored.convert_to_output_meta_info()
        )
        self.assertNotIn(
            "scheduler_completion_time", restored.convert_to_output_meta_info()
        )


class TestGeneratedTokenTiming(CustomTestCase):
    def test_retraction_and_prefill_retry_preserve_first_commit(self):
        stats = rts.SchedulerReqTimeStats()
        stats.set_first_token_generated_time(ts=5.0)
        stats.set_retract_time(ts=6.0)
        stats.reset_prefill_retry_time()
        stats.set_first_token_generated_time(ts=8.0)
        self.assertEqual(stats.first_token_generated_time, 5.0)

    def test_native_timestamps_reach_file_exporter(self):
        """Buffered first delivery must not replace the scheduler commit clock."""
        scheduler_stats = rts.SchedulerReqTimeStats(enable_metrics=True)
        scheduler_stats.set_first_token_generated_time(ts=5.0)
        scheduler_stats.completion_time = 9.0
        api_stats = rts.APIServerReqTimeStats()
        api_stats.created_time = 1.0
        api_stats.first_token_time = 12.0
        api_stats.finished_time = 13.0
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 100.0):
            meta_info = scheduler_stats.convert_to_output_meta_info()
            meta_info.update(api_stats.convert_to_output_meta_info(scheduler_stats, 3))
        out = {"meta_info": meta_info, "text": "example"}
        with tempfile.TemporaryDirectory() as directory:
            exporter = FileRequestMetricsExporter(
                ServerArgs(model_path="model", export_metrics_to_file_dir=directory),
                obj_skip_names=None,
                out_skip_names=None,
            )
            try:
                asyncio.run(exporter.write_record(GenerateReqInput(rid="test"), out))
            finally:
                exporter.close()
            files = list(Path(directory).glob("*.log"))
            self.assertEqual(len(files), 1)
            record = json.loads(files[0].read_text())
        self.assertEqual(record["first_token_generated_time"], 105.0)
        self.assertEqual(record["scheduler_completion_time"], 109.0)
        self.assertEqual(record["request_received_ts"], 101.0)
        self.assertEqual(record["request_finished_ts"], 113.0)
        self.assertEqual(
            record["scheduler_completion_time"] - record["first_token_generated_time"],
            4.0,
        )


class TestConvertToGenAiSpanAttrs(CustomTestCase):
    def _stats_after_first_token(self) -> rts.APIServerReqTimeStats:
        stats = rts.APIServerReqTimeStats()
        stats.created_time = 1.0
        stats.api_server_dispatch_finish_time = 1.1
        stats.first_token_time = 1.5
        return stats

    def test_prefill_without_finished_time_omits_e2e_and_decode(self):
        attrs = self._stats_after_first_token().convert_to_gen_ai_span_attrs()
        self.assertIn(SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL, attrs)
        self.assertIn(SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN, attrs)
        self.assertNotIn(SpanAttributes.GEN_AI_LATENCY_E2E, attrs)
        self.assertNotIn(SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE, attrs)
        self.assertNotIn(SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE, attrs)

    def test_finished_time_populates_e2e_and_decode(self):
        stats = self._stats_after_first_token()
        stats.finished_time = 2.0
        attrs = stats.convert_to_gen_ai_span_attrs()
        self.assertAlmostEqual(attrs[SpanAttributes.GEN_AI_LATENCY_E2E], 1.0)
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE], 0.5
        )
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL], 0.4
        )
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE], 0.9
        )


class TestSetFinishedTimeSpanAttrs(CustomTestCase):
    def _tracing_stats(self) -> rts.APIServerReqTimeStats:
        stats = rts.APIServerReqTimeStats()
        stats.trace_ctx = mock.MagicMock()
        stats.trace_ctx.tracing_enable = True
        return stats

    def test_passes_caller_attrs_into_trace_req_finish(self):
        stats = self._tracing_stats()

        stats.set_finished_time(ts=1.25, span_attrs={"gen_ai.request.id": "rid-1"})

        self.assertEqual(stats.finished_time, 1.25)
        stats.trace_ctx.trace_req_finish.assert_called_once_with(
            mock.ANY, attrs={"gen_ai.request.id": "rid-1"}
        )

    def test_merges_latency_attrs_derived_from_finished_time(self):
        stats = self._tracing_stats()
        stats.created_time = 1.0
        stats.api_server_dispatch_finish_time = 1.1
        stats.first_token_time = 1.5

        stats.set_finished_time(ts=2.0, span_attrs={"gen_ai.request.id": "rid-1"})

        attrs = stats.trace_ctx.trace_req_finish.call_args.kwargs["attrs"]
        self.assertEqual(attrs["gen_ai.request.id"], "rid-1")
        self.assertAlmostEqual(attrs[SpanAttributes.GEN_AI_LATENCY_E2E], 1.0)
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE], 0.5
        )
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE], 0.9
        )
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL], 0.4
        )

    def test_caller_attrs_not_mutated(self):
        stats = self._tracing_stats()
        stats.created_time = 1.0
        caller_attrs = {"gen_ai.request.id": "rid-1"}

        stats.set_finished_time(ts=2.0, span_attrs=caller_attrs)

        self.assertEqual(caller_attrs, {"gen_ai.request.id": "rid-1"})


if __name__ == "__main__":
    unittest.main()
