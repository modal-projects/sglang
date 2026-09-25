import unittest
from array import array
from contextlib import ExitStack
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import torch
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Summary

from sglang.srt.environ import envs
from sglang.srt.layers.attention import trtllm_mla_backend as trt
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dots_hybrid_backend import (
    DotsHybridAttnBackend,
    DotsSWAMLAAttnBackend,
)
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.fp8_range import Fp8RangeObserver
from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollector,
    SchedulerMetricsCollectorContext,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _backend(observer):
    backend = object.__new__(trt.TRTLLMMLABackend)
    backend.data_type = torch.float8_e4m3fn
    backend.workspace_buffer = None
    backend.fp8_range_observer = observer
    return backend


def _run(backend, q, *, causal=True, layer_id=0):
    kernel = SimpleNamespace(
        prefill=SimpleNamespace(
            trtllm_ragged_attention_deepseek=lambda **kw: kw["query"]
        )
    )
    with patch.object(trt, "flashinfer", kernel, create=True):
        return backend._run_prefill_kernel(
            q=q,
            k=q,
            v=q,
            layer=RadixAttention(
                num_heads=q.shape[1],
                head_dim=q.shape[2],
                num_kv_heads=q.shape[1],
                layer_id=layer_id,
                scaling=1.0,
            ),
            batch_size=1,
            cum_seq_lens_q=None,
            max_q_len=q.shape[0],
            seq_lens_kv=None,
            cum_seq_lens_kv=None,
            max_kv_len=q.shape[0],
            is_causal=causal,
            return_lse=False,
            out_buffer=None,
        )


def _collector(stack, *, labels, every=1):
    registry = CollectorRegistry()
    for name, cls in (
        ("Counter", Counter),
        ("Gauge", Gauge),
        ("Histogram", Histogram),
        ("Summary", Summary),
    ):
        stack.enter_context(
            patch(f"prometheus_client.{name}", partial(cls, registry=registry))
        )
    stack.enter_context(envs.SGLANG_DEBUG_FP8_RANGE_EVERY.override(every))
    override = get_context().override_server_args()
    server_args = override.install()
    stack.callback(override.restore)
    labels = {"moe_ep_rank": 1, **labels}
    return SchedulerMetricsCollector(labels=labels, server_args=server_args), registry


def _reporter(stack, *, backend, collector=None, logging_rank=False):
    override = get_context().override_server_args()
    override.install()
    stack.callback(override.restore)
    scheduler = SimpleNamespace(
        device="cpu",
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(attn_backend=backend)),
        draft_worker=None,
    )
    context = SchedulerMetricsCollectorContext(
        enable_metrics=collector is not None,
        is_stats_logging_rank=logging_rank,
        current_scheduler_metrics_enabled=False,
        enable_kv_cache_events=False,
        collector=collector,
    )
    return SchedulerMetricsReporter(
        scheduler=scheduler,
        tp_rank=0,
        pp_rank=0,
        dp_rank=None,
        metrics_collector_context=context,
        metrics_collector=collector,
    )


class TestFp8Range(CustomTestCase):
    def test_strict_finite_thresholds_and_nonfinite_rows(self):
        observer = Fp8RangeObserver(num_layers=2, device="cpu", every=1)
        q = torch.tensor(
            [
                [448, -448],
                [449, 0],
                [464, 0],
                [-465, 0],
                [float("inf"), 500],
                [float("nan"), 500],
                [-float("inf"), 0],
            ]
        ).reshape(7, 1, 2)
        before = q.clone()
        observer.observe(layer_id=1, q=q)
        torch.testing.assert_close(q, before, equal_nan=True)
        self.assertEqual(
            observer.drain(),
            [(1, "gt448", 3), (1, "gt464", 1), (1, "nonfinite", 3)],
        )
        self.assertEqual(observer.drain(), [])
        observer.observe(layer_id=1, q=torch.ones(1, 2))
        self.assertEqual(observer.drain(), [])

    def test_causal_producer_observes_before_conversion_per_layer(self):
        """Prefix reuse must not advance sampling or count already-converted Q."""
        observer = Fp8RangeObserver(num_layers=2, device="cpu", every=2)
        backend = _backend(observer)
        q = torch.tensor([[[500.0, 1.0]]])
        for _ in range(3):
            _run(backend, q, causal=False)
        _run(backend, q, layer_id=0)
        _run(backend, q, layer_id=1)
        self.assertEqual(backend.drain_fp8_range_observations(), [])
        _run(backend, q, layer_id=1)
        self.assertEqual(
            backend.drain_fp8_range_observations(),
            [(1, "gt448", 1), (1, "gt464", 1)],
        )
        _run(backend, q, layer_id=0)
        self.assertEqual(
            backend.drain_fp8_range_observations(),
            [(0, "gt448", 1), (0, "gt464", 1)],
        )

    def test_empty_and_capture_do_not_advance_sampling(self):
        observer = Fp8RangeObserver(num_layers=1, device="cpu", every=2)
        observer.observe(layer_id=0, q=torch.empty(0, 1, 2))
        q = torch.tensor([[500.0]])
        with (
            patch.object(
                torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
        ):
            observer.observe(layer_id=0, q=q)
        observer.observe(layer_id=0, q=q)
        self.assertEqual(observer.drain(), [])
        observer.observe(layer_id=0, q=q)
        self.assertEqual(observer.drain(), [(0, "gt448", 1), (0, "gt464", 1)])

    def test_disabled_producer_and_reporter_do_no_observation_work(self):
        q = torch.tensor([[[500.0]]])
        backend = _backend(None)
        with patch.object(Fp8RangeObserver, "observe", side_effect=AssertionError):
            output = _run(backend, q)
        self.assertEqual(output.dtype, torch.float8_e4m3fn)
        self.assertEqual(backend.drain_fp8_range_observations(), [])
        with ExitStack() as stack:
            collector, registry = _collector(stack, labels={}, every=0)
            reporter = _reporter(stack, backend=backend, collector=collector)
            with patch.object(
                backend,
                "drain_fp8_range_observations",
                wraps=backend.drain_fp8_range_observations,
            ) as drain:
                reporter.report_prefill_stats(None, None, False)
            drain.assert_not_called()
            self.assertIsNone(collector.fp8_range_rows)
            self.assertNotIn(
                "sglang:fp8_range_rows", {m.name for m in registry.collect()}
            )

    def test_wrapped_backend_drains_each_observer_once(self):
        first = _backend(Fp8RangeObserver(num_layers=1, device="cpu", every=1))
        second = _backend(Fp8RangeObserver(num_layers=1, device="cpu", every=1))
        _run(first, torch.tensor([[[500.0]]]))
        _run(second, torch.tensor([[[float("inf")]]]))
        hybrid = object.__new__(HybridAttnBackend)
        hybrid.prefill_backend = first
        hybrid.decode_backend = first
        linear = object.__new__(HybridLinearAttnBackend)
        linear.full_attn_backend = hybrid
        linear.linear_attn_backend = SimpleNamespace(
            drain_fp8_range_observations=lambda: (
                AttentionBackend.drain_fp8_range_observations(None)
            )
        )
        tbo = object.__new__(TboAttnBackend)
        tbo.primary = linear
        tbo.children = [second, linear, second]
        self.assertEqual(
            tbo.drain_fp8_range_observations(),
            [(0, "gt448", 1), (0, "gt464", 1), (0, "nonfinite", 1)],
        )
        self.assertEqual(tbo.drain_fp8_range_observations(), [])

    def test_rank_local_export_before_logging_rank_early_return(self):
        with ExitStack() as stack:
            collector, registry = _collector(stack, labels={"tp_rank": "1"})
            backend = _backend(Fp8RangeObserver(num_layers=1, device="cpu", every=1))
            _run(backend, torch.tensor([[[500.0]], [[float("nan")]]]))
            reporter = _reporter(stack, backend=backend, collector=collector)
            reporter.report_prefill_stats(None, None, False)
            for kind in ("gt448", "gt464", "nonfinite"):
                self.assertEqual(
                    registry.get_sample_value(
                        "sglang:fp8_range_rows_total",
                        {
                            "moe_ep_rank": "1",
                            "tp_rank": "1",
                            "layer": "0",
                            "kind": kind,
                        },
                    ),
                    1,
                )
            self.assertEqual(backend.drain_fp8_range_observations(), [])

    def test_dots_wrappers_export_sampled_child_observations(self):
        """Dots wrappers must not hide sampled rows from the scheduler drain."""
        with ExitStack() as stack:
            collector, registry = _collector(stack, labels={"tp_rank": "0"})
            first = _backend(Fp8RangeObserver(num_layers=2, device="cpu", every=1))
            second = _backend(Fp8RangeObserver(num_layers=2, device="cpu", every=1))
            _run(first, torch.tensor([[[500.0]]]), layer_id=0)
            _run(second, torch.tensor([[[float("inf")]]]), layer_id=1)
            swa = object.__new__(DotsSWAMLAAttnBackend)
            swa.backend = first
            outer = object.__new__(DotsHybridAttnBackend)
            outer.dsa_backend = second
            outer.swa_backend = swa
            reporter = _reporter(stack, backend=outer, collector=collector)
            reporter._report_fp8_range_observations()
            for layer, kind in (("0", "gt448"), ("0", "gt464"), ("1", "nonfinite")):
                self.assertEqual(
                    registry.get_sample_value(
                        "sglang:fp8_range_rows_total",
                        {
                            "moe_ep_rank": "1",
                            "tp_rank": "0",
                            "layer": layer,
                            "kind": kind,
                        },
                    ),
                    1,
                )
            self.assertEqual(outer.drain_fp8_range_observations(), [])
            # Shared wrapper children represent one producer, not two samples.
            outer.dsa_backend = swa
            _run(first, torch.tensor([[[500.0]]]), layer_id=0)
            self.assertEqual(
                outer.drain_fp8_range_observations(),
                [(0, "gt448", 1), (0, "gt464", 1)],
            )

    def test_converted_extend_completion_drains_observations(self):
        """An EXTEND completion without prefill stats must consume recorded rows."""
        for every in (1, 0):
            with self.subTest(every=every), ExitStack() as stack:
                collector, registry = _collector(
                    stack, labels={"tp_rank": "1"}, every=every
                )
                observer = (
                    Fp8RangeObserver(num_layers=1, device="cpu", every=every)
                    if every
                    else None
                )
                backend = _backend(observer)
                reporter = _reporter(stack, backend=backend, collector=collector)
                params = SamplingParams(max_new_tokens=32)
                params.normalize(None)
                req = Req(
                    rid="converted",
                    origin_input_text="",
                    origin_input_ids=array("q", [1, 2]),
                    sampling_params=params,
                    vocab_size=128,
                )
                req.output_ids.append(3)
                batch = ScheduleBatch(reqs=[req])
                batch.forward_mode = ForwardMode.DECODE
                batch.seq_lens_cpu = torch.tensor([3])
                batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)
                batch.convert_decode_to_extend()
                self.assertIsNone(batch.prefill_stats)
                _run(backend, torch.tensor([[[500.0]]]))
                processor = SchedulerBatchResultProcessor(
                    is_generation=True,
                    disaggregation_mode=None,
                    enable_overlap=False,
                    enable_overlap_mlx=False,
                    model_config=SimpleNamespace(think_end_ids=None),
                    token_to_kv_pool_allocator=Mock(),
                    tree_cache=None,
                    hisparse_coordinator=None,
                    req_to_token_pool=None,
                    decode_offload_manager=None,
                    metrics_collector=collector,
                    metrics_reporter=reporter,
                    draft_worker=None,
                    model_worker=Mock(),
                    logprob_result_processor=None,
                    output_streamer=Mock(),
                    beam_coordinator=Mock(),
                    abort_request=Mock(),
                )
                result = GenerationBatchResult(
                    next_token_ids=torch.tensor([4]),
                    logits_output=LogitsProcessorOutput(next_token_logits=None),
                    extend_input_len_per_req=[1],
                    extend_logprob_start_len_per_req=[0],
                )
                with (
                    patch.object(
                        backend,
                        "drain_fp8_range_observations",
                        wraps=backend.drain_fp8_range_observations,
                    ) as drain,
                    patch.object(
                        reporter,
                        "report_prefill_stats",
                        wraps=reporter.report_prefill_stats,
                    ) as prefill_stats,
                ):
                    processor.process_batch_result_prefill(batch, result)
                prefill_stats.assert_not_called()
                self.assertEqual(list(req.output_ids), [3, 4])
                for kind in ("gt448", "gt464"):
                    self.assertEqual(
                        registry.get_sample_value(
                            "sglang:fp8_range_rows_total",
                            {
                                "moe_ep_rank": "1",
                                "tp_rank": "1",
                                "layer": "0",
                                "kind": kind,
                            },
                        ),
                        1 if every else None,
                    )
                self.assertEqual(drain.call_count, int(every > 0))
                self.assertEqual(backend.drain_fp8_range_observations(), [])

    def test_warning_groups_kinds_and_is_bounded(self):
        observer = Fp8RangeObserver(num_layers=1, device="cpu", every=1)
        backend = _backend(observer)
        module = "sglang.srt.managers.scheduler_components.metrics_reporter"
        with ExitStack() as stack:
            stack.enter_context(envs.SGLANG_DEBUG_FP8_RANGE_EVERY.override(1))
            reporter = _reporter(stack, backend=backend, logging_rank=True)
            with patch(f"{module}.time.monotonic", side_effect=[1, 2, 61]):
                with self.assertLogs(module, level="WARNING") as logs:
                    for _ in range(3):
                        _run(backend, torch.tensor([[[500.0]], [[float("inf")]]]))
                        reporter._report_fp8_range_observations()
        self.assertEqual(len(logs.output), 2)
        for line in logs.output:
            for kind in ("gt448", "gt464", "nonfinite"):
                self.assertIn(kind, line)

    def test_standard_labels_and_reserved_label_rejection(self):
        with ExitStack() as stack:
            collector, registry = _collector(stack, labels={})
            collector.increment_fp8_range_rows(0, "gt448", 2)
            self.assertEqual(
                registry.get_sample_value(
                    "sglang:fp8_range_rows_total",
                    {"moe_ep_rank": "1", "layer": "0", "kind": "gt448"},
                ),
                2,
            )
        for reserved in ("layer", "kind"):
            with ExitStack() as stack, self.assertRaisesRegex(ValueError, "reserve"):
                _collector(stack, labels={reserved: "user"})


if __name__ == "__main__":
    unittest.main()
