"""Producer outcomes must be counted only after wrapping or copying succeeds."""

import asyncio
import functools
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import prometheus_client
import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Processor(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


class _CudaTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True

    def cpu(self):
        return self.as_subclass(torch.Tensor)


def _collector(labels):
    registry = prometheus_client.CollectorRegistry()

    class Collector(TokenizerMetricsCollector):
        _counter_cls = functools.partial(prometheus_client.Counter, registry=registry)
        _gauge_cls = functools.partial(prometheus_client.Gauge, registry=registry)
        _histogram_cls = functools.partial(
            prometheus_client.Histogram, registry=registry
        )

    return Collector(labels=labels), registry


def _processor(observer=None):
    processor = object.__new__(_Processor)
    processor.use_cuda_ipc = True
    processor.use_ipc_pool_handle_cache = True
    processor.cudaipc_mmfeature_pool = Mock()
    processor.mm_feature_transport_observer = observer
    return processor


class TestFeatureTransportMetrics(CustomTestCase):
    def setUp(self):
        self.enterContext(get_context().override_server_args())

    def test_injected_counter_backend_observes_both_transports(self):
        class Metric:
            def __init__(self, *args, labelnames=(), **kwargs):
                self.labelnames = tuple(labelnames)
                self.total = 0
                self.children = {}

            def labels(self, **labels):
                key = tuple(sorted(labels.items()))
                if key not in self.children:
                    self.children[key] = Metric()
                return self.children[key]

            def inc(self, value=1):
                self.total += value

        class Collector(TokenizerMetricsCollector):
            _counter_cls = Metric
            _gauge_cls = Metric
            _histogram_cls = Metric

        for labels in ({}, {"model_name": "synthetic"}):
            with self.subTest(labels=labels):
                collector = Collector(labels=labels)
                collector.observe_mm_feature_transport("cuda_ipc", 8)
                collector.observe_mm_feature_transport("cpu_fallback", 16, 0.5)
                self.assertEqual(
                    collector.mm_feature_transport_bytes_total.labels(
                        **labels, transport="cuda_ipc"
                    ).total,
                    8,
                )
                self.assertEqual(
                    collector.mm_feature_transport_tensors_total.labels(
                        **labels, transport="cpu_fallback"
                    ).total,
                    1,
                )
                duration = collector.mm_feature_cpu_fallback_seconds_total
                if labels:
                    duration = duration.labels(**labels)
                self.assertEqual(duration.total, 0.5)

    def test_real_counters_with_and_without_base_labels(self):
        """Unlabeled duration counters must work at construction and observation."""
        for labels in ({}, {"model_name": "synthetic"}):
            with self.subTest(labels=labels):
                collector, registry = _collector(labels)
                for transport in ("cuda_ipc", "cpu_fallback"):
                    for kind in ("tensors", "bytes"):
                        self.assertEqual(
                            registry.get_sample_value(
                                f"sglang:mm_feature_transport_{kind}_total",
                                {**labels, "transport": transport},
                            ),
                            0,
                        )
                self.assertEqual(
                    registry.get_sample_value(
                        "sglang:mm_feature_cpu_fallback_seconds_total", labels
                    ),
                    0,
                )
                collector.observe_mm_feature_transport("cuda_ipc", 24)
                collector.observe_mm_feature_transport("cpu_fallback", 12, 0.25)
                collector.observe_mm_feature_transport("cpu_fallback", 8, 0.5)
                self.assertEqual(
                    registry.get_sample_value(
                        "sglang:mm_feature_cpu_fallback_seconds_total", labels
                    ),
                    0.75,
                )
                for transport, count, nbytes in (
                    ("cuda_ipc", 1, 24),
                    ("cpu_fallback", 2, 20),
                ):
                    for kind, expected in (("tensors", count), ("bytes", nbytes)):
                        self.assertEqual(
                            registry.get_sample_value(
                                f"sglang:mm_feature_transport_{kind}_total",
                                {**labels, "transport": transport},
                            ),
                            expected,
                        )

    def test_transport_omits_request_dimensions_but_retains_static_labels(self):
        """Producer metrics must not publish empty priority/custom request series."""
        for static_labels in ({}, {"model_name": "synthetic", "replica": ""}):
            with (
                self.subTest(static_labels=static_labels),
                get_context().override_server_args(
                    enable_priority_scheduling=True,
                    tokenizer_metrics_allowed_custom_labels=["request_class"],
                ),
            ):
                labels = {**static_labels, "priority": "", "request_class": ""}
                collector, registry = _collector(labels)
                processor = _processor(collector.observe_mm_feature_transport)
                processor.cudaipc_mmfeature_pool.wrap_tensor.return_value = None
                tensor = torch.ones(3).as_subclass(_CudaTensor)
                with patch("time.perf_counter", side_effect=[0.0, 0.25]):
                    self.assertFalse(
                        processor._wrap_tensor_for_cuda_ipc(tensor).is_cuda
                    )
                self.assertEqual(collector.labels, labels)
                for metric in registry.collect():
                    if metric.name.startswith("sglang:mm_feature_"):
                        for sample in metric.samples:
                            self.assertNotIn("priority", sample.labels)
                            self.assertNotIn("request_class", sample.labels)
                            for key, value in static_labels.items():
                                self.assertEqual(sample.labels[key], value)
                self.assertEqual(
                    registry.get_sample_value(
                        "sglang:mm_feature_transport_bytes_total",
                        {**static_labels, "transport": "cpu_fallback"},
                    ),
                    12,
                )
                self.assertEqual(
                    registry.get_sample_value(
                        "sglang:mm_feature_cpu_fallback_seconds_total", static_labels
                    ),
                    0.25,
                )

    def test_request_only_transport_label_does_not_collide_with_outcome(self):
        with get_context().override_server_args(
            tokenizer_metrics_allowed_custom_labels=["transport"]
        ):
            collector, registry = _collector({"transport": ""})
            collector.observe_mm_feature_transport("cpu_fallback", 12, 0.25)
            collector.observe_one_aborted_request({"transport": "request-value"})
        self.assertEqual(
            registry.get_sample_value(
                "sglang:mm_feature_transport_bytes_total", {"transport": "cpu_fallback"}
            ),
            12,
        )
        self.assertEqual(
            registry.get_sample_value("sglang:mm_feature_cpu_fallback_seconds_total"),
            0.25,
        )
        self.assertEqual(
            registry.get_sample_value(
                "sglang:num_aborted_requests_total", {"transport": "request-value"}
            ),
            1,
        )

    def test_transport_labels_are_bounded_and_reserved(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            _collector({"transport": "custom"})
        collector, registry = _collector({})
        before = prometheus_client.generate_latest(registry)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            collector.observe_mm_feature_transport("custom", 1)
        self.assertEqual(before, prometheus_client.generate_latest(registry))

    def test_copy_timing_and_logical_view_bytes(self):
        """Pool waiting and observer overhead must not inflate host-copy time."""
        observations = []
        processor = _processor(lambda *args: observations.append(args))
        tensor = torch.arange(40, dtype=torch.float32)[::4].as_subclass(_CudaTensor)
        clock = [10.0]

        def wrap(*args, **kwargs):
            clock[0] += 7.0
            return None

        def copy(tensor):
            clock[0] += 0.25
            return tensor.as_subclass(torch.Tensor)

        processor.cudaipc_mmfeature_pool.wrap_tensor.side_effect = wrap
        with (
            patch(
                "time.perf_counter",
                side_effect=lambda: clock[0],
            ),
            patch.object(_CudaTensor, "cpu", copy),
        ):
            result = processor._wrap_tensor_for_cuda_ipc(tensor)
        self.assertTrue(torch.equal(result, tensor))
        self.assertFalse(result.is_cuda)
        self.assertEqual(observations, [("cpu_fallback", 40, 0.25)])

    def test_ipc_completion_cpu_bypass_and_failed_producers(self):
        collector, registry = _collector({})
        processor = _processor(collector.observe_mm_feature_transport)
        tensor = torch.ones(3).as_subclass(_CudaTensor)
        proxy = object()
        processor.cudaipc_mmfeature_pool.wrap_tensor.return_value = proxy
        self.assertIs(processor._wrap_tensor_for_cuda_ipc(tensor), proxy)
        cpu_tensor = torch.ones(3)
        self.assertIs(processor._wrap_tensor_for_cuda_ipc(cpu_tensor), cpu_tensor)
        processor.cudaipc_mmfeature_pool.wrap_tensor.side_effect = RuntimeError("wrap")
        with self.assertRaisesRegex(RuntimeError, "wrap"):
            processor._wrap_tensor_for_cuda_ipc(tensor)
        processor.cudaipc_mmfeature_pool.wrap_tensor.side_effect = None
        processor.cudaipc_mmfeature_pool.wrap_tensor.return_value = None
        with (
            patch.object(_CudaTensor, "cpu", side_effect=RuntimeError("copy")),
            self.assertRaisesRegex(RuntimeError, "copy"),
        ):
            processor._wrap_tensor_for_cuda_ipc(tensor)
        self.assertEqual(
            registry.get_sample_value(
                "sglang:mm_feature_transport_tensors_total", {"transport": "cuda_ipc"}
            ),
            1,
        )
        self.assertEqual(
            registry.get_sample_value(
                "sglang:mm_feature_transport_tensors_total",
                {"transport": "cpu_fallback"},
            ),
            0,
        )
        self.assertEqual(
            registry.get_sample_value("sglang:mm_feature_cpu_fallback_seconds_total"), 0
        )

    def test_disabled_observer_does_no_timing_or_size_work(self):
        processor = _processor()
        tensor = torch.ones(3).as_subclass(_CudaTensor)
        with (
            patch(
                "time.perf_counter",
                side_effect=AssertionError("clock"),
            ),
            patch.object(_CudaTensor, "numel", side_effect=AssertionError("size")),
        ):
            proxy = object()
            processor.cudaipc_mmfeature_pool.wrap_tensor.return_value = proxy
            self.assertIs(processor._wrap_tensor_for_cuda_ipc(tensor), proxy)
            processor.cudaipc_mmfeature_pool.wrap_tensor.return_value = None
            self.assertFalse(processor._wrap_tensor_for_cuda_ipc(tensor).is_cuda)

    def test_backend_failure_does_not_change_transport(self):
        processor = _processor(Mock(side_effect=RuntimeError("backend unavailable")))
        tensor = torch.ones(3).as_subclass(_CudaTensor)
        proxy = object()
        processor.cudaipc_mmfeature_pool.wrap_tensor.return_value = proxy
        self.assertIs(processor._wrap_tensor_for_cuda_ipc(tensor), proxy)
        processor.cudaipc_mmfeature_pool.wrap_tensor.return_value = None
        self.assertFalse(processor._wrap_tensor_for_cuda_ipc(tensor).is_cuda)

    def test_cancel_during_observation_releases_current_and_prior_proxies(self):
        """An interrupted callback must not strand a proxy before batch tracking."""
        processor = _processor(Mock(side_effect=[None, asyncio.CancelledError()]))
        tensors = [torch.ones(3).as_subclass(_CudaTensor) for _ in range(2)]
        proxies = [object.__new__(CudaIpcTensorTransportProxy) for _ in range(2)]
        processor.cudaipc_mmfeature_pool.wrap_tensor.side_effect = proxies
        items = [
            MultimodalDataItem(modality=Modality.IMAGE, feature=t) for t in tensors
        ]
        with self.assertRaises(asyncio.CancelledError):
            processor._prepare_mm_items_for_transport(items)
        self.assertTrue(all(item.feature is t for item, t in zip(items, tensors)))
        self.assertEqual(
            [
                call.args[0]
                for call in processor.cudaipc_mmfeature_pool.cancel_proxy.call_args_list
            ],
            list(reversed(proxies)),
        )

    def test_manager_attaches_only_enabled_optional_hook(self):
        for enabled, has_hook in ((False, True), (True, False), (True, True)):
            with self.subTest(enabled=enabled, has_hook=has_hook):
                observer = Mock()
                collector = SimpleNamespace()
                if has_hook:
                    collector.observe_mm_feature_transport = observer
                processor = _processor()
                manager = SimpleNamespace(
                    enable_metrics=enabled,
                    enable_priority_scheduling=False,
                    mm_processor=processor,
                    server_args=None,
                )
                with (
                    get_context().override_server_args(gc_warning_threshold_secs=0),
                    patch(
                        "sglang.srt.managers.tokenizer_manager.resolve_collector_class",
                        return_value=lambda **kwargs: collector,
                    ),
                    patch(
                        "sglang.srt.managers.tokenizer_manager.start_cpu_monitor_thread"
                    ),
                    patch("sglang.srt.managers.tokenizer_manager.Watchdog.create"),
                ):
                    TokenizerManager.init_metric_collector_watchdog(manager)
                self.assertIs(
                    processor.mm_feature_transport_observer,
                    observer if enabled and has_hook else None,
                )


if __name__ == "__main__":
    unittest.main()
