"""Capacity producers must retain their limiting bound through startup export."""

import unittest
from functools import partial
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Summary

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components import kv_pool_runtime
from sglang.srt.model_executor.pool_configurator import (
    MemoryPoolConfig,
    MemoryPoolConfigurator,
)
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestEffectiveCapacityMetrics(CustomTestCase):
    def setUp(self):
        self.context = get_context().override_server_args(
            enable_metrics=True,
            max_queued_requests=None,
            disable_radix_cache=True,
            served_model_name="synthetic",
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend=Backend.DISABLED)
            ),
        )
        self.server_args = self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)
        self.registry = CollectorRegistry()
        # GaugeHistogram constructs its Gauge directly instead of using collector DI.
        self.enterContext(
            patch("prometheus_client.Gauge", partial(Gauge, registry=self.registry))
        )

        class Collector(SchedulerMetricsCollector):
            _counter_cls = staticmethod(partial(Counter, registry=self.registry))
            _gauge_cls = staticmethod(partial(Gauge, registry=self.registry))
            _histogram_cls = staticmethod(partial(Histogram, registry=self.registry))
            _summary_cls = staticmethod(partial(Summary, registry=self.registry))

        self.collector_cls = Collector

    def _configurator(self, ps, *, mamba=False):
        configurator = KVCacheConfigurator.__new__(KVCacheConfigurator)
        configurator.ps = ps
        configurator.model_config = SimpleNamespace(context_len=8192)
        configurator.mambaish_config = object() if mamba else None
        return configurator

    def _resolve(self, configurator, tokens):
        config = MemoryPoolConfig(max_total_num_tokens=tokens)
        with (
            patch.object(
                KVCacheConfigurator, "_profile_available_bytes", return_value=0
            ),
            patch.object(
                KVCacheConfigurator, "config_from_budget", return_value=config
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.create_memory_pool_configurator",
                return_value=MemoryPoolConfigurator(),
            ),
        ):
            return configurator._resolve_memory_pool_config(0)

    def _scheduler(
        self,
        config,
        *,
        ps=None,
        dp_attention=False,
        labels=None,
        queue=None,
        collector=None,
    ):
        ps = ps or ParallelState.trivial()
        if collector is None:
            collector = self.collector_cls(
                labels={"model_name": "synthetic"} if labels is None else labels,
                server_args=self.server_args,
            )
        runner = SimpleNamespace(memory_pool_config=config, weight_load_mem_usage=1.0)
        scheduler = SimpleNamespace(
            metrics_collector=collector,
            max_total_num_tokens=config.max_total_num_tokens,
            max_running_requests=config.max_running_requests,
            max_queued_requests=queue,
            swa_tokens_per_layer=None,
            tp_worker=SimpleNamespace(model_runner=runner, graph_memory_usage={}),
            draft_worker=None,
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: SimpleNamespace(mem_usage=2.0)
            ),
            model_config=SimpleNamespace(context_len=8192),
            page_size=1,
            startup_available_gpu_memory_gb=3.0,
            device="cpu",
            ps=ps,
            enable_dp_attention=dp_attention,
        )
        return scheduler

    def _samples(self, name):
        return [
            sample
            for metric in self.registry.collect()
            for sample in metric.samples
            if sample.name == "sglang:" + name
        ]

    def test_resolved_caps_and_ties_reach_scrape(self):
        # A smaller cap must carry its own source, including after DP division.
        cases = (
            (120, 10000, None, 1, 120, "requested"),
            (None, 10000, None, 1, 2048, "estimated"),
            (120, 160, None, 1, 80, "kv_capacity"),
            (None, 160, None, 1, 80, "kv_capacity"),
            (120, 10000, 70, 1, 70, "mamba_pool"),
            (120, 240, 120, 1, 120, "requested"),
            (None, 4096, 2048, 1, 2048, "estimated"),
            (120, 160, 80, 1, 80, "kv_capacity"),
            (120, 10000, None, 3, 40, "requested"),
        )
        collector = self.collector_cls(labels={}, server_args=self.server_args)
        for requested, tokens, mamba, dp, expected, source in cases:
            with self.subTest(
                source=source, tokens=tokens, requested=requested, mamba=mamba
            ):
                with get_context().override_server_args(
                    enable_metrics=True,
                    max_running_requests=requested,
                    max_mamba_cache_size=mamba,
                    disable_radix_cache=True,
                ):
                    ps = ParallelState.trivial(attn_dp_size=dp)
                    config = self._resolve(
                        self._configurator(ps, mamba=mamba is not None), tokens
                    )
                    # Reuse the collector to verify only one metric family is registered.
                    scheduler = self._scheduler(
                        config, ps=ps, dp_attention=dp > 1, collector=collector
                    )
                    Scheduler.emit_metrics_constants(scheduler)
                    self.assertEqual(config.max_running_requests, expected)
                    self.assertEqual(config.max_running_requests_cap_source, source)
                    self.assertEqual(
                        self.registry.get_sample_value(
                            "sglang:max_running_requests", {"cap_source": source}
                        ),
                        expected,
                    )
        self.assertEqual(
            sum(
                m.name == "sglang:max_running_requests" for m in self.registry.collect()
            ),
            1,
        )
        self.assertFalse(self._samples("max_queued_requests"))

    def test_post_capture_resize_relays_changed_source_and_preserves_ties(self):
        # Post-capture resolution may select another bound at the same value;
        # the installed capacity and its source change only on strict reduction.
        for tokens, expected, source in (
            (80, 40, "kv_capacity"),
            (100, 50, "mamba_pool"),
        ):
            with self.subTest(tokens=tokens):
                with get_context().override_server_args(
                    enable_metrics=True,
                    max_running_requests=100,
                    max_mamba_cache_size=50,
                    disable_radix_cache=True,
                    cuda_graph_config=CudaGraphConfig(
                        decode=PhaseConfig(backend=Backend.DISABLED)
                    ),
                ):
                    configurator = self._configurator(
                        ParallelState.trivial(), mamba=True
                    )
                    initial = self._resolve(configurator, 200)
                    final = MemoryPoolConfig(max_total_num_tokens=tokens)
                    runner = SimpleNamespace(
                        memory_pool_config=initial,
                        token_to_kv_pool=Mock(
                            post_capture_backed_bytes=0, dtype="bfloat16"
                        ),
                        device="cpu",
                        gpu_id=0,
                        pre_model_load_memory=32,
                        mem_fraction_static=0.8,
                        max_running_requests=50,
                        max_total_num_tokens=200,
                        model_config=SimpleNamespace(is_multimodal=False),
                        canary_manager=None,
                        kv_cache_configurator=configurator,
                        token_to_kv_pool_allocator=Mock(),
                        req_to_token_pool=Mock(),
                        is_hybrid_swa=False,
                    )
                    with (
                        patch.object(
                            KVCacheConfigurator,
                            "config_from_budget",
                            return_value=final,
                        ),
                        patch.object(kv_pool_runtime.torch.cuda, "synchronize"),
                        patch.object(
                            kv_pool_runtime,
                            "get_world_group",
                            return_value=SimpleNamespace(world_size=1, cpu_group=None),
                        ),
                        patch.object(
                            kv_pool_runtime, "get_available_gpu_memory", return_value=20
                        ),
                        patch.object(
                            kv_pool_runtime, "mambaish_config", return_value=None
                        ),
                        patch.object(
                            kv_pool_runtime,
                            "graph_pool_borrow_enabled",
                            return_value=True,
                        ),
                        patch.object(
                            kv_pool_runtime, "mm_runtime_reservation_gb", return_value=0
                        ),
                    ):
                        ModelRunner.post_capture_resize_kv_pool(runner)
                    self.assertEqual(runner.max_running_requests, expected)
                    self.assertEqual(initial.max_running_requests_cap_source, source)
                    scheduler = self._scheduler(initial)
                    Scheduler.emit_metrics_constants(scheduler)
                    self.assertEqual(
                        self._samples("max_running_requests")[0].value, expected
                    )
                    self.assertEqual(
                        self._samples("max_running_requests")[0].labels["cap_source"],
                        source,
                    )
                    # Each startup gets a separate registry, as separate processes do.
                    for metric in list(self.registry._collector_to_names):
                        self.registry.unregister(metric)

    def test_configured_queue_and_topology(self):
        for tp, pp, dp, attention, expected in (
            (1, 1, 1, False, 1),
            (4, 2, 3, False, 24),
            (4, 2, 2, True, 8),
        ):
            with self.subTest(tp=tp, pp=pp, dp=dp, attention=attention):
                with get_context().override_server_args(
                    enable_metrics=True, max_queued_requests=0
                ):
                    scheduler = self._scheduler(
                        MemoryPoolConfig(
                            max_total_num_tokens=100,
                            max_running_requests=20,
                            max_running_requests_cap_source="requested",
                        ),
                        ps=ParallelState.trivial(tp_size=tp, pp_size=pp, dp_size=dp),
                        dp_attention=attention,
                        labels={},
                        queue=0,
                    )
                    Scheduler.emit_metrics_constants(scheduler)
                    self.assertEqual(
                        self._samples("configured_device_count")[0].value, expected
                    )
                    self.assertEqual(
                        self._samples("configured_device_count")[0].labels,
                        {"device_type": "cpu"},
                    )
                    self.assertEqual(self._samples("max_queued_requests")[0].value, 0)
                    for metric in list(self.registry._collector_to_names):
                        self.registry.unregister(metric)

    def test_disaggregated_queues_do_not_export_ordinary_queue_limit(self):
        # PD requests bypass the ordinary waiting-queue cap, so a configured
        # value must not advertise a limit on bootstrap or preallocation queues.
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            for metric in list(self.registry._collector_to_names):
                self.registry.unregister(metric)
            with (
                self.subTest(mode=mode),
                get_context().override_server_args(
                    enable_metrics=True,
                    max_queued_requests=1,
                    disaggregation_mode=mode.value,
                ),
            ):
                scheduler = self._scheduler(
                    MemoryPoolConfig(max_total_num_tokens=100, max_running_requests=20),
                    queue=1,
                    labels={},
                )
                scheduler.disaggregation_mode = mode
                scheduler.enable_priority_scheduling = False
                scheduler.abort_on_priority_when_disabled = False
                scheduler._set_or_validate_priority = MethodType(
                    Scheduler._set_or_validate_priority, scheduler
                )
                scheduler._prefetch_kvcache = Mock()
                scheduler.model_config.num_key_value_heads = 8
                queued = []
                queue = SimpleNamespace(
                    add=lambda req, *args, **kwargs: queued.append(req)
                )
                scheduler.disagg_prefill_bootstrap_queue = queue
                scheduler.disagg_decode_prealloc_queue = queue
                for _ in range(2):
                    Scheduler._add_request_to_queue(
                        scheduler, SimpleNamespace(priority=None, time_stats=Mock())
                    )
                self.assertEqual(len(queued), 2)
                Scheduler.emit_metrics_constants(scheduler)
                self.assertFalse(self._samples("max_queued_requests"))

    def test_plain_dp_identity_and_disabled_metrics(self):
        for rank in (0, 1):
            ps = ParallelState.trivial(dp_rank=rank, dp_size=2)
            context = self.collector_cls.init_new(
                server_args=self.server_args,
                ps=ps,
                tp_rank=0,
                pp_rank=0,
                dp_rank=rank,
                enable_priority_scheduling=False,
                enable_lora=False,
                enable_hierarchical_cache=False,
            )
            self.assertEqual(context.collector.labels["dp_rank"], rank)
            for metric in list(self.registry._collector_to_names):
                self.registry.unregister(metric)
        with get_context().override_server_args(enable_metrics=False):
            Scheduler.emit_metrics_constants(SimpleNamespace())
        self.assertFalse(list(self.registry.collect()))

    def test_custom_pool_without_capacity_source_is_unknown(self):
        scheduler = self._scheduler(
            MemoryPoolConfig(max_total_num_tokens=100, max_running_requests=20)
        )
        scheduler.tp_worker.model_runner.memory_pool_config = None
        Scheduler.emit_metrics_constants(scheduler)
        samples = self._samples("max_running_requests")
        self.assertEqual(
            [(s.labels["cap_source"], s.value) for s in samples], [("unknown", 20)]
        )

    def test_reserved_labels_fail_before_registration(self):
        for label in ("cap_source", "device_type"):
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(ValueError, "Reserved"),
            ):
                self.collector_cls(labels={label: "user"}, server_args=self.server_args)
        self.assertFalse(list(self.registry.collect()))


if __name__ == "__main__":
    unittest.main()
