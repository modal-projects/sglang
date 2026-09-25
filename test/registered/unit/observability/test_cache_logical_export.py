"""CPU contracts for rank-aware cache metrics and custom collector compatibility."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
from prometheus_client import CollectorRegistry, Counter, Histogram
from transformers import LlamaConfig

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.environ import envs
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv_cache_builder import build_kv_cache
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.metrics_collector import RadixCacheMetricsCollector
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _CPUAllocator:
    device = torch.device("cpu")


class _LabelsOnlyCollector:
    def __init__(self, labels):
        self.labels = labels


class _KwargsCollector:
    def __init__(self, **kwargs):
        self.options = kwargs


class _DisabledCollector:
    def __init__(self, **kwargs):
        self.emit_cache_metrics = False


class _LegacyBuiltinCollector(RadixCacheMetricsCollector):
    def __init__(self, labels):
        super().__init__(labels=labels)


class TestCacheLogicalExport(CustomTestCase):
    def setUp(self):
        self.server_args = self.enterContext(
            get_context().override_server_args(
                extra_metric_labels={"model": "test-model"},
                disaggregation_decode_retraction_backup="cpu_tensor",
            )
        )
        self.enterContext(
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python")
        )
        self.registry = CollectorRegistry()
        self.enterContext(
            patch.object(
                RadixCacheMetricsCollector,
                "_counter_cls",
                partial(Counter, registry=self.registry),
            )
        )
        self.enterContext(
            patch.object(
                RadixCacheMetricsCollector,
                "_histogram_cls",
                partial(Histogram, registry=self.registry),
            )
        )

    @staticmethod
    def _cache(**params):
        return UnifiedRadixCache(
            CacheInitParams(
                **{
                    "disable": False,
                    "req_to_token_pool": None,
                    "token_to_kv_pool_allocator": _CPUAllocator(),
                    "page_size": 2,
                    "tree_components": (ComponentType.FULL,),
                    "enable_metrics": True,
                    **params,
                }
            )
        )

    def test_tp_cp_gate_retains_physical_metrics_and_logical_dp_pp_labels(self):
        """Logical ownership must not suppress per-rank transfer observability."""
        for tp, cp, expected in ((0, 0, True), (1, 0, False), (0, 1, False)):
            registry = CollectorRegistry()
            with (
                self.subTest(tp=tp, cp=cp),
                patch.object(
                    RadixCacheMetricsCollector,
                    "_counter_cls",
                    partial(Counter, registry=registry),
                ),
                patch.object(
                    RadixCacheMetricsCollector,
                    "_histogram_cls",
                    partial(Histogram, registry=registry),
                ),
            ):
                cache = self._cache(
                    attn_tp_rank=tp, attn_cp_rank=cp, dp_rank=3, pp_rank=2
                )
                metrics = cache.metrics_collector
                self.assertEqual(cache.emit_logical_cache_metrics, expected)
                self.assertEqual(metrics.emit_cache_metrics, expected)
                self.assertEqual(
                    metrics.logical_labels,
                    {
                        "cache_type": "UnifiedRadixCache",
                        "model": "test-model",
                        "dp_rank": "3",
                        "pp_rank": "2",
                    },
                )
                metrics.increment_eviction_num_tokens(8)
                metrics.observe_eviction_duration(0.25)
                metrics.increment_load_back_num_tokens(4, "kv")
                metrics.observe_load_back_duration(0.1)
                self.assertEqual(
                    registry.get_sample_value(
                        "sglang:evicted_tokens_total", metrics.labels
                    ),
                    8,
                )
                self.assertEqual(
                    registry.get_sample_value(
                        "sglang:eviction_duration_seconds_sum", metrics.labels
                    ),
                    0.25,
                )
                self.assertEqual(
                    registry.get_sample_value(
                        "sglang:load_back_tokens_total",
                        {**metrics.labels, "pool": "kv"},
                    ),
                    4,
                )
                self.assertEqual(
                    registry.get_sample_value(
                        "sglang:load_back_duration_seconds_sum", metrics.labels
                    ),
                    0.1,
                )

    def test_custom_signatures_missing_flag_and_explicit_opt_out(self):
        """Older labels-only plugins remain usable; explicit false takes precedence."""
        for collector in (_LabelsOnlyCollector, _KwargsCollector, _DisabledCollector):
            for rank in (0, 1):
                with (
                    self.subTest(collector=collector.__name__, rank=rank),
                    patch(
                        "sglang.srt.mem_cache.base_prefix_cache.resolve_collector_class",
                        return_value=collector,
                    ),
                ):
                    cache = self._cache(attn_tp_rank=rank, dp_rank=3)
                    self.assertEqual(
                        cache.emit_logical_cache_metrics,
                        rank == 0 and collector is not _DisabledCollector,
                    )
                    if collector is _LabelsOnlyCollector:
                        self.assertEqual(
                            cache.metrics_collector.labels["model"], "test-model"
                        )
                    elif collector is _KwargsCollector:
                        options = cache.metrics_collector.options
                        self.assertEqual(options["emit_cache_metrics"], rank == 0)
                        self.assertEqual(options["logical_labels"]["dp_rank"], "3")

    def test_legacy_builtin_subclass_keeps_logical_identity(self):
        """A labels-only subclass still needs DP/PP identity for inherited metrics."""
        with patch(
            "sglang.srt.mem_cache.base_prefix_cache.resolve_collector_class",
            return_value=_LegacyBuiltinCollector,
        ):
            cache = self._cache(dp_rank=3, pp_rank=2)
        self.assertTrue(cache.emit_logical_cache_metrics)
        self.assertEqual(
            cache.metrics_collector.logical_labels,
            {
                "cache_type": "UnifiedRadixCache",
                "model": "test-model",
                "dp_rank": "3",
                "pp_rank": "2",
            },
        )

    def test_builder_preserves_plain_dp_and_attention_dp_identity(self):
        """Plain DP workers can share attention-DP rank zero without sharing identity."""
        config = SimpleNamespace(
            hf_config=LlamaConfig(),
            _resolved_model_impl=ModelImpl.SGLANG,
            linear_attn_registry_result=None,
            is_multimodal=False,
            is_draft_model=False,
        )
        allocator = _CPUAllocator()
        worker = SimpleNamespace(
            is_hybrid_swa=False,
            model_runner=SimpleNamespace(
                model_config=config, mtp_draft_device_pools=()
            ),
            get_memory_pool=lambda: (None, allocator),
        )
        for attention_dp, dp_rank, expected in (
            (False, 3, "3"),
            (True, 3, "1"),
            (False, None, "0"),
        ):
            with (
                self.subTest(attention_dp=attention_dp, dp_rank=dp_rank),
                get_context().override_server_args(
                    enable_dp_attention=attention_dp,
                    disaggregation_decode_retraction_backup="cpu_tensor",
                    extra_metric_labels={},
                ) as server_args,
                patch.object(
                    RadixCacheMetricsCollector,
                    "_counter_cls",
                    partial(Counter, registry=CollectorRegistry()),
                ),
                patch.object(
                    RadixCacheMetricsCollector,
                    "_histogram_cls",
                    partial(Histogram, registry=CollectorRegistry()),
                ),
                patch("sglang.srt.managers.mm_schedule.embedding_cache", None),
            ):
                result = build_kv_cache(
                    server_args=server_args,
                    model_config=config,
                    tp_worker=worker,
                    page_size=2,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    attn_tp_cpu_group=None,
                    tp_cpu_group=None,
                    attn_cp_cpu_group=None,
                    enable_metrics=True,
                    enable_kv_cache_events=False,
                    ps=SimpleNamespace(
                        dp_rank=dp_rank,
                        attn_dp_rank=1,
                        attn_tp_rank=0,
                        pp_rank=2,
                        pp_size=1,
                        attn_cp_rank=0,
                        attn_cp_size=1,
                        tp_rank=0,
                        tp_size=1,
                    ),
                    tp_group=None,
                    pp_group=SimpleNamespace(cpu_group=None),
                    enable_hierarchical_cache=False,
                )
                self.assertIsInstance(result.tree_cache, UnifiedRadixCache)
                self.assertEqual(
                    result.tree_cache.metrics_collector.logical_labels["dp_rank"],
                    expected,
                )
                self.assertEqual(
                    result.tree_cache.metrics_collector.logical_labels["pp_rank"], "2"
                )
                self.assertTrue(result.tree_cache.emit_logical_cache_metrics)


if __name__ == "__main__":
    unittest.main()
