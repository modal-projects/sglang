"""CPU producer-to-Prometheus coverage for Unified cache eviction causes."""

import unittest
from array import array
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from prometheus_client import CollectorRegistry, Counter, Histogram
from test_unified_radix_cache_unittest import CacheConfig, build_fixture

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import (
    C128SidecarComponent,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    EvictionFreeCause,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.metrics_collector import RadixCacheMetricsCollector
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

FULL = ComponentType.FULL
MAMBA = ComponentType.MAMBA
SWA = ComponentType.SWA
KV_COUNTER = "sglang:kv_evicted_tokens_by_cause_total"
MAMBA_COUNTER = "sglang:mamba_states_evicted_total"


def make_collector_class():
    registry = CollectorRegistry()

    class IsolatedCollector(RadixCacheMetricsCollector):
        _counter_cls = staticmethod(partial(Counter, registry=registry))
        _histogram_cls = staticmethod(partial(Histogram, registry=registry))

    return IsolatedCollector, registry


def make_collector(*, enabled=True):
    collector_cls, registry = make_collector_class()
    collector = collector_cls(
        {"model_name": "unit", "tp_rank": "0"},
        emit_cache_metrics=enabled,
        logical_labels={"model_name": "unit"},
    )
    return collector, registry


class TestUnifiedEvictionCauses(CustomTestCase):
    def make_cache(self, components=(FULL,), *, page_size=1):
        cfg = CacheConfig(
            page_size=page_size,
            components=components,
            num_layers=2,
            full_attention_layer_ids=(0,),
            sliding_window_size=4 if SWA in components else None,
            kv_size=64,
            mamba_cache_size=8,
        )
        with (
            patch("test_unified_radix_cache_unittest.get_device", return_value="cpu"),
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"),
        ):
            cache, _, _ = build_fixture(cfg)
        cache.metrics_collector, registry = make_collector()
        cache.emit_logical_cache_metrics = True
        return cache, registry

    def insert(self, cache, tokens):
        allocator = cache.token_to_kv_pool_allocator
        if SWA in cache.tree_components:
            full = allocator.full_attn_allocator.alloc(len(tokens))
            swa = allocator.swa_attn_allocator.alloc(len(tokens))
            allocator.full_to_swa_index_mapping[full] = swa
            value = full
        else:
            value = allocator.alloc(len(tokens))
        self.assertIsNotNone(value)
        params = InsertParams(key=RadixKey(array("q", tokens)), value=value)
        if MAMBA in cache.tree_components:
            params.mamba_value = cache.req_to_token_pool.mamba_allocator.alloc(1)
            self.assertIsNotNone(params.mamba_value)
        return cache.insert(params).last_device_node

    def causes(self, registry):
        result = {}
        for family in registry.collect():
            for sample in family.samples:
                if sample.name not in (KV_COUNTER, MAMBA_COUNTER):
                    continue
                labels = sample.labels
                self.assertEqual(labels["model_name"], "unit")
                self.assertNotIn("tp_rank", labels)
                component = labels.get("component", "mamba")
                result[component, labels["tier"], labels["cause"]] = sample.value
        return result

    def test_full_pressure_counts_actual_leaf_and_collateral_state(self):
        cache, registry = self.make_cache((FULL, MAMBA))
        self.insert(cache, [1, 2, 3, 4])
        available = cache.token_to_kv_pool_allocator.available_size()
        result = cache.evict(EvictParams(num_tokens=1))
        self.assertEqual(result.num_tokens_evicted, 4)
        self.assertEqual(result.mamba_num_evicted, 1)
        self.assertEqual(
            cache.token_to_kv_pool_allocator.available_size() - available, 4
        )
        self.assertEqual(
            self.causes(registry),
            {
                ("full", "device", "full_pressure"): 4,
                ("mamba", "device", "full_pressure"): 1,
            },
        )
        self.assertEqual(
            registry.get_sample_value(
                "sglang:evicted_tokens_total", {"model_name": "unit", "tp_rank": "0"}
            ),
            4,
        )

    def test_mamba_pressure_tombstones_interior_without_freeing_full(self):
        cache, registry = self.make_cache((FULL, MAMBA))
        interior = self.insert(cache, [1, 2])
        self.insert(cache, [1, 2, 3, 4])
        result = cache.evict(EvictParams(mamba_num=1))
        self.assertEqual(result.num_tokens_evicted, 0)
        self.assertEqual(result.mamba_num_evicted, 1)
        self.assertIsNotNone(cache.tree_core.get_component_device_value(interior, FULL))
        self.assertIsNone(cache.tree_core.get_component_device_value(interior, MAMBA))
        self.assertEqual(
            self.causes(registry), {("mamba", "device", "mamba_pressure"): 1}
        )

    def test_swa_pressure_leaf_cascade_uses_one_trigger_cause(self):
        cache, registry = self.make_cache((FULL, SWA), page_size=4)
        self.insert(cache, [1, 2, 3, 4])
        result = cache.evict(EvictParams(swa_num_tokens=1))
        self.assertEqual(
            (result.num_tokens_evicted, result.swa_num_tokens_evicted), (4, 4)
        )
        self.assertEqual(
            self.causes(registry),
            {
                ("full", "device", "swa_pressure"): 4,
                ("swa", "device", "swa_pressure"): 4,
            },
        )

    def test_request_owned_swa_ring_is_not_counted_as_freed_cache(self):
        cache, registry = self.make_cache((FULL, SWA), page_size=4)
        allocator = cache.token_to_kv_pool_allocator
        allocator._swa_req_ring = True
        value = allocator.full_attn_allocator.alloc(4)
        cache.insert(InsertParams(key=RadixKey(array("q", [1, 2, 3, 4])), value=value))
        available = allocator.swa_attn_allocator.available_size()
        result = cache.evict(EvictParams(num_tokens=1))
        self.assertEqual(result.num_tokens_evicted, 4)
        self.assertEqual(allocator.swa_attn_allocator.available_size(), available)
        self.assertEqual(
            self.causes(registry), {("full", "device", "full_pressure"): 4}
        )

    def attach_host_pool(self, cache):
        released = []
        cache.components[FULL]._full_kv_pool_host = object()
        cache.host_pool_group = SimpleNamespace(
            free=lambda values, **kwargs: released.extend(values.tolist())
        )
        return released

    def test_device_eviction_keeps_host_copy_then_host_pressure_frees_it(self):
        cache, registry = self.make_cache()
        node_id = self.insert(cache, [1, 2, 3, 4])
        node = cache.tree_core.node_by_id(node_id)
        node.component_data[FULL].host_value = torch.arange(100, 104)
        cache.tree_core._update_evictable_leaf_sets(node)
        released = self.attach_host_pool(cache)
        cache.evict(EvictParams(num_tokens=1))
        self.assertEqual(released, [])
        self.assertIsNotNone(node.component_data[FULL].host_value)
        self.assertEqual(
            self.causes(registry), {("full", "device", "full_pressure"): 4}
        )
        self.assertEqual(cache.evict_host(1), 4)
        self.assertEqual(released, [100, 101, 102, 103])
        self.assertEqual(
            self.causes(registry),
            {
                ("full", "device", "full_pressure"): 4,
                ("full", "host", "host_pressure"): 4,
            },
        )

    def test_failed_write_back_counts_actual_drops_as_host_pressure(self):
        cache, registry = self.make_cache((FULL, MAMBA))
        self.insert(cache, [1, 2, 3, 4])
        cache.is_write_back = True
        with patch.object(cache, "_execute_and_commit_kv_backup", return_value=0):
            result = cache.evict(EvictParams(num_tokens=1))
        self.assertEqual(result.num_tokens_evicted, 4)
        self.assertEqual(result.mamba_num_evicted, 1)
        self.assertEqual(
            self.causes(registry),
            {
                ("full", "device", "host_pressure"): 4,
                ("mamba", "device", "host_pressure"): 1,
            },
        )

    def test_absent_host_pool_does_not_report_successful_free(self):
        cache, registry = self.make_cache()
        cache._free_values({}, {FULL: [torch.arange(4)]}, cause="host_pressure")
        self.assertEqual(self.causes(registry), {})

    def test_full_host_leaf_drains_c128_and_remaining_mamba_values(self):
        c128 = ComponentType.C128
        with patch.dict(
            "sglang.srt.mem_cache.unified_radix_cache.COMPONENT_REGISTRY",
            {c128: C128SidecarComponent},
        ):
            cache, registry = self.make_cache((FULL, c128, MAMBA))
        released = []
        expected_releases = [
            (PoolName.KV, [100, 101, 102, 103]),
            (PoolName.DEEPSEEK_V4_C128, [200, 201]),
            (PoolName.MAMBA, [300]),
        ]
        # Check release completion even if metric observation raises midway.
        self.addCleanup(self.assertEqual, released, expected_releases)
        cache.host_pool_group = SimpleNamespace(
            free=lambda values, *, pool: released.append((pool, values.tolist()))
        )
        cache.components[FULL]._full_kv_pool_host = object()
        cache.components[MAMBA]._mamba_pool_host = object()
        cache.components[c128]._c128_kv_pool_host = SimpleNamespace(
            free=lambda values: released.append(
                (PoolName.DEEPSEEK_V4_C128, values.tolist())
            )
        )
        core = cache.tree_core
        inserted = core.insert_host(
            cache.root_node_handle(),
            RadixKey(array("q", [1, 2, 3, 4])),
            torch.arange(100, 104),
            ["h1", "h2", "h3", "h4"],
        )
        node = core.node_by_id(inserted.inserted_host_node)
        for component, values in (
            (c128, torch.arange(200, 202)),
            (MAMBA, torch.tensor([300])),
        ):
            node.component_data[component].host_value = values
            core.host_lru_lists[component].insert_mru(node)
        self.assertTrue(core.is_host_evictable_leaf(node.id))
        self.assertEqual(cache.evict_host(1), 4)
        self.assertEqual(
            self.causes(registry),
            {
                ("full", "host", "host_pressure"): 4,
                ("mamba", "host", "host_pressure"): 1,
            },
        )
        self.assertEqual(core.root_node.children, {})

    def test_legacy_void_host_free_finishes_all_values_without_metrics(self):
        class LegacyFullComponent(FullComponent):
            def free_host_values(self, host_values):
                super().free_host_values(host_values)

        cache, registry = self.make_cache()
        cache.components[FULL] = LegacyFullComponent(cache, cache.cache_init_params)
        released = self.attach_host_pool(cache)
        self.addCleanup(self.assertEqual, released, [100, 101, 200, 201, 202])
        cache._free_values(
            {},
            {FULL: [torch.arange(100, 102), torch.arange(200, 203)]},
            cause="host_pressure",
        )
        self.assertEqual(self.causes(registry), {})

    def test_mamba_path_cap_preserves_full_and_counts_only_released_state(self):
        cache, registry = self.make_cache((FULL, MAMBA))
        cache.components[MAMBA].mamba_max_states_per_path = 1
        interior = self.insert(cache, [1, 2])
        self.insert(cache, [1, 2, 3, 4])
        self.assertIsNotNone(cache.tree_core.get_component_device_value(interior, FULL))
        self.assertIsNone(cache.tree_core.get_component_device_value(interior, MAMBA))
        self.assertEqual(
            self.causes(registry), {("mamba", "device", "mamba_path_cap"): 1}
        )

    def test_mamba_donor_uses_real_full_eviction_walk(self):
        cache, registry = self.make_cache((FULL, MAMBA))
        # Spare Mamba IDs exist, but byte capacity requires Full cache donation.
        self.insert(cache, [1, 2, 3, 4])
        available = cache.token_to_kv_pool_allocator.available_size()
        original_capacity = cache._component_available_size
        donor = Mock()
        donor.full_tokens_before_mamba_recheck.return_value = 1

        def capacity(component):
            if component == MAMBA:
                return int(
                    cache.token_to_kv_pool_allocator.available_size() > available
                )
            return original_capacity(component)

        with (
            patch.object(cache, "_component_available_size", side_effect=capacity),
            patch.object(
                cache.token_to_kv_pool_allocator,
                "mamba_full_cache_donor",
                return_value=donor,
            ),
        ):
            result = cache.evict_for_alloc(EvictParams(mamba_num=1))
        self.assertEqual(result.num_tokens_evicted, 4)
        self.assertEqual(result.mamba_num_evicted, 1)
        donor.prepare_mamba_allocation.assert_called()
        self.assertEqual(
            self.causes(registry),
            {
                ("full", "device", "mamba_donor"): 4,
                ("mamba", "device", "mamba_donor"): 1,
            },
        )

    def test_locked_noop_and_duplicate_insert_do_not_count_evictions(self):
        cache, registry = self.make_cache()
        node = self.insert(cache, [1, 2, 3, 4])
        self.insert(cache, [1, 2, 3, 4])
        receipt = cache.inc_lock_ref(node)
        result = cache.evict(EvictParams(num_tokens=100))
        self.assertEqual(result.num_tokens_evicted, 0)
        cache.dec_lock_ref(node, receipt.to_dec_params())
        self.assertEqual(cache.evict(EvictParams()).num_tokens_evicted, 0)
        self.assertEqual(self.causes(registry), {})

    def test_request_finish_free_is_excluded_from_eviction_counters(self):
        cache, registry = self.make_cache()
        req = Req(
            rid="finished",
            origin_input_text="",
            origin_input_ids=array("q", [1, 2]),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        cache.req_to_token_pool.alloc([req])
        allocator = cache.token_to_kv_pool_allocator
        indices = allocator.alloc(2)
        cache.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :2] = indices
        available = allocator.available_size()
        cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=2)
        self.assertEqual(allocator.available_size() - available, 2)
        self.assertEqual(self.causes(registry), {})

    def test_unattributed_free_uses_other(self):
        cache, registry = self.make_cache()
        indices = cache.token_to_kv_pool_allocator.alloc(3)
        cache._free_values({FULL: [indices]}, {})
        self.assertEqual(self.causes(registry), {("full", "device", "other"): 3})

    def test_partial_free_failure_counts_successes_and_drains_host(self):
        cache, registry = self.make_cache()
        allocator = cache.token_to_kv_pool_allocator
        first, second = allocator.alloc(2), allocator.alloc(3)
        released = self.attach_host_pool(cache)
        real_free = allocator.free_segment

        def fail_second(indices, **kwargs):
            if torch.equal(indices, second):
                raise RuntimeError("free failed")
            real_free(indices, **kwargs)

        with patch.object(allocator, "free_segment", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "free failed"):
                cache._free_values(
                    {FULL: [first, second]},
                    {FULL: [torch.arange(100, 104)]},
                    cause="full_pressure",
                )
        self.assertEqual(released, [100, 101, 102, 103])
        self.assertEqual(
            self.causes(registry),
            {
                ("full", "device", "full_pressure"): 2,
                ("full", "host", "full_pressure"): 4,
            },
        )

    def test_per_free_override_preserves_totals_and_separates_tiers(self):
        cache, registry = self.make_cache()
        allocator = cache.token_to_kv_pool_allocator
        device_values = [allocator.alloc(2), allocator.alloc(3)]
        released = self.attach_host_pool(cache)
        available = allocator.available_size()
        cache._free_values(
            {FULL: device_values},
            {FULL: [torch.arange(100, 104), torch.arange(200, 206)]},
            cause="full_pressure",
            free_causes=[
                EvictionFreeCause(FULL, "device", 1),
                EvictionFreeCause(FULL, "host", 0),
            ],
        )
        self.assertEqual(allocator.available_size() - available, 5)
        self.assertEqual(len(released), 10)
        causes = self.causes(registry)
        self.assertEqual(
            causes,
            {
                ("full", "device", "full_pressure"): 2,
                ("full", "device", "other"): 3,
                ("full", "host", "other"): 4,
                ("full", "host", "full_pressure"): 6,
            },
        )
        self.assertEqual(sum(causes.values()), 15)

    def test_disabled_exporter_still_frees_without_logical_samples(self):
        cache, _ = self.make_cache()
        cache.metrics_collector, registry = make_collector(enabled=False)
        cache.emit_logical_cache_metrics = False
        self.insert(cache, [1, 2])
        self.assertEqual(cache.evict(EvictParams(num_tokens=1)).num_tokens_evicted, 2)
        self.assertEqual(self.causes(registry), {})

    def test_logical_exporter_configures_legacy_subclasses_before_registration(self):
        self.enterContext(get_context().override_server_args(extra_metric_labels={}))
        for legacy in (False, True):
            for tp, cp, dp in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
                with self.subTest(legacy=legacy, tp=tp, cp=cp, dp=dp):
                    collector_cls, registry = make_collector_class()
                    if legacy:

                        class LabelsOnlyCollector(collector_cls):
                            def __init__(self, labels):
                                super().__init__(labels)

                        collector_cls = LabelsOnlyCollector
                    cache = object.__new__(UnifiedRadixCache)
                    params = CacheInitParams(
                        disable=False,
                        req_to_token_pool=None,
                        token_to_kv_pool_allocator=None,
                        page_size=1,
                        attn_tp_rank=tp,
                        attn_cp_rank=cp,
                        dp_rank=dp,
                        pp_rank=2,
                    )
                    with patch(
                        "sglang.srt.mem_cache.base_prefix_cache.resolve_collector_class",
                        return_value=collector_cls,
                    ):
                        cache.init_metrics_collector(params)
                    collector = cache.metrics_collector
                    collector.increment_eviction_cause(
                        3, "full", "device", "full_pressure"
                    )
                    collector.increment_eviction_num_tokens(3)
                    expected = 3 if tp == 0 and cp == 0 else None
                    samples = [
                        sample.value
                        for family in registry.collect()
                        for sample in family.samples
                        if sample.name == KV_COUNTER
                    ]
                    self.assertEqual(samples, [] if expected is None else [3])
                    self.assertEqual(
                        registry.get_sample_value(
                            KV_COUNTER,
                            {
                                "cache_type": "UnifiedRadixCache",
                                "dp_rank": str(dp),
                                "pp_rank": "2",
                                "component": "full",
                                "tier": "device",
                                "cause": "full_pressure",
                            },
                        ),
                        expected,
                    )
                    self.assertEqual(
                        registry.get_sample_value(
                            "sglang:evicted_tokens_total",
                            {"cache_type": "UnifiedRadixCache"},
                        ),
                        3,
                    )

    def test_consumer_normalizes_unknown_cause_and_keeps_units_separate(self):
        collector, registry = make_collector()
        collector.increment_eviction_cause(7, "full", "device", "unexpected-detail")
        collector.increment_eviction_cause(2, "mamba", "host", "host_pressure")
        collector.increment_eviction_cause(0, "swa", "device", "swa_pressure")
        self.assertEqual(
            self.causes(registry),
            {("full", "device", "other"): 7, ("mamba", "host", "host_pressure"): 2},
        )


if __name__ == "__main__":
    unittest.main()
