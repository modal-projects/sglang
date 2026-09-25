"""Host occupancy regressions using real CPU allocators and metric export."""

import threading
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Summary

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, SidecarPoolSpec
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollector,
    SchedulerStats,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _allocator(pool_type, *, size, dcp_size=1):
    # Exercise production free lists without allocating the backing KV tensors.
    pool = object.__new__(pool_type)
    pool.size = size
    pool.dcp_size = dcp_size
    pool.page_size = 1
    pool.device = "cpu"
    pool.layout = "layer_first"
    pool.lock = threading.RLock()
    pool.can_use_write_back_jit = False
    pool.clear()
    return pool


def _entry(name, pool):
    return PoolEntry(
        name=name,
        host_pool=pool,
        device_pool=None,
        layer_mapper=lambda _: 0,
        is_primary_index_anchor=name == PoolName.KV,
    )


class _SidecarPool(LogicalHostPool):
    def available_size(self):
        raise AssertionError("Sidecar free lists do not measure occupancy")


class _UnavailablePool(LogicalHostPool):
    def available_size(self):
        raise NotImplementedError("No independent allocator")


class TestHiCachePoolStats(CustomTestCase):
    def setUp(self):
        self._init_collector()
        self.reporter = object.__new__(SchedulerMetricsReporter)
        self.reporter.stats = SchedulerStats()
        self.reporter.scheduler = SimpleNamespace(enable_hierarchical_cache=True)

    def _init_collector(self, extra_metric_labels=None):
        override = get_context().override_server_args(
            enable_metrics=True, extra_metric_labels=extra_metric_labels
        )
        server_args = override.install()
        self.addCleanup(override.restore)
        self.registry = CollectorRegistry()

        for metric in (Counter, Gauge, Histogram, Summary):
            self.enterContext(
                patch(
                    "prometheus_client." + metric.__name__,
                    partial(metric, registry=self.registry),
                )
            )

        context = SchedulerMetricsCollector.init_new(
            server_args=server_args,
            ps=ParallelState.trivial(),
            tp_rank=0,
            pp_rank=0,
            dp_rank=0,
            enable_priority_scheduling=False,
            enable_lora=False,
            enable_hierarchical_cache=True,
        )
        self.collector = context.collector
        self.labels = {key: str(value) for key, value in self.collector.labels.items()}

    def _attach(self, entries, specs=()):
        group = HostPoolGroup(entries)
        cache = object.__new__(UnifiedRadixCache)
        cache.host_pool_group = group
        cache.sidecar_pool_specs = list(specs)
        cache.full_kv_pool_host = group.anchor_entry.host_pool
        self.reporter.scheduler.tree_cache = cache
        return group

    def _scrape(self):
        self.reporter._log_hicache_stats()
        self.collector.log_stats(self.reporter.stats)

    def _value(self, metric, pool, source=""):
        return self.registry.get_sample_value(
            "sglang:hicache_host_pool_" + metric + "_slots",
            dict(self.labels, pool=pool, indices_from_pool=source),
        )

    def test_draft_tracks_dcp_logical_allocation_and_release(self):
        """Draft usage follows target logical slots even while its free list is unused."""
        kv = _allocator(MHATokenToKVPoolHost, size=8, dcp_size=2)
        self._attach(
            [_entry(PoolName.DRAFT, _SidecarPool(16, 1)), _entry(PoolName.KV, kv)],
            [SidecarPoolSpec(pool_name=PoolName.DRAFT, indices_from_pool=PoolName.KV)],
        )
        slots = kv.alloc(10)
        self._scrape()
        self.assertEqual(self._value("used", "kv"), 10)
        self.assertEqual(self._value("used", "draft", "kv"), 10)
        self.assertEqual(self._value("total", "kv"), 16)
        self.assertEqual(self._value("total", "draft", "kv"), 16)
        self.assertIsNone(self._value("total", "draft"))
        kv.free(slots)
        self._scrape()
        self.assertEqual(self._value("used", "kv"), 0)
        self.assertEqual(self._value("used", "draft", "kv"), 0)
        self.assertEqual(kv.num_release_slots, 10)

    def test_mamba_released_checkpoints_are_available_before_free_list_merge(self):
        kv = LogicalHostPool(16, 1)
        mamba = _allocator(MambaPoolHost, size=8)
        self._attach([_entry(PoolName.KV, kv), _entry(PoolName.MAMBA, mamba)])
        slots = mamba.alloc(6)
        mamba.free(slots[:4])
        self._scrape()
        self.assertEqual(self._value("used", "mamba"), 2)
        self.assertEqual(self._value("total", "mamba"), 8)
        self.assertEqual(len(mamba.free_slots), 2)
        self.assertEqual(mamba.num_release_slots, 4)

    def test_sidecar_selects_its_declared_source(self):
        kv = LogicalHostPool(16, 1)
        swa = LogicalHostPool(8, 1)
        self._attach(
            [
                _entry(PoolName.KV, kv),
                _entry(PoolName.SWA, swa),
                _entry(PoolName.DRAFT_SWA, _SidecarPool(8, 1)),
            ],
            [
                SidecarPoolSpec(
                    pool_name=PoolName.DRAFT_SWA, indices_from_pool=PoolName.SWA
                )
            ],
        )
        kv.alloc(12)
        swa.alloc(3)
        self._scrape()
        self.assertEqual(self._value("used", "draft_swa", "swa"), 3)
        self.assertEqual(self._value("total", "draft_swa", "swa"), 8)
        self.assertIsNone(self._value("used", "draft_swa", "kv"))

    def test_unavailable_source_and_absent_sidecars_have_no_series(self):
        self._attach(
            [
                _entry(PoolName.KV, LogicalHostPool(16, 1)),
                _entry(PoolName.SWA, _UnavailablePool(8, 1)),
                _entry(PoolName.DRAFT_SWA, _SidecarPool(8, 1)),
                _entry(PoolName.DRAFT_INDEXER, _SidecarPool(8, 1)),
            ],
            [
                SidecarPoolSpec(
                    pool_name=PoolName.DRAFT, indices_from_pool=PoolName.KV
                ),
                SidecarPoolSpec(
                    pool_name=PoolName.DRAFT_SWA, indices_from_pool=PoolName.SWA
                ),
                SidecarPoolSpec(
                    pool_name=PoolName.DRAFT_INDEXER, indices_from_pool=PoolName.INDEXER
                ),
            ],
        )
        self._scrape()
        samples = [
            sample
            for metric in self.registry.collect()
            if metric.name.startswith("sglang:hicache_host_pool_")
            for sample in metric.samples
        ]
        self.assertEqual(len(samples), 2)
        self.assertEqual({sample.labels["pool"] for sample in samples}, {"kv"})

    def test_allocator_errors_propagate(self):
        class BrokenPool(LogicalHostPool):
            def available_size(self):
                raise RuntimeError("Allocator failure")

        self._attach(
            [
                _entry(PoolName.KV, LogicalHostPool(16, 1)),
                _entry(PoolName.SWA, BrokenPool(8, 1)),
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "Allocator failure"):
            self.reporter._log_hicache_stats()

    def test_extra_labels_do_not_shadow_intrinsic_pool_labels(self):
        """Custom labels must not duplicate gauge dimensions or change pool identity."""
        self._init_collector(
            extra_metric_labels={
                "pool": "custom-pool",
                "indices_from_pool": "custom-source",
            }
        )
        kv = LogicalHostPool(16, 1)
        self._attach(
            [_entry(PoolName.KV, kv), _entry(PoolName.DRAFT, _SidecarPool(16, 1))],
            [SidecarPoolSpec(pool_name=PoolName.DRAFT, indices_from_pool=PoolName.KV)],
        )
        kv.alloc(4)
        self._scrape()
        for pool, source in (("kv", ""), ("draft", "kv")):
            self.assertEqual(self._value("used", pool, source), 4)
            self.assertEqual(self._value("total", pool, source), 16)
        self.assertEqual(
            self.registry.get_sample_value(
                "sglang:hicache_host_used_tokens", self.labels
            ),
            4,
        )
        self.assertEqual(self.labels["pool"], "custom-pool")
        self.assertEqual(self.labels["indices_from_pool"], "custom-source")

    def test_idle_reports_initial_capacity_and_released_slots(self):
        """Idle reports must expose initial capacity and refresh post-batch releases."""
        kv = LogicalHostPool(16, 1)
        self._attach(
            [_entry(PoolName.KV, kv), _entry(PoolName.DRAFT, _SidecarPool(16, 1))],
            [SidecarPoolSpec(pool_name=PoolName.DRAFT, indices_from_pool=PoolName.KV)],
        )
        scheduler = self.reporter.scheduler
        scheduler.running_batch = SimpleNamespace(reqs=[])
        scheduler.waiting_queue = []
        scheduler.grammar_manager = []
        scheduler.enable_priority_scheduling = False
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.pool_stats_observer = SimpleNamespace(
            get_pool_stats=lambda: SimpleNamespace(
                update_scheduler_stats=lambda _: None
            ),
            streaming_session_count=lambda: 0,
            session_held_tokens=lambda: 0,
        )
        self.reporter.current_scheduler_metrics_enabled = True
        self.reporter.metrics_collector = self.collector
        self.collector.last_log_time = 0
        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.time.perf_counter",
            return_value=31.0,
        ):
            self.reporter._maybe_log_idle_metrics()
        self.assertEqual(self._value("total", "kv"), 16)
        self.assertEqual(self._value("total", "draft", "kv"), 16)
        self.assertEqual(self._value("used", "draft", "kv"), 0)

        slots = kv.alloc(4)
        self._scrape()
        self.assertEqual(self._value("used", "draft", "kv"), 4)
        kv.free(slots)
        self.collector.last_log_time = 0
        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.time.perf_counter",
            return_value=31.0,
        ):
            self.reporter._maybe_log_idle_metrics()

        self.assertEqual(self._value("used", "kv"), 0)
        self.assertEqual(self._value("used", "draft", "kv"), 0)
        self.assertEqual(
            self.registry.get_sample_value(
                "sglang:hicache_host_used_tokens", self.labels
            ),
            0,
        )

    def test_legacy_cache_keeps_only_aggregate_metrics(self):
        kv = LogicalHostPool(16, 1)
        kv.alloc(4)
        self.reporter.scheduler.tree_cache = SimpleNamespace(token_to_kv_pool_host=kv)
        self._scrape()
        self.assertEqual(
            self.registry.get_sample_value(
                "sglang:hicache_host_used_tokens", self.labels
            ),
            4,
        )
        self.assertIsNone(self._value("used", "kv"))


if __name__ == "__main__":
    unittest.main()
