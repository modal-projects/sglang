"""CPU producer/consumer contracts for bounded KV eviction history.

The real cache, tree core, Full component and Prometheus collector run together.
Only KV storage allocation and the monotonic clock are external fixtures.
"""

import sys
import unittest
from array import array
from functools import partial
from unittest.mock import patch

import torch
from prometheus_client import CollectorRegistry, Counter, Histogram

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import EvictLayer
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.metrics_collector import RadixCacheMetricsCollector
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _CPUAllocator:
    device = torch.device("cpu")

    def __init__(self):
        self.freed = []

    def free(self, indices, *, pool=None):
        self.freed.extend(indices.tolist())

    def free_segment(self, indices, *, start_pos):
        self.free(indices)


class _CPUSWAAllocator(SWATokenToKVPoolAllocator):
    def __init__(self):
        self.device = torch.device("cpu")
        self.full_attn_allocator = _CPUAllocator()
        self.swa_attn_allocator = _CPUAllocator()

    def translate_loc_from_full_to_swa(self, indices):
        return indices

    def free_segment(self, indices, *, start_pos):
        self.full_attn_allocator.free(indices)
        self.swa_attn_allocator.free(indices)

    def free_full_segment(self, indices, *, start_pos):
        self.full_attn_allocator.free(indices)

    def free_swa_segment(self, indices, *, start_pos):
        self.swa_attn_allocator.free(indices)


class _LabelsOnlyCollector:
    def __init__(self, labels):
        self.labels = labels


class _KwargsCollector:
    def __init__(self, **kwargs):
        self.options = kwargs


class _LegacyBuiltinCollector(RadixCacheMetricsCollector):
    def __init__(self, labels):
        super().__init__(labels=labels)


class _DisabledCollector(RadixCacheMetricsCollector):
    def __init__(self, **kwargs):
        super().__init__(**{**kwargs, "emit_cache_metrics": False})


class TestKVGhostMetrics(CustomTestCase):
    def setUp(self):
        self.enterContext(get_context().override_server_args(extra_metric_labels={}))
        self.enterContext(
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python")
        )
        self.enterContext(envs.SGLANG_KV_GHOST_CAPACITY_PAGES.override(32))
        self.enterContext(envs.SGLANG_KV_GHOST_TTL_SECONDS.override(300))
        self.clock = self.enterContext(
            patch("sglang.srt.mem_cache.kv_ghost.time.monotonic", return_value=10.0)
        )

    def _cache(self, **params):
        registry = CollectorRegistry()
        params = CacheInitParams(
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
        with (
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
            cache = UnifiedRadixCache(params)
        cache.host_pool_group = _CPUAllocator()
        return cache, registry

    @staticmethod
    def _key(tokens, *, salt=None, extra=None, bigram=False):
        return RadixKey(
            array("q", tokens), extra_key=extra, cache_salt=salt, is_bigram=bigram
        )

    def _insert(self, cache, tokens, *, restored=False, **namespace):
        key = self._key(tokens, **namespace)
        return cache.insert(
            InsertParams(
                key=key,
                value=torch.arange(len(key), dtype=torch.int64),
                restored_from_cache=restored,
            )
        )

    @staticmethod
    def _evict_leaf(cache, node_id):
        result = cache.tree_core.evict_device_leaf(node_id, is_write_back=False)
        cache._free_values(result.device_frees, result.host_frees)

    @staticmethod
    def _metric(registry, name, **labels):
        return registry.get_sample_value(
            "sglang:" + name,
            {
                "cache_type": "UnifiedRadixCache",
                "dp_rank": "0",
                "pp_rank": "0",
                **labels,
            },
        )

    def test_capacity_eviction_reinsert_counts_only_fresh_adopted_pages(self):
        """A repeated cache hit cannot inflate the recomputation denominator."""
        cache, registry = self._cache()
        self._insert(cache, range(4))
        self._insert(cache, range(4))
        self.assertEqual(self._metric(registry, "kv_inserted_tokens_total"), 4)
        cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(cache.token_to_kv_pool_allocator.freed, list(range(4)) * 2)
        self.assertEqual(
            self._metric(registry, "kv_ghost_list_recorded_pages_total"), 2
        )
        self.clock.return_value = 17.0
        self._insert(cache, range(4))
        self.assertEqual(self._metric(registry, "kv_inserted_tokens_total"), 8)
        self.assertEqual(
            self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 4
        )
        self.assertEqual(
            self._metric(registry, "kv_reinsert_after_evict_seconds_count"), 1
        )
        self.assertEqual(
            self._metric(registry, "kv_reinsert_after_evict_seconds_sum"), 7
        )
        self.assertEqual(
            self._metric(
                registry, "kv_ghost_list_discarded_pages_total", reason="reinsert"
            ),
            2,
        )

    def test_host_survivor_excludes_demotion_and_readoption(self):
        """Only eviction of the last local copy creates capacity history."""
        cache, registry = self._cache()
        result = self._insert(cache, range(4))
        core = cache.tree_core
        core.commit_backup(result.last_device_node, torch.arange(4), {})
        self._evict_leaf(cache, result.last_device_node)
        self.assertTrue(core.is_backuped(result.last_device_node))
        self.assertEqual(
            self._metric(registry, "kv_ghost_list_recorded_pages_total"), 0
        )
        self._insert(cache, range(4))
        self.assertEqual(self._metric(registry, "kv_inserted_tokens_total"), 4)
        self.assertEqual(
            self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 0
        )
        self._evict_leaf(cache, result.last_device_node)
        dropped = core.drive_host_eviction(ComponentType.FULL, 4)
        self.assertEqual(dropped.tracker[ComponentType.FULL], 4)
        cache._free_values(dropped.device_frees, dropped.host_frees)
        self.assertEqual(
            self._metric(registry, "kv_ghost_list_recorded_pages_total"), 2
        )
        self._insert(cache, range(4))
        self.assertEqual(self._metric(registry, "kv_inserted_tokens_total"), 8)
        self.assertEqual(
            self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 4
        )

    def test_storage_restore_consumes_history_without_recomputation(self):
        """Host prefetch and direct device restore retire ghosts without a hit."""
        for tier in ("host", "device"):
            with self.subTest(tier=tier):
                cache, registry = self._cache()
                self._insert(cache, range(4))
                cache.evict(EvictParams(num_tokens=4))
                if tier == "host":
                    result = cache.tree_core.insert_host(
                        cache.tree_core.root_node.id,
                        RadixKey(list(range(4))),
                        torch.arange(4),
                        ["page0", "page1"],
                    )
                    self.assertIsNotNone(result.inserted_host_node)
                    cache.insert(
                        InsertParams(
                            key=RadixKey(list(range(4))), value=torch.arange(4)
                        )
                    )
                else:
                    self._insert(cache, range(4), restored=True)
                self.assertEqual(self._metric(registry, "kv_inserted_tokens_total"), 4)
                self.assertEqual(
                    self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 0
                )
                self.assertEqual(
                    self._metric(
                        registry,
                        "kv_ghost_list_discarded_pages_total",
                        reason="restored",
                    ),
                    2,
                )
                self.assertEqual(len(cache.tree_core.ghost_tracker._entries), 0)

    def test_retired_host_attachment_cannot_consume_replacement_ghosts(self):
        """A late restore to an unusable generation cannot hide fresh capacity misses."""
        cache, registry = self._cache()
        core = cache.tree_core
        old = self._insert(cache, [1, 2]).last_device_node
        ref = core.capture_prefix_ref(old, 2)
        core.invalidate_prefix_ref(ref, 0)
        replacement = self._insert(cache, [1, 2, 3, 4]).last_device_node
        self._evict_leaf(cache, replacement)
        self.assertEqual(len(core.ghost_tracker._entries), 2)

        core.is_write_back = True
        restored = core.insert_host(
            old,
            self._key([3, 4]),
            torch.tensor([7, 8]),
            ["suffix"],
        )
        self.assertTrue(core.node_by_id(restored.inserted_host_node).retired)
        self.assertEqual(len(core.ghost_tracker._entries), 2)
        self.assertEqual(
            self._metric(
                registry, "kv_ghost_list_discarded_pages_total", reason="restored"
            ),
            0,
        )
        self._insert(cache, [1, 2, 3, 4])
        self.assertEqual(
            self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 4
        )

    def test_split_preserves_salted_bigram_page_identity(self):
        """Splitting at page boundaries must preserve the evicted suffix identity."""
        for bigram in (False, True):
            with self.subTest(bigram=bigram):
                cache, registry = self._cache()
                namespace = {"salt": "tenant-a", "extra": "adapter", "bigram": bigram}
                tokens = list(range(9 if bigram else 8))
                original = self._insert(cache, tokens, **namespace)
                branch = tokens[: 5 if bigram else 4] + [90, 91, 92, 93]
                self._insert(cache, branch, **namespace)
                core = cache.tree_core
                suffix = core.node_by_id(original.last_device_node)
                self.assertEqual(len(suffix.key), 4)
                self.assertEqual(len(suffix.parent.key), 4)
                self._evict_leaf(cache, suffix.id)
                self._insert(cache, tokens, **namespace)
                self.assertEqual(
                    self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 4
                )
                self.assertEqual(
                    self._metric(
                        registry,
                        "kv_ghost_list_discarded_pages_total",
                        reason="reinsert",
                    ),
                    2,
                )

    def test_swa_commit_splits_count_the_entire_fresh_full_span(self):
        """SWA commit can split a fresh Full leaf into several logical fragments."""
        for floor in (0, 4):
            with self.subTest(floor=floor):
                cache, registry = self._cache(
                    token_to_kv_pool_allocator=_CPUSWAAllocator(),
                    tree_components=(ComponentType.FULL, ComponentType.SWA),
                    sliding_window_size=2,
                )
                for insertion in (1, 2):
                    cache.insert(
                        InsertParams(
                            key=self._key(range(8)),
                            value=torch.arange(8),
                            swa_evicted_seqlen=floor,
                        )
                    )
                    self.assertEqual(
                        self._metric(registry, "kv_inserted_tokens_total"),
                        8 * insertion,
                    )
                    self.assertEqual(
                        self._metric(
                            registry, "kv_recomputed_after_evict_tokens_total"
                        ),
                        8 * (insertion - 1),
                    )
                    if insertion == 1:
                        cache.evict(EvictParams(num_tokens=8))
                        self.assertEqual(
                            self._metric(
                                registry, "kv_ghost_list_recorded_pages_total"
                            ),
                            4,
                        )

    def test_namespaces_and_bigram_mode_do_not_share_ghosts(self):
        """Identical raw token bytes can denote different tenants or logical pages."""
        variants = ({"salt": "other"}, {"extra": "other"}, {"bigram": True})
        for namespace in variants:
            with self.subTest(namespace=namespace):
                cache, registry = self._cache()
                self._insert(cache, range(4))
                cache.evict(EvictParams(num_tokens=4))
                self._insert(cache, range(4), **namespace)
                self.assertEqual(
                    self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 0
                )
                self.assertEqual(len(cache.tree_core.ghost_tracker._entries), 2)
                self._insert(cache, range(4))
                self.assertEqual(
                    self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 4
                )

    def test_invalidation_excludes_retired_pages_and_old_missing_descendants(self):
        """Invalidation must not inherit capacity evidence from a missing child."""
        cache, registry = self._cache()
        self._insert(cache, range(4))
        tail = self._insert(cache, range(8))
        core = cache.tree_core
        ref = core.capture_prefix_ref(tail.last_device_node, 8)
        self._evict_leaf(cache, tail.last_device_node)
        self.assertEqual(
            self._metric(registry, "kv_ghost_list_recorded_pages_total"), 2
        )
        core.invalidate_prefix_ref(ref, 0)
        self.assertEqual(
            self._metric(
                registry, "kv_ghost_list_discarded_pages_total", reason="invalidate"
            ),
            2,
        )
        cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(
            self._metric(registry, "kv_ghost_list_recorded_pages_total"), 2
        )
        self._insert(cache, range(8))
        self.assertEqual(
            self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 0
        )
        core.release_prefix_ref(ref)

    def test_invalidation_after_receipt_nodes_are_all_evicted_clears_history(self):
        """A receipt can outlive every tree node it invalidates."""
        cache, registry = self._cache()
        result = self._insert(cache, range(4))
        core = cache.tree_core
        ref = core.capture_prefix_ref(result.last_device_node, 4)
        cache.evict(EvictParams(num_tokens=4))
        core.invalidate_prefix_ref(ref, 0)
        self.assertEqual(
            self._metric(
                registry, "kv_ghost_list_discarded_pages_total", reason="invalidate"
            ),
            2,
        )
        self._insert(cache, range(4))
        self.assertEqual(
            self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 0
        )
        core.release_prefix_ref(ref)

    def test_inactive_and_foreign_receipts_preserve_capacity_history(self):
        """Unrelated or obsolete invalidation receipts cannot erase new history."""
        for receipt_state in ("released", "reset", "foreign"):
            with self.subTest(receipt_state=receipt_state):
                cache, registry = self._cache()
                core = cache.tree_core
                result = self._insert(cache, range(4))
                ref = core.capture_prefix_ref(result.last_device_node, 4)
                if receipt_state == "released":
                    core.release_prefix_ref(ref)
                elif receipt_state == "reset":
                    cache.reset()
                    self._insert(cache, range(4))
                else:
                    other, _ = self._cache()
                    result = self._insert(other, range(4))
                    ref = other.tree_core.capture_prefix_ref(result.last_device_node, 4)
                cache.evict(EvictParams(num_tokens=4))
                if receipt_state == "foreign":
                    with self.assertRaises(ValueError):
                        core.invalidate_prefix_ref(ref, 0)
                else:
                    core.invalidate_prefix_ref(ref, 0)
                self.assertEqual(
                    self._metric(
                        registry,
                        "kv_ghost_list_discarded_pages_total",
                        reason="invalidate",
                    ),
                    0,
                )
                self._insert(cache, range(4))
                self.assertEqual(
                    self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 4
                )

    def _assert_history_balance(self, registry, tracker):
        discarded = sum(
            self._metric(registry, "kv_ghost_list_discarded_pages_total", reason=reason)
            for reason in (
                "capacity",
                "ttl",
                "flush",
                "reinsert",
                "restored",
                "invalidate",
            )
        )
        self.assertEqual(
            self._metric(registry, "kv_ghost_list_recorded_pages_total"),
            discarded + len(tracker._entries),
        )
        self.assertLessEqual(len(tracker._entries), tracker.capacity)
        self.assertEqual(tracker._cutoff_count, sum(map(len, tracker._by_ref.values())))
        self.assertLessEqual(tracker._cutoff_count, tracker.capacity)

    def test_invalidation_preserves_disjoint_prefix_and_salt_history(self):
        """Invalidating one request cannot erase another prefix or tenant's history."""
        for evict_captured in (False, True):
            with self.subTest(evict_captured=evict_captured):
                cache, registry = self._cache()
                core = cache.tree_core
                target = self._insert(cache, range(4))
                ref = core.capture_prefix_ref(target.last_device_node, 4)
                disjoint = self._insert(cache, range(20, 24))
                salted = self._insert(cache, range(4), salt="other-tenant")
                self._evict_leaf(cache, disjoint.last_device_node)
                self._evict_leaf(cache, salted.last_device_node)
                if evict_captured:
                    self._evict_leaf(cache, target.last_device_node)
                core.invalidate_prefix_ref(ref, 0)
                self.assertEqual(
                    self._metric(
                        registry,
                        "kv_ghost_list_discarded_pages_total",
                        reason="invalidate",
                    ),
                    2 if evict_captured else 0,
                )
                self._insert(cache, range(4))
                self.assertEqual(
                    self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 0
                )
                self._insert(cache, range(20, 24))
                self._insert(cache, range(4), salt="other-tenant")
                self.assertEqual(
                    self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 8
                )
                self._assert_history_balance(registry, core.ghost_tracker)
                core.release_prefix_ref(ref)

    def test_deleted_receipt_partial_invalidation_preserves_earlier_pages(self):
        """Deletion must retain a receipt's page boundary and captured endpoint."""
        for start, removed_pages, retained_tokens in ((2, 3, 2), (6, 0, 8)):
            with self.subTest(start=start):
                cache, registry = self._cache()
                core = cache.tree_core
                result = self._insert(cache, range(8))
                ref = core.capture_prefix_ref(result.last_device_node, 6)
                self._evict_leaf(cache, result.last_device_node)
                unrelated = self._insert(cache, [90, 91])
                self._evict_leaf(cache, unrelated.last_device_node)
                self.assertEqual(core._prefix_refs.counts(), (1, 0))
                core.invalidate_prefix_ref(ref, start)
                self.assertEqual(
                    self._metric(
                        registry,
                        "kv_ghost_list_discarded_pages_total",
                        reason="invalidate",
                    ),
                    removed_pages,
                )
                self._insert(cache, range(8))
                self.assertEqual(
                    self._metric(registry, "kv_recomputed_after_evict_tokens_total"),
                    retained_tokens,
                )
                self._insert(cache, [90, 91])
                self.assertEqual(
                    self._metric(registry, "kv_recomputed_after_evict_tokens_total"),
                    retained_tokens + 2,
                )
                self._assert_history_balance(registry, core.ghost_tracker)
                core.release_prefix_ref(ref)

    def test_uncaptured_deleted_descendant_follows_ancestor_invalidation(self):
        """Retiring a captured ancestor also invalidates its uncaptured missing tail."""
        for delete_ancestor in (False, True):
            with self.subTest(delete_ancestor=delete_ancestor):
                cache, registry = self._cache()
                core = cache.tree_core
                ancestor = self._insert(cache, range(4))
                ref = core.capture_prefix_ref(ancestor.last_device_node, 4)
                descendant = self._insert(cache, range(8))
                self._evict_leaf(cache, descendant.last_device_node)
                if delete_ancestor:
                    self._evict_leaf(cache, ancestor.last_device_node)
                core.invalidate_prefix_ref(ref, 2)
                self.assertEqual(
                    self._metric(
                        registry,
                        "kv_ghost_list_discarded_pages_total",
                        reason="invalidate",
                    ),
                    3 if delete_ancestor else 2,
                )
                self._insert(cache, range(8))
                self.assertEqual(
                    self._metric(registry, "kv_recomputed_after_evict_tokens_total"),
                    2 if delete_ancestor else 0,
                )
                self._assert_history_balance(registry, core.ghost_tracker)
                core.release_prefix_ref(ref)

    def test_split_keeps_missing_descendant_provenance_out_of_new_branch(self):
        """A split must separate an old missing descendant from a new sibling."""
        cache, registry = self._cache()
        core = cache.tree_core
        ancestor = self._insert(cache, [0, 1, 2, 3])
        ref = core.capture_prefix_ref(ancestor.last_device_node, 4)
        descendant = self._insert(cache, range(8))
        self._evict_leaf(cache, descendant.last_device_node)
        sibling = self._insert(cache, [0, 1, 90, 91])
        self.assertEqual(len(core.node_by_id(ancestor.last_device_node).key), 2)
        self._evict_leaf(cache, sibling.last_device_node)
        core.invalidate_prefix_ref(ref, 2)
        self.assertEqual(
            self._metric(
                registry, "kv_ghost_list_discarded_pages_total", reason="invalidate"
            ),
            2,
        )
        self._insert(cache, range(8))
        self.assertEqual(
            self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 0
        )
        self._insert(cache, [0, 1, 90, 91])
        self.assertEqual(
            self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 2
        )
        self._assert_history_balance(registry, core.ghost_tracker)
        core.release_prefix_ref(ref)

    def test_invalidation_split_preserves_prefix_ghost_on_live_tombstone(self):
        """A partial retirement keeps ghosts belonging to the surviving prefix."""
        cache, registry = self._cache()
        core = cache.tree_core
        ancestor = self._insert(cache, range(4))
        descendant = self._insert(cache, range(8))
        ref = core.capture_prefix_ref(ancestor.last_device_node, 4)
        node = core.node_by_id(ancestor.last_device_node)
        full = ComponentType.FULL
        component = core.components_by_type[full]
        device_frees, host_frees = {full: []}, {full: []}
        tracker = {full: 0}
        core._evict_component_and_detach_lru(
            node,
            component,
            device_frees,
            host_frees,
            target=EvictLayer.DEVICE,
            tracker=tracker,
        )
        core._cascade_evict(
            node,
            component,
            tracker,
            device_frees,
            host_frees,
            target=EvictLayer.DEVICE,
        )
        cache._free_values(device_frees, host_frees)
        self.assertTrue(node.evicted)
        self.assertFalse(core.node_by_id(descendant.last_device_node).evicted)
        self.assertEqual(
            self._metric(registry, "kv_ghost_list_recorded_pages_total"), 2
        )
        core.invalidate_prefix_ref(ref, 2)
        self.assertEqual(len(node.key), 2)
        self.assertEqual(len(node.parent.key), 2)
        self.assertFalse(node.parent.retired)
        self.assertTrue(core.is_invalidated(descendant.last_device_node))
        self.assertEqual(
            self._metric(
                registry, "kv_ghost_list_discarded_pages_total", reason="invalidate"
            ),
            1,
        )
        self._insert(cache, range(2))
        self.assertEqual(
            self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 2
        )
        self._assert_history_balance(registry, core.ghost_tracker)
        core.release_prefix_ref(ref)

    def test_deleted_ancestor_chain_preserves_farthest_receipt_cutoff(self):
        """Moving history up a deleted path must not shorten its captured endpoint."""
        cache, registry = self._cache()
        core = cache.tree_core
        prefix = self._insert(cache, range(2))
        ancestor = self._insert(cache, range(4))
        descendant = self._insert(cache, range(6))
        ref = core.capture_prefix_ref(ancestor.last_device_node, 4)
        for result in (descendant, ancestor, prefix):
            self._evict_leaf(cache, result.last_device_node)
        self.assertEqual(core._prefix_refs.counts(), (1, 0))
        core.invalidate_prefix_ref(ref, 2)
        self.assertEqual(
            self._metric(
                registry, "kv_ghost_list_discarded_pages_total", reason="invalidate"
            ),
            2,
        )
        self._insert(cache, range(6))
        self.assertEqual(
            self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 2
        )
        self._assert_history_balance(registry, core.ghost_tracker)
        core.release_prefix_ref(ref)

    def test_late_ancestor_receipt_covers_already_evicted_descendant(self):
        """Capturing a live ancestor also covers history already anchored beneath it."""
        for delete_ancestor in (False, True):
            with self.subTest(delete_ancestor=delete_ancestor):
                cache, registry = self._cache()
                core = cache.tree_core
                ancestor = self._insert(cache, range(4))
                descendant = self._insert(cache, range(8))
                self._evict_leaf(cache, descendant.last_device_node)
                ref = core.capture_prefix_ref(ancestor.last_device_node, 4)
                unrelated = self._insert(cache, [90, 91])
                self._evict_leaf(cache, unrelated.last_device_node)
                if delete_ancestor:
                    self._evict_leaf(cache, ancestor.last_device_node)
                core.invalidate_prefix_ref(ref, 0)
                self.assertEqual(
                    self._metric(
                        registry,
                        "kv_ghost_list_discarded_pages_total",
                        reason="invalidate",
                    ),
                    4 if delete_ancestor else 2,
                )
                self._insert(cache, range(8))
                self.assertEqual(
                    self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 0
                )
                self._insert(cache, [90, 91])
                self.assertEqual(
                    self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 2
                )
                self._assert_history_balance(registry, core.ghost_tracker)
                core.release_prefix_ref(ref)

    def test_old_receipt_cannot_remove_new_same_key_eviction(self):
        """Recomputation consumes old evidence; a later eviction is a new event."""
        cache, registry = self._cache()
        core = cache.tree_core
        original = self._insert(cache, range(4))
        ref = core.capture_prefix_ref(original.last_device_node, 4)
        self._evict_leaf(cache, original.last_device_node)
        replacement = self._insert(cache, range(4))
        self._evict_leaf(cache, replacement.last_device_node)
        core.invalidate_prefix_ref(ref, 0)
        self.assertEqual(
            self._metric(
                registry, "kv_ghost_list_discarded_pages_total", reason="invalidate"
            ),
            0,
        )
        self._insert(cache, range(4))
        self.assertEqual(
            self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 8
        )
        self._assert_history_balance(registry, core.ghost_tracker)
        core.release_prefix_ref(ref)

    def test_receipt_metadata_overflow_discards_only_affected_history(self):
        """Many receipts cannot exceed the history budget or evict unrelated evidence."""
        with envs.SGLANG_KV_GHOST_CAPACITY_PAGES.override(4):
            cache, registry = self._cache()
        core = cache.tree_core
        unrelated = self._insert(cache, [90, 91])
        self._evict_leaf(cache, unrelated.last_device_node)
        result = self._insert(cache, [0, 1])
        refs = [core.capture_prefix_ref(result.last_device_node, 2) for _ in range(12)]
        references = core._prefix_refs.references
        memberships = []

        def counted_references(node_id):
            for membership in references(node_id):
                memberships.append(membership)
                yield membership

        with patch.object(core._prefix_refs, "references", counted_references):
            self._evict_leaf(cache, result.last_device_node)
        tracker = core.ghost_tracker
        self.assertLessEqual(len(memberships), tracker.capacity + 1)
        self._assert_history_balance(registry, tracker)
        self.assertEqual(len(tracker._entries), 1)
        self.assertEqual(tracker._cutoff_count, 0)
        self.assertEqual(tracker._by_ref, {})
        self.assertEqual(
            self._metric(
                registry, "kv_ghost_list_discarded_pages_total", reason="capacity"
            ),
            1,
        )
        core.invalidate_prefix_ref(refs[-1], 0)
        self._insert(cache, [0, 1])
        self.assertEqual(
            self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 0
        )
        self._insert(cache, [90, 91])
        self.assertEqual(
            self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 2
        )
        for ref in refs:
            core.release_prefix_ref(ref)
        self._assert_history_balance(registry, tracker)

    def test_released_receipts_keep_cutoff_dict_storage_linear(self):
        """Released memberships must not leave quadratic dictionary backing storage."""
        capacity = 64
        with envs.SGLANG_KV_GHOST_CAPACITY_PAGES.override(capacity):
            cache, registry = self._cache()
        core = cache.tree_core
        tracker = core.ghost_tracker
        retained_refs = []
        for index in range(capacity):
            result = self._insert(cache, [2 * index, 2 * index + 1])
            refs = [
                core.capture_prefix_ref(result.last_device_node, 2)
                for _ in range(capacity - index)
            ]
            self._evict_leaf(cache, result.last_device_node)
            for ref in refs[:-1]:
                core.release_prefix_ref(ref)
            retained_refs.append(refs[-1])
        self.assertEqual(len(tracker._entries), capacity)
        self.assertEqual(tracker._cutoff_count, capacity)
        self._assert_history_balance(registry, tracker)
        cutoffs = [entry.cutoffs for entry in tracker._entries.values()]
        self.assertTrue(all(len(links) == 1 for links in cutoffs))
        retained_bytes = sum(sys.getsizeof(links) for links in cutoffs)
        compact_bytes = sum(
            sys.getsizeof({ref_id: cutoff for ref_id, cutoff in links.items()})
            for links in cutoffs
        )
        self.assertLessEqual(retained_bytes, 4 * compact_bytes)
        for ref in retained_refs:
            core.release_prefix_ref(ref)
        self.assertEqual(tracker._cutoff_count, 0)
        self._assert_history_balance(registry, tracker)

    def test_release_and_ttl_remove_receipt_links_without_leaking_history(self):
        """Receipt release preserves ghosts; TTL expiration removes their metadata."""
        cache, registry = self._cache()
        core = cache.tree_core
        result = self._insert(cache, range(4))
        ref = core.capture_prefix_ref(result.last_device_node, 4)
        self._evict_leaf(cache, result.last_device_node)
        tracker = core.ghost_tracker
        self.assertEqual(tracker._cutoff_count, 2)
        core.release_prefix_ref(ref)
        self.assertEqual(tracker._cutoff_count, 0)
        self.assertEqual(tracker._by_ref, {})
        self.assertEqual(len(tracker._entries), 2)
        result = self._insert(cache, range(4))
        ref = core.capture_prefix_ref(result.last_device_node, 4)
        self._evict_leaf(cache, result.last_device_node)
        self.assertEqual(tracker._cutoff_count, 2)
        self.clock.return_value = 310.0
        self._insert(cache, [90, 91])
        self.assertEqual(tracker._cutoff_count, 0)
        self.assertEqual(tracker._by_ref, {})
        self.assertEqual(len(tracker._entries), 0)
        self.assertEqual(
            self._metric(registry, "kv_ghost_list_discarded_pages_total", reason="ttl"),
            2,
        )
        self._assert_history_balance(registry, tracker)
        core.release_prefix_ref(ref)

    def test_capacity_ttl_and_reset_conserve_history(self):
        """Recorded pages equal discarded pages plus the bounded live history."""
        with envs.SGLANG_KV_GHOST_CAPACITY_PAGES.override(2):
            cache, registry = self._cache()
        self._insert(cache, range(6))
        cache.evict(EvictParams(num_tokens=6))
        self.assertEqual(
            self._metric(
                registry, "kv_ghost_list_discarded_pages_total", reason="capacity"
            ),
            1,
        )
        self.clock.return_value = 310.0
        self._insert(cache, range(6))
        self.assertEqual(
            self._metric(registry, "kv_ghost_list_discarded_pages_total", reason="ttl"),
            2,
        )
        self.assertEqual(
            self._metric(registry, "kv_recomputed_after_evict_tokens_total"), 0
        )
        cache.evict(EvictParams(num_tokens=6))
        cache.reset()
        self.assertEqual(
            self._metric(
                registry, "kv_ghost_list_discarded_pages_total", reason="flush"
            ),
            2,
        )
        recorded = self._metric(registry, "kv_ghost_list_recorded_pages_total")
        discarded = sum(
            self._metric(registry, "kv_ghost_list_discarded_pages_total", reason=reason)
            for reason in (
                "capacity",
                "ttl",
                "flush",
                "reinsert",
                "restored",
                "invalidate",
            )
        )
        self.assertEqual(
            recorded, discarded + len(cache.tree_core.ghost_tracker._entries)
        )

    def test_nonexporters_do_not_construct_tracker_or_hash_pages(self):
        """TP/CP replicas and opt-outs incur no ghost construction or hashing."""
        cases = (
            {"attn_tp_rank": 1},
            {"attn_cp_rank": 1},
            {"enable_metrics": False},
            {"disable": True},
        )
        for params in cases:
            with (
                self.subTest(params=params),
                patch(
                    "sglang.srt.mem_cache.unified_radix_cache.KVGhostTracker",
                    side_effect=AssertionError("nonexporter constructed ghost tracker"),
                ),
                patch(
                    "sglang.srt.mem_cache.kv_ghost._node_digests",
                    side_effect=AssertionError("nonexporter hashed pages"),
                ),
            ):
                cache, registry = self._cache(**params)
                self.assertIsNone(cache.tree_core.ghost_tracker)
                self._insert(cache, range(4))
                cache.evict(EvictParams(num_tokens=4))
                self.assertIsNone(self._metric(registry, "kv_inserted_tokens_total"))
        with (
            envs.SGLANG_KV_GHOST_CAPACITY_PAGES.override(0),
            patch(
                "sglang.srt.mem_cache.unified_radix_cache.KVGhostTracker",
                side_effect=AssertionError(
                    "disabled feature constructed ghost tracker"
                ),
            ),
            patch(
                "sglang.srt.mem_cache.kv_ghost._node_digests",
                side_effect=AssertionError("disabled feature hashed pages"),
            ),
        ):
            cache, registry = self._cache()
            self._insert(cache, range(4))
            cache.evict(EvictParams(num_tokens=4))
            self.assertIsNone(cache.tree_core.ghost_tracker)
            self.assertIsNone(self._metric(registry, "kv_inserted_tokens_total"))

    def test_custom_collectors_keep_constructor_compatibility_and_opt_out(self):
        """Missing optional ghost methods and explicit emission opt-out are safe."""
        for collector in (_LabelsOnlyCollector, _KwargsCollector, _DisabledCollector):
            with (
                self.subTest(collector=collector.__name__),
                patch(
                    "sglang.srt.mem_cache.base_prefix_cache.resolve_collector_class",
                    return_value=collector,
                ),
                patch(
                    "sglang.srt.mem_cache.unified_radix_cache.KVGhostTracker",
                    side_effect=AssertionError(
                        "unsupported collector constructed tracker"
                    ),
                ),
            ):
                cache, registry = self._cache()
                self.assertIsNone(cache.tree_core.ghost_tracker)
                self.assertIsNone(self._metric(registry, "kv_inserted_tokens_total"))
                self.assertEqual(
                    cache.emit_logical_cache_metrics,
                    collector is not _DisabledCollector,
                )
                if collector is _KwargsCollector:
                    self.assertTrue(
                        cache.metrics_collector.options["emit_cache_metrics"]
                    )
                    self.assertEqual(
                        cache.metrics_collector.options["logical_labels"]["dp_rank"],
                        "0",
                    )

    def test_legacy_builtin_collector_exports_one_correctly_identified_series(self):
        """A legacy constructor must inherit both rank gating and logical labels."""
        for rank in (0, 1):
            with (
                self.subTest(rank=rank),
                patch(
                    "sglang.srt.mem_cache.base_prefix_cache.resolve_collector_class",
                    return_value=_LegacyBuiltinCollector,
                ),
            ):
                cache, registry = self._cache(attn_tp_rank=rank, dp_rank=3, pp_rank=2)
                self._insert(cache, range(4))
                self.assertEqual(
                    self._metric(
                        registry, "kv_inserted_tokens_total", dp_rank="3", pp_rank="2"
                    ),
                    4 if rank == 0 else None,
                )
                if rank:
                    self.assertIsNone(cache.tree_core.ghost_tracker)
                self.assertIsNone(self._metric(registry, "kv_inserted_tokens_total"))

    def test_rank_export_preserves_dp_identity_and_physical_eviction_metrics(self):
        """Logical pages export once per DP replica; physical eviction remains local."""
        logical = []
        for dp_rank in (0, 1):
            for tp_rank, cp_rank in ((0, 0), (1, 0), (0, 1)):
                cache, registry = self._cache(
                    dp_rank=dp_rank, attn_tp_rank=tp_rank, attn_cp_rank=cp_rank
                )
                self._insert(cache, range(4))
                cache.evict(EvictParams(num_tokens=4))
                physical = {"cache_type": "UnifiedRadixCache"}
                self.assertEqual(
                    registry.get_sample_value("sglang:evicted_tokens_total", physical),
                    4,
                )
                self.assertEqual(
                    registry.get_sample_value(
                        "sglang:eviction_duration_seconds_count", physical
                    ),
                    1,
                )
                logical.extend(
                    sample
                    for metric in registry.collect()
                    for sample in metric.samples
                    if sample.name == "sglang:kv_inserted_tokens_total"
                )
        self.assertEqual(len(logical), 2)
        self.assertEqual({sample.labels["dp_rank"] for sample in logical}, {"0", "1"})
        self.assertEqual(sum(sample.value for sample in logical), 8)


if __name__ == "__main__":
    unittest.main()
