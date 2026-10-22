# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tri-pool composite (`UnifiedMambaSWATokenToKVPoolAllocator`) -- full KV +
SWA KV + mamba/conv state in ONE unified byte buffer, chain
``[mamba (up END) | swa (FLOAT) | full (down END)]``.

Pure CPU; fakes stand in for the KV pools (data markers verify moves).
"""

import inspect
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    kv_mamba_slots,
)
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedMambaSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.unified_sub_pool import (
    FloatMultiEndedAllocator,
    MultiEndedAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.kv_loc_plan import IdSpaceKind
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    DraftStateGeometry,
    DraftStateRegion,
    FusedDraftPlacement,
)
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_memory_pool import (
    UnifiedHybridLinearKVPool,
    UnifiedKVPool,
    UnifiedMambaPool,
    UnifiedMambaSlotAllocator,
    UnifiedMHATokenToKVPool,
    init_unified_mamba_swa_pools,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import (
    get_context,
    get_memory,
    get_parallel,
    get_spec,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    FakeKVCache,
    FakeUnifiedSWAKVPool,
    tri_sub_pool_specs,
)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_DEV = "cpu"


class _NoMambaHostReadback(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (
            torch.ops.aten._local_scalar_dense.default,
            torch.ops.aten.nonzero.default,
        ):
            raise AssertionError(f"Host readback in tracking-slot bookkeeping: {func}")
        if func is torch.ops.aten.index.Tensor and any(
            index is not None and index.dtype == torch.bool for index in args[1]
        ):
            raise AssertionError("Boolean indexing needs a host-visible output size")
        return func(*args, **(kwargs or {}))


class _NoPageableCountReadback(TorchDispatchMode):
    """Boolean indexing reads its output size into pageable memory, which
    waits behind every in-flight HiCache D2H copy on the device."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.nonzero.default or (
            func is torch.ops.aten.index.Tensor
            and any(
                index is not None and index.dtype == torch.bool for index in args[1]
            )
        ):
            raise AssertionError(f"Pageable count readback: {func}")
        return func(*args, **(kwargs or {}))


class TestUnifiedTriPool(unittest.TestCase):
    def _build(
        self,
        n_full=32,
        n_swa=16,
        n_state=8,
        page_size=1,
        lazy_compaction=False,
    ):
        full, swa, mamba = tri_sub_pool_specs()
        total = (
            n_full * full.entry_bytes()
            + n_swa * swa.entry_bytes()
            + n_state * mamba.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa, mamba],
            device=_DEV,
            enable_memory_saver=False,
            page_size=page_size,
        )
        kvcache = FakeUnifiedSWAKVPool(pool)
        mamba_kv = FakeKVCache(pool.max_slots("mamba"))
        allocator = UnifiedMambaSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            mamba_kvcache=mamba_kv,
            device=_DEV,
            full_max_total_num_tokens=n_full,
            swa_max_total_num_tokens=n_swa,
            page_size=page_size,
            need_sort=False,
            forward_stream=None,
            lazy_compaction=lazy_compaction,
        )
        return pool, allocator, kvcache, mamba_kv

    def _stamp(self, allocator, kvcache, v):
        fa = allocator.full_attn_allocator
        sa = allocator.swa_attn_allocator
        kvcache.full_kv_pool.buf[fa.virtual_to_physical[v]] = v
        kvcache.swa_kv_pool.buf[sa.virtual_to_physical[v]] = v

    # -- construction --

    def test_chain_wiring_and_float_swa(self):
        pool, allocator, kvcache, _ = self._build()
        fa = allocator.full_attn_allocator
        sa = allocator.swa_attn_allocator
        ma = allocator.mamba_allocator
        self.assertIsInstance(sa, FloatMultiEndedAllocator)
        self.assertIs(ma.high_peer, sa)
        self.assertIs(sa.low_peer, ma)
        self.assertIs(sa.high_peer, fa)
        self.assertIs(fa.low_peer, sa)
        # Canonical chain order in the pool.
        self.assertEqual(
            [s.name for s in pool.sub_pool_specs], ["mamba", "swa", "full"]
        )
        # KV pool got the allocators.
        self.assertIs(kvcache._full_allocator, fa)
        self.assertIs(kvcache._swa_allocator, sa)

    def test_pd_preallocation_binds_only_swa_tail_pages(self):
        _, allocator, _, _ = self._build(page_size=4)
        before = allocator.available_size()
        prefix = torch.tensor([0], dtype=torch.int64)
        seq = torch.tensor([12], dtype=torch.int64)
        # With an empty prefix, ordinary allocation supplies the same virtual
        # pages without launching the GPU extend kernel. All page binding and
        # capacity accounting still run through the real sub-allocators.
        full = allocator.full_attn_allocator
        with patch.object(
            full, "alloc_extend", side_effect=lambda *a, **kw: full.alloc(12)
        ):
            virtual = allocator.alloc_extend_swa_tail(
                prefix, prefix, seq, seq, torch.tensor([-1]), 12, swa_tail_len=5
            )
        self.assertIsNotNone(virtual)
        self.assertEqual(len(virtual), 12)
        self.assertEqual(allocator.full_attn_allocator.allocated_count(), 12)
        self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 8)
        swa_pages = allocator.swa_v2p_page_table[virtual[::4] // 4]
        self.assertLessEqual(swa_pages[0].item(), 0)
        self.assertTrue(torch.all(swa_pages[1:] > 0).item())
        allocator.free(virtual)
        self.assertEqual(allocator.full_attn_allocator.allocated_count(), 0)
        self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 0)
        self.assertEqual(allocator.available_size(), before)

    def test_pd_short_tail_fits_beyond_joint_capacity(self):
        for lazy in (False, True):
            for tail_len in (0, 5):
                with self.subTest(lazy=lazy, tail_len=tail_len):
                    _, allocator, _, _ = self._build(page_size=4, lazy_compaction=lazy)
                    full = allocator.full_attn_allocator
                    length = allocator.available_size() + 4
                    self.assertFalse(allocator.can_reserve(length, length))
                    self.assertTrue(allocator.can_reserve(length, tail_len))
                    prefix = torch.tensor([0], dtype=torch.int64)
                    seq = torch.tensor([length], dtype=torch.int64)
                    with patch.object(
                        full,
                        "alloc_extend",
                        side_effect=lambda *a, **kw: full.alloc(length),
                    ):
                        virtual = allocator.alloc_extend_swa_tail(
                            prefix,
                            prefix,
                            seq,
                            seq,
                            torch.tensor([-1]),
                            length,
                            tail_len,
                        )
                    self.assertIsNotNone(virtual)
                    self.assertEqual(full.allocated_count(), length)
                    self.assertEqual(
                        allocator.swa_attn_allocator.allocated_count(),
                        -(-tail_len // 4) * 4,
                    )
                    self.assertEqual(allocator.verify_byte_accounting(), [])
                    allocator.free(virtual)
                    self.assertEqual(full.allocated_count(), 0)
                    self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 0)

    def test_pd_tail_rejects_full_capacity_shortfall(self):
        _, allocator, _, _ = self._build(page_size=4)
        full = allocator.full_attn_allocator
        length = full.available_size() + 4
        prefix = torch.tensor([0], dtype=torch.int64)
        seq = torch.tensor([length], dtype=torch.int64)
        with patch.object(full, "alloc_extend") as extend:
            self.assertIsNone(
                allocator.alloc_extend_swa_tail(
                    prefix, prefix, seq, seq, torch.tensor([-1]), length, 0
                )
            )
        extend.assert_not_called()
        self.assertEqual(full.allocated_count(), 0)
        self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 0)

    def test_empty_float_is_transparent_to_the_ends(self):
        _, allocator, _, _ = self._build()
        fa = allocator.full_attn_allocator
        ma = allocator.mamba_allocator
        self.assertTrue(allocator.swa_attn_allocator._is_frontier_transparent())
        # full's chain gap reaches the mamba end's frontier straight through.
        self.assertEqual(
            fa._current_gap_bytes(),
            fa._byte_low_frontier() - ma._byte_high_frontier(),
        )

    # -- the joint availability contract --

    def test_available_shrinks_as_state_slots_grow(self):
        _, allocator, _, _ = self._build()
        before = allocator.available_size()
        slots = allocator.mamba_allocator.alloc(4)
        self.assertIsNotNone(slots)
        after = allocator.available_size()
        self.assertLess(after, before)
        allocator.mamba_allocator.free(slots)
        self.assertEqual(allocator.available_size(), before)

    # -- steady-state SWA churn: tombstones -> holes -> in-place reuse --

    def _swa_interior_block(self, allocator, blocks):
        """The block whose SWA-physical pages touch neither float boundary:
        `free_swa` on it holes; a boundary block would be absorbed instead."""
        sa = allocator.swa_attn_allocator
        for v in blocks:
            pages = set(int(x) for x in sa.virtual_to_physical[v].tolist())
            if sa.low_wm_page not in pages and (sa.high_wm_page - 1) not in pages:
                return v
        raise AssertionError("no interior block in layout")

    def test_free_swa_holes_recycled_in_place_zero_copy(self):
        _, allocator, kvcache, _ = self._build()
        blocks = [allocator.alloc(4) for _ in range(3)]
        for v in blocks:
            self.assertIsNotNone(v)
            self._stamp(allocator, kvcache, v)
        sa = allocator.swa_attn_allocator
        fa = allocator.full_attn_allocator
        span_before = (sa.low_wm_page, sa.high_wm_page)

        # Window slide: an INTERIOR block ages out of the SWA window.
        v_mid = self._swa_interior_block(allocator, blocks)
        allocator.free_swa(v_mid)
        # The full side keeps the token; the swa side tombstoned it.
        self.assertTrue(bool((fa.virtual_to_physical[v_mid] >= 0).all()))
        self.assertTrue(bool((sa.virtual_to_physical[v_mid] == -1).all()))
        self.assertEqual(sa._hole_pages(), 4)
        self.assertEqual((sa.low_wm_page, sa.high_wm_page), span_before)

        # The next alloc recycles the holes IN PLACE: no span growth, no moves.
        vd = allocator.alloc(4)
        self.assertIsNotNone(vd)
        self._stamp(allocator, kvcache, vd)
        self.assertEqual(sa._hole_pages(), 0)
        self.assertEqual((sa.low_wm_page, sa.high_wm_page), span_before)
        self.assertEqual(len(sa._inverse_history), 0)  # zero copies

    def test_free_swa_boundary_block_absorbed_zero_copy(self):
        # A boundary block's tombstones shrink the span instead of holing it.
        # The shrink needs the hole set on the host, so it is deferred out of
        # the per-step free and lands in the opportunistic flush.
        _, allocator, kvcache, _ = self._build()
        blocks = [allocator.alloc(4) for _ in range(2)]
        for v in blocks:
            self._stamp(allocator, kvcache, v)
        sa = allocator.swa_attn_allocator
        span_pages = sa._span_pages()
        # Pick a block holding a span-boundary page.
        boundary = None
        for v in blocks:
            pages = set(int(x) for x in sa.virtual_to_physical[v].tolist())
            if sa.low_wm_page in pages or (sa.high_wm_page - 1) in pages:
                boundary = v
                break
        self.assertIsNotNone(boundary)
        allocator.free_swa(boundary)
        allocator.flush_opportunistic()  # the deferred reclaim point
        self.assertEqual(sa._hole_pages(), 0)  # absorbed, not holed
        self.assertEqual(sa._span_pages(), span_pages - 4)
        self.assertEqual(len(sa._inverse_history), 0)  # zero copies

    def test_free_releases_both_sides_and_filters_tombstones(self):
        _, allocator, kvcache, _ = self._build()
        va = allocator.alloc(4)
        self._stamp(allocator, kvcache, va)
        allocator.free_swa(va)  # tombstone first (aged out of window)
        allocator.free(va)  # then the request finishes
        fa = allocator.full_attn_allocator
        sa = allocator.swa_attn_allocator
        self.assertTrue(bool((fa.virtual_to_physical[va] == -1).all()))
        self.assertTrue(bool((sa.virtual_to_physical[va] == -1).all()))
        # Fully-freed float parks and is transparent again.
        self.assertTrue(sa._is_frontier_transparent())

    # -- per-request state surface --

    def test_mamba_slot_allocator_surface(self):
        pool, allocator, _, mamba_kv = self._build()
        slot_alloc = UnifiedMambaSlotAllocator(
            allocator.mamba_allocator,
            max_size=pool.max_slots("mamba") - 1,
            device=_DEV,
        )
        v = slot_alloc.alloc(3)
        self.assertIsNotNone(v)
        p = slot_alloc.translate(v)
        self.assertTrue(bool((p >= 0).all()))
        mamba_kv.buf[p] = v
        self.assertEqual(
            slot_alloc.available_size(),
            (pool.max_slots("mamba") - 1) - 3,
        )
        slot_alloc.free(v)
        self.assertEqual(slot_alloc.available_size(), pool.max_slots("mamba") - 1)
        # Group prefetch draw-down + surplus return.
        slot_alloc.alloc_group_begin(4)
        s1 = slot_alloc.alloc(1)
        self.assertIsNotNone(s1)
        slot_alloc.alloc_group_end()
        self.assertEqual(
            slot_alloc.available_size(),
            (pool.max_slots("mamba") - 1) - 1,
        )

    # -- flush semantics --

    def test_urgent_flush_preserves_float_holes(self):
        _, allocator, kvcache, _ = self._build(lazy_compaction=True)
        blocks = [allocator.alloc(4) for _ in range(3)]
        for v in blocks:
            self._stamp(allocator, kvcache, v)
        allocator.free_swa(self._swa_interior_block(allocator, blocks))
        sa = allocator.swa_attn_allocator
        holes = sa._hole_pages()
        self.assertGreater(holes, 0)
        from sglang.srt.mem_cache.allocator.unified_sub_pool import _relieve_for_alloc

        _relieve_for_alloc(allocator, 1)
        self.assertEqual(sa._hole_pages(), holes)  # holes are assets, not backlog

    def test_full_donor_stops_after_float_exposes_mamba_capacity(self):
        for page_size, lazy_compaction in ((1, False), (4, True)):
            with self.subTest(page_size=page_size, lazy_compaction=lazy_compaction):
                _, allocator, _, _ = self._build(
                    page_size=page_size, lazy_compaction=lazy_compaction
                )
                mamba_slots = UnifiedMambaSlotAllocator(
                    allocator.mamba_allocator,
                    max_size=allocator.mamba_allocator.max_slots - 1,
                    device=_DEV,
                )

                full_leaves = []
                while True:
                    indices = allocator.alloc(allocator.page_size)
                    if indices is None:
                        break
                    full_leaves.append(indices)
                self.assertTrue(full_leaves)
                residual_mamba = mamba_slots.schedulable_available_size()
                if residual_mamba:
                    self.assertIsNotNone(mamba_slots.alloc(residual_mamba))
                while mamba_slots.alloc(1) is not None:
                    pass
                self.assertGreater(mamba_slots.available_size(), 0)
                self.assertEqual(mamba_slots.schedulable_available_size(), 0)

                cache = object.__new__(UnifiedRadixCache)
                cache.disable = False
                cache.tree_components = (
                    ComponentType.FULL,
                    ComponentType.SWA,
                    ComponentType.MAMBA,
                )
                cache.is_swa_enabled = True
                cache.cache_controller = None
                cache.metrics_collector = None
                cache.token_to_kv_pool_allocator = allocator
                cache.req_to_token_pool = MagicMock(mamba_allocator=mamba_slots)

                tree_core = MagicMock()
                tree_core.full_evictable_size.return_value = len(full_leaves)
                tree_core.mamba_evictable_size.return_value = 0
                walk = {"request_cnt": 0, "freed_leaves": 0}

                def start(component_type, request_cnt):
                    self.assertEqual(component_type, ComponentType.FULL)
                    walk["request_cnt"] = request_cnt

                def next_node(component_type, tracker):
                    self.assertEqual(component_type, ComponentType.FULL)
                    if tracker[ComponentType.FULL] >= walk["request_cnt"] or walk[
                        "freed_leaves"
                    ] >= len(full_leaves):
                        return None, False
                    return walk["freed_leaves"] + 1, True

                def evict_leaf(node_id, tracker):
                    self.assertEqual(node_id, walk["freed_leaves"] + 1)
                    indices = full_leaves[-node_id]
                    tracker[ComponentType.FULL] += int(indices.numel())
                    allocator.full_attn_allocator.free(indices)
                    walk["freed_leaves"] += 1
                    return None

                tree_core.evict_device_start.side_effect = start
                cache.tree_core = tree_core
                cache._evict_device_next_node = MagicMock(side_effect=next_node)
                cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)

                swa_live_before = allocator.swa_attn_allocator._live_pages()
                result = cache.evict_for_alloc(EvictParams(mamba_num=1))

                self.assertEqual(walk["freed_leaves"], 1)
                self.assertEqual(result.num_tokens_evicted, page_size)
                self.assertEqual(result.swa_num_tokens_evicted, 0)
                self.assertEqual(result.mamba_num_evicted, 0)
                self.assertEqual(
                    allocator.swa_attn_allocator._live_pages(), swa_live_before
                )
                self.assertGreaterEqual(mamba_slots.schedulable_available_size(), 1)
                self.assertIsNotNone(mamba_slots.alloc(1))
                self.assertEqual(allocator.verify_byte_accounting(), [])


class TestTriPagedFreeGroup(unittest.TestCase):
    """The tri composite at PAGE SIZE > 1, driven through the production free
    path: free_group_begin -> free_segment -> free_group_end.

    Regression: every other tri test runs at page_size=1, where the
    page-representative machinery is dead code. At ps>1 the composite releases
    reps via `swa_attn_allocator.free(..., _pages=...)`, which the float must
    accept or the first decode batch dies; honouring `_pages` is also what
    keeps the free path off the data-dependent `torch.unique` host sync.
    """

    def _build_paged(self, page_size=4, n_full=64, n_swa=32, n_state=8):
        full, swa, mamba = tri_sub_pool_specs()
        total = (
            n_full * full.entry_bytes()
            + n_swa * swa.entry_bytes()
            + n_state * mamba.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa, mamba],
            device=_DEV,
            enable_memory_saver=False,
            page_size=page_size,
        )
        kvcache = FakeUnifiedSWAKVPool(pool)
        mamba_kv = FakeKVCache(pool.max_slots("mamba"))
        allocator = UnifiedMambaSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            mamba_kvcache=mamba_kv,
            device=_DEV,
            full_max_total_num_tokens=n_full,
            swa_max_total_num_tokens=n_swa,
            page_size=page_size,
            need_sort=False,
            forward_stream=None,
        )
        return pool, allocator

    def test_free_group_segment_release_reaches_the_float(self):
        """The exact production sequence the scheduler runs per decode batch."""
        pool, allocator = self._build_paged()
        v = allocator.alloc(8)
        self.assertIsNotNone(v)
        before = allocator.available_size()

        allocator.free_group_begin()
        allocator.free_segment(v, start_pos=0)
        allocator.free_group_end()  # -> _release_page_reps -> float.free(_pages=)

        self.assertEqual(allocator.verify_byte_accounting(), [])
        self.assertGreaterEqual(allocator.available_size(), before)
        # Capacity fully recovered: the float parked, both ends rewound.
        self.assertTrue(allocator.swa_attn_allocator._is_frontier_transparent())

    def test_mamba_donor_flushes_full_only_group_without_closing_it(self):
        _, allocator = self._build_paged(page_size=1)
        full_indices = allocator.alloc(8)
        self.assertIsNotNone(full_indices)
        allocator.free_swa(full_indices)
        allocated_before = allocator.full_attn_allocator.allocated_count()

        allocator.free_group_begin()
        allocator.free_full_segment(full_indices, start_pos=0)
        self.assertTrue(allocator.full_page_reps_group)

        donor = allocator.mamba_full_cache_donor()
        self.assertIsNotNone(donor)
        donor.flush_deferred_full_frees()

        self.assertEqual(allocator.free_group, [])
        self.assertEqual(allocator.free_page_reps_group, [])
        self.assertEqual(allocator.full_page_reps_group, [])
        self.assertEqual(allocator.full_free_group, [])
        self.assertLess(
            allocator.full_attn_allocator.allocated_count(), allocated_before
        )
        self.assertEqual(allocator.verify_byte_accounting(), [])
        allocator.free_group_end()

    def test_ungrouped_segment_free_also_reaches_the_float(self):
        pool, allocator = self._build_paged()
        v = allocator.alloc(8)
        self.assertIsNotNone(v)
        allocator.free_segment(v, start_pos=0)
        self.assertEqual(allocator.verify_byte_accounting(), [])


class TestTriFreeSwaNoHostSync(unittest.TestCase):
    """The tri's swa side is the FLOAT, which can never run the lazy event
    pipeline: unless the per-step frees carry caller-derived page ids, the tri
    reintroduces the host syncs the sync-free free path removed. Fixtures run
    at page_size > 1; ps==1 short-circuits the page machinery and hides this.
    """

    PS = 4

    def _tri(self):
        inst = TestTriPagedFreeGroup(
            [m for m in dir(TestTriPagedFreeGroup) if m.startswith("test_")][0]
        )
        return inst._build_paged(page_size=self.PS)[1]

    def test_ratchet_shape_free_swa_never_syncs_on_the_float(self):
        alloc = self._tri()
        v = alloc.alloc(8 * self.PS)
        self.assertIsNotNone(v)
        from unittest import mock

        with (
            mock.patch.object(
                torch, "unique", side_effect=AssertionError("unique = host sync")
            ),
            mock.patch.object(
                torch.Tensor, "item", side_effect=AssertionError("item = host sync")
            ),
        ):
            alloc.free_swa_segment(v[: 4 * self.PS], start_pos=0)
        self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_fallback_free_swa_still_correct_for_radix_shapes(self):
        """Radix eviction hands arbitrary node values (no start_pos): the
        dedup fallback must end in the same state as the stride path."""
        a1, a2 = self._tri(), self._tri()
        v1, v2 = a1.alloc(6 * self.PS), a2.alloc(6 * self.PS)
        self.assertTrue(torch.equal(v1, v2))
        a1.free_swa_segment(v1[: 4 * self.PS], start_pos=0)
        a2.free_swa(v2[: 4 * self.PS])
        self.assertTrue(
            torch.equal(
                a1.swa_attn_allocator.virtual_to_physical,
                a2.swa_attn_allocator.virtual_to_physical,
            )
        )
        self.assertEqual(a1.available_size(), a2.available_size())
        self.assertEqual(a1.verify_byte_accounting(), [])
        self.assertEqual(a2.verify_byte_accounting(), [])


class TestGeneralizedRebalance(unittest.TestCase):
    """The float must yield to WHICHEVER end is short, with the direction
    derived from the caller's growth direction rather than hard-coded to the
    token path's side. `make_room` was always side-agnostic; these pin the
    POLICY in `_ask_float_for_room`."""

    PS = 4

    def _tri(self):
        inst = TestTriPagedFreeGroup(
            [m for m in dir(TestTriPagedFreeGroup) if m.startswith("test_")][0]
        )
        return inst._build_paged(page_size=self.PS)[1]

    def test_direction_is_derived_from_growth_on_both_ends(self):
        """Raw end+float+end chain, BOTH orientations in one fixture: the
        up-growing end opens the float's LOW side, the down-growing end its
        HIGH side."""
        from test_multi_ended_allocator import TestFloatMultiEndedAllocator

        inst = TestFloatMultiEndedAllocator(
            [m for m in dir(TestFloatMultiEndedAllocator) if m.startswith("test_")][0]
        )
        _pool, up_end, fla, down_end, _kv = inst._build_tri()
        v = fla.alloc(8)  # opaque float mid-region
        self.assertIsNotNone(v)

        # UP end: exhaust its band below the float, then ask for more.
        e_up = up_end.entry_bytes_per_page
        fit = int((fla._byte_low_frontier() - up_end._byte_high_frontier()) // e_up)
        if fit > 0:
            self.assertIsNotNone(up_end.alloc(fit * up_end.page_size))
        low_before = fla.low_wm_page
        self.assertIsNotNone(up_end.alloc(up_end.page_size))
        self.assertGreater(fla.low_wm_page, low_before)  # opened LOW side

        # DOWN end: exhaust its band above the float, then ask for more.
        e_dn = down_end.entry_bytes_per_page
        fit = int((down_end._byte_low_frontier() - fla._byte_high_frontier()) // e_dn)
        if fit > 0:
            self.assertIsNotNone(down_end.alloc(fit * down_end.page_size))
        high_before = fla.high_wm_page
        self.assertIsNotNone(down_end.alloc(down_end.page_size))
        self.assertLess(fla.high_wm_page, high_before)  # opened HIGH side

    def test_two_pool_chain_rebalance_is_a_noop(self):
        from test_multi_ended_allocator import (
            TestPagedMultiEndedAllocator as _PagedFixture,
        )

        inst = _PagedFixture(
            [m for m in dir(_PagedFixture) if m.startswith("test_")][0]
        )
        _pool, full, swa, _fkv, _skv = inst._build()
        v = full.alloc(full.page_size * 2)
        self.assertIsNotNone(v)
        wm = full.watermark_physical
        full._ask_float_for_room(full.page_size * 1000)  # absurd ask
        self.assertEqual(full.watermark_physical, wm)  # untouched

    def test_index_cap_guard_never_moves_data_uselessly(self):
        """When the caller's own INDEX space binds, no amount of float
        movement helps -- make_room must not be called (poisoned)."""
        from unittest import mock

        alloc = self._tri()
        alloc.alloc(4 * self.PS)
        ma = alloc.mamba_allocator
        sa = alloc.swa_attn_allocator
        huge = (ma.num_pages + 10) * ma.page_size  # beyond index space
        with mock.patch.object(
            sa, "make_room", side_effect=AssertionError("useless make_room")
        ):
            ma._ask_float_for_room(huge)


class TestComputedShortSide(unittest.TestCase):
    """`_ask_float_for_room` must open the side that MEASURES short -- never
    "the side facing full" -- including the shape with coupled ends on BOTH
    sides of the float.
    """

    PS = 4

    def _tri(self):
        inst = TestTriPagedFreeGroup(
            [m for m in dir(TestTriPagedFreeGroup) if m.startswith("test_")][0]
        )
        return inst._build_paged(page_size=self.PS)[1]

    def _sides(self, alloc):
        sa = alloc.swa_attn_allocator
        low = max(0, sa._byte_low_frontier() - sa._chain_high_frontier_below_bytes())
        high = max(0, sa._chain_low_frontier_above_bytes() - sa._byte_high_frontier())
        return low, high

    def test_float_share_short_opens_the_state_side(self):
        """RED-LINE: full's demand fits its band, the float's own share fits
        NEITHER band, and the state side has the larger surplus -- so the
        policy must open the STATE side, i.e. slide the float toward full
        during a TOKEN alloc.
        """
        from unittest import mock

        alloc = self._tri()
        v = alloc.alloc(6 * self.PS)  # places the float mid-region
        self.assertIsNotNone(v)
        sa = alloc.swa_attn_allocator
        fa = alloc.full_attn_allocator
        e_f, e_s = fa.entry_bytes_per_page, sa.entry_bytes_per_page

        # Position: slide the float LOW (setup uses the mechanism directly),
        # so the low band is small and the geometry below is expressible.
        b_low0, b_high0 = self._sides(alloc)
        # Pack the float low (leapfrog over-opens by design), then reopen LOW
        # to ~2 full-pages: F must outgrow it, yet the need_n window below
        # must stay non-empty.
        sa.make_room(side="high", min_bytes=b_low0 + b_high0 - 2 * e_f)
        sa.make_room(side="low", min_bytes=2 * e_f)

        # Find a need_n where: D_high = need_n*e_f fits band_high, F =
        # need_n*e_s exceeds BOTH surpluses, and low has the larger surplus.
        chosen = None
        for need_n in range(1, 64):
            b_low, b_high = self._sides(alloc)
            s_low = b_low  # no coupled end on the low side
            s_high = b_high - need_n * e_f
            if s_high < 0:
                break
            F = need_n * e_s
            if F > s_low and F > s_high and s_low >= s_high:
                chosen = need_n
                break
        self.assertIsNotNone(chosen, "fixture cannot express the geometry")

        calls = []
        real = sa.make_room
        with mock.patch.object(
            sa, "make_room", side_effect=lambda **kw: calls.append(kw) or real(**kw)
        ):
            alloc._ask_float_for_room(chosen * self.PS)
        self.assertEqual(len(calls), 1, calls)
        self.assertEqual(calls[0]["side"], "low")  # the STATE side

    def test_full_side_short_target_matches_the_closed_form(self):
        """When the full side is the short one, the ask must equal the
        closed form demand + max(0, F - far_surplus) + slack."""
        from unittest import mock

        alloc = self._tri()
        v = alloc.alloc(4 * self.PS)
        self.assertIsNotNone(v)
        sa, fa = alloc.swa_attn_allocator, alloc.full_attn_allocator
        e_f, e_s = fa.entry_bytes_per_page, sa.entry_bytes_per_page
        chosen = None
        for need_n in range(1, 256):
            b_low, b_high = self._sides(alloc)
            if b_high - need_n * e_f < 0 and b_low >= 0:
                chosen = need_n
                break
        self.assertIsNotNone(chosen)
        b_low, b_high = self._sides(alloc)
        F = max(0, chosen - sa._hole_pages()) * e_s
        want = chosen * e_f + max(0, F - b_low) + max(e_f, e_s)
        calls = []
        with mock.patch.object(
            sa, "make_room", side_effect=lambda **kw: calls.append(kw)
        ):
            alloc._ask_float_for_room(chosen * self.PS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["side"], "high")
        self.assertEqual(calls[0]["min_bytes"], want)

    def test_two_coupled_ends_lands_demand_on_both_sides(self):
        """A coupled set with ends on BOTH sides of the float: one-side-short
        must open that side; BOTH-sides-short must not move at all
        (relocation is zero-sum between the bands)."""
        from unittest import mock

        alloc = self._tri()
        v = alloc.alloc(6 * self.PS)
        self.assertIsNotNone(v)
        sa, fa, ma = (
            alloc.swa_attn_allocator,
            alloc.full_attn_allocator,
            alloc.mamba_allocator,
        )
        # Synthetic coupling: the state end joins the demand vector, the
        # override a composite with ends on both sides would ship.
        need = lambda self, t: {
            fa: -(-t // self.page_size),
            sa: -(-t // self.page_size),
            ma: -(-t // self.page_size),
        }
        with mock.patch.object(type(alloc), "_alloc_demand", need):
            # (a) both sides short: absurd need -> both demands exceed their
            # bands -> make_room must NOT be called.
            with mock.patch.object(
                sa, "make_room", side_effect=AssertionError("zero-sum move")
            ):
                alloc._ask_float_for_room(10_000 * self.PS)

            # (b) one side short: find a need where the HIGH side (full) is
            # short while the LOW side (mamba demand) still fits.
            e_f, e_m = fa.entry_bytes_per_page, ma.entry_bytes_per_page
            chosen = None
            for need_n in range(1, 256):
                b_low, b_high = self._sides(alloc)
                if b_high - need_n * e_f < 0 and b_low - need_n * e_m >= 0:
                    chosen = need_n
                    break
            if chosen is not None:
                calls = []
                with mock.patch.object(
                    sa, "make_room", side_effect=lambda **kw: calls.append(kw)
                ):
                    alloc._ask_float_for_room(chosen * self.PS)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["side"], "high")

    def test_nothing_short_means_no_relocation(self):
        """Everything fits -> no page moves. The tri's token vector carries
        {mamba: 0}: a zero entry must not move the float either."""
        from unittest import mock

        alloc = self._tri()
        alloc.alloc(4 * self.PS)
        demand = alloc._alloc_demand(2 * self.PS)
        self.assertEqual(demand[alloc.mamba_allocator], 0)
        sa = alloc.swa_attn_allocator
        with mock.patch.object(
            sa, "make_room", side_effect=AssertionError("needless move")
        ):
            alloc._ask_float_for_room(1)


class TestFloatPolicyTotalTarget(unittest.TestCase):
    """`make_room`'s min_bytes is a TARGET for the whole band, not a delta.

    Regression: an ask shaped as `deficit + one page` lands BELOW the current
    gap when the band is only PARTIALLY free, so `make_room` no-ops and the
    allocation fails though the float had room to slide.
    """

    PS = 4

    def _tri(self):
        inst = TestTriPagedFreeGroup(
            [m for m in dir(TestTriPagedFreeGroup) if m.startswith("test_")][0]
        )
        return inst._build_paged(page_size=self.PS)[1]

    def test_partial_gap_state_alloc_still_succeeds(self):
        alloc = self._tri()
        v = alloc.alloc(6 * self.PS)
        self.assertIsNotNone(v)
        ma, sa = alloc.mamba_allocator, alloc.swa_attn_allocator
        e_m = ma.entry_bytes_per_page
        gap_slots = int((sa._byte_low_frontier() - ma._byte_high_frontier()) // e_m)
        self.assertGreater(gap_slots, 2)
        low_before = sa.low_wm_page
        # partial-gap + 3: a delta-shaped ask lands below the current gap.
        got = ma.alloc((gap_slots + 3) * ma.page_size)
        self.assertIsNotNone(got, "partial-gap shortfall must relocate, not fail")
        self.assertGreater(sa.low_wm_page, low_before)
        self.assertEqual(alloc.verify_byte_accounting(), [])


class TestTriDeferredAbsorption(unittest.TestCase):
    """Boundary absorption is deferred out of the per-step free and paid once
    at a quiescent point. These pin WHERE it is paid, and that skipping it
    stays merely conservative."""

    PS = 4

    def _tri(self):
        inst = TestTriPagedFreeGroup(
            [m for m in dir(TestTriPagedFreeGroup) if m.startswith("test_")][0]
        )
        return inst._build_paged(page_size=self.PS)[1]

    def test_per_step_flush_reclaims_the_span(self):
        alloc = self._tri()
        v = alloc.alloc(8 * self.PS)
        sa = alloc.swa_attn_allocator
        span = sa._span_pages()
        alloc.free_swa_segment(v[6 * self.PS :], start_pos=6 * self.PS)  # high edge
        self.assertGreater(sa._hole_pages(), 0)  # deferred
        self.assertEqual(sa._span_pages(), span)
        moved = alloc.flush_opportunistic()
        self.assertGreater(moved, 0)
        self.assertLess(sa._span_pages(), span)
        self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_shortfall_ladder_absorbs_before_the_deficit_math(self):
        """The zero-copy rung must run FIRST: a stale-wide span inflates the
        rebalance deficit and buys a relocation the shrink already covers."""
        alloc = self._tri()
        v = alloc.alloc(8 * self.PS)
        sa = alloc.swa_attn_allocator
        alloc.free_swa_segment(v[6 * self.PS :], start_pos=6 * self.PS)
        self.assertGreater(sa._hole_pages(), 0)
        moves_before = len(sa._inverse_history)
        from sglang.srt.mem_cache.allocator.unified_sub_pool import _relieve_for_alloc

        _relieve_for_alloc(alloc, 1)  # the ladder
        self.assertEqual(sa._hole_pages(), 0)  # rung 0 ran
        self.assertEqual(len(sa._inverse_history), moves_before)  # zero copies

    def test_deferral_is_conservative_never_over_reports(self):
        """Availability with a stale-wide span must never EXCEED the absorbed
        value -- under-reporting is safe, over-reporting would over-admit."""
        alloc = self._tri()
        v = alloc.alloc(8 * self.PS)
        alloc.free_swa_segment(v[6 * self.PS :], start_pos=6 * self.PS)
        deferred = alloc.available_size()
        alloc.swa_attn_allocator._flush(urgent=False)
        absorbed = alloc.available_size()
        self.assertLessEqual(deferred, absorbed)
        self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_clean_flush_skips_the_d2h_entirely(self):
        """Only `free` can put a hole ON a boundary (alloc DRAINS holes into
        live pages, extension adds live pages), so with nothing freed since
        the last absorb the walk provably finds nothing and must not pay the
        D2H."""
        from unittest import mock

        alloc = self._tri()
        v = alloc.alloc(8 * self.PS)
        alloc.free_swa_segment(v[2 * self.PS : 4 * self.PS], start_pos=2 * self.PS)
        alloc.flush_opportunistic()  # consumes the dirty flag
        sa = alloc.swa_attn_allocator
        self.assertGreater(sa._hole_pages(), 0)  # interior holes remain
        with mock.patch.object(
            torch.Tensor, "tolist", side_effect=AssertionError("tolist = D2H")
        ):
            self.assertEqual(alloc.flush_opportunistic(), 0)
            self.assertEqual(sa._flush(urgent=False), 0)

    def test_alloc_between_frees_cannot_hide_a_boundary_hole(self):
        """An alloc drains holes and can restore a previously-seen hole
        COUNT, so the flag must be armed by `free`, not read off `numel()`."""
        alloc = self._tri()
        v = alloc.alloc(8 * self.PS)
        sa = alloc.swa_attn_allocator
        alloc.free_swa_segment(v[: 2 * self.PS], start_pos=0)  # low-edge holes
        n_after_free = sa._hole_pages()
        alloc.alloc(2 * self.PS)  # drains them back to live
        alloc.free_swa_segment(v[6 * self.PS :], start_pos=6 * self.PS)  # high edge
        self.assertEqual(sa._hole_pages(), n_after_free)  # same COUNT as before
        span = sa._span_pages()
        self.assertGreater(alloc.flush_opportunistic(), 0)  # still absorbed
        self.assertLess(sa._span_pages(), span)
        self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_transparency_still_exact_without_absorption(self):
        """Park-on-empty stays in `free` because it is sync-free -- a float
        that empties must go transparent immediately, with no flush needed."""
        alloc = self._tri()
        v = alloc.alloc(4 * self.PS)
        sa = alloc.swa_attn_allocator
        self.assertFalse(sa._is_frontier_transparent())
        from unittest import mock

        with mock.patch.object(
            torch.Tensor, "tolist", side_effect=AssertionError("tolist = D2H")
        ):
            alloc.free_swa_segment(v, start_pos=0)
        self.assertTrue(sa._is_frontier_transparent())
        self.assertEqual(sa._hole_pages(), 0)


class TestTriFactorySizing(CustomTestCase):
    """Factory-level contracts: byte-budget sizing, the bs=1 feasibility
    floor, and the boot signature the GPU harness greps for (3 sub-pools,
    swa grow=float)."""

    def _factory_kwargs(self, **over):
        import types

        cp = types.SimpleNamespace(
            shape=types.SimpleNamespace(conv=[(3, 8)], temporal=(0, 0, 0)),
            dtype=types.SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32),
            layers=[0, 1],
        )
        kw = dict(
            device=_DEV,
            kv_cache_dtype=torch.float16,
            head_num=2,
            head_dim=4,
            v_head_dim=4,
            swa_head_num=2,
            swa_head_dim=4,
            swa_v_head_dim=4,
            page_size=1,
            start_layer=0,
            end_layer=2,
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0],
            mamba_layer_ids=[0, 1],
            mamba2_cache_params=cp,
            full_max_total_num_tokens=64,
            swa_max_total_num_tokens=32,
            max_mamba_cache_size=4,
            model_context_len=16,
            extra_max_context_len=4,
            max_num_reqs=4,
            enable_memory_saver=False,
            enable_mamba_extra_buffer=False,
            enable_mamba_extra_buffer_lazy=False,
            disable_overlap_schedule=True,
            need_sort=False,
        )
        kw.update(over)
        return kw

    def test_streaming_session_lazy_checkpoint_cleanup(self):
        from types import SimpleNamespace

        from sglang.srt.managers.schedule_batch import ReqKvInfo
        from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.session.streaming_session import SessionSlot, StreamingSession

        for lazy_compaction in (None, False, True):
            for ids in ((0, -1), (-1, 0), (-1, -1), (0, 1)):
                with self.subTest(compaction=lazy_compaction, ids=ids):
                    kw = self._factory_kwargs(
                        enable_mamba_extra_buffer=True,
                        enable_mamba_extra_buffer_lazy=True,
                        lazy_compaction=lazy_compaction,
                    )
                    with get_parallel().override(attn_dcp_size=1):
                        pool = init_unified_mamba_swa_pools(**kw).req_to_token_pool
                    if lazy_compaction is None:
                        pool = SimpleNamespace(
                            mamba_allocator=MambaSlotAllocator(16, "cpu")
                        )
                    allocator = pool.mamba_allocator
                    available = allocator.available_size()
                    count = sum(i != -1 for i in ids)
                    live = allocator.alloc(count + 1)
                    track = live.new_full((2,), -1)
                    for position, index in enumerate(ids):
                        if index != -1:
                            track[position] = live[index + 1]
                    slot = SessionSlot(
                        kv=ReqKvInfo(
                            req_pool_idx=1,
                            mamba_pool_idx=live[0],
                            mamba_ping_pong_track_buffer=track,
                            mamba_ping_pong_allocated=sum(
                                1 << i for i, index in enumerate(ids) if index != -1
                            ),
                        )
                    )
                    session = StreamingSession(SimpleNamespace(req_to_token_pool=pool))
                    session.slots = {"test": slot}
                    self.assertEqual(kv_mamba_slots(slot.kv), count + 1)
                    session._free_slot_mamba(slot)
                    self.assertEqual(kv_mamba_slots(slot.kv), 0)
                    self.assertEqual(slot.kv.mamba_ping_pong_allocated, 0)
                    self.assertEqual(allocator.available_size(), available)
                    session._free_slot_mamba(slot)
                    self.assertEqual(allocator.available_size(), available)

    def test_mamba_lazy_checkpoint_cleanup(self):
        from types import SimpleNamespace

        from sglang.srt.runtime_context import get_parallel

        for tri_pool in (False, True):
            for lazy in (False, True):
                for keep_checkpoint in (False, True):
                    with self.subTest(
                        tri_pool=tri_pool, lazy=lazy, keep=keep_checkpoint
                    ):
                        from sglang.srt.mem_cache import kv_cache_configurator as cfg

                        kw = self._factory_kwargs()
                        model = SimpleNamespace(
                            get_num_kv_heads=lambda tp, dcp: 2,
                            get_swa_num_kv_heads=lambda tp: 2,
                            head_dim=4,
                            v_head_dim=4,
                            swa_head_dim=4,
                            swa_v_head_dim=4,
                            context_len=16,
                            full_attention_layer_ids=[0],
                            swa_attention_layer_ids=[1],
                            sliding_window_size=8,
                        )
                        configurator = SimpleNamespace(
                            mambaish_config=SimpleNamespace(
                                mamba2_cache_params=kw["mamba2_cache_params"],
                                full_attention_layer_ids=[0],
                            ),
                            model_config=model,
                            layer_info=SimpleNamespace(
                                swa_attention_layer_ids=[1],
                                full_attention_layer_ids=[0],
                                start_layer=0,
                                end_layer=2,
                            ),
                            device=_DEV,
                            kv_cache_dtype=torch.float16,
                            page_size=1,
                            is_draft_worker=False,
                            use_mla_backend=False,
                            is_hybrid_swa=tri_pool,
                            is_hybrid_swa_compress=False,
                            forward_stream=None,
                            # Spec off: no draft region to fuse.
                            _fused_draft_for_mamba_factory=lambda: None,
                            _unified_swa_head_geometry=lambda: (
                                cfg.KVCacheConfigurator._unified_swa_head_geometry(
                                    configurator
                                )
                            ),
                        )
                        # Run the production configurator AND factory. Reverting
                        # either top-level flag forwarding must break cleanup.
                        with (
                            get_parallel().override(attn_dcp_size=1, attn_tp_size=1),
                            patch.object(
                                cfg,
                                "get_exec",
                                return_value=SimpleNamespace(
                                    features=SimpleNamespace(enable_memory_saver=False),
                                    mamba=SimpleNamespace(
                                        enable_mamba_extra_buffer=True,
                                        enable_mamba_extra_buffer_lazy=lazy,
                                    ),
                                ),
                            ),
                            patch.object(
                                cfg,
                                "get_spec",
                                return_value=SimpleNamespace(
                                    speculative_num_draft_tokens=None,
                                ),
                            ),
                            patch.object(
                                cfg,
                                "get_schedule",
                                return_value=SimpleNamespace(
                                    max_mamba_cache_size=4,
                                    disable_overlap_schedule=False,
                                    mamba_full_memory_ratio=None,
                                ),
                            ),
                            patch.object(
                                cfg,
                                "get_disagg",
                                return_value=SimpleNamespace(
                                    disaggregation_mode="null",
                                ),
                            ),
                            patch.object(
                                cfg, "_should_enable_lazy_compaction", return_value=True
                            ),
                        ):
                            if tri_pool:
                                bundle = cfg.KVCacheConfigurator._init_unified_mamba_swa_pools(
                                    configurator,
                                    max_num_reqs=4,
                                    full_max_total_num_tokens=64,
                                    swa_max_total_num_tokens=32,
                                )
                            else:
                                bundle = (
                                    cfg.KVCacheConfigurator._init_unified_mamba_pools(
                                        configurator,
                                        max_num_reqs=4,
                                        max_total_num_tokens=64,
                                    )
                                )
                            pool = bundle.req_to_token_pool
                        self.assertEqual(pool.enable_mamba_extra_buffer_lazy, lazy)
                        allocator = pool.mamba_allocator
                        initial_available = allocator.available_size()
                        req = SimpleNamespace(
                            kv=SimpleNamespace(
                                req_pool_idx=0,
                                mamba_pool_idx=allocator.alloc(1)[0],
                                dflash_context_state=None,
                            )
                        )
                        pool._alloc_ping_pong_buffer(req)
                        buf = req.kv.mamba_ping_pong_track_buffer
                        self.assertEqual(int((buf != -1).sum()), 1 if lazy else 2)
                        if lazy:
                            # A decode boundary replaces the old checkpoint and
                            # leaves its ping-pong entry unallocated (-1).
                            replacement = allocator.alloc(1)
                            allocator.free(buf[:1].clone())
                            pool.set_mamba_ping_pong_slot(req, 0, -1)
                            pool.set_mamba_ping_pong_slot(req, 1, replacement[0])
                        else:
                            pool.set_mamba_ping_pong_slot(req, 1, buf[1])
                        retained = buf[1:2].clone()
                        with patch.object(
                            allocator, "free", wraps=allocator.free
                        ) as free:
                            pool.free_mamba_cache(req, 1 if keep_checkpoint else None)
                            for call in free.call_args_list:
                                self.assertTrue(bool((call.args[0] >= 0).all()))
                        self.assertEqual(
                            allocator.available_size(),
                            initial_available - int(keep_checkpoint),
                        )
                        self.assertIsNone(req.kv.mamba_ping_pong_track_buffer)
                        if keep_checkpoint:
                            self.assertTrue(
                                bool((allocator.translate(retained) >= 0).all())
                            )
                            allocator.free(retained)
                        self.assertEqual(allocator.available_size(), initial_available)

    def test_lazy_tracking_cleanup_without_host_readbacks(self):
        """Free every owned slot except the donated one, without GPU filtering."""
        from sglang.srt.managers.schedule_batch import ReqKvInfo

        for overlap in (False, True):
            size = 2 if overlap else 1
            for mask in range(1 << size):
                for keep in [None] + [i for i in range(size) if mask & (1 << i)]:
                    with self.subTest(overlap=overlap, mask=mask, keep=keep):
                        kw = self._factory_kwargs(
                            enable_mamba_extra_buffer=True,
                            enable_mamba_extra_buffer_lazy=True,
                            disable_overlap_schedule=not overlap,
                            lazy_compaction=True,
                        )
                        with get_parallel().override(attn_dcp_size=1):
                            pool = init_unified_mamba_swa_pools(**kw).req_to_token_pool
                        allocator = pool.mamba_allocator
                        available = allocator.available_size()
                        req = SimpleNamespace(
                            kv=ReqKvInfo(
                                req_pool_idx=0, mamba_pool_idx=allocator.alloc(1)[0]
                            )
                        )
                        pool._alloc_ping_pong_buffer(req)
                        buf = req.kv.mamba_ping_pong_track_buffer
                        for i in range(size):
                            allocated = bool(buf[i] != -1)
                            if mask & (1 << i):
                                if not allocated:
                                    pool.set_mamba_ping_pong_slot(
                                        req, i, allocator.alloc(1)[0]
                                    )
                            elif allocated:
                                allocator.free(buf[i : i + 1].clone())
                                pool.set_mamba_ping_pong_slot(req, i, -1)
                        pool.req_index_to_mamba_ping_pong_track_buffer_mapping[0] = buf
                        retained = (
                            None if keep is None else buf[keep : keep + 1].clone()
                        )

                        with _NoMambaHostReadback():
                            pool.free_mamba_cache(req, keep)
                        self.assertEqual(
                            allocator.available_size(),
                            available - int(keep is not None),
                        )
                        self.assertIsNone(req.kv.mamba_ping_pong_track_buffer)
                        self.assertEqual(req.kv.mamba_ping_pong_allocated, 0)
                        if retained is not None:
                            self.assertTrue(
                                bool((allocator.translate(retained) >= 0).all())
                            )
                            allocator.free(retained)
                        self.assertEqual(allocator.available_size(), available)

    def test_lazy_pending_slot_ownership_before_confirmation(self):
        """A pending crossing owns its slot before its result promotes it."""
        from sglang.srt.managers.schedule_batch import ReqKvInfo, ScheduleBatch
        from sglang.srt.managers.scheduler_components.batch_result_processor import (
            SchedulerBatchResultProcessor,
        )

        kw = self._factory_kwargs(
            enable_mamba_extra_buffer=True,
            enable_mamba_extra_buffer_lazy=True,
            disable_overlap_schedule=False,
            lazy_compaction=True,
        )
        with get_parallel().override(attn_dcp_size=1):
            pool = init_unified_mamba_swa_pools(**kw).req_to_token_pool
        req = SimpleNamespace(
            kv=ReqKvInfo(
                req_pool_idx=0,
                mamba_pool_idx=pool.mamba_allocator.alloc(1)[0],
                kv_committed_len=63,
            ),
            finished=lambda: False,
        )
        pool._alloc_ping_pong_buffer(req)
        pool.req_index_to_mamba_ping_pong_track_buffer_mapping[0] = (
            req.kv.mamba_ping_pong_track_buffer
        )
        batch = ScheduleBatch(reqs=[req], req_to_token_pool=pool)
        initial_available = pool.mamba_allocator.available_size()
        with (
            patch.object(pool.mamba_allocator, "alloc", return_value=None),
            _NoMambaHostReadback(),
        ):
            batch.mamba_lazy_spec_prepare(64, 16)
        self.assertEqual(batch.mamba_lazy_spec_track_positions_cpu, [0])
        self.assertEqual(pool.mamba_allocator.available_size(), initial_available)

        with _NoMambaHostReadback():
            batch.mamba_lazy_spec_prepare(64, 16)
        pending = batch.mamba_lazy_spec_track_positions_cpu[0]
        self.assertNotEqual(pending, req.kv.mamba_next_track_idx)
        available = pool.mamba_allocator.available_size()
        with _NoMambaHostReadback():
            batch.mamba_lazy_spec_prepare(64, 16)
        self.assertEqual(pool.mamba_allocator.available_size(), available)
        with _NoMambaHostReadback():
            SchedulerBatchResultProcessor._mamba_lazy_spec_update(
                None, req, batch, 0, True, 64
            )
        self.assertEqual(req.kv.mamba_next_track_idx, pending)
        self.assertEqual(pool.mamba_allocator.available_size(), available + 1)
        with _NoMambaHostReadback():
            batch.mamba_lazy_spec_prepare(64, 16)
            SchedulerBatchResultProcessor._mamba_lazy_spec_update(
                None, req, batch, 0, False, 0
            )
        self.assertEqual(req.kv.mamba_next_track_idx, pending)
        self.assertEqual(pool.mamba_allocator.available_size(), available)
        with _NoMambaHostReadback():
            pool.free_mamba_cache(req)

    def test_budget_sizing_and_boot_signature(self):
        from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
            UnifiedSWAAllocatorBase,
            UnifiedSWATokenToKVPoolAllocator,
        )

        budget = 1 << 20
        bundle = init_unified_mamba_swa_pools(
            **self._factory_kwargs(unified_total_bytes=budget)
        )
        pool = bundle.unified_memory_pool
        allocator = bundle.token_to_kv_pool_allocator
        self.assertIsInstance(allocator, UnifiedSWAAllocatorBase)
        self.assertNotIsInstance(allocator, UnifiedSWATokenToKVPoolAllocator)
        # Buffer = budget + the state pool's bytes (budget captured AFTER the
        # state carve-out), never the token-count re-sum.
        state_bytes = 4 * pool.spec("mamba").entry_bytes()
        self.assertEqual(pool.total_bytes, budget + state_bytes)
        # Boot signature: 3 sub-pools in chain order, swa is the float.
        self.assertEqual(len(pool.sub_pool_specs), 3)
        self.assertEqual(
            [(sp.name, sp.grow_direction) for sp in pool.sub_pool_specs],
            [("mamba", "up"), ("swa", "float"), ("full", "down")],
        )

    def test_fallback_is_the_token_count_resum(self):
        bundle = init_unified_mamba_swa_pools(**self._factory_kwargs())
        pool = bundle.unified_memory_pool
        want = (
            64 * pool.spec("full").entry_bytes()
            + 32 * pool.spec("swa").entry_bytes()
            + 4 * pool.spec("mamba").entry_bytes()
        )
        self.assertEqual(pool.total_bytes, want)

    def test_full_side_carries_the_fused_draft_region(self):
        """The boot solve prices the fused entry whenever a placement resolves,
        so a factory that dropped the kwarg would allocate UNFUSED under that
        price and under-budget the private draft pool."""
        region = DenseDraftRegion(
            lane_num=1, head_num=2, head_dim=4, store_dtype=torch.float16
        )
        placement = FusedDraftPlacement.from_counts(
            counts={"full": [1]}, regions={"full": region}
        )
        bundle = init_unified_mamba_swa_pools(
            **self._factory_kwargs(fused_draft=placement)
        )
        pool = bundle.unified_memory_pool
        self.assertIs(pool.fused_draft, placement)
        self.assertIs(pool.spec("full").draft_region, region)
        self.assertIs(pool.require_draft_host_spec("full"), pool.spec("full"))
        for other in ("swa", "mamba"):
            self.assertIsNone(pool.spec(other).draft_region, other)

    def _state_placement(self):
        region = DraftStateRegion(
            lane_num=2,
            state=DraftStateGeometry(
                conv_state_shapes=((3, 8),),
                conv_dtype=torch.bfloat16,
                temporal_state_shape=(0, 0, 0),
                temporal_dtype=torch.float32,
            ),
        )
        return FusedDraftPlacement.from_counts(
            counts={"mamba": [1, 1]}, regions={"mamba": region}
        )

    def test_fused_draft_state_clone_shares_the_request_slot_space(self):
        """A draft runner's req pool over the fused state block keeps the
        target's request->slot mappings and slot allocator (the draft's slot
        ids ARE the target's), and views only its own block."""
        bundle = init_unified_mamba_swa_pools(
            **self._factory_kwargs(fused_draft=self._state_placement())
        )
        target = bundle.req_to_token_pool
        clone = target.clone_for_fused_draft(layer_lanes={1: 1})
        self.assertIs(
            clone.req_index_to_mamba_index_mapping,
            target.req_index_to_mamba_index_mapping,
        )
        self.assertIs(clone.mamba_allocator, target.mamba_allocator)
        self.assertIs(clone.req_to_token, target.req_to_token)
        self.assertEqual(clone.mamba_map, {1: 0})
        conv = clone.mamba2_layer_cache(1).conv[0]
        self.assertEqual(tuple(conv.shape), (target.mamba_pool.size + 1, 3, 8))
        draft_conv, _ = bundle.unified_memory_pool.build_draft_state_views("mamba")
        self.assertEqual(conv.data_ptr(), draft_conv[0][1].data_ptr())

    def test_bs1_floor_fails_loud_before_construction(self):
        """A budget far below one worst-case request must raise BEFORE any
        pool construction -- under-sizing is a retract LIVELOCK at runtime."""
        with self.assertRaisesRegex(RuntimeError, "bs=1 floor"):
            init_unified_mamba_swa_pools(
                **self._factory_kwargs(
                    unified_total_bytes=1024,  # << ctx * e_f alone
                    model_context_len=100_000,
                    sliding_window_size=64,
                )
            )


class TestSeparateDraftSWAPool(CustomTestCase):
    def _build_pools(self, page_size):
        full, swa, mamba = tri_sub_pool_specs(full_layer_num=2, swa_layer_num=6)
        swa = replace(
            swa,
            head_num=1,
            head_dim=8,
            v_head_dim=8,
            store_dtype=torch.bfloat16,
            kv_cache_dtype=torch.bfloat16,
        )
        mamba = replace(mamba, temporal_state_shape=(2, 2, 4))
        shared = UnifiedKVPool(
            total_bytes=64 * full.entry_bytes()
            + 32 * swa.entry_bytes()
            + 8 * mamba.entry_bytes(),
            sub_pool_specs=[full, swa, mamba],
            device=_DEV,
            enable_memory_saver=False,
            page_size=page_size,
        )
        state = UnifiedMambaPool(
            unified_buffer=shared,
            sub_pool_name="mamba",
            spec_state_size=1,
            mamba_layer_ids=[0, 2],
        )
        full_pool = UnifiedMHATokenToKVPool(
            unified_buffer=shared, sub_pool_name="full", page_size=page_size
        )
        target = UnifiedHybridLinearKVPool(
            size=64,
            dtype=full.kv_cache_dtype,
            page_size=page_size,
            head_num=full.head_num,
            head_dim=full.head_dim,
            full_attention_layer_ids=[1, 3],
            device=_DEV,
            mamba_pool=state,
            full_kv_pool=full_pool,
        )
        draft = UnifiedMHATokenToKVPool(
            unified_buffer=shared, sub_pool_name="swa", page_size=page_size
        )
        return shared, target, draft, state

    def _allocator(self, shared, target, draft, state, page_size):
        return UnifiedMambaSWATokenToKVPoolAllocator(
            unified_buffer=shared,
            kvcache=target,
            swa_kvcache=draft,
            mamba_kvcache=state,
            device=_DEV,
            full_max_total_num_tokens=64,
            swa_max_total_num_tokens=32,
            page_size=page_size,
        )

    def test_draft_reclaim_preserves_target_kv_and_recurrent_state(self):
        """SWA recycling and Full compaction must preserve both models' rows
        despite overlapping layer IDs, different geometry, and different dtypes.
        """
        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                shared, target, draft, state = self._build_pools(page_size)
                alloc = self._allocator(shared, target, draft, state, page_size)
                self.assertIs(alloc.get_kvcache(), target)
                ids = alloc.alloc(3 * page_size)
                state_ids = alloc.mamba_allocator.alloc(1)
                self.assertIsNotNone(ids)
                self.assertIsNotNone(state_ids)
                state_phys = alloc.mamba_allocator.translate_kv_loc(state_ids)
                state.mamba_cache.conv[0][:, state_phys] = 42
                state.mamba_cache.temporal[:, state_phys] = 43

                buffers = []
                for pool, layer_ids, translate, base in (
                    (target, (1, 3), alloc.translate_kv_loc, 10),
                    (draft, range(6), alloc.translate_loc_from_full_to_swa, 100),
                ):
                    for layer_id in layer_ids:
                        for offset, get_buffer in enumerate(
                            (pool.get_key_buffer, pool.get_value_buffer)
                        ):
                            buf = get_buffer(layer_id)
                            values = (ids + base + 10 * layer_id + offset).to(buf.dtype)
                            values = values[:, None, None].expand(-1, *buf.shape[1:])
                            buf[translate(ids)] = values
                            buffers.append((pool, buf, translate, values))

                alloc.free_swa(ids[:page_size])
                self.assertEqual(alloc.full_attn_allocator.allocated_count(), len(ids))
                self.assertEqual(
                    alloc.swa_attn_allocator.allocated_count(), 2 * page_size
                )
                self.assertTrue(
                    bool((alloc.swa_v2p_page_table[ids[:1] // page_size] == -1).all())
                )
                alloc.swa_attn_allocator.alloc_with_virtual(ids[:1] // page_size)
                self.assertEqual(alloc.swa_attn_allocator.allocated_count(), len(ids))
                for pool, buf, translate, values in buffers:
                    if pool is draft:
                        buf[translate(ids[:page_size])] = values[:page_size]
                alloc.free(ids[page_size : 2 * page_size])
                remaining = torch.cat((ids[:page_size], ids[2 * page_size :]))
                swa = alloc.swa_attn_allocator
                old_swa_phys = alloc.translate_loc_from_full_to_swa(remaining)
                swa.make_room(
                    side="low",
                    min_bytes=swa._byte_low_frontier()
                    - swa._chain_high_frontier_below_bytes()
                    + swa.entry_bytes_per_page,
                )
                self.assertFalse(
                    torch.equal(
                        old_swa_phys, alloc.translate_loc_from_full_to_swa(remaining)
                    )
                )
                for _, buf, translate, values in buffers:
                    expected = torch.cat((values[:page_size], values[2 * page_size :]))
                    torch.testing.assert_close(
                        buf[translate(remaining)], expected, rtol=0, atol=0
                    )
                state_phys = alloc.mamba_allocator.translate_kv_loc(state_ids)
                self.assertTrue(
                    bool((state.mamba_cache.conv[0][:, state_phys] == 42).all())
                )
                self.assertTrue(
                    bool((state.mamba_cache.temporal[:, state_phys] == 43).all())
                )
                alloc.free(remaining)
                alloc.mamba_allocator.free(state_ids)
                for member in (
                    alloc.full_attn_allocator,
                    alloc.swa_attn_allocator,
                    alloc.mamba_allocator,
                ):
                    self.assertEqual(member.allocated_count(), 0)
                self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_rejects_draft_view_of_wrong_storage(self):
        shared, target, _, state = self._build_pools(page_size=1)
        _, _, foreign_draft, _ = self._build_pools(page_size=1)
        wrong_page_size = UnifiedMHATokenToKVPool(
            unified_buffer=shared, sub_pool_name="swa", page_size=2
        )
        for wrong_pool in (target.full_kv_pool, foreign_draft, wrong_page_size):
            with (
                self.subTest(pool=wrong_pool),
                self.assertRaisesRegex(
                    ValueError, "SWA view of the same unified buffer"
                ),
            ):
                self._allocator(shared, target, wrong_pool, state, page_size=1)


class TestDraftSWAFactory(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            enable_dflash_swa_cache=True,
            enable_hierarchical_cache=True,
        )
        override.install()
        self.addCleanup(override.restore)
        self.enterContext(get_parallel().override(attn_dcp_rank=0))

    def _factory_kwargs(self, **overrides):
        kwargs = TestTriFactorySizing._factory_kwargs(self)
        kwargs.update(
            start_layer=4,
            end_layer=8,
            full_attention_layer_ids=[5, 7],
            mamba_layer_ids=[4, 6],
            swa_attention_layer_ids=[],
            swa_head_num=1,
            swa_head_dim=8,
            swa_v_head_dim=16,
            draft_swa_layer_ids=list(range(2, 8)),
            draft_kv_cache_dtype=torch.bfloat16,
            sliding_window_size=8,
        )
        kwargs["mamba2_cache_params"].layers = [4, 6]
        kwargs["mamba2_cache_params"].shape.temporal = (2, 2, 4)
        kwargs.update(overrides)
        return kwargs

    def test_bounded_draft_windows_share_pages_and_survive_canonicalization(self):
        from types import SimpleNamespace

        from sglang.srt.managers.schedule_batch import ReqKvInfo
        from sglang.srt.speculative.dflash_cache import DFlashCacheResidency
        from sglang.srt.speculative.dflash_feature_tail import (
            DFlashContextState,
            DFlashContextWindow,
        )
        from sglang.test.kernels.dflash_window import (
            dflash_unbound_full_pages_reference,
            dflash_update_window_reference,
        )

        self.enterContext(
            patch(
                "sglang.srt.speculative.dflash_cache.dflash_compact_window_pages",
                side_effect=lambda held, mapping, num_live: held[mapping[held] > 0],
            )
        )
        self.enterContext(
            patch(
                "sglang.srt.speculative.dflash_cache.dflash_unbound_full_pages",
                side_effect=dflash_unbound_full_pages_reference,
            )
        )

        self.enterContext(
            patch(
                "sglang.srt.speculative.dflash_cache.dflash_update_window",
                side_effect=dflash_update_window_reference,
            )
        )

        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                bundle = init_unified_mamba_swa_pools(
                    **self._factory_kwargs(
                        page_size=page_size,
                        bounded_draft_swa=True,
                        model_context_len=80,
                        unified_total_bytes=131072,
                        full_max_total_num_tokens=128,
                        swa_max_total_num_tokens=32,
                    )
                )
                alloc = bundle.token_to_kv_pool_allocator
                table = bundle.req_to_token_pool
                draft = bundle.draft_token_to_kv_pool
                ids = alloc.alloc(64)
                self.assertIsNotNone(ids)
                self.assertEqual(alloc.swa_attn_allocator.allocated_count(), 0)
                table.req_to_token[0, :64] = ids
                table.req_to_token[1, :64] = ids
                a, b = (SimpleNamespace(kv=ReqKvInfo(req_pool_idx=i)) for i in (0, 1))
                for req, start in zip((a, b), (40, 88)):
                    # Each live tail holds prompt rows [40, 48); b's are distinct.
                    req.kv.dflash_context_state = DFlashContextState.from_tail(
                        DFlashContextWindow(
                            40, (torch.arange(start, start + 8).float()[:, None],)
                        ),
                        8,
                    )
                alloc.dflash_checkpoint_pool = SimpleNamespace(revision=0)
                cache = DFlashCacheResidency(alloc, table, window_size=8)

                def restore(*, tail, cache_loc, positions):
                    if tail.revision != alloc.dflash_checkpoint_pool.revision:
                        tail = DFlashContextWindow(
                            tail.start_pos,
                            (tail.as_tensor() + 1000,),
                            alloc.dflash_checkpoint_pool.revision,
                        )
                    target_hidden = tail.as_tensor()[positions - tail.start_pos]
                    draft.get_key_buffer(2)[cache_loc] = target_hidden[
                        :, :1, None
                    ].bfloat16()
                    return tail

                batch = SimpleNamespace(
                    reqs=[a, b],
                    prefix_lens=[0, 0],
                    extend_lens=[48, 48],
                    seq_lens=torch.tensor([48, 48]),
                )
                cache.prepare(batch, restore=restore)
                reserved = 2 * ((8 + 2 * page_size - 2) // page_size) * page_size
                self.assertEqual(alloc.swa_attn_allocator.allocated_count(), reserved)
                # The windows share their eight materialized tokens. Additional
                # bindings provision the bounded pool for asynchronous growth.
                self.assertEqual(
                    alloc.translate_loc_from_full_to_swa(table.req_to_token[:2, 40:48])
                    .unique()
                    .numel(),
                    8,
                )
                capture = SimpleNamespace(
                    entries=[
                        SimpleNamespace(owner=req.kv.dflash_context_state)
                        for req in (a, b)
                    ]
                )
                cache.retain_prefill(batch, capture)
                # The tree insert adopts canonical pages and records the rows.
                canonical = alloc.alloc(8)
                table.req_to_token[1, 40:48] = canonical
                b.kv.dflash_context_state.note_canonical_remap(40, 48)
                cache.prepare(batch, restore=restore)
                self.assertEqual(alloc.swa_attn_allocator.allocated_count(), reserved)
                values = draft.get_key_buffer(2)[
                    alloc.translate_loc_from_full_to_swa(canonical)
                ]
                torch.testing.assert_close(
                    values[:, 0, 0].float(), torch.arange(88, 96).float()
                )
                old_window = b.kv.dflash_context_state.snapshot()
                alloc.dflash_checkpoint_pool.revision += 1
                cache.prepare(batch, restore=restore)
                for req in (a, b):
                    self.assertEqual(req.kv.dflash_context_state.revision, 1)
                torch.testing.assert_close(
                    old_window.as_tensor()[:, 0], torch.arange(88, 96).float()
                )
                torch.testing.assert_close(
                    b.kv.dflash_context_state.snapshot().as_tensor()[:, 0],
                    torch.arange(1088, 1096).float(),
                )
                a.kv.req_pool_idx = None
                batch.reqs = [b]
                batch.prefix_lens = [48]
                batch.extend_lens = [16]
                batch.seq_lens = torch.tensor([64])
                cache.prepare(batch, restore=restore)
                # Retired bindings remain reusable within the fixed window cap.
                self.assertEqual(alloc.swa_attn_allocator.allocated_count(), reserved)
                self.assertEqual(alloc.full_attn_allocator.allocated_count(), 72)
                future = alloc.alloc(16)
                table.req_to_token[1, 64:80] = future
                limit = max(
                    reserved, ((8 + 2 + 2 * page_size - 2) // page_size) * page_size
                )
                for prefix in range(64, 77):
                    # Host progress trails the GPU by at most one verify block.
                    b.kv.kv_committed_len = prefix - 2
                    b.kv.kv_allocated_len = prefix + 2
                    cache.prepare(
                        batch,
                        prefix_lens=torch.tensor([prefix]),
                        block_size=2,
                        restore=restore,
                    )
                    window = table.req_to_token[1, prefix - 8 : prefix + 2]
                    self.assertTrue(
                        (
                            alloc.swa_attn_allocator.virtual_to_physical[
                                window // page_size
                            ]
                            > 0
                        ).all()
                    )
                    self.assertLessEqual(
                        alloc.swa_attn_allocator.allocated_count(), limit
                    )
                    self.assertEqual(alloc.verify_byte_accounting(), [])
                self.assertEqual(alloc.swa_attn_allocator.allocated_count(), limit)
                # Releasing pages with draft bindings runs during result
                # processing, right after donations queue their host copies.
                with _NoPageableCountReadback():
                    alloc.free_group_begin()
                    alloc.free_segment(ids, start_pos=0)
                    alloc.free_segment(canonical, start_pos=40)
                    alloc.free_segment(future, start_pos=64)
                    self.assertTrue(alloc.evict_to_free_tokens(None, 128))
                    alloc.free_group_end()
                self.assertEqual(alloc.full_attn_allocator.allocated_count(), 0)
                # Draft bindings outlive the freed target pages as spares, and
                # the next window reuses them without growing the reservation.
                self.assertEqual(alloc.swa_attn_allocator.allocated_count(), limit)
                self.assertEqual(alloc.verify_byte_accounting(), [])
                b.kv.req_pool_idx = None
                c = SimpleNamespace(kv=ReqKvInfo(req_pool_idx=0))
                fresh = alloc.alloc(16)
                table.req_to_token[0, :16] = fresh
                batch.reqs = [c]
                batch.seq_lens = torch.tensor([16])
                cache.prepare(batch, restore=restore)
                self.assertTrue(
                    (
                        alloc.swa_attn_allocator.virtual_to_physical[
                            table.req_to_token[0, 8:16] // page_size
                        ]
                        > 0
                    ).all()
                )
                self.assertEqual(alloc.swa_attn_allocator.allocated_count(), limit)
                self.assertEqual(alloc.verify_byte_accounting(), [])
                alloc.free(fresh)
                states = table.mamba_allocator.alloc(4)
                self.assertIsNotNone(states)
                self.assertIsNone(table.mamba_allocator.alloc(1))
                table.mamba_allocator.free(states)

    def test_model_views_and_recurrent_slot_wiring(self):
        """Each model keeps its own layer IDs, head geometry, and dtype;
        the target's recurrent state uses the shared slot allocator.
        """
        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                bundle = init_unified_mamba_swa_pools(
                    **self._factory_kwargs(page_size=page_size)
                )
                shared = bundle.unified_memory_pool
                target = bundle.token_to_kv_pool
                draft = bundle.draft_token_to_kv_pool
                req_pool = bundle.req_to_token_pool
                alloc = bundle.token_to_kv_pool_allocator
                self.assertIsInstance(target, UnifiedHybridLinearKVPool)
                self.assertIsInstance(draft, UnifiedMHATokenToKVPool)
                self.assertIs(alloc.get_kvcache(), target)
                self.assertIs(target.mamba_pool, req_pool.mamba_pool)
                ids = alloc.alloc(2 * page_size)
                self.assertIsNotNone(ids)
                buffers = []
                for pool, layers, translate, shape, dtype, base in (
                    (
                        target,
                        (5, 7),
                        alloc.translate_kv_loc,
                        (2, 4, 4),
                        torch.float16,
                        20,
                    ),
                    (
                        draft,
                        range(2, 8),
                        alloc.translate_loc_from_full_to_swa,
                        (1, 8, 16),
                        torch.bfloat16,
                        100,
                    ),
                ):
                    for layer_id in layers:
                        for offset, (get_buffer, head_dim) in enumerate(
                            (
                                (pool.get_key_buffer, shape[1]),
                                (pool.get_value_buffer, shape[2]),
                            )
                        ):
                            buf = get_buffer(layer_id)
                            self.assertEqual(buf.shape[1:], (shape[0], head_dim))
                            self.assertEqual(buf.dtype, dtype)
                            self.assertEqual(
                                buf.untyped_storage().data_ptr(), shared._raw.data_ptr()
                            )
                            value = base + layer_id + offset
                            buf[translate(ids)] = value
                            buffers.append((buf, translate(ids), value))
                for buf, physical_ids, value in buffers:
                    self.assertTrue(bool((buf[physical_ids] == value).all()))

                state_ids = req_pool.mamba_allocator.alloc(2)
                self.assertIsNotNone(state_ids)
                physical_state_ids = alloc.mamba_allocator.translate_kv_loc(state_ids)
                torch.testing.assert_close(
                    req_pool.translate_mamba_indices(state_ids),
                    physical_state_ids.to(torch.int32),
                )
                torch.testing.assert_close(
                    target._mamba_translate(state_ids), physical_state_ids
                )
                req_pool.mamba_allocator.free(state_ids)
                alloc.free(ids)
                self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_draft_capacity_and_geometry_determine_byte_budget(self):
        # Full: 2 layers * 2 heads * (4 K + 4 V) * 2 bytes = 64 bytes/token.
        # Draft: 6 layers * 1 head * (8 K + 16 V) * 2 bytes = 288 bytes/token.
        # State: 2 layers * (3*8 BF16 conv + 2*2*4 FP32 recurrent) = 224 bytes.
        for token_budget in (None, 20_000):
            for draft_tokens in (16, 32):
                with self.subTest(token_budget=token_budget, draft_tokens=draft_tokens):
                    bundle = init_unified_mamba_swa_pools(
                        **self._factory_kwargs(
                            swa_max_total_num_tokens=draft_tokens,
                            unified_total_bytes=token_budget,
                        )
                    )
                    token_bytes = (
                        64 * 64 + draft_tokens * 288
                        if token_budget is None
                        else token_budget
                    )
                    self.assertEqual(
                        bundle.unified_memory_pool.total_bytes,
                        token_bytes + 4 * 224,
                    )
                    target = bundle.token_to_kv_pool.full_kv_pool
                    draft = bundle.draft_token_to_kv_pool
                    self.assertEqual(target.host_capacity_tokens, 64)
                    self.assertEqual(target.host_capacity_bytes, 64 * 64)
                    self.assertEqual(draft.host_capacity_tokens, draft_tokens)
                    self.assertEqual(draft.host_capacity_bytes, draft_tokens * 288)

    def test_bounded_recurrent_slots_do_not_reduce_full_token_budget(self):
        bundle = init_unified_mamba_swa_pools(
            **self._factory_kwargs(
                bounded_draft_swa=True,
                unified_total_bytes=20_000,
                swa_max_total_num_tokens=16,
            )
        )
        allocator = bundle.token_to_kv_pool_allocator
        states = bundle.req_to_token_pool.mamba_allocator
        full_capacity = allocator.available_size()
        state_ids = states.alloc(states.size)
        self.assertIsNotNone(state_ids)
        self.assertEqual(states.schedulable_available_size(), 0)
        self.assertEqual(allocator.available_size(), full_capacity)
        self.assertEqual(allocator.mamba_slot_full_token_cost(), 0)
        tokens = allocator.alloc(full_capacity)
        self.assertIsNotNone(tokens)
        self.assertTrue(allocator.ensure_capacity(0, 16))
        allocator.swa_attn_allocator.alloc_with_virtual(tokens[:16])
        self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 16)
        allocator.free(tokens)
        states.free(state_ids)
        self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_bounded_pool_accounting_when_draft_splits_free_space(self):
        """A live draft region must not appear as leaked target tokens."""
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            SchedulerInvariantChecker,
        )

        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                bundle = init_unified_mamba_swa_pools(
                    **self._factory_kwargs(
                        bounded_draft_swa=True,
                        full_max_total_num_tokens=1024,
                        swa_max_total_num_tokens=16,
                        unified_total_bytes=80_000,
                        page_size=page_size,
                    )
                )
                allocator = bundle.token_to_kv_pool_allocator
                states = bundle.req_to_token_pool.mamba_allocator
                state_ids = states.alloc(states.size)
                tokens = allocator.alloc(32)
                self.assertTrue(allocator.ensure_capacity(0, 16))
                allocator.swa_attn_allocator.alloc_with_virtual(
                    tokens[:16:page_size] // page_size
                )
                self.assertLess(allocator.available_size(), 1024 - 32)
                for cached, expected_leak in ((32, False), (32 - page_size, True)):
                    leak, message = SchedulerInvariantChecker._check_pool_invariant(
                        "full", allocator.available_size(), cached, 0, 0, allocator.size
                    )
                    self.assertEqual(leak, expected_leak, message)

                # The fixed admission cap remains usable after moving the draft
                # region; reporting current capacity must not shrink that budget.
                remaining = allocator.alloc(1024 - 32)
                self.assertIsNotNone(remaining)
                self.assertIsNone(allocator.alloc(page_size))
                allocator.free(torch.cat((tokens, remaining)))
                states.free(state_ids)
                self.assertEqual(allocator.available_size(), 1024)
                self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_rejects_ambiguous_draft_layouts(self):
        region = DenseDraftRegion(
            lane_num=1, head_num=2, head_dim=4, store_dtype=torch.float16
        )
        fused = FusedDraftPlacement.from_counts(
            counts={"full": [1]}, regions={"full": region}
        )
        for overrides in (
            {"draft_swa_layer_ids": None},
            {"draft_kv_cache_dtype": None},
            {"draft_swa_layer_ids": []},
            {"draft_swa_layer_ids": [2, 4]},
            {"draft_swa_layer_ids": [3, 2]},
            {"draft_swa_layer_ids": [-1, 0]},
            {"swa_attention_layer_ids": [4]},
            {"fused_draft": fused},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                init_unified_mamba_swa_pools(**self._factory_kwargs(**overrides))

    def _draft_configurator(self, page_size):
        cfg = KVCacheConfigurator.__new__(KVCacheConfigurator)
        cfg.is_draft_worker = True
        cfg.draft_model_idx = None
        cfg.page_size = page_size
        cfg.kv_cache_dtype = torch.bfloat16
        cfg.model = torch.nn.ModuleList(
            RadixAttention(
                num_heads=2,
                num_kv_heads=1,
                head_dim=8,
                v_head_dim=16,
                scaling=1,
                layer_id=layer_id,
                sliding_window_size=8,
            )
            for layer_id in range(2, 8)
        )
        return cfg

    def test_draft_binding_requires_opt_in_and_supports_gpu_only(self):
        bundle = init_unified_mamba_swa_pools(**self._factory_kwargs(page_size=4))
        with get_spec().override(enable_dflash_swa_cache=False):
            with self.assertRaisesRegex(ValueError, "--enable-dflash-swa-cache"):
                self._draft_configurator(page_size=4)._init_pools(
                    sizes=None,
                    req_to_token_pool=bundle.req_to_token_pool,
                    token_to_kv_pool_allocator=bundle.token_to_kv_pool_allocator,
                )
        with get_memory().override(enable_hierarchical_cache=False):
            pools = self._draft_configurator(page_size=4)._init_pools(
                sizes=None,
                req_to_token_pool=bundle.req_to_token_pool,
                token_to_kv_pool_allocator=bundle.token_to_kv_pool_allocator,
            )
            self.assertIs(pools.token_to_kv_pool, bundle.draft_token_to_kv_pool)

    def test_draft_binding_and_shared_plan_use_swa_physical_ids(self):
        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                bundle = init_unified_mamba_swa_pools(
                    **self._factory_kwargs(page_size=page_size)
                )
                alloc = bundle.token_to_kv_pool_allocator
                cfg = self._draft_configurator(page_size)
                pools = cfg._init_pools(
                    sizes=None,
                    req_to_token_pool=bundle.req_to_token_pool,
                    token_to_kv_pool_allocator=alloc,
                )
                self.assertIs(pools.token_to_kv_pool, bundle.draft_token_to_kv_pool)
                self.assertIs(pools.req_to_token_pool, bundle.req_to_token_pool)
                self.assertIs(pools.token_to_kv_pool_allocator, alloc)
                self.assertIsNone(pools.unified_memory_pool)

                ids = alloc.alloc(3 * page_size)
                self.assertIsNotNone(ids)
                req_to_token = pools.req_to_token_pool.req_to_token
                req_to_token[0, : ids.numel()] = ids.to(req_to_token.dtype)
                target, draft = (
                    KVIndexTranslator(
                        req_to_token=req_to_token,
                        token_to_kv_pool_allocator=alloc,
                        token_to_kv_pool=pool,
                        page_size=page_size,
                        device=_DEV,
                        is_draft_worker=is_draft,
                    )
                    for pool, is_draft in (
                        (bundle.token_to_kv_pool, False),
                        (pools.token_to_kv_pool, True),
                    )
                )
                self.assertTrue(draft.is_translating)
                self.assertEqual(draft.window_kv_kind, IdSpaceKind.SLIDING_WINDOW)
                self.assertEqual(
                    draft.capture_token_capacity(1),
                    alloc.full_v2p_page_table.numel() * page_size,
                )
                values = torch.arange(ids.numel(), dtype=torch.bfloat16) + 10
                buffer = pools.token_to_kv_pool.get_key_buffer(2)
                buffer[alloc.translate_loc_from_full_to_swa(ids)] = values[
                    :, None, None
                ]

                def check_plans():
                    expected_full = alloc.translate_kv_loc(ids)
                    expected_swa = alloc.translate_loc_from_full_to_swa(ids)
                    self.assertFalse(torch.equal(expected_full, expected_swa))
                    for source in (target, draft):
                        plan = source.plan(
                            req_pool_indices=torch.tensor([0]),
                            seq_lens=torch.tensor([ids.numel()]),
                            seq_lens_cpu=torch.tensor([ids.numel()]),
                            write_virtual=ids,
                        )
                        for reader, expected in (
                            (target, expected_full),
                            (draft, expected_swa),
                        ):
                            batch = SimpleNamespace()
                            plan.bind(batch, reader)
                            torch.testing.assert_close(batch.out_cache_loc, expected)
                            self.assertIs(batch.out_cache_loc_virtual, ids)
                            # Exercise the gather source before and after a table reader.
                            view = reader.read_source(
                                plan, req_pool_indices=plan.req_pool_indices, bs=1
                            )
                            self.assertIs(
                                view.v2p, reader.space(IdSpaceKind.FULL).read_v2p
                            )
                            table = reader.read_table(plan)
                            torch.testing.assert_close(
                                table.ids[0].long(), expected[::page_size] // page_size
                            )
                            self.assertIs(
                                reader.read_source(
                                    plan, req_pool_indices=plan.req_pool_indices, bs=1
                                ).ids,
                                table.ids,
                            )
                        self.assertIs(
                            plan.write_ids(draft),
                            plan.write_ids(target, kind=IdSpaceKind.SLIDING_WINDOW),
                        )
                        self.assertIs(
                            draft.read_table(plan).ids,
                            target.read_table(
                                plan, kind=IdSpaceKind.SLIDING_WINDOW
                            ).ids,
                        )
                        capture_table = torch.full((2, 3), -1, dtype=torch.int32)
                        draft.copy_page_table(plan, out=capture_table)
                        torch.testing.assert_close(
                            capture_table[0].long(),
                            expected_swa[::page_size] // page_size,
                        )
                        self.assertTrue(bool((capture_table[1] == 0).all()))

                check_plans()
                # Reclaiming the draft prefix leaves the target's IDs live.
                alloc.free_swa(ids[:page_size])
                check_plans()
                alloc.swa_attn_allocator.alloc_with_virtual(ids[:1] // page_size)
                buffer[alloc.translate_loc_from_full_to_swa(ids[:page_size])] = values[
                    :page_size, None, None
                ]
                swa = alloc.swa_attn_allocator
                before = alloc.translate_loc_from_full_to_swa(ids)
                swa.make_room(
                    side="low",
                    min_bytes=swa._byte_low_frontier()
                    - swa._chain_high_frontier_below_bytes()
                    + swa.entry_bytes_per_page,
                )
                self.assertFalse(
                    torch.equal(before, alloc.translate_loc_from_full_to_swa(ids))
                )
                check_plans()
                torch.testing.assert_close(
                    buffer[alloc.translate_loc_from_full_to_swa(ids), 0, 0], values
                )
                # Capture slots already name the draft's storage; no second translation.
                slots = alloc.translate_loc_from_full_to_swa(ids)
                capture = SimpleNamespace(out_cache_loc=slots)
                draft.bind_runner_slots(capture)
                self.assertIs(
                    draft.write_ids(capture, IdSpaceKind.SLIDING_WINDOW), slots
                )
                alloc.free(ids)
                self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_draft_binding_rejects_model_or_storage_mismatch(self):
        bundle = init_unified_mamba_swa_pools(**self._factory_kwargs(page_size=4))
        bundle.token_to_kv_pool_allocator.dflash_cache_sizing = SimpleNamespace(
            window_size=9
        )
        mutations = {
            "page size": lambda cfg: setattr(cfg, "page_size", 1),
            "dtype": lambda cfg: setattr(cfg, "kv_cache_dtype", torch.float16),
            "second runner": lambda cfg: setattr(cfg, "draft_model_idx", 1),
            "layer IDs": lambda cfg: setattr(cfg.model[0], "layer_id", 0),
            "K width": lambda cfg: setattr(cfg.model[0], "head_dim", 16),
            "V width": lambda cfg: setattr(cfg.model[0], "v_head_dim", 8),
            "K heads": lambda cfg: setattr(cfg.model[0], "tp_k_head_num", 2),
            "V heads": lambda cfg: setattr(cfg.model[0], "tp_v_head_num", 2),
            "full attention": lambda cfg: setattr(
                cfg.model[0], "sliding_window_size", -1
            ),
            "window": lambda cfg: setattr(cfg.model[0], "sliding_window_size", 7),
            "recurrent state": lambda cfg: cfg.model.append(
                RadixLinearAttention(
                    layer_id=8,
                    num_q_heads=1,
                    num_k_heads=1,
                    num_v_heads=1,
                    head_q_dim=8,
                    head_k_dim=8,
                    head_v_dim=8,
                )
            ),
        }
        for name, mutation in mutations.items():
            cfg = self._draft_configurator(page_size=4)
            mutation(cfg)
            with (
                self.subTest(mismatch=name),
                self.assertRaisesRegex(ValueError, "draft SWA"),
            ):
                cfg._init_pools(
                    sizes=None,
                    req_to_token_pool=bundle.req_to_token_pool,
                    token_to_kv_pool_allocator=bundle.token_to_kv_pool_allocator,
                )
        with self.assertRaisesRegex(ValueError, "request table"):
            self._draft_configurator(page_size=4)._init_pools(
                sizes=None,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=bundle.token_to_kv_pool_allocator,
            )

    def test_draft_plan_rejects_foreign_owner_and_unbound_views(self):
        bundle, foreign = (
            init_unified_mamba_swa_pools(**self._factory_kwargs()) for _ in range(2)
        )
        alloc = bundle.token_to_kv_pool_allocator
        ids = alloc.alloc(2)
        req_to_token = bundle.req_to_token_pool.req_to_token
        target, draft, foreign_draft, unbound = (
            KVIndexTranslator(
                req_to_token=req_to_token,
                token_to_kv_pool_allocator=allocator,
                token_to_kv_pool=pool,
                page_size=1,
                device=_DEV,
            )
            for allocator, pool in (
                (alloc, bundle.token_to_kv_pool),
                (alloc, bundle.draft_token_to_kv_pool),
                (foreign.token_to_kv_pool_allocator, foreign.draft_token_to_kv_pool),
                (
                    alloc,
                    UnifiedMHATokenToKVPool(
                        unified_buffer=bundle.unified_memory_pool, sub_pool_name="swa"
                    ),
                ),
            )
        )
        plan = target.plan(
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([2]),
            seq_lens_cpu=torch.tensor([2]),
            write_virtual=ids,
        )
        self.assertTrue(plan.is_read_by(draft))
        # The row tensor alone cannot establish logical-ID ownership.
        with self.assertRaisesRegex(AssertionError, "own.*req_to_token"):
            foreign_draft.read_table(plan)
        # Sharing the raw buffer alone cannot establish a physical mapping.
        self.assertFalse(unbound.is_translating)
        self.assertIs(plan.write_ids(unbound), ids)
        alloc.free(ids)


class TestTriPoolHardening(unittest.TestCase):
    """Pressure lanes: the planned-rebalance remedy in the alloc path (a
    mis-positioned float must not fail an alloc that fits in total bytes),
    retract-loop convergence through `check_decode_capacity`, and bounded copy
    traffic under alternating end pressure.
    """

    def _build(self, **kw):
        return TestUnifiedTriPool._build(self, **kw)

    def test_alloc_rebalances_a_blocking_float(self):
        # Fill much of the high band so the midpoint-placed float walls the
        # low band's free bytes off from `full`; the next alloc must SLIDE it
        # rather than fail while total bytes suffice.
        _, allocator, kvcache, _ = self._build(n_full=32, n_swa=24, n_state=8)
        sa = allocator.swa_attn_allocator
        v0 = allocator.alloc(4)  # places the float at the region midpoint
        self.assertIsNotNone(v0)
        TestUnifiedTriPool._stamp(self, allocator, kvcache, v0)
        # Exhaust the high band directly on the full end (full-only growth,
        # e.g. long decode of already-admitted requests).
        fa = allocator.full_attn_allocator
        b_high_pages = fa._current_gap_bytes() // fa.entry_bytes_per_page
        grab = fa.alloc(max(0, (b_high_pages - 2)))
        self.assertIsNotNone(grab)
        # The gate under-reports (no slide credit), so asking BEYOND it fires
        # the remedy; the ask still fits the free bytes the LOW band holds.
        avail = allocator.available_size()
        need = avail + 4
        live_before = sa._live_pages()
        moves_before = len(sa._inverse_history)
        v1 = allocator.alloc(need)
        self.assertIsNotNone(
            v1, "alloc must rebalance the blocking float instead of failing"
        )
        self.assertEqual(int(v1.numel()), need)
        moved = sum(int(s.numel()) for s, _, _ in sa._inverse_history[moves_before:])
        self.assertGreater(moved, 0, "the rebalance path must have fired")
        # Cost bound min(L_live, G): never more than the live pages present
        # when the slide ran (the leapfrog cap).
        self.assertLessEqual(moved, live_before)
        self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_check_decode_capacity_retract_convergence(self):
        # Retract loop: token blocks freed one at a time until the next step
        # fits; must converge before bs=1 and never report capacity early.
        _, allocator, _, _ = self._build(n_full=32, n_swa=24, n_state=8)
        reqs = []
        while True:
            v = allocator.alloc(4)
            if v is None or allocator.available_size() < 4:
                if v is not None:
                    reqs.append(v)
                break
            reqs.append(v)
        self.assertGreater(len(reqs), 2)
        # Pool saturated: a large decode step does not fit.
        need = 16
        while not allocator.check_decode_capacity(num_tokens=need, tree_cache=None):
            self.assertGreater(len(reqs), 1, "retract must converge before bs=1")
            allocator.free(reqs.pop())
        self.assertGreaterEqual(allocator.available_size(), need)
        self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_alternating_pressure_copy_traffic_bounded(self):
        # Alternating full-grow / swa-churn: hole recycling and absorption do
        # the steady-state work, so total float moves stay bounded.
        _, allocator, kvcache, _ = self._build(n_full=48, n_swa=32, n_state=8)
        sa = allocator.swa_attn_allocator
        fa = allocator.full_attn_allocator
        total_alloc_pages = 0
        for _ in range(6):
            v = allocator.alloc(8)
            self.assertIsNotNone(v)
            total_alloc_pages += 8
            TestUnifiedTriPool._stamp(self, allocator, kvcache, v)
            allocator.free_swa(v)  # window slide: tombstones -> holes/absorb
            g = fa.alloc(4)  # full-side decode growth
            self.assertIsNotNone(g)
            total_alloc_pages += 4
            fa.free(g)
        moved = sum(int(s.numel()) for s, _, _ in sa._inverse_history)
        self.assertLessEqual(
            moved,
            total_alloc_pages // 2,
            "steady-state churn must be predominantly zero-copy",
        )
        self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_joint_eviction_loop_stops_without_progress(self):
        # tree_cache=None: the default helper no-ops; the bounded loop must
        # return promptly (no infinite re-check) and the gate reports honestly.
        _, allocator, _, _ = self._build()
        big = allocator.available_size() + 64
        self.assertFalse(
            allocator.check_decode_capacity(num_tokens=big, tree_cache=None)
        )


class TestJointCapacityIsHonoured(unittest.TestCase):
    """`alloc(available_size())` must never fail.

    REGRESSION: the joint predicate priced the swa float's extension in RAW
    BYTES, but `take_physical_pages` yields only whole pages on the float's
    OWN grid, whose low edge is rounded up to a multiple of the NEIGHBOUR's
    entry size -- so the budget credited a page the grid could not yield.
    Swept over geometries so a symmetric mistake on the FULL side surfaces
    here too.
    """

    def _build(self, *, page_size, n_full, n_swa, n_state, lazy, specs):
        full, swa, mamba = specs
        total = (
            n_full * full.entry_bytes()
            + n_swa * swa.entry_bytes()
            + n_state * mamba.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa, mamba],
            device=_DEV,
            enable_memory_saver=False,
            page_size=page_size,
        )
        kvcache = FakeUnifiedSWAKVPool(pool)
        return pool, UnifiedMambaSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            mamba_kvcache=FakeKVCache(pool.max_slots("mamba")),
            device=_DEV,
            full_max_total_num_tokens=n_full,
            swa_max_total_num_tokens=n_swa,
            need_sort=False,
            forward_stream=None,
            lazy_compaction=lazy,
        )

    def test_fresh_boot_alloc_of_available_size_succeeds(self):
        # Geometries chosen so the mamba end's frontier (a multiple of the
        # STATE entry size) lands off the swa float's page grid.
        for page_size in (1, 2, 4):
            for fl, sl, ml in ((4, 3, 1), (4, 2, 2), (6, 3, 1), (3, 5, 2)):
                for n_full, n_swa, n_state in ((24, 16, 4), (32, 16, 8), (20, 12, 6)):
                    for lazy in (False, True):
                        specs = tri_sub_pool_specs(
                            full_layer_num=fl,
                            swa_layer_num=sl,
                            state_layer_num=ml,
                            head_num=1,
                            head_dim=8,
                        )
                        with self.subTest(
                            ps=page_size,
                            layers=(fl, sl, ml),
                            n=(n_full, n_swa, n_state),
                            lazy=lazy,
                        ):
                            _pool, alloc = self._build(
                                page_size=page_size,
                                n_full=n_full * page_size,
                                n_swa=n_swa * page_size,
                                n_state=n_state,
                                lazy=lazy,
                                specs=specs,
                            )
                            n = alloc.available_size()
                            if n <= 0:
                                continue
                            # The whole point: the number the scheduler reads
                            # must be allocatable, with no backstop assert.
                            out = alloc.alloc(n)
                            self.assertIsNotNone(
                                out,
                                f"alloc(available_size()={n}) returned None",
                            )
                            self.assertEqual(out.numel(), n)
                            # Both sides bound for every allocated virtual id.
                            fa = alloc.full_attn_allocator
                            sa = alloc.swa_attn_allocator
                            self.assertTrue(
                                bool((fa.virtual_to_physical[out] >= 0).all())
                            )
                            self.assertTrue(
                                bool((sa.virtual_to_physical[out] >= 0).all())
                            )

    def test_available_size_never_exceeds_the_float_page_grid(self):
        """Direct form: the joint answer, converted to float pages, must fit
        inside what `_region_bounds_pages` actually offers."""
        for page_size in (1, 4):
            specs = tri_sub_pool_specs(
                full_layer_num=4,
                swa_layer_num=3,
                state_layer_num=1,
                head_num=1,
                head_dim=8,
            )
            _pool, alloc = self._build(
                page_size=page_size,
                n_full=24 * page_size,
                n_swa=16 * page_size,
                n_state=4,
                lazy=False,
                specs=specs,
            )
            sa = alloc.swa_attn_allocator
            n_pages = alloc.available_size() // page_size
            lo, hi = sa._region_bounds_pages()
            with self.subTest(ps=page_size):
                self.assertLessEqual(
                    n_pages - sa._hole_pages(),
                    max(0, hi - lo),
                    "joint available_size() promises more float pages than the "
                    "float's own page grid can yield",
                )


class TestFloatRelocationIsOrderedAgainstTheForward(unittest.TestCase):
    """Float relocation must settle the in-flight forward BEFORE its first copy.

    REGRESSION: `make_room` / `compact_holes` issued `move_kv_cache` and
    rebound `virtual_to_physical` with no ordering against the running
    forward, so a copy could carry pre-write bytes and every later reader saw
    a destination that never received those writes -- silently wrong KV, no
    crash. The END pools guard the same hazard via `_settle_inflight_forward`.
    """

    def _tri(self, lazy=True):
        full, swa, mamba = tri_sub_pool_specs(head_num=1, head_dim=8)
        total = (
            48 * full.entry_bytes() + 32 * swa.entry_bytes() + 8 * mamba.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = FakeUnifiedSWAKVPool(pool)
        alloc = UnifiedMambaSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            mamba_kvcache=FakeKVCache(pool.max_slots("mamba")),
            device=_DEV,
            full_max_total_num_tokens=48,
            swa_max_total_num_tokens=32,
            need_sort=False,
            forward_stream=None,
            lazy_compaction=lazy,
        )
        return pool, alloc, kvcache

    def _trace(self, flt):
        """Record the order of (settle, move) on the float."""
        order = []
        real_settle = flt._settle_inflight_forward
        real_move = flt._move_pages_and_rebind

        def settle():
            order.append("settle")
            return real_settle()

        def move(src, dst):
            order.append("move")
            return real_move(src, dst)

        flt._settle_inflight_forward = settle
        flt._move_pages_and_rebind = move
        return order

    def test_make_room_settles_before_the_first_move(self):
        _pool, alloc, _kv = self._tri()
        flt = alloc.swa_attn_allocator
        # Occupy the float, then free an interior page so a relocation has
        # something to move and somewhere to move it.
        v = alloc.alloc(12)
        self.assertIsNotNone(v)
        alloc.free(v[:4])
        order = self._trace(flt)
        flt.make_room(side="low", min_bytes=flt.entry_bytes_per_page)
        self.assertIn("settle", order, "make_room never settled the forward")
        if "move" in order:
            self.assertLess(
                order.index("settle"),
                order.index("move"),
                f"a copy was issued before the settle: {order}",
            )

    def test_compact_holes_settles_before_the_first_move(self):
        _pool, alloc, _kv = self._tri()
        flt = alloc.swa_attn_allocator
        v = alloc.alloc(12)
        self.assertIsNotNone(v)
        alloc.free(v[2:6])  # interior holes, so compact_holes has work
        order = self._trace(flt)
        flt.compact_holes(retreat_side="high")
        if not order:
            self.skipTest("no holes reached compact_holes in this geometry")
        self.assertEqual(order[0], "settle", f"first action was not a settle: {order}")

    def test_the_settle_is_a_stream_wait_not_a_host_sync(self):
        """The settle before a float move must be a stream wait: a host sync
        there would land on the alloc-shortfall path every time the float moves."""
        src = inspect.getsource(MultiEndedAllocator._settle_inflight_forward)
        self.assertIn("wait_event", src)
        self.assertNotIn(".item()", src)
        self.assertNotIn("synchronize()", src)


class TestFloatHoleCreditIsPerSide(unittest.TestCase):
    """A float's schedulable credit must follow the side the holes are on.

    REGRESSION: the base `_peer_drainable_hole_bytes` picks its neighbour from
    `grow_direction`, which for a float is "float" -- it fell through to
    `low_peer`, never saw the HIGH neighbour, and its single scalar was added
    to `max(gap_low, gap_high)`, landing a LOW neighbour's holes on the HIGH
    gap. Over-reporting `schedulable_available_size` admits work the shortfall
    ladder cannot satisfy.
    """

    def _float(self):
        full, swa, mamba = tri_sub_pool_specs(head_num=1, head_dim=8)
        total = (
            48 * full.entry_bytes() + 32 * swa.entry_bytes() + 8 * mamba.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        alloc = UnifiedMambaSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=FakeUnifiedSWAKVPool(pool),
            mamba_kvcache=FakeKVCache(pool.max_slots("mamba")),
            device=_DEV,
            full_max_total_num_tokens=48,
            swa_max_total_num_tokens=32,
            need_sort=False,
            forward_stream=None,
            lazy_compaction=True,
        )
        return alloc, alloc.swa_attn_allocator

    def test_schedulable_never_exceeds_the_sum_of_the_two_sides(self):
        """Upper bound that the undirected scalar could violate: no side may be
        credited with the other side's holes on top of its own gap."""
        alloc, flt = self._float()
        v = alloc.alloc(10)
        self.assertIsNotNone(v)
        alloc.free(v[:3])
        epp = flt.entry_bytes_per_page
        gap_low, gap_high = flt._gap_pages()
        c_low = flt._side_drainable_hole_bytes("low") // epp
        c_high = flt._side_drainable_hole_bytes("high") // epp
        bound = (
            min(
                max(gap_low + c_low, gap_high + c_high),
                flt.num_pages - flt.min_page_index - flt._live_pages(),
            )
            + flt._hole_pages()
        ) * flt.page_size
        self.assertLessEqual(flt.schedulable_available_size(), bound)

    def test_memo_verifier_agrees_with_the_per_side_formula(self):
        """The staleness verifier recomputes through the same entry, so the
        override must not make the memo look stale."""
        _alloc, flt = self._float()
        flt.available_size()
        flt.schedulable_available_size()
        self.assertEqual(flt._byte_accounting_violations(), [])


class TestPreallocIsPricedOnTheSharedGrid(unittest.TestCase):
    """REGRESSION: PD admission compared FULL and SWA against per-side token
    budgets, but each side's `available_size` credits the peer's drainable
    holes, so a pair that each side can host alone can be jointly infeasible.
    Such a pair was admitted and then refused inside `alloc_extend_swa_tail`."""

    def _build(self, **kw):
        return TestUnifiedTriPool._build(self, **kw)

    def test_a_pair_each_side_can_host_alone_is_still_refused(self):
        # page_size 1 leaves no slack between the per-side and joint views;
        # the double-count only has room to show on a paged grid.
        _, allocator, _, _ = self._build(page_size=4)
        full_demand = allocator.full_available_size()
        swa_demand = allocator.swa_available_size()
        self.assertGreater(min(full_demand, swa_demand), 0)
        # Each side alone reports room for its own half ...
        self.assertLessEqual(full_demand, allocator.full_available_size())
        self.assertLessEqual(swa_demand, allocator.swa_available_size())
        # ... yet the two draw on the same bytes, so the grid refuses the pair.
        self.assertFalse(
            allocator._fits_page_demand(
                -(-full_demand // allocator.page_size),
                -(-swa_demand // allocator.page_size),
            )
        )
        self.assertFalse(
            allocator.prealloc_fits(
                MagicMock(),
                full_demand,
                swa_demand,
                full_budget_tokens=full_demand,
                swa_budget_tokens=swa_demand,
            )
        )

    def test_the_scheduler_budget_still_binds(self):
        _, allocator, _, _ = self._build()
        page_size = allocator.page_size
        self.assertTrue(
            allocator.prealloc_fits(
                MagicMock(),
                page_size,
                page_size,
                full_budget_tokens=page_size,
                swa_budget_tokens=page_size,
            )
        )
        self.assertFalse(
            allocator.prealloc_fits(
                MagicMock(),
                page_size,
                page_size,
                full_budget_tokens=page_size - 1,
                swa_budget_tokens=page_size,
            )
        )


if __name__ == "__main__":
    unittest.main()
