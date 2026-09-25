"""Staged L3 prefetch lifecycle through the buffer pipeline; no GPU kernels."""

import tempfile
import unittest
from array import array
from collections import defaultdict, deque
from datetime import timedelta
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodePrefixMatch,
)
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    InitLoadBackParams,
)
from sglang.srt.mem_cache.buffer_mode.pipeline import (
    BufferModePipeline,
    _StagedPrefetch,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.multimodal_key import MultimodalKeySpan, slice_mm_spans
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_cache.components import BASE_COMPONENT_TYPE
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _OngoingPrefetch,
)
from sglang.srt.mem_cache.utils import get_hash_str, get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_REQ = CacheRequestHandle("r", 0)


def _staged_fixture(full_match=2):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(
        page_size=2,
        is_eagle=False,
        enable_storage=True,
        prefetch_anchor_info=lambda node: (None, None),
        match_full_device_prefix=Mock(return_value=(full_match, 1, full_match)),
        collect_full_device_indices=Mock(return_value=torch.arange(8)),
        inc_full_pin=Mock(),
        dec_full_pin=Mock(),
        empty_match_result=SimpleNamespace(
            last_device_node=0, device_indices=torch.arange(0)
        ),
    )
    cache.host_memory_mode = "buffer_only"
    cache.linker = None
    cache.storage_prefetch_retries = StoragePrefetchRetries()
    cache.prefetch_loaded_tokens_by_reqid = {_REQ: 6}
    cache.prefetch_loaded_storage_start_by_reqid = {_REQ: 2}
    cache._storage_prefetch_hit_remaining_by_reqid = {}
    cache.enable_storage_metrics = False
    cache.storage_metrics_collector = None
    cache.ongoing_prefetch = {}
    cache._prefetch_outcome_stats = defaultdict(int)
    cache.tree_components = []
    cache.prefetch_threshold = 2
    cache._build_sidecar_transfers = Mock(return_value=[])
    cache.supports_swa = lambda: True
    cache.evict_for_alloc = Mock()
    cache.token_to_kv_pool_allocator = SimpleNamespace(
        full_available_size=Mock(return_value=100)
    )
    cc = HybridCacheController.__new__(HybridCacheController)
    cc.page_size = 2
    cc.get_hash_str = get_hash_str
    cc.prefetch_queue = Queue()
    cc.prefetch_tokens_occupied = 6
    cc.prefetch_rate_limited = lambda: False
    cc.load = Mock(return_value=None)
    cc.storage_backend = Mock()
    cc.storage_backend.batch_exists.return_value = 0
    cc.mem_pool_host = SimpleNamespace(
        free=Mock(),
        entry_map={
            PoolName.SWA: SimpleNamespace(host_pool=SimpleNamespace(free=Mock()))
        },
    )
    cache.cache_controller = cc
    pipeline = BufferModePipeline.__new__(BufferModePipeline)
    pipeline._cache = cache
    pipeline.reset()
    cache.buffer_pipeline = pipeline
    pipeline.staged_prefetches[_REQ] = _StagedPrefetch(
        request=_REQ,
        key_tokens=array("q", range(8)),
        extra_key=None,
        cache_salt=None,
        matched_len=2,
        num_tokens=6,
        occupied_tokens=6,
        host_indices=torch.arange(6),
        aux_xfers=[PoolTransfer(name=PoolName.SWA, host_indices=torch.arange(4))],
        hash_values=["a", "b", "c"],
        operation_id=1,
    )
    req = SimpleNamespace(
        rid="r",
        cache_request_handle=_REQ,
        prefix_indices=torch.arange(2),
        last_node=1,
        kv=SimpleNamespace(cache_protected_len=2),
        extra_key=None,
        cache_salt=None,
        host_hit_length=0,
        swa_host_hit_length=0,
        host_hit_is_storage=False,
        host_loaded_length=0,
        storage_prefetch_last_match_len=4,
        storage_prefetch_retry_attempts=0,
    )
    return cache, pipeline, req


def _hit_drain_fixture():
    """A buffer-mode cache whose hit drain and outcome accounting are real."""
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.host_memory_mode = "buffer_only"
    cache.prefetch_threshold = 2
    cache.enable_storage_metrics = False
    cache.storage_metrics_collector = None
    cache.storage_prefetch_retries = StoragePrefetchRetries()
    cache._prefetch_outcome_stats = defaultdict(int)
    cache._storage_prefetch_hit_remaining_by_reqid = {}
    cache._record_storage_prefetch_hit = Mock()
    cache.revoke_pending_prefetch = Mock()
    cache.buffer_pipeline = SimpleNamespace(pending_hit_allocs=deque())
    cache.cache_controller = SimpleNamespace(
        prefetch_hit_queue=Queue(),
        ack_prefetch_queue=Queue(),
        ack_backup_queue=Queue(),
        host_mem_release_queue=Queue(),
        extra_host_mem_release_queues={},
    )
    cache.ongoing_prefetch = {}
    return cache


def _terminated_query(cache, rid, hit_tokens):
    handle = CacheRequestHandle(rid, 0)
    operation = SimpleNamespace(
        request_id=rid,
        handle=handle,
        storage_hit_count=hit_tokens,
        stats_requested_tokens=8,
        is_terminated=lambda: True,
    )
    cache.ongoing_prefetch[handle] = _OngoingPrefetch(
        0, RadixKey(array("q", range(8))), None, operation, None, {}
    )
    cache.cache_controller.prefetch_hit_queue.put(operation)


def _two_rank_retry_trace(rank, rendezvous):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        cache, pipeline, req = _staged_fixture()
        held = pipeline.staged_prefetches.pop(req.cache_request_handle)
        cache.cache_controller.prefetch_tokens_occupied = 0
        waiting = [SimpleNamespace(rid="head", storage_prefetch_retry_attempts=0), req]
        published = False
        issued = []
        for step in range(7):
            # Native completion arrives one pass earlier on rank 0. Only the
            # agreed completion may publish staging to the admission path.
            ready = torch.tensor([int(step >= rank + 1)])
            torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
            if ready.item() and not published:
                pipeline.staged_prefetches[req.cache_request_handle] = held
                cache.cache_controller.prefetch_tokens_occupied = 6
                published = True
            for retry_req, hit_end in cache.storage_prefetch_retries.pop_ready(
                waiting, 2, 8
            ):
                issued.append((step, retry_req.rid, hit_end))
            if pipeline.has_staged(req.cache_request_handle):
                full_match = 2 if step < 3 else 0
                cache.tree_core.match_full_device_prefix.return_value = (
                    full_match,
                    1,
                    full_match,
                )
                req.prefix_indices = torch.arange(full_match)
                if cache.buffer_pipeline.prepare_staged_prefetch(req):
                    assert (
                        cache.init_load_back(
                            InitLoadBackParams(None, req.host_hit_length, req=req)
                        )
                        is None
                    )
            snapshot = (
                list(issued),
                pipeline.has_staged(req.cache_request_handle),
                cache.cache_controller.prefetch_tokens_occupied,
            )
            snapshots = [None, None]
            torch.distributed.all_gather_object(snapshots, snapshot)
            assert snapshots[0] == snapshots[1], (step, snapshots)
        assert issued == [(4, "r", 8)], issued
        assert cache.cache_controller.load.call_count == 1
        assert cache.cache_controller.prefetch_tokens_occupied == 0
    finally:
        torch.distributed.destroy_process_group()


class TestStagedPrefetchLifecycle(unittest.TestCase):
    def test_trim_and_stage_preserve_raw_token_boundaries(self):
        for bigram in (False, True):
            for trims in ((2,), (2, 2), (8,), (2, 6)):
                with self.subTest(bigram=bigram, trims=trims):
                    cache, pipeline, req = _staged_fixture()
                    pipeline.staged_prefetches.clear()
                    cache.tree_core.is_eagle = bigram
                    tokens = array("q", range(10 + int(bigram)))
                    cache.prefetch_from_storage(
                        req.cache_request_handle,
                        0,
                        tokens[2:],
                        matched_prefix_tokens=tokens[:2],
                        storage_hit_end=10,
                    )
                    info = cache.ongoing_prefetch[req.cache_request_handle]
                    operation = info.operation
                    self.assertEqual(len(info.prefetch_key), 8)
                    self.assertTrue(operation.assume_stored)
                    operation.hash_value = ["h0", "h1", "h2", "h3"]
                    operation.storage_hit_count = 8
                    matched_len, hit_tokens = 2, 8
                    for trim in trims:
                        matched_len += trim
                        info, hit_tokens, aux_tokens = (
                            cache._trim_buffer_prefetch_full_head(
                                req.cache_request_handle,
                                info,
                                operation,
                                matched_len,
                                hit_tokens,
                            )
                        )
                        self.assertEqual(aux_tokens, 8)
                        self.assertEqual(
                            pipeline._prefetch_prefix_ctx[req.cache_request_handle][0],
                            list(tokens[:matched_len]),
                        )
                        self.assertEqual(
                            list(info.prefetch_key.raw_token_ids()),
                            list(tokens[matched_len:]),
                        )
                    # A capacity retry must retain the endpoint even when
                    # there is no FULL suffix and only SWA remains to load.
                    cache.tree_core.match_full_device_prefix.return_value = (
                        matched_len,
                        1,
                        matched_len,
                    )
                    pipeline.anchor_lock_cap_tokens = 100
                    pipeline.try_lock_anchor(req.cache_request_handle, hit_tokens)
                    anchor_key = (
                        cache.tree_core.match_full_device_prefix.call_args.args[0]
                    )
                    self.assertEqual(anchor_key.raw_token_ids(), tokens)
                    pipeline.release_anchor_lock(req.cache_request_handle)
                    cache.tree_core.match_full_device_prefix.reset_mock()
                    swa = PoolTransfer(name=PoolName.SWA, host_indices=torch.arange(4))
                    operation.pool_transfers = [swa]
                    operation.host_indices = torch.arange(hit_tokens)
                    cache.ongoing_prefetch[req.cache_request_handle] = info._replace(
                        host_indices=operation.host_indices, comp_xfers={"swa": [swa]}
                    )
                    cache.cache_controller.prefetch_tokens_occupied = hit_tokens
                    cache.storage_existence_cache = Mock()
                    pipeline.stage_completed_prefetch(
                        req.cache_request_handle, hit_tokens, operation.hash_value
                    )
                    held = pipeline.staged_prefetches[req.cache_request_handle]
                    self.assertEqual(held.key_tokens, tokens)
                    self.assertEqual(held.matched_len, matched_len)
                    self.assertEqual(held.num_tokens, hit_tokens)
                    self.assertEqual(
                        len(RadixKey(held.key_tokens, is_bigram=bigram)), 10
                    )
                    # A joint match counts bigrams, not their extra raw boundary token.
                    req.prefix_indices = torch.arange(10)
                    req.kv.cache_protected_len = 10
                    self.assertTrue(pipeline.prepare_staged_prefetch(req))
                    self.assertFalse(pipeline.has_staged(req.cache_request_handle))
                    cache.tree_core.match_full_device_prefix.assert_not_called()

    def test_two_rank_completion_capacity_and_anchor_loss(self):
        with tempfile.TemporaryDirectory(prefix="prefetch-rank-test-") as directory:
            torch.multiprocessing.spawn(
                _two_rank_retry_trace, args=(f"{directory}/store",), nprocs=2, join=True
            )

    def test_next_pass_uses_fresh_joint_match(self):
        cache, pipeline, req = _staged_fixture(full_match=8)
        self.assertTrue(cache.buffer_pipeline.prepare_staged_prefetch(req))
        self.assertEqual((req.host_hit_length, req.swa_host_hit_length), (0, 4))
        self.assertEqual(req.kv.cache_protected_len, 8)
        plan = req.staged_prefetch_plan
        self.assertIs(
            plan.key.token_ids,
            pipeline.staged_prefetches[req.cache_request_handle].key_tokens,
        )
        cache.tree_core.match_full_device_prefix.assert_called_once()
        # A twin finished first: the next pass's joint match runs past the
        # staged span and is kept as is (a shrink would strand its recompute).
        req.prefix_indices = torch.arange(12)
        req.last_node = 9
        req.kv.cache_protected_len = 12
        self.assertTrue(cache.buffer_pipeline.prepare_staged_prefetch(req))
        self.assertIsNone(req.staged_prefetch_plan)
        self.assertEqual(len(req.prefix_indices), 12)
        self.assertEqual((req.last_node, req.kv.cache_protected_len), (9, 12))
        self.assertEqual((req.host_hit_length, req.swa_host_hit_length), (0, 0))
        self.assertFalse(pipeline.has_staged(req.cache_request_handle))
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)
        cache.cache_controller.mem_pool_host.free.assert_called_once()
        cache.tree_core.match_full_device_prefix.assert_called_once()

    def test_capacity_retry_keeps_buffers_without_a_storage_retry(self):
        for available in (0, 100):
            with self.subTest(full_available=available):
                cache, pipeline, req = _staged_fixture()
                cache.cache_controller.load.return_value = None
                cache.token_to_kv_pool_allocator.full_available_size.return_value = (
                    available
                )
                held = pipeline.staged_prefetches[req.cache_request_handle]
                pipeline.anchor_lock_cap_tokens = 8
                pipeline._prefetch_prefix_ctx[req.cache_request_handle] = (
                    [0, 1],
                    None,
                    None,
                    (),
                )
                self.assertEqual(
                    pipeline.try_lock_anchor(req.cache_request_handle, 0), ("locked", 2)
                )
                anchor = pipeline.anchor_locks[req.cache_request_handle]
                cache.tree_core.match_full_device_prefix.reset_mock()
                for attempt in range(1, 4):
                    req.prefix_indices = torch.arange(2)
                    self.assertTrue(cache.buffer_pipeline.prepare_staged_prefetch(req))
                    self.assertIsNone(
                        cache.init_load_back(
                            InitLoadBackParams(
                                best_match_node=None,
                                host_hit_length=req.host_hit_length,
                                req=req,
                            )
                        )
                    )
                    self.assertIs(
                        pipeline.staged_prefetches[req.cache_request_handle], held
                    )
                    self.assertIs(
                        pipeline.anchor_locks[req.cache_request_handle], anchor
                    )
                    self.assertEqual(pipeline.anchor_locked_tokens_, 2)
                    self.assertEqual(
                        cache.storage_prefetch_retries.pop_ready([req], 0, 8), []
                    )
                    self.assertEqual(
                        cache.tree_core.match_full_device_prefix.call_count, attempt
                    )
                    self.assertEqual(
                        cache.tree_core.collect_full_device_indices.call_count, attempt
                    )
                cache.cache_controller.mem_pool_host.free.assert_not_called()
                self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 6)
                cache.tree_core.dec_full_pin.assert_not_called()
                pipeline.release_staged_hold(req.cache_request_handle)
                cache.tree_core.dec_full_pin.assert_called_once_with(anchor.node_id)
                self.assertEqual(pipeline.anchor_locked_tokens_, 0)

    def test_next_pass_replans_growth_and_refetches_anchor_loss_once(self):
        cache, pipeline, req = _staged_fixture()
        self.assertTrue(cache.buffer_pipeline.prepare_staged_prefetch(req))
        self.assertEqual((req.host_hit_length, req.swa_host_hit_length), (6, 4))

        # Tree changes occur while queued, before the next preparation pass.
        cache.tree_core.match_full_device_prefix.return_value = (6, 1, 6)
        req.prefix_indices = torch.arange(2)
        self.assertTrue(cache.buffer_pipeline.prepare_staged_prefetch(req))
        self.assertEqual((req.host_hit_length, req.swa_host_hit_length), (2, 4))
        self.assertTrue(pipeline.has_staged(req.cache_request_handle))
        self.assertEqual(cache.storage_prefetch_retries.pop_ready([req], 0, 8), [])

        cache.tree_core.match_full_device_prefix.return_value = (0, 0, 0)
        req.prefix_indices = torch.arange(0)
        self.assertFalse(cache.buffer_pipeline.prepare_staged_prefetch(req))
        self.assertFalse(pipeline.has_staged(req.cache_request_handle))
        self.assertEqual(
            cache.storage_prefetch_retries.pop_ready([req], 0, 8), [(req, 8)]
        )
        self.assertEqual(cache.storage_prefetch_retries.pop_ready([req], 0, 8), [])
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)
        cache.cache_controller.load.assert_not_called()

    def test_retry_budget_bounds_reissues_and_paces_capacity_misses(self):
        """Past --hicache-storage-prefetch-retry-max-attempts a request stops
        re-issuing; a rate-limited cache-mode query is paced, not re-issued."""
        retries = StoragePrefetchRetries()
        head = SimpleNamespace(rid="head", storage_prefetch_retry_attempts=0)
        req = SimpleNamespace(rid="r", storage_prefetch_retry_attempts=8)
        retries.refetch(req.rid, 8)
        self.assertEqual(retries.pop_ready([head, req], 0, 8), [])
        req.storage_prefetch_retry_attempts = 7
        retries.refetch(req.rid, 8)
        self.assertEqual(retries.pop_ready([head, req], 0, 8), [(req, 8)])
        # A paced retry yields to the queue head; an immediate one is re-issued.
        retries.poll_miss(head.rid)
        retries.refetch(req.rid, 8)
        self.assertEqual(retries.pop_ready([head, req], 2, 8), [(req, 8)])

        cache, _, req = _staged_fixture()
        cache.host_memory_mode = "cache"
        cache.buffer_pipeline = None
        cache.cache_controller.prefetch_rate_limited = lambda: True
        tokens = array("q", range(10))
        cache.prefetch_from_storage(
            req.cache_request_handle,
            0,
            tokens[2:],
            matched_prefix_tokens=tokens[:2],
            storage_hit_end=10,
        )
        self.assertEqual(cache.ongoing_prefetch, {})
        retries = cache.storage_prefetch_retries
        self.assertEqual(retries.pop_ready([head, req], 2, 8), [])
        self.assertEqual(retries.pop_ready([head, req], 2, 8), [])
        self.assertEqual(retries.pop_ready([head, req], 2, 8), [(req, 10)])

    def test_staged_hold_drops_after_bounded_admission_deferrals(self):
        """A hold that cannot be materialized after max_staged_admission_defers
        passes is released, and the request re-plans without a new L3 query."""
        cache, pipeline, req = _staged_fixture()
        cache.cache_controller.load.return_value = None
        pipeline.max_staged_admission_defers = 3
        params = lambda: InitLoadBackParams(None, req.host_hit_length, req=req)
        for attempt in range(1, 4):
            req.prefix_indices = torch.arange(2)
            self.assertTrue(pipeline.prepare_staged_prefetch(req))
            self.assertIsNone(cache.init_load_back(params()))
            self.assertEqual(pipeline.has_staged(req.cache_request_handle), attempt < 3)
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)
        self.assertEqual(cache.storage_prefetch_retries.pop_ready([req], 0, 8), [])
        self.assertTrue(pipeline.prepare_staged_prefetch(req))
        self.assertEqual((req.staged_prefetch_plan, req.storage_hit_length), (None, 0))

    def test_controller_terminated_query_counts_as_an_l3_miss(self):
        """A query the controller terminated (store miss or short hit) must feed
        the L3-miss counters, or a store that lost pages reads as zero misses."""
        cache = _hit_drain_fixture()
        _terminated_query(cache, "miss", hit_tokens=0)
        _terminated_query(cache, "short", hit_tokens=2)
        cache._drain_storage_control_queues_impl(
            n_storage_hit=2,
            n_ack_prefetch=0,
            n_backup=0,
            n_release=0,
            extra_release_counts={},
            log_metrics=False,
        )
        stats = cache._prefetch_outcome_stats
        self.assertEqual(
            (
                stats["revoked_full_miss"],
                stats["revoked_insufficient"],
                stats["l3_miss_tokens"],
            ),
            (1, 1, 14),
        )
        self.assertEqual(cache.revoke_pending_prefetch.call_count, 2)
        head = SimpleNamespace(rid="head", storage_prefetch_retry_attempts=0)
        req = SimpleNamespace(rid="miss", storage_prefetch_retry_attempts=0)
        retries = cache.storage_prefetch_retries
        self.assertEqual(retries.pop_ready([head, req], 1, 8), [])
        self.assertEqual(retries.pop_ready([head, req], 1, 8), [(req, None)])


class TestMultimodalStorageIdentity(CustomTestCase):
    _A = "sha256:" + "a" * 64
    _B = "sha256:" + "b" * 64
    _C = "sha256:" + "c" * 64

    def test_decode_prefetch_retains_confirmed_bigram_endpoint(self):
        """A confirmed logical hit includes its final raw media token in bigram mode."""
        tokens = array("q", range(11))
        spans = (
            MultimodalKeySpan(1, 3, self._A),
            MultimodalKeySpan(6, 7, self._B),
        )
        for cache_type in (HiRadixCache, UnifiedRadixCache):
            for bigram in (False, True):
                with self.subTest(cache=cache_type.__name__, bigram=bigram):
                    key = RadixKey(tokens, is_bigram=bigram, mm_spans=spans)
                    last_hash = get_storage_hash_str(key[:2])
                    if cache_type is UnifiedRadixCache:
                        cache, pipeline, _ = _staged_fixture()
                        pipeline.staged_prefetches.clear()
                        cache.tree_core.is_eagle = bigram
                        host = 0
                    else:
                        cache = HiRadixCache.__new__(HiRadixCache)
                        cache.page_size = 2
                        cache.is_eagle = bigram
                        cache.enable_storage = True
                        cache.prefetch_threshold = 2
                        cache.ongoing_prefetch = {}
                        cache._get_extra_pools = lambda: {}
                        controller = HiCacheController.__new__(HiCacheController)
                        controller.prefetch_rate_limited = lambda: False
                        controller.prefetch_queue = Queue()
                        controller.prefetch_tokens_occupied = 0
                        cache.cache_controller = controller
                        host = SimpleNamespace(protect_host=lambda: None)
                    cache.get_last_hash_value = lambda _: last_hash
                    cache.hicache_storage_pass_prefix_keys = False
                    req = SimpleNamespace(
                        rid="r",
                        cache_request_handle=_REQ,
                        origin_input_ids=tokens,
                        extra_key=None,
                        cache_salt=None,
                        mm_cache_spans=spans,
                    )
                    match = DecodePrefixMatch(
                        prefix_indices=torch.arange(2),
                        l2_host_hit_length=0,
                        l3_storage_hit_length=4,
                        last_device_node=host,
                        last_host_node=host,
                    )
                    DecodeHiCachePreallocMixin._start_hicache_prefetch(
                        SimpleNamespace(tree_cache=cache), req, match
                    )
                    self.assertTrue(match.prefetch_registered)
                    operation = cache.cache_controller.prefetch_queue.get_nowait()
                    fetched = operation.token_ids
                    expected = key[2:6]
                    self.assertEqual(len(fetched), 4)
                    self.assertEqual(fetched.raw_token_ids(), expected.raw_token_ids())
                    self.assertEqual(fetched.mm_spans, expected.mm_spans)
                    self.assertEqual(
                        get_storage_hash_str(fetched, last_hash, page_size=2),
                        get_storage_hash_str(expected, last_hash, page_size=2),
                    )

    def test_storage_probe_keeps_text_and_earlier_media_hits(self):
        """A later one-token media change must not alias the stored suffix or salt its prefix."""
        tokens = array("q", [1, 2, 99, 3, 4, 5, 99, 6, 7, 8])
        stored_spans = (
            MultimodalKeySpan(2, 3, self._A),
            MultimodalKeySpan(6, 7, self._A),
        )
        stored = RadixKey(tokens, mm_spans=stored_spans)
        hashes = get_storage_hash_str(stored, page_size=2)

        def exists(keys, _extra_info):
            for index, key in enumerate(keys):
                if key not in hashes:
                    return index
            return len(keys)

        for cache_type in (HiRadixCache, UnifiedRadixCache):
            with self.subTest(cache=cache_type.__name__):
                cache = cache_type.__new__(cache_type)
                cache.tree_core = SimpleNamespace(
                    enable_storage=True, page_size=2, is_eagle=False
                )
                cache.enable_storage = True
                cache.page_size = 2
                cache.is_eagle = False
                cache.prefetch_threshold = 2
                cache._all_reduce_attn_groups = lambda *_: None
                controller = HiCacheController.__new__(HiCacheController)
                controller.page_size = 2
                controller.prefetch_rate_limited = lambda: False
                controller.storage_backend = SimpleNamespace(batch_exists=exists)
                cache.cache_controller = controller
                suffix = tokens[2:]
                for second_identity, expected_hit in ((self._A, 8), (self._B, 4)):
                    spans = (
                        stored_spans[0],
                        MultimodalKeySpan(6, 7, second_identity),
                    )
                    hit = cache.query_storage_hit_length(
                        0,
                        suffix,
                        last_hash=hashes[0],
                        mm_spans=slice_mm_spans(spans, 2, len(tokens)),
                    )
                    self.assertEqual(hit, expected_hit)
                changed_first = (
                    MultimodalKeySpan(2, 3, self._B),
                    stored_spans[1],
                )
                self.assertEqual(
                    cache.query_storage_hit_length(0, tokens, mm_spans=changed_first),
                    2,
                )
                host = SimpleNamespace(protect_host=lambda: None)
                cache.is_backuped = lambda _: True
                cache.get_last_hash_value = lambda _: hashes[0]
                cache.hicache_storage_pass_prefix_keys = False
                req = SimpleNamespace(
                    rid="r",
                    cache_request_handle=_REQ,
                    origin_input_ids=tokens,
                    extra_key=None,
                    cache_salt=None,
                    mm_cache_spans=spans,
                )
                harness = SimpleNamespace(
                    scheduler=SimpleNamespace(enable_decode_hicache=True),
                    tree_cache=cache,
                )
                match = DecodeHiCachePreallocMixin._build_decode_prefix_match(
                    harness,
                    req,
                    SimpleNamespace(
                        device_indices=torch.arange(2),
                        host_hit_length=0,
                        last_device_node=host,
                        last_host_node=host,
                    ),
                )
                self.assertEqual(match.l3_storage_hit_length, 4)
                if cache_type is HiRadixCache:
                    cache.ongoing_prefetch = {}
                    cache._get_extra_pools = lambda: {}
                    controller.prefetch_queue = Queue()
                    controller.prefetch_tokens_occupied = 0
                    DecodeHiCachePreallocMixin._start_hicache_prefetch(
                        harness, req, match
                    )
                    self.assertTrue(match.prefetch_registered)
                    operation = controller.prefetch_queue.get_nowait()
                    self.assertEqual(len(operation.token_ids), 4)
                    self.assertEqual(controller._storage_hit_query(operation)[1], 4)

    def test_trim_stage_and_admission_preserve_media_offsets(self):
        """Trimmed fetches and retries retain media at shared bigram boundaries."""
        for bigram in (False, True):
            for trims in ((2, 2), (8,)):
                with self.subTest(bigram=bigram, trims=trims):
                    cache, pipeline, req = _staged_fixture()
                    pipeline.staged_prefetches.clear()
                    cache.tree_core.is_eagle = bigram
                    tokens = array("q", range(10 + int(bigram)))
                    spans = (
                        MultimodalKeySpan(1, 3, self._A),
                        MultimodalKeySpan(4, 5, self._B),
                        MultimodalKeySpan(6, 9, self._A),
                    )
                    if bigram:
                        spans += (MultimodalKeySpan(10, 11, self._B),)
                    expected = RadixKey(tokens, is_bigram=bigram, mm_spans=spans)
                    cache.prefetch_from_storage(
                        req.cache_request_handle,
                        0,
                        tokens[2:],
                        matched_prefix_tokens=tokens[:2],
                        mm_spans=slice_mm_spans(spans, 2, len(tokens)),
                        matched_prefix_mm_spans=slice_mm_spans(spans, 0, 2),
                        storage_hit_end=10,
                    )
                    info = cache.ongoing_prefetch[req.cache_request_handle]
                    operation = info.operation
                    operation.hash_value = get_storage_hash_str(
                        info.prefetch_key, page_size=2
                    )
                    operation.storage_hit_count = 8
                    matched, hit = 2, 8
                    for trim in trims:
                        matched += trim
                        info, hit, _ = cache._trim_buffer_prefetch_full_head(
                            req.cache_request_handle, info, operation, matched, hit
                        )
                    cache.tree_core.match_full_device_prefix.side_effect = lambda key: (
                        min(matched, key.match(expected)),
                        1,
                        min(matched, key.match(expected)),
                    )
                    cache.tree_core.collect_full_device_indices.return_value = (
                        torch.arange(10)
                    )
                    pipeline.anchor_lock_cap_tokens = 100
                    self.assertEqual(
                        pipeline.try_lock_anchor(req.cache_request_handle, hit),
                        ("locked", matched),
                    )
                    anchor_key = (
                        cache.tree_core.match_full_device_prefix.call_args.args[0]
                    )
                    self.assertEqual(anchor_key.match(expected), 10)
                    pipeline.release_anchor_lock(req.cache_request_handle)
                    swa = PoolTransfer(name=PoolName.SWA, host_indices=torch.arange(4))
                    operation.pool_transfers = [swa]
                    operation.host_indices = torch.arange(hit)
                    cache.ongoing_prefetch[req.cache_request_handle] = info._replace(
                        host_indices=operation.host_indices, comp_xfers={"swa": [swa]}
                    )
                    cache.cache_controller.prefetch_tokens_occupied = hit
                    cache.storage_existence_cache = Mock()
                    pipeline.stage_completed_prefetch(
                        req.cache_request_handle, hit, operation.hash_value
                    )
                    req.prefix_indices = torch.arange(0)
                    self.assertTrue(pipeline.prepare_staged_prefetch(req))
                    restored_key = req.staged_prefetch_plan.key
                    self.assertEqual(restored_key.raw_token_ids(), tokens)
                    self.assertEqual(restored_key.match(expected), 10)
                    self.assertEqual(
                        get_storage_hash_str(restored_key, page_size=2),
                        get_storage_hash_str(expected, page_size=2),
                    )

    def test_split_backup_and_snapshot_keep_short_and_repeated_media(self):
        """Backup snapshots must preserve identities after live node splits."""
        for bigram in (False, True):
            with self.subTest(bigram=bigram):
                tokens = array("q", range(12 + int(bigram)))
                spans = (
                    MultimodalKeySpan(2, 3, self._A),
                    MultimodalKeySpan(4, 9, self._B),
                    MultimodalKeySpan(10, 11, self._A),
                )
                key = RadixKey(tokens, is_bigram=bigram, mm_spans=spans)
                hashes = get_storage_hash_str(key, page_size=2)
                root = SimpleNamespace(id=0, get_last_hash_value=lambda: None)
                top = SimpleNamespace(
                    id=1,
                    parent=root,
                    key=key[:6],
                    hash_value=hashes[:3],
                    host_value=torch.arange(6),
                )
                leaf = SimpleNamespace(
                    id=2,
                    parent=top,
                    key=key[6:],
                    hash_value=hashes[3:],
                    host_value=torch.arange(6, 12),
                )
                cache = HiRadixCache.__new__(HiRadixCache)
                cache.root_node = root
                _, joined, joined_hashes, host = cache._concat_split_chain(leaf, 12)
                self.assertEqual(list(joined.token_ids), list(tokens))
                self.assertEqual(joined_hashes, hashes)
                self.assertEqual(get_storage_hash_str(joined, page_size=2), hashes)
                torch.testing.assert_close(host, torch.arange(12))

                node = SimpleNamespace(
                    id=3,
                    parent=root,
                    key=key,
                    hash_value=hashes,
                    component_data={BASE_COMPONENT_TYPE: SimpleNamespace(value=host)},
                )
                core = UnifiedTreeCore.__new__(UnifiedTreeCore)
                core.root_node = root
                core._node_arena = {node.id: node}
                snapshot = core.snapshot_buffer_backup(node.id, False)
                node.key = key[6:]
                self.assertEqual(snapshot.key.match(key), 12)
                self.assertEqual(
                    get_storage_hash_str(snapshot.key, page_size=2), hashes
                )
                changed = RadixKey(
                    tokens,
                    is_bigram=bigram,
                    mm_spans=spans[:2] + (MultimodalKeySpan(10, 11, self._C),),
                )
                self.assertEqual(snapshot.key.match(changed), 10 - int(bigram))


if __name__ == "__main__":
    unittest.main()
