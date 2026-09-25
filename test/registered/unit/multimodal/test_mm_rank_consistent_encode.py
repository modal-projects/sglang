"""Cache asymmetry must not change encoder collectives or release live features."""

import tempfile
import unittest
from contextlib import nullcontext
from datetime import timedelta
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.managers import mm_schedule, schedule_batch
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.transport import cuda_ipc, memory_pool
from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class _PeerGroup:
    def __init__(self, peer_flags):
        self.peer_flags = peer_flags
        self.flags = []

    def all_reduce(self, flags):
        self.flags.append(flags.tolist())
        peer = torch.tensor(self.peer_flags, dtype=flags.dtype)
        assert flags.shape == peer.shape
        return flags + peer


class _GlooGroup:
    def all_reduce(self, flags):
        dist.all_reduce(flags)
        return flags


def _item(identity, rows):
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=identity,
        offsets=[(0, rows - 1)],
        feature=torch.full((rows, 2), float(identity)),
    )


def _content_item(value, rows):
    item = _item(value, rows)
    item.set_pad_value()
    item.set_hash(11)
    return item


def _offsets(items):
    offsets = []
    cursor = 0
    for item in items:
        rows = sum(end - start + 1 for start, end in item.offsets)
        offsets.append((cursor, cursor + rows - 1))
        cursor += rows
    return offsets, cursor


def _run_route(route, items, encode, *, prefix=0, extend=None):
    offsets, rows = _offsets(items)
    if extend is None:
        extend = rows - prefix
    if route == "full":
        return mm_schedule._get_chunked_embedding_full(
            encode,
            items,
            offsets,
            prefix,
            extend,
            torch.zeros(extend, dtype=torch.long),
            torch.device("cpu"),
        )[0]
    if route == "by_item":
        return mm_schedule._get_chunked_embedding_by_item(
            encode, items, offsets, prefix, extend, torch.device("cpu")
        )
    request = mm_schedule.PerImageRequestInfo(0, items, offsets, prefix, extend)
    embeddings = mm_schedule._batch_encode_per_image_misses(
        encode, [request], torch.device("cpu")
    )
    return mm_schedule._assemble_per_image_chunk(
        request.overlapping, embeddings, prefix, extend
    )


def _cache(route, items, hits, *, stale=False):
    mm_schedule.init_mm_embedding_cache(1 << 20)
    if route == "full":
        if hits:
            key = mm_schedule.MultiModalStaticCache.combine_hashes(
                [item.cache_key for item in items]
            )
            value = torch.cat([item.feature for item in items])
            if stale:
                value = value[:-1]
            mm_schedule.embedding_cache.set(
                key, mm_schedule.EmbeddingResult(embedding=value)
            )
    else:
        for i in hits:
            value = items[i].feature
            if stale:
                value = value[:-1]
            mm_schedule.embedding_cache.set(
                items[i].cache_key, mm_schedule.EmbeddingResult(embedding=value)
            )


def _parallel(group, rank=0):
    return SimpleNamespace(
        attn_tp_size=2, attn_tp_rank=rank, attn_tp_group=group, tp_size=4
    )


def _collective_worker(rank, rendezvous):
    # An order mismatch is reported at the encoder boundary; missing participation
    # is bounded by the process-group timeout instead of hanging the CPU suite.
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:
        with patch.object(
            mm_schedule, "get_parallel", return_value=_parallel(_GlooGroup(), rank)
        ):
            for route, content_keys in product(
                ("batch", "by_item", "full"), (False, True)
            ):
                for mode in ("asymmetric", "all_hit", "all_miss", "stale"):
                    if content_keys:
                        items = [
                            _content_item(11, 2),
                            _content_item(22, 2),
                            _content_item(11, 2),
                        ]
                        assert len({item.hash for item in items}) == 1
                        assert len({item.cache_key for item in items}) == 2
                    else:
                        items = [_item(11, 2), _item(22, 3)]
                    if mode == "asymmetric":
                        hits = [rank] if route != "full" else ([0] if rank == 0 else [])
                    elif mode == "all_hit":
                        hits = [0, 1]
                    elif mode == "stale":
                        hits = [0, 1]
                    else:
                        hits = []
                    _cache(route, items, hits, stale=mode == "stale" and rank == 1)
                    calls = []

                    def encode(batch):
                        identities = [
                            (item.cache_key, len(item.feature)) for item in batch
                        ]
                        peers = [None, None]
                        dist.all_gather_object(peers, identities)
                        assert peers[0] == peers[1], peers
                        calls.append(identities)
                        value = torch.cat([item.feature for item in batch])
                        dist.all_reduce(value)
                        return value / 2

                    result = _run_route(route, items, encode)
                    assert torch.equal(
                        result, torch.cat([item.feature for item in items])
                    )
                    assert len(calls) == (0 if mode == "all_hit" else 1)
    finally:
        dist.destroy_process_group()


class TestRankConsistentEncode(CustomTestCase):
    def setUp(self):
        self.old_cache = mm_schedule.embedding_cache

    def tearDown(self):
        mm_schedule.embedding_cache = self.old_cache

    def test_content_keys_with_shared_routing_hints_and_repeated_spans(self):
        """Local hits and TP miss unions retain each distinct content identity."""
        for route, mode in product(
            ("batch", "by_item", "full"),
            ("all_hit", "all_miss", "mixed", "peer_miss"),
        ):
            with self.subTest(route=route, mode=mode):
                items = [
                    _content_item(11, 2),
                    _content_item(22, 2),
                    _content_item(11, 2),
                ]
                a, b, repeated_a = items
                self.assertEqual(a.hash, b.hash)
                self.assertEqual(a.pad_value, b.pad_value)
                self.assertNotEqual(a.cache_key, b.cache_key)
                self.assertEqual(a.cache_key, repeated_a.cache_key)
                hits = [] if mode == "all_miss" else [0]
                if mode == "all_hit":
                    hits = [0, 1]
                cached_items = (
                    [a, repeated_a, a] if route == "full" and mode == "mixed" else items
                )
                _cache(route, cached_items, hits)
                peer_flags = [int(mode == "peer_miss")]
                if route != "full":
                    peer_flags.append(0)
                group = _PeerGroup(peer_flags)
                encoded = []

                def encode(batch):
                    encoded.extend(item.cache_key for item in batch)
                    return torch.cat([item.feature + 100 for item in batch])

                with (
                    patch.object(dist, "is_initialized", return_value=True),
                    patch.object(
                        mm_schedule, "get_parallel", return_value=_parallel(group)
                    ),
                ):
                    result = _run_route(route, items, encode, prefix=1, extend=4)

                if mode == "all_hit":
                    expected_encoded = []
                    values = [item.feature for item in items]
                elif mode == "mixed" and route != "full":
                    expected_encoded = [b.cache_key]
                    values = [a.feature, b.feature + 100, repeated_a.feature]
                else:
                    expected_encoded = [a.cache_key, b.cache_key]
                    if route != "batch":
                        expected_encoded.append(a.cache_key)
                    values = [item.feature + 100 for item in items]
                self.assertEqual(encoded, expected_encoded)
                torch.testing.assert_close(result, torch.cat(values)[1:5])

    def test_asymmetric_hits_keep_original_encode_order(self):
        """A local hit before a local miss must retain its batch position."""
        for route in ("batch", "by_item", "full"):
            for per_item_result in (False, True):
                with self.subTest(route=route, per_item_result=per_item_result):
                    items = [_item(11, 2), _item(22, 3)]
                    _cache(route, items, [0])
                    group = _PeerGroup([1] if route == "full" else [1, 0])
                    encoded = []

                    def encode(batch):
                        encoded.extend(item.hash for item in batch)
                        tensors = [item.feature + 100 for item in batch]
                        return tensors if per_item_result else torch.cat(tensors)

                    with (
                        patch.object(dist, "is_initialized", return_value=True),
                        patch.object(
                            mm_schedule, "get_parallel", return_value=_parallel(group)
                        ),
                    ):
                        result = _run_route(route, items, encode, prefix=1, extend=3)
                    self.assertEqual(encoded, [11, 22])
                    torch.testing.assert_close(
                        result, torch.cat([item.feature + 100 for item in items])[1:4]
                    )

    def test_row_count_is_part_of_collective_identity(self):
        """A stale row count is a miss; distinct spans must not be deduplicated."""
        for route in ("batch", "by_item"):
            with self.subTest(route=route):
                items = [_item(11, 2), _item(11, 3), _item(11, 2)]
                _cache(route, items, [0])
                group = _PeerGroup([1, 0])
                encoded = []

                def encode(batch):
                    encoded.extend(len(item.feature) for item in batch)
                    return [item.feature + 100 for item in batch]

                with (
                    patch.object(dist, "is_initialized", return_value=True),
                    patch.object(
                        mm_schedule, "get_parallel", return_value=_parallel(group)
                    ),
                ):
                    result = _run_route(route, items, encode)
                self.assertEqual(encoded, [2, 3] if route == "batch" else [2, 3, 2])
                torch.testing.assert_close(
                    result, torch.cat([item.feature + 100 for item in items])
                )

    def test_cross_request_batch_deduplicates_in_original_order(self):
        """A forced miss shared by requests is encoded once and sliced per request."""
        a, b, repeated_a, c = _item(11, 2), _item(22, 3), _item(11, 2), _item(33, 2)
        _cache("batch", [a], [0])
        encoded = []

        def encode(batch):
            encoded.extend(item.hash for item in batch)
            return torch.cat([item.feature + 100 for item in batch])

        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(
                mm_schedule,
                "get_parallel",
                return_value=_parallel(_PeerGroup([1, 0, 0])),
            ),
            patch.object(mm_schedule, "_is_hip", False),
            patch.object(mm_schedule, "_is_npu", False),
            patch.object(mm_schedule, "_is_xpu", False),
        ):
            result, _ = mm_schedule._get_chunked_prefill_embedding(
                encode,
                [a, b, repeated_a, c],
                [0, 2, 4],
                [1, 0],
                [6, 4],
                [[(0, 1), (4, 6)], [(0, 1), (2, 3)]],
                torch.zeros(10, dtype=torch.long),
            )
        self.assertEqual(encoded, [11, 22, 33])
        torch.testing.assert_close(
            result,
            torch.cat([a.feature[1:], b.feature, repeated_a.feature, c.feature]) + 100,
        )

    def test_all_hit_and_single_rank_preserve_cached_outputs(self):
        """Agreement cannot convert unanimous hits into extra encoder work."""
        for route in ("batch", "by_item", "full"):
            for distributed, size in ((True, 2), (True, 1), (False, 2)):
                with self.subTest(route=route, distributed=distributed, size=size):
                    items = [_item(11, 2)]
                    _cache(route, items, [0])
                    group = _PeerGroup([0])
                    parallel = _parallel(group)
                    parallel.attn_tp_size = size

                    def encode(batch):
                        self.fail("Unanimous cache hits entered the encoder")

                    with (
                        patch.object(dist, "is_initialized", return_value=distributed),
                        patch.object(
                            mm_schedule, "get_parallel", return_value=parallel
                        ),
                    ):
                        result = _run_route(route, items, encode)
                    torch.testing.assert_close(result, items[0].feature)
                    self.assertEqual(
                        group.flags, [[0]] if distributed and size == 2 else []
                    )

    def test_forced_reencode_is_reused_by_subsequent_hits(self):
        """Batch-dependent rounding must not leave different encodings cached."""
        for route, content_keys in product(("batch", "by_item", "full"), (False, True)):
            with self.subTest(route=route, content_keys=content_keys):
                if content_keys:
                    items = [
                        _content_item(11, 2),
                        _content_item(22, 2),
                        _content_item(11, 2),
                    ]
                    self.assertEqual(len({item.hash for item in items}), 1)
                    self.assertEqual(len({item.pad_value for item in items}), 1)
                    self.assertNotEqual(items[0].cache_key, items[1].cache_key)
                    self.assertEqual(items[0].cache_key, items[2].cache_key)
                    for item in items:
                        self.assertRegex(item.cache_key, r"^sha256:[0-9a-f]{64}$")
                else:
                    items = [_item(11, 2)]
                unique_items = {item.cache_key: item for item in items}
                stored_items = items if route == "full" else unique_items.values()
                fresh = torch.cat([item.feature for item in stored_items])
                expected = torch.cat([item.feature for item in items])
                flag_count = 1 if route == "full" else len(unique_items)

                def encode(batch):
                    return torch.cat([item.feature for item in batch])

                def encode_prior(batch):
                    value = encode(batch)
                    return torch.nextafter(value, torch.full_like(value, float("inf")))

                caches = []
                for rank in (0, 1):
                    mm_schedule.init_mm_embedding_cache(1 << 20)
                    cache = mm_schedule.embedding_cache
                    caches.append(cache)
                    if rank == 0:
                        with patch.object(dist, "is_initialized", return_value=False):
                            _run_route(route, items, encode_prior)
                    cache.set(
                        99, mm_schedule.EmbeddingResult(embedding=torch.ones(1, 2))
                    )

                first_results = []
                next_results = []
                for rank, cache in enumerate(caches):
                    mm_schedule.embedding_cache = cache
                    with (
                        patch.object(dist, "is_initialized", return_value=True),
                        patch.object(
                            mm_schedule,
                            "get_parallel",
                            return_value=_parallel(
                                _PeerGroup([1 - rank] * flag_count), rank
                            ),
                        ),
                    ):
                        first_results.append(_run_route(route, items, encode))

                    def unexpected_encode(batch):
                        self.fail("A subsequent unanimous hit entered the encoder")

                    with (
                        patch.object(dist, "is_initialized", return_value=True),
                        patch.object(
                            mm_schedule,
                            "get_parallel",
                            return_value=_parallel(_PeerGroup([0] * flag_count), rank),
                        ),
                    ):
                        next_results.append(_run_route(route, items, unexpected_encode))
                    self.assertTrue(torch.equal(first_results[-1], next_results[-1]))
                    self.assertTrue(torch.equal(first_results[-1], expected))
                    if route == "full":
                        self.assertTrue(
                            cache.has(tuple(item.cache_key for item in items))
                        )
                    self.assertTrue(
                        torch.equal(cache.get_single(99).embedding, torch.ones(1, 2))
                    )
                    self.assertEqual(
                        cache.current_size, fresh.numel() * fresh.element_size() + 8
                    )
                self.assertTrue(torch.equal(next_results[0], next_results[1]))

    def test_dispatcher_uses_agreement_on_each_route(self):
        """Platform fallback and combined requests must retain the agreement gate."""
        for route in ("batch", "by_item", "full"):
            with self.subTest(route=route):
                item = _item(11, 4)
                if route == "full":
                    item.offsets = [(0, 1), (2, 3)]
                _cache(route, [item], [0])
                with (
                    patch.object(dist, "is_initialized", return_value=True),
                    patch.object(
                        mm_schedule,
                        "get_parallel",
                        return_value=_parallel(_PeerGroup([1])),
                    ),
                    patch.object(mm_schedule, "_is_hip", route == "by_item"),
                    patch.object(mm_schedule, "_is_npu", False),
                    patch.object(mm_schedule, "_is_xpu", False),
                ):
                    result, _ = mm_schedule._get_chunked_prefill_embedding(
                        lambda batch: torch.cat([x.feature + 100 for x in batch]),
                        [item],
                        [0, 1],
                        [0],
                        [4],
                        [[(0, 3)]],
                        torch.zeros(4, dtype=torch.long),
                    )
                torch.testing.assert_close(result, item.feature + 100)

    def test_deferred_feature_ack_waits_for_agreement(self):
        """Cache agreement must retain each rank's features for a later eviction."""
        for route in ("full", "by_item"):
            for peer_miss in (False, True):
                for rank in (0, 1, 2, 3):
                    with self.subTest(route=route, peer_miss=peer_miss, rank=rank):
                        item = _item(11, 2)
                        item.set_pad_value()
                        expected = item.feature.clone()
                        _cache(route, [item], [0])
                        mm_schedule.embedding_cache.max_size = 16
                        storage = torch.zeros(512, dtype=torch.uint8)
                        data = storage[256:272]
                        data.copy_(expected.view(torch.uint8).reshape(-1))
                        pool_id = uuid4().hex
                        handles = tuple(
                            (0, pool_id, 512, 0, (pool_id, peer), 0, b"event", False)
                            for peer in range(4)
                        )
                        proxy = CudaIpcTensorTransportProxy(
                            data=data,
                            info_data=expected,
                            pool_ipc_handle=handles[0],
                            pool_ipc_handles=handles,
                            pool_id=pool_id,
                            pool_byte_offset=256,
                            ready_byte_offset=32,
                            ack_byte_offset=64,
                            generation=7,
                            total_consumer_count=4,
                            use_pool_handle_cache=True,
                        )
                        item.feature = proxy
                        item.pad_value = 1001
                        item.model_specific_data[
                            cuda_ipc.DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY
                        ] = True
                        writes = []
                        encoded = []
                        real_empty = torch.empty

                        def cpu_empty(*args, **kwargs):
                            kwargs["device"] = "cpu"
                            return real_empty(*args, **kwargs)

                        def encode(batch):
                            if not encoded:
                                self.assertEqual(
                                    proxy._consumer_acknowledged, not peer_miss
                                )
                            for entry in batch:
                                entry.materialize_deferred_cuda_ipc_feature()
                            encoded.append([entry.feature.clone() for entry in batch])
                            return torch.cat([entry.feature + 100 for entry in batch])

                        with (
                            patch.object(dist, "is_initialized", return_value=True),
                            patch.object(
                                mm_schedule,
                                "get_parallel",
                                return_value=_parallel(
                                    _PeerGroup([int(peer_miss)]), rank % 2
                                ),
                            ),
                            patch.object(
                                memory_pool,
                                "get_parallel",
                                return_value=SimpleNamespace(tp_rank=rank),
                            ),
                            patch.object(
                                schedule_batch,
                                "get_parallel",
                                return_value=SimpleNamespace(
                                    tp_size=4,
                                    tp_rank=rank,
                                    attn_tp_size=2,
                                    attn_tp_rank=rank % 2,
                                    attn_cp_size=2,
                                    attn_cp_rank=rank // 2,
                                ),
                            ),
                            patch.object(torch, "empty", side_effect=cpu_empty),
                            patch.object(torch.cuda, "current_device", return_value=0),
                            patch.object(
                                torch.cuda, "current_stream", return_value=Mock()
                            ),
                            patch.object(
                                torch.cuda, "device", return_value=nullcontext()
                            ),
                            patch.object(
                                cuda_ipc,
                                "_open_pooled_storage_uncached",
                                return_value=storage.untyped_storage(),
                            ),
                            patch.object(memory_pool, "stream_wait_value32"),
                            patch.object(cuda_ipc, "_release_ipc_export"),
                            patch.object(
                                cuda_ipc,
                                "stream_write_value32",
                                side_effect=lambda *args: writes.append(args),
                            ),
                        ):
                            result = _run_route(route, [item], encode)
                            self.assertTrue(proxy._consumer_acknowledged)
                            self.assertEqual(
                                [
                                    (address - storage.data_ptr(), value)
                                    for _, address, value, _ in writes
                                ],
                                [(64 + rank * 4, 7)],
                            )
                            self.assertEqual(len(encoded), int(peer_miss))
                            torch.testing.assert_close(
                                result, expected + (100 if peer_miss else 0)
                            )

                            # Recycle/overwrite the producer bytes, then evict the
                            # embedding using the real bounded LRU admission path.
                            data.fill_(255)
                            mm_schedule.embedding_cache.set(
                                99,
                                mm_schedule.EmbeddingResult(embedding=expected + 200),
                            )
                            result = _run_route(route, [item], encode)
                            torch.testing.assert_close(result, expected + 100)
                            torch.testing.assert_close(item.feature, expected)
                            self.assertEqual(len(encoded), int(peer_miss) + 1)
                            item.release_transport_proxies(4)
                            self.assertEqual(len(writes), 1)

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "Gloo is required"
    )
    def test_two_ranks_enter_matching_encoder_collectives(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _collective_worker,
                args=(f"{directory}/rendezvous",),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
