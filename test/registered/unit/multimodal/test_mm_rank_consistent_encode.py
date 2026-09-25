"""Cache asymmetry must not change encoder collectives or release live features."""

import tempfile
import unittest
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.transport import memory_pool
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
                [item.hash for item in items]
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
                items[i].hash, mm_schedule.EmbeddingResult(embedding=value)
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
            for route in ("batch", "by_item", "full"):
                for mode in ("asymmetric", "all_hit", "all_miss", "stale"):
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
                        identities = [(item.hash, len(item.feature)) for item in batch]
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
        """A peer miss must keep the real proxy live until the encoder consumes it."""
        for route in ("full", "by_item"):
            for peer_miss in (False, True):
                for rank in (0, 1):
                    with self.subTest(route=route, peer_miss=peer_miss, rank=rank):
                        item = _item(11, 2)
                        expected = item.feature.clone()
                        _cache(route, [item], [0])
                        proxy = CudaIpcTensorTransportProxy(
                            data=expected,
                            info_data=expected,
                            pool_ipc_handle=(0,),
                            pool_byte_offset=0,
                            ready_byte_offset=32,
                            ack_byte_offset=64,
                            generation=7,
                            total_consumer_count=4,
                            use_pool_handle_cache=True,
                        )
                        item.feature = proxy
                        writes = []

                        def encode(batch):
                            self.assertFalse(proxy._consumer_acknowledged)
                            proxy.acknowledge_consumption(4)
                            return expected + 100

                        with (
                            patch.object(dist, "is_initialized", return_value=True),
                            patch.object(
                                mm_schedule,
                                "get_parallel",
                                return_value=_parallel(
                                    _PeerGroup([int(peer_miss)]), rank
                                ),
                            ),
                            patch.object(torch.cuda, "current_device", return_value=0),
                            patch.object(
                                torch.cuda, "device", return_value=nullcontext()
                            ),
                            patch.object(
                                proxy,
                                "_open_pool_slice",
                                return_value=(expected, expected.untyped_storage()),
                            ),
                            patch.object(memory_pool, "stream_wait_value32"),
                            patch.object(
                                memory_pool,
                                "stream_write_value32",
                                side_effect=lambda *args: writes.append(args),
                            ),
                        ):
                            result = _run_route(route, [item], encode)
                            self.assertEqual(
                                proxy._consumer_acknowledged, peer_miss or rank == 0
                            )
                            self.assertEqual(
                                len(writes), 4 if peer_miss or rank == 0 else 0
                            )
                            # Subsequent request cleanup must not acknowledge twice.
                            item.release_transport_proxies(4)
                            self.assertEqual(len(writes), 4)
                        torch.testing.assert_close(
                            result, expected + (100 if peer_miss else 0)
                        )

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
