"""Unit tests for HiCache PP synchronization."""

import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeWork:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class _Holder:
    """Minimal carrier exposing only what _drain_async_work touches."""


class TestPPSyncDrain(unittest.TestCase):
    def test_drain_waits_all_and_clears(self):
        holder = _Holder()
        works = [_FakeWork(), _FakeWork(), _FakeWork()]
        holder.work_list = list(works)

        UnifiedRadixCache._drain_async_work(holder)

        self.assertTrue(all(w.waited for w in works))
        self.assertEqual(holder.work_list, [])

    def test_drain_empty_is_noop(self):
        holder = _Holder()
        holder.work_list = []

        UnifiedRadixCache._drain_async_work(holder)

        self.assertEqual(holder.work_list, [])


class TestUnifiedPPSyncBatching(unittest.TestCase):
    def _make_cache(self, pp_rank, write_ready, load_ready):
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(
            enable_storage=False,
            write_back_duplicate_reclaim_digest=0,
        )
        cache.pp_rank = pp_rank
        cache.pp_size = 2
        cache.host_memory_mode = "cache"
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.buffer_pipeline = None
        cache.linker = None
        cache.dflash_checkpoint_pool = None
        cache._drain_async_work = MagicMock()
        cache._all_reduce = MagicMock()
        cache.writing_check = MagicMock()
        cache.loading_check = MagicMock()
        cache.cache_controller = SimpleNamespace(
            start_writing=MagicMock(),
            ack_write_queue=[
                SimpleNamespace(
                    finish_event=SimpleNamespace(query=MagicMock(return_value=ready))
                )
                for ready in write_ready
            ],
            ack_load_queue=[
                SimpleNamespace(
                    finish_event=SimpleNamespace(query=MagicMock(return_value=ready))
                )
                for ready in load_ready
            ],
        )
        return cache

    def test_pp_batches_write_and_load_counts_once(self):
        leader = self._make_cache(0, [True, False], [True, True])
        leader.check_hicache_events()

        leader._all_reduce.assert_called_once()
        self.assertEqual(leader._all_reduce.call_args.args[0].tolist(), [1, 2, 0, 0])
        leader.writing_check.assert_called_once_with(finish_count=1)
        leader.loading_check.assert_called_once_with(finish_count=2)

        follower = self._make_cache(1, [True], [True])

        def reduce_to_min(counts, _):
            counts.copy_(torch.tensor([1, 1, 0, 0], dtype=torch.int64))

        follower._all_reduce.side_effect = reduce_to_min
        follower.check_hicache_events()

        for queue in (
            follower.cache_controller.ack_write_queue,
            follower.cache_controller.ack_load_queue,
        ):
            queue[0].finish_event.query.assert_not_called()
        follower._all_reduce.assert_called_once()
        follower.writing_check.assert_called_once_with(finish_count=1)
        follower.loading_check.assert_called_once_with(finish_count=1)

    def test_buffer_mode_follower_drains_rank_local_completions(self):
        cache = self._make_cache(1, [True, False], [True, True])
        cache.host_memory_mode = "buffer_only"
        cache.tree_core.enable_storage = True
        cache._all_reduce_attn_groups = MagicMock()
        cache._drain_storage_control_queues_impl = MagicMock()
        cc = cache.cache_controller
        for size, name in enumerate(
            (
                "prefetch_hit_queue",
                "ack_prefetch_queue",
                "ack_backup_queue",
                "host_mem_release_queue",
            ),
            start=1,
        ):
            queue = Queue()
            for _ in range(size):
                queue.put(object())
            setattr(cc, name, queue)

        cache.check_hicache_events()

        cache._all_reduce.assert_not_called()
        cache._all_reduce_attn_groups.assert_called_once()
        cache.writing_check.assert_called_once_with(finish_count=1)
        cache.loading_check.assert_called_once_with(finish_count=2)
        cache._drain_storage_control_queues_impl.assert_called_once_with(
            n_storage_hit=1,
            n_ack_prefetch=2,
            n_backup=3,
            n_release=4,
            extra_release_counts={},
            log_metrics=True,
        )


class TestHiCacheReadinessPipeline(CustomTestCase):
    def setUp(self):
        self.cache = object.__new__(UnifiedRadixCache)
        self.cache.tree_core = SimpleNamespace(
            enable_storage=False, write_back_duplicate_reclaim_digest=17
        )
        self.cache.dflash_checkpoint_pool = object()
        self.cache.pp_size = 1
        self.cache.attn_cp_group = None
        self.cache.attn_tp_group = object()
        self.cache.tp_world_size = 2
        self.cache.cache_controller = SimpleNamespace(
            ack_write_queue=[], ack_load_queue=[]
        )
        self.works = []

        def reduce(counts, *, op, group, async_op):
            self.assertTrue(async_op)
            work = _FakeWork()
            self.works.append(work)
            return work

        for name, kwargs in (
            ("get_world_size", {"return_value": 2}),
            ("all_reduce", {"side_effect": reduce}),
        ):
            patcher = patch("torch.distributed." + name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def ack(ready=True):
        return SimpleNamespace(finish_event=SimpleNamespace(query=lambda: ready))

    def test_snapshot_does_not_complete_later_acks_after_blocking_drain(self):
        """A count agreed before eviction must never apply to replacement acks."""
        cc = self.cache.cache_controller
        first, second, third = [self.ack() for _ in range(3)]
        cc.ack_write_queue[:] = [first, second, third]
        cc.ack_load_queue[:] = [self.ack()]
        self.cache._start_hicache_ready_sync()
        pending = self.cache._pending_hicache_ready
        pending.counts[0] = 2  # The other rank has only two ready writes.
        self.assertFalse(self.works[0].waited)

        # Blocking eviction consumes the first write and the old load. New
        # transfers can even be ready already, but belong to a later consensus.
        cc.ack_write_queue.pop(0)
        cc.ack_load_queue[:] = [self.ack()]
        self.assertEqual(self.cache._sync_hicache_ready_counts(), (1, 0, (), ()))
        self.assertTrue(self.works[0].waited)
        self.assertIsNone(self.cache._pending_hicache_ready)
        self.assertEqual(len(self.works), 1)

        cc.ack_write_queue.pop(0)
        self.assertEqual(self.cache._sync_hicache_ready_counts(), (1, 1, (), ()))

    def test_digest_checks_same_snapshot_on_each_rank(self):
        self.cache._start_hicache_ready_sync()
        self.cache.tree_core.write_back_duplicate_reclaim_digest = 23
        self.assertEqual(self.cache._sync_hicache_ready_counts(), (0, 0, (), ()))
        self.cache._start_hicache_ready_sync()
        self.cache._pending_hicache_ready.counts[-1] = -24
        with self.assertRaisesRegex(AssertionError, "victims diverged"):
            self.cache._sync_hicache_ready_counts()

    def test_release_waits_before_destroying_transfer_resources(self):
        self.cache._start_hicache_ready_sync()
        work = self.works[0]
        self.cache.linker = None
        self.cache.host_pool_group = None
        destroyed = []

        def destroy():
            self.assertTrue(work.waited)
            destroyed.append(True)

        self.cache.dflash_checkpoint_pool = SimpleNamespace(destroy=destroy)
        self.cache.release_host_resources()
        self.assertEqual(destroyed, [True])
        self.assertIsNone(self.cache._pending_hicache_ready)


if __name__ == "__main__":
    unittest.main()
