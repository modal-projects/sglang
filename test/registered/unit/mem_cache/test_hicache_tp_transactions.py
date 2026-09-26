"""CPU tests for reversible hybrid HiCache transfers and TP consensus."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import IncLockRefResult
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    CacheOperation,
    HybridCacheController,
    HybridLoadReservation,
    HybridWriteReservation,
)
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
    PrepareLoadBackResult,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _indices(start: int, count: int) -> torch.Tensor:
    return torch.arange(start, start + count, dtype=torch.int64)


class _TrackingPool:
    def __init__(
        self,
        start: int,
        *,
        fail: bool = False,
        raise_on_alloc: bool = False,
        raise_on_free: bool = False,
    ):
        self.next_index = start
        self.fail = fail
        self.raise_on_alloc = raise_on_alloc
        self.raise_on_free = raise_on_free
        self.alloc_calls: list[int] = []
        self.freed: list[torch.Tensor] = []

    def alloc(self, size: int):
        self.alloc_calls.append(size)
        if self.raise_on_alloc:
            raise RuntimeError("injected allocation failure")
        if self.fail:
            return None
        result = _indices(self.next_index, size)
        self.next_index += size
        return result

    def free(self, indices: torch.Tensor) -> None:
        self.freed.append(indices.clone())
        if self.raise_on_free:
            raise RuntimeError("injected free failure")


def _pool_entry(host_pool, device_pool):
    return SimpleNamespace(
        host_pool=host_pool,
        device_pool=device_pool,
        device_alloc_fn=None,
        device_free_fn=None,
        host_evict_fn=mock.Mock(),
        device_evict_fn=mock.Mock(),
    )


def _make_controller(
    *,
    fail_host_pool=None,
    fail_device_pool=None,
    raise_host_pool=None,
    raise_device_pool=None,
    raise_free_host_pool=None,
    raise_free_device_pool=None,
):
    pools = SimpleNamespace(
        primary_host=_TrackingPool(100),
        swa_host=_TrackingPool(
            200,
            fail=fail_host_pool == PoolName.SWA,
            raise_on_alloc=raise_host_pool == PoolName.SWA,
            raise_on_free=raise_free_host_pool == PoolName.SWA,
        ),
        mamba_host=_TrackingPool(
            300,
            fail=fail_host_pool == PoolName.MAMBA,
            raise_on_alloc=raise_host_pool == PoolName.MAMBA,
            raise_on_free=raise_free_host_pool == PoolName.MAMBA,
        ),
        primary_device=_TrackingPool(400),
        swa_device=_TrackingPool(
            500,
            fail=fail_device_pool == PoolName.SWA,
            raise_on_alloc=raise_device_pool == PoolName.SWA,
            raise_on_free=raise_free_device_pool == PoolName.SWA,
        ),
        mamba_device=_TrackingPool(
            600,
            fail=fail_device_pool == PoolName.MAMBA,
            raise_on_alloc=raise_device_pool == PoolName.MAMBA,
            raise_on_free=raise_free_device_pool == PoolName.MAMBA,
        ),
    )
    entries = {
        PoolName.SWA: _pool_entry(pools.swa_host, pools.swa_device),
        PoolName.MAMBA: _pool_entry(pools.mamba_host, pools.mamba_device),
        PoolName.DEEPSEEK_V4_C4: _pool_entry(None, None),
    }
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.mem_pool_host = SimpleNamespace(
        alloc=pools.primary_host.alloc,
        free=pools.primary_host.free,
        available_size=mock.Mock(return_value=1_000),
        entry_map=entries,
    )
    controller.mem_pool_device_allocator = SimpleNamespace(
        full_attn_allocator=pools.primary_device
    )
    controller.device = torch.device("cpu")
    controller.write_queue = []
    controller.load_queue = []
    controller.start_writing = mock.Mock()
    return controller, pools, entries


def _write_transfers():
    return [
        PoolTransfer(name=PoolName.SWA, device_indices=_indices(10, 4)),
        PoolTransfer(name=PoolName.MAMBA, device_indices=_indices(20, 1)),
        PoolTransfer(
            name=PoolName.DEEPSEEK_V4_C4,
            indices_from_pool=PoolName.KV,
        ),
    ]


def _load_transfers():
    mamba_host = _indices(30, 1)
    return [
        PoolTransfer(name=PoolName.SWA, host_indices=_indices(10, 4)),
        PoolTransfer(name=PoolName.MAMBA, host_indices=mamba_host),
        # This request-owned CoW slot is preallocated by MambaComponent and is
        # therefore not owned by the controller transaction.
        PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=mamba_host,
            device_indices=_indices(900, 1),
        ),
        PoolTransfer(
            name=PoolName.DEEPSEEK_V4_C4,
            indices_from_pool=PoolName.KV,
        ),
    ]


def _assert_one_free(test, pool: _TrackingPool, expected: torch.Tensor):
    test.assertEqual(len(pool.freed), 1)
    torch.testing.assert_close(pool.freed[0], expected)


class TestHybridControllerWriteTransaction(unittest.TestCase):
    def test_reserve_then_abort_releases_primary_swa_mamba_and_bindings(self):
        controller, pools, entries = _make_controller()
        transfers = _write_transfers()

        reservation = controller.reserve_write(
            _indices(0, 4), node_id=7, extra_pools=transfers
        )

        self.assertIsInstance(reservation, HybridWriteReservation)
        self.assertEqual(controller.write_queue, [])
        controller.start_writing.assert_not_called()
        for entry in entries.values():
            entry.host_evict_fn.assert_not_called()
        self.assertIsNotNone(transfers[0].host_indices)
        self.assertIsNotNone(transfers[1].host_indices)
        self.assertIs(transfers[2].host_indices, reservation.host_indices)

        controller.abort_write(reservation)

        _assert_one_free(self, pools.primary_host, reservation.host_indices)
        _assert_one_free(self, pools.swa_host, _indices(200, 4))
        _assert_one_free(self, pools.mamba_host, _indices(300, 1))
        self.assertIsNone(transfers[0].host_indices)
        self.assertIsNone(transfers[1].host_indices)
        self.assertIsNone(transfers[2].host_indices)
        self.assertIsNone(transfers[2].device_indices)
        self.assertEqual(controller.write_queue, [])

    def test_commit_populates_real_queue_without_releasing_reservations(self):
        controller, pools, _ = _make_controller()
        transfers = _write_transfers()
        reservation = controller.reserve_write(
            _indices(0, 4), node_id=7, extra_pools=transfers
        )

        host_indices = controller.commit_write(reservation)

        self.assertIs(host_indices, reservation.host_indices)
        self.assertEqual(len(controller.write_queue), 1)
        operation = controller.write_queue[0]
        self.assertIsInstance(operation, CacheOperation)
        self.assertEqual(operation.node_ids, [7])
        self.assertIs(operation.pool_transfers, transfers)
        controller.start_writing.assert_called_once()
        self.assertEqual(pools.primary_host.freed, [])
        self.assertEqual(pools.swa_host.freed, [])
        self.assertEqual(pools.mamba_host.freed, [])

    def test_partial_aux_failure_rolls_back_without_eviction_or_queueing(self):
        controller, pools, entries = _make_controller(fail_host_pool=PoolName.MAMBA)
        transfers = _write_transfers()

        reservation = controller.reserve_write(
            _indices(0, 4), node_id=7, extra_pools=transfers
        )

        self.assertIsNone(reservation)
        _assert_one_free(self, pools.primary_host, _indices(100, 4))
        _assert_one_free(self, pools.swa_host, _indices(200, 4))
        self.assertEqual(pools.mamba_host.freed, [])
        self.assertIsNone(transfers[0].host_indices)
        self.assertIsNone(transfers[1].host_indices)
        self.assertEqual(controller.write_queue, [])
        controller.start_writing.assert_not_called()
        # Reclaim-on-alloc (the local eviction model, kept deliberately): the
        # failing pool's evict callback fires once before the reservation is
        # declared failed; untouched pools never see theirs.
        entries[PoolName.MAMBA].host_evict_fn.assert_called_once_with(1)
        entries[PoolName.SWA].host_evict_fn.assert_not_called()
        entries[PoolName.DEEPSEEK_V4_C4].host_evict_fn.assert_not_called()

    def test_raising_aux_allocator_rolls_back_primary_and_prior_aux(self):
        controller, pools, _ = _make_controller(raise_host_pool=PoolName.MAMBA)
        transfers = _write_transfers()

        with self.assertRaisesRegex(RuntimeError, "injected allocation failure"):
            controller.reserve_write(_indices(0, 4), node_id=7, extra_pools=transfers)

        _assert_one_free(self, pools.primary_host, _indices(100, 4))
        _assert_one_free(self, pools.swa_host, _indices(200, 4))
        self.assertEqual(pools.mamba_host.freed, [])
        self.assertIsNone(transfers[0].host_indices)
        self.assertIsNone(transfers[1].host_indices)
        self.assertEqual(controller.write_queue, [])

    def test_aux_free_error_still_releases_primary_and_other_aux(self):
        controller, pools, _ = _make_controller(raise_free_host_pool=PoolName.SWA)
        transfers = _write_transfers()
        reservation = controller.reserve_write(
            _indices(0, 4), node_id=7, extra_pools=transfers
        )

        with self.assertRaisesRegex(RuntimeError, "injected free failure"):
            controller.abort_write(reservation)

        _assert_one_free(self, pools.primary_host, _indices(100, 4))
        _assert_one_free(self, pools.swa_host, _indices(200, 4))
        _assert_one_free(self, pools.mamba_host, _indices(300, 1))
        self.assertIsNone(transfers[0].host_indices)
        self.assertIsNone(transfers[1].host_indices)
        self.assertEqual(controller.write_queue, [])

    def test_unregistered_explicit_or_derived_pool_aborts_whole_reservation(self):
        missing_transfers = (
            PoolTransfer(name=PoolName.INDEXER, device_indices=_indices(20, 4)),
            PoolTransfer(
                name=PoolName.INDEXER,
                indices_from_pool=PoolName.KV,
            ),
        )
        for missing in missing_transfers:
            with self.subTest(derived=missing.indices_from_pool is not None):
                controller, pools, _ = _make_controller()
                transfers = [
                    PoolTransfer(name=PoolName.SWA, device_indices=_indices(10, 4)),
                    missing,
                ]

                reservation = controller.reserve_write(
                    _indices(0, 4), node_id=7, extra_pools=transfers
                )

                self.assertIsNone(reservation)
                _assert_one_free(self, pools.primary_host, _indices(100, 4))
                _assert_one_free(self, pools.swa_host, _indices(200, 4))
                self.assertIsNone(transfers[0].host_indices)
                self.assertEqual(controller.write_queue, [])

    def test_fire_and_forget_write_still_commits_through_the_reservation(self):
        controller, pools, _ = _make_controller()
        transfers = _write_transfers()

        host_indices = controller.write(
            _indices(0, 4), node_id=7, extra_pools=transfers
        )

        self.assertIsNotNone(host_indices)
        self.assertEqual(len(controller.write_queue), 1)
        controller.start_writing.assert_called_once()
        self.assertEqual(pools.primary_host.freed, [])

    def test_fire_and_forget_write_returns_none_on_reservation_failure(self):
        controller, pools, _ = _make_controller(fail_host_pool=PoolName.MAMBA)
        transfers = _write_transfers()

        host_indices = controller.write(
            _indices(0, 4), node_id=7, extra_pools=transfers
        )

        self.assertIsNone(host_indices)
        self.assertEqual(controller.write_queue, [])
        _assert_one_free(self, pools.primary_host, _indices(100, 4))
        _assert_one_free(self, pools.swa_host, _indices(200, 4))


class TestHybridControllerLoadTransaction(unittest.TestCase):
    def test_reserve_then_abort_releases_owned_primary_swa_mamba_only(self):
        controller, pools, entries = _make_controller()
        transfers = _load_transfers()
        request_mamba_indices = transfers[2].device_indices

        reservation = controller.reserve_load(
            _indices(0, 4), node_id=9, extra_pools=transfers
        )

        self.assertIsInstance(reservation, HybridLoadReservation)
        self.assertEqual(controller.load_queue, [])
        for entry in entries.values():
            entry.device_evict_fn.assert_not_called()
        self.assertIsNotNone(transfers[0].device_indices)
        self.assertIsNotNone(transfers[1].device_indices)
        self.assertIs(transfers[3].device_indices, reservation.device_indices)

        controller.abort_load(reservation)

        _assert_one_free(self, pools.primary_device, reservation.device_indices)
        _assert_one_free(self, pools.swa_device, _indices(500, 4))
        _assert_one_free(self, pools.mamba_device, _indices(600, 1))
        self.assertIsNone(transfers[0].device_indices)
        self.assertIsNone(transfers[1].device_indices)
        self.assertIs(transfers[2].device_indices, request_mamba_indices)
        self.assertIsNone(transfers[3].host_indices)
        self.assertIsNone(transfers[3].device_indices)
        self.assertEqual(controller.load_queue, [])

    def test_commit_populates_real_queue_without_releasing_reservations(self):
        controller, pools, _ = _make_controller()
        transfers = _load_transfers()
        reservation = controller.reserve_load(
            _indices(0, 4), node_id=9, extra_pools=transfers
        )

        device_indices = controller.commit_load(reservation)

        self.assertIs(device_indices, reservation.device_indices)
        self.assertEqual(len(controller.load_queue), 1)
        operation = controller.load_queue[0]
        self.assertIsInstance(operation, CacheOperation)
        self.assertEqual(operation.node_ids, [9])
        self.assertIs(operation.pool_transfers, transfers)
        self.assertEqual(pools.primary_device.freed, [])
        self.assertEqual(pools.swa_device.freed, [])
        self.assertEqual(pools.mamba_device.freed, [])

    def test_partial_aux_failure_rolls_back_without_eviction_or_queueing(self):
        controller, pools, entries = _make_controller(fail_device_pool=PoolName.MAMBA)
        transfers = _load_transfers()

        reservation = controller.reserve_load(
            _indices(0, 4), node_id=9, extra_pools=transfers
        )

        self.assertIsNone(reservation)
        _assert_one_free(self, pools.primary_device, _indices(400, 4))
        _assert_one_free(self, pools.swa_device, _indices(500, 4))
        self.assertEqual(pools.mamba_device.freed, [])
        self.assertIsNone(transfers[0].device_indices)
        self.assertIsNone(transfers[1].device_indices)
        self.assertEqual(controller.load_queue, [])
        # Same reclaim-on-alloc contract as the write side.
        entries[PoolName.MAMBA].device_evict_fn.assert_called_once_with(1)
        entries[PoolName.SWA].device_evict_fn.assert_not_called()
        entries[PoolName.DEEPSEEK_V4_C4].device_evict_fn.assert_not_called()

    def test_raising_aux_allocator_rolls_back_primary_and_prior_aux(self):
        controller, pools, _ = _make_controller(raise_device_pool=PoolName.MAMBA)
        transfers = _load_transfers()

        with self.assertRaisesRegex(RuntimeError, "injected allocation failure"):
            controller.reserve_load(_indices(0, 4), node_id=9, extra_pools=transfers)

        _assert_one_free(self, pools.primary_device, _indices(400, 4))
        _assert_one_free(self, pools.swa_device, _indices(500, 4))
        self.assertEqual(pools.mamba_device.freed, [])
        self.assertIsNone(transfers[0].device_indices)
        self.assertIsNone(transfers[1].device_indices)
        self.assertEqual(controller.load_queue, [])

    def test_zero_kv_load_holds_no_primary_and_aborts_side_pools_only(self):
        controller, pools, _ = _make_controller()
        transfers = [
            PoolTransfer(name=PoolName.SWA, host_indices=_indices(10, 4)),
        ]

        reservation = controller.reserve_load(
            torch.empty((0,), dtype=torch.int64), node_id=11, extra_pools=transfers
        )

        self.assertIsNotNone(reservation)
        self.assertEqual(reservation.device_indices.numel(), 0)
        self.assertIsNone(reservation.primary_free_fn)
        self.assertEqual(pools.primary_device.alloc_calls, [])

        controller.abort_load(reservation)

        self.assertEqual(pools.primary_device.freed, [])
        _assert_one_free(self, pools.swa_device, _indices(500, 4))
        self.assertIsNone(transfers[0].device_indices)
        self.assertEqual(controller.load_queue, [])

    def test_fire_and_forget_load_still_commits_through_the_reservation(self):
        controller, pools, _ = _make_controller()
        transfers = _load_transfers()

        device_indices = controller.load(
            _indices(0, 4), node_id=9, extra_pools=transfers
        )

        self.assertIsNotNone(device_indices)
        self.assertEqual(len(controller.load_queue), 1)
        self.assertEqual(pools.primary_device.freed, [])


def _component(component_type, *, prep=None):
    component = mock.Mock()
    component.component_type = component_type
    component.prepare_load_back.return_value = prep or PrepareLoadBackResult()
    return component


def _make_unified_cache(controller, components=()):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.cache_controller = controller
    cache.tp_world_size = 2
    cache._dedup_transaction_depth = 0
    cache.pp_rank = 0
    cache.pp_size = 1
    cache.pp_group = None
    cache.attn_cp_group = None
    cache.attn_tp_group = None
    cache.tp_group = object()
    cache._components_tuple = tuple(components)
    cache.sidecar_pool_specs = []
    cache.buffer_pipeline = None
    cache.load_back_threshold = 1
    cache.ongoing_load_back = {}
    return cache


@contextlib.contextmanager
def _mocked_tp_group(
    *, peer_ok=True, peer_error=False, before_gather=None, gather_error=None
):
    """Mock a 2-rank TP group: rank 0 is local, rank 1 is the peer."""

    def all_gather(gathered, local, group=None):
        if gather_error is not None:
            raise gather_error
        if before_gather is not None:
            before_gather(local)
        gathered[0].copy_(local)
        peer = local.clone()
        if not peer_ok:
            peer[0] = 0
        if peer_error:
            peer[0] = 0
            peer[1] = 1
        gathered[1].copy_(peer)

    with (
        mock.patch.object(
            torch.distributed, "all_gather", side_effect=all_gather
        ) as gather_mock,
        mock.patch.object(torch.distributed, "get_world_size", return_value=2),
    ):
        yield gather_mock


class TestTPTransactionConsensus(unittest.TestCase):
    """The fixed-shape fingerprint validation, as a pure function."""

    def _consensus(self, cache, **kwargs):
        return UnifiedRadixCache._tp_transaction_consensus(cache, **kwargs)

    def test_fingerprint_rejects_opcode_mismatch_and_reports_peer_error(self):
        controller, _, _ = _make_controller()
        cache = _make_unified_cache(controller)

        def opcode_mismatch(local):
            def patch(gathered, tensor, group=None):
                gathered[0].copy_(tensor)
                peer = tensor.clone()
                peer[2] = 2
                gathered[1].copy_(peer)

            return mock.patch.object(torch.distributed, "all_gather", patch)

        with (
            opcode_mismatch(None),
            mock.patch.object(torch.distributed, "get_world_size", return_value=2),
        ):
            group_ok, peer_error = self._consensus(
                cache,
                local_ok=True,
                local_error=False,
                opcode=1,
                node_id=17,
                target_tokens=4,
                mamba_tree_rows=1,
                request_mamba_rows=0,
            )
        self.assertFalse(group_ok)
        self.assertFalse(peer_error)

        with _mocked_tp_group(peer_error=True) as gather_mock:
            group_ok, peer_error = self._consensus(
                cache,
                local_ok=True,
                local_error=False,
                opcode=1,
                node_id=17,
                target_tokens=4,
                mamba_tree_rows=1,
                request_mamba_rows=0,
            )
        gather_mock.assert_called_once()
        self.assertFalse(group_ok)
        self.assertTrue(peer_error)

    def test_single_rank_short_circuits_without_a_collective(self):
        controller, _, _ = _make_controller()
        cache = _make_unified_cache(controller)
        cache.tp_group = None

        with mock.patch.object(torch.distributed, "all_gather") as gather_mock:
            group_ok, peer_error = self._consensus(
                cache,
                local_ok=False,
                local_error=False,
                opcode=1,
                node_id=17,
                target_tokens=4,
                mamba_tree_rows=0,
                request_mamba_rows=0,
            )

        gather_mock.assert_not_called()
        self.assertFalse(group_ok)
        self.assertFalse(peer_error)


class TestDedupConsensusGroupSelection(unittest.TestCase):
    """The consensus spans exactly the broadcast group's ranks."""

    def _gathered_group(self, *, dp_attention: bool):
        controller, _, _ = _make_controller()
        cache = _make_unified_cache(controller)
        tp_sentinel = object()
        attn_sentinel = object()
        cache.tp_group = tp_sentinel
        cache.attn_tp_group = attn_sentinel
        seen = {}

        def all_gather(gathered, local, group=None):
            seen["group"] = group
            gathered[0].copy_(local)
            gathered[1].copy_(local)

        with (
            mock.patch.object(torch.distributed, "all_gather", side_effect=all_gather),
            mock.patch.object(torch.distributed, "get_world_size", return_value=2),
            mock.patch(
                "sglang.srt.mem_cache.unified_radix_cache.is_dp_attention_enabled",
                return_value=dp_attention,
            ),
        ):
            cache._tp_transaction_consensus(
                local_ok=True,
                local_error=False,
                opcode=1,
                node_id=1,
                target_tokens=1,
                mamba_tree_rows=0,
                request_mamba_rows=0,
            )
        return seen["group"], tp_sentinel, attn_sentinel

    def test_without_dp_attention_the_consensus_uses_the_full_tp_group(self):
        group, tp_sentinel, _ = self._gathered_group(dp_attention=False)
        self.assertIs(group, tp_sentinel)

    def test_with_dp_attention_the_consensus_uses_the_attn_tp_group(self):
        group, _, attn_sentinel = self._gathered_group(dp_attention=True)
        self.assertIs(group, attn_sentinel)


class TestUnifiedWriteConsensus(unittest.TestCase):
    def _setup(self, *, raise_host_pool=None, fail_host_pool=None, dedup_active=True):
        controller, pools, entries = _make_controller(
            raise_host_pool=raise_host_pool,
            fail_host_pool=fail_host_pool,
        )
        controller.mla_broadcaster = (
            SimpleNamespace(is_src=True) if dedup_active else None
        )
        cache = _make_unified_cache(controller)
        device_value = _indices(0, 4)
        swa_transfer = PoolTransfer(name=PoolName.SWA, device_indices=_indices(10, 4))
        mamba_transfer = PoolTransfer(
            name=PoolName.MAMBA, device_indices=_indices(20, 1)
        )
        comp_xfers = {
            ComponentType.SWA: [swa_transfer],
            ComponentType.MAMBA: [mamba_transfer],
        }
        cache.tree_core = mock.Mock()
        cache.tree_core.build_backup_spec.return_value = (device_value, comp_xfers)
        cache.evict_host = mock.Mock()
        cache.inc_lock_ref = mock.Mock(return_value=IncLockRefResult(delta=0))
        cache._track_write_through_node = mock.Mock()
        return (
            cache,
            controller,
            pools,
            entries,
            device_value,
            (swa_transfer, mamba_transfer),
        )

    @staticmethod
    def _run_backup(cache, node_id=7, write_back=False):
        action = SimpleNamespace(node_ids=[node_id])
        return cache._execute_and_commit_kv_backup(action, write_back=write_back)

    def test_success_queues_then_commits_tree_state(self):
        (
            cache,
            controller,
            pools,
            _,
            device_value,
            (swa_transfer, mamba_transfer),
        ) = self._setup()

        def before_gather(local):
            # The fingerprint covers the exact DMA shape: opcode 1 (write),
            # node 7, 4 KV tokens, 1 mamba tree row, no request slot.
            self.assertEqual(local.tolist(), [1, 0, 1, 7, 4, 1, 0])
            self.assertEqual(controller.write_queue, [])
            cache.tree_core.commit_backup.assert_not_called()
            cache.inc_lock_ref.assert_not_called()
            cache._track_write_through_node.assert_not_called()
            self.assertEqual(pools.primary_host.alloc_calls, [4])
            self.assertEqual(pools.swa_host.alloc_calls, [4])
            self.assertEqual(pools.mamba_host.alloc_calls, [1])

        with _mocked_tp_group(before_gather=before_gather) as gather_mock:
            written = self._run_backup(cache)

        self.assertEqual(written, 4)
        gather_mock.assert_called_once()
        self.assertEqual(len(controller.write_queue), 1)
        operation = controller.write_queue[0]
        self.assertIsInstance(operation, CacheOperation)
        self.assertEqual(operation.node_ids, [7])
        controller.start_writing.assert_called_once()
        self.assertEqual(pools.primary_host.freed, [])
        self.assertEqual(pools.swa_host.freed, [])
        self.assertEqual(pools.mamba_host.freed, [])
        cache.tree_core.commit_backup.assert_called_once()
        cache.inc_lock_ref.assert_called_once_with(7)
        cache._track_write_through_node.assert_called_once()

    def test_peer_failure_aborts_every_allocation_without_queue_or_mutation(self):
        (
            cache,
            controller,
            pools,
            entries,
            _,
            (swa_transfer, mamba_transfer),
        ) = self._setup()

        with _mocked_tp_group(peer_ok=False) as gather_mock:
            written = self._run_backup(cache)

        self.assertEqual(written, 0)
        gather_mock.assert_called_once()
        self.assertEqual(controller.write_queue, [])
        controller.start_writing.assert_not_called()
        _assert_one_free(self, pools.primary_host, _indices(100, 4))
        _assert_one_free(self, pools.swa_host, _indices(200, 4))
        _assert_one_free(self, pools.mamba_host, _indices(300, 1))
        self.assertIsNone(swa_transfer.host_indices)
        self.assertIsNone(mamba_transfer.host_indices)
        cache.tree_core.commit_backup.assert_not_called()
        cache.inc_lock_ref.assert_not_called()
        cache._track_write_through_node.assert_not_called()

    def test_local_soft_failure_participates_and_rejects_symmetrically(self):
        (
            cache,
            controller,
            pools,
            entries,
            _,
            (swa_transfer, mamba_transfer),
        ) = self._setup(fail_host_pool=PoolName.MAMBA)

        def before_gather(local):
            # Local reservation failed softly: ok=0, error=0, shape intact.
            self.assertEqual(local.tolist(), [0, 0, 1, 7, 4, 1, 0])

        with _mocked_tp_group(before_gather=before_gather) as gather_mock:
            written = self._run_backup(cache)

        self.assertEqual(written, 0)
        gather_mock.assert_called_once()
        _assert_one_free(self, pools.primary_host, _indices(100, 4))
        _assert_one_free(self, pools.swa_host, _indices(200, 4))
        self.assertEqual(pools.mamba_host.freed, [])
        self.assertIsNone(swa_transfer.host_indices)
        self.assertIsNone(mamba_transfer.host_indices)
        self.assertEqual(controller.write_queue, [])
        cache.tree_core.commit_backup.assert_not_called()

    def test_local_allocator_exception_is_cleaned_then_reduced_as_failure(self):
        (
            cache,
            controller,
            pools,
            _,
            _,
            (swa_transfer, mamba_transfer),
        ) = self._setup(raise_host_pool=PoolName.MAMBA)

        def before_gather(local):
            self.assertEqual(local.tolist(), [0, 1, 1, 7, 4, 1, 0])

        with (
            _mocked_tp_group(before_gather=before_gather) as gather_mock,
            self.assertRaisesRegex(RuntimeError, "injected allocation failure"),
        ):
            self._run_backup(cache)

        gather_mock.assert_called_once()
        _assert_one_free(self, pools.primary_host, _indices(100, 4))
        _assert_one_free(self, pools.swa_host, _indices(200, 4))
        self.assertEqual(pools.mamba_host.freed, [])
        self.assertIsNone(swa_transfer.host_indices)
        self.assertIsNone(mamba_transfer.host_indices)
        self.assertEqual(controller.write_queue, [])
        cache.tree_core.commit_backup.assert_not_called()

    def test_consensus_exception_aborts_every_local_reservation(self):
        (
            cache,
            controller,
            pools,
            _,
            _,
            (swa_transfer, mamba_transfer),
        ) = self._setup()

        with (
            _mocked_tp_group(
                gather_error=RuntimeError("injected gather failure")
            ) as gather_mock,
            self.assertRaisesRegex(RuntimeError, "injected gather failure"),
        ):
            self._run_backup(cache)

        gather_mock.assert_called_once()
        _assert_one_free(self, pools.primary_host, _indices(100, 4))
        _assert_one_free(self, pools.swa_host, _indices(200, 4))
        _assert_one_free(self, pools.mamba_host, _indices(300, 1))
        self.assertIsNone(swa_transfer.host_indices)
        self.assertIsNone(mamba_transfer.host_indices)
        self.assertEqual(controller.write_queue, [])
        cache.tree_core.commit_backup.assert_not_called()

    def test_multi_rank_write_back_runs_the_consensus_and_demotes_on_success(self):
        cache, controller, pools, _, _, _ = self._setup()

        with _mocked_tp_group() as gather_mock:
            written = self._run_backup(cache, write_back=True)

        self.assertEqual(written, 4)
        gather_mock.assert_called_once()
        self.assertEqual(len(controller.write_queue), 1)
        cache.tree_core.commit_backup.assert_called_once()
        # Write-back demotes instead of taking the write-through lock.
        cache.inc_lock_ref.assert_not_called()

    def test_admission_failure_still_enters_the_gather_and_rejects(self):
        (
            cache,
            controller,
            pools,
            _,
            _,
            _,
        ) = self._setup()
        controller.mem_pool_host.available_size.return_value = 0
        cache.evict_host.return_value = 0

        def before_gather(local):
            # A rank that cannot make room votes instead of returning
            # early: ok=0, error=0, shape intact, so the group rejects
            # together and no peer is stranded in the gather.
            self.assertEqual(local.tolist(), [0, 0, 1, 7, 4, 1, 0])

        with _mocked_tp_group(before_gather=before_gather) as gather_mock:
            written = self._run_backup(cache)

        self.assertEqual(written, 0)
        gather_mock.assert_called_once()
        cache.evict_host.assert_called_once_with(4)
        self.assertEqual(pools.primary_host.alloc_calls, [])
        self.assertEqual(controller.write_queue, [])
        controller.start_writing.assert_not_called()
        cache.tree_core.commit_backup.assert_not_called()
        cache.inc_lock_ref.assert_not_called()

    def test_nested_write_back_commits_locally_without_a_nested_gather(self):
        cache, controller, pools, _, _, _ = self._setup()
        # A reclaim callback re-entering the backup seam while an outer
        # dedup transaction is in flight on this thread must not join a
        # second consensus gather; it commits the identical reservation
        # locally instead.
        cache._dedup_transaction_depth = 1

        with _mocked_tp_group() as gather_mock:
            written = self._run_backup(cache)

        self.assertEqual(written, 4)
        gather_mock.assert_not_called()
        self.assertEqual(len(controller.write_queue), 1)
        controller.start_writing.assert_called_once()
        cache.tree_core.commit_backup.assert_called_once()
        # The guard is owned by the outer transaction; the nested call
        # leaves it untouched.
        self.assertEqual(cache._dedup_transaction_depth, 1)

    def test_admission_exception_is_reduced_as_error_and_reraised(self):
        cache, controller, pools, _, _, _ = self._setup()
        controller.mem_pool_host.available_size.return_value = 0
        cache.evict_host.side_effect = RuntimeError("injected eviction failure")

        def before_gather(local):
            # The exception is carried into the consensus as error=1, so
            # the group rejects together instead of stranding peers.
            self.assertEqual(local.tolist(), [0, 1, 1, 7, 4, 1, 0])

        with (
            _mocked_tp_group(before_gather=before_gather) as gather_mock,
            self.assertRaisesRegex(RuntimeError, "injected eviction failure"),
        ):
            self._run_backup(cache)

        gather_mock.assert_called_once()
        self.assertEqual(pools.primary_host.alloc_calls, [])
        self.assertEqual(controller.write_queue, [])
        cache.tree_core.commit_backup.assert_not_called()

    def test_transaction_depth_returns_to_zero_after_commit_and_reject(self):
        for peer_ok in (True, False):
            with self.subTest(peer_ok=peer_ok):
                cache, _, _, _, _, _ = self._setup()

                with _mocked_tp_group(peer_ok=peer_ok):
                    self._run_backup(cache)

                self.assertEqual(cache._dedup_transaction_depth, 0)

    def test_non_dedup_path_runs_no_collective(self):
        cache, controller, pools, _, _, _ = self._setup(dedup_active=False)

        with _mocked_tp_group() as gather_mock:
            written = self._run_backup(cache)

        self.assertEqual(written, 4)
        gather_mock.assert_not_called()
        self.assertEqual(len(controller.write_queue), 1)
        cache.tree_core.commit_backup.assert_called_once()


class TestUnifiedLoadConsensus(unittest.TestCase):
    def _setup(
        self,
        *,
        raise_device_pool=None,
        dedup_active=True,
        kv_tokens=4,
        request_slot=True,
    ):
        controller, pools, entries = _make_controller(
            raise_device_pool=raise_device_pool
        )
        controller.mla_broadcaster = (
            SimpleNamespace(is_src=True) if dedup_active else None
        )
        prep = PrepareLoadBackResult(
            allocated_mamba_slot=_indices(700, 1) if request_slot else None
        )
        mamba_comp = _component(ComponentType.MAMBA, prep=prep)
        cache = _make_unified_cache(controller, [mamba_comp])
        kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=_indices(0, kv_tokens))
        tree_transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=_indices(30, 1),
            nodes_to_load=[9],
        )
        request_transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=_indices(30, 1),
            device_indices=prep.allocated_mamba_slot,
        )
        comp_xfers = {ComponentType.MAMBA: [tree_transfer, request_transfer]}
        cache.tree_core = mock.Mock()
        cache.tree_core.build_load_back_spec.return_value = (kv_xfer, comp_xfers)
        cache.tree_core.commit_load_back.return_value = []
        cache._component_available_size = mock.Mock(return_value=1_000)
        cache.token_to_kv_pool_allocator = SimpleNamespace()
        cache.evict_for_alloc = mock.Mock()
        cache.inc_lock_ref = mock.Mock(
            side_effect=lambda node_id: IncLockRefResult(delta=0, node_id=node_id)
        )
        cache.inc_host_lock_ref = mock.Mock(
            side_effect=lambda node_id: IncLockRefResult(delta=0, node_id=node_id)
        )
        cache.dec_lock_ref = mock.Mock()
        cache.dec_host_lock_ref = mock.Mock()
        cache._apply_cache_actions = mock.Mock()
        return (
            cache,
            controller,
            pools,
            entries,
            kv_xfer,
            (tree_transfer, request_transfer),
            mamba_comp,
        )

    def test_success_queues_then_commits_tree_and_request_state(self):
        (
            cache,
            controller,
            pools,
            _,
            kv_xfer,
            (tree_transfer, request_transfer),
            mamba_comp,
        ) = self._setup()

        def before_gather(local):
            # opcode 2 (load), node 9, 4 KV tokens, 1 mamba tree row,
            # 1 request-slot row from prepare_load_back.
            self.assertEqual(local.tolist(), [1, 0, 2, 9, 4, 1, 1])
            self.assertEqual(controller.load_queue, [])
            cache.tree_core.commit_load_back.assert_not_called()

        with _mocked_tp_group(before_gather=before_gather) as gather_mock:
            loaded = cache.load_back(9)

        self.assertTrue(loaded)
        gather_mock.assert_called_once()
        self.assertEqual(len(controller.load_queue), 1)
        operation = controller.load_queue[0]
        self.assertIsInstance(operation, CacheOperation)
        self.assertEqual(operation.node_ids, [9])
        self.assertEqual(pools.primary_device.freed, [])
        self.assertEqual(pools.mamba_device.freed, [])
        torch.testing.assert_close(tree_transfer.device_indices, _indices(600, 1))
        torch.testing.assert_close(request_transfer.device_indices, _indices(700, 1))
        cache.tree_core.commit_load_back.assert_called_once()
        cache._apply_cache_actions.assert_called_once()
        self.assertIn(9, cache.ongoing_load_back)
        cache.dec_lock_ref.assert_called_once()
        cache.dec_host_lock_ref.assert_not_called()
        prep = mamba_comp.prepare_load_back.return_value
        mamba_comp.finalize_load_back.assert_called_once_with(None, prep, True)

    def test_peer_failure_aborts_primary_mamba_and_the_request_slot(self):
        (
            cache,
            controller,
            pools,
            _,
            _,
            (tree_transfer, request_transfer),
            mamba_comp,
        ) = self._setup()

        with _mocked_tp_group(peer_ok=False) as gather_mock:
            loaded = cache.load_back(9)

        self.assertFalse(loaded)
        gather_mock.assert_called_once()
        # No broadcast collective is ever queued on this rank: the op never
        # reaches the load queue that start_loading drains.
        self.assertEqual(controller.load_queue, [])
        _assert_one_free(self, pools.primary_device, _indices(400, 4))
        _assert_one_free(self, pools.mamba_device, _indices(600, 1))
        # Allocator metadata returns to its pre-reservation state, so every
        # rank's (dummy) pools stay identical after the abort.
        self.assertIsNone(tree_transfer.device_indices)
        torch.testing.assert_close(request_transfer.device_indices, _indices(700, 1))
        cache.tree_core.commit_load_back.assert_not_called()
        self.assertEqual(cache.ongoing_load_back, {})
        cache.dec_lock_ref.assert_called_once()
        cache.dec_host_lock_ref.assert_called_once()
        # The request slot held by prepare_load_back is released through the
        # component's failure finalize.
        prep = mamba_comp.prepare_load_back.return_value
        mamba_comp.finalize_load_back.assert_called_once_with(None, prep, False)

    def test_local_allocator_exception_is_cleaned_then_reduced_as_failure(self):
        (
            cache,
            controller,
            pools,
            _,
            _,
            (tree_transfer, _),
            mamba_comp,
        ) = self._setup(raise_device_pool=PoolName.MAMBA)

        def before_gather(local):
            self.assertEqual(local.tolist(), [0, 1, 2, 9, 4, 1, 1])

        with (
            _mocked_tp_group(before_gather=before_gather) as gather_mock,
            self.assertRaisesRegex(RuntimeError, "injected allocation failure"),
        ):
            cache.load_back(9)

        gather_mock.assert_called_once()
        _assert_one_free(self, pools.primary_device, _indices(400, 4))
        self.assertEqual(pools.mamba_device.freed, [])
        self.assertIsNone(tree_transfer.device_indices)
        self.assertEqual(controller.load_queue, [])
        cache.tree_core.commit_load_back.assert_not_called()
        prep = mamba_comp.prepare_load_back.return_value
        mamba_comp.finalize_load_back.assert_called_once_with(None, prep, False)

    def test_consensus_exception_aborts_controller_and_request_mamba(self):
        (
            cache,
            controller,
            pools,
            _,
            _,
            (tree_transfer, _),
            mamba_comp,
        ) = self._setup()

        with (
            _mocked_tp_group(
                gather_error=RuntimeError("injected gather failure")
            ) as gather_mock,
            self.assertRaisesRegex(RuntimeError, "injected gather failure"),
        ):
            cache.load_back(9)

        gather_mock.assert_called_once()
        _assert_one_free(self, pools.primary_device, _indices(400, 4))
        _assert_one_free(self, pools.mamba_device, _indices(600, 1))
        self.assertIsNone(tree_transfer.device_indices)
        self.assertEqual(controller.load_queue, [])
        cache.tree_core.commit_load_back.assert_not_called()
        cache.dec_host_lock_ref.assert_called_once()
        prep = mamba_comp.prepare_load_back.return_value
        mamba_comp.finalize_load_back.assert_called_once_with(None, prep, False)

    def test_single_rank_short_circuits_without_a_collective(self):
        cache, controller, pools, _, _, _, _ = self._setup()
        cache.tp_world_size = 1
        cache.tp_group = None

        with mock.patch.object(torch.distributed, "all_gather") as gather_mock:
            loaded = cache.load_back(9)

        self.assertTrue(loaded)
        gather_mock.assert_not_called()
        self.assertEqual(len(controller.load_queue), 1)
        cache.tree_core.commit_load_back.assert_called_once()

    def test_non_dedup_path_runs_no_collective(self):
        cache, controller, pools, _, _, _, _ = self._setup(dedup_active=False)

        with _mocked_tp_group() as gather_mock:
            loaded = cache.load_back(9)

        self.assertTrue(loaded)
        gather_mock.assert_not_called()
        self.assertEqual(len(controller.load_queue), 1)
        cache.tree_core.commit_load_back.assert_called_once()

    def test_admission_failure_still_enters_the_gather_and_rejects(self):
        (
            cache,
            controller,
            pools,
            _,
            _,
            _,
            mamba_comp,
        ) = self._setup()
        cache._component_available_size.return_value = 0

        def before_gather(local):
            # Same contract as the write seam: a local capacity shortfall
            # is a vote (ok=0), never an early return that strands peers.
            self.assertEqual(local.tolist(), [0, 0, 2, 9, 4, 1, 1])

        with _mocked_tp_group(before_gather=before_gather) as gather_mock:
            loaded = cache.load_back(9)

        self.assertFalse(loaded)
        gather_mock.assert_called_once()
        cache.evict_for_alloc.assert_called_once()
        self.assertEqual(pools.primary_device.alloc_calls, [])
        self.assertEqual(controller.load_queue, [])
        cache.tree_core.commit_load_back.assert_not_called()
        cache.dec_lock_ref.assert_called_once()
        cache.dec_host_lock_ref.assert_called_once()
        prep = mamba_comp.prepare_load_back.return_value
        mamba_comp.finalize_load_back.assert_called_once_with(None, prep, False)

    def test_admission_exception_is_reduced_as_error_and_reraised(self):
        (
            cache,
            controller,
            pools,
            _,
            _,
            _,
            mamba_comp,
        ) = self._setup()
        cache._component_available_size.return_value = 0
        cache.evict_for_alloc.side_effect = RuntimeError("injected eviction failure")

        def before_gather(local):
            self.assertEqual(local.tolist(), [0, 1, 2, 9, 4, 1, 1])

        with (
            _mocked_tp_group(before_gather=before_gather) as gather_mock,
            self.assertRaisesRegex(RuntimeError, "injected eviction failure"),
        ):
            cache.load_back(9)

        gather_mock.assert_called_once()
        self.assertEqual(pools.primary_device.alloc_calls, [])
        self.assertEqual(controller.load_queue, [])
        cache.tree_core.commit_load_back.assert_not_called()
        cache.dec_lock_ref.assert_called_once()
        cache.dec_host_lock_ref.assert_called_once()
        prep = mamba_comp.prepare_load_back.return_value
        mamba_comp.finalize_load_back.assert_called_once_with(None, prep, False)

    def test_zero_kv_mamba_only_load_commits_sidecar_only(self):
        cache, controller, pools, _, kv_xfer, (tree_transfer, _), _ = self._setup(
            kv_tokens=0
        )

        def before_gather(local):
            # Empty KV payload: the sidecar rows still shape the fingerprint.
            self.assertEqual(local.tolist(), [1, 0, 2, 9, 0, 1, 1])

        with _mocked_tp_group(before_gather=before_gather) as gather_mock:
            loaded = cache.load_back(9)

        self.assertTrue(loaded)
        gather_mock.assert_called_once()
        self.assertEqual(pools.primary_device.alloc_calls, [])
        self.assertEqual(len(controller.load_queue), 1)
        operation = controller.load_queue[0]
        self.assertEqual(operation.device_indices.numel(), 0)
        torch.testing.assert_close(tree_transfer.device_indices, _indices(600, 1))
        cache.tree_core.commit_load_back.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
