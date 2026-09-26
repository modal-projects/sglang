"""Unit tests for HiCache staged write-back host-pool dispatch."""

import unittest
from array import array
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers import cache_controller as manager_cache_controller
from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache import l2_transfer as transfer_module
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.buffer_mode.pipeline import BufferModePipeline
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
    LogicalHostPool,
)
from sglang.srt.mem_cache.mla_host_dedup import enforce_hicache_host_budget
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MEMORY_POOL_HOST_MODULE = "sglang.srt.mem_cache.memory_pool_host"
DSA_POOL_HOST_MODULE = "sglang.srt.mem_cache.pool_host.dsa"
MHA_POOL_HOST_MODULE = "sglang.srt.mem_cache.pool_host.mha"
MLA_POOL_HOST_MODULE = "sglang.srt.mem_cache.pool_host.mla"


def _indices(start: int, end: int) -> torch.Tensor:
    return torch.arange(start, end, dtype=torch.int64)


def _ptr_key_from_layers(src_layers) -> tuple[int, ...]:
    return tuple(int(src_layers[i].data_ptr()) for i in range(len(src_layers)))


def _ptr_key_from_tensor(ptrs: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(ptr) for ptr in ptrs.cpu().tolist())


def _device_pool_stub(*, layer_num: int, **fields) -> SimpleNamespace:
    """Minimal device-pool stand-in with layer-split fields real pools expose."""
    return SimpleNamespace(
        layer_num=layer_num,
        layer_shard_enabled=False,
        **fields,
    )


def _host_group_stub(captured, *, can_use_write_back_jit: bool) -> SimpleNamespace:
    class FakeHostPool:
        size_per_token = 2

        def backup_from_device_all_layer(
            self, device_pool, host_indices, device_indices, io_backend
        ):
            captured.append(host_indices)

    entries = [
        PoolEntry(
            name=name,
            host_pool=FakeHostPool(),
            device_pool=None,
            layer_mapper=lambda layer_id: layer_id,
            is_primary_index_anchor=name == PoolName.KV,
        )
        for name in (PoolName.KV, PoolName.SWA, PoolName.DEEPSEEK_V4_C4)
    ]
    return SimpleNamespace(
        layout="page_first",
        can_use_write_back_jit=can_use_write_back_jit,
        anchor_entry=entries[0],
        entry_map={entry.name: entry for entry in entries},
    )


def _cpu_staged_lf_pf_copy(
    src_registry,
    *,
    ptr_src,
    src_indices,
    dst_indices,
    dst,
    **_,
):
    src_layers = src_registry[_ptr_key_from_tensor(ptr_src)]
    src_indices = src_indices.to(dtype=torch.int64, device="cpu")
    dst_indices = dst_indices.to(dtype=torch.int64, device="cpu")
    for layer_id, src in enumerate(src_layers):
        dst[dst_indices, layer_id] = src[src_indices]


def _cpu_staged_mha_lf_pf_copy(
    src_registry,
    *,
    k_ptr_src,
    v_ptr_src,
    src_indices,
    dst_indices,
    dst_k,
    dst_v,
    **_,
):
    k_src_layers = src_registry[_ptr_key_from_tensor(k_ptr_src)]
    v_src_layers = src_registry[_ptr_key_from_tensor(v_ptr_src)]
    src_indices = src_indices.to(dtype=torch.int64, device="cpu")
    dst_indices = dst_indices.to(dtype=torch.int64, device="cpu")
    for layer_id, (k_src, v_src) in enumerate(zip(k_src_layers, v_src_layers)):
        dst_k[dst_indices, layer_id] = k_src[src_indices]
        dst_v[dst_indices, layer_id] = v_src[src_indices]


def _cpu_jit_one_layer_mha_copy(
    *,
    k_cache_dst,
    v_cache_dst,
    k_cache_src,
    v_cache_src,
    indices_dst,
    indices_src,
    **_,
):
    indices_dst = indices_dst.to(dtype=torch.int64, device="cpu")
    indices_src = indices_src.to(dtype=torch.int64, device="cpu")
    k_cache_dst[indices_dst] = k_cache_src[indices_src]
    v_cache_dst[indices_dst] = v_cache_src[indices_src]


def _cpu_jit_one_layer_mla_copy(
    *,
    cache_dst,
    cache_src,
    indices_dst,
    indices_src,
    **_,
):
    indices_dst = indices_dst.to(dtype=torch.int64, device="cpu")
    indices_src = indices_src.to(dtype=torch.int64, device="cpu")
    cache_dst[indices_dst] = cache_src[indices_src]


def _cpu_per_layer_pf_lf_copy(
    *,
    src,
    dst,
    src_indices,
    dst_indices,
    layer_id,
    **_,
):
    src_indices = src_indices.to(dtype=torch.int64, device="cpu")
    dst_indices = dst_indices.to(dtype=torch.int64, device="cpu")
    dst[dst_indices] = src[src_indices, layer_id]


class _FakeEvent:
    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing

    def record(self):
        pass

    def wait(self, stream=None):
        pass


class _FakeDeviceModule:
    Event = _FakeEvent

    @staticmethod
    def Stream():
        return object()

    @staticmethod
    @contextmanager
    def stream(stream):
        yield


class _QueuedStream:
    def __init__(self):
        self.operations = []
        self.position = 0

    def enqueue(self, operation):
        self.operations.append(operation)

    def drain(self, stop=None):
        stop = len(self.operations) if stop is None else stop
        while self.position < stop:
            operation = self.operations[self.position]
            self.position += 1
            operation()

    def wait_stream(self, stream):
        stop = len(stream.operations)
        self.enqueue(lambda: stream.drain(stop))


class _QueuedDeviceModule:
    Stream = _QueuedStream

    def __init__(self):
        self.current = self.Stream()

    def Event(self, **_):
        device = self

        class Event:
            def record(self):
                self.stream = device.current
                self.stop = len(self.stream.operations)

            def wait(self, stream=None):
                stream = device.current if stream is None else stream
                source, stop = self.stream, self.stop
                stream.enqueue(lambda: source.drain(stop))

        return Event()

    @contextmanager
    def stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous


class TestHiCacheStagedWriteBackDispatch(CustomTestCase):
    def setUp(self):
        transfer_module._timing_events_supported.cache_clear()
        self.addCleanup(transfer_module._timing_events_supported.cache_clear)

    @staticmethod
    def _start_writing(controller):
        with mock.patch.object(transfer_module, "device_module", _FakeDeviceModule):
            controller.l2_transfer_engine = L2TransferEngine("kernel")
            controller.start_writing()

    def test_backup_precedes_later_forward_mutation(self):
        """A later forward must not overwrite device state before backup reads it."""
        for controller_type in (HiCacheController, HybridCacheController):
            with self.subTest(controller=controller_type.__name__):
                device = _QueuedDeviceModule()

                class HostPool:
                    layout = "page_first"
                    can_use_write_back_jit = True
                    size_per_token = 4

                    def __init__(self):
                        self.data = torch.full((4,), -1)

                    def backup_from_device_all_layer(
                        self, device_pool, host_indices, device_indices, io_backend
                    ):
                        device.current.enqueue(lambda: self.data.copy_(device_pool))

                pools = [HostPool(), HostPool()]
                sources = [torch.full((4,), 3), torch.full((4,), 5)]
                controller = controller_type.__new__(controller_type)
                controller.io_backend = "kernel"
                controller.mem_pool_device = sources[0]
                controller.mem_pool_host = pools[0]
                extra_pools = None
                if controller_type is HybridCacheController:
                    entries = [
                        PoolEntry(
                            name=name,
                            host_pool=pool,
                            device_pool=source,
                            layer_mapper=None,
                        )
                        for name, pool, source in zip(
                            (PoolName.KV, PoolName.MAMBA), pools, sources
                        )
                    ]
                    controller.mem_pool_host = SimpleNamespace(
                        layout="page_first",
                        can_use_write_back_jit=True,
                        anchor_entry=entries[0],
                        entry_map={entry.name: entry for entry in entries},
                    )
                    extra_pools = [
                        PoolTransfer(PoolName.MAMBA, _indices(0, 4), _indices(0, 4))
                    ]
                controller.write_queue = [
                    CacheOperation(
                        _indices(0, 4), _indices(0, 4), 1, pool_transfers=extra_pools
                    )
                ]
                controller.ack_write_queue = []
                with mock.patch.object(transfer_module, "device_module", device):
                    controller.l2_transfer_engine = L2TransferEngine("kernel")
                    controller.start_writing()

                self.assertTrue(torch.all(pools[0].data == -1))
                forward = device.Stream()
                forward.wait_stream(device.current)
                for source in sources:
                    forward.enqueue(lambda source=source: source.fill_(7))
                forward.drain()
                controller.l2_transfer_engine.device_to_host_stream.drain()

                self.assertTrue(torch.all(pools[0].data == 3))
                if extra_pools:
                    self.assertTrue(torch.all(pools[1].data == 5))
                self.assertTrue(all(torch.all(source == 7) for source in sources))

    def test_hybrid_load_forwards_merged_pool_transfers(self):
        transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=_indices(0, 2),
            device_indices=_indices(2, 4),
            keys=["page-key"],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        op = CacheOperation(_indices(0, 4), _indices(4, 8), 7)
        op.pool_transfers = [transfer]
        controller = mock.Mock(spec=HybridCacheController)
        controller.load_queue = [op, op]
        controller.layer_done_counter = mock.MagicMock()
        controller.layer_done_counter.update_producer.return_value = 0
        controller._move_op_indices.side_effect = lambda op: (
            op.host_indices,
            op.device_indices,
            op.pool_transfers,
        )
        controller.mem_pool_host = _host_group_stub([], can_use_write_back_jit=False)
        controller._l2_transfers.side_effect = lambda *args: (
            HybridCacheController._l2_transfers(controller, *args)
        )
        controller._l2_load_transfers.side_effect = lambda *args: (
            HybridCacheController._l2_load_transfers(controller, *args)
        )
        controller._num_tokens_by_pool.return_value = {}
        controller._transfer_num_bytes.return_value = 0
        controller.l2_transfer_engine = mock.Mock()
        controller.load_fence_stream = None
        controller.mla_broadcast_enabled = False
        completion = SimpleNamespace(
            start_event=object(), finish_event=object(), timing_enabled=False
        )
        controller.l2_transfer_engine.submit_host_to_device.return_value = completion
        controller.layer_num = 2
        controller.ack_load_queue = []

        self.assertEqual(HybridCacheController.start_loading(controller), 0)

        merged_op = controller._move_op_indices.call_args.args[0]
        merged_transfer = merged_op.pool_transfers[0]
        self.assertEqual(merged_transfer.host_indices.tolist(), [0, 1, 0, 1])
        self.assertEqual(merged_transfer.keys, ["page-key", "page-key"])
        self.assertEqual(merged_transfer.hit_policy, PoolHitPolicy.TRAILING_PAGES)
        controller._l2_load_transfers.assert_called_once()
        l2_transfers = (
            controller.l2_transfer_engine.submit_host_to_device.call_args.args[0]
        )
        self.assertEqual(len(l2_transfers), 2)
        self.assertEqual(l2_transfers[1].host_indices.tolist(), [0, 1, 0, 1])
        self.assertEqual(
            len(
                HybridCacheController._l2_transfers(
                    controller, _indices(0, 0), _indices(0, 0), [merged_transfer]
                )
            ),
            1,
        )
        controller._num_tokens_by_pool.assert_called_once_with(merged_op)
        self.assertEqual(controller.ack_load_queue[0].node_ids, [7, 7])

    def _short_swa_tail_pipeline(self, swa_page_size: int) -> BufferModePipeline:
        """Pipeline holding one staged span [2, 8) whose 4-slot trailing SWA
        window outruns the splice left by a device prefix of 6."""
        handle = CacheRequestHandle("r", 0)
        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        pipeline._cache = mock.Mock()
        pipeline._cache.cache_controller.mem_pool_host.entry_map = {
            PoolName.SWA: SimpleNamespace(
                host_pool=SimpleNamespace(page_size=swa_page_size)
            )
        }
        pipeline.release_staged_hold = mock.Mock(return_value=True)
        pipeline.staged_prefetches = {
            handle: SimpleNamespace(
                request=handle,
                key_tokens=array("q", range(8)),
                extra_key=None,
                cache_salt=None,
                matched_len=2,
                num_tokens=6,
                occupied_tokens=6,
                host_indices=_indices(0, 6),
                aux_xfers=[
                    PoolTransfer(
                        name=PoolName.SWA,
                        host_indices=_indices(0, 4),
                    )
                ],
                hash_values=[],
                operation_id=1,
            )
        }
        return pipeline

    def test_short_staged_swa_tail_keeps_complete_window(self):
        """FULL-prefix growth trims only FULL; SWA keeps its complete window."""
        handle = CacheRequestHandle("r", 0)
        pipeline = self._short_swa_tail_pipeline(swa_page_size=2)
        pipeline._cache.tree_core.is_eagle = False
        pipeline._cache.tree_core.match_full_device_prefix.return_value = (6, 1, 6)
        pipeline._cache.tree_core.collect_full_device_indices.return_value = _indices(
            0, 6
        )
        req = SimpleNamespace(
            rid="r",
            cache_request_handle=handle,
            prefix_indices=_indices(0, 0),
            kv=SimpleNamespace(cache_protected_len=0),
        )
        self.assertTrue(pipeline.prepare_staged_prefetch(req))
        self.assertEqual((req.host_hit_length, req.swa_host_hit_length), (2, 4))
        pipeline.release_staged_hold.assert_not_called()

        pipeline = self._short_swa_tail_pipeline(swa_page_size=4)
        pipeline._cache.tree_core.is_eagle = False
        pipeline._cache.tree_core.match_full_device_prefix.return_value = (6, 1, 6)
        pipeline._cache.tree_core.collect_full_device_indices.return_value = _indices(
            0, 6
        )
        req = SimpleNamespace(
            rid="r",
            cache_request_handle=handle,
            prefix_indices=_indices(0, 0),
            kv=SimpleNamespace(cache_protected_len=0),
        )
        self.assertTrue(pipeline.prepare_staged_prefetch(req))
        self.assertEqual((req.host_hit_length, req.swa_host_hit_length), (2, 4))
        pipeline.release_staged_hold.assert_not_called()

    def test_l2_transfer_maps_global_layers(self):
        host_pool = mock.Mock()
        transfer = L2Transfer(
            host_pool=host_pool,
            device_pool=mock.sentinel.device_pool,
            host_indices=_indices(0, 2),
            device_indices=_indices(2, 4),
            layer_mapper={1: 0, 3: 1}.get,
        )
        with mock.patch.object(transfer_module, "device_module", _FakeDeviceModule):
            L2TransferEngine("kernel").submit_host_to_device([transfer], layer_num=4)

        self.assertEqual(
            [
                call.args[3]
                for call in host_pool.load_to_device_per_layer.call_args_list
            ],
            [0, 1],
        )

    def test_packed_draft_load_is_flattened_into_l2_transfers(self):
        host_pool = mock.Mock()
        controller = HybridCacheController.__new__(HybridCacheController)
        entry = PoolEntry(
            name=PoolName.KV,
            host_pool=host_pool,
            device_pool=mock.sentinel.target_device_pool,
            layer_mapper={0: 0, 1: 1, 2: 2}.get,
            is_primary_index_anchor=True,
            packed_draft_device_pools=(mock.sentinel.draft_device_pool,),
        )
        controller.mem_pool_host = SimpleNamespace(
            anchor_entry=entry,
            entry_map={entry.name: entry},
        )
        controller.layer_num = 2

        self.assertEqual(
            len(controller._l2_transfers(_indices(0, 2), _indices(2, 4))), 1
        )
        transfers = controller._l2_load_transfers(_indices(0, 2), _indices(2, 4))

        self.assertEqual(len(transfers), 2)
        self.assertFalse(transfers[0].is_draft)
        self.assertTrue(transfers[1].is_draft)
        with mock.patch.object(transfer_module, "device_module", _FakeDeviceModule):
            L2TransferEngine("kernel").submit_host_to_device(transfers, layer_num=2)
        self.assertEqual(
            [
                call.args[3]
                for call in host_pool.load_to_device_per_layer.call_args_list
            ],
            [0, 2, 1],
        )
        self.assertIs(
            host_pool.load_to_device_per_layer.call_args_list[1].args[0],
            mock.sentinel.draft_device_pool,
        )
        self.assertTrue(
            host_pool.load_to_device_per_layer.call_args_list[1].kwargs["is_draft"]
        )

    def test_mixed_staged_write_resolves_indices_per_pool(self):
        anchor_host_pool = SimpleNamespace(can_use_write_back_jit=True)
        extra_host_pool = SimpleNamespace(can_use_write_back_jit=False)
        anchor_entry = PoolEntry(
            name=PoolName.KV,
            host_pool=anchor_host_pool,
            device_pool=None,
            layer_mapper=lambda layer_id: layer_id,
            is_primary_index_anchor=True,
        )
        extra_entry = PoolEntry(
            name=PoolName.SWA,
            host_pool=extra_host_pool,
            device_pool=None,
            layer_mapper=lambda layer_id: layer_id,
        )
        host_group = SimpleNamespace(
            layout="page_first",
            can_use_write_back_jit=False,
            supports_per_pool_backup_indices=True,
            anchor_entry=anchor_entry,
            entry_map={PoolName.KV: anchor_entry, PoolName.SWA: extra_entry},
        )
        transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=_indices(4, 6),
            device_indices=_indices(6, 8),
        )
        op = CacheOperation(
            host_indices=_indices(0, 2),
            device_indices=_indices(2, 4),
            node_id=1,
            pool_transfers=[transfer],
        )
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.io_backend = "kernel"
        controller.mem_pool_host = host_group
        controller.move_indices = mock.Mock(
            return_value=(mock.sentinel.host_indices, mock.sentinel.device_indices)
        )

        host_indices, device_indices, pool_transfers = controller._move_write_operation(
            op
        )

        self.assertIs(host_indices, op.host_indices)
        self.assertIs(device_indices, op.device_indices)
        controller.move_indices.assert_called_once_with(
            transfer.host_indices, transfer.device_indices
        )
        self.assertIs(pool_transfers[0].host_indices, mock.sentinel.host_indices)
        self.assertIs(pool_transfers[0].device_indices, mock.sentinel.device_indices)

    def _patched_transfers(self, src_registry=None, module=MEMORY_POOL_HOST_MODULE):
        staged_side_effect = None
        if src_registry is not None:
            staged_side_effect = lambda **kwargs: _cpu_staged_lf_pf_copy(
                src_registry, **kwargs
            )
        return (
            mock.patch(
                f"{module}.jit_transfer_hicache_all_layer_mla_staged_lf_pf",
                side_effect=staged_side_effect,
            ),
            mock.patch(
                f"{module}.transfer_kv_all_layer_mla_lf_pf",
                create=True,
            ),
            mock.patch(
                f"{module}.transfer_kv_per_layer_mla_pf_lf",
                side_effect=_cpu_per_layer_pf_lf_copy,
                create=True,
            ),
        )

    def test_mha_backup_then_load_roundtrip_uses_staged(self):
        layer_num = 2
        head_num = 1
        head_dim = 4
        host_indices = _indices(0, 4)
        device_indices = _indices(4, 8)
        k_layers = [
            (torch.arange(8 * head_num * head_dim, dtype=torch.uint8) + layer_id * 40)
            .reshape(8, head_num, head_dim)
            .clone()
            for layer_id in range(layer_num)
        ]
        v_layers = [
            (
                torch.arange(8 * head_num * head_dim, dtype=torch.uint8)
                + 100
                + layer_id * 40
            )
            .reshape(8, head_num, head_dim)
            .clone()
            for layer_id in range(layer_num)
        ]
        expected_k = [layer[device_indices].clone() for layer in k_layers]
        expected_v = [layer[device_indices].clone() for layer in v_layers]
        device_pool = _device_pool_stub(
            layer_num=layer_num,
            k_buffer=k_layers,
            v_buffer=v_layers,
            k_data_ptrs=torch.tensor(
                [layer.data_ptr() for layer in k_layers], dtype=torch.uint64
            ),
            v_data_ptrs=torch.tensor(
                [layer.data_ptr() for layer in v_layers], dtype=torch.uint64
            ),
        )

        host = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        host.layout = "page_first"
        host.page_size = 1
        host.layer_num = layer_num
        host.head_num = head_num
        host.head_dim = head_dim
        host.element_dim = head_num * head_dim
        host.token_stride_size = host.element_dim
        host.layout_dim = host.token_stride_size * layer_num
        host.dtype = torch.uint8
        host.can_use_jit = True
        host.can_use_write_back_jit = True
        host.kv_buffer = torch.zeros(
            2, 8, layer_num, head_num, head_dim, dtype=torch.uint8
        )
        host.k_data_refs = [host.k_buffer.transpose(0, 1)[i] for i in range(layer_num)]
        host.v_data_refs = [host.v_buffer.transpose(0, 1)[i] for i in range(layer_num)]
        host.staging_k_buffer = torch.empty(
            4, layer_num, head_num, head_dim, dtype=torch.uint8
        )
        host.staging_v_buffer = torch.empty_like(host.staging_k_buffer)
        src_registry = {
            _ptr_key_from_layers(k_layers): k_layers,
            _ptr_key_from_layers(v_layers): v_layers,
        }

        with (
            mock.patch(
                f"{MHA_POOL_HOST_MODULE}.jit_transfer_hicache_all_layer_staged_lf_pf",
                side_effect=lambda **kwargs: _cpu_staged_mha_lf_pf_copy(
                    src_registry, **kwargs
                ),
            ) as staged,
            mock.patch(
                f"{MHA_POOL_HOST_MODULE}.transfer_kv_all_layer_lf_pf",
                create=True,
            ) as fallback,
            mock.patch(
                f"{MHA_POOL_HOST_MODULE}.jit_transfer_hicache_one_layer",
                side_effect=_cpu_jit_one_layer_mha_copy,
            ) as load,
            mock.patch(
                f"{MHA_POOL_HOST_MODULE}.can_use_write_back_jit_kernel",
                return_value=True,
            ) as can_use_write_back_jit_kernel,
        ):
            host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="kernel"
            )
            for layer in k_layers + v_layers:
                layer.zero_()
            for layer_id in range(layer_num):
                host.load_to_device_per_layer(
                    device_pool,
                    host_indices,
                    device_indices,
                    layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 1)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, layer_num)
        can_use_write_back_jit_kernel.assert_not_called()
        for layer_id in range(layer_num):
            self.assertTrue(
                torch.equal(k_layers[layer_id][device_indices], expected_k[layer_id])
            )
            self.assertTrue(
                torch.equal(v_layers[layer_id][device_indices], expected_v[layer_id])
            )
            self.assertTrue(
                torch.equal(host.k_buffer[host_indices, layer_id], expected_k[layer_id])
            )
            self.assertTrue(
                torch.equal(host.v_buffer[host_indices, layer_id], expected_v[layer_id])
            )

    def test_mla_backup_then_load_roundtrip_uses_staged(self):
        layer_num = 2
        kv_cache_dim = 5
        host_indices = _indices(0, 4)
        device_indices = _indices(4, 8)
        device_layers = [
            (torch.arange(8 * kv_cache_dim, dtype=torch.uint8) + layer_id * 50)
            .reshape(8, 1, kv_cache_dim)
            .clone()
            for layer_id in range(layer_num)
        ]
        expected = [layer[device_indices].clone() for layer in device_layers]
        device_pool = _device_pool_stub(
            layer_num=layer_num,
            kv_buffer=device_layers,
            data_ptrs=torch.tensor(
                [layer.data_ptr() for layer in device_layers], dtype=torch.uint64
            ),
        )

        host = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        host.device_pool = device_pool
        host.layout = "page_first"
        host.page_size = 1
        host.layer_num = layer_num
        host.kv_cache_dim = kv_cache_dim
        host.token_stride_size = kv_cache_dim
        host.layout_dim = host.token_stride_size * layer_num
        host.dtype = torch.uint8
        host.can_use_jit = True
        host.can_use_write_back_jit = True
        host.kv_buffer = torch.zeros(8, layer_num, 1, kv_cache_dim, dtype=torch.uint8)
        host.data_refs = [host.kv_buffer.transpose(0, 1)[i] for i in range(layer_num)]
        host.staging_buffer = torch.empty(
            4, layer_num, 1, kv_cache_dim, dtype=torch.uint8
        )
        src_registry = {_ptr_key_from_layers(device_layers): device_layers}

        staged_patch, fallback_patch, _ = self._patched_transfers(
            src_registry, module=MLA_POOL_HOST_MODULE
        )
        with (
            staged_patch as staged,
            fallback_patch as fallback,
            mock.patch(
                f"{MLA_POOL_HOST_MODULE}.jit_transfer_hicache_one_layer_mla",
                side_effect=_cpu_jit_one_layer_mla_copy,
            ) as load,
            mock.patch(
                f"{MLA_POOL_HOST_MODULE}.can_use_write_back_jit_kernel",
                return_value=True,
            ) as can_use_write_back_jit_kernel,
        ):
            host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="kernel"
            )
            for layer in device_layers:
                layer.zero_()
            for layer_id in range(layer_num):
                host.load_to_device_per_layer(
                    device_pool,
                    host_indices,
                    device_indices,
                    layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 1)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, layer_num)
        can_use_write_back_jit_kernel.assert_not_called()
        for layer_id, layer in enumerate(device_layers):
            self.assertTrue(torch.equal(layer[device_indices], expected[layer_id]))
            self.assertTrue(
                torch.equal(host.kv_buffer[host_indices, layer_id], expected[layer_id])
            )

    @unittest.skip(
        "TODO: Mamba pool is currently incompatible with write-back staging "
        "kernel; re-enable once the staging bug is fixed."
    )
    def test_mamba_backup_then_load_roundtrip_uses_staged(self):
        num_layers = 2
        host_indices = _indices(0, 4)
        device_indices = _indices(4, 8)
        temporal = torch.arange(num_layers * 8 * 3, dtype=torch.uint8).reshape(
            num_layers, 8, 1, 3
        )
        conv = (torch.arange(num_layers * 8 * 2, dtype=torch.uint8) + 97).reshape(
            num_layers, 8, 1, 2
        )
        device_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(temporal=temporal.clone(), conv=[conv.clone()])
        )
        expected_temporal = device_pool.mamba_cache.temporal[:, device_indices].clone()
        expected_conv = device_pool.mamba_cache.conv[0][:, device_indices].clone()

        host = MambaPoolHost.__new__(MambaPoolHost)
        host.layout = "page_first"
        host.num_mamba_layers = num_layers
        host.device_pool = SimpleNamespace(device="cpu")
        host.temporal_buffer = torch.zeros(8, num_layers, 1, 3, dtype=torch.uint8)
        host.conv_buffer = [
            torch.zeros(8, num_layers, 1, 2, dtype=torch.uint8),
        ]
        host.conv_state_shapes = [(2,)]
        host.temporal_staging_buffer = torch.empty(
            4, num_layers, 1, 3, dtype=torch.uint8
        )
        host.conv_staging_buffers = [
            torch.empty(4, num_layers, 1, 2, dtype=torch.uint8),
        ]
        host._temporal_can_use_jit = True
        host._conv_can_use_jit = [True]
        host.can_use_write_back_jit = True
        host.temporal_device_ptrs = torch.tensor(
            [layer.data_ptr() for layer in device_pool.mamba_cache.temporal],
            dtype=torch.uint64,
        )
        host.conv_device_ptrs = [
            torch.tensor(
                [layer.data_ptr() for layer in device_pool.mamba_cache.conv[0]],
                dtype=torch.uint64,
            )
        ]

        src_registry = {
            _ptr_key_from_layers(device_pool.mamba_cache.temporal): list(
                device_pool.mamba_cache.temporal
            ),
            _ptr_key_from_layers(device_pool.mamba_cache.conv[0]): list(
                device_pool.mamba_cache.conv[0]
            ),
        }

        staged_patch, fallback_patch, load_patch = self._patched_transfers(src_registry)
        with staged_patch as staged, fallback_patch as fallback, load_patch as load:
            host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="kernel"
            )
            device_pool.mamba_cache.temporal.zero_()
            device_pool.mamba_cache.conv[0].zero_()
            for layer_id in range(num_layers):
                host.load_to_device_per_layer(
                    device_pool,
                    host_indices,
                    device_indices,
                    layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 2)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, 4)
        self.assertTrue(
            torch.equal(
                device_pool.mamba_cache.temporal[:, device_indices], expected_temporal
            )
        )
        self.assertTrue(
            torch.equal(
                device_pool.mamba_cache.conv[0][:, device_indices], expected_conv
            )
        )

    def test_deepseek_v4_paged_pool_backup_then_load_roundtrip_uses_staged(self):
        layer_num = 2
        slot_page_size = 2
        host_indices = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
        device_indices = torch.tensor([2, 3, 6, 7], dtype=torch.int64)
        host_rows = torch.tensor([0, 2], dtype=torch.int64)
        device_rows = torch.tensor([1, 3], dtype=torch.int64)
        device_buffers = [
            (torch.arange(5 * 4, dtype=torch.uint8) + layer_id * 50).reshape(5, 4)
            for layer_id in range(layer_num)
        ]
        expected = [buffer[device_rows].clone() for buffer in device_buffers]

        host = DeepSeekV4PagedHostPool.__new__(DeepSeekV4PagedHostPool)
        host.pool_name = "c4"
        host.layout = "page_first"
        host.slot_page_size = slot_page_size
        host.layer_num = layer_num
        host.item_bytes = 4
        host.dtype = torch.uint8
        host.device_buffers = device_buffers
        host.device_ptrs = torch.tensor(
            [buffer.data_ptr() for buffer in device_buffers], dtype=torch.uint64
        )
        host.kv_buffer = torch.zeros(
            4, host.layer_num, host.item_bytes, dtype=torch.uint8
        )
        host.staging_buffer = torch.empty(
            4, host.layer_num, host.item_bytes, dtype=torch.uint8
        )
        host.can_use_jit = False
        host.can_use_write_back_jit = True
        src_registry = {_ptr_key_from_layers(device_buffers): device_buffers}

        staged_patch, fallback_patch, load_patch = self._patched_transfers(src_registry)
        with staged_patch as staged, fallback_patch as fallback, load_patch as load:
            host.backup_from_device_all_layer(
                device_pool=None,
                host_indices=host_indices,
                device_indices=device_indices,
                io_backend="kernel",
            )
            for buffer in device_buffers:
                buffer.zero_()
            for layer_id in range(layer_num):
                host.load_to_device_per_layer(
                    device_pool=None,
                    host_indices=host_indices,
                    device_indices=device_indices,
                    layer_id=layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 1)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, layer_num)
        for layer_id, buffer in enumerate(device_buffers):
            self.assertTrue(torch.equal(buffer[device_rows], expected[layer_id]))
            self.assertTrue(
                torch.equal(host.kv_buffer[host_rows, layer_id], expected[layer_id])
            )

    def test_deepseek_v4_state_pool_backup_then_load_roundtrip_uses_staged(self):
        layer_num = 2
        swa_page_size = 2
        host_indices = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
        device_indices = torch.tensor([2, 3, 6, 7], dtype=torch.int64)
        host_rows = torch.tensor([0, 2], dtype=torch.int64)
        device_rows = torch.tensor([1, 3], dtype=torch.int64)
        device_page_views = [
            (torch.arange(5 * 5, dtype=torch.uint8) + layer_id * 60).reshape(5, 5)
            for layer_id in range(layer_num)
        ]
        expected = [buffer[device_rows].clone() for buffer in device_page_views]

        host = DeepSeekV4StateHostPool.__new__(DeepSeekV4StateHostPool)
        host.pool_name = "c4_state"
        host.layout = "page_first"
        host.swa_page_size = swa_page_size
        host.layer_num = layer_num
        host.state_page_bytes = 5
        host.dtype = torch.uint8
        host.device_page_views = device_page_views
        host.device_ptrs = torch.tensor(
            [buffer.data_ptr() for buffer in device_page_views], dtype=torch.uint64
        )
        host.kv_buffer = torch.zeros(
            4, host.layer_num, host.state_page_bytes, dtype=torch.uint8
        )
        host.staging_buffer = torch.empty(
            4, host.layer_num, host.state_page_bytes, dtype=torch.uint8
        )
        host.can_use_jit = False
        host.can_use_write_back_jit = True
        src_registry = {_ptr_key_from_layers(device_page_views): device_page_views}

        staged_patch, fallback_patch, load_patch = self._patched_transfers(src_registry)
        with staged_patch as staged, fallback_patch as fallback, load_patch as load:
            host.backup_from_device_all_layer(
                device_pool=None,
                host_indices=host_indices,
                device_indices=device_indices,
                io_backend="kernel",
            )
            for buffer in device_page_views:
                buffer.zero_()
            for layer_id in range(layer_num):
                host.load_to_device_per_layer(
                    device_pool=None,
                    host_indices=host_indices,
                    device_indices=device_indices,
                    layer_id=layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 1)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, layer_num)
        for layer_id, buffer in enumerate(device_page_views):
            self.assertTrue(torch.equal(buffer[device_rows], expected[layer_id]))
            self.assertTrue(
                torch.equal(host.kv_buffer[host_rows, layer_id], expected[layer_id])
            )

    def test_dsa_indexer_backup_then_load_roundtrip_uses_staged(self):
        layer_num = 2
        page_size = 2
        host_indices = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
        device_indices = torch.tensor([2, 3, 6, 7], dtype=torch.int64)
        host_page_indices = torch.tensor([0, 2], dtype=torch.int64)
        device_page_indices = torch.tensor([1, 3], dtype=torch.int64)
        indexer_page_stride_size = 8
        device_layers = [
            (
                torch.arange(5 * indexer_page_stride_size, dtype=torch.uint8)
                + layer_id * 70
            ).reshape(5, 1, indexer_page_stride_size)
            for layer_id in range(layer_num)
        ]
        expected = [buffer[device_page_indices].clone() for buffer in device_layers]
        device_pool = _device_pool_stub(
            layer_num=layer_num,
            index_k_with_scale_buffer=device_layers,
        )

        host = DSAIndexerPoolHost.__new__(DSAIndexerPoolHost)
        host.device_pool = device_pool
        host.layout = "page_first"
        host.page_size = page_size
        host.layer_num = layer_num
        host.indexer_page_stride_size = indexer_page_stride_size
        host.indexer_layout_dim = host.layer_num * host.indexer_page_stride_size
        host.index_k_device_ptrs = torch.tensor(
            [buffer.data_ptr() for buffer in device_layers], dtype=torch.uint64
        )
        host.index_k_with_scale_buffer = torch.zeros(
            4, host.layer_num, 1, host.indexer_page_stride_size, dtype=torch.uint8
        )
        host.staging_buffer = torch.empty(
            4, host.layer_num, 1, host.indexer_page_stride_size, dtype=torch.uint8
        )
        host.can_use_jit = False
        host.can_use_write_back_jit = True
        src_registry = {_ptr_key_from_layers(device_layers): device_layers}

        staged_patch, fallback_patch, load_patch = self._patched_transfers(
            src_registry, module=DSA_POOL_HOST_MODULE
        )
        with staged_patch as staged, fallback_patch as fallback, load_patch as load:
            host.backup_from_device_all_layer(
                device_pool=device_pool,
                host_indices=host_indices,
                device_indices=device_indices,
                io_backend="kernel",
            )
            for buffer in device_layers:
                buffer.zero_()
            for layer_id in range(layer_num):
                host.load_to_device_per_layer(
                    device_pool=device_pool,
                    host_indices=host_indices,
                    device_indices=device_indices,
                    layer_id=layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 1)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, layer_num)
        for layer_id, buffer in enumerate(device_layers):
            self.assertTrue(
                torch.equal(buffer[device_page_indices], expected[layer_id])
            )
            self.assertTrue(
                torch.equal(
                    host.index_k_with_scale_buffer[host_page_indices, layer_id],
                    expected[layer_id],
                )
            )

    def test_logical_host_pool_preserves_page_first_group_layout(self):
        logical_host_pool = LogicalHostPool(8, 2, layout="page_first")
        group = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=logical_host_pool,
                    device_pool=None,
                    layer_mapper=lambda _: 0,
                    is_primary_index_anchor=True,
                )
            ]
        )

        self.assertEqual(group.layout, "page_first")
        self.assertTrue(group.can_use_write_back_jit)

    def test_host_pool_group_destroys_logical_anchor(self):
        logical_host_pool = LogicalHostPool(8, 2, layout="page_first")
        group = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=logical_host_pool,
                    device_pool=None,
                    layer_mapper=lambda _: 0,
                    is_primary_index_anchor=True,
                )
            ]
        )

        self.assertIsNone(group.destroy())

    def test_write_back_jit_hybrid_write_keeps_extra_host_indices_on_cpu(self):
        captured = []

        controller = HybridCacheController.__new__(HybridCacheController)
        controller.write_queue = [
            CacheOperation(
                host_indices=_indices(0, 4),
                device_indices=_indices(4, 8),
                node_id=1,
                pool_transfers=[
                    PoolTransfer(
                        name=PoolName.DEEPSEEK_V4_C4,
                        host_indices=_indices(0, 4),
                        device_indices=_indices(4, 8),
                    )
                ],
            )
        ]
        controller.io_backend = "kernel"
        controller.mem_pool_host = _host_group_stub(
            captured, can_use_write_back_jit=True
        )
        controller.mem_pool_device = None
        controller.ack_write_queue = []
        controller.move_hybrid_indices = mock.Mock(
            side_effect=AssertionError(
                "write-back JIT kernel write should not move indices"
            )
        )

        self._start_writing(controller)

        controller.move_hybrid_indices.assert_not_called()
        self.assertEqual([indices.device.type for indices in captured], ["cpu", "cpu"])

    def test_hybrid_write_moves_indices_without_write_back_jit(self):
        captured = []

        op = CacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
            pool_transfers=[
                PoolTransfer(
                    name=PoolName.DEEPSEEK_V4_C4,
                    host_indices=_indices(0, 4),
                    device_indices=_indices(4, 8),
                )
            ],
        )
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.write_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host = _host_group_stub(
            captured, can_use_write_back_jit=False
        )
        controller.mem_pool_device = None
        controller.ack_write_queue = []
        controller.move_hybrid_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, op.pool_transfers)
        )

        self._start_writing(controller)

        controller.move_hybrid_indices.assert_called_once()
        self.assertEqual([indices.device.type for indices in captured], ["cpu", "cpu"])

    def test_write_back_jit_cache_controller_keeps_host_indices_on_cpu(self):
        captured = {}

        class FakeHostPool:
            layout = "page_first"
            can_use_write_back_jit = True
            size_per_token = 2

            def backup_from_device_all_layer(
                self, device_pool, host_indices, device_indices, io_backend
            ):
                captured["host_indices"] = host_indices

        controller = HiCacheController.__new__(HiCacheController)
        controller.write_queue = [
            CacheOperation(
                host_indices=_indices(0, 4),
                device_indices=_indices(4, 8),
                node_id=1,
            )
        ]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeHostPool()
        controller.mem_pool_device = None
        controller.device = "cuda"
        controller.ack_write_queue = []
        controller.move_indices = mock.Mock(
            side_effect=AssertionError(
                "write-back JIT kernel write should not move indices"
            )
        )

        self._start_writing(controller)

        controller.move_indices.assert_not_called()
        self.assertEqual(captured["host_indices"].device.type, "cpu")

    def test_cache_controller_moves_indices_without_write_back_jit(self):
        captured = {}

        class FakeHostPool:
            layout = "page_first"
            can_use_write_back_jit = False
            size_per_token = 2

            def backup_from_device_all_layer(
                self, device_pool, host_indices, device_indices, io_backend
            ):
                captured["host_indices"] = host_indices

        op = CacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.write_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeHostPool()
        controller.mem_pool_device = None
        controller.device = "cuda"
        controller.ack_write_queue = []
        controller.move_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices)
        )

        self._start_writing(controller)

        controller.move_indices.assert_called_once()
        self.assertEqual(captured["host_indices"].device.type, "cpu")


class _FakeProducerEvent:
    def __init__(self, operations=None):
        self.start_event = _FakeEvent()
        self.finish_event = _FakeEvent()
        self._operations = operations
        self.completed_layers = []

    def complete(self, layer_index):
        self.completed_layers.append(layer_index)
        if self._operations is not None:
            self._operations.append(("complete", layer_index))


def _dedup_broadcaster_stub(operations=None, *, is_src):
    return SimpleNamespace(
        is_src=is_src,
        prepare_broadcast=lambda device_indices, stream: (device_indices, None),
        broadcast_loaded_layer=lambda layer_id, prepared, trace=None: (
            operations.append(("broadcast", layer_id))
            if operations is not None
            else None
        ),
    )


class TestMLAHostDedupDispatch(CustomTestCase):
    """Dedup wiring on the destination's L2-transfer-engine controllers."""

    def setUp(self):
        transfer_module._timing_events_supported.cache_clear()
        self.addCleanup(transfer_module._timing_events_supported.cache_clear)

    @staticmethod
    def _run_start_loading(controller):
        with (
            mock.patch.object(
                manager_cache_controller, "device_module", _FakeDeviceModule
            ),
            mock.patch.object(transfer_module, "device_module", _FakeDeviceModule),
        ):
            controller.l2_transfer_engine = L2TransferEngine("kernel")
            return controller.start_loading()

    def test_mla_dedup_dummy_host_pools_are_allocator_only(self):
        mla_device_pool = _device_pool_stub(
            layer_num=2,
            store_dtype=torch.float16,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            size=8,
            start_layer=0,
            end_layer=2,
        )
        mla_host = MLATokenToKVPoolHost(
            mla_device_pool,
            host_to_device_ratio=2,
            host_size=0,
            page_size=2,
            layout="page_first",
            pin_memory=False,
            is_dummy=True,
        )

        self.assertTrue(mla_host._is_dummy)
        self.assertIsNone(mla_host.kv_buffer)
        self.assertIsNone(mla_host.data_ptrs)
        self.assertEqual(mla_host.get_contiguous_buf_infos(), ([], [], []))
        slots = mla_host.alloc(2)
        self.assertIsNotNone(slots)
        self.assertEqual(slots.tolist(), [0, 1])
        with self.assertRaisesRegex(AssertionError, "load on a dummy"):
            mla_host.load_to_device_per_layer(
                mla_device_pool, slots, slots, layer_id=0, io_backend="kernel"
            )

        dsa_device_pool = _device_pool_stub(
            layer_num=2,
            store_dtype=torch.float16,
            size=8,
            start_layer=0,
            end_layer=2,
            index_head_dim=8,
            quant_block_size=4,
        )
        indexer_host = DSAIndexerPoolHost(
            dsa_device_pool,
            mla_host,
            layout="page_first",
            pin_memory=False,
            is_dummy=True,
        )

        self.assertTrue(indexer_host._is_dummy)
        self.assertIsNone(indexer_host.index_k_with_scale_buffer)
        self.assertIsNone(indexer_host.index_k_device_ptrs)
        self.assertEqual(indexer_host.size, mla_host.size)
        self.assertEqual(indexer_host.dcp_size, mla_host.dcp_size)
        self.assertEqual(indexer_host.logical_size, mla_host.logical_size)
        with self.assertRaisesRegex(AssertionError, "load on a dummy"):
            indexer_host.load_to_device_per_layer(
                dsa_device_pool, slots, slots, layer_id=0, io_backend="kernel"
            )

    def test_mla_dedup_peer_skips_target_host_io(self):
        writes = []

        class FakeTargetHostPool:
            _is_dummy = True
            layout = "page_first"
            can_use_write_back_jit = False
            size_per_token = 2

            def backup_from_device_all_layer(self, *args):
                writes.append(args)

        op = CacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.write_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeTargetHostPool()
        controller.mem_pool_device = object()
        controller.mla_broadcaster = SimpleNamespace(is_src=False)
        controller.ack_write_queue = []
        controller.move_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices)
        )

        with mock.patch.object(transfer_module, "device_module", _FakeDeviceModule):
            controller.l2_transfer_engine = L2TransferEngine("kernel")
            controller.start_writing()

        self.assertEqual(writes, [])
        self.assertEqual(len(controller.ack_write_queue), 1)

    def _hybrid_dedup_controller(
        self,
        *,
        entries,
        group_attrs=(),
        broadcaster,
        op,
        layer_num=2,
    ):
        group = SimpleNamespace(
            layout="page_first",
            can_use_write_back_jit=False,
            supports_per_pool_backup_indices=False,
            anchor_entry=entries[0],
            entry_map={entry.name: entry for entry in entries},
            **dict(group_attrs),
        )
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.io_backend = "kernel"
        controller.mem_pool_host = group
        controller.mem_pool_device = object()
        controller.layer_num = layer_num
        controller.mla_broadcaster = broadcaster
        controller.ack_write_queue = []
        controller.ack_load_queue = []
        controller.load_queue = [op]
        controller.write_queue = [op]
        controller.load_fence_stream = None
        controller._mla_trace_pending = []
        controller._mla_trace_issued = 0
        controller.move_hybrid_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, op.pool_transfers)
        )
        return controller

    def test_hybrid_mla_dedup_peer_still_writes_rank_local_mamba(self):
        writes = []
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=_indices(0, 2),
            device_indices=_indices(2, 4),
        )
        op = CacheOperation(
            host_indices=_indices(0, 2),
            device_indices=_indices(2, 4),
            node_id=1,
            pool_transfers=[transfer],
        )

        class FakeAnchorHostPool:
            _is_dummy = True
            size_per_token = 2

            def backup_from_device_all_layer(self, *args, **kwargs):
                raise AssertionError("peer must not write target MLA")

        class FakeMambaHostPool:
            can_use_write_back_jit = False
            size_per_token = 2

            def backup_from_device_all_layer(
                self, device_pool, host_indices, device_indices, io_backend
            ):
                writes.append((host_indices, device_indices))

        entries = [
            PoolEntry(
                name=PoolName.KV,
                host_pool=FakeAnchorHostPool(),
                device_pool=object(),
                layer_mapper=lambda layer_id: layer_id,
                is_primary_index_anchor=True,
            ),
            PoolEntry(
                name=PoolName.MAMBA,
                host_pool=FakeMambaHostPool(),
                device_pool=object(),
                layer_mapper=lambda layer_id: layer_id,
            ),
        ]
        controller = self._hybrid_dedup_controller(
            entries=entries,
            broadcaster=SimpleNamespace(is_src=False),
            op=op,
        )

        with mock.patch.object(transfer_module, "device_module", _FakeDeviceModule):
            controller.l2_transfer_engine = L2TransferEngine("kernel")
            controller.start_writing()

        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][0].tolist(), transfer.host_indices.tolist())
        self.assertEqual(len(controller.ack_write_queue), 1)

    def test_hybrid_mla_dedup_peer_still_writes_local_draft_pool(self):
        target_writes = []
        draft_writes = []
        transfer = PoolTransfer(
            name=PoolName.DRAFT,
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            indices_from_pool=PoolName.KV,
        )
        op = CacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
            pool_transfers=[transfer],
        )

        class FakeTargetHostPool:
            _is_dummy = True
            size_per_token = 2

            def backup_from_device_all_layer(self, *args, **kwargs):
                target_writes.append(args)

        class FakeDraftHostPool:
            can_use_write_back_jit = False
            size_per_token = 2

            def backup_from_device_all_layer(
                self, device_pool, host_indices, device_indices, io_backend
            ):
                draft_writes.append((host_indices, device_indices))

        entries = [
            PoolEntry(
                name=PoolName.KV,
                host_pool=FakeTargetHostPool(),
                device_pool=object(),
                layer_mapper=lambda layer_id: layer_id,
                is_primary_index_anchor=True,
            ),
            PoolEntry(
                name=PoolName.DRAFT,
                host_pool=FakeDraftHostPool(),
                device_pool=object(),
                layer_mapper=lambda layer_id: layer_id,
            ),
        ]
        controller = self._hybrid_dedup_controller(
            entries=entries,
            broadcaster=SimpleNamespace(is_src=False),
            op=op,
        )

        with mock.patch.object(transfer_module, "device_module", _FakeDeviceModule):
            controller.l2_transfer_engine = L2TransferEngine("kernel")
            controller.start_writing()

        # The deduplicated target moves no data on a peer; the rank-local
        # mirrored draft pool is still written on every rank.
        self.assertEqual(target_writes, [])
        self.assertEqual(len(draft_writes), 1)

    def test_mla_dedup_source_load_and_broadcast_are_layerwise(self):
        operations = []

        class FakeTargetHostPool:
            size_per_token = 2

            def load_to_device_per_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
            ):
                operations.append(("target", layer_id))

        op = CacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        broadcaster = _dedup_broadcaster_stub(operations, is_src=True)
        broadcaster.prepare_broadcast = lambda device_indices, stream: (
            operations.append(("prepare", None)) or (device_indices, None)
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.load_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeTargetHostPool()
        controller.mem_pool_device = object()
        controller.layer_num = 3
        controller.layer_done_counter = SimpleNamespace(
            update_producer=lambda: 0, events=[_FakeProducerEvent(operations)]
        )
        controller.mla_broadcaster = broadcaster
        controller.load_fence_stream = None
        controller.ack_load_queue = []
        controller._mla_trace_pending = []
        controller._mla_trace_issued = 0
        controller.move_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices)
        )

        producer_id = self._run_start_loading(controller)

        self.assertEqual(producer_id, 0)
        self.assertEqual(
            operations,
            [
                ("prepare", None),
                ("target", 0),
                ("broadcast", 0),
                ("complete", 0),
                ("target", 1),
                ("broadcast", 1),
                ("complete", 1),
                ("target", 2),
                ("broadcast", 2),
                ("complete", 2),
            ],
        )
        self.assertEqual(len(controller.ack_load_queue), 1)
        self.assertEqual(controller.ack_load_queue[0].num_tokens, 4)

    def test_mla_dedup_load_restores_draft_on_every_rank(self):
        operations = []
        transfer = PoolTransfer(
            name=PoolName.DRAFT,
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            indices_from_pool=PoolName.KV,
        )
        op = CacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
            pool_transfers=[transfer],
        )

        class FakeTargetHostPool:
            _is_dummy = True
            size_per_token = 2

            def load_to_device_per_layer(self, *args, **kwargs):
                raise AssertionError("peer must not H2D the dedup target")

        class FakeDraftHostPool:
            layer_num = 2
            size_per_token = 2

            def load_to_device_per_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
                is_draft=False,
            ):
                operations.append(("draft", layer_id))

        entries = [
            PoolEntry(
                name=PoolName.KV,
                host_pool=FakeTargetHostPool(),
                device_pool=object(),
                layer_mapper=lambda layer_id: layer_id,
                is_primary_index_anchor=True,
            ),
            PoolEntry(
                name=PoolName.DRAFT,
                host_pool=FakeDraftHostPool(),
                device_pool=object(),
                layer_mapper=lambda layer_id: layer_id if layer_id < 2 else None,
            ),
        ]
        producer_event = _FakeProducerEvent(operations)
        controller = self._hybrid_dedup_controller(
            entries=entries,
            broadcaster=_dedup_broadcaster_stub(operations, is_src=False),
            op=op,
            layer_num=3,
        )
        controller.layer_done_counter = SimpleNamespace(
            update_producer=lambda: 0, events=[producer_event]
        )

        producer_id = self._run_start_loading(controller)

        self.assertEqual(producer_id, 0)
        self.assertEqual(
            operations,
            [
                ("draft", 0),
                ("broadcast", 0),
                ("complete", 0),
                ("draft", 1),
                ("broadcast", 1),
                ("complete", 1),
                ("broadcast", 2),
                ("complete", 2),
            ],
        )
        self.assertEqual(len(controller.ack_load_queue), 1)
        ack = controller.ack_load_queue[0]
        self.assertEqual(ack.num_tokens, 4)
        self.assertIsNot(ack.start_event, producer_event.start_event)
        self.assertIsNot(ack.finish_event, producer_event.finish_event)

    def test_hybrid_mla_dedup_loads_extra_pools_layerwise(self):
        operations = []
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=_indices(0, 2),
            device_indices=_indices(2, 4),
        )
        op = CacheOperation(
            host_indices=_indices(0, 2),
            device_indices=_indices(2, 4),
            node_id=1,
            pool_transfers=[transfer],
        )

        class FakeTargetHostPool:
            size_per_token = 2

            def load_to_device_per_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
            ):
                operations.append(("target", layer_id))

        class FakeMambaHostPool:
            size_per_token = 2

            def load_to_device_per_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
                is_draft=False,
            ):
                operations.append(("mamba", layer_id))

        entries = [
            PoolEntry(
                name=PoolName.KV,
                host_pool=FakeTargetHostPool(),
                device_pool=object(),
                layer_mapper=lambda layer_id: layer_id,
                is_primary_index_anchor=True,
            ),
            PoolEntry(
                name=PoolName.MAMBA,
                host_pool=FakeMambaHostPool(),
                device_pool=object(),
                layer_mapper=lambda layer_id: layer_id,
            ),
        ]
        controller = self._hybrid_dedup_controller(
            entries=entries,
            broadcaster=_dedup_broadcaster_stub(operations, is_src=True),
            op=op,
            layer_num=2,
        )
        controller.layer_done_counter = SimpleNamespace(
            update_producer=lambda: 0, events=[_FakeProducerEvent(operations)]
        )

        self._run_start_loading(controller)

        self.assertEqual(
            operations,
            [
                ("target", 0),
                ("mamba", 0),
                ("broadcast", 0),
                ("complete", 0),
                ("target", 1),
                ("mamba", 1),
                ("broadcast", 1),
                ("complete", 1),
            ],
        )

    def test_hybrid_mla_dedup_peer_broadcasts_only_target_and_loads_mamba(self):
        operations = []
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=_indices(0, 2),
            device_indices=_indices(2, 4),
        )
        op = CacheOperation(
            host_indices=_indices(0, 2),
            device_indices=_indices(2, 4),
            node_id=1,
            pool_transfers=[transfer],
        )

        class FakeTargetHostPool:
            _is_dummy = True
            size_per_token = 2

            def load_to_device_per_layer(self, *args, **kwargs):
                raise AssertionError("peer must not H2D the dedup target")

        class FakeMambaHostPool:
            size_per_token = 2

            def load_to_device_per_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
                is_draft=False,
            ):
                operations.append(("mamba", layer_id))

        entries = [
            PoolEntry(
                name=PoolName.KV,
                host_pool=FakeTargetHostPool(),
                device_pool=object(),
                # Global transfer ids {1, 3} are target MLA layers.
                layer_mapper=lambda layer_id: {1: 0, 3: 1}.get(layer_id),
                is_primary_index_anchor=True,
            ),
            PoolEntry(
                name=PoolName.MAMBA,
                host_pool=FakeMambaHostPool(),
                device_pool=object(),
                # Mamba state lands on the interleaved transfer layers.
                layer_mapper=lambda layer_id: layer_id if layer_id in (0, 2) else None,
            ),
        ]
        controller = self._hybrid_dedup_controller(
            entries=entries,
            broadcaster=_dedup_broadcaster_stub(operations, is_src=False),
            op=op,
            layer_num=4,
        )
        controller.layer_done_counter = SimpleNamespace(
            update_producer=lambda: 0, events=[_FakeProducerEvent(operations)]
        )

        self._run_start_loading(controller)

        # Only the target MLA layers broadcast; the interleaved Mamba layers
        # load rank-locally on every rank, including dedup peers.
        self.assertEqual(
            operations,
            [
                ("mamba", 0),
                ("complete", 0),
                ("broadcast", 0),
                ("complete", 1),
                ("mamba", 2),
                ("complete", 2),
                ("broadcast", 1),
                ("complete", 3),
            ],
        )

    def test_mla_dedup_zero_kv_load_op_runs_sidecar_layerwise(self):
        """A mamba-only (zero-KV) dedup load op runs to completion.

        The anchor L2 transfer is dropped for the empty KV range, the mamba
        sidecar still loads rank-locally on every layer, the per-layer
        broadcast and completion events still fire so every rank stays
        collective-aligned, and the ack reports zero KV tokens.
        """
        operations = []
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=_indices(0, 2),
            device_indices=_indices(2, 4),
        )
        op = CacheOperation(
            host_indices=torch.empty((0,), dtype=torch.int64),
            device_indices=torch.empty((0,), dtype=torch.int64),
            node_id=23,
            pool_transfers=[transfer],
        )

        class FakeTargetHostPool:
            size_per_token = 2

            def load_to_device_per_layer(self, *args, **kwargs):
                raise AssertionError("zero-KV op must not H2D the target pool")

        class FakeMambaHostPool:
            size_per_token = 2

            def load_to_device_per_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
                is_draft=False,
            ):
                operations.append(("mamba", layer_id))

        entries = [
            PoolEntry(
                name=PoolName.KV,
                host_pool=FakeTargetHostPool(),
                device_pool=object(),
                layer_mapper=lambda layer_id: layer_id,
                is_primary_index_anchor=True,
            ),
            PoolEntry(
                name=PoolName.MAMBA,
                host_pool=FakeMambaHostPool(),
                device_pool=object(),
                layer_mapper=lambda layer_id: layer_id,
            ),
        ]
        controller = self._hybrid_dedup_controller(
            entries=entries,
            broadcaster=_dedup_broadcaster_stub(operations, is_src=True),
            op=op,
            layer_num=2,
        )
        controller.layer_done_counter = SimpleNamespace(
            update_producer=lambda: 0, events=[_FakeProducerEvent(operations)]
        )

        self._run_start_loading(controller)

        self.assertEqual(
            operations,
            [
                ("mamba", 0),
                ("broadcast", 0),
                ("complete", 0),
                ("mamba", 1),
                ("broadcast", 1),
                ("complete", 1),
            ],
        )
        self.assertEqual(len(controller.ack_load_queue), 1)
        ack = controller.ack_load_queue[0]
        self.assertEqual(ack.node_ids, [23])
        self.assertEqual(ack.num_tokens, 0)

    def test_mla_dedup_aggregate_host_budget_is_fail_closed(self):
        budget = mock.Mock(get=mock.Mock(return_value=1))
        with mock.patch(
            "sglang.srt.environ.envs.SGLANG_HICACHE_HOST_BUDGET_GIB", budget
        ):
            with self.assertRaisesRegex(ValueError, "requires 1.12 GiB"):
                enforce_hicache_host_budget(
                    target_bytes=600_000_000,
                    rank_local_bytes={"mamba": 300_000_000},
                    tp_size=2,
                    context="unit-test",
                )

        budget.get.return_value = 2
        with mock.patch(
            "sglang.srt.environ.envs.SGLANG_HICACHE_HOST_BUDGET_GIB", budget
        ):
            self.assertEqual(
                enforce_hicache_host_budget(
                    target_bytes=600_000_000,
                    rank_local_bytes={"mamba": 300_000_000},
                    tp_size=2,
                    context="unit-test",
                ),
                1_200_000_000,
            )

    def test_hybrid_mla_dedup_preflight_includes_draft_before_host_alloc(self):
        params = SimpleNamespace(
            page_size=64,
            hicache_draft_kv_pool=mock.sentinel.draft_pool,
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
            mtp_draft_device_pools=(),
            req_to_token_pool=SimpleNamespace(mamba_allocator=object()),
        )
        memory = SimpleNamespace(
            hicache_ratio=3.0,
            hicache_size=230,
        )

        with (
            mock.patch.object(hybrid_pool_assembler, "get_memory", return_value=memory),
            mock.patch.object(
                hybrid_pool_assembler,
                "estimate_mla_host_pool_bytes",
                return_value=(100, 10),
            ),
            mock.patch.object(
                hybrid_pool_assembler,
                "estimate_mamba_host_pool_bytes",
                return_value=(20, 2),
            ),
            mock.patch.object(
                hybrid_pool_assembler,
                "estimate_draft_host_pool_bytes",
                return_value=(30, 10),
            ),
            mock.patch.object(
                hybrid_pool_assembler,
                "mla_dedup_rank_and_size",
                return_value=(0, 8),
            ),
            mock.patch.object(
                hybrid_pool_assembler, "enforce_hicache_host_budget"
            ) as enforce,
            mock.patch.object(
                hybrid_pool_assembler,
                "maybe_prebuild_mla_host_dedup",
                side_effect=RuntimeError("stop before allocation"),
            ),
            mock.patch.object(
                hybrid_pool_assembler, "build_kv_host_pool"
            ) as build_host_pool,
        ):
            with self.assertRaisesRegex(RuntimeError, "stop before allocation"):
                hybrid_pool_assembler.build_hybrid_mamba_stack(
                    params=params,
                    kv_pool=mock.sentinel.kv_pool,
                    mamba_pool=mock.sentinel.mamba_pool,
                    full_layer_mapping={0: 0},
                    mamba_layer_mapping={1: 0},
                    load_cache_event=None,
                    storage_backend=None,
                    use_mla=True,
                    enable_mla_hicache_host_dedup=True,
                )

        enforce.assert_called_once_with(
            target_bytes=100,
            rank_local_bytes={
                "mamba": 20,
                "allocator_metadata": 218,
                "draft": 30,
            },
            tp_size=8,
            context=(
                "hybrid MLA+Mamba L2 (target_tokens=10, mamba_slots=2, draft_tokens=10)"
            ),
        )
        build_host_pool.assert_not_called()

    def test_mamba_host_sizing_preserves_legacy_fixed_size_without_opt_in(self):
        self.assertEqual(
            hybrid_pool_assembler._get_mamba_host_sizing(
                hicache_ratio=3.0,
                fixed_size=230,
                dedup_enabled=False,
            ),
            (3.0, 230),
        )

    def test_mamba_host_sizing_is_independent_for_dedup(self):
        self.assertEqual(
            hybrid_pool_assembler._get_mamba_host_sizing(
                hicache_ratio=3.0,
                fixed_size=230,
                dedup_enabled=True,
            ),
            (3.0, 0),
        )

    def test_dedup_fixed_hicache_size_fully_sizes_target(self):
        params = SimpleNamespace(
            page_size=64,
            hicache_draft_kv_pool=None,
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
            mtp_draft_device_pools=(),
            token_to_kv_pool_allocator=object(),
            req_to_token_pool=SimpleNamespace(
                mamba_allocator=SimpleNamespace(
                    alloc=lambda *args: None, free=lambda *args: None
                )
            ),
        )
        memory = SimpleNamespace(
            hicache_ratio=3.0,
            hicache_size=230,
            hicache_write_policy="write_through_selective",
            hicache_io_backend="kernel",
            hicache_host_memory_mode="cache",
            hicache_mem_layout="page_first",
        )
        kv_pool = mock.Mock()
        kv_pool.get_kv_size_bytes.return_value = 100
        mamba_pool = mock.Mock()
        mamba_pool.get_kv_size_bytes.return_value = 100
        base_kwargs = dict(
            params=params,
            kv_pool=kv_pool,
            mamba_pool=mamba_pool,
            full_layer_mapping={0: 0},
            mamba_layer_mapping={1: 0},
            load_cache_event=None,
            storage_backend=None,
            use_mla=True,
        )

        with (
            mock.patch.object(hybrid_pool_assembler, "get_memory", return_value=memory),
            mock.patch.object(
                hybrid_pool_assembler,
                "estimate_mla_host_pool_bytes",
                return_value=(100, 10),
            ),
            mock.patch.object(
                hybrid_pool_assembler,
                "estimate_mamba_host_pool_bytes",
                return_value=(20, 2),
            ),
            mock.patch.object(
                hybrid_pool_assembler,
                "mla_dedup_rank_and_size",
                return_value=(0, 8),
            ),
            mock.patch.object(hybrid_pool_assembler, "enforce_hicache_host_budget"),
            mock.patch.object(
                hybrid_pool_assembler,
                "maybe_prebuild_mla_host_dedup",
                return_value=None,
            ),
            mock.patch.object(
                hybrid_pool_assembler,
                "is_mla_dedup_dummy_rank",
                return_value=False,
            ),
            mock.patch.object(
                hybrid_pool_assembler, "build_kv_host_pool"
            ) as build_host_pool,
            mock.patch.object(hybrid_pool_assembler, "MambaPoolHost"),
            mock.patch.object(hybrid_pool_assembler, "HybridCacheController"),
            mock.patch.object(
                hybrid_pool_assembler, "_get_allocator_type", return_value="default"
            ),
        ):
            hybrid_pool_assembler.build_hybrid_mamba_stack(
                enable_mla_hicache_host_dedup=True, **base_kwargs
            )

        # The whole fixed budget sizes the single deduplicated target pool,
        # matching the preflight estimate; Mamba sizes off the ratio.
        self.assertEqual(build_host_pool.call_args.kwargs["host_size"], 230)

        with (
            mock.patch.object(hybrid_pool_assembler, "get_memory", return_value=memory),
            mock.patch.object(
                hybrid_pool_assembler, "build_kv_host_pool"
            ) as build_host_pool,
            mock.patch.object(hybrid_pool_assembler, "MambaPoolHost"),
            mock.patch.object(hybrid_pool_assembler, "HybridCacheController"),
            mock.patch.object(
                hybrid_pool_assembler, "_get_allocator_type", return_value="default"
            ),
        ):
            hybrid_pool_assembler.build_hybrid_mamba_stack(
                enable_mla_hicache_host_dedup=False, **base_kwargs
            )

        # Legacy non-dedup behavior still splits the fixed size across pools.
        self.assertEqual(build_host_pool.call_args.kwargs["host_size"], 115)

    def test_mla_dedup_requires_dense_stage_local_layer_ids(self):
        hybrid_pool_assembler._require_dense_layer_ids(
            mappings=({0: 0, 2: 1}, {1: 0, 3: 1}),
            transfer_layer_num=4,
            context="unit-test",
        )

        with self.assertRaisesRegex(ValueError, "dense stage-local layer ids"):
            hybrid_pool_assembler._require_dense_layer_ids(
                mappings=({4: 0}, {5: 0}),
                transfer_layer_num=2,
                context="unit-test",
            )

    def test_dedup_draft_requires_the_target_slot_domain(self):
        controller = SimpleNamespace(
            mla_broadcast_enabled=True,
            mem_pool_device=SimpleNamespace(size=1024, page_size=64),
        )
        dflash = SimpleNamespace(is_dflash=lambda: True)
        kv_cache_builder._validate_dedup_draft_index_domain(
            cache_controller=controller,
            draft_pool=SimpleNamespace(size=1024, page_size=64),
            spec_algorithm=dflash,
        )

        with self.assertRaisesRegex(ValueError, "requires DFlash"):
            kv_cache_builder._validate_dedup_draft_index_domain(
                cache_controller=controller,
                draft_pool=SimpleNamespace(size=1024, page_size=64),
                spec_algorithm=SimpleNamespace(is_dflash=lambda: False),
            )

        with self.assertRaisesRegex(ValueError, "share one global KV slot domain"):
            kv_cache_builder._validate_dedup_draft_index_domain(
                cache_controller=controller,
                draft_pool=SimpleNamespace(size=512, page_size=64),
                spec_algorithm=dflash,
            )

    def test_mla_dedup_dummy_prefetch_reports_only_through_acks(self):
        operation = SimpleNamespace(
            hash_value=["p0", "p1", "p2"], completed_tokens=0, request_id="r"
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = 2
        controller.prefetch_sync_queue = mock.Mock()

        completed = controller._page_transfer_dummy(operation)

        self.assertEqual(completed, 3)
        # The optimistic count lives only in the ack stream; the ACK drain
        # owns operation.completed_tokens after the cross-rank MIN, so a
        # short source read cannot trip its assertion on this rank.
        self.assertEqual(operation.completed_tokens, 0)
        [ack] = [
            call.args[0] for call in controller.prefetch_sync_queue.put.call_args_list
        ]
        self.assertEqual(ack.completed_tokens, 6)


if __name__ == "__main__":
    unittest.main()
