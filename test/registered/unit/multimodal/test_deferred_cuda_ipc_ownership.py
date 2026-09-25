"""Deferred feature reuse with native ownership logic and CPU CUDA boundaries.

The storage mapping, allocations, streams, and driver calls run on CPU. These
tests establish byte ownership and protocol transitions, not GPU stream timing.
"""

import pickle
import threading
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import torch

from sglang.srt.managers import mm_schedule, schedule_batch
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.multimodal import mm_utils
from sglang.srt.multimodal.transport import cuda_ipc, memory_pool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDeferredCudaIpcOwnership(CustomTestCase):
    def setUp(self):
        self.old_cache = mm_schedule.embedding_cache
        self.addCleanup(setattr, mm_schedule, "embedding_cache", self.old_cache)
        mm_schedule.init_mm_embedding_cache(16)
        self.expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        self.parallel = SimpleNamespace()
        self._configure_pool(consumers=2, attn_tp=2, attn_cp=1)

        def cpu_allocation(function):
            def allocate(*args, **kwargs):
                device = kwargs.get("device")
                if device is not None and torch.device(device).type == "cuda":
                    kwargs["device"] = "cpu"
                return function(*args, **kwargs)

            return allocate

        patches = [
            patch.object(torch, "empty", side_effect=cpu_allocation(torch.empty)),
            patch.object(torch, "tensor", side_effect=cpu_allocation(torch.tensor)),
            patch.object(torch.cuda, "device", side_effect=lambda *_: nullcontext()),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "current_stream", return_value=Mock()),
            patch.object(memory_pool, "get_parallel", return_value=self.parallel),
            patch.object(mm_schedule, "get_parallel", return_value=self.parallel),
            patch.object(schedule_batch, "get_parallel", return_value=self.parallel),
            patch.object(mm_utils, "get_parallel", return_value=self.parallel),
            patch.object(memory_pool, "stream_wait_value32", side_effect=self._wait),
            patch.object(cuda_ipc, "stream_write_value32", side_effect=self._write),
            patch.object(cuda_ipc, "_release_ipc_export"),
            patch.object(
                cuda_ipc,
                "_open_pooled_storage_uncached",
                side_effect=lambda _: self.storage.untyped_storage(),
            ),
        ]
        for context in patches:
            context.start()
            self.addCleanup(context.stop)
        self.addCleanup(cuda_ipc._pool_handle_cache_clear)

    def _configure_pool(self, *, consumers, attn_tp, attn_cp):
        self.storage = torch.zeros(512, dtype=torch.uint8)
        self.pool_id = uuid4().hex
        self.writes = []
        self.parallel.tp_size = consumers
        self.parallel.attn_tp_size = attn_tp
        self.parallel.attn_cp_size = attn_cp
        self.parallel.pp_size = 1
        self._select_rank(0)
        # Construct the CPU storage boundary without the CUDA-only constructor.
        # Allocation, recycling, and all consumer methods remain production code.
        self.pool = object.__new__(memory_pool.StreamOrderedMmFeaturePool)
        self.pool.device_id = 0
        self.pool._lock = threading.Lock()
        words = consumers + 1
        self.pool._control_words = (
            self.storage[: words * 4].view(torch.int32).reshape(1, words)
        )
        self.pool._available_ranges = [(256, 512)]
        self.pool._available_slots = [0]
        self.pool._slot_generations = [0]
        self.pool._occupied = {}
        self.pool.control_words_per_slot = words
        self.pool.transport_name = "CUDA IPC"

    def _select_rank(self, rank):
        self.parallel.tp_rank = rank
        self.parallel.attn_tp_rank = rank % self.parallel.attn_tp_size
        self.parallel.attn_cp_rank = (
            rank // self.parallel.attn_tp_size
        ) % self.parallel.attn_cp_size

    def _word(self, address):
        return (address - self.storage.data_ptr()) // memory_pool.CONTROL_WORD_BYTES

    def _wait(self, device_id, address, generation, transport_name):
        self.assertEqual(
            int(self.storage.view(torch.int32)[self._word(address)]), generation
        )

    def _write(self, device_id, address, generation, transport_name):
        word = self._word(address)
        self.storage.view(torch.int32)[word] = generation
        self.writes.append((word, generation))

    def _publish(self, value):
        lease = self.pool._allocate_locked(value.numel() * value.element_size())
        self.assertIsNotNone(lease)
        data = self.storage[lease.start : lease.start + lease.nbytes]
        data.copy_(value.view(torch.uint8).reshape(-1))
        self.pool._control_words[0, 0] = lease.generation
        handles = tuple(
            (0, self.pool_id, 512, 0, (lease.generation, rank), 0, b"event", False)
            for rank in range(self.parallel.tp_size)
        )
        return cuda_ipc.CudaIpcTensorTransportProxy(
            data=data,
            info_data=value,
            pool_ipc_handle=handles[0],
            pool_ipc_handles=handles,
            pool_id=self.pool_id,
            pool_byte_offset=lease.start,
            ready_byte_offset=lease.ready_byte_offset,
            ack_byte_offset=lease.ack_byte_offset,
            generation=lease.generation,
            total_consumer_count=self.parallel.tp_size,
            use_pool_handle_cache=True,
        )

    def _item(self, proxy):
        return MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=11,
            pad_value=1001,
            offsets=[(0, 1)],
            feature=pickle.loads(pickle.dumps(proxy)),
            model_specific_data={
                cuda_ipc.DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY: True,
                "image_grid_thw": torch.tensor([[1, 1, 2]]),
            },
        )

    def _recycle(self):
        with self.pool._lock:
            self.pool._recycle_ready_leases_locked()

    def _embed(self, item, encode, *, prefix):
        return mm_schedule.get_embedding_and_mask(
            encode,
            [item],
            torch.tensor([1001]),
            torch.tensor([1001, 1001] if prefix == 0 else [1, 2]),
            [0, 1],
            [prefix],
            [2],
            [[(0, 1)]],
        )

    def test_prefix_resident_features_drain_and_survive_pool_reuse(self):
        """A skipped prefix must retain features before reuse and later retraction."""
        proxy = self._publish(self.expected)
        items = [self._item(proxy), self._item(proxy)]
        for rank, item in enumerate(items):
            self._select_rank(rank)
            embedding, mask, ids = self._embed(
                item,
                lambda _: self.fail("Prefix-resident image entered encoder"),
                prefix=2,
            )
            self.assertIsNone(embedding)
            self.assertIsNone(mask)
            self.assertEqual(ids.tolist(), [1, 2])
            self._recycle()
            self.assertEqual(self.pool.active_lease_count, 1 if rank == 0 else 0)
        self.assertEqual(self.writes, [(1, 1), (2, 1)])

        replacement = self._publish(torch.full_like(self.expected, -500))
        self.assertEqual(replacement.generation, 2)
        self.assertEqual(replacement.proxy_state["ipc_extra"]["pool_byte_offset"], 256)
        for rank, item in enumerate(items):
            self._select_rank(rank)
            mm_schedule.embedding_cache.clear()
            embedding, mask, _ = self._embed(
                item,
                lambda batch: torch.cat([x.feature * 3 + 7 for x in batch]),
                prefix=0,
            )
            torch.testing.assert_close(embedding, self.expected * 3 + 7)
            torch.testing.assert_close(item.feature, self.expected)
            self.assertTrue(mask.all())
        self.assertEqual(self.writes, [(1, 1), (2, 1)])

    def test_receiver_subgroup_drains_after_delayed_peer_copies(self):
        """Inactive global ranks cannot leak a lease or release a delayed reader."""
        for attn_tp, attn_cp in ((2, 1), (1, 2)):
            with self.subTest(attn_tp=attn_tp, attn_cp=attn_cp):
                self._configure_pool(consumers=4, attn_tp=attn_tp, attn_cp=attn_cp)
                proxy = self._publish(self.expected)
                leader, peer = self._item(proxy), self._item(proxy)
                self._select_rank(2)
                self._embed(
                    leader,
                    lambda _: self.fail("Prefix-resident image entered encoder"),
                    prefix=2,
                )
                self._recycle()
                self.assertEqual(self.pool.active_lease_count, 1)
                self.assertEqual(int(self.pool._control_words[0, 4]), 0)
                self.assertIsNone(self.pool._allocate_locked(16))

                self._select_rank(3)
                self._embed(
                    peer,
                    lambda _: self.fail("Prefix-resident image entered encoder"),
                    prefix=2,
                )
                self._recycle()
                self.assertEqual(self.pool.active_lease_count, 0)
                self.assertEqual(self.writes, [(3, 1), (1, 1), (2, 1), (4, 1)])
                replacement = self._publish(torch.full_like(self.expected, -500))
                self.assertEqual(replacement.generation, 2)
                torch.testing.assert_close(leader.feature, self.expected)
                torch.testing.assert_close(peer.feature, self.expected)

    def test_rejected_receiver_subgroup_drains_without_copying(self):
        """Early rejection must release absent global consumers and both live ranks."""
        for attn_tp, attn_cp in ((2, 1), (1, 2)):
            with self.subTest(attn_tp=attn_tp, attn_cp=attn_cp):
                self._configure_pool(consumers=4, attn_tp=attn_tp, attn_cp=attn_cp)
                proxy = self._publish(self.expected)
                leader, peer = self._item(proxy), self._item(proxy)
                proxies = [leader.feature, peer.feature]
                self._select_rank(2)
                MultimodalInputs(mm_items=[leader]).release_features()
                self._recycle()
                self.assertIsNone(leader.feature)
                self.assertEqual(self.pool.active_lease_count, 1)
                self.assertEqual(int(self.pool._control_words[0, 4]), 0)
                self.assertIsNone(self.pool._allocate_locked(16))

                self._select_rank(3)
                MultimodalInputs(mm_items=[peer]).release_features()
                self._recycle()
                self.assertIsNone(peer.feature)
                self.assertEqual(self.pool.active_lease_count, 0)
                self.assertTrue(all(p.reconstruct_tensor is None for p in proxies))
                self.assertEqual(self.writes, [(3, 1), (1, 1), (2, 1), (4, 1)])
                replacement = self._publish(torch.full_like(self.expected, -500))
                self.assertEqual(replacement.generation, 2)

    def test_k3_nonowner_can_become_owner_after_pool_reuse(self):
        """An image skipped by one DP rank must survive a later owner reassignment."""
        from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration

        class Tower:
            device = torch.device("cuda:0")
            merge_kernel_size = (1, 1)
            config = SimpleNamespace(hidden_size=2)
            patch_embed = SimpleNamespace(proj=SimpleNamespace(weight=torch.empty(1)))

            def __init__(self):
                self.inputs = []

            def __call__(self, features, **kwargs):
                self.inputs.append(features.clone())
                return (features * 3 + 7).unsqueeze(1)

        expected_embedding = (self.expected * 3 + 7).unsqueeze(1)
        companion = torch.full((3, 2), 42.0)

        class PeerGroup:
            def broadcast(self, output, src):
                output.copy_(expected_embedding)

            def all_gather(self, local, dim):
                return torch.cat([(companion * 3 + 7).unsqueeze(1), local], dim=dim)

        model = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.use_data_parallel = True
        model.vision_tower = Tower()
        model.mm_projector = lambda value: value
        self.parallel.attn_tp_size = 2
        self.parallel.attn_tp_rank = 1
        self.parallel.tp_rank = 1
        self.parallel.attn_tp_group = PeerGroup()
        proxy = self._publish(self.expected)
        item, peer_item = self._item(proxy), self._item(proxy)

        first = model.get_image_feature([item])
        torch.testing.assert_close(first, expected_embedding)
        self.assertEqual(model.vision_tower.inputs, [])
        self.assertEqual(self.writes, [(2, 1)])
        self.parallel.tp_rank = 0
        peer_item.materialize_deferred_cuda_ipc_feature()
        self._recycle()
        self.assertEqual(self.pool.active_lease_count, 0)
        self._publish(torch.full_like(self.expected, -500))

        self.parallel.tp_rank = 1
        larger_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=companion,
            model_specific_data={"image_grid_thw": torch.tensor([[1, 1, 3]])},
        )
        result = model.get_image_feature([item, larger_item])
        self.assertEqual(len(model.vision_tower.inputs), 1)
        torch.testing.assert_close(model.vision_tower.inputs[0], self.expected)
        torch.testing.assert_close(
            result, torch.cat([expected_embedding, (companion * 3 + 7).unsqueeze(1)])
        )
        self.assertEqual(self.writes, [(2, 1), (1, 1)])


if __name__ == "__main__":
    unittest.main()
