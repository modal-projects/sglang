"""PP intake owns CUDA IPC bytes before the raw request is pickled for relay.

Native proxy/caller and PP serialization code run unchanged. CUDA storage,
allocation, streams, driver operations, and network calls use CPU boundaries.
"""

import pickle
import unittest
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import schedule_batch, scheduler
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.multimodal.transport import cuda_ipc, memory_pool
from sglang.srt.utils.common import point_to_point_pyobj

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestCudaIpcPipelineIntake(CustomTestCase):
    def setUp(self):
        self.parallel = SimpleNamespace(
            tp_size=2,
            tp_rank=0,
            pp_size=2,
            pp_rank=0,
            attn_tp_size=2,
            attn_tp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            attn_dp_rank=0,
        )
        self.config = SimpleNamespace(enable_broadcast_mm_inputs_process=True)
        self.storages = {}
        self.opens = []
        self.releases = []
        self.writes = []
        self.broadcasts = []
        self.wire = deque()
        self.fail_allocations = False
        real_empty = torch.empty

        def cpu_empty(*args, **kwargs):
            device = kwargs.get("device")
            if device is not None and torch.device(device).type == "cuda":
                if self.fail_allocations and args[0] != 0:
                    raise RuntimeError("consumer copy allocation failed")
                kwargs["device"] = "cpu"
            return real_empty(*args, **kwargs)

        patches = [
            patch.object(cuda_ipc, "_pool_imported_generations", {}),
            patch.object(cuda_ipc, "_pool_acknowledged_generations", {}),
            patch.object(cuda_ipc, "_pool_storage_cache", {}),
            patch.object(torch, "empty", side_effect=cpu_empty),
            patch.object(torch.cuda, "device", side_effect=lambda *_: nullcontext()),
            patch.object(
                torch.cuda, "current_device", side_effect=lambda: self.parallel.tp_rank
            ),
            patch.object(torch.cuda, "current_stream", return_value=Mock()),
            patch.object(memory_pool, "get_parallel", return_value=self.parallel),
            patch.object(schedule_batch, "get_parallel", return_value=self.parallel),
            patch.object(scheduler, "get_parallel", return_value=self.parallel),
            patch.object(scheduler, "get_mm", return_value=self.config),
            patch.object(
                schedule_batch.envs.SGLANG_MM_BUFFER_SIZE_MB, "get", return_value=0
            ),
            patch.object(
                cuda_ipc, "_open_pooled_storage_uncached", side_effect=self._open
            ),
            patch.object(
                cuda_ipc, "_release_ipc_export", side_effect=self.releases.append
            ),
            patch.object(memory_pool, "stream_wait_value32"),
            patch.object(cuda_ipc, "stream_write_value32", side_effect=self._write),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(
                torch.distributed, "broadcast_object_list", side_effect=self._broadcast
            ),
            patch.object(
                torch.distributed,
                "send",
                side_effect=lambda tensor, *a, **kw: self.wire.append(tensor.clone()),
            ),
            patch.object(torch.distributed, "irecv", side_effect=self._receive),
        ]
        for context in patches:
            context.start()
            self.addCleanup(context.stop)

    def _open(self, handle):
        self.opens.append(handle)
        return self.storages[handle[1]].untyped_storage()

    def _write(self, device, address, generation, transport):
        for name, storage in self.storages.items():
            offset = address - storage.data_ptr()
            if 0 <= offset < storage.numel():
                storage.view(torch.int32)[offset // 4] = generation
                self.writes.append((name, offset, generation))
                return
        self.fail("Acknowledgement outside the producer allocation")

    def _broadcast(self, objects, **kwargs):
        if self.parallel.tp_rank == 0:
            self.broadcasts.append(pickle.dumps(objects[0]))
        else:
            objects[0] = pickle.loads(self.broadcasts[-1])

    def _receive(self, tensor, **kwargs):
        tensor.copy_(self.wire.popleft())
        return SimpleNamespace(wait=lambda: None)

    def _scheduler(self, rank):
        self.parallel.tp_rank = rank
        self.parallel.attn_tp_rank = rank
        result = Scheduler.__new__(Scheduler)
        result.dp_tp_group = SimpleNamespace(rank_in_group=rank, first_rank=0)
        result.dp_tp_cpu_group = object()
        result.ps = self.parallel
        result.world_group = SimpleNamespace(cpu_group=object())
        return result

    def _item(self, field, value=11):
        expected = torch.full((2, 2), float(value))
        name = uuid4().hex
        storage = torch.zeros(512, dtype=torch.uint8)
        storage.view(torch.int32)[0] = 1
        storage[256:272].copy_(expected.view(torch.uint8).reshape(-1))
        self.storages[name] = storage
        handles = tuple(
            (0, name, 512, 0, f"{name}-recipient-{rank}", 0, b"event", False)
            for rank in range(2)
        )
        proxy = cuda_ipc.CudaIpcTensorTransportProxy(
            data=storage[256:272],
            info_data=expected,
            pool_ipc_handle=handles[0],
            pool_ipc_handles=handles,
            pool_id=name,
            pool_byte_offset=256,
            ready_byte_offset=0,
            ack_byte_offset=4,
            generation=1,
            total_consumer_count=2,
            use_pool_handle_cache=True,
        )
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=value,
            pad_value=value,
            offsets=[(0, 1)],
            model_specific_data={
                cuda_ipc.DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY: True
            },
        )
        if field == "auxiliary":
            item.feature = torch.ones(2, 2)
            item.model_specific_data[field] = expected
            item.set_pad_value()
            item.model_specific_data[field] = proxy
        else:
            setattr(item, field, expected)
            item.set_pad_value()
            setattr(item, field, proxy)
        return item, expected

    @staticmethod
    def _field(item, field):
        return (
            item.model_specific_data[field]
            if field == "auxiliary"
            else getattr(item, field)
        )

    def _relay(self, sender, raw):
        sender._pp_send_pyobj_to_next_stage([SimpleNamespace(mm_inputs=raw)])
        return point_to_point_pyobj(
            [], rank=2, src=0, dst=2, group=sender.world_group.cpu_group
        )[0].mm_inputs

    def test_broadcast_enabled_claims_every_raw_ipc_recipient(self):
        """Leader-only MM processing cannot leave another recipient's ticket unopened."""
        for field in ("feature", "precomputed_embeddings", "auxiliary"):
            with self.subTest(field=field):
                item, expected = self._item(field)
                encoded = pickle.dumps(MultimodalProcessorOutput(mm_items=[item]))
                before = len(self.opens)
                for rank in (0, 1):
                    raw = pickle.loads(encoded)
                    result = self._scheduler(rank)._get_multimodal_inputs(raw)
                    self.assertFalse(raw.mm_items[0].has_cuda_ipc_proxy())
                    torch.testing.assert_close(
                        self._field(raw.mm_items[0], field), expected
                    )
                    torch.testing.assert_close(
                        self._field(result.mm_items[0], field), expected
                    )
                self.assertEqual(len(self.opens) - before, 2)
                self.assertEqual(self.opens[-2][4], f"{self.opens[-2][1]}-recipient-0")
                self.assertEqual(self.opens[-1][4], f"{self.opens[-1][1]}-recipient-1")
                self.assertEqual(
                    self.storages[self.opens[-1][1]][:12].view(torch.int32).tolist(),
                    [1, 1, 1],
                )

    def test_pipeline_pickle_relay_contains_owned_fields(self):
        """Later PP stages receive stable tensors after all first-stage tickets retire."""
        for prebuilt in (False, True):
            with self.subTest(prebuilt=prebuilt):
                fields = ("feature", "precomputed_embeddings", "auxiliary")
                pairs = [
                    self._item(field, 20 + index) for index, field in enumerate(fields)
                ]
                items = [item for item, _ in pairs]
                raw = (
                    MultimodalInputs(mm_items=items)
                    if prebuilt
                    else MultimodalProcessorOutput(mm_items=items)
                )
                encoded = pickle.dumps(raw)
                sender = self._scheduler(0)
                sender._get_multimodal_inputs(raw)
                self._scheduler(1)._get_multimodal_inputs(pickle.loads(encoded))
                for storage in self.storages.values():
                    storage[256:272].fill_(255)
                claims = len(self.opens), len(self.writes), len(self.releases)
                sender = self._scheduler(0)
                received = self._relay(sender, raw)
                self.assertFalse(
                    any(item.has_cuda_ipc_proxy() for item in received.mm_items)
                )
                for item, field, (_, expected) in zip(received.mm_items, fields, pairs):
                    torch.testing.assert_close(self._field(item, field), expected)
                self.assertEqual(
                    [item.cache_key for item in received.mm_items],
                    [item.cache_key for item in items],
                )
                self.parallel.pp_rank = 1
                later = self._scheduler(0)._get_multimodal_inputs(received)
                self.assertFalse(
                    any(item.has_cuda_ipc_proxy() for item in later.mm_items)
                )
                self.assertEqual(
                    (len(self.opens), len(self.writes), len(self.releases)), claims
                )
                self.parallel.pp_rank = 0

    def test_single_stage_keeps_selected_deferred_feature(self):
        """The single-stage lazy path keeps its ticket until local feature use."""
        self.parallel.pp_size = 1
        item, expected = self._item("feature")
        identity = item.cache_key
        encoded = pickle.dumps(MultimodalProcessorOutput(mm_items=[item]))
        for rank in (0, 1):
            raw = pickle.loads(encoded)
            result = self._scheduler(rank)._get_multimodal_inputs(raw)
            self.assertIsInstance(
                result.mm_items[0].feature, cuda_ipc.CudaIpcTensorTransportProxy
            )
            self.assertEqual(len(self.opens), rank)
            result.mm_items[0].materialize_deferred_cuda_ipc_feature()
            torch.testing.assert_close(result.mm_items[0].feature, expected)
            self.assertEqual(result.mm_items[0].cache_key, identity)
        self.assertEqual(len(self.opens), 2)

    def test_single_stage_reconstructs_before_computing_missing_identity(self):
        """An old sender's routing hash keeps the eager intake fallback valid."""
        self.parallel.pp_size = 1
        item, expected = self._item("feature")
        identity = item.cache_key
        item.cache_identity = None
        encoded = pickle.dumps(MultimodalProcessorOutput(mm_items=[item]))
        for rank in (0, 1):
            raw = pickle.loads(encoded)
            result = self._scheduler(rank)._get_multimodal_inputs(raw)
            torch.testing.assert_close(result.mm_items[0].feature, expected)
            self.assertEqual(result.mm_items[0].cache_key, identity)
            self.assertEqual(len(self.opens), rank + 1)
        self.assertEqual(len(self.opens), 2)

    def test_cpu_inputs_keep_broadcast_processing_knob(self):
        """CPU preprocessing still broadcasts the leader result only when enabled."""
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                self.config.enable_broadcast_mm_inputs_process = enabled
                results = []
                for rank in (0, 1):
                    item = MultimodalDataItem(
                        modality=Modality.IMAGE,
                        hash=10,
                        pad_value=10,
                        feature=torch.full((2, 2), float(rank)),
                    )
                    result = self._scheduler(rank)._get_multimodal_inputs(
                        MultimodalProcessorOutput(mm_items=[item])
                    )
                    results.append(result.mm_items[0].feature)
                self.assertEqual(float(results[1][0, 0]), 0 if enabled else 1)

    def test_copy_failure_propagates_before_pipeline_relay(self):
        """Failed eager conversion cannot forward an unconsumed native ticket."""
        item, _ = self._item("feature")
        raw = MultimodalProcessorOutput(mm_items=[item])
        receiver = self._scheduler(0)
        self.fail_allocations = True
        with self.assertRaisesRegex(RuntimeError, "consumer copy allocation failed"):
            receiver._get_multimodal_inputs(raw)
            self._relay(receiver, raw)
        self.assertFalse(self.wire)
        self.assertTrue(item.feature._consumer_acknowledged)
        self.assertEqual(
            self.storages[self.opens[0][1]][:12].view(torch.int32).tolist(), [1, 1, 0]
        )


if __name__ == "__main__":
    unittest.main()
