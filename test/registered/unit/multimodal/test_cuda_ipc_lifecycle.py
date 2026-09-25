"""CPU protocol regressions for pooled CUDA IPC lease ownership."""

import pickle
import threading
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import torch

from sglang.srt.multimodal.transport import cuda_ipc, memory_pool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestCudaIpcLeaseLifecycle(CustomTestCase):
    def setUp(self):
        self.source = torch.arange(4, dtype=torch.float32)
        self.bytes = self.source.view(torch.uint8)
        self.writes = []
        self.stream = Mock()
        self.write_effect = self._write
        self.write = Mock(side_effect=lambda *args: self.write_effect(*args))
        real_empty = torch.empty

        def cpu_empty(*args, **kwargs):
            kwargs["device"] = "cpu"
            return real_empty(*args, **kwargs)

        patches = [
            patch.object(cuda_ipc, "_pool_acknowledged_generations", {}, create=True),
            patch.object(cuda_ipc, "_pool_imported_generations", {}, create=True),
            patch.object(torch, "empty", side_effect=cpu_empty),
            patch.object(torch.cuda, "device", side_effect=lambda *_: nullcontext()),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "current_stream", return_value=self.stream),
            patch.object(memory_pool, "stream_wait_value32"),
            patch.object(cuda_ipc, "stream_wait_value32", create=True),
            patch.object(cuda_ipc, "stream_write_value32", self.write, create=True),
            patch.object(memory_pool, "stream_write_value32", self.write),
            patch.object(
                cuda_ipc.CudaIpcTensorTransportProxy,
                "_open_pool_slice",
                return_value=(self.bytes, self.bytes.untyped_storage()),
            ),
        ]
        for context in patches:
            context.start()
            self.addCleanup(context.stop)

    def _write(self, device_id, address, generation, transport_name):
        self.writes.append((address, generation))

    def _proxy(self, generation=1, *, handle=("pool",), slot=0, consumers=2):
        ready = slot * (consumers + 1) * memory_pool.CONTROL_WORD_BYTES
        return cuda_ipc.CudaIpcTensorTransportProxy(
            data=self.bytes,
            info_data=self.source,
            pool_ipc_handle=handle,
            pool_byte_offset=256,
            ready_byte_offset=ready,
            ack_byte_offset=ready + memory_pool.CONTROL_WORD_BYTES,
            generation=generation,
            total_consumer_count=consumers,
            use_pool_handle_cache=True,
        )

    def test_serialized_alias_cannot_read_or_acknowledge_reused_slot(self):
        """A delayed alias must not read recycled bytes or overwrite newer acks."""
        owner = self._proxy()
        serialized = pickle.dumps(owner)
        owner.acknowledge_consumption(2)
        for generation in (1, 2):
            with self.subTest(generation=generation):
                if generation == 2:
                    self._proxy(generation=2).acknowledge_consumption(2)
                alias = pickle.loads(serialized)
                with self.assertRaisesRegex(RuntimeError, "acknowledged CUDA IPC"):
                    alias.reconstruct_on_target_device(0, consumer_rank=0)
                with patch.object(
                    cuda_ipc, "resolve_consumer_rank", return_value=0, create=True
                ):
                    with self.assertRaisesRegex(RuntimeError, "acknowledged CUDA IPC"):
                        alias.borrow_on_target_device(0)
                written = list(self.writes)
                alias.acknowledge_consumption(2)
                self.assertEqual(self.writes, written)
                self.assertTrue(alias._consumer_acknowledged)

    def test_acknowledgements_are_independent_per_consumer_and_pool(self):
        """One rank's release cannot suppress another rank or another pool."""
        self._proxy().acknowledge_consumption(consumer_rank=0)
        other_rank = self._proxy()
        other_rank.acknowledge_consumption(consumer_rank=1)
        self._proxy(handle=("another",)).acknowledge_consumption(2)
        self.assertEqual(len(self.writes), 4)
        self.assertEqual(self.writes[0][0] + 4, self.writes[1][0])

    def test_partial_acknowledgement_retries_only_missing_ranks(self):
        """An enqueue failure must preserve successful writes without retiring others."""
        proxy = self._proxy(consumers=3)
        base_address = self.bytes.untyped_storage().data_ptr()

        def fail_second(device_id, address, generation, transport_name):
            if address == base_address + 8:
                raise RuntimeError("write failed")
            self._write(device_id, address, generation, transport_name)

        with patch.object(self, "write_effect", fail_second):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                proxy.acknowledge_consumption(3)
        self.assertFalse(proxy._consumer_acknowledged)
        proxy.acknowledge_consumption(3)
        self.assertEqual(
            self.writes,
            [(base_address + offset, 1) for offset in (4, 8, 12)],
        )
        self.assertTrue(proxy._consumer_acknowledged)

    def test_failed_first_ack_does_not_retire_the_lease(self):
        """A failed enqueue cannot make subsequent release silently disappear."""
        proxy = self._proxy()
        with patch.object(
            self, "write_effect", Mock(side_effect=RuntimeError("write failed"))
        ):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                proxy.acknowledge_consumption(2)
        self.assertFalse(proxy._consumer_acknowledged)
        self._proxy().acknowledge_consumption(2)
        self.assertEqual(len(self.writes), 2)

    def test_reconstruction_retry_keeps_owned_copy_and_orders_its_stream(self):
        """Release retries must not reread bytes already released by another alias."""
        proxy = self._proxy()
        with (
            patch.object(
                self, "write_effect", Mock(side_effect=RuntimeError("write failed"))
            ),
            self.assertRaisesRegex(RuntimeError, "write failed"),
        ):
            proxy.reconstruct_on_target_device(0, consumer_count=2, consumer_rank=0)
        self._proxy().acknowledge_consumption(2)
        expected = self.source.clone()
        self.source.fill_(-1)
        next_stream = Mock()
        with patch.object(torch.cuda, "current_stream", return_value=next_stream):
            reconstructed = proxy.reconstruct_on_target_device(
                0, consumer_count=2, consumer_rank=0
            )
        self.assertTrue(torch.equal(reconstructed, expected))
        next_stream.wait_stream.assert_called_once_with(self.stream)
        self.assertEqual(len(self.writes), 2)

    def test_uncached_release_retry_retains_mapping_until_stream_completion(self):
        """Failed release must keep its mapping alive through a retry on another stream."""
        proxy = self._proxy()
        proxy.proxy_state["ipc_extra"]["use_pool_handle_cache"] = False
        with patch.object(
            proxy,
            "_open_pool_slice",
            return_value=(self.bytes, self.bytes.untyped_storage()),
        ) as open_slice:
            with patch.object(
                self, "write_effect", Mock(side_effect=RuntimeError("write failed"))
            ):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    proxy.acknowledge_consumption(2)
            self.assertIsNotNone(proxy._pool_storage)
            next_stream = Mock()
            ordering = []
            next_stream.wait_stream.side_effect = lambda stream: ordering.append(
                ("wait", stream)
            )

            def write_after_wait(*args):
                ordering.append(("write", None))
                self._write(*args)

            with (
                patch.object(torch.cuda, "current_stream", return_value=next_stream),
                patch.object(self, "write_effect", write_after_wait),
            ):
                proxy.acknowledge_consumption(2)
            self.assertEqual(ordering[0], ("wait", self.stream))
            self.assertEqual(len(self.writes), 2)
            self.assertEqual(open_slice.call_count, 1)
            next_stream.synchronize.assert_called_once()
            self.assertIsNone(proxy._pool_storage)

    def test_slot_history_is_bounded_and_survives_mapping_eviction(self):
        """Slot reuse updates one record; closing a mapping cannot revive old aliases."""
        for generation in range(1, 65):
            for slot in (0, 1):
                self._proxy(generation=generation, slot=slot).acknowledge_consumption(2)
        self.assertEqual(len(cuda_ipc._pool_acknowledged_generations[("pool",)]), 2)
        cuda_ipc._pool_handle_cache_clear()
        with self.assertRaisesRegex(RuntimeError, "acknowledged CUDA IPC"):
            self._proxy().reconstruct_on_target_device(0, consumer_rank=0)

    def test_producer_snapshot_owns_bytes_and_rejects_inactive_lease(self):
        """Sampling clones need independent bytes while the producer retains its lease."""
        pool = object.__new__(cuda_ipc.MmItemMemoryPool)
        pool.device_id = 0
        pool._pool_id = ("pool",)
        pool.memory_pool = torch.zeros(512, dtype=torch.uint8)
        pool.memory_pool[256:272].copy_(self.bytes)
        pool._pool = object.__new__(memory_pool.StreamOrderedMmFeaturePool)
        pool._pool._lock = threading.Lock()
        pool._pool.control_words_per_slot = 3
        pool._pool.base_address = pool.memory_pool.data_ptr()
        pool._pool._occupied = {
            0: memory_pool.PoolLease(
                start=256,
                end=512,
                nbytes=16,
                slot=0,
                generation=1,
                ready_byte_offset=0,
                ack_byte_offset=4,
            )
        }
        proxy = self._proxy()
        snapshot = pool.copy_proxy_to_cpu(proxy)
        pool.memory_pool.zero_()
        self.assertTrue(torch.equal(snapshot, self.source))
        self.assertEqual(self.writes, [])
        self.assertEqual(len(pool._pool._occupied), 1)
        for invalid in (self._proxy(generation=2), self._proxy(handle=("another",))):
            with self.assertRaises(RuntimeError):
                pool.copy_proxy_to_cpu(invalid)

    def test_dp_subgroup_drains_pool_only_after_every_live_reader_copies(self):
        """Inactive ranks may be released by the leader; delayed live ranks retain the slice."""
        real_tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return real_tensor(*args, **kwargs)

        for first_rank in (0, 2):
            with self.subTest(first_rank=first_rank):
                pool = object.__new__(memory_pool.StreamOrderedMmFeaturePool)
                pool._available_ranges = [(256, 512)]
                pool._available_slots = [0]
                pool._slot_generations = [0]
                pool._occupied = {}
                pool._lock = threading.Lock()
                pool.device_id = 0
                pool.control_words_per_slot = 5
                pool.transport_name = "CUDA IPC"
                backing = torch.zeros(512, dtype=torch.uint8)
                pool._control_words = backing[:20].view(torch.int32).reshape(1, 5)
                lease = pool._allocate_locked(self.bytes.numel())
                pool._control_words[0, 0] = lease.generation
                payload = backing[lease.start : lease.start + lease.nbytes]
                payload.copy_(self.bytes)
                leader = self._proxy(handle=("subgroup", first_rank), consumers=4)
                peer = self._proxy(handle=("subgroup", first_rank), consumers=4)
                inactive = tuple(
                    rank for rank in range(4) if rank // 2 != first_rank // 2
                )

                def write_control(device_id, address, generation, transport_name):
                    index = (
                        address - backing.data_ptr()
                    ) // memory_pool.CONTROL_WORD_BYTES
                    pool._control_words[0, index] = generation

                with (
                    patch.object(torch, "tensor", side_effect=cpu_tensor),
                    patch.object(self, "write_effect", write_control),
                    patch.object(
                        cuda_ipc.CudaIpcTensorTransportProxy,
                        "_open_pool_slice",
                        return_value=(payload, backing.untyped_storage()),
                    ),
                ):
                    owned = leader.reconstruct_on_target_device(
                        0,
                        consumer_rank=first_rank,
                        acknowledge_ranks=(first_rank, *inactive),
                    )
                    pool._recycle_ready_leases_locked()
                    self.assertEqual(pool.active_lease_count, 1)
                    self.assertEqual(pool._control_words[0, first_rank + 2], 0)
                    self.assertIsNone(pool._allocate_locked(self.bytes.numel()))
                    delayed = peer.reconstruct_on_target_device(
                        0, consumer_rank=first_rank + 1
                    )
                    pool._recycle_ready_leases_locked()
                    self.assertEqual(pool.active_lease_count, 0)
                    next_lease = pool._allocate_locked(self.bytes.numel())
                    self.assertEqual(next_lease.generation, lease.generation + 1)
                    payload.zero_()
                    self.assertTrue(torch.equal(owned, self.source))
                    self.assertTrue(torch.equal(delayed, self.source))

    def test_explicit_rank_cleanup_preserves_live_peer_after_copy_failure(self):
        """Generic cleanup must keep a reader's selected ranks after copy failure."""
        proxy = self._proxy(consumers=4)
        with patch.object(
            proxy, "_open_pool_slice", side_effect=RuntimeError("open failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "open failed"):
                proxy.reconstruct_on_target_device(
                    0, consumer_rank=2, acknowledge_ranks=(2, 0, 1)
                )
        proxy.release_without_reconstruction(consumer_count=4)
        base_address = self.bytes.untyped_storage().data_ptr()
        self.assertEqual(
            {address - base_address for address, _ in self.writes}, {4, 8, 12}
        )

    def test_rejected_subgroup_releases_inactive_ranks_without_reconstruction(self):
        """A request rejected before copying must account for inactive slots only once."""
        leader = self._proxy(consumers=4)
        peer = self._proxy(consumers=4)
        leader.release_without_reconstruction(
            consumer_rank=2, acknowledge_ranks=(2, 0, 1)
        )
        base_address = self.bytes.untyped_storage().data_ptr()
        self.assertEqual(
            {address - base_address for address, _ in self.writes}, {4, 8, 12}
        )
        self.assertIsNone(leader.reconstruct_tensor)
        self.assertFalse(peer._consumer_acknowledged)
        leader.release_without_reconstruction()
        peer.release_without_reconstruction(consumer_rank=3)
        self.assertEqual(len(self.writes), 4)
        self.assertEqual(self.writes[-1], (base_address + 16, 1))

    def test_explicit_ranks_reject_invalid_or_changed_ownership(self):
        """Invalid rank ownership must fail before enqueuing a read or release."""
        for ranks in ((0, 0), (0, 4), (0, -1), (1, 2), ()):
            with self.subTest(ranks=ranks):
                with self.assertRaisesRegex(ValueError, "acknowledgement ranks"):
                    self._proxy(consumers=4).reconstruct_on_target_device(
                        0, consumer_rank=0, acknowledge_ranks=ranks
                    )
        proxy = self._proxy(consumers=4)
        with patch.object(
            proxy, "_open_pool_slice", side_effect=RuntimeError("open failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "open failed"):
                proxy.acknowledge_consumption(
                    consumer_rank=0, acknowledge_ranks=(0, 2, 3)
                )
        with self.assertRaisesRegex(ValueError, "Cannot change"):
            proxy.acknowledge_consumption(
                consumer_rank=0, acknowledge_ranks=(0, 1, 2, 3)
            )
        self.assertEqual(self.writes, [])


class TestCudaIpcNativeExports(CustomTestCase):
    def setUp(self):
        self.backing = torch.zeros(512, dtype=torch.uint8)
        self.feature = torch.arange(4, dtype=torch.float32)
        self.backing[256:272].copy_(self.feature.view(torch.uint8))
        self.opened = []
        self.released = []
        self.stream = Mock()
        real_empty = torch.empty

        def cpu_empty(*args, **kwargs):
            kwargs["device"] = "cpu"
            return real_empty(*args, **kwargs)

        def open_storage(handle):
            self.opened.append(handle[5])
            return self.backing.untyped_storage()

        for context in (
            patch.object(cuda_ipc, "_pool_acknowledged_generations", {}),
            patch.object(cuda_ipc, "_pool_imported_generations", {}),
            patch.object(cuda_ipc, "_pool_storage_cache", {}),
            patch.object(torch, "empty", side_effect=cpu_empty),
            patch.object(torch.cuda, "device", side_effect=lambda *_: nullcontext()),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "current_stream", return_value=self.stream),
            patch.object(torch.cuda, "ipc_collect"),
            patch.object(memory_pool, "stream_wait_value32"),
            patch.object(cuda_ipc, "stream_write_value32"),
            patch.object(
                cuda_ipc, "_open_pooled_storage_uncached", side_effect=open_storage
            ),
            patch.object(
                cuda_ipc,
                "_release_ipc_export",
                side_effect=lambda handle: self.released.append(handle[5]),
            ),
        ):
            context.start()
            self.addCleanup(context.stop)

    def _proxy(self, generation=1, cache=True):
        handles = tuple(
            (
                0,
                b"allocation",
                512,
                0,
                b"counter",
                generation * 2 + rank,
                b"event",
                False,
            )
            for rank in range(2)
        )
        return cuda_ipc.CudaIpcTensorTransportProxy(
            data=self.feature.view(torch.uint8),
            info_data=self.feature,
            pool_ipc_handle=handles[0],
            pool_byte_offset=256,
            ready_byte_offset=0,
            ack_byte_offset=4,
            generation=generation,
            total_consumer_count=2,
            use_pool_handle_cache=cache,
            pool_id="native-pool",
            pool_ipc_handles=handles,
        )

    def test_exports_use_distinct_zero_copy_storage_owners(self):
        """Exports must not stack their native ownership on the persistent pool storage."""
        storage = self.backing.untyped_storage()
        observed = []

        def share(exported):
            observed.append((exported._cdata, exported.data_ptr()))
            return (
                0,
                b"allocation",
                512,
                0,
                b"counter",
                len(observed),
                b"event",
                False,
            )

        with patch.object(torch.UntypedStorage, "_share_cuda_", share):
            handles = cuda_ipc._export_pool_storage(storage, 2)
        self.assertEqual(len(handles), 2)
        for identity, address in observed:
            self.assertNotEqual(identity, storage._cdata)
            self.assertEqual(address, storage.data_ptr())

    def test_shutdown_serializes_with_export_and_rejects_late_publishers(self):
        """Closing must finish an active export and reject later executor callbacks."""
        pool = object.__new__(cuda_ipc.MmItemMemoryPool)
        pool._pool_id = "native-pool"
        pool.device_id = 0
        pool._export_lock = threading.Lock()
        pool._closed = False
        pool._exports = {}
        pool._cancel_states = {}
        pool.consumer_count = 2
        pool.device_id = 0
        pool.memory_pool = self.backing
        pool._pool = Mock()
        lease = memory_pool.PoolLease(256, 512, 16, 0, 1, 0, 4)
        pool._pool._lock = threading.Lock()
        pool._pool.control_words_per_slot = 3
        pool._pool._occupied = {0: lease}
        pool._pool.base_address = self.backing.data_ptr()
        pool._pool.copy_tensor.return_value = (lease, self.backing[256:272])
        entered = threading.Event()
        finish = threading.Event()
        close_entered = threading.Event()
        close_finished = threading.Event()
        result = []
        errors = []
        handles = self._proxy().proxy_state["ipc_extra"]["pool_handles"]

        def export(*args):
            entered.set()
            if not finish.wait(5):
                raise TimeoutError("export was not released")
            return handles

        def publish():
            try:
                result.append(
                    pool.wrap_tensor(self.feature, use_pool_handle_cache=True)
                )
            except Exception as error:
                errors.append(error)

        def close():
            close_entered.set()
            try:
                pool.shutdown()
            except Exception as error:
                errors.append(error)
            finally:
                close_finished.set()

        publisher = threading.Thread(target=publish)
        closer = threading.Thread(target=close)
        with patch.object(cuda_ipc, "_export_pool_storage", side_effect=export):
            try:
                publisher.start()
                self.assertTrue(entered.wait(5))
                closer.start()
                self.assertTrue(close_entered.wait(5))
                self.assertFalse(close_finished.wait(0.1))
            finally:
                finish.set()
                publisher.join(5)
                closer.join(5)
        self.assertFalse(publisher.is_alive())
        self.assertFalse(closer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(result), 1)
        self.assertTrue(pool.owns_proxy(result[0]))
        self.assertTrue(pool._closed)
        with self.assertRaisesRegex(RuntimeError, "closed CUDA IPC"):
            pool.wrap_tensor(self.feature, use_pool_handle_cache=True)
        pool._pool.copy_tensor.assert_called_once()
        pool.shutdown()
        pool._pool.shutdown.assert_called_once()
        with patch.object(cuda_ipc, "stream_wait_value32"):
            copied = pool.copy_proxy_to_cpu(result[0])
        self.assertTrue(torch.equal(copied, self.feature))
        pool.cancel_proxy(result[0])
        self.assertEqual(self.released, [2, 3])
        pool._pool.cancel_lease.assert_called_once()

    def test_cache_hit_and_serialized_alias_consume_each_export_once(self):
        """Fresh cache-hit reservations retire once, even with duplicate serialized aliases."""
        first = self._proxy()
        first.reconstruct_on_target_device(0, consumer_rank=0)
        next_proxy = self._proxy(generation=2)
        alias = pickle.loads(pickle.dumps(next_proxy))
        next_proxy.reconstruct_on_target_device(0, consumer_rank=0)
        alias.acknowledge_consumption(consumer_rank=0)
        self.assertEqual(self.opened, [2])
        self.assertEqual(self.released, [4])
        self.assertEqual(len(cuda_ipc._pool_imported_generations["native-pool"]), 1)

    def test_post_import_view_failure_retries_without_reopening(self):
        """A view construction failure must retain the already claimed native mapping."""
        proxy = self._proxy(cache=False)
        alias = pickle.loads(pickle.dumps(proxy))
        proxy.proxy_state["ipc_extra"]["stride"] = (1, 1)
        with self.assertRaises(RuntimeError):
            proxy._open_pool_slice(0, consumer_rank=0)
        result = alias.reconstruct_on_target_device(0, consumer_rank=0)
        self.assertTrue(torch.equal(result, self.feature))
        self.assertEqual(self.opened, [2])
        self.assertEqual(self.released, [])

    def test_new_import_generation_blocks_old_alias_before_new_ack(self):
        """An older alias cannot consume a reservation after a newer import is observed."""
        old = self._proxy()
        newer = self._proxy(generation=2)
        newer._open_pool_slice(0, consumer_rank=0)
        with self.assertRaisesRegex(RuntimeError, "acknowledged"):
            old.reconstruct_on_target_device(0, consumer_rank=0)
        old.acknowledge_consumption(consumer_rank=0)
        self.assertEqual(self.opened, [4])
        self.assertEqual(self.released, [])

    def test_inactive_export_release_survives_failed_ack_retry(self):
        """Inactive reservations retire once even when their GPU acknowledgement fails."""
        proxy = self._proxy()
        with patch.object(
            cuda_ipc,
            "stream_write_value32",
            side_effect=[None, RuntimeError("write failed")],
        ):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                proxy.reconstruct_on_target_device(0, consumer_count=2, consumer_rank=0)
        proxy.release_without_reconstruction(2, consumer_rank=0)
        self.assertEqual(self.opened, [2])
        self.assertEqual(self.released, [3])

    def test_cache_eviction_reopens_only_a_fresh_generation(self):
        """After mapping eviction, another transfer uses its own fresh reservation."""
        self._proxy().reconstruct_on_target_device(0, consumer_rank=0)
        cuda_ipc._pool_handle_cache_clear()
        self._proxy(generation=2).reconstruct_on_target_device(0, consumer_rank=0)
        self.assertEqual(self.opened, [2, 4])
        self.assertEqual(self.released, [])

    def test_failed_native_open_is_not_retried_or_compensated(self):
        """An uncertain native-open failure must not double-consume its reservation."""
        proxy = self._proxy()
        with patch.object(
            cuda_ipc,
            "_open_pooled_storage_uncached",
            side_effect=RuntimeError("open failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "open failed"):
                proxy._open_pool_slice(0, consumer_rank=0)
        with self.assertRaisesRegex(RuntimeError, "retired or failed"):
            proxy._open_pool_slice(0, consumer_rank=0)
        self.assertEqual(self.released, [])

    def test_producer_cancel_validates_then_releases_each_unused_export(self):
        """Undispatched cancellation must balance all reservations without opening them."""
        proxy = self._proxy()
        pool = object.__new__(cuda_ipc.MmItemMemoryPool)
        pool._pool_id = "native-pool"
        pool.device_id = 0
        pool._export_lock = threading.Lock()
        handles = proxy.proxy_state["ipc_extra"]["pool_handles"]
        pool._exports = {0: (1, handles, set())}
        pool._cancel_states = {}
        pool._pool = object.__new__(memory_pool.StreamOrderedMmFeaturePool)
        pool._pool._lock = threading.Lock()
        pool._pool.control_words_per_slot = 3
        pool._pool._occupied = {}
        with self.assertRaisesRegex(RuntimeError, "inactive"):
            pool.cancel_proxy(proxy)
        self.assertEqual(self.released, [])
        pool._pool._occupied[0] = memory_pool.PoolLease(256, 512, 16, 0, 1, 0, 4)
        with patch.object(pool._pool, "cancel_lease"):
            pool.cancel_proxy(proxy)
            pool.cancel_proxy(proxy)
        self.assertEqual(self.released, [2, 3])
        self.assertEqual(self.opened, [])

    def test_producer_cancel_cannot_write_old_generation_after_slot_reuse(self):
        """Duplicate or retried cancellation must not overwrite a later lease's words."""

        class QueuedStream:
            def __init__(self):
                self.operations = []
                self.position = 0

            def wait_stream(self, prior):
                end = len(prior.operations)
                self.operations.append(lambda: prior.run(end))

            def run(self, end=None):
                end = len(self.operations) if end is None else end
                while self.position < end:
                    operation = self.operations[self.position]
                    self.position += 1
                    operation()

        real_tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return real_tensor(*args, **kwargs)

        for partial in (False, True):
            with self.subTest(partial=partial):
                pool = object.__new__(cuda_ipc.MmItemMemoryPool)
                pool._pool_id = "native-pool"
                pool.device_id = 0
                pool._export_lock = threading.Lock()
                pool._cancel_states = {}
                pool._pool = object.__new__(memory_pool.StreamOrderedMmFeaturePool)
                inner = pool._pool
                inner._lock = threading.Lock()
                inner._available_ranges = [(256, 512)]
                inner._available_slots = [0]
                inner._slot_generations = [0]
                inner._occupied = {}
                inner.consumer_count = 2
                inner.control_words_per_slot = 3
                inner.device_id = 0
                inner.base_address = self.backing.data_ptr()
                inner.transport_name = "CUDA IPC"
                inner._control_words = self.backing[:12].view(torch.int32).reshape(1, 3)
                inner._control_words.zero_()
                first = inner._allocate_locked(16)
                inner._control_words[0, 0] = first.generation
                proxy = self._proxy()
                pool._exports = {
                    0: (1, proxy.proxy_state["ipc_extra"]["pool_handles"], set())
                }
                original, retry = QueuedStream(), QueuedStream()
                current = [original]

                def write(device, address, generation, transport_name):
                    index = (address - inner.base_address) // 4
                    if partial and current[0] is original and index == 2:
                        raise RuntimeError("write failed")
                    current[0].operations.append(
                        lambda: inner._control_words[0].__setitem__(index, generation)
                    )

                with (
                    patch.object(torch, "tensor", side_effect=cpu_tensor),
                    patch.object(
                        torch.cuda, "current_stream", side_effect=lambda *_: current[0]
                    ),
                    patch.object(
                        memory_pool, "stream_write_value32", side_effect=write
                    ),
                ):
                    if partial:
                        with self.assertRaisesRegex(RuntimeError, "write failed"):
                            pool.cancel_proxy(proxy)
                    else:
                        pool.cancel_proxy(proxy)
                    current[0] = retry
                    pool.cancel_proxy(proxy)
                    (retry if partial else original).run()
                    inner._recycle_ready_leases_locked()
                    self.assertEqual(pool.active_lease_count, 0)
                    reused = inner._allocate_locked(16)
                    self.assertEqual(reused.generation, 2)
                    inner._control_words.fill_(reused.generation)
                    original.run()
                    retry.run()
                    self.assertTrue(bool((inner._control_words == 2).all()))

    def test_legacy_single_consumer_export_remains_supported_once(self):
        """A single exported reservation remains usable, but cannot fund another lease."""
        handle = self._proxy().proxy_state["ipc_extra"]["pool_handles"][0]

        def legacy(generation):
            return cuda_ipc.CudaIpcTensorTransportProxy(
                data=self.feature.view(torch.uint8),
                info_data=self.feature,
                pool_ipc_handle=handle,
                pool_byte_offset=256,
                ready_byte_offset=0,
                ack_byte_offset=4,
                generation=generation,
                total_consumer_count=1,
                use_pool_handle_cache=True,
            )

        proxy = legacy(1)
        alias = pickle.loads(pickle.dumps(proxy))
        self.assertTrue(
            torch.equal(proxy.reconstruct_on_target_device(0), self.feature)
        )
        alias.acknowledge_consumption()
        with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
            legacy(2).reconstruct_on_target_device(0)
        self.assertEqual(self.opened, [2])
        self.assertEqual(self.released, [])


if __name__ == "__main__":
    unittest.main()
