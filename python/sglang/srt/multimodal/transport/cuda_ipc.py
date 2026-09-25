import logging
import threading
import uuid
from typing import Any, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.multimodal.transport.memory_pool import (
    CONTROL_WORD_BYTES,
    DEFAULT_MAX_INFLIGHT_SLICES,
    StreamOrderedMmFeaturePool,
    StreamOrderedPoolConsumerMixin,
    resolve_consumer_rank,
    stream_wait_value32,
    stream_write_value32,
)

logger = logging.getLogger(__name__)

MM_FEATURE_CACHE_SIZE = envs.SGLANG_MM_FEATURE_CACHE_MB.get() * 1024 * 1024

MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL = (
    envs.SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC.get()
)

# Defer producer-backed features until scheduling; each live reader still
# materializes an owned copy before acknowledging its lease.
DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY = (
    "_sglang_defer_cuda_ipc_feature_reconstruction"
)
BORROW_CUDA_IPC_FEATURE_KEY = "_sglang_borrow_cuda_ipc_feature"
CUDA_IPC_FEATURE_COPY_EVENT_KEY = "_sglang_cuda_ipc_feature_copy_event"
RETAINED_CUDA_IPC_FEATURE_PROXY_KEY = "_sglang_retained_cuda_ipc_feature_proxy"


def get_mm_feature_pool_size_per_worker(
    total_pool_size: int, tokenizer_worker_num: int
) -> int:
    """Split the CUDA IPC feature-pool budget without exceeding it.

    Each tokenizer worker owns a distinct CUDA allocation, even though all pools
    are created on ``base_gpu_id``.  Therefore a minimum per-worker allocation
    would make the aggregate HBM reservation larger than the configured budget.
    Keep the configured value as a hard per-node cap and leave at most
    ``tokenizer_worker_num - 1`` bytes unused when it is not evenly divisible.
    """
    if total_pool_size <= 0:
        raise ValueError("total_pool_size must be positive")
    if tokenizer_worker_num <= 0:
        raise ValueError("tokenizer_worker_num must be positive")

    return total_pool_size // tokenizer_worker_num


# Cache for pool-level IPC handles on the consumer side.
# Key: consumer device, allocation handle and storage offset.
_pool_storage_cache: dict = {}
_pool_cache_lock = threading.Lock()

# One high-water generation and rank mask per control slot, independent of
# mapping eviction: a serialized alias can outlive its original mapping.
_pool_acknowledged_generations: dict[Any, dict[int, tuple[int, int]]] = {}
_pool_consumer_lock = threading.RLock()
_pool_imported_generations: dict[Any, dict[int, tuple[int, dict]]] = {}
_IMPORT_FAILED = object()


def _release_ipc_export(handle) -> None:
    torch.UntypedStorage._release_ipc_counter_cuda(handle[4], handle[5])


def _export_pool_storage(storage, consumer_count: int) -> tuple:
    handles = []
    try:
        for _ in range(consumer_count):
            # A distinct StorageImpl retains the pool without stacking export
            # contexts on the pool's persistent DataPtr.
            exported = storage[:]
            handles.append(exported._share_cuda_())
            del exported
    except BaseException:
        for handle in handles:
            _release_ipc_export(handle)
        torch.cuda.ipc_collect()
        raise
    torch.cuda.ipc_collect()
    return tuple(handles)


def _normalize_pool_cache_key(pool_handle, device_index: int) -> tuple[Any, ...]:
    return (device_index, pool_handle[1], pool_handle[3])


def _open_pooled_storage_uncached(pool_handle):
    return torch.UntypedStorage._new_shared_cuda(*pool_handle)


def _pool_handle_cache_clear():
    with _pool_cache_lock:
        _pool_storage_cache.clear()


class MmItemMemoryPool:
    def __init__(
        self,
        memory_size: int,
        recycle_interval: float,
        base_gpu_id: int,
        consumer_count: int,
        max_inflight_slices: int = DEFAULT_MAX_INFLIGHT_SLICES,
    ):
        self.device_id = base_gpu_id
        self.consumer_count = consumer_count
        self.memory_pool = torch.empty(
            memory_size, dtype=torch.uint8, device=f"cuda:{base_gpu_id}"
        ).contiguous()
        self._pool = StreamOrderedMmFeaturePool(
            memory_size=memory_size,
            byte_tensor=self.memory_pool,
            base_address=self.memory_pool.data_ptr(),
            device_id=base_gpu_id,
            consumer_count=consumer_count,
            recycle_interval=recycle_interval,
            transport_name="CUDA IPC",
            max_inflight_slices=max_inflight_slices,
        )
        self._pool_id = uuid.uuid4().hex
        self._export_lock = threading.Lock()
        self._closed = False
        self._exports = {}
        self._cancel_states = {}
        self._pool_full_warned = False

        logger.debug(
            f"[MmItemMemoryPool] init: memory_size={memory_size}, "
            f"recycle_interval={recycle_interval}s"
        )

    def shutdown(self):
        with self._export_lock:
            if self._closed:
                return
            self._closed = True
            self._pool.shutdown()
            with torch.cuda.device(self.device_id):
                torch.cuda.ipc_collect()

    def owns_proxy(self, proxy: "CudaIpcTensorTransportProxy") -> bool:
        return proxy.proxy_state["ipc_extra"]["pool_id"] == self._pool_id

    @property
    def active_lease_count(self) -> int:
        return self._pool.active_lease_count

    def wrap_tensor(
        self, tensor: torch.Tensor, *, use_pool_handle_cache: bool
    ) -> Optional["CudaIpcTensorTransportProxy"]:
        with self._export_lock:
            if self._closed:
                raise RuntimeError("Cannot export from a closed CUDA IPC pool")
            return self._wrap_tensor(
                tensor, use_pool_handle_cache=use_pool_handle_cache
            )

    def _wrap_tensor(
        self, tensor: torch.Tensor, *, use_pool_handle_cache: bool
    ) -> Optional["CudaIpcTensorTransportProxy"]:
        lease, destination = self._pool.copy_tensor(tensor)
        if lease is None:
            nbytes = tensor.numel() * tensor.element_size()
            self._warn_pool_full_once(nbytes)
            return None

        try:
            handles = _export_pool_storage(
                self.memory_pool.untyped_storage(), self.consumer_count
            )
        except BaseException:
            self._pool.cancel_lease(
                ready_byte_offset=lease.ready_byte_offset,
                ack_byte_offset=lease.ack_byte_offset,
                generation=lease.generation,
            )
            raise
        self._exports[lease.ready_byte_offset] = (lease.generation, handles, set())
        self._cancel_states.pop(lease.ready_byte_offset, None)
        return CudaIpcTensorTransportProxy(
            data=destination,
            info_data=tensor,
            pool_ipc_handle=handles[0],
            pool_byte_offset=lease.start,
            ready_byte_offset=lease.ready_byte_offset,
            ack_byte_offset=lease.ack_byte_offset,
            generation=lease.generation,
            total_consumer_count=self.consumer_count,
            use_pool_handle_cache=use_pool_handle_cache,
            pool_id=self._pool_id,
            pool_ipc_handles=handles,
        )

    def cancel_proxy(self, proxy: "CudaIpcTensorTransportProxy") -> None:
        """Return a published slice when its request was never dispatched."""
        if not self.owns_proxy(proxy):
            raise RuntimeError("CUDA IPC proxy does not belong to this pool")
        with self._export_lock:
            entry = self._exports.get(proxy.ready_byte_offset)
            if entry is None or entry[0] != proxy.generation:
                raise RuntimeError("Cannot cancel inactive CUDA IPC native exports")
            _, handles, released = entry
            with self._pool._lock:
                stride = self._pool.control_words_per_slot * CONTROL_WORD_BYTES
                lease = self._pool._occupied.get(proxy.ready_byte_offset // stride)
                if (
                    lease is None
                    or lease.generation != proxy.generation
                    or lease.ready_byte_offset != proxy.ready_byte_offset
                    or lease.ack_byte_offset != proxy.ack_byte_offset
                ):
                    raise RuntimeError("Cannot cancel inactive CUDA IPC pool lease")
                previous = self._cancel_states.get(proxy.ready_byte_offset)
                if (
                    previous is not None
                    and previous[0] == proxy.generation
                    and previous[2]
                ):
                    return
                for rank, handle in enumerate(handles):
                    if rank not in released:
                        _release_ipc_export(handle)
                        released.add(rank)
            with torch.cuda.device(self.device_id):
                stream = torch.cuda.current_stream(self.device_id)
                if previous is not None and previous[0] == proxy.generation:
                    if stream != previous[1]:
                        stream.wait_stream(previous[1])
                self._cancel_states[proxy.ready_byte_offset] = (
                    proxy.generation,
                    stream,
                    False,
                )
                self._pool.cancel_lease(
                    ready_byte_offset=proxy.ready_byte_offset,
                    ack_byte_offset=proxy.ack_byte_offset,
                    generation=proxy.generation,
                )
                self._cancel_states[proxy.ready_byte_offset] = (
                    proxy.generation,
                    stream,
                    True,
                )
            torch.cuda.ipc_collect()

    def copy_proxy_to_cpu(self, proxy: "CudaIpcTensorTransportProxy") -> torch.Tensor:
        """Snapshot an undispatched lease without acknowledging its consumers."""
        ipc_extra = proxy.proxy_state["ipc_extra"]
        if not self.owns_proxy(proxy):
            raise RuntimeError("CUDA IPC proxy does not belong to this pool")
        pool = self._pool
        slot_stride = pool.control_words_per_slot * CONTROL_WORD_BYTES
        with pool._lock:
            lease = pool._occupied.get(proxy.ready_byte_offset // slot_stride)
            if (
                lease is None
                or lease.generation != proxy.generation
                or lease.ready_byte_offset != proxy.ready_byte_offset
                or lease.ack_byte_offset != proxy.ack_byte_offset
                or lease.start != ipc_extra["pool_byte_offset"]
                or lease.nbytes != ipc_extra["nbytes"]
            ):
                raise RuntimeError("Cannot copy inactive CUDA IPC pool lease")
            with torch.cuda.device(self.device_id):
                stream_wait_value32(
                    self.device_id,
                    pool.base_address + lease.ready_byte_offset,
                    lease.generation,
                    "CUDA IPC",
                )
                return (
                    self.memory_pool[lease.start : lease.start + lease.nbytes]
                    .view(ipc_extra["recons_dtype"])
                    .reshape(ipc_extra["recons_shape"])
                    .to(device="cpu", copy=True)
                )

    def _warn_pool_full_once(self, nbytes: int):
        if self._pool_full_warned:
            return
        self._pool_full_warned = True
        pool_mb = (
            self.memory_pool.numel() * self.memory_pool.element_size() / (1024 * 1024)
        )
        need_mb = nbytes / (1024 * 1024)
        logger.warning(
            "MmItemMemoryPool has no free chunk large enough for a %.2f MiB tensor "
            "(pool size: %.2f MiB); falling back to non-IPC transport. "
            "Consider increasing SGLANG_MM_FEATURE_CACHE_MB.",
            need_mb,
            pool_mb,
        )


class CudaIpcTensorTransportProxy(StreamOrderedPoolConsumerMixin):
    """Serializable view of one tensor stored in a CUDA IPC memory pool.

    The producer-ready word and one acknowledgement word per consumer live in
    the same CUDA allocation as the tensor. CUDA stream memory operations order
    the producer copy, consumer copy, and pool reuse without CPU shared memory
    or device-wide synchronization.
    """

    def __init__(
        self,
        data: torch.Tensor,
        info_data: torch.Tensor,
        pool_ipc_handle,
        pool_byte_offset: int,
        ready_byte_offset: int,
        ack_byte_offset: int,
        generation: int,
        total_consumer_count: int,
        use_pool_handle_cache: bool,
        *,
        pool_id=None,
        pool_ipc_handles=None,
    ):
        if (not isinstance(data, torch.Tensor)) or (
            not isinstance(info_data, torch.Tensor)
        ):
            raise TypeError(
                f"Input 'data' must be a torch.Tensor, but got {type(data)}"
            )

        self._init_stream_ordered_consumer(
            ready_byte_offset=ready_byte_offset,
            ack_byte_offset=ack_byte_offset,
            generation=generation,
            total_consumer_count=total_consumer_count,
            transport_name="CUDA IPC",
        )

        self.proxy_state = {
            "ipc_extra": {
                "pool_handle": pool_ipc_handle,
                "pool_handles": (
                    (pool_ipc_handle,)
                    if pool_ipc_handles is None and total_consumer_count == 1
                    else pool_ipc_handles
                ),
                "single_export": pool_ipc_handles is None and total_consumer_count == 1,
                "pool_id": tuple(pool_ipc_handle) if pool_id is None else pool_id,
                "pool_byte_offset": pool_byte_offset,
                "shape": data.shape,
                "dtype": data.dtype,
                "stride": data.stride(),
                "storage_offset": 0,
                "nbytes": data.numel() * data.element_size(),
                "recons_shape": info_data.shape,
                "recons_dtype": info_data.dtype,
                "use_pool_handle_cache": use_pool_handle_cache,
            },
            "tensor_data": None,
        }
        self.reconstruct_tensor = None
        self._reconstruct_device_idx = None
        self._reconstruct_stream = None
        self._acknowledge_ranks = None
        # Keep uncached mappings alive until the work enqueued on the consumer
        # stream has completed.
        self._pool_storage = None
        self._pool_storage_device_id = None
        self._pool_storage_stream = None
        self._borrowed_storage = None
        self._borrowed_base_address = None
        self._borrowed_device_id = None

    def _acknowledged_generation(self) -> tuple[int, int]:
        pool_key = self.proxy_state["ipc_extra"]["pool_id"]
        return _pool_acknowledged_generations.get(pool_key, {}).get(
            self.ready_byte_offset, (0, 0)
        )

    def _native_slot(self, *, create: bool = False):
        extra = self.proxy_state["ipc_extra"]
        slots = _pool_imported_generations.get(extra["pool_id"], {})
        entry = slots.get(self.ready_byte_offset)
        if (
            create
            and extra["single_export"]
            and slots
            and (entry is None or entry[0] != self.generation)
        ):
            raise RuntimeError(
                "A single CUDA IPC export cannot be reused for another lease"
            )
        if create and (entry is None or entry[0] < self.generation):
            slots = _pool_imported_generations.setdefault(extra["pool_id"], {})
            entry = (self.generation, {})
            slots[self.ready_byte_offset] = entry
        return entry

    def _retire_unused_export(self, rank: int) -> None:
        handles = self.proxy_state["ipc_extra"]["pool_handles"]
        if handles is None:
            return
        entry = self._native_slot(create=True)
        if entry[0] != self.generation:
            return
        imports = entry[1]
        if rank not in imports:
            imports[rank] = _IMPORT_FAILED
            _release_ipc_export(handles[rank])
            imports[rank] = None
        elif imports[rank] is _IMPORT_FAILED:
            raise RuntimeError("CUDA IPC import failed with uncertain native ownership")

    def _check_read(self, consumer_rank: Optional[int] = None) -> None:
        rank = resolve_consumer_rank(
            self.total_consumer_count, consumer_rank, self.transport_name
        )
        generation, rank_mask = self._acknowledged_generation()
        native = self._native_slot()
        if native is not None and native[0] > self.generation:
            raise RuntimeError("Cannot read an acknowledged CUDA IPC pool lease")
        if generation > self.generation or (
            generation == self.generation and rank_mask & (1 << rank)
        ):
            raise RuntimeError(
                "Cannot read an acknowledged CUDA IPC pool lease "
                f"(offset={self.ready_byte_offset}, generation={self.generation})"
            )

    def _set_acknowledge_ranks(
        self, acknowledge_ranks, consumer_count: int, consumer_rank: Optional[int]
    ) -> None:
        if acknowledge_ranks is None:
            return
        ranks = tuple(acknowledge_ranks)
        own_rank = resolve_consumer_rank(
            self.total_consumer_count, consumer_rank, self.transport_name
        )
        if (
            consumer_count != 1
            or own_rank not in ranks
            or len(set(ranks)) != len(ranks)
            or any(rank < 0 or rank >= self.total_consumer_count for rank in ranks)
        ):
            raise ValueError(
                "Explicit CUDA IPC acknowledgement ranks must contain the reader "
                "and unique valid ranks, with consumer_count=1"
            )
        if self._acknowledge_ranks is not None and self._acknowledge_ranks != ranks:
            raise ValueError("Cannot change CUDA IPC acknowledgement ranks")
        # Keep the selected ranks through failures so generic request cleanup
        # cannot acknowledge a live peer that this reader does not own.
        self._acknowledge_ranks = ranks

    def _pending_consumer_ranks(self, consumer_count, consumer_rank):
        ranks = (
            self._acknowledge_ranks
            if self._acknowledge_ranks is not None
            else self._consumer_ranks(consumer_count, consumer_rank)
        )
        generation, rank_mask = self._acknowledged_generation()
        native = self._native_slot()
        if generation > self.generation or (
            native is not None and native[0] > self.generation
        ):
            return ()
        return tuple(
            rank
            for rank in ranks
            if generation < self.generation or not rank_mask & (1 << rank)
        )

    def _acknowledge_on_stream(
        self,
        base_address: int,
        device_id: int,
        consumer_count: int,
        consumer_rank: Optional[int] = None,
    ) -> None:
        with _pool_consumer_lock:
            if self._consumer_acknowledged:
                return
            pool_key = self.proxy_state["ipc_extra"]["pool_id"]
            for rank in self._pending_consumer_ranks(consumer_count, consumer_rank):
                self._retire_unused_export(rank)
                native = self._native_slot()
                imported = None if native is None else native[1].get(rank)
                if imported is not None:
                    stream = torch.cuda.current_stream(device_id)
                    if stream != imported[2]:
                        stream.wait_stream(imported[2])
                stream_write_value32(
                    device_id,
                    base_address + self.ack_byte_offset + rank * CONTROL_WORD_BYTES,
                    self.generation,
                    self.transport_name,
                )
                generation, rank_mask = self._acknowledged_generation()
                if generation != self.generation:
                    rank_mask = 0
                _pool_acknowledged_generations.setdefault(pool_key, {})[
                    self.ready_byte_offset
                ] = (self.generation, rank_mask | (1 << rank))
                if native is not None:
                    native[1][rank] = None
            self._consumer_acknowledged = True

    def _open_pool_slice(self, rebuild_device_idx: int, consumer_rank=None):
        ipc_extra = self.proxy_state["ipc_extra"]
        handles = ipc_extra["pool_handles"]
        if handles is None or len(handles) != self.total_consumer_count:
            raise RuntimeError("CUDA IPC requires one native export per consumer")
        rank = resolve_consumer_rank(
            self.total_consumer_count, consumer_rank, self.transport_name
        )
        with _pool_consumer_lock, torch.cuda.device(rebuild_device_idx):
            self._check_read(rank)
            entry = self._native_slot(create=True)
            imports = entry[1]
            stream = torch.cuda.current_stream(rebuild_device_idx)
            if rank in imports:
                imported = imports[rank]
                if imported is None or imported is _IMPORT_FAILED:
                    raise RuntimeError(
                        "CUDA IPC native export was already retired or failed"
                    )
                storage, mapped_device, prior_stream = imported
                if mapped_device != rebuild_device_idx:
                    raise RuntimeError(
                        "Cannot reopen a CUDA IPC export on another device"
                    )
                if stream != prior_stream:
                    stream.wait_stream(prior_stream)
            else:
                handle = handles[rank]
                cache_key = _normalize_pool_cache_key(handle, rebuild_device_idx)
                storage = (
                    _pool_storage_cache.get(cache_key)
                    if ipc_extra["use_pool_handle_cache"]
                    else None
                )
                if storage is None:
                    # A failed native open can already have consumed its counter;
                    # never retry that reservation or guess a compensating decrement.
                    imports[rank] = _IMPORT_FAILED
                    redirected_handle = (rebuild_device_idx,) + tuple(handle)[1:]
                    storage = _open_pooled_storage_uncached(redirected_handle)
                    imports[rank] = (storage, rebuild_device_idx, stream)
                    if ipc_extra["use_pool_handle_cache"]:
                        with _pool_cache_lock:
                            _pool_storage_cache[cache_key] = storage
                else:
                    imports[rank] = _IMPORT_FAILED
                    _release_ipc_export(handle)
            imports[rank] = (storage, rebuild_device_idx, stream)
            slice_storage = storage[
                ipc_extra["pool_byte_offset"] : ipc_extra["pool_byte_offset"]
                + ipc_extra["nbytes"]
            ]
            slice_tensor = torch.empty(
                0, dtype=ipc_extra["dtype"], device=f"cuda:{rebuild_device_idx}"
            ).set_(
                slice_storage,
                storage_offset=ipc_extra["storage_offset"],
                size=ipc_extra["shape"],
                stride=ipc_extra["stride"],
            )
            return slice_tensor, storage

    def _retain_storage_until_stream_completes(self, storage, device_id: int) -> None:
        if self.proxy_state["ipc_extra"]["use_pool_handle_cache"]:
            # The process-wide cache owns the mapping after this proxy is
            # replaced by its reconstructed tensor.
            self._pool_storage = storage
        else:
            # An uncached mapping is owned only by this proxy. The caller
            # replaces the proxy immediately, so finish the current stream
            # before allowing the mapping to close.
            stream = torch.cuda.current_stream(device_id)
            if self._pool_storage_stream is not None:
                stream.wait_stream(self._pool_storage_stream)
            stream.synchronize()
            self._pool_storage = None
            self._pool_storage_device_id = None
            self._pool_storage_stream = None

    def acknowledge_consumption(
        self,
        consumer_count: int = 1,
        consumer_rank: Optional[int] = None,
        *,
        acknowledge_ranks: Optional[tuple[int, ...]] = None,
    ) -> None:
        """Stream-order pool release when a cache hit needs no tensor copy."""
        with _pool_consumer_lock:
            self._set_acknowledge_ranks(
                acknowledge_ranks, consumer_count, consumer_rank
            )
            if self._consumer_acknowledged:
                return
            if not self._pending_consumer_ranks(consumer_count, consumer_rank):
                self._consumer_acknowledged = True
                if self._pool_storage is not None:
                    self._retain_storage_until_stream_completes(
                        self._pool_storage, self._pool_storage_device_id
                    )
                return
            device_id = (
                self._pool_storage_device_id
                if self._pool_storage is not None
                else torch.cuda.current_device()
            )
            with torch.cuda.device(device_id):
                stream = torch.cuda.current_stream(device_id)
                if self._pool_storage is None:
                    _, storage = self._open_pool_slice(device_id, consumer_rank)
                    self._pool_storage = storage
                    self._pool_storage_device_id = device_id
                    self._pool_storage_stream = stream
                else:
                    storage = self._pool_storage
                    if stream != self._pool_storage_stream:
                        stream.wait_stream(self._pool_storage_stream)
                base_address = storage.data_ptr()
                self._wait_until_ready(base_address, device_id)
                self._acknowledge_on_stream(
                    base_address, device_id, consumer_count, consumer_rank
                )
            self._retain_storage_until_stream_completes(storage, device_id)

    def borrow_on_target_device(
        self, rebuild_device_idx: int
    ) -> Optional[torch.Tensor]:
        """Return a zero-copy view whose lease remains owned by this proxy."""
        ipc_extra = self.proxy_state["ipc_extra"]
        if not ipc_extra["use_pool_handle_cache"] or self._consumer_acknowledged:
            return None

        with _pool_consumer_lock, torch.cuda.device(rebuild_device_idx):
            self._check_read()
            slice_tensor, storage = self._open_pool_slice(rebuild_device_idx)
            base_address = storage.data_ptr()
            self._wait_until_ready(base_address, rebuild_device_idx)
            borrowed = slice_tensor.view(ipc_extra["recons_dtype"]).reshape(
                ipc_extra["recons_shape"]
            )

        self._borrowed_storage = storage
        self._borrowed_base_address = base_address
        self._borrowed_device_id = rebuild_device_idx
        return borrowed

    def release_borrowed_on_current_stream(
        self, consumer_count: int = 1, consumer_rank: Optional[int] = None
    ) -> None:
        """Release a borrowed view after all current-stream reads are enqueued."""
        storage = self._borrowed_storage
        if storage is None:
            return
        device_id = self._borrowed_device_id
        with torch.cuda.device(device_id):
            self._acknowledge_on_stream(
                self._borrowed_base_address,
                device_id,
                consumer_count,
                consumer_rank,
            )
        self._retain_storage_until_stream_completes(storage, device_id)
        self._borrowed_storage = None
        self._borrowed_base_address = None
        self._borrowed_device_id = None

    def release_without_reconstruction(
        self,
        consumer_count: int = 1,
        consumer_rank: Optional[int] = None,
        *,
        acknowledge_ranks: Optional[tuple[int, ...]] = None,
    ) -> None:
        """Release a pool slice when its request abandons this proxy."""
        with _pool_consumer_lock:
            self._set_acknowledge_ranks(
                acknowledge_ranks, consumer_count, consumer_rank
            )
            if self._borrowed_storage is not None:
                self.release_borrowed_on_current_stream(consumer_count, consumer_rank)
            else:
                self.acknowledge_consumption(consumer_count, consumer_rank)

    def reconstruct_on_target_device(
        self,
        rebuild_device_idx,
        consumer_count: int = 1,
        consumer_rank: Optional[int] = None,
        *,
        acknowledge_ranks: Optional[tuple[int, ...]] = None,
    ):
        rebuild_device = torch.device(f"cuda:{rebuild_device_idx}")
        with _pool_consumer_lock, torch.cuda.device(rebuild_device):
            self._set_acknowledge_ranks(
                acknowledge_ranks, consumer_count, consumer_rank
            )
            if (
                self.reconstruct_tensor is not None
                and self._reconstruct_device_idx == rebuild_device_idx
            ):
                current_stream = torch.cuda.current_stream(rebuild_device_idx)
                if current_stream != self._reconstruct_stream:
                    current_stream.wait_stream(self._reconstruct_stream)
                if self._consumer_acknowledged:
                    return self.reconstruct_tensor
                storage = self._pool_storage
            else:
                self._check_read(consumer_rank)
                ipc_extra = self.proxy_state["ipc_extra"]
                slice_tensor, storage = self._open_pool_slice(
                    rebuild_device_idx, consumer_rank
                )
                self._pool_storage = storage
                self._pool_storage_device_id = rebuild_device_idx
                self._pool_storage_stream = torch.cuda.current_stream(
                    rebuild_device_idx
                )
                self._wait_until_ready(storage.data_ptr(), rebuild_device_idx)
                reconstructed_tensor = torch.empty(
                    ipc_extra["recons_shape"],
                    dtype=ipc_extra["recons_dtype"],
                    device=rebuild_device,
                ).contiguous()
                reconstructed_tensor.view(torch.uint8).reshape(-1).copy_(slice_tensor)
                # Keep the owned copy and mapping if a later acknowledgement
                # fails; retrying must not read a partially released lease.
                self.reconstruct_tensor = reconstructed_tensor
                self._reconstruct_device_idx = rebuild_device_idx
                self._reconstruct_stream = torch.cuda.current_stream(rebuild_device_idx)
                self._pool_storage = storage
            self._acknowledge_on_stream(
                storage.data_ptr(), rebuild_device_idx, consumer_count, consumer_rank
            )
            self._retain_storage_until_stream_completes(storage, rebuild_device_idx)
            return self.reconstruct_tensor
