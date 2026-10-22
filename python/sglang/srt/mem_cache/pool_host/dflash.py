"""Bounded DFlash checkpoint windows, on GPU or offloaded through HiCache."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from sglang.srt.mem_cache import l2_transfer
from sglang.srt.mem_cache.l2_transfer import (
    L2Transfer,
    L2TransferEngine,
    TransferCompletion,
)
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_bytes
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    HostTensorAllocator,
    _cuda_host_unregister,
)
from sglang.srt.speculative.dflash_feature_tail import (
    CONTEXT_CHUNK_SIZE,
    DFlashContextLayout,
    DFlashContextWindow,
)

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class DFlashCheckpoint:
    """One checkpoint window; only the owning pool may release its slot."""

    start_pos: int
    num_tokens: int
    _indices: torch.Tensor
    revision: int = 0

    @property
    def end_pos(self) -> int:
        return self.start_pos + self.num_tokens


class DFlashCheckpointPool:
    """Context windows have independent residency from recurrent checkpoints.

    The cache owner calls backup/restore/release and polls transfer completions.
    Calls are serialized by that owner. It must keep each handle attached to the
    exact checkpoint/target identity; this pool never matches token prefixes.
    Active GPU tails and admission of restore destinations belong to the caller.
    """

    layer_num = 1

    def __init__(
        self,
        *,
        capacity: int,
        window_size: int,
        feature_size: int,
        dtype: torch.dtype,
        transfer_engine: L2TransferEngine | None = None,
        device: str | torch.device = "cpu",
        layout: DFlashContextLayout | None = None,
        pin_memory: bool = True,
        max_pending: int | None = None,
    ):
        if min(capacity, window_size, feature_size) <= 0 or not (
            dtype.is_floating_point or dtype == torch.uint8
        ):
            raise ValueError(
                "Context windows require positive sizes and a floating or byte dtype"
            )
        if max_pending is not None and max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self.device = (
            transfer_engine.device_to_host_stream.device
            if transfer_engine is not None
            else torch.device(device)
        )
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Feature-tail host transfers support CPU and CUDA devices")
        if (
            transfer_engine is not None
            and self.device.type == "cuda"
            and not pin_memory
        ):
            raise ValueError("GPU feature-tail transfers require pinned host memory")
        requested_bytes = capacity * window_size * feature_size * dtype.itemsize
        if transfer_engine is not None and requested_bytes > host_memory_budget_bytes(
            requested_bytes
        ):
            raise ValueError("Insufficient host memory for the feature-tail pool")
        self.window_size = window_size
        self.feature_size = feature_size
        self.dtype = dtype
        self.layout = layout
        self.revision = 0
        self._engine = transfer_engine
        self._max_pending = max_pending
        self._pin_memory = pin_memory and transfer_engine is not None
        self._slots = LogicalHostPool(capacity, page_size=1)
        shape = (capacity, window_size, feature_size)
        self._buffer = (
            ALLOC_MEMORY_FUNCS[self.device.type](
                shape,
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory,
                allocator=HostTensorAllocator(),
            )
            if transfer_engine is not None
            else torch.empty(shape, dtype=dtype, device=self.device)
        )
        if transfer_engine is None:
            self.device = self._buffer.device
        # Each descriptor's device pool is one owned snapshot, addressed as slot 0.
        self._device_slot = torch.zeros(1, dtype=torch.int64)
        # None reserves capacity for a checkpoint whose forward is still pending.
        self._live: dict[DFlashCheckpoint, torch.cuda.Event | bool | None] = {}
        self._pending: list[tuple[TransferCompletion, DFlashContextWindow]] = []
        # Logical release is deterministic across ranks. A new writer waits for
        # the previous occupant's final GPU use instead of delaying allocation.
        self._slot_fences: dict[int, torch.cuda.Event | bool] = {}
        logger.info(
            "DFLASH checkpoint pool (%s): %d windows, %.3f GiB per rank, "
            "pending transfer limit=%s",
            "host" if transfer_engine is not None else self.device,
            capacity,
            requested_bytes / 2**30,
            max_pending,
        )

    def poll(self) -> None:
        """Release completed copy sources without blocking the CPU."""
        self._pending = [
            copy for copy in self._pending if not copy[0].finish_event.query()
        ]

    def available_size(self) -> int:
        self.poll()
        return self._slots.available_size() if self._buffer is not None else 0

    def reserve(self, end_pos: int) -> DFlashCheckpoint | None:
        """Reserve a checkpoint's exact window before publishing its state."""
        if self._buffer is None:
            raise RuntimeError("Checkpoint pool is closed")
        if end_pos <= 0:
            raise ValueError("A cached feature tail must have a positive boundary")
        if self.available_size() == 0:
            return None
        indices = self._slots.alloc(1)
        if indices is None:
            return None
        start = max(0, end_pos - self.window_size)
        handle = DFlashCheckpoint(start, end_pos - start, indices)
        self._live[handle] = None
        return handle

    def contains(self, handle: DFlashCheckpoint) -> bool:
        return handle in self._live

    def is_ready(self, handle: DFlashCheckpoint | None) -> bool:
        """The backup is submitted; restore can order its H2D after that copy."""
        return self._live.get(handle) is not None

    def is_usable(self, handle: DFlashCheckpoint | None) -> bool:
        return self.is_ready(handle) and (
            handle.revision == self.revision
            or self.layout is None
            or self.layout.feature_size > 0
        )

    def invalidate_materialized(self) -> int:
        """Call at a drained draft-update boundary; raw taps remain compatible."""
        if self.layout is None or not self.layout.feature_size:
            raise ValueError("Draft updates require --enable-dflash-tap-cache")
        self.revision += 1
        return self.revision

    def refresh(self, handle: DFlashCheckpoint, tail: DFlashContextWindow) -> None:
        """Replace stale projected rows, preserving the checkpoint's logical slot."""
        event = self._get_last_use(handle)
        if event is not None:
            self._slot_fences[int(handle._indices[0])] = event
        self._live[handle] = None
        self.write_reserved(handle, tail)

    def _validate_tail(self, tail: DFlashContextWindow) -> None:
        if (
            not tail.parts
            or any(
                part.ndim != 2
                or part.shape[1] != self.feature_size
                or part.dtype != self.dtype
                or part.device != self.device
                or not part.is_contiguous()
                for part in tail.parts
            )
            or tail.num_tokens > self.window_size
            or tail.start_pos != max(0, tail.end_pos - self.window_size)
        ):
            raise ValueError(
                "Checkpoint context must match the pool's window, row layout, dtype and device"
            )

    def write_reserved(
        self, handle: DFlashCheckpoint, tail: DFlashContextWindow
    ) -> None:
        """Fill a reservation once; retain its immutable source until DMA finishes."""
        if self._get_last_use(handle) is not None:
            raise ValueError("Feature-tail reservation is already filled")
        self._validate_tail(tail)
        if (tail.start_pos, tail.end_pos) != (handle.start_pos, handle.end_pos):
            raise ValueError("Feature tail does not match the reserved checkpoint")
        handle.revision = tail.revision
        if self._engine is None:
            event = self._slot_fences.pop(int(handle._indices[0]), None)
            if self.device.type == "cuda" and event is not None:
                event.wait(torch.cuda.current_stream(self.device))
            self.backup_from_device_all_layer_physical(
                tail, handle._indices, None, None
            )
            if self.device.type == "cuda":
                for part in tail.parts:
                    part.record_stream(torch.cuda.current_stream(self.device))
            self._live[handle] = self._record_device_use()
            return
        self._wait_for_transfer_capacity()
        slot = int(handle._indices[0])
        previous_use = self._slot_fences.get(slot)
        if previous_use is not None:
            self._engine.device_to_host_stream.wait_event(previous_use)
        try:
            completion = self._engine.submit_device_to_host(
                [self._transfer(handle, tail)]
            )
        except Exception:
            # The HiCache engine drains its transfer stream on submission failure.
            self.release(handle)
            raise
        self._slot_fences.pop(slot, None)
        self._live[handle] = completion.finish_event
        self._pending.append((completion, tail))

    def backup(self, tail: DFlashContextWindow) -> DFlashCheckpoint | None:
        """Queue a copy after the calling stream; None means capacity is full."""
        if self._buffer is None:
            raise RuntimeError("Checkpoint pool is closed")
        self._validate_tail(tail)
        handle = self.reserve(tail.end_pos)
        if handle is not None:
            self.write_reserved(handle, tail)
        return handle

    def restore(self, handle: DFlashCheckpoint) -> DFlashContextWindow:
        """Return independent GPU context ordered before use on the caller's stream.

        Restoring needs no Mamba slot or transfer. A resident recurrent state can
        use this path too. Multiple consumers get independent mutable storage.
        """
        event = self._get_last_use(handle)
        if event is None:
            raise RuntimeError("Feature-tail capture has not completed")
        self._wait_for_transfer_capacity()
        tail = DFlashContextWindow(
            handle.start_pos,
            tuple(
                torch.empty(
                    (
                        min(CONTEXT_CHUNK_SIZE, handle.num_tokens - offset),
                        self.feature_size,
                    ),
                    dtype=self.dtype,
                    device=self.device,
                )
                for offset in range(0, handle.num_tokens, CONTEXT_CHUNK_SIZE)
            ),
            handle.revision,
        )
        if self._engine is None:
            if self.device.type == "cuda":
                event.wait(torch.cuda.current_stream(self.device))
            self.load_to_device_per_layer_physical(tail, handle._indices, None, 0, None)
            self._live[handle] = self._record_device_use()
            return tail
        self._engine.host_to_device_stream.wait_event(event)
        completion = self._engine.submit_host_to_device(
            [self._transfer(handle, tail)], transfer_layer_id_max=1
        )
        self._live[handle] = completion.finish_event
        self._pending.append((completion, tail))
        completion.finish_event.wait(l2_transfer.device_module.current_stream())
        return tail

    def _record_device_use(self):
        if self.device.type == "cpu":
            return True
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        return event

    def _wait_for_transfer_capacity(self) -> None:
        # Unfilled reservations own no transfer source. Bound actual GPU leases
        # at submission, retaining every reserved checkpoint. Only saturation waits.
        self.poll()
        while self._max_pending is not None and len(self._pending) >= self._max_pending:
            self._pending[0][0].finish_event.synchronize()
            self.poll()

    def release(self, handle: DFlashCheckpoint) -> None:
        """Release logical capacity now; fence the next write after pending copies."""
        event = self._get_last_use(handle)
        del self._live[handle]
        # Each load waits for the previous use. Its completion therefore fences
        # every reader and writer of the slot, on either storage tier.
        if event is not None:
            self._slot_fences[int(handle._indices[0])] = event
        self._slots.free(handle._indices)
        self.poll()

    def _get_last_use(self, handle: DFlashCheckpoint) -> torch.cuda.Event | bool | None:
        if handle not in self._live:
            raise ValueError("Feature tail is released or belongs to another pool")
        return self._live[handle]

    def clear(self) -> None:
        """Invalidate all bundles; in-flight transfers still fence slot reuse."""
        for handle in list(self._live):
            self.release(handle)

    def destroy(self) -> None:
        """Drain readers and writers before releasing storage; safe to call twice."""
        if self._buffer is None:
            return
        if self._engine is None and self.device.type == "cuda":
            for event in (*self._live.values(), *self._slot_fences.values()):
                if event is not None:
                    event.synchronize()
        for completion, _ in self._pending:
            completion.finish_event.synchronize()
        self.poll()
        self._live.clear()
        self._slot_fences.clear()
        if self._pin_memory:
            _cuda_host_unregister(self._buffer)
        self._buffer = None

    def _transfer(
        self, handle: DFlashCheckpoint, tail: DFlashContextWindow
    ) -> L2Transfer:
        return L2Transfer(self, tail, handle._indices, self._device_slot)

    def prepare_transfer_indices(self, host_indices, device_indices, io_backend):
        # Host slot IDs are CPU metadata; no device scalar read is needed.
        return host_indices, device_indices

    def _host_window(self, host_indices, tail):
        return self._buffer[int(host_indices[0]), : tail.num_tokens]

    def backup_from_device_all_layer_physical(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        host = self._host_window(host_indices, device_pool)
        offset = 0
        for part in device_pool.parts:
            host[offset : offset + part.shape[0]].copy_(part, non_blocking=True)
            offset += part.shape[0]

    def load_to_device_per_layer_physical(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft=False,
    ):
        if layer_id != 0:
            return
        host = self._host_window(host_indices, device_pool)
        offset = 0
        for part in device_pool.parts:
            part.copy_(host[offset : offset + part.shape[0]], non_blocking=True)
            offset += part.shape[0]
