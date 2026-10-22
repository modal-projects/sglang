"""Owned DFlash checkpoint windows: materialized KV and optional target taps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from sglang.kernels.ops.kvcache.dflash_features import capture_dflash_features

CONTEXT_CHUNK_SIZE = 256

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.pool_host.dflash import (
        DFlashCheckpointPool,
        DFlashCheckpoint,
    )


@dataclass(frozen=True)
class DFlashContextLayout:
    """Byte layout of one token's KV and optional unprojected target features.

    Reinterpreting bytes preserves different target and KV dtypes exactly.
    The window lifecycle treats rows as opaque data, independent of projection.
    """

    num_layers: int
    kv_heads: int
    head_dim: int
    kv_dtype: torch.dtype
    feature_size: int = 0
    feature_dtype: torch.dtype = torch.bfloat16

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def kv_bytes(self) -> int:
        return self.num_layers * 2 * self.kv_width * self.kv_dtype.itemsize

    @property
    def row_bytes(self) -> int:
        return self.kv_bytes + self.feature_size * self.feature_dtype.itemsize

    def kv(self, rows: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        width = self.kv_width * self.kv_dtype.itemsize
        start = layer * 2 * width
        shape = (rows.shape[0], self.kv_heads, self.head_dim)
        return (
            rows[:, start : start + width].view(self.kv_dtype).view(shape),
            rows[:, start + width : start + 2 * width].view(self.kv_dtype).view(shape),
        )

    def features(self, rows: torch.Tensor) -> torch.Tensor:
        if not self.feature_size:
            raise RuntimeError("Draft reprojection requires --enable-dflash-tap-cache")
        return rows[:, self.kv_bytes :].view(self.feature_dtype)


@dataclass(frozen=True)
class DFlashContextWindow:
    """Context rows for [start_pos, end_pos) at one draft revision.

    Treat ``parts`` as read-only. Overlapping tails share their backing chunks.
    Appending may fill unused rows in a chunk, but never overwrites these views.
    The caller owns branch/target compatibility and stream ordering; this object
    leases feature rows, not a recurrent state or a cache entry.
    """

    start_pos: int
    parts: tuple[torch.Tensor, ...]
    revision: int = 0

    @property
    def num_tokens(self) -> int:
        return sum(part.shape[0] for part in self.parts)

    @property
    def end_pos(self) -> int:
        return self.start_pos + self.num_tokens

    def as_tensor(self) -> torch.Tensor:
        """Join shared chunks only when a consumer needs contiguous rows."""
        return self.parts[0] if len(self.parts) == 1 else torch.cat(self.parts)

    @classmethod
    def capture(
        cls,
        target_hidden: torch.Tensor,
        *,
        start_pos: int,
        end_pos: int,
        window_size: int,
        prefix_tail: DFlashContextWindow | None = None,
    ) -> DFlashContextWindow:
        """Own the window ending at a committed checkpoint inside this forward.

        ``target_hidden`` starts at ``start_pos`` and may include padding or
        uncommitted rows after ``end_pos``. A prefix tail must end at start_pos
        when old rows are needed. Missing coverage is an error, not a replay
        request. Capture an interior prefill checkpoint before reusing its
        forward's output buffer; the final live tail may no longer cover it.

        Only host position metadata is inspected. The copy runs on the current
        stream and must precede reuse of either source on another stream.
        """
        if (
            target_hidden.ndim != 2
            or target_hidden.shape[1] == 0
            or not (
                target_hidden.is_floating_point() or target_hidden.dtype == torch.uint8
            )
        ):
            raise ValueError("target_hidden must be rank-2 floating or byte rows")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0 <= start_pos <= end_pos <= start_pos + target_hidden.shape[0]:
            raise ValueError("end_pos must be within the supplied forward's positions")

        window_start = max(0, end_pos - window_size)
        parts = []
        if window_start < start_pos:
            if (
                prefix_tail is None
                or prefix_tail.end_pos != start_pos
                or prefix_tail.start_pos > window_start
            ):
                raise ValueError(
                    f"Missing target feature coverage for [{window_start}, {start_pos})"
                )
            previous = prefix_tail.parts[0]
            if (
                previous.shape[1] != target_hidden.shape[1]
                or previous.dtype != target_hidden.dtype
                or previous.device != target_hidden.device
            ):
                raise ValueError(
                    "prefix and new target features must match shape/dtype/device"
                )
            pos = prefix_tail.start_pos
            for part in prefix_tail.parts:
                if pos + part.shape[0] > window_start:
                    parts.append(part[max(0, window_start - pos) :])
                pos += part.shape[0]
        new = target_hidden[
            max(window_start - start_pos, 0) : end_pos - start_pos
        ].detach()
        # Copy new rows before graph-output reuse. Prefix rows are already owned
        # and immutable; keep their chunks without copying them again.
        parts.extend(
            new[offset : offset + CONTEXT_CHUNK_SIZE].clone()
            for offset in range(0, new.shape[0], CONTEXT_CHUNK_SIZE)
        )
        return cls(window_start, tuple(parts) if parts else (new.clone(),))


@dataclass
class _FeatureChunk:
    start_pos: int
    buffer: torch.Tensor
    used: int

    @property
    def end_pos(self) -> int:
        return self.start_pos + self.used


@dataclass(eq=False)
class _CheckpointSlot:
    index: int
    tail: DFlashContextWindow | None = None
    pending_end_pos: int | None = None
    destination: tuple[DFlashCheckpointPool, DFlashCheckpoint] | None = None

    def cancel_donation(self) -> None:
        if self.destination is not None:
            pool, handle = self.destination
            if pool.contains(handle):
                pool.release(handle)
            self.destination = None


class DFlashContextState:
    """A request's recent context rows and shared checkpoint windows.

    The scheduler consumes completed forwards in order. Only new rows enter
    append-only chunks; checkpoints retain views of the rows they need.
    A prefix hit without restored features has incomplete coverage until a
    full new window arrives. Such coverage never yields a checkpoint tail.
    """

    def __init__(self, window_size: int, revision: int = 0):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.revision = revision
        self.end_pos: int | None = None
        self._start_pos = 0
        self._chunks: list[_FeatureChunk] = []
        self._checkpoints: dict[int, _CheckpointSlot] = {}
        # [start, end) rows whose KV ids a tree insert replaced with canonical ids.
        self._canonical_remap: tuple[int, int] | None = None
        self.closed = False

    def _checkpoint_slot(self, index: int) -> _CheckpointSlot:
        if index not in (0, 1):
            raise ValueError("Invalid recurrent tracking position")
        if index not in self._checkpoints:
            self._checkpoints[index] = _CheckpointSlot(index)
        return self._checkpoints[index]

    def note_canonical_remap(self, start_pos: int, end_pos: int) -> None:
        if self._canonical_remap is not None:
            start_pos = min(start_pos, self._canonical_remap[0])
            end_pos = max(end_pos, self._canonical_remap[1])
        self._canonical_remap = (start_pos, end_pos)

    def take_canonical_remap(self) -> tuple[int, int] | None:
        remap, self._canonical_remap = self._canonical_remap, None
        return remap

    def invalidate_checkpoint(self, index: int) -> None:
        # Identity also rejects captures queued before this slot was replaced.
        slot = self._checkpoints.pop(index, None)
        if slot is not None:
            slot.tail = None

    def checkpoint(self, index: int, end_pos: int) -> DFlashContextWindow | None:
        slot = self._checkpoints.get(index)
        tail = slot.tail if slot is not None else None
        return tail if tail is not None and tail.end_pos == end_pos else None

    def has_checkpoint(self, index: int, end_pos: int) -> bool:
        slot = self._checkpoints.get(index)
        return slot is not None and (
            slot.pending_end_pos == end_pos
            or (slot.tail is not None and slot.tail.end_pos == end_pos)
        )

    def donate_checkpoint(
        self, index: int, end_pos: int, pool: DFlashCheckpointPool
    ) -> DFlashCheckpoint | None:
        """Move the completed or pending snapshot with its recurrent state."""
        if not self.has_checkpoint(index, end_pos):
            return None
        handle = pool.reserve(end_pos)
        if handle is None:
            return None
        slot = self._checkpoints.pop(index)
        handle.revision = slot.tail.revision if slot.tail is not None else self.revision
        if slot.tail is not None:
            pool.write_reserved(handle, slot.tail)
            slot.tail = None
        else:
            slot.destination = (pool, handle)
        return handle

    @classmethod
    def from_tail(
        cls, tail: DFlashContextWindow, window_size: int
    ) -> DFlashContextState:
        state = cls(window_size, tail.revision)
        if tail.start_pos != max(0, tail.end_pos - window_size):
            raise ValueError("Restored features must cover the complete window")
        state.end_pos = tail.end_pos
        state._start_pos = tail.start_pos
        pos = tail.start_pos
        for i, part in enumerate(tail.parts):
            if part.shape[0] > CONTEXT_CHUNK_SIZE:
                # Reprojection can return one contiguous window. Restore the
                # bounded chunks used during capture before it starts sliding.
                for chunk in part.split(CONTEXT_CHUNK_SIZE):
                    state._chunks.append(
                        _FeatureChunk(pos, chunk.clone(), chunk.shape[0])
                    )
                    pos += chunk.shape[0]
                continue
            if (
                i == len(tail.parts) - 1
                and part.untyped_storage().nbytes() > part.numel() * part.element_size()
            ):
                # Seal a partial shared chunk so its unused suffix cannot become
                # internal padding after this branch appends. HiCache restores
                # already own exact-sized chunks and need no copy here.
                part = part.clone()
            state._chunks.append(_FeatureChunk(pos, part, part.shape[0]))
            pos += part.shape[0]
        return state

    def replace_live_window(self, tail: DFlashContextWindow) -> None:
        """Refresh live KV without changing snapshots held by older checkpoints."""
        restored = self.from_tail(tail, self.window_size)
        self._chunks = restored._chunks
        self._start_pos = restored._start_pos
        self.end_pos = restored.end_pos
        self.revision = restored.revision

    def close(self) -> None:
        self.closed = True
        self._canonical_remap = None
        self._chunks.clear()
        for slot in self._checkpoints.values():
            slot.tail = None
        self._checkpoints.clear()

    def _append(
        self,
        features: torch.Tensor,
        start_pos: int,
        stream: torch.cuda.Stream | None = None,
        *,
        owned_chunk: bool = False,
    ) -> None:
        if (
            features.ndim != 2
            or features.shape[1] == 0
            or not (features.is_floating_point() or features.dtype == torch.uint8)
        ):
            raise ValueError("Feature rows must be rank-2 floating or byte rows")
        if features.shape[0] == 0:
            return
        end_pos = start_pos + features.shape[0]
        # A long prefill may supply just the windows needed at its checkpoints.
        if self.end_pos is None or start_pos > self.end_pos:
            self._start_pos = start_pos
        elif start_pos != self.end_pos:
            raise ValueError("Feature rows must arrive in forward order")
        if self._chunks:
            previous = self._chunks[-1].buffer
            if (
                features.shape[1] != previous.shape[1]
                or features.dtype != previous.dtype
                or features.device != previous.device
            ):
                raise ValueError("Feature rows must match the request's chunks")
            del previous
        features = features[-self.window_size :]
        start_pos = end_pos - features.shape[0]
        if self.end_pos is None or start_pos > self.end_pos:
            self._chunks.clear()
        self._start_pos = max(self._start_pos, end_pos - self.window_size)
        # Drop expired live ownership before allocating the new rows. Checkpoint
        # and transfer leases independently retain any old chunks they need.
        self._chunks = [c for c in self._chunks if c.end_pos > self._start_pos]
        if owned_chunk and features.device.type == "cuda":
            # Prefill captures already own independent, bounded chunks. Their
            # allocation stream differs from result consumption's stream.
            if stream is None:
                stream = torch.cuda.current_stream(features.device)
            features.record_stream(stream)
        offset = 0
        while offset < features.shape[0]:
            chunk = self._chunks[-1] if self._chunks else None
            if chunk is None or chunk.used == chunk.buffer.shape[0]:
                if owned_chunk:
                    part = features[offset:]
                    self._chunks.append(
                        _FeatureChunk(start_pos + offset, part, part.shape[0])
                    )
                    break
                chunk = _FeatureChunk(
                    start_pos + offset,
                    features.new_empty(
                        (min(self.window_size, CONTEXT_CHUNK_SIZE), features.shape[1])
                    ),
                    0,
                )
                self._chunks.append(chunk)
            count = min(features.shape[0] - offset, chunk.buffer.shape[0] - chunk.used)
            if chunk.buffer.device.type == "cuda":
                # Restore and consumption can use different streams. The views
                # retained by checkpoints share this allocation's lifetime.
                if stream is None:
                    stream = torch.cuda.current_stream(features.device)
                chunk.buffer.record_stream(stream)
            chunk.buffer[chunk.used : chunk.used + count].copy_(
                features[offset : offset + count]
            )
            chunk.used += count
            offset += count
        self.end_pos = end_pos

    def snapshot(self) -> DFlashContextWindow | None:
        """Lease the current complete window; missing history is an explicit miss."""
        if self.closed or self.end_pos is None:
            return None
        start_pos = max(0, self.end_pos - self.window_size)
        if self._start_pos > start_pos or not self._chunks:
            return None
        return DFlashContextWindow(
            start_pos,
            tuple(
                chunk.buffer[max(0, start_pos - chunk.start_pos) : chunk.used]
                for chunk in self._chunks
                if chunk.end_pos > start_pos
            ),
            self.revision,
        )


@dataclass
class _FeatureCaptureEntry:
    owner: DFlashContextState
    # Offsets are relative to this forward's first token, not packed batch rows.
    slices: list[tuple[int, torch.Tensor]]
    checkpoint_slot: _CheckpointSlot | None


@dataclass
class DFlashContextCapture:
    """Owned forward context, consumed after the existing result completion.

    Only the small position/commit metadata travels to CPU. Context rows stay on
    GPU and are appended to owned chunks on the result-processing stream.
    The result owns this object, so cancellation cannot orphan a capture in a
    global map or let it mutate a later occupant of a recycled request row.
    """

    entries: list[_FeatureCaptureEntry | None]
    # [batch, 3]: forward start, committed rows, recurrent checkpoint end (-1 if none).
    metadata: torch.Tensor
    # Decode entry views share this allocation; fence it once per consumption.
    # Prefill entries instead own independent chunks that consumption can adopt.
    features: torch.Tensor | None

    @staticmethod
    def _owner(
        req: Req, window_size: int, revision: int = 0
    ) -> DFlashContextState | None:
        if req.finished() or req.is_retracted or not req.kv.holds_kv:
            return None
        if req.kv.dflash_context_state is None:
            req.kv.dflash_context_state = DFlashContextState(window_size, revision)
        owner = req.kv.dflash_context_state
        if (
            owner.closed
            or owner.window_size != window_size
            or owner.revision != revision
        ):
            raise ValueError("Invalid request context lifetime, window or revision")
        return owner

    @classmethod
    def prefill(
        cls,
        *,
        reqs: list[Req],
        target_hidden: torch.Tensor,
        prefix_lens: list[int],
        extend_lens: list[int],
        checkpoints: list[tuple[int, int] | None] | None,
        window_size: int,
        materialize: Callable[..., torch.Tensor] | None = None,
        revision: int = 0,
    ) -> DFlashContextCapture:
        if target_hidden.ndim != 2 or sum(extend_lens) > target_hidden.shape[0]:
            raise ValueError("Prefill features do not cover the batch")
        if checkpoints is None:
            checkpoints = [None] * len(reqs)
        entries, metadata, selected, live_rows, selected_lens = [], [], [], [], []
        selected_count = 0
        offset = 0
        for req, prefix, length, checkpoint in zip(
            reqs, prefix_lens, extend_lens, checkpoints, strict=True
        ):
            owner = cls._owner(req, window_size, revision)
            checkpoint_end = checkpoint[1] if checkpoint is not None else -1
            metadata.append((prefix, length, checkpoint_end))
            if (
                checkpoint is not None
                and not prefix < checkpoint_end <= prefix + length
            ):
                raise ValueError("Prefill checkpoint must belong to this forward")
            if owner is None:
                entries.append(None)
                selected_lens.append(0)
                offset += length
                continue
            slot = None
            if checkpoint is not None:
                # Prefill can overwrite the same physical tracking position
                # before the preceding result is consumed; each write has an owner.
                owner.invalidate_checkpoint(checkpoint[0])
                slot = owner._checkpoint_slot(checkpoint[0])
                slot.pending_end_pos = checkpoint_end
            entry = _FeatureCaptureEntry(owner, [], slot)
            entries.append(entry)
            selected_start = selected_count
            intervals = []
            if checkpoint is not None:
                end = checkpoint_end - prefix
                intervals.append((max(0, end - window_size), end))
            start = max(0, length - window_size)
            if intervals and start <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], length)
            else:
                intervals.append((start, length))
            for start, end in intervals:
                chunk_size = min(window_size, CONTEXT_CHUNK_SIZE)
                for pos in range(start, end, chunk_size):
                    stop = min(pos + chunk_size, end)
                    selected.append((offset + pos, offset + stop))
                    live_start = max(pos, length - window_size)
                    live_rows.extend(
                        range(
                            selected_count + live_start - pos,
                            selected_count + stop - pos,
                        )
                    )
                    selected_count += stop - pos
                    entry.slices.append(
                        (
                            pos,
                            target_hidden[offset + pos : offset + stop],
                        )
                    )
            selected_lens.append(selected_count - selected_start)
            offset += length
        if materialize is not None and selected:
            rows = torch.cat(
                [
                    torch.arange(start, end, device=target_hidden.device)
                    for start, end in selected
                ]
            )
            context = materialize(
                rows,
                torch.tensor(live_rows, dtype=torch.int64, device=rows.device),
                selected_lens,
            )
            offset = 0
            for entry in entries:
                if entry is None:
                    continue
                for i, (pos, part) in enumerate(entry.slices):
                    count = part.shape[0]
                    entry.slices[i] = (pos, context[offset : offset + count].clone())
                    offset += count
        else:
            for entry in entries:
                if entry is not None:
                    entry.slices = [
                        (pos, part.detach().clone()) for pos, part in entry.slices
                    ]
        return cls(
            entries, torch.tensor(metadata, dtype=torch.int64, device="cpu"), None
        )

    @classmethod
    def decode(
        cls,
        *,
        reqs: list[Req],
        target_hidden: torch.Tensor,
        start_positions: torch.Tensor,
        commit_lens: torch.Tensor,
        checkpoint_steps: torch.Tensor | None,
        checkpoint_indices: torch.Tensor | None,
        checkpoint_slots: list[int] | None,
        window_size: int,
        revision: int = 0,
    ) -> DFlashContextCapture:
        if target_hidden.ndim != 3 or target_hidden.shape[0] != len(reqs):
            raise ValueError("Decode features must have shape [batch, block, features]")
        if checkpoint_steps is not None:
            if checkpoint_indices is None or checkpoint_slots is None:
                raise ValueError("Checkpoint steps require their tracking destinations")
        packed, metadata = capture_dflash_features(
            target_hidden,
            start_positions,
            commit_lens,
            checkpoint_steps,
            checkpoint_indices,
        )
        entries = []
        for i, req in enumerate(reqs):
            owner = cls._owner(req, window_size, revision)
            slot = (
                owner._checkpoint_slot(checkpoint_slots[i])
                if owner is not None and checkpoint_slots is not None
                else None
            )
            entries.append(
                _FeatureCaptureEntry(owner, [(0, packed[i])], slot)
                if owner is not None
                else None
            )
        return cls(entries, metadata, packed)

    def map_device_tensors(self, copy_to_cpu) -> None:
        if self.metadata.device.type != "cpu":
            self.metadata = copy_to_cpu(self.metadata)

    def consume(self, reqs: list[Req]) -> None:
        if self.metadata.device.type != "cpu":
            raise RuntimeError("Feature capture requires completed host metadata")
        stream = None
        if self.features is not None and self.features.device.type == "cuda":
            stream = torch.cuda.current_stream(self.features.device)
            self.features.record_stream(stream)
        for req, entry, (start, count, checkpoint_end) in zip(
            reqs, self.entries, self.metadata.tolist(), strict=True
        ):
            if entry is None:
                continue
            owner = entry.owner
            if (
                owner.closed
                or req.kv.dflash_context_state is not owner
                or req.finished()
                or req.is_retracted
            ):
                if entry.checkpoint_slot is not None:
                    entry.checkpoint_slot.cancel_donation()
                continue
            if owner.end_pos is not None and owner.end_pos != start:
                raise ValueError("Completed feature captures must be consumed in order")
            end = start + count
            if count < 0 or (
                checkpoint_end != -1 and not start < checkpoint_end <= end
            ):
                raise ValueError("Invalid committed feature or checkpoint boundary")
            for offset, features in entry.slices:
                features = features[: max(0, count - offset)]
                lo, hi = start + offset, start + offset + features.shape[0]
                if lo < checkpoint_end <= hi:
                    split = checkpoint_end - lo
                    owner._append(
                        features[:split], lo, stream, owned_chunk=self.features is None
                    )
                    slot = entry.checkpoint_slot
                    if slot is not None:
                        if slot.destination is not None:
                            pool, handle = slot.destination
                            if pool.contains(handle):
                                tail = owner.snapshot()
                                if tail is None:
                                    pool.release(handle)
                                else:
                                    pool.write_reserved(handle, tail)
                            slot.destination = None
                        elif owner._checkpoints.get(slot.index) is slot:
                            slot.tail = owner.snapshot()
                            slot.pending_end_pos = None
                    owner._append(
                        features[split:],
                        checkpoint_end,
                        stream,
                        owned_chunk=self.features is None,
                    )
                else:
                    owner._append(
                        features, lo, stream, owned_chunk=self.features is None
                    )
            if count and owner.end_pos != end:
                raise ValueError("Capture does not cover the committed end")
        self.entries.clear()
        self.features = None
