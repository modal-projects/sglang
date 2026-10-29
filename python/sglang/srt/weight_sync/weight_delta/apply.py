"""Inside the pause: copy this GPU's staged version to the GPU and apply it to the serving tensors."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import triton

from sglang.srt.weight_sync.weight_delta.decode import (
    GapDecoder,
    SnappyDecoder,
    check_decode,
)
from sglang.srt.weight_sync.weight_delta.decode import warm_up as warm_up_decode
from sglang.srt.weight_sync.weight_delta.map import TORCH_DTYPES, DeltaMap
from sglang.srt.weight_sync.weight_delta.prepare import Prepared, ragged_rows
from sglang.srt.weight_sync.weight_delta.xor_kernel import ByteApplyGroup, plan_xor
from sglang.srt.weight_sync.weight_delta.xor_kernel import warm_up as warm_up_xor

# Each decode slot holds this much, or the largest changed tensor if that is more.
SLOT_NBYTES = 1 << 30
SLOT_ALIGNMENT = 256


@dataclass
class Batch:
    """Consecutive xor tensors decoded into one slot together, then XORed into their places."""

    frames: np.ndarray  # per frame: input offset, encoded, decoded, slot offset
    input_start: int  # where the batch's inner bytes start in the staged bytes
    input_nbytes: int
    slot_nbytes: int
    # Slot bytes no frame writes, as (offset, nbytes) rows; None zeroes the whole slot.
    zero_ranges: np.ndarray | None
    xor: ByteApplyGroup
    xor_table: np.ndarray  # pointer rows, sources as slot offsets, then metadata


@dataclass
class Plan:
    batches: list[Batch]
    raw_nbytes: int  # the raw values are the staged bytes [0, raw_nbytes)
    raw_buffer: torch.Tensor
    # (targets, sources) groups, one multi-tensor copy each: same-dtype values as bytes, so the
    # copy is exact, then each cast by its dtype pair.
    raw_copies: list[tuple[list[torch.Tensor], list[torch.Tensor]]]


class Applier:
    """One GPU's decoder and the plans it reuses: an RL step changes the same tensors every time,
    so a batch's XOR plan and the raw copies are built once."""

    def __init__(self, delta_map: DeltaMap, codec: str, device: torch.device) -> None:
        self.delta_map, self.device = delta_map, torch.device(device)
        self.decoder = GapDecoder() if codec == "gap-zstd" else SnappyDecoder(device)
        self._xor_plans: dict[tuple, tuple[ByteApplyGroup, np.ndarray]] = {}
        self._raw_plans: dict[tuple, tuple] = {}
        warm_up_decode(self.device)
        warm_up_xor(self.device.index)

    def plan(self, prepared: Prepared) -> Plan:
        """Before the pause."""
        raw = np.flatnonzero(prepared.raw)
        xor = np.flatnonzero(~prepared.raw)
        raw_nbytes = (
            int(prepared.staged_offset[raw[-1]] + prepared.staged_nbytes[raw[-1]])
            if len(raw)
            else 0
        )
        raw_buffer, raw_copies = self._raw_plan(prepared, raw, raw_nbytes)
        frame_rows, frame_end = ragged_rows(prepared.frame_end, xor)
        frames = {key: column[frame_rows] for key, column in prepared.frames.items()}
        frame_starts = np.concatenate(([0], frame_end))
        batches = []
        for members in _batches(prepared.nbytes[xor]):
            batch_rows = xor[members]
            first, last = int(batch_rows[0]), int(batch_rows[-1])
            input_start = int(prepared.staged_offset[first])
            spans = -(-prepared.nbytes[batch_rows] // SLOT_ALIGNMENT) * SLOT_ALIGNMENT
            slot_offset = np.cumsum(spans) - spans
            in_batch = np.arange(
                frame_starts[members[0]], frame_starts[members[-1] + 1]
            )
            member = np.repeat(
                np.arange(len(batch_rows)), np.diff(frame_end, prepend=0)[members]
            )
            frame_table = np.stack(
                [
                    prepared.staged_offset[batch_rows][member]
                    - input_start
                    + frames["inner_offset"][in_batch],
                    frames["inner_nbytes"][in_batch],
                    frames["nbytes"][in_batch],
                    slot_offset[member] + frames["offset"][in_batch],
                ]
            ).astype(np.int64)
            self.decoder.check(frame_table)
            zero_ranges = None
            if self.decoder.whole_frames:
                zero_ranges = _uncovered(
                    slot_offset,
                    prepared.nbytes[batch_rows],
                    frame_table[3],
                    frame_table[2],
                    np.diff(frame_end, prepend=0)[members],
                )
            xor_plan, xor_table = self._xor_plan(
                prepared.table_rows[batch_rows], slot_offset
            )
            batches.append(
                Batch(
                    frames=frame_table,
                    input_start=input_start,
                    input_nbytes=int(
                        prepared.staged_offset[last] + prepared.staged_nbytes[last]
                    )
                    - input_start,
                    slot_nbytes=int(slot_offset[-1] + spans[-1]),
                    zero_ranges=zero_ranges,
                    xor=xor_plan,
                    xor_table=xor_table,
                )
            )
        return Plan(batches, raw_nbytes, raw_buffer, raw_copies)

    def apply(self, plan: Plan, staged: torch.Tensor) -> dict:
        """Inside the pause. Any failure raises with the weights partly updated, so the caller
        takes the engine down."""
        started = time.perf_counter()
        # An in-place pause leaves the last launched batch running.
        torch.cuda.synchronize(self.device)
        synced = time.perf_counter()
        arena = None
        try:
            arena = _DeviceArena(self._arena_nbytes(plan), self.device)
            return self._apply(plan, staged, arena, started, synced)
        finally:
            # Nothing may be freed under a running decode or copy.
            torch.cuda.synchronize(self.device)
            if arena is not None:
                arena.free()

    def _apply(self, plan, staged, arena, started, synced) -> dict:
        batches, device = plan.batches, self.device
        buffers = min(2, len(batches))
        slot_nbytes = max((batch.slot_nbytes for batch in batches), default=0)
        input_nbytes = max((batch.input_nbytes for batch in batches), default=0)
        slots = [arena.take(slot_nbytes) for _ in range(buffers)]
        inputs = [arena.take(input_nbytes) for _ in range(buffers)]
        temp = arena.take(self.decoder.temp_nbytes)
        counts = [batch.frames.shape[1] for batch in batches]
        decode_metadata = arena.take(32 * sum(counts)).view(torch.int64)
        statuses = arena.take(4 * sum(counts)).view(torch.int32)
        actual_sizes = arena.take(8 * sum(counts)).view(torch.int64)
        error = torch.zeros(1, dtype=torch.int32, device=device)

        host_metadata, host_xor_tables = [], []
        for index, batch in enumerate(batches):
            slot = slots[index % 2]
            host_metadata.append(
                self.decoder.metadata(
                    batch.frames, inputs[index % 2].data_ptr(), slot.data_ptr()
                ).ravel()
            )
            if not batch.xor.bind(slot):
                raise RuntimeError("a decode slot is not 16-byte aligned")
            xor_table = batch.xor_table.copy()
            xor_table[list(batch.xor.source_indices)] += slot.data_ptr()
            host_xor_tables.append(xor_table)
        if batches:
            decode_metadata.copy_(torch.from_numpy(np.concatenate(host_metadata)))
        xor_tables = (
            torch.from_numpy(np.concatenate(host_xor_tables)).to(device)
            if batches
            else None
        )
        frames_at = np.cumsum([0, *counts]).tolist()
        launches, table_at = [], 0
        for batch, xor_table in zip(batches, host_xor_tables):
            pointers = 2 * len(batch.xor.sources)
            view = xor_tables[table_at : table_at + len(xor_table)]
            launches.append(
                batch.xor.bind_launch(view[:pointers], view[pointers:], error)
            )
            table_at += len(xor_table)
        bound = time.perf_counter()

        current = torch.cuda.current_stream(device)
        copy_stream, decode_stream, xor_stream = (
            torch.cuda.Stream(device) for _ in range(3)
        )
        staged_ready = [torch.cuda.Event() for _ in inputs]
        input_free = [torch.cuda.Event() for _ in inputs]
        slot_ready = [torch.cuda.Event() for _ in slots]
        slot_free = [torch.cuda.Event() for _ in slots]
        begin, applied = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record(current)
        for stream in (copy_stream, decode_stream, xor_stream):
            stream.wait_stream(current)

        def copy_input(index):
            batch = batches[index]
            with torch.cuda.stream(copy_stream):
                if index >= 2:
                    copy_stream.wait_event(input_free[index % 2])
                inputs[index % 2][: batch.input_nbytes].copy_(
                    staged[batch.input_start : batch.input_start + batch.input_nbytes],
                    non_blocking=True,
                )
                staged_ready[index % 2].record(copy_stream)

        def decode_batch(index):
            batch, count = batches[index], counts[index]
            start = 4 * frames_at[index]
            metadata = decode_metadata[start : start + 4 * count].view(4, count)
            batch_statuses = statuses[frames_at[index] : frames_at[index] + count]
            batch_sizes = actual_sizes[frames_at[index] : frames_at[index] + count]
            with torch.cuda.stream(decode_stream):
                if index >= 2:
                    decode_stream.wait_event(slot_free[index % 2])
                decode_stream.wait_event(staged_ready[index % 2])
                slot = slots[index % 2]
                if batch.zero_ranges is None:
                    slot[: batch.slot_nbytes].zero_()
                elif len(batch.zero_ranges):
                    torch._foreach_zero_(
                        [
                            slot[at : at + size]
                            for at, size in batch.zero_ranges.tolist()
                        ]
                    )
                if count:
                    self.decoder.enqueue(
                        metadata, batch_statuses, batch_sizes, temp, count
                    )
                    check_decode[(triton.cdiv(count, 1024),)](
                        batch_statuses,
                        batch_sizes,
                        self.decoder.expected_sizes(metadata),
                        error,
                        count,
                        BLOCK=1024,
                    )
                input_free[index % 2].record(decode_stream)
                slot_ready[index % 2].record(decode_stream)
            # Queued only now: a stream waits on an event's latest record at queue time.
            if index + 2 < len(batches):
                copy_input(index + 2)

        with torch.cuda.stream(xor_stream):
            # Raw targets never overlap XOR targets, so they go beside the first decode.
            if plan.raw_nbytes:
                plan.raw_buffer.copy_(staged[: plan.raw_nbytes], non_blocking=True)
                for targets, sources in plan.raw_copies:
                    torch._foreach_copy_(targets, sources)
        for index in range(buffers):
            copy_input(index)
        if batches:
            decode_batch(0)
        for index, launch in enumerate(launches):
            with torch.cuda.stream(xor_stream):
                xor_stream.wait_event(slot_ready[index % 2])
                if launch is not None:
                    launch()
                slot_free[index % 2].record(xor_stream)
            # After this XOR is queued: nvCOMP may wait for earlier work on its stream.
            if index + 1 < len(batches):
                decode_batch(index + 1)
        with torch.cuda.stream(xor_stream):
            for copied in self.delta_map.copies:
                copied.dst.copy_(copied.src)
        # Recomputing derived tensors allocates temporaries: on the caller's stream, whose cached
        # blocks SGLang reuses. On a stream made for this apply they would stay cached for good.
        current.wait_stream(xor_stream)
        for derived in self.delta_map.derived:
            for value, dst in derived.values():
                dst.copy_(value)
        applied.record(current)
        enqueued = time.perf_counter()
        applied.synchronize()
        finished = time.perf_counter()
        if error.item():
            raise RuntimeError("a frame failed to decode or decoded to the wrong size")
        return {
            "batches": len(batches),
            "raw": sum(len(targets) for targets, _ in plan.raw_copies),
            "slot_nbytes": slot_nbytes,
            "input_nbytes": input_nbytes,
            "sync_s": synced - started,
            "bind_s": bound - synced,
            "enqueue_s": enqueued - bound,
            "wait_s": finished - enqueued,
            "gpu_s": begin.elapsed_time(applied) / 1e3,
        }

    def _arena_nbytes(self, plan: Plan) -> int:
        batches = plan.batches
        buffers = min(2, len(batches))
        count = sum(batch.frames.shape[1] for batch in batches)
        return (
            buffers * max((batch.slot_nbytes for batch in batches), default=0)
            + buffers * max((batch.input_nbytes for batch in batches), default=0)
            + self.decoder.temp_nbytes
            + (32 + 4 + 8) * count
            + (2 * buffers + 4) * _DeviceArena.ALIGNMENT
        )

    def _raw_plan(self, prepared: Prepared, raw: np.ndarray, nbytes: int):
        key = tuple(prepared.table_rows[raw].tolist())
        if key not in self._raw_plans:
            inventory, entries = self.delta_map.inventory, self.delta_map.table.entries
            names = self.delta_map.table.names
            buffer = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
            copies: dict[tuple, tuple[list, list]] = {}
            for index in raw.tolist():
                row = int(prepared.table_rows[index])
                tensor = inventory[names[row].decode()]
                offset, size = (
                    int(prepared.staged_offset[index]),
                    int(prepared.nbytes[index]),
                )
                value = (
                    buffer[offset : offset + size]
                    .view(TORCH_DTYPES[tensor.dtype])
                    .reshape(tensor.shape)
                )
                for place in entries[row]:
                    source = value[place.src]
                    target = place.dst.view(source.shape)
                    if source.dtype == target.dtype:
                        source, target = _bytes(source), _bytes(target)
                    targets, sources = copies.setdefault(
                        (source.dtype, target.dtype), ([], [])
                    )
                    targets.append(target)
                    sources.append(source)
            self._raw_plans[key] = (buffer, list(copies.values()))
        return self._raw_plans[key]

    def _xor_plan(self, table_rows: np.ndarray, slot_offset: np.ndarray):
        key = tuple(table_rows.tolist())
        if key not in self._xor_plans:
            inventory, entries = self.delta_map.inventory, self.delta_map.table.entries
            names = self.delta_map.table.names
            end = int(slot_offset[-1] + self.delta_map.table.nbytes[table_rows[-1]])
            # Sources view a meta stand-in for the slot, which exists only in the pause.
            slot = torch.empty(end, dtype=torch.uint8, device="meta")
            pairs = []
            for row, offset in zip(table_rows.tolist(), slot_offset.tolist()):
                tensor = inventory[names[row].decode()]
                nbytes = int(self.delta_map.table.nbytes[row])
                canonical = (
                    slot[offset : offset + nbytes]
                    .view(TORCH_DTYPES[tensor.dtype])
                    .reshape(tensor.shape)
                )
                for place in entries[row]:
                    pairs.append((_bytes(canonical[place.src]), _bytes(place.dst)))
            group = plan_xor(pairs)
            group.prepare_metadata(self.device.index)
            pointers = np.fromiter(group.pointer_rows(0), dtype=np.int64)
            self._xor_plans[key] = (
                group,
                np.concatenate(
                    [pointers, np.asarray(group.metadata_rows, dtype=np.int64)]
                ),
            )
        return self._xor_plans[key]


def _batches(nbytes: np.ndarray) -> list[np.ndarray]:
    """Consecutive tensors per batch, each batch's aligned bytes within one slot."""
    spans = -(-nbytes // SLOT_ALIGNMENT) * SLOT_ALIGNMENT
    capacity = max(SLOT_NBYTES, int(spans.max(initial=0)))
    batches, start, used = [], 0, 0
    for index, span in enumerate(spans.tolist()):
        if used + span > capacity:
            batches.append(np.arange(start, index))
            start, used = index, 0
        used += span
    if len(spans):
        batches.append(np.arange(start, len(spans)))
    return batches


def _uncovered(
    tensor_offset: np.ndarray,
    tensor_nbytes: np.ndarray,
    frame_offset: np.ndarray,
    frame_nbytes: np.ndarray,
    frame_counts: np.ndarray,
) -> np.ndarray:
    """The (offset, nbytes) of each range of the tensors' slot regions that no frame covers:
    pair every region start and frame end with the next frame start or region end."""
    first = np.cumsum(frame_counts) - frame_counts
    ends = np.insert(frame_offset + frame_nbytes, first, tensor_offset)
    starts = np.insert(
        frame_offset, first + frame_counts, tensor_offset + tensor_nbytes
    )
    gaps = starts > ends
    return np.stack([ends[gaps], starts[gaps] - ends[gaps]], axis=1)


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape or (1,)).view(torch.uint8)


class _DeviceArena:
    """GPU memory taken with cuMemAlloc for one pause, outside PyTorch's allocator: memory the
    decompression engine can read, given back before the engine resumes."""

    ALIGNMENT = 256

    def __init__(self, nbytes: int, device: torch.device) -> None:
        from cuda.bindings import driver

        self._driver, self.device = driver, device
        self._address, self._used, self.nbytes = 0, 0, nbytes
        if nbytes:
            with torch.cuda.device(device):
                status, address = driver.cuMemAlloc(nbytes)
            if status:
                raise RuntimeError(f"cuMemAlloc of {nbytes} bytes failed: {status}")
            self._address = int(address)

    def take(self, nbytes: int) -> torch.Tensor:
        start = -(-self._used // self.ALIGNMENT) * self.ALIGNMENT
        if start + nbytes > self.nbytes:
            raise RuntimeError("the weight-delta device arena is too small")
        self._used = start + nbytes
        return torch.as_tensor(
            _DeviceBytes(self._address + start, nbytes), device=self.device
        )

    def free(self) -> None:
        if self._address:
            with torch.cuda.device(self.device):
                status = self._driver.cuMemFree(self._address)[0]
            self._address = 0
            if status:
                raise RuntimeError(f"cuMemFree failed: {status}")


class _DeviceBytes:
    def __init__(self, address: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (address, False),
            "version": 3,
        }
