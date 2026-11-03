"""Inside the pause: copy this GPU's staged version to the GPU and apply it to the serving tensors."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import triton

from sglang.srt.weight_sync.weight_delta.decode import (
    GAP_NBYTES,
    SnappyDecoder,
    check_decode,
    check_gap_frames,
    gap_places,
    gap_scatter,
)
from sglang.srt.weight_sync.weight_delta.decode import warm_up as warm_up_decode
from sglang.srt.weight_sync.weight_delta.map import TORCH_DTYPES, DeltaMap
from sglang.srt.weight_sync.weight_delta.prepare import Prepared, ragged_rows, ranges
from sglang.srt.weight_sync.weight_delta.xor_kernel import (
    ByteApplyGroup,
    plan_xor,
    raw_checksums,
)
from sglang.srt.weight_sync.weight_delta.xor_kernel import warm_up as warm_up_xor

# Gap frames reach the GPU in chunks of about this many inner bytes, each copied while the one
# before is applied.
CHUNK_NBYTES = 64 << 20
# Each decode slot holds this much, or the largest tensor if that is more.
SLOT_NBYTES = 1 << 30
SLOT_ALIGNMENT = 256


@dataclass
class Chunk:
    """Consecutive gap frames copied to the GPU together, then XORed into their places."""

    input_start: int  # where the chunk's inner bytes start in the staged bytes
    input_nbytes: int
    # gap_scatter's items, one per frame and place it touches, inner offsets in the chunk.
    items: np.ndarray


@dataclass
class Batch:
    """Tensors of one static slot layout decoded together, then XORed into their places."""

    frames: np.ndarray  # per frame: input offset, encoded, decoded, slot offset
    input_start: int  # where the batch's inner bytes start in the staged bytes
    input_nbytes: int
    slot_nbytes: int
    # Slot bytes no frame writes, as (offset, nbytes) rows: the version's frames do not cover
    # every byte, nor any of the tensors it does not carry.
    zero_ranges: np.ndarray
    xor: ByteApplyGroup
    xor_table: np.ndarray  # pointer rows, sources as slot offsets, then metadata


@dataclass
class _Layout:
    """A snappy-zstd batch fixed at startup: each version's Batch of it reuses its XOR plan."""

    rows: np.ndarray
    slot_offset: np.ndarray
    slot_nbytes: int
    xor: ByteApplyGroup
    xor_table: np.ndarray


@dataclass
class RawPlan:
    buffer: torch.Tensor
    # (targets, sources) groups, one multi-tensor copy each: same-dtype values as bytes, so the
    # copy is exact, then each cast by its dtype pair.
    copies: list[tuple[list[torch.Tensor], list[torch.Tensor]]]
    # Each target as the canonical region it holds, cast back to the checkpoint dtype (casts
    # only widen, so exactly), and per byte its canonical position and checksum row.
    checked: list[tuple[torch.Tensor, torch.dtype]]
    positions: np.ndarray
    rows: np.ndarray


@dataclass
class Plan:
    chunks: list[Chunk]  # gap-zstd's
    batches: list[Batch]  # snappy-zstd's
    raw_nbytes: int  # the raw values are the staged bytes [0, raw_nbytes)
    raw: RawPlan
    # The checksum rows (MapTable rows) of the version's tensors this GPU holds, and the
    # trainer's checksum of each; the rows of tensors split across a group, per group.
    rows: np.ndarray
    expected: np.ndarray
    split: list[tuple[object, np.ndarray]]


class Applier:
    """Built at startup: where a tensor's bytes go does not depend on what a version carries."""

    def __init__(self, delta_map: DeltaMap, codec: str, device: torch.device) -> None:
        self.delta_map, self.device = delta_map, torch.device(device)
        table = delta_map.table
        self.places, self.decoder, self._layouts = None, None, []
        if codec == "gap-zstd":
            self.places = gap_places(delta_map, self.device)
        else:
            self.decoder = SnappyDecoder(device)
            self._layouts = _layouts(delta_map, self.device)
        self._layout_of = np.full(len(table.names), -1, np.int64)
        self._slot_offset = np.zeros(len(table.names), np.int64)
        for index, layout in enumerate(self._layouts):
            self._layout_of[layout.rows] = index
            self._slot_offset[layout.rows] = layout.slot_offset
        self._groups, self._group_of = [], np.full(len(table.names), -1, np.int64)
        for row, entry in enumerate(table.entries):
            if isinstance(entry, tuple) and (group := entry[0].split_group) is not None:
                if group not in self._groups:
                    self._groups.append(group)
                self._group_of[row] = self._groups.index(group)
        # Which tensors arrive raw is the trainer's choice: their positions are found on arrival.
        self._raw_places: dict[int, list[tuple[object, np.ndarray]]] = {}
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
        group = self._group_of[prepared.table_rows]
        return Plan(
            self._gap_chunks(prepared, xor) if self.places is not None else [],
            self._snappy_batches(prepared, xor) if self.decoder is not None else [],
            raw_nbytes,
            self._raw_plan(prepared, raw, raw_nbytes),
            prepared.table_rows,
            prepared.checksum,
            [
                (self._groups[index], prepared.table_rows[group == index])
                for index in np.unique(group[group >= 0]).tolist()
            ],
        )

    def _gap_chunks(self, prepared: Prepared, xor: np.ndarray) -> list[Chunk]:
        frame_rows, frame_end = ragged_rows(prepared.frame_end, xor)
        frames = {key: column[frame_rows] for key, column in prepared.frames.items()}
        check_gap_frames(frames["inner_nbytes"], frames["nbytes"])
        tensor = np.repeat(xor, np.diff(frame_end, prepend=0))
        inner_start = prepared.staged_offset[tensor] + frames["inner_offset"]
        inner_end = inner_start + frames["inner_nbytes"]
        places, row = self.places, prepared.table_rows[tensor]
        frame = np.repeat(np.arange(len(tensor)), places.count[row])
        place = ranges(places.start[row], places.count[row])
        start = frames["offset"][frame]
        stop = start + frames["nbytes"][frame]
        touched = (start < places.high[place]) & (stop > places.low[place])
        frame, place = frame[touched], place[touched]
        if not len(frame):
            return []
        chunk = (inner_end[frame] - 1 - inner_start[frame[0]]) // CHUNK_NBYTES
        chunks = []
        for items in np.split(
            np.arange(len(frame)), np.flatnonzero(np.diff(chunk)) + 1
        ):
            members, records = frame[items], places.record[place[items]]
            input_start = int(inner_start[members[0]])
            chunks.append(
                Chunk(
                    input_start,
                    int(inner_end[members[-1]]) - input_start,
                    np.stack(
                        [
                            inner_start[members] - input_start,
                            frames["inner_nbytes"][members] // (GAP_NBYTES + 1),
                            frames["offset"][members],
                            frames["nbytes"][members],
                            records,
                        ]
                    ).astype(np.int64),
                )
            )
        return chunks

    def _snappy_batches(self, prepared: Prepared, xor: np.ndarray) -> list[Batch]:
        if not len(xor):
            return []
        rows = prepared.table_rows[xor]
        layout_of = self._layout_of[rows]
        frame_rows, frame_end = ragged_rows(prepared.frame_end, xor)
        frames = {key: column[frame_rows] for key, column in prepared.frames.items()}
        frame_counts = np.diff(frame_end, prepend=0)
        frame_starts = np.concatenate(([0], frame_end))
        batches = []
        # Staged in table-row order, so a layout's tensors in the version are consecutive.
        for members in np.split(
            np.arange(len(xor)), np.flatnonzero(np.diff(layout_of)) + 1
        ):
            layout = self._layouts[int(layout_of[members[0]])]
            tensors = xor[members]
            input_start = int(prepared.staged_offset[tensors[0]])
            in_batch = np.arange(
                frame_starts[members[0]], frame_starts[members[-1] + 1]
            )
            member = np.repeat(np.arange(len(tensors)), frame_counts[members])
            frame_table = np.stack(
                [
                    prepared.staged_offset[tensors][member]
                    - input_start
                    + frames["inner_offset"][in_batch],
                    frames["inner_nbytes"][in_batch],
                    frames["nbytes"][in_batch],
                    self._slot_offset[rows[members]][member]
                    + frames["offset"][in_batch],
                ]
            ).astype(np.int64)
            self.decoder.check(frame_table)
            counts = np.zeros(len(layout.rows), np.int64)
            counts[np.searchsorted(layout.rows, rows[members])] = frame_counts[members]
            batches.append(
                Batch(
                    frames=frame_table,
                    input_start=input_start,
                    input_nbytes=int(
                        prepared.staged_offset[tensors[-1]]
                        + prepared.staged_nbytes[tensors[-1]]
                    )
                    - input_start,
                    slot_nbytes=layout.slot_nbytes,
                    zero_ranges=_uncovered(
                        layout.slot_offset,
                        self.delta_map.table.nbytes[layout.rows],
                        frame_table[3],
                        frame_table[2],
                        counts,
                    ),
                    xor=layout.xor,
                    xor_table=layout.xor_table,
                )
            )
        return batches

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
        device = self.device
        error = torch.zeros(1, dtype=torch.int32, device=device)
        checksums = arena.take(8 * len(self.delta_map.table.names)).view(torch.int64)
        checksums.zero_()
        raw, split = plan.raw, plan.split
        lengths = [
            len(raw.positions),
            len(raw.rows),
            len(plan.rows),
            len(plan.expected),
        ]
        lengths += [len(rows) for _, rows in split]
        host = np.concatenate(
            [raw.positions, raw.rows, plan.rows, plan.expected.view(np.int64)]
            + [rows for _, rows in split]
        ).astype(np.int64)
        raw_positions, raw_rows, rows, expected, *split_rows = (
            torch.from_numpy(host).to(device).split(lengths)
        )
        current = torch.cuda.current_stream(device)
        apply_stream = torch.cuda.Stream(device)
        begin, applied = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        if self.places is not None:
            enqueue = self._scatter(plan, staged, arena, error, checksums, apply_stream)
        else:
            enqueue = self._decode_and_xor(
                plan, staged, arena, error, checksums, apply_stream
            )
        bound = time.perf_counter()

        begin.record(current)
        apply_stream.wait_stream(current)
        with torch.cuda.stream(apply_stream):
            # The XORs' stream: a Snappy layout also XORs zeros over tensors that arrive raw.
            if plan.raw_nbytes:
                raw.buffer.copy_(staged[: plan.raw_nbytes], non_blocking=True)
                for targets, sources in raw.copies:
                    torch._foreach_copy_(targets, sources)
                values = [
                    target.to(dtype).reshape(-1).view(torch.uint8)
                    for target, dtype in raw.checked
                ]
                raw_checksums(torch.cat(values), raw_positions, raw_rows, checksums)
        enqueue(current)
        with torch.cuda.stream(apply_stream):
            # A split tensor's parts are summed over the group whose GPUs hold them, on every
            # GPU before any can raise, so none waits in a collective.
            for (group, _), split_at in zip(split, split_rows):
                parts = checksums[split_at]
                torch.distributed.all_reduce(parts, group=group.device_group)
                checksums[split_at] = parts
            for copied in self.delta_map.copies:
                copied.dst.copy_(copied.src)
        # Recomputing derived tensors allocates temporaries: on the caller's stream, whose cached
        # blocks SGLang reuses. On a stream made for this apply they would stay cached for good.
        current.wait_stream(apply_stream)
        for derived in self.delta_map.derived:
            for value, dst in derived.values():
                dst.copy_(value)
        applied.record(current)
        enqueued = time.perf_counter()
        applied.synchronize()
        finished = time.perf_counter()
        if error.item():
            raise RuntimeError("a frame failed to decode or decoded to the wrong size")
        wrong = torch.nonzero(checksums[rows] != expected).flatten().tolist()
        if wrong:
            names = [
                self.delta_map.table.names[plan.rows[i]].decode() for i in wrong[:3]
            ]
            raise RuntimeError(
                f"{len(wrong)} tensors differ from the trainer's after applying, e.g. {names}"
            )
        return {
            "batches": len(plan.chunks) + len(plan.batches),
            "raw": sum(len(targets) for targets, _ in raw.copies),
            "slot_nbytes": max(
                (batch.slot_nbytes for batch in plan.batches), default=0
            ),
            "input_nbytes": max(
                (part.input_nbytes for part in (*plan.chunks, *plan.batches)), default=0
            ),
            "sync_s": synced - started,
            "bind_s": bound - synced,
            "enqueue_s": enqueued - bound,
            "wait_s": finished - enqueued,
            "gpu_s": begin.elapsed_time(applied) / 1e3,
        }

    def _scatter(self, plan, staged, arena, error, checksums, apply_stream):
        """Binds the gap chunks; the returned call queues their copies and scatters."""
        device, chunks = self.device, plan.chunks
        buffers = min(2, len(chunks))
        input_nbytes = max((chunk.input_nbytes for chunk in chunks), default=0)
        inputs = [arena.take(input_nbytes) for _ in range(buffers)]
        host = []
        for index, chunk in enumerate(chunks):
            items = chunk.items.copy()
            items[0] += inputs[index % 2].data_ptr()
            host.append(items.ravel())
        items = torch.from_numpy(np.concatenate(host)).to(device) if host else None
        item_at = np.cumsum([0, *(len(part) for part in host)]).tolist()

        def enqueue(current):
            copy_stream = torch.cuda.Stream(device)
            copy_stream.wait_stream(current)
            staged_ready = [torch.cuda.Event() for _ in inputs]
            input_free = [torch.cuda.Event() for _ in inputs]

            def copy_input(index):
                chunk = chunks[index]
                with torch.cuda.stream(copy_stream):
                    if index >= 2:
                        copy_stream.wait_event(input_free[index % 2])
                    inputs[index % 2][: chunk.input_nbytes].copy_(
                        staged[
                            chunk.input_start : chunk.input_start + chunk.input_nbytes
                        ],
                        non_blocking=True,
                    )
                    staged_ready[index % 2].record(copy_stream)

            for index in range(buffers):
                copy_input(index)
            for index, chunk in enumerate(chunks):
                with torch.cuda.stream(apply_stream):
                    apply_stream.wait_event(staged_ready[index % 2])
                    gap_scatter(
                        items[item_at[index] : item_at[index + 1]],
                        self.places,
                        error,
                        checksums,
                        chunk.items.shape[1],
                    )
                    input_free[index % 2].record(apply_stream)
                # Queued only now: a stream waits on an event's latest record at queue time.
                if index + 2 < len(chunks):
                    copy_input(index + 2)

        return enqueue

    def _decode_and_xor(self, plan, staged, arena, error, checksums, apply_stream):
        """Binds the Snappy batches; the returned call queues their copies, decodes and XORs."""
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
            pointers = 4 * len(batch.xor.sources)
            view = xor_tables[table_at : table_at + len(xor_table)]
            launches.append(
                batch.xor.bind_launch(
                    view[:pointers], view[pointers:], error, checksums
                )
            )
            table_at += len(xor_table)

        def enqueue(current):
            copy_stream, decode_stream = (torch.cuda.Stream(device) for _ in range(2))
            for stream in (copy_stream, decode_stream):
                stream.wait_stream(current)
            staged_ready = [torch.cuda.Event() for _ in inputs]
            input_free = [torch.cuda.Event() for _ in inputs]
            slot_ready = [torch.cuda.Event() for _ in slots]
            slot_free = [torch.cuda.Event() for _ in slots]

            def copy_input(index):
                batch = batches[index]
                with torch.cuda.stream(copy_stream):
                    if index >= 2:
                        copy_stream.wait_event(input_free[index % 2])
                    inputs[index % 2][: batch.input_nbytes].copy_(
                        staged[
                            batch.input_start : batch.input_start + batch.input_nbytes
                        ],
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
                    if len(batch.zero_ranges):
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

            for index in range(buffers):
                copy_input(index)
            if batches:
                decode_batch(0)
            for index, launch in enumerate(launches):
                with torch.cuda.stream(apply_stream):
                    apply_stream.wait_event(slot_ready[index % 2])
                    if launch is not None:
                        launch()
                    slot_free[index % 2].record(apply_stream)
                # After this XOR is queued: nvCOMP may wait for earlier work on its stream.
                if index + 1 < len(batches):
                    decode_batch(index + 1)

        return enqueue

    def _arena_nbytes(self, plan: Plan) -> int:
        tables = 8 * len(self.delta_map.table.names)
        if self.places is not None:
            buffers = min(2, len(plan.chunks))
            return (
                buffers * max((chunk.input_nbytes for chunk in plan.chunks), default=0)
                + tables
                + (buffers + 2) * _DeviceArena.ALIGNMENT
            )
        batches = plan.batches
        buffers = min(2, len(batches))
        count = sum(batch.frames.shape[1] for batch in batches)
        return (
            buffers * max((batch.slot_nbytes for batch in batches), default=0)
            + buffers * max((batch.input_nbytes for batch in batches), default=0)
            + self.decoder.temp_nbytes
            + (32 + 4 + 8) * count
            + tables
            + (2 * buffers + 5) * _DeviceArena.ALIGNMENT
        )

    def _raw_plan(self, prepared: Prepared, raw: np.ndarray, nbytes: int) -> RawPlan:
        inventory, names = self.delta_map.inventory, self.delta_map.table.names
        buffer = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
        copies: dict[tuple, tuple[list, list]] = {}
        checked, positions, rows = [], [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
        for index in raw.tolist():
            row = int(prepared.table_rows[index])
            tensor = inventory[names[row].decode()]
            dtype = TORCH_DTYPES[tensor.dtype]
            offset, size = (
                int(prepared.staged_offset[index]),
                int(prepared.nbytes[index]),
            )
            value = buffer[offset : offset + size].view(dtype).reshape(tensor.shape)
            for place, held in self._raw_place_positions(row):
                source = value[place.src]
                target = place.dst.view(source.shape)
                checked.append((target, dtype))
                positions.append(held)
                rows.append(np.full(len(held), row, np.int64))
                if source.dtype == target.dtype:
                    source, target = _bytes(source), _bytes(target)
                targets, sources = copies.setdefault(
                    (source.dtype, target.dtype), ([], [])
                )
                targets.append(target)
                sources.append(source)
        return RawPlan(
            buffer,
            list(copies.values()),
            checked,
            np.concatenate(positions),
            np.concatenate(rows),
        )

    def _raw_place_positions(self, row: int) -> list[tuple[object, np.ndarray]]:
        if row not in self._raw_places:
            tensor = self.delta_map.inventory[self.delta_map.table.names[row].decode()]
            itemsize = TORCH_DTYPES[tensor.dtype].itemsize
            size = int(self.delta_map.table.nbytes[row])
            byte_positions = torch.arange(size).view(*tensor.shape, itemsize)
            self._raw_places[row] = [
                (place, byte_positions[place.src].reshape(-1).numpy())
                for place in self.delta_map.table.entries[row]
            ]
        return self._raw_places[row]


def _layouts(delta_map: DeltaMap, device: torch.device) -> list[_Layout]:
    """Every tensor the GPU can take XORed, in table-row order (prepare's), cut into slots."""
    table = delta_map.table
    rows = np.flatnonzero(table.held & ~table.raw_only)
    layouts = []
    for members in _batches(table.nbytes[rows]):
        batch_rows = rows[members]
        spans = -(-table.nbytes[batch_rows] // SLOT_ALIGNMENT) * SLOT_ALIGNMENT
        slot_offset = np.cumsum(spans) - spans
        xor, xor_table = _xor_plan(delta_map, batch_rows, slot_offset, device)
        layouts.append(
            _Layout(
                batch_rows,
                slot_offset,
                int(slot_offset[-1] + spans[-1]),
                xor,
                xor_table,
            )
        )
    return layouts


def _xor_plan(delta_map, table_rows, slot_offset, device):
    inventory, entries = delta_map.inventory, delta_map.table.entries
    names = delta_map.table.names
    end = int(slot_offset[-1] + delta_map.table.nbytes[table_rows[-1]])
    # Sources view a meta stand-in for the slot, which exists only in the pause.
    slot = torch.empty(end, dtype=torch.uint8, device="meta")
    items = []
    for row, offset in zip(table_rows.tolist(), slot_offset.tolist()):
        tensor = inventory[names[row].decode()]
        nbytes = int(delta_map.table.nbytes[row])
        canonical = (
            slot[offset : offset + nbytes]
            .view(TORCH_DTYPES[tensor.dtype])
            .reshape(tensor.shape)
        )
        for place in entries[row]:
            source = _bytes(canonical[place.src])
            start = source.storage_offset() - offset
            items.append((source, _bytes(place.dst), start, row))
    group = plan_xor(items)
    group.prepare_metadata(device.index)
    pointers = np.fromiter(group.pointer_rows(0), dtype=np.int64)
    return group, np.concatenate(
        [pointers, np.asarray(group.metadata_rows, dtype=np.int64)]
    )


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
