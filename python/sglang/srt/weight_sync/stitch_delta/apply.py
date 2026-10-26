"""Inside the pause: decode each batch on the decompression engine into one of two
scratch slots and XOR it into the serving tensors while the next batch decodes; then the
raw tensors and the copies. Any failure raises; the caller lets it take the engine down.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from sglang.srt.weight_sync.stitch_delta.fetch import Fetched
from sglang.srt.weight_sync.stitch_delta.map import DeltaMap
from sglang.srt.weight_sync.stitch_delta.memory import (
    device_empty,
    record_device_stream,
)


def apply(fetched: Fetched, delta_map: DeltaMap, device: torch.device) -> dict:
    started = time.perf_counter()
    # An in-place pause leaves the last launched batch running.
    torch.cuda.synchronize(device)
    synced = time.perf_counter()
    try:
        with torch.cuda.nvtx.range("gpu_delta.apply"):
            return _apply(fetched, delta_map, device, started, synced)
    except BaseException:
        # Nothing may be freed under a running decode or copy.
        torch.cuda.synchronize(device)
        raise


def _apply(fetched, delta_map, device, started, synced) -> dict:
    batches = fetched.batches
    free_bytes = torch.cuda.mem_get_info(device)[0]
    slots = [
        device_empty(fetched.scratch_bytes, torch.uint8, device)
        for _ in range(min(2, len(batches)))
    ]

    tables = []
    for index, batch in enumerate(batches):
        slot = slots[index % 2]
        batch.decode.bind(slot)
        if not batch.xor.bind(slot):
            raise RuntimeError("decode slot is not 16-byte aligned")
        table = batch.xor_table.copy()
        table[list(batch.xor.source_indices)] += slot.data_ptr()
        tables.append(table)
    rows = torch.from_numpy(np.concatenate(tables)).to(device) if tables else None
    launches, position = [], 0
    for batch, table in zip(batches, tables):
        pointers = 2 * len(batch.xor.sources)
        view = rows[position : position + len(table)]
        launches.append(
            batch.xor.bind_launch(view[:pointers], view[pointers:], fetched.error)
        )
        position += len(table)
    bound = time.perf_counter()

    current = torch.cuda.current_stream(device)
    decode_stream, xor_stream = torch.cuda.Stream(device), torch.cuda.Stream(device)
    ready = [torch.cuda.Event() for _ in slots]
    free = [torch.cuda.Event() for _ in slots]
    begin, decoded, applied = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    begin.record(current)
    for stream in (decode_stream, xor_stream):
        stream.wait_stream(current)
        for slot in slots:
            record_device_stream(slot, stream)

    def decode(index):
        batch, slot = batches[index], slots[index % 2]
        with torch.cuda.stream(decode_stream):
            if index >= 2:
                decode_stream.wait_event(free[index % 2])
            if batch.zero_ranges:
                torch._foreach_zero_([slot[o : o + n] for o, n in batch.zero_ranges])
            batch.decode.enqueue(decode_stream)
            ready[index % 2].record(decode_stream)

    with torch.cuda.stream(xor_stream):
        # Raw targets never overlap XOR targets, so they go beside the first decode. The
        # values are already on the GPU.
        for value, places in fetched.raw:
            for place in places:
                place.dst.copy_(value[place.src])
    if batches:
        decode(0)
    for index, launch in enumerate(launches):
        with torch.cuda.stream(xor_stream):
            xor_stream.wait_event(ready[index % 2])
            batches[index].status_check()
            if launch is not None:
                launch()
            free[index % 2].record(xor_stream)
        # After this XOR is queued: nvCOMP may wait for earlier work on its stream.
        if index + 1 < len(batches):
            decode(index + 1)
    decoded.record(decode_stream)
    with torch.cuda.stream(xor_stream):
        for copy in delta_map.copies:
            copy.dst.copy_(copy.src)
        applied.record(xor_stream)
    enqueued = time.perf_counter()
    applied.synchronize()
    finished = time.perf_counter()
    if fetched.error.item():
        raise RuntimeError(
            "a Snappy frame failed to decode or decoded to the wrong size"
        )
    return {
        "batches": len(batches),
        "slot_bytes": fetched.scratch_bytes if slots else 0,
        "free_bytes_before_slots": free_bytes,
        "sync_s": synced - started,
        "bind_s": bound - synced,
        "enqueue_s": enqueued - bound,
        "wait_s": finished - enqueued,
        "total_s": finished - started,
        "gpu_decode_s": begin.elapsed_time(decoded) / 1e3,
        "gpu_apply_s": begin.elapsed_time(applied) / 1e3,
    }
