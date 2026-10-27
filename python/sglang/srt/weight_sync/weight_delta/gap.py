"""Codec gap-zstd: each changed frame lists only its changed bytes. A frame of n changes is
5n bytes: the gaps between its changed in-frame positions (the first from 0) as little-endian
u32 in four byte planes, then the XOR value at each. Decoding writes those values into a zeroed
scratch frame, which the XOR step then applies like a Snappy-decoded one; the plan has the
interface of nvcomp.DecodePlan, staging its frames on the GPU first."""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.weight_sync.weight_delta.memory import device_empty, record_device_stream

BLOCK = 1024


@triton.jit(do_not_specialize=["count"])
def _gap_decode(rows, statuses, count, BLOCK: tl.constexpr):
    frame = tl.program_id(0)
    source = tl.load(rows + frame).to(tl.pointer_type(tl.uint8))
    n = tl.load(rows + count + frame)
    target = tl.load(rows + 2 * count + frame).to(tl.pointer_type(tl.uint8))
    limit = tl.load(rows + 3 * count + frame)
    base = n * 0
    bad = (n * 0).to(tl.int32)
    for start in range(0, n, BLOCK):
        index = start + tl.arange(0, BLOCK)
        live = index < n
        gap = tl.load(source + index, mask=live, other=0).to(tl.int64)
        gap |= tl.load(source + n + index, mask=live, other=0).to(tl.int64) << 8
        gap |= tl.load(source + 2 * n + index, mask=live, other=0).to(tl.int64) << 16
        gap |= tl.load(source + 3 * n + index, mask=live, other=0).to(tl.int64) << 24
        value = tl.load(source + 4 * n + index, mask=live, other=0)
        position = base + tl.cumsum(gap, 0)
        tl.store(target + position, value, mask=live & (position < limit))
        bad |= tl.max((live & (position >= limit)).to(tl.int32), 0)
        base += tl.sum(gap, 0)
    tl.store(statuses + frame, bad)


class GapDecoder:
    hardware = False

    def __init__(self, device: torch.device):
        self.device = torch.device(device)

    def plan(self, frames: np.ndarray, host: torch.Tensor) -> GapDecodePlan:
        """frames: as for nvcomp.SnappyDecoder.plan."""
        inputs, encoded, decoded, outputs = frames
        if frames.shape[1] and (
            np.any(encoded <= 0)
            or np.any(encoded % 5)
            or np.any(decoded <= 0)
            or np.any(inputs < 0)
            or np.any(inputs + encoded > host.numel())
            or np.any(outputs[1:] < outputs[:-1] + decoded[:-1])
        ):
            raise ValueError("gap frames out of bounds, malformed or overlapping")
        return GapDecodePlan(self, frames, host)


class GapDecodePlan:
    def __init__(self, decoder: GapDecoder, frames: np.ndarray, host):
        self.decoder, self.host, self.count = decoder, host, frames.shape[1]
        self.output_offsets = frames[3].copy()
        self.output_bound = int((frames[3] + frames[2]).max()) if self.count else 0
        self.input_start = int(frames[0].min()) if self.count else 0
        self.input_bytes = (
            int((frames[0] + frames[1]).max()) - self.input_start if self.count else 0
        )
        self.input_offsets = frames[0] - self.input_start
        device = decoder.device
        rows = np.empty((4, self.count), dtype=np.int64)
        rows[1] = frames[1] // 5
        rows[3] = frames[2]
        self.host_rows = torch.from_numpy(rows).pin_memory()
        self.rows = device_empty((4, self.count), torch.int64, device)
        self.rows[1:4:2].copy_(self.host_rows[1:4:2], non_blocking=True)
        self.temp = device_empty((0,), torch.uint8, device)
        self.statuses = device_empty((self.count,), torch.int32, device)
        self.expected_sizes = self.rows[3]
        # The kernel reports out-of-frame positions through the statuses alone.
        self.actual_sizes = device_empty((self.count,), torch.int64, device)
        self.actual_sizes.copy_(self.rows[3], non_blocking=True)

    def bind(self, output: torch.Tensor, staging: torch.Tensor | None = None) -> None:
        if output.numel() < self.output_bound:
            raise ValueError("decode output buffer is too small")
        if staging is None or staging.numel() < self.input_bytes:
            raise ValueError("decode staging buffer is missing or too small")
        self.host_rows[0] = torch.from_numpy(self.input_offsets + staging.data_ptr())
        self.host_rows[2] = torch.from_numpy(self.output_offsets + output.data_ptr())
        self.rows[0:3:2].copy_(self.host_rows[0:3:2], non_blocking=True)

    def stage(self, staging: torch.Tensor) -> None:
        if self.input_bytes:
            end = self.input_start + self.input_bytes
            staging[: self.input_bytes].copy_(
                self.host[self.input_start : end], non_blocking=True
            )

    def enqueue(self, stream: torch.cuda.Stream) -> None:
        if not self.count:
            return
        for tensor in (self.rows, self.statuses, self.actual_sizes):
            record_device_stream(tensor, stream)
        with torch.cuda.stream(stream):
            _gap_decode[(self.count,)](self.rows, self.statuses, self.count, BLOCK)


def warm_up(device_index: int) -> None:
    """Compile the decode kernel before the first update needs it."""
    device = torch.device("cuda", device_index)
    record = torch.zeros(5, dtype=torch.uint8, device=device)
    out = torch.zeros(4, dtype=torch.uint8, device=device)
    rows = torch.tensor(
        [[record.data_ptr()], [1], [out.data_ptr()], [4]], dtype=torch.int64, device=device
    )
    statuses = torch.zeros(1, dtype=torch.int32, device=device)
    _gap_decode[(1,)](rows, statuses, 1, BLOCK)
    torch.cuda.synchronize(device)
