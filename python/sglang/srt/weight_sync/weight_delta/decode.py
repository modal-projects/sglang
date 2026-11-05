"""Inside the pause: XOR gap frames into serving tensors, or decode Snappy frames into slots."""

from __future__ import annotations

import ctypes
import importlib.metadata
from dataclasses import dataclass
from functools import cache

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.weight_sync.weight_delta.map import TORCH_DTYPES, DeltaMap
from sglang.srt.weight_sync.weight_delta.xor_kernel import checksum_terms

GAP_NBYTES = 3
MAX_FRAME_NBYTES = 4 << 20
HARDWARE, CUDA_KERNELS = 1, 2  # nvcompDecompressBackend_t
SCATTER_BLOCK = 1024


@dataclass(frozen=True)
class Places:
    """The places a held tensor can arrive XORed to; row r's are [start[r], start[r] + count[r])."""

    # Per place, int64: table row, serving address, item size, region axes, view axes, then
    # (extent, first, length, step) per region axis and (size, stride) per view axis.
    records: torch.Tensor
    record_offset: np.ndarray  # each place's record, in records
    low: np.ndarray  # each place holds the canonical bytes [low, high)
    high: np.ndarray
    start: np.ndarray
    count: np.ndarray


def gap_places(delta_map: DeltaMap, device: torch.device) -> Places:
    """Once at startup."""
    table = delta_map.table
    words, record_offset, low, high = [], [], [], []
    start = np.zeros(len(table.names), np.int64)
    count = np.zeros(len(table.names), np.int64)
    shape_start = np.concatenate(([0], table.shape_end[:-1]))
    for row, entry in enumerate(table.entries):
        start[row] = len(record_offset)
        if not isinstance(entry, tuple) or table.raw_only[row]:
            continue
        shape = tuple(table.shape[shape_start[row] : table.shape_end[row]].tolist())
        itemsize = TORCH_DTYPES[table.dtypes[row]].itemsize
        for place in entry:
            region, first, last = _region_axes(shape, place.src)
            view = _view_axes(place.dst)
            record_offset.append(len(words))
            low.append(first * itemsize)
            high.append((last + 1) * itemsize)
            words += [row, place.dst.data_ptr(), itemsize, len(region), len(view)]
            words += [value for axis in region + view for value in axis]
        count[row] = len(record_offset) - start[row]
    return Places(
        torch.tensor(words or [0], dtype=torch.int64, device=device),
        np.array(record_offset, np.int64),
        np.array(low, np.int64),
        np.array(high, np.int64),
        start,
        count,
    )


def _region_axes(shape, src) -> tuple[list[list[int]], int, int]:
    """(extent, first, length, step) axes, a wholly covered axis folded into the one outside it."""
    axes = []
    for extent, part in zip(shape, src, strict=True):
        first, stop, step = part.indices(extent)
        if step != 1:
            raise ValueError("a place's region must step by one along every axis")
        if extent != 1:
            axes.append([extent, first, stop - first])
    axis = len(axes) - 1
    while axis > 0:
        extent, first, length = axes[axis]
        if first == 0 and length == extent:
            outer = axes.pop(axis - 1)
            axes[axis - 1] = [outer[0] * extent, outer[1] * extent, outer[2] * extent]
        axis -= 1
    axes = axes or [[1, 0, 1]]
    first = last = 0
    for extent, start, length in axes:
        first, last = first * extent + start, last * extent + start + length - 1
    step = 1
    for axis in reversed(axes):
        axis.append(step)
        step *= axis[2]
    return axes, first, last


def _view_axes(view: torch.Tensor) -> list[list[int]]:
    """A serving view as (size, stride) axes in elements, contiguous neighbours merged."""
    axes = []
    for size, stride in zip(view.shape, view.stride()):
        if size == 1:
            continue
        if stride == 0:
            raise ValueError("a serving view repeats an element")
        if axes and axes[-1][1] == size * stride:
            axes[-1] = [axes[-1][0] * size, stride]
        else:
            axes.append([size, stride])
    return axes or [[1, 1]]


def check_gap_frames(inner_nbytes: np.ndarray, nbytes: np.ndarray) -> None:
    if np.any(inner_nbytes <= 0) or np.any(nbytes <= 0):
        raise ValueError("a gap frame is empty")
    if np.any(inner_nbytes % (GAP_NBYTES + 1)):
        raise ValueError("a gap frame's length is not a whole number of changes")


def gap_scatter(
    items: torch.Tensor, places: Places, error, checksums, count: int, checked: bool
):
    """items: int64 rows of inner address, changes, frame offset, frame nbytes, record."""
    if count:
        _gap_scatter[(count,)](
            items,
            places.records,
            error,
            checksums,
            count,
            BLOCK=SCATTER_BLOCK,
            CHECKSUM=checked,
        )


# Not specialized on alignment: a version with new offsets would compile a kernel in the pause.
@triton.jit(
    do_not_specialize=["count"],
    do_not_specialize_on_alignment=["items", "records", "error", "checksums"],
)
def _gap_scatter(
    items, records, error, checksums, count, BLOCK: tl.constexpr, CHECKSUM: tl.constexpr
):
    """A frame of n changes: n little-endian u24 gaps in three byte planes, then n XOR values."""
    item = tl.program_id(0)
    source = tl.load(items + item).to(tl.pointer_type(tl.uint8))
    changes = tl.load(items + count + item)
    start = tl.load(items + 2 * count + item)
    limit = tl.load(items + 3 * count + item)
    record = tl.load(items + 4 * count + item)
    row = tl.load(records + record)
    target = tl.load(records + record + 1).to(tl.pointer_type(tl.uint8))
    itemsize = tl.load(records + record + 2)
    region_axes = tl.load(records + record + 3)
    view_axes = tl.load(records + record + 4)
    region = record + 5
    view = region + 4 * region_axes
    base = changes * 0
    bad = (changes * 0).to(tl.int32)
    sums = tl.zeros((BLOCK,), tl.uint64)
    for first in range(0, changes, BLOCK):
        index = first + tl.arange(0, BLOCK)
        live = index < changes
        gap = tl.load(source + index, mask=live, other=0).to(tl.int64)
        gap |= tl.load(source + changes + index, mask=live, other=0).to(tl.int64) << 8
        gap |= (
            tl.load(source + 2 * changes + index, mask=live, other=0).to(tl.int64) << 16
        )
        value = tl.load(source + 3 * changes + index, mask=live, other=0)
        local = base + tl.cumsum(gap, 0)
        bad |= tl.max((live & (local >= limit)).to(tl.int32), 0)
        position = start + local
        element = position // itemsize
        # The element's row-major index in the place's region, if the region holds it.
        member = live & (local < limit)
        rest = element
        offset = element * 0
        axis = region_axes - 1
        while axis > 0:
            extent = tl.load(records + region + 4 * axis)
            coordinate = rest % extent - tl.load(records + region + 4 * axis + 1)
            member &= (coordinate >= 0) & (
                coordinate < tl.load(records + region + 4 * axis + 2)
            )
            offset += coordinate * tl.load(records + region + 4 * axis + 3)
            rest = rest // extent
            axis -= 1
        coordinate = rest - tl.load(records + region + 1)
        member &= (coordinate >= 0) & (coordinate < tl.load(records + region + 2))
        offset += coordinate * tl.load(records + region + 3)
        # That index's element of the serving view, row-major over the view's own shape.
        rest = offset
        address = offset * 0
        axis = view_axes - 1
        while axis > 0:
            size = tl.load(records + view + 2 * axis)
            address += (rest % size) * tl.load(records + view + 2 * axis + 1)
            rest = rest // size
            axis -= 1
        address += rest * tl.load(records + view + 1)
        address = address * itemsize + position - element * itemsize
        previous = tl.load(target + address, mask=member, other=0)
        new = previous ^ value
        tl.store(target + address, new, mask=member)
        if CHECKSUM:
            terms = checksum_terms(
                position, new.to(tl.uint32) << ((position & 3) * 8).to(tl.uint32)
            )
            sums += terms * (member & (value != 0)).to(tl.uint64)
        base += tl.sum(gap, 0)
    if bad != 0:
        tl.atomic_or(error, 1)
    if CHECKSUM:
        tl.atomic_add(checksums + row, tl.sum(sums, axis=0).to(tl.int64, bitcast=True))


class _Options(ctypes.Structure):
    _fields_ = [
        ("backend", ctypes.c_int),
        ("sort_before_hw_decompress", ctypes.c_int),
        ("reserved", ctypes.c_char * 56),
    ]


class _Alignments(ctypes.Structure):
    _fields_ = [
        ("input", ctypes.c_size_t),
        ("output", ctypes.c_size_t),
        ("temp", ctypes.c_size_t),
    ]


class SnappyDecoder:
    """Codec snappy-zstd via nvCOMP's C API; from sglang#42210's weight_sync/gpu_delta/codec.py."""

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        cuda_major = torch.version.cuda.split(".")[0]
        package = importlib.metadata.distribution(f"nvidia-libnvcomp-cu{cuda_major}")
        version = tuple(int(part) for part in package.version.split(".")[:2])
        if not (5, 3) <= version < (6, 0):
            raise RuntimeError(
                f"Weight delta needs nvCOMP 5.3, found {package.version}"
            )
        self.hardware = has_decompression_engine(self.device.index)
        self.max_frame_nbytes = MAX_FRAME_NBYTES
        if self.hardware:
            self.max_frame_nbytes = min(
                MAX_FRAME_NBYTES, _hardware_limit(self.device.index)
            )
        # Always named: nvCOMP's default may pick its software decoder on a GPU with the engine.
        self._options = _Options(backend=HARDWARE if self.hardware else CUDA_KERNELS)
        library = ctypes.CDLL(
            str(package.locate_file("nvidia/libnvcomp/lib64/libnvcomp.so.5"))
        )
        pointer, size = ctypes.c_void_p, ctypes.c_size_t

        def bind(suffix, arguments):
            function = getattr(library, f"nvcompBatchedSnappyDecompress{suffix}")
            function.argtypes, function.restype = arguments, ctypes.c_int
            return function

        self._temp_size = bind(
            "GetTempSizeAsync", [size, size, _Options, ctypes.POINTER(size), size]
        )
        self._decode = bind(
            "Async",
            [pointer, pointer, pointer, pointer, size]
            + [pointer, size, pointer, _Options, pointer, pointer],
        )
        self.alignments = _Alignments()
        self._check(
            bind("GetRequiredAlignments", [_Options, ctypes.POINTER(_Alignments)])(
                self._options, ctypes.byref(self.alignments)
            )
        )
        self.temp_nbytes = 0

    def check(self, frames: np.ndarray) -> None:
        _check_frames(frames)
        inputs, _, decoded, outputs = frames
        if (
            np.any(decoded > self.max_frame_nbytes)
            or np.any(inputs % self.alignments.input)
            or np.any(outputs % self.alignments.output)
        ):
            raise ValueError("Snappy frames too large or misaligned")
        if frames.shape[1]:
            temp = ctypes.c_size_t(0)
            with torch.cuda.device(self.device):
                self._check(
                    self._temp_size(
                        frames.shape[1],
                        int(decoded.max()),
                        self._options,
                        ctypes.byref(temp),
                        int(decoded.sum()),
                    )
                )
            self.temp_nbytes = max(self.temp_nbytes, temp.value)

    def metadata(self, frames: np.ndarray, input_address: int, output_address: int):
        metadata = frames.copy()
        metadata[0] += input_address
        metadata[3] += output_address
        return metadata

    def expected_sizes(self, metadata: torch.Tensor) -> torch.Tensor:
        return metadata[2]

    def enqueue(self, metadata, statuses, actual_sizes, temp, count: int) -> None:
        self._check(
            self._decode(
                metadata[0].data_ptr(),
                metadata[1].data_ptr(),
                metadata[2].data_ptr(),
                actual_sizes.data_ptr(),
                count,
                temp.data_ptr(),
                temp.numel(),
                metadata[3].data_ptr(),
                self._options,
                statuses.data_ptr(),
                torch.cuda.current_stream(self.device).cuda_stream,
            )
        )

    def _check(self, status: int) -> None:
        if status != 0:
            backend = "hardware" if self.hardware else "CUDA"
            raise RuntimeError(f"nvCOMP Snappy ({backend}) failed: status {status}")


@triton.jit(
    do_not_specialize=["count"],
    do_not_specialize_on_alignment=["statuses", "actual_sizes", "expected_sizes"],
)
def check_decode(
    statuses, actual_sizes, expected_sizes, error, count, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < count
    status = tl.load(statuses + index, active, other=0)
    actual = tl.load(actual_sizes + index, active, other=0)
    expected = tl.load(expected_sizes + index, active, other=0)
    if tl.sum(((status != 0) | (actual != expected)).to(tl.int32), 0) != 0:
        tl.atomic_or(error, 1)


def warm_up(device: torch.device) -> None:
    """Compile every kernel at startup, so none compiles in a pause."""
    inner = torch.zeros(4, dtype=torch.uint8, device=device)
    target = torch.zeros(4, dtype=torch.uint8, device=device)
    records = torch.tensor(
        [0, target.data_ptr(), 1, 1, 1, 4, 0, 4, 1, 4, 1],
        dtype=torch.int64,
        device=device,
    )
    items = torch.tensor(
        [inner.data_ptr(), 1, 0, 4, 0], dtype=torch.int64, device=device
    )
    error = torch.zeros(1, dtype=torch.int32, device=device)
    checksums = torch.zeros(1, dtype=torch.int64, device=device)
    for checked in (True, False):
        _gap_scatter[(1,)](
            items, records, error, checksums, 1, BLOCK=SCATTER_BLOCK, CHECKSUM=checked
        )
    sizes = torch.zeros(1, dtype=torch.int64, device=device)
    check_decode[(1,)](error, sizes, sizes, error, 1, BLOCK=1024)
    torch.cuda.synchronize(device)


def _check_frames(frames: np.ndarray) -> None:
    inputs, encoded, decoded, outputs = frames
    if frames.shape[1] and (
        np.any(encoded <= 0)
        or np.any(decoded <= 0)
        or np.any(inputs < 0)
        or np.any(outputs[1:] < outputs[:-1] + decoded[:-1])
    ):
        raise ValueError("frames empty, out of bounds or overlapping")


@cache
def has_decompression_engine(device_index: int) -> bool:
    driver = _driver()
    mask = _attribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK,
        device_index,
    )
    return bool(
        mask & driver.CUmemDecompressAlgorithm.CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY
    )


def _hardware_limit(device_index: int) -> int:
    driver = _driver()
    limit = _attribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH,
        device_index,
    )
    if limit <= 0:
        raise RuntimeError("this GPU reports no hardware decompression limit")
    return limit


def _attribute(attribute, device_index: int) -> int:
    status, value = _driver().cuDeviceGetAttribute(attribute, device_index)
    if status:
        raise RuntimeError(f"cuDeviceGetAttribute({attribute}) failed: CUDA {status}")
    return value


@cache
def _driver():
    from cuda.bindings import driver

    driver.cuInit(0)
    return driver
