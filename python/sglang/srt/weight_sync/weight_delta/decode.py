"""Inside the pause: decode a batch's frames from a GPU input buffer into its decode slot."""

from __future__ import annotations

import ctypes
import importlib.metadata
from functools import cache

import numpy as np
import torch
import triton
import triton.language as tl

GAP_NBYTES = 3
MAX_FRAME_NBYTES = 4 << 20
HARDWARE, CUDA_KERNELS = 1, 2  # nvcompDecompressBackend_t


class GapDecoder:
    """Codec gap-zstd. A frame of n changes is 4n bytes: each changed position's gap from the
    previous one (the first from 0) as a little-endian u24 in three byte planes, then the XOR
    value at each. Decoding writes the values into a zeroed slot, which is then XORed like a
    Snappy-decoded one; a position past the frame flags its status."""

    whole_frames = False
    temp_nbytes = 0

    def check(self, frames: np.ndarray) -> None:
        """frames: int64 rows (input offset, encoded bytes, decoded bytes, output offset)."""
        _check_frames(frames)
        if np.any(frames[1] % (GAP_NBYTES + 1)):
            raise ValueError("a gap frame's length is not a whole number of changes")

    def metadata(self, frames: np.ndarray, input_address: int, output_address: int):
        metadata = frames.copy()
        metadata[0] += input_address
        metadata[1] //= GAP_NBYTES + 1
        metadata[2], metadata[3] = frames[3] + output_address, frames[2]
        return metadata

    def expected_sizes(self, metadata: torch.Tensor) -> torch.Tensor:
        return metadata[3]

    def enqueue(self, metadata, statuses, actual_sizes, temp, count: int) -> None:
        actual_sizes.copy_(metadata[3])
        _gap_decode[(count,)](metadata, statuses, count, BLOCK=1024)


# The decode kernels' pointers are slices at offsets each version's frames set: specialized on
# alignment, a version with a new alignment would compile a kernel in the pause.
@triton.jit(
    do_not_specialize=["count"],
    do_not_specialize_on_alignment=["metadata", "statuses"],
)
def _gap_decode(metadata, statuses, count, BLOCK: tl.constexpr):
    frame = tl.program_id(0)
    source = tl.load(metadata + frame).to(tl.pointer_type(tl.uint8))
    changes = tl.load(metadata + count + frame)
    target = tl.load(metadata + 2 * count + frame).to(tl.pointer_type(tl.uint8))
    limit = tl.load(metadata + 3 * count + frame)
    base = changes * 0
    bad = (changes * 0).to(tl.int32)
    for start in range(0, changes, BLOCK):
        index = start + tl.arange(0, BLOCK)
        live = index < changes
        gap = tl.load(source + index, mask=live, other=0).to(tl.int64)
        gap |= tl.load(source + changes + index, mask=live, other=0).to(tl.int64) << 8
        gap |= (
            tl.load(source + 2 * changes + index, mask=live, other=0).to(tl.int64) << 16
        )
        value = tl.load(source + 3 * changes + index, mask=live, other=0)
        position = base + tl.cumsum(gap, 0)
        tl.store(target + position, value, mask=live & (position < limit))
        bad |= tl.max((live & (position >= limit)).to(tl.int32), 0)
        base += tl.sum(gap, 0)
    tl.store(statuses + frame, bad)


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
    """Codec snappy-zstd, through nvCOMP's C API: on the decompression engine where the GPU has
    one (Blackwell), otherwise on nvCOMP's CUDA kernels. The backend is always named: nvCOMP's
    default may silently pick its software decoder on a GPU that has the engine. Adapted from
    sglang#42210 (weight_sync/gpu_delta/codec.py)."""

    whole_frames = True

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
        self.maximum_frame_nbytes = MAX_FRAME_NBYTES
        if self.hardware:
            self.maximum_frame_nbytes = min(
                MAX_FRAME_NBYTES, _hardware_limit(self.device.index)
            )
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
            np.any(decoded > self.maximum_frame_nbytes)
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
    """Compile the decode kernels before the first update needs them."""
    encoded = torch.zeros(4, dtype=torch.uint8, device=device)
    decoded = torch.zeros(4, dtype=torch.uint8, device=device)
    metadata = torch.tensor(
        [[encoded.data_ptr()], [1], [decoded.data_ptr()], [4]],
        dtype=torch.int64,
        device=device,
    )
    statuses = torch.zeros(1, dtype=torch.int32, device=device)
    _gap_decode[(1,)](metadata, statuses, 1, BLOCK=1024)
    check_decode[(1,)](statuses, metadata[3], metadata[3], statuses, 1, BLOCK=1024)
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
