"""Batched Snappy decode on the Blackwell decompression engine, through nvCOMP's C API.

Adapted from sglang#42210 (weight_sync/gpu_delta/codec.py). The hardware backend is
forced: nvCOMP's default may silently pick its software decoder. A plan is built before
the pause; only the output addresses are filled in once the scratch buffer exists.
"""

from __future__ import annotations

import ctypes
import importlib.metadata

import numpy as np
import torch

from sglang.srt.weight_sync.stitch_delta.memory import (
    _driver,
    device_empty,
    record_device_stream,
    require_de_capable,
)

MAX_FRAME_BYTES = 4 << 20


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


def _hardware_limit(device: torch.device) -> int:
    driver = _driver()
    attributes = driver.CUdevice_attribute
    status, mask = driver.cuDeviceGetAttribute(
        attributes.CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK, device.index
    )
    if status:
        raise RuntimeError(f"decompression engine query failed: CUDA {status}")
    if not mask & driver.CUmemDecompressAlgorithm.CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY:
        raise RuntimeError("this GPU has no hardware Snappy decompressor")
    status, maximum = driver.cuDeviceGetAttribute(
        attributes.CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH, device.index
    )
    if status or maximum <= 0:
        raise RuntimeError("this GPU reports no hardware decompression limit")
    return maximum


class SnappyDecoder:
    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        cuda_major = torch.version.cuda.split(".")[0]
        package = importlib.metadata.distribution(f"nvidia-libnvcomp-cu{cuda_major}")
        version = tuple(int(v) for v in package.version.split(".")[:2])
        if not (5, 3) <= version < (6, 0):
            raise RuntimeError(f"GPU delta needs nvCOMP 5.3+, found {package.version}")
        self.maximum_frame_bytes = min(MAX_FRAME_BYTES, _hardware_limit(self.device))
        self._options = _Options(backend=1)  # 1 = hardware decompression engine
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
            [
                pointer,
                pointer,
                pointer,
                pointer,
                size,
                pointer,
                size,
                pointer,
                _Options,
                pointer,
                pointer,
            ],
        )
        self.alignments = _Alignments()
        self._check(
            bind("GetRequiredAlignments", [_Options, ctypes.POINTER(_Alignments)])(
                self._options, ctypes.byref(self.alignments)
            )
        )

    @staticmethod
    def _check(status):
        if status != 0:
            raise RuntimeError(f"nvCOMP Snappy (hardware) failed: status {status}")

    def plan(self, frames: np.ndarray, host: torch.Tensor) -> DecodePlan:
        """frames: int64 rows (input offset, encoded bytes, decoded bytes, output offset),
        one column per frame; input offsets are into `host`, output offsets into the
        output buffer given to DecodePlan.bind."""
        inputs, encoded, decoded, outputs = frames
        count = frames.shape[1]
        if count and (
            np.any(encoded <= 0)
            or np.any(decoded <= 0)
            or np.any(decoded > self.maximum_frame_bytes)
            or np.any(inputs < 0)
            or np.any(inputs + encoded > host.numel())
            or np.any((inputs + host.data_ptr()) % self.alignments.input)
            or np.any(outputs % self.alignments.output)
            or np.any(outputs[1:] < outputs[:-1] + decoded[:-1])
        ):
            raise ValueError("Snappy frames out of bounds, misaligned or overlapping")
        temp = ctypes.c_size_t(0)
        if count:
            with torch.cuda.device(self.device):
                self._check(
                    self._temp_size(
                        count,
                        int(decoded.max()),
                        self._options,
                        ctypes.byref(temp),
                        int(decoded.sum()),
                    )
                )
        return DecodePlan(self, frames, host, temp.value)


class DecodePlan:
    def __init__(self, decoder: SnappyDecoder, frames: np.ndarray, host, temp_bytes):
        self.decoder, self.host, self.count = decoder, host, frames.shape[1]
        self.output_offsets = frames[3].copy()
        self.output_bound = int((frames[3] + frames[2]).max()) if self.count else 0
        device = decoder.device
        rows = np.empty((4, self.count), dtype=np.int64)
        rows[0] = frames[0] + host.data_ptr()
        rows[1:3] = frames[1:3]
        self.host_rows = torch.from_numpy(rows).pin_memory()
        self.rows = device_empty((4, self.count), torch.int64, device)
        self.temp = device_empty((temp_bytes,), torch.uint8, device)
        self.statuses = device_empty((self.count,), torch.int32, device)
        self.actual_sizes = device_empty((self.count,), torch.int64, device)
        self.expected_sizes = self.rows[2]
        self.rows[:3].copy_(self.host_rows[:3], non_blocking=True)

    def bind(self, output: torch.Tensor) -> None:
        """Point every frame at `output` (a decompression-engine buffer) plus its offset."""
        if output.numel() < self.output_bound:
            raise ValueError("decode output buffer is too small")
        require_de_capable(output.data_ptr())
        self.host_rows[3] = torch.from_numpy(self.output_offsets + output.data_ptr())
        self.rows[3].copy_(self.host_rows[3], non_blocking=True)

    def enqueue(self, stream: torch.cuda.Stream) -> None:
        """Launch only; check statuses and actual sizes on the device afterwards."""
        if not self.count:
            return
        for tensor in (self.rows, self.temp, self.statuses, self.actual_sizes):
            record_device_stream(tensor, stream)
        rows = self.rows
        self.decoder._check(
            self.decoder._decode(
                rows[0].data_ptr(),
                rows[1].data_ptr(),
                rows[2].data_ptr(),
                self.actual_sizes.data_ptr(),
                self.count,
                self.temp.data_ptr(),
                self.temp.numel(),
                rows[3].data_ptr(),
                self.decoder._options,
                self.statuses.data_ptr(),
                stream.cuda_stream,
            )
        )
