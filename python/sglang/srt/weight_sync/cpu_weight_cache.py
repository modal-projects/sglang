"""Compile canonical checkpoints into rank-ready CPU weight images.

Staging deliberately reuses the model's ordinary loader and quantization
hooks. The canonical checkpoint can remain in memory shared by a model
replica's local workers or be materialized on host-local disk. Compilation
writes TP-sharded tensors into the persistent rank image, then runs reload-safe
post-load transforms on CPU. Quantization methods that require device kernels
retain the bounded GPU-staging path. The resulting runtime bytes remain in
:class:`~sglang.srt.weight_sync.cpu_weight_image.CPUWeightImage`; the live
model is never rebound or overwritten during staging.

Each delta is verified in canonical checkpoint space before the staged image
becomes valid. Disk-backed staging shares each bounded target buffer between
checkpoint persistence and runtime compilation. Neither path depends on
tensor-level sparsity.
"""

from __future__ import annotations

import copy
import errno
import functools
import gc
import hashlib
import json
import logging
import math
import mmap
import os
import struct
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import safe_open

from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.utils import DEFERRED_WEIGHT_COPY_SAFE_ATTR
from sglang.srt.weight_sync.cpu_weight_image import (
    CPUWeightImage,
    iter_weight_tensors,
)
from sglang.srt.weight_sync.host_shared_memory import HostSharedMemoryBuffer

logger = logging.getLogger(__name__)

_POSITIONAL_IO_CHUNK_BYTES = 64 << 20
_DIRECT_IO_ALIGNMENT = 4096


def _safetensors_dtypes() -> dict[str, torch.dtype]:
    result = {
        "BOOL": torch.bool,
        "I8": torch.int8,
        "U8": torch.uint8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
        "C64": torch.complex64,
    }
    optional = {
        "U16": "uint16",
        "U32": "uint32",
        "U64": "uint64",
        "F8_E4M3": "float8_e4m3fn",
        "F8_E4M3FNUZ": "float8_e4m3fnuz",
        "F8_E5M2": "float8_e5m2",
        "F8_E5M2FNUZ": "float8_e5m2fnuz",
        "F8_E8M0": "float8_e8m0fnu",
        "F4": "float4_e2m1fn_x2",
    }
    for code, name in optional.items():
        dtype = getattr(torch, name, None)
        if dtype is not None:
            result[code] = dtype
    return result


_SAFETENSORS_DTYPES = _safetensors_dtypes()


@dataclass(frozen=True)
class _SafetensorsEntry:
    dtype: torch.dtype
    dtype_code: str
    shape: tuple[int, ...]
    relative_begin: int
    relative_end: int


@dataclass(frozen=True)
class _SafetensorsLayout:
    data_offset: int
    file_nbytes: int
    tensors: dict[str, _SafetensorsEntry]


def _parse_safetensors_layout(
    *,
    header_nbytes: int,
    header_bytes: bytes,
    file_nbytes: int,
) -> _SafetensorsLayout:
    data_offset = 8 + header_nbytes
    if (
        header_nbytes <= 0
        or len(header_bytes) != header_nbytes
        or data_offset > file_nbytes
    ):
        raise ValueError(
            "invalid safetensors header length: "
            f"header={header_nbytes} file={file_nbytes}"
        )
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not an object")
    header.pop("__metadata__", None)
    tensors = {}
    for name, metadata in header.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise ValueError("invalid safetensors tensor metadata")
        dtype_code = metadata.get("dtype")
        dtype = _SAFETENSORS_DTYPES.get(dtype_code)
        if dtype is None:
            raise TypeError(f"unsupported safetensors dtype {dtype_code!r}")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(shape, list)
            or not all(isinstance(value, int) and value >= 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
        ):
            raise ValueError(f"invalid safetensors metadata for {name!r}")
        relative_begin, relative_end = offsets
        begin = data_offset + relative_begin
        end = data_offset + relative_end
        if relative_begin < 0 or begin > end or end > file_nbytes:
            raise ValueError(
                f"safetensors offsets are out of bounds for {name!r}: {offsets}"
            )
        shape_tuple = tuple(shape)
        if dtype_code == "F4":
            if not shape_tuple or shape_tuple[-1] % 2:
                raise ValueError(
                    f"F4 tensor {name!r} must have an even final dimension"
                )
            tensor_shape = shape_tuple[:-1] + (shape_tuple[-1] // 2,)
        else:
            tensor_shape = shape_tuple
        expected_bytes = (
            math.prod(tensor_shape) * torch.empty((), dtype=dtype).element_size()
        )
        actual_bytes = relative_end - relative_begin
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"tensor byte size mismatch for {name!r}: "
                f"expected={expected_bytes} actual={actual_bytes}"
            )
        tensors[name] = _SafetensorsEntry(
            dtype=dtype,
            dtype_code=dtype_code,
            shape=tensor_shape,
            relative_begin=relative_begin,
            relative_end=relative_end,
        )
    cursor = 0
    for name, entry in sorted(
        tensors.items(),
        key=lambda item: (
            item[1].relative_begin,
            item[1].relative_end,
            item[0],
        ),
    ):
        if entry.relative_begin != cursor:
            relation = (
                "overlaps another tensor"
                if entry.relative_begin < cursor
                else "leaves a gap"
            )
            raise ValueError(
                f"safetensors range for {name!r} {relation}: "
                f"expected_begin={cursor} actual_begin={entry.relative_begin}"
            )
        cursor = entry.relative_end
    data_nbytes = file_nbytes - data_offset
    if cursor != data_nbytes:
        raise ValueError(
            "safetensors tensor ranges do not cover the data buffer: "
            f"covered={cursor} data={data_nbytes}"
        )
    return _SafetensorsLayout(
        data_offset=data_offset,
        file_nbytes=file_nbytes,
        tensors=tensors,
    )


def _read_safetensors_layout(path: Path) -> _SafetensorsLayout:
    with path.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError(f"safetensors source is shorter than its header: {path}")
        header_nbytes = int.from_bytes(prefix, "little")
        header_bytes = file.read(header_nbytes)
    return _parse_safetensors_layout(
        header_nbytes=header_nbytes,
        header_bytes=header_bytes,
        file_nbytes=path.stat().st_size,
    )


class _InMemorySafetensorsFile:
    """Expose safetensors views from one bounded CPU or CUDA byte tensor.

    The header is parsed once from the CPU source. The same validated layout is
    then used for views into the one bulk CUDA copy, avoiding thousands of
    small CPU-to-GPU tensor transfers.
    """

    def __init__(
        self,
        buffer: torch.Tensor,
        *,
        layout: _SafetensorsLayout | None = None,
    ):
        if (
            buffer.device.type not in {"cpu", "cuda"}
            or buffer.dtype != torch.uint8
            or buffer.ndim != 1
            or not buffer.is_contiguous()
        ):
            raise ValueError("safetensors source must be contiguous CPU or CUDA bytes")
        if layout is None:
            if buffer.device.type != "cpu":
                raise ValueError("safetensors headers must be parsed from CPU bytes")
            layout = self._parse_layout(buffer)
        if layout.file_nbytes != buffer.numel():
            raise ValueError(
                "safetensors layout file size differs from source buffer: "
                f"layout={layout.file_nbytes} buffer={buffer.numel()}"
            )
        if layout.data_offset > buffer.numel():
            raise ValueError(
                "safetensors layout exceeds source buffer: "
                f"data_offset={layout.data_offset} file={buffer.numel()}"
            )
        self.buffer = buffer
        self.layout = layout
        self.data_offset = layout.data_offset
        self.tensors = layout.tensors

    @staticmethod
    def _parse_layout(buffer: torch.Tensor) -> _SafetensorsLayout:
        if buffer.numel() < 8:
            raise ValueError("safetensors source is shorter than its header prefix")
        prefix = buffer[:8].numpy().tobytes()
        header_nbytes = int.from_bytes(prefix, "little")
        data_offset = 8 + header_nbytes
        return _parse_safetensors_layout(
            header_nbytes=header_nbytes,
            header_bytes=buffer[8:data_offset].numpy().tobytes(),
            file_nbytes=buffer.numel(),
        )

    def get_tensor(self, name: str) -> torch.Tensor:
        entry = self.tensors.get(name)
        if entry is None:
            raise KeyError(f"safetensors source has no tensor {name!r}")
        begin = self.data_offset + entry.relative_begin
        end = self.data_offset + entry.relative_end
        source = self.buffer[begin:end]
        try:
            tensor = source.view(entry.dtype).reshape(entry.shape)
            # The host-shared source remains immutable until every local rank
            # crosses the reuse barrier. Native loaders may therefore batch
            # independent CPU copies without retaining a reused stream buffer.
            setattr(tensor, DEFERRED_WEIGHT_COPY_SAFE_ATTR, True)
            return tensor
        except RuntimeError as exc:
            raise ValueError(f"cannot construct safetensors view for {name!r}") from exc

    def get_tensor_bytes(self, name: str) -> torch.Tensor:
        """Return the canonical encoded bytes for an in-place source transform."""

        entry = self.tensors.get(name)
        if entry is None:
            raise KeyError(f"safetensors source has no tensor {name!r}")
        begin = self.data_offset + entry.relative_begin
        end = self.data_offset + entry.relative_end
        return self.buffer[begin:end]


def _pread_file_to_tensor(
    path: Path,
    target: torch.Tensor,
    *,
    drop_cache_after_read: bool = False,
) -> float:
    if target.numel() != path.stat().st_size:
        raise ValueError(
            f"source buffer size mismatch for {path}: "
            f"buffer={target.numel()} file={path.stat().st_size}"
        )
    return _pread_range_to_tensor(
        path,
        target,
        file_offset=0,
        drop_cache_after_read=drop_cache_after_read,
    ).wall_s


@dataclass(frozen=True)
class _PositionalReadResult:
    wall_s: float
    direct_io: bool


def _pread_range_to_tensor(
    path: Path,
    target: torch.Tensor,
    *,
    file_offset: int,
    direct_io: bool = False,
    drop_cache_after_read: bool = False,
) -> _PositionalReadResult:
    if (
        target.device.type != "cpu"
        or target.dtype != torch.uint8
        or target.ndim != 1
        or not target.is_contiguous()
    ):
        raise ValueError("positional read target must be contiguous CPU bytes")
    if file_offset < 0 or file_offset + target.numel() > path.stat().st_size:
        raise ValueError(
            f"source range exceeds {path}: offset={file_offset} "
            f"bytes={target.numel()} file={path.stat().st_size}"
        )
    if direct_io and (
        file_offset % _DIRECT_IO_ALIGNMENT
        or target.numel() % _DIRECT_IO_ALIGNMENT
        or target.data_ptr() % _DIRECT_IO_ALIGNMENT
    ):
        raise ValueError(
            "direct positional reads require aligned file offsets, buffers, "
            f"and lengths: offset={file_offset} address={target.data_ptr()} "
            f"bytes={target.numel()}"
        )

    started = time.perf_counter()
    array = target.numpy()
    view = memoryview(array).cast("B")
    use_direct_io = direct_io and hasattr(os, "O_DIRECT")
    flags = os.O_RDONLY | (os.O_DIRECT if use_direct_io else 0)
    fd = None
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if not use_direct_io or exc.errno not in {
            errno.EINVAL,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
            errno.EPERM,
        }:
            view.release()
            raise
        use_direct_io = False
        try:
            fd = os.open(path, os.O_RDONLY)
        except Exception:
            view.release()
            raise
    offset = 0
    try:
        try:
            while offset < target.numel():
                end = min(offset + _POSITIONAL_IO_CHUNK_BYTES, target.numel())
                nread = os.preadv(fd, [view[offset:end]], file_offset + offset)
                if nread <= 0:
                    raise EOFError(
                        f"unexpected EOF reading {path}: "
                        f"offset={file_offset + offset} size={target.numel()}"
                    )
                offset += nread
        except OSError as exc:
            if not use_direct_io or exc.errno not in {
                errno.EINVAL,
                errno.ENOTSUP,
                errno.EOPNOTSUPP,
                errno.EPERM,
            }:
                raise
            os.close(fd)
            fd = None
            use_direct_io = False
            fd = os.open(path, os.O_RDONLY)
            offset = 0
            while offset < target.numel():
                end = min(offset + _POSITIONAL_IO_CHUNK_BYTES, target.numel())
                nread = os.preadv(fd, [view[offset:end]], file_offset + offset)
                if nread <= 0:
                    raise EOFError(
                        f"unexpected EOF reading {path}: "
                        f"offset={file_offset + offset} size={target.numel()}"
                    )
                offset += nread
        if drop_cache_after_read and hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(
                    fd,
                    file_offset,
                    target.numel(),
                    os.POSIX_FADV_DONTNEED,
                )
            except OSError:
                # Cache eviction is a memory-pressure optimization. Some remote
                # or virtual filesystems do not implement fadvise; correctness
                # does not depend on it.
                pass
    finally:
        if fd is not None:
            os.close(fd)
        view.release()
    return _PositionalReadResult(
        wall_s=time.perf_counter() - started,
        direct_io=use_direct_io,
    )


class _HostSharedCheckpoint(HostSharedMemoryBuffer):
    """One canonical checkpoint mapping shared by local TP ranks."""

    def __init__(
        self,
        *,
        capacity: int,
        cpu_group: Any,
    ):
        setup_started = time.perf_counter()
        super().__init__(
            nbytes=capacity,
            cpu_group=cpu_group,
            name="weight-checkpoint",
        )
        self.capacity = self.nbytes
        self.setup_wall_s = time.perf_counter() - setup_started

    def stats(self) -> dict[str, Any]:
        return {
            "transport": "host_shared_checkpoint",
            "bytes": self.capacity,
            "physical_host_copies": 1,
            "setup_wall_s": round(self.setup_wall_s, 6),
        }


@dataclass(frozen=True)
class _DiskTensorSource:
    filename: str
    file_begin: int
    file_end: int
    entry: _SafetensorsEntry


@dataclass(frozen=True)
class _SharedTensorSource:
    filename: str
    file_offset: int
    buffer_offset: int
    entry: _SafetensorsEntry


@dataclass(frozen=True)
class _SharedCheckpointRead:
    filename: str
    file_offset: int
    buffer_offset: int
    nbytes: int
    direct_io: bool


@dataclass(frozen=True)
class _SharedCheckpointGroup:
    path: str
    tensors: dict[str, _SharedTensorSource]
    reads: tuple[_SharedCheckpointRead, ...]
    source_bytes: int
    buffer_bytes: int


class _SharedCheckpointGroupView:
    """Expose one checkpoint group from a reusable shared buffer."""

    def __init__(
        self,
        buffer: HostSharedMemoryBuffer,
        tensors: dict[str, _SharedTensorSource],
    ):
        self.buffer = buffer
        self.tensors = tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        source = self.tensors.get(name)
        if source is None:
            raise KeyError(f"shared checkpoint group has no tensor {name!r}")
        entry = source.entry
        nbytes = entry.relative_end - entry.relative_begin
        try:
            tensor = (
                self.buffer.view(nbytes, offset=source.buffer_offset)
                .view(entry.dtype)
                .reshape(entry.shape)
            )
        except RuntimeError as exc:
            raise ValueError(
                f"cannot construct shared checkpoint view for {name!r}"
            ) from exc
        setattr(tensor, DEFERRED_WEIGHT_COPY_SAFE_ATTR, True)
        return tensor

    def get_tensor_bytes(self, name: str) -> torch.Tensor:
        source = self.tensors.get(name)
        if source is None:
            raise KeyError(f"shared checkpoint group has no tensor {name!r}")
        nbytes = source.entry.relative_end - source.entry.relative_begin
        return self.buffer.view(nbytes, offset=source.buffer_offset)


def _canonical_transform_owners(
    sources: list[_SharedCheckpointGroup],
    operations_by_file: dict[str, dict[str, Any]],
    world_size: int,
) -> dict[str, int]:
    """Balance canonical tensor reconstruction within each compile group."""

    if world_size <= 0:
        raise ValueError("canonical transform world size must be positive")
    owners = {}
    for group in sources:
        rank_bytes = [0] * world_size
        tensors = [
            (
                name,
                (source.entry.relative_end - source.entry.relative_begin)
                * len(operations_by_file[source.filename][name]),
            )
            for name, source in group.tensors.items()
            if name in operations_by_file.get(source.filename, {})
        ]
        for name, nbytes in sorted(tensors, key=lambda item: (-item[1], item[0])):
            owner = min(range(world_size), key=lambda rank: (rank_bytes[rank], rank))
            owners[name] = owner
            rank_bytes[owner] += nbytes

    expected_names = {name for names in operations_by_file.values() for name in names}
    if owners.keys() != expected_names:
        raise RuntimeError(
            "canonical checkpoint groups do not cover every delta tensor: "
            f"missing={sorted(expected_names - owners.keys())[:20]} "
            f"extra={sorted(owners.keys() - expected_names)[:20]}"
        )
    return owners


class _CanonicalCheckpointWriter:
    """Persist verified shared-buffer tensors once, then release their pages."""

    def __init__(
        self,
        *,
        root: Path,
        sources: list[_SharedCheckpointGroup],
        operations_by_file: dict[str, dict[str, Any]],
        rank: int,
        world_size: int,
        drop_cache_after_write: bool,
    ):
        from sglang.srt.weight_sync.disk_checkpoint import drop_file_page_cache

        self.root = root
        self.drop_file_page_cache = drop_file_page_cache
        self.drop_cache_after_write = drop_cache_after_write
        filesystem = os.statvfs(root)
        self.write_block_bytes = filesystem.f_frsize or filesystem.f_bsize
        filenames = sorted(operations_by_file)
        owners = _canonical_transform_owners(
            sources,
            operations_by_file,
            world_size,
        )
        self.sync_files = {
            filename
            for index, filename in enumerate(filenames)
            if index % world_size == rank
        }
        self.transform_sources = {
            name: source
            for group in sources
            for name, source in group.tensors.items()
            if owners.get(name) == rank
        }
        self.expected_names = set(self.transform_sources)
        open_files = self.sync_files | {
            source.filename for source in self.transform_sources.values()
        }
        self.last_group = {}
        for group_index, group in enumerate(sources):
            for name, source in group.tensors.items():
                if name in owners and source.filename in open_files:
                    self.last_group[source.filename] = group_index
        self.fds = {}
        self.mappings = {}
        try:
            for filename in open_files:
                fd = os.open(root / filename, os.O_RDWR)
                self.fds[filename] = fd
                try:
                    self.mappings[filename] = mmap.mmap(
                        fd,
                        0,
                        access=mmap.ACCESS_WRITE,
                    )
                except (OSError, ValueError):
                    self.mappings[filename] = None
        except Exception:
            self.close()
            raise
        self.mapped_file_count = sum(
            mapping is not None for mapping in self.mappings.values()
        )
        self.dirty_files = set()
        self.written_names = set()
        self.logical_bytes = 0
        self.write_bytes = 0
        self.write_worker_s = 0.0
        self.flush_worker_s = 0.0
        self._pending_writes = []
        self.lock = threading.Lock()

    def owns(self, name: str) -> bool:
        return name in self.transform_sources

    def record_dirty_ranges(
        self,
        name: str,
        _region: Any,
        ranges: list[tuple[int, int]],
    ) -> None:
        if name not in self.transform_sources:
            raise RuntimeError(f"rank does not own canonical tensor {name!r}")
        with self.lock:
            self._pending_writes.append((name, ranges))

    def persist_pending_group(
        self,
        get_tensor_bytes: Callable[[str], torch.Tensor],
    ) -> None:
        with self.lock:
            pending_writes = self._pending_writes
            self._pending_writes = []

        writes_by_file = {}
        for name, ranges in pending_writes:
            source = self.transform_sources[name]
            writes_by_file.setdefault(source.filename, []).append(
                (name, source.file_offset, ranges)
            )

        for filename, writes in sorted(writes_by_file.items()):
            fd = self.fds[filename]
            mapping = self.mappings[filename]
            write_bytes = 0
            started = time.perf_counter()
            for name, file_offset, ranges in sorted(
                writes,
                key=lambda item: item[1],
            ):
                view = memoryview(get_tensor_bytes(name).numpy()).cast("B")
                self.logical_bytes += len(view)
                self.written_names.add(name)
                for begin, end in ranges:
                    position = begin
                    while position < end:
                        chunk_end = min(
                            position + _POSITIONAL_IO_CHUNK_BYTES,
                            end,
                        )
                        if mapping is None:
                            written = os.pwrite(
                                fd,
                                view[position:chunk_end],
                                file_offset + position,
                            )
                            if written <= 0:
                                raise RuntimeError(
                                    "short canonical checkpoint write for "
                                    f"{name!r}: range=({begin}, {end}) "
                                    f"actual={position - begin}"
                                )
                        else:
                            mapping[
                                file_offset + position : file_offset + chunk_end
                            ] = view[position:chunk_end]
                            written = chunk_end - position
                        position += written
                        write_bytes += written
            with self.lock:
                self.write_worker_s += time.perf_counter() - started
                self.write_bytes += write_bytes
                if write_bytes:
                    self.dirty_files.add(filename)

    def finish_group(self, group_index: int) -> None:
        if self._pending_writes:
            raise RuntimeError("canonical checkpoint writes are still pending")
        filenames = [
            filename
            for filename, last_group in self.last_group.items()
            if last_group == group_index
        ]
        for filename in filenames:
            fd = self.fds.get(filename)
            mapping = self.mappings.get(filename)
            if fd is None:
                raise RuntimeError(
                    f"canonical checkpoint file is already closed: {filename}"
                )
            try:
                if filename in self.sync_files:
                    started = time.perf_counter()
                    os.fsync(fd)
                    self.flush_worker_s += time.perf_counter() - started
            finally:
                if mapping is not None:
                    mapping.close()
                os.close(fd)
                self.mappings.pop(filename, None)
                self.fds.pop(filename, None)
            if self.drop_cache_after_write and filename in self.sync_files:
                self.drop_file_page_cache(str(self.root / filename))

    def close(self) -> None:
        for mapping in self.mappings.values():
            if mapping is not None:
                mapping.close()
        self.mappings.clear()
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()

    def validate(self) -> None:
        if self.written_names != self.expected_names:
            raise RuntimeError(
                "canonical checkpoint transform did not persist every delta tensor: "
                f"missing={sorted(self.expected_names - self.written_names)[:20]} "
                f"extra={sorted(self.written_names - self.expected_names)[:20]}"
            )
        if self.fds:
            raise RuntimeError(
                "canonical checkpoint files were not finalized: "
                f"{sorted(self.fds)[:20]}"
            )

    def stats(self) -> dict[str, Any]:
        return {
            "target_logical_bytes": self.logical_bytes,
            "target_write_bytes": self.write_bytes,
            "target_write_worker_s": round(self.write_worker_s, 6),
            "target_flush_worker_s": round(self.flush_worker_s, 6),
            "target_files": len(self.dirty_files),
            "target_mapped_files": self.mapped_file_count,
        }


def _shared_checkpoint_groups(
    *,
    root: Path,
    weight_map: dict[str, str],
    names_by_group: dict[str, list[str]],
    read_parallelism: int = 1,
) -> dict[str, _SharedCheckpointGroup]:
    """Plan bounded, physically ordered reads for every runtime module group."""

    if read_parallelism <= 0:
        raise ValueError("checkpoint read parallelism must be positive")
    layouts = {
        filename: _read_safetensors_layout(root / filename)
        for filename in sorted(set(weight_map.values()))
    }
    disk_tensors = {}
    for name, filename in weight_map.items():
        layout = layouts[filename]
        try:
            entry = layout.tensors[name]
        except KeyError as exc:
            raise ValueError(
                f"checkpoint index maps {name!r} to {filename!r}, "
                "but the tensor is absent from that file"
            ) from exc
        disk_tensors[name] = _DiskTensorSource(
            filename=filename,
            file_begin=layout.data_offset + entry.relative_begin,
            file_end=layout.data_offset + entry.relative_end,
            entry=entry,
        )

    result = {}
    for group_path, names in names_by_group.items():
        sources = []
        for name in names:
            try:
                sources.append((name, disk_tensors[name]))
            except KeyError as exc:
                raise ValueError(
                    f"runtime group {group_path!r} references unknown "
                    f"checkpoint tensor {name!r}"
                ) from exc
        sources.sort(
            key=lambda item: (
                item[1].filename,
                item[1].file_begin,
                item[1].file_end,
                item[0],
            )
        )

        reads = []
        tensors = {}
        source_bytes = 0
        buffer_bytes = 0
        run = []

        def finish_run() -> None:
            nonlocal buffer_bytes
            if not run:
                return
            first = run[0][1]
            last = run[-1][1]
            run_nbytes = last.file_end - first.file_begin
            aligned_buffer_begin = (
                (buffer_bytes + _DIRECT_IO_ALIGNMENT - 1)
                // _DIRECT_IO_ALIGNMENT
                * _DIRECT_IO_ALIGNMENT
            )
            data_buffer_begin = (
                aligned_buffer_begin + first.file_begin % _DIRECT_IO_ALIGNMENT
            )
            direct_begin = (
                (first.file_begin + _DIRECT_IO_ALIGNMENT - 1)
                // _DIRECT_IO_ALIGNMENT
                * _DIRECT_IO_ALIGNMENT
            )
            direct_end = last.file_end // _DIRECT_IO_ALIGNMENT * _DIRECT_IO_ALIGNMENT
            if direct_begin < direct_end:
                if first.file_begin < direct_begin:
                    reads.append(
                        _SharedCheckpointRead(
                            filename=first.filename,
                            file_offset=first.file_begin,
                            buffer_offset=data_buffer_begin,
                            nbytes=direct_begin - first.file_begin,
                            direct_io=False,
                        )
                    )
                direct_nbytes = direct_end - direct_begin
                read_chunks = min(
                    read_parallelism,
                    math.ceil(direct_nbytes / _POSITIONAL_IO_CHUNK_BYTES),
                )
                chunk_blocks = math.ceil(
                    direct_nbytes / _DIRECT_IO_ALIGNMENT / read_chunks
                )
                chunk_bytes = chunk_blocks * _DIRECT_IO_ALIGNMENT
                reads.extend(
                    (
                        _SharedCheckpointRead(
                            filename=first.filename,
                            file_offset=direct_begin + offset,
                            buffer_offset=(
                                data_buffer_begin
                                + direct_begin
                                - first.file_begin
                                + offset
                            ),
                            nbytes=min(chunk_bytes, direct_nbytes - offset),
                            direct_io=True,
                        )
                        for offset in range(0, direct_nbytes, chunk_bytes)
                    )
                )
                if direct_end < last.file_end:
                    reads.append(
                        _SharedCheckpointRead(
                            filename=first.filename,
                            file_offset=direct_end,
                            buffer_offset=(
                                data_buffer_begin + direct_end - first.file_begin
                            ),
                            nbytes=last.file_end - direct_end,
                            direct_io=False,
                        )
                    )
            elif run_nbytes:
                read_chunks = min(
                    read_parallelism,
                    math.ceil(run_nbytes / _POSITIONAL_IO_CHUNK_BYTES),
                )
                chunk_bytes = math.ceil(run_nbytes / read_chunks)
                reads.extend(
                    _SharedCheckpointRead(
                        filename=first.filename,
                        file_offset=first.file_begin + offset,
                        buffer_offset=data_buffer_begin + offset,
                        nbytes=min(chunk_bytes, run_nbytes - offset),
                        direct_io=False,
                    )
                    for offset in range(0, run_nbytes, chunk_bytes)
                )
            for name, source in run:
                tensors[name] = _SharedTensorSource(
                    filename=source.filename,
                    file_offset=source.file_begin,
                    buffer_offset=(
                        data_buffer_begin + source.file_begin - first.file_begin
                    ),
                    entry=source.entry,
                )
            buffer_bytes = data_buffer_begin + run_nbytes
            run.clear()

        for name, source in sources:
            nbytes = source.file_end - source.file_begin
            source_bytes += nbytes
            if run:
                previous = run[-1][1]
                if (
                    source.filename != previous.filename
                    or source.file_begin != previous.file_end
                ):
                    finish_run()
            run.append((name, source))
        finish_run()
        result[group_path] = _SharedCheckpointGroup(
            path=group_path,
            tensors=tensors,
            reads=tuple(reads),
            source_bytes=source_bytes,
            buffer_bytes=buffer_bytes,
        )
    return result


def _populate_shared_checkpoint_group(
    *,
    root: Path,
    group: _SharedCheckpointGroup,
    buffer: HostSharedMemoryBuffer,
    cpu_group: Any,
    drop_cache_after_read: bool = False,
) -> dict[str, Any]:
    """Read every source range exactly once across the local TP ranks."""

    started = time.perf_counter()
    distributed = torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size(group=cpu_group) if distributed else 1
    rank = torch.distributed.get_rank(group=cpu_group) if distributed else 0
    owned_reads = []
    for read_index, read in enumerate(group.reads):
        if read_index % world_size != rank:
            continue
        target = buffer.view(
            read.nbytes,
            offset=read.buffer_offset,
        )
        try:
            read_result = _pread_range_to_tensor(
                root / read.filename,
                target,
                file_offset=read.file_offset,
                direct_io=read.direct_io,
                drop_cache_after_read=drop_cache_after_read,
            )
        finally:
            del target
        owned_reads.append(
            {
                "filename": read.filename,
                "file_offset": read.file_offset,
                "bytes": read.nbytes,
                "direct_io": read_result.direct_io,
                "wall_s": round(read_result.wall_s, 6),
            }
        )
    return {
        "owner_rank": rank,
        "owned_reads": owned_reads,
        "owned_bytes": sum(read["bytes"] for read in owned_reads),
        "owned_direct_io_bytes": sum(
            read["bytes"] for read in owned_reads if read["direct_io"]
        ),
        "wall_s": round(time.perf_counter() - started, 6),
    }


def _validate_shared_checkpoint_groups(
    groups: dict[str, _SharedCheckpointGroup],
    *,
    cpu_group: Any,
) -> None:
    """Fail before shared allocation if local TP layouts disagree."""

    if not torch.distributed.is_initialized():
        return
    digest = hashlib.sha256()
    for path, group in groups.items():
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(str(group.source_bytes).encode())
        digest.update(b"\0")
        digest.update(str(group.buffer_bytes).encode())
        digest.update(b"\0")
        for name, source in sorted(group.tensors.items()):
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(source.entry.dtype_code.encode())
            digest.update(b"\0")
            digest.update(str(source.entry.shape).encode())
            digest.update(b"\0")
            digest.update(str(source.buffer_offset).encode())
            digest.update(b"\0")
        for read in group.reads:
            digest.update(read.filename.encode())
            digest.update(b"\0")
            digest.update(
                f"{read.file_offset}:{read.buffer_offset}:{read.nbytes}".encode()
            )
            digest.update(b"\0")
    signature = digest.hexdigest()
    signatures = [None] * torch.distributed.get_world_size(group=cpu_group)
    torch.distributed.all_gather_object(
        signatures,
        signature,
        group=cpu_group,
    )
    if len(set(signatures)) != 1:
        raise RuntimeError(
            "disk-backed CPU weight compilation requires identical checkpoint "
            f"groups on every local TP rank: {signatures}"
        )


@dataclass(frozen=True)
class _WeightModuleGroup:
    """A model subtree small enough to compile without a second full model."""

    path: str
    nbytes: int


@dataclass
class _LoadedWeightGroup:
    """One loaded module group awaiting its post-load transforms."""

    group_index: int
    group: _WeightModuleGroup
    checkpoint_tensors: int
    cpu_shadow: torch.nn.Module
    cpu_image_storage_keys: set[tuple[int, int]]
    group_started: float
    cpu_clone_s: float
    restore_s: float
    cpu_load_s: float


class _HostRankPhaseCoordinator:
    """Coordinate repeated host-local phases without network collectives."""

    def __init__(self, cpu_group: Any):
        distributed = torch.distributed.is_initialized()
        self.world_size = (
            torch.distributed.get_world_size(group=cpu_group) if distributed else 1
        )
        self.rank = torch.distributed.get_rank(group=cpu_group) if distributed else 0
        self.phase = 0
        self.buffer = (
            HostSharedMemoryBuffer(
                nbytes=self.world_size * 16,
                cpu_group=cpu_group,
                name="weight-phase-coordinator",
            )
            if self.world_size > 1
            else None
        )
        self._format = f"{self.world_size}q"

    def arrive(self, failed: bool) -> bool:
        if self.buffer is None:
            return failed
        self.phase += 1
        value = -self.phase if failed else self.phase
        struct.pack_into(
            "q",
            self.buffer.mapping,
            self.rank * 8,
            value,
        )
        while True:
            values = struct.unpack_from(self._format, self.buffer.mapping)
            if all(abs(value) == self.phase for value in values):
                any_failed = any(value < 0 for value in values)
                break
            time.sleep(0.001)
        struct.pack_into(
            "q",
            self.buffer.mapping,
            (self.world_size + self.rank) * 8,
            self.phase,
        )
        while True:
            departed = struct.unpack_from(
                self._format,
                self.buffer.mapping,
                self.world_size * 8,
            )
            if all(value == self.phase for value in departed):
                return any_failed
            time.sleep(0.001)

    def close(self) -> None:
        if self.buffer is not None:
            self.buffer.close()
            self.buffer = None


def _storage_key(tensor: torch.Tensor) -> tuple[int | None, int, int]:
    storage = tensor.untyped_storage()
    return tensor.device.index, storage.data_ptr(), storage.nbytes()


def _direct_weight_tensors(
    module: torch.nn.Module,
) -> Iterable[torch.Tensor]:
    yield from (value for value in module._parameters.values() if value is not None)
    yield from (
        value
        for name, value in module._buffers.items()
        if value is not None and name not in module._non_persistent_buffers_set
    )
    get_extra = getattr(module, "get_additional_weight_tensors", None)
    if get_extra is not None:
        for _, tensor in get_extra():
            yield tensor


def _build_weight_module_groups(
    model: torch.nn.Module,
    *,
    max_group_bytes: int,
    device_type: str = "cuda",
) -> list[_WeightModuleGroup]:
    """Partition the runtime module tree into bounded, storage-complete groups."""

    if max_group_bytes <= 0:
        raise ValueError("weight compilation group budget must be positive")

    subtree_keys: dict[str, set[tuple[int | None, int, int]]] = {}
    direct_keys: dict[str, set[tuple[int | None, int, int]]] = {}
    storage_nbytes: dict[tuple[int | None, int, int], int] = {}

    def collect(path: str, module: torch.nn.Module):
        direct: set[tuple[int | None, int, int]] = set()
        for tensor in _direct_weight_tensors(module):
            if tensor.device.type != device_type:
                continue
            key = _storage_key(tensor)
            direct.add(key)
            storage_nbytes[key] = key[2]
        direct_keys[path] = direct
        subtree = set(direct)
        prefix = f"{path}." if path else ""
        for child_name, child in module.named_children():
            subtree.update(collect(f"{prefix}{child_name}", child))
        subtree_keys[path] = subtree
        return subtree

    collect("", model)
    groups: list[_WeightModuleGroup] = []

    def visit(path: str, module: torch.nn.Module) -> None:
        keys = subtree_keys[path]
        if not keys:
            return
        nbytes = sum(storage_nbytes[key] for key in keys)
        prefix = f"{path}." if path else ""
        children = [
            (f"{prefix}{name}", child)
            for name, child in module.named_children()
            if subtree_keys[f"{prefix}{name}"]
        ]
        if path and (nbytes <= max_group_bytes or not children):
            if nbytes > max_group_bytes:
                logger.warning(
                    "indivisible weight module exceeds compilation budget: "
                    "path=%s bytes=%d budget=%d",
                    path,
                    nbytes,
                    max_group_bytes,
                )
            groups.append(_WeightModuleGroup(path=path, nbytes=nbytes))
            return
        if direct_keys[path]:
            raise ValueError(
                "cannot split a weight module that owns direct tensors and "
                f"child subtrees: path={path or '<root>'!r} bytes={nbytes} "
                f"budget={max_group_bytes}"
            )
        for child_path, child in children:
            visit(child_path, child)

    visit("", model)
    if not groups:
        raise ValueError(f"model has no {device_type} weight module groups")
    owners: dict[tuple[int | None, int, int], str] = {}
    for group in groups:
        for key in subtree_keys[group.path]:
            previous = owners.setdefault(key, group.path)
            if previous != group.path:
                raise ValueError(
                    "weight storage spans independent compilation groups: "
                    f"{previous!r} and {group.path!r}; use the disk update path "
                    "for this model"
                )
    return groups


def _clone_tensor(
    tensor: torch.Tensor,
    tensor_memo: dict[int, torch.Tensor],
    storage_memo: dict[tuple[int | None, int, int], torch.Tensor],
    tensor_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    target_device: torch.device | None = None,
    copy_data: bool = True,
    storage_factory: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    cached = tensor_memo.get(id(tensor))
    if cached is not None:
        return cached

    key = _storage_key(tensor)
    storage_bytes = storage_memo.get(key)
    if storage_bytes is None:
        source = torch.empty(
            0,
            dtype=torch.uint8,
            device=tensor.device,
        ).set_(
            tensor.untyped_storage(),
            0,
            (tensor.untyped_storage().nbytes(),),
            (1,),
        )
        if storage_factory is not None:
            storage_bytes = storage_factory(tensor, source)
        else:
            device = tensor.device if target_device is None else target_device
            storage_bytes = torch.empty(
                source.numel(),
                dtype=torch.uint8,
                device=device,
            )
            if copy_data:
                storage_bytes.copy_(source, non_blocking=True)
        if (
            storage_bytes.dtype != torch.uint8
            or storage_bytes.ndim != 1
            or storage_bytes.numel() != source.numel()
        ):
            raise ValueError(
                "cloned storage must be a flat byte tensor with the source size"
            )
        storage_memo[key] = storage_bytes
    storage_byte_offset = storage_bytes.storage_offset()
    if storage_byte_offset % tensor.element_size():
        raise ValueError(
            "cloned byte storage offset is not aligned for the tensor dtype: "
            f"offset={storage_byte_offset} dtype={tensor.dtype}"
        )
    view = torch.empty(0, dtype=tensor.dtype, device=storage_bytes.device).set_(
        storage_bytes.untyped_storage(),
        storage_byte_offset // tensor.element_size() + tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )
    if isinstance(tensor, torch.nn.Parameter):
        cloned = type(tensor)._make_subclass(
            type(tensor),
            view,
            tensor.requires_grad,
        )
    else:
        cloned = view.requires_grad_(tensor.requires_grad)
    tensor_memo[id(tensor)] = cloned
    tensor_pairs.append((tensor, cloned))
    return cloned


def _clone_attribute(
    value: Any,
    tensor_memo: dict[int, torch.Tensor],
    storage_memo: dict[tuple[int | None, int, int], torch.Tensor],
    tensor_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    container_memo: dict[int, Any],
    *,
    target_device: torch.device | None,
    copy_data: bool,
    storage_factory: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
) -> Any:
    if isinstance(value, torch.Tensor):
        return _clone_tensor(
            value,
            tensor_memo,
            storage_memo,
            tensor_pairs,
            target_device=target_device,
            copy_data=copy_data,
            storage_factory=storage_factory,
        )
    cached = container_memo.get(id(value))
    if cached is _CONTAINER_CLONE_IN_PROGRESS:
        raise ValueError("cyclic immutable loader state cannot be cloned safely")
    if cached is not None:
        return cached
    if isinstance(value, dict):
        cloned = copy.copy(value)
        cloned.clear()
        container_memo[id(value)] = cloned
        cloned.update(
            (
                key,
                _clone_attribute(
                    child,
                    tensor_memo,
                    storage_memo,
                    tensor_pairs,
                    container_memo,
                    target_device=target_device,
                    copy_data=copy_data,
                    storage_factory=storage_factory,
                ),
            )
            for key, child in value.items()
        )
        return cloned
    if isinstance(value, list):
        cloned = copy.copy(value)
        cloned.clear()
        container_memo[id(value)] = cloned
        cloned.extend(
            _clone_attribute(
                child,
                tensor_memo,
                storage_memo,
                tensor_pairs,
                container_memo,
                target_device=target_device,
                copy_data=copy_data,
                storage_factory=storage_factory,
            )
            for child in value
        )
        return cloned
    if isinstance(value, set):
        cloned = copy.copy(value)
        cloned.clear()
        container_memo[id(value)] = cloned
        cloned.update(
            _clone_attribute(
                child,
                tensor_memo,
                storage_memo,
                tensor_pairs,
                container_memo,
                target_device=target_device,
                copy_data=copy_data,
                storage_factory=storage_factory,
            )
            for child in value
        )
        return cloned
    if isinstance(value, (tuple, frozenset)):
        container_memo[id(value)] = _CONTAINER_CLONE_IN_PROGRESS
        children = [
            _clone_attribute(
                child,
                tensor_memo,
                storage_memo,
                tensor_pairs,
                container_memo,
                target_device=target_device,
                copy_data=copy_data,
                storage_factory=storage_factory,
            )
            for child in value
        ]
        if isinstance(value, frozenset):
            cloned = frozenset(children)
        elif hasattr(value, "_fields"):
            cloned = type(value)(*children)
        else:
            cloned = tuple(children)
        container_memo[id(value)] = cloned
        return cloned
    return value


_CONTAINER_CLONE_IN_PROGRESS = object()


def _rebind_cloned_method_owners(
    value: Any,
    owner_memo: dict[int, Any],
    value_memo: dict[int, Any],
) -> Any:
    cloned_value = owner_memo.get(id(value))
    if cloned_value is not None:
        return cloned_value
    owner = getattr(value, "__self__", None)
    function = getattr(value, "__func__", None)
    if owner is not None and function is not None:
        cloned_owner = owner_memo.get(id(owner))
        return value if cloned_owner is None else function.__get__(cloned_owner)

    cached = value_memo.get(id(value))
    if cached is _CONTAINER_CLONE_IN_PROGRESS:
        raise ValueError("cyclic immutable loader state cannot be rebound safely")
    if cached is not None:
        return cached
    if isinstance(value, functools.partial):
        cloned = functools.partial(
            _rebind_cloned_method_owners(value.func, owner_memo, value_memo),
            *(
                _rebind_cloned_method_owners(item, owner_memo, value_memo)
                for item in value.args
            ),
            **{
                key: _rebind_cloned_method_owners(item, owner_memo, value_memo)
                for key, item in (value.keywords or {}).items()
            },
        )
        value_memo[id(value)] = cloned
        cloned.__dict__.update(
            {
                key: _rebind_cloned_method_owners(item, owner_memo, value_memo)
                for key, item in value.__dict__.items()
            }
        )
        return cloned
    if isinstance(value, dict):
        cloned = copy.copy(value)
        cloned.clear()
        value_memo[id(value)] = cloned
        cloned.update(
            (
                key,
                _rebind_cloned_method_owners(item, owner_memo, value_memo),
            )
            for key, item in value.items()
        )
        return cloned
    if isinstance(value, list):
        cloned = copy.copy(value)
        cloned.clear()
        value_memo[id(value)] = cloned
        cloned.extend(
            _rebind_cloned_method_owners(item, owner_memo, value_memo) for item in value
        )
        return cloned
    if isinstance(value, set):
        cloned = copy.copy(value)
        cloned.clear()
        value_memo[id(value)] = cloned
        cloned.update(
            _rebind_cloned_method_owners(item, owner_memo, value_memo) for item in value
        )
        return cloned
    if isinstance(value, (tuple, frozenset)):
        value_memo[id(value)] = _CONTAINER_CLONE_IN_PROGRESS
        items = [
            _rebind_cloned_method_owners(item, owner_memo, value_memo) for item in value
        ]
        if isinstance(value, frozenset):
            cloned = frozenset(items)
        elif hasattr(value, "_fields"):
            cloned = type(value)(*items)
        else:
            cloned = tuple(items)
        value_memo[id(value)] = cloned
        return cloned
    return value


def _clone_weight_module(
    module: torch.nn.Module,
    *,
    target_device: torch.device | None = None,
    copy_data: bool = True,
    storage_factory: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.nn.Module:
    """Clone tensor state and mutable objects used by weight-loading hooks."""

    tensor_memo: dict[int, torch.Tensor] = {}
    storage_memo: dict[tuple[int | None, int, int], torch.Tensor] = {}
    tensor_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    container_memo: dict[int, Any] = {}
    loader_object_memo: dict[int, Any] = {}
    module_memo: dict[int, torch.nn.Module] = {}

    def clone(current: torch.nn.Module) -> torch.nn.Module:
        result = copy.copy(current)
        module_memo[id(current)] = result
        result._parameters = {
            name: (
                None
                if parameter is None
                else _clone_tensor(
                    parameter,
                    tensor_memo,
                    storage_memo,
                    tensor_pairs,
                    target_device=target_device,
                    copy_data=copy_data,
                    storage_factory=storage_factory,
                )
            )
            for name, parameter in current._parameters.items()
        }
        result._buffers = {
            name: (
                None
                if buffer is None
                else _clone_tensor(
                    buffer,
                    tensor_memo,
                    storage_memo,
                    tensor_pairs,
                    target_device=target_device,
                    copy_data=copy_data,
                    storage_factory=storage_factory,
                )
            )
            for name, buffer in current._buffers.items()
        }
        result._modules = {
            name: None if child is None else clone(child)
            for name, child in current._modules.items()
        }
        result._non_persistent_buffers_set = current._non_persistent_buffers_set.copy()
        for name, value in vars(current).items():
            if name in {"_parameters", "_buffers", "_modules"}:
                continue
            clone_for_update = getattr(
                value,
                "clone_for_weight_update",
                None,
            )
            if callable(clone_for_update):
                if id(value) not in loader_object_memo:
                    cloned_value = clone_for_update()
                    if cloned_value is value:
                        raise RuntimeError(
                            "clone_for_weight_update() returned the live object"
                        )
                    loader_object_memo[id(value)] = cloned_value
                result.__dict__[name] = loader_object_memo[id(value)]
                continue
            if name in {"quant_method", "scheme"} and value is not None:
                if id(value) not in loader_object_memo:
                    loader_object_memo[id(value)] = copy.copy(value)
                result.__dict__[name] = loader_object_memo[id(value)]
                continue
            result.__dict__[name] = _clone_attribute(
                value,
                tensor_memo,
                storage_memo,
                tensor_pairs,
                container_memo,
                target_device=target_device,
                copy_data=copy_data,
                storage_factory=storage_factory,
            )
        return result

    result = clone(module)
    # Parameter loaders and quantization methods keep tensors in custom tensor
    # attributes. Clone those attributes after the registered model state has
    # populated the tensor memo, preserving aliases without sharing storage.
    pair_index = 0
    while pair_index < len(tensor_pairs):
        source, cloned = tensor_pairs[pair_index]
        pair_index += 1
        if not hasattr(source, "__dict__"):
            continue
        cloned.__dict__.update(
            {
                name: _clone_attribute(
                    value,
                    tensor_memo,
                    storage_memo,
                    tensor_pairs,
                    container_memo,
                    target_device=target_device,
                    copy_data=copy_data,
                    storage_factory=storage_factory,
                )
                for name, value in vars(source).items()
            }
        )
    owner_memo = {**module_memo, **loader_object_memo}
    value_memo: dict[int, Any] = {}
    for cloned_module in module_memo.values():
        for name, value in vars(cloned_module).items():
            if name in {"_parameters", "_buffers", "_modules"}:
                continue
            cloned_module.__dict__[name] = _rebind_cloned_method_owners(
                value,
                owner_memo,
                value_memo,
            )
    for cloned_object in loader_object_memo.values():
        if not hasattr(cloned_object, "__dict__"):
            continue
        for name, value in vars(cloned_object).items():
            cloned_object.__dict__[name] = _rebind_cloned_method_owners(
                value,
                owner_memo,
                value_memo,
            )
    for cloned_tensor in tensor_memo.values():
        for name, value in vars(cloned_tensor).items():
            cloned_tensor.__dict__[name] = _rebind_cloned_method_owners(
                value,
                owner_memo,
                value_memo,
            )
    return result


def _build_weight_loader_proxy(
    model: torch.nn.Module,
    path: str,
    *,
    target_device: torch.device | None = None,
    copy_data: bool = True,
    storage_factory: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Build a loader proxy with one isolated, tensor-cloned subtree."""

    def clone_shell(module: torch.nn.Module) -> torch.nn.Module:
        result = copy.copy(module)
        result._parameters = module._parameters.copy()
        result._buffers = module._buffers.copy()
        result._modules = module._modules.copy()
        result._non_persistent_buffers_set = module._non_persistent_buffers_set.copy()
        for name in ("quant_method", "scheme"):
            value = getattr(module, name, None)
            if value is not None:
                result.__dict__[name] = copy.copy(value)
        for name, value in vars(module).items():
            clone_for_update = getattr(
                value,
                "clone_for_weight_update",
                None,
            )
            if callable(clone_for_update):
                result.__dict__[name] = clone_for_update()
        return result

    parts = path.split(".")
    live = model
    proxy = clone_shell(model)
    proxy_cursor = proxy
    for index, part in enumerate(parts):
        live_child = live._modules.get(part)
        if live_child is None:
            raise KeyError(f"module path {path!r} is missing component {part!r}")
        if index == len(parts) - 1:
            shadow = _clone_weight_module(
                live_child,
                target_device=target_device,
                copy_data=copy_data,
                storage_factory=storage_factory,
            )
            proxy_cursor._modules[part] = shadow
            return proxy, shadow
        proxy_child = clone_shell(live_child)
        proxy_cursor._modules[part] = proxy_child
        proxy_cursor = proxy_child
        live = live_child
    raise AssertionError("empty module path")


def _map_checkpoint_name(
    model: torch.nn.Module,
    name: str,
) -> str | None:
    """Apply the same authoritative name mapper as the ordinary loader."""

    mapper = getattr(model, "hf_to_sglang_mapper", None)
    return name if mapper is None else mapper._map_name(name)


def _longest_group_prefix(
    name: str,
    paths: set[str],
) -> str | None:
    parts = name.split(".")
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        if candidate in paths:
            return candidate
    return None


def _map_checkpoint_names_to_groups(
    model: torch.nn.Module,
    names: Iterable[str],
    groups: list[_WeightModuleGroup],
) -> dict[str, str | None]:
    """Map checkpoint tensors to the bounded runtime subtree that loads them."""

    paths = {group.path for group in groups}
    root_prefixes = {path.split(".", 1)[0] for path in paths}
    result: dict[str, str | None] = {}
    for name in names:
        mapped = _map_checkpoint_name(model, name)
        if mapped is None:
            result[name] = None
            continue
        direct = _longest_group_prefix(mapped, paths)
        if direct is not None:
            result[name] = direct
            continue

        # Some wrapper models delegate unprefixed checkpoint names into one
        # named runtime child. Only infer that wrapper prefix when the
        # authoritative mapped name has no direct runtime match.
        matches = {
            match
            for root in root_prefixes
            if (match := _longest_group_prefix(f"{root}.{mapped}", paths)) is not None
        }
        if len(matches) > 1:
            raise ValueError(f"ambiguous checkpoint group for {name!r}: {matches}")
        result[name] = next(iter(matches), None)
    return result


def _checkpoint_weight_map(checkpoint_dir: str) -> tuple[dict[str, str], Path]:
    root = Path(checkpoint_dir)
    model_index = root / "model.safetensors.index.json"
    indexes = (
        [model_index]
        if model_index.is_file()
        else sorted(root.glob("*.safetensors.index.json"))
    )
    if len(indexes) > 1:
        raise ValueError(
            f"expected at most one safetensors index in {checkpoint_dir!r}, "
            f"found {indexes}"
        )
    if indexes:
        payload = json.loads(indexes[0].read_text())
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid safetensors weight map: {indexes[0]}")
        if not all(
            isinstance(name, str) and isinstance(filename, str)
            for name, filename in weight_map.items()
        ):
            raise ValueError(f"invalid safetensors weight map: {indexes[0]}")
        return weight_map, root

    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no safetensors weights found in {checkpoint_dir!r}")
    weight_map = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as tensor_file:
            for name in tensor_file.keys():
                if name in weight_map:
                    raise ValueError(
                        f"duplicate safetensors tensor {name!r} in "
                        f"{weight_map[name]!r} and {path.name!r}"
                    )
                weight_map[name] = path.name
    return weight_map, root


def _canonical_checkpoint_layout(
    root: Path,
    filenames: list[str],
) -> tuple[
    dict[str, int],
    dict[str, int],
    int,
    tuple[str, tuple[tuple[str, int], ...]],
]:
    """Lay complete checkpoint files out once in a page-aligned CPU image."""

    file_sizes = {filename: (root / filename).stat().st_size for filename in filenames}
    offsets = {}
    capacity = 0
    for filename in filenames:
        capacity = (capacity + 4095) // 4096 * 4096
        offsets[filename] = capacity
        capacity += file_sizes[filename]
    capacity = (capacity + 4095) // 4096 * 4096
    signature = (
        os.path.realpath(root),
        tuple((filename, file_sizes[filename]) for filename in filenames),
    )
    return file_sizes, offsets, capacity, signature


class _NoOpCheckpointTransform:
    """Validate the persistent baseline without mutating its bytes."""

    canonical_version = 0

    @staticmethod
    def transform_file(
        filename: str,
        _tensor_file: _InMemorySafetensorsFile,
    ) -> dict[str, Any]:
        return {
            "operation": "canonical_baseline",
            "filename": filename,
            "delta_tensors": 0,
            "target_tensor_bytes": 0,
            "compressed_bytes": 0,
            "wall_s": 0.0,
        }


def _transform_canonical_checkpoint(
    *,
    filenames: list[str],
    file_sizes: dict[str, int],
    offsets: dict[str, int],
    checkpoint: _HostSharedCheckpoint,
    checkpoint_transform: Any,
    rank: int,
    world_size: int,
    cpu_group: Any,
) -> dict[str, Any]:
    """Advance and verify the complete canonical image before compiling it.

    The canonical files are disjoint, so local TP ranks can own a strided
    subset without locking. One collective publishes either complete success
    or every rank's error after all owned tensor checksums have run. Runtime
    layout compilation then consumes an immutable, fully verified checkpoint;
    it does not need a collective at every checkpoint-file batch.
    """

    started = time.perf_counter()
    owned_transforms = []
    local_error = None
    try:
        for file_index, filename in enumerate(filenames):
            if file_index % world_size != rank:
                continue
            file_nbytes = file_sizes[filename]
            source_view = checkpoint.view(
                file_nbytes,
                offset=offsets[filename],
            )
            try:
                transform = checkpoint_transform.transform_file(
                    filename,
                    _InMemorySafetensorsFile(source_view),
                )
            finally:
                del source_view
            owned_transforms.append(transform)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    verify_started = time.perf_counter()
    if world_size > 1:
        errors: list[str | None] = [None] * world_size
        torch.distributed.all_gather_object(
            errors,
            local_error,
            group=cpu_group,
        )
    else:
        errors = [local_error]
    verify_barrier_s = time.perf_counter() - verify_started
    errors = [value for value in errors if value is not None]
    if errors:
        raise RuntimeError(
            "canonical checkpoint transform failed before runtime compilation: "
            + "; ".join(errors)
        )

    stats = {
        "operation": "transform_canonical_checkpoint",
        "files": len(filenames),
        "owner_rank": rank,
        "owned_files": len(owned_transforms),
        "owned_transforms": owned_transforms,
        "transform_wall_s": round(
            sum(value.get("wall_s", 0.0) for value in owned_transforms),
            6,
        ),
        "verify_barrier_s": round(verify_barrier_s, 6),
        "wall_s": round(time.perf_counter() - started, 6),
    }
    logger.info(
        "Transformed canonical checkpoint on rank %d: files=%d "
        "transform_time=%.3fs synchronization_time=%.3fs wall_time=%.3fs",
        rank,
        len(owned_transforms),
        stats["transform_wall_s"],
        stats["verify_barrier_s"],
        stats["wall_s"],
    )
    return stats


class CPUWeightCache:
    """Compile safetensors targets into complete pinned host images."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        max_group_bytes: int,
        host_cpu_group: Any = None,
        canonical_checkpoint_storage: Literal["memory", "disk"] = "memory",
        drop_cache_after_load: bool = False,
    ):
        if getattr(model, "secondary_weights", None):
            raise NotImplementedError(
                "CPU weight staging does not support models with secondary "
                "checkpoint sources; use the disk update path for this model"
            )
        self.model = model
        self.target_device = torch.device("cuda", torch.cuda.current_device())
        self.max_group_bytes = max_group_bytes
        self.host_cpu_group = host_cpu_group
        self.canonical_checkpoint_storage = canonical_checkpoint_storage
        self.drop_cache_after_load = drop_cache_after_load
        if canonical_checkpoint_storage not in {"memory", "disk"}:
            raise ValueError("canonical_checkpoint_storage must be 'memory' or 'disk'")
        self.groups = _build_weight_module_groups(
            model,
            max_group_bytes=max_group_bytes,
        )
        self._weight_update_postprocess_device(model)
        self.image = CPUWeightImage(model)
        self._compile_stream = torch.cuda.Stream(device=self.target_device)
        self._canonical_checkpoint: _HostSharedCheckpoint | None = None
        self._canonical_checkpoint_signature: (
            tuple[
                str,
                tuple[tuple[str, int], ...],
            ]
            | None
        ) = None
        self._canonical_lineage: tuple[str, str] | None = None
        self._canonical_checkpoint_version: int | None = None
        self._host_rank_coordinator: _HostRankPhaseCoordinator | None = None
        self._host_rank_sync_s = 0.0
        logger.info(
            "CPU weight cache layout: groups=%d storages=%d bytes=%d "
            "max_compile_group_bytes=%d canonical_checkpoint_storage=%s",
            len(self.groups),
            len(self.image.segments),
            self.image.image_nbytes,
            self.max_group_bytes,
            self.canonical_checkpoint_storage,
        )

    def initialize_host_rank_coordinator(self) -> None:
        if self._host_rank_coordinator is not None:
            raise RuntimeError("host-rank coordinator is already initialized")
        self._host_rank_coordinator = _HostRankPhaseCoordinator(self.host_cpu_group)

    def _run_on_all_host_ranks(self, description: str, function: Callable[[], Any]):
        result = None
        error = None
        try:
            result = function()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        distributed = torch.distributed.is_initialized()
        world_size = (
            torch.distributed.get_world_size(group=self.host_cpu_group)
            if distributed
            else 1
        )
        if world_size > 1:
            sync_started = time.perf_counter()
            coordinator = self._host_rank_coordinator
            if coordinator is None:
                failed = torch.tensor([error is not None], dtype=torch.int32)
                torch.distributed.all_reduce(
                    failed,
                    op=torch.distributed.ReduceOp.MAX,
                    group=self.host_cpu_group,
                )
                any_failed = bool(failed.item())
            else:
                any_failed = coordinator.arrive(error is not None)
            self._host_rank_sync_s += time.perf_counter() - sync_started
            if any_failed:
                errors: list[str | None] = [None] * world_size
                torch.distributed.all_gather_object(
                    errors,
                    error,
                    group=self.host_cpu_group,
                )
            else:
                errors = []
        else:
            errors = [error]
        errors = [value for value in errors if value is not None]
        if errors:
            raise RuntimeError(f"{description} failed: " + "; ".join(errors))
        return result

    def initialize_from_checkpoint(
        self,
        *,
        checkpoint_dir: str,
        seed_from_active_weights: bool,
    ) -> dict[str, Any]:
        """Populate the canonical checkpoint and rank-ready weight image."""

        started = time.perf_counter()
        registration = self._run_on_all_host_ranks(
            "CPU weight image registration",
            self.image.register_host_memory,
        )
        seed = self._run_on_all_host_ranks(
            "active weight capture",
            self.image.capture_active_weights,
        )

        def inspect_checkpoint():
            weight_map, root = _checkpoint_weight_map(checkpoint_dir)
            filenames = sorted(set(weight_map.values()))
            checkpoint_bytes = sum((root / name).stat().st_size for name in filenames)
            return root, filenames, checkpoint_bytes

        root, filenames, canonical_checkpoint_bytes = self._run_on_all_host_ranks(
            "CPU weight cache checkpoint inspection",
            inspect_checkpoint,
        )
        if self.canonical_checkpoint_storage == "memory":
            (
                file_sizes,
                offsets,
                capacity,
                signature,
            ) = self._run_on_all_host_ranks(
                "CPU weight cache checkpoint layout inspection",
                lambda: _canonical_checkpoint_layout(root, filenames),
            )
            checkpoint, created = self._get_canonical_checkpoint(
                capacity=capacity,
                signature=signature,
            )
            if not created:
                raise RuntimeError(
                    "canonical checkpoint already exists during cache initialization"
                )
            canonical_checkpoint_stats = self._populate_canonical_checkpoint(
                root=root,
                filenames=filenames,
                file_sizes=file_sizes,
                offsets=offsets,
                checkpoint=checkpoint,
            )
            self._canonical_checkpoint_version = 0
        else:
            canonical_checkpoint_stats = {
                "setup_wall_s": 0.0,
                "wall_s": 0.0,
            }
        if seed_from_active_weights:
            initial_compile_wall_s = 0.0
            validation_wall_s = 0.0
            rank_image_source = "active_model"
        else:
            baseline_stage = self._stage_from_checkpoint(
                checkpoint_dir=checkpoint_dir,
                target_version=0,
                checkpoint_transform=_NoOpCheckpointTransform(),
            )
            validation = self.image.validate_against_active()
            self.image.accept_staged_baseline()
            initial_compile_wall_s = baseline_stage["wall_s"]
            validation_wall_s = validation["wall_s"]
            rank_image_source = "checkpoint"
        stats = {
            "operation": "initialize_cpu_weight_cache",
            "canonical_checkpoint_storage": self.canonical_checkpoint_storage,
            "canonical_checkpoint_bytes": canonical_checkpoint_bytes,
            "rank_image_bytes": self.image.image_nbytes,
            "rank_weight_bytes": self.image.weight_nbytes,
            "rank_image_source": rank_image_source,
            "compile_group_limit_bytes": self.max_group_bytes,
            "registration_wall_s": registration["wall_s"],
            "capture_wall_s": seed["wall_s"],
            "canonical_setup_wall_s": canonical_checkpoint_stats["setup_wall_s"],
            "canonical_load_wall_s": canonical_checkpoint_stats["wall_s"],
            "initial_compile_wall_s": initial_compile_wall_s,
            "validation_wall_s": validation_wall_s,
            "wall_s": round(time.perf_counter() - started, 6),
        }
        logger.info(
            "CPU weight cache ready: rank_image_bytes=%d "
            "canonical_checkpoint_bytes=%d canonical_checkpoint_storage=%s "
            "compile_time=%.3fs wall_time=%.3fs",
            self.image.image_nbytes,
            stats["canonical_checkpoint_bytes"],
            stats["canonical_checkpoint_storage"],
            stats["initial_compile_wall_s"],
            stats["wall_s"],
        )
        return stats

    def _copy_shadow_to_image(
        self,
        path: str,
        shadow: torch.nn.Module,
    ) -> tuple[set[int], int]:
        updated: set[int] = set()
        copied_bytes = 0
        seen_shadow_storages: set[tuple[int | None, int, int]] = set()
        copies = []
        for relative_name, tensor in iter_weight_tensors(shadow):
            if tensor.device.type != "cuda":
                continue
            shadow_key = _storage_key(tensor)
            if shadow_key in seen_shadow_storages:
                continue
            seen_shadow_storages.add(shadow_key)
            full_name = f"{path}.{relative_name}" if relative_name else path
            segment = self.image.segments_by_name.get(full_name)
            if segment is None:
                raise RuntimeError(
                    f"compiled shadow produced unknown weight {full_name!r}"
                )
            source = torch.empty(
                0,
                dtype=torch.uint8,
                device=tensor.device,
            ).set_(
                tensor.untyped_storage(),
                0,
                (tensor.untyped_storage().nbytes(),),
                (1,),
            )
            copies.append((segment, source))
            updated.add(id(segment))
            copied_bytes += segment.nbytes
        self.image.copy_device_segments_to_image(copies)
        return updated, copied_bytes

    def _copy_cpu_shadow_to_image(
        self,
        path: str,
        shadow: torch.nn.Module,
    ) -> tuple[set[int], int, int]:
        """Publish rebound CPU tensors while retaining image-backed writes."""

        updated: set[int] = set()
        runtime_bytes = 0
        copied_bytes = 0
        seen_shadow_storages: set[tuple[int | None, int, int]] = set()
        for relative_name, tensor in iter_weight_tensors(shadow):
            if tensor.device.type != "cpu":
                continue
            shadow_key = _storage_key(tensor)
            if shadow_key in seen_shadow_storages:
                continue
            seen_shadow_storages.add(shadow_key)
            full_name = f"{path}.{relative_name}" if relative_name else path
            segment = self.image.segments_by_name.get(full_name)
            if segment is None:
                raise RuntimeError(
                    f"compiled shadow produced unknown weight {full_name!r}"
                )
            source = torch.empty(0, dtype=torch.uint8).set_(
                tensor.untyped_storage(),
                0,
                (tensor.untyped_storage().nbytes(),),
                (1,),
            )
            if source.numel() != segment.nbytes:
                raise RuntimeError(
                    "compiled shadow storage size changed: "
                    f"name={full_name!r} source={source.numel()} "
                    f"target={segment.nbytes}"
                )
            target = self.image.image[
                segment.image_offset : segment.image_offset + segment.nbytes
            ]
            if source.data_ptr() != target.data_ptr():
                target.copy_(source)
                copied_bytes += segment.nbytes
            updated.add(id(segment))
            runtime_bytes += segment.nbytes
        return updated, runtime_bytes, copied_bytes

    @staticmethod
    def _weight_update_postprocess_device(shadow: torch.nn.Module) -> str:
        device = "cpu"
        for module_name, module in shadow.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is None:
                continue
            get_device = getattr(
                quant_method,
                "weight_update_postprocess_device",
                None,
            )
            method_device = get_device(module) if callable(get_device) else None
            if method_device not in {"cpu", "cuda"}:
                raise NotImplementedError(
                    "CPU weight staging is unsupported for quantization "
                    f"method {type(quant_method).__name__} at "
                    f"{module_name or '<root>'}"
                )
            if method_device == "cuda":
                device = "cuda"
        return device

    def _load_group_into_cpu_image(
        self,
        *,
        group_index: int,
        group: _WeightModuleGroup,
        names: list[str],
        get_tensor: Callable[[str], torch.Tensor],
    ) -> _LoadedWeightGroup:
        """Run the authoritative loader directly into one host-image range."""

        group_started = time.perf_counter()
        logger.debug(
            "Compiling CPU weight group %d/%d: path=%s "
            "estimated_bytes=%d checkpoint_tensors=%d",
            group_index,
            len(self.groups),
            group.path,
            group.nbytes,
            len(names),
        )

        cpu_image_storage_keys: set[tuple[int, int]] = set()

        def cpu_storage_factory(
            tensor: torch.Tensor,
            source_bytes: torch.Tensor,
        ) -> torch.Tensor:
            try:
                storage_bytes = self.image.storage_image_bytes(tensor)
            except KeyError:
                # Tensor attributes outside the CPU-image contract may
                # carry loader metadata. Keep them in bounded group scratch.
                storage_bytes = source_bytes.to("cpu").clone()
            if tensor.device.type == "cuda":
                storage = storage_bytes.untyped_storage()
                cpu_image_storage_keys.add(
                    (storage.data_ptr(), storage.nbytes()),
                )
            return storage_bytes

        phase_started = time.perf_counter()
        proxy, cpu_shadow = _build_weight_loader_proxy(
            self.model,
            group.path,
            target_device=torch.device("cpu"),
            copy_data=False,
            storage_factory=cpu_storage_factory,
        )
        cpu_clone_s = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        DefaultModelLoader.restore_weights_before_loading(
            cpu_shadow, torch.device("cpu")
        )
        restore_s = time.perf_counter() - phase_started

        weights = ((name, get_tensor(name)) for name in names)
        phase_started = time.perf_counter()
        with DefaultModelLoader.weight_loading_context(proxy):
            proxy.load_weights(weights)
        cpu_load_s = time.perf_counter() - phase_started
        del proxy, weights

        return _LoadedWeightGroup(
            group_index=group_index,
            group=group,
            checkpoint_tensors=len(names),
            cpu_shadow=cpu_shadow,
            cpu_image_storage_keys=cpu_image_storage_keys,
            group_started=group_started,
            cpu_clone_s=cpu_clone_s,
            restore_s=restore_s,
            cpu_load_s=cpu_load_s,
        )

    def _finalize_cpu_image_group(
        self,
        loaded: _LoadedWeightGroup,
    ) -> tuple[set[int], int, dict[str, Any]]:
        """Run post-load transforms and write final runtime bytes to the image."""

        group = loaded.group
        cpu_shadow = loaded.cpu_shadow
        background_h2d_bytes = sum(
            nbytes for _, nbytes in loaded.cpu_image_storage_keys
        )
        postprocess_device = self._weight_update_postprocess_device(cpu_shadow)
        if postprocess_device == "cpu":
            phase_started = time.perf_counter()
            for _, module in cpu_shadow.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)
            quant_submit_s = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            group_updated, group_bytes, cpu_image_copy_bytes = (
                self._copy_cpu_shadow_to_image(
                    group.path,
                    cpu_shadow,
                )
            )
            image_copy_s = time.perf_counter() - phase_started
            h2d_submit_s = 0.0
            device_sync_s = 0.0
            background_h2d_bytes = 0
            background_d2h_bytes = 0
            gpu_shadow = None
        else:
            model_state_ids = {
                id(tensor) for _, tensor in iter_weight_tensors(cpu_shadow)
            }

            def gpu_storage_factory(
                tensor: torch.Tensor,
                source_bytes: torch.Tensor,
            ) -> torch.Tensor:
                storage = tensor.untyped_storage()
                source_key = (storage.data_ptr(), storage.nbytes())
                target_device = (
                    self.target_device
                    if (
                        source_key in loaded.cpu_image_storage_keys
                        or id(tensor) in model_state_ids
                    )
                    else tensor.device
                )
                storage_bytes = torch.empty(
                    source_bytes.numel(),
                    dtype=torch.uint8,
                    device=target_device,
                )
                storage_bytes.copy_(source_bytes, non_blocking=True)
                return storage_bytes

            with torch.cuda.stream(self._compile_stream):
                phase_started = time.perf_counter()
                gpu_shadow = _clone_weight_module(
                    cpu_shadow,
                    target_device=self.target_device,
                    copy_data=True,
                    storage_factory=gpu_storage_factory,
                )
                h2d_submit_s = time.perf_counter() - phase_started

                phase_started = time.perf_counter()
                for _, module in gpu_shadow.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        quant_method.process_weights_after_loading(module)
                quant_submit_s = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            self._compile_stream.synchronize()
            device_sync_s = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            group_updated, group_bytes = self._copy_shadow_to_image(
                group.path,
                gpu_shadow,
            )
            image_copy_s = time.perf_counter() - phase_started
            cpu_image_copy_bytes = 0
            background_d2h_bytes = group_bytes
        stats = {
            "path": group.path,
            "checkpoint_tensors": loaded.checkpoint_tensors,
            "bytes": group_bytes,
            "postprocess_device": postprocess_device,
            "background_h2d_bytes": background_h2d_bytes,
            "background_d2h_bytes": background_d2h_bytes,
            "cpu_image_copy_bytes": cpu_image_copy_bytes,
            "cpu_clone_s": round(loaded.cpu_clone_s, 6),
            "restore_s": round(loaded.restore_s, 6),
            "cpu_load_s": round(loaded.cpu_load_s, 6),
            "h2d_submit_s": round(h2d_submit_s, 6),
            "quant_submit_s": round(quant_submit_s, 6),
            "device_sync_s": round(device_sync_s, 6),
            "image_copy_s": round(image_copy_s, 6),
            "wall_s": round(time.perf_counter() - loaded.group_started, 6),
        }
        logger.debug(
            "Compiled CPU weight group %d/%d: path=%s "
            "bytes=%d postprocess_device=%s wall_s=%.6f "
            "cpu_load_s=%.6f h2d_submit_s=%.6f "
            "device_sync_s=%.6f image_copy_s=%.6f",
            loaded.group_index,
            len(self.groups),
            group.path,
            group_bytes,
            stats["postprocess_device"],
            stats["wall_s"],
            stats["cpu_load_s"],
            stats["h2d_submit_s"],
            stats["device_sync_s"],
            stats["image_copy_s"],
        )
        del cpu_shadow, gpu_shadow
        gc.collect(0)
        return group_updated, group_bytes, stats

    def _get_canonical_checkpoint(
        self,
        *,
        capacity: int,
        signature: tuple[str, tuple[tuple[str, int], ...]],
    ) -> tuple[_HostSharedCheckpoint, bool]:
        if self._canonical_checkpoint is not None:
            if (
                self._canonical_checkpoint.capacity < capacity
                or self._canonical_checkpoint_signature != signature
            ):
                self._discard_canonical_checkpoint(
                    "canonical checkpoint layout changed",
                )
            else:
                return self._canonical_checkpoint, False
        self._canonical_checkpoint = _HostSharedCheckpoint(
            capacity=capacity,
            cpu_group=self.host_cpu_group,
        )
        self._canonical_checkpoint_signature = signature
        self._canonical_checkpoint_version = None
        return self._canonical_checkpoint, True

    def _populate_canonical_checkpoint(
        self,
        *,
        root: Path,
        filenames: list[str],
        file_sizes: dict[str, int],
        offsets: dict[str, int],
        checkpoint: _HostSharedCheckpoint,
    ) -> dict[str, Any]:
        """Read one complete canonical checkpoint into host-shared CPU memory."""

        started = time.perf_counter()
        distributed = torch.distributed.is_initialized()
        cpu_group = self.host_cpu_group
        world_size = (
            torch.distributed.get_world_size(group=cpu_group) if distributed else 1
        )
        rank = torch.distributed.get_rank(group=cpu_group) if distributed else 0
        owned_reads = []
        local_error = None
        try:
            for file_index, filename in enumerate(filenames):
                if file_index % world_size != rank:
                    continue
                file_nbytes = file_sizes[filename]
                wall_s = _pread_file_to_tensor(
                    root / filename,
                    checkpoint.view(
                        file_nbytes,
                        offset=offsets[filename],
                    ),
                    drop_cache_after_read=True,
                )
                owned_reads.append(
                    {
                        "filename": filename,
                        "bytes": file_nbytes,
                        "wall_s": round(wall_s, 6),
                    }
                )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"

        if world_size > 1:
            errors: list[str | None] = [None] * world_size
            torch.distributed.all_gather_object(
                errors,
                local_error,
                group=cpu_group,
            )
        else:
            errors = [local_error]
        errors = [error for error in errors if error is not None]
        if errors:
            raise RuntimeError(
                "failed to populate canonical CPU checkpoint: " + "; ".join(errors)
            )

        stats = checkpoint.stats()
        stats.update(
            {
                "operation": "populate_canonical_cpu_checkpoint",
                "persistent_canonical_checkpoint": True,
                "base_version": 0,
                "files": len(filenames),
                "checkpoint_bytes": sum(file_sizes.values()),
                "owner_rank": rank,
                "owned_reads": owned_reads,
                "owned_bytes": sum(item["bytes"] for item in owned_reads),
                "wall_s": round(time.perf_counter() - started, 6),
            }
        )
        return stats

    def _discard_canonical_checkpoint(self, reason: str) -> None:
        checkpoint = self._canonical_checkpoint
        self._canonical_checkpoint = None
        self._canonical_checkpoint_signature = None
        self._canonical_lineage = None
        self._canonical_checkpoint_version = None
        gc.collect()
        if checkpoint is not None:
            checkpoint.close()
        logger.warning("Discarded canonical CPU checkpoint: %s", reason)

    def _canonical_checkpoint_version_for_lineage(
        self,
        *,
        base_checkpoint_dir: str,
        checkpoint_source_dir: str,
    ) -> int:
        """Return a reusable canonical version, or reset another lineage."""

        if self._canonical_checkpoint is None:
            return 0
        requested_lineage = (
            os.path.realpath(base_checkpoint_dir),
            os.path.realpath(checkpoint_source_dir),
        )
        if (
            self._canonical_checkpoint_version == 0
            and self._canonical_checkpoint_signature is not None
            and self._canonical_checkpoint_signature[0] == requested_lineage[0]
        ):
            # The same immutable v0 checkpoint may anchor a new publisher run.
            self._canonical_lineage = requested_lineage
            return 0
        if self._canonical_lineage != requested_lineage:
            self._discard_canonical_checkpoint(
                "checkpoint lineage changed",
            )
            return 0
        return self._canonical_checkpoint_version or 0

    def _compile_memory_checkpoint(
        self,
        *,
        root: Path,
        weight_map: dict[str, str],
        names_by_group: dict[str, list[str]],
        source_stats: list[dict[str, Any]],
        checkpoint_transform: Any,
    ):
        filenames = sorted(set(weight_map.values()))
        (
            file_sizes,
            offsets,
            capacity,
            signature,
        ) = self._run_on_all_host_ranks(
            "canonical checkpoint layout inspection",
            lambda: _canonical_checkpoint_layout(root, filenames),
        )
        checkpoint, created = self._get_canonical_checkpoint(
            capacity=capacity,
            signature=signature,
        )
        expected_version = int(getattr(checkpoint_transform, "canonical_version", 0))
        if created:
            source_stats.append(
                self._populate_canonical_checkpoint(
                    root=root,
                    filenames=filenames,
                    file_sizes=file_sizes,
                    offsets=offsets,
                    checkpoint=checkpoint,
                )
            )
            self._canonical_checkpoint_version = expected_version
        elif self._canonical_checkpoint_version != expected_version:
            raise RuntimeError(
                "canonical CPU checkpoint version mismatch: "
                f"checkpoint={self._canonical_checkpoint_version} "
                f"delta_base={expected_version}"
            )

        distributed = torch.distributed.is_initialized()
        cpu_group = self.host_cpu_group
        world_size = (
            torch.distributed.get_world_size(group=cpu_group) if distributed else 1
        )
        rank = torch.distributed.get_rank(group=cpu_group) if distributed else 0
        source_stats.append(
            _transform_canonical_checkpoint(
                filenames=filenames,
                file_sizes=file_sizes,
                offsets=offsets,
                checkpoint=checkpoint,
                checkpoint_transform=checkpoint_transform,
                rank=rank,
                world_size=world_size,
                cpu_group=cpu_group,
            )
        )

        handles = {
            filename: _InMemorySafetensorsFile(
                checkpoint.view(file_sizes[filename], offset=offsets[filename])
            )
            for filename in filenames
        }
        try:
            for group_index, group in enumerate(self.groups, start=1):
                loaded = self._load_group_into_cpu_image(
                    group_index=group_index,
                    group=group,
                    names=names_by_group[group.path],
                    get_tensor=lambda name: handles[weight_map[name]].get_tensor(name),
                )
                yield self._finalize_cpu_image_group(loaded)
        finally:
            handles.clear()
            gc.collect()

    def _compile_disk_checkpoint(
        self,
        *,
        root: Path,
        weight_map: dict[str, str],
        names_by_group: dict[str, list[str]],
        source_stats: list[dict[str, Any]] | None = None,
        checkpoint_transform: Any = None,
    ):
        """Compile bounded groups from canonical bytes read once per host."""

        if source_stats is None:
            source_stats = []
        if checkpoint_transform is None:
            checkpoint_transform = _NoOpCheckpointTransform()
        read_parallelism = (
            torch.distributed.get_world_size(group=self.host_cpu_group)
            if torch.distributed.is_initialized()
            else 1
        )
        groups = _shared_checkpoint_groups(
            root=root,
            weight_map=weight_map,
            names_by_group=names_by_group,
            read_parallelism=read_parallelism,
        )
        _validate_shared_checkpoint_groups(
            groups,
            cpu_group=self.host_cpu_group,
        )
        sources = [groups[group.path] for group in self.groups]
        distributed = torch.distributed.is_initialized()
        world_size = (
            torch.distributed.get_world_size(group=self.host_cpu_group)
            if distributed
            else 1
        )
        rank = (
            torch.distributed.get_rank(group=self.host_cpu_group) if distributed else 0
        )
        writer = None
        if not isinstance(checkpoint_transform, _NoOpCheckpointTransform):

            def build_writer():
                nonlocal writer
                writer = _CanonicalCheckpointWriter(
                    root=root,
                    sources=sources,
                    operations_by_file=checkpoint_transform.operations_by_file,
                    rank=rank,
                    world_size=world_size,
                    drop_cache_after_write=self.drop_cache_after_load,
                )

            try:
                self._run_on_all_host_ranks(
                    "canonical checkpoint writer construction",
                    build_writer,
                )
            except Exception:
                if writer is not None:
                    writer.close()
                raise
        capacities = [
            max(
                (
                    source.buffer_bytes
                    for index, source in enumerate(sources)
                    if index % 2 == buffer_index
                ),
                default=0,
            )
            for buffer_index in range(2)
        ]
        shared_buffers = [
            HostSharedMemoryBuffer(
                nbytes=max(1, capacity),
                cpu_group=self.host_cpu_group,
                name=f"weight-checkpoint-group-{index}",
            )
            for index, capacity in enumerate(capacities)
        ]
        shared_buffer_bytes = sum(buffer.nbytes for buffer in shared_buffers)
        try:
            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="weight-checkpoint-read",
            ) as reader:

                def submit_read(index: int):
                    return reader.submit(
                        _populate_shared_checkpoint_group,
                        root=root,
                        group=sources[index],
                        buffer=shared_buffers[index % 2],
                        cpu_group=self.host_cpu_group,
                        drop_cache_after_read=self.drop_cache_after_load,
                    )

                pending_read = submit_read(0) if sources else None
                for source_index, (group, source) in enumerate(
                    zip(self.groups, sources)
                ):
                    wait_started = time.perf_counter()
                    read_result = None
                    read_error = None
                    try:
                        assert pending_read is not None
                        read_result = pending_read.result()
                    except Exception as exc:
                        read_error = f"{type(exc).__name__}: {exc}"

                    def finish_read():
                        if read_error is not None:
                            raise RuntimeError(read_error)
                        return read_result

                    read_result = self._run_on_all_host_ranks(
                        f"canonical checkpoint read for group {group.path!r}",
                        finish_read,
                    )
                    read_wait_s = time.perf_counter() - wait_started
                    next_index = source_index + 1
                    pending_read = (
                        submit_read(next_index) if next_index < len(sources) else None
                    )
                    tensor_file = _SharedCheckpointGroupView(
                        shared_buffers[source_index % 2],
                        source.tensors,
                    )
                    transform_stats = None
                    if writer is not None:

                        def transform_group():
                            owned_names = [
                                name
                                for name in names_by_group[group.path]
                                if writer.owns(name)
                            ]
                            result = checkpoint_transform.transform_tensors(
                                {
                                    name: tensor_file.get_tensor_bytes(name)
                                    for name in owned_names
                                },
                                description=group.path,
                                write_tensor=writer.record_dirty_ranges,
                                write_block_bytes=writer.write_block_bytes,
                            )
                            return result

                        transform_stats = self._run_on_all_host_ranks(
                            f"canonical delta transform for group {group.path!r}",
                            transform_group,
                        )
                        source_stats.append(transform_stats)
                        self._run_on_all_host_ranks(
                            f"canonical checkpoint writes for group {group.path!r}",
                            functools.partial(
                                writer.persist_pending_group,
                                tensor_file.get_tensor_bytes,
                            ),
                        )

                    def compile_group():
                        if writer is not None:
                            # The write barrier above makes every positional
                            # update visible before one rank flushes and closes
                            # the completed checkpoint files.
                            writer.finish_group(source_index)
                        loaded = self._load_group_into_cpu_image(
                            group_index=next_index,
                            group=group,
                            names=names_by_group[group.path],
                            get_tensor=tensor_file.get_tensor,
                        )
                        try:
                            return self._finalize_cpu_image_group(loaded)
                        finally:
                            del loaded

                    result = self._run_on_all_host_ranks(
                        f"runtime compilation for group {group.path!r}",
                        compile_group,
                    )
                    stats = result[2]
                    stats["canonical_read_s"] = read_result["wall_s"]
                    stats["canonical_read_wait_s"] = round(read_wait_s, 6)
                    stats["canonical_read_bytes"] = source.source_bytes
                    stats["canonical_read_runs"] = len(source.reads)
                    stats["canonical_direct_io_bytes"] = sum(
                        read.nbytes for read in source.reads if read.direct_io
                    )
                    stats["canonical_rank_direct_io_bytes"] = read_result.get(
                        "owned_direct_io_bytes",
                        0,
                    )
                    stats["canonical_buffer_bytes"] = source.buffer_bytes
                    stats["canonical_shared_buffer_bytes"] = shared_buffer_bytes
                    if transform_stats is not None:
                        stats["canonical_transform"] = transform_stats
                        stats["canonical_transform_s"] = transform_stats["wall_s"]
                    stats["wall_s"] = round(
                        stats["wall_s"]
                        + read_wait_s
                        + (
                            0.0
                            if transform_stats is None
                            else transform_stats["wall_s"]
                        ),
                        6,
                    )
                    yield result
            if writer is not None:
                self._run_on_all_host_ranks(
                    "canonical checkpoint persistence",
                    writer.validate,
                )
        finally:
            logger.info(
                "Releasing shared canonical checkpoint group buffer: bytes=%d",
                shared_buffer_bytes,
            )
            started = time.perf_counter()
            gc.collect()
            for shared_buffer in shared_buffers:
                shared_buffer.close()
            if writer is not None:
                source_stats.append(
                    {
                        "operation": "persist_canonical_checkpoint",
                        **writer.stats(),
                    }
                )
                writer.close()
            logger.info(
                "Released shared canonical checkpoint group buffer: "
                "bytes=%d wall_time=%.3fs",
                shared_buffer_bytes,
                time.perf_counter() - started,
            )

    def _stage_from_checkpoint(
        self,
        *,
        checkpoint_dir: str,
        target_version: int,
        checkpoint_transform: Any,
    ) -> dict[str, Any]:
        """Compile every bounded module from a complete canonical checkpoint."""

        started = time.perf_counter()
        host_rank_sync_started = self._host_rank_sync_s

        def inspect_checkpoint():
            weight_map, root = _checkpoint_weight_map(checkpoint_dir)
            return (
                weight_map,
                root,
                _map_checkpoint_names_to_groups(
                    self.model,
                    weight_map,
                    self.groups,
                ),
            )

        weight_map, root, group_for_name = self._run_on_all_host_ranks(
            "checkpoint inspection",
            inspect_checkpoint,
        )
        names_by_group: dict[str, list[str]] = {group.path: [] for group in self.groups}
        unmapped = []
        for name, group_path in group_for_name.items():
            if group_path is None:
                unmapped.append(name)
            else:
                names_by_group[group_path].append(name)
        if unmapped:
            raise RuntimeError(
                "CPU weight staging cannot map every checkpoint tensor "
                f"to a runtime weight group; unmapped={unmapped[:20]}"
            )

        updated_segments: set[int] = set()
        copied_bytes = 0
        group_stats = []
        source_stats = []
        try:

            def begin_stage():
                if not self.image.staging and not self.image.valid:
                    # A failed stage may have partially overwritten the
                    # sole host image while the active CUDA model remains
                    # unchanged. Restore the ordinary in-place reload state so
                    # checkpoint-optional weights cannot inherit failed bytes.
                    self.image.capture_active_weights()
                self.image.begin_stage(target_version)
                self.image.register_host_memory()

            self._run_on_all_host_ranks(
                f"CPU weight image staging of version {target_version}",
                begin_stage,
            )
            progress_interval = max(1, math.ceil(len(self.groups) / 10))
            if self.canonical_checkpoint_storage == "memory":
                compiler = self._compile_memory_checkpoint(
                    root=root,
                    weight_map=weight_map,
                    names_by_group=names_by_group,
                    source_stats=source_stats,
                    checkpoint_transform=checkpoint_transform,
                )
            else:
                compiler = self._compile_disk_checkpoint(
                    root=root,
                    weight_map=weight_map,
                    names_by_group=names_by_group,
                    source_stats=source_stats,
                    checkpoint_transform=checkpoint_transform,
                )
            for (
                group_updated,
                group_bytes,
                stats,
            ) in compiler:
                updated_segments.update(group_updated)
                copied_bytes += group_bytes
                group_stats.append(stats)
                completed_groups = len(group_stats)
                if (
                    completed_groups == 1
                    or completed_groups % progress_interval == 0
                    or completed_groups == len(self.groups)
                ):
                    logger.info(
                        "CPU weight image %s progress: groups=%d/%d "
                        "bytes=%d elapsed=%.3fs",
                        target_version,
                        completed_groups,
                        len(self.groups),
                        copied_bytes,
                        time.perf_counter() - started,
                    )

            expected_segments = {id(value) for value in self.image.segments}
            missing = expected_segments - updated_segments
            if missing:
                missing_names = [
                    segment.name
                    for segment in self.image.segments
                    if id(segment) in missing
                ]
                raise RuntimeError(
                    "checkpoint did not produce every runtime weight "
                    f"storage; missing={missing_names[:20]}"
                )
            self.image.finish_stage(target_version)
        except Exception as exc:
            self.image.invalidate(
                f"checkpoint compilation of version {target_version} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            raise

        phase_totals = {
            phase: round(
                sum(value.get(phase, 0.0) for value in group_stats),
                6,
            )
            for phase in (
                "canonical_read_s",
                "canonical_read_wait_s",
                "canonical_transform_s",
                "cpu_clone_s",
                "restore_s",
                "cpu_load_s",
                "h2d_submit_s",
                "quant_submit_s",
                "device_sync_s",
                "image_copy_s",
            )
        }
        phase_totals["host_rank_sync_s"] = round(
            self._host_rank_sync_s - host_rank_sync_started,
            6,
        )
        traffic = {
            name: sum(value.get(name, 0) for value in group_stats)
            for name in (
                "background_h2d_bytes",
                "background_d2h_bytes",
                "cpu_image_copy_bytes",
            )
        }
        postprocess_bytes = {
            device: sum(
                value["bytes"]
                for value in group_stats
                if value.get("postprocess_device") == device
            )
            for device in ("cpu", "cuda")
        }
        if self.canonical_checkpoint_storage == "memory":
            load_stats = [
                value
                for value in source_stats
                if value.get("operation") == "populate_canonical_cpu_checkpoint"
            ]
            transform_stats = next(
                value
                for value in source_stats
                if value.get("operation") == "transform_canonical_checkpoint"
            )
            source_summary = {
                "storage": "memory",
                "checkpoint_bytes": (
                    self._canonical_checkpoint.capacity
                    if self._canonical_checkpoint is not None
                    else 0
                ),
                "loaded_from_disk": bool(load_stats),
                "load_wall_s": round(
                    sum(value["wall_s"] for value in load_stats),
                    6,
                ),
                "transform_wall_s": transform_stats["transform_wall_s"],
                "synchronization_wall_s": transform_stats["verify_barrier_s"],
            }
        else:
            canonical_read_s = sum(value["canonical_read_s"] for value in group_stats)
            canonical_read_wait_s = sum(
                value["canonical_read_wait_s"] for value in group_stats
            )
            cpu_load_s = sum(value["cpu_load_s"] for value in group_stats)
            transform_stats = [
                value
                for value in source_stats
                if value.get("operation") == "canonical_delta_transform"
            ]
            persist_stats = next(
                (
                    value
                    for value in source_stats
                    if value.get("operation") == "persist_canonical_checkpoint"
                ),
                {},
            )
            source_summary = {
                "storage": "disk",
                "checkpoint_bytes": sum(
                    (root / filename).stat().st_size
                    for filename in set(weight_map.values())
                ),
                "loaded_from_disk": True,
                "physical_host_read_bytes": sum(
                    value["canonical_read_bytes"] for value in group_stats
                ),
                "direct_io_bytes": sum(
                    value["canonical_direct_io_bytes"] for value in group_stats
                ),
                "rank_direct_io_bytes": sum(
                    value["canonical_rank_direct_io_bytes"] for value in group_stats
                ),
                "shared_buffer_bytes": max(
                    value["canonical_shared_buffer_bytes"] for value in group_stats
                ),
                "read_runs": sum(value["canonical_read_runs"] for value in group_stats),
                "read_wall_s": round(canonical_read_s, 6),
                "read_wait_wall_s": round(canonical_read_wait_s, 6),
                "cpu_load_wall_s": round(cpu_load_s, 6),
                "load_wall_s": round(canonical_read_s + cpu_load_s, 6),
                "transform_wall_s": round(
                    sum(value["wall_s"] for value in transform_stats),
                    6,
                ),
                "synchronization_wall_s": 0.0,
                **{
                    key: value
                    for key, value in persist_stats.items()
                    if key != "operation"
                },
            }
        wall_s = round(time.perf_counter() - started, 6)
        logger.info(
            "Staged CPU weight image %s: bytes=%d wall_time=%.3fs "
            "source=%s phases=%s",
            target_version,
            copied_bytes,
            wall_s,
            source_summary,
            phase_totals,
        )
        return {
            "operation": "stage_cpu_weight_update",
            "target_version": target_version,
            "groups": len(self.groups),
            "checkpoint_tensors": len(weight_map),
            "runtime_storages": len(updated_segments),
            "bytes": copied_bytes,
            "transport": (
                "canonical_cpu_checkpoint"
                if self.canonical_checkpoint_storage == "memory"
                else "canonical_disk_checkpoint"
            ),
            "wall_s": wall_s,
            "compile_wall_s": round(
                sum(value["wall_s"] for value in group_stats),
                6,
            ),
            "source": source_summary,
            "phases": phase_totals,
            "postprocess_bytes": postprocess_bytes,
            "traffic": traffic,
        }

    def stage_from_checkpoint(
        self,
        *,
        checkpoint_dir: str,
        target_version: int,
    ) -> dict[str, Any]:
        """Compile a verified local checkpoint into the rank-ready CPU image."""

        if self.canonical_checkpoint_storage != "disk":
            raise RuntimeError(
                "stage_from_checkpoint requires a disk-backed canonical checkpoint"
            )
        stats = self._stage_from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            target_version=target_version,
            checkpoint_transform=_NoOpCheckpointTransform(),
        )
        stats["canonical_checkpoint"] = {
            "version": target_version,
            "bytes": stats["source"]["checkpoint_bytes"],
            "storage": "disk",
            "physical_host_copies": 1,
        }
        return stats

    def stage_from_disk_delta_lineage(
        self,
        *,
        checkpoint_dir: str,
        base_checkpoint_dir: str,
        checkpoint_source_dir: str,
        target_version: int,
    ) -> dict[str, Any]:
        """Persist and compile verified delta bytes from one bounded buffer."""

        if self.canonical_checkpoint_storage != "disk":
            raise RuntimeError(
                "stage_from_disk_delta_lineage requires a disk-backed "
                "canonical checkpoint"
            )

        from sglang.srt.weight_sync import disk_checkpoint
        from sglang.srt.weight_sync.cpu_delta_checkpoint import (
            DeltaCheckpointTransform,
        )

        distributed = torch.distributed.is_initialized()
        world_size = (
            torch.distributed.get_world_size(group=self.host_cpu_group)
            if distributed
            else 1
        )
        rank = (
            torch.distributed.get_rank(group=self.host_cpu_group) if distributed else 0
        )
        transaction = None
        transform = None
        mutation_started = False
        started = time.perf_counter()
        try:

            def acquire_transaction():
                nonlocal transaction
                if rank == 0:
                    candidate = disk_checkpoint.LocalCheckpointTransaction(
                        checkpoint_dir
                    )
                    candidate.__enter__()
                    transaction = candidate

            self._run_on_all_host_ranks(
                "canonical checkpoint transaction acquisition",
                acquire_transaction,
            )
            versions = [None] * world_size
            local_version = (
                transaction.initial_version if transaction is not None else None
            )
            if distributed:
                torch.distributed.all_gather_object(
                    versions,
                    local_version,
                    group=self.host_cpu_group,
                )
            else:
                versions[0] = local_version
            canonical_version = versions[0]

            if canonical_version is not None and canonical_version <= target_version:
                transform = DeltaCheckpointTransform(
                    base_checkpoint_dir=base_checkpoint_dir,
                    checkpoint_source_dir=checkpoint_source_dir,
                    target_version=target_version,
                    cpu_group=self.host_cpu_group,
                    canonical_version=canonical_version,
                    canonical_checkpoint_layout_dir=checkpoint_dir,
                )
                can_fuse = (
                    transform.canonical_version == canonical_version
                    and os.path.realpath(transform.checkpoint_root)
                    == os.path.realpath(checkpoint_dir)
                )
            else:
                can_fuse = False

            if not can_fuse:
                logger.info(
                    "Rebuilding the NVMe canonical checkpoint before staging "
                    "version %d",
                    target_version,
                )
                if transform is not None:
                    transform.close()
                    transform = None
                if transaction is not None:
                    transaction.__exit__(None, None, None)
                    transaction = None
                self._run_on_all_host_ranks(
                    "canonical checkpoint transaction release",
                    lambda: None,
                )
                materialization = None

                def materialize_checkpoint():
                    nonlocal materialization
                    if rank == 0:
                        materialization = disk_checkpoint.materialize(
                            local_checkpoint_dir=checkpoint_dir,
                            base_checkpoint_dir=base_checkpoint_dir,
                            checkpoint_source_dir=checkpoint_source_dir,
                            target_version=target_version,
                            drop_cache_after_seed=self.drop_cache_after_load,
                        )

                self._run_on_all_host_ranks(
                    "canonical checkpoint fallback materialization",
                    materialize_checkpoint,
                )
                stats = self._stage_from_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    target_version=target_version,
                    checkpoint_transform=_NoOpCheckpointTransform(),
                )
                stats["canonical_checkpoint_materialization"] = materialization
                return stats

            assert canonical_version is not None
            logger.info(
                "Staging version %d from NVMe canonical checkpoint version %d "
                "with one bounded read/write pass",
                target_version,
                canonical_version,
            )

            def begin_mutation():
                if transaction is not None:
                    transaction.begin(canonical_version)

            self._run_on_all_host_ranks(
                "canonical checkpoint mutation",
                begin_mutation,
            )
            mutation_started = True
            stats = self._stage_from_checkpoint(
                checkpoint_dir=checkpoint_dir,
                target_version=target_version,
                checkpoint_transform=transform,
            )

            def commit_mutation():
                if transaction is not None:
                    transaction.commit(target_version)

            self._run_on_all_host_ranks(
                "canonical checkpoint commit",
                commit_mutation,
            )
            stats["delta_setup"] = transform.setup_stats
            stats["canonical_checkpoint"] = {
                "version": target_version,
                "bytes": stats["source"]["checkpoint_bytes"],
                "storage": "disk",
                "physical_host_copies": 1,
            }
            stats["wall_s"] = round(time.perf_counter() - started, 6)
            return stats
        except Exception:
            if mutation_started:
                self.image.invalidate(
                    f"disk-backed canonical staging of v{target_version} failed"
                )
            raise
        finally:
            if transform is not None:
                transform.close()
            if transaction is not None:
                transaction.__exit__(None, None, None)

    def stage_from_delta_lineage(
        self,
        *,
        base_checkpoint_dir: str,
        checkpoint_source_dir: str,
        target_version: int,
    ) -> dict[str, Any]:
        """Reconstruct and compile a target without materializing it on disk."""

        if self.canonical_checkpoint_storage != "memory":
            raise RuntimeError(
                "stage_from_delta_lineage requires an in-memory canonical checkpoint"
            )
        started = time.perf_counter()
        from sglang.srt.weight_sync.cpu_delta_checkpoint import (
            DeltaCheckpointTransform,
            validate_delta_target,
        )

        validate_delta_target(checkpoint_source_dir, target_version)
        canonical_version = self._canonical_checkpoint_version_for_lineage(
            base_checkpoint_dir=base_checkpoint_dir,
            checkpoint_source_dir=checkpoint_source_dir,
        )
        if target_version < canonical_version:
            self._discard_canonical_checkpoint(
                f"requested rollback from v{canonical_version} to v{target_version}",
            )
            canonical_version = 0
        try:
            with DeltaCheckpointTransform(
                base_checkpoint_dir=base_checkpoint_dir,
                checkpoint_source_dir=checkpoint_source_dir,
                target_version=target_version,
                cpu_group=self.host_cpu_group,
                canonical_version=canonical_version,
                canonical_checkpoint_layout_dir=(
                    None
                    if self._canonical_checkpoint_signature is None
                    else self._canonical_checkpoint_signature[0]
                ),
            ) as delta_transform:
                delta_setup_stats = delta_transform.setup_stats
                stats = self._stage_from_checkpoint(
                    checkpoint_dir=str(delta_transform.checkpoint_root),
                    target_version=target_version,
                    checkpoint_transform=delta_transform,
                )
            self._canonical_checkpoint_version = (
                target_version if self._canonical_checkpoint is not None else None
            )
            if self._canonical_checkpoint is not None:
                self._canonical_lineage = (
                    os.path.realpath(base_checkpoint_dir),
                    os.path.realpath(checkpoint_source_dir),
                )
        except Exception:
            # A transform or loader failure may leave an unknown subset of the
            # canonical bytes advanced. Fail closed: the live GPU model remains
            # untouched, and a retry reconstructs from the immutable base
            # checkpoint.
            self._discard_canonical_checkpoint(
                f"staging of v{target_version} did not complete",
            )
            raise
        stats["delta_setup"] = delta_setup_stats
        stats["canonical_checkpoint"] = {
            "version": self._canonical_checkpoint_version,
            "bytes": (
                0
                if self._canonical_checkpoint is None
                else self._canonical_checkpoint.capacity
            ),
            "storage": "memory",
            "physical_host_copies": 1,
        }
        stats["wall_s"] = round(time.perf_counter() - started, 6)
        return stats

    def invalidate_stage(self, reason: str) -> None:
        """Invalidate the staged image and canonical checkpoint after any rank fails."""

        self.image.invalidate(reason)
        self._discard_canonical_checkpoint(reason)

    def close(self, reason: str) -> None:
        self._discard_canonical_checkpoint(reason)
        self.image.close()
        if self._host_rank_coordinator is not None:
            self._host_rank_coordinator.close()
            self._host_rank_coordinator = None

    def commit(
        self,
        target_version: int,
    ) -> dict[str, float | int | str]:
        return self.image.commit(target_version)
