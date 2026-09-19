"""Bounded positional I/O used during weight staging."""

from __future__ import annotations

import errno
import os
import resource
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_POSITIONAL_IO_CHUNK_BYTES = 64 << 20
_DIRECT_IO_ALIGNMENT = 4096
_DIRECT_IO_FALLBACK_ERRORS = {
    errno.EINVAL,
    errno.ENOTSUP,
    errno.EOPNOTSUPP,
    errno.EPERM,
}


@dataclass(frozen=True)
class PositionalReadResult:
    wall_s: float
    direct_io: bool


def read_exact(reader: Any, nbytes: int) -> bytearray:
    """Read exactly ``nbytes`` from a binary stream."""

    result = bytearray(nbytes)
    view = memoryview(result)
    position = 0
    while position < nbytes:
        nread = reader.readinto(view[position:])
        if not nread:
            raise EOFError(
                "unexpected end of compressed payload: "
                f"expected={nbytes} actual={position}"
            )
        position += nread
    return result


class PositionalFileRangeReader:
    """Expose one immutable file range without advancing a shared descriptor."""

    def __init__(
        self,
        fd: int,
        offset: int,
        nbytes: int,
        path: str | Path,
        *,
        max_read_bytes: int,
    ):
        if offset < 0 or nbytes < 0 or max_read_bytes <= 0:
            raise ValueError("positional file ranges must be non-negative and bounded")
        self.fd = fd
        self.offset = offset
        self.nbytes = nbytes
        self.path = Path(path)
        self.max_read_bytes = max_read_bytes
        self.position = 0
        self.read_wall_s = 0.0

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        remaining = self.nbytes - self.position
        if remaining == 0:
            return b""
        size = min(
            remaining,
            self.max_read_bytes if size < 0 else size,
            self.max_read_bytes,
        )
        started = time.perf_counter()
        data = os.pread(self.fd, size, self.offset + self.position)
        self.read_wall_s += time.perf_counter() - started
        if not data:
            raise EOFError(
                f"unexpected EOF reading {self.path}: offset={self.offset} "
                f"expected={self.nbytes} actual={self.position}"
            )
        self.position += len(data)
        return data


@dataclass
class _CachedFileDescriptor:
    fd: int
    users: int = 0
    dirty: bool = False
    last_used: int = 0


class FileDescriptorCache:
    """Share positional-I/O descriptors under a fixed process-local bound."""

    def __init__(self, flags: int, limit: int):
        if limit <= 0:
            raise ValueError("file descriptor cache limit must be positive")
        self.flags = flags
        self.limit = limit
        self.entries: dict[str, _CachedFileDescriptor] = {}
        self.condition = threading.Condition()
        self.clock = 0
        self.peak_open_files = 0

    @contextmanager
    def acquire(self, path: str | Path, *, write: bool = False):
        path = str(path)
        with self.condition:
            entry = self.entries.get(path)
            while entry is None:
                if len(self.entries) < self.limit:
                    entry = _CachedFileDescriptor(fd=os.open(path, self.flags))
                    self.entries[path] = entry
                    self.peak_open_files = max(
                        self.peak_open_files,
                        len(self.entries),
                    )
                    break
                idle_path, idle = min(
                    (
                        (candidate_path, candidate)
                        for candidate_path, candidate in self.entries.items()
                        if candidate.users == 0
                    ),
                    key=lambda item: item[1].last_used,
                    default=(None, None),
                )
                if idle is None:
                    self.condition.wait()
                    entry = self.entries.get(path)
                    continue
                del self.entries[idle_path]
                self._close(idle)
            entry.users += 1
            self.clock += 1
            entry.last_used = self.clock
            if write:
                entry.dirty = True
        try:
            yield entry.fd
        finally:
            with self.condition:
                entry.users -= 1
                self.clock += 1
                entry.last_used = self.clock
                self.condition.notify()

    def flush(self) -> None:
        with self.condition:
            for entry in self.entries.values():
                if entry.dirty:
                    os.fsync(entry.fd)
                    entry.dirty = False

    def close(self) -> None:
        with self.condition:
            entries = list(self.entries.values())
            self.entries.clear()
        error = None
        for entry in entries:
            try:
                self._close(entry)
            except OSError as exc:
                error = error or exc
        if error is not None:
            raise error

    @staticmethod
    def _close(entry: _CachedFileDescriptor) -> None:
        try:
            if entry.dirty:
                os.fsync(entry.fd)
        finally:
            os.close(entry.fd)


def file_descriptor_cache_limit(
    *,
    concurrent_caches: int = 1,
    max_cached_file_descriptors: int = 256,
) -> int:
    """Return a per-cache limit while reserving descriptors for the server."""

    if concurrent_caches <= 0:
        raise ValueError("concurrent_caches must be positive")
    if max_cached_file_descriptors <= 0:
        raise ValueError("max_cached_file_descriptors must be positive")
    reserve = 64
    try:
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        soft_limit = max_cached_file_descriptors * concurrent_caches + reserve
    if soft_limit == resource.RLIM_INFINITY:
        soft_limit = max_cached_file_descriptors * concurrent_caches + reserve
    return max(
        1,
        min(
            max_cached_file_descriptors,
            max(concurrent_caches, soft_limit - reserve) // concurrent_caches,
        ),
    )


def read_file_into_tensor(
    path: str | Path,
    target: torch.Tensor,
    *,
    drop_cache_after_read: bool = False,
) -> PositionalReadResult:
    path = Path(path)
    file_nbytes = path.stat().st_size
    if target.numel() != file_nbytes:
        raise ValueError(
            f"source buffer size mismatch for {path}: "
            f"buffer={target.numel()} file={file_nbytes}"
        )
    return read_range_into_tensor(
        path,
        target,
        file_offset=0,
        drop_cache_after_read=drop_cache_after_read,
    )


def read_range_into_tensor(
    path: str | Path,
    target: torch.Tensor,
    *,
    file_offset: int,
    direct_io: bool = False,
    drop_cache_after_read: bool = False,
) -> PositionalReadResult:
    """Read one immutable file range into contiguous CPU byte storage."""

    path = Path(path)
    if (
        target.device.type != "cpu"
        or target.dtype != torch.uint8
        or target.ndim != 1
        or not target.is_contiguous()
    ):
        raise ValueError("positional read target must be contiguous CPU bytes")
    file_nbytes = path.stat().st_size
    if file_offset < 0 or file_offset + target.numel() > file_nbytes:
        raise ValueError(
            f"source range exceeds {path}: offset={file_offset} "
            f"bytes={target.numel()} file={file_nbytes}"
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
    view = memoryview(target.numpy()).cast("B")
    use_direct_io = direct_io and hasattr(os, "O_DIRECT")
    flags = os.O_RDONLY | (os.O_DIRECT if use_direct_io else 0)
    fd = None
    try:
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            if not use_direct_io or exc.errno not in _DIRECT_IO_FALLBACK_ERRORS:
                raise
            use_direct_io = False
            fd = os.open(path, os.O_RDONLY)

        offset = 0
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
            if not use_direct_io or exc.errno not in _DIRECT_IO_FALLBACK_ERRORS:
                raise
            os.close(fd)
            fd = os.open(path, os.O_RDONLY)
            use_direct_io = False
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
                # Cache eviction only reduces host memory pressure. Some remote
                # and virtual filesystems do not implement this advice.
                pass
    finally:
        if fd is not None:
            os.close(fd)
        view.release()

    return PositionalReadResult(
        wall_s=time.perf_counter() - started,
        direct_io=use_direct_io,
    )
