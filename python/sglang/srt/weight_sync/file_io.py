"""Bounded positional I/O used during weight staging."""

from __future__ import annotations

import errno
import os
import time
from dataclasses import dataclass
from pathlib import Path

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
