"""Bounded positional file readers used during weight staging."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


def read_exact(reader: Any, nbytes: int) -> bytearray:
    """Read exactly ``nbytes`` from a stream into one bounded output buffer."""

    result = bytearray(nbytes)
    view = memoryview(result)
    position = 0
    while position < nbytes:
        nread = reader.readinto(view[position:])
        if not nread:
            raise EOFError(
                f"unexpected end of compressed payload: "
                f"expected={nbytes} actual={position}"
            )
        position += nread
    return result


class PositionalFileRangeReader:
    """Expose one immutable file range without advancing the shared descriptor."""

    def __init__(
        self,
        fd: int,
        offset: int,
        nbytes: int,
        path: str | Path,
        *,
        max_read_bytes: int,
    ):
        self.fd = fd
        self.offset = offset
        self.nbytes = nbytes
        self.path = path
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
            raise FileNotFoundError(
                f"incomplete source range {self.path}: offset={self.offset} "
                f"expected={self.nbytes} actual={self.position}"
            )
        self.position += len(data)
        return data
