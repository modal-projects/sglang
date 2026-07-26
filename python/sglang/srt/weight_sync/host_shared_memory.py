from __future__ import annotations

import errno
import mmap
import os
import socket
import uuid
from pathlib import Path
from typing import Any

import torch

_SHARED_MEMORY_ROOT = Path("/dev/shm")
_ALIGNMENT = 4096


class HostSharedMemoryBuffer:
    """Own one unlinked CPU byte mapping shared by local model workers."""

    def __init__(self, *, nbytes: int, cpu_group: Any, name: str):
        if nbytes <= 0:
            raise ValueError("host-shared buffer size must be positive")
        self.nbytes = (nbytes + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
        distributed = torch.distributed.is_initialized()
        world_size = (
            torch.distributed.get_world_size(group=cpu_group) if distributed else 1
        )
        rank = torch.distributed.get_rank(group=cpu_group) if distributed else 0
        if world_size > 1 and cpu_group is None:
            raise RuntimeError("host-shared memory requires a host-local process group")
        if world_size > 1:
            hosts: list[str | None] = [None] * world_size
            torch.distributed.all_gather_object(
                hosts,
                socket.gethostname(),
                group=cpu_group,
            )
            if len(set(hosts)) != 1:
                raise RuntimeError(
                    f"host-shared memory group spans multiple hosts: {hosts}"
                )

        path: Path | None = None
        creator_fd: int | None = None
        create_error: str | None = None
        if rank == 0:
            path = _SHARED_MEMORY_ROOT / f"sglang-{name}-{uuid.uuid4().hex}"
            try:
                filesystem = os.statvfs(_SHARED_MEMORY_ROOT)
                available = filesystem.f_bavail * filesystem.f_frsize
                if self.nbytes > available:
                    raise OSError(
                        errno.ENOSPC,
                        "insufficient /dev/shm capacity for host-shared "
                        f"weight memory: requested={self.nbytes} "
                        f"available={available}",
                    )
                creator_fd = os.open(
                    path,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                os.ftruncate(creator_fd, self.nbytes)
            except Exception as exc:
                create_error = f"{type(exc).__name__}: {exc}"
                if creator_fd is not None:
                    os.close(creator_fd)
                    creator_fd = None
                if path.exists():
                    path.unlink()

        creation = [None if path is None else str(path), create_error]
        if world_size > 1:
            global_rank_zero = torch.distributed.get_global_rank(cpu_group, 0)
            torch.distributed.broadcast_object_list(
                creation,
                src=global_rank_zero,
                group=cpu_group,
            )
        if creation[1] is not None:
            raise RuntimeError(f"failed to create host-shared memory: {creation[1]}")
        if creation[0] is None:
            raise RuntimeError("host-shared memory path was not distributed")
        self.path = Path(creation[0])

        local_error = None
        try:
            fd = creator_fd
            if fd is None:
                fd = os.open(self.path, os.O_RDWR)
            try:
                self.mapping = mmap.mmap(
                    fd,
                    self.nbytes,
                    flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                )
            finally:
                os.close(fd)
            self.tensor = torch.frombuffer(
                self.mapping,
                dtype=torch.uint8,
                count=self.nbytes,
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
            self.close()
            if rank == 0:
                self.path.unlink(missing_ok=True)
            raise RuntimeError("failed to map host-shared memory: " + "; ".join(errors))

        if world_size > 1:
            torch.distributed.barrier(group=cpu_group)
        if rank == 0:
            self.path.unlink()
        if world_size > 1:
            torch.distributed.barrier(group=cpu_group)

    def view(self, nbytes: int, *, offset: int = 0) -> torch.Tensor:
        if offset < 0 or nbytes < 0 or offset + nbytes > self.nbytes:
            raise ValueError(
                "host-shared memory view exceeds capacity: "
                f"offset={offset} requested={nbytes} capacity={self.nbytes}"
            )
        return self.tensor[offset : offset + nbytes]

    def close(self) -> None:
        tensor = getattr(self, "tensor", None)
        if tensor is not None:
            del self.tensor
        mapping = getattr(self, "mapping", None)
        if mapping is not None:
            mapping.close()
            del self.mapping
