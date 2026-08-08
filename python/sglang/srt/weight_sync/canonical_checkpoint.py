"""Host-local canonical safetensors checkpoints for weight staging."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from sglang.srt.model_loader.utils import DEFERRED_WEIGHT_COPY_SAFE_ATTR
from sglang.srt.weight_sync.file_io import (
    read_file_into_tensor,
    read_range_into_tensor,
)
from sglang.srt.weight_sync.host_local_buffer import HostLocalSharedBuffer
from sglang.srt.weight_sync.safetensors_buffer import (
    SafetensorsBuffer,
    SafetensorsEntry,
    SafetensorsLayout,
)

_ALIGNMENT = 4096
_POSITIONAL_IO_CHUNK_BYTES = 64 << 20

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CheckpointFile:
    name: str
    path: Path
    offset: int
    nbytes: int


def _align(value: int) -> int:
    return (value + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def _load_weight_map(root: Path) -> dict[str, str] | None:
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if not indexes:
        return None
    if len(indexes) != 1:
        raise ValueError(
            f"expected at most one safetensors index in {root}, found {indexes}"
        )
    try:
        payload = json.loads(indexes[0].read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid safetensors index: {indexes[0]}") from exc
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid safetensors weight map: {indexes[0]}")
    if not all(
        isinstance(name, str) and name and isinstance(filename, str) and filename
        for name, filename in weight_map.items()
    ):
        raise ValueError(f"invalid safetensors weight map: {indexes[0]}")
    return weight_map


def _checkpoint_manifest(
    checkpoint_dir: str | Path,
) -> tuple[dict[str, str] | None, list[_CheckpointFile], int, str]:
    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")

    indexed_weight_map = _load_weight_map(root)
    if indexed_weight_map is None:
        filenames = [path.name for path in sorted(root.glob("*.safetensors"))]
        if not filenames:
            raise FileNotFoundError(f"checkpoint has no safetensors files: {root}")
    else:
        filenames = sorted(set(indexed_weight_map.values()))

    files = []
    capacity = 0
    signature_files = []
    for filename in filenames:
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid checkpoint shard path: {filename!r}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint shard does not exist: {path}")
        file_nbytes = path.stat().st_size
        if file_nbytes <= 0:
            raise ValueError(f"checkpoint shard is empty: {path}")
        capacity = _align(capacity)
        files.append(
            _CheckpointFile(
                name=filename,
                path=path,
                offset=capacity,
                nbytes=file_nbytes,
            )
        )
        capacity += file_nbytes
        signature_files.append((filename, file_nbytes))
    signature_payload = json.dumps(
        {
            "files": signature_files,
            "weight_map": (
                sorted(indexed_weight_map.items())
                if indexed_weight_map is not None
                else None
            ),
        },
        separators=(",", ":"),
    ).encode()
    signature = hashlib.sha256(signature_payload).hexdigest()
    return indexed_weight_map, files, _align(capacity), signature


class CanonicalCheckpoint:
    """Expose one complete safetensors checkpoint to local model workers.

    Memory storage copies each checkpoint once into host-shared pageable RAM.
    Disk storage maps a materialized host-local checkpoint without allocating
    another canonical copy. Callers must synchronize before mutating memory
    storage; disk storage is immutable and must be advanced by the disk
    checkpoint materializer.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        host_group: torch.distributed.ProcessGroup | None,
        version: int = 0,
        storage: Literal["memory", "disk"] = "memory",
    ):
        if version < 0:
            raise ValueError("canonical checkpoint version must be non-negative")
        if storage not in {"memory", "disk"}:
            raise ValueError("canonical checkpoint storage must be memory or disk")
        distributed = torch.distributed.is_initialized()
        world_size = (
            torch.distributed.get_world_size(group=host_group) if distributed else 1
        )
        rank = torch.distributed.get_rank(group=host_group) if distributed else 0
        self._host_rank = rank
        if world_size > 1 and host_group is None:
            raise RuntimeError(
                "canonical checkpoint caching requires a host-local process group"
            )

        local_error = None
        local_exception = None
        manifest = None
        try:
            manifest = _checkpoint_manifest(checkpoint_dir)
        except Exception as exc:
            local_exception = exc
            local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        if world_size > 1:
            errors: list[str | None] = [None] * world_size
            torch.distributed.all_gather_object(
                errors,
                local_error,
                group=host_group,
            )
        else:
            errors = [local_error]
        errors = [error for error in errors if error is not None]
        if errors:
            if world_size == 1 and local_exception is not None:
                raise local_exception
            raise RuntimeError(
                "failed to discover canonical checkpoint: " + "; ".join(errors)
            )
        if manifest is None:
            raise RuntimeError("canonical checkpoint discovery returned no manifest")
        indexed_weight_map, files, capacity, signature = manifest

        local_manifest = (capacity, signature)
        if world_size > 1:
            manifests: list[tuple[int, str] | None] = [None] * world_size
            torch.distributed.all_gather_object(
                manifests,
                local_manifest,
                group=host_group,
            )
        else:
            manifests = [local_manifest]
        if any(manifest != local_manifest for manifest in manifests):
            raise RuntimeError(
                f"checkpoint manifests differ across local workers: {manifests}"
            )

        self.storage = storage
        self.root = Path(checkpoint_dir)
        self._storage = None
        started = time.perf_counter()
        local_error = None
        if storage == "memory":
            self._storage = HostLocalSharedBuffer(
                nbytes=capacity,
                host_group=host_group,
                name="weight-checkpoint",
            )
            try:
                for index, checkpoint_file in enumerate(files):
                    if index % world_size != rank:
                        continue
                    target = self._storage.view(
                        checkpoint_file.nbytes,
                        offset=checkpoint_file.offset,
                    )
                    try:
                        read_file_into_tensor(
                            checkpoint_file.path,
                            target,
                            drop_cache_after_read=True,
                        )
                    finally:
                        del target
            except Exception as exc:
                local_error = f"rank {rank}: {type(exc).__name__}: {exc}"

            if world_size > 1:
                errors: list[str | None] = [None] * world_size
                torch.distributed.all_gather_object(
                    errors,
                    local_error,
                    group=host_group,
                )
            else:
                errors = [local_error]
            errors = [error for error in errors if error is not None]
            if errors:
                self._storage.close()
                raise RuntimeError(
                    "failed to cache canonical checkpoint: " + "; ".join(errors)
                )

        self._files = {}
        discovered_weight_map = {}
        layout_error = None
        layout_exception = None
        try:
            for checkpoint_file in files:
                if self._storage is None:
                    source = torch.from_file(
                        str(checkpoint_file.path),
                        shared=False,
                        size=checkpoint_file.nbytes,
                        dtype=torch.uint8,
                    )
                else:
                    source = self._storage.view(
                        checkpoint_file.nbytes,
                        offset=checkpoint_file.offset,
                    )
                tensor_file = SafetensorsBuffer(source)
                self._files[checkpoint_file.name] = tensor_file
                for name in tensor_file.layout.tensors:
                    previous = discovered_weight_map.setdefault(
                        name, checkpoint_file.name
                    )
                    if previous != checkpoint_file.name:
                        raise ValueError(
                            f"checkpoint tensor {name!r} appears in both "
                            f"{previous!r} and {checkpoint_file.name!r}"
                        )
            if not discovered_weight_map:
                raise ValueError(f"checkpoint contains no tensors: {checkpoint_dir}")
            if (
                indexed_weight_map is not None
                and indexed_weight_map != discovered_weight_map
            ):
                missing = sorted(set(indexed_weight_map) - set(discovered_weight_map))
                extra = sorted(set(discovered_weight_map) - set(indexed_weight_map))
                misplaced = sorted(
                    name
                    for name in set(indexed_weight_map) & set(discovered_weight_map)
                    if indexed_weight_map[name] != discovered_weight_map[name]
                )
                raise ValueError(
                    "safetensors index does not match checkpoint shards: "
                    f"missing={missing[:8]} extra={extra[:8]} "
                    f"misplaced={misplaced[:8]}"
                )
        except Exception as exc:
            layout_exception = exc
            layout_error = f"rank {rank}: {type(exc).__name__}: {exc}"

        if world_size > 1:
            layout_errors: list[str | None] = [None] * world_size
            torch.distributed.all_gather_object(
                layout_errors,
                layout_error,
                group=host_group,
            )
        else:
            layout_errors = [layout_error]
        errors = [error for error in layout_errors if error is not None]
        if errors:
            self._files.clear()
            if self._storage is not None:
                self._storage.close()
            if world_size == 1 and layout_exception is not None:
                raise layout_exception
            raise RuntimeError(
                "failed to parse canonical checkpoint: " + "; ".join(errors)
            )

        self.weight_map = indexed_weight_map or discovered_weight_map
        self._checkpoint_files = files
        self.checkpoint_bytes = sum(file.nbytes for file in files)
        self.read_wall_s = time.perf_counter() - started
        self.version = version
        self.valid = True
        self.updating = False
        self.invalid_reason: str | None = None

    def _require_readable(self) -> None:
        if not self.valid:
            reason = f": {self.invalid_reason}" if self.invalid_reason else ""
            raise RuntimeError(f"canonical checkpoint is invalid{reason}")
        if self.updating:
            raise RuntimeError("canonical checkpoint update is in progress")

    def begin_update(self, target_version: int) -> None:
        self._require_readable()
        if self.storage != "memory":
            raise RuntimeError(
                "disk-backed canonical checkpoints must be advanced by the "
                "disk checkpoint materializer"
            )
        if target_version <= self.version:
            raise ValueError(
                f"target version {target_version} must follow canonical "
                f"version {self.version}"
            )
        self.valid = False
        self.updating = True
        self.invalid_reason = (
            f"update from version {self.version} to {target_version} is incomplete"
        )

    def get_update_tensor_bytes(self, name: str) -> torch.Tensor:
        if not self.updating:
            raise RuntimeError("canonical checkpoint update is not in progress")
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(f"canonical checkpoint has no tensor {name!r}")
        return self._files[filename].get_tensor_bytes(name)

    def finish_update(self, target_version: int) -> None:
        if not self.updating:
            raise RuntimeError("canonical checkpoint update is not in progress")
        if target_version <= self.version:
            raise ValueError(
                f"target version {target_version} must follow canonical "
                f"version {self.version}"
            )
        self.version = target_version
        self.invalid_reason = None
        self.updating = False
        self.valid = True

    def fail_update(self, reason: str) -> None:
        if not self.updating:
            raise RuntimeError("canonical checkpoint update is not in progress")
        self.invalid_reason = reason
        self.updating = False
        self.valid = False

    def get_tensor(self, name: str) -> torch.Tensor:
        self._require_readable()
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(f"canonical checkpoint has no tensor {name!r}")
        tensor = self._files[filename].get_tensor(name)
        setattr(tensor, DEFERRED_WEIGHT_COPY_SAFE_ATTR, True)
        return tensor

    def get_tensor_bytes(self, name: str) -> torch.Tensor:
        self._require_readable()
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(f"canonical checkpoint has no tensor {name!r}")
        return self._files[filename].get_tensor_bytes(name)

    @contextmanager
    def tensor_group(self, _path: str, _names: list[str]):
        """Expose a stable group of tensors to the bounded CPU compiler."""

        self._require_readable()
        yield self

    def run_on_host_ranks(self, _operation: str, function):
        """Run a compiler phase; memory and mapped sources need no inner sync."""

        return function()

    def stats(self) -> dict[str, int | float | str]:
        self._require_readable()
        return {
            "storage": (
                "host_shared_memory" if self._storage is not None else "host_local_disk"
            ),
            "checkpoint_bytes": self.checkpoint_bytes,
            "allocated_bytes": self._storage.nbytes if self._storage is not None else 0,
            "files": len(self._files),
            "tensors": len(self.weight_map),
            "physical_host_copies": 1 if self._storage is not None else 0,
            "read_wall_s": round(self.read_wall_s, 6),
            "version": self.version,
        }

    def release_cached_pages(self) -> dict[str, int | float]:
        """Release clean file-backed pages after compiling a disk checkpoint."""

        started = time.perf_counter()
        files = 0
        if (
            self.storage == "disk"
            and self._host_rank == 0
            and hasattr(os, "posix_fadvise")
        ):
            for checkpoint_file in self._checkpoint_files:
                try:
                    fd = os.open(checkpoint_file.path, os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                    files += 1
                except OSError:
                    # Cache eviction is a memory-pressure hint, not a
                    # checkpoint correctness requirement.
                    pass
        return {
            "files": files,
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def close(self) -> None:
        self._files.clear()
        self._checkpoint_files = []
        if self._storage is not None:
            self._storage.close()
            self._storage = None


@dataclass(frozen=True)
class _DiskTensorSource:
    filename: str
    file_offset: int
    buffer_offset: int
    entry: SafetensorsEntry


@dataclass(frozen=True)
class _DiskGroupRead:
    filename: str
    file_offset: int
    buffer_offset: int
    nbytes: int
    direct_io: bool


@dataclass(frozen=True)
class _DiskTensorGroup:
    path: str
    tensors: dict[str, _DiskTensorSource]
    reads: tuple[_DiskGroupRead, ...]
    source_bytes: int
    buffer_bytes: int


class _DiskTensorGroupView:
    """Expose one canonical tensor group from a reusable shared buffer."""

    def __init__(
        self,
        buffer: HostLocalSharedBuffer,
        tensors: dict[str, _DiskTensorSource],
    ):
        self._buffer = buffer
        self._tensors = tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        source = self._tensors.get(name)
        if source is None:
            raise KeyError(f"canonical tensor group has no tensor {name!r}")
        nbytes = source.entry.relative_end - source.entry.relative_begin
        try:
            tensor = (
                self._buffer.view(nbytes, offset=source.buffer_offset)
                .view(source.entry.dtype)
                .reshape(source.entry.shape)
            )
        except RuntimeError as exc:
            raise ValueError(
                f"cannot construct canonical tensor view for {name!r}"
            ) from exc
        setattr(tensor, DEFERRED_WEIGHT_COPY_SAFE_ATTR, True)
        return tensor

    def get_tensor_bytes(self, name: str) -> torch.Tensor:
        source = self._tensors.get(name)
        if source is None:
            raise KeyError(f"canonical tensor group has no tensor {name!r}")
        nbytes = source.entry.relative_end - source.entry.relative_begin
        return self._buffer.view(nbytes, offset=source.buffer_offset)


def _disk_tensor_groups(
    *,
    weight_map: dict[str, str],
    layouts: dict[str, SafetensorsLayout],
    names_by_group: dict[str, list[str]],
    read_parallelism: int,
) -> list[_DiskTensorGroup]:
    """Plan bounded, physically ordered reads for runtime compile groups."""

    if read_parallelism <= 0:
        raise ValueError("checkpoint read parallelism must be positive")
    for filename in sorted(set(weight_map.values())):
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid checkpoint shard path: {filename!r}")
        if filename not in layouts:
            raise ValueError(f"checkpoint has no parsed shard {filename!r}")

    disk_tensors = {}
    for name, filename in weight_map.items():
        try:
            entry = layouts[filename].tensors[name]
        except KeyError as exc:
            raise ValueError(
                f"checkpoint index maps {name!r} to {filename!r}, but the "
                "tensor is absent from that shard"
            ) from exc
        disk_tensors[name] = (
            filename,
            layouts[filename].data_offset + entry.relative_begin,
            layouts[filename].data_offset + entry.relative_end,
            entry,
        )

    groups = []
    for group_path, names in names_by_group.items():
        sources = []
        for name in names:
            try:
                sources.append((name, disk_tensors[name]))
            except KeyError as exc:
                raise ValueError(
                    f"runtime group {group_path!r} references unknown checkpoint "
                    f"tensor {name!r}"
                ) from exc
        sources.sort(key=lambda item: (*item[1][:3], item[0]))

        reads = []
        tensors = {}
        source_bytes = 0
        buffer_bytes = 0
        run = []

        def finish_run() -> None:
            nonlocal buffer_bytes
            if not run:
                return
            _, (filename, first_begin, _, _) = run[0]
            last_end = run[-1][1][2]
            run_nbytes = last_end - first_begin
            aligned_buffer_begin = _align(buffer_bytes)
            data_buffer_begin = aligned_buffer_begin + first_begin % _ALIGNMENT
            direct_begin = _align(first_begin)
            direct_end = last_end // _ALIGNMENT * _ALIGNMENT

            if direct_begin < direct_end:
                if first_begin < direct_begin:
                    reads.append(
                        _DiskGroupRead(
                            filename=filename,
                            file_offset=first_begin,
                            buffer_offset=data_buffer_begin,
                            nbytes=direct_begin - first_begin,
                            direct_io=False,
                        )
                    )
                direct_nbytes = direct_end - direct_begin
                chunks = min(
                    read_parallelism,
                    math.ceil(direct_nbytes / _POSITIONAL_IO_CHUNK_BYTES),
                )
                chunk_blocks = math.ceil(direct_nbytes / _ALIGNMENT / chunks)
                chunk_bytes = chunk_blocks * _ALIGNMENT
                for offset in range(0, direct_nbytes, chunk_bytes):
                    reads.append(
                        _DiskGroupRead(
                            filename=filename,
                            file_offset=direct_begin + offset,
                            buffer_offset=(
                                data_buffer_begin + direct_begin - first_begin + offset
                            ),
                            nbytes=min(chunk_bytes, direct_nbytes - offset),
                            direct_io=True,
                        )
                    )
                tail_begin = direct_end
            else:
                tail_begin = first_begin
            if tail_begin < last_end:
                reads.append(
                    _DiskGroupRead(
                        filename=filename,
                        file_offset=tail_begin,
                        buffer_offset=data_buffer_begin + tail_begin - first_begin,
                        nbytes=last_end - tail_begin,
                        direct_io=False,
                    )
                )

            for name, (source_filename, begin, end, entry) in run:
                tensors[name] = _DiskTensorSource(
                    filename=source_filename,
                    file_offset=begin,
                    buffer_offset=data_buffer_begin + begin - first_begin,
                    entry=entry,
                )
            buffer_bytes = data_buffer_begin + run_nbytes
            run.clear()

        for name, source in sources:
            _, begin, end, _ = source
            source_bytes += end - begin
            if run:
                previous = run[-1][1]
                if source[0] != previous[0] or begin != previous[2]:
                    finish_run()
            run.append((name, source))
        finish_run()
        groups.append(
            _DiskTensorGroup(
                path=group_path,
                tensors=tensors,
                reads=tuple(reads),
                source_bytes=source_bytes,
                buffer_bytes=buffer_bytes,
            )
        )
    return groups


class DiskCanonicalCheckpointUpdate:
    """Compile and persist an NVMe canonical update from the same bytes.

    Two bounded host-shared buffers overlap the next canonical read with the
    current runtime compilation. Delta transforms, checksum verification, and
    sparse persistence all operate on those buffers; canonical bytes are never
    reread solely for compilation.
    """

    def __init__(
        self,
        checkpoint: CanonicalCheckpoint,
        *,
        names_by_group: dict[str, list[str]],
        transform: Any,
        target_version: int,
        host_group: torch.distributed.ProcessGroup | None,
    ):
        if checkpoint.storage != "disk":
            raise ValueError("disk canonical updates require disk storage")
        if target_version <= checkpoint.version:
            raise ValueError("disk canonical updates must advance the version")
        distributed = torch.distributed.is_initialized()
        self.world_size = (
            torch.distributed.get_world_size(group=host_group) if distributed else 1
        )
        self.rank = torch.distributed.get_rank(group=host_group) if distributed else 0
        if self.world_size > 1 and host_group is None:
            raise RuntimeError("disk canonical updates require a host-local group")

        self.root = checkpoint.root
        self.weight_map = dict(checkpoint.weight_map)
        self.initial_version = checkpoint.version
        self.target_version = target_version
        self.host_group = host_group
        self.transform = transform
        self._groups = _disk_tensor_groups(
            weight_map=self.weight_map,
            layouts={
                filename: tensor_file.layout
                for filename, tensor_file in checkpoint._files.items()
            },
            names_by_group=names_by_group,
            read_parallelism=self.world_size,
        )
        if not self._groups:
            raise ValueError("canonical checkpoint has no runtime tensor groups")
        if len({group.path for group in self._groups}) != len(self._groups):
            raise ValueError("canonical runtime tensor group paths are not unique")
        self._validate_plan()

        buffer_count = min(2, len(self._groups))
        capacities = [
            max(
                group.buffer_bytes
                for index, group in enumerate(self._groups)
                if index % buffer_count == buffer_index
            )
            for buffer_index in range(buffer_count)
        ]
        self._buffers = []
        try:
            for index, capacity in enumerate(capacities):
                self._buffers.append(
                    HostLocalSharedBuffer(
                        nbytes=max(1, capacity),
                        host_group=host_group,
                        name=f"canonical-update-{index}",
                    )
                )
        except Exception:
            self.close()
            raise
        self._reader = None
        reader_exception = None
        reader_error = None
        try:
            self._reader = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="canonical-checkpoint-read",
            )
        except Exception as exc:
            reader_exception = exc
            reader_error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        reader_errors = [
            error for error in self._gather(reader_error) if error is not None
        ]
        if reader_errors:
            self.close()
            raise RuntimeError(
                "failed to create canonical checkpoint readers: "
                + "; ".join(reader_errors)
            ) from reader_exception
        changed_names = set(transform.operations_by_name)
        sources_by_name = {
            name: source
            for group in self._groups
            for name, source in group.tensors.items()
        }
        missing = changed_names - sources_by_name.keys()
        if missing:
            self.close()
            raise RuntimeError(
                "runtime groups do not cover every delta tensor: "
                f"missing={sorted(missing)[:20]}"
            )
        self._sources_by_name = sources_by_name
        self._transform_owners = self._assign_transform_owners(changed_names)
        changed_files = sorted(
            {sources_by_name[name].filename for name in changed_names}
        )
        self._file_owners = {
            filename: index % self.world_size
            for index, filename in enumerate(changed_files)
        }
        self._expected_names = {
            name
            for name in changed_names
            if self._file_owners[sources_by_name[name].filename] == self.rank
        }
        self._fds = {}
        open_exception = None
        open_error = None
        try:
            for filename, owner in self._file_owners.items():
                if owner == self.rank:
                    self._fds[filename] = os.open(self.root / filename, os.O_RDWR)
        except Exception as exc:
            open_exception = exc
            open_error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        open_errors = [error for error in self._gather(open_error) if error is not None]
        if open_errors:
            self.close()
            raise RuntimeError(
                "failed to open canonical checkpoint writers: " + "; ".join(open_errors)
            ) from open_exception
        available_cpus = os.cpu_count() or 1
        self._writer = None
        writer_exception = None
        writer_error = None
        try:
            self._writer = ThreadPoolExecutor(
                max_workers=max(
                    1,
                    min(len(self._fds), max(1, available_cpus // self.world_size)),
                ),
                thread_name_prefix="canonical-checkpoint-write",
            )
        except Exception as exc:
            writer_exception = exc
            writer_error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        writer_errors = [
            error for error in self._gather(writer_error) if error is not None
        ]
        if writer_errors:
            self.close()
            raise RuntimeError(
                "failed to create canonical checkpoint writers: "
                + "; ".join(writer_errors)
            ) from writer_exception
        filesystem = os.statvfs(self.root)
        self._write_block_bytes = filesystem.f_frsize or filesystem.f_bsize
        self._pending_reads: dict[int, Future] = {}
        self._next_group = 0
        self._transaction = None
        self._entered = False
        self._finished = False
        self._started = 0.0
        self._read_stats = []
        self._transform_stats = []
        self._written_names = set()
        self._dirty_files = set()
        self._logical_write_bytes = 0
        self._physical_write_bytes = 0
        self._write_worker_s = 0.0
        self._write_wait_s = 0.0
        self._fsync_worker_s = 0.0
        self._persistence_stats = None
        logger.info(
            "Prepared NVMe canonical update v%d to v%d: groups=%d "
            "shared_buffer_bytes=%d",
            self.initial_version,
            self.target_version,
            len(self._groups),
            sum(buffer.nbytes for buffer in self._buffers),
        )

    def _validate_plan(self) -> None:
        digest = hashlib.sha256()
        for group in self._groups:
            digest.update(f"{group.path}:{group.buffer_bytes}\n".encode())
            for name, source in sorted(group.tensors.items()):
                digest.update(
                    (
                        f"{name}:{source.filename}:{source.file_offset}:"
                        f"{source.buffer_offset}:{source.entry.dtype_code}:"
                        f"{source.entry.shape}\n"
                    ).encode()
                )
        signatures = self._gather(digest.hexdigest())
        if len(set(signatures)) != 1:
            raise RuntimeError(
                f"canonical update plans differ across local workers: {signatures}"
            )

    def _gather(self, value: Any) -> list[Any]:
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        torch.distributed.all_gather_object(values, value, group=self.host_group)
        return values

    def run_on_host_ranks(self, operation: str, function):
        result = None
        exception = None
        error = None
        try:
            result = function()
        except Exception as exc:
            exception = exc
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [value for value in self._gather(error) if value is not None]
        if errors:
            raise RuntimeError(
                f"{operation} failed: " + "; ".join(errors)
            ) from exception
        return result

    def _assign_transform_owners(self, changed_names: set[str]) -> dict[str, int]:
        owners = {}
        for group in self._groups:
            rank_bytes = [0] * self.world_size
            weighted_names = [
                (
                    name,
                    (source.entry.relative_end - source.entry.relative_begin)
                    * len(self.transform.operations_by_name[name]),
                )
                for name, source in group.tensors.items()
                if name in changed_names
            ]
            for name, nbytes in sorted(
                weighted_names, key=lambda item: (-item[1], item[0])
            ):
                owner = min(
                    range(self.world_size),
                    key=lambda candidate: (rank_bytes[candidate], candidate),
                )
                owners[name] = owner
                rank_bytes[owner] += nbytes
        return owners

    def _read_group(self, index: int) -> dict[str, Any]:
        started = time.perf_counter()
        group = self._groups[index]
        buffer = self._buffers[index % len(self._buffers)]
        reads = []
        for read_index, source in enumerate(group.reads):
            if read_index % self.world_size != self.rank:
                continue
            target = buffer.view(source.nbytes, offset=source.buffer_offset)
            try:
                result = read_range_into_tensor(
                    self.root / source.filename,
                    target,
                    file_offset=source.file_offset,
                    direct_io=source.direct_io,
                    drop_cache_after_read=True,
                )
            finally:
                del target
            reads.append(
                {
                    "bytes": source.nbytes,
                    "direct_io": result.direct_io,
                    "wall_s": result.wall_s,
                }
            )
        return {
            "owner_rank": self.rank,
            "owned_bytes": sum(read["bytes"] for read in reads),
            "owned_direct_io_bytes": sum(
                read["bytes"] for read in reads if read["direct_io"]
            ),
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def _submit_read(self, index: int) -> None:
        self._pending_reads[index] = self._reader.submit(self._read_group, index)

    def _finish_read(self, index: int) -> None:
        wait_started = time.perf_counter()
        local = self.run_on_host_ranks(
            f"canonical read for group {self._groups[index].path!r}",
            self._pending_reads.pop(index).result,
        )
        ranks = self._gather(local)
        self._read_stats.append(
            {
                "group": self._groups[index].path,
                "bytes": sum(result["owned_bytes"] for result in ranks),
                "direct_io_bytes": sum(
                    result["owned_direct_io_bytes"] for result in ranks
                ),
                "worker_s": round(sum(result["wall_s"] for result in ranks), 6),
                "wait_s": round(time.perf_counter() - wait_started, 6),
            }
        )

    def _submit_writes(
        self,
        view: _DiskTensorGroupView,
        writes: list[tuple[str, list[tuple[int, int]]]],
    ) -> list[Future]:
        writes_by_file = {}
        for name, ranges in writes:
            source = self._sources_by_name[name]
            if self._file_owners[source.filename] == self.rank:
                writes_by_file.setdefault(source.filename, []).append(
                    (name, source.file_offset, ranges)
                )

        def write_file(item):
            filename, file_writes = item
            fd = self._fds[filename]
            processed_names = set()
            logical_bytes = 0
            physical_bytes = 0
            started = time.perf_counter()
            for name, file_offset, ranges in sorted(
                file_writes, key=lambda value: value[1]
            ):
                tensor = view.get_tensor_bytes(name)
                payload = memoryview(tensor.numpy()).cast("B")
                try:
                    processed_names.add(name)
                    logical_bytes += len(payload)
                    for begin, end in ranges:
                        position = begin
                        while position < end:
                            chunk_end = min(position + _POSITIONAL_IO_CHUNK_BYTES, end)
                            written = os.pwrite(
                                fd,
                                payload[position:chunk_end],
                                file_offset + position,
                            )
                            if written <= 0:
                                raise RuntimeError(
                                    "short canonical checkpoint write for "
                                    f"{name!r}: range=({begin}, {end})"
                                )
                            position += written
                            physical_bytes += written
                finally:
                    payload.release()
                    del tensor
            return {
                "filename": filename,
                "processed_names": processed_names,
                "logical_bytes": logical_bytes,
                "physical_bytes": physical_bytes,
                "wall_s": time.perf_counter() - started,
            }

        return [
            self._writer.submit(write_file, item)
            for item in sorted(writes_by_file.items())
        ]

    def _finish_writes(self, futures: list[Future]) -> None:
        wait_started = time.perf_counter()
        for future in futures:
            result = future.result()
            self._written_names.update(result["processed_names"])
            self._logical_write_bytes += result["logical_bytes"]
            self._physical_write_bytes += result["physical_bytes"]
            self._write_worker_s += result["wall_s"]
            if result["physical_bytes"]:
                self._dirty_files.add(result["filename"])
        self._write_wait_s += time.perf_counter() - wait_started

    def __enter__(self):
        if self._entered:
            raise RuntimeError("canonical checkpoint update is already active")
        self._started = time.perf_counter()

        def begin_transaction():
            if self.rank != 0:
                return
            from sglang.srt.weight_sync.disk_checkpoint import (
                LocalCheckpointTransaction,
            )

            self._transaction = LocalCheckpointTransaction(str(self.root))
            self._transaction.__enter__()
            self._transaction.begin(self.initial_version)

        try:
            self.run_on_host_ranks(
                "canonical checkpoint transaction", begin_transaction
            )
        except Exception:
            if self._transaction is not None:
                self._transaction.__exit__(None, None, None)
                self._transaction = None
            raise
        self._entered = True
        self._submit_read(0)
        return self

    @contextmanager
    def tensor_group(self, path: str, names: list[str]):
        if not self._entered or self._finished:
            raise RuntimeError("canonical checkpoint update is not active")
        index = self._next_group
        if index >= len(self._groups):
            raise RuntimeError("canonical compiler requested too many tensor groups")
        group = self._groups[index]
        if group.path != path or set(group.tensors) != set(names):
            raise RuntimeError(
                "canonical compiler group order changed: "
                f"expected={group.path!r} actual={path!r}"
            )
        self._finish_read(index)
        next_index = index + 1
        if next_index < len(self._groups):
            self._submit_read(next_index)
        view = _DiskTensorGroupView(
            self._buffers[index % len(self._buffers)], group.tensors
        )

        local_writes = []
        write_lock = threading.Lock()

        def record_write(name, _tensor, ranges):
            with write_lock:
                local_writes.append((name, ranges))

        owned_names = [
            name for name in names if self._transform_owners.get(name) == self.rank
        ]
        local_transform = self.run_on_host_ranks(
            f"canonical delta transform for group {path!r}",
            lambda: self.transform.transform_tensors(
                {name: view.get_tensor_bytes(name) for name in owned_names},
                description=path,
                write_tensor=record_write,
                write_block_bytes=self._write_block_bytes,
            ),
        )
        rank_transforms = self._gather(local_transform)
        self._transform_stats.append(
            {
                "group": path,
                "delta_tensors": sum(
                    stats["delta_tensors"] for stats in rank_transforms
                ),
                "delta_fragments": sum(
                    stats["delta_fragments"] for stats in rank_transforms
                ),
                "target_tensor_bytes": sum(
                    stats["target_tensor_bytes"] for stats in rank_transforms
                ),
                "compressed_bytes": sum(
                    stats["compressed_bytes"] for stats in rank_transforms
                ),
                "wall_s": max(stats["wall_s"] for stats in rank_transforms),
            }
        )
        writes = [
            write for rank_writes in self._gather(local_writes) for write in rank_writes
        ]
        futures = self.run_on_host_ranks(
            f"canonical write submission for group {path!r}",
            lambda: self._submit_writes(view, writes),
        )

        body_exception = None
        try:
            yield view
        except Exception as exc:
            body_exception = exc

        write_exception = None
        try:
            self._finish_writes(futures)
        except Exception as exc:
            write_exception = exc
        local_error = body_exception or write_exception
        error = (
            None
            if local_error is None
            else f"rank {self.rank}: {type(local_error).__name__}: {local_error}"
        )
        errors = [value for value in self._gather(error) if value is not None]
        if errors:
            raise RuntimeError(
                f"canonical group {path!r} failed: " + "; ".join(errors)
            ) from local_error
        self._next_group += 1

    def finish(self) -> dict[str, Any]:
        if self._next_group != len(self._groups):
            raise RuntimeError(
                "canonical compiler did not consume every tensor group: "
                f"completed={self._next_group} expected={len(self._groups)}"
            )

        def persist():
            if self._written_names != self._expected_names:
                raise RuntimeError(
                    "canonical update did not persist every delta tensor: "
                    f"missing={sorted(self._expected_names - self._written_names)[:20]} "
                    f"extra={sorted(self._written_names - self._expected_names)[:20]}"
                )
            for filename in self._dirty_files:
                fd = self._fds[filename]
                started = time.perf_counter()
                os.fsync(fd)
                self._fsync_worker_s += time.perf_counter() - started
                if hasattr(os, "posix_fadvise"):
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    except OSError:
                        pass

        self.run_on_host_ranks("canonical checkpoint persistence", persist)
        rank_stats = self._gather(
            {
                "logical_bytes": self._logical_write_bytes,
                "physical_bytes": self._physical_write_bytes,
                "worker_s": self._write_worker_s,
                "wait_s": self._write_wait_s,
                "fsync_worker_s": self._fsync_worker_s,
                "files": len(self._dirty_files),
            }
        )
        self._persistence_stats = {
            "logical_bytes": sum(stats["logical_bytes"] for stats in rank_stats),
            "physical_bytes": sum(stats["physical_bytes"] for stats in rank_stats),
            "worker_s": round(sum(stats["worker_s"] for stats in rank_stats), 6),
            "wait_s": round(max(stats["wait_s"] for stats in rank_stats), 6),
            "fsync_worker_s": round(
                sum(stats["fsync_worker_s"] for stats in rank_stats), 6
            ),
            "fsync_wall_s": round(
                max(stats["fsync_worker_s"] for stats in rank_stats), 6
            ),
            "files": sum(stats["files"] for stats in rank_stats),
        }

        def commit():
            if self.rank == 0:
                self._transaction.commit(self.target_version)

        self.run_on_host_ranks("canonical checkpoint publication", commit)
        self._finished = True
        stats = self.stats()
        logger.info(
            "Persisted NVMe canonical update v%d: read_bytes=%d write_bytes=%d "
            "wall_time=%.3fs",
            self.target_version,
            stats["canonical_read_bytes"],
            stats["persistence"]["physical_bytes"],
            stats["wall_s"],
        )
        return stats

    def stats(self) -> dict[str, Any]:
        delta_transform = {
            "operation": "canonical_delta_transform",
            "delta_tensors": sum(
                stats["delta_tensors"] for stats in self._transform_stats
            ),
            "delta_fragments": sum(
                stats["delta_fragments"] for stats in self._transform_stats
            ),
            "target_tensor_bytes": sum(
                stats["target_tensor_bytes"] for stats in self._transform_stats
            ),
            "compressed_bytes": sum(
                stats["compressed_bytes"] for stats in self._transform_stats
            ),
            "wall_s": round(sum(stats["wall_s"] for stats in self._transform_stats), 6),
        }
        return {
            "operation": "stream_canonical_checkpoint_update",
            "initial_version": self.initial_version,
            "target_version": self.target_version,
            "groups": len(self._groups),
            "canonical_read_bytes": sum(stats["bytes"] for stats in self._read_stats),
            "canonical_direct_io_bytes": sum(
                stats["direct_io_bytes"] for stats in self._read_stats
            ),
            "canonical_read_wait_s": round(
                sum(stats["wait_s"] for stats in self._read_stats), 6
            ),
            "shared_buffer_bytes": sum(buffer.nbytes for buffer in self._buffers),
            "delta_transform": delta_transform,
            "persistence": self._persistence_stats,
            "wall_s": round(time.perf_counter() - self._started, 6),
        }

    def close(self) -> None:
        reader = getattr(self, "_reader", None)
        if reader is not None:
            reader.shutdown(wait=True)
            self._reader = None
        writer = getattr(self, "_writer", None)
        if writer is not None:
            writer.shutdown(wait=True)
            self._writer = None
        for fd in getattr(self, "_fds", {}).values():
            os.close(fd)
        self._fds = {}
        for buffer in getattr(self, "_buffers", []):
            buffer.close()
        self._buffers = []

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.close()
        finally:
            if self._transaction is not None:
                self._transaction.__exit__(exc_type, exc, traceback)
                self._transaction = None
