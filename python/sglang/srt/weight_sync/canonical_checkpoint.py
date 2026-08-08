"""Host-local canonical safetensors checkpoints for weight staging."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from sglang.srt.model_loader.utils import DEFERRED_WEIGHT_COPY_SAFE_ATTR
from sglang.srt.weight_sync.file_io import read_file_into_tensor
from sglang.srt.weight_sync.host_local_buffer import HostLocalSharedBuffer
from sglang.srt.weight_sync.safetensors_buffer import SafetensorsBuffer

_ALIGNMENT = 4096


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
