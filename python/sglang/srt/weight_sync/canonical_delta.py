"""Verified delta transforms over canonical host checkpoints."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zstandard

from sglang.srt.environ import envs
from sglang.srt.weight_sync.canonical_checkpoint import CanonicalCheckpoint
from sglang.srt.weight_sync.checksum import create_checksum
from sglang.srt.weight_sync.delta_checkpoint import (
    read_delta_checkpoint,
    version_dir,
)
from sglang.srt.weight_sync.file_io import (
    FileDescriptorCache,
    PositionalFileRangeReader,
    file_descriptor_cache_limit,
    read_exact,
)

logger = logging.getLogger(__name__)

_STREAM_CHUNK_BYTES = 4 << 20
_HOST_WORKING_MEMORY_BYTES = 8 << 30


@dataclass(frozen=True)
class _DeltaOperation:
    version: int
    encoding: str
    checksum_algorithm: str
    expected_checksum: str
    source_path: Path
    source_offset: int
    compressed_nbytes: int


class _ByteBudget:
    """Bound concurrent payload buffers without rejecting one large tensor."""

    def __init__(self, limit: int):
        if limit <= 0:
            raise ValueError("delta working-memory budget must be positive")
        self.limit = limit
        self.used = 0
        self.condition = threading.Condition()

    @contextmanager
    def reserve(self, requested: int):
        charge = min(requested, self.limit)
        with self.condition:
            self.condition.wait_for(lambda: self.used + charge <= self.limit)
            self.used += charge
        try:
            yield
        finally:
            with self.condition:
                self.used -= charge
                self.condition.notify_all()


def _gather_objects(value: Any, group: Any, world_size: int) -> list[Any]:
    if world_size == 1:
        return [value]
    values = [None] * world_size
    torch.distributed.all_gather_object(values, value, group=group)
    return values


class CanonicalDeltaTransform:
    """Advance a canonical checkpoint through a verified delta lineage.

    Planning validates every published artifact before any canonical byte is
    changed. Once mutation starts, any failure invalidates the checkpoint; a
    caller must seed a new canonical image rather than consume partial state.
    """

    def __init__(
        self,
        checkpoint: CanonicalCheckpoint,
        *,
        checkpoint_source_dir: str | Path,
        target_version: int,
        host_group: torch.distributed.ProcessGroup | None,
        max_working_memory_bytes: int = _HOST_WORKING_MEMORY_BYTES,
    ):
        checkpoint.stats()
        if target_version < checkpoint.version:
            raise ValueError(
                f"target version {target_version} precedes canonical "
                f"version {checkpoint.version}"
            )
        if max_working_memory_bytes <= 0:
            raise ValueError("delta working-memory budget must be positive")
        distributed = torch.distributed.is_initialized()
        self.world_size = (
            torch.distributed.get_world_size(group=host_group) if distributed else 1
        )
        self.rank = torch.distributed.get_rank(group=host_group) if distributed else 0
        if self.world_size > 1 and host_group is None:
            raise RuntimeError("delta transforms require a host-local process group")

        self.checkpoint = checkpoint
        self.host_group = host_group
        self.target_version = target_version
        self.working_memory_budget_bytes = max(
            1, max_working_memory_bytes // self.world_size
        )
        self.operations_by_file: dict[str, dict[str, list[_DeltaOperation]]] = {}
        started = time.perf_counter()
        plan_error = None
        plan_exception = None
        plan_hasher = hashlib.sha256()
        plan_hasher.update(f"{checkpoint.version}:{target_version}\n".encode())
        fragment_count = 0
        source_paths = set()
        compressed_bytes = 0
        delta_blob_bytes = 0
        try:
            expected_base_version = checkpoint.version
            for version in range(checkpoint.version + 1, target_version + 1):
                root = version_dir(checkpoint_source_dir, version)
                delta = read_delta_checkpoint(
                    root,
                    expected_version=version,
                    expected_base_version=expected_base_version,
                )
                for tensor in delta.tensors:
                    canonical_filename = checkpoint.weight_map.get(tensor.name)
                    if canonical_filename is None:
                        raise KeyError(
                            f"delta tensor {tensor.name!r} is absent from the "
                            "canonical checkpoint"
                        )
                    operation = _DeltaOperation(
                        version=version,
                        encoding=delta.encoding,
                        checksum_algorithm=tensor.checksum_algorithm,
                        expected_checksum=tensor.expected_checksum,
                        source_path=tensor.source_path,
                        source_offset=tensor.source_offset,
                        compressed_nbytes=tensor.compressed_nbytes,
                    )
                    self.operations_by_file.setdefault(
                        canonical_filename, {}
                    ).setdefault(tensor.name, []).append(operation)
                    plan_hasher.update(
                        json.dumps(
                            (
                                tensor.name,
                                canonical_filename,
                                version,
                                operation.encoding,
                                operation.checksum_algorithm,
                                operation.expected_checksum,
                                str(tensor.source_path.relative_to(root.resolve())),
                                operation.source_offset,
                                operation.compressed_nbytes,
                            ),
                            separators=(",", ":"),
                        ).encode()
                    )
                    plan_hasher.update(b"\n")
                    fragment_count += 1
                    compressed_bytes += operation.compressed_nbytes
                    if tensor.source_path not in source_paths:
                        source_paths.add(tensor.source_path)
                        delta_blob_bytes += tensor.source_path.stat().st_size
                expected_base_version = version
        except Exception as exc:
            plan_exception = exc
            plan_error = f"rank {self.rank}: {type(exc).__name__}: {exc}"

        errors = [
            error
            for error in _gather_objects(plan_error, host_group, self.world_size)
            if error is not None
        ]
        if errors:
            if self.world_size == 1 and plan_exception is not None:
                raise plan_exception
            raise RuntimeError("failed to plan delta lineage: " + "; ".join(errors))

        signature = plan_hasher.hexdigest()
        signatures = _gather_objects(signature, host_group, self.world_size)
        if any(value != signature for value in signatures):
            raise RuntimeError(f"delta plans differ across local workers: {signatures}")

        self.operations_by_name = {
            name: operations
            for names in self.operations_by_file.values()
            for name, operations in names.items()
        }

        self.setup_stats = {
            "operation": "plan_delta_checkpoint",
            "canonical_version": checkpoint.version,
            "target_version": target_version,
            "delta_versions": list(range(checkpoint.version + 1, target_version + 1)),
            "delta_shards": len(source_paths),
            "delta_tensors": sum(
                len(names) for names in self.operations_by_file.values()
            ),
            "delta_fragments": fragment_count,
            "compressed_bytes": compressed_bytes,
            "delta_blob_bytes": delta_blob_bytes,
            "working_memory_budget_bytes": self.working_memory_budget_bytes,
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def apply(self) -> dict[str, Any]:
        if self.target_version == self.checkpoint.version:
            return {
                "operation": "canonical_delta_transform",
                "canonical_version": self.checkpoint.version,
                "target_version": self.target_version,
                "delta_tensors": 0,
                "folded_tensors": 0,
                "target_tensor_bytes": 0,
                "compressed_bytes": 0,
                "wall_s": 0.0,
            }

        self.checkpoint.begin_update(self.target_version)
        started = time.perf_counter()
        local_error = None
        local_stats = []
        try:
            for file_index, filename in enumerate(sorted(self.operations_by_file)):
                if file_index % self.world_size != self.rank:
                    continue
                local_stats.append(self._apply_file(filename))
        except Exception as exc:
            local_error = f"rank {self.rank}: {type(exc).__name__}: {exc}"

        errors = [
            error
            for error in _gather_objects(local_error, self.host_group, self.world_size)
            if error is not None
        ]
        if errors:
            reason = "delta transform failed: " + "; ".join(errors)
            self.checkpoint.fail_update(reason)
            raise RuntimeError(reason)

        self.checkpoint.finish_update(self.target_version)
        all_stats = _gather_objects(local_stats, self.host_group, self.world_size)
        file_stats = [stats for rank_stats in all_stats for stats in rank_stats]
        result = {
            "operation": "canonical_delta_transform",
            "canonical_version": self.setup_stats["canonical_version"],
            "target_version": self.target_version,
            "delta_tensors": sum(value["delta_tensors"] for value in file_stats),
            "delta_fragments": sum(value["delta_fragments"] for value in file_stats),
            "folded_tensors": sum(value["folded_tensors"] for value in file_stats),
            "source_files": sum(value["source_files"] for value in file_stats),
            "target_tensor_bytes": sum(
                value["target_tensor_bytes"] for value in file_stats
            ),
            "compressed_bytes": sum(value["compressed_bytes"] for value in file_stats),
            "source_read_worker_s": round(
                sum(value["source_read_worker_s"] for value in file_stats), 6
            ),
            "working_memory_budget_bytes": self.working_memory_budget_bytes,
            "wall_s": round(time.perf_counter() - started, 6),
        }
        logger.info(
            "Advanced canonical checkpoint from v%d to v%d: tensors=%d "
            "target_bytes=%d wall_time=%.3fs",
            result["canonical_version"],
            result["target_version"],
            result["delta_tensors"],
            result["target_tensor_bytes"],
            result["wall_s"],
        )
        return result

    def _apply_file(self, filename: str) -> dict[str, Any]:
        operations_by_name = self.operations_by_file[filename]
        tensors = {
            name: self.checkpoint.get_update_tensor_bytes(name)
            for name in operations_by_name
        }
        try:
            stats = self._transform_tensors(tensors, description=filename)
            stats["filename"] = filename
            return stats
        finally:
            tensors.clear()

    def _transform_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        description: str,
    ) -> dict[str, Any]:
        """Apply and verify deltas in caller-owned canonical byte tensors."""
        for name, tensor in tensors.items():
            if (
                tensor.device.type != "cpu"
                or tensor.dtype != torch.uint8
                or tensor.ndim != 1
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"canonical bytes for {name!r} must be contiguous CPU uint8"
                )
        operations_by_name = {
            name: self.operations_by_name[name]
            for name in tensors
            if name in self.operations_by_name
        }
        if not operations_by_name:
            return {
                "operation": "canonical_delta_transform",
                "description": description,
                "delta_tensors": 0,
                "delta_fragments": 0,
                "folded_tensors": 0,
                "source_files": 0,
                "target_tensor_bytes": 0,
                "compressed_bytes": 0,
                "source_read_worker_s": 0.0,
                "working_memory_budget_bytes": self.working_memory_budget_bytes,
                "workers": 0,
                "wall_s": 0.0,
            }

        try:
            available_cpus = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available_cpus = os.cpu_count() or 1
        workers = min(
            (
                available_cpus
                if envs.SGLANG_SET_CPU_AFFINITY.get()
                else max(1, available_cpus // self.world_size)
            ),
            len(operations_by_name),
        )
        budget = _ByteBudget(self.working_memory_budget_bytes)
        source_paths = {
            operation.source_path
            for operations in operations_by_name.values()
            for operation in operations
        }
        file_cache_limit = file_descriptor_cache_limit()
        source_files = FileDescriptorCache(os.O_RDONLY, file_cache_limit)

        def apply_tensor(
            item: tuple[str, list[_DeltaOperation]],
        ) -> tuple[int, int, int, float, int]:
            name, operations = item
            target_tensor = tensors[name]
            target = target_tensor.numpy()
            has_overwrite = any(
                operation.encoding == "overwrite" for operation in operations
            )
            folds_lineage = len(operations) > 1
            working_nbytes = _STREAM_CHUNK_BYTES
            if folds_lineage:
                working_nbytes += target.size
            if has_overwrite:
                working_nbytes += 4 * target.size

            def apply_operation(
                operation: _DeltaOperation,
                destination: np.ndarray,
                decompressor: zstandard.ZstdDecompressor,
            ) -> float:
                with source_files.acquire(operation.source_path) as source_fd:
                    source = PositionalFileRangeReader(
                        source_fd,
                        operation.source_offset,
                        operation.compressed_nbytes,
                        operation.source_path,
                        max_read_bytes=_STREAM_CHUNK_BYTES,
                    )
                    with decompressor.stream_reader(source, closefd=False) as reader:
                        if operation.encoding == "xor":
                            position = 0
                            while position < target.size:
                                block = reader.read(
                                    min(
                                        _STREAM_CHUNK_BYTES,
                                        target.size - position,
                                    )
                                )
                                if not block:
                                    break
                                delta = np.frombuffer(block, dtype=np.uint8)
                                region = destination[position : position + delta.size]
                                np.bitwise_xor(region, delta, out=region)
                                position += delta.size
                            if position != target.size or reader.read(1):
                                raise RuntimeError(
                                    "decompressed XOR size mismatch for "
                                    f"{name!r}: expected={target.size} "
                                    f"actual={position}"
                                )
                        else:
                            count = int.from_bytes(read_exact(reader, 4), "little")
                            if count > target.size:
                                raise RuntimeError(
                                    f"overwrite payload for {name!r} is invalid"
                                )
                            positions_payload = read_exact(reader, 4 * count)
                            positions = np.frombuffer(
                                positions_payload, dtype="<u4", count=count
                            )
                            if count and (
                                int(positions[-1]) >= target.size
                                or np.any(positions[1:] <= positions[:-1])
                            ):
                                raise RuntimeError(
                                    f"overwrite payload for {name!r} is invalid"
                                )
                            position = 0
                            while position < count:
                                block_nbytes = min(
                                    _STREAM_CHUNK_BYTES,
                                    count - position,
                                )
                                values = np.frombuffer(
                                    read_exact(reader, block_nbytes),
                                    dtype=np.uint8,
                                )
                                destination[
                                    positions[position : position + block_nbytes]
                                ] = values
                                position += block_nbytes
                            if reader.read(1):
                                raise RuntimeError(
                                    f"overwrite payload for {name!r} is oversized"
                                )
                    if source.position != operation.compressed_nbytes:
                        raise RuntimeError(
                            "compressed delta range was not fully consumed for "
                            f"{name!r}: expected={operation.compressed_nbytes} "
                            f"actual={source.position}"
                        )
                    return source.read_wall_s

            with budget.reserve(working_nbytes):
                if not folds_lineage:
                    destination = target
                elif has_overwrite:
                    destination = target.copy()
                else:
                    destination = np.zeros(target.size, dtype=np.uint8)
                compressed_bytes = 0
                source_read_wall_s = 0.0
                decompressor = zstandard.ZstdDecompressor()
                for operation in operations:
                    source_read_wall_s += apply_operation(
                        operation,
                        destination,
                        decompressor,
                    )
                    compressed_bytes += operation.compressed_nbytes

                if folds_lineage and not has_overwrite:
                    np.bitwise_xor(target, destination, out=target)
                    final_bytes = target
                elif folds_lineage:
                    final_bytes = destination
                else:
                    final_bytes = target

                final = operations[-1]
                hasher = create_checksum(final.checksum_algorithm)
                hasher.update(final_bytes)
                actual = hasher.hexdigest()
                if actual != final.expected_checksum:
                    raise RuntimeError(
                        f"checksum mismatch after reconstructing {name!r}: "
                        f"expected={final.expected_checksum} actual={actual}"
                    )

                if folds_lineage and has_overwrite:
                    np.copyto(target, destination)
                return (
                    target.size,
                    len(operations),
                    compressed_bytes,
                    source_read_wall_s,
                    int(folds_lineage),
                )

        started = time.perf_counter()
        work = sorted(
            operations_by_name.items(),
            key=lambda item: (-tensors[item[0]].numel(), item[0]),
        )
        try:
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="weight-delta",
            ) as pool:
                results = list(pool.map(apply_tensor, work))
        finally:
            source_files.close()

        return {
            "operation": "canonical_delta_transform",
            "description": description,
            "delta_tensors": len(operations_by_name),
            "delta_fragments": sum(value[1] for value in results),
            "source_files": len(source_paths),
            "target_tensor_bytes": sum(value[0] for value in results),
            "compressed_bytes": sum(value[2] for value in results),
            "source_read_worker_s": round(sum(value[3] for value in results), 6),
            "folded_tensors": sum(value[4] for value in results),
            "workers": workers,
            "file_descriptor_cache_limit": file_cache_limit,
            "peak_source_file_descriptors": source_files.peak_open_files,
            "wall_s": round(time.perf_counter() - started, 6),
        }
