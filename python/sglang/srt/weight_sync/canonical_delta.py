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
    read_exact,
    read_file_into_tensor,
)

logger = logging.getLogger(__name__)

_STREAM_CHUNK_BYTES = 4 << 20
_HOST_WORKING_MEMORY_BYTES = 8 << 30
_MAX_DELTA_TRANSFORM_WORKERS = 8


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
        self.operations_by_version_source: dict[
            int, dict[Path, list[tuple[str, _DeltaOperation]]]
        ] = {}
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
                root = version_dir(checkpoint_source_dir, version).resolve()
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
                    self.operations_by_version_source.setdefault(
                        version, {}
                    ).setdefault(tensor.source_path, []).append(
                        (tensor.name, operation)
                    )
                    plan_hasher.update(
                        json.dumps(
                            (
                                tensor.name,
                                canonical_filename,
                                version,
                                operation.encoding,
                                operation.checksum_algorithm,
                                operation.expected_checksum,
                                str(tensor.source_path.relative_to(root)),
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
                "delta_fragments": 0,
                "folded_tensors": 0,
                "source_files": 0,
                "source_blob_bytes": 0,
                "target_tensor_bytes": 0,
                "compressed_bytes": 0,
                "source_read_worker_s": 0.0,
                "wall_s": 0.0,
            }

        self.checkpoint.begin_update(self.target_version)
        started = time.perf_counter()
        local_source_stats = []
        budget = _ByteBudget(self.working_memory_budget_bytes)
        try:
            for version in range(
                self.setup_stats["canonical_version"] + 1,
                self.target_version + 1,
            ):
                local_error = None
                try:
                    sources = sorted(
                        self.operations_by_version_source.get(version, {}).items(),
                        key=lambda item: str(item[0]),
                    )
                    for source_index, (source_path, operations) in enumerate(sources):
                        if source_index % self.world_size != self.rank:
                            continue
                        tensors = {
                            name: self.checkpoint.get_update_tensor_bytes(name)
                            for name, _ in operations
                        }
                        try:
                            local_source_stats.append(
                                self._apply_source(
                                    source_path,
                                    operations,
                                    tensors,
                                    budget,
                                )
                            )
                        finally:
                            tensors.clear()
                except Exception as exc:
                    local_error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
                errors = [
                    error
                    for error in _gather_objects(
                        local_error,
                        self.host_group,
                        self.world_size,
                    )
                    if error is not None
                ]
                if errors:
                    raise RuntimeError(
                        f"delta v{version} transform failed: " + "; ".join(errors)
                    )

            local_error = None
            local_verify_stats = None
            try:
                tensors = {}
                local_operations = {}
                for file_index, filename in enumerate(sorted(self.operations_by_file)):
                    if file_index % self.world_size != self.rank:
                        continue
                    for name, operations in self.operations_by_file[filename].items():
                        tensors[name] = self.checkpoint.get_update_tensor_bytes(name)
                        local_operations[name] = operations
                try:
                    local_verify_stats = self._verify_tensors(
                        tensors,
                        local_operations,
                    )
                finally:
                    tensors.clear()
            except Exception as exc:
                local_error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
            errors = [
                error
                for error in _gather_objects(
                    local_error,
                    self.host_group,
                    self.world_size,
                )
                if error is not None
            ]
            if errors:
                raise RuntimeError("delta verification failed: " + "; ".join(errors))
            if local_verify_stats is None:
                raise RuntimeError("delta verification returned no statistics")
        except Exception as exc:
            reason = str(exc)
            self.checkpoint.fail_update(reason)
            raise

        self.checkpoint.finish_update(self.target_version)
        all_source_stats = _gather_objects(
            local_source_stats,
            self.host_group,
            self.world_size,
        )
        source_stats = [
            stats for rank_stats in all_source_stats for stats in rank_stats
        ]
        verify_stats = _gather_objects(
            local_verify_stats,
            self.host_group,
            self.world_size,
        )
        result = {
            "operation": "canonical_delta_transform",
            "canonical_version": self.setup_stats["canonical_version"],
            "target_version": self.target_version,
            "delta_tensors": sum(value["delta_tensors"] for value in verify_stats),
            "delta_fragments": sum(value["delta_fragments"] for value in source_stats),
            "folded_tensors": sum(value["folded_tensors"] for value in verify_stats),
            "source_files": len(source_stats),
            "source_blob_bytes": sum(
                value["source_blob_bytes"] for value in source_stats
            ),
            "target_tensor_bytes": sum(
                value["target_tensor_bytes"] for value in verify_stats
            ),
            "compressed_bytes": sum(
                value["compressed_bytes"] for value in source_stats
            ),
            "source_read_worker_s": round(
                sum(value["source_read_worker_s"] for value in source_stats), 6
            ),
            "working_memory_budget_bytes": self.working_memory_budget_bytes,
            "workers": max(
                (value["workers"] for value in source_stats),
                default=0,
            ),
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

    def _worker_count(self, work_items: int) -> int:
        if work_items <= 0:
            return 0
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available_cpus = os.cpu_count() or 1
        return min(
            (
                available_cpus
                if envs.SGLANG_SET_CPU_AFFINITY.get()
                else max(1, available_cpus // self.world_size)
            ),
            _MAX_DELTA_TRANSFORM_WORKERS,
            work_items,
        )

    def _apply_source(
        self,
        source_path: Path,
        operations: list[tuple[str, _DeltaOperation]],
        tensors: dict[str, torch.Tensor],
        budget: _ByteBudget,
    ) -> dict[str, Any]:
        """Read one immutable delta shard sequentially, then transform its tensors."""

        source_nbytes = source_path.stat().st_size
        source = torch.empty(source_nbytes, dtype=torch.uint8)
        read_stats = read_file_into_tensor(
            source_path,
            source,
            drop_cache_after_read=True,
        )
        source_view = memoryview(source.numpy())

        def apply_operation(
            item: tuple[str, _DeltaOperation],
            decompressor: zstandard.ZstdDecompressor,
        ) -> int:
            name, operation = item
            if operation.source_path != source_path:
                raise ValueError(
                    f"delta operation for {name!r} belongs to "
                    f"{operation.source_path}, not {source_path}"
                )
            end = operation.source_offset + operation.compressed_nbytes
            if end > source_nbytes:
                raise EOFError(
                    f"compressed delta range exceeds {source_path}: "
                    f"offset={operation.source_offset} "
                    f"bytes={operation.compressed_nbytes} file={source_nbytes}"
                )
            target = tensors[name].numpy()
            working_nbytes = _STREAM_CHUNK_BYTES
            if operation.encoding == "overwrite":
                working_nbytes += 4 * target.size
            with budget.reserve(working_nbytes):
                payload = source_view[operation.source_offset : end]
                with decompressor.stream_reader(payload, closefd=False) as reader:
                    if operation.encoding == "xor":
                        position = 0
                        while position < target.size:
                            block = reader.read(
                                min(_STREAM_CHUNK_BYTES, target.size - position)
                            )
                            if not block:
                                break
                            delta = np.frombuffer(block, dtype=np.uint8)
                            region = target[position : position + delta.size]
                            np.bitwise_xor(region, delta, out=region)
                            position += delta.size
                        if position != target.size or reader.read(1):
                            raise RuntimeError(
                                "decompressed XOR size mismatch for "
                                f"{name!r}: expected={target.size} actual={position}"
                            )
                    else:
                        count = int.from_bytes(read_exact(reader, 4), "little")
                        if count > target.size:
                            raise RuntimeError(
                                f"overwrite payload for {name!r} is invalid"
                            )
                        positions_payload = read_exact(reader, 4 * count)
                        positions = np.frombuffer(
                            positions_payload,
                            dtype="<u4",
                            count=count,
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
                            target[positions[position : position + block_nbytes]] = (
                                values
                            )
                            position += block_nbytes
                        if reader.read(1):
                            raise RuntimeError(
                                f"overwrite payload for {name!r} is oversized"
                            )
            return operation.compressed_nbytes

        work = sorted(operations, key=lambda item: item[1].source_offset)
        workers = self._worker_count(len(work))

        def apply_partition(
            partition: list[tuple[str, _DeltaOperation]],
        ) -> int:
            decompressor = zstandard.ZstdDecompressor()
            return sum(apply_operation(item, decompressor) for item in partition)

        partitions = [work[index::workers] for index in range(workers)]
        try:
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="weight-delta",
            ) as pool:
                compressed_nbytes = sum(pool.map(apply_partition, partitions))
        finally:
            source_view.release()
            del source
        return {
            "delta_fragments": len(work),
            "compressed_bytes": compressed_nbytes,
            "source_blob_bytes": source_nbytes,
            "source_read_worker_s": read_stats.wall_s,
            "workers": workers,
        }

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
                "source_blob_bytes": 0,
                "target_tensor_bytes": 0,
                "compressed_bytes": 0,
                "source_read_worker_s": 0.0,
                "working_memory_budget_bytes": self.working_memory_budget_bytes,
                "workers": 0,
                "wall_s": 0.0,
            }
        budget = _ByteBudget(self.working_memory_budget_bytes)
        started = time.perf_counter()
        source_stats = []
        for version in range(
            self.setup_stats["canonical_version"] + 1,
            self.target_version + 1,
        ):
            sources = self.operations_by_version_source.get(version, {})
            for source_path in sorted(sources, key=str):
                operations = [
                    (name, operation)
                    for name, operation in sources[source_path]
                    if name in tensors
                ]
                if operations:
                    source_stats.append(
                        self._apply_source(
                            source_path,
                            operations,
                            tensors,
                            budget,
                        )
                    )
        verify_stats = self._verify_tensors(tensors, operations_by_name)

        return {
            "operation": "canonical_delta_transform",
            "description": description,
            "delta_tensors": len(operations_by_name),
            "delta_fragments": sum(value["delta_fragments"] for value in source_stats),
            "source_files": len(source_stats),
            "source_blob_bytes": sum(
                value["source_blob_bytes"] for value in source_stats
            ),
            "target_tensor_bytes": verify_stats["target_tensor_bytes"],
            "compressed_bytes": sum(
                value["compressed_bytes"] for value in source_stats
            ),
            "source_read_worker_s": round(
                sum(value["source_read_worker_s"] for value in source_stats),
                6,
            ),
            "folded_tensors": verify_stats["folded_tensors"],
            "workers": max(
                (value["workers"] for value in source_stats),
                default=0,
            ),
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def _verify_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        operations_by_name: dict[str, list[_DeltaOperation]],
    ) -> dict[str, int]:
        def verify(item: tuple[str, list[_DeltaOperation]]) -> int:
            name, operations = item
            final = operations[-1]
            hasher = create_checksum(final.checksum_algorithm)
            hasher.update(tensors[name].numpy())
            actual = hasher.hexdigest()
            if actual != final.expected_checksum:
                raise RuntimeError(
                    f"checksum mismatch after reconstructing {name!r}: "
                    f"expected={final.expected_checksum} actual={actual}"
                )
            return tensors[name].numel()

        work = sorted(operations_by_name.items())
        workers = self._worker_count(len(work))
        if not work:
            return {
                "delta_tensors": 0,
                "folded_tensors": 0,
                "target_tensor_bytes": 0,
            }
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="weight-checksum",
        ) as pool:
            target_nbytes = sum(pool.map(verify, work))
        return {
            "delta_tensors": len(work),
            "folded_tensors": sum(len(operations) > 1 for _, operations in work),
            "target_tensor_bytes": target_nbytes,
        }
