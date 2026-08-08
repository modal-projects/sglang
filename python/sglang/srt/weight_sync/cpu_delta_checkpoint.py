"""Verified delta transforms over host-shared canonical checkpoints."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zstandard

from sglang.srt.environ import envs
from sglang.srt.weight_sync.canonical_checkpoint import CanonicalCheckpoint
from sglang.srt.weight_sync.checksum import create_checksum, validate_checksum
from sglang.srt.weight_sync.file_io import PositionalFileRangeReader, read_exact
from sglang.srt.weight_sync.safetensors_buffer import (
    MAX_SAFETENSORS_HEADER_BYTES,
    SafetensorsLayout,
    parse_safetensors_header,
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


def _version_dir(checkpoint_source_dir: str | Path, version: int) -> Path:
    return Path(checkpoint_source_dir) / f"weight_v{version:06d}"


def _load_delta_index(root: Path, expected_version: int) -> dict[str, Any]:
    path = root / "model.safetensors.index.json"
    try:
        index = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"checkpoint index is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid checkpoint index: {path}") from exc
    if not isinstance(index, dict):
        raise ValueError(f"invalid checkpoint index: {path}")
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict):
        raise ValueError(f"invalid checkpoint index: {path}")
    if not all(
        isinstance(name, str) and name and isinstance(filename, str) and filename
        for name, filename in weight_map.items()
    ):
        raise ValueError(f"invalid checkpoint weight map: {path}")
    encoding = metadata.get("delta_encoding")
    if encoding is None:
        raise ValueError(
            f"weight_v{expected_version:06d} is a full checkpoint; "
            "replace the canonical checkpoint before applying later deltas"
        )
    if encoding not in {"xor", "overwrite"}:
        raise NotImplementedError(
            f"delta v{expected_version} encoding {encoding!r} is unsupported"
        )
    try:
        version = int(metadata["version"])
        base_version = int(metadata["base_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint has no valid delta lineage: {path}") from exc
    if version != expected_version:
        raise ValueError(
            f"checkpoint version mismatch: directory is v{expected_version}, "
            f"index declares v{version}"
        )
    if base_version != expected_version - 1:
        raise ValueError(
            f"delta v{expected_version} builds on v{base_version}, "
            f"expected v{expected_version - 1}"
        )
    if metadata.get("compression_format") != "zstd":
        raise NotImplementedError(
            f"delta v{expected_version} compression "
            f"{metadata.get('compression_format')!r} is unsupported"
        )
    checksum_algorithm = metadata.get("checksum_format")
    if not isinstance(checksum_algorithm, str):
        raise ValueError(f"delta v{expected_version} has no checksum format")
    create_checksum(checksum_algorithm)
    return index


def _read_delta_header(path: Path) -> tuple[SafetensorsLayout, dict[str, str]]:
    file_nbytes = path.stat().st_size
    with path.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError(f"delta source is shorter than its header: {path}")
        header_nbytes = int.from_bytes(prefix, "little")
        if (
            header_nbytes <= 0
            or header_nbytes > MAX_SAFETENSORS_HEADER_BYTES
            or 8 + header_nbytes > file_nbytes
        ):
            raise ValueError(
                f"invalid delta header length in {path}: "
                f"header={header_nbytes} file={file_nbytes}"
            )
        header_bytes = file.read(header_nbytes)
    layout, metadata = parse_safetensors_header(
        header_nbytes=header_nbytes,
        header_bytes=header_bytes,
        file_nbytes=file_nbytes,
    )
    for name, entry in layout.tensors.items():
        if entry.dtype_code != "U8" or entry.shape != (
            entry.relative_end - entry.relative_begin,
        ):
            raise ValueError(
                f"compressed delta tensor {name!r} must be one-dimensional U8"
            )
    return layout, metadata


def _gather_objects(value: Any, group: Any, world_size: int) -> list[Any]:
    if world_size == 1:
        return [value]
    values = [None] * world_size
    torch.distributed.all_gather_object(values, value, group=group)
    return values


class DeltaCheckpointTransform:
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
        plan_hasher = hashlib.sha256()
        plan_hasher.update(f"{checkpoint.version}:{target_version}\n".encode())
        fragment_count = 0
        source_paths = set()
        compressed_bytes = 0
        delta_blob_bytes = 0
        try:
            for version in range(checkpoint.version + 1, target_version + 1):
                root = _version_dir(checkpoint_source_dir, version)
                index = _load_delta_index(root, version)
                metadata = index["metadata"]
                weight_map = index["weight_map"]
                for filename in sorted(set(weight_map.values())):
                    relative = Path(filename)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"invalid delta shard path: {filename!r}")
                    path = root / relative
                    try:
                        layout, checksums = _read_delta_header(path)
                    except FileNotFoundError as exc:
                        raise FileNotFoundError(
                            f"incomplete source version {root}: "
                            f"missing blob {filename}"
                        ) from exc
                    expected_names = {
                        name
                        for name, mapped_filename in weight_map.items()
                        if mapped_filename == filename
                    }
                    actual_names = set(layout.tensors)
                    if actual_names != expected_names:
                        raise ValueError(
                            f"delta blob/index tensor mismatch for {path}: "
                            f"missing={sorted(expected_names - actual_names)[:8]} "
                            f"extra={sorted(actual_names - expected_names)[:8]}"
                        )
                    checksum_names = set(checksums)
                    if checksum_names != expected_names:
                        raise ValueError(
                            f"delta checksum/tensor mismatch for {path}: "
                            f"missing={sorted(expected_names - checksum_names)[:8]} "
                            f"extra={sorted(checksum_names - expected_names)[:8]}"
                        )
                    for name in sorted(layout.tensors):
                        entry = layout.tensors[name]
                        canonical_filename = checkpoint.weight_map.get(name)
                        if canonical_filename is None:
                            raise KeyError(
                                f"delta tensor {name!r} is absent from the "
                                "canonical checkpoint"
                            )
                        expected_checksum = validate_checksum(
                            metadata["checksum_format"], checksums.get(name)
                        )
                        operation = _DeltaOperation(
                            version=version,
                            encoding=metadata["delta_encoding"],
                            checksum_algorithm=metadata["checksum_format"],
                            expected_checksum=expected_checksum,
                            source_path=path,
                            source_offset=(layout.data_offset + entry.relative_begin),
                            compressed_nbytes=(
                                entry.relative_end - entry.relative_begin
                            ),
                        )
                        self.operations_by_file.setdefault(
                            canonical_filename, {}
                        ).setdefault(name, []).append(operation)
                        plan_hasher.update(
                            json.dumps(
                                (
                                    name,
                                    canonical_filename,
                                    version,
                                    operation.encoding,
                                    operation.checksum_algorithm,
                                    operation.expected_checksum,
                                    str(relative),
                                    operation.source_offset,
                                    operation.compressed_nbytes,
                                ),
                                separators=(",", ":"),
                            ).encode()
                        )
                        plan_hasher.update(b"\n")
                        fragment_count += 1
                        compressed_bytes += operation.compressed_nbytes
                    source_paths.add(path)
                    delta_blob_bytes += layout.file_nbytes
        except Exception as exc:
            plan_error = f"rank {self.rank}: {type(exc).__name__}: {exc}"

        errors = [
            error
            for error in _gather_objects(plan_error, host_group, self.world_size)
            if error is not None
        ]
        if errors:
            raise RuntimeError("failed to plan delta lineage: " + "; ".join(errors))

        signature = plan_hasher.hexdigest()
        signatures = _gather_objects(signature, host_group, self.world_size)
        if any(value != signature for value in signatures):
            raise RuntimeError(f"delta plans differ across local workers: {signatures}")

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
            return self._apply_tensors(filename, tensors, operations_by_name)
        finally:
            tensors.clear()

    def _apply_tensors(
        self,
        filename: str,
        tensors: dict[str, torch.Tensor],
        operations_by_name: dict[str, list[_DeltaOperation]],
    ) -> dict[str, Any]:
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
            len(tensors),
        )
        budget = _ByteBudget(self.working_memory_budget_bytes)
        final_operations = {
            name: operations[-1] for name, operations in operations_by_name.items()
        }
        operations_by_source = {}
        for name, operations in operations_by_name.items():
            for operation in operations:
                operations_by_source.setdefault(
                    (operation.version, operation.source_path), []
                ).append((name, operation))

        def apply_operation(
            item: tuple[str, _DeltaOperation],
            source_fd: int,
            decompressor: zstandard.ZstdDecompressor,
        ) -> tuple[int, int, float]:
            name, operation = item
            target = tensors[name].numpy()
            is_final = operation is final_operations[name]
            hasher = create_checksum(operation.checksum_algorithm) if is_final else None
            source = PositionalFileRangeReader(
                source_fd,
                operation.source_offset,
                operation.compressed_nbytes,
                operation.source_path,
                max_read_bytes=_STREAM_CHUNK_BYTES,
            )
            with decompressor.stream_reader(source, closefd=False) as reader:
                if operation.encoding == "xor":
                    with budget.reserve(_STREAM_CHUNK_BYTES):
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
                            if hasher is not None:
                                hasher.update(region)
                            position += delta.size
                        if position != target.size or reader.read(1):
                            raise RuntimeError(
                                f"decompressed XOR size mismatch for {name!r}: "
                                f"expected={target.size} actual={position}"
                            )
                else:
                    count = int.from_bytes(read_exact(reader, 4), "little")
                    if count > target.size:
                        raise RuntimeError(f"overwrite payload for {name!r} is invalid")
                    with budget.reserve(4 * count + _STREAM_CHUNK_BYTES):
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
                            block_nbytes = min(_STREAM_CHUNK_BYTES, count - position)
                            values = np.frombuffer(
                                read_exact(reader, block_nbytes), dtype=np.uint8
                            )
                            target[positions[position : position + block_nbytes]] = (
                                values
                            )
                            position += block_nbytes
                    if reader.read(1):
                        raise RuntimeError(
                            f"overwrite payload for {name!r} is oversized"
                        )
                    if hasher is not None:
                        hasher.update(target)
            if source.position != operation.compressed_nbytes:
                raise RuntimeError(
                    f"compressed delta range was not fully consumed for {name!r}: "
                    f"expected={operation.compressed_nbytes} "
                    f"actual={source.position}"
                )

            if hasher is not None:
                actual = hasher.hexdigest()
                if actual != operation.expected_checksum:
                    raise RuntimeError(
                        f"checksum mismatch after reconstructing {name!r}: "
                        f"expected={operation.expected_checksum} actual={actual}"
                    )
            return (
                target.size if is_final else 0,
                operation.compressed_nbytes,
                source.read_wall_s,
            )

        results = []
        started = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="weight-delta",
        ) as pool:
            for (_, source_path), operations in sorted(
                operations_by_source.items(),
                key=lambda item: (item[0][0], str(item[0][1])),
            ):
                source_fd = os.open(source_path, os.O_RDONLY)
                futures = []
                try:
                    chunk_count = min(workers, len(operations))
                    chunks = [[] for _ in range(chunk_count)]
                    chunk_bytes = [0] * chunk_count
                    for operation in sorted(
                        operations,
                        key=lambda item: (-tensors[item[0]].numel(), item[0]),
                    ):
                        chunk_index = min(
                            range(chunk_count), key=chunk_bytes.__getitem__
                        )
                        chunks[chunk_index].append(operation)
                        chunk_bytes[chunk_index] += tensors[operation[0]].numel()

                    def apply_chunk(chunk):
                        decompressor = zstandard.ZstdDecompressor()
                        return [
                            apply_operation(operation, source_fd, decompressor)
                            for operation in chunk
                        ]

                    futures = [
                        pool.submit(apply_chunk, chunk) for chunk in chunks if chunk
                    ]
                    wait(futures)
                    for future in futures:
                        results.extend(future.result())
                finally:
                    wait(futures)
                    os.close(source_fd)

        return {
            "filename": filename,
            "delta_tensors": len(tensors),
            "delta_fragments": len(results),
            "source_files": len(operations_by_source),
            "target_tensor_bytes": sum(value[0] for value in results),
            "compressed_bytes": sum(value[1] for value in results),
            "source_read_worker_s": round(sum(value[2] for value in results), 6),
            "workers": workers,
            "wall_s": round(time.perf_counter() - started, 6),
        }
