"""Checkpoint-delta reconstruction for CPU weight staging.

CPU weight staging maintains one canonical checkpoint in host memory or on
host-local disk. Updates apply only the delta versions after that checkpoint,
and every changed tensor is verified before the ordinary SGLang loader can
publish a rank-ready CPU image.

Compressed payloads are streamed once into bounded work buffers while exactly
one local rank owns reconstruction of each canonical checkpoint file. The
complete checkpoint and rank-ready image stay in CPU memory; no lineage-sized
delta arena, runtime-layout assumption, or tensor-sparsity assumption enters
this module. Disk-backed callers may persist each verified bounded buffer while
the loader consumes the same bytes.
"""

from __future__ import annotations

import json
import logging
import math
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
from sglang.srt.weight_sync.checksum import create_checksum
from sglang.srt.weight_sync.file_io import PositionalFileRangeReader, read_exact

logger = logging.getLogger(__name__)

_POSITIONAL_IO_CHUNK_BYTES = 64 << 20
_XOR_STREAM_CHUNK_BYTES = 4 << 20
_HOST_DELTA_APPLY_MEMORY_BYTES = 8 << 30


def _version_dir(checkpoint_source_dir: str, version: int) -> Path:
    return Path(checkpoint_source_dir) / f"weight_v{version:06d}"


def _checkpoint_index(
    root: Path,
    *,
    expected_version: int | None = None,
) -> dict[str, Any]:
    path = root / "model.safetensors.index.json"
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        if expected_version is None:
            # Base checkpoints follow the ordinary loader and may be a single
            # unindexed safetensors file.
            from sglang.srt.weight_sync.cpu_weight_cache import (
                _checkpoint_weight_map,
            )

            weight_map, _ = _checkpoint_weight_map(str(root))
            return {"metadata": {}, "weight_map": weight_map}
        raise FileNotFoundError(f"checkpoint index is missing: {path}") from exc
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"invalid checkpoint weight map: {path}")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid checkpoint metadata: {path}")
    if expected_version is not None:
        try:
            published_version = int(metadata["version"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"checkpoint manifest has no valid version: {path}"
            ) from exc
        if published_version != expected_version:
            raise ValueError(
                f"checkpoint version mismatch: directory is v{expected_version}, "
                f"manifest declares v{published_version}"
            )
    if not weight_map and not metadata.get("delta_encoding"):
        raise ValueError(f"full checkpoint has an empty weight map: {path}")
    return payload


def validate_delta_target(checkpoint_source_dir: str, target_version: int) -> None:
    """Reject CPU staging for base or full-checkpoint targets."""

    if target_version <= 0:
        raise ValueError("CPU weight staging requires a delta target")
    index = _checkpoint_index(
        _version_dir(checkpoint_source_dir, target_version),
        expected_version=target_version,
    )
    metadata = index.get("metadata") or {}
    if not metadata.get("delta_encoding"):
        raise ValueError(
            "CPU weight staging supports delta targets only; "
            f"weight_v{target_version:06d} is a full checkpoint"
        )


def _pread_exact(fd: int, offset: int, nbytes: int, path: Path) -> bytearray:
    result = bytearray(nbytes)
    view = memoryview(result)
    position = 0
    while position < nbytes:
        end = min(position + _POSITIONAL_IO_CHUNK_BYTES, nbytes)
        nread = os.preadv(fd, [view[position:end]], offset + position)
        if nread <= 0:
            raise FileNotFoundError(
                f"incomplete delta blob {path}: offset={offset} "
                f"expected={nbytes} actual={position}"
            )
        position += nread
    return result


def _read_delta_header(path: Path) -> tuple[int, int, dict[str, Any]]:
    file_nbytes = path.stat().st_size
    fd = os.open(path, os.O_RDONLY)
    try:
        prefix = _pread_exact(fd, 0, 8, path)
        header_nbytes = int.from_bytes(prefix, "little")
        if header_nbytes <= 0 or 8 + header_nbytes > file_nbytes:
            raise FileNotFoundError(
                f"incomplete delta blob {path}: invalid header length "
                f"{header_nbytes}"
            )
        header = json.loads(_pread_exact(fd, 8, header_nbytes, path))
    finally:
        os.close(fd)
    if not isinstance(header, dict):
        raise ValueError(f"invalid delta header: {path}")
    return file_nbytes, 8 + header_nbytes, header


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
    """Bound concurrent decompression buffers without rejecting large tensors."""

    def __init__(self, limit: int):
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


def _resolve_lineage(
    *,
    base_checkpoint_dir: str,
    canonical_checkpoint_layout_dir: str | None,
    checkpoint_source_dir: str,
    target_version: int,
    canonical_version: int = 0,
) -> tuple[Path, int, list[tuple[int, Path, dict[str, Any]]]]:
    """Find the newest full source and the ordered deltas needed for target."""

    if target_version < 0:
        raise ValueError("target_version must be non-negative")
    if canonical_version < 0:
        raise ValueError("canonical_version must be non-negative")
    if target_version < canonical_version:
        raise ValueError(
            f"target version {target_version} precedes canonical "
            f"version {canonical_version}"
        )
    if canonical_checkpoint_layout_dir is not None:
        checkpoint_root = Path(canonical_checkpoint_layout_dir)
        _checkpoint_index(checkpoint_root)
        deltas = []
        previous = canonical_version
        for version in range(canonical_version + 1, target_version + 1):
            root = _version_dir(checkpoint_source_dir, version)
            index = _checkpoint_index(root, expected_version=version)
            metadata = index.get("metadata", {})
            if "delta_encoding" not in metadata:
                # A full target is never staged in CPU memory. A later delta
                # may still use it as its canonical reconstruction anchor.
                return _resolve_lineage(
                    base_checkpoint_dir=base_checkpoint_dir,
                    canonical_checkpoint_layout_dir=None,
                    checkpoint_source_dir=checkpoint_source_dir,
                    target_version=target_version,
                    canonical_version=0,
                )
            if int(metadata.get("base_version", -1)) != previous:
                raise RuntimeError(
                    f"out-of-order delta v{version}: builds on "
                    f"{metadata.get('base_version')}, expected {previous}"
                )
            if metadata.get("compression_format") != "zstd":
                raise NotImplementedError(
                    f"delta v{version} compression "
                    f"{metadata.get('compression_format')!r} is unsupported"
                )
            encoding = metadata.get("delta_encoding")
            if encoding not in {"xor", "overwrite"}:
                raise NotImplementedError(
                    f"delta v{version} encoding {encoding!r} is unsupported"
                )
            deltas.append((version, root, index))
            previous = version
        return checkpoint_root, canonical_version, deltas
    if canonical_version:
        raise ValueError(
            "canonical_checkpoint_layout_dir is required when advancing a "
            "canonical checkpoint"
        )
    if target_version == 0:
        root = Path(base_checkpoint_dir)
        _checkpoint_index(root)
        return root, 0, []

    start = target_version
    indexes: dict[int, dict[str, Any]] = {}
    while start > 0:
        root = _version_dir(checkpoint_source_dir, start)
        index = _checkpoint_index(root, expected_version=start)
        indexes[start] = index
        if "delta_encoding" not in index.get("metadata", {}):
            break
        start -= 1

    if start == 0:
        checkpoint_root = Path(base_checkpoint_dir)
        _checkpoint_index(checkpoint_root)
    else:
        checkpoint_root = _version_dir(checkpoint_source_dir, start)

    deltas = []
    previous = start
    for version in range(start + 1, target_version + 1):
        root = _version_dir(checkpoint_source_dir, version)
        index = indexes.get(version) or _checkpoint_index(
            root, expected_version=version
        )
        metadata = index.get("metadata", {})
        if "delta_encoding" not in metadata:
            raise ValueError(
                f"full checkpoint v{version} appears inside delta lineage "
                f"v{start + 1}..v{target_version}"
            )
        if int(metadata.get("base_version", -1)) != previous:
            raise RuntimeError(
                f"out-of-order delta v{version}: builds on "
                f"{metadata.get('base_version')}, expected {previous}"
            )
        if metadata.get("compression_format") != "zstd":
            raise NotImplementedError(
                f"delta v{version} compression "
                f"{metadata.get('compression_format')!r} is unsupported"
            )
        encoding = metadata.get("delta_encoding")
        if encoding not in {"xor", "overwrite"}:
            raise NotImplementedError(
                f"delta v{version} encoding {encoding!r} is unsupported"
            )
        deltas.append((version, root, index))
        previous = version
    return checkpoint_root, start, deltas


class DeltaCheckpointTransform:
    """Apply a verified delta lineage to a canonical CPU checkpoint."""

    def __init__(
        self,
        *,
        base_checkpoint_dir: str,
        checkpoint_source_dir: str,
        target_version: int,
        cpu_group: Any,
        canonical_version: int = 0,
        canonical_checkpoint_layout_dir: str | None = None,
    ):
        started = time.perf_counter()
        self.canonical_version = canonical_version
        distributed = torch.distributed.is_initialized()
        self.world_size = (
            torch.distributed.get_world_size(group=cpu_group) if distributed else 1
        )
        self.rank = torch.distributed.get_rank(group=cpu_group) if distributed else 0
        self.working_memory_budget_bytes = max(
            1,
            _HOST_DELTA_APPLY_MEMORY_BYTES // self.world_size,
        )
        self.operations_by_file: dict[str, dict[str, list[_DeltaOperation]]] = {}
        self.setup_stats: dict[str, Any] = {
            "operation": "plan_delta_checkpoint",
            "canonical_version": canonical_version,
            "target_version": target_version,
            "delta_versions": [],
            "compressed_bytes": 0,
            "delta_blob_bytes": 0,
            "working_memory_budget_bytes": self.working_memory_budget_bytes,
            "wall_s": 0.0,
        }

        checkpoint_root = Path(base_checkpoint_dir)
        deltas = []
        plan_error = None
        delta_blob_bytes = 0
        compressed_bytes = 0
        source_paths = set()
        try:
            checkpoint_root, resolved_canonical_version, deltas = _resolve_lineage(
                base_checkpoint_dir=base_checkpoint_dir,
                canonical_checkpoint_layout_dir=canonical_checkpoint_layout_dir,
                checkpoint_source_dir=checkpoint_source_dir,
                target_version=target_version,
                canonical_version=canonical_version,
            )
            self.canonical_version = resolved_canonical_version
            self.setup_stats["canonical_version"] = resolved_canonical_version
            base_weight_map = _checkpoint_index(checkpoint_root)["weight_map"]
            for version, root, index in deltas:
                metadata = index["metadata"]
                checksum_algorithm = metadata.get("checksum_format")
                if not isinstance(checksum_algorithm, str):
                    raise ValueError(f"delta v{version} has no valid checksum format")
                create_checksum(checksum_algorithm)
                self.setup_stats["delta_versions"].append(version)
                for filename in sorted(set(index["weight_map"].values())):
                    path = root / filename
                    try:
                        file_nbytes, data_offset, header = _read_delta_header(path)
                    except FileNotFoundError as exc:
                        raise FileNotFoundError(
                            f"incomplete source version {root}: "
                            f"missing blob {filename}"
                        ) from exc
                    checksums = header.get("__metadata__", {})
                    if not isinstance(checksums, dict):
                        raise ValueError(f"invalid delta checksums: {path}")
                    ranges = []
                    seen_names = set()
                    for name, info in header.items():
                        if name == "__metadata__":
                            continue
                        if index["weight_map"].get(name) != path.name:
                            raise ValueError(
                                f"delta blob {path} contains unindexed tensor {name!r}"
                            )
                        seen_names.add(name)
                        if not isinstance(info, dict):
                            raise ValueError(
                                f"invalid delta metadata for {name!r} in {path}"
                            )
                        offsets = info.get("data_offsets")
                        if (
                            not isinstance(offsets, list)
                            or len(offsets) != 2
                            or not all(isinstance(value, int) for value in offsets)
                        ):
                            raise ValueError(
                                f"invalid delta offsets for {name!r} in {path}"
                            )
                        begin, end = offsets
                        if begin < 0 or begin > end or data_offset + end > file_nbytes:
                            raise FileNotFoundError(
                                f"incomplete delta range for {name!r} in {path}"
                            )
                        ranges.append((begin, end, name))
                        base_filename = base_weight_map.get(name)
                        if base_filename is None:
                            raise KeyError(
                                f"delta tensor {name!r} is absent from full "
                                f"checkpoint {checkpoint_root}"
                            )
                        expected_checksum = checksums.get(name)
                        if (
                            not isinstance(expected_checksum, str)
                            or not expected_checksum
                        ):
                            raise ValueError(
                                f"delta tensor {name!r} has no target checksum"
                            )
                        self.operations_by_file.setdefault(
                            base_filename,
                            {},
                        ).setdefault(name, []).append(
                            _DeltaOperation(
                                version=version,
                                encoding=metadata["delta_encoding"],
                                checksum_algorithm=checksum_algorithm,
                                expected_checksum=expected_checksum,
                                source_path=path,
                                source_offset=data_offset + begin,
                                compressed_nbytes=end - begin,
                            )
                        )
                        compressed_bytes += end - begin
                    cursor = 0
                    for begin, end, name in sorted(ranges):
                        if begin != cursor:
                            relation = "overlaps" if begin < cursor else "leaves a gap"
                            raise ValueError(
                                f"delta range for {name!r} {relation} in "
                                f"{path}: expected_begin={cursor} "
                                f"actual_begin={begin}"
                            )
                        cursor = end
                    expected_file_nbytes = data_offset + cursor
                    if expected_file_nbytes != file_nbytes:
                        raise FileNotFoundError(
                            f"incomplete delta blob {path}: file has "
                            f"{file_nbytes} bytes, header declares "
                            f"{expected_file_nbytes}"
                        )
                    expected_names = {
                        name
                        for name, expected_filename in index["weight_map"].items()
                        if expected_filename == path.name
                    }
                    if seen_names != expected_names:
                        raise ValueError(
                            f"delta blob/index tensor mismatch for {path}: "
                            f"missing={sorted(expected_names - seen_names)[:20]} "
                            f"extra={sorted(seen_names - expected_names)[:20]}"
                        )
                    source_paths.add(path)
                    delta_blob_bytes += file_nbytes
        except Exception as exc:
            plan_error = f"{type(exc).__name__}: {exc}"

        if self.world_size > 1:
            plan_errors: list[str | None] = [None] * self.world_size
            torch.distributed.all_gather_object(
                plan_errors,
                plan_error,
                group=cpu_group,
            )
        else:
            plan_errors = [plan_error]
        plan_errors = [value for value in plan_errors if value is not None]
        if plan_errors:
            raise RuntimeError(
                "failed to plan delta lineage: " + "; ".join(plan_errors)
            )

        for names in self.operations_by_file.values():
            for operations in names.values():
                operations.sort(key=lambda value: value.version)
        self.operations_by_name = {
            name: operations
            for names in self.operations_by_file.values()
            for name, operations in names.items()
        }

        self.checkpoint_root = checkpoint_root
        self.setup_stats.update(
            {
                "compressed_bytes": compressed_bytes,
                "delta_blob_bytes": delta_blob_bytes,
                "delta_shards": len(source_paths),
                "delta_tensors": sum(
                    len(names) for names in self.operations_by_file.values()
                ),
                "delta_fragments": sum(
                    len(operations)
                    for names in self.operations_by_file.values()
                    for operations in names.values()
                ),
                "wall_s": round(time.perf_counter() - started, 6),
            }
        )
        logger.info(
            "Planned CPU delta versions %s: compressed_bytes=%d wall_time=%.3fs",
            self.setup_stats["delta_versions"],
            self.setup_stats["compressed_bytes"],
            self.setup_stats["wall_s"],
        )

    def transform_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        description: str,
        write_tensor: Any = None,
        write_block_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Apply and verify canonical target tensors in caller-owned buffers."""

        if write_tensor is not None and (
            write_block_bytes is None or write_block_bytes <= 0
        ):
            raise ValueError("write_block_bytes must be positive when persisting")
        if write_tensor is not None:
            assert write_block_bytes is not None
        names = {
            name: self.operations_by_name[name]
            for name in tensors
            if name in self.operations_by_name
        }
        if not names:
            return {
                "operation": "canonical_delta_transform",
                "description": description,
                "delta_tensors": 0,
                "target_tensor_bytes": 0,
                "compressed_bytes": 0,
                "working_memory_budget_bytes": self.working_memory_budget_bytes,
                "wall_s": 0.0,
            }

        started = time.perf_counter()
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available_cpus = os.cpu_count() or 1
        worker_count = min(
            (
                available_cpus
                if envs.SGLANG_SET_CPU_AFFINITY.get()
                else max(1, available_cpus // self.world_size)
            ),
            len(names),
        )
        memory_budget = _ByteBudget(self.working_memory_budget_bytes)
        final_operation = {name: operations[-1] for name, operations in names.items()}
        dirty_blocks = (
            {
                name: np.zeros(
                    math.ceil(tensors[name].numel() / write_block_bytes),
                    dtype=np.bool_,
                )
                for name in names
            }
            if write_tensor is not None
            else {}
        )
        operations_by_source = {}
        for name, operations in names.items():
            for operation in operations:
                operations_by_source.setdefault(
                    (operation.version, operation.source_path),
                    [],
                ).append((name, operation))

        def apply_operation(
            item: tuple[str, _DeltaOperation],
            source_fd: int,
            decompressor: zstandard.ZstdDecompressor,
        ) -> tuple[int, int, float]:
            name, operation = item
            region_tensor = tensors[name]
            region = region_tensor.numpy()
            is_final = operation is final_operation[name]
            hasher = create_checksum(operation.checksum_algorithm) if is_final else None
            max_payload_nbytes = (
                _XOR_STREAM_CHUNK_BYTES
                if operation.encoding == "xor"
                else 4 + 5 * region.size
            )
            with memory_budget.reserve(max_payload_nbytes):
                source = PositionalFileRangeReader(
                    source_fd,
                    operation.source_offset,
                    operation.compressed_nbytes,
                    operation.source_path,
                    max_read_bytes=_XOR_STREAM_CHUNK_BYTES,
                )
                if operation.encoding == "xor":
                    with decompressor.stream_reader(
                        source,
                        closefd=False,
                    ) as reader:
                        position = 0
                        while position < region.size:
                            block = reader.read(
                                min(
                                    _XOR_STREAM_CHUNK_BYTES,
                                    region.size - position,
                                )
                            )
                            if not block:
                                break
                            delta = np.frombuffer(block, dtype=np.uint8)
                            has_changes = True
                            if write_tensor is not None:
                                block_begin = position // write_block_bytes
                                complete_bytes = (
                                    delta.size // write_block_bytes * write_block_bytes
                                )
                                block_changes = np.any(
                                    delta[:complete_bytes].reshape(
                                        -1,
                                        write_block_bytes,
                                    ),
                                    axis=1,
                                )
                                if complete_bytes:
                                    dirty_blocks[name][
                                        block_begin : block_begin
                                        + complete_bytes // write_block_bytes
                                    ] |= block_changes
                                tail_has_changes = (
                                    complete_bytes != delta.size
                                    and np.any(delta[complete_bytes:])
                                )
                                if tail_has_changes:
                                    dirty_blocks[name][
                                        block_begin
                                        + complete_bytes // write_block_bytes
                                    ] = True
                                has_changes = bool(
                                    np.any(block_changes) or tail_has_changes
                                )
                            if has_changes:
                                np.bitwise_xor(
                                    region[position : position + delta.size],
                                    delta,
                                    out=region[position : position + delta.size],
                                )
                            if hasher is not None:
                                hasher.update(region[position : position + delta.size])
                            position += delta.size
                        if position != region.size or reader.read(1):
                            raise RuntimeError(
                                f"decompressed XOR size mismatch for {name!r}: "
                                f"expected={region.size} actual={position}"
                            )
                else:
                    with decompressor.stream_reader(
                        source,
                        closefd=False,
                    ) as reader:
                        count = int.from_bytes(read_exact(reader, 4), "little")
                        if count > region.size:
                            raise RuntimeError(
                                f"overwrite payload for {name!r} is invalid"
                            )
                        payload_nbytes = 5 * count
                        payload = read_exact(reader, payload_nbytes)
                        if reader.read(1):
                            raise RuntimeError(
                                f"overwrite payload for {name!r} is oversized"
                            )
                        positions_nbytes = 4 * count
                        positions = np.frombuffer(
                            payload,
                            dtype="<u4",
                            count=count,
                        )
                        values = np.frombuffer(
                            payload,
                            dtype=np.uint8,
                            count=count,
                            offset=positions_nbytes,
                        )
                        if count and int(positions.max()) >= region.size:
                            raise RuntimeError(
                                f"overwrite payload for {name!r} is invalid"
                            )
                        region[positions] = values
                        if write_tensor is not None and count:
                            dirty_blocks[name][
                                np.unique(positions // write_block_bytes)
                            ] = True
                        if hasher is not None:
                            hasher.update(region)
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
                if write_tensor is not None:
                    changes = np.flatnonzero(
                        np.diff(
                            np.concatenate(
                                (
                                    np.array([False]),
                                    dirty_blocks[name],
                                    np.array([False]),
                                )
                            )
                        )
                    )
                    write_tensor(
                        name,
                        region,
                        [
                            (
                                int(begin) * write_block_bytes,
                                min(int(end) * write_block_bytes, region.size),
                            )
                            for begin, end in changes.reshape(-1, 2)
                        ],
                    )
            return (
                region.size if is_final else 0,
                operation.compressed_nbytes,
                source.read_wall_s,
            )

        results = []
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="weight-delta",
        ) as pool:
            for (_, source_path), operations in sorted(
                operations_by_source.items(),
                key=lambda item: (item[0][0], str(item[0][1])),
            ):
                source_fd = os.open(source_path, os.O_RDONLY)
                futures = []
                try:
                    # Keep executor and decompressor setup proportional to
                    # workers rather than checkpoint tensor count.
                    chunk_count = min(worker_count, len(operations))
                    chunks = [[] for _ in range(chunk_count)]
                    chunk_bytes = [0] * chunk_count
                    for operation in sorted(
                        operations,
                        key=lambda item: (-tensors[item[0]].numel(), item[0]),
                    ):
                        chunk_index = min(
                            range(chunk_count),
                            key=chunk_bytes.__getitem__,
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
        stats = {
            "operation": "canonical_delta_transform",
            "description": description,
            "delta_tensors": len(names),
            "delta_fragments": len(results),
            "source_files": len(operations_by_source),
            "target_tensor_bytes": sum(value[0] for value in results),
            "compressed_bytes": sum(value[1] for value in results),
            "source_read_worker_s": round(sum(value[2] for value in results), 6),
            "working_memory_budget_bytes": self.working_memory_budget_bytes,
            "workers": worker_count,
            "wall_s": round(time.perf_counter() - started, 6),
        }
        logger.debug(
            "Applied canonical delta group %s: tensors=%d target_bytes=%d "
            "wall_time=%.3fs",
            description,
            stats["delta_tensors"],
            stats["target_tensor_bytes"],
            stats["wall_s"],
        )
        return stats

    def transform_file(self, filename: str, tensor_file: Any) -> dict[str, Any]:
        """Apply and verify one in-memory canonical checkpoint file."""

        names = self.operations_by_file.get(filename, {})
        stats = self.transform_tensors(
            {name: tensor_file.get_tensor_bytes(name) for name in names},
            description=filename,
        )
        stats["filename"] = filename
        return stats

    def close(self) -> None:
        self.operations_by_file.clear()
        self.operations_by_name.clear()

    def __enter__(self) -> DeltaCheckpointTransform:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
