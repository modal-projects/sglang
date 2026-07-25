"""Delta reconstruction for CPU weight preparation.

The disk reload path materializes a complete target checkpoint. CPU weight
preparation instead maintains one node-shared canonical CPU snapshot.
CPU-rich cache initialization populates it from the immutable full checkpoint;
updates then apply only the delta versions after the resident snapshot. Every
changed canonical tensor is verified before the ordinary SGLang loader compiles
the target into rank-ready CPU images.

Compressed delta blobs are prefetched once into a node-shared CPU arena.  Base
checkpoint files are read once, and exactly one local rank owns reconstruction
of each file before publishing it to the other ranks. No runtime-layout or
tensor-sparsity assumption enters this module.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zstandard

from sglang.srt.weight_sync.host_shared_memory import HostSharedMemoryBuffer
from sglang.srt.weight_sync.local_checkpoint import checksum

logger = logging.getLogger(__name__)

_POSITIONAL_IO_CHUNK_BYTES = 64 << 20
_XOR_STREAM_CHUNK_BYTES = 4 << 20


def _version_dir(source_dir: str, version: int) -> Path:
    return Path(source_dir) / f"weight_v{version:06d}"


def _checkpoint_index(root: Path) -> dict[str, Any]:
    path = root / "model.safetensors.index.json"
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"checkpoint index is missing: {path}") from exc
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"invalid checkpoint weight map: {path}")
    metadata = payload.get("metadata") or {}
    if not weight_map and not metadata.get("delta_encoding"):
        raise ValueError(f"full checkpoint has an empty weight map: {path}")
    return payload


def validate_delta_target(source_dir: str, target_version: int) -> None:
    """Reject CPU preparation for base or full-checkpoint targets."""

    if target_version <= 0:
        raise ValueError("CPU weight preparation requires a delta target")
    index = _checkpoint_index(_version_dir(source_dir, target_version))
    metadata = index.get("metadata") or {}
    if not metadata.get("delta_encoding"):
        raise ValueError(
            "CPU weight preparation supports delta targets only; "
            f"weight_v{target_version:06d} is a full checkpoint"
        )


def _pread_file_to_tensor(path: Path, target: torch.Tensor) -> float:
    file_nbytes = path.stat().st_size
    if target.numel() != file_nbytes:
        raise ValueError(
            f"delta arena size mismatch for {path}: "
            f"buffer={target.numel()} file={file_nbytes}"
        )
    started = time.perf_counter()
    view = memoryview(target.numpy()).cast("B")
    fd = os.open(path, os.O_RDONLY)
    offset = 0
    try:
        while offset < file_nbytes:
            end = min(offset + _POSITIONAL_IO_CHUNK_BYTES, file_nbytes)
            nread = os.preadv(fd, [view[offset:end]], offset)
            if nread <= 0:
                raise EOFError(
                    f"unexpected EOF staging delta blob {path}: "
                    f"offset={offset} size={file_nbytes}"
                )
            offset += nread
    finally:
        os.close(fd)
        view.release()
    return time.perf_counter() - started


class HostSharedDeltaBuffer(HostSharedMemoryBuffer):
    """One unlinked shared mapping containing every compressed delta blob."""

    def __init__(self, *, nbytes: int, cpu_group: Any):
        super().__init__(
            nbytes=nbytes,
            cpu_group=cpu_group,
            name="weight-delta",
        )


@dataclass(frozen=True)
class DeltaBlob:
    path: Path
    arena_offset: int
    file_nbytes: int


@dataclass(frozen=True)
class DeltaOperation:
    version: int
    encoding: str
    checksum_algorithm: str
    expected_checksum: str
    arena_offset: int
    compressed_nbytes: int


def _resolve_lineage(
    *,
    base_checkpoint_dir: str,
    current_checkpoint_dir: str | None,
    source_dir: str,
    target_version: int,
    base_version: int = 0,
) -> tuple[Path, int, list[tuple[int, Path, dict[str, Any]]]]:
    """Find the newest full source and the ordered deltas needed for target."""

    if target_version < 0:
        raise ValueError("target_version must be non-negative")
    if base_version < 0:
        raise ValueError("base_version must be non-negative")
    if target_version < base_version:
        raise ValueError(
            f"target version {target_version} precedes resident base "
            f"version {base_version}"
        )
    if base_version:
        if current_checkpoint_dir is None:
            raise ValueError(
                "current_checkpoint_dir is required when advancing a "
                "resident checkpoint"
            )
        checkpoint_root = Path(current_checkpoint_dir)
        _checkpoint_index(checkpoint_root)
        deltas = []
        previous = base_version
        for version in range(base_version + 1, target_version + 1):
            root = _version_dir(source_dir, version)
            index = _checkpoint_index(root)
            metadata = index.get("metadata", {})
            if "delta_encoding" not in metadata:
                # The FULL target itself is never prepared in CPU memory. A
                # later DELTA may use it as its canonical preparation anchor.
                return _resolve_lineage(
                    base_checkpoint_dir=base_checkpoint_dir,
                    current_checkpoint_dir=None,
                    source_dir=source_dir,
                    target_version=target_version,
                    base_version=0,
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
        return checkpoint_root, base_version, deltas
    if target_version == 0:
        root = Path(base_checkpoint_dir)
        _checkpoint_index(root)
        return root, 0, []

    start = target_version
    indexes: dict[int, dict[str, Any]] = {}
    while start > 0:
        root = _version_dir(source_dir, start)
        index = _checkpoint_index(root)
        indexes[start] = index
        if "delta_encoding" not in index.get("metadata", {}):
            break
        start -= 1

    if start == 0:
        checkpoint_root = Path(base_checkpoint_dir)
        _checkpoint_index(checkpoint_root)
    else:
        checkpoint_root = _version_dir(source_dir, start)

    deltas = []
    previous = start
    for version in range(start + 1, target_version + 1):
        root = _version_dir(source_dir, version)
        index = indexes.get(version) or _checkpoint_index(root)
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


class DeltaCheckpointOverlay:
    """Apply a verified delta lineage to a canonical CPU checkpoint."""

    def __init__(
        self,
        *,
        base_checkpoint_dir: str,
        source_dir: str,
        target_version: int,
        cpu_group: Any,
        base_version: int = 0,
        current_checkpoint_dir: str | None = None,
    ):
        started = time.perf_counter()
        self.base_version = base_version
        self.target_version = target_version
        self.cpu_group = cpu_group
        distributed = torch.distributed.is_initialized()
        self.world_size = (
            torch.distributed.get_world_size(group=cpu_group) if distributed else 1
        )
        self.rank = torch.distributed.get_rank(group=cpu_group) if distributed else 0
        self.arena: HostSharedDeltaBuffer | None = None
        self.operations_by_file: dict[str, dict[str, list[DeltaOperation]]] = {}
        self.stage_stats: dict[str, Any] = {
            "operation": "stage_delta_checkpoint",
            "base_version": base_version,
            "target_version": target_version,
            "delta_versions": [],
            "compressed_bytes": 0,
            "wall_s": 0.0,
        }

        checkpoint_root = Path(base_checkpoint_dir)
        deltas = []
        base_weight_map = {}
        blobs: list[DeltaBlob] = []
        blob_indexes: list[tuple[int, dict[str, Any], dict[str, Any], DeltaBlob]] = []
        arena_nbytes = 0
        plan_error = None
        try:
            checkpoint_root, resolved_base_version, deltas = _resolve_lineage(
                base_checkpoint_dir=base_checkpoint_dir,
                current_checkpoint_dir=current_checkpoint_dir,
                source_dir=source_dir,
                target_version=target_version,
                base_version=base_version,
            )
            self.base_version = resolved_base_version
            self.stage_stats["base_version"] = resolved_base_version
            base_weight_map = _checkpoint_index(checkpoint_root)["weight_map"]
            for version, root, index in deltas:
                metadata = index["metadata"]
                self.stage_stats["delta_versions"].append(version)
                for filename in sorted(set(index["weight_map"].values())):
                    path = root / filename
                    try:
                        file_nbytes = path.stat().st_size
                    except FileNotFoundError as exc:
                        raise FileNotFoundError(
                            f"incomplete source version {root}: "
                            f"missing blob {filename}"
                        ) from exc
                    arena_nbytes = (arena_nbytes + 4095) // 4096 * 4096
                    blob = DeltaBlob(
                        path=path,
                        arena_offset=arena_nbytes,
                        file_nbytes=file_nbytes,
                    )
                    blobs.append(blob)
                    blob_indexes.append((version, metadata, index, blob))
                    arena_nbytes += file_nbytes
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
                "failed to resolve delta lineage: " + "; ".join(plan_errors)
            )

        self.checkpoint_root = checkpoint_root
        if not deltas:
            self.stage_stats["wall_s"] = round(time.perf_counter() - started, 6)
            return
        if not blobs:
            self.stage_stats.update(
                {
                    "physical_host_copies": 0,
                    "arena_bytes": 0,
                    "owned_bytes": 0,
                    "owned_read_wall_s": 0.0,
                    "changed_tensors": 0,
                    "wall_s": round(time.perf_counter() - started, 6),
                }
            )
            return

        self.arena = HostSharedDeltaBuffer(
            nbytes=arena_nbytes,
            cpu_group=cpu_group,
        )
        owned_stats = []
        local_error = None
        try:
            for blob_index, blob in enumerate(blobs):
                if blob_index % self.world_size != self.rank:
                    continue
                wall_s = _pread_file_to_tensor(
                    blob.path,
                    self.arena.view(
                        blob.file_nbytes,
                        offset=blob.arena_offset,
                    ),
                )
                owned_stats.append(
                    {
                        "path": str(blob.path),
                        "bytes": blob.file_nbytes,
                        "wall_s": round(wall_s, 6),
                    }
                )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"

        if self.world_size > 1:
            statuses: list[str | None] = [None] * self.world_size
            torch.distributed.all_gather_object(
                statuses,
                local_error,
                group=cpu_group,
            )
        else:
            statuses = [local_error]
        errors = [value for value in statuses if value is not None]
        if errors:
            self.close()
            raise RuntimeError(
                "failed to prefetch complete delta blobs into CPU memory: "
                + "; ".join(errors)
            )

        parse_error = None
        try:
            for version, metadata, index, blob in blob_indexes:
                file_tensor = self.arena.view(
                    blob.file_nbytes,
                    offset=blob.arena_offset,
                )
                if file_tensor.numel() < 8:
                    raise FileNotFoundError(
                        f"incomplete delta blob {blob.path}: "
                        "shorter than header prefix"
                    )
                header_nbytes = int.from_bytes(
                    file_tensor[:8].numpy().tobytes(), "little"
                )
                data_offset = 8 + header_nbytes
                if header_nbytes <= 0 or data_offset > file_tensor.numel():
                    raise FileNotFoundError(
                        f"incomplete delta blob {blob.path}: invalid header "
                        f"length {header_nbytes}"
                    )
                header = json.loads(
                    file_tensor[8:data_offset].numpy().tobytes().decode("utf-8")
                )
                checksums = header.get("__metadata__", {})
                declared_end = 0
                seen_names = set()
                for name, info in header.items():
                    if name == "__metadata__":
                        continue
                    if index["weight_map"].get(name) != blob.path.name:
                        raise ValueError(
                            f"delta blob {blob.path} contains unindexed tensor "
                            f"{name!r}"
                        )
                    seen_names.add(name)
                    offsets = info.get("data_offsets")
                    if (
                        not isinstance(offsets, list)
                        or len(offsets) != 2
                        or not all(isinstance(value, int) for value in offsets)
                    ):
                        raise ValueError(
                            f"invalid delta offsets for {name!r} in {blob.path}"
                        )
                    begin, end = offsets
                    declared_end = max(declared_end, end)
                    if begin < 0 or begin > end or data_offset + end > blob.file_nbytes:
                        raise FileNotFoundError(
                            f"incomplete delta range for {name!r} in {blob.path}"
                        )
                    base_filename = base_weight_map.get(name)
                    if base_filename is None:
                        raise KeyError(
                            f"delta tensor {name!r} is absent from full "
                            f"checkpoint {self.checkpoint_root}"
                        )
                    expected_checksum = checksums.get(name)
                    if not isinstance(expected_checksum, str) or not expected_checksum:
                        raise ValueError(
                            f"delta tensor {name!r} has no target checksum"
                        )
                    operation = DeltaOperation(
                        version=version,
                        encoding=metadata["delta_encoding"],
                        checksum_algorithm=metadata["checksum_format"],
                        expected_checksum=expected_checksum,
                        arena_offset=blob.arena_offset + data_offset + begin,
                        compressed_nbytes=end - begin,
                    )
                    self.operations_by_file.setdefault(base_filename, {}).setdefault(
                        name,
                        [],
                    ).append(operation)
                expected_file_nbytes = data_offset + declared_end
                if expected_file_nbytes != blob.file_nbytes:
                    raise FileNotFoundError(
                        f"incomplete delta blob {blob.path}: file has "
                        f"{blob.file_nbytes} bytes, header declares "
                        f"{expected_file_nbytes}"
                    )
                expected_names = {
                    name
                    for name, filename in index["weight_map"].items()
                    if filename == blob.path.name
                }
                if seen_names != expected_names:
                    raise ValueError(
                        f"delta blob/index tensor mismatch for {blob.path}: "
                        f"missing={sorted(expected_names - seen_names)[:20]} "
                        f"extra={sorted(seen_names - expected_names)[:20]}"
                    )
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

        if self.world_size > 1:
            parse_errors: list[str | None] = [None] * self.world_size
            torch.distributed.all_gather_object(
                parse_errors,
                parse_error,
                group=cpu_group,
            )
        else:
            parse_errors = [parse_error]
        parse_errors = [value for value in parse_errors if value is not None]
        if parse_errors:
            self.close()
            raise RuntimeError(
                "failed to parse staged delta blobs: " + "; ".join(parse_errors)
            )

        for names in self.operations_by_file.values():
            for operations in names.values():
                operations.sort(key=lambda value: value.version)

        self.stage_stats.update(
            {
                "compressed_bytes": sum(blob.file_nbytes for blob in blobs),
                "physical_host_copies": 1,
                "arena_bytes": self.arena.nbytes,
                "owned_bytes": sum(value["bytes"] for value in owned_stats),
                "owned_read_wall_s": round(
                    sum(value["wall_s"] for value in owned_stats), 6
                ),
                "changed_tensors": sum(
                    len(names) for names in self.operations_by_file.values()
                ),
                "wall_s": round(time.perf_counter() - started, 6),
            }
        )
        logger.info(
            "Staged delta versions %s in CPU memory: compressed_bytes=%d "
            "wall_time=%.3fs",
            self.stage_stats["delta_versions"],
            self.stage_stats["compressed_bytes"],
            self.stage_stats["wall_s"],
        )

    def _compressed_view(self, operation: DeltaOperation) -> memoryview:
        if self.arena is None:
            raise RuntimeError("staged delta buffer is not available")
        tensor = self.arena.view(
            operation.compressed_nbytes,
            offset=operation.arena_offset,
        )
        return memoryview(tensor.numpy()).cast("B")

    def transform_file(self, filename: str, tensor_file: Any) -> dict[str, Any]:
        """Apply and verify this file's canonical target bytes in place."""

        names = self.operations_by_file.get(filename)
        if not names:
            return {
                "operation": "canonical_delta_overlay",
                "filename": filename,
                "changed_tensors": 0,
                "target_tensor_bytes": 0,
                "compressed_bytes": 0,
                "wall_s": 0.0,
            }

        started = time.perf_counter()
        worker_count = min(
            8,
            max(1, (os.cpu_count() or self.world_size) // self.world_size),
            len(names),
        )

        def apply_tensor(item: tuple[str, list[DeltaOperation]]) -> tuple[int, int]:
            name, operations = item
            region_tensor = tensor_file.get_tensor_bytes(name)
            region = region_tensor.numpy()
            for operation in operations:
                compressed = self._compressed_view(operation)
                try:
                    decompressor = zstandard.ZstdDecompressor()
                    if operation.encoding == "xor":
                        with decompressor.stream_reader(compressed) as reader:
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
                                np.bitwise_xor(
                                    region[position : position + delta.size],
                                    delta,
                                    out=region[position : position + delta.size],
                                )
                                position += delta.size
                            if position != region.size or reader.read(1):
                                raise RuntimeError(
                                    f"decompressed XOR size mismatch for {name!r}: "
                                    f"expected={region.size} actual={position}"
                                )
                    else:
                        payload = decompressor.decompress(compressed)
                        if len(payload) < 4:
                            raise RuntimeError(
                                f"overwrite delta for {name!r} is truncated"
                            )
                        count = int.from_bytes(payload[:4], "little")
                        positions_end = 4 + 4 * count
                        if positions_end > len(payload):
                            raise RuntimeError(
                                f"overwrite positions for {name!r} are truncated"
                            )
                        positions = np.frombuffer(
                            payload[4:positions_end],
                            dtype="<u4",
                        )
                        values = np.frombuffer(payload[positions_end:], dtype=np.uint8)
                        if values.size != count or (
                            count and int(positions.max()) >= region.size
                        ):
                            raise RuntimeError(
                                f"overwrite payload for {name!r} is invalid"
                            )
                        region[positions] = values
                finally:
                    compressed.release()

            final = operations[-1]
            actual = checksum(final.checksum_algorithm, region)
            if actual != final.expected_checksum:
                raise RuntimeError(
                    f"checksum mismatch after reconstructing {name!r}: "
                    f"expected={final.expected_checksum} actual={actual}"
                )
            return region.size, sum(
                operation.compressed_nbytes for operation in operations
            )

        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="weight-delta-xor",
        ) as pool:
            sizes = list(pool.map(apply_tensor, names.items()))
        stats = {
            "operation": "canonical_delta_overlay",
            "filename": filename,
            "changed_tensors": len(names),
            "target_tensor_bytes": sum(value[0] for value in sizes),
            "compressed_bytes": sum(value[1] for value in sizes),
            "workers": worker_count,
            "wall_s": round(time.perf_counter() - started, 6),
        }
        logger.debug(
            "Applied canonical delta file %s: tensors=%d target_bytes=%d "
            "wall_time=%.3fs",
            filename,
            stats["changed_tensors"],
            stats["target_tensor_bytes"],
            stats["wall_s"],
        )
        return stats

    def close(self) -> None:
        self.operations_by_file.clear()
        arena = getattr(self, "arena", None)
        if arena is not None:
            arena.close()
            self.arena = None

    def __enter__(self) -> DeltaCheckpointOverlay:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
