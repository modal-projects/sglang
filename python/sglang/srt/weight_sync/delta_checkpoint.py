"""Validation and indexing for versioned compressed checkpoint deltas."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sglang.srt.weight_sync.checksum import create_checksum, validate_checksum
from sglang.srt.weight_sync.safetensors_buffer import (
    read_safetensors_file_layout,
)


@dataclass(frozen=True)
class DeltaTensor:
    name: str
    source_path: Path
    source_offset: int
    compressed_nbytes: int
    checksum_algorithm: str
    expected_checksum: str


@dataclass(frozen=True)
class DeltaCheckpoint:
    version: int
    base_version: int
    encoding: str
    tensors: tuple[DeltaTensor, ...]
    metadata_wall_s: float
    source_setup_wall_s: float


def version_dir(checkpoint_source_dir: str | Path, version: int) -> Path:
    return Path(checkpoint_source_dir) / f"weight_v{version:06d}"


def read_delta_checkpoint(
    root: str | Path,
    *,
    expected_version: int,
    expected_base_version: int,
) -> DeltaCheckpoint:
    """Validate one published delta and index every compressed tensor range."""

    root = Path(root)
    metadata_started = time.perf_counter()
    index_path = root / "model.safetensors.index.json"
    try:
        index: Any = json.loads(index_path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"checkpoint index is missing: {index_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid checkpoint index: {index_path}") from exc
    if not isinstance(index, dict):
        raise ValueError(f"invalid checkpoint index: {index_path}")
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict):
        raise ValueError(f"invalid checkpoint index: {index_path}")
    if not all(
        isinstance(name, str) and name and isinstance(filename, str) and filename
        for name, filename in weight_map.items()
    ):
        raise ValueError(f"invalid checkpoint weight map: {index_path}")

    try:
        version = int(metadata["version"])
        base_version = int(metadata["base_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"checkpoint has no valid delta lineage: {index_path}"
        ) from exc
    if version != expected_version:
        raise ValueError(
            f"checkpoint version mismatch: directory is v{expected_version}, "
            f"index declares v{version}"
        )
    if base_version != expected_base_version:
        raise ValueError(
            f"delta v{expected_version} builds on v{base_version}, "
            f"expected v{expected_base_version}"
        )

    encoding = metadata.get("delta_encoding")
    if encoding is None:
        raise ValueError(f"weight_v{expected_version:06d} is a full checkpoint")
    if encoding not in {"xor", "overwrite"}:
        raise NotImplementedError(
            f"delta v{expected_version} encoding {encoding!r} is unsupported"
        )
    compression = metadata.get("compression_format")
    if compression != "zstd":
        raise NotImplementedError(
            f"delta v{expected_version} compression {compression!r} is unsupported"
        )
    checksum_algorithm = metadata.get("checksum_format")
    if not isinstance(checksum_algorithm, str):
        raise ValueError(f"delta v{expected_version} has no checksum format")
    create_checksum(checksum_algorithm)
    metadata_wall_s = time.perf_counter() - metadata_started

    source_started = time.perf_counter()
    tensors: list[DeltaTensor] = []
    seen_names: set[str] = set()
    for filename in sorted(set(weight_map.values())):
        source_path = _resolve_source_path(root, filename)
        layout, checksums = _read_delta_header(source_path)
        expected_names = {
            name
            for name, mapped_filename in weight_map.items()
            if mapped_filename == filename
        }
        actual_names = set(layout.tensors)
        if actual_names != expected_names:
            raise ValueError(
                f"delta blob/index tensor mismatch for {source_path}: "
                f"missing={sorted(expected_names - actual_names)[:20]} "
                f"extra={sorted(actual_names - expected_names)[:20]}"
            )
        checksum_names = set(checksums)
        if checksum_names != expected_names:
            raise ValueError(
                f"delta checksum/index tensor mismatch for {source_path}: "
                f"missing={sorted(expected_names - checksum_names)[:20]} "
                f"extra={sorted(checksum_names - expected_names)[:20]}"
            )
        for name in sorted(layout.tensors):
            if name in seen_names:
                raise ValueError(f"duplicate delta tensor {name!r}")
            seen_names.add(name)
            entry = layout.tensors[name]
            tensors.append(
                DeltaTensor(
                    name=name,
                    source_path=source_path,
                    source_offset=layout.data_offset + entry.relative_begin,
                    compressed_nbytes=entry.relative_end - entry.relative_begin,
                    checksum_algorithm=checksum_algorithm,
                    expected_checksum=validate_checksum(
                        checksum_algorithm, checksums.get(name)
                    ),
                )
            )

    return DeltaCheckpoint(
        version=version,
        base_version=base_version,
        encoding=encoding,
        tensors=tuple(tensors),
        metadata_wall_s=metadata_wall_s,
        source_setup_wall_s=time.perf_counter() - source_started,
    )


def _resolve_source_path(root: Path, filename: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid delta shard path: {filename!r}")
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if not path.is_relative_to(resolved_root):
        raise ValueError(f"invalid delta shard path: {filename!r}")
    if not path.exists():
        raise FileNotFoundError(
            f"incomplete source version {root}: missing blob {filename}"
        )
    return path


def _read_delta_header(path: Path):
    layout, metadata = read_safetensors_file_layout(path)
    for name, entry in layout.tensors.items():
        if entry.dtype_code != "U8" or entry.shape != (
            entry.relative_end - entry.relative_begin,
        ):
            raise ValueError(
                f"compressed delta tensor {name!r} must be one-dimensional U8"
            )
    return layout, metadata
