"""Read a weight-delta version Miles writes: manifest.json, then each owner file's columns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import orjson
from safetensors import safe_open

FORMAT = "weight-delta/v1"
CODECS = ("gap-zstd", "snappy-zstd")
_REGION_FIELDS = ("offset", "nbytes", "inner_offset", "inner_nbytes")
_COLUMNS = (
    "tensors.name",
    "tensors.name_end",
    "tensors.dtype",
    "tensors.shape",
    "tensors.shape_end",
    "tensors.raw",
    "tensors.offset",
    "tensors.nbytes",
    "tensors.inner_nbytes",
    "tensors.chunk_end",
    "tensors.frame_end",
    "tensors.checksum",
    *(f"{table}.{field}" for table in ("chunks", "frames") for field in _REGION_FIELDS),
)
# Each ragged column's row ends, and the column they end into.
_ENDS = {
    "tensors.name_end": "tensors.name",
    "tensors.shape_end": "tensors.shape",
    "tensors.chunk_end": "chunks.offset",
    "tensors.frame_end": "frames.offset",
}


@dataclass(frozen=True)
class Manifest:
    directory: Path
    codec: str
    frame_nbytes: int
    base_version: int
    target_version: int
    files: dict[str, int]  # owner file name -> size in bytes


@dataclass(frozen=True)
class Tensors:
    """Every changed tensor of a version, one row each, the owner files' rows back to back. Row i's
    dims, chunks and frames are rows [end[i - 1], end[i]) of theirs. A chunk's offset is in the
    owner file; a frame's offset is in the tensor and its inner_offset in the tensor's inner bytes."""

    names: np.ndarray  # fixed-width bytes
    dtypes: np.ndarray  # safetensors dtype names
    shape: np.ndarray
    shape_end: np.ndarray
    raw: np.ndarray
    file: np.ndarray  # index into Manifest.files
    offset: np.ndarray  # in the owner file
    nbytes: np.ndarray
    inner_nbytes: np.ndarray
    chunk_end: np.ndarray
    frame_end: np.ndarray
    # Of the tensor's new bytes where they changed, every byte for a raw one (Miles'
    # publication.checksum); the engine compares it after applying.
    checksum: np.ndarray
    chunks: dict[str, np.ndarray]
    frames: dict[str, np.ndarray]


def read_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    data = orjson.loads(path.read_bytes())
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: format {data.get('format')!r}, expected {FORMAT}")
    if data.get("codec") not in CODECS:
        raise ValueError(f"{path}: codec {data.get('codec')!r} is not one of {CODECS}")
    try:
        return Manifest(
            directory=path.parent,
            codec=data["codec"],
            frame_nbytes=data["frame_nbytes"],
            base_version=data["base_version"],
            target_version=data["target_version"],
            files={file["name"]: file["nbytes"] for file in data["files"]},
        )
    except KeyError as missing:
        raise ValueError(f"{path}: no {missing} field") from None


def read_tensors(manifest: Manifest) -> Tensors:
    owners = [
        _read_owner_file(manifest.directory / name, nbytes)
        for name, nbytes in manifest.files.items()
    ] or [_no_rows()]
    column = {
        key: np.concatenate([owner[key] for owner in owners]) for key in owners[0]
    }
    ends = {
        key: np.cumsum(
            np.concatenate([np.diff(owner[key], prepend=0) for owner in owners])
        )
        for key in _ENDS
    }
    file = np.repeat(
        np.arange(len(owners)), [len(owner["tensors.offset"]) for owner in owners]
    )
    offset = column["tensors.offset"] + column["payload_offset"][file]
    chunks = {field: column[f"chunks.{field}"] for field in _REGION_FIELDS}
    chunks["offset"] = chunks["offset"] + np.repeat(
        offset, np.diff(ends["tensors.chunk_end"], prepend=0)
    )
    return Tensors(
        names=_fixed_width(column["tensors.name"], ends["tensors.name_end"]),
        dtypes=column["tensors.dtype"],
        shape=column["tensors.shape"],
        shape_end=ends["tensors.shape_end"],
        raw=column["tensors.raw"],
        file=file,
        offset=offset,
        nbytes=column["tensors.nbytes"],
        inner_nbytes=column["tensors.inner_nbytes"],
        chunk_end=ends["tensors.chunk_end"],
        frame_end=ends["tensors.frame_end"],
        checksum=column["tensors.checksum"],
        chunks=chunks,
        frames={field: column[f"frames.{field}"] for field in _REGION_FIELDS},
    )


def _read_owner_file(path: Path, nbytes: int) -> dict[str, np.ndarray]:
    """The file's columns, with dtype codes as names and where its payload starts."""
    if path.stat().st_size != nbytes:
        raise ValueError(
            f"{path}: {path.stat().st_size} bytes, the manifest says {nbytes}"
        )
    with safe_open(path, framework="numpy") as owner:
        missing = sorted(set(_COLUMNS) - set(owner.keys()))
        dtypes = (owner.metadata() or {}).get("dtypes")
        if missing or dtypes is None:
            raise ValueError(f"{path}: no {missing or 'dtypes metadata'}")
        column = {key: owner.get_tensor(key) for key in _COLUMNS}
    with path.open("rb") as source:
        header_nbytes = int.from_bytes(source.read(8), "little")
        payload_start, payload_end = orjson.loads(source.read(header_nbytes))[
            "payload"
        ]["data_offsets"]
    dtypes = np.array(dtypes.split(","))
    for end, values in _ENDS.items():
        if (column[end][-1] if len(column[end]) else 0) != len(column[values]):
            raise ValueError(f"{path}: {end} does not end at {values}'s length")
    if np.any(column["tensors.dtype"] >= len(dtypes)):
        raise ValueError(f"{path}: a dtype code past its {len(dtypes)} dtypes")
    regions_end = column["tensors.offset"] + column["tensors.nbytes"]
    if np.any(regions_end > payload_end - payload_start):
        raise ValueError(f"{path}: a tensor's bytes lie past the payload")
    column["tensors.dtype"] = dtypes[column["tensors.dtype"]]
    column["payload_offset"] = np.array([8 + header_nbytes + payload_start])
    return column


def _no_rows() -> dict[str, np.ndarray]:
    column: dict[str, np.ndarray] = {
        key: np.zeros(0, dtype=np.int64) for key in _COLUMNS
    }
    column["tensors.name"] = np.zeros(0, dtype=np.uint8)
    column["tensors.dtype"] = np.zeros(0, dtype=str)
    column["tensors.raw"] = np.zeros(0, dtype=bool)
    column["tensors.checksum"] = np.zeros(0, dtype=np.uint64)
    column["payload_offset"] = np.zeros(1, dtype=np.int64)
    return column


def _fixed_width(blob: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Byte strings given back to back, as one fixed-width bytes array."""
    lengths = np.diff(ends, prepend=0)
    width = max(int(lengths.max(initial=0)), 1)
    matrix = np.zeros((len(ends), width), dtype=np.uint8)
    rows = np.repeat(np.arange(len(ends)), lengths)
    matrix[rows, np.arange(len(blob)) - np.repeat(ends - lengths, lengths)] = blob
    return matrix.view(f"S{width}").ravel()
