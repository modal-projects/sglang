"""Weight-delta versions as Miles writes them, for the weight-delta tests."""

import json
from pathlib import Path

import numpy as np
import zstandard
from safetensors.numpy import save_file

FRAME_NBYTES = 1 << 20


def checksum(values: bytes, counted: np.ndarray | None = None) -> int:
    """Miles' publication.checksum: over each 4-byte word w of the bytes (the last one
    zero-padded), the sum mod 2^64 of (murmur3's 32-bit finalizer of w mod 2^32, made odd) x the
    word with the bytes outside `counted` zeroed; every byte counts without it."""
    data = np.frombuffer(values, dtype=np.uint8)
    if counted is not None:
        data = np.where(counted, data, np.uint8(0))
    words = np.zeros(-(-data.size // 4), dtype="<u4")
    words.view(np.uint8)[: data.size] = data
    h = (np.arange(words.size) % (1 << 32)).astype(np.uint32)
    h ^= h >> np.uint32(16)
    h *= np.uint32(0x85EBCA6B)
    h ^= h >> np.uint32(13)
    h *= np.uint32(0xC2B2AE35)
    h ^= h >> np.uint32(16)
    h |= np.uint32(1)
    return int((h.astype(np.uint64) * words.astype(np.uint64)).sum(dtype=np.uint64))


def region_checksum(
    previous: bytes | None, current: bytes, itemsize: int, shape: tuple, region: tuple
) -> int:
    """What a GPU holding the canonical `region` computes: the bytes there that changed from
    `previous`, or all of them for a raw tensor (no previous)."""
    counted = np.zeros((*shape, itemsize), dtype=bool)
    counted[region] = True
    counted = counted.reshape(-1)
    if previous is not None:
        counted &= np.frombuffer(previous, np.uint8) != np.frombuffer(current, np.uint8)
    return checksum(current, counted)


def raw_tensor(
    name: str, dtype: str, shape: tuple, data: bytes, expected: int | None = None
) -> dict:
    """expected: the checksum the engine computes, by default that of every byte."""
    return {
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "raw": data,
        "checksum": checksum(data) if expected is None else expected,
    }


def xor_tensor(
    name: str,
    dtype: str,
    shape: tuple,
    frames: list[dict],
    inner: bytes,
    chunk_nbytes: int = 1 << 20,
    expected: int = 0,
) -> dict:
    """A changed tensor's frames and inner bytes, cut into outer Zstd chunks of chunk_nbytes."""
    pieces = [
        inner[at : at + chunk_nbytes] for at in range(0, len(inner), chunk_nbytes)
    ]
    return {
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "frames": frames,
        "pieces": pieces,
        "checksum": expected,
    }


def gap_frames(previous: bytes, current: bytes) -> tuple[list[dict], bytes]:
    """Codec gap: each changed frame's u24 gaps in three byte planes, then the XOR values."""

    def encode(frame: np.ndarray) -> bytes:
        positions = np.flatnonzero(frame)
        gaps = np.diff(positions, prepend=0).astype(np.uint32)
        planes = [((gaps >> (8 * plane)) & 0xFF).astype(np.uint8) for plane in range(3)]
        return (
            b"".join(plane.tobytes() for plane in planes) + frame[positions].tobytes()
        )

    return _frames(previous, current, encode)


def snappy_frames(previous: bytes, current: bytes) -> tuple[list[dict], bytes]:
    """Codec Snappy: each changed frame's XOR as one raw Snappy block."""
    import pyarrow

    codec = pyarrow.Codec("snappy")
    return _frames(
        previous, current, lambda frame: codec.compress(frame.tobytes(), asbytes=True)
    )


def write_version(
    directory: Path,
    owners: list[list[dict]],
    base_version: int,
    codec: str = "gap-zstd",
) -> Path:
    """One owner file per list of tensors, then the manifest; returns the manifest's path."""
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for index, tensors in enumerate(owners):
        path = directory / f"owner-{index:05d}.safetensors"
        files.append({"name": path.name, "nbytes": write_owner_file(path, tensors)})
    manifest = {
        "format": "weight-delta/v1",
        "codec": codec,
        "frame_nbytes": FRAME_NBYTES,
        "base_version": base_version,
        "target_version": base_version + 1,
        "files": files,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory / "manifest.json"


def write_owner_file(path: Path, tensors: list[dict]) -> int:
    """An owner file as Miles writes it: payload, then where each tensor's bytes lie."""
    payload, rows, chunks, frames = bytearray(), [], [], []
    dtypes = list(dict.fromkeys(tensor["dtype"] for tensor in tensors))
    for tensor in tensors:
        payload += bytes(-len(payload) % 16)
        offset, inner_offset = len(payload), 0
        for piece in tensor.get("pieces", []):
            frame = zstandard.ZstdCompressor().compress(piece)
            chunks.append((len(payload) - offset, len(frame), inner_offset, len(piece)))
            payload += frame
            inner_offset += len(piece)
        payload += tensor.get("raw", b"")
        frames += [tuple(frame.values()) for frame in tensor.get("frames", [])]
        rows.append((tensor, offset, len(payload) - offset, inner_offset))
        rows[-1] += (len(chunks), len(frames))
    names = [tensor["name"].encode() for tensor, *_ in rows]

    def column(values) -> np.ndarray:
        return np.array(values, dtype=np.int64).reshape(-1)

    columns = {
        "tensors.name": np.frombuffer(b"".join(names), dtype=np.uint8),
        "tensors.name_end": np.cumsum([len(name) for name in names]),
        "tensors.dtype": np.array(
            [dtypes.index(tensor["dtype"]) for tensor, *_ in rows], dtype=np.uint8
        ),
        "tensors.shape": column(
            [dim for tensor, *_ in rows for dim in tensor["shape"]]
        ),
        "tensors.shape_end": np.cumsum([len(tensor["shape"]) for tensor, *_ in rows]),
        "tensors.raw": np.array(["raw" in tensor for tensor, *_ in rows]),
        "tensors.offset": column([row[1] for row in rows]),
        "tensors.nbytes": column([row[2] for row in rows]),
        "tensors.inner_nbytes": column([row[3] for row in rows]),
        "tensors.chunk_end": column([row[4] for row in rows]),
        "tensors.frame_end": column([row[5] for row in rows]),
        "tensors.checksum": np.array(
            [tensor["checksum"] for tensor, *_ in rows], dtype=np.uint64
        ),
        "payload": np.frombuffer(bytes(payload), dtype=np.uint8),
    }
    fields = ("offset", "nbytes", "inner_offset", "inner_nbytes")
    for table, records in (("chunks", chunks), ("frames", frames)):
        for index, field in enumerate(fields):
            columns[f"{table}.{field}"] = column([record[index] for record in records])
    save_file(columns, str(path), metadata={"dtypes": ",".join(dtypes)})
    return path.stat().st_size


def _frames(previous: bytes, current: bytes, encode) -> tuple[list[dict], bytes]:
    xor = np.frombuffer(previous, dtype=np.uint8) ^ np.frombuffer(
        current, dtype=np.uint8
    )
    frames, inner = [], bytearray()
    for start in range(0, len(xor), FRAME_NBYTES):
        frame = xor[start : start + FRAME_NBYTES]
        if not frame.any():
            continue
        encoded = encode(frame)
        inner += bytes(-len(inner) % 16)
        frames.append(
            {
                "offset": start,
                "nbytes": len(frame),
                "inner_offset": len(inner),
                "inner_nbytes": len(encoded),
            }
        )
        inner += encoded
    return frames, bytes(inner)
