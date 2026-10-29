"""Weight-delta versions as Miles writes them, for the weight-delta tests."""

import json
from pathlib import Path

import numpy as np
import zstandard
from safetensors.numpy import save_file

FRAME_NBYTES = 1 << 20


def raw_tensor(name: str, dtype: str, shape: tuple, data: bytes) -> dict:
    return {"name": name, "dtype": dtype, "shape": shape, "raw": data}


def xor_tensor(
    name: str,
    dtype: str,
    shape: tuple,
    frames: list[dict],
    inner: bytes,
    chunk_nbytes: int = 1 << 20,
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
