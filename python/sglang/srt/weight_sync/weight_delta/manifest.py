from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import orjson

FORMAT = "weight-delta/v1"
# snappy-zstd: changed frames Snappy-compressed; gap-zstd: their changed bytes (gap.py).
CODECS = ("snappy-zstd", "gap-zstd")


@dataclass(frozen=True)
class Manifest:
    """One version's delta: tensor entries (encoding xor_bytes or raw_bytes, with their
    payload regions when changed) and the payload files they live in."""

    directory: Path
    stream_id: str
    base_version: int
    target_version: int
    codec: str
    frame_bytes: int
    files: dict[str, dict]
    tensors: dict[str, dict]


def read_manifest(path: str) -> Manifest:
    data = orjson.loads(Path(path).read_bytes())
    if data.get("format") != FORMAT or data.get("codec") not in CODECS:
        raise ValueError(f"{path}: expected {FORMAT} with a codec of {CODECS}")
    return Manifest(
        directory=Path(path).parent,
        stream_id=data["stream_id"],
        base_version=data["base_version"],
        target_version=data["target_version"],
        codec=data["codec"],
        frame_bytes=data["frame_bytes"],
        files={f["name"]: f for f in data["files"]},
        tensors={t["name"]: t for t in data["tensors"]},
    )
