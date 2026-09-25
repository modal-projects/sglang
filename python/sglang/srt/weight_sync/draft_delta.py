"""Wire format and CPU codec for fixed-shape, draft-only weight updates.

The payload is the adaptive-spec v1 export: independent XOR/Zstandard frames
over canonical (unsharded, unpacked) serving tensors. No pickle or client paths
cross the HTTP boundary. Identity is independent of safetensors file layout.
"""

import hashlib
import json
import math
import struct
import sys

import numpy as np
import torch
from transformers import Qwen3Config

FORMAT = "adaptive-spec-draft-export"
HEADER = struct.Struct("<8sQ")
MAGIC = b"SGLDD001"
MAX_MANIFEST_BYTES = 8 * 2**20
MAX_DELTA_BYTES = 4 * 2**30
MAX_WEIGHT_BYTES = 4 * 2**30
DTYPES = {"bfloat16": 2, "float16": 2, "float32": 4}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def config_hash(config):
    value = Qwen3Config.from_dict(config).to_dict()
    for key in ("_name_or_path", "_commit_hash", "transformers_version"):
        value.pop(key, None)
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def weights_id(config_sha256, tensors):
    return hashlib.sha256(
        b"adaptive-spec-draft-v1\0"
        + config_sha256.encode()
        + canonical_json(tensors).encode()
    ).hexdigest()


def tensor_bytes(tensor):
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("expected contiguous CPU weights")
    return tensor.detach().reshape(-1).view(torch.uint8).numpy()


def tensor_info(name, tensor):
    return {
        "name": name,
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "nbytes": tensor.numel() * tensor.element_size(),
        "sha256": hashlib.sha256(tensor_bytes(tensor)).hexdigest(),
    }


def _hash(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def validate_manifest(manifest):
    """Bound allocations and validate every descriptor before reading frames."""
    if not isinstance(manifest, dict) or (
        manifest.get("format"),
        manifest.get("version"),
        sys.byteorder,
    ) != (FORMAT, 1, "little"):
        raise ValueError("expected a little-endian adaptive-spec draft export v1")
    if not _hash(manifest.get("config_sha256")) or not _hash(
        manifest.get("weights_id")
    ):
        raise ValueError("invalid draft config/result identity")
    delta = manifest.get("delta")
    if not isinstance(delta, dict) or (
        delta.get("codec"),
        delta.get("byte_order"),
        delta.get("file"),
    ) != ("xor-zstd", "little", "model.delta.zst"):
        raise ValueError("expected XOR/Zstandard draft delta")
    if not _hash(delta.get("base_weights_id")):
        raise ValueError("invalid delta base identity")
    entries = delta.get("tensors")
    if not isinstance(entries, list) or not entries or len(entries) > 16384:
        raise ValueError("invalid delta tensor list")
    offset, weight_bytes, infos, names = 0, 0, [], []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid delta tensor descriptor")
        name, shape, dtype = entry.get("name"), entry.get("shape"), entry.get("dtype")
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 1024
            or any(part in ("lm_head", "embed_tokens") for part in name.split("."))
        ):
            raise ValueError("invalid or target-owned tensor name")
        if (
            not isinstance(dtype, str)
            or dtype not in DTYPES
            or not isinstance(shape, list)
            or not shape
            or len(shape) > 8
            or any(type(n) is not int or n <= 0 for n in shape)
        ):
            raise ValueError(f"invalid dtype/shape: {name}")
        nbytes = math.prod(shape) * DTYPES[dtype]
        if type(entry.get("nbytes")) is not int or entry["nbytes"] != nbytes:
            raise ValueError(f"invalid tensor byte count: {name}")
        if not _hash(entry.get("sha256")) or not _hash(entry.get("base_sha256")):
            raise ValueError(f"invalid tensor checksum: {name}")
        length = entry.get("length")
        if (
            type(entry.get("offset")) is not int
            or entry["offset"] != offset
            or type(length) is not int
            or length < 0
            or length > nbytes + max(1024 * 1024, nbytes // 128)
            or (length == 0 and entry["sha256"] != entry["base_sha256"])
        ):
            raise ValueError(f"invalid delta frame bounds: {name}")
        offset += length
        weight_bytes += nbytes
        if offset > MAX_DELTA_BYTES or weight_bytes > MAX_WEIGHT_BYTES:
            raise ValueError("draft delta exceeds the 4 GiB payload/weight limit")
        names.append(name)
        infos.append(
            {k: entry[k] for k in ("name", "dtype", "shape", "nbytes", "sha256")}
        )
    if names != sorted(set(names)):
        raise ValueError("delta tensors must have unique, sorted canonical names")
    if (
        type(delta.get("compressed_bytes")) is not int
        or offset != delta["compressed_bytes"]
        or type(manifest.get("weight_bytes")) is not int
        or weight_bytes != manifest["weight_bytes"]
        or infos != manifest.get("tensors")
    ):
        raise ValueError("delta manifest tensor/payload size mismatch")
    if weights_id(manifest["config_sha256"], infos) != manifest["weights_id"]:
        raise ValueError("delta result identity mismatch")
    base_infos = [
        {**info, "sha256": entry["base_sha256"]}
        for info, entry in zip(infos, entries, strict=True)
    ]
    if weights_id(manifest["config_sha256"], base_infos) != delta["base_weights_id"]:
        raise ValueError("delta base identity mismatch")
    return entries


def decode_tensor(base, entry, compressed):
    import zstandard as zstd

    before = tensor_info(entry["name"], base)
    for key in ("dtype", "shape", "nbytes"):
        if before[key] != entry[key]:
            raise ValueError(f"delta base schema mismatch: {entry['name']}")
    if before["sha256"] != entry["base_sha256"]:
        raise ValueError(f"delta base checksum mismatch: {entry['name']}")
    result = base.clone()
    if entry["length"]:
        frame = zstd.get_frame_parameters(compressed)
        if frame.content_size != entry["nbytes"] or frame.window_size > max(
            entry["nbytes"], 1024 * 1024
        ):
            raise ValueError(f"invalid delta frame output size: {entry['name']}")
        raw_delta = zstd.ZstdDecompressor().decompress(
            compressed, max_output_size=entry["nbytes"], allow_extra_data=False
        )
        raw = tensor_bytes(result)
        np.bitwise_xor(raw, np.frombuffer(raw_delta, dtype=np.uint8), out=raw)
    if tensor_info(entry["name"], result)["sha256"] != entry["sha256"]:
        raise ValueError(f"delta result checksum mismatch: {entry['name']}")
    if not torch.isfinite(result).all():
        raise ValueError(f"nonfinite draft weights: {entry['name']}")
    return result


def encode_upload_header(manifest, expected_draft_version):
    """Follow this header with model.delta.zst bytes in an HTTP POST body."""
    validate_manifest(manifest)
    if type(expected_draft_version) is not int or expected_draft_version < 0:
        raise ValueError("expected_draft_version must be a nonnegative integer")
    metadata = canonical_json(
        {"manifest": manifest, "expected_draft_version": expected_draft_version}
    ).encode()
    if len(metadata) > MAX_MANIFEST_BYTES:
        raise ValueError("oversized draft delta manifest")
    return HEADER.pack(MAGIC, len(metadata)) + metadata
