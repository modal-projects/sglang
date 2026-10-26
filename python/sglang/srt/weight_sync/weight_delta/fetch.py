"""Before the pause, while the engine serves: read this GPU's part of deltas N+1..M from
their files, undo the outer Zstd on CPU into pinned host memory, put the raw values on the
GPU, and plan every decode and XOR, so the pause only binds the decode slots (and, without a
decompression engine, the staging buffers) and launches."""

from __future__ import annotations

import hashlib
import mmap
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch

from sglang.srt.weight_sync.weight_delta.manifest import Manifest, read_manifest
from sglang.srt.weight_sync.weight_delta.map import TORCH_DTYPES, DeltaMap, Place
from sglang.srt.weight_sync.weight_delta.memory import HostArena
from sglang.srt.weight_sync.weight_delta.nvcomp import DecodePlan, SnappyDecoder
from sglang.srt.weight_sync.weight_delta.xor_kernel import (
    ByteApplyGroup,
    plan_xor,
    prepare_status_check,
)

# Decoded tensors and host arenas start on this boundary, which covers nvCOMP's and the
# XOR kernel's alignment needs.
ALIGN = 256
# Scratch per batch, so each of the two decode slots in the pause holds about this much.
BATCH_BYTES = 2 << 30
# The outer Zstd is undone in groups of about this much output, then copied into the
# arena: decoding straight into pinned memory is several times slower.
ZSTD_GROUP_BYTES = 1 << 30


@dataclass
class Batch:
    """Consecutive changed tensors of one version: decode them into scratch, then XOR."""

    target_version: int
    tensors: list[str]
    offsets: list[int]  # where each tensor's decoded bytes start in scratch
    scratch_bytes: int
    zero_ranges: list[
        tuple[int, int]
    ]  # scratch bytes no frame writes: unchanged frames
    decode: DecodePlan
    status_check: object  # sets the error flag if a frame failed; the XOR then skips
    xor: ByteApplyGroup
    xor_table: np.ndarray  # pointer rows (sources as scratch offsets), then metadata


@dataclass
class Fetched:
    base_version: int
    target_version: int
    batches: list[Batch]
    raw: list[
        tuple[torch.Tensor, tuple[Place, ...]]
    ]  # newest value, already on the GPU; its places
    scratch_bytes: int
    staging_bytes: int  # largest batch's frames; 0 when the decompression engine reads host
    error: torch.Tensor
    plans: dict  # XOR plans by batch membership, reused by the next prepare
    stats: dict = field(default_factory=dict)


def _aligned(size: int) -> int:
    return -(-size // ALIGN) * ALIGN


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape or (1,)).view(torch.uint8)


def _check_chain(manifests: list[Manifest], version: str) -> None:
    if str(manifests[0].base_version) != str(version):
        raise ValueError(
            f"the first delta starts at version {manifests[0].base_version}, "
            f"the engine is at {version}"
        )
    for previous, current in zip(manifests, manifests[1:]):
        if (
            current.stream_id != previous.stream_id
            or current.base_version != previous.target_version
        ):
            raise ValueError(
                f"delta {current.base_version}->{current.target_version} does not follow "
                f"{previous.base_version}->{previous.target_version}"
            )


def _check_files(manifest: Manifest, verify_sha256: bool) -> None:
    for name, record in manifest.files.items():
        path = manifest.directory / name
        if path.stat().st_size != record["nbytes"]:
            raise ValueError(f"{path}: size differs from the manifest")
        if verify_sha256:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(64 << 20):
                    digest.update(chunk)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError(f"{path}: SHA256 differs from the manifest")


def _map_files(manifest: Manifest) -> dict[str, memoryview]:
    """The payload files, memory-mapped: regions are slices, with no read per tensor."""
    views = {}
    for name, record in manifest.files.items():
        if not record["nbytes"]:
            views[name] = memoryview(b"")
            continue
        with (manifest.directory / name).open("rb") as source:
            views[name] = memoryview(
                mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
            )
    return views


def _undo_outer_zstd(
    chunks, host: np.ndarray, pool: ThreadPoolExecutor, workers: int
) -> None:
    """chunks: (encoded bytes, arena offset, decoded size). Decoded in C threads, a group
    at a time; every chunk must decode to its declared size."""
    import zstandard

    decompressor = zstandard.ZstdDecompressor()
    group, size = [], 0
    for chunk in [*chunks, None]:
        if chunk is not None and (not group or size + chunk[2] <= ZSTD_GROUP_BYTES):
            group.append(chunk)
            size += chunk[2]
            continue
        if group:
            sizes = np.array([decoded for _, _, decoded in group], dtype=np.uint64)
            result = decompressor.multi_decompress_to_buffer(
                [encoded for encoded, _, _ in group],
                decompressed_sizes=sizes.tobytes(),
                threads=workers,
            )

            def place(i):
                _, offset, decoded = group[i]
                segment = result[i]
                if len(segment) != decoded:
                    raise ValueError("an outer Zstd chunk decoded to the wrong size")
                np.copyto(
                    host[offset : offset + decoded], np.frombuffer(segment, np.uint8)
                )

            list(pool.map(place, range(len(group))))
            del result
        if chunk is not None:
            group, size = [chunk], chunk[2]


def _plan_batch(
    target_version, members, delta_map, decoder, host, error, plans
) -> Batch:
    offsets, size = [], 0
    for entry, _ in members:
        size = _aligned(size)
        offsets.append(size)
        size += entry["nbytes"]
    columns, zero_ranges = [], []
    for (entry, arena), offset in zip(members, offsets):
        cursor = 0
        for frame in entry["frames"]:
            if frame["decoded_offset"] > cursor:
                zero_ranges.append((offset + cursor, frame["decoded_offset"] - cursor))
            columns.append(
                (
                    arena + frame["encoded_offset"],
                    frame["encoded_bytes"],
                    frame["decoded_bytes"],
                    offset + frame["decoded_offset"],
                )
            )
            cursor = frame["decoded_offset"] + frame["decoded_bytes"]
        if cursor < entry["nbytes"]:
            zero_ranges.append((offset + cursor, entry["nbytes"] - cursor))
    frames = np.ascontiguousarray(np.array(columns, dtype=np.int64).reshape(-1, 4).T)
    decode = decoder.plan(frames, host)
    # The XOR plan depends only on which tensors the batch holds: an RL step changes the
    # same tensors every update, so the previous prepare's plan is reused.
    key = tuple(entry["name"] for entry, _ in members)
    if key not in plans:
        # Sources view a meta stand-in for the scratch buffer, which exists only in the pause.
        scratch = torch.empty(size, dtype=torch.uint8, device="meta")
        pairs = []
        for (entry, _), offset in zip(members, offsets):
            tensor = delta_map.inventory[entry["name"]]
            canonical = (
                scratch[offset : offset + entry["nbytes"]]
                .view(TORCH_DTYPES[tensor.dtype])
                .reshape(tensor.shape)
            )
            for place in delta_map.entries[entry["name"]]:
                pairs.append((_bytes(canonical[place.src]), _bytes(place.dst)))
        group = plan_xor(pairs)
        group.prepare_metadata(decoder.device.index)
        pointers = np.fromiter(group.pointer_rows(0), dtype=np.int64)
        plans[key] = (
            group,
            np.concatenate([pointers, np.asarray(group.metadata_rows, dtype=np.int64)]),
        )
    group, rows = plans[key]
    return Batch(
        target_version,
        list(key),
        offsets,
        size,
        zero_ranges,
        decode,
        prepare_status_check(decode, error),
        group,
        rows,
    )


def fetch(
    delta_map: DeltaMap,
    manifest_paths: list[str],
    version: str,
    decoder: SnappyDecoder,
    workers: int,
    arena: HostArena,
    plans: dict,
    verify_sha256: bool = False,
) -> Fetched:
    """Any failure here leaves the engine as it was. `plans` is the previous prepare's."""
    started = time.perf_counter()
    with torch.cuda.nvtx.range("weight_delta.prepare.check"):
        manifests = [read_manifest(path) for path in manifest_paths]
        if not manifests:
            raise ValueError("no deltas given")
        _check_chain(manifests, version)
        encodings = {}
        for manifest in manifests:
            delta_map.check_manifest(manifest)
            _check_files(manifest, verify_sha256)
            for name, entry in manifest.tensors.items():
                if encodings.setdefault(name, entry["encoding"]) != entry["encoding"]:
                    raise ValueError(f"{name} changes encoding between versions")

        xor_work, newest_raw = [], {}
        for index, manifest in enumerate(manifests):
            for name, entry in manifest.tensors.items():
                if not isinstance(delta_map.entries[name], tuple):
                    continue  # held by another GPU
                if entry["encoding"] == "raw_bytes":
                    if "raw" in entry:
                        newest_raw[name] = (index, entry)
                elif "outer" in entry:
                    xor_work.append((index, entry))
    checked = time.perf_counter()

    arenas, host_bytes = [], 0
    for _, entry in xor_work:
        arenas.append(host_bytes)
        host_bytes += _aligned(entry["outer"]["decoded_bytes"])
    host_tensor = arena.fit(host_bytes)
    allocated = time.perf_counter()

    with torch.cuda.nvtx.range("weight_delta.prepare.outer_zstd"):
        files = [_map_files(manifest) for manifest in manifests]
        chunks = []
        for (index, entry), offset in zip(xor_work, arenas):
            outer = entry["outer"]
            region = files[index][outer["file"]][
                outer["encoded_offset"] : outer["encoded_offset"]
                + outer["encoded_bytes"]
            ]
            if len(region) != outer["encoded_bytes"]:
                raise ValueError(f"{entry['name']}: payload is truncated")
            for chunk in outer["frames"]:
                start = chunk["encoded_offset"]
                chunks.append(
                    (
                        region[start : start + chunk["encoded_bytes"]],
                        offset + chunk["decoded_offset"],
                        chunk["decoded_bytes"],
                    )
                )
        with ThreadPoolExecutor(workers, thread_name_prefix="weight-delta-zstd") as pool:
            _undo_outer_zstd(chunks, host_tensor.numpy(), pool, workers)
        raw = []
        for name, (index, entry) in newest_raw.items():
            record = entry["raw"]
            start = record["encoded_offset"]
            payload = files[index][record["file"]][
                start : start + record["encoded_bytes"]
            ]
            if len(payload) != record["encoded_bytes"]:
                raise ValueError(f"{name}: payload is truncated")
            value = (
                torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                .to(decoder.device, non_blocking=False)
                .view(TORCH_DTYPES[entry["dtype"]])
                .reshape(entry["shape"])
            )
            raw.append((value, delta_map.entries[name]))
        del chunks, files
    decoded = time.perf_counter()

    with torch.cuda.nvtx.range("weight_delta.prepare.plan"):
        # Batches fill in order up to BATCH_BYTES of scratch; a bigger tensor is a batch
        # of its own. A batch never spans versions, so no tensor appears twice in one.
        groups, size = [], 0
        for (index, entry), offset in zip(xor_work, arenas):
            end = _aligned(size) + entry["nbytes"]
            if not groups or groups[-1][0] != index or end > BATCH_BYTES:
                groups.append((index, []))
                end = entry["nbytes"]
            groups[-1][1].append((entry, offset))
            size = end
        error = torch.zeros(1, dtype=torch.int32, device=decoder.device)
        known = set(plans)
        used, batches = {}, []
        for index, members in groups:
            batch = _plan_batch(
                manifests[index].target_version,
                members,
                delta_map,
                decoder,
                host_tensor,
                error,
                plans,
            )
            used[tuple(batch.tensors)] = plans[tuple(batch.tensors)]
            batches.append(batch)
    planned = time.perf_counter()
    return Fetched(
        base_version=manifests[0].base_version,
        target_version=manifests[-1].target_version,
        batches=batches,
        raw=raw,
        scratch_bytes=max((batch.scratch_bytes for batch in batches), default=0),
        staging_bytes=0
        if decoder.hardware
        else max((batch.decode.input_bytes for batch in batches), default=0),
        error=error,
        plans=used,
        stats={
            "versions": len(manifests),
            "xor_tensors": len(xor_work),
            "raw_tensors": len(raw),
            "host_bytes": host_bytes,
            "frames": sum(batch.decode.count for batch in batches),
            "decode_temp_bytes": sum(batch.decode.temp.numel() for batch in batches),
            "batches": len(batches),
            "xor_pairs": sum(len(batch.xor.sources) for batch in batches),
            "plans_reused": sum(1 for key in used if key in known),
            "check_s": checked - started,
            "host_alloc_s": allocated - checked,
            "outer_zstd_s": decoded - allocated,
            "plan_s": planned - decoded,
        },
    )
