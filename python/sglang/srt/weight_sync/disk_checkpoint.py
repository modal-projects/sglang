"""Materialize versioned checkpoints on host-local storage.

Each ``weight_v{N:06d}`` source is either a full Hugging Face checkpoint or a
compressed per-tensor delta over version N-1. Materialization seeds the newest
full checkpoint, applies and verifies the remaining deltas, then records the
target version.

A host-local file lock makes concurrent calls from model workers safe. Updated
files are synchronized before the version marker, and a checksum mismatch
triggers one clean reseed before failing loudly.
"""

from __future__ import annotations

import fcntl
import glob
import json
import logging
import mmap
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import zstandard

from sglang.srt.utils import dynamic_import
from sglang.srt.weight_sync.checksum import calculate_checksum, create_checksum

logger = logging.getLogger(__name__)

# XOR targets are read and written through ordinary positional I/O instead of
# mmap. An XOR delta must read and checksum every target byte, and positional
# I/O keeps that access sequential instead of making progress page-fault bound.
# Bound compressed, decompressed, and target work buffers across workers. A
# single tensor larger than the budget is admitted alone.
_DELTA_APPLY_MEMORY_BYTES = 8 << 30
_POSITIONAL_IO_CHUNK_BYTES = 64 << 20

_MAX_SEED_COPY_WORKERS = 8
_MAX_DELTA_APPLY_WORKERS = 32
_MAX_SAFETENSORS_HEADER_BYTES = 100 << 20
_SEED_COPY_CHUNK_BYTES = 16 << 20
_SEED_LOG_STEP_GB = 50

# Per-checkpoint dir holding the applied-version marker and materialization lock.
_SYNC_DIR = ".weight_sync"


class _ChecksumMismatchError(RuntimeError):
    """The reconstructed local tensor bytes do not match the published target."""


def _available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _paths_overlap(left: str, right: str) -> bool:
    left = os.path.realpath(left)
    right = os.path.realpath(right)
    common = os.path.commonpath((left, right))
    return common == left or common == right


def materialize(
    local_checkpoint_dir: str,
    base_checkpoint_dir: str,
    checkpoint_source_dir: str,
    target_version: int,
    checkpoint_source_refresh_hook: str | None = None,
) -> dict:
    """Bring the host-local checkpoint up to ``target_version``.

    The optional refresh hook can make a newly published target visible before
    the first local worker reads it.

    Missing or incomplete source files raise ``FileNotFoundError`` without
    reseeding. A checksum mismatch on a complete source is treated as corrupt
    local state and gets one replay from a clean seed.
    """
    if target_version < 0:
        raise ValueError("target_version must be non-negative")
    if _paths_overlap(local_checkpoint_dir, base_checkpoint_dir):
        raise ValueError(
            "local_checkpoint_dir must not overlap the immutable "
            "base_checkpoint_dir tree"
        )
    if _paths_overlap(local_checkpoint_dir, checkpoint_source_dir):
        raise ValueError(
            "local_checkpoint_dir must not overlap the published "
            "checkpoint_source_dir tree"
        )
    started = time.perf_counter()
    lock_started = time.perf_counter()
    with _materialization_lock(local_checkpoint_dir):
        lock_wait_s = time.perf_counter() - lock_started
        applied = _read_applied_version(local_checkpoint_dir)
        if applied == target_version:
            # A co-located rank already brought this host up to the target.
            return {
                "operation": "noop",
                "initial_version": applied,
                "target_version": target_version,
                "lock_wait_s": round(lock_wait_s, 6),
                "source_refresh_wall_s": 0.0,
                "wall_s": round(time.perf_counter() - started, 6),
            }
        refresh_started = time.perf_counter()
        refresh_checkpoint_source(
            checkpoint_source_dir,
            target_version,
            checkpoint_source_refresh_hook,
        )
        source_refresh_wall_s = time.perf_counter() - refresh_started
        try:
            stats = _materialize_locked(
                local_checkpoint_dir,
                base_checkpoint_dir,
                checkpoint_source_dir,
                target_version,
                reseed=applied is not None and applied > target_version,
            )
        except FileNotFoundError:
            # A source version is missing or not fully materialized — a readiness
            # failure the caller owns, not local corruption. Reseeding cannot
            # conjure absent bytes, so record what the mount shows and fail fast;
            # the caller reloads and retries.
            _log_missing_source(checkpoint_source_dir, target_version)
            raise
        except _ChecksumMismatchError:
            # A checksum mismatch on staged, complete bytes == corrupt local
            # state (incomplete sources are reclassified to FileNotFoundError
            # above and never reach here). Reseed from the pristine base and
            # replay once; a failure on that fresh state re-raises.
            logger.exception(
                "materialization of v%d failed on staged sources; "
                "reseeding from base and replaying",
                target_version,
            )
            stats = _materialize_locked(
                local_checkpoint_dir,
                base_checkpoint_dir,
                checkpoint_source_dir,
                target_version,
                reseed=True,
            )
            stats["reseed_after_failed_apply"] = True
        stats["initial_version"] = applied
        stats["target_version"] = target_version
        stats["lock_wait_s"] = round(lock_wait_s, 6)
        stats["source_refresh_wall_s"] = round(source_refresh_wall_s, 6)
        stats["wall_s"] = round(time.perf_counter() - started, 6)
        return stats


def refresh_checkpoint_source(
    checkpoint_source_dir: str,
    target_version: int,
    checkpoint_source_refresh_hook: str | None,
) -> None:
    """Run an optional visibility hook before reading a published target."""

    if target_version > 0 and checkpoint_source_refresh_hook is not None:
        dynamic_import(checkpoint_source_refresh_hook)(
            checkpoint_source_dir,
            target_version,
        )


def _materialize_locked(
    local_checkpoint_dir: str,
    base_checkpoint_dir: str,
    checkpoint_source_dir: str,
    target_version: int,
    reseed: bool,
) -> dict:
    # A torn local state (reseed=True) is treated like a fresh host: the
    # applied-version marker can't be trusted over partially-mutated files.
    applied = None if reseed else _read_applied_version(local_checkpoint_dir)
    # Scan back from the target for the newest full version. Stop at the
    # local state — below it a reset can never be needed (or, on a fresh
    # host, at 0 = the engine's base).
    floor = applied if applied is not None else 0
    start = target_version
    while start > floor and _is_delta(
        _version_dir(checkpoint_source_dir, start), start
    ):
        start -= 1
    if applied is None or start > applied:
        seed_dir = (
            base_checkpoint_dir
            if start == 0
            else _version_dir(checkpoint_source_dir, start)
        )
        seed_started = time.perf_counter()
        _reset_checkpoint(seed_dir, local_checkpoint_dir, start)
        seed_wall_s = time.perf_counter() - seed_started
    else:
        start = applied
        seed_wall_s = 0.0
    if start == target_version:
        apply_stats = {
            "operation": "seed_only" if seed_wall_s else "noop",
            "wall_s": 0.0,
        }
    else:
        applied_deltas = [
            _apply_delta(
                local_checkpoint_dir,
                _version_dir(checkpoint_source_dir, version),
                version,
            )
            for version in range(start + 1, target_version + 1)
        ]
        apply_stats = {
            "operation": "apply_deltas",
            "versions": len(applied_deltas),
            "wall_s": round(sum(value["wall_s"] for value in applied_deltas), 6),
        }
        if len(applied_deltas) == 1:
            apply_stats = applied_deltas[0]
    return {
        "operation": "reseed_and_apply" if reseed else "materialize",
        "seed_wall_s": round(seed_wall_s, 6),
        "apply": apply_stats,
    }


def _log_missing_source(source_dir: str, target_version: int) -> None:
    """Record the visible source state after an incomplete read."""
    vdir = _version_dir(source_dir, target_version)
    try:
        versions = sorted(n for n in os.listdir(source_dir) if n.startswith("weight_v"))
    except OSError as e:
        versions = [f"<listdir {source_dir} failed: {e}>"]
    target_contents = None
    if os.path.isdir(vdir):
        try:
            target_contents = sorted(os.listdir(vdir))
        except OSError as e:
            target_contents = [f"<listdir failed: {e}>"]
    logger.error(
        "Checkpoint source v%d is incomplete: versions=%s isdir(%s)=%s contents=%s",
        target_version,
        versions,
        vdir,
        os.path.isdir(vdir),
        target_contents,
    )


def _version_dir(source_dir: str, version: int) -> str:
    return os.path.join(source_dir, f"weight_v{version:06d}")


def _validate_published_version(
    metadata: object,
    expected_version: int,
    index_path: str,
) -> dict:
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid checkpoint metadata: {index_path}")
    try:
        published_version = int(metadata["version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"checkpoint manifest has no valid version: {index_path}"
        ) from exc
    if published_version != expected_version:
        raise ValueError(
            f"checkpoint version mismatch: directory is v{expected_version}, "
            f"manifest declares v{published_version}"
        )
    return metadata


def _is_delta(version_dir: str, expected_version: int) -> bool:
    """Read the published manifest and return whether this version is a delta."""
    if not os.path.isdir(version_dir):
        raise FileNotFoundError(f"published weight version missing: {version_dir}")
    index_path = os.path.join(version_dir, "model.safetensors.index.json")
    try:
        with open(index_path) as file:
            index = json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"published weight version has no manifest: {index_path}"
        ) from exc
    metadata = _validate_published_version(
        index.get("metadata"), expected_version, index_path
    )
    return "delta_encoding" in metadata


@contextmanager
def _materialization_lock(local_checkpoint_dir: str):
    sync = os.path.join(local_checkpoint_dir, _SYNC_DIR)
    os.makedirs(sync, exist_ok=True)
    with open(os.path.join(sync, "lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read_applied_version(local_checkpoint_dir: str) -> int | None:
    try:
        with open(os.path.join(local_checkpoint_dir, _SYNC_DIR, "state.json")) as f:
            return int(json.load(f)["version"])
    except FileNotFoundError:
        return None


def _write_applied_version(local_checkpoint_dir: str, version: int) -> None:
    sync_dir = os.path.join(local_checkpoint_dir, _SYNC_DIR)
    path = os.path.join(sync_dir, "state.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"version": f"{version:06d}"}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(sync_dir)


def _clear_applied_version(local_checkpoint_dir: str) -> None:
    sync_dir = os.path.join(local_checkpoint_dir, _SYNC_DIR)
    try:
        os.remove(os.path.join(sync_dir, "state.json"))
    except FileNotFoundError:
        return
    _fsync_dir(sync_dir)


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _drop_page_cache(path: str) -> None:
    """Evict a file from the page cache (POSIX_FADV_DONTNEED)."""
    if not hasattr(os, "posix_fadvise"):  # POSIX-only (absent on macOS/Windows)
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def _reset_checkpoint(src_dir: str, local_checkpoint_dir: str, version: int) -> None:
    """Make local_checkpoint_dir an exact copy of the full checkpoint in src_dir
    (files the new checkpoint doesn't have — e.g. differently-sharded old ones —
    are pruned). Later deltas chain on top of this state."""
    if os.path.realpath(src_dir) == os.path.realpath(local_checkpoint_dir):
        raise ValueError(
            "a published full checkpoint cannot also be the mutable " "local checkpoint"
        )
    _validate_full_checkpoint(src_dir, version=version)
    os.makedirs(local_checkpoint_dir, exist_ok=True)
    # A full seed replaces every checkpoint byte. Invalidate the old marker
    # before the first mutation so an interrupted copy cannot be mistaken for
    # the previously committed version.
    _clear_applied_version(local_checkpoint_dir)
    src_files = [entry for entry in os.scandir(src_dir) if entry.is_file()]
    total_gb = sum(entry.stat().st_size for entry in src_files) / 1e9
    workers = min(
        _MAX_SEED_COPY_WORKERS,
        _available_cpu_count(),
        len(src_files) or 1,
    )
    logger.info(
        "staging checkpoint v%d to local disk: %.0f GB in %d files, %d parallel streams (%s -> %s)",
        version,
        total_gb,
        len(src_files),
        workers,
        src_dir,
        local_checkpoint_dir,
    )
    start = time.monotonic()
    progress = {"done_gb": 0.0, "next_log_gb": _SEED_LOG_STEP_GB}
    progress_lock = threading.Lock()

    def copy_one(entry) -> None:
        dst = os.path.join(local_checkpoint_dir, entry.name)
        with open(entry.path, "rb") as src, open(dst, "wb") as out:
            while chunk := src.read(_SEED_COPY_CHUNK_BYTES):
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        # Don't let the source evict the local copy we keep resident.
        _drop_page_cache(entry.path)
        with progress_lock:
            progress["done_gb"] += entry.stat().st_size / 1e9
            done = progress["done_gb"]
            if done >= progress["next_log_gb"] or done >= total_gb:
                rate = done / max(time.monotonic() - start, 1e-3)
                logger.info(
                    "staging checkpoint v%d to local disk: %.0f/%.0f GB (%.0f%%), %.1f GB/s",
                    version,
                    done,
                    total_gb,
                    100 * done / max(total_gb, 1e-9),
                    rate,
                )
                progress["next_log_gb"] += _SEED_LOG_STEP_GB

    # The size check below fails loud if the mount served a short read on any shard.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(copy_one, src_files))
    names = {entry.name for entry in src_files}
    for entry in os.scandir(local_checkpoint_dir):
        if entry.is_file() and entry.name not in names:
            os.remove(entry.path)
    # a truncated copy (e.g. an object-store mount surfacing metadata before
    # bytes) must fail loud, not serve bad weights
    for entry in src_files:
        copied = os.path.getsize(os.path.join(local_checkpoint_dir, entry.name))
        if copied != entry.stat().st_size:
            raise RuntimeError(
                f"size mismatch copying {entry.name}: src {entry.stat().st_size} != local {copied}"
            )
    _fsync_dir(local_checkpoint_dir)
    _write_applied_version(local_checkpoint_dir, version)


def _tensor_locations(ckpt_dir: str) -> dict:
    """Map each tensor name to (file, byte offset, nbytes) by reading every safetensors header."""
    locations = {}
    for path in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
        header_len, header = _read_safetensors_header(path)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in locations:
                raise ValueError(f"duplicate checkpoint tensor {name!r}")
            begin, end = info["data_offsets"]
            locations[name] = (path, 8 + header_len + begin, end - begin)
    return locations


def _read_safetensors_header(path: str) -> tuple[int, dict]:
    """Read a complete header and reject a truncated or oversized payload."""

    with open(path, "rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise FileNotFoundError(f"incomplete safetensors header: {path}")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len > _MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(
                f"safetensors header exceeds {_MAX_SAFETENSORS_HEADER_BYTES} bytes: "
                f"{path}"
            )
        header_bytes = file.read(header_len)
    if len(header_bytes) != header_len:
        raise FileNotFoundError(f"incomplete safetensors header: {path}")
    try:
        header = json.loads(header_bytes)
    except ValueError as exc:
        raise ValueError(f"invalid safetensors header: {path}") from exc
    end = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        end = max(end, info["data_offsets"][1])
    expected_size = 8 + header_len + end
    actual_size = os.path.getsize(path)
    if actual_size != expected_size:
        raise FileNotFoundError(
            f"incomplete source blob {path}: {actual_size}B, "
            f"header declares {expected_size}B"
        )
    return header_len, header


def _validate_full_checkpoint(src_dir: str, *, version: int) -> None:
    """Reject incomplete full checkpoints before copying or publishing a marker."""

    headers = {}
    for path in sorted(glob.glob(os.path.join(src_dir, "*.safetensors"))):
        _, headers[os.path.basename(path)] = _read_safetensors_header(path)
    if not headers:
        raise FileNotFoundError(f"full checkpoint has no safetensors files: {src_dir}")

    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    try:
        with open(index_path) as file:
            index = json.load(file)
    except FileNotFoundError as exc:
        if version == 0:
            return
        raise FileNotFoundError(
            f"published full checkpoint has no manifest: {index_path}"
        ) from exc
    if version > 0:
        _validate_published_version(index.get("metadata"), version, index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"published full checkpoint has no weight map: {index_path}")
    for name, filename in weight_map.items():
        header = headers.get(filename)
        if header is None:
            raise FileNotFoundError(
                f"published full checkpoint is missing blob {filename!r}"
            )
        if name not in header:
            raise ValueError(
                f"published full checkpoint blob {filename!r} "
                f"does not contain indexed tensor {name!r}"
            )


@dataclass(frozen=True)
class _DiskDeltaItem:
    name: str
    source_path: str
    source_offset: int
    compressed_nbytes: int
    target_path: str
    target_offset: int
    target_nbytes: int
    expected_checksum: str


class _ByteBudget:
    """Bound concurrent full-tensor work buffers without rejecting large tensors."""

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


def _pread_exact(
    fd: int,
    offset: int,
    nbytes: int,
    *,
    source_path: str | None = None,
) -> bytearray:
    result = bytearray(nbytes)
    view = memoryview(result)
    position = 0
    while position < nbytes:
        count = min(_POSITIONAL_IO_CHUNK_BYTES, nbytes - position)
        read = os.preadv(
            fd,
            [view[position : position + count]],
            offset + position,
        )
        if read <= 0:
            error_type = FileNotFoundError if source_path is not None else RuntimeError
            raise error_type(
                f"short positional read: offset={offset} expected={nbytes} "
                f"actual={position}"
                + (f" source={source_path}" if source_path is not None else "")
            )
        position += read
    return result


def _pwrite_all(fd: int, offset: int, data: bytearray) -> None:
    view = memoryview(data)
    position = 0
    while position < len(view):
        end = min(position + _POSITIONAL_IO_CHUNK_BYTES, len(view))
        written = os.pwrite(fd, view[position:end], offset + position)
        if written <= 0:
            raise RuntimeError(
                f"short positional write: offset={offset} expected={len(view)} "
                f"actual={position}"
            )
        position += written


def _apply_delta(
    local_checkpoint_dir: str,
    version_dir: str,
    expected_version: int,
) -> dict:
    """Apply one version's delta in place and fail on any checksum mismatch."""
    started = time.perf_counter()
    metadata_started = time.perf_counter()
    with open(os.path.join(version_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    meta = _validate_published_version(
        index.get("metadata"),
        expected_version,
        os.path.join(version_dir, "model.safetensors.index.json"),
    )
    applied = _read_applied_version(local_checkpoint_dir)
    if applied == int(meta["version"]):
        return {
            "operation": "noop",
            "version": int(meta["version"]),
            "wall_s": round(time.perf_counter() - started, 6),
        }
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"invalid delta weight map: {version_dir}")
    # Validate the source before reading it: every blob the index names must be
    # present, or a half-propagated version would apply only the blobs that made
    # it and report the rest as a checksum mismatch (misread as corruption). A
    # missing blob is a not-ready source, so raise FileNotFoundError — staging
    # fails fast and the caller reloads + retries instead of reseeding.
    for blob in sorted(set(weight_map.values())):
        if not os.path.exists(os.path.join(version_dir, blob)):
            raise FileNotFoundError(
                f"incomplete source version {version_dir}: missing blob {blob}"
            )
    if applied != int(meta["base_version"]):
        raise RuntimeError(
            f"out-of-order delta: local at {applied}, delta builds on {meta['base_version']}"
        )
    if meta["compression_format"] != "zstd":
        raise NotImplementedError(
            f"compression {meta['compression_format']!r} not supported"
        )
    encoding = meta["delta_encoding"]
    algorithm = meta["checksum_format"]
    expected_delta_names = set(weight_map)
    expected_delta_files = set(weight_map.values())
    locations = _tensor_locations(local_checkpoint_dir)
    metadata_wall_s = time.perf_counter() - metadata_started
    mismatches = []
    lock = threading.Lock()
    items: list[_DiskDeltaItem] = []
    seen_delta_names = set()
    source_setup_started = time.perf_counter()
    for delta_filename in sorted(expected_delta_files):
        delta_file = os.path.join(version_dir, delta_filename)
        header_len, header = _read_safetensors_header(delta_file)
        want_checksums = header.get("__metadata__", {})
        if not isinstance(want_checksums, dict):
            raise ValueError(f"invalid delta checksum metadata: {delta_file}")
        seen_file_names = set()
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if weight_map.get(name) != delta_filename:
                raise ValueError(
                    f"delta blob {delta_file} contains unindexed tensor {name!r}"
                )
            if name in seen_delta_names:
                raise ValueError(f"duplicate delta tensor {name!r}")
            seen_file_names.add(name)
            seen_delta_names.add(name)
            begin, end = info["data_offsets"]
            if begin < 0 or begin > end:
                raise ValueError(f"invalid delta offsets for {name!r} in {delta_file}")
            try:
                path, offset, nbytes = locations[name]
            except KeyError as exc:
                raise ValueError(
                    f"delta tensor {name!r} is absent from the local checkpoint"
                ) from exc
            data_start = 8 + header_len
            want_checksum = want_checksums.get(name)
            if not isinstance(want_checksum, str) or not want_checksum:
                raise ValueError(f"delta tensor {name!r} has no target checksum")
            items.append(
                _DiskDeltaItem(
                    name=name,
                    source_path=delta_file,
                    source_offset=data_start + begin,
                    compressed_nbytes=end - begin,
                    target_path=path,
                    target_offset=offset,
                    target_nbytes=nbytes,
                    expected_checksum=want_checksum,
                )
            )
        expected_file_names = {
            name for name, filename in weight_map.items() if filename == delta_filename
        }
        if seen_file_names != expected_file_names:
            raise ValueError(
                f"delta blob/index tensor mismatch for {delta_file}: "
                f"missing={sorted(expected_file_names - seen_file_names)[:20]} "
                f"extra={sorted(seen_file_names - expected_file_names)[:20]}"
            )
    if seen_delta_names != expected_delta_names:
        raise ValueError(
            f"delta index tensor mismatch for {version_dir}: "
            f"missing={sorted(expected_delta_names - seen_delta_names)[:20]} "
            f"extra={sorted(seen_delta_names - expected_delta_names)[:20]}"
        )
    source_setup_wall_s = time.perf_counter() - source_setup_started

    # The target is about to change in place. Remove the base marker only after
    # the complete source has been validated; an interrupted mutation must
    # never be mistaken for either the base or target version.
    _clear_applied_version(local_checkpoint_dir)
    source_files = {}
    target_files = {}
    prefetch_wall_s = 0.0
    close_wall_s = 0.0
    try:
        source_files = {
            path: os.open(path, os.O_RDONLY)
            for path in {item.source_path for item in items}
        }
        target_files = {
            path: os.open(path, os.O_RDWR)
            for path in {item.target_path for item in items}
        }
        memory_budget = _ByteBudget(_DELTA_APPLY_MEMORY_BYTES)
        if encoding == "xor":

            def apply_xor(item: _DiskDeltaItem) -> None:
                working_nbytes = 2 * item.target_nbytes + item.compressed_nbytes
                with memory_budget.reserve(working_nbytes):
                    compressed = _pread_exact(
                        source_files[item.source_path],
                        item.source_offset,
                        item.compressed_nbytes,
                        source_path=item.source_path,
                    )
                    region = _pread_exact(
                        target_files[item.target_path],
                        item.target_offset,
                        item.target_nbytes,
                    )

                    delta = zstandard.ZstdDecompressor().decompress(
                        compressed,
                        max_output_size=item.target_nbytes,
                    )
                    if len(delta) != item.target_nbytes:
                        raise RuntimeError(
                            f"decompressed delta size mismatch for {item.name!r}: "
                            f"expected={item.target_nbytes} actual={len(delta)}"
                        )

                    region_view = np.frombuffer(region, dtype=np.uint8)
                    delta_view = np.frombuffer(delta, dtype=np.uint8)
                    hasher = create_checksum(algorithm)
                    for position in range(
                        0,
                        item.target_nbytes,
                        _POSITIONAL_IO_CHUNK_BYTES,
                    ):
                        end = min(
                            position + _POSITIONAL_IO_CHUNK_BYTES,
                            item.target_nbytes,
                        )
                        np.bitwise_xor(
                            region_view[position:end],
                            delta_view[position:end],
                            out=region_view[position:end],
                        )
                        hasher.update(region_view[position:end])
                    actual = hasher.hexdigest()
                    if actual != item.expected_checksum:
                        with lock:
                            mismatches.append(item.name)
                    else:
                        _pwrite_all(
                            target_files[item.target_path],
                            item.target_offset,
                            region,
                        )

            apply_tensor = apply_xor
            io_backend = "pread_pwrite"
        elif encoding == "overwrite":
            open_mmaps = {
                path: (
                    (fh := open(path, "r+b")),
                    mmap.mmap(fh.fileno(), 0),
                )
                for path in target_files
            }

            prefetch_started = time.perf_counter()
            for _, mm in open_mmaps.values():
                try:
                    mm.madvise(mmap.MADV_WILLNEED)
                except (OSError, AttributeError, ValueError):
                    pass
            prefetch_wall_s = time.perf_counter() - prefetch_started

            def apply_overwrite(item: _DiskDeltaItem) -> None:
                # The sparse format stores one uint32 position and one byte per
                # changed target byte. Reject duplicate-heavy or malformed
                # payloads by bounding the decompressed size accordingly.
                max_payload_nbytes = 4 + 5 * item.target_nbytes
                with memory_budget.reserve(item.compressed_nbytes + max_payload_nbytes):
                    compressed = _pread_exact(
                        source_files[item.source_path],
                        item.source_offset,
                        item.compressed_nbytes,
                        source_path=item.source_path,
                    )
                    payload = zstandard.ZstdDecompressor().decompress(
                        compressed,
                        max_output_size=max_payload_nbytes,
                    )
                    if len(payload) < 4:
                        raise RuntimeError(
                            f"overwrite delta for {item.name!r} is truncated"
                        )
                    count = int.from_bytes(payload[:4], "little")
                    positions_end = 4 + 4 * count
                    if positions_end > len(payload):
                        raise RuntimeError(
                            f"overwrite positions for {item.name!r} are truncated"
                        )
                    positions = np.frombuffer(payload[4:positions_end], dtype="<u4")
                    values = np.frombuffer(payload[positions_end:], dtype=np.uint8)
                    if (
                        count > item.target_nbytes
                        or values.size != count
                        or (count and int(positions.max()) >= item.target_nbytes)
                    ):
                        raise RuntimeError(
                            f"overwrite payload for {item.name!r} is invalid"
                        )
                    region = np.ndarray(
                        (item.target_nbytes,),
                        dtype=np.uint8,
                        buffer=open_mmaps[item.target_path][1],
                        offset=item.target_offset,
                    )
                    region[positions] = values
                    if calculate_checksum(algorithm, region) != item.expected_checksum:
                        with lock:
                            mismatches.append(item.name)

            apply_tensor = apply_overwrite
            io_backend = "mmap_sparse_overwrite"
        else:
            raise NotImplementedError(f"delta encoding {encoding!r} not supported")

        apply_started = time.perf_counter()
        # Decompression, XOR/scatter, and checksumming release the GIL and are
        # memory-bandwidth bound. Bound concurrency by both available CPUs and
        # independent tensors; more than 32 streams only add scheduler pressure.
        workers = min(
            _MAX_DELTA_APPLY_WORKERS,
            _available_cpu_count(),
            len(items) or 1,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(apply_tensor, items))
        apply_wall_s = time.perf_counter() - apply_started

        # Persist target bytes before the applied-version marker. A failure
        # before the marker causes a checksum failure and clean reseed on retry.
        flush_started = time.perf_counter()
        if encoding == "overwrite":
            for _, mm in open_mmaps.values():
                mm.flush()
        else:
            for fd in target_files.values():
                os.fsync(fd)
        flush_wall_s = time.perf_counter() - flush_started
    finally:
        close_started = time.perf_counter()
        if encoding == "overwrite" and "open_mmaps" in locals():
            for fh, mm in open_mmaps.values():
                mm.close()
                fh.close()
        for fd in target_files.values():
            os.close(fd)
        for fd in source_files.values():
            os.close(fd)
        close_wall_s = time.perf_counter() - close_started

    if mismatches:
        raise _ChecksumMismatchError(
            f"checksum mismatch for {len(mismatches)} tensors after applying {version_dir}: "
            f"{sorted(mismatches)[:20]}"
        )
    marker_started = time.perf_counter()
    _write_applied_version(local_checkpoint_dir, int(meta["version"]))
    marker_wall_s = time.perf_counter() - marker_started
    stats = {
        "operation": f"apply_{encoding}",
        "version": int(meta["version"]),
        "delta_tensors": len(items),
        "checkpoint_shards": len(target_files),
        "compressed_bytes": sum(item.compressed_nbytes for item in items),
        "target_tensor_bytes": sum(item.target_nbytes for item in items),
        "workers": workers,
        "io_backend": io_backend,
        "working_memory_budget_bytes": _DELTA_APPLY_MEMORY_BYTES,
        "phases": {
            "metadata_wall_s": round(metadata_wall_s, 6),
            "source_setup_wall_s": round(source_setup_wall_s, 6),
            "prefetch_wall_s": round(prefetch_wall_s, 6),
            "apply_wall_s": round(apply_wall_s, 6),
            "flush_wall_s": round(flush_wall_s, 6),
            "close_wall_s": round(close_wall_s, 6),
            "marker_wall_s": round(marker_wall_s, 6),
        },
        "wall_s": round(time.perf_counter() - started, 6),
    }
    logger.info(
        "Applied checkpoint delta v%d: tensors=%d target_bytes=%d wall_time=%.3fs",
        stats["version"],
        stats["delta_tensors"],
        stats["target_tensor_bytes"],
        stats["wall_s"],
    )
    return stats
