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
import os
import resource
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zstandard

from sglang.srt.utils import dynamic_import
from sglang.srt.weight_sync.checksum import (
    calculate_checksum,
    create_checksum,
    validate_checksum,
)
from sglang.srt.weight_sync.file_io import PositionalFileRangeReader, read_exact

logger = logging.getLogger(__name__)

# XOR targets are read and written through ordinary positional I/O instead of
# mmap. An XOR delta must read and checksum every target byte, and positional
# I/O keeps that access sequential instead of making progress page-fault bound.
# Bound target and sparse-overwrite work buffers across workers. Compressed XOR
# bytes are decoded through a fixed-size stream buffer. A single tensor larger
# than the budget is admitted alone.
_DELTA_APPLY_MEMORY_BYTES = 8 << 30
_POSITIONAL_IO_CHUNK_BYTES = 64 << 20
_DELTA_STREAM_CHUNK_BYTES = 4 << 20

_MAX_SEED_COPY_WORKERS = 8
_MAX_DELTA_APPLY_WORKERS = 32
_MAX_OPEN_FILES_PER_CACHE = 256
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
        versions = list(range(start + 1, target_version + 1))
        if len(versions) == 1:
            apply_stats = _apply_delta(
                local_checkpoint_dir,
                _version_dir(checkpoint_source_dir, target_version),
                target_version,
            )
        else:
            apply_stats = _apply_delta_lineage(
                local_checkpoint_dir,
                [
                    (_version_dir(checkpoint_source_dir, version), version)
                    for version in versions
                ],
            )
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
    encoding: str
    name: str
    source_path: str
    source_offset: int
    compressed_nbytes: int
    target_path: str
    target_offset: int
    target_nbytes: int
    checksum_algorithm: str
    expected_checksum: str


@dataclass(frozen=True)
class _DiskDeltaPlan:
    version: int
    base_version: int
    encoding: str
    items: tuple[_DiskDeltaItem, ...]
    metadata_wall_s: float
    source_setup_wall_s: float


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


@dataclass
class _CachedFileDescriptor:
    fd: int
    users: int = 0
    dirty: bool = False
    last_used: int = 0


class _FileDescriptorCache:
    """Share positional-I/O descriptors under a fixed process-local bound."""

    def __init__(self, flags: int, limit: int):
        self.flags = flags
        self.limit = limit
        self.entries: dict[str, _CachedFileDescriptor] = {}
        self.condition = threading.Condition()
        self.clock = 0
        self.peak_open_files = 0

    @contextmanager
    def acquire(self, path: str, *, write: bool = False):
        with self.condition:
            entry = self.entries.get(path)
            while entry is None:
                if len(self.entries) < self.limit:
                    entry = _CachedFileDescriptor(fd=os.open(path, self.flags))
                    self.entries[path] = entry
                    self.peak_open_files = max(
                        self.peak_open_files,
                        len(self.entries),
                    )
                    break
                idle_path, idle = min(
                    (
                        (candidate_path, candidate)
                        for candidate_path, candidate in self.entries.items()
                        if candidate.users == 0
                    ),
                    key=lambda item: item[1].last_used,
                    default=(None, None),
                )
                if idle is None:
                    self.condition.wait()
                    entry = self.entries.get(path)
                    continue
                del self.entries[idle_path]
                self._close(idle)
            entry.users += 1
            self.clock += 1
            entry.last_used = self.clock
            if write:
                entry.dirty = True
        try:
            yield entry.fd
        finally:
            with self.condition:
                entry.users -= 1
                self.clock += 1
                entry.last_used = self.clock
                self.condition.notify()

    def flush(self) -> None:
        with self.condition:
            for entry in self.entries.values():
                if entry.dirty:
                    os.fsync(entry.fd)
                    entry.dirty = False

    def close(self) -> None:
        with self.condition:
            entries = list(self.entries.values())
            self.entries.clear()
        error = None
        for entry in entries:
            try:
                self._close(entry)
            except OSError as exc:
                error = error or exc
        if error is not None:
            raise error

    @staticmethod
    def _close(entry: _CachedFileDescriptor) -> None:
        try:
            if entry.dirty:
                os.fsync(entry.fd)
        finally:
            os.close(entry.fd)


def _file_descriptor_cache_limit() -> int:
    """Reserve descriptors for the server while bounding long lineages."""

    try:
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        soft_limit = _MAX_OPEN_FILES_PER_CACHE * 2 + 64
    if soft_limit == resource.RLIM_INFINITY:
        soft_limit = _MAX_OPEN_FILES_PER_CACHE * 2 + 64
    return max(
        1,
        min(
            _MAX_OPEN_FILES_PER_CACHE,
            max(2, soft_limit - 64) // 2,
        ),
    )


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


def _resolve_delta_path(version_dir: str, filename: object) -> str:
    if not isinstance(filename, str):
        raise ValueError(f"invalid delta shard path: {filename!r}")
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid delta shard path: {filename!r}")
    root = os.path.realpath(version_dir)
    path = os.path.realpath(os.path.join(root, filename))
    if os.path.commonpath((root, path)) != root:
        raise ValueError(f"invalid delta shard path: {filename!r}")
    return path


def _build_delta_plan(
    version_dir: str,
    expected_version: int,
    *,
    expected_base_version: int | None,
    locations: dict[str, tuple[str, int, int]],
) -> _DiskDeltaPlan:
    """Validate one published delta and resolve its source and target ranges."""

    metadata_started = time.perf_counter()
    index_path = os.path.join(version_dir, "model.safetensors.index.json")
    with open(index_path) as file:
        index = json.load(file)
    metadata = _validate_published_version(
        index.get("metadata"),
        expected_version,
        index_path,
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"invalid delta weight map: {version_dir}")
    try:
        base_version = int(metadata["base_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"delta has no valid base version: {index_path}") from exc
    if base_version != expected_base_version:
        raise RuntimeError(
            f"out-of-order delta v{expected_version}: builds on "
            f"{base_version}, expected {expected_base_version}"
        )
    if metadata.get("compression_format") != "zstd":
        raise NotImplementedError(
            f"compression {metadata.get('compression_format')!r} not supported"
        )
    encoding = metadata.get("delta_encoding")
    if encoding not in {"xor", "overwrite"}:
        raise NotImplementedError(f"delta encoding {encoding!r} not supported")
    checksum_algorithm = metadata.get("checksum_format")
    if not isinstance(checksum_algorithm, str):
        raise ValueError(f"delta has no valid checksum format: {index_path}")
    create_checksum(checksum_algorithm)
    if not all(isinstance(name, str) for name in weight_map):
        raise ValueError(f"invalid delta tensor name in {index_path}")
    if not all(isinstance(filename, str) for filename in weight_map.values()):
        raise ValueError(f"invalid delta shard path in {index_path}")
    expected_delta_names = set(weight_map)
    metadata_wall_s = time.perf_counter() - metadata_started

    source_setup_started = time.perf_counter()
    source_paths = {
        filename: _resolve_delta_path(version_dir, filename)
        for filename in set(weight_map.values())
    }
    for filename, path in source_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"incomplete source version {version_dir}: missing blob {filename}"
            )

    items: list[_DiskDeltaItem] = []
    seen_delta_names = set()
    for delta_filename, delta_file in sorted(source_paths.items()):
        header_len, header = _read_safetensors_header(delta_file)
        expected_checksums = header.get("__metadata__", {})
        if not isinstance(expected_checksums, dict):
            raise ValueError(f"invalid delta checksum metadata: {delta_file}")
        expected_file_names = {
            name for name, filename in weight_map.items() if filename == delta_filename
        }
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
            if not isinstance(info, dict):
                raise ValueError(f"invalid delta tensor metadata for {name!r}")
            offsets = info.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
            ):
                raise ValueError(f"invalid delta offsets for {name!r} in {delta_file}")
            seen_file_names.add(name)
            seen_delta_names.add(name)
            begin, end = offsets
            if begin < 0 or begin > end:
                raise ValueError(f"invalid delta offsets for {name!r} in {delta_file}")
            if info.get("dtype") != "U8" or info.get("shape") != [end - begin]:
                raise ValueError(
                    f"compressed delta tensor {name!r} must be one-dimensional U8"
                )
            try:
                path, offset, nbytes = locations[name]
            except KeyError as exc:
                raise ValueError(
                    f"delta tensor {name!r} is absent from the local checkpoint"
                ) from exc
            expected_checksum = validate_checksum(
                checksum_algorithm,
                expected_checksums.get(name),
            )
            items.append(
                _DiskDeltaItem(
                    encoding=encoding,
                    name=name,
                    source_path=delta_file,
                    source_offset=8 + header_len + begin,
                    compressed_nbytes=end - begin,
                    target_path=path,
                    target_offset=offset,
                    target_nbytes=nbytes,
                    checksum_algorithm=checksum_algorithm,
                    expected_checksum=expected_checksum,
                )
            )
        if seen_file_names != expected_file_names:
            raise ValueError(
                f"delta blob/index tensor mismatch for {delta_file}: "
                f"missing={sorted(expected_file_names - seen_file_names)[:20]} "
                f"extra={sorted(seen_file_names - expected_file_names)[:20]}"
            )
        if set(expected_checksums) != expected_file_names:
            raise ValueError(
                f"delta checksum/index tensor mismatch for {delta_file}: "
                f"missing={sorted(expected_file_names - set(expected_checksums))[:20]} "
                f"extra={sorted(set(expected_checksums) - expected_file_names)[:20]}"
            )
    if seen_delta_names != expected_delta_names:
        raise ValueError(
            f"delta index tensor mismatch for {version_dir}: "
            f"missing={sorted(expected_delta_names - seen_delta_names)[:20]} "
            f"extra={sorted(seen_delta_names - expected_delta_names)[:20]}"
        )
    return _DiskDeltaPlan(
        version=expected_version,
        base_version=base_version,
        encoding=encoding,
        items=tuple(items),
        metadata_wall_s=metadata_wall_s,
        source_setup_wall_s=time.perf_counter() - source_setup_started,
    )


def _apply_delta(
    local_checkpoint_dir: str,
    version_dir: str,
    expected_version: int,
) -> dict:
    """Apply one version's delta in place and fail on checksum mismatch."""

    applied = _read_applied_version(local_checkpoint_dir)
    if applied == expected_version:
        return {
            "operation": "noop",
            "version": expected_version,
            "wall_s": 0.0,
        }
    return _apply_delta_lineage(
        local_checkpoint_dir,
        [(version_dir, expected_version)],
    )


def _apply_delta_lineage(
    local_checkpoint_dir: str,
    versions: list[tuple[str, int]],
) -> dict:
    """Apply an ordered lineage with one target read and write per tensor."""

    if not versions:
        raise ValueError("delta lineage must not be empty")
    started = time.perf_counter()
    applied = _read_applied_version(local_checkpoint_dir)
    expected_base_version = applied
    locations = _tensor_locations(local_checkpoint_dir)
    plans = []
    for version_dir, version in versions:
        plan = _build_delta_plan(
            version_dir,
            version,
            expected_base_version=expected_base_version,
            locations=locations,
        )
        plans.append(plan)
        expected_base_version = version

    operations_by_name: dict[str, list[_DiskDeltaItem]] = {}
    for plan in plans:
        for item in plan.items:
            operations = operations_by_name.setdefault(item.name, [])
            if operations:
                first = operations[0]
                if (
                    item.target_path,
                    item.target_offset,
                    item.target_nbytes,
                ) != (
                    first.target_path,
                    first.target_offset,
                    first.target_nbytes,
                ):
                    raise ValueError(
                        f"target location changed across deltas for {item.name!r}"
                    )
            operations.append(item)

    operations_by_target_path: dict[
        str,
        list[tuple[str, list[_DiskDeltaItem]]],
    ] = {}
    for name, operations in operations_by_name.items():
        operations_by_target_path.setdefault(
            operations[0].target_path,
            [],
        ).append((name, operations))
    for tensor_operations in operations_by_target_path.values():
        tensor_operations.sort(key=lambda value: value[1][0].target_offset)
    target_batches = sorted(
        operations_by_target_path.items(),
        key=lambda value: sum(
            operations[0].target_nbytes for _, operations in value[1]
        ),
        reverse=True,
    )
    all_items = [
        item for operations in operations_by_name.values() for item in operations
    ]

    # Planning above validates the complete lineage before any local byte changes.
    # An interruption after this point leaves no marker, forcing a clean seed.
    _clear_applied_version(local_checkpoint_dir)
    file_cache_limit = _file_descriptor_cache_limit()
    source_files = _FileDescriptorCache(os.O_RDONLY, file_cache_limit)
    target_files = _FileDescriptorCache(os.O_RDWR, file_cache_limit)
    memory_budget = _ByteBudget(_DELTA_APPLY_MEMORY_BYTES)
    mismatches = []
    mismatch_lock = threading.Lock()
    close_wall_s = 0.0

    def apply_tensor(
        name: str,
        operations: list[_DiskDeltaItem],
        target_fd: int,
        decompressor: zstandard.ZstdDecompressor,
    ) -> None:
        first = operations[0]
        has_overwrite = any(item.encoding == "overwrite" for item in operations)
        working_nbytes = first.target_nbytes + _DELTA_STREAM_CHUNK_BYTES
        if has_overwrite:
            working_nbytes += 4 * first.target_nbytes
        with memory_budget.reserve(working_nbytes):
            region = _pread_exact(
                target_fd,
                first.target_offset,
                first.target_nbytes,
            )
            region_view = np.frombuffer(region, dtype=np.uint8)
            for operation in operations:
                with source_files.acquire(operation.source_path) as source_fd:
                    source = PositionalFileRangeReader(
                        source_fd,
                        operation.source_offset,
                        operation.compressed_nbytes,
                        operation.source_path,
                        max_read_bytes=_DELTA_STREAM_CHUNK_BYTES,
                    )
                    with decompressor.stream_reader(source, closefd=False) as reader:
                        if operation.encoding == "xor":
                            position = 0
                            while position < operation.target_nbytes:
                                block = reader.read(
                                    min(
                                        _DELTA_STREAM_CHUNK_BYTES,
                                        operation.target_nbytes - position,
                                    )
                                )
                                if not block:
                                    break
                                end = position + len(block)
                                target = region_view[position:end]
                                np.bitwise_xor(
                                    target,
                                    np.frombuffer(block, dtype=np.uint8),
                                    out=target,
                                )
                                position = end
                            if position != operation.target_nbytes or reader.read(1):
                                raise RuntimeError(
                                    f"decompressed XOR size mismatch for {name!r}: "
                                    f"expected={operation.target_nbytes} "
                                    f"actual={position}"
                                )
                        else:
                            count = int.from_bytes(read_exact(reader, 4), "little")
                            if count > operation.target_nbytes:
                                raise RuntimeError(
                                    f"overwrite payload for {name!r} is invalid"
                                )
                            positions_buffer = read_exact(reader, 4 * count)
                            positions = np.frombuffer(positions_buffer, dtype="<u4")
                            if (
                                count
                                and int(positions.max()) >= operation.target_nbytes
                            ):
                                raise RuntimeError(
                                    f"overwrite payload for {name!r} is invalid"
                                )
                            value_offset = 0
                            while value_offset < count:
                                values = reader.read(
                                    min(
                                        _DELTA_STREAM_CHUNK_BYTES,
                                        count - value_offset,
                                    )
                                )
                                if not values:
                                    break
                                value_end = value_offset + len(values)
                                region_view[positions[value_offset:value_end]] = (
                                    np.frombuffer(values, dtype=np.uint8)
                                )
                                value_offset = value_end
                            if value_offset != count or reader.read(1):
                                raise RuntimeError(
                                    f"overwrite payload size mismatch for {name!r}: "
                                    f"expected={count} actual={value_offset}"
                                )
                    if source.position != operation.compressed_nbytes:
                        raise RuntimeError(
                            f"compressed delta range was not fully consumed for "
                            f"{name!r}: expected={operation.compressed_nbytes} "
                            f"actual={source.position}"
                        )

            final = operations[-1]
            if (
                calculate_checksum(final.checksum_algorithm, region)
                != final.expected_checksum
            ):
                with mismatch_lock:
                    mismatches.append(name)
            else:
                _pwrite_all(target_fd, first.target_offset, region)

    def apply_checkpoint_shard(
        batch: tuple[str, list[tuple[str, list[_DiskDeltaItem]]]],
    ) -> None:
        target_path, tensor_operations = batch
        decompressor = zstandard.ZstdDecompressor()
        with target_files.acquire(target_path, write=True) as target_fd:
            for name, operations in tensor_operations:
                apply_tensor(name, operations, target_fd, decompressor)

    try:
        apply_started = time.perf_counter()
        workers = min(
            _MAX_DELTA_APPLY_WORKERS,
            _available_cpu_count(),
            len(target_batches) or 1,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(apply_checkpoint_shard, target_batches))
        apply_wall_s = time.perf_counter() - apply_started

        flush_started = time.perf_counter()
        target_files.flush()
        flush_wall_s = time.perf_counter() - flush_started
    finally:
        close_started = time.perf_counter()
        try:
            target_files.close()
        finally:
            source_files.close()
        close_wall_s = time.perf_counter() - close_started

    if mismatches:
        raise _ChecksumMismatchError(
            f"checksum mismatch for {len(mismatches)} tensors after applying "
            f"delta lineage v{plans[0].version}..v{plans[-1].version}: "
            f"{sorted(mismatches)[:20]}"
        )

    marker_started = time.perf_counter()
    _write_applied_version(local_checkpoint_dir, plans[-1].version)
    marker_wall_s = time.perf_counter() - marker_started
    encodings = {plan.encoding for plan in plans}
    if len(plans) == 1:
        operation = f"apply_{plans[0].encoding}"
    elif encodings == {"xor"}:
        operation = "apply_xor_chain"
    else:
        operation = "apply_delta_lineage"
    stats = {
        "operation": operation,
        "base_version": plans[0].base_version,
        "version": plans[-1].version,
        "versions": len(plans),
        "delta_tensors": len(operations_by_name),
        "delta_fragments": len(all_items),
        "checkpoint_shards": len(target_batches),
        "delta_shards": len({item.source_path for item in all_items}),
        "compressed_bytes": sum(item.compressed_nbytes for item in all_items),
        "target_tensor_bytes": sum(
            operations[0].target_nbytes for operations in operations_by_name.values()
        ),
        "apply_work_items": len(target_batches),
        "scheduling": "checkpoint_shard_offset_order",
        "workers": workers,
        "io_backend": (
            "pread_pwrite" if len(plans) == 1 else "pread_pwrite_delta_lineage"
        ),
        "working_memory_budget_bytes": _DELTA_APPLY_MEMORY_BYTES,
        "file_descriptor_cache_limit": file_cache_limit,
        "peak_source_file_descriptors": source_files.peak_open_files,
        "peak_target_file_descriptors": target_files.peak_open_files,
        "phases": {
            "metadata_wall_s": round(sum(plan.metadata_wall_s for plan in plans), 6),
            "source_setup_wall_s": round(
                sum(plan.source_setup_wall_s for plan in plans), 6
            ),
            "apply_wall_s": round(apply_wall_s, 6),
            "flush_wall_s": round(flush_wall_s, 6),
            "close_wall_s": round(close_wall_s, 6),
            "marker_wall_s": round(marker_wall_s, 6),
        },
        "wall_s": round(time.perf_counter() - started, 6),
    }
    logger.info(
        "Applied checkpoint delta lineage v%d..v%d: versions=%d tensors=%d "
        "target_bytes=%d wall_time=%.3fs",
        plans[0].version,
        plans[-1].version,
        stats["versions"],
        stats["delta_tensors"],
        stats["target_tensor_bytes"],
        stats["wall_s"],
    )
    return stats
