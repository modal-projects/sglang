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
import resource
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
import zstandard

from sglang.srt.utils import dynamic_import
from sglang.srt.weight_sync.checksum import calculate_checksum, create_checksum
from sglang.srt.weight_sync.file_io import PositionalFileRangeReader, read_exact

logger = logging.getLogger(__name__)

# XOR targets are read and written through ordinary positional I/O instead of
# mmap. An XOR delta must read and checksum every target byte, and positional
# I/O keeps that access sequential instead of making progress page-fault bound.
# Bound compressed, decompressed, and target work buffers across workers. A
# single tensor larger than the budget is admitted alone.
_DELTA_APPLY_MEMORY_BYTES = 8 << 30
_POSITIONAL_IO_CHUNK_BYTES = 64 << 20
_XOR_STREAM_CHUNK_BYTES = 4 << 20

_MAX_SEED_COPY_WORKERS = 8
_MAX_DELTA_APPLY_WORKERS = 32
_MAX_OPEN_FILES_PER_CACHE = 256
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
    checkpoint_source_refresh_hook: Optional[str] = None,
    *,
    drop_cache_after_seed: bool = False,
) -> dict:
    """Bring the host-local checkpoint up to ``target_version``.

    Missing or incomplete source files raise ``FileNotFoundError`` without
    reseeding. A checksum mismatch on a complete source is treated as corrupt
    local state and gets one replay from a clean seed. ``drop_cache_after_seed``
    releases each durable local shard instead of retaining it for a consumer.
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
            # A co-located rank already brought this host up to the target;
            # don't refresh the source again (avoids concurrent-refresh churn).
            return {
                "operation": "noop",
                "initial_version": applied,
                "target_version": target_version,
                "lock_wait_s": round(lock_wait_s, 6),
                "source_refresh_wall_s": 0.0,
                "wall_s": round(time.perf_counter() - started, 6),
            }
        # A custom source may require an explicit visibility refresh.
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
                drop_cache_after_seed=drop_cache_after_seed,
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
                drop_cache_after_seed=drop_cache_after_seed,
            )
            stats["reseed_after_failed_apply"] = True
        stats["initial_version"] = applied
        stats["target_version"] = target_version
        stats["lock_wait_s"] = round(lock_wait_s, 6)
        stats["source_refresh_wall_s"] = round(source_refresh_wall_s, 6)
        stats["wall_s"] = round(time.perf_counter() - started, 6)
        return stats


def _materialize_locked(
    local_checkpoint_dir: str,
    base_checkpoint_dir: str,
    checkpoint_source_dir: str,
    target_version: int,
    reseed: bool,
    drop_cache_after_seed: bool,
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
        _reset_checkpoint(
            seed_dir,
            local_checkpoint_dir,
            start,
            drop_cache_after_copy=drop_cache_after_seed,
        )
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
            apply_stats = _apply_delta_chain(
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


def refresh_checkpoint_source(
    checkpoint_source_dir: str,
    target_version: int,
    checkpoint_source_refresh_hook: Optional[str],
) -> None:
    """Make a published version visible before reading any of its metadata."""

    if target_version > 0 and checkpoint_source_refresh_hook:
        dynamic_import(checkpoint_source_refresh_hook)(
            checkpoint_source_dir, target_version
        )


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


def _read_applied_version(local_checkpoint_dir: str) -> Optional[int]:
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


def drop_checkpoint_page_cache(checkpoint_dir: str) -> int:
    """Release clean safetensors pages after a checkpoint consumer is done."""

    paths = glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))
    for path in paths:
        _drop_page_cache(path)
    return len(paths)


def _reset_checkpoint(
    src_dir: str,
    local_checkpoint_dir: str,
    version: int,
    *,
    drop_cache_after_copy: bool,
) -> None:
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
        # The immutable source no longer needs to occupy the page cache.
        _drop_page_cache(entry.path)
        if drop_cache_after_copy:
            _drop_page_cache(dst)
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
    for path in glob.glob(os.path.join(ckpt_dir, "*.safetensors")):
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        for name, info in header.items():
            if name == "__metadata__":
                continue
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
    for path in glob.glob(os.path.join(src_dir, "*.safetensors")):
        _, headers[os.path.basename(path)] = _read_safetensors_header(path)
    if not headers:
        raise FileNotFoundError(f"full checkpoint has no safetensors files: {src_dir}")
    if version == 0:
        return

    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    try:
        with open(index_path) as file:
            index = json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"published full checkpoint has no manifest: {index_path}"
        ) from exc
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
    checksum_algorithm: str
    expected_checksum: str


@dataclass(frozen=True)
class _DiskDeltaPlan:
    version_dir: str
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
    """Reserve descriptors for the server while bounding long delta lineages."""

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


def _build_delta_plan(
    local_checkpoint_dir: str,
    version_dir: str,
    expected_version: int,
    expected_base_version: Optional[int],
    locations: Optional[dict] = None,
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
    # Reject an unsupported format before invalidating the local version marker.
    create_checksum(checksum_algorithm)
    if locations is None:
        locations = _tensor_locations(local_checkpoint_dir)
    metadata_wall_s = time.perf_counter() - metadata_started

    source_setup_started = time.perf_counter()
    expected_delta_names = set(weight_map)
    if any(not isinstance(filename, str) for filename in weight_map.values()):
        raise ValueError(f"invalid delta filename in {index_path}")
    expected_delta_files = set(weight_map.values())
    for filename in expected_delta_files:
        if not os.path.exists(os.path.join(version_dir, filename)):
            raise FileNotFoundError(
                f"incomplete source version {version_dir}: missing blob {filename}"
            )

    items = []
    seen_delta_names = set()
    for delta_filename in sorted(expected_delta_files):
        delta_file = os.path.join(version_dir, delta_filename)
        header_len, header = _read_safetensors_header(delta_file)
        expected_checksums = header.get("__metadata__", {})
        if not isinstance(expected_checksums, dict):
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
            if not isinstance(info, dict):
                raise ValueError(f"invalid delta tensor metadata for {name!r}")
            offsets = info.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
            ):
                raise ValueError(f"invalid delta offsets for {name!r} in {delta_file}")
            begin, end = offsets
            if begin < 0 or begin > end:
                raise ValueError(f"invalid delta offsets for {name!r} in {delta_file}")
            try:
                target_path, target_offset, target_nbytes = locations[name]
            except KeyError as exc:
                raise ValueError(
                    f"delta tensor {name!r} is absent from the local checkpoint"
                ) from exc
            expected_checksum = expected_checksums.get(name)
            if not isinstance(expected_checksum, str) or not expected_checksum:
                raise ValueError(f"delta tensor {name!r} has no target checksum")
            items.append(
                _DiskDeltaItem(
                    name=name,
                    source_path=delta_file,
                    source_offset=8 + header_len + begin,
                    compressed_nbytes=end - begin,
                    target_path=target_path,
                    target_offset=target_offset,
                    target_nbytes=target_nbytes,
                    checksum_algorithm=checksum_algorithm,
                    expected_checksum=expected_checksum,
                )
            )
            seen_file_names.add(name)
            seen_delta_names.add(name)
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
    return _DiskDeltaPlan(
        version_dir=version_dir,
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
    """Apply one version's delta in place and fail on any checksum mismatch."""

    started = time.perf_counter()
    applied = _read_applied_version(local_checkpoint_dir)
    if applied == expected_version:
        return {
            "operation": "noop",
            "version": expected_version,
            "wall_s": round(time.perf_counter() - started, 6),
        }
    delta = _build_delta_plan(
        local_checkpoint_dir,
        version_dir,
        expected_version,
        expected_base_version=applied,
    )
    return _apply_delta_plan(local_checkpoint_dir, delta, started=started)


def _apply_delta_chain(
    local_checkpoint_dir: str,
    versions: list[tuple[str, int]],
) -> dict:
    """Apply an ordered delta lineage, folding consecutive XOR versions."""

    started = time.perf_counter()
    expected_base_version = _read_applied_version(local_checkpoint_dir)
    locations = _tensor_locations(local_checkpoint_dir)
    deltas = []
    for version_dir, version in versions:
        delta = _build_delta_plan(
            local_checkpoint_dir,
            version_dir,
            version,
            expected_base_version=expected_base_version,
            locations=locations,
        )
        deltas.append(delta)
        expected_base_version = version

    if all(delta.encoding == "xor" for delta in deltas):
        return _apply_xor_delta_chain(local_checkpoint_dir, deltas, started=started)

    xor_run = []
    for delta in deltas:
        if delta.encoding == "xor":
            xor_run.append(delta)
            continue
        if xor_run:
            _apply_xor_delta_chain(
                local_checkpoint_dir,
                xor_run,
                started=time.perf_counter(),
            )
            xor_run = []
        _apply_delta_plan(local_checkpoint_dir, delta)
    if xor_run:
        _apply_xor_delta_chain(
            local_checkpoint_dir,
            xor_run,
            started=time.perf_counter(),
        )
    return {
        "operation": "apply_deltas",
        "versions": len(deltas),
        "wall_s": round(time.perf_counter() - started, 6),
    }


def _apply_xor_delta_chain(
    local_checkpoint_dir: str,
    deltas: list[_DiskDeltaPlan],
    *,
    started: float,
) -> dict:
    """Fold an XOR lineage into one read and write of each target tensor."""

    applied = _read_applied_version(local_checkpoint_dir)
    if applied != deltas[0].base_version:
        raise RuntimeError(
            f"out-of-order delta: local at {applied}, "
            f"delta builds on {deltas[0].base_version}"
        )
    operations_by_name: dict[str, list[_DiskDeltaItem]] = {}
    for delta in deltas:
        for item in delta.items:
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

    # All lineage metadata, source blobs, and target ranges are valid before the
    # first mutation. An interruption after this point leaves no version marker,
    # so a retry starts from a clean checkpoint seed.
    _clear_applied_version(local_checkpoint_dir)
    all_items = [
        item for operations in operations_by_name.values() for item in operations
    ]
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
        tensor_operations.sort(key=lambda item: item[1][0].target_offset)
    target_batches = sorted(
        operations_by_target_path.items(),
        key=lambda item: sum(operations[0].target_nbytes for _, operations in item[1]),
        reverse=True,
    )
    source_paths = {item.source_path for item in all_items}
    target_paths = {item.target_path for item in all_items}
    file_cache_limit = _file_descriptor_cache_limit()
    source_files = _FileDescriptorCache(os.O_RDONLY, file_cache_limit)
    target_files = _FileDescriptorCache(os.O_RDWR, file_cache_limit)
    mismatches = []
    mismatch_lock = threading.Lock()
    close_wall_s = 0.0
    try:
        memory_budget = _ByteBudget(_DELTA_APPLY_MEMORY_BYTES)

        def apply_tensor(
            name: str,
            operations: list[_DiskDeltaItem],
            target_fd: int,
            decompressor: zstandard.ZstdDecompressor,
        ) -> None:
            first = operations[0]
            working_nbytes = first.target_nbytes + _XOR_STREAM_CHUNK_BYTES
            with memory_budget.reserve(working_nbytes):
                region = _pread_exact(
                    target_fd,
                    first.target_offset,
                    first.target_nbytes,
                )
                region_view = np.frombuffer(region, dtype=np.uint8)
                final = operations[-1]
                hasher = create_checksum(final.checksum_algorithm)
                for operation_index, operation in enumerate(operations):
                    is_final = operation_index == len(operations) - 1
                    with source_files.acquire(operation.source_path) as source_fd:
                        source = PositionalFileRangeReader(
                            source_fd,
                            operation.source_offset,
                            operation.compressed_nbytes,
                            operation.source_path,
                            max_read_bytes=_XOR_STREAM_CHUNK_BYTES,
                        )
                        with decompressor.stream_reader(
                            source,
                            closefd=False,
                        ) as reader:
                            position = 0
                            while position < operation.target_nbytes:
                                block = reader.read(
                                    min(
                                        _XOR_STREAM_CHUNK_BYTES,
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
                                if is_final:
                                    hasher.update(target)
                                position = end
                            if position != operation.target_nbytes or reader.read(1):
                                raise RuntimeError(
                                    f"decompressed delta size mismatch for {name!r}: "
                                    f"expected={operation.target_nbytes} "
                                    f"actual={position}"
                                )
                        if source.position != operation.compressed_nbytes:
                            raise RuntimeError(
                                f"compressed delta range was not fully consumed "
                                f"for {name!r}: "
                                f"expected={operation.compressed_nbytes} "
                                f"actual={source.position}"
                            )

                if hasher.hexdigest() != final.expected_checksum:
                    with mismatch_lock:
                        mismatches.append(name)
                else:
                    _pwrite_all(
                        target_fd,
                        final.target_offset,
                        region,
                    )

        def apply_checkpoint_shard(
            batch: tuple[str, list[tuple[str, list[_DiskDeltaItem]]]],
        ) -> None:
            target_path, tensor_operations = batch
            decompressor = zstandard.ZstdDecompressor()
            with target_files.acquire(target_path, write=True) as target_fd:
                for name, operations in tensor_operations:
                    apply_tensor(name, operations, target_fd, decompressor)

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
        scope = (
            f"delta v{deltas[-1].version}"
            if len(deltas) == 1
            else f"delta chain v{deltas[0].version}..v{deltas[-1].version}"
        )
        raise _ChecksumMismatchError(
            f"checksum mismatch for {len(mismatches)} tensors after applying {scope}: "
            f"{sorted(mismatches)[:20]}"
        )
    marker_started = time.perf_counter()
    _write_applied_version(local_checkpoint_dir, deltas[-1].version)
    marker_wall_s = time.perf_counter() - marker_started
    stats = {
        "operation": "apply_xor" if len(deltas) == 1 else "apply_xor_chain",
        "version": deltas[-1].version,
        "delta_tensors": len(operations_by_name),
        "checkpoint_shards": len(target_paths),
        "delta_shards": len(source_paths),
        "compressed_bytes": sum(item.compressed_nbytes for item in all_items),
        "target_tensor_bytes": sum(
            operations[0].target_nbytes for operations in operations_by_name.values()
        ),
        "apply_work_items": len(target_batches),
        "scheduling": "checkpoint_shard_offset_order",
        "workers": workers,
        "io_backend": (
            "pread_pwrite" if len(deltas) == 1 else "pread_pwrite_xor_chain"
        ),
        "working_memory_budget_bytes": _DELTA_APPLY_MEMORY_BYTES,
        "file_descriptor_cache_limit": file_cache_limit,
        "peak_source_file_descriptors": source_files.peak_open_files,
        "peak_target_file_descriptors": target_files.peak_open_files,
        "phases": {
            "metadata_wall_s": round(sum(delta.metadata_wall_s for delta in deltas), 6),
            "source_setup_wall_s": round(
                sum(delta.source_setup_wall_s for delta in deltas), 6
            ),
            "prefetch_wall_s": 0.0,
            "apply_wall_s": round(apply_wall_s, 6),
            "flush_wall_s": round(flush_wall_s, 6),
            "close_wall_s": round(close_wall_s, 6),
            "marker_wall_s": round(marker_wall_s, 6),
        },
        "wall_s": round(time.perf_counter() - started, 6),
    }
    if len(deltas) == 1:
        logger.info(
            "Applied checkpoint delta v%d: tensors=%d target_bytes=%d "
            "wall_time=%.3fs",
            stats["version"],
            stats["delta_tensors"],
            stats["target_tensor_bytes"],
            stats["wall_s"],
        )
    else:
        stats.update(
            {
                "base_version": deltas[0].base_version,
                "versions": len(deltas),
                "delta_fragments": len(all_items),
            }
        )
        logger.info(
            "Applied checkpoint XOR delta chain v%d..v%d: versions=%d "
            "tensors=%d target_bytes=%d wall_time=%.3fs",
            deltas[0].version,
            deltas[-1].version,
            stats["versions"],
            stats["delta_tensors"],
            stats["target_tensor_bytes"],
            stats["wall_s"],
        )
    return stats


def _apply_delta_plan(
    local_checkpoint_dir: str,
    delta: _DiskDeltaPlan,
    *,
    started: Optional[float] = None,
) -> dict:
    """Apply one validated delta and publish its version marker."""

    if started is None:
        started = time.perf_counter()
    applied = _read_applied_version(local_checkpoint_dir)
    if applied != delta.base_version:
        raise RuntimeError(
            f"out-of-order delta: local at {applied}, "
            f"delta builds on {delta.base_version}"
        )
    if delta.encoding == "xor":
        return _apply_xor_delta_chain(
            local_checkpoint_dir,
            [delta],
            started=started,
        )

    _clear_applied_version(local_checkpoint_dir)
    items = delta.items
    source_files = {}
    target_files = {}
    mismatches = []
    mismatch_lock = threading.Lock()
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
        open_mmaps = {
            path: (
                (file := open(path, "r+b")),
                mmap.mmap(file.fileno(), 0),
            )
            for path in target_files
        }
        prefetch_started = time.perf_counter()
        for _, mapped_file in open_mmaps.values():
            try:
                mapped_file.madvise(mmap.MADV_WILLNEED)
            except (OSError, AttributeError, ValueError):
                pass
        prefetch_wall_s = time.perf_counter() - prefetch_started

        def apply_overwrite(item: _DiskDeltaItem) -> None:
            max_payload_nbytes = 4 + 5 * item.target_nbytes
            with memory_budget.reserve(max_payload_nbytes):
                source = PositionalFileRangeReader(
                    source_files[item.source_path],
                    item.source_offset,
                    item.compressed_nbytes,
                    item.source_path,
                    max_read_bytes=_XOR_STREAM_CHUNK_BYTES,
                )
                with zstandard.ZstdDecompressor().stream_reader(
                    source,
                    closefd=False,
                ) as reader:
                    count = int.from_bytes(read_exact(reader, 4), "little")
                    if count > item.target_nbytes:
                        raise RuntimeError(
                            f"overwrite payload for {item.name!r} is invalid"
                        )
                    payload = read_exact(reader, 5 * count)
                    if reader.read(1):
                        raise RuntimeError(
                            f"overwrite payload for {item.name!r} is oversized"
                        )
                if source.position != item.compressed_nbytes:
                    raise RuntimeError(
                        f"compressed delta range was not fully consumed for "
                        f"{item.name!r}: expected={item.compressed_nbytes} "
                        f"actual={source.position}"
                    )
                positions = np.frombuffer(payload, dtype="<u4", count=count)
                values = np.frombuffer(
                    payload,
                    dtype=np.uint8,
                    count=count,
                    offset=4 * count,
                )
                if values.size != count or (
                    count and int(positions.max()) >= item.target_nbytes
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
                if (
                    calculate_checksum(item.checksum_algorithm, region)
                    != item.expected_checksum
                ):
                    with mismatch_lock:
                        mismatches.append(item.name)

        apply_started = time.perf_counter()
        workers = min(
            _MAX_DELTA_APPLY_WORKERS,
            _available_cpu_count(),
            len(items) or 1,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(apply_overwrite, items))
        apply_wall_s = time.perf_counter() - apply_started

        flush_started = time.perf_counter()
        for _, mapped_file in open_mmaps.values():
            mapped_file.flush()
        flush_wall_s = time.perf_counter() - flush_started
    finally:
        close_started = time.perf_counter()
        if "open_mmaps" in locals():
            for file, mapped_file in open_mmaps.values():
                mapped_file.close()
                file.close()
        for fd in target_files.values():
            os.close(fd)
        for fd in source_files.values():
            os.close(fd)
        close_wall_s = time.perf_counter() - close_started

    if mismatches:
        raise _ChecksumMismatchError(
            f"checksum mismatch for {len(mismatches)} tensors after applying "
            f"{delta.version_dir}: {sorted(mismatches)[:20]}"
        )
    marker_started = time.perf_counter()
    _write_applied_version(local_checkpoint_dir, delta.version)
    marker_wall_s = time.perf_counter() - marker_started
    stats = {
        "operation": f"apply_{delta.encoding}",
        "version": delta.version,
        "delta_tensors": len(items),
        "checkpoint_shards": len(target_files),
        "compressed_bytes": sum(item.compressed_nbytes for item in items),
        "target_tensor_bytes": sum(item.target_nbytes for item in items),
        "workers": workers,
        "io_backend": "mmap_sparse_overwrite",
        "working_memory_budget_bytes": _DELTA_APPLY_MEMORY_BYTES,
        "phases": {
            "metadata_wall_s": round(delta.metadata_wall_s, 6),
            "source_setup_wall_s": round(delta.source_setup_wall_s, 6),
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
