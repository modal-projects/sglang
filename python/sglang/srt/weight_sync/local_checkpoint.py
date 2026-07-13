"""Host-local pull of published weights (the /pull_weights endpoint).

A trainer publishes each weight sync as a version directory ``weight_v{N:06d}/``
under a shared ``source_dir``. Each version is a canonical HF checkpoint
directory of one of two kinds, distinguished by its index metadata:

- **full**: an ordinary checkpoint. Pulling it copies it into the host-local
  ``local_checkpoint_dir``, replacing whatever is there — no history needed.
- **delta** (index metadata carries ``delta_encoding``): safetensors files
  holding zstd-compressed per-tensor diffs against version N-1, plus per-tensor
  checksums of the new state. Pulling it patches the local checkpoint in place.

Version 0 is the engine's own base checkpoint (``model_path``). Every host of a
(possibly multi-node) deployment runs the same pull; the engine then reloads the
local checkpoint through the ordinary ``update_weights_from_disk`` path.

``pull()`` is safe to call concurrently from every scheduler rank on a host: a
per-host file lock serializes the work and an applied-version marker makes the
extra calls no-ops.

A pull that dies mid-mutation (preemption, power loss) never wedges the host.
The next pull re-applies the interrupted version; a changed tensor that was
already XORed reverts under the second XOR and so fails its checksum, which
triggers one reseed from the newest full version (or the engine's base) plus a
replay of the delta chain. The delta's dirty pages are msync'd before the
applied-version marker is written, so the marker can never become durable over
bytes that never reached disk. Only a failure on the fresh reseed raises.
"""

from __future__ import annotations

import fcntl
import glob
import importlib
import json
import logging
import mmap
import os
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Optional

import numpy as np
import zstandard

logger = logging.getLogger(__name__)

# The delta-apply phases (decompress, XOR/scatter, checksum) are memory-bandwidth
# bound and release the GIL, so a thread pool over tensors recovers the
# bandwidth one thread leaves idle.
NUM_WORKERS = min(32, (os.cpu_count() or 8))

# Per-checkpoint dir holding the applied-version marker and the pull lock.
SYNC_DIR = ".weight_sync"


def pull(
    local_checkpoint_dir: str,
    base_dir: str,
    source_dir: str,
    target_version: int,
    pre_read_hook: Optional[str] = None,
) -> None:
    """Bring the host-local checkpoint up to ``target_version``.

    Seeds from the newest full checkpoint at or below the target — the engine's
    own pristine base (``base_dir``) for a pure-delta stream, a published full
    version otherwise — then applies the remaining deltas in order. A local
    checkpoint already past the seed point just continues its delta chain.

    Runs under a per-host lock. Every co-located TP rank calls this, but only the
    lock winner reloads + applies; a rank that finds the checkpoint already at or
    past the target returns immediately. So the shared volume is reloaded at most
    once per host per pull — never by several ranks at once, which can flip one
    rank's mount to a stale snapshot *mid-apply* and corrupt the result.

    Each delta blob is read whole-file into memory and size-verified against its
    own safetensors header before it is applied: the XOR reads from that in-memory
    buffer, so it never streams from an eventually-consistent mount that can change
    under it mid-apply. Two failure classes stay distinct:

      * A missing or incomplete *source* version raises ``FileNotFoundError`` and
        stops — we never reseed to paper over a not-yet-materialized source; the
        caller reloads + retries.
      * A checksum mismatch on staged, complete bytes == corrupt *local* state (a
        torn mid-write apply, bit rot). That triggers one reseed from the pristine
        base plus a replay of the chain; a failure on the fresh state re-raises
        (fail loud, never serve bad weights).
    """
    with _pull_lock(local_checkpoint_dir):
        applied = _read_applied_version(local_checkpoint_dir)
        if applied is not None and applied >= target_version:
            # A co-located rank already brought this host up to the target;
            # don't reload the volume again (avoids concurrent-reload churn).
            return
        # Object-store-backed volumes lack cross-host read-after-write
        # consistency: the publisher's files appear here only after an explicit
        # refresh, which the deployment supplies as this hook. POSIX shared
        # filesystems (NFS, Lustre, ...) pass no hook and need none.
        if target_version > 0 and pre_read_hook:
            _load_hook(pre_read_hook)(source_dir, target_version)
        try:
            _pull_locked(
                local_checkpoint_dir, base_dir, source_dir, target_version, reseed=False
            )
        except FileNotFoundError:
            # A source version is missing or not fully materialized — a readiness
            # failure the caller owns, not local corruption. Reseeding cannot
            # conjure absent bytes, so record what the mount shows and fail fast;
            # the caller reloads and retries.
            _log_pull_not_found(source_dir, target_version)
            raise
        except Exception:
            # A checksum mismatch on staged, complete bytes == corrupt local
            # state (incomplete sources are reclassified to FileNotFoundError
            # above and never reach here). Reseed from the pristine base and
            # replay once; a failure on that fresh state re-raises.
            logger.exception(
                "pull to v%d failed on staged sources; reseeding from base and replaying",
                target_version,
            )
            _pull_locked(
                local_checkpoint_dir, base_dir, source_dir, target_version, reseed=True
            )


def _pull_locked(
    local_checkpoint_dir: str,
    base_dir: str,
    source_dir: str,
    target_version: int,
    reseed: bool,
) -> None:
    # A torn local state (reseed=True) is treated like a fresh host: the
    # applied-version marker can't be trusted over partially-mutated files.
    applied = None if reseed else _read_applied_version(local_checkpoint_dir)
    # Scan back from the target for the newest full version. Stop at the
    # local state — below it a reset can never be needed (or, on a fresh
    # host, at 0 = the engine's base).
    floor = applied if applied is not None else 0
    start = target_version
    while start > floor and _is_delta(_version_dir(source_dir, start)):
        start -= 1
    if applied is None or start > applied:
        seed_dir = base_dir if start == 0 else _version_dir(source_dir, start)
        _reset_checkpoint(seed_dir, local_checkpoint_dir, start)
    else:
        start = applied
    for version in range(start + 1, target_version + 1):
        _apply_delta(local_checkpoint_dir, _version_dir(source_dir, version))


def _load_hook(path: str):
    module_path, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module_path), name)


def _log_pull_not_found(source_dir: str, target_version: int) -> None:
    """A pull reached the engine with a missing/incomplete source version — the
    pre-read hook is supposed to block until the version is fully materialized,
    so this should be rare. Record what the mount actually shows (visible version
    dirs, whether the target is a dir and its contents, the `latest` pointer) to
    tell a still-propagating source from a path/logic bug."""
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
        "[pull-diag] MISSING v%d: mount shows versions=%s ; isdir(%s)=%s contents=%s ; "
        "latest_pointer=%s",
        target_version,
        versions,
        vdir,
        os.path.isdir(vdir),
        target_contents,
        _read_pointer(source_dir),
    )


def _read_pointer(source_dir: str):
    # The `latest` pointer lives at the transport root (source_dir's parent).
    for path in (
        os.path.join(source_dir, "latest"),
        os.path.join(os.path.dirname(source_dir.rstrip("/")), "latest"),
    ):
        try:
            with open(path) as f:
                return f"{path}={f.read().strip()!r}"
        except OSError:
            continue
    return "<no latest pointer found>"


def _version_dir(source_dir: str, version: int) -> str:
    return os.path.join(source_dir, f"weight_v{version:06d}")


def _is_delta(version_dir: str) -> bool:
    """A version is a delta iff its index metadata declares an encoding; an
    ordinary HF checkpoint (with or without an index) is a full version."""
    if not os.path.isdir(version_dir):
        raise FileNotFoundError(f"published weight version missing: {version_dir}")
    try:
        with open(os.path.join(version_dir, "model.safetensors.index.json")) as f:
            return "delta_encoding" in json.load(f).get("metadata", {})
    except FileNotFoundError:
        return False


class _Adler32:
    """adler32 behind the incremental .update / .hexdigest interface the hash objects expose."""

    def __init__(self):
        self._value = 1

    def update(self, data) -> None:
        self._value = zlib.adler32(data, self._value)

    def hexdigest(self) -> str:
        return f"{self._value:08x}"


def _new_hasher(algorithm: str):
    if algorithm == "xxh3-128":
        import xxhash

        return xxhash.xxh3_128()
    if algorithm == "blake3":
        import blake3

        return blake3.blake3()
    if algorithm == "adler32":
        return _Adler32()
    raise KeyError(f"unknown checksum algorithm {algorithm!r}")


def _checksum(algorithm: str, buf) -> str:
    hasher = _new_hasher(algorithm)
    hasher.update(buf)
    return hasher.hexdigest()


@contextmanager
def _pull_lock(local_checkpoint_dir: str):
    sync = os.path.join(local_checkpoint_dir, SYNC_DIR)
    os.makedirs(sync, exist_ok=True)
    with open(os.path.join(sync, "lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read_applied_version(local_checkpoint_dir: str) -> Optional[int]:
    try:
        with open(os.path.join(local_checkpoint_dir, SYNC_DIR, "state.json")) as f:
            return int(json.load(f)["version"])
    except FileNotFoundError:
        return None


def _write_applied_version(local_checkpoint_dir: str, version: int) -> None:
    path = os.path.join(local_checkpoint_dir, SYNC_DIR, "state.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"version": f"{version:06d}"}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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
    logger.info(
        "Pulling full checkpoint v%d %s -> %s", version, src_dir, local_checkpoint_dir
    )
    os.makedirs(local_checkpoint_dir, exist_ok=True)
    src_files = [entry for entry in os.scandir(src_dir) if entry.is_file()]
    for entry in src_files:
        dst = os.path.join(local_checkpoint_dir, entry.name)
        # Seeding from a dir that overlaps the local checkpoint (e.g. base_dir ==
        # local_checkpoint_dir, or a shared mount surfacing the same inode) leaves
        # the file already in place, so skip it.
        if os.path.exists(dst) and os.path.samefile(entry.path, dst):
            continue
        # Whole-file read like the delta apply: fetches the lazy mount in one
        # shot; shutil.copy2 reads in 64 KiB chunks, ~40x slower on this mount.
        # The size check below fails loud if the mount served a short read.
        with open(entry.path, "rb") as f:
            data = f.read()
        with open(dst, "wb") as f:
            f.write(data)
        # don't let the source evict the local copy we keep resident
        _drop_page_cache(entry.path)
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


def _safetensors_size(blob: bytes) -> Optional[int]:
    """Total byte length a safetensors payload must have per its own header — 8
    (header-length prefix) + header + the largest tensor end-offset. Returns None
    if the bytes are too short to even hold the declared header, the signal of a
    torn or not-yet-fully-materialized read."""
    if len(blob) < 8:
        return None
    header_len = struct.unpack("<Q", blob[:8])[0]
    if len(blob) < 8 + header_len:
        return None
    try:
        header = json.loads(blob[8 : 8 + header_len])
    except ValueError:
        return None
    end = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        end = max(end, info["data_offsets"][1])
    return 8 + header_len + end


def _apply_delta(local_checkpoint_dir: str, version_dir: str) -> None:
    """Apply one version's delta in place: decompress + apply + checksum each tensor across a thread
    pool (each writes a distinct mmap region, so the writes don't conflict). Any mismatch raises.
    """
    with open(os.path.join(version_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    meta = index["metadata"]
    applied = _read_applied_version(local_checkpoint_dir)
    if applied == int(meta["version"]):
        return
    # Validate the source before reading it: every blob the index names must be
    # present, or a half-propagated version would apply only the blobs that made
    # it and report the rest as a checksum mismatch (misread as corruption). A
    # missing blob is a not-ready source, so raise FileNotFoundError — the pull
    # fails fast and the caller reloads + retries instead of reseeding.
    for blob in sorted(set(index.get("weight_map", {}).values())):
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
    locations = _tensor_locations(local_checkpoint_dir)
    open_mmaps = {}
    mismatches = []
    lock = threading.Lock()
    file_bytes = []  # keep alive: items hold zero-copy views into these
    items = []  # (name, compressed_view, path, offset, nbytes, want_checksum)
    try:
        for delta_file in sorted(glob.glob(os.path.join(version_dir, "*.safetensors"))):
            with open(delta_file, "rb") as f:
                # One whole-file read: fetches the lazy volume mount in a single
                # shot, and the XOR below reads from this in-memory buffer, so the
                # apply is immune to the mount changing under it. (No on-disk
                # staging step — the buffer IS the stable snapshot.)
                blob = f.read()
            # Verify the whole file arrived (its length matches its own
            # safetensors header) BEFORE building any item: a short read == a
            # not-yet-materialized source, so fail fast (FileNotFoundError ->
            # caller reloads + retries) rather than XOR a partial delta and
            # corrupt the checkpoint.
            expected = _safetensors_size(blob)
            if expected is None or len(blob) != expected:
                raise FileNotFoundError(
                    f"incomplete source blob {delta_file}: {len(blob)}B, header "
                    f"declares {expected}B (not fully materialized)"
                )
            file_bytes.append(blob)
            (header_len,) = struct.unpack("<Q", blob[:8])
            header = json.loads(blob[8 : 8 + header_len])
            want_checksums = header.get("__metadata__", {})
            view = memoryview(blob)
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                begin, end = info["data_offsets"]
                path, offset, nbytes = locations[name]
                if path not in open_mmaps:
                    fh = open(path, "r+b")
                    open_mmaps[path] = (fh, mmap.mmap(fh.fileno(), 0))
                data_start = 8 + header_len
                items.append(
                    (
                        name,
                        view[data_start + begin : data_start + end],
                        path,
                        offset,
                        nbytes,
                        want_checksums.get(name),
                    )
                )

        # prefetch into page cache (evicted during the rollout) so the apply
        # doesn't fault from cold storage
        for _, mm in open_mmaps.values():
            try:
                mm.madvise(mmap.MADV_WILLNEED)
            except (OSError, AttributeError, ValueError):
                pass

        def apply_xor(item) -> None:
            name, compressed, path, offset, nbytes, want = item
            region = np.ndarray(
                (nbytes,), dtype=np.uint8, buffer=open_mmaps[path][1], offset=offset
            )
            hasher = _new_hasher(algorithm)
            reader = zstandard.ZstdDecompressor().stream_reader(compressed)
            pos = 0
            # 2 MB chunks stay L2-resident across decompress -> XOR -> checksum
            while pos < nbytes:
                block = reader.read(min(2 << 20, nbytes - pos))
                if not block:
                    break
                chunk = np.frombuffer(block, dtype=np.uint8)
                region[pos : pos + chunk.size] ^= chunk
                hasher.update(region[pos : pos + chunk.size])
                pos += chunk.size
            if hasher.hexdigest() != want:
                with lock:
                    mismatches.append(name)

        def apply_overwrite(item) -> None:
            name, compressed, path, offset, nbytes, want = item
            delta = np.frombuffer(
                zstandard.ZstdDecompressor().decompress(compressed), dtype=np.uint8
            )
            region = np.ndarray(
                (nbytes,), dtype=np.uint8, buffer=open_mmaps[path][1], offset=offset
            )
            count = int.from_bytes(delta[:4], "little")
            positions = np.frombuffer(delta[4 : 4 + 4 * count], dtype="<u4")
            region[positions] = delta[4 + 4 * count :]
            if _checksum(algorithm, region) != want:
                with lock:
                    mismatches.append(name)

        if encoding == "xor":
            apply_tensor = apply_xor
        elif encoding == "overwrite":
            apply_tensor = apply_overwrite
        else:
            raise NotImplementedError(f"delta encoding {encoding!r} not supported")
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
            list(pool.map(apply_tensor, items))
        # msync BEFORE the applied-version marker: the marker must never become
        # durable over data pages that never made it to disk, or a power loss
        # after a "successful" apply would silently serve stale bytes forever.
        # Only the delta's dirty pages get written, so the cost is O(delta).
        for _, mm in open_mmaps.values():
            mm.flush()
    finally:
        for fh, mm in open_mmaps.values():
            mm.close()
            fh.close()
    if mismatches:
        raise RuntimeError(
            f"checksum mismatch for {len(mismatches)} tensors after applying {version_dir}: "
            f"{sorted(mismatches)[:20]}"
        )
    _write_applied_version(local_checkpoint_dir, int(meta["version"]))
