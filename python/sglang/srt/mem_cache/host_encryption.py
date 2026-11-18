"""Encryption at rest for the HiCache host (L2) pool.

GPU transfer kernels read and write the host pool directly, so plaintext KV
necessarily exists in host memory while a transfer runs. This module keeps a
page as AES-256-GCM ciphertext for the rest of its life in the pool:

- after a device-to-host backup completes, its pages are encrypted in place;
- before a host-to-device load is submitted, its pages are decrypted in place,
  and re-encrypted once the load completes.

The key is generated per process and never leaves it, which matches the
lifetime of the host pool. Tags live in process memory outside the pool. A
page's nonce is derived from its pool, its slot and a counter that advances
every time the slot is encrypted, so a page that is moved, replayed or modified
fails authentication.

Encryption runs in a small C library built on first use (see
host_encryption_native.c). A transfer is handed to it as arrays of page
numbers, so neither the scheduler thread nor the worker threads spend time in
the interpreter per page.
"""

from __future__ import annotations

import collections
import ctypes
import ctypes.util
import hashlib
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_TAG_BYTES = 16
# A page larger than this is encrypted as several independent units, so that
# one large page (a Mamba state is tens of MB) still spreads across threads.
_UNIT_BYTES = 4 << 20
# Stashed on a device copy of host indices so page bookkeeping never needs a
# device-to-host readback (which would synchronize the transfer stream).
CPU_INDICES_ATTR = "_hicache_cpu_host_indices"

# Page states.
_PLAIN = 0  # never encrypted: unused, or written before this engine saw it
_PENDING = 1  # plaintext: backup kernel submitted, not yet encrypted
_SEALED = 2  # ciphertext
_OPEN = 3  # plaintext: decrypted for one or more in-flight loads
_BUSY = 4  # a worker is transforming the page right now

_ITEM = np.dtype(
    [("page", "<u8"), ("generation", "<u8"), ("unit", "<u4"), ("reserved", "<u4")]
)
_BUFFER = np.dtype([("base", "<u8"), ("stride", "<u8"), ("size", "<u8")])


class HostEncryptionError(RuntimeError):
    pass


def _build_native_library() -> Path:
    source = Path(__file__).with_name("host_encryption_native.c")
    libcrypto = ctypes.util.find_library("crypto")
    if libcrypto is None:
        raise HostEncryptionError(
            "HiCache host encryption needs OpenSSL's libcrypto, which was not found."
        )
    digest = hashlib.sha256(source.read_bytes() + libcrypto.encode()).hexdigest()[:16]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "sglang"
    target = cache / f"hicache_host_encryption_{digest}.so"
    if target.exists():
        return target
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache, suffix=".so", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            ["cc", "-O2", "-shared", "-fPIC", "-o", str(tmp_path), str(source)]
            + [f"-l:{libcrypto}"],
            check=True,
            capture_output=True,
            text=True,
        )
        # Ranks build concurrently; whichever rename lands last wins, and every
        # result is identical.
        tmp_path.replace(target)
    except (OSError, subprocess.CalledProcessError) as e:
        raise HostEncryptionError(
            "HiCache host encryption could not build its native library: "
            f"{getattr(e, 'stderr', None) or e}"
        ) from e
    finally:
        tmp_path.unlink(missing_ok=True)
    return target


class _Pool:
    """Encryption state of one host pool: where its pages live and what each holds."""

    def __init__(self, pool_id: int, host_pool: Any):
        self.pool_id = pool_id
        self.name = type(host_pool).__name__
        self.page_size = host_pool.page_size
        self.buffers = self._describe(host_pool)
        self.page_bytes = int(self.buffers["size"].sum())
        self.units_per_page = -(-self.page_bytes // _UNIT_BYTES)
        if self.units_per_page >= 1 << 16:
            raise HostEncryptionError(f"{self.name} pages are too large to encrypt.")
        # Sized once for the whole pool: worker threads write tags while the
        # scheduler thread updates page state, so these never reallocate.
        self.num_pages = host_pool.size // self.page_size + 1
        self.state = np.zeros(self.num_pages, dtype=np.uint8)
        self.epoch = np.zeros(self.num_pages, dtype=np.uint32)
        self.generation = np.zeros(self.num_pages, dtype=np.uint64)
        self.readers = np.zeros(self.num_pages, dtype=np.int32)
        self.tags = np.zeros(
            self.num_pages * self.units_per_page * _TAG_BYTES, dtype=np.uint8
        )

    @staticmethod
    def _describe(host_pool: Any) -> np.ndarray:
        """Every host buffer as (base, bytes between pages, bytes per page).

        Asks the pool where three of its pages live and checks that page
        addresses are linear in the page number, which holds for the
        page-first layouts.
        """
        from sglang.srt.mem_cache.pool_host.base import uses_shared_host_layout

        if uses_shared_host_layout(host_pool) or not hasattr(
            host_pool, "get_page_buffer_meta"
        ):
            raise HostEncryptionError(
                "HiCache host encryption does not support the "
                f"{type(host_pool).__name__} host pool."
            )
        page_size = host_pool.page_size
        last = host_pool.size // page_size - 1

        def probe(page: int):
            indices = torch.arange(page * page_size, (page + 1) * page_size)
            ptrs, sizes = host_pool.get_page_buffer_meta(indices)
            return [int(p) for p in ptrs], [int(n) for n in sizes]

        (ptrs0, sizes0), (ptrs1, sizes1), (ptrs2, sizes2) = probe(0), probe(1), probe(last)
        buffers = np.zeros(len(ptrs0), dtype=_BUFFER)
        for i, (base, size) in enumerate(zip(ptrs0, sizes0)):
            stride = ptrs1[i] - base
            if (
                stride < size
                or sizes1[i] != size
                or sizes2[i] != size
                or ptrs2[i] != base + last * stride
            ):
                raise HostEncryptionError(
                    "HiCache host encryption needs page-linear host buffers; "
                    f"{type(host_pool).__name__} layout "
                    f"{getattr(host_pool, 'layout', None)!r} is not."
                )
            buffers[i] = (base, stride, size)
        return buffers

    def check(self, pages: np.ndarray) -> None:
        if len(pages) and (pages[0] < 0 or pages[-1] >= self.num_pages):
            raise HostEncryptionError(
                f"{self.name} transfer names a page outside the pool."
            )

    def items(self, pages: np.ndarray) -> np.ndarray:
        """One entry per unit of every page, with the page's current generation."""
        units = self.units_per_page
        items = np.zeros(len(pages) * units, dtype=_ITEM)
        items["page"] = np.repeat(pages, units)
        items["generation"] = np.repeat(self.generation[pages], units)
        items["unit"] = np.tile(np.arange(units, dtype=np.uint32), len(pages))
        return items


class _Ticket:
    """The pages one transfer submission touches, with the write epoch of each."""

    __slots__ = ("parts",)

    def __init__(self):
        self.parts: list[tuple[_Pool, np.ndarray, np.ndarray]] = []


class HostPoolEncryption:
    def __init__(self, num_threads: int):
        self._lib = ctypes.CDLL(str(_build_native_library()))
        vp, i64, u32, u64 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint32, ctypes.c_uint64
        common = [vp, u32, vp, ctypes.c_int, u64, u32, i64, vp, vp]
        self._lib.hc_seal.argtypes = common
        self._lib.hc_open.argtypes = common + [vp]
        self._lib.hc_seal.restype = self._lib.hc_open.restype = ctypes.c_int
        self._key = os.urandom(32)
        self._num_threads = max(1, num_threads)
        self._executor = ThreadPoolExecutor(
            self._num_threads, thread_name_prefix="hicache-host-crypto"
        )
        self._cv = threading.Condition()
        self._pools: dict[int, _Pool] = {}
        self._jobs: queue.Queue = queue.Queue()
        self._stats = collections.Counter()
        # Per load: (seconds waiting on busy pages, seconds decrypting, bytes
        # decrypted, bytes found already in plaintext, total seconds).
        self._recent_loads: collections.deque = collections.deque(maxlen=2048)
        # Per backup: (seconds waiting on busy pages, total seconds, bytes).
        self._recent_backups: collections.deque = collections.deque(maxlen=2048)
        self._last_report = time.monotonic()
        self._reported_bytes = 0
        self._last_submission = 0.0
        self._quiet_seconds = envs.SGLANG_HICACHE_HOST_ENCRYPTION_QUIET_MS.get() / 1e3
        self._max_delay_seconds = max(
            self._quiet_seconds,
            envs.SGLANG_HICACHE_HOST_ENCRYPTION_MAX_DELAY_MS.get() / 1e3,
        )
        threading.Thread(
            target=self._after_transfer_loop,
            name="hicache-host-crypto-seal",
            daemon=True,
        ).start()
        logger.info(
            "HiCache host pool encryption enabled: AES-256-GCM, %d threads, "
            "per-process key.",
            self._num_threads,
        )

    @classmethod
    def maybe_create(cls):
        if envs.SGLANG_TEST_HICACHE_HOST_ENCRYPTION_OBSERVE.get():
            return HostPoolObserver()
        if not envs.SGLANG_ENABLE_HICACHE_HOST_ENCRYPTION.get():
            return None
        return cls(envs.SGLANG_HICACHE_HOST_ENCRYPTION_THREADS.get())

    # ---- page resolution -------------------------------------------------

    def _pool(self, host_pool: Any) -> _Pool:
        pool = self._pools.get(id(host_pool))
        if pool is None:
            if len(self._pools) >= 256:
                raise HostEncryptionError("Too many HiCache host pools to encrypt.")
            pool = self._pools[id(host_pool)] = _Pool(len(self._pools), host_pool)
            logger.info(
                "HiCache host encryption covers %s: %d buffer(s), %.2f MB per page "
                "in %d unit(s).",
                pool.name,
                len(pool.buffers),
                pool.page_bytes / 1e6,
                pool.units_per_page,
            )
        return pool

    def _cpu_indices(self, host_indices: torch.Tensor) -> torch.Tensor:
        if not host_indices.is_cuda:
            return host_indices
        stashed = getattr(host_indices, CPU_INDICES_ATTR, None)
        if stashed is not None:
            return stashed
        self._stats["index_readbacks"] += 1
        return host_indices.cpu()

    def _resolve(self, transfers) -> list[tuple[_Pool, np.ndarray]]:
        """The distinct pages of each pool that the transfers touch."""
        by_pool: dict[int, tuple[_Pool, list[np.ndarray]]] = {}
        seen: set[tuple[int, int]] = set()
        for transfer in transfers:
            key = (id(transfer.host_pool), id(transfer.host_indices))
            if key in seen or transfer.host_indices.numel() == 0:
                continue
            seen.add(key)
            pool = self._pool(transfer.host_pool)
            page_size = pool.page_size
            # numpy from here on: these run on the scheduler thread for every
            # transfer, and small torch ops cost several times as much.
            indices = self._cpu_indices(transfer.host_indices).numpy()
            firsts = indices[::page_size]
            if (
                len(indices) % page_size != 0
                or (firsts % page_size != 0).any()
                or (indices[page_size - 1 :: page_size] - firsts != page_size - 1).any()
            ):
                raise HostEncryptionError(
                    "HiCache host encryption needs transfers of whole, aligned "
                    f"pages; got {len(indices)} indices for {pool.name} with "
                    f"page size {page_size}."
                )
            pages = (firsts // page_size).astype(np.int64, copy=False)
            by_pool.setdefault(pool.pool_id, (pool, []))[1].append(pages)
        resolved = []
        for pool, parts in by_pool.values():
            pages = parts[0] if len(parts) == 1 else np.concatenate(parts)
            # Transfers almost always name ascending pages; only sort and
            # deduplicate when they do not.
            if len(pages) > 1 and not (pages[1:] > pages[:-1]).all():
                pages = np.unique(pages)
            resolved.append((pool, pages))
        return resolved

    # ---- crypto ----------------------------------------------------------

    def _jobs_for(self, work: list[tuple[_Pool, np.ndarray]]) -> list:
        """Split (pool, items) work into per-thread jobs of similar size."""
        total = sum(len(items) * min(pool.page_bytes, _UNIT_BYTES) for pool, items in work)
        target = max(1, total // self._num_threads)
        jobs = []
        for pool, items in work:
            if len(items) == 0:
                continue
            nbytes = len(items) * min(pool.page_bytes, _UNIT_BYTES)
            parts = min(len(items), max(1, round(nbytes / target)))
            jobs.extend((pool, part) for part in np.array_split(items, parts))
        return jobs

    def _call(self, fn, pool: _Pool, items: np.ndarray, *extra) -> int:
        return fn(
            self._key,
            pool.pool_id,
            pool.buffers.ctypes.data,
            len(pool.buffers),
            _UNIT_BYTES,
            pool.units_per_page,
            len(items),
            items.ctypes.data,
            pool.tags.ctypes.data,
            *extra,
        )

    def _map(self, fn, jobs: list) -> list:
        if len(jobs) <= 1:
            return [fn(job) for job in jobs]
        return list(self._executor.map(fn, jobs))

    def _seal(self, claimed: list[tuple[_Pool, np.ndarray]]) -> None:
        """Encrypt pages already marked _BUSY, then publish them as _SEALED."""
        if not any(len(pages) for _, pages in claimed):
            return
        start = time.perf_counter()
        # The caller advanced each page's generation under the lock.
        work = [(pool, pool.items(pages)) for pool, pages in claimed]

        def seal(job):
            pool, items = job
            return self._call(self._lib.hc_seal, pool, np.ascontiguousarray(items))

        failed = [rc for rc in self._map(seal, self._jobs_for(work)) if rc]
        if failed:
            # The pages hold a mix of plaintext and ciphertext that nothing
            # can read back; leaving them _BUSY keeps every later load waiting.
            raise HostEncryptionError("AES-GCM encryption failed.")
        with self._cv:
            for pool, pages in claimed:
                pool.state[pages] = _SEALED
                self._stats["sealed_bytes"] += len(pages) * pool.page_bytes
            self._stats["seal_seconds"] += time.perf_counter() - start
            self._cv.notify_all()

    def _wait_not_busy(self, resolved) -> float:
        """Caller holds the lock. Returns the seconds spent waiting."""
        if not any((pool.state[pages] == _BUSY).any() for pool, pages in resolved):
            return 0.0
        start = time.perf_counter()
        while any((pool.state[pages] == _BUSY).any() for pool, pages in resolved):
            self._cv.wait()
        return time.perf_counter() - start

    # ---- device -> host --------------------------------------------------

    def begin_backup(self, transfers) -> _Ticket:
        """Call before submitting backup kernels: the pages become plaintext."""
        start = self._last_submission = time.perf_counter()
        ticket = _Ticket()
        resolved = self._resolve(transfers)
        with self._cv:
            for pool, pages in resolved:
                pool.check(pages)
            waited = self._wait_not_busy(resolved)
            nbytes = 0
            for pool, pages in resolved:
                pool.epoch[pages] += 1
                pool.state[pages] = _PENDING
                pool.readers[pages] = 0
                ticket.parts.append((pool, pages, pool.epoch[pages]))
                nbytes += len(pages) * pool.page_bytes
            self._recent_backups.append((waited, time.perf_counter() - start, nbytes))
        return ticket

    def finish_backup(self, ticket: _Ticket, finish_event: Any) -> None:
        self._jobs.put((self._claim_backup, ticket, finish_event, time.perf_counter()))

    def _claim_backup(self, ticket: _Ticket) -> list[tuple[_Pool, np.ndarray]]:
        """Caller holds the lock."""
        claimed = []
        for pool, pages, epochs in ticket.parts:
            # A newer backup, or a load that found the page still in
            # plaintext, owns it now and seals it when it is done.
            mine = pages[
                (pool.epoch[pages] == epochs) & (pool.state[pages] == _PENDING)
            ]
            pool.state[mine] = _BUSY
            pool.generation[mine] += 1
            claimed.append((pool, mine))
        return claimed

    # ---- host -> device --------------------------------------------------

    def begin_load(self, transfers) -> _Ticket:
        """Call before submitting load kernels: decrypts the pages in place."""
        start = self._last_submission = time.perf_counter()
        ticket = _Ticket()
        resolved = self._resolve(transfers)
        to_open = []
        plain_bytes = 0
        with self._cv:
            for pool, pages in resolved:
                pool.check(pages)
            waited = self._wait_not_busy(resolved)
            for pool, pages in resolved:
                sealed = pages[pool.state[pages] == _SEALED]
                pool.state[pages] = _OPEN
                pool.state[sealed] = _BUSY
                pool.readers[pages] += 1
                ticket.parts.append((pool, pages, pool.epoch[pages]))
                to_open.append((pool, sealed, pool.items(sealed)))
                plain_bytes += (len(pages) - len(sealed)) * pool.page_bytes
        open_start = time.perf_counter()
        opened_bytes = self._open(to_open)
        end = time.perf_counter()
        with self._cv:
            self._recent_loads.append(
                (waited, end - open_start, opened_bytes, plain_bytes, end - start)
            )
        return ticket

    def _open(self, claimed: list[tuple[_Pool, np.ndarray, np.ndarray]]) -> int:
        opened_bytes = sum(len(pages) * pool.page_bytes for pool, pages, _ in claimed)
        if opened_bytes == 0:
            return 0
        start = time.perf_counter()

        def open_(job):
            pool, items = job
            first_failed = ctypes.c_int64(-1)
            failures = self._call(
                self._lib.hc_open,
                pool,
                np.ascontiguousarray(items),
                ctypes.byref(first_failed),
            )
            return pool, failures, int(items["page"][first_failed.value]) if failures else -1

        results = self._map(open_, self._jobs_for([(p, items) for p, _, items in claimed]))
        with self._cv:
            for pool, pages, _ in claimed:
                pool.state[pages] = _OPEN
            self._stats["opened_bytes"] += opened_bytes
            self._stats["open_seconds"] += time.perf_counter() - start
            self._stats["open_calls"] += 1
            self._cv.notify_all()
        failed = [(pool, count, page) for pool, count, page in results if count]
        if failed:
            pool, _, page = failed[0]
            raise HostEncryptionError(
                "HiCache host pool authentication failed for "
                f"{sum(count for _, count, _ in failed)} unit(s), first in "
                f"{pool.name} page {page}; refusing to load them."
            )
        return opened_bytes

    def finish_load(self, ticket: _Ticket, finish_event: Any) -> None:
        self._jobs.put((self._claim_load, ticket, finish_event, time.perf_counter()))

    def _claim_load(self, ticket: _Ticket) -> list[tuple[_Pool, np.ndarray]]:
        """Caller holds the lock."""
        claimed = []
        for pool, pages, epochs in ticket.parts:
            current = pages[pool.epoch[pages] == epochs]
            pool.readers[current] -= 1
            idle = current[
                (pool.readers[current] <= 0) & (pool.state[current] == _OPEN)
            ]
            pool.state[idle] = _BUSY
            pool.generation[idle] += 1
            claimed.append((pool, idle))
        return claimed

    # ---- background ------------------------------------------------------

    def _after_transfer_loop(self) -> None:
        """Encrypt the pages of finished transfers.

        Waits for the scheduler thread to stop submitting transfers, so that
        encryption does not compete with a burst of them (an eviction is a
        dozen small backups followed by a load), then encrypts everything
        that finished in the meantime as one batch. A page still never stays
        in plaintext longer than the maximum delay after its transfer.
        """
        while True:
            try:
                batch = [self._jobs.get(timeout=5)]
            except queue.Empty:
                self._maybe_report()
                continue
            try:
                deadline = batch[0][3] + self._max_delay_seconds
                while True:
                    now = time.perf_counter()
                    quiet_at = self._last_submission + self._quiet_seconds
                    if now >= min(quiet_at, deadline):
                        break
                    time.sleep(min(quiet_at, deadline) - now)
                while True:
                    try:
                        batch.append(self._jobs.get_nowait())
                    except queue.Empty:
                        break
                # Events complete in submission order per stream, so waiting
                # in order costs no more than waiting for the last.
                for _, _, finish_event, _ in batch:
                    if finish_event is not None:
                        finish_event.synchronize()
                merged: dict[int, tuple[_Pool, list[np.ndarray]]] = {}
                with self._cv:
                    for claim, ticket, _, _ in batch:
                        for pool, pages in claim(ticket):
                            merged.setdefault(pool.pool_id, (pool, []))[1].append(pages)
                    self._stats["seal_batches"] += 1
                    self._stats["seal_jobs"] += len(batch)
                self._seal(
                    [(pool, np.concatenate(parts)) for pool, parts in merged.values()]
                )
                self._maybe_report()
            except Exception:
                logger.exception("HiCache host encryption worker failed")

    def stats(self) -> dict:
        with self._cv:
            stats = dict(self._stats)
            stats["recent_loads"] = list(self._recent_loads)
            stats["recent_backups"] = list(self._recent_backups)
            sealed = sum(int((p.state == _SEALED).sum()) for p in self._pools.values())
            plain = sum(
                int(((p.state != _SEALED) & (p.state != _PLAIN)).sum())
                for p in self._pools.values()
            )
        stats["pages_sealed"], stats["pages_plaintext"] = sealed, plain
        return stats

    def _maybe_report(self) -> None:
        now = time.monotonic()
        if now - self._last_report < 30:
            return
        self._last_report = now
        s = self.stats()
        moved = s.get("sealed_bytes", 0) + s.get("opened_bytes", 0)
        if moved == self._reported_bytes:
            return
        self._reported_bytes = moved

        def rate(nbytes, seconds):
            return nbytes / seconds / 1e9 if seconds else 0.0

        def pct(values, q):
            values = sorted(values)
            return values[min(len(values) - 1, int(q * len(values)))] if values else 0.0

        loads, backups = s["recent_loads"], s["recent_backups"]
        logger.info(
            "HiCache host encryption: sealed %.2f GB (%.1f GB/s), opened %.2f GB "
            "(%.1f GB/s) in %d loads, %d pages sealed, %d in plaintext, %d index "
            "readbacks, %d transfers encrypted in %d batches. Recent loads, ms: total p50 %.2f p95 %.2f max %.2f, "
            "decrypt p50 %.2f p95 %.2f, waiting on busy pages p95 %.2f max %.2f; "
            "MB decrypted p50 %.0f, already plaintext p50 %.0f. Recent backups, "
            "ms: prepare p50 %.3f p95 %.3f max %.3f, waiting on busy pages max %.3f.",
            s.get("sealed_bytes", 0) / 1e9,
            rate(s.get("sealed_bytes", 0), s.get("seal_seconds", 0)),
            s.get("opened_bytes", 0) / 1e9,
            rate(s.get("opened_bytes", 0), s.get("open_seconds", 0)),
            s.get("open_calls", 0),
            s["pages_sealed"],
            s["pages_plaintext"],
            s.get("index_readbacks", 0),
            s.get("seal_jobs", 0),
            s.get("seal_batches", 0),
            pct([x[4] for x in loads], 0.5) * 1e3,
            pct([x[4] for x in loads], 0.95) * 1e3,
            pct([x[4] for x in loads], 1.0) * 1e3,
            pct([x[1] for x in loads], 0.5) * 1e3,
            pct([x[1] for x in loads], 0.95) * 1e3,
            pct([x[0] for x in loads], 0.95) * 1e3,
            pct([x[0] for x in loads], 1.0) * 1e3,
            pct([x[2] for x in loads], 0.5) / 1e6,
            pct([x[3] for x in loads], 0.5) / 1e6,
            pct([x[1] for x in backups], 0.5) * 1e3,
            pct([x[1] for x in backups], 0.95) * 1e3,
            pct([x[1] for x in backups], 1.0) * 1e3,
            pct([x[0] for x in backups], 1.0) * 1e3,
        )


def describe_host_pool(host_pool: Any, indices: Optional[torch.Tensor] = None) -> str:
    """The host buffers of a pool and how it exposes a page, for bring-up."""
    lines = [
        f"{' > '.join(c.__name__ for c in type(host_pool).__mro__[:-1])}: "
        + ", ".join(
            f"{name}={getattr(host_pool, name, None)!r}"
            for name in ("layout", "page_size", "size", "page_num", "dtype")
        )
    ]
    buffers = []

    def visit(name, value):
        if isinstance(value, torch.Tensor):
            if value.device.type == "cpu" and value.numel() * value.element_size() >= 1 << 20:
                nbytes = value.numel() * value.element_size()
                buffers.append((value.data_ptr(), nbytes))
                lines.append(
                    f"  {name}: shape={tuple(value.shape)} dtype={value.dtype} "
                    f"bytes={nbytes} ptr={value.data_ptr():#x}"
                )
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                visit(f"{name}[{i}]", item)

    for name, value in vars(host_pool).items():
        visit(name, value)
    if indices is not None and indices.numel() >= host_pool.page_size:
        first = indices[: host_pool.page_size]
        try:
            page = host_pool.get_data_page(int(first[0]), flat=True)
            ptr, nbytes = page.data_ptr(), page.numel() * page.element_size()
            view = any(lo <= ptr < lo + size for lo, size in buffers)
            lines.append(
                f"  get_data_page(flat): shape={tuple(page.shape)} dtype={page.dtype} "
                f"bytes={nbytes} {'view' if view else 'copy'}"
            )
        except Exception as e:  # bring-up aid: report whatever the pool says
            lines.append(f"  get_data_page(flat) raised {type(e).__name__}: {e}")
        try:
            ptrs, sizes = host_pool.get_page_buffer_meta(first)
            lines.append(
                "  get_page_buffer_meta: "
                + ", ".join(f"{int(p):#x}+{int(n)}" for p, n in zip(ptrs, sizes))
            )
        except Exception as e:
            lines.append(f"  get_page_buffer_meta raised {type(e).__name__}: {e}")
    return "\n".join(lines)


class HostPoolObserver:
    """Logs what each transfer touches without encrypting anything."""

    def __init__(self):
        self._seen: dict[tuple[int, str], int] = {}
        logger.info("HiCache host pool observer enabled.")

    def _observe(self, direction: str, transfers) -> None:
        for transfer in transfers:
            host_pool, indices = transfer.host_pool, transfer.host_indices
            key = (id(host_pool), direction)
            count = self._seen.get(key, 0)
            if count >= 4:
                continue
            self._seen[key] = count + 1
            cpu = getattr(indices, CPU_INDICES_ATTR, None)
            stashed = cpu is not None
            cpu = indices.cpu() if cpu is None else cpu
            page_size = host_pool.page_size
            whole = cpu.numel() % page_size == 0 and cpu.numel() > 0
            aligned = contiguous = None
            if whole:
                pages = cpu.reshape(-1, page_size)
                aligned = bool((pages[:, 0] % page_size == 0).all())
                contiguous = bool(
                    (pages == pages[:, :1] + torch.arange(page_size)).all()
                )
            logger.info(
                "HiCache host observer %s #%d pool=%s indices: device=%s stashed=%s "
                "numel=%d min=%s max=%s whole_pages=%s aligned=%s contiguous=%s "
                "is_draft=%s layer_mapper=%s shared_layout=%s",
                direction,
                count,
                type(host_pool).__name__,
                indices.device,
                stashed,
                cpu.numel(),
                int(cpu.min()) if cpu.numel() else None,
                int(cpu.max()) if cpu.numel() else None,
                whole,
                aligned,
                contiguous,
                transfer.is_draft,
                transfer.layer_mapper is not None,
                getattr(host_pool, "shared_allocation_domain", None) is not None,
            )
            if count == 0:
                logger.info("HiCache host observer pool:\n%s", describe_host_pool(host_pool, cpu))

    def begin_backup(self, transfers):
        self._observe("backup", transfers)

    def begin_load(self, transfers):
        self._observe("load", transfers)

    def finish_backup(self, ticket, finish_event) -> None:
        pass

    def finish_load(self, ticket, finish_event) -> None:
        pass
