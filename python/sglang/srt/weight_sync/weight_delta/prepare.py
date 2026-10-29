"""Before the pause, while the engine serves: stage this GPU's part of one weight-delta version."""

from __future__ import annotations

import ctypes
import dataclasses
import logging
import mmap
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import zmq
import zstandard

from sglang.srt.weight_sync.weight_delta.manifest import (
    Tensors,
    read_manifest,
    read_tensors,
)
from sglang.srt.weight_sync.weight_delta.map import Ignored, MapTable

logger = logging.getLogger(__name__)

# Compressed bytes per Zstd call; the calls run on `threads` threads at once.
ZSTD_GROUP_NBYTES = 64 << 20
STAGING_GRANULE = 64 << 20
# Where each staged region starts: nvCOMP's frame inputs and every dtype's views need it.
STAGED_ALIGNMENT = 16


@dataclass
class Prepared:
    """This GPU's tensors of one version, raw ones first: their MapTable rows, and each one's raw
    value or inner bytes in the staging memory, in row order on aligned starts. Row i's frames are
    frames rows [frame_end[i - 1], frame_end[i]), in the tensor and its inner bytes."""

    base_version: int
    target_version: int
    codec: str
    table_rows: np.ndarray
    raw: np.ndarray
    nbytes: np.ndarray
    staged_offset: np.ndarray
    staged_nbytes: np.ndarray
    staging_capacity: int
    frame_end: np.ndarray
    frames: dict[str, np.ndarray]
    stats: dict = field(default_factory=dict)

    @property
    def staged_end(self) -> int:
        if not len(self.table_rows):
            return 0
        return int(self.staged_offset[-1] + self.staged_nbytes[-1])


class Staging:
    """A memfd holding a version's staged bytes, reused by every version; it only grows."""

    def __init__(self, fd: int) -> None:
        self.fd = fd
        self.capacity = os.fstat(fd).st_size
        self._view: mmap.mmap | None = None

    def fit(self, nbytes: int) -> np.ndarray:
        if nbytes > self.capacity:
            if self._view is not None:
                self._view.close()
                self._view = None
            self.capacity = -(-nbytes * 5 // 4 // STAGING_GRANULE) * STAGING_GRANULE
            os.ftruncate(self.fd, self.capacity)
            logger.info(f"Weight delta staging grew to {self.capacity / 2**30:.2f} GiB")
        if not nbytes:
            return np.zeros(0, dtype=np.uint8)
        if self._view is None:
            self._view = mmap.mmap(self.fd, self.capacity)
        return np.frombuffer(self._view, dtype=np.uint8, count=nbytes)


class PrepareProcess:
    """A process per GPU that prepares versions, so prepare's Python never shares the scheduler's
    GIL. Staging is a memfd both processes map; the scheduler pins its mapping whenever it grew."""

    def __init__(self, table: MapTable, threads: int) -> None:
        self._fd = os.memfd_create("weight-delta-staging")
        address = f"ipc:///tmp/weight-delta-prepare-{os.getpid()}"
        self._socket = zmq.Context.instance().socket(zmq.PAIR)
        self._socket.bind(address)
        # The places hold this process's serving tensors; the other process needs only the reasons.
        reasons = [
            entry if isinstance(entry, Ignored) else None for entry in table.entries
        ]
        self._process = mp.get_context("spawn").Process(
            target=_serve,
            args=(
                os.getpid(),
                self._fd,
                address,
                dataclasses.replace(table, entries=reasons),
                threads,
            ),
            name="weight-delta-prepare",
            daemon=True,
        )
        self._process.start()
        self._view: mmap.mmap | None = None
        self._pointer = 0
        self._staged = torch.empty(0, dtype=torch.uint8)

    def prepare(
        self, manifest_path: str | Path, engine_version: str
    ) -> tuple[Prepared, torch.Tensor]:
        """The version's preparation and its staged bytes, pinned, valid until the next prepare."""
        self.start(manifest_path, engine_version)
        while (result := self.poll()) is None:
            self._socket.poll(1000)
        return result

    def start(self, manifest_path: str | Path, engine_version: str) -> None:
        self._socket.send_pyobj((str(manifest_path), engine_version))

    def poll(self) -> tuple[Prepared, torch.Tensor] | None:
        """What prepare returns once the started preparation is done, else None."""
        if not self._socket.poll(0):
            if not self._process.is_alive():
                raise RuntimeError(
                    f"the weight-delta prepare process exited with {self._process.exitcode}"
                )
            return None
        prepared = self._socket.recv_pyobj()
        if isinstance(prepared, BaseException):
            raise prepared
        if prepared.staging_capacity != len(self._staged):
            self._pin(prepared.staging_capacity)
        return prepared, self._staged[: prepared.staged_end]

    def close(self) -> None:
        self._process.terminate()
        self._unpin()
        os.close(self._fd)

    def _pin(self, capacity: int) -> None:
        from cuda.bindings import runtime

        self._unpin()
        self._view = mmap.mmap(self._fd, capacity)
        self._pointer = ctypes.addressof(ctypes.c_ubyte.from_buffer(self._view))
        _check(
            runtime.cudaHostRegister(
                self._pointer, capacity, runtime.cudaHostRegisterPortable
            ),
            "cudaHostRegister",
        )
        memory = (ctypes.c_ubyte * capacity).from_address(self._pointer)
        self._staged = torch.frombuffer(memory, dtype=torch.uint8)

    def _unpin(self) -> None:
        from cuda.bindings import runtime

        if self._view is None:
            return
        self._staged = torch.empty(0, dtype=torch.uint8)
        _check(runtime.cudaHostUnregister(self._pointer), "cudaHostUnregister")
        self._pointer = 0
        self._view.close()
        self._view = None


def _serve(
    scheduler_pid: int, fd: int, address: str, table: MapTable, threads: int
) -> None:
    from sglang.srt.utils.common import kill_itself_when_parent_died

    kill_itself_when_parent_died()
    socket = zmq.Context.instance().socket(zmq.PAIR)
    socket.connect(address)
    staging = Staging(os.open(f"/proc/{scheduler_pid}/fd/{fd}", os.O_RDWR))
    while True:
        manifest_path, engine_version = socket.recv_pyobj()
        try:
            reply = prepare(table, manifest_path, engine_version, staging, threads)
        except Exception as error:
            reply = error
        socket.send_pyobj(reply)


def prepare(
    table: MapTable,
    manifest_path: str | Path,
    engine_version: str,
    staging: Staging,
    threads: int,
) -> Prepared:
    """Any failure here leaves the engine as it was."""
    started = time.perf_counter()
    manifest = read_manifest(manifest_path)
    if str(manifest.base_version) != engine_version:
        raise ValueError(
            f"the delta starts at version {manifest.base_version}, "
            f"the engine is at {engine_version}"
        )
    paths = [manifest.directory / name for name in manifest.files]
    _read_into_page_cache(paths, threads)
    read = time.perf_counter()

    tensors = read_tensors(manifest)
    rows = _join(table, tensors)
    held = np.flatnonzero(table.held[rows])
    held = held[np.argsort(~tensors.raw[held], kind="stable")]
    raw = tensors.raw[held]
    staged_nbytes = np.where(raw, tensors.nbytes[held], tensors.inner_nbytes[held])
    spans = -(-staged_nbytes // STAGED_ALIGNMENT) * STAGED_ALIGNMENT
    staged_offset = np.cumsum(spans) - spans
    frame_rows, frame_end = ragged_rows(tensors.frame_end, held)
    joined = time.perf_counter()

    staged = staging.fit(int(spans.sum()))
    mapped = [_mmap(path) for path in paths]
    try:
        _stage_raw(tensors, held[raw], staged_offset[raw], mapped, staged)
        _stage_inner(tensors, held[~raw], staged_offset[~raw], mapped, staged, threads)
    finally:
        del mapped, staged
    return Prepared(
        base_version=manifest.base_version,
        target_version=manifest.target_version,
        codec=manifest.codec,
        table_rows=rows[held],
        raw=raw,
        nbytes=table.nbytes[rows[held]],
        staged_offset=staged_offset,
        staged_nbytes=staged_nbytes,
        staging_capacity=staging.capacity,
        frame_end=frame_end,
        frames={key: column[frame_rows] for key, column in tensors.frames.items()},
        stats={
            "files": len(paths),
            "tensors": len(rows),
            "held": len(held),
            "raw": int(raw.sum()),
            "staged_nbytes": int(staged_nbytes.sum()),
            "read_s": read - started,
            "join_s": joined - read,
            "stage_s": time.perf_counter() - joined,
        },
    )


def _read_into_page_cache(paths: list[Path], threads: int) -> None:
    """Large sequential reads: what a volume serves fast. Later reads of ranges then hit memory."""
    buffers = threading.local()

    def read(path: Path) -> None:
        if not hasattr(buffers, "data"):
            buffers.data = bytearray(64 << 20)
        with path.open("rb", buffering=0) as source:
            while source.readinto(buffers.data):
                pass

    with ThreadPoolExecutor(threads, thread_name_prefix="weight-delta-read") as pool:
        list(pool.map(read, paths))


def _join(table: MapTable, tensors: Tensors) -> np.ndarray:
    """Each version row's table row, once every row is one this GPU can take."""
    rows = np.searchsorted(table.names, tensors.names)
    known = table.names[np.minimum(rows, len(table.names) - 1)] == tensors.names
    _refuse(~known, tensors, "are not in the loaded checkpoint")
    unique, counts = np.unique(rows, return_counts=True)
    _refuse(np.isin(rows, unique[counts > 1]), tensors, "appear twice")
    _refuse(table.dtypes[rows] != tensors.dtypes, tensors, "have another dtype")
    ndim = np.diff(tensors.shape_end, prepend=0)
    loaded_ndim = np.diff(table.shape_end, prepend=0)[rows]
    _refuse(ndim != loaded_ndim, tensors, "have another shape")
    dims, _ = ragged_rows(table.shape_end, rows)
    other_dim = np.bincount(
        np.repeat(np.arange(len(rows)), ndim),
        weights=table.shape[dims] != tensors.shape,
        minlength=len(rows),
    )
    _refuse(other_dim > 0, tensors, "have another shape")
    ignored = table.ignored[rows]
    if ignored.any():
        first = int(np.flatnonzero(ignored)[0])
        _refuse(
            ignored, tensors, f"are not applied ({table.entries[rows[first]].reason})"
        )
    _refuse(
        table.raw_only[rows] & ~tensors.raw,
        tensors,
        "are stored wider, so must arrive raw",
    )
    _refuse(
        tensors.raw & (tensors.nbytes != table.nbytes[rows]),
        tensors,
        "arrive raw with another size",
    )
    return rows


def _refuse(mask: np.ndarray, tensors: Tensors, problem: str) -> None:
    if mask.any():
        names = [name.decode() for name in tensors.names[mask][:5]]
        raise ValueError(f"{int(mask.sum())} tensors {problem}, e.g. {names}")


def ragged_rows(ends: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rows of a ragged column that `rows` own, in order, and their ends."""
    starts = (ends - np.diff(ends, prepend=0))[rows]
    counts = ends[rows] - starts
    return ranges(starts, counts), np.cumsum(counts)


def ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """starts[i], starts[i] + 1, ..., starts[i] + counts[i] - 1, for every i, back to back."""
    return np.repeat(starts - np.cumsum(counts) + counts, counts) + np.arange(
        counts.sum()
    )


def _mmap(path: Path) -> mmap.mmap:
    with path.open("rb") as source:
        return mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)


def _stage_raw(
    tensors: Tensors,
    rows: np.ndarray,
    staged_offset: np.ndarray,
    mapped: list[mmap.mmap],
    staged: np.ndarray,
) -> None:
    files, nbytes = tensors.file[rows], tensors.nbytes[rows]
    for file in np.unique(files):
        in_file = files == file
        content = np.frombuffer(mapped[file], dtype=np.uint8)
        staged[ranges(staged_offset[in_file], nbytes[in_file])] = content[
            ranges(tensors.offset[rows][in_file], nbytes[in_file])
        ]


def _stage_inner(
    tensors: Tensors,
    rows: np.ndarray,
    staged_offset: np.ndarray,
    mapped: list[mmap.mmap],
    staged: np.ndarray,
    threads: int,
) -> None:
    """Undo the outer Zstd of every chunk of `rows` into its place in `staged`."""
    chunks, chunk_end = ragged_rows(tensors.chunk_end, rows)
    if not len(chunks):
        return
    row = np.repeat(np.arange(len(rows)), np.diff(chunk_end, prepend=0))
    files = tensors.file[rows][row]
    offset = tensors.chunks["offset"][chunks]
    nbytes = tensors.chunks["nbytes"][chunks]
    inner_nbytes = tensors.chunks["inner_nbytes"][chunks]
    staged_at = staged_offset[row] + tensors.chunks["inner_offset"][chunks]
    bucket = np.cumsum(nbytes) // ZSTD_GROUP_NBYTES
    boundaries = np.flatnonzero(
        np.diff(files * (bucket.max(initial=0) + 1) + bucket, prepend=-1)
    )
    groups = np.split(np.arange(len(chunks)), boundaries[1:])
    output = memoryview(staged)

    def decode_group(group: np.ndarray) -> None:
        segments = np.stack([offset[group], nbytes[group]], axis=1).astype("<u8")
        compressed = zstandard.BufferWithSegments(
            mapped[files[group[0]]], segments.tobytes()
        )
        decoded = zstandard.ZstdDecompressor().multi_decompress_to_buffer(
            compressed,
            decompressed_sizes=inner_nbytes[group].astype("<u8").tobytes(),
            threads=1,
        )
        destinations = zip(staged_at[group].tolist(), inner_nbytes[group].tolist())
        for index, (start, size) in enumerate(destinations):
            segment = decoded[index]
            if len(segment) != size:
                raise ValueError("an outer Zstd chunk decoded to the wrong size")
            output[start : start + size] = segment

    with ThreadPoolExecutor(threads, thread_name_prefix="weight-delta-zstd") as pool:
        list(pool.map(decode_group, groups))


def _check(result, operation: str):
    if int(result[0]):
        raise RuntimeError(f"Weight delta {operation} failed: CUDA {result[0]}")
    return result[1] if len(result) > 1 else None
