"""One process per node that reads LoRA adapters into slot order while the schedulers
serve.

Reading and placing an adapter inside a scheduler would share its GIL with decoding.
The process is a clean interpreter running this file, so this module imports no SGLang
code: only the standard library and torch, for the install's arithmetic. It sees no GPU.

Each GPU's scheduler connects over a Unix socket in a directory only this user can use,
and sends JSON requests; bulk data is shared memory, passed as file descriptors:
- per GPU, staging, which its scheduler pins once and copies into the slot from, reused
  by every update and only grown, because pinning host memory stalls the GPU's work;
- per adapter version, one shared copy holding each distinct buffer region once. A
  region several GPUs hold alike (a replicated module) is built by the first request
  that needs it, and every GPU maps the copy for the regions it uses as its
  LoRASlotWeights.

The scheduler that binds the node's socket first starts the process and hands it the
listening socket, so the process's exit resets every connection; that scheduler starts
a new process at its next request.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import ctypes
import errno
import hashlib
import json
import logging
import math
import mmap
import os
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from multiprocessing.connection import Connection
from typing import Dict, List, Optional, Sequence, Tuple

# The process runs this file in a clean interpreter: nothing from SGLang is imported.
import torch

logger = logging.getLogger(__name__)

# Staging grows in whole steps, with a quarter of headroom, so it is re-pinned rarely.
_STAGING_STEP = 64 << 20
_PREPARE_TIMEOUT_S = 600
# One call maps every page: the threads of one process would take each fault in turn.
_POPULATE = mmap.MAP_POPULATE
# Read as stored: like the install, scaled in the file's dtype and cast at the copy.
_FLOAT_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

# (name, file, shape), sorted by name.
FileTensor = Tuple[str, str, Tuple[int, ...]]
# lora_slot_weights._SlotCopy as sent: (tensor_index, src_offset, src_strides,
# dst_offset, dst_strides, shape, scaled).
SlotCopy = Tuple[int, int, Tuple[int, ...], int, Tuple[int, ...], Tuple[int, ...], bool]
# A GPU's slot: its buffer regions (offset, numel) in offset order, and the copies.
SlotLayout = Tuple[List[Tuple[int, int]], List[SlotCopy]]


def node_address(instance_id: str) -> str:
    digest = hashlib.blake2s(instance_id.encode(), digest_size=4).hexdigest()
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in instance_id)
    directory = os.path.join(
        tempfile.gettempdir(), f"sglang-lora-prepare-{os.getuid()}"
    )
    return os.path.join(directory, f"{safe}_{digest}.sock")


class LoRAPrepareClient:
    """A scheduler's side: connects to the node's process, owns this GPU's staging."""

    def __init__(self, address: str):
        self._address = address
        self._staging_fd = os.memfd_create("sglang-lora-staging")
        self._staging: Optional[torch.Tensor] = None
        # Recorded after the GPU's reads of staging are queued; a read waits for it.
        self._staging_read: Optional[torch.cuda.Event] = None
        self._process: Optional[subprocess.Popen] = None
        self._listener: Optional[socket.socket] = None
        self._connection: Optional[Connection] = None
        self._socket: Optional[socket.socket] = None
        self._layouts_sent: set = set()
        self._started: Optional[float] = None
        self._owner = False
        self._unavailable: Optional[str] = None
        try:
            _make_private_directory(os.path.dirname(address))
            self._owner = self._bind()
        except OSError as error:
            self._unavailable = f"no LoRA prepare socket at {address}: {error}"

    def list_tensors(self, prepare_id: str, files: Sequence[str]) -> None:
        self._send({"op": "list", "prepare_id": prepare_id, "files": list(files)})

    def request_tensors(self, prepare_id: str) -> None:
        self._send({"op": "tensors", "prepare_id": prepare_id})

    def read(
        self,
        prepare_id: str,
        layout_id: str,
        layout: SlotLayout,
        numel: int,
        dtype: torch.dtype,
        scaling: float,
    ) -> None:
        """Ask to read the listed files into this GPU's slot regions, in staging and in
        the version's shared copy, which poll replies with."""
        if self._staging_read is not None:
            self._staging_read.synchronize()
            self._staging_read = None
        self._fit_staging(numel * dtype.itemsize)
        # A new connection holds no layouts: decide what to send once connected.
        self._connect()
        self._send(
            {
                "op": "read",
                "prepare_id": prepare_id,
                "layout_id": layout_id,
                "layout": None if layout_id in self._layouts_sent else layout,
                "dtype": str(dtype).removeprefix("torch."),
                "scaling": scaling,
            }
        )
        self._layouts_sent.add(layout_id)

    def poll(self) -> Optional[Tuple[dict, Optional[torch.Tensor]]]:
        """The reply to the request in flight once it is done, else None. A read's
        reply comes with the version's shared copy, mapped as bytes."""
        if self._started is None:
            return None
        shared = None
        try:
            if time.monotonic() - self._started > _PREPARE_TIMEOUT_S:
                raise TimeoutError(f"no reply in {_PREPARE_TIMEOUT_S} s")
            if not self._connection.poll(0):
                return None
            reply = json.loads(self._connection.recv_bytes())
            if "region_offsets" in reply:
                _, fds, _, _ = socket.recv_fds(self._socket, 1, 1)
                if not fds:
                    raise RuntimeError("the shared copy did not arrive")
                try:
                    shared = _map(fds[0])
                finally:
                    for fd in fds:
                        os.close(fd)
        except Exception as error:
            self._disconnect()
            reply, shared = {"error": f"LoRA prepare process: {error!r}"}, None
        self._started = None
        return reply, shared

    def staged(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        """Pinned view of what the last read staged. Call record_staging_reads after
        queuing the copies from it: the next read waits for them before writing."""
        return self._staging[: numel * dtype.itemsize].view(dtype)

    def record_staging_reads(self) -> None:
        event = torch.cuda.Event()
        event.record()
        self._staging_read = event

    def _bind(self) -> bool:
        """Whether this scheduler owns the node's process: the first to bind does."""
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(self._address)
            listener.listen(64)
        except OSError as error:
            listener.close()
            if error.errno == errno.EADDRINUSE:
                return False
            raise
        self._listener = listener
        atexit.register(_unlink_quietly, self._address)
        return True

    def _send(self, request: dict) -> None:
        assert self._started is None, "a LoRA prepare request is still running"
        if self._unavailable is not None:
            raise RuntimeError(self._unavailable)
        self._connect()
        self._connection.send_bytes(json.dumps(request).encode())
        self._started = time.monotonic()

    def _connect(self) -> None:
        if self._owner and (self._process is None or self._process.poll() is not None):
            self._start_process()
        # Idle, the connection has nothing to read unless the process went away.
        if self._connection is not None and self._connection.poll(0):
            self._disconnect()
        if self._connection is not None:
            return
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # The owner listens a moment after it binds, and binds again after its
        # process exits.
        deadline = time.monotonic() + 0.1
        while True:
            try:
                sock.connect(self._address)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > deadline:
                    sock.close()
                    raise
                time.sleep(0.001)
        self._socket = sock
        self._connection = Connection(os.dup(sock.fileno()))
        # Staging goes once per connection; the process maps it for every read.
        self._connection.send_bytes(json.dumps({"op": "staging"}).encode())
        socket.send_fds(sock, [b"\0"], [self._staging_fd])

    def _start_process(self) -> None:
        if self._process is not None:
            logger.warning(
                f"The LoRA prepare process exited with {self._process.returncode}; "
                "starting a new one."
            )
            self._disconnect()
        if self._listener is None:
            # The last process took the listening socket with it.
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self._address)
            if not self._bind():
                raise RuntimeError(f"{self._address} was taken by another process")
        try:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    __file__,
                    "--listen-fd",
                    str(self._listener.fileno()),
                    "--parent-pid",
                    str(os.getpid()),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                pass_fds=(self._listener.fileno(),),
                start_new_session=True,
                env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
            )
        finally:
            # Only the process holds it now: when the process exits, connections
            # waiting to be accepted are reset instead of waiting forever.
            self._listener.close()
            self._listener = None

    def _disconnect(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._socket.close()
        self._connection = self._socket = None
        self._layouts_sent.clear()
        if self._started is not None:
            # The process may still be writing the old staging: take a new one.
            self._started = None
            self._drop_staging()
            os.close(self._staging_fd)
            self._staging_fd = os.memfd_create("sglang-lora-staging")

    def _drop_staging(self) -> None:
        if self._staging_read is not None:
            self._staging_read.synchronize()
            self._staging_read = None
        if self._staging is not None:
            err = torch.cuda.cudart().cudaHostUnregister(self._staging.data_ptr())
            if int(err) != 0:
                logger.warning(f"cudaHostUnregister of LoRA staging failed: {err}")
            # The old mapping closes once no view of it is left.
            self._staging = None

    def _fit_staging(self, nbytes: int) -> None:
        capacity = 0 if self._staging is None else self._staging.numel()
        if nbytes <= capacity:
            return
        capacity = math.ceil(nbytes * 1.25 / _STAGING_STEP) * _STAGING_STEP
        self._drop_staging()
        os.ftruncate(self._staging_fd, capacity)
        staging = torch.frombuffer(
            mmap.mmap(self._staging_fd, capacity), dtype=torch.uint8
        )
        err = torch.cuda.cudart().cudaHostRegister(staging.data_ptr(), capacity, 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister({capacity} bytes) failed: {err}")
        self._staging = staging
        logger.info(f"LoRA staging grew to {capacity / 2**30:.2f} GiB")


def _unlink_quietly(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


def _make_private_directory(directory: str) -> None:
    os.makedirs(directory, mode=0o700, exist_ok=True)
    info = os.stat(directory)
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError(f"{directory} is not private to this user")


def list_file_tensors(
    files: Sequence[str],
) -> Optional[Tuple[List[FileTensor], List[Tuple[torch.dtype, int]]]]:
    """Tensors and their (dtype, offset) by name; None unless uniquely named floats."""
    entries = []
    for file in files:
        with open(file, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            (header_len,) = struct.unpack("<Q", f.read(8)) if size >= 8 else (size,)
            if size < 8 + header_len:
                raise ValueError(
                    f"{file} is {size:,} bytes, shorter than its header: an "
                    "incomplete file?"
                )
            header = json.loads(f.read(header_len))
        end = 8 + header_len
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = _FLOAT_DTYPES.get(info["dtype"])
            if dtype is None:
                return None
            entry = (name, file, tuple(info["shape"]))
            entries.append((entry, (dtype, 8 + header_len + info["data_offsets"][0])))
            end = max(end, 8 + header_len + info["data_offsets"][1])
        if size < end:
            raise ValueError(
                f"{file} is {size:,} bytes but its header needs {end:,}: an "
                "incomplete file?"
            )
    if len({entry[0] for entry, _ in entries}) != len(entries):
        return None
    entries.sort(key=lambda e: e[0])
    return [entry for entry, _ in entries], [location for _, location in entries]


def inventory_digest(tensors: Sequence[FileTensor]) -> str:
    """Names and shapes: what a placement depends on in the files."""
    inventory = [(name, shape) for name, _, shape in tensors]
    return hashlib.sha256(json.dumps(inventory).encode()).hexdigest()


def read_slot_weights(
    data: torch.Tensor,
    copies: Sequence[SlotCopy],
    sources: Sequence[torch.Tensor],
    scaling: float,
) -> None:
    """Fill `data` as the copies place `sources`, cast as the install casts."""
    data.zero_()
    groups: Dict[Tuple[torch.dtype, bool], Tuple[list, list]] = {}
    for (
        tensor_index,
        src_offset,
        src_strides,
        dst_offset,
        dst_strides,
        shape,
        scaled,
    ) in copies:
        source = sources[tensor_index]
        dsts, srcs = groups.setdefault((source.dtype, scaled), ([], []))
        dsts.append(strided(data, dst_offset, dst_strides, shape))
        srcs.append(strided(source, src_offset, src_strides, shape))
    for (_, scaled), (dsts, srcs) in groups.items():
        if scaled:
            # The install's op: weight * scaling on CPU, in the file's dtype.
            srcs = torch._foreach_mul(srcs, scaling)
        torch._foreach_copy_(dsts, srcs)


def strided(
    flat: torch.Tensor, offset: int, strides: Sequence[int], shape: Sequence[int]
) -> torch.Tensor:
    return flat.as_strided(tuple(shape), tuple(strides), flat.storage_offset() + offset)


def _map(
    fd: int, length: Optional[int] = None, offset: int = 0, populate: bool = False
) -> torch.Tensor:
    if length is None:
        length = os.fstat(fd).st_size - offset
    if length == 0:
        return torch.empty(0, dtype=torch.uint8)
    flags = mmap.MAP_SHARED | (_POPULATE if populate else 0)
    return torch.frombuffer(
        mmap.mmap(fd, length, flags=flags, offset=offset), dtype=torch.uint8
    )


class _Region:
    """A buffer region of one version in the shared copy, built by its first reader."""

    def __init__(self, offset: int):
        self.offset = offset
        self.built = threading.Event()
        self.error: Optional[str] = None


class _Version:
    """One update's listing and shared copy; its regions are found by digest."""

    def __init__(self, prepare_id: str):
        self.prepare_id = prepare_id
        self.listed = threading.Event()
        self.tensors: Optional[List[FileTensor]] = None
        self.locations: List[Tuple[torch.dtype, int]] = []
        # Mapped only while a request uses it: no file stays open between updates.
        self.sources: Optional[List[torch.Tensor]] = None
        self.digest: Optional[str] = None
        self.error: Optional[str] = None
        self.shared_fd = os.memfd_create("sglang-lora-weights")
        self.size = 0
        self.regions: Dict[str, _Region] = {}
        self.lock = threading.Lock()
        # Requests using it; a retired version closes when the last one ends.
        self.users = 0
        self.retired = False

    def claim(self, digest: str, nbytes: int) -> Tuple[_Region, bool]:
        """The region of `digest`, and whether the caller builds it."""
        with self.lock:
            region = self.regions.get(digest)
            if region is not None:
                return region, False
            # Page-aligned, so a region maps on its own.
            region = _Region(self.size)
            self.size += -(-nbytes // mmap.PAGESIZE) * mmap.PAGESIZE
            os.ftruncate(self.shared_fd, self.size)
            self.regions[digest] = region
            return region, True


class _Node:
    def __init__(self):
        self.lock = threading.Lock()
        self.version: Optional[_Version] = None

    def use(self, prepare_id: str, create: bool) -> Tuple[Optional[_Version], bool]:
        """The version of `prepare_id`, held for the caller until release, and whether
        the caller lists it (only with `create`). A newer update retires the older
        one; the GPUs' maps of its shared copy keep that memory until they drop it."""
        with self.lock:
            version = self.version
            if version is not None and version.prepare_id == prepare_id:
                version.users += 1
                return version, False
            if not create:
                return None, False
            if version is not None:
                version.retired = True
                if version.users == 0:
                    os.close(version.shared_fd)
            self.version = _Version(prepare_id)
            self.version.users = 1
            return self.version, True

    def release(self, version: _Version) -> None:
        with self.lock:
            version.users -= 1
            if version.users == 0:
                version.sources = None
            if version.retired and version.users == 0:
                os.close(version.shared_fd)


class _Layout:
    """A GPU's regions, each with its copies rebased to the region and a digest."""

    def __init__(self, layout: SlotLayout):
        regions = [tuple(region) for region in layout[0]]
        copies = [
            (c[0], c[1], tuple(c[2]), c[3], tuple(c[4]), tuple(c[5]), c[6])
            for c in layout[1]
        ]
        by_region: List[List[SlotCopy]] = [[] for _ in regions]
        index = 0
        for copy in sorted(copies, key=lambda c: c[3]):
            while copy[3] >= sum(regions[index]):
                index += 1
            start = regions[index][0]
            by_region[index].append(copy[:3] + (copy[3] - start,) + copy[4:])
        self.regions = regions
        self.copies = by_region
        self.digests = [
            hashlib.sha256(json.dumps([numel, sorted(c)]).encode()).hexdigest()
            for (_, numel), c in zip(regions, by_region)
        ]


def _serve(node: _Node, sock: socket.socket) -> None:
    """One GPU's requests, in order."""
    connection = Connection(os.dup(sock.fileno()))
    staging_fd: Optional[int] = None
    staging = torch.empty(0, dtype=torch.uint8)
    layouts: Dict[str, _Layout] = {}
    while True:
        try:
            request = json.loads(connection.recv_bytes())
        except (EOFError, OSError):
            break
        if request["op"] == "staging":
            _, fds, _, _ = socket.recv_fds(sock, 1, 1)
            if staging_fd is not None:
                os.close(staging_fd)
            staging_fd, staging = fds[0], torch.empty(0, dtype=torch.uint8)
            continue
        version, lists = node.use(request["prepare_id"], create=request["op"] == "list")
        try:
            if version is None:
                raise RuntimeError("the LoRA adapter was not listed for this request")
            if lists:
                _list(version, request["files"])
            version.listed.wait()
            if version.error is not None:
                raise RuntimeError(version.error)
            if request["op"] == "list":
                reply: dict = {"digest": version.digest}
            elif request["op"] == "tensors":
                reply = {"tensors": version.tensors}
            else:
                if request["layout"] is not None:
                    layouts[request["layout_id"]] = _Layout(request["layout"])
                layout = layouts[request["layout_id"]]
                dtype = getattr(torch, request["dtype"])
                nbytes = sum(numel for _, numel in layout.regions) * dtype.itemsize
                if staging.numel() < nbytes:
                    staging = _map(staging_fd, populate=True)
                reply = _read(version, layout, dtype, request["scaling"], staging)
        except Exception as error:
            reply = {"error": f"{type(error).__name__}: {error}"}
        try:
            connection.send_bytes(json.dumps(reply).encode())
            if "region_offsets" in reply:
                socket.send_fds(sock, [b"\0"], [version.shared_fd])
        except OSError:
            break
        finally:
            if version is not None:
                node.release(version)
    connection.close()
    sock.close()
    if staging_fd is not None:
        os.close(staging_fd)


def _list(version: _Version, files: Sequence[str]) -> None:
    try:
        listed = list_file_tensors(files)
        if listed is not None:
            version.tensors, version.locations = listed
            version.digest = inventory_digest(version.tensors)
            # Mapping populates: the files' pages are read while the GPUs trace.
            version.map_sources()
    except Exception as error:
        version.error = f"{type(error).__name__}: {error}"
    version.listed.set()


    def map_sources(self) -> List[torch.Tensor]:
        with self.lock:
            if self.sources is None:
                self.sources = _map_file_tensors(self.tensors, self.locations)
            return self.sources

def _read(
    version: _Version,
    layout: _Layout,
    dtype: torch.dtype,
    scaling: float,
    staging: torch.Tensor,
) -> dict:
    """Build this GPU's regions that no request has built yet into the version's shared
    copy, wait for the ones another request builds, copy all of them into this GPU's
    staging, and reply where they sit in the copy. A request builds each region it
    claims at once, so it never waits while holding an unbuilt one."""
    offsets = []
    for (offset, numel), copies, digest in zip(
        layout.regions, layout.copies, layout.digests
    ):
        nbytes = numel * dtype.itemsize
        region, builds = version.claim(digest, nbytes)
        if builds:
            try:
                shared = _map(version.shared_fd, nbytes, region.offset, populate=True)
                read_slot_weights(shared.view(dtype), copies, version.sources, scaling)
            except Exception as error:
                region.error = f"{type(error).__name__}: {error}"
            finally:
                region.built.set()
        else:
            region.built.wait()
        if region.error is not None:
            raise RuntimeError(region.error)
        if not builds:
            shared = _map(version.shared_fd, nbytes, region.offset, populate=True)
        start = offset * dtype.itemsize
        staging[start : start + nbytes].copy_(shared)
        offsets.append(region.offset)
    return {"region_offsets": offsets}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-fd", type=int, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    # PR_SET_PDEATHSIG: exit with the scheduler that started this process.
    ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)
    if os.getppid() != args.parent_pid:
        return
    # The GPUs' requests run in parallel threads; op threads would take cores.
    torch.set_num_threads(1)
    listener = socket.socket(fileno=args.listen_fd)
    node = _Node()
    while True:
        sock, _ = listener.accept()
        threading.Thread(target=_serve, args=(node, sock), daemon=True).start()


if __name__ == "__main__":
    main()

def _map_file_tensors(
    tensors: Sequence[FileTensor], locations: Sequence[Tuple[torch.dtype, int]]
) -> List[torch.Tensor]:
    mapped = {}
    sources = []
    for (_, file, shape), (dtype, offset) in zip(tensors, locations):
        if file not in mapped:
            with open(file, "rb") as f:
                mapped[file] = mmap.mmap(
                    f.fileno(),
                    0,
                    flags=mmap.MAP_SHARED | _POPULATE,
                    prot=mmap.PROT_READ,
                )
        count = math.prod(shape)
        if count == 0:
            sources.append(torch.empty(0, dtype=dtype))
            continue
        with warnings.catch_warnings():
            # Read only, as the reader uses it.
            warnings.filterwarnings("ignore", "The given buffer is not writable")
            sources.append(
                torch.frombuffer(mapped[file], dtype=dtype, count=count, offset=offset)
            )
    return sources

