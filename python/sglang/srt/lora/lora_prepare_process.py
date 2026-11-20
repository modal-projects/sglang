"""A process per node that reads LoRA adapters into slot order while serving."""

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
import uuid
import warnings
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# The process runs this file in a clean interpreter: nothing from SGLang is imported.
import torch

logger = logging.getLogger(__name__)

_SEGMENT_MIN = 256 << 20
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
    """A scheduler's side of the node's prepare process; poll() returns each reply."""

    def __init__(self, address: str):
        self._address = address
        # Names this scheduler to the process: its holds outlive a connection.
        self._client_id = uuid.uuid4().hex
        self._process: Optional[subprocess.Popen] = None
        self._listener: Optional[socket.socket] = None
        self._connection: Optional[Connection] = None
        self._socket: Optional[socket.socket] = None
        self._layouts_sent: set = set()
        self._request: Optional[dict] = None
        self._started: Optional[float] = None
        self._pinning: Optional[Tuple[dict, threading.Thread]] = None
        self._segments: Dict[Tuple[str, int], _Segment] = {}
        self._buffer_id: Optional[str] = None
        self._holds: Dict[str, Tuple[str, List[int]]] = {}
        self._releases: List[Tuple[str, Any]] = []
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
        dtype: torch.dtype,
        scaling: float,
    ) -> None:
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

    def poll(self) -> Optional[dict]:
        """The reply once done, else None; a read once its segments are pinned."""
        if self._started is None:
            return None
        try:
            if time.monotonic() - self._started > _PREPARE_TIMEOUT_S:
                raise TimeoutError(f"no reply in {_PREPARE_TIMEOUT_S} s")
            if self._pinning is not None:
                reply, thread = self._pinning
                if thread.is_alive():
                    return None
                self._pinning = None
            else:
                if not self._connection.poll(0):
                    return None
                reply = json.loads(self._connection.recv_bytes())
                if "regions" in reply:
                    thread = self._receive_segments(reply)
                    if thread is not None:
                        self._pinning = (reply, thread)
                        return None
            if "regions" in reply:
                self._hold(reply)
        except Exception as error:
            self._disconnect()
            reply = {"error": f"LoRA prepare process: {error!r}"}
        self._started = self._request = None
        return reply

    def region_views(self, reply: dict, sizes: Sequence[int]) -> List[torch.Tensor]:
        buffer_id = reply["buffer"]
        return [
            self._segments[(buffer_id, segment)].data[offset : offset + size]
            for (segment, offset), size in zip(reply["regions"], sizes)
        ]

    def release(self, prepare_id: str) -> None:
        """The process may reuse prepare_id's regions once the queued GPU work ran."""
        self._releases.append((prepare_id, _reads_fence()))

    def _receive_segments(self, reply: dict) -> Optional[threading.Thread]:
        new = reply["segments"]
        fds: List[int] = []
        if new:
            _, fds, _, _ = socket.recv_fds(self._socket, 1, len(new))
            if len(fds) != len(new):
                for fd in fds:
                    os.close(fd)
                raise RuntimeError("the node buffer's segments did not arrive")
        if reply["buffer"] != self._buffer_id:
            # A new process: the old one's segments go once no version uses them.
            self._buffer_id = reply["buffer"]
            self._drop_unused_segments()
        to_pin = []
        for (segment, size), fd in zip(new, fds):
            key = (reply["buffer"], segment)
            if key in self._segments:
                os.close(fd)
            else:
                self._segments[key] = _Segment(fd, size)
                to_pin.append(self._segments[key])
        if not to_pin:
            return None
        thread = threading.Thread(
            target=lambda: [segment.pin() for segment in to_pin], daemon=True
        )
        thread.start()
        return thread

    def _hold(self, reply: dict) -> None:
        prepare_id = self._request["prepare_id"]
        used = sorted({segment for segment, _ in reply["regions"]})
        segments = [self._segments[(reply["buffer"], s)] for s in used]
        self._holds[prepare_id] = (reply["buffer"], used)
        for segment in segments:
            segment.held_versions += 1
        errors = [segment.error for segment in segments if segment.error is not None]
        if errors:
            self._releases.append((prepare_id, None))
            raise RuntimeError(f"pinning the node buffer failed: {errors[0]}")

    def _send_releases(self) -> None:
        done = [p for p, fence in self._releases if fence is None or fence.query()]
        if not done:
            return
        self._releases = [(p, f) for p, f in self._releases if p not in done]
        current = []
        for prepare_id in done:
            buffer_id, used = self._holds.pop(prepare_id, (self._buffer_id, []))
            for segment in used:
                self._segments[(buffer_id, segment)].held_versions -= 1
            if buffer_id == self._buffer_id:
                current.append(prepare_id)
        self._drop_unused_segments()
        if current:
            self._connection.send_bytes(
                json.dumps({"op": "release", "prepare_ids": current}).encode()
            )

    def _drop_unused_segments(self) -> None:
        for key, segment in list(self._segments.items()):
            if key[0] != self._buffer_id and segment.held_versions == 0:
                segment.unpin()
                del self._segments[key]

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
        self._send_releases()
        self._connection.send_bytes(json.dumps(request).encode())
        self._request, self._started = request, time.monotonic()

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
        self._connection.send_bytes(
            json.dumps(
                {"op": "hello", "client_id": self._client_id, "pid": os.getpid()}
            ).encode()
        )

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
        if self._request is not None and self._request["op"] == "read":
            # Its reply, and the hold it gave, may never be read here.
            self._releases.append((self._request["prepare_id"], None))
        self._started = self._request = None
        self._pinning = None


class _Segment:
    def __init__(self, fd: int, size: int):
        self.fd = fd
        self.size = size
        self.data: Optional[torch.Tensor] = None
        self.error: Optional[str] = None
        self.held_versions = 0

    def pin(self) -> None:
        try:
            data = _map(self.fd, self.size, populate=True)
            _host_register(data)
            self.data = data
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            os.close(self.fd)

    def unpin(self) -> None:
        if self.data is not None:
            _host_unregister(self.data)
            self.data = None


def _host_register(data: torch.Tensor) -> None:
    err = int(torch.cuda.cudart().cudaHostRegister(data.data_ptr(), data.numel(), 0))
    if err != 0:
        raise RuntimeError(f"cudaHostRegister({data.numel()} bytes) failed: {err}")


def _host_unregister(data: torch.Tensor) -> None:
    err = int(torch.cuda.cudart().cudaHostUnregister(data.data_ptr()))
    if err != 0:
        logger.warning(f"cudaHostUnregister of the LoRA node buffer failed: {err}")


def _reads_fence():
    """Done once the queued GPU work, host-memory copies included, ran."""
    event = torch.cuda.Event()
    event.record()
    return event


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


def _pages(nbytes: int) -> int:
    return -(-nbytes // mmap.PAGESIZE) * mmap.PAGESIZE


class _Buffer:
    """Every version's regions, in memfd segments that each GPU maps and pins once."""

    def __init__(self):
        # Names this process's buffer: an earlier process's segments are other memory.
        self.id = uuid.uuid4().hex
        # (fd, bytes mapped here) per segment.
        self.segments: List[Tuple[int, torch.Tensor]] = []
        # Per segment, its free [offset, size] extents in offset order.
        self._free: List[List[List[int]]] = []
        self._lock = threading.Lock()

    def allocate(self, nbytes: int) -> Tuple[int, int]:
        nbytes = _pages(nbytes)
        with self._lock:
            for segment, free in enumerate(self._free):
                for extent in free:
                    if extent[1] >= nbytes:
                        offset = extent[0]
                        extent[0] += nbytes
                        extent[1] -= nbytes
                        if extent[1] == 0:
                            free.remove(extent)
                        return segment, offset
            total = sum(data.numel() for _, data in self.segments)
            in_use = total - sum(size for free in self._free for _, size in free)
            # Twice what is in use, as the GPU-delta host arenas grow: every GPU
            # pins each new segment, which can stall serving.
            size = _pages(max(nbytes, 2 * (in_use + nbytes) - total, _SEGMENT_MIN))
            fd = os.memfd_create("sglang-lora-node-buffer")
            os.ftruncate(fd, size)
            self.segments.append((fd, _map(fd, size, populate=True)))
            self._free.append([[nbytes, size - nbytes]] if size > nbytes else [])
            return len(self.segments) - 1, 0

    def free(self, segment: int, offset: int, nbytes: int) -> None:
        with self._lock:
            extents = sorted(self._free[segment] + [[offset, _pages(nbytes)]])
            merged = [extents[0]]
            for start, size in extents[1:]:
                if merged[-1][0] + merged[-1][1] == start:
                    merged[-1][1] += size
                else:
                    merged.append([start, size])
            self._free[segment] = merged


class _Region:
    def __init__(self, segment: int, offset: int, nbytes: int):
        self.segment = segment
        self.offset = offset
        self.nbytes = nbytes
        self.built = threading.Event()
        self.error: Optional[str] = None


class _Version:
    def __init__(self, prepare_id: str, buffer: _Buffer):
        self.prepare_id = prepare_id
        self.buffer = buffer
        self.listed = threading.Event()
        self.tensors: Optional[List[FileTensor]] = None
        self.locations: List[Tuple[torch.dtype, int]] = []
        # Mapped only while a request uses it: no file stays open between updates.
        self.sources: Optional[List[torch.Tensor]] = None
        self.digest: Optional[str] = None
        self.error: Optional[str] = None
        self.regions: Dict[str, _Region] = {}
        self.lock = threading.Lock()
        # Requests using it now, and the clients whose GPUs hold its regions.
        self.users = 0
        self.holders: Set[str] = set()

    def claim(self, digest: str, nbytes: int) -> Tuple[_Region, bool]:
        """The region of `digest`, and whether the caller builds it."""
        with self.lock:
            region = self.regions.get(digest)
            if region is not None:
                return region, False
            region = _Region(*self.buffer.allocate(nbytes), nbytes)
            self.regions[digest] = region
            return region, True

    def free(self) -> None:
        for region in self.regions.values():
            self.buffer.free(region.segment, region.offset, region.nbytes)
        self.regions.clear()
        self.sources = None


    def map_sources(self) -> List[torch.Tensor]:
        with self.lock:
            if self.sources is None:
                self.sources = _map_file_tensors(self.tensors, self.locations)
            return self.sources

class _Node:
    def __init__(self):
        self.lock = threading.Lock()
        self.buffer = _Buffer()
        self.versions: Dict[str, _Version] = {}
        # The newest version stays until the next: its GPUs read it one by one.
        self.latest: Optional[str] = None
        # client_id -> (pid, open connections); a client gone with none loses its holds.
        self.clients: Dict[str, Tuple[int, int]] = {}

    def connect(self, client_id: str, pid: int) -> None:
        with self.lock:
            _, connections = self.clients.get(client_id, (pid, 0))
            self.clients[client_id] = (pid, connections + 1)

    def disconnect(self, client_id: str) -> None:
        with self.lock:
            pid, connections = self.clients[client_id]
            self.clients[client_id] = (pid, connections - 1)

    def use(self, prepare_id: str, create: bool) -> Tuple[Optional[_Version], bool]:
        """The version of `prepare_id`, used until done(), and whether to list it."""
        with self.lock:
            version = self.versions.get(prepare_id)
            if version is not None:
                version.users += 1
                return version, False
            if not create:
                return None, False
            self._reap()
            previous, self.latest = self.versions.get(self.latest), prepare_id
            version = self.versions[prepare_id] = _Version(prepare_id, self.buffer)
            version.users = 1
            if previous is not None:
                self._free_if_unused(previous)
            return version, True

    def done(self, version: _Version, holder: Optional[str]) -> None:
        """A request on `version` ended; `holder`'s GPU now holds its regions."""
        with self.lock:
            version.users -= 1
            if version.users == 0:
                version.sources = None
            if holder is not None:
                version.holders.add(holder)
            self._free_if_unused(version)

    def release(self, client_id: str, prepare_ids: Sequence[str]) -> None:
        with self.lock:
            for prepare_id in prepare_ids:
                version = self.versions.get(prepare_id)
                if version is not None:
                    version.holders.discard(client_id)
                    self._free_if_unused(version)

    def _reap(self) -> None:
        """Release the holds of clients gone without releasing them."""
        for client_id, (pid, connections) in list(self.clients.items()):
            if connections == 0 and not _alive(pid):
                del self.clients[client_id]
                for version in list(self.versions.values()):
                    version.holders.discard(client_id)
                    self._free_if_unused(version)

    def _free_if_unused(self, version: _Version) -> None:
        if (
            version.users == 0
            and not version.holders
            and version.prepare_id != self.latest
        ):
            version.free()
            del self.versions[version.prepare_id]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


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
    client_id: Optional[str] = None
    sent_segments: Set[int] = set()
    layouts: Dict[str, _Layout] = {}
    while True:
        try:
            request = json.loads(connection.recv_bytes())
        except (EOFError, OSError):
            break
        if request["op"] == "hello":
            client_id = request["client_id"]
            node.connect(client_id, request["pid"])
            continue
        if request["op"] == "release":
            node.release(client_id, request["prepare_ids"])
            continue
        version, lists = node.use(request["prepare_id"], create=request["op"] == "list")
        holder = None
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
                reply = _read(
                    version=version,
                    layout=layout,
                    dtype=dtype,
                    scaling=request["scaling"],
                )
                holder = client_id
        except Exception as error:
            reply = {"error": f"{type(error).__name__}: {error}"}
        new: List[int] = []
        if "regions" in reply:
            new = sorted({segment for segment, _ in reply["regions"]} - sent_segments)
            reply["buffer"] = node.buffer.id
            reply["segments"] = [(s, node.buffer.segments[s][1].numel()) for s in new]
        # Before the reply: a GPU that has it finds the files unmapped unless another
        # request reads them.
        if version is not None:
            node.done(version, holder)
        try:
            connection.send_bytes(json.dumps(reply).encode())
            if new:
                socket.send_fds(
                    sock, [b"\0"], [node.buffer.segments[s][0] for s in new]
                )
                sent_segments.update(new)
        except OSError:
            break
    connection.close()
    sock.close()
    if client_id is not None:
        node.disconnect(client_id)


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


def _read(
    version: _Version, layout: _Layout, dtype: torch.dtype, scaling: float
) -> dict:
    """Build this GPU's regions no request built yet; wait for the others."""
    regions = []
    for (_, numel), copies, digest in zip(
        layout.regions, layout.copies, layout.digests
    ):
        nbytes = numel * dtype.itemsize
        region, builds = version.claim(digest, nbytes)
        if builds:
            try:
                data = version.buffer.segments[region.segment][1]
                read_slot_weights(
                    data=data[region.offset : region.offset + nbytes].view(dtype),
                    copies=copies,
                    sources=version.map_sources(),
                    scaling=scaling,
                )
            except Exception as error:
                region.error = f"{type(error).__name__}: {error}"
            finally:
                region.built.set()
        else:
            region.built.wait()
        if region.error is not None:
            raise RuntimeError(region.error)
        regions.append((region.segment, region.offset))
    return {"regions": regions}


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
