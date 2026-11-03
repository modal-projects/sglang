"""CPU-only snapshots for the SGLang scheduler.

The scheduler captures immutable CPU metadata and changed token bytes. A
separate process hashes and publishes observations; one acknowledged capture
may be in flight. No tensors or cache access state are modified. Caches without
the native delta publish load only, and any capture failure disables
observations instead of stopping the scheduler.
Enable explicitly with SGLANG_SPINDLE_ROUTING_OBSERVATIONS=1.
"""

import atexit
import json
import logging
import os
import pickle
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import xxhash

if __name__ != "__main__":
    from sglang.srt.runtime_context import get_parallel


ENABLED = os.environ.get("SGLANG_SPINDLE_ROUTING_OBSERVATIONS") == "1"
PATH = Path("/tmp/spindle-routing-work.json")
publisher = None
last_snapshot = 0.0
sequence = 0
load_only_logged = False


class IncrementalPrefixes:
    """Private publisher state; never reads the live engine tree."""

    def __init__(self):
        self.tokens = {}
        self.hashes = {}
        self.root = None

    def export(self, root, rows, adapter_names, hybrid):
        if root != self.root:
            self.tokens.clear()
            self.hashes.clear()
            self.root = root
        live = {row[0] for row in rows}
        self.tokens = {key: value for key, value in self.tokens.items() if key in live}
        children = {}
        for row in rows:
            node_id, parent, _, token_bytes, _, _, length = row
            if token_bytes is not None:
                self.tokens[node_id] = token_bytes
            if node_id not in self.tokens or len(self.tokens[node_id]) != length * 8:
                raise ValueError(
                    "Incomplete routing delta; publisher must restart with a full inventory"
                )
            children.setdefault(parent, []).append(row)
        boundaries_by_hash = {}
        fresh_hashes = {}
        stack = [(row, None, 0, None) for row in children.get(root, ())]
        while stack:
            row, parent_digest, start, parent_key = stack.pop()
            node_id, _, extra_key, _, mamba, accessed, length = row
            adapter = adapter_names.get(extra_key)
            if adapter is None:
                continue
            tokens = self.tokens[node_id]
            context = (
                adapter,
                start,
                parent_digest.hexdigest() if parent_digest else None,
            )
            previous = self.hashes.get(node_id)
            if (
                previous is not None
                and previous[0] is tokens
                and previous[1] == context
            ):
                digest, boundaries = previous[2:]
            else:
                digest = (
                    parent_digest.copy()
                    if parent_digest is not None
                    else xxhash.xxh3_128((adapter + "\0").encode())
                )
                offset = 0
                boundaries = []
                view = memoryview(tokens)
                for end in range((start // 64 + 1) * 64, start + length + 1, 64):
                    take = (end - start) * 8
                    digest.update(view[offset:take])
                    boundaries.append((digest.hexdigest(), end))
                    offset = take
                digest.update(view[offset:])
            fresh_hashes[node_id] = (tokens, context, digest, boundaries)
            end_node = start + length
            for key, end in boundaries:
                reusable = not hybrid or (mamba and end == end_node)
                boundaries_by_hash[key] = (parent_key, reusable, accessed)
                parent_key = key
            stack.extend(
                (child, digest, end_node, parent_key)
                for child in children.get(node_id, ())
            )
        self.hashes = fresh_hashes
        return select_prefixes(boundaries_by_hash)


def select_prefixes(boundaries, limit=8192):
    """Retain recent endpoints and their bounded reusable ancestors.

    The router consumes only hashes. An ancestor budget preserves shorter
    matches when long recent branches fill the endpoint budget.
    """
    endpoints = sorted(
        (
            (accessed, key)
            for key, (_, reusable, accessed) in boundaries.items()
            if reusable
        ),
        reverse=True,
    )
    selected = set()
    for key in sorted(boundaries, key=lambda key: boundaries[key][2], reverse=True):
        if len(selected) == limit:
            break
        if not boundaries[key][1]:
            continue
        path = []
        while key is not None and key not in selected:
            path.append(key)
            key = boundaries[key][0]
        if len(selected) + len(path) <= limit:
            selected.update(path)
    return sorted(
        {key for _, key in endpoints[:limit]}
        | {key for key in selected if boundaries[key][1]}
    )


def send_packet(connection, packet):
    payload = pickle.dumps(packet, protocol=5)
    connection.sendall(struct.pack("!Q", len(payload)))
    connection.sendall(payload)


def receive_packet(connection):
    def receive(size):
        chunks = bytearray()
        while len(chunks) < size:
            chunk = connection.recv(size - len(chunks))
            if not chunk:
                raise EOFError("Cache publisher pipe closed")
            chunks.extend(chunk)
        return chunks

    (size,) = struct.unpack("!Q", receive(8))
    return pickle.loads(receive(size))


def publish_process(connection, path):
    """One CPU-only subprocess, with no Torch/CUDA imports or scheduler GIL."""
    prefixes = IncrementalPrefixes()
    while True:
        try:
            state, root, rows, names, hybrid = receive_packet(connection)
        except EOFError:
            return
        started = time.monotonic()
        state["prefixes"] = prefixes.export(root, rows, names, hybrid)
        state["export_processing_seconds"] = time.monotonic() - started
        state["publication_seconds"] = time.time() - state["timestamp"]
        # Keep the capture timestamp. A slow publisher must not make an old
        # cache look freshly observed. The normal router freshness limit is 3s.
        if state["publication_seconds"] < 2.5:
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, separators=(",", ":")))
            os.replace(temporary, path)
        send_packet(connection, state["sequence"])


class BackgroundPublisher:
    """One acknowledged delta in flight; skip sampling instead of queueing work.

    A private pipe and subprocess isolate hashing, sorting, JSON and file I/O.
    The feeder thread handles pipe writes and process startup. Only immutable
    CPU observations cross the boundary. On failure the next capture supplies
    every token segment again, so no dropped-delta replay is required.
    """

    def __init__(self, path=PATH):
        self.path = path
        self.jobs = Queue(maxsize=1)
        self.acks = Queue(maxsize=1)
        self.known = {}
        self.available = False
        self.tree = None
        self.closed = threading.Event()
        self.process = None
        self.failures = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def ready(self, tree):
        try:
            self.known = self.acks.get_nowait()
            self.available = True
        except Empty:
            pass
        if self.available and tree is not self.tree:
            self.known = {}
            self.tree = tree
        return self.available and not self.closed.is_set()

    def submit(self, packet):
        assert self.available
        self.available = False
        self.jobs.put_nowait(packet)

    def close(self):
        self.closed.set()
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def run(self):
        while not self.closed.is_set():
            connection = None
            try:
                parent, child = socket.socketpair()
                connection = parent
                connection.settimeout(5)
                try:
                    self.process = subprocess.Popen(
                        [
                            sys.executable,
                            __file__,
                            str(child.fileno()),
                            str(self.path),
                        ],
                        pass_fds=(child.fileno(),),
                        stdin=subprocess.DEVNULL,
                    )
                finally:
                    child.close()
                self.acks.put({})
                while not self.closed.is_set():
                    try:
                        packet = self.jobs.get(timeout=1)
                    except Empty:
                        continue
                    send_packet(connection, packet)
                    if receive_packet(connection) != packet[0]["sequence"]:
                        raise ValueError(
                            "Cache publisher acknowledged the wrong capture"
                        )
                    self.acks.put({row[0]: row[6] for row in packet[2]})
            except Exception:
                self.failures += 1
                logging.exception(
                    "Cache observation publisher failed; rebuilding its inventory"
                )
            finally:
                if connection is not None:
                    connection.close()
                process = self.process
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            self.closed.wait(1)


def native_delta(cache):
    """The incremental export, or None when this cache or binary lacks it."""
    if type(cache).__name__ != "UnifiedRadixCache" or cache.is_swa_enabled:
        return None
    tree_core = cache.tree_core
    if type(tree_core).__name__ != "RustUnifiedTreeCore":
        return None
    return getattr(tree_core._binding, "routing_snapshot_delta", None)


def snapshot(scheduler, running_batch, last_batch):
    global ENABLED
    if not ENABLED:
        return
    try:
        capture(scheduler, running_batch, last_batch)
    except Exception:
        ENABLED = False
        logging.exception("Spindle routing observations failed and are now disabled")


def capture(scheduler, running_batch, last_batch):
    global last_snapshot, sequence, publisher, load_only_logged
    if (
        get_parallel().tp_rank != 0
        or get_parallel().dp_size != 1
        or get_parallel().pp_size != 1
    ):
        return
    now = time.monotonic()
    if now - last_snapshot < 1:
        return
    last_snapshot = now
    if not scheduler.enable_lora:
        return
    captured = time.time()
    manager = scheduler.tp_worker.model_runner.lora_manager
    names = {uid: ref.lora_name for uid, ref in manager.lora_refs.items()}
    cache = scheduler.tree_cache
    if publisher is None:
        publisher = BackgroundPublisher()
    if not publisher.ready(cache):
        return
    delta = native_delta(cache)
    if delta is not None:
        root, cache_rows = delta(publisher.known)
        hybrid = cache.is_mamba_enabled
    else:
        # Copying a Python tree here costs the scheduler O(cached tokens) each
        # second, so these workers report load and adapters without prefixes.
        if not load_only_logged:
            load_only_logged = True
            logging.warning(
                "Spindle routing observations omit prefixes for %s",
                type(cache).__name__,
            )
        root, cache_rows, hybrid = None, [], False
    requests = {r.rid: r for r in scheduler.waiting_queue}
    for batch in (running_batch, last_batch):
        if batch is not None:
            requests.update((r.rid, r) for r in batch.reqs)
    if scheduler.chunked_req is not None:
        requests[scheduler.chunked_req.rid] = scheduler.chunked_req
    rows = []
    for req in requests.values():
        if req.finished():
            continue
        decoded = len(req.output_ids)
        rows.append(
            dict(
                rid=req.rid,
                adapter=names.get(req.lora_id, "base"),
                prefill_tokens=0
                if decoded
                else max(0, len(req.origin_input_ids) - len(req.prefix_indices)),
                decode_tokens=max(0, req.sampling_params.max_new_tokens - decoded),
            )
        )
    sequence += 1
    # These are scheduler-owned CPU counters. Evictable cached blocks are not
    # pinned work, so a full cache alone must not look like an overloaded worker.
    pool = scheduler.pool_stats_observer.get_pool_stats()
    state = dict(
        timestamp=captured,
        sequence=sequence,
        incarnation=os.getpid(),
        requests=rows,
        cpu_adapters=[names[uid] for uid in manager.loras if uid in names],
        gpu_adapters=[
            names[uid] for uid in manager.memory_pool.uid_to_buffer_id if uid in names
        ],
        snapshot_seconds=time.monotonic() - now,
        cache_kind=type(scheduler.tree_cache).__name__,
        running_requests=len(running_batch.reqs) if running_batch is not None else 0,
        waiting_requests=len(scheduler.waiting_queue),
        kv_used_tokens=pool.full_num_used,
        kv_capacity_tokens=(
            pool.full_num_used + pool.full_available_size + pool.full_evictable_size
        ),
    )
    state["export_token_bytes"] = sum(
        len(row[3]) for row in cache_rows if row[3] is not None
    )
    state["publisher_failures"] = publisher.failures
    publisher.submit((state, root, cache_rows, names, hybrid))


if __name__ == "__main__":
    publish_process(socket.socket(fileno=int(sys.argv[1])), Path(sys.argv[2]))
