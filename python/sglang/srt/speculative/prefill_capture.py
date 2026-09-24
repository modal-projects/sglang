"""Bounded, opt-in DFlash target-feature capture. Linux/CUDA, one producer.

The file is a MAP_SHARED arena registered with CUDA, not a CUDA IPC allocation.
All shared metadata transitions use flock(LOCK_EX). A consumer claims READY ->
READING, reads outside the lock, then releases READING -> FREE with the matching
generation. Payloads remain immutable until release. See the ABI constants below;
JSON descriptors contain byte offsets, never process-local pointers.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import mmap
import os
import queue
import struct
import threading
import uuid
import weakref
from collections import Counter
from dataclasses import dataclass

import torch

from sglang.srt.speculative.capture_identity import HASH_SCHEME, PrefixCursor

logger = logging.getLogger(__name__)

MAGIC = b"SGDFCAP\0"
ABI_VERSION = 3
HEADER_BYTES = 4096
SLOT_HEADER_BYTES = 4096
# magic, version, JSON length, total arena bytes; JSON follows this prefix.
FILE_HEADER = struct.Struct("<8sIIQ")
# state, JSON length, generation; JSON follows this prefix.
SLOT_HEADER = struct.Struct("<IIQ")
FREE, COPYING, READY, READING = range(4)


def _json_bytes(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _align(value, alignment=4096):
    return (value + alignment - 1) // alignment * alignment


@dataclass
class _Capture:
    request: weakref.ReferenceType
    slot: int
    generation: int
    rid: str
    prompt_length: int
    start: int
    end: int
    event: torch.cuda.Event
    seal_reason: str | None = None
    kind: str = "prefill_tail"
    prefix: dict | None = None
    receipt: dict | None = None
    target_epoch: str = ""
    block_size: int = 0


@dataclass
class _VerifyOutput:
    owner: PrefillCapture
    tickets: tuple

    def copy_to_host(self, copy_tensor):
        # Copies are already enqueued; publication waits for their completion.
        return self

    def consume(self, batch, commits):
        for index, capture in self.tickets:
            req = batch.reqs[index]
            commit = commits[index]
            if (
                capture.request() is not req
                or commit is None
                or commit.output_index < 1
                or req.is_retracted
                or capture.target_epoch != self.owner._target_epoch
            ):
                capture.receipt = {"discard": True}
                continue
            start = len(req.origin_input_ids) + commit.output_index - 1
            capture.receipt = {
                "start": start,
                "prefix": self.owner._prefix_for(req, start),
                "expected_tokens": (
                    req.output_ids[commit.output_index - 1],
                    *commit.token_ids,
                ),
                "output_index": commit.output_index,
                "output_tokens": commit.token_ids,
                "finished": req.finished(),
            }


class PrefillCapture:
    """Best-effort prompt tails and committed decode rows, without GPU staging.

    `offer` must run on the stream producing `hidden_states`, before any reuse of
    its storage. Only graceful shutdown synchronizes the host with CUDA. A full
    arena drops capture work instead of waiting. One scheduler thread offers
    data; the publisher alone reclaims shared slots into bounded local queues.
    Pending-map snapshots and entry replacement use CPython's atomic dict
    operations; only sealed captures are read by the publisher.
    """

    def __init__(
        self,
        *,
        path: str,
        slots: int,
        window: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        metadata: dict,
        sample_rate: float = 1.0,
        verify_slots: int | None = None,
        teacher_hidden_size: int = 0,
        overlap: bool = False,
    ):
        if min(slots, window, hidden_size) <= 0:
            raise ValueError("Capture slots, window, and hidden_size must be positive.")
        if verify_slots is not None and verify_slots <= 0:
            raise ValueError("Capture verify_slots must be positive.")
        if teacher_hidden_size < 0:
            raise ValueError("Capture teacher_hidden_size must be nonnegative.")
        if not 0 <= sample_rate <= 1:
            raise ValueError("Capture sample_rate must be between 0 and 1.")
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"Unsupported capture dtype: {dtype}")
        self.device = torch.device(device)
        if self.device.type != "cuda" or torch.version.hip:
            raise ValueError("Prefill capture currently requires CUDA.")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        self.path = os.path.abspath(path)
        self.slots, self.window, self.hidden_size = slots, window, hidden_size
        self.teacher_hidden_size = teacher_hidden_size
        self.capacity = max(window, metadata.get("block_size", 1))
        self.verify_capacity = metadata.get("block_size", min(window, 16))
        self.verify_slots = max(64, 8 * slots) if verify_slots is None else verify_slots
        self.dtype, self.sample_rate = dtype, sample_rate
        self.stats = Counter()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pending: dict[int, _Capture] = {}
        self._available = (queue.SimpleQueue(), queue.SimpleQueue())
        self._thread = None
        self._fd = None
        self._mapping = None
        self._host = None
        self._registered = False
        self._disabled = False
        self._warned_invalid = False
        # Opt-in only for a caller that fences target-buffer reuse. Metadata
        # copies stay on the producing stream because draft/position/accept
        # buffers can be reused before the next target forward.
        self._copy_stream = torch.cuda.Stream(device=self.device) if overlap else None
        self._last_copy_event = None

        itemsize = torch.empty((), dtype=dtype, device="meta").element_size()
        self._verify_offset = SLOT_HEADER_BYTES + self.capacity * 8
        self._aux_offset = _align(self._verify_offset + 16, 64)
        self._teacher_offset = _align(
            self._aux_offset + self.capacity * hidden_size * itemsize, 64
        )
        self._stride = _align(
            self._teacher_offset + self.capacity * teacher_hidden_size * itemsize
        )
        self._decode_offset = HEADER_BYTES + slots * self._stride
        self._decode_scalars = SLOT_HEADER_BYTES + self.verify_capacity * 8
        self._decode_aux = _align(self._decode_scalars + 16, 64)
        self._decode_teacher = _align(
            self._decode_aux + self.verify_capacity * hidden_size * itemsize, 64
        )
        self._decode_stride = _align(
            self._decode_teacher + self.verify_capacity * teacher_hidden_size * itemsize
        )
        self.nbytes = self._decode_offset + self.verify_slots * self._decode_stride
        self._target_epoch = uuid.uuid4().hex
        self.descriptor = {
            "kind": "dflash_target_features",
            "producer_pid": os.getpid(),
            "producer_epoch": uuid.uuid4().hex,
            "initial_target_epoch": self._target_epoch,
            "slots": slots,
            "window": window,
            "capacity": self.capacity,
            "prefix_hash_scheme": HASH_SCHEME,
            "hidden_size": hidden_size,
            "dtype": str(dtype).removeprefix("torch."),
            "byte_order": "little",
            "slot_offset": HEADER_BYTES,
            "slot_stride": self._stride,
            "slot_header_bytes": SLOT_HEADER_BYTES,
            "token_offset": SLOT_HEADER_BYTES,
            "token_dtype": "int64",
            "aux_offset": self._aux_offset,
            "verify_slots": self.verify_slots,
            "verify_slot_offset": self._decode_offset,
            "verify_slot_stride": self._decode_stride,
            "verify_aux_offset": self._decode_aux,
            "verify_capacity": self.verify_capacity,
            "metadata": metadata,
            "payload_copy_stream": "separate" if overlap else "producing",
        }
        if teacher_hidden_size:
            self.descriptor.update(
                teacher_hidden_size=teacher_hidden_size,
                teacher_dtype=self.descriptor["dtype"],
                teacher_offset=self._teacher_offset,
                verify_teacher_offset=self._decode_teacher,
                teacher_state="normalized_lm_head_input",
                teacher_distribution="raw_model",
                teacher_prediction_offset=1,
            )
        # A producer epoch conservatively isolates unresolved revisions/local
        # weights across restarts. Never infer weight equality from a model path.
        self._identity_config = {
            "producer_epoch": self.descriptor["producer_epoch"],
            "target_model": metadata.get("target_model"),
            "target_revision": metadata.get("target_revision"),
            "target_config_hash": metadata.get("target_config_hash"),
            "target_layer_ids": metadata.get("target_layer_ids"),
            "dtype": self.descriptor["dtype"],
            "hidden_size": hidden_size,
            "teacher_hidden_size": teacher_hidden_size,
            "position_semantics": "causal-text-zero-based",
        }
        descriptor = _json_bytes(self.descriptor)
        if len(descriptor) > HEADER_BYTES - FILE_HEADER.size:
            raise ValueError("Capture descriptor exceeds header capacity.")

        # Exclusive creation avoids truncating a live consumer's arena or silently
        # reusing a stale producer epoch. Prefer a path under /dev/shm (tmpfs).
        self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        self._inode = os.fstat(self._fd).st_ino
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            # Reserve backing space now, so a later D2H cannot SIGBUS on ENOSPC.
            os.posix_fallocate(self._fd, 0, self.nbytes)
            self._mapping = mmap.mmap(self._fd, self.nbytes, flags=mmap.MAP_SHARED)
            self._host = torch.frombuffer(self._mapping, dtype=torch.uint8)
            from cuda.bindings import runtime as cuda_rt

            self._cuda_rt = cuda_rt
            with torch.cuda.device(self.device):
                self._check_cuda(
                    cuda_rt.cudaHostRegister(
                        self._host.data_ptr(),
                        self.nbytes,
                        cuda_rt.cudaHostRegisterPortable,
                    )
                )
            self._registered = True
            self._mapping[FILE_HEADER.size : FILE_HEADER.size + len(descriptor)] = (
                descriptor
            )
            FILE_HEADER.pack_into(
                self._mapping,
                0,
                MAGIC,
                ABI_VERSION if teacher_hidden_size else 2,
                len(descriptor),
                self.nbytes,
            )
            self._claim_free_slots_locked()
        except BaseException:
            self._dispose()
            raise
        finally:
            if self._fd is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)

        self._thread = threading.Thread(
            target=self._poll, name="dflash-feature-capture", daemon=True
        )
        self._thread.start()
        logger.info(
            "DFlash feature capture: %s, %d prefill slots x %d tokens, "
            "%d verify slots x %d tokens, %.1f MiB pinned",
            self.path,
            slots,
            window,
            self.verify_slots,
            self.verify_capacity,
            self.nbytes / 2**20,
        )

    @staticmethod
    def _check_cuda(result):
        if int(result[0]) != 0:
            raise RuntimeError(f"CUDA host-memory registration error: {result[0]}")

    def _try_metadata_lock(self):
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def _slot_offset(self, slot):
        if slot >= self.slots:
            return self._decode_offset + (slot - self.slots) * self._decode_stride
        return HEADER_BYTES + slot * self._stride

    def _slot_aux_offset(self, slot):
        return self._aux_offset if slot < self.slots else self._decode_aux

    def invalidate_target(self):
        """New identity after any attempted target weight update, even failure."""
        self.wait_before_target_forward()
        self._target_epoch = uuid.uuid4().hex

    def wait_before_target_forward(self):
        """GPU-only fence before target graph outputs can be overwritten.

        Draft computation can overlap payload D2H because it only reads target
        features or already-projected draft KV. The next target forward must
        wait before reusing graph outputs. Allocator lifetime is handled
        separately by record_stream on the actual copy stream.
        """
        if self._last_copy_event is not None:
            torch.cuda.current_stream(self.device).wait_event(self._last_copy_event)
            self._last_copy_event = None

    def _payload_stream(self, producing_stream):
        if self._copy_stream is None:
            return producing_stream
        self._copy_stream.wait_stream(producing_stream)
        return self._copy_stream

    def _eligible(self, req):
        # Mutable adapters and non-token input semantics need their own versioned
        # identities. Skip them instead of silently conflating causal prefixes.
        if (
            any(
                getattr(req, field, None) is not None
                for field in (
                    "multimodal_inputs",
                    "input_embeds",
                    "positional_embed_overrides",
                    "lora_id",
                    "token_type_ids",
                )
            )
            or req.is_retracted
        ):
            return False
        if self.sample_rate < 1:
            digest = hashlib.blake2b(req.rid.encode(), digest_size=8).digest()
            return int.from_bytes(digest, "little") < self.sample_rate * 2**64
        return True

    def _prefix_for(self, req, end):
        # Per-Req O(1) state; no history table growing with generated tokens.
        epoch = (self.descriptor["producer_epoch"], self._target_epoch)
        saved = getattr(req, "_dflash_capture_cursor", None)
        if saved is None or saved[0] != epoch or saved[1].position > end:
            namespace = hashlib.sha256(
                _json_bytes(
                    {
                        **self._identity_config,
                        "target_epoch": self._target_epoch,
                        "extra_key": getattr(req, "extra_key", None),
                        "cache_salt": getattr(req, "cache_salt", None),
                    }
                )
            ).hexdigest()
            saved = (epoch, PrefixCursor(namespace))
            req._dflash_capture_cursor = saved
        cursor = saved[1]
        prompt_len = len(req.origin_input_ids)
        while cursor.position < end:
            stop = min(end, cursor.position + 1024)
            if cursor.position < prompt_len:
                stop = min(stop, prompt_len)
                tokens = req.origin_input_ids[cursor.position : stop]
            else:
                tokens = req.output_ids[
                    cursor.position - prompt_len : stop - prompt_len
                ]
            if len(tokens) != stop - cursor.position:
                raise ValueError("Capture prefix exceeds committed request tokens")
            cursor.advance(tokens)
        return cursor.snapshot()

    def _claim_free_slots_locked(self):
        # The publisher owns shared metadata. Claim FREE slots ahead of time,
        # then hand bounded producer-owned tickets to the scheduler. COPY/offer
        # never acquires flock or contends with the publisher/consumer.
        for slot in range(self.slots + self.verify_slots):
            offset = self._slot_offset(slot)
            state, _, generation = SLOT_HEADER.unpack_from(self._mapping, offset)
            if state != FREE:
                continue
            generation += 1
            SLOT_HEADER.pack_into(self._mapping, offset, COPYING, 0, generation)
            self._available[slot >= self.slots].put((slot, generation))

    def _refill_available(self):
        if self._try_metadata_lock():
            try:
                self._claim_free_slots_locked()
            finally:
                fcntl.flock(self._fd, fcntl.LOCK_UN)

    def _reserve(self, req, start, prompt_length, *, verify=False, event=None):
        if len(_json_bytes(req.rid)) > 1024:
            self.stats["dropped_metadata"] += 1
            return None
        try:
            slot, generation = self._available[verify].get_nowait()
        except queue.Empty:
            self.stats["dropped_full"] += 1
            return None
        capture = _Capture(
            weakref.ref(req),
            slot,
            generation,
            req.rid,
            prompt_length,
            start,
            start,
            torch.cuda.Event() if event is None else event,
            kind="verify_committed" if verify else "prefill_tail",
            target_epoch=self._target_epoch,
        )
        self._pending[slot] = capture
        self.stats["admitted"] += 1
        return capture

    def _valid_teacher(self, teacher, rows):
        return not self.teacher_hidden_size or (
            teacher is not None
            and teacher.device == self.device
            and teacher.dtype == self.dtype
            and teacher.ndim == 2
            and teacher.shape[0] >= rows
            and teacher.shape[1] == self.teacher_hidden_size
            and teacher.is_contiguous()
        )

    def _copy_teacher(self, slot, teacher, source_start, count, written, stream):
        if not self.teacher_hidden_size:
            return
        offset = self._slot_offset(slot) + (
            self._teacher_offset if slot < self.slots else self._decode_teacher
        )
        row_bytes = self.teacher_hidden_size * teacher.element_size()
        offset += written * row_bytes
        self._check_cuda(
            self._cuda_rt.cudaMemcpyAsync(
                self._host.data_ptr() + offset,
                teacher.data_ptr() + source_start * row_bytes,
                count * row_bytes,
                self._cuda_rt.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream.cuda_stream,
            )
        )

    def offer(
        self,
        reqs,
        prefix_lens,
        extend_lens,
        hidden_states,
        *,
        teacher_hidden_states=None,
    ):
        """Offer rows from one target prefill batch (padding rows are ignored)."""
        if self._disabled or self._stop.is_set() or self.sample_rate == 0:
            return
        if not (
            hidden_states.is_cuda
            and hidden_states.device == self.device
            and hidden_states.dtype == self.dtype
            and hidden_states.ndim == 2
            and hidden_states.shape[1] == self.hidden_size
            and hidden_states.shape[0] >= sum(extend_lens)
            and hidden_states.is_contiguous()
            and len(reqs) == len(prefix_lens) == len(extend_lens)
            and self._valid_teacher(teacher_hidden_states, sum(extend_lens))
        ):
            if not self._warned_invalid:
                logger.warning(
                    "Skipping prefill capture: unexpected feature tensor layout."
                )
                self._warned_invalid = True
            self.stats["dropped_layout"] += 1
            return
        if torch.cuda.is_current_stream_capturing():
            return
        producing_stream = torch.cuda.current_stream(hidden_states.device)
        stream = None
        row = 0
        for req, prefix, length in zip(reqs, prefix_lens, extend_lens):
            prompt_length = len(req.origin_input_ids)
            start = max(prefix, prompt_length - self.window, 0)
            end = min(prefix + length, prompt_length)
            row_start = row + start - prefix
            row += length
            # Only initial text prefill in this version; no decode/retraction
            # reconstruction, multimodal embeddings, or prompt regeneration.
            if end <= start or req.output_ids or not self._eligible(req):
                continue
            capture = next(
                (
                    c
                    for c in tuple(self._pending.values())
                    if c.request() is req and c.kind == "prefill_tail"
                ),
                None,
            )
            if capture is None:
                capture = self._reserve(req, start, prompt_length)
                if capture is not None:
                    capture.prefix = self._prefix_for(req, start)
            if capture is None or capture.seal_reason is not None:
                continue
            if start != capture.end or prompt_length != capture.prompt_length:
                capture.seal_reason = "gap"
                continue
            count = end - start
            written = capture.end - capture.start
            offset = self._slot_offset(capture.slot)
            struct.pack_into(
                f"<{count}q",
                self._mapping,
                offset + SLOT_HEADER_BYTES + written * 8,
                *req.origin_input_ids[start:end],
            )
            if stream is None:
                stream = self._payload_stream(producing_stream)
            row_bytes = self.hidden_size * hidden_states.element_size()
            self._check_cuda(
                self._cuda_rt.cudaMemcpyAsync(
                    self._host.data_ptr()
                    + offset
                    + self._aux_offset
                    + written * row_bytes,
                    hidden_states.data_ptr() + row_start * row_bytes,
                    count * row_bytes,
                    self._cuda_rt.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream.cuda_stream,
                )
            )
            hidden_states.record_stream(stream)
            self._copy_teacher(
                capture.slot, teacher_hidden_states, row_start, count, written, stream
            )
            if self.teacher_hidden_size:
                teacher_hidden_states.record_stream(stream)
            capture.event.record(stream)
            if self._copy_stream is not None:
                self._last_copy_event = capture.event
            capture.end = end
            if end == prompt_length:
                capture.seal_reason = "prefill"

    def offer_verify(
        self,
        reqs,
        hidden_states,
        token_ids,
        positions,
        commit_lens,
        *,
        teacher_hidden_states=None,
    ):
        """Copy bounded verify blocks now, publish only after scheduler commit.

        Copying the whole block avoids a GPU gather or host synchronization for
        accept lengths. Rejected rows are outside the published token_count.
        """
        if self._disabled or self._stop.is_set() or self.sample_rate == 0:
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        block = hidden_states.shape[1] if hidden_states.ndim == 3 else 0
        if not (
            hidden_states.ndim == 3
            and hidden_states.shape[0] == len(reqs)
            and hidden_states.shape[2] == self.hidden_size
            and hidden_states.dtype == self.dtype
            and hidden_states.device == self.device
            and hidden_states.is_contiguous()
            and block > 0
            and block <= self.verify_capacity
            and self._valid_teacher(teacher_hidden_states, len(reqs) * block)
            and token_ids.shape == (len(reqs), block)
            and token_ids.dtype == torch.int64
            and token_ids.stride(-1) == 1
            and positions.numel() == len(reqs) * block
            and positions.dtype == torch.int64
            and commit_lens.numel() == len(reqs)
            and commit_lens.ndim == 1
            and commit_lens.dtype == torch.int32
            and all(
                t.device == self.device for t in (token_ids, positions, commit_lens)
            )
        ):
            self.stats["dropped_layout"] += len(reqs)
            return None
        tickets = []
        producing_stream = torch.cuda.current_stream(self.device)
        positions = positions.view(len(reqs), block)
        # One event covers the batch, including fragmented slots. The payload
        # stream waits for producing-stream work before copying feature rows.
        event = torch.cuda.Event()
        for i, req in enumerate(reqs):
            if not self._eligible(req):
                continue
            capture = self._reserve(
                req, -1, len(req.origin_input_ids), verify=True, event=event
            )
            if capture is None:
                continue
            capture.kind = "verify_committed"
            capture.block_size = block
            capture.seal_reason = "verify"
            tickets.append((i, capture))
        if tickets:
            # Consecutive source requests and destination slots form a pitched
            # copy. Preserve per-request metadata and commit validation while
            # avoiding O(batch) tensor slices/views and tiny memcpy launches.
            itemsize = hidden_states.element_size()
            sources = [
                (
                    token_ids.data_ptr(),
                    token_ids.stride(0) * 8,
                    SLOT_HEADER_BYTES,
                    block * 8,
                ),
                (
                    positions.data_ptr(),
                    positions.stride(0) * 8,
                    self._decode_scalars,
                    8,
                ),
                (
                    commit_lens.data_ptr(),
                    commit_lens.stride(0) * 4,
                    self._decode_scalars + 8,
                    4,
                ),
                (
                    hidden_states.data_ptr(),
                    block * self.hidden_size * itemsize,
                    self._decode_aux,
                    block * self.hidden_size * itemsize,
                ),
            ]
            if self.teacher_hidden_size:
                size = block * self.teacher_hidden_size * itemsize
                sources.append(
                    (teacher_hidden_states.data_ptr(), size, self._decode_teacher, size)
                )
            runs = []
            run_start = 0
            for stop in range(1, len(tickets) + 1):
                if stop < len(tickets) and (
                    tickets[stop][0] == tickets[stop - 1][0] + 1
                    and tickets[stop][1].slot == tickets[stop - 1][1].slot + 1
                ):
                    continue
                index, first = tickets[run_start]
                destination = self._host.data_ptr() + self._slot_offset(first.slot)
                runs.append((index, destination, stop - run_start))
                run_start = stop
            if self._copy_stream is None:
                stream = producing_stream
                self._copy_verify_runs(runs, sources, stream)
            else:
                self._copy_verify_runs(runs, sources[:3], producing_stream)
                stream = self._payload_stream(producing_stream)
                self._copy_verify_runs(runs, sources[3:], stream)
            event.record(stream)
            if self._copy_stream is not None:
                self._last_copy_event = event
            for tensor in (token_ids, positions, commit_lens):
                tensor.record_stream(producing_stream)
            hidden_states.record_stream(stream)
            if self.teacher_hidden_size:
                teacher_hidden_states.record_stream(stream)
        return _VerifyOutput(self, tuple(tickets)) if tickets else None

    def _copy_verify_runs(self, runs, sources, stream):
        for index, destination, count in runs:
            for pointer, pitch, offset, width in sources:
                self._check_cuda(
                    self._cuda_rt.cudaMemcpy2DAsync(
                        destination + offset,
                        self._decode_stride,
                        pointer + index * pitch,
                        pitch,
                        width,
                        count,
                        self._cuda_rt.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        stream.cuda_stream,
                    )
                )

    def _verify_info(self, capture):
        receipt = capture.receipt
        if receipt.get("discard") or capture.target_epoch != self._target_epoch:
            return None
        offset = self._slot_offset(capture.slot)
        start, accepted = struct.unpack_from(
            "<qi", self._mapping, offset + self._decode_scalars
        )
        count = min(accepted, len(receipt["expected_tokens"]))
        actual = (
            struct.unpack_from(f"<{count}q", self._mapping, offset + SLOT_HEADER_BYTES)
            if 0 < count <= capture.block_size
            else ()
        )
        if (
            start != receipt["start"]
            or not actual
            or actual != receipt["expected_tokens"][:count]
        ):
            self.stats["dropped_commit_mismatch"] += 1
            return None
        return {
            "rid": capture.rid,
            "kind": capture.kind,
            "prompt_length": capture.prompt_length,
            "start": start,
            "end": start + count,
            "token_count": count,
            "prefix": receipt["prefix"],
            "proposed_count": capture.block_size - 1,
            "accepted_draft_count": accepted - 1,
            "output_index": receipt["output_index"],
            "output_tokens": receipt["output_tokens"],
            "finished": receipt["finished"],
        }

    def _sealed(self):
        # Snapshot the producer's pending entries. Sealed copies cannot be
        # appended/reused; the local lock serializes publisher/test operations.
        sealed = []
        for capture in tuple(self._pending.values()):
            if capture.kind == "verify_committed" and capture.receipt is None:
                if capture.request() is not None:
                    continue
                capture.receipt = {"discard": True}
            if capture.seal_reason is None:
                req = capture.request()
                if req is None or req.finished() or req.is_retracted or req.output_ids:
                    capture.seal_reason = "interrupted"
            if capture.seal_reason is not None:
                sealed.append(capture)
        return sealed

    def _prepare_publication(self, sealed):
        completed = []
        for capture in sealed:
            if not capture.event.query():
                continue
            expected_start = max(0, capture.prompt_length - self.window)
            info_dict = (
                self._verify_info(capture)
                if capture.kind == "verify_committed"
                else {
                    "rid": capture.rid,
                    "kind": "prefill_tail",
                    "prompt_length": capture.prompt_length,
                    "expected_start": expected_start,
                    "start": capture.start,
                    "end": capture.end,
                    "token_count": capture.end - capture.start,
                    "tail_complete": capture.start == expected_start
                    and capture.end == capture.prompt_length,
                    "seal_reason": capture.seal_reason,
                    "prefix": capture.prefix,
                }
            )
            if info_dict is not None:
                info_dict["target_epoch"] = capture.target_epoch
            completed.append(
                (capture, _json_bytes(info_dict) if info_dict is not None else None)
            )
        return completed

    def _publish_completed(self, completed=None):
        # Caller holds the local lock, serializing flock's open-file description.
        # Only tests query events here; the poller does that outside this lock.
        if completed is None:
            completed = self._prepare_publication(self._sealed())
        if not completed or not self._try_metadata_lock():
            return
        try:
            for capture, info in completed:
                if self._pending.get(capture.slot) is not capture:
                    continue
                offset = self._slot_offset(capture.slot)
                if info is None or capture.target_epoch != self._target_epoch:
                    SLOT_HEADER.pack_into(
                        self._mapping, offset, FREE, 0, capture.generation
                    )
                    del self._pending[capture.slot]
                    self.stats["discarded"] += 1
                    continue
                if len(info) > SLOT_HEADER_BYTES - SLOT_HEADER.size:
                    SLOT_HEADER.pack_into(
                        self._mapping, offset, FREE, 0, capture.generation
                    )
                    del self._pending[capture.slot]
                    self.stats["dropped_metadata"] += 1
                    continue
                self._mapping[
                    offset + SLOT_HEADER.size : offset + SLOT_HEADER.size + len(info)
                ] = info
                SLOT_HEADER.pack_into(
                    self._mapping, offset, READY, len(info), capture.generation
                )
                del self._pending[capture.slot]
                self.stats["published"] += 1
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)

    def _poll(self):
        try:
            with torch.cuda.device(self.device):
                while not self._stop.wait(0.002):
                    if self._lock.acquire(blocking=False):
                        try:
                            sealed = self._sealed()
                        finally:
                            self._lock.release()
                        # Keep event queries and serialization outside metadata
                        # transactions. Admission only takes preclaimed tickets.
                        completed = self._prepare_publication(sealed)
                        if self._lock.acquire(blocking=False):
                            try:
                                self._publish_completed(completed)
                                self._refill_available()
                            finally:
                                self._lock.release()
        except Exception:
            # Keep the arena registered; in-flight copies may still reference it.
            # The scheduler can continue, and graceful shutdown drains the events.
            self._disabled = True
            logger.exception(
                "Disabling feature capture after completion polling failed."
            )

    def close(self):
        """Drain outstanding copies and release registration on graceful shutdown."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            if self._fd is None:
                return
            with torch.cuda.device(self.device):
                if self._copy_stream is not None:
                    self._copy_stream.synchronize()
                for capture in self._pending.values():
                    capture.event.synchronize()
                self._pending.clear()
                self._dispose()
        logger.info("DFlash feature capture closed: %s", dict(self.stats))

    def _dispose(self):
        if self._registered:
            self._check_cuda(self._cuda_rt.cudaHostUnregister(self._host.data_ptr()))
            self._registered = False
        self._host = None
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._fd is not None:
            try:
                if os.stat(self.path).st_ino == self._inode:
                    os.unlink(self.path)
            except FileNotFoundError:
                pass
            os.close(self._fd)
            self._fd = None
