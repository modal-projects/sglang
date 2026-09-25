"""Bounded binary upload for a remote draft updater."""

import asyncio
import json
import tempfile
from pathlib import Path

from fastapi import HTTPException

from sglang.srt.managers.io_struct import UpdateDraftWeightsReqInput
from sglang.srt.weight_sync.draft_delta import (
    HEADER,
    MAGIC,
    MAX_DELTA_BYTES,
    MAX_MANIFEST_BYTES,
    validate_manifest,
)

# Keep a strong reference after the frontend request is cancelled. asyncio's
# task registry itself uses weak references.
_upload_tasks = set()


async def receive_upload(request, path):
    """Stream a bounded envelope and opaque delta bytes to server-owned storage."""
    content_length = request.headers.get("content-length")
    if (
        content_length is not None
        and int(content_length) > HEADER.size + MAX_MANIFEST_BYTES + MAX_DELTA_BYTES
    ):
        raise HTTPException(status_code=413, detail="Draft delta upload is too large")
    header, needed, envelope, written = bytearray(), HEADER.size, None, 0
    metadata_length = None
    with path.open("wb") as handle:
        async for chunk in request.stream():
            remaining = memoryview(chunk)
            while remaining:
                if envelope is None:
                    count = min(needed - len(header), len(remaining))
                    header.extend(remaining[:count])
                    remaining = remaining[count:]
                    if len(header) != needed:
                        continue
                    if metadata_length is None:
                        magic, metadata_length = HEADER.unpack(header)
                        if (
                            magic != MAGIC
                            or not 0 < metadata_length <= MAX_MANIFEST_BYTES
                        ):
                            raise ValueError("Invalid draft delta upload header")
                        needed += metadata_length
                        continue
                    envelope = json.loads(header[HEADER.size :])
                    if not isinstance(envelope, dict) or set(envelope) != {
                        "manifest",
                        "expected_draft_version",
                    }:
                        raise ValueError("Invalid draft delta envelope")
                    version = envelope["expected_draft_version"]
                    if type(version) is not int or version < 0:
                        raise ValueError(
                            "expected_draft_version must be a nonnegative integer"
                        )
                    validate_manifest(envelope["manifest"])
                    header.clear()
                else:
                    written += len(remaining)
                    if written > envelope["manifest"]["delta"]["compressed_bytes"]:
                        raise ValueError("Trailing bytes in draft delta upload")
                    await asyncio.to_thread(handle.write, remaining)
                    remaining = remaining[len(remaining) :]
    if envelope is None or written != envelope["manifest"]["delta"]["compressed_bytes"]:
        raise ValueError("Truncated draft delta upload")
    return envelope


async def _upload_and_update(manager, request):
    # At most one upload/staged file per frontend. Streaming goes to ordinary
    # host storage, with no model-sized GPU allocation or base64 expansion.
    async with manager.draft_delta_upload_lock:
        with tempfile.TemporaryDirectory(prefix="sglang-draft-delta-") as directory:
            path = Path(directory) / "model.delta.zst"
            envelope = await receive_upload(request, path)
            return await manager.update_draft_weights(
                UpdateDraftWeightsReqInput(
                    action="apply",
                    manifest=envelope["manifest"],
                    expected_draft_version=envelope["expected_draft_version"],
                    payload_path=str(path),
                )
            )


async def upload_and_update(manager, request):
    # A lost client must not delete a file still being read by TP workers or
    # cancel their acknowledgement waiter. The owned task retains both the
    # upload lock and temporary directory until the scheduler replies.
    task = asyncio.create_task(_upload_and_update(manager, request))
    _upload_tasks.add(task)
    task.add_done_callback(_upload_tasks.discard)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:

        def completed(future):
            if not future.cancelled():
                future.exception()

        task.add_done_callback(completed)
        raise
