from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from safetensors.torch import load_file

from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.auth import AuthLevel, auth_level

MERGE_LOADER = "sglang.srt.model_loader.lora_merge_loader.merge_lora_tensors_inplace"

router = APIRouter()


class LocalLoraMergeRequest(BaseModel):
    adapter_config_path: str
    adapter_weights_path: str
    strict: bool = True
    flush_cache: bool = False
    atomic_pause_mode: Literal["abort", "retract", "in_place"] | None = "in_place"
    abort_all_requests: bool = False
    weight_version: str | None = None
    prestage_before_pause: bool = False
    peak_device_bytes: int | str | None = None
    vram_headroom_gb: float | None = None


def _load_adapter(
    config_path: str, weights_path: str
) -> tuple[dict[str, Any], list[tuple[str, Any]]]:
    with open(Path(config_path), "r") as f:
        adapter_config = json.load(f)
    adapter_tensors = list(load_file(str(Path(weights_path)), device="cpu").items())
    return adapter_config, adapter_tensors


@router.post("/admin/update_merged_lora_from_local")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_merged_lora_from_local(req: LocalLoraMergeRequest, request: Request):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    started_at = time.monotonic()
    load_started_at = time.monotonic()
    adapter_config, adapter_tensors = await asyncio.to_thread(
        _load_adapter, req.adapter_config_path, req.adapter_weights_path
    )
    adapter_load_ms = (time.monotonic() - load_started_at) * 1000

    from sglang.srt.entrypoints.http_server import get_global_state

    tokenizer_manager = get_global_state().tokenizer_manager
    tp_size = tokenizer_manager.server_args.tp_size
    serialize_started_at = time.monotonic()
    serialized = MultiprocessingSerializer.serialize(adapter_tensors)
    serialize_ms = (time.monotonic() - serialize_started_at) * 1000
    manifest = {
        "adapter_config": adapter_config,
        "strict": req.strict,
        "lora_merge_prestage_before_pause": req.prestage_before_pause,
    }
    if req.peak_device_bytes is not None:
        manifest["lora_merge_peak_device_bytes"] = req.peak_device_bytes
    if req.vram_headroom_gb is not None:
        manifest["lora_merge_vram_headroom_gb"] = req.vram_headroom_gb

    update_req = UpdateWeightsFromTensorReqInput(
        serialized_named_tensors=[serialized for _ in range(tp_size)],
        manifest=manifest,
        load_format=MERGE_LOADER,
        flush_cache=req.flush_cache,
        atomic_pause_mode=req.atomic_pause_mode,
        abort_all_requests=req.abort_all_requests,
        weight_version=req.weight_version or req.adapter_weights_path,
        trace={
            "request_id": request_id,
            "source": "local_lora_merge_router",
            "adapter_config_path": req.adapter_config_path,
            "adapter_weights_path": req.adapter_weights_path,
            "trace_start_monotonic": time.monotonic(),
        },
    )

    update_started_at = time.monotonic()
    success, message = await tokenizer_manager.update_weights_from_tensor(
        update_req, request
    )
    update_ms = (time.monotonic() - update_started_at) * 1000
    total_ms = (time.monotonic() - started_at) * 1000
    content = {
        "success": success,
        "message": message,
        "request_id": request_id,
        "weight_version": update_req.weight_version,
        "tensor_count": len(adapter_tensors),
        "trace": {
            "adapter_load_ms": round(adapter_load_ms, 3),
            "serialize_ms": round(serialize_ms, 3),
            "update_weights_ms": round(update_ms, 3),
            "total_ms": round(total_ms, 3),
            "sglang_update_trace": update_req.trace,
        },
    }
    return ORJSONResponse(content, status_code=200 if success else 400)
