from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load as load_safetensors

from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket


def normalize_weight_update_transport(
    *,
    load_format: Optional[str],
    transport_format: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    if transport_format is None and load_format == "flattened_bucket":
        return None, "flattened_bucket"
    return load_format, transport_format


def _tensor_num_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _bucket_named_tensors(
    named_tensors: List[Tuple[str, torch.Tensor]],
    *,
    max_bucket_bytes: Optional[int],
) -> List[List[Tuple[str, torch.Tensor]]]:
    if max_bucket_bytes is None:
        return [named_tensors]
    if max_bucket_bytes <= 0:
        raise ValueError("transport_bucket_bytes must be a positive integer.")

    buckets: List[List[Tuple[str, torch.Tensor]]] = []
    current_bucket: List[Tuple[str, torch.Tensor]] = []
    current_bucket_bytes = 0
    for item in named_tensors:
        tensor_bytes = _tensor_num_bytes(item[1])
        if current_bucket and current_bucket_bytes + tensor_bytes > max_bucket_bytes:
            buckets.append(current_bucket)
            current_bucket = []
            current_bucket_bytes = 0
        current_bucket.append(item)
        current_bucket_bytes += tensor_bytes
        if current_bucket_bytes >= max_bucket_bytes:
            buckets.append(current_bucket)
            current_bucket = []
            current_bucket_bytes = 0
    if current_bucket:
        buckets.append(current_bucket)
    return buckets


def _serialize_flattened_bucket_transport(
    named_tensors: List[Tuple[str, torch.Tensor]],
    *,
    tp_size: int,
    transport_bucket_bytes: Optional[int],
) -> tuple[List[bytes], Dict[str, Any]]:
    bucket_named_tensors = _bucket_named_tensors(
        named_tensors,
        max_bucket_bytes=transport_bucket_bytes,
    )
    bucket_payloads = []
    bucket_sizes_bytes = []
    total_tensor_bytes = 0
    max_tensor_bytes = 0
    for bucket_items in bucket_named_tensors:
        bucket = FlattenedTensorBucket(named_tensors=bucket_items)
        flattened_tensor = bucket.get_flattened_tensor()
        bucket_payloads.append(
            {
                "flattened_tensor": flattened_tensor,
                "metadata": bucket.get_metadata(),
            }
        )
        bucket_bytes = int(flattened_tensor.numel())
        bucket_sizes_bytes.append(bucket_bytes)
        for _, tensor in bucket_items:
            tensor_bytes = _tensor_num_bytes(tensor)
            total_tensor_bytes += tensor_bytes
            max_tensor_bytes = max(max_tensor_bytes, tensor_bytes)

    serialized_payload = MultiprocessingSerializer.serialize(bucket_payloads)
    return (
        [serialized_payload for _ in range(tp_size)],
        {
            "bucket_count": len(bucket_payloads),
            "bucket_sizes_bytes": bucket_sizes_bytes,
            "transport_bucket_bytes": transport_bucket_bytes,
            "tensor_count": len(named_tensors),
            "total_tensor_bytes": total_tensor_bytes,
            "max_tensor_bytes": max_tensor_bytes,
        },
    )


def load_named_tensors_from_bytes(
    payload: bytes,
    tensor_format: str = "safetensors",
) -> List[Tuple[str, torch.Tensor]]:
    if tensor_format != "safetensors":
        raise ValueError(f"Unsupported tensor_format={tensor_format!r}.")

    tensor_dict = load_safetensors(payload)
    return list(tensor_dict.items())


def build_update_weights_request_from_named_tensors(
    named_tensors: List[Tuple[str, torch.Tensor]],
    *,
    tp_size: int,
    load_format: Optional[str] = None,
    transport_format: Optional[str] = None,
    transport_bucket_bytes: Optional[int] = None,
    flush_cache: bool = True,
    abort_all_requests: bool = False,
    base_weight_version: Optional[str] = None,
    weight_version: Optional[str] = None,
    payload_digest: Optional[str] = None,
    loader_metadata: Optional[Dict[str, Any]] = None,
    crash_on_error: bool = False,
    disable_draft_model: Optional[bool] = None,
) -> UpdateWeightsFromTensorReqInput:
    load_format, transport_format = normalize_weight_update_transport(
        load_format=load_format,
        transport_format=transport_format,
    )
    supported_params = inspect.signature(UpdateWeightsFromTensorReqInput).parameters
    effective_loader_metadata = loader_metadata
    effective_transport_encoding = transport_format
    if (
        transport_format == "flattened_bucket"
        and "transport_format" not in supported_params
    ):
        effective_loader_metadata = dict(loader_metadata or {})
        if load_format is not None:
            effective_loader_metadata.setdefault("inner_load_format", load_format)
        load_format = "flattened_bucket"
        transport_format = None
    if effective_transport_encoding == "flattened_bucket":
        serialized_named_tensors, transport_metadata = (
            _serialize_flattened_bucket_transport(
                named_tensors,
                tp_size=tp_size,
                transport_bucket_bytes=transport_bucket_bytes,
            )
        )
    elif transport_format is None:
        serialized_named_tensors = [
            MultiprocessingSerializer.serialize(named_tensors) for _ in range(tp_size)
        ]
        transport_metadata = None
    else:
        raise ValueError(f"Unsupported transport_format={transport_format!r}.")

    request_kwargs = {
        "serialized_named_tensors": serialized_named_tensors,
        "load_format": load_format,
        "transport_format": transport_format,
        "transport_metadata": transport_metadata,
        "flush_cache": flush_cache,
        "abort_all_requests": abort_all_requests,
        "base_weight_version": base_weight_version,
        "weight_version": weight_version,
        "payload_digest": payload_digest,
        "loader_metadata": effective_loader_metadata,
        "crash_on_error": crash_on_error,
        "disable_draft_model": disable_draft_model,
    }
    filtered_kwargs = {
        key: value for key, value in request_kwargs.items() if key in supported_params
    }
    return UpdateWeightsFromTensorReqInput(**filtered_kwargs)
