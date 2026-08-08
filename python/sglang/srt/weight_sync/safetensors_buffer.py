"""Validated safetensors views over bounded CPU or CUDA byte buffers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

MAX_SAFETENSORS_HEADER_BYTES = 100 << 20


def _safetensors_dtypes() -> dict[str, torch.dtype]:
    result = {
        "BOOL": torch.bool,
        "I8": torch.int8,
        "U8": torch.uint8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
        "C64": torch.complex64,
    }
    optional = {
        "U16": "uint16",
        "U32": "uint32",
        "U64": "uint64",
        "F8_E4M3": "float8_e4m3fn",
        "F8_E4M3FNUZ": "float8_e4m3fnuz",
        "F8_E5M2": "float8_e5m2",
        "F8_E5M2FNUZ": "float8_e5m2fnuz",
        "F8_E8M0": "float8_e8m0fnu",
        "F4": "float4_e2m1fn_x2",
    }
    for code, name in optional.items():
        dtype = getattr(torch, name, None)
        if dtype is not None:
            result[code] = dtype
    return result


_SAFETENSORS_DTYPES = _safetensors_dtypes()


@dataclass(frozen=True)
class SafetensorsEntry:
    dtype: torch.dtype
    dtype_code: str
    shape: tuple[int, ...]
    relative_begin: int
    relative_end: int


@dataclass(frozen=True)
class SafetensorsLayout:
    data_offset: int
    file_nbytes: int
    tensors: dict[str, SafetensorsEntry]


def parse_safetensors_header(
    *,
    header_nbytes: int,
    header_bytes: bytes,
    file_nbytes: int,
) -> tuple[SafetensorsLayout, dict[str, str]]:
    data_offset = 8 + header_nbytes
    if header_nbytes > MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError(
            "safetensors header exceeds "
            f"{MAX_SAFETENSORS_HEADER_BYTES} bytes: {header_nbytes}"
        )
    if (
        header_nbytes <= 0
        or len(header_bytes) != header_nbytes
        or data_offset > file_nbytes
    ):
        raise ValueError(
            "invalid safetensors header length: "
            f"header={header_nbytes} file={file_nbytes}"
        )
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not an object")
    file_metadata: Any = header.pop("__metadata__", {})
    if not isinstance(file_metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in file_metadata.items()
    ):
        raise ValueError("invalid safetensors file metadata")

    tensors = {}
    for name, tensor_metadata in header.items():
        if not isinstance(name, str) or not isinstance(tensor_metadata, dict):
            raise ValueError("invalid safetensors tensor metadata")
        dtype_code = tensor_metadata.get("dtype")
        dtype = _SAFETENSORS_DTYPES.get(dtype_code)
        if dtype is None:
            raise TypeError(f"unsupported safetensors dtype {dtype_code!r}")
        shape = tensor_metadata.get("shape")
        offsets = tensor_metadata.get("data_offsets")
        if (
            not isinstance(shape, list)
            or not all(isinstance(value, int) and value >= 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
        ):
            raise ValueError(f"invalid safetensors metadata for {name!r}")
        relative_begin, relative_end = offsets
        begin = data_offset + relative_begin
        end = data_offset + relative_end
        if relative_begin < 0 or begin > end or end > file_nbytes:
            raise ValueError(
                f"safetensors offsets are out of bounds for {name!r}: {offsets}"
            )

        shape_tuple = tuple(shape)
        if dtype_code == "F4":
            if not shape_tuple or shape_tuple[-1] % 2:
                raise ValueError(
                    f"F4 tensor {name!r} must have an even final dimension"
                )
            storage_shape = shape_tuple[:-1] + (shape_tuple[-1] // 2,)
        else:
            storage_shape = shape_tuple
        expected_bytes = (
            math.prod(storage_shape) * torch.empty((), dtype=dtype).element_size()
        )
        actual_bytes = relative_end - relative_begin
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"tensor byte size mismatch for {name!r}: "
                f"expected={expected_bytes} actual={actual_bytes}"
            )
        tensors[name] = SafetensorsEntry(
            dtype=dtype,
            dtype_code=dtype_code,
            shape=storage_shape,
            relative_begin=relative_begin,
            relative_end=relative_end,
        )

    cursor = 0
    for name, entry in sorted(
        tensors.items(),
        key=lambda item: (
            item[1].relative_begin,
            item[1].relative_end,
            item[0],
        ),
    ):
        if entry.relative_begin != cursor:
            relation = (
                "overlaps another tensor"
                if entry.relative_begin < cursor
                else "leaves a gap"
            )
            raise ValueError(
                f"safetensors range for {name!r} {relation}: "
                f"expected_begin={cursor} actual_begin={entry.relative_begin}"
            )
        cursor = entry.relative_end
    data_nbytes = file_nbytes - data_offset
    if cursor != data_nbytes:
        raise ValueError(
            "safetensors tensor ranges do not cover the data buffer: "
            f"covered={cursor} data={data_nbytes}"
        )
    return (
        SafetensorsLayout(
            data_offset=data_offset,
            file_nbytes=file_nbytes,
            tensors=tensors,
        ),
        file_metadata,
    )


def parse_safetensors_layout(
    *,
    header_nbytes: int,
    header_bytes: bytes,
    file_nbytes: int,
) -> SafetensorsLayout:
    return parse_safetensors_header(
        header_nbytes=header_nbytes,
        header_bytes=header_bytes,
        file_nbytes=file_nbytes,
    )[0]


def read_safetensors_layout(path: str | Path) -> SafetensorsLayout:
    path = Path(path)
    file_nbytes = path.stat().st_size
    with path.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError(f"safetensors source is shorter than its header: {path}")
        header_nbytes = int.from_bytes(prefix, "little")
        if header_nbytes > MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(
                "safetensors header exceeds "
                f"{MAX_SAFETENSORS_HEADER_BYTES} bytes: {path}"
            )
        header_bytes = file.read(header_nbytes)
    return parse_safetensors_layout(
        header_nbytes=header_nbytes,
        header_bytes=header_bytes,
        file_nbytes=file_nbytes,
    )


class SafetensorsBuffer:
    """Expose zero-copy tensor views from one bounded byte tensor."""

    def __init__(
        self,
        buffer: torch.Tensor,
        *,
        layout: SafetensorsLayout | None = None,
    ):
        if (
            buffer.device.type not in {"cpu", "cuda"}
            or buffer.dtype != torch.uint8
            or buffer.ndim != 1
            or not buffer.is_contiguous()
        ):
            raise ValueError("safetensors source must be contiguous CPU or CUDA bytes")
        if layout is None:
            if buffer.device.type != "cpu":
                raise ValueError("safetensors headers must be parsed from CPU bytes")
            layout = self._parse_layout(buffer)
        if layout.file_nbytes != buffer.numel():
            raise ValueError(
                "safetensors layout file size differs from source buffer: "
                f"layout={layout.file_nbytes} buffer={buffer.numel()}"
            )
        if layout.data_offset > buffer.numel():
            raise ValueError(
                "safetensors layout exceeds source buffer: "
                f"data_offset={layout.data_offset} file={buffer.numel()}"
            )
        self.buffer = buffer
        self.layout = layout

    @staticmethod
    def _parse_layout(buffer: torch.Tensor) -> SafetensorsLayout:
        if buffer.numel() < 8:
            raise ValueError("safetensors source is shorter than its header prefix")
        prefix = buffer[:8].numpy().tobytes()
        header_nbytes = int.from_bytes(prefix, "little")
        if header_nbytes > MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(
                "safetensors header exceeds "
                f"{MAX_SAFETENSORS_HEADER_BYTES} bytes: {header_nbytes}"
            )
        data_offset = 8 + header_nbytes
        return parse_safetensors_layout(
            header_nbytes=header_nbytes,
            header_bytes=buffer[8:data_offset].numpy().tobytes(),
            file_nbytes=buffer.numel(),
        )

    def get_tensor(self, name: str) -> torch.Tensor:
        entry = self.layout.tensors.get(name)
        if entry is None:
            raise KeyError(f"safetensors source has no tensor {name!r}")
        begin = self.layout.data_offset + entry.relative_begin
        end = self.layout.data_offset + entry.relative_end
        try:
            return self.buffer[begin:end].view(entry.dtype).reshape(entry.shape)
        except RuntimeError as exc:
            raise ValueError(f"cannot construct safetensors view for {name!r}") from exc

    def get_tensor_bytes(self, name: str) -> torch.Tensor:
        """Return canonical encoded bytes for an in-place source transform."""

        entry = self.layout.tensors.get(name)
        if entry is None:
            raise KeyError(f"safetensors source has no tensor {name!r}")
        begin = self.layout.data_offset + entry.relative_begin
        end = self.layout.data_offset + entry.relative_end
        return self.buffer[begin:end]
