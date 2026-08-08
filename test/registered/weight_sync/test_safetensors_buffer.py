from __future__ import annotations

import json
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file

import sglang.srt.weight_sync.file_io as file_io
from sglang.srt.weight_sync.file_io import read_file_into_tensor
from sglang.srt.weight_sync.safetensors_buffer import SafetensorsBuffer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_file_reader_and_buffer_reconstruct_tensor_views(tmp_path):
    tensors = {
        "layer.a": torch.arange(17, dtype=torch.uint8),
        "layer.b": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "layer.c": torch.arange(8, dtype=torch.bfloat16),
    }
    if hasattr(torch, "float4_e2m1fn_x2"):
        tensors["layer.fp4"] = (
            torch.arange(8, dtype=torch.uint8)
            .view(torch.float4_e2m1fn_x2)
            .reshape(2, 4)
        )
    path = tmp_path / "model.safetensors"
    save_file(tensors, path)
    source = torch.empty(path.stat().st_size, dtype=torch.uint8)

    with mock.patch.object(file_io, "_POSITIONAL_IO_CHUNK_BYTES", 16):
        result = read_file_into_tensor(path, source)

    assert result.wall_s >= 0
    parsed = SafetensorsBuffer(source)
    copied = SafetensorsBuffer(source.clone(), layout=parsed.layout)
    for name, expected in tensors.items():
        for actual in (parsed.get_tensor(name), copied.get_tensor(name)):
            if expected.dtype == getattr(torch, "float4_e2m1fn_x2", None):
                torch.testing.assert_close(
                    actual.view(torch.uint8),
                    expected.view(torch.uint8),
                )
            else:
                torch.testing.assert_close(actual, expected)

    entry = parsed.layout.tensors["layer.b"]
    expected_ptr = source.data_ptr() + parsed.layout.data_offset + entry.relative_begin
    assert parsed.get_tensor("layer.b").data_ptr() == expected_ptr


def test_invalid_header_length_fails_loudly():
    source = torch.tensor(
        list((101 << 20).to_bytes(8, "little")) + [0] * 8,
        dtype=torch.uint8,
    )

    with pytest.raises(ValueError, match="header exceeds"):
        SafetensorsBuffer(source)


def test_overlapping_tensor_ranges_fail_loudly():
    header = json.dumps(
        {
            "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
            "b": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]},
        }
    ).encode()
    source = torch.tensor(
        list(len(header).to_bytes(8, "little")) + list(header) + [1, 2, 3],
        dtype=torch.uint8,
    )

    with pytest.raises(ValueError, match="overlaps another tensor"):
        SafetensorsBuffer(source)


def test_file_reader_rejects_non_byte_and_out_of_bounds_targets(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"abcd")

    with pytest.raises(ValueError, match="contiguous CPU bytes"):
        file_io.read_range_into_tensor(
            path,
            torch.empty(4, dtype=torch.int8),
            file_offset=0,
        )
    with pytest.raises(ValueError, match="source range exceeds"):
        file_io.read_range_into_tensor(
            path,
            torch.empty(4, dtype=torch.uint8),
            file_offset=1,
        )
