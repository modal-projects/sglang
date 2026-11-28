"""Unit tests for int16 routed-experts caches and compact wire dtypes."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.srt.state_capturer.base import BaseDeviceCache
from sglang.srt.state_capturer.routed_experts import (
    _infer_num_routed_experts,
    encode_routed_experts_tensor,
    extract_routed_experts_from_meta_info,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_device_cache_int16_halves_memory():
    c32 = BaseDeviceCache(
        max_batch_size=64, num_layers=4, topk_size=8, device="cpu", name="test32"
    )
    c16 = BaseDeviceCache(
        max_batch_size=64,
        num_layers=4,
        topk_size=8,
        device="cpu",
        name="test16",
        dtype=torch.int16,
    )
    assert c16.buffer.dtype == torch.int16
    assert c16.get_buffer_size_bytes() * 2 == c32.get_buffer_size_bytes()
    src = torch.arange(16, dtype=torch.int32).reshape(2, 8)
    c16.capture(0, src)
    assert torch.equal(c16.buffer[:2, 0, :].to(torch.int32), src)


@pytest.mark.parametrize(
    ("wire_dtype", "expected_dtype", "bytes_per_elem"),
    [
        (None, "int32", 4),
        ("int32", "int32", 4),
        ("int16", "int16", 2),
        ("uint16", "uint16", 2),
        ("uint8", "uint8", 1),
    ],
)
def test_encode_and_extract_routed_experts_wire_dtypes(
    wire_dtype, expected_dtype, bytes_per_elem
):
    tensor = torch.arange(48, dtype=torch.int16).reshape(3, 2, 8)
    b64, actual = encode_routed_experts_tensor(tensor, wire_dtype)
    assert actual == expected_dtype
    meta = {"routed_experts": b64}
    if wire_dtype is not None:
        meta["routed_experts_dtype"] = actual
    decoded = extract_routed_experts_from_meta_info({"meta_info": meta})
    assert decoded.dtype.itemsize == bytes_per_elem
    np.testing.assert_array_equal(decoded, tensor.numpy().reshape(-1))


def test_unsigned_wire_dtypes_fall_back_safely_to_int16_or_int32():
    wide = torch.tensor([[[0, 287], [1, 255]]], dtype=torch.int16)
    b64, actual = encode_routed_experts_tensor(wide, "uint8")
    assert actual == "int16"
    decoded = extract_routed_experts_from_meta_info(
        {"meta_info": {"routed_experts": b64, "routed_experts_dtype": actual}}
    )
    assert decoded.dtype == np.int16
    np.testing.assert_array_equal(decoded, wide.numpy().reshape(-1))

    padded = torch.tensor([[[-1, -1], [2, 3]]], dtype=torch.int16)
    for req_dtype in ("uint8", "uint16"):
        b64_pad, actual_pad = encode_routed_experts_tensor(padded, req_dtype)
        assert actual_pad == "int16"
        decoded_pad = extract_routed_experts_from_meta_info(
            {
                "meta_info": {
                    "routed_experts": b64_pad,
                    "routed_experts_dtype": actual_pad,
                }
            }
        )
        assert decoded_pad.dtype == np.int16
        np.testing.assert_array_equal(decoded_pad, padded.numpy().reshape(-1))


def test_infer_num_routed_experts_and_invalid_dtype():
    assert (
        _infer_num_routed_experts(
            SimpleNamespace(hf_text_config=SimpleNamespace(n_routed_experts=288))
        )
        == 288
    )
    assert (
        _infer_num_routed_experts(
            SimpleNamespace(hf_text_config=SimpleNamespace(num_experts=128))
        )
        == 128
    )
    with pytest.raises(ValueError, match="unsupported routed_experts_dtype"):
        encode_routed_experts_tensor(
            torch.zeros((1, 1, 1), dtype=torch.int16), "float32"
        )
