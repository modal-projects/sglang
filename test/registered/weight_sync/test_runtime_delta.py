import json
import os
import struct
import tempfile
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import zstandard

from sglang.srt.weight_sync.runtime_delta import (
    RuntimeDeltaCoverageError,
    RuntimeDeltaPlan,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    _interleave_trtllm_nvfp4_scales,
    _shuffle_trtllm_epilogue_rows,
)


@dataclass
class _Segment:
    image_offset: int
    device_bytes: torch.Tensor


class _CopyModel(torch.nn.Module):
    def __init__(self, *, quantized: bool = False):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.zeros(8, dtype=torch.int32),
            requires_grad=False,
        )
        self.weight.weight_loader = self._load
        if quantized:
            self.quant_method = _QuantAdapterWithoutApply()

    @staticmethod
    def _load(parameter, loaded_weight):
        # The loader receives the tagged source and performs its sharding
        # internally, matching SGLang's tensor-parallel loaders.
        parameter.copy_(loaded_weight[2:10])

    def load_weights(self, weights):
        for name, tensor in weights:
            assert name == "weight"
            self.weight.weight_loader(self.weight, tensor)


class _QuantAdapterWithoutApply:
    @staticmethod
    def host_runtime_delta_parameter_names(layer):
        return ("weight",)


class _PaddedCopyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.full((10,), -1, dtype=torch.int32),
            requires_grad=False,
        )
        self.weight.weight_loader = self._load

    @staticmethod
    def _load(parameter, loaded_weight):
        parameter[: loaded_weight.shape[0]].copy_(loaded_weight)
        parameter[loaded_weight.shape[0] :].fill_(0)

    def load_weights(self, weights):
        for _, tensor in weights:
            self.weight.weight_loader(self.weight, tensor)


def _storage_bytes(tensor: torch.Tensor) -> torch.Tensor:
    storage = tensor.untyped_storage()
    return torch.empty(0, dtype=torch.uint8).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def _write_delta(root: str, old: torch.Tensor, new: torch.Tensor) -> None:
    version_dir = os.path.join(root, "weight_v000001")
    os.makedirs(version_dir)
    delta = np.bitwise_xor(
        old.contiguous().view(torch.uint8).numpy(),
        new.contiguous().view(torch.uint8).numpy(),
    ).tobytes()
    compressed = zstandard.ZstdCompressor().compress(delta)
    header = {
        "weight": {
            "dtype": "U8",
            "shape": [len(compressed)],
            "data_offsets": [0, len(compressed)],
        }
    }
    encoded = json.dumps(header).encode()
    filename = "model-00000-of-00001.safetensors"
    with open(os.path.join(version_dir, filename), "wb") as file:
        file.write(struct.pack("<Q", len(encoded)))
        file.write(encoded)
        file.write(compressed)
    with open(
        os.path.join(version_dir, "model.safetensors.index.json"), "w"
    ) as file:
        json.dump(
            {
                "metadata": {
                    "version": "000001",
                    "base_version": "000000",
                    "delta_encoding": "xor",
                    "compression_format": "zstd",
                    "checksum_format": "adler32",
                },
                "weight_map": {"weight": filename},
            },
            file,
        )


def _record_and_finalize(model: _CopyModel, source: torch.Tensor):
    plan = RuntimeDeltaPlan()
    plan.record(model, [("weight", source)])
    segment = _Segment(image_offset=0, device_bytes=_storage_bytes(model.weight))
    stats = plan.finalize(model, [segment])
    return plan, stats


def test_direct_xor_advances_tp_view_byte_exactly():
    old = torch.arange(12, dtype=torch.int32)
    new = old.clone()
    new[3] = 700
    new[8] = -123
    model = _CopyModel()
    plan, stats = _record_and_finalize(model, old)
    assert stats["direct_sources"] == 1

    host_image = _storage_bytes(model.weight).clone()
    with tempfile.TemporaryDirectory() as source_dir:
        _write_delta(source_dir, old, new)
        plan.apply_versions(
            model=model,
            host_image=host_image,
            source_dir=source_dir,
            base_version=0,
            target_version=1,
        )

    expected = new[2:10]
    actual = host_image.view(torch.int32)
    assert torch.equal(actual, expected)


def test_invariant_zero_padding_does_not_require_a_hook():
    source = torch.arange(8, dtype=torch.int32)
    model = _PaddedCopyModel()
    plan = RuntimeDeltaPlan()
    plan.record(model, [("weight", source)])
    segment = _Segment(image_offset=0, device_bytes=_storage_bytes(model.weight))
    stats = plan.finalize(model, [segment])
    assert stats["direct_sources"] == 1


def test_quantized_destination_requires_explicit_hook_before_mutation():
    old = torch.arange(12, dtype=torch.int32)
    new = old.clone()
    new[4] = 999
    model = _CopyModel(quantized=True)
    plan, stats = _record_and_finalize(model, old)
    assert stats["hook_sources"] == 1

    host_image = _storage_bytes(model.weight).clone()
    before = host_image.clone()
    with tempfile.TemporaryDirectory() as source_dir:
        _write_delta(source_dir, old, new)
        with pytest.raises(RuntimeDeltaCoverageError):
            plan.apply_versions(
                model=model,
                host_image=host_image,
                source_dir=source_dir,
                base_version=0,
                target_version=1,
            )
    assert torch.equal(host_image, before)


def test_trtllm_row_shuffle_matches_flashinfer_index_definition():
    value = torch.arange(64 * 7, dtype=torch.int32).reshape(64, 7)
    gated = (
        value.reshape(2, 32, 7).transpose(0, 1).reshape(64, 7)
    )
    src_to_dst = torch.tensor(
        [
            0,
            8,
            16,
            24,
            1,
            9,
            17,
            25,
            2,
            10,
            18,
            26,
            3,
            11,
            19,
            27,
            4,
            12,
            20,
            28,
            5,
            13,
            21,
            29,
            6,
            14,
            22,
            30,
            7,
            15,
            23,
            31,
        ]
    )
    expected = torch.empty_like(gated)
    for block in range(2):
        begin = block * 32
        expected[begin + src_to_dst] = gated[begin : begin + 32]

    actual = _shuffle_trtllm_epilogue_rows(value, gated_w13=True)
    assert torch.equal(actual, expected)


def test_trtllm_scale_interleave_matches_128x4_offset_definition():
    value = torch.arange(128 * 8, dtype=torch.int32).reshape(128, 8)
    expected = torch.empty_like(value).view(-1)
    for row in range(128):
        for column in range(8):
            destination = (
                (column // 4) * 512
                + (row % 32) * 16
                + ((row % 128) // 32) * 4
                + column % 4
            )
            expected[destination] = value[row, column]

    actual = _interleave_trtllm_nvfp4_scales(value)
    assert torch.equal(actual, expected.reshape(128, 8))
