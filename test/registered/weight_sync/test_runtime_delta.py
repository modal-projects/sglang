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
            self.quant_method = object()

    @staticmethod
    def _load(parameter, loaded_weight):
        # The loader receives the tagged source and performs its sharding
        # internally, matching SGLang's tensor-parallel loaders.
        parameter.copy_(loaded_weight[2:10])

    def load_weights(self, weights):
        for name, tensor in weights:
            assert name == "weight"
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
