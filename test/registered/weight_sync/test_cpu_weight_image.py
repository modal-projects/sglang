from __future__ import annotations

import pytest
import torch

from sglang.srt.weight_sync.cpu_weight_image import (
    CPUWeightImage,
    CPUWeightSegment,
    build_cpu_weight_image_plan,
    iter_weight_tensors,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ModelWithDerivedWeight(torch.nn.Module):
    def __init__(self):
        super().__init__()
        storage = torch.arange(32, dtype=torch.uint8)
        self.first = torch.nn.Parameter(storage[:16], requires_grad=False)
        self.alias = torch.nn.Parameter(storage[8:24], requires_grad=False)
        self.register_buffer("buffer", torch.ones(7))
        self.derived = torch.arange(9, dtype=torch.int32)
        self.unrelated = torch.arange(11, dtype=torch.int16)

    def get_additional_weight_tensors(self):
        yield "derived", self.derived


def test_weight_plan_deduplicates_aliases_and_requires_explicit_derived_state():
    model = _ModelWithDerivedWeight()

    names = [name for name, _ in iter_weight_tensors(model)]
    assert "first" in names
    assert "alias" in names
    assert "buffer" in names
    assert "derived" in names
    assert "unrelated" not in names

    segments, image_nbytes = build_cpu_weight_image_plan(
        model,
        device_type="cpu",
    )
    storage_sizes = sorted(segment.nbytes for segment in segments)
    assert storage_sizes == sorted(
        [
            model.first.untyped_storage().nbytes(),
            model.buffer.untyped_storage().nbytes(),
            model.derived.untyped_storage().nbytes(),
        ]
    )
    assert image_nbytes >= sum(storage_sizes)
    assert image_nbytes % 4096 == 0


def test_weight_plan_excludes_non_persistent_runtime_buffers():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2))
    model.register_buffer("checkpoint_buffer", torch.ones(2))
    model.register_buffer("runtime_cache", torch.ones(2), persistent=False)

    assert [name for name, _ in iter_weight_tensors(model)] == [
        "weight",
        "checkpoint_buffer",
    ]


def test_additional_weight_contract_rejects_invalid_entries():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2))
    model.get_additional_weight_tensors = lambda: [("derived", object())]

    with pytest.raises(TypeError, match="must yield"):
        list(iter_weight_tensors(model))


def test_empty_storage_has_an_empty_shadow_outside_the_image():
    tensor = torch.empty(0)
    storage = tensor.untyped_storage()
    key = (tensor.device.index, storage.data_ptr(), storage.nbytes())
    image = CPUWeightImage.__new__(CPUWeightImage)
    image._segments_by_device_storage = {
        key: CPUWeightSegment(
            name="empty",
            image_offset=0,
            nbytes=0,
            device_bytes=torch.empty(0, dtype=torch.uint8),
        )
    }
    image._image_buffer = memoryview(bytearray())

    shadow = image.storage_image_bytes(tensor)

    assert shadow.dtype == torch.uint8
    assert shadow.numel() == 0


def test_staging_state_fails_closed_until_a_complete_image_is_ready():
    image = CPUWeightImage.__new__(CPUWeightImage)
    image.target_version = None
    image.staged = False
    image.staging = False
    image.valid = False
    image.invalid_reason = "image has not been initialized"
    image.registered = False

    image.begin_stage(7)
    assert image.target_version == 7
    assert image.staging
    assert not image.valid

    image.finish_stage(7)
    assert image.valid
    assert image.staged
    assert not image.staging

    with pytest.raises(RuntimeError, match="not registered"):
        image.validate_commit(7)

    image.registered = True
    image.validate_commit(7)
    image.accept_staged_baseline()
    assert image.target_version is None
    assert not image.staged

    image.invalidate("incomplete update")
    assert not image.valid
    assert image.invalid_reason == "incomplete update"
