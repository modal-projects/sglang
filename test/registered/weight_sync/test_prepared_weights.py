from __future__ import annotations

import torch

from sglang.srt.weight_sync.prepared_weights import (
    build_prepared_weight_plan,
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

    def prepared_weight_tensors(self):
        yield "derived", self.derived


def test_weight_plan_deduplicates_aliases_and_requires_explicit_derived_state():
    model = _ModelWithDerivedWeight()

    names = [name for name, _ in iter_weight_tensors(model)]
    assert "first" in names
    assert "alias" in names
    assert "buffer" in names
    assert "derived" in names
    assert "unrelated" not in names

    segments, image_nbytes = build_prepared_weight_plan(
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
