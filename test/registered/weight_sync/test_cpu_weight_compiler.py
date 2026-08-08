from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.weight_sync.cpu_weight_compiler import (
    CPUWeightCompiler,
    _staging_postprocess_device,
)
from sglang.srt.weight_sync.cpu_weight_image import (
    build_cpu_weight_image_plan,
    iter_weight_tensors,
)
from sglang.srt.weight_sync.weight_loader_isolation import build_weight_module_groups
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _QuantMethod:
    def __init__(self, device):
        self.device = device

    def weight_staging_postprocess_device(self, _module):
        return self.device


class _MutatingQuantMethod(_QuantMethod):
    def process_weights_after_loading(self, module):
        module.weight.data.add_(1)


def test_postprocess_device_is_fail_closed_and_promotes_to_cuda():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    model[0].quant_method = _QuantMethod("cpu")
    model[1].quant_method = _QuantMethod("cuda")

    assert _staging_postprocess_device(model) == "cuda"

    model[1].quant_method = _QuantMethod(None)
    with pytest.raises(NotImplementedError, match="CPU weight staging is unsupported"):
        _staging_postprocess_device(model)


def test_native_weight_loading_context_scopes_online_nvfp4_math(monkeypatch):
    variable = "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"
    monkeypatch.delenv(variable, raising=False)
    model = SimpleNamespace(
        quant_config=SimpleNamespace(is_nvfp4_online=True),
    )

    with DefaultModelLoader.weight_loading_context(model):
        assert os.environ[variable] == "1"
    assert variable not in os.environ


class _LoadableBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(8), requires_grad=False)


class _LoadableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _LoadableBlock()

    def load_weights(self, weights):
        parameters = dict(self.named_parameters())
        for name, tensor in weights:
            parameters[name].data.copy_(tensor)


class _CPUImage:
    def __init__(self, model):
        self.segments, self.image_nbytes = build_cpu_weight_image_plan(
            model,
            device_type="cpu",
        )
        self.image = torch.empty(self.image_nbytes, dtype=torch.uint8)
        self._buffer = memoryview(self.image.numpy()).cast("B")
        self._segments_by_storage = {
            _storage_key(segment.device_bytes): segment for segment in self.segments
        }
        self.segments_by_name = {
            name: self._segments_by_storage[_storage_key(tensor)]
            for name, tensor in iter_weight_tensors(model)
        }
        self.registered = True
        self.valid = True
        self.staging = False
        self.staged = False
        self.target_version = None

    def storage_image_bytes(self, tensor):
        segment = self._segments_by_storage[_storage_key(tensor)]
        begin = segment.image_offset
        return torch.frombuffer(
            self._buffer[begin : begin + segment.nbytes],
            dtype=torch.uint8,
        )

    def begin_stage(self, target_version):
        self.valid = False
        self.staging = True
        self.target_version = target_version

    def finish_stage(self, target_version):
        assert self.target_version == target_version
        self.valid = True
        self.staging = False
        self.staged = True

    def invalidate(self, _reason):
        self.valid = False
        self.staging = False
        self.staged = False


def _storage_key(tensor):
    storage = tensor.untyped_storage()
    return tensor.device.index, storage.data_ptr(), storage.nbytes()


def test_copy_shadow_to_image_supports_root_group():
    model = _LoadableModel()
    expected = torch.arange(8, dtype=torch.float32)
    model.layer.weight.data.copy_(expected)
    image = _CPUImage(model)
    compiler = CPUWeightCompiler.__new__(CPUWeightCompiler)
    compiler.image = image

    updated, runtime_bytes, cpu_copy_bytes, d2h_bytes = compiler._copy_shadow_to_image(
        "", model
    )

    segment = image.segments_by_name["layer.weight"]
    actual = image.image[
        segment.image_offset : segment.image_offset + segment.nbytes
    ].view(torch.float32)
    assert updated == {id(segment)}
    assert runtime_bytes == segment.nbytes
    assert cpu_copy_bytes == segment.nbytes
    assert d2h_bytes == 0
    torch.testing.assert_close(actual, expected)


def test_compile_loads_rank_image_without_mutating_live_weights():
    model = _LoadableModel()
    image = _CPUImage(model)
    compiler = CPUWeightCompiler.__new__(CPUWeightCompiler)
    compiler.model = model
    compiler.groups = build_weight_module_groups(
        model,
        max_group_bytes=16,
        device_type="cpu",
    )
    compiler.image = image

    expected = torch.arange(8, dtype=torch.float32)
    checkpoint = SimpleNamespace(
        weight_map={"layer.weight": "model.safetensors"},
        get_tensor=lambda _name: expected,
        run_on_host_ranks=lambda _operation, function: function(),
    )

    @contextmanager
    def tensor_group(_path, _names):
        yield checkpoint

    checkpoint.tensor_group = tensor_group
    stats = compiler.compile(checkpoint, target_version=3)

    assert stats["target_version"] == 3
    assert image.valid and image.staged
    assert torch.all(model.layer.weight == 0)
    segment = image.segments_by_name["layer.weight"]
    actual = image.image[
        segment.image_offset : segment.image_offset + segment.nbytes
    ].view(torch.float32)
    torch.testing.assert_close(actual, expected)


def test_compile_preserves_groups_absent_from_canonical_checkpoint():
    model = _LoadableModel()
    model.loaded = model.layer
    del model.layer
    model.runtime_only = _LoadableBlock()
    model.runtime_only.quant_method = _MutatingQuantMethod("cpu")
    model.loaded.weight.data.fill_(3)
    model.runtime_only.weight.data.fill_(7)
    image = _CPUImage(model)
    for name, tensor in iter_weight_tensors(model):
        segment = image.segments_by_name[name]
        target = image.image[
            segment.image_offset : segment.image_offset + segment.nbytes
        ]
        target.copy_(tensor.detach().view(torch.uint8))

    compiler = CPUWeightCompiler.__new__(CPUWeightCompiler)
    compiler.model = model
    compiler.groups = build_weight_module_groups(
        model,
        max_group_bytes=16,
        device_type="cpu",
    )
    compiler.image = image

    expected = torch.arange(8, dtype=torch.float32)
    checkpoint = SimpleNamespace(
        weight_map={"loaded.weight": "model.safetensors"},
        get_tensor=lambda _name: expected,
        run_on_host_ranks=lambda _operation, function: function(),
    )

    @contextmanager
    def tensor_group(_path, _names):
        yield checkpoint

    checkpoint.tensor_group = tensor_group
    stats = compiler.compile(checkpoint, target_version=1)

    loaded_segment = image.segments_by_name["loaded.weight"]
    loaded = image.image[
        loaded_segment.image_offset : loaded_segment.image_offset
        + loaded_segment.nbytes
    ].view(torch.float32)
    runtime_segment = image.segments_by_name["runtime_only.weight"]
    runtime_only = image.image[
        runtime_segment.image_offset : runtime_segment.image_offset
        + runtime_segment.nbytes
    ].view(torch.float32)

    torch.testing.assert_close(loaded, expected)
    torch.testing.assert_close(runtime_only, torch.full((8,), 7.0))
    assert stats["preserved_storages"] == 1
    assert stats["preserved_bytes"] == model.runtime_only.weight.nbytes


def test_compile_failure_identifies_weight_group():
    model = _LoadableModel()
    compiler = CPUWeightCompiler.__new__(CPUWeightCompiler)
    compiler.model = model
    compiler.groups = build_weight_module_groups(
        model,
        max_group_bytes=16,
        device_type="cpu",
    )
    compiler.image = _CPUImage(model)

    checkpoint = SimpleNamespace(
        weight_map={"layer.weight": "model.safetensors"},
        get_tensor=lambda _name: (_ for _ in ()).throw(ValueError("bad tensor")),
        run_on_host_ranks=lambda _operation, function: function(),
    )

    @contextmanager
    def tensor_group(_path, _names):
        yield checkpoint

    checkpoint.tensor_group = tensor_group
    with pytest.raises(
        RuntimeError,
        match=r"group 1/1 at 'layer' during load",
    ) as error:
        compiler.compile(checkpoint, target_version=1)

    assert isinstance(error.value.__cause__, ValueError)
