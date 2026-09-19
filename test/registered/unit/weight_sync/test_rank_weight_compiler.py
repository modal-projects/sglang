from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.weight_sync.rank_weight_compiler import (
    RankWeightCompiler,
    _checkpoint_groups,
)
from sglang.srt.weight_sync.rank_weight_image import (
    build_rank_weight_image_plan,
    iter_weight_tensors,
)
from sglang.srt.weight_sync.weight_load_isolation import (
    WeightLoadGroup,
    build_weight_load_groups,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _storage_key(tensor):
    storage = tensor.untyped_storage()
    return tensor.device.index, storage.data_ptr(), storage.nbytes()


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


class _RootLoadableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(8), requires_grad=False)

    def load_weights(self, weights):
        for name, tensor in weights:
            assert name == "weight"
            self.weight.data.copy_(tensor)


class _CPUImage:
    def __init__(self, model):
        self.segments, self.image_nbytes = build_rank_weight_image_plan(
            model,
            device_type="cpu",
        )
        self.weight_nbytes = sum(segment.nbytes for segment in self.segments)
        self.device = torch.device("cpu")
        self.image = torch.empty(self.image_nbytes, dtype=torch.uint8)
        self._buffer = memoryview(self.image.numpy()).cast("B")
        self._segments_by_storage = {
            _storage_key(segment.device_bytes): segment for segment in self.segments
        }
        self.segments_by_name = {
            name: self._segments_by_storage[_storage_key(tensor)]
            for name, tensor in iter_weight_tensors(model)
            if tensor.untyped_storage().nbytes() > 0
        }
        self.registered = False
        self.valid = False
        self.staging = False
        self.staged = False
        self.target_version = None
        self.invalid_reason = None

    def storage_image_bytes(self, tensor):
        segment = self._segments_by_storage[_storage_key(tensor)]
        begin = segment.image_offset
        return torch.frombuffer(
            self._buffer[begin : begin + segment.nbytes],
            dtype=torch.uint8,
        )

    def register_host_memory(self):
        self.registered = True
        return {}

    def capture_active_weights(self):
        for segment in self.segments:
            target = self.image[
                segment.image_offset : segment.image_offset + segment.nbytes
            ]
            target.copy_(segment.device_bytes)
        self.valid = True
        self.staged = False
        self.staging = False
        self.target_version = None
        return {}

    def begin_stage(self, target_version):
        self.valid = False
        self.staging = True
        self.staged = False
        self.target_version = target_version

    def finish_stage(self, target_version, commit_segments):
        assert self.target_version == target_version
        self.commit_segments = tuple(commit_segments)
        self.valid = True
        self.staging = False
        self.staged = True

    def invalidate(self, reason):
        self.valid = False
        self.staging = False
        self.staged = False
        self.invalid_reason = reason

    def copy_device_segments_to_image(self, _segments):
        raise AssertionError("CPU compiler test must not request a device copy")

    def close(self):
        pass


def _compiler(model):
    compiler = RankWeightCompiler.__new__(RankWeightCompiler)
    compiler.model = model
    compiler.groups = build_weight_load_groups(
        model,
        max_group_bytes=16,
        device_type="cpu",
    )
    compiler.image = _CPUImage(model)
    compiler._stream = None
    compiler._prepared_groups = None
    compiler._checkpoint_names = None
    compiler._ignored_checkpoint_names = frozenset()
    return compiler


def _use_plain_loader(monkeypatch):
    monkeypatch.setattr(
        DefaultModelLoader,
        "restore_weights_before_loading",
        lambda _model, _device: None,
    )
    monkeypatch.setattr(
        DefaultModelLoader,
        "load_weights_only",
        lambda model, weights, _device: model.load_weights(weights),
    )
    monkeypatch.setattr(
        DefaultModelLoader,
        "postprocess_weights",
        lambda _model, _device: None,
    )


def test_checkpoint_groups_use_checkpoint_mapper_and_explicit_exclusions():
    model = _LoadableModel()
    model.checkpoint_name_mapper = WeightsMapper(
        orig_to_new_prefix={
            "checkpoint.": "layer.",
            "draft.": None,
        }
    )

    groups, ignored = _checkpoint_groups(
        model,
        ["checkpoint.weight", "draft.weight"],
        [WeightLoadGroup("layer", 32)],
    )

    assert groups == {"layer": ["checkpoint.weight"]}
    assert ignored == {"draft.weight"}

    compiler = _compiler(model)
    compiler.prepare_loader_views(
        {
            "checkpoint.weight": "model.safetensors",
            "draft.weight": "model.safetensors",
        }
    )
    with pytest.raises(ValueError, match="draft.weight"):
        compiler.validate_delta_names(["checkpoint.weight", "draft.weight"])


def test_checkpoint_groups_fall_back_to_native_mapper():
    model = _LoadableModel()
    model.hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={"checkpoint.": "layer."}
    )

    groups, ignored = _checkpoint_groups(
        model,
        ["checkpoint.weight"],
        [WeightLoadGroup("layer", 32)],
    )

    assert groups == {"layer": ["checkpoint.weight"]}
    assert ignored == set()


def test_checkpoint_groups_reject_unmapped_names():
    model = _LoadableModel()
    with pytest.raises(ValueError, match="do not map"):
        _checkpoint_groups(
            model,
            ["other.weight"],
            [WeightLoadGroup("layer", 32)],
        )


def test_compile_reuses_loader_views_without_mutating_live_weights(monkeypatch):
    _use_plain_loader(monkeypatch)
    model = _LoadableModel()
    compiler = _compiler(model)
    value = torch.arange(8, dtype=torch.float32)
    checkpoint = SimpleNamespace(
        version=1,
        weight_map={"layer.weight": "model.safetensors"},
        get_tensor=lambda _name: value,
    )

    first = compiler.compile(checkpoint, target_version=1)

    assert compiler.image.valid and compiler.image.staged
    assert torch.all(model.layer.weight == 0)
    segment = compiler.image.segments_by_name["layer.weight"]
    staged = compiler.image.image[
        segment.image_offset : segment.image_offset + segment.nbytes
    ].view(torch.float32)
    torch.testing.assert_close(staged, value)
    assert not first["loader_views"]["reused"]

    compiler.image.staged = False
    value = value + 10
    checkpoint.version = 2
    second = compiler.compile(checkpoint, target_version=2)

    torch.testing.assert_close(staged, value)
    assert second["loader_views"]["reused"]


def test_prepare_loader_views_does_not_mutate_the_rank_image():
    model = _LoadableModel()
    compiler = _compiler(model)
    compiler.image.image.copy_(
        torch.arange(compiler.image.image_nbytes, dtype=torch.uint8)
    )
    expected = compiler.image.image.clone()

    compiler.prepare_loader_views({"layer.weight": "model.safetensors"})

    torch.testing.assert_close(compiler.image.image, expected)


def test_compile_supports_a_root_owned_weight(monkeypatch):
    _use_plain_loader(monkeypatch)
    model = _RootLoadableModel()
    compiler = _compiler(model)
    value = torch.arange(8, dtype=torch.float32)
    checkpoint = SimpleNamespace(
        version=1,
        weight_map={"weight": "model.safetensors"},
        get_tensor=lambda _name: value,
    )

    compiler.compile(checkpoint, target_version=1)

    segment = compiler.image.segments_by_name["weight"]
    staged = compiler.image.image[
        segment.image_offset : segment.image_offset + segment.nbytes
    ].view(torch.float32)
    torch.testing.assert_close(staged, value)


def test_compile_preserves_runtime_storage_absent_from_checkpoint(monkeypatch):
    _use_plain_loader(monkeypatch)
    model = _LoadableModel()
    model.auxiliary = _LoadableBlock()
    model.auxiliary.weight.data.fill_(17)
    compiler = _compiler(model)
    checkpoint = SimpleNamespace(
        version=1,
        weight_map={"layer.weight": "model.safetensors"},
        get_tensor=lambda _name: torch.arange(8, dtype=torch.float32),
    )

    stats = compiler.compile(checkpoint, target_version=1)

    segment = compiler.image.segments_by_name["auxiliary.weight"]
    preserved = compiler.image.image[
        segment.image_offset : segment.image_offset + segment.nbytes
    ].view(torch.float32)
    torch.testing.assert_close(preserved, model.auxiliary.weight)
    assert stats["preserved_storages"] == 1
    assert stats["preserved_bytes"] == model.auxiliary.weight.nbytes
    assert stats["commit_bytes"] == model.layer.weight.nbytes
    assert segment not in compiler.image.commit_segments


def test_compile_failure_invalidates_the_image(monkeypatch):
    _use_plain_loader(monkeypatch)
    model = _LoadableModel()
    compiler = _compiler(model)
    checkpoint = SimpleNamespace(
        version=1,
        weight_map={"layer.weight": "model.safetensors"},
        get_tensor=lambda _name: (_ for _ in ()).throw(ValueError("bad tensor")),
    )

    with pytest.raises(RuntimeError, match="rank weight compilation failed"):
        compiler.compile(checkpoint, target_version=1)

    assert not compiler.image.valid
    assert not compiler.image.staged
    assert "bad tensor" in compiler.image.invalid_reason


def test_image_copy_rejects_split_runtime_aliases():
    model = torch.nn.Module()
    storage = torch.arange(8, dtype=torch.float32)
    model.first = torch.nn.Parameter(storage[:4], requires_grad=False)
    model.second = torch.nn.Parameter(storage[4:], requires_grad=False)
    compiler = RankWeightCompiler.__new__(RankWeightCompiler)
    compiler.image = _CPUImage(model)
    compiler._stream = None

    shadow = torch.nn.Module()
    first_storage = storage.clone()
    second_storage = storage.clone()
    shadow.first = torch.nn.Parameter(first_storage[:4], requires_grad=False)
    shadow.second = torch.nn.Parameter(second_storage[4:], requires_grad=False)

    with pytest.raises(RuntimeError, match="split aliased runtime storage"):
        compiler._copy_shadow_to_image("", shadow)


def test_compile_rejects_wrong_or_overlapping_targets(monkeypatch):
    _use_plain_loader(monkeypatch)
    model = _LoadableModel()
    compiler = _compiler(model)
    checkpoint = SimpleNamespace(
        version=2,
        weight_map={"layer.weight": "model.safetensors"},
        get_tensor=lambda _name: torch.arange(8, dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="does not match"):
        compiler.compile(checkpoint, target_version=1)

    compiler.image.valid = True
    compiler.image.staged = True
    compiler.image.target_version = 1
    with pytest.raises(RuntimeError, match="already staged"):
        compiler.compile(checkpoint, target_version=2)
