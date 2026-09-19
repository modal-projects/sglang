from __future__ import annotations

import pytest
import torch

from sglang.srt.weight_sync.weight_load_isolation import (
    build_weight_load_groups,
    build_weight_loader_view,
    clone_module_for_weight_loading,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Block(torch.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        storage = torch.arange(size + 4, dtype=torch.uint8)
        self.weight = torch.nn.Parameter(storage[:size], requires_grad=False)
        self.alias = torch.nn.Parameter(storage[2 : size + 2], requires_grad=False)
        self.derived = torch.arange(size, dtype=torch.int32)

    def get_additional_weight_tensors(self):
        yield "derived", self.derived


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(8), _Block(12)])


def test_groups_are_bounded_and_storage_complete():
    model = _Model()

    groups = build_weight_load_groups(
        model,
        max_group_bytes=80,
        device_type="cpu",
    )

    assert [group.path for group in groups] == ["layers.0", "layers.1"]
    assert all(group.nbytes <= 80 for group in groups)


def test_declared_indivisible_subtree_is_one_group():
    model = torch.nn.Module()
    model.block = torch.nn.Module()
    model.block.weight_load_indivisible = True
    model.block.left = torch.nn.Linear(8, 8)
    model.block.right = torch.nn.Linear(8, 8)

    groups = build_weight_load_groups(
        model,
        max_group_bytes=300,
        device_type="cpu",
    )

    assert [group.path for group in groups] == ["block"]
    assert groups[0].nbytes > 300


def test_group_budget_includes_nonpersistent_loader_state():
    model = _Model()
    before = build_weight_load_groups(
        model,
        max_group_bytes=80,
        device_type="cpu",
    )
    before_bytes = next(group.nbytes for group in before if group.path == "layers.0")
    model.layers[0].register_buffer(
        "workspace",
        torch.zeros(32, dtype=torch.uint8),
        persistent=False,
    )

    groups = build_weight_load_groups(
        model,
        max_group_bytes=80,
        device_type="cpu",
    )

    first = next(group for group in groups if group.path == "layers.0")
    assert first.nbytes == before_bytes + 32


def test_runtime_cache_without_weights_does_not_create_a_load_group():
    model = torch.nn.Module()
    model.weight = torch.nn.Linear(2, 2)
    model.cache = torch.nn.Module()
    model.cache.register_buffer(
        "workspace",
        torch.zeros(1024, dtype=torch.uint8),
        persistent=False,
    )

    groups = build_weight_load_groups(
        model,
        max_group_bytes=64,
        device_type="cpu",
    )

    assert [group.path for group in groups] == ["weight"]


def test_grouping_rejects_storage_shared_across_units():
    model = torch.nn.Module()
    model.left = torch.nn.Linear(8, 8)
    model.right = torch.nn.Linear(8, 8)
    model.right.weight = model.left.weight

    with pytest.raises(ValueError, match="spans independent load groups"):
        build_weight_load_groups(
            model,
            max_group_bytes=300,
            device_type="cpu",
        )


def test_clone_preserves_storage_aliases_and_values():
    source = _Block(16)

    copied = clone_module_for_weight_loading(source)

    assert copied.weight.data_ptr() != source.weight.data_ptr()
    assert (
        copied.weight.untyped_storage().data_ptr()
        == copied.alias.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(copied.weight, source.weight)
    torch.testing.assert_close(copied.derived, source.derived)


def test_clone_rebinds_parameter_loaders_to_copied_modules():
    class LoadableBlock(_Block):
        def __init__(self):
            super().__init__(16)
            self.weight.weight_loader = self.load_weight

        def load_weight(self, parameter, value):
            parameter.data.copy_(value)
            self.loaded = True

    source = LoadableBlock()
    copied = clone_module_for_weight_loading(source)

    assert copied.weight.weight_loader.__self__ is copied
    copied.weight.weight_loader(copied.weight, torch.full_like(copied.weight, 7))
    assert copied.loaded
    assert not hasattr(source, "loaded")
    assert torch.all(copied.weight == 7)
    assert torch.any(source.weight != 7)


def test_clone_isolates_mutable_post_load_objects():
    class Dispatcher:
        def __init__(self, child=None):
            self.quant_config = {}
            self.child = child

        def set_quant_config(self, config):
            self.quant_config = config
            if self.child is not None:
                self.child.set_quant_config(config)

    source = _Block(16)
    source.dispatcher = Dispatcher(Dispatcher())
    copied = clone_module_for_weight_loading(source)

    copied.dispatcher.set_quant_config({"scale": copied.derived})

    assert copied.dispatcher is not source.dispatcher
    assert copied.dispatcher.child is not source.dispatcher.child
    assert copied.dispatcher.quant_config["scale"] is copied.derived
    assert source.dispatcher.quant_config == {}
    assert source.dispatcher.child.quant_config == {}


def test_clone_rejects_cyclic_immutable_loader_state():
    source = _Block(16)
    cycle = []
    source.loader_state = (cycle,)
    cycle.append(source.loader_state)

    with pytest.raises(ValueError, match="cyclic immutable loader state"):
        clone_module_for_weight_loading(source)


def test_loader_view_isolates_only_the_target_subtree():
    model = _Model()

    view, shadow = build_weight_loader_view(model, "layers.0")

    assert view is not model
    assert view.layers is not model.layers
    assert shadow is view.layers[0]
    assert shadow is not model.layers[0]
    assert view.layers[1] is model.layers[1]
    shadow.weight.data.fill_(42)
    assert torch.all(model.layers[0].weight != 42)


def test_replacement_storage_offset_uses_bytes():
    source = torch.nn.Linear(4, 4)
    views = []

    def storage_factory(_tensor, source_bytes):
        backing = torch.empty(source_bytes.numel() + 64, dtype=torch.uint8)
        view = backing[64:]
        view.copy_(source_bytes)
        views.append(view)
        return view

    copied = clone_module_for_weight_loading(
        source,
        storage_factory=storage_factory,
    )

    torch.testing.assert_close(copied.weight, source.weight)
    torch.testing.assert_close(copied.bias, source.bias)
    assert all(view.storage_offset() == 64 for view in views)
