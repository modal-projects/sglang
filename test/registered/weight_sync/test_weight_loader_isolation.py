from __future__ import annotations

import pytest
import torch

from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.weight_sync.weight_loader_isolation import (
    batch_weight_module_groups,
    build_weight_loader_proxy,
    build_weight_module_groups,
    clone_weight_module,
    map_checkpoint_names_to_groups,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _DerivedBlock(torch.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        storage = torch.arange(size + 4, dtype=torch.uint8)
        self.weight = torch.nn.Parameter(storage[:size], requires_grad=False)
        self.alias = torch.nn.Parameter(storage[2 : size + 2], requires_grad=False)
        self.derived = torch.arange(size, dtype=torch.int32)

    def get_additional_weight_tensors(self):
        yield "derived", self.derived


class _GroupedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_DerivedBlock(8), _DerivedBlock(12)])


class _WrappedModel(torch.nn.Module):
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={"hf_layers.": "language_model.layers."},
    )

    def __init__(self):
        super().__init__()
        self.vision_tower = torch.nn.Linear(2, 2)
        self.mm_projector = torch.nn.Linear(2, 2)
        self.language_model = _GroupedModel()


def test_groups_are_bounded_storage_complete_and_batchable():
    model = _GroupedModel()
    groups = build_weight_module_groups(
        model,
        max_group_bytes=80,
        device_type="cpu",
    )

    assert [group.path for group in groups] == ["layers.0", "layers.1"]
    assert all(group.nbytes <= 80 for group in groups)
    assert [
        len(batch) for batch in batch_weight_module_groups(groups, max_batch_bytes=80)
    ] == [1, 1]


def test_groups_support_a_model_with_root_weights():
    model = torch.nn.Linear(4, 4)

    groups = build_weight_module_groups(
        model,
        max_group_bytes=1024,
        device_type="cpu",
    )

    assert [group.path for group in groups] == [""]
    proxy, shadow = build_weight_loader_proxy(model, "")
    assert proxy is shadow
    assert shadow.weight.data_ptr() != model.weight.data_ptr()


def test_groups_exclude_non_persistent_runtime_buffers():
    model = _GroupedModel()
    model.layers[0].register_buffer(
        "runtime_cache",
        torch.zeros(1024),
        persistent=False,
    )

    groups = build_weight_module_groups(
        model,
        max_group_bytes=80,
        device_type="cpu",
    )

    assert [group.path for group in groups] == ["layers.0", "layers.1"]


def test_groups_reject_storage_shared_across_independent_subtrees():
    model = torch.nn.Module()
    model.left = torch.nn.Linear(8, 8)
    model.right = torch.nn.Linear(8, 8)
    model.right.weight = model.left.weight

    with pytest.raises(ValueError, match="spans independent compilation groups"):
        build_weight_module_groups(
            model,
            max_group_bytes=300,
            device_type="cpu",
        )


def test_clone_preserves_tensor_subclasses_aliases_and_values():
    source = _DerivedBlock(16)
    cloned = clone_weight_module(source)

    assert type(cloned.weight) is type(source.weight)
    assert cloned.weight.data_ptr() != source.weight.data_ptr()
    assert (
        cloned.weight.untyped_storage().data_ptr()
        == cloned.alias.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(cloned.weight, source.weight)
    torch.testing.assert_close(cloned.derived, source.derived)


def test_clone_preserves_shared_module_identity():
    source = torch.nn.Module()
    shared = torch.nn.Linear(2, 2)
    source.left = shared
    source.right = shared

    cloned = clone_weight_module(source)

    assert cloned.left is cloned.right
    assert cloned.left is not shared


def test_clone_converts_byte_storage_offsets_to_tensor_elements():
    source = torch.nn.Linear(4, 4)
    storage_views = []

    def storage_factory(_tensor, source_bytes):
        backing = torch.empty(source_bytes.numel() + 64, dtype=torch.uint8)
        view = backing[64:]
        view.copy_(source_bytes)
        storage_views.append(view)
        return view

    cloned = clone_weight_module(source, storage_factory=storage_factory)

    torch.testing.assert_close(cloned.weight, source.weight)
    torch.testing.assert_close(cloned.bias, source.bias)
    assert all(view.storage_offset() == 64 for view in storage_views)


def test_clone_isolates_loader_objects_and_rebinds_callbacks():
    class Loader:
        def callback(self):
            self.called = True

        def __init__(self):
            self.callback_ref = self.callback

    source = _DerivedBlock(16)
    source.quant_method = Loader()
    source.scheme = source.quant_method
    cloned = clone_weight_module(source)

    assert cloned.quant_method is not source.quant_method
    assert cloned.scheme is cloned.quant_method
    assert cloned.quant_method.callback_ref.__self__ is cloned.quant_method
    cloned.quant_method.callback_ref()
    assert cloned.quant_method.called
    assert not hasattr(source.quant_method, "called")


def test_clone_rebinds_parameter_loaders_to_shadow_modules():
    class LoaderBlock(_DerivedBlock):
        def __init__(self):
            super().__init__(16)
            self.weight.weight_loader = self.load_weight

        def load_weight(self, parameter, value):
            parameter.data.copy_(value)
            self.loader_called = True

    source = LoaderBlock()
    cloned = clone_weight_module(source)

    assert cloned.weight.weight_loader.__self__ is cloned
    cloned.weight.weight_loader(cloned.weight, torch.full_like(cloned.weight, 7))
    assert cloned.loader_called
    assert not hasattr(source, "loader_called")
    assert torch.all(cloned.weight == 7)
    assert torch.any(source.weight != 7)


def test_clone_isolates_nested_tensor_state_and_staging_protocols():
    class MutableHelper:
        def __init__(self):
            self.values = []

        def clone_for_weight_staging(self):
            cloned = MutableHelper()
            cloned.values = self.values.copy()
            return cloned

    source = _DerivedBlock(16)
    source.weight.runtime_buffer = source.derived
    source.loader_state = {"levels": [[[{"tensor": source.derived}]]]}
    source.loader_state_alias = source.loader_state
    source.helper = MutableHelper()

    cloned = clone_weight_module(source)
    cloned_tensor = cloned.loader_state["levels"][0][0][0]["tensor"]
    cloned.helper.values.append("updated")

    assert cloned.loader_state is cloned.loader_state_alias
    assert cloned_tensor is cloned.derived
    assert cloned.weight.runtime_buffer is cloned.derived
    assert cloned.helper is not source.helper
    assert source.helper.values == []


def test_clone_rejects_cyclic_immutable_loader_state():
    source = _DerivedBlock(16)
    cycle = []
    source.loader_state = (cycle,)
    cycle.append(source.loader_state)

    with pytest.raises(ValueError, match="cyclic immutable loader state"):
        clone_weight_module(source)


def test_clone_isolates_dispatcher_state():
    class Dispatcher(BaseDispatcher):
        def dispatch(self, hidden_states, topk_output):
            raise NotImplementedError

        def combine(self, combine_input):
            raise NotImplementedError

    source = _DerivedBlock(16)
    source.dispatcher = Dispatcher()
    source.dispatcher.child = Dispatcher()

    cloned = clone_weight_module(source)
    cloned.dispatcher.set_quant_config({"weight_dtype": "fp4"})
    cloned.dispatcher.child.set_quant_config({"weight_dtype": "fp8"})

    assert cloned.dispatcher is not source.dispatcher
    assert cloned.dispatcher.child is not source.dispatcher.child
    assert source.dispatcher.quant_config == {}
    assert source.dispatcher.child.quant_config == {}


def test_proxy_ancestors_do_not_share_module_registries():
    model = _GroupedModel()
    proxy, _ = build_weight_loader_proxy(model, "layers.0")

    assert proxy._modules is not model._modules
    assert proxy.layers._modules is not model.layers._modules
    proxy.layers.register_buffer("temporary", torch.ones(1))
    assert "temporary" not in model.layers._buffers


def test_checkpoint_names_map_to_runtime_groups():
    model = _WrappedModel()
    groups = build_weight_module_groups(
        model,
        max_group_bytes=80,
        device_type="cpu",
    )
    mapping = map_checkpoint_names_to_groups(
        model,
        [
            "mm_projector.weight",
            "vision_tower.weight",
            "hf_layers.0.weight",
        ],
        groups,
    )

    assert mapping == {
        "mm_projector.weight": "mm_projector",
        "vision_tower.weight": "vision_tower",
        "hf_layers.0.weight": "language_model.layers.0",
    }


def test_unknown_checkpoint_name_has_no_group_without_a_wrapper_prefix():
    model = _GroupedModel()
    groups = build_weight_module_groups(
        model,
        max_group_bytes=80,
        device_type="cpu",
    )

    assert map_checkpoint_names_to_groups(model, ["unused.weight"], groups) == {
        "unused.weight": None
    }
