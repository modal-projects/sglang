from __future__ import annotations

from enum import Enum, auto

import pytest
import torch

from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.weight_sync.cpu_weight_cache import (
    _build_weight_loader_proxy,
    _build_weight_module_groups,
    _clone_weight_module,
    _map_checkpoint_names_to_groups,
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


def test_groups_are_bounded_and_storage_complete():
    model = _GroupedModel()
    groups = _build_weight_module_groups(
        model,
        max_group_bytes=80,
        device_type="cpu",
    )
    assert [group.path for group in groups] == ["layers.0", "layers.1"]
    assert all(group.nbytes <= 80 for group in groups)


def test_groups_exclude_non_persistent_runtime_buffers():
    model = _GroupedModel()
    model.layers[0].register_buffer(
        "runtime_cache",
        torch.zeros(1024),
        persistent=False,
    )

    groups = _build_weight_module_groups(
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
        _build_weight_module_groups(
            model,
            max_group_bytes=512,
            device_type="cpu",
        )


def test_clone_preserves_tensor_subclasses_aliases_and_source_values():
    source = _DerivedBlock(16)
    cloned = _clone_weight_module(source)

    assert type(cloned.weight) is type(source.weight)
    assert cloned.weight.data_ptr() != source.weight.data_ptr()
    assert (
        cloned.weight.untyped_storage().data_ptr()
        == cloned.alias.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(cloned.weight, source.weight)
    torch.testing.assert_close(cloned.derived, source.derived)


def test_clone_converts_byte_storage_offsets_to_tensor_elements():
    source = torch.nn.Linear(4, 4)
    storage_views = []

    def storage_factory(_tensor, source_bytes):
        backing = torch.empty(source_bytes.numel() + 64, dtype=torch.uint8)
        view = backing[64:]
        view.copy_(source_bytes)
        storage_views.append(view)
        return view

    cloned = _clone_weight_module(source, storage_factory=storage_factory)

    torch.testing.assert_close(cloned.weight, source.weight)
    torch.testing.assert_close(cloned.bias, source.bias)
    assert all(view.storage_offset() == 64 for view in storage_views)


def test_clone_isolates_loader_owned_objects():
    class Loader:
        def callback(self):
            self.called = True

        def __init__(self):
            self.callback_ref = self.callback

    source = _DerivedBlock(16)
    source.quant_method = Loader()
    source.scheme = source.quant_method
    cloned = _clone_weight_module(source)

    assert cloned.quant_method is not source.quant_method
    assert cloned.scheme is cloned.quant_method
    assert cloned.quant_method.callback_ref.__self__ is cloned.quant_method
    cloned.quant_method.callback_ref()
    assert cloned.quant_method.called
    assert not hasattr(source.quant_method, "called")


def test_clone_rebinds_parameter_weight_loaders_to_shadow_modules():
    class LoaderBlock(_DerivedBlock):
        def __init__(self):
            super().__init__(16)
            self.weight.weight_loader = self.load_weight

        def load_weight(self, parameter, value):
            parameter.data.copy_(value)
            self.loader_called = True

    source = LoaderBlock()
    cloned = _clone_weight_module(source)

    assert cloned.weight.weight_loader.__self__ is cloned
    cloned.weight.weight_loader(cloned.weight, torch.full_like(cloned.weight, 7))
    assert cloned.loader_called
    assert not hasattr(source, "loader_called")
    assert torch.all(cloned.weight == 7)
    assert torch.any(source.weight != 7)


def test_clone_isolates_tensor_attributes_and_preserves_their_aliases():
    source = _DerivedBlock(16)
    source.weight.runtime_buffer = source.derived

    cloned = _clone_weight_module(source)

    assert cloned.weight.runtime_buffer is cloned.derived
    assert cloned.weight.runtime_buffer is not source.weight.runtime_buffer
    cloned.weight.runtime_buffer.add_(1)
    torch.testing.assert_close(cloned.derived, source.derived + 1)
    torch.testing.assert_close(source.weight.runtime_buffer, source.derived)


def test_clone_uses_weight_update_protocol():
    class MutableHelper:
        def __init__(self):
            self.value = []

        def clone_for_weight_update(self):
            cloned = MutableHelper()
            cloned.value = self.value.copy()
            return cloned

    source = _DerivedBlock(16)
    source.helper = MutableHelper()
    cloned = _clone_weight_module(source)

    cloned.helper.value.append("updated")
    assert cloned.helper is not source.helper
    assert source.helper.value == []


def test_clone_isolates_deeply_nested_loader_tensor_state():
    source = _DerivedBlock(16)
    state_tensor = torch.arange(4)
    source.loader_state = {"levels": [[[{"tensor": state_tensor}]]]}
    source.loader_state_alias = source.loader_state

    cloned = _clone_weight_module(source)
    cloned_tensor = cloned.loader_state["levels"][0][0][0]["tensor"]

    assert cloned.loader_state is cloned.loader_state_alias
    assert cloned_tensor.data_ptr() != state_tensor.data_ptr()
    torch.testing.assert_close(cloned_tensor, state_tensor)


def test_clone_rejects_cyclic_immutable_loader_state():
    source = _DerivedBlock(16)
    cycle = []
    source.loader_state = (cycle,)
    cycle.append(source.loader_state)

    with pytest.raises(ValueError, match="cyclic immutable loader state"):
        _clone_weight_module(source)


def test_clone_isolates_nested_dispatcher_state():
    class Dispatcher(BaseDispatcher):
        def dispatch(self, hidden_states, topk_output):
            raise NotImplementedError

        def combine(self, combine_input):
            raise NotImplementedError

    source = _DerivedBlock(16)
    source.dispatcher = Dispatcher()
    source.dispatcher.child = Dispatcher()
    source.dispatcher.children = [Dispatcher()]

    cloned = _clone_weight_module(source)
    cloned.dispatcher.set_quant_config({"weight_dtype": "fp4"})
    cloned.dispatcher.child.set_quant_config({"weight_dtype": "fp8"})
    cloned.dispatcher.children[0].set_quant_config({"weight_dtype": "int8"})

    assert cloned.dispatcher is not source.dispatcher
    assert cloned.dispatcher.child is not source.dispatcher.child
    assert cloned.dispatcher.children[0] is not source.dispatcher.children[0]
    assert source.dispatcher.quant_config == {}
    assert source.dispatcher.child.quant_config == {}
    assert source.dispatcher.children[0].quant_config == {}


def test_dispatcher_clone_drops_serving_hooks_and_live_method_bindings():
    class Stage(Enum):
        INITIAL = auto()
        DISPATCHING = auto()

    class Dispatcher(BaseDispatcher):
        def dispatch(self, hidden_states, topk_output):
            return hidden_states

        def combine(self, combine_input):
            return combine_input

    source = Dispatcher()
    source._deepep_dispatch_hooks = object()
    source._stage = Stage.DISPATCHING
    source._dispatch_intermediate_state = torch.ones(4)
    source._combine_intermediate_state = torch.ones(4)
    hook_calls = []
    source.register_post_dispatch_hook(
        lambda _dispatcher, output: hook_calls.append(output)
    )
    source.register_post_combine_hook(
        lambda _dispatcher, output: hook_calls.append(output)
    )
    source.set_overlap_args(object(), {"serving": True})

    cloned = source.clone_for_weight_update()

    assert cloned.dispatch("dispatch", None) == "dispatch"
    assert cloned.combine("combine") == "combine"
    assert hook_calls == []
    assert cloned.dispatch.__self__ is cloned
    assert cloned.combine.__self__ is cloned
    assert source._post_dispatch_hooks is not None
    assert source._post_combine_hooks is not None
    assert cloned._post_dispatch_hooks is None
    assert cloned._post_combine_hooks is None
    assert cloned._deepep_dispatch_hooks is None
    assert cloned._stage.name == "INITIAL"
    assert not hasattr(cloned, "_dispatch_intermediate_state")
    assert not hasattr(cloned, "_combine_intermediate_state")
    assert cloned.overlap_args is None
    assert cloned.meta_overlap_args is None


def test_dispatcher_clone_tolerates_subclasses_without_base_initialization():
    class Dispatcher(BaseDispatcher):
        def __init__(self):
            self.child = None

        def dispatch(self, hidden_states, topk_output):
            raise NotImplementedError

        def combine(self, combine_input):
            raise NotImplementedError

    cloned = Dispatcher().clone_for_weight_update()

    assert cloned.quant_config == {}
    assert cloned._pre_dispatch_hooks is None


def test_dispatcher_clone_preserves_unconfigured_quantization_state():
    class Dispatcher(BaseDispatcher):
        def __init__(self):
            super().__init__()
            self.quant_config = None

        def dispatch(self, hidden_states, topk_output):
            raise NotImplementedError

        def combine(self, combine_input):
            raise NotImplementedError

    cloned = Dispatcher().clone_for_weight_update()

    assert cloned.quant_config is None


def test_dispatcher_clone_isolates_composite_quantization_state():
    class QuantizationState:
        def __init__(self):
            self.quant_config = {}

        def clone_for_weight_update(self):
            cloned = QuantizationState()
            cloned.quant_config = self.quant_config.copy()
            return cloned

        def set_quant_config(self, quant_config):
            self.quant_config = quant_config

    class Dispatcher(BaseDispatcher):
        def __init__(self):
            super().__init__()
            self.inner = QuantizationState()

        def dispatch(self, hidden_states, topk_output):
            raise NotImplementedError

        def combine(self, combine_input):
            raise NotImplementedError

        def set_quant_config(self, quant_config):
            super().set_quant_config(quant_config)
            self.inner.set_quant_config(quant_config)

    source = Dispatcher()
    cloned = source.clone_for_weight_update()
    cloned.set_quant_config({"weight_dtype": "fp4"})

    assert cloned.inner is not source.inner
    assert source.quant_config == {}
    assert source.inner.quant_config == {}


def test_proxy_ancestors_do_not_share_module_registries():
    model = _GroupedModel()
    proxy, _ = _build_weight_loader_proxy(model, "layers.0")

    assert proxy._modules is not model._modules
    assert proxy.layers._modules is not model.layers._modules
    proxy.layers.register_buffer("temporary", torch.ones(1))
    assert "temporary" not in model.layers._buffers


def test_checkpoint_names_map_to_longest_runtime_group():
    model = _GroupedModel()
    groups = _build_weight_module_groups(
        model,
        max_group_bytes=80,
        device_type="cpu",
    )
    mapping = _map_checkpoint_names_to_groups(
        model,
        [
            "layers.0.weight",
            "layers.1.weight",
            "unused.weight",
        ],
        groups,
    )
    assert mapping == {
        "layers.0.weight": "layers.0",
        "layers.1.weight": "layers.1",
        "unused.weight": None,
    }


def test_checkpoint_mapping_prefers_authoritative_direct_runtime_root():
    model = _WrappedModel()
    groups = _build_weight_module_groups(
        model,
        max_group_bytes=80,
        device_type="cpu",
    )
    mapping = _map_checkpoint_names_to_groups(
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
