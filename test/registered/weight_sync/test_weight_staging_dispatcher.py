from __future__ import annotations

from enum import Enum, auto

import torch

from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Dispatcher(BaseDispatcher):
    def dispatch(self, hidden_states, topk_output):
        return hidden_states

    def combine(self, combine_input):
        return combine_input


def test_dispatcher_clone_drops_serving_hooks_and_live_method_bindings():
    class Stage(Enum):
        INITIAL = auto()
        DISPATCHING = auto()

    source = _Dispatcher()
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

    cloned = source.clone_for_weight_staging()

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
    assert cloned._stage is Stage.INITIAL
    assert not hasattr(cloned, "_dispatch_intermediate_state")
    assert not hasattr(cloned, "_combine_intermediate_state")
    assert cloned.overlap_args is None
    assert cloned.meta_overlap_args is None


def test_dispatcher_clone_tolerates_subclasses_without_base_initialization():
    class Dispatcher(_Dispatcher):
        def __init__(self):
            self.child = None

    cloned = Dispatcher().clone_for_weight_staging()

    assert cloned.quant_config == {}
    assert cloned._pre_dispatch_hooks is None


def test_dispatcher_clone_preserves_unconfigured_quantization_state():
    source = _Dispatcher()
    source.quant_config = None

    cloned = source.clone_for_weight_staging()

    assert cloned.quant_config is None


def test_dispatcher_clone_isolates_nested_staging_state():
    class QuantizationState:
        def __init__(self):
            self.quant_config = {}

        def clone_for_weight_staging(self):
            cloned = QuantizationState()
            cloned.quant_config = self.quant_config.copy()
            return cloned

    source = _Dispatcher()
    source.inner = QuantizationState()
    source.children = [QuantizationState()]

    cloned = source.clone_for_weight_staging()
    cloned.inner.quant_config["weight_dtype"] = "fp4"
    cloned.children[0].quant_config["weight_dtype"] = "fp8"

    assert source.inner.quant_config == {}
    assert source.children[0].quant_config == {}
