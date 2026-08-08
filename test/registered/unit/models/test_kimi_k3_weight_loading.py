from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_loader.utils import DEFERRED_WEIGHT_COPY_SAFE_ATTR
from sglang.srt.models.kimi_k3 import (
    KimiK3LinearForCausalLM,
    _expert_mapping_candidates,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_expert_mapping_uses_encoded_expert_id():
    mappings = [
        ("w13", "w1", 0, "w1"),
        ("w13", "w3", 0, "w3"),
        ("w2", "w2", 0, "w2"),
        ("w13", "w1", 1, "w1"),
        ("w13", "w3", 1, "w3"),
        ("w2", "w2", 1, "w2"),
    ]
    by_expert = {
        expert_id: [mapping for mapping in mappings if mapping[2] == expert_id]
        for expert_id in range(2)
    }

    candidates = _expert_mapping_candidates(
        "model.layers.3.mlp.experts.1.w2.weight",
        mappings,
        by_expert,
    )

    assert candidates == by_expert[1]


class _BatchOwner:
    def __init__(self):
        self.immediate_calls = []
        self.batched_calls = []

    def supports_batched_weight_loading(self):
        return True

    def weight_loader(self, *args, **kwargs):
        self.immediate_calls.append((args, kwargs))

    def batched_weight_loader(self, calls, *, executor):
        self.batched_calls.extend(calls)


def _weight_loading_model(owner):
    param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    param.weight_loader = owner.weight_loader
    return SimpleNamespace(
        config=SimpleNamespace(
            linear_attn_config={},
            is_moe=True,
            num_experts=1,
            num_hidden_layers=1,
            is_linear_attn=False,
        ),
        model=SimpleNamespace(start_layer=0, end_layer=1),
        named_parameters=lambda: [("model.layers.0.mlp.experts.w13_weight", param)],
        post_load_weights=lambda **_: None,
    )


def test_load_weights_only_defers_explicitly_stable_sources():
    owner = _BatchOwner()
    model = _weight_loading_model(owner)
    stable = torch.ones(1)
    setattr(stable, DEFERRED_WEIGHT_COPY_SAFE_ATTR, True)

    KimiK3LinearForCausalLM.load_weights(
        model,
        [
            ("model.layers.0.mlp.experts.0.w1.weight", stable),
            ("model.layers.0.mlp.experts.0.w3.weight", torch.ones(1)),
        ],
    )

    assert len(owner.batched_calls) == 1
    assert owner.batched_calls[0][0][1] is stable
    assert len(owner.immediate_calls) == 1


class _FakeMoE:
    def __init__(self):
        self.merge_calls = 0
        self.gate = SimpleNamespace(
            e_score_correction_bias=torch.nn.Parameter(
                torch.zeros(1, dtype=torch.bfloat16)
            )
        )

    def _merge_front_weights(self):
        self.merge_calls += 1


class _FakeDeltaAttention:
    def __init__(self):
        self.merge_calls = 0
        self.prepare_calls = 0
        self.dt_bias = torch.zeros(1)

    def _merge_bfa_weights(self):
        self.merge_calls += 1

    def _prepare_fused_decode(self):
        self.prepare_calls += 1


def _model():
    layers = [
        SimpleNamespace(
            use_attn_residuals=False,
            mlp=_FakeMoE(),
            self_attn=_FakeDeltaAttention(),
        )
        for _ in range(2)
    ]
    return SimpleNamespace(
        config=SimpleNamespace(full_attention_layer_ids=[]),
        model=SimpleNamespace(start_layer=0, end_layer=2, layers=layers),
    )


@patch("sglang.srt.models.kimi_k3.KimiK3MoE", _FakeMoE)
@patch("sglang.srt.models.kimi_k3.KimiK3DeltaAttention", _FakeDeltaAttention)
def test_post_load_only_processes_loaded_layers():
    model = _model()

    KimiK3LinearForCausalLM.post_load_weights(
        model,
        weight_names={"model.layers.1.mlp.experts.w13_weight"},
    )

    untouched, loaded = model.model.layers
    assert untouched.mlp.merge_calls == 0
    assert untouched.self_attn.merge_calls == 0
    assert untouched.self_attn.prepare_calls == 0
    assert loaded.mlp.merge_calls == 1
    assert loaded.self_attn.merge_calls == 1
    assert loaded.self_attn.prepare_calls == 1
    assert loaded.mlp.gate.e_score_correction_bias.dtype == torch.float32


@patch("sglang.srt.models.kimi_k3.KimiK3MoE", _FakeMoE)
@patch("sglang.srt.models.kimi_k3.KimiK3DeltaAttention", _FakeDeltaAttention)
def test_post_load_hook_processes_full_model():
    model = _model()

    KimiK3LinearForCausalLM.post_load_weights(model)

    for layer in model.model.layers:
        assert layer.mlp.merge_calls == 1
        assert layer.self_attn.merge_calls == 1
        assert layer.self_attn.prepare_calls == 1
