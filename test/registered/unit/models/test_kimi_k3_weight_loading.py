from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.kimi_k3 import KimiK3LinearForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


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
def test_kimi_k3_post_load_only_processes_loaded_layers():
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
def test_kimi_k3_post_load_hook_processes_the_full_model():
    model = _model()

    KimiK3LinearForCausalLM.post_load_weights(model)

    for layer in model.model.layers:
        assert layer.mlp.merge_calls == 1
        assert layer.self_attn.merge_calls == 1
        assert layer.self_attn.prepare_calls == 1
