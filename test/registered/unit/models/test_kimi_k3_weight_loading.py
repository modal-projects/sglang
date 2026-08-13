from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attn_residual import get_cw
from sglang.srt.model_loader.utils import DEFERRED_WEIGHT_COPY_SAFE_ATTR
from sglang.srt.models.kimi_k3 import (
    KimiK3DeltaAttention,
    KimiK3LinearForCausalLM,
    _expert_mapping_candidates,
    _merge_weights_as_views,
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


def test_output_attention_residual_cache_requires_both_source_modules():
    model = _model()
    model.model.output_attn_res_proj = SimpleNamespace()
    model.model.output_attn_res_norm = SimpleNamespace()

    with patch("sglang.srt.models.kimi_k3.get_cw") as get_cw_mock:
        KimiK3LinearForCausalLM.post_load_weights(
            model,
            weight_names={"model.output_attn_res_proj.weight"},
        )
        assert get_cw_mock.call_count == 0

        KimiK3LinearForCausalLM.post_load_weights(
            model,
            weight_names={
                "model.output_attn_res_proj.weight",
                "model.output_attn_res_norm.weight",
            },
        )

    assert get_cw_mock.call_count == 2


def test_decoder_layers_are_indivisible_weight_staging_units():
    from sglang.srt.models.kimi_k3 import KimiK3DecoderLayer

    assert KimiK3DecoderLayer.weight_staging_indivisible


def test_weight_commit_refreshes_attention_residual_caches():
    def projection():
        return SimpleNamespace(_attn_res_cw_cache={torch.float32: torch.ones(1)})

    layer = SimpleNamespace(
        use_attn_residuals=True,
        self_attention_res_proj=projection(),
        self_attention_res_norm=SimpleNamespace(),
        mlp_res_proj=projection(),
        mlp_res_norm=SimpleNamespace(),
    )
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[layer],
            output_attn_res_proj=projection(),
            output_attn_res_norm=SimpleNamespace(),
        )
    )

    with patch("sglang.srt.models.kimi_k3.get_cw") as get_cw_mock:
        KimiK3LinearForCausalLM.process_weights_after_weight_commit(model)

    assert get_cw_mock.call_count == 6
    assert all(call.kwargs.get("refresh") for call in get_cw_mock.call_args_list)


def test_attention_residual_cache_refresh_preserves_storage():
    projection = SimpleNamespace(weight=torch.tensor([[2.0, 3.0]]))
    norm = SimpleNamespace(weight=torch.tensor([5.0, 7.0]))
    cached = get_cw(projection, norm)
    data_ptr = cached.data_ptr()

    projection.weight.copy_(torch.tensor([[11.0, 13.0]]))
    refreshed = get_cw(projection, norm, refresh=True)

    assert refreshed.data_ptr() == data_ptr
    torch.testing.assert_close(refreshed, torch.tensor([55.0, 91.0]))


def test_mla_post_load_refresh_preserves_runtime_storage():
    kv_b_proj = SimpleNamespace(
        weight=torch.arange(24, dtype=torch.float32).reshape(6, 4)
    )
    self_attn = SimpleNamespace(
        kv_b_proj=kv_b_proj,
        qk_nope_head_dim=1,
        v_head_dim=2,
        w_kc=torch.empty(2, 1, 4),
        w_vc=torch.empty(2, 4, 2),
    )
    layer = SimpleNamespace(
        use_attn_residuals=False,
        mlp=SimpleNamespace(),
        self_attn=self_attn,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(full_attention_layer_ids=[0]),
        model=SimpleNamespace(start_layer=0, end_layer=1, layers=[layer]),
    )

    KimiK3LinearForCausalLM.post_load_weights(model)
    pointers = (self_attn.w_kc.data_ptr(), self_attn.w_vc.data_ptr())
    kv_b_proj.weight.add_(1)
    KimiK3LinearForCausalLM.post_load_weights(model)

    assert (self_attn.w_kc.data_ptr(), self_attn.w_vc.data_ptr()) == pointers
    w_kc, w_vc = kv_b_proj.weight.unflatten(0, (-1, 3)).split([1, 2], dim=1)
    torch.testing.assert_close(
        self_attn.w_kc,
        w_kc.transpose(1, 2).contiguous().transpose(1, 2),
    )
    torch.testing.assert_close(self_attn.w_vc, w_vc.contiguous().transpose(1, 2))


def test_weight_merge_reuses_existing_runtime_storage():
    modules = [torch.nn.Linear(2, 2, bias=False) for _ in range(2)]
    merged, sizes = _merge_weights_as_views(modules, pad_rows_to=8)
    data_ptr = merged.data_ptr()

    modules[0].weight.data.fill_(3)
    modules[1].weight.data.fill_(5)
    reloaded, reloaded_sizes = _merge_weights_as_views(
        modules,
        pad_rows_to=8,
        merged=merged,
    )

    assert reloaded.data_ptr() == data_ptr
    assert reloaded_sizes == sizes
    torch.testing.assert_close(reloaded[:2], torch.full((2, 2), 3.0))
    torch.testing.assert_close(reloaded[2:4], torch.full((2, 2), 5.0))


def test_fused_kda_decode_refresh_preserves_runtime_storage():
    segment = 12 * 128
    attention = SimpleNamespace(
        conv_weights=torch.arange(3 * segment * 4, dtype=torch.float32).reshape(
            3 * segment, 4
        ),
        bias=torch.arange(3 * segment, dtype=torch.float32),
        A_log=torch.arange(12, dtype=torch.float32),
        dt_bias=torch.arange(segment, dtype=torch.float32),
    )
    model = SimpleNamespace(
        attn=attention,
        o_norm=SimpleNamespace(
            weight=torch.arange(segment, dtype=torch.float32),
            eps=1e-5,
        ),
        _kda_fused_decode_ready=False,
    )

    KimiK3DeltaAttention._prepare_fused_decode(model)
    pointers = [tensor.data_ptr() for tensor in attention._k3_fused_decode_args[:-1]]
    attention.conv_weights.add_(1)
    KimiK3DeltaAttention._prepare_fused_decode(model)

    assert [
        tensor.data_ptr() for tensor in attention._k3_fused_decode_args[:-1]
    ] == pointers
    additional = dict(KimiK3DeltaAttention.get_additional_weight_tensors(model))
    assert len(additional) == 6
    torch.testing.assert_close(
        additional["k3_fused_decode_arg_0"],
        attention.conv_weights.t()[:, :segment],
    )
