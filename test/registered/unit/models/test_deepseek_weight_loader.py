import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.model_loader.utils import DEFERRED_WEIGHT_COPY_SAFE_ATTR
from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
    NextNDisabledConfig,
    _expert_mapping_candidates,
    _normalize_modelopt_fp4_expert_weight,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepseekExpertMappingIndex(unittest.TestCase):
    def setUp(self):
        self.mappings = [
            (
                "experts.w13_",
                f"experts.{expert_id}.gate_proj.",
                expert_id,
                "w1",
            )
            for expert_id in range(384)
        ]
        self.index = {
            expert_id: [mapping] for expert_id, mapping in enumerate(self.mappings)
        }

    def test_standard_expert_name_uses_only_its_bucket(self):
        candidates = _expert_mapping_candidates(
            "model.layers.7.mlp.experts.271.gate_proj.weight",
            self.mappings,
            self.index,
        )
        self.assertEqual(candidates, [self.mappings[271]])

    def test_nonstandard_name_preserves_complete_fallback(self):
        candidates = _expert_mapping_candidates(
            "model.layers.7.mlp.routed.gate_proj.weight",
            self.mappings,
            self.index,
        )
        self.assertIs(candidates, self.mappings)

    def test_unknown_expert_preserves_complete_fallback(self):
        candidates = _expert_mapping_candidates(
            "model.layers.7.mlp.experts.999.gate_proj.weight",
            self.mappings,
            self.index,
        )
        self.assertIs(candidates, self.mappings)


class _BatchOwner:
    def __init__(self):
        self.immediate_calls = []
        self.batched_calls = []
        self.quant_method = object.__new__(ModelOptNvFp4FusedMoEMethod)

    def supports_batched_weight_loading(self):
        return self.quant_method.supports_batched_weight_loading()

    def weight_loader(self, *args, **kwargs):
        self.immediate_calls.append((args, kwargs))

    def batched_weight_loader(self, calls, *, executor):
        self.batched_calls.extend(calls)


def _deepseek_weight_loading_model(owner):
    param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    param.weight_loader = owner.weight_loader
    return SimpleNamespace(
        config=SimpleNamespace(n_routed_experts=1),
        model=SimpleNamespace(),
        num_fused_shared_experts=0,
        quant_config=SimpleNamespace(get_name=lambda: "modelopt_fp4"),
        _initialize_nextn_conf=lambda _: NextNDisabledConfig(),
        _maybe_quant_weights_to_fp8_ue8m0=lambda weights, *_: weights,
        named_parameters=lambda: [("model.layers.0.mlp.experts.w13_weight", param)],
        post_load_weights=lambda **_: None,
    )


class TestDeepseekStableWeightBatching(unittest.TestCase):
    def test_only_immutable_sources_are_deferred(self):
        owner = _BatchOwner()
        model = _deepseek_weight_loading_model(owner)
        stable = torch.ones(1)
        setattr(stable, DEFERRED_WEIGHT_COPY_SAFE_ATTR, True)

        DeepseekV2WeightLoaderMixin.do_load_weights(
            model,
            [
                ("model.layers.0.mlp.experts.0.gate_proj.weight", stable),
                (
                    "model.layers.0.mlp.experts.0.up_proj.weight",
                    torch.ones(1),
                ),
            ],
        )

        self.assertEqual(len(owner.batched_calls), 1)
        self.assertIs(owner.batched_calls[0][0][1], stable)
        self.assertEqual(len(owner.immediate_calls), 1)


class TestModelOptFp4ExpertWeightNormalization(unittest.TestCase):
    def test_scale_names(self):
        weight = torch.empty(1)

        name, _ = _normalize_modelopt_fp4_expert_weight(
            "model.layers.1.mlp.experts.2.gate_proj.weight_scale_inv",
            weight,
        )
        self.assertEqual(name, "model.layers.1.mlp.experts.2.gate_proj.weight_scale")

        name, _ = _normalize_modelopt_fp4_expert_weight(
            "model.layers.1.mlp.experts.2.gate_proj.weight_scale_global",
            weight,
        )
        self.assertEqual(name, "model.layers.1.mlp.experts.2.gate_proj.weight_scale_2")

    @unittest.skipUnless(
        hasattr(torch, "float4_e2m1fn_x2"), "Torch FP4 dtype is unavailable"
    )
    def test_fp4_storage_is_viewed_as_packed_uint8(self):
        weight = torch.empty(4, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        setattr(weight, DEFERRED_WEIGHT_COPY_SAFE_ATTR, True)
        setattr(weight, RUNAI_STREAMER_TENSOR_ATTR, True)

        _, packed_weight = _normalize_modelopt_fp4_expert_weight(
            "model.layers.1.mlp.experts.2.gate_proj.weight",
            weight,
        )

        self.assertEqual(packed_weight.dtype, torch.uint8)
        self.assertEqual(packed_weight.shape, weight.shape)
        self.assertEqual(packed_weight.data_ptr(), weight.data_ptr())
        self.assertTrue(getattr(packed_weight, DEFERRED_WEIGHT_COPY_SAFE_ATTR))
        self.assertTrue(getattr(packed_weight, RUNAI_STREAMER_TENSOR_ATTR))


class TestDeepseekDerivedWeightInventory(unittest.TestCase):
    def test_mla_exposes_only_tensor_derived_weights(self):
        attention = DeepseekV2AttentionMLA.__new__(DeepseekV2AttentionMLA)
        attention.w_kc = torch.ones(1)
        attention.w_vc = torch.ones(2)
        attention.w_scale = 1.0
        attention.w_scale_k = torch.ones(3)
        attention.w_scale_v = None
        attention.w_kc_qrep = torch.ones(4)
        attention.q_b_proj_qrep_weight = torch.ones(5)

        tensors = dict(attention.get_additional_weight_tensors())

        self.assertEqual(
            set(tensors),
            {"w_kc", "w_vc", "w_scale_k", "w_kc_qrep", "q_b_proj_qrep_weight"},
        )


if __name__ == "__main__":
    unittest.main()
