import unittest

import torch

from sglang.srt.model_loader.utils import DEFERRED_WEIGHT_COPY_SAFE_ATTR
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _expert_mapping_candidates,
    _normalize_modelopt_fp4_expert_weight,
)
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

        _, packed_weight = _normalize_modelopt_fp4_expert_weight(
            "model.layers.1.mlp.experts.2.gate_proj.weight",
            weight,
        )

        self.assertEqual(packed_weight.dtype, torch.uint8)
        self.assertEqual(packed_weight.shape, weight.shape)
        self.assertEqual(packed_weight.data_ptr(), weight.data_ptr())
        self.assertTrue(getattr(packed_weight, DEFERRED_WEIGHT_COPY_SAFE_ATTR))


if __name__ == "__main__":
    unittest.main()
