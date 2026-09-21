import unittest

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _normalize_modelopt_fp4_expert_weight,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestExpertMappingCandidates(unittest.TestCase):
    def test_uses_encoded_expert_id(self):
        mappings = [
            ("w13", "w1", 0, "w1"),
            ("w2", "w2", 0, "w2"),
            ("w13", "w1", 1, "w1"),
            ("w2", "w2", 1, "w2"),
        ]
        by_expert = {
            expert_id: [mapping for mapping in mappings if mapping[2] == expert_id]
            for expert_id in range(2)
        }

        candidates = FusedMoE.get_expert_params_mapping_candidates(
            "model.layers.3.mlp.experts.1.w2.weight",
            mappings,
            by_expert,
        )

        self.assertEqual(candidates, by_expert[1])

    def test_mapping_candidates_fall_back_without_an_explicit_expert(self):
        mappings = [("w13", "w1", 0, "w1")]
        by_expert = FusedMoE.index_expert_params_mapping(mappings)

        self.assertIs(
            FusedMoE.get_expert_params_mapping_candidates(
                "model.layers.3.mlp.shared_experts.gate_proj.weight",
                mappings,
                by_expert,
            ),
            mappings,
        )

    def test_mapping_candidates_reject_an_unknown_explicit_expert(self):
        mappings = [("w13", "w1", 0, "w1")]
        by_expert = FusedMoE.index_expert_params_mapping(mappings)

        self.assertEqual(
            FusedMoE.get_expert_params_mapping_candidates(
                "model.layers.3.mlp.experts.7.gate_proj.weight",
                mappings,
                by_expert,
            ),
            [],
        )


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
        setattr(weight, RUNAI_STREAMER_TENSOR_ATTR, True)

        _, packed_weight = _normalize_modelopt_fp4_expert_weight(
            "model.layers.1.mlp.experts.2.gate_proj.weight",
            weight,
        )

        self.assertEqual(packed_weight.dtype, torch.uint8)
        self.assertEqual(packed_weight.shape, weight.shape)
        self.assertEqual(packed_weight.data_ptr(), weight.data_ptr())
        self.assertTrue(getattr(packed_weight, RUNAI_STREAMER_TENSOR_ATTR))


if __name__ == "__main__":
    unittest.main()
