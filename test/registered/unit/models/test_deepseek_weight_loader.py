import unittest

import torch

from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _normalize_modelopt_fp4_expert_weight,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
