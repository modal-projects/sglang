import unittest

from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _expert_mapping_candidates,
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


if __name__ == "__main__":
    unittest.main()
