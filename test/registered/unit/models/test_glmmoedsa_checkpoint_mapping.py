import unittest
from types import SimpleNamespace

from sglang.srt.models.glm4_moe import _glm_moe_dsa_checkpoint_name_mapper
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


class TestGlmMoeDsaCheckpointMapping(CustomTestCase):
    def test_target_excludes_checkpoint_nextn_layers(self):
        mapper = _glm_moe_dsa_checkpoint_name_mapper(
            SimpleNamespace(num_hidden_layers=78, num_nextn_predict_layers=2)
        )

        self.assertEqual(
            mapper.apply_list(
                [
                    "model.layers.77.mlp.gate.weight",
                    "model.layers.78.eh_proj.weight",
                    "model.layers.79.mlp.experts.0.down_proj.weight",
                ]
            ),
            ["model.layers.77.mlp.gate.weight"],
        )

    def test_target_without_nextn_preserves_checkpoint_names(self):
        mapper = _glm_moe_dsa_checkpoint_name_mapper(
            SimpleNamespace(num_hidden_layers=78)
        )

        self.assertEqual(
            mapper.apply_list(["model.layers.77.mlp.gate.weight"]),
            ["model.layers.77.mlp.gate.weight"],
        )


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


if __name__ == "__main__":
    unittest.main()
