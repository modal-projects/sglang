import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteForCausalLM
from sglang.srt.models.glm4_moe_lite_nextn import Glm4MoeLiteForCausalLMNextN
from sglang.srt.models.utils import WeightsMapper
from sglang.test.test_utils import CustomTestCase


class TestGlm4MoeLiteCheckpointMapping(CustomTestCase):
    def mapper(self, model_type, config):
        model = model_type.__new__(model_type)
        torch.nn.Module.__init__(model)
        model.config = config
        return getattr(model, "checkpoint_name_mapper", None) or WeightsMapper()

    def test_target_excludes_only_configured_nextn_layers(self):
        mapper = self.mapper(
            Glm4MoeLiteForCausalLM,
            SimpleNamespace(num_hidden_layers=47, num_nextn_predict_layers=2),
        )
        names = [
            "model.layers.46.mlp.gate.weight",
            "model.layers.47.eh_proj.weight",
            "model.layers.48.shared_head.head.weight",
            "model.layers.49.mlp.gate.weight",
            "model.layers.470.mlp.gate.weight",
            "unknown.weight",
        ]
        self.assertEqual(mapper.apply_list(names), [names[0], *names[3:]])

    def test_target_without_nextn_preserves_checkpoint_names(self):
        for config in (
            SimpleNamespace(num_hidden_layers=47),
            SimpleNamespace(num_hidden_layers=47, num_nextn_predict_layers=0),
        ):
            with self.subTest(config=config):
                mapper = self.mapper(Glm4MoeLiteForCausalLM, config)
                names = ["model.layers.46.mlp.gate.weight"]
                self.assertEqual(mapper.apply_list(names), names)

    def test_draft_preserves_its_nextn_weights(self):
        mapper = self.mapper(
            Glm4MoeLiteForCausalLMNextN,
            SimpleNamespace(num_hidden_layers=47, num_nextn_predict_layers=1),
        )
        names = ["model.layers.47.eh_proj.weight"]
        self.assertEqual(mapper.apply_list(names), names)


if __name__ == "__main__":
    unittest.main()
