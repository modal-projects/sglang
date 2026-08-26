"""NextN draft quant-config rename: ignore entries written against the
checkpoint spelling (model.layers.{N}.*) must survive the wrapper's rename to
model.decoder.* / hoisted root modules, for any layer index N."""

import unittest
from types import SimpleNamespace

from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN


class TestNextNQuantMapper(unittest.TestCase):
    def _mapper(self, n):
        return DeepseekV3ForCausalLMNextN.get_hf_to_sglang_mapper(
            SimpleNamespace(num_hidden_layers=n)
        )

    def test_layer_index_is_config_dependent(self):
        for n in (61, 78, 92):
            mapper = self._mapper(n)
            self.assertEqual(
                mapper._map_name(f"model.layers.{n}.input_layernorm"),
                "model.decoder.input_layernorm",
            )

    def test_hoisted_modules_map_to_model_root(self):
        mapper = self._mapper(78)
        cases = {
            "model.layers.78.eh_proj": "model.eh_proj",
            "model.layers.78.enorm": "model.enorm",
            "model.layers.78.hnorm": "model.hnorm",
            "model.layers.78.shared_head.norm": "model.shared_head.norm",
            "model.layers.78.self_attn.q_a_layernorm": (
                "model.decoder.self_attn.q_a_layernorm"
            ),
        }
        for orig, renamed in cases.items():
            self.assertEqual(mapper._map_name(orig), renamed)

    def test_apply_list_full_ignore_list(self):
        mapper = self._mapper(78)
        ignore = [
            "model.layers.78.eh_proj",
            "model.layers.78.enorm",
            "model.layers.78.input_layernorm",
            "lm_head",
        ]
        mapped = mapper.apply_list(ignore)
        self.assertIn("model.eh_proj", mapped)
        self.assertIn("model.enorm", mapped)
        self.assertIn("model.decoder.input_layernorm", mapped)
        self.assertIn("lm_head", mapped)  # non-layer entries pass through

    def test_already_renamed_entries_pass_through(self):
        # Idempotence for checkpoints patched with mirrored spellings.
        mapper = self._mapper(78)
        for name in ("model.decoder.input_layernorm", "model.eh_proj", "eh_proj"):
            self.assertEqual(mapper._map_name(name), name)


if __name__ == "__main__":
    unittest.main()
