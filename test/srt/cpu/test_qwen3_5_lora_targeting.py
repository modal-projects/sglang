import unittest

try:
    from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"Missing test dependency: {exc}")


class TestQwen35LoRATargeting(unittest.TestCase):
    def test_runtime_lora_pattern_matches_qwen3_5_module_names(self):
        pattern = Qwen3_5ForConditionalGeneration._lora_pattern

        self.assertTrue(pattern.match("model.layers.0.qkv_proj"))
        self.assertTrue(pattern.match("model.layers.0.o_proj"))
        self.assertTrue(pattern.match("model.layers.0.self_attn.qkv_proj"))
        self.assertTrue(pattern.match("model.layers.0.mlp.gate_up_proj"))
        self.assertTrue(pattern.match("model.layers.0.mlp.shared_expert.down_proj"))
        self.assertTrue(pattern.match("model.layers.0.linear_attn.in_proj_qkvz"))
        self.assertFalse(pattern.match("model.layers.0.self_attn"))


if __name__ == "__main__":
    unittest.main()
