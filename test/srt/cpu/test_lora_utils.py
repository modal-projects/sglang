import unittest
from types import SimpleNamespace

from sglang.srt.lora.utils import (
    get_hidden_dim,
    get_runtime_lora_target_module_name,
)


class TestLoRAUtils(unittest.TestCase):
    def test_runtime_target_name_distinguishes_shared_expert_modules(self):
        target_modules = {"gate_up_proj", "down_proj"}

        self.assertEqual(
            get_runtime_lora_target_module_name(
                "model.layers.0.mlp.down_proj", target_modules
            ),
            "down_proj",
        )
        self.assertEqual(
            get_runtime_lora_target_module_name(
                "model.layers.0.mlp.shared_expert.down_proj", target_modules
            ),
            "shared_expert.down_proj",
        )
        self.assertEqual(
            get_runtime_lora_target_module_name(
                "model.layers.0.mlp.shared_experts.gate_up_proj", target_modules
            ),
            "shared_expert.gate_up_proj",
        )

    def test_hidden_dim_uses_shared_expert_intermediate_size(self):
        config = SimpleNamespace(
            hidden_size=7168,
            intermediate_size=5632,
            shared_expert_intermediate_size=512,
        )
        base_model = SimpleNamespace()

        self.assertEqual(
            get_hidden_dim("shared_expert.down_proj", config, base_model, 0),
            (512, 7168),
        )
        self.assertEqual(
            get_hidden_dim("shared_expert.gate_up_proj", config, base_model, 0),
            (7168, 1024),
        )

    def test_hidden_dim_accounts_for_attention_output_gate(self):
        config = SimpleNamespace(
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=256,
            attn_output_gate=True,
        )
        base_model = SimpleNamespace()

        self.assertEqual(
            get_hidden_dim("qkv_proj", config, base_model, 0),
            (2048, 9216),
        )


if __name__ == "__main__":
    unittest.main()
