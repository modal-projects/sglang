"""Unit tests for Kimi K3 LoRA target names and buffer dims.

Usage:
    python -m pytest test/registered/unit/lora/test_kimi_k3_lora_unit.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

from sglang.srt.lora.utils import get_default_hidden_dim, get_normalized_target_modules
from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration

K3 = SimpleNamespace(
    lora_target_module_aliases=KimiK3ForConditionalGeneration.lora_target_module_aliases
)
# The HF targets Miles exports for Kimi K3 adapters.
MILES_TARGETS = [
    f"language_model.model.layers.*.{module}"
    for module in (
        "self_attn.o_proj",
        "self_attn.q_a_proj",
        "self_attn.kv_a_proj_with_mqa",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "block_sparse_moe.experts.*.w1",
        "block_sparse_moe.experts.*.w2",
        "block_sparse_moe.experts.*.w3",
        "block_sparse_moe.shared_experts.gate_proj",
        "block_sparse_moe.shared_experts.up_proj",
        "block_sparse_moe.shared_experts.down_proj",
    )
]


def _k3_text_config():
    return SimpleNamespace(
        hidden_size=7168,
        num_attention_heads=96,
        intermediate_size=33792,
        moe_intermediate_size=3072,
        routed_expert_hidden_size=3584,
        num_shared_experts=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
    )


class TestKimiK3LoRA(unittest.TestCase):
    def test_miles_expert_targets_normalize(self):
        self.assertEqual(
            get_normalized_target_modules(MILES_TARGETS, K3),
            {"fused_qkv_a_proj_with_mqa", "o_proj", "gate_up_proj", "down_proj"},
        )

    def test_routed_experts_use_the_latent_width(self):
        config = _k3_text_config()
        self.assertEqual(
            get_default_hidden_dim("gate_up_proj_moe", config, 1), (3584, 6144)
        )
        self.assertEqual(
            get_default_hidden_dim("down_proj_moe", config, 1), (3072, 3584)
        )

    def test_shared_experts_count_from_num_shared_experts(self):
        config = _k3_text_config()
        self.assertEqual(
            get_default_hidden_dim("gate_up_proj", config, 1), (7168, 12288)
        )
        self.assertEqual(get_default_hidden_dim("down_proj", config, 1), (6144, 7168))
        # Layer 0 is the dense MLP.
        self.assertEqual(
            get_default_hidden_dim("gate_up_proj", config, 0), (7168, 67584)
        )


if __name__ == "__main__":
    unittest.main()
