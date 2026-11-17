"""Qwen3.5 GDN LoRA packed-projection contracts."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.utils import (
    ATTN_TP_LORA_MODULE_NAMES,
    get_normalized_target_modules,
    get_stacked_multiply,
)
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.utils.common import SUPPORTED_LORA_TARGET_MODULES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PREFIX = "base_model.model.model.layers.0.linear_attn"


def _weight_name(module_name: str, factor: str) -> str:
    return f"{_PREFIX}.{module_name}.lora_{factor}.weight"


def _adapter() -> LoRAAdapter:
    return LoRAAdapter.__new__(LoRAAdapter)


class TestQwen3_5GDNTargetContracts(CustomTestCase):
    def test_checkpoint_targets_normalize_to_fused_runtime_modules(self):
        fused = {"in_proj_qkvz", "in_proj_ba", "out_proj"}
        for targets in (
            ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"],
            [
                "in_proj_q",
                "in_proj_k",
                "in_proj_v",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "out_proj",
            ],
            ["in_proj_qkvz", "in_proj_ba", "out_proj"],
        ):
            with self.subTest(targets=targets):
                self.assertEqual(get_normalized_target_modules(targets), fused)

    def test_fused_runtime_modules_are_declared_and_sized(self):
        self.assertIn("in_proj_ba", Qwen3_5ForCausalLM.supported_lora_modules)
        self.assertEqual(get_stacked_multiply("in_proj_qkvz"), 4)
        self.assertEqual(get_stacked_multiply("in_proj_ba"), 2)
        self.assertIn("in_proj_ba", ATTN_TP_LORA_MODULE_NAMES)
        for module_name in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
            self.assertIn(module_name, SUPPORTED_LORA_TARGET_MODULES)

        model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(
            hidden_size=4096,
            head_dim=128,
            num_attention_heads=32,
            linear_num_value_heads=32,
        )
        self.assertEqual(model.get_hidden_dim("in_proj_ba", 0), (4096, 64))


class TestQwen3_5GDNWeightNormalization(CustomTestCase):
    def test_fused_in_proj_export_stacks_into_runtime_modules(self):
        # Megatron-Bridge exports the fused in_proj adapter as four modules
        # that share one lora_A.
        lora_a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        lora_b = {
            "in_proj_qkv": torch.arange(12, dtype=torch.float32).reshape(6, 2),
            "in_proj_z": torch.full((2, 2), 100.0),
            "in_proj_b": torch.full((1, 2), 200.0),
            "in_proj_a": torch.full((1, 2), 300.0),
        }
        weights = {}
        for module_name, b in lora_b.items():
            weights[_weight_name(module_name, "A")] = lora_a.clone()
            weights[_weight_name(module_name, "B")] = b

        adapter = _adapter()
        adapter._normalize_in_proj_qkvz(weights)
        adapter._normalize_in_proj_ba(weights)

        self.assertEqual(
            set(weights),
            {
                _weight_name(module_name, factor)
                for module_name in ("in_proj_qkvz", "in_proj_ba")
                for factor in ("A", "B")
            },
        )
        torch.testing.assert_close(
            weights[_weight_name("in_proj_qkvz", "A")], lora_a.repeat(4, 1)
        )
        torch.testing.assert_close(
            weights[_weight_name("in_proj_qkvz", "B")],
            torch.cat((lora_b["in_proj_qkv"], lora_b["in_proj_z"])),
        )
        torch.testing.assert_close(
            weights[_weight_name("in_proj_ba", "A")], lora_a.repeat(2, 1)
        )
        torch.testing.assert_close(
            weights[_weight_name("in_proj_ba", "B")],
            torch.cat((lora_b["in_proj_b"], lora_b["in_proj_a"])),
        )

    def test_already_fused_ba_repeats_only_the_shared_a_factor(self):
        fused_a = _weight_name("in_proj_ba", "A")
        fused_b = _weight_name("in_proj_ba", "B")
        lora_a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        lora_b = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        weights = {fused_a: lora_a.clone(), fused_b: lora_b.clone()}

        _adapter()._normalize_in_proj_ba(weights)

        torch.testing.assert_close(weights[fused_a], lora_a.repeat(2, 1))
        torch.testing.assert_close(weights[fused_b], lora_b)


if __name__ == "__main__":
    unittest.main()
