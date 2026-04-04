import unittest

try:
    import torch

    from sglang.srt.lora.peft import compile_peft_lora_payload
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"Missing test dependency: {exc}")


class TestPeftLoRACompiler(unittest.TestCase):
    def test_compile_qwen3_5_dense_targets(self):
        named_tensors = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(
                2, 3
            ),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(
                4, 2
            ),
            "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight": torch.full(
                (2, 3), 2.0
            ),
            "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight": torch.full(
                (5, 2), 3.0
            ),
            "base_model.model.model.layers.0.mlp.shared_expert.down_proj.lora_A.weight": torch.full(
                (2, 6), 4.0
            ),
            "base_model.model.model.layers.0.mlp.shared_expert.down_proj.lora_B.weight": torch.full(
                (3, 2), 5.0
            ),
        }

        payload_tensors, loader_metadata = compile_peft_lora_payload(
            named_tensors,
            target_resolver="qwen3_5",
            loader_metadata={
                "adapter_config": {"r": 2, "lora_alpha": 4},
                "custom_note": "keep-me",
                "strict": False,
            },
        )

        self.assertEqual(set(dict(payload_tensors)), set(named_tensors))
        self.assertEqual(loader_metadata["custom_note"], "keep-me")
        self.assertEqual(loader_metadata["rank"], 2)
        self.assertEqual(loader_metadata["lora_alpha"], 4.0)
        self.assertNotIn("adapter_config", loader_metadata)
        self.assertNotIn("strict", loader_metadata)

        targets = {
            item["target_name"]: item["components"]
            for item in loader_metadata["targets"]
        }
        self.assertEqual(
            targets["model.layers.0.qkv_proj.weight"][0]["shard_id"],
            "q",
        )
        self.assertEqual(
            targets["model.layers.0.mlp.gate_up_proj.weight"][0]["shard_id"],
            1,
        )
        self.assertEqual(
            targets["model.layers.0.mlp.shared_expert.down_proj.weight"][0][
                "component_id"
            ],
            "down_proj",
        )

    def test_compile_qwen3_5_stacks_routed_experts_independent_of_order(self):
        named_tensors = [
            (
                "base_model.model.model.layers.0.mlp.experts.2.gate_proj.lora_A.weight",
                torch.full((2, 3), 2.0),
            ),
            (
                "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight",
                torch.full((2, 3), 1.0),
            ),
            (
                "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_B.weight",
                torch.full((4, 2), 3.0),
            ),
            (
                "base_model.model.model.layers.0.mlp.experts.2.gate_proj.lora_B.weight",
                torch.full((4, 2), 4.0),
            ),
        ]

        payload_tensors, loader_metadata = compile_peft_lora_payload(
            named_tensors,
            target_resolver="qwen3_5",
        )

        payload_dict = dict(payload_tensors)
        packed_a_name = next(
            name
            for name in payload_dict
            if name.endswith("model.layers.0.mlp.experts.w13_weight.w1.lora_a.weight")
        )
        packed_b_name = next(
            name
            for name in payload_dict
            if name.endswith("model.layers.0.mlp.experts.w13_weight.w1.lora_b.weight")
        )
        self.assertEqual(tuple(payload_dict[packed_a_name].shape), (3, 2, 3))
        self.assertEqual(tuple(payload_dict[packed_b_name].shape), (3, 4, 2))
        self.assertTrue(torch.equal(payload_dict[packed_a_name][1], torch.zeros(2, 3)))

        targets = {
            item["target_name"]: item["components"]
            for item in loader_metadata["targets"]
        }
        self.assertEqual(
            targets["model.layers.0.mlp.experts.w13_weight"][0]["shard_id"],
            "w1",
        )

    def test_compile_qwen3_5_rejects_mismatched_routed_expert_factor_sets(self):
        named_tensors = {
            "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.ones(
                2, 3
            ),
            "base_model.model.model.layers.0.mlp.experts.2.gate_proj.lora_B.weight": torch.ones(
                4, 2
            ),
        }

        with self.assertRaisesRegex(ValueError, "identical expert ids"):
            compile_peft_lora_payload(
                named_tensors,
                target_resolver="qwen3_5",
            )


if __name__ == "__main__":
    unittest.main()
