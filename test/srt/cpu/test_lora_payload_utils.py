import unittest

try:
    import torch

    from sglang.srt.weight_sync.lora_payload_utils import (
        convert_peft_lora_tensors_to_weight_sync_payload,
    )
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"Missing test dependency: {exc}")


def _targets_by_name(loader_metadata):
    return {target["target_name"]: target for target in loader_metadata["targets"]}


class TestLoRAPayloadUtils(unittest.TestCase):
    def test_converts_dense_qkv_and_gate_up_targets(self):
        adapter_tensors = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(
                2, 4
            ),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(
                6, 2
            ),
            "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight": torch.full(
                (2, 4), 2.0
            ),
            "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight": torch.full(
                (4, 2), 3.0
            ),
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": torch.full(
                (2, 4), 4.0
            ),
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": torch.full(
                (4, 2), 5.0
            ),
            "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight": torch.ones(
                2, 4
            ),
            "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": torch.ones(
                8, 2
            ),
            "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight": torch.full(
                (2, 4), 6.0
            ),
            "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight": torch.full(
                (8, 2), 7.0
            ),
            "base_model.model.model.unembed_tokens.lora_A.weight": torch.full(
                (2, 4), 8.0
            ),
            "base_model.model.model.unembed_tokens.lora_B.weight": torch.full(
                (10, 2), 9.0
            ),
        }

        prepared = convert_peft_lora_tensors_to_weight_sync_payload(
            adapter_tensors,
            adapter_config={"lora_alpha": 16, "r": 8},
        )
        targets = _targets_by_name(prepared.loader_metadata)
        tensor_dict = dict(prepared.named_tensors)

        self.assertEqual(prepared.skipped_tensor_names, [])
        self.assertEqual(prepared.loader_metadata["lora_alpha"], 16)
        self.assertEqual(prepared.loader_metadata["rank"], 8)

        qkv_target = targets["model.layers.0.self_attn.qkv_proj.weight"]
        self.assertEqual(
            {component["shard_id"] for component in qkv_target["components"]},
            {"q", "k", "v"},
        )
        self.assertIn(
            "model.layers.0.self_attn.qkv_proj.weight.q.lora_A",
            tensor_dict,
        )
        self.assertIn(
            "model.layers.0.self_attn.qkv_proj.weight.k.lora_B",
            tensor_dict,
        )

        gate_up_target = targets["model.layers.0.mlp.gate_up_proj.weight"]
        self.assertEqual(
            {component["shard_id"] for component in gate_up_target["components"]},
            {0, 1},
        )
        self.assertIn("model.layers.0.mlp.gate_up_proj.weight.gate.lora_A", tensor_dict)
        self.assertIn("model.layers.0.mlp.gate_up_proj.weight.up.lora_B", tensor_dict)

        lm_head_target = targets["lm_head.weight"]
        self.assertEqual(len(lm_head_target["components"]), 1)
        self.assertIn("lm_head.weight.lora_A", tensor_dict)
        self.assertIn("lm_head.weight.lora_B", tensor_dict)

    def test_stacks_per_expert_moe_tensors(self):
        adapter_tensors = {
            "base_model.model.model.layers.1.mlp.experts.0.gate_proj.lora_A.weight": torch.tensor(
                [[1.0, 2.0, 3.0]]
            ),
            "base_model.model.model.layers.1.mlp.experts.1.gate_proj.lora_A.weight": torch.tensor(
                [[4.0, 5.0, 6.0]]
            ),
            "base_model.model.model.layers.1.mlp.experts.0.gate_proj.lora_B.weight": torch.tensor(
                [[1.0], [2.0]]
            ),
            "base_model.model.model.layers.1.mlp.experts.1.gate_proj.lora_B.weight": torch.tensor(
                [[3.0], [4.0]]
            ),
            "base_model.model.model.layers.1.mlp.experts.0.up_proj.lora_A.weight": torch.tensor(
                [[7.0, 8.0, 9.0]]
            ),
            "base_model.model.model.layers.1.mlp.experts.1.up_proj.lora_A.weight": torch.tensor(
                [[10.0, 11.0, 12.0]]
            ),
            "base_model.model.model.layers.1.mlp.experts.0.up_proj.lora_B.weight": torch.tensor(
                [[5.0], [6.0]]
            ),
            "base_model.model.model.layers.1.mlp.experts.1.up_proj.lora_B.weight": torch.tensor(
                [[7.0], [8.0]]
            ),
            "base_model.model.model.layers.1.mlp.experts.0.down_proj.lora_A.weight": torch.tensor(
                [[1.0, 0.0]]
            ),
            "base_model.model.model.layers.1.mlp.experts.1.down_proj.lora_A.weight": torch.tensor(
                [[0.0, 1.0]]
            ),
            "base_model.model.model.layers.1.mlp.experts.0.down_proj.lora_B.weight": torch.tensor(
                [[2.0], [3.0], [4.0]]
            ),
            "base_model.model.model.layers.1.mlp.experts.1.down_proj.lora_B.weight": torch.tensor(
                [[5.0], [6.0], [7.0]]
            ),
        }

        prepared = convert_peft_lora_tensors_to_weight_sync_payload(adapter_tensors)
        targets = _targets_by_name(prepared.loader_metadata)
        tensor_dict = dict(prepared.named_tensors)

        w13_target = targets["model.layers.1.mlp.experts.w13_weight"]
        self.assertEqual(
            {component["shard_id"] for component in w13_target["components"]},
            {"w1", "w3"},
        )
        self.assertTrue(all(component["fused_experts"] for component in w13_target["components"]))
        self.assertEqual(
            tuple(tensor_dict["model.layers.1.mlp.experts.w13_weight.w1.lora_A"].shape),
            (2, 1, 3),
        )
        self.assertEqual(
            tuple(tensor_dict["model.layers.1.mlp.experts.w13_weight.w3.lora_B"].shape),
            (2, 2, 1),
        )

        w2_target = targets["model.layers.1.mlp.experts.w2_weight"]
        self.assertEqual(w2_target["components"][0]["shard_id"], "w2")
        self.assertTrue(w2_target["components"][0]["fused_experts"])
        self.assertEqual(
            tuple(tensor_dict["model.layers.1.mlp.experts.w2_weight.w2.lora_B"].shape),
            (2, 3, 1),
        )

    def test_splits_direct_fused_gate_up_and_skips_visual_tensors(self):
        adapter_tensors = {
            "base_model.model.model.layers.2.mlp.experts.gate_up_proj.lora_A.weight": torch.arange(
                12, dtype=torch.float32
            ).reshape(2, 2, 3),
            "base_model.model.model.layers.2.mlp.experts.gate_up_proj.lora_B.weight": torch.arange(
                16, dtype=torch.float32
            ).reshape(2, 4, 2),
            "base_model.model.visual.blocks.0.attn.q_proj.lora_A.weight": torch.ones(
                2, 2
            ),
        }

        prepared = convert_peft_lora_tensors_to_weight_sync_payload(adapter_tensors)
        targets = _targets_by_name(prepared.loader_metadata)
        tensor_dict = dict(prepared.named_tensors)

        self.assertIn(
            "base_model.model.visual.blocks.0.attn.q_proj.lora_A.weight",
            prepared.skipped_tensor_names,
        )
        w13_target = targets["model.layers.2.mlp.experts.w13_weight"]
        self.assertEqual(len(w13_target["components"]), 2)
        self.assertEqual(
            tuple(tensor_dict["model.layers.2.mlp.experts.w13_weight.w1.lora_A"].shape),
            (2, 1, 3),
        )
        self.assertEqual(
            tuple(tensor_dict["model.layers.2.mlp.experts.w13_weight.w3.lora_B"].shape),
            (2, 2, 2),
        )

    def test_converts_direct_fused_w1_w2_w3_expert_tensors(self):
        adapter_tensors = {
            "base_model.model.model.layers.3.mlp.experts.w1.lora_A.weight": torch.ones(
                2, 1, 4
            ),
            "base_model.model.model.layers.3.mlp.experts.w1.lora_B.weight": torch.ones(
                2, 8, 1
            ),
            "base_model.model.model.layers.3.mlp.experts.w2.lora_A.weight": torch.full(
                (2, 1, 4), 2.0
            ),
            "base_model.model.model.layers.3.mlp.experts.w2.lora_B.weight": torch.full(
                (2, 4, 1), 3.0
            ),
            "base_model.model.model.layers.3.mlp.experts.w3.lora_A.weight": torch.full(
                (2, 1, 4), 4.0
            ),
            "base_model.model.model.layers.3.mlp.experts.w3.lora_B.weight": torch.full(
                (2, 8, 1), 5.0
            ),
        }

        prepared = convert_peft_lora_tensors_to_weight_sync_payload(adapter_tensors)
        targets = _targets_by_name(prepared.loader_metadata)
        tensor_dict = dict(prepared.named_tensors)

        self.assertEqual(prepared.skipped_tensor_names, [])

        w13_target = targets["model.layers.3.mlp.experts.w13_weight"]
        self.assertEqual(
            {component["shard_id"] for component in w13_target["components"]},
            {"w1", "w3"},
        )
        self.assertTrue(all(component["fused_experts"] for component in w13_target["components"]))
        self.assertEqual(
            tuple(tensor_dict["model.layers.3.mlp.experts.w13_weight.w1.lora_A"].shape),
            (2, 1, 4),
        )
        self.assertEqual(
            tuple(tensor_dict["model.layers.3.mlp.experts.w13_weight.w3.lora_B"].shape),
            (2, 8, 1),
        )

        w2_target = targets["model.layers.3.mlp.experts.w2_weight"]
        self.assertEqual(w2_target["components"][0]["shard_id"], "w2")
        self.assertTrue(w2_target["components"][0]["fused_experts"])
        self.assertEqual(
            tuple(tensor_dict["model.layers.3.mlp.experts.w2_weight.w2.lora_A"].shape),
            (2, 1, 4),
        )


if __name__ == "__main__":
    unittest.main()
