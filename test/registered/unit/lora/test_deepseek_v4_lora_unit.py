"""Unit tests for DeepSeek-V4 LoRA target mapping and adapter normalization.

Usage:
    python -m pytest test/registered/unit/lora/test_deepseek_v4_lora_unit.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.lora import deepseek_v4_targets as v4_lora
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.utils import get_normalized_target_modules, get_target_module_name

PHASE1_TARGETS = ["wq_a", "wkv", "wq_b", "wo_b", "w1", "w2", "w3"]
PHASE1_MODULES = {
    "self_attn.wqkv_a",
    "self_attn.wq_b",
    "self_attn.wo_b",
    "shared_experts.gate_up_proj",
    "shared_experts.down_proj",
}
V4_ALIASES = SimpleNamespace(lora_target_module_aliases=v4_lora.TARGET_MODULE_ALIASES)

V41_DIMS = dict(
    hidden_size=5120,
    q_lora_rank=1280,
    head_dim=512,
    num_attention_heads=64,
    o_groups=8,
    o_lora_rank=1024,
    moe_intermediate_size=2304,
    n_shared_experts=1,
    num_hidden_layers=2,
    vocab_size=129280,
)


def _small_config():
    return SimpleNamespace(
        hidden_size=16,
        q_lora_rank=6,
        head_dim=4,
        num_attention_heads=2,
        o_groups=2,
        o_lora_rank=3,
        moe_intermediate_size=5,
        n_shared_experts=1,
        num_hidden_layers=2,
        vocab_size=32,
    )


def _make_v4_model(config):
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

    model = object.__new__(DeepseekV4ForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = config
    model.quant_config = None
    model.num_fused_shared_experts = 0
    return model


class TestTargetNormalization(unittest.TestCase):
    def test_fused_targets(self):
        got = get_normalized_target_modules(PHASE1_TARGETS, V4_ALIASES)
        self.assertEqual(got, PHASE1_MODULES)

    def test_parent_qualified_cli_targets(self):
        from sglang.srt.utils.common import SUPPORTED_LORA_TARGET_MODULES

        cli = [
            "attn.wq_a",
            "attn.wkv",
            "attn.wq_b",
            "attn.wo_b",
            "ffn.shared_experts.w1",
            "ffn.shared_experts.w2",
            "ffn.shared_experts.w3",
        ]
        self.assertTrue(set(cli) <= set(SUPPORTED_LORA_TARGET_MODULES))
        self.assertEqual(get_normalized_target_modules(cli, V4_ALIASES), PHASE1_MODULES)

    def test_other_models_keep_generic_mapping(self):
        self.assertEqual(get_normalized_target_modules(["wq_b"]), {"indexer.wq_b"})
        self.assertEqual(
            get_normalized_target_modules(["gate_proj"], torch.nn.Linear(1, 1)),
            {"gate_up_proj"},
        )

    def test_targets_do_not_match_colliding_modules(self):
        targets = get_normalized_target_modules(PHASE1_TARGETS, V4_ALIASES)
        expected = {
            "model.layers.3.self_attn.wqkv_a": "self_attn.wqkv_a",
            "model.layers.3.self_attn.wq_b": "self_attn.wq_b",
            "model.layers.3.self_attn.wo_b": "self_attn.wo_b",
            "model.layers.3.mlp.shared_experts.gate_up_proj": (
                "shared_experts.gate_up_proj"
            ),
            "model.layers.3.mlp.shared_experts.down_proj": "shared_experts.down_proj",
        }
        for module_name, target in expected.items():
            self.assertEqual(get_target_module_name(module_name, targets), target)
        for module_name in (
            "model.layers.3.self_attn.indexer.wq_b",
            "model.layers.3.self_attn.compressor.wkv",
            "model.layers.0.engram.wkv",
            "model.layers.3.mlp.experts",
        ):
            with self.assertRaises(ValueError):
                get_target_module_name(module_name, targets)


class TestWeightNames(unittest.TestCase):
    def test_checkpoint_names(self):
        sglang_names = (
            "base_model.model.model.layers.3.self_attn.wo_b.lora_A.weight",
            "base_model.model.model.layers.3.mlp.shared_experts.down_proj.lora_B.weight",
        )
        cases = {
            "base_model.model.layers.3.attn.wq_a.lora_A.weight": (
                "base_model.model.layers.3.self_attn.wq_a.lora_A.weight"
            ),
            "base_model.model.layers.3.ffn.shared_experts.w1.lora_B.weight": (
                "base_model.model.layers.3.mlp.shared_experts.gate_proj.lora_B.weight"
            ),
            "base_model.model.layers.3.ffn.shared_experts.w3.lora_A.weight": (
                "base_model.model.layers.3.mlp.shared_experts.up_proj.lora_A.weight"
            ),
            "base_model.model.layers.3.ffn.shared_experts.w2.lora_A.weight": (
                "base_model.model.layers.3.mlp.shared_experts.down_proj.lora_A.weight"
            ),
            **{name: name for name in sglang_names},
        }
        for src, dst in cases.items():
            self.assertEqual(v4_lora.remap_weight_name(src), dst)


class TestFuseWqaWkv(unittest.TestCase):
    prefix = "base_model.model.layers.0.self_attn"

    def _weights(self, r, hidden, q_dim, kv_dim, parts=("wq_a", "wkv")):
        torch.manual_seed(0)
        dims = {"wq_a": q_dim, "wkv": kv_dim}
        w = {}
        for part in parts:
            w[f"{self.prefix}.{part}.lora_A.weight"] = torch.randn(r, hidden)
            w[f"{self.prefix}.{part}.lora_B.weight"] = torch.randn(dims[part], r)
        return w

    def test_stacked_slices_reproduce_each_projection(self):
        r, hidden, q_dim, kv_dim = 4, 16, 6, 3
        w = self._weights(r, hidden, q_dim, kv_dim)
        ref = {k: v.clone() for k, v in w.items()}
        v4_lora.fuse_wqa_wkv_weights(w, q_dim, kv_dim)
        self.assertEqual(
            set(w),
            {
                f"{self.prefix}.wqkv_a.lora_A.weight",
                f"{self.prefix}.wqkv_a.lora_B.weight",
            },
        )
        a = w[f"{self.prefix}.wqkv_a.lora_A.weight"]
        b = w[f"{self.prefix}.wqkv_a.lora_B.weight"]
        self.assertEqual(a.shape, (2 * r, hidden))
        self.assertEqual(b.shape, (q_dim + kv_dim, r))

        x = torch.randn(5, hidden)
        xa = x @ a.T
        out = torch.cat([xa[:, :r] @ b[:q_dim].T, xa[:, r:] @ b[q_dim:].T], dim=1)
        expected = torch.cat(
            [
                x
                @ ref[f"{self.prefix}.wq_a.lora_A.weight"].T
                @ ref[f"{self.prefix}.wq_a.lora_B.weight"].T,
                x
                @ ref[f"{self.prefix}.wkv.lora_A.weight"].T
                @ ref[f"{self.prefix}.wkv.lora_B.weight"].T,
            ],
            dim=1,
        )
        torch.testing.assert_close(out, expected)

    def test_missing_side_is_zero(self):
        r, hidden, q_dim, kv_dim = 4, 16, 6, 3
        w = self._weights(r, hidden, q_dim, kv_dim, parts=("wkv",))
        v4_lora.fuse_wqa_wkv_weights(w, q_dim, kv_dim)
        a = w[f"{self.prefix}.wqkv_a.lora_A.weight"]
        b = w[f"{self.prefix}.wqkv_a.lora_B.weight"]
        self.assertEqual(a.shape, (2 * r, hidden))
        self.assertEqual(b.shape, (q_dim + kv_dim, r))
        self.assertEqual(a[:r].abs().sum().item(), 0)
        self.assertEqual(b[:q_dim].abs().sum().item(), 0)


class TestModelHooks(unittest.TestCase):
    def setUp(self):
        self.model = _make_v4_model(SimpleNamespace(**V41_DIMS))

    def test_buffer_shapes_match_v41(self):
        expected = {
            "self_attn.wqkv_a": ((5120, 1280 + 512), 2),
            "self_attn.wq_b": ((1280, 64 * 512), 1),
            "self_attn.wo_b": ((8 * 1024, 5120), 1),
            "shared_experts.gate_up_proj": ((5120, 2 * 2304), 2),
            "shared_experts.down_proj": ((2304, 5120), 1),
        }
        for name, (dims, stacked) in expected.items():
            self.assertEqual(self.model.get_hidden_dim(name, 0), dims)
            self.assertEqual(self.model.get_stacked_multiply(name), stacked)
        self.assertEqual(self.model.get_stacked_multiply("gate_up_proj_moe"), 2)

    def test_validation(self):
        targets = get_normalized_target_modules(PHASE1_TARGETS, self.model)
        parallel = SimpleNamespace(attn_cp_size=1)
        with (
            mock.patch(
                "sglang.srt.models.deepseek_v4.get_parallel", return_value=parallel
            ),
            envs.SGLANG_OPT_FUSE_WQA_WKV.override(True),
        ):
            self.model.validate_lora_target_modules(targets)
            with self.assertRaisesRegex(ValueError, "does not support target modules"):
                self.model.validate_lora_target_modules(targets | {"indexer.wq_b"})
            with envs.SGLANG_OPT_FUSE_WQA_WKV.override(False):
                with self.assertRaisesRegex(ValueError, "SGLANG_OPT_FUSE_WQA_WKV=1"):
                    self.model.validate_lora_target_modules(targets)
            self.model.num_fused_shared_experts = 1
            with self.assertRaisesRegex(ValueError, "unfused shared experts"):
                self.model.validate_lora_target_modules(targets)
            self.model.num_fused_shared_experts = 0
            parallel.attn_cp_size = 2
            with self.assertRaisesRegex(ValueError, "context parallelism"):
                self.model.validate_lora_target_modules(targets)
            parallel.attn_cp_size = 1
            self.model.quant_config = object()
            with self.assertRaisesRegex(ValueError, "BF16"):
                self.model.validate_lora_target_modules(targets)


class TestFusedQkvAFirstOutputDim(unittest.TestCase):
    """LoRAManager marks where q_a ends in the fused q_a + kv_a LoRA B."""

    def test_v4_and_mla(self):
        for leaf, target in (
            ("wqkv_a", "self_attn.wqkv_a"),
            ("fused_qkv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa"),
        ):
            with self.subTest(leaf):
                attn = torch.nn.Module()
                attn.add_module(leaf, torch.nn.Linear(16, 10, bias=False))
                model = torch.nn.Module()
                model.layers = torch.nn.ModuleList([torch.nn.Module()])
                model.layers[0].self_attn = attn

                manager = LoRAManager.__new__(LoRAManager)
                manager.base_model = model
                manager.base_hf_config = SimpleNamespace(
                    num_hidden_layers=1, q_lora_rank=6
                )
                manager.target_modules = {target}
                with mock.patch.object(
                    manager,
                    "set_lora_module",
                    side_effect=lambda *_: SimpleNamespace(first_output_dim=0),
                ):
                    manager.init_lora_modules()

                lora_module = manager.lora_modules[0][f"layers.0.self_attn.{leaf}"]
                self.assertEqual(lora_module.first_output_dim, 6)


class TestAdapterNormalization(unittest.TestCase):
    """Checkpoint-named PEFT tensors end up in the buffers' module layout."""

    def test_peft_adapter_from_checkpoint_names(self):
        config = _small_config()
        model = _make_v4_model(config)
        r, layer = 2, 1
        prefix = f"base_model.model.layers.{layer}"
        torch.manual_seed(0)
        shapes = {
            "attn.wq_a": (config.q_lora_rank, config.hidden_size),
            "attn.wkv": (config.head_dim, config.hidden_size),
            "attn.wq_b": (
                config.num_attention_heads * config.head_dim,
                config.q_lora_rank,
            ),
            "attn.wo_b": (config.hidden_size, config.o_groups * config.o_lora_rank),
            "ffn.shared_experts.w1": (
                config.moe_intermediate_size,
                config.hidden_size,
            ),
            "ffn.shared_experts.w3": (
                config.moe_intermediate_size,
                config.hidden_size,
            ),
            "ffn.shared_experts.w2": (
                config.hidden_size,
                config.moe_intermediate_size,
            ),
        }
        tensors = {}
        for module, (out_dim, in_dim) in shapes.items():
            tensors[f"{prefix}.{module}.lora_A.weight"] = torch.randn(r, in_dim)
            tensors[f"{prefix}.{module}.lora_B.weight"] = torch.randn(out_dim, r)

        lora_config = LoRAConfig.from_dict(
            {
                "peft_type": "LORA",
                "r": r,
                "lora_alpha": 4,
                "target_modules": PHASE1_TARGETS,
            }
        )
        targets = get_normalized_target_modules(lora_config.target_modules, model)
        with (
            mock.patch(
                "sglang.srt.models.deepseek_v4.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=1),
            ),
            envs.SGLANG_OPT_FUSE_WQA_WKV.override(True),
        ):
            model.validate_lora_target_modules(targets)
            adapter = LoRAAdapter(
                "a", lora_config, config, None, None, base_model=model
            )
            adapter.initialize_weights_from_tensors(tensors)

        names = set(adapter.layers[layer].weights)
        m = f"{prefix}.self_attn"
        s = f"{prefix}.mlp.shared_experts"
        self.assertEqual(
            names,
            {
                f"{m}.wqkv_a.lora_A.weight",
                f"{m}.wqkv_a.lora_B.weight",
                f"{m}.wq_b.lora_A.weight",
                f"{m}.wq_b.lora_B.weight",
                f"{m}.wo_b.lora_A.weight",
                f"{m}.wo_b.lora_B.weight",
                f"{s}.gate_up_proj.lora_A.weight",
                f"{s}.gate_up_proj.lora_B.weight",
                f"{s}.down_proj.lora_A.weight",
                f"{s}.down_proj.lora_B.weight",
            },
        )
        weights = adapter.layers[layer].weights
        self.assertEqual(weights[f"{m}.wqkv_a.lora_A.weight"].shape, (2 * r, 16))
        self.assertEqual(weights[f"{m}.wqkv_a.lora_B.weight"].shape, (6 + 4, r))
        self.assertEqual(weights[f"{s}.gate_up_proj.lora_B.weight"].shape, (2 * 5, r))
        for name in names:
            get_target_module_name(name, targets)
        self.assertEqual(adapter.layers[0].weights, {})


if __name__ == "__main__":
    unittest.main()
