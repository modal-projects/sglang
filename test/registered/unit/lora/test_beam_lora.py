import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.beam import BeamConfig
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.lora.layers import ReplicatedLinearWithLoRA
from sglang.srt.lora.mem_pool import EMPTY_SLOT, LoRAMemoryPool
from sglang.srt.models.beam import BeamModel
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestBeamLoraConstruction(CustomTestCase):
    def test_lora_keeps_wrappable_projections_and_expert_dispatch(self):
        """Packed projections hide adapter targets; deferred sums bypass LoRA.

        Tiny serving fixtures never enter the hidden-size-5120 fusion path.
        """
        config = BeamConfig(
            vocab_size=128,
            hidden_size=5120,
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            layer_types=["full_attention"],
            mlp_layer_types=["sparse"],
        )
        pp_group = SimpleNamespace(
            is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1
        )
        # CUDA metadata suffices for eligibility; no GPU storage or collective.
        hidden = SimpleNamespace(
            is_cuda=True,
            dtype=torch.bfloat16,
            shape=(4, 5120),
            device=torch.device("cuda"),
            is_contiguous=lambda: True,
            data_ptr=lambda: 16,
        )
        for enabled in (False, True):
            with (
                self.subTest(enable_lora=enabled),
                get_context().override_server_args(
                    enable_lora=enabled, disable_custom_all_reduce=False
                ),
                get_parallel().override(
                    tp_size=2,
                    tp_rank=0,
                    attn_tp_size=2,
                    attn_tp_rank=0,
                    attn_cp_size=1,
                    attn_dp_size=1,
                    moe_tp_size=2,
                    moe_ep_size=1,
                    pp_group=pp_group,
                ),
                patch("torch.cuda.is_available", return_value=False),
                patch("torch.cuda.get_device_capability", return_value=(9, 0)),
                patch(
                    "sglang.srt.models.beam.BeamMoE", return_value=torch.nn.Identity()
                ),
                torch.device("meta"),
            ):
                model = BeamModel(config)
                projections = set(dict(model.layers[0].self_attn.named_children()))
                self.assertEqual(
                    projections & {"qkv_gate_proj", "qkv_proj", "gate_proj"},
                    {"qkv_proj", "gate_proj"} if enabled else {"qkv_gate_proj"},
                )
                attention_limit, expert_limit = model._all_reduce_limits(hidden)
                self.assertGreater(attention_limit, 0)
                self.assertEqual(expert_limit > 0, not enabled)
                self.assertTrue(model._can_replace_mlp_all_reduce())


class TestBeamHybridBuffers(CustomTestCase):
    def _make_pool(self, has_dense=True, max_loras_per_batch=1):
        config = SimpleNamespace(
            num_hidden_layers=3,
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=16,
            moe_intermediate_size=8,
            num_experts=2,
        )
        model = torch.nn.Module()
        model.config = config
        model.register_parameter("anchor", torch.nn.Parameter(torch.empty(0)))
        dense = torch.nn.Module()
        dense.gate_up_proj = ReplicatedLinear(8, 32, bias=False)
        dense.down_proj = ReplicatedLinear(16, 8, bias=False)
        experts = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(experts)
        model.layers = torch.nn.ModuleList([dense, experts, deepcopy(dense)])
        if not has_dense:
            model.layers[0] = torch.nn.Module()
            model.layers[2] = torch.nn.Module()
        with get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1):
            return LoRAMemoryPool(
                base_hf_config=config,
                max_loras_per_batch=max_loras_per_batch,
                dtype=torch.float32,
                max_lora_rank=2,
                target_modules={"gate_up_proj", "down_proj"},
                base_model=model,
                eviction_policy="lru",
                lora_added_tokens_size=0,
            )

    def test_hybrid_mlp_allocates_dense_and_expert_buffers(self):
        """A dense prefix needs 3D buffers even without shared experts."""
        for has_dense in (True, False):
            with self.subTest(has_dense=has_dense):
                pool = self._make_pool(has_dense=has_dense)
                for buffer in (pool.A_buffer, pool.B_buffer):
                    for name in ("gate_up_proj", "down_proj"):
                        self.assertEqual(name in buffer, has_dense)
                        if has_dense:
                            self.assertEqual(buffer[name][0].ndim, 3)
                            self.assertIsNone(buffer[name][1])
                            self.assertEqual(buffer[name][2].ndim, 3)
                        self.assertEqual(buffer[name + "_moe"][0].ndim, 4)
                        self.assertEqual(buffer[name + "_moe"][1].ndim, 4)
                        self.assertEqual(buffer[name + "_moe"][2].ndim, 4)

    def _tensors(self, pool):
        for buffer in (pool.A_buffer, pool.B_buffer):
            for layers in buffer.values():
                for tensor in layers:
                    if tensor is not None:
                        yield tensor

    def _prepare(self, pool, uids, adapters=None, modules=None):
        pool.prepare_lora_batch(
            cur_uids=uids,
            lora_adapters=adapters or {},
            lora_modules=modules or [{}, {}, {}],
            lora_refs={},
            lora_embed_tokens_module=None,
            lora_lm_head_module=None,
        )

    def test_hybrid_startup_prepares_base_slot(self):
        """Base-only startup must clear hybrid buffers without touching other slots."""
        pool = self._make_pool(max_loras_per_batch=2)
        for tensor in self._tensors(pool):
            tensor[0].fill_(7)
            tensor[1].fill_(11)

        self._prepare(pool, {None})

        self.assertEqual(pool.uid_to_buffer_id, {None: 0})
        self.assertEqual(pool.buffer_id_to_uid, [None, EMPTY_SLOT])
        for tensor in self._tensors(pool):
            self.assertTrue(torch.all(tensor[0] == 0))
            self.assertTrue(torch.all(tensor[1] == 11))

    def test_hybrid_adapter_removal_and_base_slot_reuse(self):
        """Removing a dense adapter clears hybrid storage for reuse as the base slot."""
        pool = self._make_pool(max_loras_per_batch=2)
        modules = [
            {
                name: ReplicatedLinearWithLoRA(getattr(layer, name), None)
                for name in ("gate_up_proj", "down_proj")
            }
            if layer_id != 1
            else {}
            for layer_id, layer in enumerate(pool.base_model.layers)
        ]
        # A dense-only adapter still uses the hybrid pool's shared slot lifecycle.
        layers = []
        for layer_id in range(pool.num_layer):
            weights = {}
            for name in modules[layer_id]:
                for kind, buffers in (("A", pool.A_buffer), ("B", pool.B_buffer)):
                    weights[f"{name}.lora_{kind}.weight"] = torch.full_like(
                        buffers[name][layer_id][0], 3
                    )
            layers.append(SimpleNamespace(weights=weights, pinned_weights={}))
        adapter = SimpleNamespace(
            config=SimpleNamespace(r=2), layers=layers, embedding_layers={}, scaling=1
        )
        for tensor in self._tensors(pool):
            tensor[0].fill_(7)
        self._prepare(
            pool,
            {"adapter", "keeper"},
            {"adapter": adapter, "keeper": adapter},
            modules,
        )
        for buffer in (pool.A_buffer, pool.B_buffer):
            for name in ("gate_up_proj", "down_proj"):
                for layer_id in (0, 2):
                    self.assertTrue(torch.all(buffer[name][layer_id][0] == 3))
        for tensor in self._tensors(pool):
            tensor[1].fill_(11)

        self.assertEqual(pool.remove_lora("adapter"), 0)
        self.assertEqual(pool.uid_to_buffer_id, {"keeper": 1})
        self.assertEqual(pool.buffer_id_to_uid, [EMPTY_SLOT, "keeper"])
        self.assertNotIn("adapter", pool.eviction_policy.access_order)
        self.assertIsNone(pool.remove_lora("adapter"))
        for tensor in self._tensors(pool):
            self.assertTrue(torch.all(tensor[0] == 0))
            self.assertTrue(torch.all(tensor[1] == 11))

        # Reuse the removed adapter's slot for a base-only batch.
        for tensor in self._tensors(pool):
            tensor[0].fill_(5)
        self._prepare(pool, {None, "keeper"})
        self.assertEqual(pool.uid_to_buffer_id, {None: 0, "keeper": 1})
        self.assertEqual(pool.buffer_id_to_uid, [None, "keeper"])
        for tensor in self._tensors(pool):
            self.assertTrue(torch.all(tensor[0] == 0))
            self.assertTrue(torch.all(tensor[1] == 11))


if __name__ == "__main__":
    unittest.main()
