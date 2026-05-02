import unittest
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
import sys

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.model_loader.lora_merge_loader import (
    merge_lora_tensors_inplace,
    prepare_lora_tensors_for_merge,
)

register_cuda_ci(est_time=20, suite="stage-b-test-1-gpu-small")


class PackedLinear(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        *,
        fill_value=0.0,
        dtype=torch.float32,
    ):
        super().__init__()
        self.output_sizes = list(output_sizes)
        self.weight = torch.nn.Parameter(
            torch.full(
                (sum(output_sizes), input_size),
                fill_value,
                dtype=dtype,
            ),
            requires_grad=False,
        )
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        if isinstance(loaded_shard_id, tuple):
            offset = 0
            for shard_id in loaded_shard_id:
                shard_size = self.output_sizes[shard_id]
                shard = loaded_weight[offset : offset + shard_size]
                self.weight_loader(param, shard, shard_id)
                offset += shard_size
            return

        if loaded_shard_id is None:
            param.data.copy_(loaded_weight.to(param.dtype))
            return

        shard_map = {"q": 0, "k": 1, "v": 2}
        shard_idx = shard_map.get(loaded_shard_id, loaded_shard_id)
        start = sum(self.output_sizes[:shard_idx])
        end = start + self.output_sizes[shard_idx]
        param.data[start:end].copy_(loaded_weight.to(param.dtype))


class DummyExperts(torch.nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(
            torch.zeros(num_experts, intermediate_size * 2, hidden_size),
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.zeros(num_experts, hidden_size, intermediate_size),
            requires_grad=False,
        )
        setattr(self.w13_weight, "weight_loader", self.w13_weight_loader)
        setattr(self.w2_weight, "weight_loader", self.w2_weight_loader)

    def w13_weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
        half = param.shape[1] // 2
        if shard_id == "w1":
            param.data[expert_id, :half].copy_(loaded_weight.to(param.dtype))
        elif shard_id == "w3":
            param.data[expert_id, half:].copy_(loaded_weight.to(param.dtype))
        else:
            raise ValueError(f"Unsupported shard_id {shard_id}")

    def w2_weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
        if shard_id != "w2":
            raise ValueError(f"Unsupported shard_id {shard_id}")
        param.data[expert_id].copy_(loaded_weight.to(param.dtype))


class FlashinferTRTLLMExperts(torch.nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size, w13_weight, w2_weight):
        super().__init__()
        self.num_local_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size
        self.moe_runner_config = SimpleNamespace(is_gated=True)
        self.quant_method = UnquantizedFusedMoEMethod(use_flashinfer_trtllm_moe=True)
        self.w13_weight = torch.nn.Parameter(w13_weight.clone(), requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2_weight.clone(), requires_grad=False)
        setattr(self.w13_weight, "weight_loader", self.weight_loader)
        setattr(self.w2_weight, "weight_loader", self.weight_loader)

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
        self.quant_method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
            self,
            param,
            weight_name,
        )
        if param is self.w13_weight:
            half = self.intermediate_size_per_partition
            if shard_id == "w1":
                param.data[expert_id, :half].copy_(loaded_weight.to(param.dtype))
            elif shard_id == "w3":
                param.data[expert_id, half:].copy_(loaded_weight.to(param.dtype))
            else:
                raise ValueError(f"Unsupported shard_id {shard_id}")
            return
        if param is self.w2_weight:
            if shard_id != "w2":
                raise ValueError(f"Unsupported shard_id {shard_id}")
            param.data[expert_id].copy_(loaded_weight.to(param.dtype))
            return
        raise ValueError("Unexpected parameter passed to FlashinferTRTLLMExperts.weight_loader")


class SharedExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = PackedLinear(3, [2, 2])
        self.down_proj = PackedLinear(2, [3])


class DenseHead(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, *, fill_value=0.0, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.full((vocab_size, hidden_size), fill_value, dtype=dtype),
            requires_grad=False,
        )


class DummyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = PackedLinear(3, [2, 2, 2])
        self.o_proj = PackedLinear(6, [3])
        self.linear_attn = torch.nn.Module()
        self.linear_attn.in_proj_qkvz = PackedLinear(3, [1, 1, 2, 2])
        self.linear_attn.out_proj = PackedLinear(2, [3])
        self.mlp = torch.nn.Module()
        self.mlp.shared_expert = SharedExpert()
        self.mlp.experts = DummyExperts(num_experts=2, hidden_size=3, intermediate_size=2)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([DummyLayer()])
        self.lm_head = DenseHead(vocab_size=5, hidden_size=3)


class LargeDirectModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.large0 = DenseHead(vocab_size=512, hidden_size=512)
        self.large1 = DenseHead(vocab_size=512, hidden_size=512)


class WeightLoaderProbe(torch.nn.Module):
    def __init__(self, fill_value=0.0, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.full((1, 1), fill_value, dtype=dtype), requires_grad=False
        )
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(self, param, loaded_weight):
        param.copy_(loaded_weight.to(dtype=param.dtype))


class TiedEmbeddingModel(torch.nn.Module):
    def __init__(self, *, tie_word_embeddings: bool, fill_value: float = 1000.0):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=tie_word_embeddings)
        self.model = torch.nn.Module()
        self.model.embed_tokens = DenseHead(
            vocab_size=1,
            hidden_size=1,
            fill_value=fill_value,
            dtype=torch.bfloat16,
        )
        self.lm_head = DenseHead(
            vocab_size=1,
            hidden_size=1,
            fill_value=fill_value,
            dtype=torch.bfloat16,
        )


class WeightLoaderPrecisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = WeightLoaderProbe(fill_value=1000.0, dtype=torch.bfloat16)


class PackedWeightPrecisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = torch.nn.Module()
        self.probe.gate_up_proj = PackedLinear(
            1,
            [1, 1],
            fill_value=1000.0,
            dtype=torch.bfloat16,
        )


class FlashinferDummyLayer(torch.nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size, w13_weight, w2_weight):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.experts = FlashinferTRTLLMExperts(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
        )


class FlashinferDummyModel(torch.nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size, w13_weight, w2_weight):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [
                FlashinferDummyLayer(
                    num_experts=num_experts,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    w13_weight=w13_weight,
                    w2_weight=w2_weight,
                )
            ]
        )


@contextmanager
def stub_flashinfer_fused_moe_core():
    saved = {
        name: sys.modules.get(name)
        for name in ("flashinfer", "flashinfer.fused_moe", "flashinfer.fused_moe.core")
    }

    def get_reorder_rows_for_gated_act_gemm_row_indices(x):
        assert x.dim() == 2
        m = x.shape[0]
        assert m % 2 == 0
        row_indices = torch.arange(m, dtype=torch.long, device=x.device)
        top = row_indices[: m // 2]
        bot = row_indices[m // 2 :]
        permuted = torch.empty_like(row_indices)
        permuted[0::2] = top
        permuted[1::2] = bot
        return permuted

    def get_shuffle_matrix_a_row_indices(input_tensor, epilogue_tile_m):
        del epilogue_tile_m
        return torch.arange(
            input_tensor.shape[0] - 1,
            -1,
            -1,
            dtype=torch.long,
            device=input_tensor.device,
        )

    def _maybe_get_cached_w3_w1_permute_indices(
        cache,
        dst_w3_w1_weight,
        epilogue_tile_m,
        num_elts_per_sf=None,
        is_gated_act_gemm=True,
    ):
        del num_elts_per_sf
        cache_key = ("w3_w1", tuple(dst_w3_w1_weight.shape))
        if cache_key not in cache:
            permute0 = (
                get_reorder_rows_for_gated_act_gemm_row_indices(dst_w3_w1_weight)
                if is_gated_act_gemm
                else torch.arange(
                    dst_w3_w1_weight.shape[0],
                    dtype=torch.long,
                    device=dst_w3_w1_weight.device,
                )
            )
            permute1 = get_shuffle_matrix_a_row_indices(
                dst_w3_w1_weight, epilogue_tile_m
            )
            cache[cache_key] = permute0[permute1]
        return cache[cache_key]

    def get_w2_permute_indices_with_cache(
        cache,
        dst_w2_weight,
        epilogue_tile_m,
        num_elts_per_sf=None,
    ):
        del num_elts_per_sf
        cache_key = ("w2", tuple(dst_w2_weight.shape))
        if cache_key not in cache:
            cache[cache_key] = get_shuffle_matrix_a_row_indices(
                dst_w2_weight, epilogue_tile_m
            )
        return cache[cache_key]

    def convert_to_block_layout(input_tensor, block_k):
        m, k = input_tensor.shape
        assert k % block_k == 0
        return (
            input_tensor.view(m, k // block_k, block_k)
            .permute(1, 0, 2)
            .contiguous()
        )

    flashinfer_mod = ModuleType("flashinfer")
    flashinfer_fused_moe_mod = ModuleType("flashinfer.fused_moe")
    flashinfer_core_mod = ModuleType("flashinfer.fused_moe.core")
    flashinfer_core_mod._maybe_get_cached_w3_w1_permute_indices = (
        _maybe_get_cached_w3_w1_permute_indices
    )
    flashinfer_core_mod.get_w2_permute_indices_with_cache = (
        get_w2_permute_indices_with_cache
    )
    flashinfer_core_mod.convert_to_block_layout = convert_to_block_layout
    flashinfer_core_mod.get_reorder_rows_for_gated_act_gemm_row_indices = (
        get_reorder_rows_for_gated_act_gemm_row_indices
    )
    flashinfer_fused_moe_mod.core = flashinfer_core_mod
    flashinfer_mod.fused_moe = flashinfer_fused_moe_mod

    sys.modules["flashinfer"] = flashinfer_mod
    sys.modules["flashinfer.fused_moe"] = flashinfer_fused_moe_mod
    sys.modules["flashinfer.fused_moe.core"] = flashinfer_core_mod
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestLoRAMergeLoader(unittest.TestCase):
    def _manifest(self):
        return {"config_dict": {"lora_alpha": 2, "r": 1}}

    def _unit_manifest(self):
        return {"config_dict": {"lora_alpha": 1, "r": 1}}

    def _merge_inplace(self, model, named_tensors, manifest=None):
        model.cuda()
        merge_lora_tensors_inplace(
            model,
            named_tensors,
            load_context={"manifest": manifest or self._manifest()},
        )
        torch.cuda.synchronize()
        model.cpu()
        return model

    def _flashinfer_trtllm_merge_fixture(self):
        num_experts = 2
        hidden_size = 64
        intermediate_size = 64

        w13_init = (
            (
                torch.arange(
                    num_experts * intermediate_size * 2 * hidden_size,
                    dtype=torch.float32,
                )
                % 211
            )
            .reshape(num_experts, intermediate_size * 2, hidden_size)
            .sub(100)
            .div(17)
            .to(torch.bfloat16)
        )
        w2_init = (
            (
                torch.arange(
                    num_experts * hidden_size * intermediate_size, dtype=torch.float32
                )
                % 157
            )
            .reshape(num_experts, hidden_size, intermediate_size)
            .sub(70)
            .div(13)
            .to(torch.bfloat16)
        )

        w1_a = torch.zeros(1, 1, hidden_size)
        w1_a[0, 0, 0] = 1.0
        w1_a[0, 0, 3] = -0.5
        w1_b = torch.zeros(num_experts, intermediate_size, 1)
        w1_b[0, 0, 0] = 1.0
        w1_b[0, 7, 0] = -2.0
        w1_b[1, 5, 0] = 3.0
        w1_b[1, 9, 0] = 1.5

        w3_a = torch.zeros(1, 1, hidden_size)
        w3_a[0, 0, 2] = 0.75
        w3_a[0, 0, 4] = -1.0
        w3_b = torch.zeros(num_experts, intermediate_size, 1)
        w3_b[0, 1, 0] = -1.25
        w3_b[0, 3, 0] = 0.5
        w3_b[1, 2, 0] = 2.0
        w3_b[1, 8, 0] = -0.25

        w2_a = torch.zeros(num_experts, 1, intermediate_size)
        w2_a[0, 0, 0] = 1.0
        w2_a[0, 0, 6] = -0.25
        w2_a[1, 0, 2] = -1.5
        w2_a[1, 0, 9] = 0.75
        w2_b = torch.zeros(1, hidden_size, 1)
        w2_b[0, 0, 0] = 0.5
        w2_b[0, 4, 0] = -1.0
        w2_b[0, 11, 0] = 1.25

        named_tensors_by_target = {
            "w1": [
                (
                    "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight",
                    w1_a,
                ),
                (
                    "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight",
                    w1_b,
                ),
            ],
            "w3": [
                (
                    "base_model.model.model.layers.0.mlp.experts.w3.lora_A.weight",
                    w3_a,
                ),
                (
                    "base_model.model.model.layers.0.mlp.experts.w3.lora_B.weight",
                    w3_b,
                ),
            ],
            "w2": [
                (
                    "base_model.model.model.layers.0.mlp.experts.w2.lora_A.weight",
                    w2_a,
                ),
                (
                    "base_model.model.model.layers.0.mlp.experts.w2.lora_B.weight",
                    w2_b,
                ),
            ],
        }
        return {
            "num_experts": num_experts,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "w13_init": w13_init,
            "w2_init": w2_init,
            "w1_a": w1_a,
            "w1_b": w1_b,
            "w3_a": w3_a,
            "w3_b": w3_b,
            "w2_a": w2_a,
            "w2_b": w2_b,
            "named_tensors_by_target": named_tensors_by_target,
        }

    def _apply_flashinfer_expected_deltas(self, experts, fixture, targets):
        scaling = (
            self._manifest()["config_dict"]["lora_alpha"]
            / self._manifest()["config_dict"]["r"]
        )
        intermediate_size = fixture["intermediate_size"]
        for expert_id in range(fixture["num_experts"]):
            if "w1" in targets:
                w1_delta = (
                    fixture["w1_b"][expert_id].float()
                    @ fixture["w1_a"][0].float()
                ).mul_(scaling).to(torch.bfloat16)
                experts.w13_weight.data[expert_id, :intermediate_size].add_(w1_delta)
            if "w3" in targets:
                w3_delta = (
                    fixture["w3_b"][expert_id].float()
                    @ fixture["w3_a"][0].float()
                ).mul_(scaling).to(torch.bfloat16)
                experts.w13_weight.data[expert_id, intermediate_size:].add_(w3_delta)
            if "w2" in targets:
                w2_delta = (
                    fixture["w2_b"][0].float()
                    @ fixture["w2_a"][expert_id].float()
                ).mul_(scaling).to(torch.bfloat16)
                experts.w2_weight.data[expert_id].add_(w2_delta)

    def _assert_flashinfer_experts_equal(self, actual_experts, expected_experts):
        self.assertEqual(
            tuple(actual_experts.w13_weight.shape),
            tuple(expected_experts.w13_weight.shape),
        )
        self.assertEqual(
            tuple(actual_experts.w2_weight.shape),
            tuple(expected_experts.w2_weight.shape),
        )
        self.assertTrue(torch.equal(actual_experts.w13_weight, expected_experts.w13_weight))
        self.assertTrue(torch.equal(actual_experts.w2_weight, expected_experts.w2_weight))

    def test_merge_self_attention_targets(self):
        model = DummyModel()
        named_tensors = [
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                torch.tensor([[1.0, 0.0, 1.0]]),
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                torch.tensor([[1.0], [2.0]]),
            ),
            (
                "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight",
                torch.tensor([[0.0, 1.0, 1.0]]),
            ),
            (
                "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight",
                torch.tensor([[3.0], [4.0]]),
            ),
            (
                "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight",
                torch.tensor([[1.0, 2.0, 0.0, 1.0, 0.0, 1.0]]),
            ),
            (
                "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight",
                torch.tensor([[1.0], [0.0], [2.0]]),
            ),
        ]

        self._merge_inplace(model, named_tensors, self._manifest())

        q_delta = 2 * torch.tensor([[1.0, 0.0, 1.0], [2.0, 0.0, 2.0]])
        v_delta = 2 * torch.tensor([[0.0, 3.0, 3.0], [0.0, 4.0, 4.0]])
        expected_qkv = torch.cat([q_delta, torch.zeros_like(q_delta), v_delta], dim=0)
        self.assertTrue(
            torch.equal(model.model.layers[0].qkv_proj.weight, expected_qkv)
        )

        expected_o = 2 * torch.tensor(
            [[1.0, 2.0, 0.0, 1.0, 0.0, 1.0], [0.0] * 6, [2.0, 4.0, 0.0, 2.0, 0.0, 2.0]]
        )
        self.assertTrue(torch.equal(model.model.layers[0].o_proj.weight, expected_o))

    def test_merge_qwen35_linear_attention_targets(self):
        model = DummyModel()
        named_tensors = [
            (
                "base_model.model.model.layers.0.linear_attn.in_proj_q.lora_A.weight",
                torch.tensor([[1.0, 2.0, 0.0]]),
            ),
            (
                "base_model.model.model.layers.0.linear_attn.in_proj_q.lora_B.weight",
                torch.tensor([[1.0]]),
            ),
            (
                "base_model.model.model.layers.0.linear_attn.in_proj_k.lora_A.weight",
                torch.tensor([[0.0, 1.0, 1.0]]),
            ),
            (
                "base_model.model.model.layers.0.linear_attn.in_proj_k.lora_B.weight",
                torch.tensor([[2.0]]),
            ),
            (
                "base_model.model.model.layers.0.linear_attn.in_proj_v.lora_A.weight",
                torch.tensor([[1.0, 0.0, 1.0]]),
            ),
            (
                "base_model.model.model.layers.0.linear_attn.in_proj_v.lora_B.weight",
                torch.tensor([[3.0], [4.0]]),
            ),
            (
                "base_model.model.model.layers.0.linear_attn.in_proj_z.lora_A.weight",
                torch.tensor([[0.0, 1.0, 2.0]]),
            ),
            (
                "base_model.model.model.layers.0.linear_attn.in_proj_z.lora_B.weight",
                torch.tensor([[5.0], [6.0]]),
            ),
            (
                "base_model.model.model.layers.0.linear_attn.out_proj.lora_A.weight",
                torch.tensor([[1.0, 0.0]]),
            ),
            (
                "base_model.model.model.layers.0.linear_attn.out_proj.lora_B.weight",
                torch.tensor([[1.0], [2.0], [3.0]]),
            ),
        ]

        self._merge_inplace(model, named_tensors, self._manifest())

        expected_qkvz = 2 * torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [0.0, 2.0, 2.0],
                [3.0, 0.0, 3.0],
                [4.0, 0.0, 4.0],
                [0.0, 5.0, 10.0],
                [0.0, 6.0, 12.0],
            ]
        )
        self.assertTrue(
            torch.equal(model.model.layers[0].linear_attn.in_proj_qkvz.weight, expected_qkvz)
        )

        expected_out = 2 * torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        self.assertTrue(
            torch.equal(model.model.layers[0].linear_attn.out_proj.weight, expected_out)
        )

    def test_merge_shared_expert_and_lm_head_targets(self):
        model = DummyModel()
        named_tensors = [
            (
                "base_model.model.model.layers.0.mlp.shared_expert.gate_proj.lora_A.weight",
                torch.tensor([[1.0, 0.0, 1.0]]),
            ),
            (
                "base_model.model.model.layers.0.mlp.shared_expert.gate_proj.lora_B.weight",
                torch.tensor([[1.0], [2.0]]),
            ),
            (
                "base_model.model.model.layers.0.mlp.shared_expert.up_proj.lora_A.weight",
                torch.tensor([[0.0, 1.0, 1.0]]),
            ),
            (
                "base_model.model.model.layers.0.mlp.shared_expert.up_proj.lora_B.weight",
                torch.tensor([[3.0], [4.0]]),
            ),
            (
                "base_model.model.model.layers.0.mlp.shared_expert.down_proj.lora_A.weight",
                torch.tensor([[1.0, 2.0]]),
            ),
            (
                "base_model.model.model.layers.0.mlp.shared_expert.down_proj.lora_B.weight",
                torch.tensor([[1.0], [0.0], [2.0]]),
            ),
            (
                "base_model.model.model.unembed_tokens.lora_A.weight",
                torch.tensor([[1.0, 0.0, 1.0]]),
            ),
            (
                "base_model.model.model.unembed_tokens.lora_B.weight",
                torch.tensor([[1.0], [0.0], [2.0], [0.0], [3.0]]),
            ),
        ]

        self._merge_inplace(model, named_tensors, self._manifest())

        expected_gate_up = 2 * torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 2.0],
                [0.0, 3.0, 3.0],
                [0.0, 4.0, 4.0],
            ]
        )
        self.assertTrue(
            torch.equal(model.model.layers[0].mlp.shared_expert.gate_up_proj.weight, expected_gate_up)
        )

        expected_down = 2 * torch.tensor([[1.0, 2.0], [0.0, 0.0], [2.0, 4.0]])
        self.assertTrue(
            torch.equal(model.model.layers[0].mlp.shared_expert.down_proj.weight, expected_down)
        )

        expected_lm_head = 2 * torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 3.0],
            ]
        )
        self.assertTrue(torch.equal(model.lm_head.weight, expected_lm_head))

    def test_unembed_tokens_updates_embed_and_lm_head_when_tied(self):
        model = TiedEmbeddingModel(tie_word_embeddings=True)
        named_tensors = [
            (
                "base_model.model.model.unembed_tokens.lora_A.weight",
                torch.tensor([[1.0]]),
            ),
            (
                "base_model.model.model.unembed_tokens.lora_B.weight",
                torch.tensor([[-999.6]]),
            ),
        ]

        self._merge_inplace(model, named_tensors, self._unit_manifest())

        expected = torch.tensor([[0.400390625]], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(model.model.embed_tokens.weight, expected))
        self.assertTrue(torch.equal(model.lm_head.weight, expected))

    def test_unembed_tokens_updates_only_lm_head_when_untied(self):
        model = TiedEmbeddingModel(tie_word_embeddings=False, fill_value=0.0)
        named_tensors = [
            (
                "base_model.model.model.unembed_tokens.lora_A.weight",
                torch.tensor([[1.0]]),
            ),
            (
                "base_model.model.model.unembed_tokens.lora_B.weight",
                torch.tensor([[2.0]]),
            ),
        ]

        self._merge_inplace(model, named_tensors, self._unit_manifest())

        self.assertTrue(
            torch.equal(
                model.model.embed_tokens.weight,
                torch.tensor([[0.0]], dtype=torch.bfloat16),
            )
        )
        self.assertTrue(
            torch.equal(
                model.lm_head.weight,
                torch.tensor([[2.0]], dtype=torch.bfloat16),
            )
        )

    def test_weight_loader_addition_accumulates_in_fp32(self):
        model = WeightLoaderPrecisionModel()
        named_tensors = [
            (
                "base_model.model.probe.lora_A.weight",
                torch.tensor([[1.0]]),
            ),
            (
                "base_model.model.probe.lora_B.weight",
                torch.tensor([[-999.6]]),
            ),
        ]

        self._merge_inplace(model, named_tensors, self._unit_manifest())

        expected = torch.tensor([[0.400390625]], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(model.probe.weight, expected))

    def test_packed_target_addition_accumulates_in_fp32(self):
        model = PackedWeightPrecisionModel()
        named_tensors = [
            (
                "base_model.model.probe.gate_proj.lora_A.weight",
                torch.tensor([[1.0]]),
            ),
            (
                "base_model.model.probe.gate_proj.lora_B.weight",
                torch.tensor([[-999.6]]),
            ),
        ]

        self._merge_inplace(model, named_tensors, self._unit_manifest())

        expected = torch.tensor(
            [[0.400390625], [1000.0]],
            dtype=torch.bfloat16,
        )
        self.assertTrue(torch.equal(model.probe.gate_up_proj.weight, expected))

    def test_merge_moe_expert_targets(self):
        model = DummyModel()
        named_tensors = [
            (
                "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight",
                torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]),
            ),
            (
                "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight",
                torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
            ),
            (
                "base_model.model.model.layers.0.mlp.experts.w3.lora_A.weight",
                torch.tensor([[[0.0, 0.0, 1.0]]]),
            ),
            (
                "base_model.model.model.layers.0.mlp.experts.w3.lora_B.weight",
                torch.tensor([[[5.0], [6.0]], [[7.0], [8.0]]]),
            ),
            (
                "base_model.model.model.layers.0.mlp.experts.w2.lora_A.weight",
                torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
            ),
            (
                "base_model.model.model.layers.0.mlp.experts.w2.lora_B.weight",
                torch.tensor([[[1.0], [2.0], [3.0]]]),
            ),
        ]

        self._merge_inplace(model, named_tensors, self._manifest())

        expected_w13 = 2 * torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0],
                    [0.0, 0.0, 6.0],
                ],
                [
                    [0.0, 3.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 0.0, 7.0],
                    [0.0, 0.0, 8.0],
                ],
            ]
        )
        self.assertTrue(
            torch.equal(model.model.layers[0].mlp.experts.w13_weight, expected_w13)
        )

        expected_w2 = 2 * torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
            ]
        )
        self.assertTrue(
            torch.equal(model.model.layers[0].mlp.experts.w2_weight, expected_w2)
        )

    def test_flashinfer_trtllm_restore_roundtrip_for_merged_moe_update(self):
        fixture = self._flashinfer_trtllm_merge_fixture()
        actual_model = FlashinferDummyModel(
            num_experts=fixture["num_experts"],
            hidden_size=fixture["hidden_size"],
            intermediate_size=fixture["intermediate_size"],
            w13_weight=fixture["w13_init"],
            w2_weight=fixture["w2_init"],
        )
        expected_model = FlashinferDummyModel(
            num_experts=fixture["num_experts"],
            hidden_size=fixture["hidden_size"],
            intermediate_size=fixture["intermediate_size"],
            w13_weight=fixture["w13_init"],
            w2_weight=fixture["w2_init"],
        )
        actual_model.cuda()
        actual_experts = actual_model.model.layers[0].mlp.experts
        expected_experts = expected_model.model.layers[0].mlp.experts

        self._apply_flashinfer_expected_deltas(expected_experts, fixture, {"w1", "w2", "w3"})

        with stub_flashinfer_fused_moe_core():
            actual_experts.quant_method.process_weights_after_loading(actual_experts)
            expected_experts.quant_method.process_weights_after_loading(expected_experts)
            self._merge_inplace(
                actual_model,
                [
                    tensor
                    for target in ("w1", "w3", "w2")
                    for tensor in fixture["named_tensors_by_target"][target]
                ],
                self._manifest(),
            )

        self._assert_flashinfer_experts_equal(actual_experts, expected_experts)

    def test_flashinfer_trtllm_partial_routed_merge_roundtrip_for_merged_moe_update(self):
        fixture = self._flashinfer_trtllm_merge_fixture()

        for target in ("w1", "w2", "w3"):
            with self.subTest(target=target):
                actual_model = FlashinferDummyModel(
                    num_experts=fixture["num_experts"],
                    hidden_size=fixture["hidden_size"],
                    intermediate_size=fixture["intermediate_size"],
                    w13_weight=fixture["w13_init"],
                    w2_weight=fixture["w2_init"],
                )
                expected_model = FlashinferDummyModel(
                    num_experts=fixture["num_experts"],
                    hidden_size=fixture["hidden_size"],
                    intermediate_size=fixture["intermediate_size"],
                    w13_weight=fixture["w13_init"],
                    w2_weight=fixture["w2_init"],
                )
                actual_model.cuda()
                actual_experts = actual_model.model.layers[0].mlp.experts
                expected_experts = expected_model.model.layers[0].mlp.experts

                self._apply_flashinfer_expected_deltas(expected_experts, fixture, {target})

                with stub_flashinfer_fused_moe_core():
                    actual_experts.quant_method.process_weights_after_loading(
                        actual_experts
                    )
                    expected_experts.quant_method.process_weights_after_loading(
                        expected_experts
                    )
                    self._merge_inplace(
                        actual_model,
                        fixture["named_tensors_by_target"][target],
                        self._manifest(),
                    )

                self._assert_flashinfer_experts_equal(actual_experts, expected_experts)

    def test_prestage_budget_stages_small_pairs_and_consumes_empty_update(self):
        model = DummyModel().cuda()
        named_tensors = [
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                torch.tensor([[1.0, 0.0, 1.0]]),
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                torch.tensor([[1.0], [2.0]]),
            ),
        ]
        request_id = "small-prestage"
        prepare_trace = {"request_id": request_id}
        prepare_manifest = {
            **self._manifest(),
            "lora_merge_prestage_request_id": request_id,
            "peak_device_bytes": "8MB",
        }
        prepare_lora_tensors_for_merge(
            model,
            named_tensors,
            load_context={"manifest": prepare_manifest, "trace": prepare_trace},
        )
        self.assertTrue(prepare_trace["lora_prestage_complete"])
        self.assertEqual(prepare_trace["lora_prestage_staged_pair_count"], 1)
        self.assertEqual(prepare_trace["lora_prestage_unstaged_pair_count"], 0)

        merge_trace = {"request_id": request_id}
        merge_lora_tensors_inplace(
            model,
            [],
            load_context={
                "manifest": {
                    **self._manifest(),
                    "lora_merge_consume_prestaged": True,
                    "lora_merge_prestage_request_id": request_id,
                },
                "trace": merge_trace,
            },
        )
        torch.cuda.synchronize()
        model.cpu()

        q_delta = 2 * torch.tensor([[1.0, 0.0, 1.0], [2.0, 0.0, 2.0]])
        expected_qkv = torch.cat([q_delta, torch.zeros_like(q_delta), torch.zeros_like(q_delta)], dim=0)
        self.assertTrue(
            torch.equal(model.model.layers[0].qkv_proj.weight, expected_qkv)
        )
        self.assertEqual(merge_trace["lora_loader_prestage_hit_count"], 1)
        self.assertEqual(merge_trace.get("lora_loader_prestage_miss_count", 0), 0)

    def test_prestage_budget_keeps_oversized_pairs_on_cpu_and_applies_from_cache(self):
        model = LargeDirectModel().cuda()
        a = torch.ones(1, 512)
        b = torch.ones(512, 1)
        named_tensors = [
            ("base_model.model.large0.lora_A.weight", a),
            ("base_model.model.large0.lora_B.weight", b),
            ("base_model.model.large1.lora_A.weight", a),
            ("base_model.model.large1.lora_B.weight", b),
        ]
        request_id = "partial-prestage"
        prepare_trace = {"request_id": request_id}
        prepare_manifest = {
            **self._unit_manifest(),
            "lora_merge_prestage_request_id": request_id,
            "peak_device_bytes": "1MB",
        }
        prepare_lora_tensors_for_merge(
            model,
            named_tensors,
            load_context={"manifest": prepare_manifest, "trace": prepare_trace},
        )
        self.assertFalse(prepare_trace["lora_prestage_complete"])
        self.assertEqual(prepare_trace["lora_prestage_staged_pair_count"], 0)
        self.assertEqual(prepare_trace["lora_prestage_unstaged_pair_count"], 2)
        self.assertGreater(
            prepare_trace["lora_prestage_max_apply_temp_bytes"],
            prepare_trace["lora_prestage_peak_device_budget_bytes"],
        )

        merge_trace = {"request_id": request_id}
        merge_lora_tensors_inplace(
            model,
            [],
            load_context={
                "manifest": {
                    **self._unit_manifest(),
                    "lora_merge_consume_prestaged": True,
                    "lora_merge_prestage_request_id": request_id,
                },
                "trace": merge_trace,
            },
        )
        torch.cuda.synchronize()
        model.cpu()

        expected = torch.ones(512, 512)
        self.assertTrue(torch.equal(model.large0.weight, expected))
        self.assertTrue(torch.equal(model.large1.weight, expected))
        self.assertEqual(merge_trace.get("lora_loader_prestage_hit_count", 0), 0)
        self.assertEqual(merge_trace["lora_loader_prestage_miss_count"], 2)

    def test_cuda_bucketed_core_targets_with_small_bucket(self):
        def named_tensors():
            return [
                (
                    "base_model.model.model.layers.0.linear_attn.in_proj_q.lora_A.weight",
                    torch.tensor([[1.0, 2.0, 0.0]]),
                ),
                (
                    "base_model.model.model.layers.0.linear_attn.in_proj_q.lora_B.weight",
                    torch.tensor([[1.0]]),
                ),
                (
                    "base_model.model.model.layers.0.linear_attn.in_proj_v.lora_A.weight",
                    torch.tensor([[1.0, 0.0, 1.0]]),
                ),
                (
                    "base_model.model.model.layers.0.linear_attn.in_proj_v.lora_B.weight",
                    torch.tensor([[3.0], [4.0]]),
                ),
                (
                    "base_model.model.model.layers.0.mlp.shared_expert.gate_proj.lora_A.weight",
                    torch.tensor([[1.0, 0.0, 1.0]]),
                ),
                (
                    "base_model.model.model.layers.0.mlp.shared_expert.gate_proj.lora_B.weight",
                    torch.tensor([[1.0], [2.0]]),
                ),
                (
                    "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight",
                    torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]),
                ),
                (
                    "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight",
                    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
                ),
                (
                    "base_model.model.model.layers.0.mlp.experts.w2.lora_A.weight",
                    torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
                ),
                (
                    "base_model.model.model.layers.0.mlp.experts.w2.lora_B.weight",
                    torch.tensor([[[1.0], [2.0], [3.0]]]),
                ),
                (
                    "base_model.model.model.unembed_tokens.lora_A.weight",
                    torch.tensor([[1.0, 0.0, 1.0]]),
                ),
                (
                    "base_model.model.model.unembed_tokens.lora_B.weight",
                    torch.tensor([[1.0], [0.0], [2.0], [0.0], [3.0]]),
                ),
            ]

        model = DummyModel()
        manifest = {
            **self._manifest(),
            "gpu_bucket_bytes": 1024 * 1024,
        }

        self._merge_inplace(model, named_tensors(), manifest)

        expected_qkvz = 2 * torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 3.0],
                [4.0, 0.0, 4.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        self.assertTrue(
            torch.equal(
                model.model.layers[0].linear_attn.in_proj_qkvz.weight, expected_qkvz
            )
        )

        expected_gate_up = 2 * torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        self.assertTrue(
            torch.equal(
                model.model.layers[0].mlp.shared_expert.gate_up_proj.weight,
                expected_gate_up,
            )
        )

        expected_w13 = 2 * torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 3.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ]
        )
        self.assertTrue(
            torch.equal(model.model.layers[0].mlp.experts.w13_weight, expected_w13)
        )

        expected_w2 = 2 * torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
            ]
        )
        self.assertTrue(
            torch.equal(model.model.layers[0].mlp.experts.w2_weight, expected_w2)
        )

        expected_lm_head = 2 * torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 3.0],
            ]
        )
        self.assertTrue(torch.equal(model.lm_head.weight, expected_lm_head))


if __name__ == "__main__":
    unittest.main()
