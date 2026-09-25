"""Live derived-weight refresh must be visible to an existing CUDA graph."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.fused_kv_materialize import FusedKVMaterializeHelper
from sglang.srt.models.dflash import CandidateSelector
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDFlashWeightRefresh(unittest.TestCase):
    @torch.no_grad()
    def test_materialize_replays_new_weights_without_recapture(self):
        torch.manual_seed(17)
        device, dtype = torch.device("cuda:0"), torch.bfloat16
        n_layers, hidden, heads, dim, tokens = 2, 64, 2, 16, 31
        phase = torch.outer(
            torch.arange(128, device=device).float(),
            10000 ** (-torch.arange(0, dim, 2, device=device).float() / dim),
        )
        rotary = SimpleNamespace(
            rotary_dim=dim,
            is_neox_style=True,
            cos_sin_cache=torch.cat((phase.cos(), phase.sin()), dim=-1).to(dtype),
        )
        layers = []
        for _ in range(n_layers):
            layers.append(
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        num_kv_heads=heads,
                        head_dim=dim,
                        q_size=heads * dim,
                        kv_size=heads * dim,
                        rotary_emb=rotary,
                        qkv_proj=SimpleNamespace(
                            weight=torch.randn(
                                3 * heads * dim, hidden, device=device, dtype=dtype
                            )
                        ),
                        k_norm=SimpleNamespace(
                            weight=torch.randn(dim, device=device, dtype=dtype),
                            variance_epsilon=1e-6,
                        ),
                    )
                )
            )
        helper = FusedKVMaterializeHelper(layers, rotary, heads, dim, device, 128)
        context = torch.randn(tokens, hidden, device=device, dtype=dtype)
        positions = torch.arange(tokens, device=device)
        output_k = torch.empty(n_layers, tokens, heads, dim, device=device, dtype=dtype)
        output_v = torch.empty_like(output_k)

        def write(index, key, value):
            output_k[index].copy_(key)
            output_v[index].copy_(value)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                helper.materialize(context, positions, write)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            helper.materialize(context, positions, write)
        graph.replay()
        before_k, before_v = output_k.clone(), output_v.clone()
        addresses = helper.flat_kv_weight_t.data_ptr(), helper.k_norm_weights.data_ptr()
        for layer in layers:
            layer.self_attn.qkv_proj.weight.mul_(0.5)
            layer.self_attn.k_norm.weight.add_(0.25)
        helper.refresh_weights(layers)
        graph.replay()
        actual_k, actual_v = output_k.clone(), output_v.clone()
        # A freshly constructed helper is the serving initialization reference.
        reference = FusedKVMaterializeHelper(layers, rotary, heads, dim, device, 128)
        reference.materialize(context, positions, write)
        torch.testing.assert_close(actual_k, output_k, rtol=0, atol=0)
        torch.testing.assert_close(actual_v, output_v, rtol=0, atol=0)
        self.assertFalse(torch.equal(before_k, actual_k))
        self.assertFalse(torch.equal(before_v, actual_v))
        self.assertEqual(
            addresses,
            (helper.flat_kv_weight_t.data_ptr(), helper.k_norm_weights.data_ptr()),
        )

    @torch.no_grad()
    def test_dflash2_selector_graph_reads_updated_parameter_storage(self):
        torch.manual_seed(23)
        selector = CandidateSelector(
            hidden_size=32, vocab_size=128, state_rank=8, top_k=4
        ).to(device="cuda", dtype=torch.bfloat16)
        for parameter in selector.parameters():
            parameter.normal_(std=0.1)
        arguments = dict(
            candidate_ids=torch.randint(0, 128, (2, 3, 4), device="cuda"),
            unary_logits=torch.randn(2, 3, 4, device="cuda"),
            hidden_states=torch.randn(2, 3, 32, device="cuda", dtype=torch.bfloat16),
            anchor_token_ids=torch.tensor([3, 7], device="cuda"),
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                selector.build_lattice(**arguments)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = selector.build_lattice(**arguments)
        graph.replay()
        before = output.clone()
        for parameter in selector.parameters():
            parameter.add_(0.1)
        graph.replay()
        torch.testing.assert_close(
            output, selector.build_lattice(**arguments), rtol=0, atol=0
        )
        self.assertFalse(torch.equal(before, output))


if __name__ == "__main__":
    unittest.main()
