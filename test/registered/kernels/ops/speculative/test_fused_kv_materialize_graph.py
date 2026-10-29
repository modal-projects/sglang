from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")

"""A CUDA graph captured over FusedKVMaterializeHelper.materialize must not
keep using the helper's growable workspace: a later, larger eager call frees it
and the graph would write into reused memory (the ROCm DFlash draft-KV tail graph
crash)."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.fused_kv_materialize import (
    FusedKVMaterializeHelper,
)
from sglang.test.test_utils import CustomTestCase

N_LAYERS, HIDDEN, KV_HEADS, HEAD_DIM, Q_SIZE = 3, 256, 1, 128, 128


def _make_helper():
    g = torch.Generator(device="cuda").manual_seed(0)
    kv_size = KV_HEADS * HEAD_DIM
    pos = torch.arange(4096, device="cuda", dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2, device="cuda") / HEAD_DIM))
    freqs = pos[:, None] * inv[None, :]
    rotary = SimpleNamespace(
        rotary_dim=HEAD_DIM,
        is_neox_style=True,
        cos_sin_cache=torch.cat([freqs.cos(), freqs.sin()], dim=-1),
    )
    layers = []
    for _ in range(N_LAYERS):
        w = torch.randn(Q_SIZE + 2 * kv_size, HIDDEN, generator=g, device="cuda").to(
            torch.bfloat16
        )
        attn = SimpleNamespace(
            num_kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            rotary_emb=rotary,
            qkv_proj=SimpleNamespace(weight=w),
            q_size=Q_SIZE,
            kv_size=kv_size,
            k_norm=SimpleNamespace(
                weight=torch.rand(HEAD_DIM, generator=g, device="cuda").to(
                    torch.bfloat16
                ),
                variance_epsilon=1e-6,
            ),
        )
        layers.append(SimpleNamespace(self_attn=attn))
    return FusedKVMaterializeHelper(
        layers=layers,
        rotary_emb=rotary,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        device=torch.device("cuda"),
        max_position_hint=4000,
    )


class TestFusedKVMaterializeGraph(CustomTestCase):
    def test_graph_survives_workspace_regrowth(self):
        helper = _make_helper()
        n = 16
        hidden = torch.randn(n, HIDDEN, device="cuda").to(torch.bfloat16)
        positions = torch.randint(0, 4000, (n,), device="cuda")
        out_k = torch.zeros(N_LAYERS, n, KV_HEADS, HEAD_DIM, device="cuda").to(
            torch.bfloat16
        )
        out_v = torch.zeros_like(out_k)

        def write(i, k, v):
            out_k[i].copy_(k)
            out_v[i].copy_(v)

        helper.materialize(hidden, positions, write)  # eager: sizes workspace to n
        ref_k, ref_v = out_k.clone(), out_v.clone()
        # Keep the pre-capture workspace alive so the test can see if the graph
        # still writes to it after a regrowth frees it from the helper.
        old_ws = [helper._proj_workspace, helper._k_workspace, helper._v_workspace]

        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            helper.materialize(hidden, positions, write)

        big = 64 * n
        helper.materialize(
            torch.randn(big, HIDDEN, device="cuda").to(torch.bfloat16),
            torch.randint(0, 4000, (big,), device="cuda"),
            lambda *_: None,
        )
        self.assertNotEqual(helper._proj_workspace.data_ptr(), old_ws[0].data_ptr())
        for t in old_ws:
            t.fill_(7.0)
        out_k.zero_()
        out_v.zero_()
        graph.replay()
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(out_k, ref_k))
        self.assertTrue(torch.equal(out_v, ref_v))
        for t in old_ws:
            self.assertTrue(torch.all(t == 7.0))


if __name__ == "__main__":
    unittest.main()
