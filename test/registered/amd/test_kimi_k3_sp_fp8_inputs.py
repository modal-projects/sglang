import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm as gemm
from sglang.srt.layers import layernorm_sp
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention, KimiK3MoE
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

HIDDEN = 7168
PROJ_TP = 1536  # 12 heads x 128 per rank at TP8
HEAD_DIM = 128
HEADS_TP = 12
WORLD, RANK = 8, 3
TOKENS = 1003


def _weight(rows, cols=HIDDEN):
    return torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda") * 0.02


class _Group:
    """A TP group of WORLD ranks seen from RANK, gathering fixed per-rank b weights."""

    world_size, rank_in_group = WORLD, RANK

    def __init__(self, b_ranks):
        self.b_ranks = b_ranks

    def all_gather_into_tensor(self, out, inp):
        assert torch.equal(inp, self.b_ranks[RANK])
        out.copy_(torch.cat(self.b_ranks))


@unittest.skipUnless(torch.cuda.is_available(), "no GPU")
class TestKimiK3SPFP8Inputs(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_kda_sp_input_matches_unsharded_projection(self):
        """The packed E4M3 + [f_a | b] gather yields the unsharded projections."""
        from sglang.srt.layers.quantization.k3_rocm_dense_fp8 import StaticFP8Weight

        b_ranks = [_weight(HEADS_TP) for _ in range(WORLD)]
        attn = SimpleNamespace(
            f_a_proj=SimpleNamespace(weight=_weight(HEAD_DIM)),
            b_proj=SimpleNamespace(weight=b_ranks[RANK]),
            _bfa_f_b_w=_weight(PROJ_TP, HEAD_DIM),
            _qkvg_fp8=StaticFP8Weight(_weight(4 * PROJ_TP)),
            split_sizes=[3 * PROJ_TP, PROJ_TP],
        )
        x = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
        parallel = SimpleNamespace(tp_group=_Group(b_ranks))
        with (
            patch.object(kimi_k3, "get_parallel", return_value=parallel),
            patch.object(layernorm_sp, "sp_num_tokens", return_value=TOKENS),
            patch.object(layernorm_sp, "sp_exit_gather", lambda t, n: t[:n]),
        ):
            KimiK3DeltaAttention._merge_sp_bfa_weights(attn)
            xq, bfa = KimiK3DeltaAttention.sp_gather_input(attn, x)
            qkv, beta, forget_gate, g = KimiK3DeltaAttention._qkvbfg_sp(attn, xq, bfa)

        fused = attn._qkvg_fp8(x)
        f_a = gemm(x, attn.f_a_proj.weight)
        self.assertTrue(torch.equal(qkv, fused[:, : 3 * PROJ_TP]))
        self.assertTrue(torch.equal(g, fused[:, 3 * PROJ_TP :]))
        torch.testing.assert_close(beta, gemm(x, b_ranks[RANK]), rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(
            forget_gate, gemm(f_a, attn._bfa_f_b_w), rtol=1e-2, atol=1e-3
        )

    def test_quantized_moe_front_matches_bf16_input(self):
        """The front on E4M3 rows equals the FP8 front on the bf16 rows."""
        from sglang.srt.layers.quantization.k3_rocm_dense_fp8 import StaticFP8Weight

        sizes = [1024, 384, 3584]
        moe = SimpleNamespace(
            _front_fp8=StaticFP8Weight(_weight(sum(sizes))),
            _front_fp32=False,
            _front_sizes=sizes,
            _moe_front_needs_dense_bf16=True,
        )
        x = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
        quantized = KimiK3MoE._front_outputs(
            moe, moe._front_fp8.quantize(x), x.dtype, quantized=True
        )
        unquantized = KimiK3MoE._front_outputs(moe, x, x.dtype)
        for a, b in zip(quantized, unquantized):
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
