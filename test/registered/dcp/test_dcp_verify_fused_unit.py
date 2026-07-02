"""
Unit test: fused DCP speculative-verify kernel vs the torch reference.

Compares ``dcp_verify_draft_merge`` (one fused Triton kernel replacing the
phases (b)+(c) of ``TRTLLMMLABackend._forward_target_verify_dcp``: the
residue-class draft-block attention plus the local base-2 LSE merge with the
phase-(a) prefix partial) against ``dcp_verify_draft_merge_torch`` (the
original, numerically validated unfused torch implementation) on random
inputs, including the edge cases the production path hits:

  - seq_lens < dcp_world_size: empty phase-(a) rows, signalled by the decode
    kernel with a +inf (or NaN) LSE sentinel and garbage/NaN outputs;
  - ranks that own ZERO residue-class positions of the draft block
    (dcp_world_size > num_draft_tokens): empty phase-(b), lse_b = -inf;
  - both phases empty simultaneously: out = 0, merged lse = -inf;
  - k_scale != 1 (softmax_scale and output_scale carry the fp8 KV dequant);
  - fp8 (e4m3) and bf16 q/k inputs; bf16 and fp32 phase-(a) outputs;
  - CUDA-graph capture + replay of the fused kernel.

Runs on a single GPU, no server launch.
"""

import math
import unittest

import torch

from sglang.srt.layers.cp.dcp.kernels import (
    dcp_verify_draft_merge,
    dcp_verify_draft_merge_torch,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu")

KV_LORA = 512
ROPE_DIM = 64


def _make_inputs(
    bs,
    draft,
    num_heads,
    seq_lens,
    dcp_world_size,
    qk_dtype=torch.bfloat16,
    oa_dtype=torch.bfloat16,
    empty_prefix_sentinel="inf",
    seed=0,
    device="cuda",
):
    """Random inputs mimicking what _forward_target_verify_dcp passes.

    Rows whose local prefix share is empty (rank-local prefix len == 0) get
    the decode kernel's empty-row behavior: lse_a = +inf (or NaN) sentinel
    and NaN-poisoned o_a, which both implementations must neutralize.
    """
    gen = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=gen, device=device).to(dtype)

    q = rand(bs, draft, num_heads, KV_LORA + ROPE_DIM, dtype=qk_dtype)
    k_latent = rand(bs, draft, KV_LORA, dtype=qk_dtype)
    k_rope = rand(bs, draft, ROPE_DIM, dtype=qk_dtype)
    o_a = rand(bs, draft, num_heads, KV_LORA, dtype=oa_dtype)
    # Plausible base-2 LSE magnitudes for a softmax over a long prefix.
    lse_a = (
        torch.rand(bs, draft, num_heads, generator=gen, device=device) * 20.0
        + math.log2(64.0)
    ).to(torch.float32)

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int64, device=device)
    # Per-request rank-local prefix length is ceil/floor of seq_len /
    # world_size; it is zero for SOME rank whenever seq_len < world_size.
    # Emulate the empty rows on requests with seq_len < world_size.
    empty = seq_lens_t < dcp_world_size
    if empty.any():
        sentinel = float("inf") if empty_prefix_sentinel == "inf" else float("nan")
        lse_a[empty] = sentinel
        o_a[empty] = float("nan")
    return q, k_latent, k_rope, o_a, lse_a, seq_lens_t


def _tolerances(qk_dtype, oa_dtype):
    if qk_dtype == torch.float8_e4m3fn:
        # Both paths dequantize the SAME fp8 values to fp32; residual
        # differences come only from summation order.
        out_tol = dict(rtol=2e-2, atol=2e-2) if oa_dtype != torch.float32 else dict(
            rtol=2e-4, atol=2e-4
        )
    elif oa_dtype == torch.float32:
        out_tol = dict(rtol=2e-4, atol=2e-4)
    else:  # bf16 output rounding dominates
        out_tol = dict(rtol=2e-2, atol=2e-2)
    lse_tol = dict(rtol=2e-4, atol=2e-4)
    return out_tol, lse_tol


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class TestDCPVerifyFusedKernel(CustomTestCase):
    def _run_case(
        self,
        bs=4,
        draft=8,
        num_heads=64,
        seq_lens=None,
        dcp_world_size=4,
        ranks=None,
        qk_dtype=torch.bfloat16,
        oa_dtype=torch.bfloat16,
        softmax_scale=0.1352,
        output_scale=1.0,
        empty_prefix_sentinel="inf",
        seed=0,
    ):
        if seq_lens is None:
            seq_lens = [1000 + 7 * i for i in range(bs)]
        assert len(seq_lens) == bs
        inputs = _make_inputs(
            bs,
            draft,
            num_heads,
            seq_lens,
            dcp_world_size,
            qk_dtype=qk_dtype,
            oa_dtype=oa_dtype,
            empty_prefix_sentinel=empty_prefix_sentinel,
            seed=seed,
        )
        out_tol, lse_tol = _tolerances(qk_dtype, oa_dtype)
        for rank in ranks if ranks is not None else range(dcp_world_size):
            with self.subTest(rank=rank):
                out_f, lse_f = dcp_verify_draft_merge(
                    *inputs, softmax_scale, output_scale, rank, dcp_world_size
                )
                out_r, lse_r = dcp_verify_draft_merge_torch(
                    *inputs, softmax_scale, output_scale, rank, dcp_world_size
                )
                self.assertEqual(out_f.dtype, out_r.dtype)
                self.assertEqual(out_f.shape, out_r.shape)
                self.assertFalse(torch.isnan(lse_f).any().item())
                self.assertFalse(torch.isnan(out_f.float()).any().item())
                # -inf == -inf compares equal in assert_close.
                torch.testing.assert_close(lse_f, lse_r, **lse_tol)
                torch.testing.assert_close(out_f, out_r, **out_tol)

    def test_basic_bf16(self):
        self._run_case()

    def test_fp8_qk(self):
        """fp8 e4m3 q/k as produced by mla_quantize_and_rope_for_fp8."""
        self._run_case(qk_dtype=torch.float8_e4m3fn, seed=1)

    def test_k_scale_not_one(self):
        """softmax_scale = layer.scaling * k_scale, output_scale = k_scale."""
        self._run_case(
            qk_dtype=torch.float8_e4m3fn,
            softmax_scale=0.1352 * 0.75,
            output_scale=0.75,
            seed=2,
        )

    def test_fp32_o_a_tight(self):
        """fp32 phase-(a) output: no output-cast rounding, tight tolerance."""
        self._run_case(oa_dtype=torch.float32, seed=3)

    def test_short_prefix_empty_phase_a_inf_sentinel(self):
        """seq_lens < dcp_world_size: +inf lse_a sentinel + NaN o_a rows."""
        self._run_case(
            seq_lens=[0, 1, 3, 4096],
            empty_prefix_sentinel="inf",
            seed=4,
        )

    def test_short_prefix_empty_phase_a_nan_sentinel(self):
        """Same, but the degenerate-row sentinel arrives as NaN."""
        self._run_case(
            seq_lens=[0, 2, 3, 77],
            empty_prefix_sentinel="nan",
            seed=5,
        )

    def test_rank_owns_zero_block_positions(self):
        """dcp_world_size > draft: ranks >= draft own no residue class of the
        fresh block (empty phase-b, lse_b = -inf on every row)."""
        self._run_case(
            dcp_world_size=16,
            seq_lens=[16, 33, 160, 4097],
            ranks=range(16),
            seed=6,
        )

    def test_both_phases_empty(self):
        """seq_len == 0 AND a rank owning no block position: out must be
        exactly 0 and merged lse exactly -inf."""
        bs, draft, num_heads, world = 2, 8, 8, 16
        inputs = _make_inputs(
            bs, draft, num_heads, [0, 0], world, seed=7
        )
        # seq_len = 0: block position j is owned by rank j % 16 -> ranks
        # 8..15 own nothing and have empty prefixes.
        for rank in range(draft, world):
            out_f, lse_f = dcp_verify_draft_merge(*inputs, 0.1352, 1.0, rank, world)
            out_r, lse_r = dcp_verify_draft_merge_torch(
                *inputs, 0.1352, 1.0, rank, world
            )
            self.assertTrue((out_f.float() == 0).all().item())
            self.assertTrue((lse_f == float("-inf")).all().item())
            self.assertTrue((out_r.float() == 0).all().item())
            self.assertTrue((lse_r == float("-inf")).all().item())

    def test_odd_shapes(self):
        """Non-default head count / batch size / draft-token count."""
        self._run_case(bs=3, num_heads=16, seed=8)
        self._run_case(bs=1, num_heads=128, dcp_world_size=2, seed=9)
        self._run_case(draft=4, seq_lens=[5, 6, 1, 4000], seed=10)

    def test_cuda_graph_capture_replay(self):
        """The fused kernel must be CUDA-graph capturable, and replays must
        track in-place input updates (no host reads baked in)."""
        bs, draft, num_heads, world, rank = 2, 8, 16, 4, 1
        scale, oscale = 0.1352, 0.75
        inputs = list(
            _make_inputs(bs, draft, num_heads, [9, 4096], world, seed=11)
        )
        static = [x.clone() for x in inputs]

        # Warmup on a side stream, then capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            dcp_verify_draft_merge(*static, scale, oscale, rank, world)
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_g, lse_g = dcp_verify_draft_merge(
                *static, scale, oscale, rank, world
            )

        # New random inputs (different seq_lens too), copied in place.
        new_inputs = _make_inputs(
            bs, draft, num_heads, [4095, 2], world, seed=12
        )
        for dst, src in zip(static, new_inputs):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()

        out_r, lse_r = dcp_verify_draft_merge_torch(
            *new_inputs, scale, oscale, rank, world
        )
        out_tol, lse_tol = _tolerances(torch.bfloat16, torch.bfloat16)
        torch.testing.assert_close(lse_g, lse_r, **lse_tol)
        torch.testing.assert_close(out_g, out_r, **out_tol)


if __name__ == "__main__":
    unittest.main()
