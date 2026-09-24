"""Fused MLA writes must not turn finite BF16 overflow into FP8 NaNs."""

import unittest

import torch

from sglang.kernels.ops.attention.set_mla_kv_concat_q import (
    set_mla_kv_concat_q_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestSetMlaKVConcatQFp8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
            raise unittest.SkipTest("fused MLA scatter requires SM90 or newer")

    def _inputs(self, values, loc_dtype, buffer_dtype):
        values = torch.tensor(values, dtype=torch.bfloat16, device="cuda")
        latent = values.repeat((3 * 576 + values.numel() - 1) // values.numel())[
            : 3 * 576
        ].reshape(3, 576)
        query = values.repeat((3 * 5 * 576 + values.numel() - 1) // values.numel())[
            : 3 * 5 * 576
        ].reshape(3, 5, 576)
        pool = torch.full((12, 576), 0x35, dtype=torch.uint8, device="cuda")
        return (
            pool.view(buffer_dtype),
            torch.tensor([7, 2, 9], dtype=loc_dtype, device="cuda"),
            latent[:, :512],
            latent[:, 512:],
            query[..., :512],
            query[..., 512:],
        )

    def _assert_fp8(self, actual, expected):
        actual = actual.view(torch.float8_e4m3fn)
        is_nan = torch.isnan(expected.float())
        self.assertTrue(torch.equal(torch.isnan(actual.float()), is_nan))
        self.assertTrue(
            torch.equal(
                actual.view(torch.uint8)[~is_nan],
                expected.view(torch.uint8)[~is_nan],
            )
        )

    def _check(self, inputs, query, world=1, rank=0):
        pool, loc, k_nope, k_rope, q_nope, q_rope = inputs
        expected_pool = torch.full_like(pool.view(torch.uint8), 0x35)
        owner = loc.remainder(world) == rank
        expected_rows = (
            torch.cat((k_nope, k_rope), dim=-1)
            .float()
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        expected_pool[loc[owner].long() // world] = expected_rows.view(torch.uint8)[
            owner
        ]
        expected_query = (
            torch.cat((q_nope, q_rope), dim=-1)
            .float()
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        self._assert_fp8(pool, expected_pool.view(torch.float8_e4m3fn))
        self._assert_fp8(query, expected_query)

    def test_finite_overflow_and_nan(self):
        """Overflow saturates in both latent halves and query halves; NaNs remain visible."""
        values = [
            0.0,
            -0.0,
            2**-9,
            -(2**-9),
            1.0,
            -1.0,
            448.0,
            -448.0,
            464.0,
            -464.0,
            480.0,
            -480.0,
            512.0,
            -512.0,
            10000.0,
            -10000.0,
            float("inf"),
            -float("inf"),
            float("nan"),
        ]
        for loc_dtype in (torch.int32, torch.int64):
            for buffer_dtype in (torch.uint8, torch.float8_e4m3fn):
                for world, rank in ((1, 0), (2, 0), (2, 1)):
                    with self.subTest(
                        loc_dtype=loc_dtype,
                        buffer_dtype=buffer_dtype,
                        world=world,
                        rank=rank,
                    ):
                        inputs = self._inputs(values, loc_dtype, buffer_dtype)
                        query = set_mla_kv_concat_q_fp8(
                            *inputs, num_warps=4, dcp_world_size=world, dcp_rank=rank
                        )
                        self._check(inputs, query, world, rank)

    def test_in_range_bytes_and_graph_replay(self):
        """Saturation preserves ordinary rounding and repeated graph writes."""
        values = torch.linspace(-448, 448, 127).tolist() + [0.0, -0.0, 2**-9]
        inputs = self._inputs(values, torch.int64, torch.uint8)
        eager = set_mla_kv_concat_q_fp8(*inputs, num_warps=1)
        self._check(inputs, eager)
        raw_query = torch.cat(inputs[4:], dim=-1).to(torch.float8_e4m3fn)
        self.assertTrue(
            torch.equal(eager.view(torch.uint8), raw_query.view(torch.uint8))
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            query = set_mla_kv_concat_q_fp8(*inputs, num_warps=1)
        for value in (512.0, -512.0):
            inputs[0].fill_(0x35)
            for component in inputs[2:]:
                component.fill_(value)
            graph.replay()
            self._check(inputs, query)


if __name__ == "__main__":
    unittest.main()
