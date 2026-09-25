"""Reserved rows and unwritten page tails stay finite across KV writers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.set_mla_kv_concat_q import set_mla_kv_concat_q_fp8
from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_dcp_sharded_triton
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    MLATokenToKVPool,
    _set_kv_buffer_prefix_valid_impl,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestPageLifecycle(CustomTestCase):
    def test_prefix_commit_skips_padding_and_uncommitted_rows(self):
        for loc_dtype in (torch.int32, torch.int64):
            with self.subTest(loc_dtype=loc_dtype):
                k = torch.arange(6 * 32, device="cuda", dtype=torch.bfloat16).reshape(
                    6, 2, 16
                )
                v = k + 2
                k[0].fill_(float("nan"))
                v[0].fill_(float("nan"))
                k_cache = torch.full((12, 2, 16), 7.0, device="cuda", dtype=k.dtype)
                v_cache = k_cache.clone()
                loc = torch.tensor(
                    [[0, 2, 4], [6, 7, 8]], dtype=loc_dtype, device="cuda"
                )
                lengths = torch.tensor([2, 0], dtype=torch.int32, device="cuda")
                _set_kv_buffer_prefix_valid_impl(
                    k, v, k_cache, v_cache, loc, lengths, 32, k.dtype
                )
                expected_k, expected_v = (
                    torch.full_like(k_cache, 7),
                    torch.full_like(v_cache, 7),
                )
                expected_k[2], expected_v[2] = k[1], v[1]
                torch.testing.assert_close(k_cache, expected_k, rtol=0, atol=0)
                torch.testing.assert_close(v_cache, expected_v, rtol=0, atol=0)

    def test_dcp_reserved_index_is_a_physical_row(self):
        for rank in (0, 1):
            for loc_dtype in (torch.int32, torch.int64):
                with self.subTest(rank=rank, loc_dtype=loc_dtype):
                    loc = torch.tensor(
                        [rank, 2 + rank, 4 + (1 - rank)], dtype=loc_dtype, device="cuda"
                    )
                    nope = torch.full(
                        (3, 1, 8), 2.0, dtype=torch.bfloat16, device="cuda"
                    )
                    rope = torch.full_like(nope, 3)
                    nope[0].fill_(float("nan"))
                    rope[0].fill_(float("nan"))
                    cache = torch.full((4, 1, 16), 7.0, dtype=nope.dtype, device="cuda")
                    with patch(
                        "sglang.kernels.ops.kvcache.mla_buffer.get_parallel",
                        return_value=SimpleNamespace(
                            attn_dcp_size=2, attn_dcp_rank=rank
                        ),
                    ):
                        set_mla_kv_buffer_dcp_sharded_triton(cache, loc, nope, rope)
                    expected = torch.full_like(cache, 7)
                    expected[1] = torch.cat((nope[1], rope[1]), dim=-1)
                    torch.testing.assert_close(cache, expected, rtol=0, atol=0)

    def test_fused_fp8_keeps_query_outputs_when_kv_writes_are_skipped(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("Fused TMA writer requires SM90+")
        for world, rank in ((1, 0), (2, 0), (2, 1)):
            for loc_dtype in (torch.int32, torch.int64):
                for num_warps in (1, 2, 4, 8):
                    with self.subTest(
                        world=world, rank=rank, loc_dtype=loc_dtype, num_warps=num_warps
                    ):
                        locations = (
                            [0, 1, 2]
                            if world == 1
                            else [rank, 2 + rank, 4 + (1 - rank)]
                        )
                        loc = torch.tensor(locations, dtype=loc_dtype, device="cuda")
                        nope = torch.full(
                            (3, 512), 2.0, dtype=torch.bfloat16, device="cuda"
                        )
                        rope = torch.full(
                            (3, 64), 3.0, dtype=torch.bfloat16, device="cuda"
                        )
                        nope[0].fill_(float("nan"))
                        rope[0].fill_(float("nan"))
                        qn = torch.full(
                            (3, 2, 512), 1.0, dtype=torch.bfloat16, device="cuda"
                        )
                        qr = torch.full(
                            (3, 2, 64), 2.0, dtype=torch.bfloat16, device="cuda"
                        )
                        cache = torch.full(
                            (4, 576), 7.0, dtype=torch.bfloat16, device="cuda"
                        ).to(torch.float8_e4m3fn)
                        expected = cache.clone()
                        expected[1] = torch.cat((nope[1], rope[1])).to(expected.dtype)
                        if world == 1:
                            expected[2] = torch.cat((nope[2], rope[2])).to(
                                expected.dtype
                            )
                        kwargs = dict(
                            num_warps=num_warps, dcp_world_size=world, dcp_rank=rank
                        )
                        # Warm compilation before capture; replay checks padded graph rows too.
                        set_mla_kv_concat_q_fp8(
                            cache, loc, nope, rope, qn, qr, **kwargs
                        )
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            query = set_mla_kv_concat_q_fp8(
                                cache, loc, nope, rope, qn, qr, **kwargs
                            )
                        graph.replay()
                        torch.testing.assert_close(
                            cache.view(torch.uint8),
                            expected.view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
                        expected_q = torch.cat((qn, qr), dim=-1).to(query.dtype)
                        torch.testing.assert_close(
                            query.view(torch.uint8),
                            expected_q.view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )

    def test_paged_handouts_clear_full_float8_envelopes(self):
        pool = MLATokenToKVPool.__new__(MLATokenToKVPool)
        pool.page_size = 4
        pool.kv_buffer = [
            torch.full((20, 1, 8), 7.0, device="cuda").to(torch.float8_e4m3fn)
        ]
        allocator = PagedTokenToKVPoolAllocator(
            16, 4, torch.float8_e4m3fn, "cuda", pool, False
        )
        first = allocator.alloc(4)
        pool.kv_buffer[0].view(torch.uint8)[first] = 0x7F
        allocator.free(first)
        prefix = torch.tensor([0], device="cuda", dtype=torch.int64)
        lengths = torch.tensor([1], device="cuda", dtype=torch.int64)
        indices = allocator.alloc_extend(
            prefix,
            prefix.cpu(),
            lengths,
            lengths.cpu(),
            torch.tensor([-1], device="cuda"),
            1,
        )
        self.assertEqual(indices.item(), 4)
        self.assertTrue(torch.all(pool.kv_buffer[0].view(torch.uint8)[4:8] == 0))
        next_index = allocator.alloc_decode(
            torch.tensor([5], device="cuda"),
            torch.tensor([5]),
            torch.tensor([7], device="cuda"),
        )
        self.assertEqual(next_index.item(), 8)
        self.assertTrue(torch.all(pool.kv_buffer[0].view(torch.uint8)[8:12] == 0))
        self.assertTrue(torch.all(pool.kv_buffer[0][12:].float() == 7))


if __name__ == "__main__":
    unittest.main()
