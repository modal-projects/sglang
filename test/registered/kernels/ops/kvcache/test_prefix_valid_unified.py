"""Prefix commits must touch only selected rows/layers in the shared byte arena."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.unified_memory_pool import (
    MHASubPoolSpec,
    UnifiedKVPool,
    UnifiedMHATokenToKVPool,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestUnifiedPrefixCommit(CustomTestCase):
    def _build(self, page_size, dtype, head_num=8, head_dim=128):
        # Six draft layers with the baseline's per-rank TP1 KV geometry.
        store_dtype = torch.uint8 if dtype == fp8_dtype else dtype
        spec = MHASubPoolSpec(
            name="swa",
            layer_num=6,
            head_num=head_num,
            head_dim=head_dim,
            store_dtype=store_dtype,
            kv_cache_dtype=dtype,
            grow_direction="up",
        )
        full = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=8,
            store_dtype=torch.float16,
            grow_direction="down",
        )
        arena = UnifiedKVPool(
            total_bytes=64 * spec.entry_bytes() + 32 * full.entry_bytes(),
            sub_pool_specs=[spec, full],
            device="cuda",
            enable_memory_saver=False,
            page_size=page_size,
        )
        pool = UnifiedMHATokenToKVPool(
            unified_buffer=arena,
            sub_pool_name="swa",
            page_size=page_size,
            start_layer=2,
            end_layer=7,
            enable_alt_stream=False,
        )
        arena._raw.fill_(0x5A)
        return arena, pool

    def _expected_bytes(self, arena, pool, layer_id, k, v, loc, lengths):
        # Derive byte offsets independently of the pool's strided views.
        expected = arena._raw.cpu().clone()
        row_bytes = pool.head_num * pool.head_dim * pool.dtype.itemsize
        layer_bytes = 2 * row_bytes
        entry_bytes = 6 * layer_bytes  # Both tested row widths are 16-byte aligned.
        for batch, length in enumerate(lengths.cpu().tolist()):
            for col in range(length):
                slot = int(loc[batch, col])
                offset = slot * entry_bytes + (layer_id - 2) * layer_bytes
                source_row = batch * loc.shape[1] + col
                for tensor in (k, v):
                    data = (
                        tensor[source_row]
                        .to(pool.dtype)
                        .contiguous()
                        .view(torch.uint8)
                        .flatten()
                        .cpu()
                    )
                    expected[offset : offset + row_bytes] = data
                    offset += row_bytes
        return expected

    def test_strided_prefix_commit_preserves_every_other_byte(self):
        for dtype in (torch.bfloat16, torch.float16, fp8_dtype):
            for page_size in (1, 4, 16):
                # A short row exercises the masked byte-tile tail, as well as
                # the baseline's wider multi-head row.
                for heads, width in ((1, 48), (8, 128)):
                    with self.subTest(
                        dtype=dtype, page_size=page_size, shape=(heads, width)
                    ):
                        arena, pool = self._build(page_size, dtype, heads, width)
                        values = torch.linspace(
                            -3, 3, 12 * 2 * heads * width, device="cuda"
                        )
                        values = values.to(torch.bfloat16).view(12, 2, heads, width)
                        k, v = values[:, 0], values[:, 1]
                        self.assertFalse(k.is_contiguous())
                        # Noncontiguous metadata; invalid addresses must never
                        # be loaded/stored outside each request's commit prefix.
                        loc = torch.empty((3, 8), dtype=torch.int64, device="cuda")[
                            :, ::2
                        ]
                        loc.copy_(
                            torch.tensor(
                                [
                                    [-1, 999999, 0, -1],
                                    [17, 31, -1, 999999],
                                    [21, 18, 25, 19],
                                ],
                                device="cuda",
                            )
                        )
                        lengths = torch.tensor(
                            [0, -9, 2, -9, 4, -9], dtype=torch.int32, device="cuda"
                        )[::2]
                        for layer_id in (2, 5, 7):
                            expected = self._expected_bytes(
                                arena, pool, layer_id, k, v, loc, lengths
                            )
                            pool.set_kv_buffer_prefix_valid(
                                SimpleNamespace(layer_id=0),
                                KVWriteLoc(loc, physical=True),
                                lengths,
                                k,
                                v,
                                1.0,
                                1.0,
                                layer_id_override=layer_id,
                            )
                            self.assertTrue(torch.equal(arena._raw.cpu(), expected))

    def test_graph_replay_reads_current_locations_and_commit_lengths(self):
        arena, pool = self._build(4, torch.bfloat16, head_num=2, head_dim=8)
        layer = SimpleNamespace(layer_id=4)
        k = torch.arange(12 * 2 * 8, device="cuda").to(torch.bfloat16).view(12, 2, 8)
        v = -k
        loc = torch.arange(16, 28, device="cuda").view(3, 4)
        lengths = torch.tensor([1, 2, 4], dtype=torch.int32, device="cuda")

        def write():
            pool.set_kv_buffer_prefix_valid(
                layer, KVWriteLoc(loc, physical=True), lengths, k, v
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            write()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            write()
        for slots, commits in (
            (torch.arange(32, 44).view(3, 4), [0, 3, 1]),
            (torch.full((3, 4), -1), [0, 0, 0]),
        ):
            arena._raw.fill_(0x5A)
            loc.copy_(slots)
            lengths.copy_(torch.tensor(commits))
            k.add_(1)
            v.sub_(1)
            expected = self._expected_bytes(
                arena, pool, layer.layer_id, k, v, loc, lengths
            )
            graph.replay()
            self.assertTrue(torch.equal(arena._raw.cpu(), expected))


if __name__ == "__main__":
    unittest.main()
