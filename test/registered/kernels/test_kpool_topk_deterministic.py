import struct
import unittest

import torch

from sglang.kernels.ops.moe.kpool_topk_transform import (
    SUPPORTED_GROUP_TOPK,
    _jit_kpool_topk_transform_module,
    fast_kpool_topk_transform_fused,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def ordered_float(value):
    bits = struct.unpack("I", struct.pack("f", value))[0]
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits | 0x80000000


def reference(scores, length, k, pool=4, tail=0, table=None, offset=0):
    selected = sorted(range(length), key=lambda i: (-ordered_float(scores[i]), i))[:k]
    tokens = [i * pool + j for i in sorted(selected) for j in range(pool)]
    tokens += list(range(length * pool, length * pool + tail))
    tokens = [table[i] if table is not None else i + offset for i in tokens]
    return tokens + [-1] * (k * pool + pool - 1 - len(tokens))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDeterministicKPool(unittest.TestCase):
    def check_rows(self, rows, k=512, pool=4, tails=None):
        tails = tails or [0] * len(rows)
        width = max(map(len, rows)) + 7
        score = torch.full((len(rows), width), float("nan"), device="cuda")
        starts = torch.tensor([3] * len(rows), dtype=torch.int32, device="cuda")
        for i, row in enumerate(rows):
            score[i, 3 : 3 + len(row)] = torch.tensor(row, device="cuda")
        lengths = torch.tensor(list(map(len, rows)), dtype=torch.int32, device="cuda")
        seq = lengths * pool + torch.tensor(tails, device="cuda", dtype=torch.int32)
        expected = torch.tensor(
            [reference(r, len(r), k, pool, tail) for r, tail in zip(rows, tails)],
            dtype=torch.int32,
        )
        for _ in range(5):
            actual = fast_kpool_topk_transform_fused(
                score, lengths, pool, k * pool, row_starts=starts, seq_lens=seq
            )
            self.assertTrue(torch.equal(actual.cpu(), expected))

    def test_ties_capacity_boundaries_and_supported_k(self):
        for k in SUPPORTED_GROUP_TOPK:
            with self.subTest(k=k):
                self.check_rows([[1.0] * n for n in [0, 1, k, k + 1, 4096, 4097, 8192]], k)

    def test_coarse_bin_overflow_distinct_scores_and_signed_zero(self):
        generator = torch.Generator().manual_seed(41)
        rows = [
            [1.0] * 4096 + [1.001] * 4096,
            [1.0 + i * 1e-7 for i in range(8192)],
            torch.randint(-8, 9, (8192,), generator=generator).float().tolist(),
            [-0.0, 0.0] * 4096,
            [-1.0] * 65536,
        ]
        self.check_rows(rows, tails=[0, 1, 2, 3, 0])

    def test_graph_replay_mutable_layout_and_physical_mapping(self):
        k, pool, batch, width = 512, 4, 4, 8208
        module = _jit_kpool_topk_transform_module(k)
        score = torch.zeros((batch, width), device="cuda")
        lengths = torch.full((batch,), 577, device="cuda", dtype=torch.int32)
        starts = torch.zeros_like(lengths)
        seq = lengths * pool
        rows = torch.tensor([5, 2, 4, 1], device="cuda", dtype=torch.int32)
        table = torch.arange(6 * width * pool, device="cuda", dtype=torch.int32).reshape(6, -1)
        out = torch.empty((batch, k * pool + pool - 1), device="cuda", dtype=torch.int32)

        def run():
            module.kpool_topk_transform(score, lengths, out, pool, table, None, starts, seq, rows)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        for iteration in range(8):
            counts = [0, 513, 4097, 8192] if iteration % 2 else [577, 512, 3, 4096]
            lengths.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
            starts.copy_(torch.tensor([1, 3, 5, 7], device="cuda", dtype=torch.int32))
            tails = [0, 1, 2, 3]
            seq.copy_(lengths * pool + torch.arange(batch, device="cuda", dtype=torch.int32))
            table.copy_(table.flip(1))
            rows.copy_(rows.roll(1))
            score.fill_(float("nan"))
            score_rows = []
            for i, n in enumerate(counts):
                values = [float((j * 37 + iteration) % 17) for j in range(n)]
                score_rows.append(values)
                score[i, 1 + 2 * i : 1 + 2 * i + n] = torch.tensor(values, device="cuda")
            out.fill_(-12345)
            graph.replay()
            actual = out.cpu()
            table_cpu, rows_cpu = table.cpu().tolist(), rows.cpu().tolist()
            expected = torch.tensor([
                reference(v, n, k, pool, tail, table_cpu[r])
                for v, n, tail, r in zip(score_rows, counts, tails, rows_cpu)
            ], dtype=torch.int32)
            self.assertTrue(torch.equal(actual, expected))

    def test_ragged_offsets_and_concurrent_streams(self):
        k, pool = 512, 4
        row = [float((i * 53) % 97) for i in range(4097)]
        score = torch.tensor([row] * 8, device="cuda")
        lengths = torch.full((8,), len(row), device="cuda", dtype=torch.int32)
        offsets = torch.arange(8, device="cuda", dtype=torch.int32) * 20000
        tails = torch.arange(8, device="cuda", dtype=torch.int32) % pool
        seq = lengths * pool + tails
        expected = torch.tensor([
            reference(row, len(row), k, pool, i % pool, offset=i * 20000)
            for i in range(8)
        ], dtype=torch.int32)
        streams = [torch.cuda.Stream() for _ in range(4)]
        outputs = []
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                outputs.append(fast_kpool_topk_transform_fused(
                    score, lengths, pool, k * pool, topk_indices_offset=offsets, seq_lens=seq
                ))
        for stream, output in zip(streams, outputs):
            torch.cuda.current_stream().wait_stream(stream)
            self.assertTrue(torch.equal(output.cpu(), expected))


if __name__ == "__main__":
    unittest.main()
