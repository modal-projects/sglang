import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.communication import q8_reduce_scatter as q8
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")


def _reduce_scatter_on_one_gpu(partials, rows):
    """pack, the all-to-all, and unpack_sum for len(partials) ranks on one GPU."""
    world = len(partials)
    width = partials[0].shape[1]
    sends = []
    for partial in partials:
        send = torch.empty(
            q8.send_bytes(rows, width, world), dtype=torch.int8, device=partial.device
        )
        q8.pack(partial, rows, world, send)
        sends.append(send.view(world, -1))
    outs = []
    for rank in range(world):
        received = torch.stack([sends[src][rank] for src in range(world)]).view(-1)
        out = partials[0].new_empty(rows // world, width)
        q8.unpack_sum(received, world, out)
        outs.append(out)
    return torch.cat(outs)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestQ8ReduceScatter(CustomTestCase):
    def test_sum_within_int8_error(self):
        """The reduce-scatter matches the fp32 sum to INT8 rounding."""
        torch.manual_seed(0)
        world, valid = 8, 1000
        rows = -(-valid // world) * world
        for width in (3584, 7168):
            partials = [
                torch.randn(valid, width, dtype=torch.bfloat16, device="cuda")
                for _ in range(world)
            ]
            out = _reduce_scatter_on_one_gpu(partials, rows).float()
            exact = torch.stack([p.float() for p in partials]).sum(0)
            groups = torch.stack(
                [p.float().abs().view(valid, -1, q8.GROUP).amax(-1) for p in partials]
            )
            bound = (groups / 254).sum(0).repeat_interleave(q8.GROUP, -1)
            err = (out[:valid] - exact).abs()
            self.assertTrue(
                bool((err <= bound + exact.abs() / 128 + 1e-6).all()),
                f"width {width}: max error {err.max().item()}",
            )
            self.assertTrue(bool((out[valid:] == 0).all()))

    def test_nan_and_inf_groups_arrive_as_nan(self):
        """A group holding a NaN or an infinity arrives as NaN, the rest stay finite."""
        world, rows, width = 4, 8, 512
        partials = [
            torch.ones(rows, width, dtype=torch.bfloat16, device="cuda")
            for _ in range(world)
        ]
        partials[0][3, 5] = float("nan")
        partials[1][6, 40] = float("inf")
        out = _reduce_scatter_on_one_gpu(partials, rows)
        nan = out.isnan()
        self.assertTrue(bool(nan[3, 0:32].all() and nan[6, 32:64].all()))
        self.assertEqual(int(nan.sum()), 2 * q8.GROUP)

    def test_zero_groups_stay_zero(self):
        """All-zero partials reduce to exact zeros."""
        world, rows, width = 8, 16, 512
        partials = [
            torch.zeros(rows, width, dtype=torch.bfloat16, device="cuda")
            for _ in range(world)
        ]
        self.assertTrue(bool((_reduce_scatter_on_one_gpu(partials, rows) == 0).all()))

    def test_graph_replay_after_larger_eager_call(self):
        """A captured reduce-scatter replays correctly after a larger eager call."""
        script = textwrap.dedent(
            """
            import torch
            from sglang.kernels.ops.communication import q8_reduce_scatter as q8

            class Group:
                world_size = 1

                def all_to_all_single(self, output, input):
                    output.copy_(input)

            group = Group()
            small = torch.randn(4096, 7168, dtype=torch.bfloat16, device="cuda")
            expected = q8.reduce_scatter_q8(small, group).clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
                out = q8.reduce_scatter_q8(small, group)
            big = torch.randn(16384, 7168, dtype=torch.bfloat16, device="cuda")
            q8.reduce_scatter_q8(big, group)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, expected)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def test_width_must_be_whole_multiples(self):
        """A width that is not a multiple of WIDTH_MULTIPLE is rejected."""
        partial = torch.zeros(8, 3 * q8.GROUP, dtype=torch.bfloat16, device="cuda")
        self.assertFalse(q8.supports(partial))
        with self.assertRaises(AssertionError):
            q8.reduce_scatter_q8(partial, SimpleNamespace(world_size=8))


if __name__ == "__main__":
    unittest.main()
