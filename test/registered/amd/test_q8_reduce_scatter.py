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
    local = rows // world * width
    chunk = local + local // q8.GROUP * 4
    sends = []
    for partial in partials:
        send = torch.empty(world * chunk, dtype=torch.int8, device=partial.device)
        q8.pack(partial, rows, world, send)
        sends.append(send.view(world, chunk))
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
        world, rows, width = 4, 8, 128
        partials = [
            torch.ones(rows, width, dtype=torch.bfloat16, device="cuda")
            for _ in range(world)
        ]
        partials[0][3, 5] = float("nan")
        partials[1][6, 40] = float("inf")
        out = _reduce_scatter_on_one_gpu(partials, rows)
        nan = out.isnan()
        self.assertTrue(bool(nan[3, 0:32].all() and nan[6, 32:64].all()))
        self.assertEqual(int(nan.sum()), 64)

    def test_zero_groups_stay_zero(self):
        """All-zero partials reduce to exact zeros."""
        world, rows, width = 8, 16, 64
        partials = [
            torch.zeros(rows, width, dtype=torch.bfloat16, device="cuda")
            for _ in range(world)
        ]
        self.assertTrue(bool((_reduce_scatter_on_one_gpu(partials, rows) == 0).all()))

    def test_width_must_be_whole_groups(self):
        """A width that is not a multiple of the group size is rejected."""
        with self.assertRaises(AssertionError):
            q8.reduce_scatter_q8(
                torch.zeros(8, 33, device="cuda"), SimpleNamespace(world_size=8)
            )


if __name__ == "__main__":
    unittest.main()
