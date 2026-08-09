"""Unit tests for asynchronous state-capture result lifetime."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.state_capturer.base import (
    BaseDeviceCache,
    BaseTopkCapturer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestStateCapturer(CustomTestCase):
    def test_overlap_output_does_not_alias_next_forward_capture(self):
        capturer = BaseTopkCapturer.__new__(BaseTopkCapturer)
        capturer.num_layers = 2
        capturer.topk_size = 2
        capturer.device_cache = BaseDeviceCache(
            max_batch_size=2,
            num_layers=2,
            topk_size=2,
            device="cpu",
            name="test",
        )
        capturer.host_cache = SimpleNamespace(
            buffer=torch.zeros((16, 2, 2), dtype=torch.int32)
        )

        first = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        capturer.capture(layer_id=0, topk_indices=first)
        out_cache_loc = torch.tensor([5, 6])
        output = capturer.on_forward_end(
            forward_batch=SimpleNamespace(out_cache_loc=out_cache_loc),
            can_run_graph=False,
            cuda_graph_batch=None,
            no_copy_to_cpu=True,
        )
        self.assertIsNotNone(output)

        # The scheduler starts the next forward before the prior result's D2H
        # copy finishes. Reusing the capture buffer must not mutate that result.
        capturer.capture(
            layer_id=0,
            topk_indices=torch.tensor([[9, 10], [11, 12]], dtype=torch.int32),
        )
        out_cache_loc.copy_(torch.tensor([7, 8]))

        torch.testing.assert_close(output.topk[:, 0], first)
        torch.testing.assert_close(output.out_cache_loc, torch.tensor([5, 6]))


if __name__ == "__main__":
    unittest.main()
