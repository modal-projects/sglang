import os
import subprocess
import sys
import textwrap
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")

_LEAD_OVER_CHAIN_MS = textwrap.dedent(
    """
    import statistics, torch
    from sglang.srt.managers.utils import exclusive_queue_stream

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16) * 0.01
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16) * 0.01
    x = torch.zeros(16, device="cuda")

    def lead_ms(f, v):
        out = []
        for _ in range(5):
            torch.cuda.synchronize()
            chain_end = torch.cuda.Event(enable_timing=True)
            done = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(f):
                for _ in range(400):
                    torch.mm(a, b, out=a)
                chain_end.record()
            with torch.cuda.stream(v):
                x.add_(1)
                done.record()
            torch.cuda.synchronize()
            out.append(done.elapsed_time(chain_end))
        return statistics.median(out)

    shared = lead_ms(torch.cuda.Stream(), torch.cuda.Stream())
    exclusive = lead_ms(exclusive_queue_stream("cuda"), torch.cuda.Stream())
    print(shared, exclusive)
    """
)


@unittest.skipUnless(
    torch.version.hip and torch.cuda.is_available(), "needs a ROCm GPU"
)
class TestExclusiveQueueStream(CustomTestCase):
    def test_kernel_beside_exclusive_stream_does_not_wait(self):
        """On one shared HW queue, a kernel waits behind another stream's work unless that stream is exclusive."""
        result = subprocess.run(
            [sys.executable, "-c", _LEAD_OVER_CHAIN_MS],
            env={**os.environ, "GPU_MAX_HW_QUEUES": "1"},
            capture_output=True,
            text=True,
            check=True,
        )
        shared, exclusive = map(float, result.stdout.splitlines()[-1].split())
        self.assertLess(shared, 0)
        self.assertGreater(exclusive, 1)


if __name__ == "__main__":
    unittest.main()
