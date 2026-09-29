import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.radix_attention import _zero_padded_pcg_tail
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestZeroPaddedPcgTail(CustomTestCase):
    def test_non_contiguous_output(self):
        buf = torch.ones(8, 6, 4).transpose(1, 2)
        _zero_padded_pcg_tail(buf, SimpleNamespace(num_tokens=8, raw_num_tokens=3))
        self.assertTrue(buf[:3].eq(1).all())
        self.assertTrue(buf[3:].eq(0).all())


if __name__ == "__main__":
    unittest.main()
