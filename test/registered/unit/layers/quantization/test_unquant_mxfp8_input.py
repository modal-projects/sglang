"""UnquantizedLinearMethod declines the DeepGEMM-packed MXFP8 input a model asks for."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.test.test_utils import CustomTestCase


class TestUnquantizedMxfp8Input(CustomTestCase):
    def test_bf16_weights_decline_the_requested_mxfp8_input(self):
        layer = ReplicatedLinear(128, 64, bias=False, params_dtype=torch.bfloat16)
        # As DeepSeek-V4.1's mega attention asks of wo_b before the weights load.
        layer.deepgemm_mxfp8_input = True
        layer.quant_method.process_weights_after_loading(layer)
        self.assertFalse(layer.deepgemm_mxfp8_input)


if __name__ == "__main__":
    unittest.main()
