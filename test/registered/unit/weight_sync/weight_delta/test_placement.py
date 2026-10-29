"""Unit tests for where the weight-delta map places a checkpoint tensor."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from pathlib import Path

import torch

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4LinearMethod,
    ModelOptNvFp4A16LinearMethod,
)
from sglang.srt.weight_sync.weight_delta.map import CanonicalTensor
from sglang.srt.weight_sync.weight_delta.rules.placement import linear_places
from sglang.test.test_utils import CustomTestCase


class TestLinearPlaces(CustomTestCase):
    def test_nvfp4_linears_are_refused(self):
        tensor = CanonicalTensor("o_proj.weight", "U8", (16, 8), Path("/dev/null"))
        for method in (ModelOptFp4LinearMethod, ModelOptNvFp4A16LinearMethod):
            with self.subTest(method=method.__name__):
                module = torch.nn.Module()
                module.quant_method = object.__new__(method)
                with self.assertRaisesRegex(ValueError, "no layout for NVFP4 linears"):
                    linear_places(module, None, None, tensor, {})


if __name__ == "__main__":
    unittest.main()
