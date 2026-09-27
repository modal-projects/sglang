"""K3 FP8 conversion must preserve scales, output shape and empty TP batches."""

import unittest

import torch

from sglang.srt.layers.k3_dense_fp8 import K3DenseFP8Linear
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestK3DenseFP8(CustomTestCase):
    def test_weight_scale_and_tuple_contract(self):
        for magnitude in (0.125, 1.0, 8.0):
            weight = torch.eye(32, device="cuda", dtype=torch.bfloat16) * magnitude
            layer = K3DenseFP8Linear(weight=weight, tuple_output=True)
            for batch in (0, 1, 17, 128):
                x = torch.full((batch, 32), 2.0, device="cuda", dtype=torch.bfloat16)
                output, bias = layer(x)
                self.assertIsNone(bias)
                self.assertEqual(output.dtype, x.dtype)
                torch.testing.assert_close(output, x * magnitude, rtol=0, atol=0)

    def test_static_activation_range(self):
        weight = torch.eye(32, device="cuda", dtype=torch.bfloat16)
        layer = K3DenseFP8Linear(weight=weight)
        x = torch.full((1, 32), 512.0, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(layer(x), torch.full_like(x, 448.0), rtol=0, atol=0)

    def test_invalid_weight_contract(self):
        for shape, dtype in (((31, 32), torch.bfloat16), ((32, 32), torch.float32)):
            with self.subTest(shape=shape, dtype=dtype), self.assertRaises(ValueError):
                K3DenseFP8Linear(weight=torch.zeros(shape, device="cuda", dtype=dtype))
        for value in (float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                K3DenseFP8Linear(
                    weight=torch.full(
                        (32, 32), value, device="cuda", dtype=torch.bfloat16
                    )
                )


if __name__ == "__main__":
    unittest.main()
