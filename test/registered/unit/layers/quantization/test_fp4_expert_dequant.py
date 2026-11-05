"""Dequantized FP4 routed experts must carry scales at the checkpoint's block size.

DeepSeek-V4.1 declares 32x32 weight blocks; scales emitted at 128x128 make the
block-FP8 MoE runner fail on the first forward.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.fp8 import (
    DSV4_DEQUANT_FP4_TABLE,
    cast_e2m1fn_to_e4m3fn,
)
from sglang.test.test_utils import CustomTestCase


def _reference_dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    u8 = packed.view(torch.uint8)
    low = DSV4_DEQUANT_FP4_TABLE[(u8 & 0x0F).long()]
    high = DSV4_DEQUANT_FP4_TABLE[(u8 >> 4).long()]
    values = torch.stack([low, high], dim=-1).flatten(1)
    return values * scale.repeat_interleave(32, dim=1)


class TestFp4ExpertDequant(CustomTestCase):
    def test_dequant_honors_checkpoint_block_size(self):
        gen = torch.Generator().manual_seed(0)
        out_dim, in_dim = 128, 256
        packed = torch.randint(
            -128, 128, (out_dim, in_dim // 2), dtype=torch.int8, generator=gen
        )
        exponents = torch.randint(-4, 2, (out_dim, in_dim // 32), generator=gen)
        scale = torch.pow(2.0, exponents.float())
        expected = _reference_dequant(packed, scale)

        for block in (32, 128):
            weight, block_scale = cast_e2m1fn_to_e4m3fn(
                packed, scale, fp8_block_size=block
            )
            self.assertEqual(weight.dtype, torch.float8_e4m3fn)
            self.assertEqual(
                tuple(block_scale.shape), (out_dim // block, in_dim // block)
            )
            dequant = weight.float() * block_scale.float().repeat_interleave(
                block, dim=0
            ).repeat_interleave(block, dim=1)
            torch.testing.assert_close(dequant, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
