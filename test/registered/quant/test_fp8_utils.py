import unittest

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    inverse_transform_scale_ue8m0,
    quant_weight_ue8m0,
    restore_scale_checkpoint_state,
    snapshot_scale_checkpoint_state,
    transform_scale_ue8m0,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-large")


class TestInverseTransformScaleUe8m0(CustomTestCase):
    def test_round_trip(self):
        for _ in range(100):
            weight_bf16 = torch.randn(
                # DeepSeek V3 kv_b_proj
                (32768, 512),
                dtype=torch.bfloat16,
                device="cuda",
            )

            weight_block_size = [128, 128]

            qweight, sf_fp32_original = quant_weight_ue8m0(
                weight_bf16, weight_block_size=weight_block_size
            )
            mn = qweight.shape[-2]

            sf_packed_original = transform_scale_ue8m0(sf_fp32_original, mn=mn)
            sf_fp32_recreated = inverse_transform_scale_ue8m0(sf_packed_original, mn=mn)

            sf_packed_recreated = transform_scale_ue8m0(sf_fp32_recreated, mn=mn)

            assert torch.all(
                sf_packed_original == sf_packed_recreated
            ), f"{sf_packed_original=} {sf_packed_recreated}"
            assert torch.all(
                sf_fp32_original == sf_fp32_recreated
            ), f"{sf_fp32_original=} {sf_fp32_recreated}"

    def test_round_trip_with_partial_last_block(self):
        mn = 385
        scales = torch.tensor(
            [
                [0.5, 1.0, 2.0, 4.0],
                [1.0, 2.0, 4.0, 8.0],
                [2.0, 4.0, 8.0, 16.0],
                [4.0, 8.0, 16.0, 32.0],
            ],
            device="cuda",
        )
        packed = transform_scale_ue8m0(scales, mn=mn)
        restored = inverse_transform_scale_ue8m0(packed, mn=mn)
        torch.testing.assert_close(restored, scales)

    def test_restore_checkpoint_scale_layout(self):
        scale = torch.nn.Parameter(torch.ones((3, 4)), requires_grad=False)
        scale.format_ue8m0 = False
        snapshot_scale_checkpoint_state(scale)

        runtime_buffer = torch.zeros((9, 2), dtype=torch.int32)
        scale.data = runtime_buffer
        scale.format_ue8m0 = True
        restore_scale_checkpoint_state(scale)

        self.assertEqual(scale.shape, (3, 4))
        self.assertEqual(scale.dtype, torch.float32)
        self.assertFalse(scale.format_ue8m0)
        self.assertEqual(
            scale._runtime_buffer.data_ptr(),
            runtime_buffer.data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
