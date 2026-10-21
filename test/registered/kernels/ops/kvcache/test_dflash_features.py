import unittest

import torch

from sglang.kernels.ops.kvcache.dflash_features import capture_dflash_features
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.dflash_features import capture_dflash_features_reference
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestDFlashFeatures(CustomTestCase):
    def test_owned_copy_and_checkpoint_metadata(self):
        for dtype in (torch.float32, torch.bfloat16):
            for features in (17, 12288):
                for checkpoints in (False, True):
                    with self.subTest(
                        dtype=dtype, features=features, checkpoints=checkpoints
                    ):
                        hidden = torch.randn(
                            (32, 16, features * 2), device="cuda", dtype=dtype
                        )[..., ::2]
                        starts = (
                            torch.arange(32, device="cuda", dtype=torch.int64) + 2**33
                        )
                        commits = (
                            torch.arange(32, device="cuda", dtype=torch.int32) % 16 + 1
                        )
                        steps = commits - 1 if checkpoints else None
                        indices = (
                            torch.arange(32, device="cuda", dtype=torch.int32)
                            if checkpoints
                            else None
                        )
                        if checkpoints:
                            steps[::3] = -1
                            indices[1::4] = -1
                        expected = capture_dflash_features_reference(
                            hidden, starts, commits, steps, indices
                        )
                        actual = capture_dflash_features(
                            hidden, starts, commits, steps, indices
                        )
                        hidden.fill_(float("nan"))
                        for result, reference in zip(actual, expected, strict=True):
                            torch.testing.assert_close(
                                result, reference, rtol=0, atol=0
                            )
                        self.assertTrue(actual[0].is_contiguous())


if __name__ == "__main__":
    unittest.main()
