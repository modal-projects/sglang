"""gpu_tensor_hash reaches every word of a tensor past 2**31 words."""

import unittest

import torch

from sglang.kernels.ops.memory.gpu_tensor_hash import gpu_tensor_hash
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestGpuTensorHash(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def test_last_word_of_a_tensor_past_2_31_words_counts(self):
        words = 2**31 + 4096
        tensor = torch.zeros(words, dtype=torch.int32, device="cuda")
        before = gpu_tensor_hash(tensor)
        tensor[-1] = 1
        self.assertNotEqual(gpu_tensor_hash(tensor), before)


if __name__ == "__main__":
    unittest.main()
