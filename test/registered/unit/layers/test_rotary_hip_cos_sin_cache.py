import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHipCosSinCacheCast(CustomTestCase):
    def test_cast_is_cached_and_master_stays_fp32(self):
        rope = SimpleNamespace(cos_sin_cache=torch.randn(16, 8))
        first = RotaryEmbedding._hip_cos_sin_cache_as(rope, torch.bfloat16)
        second = RotaryEmbedding._hip_cos_sin_cache_as(rope, torch.bfloat16)
        self.assertEqual(first.dtype, torch.bfloat16)
        self.assertIs(first, second)
        self.assertEqual(rope.cos_sin_cache.dtype, torch.float32)
        self.assertIs(
            RotaryEmbedding._hip_cos_sin_cache_as(rope, torch.float32),
            rope.cos_sin_cache,
        )


if __name__ == "__main__":
    unittest.main()
