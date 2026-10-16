import unittest

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import _mamba_host_sizing
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMambaHostRatio(CustomTestCase):
    def test_unset_keeps_hicache_sizing(self):
        self.assertEqual(_mamba_host_sizing(2.0, 64), (2.0, 64))

    def test_override_replaces_ratio_and_size(self):
        with envs.SGLANG_HICACHE_MAMBA_HOST_RATIO.override(0.5):
            self.assertEqual(_mamba_host_sizing(2.0, 64), (0.5, 0))


if __name__ == "__main__":
    unittest.main()
