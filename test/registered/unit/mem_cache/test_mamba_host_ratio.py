import os
import tempfile
import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _drop_checkpoint_page_cache,
    _mamba_host_sizing,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMambaHostRatio(CustomTestCase):
    def test_unset_keeps_hicache_sizing(self):
        self.assertEqual(_mamba_host_sizing(2.0, 64), (2.0, 64))

    def test_override_replaces_ratio_and_size(self):
        with envs.SGLANG_HICACHE_MAMBA_HOST_RATIO.override(0.5):
            self.assertEqual(_mamba_host_sizing(2.0, 64), (0.5, 0))


class TestDropCheckpointPageCache(CustomTestCase):
    def _run(self, enabled):
        with tempfile.TemporaryDirectory() as d:
            for name in ("a.safetensors", "b.safetensors", "config.json"):
                open(os.path.join(d, name), "w").close()
            with mock.patch("os.posix_fadvise") as fadvise:
                with envs.SGLANG_HICACHE_DROP_CHECKPOINT_PAGE_CACHE.override(enabled):
                    _drop_checkpoint_page_cache(d)
            return fadvise.call_count

    def test_off_by_default(self):
        self.assertEqual(self._run(False), 0)

    def test_drops_only_safetensors(self):
        self.assertEqual(self._run(True), 2)


if __name__ == "__main__":
    unittest.main()
