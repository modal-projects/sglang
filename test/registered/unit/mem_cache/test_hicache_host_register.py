import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host.common import _cuda_host_register
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeBuffer:
    def __init__(self, base: int, size: int):
        self._base = base
        self._size = size

    def data_ptr(self) -> int:
        return self._base

    def numel(self) -> int:
        return self._size

    def element_size(self) -> int:
        return 1


class _FakeCudart:
    def __init__(self):
        self.registrations = []

    def cudaHostRegister(self, ptr: int, size: int, flags: int) -> int:
        self.registrations.append((ptr, size, flags))
        return 0


class TestHiCacheHostRegister(unittest.TestCase):
    def test_registration_boundaries_honor_page_copy_granularity(self):
        mib = 1024**2
        gib = 1024**3
        base = 0x10000000
        total = 2500 * mib
        page_copy_bytes = 300 * mib
        cudart = _FakeCudart()

        with (
            mock.patch.object(
                envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB,
                "get",
                return_value=1,
            ),
            mock.patch.object(torch.cuda, "cudart", return_value=cudart),
        ):
            _cuda_host_register(
                _FakeBuffer(base, total),
                registration_granularity_bytes=page_copy_bytes,
            )

        aligned_chunk = 900 * mib
        self.assertLessEqual(aligned_chunk, gib)
        self.assertEqual(
            cudart.registrations,
            [
                (base, aligned_chunk, 0),
                (base + aligned_chunk, aligned_chunk, 0),
                (base + 2 * aligned_chunk, 700 * mib, 0),
            ],
        )
        for ptr, _, _ in cudart.registrations:
            self.assertEqual((ptr - base) % page_copy_bytes, 0)


if __name__ == "__main__":
    unittest.main()
