"""Unit tests for the weight-delta prepare process and its pinned staging."""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from versions import raw_tensor, write_version

from sglang.srt.weight_sync.weight_delta.map import CanonicalTensor, Place, _table
from sglang.srt.weight_sync.weight_delta.prepare import PrepareProcess
from sglang.test.test_utils import CustomTestCase

BIG = 300 << 20
LOADED = {"a.bias": ("BF16", (4,)), "b.scale": ("F32", (2,)), "c.bias": ("U8", (BIG,))}
VALUES = {
    "a.bias": bytes(range(8)),
    "b.scale": bytes(range(100, 108)),
    "c.bias": np.random.default_rng(0).integers(0, 256, BIG, dtype=np.uint8).tobytes(),
}


def _table_of_loaded():
    inventory = {
        name: CanonicalTensor(name, dtype, shape, Path("/dev/null"))
        for name, (dtype, shape) in LOADED.items()
    }
    entries = {
        name: (
            Place((slice(None),), torch.empty(shape, dtype=torch.uint8, device="meta")),
        )
        for name, (_, shape) in LOADED.items()
    }
    return _table(inventory, entries)


def _write_version(directory: Path, names: list[str], base_version: int = 4) -> Path:
    for path in directory.iterdir():
        path.unlink()
    tensors = [raw_tensor(name, *LOADED[name], VALUES[name]) for name in names]
    return write_version(directory, [tensors], base_version)


def _staged_bytes(names: list[str]) -> bytes:
    """The raw values back to back, each starting on a 16-byte boundary."""
    staged = b""
    for name in names:
        staged += bytes(-len(staged) % 16) + VALUES[name]
    return staged


class TestPrepareProcess(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.folder.name) / "weight_v000005"
        cls.directory.mkdir()
        cls.process = PrepareProcess(_table_of_loaded(), threads=2)

    @classmethod
    def tearDownClass(cls):
        cls.process.close()
        cls.folder.cleanup()

    def _staged_on_gpu(self, names: list[str]) -> bytes:
        prepared, staged = self.process.prepare(
            _write_version(self.directory, names), "4"
        )
        self.assertTrue(staged.is_pinned())
        self.assertEqual(prepared.staged_end, staged.numel())
        on_gpu = staged.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        return bytes(on_gpu.cpu().numpy())

    def test_the_process_stages_a_version_into_pinned_memory(self):
        self.assertEqual(
            self._staged_on_gpu(["a.bias", "b.scale"]),
            _staged_bytes(["a.bias", "b.scale"]),
        )

    def test_staging_that_grows_is_pinned_again(self):
        self.assertEqual(
            self._staged_on_gpu(["a.bias", "c.bias"]),
            _staged_bytes(["a.bias", "c.bias"]),
        )
        self.assertEqual(self._staged_on_gpu(["b.scale"]), _staged_bytes(["b.scale"]))

    def test_a_refusal_comes_back_as_the_same_error(self):
        with self.assertRaisesRegex(ValueError, "the delta starts at version 3"):
            self.process.prepare(
                _write_version(self.directory, ["a.bias"], base_version=3), "4"
            )

    def test_the_process_stops_with_close(self):
        process = PrepareProcess(_table_of_loaded(), threads=1)
        process.close()
        process._process.join(timeout=30)
        self.assertFalse(process._process.is_alive())


if __name__ == "__main__":
    unittest.main()
