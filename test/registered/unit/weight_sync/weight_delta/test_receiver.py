"""Unit tests for the scheduler's side of weight delta: prepare while serving, apply in the pause."""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from versions import (
    gap_frames,
    raw_tensor,
    region_checksum,
    write_version,
    xor_tensor,
)

from sglang.srt.managers.io_struct import (
    PrepareWeightsFromDeltaReqInput,
    UpdateWeightsFromDeltaReqInput,
)
from sglang.srt.weight_sync.weight_delta import receiver as receiver_module
from sglang.srt.weight_sync.weight_delta.map import (
    TORCH_DTYPES,
    CanonicalTensor,
    DeltaMap,
    Place,
    _table,
)
from sglang.srt.weight_sync.weight_delta.receiver import Receiver
from sglang.test.test_utils import CustomTestCase

# name -> dtype, shape
CANONICAL = {
    "w.a": ("U8", (64, 4096)),
    "w.b": ("BF16", (128, 512)),
    "n.bias": ("BF16", (256,)),
}


class _WeightUpdater:
    def __init__(self) -> None:
        self.versions = []

    def record_weight_version_after_update(self, version: str) -> None:
        self.versions.append(version)

    def flush_cache_after_weight_update(self, req, success: bool, message: str):
        return success, message


def _bytes_of(seed: int) -> dict[str, bytes]:
    rng = np.random.default_rng(seed)
    return {
        name: rng.integers(0, 256, int(np.prod(shape)) * TORCH_DTYPES[dtype].itemsize)
        .astype(np.uint8)
        .tobytes()
        for name, (dtype, shape) in CANONICAL.items()
    }


def _canonical(name: str, data: bytes) -> torch.Tensor:
    dtype, shape = CANONICAL[name]
    return (
        torch.frombuffer(bytearray(data), dtype=torch.uint8)
        .view(TORCH_DTYPES[dtype])
        .reshape(shape)
    )


class TestReceiver(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.patches = [
            (receiver_module, "get_serving", receiver_module.get_serving),
            (receiver_module, "get_parallel", receiver_module.get_parallel),
        ]
        cls.serving_state = SimpleNamespace(weight_version="4")
        receiver_module.get_serving = lambda: cls.serving_state
        receiver_module.get_parallel = lambda: SimpleNamespace(tp_size=1)
        cuda = torch.device("cuda", 0)
        cls.serving = {
            "w.a": torch.zeros((64, 4096), dtype=torch.uint8, device=cuda),
            "w.b": torch.zeros((64, 512), dtype=torch.bfloat16, device=cuda),
            "n.bias": torch.zeros(256, dtype=torch.bfloat16, device=cuda),
        }
        cls.sources = {
            "w.a": (slice(None), slice(None)),
            "w.b": (slice(32, 96), slice(None)),
            "n.bias": (slice(None),),
        }
        inventory = {
            name: CanonicalTensor(name, dtype, shape, Path("/dev/null"))
            for name, (dtype, shape) in CANONICAL.items()
        }
        entries = {
            name: (Place(cls.sources[name], cls.serving[name]),) for name in CANONICAL
        }
        cls.replies = []
        cls.receiver = Receiver(
            DeltaMap(inventory, entries, [], _table(inventory, entries)),
            "gap-zstd",
            cuda,
            lambda output, req: cls.replies.append(output),
        )
        cls.folder = tempfile.TemporaryDirectory()
        cls.versions = 0

    @classmethod
    def tearDownClass(cls):
        cls.receiver.process.close()
        cls.folder.cleanup()
        for module, name, value in cls.patches:
            setattr(module, name, value)

    def setUp(self):
        self.serving_state.weight_version = "4"
        self.base = _bytes_of(0)
        for name, tensor in self.serving.items():
            tensor.copy_(_canonical(name, self.base[name])[self.sources[name]])

    def _version(self, current, base_version=4, codec="gap-zstd") -> str:
        tensors = []
        for name, (dtype, shape) in CANONICAL.items():
            itemsize, region = TORCH_DTYPES[dtype].itemsize, self.sources[name]
            if name == "n.bias":
                expected = region_checksum(None, current[name], itemsize, shape, region)
                tensors.append(raw_tensor(name, dtype, shape, current[name], expected))
            else:
                expected = region_checksum(
                    self.base[name], current[name], itemsize, shape, region
                )
                frames, inner = gap_frames(self.base[name], current[name])
                tensors.append(
                    xor_tensor(name, dtype, shape, frames, inner, expected=expected)
                )
        type(self).versions += 1
        directory = Path(self.folder.name) / f"weight_v{self.versions:06d}"
        return str(write_version(directory, [tensors], base_version, codec))

    def _prepare(self, manifest: str):
        request = PrepareWeightsFromDeltaReqInput(manifest=manifest)
        self.assertIsNone(self.receiver.prepare(request))
        return self._reply()

    def _reply(self):
        """Run the scheduler loop's check until the prepare replies."""
        deadline = time.monotonic() + 120
        while not self.replies and time.monotonic() < deadline:
            self.receiver.check_pending()
            time.sleep(0.01)
        return self.replies.pop()

    def _update(self, manifest: str, weight_version: str, updater=None):
        request = UpdateWeightsFromDeltaReqInput(
            manifest=manifest, weight_version=weight_version
        )
        return self.receiver.update(request, updater or _WeightUpdater())

    def _changed(self) -> dict[str, bytes]:
        rng = np.random.default_rng(1)
        current = {}
        for name, data in self.base.items():
            flat = np.frombuffer(data, dtype=np.uint8).copy()
            flat[rng.choice(len(flat), size=len(flat) // 50, replace=False)] ^= 0x5A
            current[name] = flat.tobytes()
        return current

    def test_a_prepared_version_applies_and_records_its_version(self):
        current = self._changed()
        manifest = self._version(current)

        prepared = self._prepare(manifest)
        updater = _WeightUpdater()
        applied = self._update(manifest, "5", updater)

        self.assertTrue(prepared.success, prepared.message)
        self.assertGreater(prepared.stats["staging_capacity"], 0)
        self.assertIn("plan_s", prepared.stats)
        self.assertTrue(applied.success, applied.message)
        self.assertIn("gpu_s", applied.stats)
        self.assertEqual(updater.versions, ["5"])
        for name, tensor in self.serving.items():
            expected = _canonical(name, current[name])[self.sources[name]]
            self.assertTrue(
                torch.equal(
                    tensor.cpu().contiguous().view(torch.uint8),
                    expected.contiguous().view(torch.uint8),
                ),
                name,
            )

    def test_an_update_needs_its_own_prepared_version(self):
        manifest = self._version(self._changed())
        refused = self._update(manifest, "5")
        self.assertFalse(refused.success)
        self.assertIn("is not prepared", refused.message)

        self.assertTrue(self._prepare(manifest).success)
        refusals = {
            "reaches version 5, not 6": (manifest, "6", "4"),
            "is not prepared": (manifest + ".other", "5", "4"),
            "the engine moved to version 7": (manifest, "5", "7"),
        }
        for message, (path, version, engine) in refusals.items():
            with self.subTest(refusal=message):
                self.serving_state.weight_version = engine
                output = self._update(path, version)
                self.assertFalse(output.success)
                self.assertIn(message, output.message)
        self.serving_state.weight_version = "4"
        self.assertTrue(self._update(manifest, "5").success)

    def test_a_prepare_while_one_runs_is_refused(self):
        manifest = self._version(self._changed())
        self.assertIsNone(
            self.receiver.prepare(PrepareWeightsFromDeltaReqInput(manifest=manifest))
        )
        second = self.receiver.prepare(
            PrepareWeightsFromDeltaReqInput(manifest=manifest)
        )
        self.assertFalse(second.success)
        self.assertIn("already running", second.message)
        self.assertTrue(self._reply().success)

    def test_a_refused_prepare_replies_with_the_reason(self):
        current = self._changed()
        refusals = {
            "the delta starts at version 3": self._version(current, base_version=3),
            "the delta is snappy-zstd": self._version(current, codec="snappy-zstd"),
        }
        for message, manifest in refusals.items():
            with self.subTest(refusal=message):
                output = self._prepare(manifest)
                self.assertFalse(output.success)
                self.assertIn(message, output.message)


if __name__ == "__main__":
    unittest.main()
