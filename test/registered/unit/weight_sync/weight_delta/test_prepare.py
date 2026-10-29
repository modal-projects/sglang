"""Unit tests for staging this GPU's part of a weight-delta version."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import tempfile
import unittest
from pathlib import Path

import torch
from versions import raw_tensor, write_version, xor_tensor

from sglang.srt.weight_sync.weight_delta.map import (
    OTHER_GPU,
    CanonicalTensor,
    Ignored,
    Place,
    _table,
)
from sglang.srt.weight_sync.weight_delta.prepare import (
    STAGING_GRANULE,
    Staging,
    prepare,
)
from sglang.test.test_utils import CustomTestCase

INNER = bytes(range(256)) * 3
RAW_BIAS = b"\x01\x02\x03\x04\x05\x06\x07\x08"
RAW_SCALE = b"\x10\x20\x30\x40\x50\x60\x70\x80"
# name -> dtype, shape, torch dtype of its place (None: another GPU's, a string: ignored)
LOADED = {
    "a.bias": ("BF16", (4,), torch.bfloat16),
    "b.weight": ("U8", (8, 1024), torch.uint8),
    "c.weight": ("U8", (4, 4), None),
    "d.scale": ("F32", (2,), torch.float32),
    "e.scale": ("F8_E4M3", (16, 16), torch.float32),
    "f.input_scale": ("F32", (), "activation calibration, never trained"),
}


def _raw(name: str, data: bytes, **changes) -> dict:
    dtype, shape, _ = LOADED[name]
    return raw_tensor(name, dtype, shape, data) | changes


def _xor(name: str, inner: bytes, chunk_nbytes: int = 1 << 20) -> dict:
    dtype, shape, _ = LOADED[name]
    frames = [
        {"offset": 0, "nbytes": 8192, "inner_offset": 0, "inner_nbytes": len(inner)}
    ]
    return xor_tensor(name, dtype, shape, frames, inner, chunk_nbytes)


class TestPrepare(CustomTestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name) / "weight_v000005"
        self.directory.mkdir()
        inventory, entries = {}, {}
        for name, (dtype, shape, place) in LOADED.items():
            inventory[name] = CanonicalTensor(name, dtype, shape, Path("/dev/null"))
            if place is None:
                entries[name] = OTHER_GPU
            elif isinstance(place, str):
                entries[name] = Ignored(place)
            else:
                dst = torch.empty(shape, dtype=place, device="meta")
                entries[name] = (Place((slice(None),) * len(shape), dst),)
        self.table = _table(inventory, entries)
        self.entries = entries
        self.staging = Staging(os.memfd_create("weight-delta-staging"))
        self.addCleanup(os.close, self.staging.fd)

    def _prepare(self, owners: list[list[dict]], base_version: int = 4):
        manifest = write_version(self.directory, owners, base_version)
        return prepare(self.table, manifest, "4", self.staging, 4)

    def test_this_gpus_tensors_are_staged_back_to_back(self):
        prepared = self._prepare(
            [
                [
                    _raw("a.bias", RAW_BIAS),
                    _xor("b.weight", INNER, chunk_nbytes=500),
                ],
                [_xor("c.weight", b"other gpu"), _raw("d.scale", RAW_SCALE)],
            ]
        )

        self.assertEqual((prepared.base_version, prepared.target_version), (4, 5))
        self.assertEqual(
            [self.table.entries[row] for row in prepared.table_rows],
            [self.entries["a.bias"], self.entries["d.scale"], self.entries["b.weight"]],
        )
        self.assertEqual(prepared.raw.tolist(), [True, True, False])
        self.assertEqual(prepared.nbytes.tolist(), [8, 8, 8192])
        self.assertEqual(prepared.staged_offset.tolist(), [0, 16, 32])
        self.assertEqual(prepared.staged_nbytes.tolist(), [8, 8, 768])
        self.assertEqual(prepared.staged_end, 800)
        padding = bytes(8)
        self.assertEqual(
            os.pread(self.staging.fd, 800, 0),
            RAW_BIAS + padding + RAW_SCALE + padding + INNER,
        )
        self.assertEqual(prepared.frame_end.tolist(), [0, 0, 1])
        self.assertEqual(
            {key: column.tolist() for key, column in prepared.frames.items()},
            {
                "offset": [0],
                "nbytes": [8192],
                "inner_offset": [0],
                "inner_nbytes": [768],
            },
        )
        self.assertEqual(
            (prepared.stats["tensors"], prepared.stats["held"], prepared.stats["raw"]),
            (4, 3, 2),
        )

    def test_a_version_this_gpu_takes_nothing_from_stages_nothing(self):
        prepared = self._prepare([[_xor("c.weight", b"other gpu")]])
        self.assertEqual(
            (len(prepared.table_rows), int(prepared.staged_nbytes.sum())), (0, 0)
        )

    def test_a_version_that_does_not_fit_the_engine_is_refused(self):
        refused = {
            "starts at version 3": ([[_raw("a.bias", RAW_BIAS)]], 3),
            "not in the loaded checkpoint": (
                [[_raw("a.bias", RAW_BIAS) | {"name": "z.bias"}]],
                4,
            ),
            "appear twice": (
                [[_raw("a.bias", RAW_BIAS)], [_raw("a.bias", RAW_BIAS)]],
                4,
            ),
            "have another dtype": ([[_raw("a.bias", RAW_BIAS, dtype="F16")]], 4),
            "have another shape": ([[_raw("a.bias", RAW_BIAS, shape=(2, 2))]], 4),
            "are not applied \\(activation calibration": (
                [[_raw("f.input_scale", RAW_BIAS[:4])]],
                4,
            ),
            "must arrive raw": ([[_xor("e.scale", INNER)]], 4),
            "arrive raw with another size": ([[_raw("d.scale", RAW_SCALE[:4])]], 4),
        }
        for problem, (owners, base_version) in refused.items():
            with self.subTest(problem=problem):
                for path in self.directory.iterdir():
                    path.unlink()
                with self.assertRaisesRegex(ValueError, problem):
                    self._prepare(owners, base_version)


class TestStaging(CustomTestCase):
    def test_staging_grows_with_headroom_and_never_shrinks(self):
        staging = Staging(os.memfd_create("weight-delta-staging"))
        self.addCleanup(os.close, staging.fd)

        self.assertEqual(len(staging.fit(100 << 20)), 100 << 20)
        self.assertEqual(staging.capacity, 2 * STAGING_GRANULE)
        staging.fit(10 << 20)
        self.assertEqual(staging.capacity, 2 * STAGING_GRANULE)
        staging.fit(staging.capacity + 1)
        self.assertEqual(staging.capacity, 3 * STAGING_GRANULE)
        self.assertEqual(os.fstat(staging.fd).st_size, 3 * STAGING_GRANULE)


if __name__ == "__main__":
    unittest.main()
