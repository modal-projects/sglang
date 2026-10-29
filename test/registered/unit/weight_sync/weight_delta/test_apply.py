"""Unit tests for applying a staged weight-delta version inside the pause."""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from versions import gap_frames, raw_tensor, snappy_frames, write_version, xor_tensor

from sglang.srt.weight_sync.weight_delta import apply as apply_module
from sglang.srt.weight_sync.weight_delta.apply import Applier
from sglang.srt.weight_sync.weight_delta.map import (
    TORCH_DTYPES,
    CanonicalTensor,
    DeltaMap,
    Place,
    _table,
)
from sglang.srt.weight_sync.weight_delta.prepare import PrepareProcess
from sglang.test.test_utils import CustomTestCase

# name -> dtype, shape; raw ones arrive whole
CANONICAL = {
    "w.whole": ("U8", (64, 2048)),
    "w.rows": ("BF16", (128, 1024)),
    "w.cols": ("BF16", (64, 512)),
    "w.frames": ("U8", (2, 600_000)),
    "e.0": ("U8", (256, 512)),
    "e.1": ("U8", (256, 512)),
    "w.strided": ("U8", (64, 256)),
    "n.bias": ("BF16", (1024,)),
    "s.scale": ("F8_E4M3", (16,)),
    "s.f32": ("F32", (4, 8)),
}
RAW = ("n.bias", "s.scale", "s.f32")


def _serving_and_places():
    """Serving tensors, and each canonical tensor's places among them, one of each kind."""
    cuda = torch.device("cuda")
    serving = {
        "whole": torch.zeros((64, 2048), dtype=torch.uint8, device=cuda),
        "rows": torch.zeros((64, 1024), dtype=torch.bfloat16, device=cuda),
        "cols": torch.zeros((64, 256), dtype=torch.bfloat16, device=cuda),
        "frames": torch.zeros((2, 600_000), dtype=torch.uint8, device=cuda),
        "experts": torch.zeros((2, 256, 512), dtype=torch.uint8, device=cuda),
        "strided": torch.zeros((64, 512), dtype=torch.uint8, device=cuda),
        "bias": torch.zeros(1024, dtype=torch.bfloat16, device=cuda),
        "scale": torch.zeros(16, dtype=torch.float32, device=cuda),
        "f32": torch.zeros((4, 4), dtype=torch.float32, device=cuda),
    }
    everything = (slice(None),)
    places = {
        "w.whole": [Place(everything * 2, serving["whole"])],
        "w.rows": [Place((slice(32, 96), slice(None)), serving["rows"])],
        "w.cols": [Place((slice(None), slice(128, 384)), serving["cols"])],
        "w.frames": [Place(everything * 2, serving["frames"])],
        "e.0": [Place(everything * 2, serving["experts"][0])],
        "e.1": [Place(everything * 2, serving["experts"][1])],
        "w.strided": [Place(everything * 2, serving["strided"][:, 128:384])],
        "n.bias": [Place(everything, serving["bias"])],
        "s.scale": [Place(everything, serving["scale"])],
        "s.f32": [Place((slice(None), slice(2, 6)), serving["f32"])],
    }
    return serving, places


def _values(seed: int) -> dict[str, bytes]:
    rng = np.random.default_rng(seed)
    values = {}
    for name, (dtype, shape) in CANONICAL.items():
        data = rng.integers(0, 256, int(np.prod(shape)) * TORCH_DTYPES[dtype].itemsize)
        # Keep FP8 clear of NaN, which compares unequal to itself after the cast.
        values[name] = (data & 0x77 if dtype == "F8_E4M3" else data).astype(np.uint8)
    return {name: data.tobytes() for name, data in values.items()}


def _changed(values: dict[str, bytes], seed: int) -> dict[str, bytes]:
    """Every tensor with about 1% of its bytes changed."""
    rng = np.random.default_rng(seed)
    changed = {}
    for name, data in values.items():
        flat = np.frombuffer(data, dtype=np.uint8).copy()
        positions = rng.choice(len(flat), size=max(1, len(flat) // 100), replace=False)
        flat[positions] ^= rng.integers(1, 256, len(positions), dtype=np.uint8)
        if CANONICAL[name][0] == "F8_E4M3":
            flat &= 0x77
        changed[name] = flat.tobytes()
    return changed


def _canonical(name: str, data: bytes, device="cpu") -> torch.Tensor:
    dtype, shape = CANONICAL[name]
    return (
        torch.frombuffer(bytearray(data), dtype=torch.uint8)
        .view(TORCH_DTYPES[dtype])
        .reshape(shape)
        .to(device)
    )


class TestApply(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.serving, cls.places = _serving_and_places()
        inventory = {
            name: CanonicalTensor(name, dtype, shape, Path("/dev/null"))
            for name, (dtype, shape) in CANONICAL.items()
        }
        entries = {name: tuple(places) for name, places in cls.places.items()}
        cls.delta_map = DeltaMap(inventory, entries, [], _table(inventory, entries))
        cls.process = PrepareProcess(cls.delta_map.table, threads=2)
        cls.versions = 0

    @classmethod
    def tearDownClass(cls):
        cls.process.close()
        cls.folder.cleanup()

    def setUp(self):
        self.base = _values(0)
        for name, places in self.places.items():
            canonical = _canonical(name, self.base[name], "cuda")
            for place in places:
                place.dst.copy_(canonical[place.src])
        self.before = {key: tensor.cpu() for key, tensor in self.serving.items()}

    def _apply(self, applier, codec, previous, current, encode=None) -> dict:
        encode = encode or (gap_frames if codec == "gap-zstd" else snappy_frames)
        owners = [[], []]
        for index, (name, (dtype, shape)) in enumerate(CANONICAL.items()):
            if name in RAW:
                tensor = raw_tensor(name, dtype, shape, current[name])
            else:
                frames, inner = encode(previous[name], current[name])
                tensor = xor_tensor(name, dtype, shape, frames, inner)
            owners[index % 2].append(tensor)
        type(self).versions += 1
        directory = Path(self.folder.name) / f"weight_v{self.versions:06d}"
        manifest = write_version(directory, owners, 4, codec)
        prepared, staged = self.process.prepare(manifest, "4")
        return applier.apply(applier.plan(prepared), staged)

    def _assert_serving(self, values: dict[str, bytes]) -> None:
        expected = {key: tensor.clone() for key, tensor in self.before.items()}
        for name, places in self.places.items():
            canonical = _canonical(name, values[name])
            for place in places:
                owner = next(
                    key
                    for key, tensor in self.serving.items()
                    if tensor.untyped_storage().data_ptr()
                    == place.dst.untyped_storage().data_ptr()
                )
                view = expected[owner].as_strided(
                    place.dst.size(), place.dst.stride(), place.dst.storage_offset()
                )
                view.copy_(canonical[place.src])
        for key, tensor in self.serving.items():
            with self.subTest(serving=key):
                self.assertTrue(
                    torch.equal(
                        tensor.cpu().contiguous().view(torch.uint8),
                        expected[key].contiguous().view(torch.uint8),
                    )
                )

    def test_a_gap_version_lands_bit_exact(self):
        current = _changed(self.base, 1)
        applier = Applier(self.delta_map, "gap-zstd", torch.device("cuda", 0))
        stats = self._apply(applier, "gap-zstd", self.base, current)
        self._assert_serving(current)
        self.assertEqual(stats["raw"], 3)

    def test_a_snappy_version_lands_bit_exact(self):
        current = _changed(self.base, 2)
        # An unchanged second frame: the version omits it, so its slot bytes must be zeroed.
        frame = 1 << 20
        current["w.frames"] = (
            current["w.frames"][:frame] + self.base["w.frames"][frame:]
        )
        applier = Applier(self.delta_map, "snappy-zstd", torch.device("cuda", 0))
        self._apply(applier, "snappy-zstd", self.base, current)
        self._assert_serving(current)

    def test_batches_split_across_slots_and_plans_are_reused(self):
        slot_nbytes = apply_module.SLOT_NBYTES
        apply_module.SLOT_NBYTES = 64 << 10
        self.addCleanup(setattr, apply_module, "SLOT_NBYTES", slot_nbytes)
        applier = Applier(self.delta_map, "gap-zstd", torch.device("cuda", 0))
        first = _changed(self.base, 3)
        stats = self._apply(applier, "gap-zstd", self.base, first)
        self.assertGreater(stats["batches"], 2)
        plans = len(applier._xor_plans)
        second = _changed(first, 4)
        self._apply(applier, "gap-zstd", first, second)
        self._assert_serving(second)
        self.assertEqual(len(applier._xor_plans), plans)

    def test_a_frame_that_decodes_past_its_end_raises(self):
        def past_the_end(previous: bytes, current: bytes):
            frames, inner = gap_frames(previous, current)
            broken = bytearray(inner)
            # The first change's gap, all three planes at 0xFF: far past a 1 MiB frame.
            changes = frames[0]["inner_nbytes"] // 4
            for plane in range(3):
                broken[frames[0]["inner_offset"] + plane * changes] = 0xFF
            return frames, bytes(broken)

        applier = Applier(self.delta_map, "gap-zstd", torch.device("cuda", 0))
        with self.assertRaisesRegex(RuntimeError, "failed to decode"):
            self._apply(
                applier, "gap-zstd", self.base, _changed(self.base, 5), past_the_end
            )


if __name__ == "__main__":
    unittest.main()
