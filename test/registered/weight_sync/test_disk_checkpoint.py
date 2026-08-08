"""CPU unit tests for weight_sync/disk_checkpoint.py.

Covers seeding, delta-chain replay, recovery from torn or corrupted local
state, rollback, and fail-loud behavior for invalid published deltas.

Checkpoints are hand-crafted safetensors files with the `adler32` checksum
format so the tests run in a CPU environment without optional hash packages.
"""

import contextlib
import json
import os
import struct
import tempfile
import unittest
import zlib
from unittest import mock

import numpy as np
import zstandard

from sglang.srt.weight_sync import disk_checkpoint
from sglang.srt.weight_sync.checksum import calculate_checksum, create_checksum
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ChecksumTest(unittest.TestCase):
    def test_supported_algorithms_are_incremental(self):
        for algorithm in ("xxh3-128", "blake3", "adler32"):
            checksum = create_checksum(algorithm)
            checksum.update(b"checkpoint ")
            checksum.update(b"bytes")
            self.assertEqual(
                checksum.hexdigest(),
                calculate_checksum(algorithm, b"checkpoint bytes"),
            )

    def test_unsupported_algorithm_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "unsupported checksum algorithm"):
            create_checksum("unknown")

    def test_checkpoint_source_refresh_hook_receives_target(self):
        hook = mock.Mock()
        with mock.patch.object(
            disk_checkpoint,
            "dynamic_import",
            return_value=hook,
        ) as dynamic_import:
            disk_checkpoint.refresh_checkpoint_source("/updates", 3, "hooks.refresh")

        dynamic_import.assert_called_once_with("hooks.refresh")
        hook.assert_called_once_with("/updates", 3)


def write_safetensors(path, tensors, metadata=None):
    """tensors: {name: bytes}. Minimal safetensors writer (U8 payloads)."""
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset = 0
    for name, data in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [len(data)],
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    encoded = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for data in tensors.values():
            f.write(data)


def adler32_hex(data) -> str:
    return f"{zlib.adler32(bytes(data), 1):08x}"


class Publisher:
    """Builds a base checkpoint plus an XOR delta chain the way the trainer
    does: each version dir carries zstd-compressed per-tensor XOR diffs with
    checksums of the new state in the safetensors metadata."""

    SHARD = "model-00001-of-00001.safetensors"

    def __init__(self, root):
        self.base_dir = os.path.join(root, "base")
        self.source_dir = os.path.join(root, "published")
        os.makedirs(self.base_dir)
        os.makedirs(self.source_dir)
        rng = np.random.default_rng(7)
        self.state = {
            "layer.a": rng.integers(0, 256, 4096, dtype=np.uint8).tobytes(),
            "layer.b": rng.integers(0, 256, 2048, dtype=np.uint8).tobytes(),
        }
        self.versions = {0: dict(self.state)}
        write_safetensors(os.path.join(self.base_dir, self.SHARD), self.state)

    def publish_delta(self, version, changed):
        """changed: {name: new_bytes}; unchanged tensors are omitted."""
        vdir = os.path.join(self.source_dir, f"weight_v{version:06d}")
        os.makedirs(vdir)
        payloads = {}
        checksums = {}
        for name, new in changed.items():
            old = self.state[name]
            diff = (
                np.frombuffer(new, dtype=np.uint8) ^ np.frombuffer(old, dtype=np.uint8)
            ).tobytes()
            payloads[name] = zstandard.ZstdCompressor().compress(diff)
            checksums[name] = adler32_hex(new)
            self.state[name] = new
        self.versions[version] = dict(self.state)
        write_safetensors(os.path.join(vdir, self.SHARD), payloads, metadata=checksums)
        with open(os.path.join(vdir, "model.safetensors.index.json"), "w") as f:
            json.dump(
                {
                    "metadata": {
                        "version": f"{version:06d}",
                        "base_version": f"{version - 1:06d}",
                        "delta_encoding": "xor",
                        "compression_format": "zstd",
                        "checksum_format": "adler32",
                    },
                    "weight_map": {name: self.SHARD for name in payloads},
                },
                f,
            )

    def publish_full(self, version):
        vdir = os.path.join(self.source_dir, f"weight_v{version:06d}")
        os.makedirs(vdir)
        state = {
            name: np.random.default_rng(40 + index)
            .integers(0, 256, len(value), dtype=np.uint8)
            .tobytes()
            for index, (name, value) in enumerate(self.state.items(), start=version)
        }
        write_safetensors(os.path.join(vdir, self.SHARD), state)
        with open(os.path.join(vdir, "model.safetensors.index.json"), "w") as file:
            json.dump(
                {
                    "metadata": {"version": f"{version:06d}"},
                    "weight_map": {name: self.SHARD for name in state},
                },
                file,
            )
        self.state = dict(state)
        self.versions[version] = dict(state)
        return vdir


def read_local(local_dir):
    path = os.path.join(local_dir, Publisher.SHARD)
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        body = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        out[name] = body[begin:end]
    return out


class MaterializeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = self._tmp.name
        self.pub = Publisher(root)
        self.local = os.path.join(root, "local")
        rng = np.random.default_rng(11)
        self.pub.publish_delta(
            1, {"layer.a": rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()}
        )
        self.pub.publish_delta(
            2, {"layer.b": rng.integers(0, 256, 2048, dtype=np.uint8).tobytes()}
        )

    def tearDown(self):
        self._tmp.cleanup()

    def materialize(self, target):
        return disk_checkpoint.materialize(
            self.local, self.pub.base_dir, self.pub.source_dir, target
        )

    def assert_at_version(self, version):
        self.assertEqual(read_local(self.local), self.pub.versions[version])
        self.assertEqual(disk_checkpoint._read_applied_version(self.local), version)

    @contextlib.contextmanager
    def spy_materialize(self):
        """Record full-checkpoint seeds and single-delta applications."""
        seeds, applies = [], []
        orig_reset = disk_checkpoint._reset_checkpoint
        orig_apply = disk_checkpoint._apply_delta

        def reset_spy(src, *a, **k):
            seeds.append(src)
            return orig_reset(src, *a, **k)

        def apply_spy(local, vdir, *a, **k):
            applies.append(vdir)
            return orig_apply(local, vdir, *a, **k)

        disk_checkpoint._reset_checkpoint = reset_spy
        disk_checkpoint._apply_delta = apply_spy
        try:
            yield seeds, applies
        finally:
            disk_checkpoint._reset_checkpoint = orig_reset
            disk_checkpoint._apply_delta = orig_apply

    def test_seed_and_chain(self):
        self.materialize(2)
        self.assert_at_version(2)

    def test_incremental_materialization_is_idempotent(self):
        stats = self.materialize(1)
        self.assertEqual(stats["apply"]["operation"], "apply_xor")
        self.assertEqual(stats["apply"]["io_backend"], "pread_pwrite")
        self.assertEqual(stats["apply"]["delta_tensors"], 1)
        self.assertEqual(stats["apply"]["target_tensor_bytes"], 4096)
        self.assertGreaterEqual(stats["apply"]["phases"]["apply_wall_s"], 0)
        self.assert_at_version(1)
        stats = self.materialize(1)  # no-op
        self.assertEqual(stats["operation"], "noop")
        self.assert_at_version(1)
        self.materialize(2)
        self.assert_at_version(2)

    def test_materialization_can_rollback_to_an_older_version(self):
        self.materialize(2)
        stats = self.materialize(1)
        self.assert_at_version(1)
        self.assertEqual(stats["operation"], "reseed_and_apply")

    def test_torn_apply_reseeds_instead_of_repatching(self):
        self.materialize(1)
        # Simulate an apply of v2 killed mid-mutation: marker still at 1, some
        # (not all) bytes of layer.b already XORed toward v2. The re-apply
        # double-XORs those bytes -> checksum mismatch -> reseed + replay.
        shard = os.path.join(self.local, Publisher.SHARD)
        locations = disk_checkpoint._tensor_locations(self.local)
        _, offset, nbytes = locations["layer.b"]
        with open(shard, "r+b") as f:
            f.seek(offset)
            f.write(self.pub.versions[2]["layer.b"][: nbytes // 2])
        self.materialize(2)
        self.assert_at_version(2)

    def test_corrupt_local_state_recovers_via_reseed(self):
        self.materialize(1)
        # Silent local divergence leaves the marker at v1. The v2 checksum
        # detects it and triggers a clean replay from the base.
        shard = os.path.join(self.local, Publisher.SHARD)
        locations = disk_checkpoint._tensor_locations(self.local)
        _, offset, _ = locations["layer.b"]
        with open(shard, "r+b") as f:
            f.seek(offset)
            f.write(bytes(16))
        self.materialize(2)
        self.assert_at_version(2)

    def test_bad_published_delta_fails_loud(self):
        # Corrupt the published v1 payload itself: reseed-and-replay hits the
        # same bad artifact and must raise, not serve bad weights.
        import shutil

        vdir = os.path.join(self.pub.source_dir, "weight_v000001")
        shard = os.path.join(vdir, Publisher.SHARD)
        with open(shard, "rb") as f:
            data = bytearray(f.read())
        data[-1] ^= 0xFF
        with open(shard, "wb") as f:
            f.write(bytes(data))
        with self.assertRaises(Exception):
            self.materialize(1)
        # The publisher fixes the artifact; the next attempt recovers by itself.
        shutil.rmtree(vdir)
        self.pub.state = dict(self.pub.versions[0])
        rng = np.random.default_rng(11)
        self.pub.publish_delta(
            1, {"layer.a": rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()}
        )
        self.materialize(1)
        self.assert_at_version(1)

    def test_delta_index_cannot_reference_a_missing_tensor(self):
        vdir = os.path.join(self.pub.source_dir, "weight_v000001")
        index_path = os.path.join(vdir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        index["weight_map"]["layer.b"] = Publisher.SHARD
        with open(index_path, "w") as f:
            json.dump(index, f)

        with self.assertRaisesRegex(ValueError, "blob/index tensor mismatch"):
            self.materialize(1)

    def test_missing_source_version_fails_fast_without_reseed(self):
        # A not-yet-visible source version (publisher/object store not caught
        # up) must raise FileNotFoundError WITHOUT reseeding: reseed can't
        # conjure the absent bytes, and for a large base the wasted full copy
        # is expensive. The caller retries once the source is visible.
        self.materialize(1)
        self.assert_at_version(1)
        reset_calls = []
        orig_reset = disk_checkpoint._reset_checkpoint

        def _spy(*args, **kwargs):
            reset_calls.append(args)
            return orig_reset(*args, **kwargs)

        disk_checkpoint._reset_checkpoint = _spy
        try:
            with self.assertRaises(FileNotFoundError):
                self.materialize(3)  # v3 never published (only v1, v2 exist)
        finally:
            disk_checkpoint._reset_checkpoint = orig_reset
        self.assertEqual(reset_calls, [], "must not reseed on a missing source version")
        # Local state is untouched: later materialization still works.
        self.materialize(2)
        self.assert_at_version(2)

    def test_source_version_without_manifest_is_not_treated_as_full(self):
        self.materialize(1)
        self.assert_at_version(1)
        vdir = os.path.join(self.pub.source_dir, "weight_v000003")
        os.makedirs(vdir)
        write_safetensors(
            os.path.join(vdir, Publisher.SHARD),
            {"layer.a": b"not a published full checkpoint"},
        )

        reset_calls = []
        orig_reset = disk_checkpoint._reset_checkpoint

        def _spy(*args, **kwargs):
            reset_calls.append(args)
            return orig_reset(*args, **kwargs)

        disk_checkpoint._reset_checkpoint = _spy
        try:
            with self.assertRaisesRegex(FileNotFoundError, "has no manifest"):
                self.materialize(3)
        finally:
            disk_checkpoint._reset_checkpoint = orig_reset
        self.assertEqual(reset_calls, [])
        self.assert_at_version(1)

    def test_manifest_version_must_match_published_directory(self):
        self.materialize(0)
        self.assert_at_version(0)
        index_path = os.path.join(
            self.pub.source_dir,
            "weight_v000001",
            "model.safetensors.index.json",
        )
        with open(index_path) as file:
            index = json.load(file)
        index["metadata"]["version"] = "000009"
        with open(index_path, "w") as file:
            json.dump(index, file)

        with self.assertRaisesRegex(ValueError, "version mismatch"):
            self.materialize(1)
        self.assert_at_version(0)

    def test_indexed_full_version_reseeds_the_local_checkpoint(self):
        vdir = self.pub.publish_full(3)

        with self.spy_materialize() as (seeds, applies):
            self.materialize(3)
        self.assert_at_version(3)
        self.assertEqual((seeds, applies), ([vdir], []))

    def test_interrupted_full_seed_invalidates_the_old_version(self):
        self.materialize(2)
        self.assert_at_version(2)
        self.pub.publish_full(3)
        write_version = disk_checkpoint._write_applied_version

        def fail_version_marker(local_checkpoint_dir, version):
            if version == 3:
                raise OSError("simulated crash before version marker")
            return write_version(local_checkpoint_dir, version)

        disk_checkpoint._write_applied_version = fail_version_marker
        try:
            with self.assertRaisesRegex(OSError, "simulated crash"):
                self.materialize(3)
        finally:
            disk_checkpoint._write_applied_version = write_version

        self.assertIsNone(disk_checkpoint._read_applied_version(self.local))
        self.materialize(2)
        self.assert_at_version(2)

    def test_interrupted_delta_invalidates_its_base_version(self):
        self.materialize(0)
        self.assert_at_version(0)
        write_version = disk_checkpoint._write_applied_version

        def fail_version_marker(local_checkpoint_dir, version):
            if version == 1:
                raise OSError("simulated crash before delta version marker")
            return write_version(local_checkpoint_dir, version)

        disk_checkpoint._write_applied_version = fail_version_marker
        try:
            with self.assertRaisesRegex(OSError, "simulated crash"):
                self.materialize(1)
        finally:
            disk_checkpoint._write_applied_version = write_version

        self.assertIsNone(disk_checkpoint._read_applied_version(self.local))
        self.materialize(0)
        self.assert_at_version(0)

    def test_incomplete_indexed_full_version_fails_before_copy(self):
        self.materialize(1)
        self.assert_at_version(1)
        vdir = os.path.join(self.pub.source_dir, "weight_v000003")
        os.makedirs(vdir)
        with open(os.path.join(vdir, "model.safetensors.index.json"), "w") as file:
            json.dump(
                {
                    "metadata": {"version": "000003"},
                    "weight_map": {"layer.a": Publisher.SHARD},
                },
                file,
            )

        reset_calls = []
        orig_reset = disk_checkpoint._reset_checkpoint

        def _spy(*args, **kwargs):
            reset_calls.append(args)
            return orig_reset(*args, **kwargs)

        disk_checkpoint._reset_checkpoint = _spy
        try:
            with self.assertRaisesRegex(FileNotFoundError, "has no safetensors"):
                self.materialize(3)
        finally:
            disk_checkpoint._reset_checkpoint = orig_reset
        self.assertEqual(len(reset_calls), 1)
        self.assert_at_version(1)

    def test_non_checksum_failure_fails_fast_without_reseed(self):
        reset_calls = []
        orig_reset = disk_checkpoint._reset_checkpoint

        def _fail(*args, **kwargs):
            reset_calls.append(args)
            raise OSError("simulated checkpoint write failure")

        disk_checkpoint._reset_checkpoint = _fail
        try:
            with self.assertRaisesRegex(OSError, "simulated checkpoint write failure"):
                self.materialize(1)
        finally:
            disk_checkpoint._reset_checkpoint = orig_reset
        self.assertEqual(
            len(reset_calls),
            1,
            "a non-checksum failure cannot be repaired by repeating the same seed",
        )

    def test_incomplete_source_version_fails_fast_then_recovers(self):
        # A version whose index is visible but whose data blob has not finished
        # propagating (object-store read-after-write lag) must raise
        # FileNotFoundError, NOT a checksum mismatch — otherwise it would be
        # misread as local corruption and trigger a needless full reseed. And it
        # must not reseed: the caller reloads + retries, and once the blob lands
        # the same materialization applies cleanly.
        self.materialize(2)
        self.assert_at_version(2)
        self.pub.publish_delta(
            3,
            {
                "layer.a": np.random.default_rng(3)
                .integers(0, 256, 4096, dtype=np.uint8)
                .tobytes()
            },
        )
        shard = os.path.join(self.pub.source_dir, "weight_v000003", Publisher.SHARD)
        with open(shard, "rb") as f:
            blob = f.read()
        os.remove(shard)  # index present, blob not yet materialized here

        reset_calls = []
        orig_reset = disk_checkpoint._reset_checkpoint

        def _spy(*args, **kwargs):
            reset_calls.append(args)
            return orig_reset(*args, **kwargs)

        disk_checkpoint._reset_checkpoint = _spy
        try:
            with self.assertRaises(FileNotFoundError):
                self.materialize(3)
        finally:
            disk_checkpoint._reset_checkpoint = orig_reset
        self.assertEqual(
            reset_calls, [], "must not reseed on an incomplete source version"
        )
        self.assert_at_version(2)  # local untouched by the failed materialization

        with open(shard, "wb") as f:
            f.write(blob)  # blob finishes propagating
        self.materialize(3)
        self.assert_at_version(3)

    def test_truncated_source_blob_fails_fast_then_recovers(self):
        # A blob present but shorter than its own safetensors header declares
        # (a half-materialized copy on an eventually-consistent mount). Staging
        # must size-verify and reject it as not-ready (FileNotFoundError, no
        # reseed) instead of applying a partial delta; the retry succeeds once
        # the full bytes land.
        self.materialize(2)
        self.assert_at_version(2)
        self.pub.publish_delta(
            3,
            {
                "layer.a": np.random.default_rng(5)
                .integers(0, 256, 4096, dtype=np.uint8)
                .tobytes()
            },
        )
        shard = os.path.join(self.pub.source_dir, "weight_v000003", Publisher.SHARD)
        with open(shard, "rb") as f:
            full = f.read()
        with open(shard, "wb") as f:
            f.write(full[:-256])  # header still declares the full length

        reset_calls = []
        orig_reset = disk_checkpoint._reset_checkpoint

        def _spy(*args, **kwargs):
            reset_calls.append(args)
            return orig_reset(*args, **kwargs)

        disk_checkpoint._reset_checkpoint = _spy
        try:
            with self.assertRaises(FileNotFoundError):
                self.materialize(3)
        finally:
            disk_checkpoint._reset_checkpoint = orig_reset
        self.assertEqual(
            reset_calls, [], "must not reseed on a truncated (not-ready) blob"
        )
        self.assert_at_version(2)

        with open(shard, "wb") as f:
            f.write(full)  # full bytes materialize
        self.materialize(3)
        self.assert_at_version(3)

    def test_multi_delta_materialization_handles_repeated_tensor_changes(self):
        rng = np.random.default_rng(23)
        self.pub.publish_delta(
            3, {"layer.a": rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()}
        )
        self.pub.publish_delta(
            4,
            {
                "layer.a": rng.integers(0, 256, 4096, dtype=np.uint8).tobytes(),
                "layer.b": rng.integers(0, 256, 2048, dtype=np.uint8).tobytes(),
            },
        )
        self.materialize(4)
        self.assert_at_version(4)

    def test_multi_delta_materialization_reseeds_corrupt_local_checkpoint(self):
        rng = np.random.default_rng(31)
        self.pub.publish_delta(
            3, {"layer.a": rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()}
        )
        self.materialize(1)
        shard = os.path.join(self.local, Publisher.SHARD)
        _, offset, _ = disk_checkpoint._tensor_locations(self.local)["layer.a"]
        with open(shard, "r+b") as f:
            f.seek(offset)
            f.write(bytes(16))  # silent local corruption
        self.materialize(3)
        self.assert_at_version(3)

    def test_materialize_zero_seeds_base_without_applying_a_delta(self):
        with self.spy_materialize() as (seeds, applies):
            self.materialize(0)
        self.assert_at_version(0)
        self.assertEqual((seeds, applies), ([self.pub.base_dir], []))

    def test_indexed_base_must_contain_every_shard(self):
        with open(
            os.path.join(self.pub.base_dir, "model.safetensors.index.json"), "w"
        ) as file:
            json.dump(
                {
                    "metadata": {},
                    "weight_map": {
                        "layer.a": Publisher.SHARD,
                        "layer.b": "model-00002-of-00002.safetensors",
                    },
                },
                file,
            )

        with self.assertRaisesRegex(FileNotFoundError, "missing blob"):
            self.materialize(0)

    def test_mutable_checkpoint_cannot_alias_its_immutable_base(self):
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            disk_checkpoint.materialize(
                self.pub.base_dir,
                self.pub.base_dir,
                self.pub.source_dir,
                0,
            )

    def test_mutable_checkpoint_cannot_overlap_published_sources(self):
        local = os.path.join(self.pub.source_dir, "weight_v000001")
        with self.assertRaisesRegex(ValueError, "published"):
            disk_checkpoint.materialize(
                local,
                self.pub.base_dir,
                self.pub.source_dir,
                1,
            )


if __name__ == "__main__":
    unittest.main()
