"""CPU unit tests for weight_sync/local_checkpoint.py.

Covers the pull recovery semantics: seeding + delta-chain replay, torn-apply
detection via the apply-intent sentinel (a mutation killed partway must
reseed, never re-patch — XOR applied twice reverts), automatic
reseed-and-replay after a checksum mismatch on corrupted local state, and
fail-loud behavior when the published delta itself is bad.

The module under test needs only numpy + zstandard; it is loaded directly so
the tests run without the full sglang import chain (and therefore in any CPU
environment). Checkpoints are hand-crafted safetensors files with the
`adler32` checksum format so no extra hash packages are needed.
"""

import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import zstandard


def _load_module():
    if "local_checkpoint_under_test" in sys.modules:
        return sys.modules["local_checkpoint_under_test"]
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/weight_sync/local_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location("local_checkpoint_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["local_checkpoint_under_test"] = module
    spec.loader.exec_module(module)
    return module


local_checkpoint = _load_module()


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


class PullTest(unittest.TestCase):
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

    def pull(self, target):
        local_checkpoint.pull(
            self.local, self.pub.base_dir, self.pub.source_dir, target
        )

    def assert_at_version(self, version):
        self.assertEqual(read_local(self.local), self.pub.versions[version])
        self.assertEqual(local_checkpoint._read_applied_version(self.local), version)

    def test_seed_and_chain(self):
        self.pull(2)
        self.assert_at_version(2)

    def test_incremental_pulls_are_idempotent(self):
        self.pull(1)
        self.assert_at_version(1)
        self.pull(1)  # no-op
        self.assert_at_version(1)
        self.pull(2)
        self.assert_at_version(2)

    def test_torn_apply_reseeds_instead_of_repatching(self):
        self.pull(1)
        # Simulate an apply of v2 killed mid-mutation: marker still at 1, some
        # (not all) bytes of layer.b already XORed toward v2. The re-apply
        # double-XORs those bytes -> checksum mismatch -> reseed + replay.
        shard = os.path.join(self.local, Publisher.SHARD)
        locations = local_checkpoint._tensor_locations(self.local)
        _, offset, nbytes = locations["layer.b"]
        with open(shard, "r+b") as f:
            f.seek(offset)
            f.write(self.pub.versions[2]["layer.b"][: nbytes // 2])
        self.pull(2)
        self.assert_at_version(2)

    def test_corrupt_local_state_recovers_via_reseed(self):
        self.pull(1)
        # Silent local divergence (bit rot, lost page): no sentinel, marker
        # says 1, but the bytes are wrong. The v2 apply must checksum-fail,
        # reseed from base, and replay the chain — not wedge.
        shard = os.path.join(self.local, Publisher.SHARD)
        locations = local_checkpoint._tensor_locations(self.local)
        _, offset, _ = locations["layer.b"]
        with open(shard, "r+b") as f:
            f.seek(offset)
            f.write(bytes(16))
        self.pull(2)
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
            self.pull(1)
        # The publisher fixes the artifact; the next pull recovers by itself
        # (the sentinel left by the failed attempt forces a clean reseed).
        shutil.rmtree(vdir)
        self.pub.state = dict(self.pub.versions[0])
        rng = np.random.default_rng(11)
        self.pub.publish_delta(
            1, {"layer.a": rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()}
        )
        self.pull(1)
        self.assert_at_version(1)

    def test_missing_source_version_fails_fast_without_reseed(self):
        # A not-yet-visible source version (publisher/object store not caught
        # up) must raise FileNotFoundError WITHOUT reseeding: reseed can't
        # conjure the absent bytes, and for a large base the wasted full copy
        # is expensive. The caller retries once the source is visible.
        self.pull(1)
        self.assert_at_version(1)
        reset_calls = []
        orig_reset = local_checkpoint._reset_checkpoint

        def _spy(*args, **kwargs):
            reset_calls.append(args)
            return orig_reset(*args, **kwargs)

        local_checkpoint._reset_checkpoint = _spy
        try:
            with self.assertRaises(FileNotFoundError):
                self.pull(3)  # v3 never published (only v1, v2 exist)
        finally:
            local_checkpoint._reset_checkpoint = orig_reset
        self.assertEqual(reset_calls, [], "must not reseed on a missing source version")
        # Local state is untouched: a later pull to a real version still works.
        self.pull(2)
        self.assert_at_version(2)


if __name__ == "__main__":
    unittest.main()
