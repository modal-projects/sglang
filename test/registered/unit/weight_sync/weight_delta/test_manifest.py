"""Unit tests for reading a weight-delta version."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sglang.srt.weight_sync.weight_delta.manifest import read_manifest, read_tensors
from sglang.test.test_utils import CustomTestCase

# The same version is in Miles' tests/fast/backends/training_utils/weight_update/protocols/
# test_weight_delta_publication.py.
GOLDEN_MANIFEST = (
    '{"base_version":4,"codec":"gap-zstd","files":[{"name":"owner-00000.safetensors","nbytes":1784}],"format":'
    '"weight-delta/v1","frame_nbytes":1048576,"target_version":5}'
)
GOLDEN_OWNER_HEADER = (
    '{"__metadata__":{"dtypes":"BF16,U8"},"tensors.name_end":{"dtype":"I64","shape":[2],"data_offsets":[0,16]}'
    ',"tensors.shape":{"dtype":"I64","shape":[3],"data_offsets":[16,40]},"tensors.shape_end":{"dtype":"I64","s'
    'hape":[2],"data_offsets":[40,56]},"tensors.offset":{"dtype":"I64","shape":[2],"data_offsets":[56,72]},"te'
    'nsors.nbytes":{"dtype":"I64","shape":[2],"data_offsets":[72,88]},"tensors.inner_nbytes":{"dtype":"I64","s'
    'hape":[2],"data_offsets":[88,104]},"tensors.chunk_end":{"dtype":"I64","shape":[2],"data_offsets":[104,120'
    ']},"tensors.frame_end":{"dtype":"I64","shape":[2],"data_offsets":[120,136]},"tensors.checksum":{"dtype":"'
    'U64","shape":[2],"data_offsets":[136,152]},"chunks.offset":{"dtype":"I64","shape":[1],"data_offsets":[152'
    ',160]},"chunks.nbytes":{"dtype":"I64","shape":[1],"data_offsets":[160,168]},"chunks.inner_offset":{"dtype'
    '":"I64","shape":[1],"data_offsets":[168,176]},"chunks.inner_nbytes":{"dtype":"I64","shape":[1],"data_offs'
    'ets":[176,184]},"frames.offset":{"dtype":"I64","shape":[1],"data_offsets":[184,192]},"frames.nbytes":{"dt'
    'ype":"I64","shape":[1],"data_offsets":[192,200]},"frames.inner_offset":{"dtype":"I64","shape":[1],"data_o'
    'ffsets":[200,208]},"frames.inner_nbytes":{"dtype":"I64","shape":[1],"data_offsets":[208,216]},"tensors.na'
    'me":{"dtype":"U8","shape":[14],"data_offsets":[216,230]},"tensors.dtype":{"dtype":"U8","shape":[2],"data_'
    'offsets":[230,232]},"tensors.raw":{"dtype":"BOOL","shape":[2],"data_offsets":[232,234]},"payload":{"dtype'
    '":"U8","shape":[22],"data_offsets":[234,256]}}    '
)
GOLDEN_OWNER_DATA = (
    "06000000000000000e00000000000000020000000000000004000000000000000400000000000000010000000000000003000000"
    "00000000000000000000000010000000000000000400000000000000060000000000000000000000000000000a00000000000000"
    "000000000000000001000000000000000000000000000000010000000000000001020304000000001032547698badcfe00000000"
    "00000000060000000000000000000000000000000a00000000000000000000000000000010000000000000000000000000000000"
    "0a00000000000000612e62696173622e77656967687400010100010203040000000000000000000000007a7374642121"
)
GOLDEN_OWNER_FILE = (
    len(GOLDEN_OWNER_HEADER).to_bytes(8, "little")
    + GOLDEN_OWNER_HEADER.encode()
    + bytes.fromhex(GOLDEN_OWNER_DATA)
)


class TestReadVersion(CustomTestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name) / "weight_v000005"
        self.directory.mkdir()
        self.path = self.directory / "manifest.json"

    def _write(self, owners: list[str], **fields) -> Path:
        files = [{"name": name, "nbytes": len(GOLDEN_OWNER_FILE)} for name in owners]
        for name in owners:
            (self.directory / name).write_bytes(GOLDEN_OWNER_FILE)
        self.path.write_text(
            json.dumps(json.loads(GOLDEN_MANIFEST) | {"files": files} | fields)
        )
        return self.path

    def test_the_version_miles_writes_is_read(self):
        self.path.write_text(GOLDEN_MANIFEST)
        (self.directory / "owner-00000.safetensors").write_bytes(GOLDEN_OWNER_FILE)

        manifest = read_manifest(self.path)
        tensors = read_tensors(manifest)

        self.assertEqual(manifest.directory, self.directory)
        self.assertEqual((manifest.codec, manifest.frame_nbytes), ("gap-zstd", 1 << 20))
        self.assertEqual((manifest.base_version, manifest.target_version), (4, 5))
        self.assertEqual(manifest.files, {"owner-00000.safetensors": 1784})
        self.assertEqual(tensors.names.tolist(), [b"a.bias", b"b.weight"])
        self.assertEqual(tensors.dtypes.tolist(), ["BF16", "U8"])
        self.assertEqual(tensors.shape.tolist(), [2, 4, 4])
        self.assertEqual(tensors.shape_end.tolist(), [1, 3])
        self.assertEqual(tensors.raw.tolist(), [True, False])
        self.assertEqual(tensors.file.tolist(), [0, 0])
        self.assertEqual(
            [
                GOLDEN_OWNER_FILE[offset : offset + nbytes]
                for offset, nbytes in zip(tensors.offset, tensors.nbytes)
            ],
            [b"\x01\x02\x03\x04", b"zstd!!"],
        )
        self.assertEqual(tensors.inner_nbytes.tolist(), [0, 10])
        self.assertEqual(tensors.chunk_end.tolist(), [0, 1])
        self.assertEqual(tensors.frame_end.tolist(), [0, 1])
        self.assertEqual(tensors.checksum.tolist(), [0x04030201, 0xFEDCBA9876543210])
        self.assertEqual(
            {field: column.tolist() for field, column in tensors.chunks.items()},
            {
                "offset": [int(tensors.offset[1])],
                "nbytes": [6],
                "inner_offset": [0],
                "inner_nbytes": [10],
            },
        )
        self.assertEqual(
            {field: column.tolist() for field, column in tensors.frames.items()},
            {"offset": [0], "nbytes": [16], "inner_offset": [0], "inner_nbytes": [10]},
        )

    def test_owner_files_are_read_back_to_back(self):
        tensors = read_tensors(
            read_manifest(
                self._write(["owner-00000.safetensors", "owner-00001.safetensors"])
            )
        )

        self.assertEqual(tensors.names.tolist(), [b"a.bias", b"b.weight"] * 2)
        self.assertEqual(tensors.file.tolist(), [0, 0, 1, 1])
        self.assertEqual(tensors.shape_end.tolist(), [1, 3, 4, 6])
        self.assertEqual(tensors.chunk_end.tolist(), [0, 1, 1, 2])
        self.assertEqual(tensors.frame_end.tolist(), [0, 1, 1, 2])
        np.testing.assert_array_equal(tensors.offset[2:], tensors.offset[:2])
        self.assertEqual(
            tensors.chunks["offset"].tolist(),
            [int(tensors.offset[1]), int(tensors.offset[3])],
        )

    def test_a_version_without_owner_files_has_no_tensors(self):
        tensors = read_tensors(read_manifest(self._write([])))
        self.assertEqual(
            (
                len(tensors.names),
                len(tensors.chunks["offset"]),
                len(tensors.frames["offset"]),
            ),
            (0, 0, 0),
        )

    def test_an_owner_file_of_another_size_is_refused(self):
        self._write(["owner-00000.safetensors"])
        with open(self.directory / "owner-00000.safetensors", "ab") as owner:
            owner.write(b"\0")
        with self.assertRaisesRegex(ValueError, "1785 bytes, the manifest says 1784"):
            read_tensors(read_manifest(self.path))

    def test_another_format_is_refused(self):
        with self.assertRaisesRegex(ValueError, "format 'gpu-delta/v1'"):
            read_manifest(self._write([], format="gpu-delta/v1"))

    def test_an_unknown_codec_is_refused(self):
        with self.assertRaisesRegex(ValueError, "codec 'lz4'"):
            read_manifest(self._write([], codec="lz4"))

    def test_a_missing_field_is_named(self):
        data = json.loads(GOLDEN_MANIFEST)
        del data["base_version"]
        self.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "no 'base_version' field"):
            read_manifest(self.path)


if __name__ == "__main__":
    unittest.main()
