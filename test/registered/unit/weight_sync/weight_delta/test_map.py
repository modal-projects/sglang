"""Unit tests for the weight-delta map's table."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from pathlib import Path

import numpy as np
import torch

from sglang.srt.weight_sync.weight_delta.map import (
    OTHER_GPU,
    CanonicalTensor,
    Ignored,
    Place,
    _table,
)
from sglang.test.test_utils import CustomTestCase


def _tensor(name: str, dtype: str, shape: tuple[int, ...]) -> CanonicalTensor:
    return CanonicalTensor(name, dtype, shape, Path("/dev/null"))


def _place(dtype: torch.dtype, shape: tuple[int, ...]) -> Place:
    return Place(
        (slice(None),) * len(shape), torch.empty(shape, dtype=dtype, device="meta")
    )


class TestMapTable(CustomTestCase):
    def setUp(self):
        self.inventory = {
            "z.weight": _tensor("z.weight", "U8", (4, 2)),
            "a.scale": _tensor("a.scale", "F8_E4M3", (2,)),
            "m.bias": _tensor("m.bias", "BF16", (3,)),
            "b.input_scale": _tensor("b.input_scale", "F32", ()),
        }
        self.entries = {
            "z.weight": (_place(torch.uint8, (4, 2)),),
            "a.scale": (_place(torch.float32, (2,)),),
            "m.bias": OTHER_GPU,
            "b.input_scale": Ignored("activation calibration, never trained"),
        }

    def test_the_table_holds_every_checkpoint_tensor_sorted_by_name(self):
        table = _table(self.inventory, self.entries)

        self.assertEqual(
            table.names.tolist(), [b"a.scale", b"b.input_scale", b"m.bias", b"z.weight"]
        )
        self.assertEqual(table.dtypes.tolist(), ["F8_E4M3", "F32", "BF16", "U8"])
        self.assertEqual(table.shape.tolist(), [2, 3, 4, 2])
        self.assertEqual(table.shape_end.tolist(), [1, 1, 2, 4])
        self.assertEqual(table.nbytes.tolist(), [2, 4, 6, 8])
        self.assertEqual(table.held.tolist(), [True, False, False, True])
        self.assertEqual(table.ignored.tolist(), [False, True, False, False])
        self.assertEqual(table.raw_only.tolist(), [True, False, False, False])
        self.assertEqual(
            table.entries, [self.entries[name.decode()] for name in table.names]
        )

    def test_a_versions_names_join_the_table_in_one_search(self):
        table = _table(self.inventory, self.entries)
        names = np.array([b"m.bias", b"z.weight", b"a.scale"], dtype="S32")

        rows = np.searchsorted(table.names, names)

        np.testing.assert_array_equal(table.names[rows], names)
        self.assertEqual(rows.tolist(), [2, 3, 0])


if __name__ == "__main__":
    unittest.main()
