import unittest

import torch

from sglang.srt.layers.quantization.utils import (
    block_scale_interleave,
    shuffle_matrix_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestBlockScaleInterleave(unittest.TestCase):
    def test_matches_128x4_layout_with_padding(self):
        batches, rows, cols = 2, 130, 5
        scale = torch.arange(
            batches * rows * cols,
            dtype=torch.int64,
        ).reshape(batches, rows, cols)
        scale = scale.remainder(251).to(torch.uint8)

        actual = block_scale_interleave(scale)

        padded_rows = 256
        padded_cols = 8
        expected = torch.zeros(
            batches * padded_rows * padded_cols,
            dtype=torch.uint8,
        )
        for batch in range(batches):
            for row in range(rows):
                for col in range(cols):
                    offset = (
                        batch * padded_rows * padded_cols
                        + (row // 128) * 128 * padded_cols
                        + (col // 4) * 512
                        + (row % 32) * 16
                        + ((row % 128) // 32) * 4
                        + col % 4
                    )
                    expected[offset] = scale[batch, row, col]

        torch.testing.assert_close(actual, expected)

    def test_two_dimensional_input_uses_one_batch(self):
        scale = (
            torch.arange(128 * 4, dtype=torch.int64).to(torch.bfloat16).reshape(128, 4)
        )

        actual = block_scale_interleave(scale)
        expected = (
            scale.reshape(1, 1, 4, 32, 1, 4)
            .permute(0, 1, 4, 3, 2, 5)
            .contiguous()
            .reshape(-1)
        )

        self.assertEqual(actual.device, scale.device)
        self.assertEqual(actual.dtype, scale.dtype)
        torch.testing.assert_close(actual, expected)

    def test_matrix_row_shuffle_preserves_columns(self):
        matrix = torch.arange(128 * 3, dtype=torch.int64).reshape(128, 3)

        actual = shuffle_matrix_rows(matrix, epilogue_tile_m=128)

        from flashinfer.utils import get_shuffle_matrix_a_row_indices

        row_indices = get_shuffle_matrix_a_row_indices(matrix, 128)
        torch.testing.assert_close(actual, matrix[row_indices])


if __name__ == "__main__":
    unittest.main()
