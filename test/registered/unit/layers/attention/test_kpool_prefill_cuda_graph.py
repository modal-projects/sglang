import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.kpool_prefill_cuda_graph import (
    _kpool_indexer_prefill_with_output,
)


class _Indexer:
    def __init__(self, result):
        self.result = result

    def _forward_cuda_impl(self, **kwargs):
        self.call = kwargs
        return self.result


class TestKPoolPrefillCudaGraph(unittest.TestCase):
    def _run(self, *, live_rows, capture_rows, result):
        forward_batch = types.SimpleNamespace(extend_num_tokens=live_rows)
        indexer = _Indexer(result)
        x = torch.zeros(capture_rows, 2)
        q_lora = torch.zeros(capture_rows, 2)
        positions = torch.zeros(capture_rows, dtype=torch.int64)
        output = torch.full((capture_rows, 3), 99, dtype=torch.int32)
        context = types.SimpleNamespace(forward_batch=forward_batch)
        with patch(
            "sglang.srt.layers.attention.dsa.kpool_prefill_cuda_graph."
            "get_tc_piecewise_forward_context",
            return_value=context,
        ):
            _kpool_indexer_prefill_with_output(
                indexer, x, q_lora, positions, output, layer_id=7
            )
        self.assertEqual(indexer.call["x"].shape[0], live_rows)
        return output

    def test_copies_live_result_and_masks_capture_padding(self):
        result = torch.tensor([[1, 2, 3]], dtype=torch.int32)
        output = self._run(live_rows=1, capture_rows=4, result=result)
        torch.testing.assert_close(output[0], result[0])
        torch.testing.assert_close(output[1:], torch.full((3, 3), -1))

    def test_copies_live_prefix_from_bucket_shaped_result(self):
        result = torch.arange(12, dtype=torch.int32).reshape(4, 3)
        output = self._run(live_rows=1, capture_rows=4, result=result)
        torch.testing.assert_close(output[0], result[0])
        torch.testing.assert_close(output[1:], torch.full((3, 3), -1))

    def test_rejects_unrelated_row_count(self):
        result = torch.zeros(2, 3, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "got \\(2, 3\\)"):
            self._run(live_rows=1, capture_rows=4, result=result)


if __name__ == "__main__":
    unittest.main()
