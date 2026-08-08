"""Unit tests for DFlash target-head sampling."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _PackedQuantMethod:
    def __init__(self, logits):
        self.logits = logits
        self.calls = []

    def apply(self, layer, hidden_states, bias):
        self.calls.append((layer, hidden_states, bias))
        return self.logits[: hidden_states.shape[0]]


class TestDFlashGreedyHead(CustomTestCase):
    def test_uses_quantization_method_for_packed_lm_head(self):
        hidden_states = torch.randn(2, 4)
        expected_logits = torch.tensor([[0.0, 4.0, 1.0, 2.0], [3.0, 1.0, 5.0, 2.0]])
        quant_method = _PackedQuantMethod(expected_logits)
        lm_head = SimpleNamespace(
            # Packed weights do not have the dense [vocab, hidden] shape.
            weight=torch.zeros((4, 2), dtype=torch.uint8),
            quant_method=quant_method,
            shard_indices=SimpleNamespace(
                num_org_elements=4,
                num_org_elements_padded=4,
                num_added_elements=0,
                org_vocab_start_index=0,
                added_vocab_start_index=4,
            ),
        )
        worker = object.__new__(DFlashWorkerV2)
        worker._draft_greedy_local_cap = 0
        worker._draft_greedy_local_max_buf = None
        worker._draft_greedy_local_arg_buf = None

        with mock.patch(
            "sglang.srt.speculative.dflash_worker_v2.get_tp_group",
            return_value=SimpleNamespace(world_size=1),
        ):
            token_ids = worker._greedy_sample_from_vocab_parallel_head(
                hidden_states=hidden_states,
                lm_head=lm_head,
            )

        torch.testing.assert_close(token_ids, torch.tensor([1, 2]))
        self.assertEqual(len(quant_method.calls), 1)
        self.assertIs(quant_method.calls[0][0], lm_head)
        torch.testing.assert_close(quant_method.calls[0][1], hidden_states)
        self.assertEqual(quant_method.calls[0][1].dtype, hidden_states.dtype)
        self.assertIsNone(quant_method.calls[0][2])


if __name__ == "__main__":
    unittest.main()
