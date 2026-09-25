import os
import unittest
from types import SimpleNamespace

import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

import triton

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.verify_validity import (
    _first_invalid_rows_kernel,
    _verify_rows_validity_kernel,
    prepare_verify_rows_,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestDFlashVerifyValidityKernels(CustomTestCase):
    def test_reductions_and_repair_across_vocab_chunks(self):
        """A bad value in any vocab chunk must survive repair as invalid metadata."""
        for is_logits in (False, True):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(is_logits=is_logits, dtype=dtype):
                    values = torch.ones((2, 5, 1031), device=DEVICE, dtype=dtype)
                    values[0, 0, 1027] = float("nan")
                    values[0, 1, 1030] = float("inf")
                    values[0, 2].fill_(-float("inf") if is_logits else 0.0)
                    values[0, 3, 1028] = -float("inf") if is_logits else -1.0
                    reference = values.cpu().clone()
                    expected_valid = prepare_verify_rows_(
                        reference, is_logits=is_logits
                    )
                    valid = torch.empty((2, 5), dtype=torch.bool, device=DEVICE)
                    args = (values, valid, 5, 1031, *values.stride())
                    _verify_rows_validity_kernel[(10,)](
                        *args, IS_LOGITS=is_logits, BLOCK=1024
                    )
                    torch.testing.assert_close(valid.cpu(), expected_valid)
                    torch.testing.assert_close(values.cpu(), reference)
                    out = torch.empty(2, dtype=torch.int32, device=DEVICE)
                    _first_invalid_rows_kernel[(2,)](
                        valid,
                        torch.tensor([4, 4], dtype=torch.int32, device=DEVICE),
                        out,
                        5,
                        BLOCK=triton.next_power_of_2(5),
                    )
                    self.assertEqual(out.tolist(), [0, -1])

    def test_selector_worker_preserves_reached_invalid_rows(self):
        """Real rejection sampling must expose bad first, middle, and bonus rows."""
        bs, block, vocab = 5, 3, 4
        logits = torch.full((bs, block, vocab), -float("inf"), device=DEVICE)
        logits[..., 1] = 0.0
        logits[0, 0].fill_(float("nan"))
        logits[1, 1].fill_(float("nan"))
        logits[2, 2].fill_(float("nan"))
        logits[3, 1].fill_(float("nan"))
        candidates = torch.ones((bs, block), dtype=torch.int64, device=DEVICE)
        candidates[3, 1] = 0
        candidate_ids = torch.ones((bs, block - 1, 1), dtype=torch.int64, device=DEVICE)
        candidate_ids[3, 0, 0] = 0
        q_rows = torch.ones((bs, block - 1, 1), device=DEVICE)
        worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
        worker.block_size = block
        worker._selector_sample = (candidate_ids, q_rows)
        worker._draft_probs_buf = None
        worker._tp_sync = SimpleNamespace(sync=lambda site, values: values)
        sampling_info = SimpleNamespace(
            is_all_greedy=False,
            temperatures=torch.ones((bs, 1), device=DEVICE),
            need_top_k_sampling=False,
            need_top_p_sampling=False,
        )
        args = dict(
            candidates=candidates,
            next_token_logits=logits.flatten(0, 1),
            sampling_info=sampling_info,
            draft_input=SimpleNamespace(max_top_k=None, uniform_top_k_value=None),
            prefix_lens=torch.zeros(bs, dtype=torch.int32, device=DEVICE),
            bs=bs,
        )
        result = worker._accept_block(**args)
        self.assertEqual(result[0].tolist(), [0, 1, 2, 0, 2])
        self.assertEqual(result[-1].tolist(), [0, 1, 2, -1, -1])
        self.assertEqual(result[2].tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(int(worker._draft_probs_buf.count_nonzero()), 0)
        original_metadata = result[-1]
        logits.fill_(-float("inf"))
        logits[..., 1] = 0.0
        next_result = worker._accept_block(**args)
        self.assertEqual(next_result[-1].tolist(), [-1] * bs)
        self.assertEqual(original_metadata.tolist(), [0, 1, 2, -1, -1])
        self.assertNotEqual(original_metadata.data_ptr(), next_result[-1].data_ptr())


if __name__ == "__main__":
    unittest.main()
