import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative.dflash_utils import (
    compute_dflash_sampling_correct_drafts_and_bonus,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class TestDFlashSamplingBoundary(CustomTestCase):
    def test_zero_rng_output_cannot_accept_zero_probability_draft(self):
        device = "cuda"
        candidates = torch.tensor([[0, 2]], dtype=torch.int64, device=device)
        logits = torch.full((2, 4), -1000.0, device=device)
        logits[:, 1] = 1000.0
        sampling_info = SimpleNamespace(
            temperatures=torch.ones((1, 1), device=device),
            top_ks=torch.ones(1, dtype=torch.int32, device=device),
            top_ps=torch.ones(1, device=device),
            need_top_k_sampling=True,
            need_top_p_sampling=False,
        )

        with mock.patch(
            "sglang.srt.speculative.dflash_utils.torch.rand",
            return_value=torch.zeros((1, 2), dtype=torch.float32, device=device),
        ):
            accept_len, bonus, target_probs = (
                compute_dflash_sampling_correct_drafts_and_bonus(
                    candidates=candidates,
                    next_token_logits=logits,
                    sampling_info=sampling_info,
                    max_top_k=1,
                    uniform_top_k_value=1,
                    threshold_single=1.0,
                    threshold_acc=1.0,
                    uniform_samples_for_final_sampling=torch.full(
                        (1,), 0.5, dtype=torch.float32, device=device
                    ),
                    return_target_probs=True,
                )
            )

        self.assertEqual(accept_len.item(), 0)
        self.assertEqual(bonus.item(), 1)
        self.assertEqual(target_probs[0, 0, 2].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
