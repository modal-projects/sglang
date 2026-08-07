import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.sampler import top_p_normalize_probs_torch
from sglang.srt.speculative.dflash_utils import (
    build_dflash_sampling_mask_output,
    build_dflash_verify_target_probs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDFlashSamplingMask(CustomTestCase):
    def test_sparse_top_k_then_top_p_matches_dense_reference(self):
        logits = torch.tensor(
            [
                [4.0, 3.0, 2.0, 1.0, 0.0],
                [0.0, 2.0, 1.0, 4.0, 3.0],
                [3.0, 0.0, 1.0, 2.0, -1.0],
                [-1.0, 1.0, 0.0, 3.0, 2.0],
            ]
        )
        sampling_info = SimpleNamespace(
            need_top_k_sampling=True,
            need_top_p_sampling=True,
            temperatures=torch.tensor([1.0, 2.0]).unsqueeze(1),
            top_ks=torch.tensor([2, 4]),
            top_ps=torch.tensor([0.7, 0.8]),
        )
        expanded_temperatures = torch.tensor([1.0, 1.0, 2.0, 2.0]).unsqueeze(1)
        scaled_logits = logits / expanded_temperatures
        repeated_top_ks = [2, 2, 4, 4]
        dense_top_k_probs = torch.zeros_like(logits)
        for row, top_k in enumerate(repeated_top_ks):
            values, indices = scaled_logits[row].topk(top_k)
            dense_top_k_probs[row, indices] = torch.softmax(values, dim=-1)
        expected = top_p_normalize_probs_torch(
            dense_top_k_probs, torch.tensor([0.7, 0.7, 0.8, 0.8])
        ).reshape(2, 2, 5)

        with patch(
            "sglang.srt.speculative.dflash_utils.top_p_renorm_prob",
            top_p_normalize_probs_torch,
        ):
            actual = build_dflash_verify_target_probs(
                next_token_logits=logits,
                sampling_info=sampling_info,
                draft_token_num=2,
                bs=2,
                max_top_k=4,
                use_sparse_topk=True,
            )

        torch.testing.assert_close(actual, expected)

    def test_top_p_probabilities_and_support_match_reference(self):
        logits = torch.tensor(
            [
                [4.0, 3.0, 2.0, 1.0, 0.0],
                [0.0, 2.0, 1.0, 4.0, 3.0],
                [3.0, 0.0, 1.0, 2.0, -1.0],
                [-1.0, 1.0, 0.0, 3.0, 2.0],
            ]
        )
        sampling_info = SimpleNamespace(
            need_top_k_sampling=False,
            need_top_p_sampling=True,
            temperatures=torch.tensor([1.0, 2.0]).unsqueeze(1),
            top_ps=torch.tensor([0.6, 0.8]),
        )
        expanded_temperatures = torch.tensor([1.0, 1.0, 2.0, 2.0]).unsqueeze(1)
        unfiltered = torch.softmax(logits / expanded_temperatures, dim=-1)
        expected = top_p_normalize_probs_torch(
            unfiltered.clone(), torch.tensor([0.6, 0.6, 0.8, 0.8])
        ).reshape(2, 2, 5)

        with patch(
            "sglang.srt.speculative.dflash_utils.top_p_renorm_prob",
            top_p_normalize_probs_torch,
        ):
            actual = build_dflash_verify_target_probs(
                next_token_logits=logits,
                sampling_info=sampling_info,
                draft_token_num=2,
                bs=2,
                use_sparse_topk=False,
            )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(
            (actual > 0).nonzero().tolist(),
            (expected > 0).nonzero().tolist(),
        )

        output_token_ids = expected.argmax(dim=-1)
        masks, logprobs = build_dflash_sampling_mask_output(
            target_probs=actual,
            output_token_ids=output_token_ids,
            output_lens=torch.tensor([2, 2]),
            return_sampling_masks=[True, True],
        )
        for request_idx in range(2):
            for token_idx in range(2):
                expected_support = (
                    expected[request_idx, token_idx].nonzero().flatten().tolist()
                )
                selected_id = output_token_ids[request_idx, token_idx]
                expected_logprob = expected[request_idx, token_idx, selected_id].log()
                self.assertEqual(masks[request_idx][token_idx], expected_support)
                self.assertAlmostEqual(
                    logprobs[request_idx][token_idx], expected_logprob.item()
                )

    def test_filtered_target_probabilities(self):
        target_probs = torch.tensor(
            [
                [
                    [0.0, 0.25, 0.0, 0.75, 0.0],
                    [0.1, 0.0, 0.2, 0.0, 0.7],
                    [1, 0, 0, 0, 0],
                ],
                [[0.0, 0.0, 0.4, 0.6, 0.0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],
            ]
        )
        masks, logprobs = build_dflash_sampling_mask_output(
            target_probs=target_probs,
            output_token_ids=torch.tensor([[3, 4, 0], [3, 1, 2]]),
            output_lens=torch.tensor([2, 1]),
            return_sampling_masks=[True, False],
        )

        self.assertEqual(masks, [[[1, 3], [0, 2, 4]], None])
        self.assertAlmostEqual(logprobs[0][0], math.log(0.75))
        self.assertAlmostEqual(logprobs[0][1], math.log(0.7))
        self.assertIsNone(logprobs[1])

    def test_greedy_masks_are_singletons(self):
        masks, logprobs = build_dflash_sampling_mask_output(
            target_probs=None,
            output_token_ids=torch.tensor([[3, 4], [2, 1]]),
            output_lens=torch.tensor([2, 1]),
            return_sampling_masks=[True, True],
        )

        self.assertEqual(masks, [[[3], [4]], [[2]]])
        self.assertEqual(logprobs, [[0.0, 0.0], [0.0]])

    def test_heterogeneous_commit_lengths_select_only_committed_rows(self):
        # The lengths represent full acceptance, early rejection plus its bonus,
        # and a one-token result after finish-time truncation.
        target_probs = torch.zeros(3, 4, 8)
        output_token_ids = torch.tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [7, 6, 5, 4],
            ]
        )
        for request_idx in range(3):
            for token_idx in range(4):
                selected_id = output_token_ids[request_idx, token_idx]
                other_id = (selected_id + 1) % target_probs.shape[-1]
                selected_prob = 0.6 + 0.01 * (request_idx * 4 + token_idx)
                target_probs[request_idx, token_idx, selected_id] = selected_prob
                target_probs[request_idx, token_idx, other_id] = 1 - selected_prob

        masks, logprobs = build_dflash_sampling_mask_output(
            target_probs=target_probs,
            output_token_ids=output_token_ids,
            output_lens=torch.tensor([4, 2, 1]),
            return_sampling_masks=[True, False, True],
        )

        self.assertEqual(len(masks[0]), 4)
        self.assertIsNone(masks[1])
        self.assertEqual(len(masks[2]), 1)
        for request_idx in (0, 2):
            expected_len = 4 if request_idx == 0 else 1
            for token_idx in range(expected_len):
                selected_id = output_token_ids[request_idx, token_idx].item()
                self.assertIn(selected_id, masks[request_idx][token_idx])
                self.assertAlmostEqual(
                    logprobs[request_idx][token_idx],
                    math.log(target_probs[request_idx, token_idx, selected_id].item()),
                )


if __name__ == "__main__":
    unittest.main()
