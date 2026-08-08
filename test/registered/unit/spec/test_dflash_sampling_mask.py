import math
import unittest

import torch

from sglang.srt.speculative.dflash_utils import build_dflash_sampling_mask_output
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDFlashSamplingMask(CustomTestCase):
    def test_sampling_mask_stays_tensor_backed_until_result_processing(self):
        target_probs = torch.tensor([[[0.25, 0.0, 0.75]]])
        pending = build_dflash_sampling_mask_output(
            target_probs=target_probs,
            output_token_ids=torch.tensor([[2]]),
            output_lens=torch.tensor([1]),
            return_sampling_masks=[True],
            max_mask_tokens=3,
        )

        copied = []

        def record_copy(tensor):
            copied.append(tensor)
            return tensor.clone()

        pending.map_device_tensors(record_copy)

        self.assertEqual(len(copied), 4)
        self.assertEqual(pending.support_mask.dtype, torch.bool)
        masks, logprobs = pending.finalize()
        self.assertEqual(masks, [[[0, 2]]])
        self.assertAlmostEqual(logprobs[0][0], math.log(0.75))

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
        pending = build_dflash_sampling_mask_output(
            target_probs=target_probs,
            output_token_ids=torch.tensor([[3, 4, 0], [3, 1, 2]]),
            output_lens=torch.tensor([2, 1]),
            return_sampling_masks=[True, False],
            max_mask_tokens=5,
        )
        masks, logprobs = pending.finalize()

        self.assertEqual(masks, [[[1, 3], [0, 2, 4]], None])
        self.assertAlmostEqual(logprobs[0][0], math.log(0.75))
        self.assertAlmostEqual(logprobs[0][1], math.log(0.7))
        self.assertIsNone(logprobs[1])

    def test_greedy_masks_are_singletons(self):
        pending = build_dflash_sampling_mask_output(
            target_probs=None,
            output_token_ids=torch.tensor([[3, 4], [2, 1]]),
            output_lens=torch.tensor([2, 1]),
            return_sampling_masks=[True, True],
            max_mask_tokens=1,
        )
        masks, logprobs = pending.finalize()

        self.assertEqual(masks, [[[3], [4]], [[2]]])
        self.assertEqual(logprobs, [[0.0, 0.0], [0.0]])

    def test_one_oversized_token_rejects_its_whole_request(self):
        target_probs = torch.tensor(
            [
                [[0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.0, 0.0]],
                [[0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            ]
        )
        pending = build_dflash_sampling_mask_output(
            target_probs=target_probs,
            output_token_ids=torch.tensor([[0, 1], [1, 0]]),
            output_lens=torch.tensor([2, 1]),
            return_sampling_masks=[True, True],
            max_mask_tokens=2,
        )
        masks, logprobs = pending.finalize()

        self.assertIsNone(masks[0])
        self.assertIsNone(logprobs[0])
        self.assertEqual(masks[1], [[0, 1]])
        self.assertAlmostEqual(logprobs[1][0], math.log(0.5))


if __name__ == "__main__":
    unittest.main()
