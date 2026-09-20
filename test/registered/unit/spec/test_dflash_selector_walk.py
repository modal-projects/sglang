import unittest

import torch

from sglang.kernels.ops.speculative.dflash import selector_walk_triton
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestDFlashSelectorWalk(unittest.TestCase):
    def test_nan_scores_and_graph_padding_stay_in_bounds(self):
        candidate_ids = torch.tensor(
            [
                [[11, 12, 13, 14], [21, 22, 23, 24]],
                [[31, 32, 33, 34], [41, 42, 43, 44]],
                [[51, 52, 53, 54], [61, 62, 63, 64]],
            ],
            dtype=torch.int64,
            device="cuda",
        )
        scores = torch.zeros((3, 2, 4, 4), dtype=torch.float32, device="cuda")
        scores[0, 0, 0].fill_(float("nan"))
        scores[0, 1, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
        scores[1, 0, 0] = torch.tensor([1.0, float("nan"), 3.0, 4.0], device="cuda")
        scores[1, 1, 1] = torch.tensor([4.0, 3.0, 2.0, 1.0], device="cuda")
        scores[2].fill_(float("nan"))

        tokens, q_rows = selector_walk_triton(
            candidate_ids=candidate_ids,
            scores=scores,
            uniforms=torch.zeros((3, 2), dtype=torch.float32, device="cuda"),
            temperatures=torch.ones(3, dtype=torch.float32, device="cuda"),
            greedy_mask=torch.ones(3, dtype=torch.bool, device="cuda"),
            logical_batch_size=torch.tensor([2], dtype=torch.int32, device="cuda"),
        )

        torch.testing.assert_close(
            tokens, torch.tensor([[11, 24], [32, 41], [0, 0]], device="cuda")
        )
        expected_q = torch.nn.functional.one_hot(
            torch.tensor([[0, 3], [1, 0], [0, 0]], device="cuda"), num_classes=4
        ).float()
        torch.testing.assert_close(q_rows, expected_q)


if __name__ == "__main__":
    unittest.main()
