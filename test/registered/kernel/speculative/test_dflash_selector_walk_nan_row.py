import unittest

import torch

from sglang.kernels.ops.speculative.dflash import selector_walk_triton
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _walk(scores, greedy):
    batch, slots, top_k, _ = scores.shape
    device = scores.device
    candidate_ids = torch.arange(batch * slots * top_k, device=device).view(
        batch, slots, top_k
    )
    tokens, q_rows = selector_walk_triton(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=torch.full((batch, slots), 0.5, device=device),
        temperatures=torch.ones(batch, device=device),
        greedy_mask=torch.full((batch,), greedy, dtype=torch.bool, device=device),
    )
    return candidate_ids.cpu(), tokens.cpu(), q_rows.cpu()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestDFlashSelectorWalkNanRow(CustomTestCase):
    def _assert_greedy_walk_matches_torch_argmax(self, scores):
        batch, slots, top_k, _ = scores.shape
        candidate_ids, tokens, q_rows = _walk(scores, greedy=True)
        scores = scores.cpu()

        previous = torch.zeros(batch, dtype=torch.int64)
        for slot in range(slots):
            rows = scores[torch.arange(batch), slot, previous]
            expected = rows.argmax(dim=-1)
            self.assertEqual(
                tokens[:, slot].tolist(),
                candidate_ids[torch.arange(batch), slot, expected].tolist(),
            )
            torch.testing.assert_close(
                q_rows[:, slot], torch.nn.functional.one_hot(expected, top_k).float()
            )
            previous = expected

    def test_all_nan_greedy_row_stays_inside_its_candidate_row(self):
        batch, slots, top_k = 2, 3, 4
        scores = torch.randn(batch, slots, top_k, top_k, device="cuda")
        scores[0] = float("nan")
        candidate_ids, tokens, q_rows = _walk(scores, greedy=True)

        for slot in range(slots):
            self.assertIn(int(tokens[0, slot]), candidate_ids[0, slot].tolist())
            self.assertEqual(float(q_rows[0, slot].sum()), 1.0)
        self._assert_greedy_walk_matches_torch_argmax(scores)

    def test_mixed_nan_greedy_row_matches_torch_argmax(self):
        torch.manual_seed(5)
        scores = torch.randn(3, 6, 8, 8, device="cuda")
        scores[torch.rand_like(scores) < 0.3] = float("nan")
        self._assert_greedy_walk_matches_torch_argmax(scores)

    def test_finite_greedy_row_matches_torch_argmax(self):
        scores = torch.randn(3, 4, 8, 8, device="cuda")
        scores[1, 2, :, :] = 0.0  # ties break to the left
        self._assert_greedy_walk_matches_torch_argmax(scores)


if __name__ == "__main__":
    unittest.main()
