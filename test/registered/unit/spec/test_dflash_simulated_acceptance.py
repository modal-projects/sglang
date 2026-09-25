import unittest

import torch

from sglang.srt.speculative.dflash_utils import apply_dflash_simulated_acceptance
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDFlashSimulatedAcceptance(CustomTestCase):
    def test_forced_length_bounds_existing_invalid_row_metadata(self):
        """Both token modes use final lengths to retain or discard row indices."""
        for mode in ("fixed", "real-draft-token"):
            for forced_commit_len in (1, 3, 4):
                with self.subTest(mode=mode, forced_commit_len=forced_commit_len):
                    candidates = torch.tensor([[10, 11, 12, 13]] * 5)
                    target_predict = torch.tensor([[20, 21, 22, 23]] * 5)
                    accept_len = torch.tensor([3, 0, 3, 0, 1], dtype=torch.int32)
                    commit_lens = accept_len + 1
                    bonus = torch.zeros(5, dtype=torch.int64)
                    out_tokens = torch.zeros_like(candidates)
                    first_invalid_rows = torch.tensor(
                        [-1, 0, 1, 2, 3], dtype=torch.int32
                    )
                    apply_dflash_simulated_acceptance(
                        candidates=candidates,
                        target_predict=target_predict,
                        accept_len=accept_len,
                        commit_lens=commit_lens,
                        bonus=bonus,
                        out_tokens=out_tokens,
                        simulate_acc_len=forced_commit_len,
                        simulate_acc_method="match-expected",
                        simulate_acc_token_mode=mode,
                        first_invalid_rows=first_invalid_rows,
                    )
                    self.assertEqual(
                        first_invalid_rows.tolist(),
                        [
                            row if row < forced_commit_len else -1
                            for row in [-1, 0, 1, 2, 3]
                        ],
                    )
                    self.assertEqual(accept_len.tolist(), [forced_commit_len - 1] * 5)
                    self.assertEqual(commit_lens.tolist(), [forced_commit_len] * 5)
                    expected = (
                        [100] * forced_commit_len
                        if mode == "fixed"
                        else [11, 12, 13][: forced_commit_len - 1]
                        + [20 + forced_commit_len - 1]
                    )
                    self.assertEqual(
                        out_tokens[:, :forced_commit_len].tolist(), [expected] * 5
                    )
                    self.assertEqual(bonus.tolist(), [expected[-1]] * 5)


if __name__ == "__main__":
    unittest.main()
