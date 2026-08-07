import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logprob_processor import compute_spec_v2_logprobs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSpecV2Logprobs(CustomTestCase):
    def test_permuted_accept_rows_use_request_temperature(self):
        logits = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 1.0, 0.0, -1.0],
                [-1.0, 0.0, 3.0, 1.0],
                [2.0, -2.0, 0.0, 1.0],
                [1.0, 3.0, 2.0, 0.0],
                [3.0, 0.0, -1.0, 2.0],
            ]
        )
        accept_index = torch.tensor([[4, 1, 5], [0, 3, 2]])
        predict = torch.tensor([3, 0, 2, 3, 1, 0])
        batch = SimpleNamespace(
            seq_lens=[10, 20],
            sampling_info=SimpleNamespace(
                is_all_greedy=False,
                temperatures=torch.tensor([0.5, 2.0]).unsqueeze(1),
            ),
            top_logprobs_nums=[2, 3],
            token_ids_logprobs=[[0, 3], [1]],
        )
        output = SimpleNamespace(next_token_logits=logits)

        with envs.SGLANG_RETURN_ORIGINAL_LOGPROB.override(False):
            compute_spec_v2_logprobs(
                batch,
                output,
                predict,
                accept_index,
                speculative_num_steps=2,
            )

        gathered_logits = logits[accept_index.flatten()]
        temperatures = torch.tensor([0.5] * 3 + [2.0] * 3).unsqueeze(1)
        expected_rows = torch.log_softmax(gathered_logits / temperatures, dim=-1)
        expected_tokens = predict[accept_index.flatten()].reshape(2, 3)
        expected = expected_rows.gather(1, expected_tokens.flatten().unsqueeze(1))
        torch.testing.assert_close(output.next_token_logprobs, expected.reshape(2, 3))

        for row, (values, indices) in enumerate(
            zip(
                output.next_token_top_logprobs_val,
                output.next_token_top_logprobs_idx,
                strict=True,
            )
        ):
            k = 2 if row < 3 else 3
            expected_values, expected_indices = expected_rows[row].topk(k)
            torch.testing.assert_close(values, expected_values)
            torch.testing.assert_close(indices, expected_indices)

        for row, (values, indices) in enumerate(
            zip(
                output.next_token_token_ids_logprobs_val,
                output.next_token_token_ids_logprobs_idx,
                strict=True,
            )
        ):
            requested_ids = [0, 3] if row < 3 else [1]
            torch.testing.assert_close(values, expected_rows[row, requested_ids])
            self.assertEqual(indices, requested_ids)

    def test_original_logprobs_ignore_sampling_temperature(self):
        logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, -1.0, 0.0]])
        batch = SimpleNamespace(
            seq_lens=[2],
            sampling_info=SimpleNamespace(
                is_all_greedy=False,
                temperatures=torch.tensor([0.25]).unsqueeze(1),
            ),
            top_logprobs_nums=[0],
            token_ids_logprobs=[None],
        )
        output = SimpleNamespace(next_token_logits=logits)
        predict = torch.tensor([1, 2])
        accept_index = torch.tensor([[0, 1]])

        with envs.SGLANG_RETURN_ORIGINAL_LOGPROB.override(True):
            compute_spec_v2_logprobs(
                batch,
                output,
                predict,
                accept_index,
                speculative_num_steps=1,
            )

        expected_rows = torch.log_softmax(logits, dim=-1)
        expected = expected_rows[torch.arange(2), predict].reshape(1, 2)
        torch.testing.assert_close(output.next_token_logprobs, expected)


if __name__ == "__main__":
    unittest.main()
