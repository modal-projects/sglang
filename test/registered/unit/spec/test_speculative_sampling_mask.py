import math

import torch

from sglang.srt.layers.logits_processor import SamplingMaskStatus
from sglang.srt.speculative.sampling_mask import SpeculativeSamplingMaskCapture
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_sampling_support_is_packed_for_committed_tokens():
    capture = SpeculativeSamplingMaskCapture(
        target_probs=torch.tensor(
            [
                [
                    [0.0, 0.25, 0.0, 0.75, 0.0],
                    [0.1, 0.0, 0.2, 0.0, 0.7],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.4, 0.6, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                ],
            ]
        ),
        return_sampling_masks=[True, False],
        max_top_k=5,
    )

    output = capture.build_output(
        out_tokens=torch.tensor([[3, 4, 0], [3, 1, 2]]),
        commit_lens=torch.tensor([2, 3]),
    )

    assert output.output_lens.tolist() == [2]
    assert output.lengths.tolist() == [[2, 3, 1]]
    assert set(output.token_ids[0, 0, :2].tolist()) == {1, 3}
    assert set(output.token_ids[0, 1, :3].tolist()) == {0, 2, 4}
    assert math.isclose(
        output.selected_logprobs[0, 0].item(), math.log(0.75), rel_tol=1e-6
    )
    assert math.isclose(
        output.selected_logprobs[0, 1].item(), math.log(0.7), rel_tol=1e-6
    )
    assert output.statuses.tolist() == [
        [SamplingMaskStatus.OK, SamplingMaskStatus.OK, SamplingMaskStatus.OK]
    ]


def test_mixed_greedy_rows_emit_singleton_support():
    capture = SpeculativeSamplingMaskCapture(
        target_probs=torch.tensor(
            [
                [[0.0, 0.25, 0.0, 0.75]],
                [[0.0, 0.25, 0.0, 0.75]],
            ]
        ),
        return_sampling_masks=[True, True],
        max_top_k=4,
        greedy_mask=torch.tensor([True, False]),
    )

    output = capture.build_output(
        out_tokens=torch.tensor([[3], [3]]),
        commit_lens=torch.tensor([1, 1]),
    )

    assert output.token_ids[0, 0, :1].tolist() == [3]
    assert output.lengths[0].tolist() == [1]
    assert output.selected_logprobs[0].tolist() == [0.0]
    assert set(output.token_ids[1, 0, :2].tolist()) == {1, 3}
    assert output.lengths[1].tolist() == [2]


def test_zero_selected_probability_is_invalid():
    capture = SpeculativeSamplingMaskCapture(
        target_probs=torch.tensor([[[0.0, 1.0]]]),
        return_sampling_masks=[True],
        max_top_k=2,
    )

    output = capture.build_output(
        out_tokens=torch.tensor([[0]]),
        commit_lens=torch.tensor([1]),
    )

    assert output.statuses.tolist() == [[SamplingMaskStatus.INVALID]]
