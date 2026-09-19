import math

import torch

from sglang.srt.layers.logits_processor import SamplingMaskStatus
from sglang.srt.layers.sampler import Sampler
from sglang.srt.speculative.dflash_utils import build_dflash_sampling_mask_output
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _sampler(max_tokens: int) -> Sampler:
    sampler = object.__new__(Sampler)
    sampler.sampling_mask_max_tokens = max_tokens
    sampler.tp_sync_group = None
    sampler.cp_sync_group = None
    return sampler


def test_filtered_support_is_packed_for_each_verify_position():
    target_probs = torch.tensor(
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
    )
    output = build_dflash_sampling_mask_output(
        sampler=_sampler(5),
        target_probs=target_probs,
        output_token_ids=torch.tensor([[3, 4, 0], [3, 1, 2]]),
        batch_indices=torch.tensor([0]),
        is_all_greedy=False,
    )

    assert output.lengths.tolist() == [[2, 3, 1]]
    assert set(output.token_ids[0, 0, :2].tolist()) == {1, 3}
    assert set(output.token_ids[0, 1, :3].tolist()) == {0, 2, 4}
    assert math.isclose(
        output.selected_logprobs[0, 0].item(), math.log(0.75), rel_tol=1e-6
    )
    assert math.isclose(
        output.selected_logprobs[0, 1].item(), math.log(0.7), rel_tol=1e-6
    )


def test_non_greedy_fallback_is_invalid_instead_of_claiming_singleton_support():
    output = build_dflash_sampling_mask_output(
        sampler=_sampler(4),
        target_probs=None,
        output_token_ids=torch.tensor([[3, 4]]),
        batch_indices=torch.tensor([0]),
        is_all_greedy=False,
    )

    assert output.statuses[0, 0].item() == SamplingMaskStatus.INVALID


def test_greedy_support_is_the_emitted_token():
    output = build_dflash_sampling_mask_output(
        sampler=_sampler(1),
        target_probs=None,
        output_token_ids=torch.tensor([[3, 4]]),
        batch_indices=torch.tensor([0]),
        is_all_greedy=True,
    )

    assert output.token_ids.squeeze(-1).tolist() == [[3, 4]]
    assert output.lengths.tolist() == [[1, 1]]
    assert output.selected_logprobs.tolist() == [[0.0, 0.0]]
    assert output.statuses.tolist() == [[SamplingMaskStatus.OK, SamplingMaskStatus.OK]]
