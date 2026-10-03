"""Model-independent contracts required by score-centering clients."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.dflash_utils import build_speculative_verify_target_probs
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_sampling_mask import (
    SpeculativeSamplingMaskCapture,
    validate_spec_sampling_mask_request,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("width", [32, 128])
def test_chain_logprobs_match_independent_dense_distribution(width):
    """Different temperatures and emitted IDs expose row/temperature misalignment."""
    generator = torch.Generator().manual_seed(407)
    batch_size, block_size, vocab_size = 3, 5, 257
    logits = torch.randn(batch_size * block_size, vocab_size, generator=generator)
    temperatures = torch.tensor([[0.7], [1.0], [1.3]])
    tokens = torch.randint(vocab_size, (batch_size, block_size), generator=generator)
    batch = SimpleNamespace(
        seq_lens=torch.tensor([3, 12, 71]),
        sampling_info=SimpleNamespace(is_all_greedy=False, temperatures=temperatures),
        top_logprobs_nums=[width] * batch_size,
        token_ids_logprobs=None,
    )
    output = SimpleNamespace(next_token_logits=logits)

    with envs.SGLANG_RETURN_ORIGINAL_LOGPROB.override(False):
        compute_spec_logprobs(batch, output, tokens.flatten(), chain_stride=block_size)

    scaled = logits.double().reshape(batch_size, block_size, vocab_size)
    scaled = scaled / temperatures.double().unsqueeze(-1)
    # Independent FP64 normalization, rather than the production log_softmax.
    expected = scaled - scaled.exp().sum(dim=-1, keepdim=True).log()
    expected_selected = expected.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(
        output.next_token_logprobs.double(), expected_selected, atol=1e-6, rtol=1e-6
    )
    flat_expected = expected.flatten(0, 1)
    for row, (values, ids) in enumerate(
        zip(output.next_token_top_logprobs_val, output.next_token_top_logprobs_idx)
    ):
        assert len(ids) == width
        assert len(set(ids.tolist())) == width
        torch.testing.assert_close(
            values.double(), flat_expected[row, ids], atol=1e-6, rtol=1e-6
        )
        assert values[-1] >= flat_expected[row].sort(descending=True).values[width]


def test_support_capture_keeps_request_rows_commit_lengths_and_greedy_rows():
    """The output can mix sampled/greedy requests and subset support capture."""
    probs = torch.tensor(
        [
            [[0.1, 0.6, 0.3, 0.0], [0.0, 0.2, 0.8, 0.0]],
            [[0.3, 0.2, 0.1, 0.4], [0.2, 0.1, 0.6, 0.1]],
            [[0.0, 0.0, 0.25, 0.75], [0.7, 0.0, 0.3, 0.0]],
        ]
    )
    # Captured request order differs from the batch; only the first captured
    # request asks for full support probabilities.
    capture = SpeculativeSamplingMaskCapture(
        target_probs=probs,
        batch_indices=torch.tensor([2, 1]),
        max_tokens=4,
        greedy_mask=torch.tensor([False, True, False]),
        support_capture_indices=torch.tensor([0]),
    )
    output = capture.build_output(
        out_tokens=torch.tensor([[1, 2], [3, 2], [3, 0]]),
        commit_lens=torch.tensor([2, 1, 2]),
    )
    assert output.num_accept_tokens.tolist() == [2, 1]
    assert output.lengths.tolist() == [[2, 2], [1, 1]]
    torch.testing.assert_close(
        output.selected_logprobs[0], torch.tensor([0.75, 0.7]).log()
    )
    torch.testing.assert_close(output.selected_logprobs[1], torch.zeros(2))
    assert output.token_ids[1, :, 0].tolist() == [3, 2]
    for position in range(2):
        ids = output.token_ids[0, position].long()
        actual = output.support_logprobs[0, position].exp()
        torch.testing.assert_close(actual, probs[2, position, ids])
        torch.testing.assert_close(actual.sum(), torch.tensor(1.0))
    assert (output.statuses == 0).all()


@pytest.mark.parametrize("capacity,expected_status", [(2, 1), (4, 0)])
def test_cutoff_ties_are_captured_completely_or_rejected(capacity, expected_status):
    probs = torch.tensor([[[0.5, 0.25, 0.25, 0.0]]])
    capture = SpeculativeSamplingMaskCapture(
        target_probs=probs,
        batch_indices=torch.tensor([0]),
        max_tokens=capacity,
        support_capture_indices=torch.tensor([0]),
    )
    output = capture.build_output(
        out_tokens=torch.tensor([[2]]), commit_lens=torch.tensor([1])
    )
    assert output.statuses.item() == expected_status
    assert output.lengths.item() == min(capacity, 3)
    if expected_status == 0:
        length = output.lengths.item()
        assert set(output.token_ids[0, 0, :length].tolist()) == {0, 1, 2}
        torch.testing.assert_close(
            output.support_logprobs[0, 0, :length].exp().sum(), torch.tensor(1.0)
        )


def test_capture_uses_server_capacity_instead_of_requested_top_k():
    info = SimpleNamespace(
        temperatures=torch.tensor([[1.0]]),
        top_ks=torch.tensor([2]),
        need_top_k_sampling=True,
        need_top_p_sampling=False,
        sampling_mask_batch_indices=torch.tensor([0]),
        sampling_support_logprobs_capture_indices=torch.tensor([0]),
        sampling_mask_top_ks=[2],
        is_all_greedy=False,
    )
    with get_context().override_server_args(
        sampling_mask_max_tokens=4, sampling_filter_order="top_k_first"
    ):
        capture = SpeculativeSamplingMaskCapture.from_logits(
            info,
            next_token_logits=torch.tensor([[0.4, 0.2, 0.2, 0.1, 0.1]]).log(),
            draft_input=SimpleNamespace(max_top_k=2, uniform_top_k_value=2),
            draft_token_num=1,
            bs=1,
        )
    output = capture.build_output(
        out_tokens=torch.tensor([[2]]), commit_lens=torch.tensor([1])
    )
    assert output.statuses.item() == 0
    assert output.lengths.item() == 3


def test_sparse_top_k_keeps_cutoff_ties_with_heterogeneous_requests():
    logits = torch.tensor([[3.0, 2.0, 2.0, 1.0, 0.0], [1.0, 3.0, 2.0, 2.0, 2.0]])
    temperatures = torch.tensor([[0.7], [1.3]])
    info = SimpleNamespace(
        temperatures=temperatures,
        top_ks=torch.tensor([2, 3]),
        need_top_k_sampling=True,
        need_top_p_sampling=False,
    )
    actual = build_speculative_verify_target_probs(
        next_token_logits=logits,
        sampling_info=info,
        draft_token_num=1,
        bs=2,
        max_top_k=3,
    ).squeeze(1)
    expected_weights = (logits.double() / temperatures.double()).exp()
    expected_weights[0, 3:] = 0.0
    expected_weights[1, 0] = 0.0
    expected = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(actual.double(), expected, atol=1e-7, rtol=1e-6)


def test_greedy_rows_keep_singleton_support_with_tied_maximum():
    info = SimpleNamespace(
        temperatures=torch.ones(2, 1),
        top_ks=torch.tensor([1, 2]),
        need_top_k_sampling=True,
        need_top_p_sampling=False,
    )
    actual = build_speculative_verify_target_probs(
        next_token_logits=torch.tensor([[3.0, 3.0, 1.0], [2.0, 1.0, 1.0]]),
        sampling_info=info,
        draft_token_num=1,
        bs=2,
        max_top_k=2,
    ).squeeze(1)
    assert (actual[0] > 0).sum() == 1
    assert actual[0].max() == 1
    assert (actual[1] > 0).sum() == 3


@pytest.mark.parametrize(
    "selector,selector_enabled,lilicorr,lilicorr_enabled,target_only,expected",
    [
        (False, False, False, False, False, False),
        (False, False, False, False, True, True),
        (True, True, False, False, False, True),
        (True, False, False, False, False, False),
        (False, False, True, True, False, True),
        (False, False, True, False, False, False),
        (True, False, False, False, True, True),
    ],
)
def test_sampling_capability_reflects_loaded_draft(
    monkeypatch,
    selector,
    selector_enabled,
    lilicorr,
    lilicorr_enabled,
    target_only,
    expected,
):
    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker.selector = object() if selector else None
    worker._selector_sampling_enabled = selector_enabled
    worker.lilicorr = object() if lilicorr else None
    worker._lilicorr_sampling_enabled = lilicorr_enabled
    monkeypatch.setattr(
        "sglang.srt.speculative.dflash_worker_v2.is_dflash_sampling_verify_available",
        lambda: target_only,
    )
    assert worker.sampling_verify_available() is expected


def test_enabled_sampling_never_silently_falls_back_to_greedy(monkeypatch):
    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker.selector = object()
    worker._selector_sampling_enabled = True
    worker.lilicorr = None
    worker._lilicorr_sampling_enabled = False
    worker._selector_sample = None
    monkeypatch.setattr(
        "sglang.srt.speculative.dflash_worker_v2.is_dflash_sampling_verify_available",
        lambda: False,
    )
    with pytest.raises(RuntimeError, match="Refusing greedy fallback"):
        worker._accept_block(
            candidates=torch.tensor([[0, 1]]),
            next_token_logits=torch.zeros(2, 3),
            sampling_info=SimpleNamespace(is_all_greedy=False),
            draft_input=None,
            prefix_lens=torch.tensor([3]),
            bs=1,
        )


@pytest.mark.parametrize("greedy,available", [(True, True), (False, False)])
def test_ordinary_inference_keeps_existing_greedy_behavior(
    monkeypatch, greedy, available
):
    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker.selector = object() if available else None
    worker._selector_sampling_enabled = available
    worker.lilicorr = None
    worker._lilicorr_sampling_enabled = False
    worker._selector_sample = None
    worker.block_size = 2
    worker._use_triton_accept_bonus = False
    worker._tp_sync = SimpleNamespace(sync=lambda site, tensor: None)
    monkeypatch.setattr(
        "sglang.srt.speculative.dflash_worker_v2.is_dflash_sampling_verify_available",
        lambda: False,
    )
    _, lengths, _, tokens, _, _ = worker._accept_block(
        candidates=torch.tensor([[0, 1]]),
        next_token_logits=torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]),
        sampling_info=SimpleNamespace(is_all_greedy=greedy),
        draft_input=None,
        prefix_lens=torch.tensor([3]),
        bs=1,
    )
    assert lengths.tolist() == [2]
    assert tokens.tolist() == [[1, 2]]


@pytest.mark.parametrize(
    "simulated,threshold_single,threshold_acc,kernel_available,expected_error",
    [
        (-1, 1.0, 1.0, True, None),
        (3, 1.0, 1.0, True, "simulated"),
        (-1, 0.5, 1.0, True, "thresholds"),
        (-1, 1.0, 0.5, True, "thresholds"),
        (-1, 1.0, 1.0, False, "sampling verification support"),
    ],
)
def test_filtered_support_rejects_approximate_sampling(
    monkeypatch,
    simulated,
    threshold_single,
    threshold_acc,
    kernel_available,
    expected_error,
):
    req = SimpleNamespace(sampling_params=SamplingParams(top_k=32, top_p=0.9))
    monkeypatch.setattr(
        "sglang.srt.speculative.spec_sampling_mask.is_dflash_sampling_verify_available",
        lambda: kernel_available,
    )
    with (
        get_context().override_server_args(
            speculative_accept_threshold_single=threshold_single,
            speculative_accept_threshold_acc=threshold_acc,
        ),
        envs.SGLANG_SIMULATE_ACC_LEN.override(simulated),
    ):
        error = validate_spec_sampling_mask_request(req, SpeculativeAlgorithm.DFLASH)
    if expected_error is None:
        assert error is None
    else:
        assert expected_error in error
