"""Both DFlash verifier families must preserve the target categorical law."""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.kernels.ops.speculative.sampling import (
    tree_speculative_sampling_target_only,
)
from sglang.srt.speculative.dflash_utils import build_speculative_verify_target_probs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("verifier", ["target_only", "rejection"])
@pytest.mark.parametrize(
    "target_weights,draft_weights",
    [
        ([1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]),
        ([0, 0, 0, 0, 1, 2, 3, 4], [1, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 1, 2, 3, 4], [0, 0, 0, 0, 1, 2, 3, 4]),
    ],
    ids=["different-proposal", "proposal-outside-filtered-support", "same-policy"],
)
def test_dflash_verifier_preserves_target_distribution(
    verifier, target_weights, draft_weights
):
    """Exercise actual CUDA kernels independently of model/logprob extraction.

    The first emitted token is an unconditional target sample whether it came
    from an accepted proposal or a rejection bonus. Checking only accepted
    tokens, or only bonuses, would compare biased conditional distributions.
    """
    batch_size, block_size = 65536, 4
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(408)
    p = torch.tensor(target_weights, dtype=torch.float32, device=device)
    p /= p.sum()
    q = torch.tensor(draft_weights, dtype=torch.float32, device=device)
    q /= q.sum()
    proposals = torch.multinomial(
        q, batch_size * (block_size - 1), replacement=True, generator=generator
    ).view(batch_size, block_size - 1)
    candidates = torch.cat(
        [torch.zeros(batch_size, 1, dtype=torch.int64, device=device), proposals],
        dim=1,
    )
    retrieve_index = torch.arange(
        batch_size * block_size, dtype=torch.int64, device=device
    ).view(batch_size, block_size)
    retrieve_next_token = (
        torch.tensor([1, 2, 3, -1], dtype=torch.int64, device=device)
        .expand(batch_size, -1)
        .contiguous()
    )
    retrieve_next_sibling = torch.full_like(retrieve_index, -1)
    predicts = torch.full(
        (batch_size * block_size,), -1, dtype=torch.int32, device=device
    )
    accept_index = torch.full(
        (batch_size, block_size), -1, dtype=torch.int32, device=device
    )
    accept_count = torch.zeros(batch_size, dtype=torch.int32, device=device)
    target_probs = p.expand(batch_size, block_size, -1).contiguous()
    draft_probs = (
        torch.zeros_like(target_probs)
        if verifier == "target_only"
        else q.expand(batch_size, block_size - 1, -1).contiguous()
    )
    # Target-only addresses a coin by verify position; rejection addresses one
    # by proposal position. Both read at most block_size - 1 coins on a chain.
    coins = torch.rand(batch_size, block_size, device=device, generator=generator)
    final_coins = torch.rand(batch_size, device=device, generator=generator)
    kernel = (
        tree_speculative_sampling_target_only
        if verifier == "target_only"
        else chain_speculative_sampling_triton
    )
    kernel(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_count,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=final_coins,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    first_tokens = predicts[accept_index[:, 0].long()]
    assert ((first_tokens >= 0) & (first_tokens < len(p))).all()
    assert (p[first_tokens.long()] > 0).all()
    frequencies = torch.bincount(first_tokens.long(), minlength=len(p)) / batch_size
    # Six binomial standard errors per category, with a one-count floor.
    tolerance = 6 * torch.sqrt(p * (1 - p) / batch_size) + 1 / batch_size
    assert ((frequencies - p).abs() <= tolerance).all(), (frequencies, p)
    assert ((accept_count >= 0) & (accept_count < block_size)).all()


@pytest.mark.parametrize("filter_order", ["top_k_first", "joint"])
@pytest.mark.parametrize("top_p", [1.0, 0.55, 0.8])
def test_sparse_filtering_matches_flashinfer_at_cutoff_ties(filter_order, top_p):
    """Verify support equality as well as probability closeness at tied cutoffs."""
    from flashinfer.sampling import top_k_renorm_probs, top_p_renorm_probs

    logits = torch.tensor(
        [[0.4, 0.2, 0.2, 0.1, 0.1], [0.1, 0.1, 0.3, 0.3, 0.2]],
        device="cuda",
    ).log()
    top_ks = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    top_ps = torch.full((2,), top_p, device="cuda")
    sampling_info = SimpleNamespace(
        temperatures=torch.ones(2, 1, device="cuda"),
        top_ks=top_ks,
        top_ps=top_ps,
        need_top_k_sampling=True,
        need_top_p_sampling=top_p < 1.0,
    )
    expected = logits.softmax(-1)
    if filter_order == "joint" and top_p < 1.0:
        expected = top_p_renorm_probs(expected, top_ps)
    expected = top_k_renorm_probs(expected, top_ks)
    if filter_order == "top_k_first" and top_p < 1.0:
        expected = top_p_renorm_probs(expected, top_ps)
    for sparse in (False, True):
        actual = build_speculative_verify_target_probs(
            next_token_logits=logits,
            sampling_info=sampling_info,
            draft_token_num=1,
            bs=2,
            max_top_k=3,
            use_sparse_topk=sparse,
            filter_apply_order=filter_order,
        ).squeeze(1)
        assert torch.equal(actual > 0, expected > 0)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("verifier", ["target_only", "rejection"])
@pytest.mark.parametrize(
    "accept_proposal", [False, True], ids=["rejection-bonus", "all-accepted-bonus"]
)
def test_qwen_vocabulary_bonus_cdf_boundaries(verifier, accept_proposal):
    """Pin bonus CDF traversal across tiles and the final partial vocabulary tile.

    Binary-exact masses and stratified midpoint uniforms make every output
    deterministic. The largest case uses about 1 GiB for target/draft matrices.
    """
    batch_size, block_size, vocab_size = 256, 2, 248320
    device = "cuda"
    proposal = 17  # Outside the bonus distribution's support.
    support = torch.tensor(
        [0, 4095, 4096, 8191, 8192, 131072, vocab_size - 2, vocab_size - 1],
        device=device,
    )
    weights = (
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 4], dtype=torch.float32, device=device) / 32
    )
    target_probs = torch.zeros(batch_size, block_size, vocab_size, device=device)
    target_probs[:, 1, support] = weights
    if accept_proposal:
        target_probs[:, 0, proposal] = 1.0
    else:
        target_probs[:, 0, support] = weights
    draft_probs = torch.zeros(
        batch_size,
        block_size if verifier == "target_only" else block_size - 1,
        vocab_size,
        device=device,
    )
    if verifier == "rejection":
        draft_probs[:, 0, proposal] = 1.0
    candidates = (
        torch.tensor([0, proposal], device=device).expand(batch_size, -1).contiguous()
    )
    retrieve_index = torch.arange(batch_size * block_size, device=device).view(
        batch_size, block_size
    )
    retrieve_next_token = (
        torch.tensor([1, -1], device=device).expand(batch_size, -1).contiguous()
    )
    retrieve_next_sibling = torch.full_like(retrieve_index, -1)
    predicts = torch.full(
        (batch_size * block_size,), -1, dtype=torch.int32, device=device
    )
    accept_index = torch.full(
        (batch_size, block_size), -1, dtype=torch.int32, device=device
    )
    accept_count = torch.zeros(batch_size, dtype=torch.int32, device=device)
    final_coins = (
        torch.arange(batch_size, dtype=torch.float32, device=device) + 0.5
    ) / batch_size
    kernel = (
        tree_speculative_sampling_target_only
        if verifier == "target_only"
        else chain_speculative_sampling_triton
    )
    kernel(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_count,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=torch.full((batch_size, block_size), 0.5, device=device),
        uniform_samples_for_final_sampling=final_coins,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    assert torch.all(accept_count == int(accept_proposal))
    row_ids = torch.arange(batch_size, device=device)
    bonus = predicts[accept_index[row_ids, accept_count.long()].long()]
    expected = support[torch.searchsorted(weights.cumsum(0), final_coins, right=True)]
    torch.testing.assert_close(bonus.long(), expected, rtol=0, atol=0)
    if accept_proposal:
        assert torch.all(predicts[accept_index[:, 0].long()] == proposal)
