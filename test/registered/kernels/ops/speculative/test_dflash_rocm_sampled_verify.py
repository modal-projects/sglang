import os
import sys

import pytest
import torch

from sglang.kernels.ops.speculative.dflash_rocm_sampled_verify import (
    dflash_sampled_verify_rocm,
)
from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.layers.sampler import top_k_top_p_normalize_probs_torch
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"
# The Triton interpreter runs one program at a time; keep CPU runs small.
SCALE = 16 if INTERPRET else 1


def _target_probs(logits, temps, top_ks, top_ps, draft):
    rep = lambda x: torch.repeat_interleave(x, draft, dim=0)
    probs = torch.softmax(logits / rep(temps)[:, None], dim=-1)
    return top_k_top_p_normalize_probs_torch(probs, rep(top_ks), rep(top_ps))


def _reference(logits, candidates, temps, top_ks, top_ps, coins, coins_final):
    """The prior ROCm path: dense target probs + one-hot draft chain sampler."""
    bs, d = candidates.shape
    target = _target_probs(logits, temps, top_ks, top_ps, d).view(bs, d, -1)
    draft = torch.zeros_like(target)
    draft[:, :-1].scatter_(2, candidates[:, 1:, None].long(), 1.0)
    idx = torch.arange(bs * d, device=logits.device, dtype=torch.int64).view(bs, d)
    predicts = torch.empty(bs * d, dtype=torch.int32, device=logits.device)
    accept_index = torch.full((bs, d), -1, dtype=torch.int32, device=logits.device)
    accept_len = torch.empty(bs, dtype=torch.int32, device=logits.device)
    chain_speculative_sampling_triton(
        predicts,
        accept_index,
        accept_len,
        candidates.long(),
        idx,
        None,
        None,
        coins,
        coins_final,
        target.contiguous(),
        draft,
        1.0,
        1.0,
        True,
    )
    rows = torch.arange(bs, device=logits.device)
    bonus = predicts[accept_index[rows, accept_len.long()].long()].long()
    return accept_len, bonus


def _fused(logits, candidates, temps, top_ks, top_ps, coins, coins_final):
    return dflash_sampled_verify_rocm(
        logits=logits,
        candidates=candidates,
        temperatures=temps,
        top_ks=top_ks,
        top_ps=top_ps,
        coins=coins,
        coins_final=coins_final,
    )


def _case(bs, d, vocab, *, temp, top_k, top_p, scale, seed, accept_bias=0.7):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(bs * d, vocab, generator=g) * scale
    temps = torch.full((bs,), temp)
    top_ks = torch.full((bs,), top_k, dtype=torch.int32)
    top_ps = torch.full((bs,), top_p)
    # Drafts mostly propose the target argmax so chains run deep.
    greedy = logits.view(bs, d, vocab).argmax(-1)
    rand = torch.randint(0, vocab, (bs, d), generator=g)
    use_greedy = torch.rand(bs, d, generator=g) < accept_bias
    candidates = torch.where(use_greedy, torch.roll(greedy, 1, dims=1), rand)
    coins = torch.rand(bs, d, generator=g)
    coins_final = torch.rand(bs, generator=g)
    to = lambda x: x.to(DEVICE)
    return tuple(
        map(to, (logits, candidates, temps, top_ks, top_ps, coins, coins_final))
    )


@pytest.mark.parametrize(
    "vocab,top_k,top_p,scale",
    [
        (5003, 1 << 30, 0.95, 4.0),  # K3 default: top_p only
        (5003, 1 << 30, 1.0, 4.0),  # no filtering
        (5003, 40, 0.95, 4.0),
        (5003, 1, 1.0, 4.0),  # greedy-equivalent
        (4096, 1 << 30, 0.5, 0.05),  # flat: nucleus spills past the coarse search
        (20000, 1 << 30, 0.9, 1.0),
    ],
)
def test_matches_reference_fixed_rng(vocab, top_k, top_p, scale):
    args = _case(
        64 // (SCALE // 4 or 1),
        8,
        vocab,
        temp=0.7,
        top_k=top_k,
        top_p=top_p,
        scale=scale,
        seed=vocab,
    )
    ref_len, ref_bonus = _reference(*args)
    len_, bonus = _fused(*args)
    # Same coins give the same decisions up to fp rounding at CDF boundaries.
    assert (len_ == ref_len).float().mean().item() >= 0.98
    same = len_ == ref_len
    assert (bonus[same] == ref_bonus[same]).float().mean().item() >= 0.98
    # The bonus must lie in the filtered support of its row.
    bs, d = args[1].shape
    target = _target_probs(args[0], args[2], args[3], args[4], d).view(bs, d, -1)
    rows = torch.arange(bs, device=bonus.device)
    p_bonus = target[rows, len_.long(), bonus]
    assert (p_bonus > 0).all()


def test_top_k_one_is_argmax():
    args = _case(64, 8, 3001, temp=0.7, top_k=1, top_p=1.0, scale=4.0, seed=3)
    logits, candidates = args[0], args[1]
    len_, bonus = _fused(*args)
    bs, d = candidates.shape
    greedy = logits.view(bs, d, -1).argmax(-1)
    match = (candidates[:, 1:] == greedy[:, :-1]).int().cumprod(dim=1).sum(dim=1)
    assert torch.equal(len_.long(), match.long())
    rows = torch.arange(bs, device=bonus.device)
    assert torch.equal(bonus, greedy[rows, len_.long()])


def test_top_k_ties_keep_exactly_k():
    bs, d, vocab = 4096 // SCALE, 2, 64
    logits = torch.zeros(bs * d, vocab, device=DEVICE)
    candidates = torch.full((bs, d), vocab - 1, device=DEVICE)  # outside kept set
    temps = torch.ones(bs, device=DEVICE)
    top_ks = torch.full((bs,), 3, dtype=torch.int32, device=DEVICE)
    top_ps = torch.ones(bs, device=DEVICE)
    coins = torch.rand(bs, d, device=DEVICE)
    len_, bonus = _fused(
        logits, candidates, temps, top_ks, top_ps, coins, torch.rand(bs, device=DEVICE)
    )
    assert (len_ == 0).all()
    assert set(bonus.tolist()) == {0, 1, 2}


def _chi2_pvalue(counts, probs, trials):
    scipy_stats = pytest.importorskip("scipy.stats")
    expected = probs / probs.sum() * trials
    keep = expected > 0
    assert counts[~keep].sum() == 0
    return scipy_stats.chisquare(counts[keep], expected[keep]).pvalue


@pytest.mark.parametrize("top_k,top_p", [(1 << 30, 0.9), (6, 1.0), (6, 0.8)])
def test_emitted_tokens_follow_target(top_k, top_p):
    """Emitted tokens are distributed exactly as sampling from the target."""
    vocab, d, trials, temp = 12, 3, 200_000 // SCALE, 0.7
    g = torch.Generator().manual_seed(top_k + int(top_p * 100))
    row_logits = torch.randn(d, vocab, generator=g) * 1.5
    # Root then a draft that is not the target argmax, then the argmax of row 1.
    probs = _target_probs(
        row_logits,
        torch.tensor([temp]),
        torch.tensor([top_k], dtype=torch.int32),
        torch.tensor([top_p]),
        d,
    ).double()
    draft_1 = int(torch.argsort(probs[0], descending=True)[1])
    draft_2 = int(probs[1].argmax())
    candidates = torch.tensor([[0, draft_1, draft_2]]).expand(trials, d).contiguous()
    logits = row_logits.repeat(trials, 1)
    out_len, bonus = _fused(
        logits.to(DEVICE),
        candidates.to(DEVICE),
        torch.full((trials,), temp, device=DEVICE),
        torch.full((trials,), top_k, dtype=torch.int32, device=DEVICE),
        torch.full((trials,), top_p, device=DEVICE),
        torch.rand(trials, d, device=DEVICE),
        torch.rand(trials, device=DEVICE),
    )
    out_len, bonus = out_len.cpu().long(), bonus.cpu()
    first = torch.where(out_len >= 1, torch.full_like(bonus, draft_1), bonus)
    counts = torch.bincount(first, minlength=vocab).double()
    assert _chi2_pvalue(counts, probs[0], trials) > 1e-4
    # Given the first draft was accepted, the next token follows row 1.
    acc = out_len >= 1
    second = torch.where(
        out_len[acc] >= 2, torch.full_like(bonus[acc], draft_2), bonus[acc]
    )
    counts = torch.bincount(second, minlength=vocab).double()
    assert _chi2_pvalue(counts, probs[1], int(acc.sum())) > 1e-4


def test_bf16_ties_match_pytorch_filter():
    """Exact bf16 ties at the top-p pivot: keep only the ties the sampler's sort keeps."""
    rows, vocab = 4000 // SCALE, 512
    g = torch.Generator().manual_seed(0)
    logits = torch.round(torch.randn(rows, vocab, generator=g) * 8) / 4
    logits = logits.bfloat16().float().to(DEVICE)
    temps = torch.ones(rows, device=DEVICE)
    top_ks = torch.full((rows,), 1 << 30, dtype=torch.int32, device=DEVICE)
    top_ps = torch.full((rows,), 0.95, device=DEVICE)
    kept = _target_probs(logits, temps, top_ks, top_ps, 1) > 0
    probs = torch.softmax(logits, dim=-1)
    pivot = torch.where(kept, probs, torch.inf).min(dim=-1, keepdim=True).values
    r, t = torch.nonzero(probs == pivot, as_tuple=True)
    assert (~kept[r, t]).any(), "no dropped ties at the pivot; test is vacuous"
    q = r.numel()
    len_, _ = _fused(
        logits[r].repeat_interleave(2, dim=0),
        torch.stack([torch.zeros_like(t), t], dim=1),
        temps[r],
        top_ks[r],
        top_ps[r],
        torch.zeros(q, 2, device=DEVICE),  # coin 0 accepts iff the draft is kept
        torch.rand(q, device=DEVICE),
    )
    assert torch.equal(len_ >= 1, kept[r, t])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
