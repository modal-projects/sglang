"""Fused ROCm sampled verify with a sparse sampled draft q (DFlash2 selector) vs
the dense path the selector uses today (scatter q -> DSpark accept_sampling's
pytorch filter + chain_speculative_sampling_triton)."""

import os, sys, time, json
import pytest
import torch

from sglang.kernels.ops.speculative.dflash_rocm_sampled_verify import (
    dflash_sampled_verify_rocm,
)
from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.layers.sampler import top_k_top_p_normalize_probs_torch
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd-mi35x")

DEVICE = "cuda"


def _target_probs(logits, temps, top_ks, top_ps, d):
    rep = lambda x: torch.repeat_interleave(x, d, dim=0)
    probs = torch.softmax(logits / rep(temps)[:, None], dim=-1)
    return top_k_top_p_normalize_probs_torch(probs, rep(top_ks), rep(top_ps))


def _dense_q(ids, q, vocab):
    bs, g, _ = ids.shape
    dq = torch.zeros(bs, g, vocab, dtype=torch.float32, device=ids.device)
    dq.scatter_(-1, ids.long(), q.float())
    return dq


def _reference(logits, cands, temps, ks, ps, coins, coins_final, ids, q):
    bs, d = cands.shape
    vocab = logits.shape[-1]
    target = _target_probs(logits, temps, ks, ps, d).view(bs, d, -1).contiguous()
    draft = _dense_q(ids, q, vocab)
    idx = torch.arange(bs * d, device=logits.device, dtype=torch.int64).view(bs, d)
    predicts = torch.empty(bs * d, dtype=torch.int32, device=logits.device)
    accept_index = torch.full((bs, d), -1, dtype=torch.int32, device=logits.device)
    accept_len = torch.empty(bs, dtype=torch.int32, device=logits.device)
    chain_speculative_sampling_triton(
        predicts,
        accept_index,
        accept_len,
        cands.long(),
        idx,
        None,
        None,
        coins,
        coins_final,
        target,
        draft,
        1.0,
        1.0,
        True,
    )
    rows = torch.arange(bs, device=logits.device)
    bonus = predicts[accept_index[rows, accept_len.long()].long()].long()
    return accept_len, bonus


def _fused(logits, cands, temps, ks, ps, coins, coins_final, ids, q, **kw):
    return dflash_sampled_verify_rocm(
        logits=logits,
        candidates=cands,
        temperatures=temps,
        top_ks=ks,
        top_ps=ps,
        coins=coins,
        coins_final=coins_final,
        draft_ids=ids,
        draft_q=q,
        **kw,
    )


def _case(
    bs, d, vocab, K, *, temp, top_k, top_p, scale, seed, dtype=torch.float32, noise=0.5
):
    """Selector-like draft: K candidates per row drawn from the target's top
    region (plus noise), q a softmax over them, and the drafted token sampled
    from q, so q(token) > 0 always."""
    g = torch.Generator().manual_seed(seed)
    logits = (torch.randn(bs * d, vocab, generator=g) * scale).to(dtype).float()
    temps = torch.full((bs,), temp)
    ks = torch.full((bs,), top_k, dtype=torch.int32)
    ps = torch.full((bs,), top_p)
    lv = logits.view(bs, d, vocab)
    noisy = lv[:, :-1] + torch.randn(bs, d - 1, vocab, generator=g) * scale * noise
    ids = noisy.topk(K, dim=-1).indices  # [bs, d-1, K], unique per row
    # q: a draft that roughly tracks the target over its K candidates.
    q = torch.softmax(
        lv[:, :-1].gather(-1, ids) / temp
        + torch.randn(bs, d - 1, K, generator=g) * 0.5,
        dim=-1,
    )
    pick = torch.multinomial(q.view(-1, K), 1, generator=g).view(bs, d - 1)
    tok = ids.gather(-1, pick[..., None])[..., 0]
    root = torch.randint(0, vocab, (bs, 1), generator=g)
    cands = torch.cat([root, tok], dim=1)
    coins = torch.rand(bs, d, generator=g)
    coins_final = torch.rand(bs, generator=g)
    to = lambda x: x.to(DEVICE)
    return tuple(map(to, (logits, cands, temps, ks, ps, coins, coins_final, ids, q)))


@pytest.mark.parametrize(
    "vocab,top_k,top_p,scale,dtype",
    [
        (163840, 1 << 30, 0.95, 4.0, torch.bfloat16),  # K3 vocab, top_p only
        (163840, 1 << 30, 1.0, 4.0, torch.bfloat16),  # no filtering
        (163840, 40, 0.95, 4.0, torch.float32),
        (20000, 1 << 30, 0.9, 1.0, torch.float32),
        (4096, 1 << 30, 0.5, 0.05, torch.float32),  # flat nucleus
    ],
)
@pytest.mark.parametrize("K", [8, 5])
@pytest.mark.parametrize("noise", [0.5, 0.02])
def test_matches_dense_reference_fixed_rng(vocab, top_k, top_p, scale, dtype, K, noise):
    args = _case(
        64,
        8,
        vocab,
        K,
        temp=0.7,
        top_k=top_k,
        top_p=top_p,
        scale=scale,
        seed=vocab + K,
        dtype=dtype,
        noise=noise,
    )
    ref_len, ref_bonus = _reference(*args)
    len_, bonus = _fused(*args)
    agree_len = (len_ == ref_len).float().mean().item()
    same = len_ == ref_len
    agree_bonus = (bonus[same] == ref_bonus[same]).float().mean().item()
    print(
        f"vocab={vocab} K={K} noise={noise} full={(len_ == 7).float().mean():.2f} len_agree={agree_len:.4f} bonus_agree={agree_bonus:.4f} "
        f"mean_len={len_.float().mean():.2f}/{ref_len.float().mean():.2f}"
    )
    assert agree_len >= 0.98 and agree_bonus >= 0.98
    # Bonus must be in the support of the residual (or of p when all accepted).
    bs, d = args[1].shape
    target = _target_probs(args[0], args[2], args[3], args[4], d).view(bs, d, -1)
    dq = _dense_q(args[7], args[8], vocab)
    rows = torch.arange(bs, device=bonus.device)
    p_b = target[rows, len_.long(), bonus]
    full = len_ == d - 1
    q_b = torch.where(
        full, torch.zeros_like(p_b), dq[rows, len_.clamp(max=d - 2).long(), bonus]
    )
    assert (p_b - q_b > 0).all()


def _chi2_pvalue(counts, probs, trials):
    from scipy import stats

    expected = probs / probs.sum() * trials
    keep = expected > 0
    assert counts[~keep].sum() == 0, "emitted token outside target support"
    return stats.chisquare(counts[keep], expected[keep]).pvalue


@pytest.mark.parametrize(
    "top_k,top_p", [(1 << 30, 0.9), (6, 1.0), (6, 0.8), (1 << 30, 1.0)]
)
def test_emitted_tokens_follow_target_sampled_q(top_k, top_p):
    """Draft token ~ q (sparse, K=4); first emitted token must be ~ target row 0,
    and given acceptance, the second ~ target row 1."""
    vocab, d, K, trials, temp = 12, 3, 4, 400_000, 0.7
    g = torch.Generator().manual_seed(top_k + int(top_p * 100))
    row_logits = torch.randn(d, vocab, generator=g) * 1.5
    probs = _target_probs(
        row_logits,
        torch.tensor([temp]),
        torch.tensor([top_k], dtype=torch.int32),
        torch.tensor([top_p]),
        d,
    ).double()
    # q puts mass on 4 tokens incl. one outside the filtered support when possible.
    ids_row = torch.stack(
        [
            torch.argsort(probs[r], descending=True)[[0, 1, 2, vocab - 1]]
            for r in range(d - 1)
        ]
    )
    q_row = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.5, 0.1, 0.3, 0.1]])
    ids = ids_row[None].expand(trials, d - 1, K).contiguous()
    q = q_row[None].expand(trials, d - 1, K).contiguous()
    pick = torch.multinomial(q.view(-1, K), 1).view(trials, d - 1)
    tok = ids.gather(-1, pick[..., None])[..., 0]
    cands = torch.cat([torch.zeros(trials, 1, dtype=torch.long), tok], dim=1)
    out_len, bonus = _fused(
        row_logits.repeat(trials, 1).to(DEVICE),
        cands.to(DEVICE),
        torch.full((trials,), temp, device=DEVICE),
        torch.full((trials,), top_k, dtype=torch.int32, device=DEVICE),
        torch.full((trials,), top_p, device=DEVICE),
        torch.rand(trials, d, device=DEVICE),
        torch.rand(trials, device=DEVICE),
        ids.to(DEVICE),
        q.to(DEVICE),
    )
    out_len, bonus = out_len.cpu().long(), bonus.cpu()
    first = torch.where(out_len >= 1, tok[:, 0], bonus)
    p1 = _chi2_pvalue(torch.bincount(first, minlength=vocab).double(), probs[0], trials)
    acc = out_len >= 1
    second = torch.where(out_len[acc] >= 2, tok[acc, 1], bonus[acc])
    p2 = _chi2_pvalue(
        torch.bincount(second, minlength=vocab).double(), probs[1], int(acc.sum())
    )
    print(
        f"chi2 top_k={top_k} top_p={top_p} p1={p1:.3g} p2={p2:.3g} acc_rate={acc.float().mean():.3f}"
    )
    assert p1 > 1e-4 and p2 > 1e-4


def test_full_vocab_chi2_k3():
    """Full-vocab (163840) chi-square on the first emitted token, bf16 logits,
    top_p=0.95, real selector-like sparse q; tokens binned by target rank."""
    from scipy import stats

    vocab, d, K, trials, temp = 163840, 2, 8, 200_000, 0.7
    g = torch.Generator().manual_seed(7)
    row = (torch.randn(d, vocab, generator=g) * 3.0).bfloat16().float()
    # Support from the same torch filter on the GPU, as the dense serving path.
    probs = (
        _target_probs(
            row.to(DEVICE),
            torch.tensor([temp], device=DEVICE),
            torch.tensor([1 << 30], dtype=torch.int32, device=DEVICE),
            torch.tensor([0.95], device=DEVICE),
            d,
        )
        .double()
        .cpu()
    )
    order = torch.argsort(probs[0], descending=True)
    ids_row = order[[0, 1, 2, 3, 5, 8, 13, vocab - 1]][None]
    q_row = torch.softmax(torch.randn(1, K, generator=g), -1)
    ids = ids_row[None].expand(trials, 1, K).contiguous()
    q = q_row[None].expand(trials, 1, K).contiguous()
    pick = torch.multinomial(q.view(-1, K), 1).view(trials, 1)
    tok = ids.gather(-1, pick[..., None])[..., 0]
    cands = torch.cat([torch.zeros(trials, 1, dtype=torch.long), tok], dim=1)
    chunk = 2000
    logits = row.to(DEVICE).repeat(chunk, 1)
    lens, bons = [], []
    for c0 in range(0, trials, chunk):
        n = min(chunk, trials - c0)
        l, b = _fused(
            logits[: n * d],
            cands[c0 : c0 + n].to(DEVICE),
            torch.full((n,), temp, device=DEVICE),
            torch.full((n,), 1 << 30, dtype=torch.int32, device=DEVICE),
            torch.full((n,), 0.95, device=DEVICE),
            torch.rand(n, d, device=DEVICE),
            torch.rand(n, device=DEVICE),
            ids[c0 : c0 + n].to(DEVICE),
            q[c0 : c0 + n].to(DEVICE),
        )
        lens.append(l.cpu())
        bons.append(b.cpu())
    out_len, bonus = torch.cat(lens), torch.cat(bons)
    first = torch.where(out_len >= 1, tok[:, 0], bonus)
    out = probs[0][first] <= 0
    if out.any():
        rk = torch.empty(vocab, dtype=torch.long)
        rk[torch.argsort(probs[0], descending=True)] = torch.arange(vocab)
        print(
            "outside support:",
            int(out.sum()),
            "ranks",
            rk[first[out]][:10].tolist(),
            "support",
            int((probs[0] > 0).sum()),
            "accepted",
            bool((out_len[out] >= 1).any()),
        )
    assert not out.any(), "token outside support"
    rank = torch.empty(vocab, dtype=torch.long)
    rank[order] = torch.arange(vocab)
    r = rank[first]
    support = int((probs[0] > 0).sum())
    # bins: top 32 ranks individually, then geometric bins over the rest of the support
    edges = list(range(33))
    e = 32
    while e < support:
        e = min(support, int(e * 1.6) + 1)
        edges.append(e)
    edges = torch.tensor(edges)
    cnt = torch.histogram(r.double(), bins=edges.double()).hist
    ps = probs[0][order]
    exp = (
        torch.stack([ps[edges[i] : edges[i + 1]].sum() for i in range(len(edges) - 1)])
        * trials
    )
    keep = exp > 5
    pv = stats.chisquare(
        cnt[keep], exp[keep] * cnt[keep].sum() / exp[keep].sum()
    ).pvalue
    print(f"full-vocab chi2 support={support} bins={int(keep.sum())} p={pv:.3g}")
    assert pv > 1e-4


def bench():
    """ms per verify call: fused sparse-q vs today's selector path (dense q
    scatter + accept_sampling with the pytorch filter)."""
    from sglang.kernels.ops.speculative.dspark.dspark_accept import (
        _accept_sampling_core,
    )  # noqa

    out = {}
    for bs in (1, 4, 16, 32, 64):
        args = _case(
            bs,
            8,
            163840,
            8,
            temp=0.7,
            top_k=1 << 30,
            top_p=0.95,
            scale=4.0,
            seed=bs,
            dtype=torch.bfloat16,
        )
        logits = args[0].bfloat16().float()

        def ref():
            # mirrors _selector_sampling_accept (scatter, filter, chain kernel, zero)
            return _reference(*args)

        def fus():
            return _fused(*args)

        res = {}
        for name, fn in (("dense_ref", ref), ("fused", fus)):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            n = 50
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            st.record()
            for _ in range(n):
                fn()
            en.record()
            torch.cuda.synchronize()
            res[name] = st.elapsed_time(en) / n
        out[bs] = res
        print(
            f"bs={bs} dense_ref={res['dense_ref']:.3f} ms fused={res['fused']:.3f} ms"
        )
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        print(json.dumps(bench()))
    else:
        sys.exit(pytest.main([__file__, "-q", "-s"]))
