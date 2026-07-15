"""Unit-test vendored FA4 sheared bias against the pip score_mod baseline on SERVING shapes.

The installed pip flash_attn.cute package does not expose rel_bias, so this test
uses pip score_mod as the reference and exercises the vendored sheared-bias path
with rel_bias=...

This closes the remaining serving-surface gaps where the sheared-bias path could
still diverge e2e:
  - LSE output (used by the backend), not just O
  - decode batch sizes incl. large-bs split heuristics, windowed (local) layers
  - chunked-prefill-sized varlen and ragged mixed batches
  - CUDA graph capture + replay (the serving execution mode), with input
    mutation between replays and a shrunk seqused_k (decode-graph semantics)

Outputs must be bitwise identical.

Run:  python benchmark/tml/fusion/test_vendored_vs_pip_serving.py
"""

import torch

from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

from flash_attn.cute import flash_attn_varlen_func as pip_fn

from sglang.jit_kernel.flash_attn.cute import flash_attn_varlen_func as ven_fn
from sglang.srt.models.inkling_common.attn import (
    get_inkling_relative_attention_score_mod,
)

DEV = "cuda"
NH, NHK, D = 64, 8, 128
PAGE_SIZE = 128
REL_EXTENT = 1024

FAILS = []


def make(b, s_kv, q_len, seed=0):
    torch.manual_seed(seed)
    total_q = b * q_len
    pages = b * ((s_kv + PAGE_SIZE - 1) // PAGE_SIZE)
    kc = torch.randn(pages, PAGE_SIZE, NHK, D, device=DEV, dtype=torch.bfloat16)
    vc = torch.randn(pages, PAGE_SIZE, NHK, D, device=DEV, dtype=torch.bfloat16)
    pt = torch.arange(pages, dtype=torch.int32, device=DEV).view(b, -1)
    q = torch.randn(total_q, NH, D, device=DEV, dtype=torch.bfloat16)
    bias = torch.randn(total_q, NH, REL_EXTENT, device=DEV, dtype=torch.bfloat16)
    cu = torch.arange(0, total_q + 1, q_len, dtype=torch.int32, device=DEV)
    sl = torch.full((b,), s_kv, dtype=torch.int32, device=DEV)
    return q, kc, vc, bias, pt, cu, sl


def check(tag, rv, rp):
    ov, lv = rv if isinstance(rv, tuple) else (rv, None)
    op, lp = rp if isinstance(rp, tuple) else (rp, None)
    out_diff = int((ov != op).sum().item())
    out_err = (ov.float() - op.float()).abs().max().item()
    lse_diff = 0
    lse_err = 0.0
    if lv is not None and lp is not None:
        lse_diff = int((lv != lp).sum().item())
        lse_err = (lv.float() - lp.float()).abs().max().item()
    ok = out_diff == 0 and lse_diff == 0
    status = "OK" if ok else "FAIL"
    print(
        f"  {tag:44s} {status} "
        f"out_diff={out_diff} out_abs={out_err:.3e} "
        f"lse_diff={lse_diff} lse_abs={lse_err:.3e}"
    )
    if not ok:
        FAILS.append(tag)


def eager_sweep(sm):
    for b, s_kv, q_len, windowed in [
        (1, 1211, 1, False),
        (64, 4097, 1, False),
        (128, 2048, 1, False),
        (64, 4097, 1, True),
        (1, 8261, 8192, False),  # chunked-prefill-sized
        (1, 8261, 8192, True),
        (8, 517, 256, False),
    ]:
        q, kc, vc, bias, pt, cu, sl = make(b, s_kv, q_len)
        kw = dict(
            cu_seqlens_q=cu,
            seqused_k=sl,
            max_seqlen_q=q_len,
            page_table=pt,
            softmax_scale=1.0 / D,
            causal=True,
            num_splits=0,
            return_lse=True,
        )
        if windowed:
            kw["window_size"] = (REL_EXTENT - 1, 0)
        tag = f"eager b={b} kv={s_kv} q={q_len} win={windowed} rel_bias"
        check(
            tag,
            ven_fn(q, kc, vc, rel_bias=bias, **kw),
            pip_fn(q, kc, vc, score_mod=sm, aux_tensors=[bias], **kw),
        )
        del q, kc, vc, bias
        torch.cuda.empty_cache()


def ragged(sm):
    q_lens = [1, 7, 256, 33]
    kv_lens = [1211, 517, 8261, 4097]
    torch.manual_seed(1)
    b = len(q_lens)
    total_q = sum(q_lens)
    pages_per = max((kv + PAGE_SIZE - 1) // PAGE_SIZE for kv in kv_lens)
    kc = torch.randn(b * pages_per, PAGE_SIZE, NHK, D, device=DEV, dtype=torch.bfloat16)
    vc = torch.randn(b * pages_per, PAGE_SIZE, NHK, D, device=DEV, dtype=torch.bfloat16)
    pt = torch.arange(b * pages_per, dtype=torch.int32, device=DEV).view(b, -1)
    q = torch.randn(total_q, NH, D, device=DEV, dtype=torch.bfloat16)
    bias = torch.randn(total_q, NH, REL_EXTENT, device=DEV, dtype=torch.bfloat16)
    cu = (
        torch.tensor([0] + q_lens, dtype=torch.int32, device=DEV)
        .cumsum(0)
        .to(torch.int32)
    )
    sl = torch.tensor(kv_lens, dtype=torch.int32, device=DEV)
    kw = dict(
        cu_seqlens_q=cu,
        seqused_k=sl,
        max_seqlen_q=max(q_lens),
        page_table=pt,
        softmax_scale=1.0 / D,
        causal=True,
        num_splits=0,
        return_lse=True,
    )
    check(
        "eager ragged mixed batch rel_bias",
        ven_fn(q, kc, vc, rel_bias=bias, **kw),
        pip_fn(q, kc, vc, score_mod=sm, aux_tensors=[bias], **kw),
    )
    torch.cuda.empty_cache()


def graph_case(sm, b, s_kv, q_len, tag):
    """Capture both impls, replay with mutated inputs and shrunk seqused_k."""
    q, kc, vc, bias, pt, cu, sl = make(b, s_kv, q_len)
    kw = dict(
        cu_seqlens_q=cu,
        seqused_k=sl,
        max_seqlen_q=q_len,
        page_table=pt,
        softmax_scale=1.0 / D,
        causal=True,
        num_splits=0,
        return_lse=True,
    )

    outs = {}
    for name, fn, extra in (
        ("ven", ven_fn, dict(rel_bias=bias)),
        ("pip", pip_fn, dict(score_mod=sm, aux_tensors=[bias])),
    ):
        fn(q, kc, vc, **extra, **kw)  # warm compile outside capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            res = fn(q, kc, vc, **extra, **kw)
        # replay 1: same inputs
        g.replay()
        torch.cuda.synchronize()
        o1 = (res[0].clone(), res[1].clone())
        # replay 2: mutate q/bias in place + shrink seqused_k (decode-graph semantics)
        torch.manual_seed(7)
        q.copy_(torch.randn_like(q))
        bias.copy_(torch.randn_like(bias))
        sl.fill_(s_kv - 173)
        g.replay()
        torch.cuda.synchronize()
        o2 = (res[0].clone(), res[1].clone())
        # restore for the other impl
        torch.manual_seed(0)
        q2, _, _, b2, _, _, _ = make(b, s_kv, q_len)
        q.copy_(q2)
        bias.copy_(b2)
        sl.fill_(s_kv)
        outs[name] = (o1, o2)
        del g
        torch.cuda.synchronize()

    check(f"graph {tag} replay1", outs["ven"][0], outs["pip"][0])
    check(f"graph {tag} replay2 (mutated+shrunk kv)", outs["ven"][1], outs["pip"][1])
    torch.cuda.empty_cache()


def main():
    sm = get_inkling_relative_attention_score_mod(REL_EXTENT)
    eager_sweep(sm)
    ragged(sm)
    graph_case(sm, 64, 4097, 1, "decode bs=64 kv=4097")
    graph_case(sm, 1, 8261, 1024, "prefill q=1024 kv=8261")
    print("ALL OK" if not FAILS else f"FAILURES: {FAILS}")
    raise SystemExit(1 if FAILS else 0)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
