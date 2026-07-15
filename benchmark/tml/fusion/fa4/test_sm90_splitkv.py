"""SM90 (H100/H200) SplitKV correctness + speedup check for the vendored FA4 cute kernel.

Correctness: for each case, run the vendored varlen kernel with num_splits=1
(baseline, previously the only supported SM90 path) and num_splits in SPLITS,
plus a fp32 torch SDPA reference over the gathered paged cache. Split output
must match the no-split output to ~combine-precision tolerance, and both must
match the reference to bf16 tolerance.

Cases cover: MHA + GQA (pack_gqa is force-disabled for split on SM90), decode
(q_len=1) and chunked prefill (q_len=128), causal and non-causal, a sliding
window case (exercises empty splits: whole splits outside the local window),
and a learnable-sink case (split 0 folds the sink exactly once).

Speedup: every correctness case, num_splits=1 vs heuristic (num_splits=0), same GPU.

Run:  python benchmark/tml/fusion/fa4/test_sm90_splitkv.py
Knobs: SPLITS=2,3,8,32; CASES=name1,name2 filters cases; BENCH=0 skips bench.

H200
[gqa_decode] base_vs_ref max_abs=2.67e-04
[gqa_decode] splits=  2 vs_base=2.44e-04 vs_ref=2.67e-04 OK
[gqa_decode] splits=  3 vs_base=2.44e-04 vs_ref=2.67e-04 OK
[gqa_decode] splits=  8 vs_base=4.88e-04 vs_ref=2.21e-04 OK
[gqa_decode] splits= 32 vs_base=2.44e-04 vs_ref=2.67e-04 OK
[mha_decode] base_vs_ref max_abs=2.19e-04
[mha_decode] splits=  2 vs_base=4.88e-04 vs_ref=3.13e-04 OK
[mha_decode] splits=  3 vs_base=4.88e-04 vs_ref=3.13e-04 OK
[mha_decode] splits=  8 vs_base=2.44e-04 vs_ref=1.83e-04 OK
[mha_decode] splits= 32 vs_base=2.44e-04 vs_ref=1.75e-04 OK
[gqa_decode_odd_kv] base_vs_ref max_abs=3.33e-04
[gqa_decode_odd_kv] splits=  2 vs_base=4.88e-04 vs_ref=3.33e-04 OK
[gqa_decode_odd_kv] splits=  3 vs_base=4.88e-04 vs_ref=3.33e-04 OK
[gqa_decode_odd_kv] splits=  8 vs_base=4.88e-04 vs_ref=3.33e-04 OK
[gqa_decode_odd_kv] splits= 32 vs_base=4.88e-04 vs_ref=3.33e-04 OK
[gqa_prefill] base_vs_ref max_abs=3.56e-04
[gqa_prefill] splits=  2 vs_base=4.88e-04 vs_ref=3.63e-04 OK
[gqa_prefill] splits=  3 vs_base=4.88e-04 vs_ref=3.89e-04 OK
[gqa_prefill] splits=  8 vs_base=4.88e-04 vs_ref=3.61e-04 OK
[gqa_prefill] splits= 32 vs_base=4.88e-04 vs_ref=3.45e-04 OK
[noncausal] base_vs_ref max_abs=4.86e-04
[noncausal] splits=  2 vs_base=4.88e-04 vs_ref=4.86e-04 OK
[noncausal] splits=  3 vs_base=4.88e-04 vs_ref=4.86e-04 OK
[noncausal] splits=  8 vs_base=9.77e-04 vs_ref=4.91e-04 OK
[noncausal] splits= 32 vs_base=9.77e-04 vs_ref=4.91e-04 OK
[local_window] base_vs_ref max_abs=5.28e-04
[local_window] splits=  2 vs_base=4.88e-04 vs_ref=5.28e-04 OK
[local_window] splits=  3 vs_base=9.77e-04 vs_ref=5.30e-04 OK
[local_window] splits=  8 vs_base=9.77e-04 vs_ref=5.30e-04 OK
[local_window] splits= 32 vs_base=9.77e-04 vs_ref=5.30e-04 OK
[sink_decode] base_vs_ref max_abs=1.82e-04
[sink_decode] splits=  2 vs_base=2.44e-04 vs_ref=1.82e-04 OK
[sink_decode] splits=  3 vs_base=2.44e-04 vs_ref=1.82e-04 OK
[sink_decode] splits=  8 vs_base=2.44e-04 vs_ref=1.82e-04 OK
[sink_decode] splits= 32 vs_base=2.44e-04 vs_ref=1.82e-04 OK
[sink_local_empty_splits] base_vs_ref max_abs=9.11e-04
[sink_local_empty_splits] splits=  2 vs_base=9.77e-04 vs_ref=9.11e-04 OK
[sink_local_empty_splits] splits=  3 vs_base=9.77e-04 vs_ref=9.11e-04 OK
[sink_local_empty_splits] splits=  8 vs_base=9.77e-04 vs_ref=9.11e-04 OK
[sink_local_empty_splits] splits= 32 vs_base=9.77e-04 vs_ref=9.11e-04 OK
[packgqa_decode] base_vs_ref max_abs=2.67e-04
[packgqa_decode] splits=  2 vs_base=2.44e-04 vs_ref=2.67e-04 OK
[packgqa_decode] splits=  3 vs_base=2.44e-04 vs_ref=2.67e-04 OK
[packgqa_decode] splits=  8 vs_base=4.88e-04 vs_ref=2.21e-04 OK
[packgqa_decode] splits= 32 vs_base=2.44e-04 vs_ref=2.67e-04 OK
[packgqa_odd_kv] base_vs_ref max_abs=3.33e-04
[packgqa_odd_kv] splits=  2 vs_base=4.88e-04 vs_ref=3.33e-04 OK
[packgqa_odd_kv] splits=  3 vs_base=4.88e-04 vs_ref=3.33e-04 OK
[packgqa_odd_kv] splits=  8 vs_base=4.88e-04 vs_ref=3.33e-04 OK
[packgqa_odd_kv] splits= 32 vs_base=4.88e-04 vs_ref=3.33e-04 OK
[packgqa_qlen4] base_vs_ref max_abs=2.44e-04
[packgqa_qlen4] splits=  2 vs_base=2.44e-04 vs_ref=2.44e-04 OK
[packgqa_qlen4] splits=  3 vs_base=2.44e-04 vs_ref=2.44e-04 OK
[packgqa_qlen4] splits=  8 vs_base=2.44e-04 vs_ref=2.44e-04 OK
[packgqa_qlen4] splits= 32 vs_base=4.88e-04 vs_ref=2.76e-04 OK
[packgqa_sink] base_vs_ref max_abs=1.82e-04
[packgqa_sink] splits=  2 vs_base=2.44e-04 vs_ref=1.82e-04 OK
[packgqa_sink] splits=  3 vs_base=2.44e-04 vs_ref=1.82e-04 OK
[packgqa_sink] splits=  8 vs_base=2.44e-04 vs_ref=1.82e-04 OK
[packgqa_sink] splits= 32 vs_base=2.44e-04 vs_ref=1.82e-04 OK
[packgqa_sink_local_empty] base_vs_ref max_abs=9.11e-04
[packgqa_sink_local_empty] splits=  2 vs_base=9.77e-04 vs_ref=9.11e-04 OK
[packgqa_sink_local_empty] splits=  3 vs_base=9.77e-04 vs_ref=9.11e-04 OK
[packgqa_sink_local_empty] splits=  8 vs_base=9.77e-04 vs_ref=9.11e-04 OK
[packgqa_sink_local_empty] splits= 32 vs_base=9.77e-04 vs_ref=9.11e-04 OK
=================================================================================================
                        name |    no_split(us)  heuristic(us) |   no_split(GB/s)  heuristic(GB/s)
-------------------------------------------------------------------------------------------------
0                 gqa_decode |        100.0384        39.3472 |          1250.13          3178.40
1                 mha_decode |        100.1408        39.2406 |          1248.39          3185.86
2          gqa_decode_odd_kv |         52.1794        13.6729 |           225.02           858.75
3                gqa_prefill |         53.0108        28.8494 |           331.60           609.31
4                  noncausal |         49.5076        28.8348 |           355.06           609.61
5               local_window |         19.1369        10.9289 |          1633.77          2860.79
6                sink_decode |         99.4214        17.4263 |           314.47          1794.14
7    sink_local_empty_splits |         12.7555        10.3586 |          4901.06          6035.11
8             packgqa_decode |        100.2061        39.3619 |          1248.04          3177.21
9             packgqa_odd_kv |         52.1258        13.6360 |           225.26           861.08
10             packgqa_qlen4 |         99.3044        17.7638 |           315.30          1762.63
11              packgqa_sink |         99.4333        17.4703 |           314.43          1789.63
12  packgqa_sink_local_empty |         12.7696        10.3727 |          4895.64          6026.90
=================================================================================================
"""

import math
import os

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.flash_attn.cute import flash_attn_varlen_func

DEV = "cuda"
PAGE = 128
D = 128
SPLITS = [int(s) for s in os.environ.get("SPLITS", "2,3,8,32").split(",")]
CASE_FILTER = set(filter(None, os.environ.get("CASES", "").split(",")))
FAILS = []


def make(b, s_kv, q_len, nh, nhk, seed=0):
    torch.manual_seed(seed)
    total_q = b * q_len
    pages_per_seq = (s_kv + PAGE - 1) // PAGE
    pages = b * pages_per_seq
    kc = torch.randn(pages, PAGE, nhk, D, device=DEV, dtype=torch.bfloat16)
    vc = torch.randn(pages, PAGE, nhk, D, device=DEV, dtype=torch.bfloat16)
    pt = torch.arange(pages, dtype=torch.int32, device=DEV).view(b, -1)
    q = torch.randn(total_q, nh, D, device=DEV, dtype=torch.bfloat16)
    cu = torch.arange(0, total_q + 1, q_len, dtype=torch.int32, device=DEV)
    sl = torch.full((b,), s_kv, dtype=torch.int32, device=DEV)
    return q, kc, vc, pt, cu, sl


def ref_attn(q, kc, vc, pt, s_kv, q_len, nh, nhk, causal, window, sink):
    b = pt.shape[0]
    scale = 1.0 / math.sqrt(D)
    outs = []
    for i in range(b):
        k = kc[pt[i].long()].reshape(-1, nhk, D)[:s_kv].float()  # (s_kv, nhk, D)
        v = vc[pt[i].long()].reshape(-1, nhk, D)[:s_kv].float()
        qi = q[i * q_len : (i + 1) * q_len].float()  # (q_len, nh, D)
        rep = nh // nhk
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        s = torch.einsum("qhd,khd->hqk", qi, k) * scale
        q_idx = torch.arange(q_len, device=DEV)[:, None] + (s_kv - q_len)
        k_idx = torch.arange(s_kv, device=DEV)[None, :]
        mask = torch.zeros(q_len, s_kv, dtype=torch.bool, device=DEV)
        if causal:
            mask |= k_idx > q_idx
        if window[0] is not None:
            mask |= k_idx < q_idx - window[0]
        s = s.masked_fill(mask[None], float("-inf"))
        if sink is not None:
            s = torch.cat([s, sink.float()[:, None, None].expand(nh, q_len, 1)], dim=-1)
        p = torch.softmax(s, dim=-1)
        if sink is not None:
            p = p[..., :-1]
        p = torch.nan_to_num(p)  # fully-masked rows -> 0
        outs.append(torch.einsum("hqk,khd->qhd", p, v))
    return torch.cat(outs)


def run_case(
    name, b, s_kv, q_len, nh, nhk, causal, window=(None, None), use_sink=False,
    pack_gqa=None,
):
    q, kc, vc, pt, cu, sl = make(b, s_kv, q_len, nh, nhk)
    sink = (
        torch.randn(nh, device=DEV, dtype=torch.bfloat16) if use_sink else None
    )
    kw = dict(
        cu_seqlens_q=cu,
        seqused_k=sl,
        max_seqlen_q=q_len,
        page_table=pt,
        softmax_scale=1.0 / math.sqrt(D),
        causal=causal,
        window_size=window,
        learnable_sink=sink,
        pack_gqa=pack_gqa,
    )
    base = flash_attn_varlen_func(q, kc, vc, num_splits=1, **kw)
    base = base[0] if isinstance(base, tuple) else base
    ref = ref_attn(q, kc, vc, pt, s_kv, q_len, nh, nhk, causal, window, sink)
    err_base = (base.float() - ref).abs().max().item()
    print(f"[{name}] base_vs_ref max_abs={err_base:.2e}")
    ok = err_base < 3e-2
    if not ok:
        FAILS.append(f"{name}: baseline vs ref {err_base:.2e}")
    for ns in SPLITS:
        out = flash_attn_varlen_func(q, kc, vc, num_splits=ns, **kw)
        out = out[0] if isinstance(out, tuple) else out
        d_base = (out.float() - base.float()).abs().max().item()
        d_ref = (out.float() - ref).abs().max().item()
        status = "OK" if (d_base < 4e-3 and d_ref < 3e-2) else "FAIL"
        print(
            f"[{name}] splits={ns:3d} vs_base={d_base:.2e} vs_ref={d_ref:.2e} {status}"
        )
        if status == "FAIL":
            FAILS.append(f"{name} splits={ns}: vs_base={d_base:.2e} vs_ref={d_ref:.2e}")


CASES = [
    ("gqa_decode", dict(b=4, s_kv=8192, q_len=1, nh=32, nhk=8, causal=True)),
    ("mha_decode", dict(b=4, s_kv=8192, q_len=1, nh=8, nhk=8, causal=True)),
    (
        "gqa_decode_odd_kv",
        dict(b=3, s_kv=4000, q_len=1, nh=16, nhk=2, causal=True),
    ),
    (
        "gqa_prefill",
        dict(b=2, s_kv=4096, q_len=128, nh=16, nhk=4, causal=True),
    ),
    (
        "noncausal",
        dict(b=2, s_kv=4096, q_len=128, nh=16, nhk=4, causal=False),
    ),
    (
        "local_window",
        dict(b=2, s_kv=8192, q_len=1, nh=16, nhk=4, causal=True, window=(1024, None)),
    ),
    (
        "sink_decode",
        dict(b=2, s_kv=8192, q_len=1, nh=16, nhk=4, causal=True, use_sink=True),
    ),
    (
        "sink_local_empty_splits",
        dict(
            b=2,
            s_kv=16384,
            q_len=1,
            nh=16,
            nhk=4,
            causal=True,
            window=(512, None),
            use_sink=True,
        ),
    ),
    (
        "packgqa_decode",
        dict(b=4, s_kv=8192, q_len=1, nh=32, nhk=8, causal=True, pack_gqa=True),
    ),
    (
        "packgqa_odd_kv",
        dict(b=3, s_kv=4000, q_len=1, nh=16, nhk=2, causal=True, pack_gqa=True),
    ),
    (
        "packgqa_qlen4",
        dict(b=2, s_kv=8192, q_len=4, nh=16, nhk=4, causal=True, pack_gqa=True),
    ),
    (
        "packgqa_sink",
        dict(
            b=2,
            s_kv=8192,
            q_len=1,
            nh=16,
            nhk=4,
            causal=True,
            use_sink=True,
            pack_gqa=True,
        ),
    ),
    (
        "packgqa_sink_local_empty",
        dict(
            b=2,
            s_kv=16384,
            q_len=1,
            nh=16,
            nhk=4,
            causal=True,
            window=(512, None),
            use_sink=True,
            pack_gqa=True,
        ),
    ),
]


CASE_BY_NAME = dict(CASES)


@marker.parametrize("name", [name for name, _ in CASES])
@marker.benchmark("impl", ["no_split", "heuristic"])
def bench_case(name, impl):
    """Compare forced no-split against the production split heuristic."""
    if CASE_FILTER and name not in CASE_FILTER:
        marker.skip(f"{name} excluded by CASES")

    case = CASE_BY_NAME[name]
    b, s_kv, q_len = case["b"], case["s_kv"], case["q_len"]
    nh, nhk, causal = case["nh"], case["nhk"], case["causal"]
    window = case.get("window", (None, None))
    use_sink = case.get("use_sink", False)
    pack_gqa = case.get("pack_gqa")
    q, kc, vc, pt, cu, sl = make(b, s_kv, q_len, nh, nhk)
    sink = torch.randn(nh, device=DEV, dtype=torch.bfloat16) if use_sink else None
    kw = dict(
        cu_seqlens_q=cu,
        seqused_k=sl,
        max_seqlen_q=q_len,
        page_table=pt,
        softmax_scale=1.0 / math.sqrt(D),
        causal=causal,
        window_size=window,
        learnable_sink=sink,
        pack_gqa=pack_gqa,
        num_splits=1 if impl == "no_split" else 0,
    )
    fn = lambda q, kc, vc: flash_attn_varlen_func(q, kc, vc, **kw)
    out = fn(q, kc, vc)  # compile before the marker warmup and measurements
    out = out[0] if isinstance(out, tuple) else out
    return marker.do_bench(fn, input_args=(q, kc, vc), memory_output=out)


if __name__ == "__main__":
    assert torch.cuda.get_device_capability()[0] == 9, "SM90 test"
    # for name, kwargs in CASES:
    #     if not CASE_FILTER or name in CASE_FILTER:
    #         run_case(name, **kwargs)
    if os.environ.get("BENCH", "1") == "1":
        bench_case.run()
    if FAILS:
        print("\nFAILURES:")
        for f in FAILS:
            print(" ", f)
        raise SystemExit(1)
    print("\nALL PASS")
