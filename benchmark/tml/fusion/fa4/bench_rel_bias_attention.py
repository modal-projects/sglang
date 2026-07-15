"""Per-kernel bench: Inkling relative-attention bias path — sheared-bias vs score-mod.

Mirrors test_varlen_rel_bias_matches_score_mod: the same rel_logits tensor is
fed to the FA4 kernel as a pre-sheared bias (rel_bias=) and as a score-mod
callback (get_inkling_relative_attention_score_mod + aux_tensors). Causal varlen,
b equal-length square sequences.

  shear              -> vendored flash_attn_varlen_func(rel_bias=...)   [new sheared-bias path]
  score_mod_shear    -> vendored flash_attn_varlen_func(score_mod=..., aux_tensors=[...])  [today's production path]
  score_mod_baseline -> baseline sgl-flash-attn flash_attn_varlen_func score-mod

shear and score_mod_shear run on the same vendored kernel, so their delta is
purely sheared-bias-vs-score-mod. shear / score_mod_shear print PASS/FAIL of
their output vs the baseline sgl-flash-attn score-mod kernel.

Run:  python benchmark/tml/fusion/bench_rel_bias_attention.py

b200
[correctness] rel_bias_attn b=1 s=512 [64h/8kv d=128 re=128] shear: PASS (abs_max=0.000e+00)
[correctness] rel_bias_attn b=1 s=8192 [64h/8kv d=128 re=128] shear: PASS (abs_max=0.000e+00)
[correctness] rel_bias_attn b=1 s=131072 [64h/8kv d=128 re=128] shear: PASS (abs_max=0.000e+00)
[correctness] rel_bias_attn b=4 s=512 [64h/8kv d=128 re=128] shear: PASS (abs_max=0.000e+00)
[correctness] rel_bias_attn b=4 s=8192 [64h/8kv d=128 re=128] shear: PASS (abs_max=0.000e+00)
[correctness] rel_bias_attn b=1 s=512 [64h/8kv d=128 re=128] score_mod_shear: PASS (abs_max=0.000e+00)
[correctness] rel_bias_attn b=1 s=8192 [64h/8kv d=128 re=128] score_mod_shear: PASS (abs_max=0.000e+00)
[correctness] rel_bias_attn b=1 s=131072 [64h/8kv d=128 re=128] score_mod_shear: PASS (abs_max=0.000e+00)
[correctness] rel_bias_attn b=4 s=512 [64h/8kv d=128 re=128] score_mod_shear: PASS (abs_max=0.000e+00)
[correctness] rel_bias_attn b=4 s=8192 [64h/8kv d=128 re=128] score_mod_shear: PASS (abs_max=0.000e+00)
=======================================================================================================================================================
          b         s |       shear(us)  score_mod_shear(us)  score_mod_baseline(us) |     shear(GB/s)  score_mod_shear(GB/s)  score_mod_baseline(GB/s)
-------------------------------------------------------------------------------------------------------------------------------------------------------
0         1       512 |         32.7591             100.5812                100.5826 |          775.07                 252.44                    252.44
1         1      8192 |       1191.8538           11004.8637              11009.0237 |          340.86                  36.92                     36.90
2         1    131072 |      228081.665           2754350.10              2754313.96 |           28.50                   2.36                      2.36
3         4       512 |         96.0106             252.7437                252.4979 |         1057.83                 401.84                    402.23
4         4      8192 |       4785.9714           43934.5932              43930.6564 |          339.53                  36.99                     36.99
5         4    131072 |             N/A                  N/A                     N/A |             N/A                    N/A                       N/A
=======================================================================================================================================================
"""

import os
import sys
import time

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.flash_attn.cute import flash_attn_varlen_func as _shear_fn
from sglang.srt.models.inkling_common.attn import (
    get_inkling_relative_attention_score_mod,
)
from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench_refs import report

try:
    from flash_attn.cute import flash_attn_varlen_func as _baseline_fn
except Exception:
    _baseline_fn = None

DEV = "cuda"
NH, NHK, D = 64, 8, 128  # Inkling full-attn: 64 q heads, 8 kv heads, head_dim 128
REL_EXTENT = 128  # must be a multiple of 128
SEQLENS = [512, 8192, 131072]
BATCHES = [1, 4]
MAX_TOKENS = 282144  # skip batch*seqlen beyond this to bound memory


def _score_mod():
    # get_inkling_relative_attention_score_mod needs the fork's SeqlenInfoQK type;
    # returns None if the fork isn't importable so score_mod impls skip cleanly.
    try:
        return get_inkling_relative_attention_score_mod(REL_EXTENT)
    except Exception:
        return None


@marker.parametrize("b", BATCHES)
@marker.parametrize("s", SEQLENS)
@marker.benchmark("impl", ["shear", "score_mod_shear", "score_mod_baseline"])
def bench_rel_bias_attention(s: int, b: int, impl: str):
    if b * s > MAX_TOKENS:
        marker.skip(f"batch*seqlen {b * s} > {MAX_TOKENS}")
    sm = _score_mod()
    if impl == "score_mod_baseline" and _baseline_fn is None:
        marker.skip("baseline flash_attn (flash-attn-4-forward) not installed")
    if impl == "score_mod_shear" and sm is None:
        marker.skip("score_mod requires the flash_attn fork's SeqlenInfoQK")

    torch.manual_seed(0)
    total = b * s
    cu = torch.arange(0, total + 1, s, dtype=torch.int32, device=DEV)  # b seqs of len s
    q = torch.randn(total, NH, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(total, NHK, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(total, NHK, D, device=DEV, dtype=torch.bfloat16)
    bias = torch.randn(total, NH, REL_EXTENT, device=DEV, dtype=torch.bfloat16) * 0.1
    scale = D**-0.5
    common = dict(
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=s,
        max_seqlen_k=s,
        softmax_scale=scale,
        causal=True,
        num_splits=0,
    )

    if impl == "shear":
        fn = lambda q, k, v, x: _shear_fn(q, k, v, rel_bias=x, **common)
    else:
        kern = _shear_fn if impl == "score_mod_shear" else _baseline_fn
        fn = lambda q, k, v, x: kern(q, k, v, score_mod=sm, aux_tensors=[x], **common)

    out = fn(q, k, v, bias)  # first call pays JIT compile; time the second
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(q, k, v, bias)
    torch.cuda.synchronize()
    per_call = time.perf_counter() - t0
    # Correctness vs the baseline fork's score-mod kernel (skip the baseline impl
    # itself -- it would only compare against itself).
    if impl != "score_mod_baseline" and _baseline_fn is not None and sm is not None:
        tag = f"rel_bias_attn b={b} s={s} [{NH}h/{NHK}kv d={D} re={REL_EXTENT}] {impl}"
        ref = _baseline_fn(q, k, v, score_mod=sm, aux_tensors=[bias], **common)
        report(tag, out[0], ref[0])
    # ponytail: ~30s budget per impl. The cuda-graph path ignores small replay_iters
    # (loop_count floor = 100 calls/graph, >=10 replays => >=1100 calls; ~50min for
    # score_mod s=131k), so for slow calls use the naive loop where launch overhead
    # is noise anyway.
    replay = max(10, min(1000, int(30.0 / per_call)))
    use_graph = per_call < 10e-3
    print(
        f"[bench] {impl} b={b} s={s}: per_call={per_call * 1e3:.1f}ms "
        f"replay={replay} cuda_graph={use_graph}",
        flush=True,
    )
    return marker.do_bench(
        fn,
        input_args=(q, k, v, bias),
        warmup_iters=max(3, replay // 20),
        replay_iters=replay,
        use_cuda_graph=use_graph,
    )


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    bench_rel_bias_attention.run()
