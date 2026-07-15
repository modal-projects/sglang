"""Per-kernel bench: Inkling relative-attention bias on the PAGED KV-CACHE path.

This is the decode / paged-prefill counterpart to bench_rel_bias_attention.py. It
exercises exactly the path the model uses (flash_attn_with_kvcache, ver=4, with a
page_table) -- the path that the raw-K/V varlen bench never touches -- so a CUDA
error in the model reproduces here, small and fast.

  shear      -> flash_attn_with_kvcache(rel_bias=...)            [new sheared-bias path]
  score_mod  -> flash_attn_with_kvcache(score_mod=..., aux=[...]) [today's production path]

Both run on the SAME vendored FA4 kernel, so their delta is purely sheared-bias vs
score-mod. `shear` prints PASS/FAIL of its output vs the `score_mod` output.

Run:  python benchmark/tml/fusion/bench_rel_bias_attention_kvcache.py

B200
NUM_SPLITS=0
[correctness] rel_bias_kvcache b=1 q=1 kv=512 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
[correctness] rel_bias_kvcache b=1 q=256 kv=512 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
[correctness] rel_bias_kvcache b=1 q=1 kv=8192 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
[correctness] rel_bias_kvcache b=1 q=256 kv=8192 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
[correctness] rel_bias_kvcache b=8 q=1 kv=512 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
[correctness] rel_bias_kvcache b=8 q=256 kv=512 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
[correctness] rel_bias_kvcache b=8 q=1 kv=8192 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
[correctness] rel_bias_kvcache b=64 q=1 kv=512 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
[correctness] rel_bias_kvcache b=64 q=1 kv=8192 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
======================================================================================================================================
           b      s_kv     q_len |   score_mod(us)      shear(us)  shear_prep(us) |   score_mod(GB/s)    shear(GB/s)  shear_prep(GB/s)
--------------------------------------------------------------------------------------------------------------------------------------
0          1       512         1 |         40.6304        16.8982          5.2445 |             49.20         118.29            381.14
1          1       512       256 |        109.7413        27.4188          9.3415 |            124.58         498.63           1463.56
2          1      8192         1 |         49.3547        24.9073          4.0951 |            634.10        1256.49           7642.28
3          1      8192       256 |        830.6243        84.9525          9.1956 |             51.73         505.80           4672.75
4          8       512         1 |         32.2469        20.8456          5.3421 |            495.90         767.13           2993.41
5          8       512       256 |        374.6436       114.3484         47.4916 |            291.94         956.51           2303.04
6          8      8192         1 |        266.3360        63.9907          5.3453 |            940.04        3912.54          46838.74
7          8      8192       256 |             N/A            N/A             N/A |               N/A            N/A               N/A
8         64       512         1 |        171.6797        72.1616         21.6627 |            745.16        1772.82           5905.52
9         64       512       256 |             N/A            N/A             N/A |               N/A            N/A               N/A
10        64      8192         1 |       1659.3207       480.0934         21.7606 |           1207.08        4171.96          92043.69
11        64      8192       256 |             N/A            N/A             N/A |               N/A            N/A               N/A
======================================================================================================================================

h200
[correctness] rel_bias_kvcache b=1 q=1 kv=512 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
[correctness] rel_bias_kvcache b=1 q=256 kv=512 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
[correctness] rel_bias_kvcache b=1 q=1 kv=8192 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
[correctness] rel_bias_kvcache b=1 q=256 kv=8192 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
[correctness] rel_bias_kvcache b=8 q=1 kv=512 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
[correctness] rel_bias_kvcache b=8 q=256 kv=512 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
[correctness] rel_bias_kvcache b=8 q=1 kv=8192 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
[correctness] rel_bias_kvcache b=64 q=1 kv=512 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
[correctness] rel_bias_kvcache b=64 q=1 kv=8192 [64h/8kv d=128 re=128]: PASS (abs_max=0.000e+00)
======================================================================================================================================
           b      s_kv     q_len |   score_mod(us)      shear(us)  shear_prep(us) |   score_mod(GB/s)    shear(GB/s)  shear_prep(GB/s)
--------------------------------------------------------------------------------------------------------------------------------------
0          1       512         1 |         25.0013        17.8912          4.1920 |             79.95         111.73            476.84
1          1       512       256 |         27.6062        25.1115          9.8119 |            495.25         544.45           1393.40
2          1      8192         1 |         31.7553        25.0153          3.7583 |            985.53        1251.07           8327.21
3          1      8192       256 |        319.0720       127.4190          9.5630 |            134.67         337.22           4493.25
4          8       512         1 |         20.9094        23.8128          5.7386 |            764.78         671.54           2786.62
5          8       512       256 |        180.3341       154.4390         52.6480 |            606.51         708.21           2077.48
6          8      8192         1 |        168.2962        80.3213          5.7365 |           1487.65        3117.06          43644.64
7          8      8192       256 |             N/A            N/A             N/A |               N/A            N/A               N/A
8         64       512         1 |        106.5658        85.2874         24.4269 |           1200.48        1499.98           5237.25
9         64       512       256 |             N/A            N/A             N/A |               N/A            N/A               N/A
10        64      8192         1 |       1270.5991       619.5760         24.4371 |           1576.37        3232.74          81962.59
11        64      8192       256 |             N/A            N/A             N/A |               N/A            N/A               N/A
======================================================================================================================================

"""

import os
import sys
from functools import cache

from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

import torch

from sglang.jit_kernel.benchmark import marker

# Dispatcher (ver=4) -- the exact entry point the fa4 backend calls in the model.
from sglang.jit_kernel.flash_attention import flash_attn_with_kvcache as _kvcache_fn
from sglang.srt.models.inkling_common.attn import (
    get_inkling_relative_attention_score_mod,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench_refs import report

DEV = "cuda"
NH, NHK, D = 64, 8, 128  # Inkling full-attn: 64 q heads, 8 kv heads, head_dim 128
REL_EXTENT = 128  # must be a multiple of 128
PAGE_SIZE = 128  # == tile_n so the FA4 paged-TMA path is used
KV_SEQLENS = [512, 8192]  # cached length per sequence
BATCHES = [1, 8, 64]
QLENS = [1, 256]  # 1 = decode, >1 = chunked-prefill against an existing cache
NUM_SPLITS = int(os.environ.get("BENCH_NUM_SPLITS", "0"))  # 0 = auto-split heuristic
NO_BIAS = (
    os.environ.get("BENCH_NO_BIAS") == "0"
)  # plain attention baseline (no bias/score_mod)
MAX_TOKENS = 131072  # skip batch*qlen*kv beyond a memory bound
_REFS = {}


@cache
def _score_mod():
    try:
        return get_inkling_relative_attention_score_mod(REL_EXTENT)
    except Exception:
        return None


def _make_paged_kv(b: int, s_kv: int):
    """Allocate a contiguous paged KV cache for b sequences each of length s_kv."""
    pages_per_seq = (s_kv + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = b * pages_per_seq
    k_cache = torch.randn(
        total_pages, PAGE_SIZE, NHK, D, device=DEV, dtype=torch.bfloat16
    )
    v_cache = torch.randn(
        total_pages, PAGE_SIZE, NHK, D, device=DEV, dtype=torch.bfloat16
    )
    page_table = torch.arange(total_pages, dtype=torch.int32, device=DEV).view(
        b, pages_per_seq
    )
    cache_seqlens = torch.full((b,), s_kv, dtype=torch.int32, device=DEV)
    return k_cache, v_cache, page_table, cache_seqlens


def _make_case(b: int, s_kv: int, q_len: int):
    """Build once per shape; both impls use identical tensors."""
    torch.manual_seed(0)
    total_q = b * q_len
    k_cache, v_cache, page_table, cache_seqlens = _make_paged_kv(b, s_kv)
    q = torch.randn(total_q, NH, D, device=DEV, dtype=torch.bfloat16)
    bias = torch.randn(total_q, NH, REL_EXTENT, device=DEV, dtype=torch.bfloat16) * 0.1
    cu_seqlens_q = torch.arange(0, total_q + 1, q_len, dtype=torch.int32, device=DEV)
    common = dict(
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=q_len,
        softmax_scale=D**-0.5,
        causal=True,
        num_splits=NUM_SPLITS,
        ver=4,
    )
    return q, k_cache, v_cache, bias, common


@marker.parametrize("b", BATCHES)
@marker.parametrize("s_kv", KV_SEQLENS)
@marker.parametrize("q_len", QLENS)
@marker.benchmark("impl", ["score_mod", "shear", "shear_prep"])
def bench_rel_bias_attention_kvcache(q_len: int, s_kv: int, b: int, impl: str):
    if b * q_len * s_kv > MAX_TOKENS * 32:
        marker.skip(f"batch*qlen*kv {b * q_len * s_kv} too large")
    if q_len > s_kv:
        marker.skip(f"q_len {q_len} > kv {s_kv}")
    sm = _score_mod()
    if impl == "score_mod" and sm is None:
        marker.skip("score_mod requires the flash_attn fork's SeqlenInfoQK")

    q, k_cache, v_cache, bias, common = _make_case(b, s_kv, q_len)
    case_key = (b, s_kv, q_len)

    if NO_BIAS:
        fn = lambda q, kc, vc, x: _kvcache_fn(q, kc, vc, **common)
    elif impl == "shear":
        fn = lambda q, kc, vc, x: _kvcache_fn(q, kc, vc, rel_bias=x, **common)
    elif impl == "shear_prep":
        # Just the shear-prep kernels (bias expansion), no attention kernel.
        def fn(q, kc, vc, x):
            os.environ["BIAS_PREP_ONLY"] = "1"
            try:
                return _kvcache_fn(q, kc, vc, rel_bias=x, **common)
            finally:
                del os.environ["BIAS_PREP_ONLY"]

    else:
        fn = lambda q, kc, vc, x: _kvcache_fn(
            q, kc, vc, score_mod=sm, aux_tensors=[x], **common
        )

    out = fn(q, k_cache, v_cache, bias)
    out = out[0] if isinstance(out, tuple) else out
    if impl == "score_mod":
        _REFS[case_key] = out.detach()

    # Correctness: sheared bias must match the score-mod path on the same kernel.
    if impl == "shear" and sm is not None and case_key in _REFS:
        tag = f"rel_bias_kvcache b={b} q={q_len} kv={s_kv} [{NH}h/{NHK}kv d={D} re={REL_EXTENT}]"
        report(tag, out, _REFS[case_key])
    return marker.do_bench(
        fn, input_args=(q, k_cache, v_cache, bias), memory_output=out
    )


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    bench_rel_bias_attention_kvcache.run()
