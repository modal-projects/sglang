"""Per-kernel bench: Inkling relative-attention (sheared bias) with fp8_e4m3 vs bf16 KV cache.

Measures the effect of the KV-cache dtype on the FA4 sheared-bias decode/prefill path
(paged KV + rel_bias). FA4 couples the Q/K/V dtype (the kernel asserts q_dtype == k_dtype),
so "fp8 KV cache" here means the whole attention runs in fp8_e4m3 with per-(batch, kv-head)
descales; bf16 is the reference. rel_bias stays bf16 in both.

  bf16      -> flash_attn_varlen_func(q,kc,vc bf16, rel_bias=..., page_table=...)
  fp8_e4m3  -> same, q/kc/vc = float8_e4m3fn + q/k/v_descale=ones([b, n_kv])

Calls the vendored cute flash_attn_varlen_func directly (paged) because the fa4 wrapper
blocks descale. The fp8 rows also report the output error vs the bf16 output on the same
underlying values (the fp8 quantization error, informational -- not a correctness gate).

Run:  python benchmark/tml/fusion/bench_rel_bias_attention_kvcache_fp8.py

B200
num split 0
[correctness] rel_bias_kvcache_fp8 b=1 q=1 kv=512 [64h/8kv d=128]: PASS (abs_max=1.187e-01)
[correctness] rel_bias_kvcache_fp8 b=1 q=256 kv=512 [64h/8kv d=128]: FAIL (abs_max=6.035e-01)
[correctness] rel_bias_kvcache_fp8 b=1 q=1 kv=8192 [64h/8kv d=128]: PASS (abs_max=3.055e-02)
[correctness] rel_bias_kvcache_fp8 b=1 q=256 kv=8192 [64h/8kv d=128]: PASS (abs_max=5.157e-02)
[correctness] rel_bias_kvcache_fp8 b=8 q=1 kv=512 [64h/8kv d=128]: PASS (abs_max=1.309e-01)
[correctness] rel_bias_kvcache_fp8 b=8 q=256 kv=512 [64h/8kv d=128]: FAIL (abs_max=8.193e-01)
[correctness] rel_bias_kvcache_fp8 b=8 q=1 kv=8192 [64h/8kv d=128]: PASS (abs_max=3.372e-02)
[correctness] rel_bias_kvcache_fp8 b=64 q=1 kv=512 [64h/8kv d=128]: PASS (abs_max=2.842e-01)
[correctness] rel_bias_kvcache_fp8 b=64 q=1 kv=8192 [64h/8kv d=128]: PASS (abs_max=5.597e-02)
===================================================================================================
           b      s_kv     q_len |        bf16(us)   fp8_e4m3(us) |      bf16(GB/s)  fp8_e4m3(GB/s)
---------------------------------------------------------------------------------------------------
0          1       512         1 |         17.1421        16.8141 |          116.61           60.35
1          1       512       256 |         28.8999        27.7816 |          473.08          386.67
2          1      8192         1 |         24.3709        23.6855 |         1284.15          661.30
3          1      8192       256 |         84.2707        83.0987 |          509.89          305.55
4          8       512         1 |         21.2216        20.2921 |          753.53          400.04
5          8       512       256 |        117.3333       110.4342 |          932.17          778.18
6          8      8192         1 |         63.8054        58.2278 |         3923.90         2151.98
7          8      8192       256 |             N/A            N/A |             N/A             N/A
8         64       512         1 |         66.7106        62.1289 |         1917.68         1045.27
9         64       512       256 |             N/A            N/A |             N/A             N/A
10        64      8192         1 |        457.9632       316.3958 |         4373.56         3168.31
11        64      8192       256 |             N/A            N/A |             N/A             N/A
===================================================================================================
"""

import os
import sys
from functools import cache

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

# Cute interface directly: supports fp8 + descale + rel_bias + page_table (the fa4
# wrapper rejects descale, so we bypass it here).
from sglang.jit_kernel.flash_attn.cute import flash_attn_varlen_func as _fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench_refs import report

DEV = "cuda"
NH, NHK, D = 64, 8, 128  # Inkling full-attn: 64 q heads, 8 kv heads, head_dim 128
REL_EXTENT = 128  # must be a multiple of 128
PAGE_SIZE = 128  # == tile_n -> FA4 paged-TMA path
KV_SEQLENS = [512, 8192]  # cached length per sequence
BATCHES = [1, 8, 64]
QLENS = [1, 256]  # 1 = decode, >1 = chunked-prefill against an existing cache
NUM_SPLITS = 0  # auto-split heuristic (decode uses split-KV)
MAX_TOKENS = 131072

_DTYPES = {"bf16": torch.bfloat16, "fp8_e4m3": torch.float8_e4m3fn}
_REFS = {}


@cache
def _build_base(b: int, s_kv: int, q_len: int):
    """bf16 base tensors (randn ~O(1) fits fp8_e4m3, so no pre-scaling needed)."""
    torch.manual_seed(0)
    pages_per_seq = (s_kv + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = b * pages_per_seq
    total_q = b * q_len
    q = torch.randn(total_q, NH, D, device=DEV, dtype=torch.bfloat16)
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
    cu_seqlens_q = torch.arange(0, total_q + 1, q_len, dtype=torch.int32, device=DEV)
    bias = torch.randn(total_q, NH, REL_EXTENT, device=DEV, dtype=torch.bfloat16) * 0.1
    return q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, bias


@cache
def _build_case(b: int, s_kv: int, q_len: int, kv_dtype: str):
    """Build once per shape/dtype; fp8 descales are benchmark inputs, not per-call work."""
    dtype = _DTYPES[kv_dtype]
    q_bf, kc_bf, vc_bf, page_table, cache_seqlens, cu_seqlens_q, bias = _build_base(
        b, s_kv, q_len
    )
    q, k_cache, v_cache = [t.to(dtype) for t in (q_bf, kc_bf, vc_bf)]
    kw = dict(
        page_table=page_table,
        seqused_k=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=q_len,
        softmax_scale=D**-0.5,
        causal=True,
        num_splits=NUM_SPLITS,
    )
    if dtype == torch.float8_e4m3fn:
        ones = torch.ones(b, NHK, device=DEV, dtype=torch.float32)
        kw.update(q_descale=ones, k_descale=ones, v_descale=ones)
    return q, k_cache, v_cache, bias, kw


def _run(q, k_cache, v_cache, bias, kw):
    out = _fn(q, k_cache, v_cache, rel_bias=bias, **kw)
    return out[0] if isinstance(out, tuple) else out


@marker.parametrize("b", BATCHES)
@marker.parametrize("s_kv", KV_SEQLENS)
@marker.parametrize("q_len", QLENS)
@marker.benchmark("kv_dtype", ["bf16", "fp8_e4m3"])
def bench_rel_bias_kvcache_fp8(q_len: int, s_kv: int, b: int, kv_dtype: str):
    if b * q_len * s_kv > MAX_TOKENS * 32:
        marker.skip(f"batch*qlen*kv {b * q_len * s_kv} too large")
    if q_len > s_kv:
        marker.skip(f"q_len {q_len} > kv {s_kv}")

    q, kc, vc, bias, kw = _build_case(b, s_kv, q_len, kv_dtype)
    case_key = (b, s_kv, q_len)
    fn = lambda q, kc, vc, x: _run(q, kc, vc, x, kw)

    out = fn(q, kc, vc, bias)
    if kv_dtype == "bf16":
        _REFS[case_key] = out.detach()
    if kv_dtype == "fp8_e4m3":
        if case_key in _REFS:
            tag = (
                f"rel_bias_kvcache_fp8 b={b} q={q_len} kv={s_kv} [{NH}h/{NHK}kv d={D}]"
            )
            report(
                tag, out.float(), _REFS[case_key].float(), tol=5e-1
            )  # fp8 quant error, informational
    return marker.do_bench(fn, input_args=(q, kc, vc, bias), memory_output=out)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    bench_rel_bias_kvcache_fp8.run()
