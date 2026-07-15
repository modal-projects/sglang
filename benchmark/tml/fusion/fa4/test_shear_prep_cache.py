"""Check rel_bias_prep_cache: one CuSeqlensToBlocks launch per step, not per layer.

Simulates N layers of one forward step by calling flash_attn_with_kvcache N times
with a shared rel_bias_prep_cache dict (as FlashAttentionBackend does via
FlashAttentionMetadata.rel_bias_prep_cache). Asserts:
  1. outputs are bitwise identical with and without the cache
  2. the profiler sees exactly 1 CuSeqlensToBlocks launch with the cache
     (vs N without)

Run:  CUDA_VISIBLE_DEVICES=<free> python benchmark/tml/fusion/fa4/test_shear_prep_cache.py
"""

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

from sglang.jit_kernel.flash_attention import flash_attn_with_kvcache

DEV = "cuda"
NH, NHK, D = 64, 8, 128
REL_EXTENT = 128
PAGE_SIZE = 128
N_LAYERS = 4
B, S_KV, Q_LEN = 8, 512, 1


def make_case():
    torch.manual_seed(0)
    total_q = B * Q_LEN
    pages_per_seq = (S_KV + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = B * pages_per_seq
    k_cache = torch.randn(total_pages, PAGE_SIZE, NHK, D, device=DEV, dtype=torch.bfloat16)
    v_cache = torch.randn(total_pages, PAGE_SIZE, NHK, D, device=DEV, dtype=torch.bfloat16)
    page_table = torch.arange(total_pages, dtype=torch.int32, device=DEV).view(B, pages_per_seq)
    cache_seqlens = torch.full((B,), S_KV, dtype=torch.int32, device=DEV)
    q = torch.randn(total_q, NH, D, device=DEV, dtype=torch.bfloat16)
    bias = torch.randn(total_q, NH, REL_EXTENT, device=DEV, dtype=torch.bfloat16) * 0.1
    cu_seqlens_q = torch.arange(0, total_q + 1, Q_LEN, dtype=torch.int32, device=DEV)
    common = dict(
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=Q_LEN,
        softmax_scale=D**-0.5,
        causal=True,
        num_splits=0,
        ver=4,
    )
    return q, k_cache, v_cache, bias, common


def run_step(q, k_cache, v_cache, bias, common, prep_cache):
    outs = []
    for _ in range(N_LAYERS):
        kwargs = {} if prep_cache is None else {"rel_bias_prep_cache": prep_cache}
        outs.append(flash_attn_with_kvcache(q, k_cache, v_cache, rel_bias=bias, **kwargs, **common))
    return outs


def count_prep_kernels(fn):
    fn()  # warm up (JIT compile) outside the profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        outs = fn()
        torch.cuda.synchronize()
    n = sum(
        e.count
        for e in prof.key_averages()
        if "CuSeqlensToBlocks" in e.key
    )
    return n, outs


def main():
    q, k_cache, v_cache, bias, common = make_case()

    n_uncached, outs_uncached = count_prep_kernels(
        lambda: run_step(q, k_cache, v_cache, bias, common, prep_cache=None)
    )
    n_cached, outs_cached = count_prep_kernels(
        lambda: run_step(q, k_cache, v_cache, bias, common, prep_cache={})
    )

    for a, b in zip(outs_uncached, outs_cached):
        assert torch.equal(a, b), "cached vs uncached outputs differ"
    assert n_uncached == N_LAYERS, f"expected {N_LAYERS} prep launches uncached, saw {n_uncached}"
    assert n_cached == 1, f"expected 1 prep launch with cache, saw {n_cached}"
    print(f"PASS: prep launches {n_uncached} -> {n_cached} across {N_LAYERS} layers, outputs bitwise equal")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    main()
