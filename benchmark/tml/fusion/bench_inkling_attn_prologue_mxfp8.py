"""Benchmark the Inkling fused attention prologue with optional MXFP8 Q/K/V quant.

This focuses on the target-verify prologue kernel in
``jit_kernel/inkling_attn_prologue.py``:

  bf16_store
      fused {K/V sconv + save_windows + Q/K norm + bf16 KV store}

  mxfp8_split
      fused prologue without KV store, then the current staged MXFP8 work:
      ``to_mxfp8(Q)`` + ``quant_store_kv_mxfp8(K,V)``

  mxfp8_fused
      one prologue launch that emits fp8 Q + SFQ and writes fp8 K/V cache +
      interleaved SFK/SFV directly

The ``marker`` harness gives one row per case and provider. Cases default to
Inkling-ish target-verify geometry (TP8 per-rank heads, head_dim=128, W=4).

Env knobs:
  CASES="B,Q,HQ,HKV;..." default "1,4,8,1;4,4,8,1;8,4,8,1;16,4,8,1;32,4,8,1"
  W=4                    short-conv kernel width
  ACT=silu|none          activation in K/V sconv (default none)
  RESIDUAL=0|1           sconv residual add (default 1)
  STORE=0|1              include bf16 KV store in bf16_store (default 1)

Run:
  PYTHONPATH=python python benchmark/tml/fusion/bench_inkling_attn_prologue_mxfp8.py

b200
[correctness] inkling_attn_prologue_mxfp8 b=1 q=4 hq=8 hkv=1: PASS (dequant_abs_max q=0.000e+00 k=0.000e+00 v=0.000e+00; raw_byte_delta exact)
[correctness] inkling_attn_prologue_mxfp8 b=4 q=4 hq=8 hkv=1: PASS (dequant_abs_max q=0.000e+00 k=0.000e+00 v=0.000e+00; raw_byte_delta exact)
[correctness] inkling_attn_prologue_mxfp8 b=8 q=4 hq=8 hkv=1: PASS (dequant_abs_max q=0.000e+00 k=0.000e+00 v=0.000e+00; raw_byte_delta exact)
[correctness] inkling_attn_prologue_mxfp8 b=16 q=4 hq=8 hkv=1: PASS (dequant_abs_max q=0.000e+00 k=0.000e+00 v=0.000e+00; raw_byte_delta exact)
[correctness] inkling_attn_prologue_mxfp8 b=32 q=4 hq=8 hkv=1: PASS (dequant_abs_max q=0.000e+00 k=0.000e+00 v=0.000e+00; raw_byte_delta exact)
=========================================================================================================================================================
          b         q        hq       hkv |   bf16_store(us)  mxfp8_split(us)  mxfp8_fused(us) |   bf16_store(GB/s)  mxfp8_split(GB/s)  mxfp8_fused(GB/s)
---------------------------------------------------------------------------------------------------------------------------------------------------------
0         1         4         8         1 |           4.1373           7.1062           4.7104 |               1.84               0.54               0.81
1         4         4         8         1 |           4.1574           7.4342           4.7104 |               7.34               2.05               3.24
2         8         4         8         1 |           4.1578           7.4950           4.7526 |              14.68               4.07               6.42
3        16         4         8         1 |           4.1798           7.5776           4.7718 |              29.20               8.05              12.79
4        32         4         8         1 |           4.2598           7.7219           4.8144 |              57.31              15.81              25.36
=========================================================================================================================================================
"""

from __future__ import annotations

import os
from functools import cache

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.inkling_attn_prologue import inkling_attn_prologue_verify
from sglang.srt.utils.common import suppress_noisy_warnings
from sglang.srt.kernels.mxfp8_quant import (
    MXFP8Tensor,
    from_mxfp8,
    quant_store_kv_mxfp8,
    to_mxfp8,
)

suppress_noisy_warnings()

DEV = "cuda"
DTYPE = torch.bfloat16
D = 128
PAGE = 128
SF_DIM = D // 32
DEFAULT_CASES = "1,4,8,1;4,4,8,1;8,4,8,1;16,4,8,1;32,4,8,1"
W = int(os.environ.get("W", "4"))
ACT_ENV = os.environ.get("ACT", "none").lower()
ACT = None if ACT_ENV in ("", "0", "none") else ACT_ENV
USE_RESIDUAL = os.environ.get("RESIDUAL", "1") == "1"
DO_BF16_STORE = os.environ.get("STORE", "1") == "1"
REPORT_TOL = float(os.environ.get("ATOL", "0.08"))
IMPLS = ["bf16_store", "mxfp8_split", "mxfp8_fused"]
_REFS = {}


def _parse_cases():
    out = []
    for case in os.environ.get("CASES", DEFAULT_CASES).split(";"):
        if not case:
            continue
        b, q, hq, hkv = (int(x) for x in case.split(","))
        out.append((b, q, hq, hkv))
    return out


@cache
def _make_case(b: int, q: int, hq: int, hkv: int):
    torch.cuda.manual_seed(1234 + b * 17 + q * 31 + hq * 43 + hkv * 59)
    t = b * q
    dq = hq * D
    dkv = hkv * D
    width = dq + 2 * dkv
    q_off, k_off, v_off = 0, dq, dq + dkv

    qkvr = torch.randn(t, width, device=DEV, dtype=DTYPE)
    k_cache = torch.randn(b, W - 1, dkv, device=DEV, dtype=DTYPE)
    v_cache = torch.randn(b, W - 1, dkv, device=DEV, dtype=DTYPE)
    cache_indices = torch.arange(b, device=DEV, dtype=torch.int32)
    cache_mask = torch.ones(b, device=DEV, dtype=torch.bool)
    k_weight = torch.randn(dkv, W, device=DEV, dtype=DTYPE) * 0.1
    v_weight = torch.randn(dkv, W, device=DEV, dtype=DTYPE) * 0.1
    k_inter = torch.empty(b, q, W - 1, dkv, device=DEV, dtype=DTYPE)
    v_inter = torch.empty_like(k_inter)
    q_gamma = torch.randn(D, device=DEV, dtype=DTYPE) * 0.1 + 1.0
    k_gamma = torch.randn(D, device=DEV, dtype=DTYPE) * 0.1 + 1.0
    loc = torch.arange(t, device=DEV, dtype=torch.int64)

    # Keep one page per benchmark case. loc is [0, T), so allocate page-rounded
    # buffers exactly like the production MXFP8 pool layout for page_size=128.
    slots = ((t + PAGE - 1) // PAGE) * PAGE
    k_buf_bf16 = torch.empty(slots, hkv, D, device=DEV, dtype=DTYPE)
    v_buf_bf16 = torch.empty_like(k_buf_bf16)
    k_buf_fp8 = torch.empty(slots, hkv, D, device=DEV, dtype=torch.float8_e4m3fn)
    v_buf_fp8 = torch.empty_like(k_buf_fp8)
    num_pages = slots // PAGE
    sf_shape = (num_pages, hkv, 32, PAGE // 32, SF_DIM)
    sfk = torch.empty(sf_shape, device=DEV, dtype=torch.float8_e8m0fnu)
    sfv = torch.empty_like(sfk)

    common = dict(
        qkvr=qkvr,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_indices=cache_indices,
        cache_mask=cache_mask,
        k_weight=k_weight,
        v_weight=v_weight,
        k_inter=k_inter,
        v_inter=v_inter,
        q_gamma=q_gamma,
        k_gamma=k_gamma,
        eps=1e-6,
        loc=loc,
        q_off=q_off,
        k_off=k_off,
        v_off=v_off,
        dq=dq,
        dkv=dkv,
        draft_token_num=q,
        activation=ACT,
        use_residual=USE_RESIDUAL,
    )
    return common, k_buf_bf16, v_buf_bf16, k_buf_fp8, v_buf_fp8, sfk, sfv


def _prologue(common, k_buf, v_buf, **kwargs):
    return inkling_attn_prologue_verify(
        common["qkvr"],
        common["k_cache"],
        common["v_cache"],
        common["cache_indices"],
        common["cache_mask"],
        common["k_weight"],
        common["v_weight"],
        common["k_inter"],
        common["v_inter"],
        common["q_gamma"],
        common["k_gamma"],
        common["eps"],
        common["loc"],
        k_buf,
        v_buf,
        common["q_off"],
        common["k_off"],
        common["v_off"],
        common["dq"],
        common["dkv"],
        common["draft_token_num"],
        activation=common["activation"],
        use_residual=common["use_residual"],
        **kwargs,
    )


def _make_fns(b: int, q: int, hq: int, hkv: int):
    common, k_bf, v_bf, k8, v8, sfk, sfv = _make_case(b, q, hq, hkv)

    def bf16_store():
        q_out, k_out, v_out, _ = _prologue(
            common, k_bf, v_bf, do_store=DO_BF16_STORE
        )
        return q_out, k_out, v_out

    def mxfp8_split():
        q_out, k_out, v_out, _ = _prologue(common, k_bf, v_bf, do_store=False)
        q_mxfp = to_mxfp8(q_out.view(b * q, hq, D))
        quant_store_kv_mxfp8(
            k_out.view(b * q, hkv, D),
            v_out.view(b * q, hkv, D),
            common["loc"],
            k8,
            v8,
            sfk,
            sfv,
            page_size=PAGE,
        )
        return q_mxfp.data, q_mxfp.scale, k8, v8, sfk, sfv

    def mxfp8_fused():
        q8, _k_out, _v_out, sfq = _prologue(
            common,
            k8,
            v8,
            do_store=True,
            mxfp8_quant=True,
            sfk=sfk,
            sfv=sfv,
            page_size=PAGE,
        )
        return q8, sfq, k8, v8, sfk, sfv

    return {
        "bf16_store": (bf16_store, ()),
        "mxfp8_split": (mxfp8_split, ()),
        "mxfp8_fused": (mxfp8_fused, ()),
    }


def _gather_interleaved_sf(sf: torch.Tensor, loc: torch.Tensor, nheads: int):
    page = loc // PAGE
    page_off = loc % PAGE
    heads = torch.arange(nheads, device=loc.device)
    return sf[
        page[:, None],
        heads[None, :],
        (page_off % 32)[:, None],
        (page_off // 32)[:, None],
        :,
    ]


def _dequant_outputs(out, b: int, q: int, hq: int, hkv: int):
    q8, sfq, k8, v8, sfk, sfv = out
    t = b * q
    loc = torch.arange(t, device=q8.device, dtype=torch.int64)
    q_bf = from_mxfp8(
        MXFP8Tensor(q8.view(t, hq, D), sfq.view(t, hq, SF_DIM)),
        out_dtype=torch.float32,
    )
    k_bf = from_mxfp8(
        MXFP8Tensor(k8[loc].view(t, hkv, D), _gather_interleaved_sf(sfk, loc, hkv)),
        out_dtype=torch.float32,
    )
    v_bf = from_mxfp8(
        MXFP8Tensor(v8[loc].view(t, hkv, D), _gather_interleaved_sf(sfv, loc, hkv)),
        out_dtype=torch.float32,
    )
    return q_bf, k_bf, v_bf


def _report_mxfp8(tag: str, fused, split, b: int, q: int, hq: int, hkv: int):
    labels = ("q8", "sfq", "k8_cache", "v8_cache", "sfk", "sfv")
    raw_max = {}
    for name, got, ref in zip(labels, fused, split):
        raw_diff = (got.view(torch.uint8).reshape(-1).to(torch.int16) -
                    ref.view(torch.uint8).reshape(-1).to(torch.int16)).abs()
        raw_max[name] = int(raw_diff.max().item())

    got_vals = _dequant_outputs(fused, b, q, hq, hkv)
    ref_vals = _dequant_outputs(split, b, q, hq, hkv)
    val_max = {
        name: (got.float() - ref.float()).abs().max().item()
        for name, got, ref in zip(("q", "k", "v"), got_vals, ref_vals)
    }
    worst = max(val_max.values())
    status = "PASS" if worst <= REPORT_TOL else "FAIL"
    raw_s = " ".join(f"{k}={v}" for k, v in raw_max.items() if v)
    if not raw_s:
        raw_s = "exact"
    val_s = " ".join(f"{k}={v:.3e}" for k, v in val_max.items())
    print(
        f"[correctness] {tag}: {status} "
        f"(dequant_abs_max {val_s}; raw_byte_delta {raw_s})"
    )


@marker.parametrize("b,q,hq,hkv", _parse_cases())
@marker.benchmark("impl", IMPLS)
def bench_inkling_attn_prologue(b: int, q: int, hq: int, hkv: int, impl: str):
    fns = _make_fns(b, q, hq, hkv)
    fn, args = fns[impl]
    out = fn(*args)

    case_key = (b, q, hq, hkv)
    if impl == "mxfp8_split":
        _REFS[case_key] = tuple(x.detach().clone() for x in out)
    elif impl == "mxfp8_fused" and case_key in _REFS:
        tag = f"inkling_attn_prologue_mxfp8 b={b} q={q} hq={hq} hkv={hkv}"
        _report_mxfp8(tag, out, _REFS[case_key], b, q, hq, hkv)

    # The prologue mutates k_inter/v_inter and optionally KV buffers. That is
    # intentional: the serving path does the same thing every layer. Passing the
    # representative output lets marker account for memory effects without an
    # extra synchronization.
    memory_output = out[0] if isinstance(out, tuple) else out
    return marker.do_bench(fn, input_args=args, memory_output=memory_output)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    bench_inkling_attn_prologue.run()
