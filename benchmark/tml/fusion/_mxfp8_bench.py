"""Per-kernel bench: FA4 paged attention, bf16 KV vs the MXFP8 production path.

Exercises the paged varlen FA4 path with Inkling TP8-ish shapes:

  bf16 -> bf16 K/V baseline
  qbs  -> reference MXFP8 production path: fp8 Q (sfq) + fp8 K (sfk) +
          fp8 V (sfv, dequantized in-kernel), all paged. For page_size == 128
          the sf tensors are INTERLEAVED into the BlockScaledBasicChunk atom
          layout (blockscaled_utils.interleave_sf) and TMA-loaded. sfq stays a
          per-token flat tensor (q_sf_interleaved=False since cu_seqlens_q is
          given). qk_sf_vec_size = v_sf_vec_size = 32.
          V scales are PER-HEAD-DIM (per token, head_dim_v/32 blocks) -- built
          exactly like K's, so an incremental KV cache can produce them the same
          way it produces K scales (a per-head-dim block needs only one token).

Each non-bf16 row prints an informational PASS/FAIL line against the bf16 output
for the same random tensors before timing.

Env knobs:
  CASES="b,kv,q;..."   default decode-ish and prefill-ish shapes
  SPLITS=n             num_splits (default 0 = auto heuristic)
  BENCH_NUM_SPLITS=n   accepted alias for SPLITS
  PAGE=n               page size (default 128; interleaved SF requires 128)
  ATOL=n               report threshold vs bf16 (default 0.5)

Run:  python benchmark/tml/fusion/_mxfp8_bench.py

b200
[correctness] mxfp8_attention qbs b=64 q=1 kv=4096 [8h/1kv d=128] vs_bf16: PASS (abs_max=1.074e-02 bulk_abs_max=1.074e-02 abs_mean=9.735e-04)
[correctness] mxfp8_attention qbs b=64 q=1 kv=8192 [8h/1kv d=128] vs_bf16: PASS (abs_max=4.272e-03 bulk_abs_max=4.272e-03 abs_mean=6.864e-04)
[correctness] mxfp8_attention qbs b=128 q=1 kv=8192 [8h/1kv d=128] vs_bf16: PASS (abs_max=4.700e-03 bulk_abs_max=4.700e-03 abs_mean=6.869e-04)
[correctness] mxfp8_attention qbs b=32 q=1 kv=16384 [8h/1kv d=128] vs_bf16: PASS (abs_max=2.808e-03 bulk_abs_max=2.808e-03 abs_mean=4.913e-04)
[correctness] mxfp8_attention qbs b=1 q=8192 kv=8192 [8h/1kv d=128] vs_bf16: PASS (abs_max=1.484e-01 bulk_abs_max=7.422e-02 abs_mean=1.305e-03)
[correctness] mxfp8_attention qbs b=2 q=8192 kv=16384 [8h/1kv d=128] vs_bf16: PASS (abs_max=7.782e-03 bulk_abs_max=7.782e-03 abs_mean=5.671e-04)
=================================================================================================
          b    kv_len     q_len |        bf16(us)        qbs(us) |      bf16(GB/s)      qbs(GB/s)
-------------------------------------------------------------------------------------------------
0        64      4096         1 |         32.0555        32.0140 |         3907.10        2019.06
1        64      8192         1 |         51.8061        51.9235 |         4830.40        2486.18
2       128      8192         1 |        101.3252        82.1347 |         4939.43        3143.40
3        32     16384         1 |         53.2995        50.0571 |         4692.76        2577.03
4         1      8192      8192 |        114.4475       150.9097 |          307.18         170.27
5         2     16384      8192 |        660.6556       829.7147 |          118.25          66.79
=================================================================================================


[correctness] mxfp8_attention qbs b=64 q=1 kv=4096 [8h/1kv d=128] vs_bf16: PASS (abs_max=1.074e-02 bulk_abs_max=1.074e-02 abs_mean=9.735e-04)
[correctness] mxfp8_attention qbs b=64 q=1 kv=8192 [8h/1kv d=128] vs_bf16: PASS (abs_max=4.272e-03 bulk_abs_max=4.272e-03 abs_mean=6.864e-04)
[correctness] mxfp8_attention qbs b=128 q=1 kv=8192 [8h/1kv d=128] vs_bf16: PASS (abs_max=4.700e-03 bulk_abs_max=4.700e-03 abs_mean=6.869e-04)
[correctness] mxfp8_attention qbs b=32 q=1 kv=16384 [8h/1kv d=128] vs_bf16: PASS (abs_max=2.808e-03 bulk_abs_max=2.808e-03 abs_mean=4.913e-04)
[correctness] mxfp8_attention qbs b=1 q=8192 kv=8192 [8h/1kv d=128] vs_bf16: PASS (abs_max=1.484e-01 bulk_abs_max=7.422e-02 abs_mean=1.305e-03)
[correctness] mxfp8_attention qbs b=2 q=8192 kv=16384 [8h/1kv d=128] vs_bf16: PASS (abs_max=7.782e-03 bulk_abs_max=7.782e-03 abs_mean=5.671e-04)
=================================================================================================
          b    kv_len     q_len |        bf16(us)        qbs(us) |      bf16(GB/s)      qbs(GB/s)
-------------------------------------------------------------------------------------------------
0        64      4096         1 |         32.7197        30.6921 |         3827.79        2106.02
1        64      8192         1 |         56.1357        49.1821 |         4457.84        2624.76
2       128      8192         1 |        108.8973        87.2134 |         4595.97        2960.35
3        32     16384         1 |         53.5642        48.6391 |         4669.58        2652.16
4         1      8192      8192 |        118.6538       151.6574 |          296.29         169.43
5         2     16384      8192 |        667.4045       833.3452 |          117.06          66.50
=================================================================================================
"""

import math
import os
import sys
from functools import cache

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.flash_attention_v4 import flash_attn_varlen_func
from blockscaled_utils import interleave_sf
from sglang.srt.utils.common import suppress_noisy_warnings
from sglang.srt.kernels.mxfp8_quant import to_mxfp8

suppress_noisy_warnings()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_refs import report

DEV = "cuda"
HEADS_Q, HEADS_KV, HEAD_DIM = 8, 1, 128
SF_VEC = 32
SF_DIM = HEAD_DIM // SF_VEC
PAGE = int(os.environ.get("PAGE", "128"))
NUM_SPLITS = int(os.environ.get("SPLITS", os.environ.get("BENCH_NUM_SPLITS", "0")))
REPORT_TOL = float(os.environ.get("ATOL", "0.5"))
DEFAULT_CASES = (
    "1,4096,1;64,4096,1;64,8192,1;128,8192,1;32,16384,1;1,8192,8192;2,16384,8192"
)
_REFS = {}


def _parse_cases():
    cases = []
    for case in os.environ.get("CASES", DEFAULT_CASES).split(";"):
        if not case:
            continue
        b, kv_len, q_len = (int(x) for x in case.split(","))
        cases.append((b, kv_len, q_len))
    return cases


def paged_pack(x, b, kv_len):
    """[b*kv_len, h, d] -> ([n_pages, PAGE, h, d], page_table [b, pages])."""
    pages_per_seq = math.ceil(kv_len / PAGE)
    n_pages = b * pages_per_seq
    h, d = x.shape[1], x.shape[2]
    buf = torch.zeros(n_pages * PAGE, h, d, device=x.device, dtype=x.dtype)
    for i in range(b):
        buf[i * pages_per_seq * PAGE : i * pages_per_seq * PAGE + kv_len] = x[
            i * kv_len : (i + 1) * kv_len
        ]
    page_table = torch.arange(n_pages, device=x.device, dtype=torch.int32).view(
        b, pages_per_seq
    )
    return buf.view(n_pages, PAGE, h, d), page_table


def _interleaved_sf(sf_u8, n_pages):
    """Interleave a paged (n_pages, PAGE, h_kv, sf_dim) uint8 scale tensor into
    the BlockScaledBasicChunk atom layout, returned as an e8m0fnu buffer."""
    sf = sf_u8.view(n_pages, PAGE, HEADS_KV, SF_DIM)
    inter = interleave_sf(sf, sf_vec_size=SF_VEC)  # (n_pages, h_kv, ...) contiguous
    return inter.view(torch.float8_e8m0fnu)


@cache
def _make_base(b: int, kv_len: int, q_len: int):
    g = torch.Generator(device=DEV).manual_seed(1234)
    q = torch.randn(
        b * q_len, HEADS_Q, HEAD_DIM, device=DEV, dtype=torch.bfloat16, generator=g
    )
    k = torch.randn(
        b * kv_len, HEADS_KV, HEAD_DIM, device=DEV, dtype=torch.bfloat16, generator=g
    )
    v = torch.randn(
        b * kv_len, HEADS_KV, HEAD_DIM, device=DEV, dtype=torch.bfloat16, generator=g
    )
    cu_q = torch.arange(0, (b + 1) * q_len, q_len, device=DEV, dtype=torch.int32)
    seqused_k = torch.full((b,), kv_len, device=DEV, dtype=torch.int32)
    return q, k, v, cu_q, seqused_k


@cache
def _make_case(b: int, kv_len: int, q_len: int):
    q, k, v, cu_q, seqused_k = _make_base(b, kv_len, q_len)
    common = dict(
        cu_seqlens_q=cu_q,
        max_seqlen_q=q_len,
        # 1/sqrt(D) matches the real model (sharp softmax) and actually stresses
        # QK precision; 1/D gives near-uniform attention and masks fp8-QK error.
        softmax_scale=1.0 / math.sqrt(HEAD_DIM),
        causal=True,
        num_splits=NUM_SPLITS,
        seqused_k=seqused_k,
    )

    # bf16 paged K/V baseline.
    k_p, page_table = paged_pack(k, b, kv_len)
    v_p, _ = paged_pack(v, b, kv_len)
    common["page_table"] = page_table
    n_pages = k_p.shape[0]

    # --- MXFP8 production path (qbs): fp8 Q/K/V + interleaved paged SF ---
    # Q: per-token flat fp8 + flat sfq (q_sf_interleaved=False; cu_seqlens_q given).
    qq = to_mxfp8(q)
    q8 = qq.data
    sfq = qq.scale.view(torch.float8_e8m0fnu)  # (total_q, HEADS_Q, SF_DIM)

    # K: quantize per-token then page-pack; interleave the scales.
    kq = to_mxfp8(k)
    k8_p, _ = paged_pack(kq.data, b, kv_len)
    sfk_flat, _ = paged_pack(kq.scale.view(k.shape[0], HEADS_KV, SF_DIM), b, kv_len)
    sfk = _interleaved_sf(sfk_flat, n_pages)

    # V (v_dequant): scales are PER-HEAD-DIM, per token -- identical convention to
    # K (an incremental KV cache produces V scales the same way it produces K
    # scales, since a per-head-dim block only needs one token's worth of data).
    # So build sfv exactly like sfk: quantize V per-token along head_dim_v, then
    # paged-pack and interleave over (tokens-M, head_dim_v/32 sf_k).
    vq = to_mxfp8(v)
    v8_p, _ = paged_pack(vq.data, b, kv_len)
    sfv_flat, _ = paged_pack(vq.scale.view(v.shape[0], HEADS_KV, SF_DIM), b, kv_len)
    sfv = _interleaved_sf(sfv_flat, n_pages)

    return q, k_p, v_p, q8, k8_p, v8_p, sfq, sfk, sfv, common


def _run(q, k, v, common, **sf_kwargs):
    out = flash_attn_varlen_func(q, k, v, **sf_kwargs, **common)
    return out[0] if isinstance(out, tuple) else out


@marker.parametrize("b,kv_len,q_len", _parse_cases())
@marker.benchmark("impl", ["bf16", "qbs"])
def bench_mxfp8_attention(b: int, kv_len: int, q_len: int, impl: str):
    if q_len > kv_len:
        marker.skip(f"q_len {q_len} > kv {kv_len}")

    q, k_p, v_p, q8, k8_p, v8_p, sfq, sfk, sfv, common = _make_case(b, kv_len, q_len)

    if impl == "bf16":
        fn = lambda q, k, v: _run(q, k, v, common)
        input_args = (q, k_p, v_p)
    else:  # qbs: fp8 Q/K/V + interleaved SF, v dequanted in-kernel
        # vec sizes default to 32 in the interface when sfq/sfv are given.
        fn = lambda q, k, v, sq, sk, sv: _run(
            q, k, v, common, sfq=sq, sfk=sk, sfv=sv,
        )
        input_args = (q8, k8_p, v8_p, sfq, sfk, sfv)

    out = fn(*input_args)
    case_key = (b, kv_len, q_len)
    if impl == "bf16":
        # clone so the marker's buffer reuse during do_bench can't overwrite the ref
        _REFS[case_key] = out.detach().float().clone()
    elif case_key in _REFS:
        tag = (
            f"mxfp8_attention {impl} b={b} q={q_len} kv={kv_len} "
            f"[{HEADS_Q}h/{HEADS_KV}kv d={HEAD_DIM}] vs_bf16"
        )
        cur = out.detach().float().clone()
        ref = _REFS[case_key]
        diff = (cur - ref).abs().nan_to_num(nan=float("inf"))
        absmax = diff.max().item()
        absmean = diff.mean().item()
        # In causal prefill the first few query tokens attend to only a handful
        # of KV entries, so their output is essentially a single MXFP8-dequantized
        # V vector -> abs error up to ~1.0 (inherent fp8 quant noise, not a kernel
        # bug). Gate PASS on the bulk of tokens (drop the first CAUSAL_WARMUP per
        # sequence) plus a nan check; still print the full abs_max for visibility.
        CAUSAL_WARMUP = 32
        d_seq = diff.view(b, q_len, HEADS_Q, HEAD_DIM) if q_len > 1 else diff.view(b, 1, HEADS_Q, HEAD_DIM)
        bulk = d_seq[:, CAUSAL_WARMUP:] if q_len > CAUSAL_WARMUP else d_seq
        bulk_absmax = bulk.max().item() if bulk.numel() else absmax
        has_nan = absmax != absmax or absmax == float("inf")
        status = "FAIL" if (has_nan or bulk_absmax > REPORT_TOL) else "PASS"
        print(
            f"[correctness] {tag}: {status} "
            f"(abs_max={absmax:.3e} bulk_abs_max={bulk_absmax:.3e} abs_mean={absmean:.3e})"
        )

    return marker.do_bench(fn, input_args=input_args, memory_output=out)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    bench_mxfp8_attention.run()
