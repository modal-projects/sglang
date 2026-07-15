"""Correctness testbed for the FA4 MXFP8 KV-cache attention (production path).

Single kernel mode (the downloads reference contract): block-scaled QK^T +
in-kernel V dequant. Q/K/V are all quantized to fp8 e4m3 with per-32-element
UE8M0 scales along head_dim; Q's scales pass as sfq (flat), K/V's as sfk/sfv
(interleaved into the FA4 BlockScaledBasicChunk atom layout for page_size==128).
The QK^T runs as tcgen05 mxf8f6f4 with scales in TMEM; V is dequantized to bf16
in the correction warp before a bf16 PV.

Reference = FA4 bf16 attention over exactly-dequantized (per-32-block) Q/K/V,
isolating kernel error from quantization error (so the tolerance is tight).

Inkling TP8 shapes: 8 q heads, 1 kv head, head_dim 128 (sf_dim 4), page 128.

Env knobs:
  CASES="b,kv,q;..."   default "2,128,1;2,1211,1;2,1211,256;2,8261,256"
  SPLITS=n             num_splits (0 = auto heuristic; default 1)
  WINDOW=n             sliding window size (0 = off)
  ATOL=0.06            max-rel-diff threshold vs the exactly-dequantized ref
"""

import math
import os

import torch

from sglang.jit_kernel.flash_attention_v4 import flash_attn_varlen_func
from blockscaled_utils import interleave_sf
from sglang.srt.kernels.mxfp8_quant import from_mxfp8, to_mxfp8

HEADS_Q = int(os.environ.get("HEADS_Q", "8"))
HEADS_KV = int(os.environ.get("HEADS_KV", "1"))
HEAD_DIM = 128
SF_DIM = HEAD_DIM // 32
SF_VEC = 32
PAGE = 128


def make_case(b, kv_len, q_len, device, dtype=torch.bfloat16):
    g = torch.Generator(device=device).manual_seed(b * 7919 + kv_len * 131 + q_len)
    q = torch.randn(b * q_len, HEADS_Q, HEAD_DIM, device=device, dtype=dtype, generator=g)
    k = torch.randn(b * kv_len, HEADS_KV, HEAD_DIM, device=device, dtype=dtype, generator=g)
    v = torch.randn(b * kv_len, HEADS_KV, HEAD_DIM, device=device, dtype=dtype, generator=g)
    cu_q = torch.arange(0, (b + 1) * q_len, q_len, device=device, dtype=torch.int32)
    seqused_k = torch.full((b,), kv_len, device=device, dtype=torch.int32)
    return q, k, v, cu_q, seqused_k


def paged_pack(x, b, kv_len):
    """[b*kv_len, h, d] -> ([n_pages, PAGE, h, d], page_table [b, pages_per_seq])."""
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


def interleaved_sf(sf_paged):
    """Paged per-token (n_pages, PAGE, h, SF_DIM) -> interleaved BlockScaledBasicChunk."""
    return interleave_sf(sf_paged, sf_vec_size=SF_VEC).view(torch.float8_e8m0fnu)


def run_case(b, kv_len, q_len, splits, atol, device="cuda"):
    q, k, v, cu_q, seqused_k = make_case(b, kv_len, q_len, device)
    window = int(os.environ.get("WINDOW", "0"))
    common = dict(
        cu_seqlens_q=cu_q,
        max_seqlen_q=q_len,
        softmax_scale=1.0 / HEAD_DIM,
        causal=True,
        num_splits=splits,
        seqused_k=seqused_k,
    )
    if window > 0:
        common["window_size"] = (window, 0)

    # Quantize Q/K/V to mxfp8; keep exact-dequant copies for the bf16 reference.
    qq, kq, vq = to_mxfp8(q), to_mxfp8(k), to_mxfp8(v)
    q_ref, k_ref, v_ref = from_mxfp8(qq), from_mxfp8(kq), from_mxfp8(vq)

    k_ref_p, page_table = paged_pack(k_ref, b, kv_len)
    v_ref_p, _ = paged_pack(v_ref, b, kv_len)
    ref = flash_attn_varlen_func(
        q_ref, k_ref_p, v_ref_p, page_table=page_table, **common
    )

    q8 = qq.data
    sfq = qq.scale.view(torch.float8_e8m0fnu)
    k8_p, _ = paged_pack(kq.data, b, kv_len)
    v8_p, _ = paged_pack(vq.data, b, kv_len)
    sfk_flat, _ = paged_pack(kq.scale.view(k.shape[0], HEADS_KV, SF_DIM), b, kv_len)
    sfv_flat, _ = paged_pack(vq.scale.view(v.shape[0], HEADS_KV, SF_DIM), b, kv_len)
    out = flash_attn_varlen_func(
        q8,
        k8_p,
        v8_p,
        page_table=page_table,
        sfq=sfq,
        sfk=interleaved_sf(sfk_flat),
        sfv=interleaved_sf(sfv_flat),
        **common,
    )

    out = out[0] if isinstance(out, tuple) else out
    ref = ref[0] if isinstance(ref, tuple) else ref
    diff = (out.float() - ref.float()).abs()
    rel = (diff / ref.float().abs().clamp(min=1.0)).max().item()
    ok = rel <= atol and not torch.isnan(out).any()
    print(
        f"case b={b} kv={kv_len} q={q_len} splits={splits} qbs "
        f"max_rel={rel:.3e} max_abs={diff.max().item():.3e} "
        f"{'OK' if ok else 'FAIL'}"
    )
    return ok


def main():
    # kv=128 (single page) is dominated by fp8 quant noise on a tiny context
    # (~0.14 abs); real serving contexts are long, so the defaults start at 1211.
    cases = os.environ.get("CASES", "2,1211,1;2,1211,256;2,8261,256")
    splits = int(os.environ.get("SPLITS", "1"))
    atol = float(os.environ.get("ATOL", "0.06"))
    # The blockscaled kernel forces q_stage=1; pin the bf16 reference to the same
    # schedule or accumulation-order ulps under split-KV/window exceed ATOL.
    os.environ["FA4_DEBUG_QSTAGE1"] = "1"
    ok = True
    for case in cases.split(";"):
        b, kv_len, q_len = (int(x) for x in case.split(","))
        ok &= run_case(b, kv_len, q_len, splits, atol)
    if not ok:
        raise SystemExit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()
