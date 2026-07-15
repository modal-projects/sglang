"""Diagnostic: does the mxfp8 (blockscaled-QK + v_dequant) path compose with the
FA4 sheared rel_bias? The server always runs rel_bias; the mxfp8 bench never did.

Compares, over the SAME values:
  A) bf16 attention + rel_bias                          (reference)
  B) mxfp8 qbs (fp8 Q/K/V + sfq/sfk/sfv) + rel_bias      (kernel under test)
  C) mxfp8 qbs WITHOUT rel_bias                          (isolates the bias)

If B diverges from A but C matches its own no-bias bf16, the bias path is broken
for mxfp8.
"""

import math
import torch

from sglang.jit_kernel.flash_attn.cute import flash_attn_varlen_func as fn
from blockscaled_utils import interleave_sf
from sglang.srt.kernels.mxfp8_quant import from_mxfp8, to_mxfp8

DEV = "cuda"
HQ, HKV, D = 8, 1, 128
SF_DIM = D // 32
SF_VEC = 32
PAGE = 128
REL_EXTENT = 2048


def paged_pack(x, b, kv_len):
    pps = math.ceil(kv_len / PAGE)
    n_pages = b * pps
    h, d = x.shape[1], x.shape[2]
    buf = torch.zeros(n_pages * PAGE, h, d, device=x.device, dtype=x.dtype)
    for i in range(b):
        buf[i * pps * PAGE : i * pps * PAGE + kv_len] = x[i * kv_len : (i + 1) * kv_len]
    pt = torch.arange(n_pages, device=x.device, dtype=torch.int32).view(b, pps)
    return buf.view(n_pages, PAGE, h, d), pt


def isf(sf_paged):
    return interleave_sf(sf_paged, sf_vec_size=SF_VEC).view(torch.float8_e8m0fnu)


def run(b, kv_len, q_len):
    g = torch.Generator(device=DEV).manual_seed(0)
    q = torch.randn(b * q_len, HQ, D, device=DEV, dtype=torch.bfloat16, generator=g)
    k = torch.randn(b * kv_len, HKV, D, device=DEV, dtype=torch.bfloat16, generator=g)
    v = torch.randn(b * kv_len, HKV, D, device=DEV, dtype=torch.bfloat16, generator=g)
    bias = torch.randn(b * q_len, HQ, REL_EXTENT, device=DEV, dtype=torch.bfloat16, generator=g) * 0.1
    cu_q = torch.arange(0, (b + 1) * q_len, q_len, device=DEV, dtype=torch.int32)
    seqused_k = torch.full((b,), kv_len, device=DEV, dtype=torch.int32)
    common = dict(cu_seqlens_q=cu_q, max_seqlen_q=q_len, softmax_scale=1.0 / D,
                  causal=True, seqused_k=seqused_k)

    qq, kq, vq = to_mxfp8(q), to_mxfp8(k), to_mxfp8(v)
    q_ref, k_ref, v_ref = from_mxfp8(qq), from_mxfp8(kq), from_mxfp8(vq)
    kr_p, pt = paged_pack(k_ref, b, kv_len)
    vr_p, _ = paged_pack(v_ref, b, kv_len)
    k8_p, _ = paged_pack(kq.data, b, kv_len)
    v8_p, _ = paged_pack(vq.data, b, kv_len)
    sfk = isf(paged_pack(kq.scale.view(k.shape[0], HKV, SF_DIM), b, kv_len)[0])
    sfv = isf(paged_pack(vq.scale.view(v.shape[0], HKV, SF_DIM), b, kv_len)[0])
    sfq = qq.scale.view(torch.float8_e8m0fnu)
    qk = dict(sfq=sfq, sfk=sfk, sfv=sfv)

    def out(t):
        return t[0] if isinstance(t, tuple) else t

    # A: bf16 + bias   B: mxfp8 + bias   C: bf16 no-bias   D: mxfp8 no-bias
    A = out(fn(q_ref, kr_p, vr_p, page_table=pt, rel_bias=bias, **common)).float()
    B = out(fn(qq.data, k8_p, v8_p, page_table=pt, rel_bias=bias, **common, **qk)).float()
    C = out(fn(q_ref, kr_p, vr_p, page_table=pt, **common)).float()
    Dd = out(fn(qq.data, k8_p, v8_p, page_table=pt, **common, **qk)).float()

    def cmp(x, y):
        return (x - y).abs().max().item()

    print(f"b={b} kv={kv_len} q={q_len}:")
    print(f"  mxfp8+bias vs bf16+bias   : {cmp(B, A):.3e}  {'OK' if cmp(B,A)<0.5 else 'FAIL'}")
    print(f"  mxfp8 no-bias vs bf16 nobias: {cmp(Dd, C):.3e}  {'OK' if cmp(Dd,C)<0.5 else 'FAIL'}")
    print(f"  bias effect (A vs C)      : {cmp(A, C):.3e} (should be nonzero)")


if __name__ == "__main__":
    for c in [(2, 4096, 1), (2, 1211, 256)]:
        run(*c)
