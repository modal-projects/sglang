"""Diagnostic: replicate the MHATokenToKVPoolMXFP8 write/read ops exactly and run
the kernel through a SCATTERED page_table (the one server integration aspect the
bench never exercised: bench uses page_table=arange; the real allocator scatters
pages, and scales are written per-token via store_sf_interleaved at absolute loc).

Compares mxfp8 (pool-style) vs bf16 over the same dequantized values.
"""

import torch

from sglang.jit_kernel.flash_attn.cute import flash_attn_varlen_func as fn
from sglang.srt.kernels.mxfp8_interleave_sf import store_sf_interleaved
from sglang.srt.kernels.mxfp8_quant import from_mxfp8, to_mxfp8

DEV = "cuda"
HQ, HKV, D = 64, 8, 128
SF_DIM = D // 32
PAGE = 128


def run(b, kv_len, q_len, scatter):
    pps = (kv_len + PAGE - 1) // PAGE
    n_pages = b * pps
    m = n_pages * PAGE
    g = torch.Generator(device=DEV).manual_seed(0)

    q = torch.randn(b * q_len, HQ, D, device=DEV, dtype=torch.bfloat16, generator=g)
    k = torch.randn(b * kv_len, HKV, D, device=DEV, dtype=torch.bfloat16, generator=g)
    v = torch.randn(b * kv_len, HKV, D, device=DEV, dtype=torch.bfloat16, generator=g)

    # page_table: contiguous or scattered (shuffled distinct pages, like the allocator)
    if scatter:
        perm = torch.randperm(n_pages, device=DEV, generator=g).to(torch.int32)
    else:
        perm = torch.arange(n_pages, device=DEV, dtype=torch.int32)
    page_table = perm.view(b, pps)

    # absolute cache slot for each (seq, token): loc = page*PAGE + offset
    loc = torch.empty(b * kv_len, device=DEV, dtype=torch.int64)
    for i in range(b):
        for t in range(kv_len):
            pg = page_table[i, t // PAGE].item()
            loc[i * kv_len + t] = pg * PAGE + t % PAGE

    qq, kq, vq = to_mxfp8(q), to_mxfp8(k), to_mxfp8(v)
    q_ref, k_ref, v_ref = from_mxfp8(qq), from_mxfp8(kq), from_mxfp8(vq)

    # --- replicate pool buffers + writes ---
    k_buf = torch.zeros(m, HKV, D, device=DEV, dtype=torch.float8_e4m3fn)
    v_buf = torch.zeros(m, HKV, D, device=DEV, dtype=torch.float8_e4m3fn)
    kref_buf = torch.zeros(m, HKV, D, device=DEV, dtype=torch.bfloat16)
    vref_buf = torch.zeros(m, HKV, D, device=DEV, dtype=torch.bfloat16)
    ksf = torch.zeros(n_pages, HKV, 32, SF_DIM, SF_DIM, device=DEV, dtype=torch.float8_e8m0fnu)
    vsf = torch.zeros(n_pages, HKV, 32, SF_DIM, SF_DIM, device=DEV, dtype=torch.float8_e8m0fnu)
    k_buf[loc] = kq.data
    v_buf[loc] = vq.data
    kref_buf[loc] = k_ref
    vref_buf[loc] = v_ref
    store_sf_interleaved(kq.scale.view(-1, HKV, SF_DIM), ksf, loc, page_size=PAGE)
    store_sf_interleaved(vq.scale.view(-1, HKV, SF_DIM), vsf, loc, page_size=PAGE)

    cu_q = torch.arange(0, (b + 1) * q_len, q_len, device=DEV, dtype=torch.int32)
    seqused_k = torch.full((b,), kv_len, device=DEV, dtype=torch.int32)
    common = dict(cu_seqlens_q=cu_q, max_seqlen_q=q_len, softmax_scale=1.0 / D,
                  causal=True, seqused_k=seqused_k, page_table=page_table)

    def o(t):
        return (t[0] if isinstance(t, tuple) else t).float()

    ref = o(fn(q_ref, kref_buf.view(n_pages, PAGE, HKV, D), vref_buf.view(n_pages, PAGE, HKV, D), **common))
    out = o(fn(qq.data, k_buf.view(n_pages, PAGE, HKV, D), v_buf.view(n_pages, PAGE, HKV, D),
              sfq=qq.scale.view(torch.float8_e8m0fnu), sfk=ksf, sfv=vsf, **common))
    d = (out - ref).abs().max().item()
    print(f"b={b} kv={kv_len} q={q_len} scatter={scatter}: max_abs={d:.3e} {'OK' if d < 0.5 else 'FAIL'}")


if __name__ == "__main__":
    for scatter in (False, True):
        run(2, 1211, 256, scatter)
        run(4, 4096, 1, scatter)
