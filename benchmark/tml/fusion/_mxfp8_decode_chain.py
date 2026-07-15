"""Per-layer decode-chain bench: bf16 KV vs the MXFP8 production path, bs1 focus.

Measures the WHOLE per-attention-layer decode-step chain, not just the FA4
kernel, at real Inkling TP8 per-rank geometry (66 attn layers: 55 SWA
w=512 h_kv=2, 11 full h_kv=1; 8 q heads, d=128, rel_extent 1024):

  bf16   -> fused store_cache KV write + FA4 bf16 (rel_bias)
  mxfp8  -> 3x to_mxfp8 (Q/K/V) + 2x index_put fp8 KV write
            + 2x store_sf_interleaved + FA4 blockscaled (sfq/sfk/sfv, rel_bias)

Attribution columns: *_fa4 = attention kernel only, *_ovh = everything else
(quant + cache/scale writes). chain ~= fa4 + ovh; the difference x66 layers
is the per-step decode overhead mxfp8 must claw back.

Each mxfp8 chain row prints an informational PASS/FAIL vs the bf16 chain
output for the same random tensors before timing.

Env knobs:
  CASES="lt,b,kv;..."  lt in {full,swa} (default "full,1,4608;swa,1,640;full,32,4608;swa,32,640")
  SPLITS=n             num_splits (default 0 = auto heuristic)
  REL=0                drop rel_bias (default 1: prod runs sheared bias)
  ATOL=n               report threshold vs bf16 chain (default 0.5)

Run:  python benchmark/tml/fusion/_mxfp8_decode_chain.py
"""

import math
import os
import sys
from functools import cache

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

from sglang.jit_kernel.flash_attn.cute import flash_attn_varlen_func as _fn
from sglang.srt.mem_cache.memory_pool import _set_kv_buffer_impl
from sglang.srt.kernels.mxfp8_interleave_sf import store_sf_interleaved
from sglang.srt.kernels.mxfp8_quant import quant_store_kv_mxfp8, to_mxfp8

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_refs import report
from blockscaled_utils import interleave_sf

DEV = "cuda"
NH, D = 8, 128  # TP8 per-rank q heads, head_dim
SF_DIM = D // 32
PAGE = 128
REL_EXTENT = 1024
WINDOW = 512  # Inkling sliding_window_size
NUM_SPLITS = int(os.environ.get("SPLITS", "0"))
USE_REL = os.environ.get("REL", "1") == "1"
REPORT_TOL = float(os.environ.get("ATOL", "0.5"))
# lt,b,kv: full = 11 layers (h_kv=1, no window), swa = 55 layers (h_kv=2, w=512).
# kv=4608 ~ 4K prompt mid-decode; swa kv=640 ~ window + page slack.
DEFAULT_CASES = "full,1,4608;swa,1,640;full,32,4608;swa,32,640"
IMPLS = [
    "bf16",
    "mxfp8",
    "mxfp8f",
    "bf16_fa4",
    "mxfp8_fa4",
    "bf16_ovh",
    "mxfp8_ovh",
    "mxfp8f_ovh",
]
_REFS = {}


def _parse_cases():
    out = []
    for case in os.environ.get("CASES", DEFAULT_CASES).split(";"):
        if case:
            lt, b, kv = case.split(",")
            out.append((lt, int(b), int(kv)))
    return out


@cache
def _build(lt: str, b: int, kv: int):
    h_kv = 1 if lt == "full" else 2
    g = torch.Generator(device=DEV).manual_seed(1234)

    def rnd(*shape):
        return torch.randn(*shape, device=DEV, dtype=torch.bfloat16, generator=g)

    pages_per_seq = math.ceil(kv / PAGE)
    n_pages = b * pages_per_seq
    # New-token projections (what the model hands the attention layer per step).
    q = rnd(b, NH, D)
    k_new = rnd(b, h_kv, D)
    v_new = rnd(b, h_kv, D)
    # KV history, written into both cache flavors. The chain overwrites slot
    # kv-1 of each row with (k_new, v_new) each iteration, like a decode step.
    hist_k = rnd(n_pages * PAGE, h_kv, D)
    hist_v = rnd(n_pages * PAGE, h_kv, D)
    kc_bf = hist_k.clone().view(n_pages, PAGE, h_kv, D)
    vc_bf = hist_v.clone().view(n_pages, PAGE, h_kv, D)

    kq, vq = to_mxfp8(hist_k), to_mxfp8(hist_v)
    kc8 = kq.data.clone().view(n_pages, PAGE, h_kv, D)
    vc8 = vq.data.clone().view(n_pages, PAGE, h_kv, D)
    sfk = (
        interleave_sf(kq.scale.view(n_pages, PAGE, h_kv, SF_DIM), sf_vec_size=32)
        .view(torch.float8_e8m0fnu)
        .clone()
    )
    sfv = (
        interleave_sf(vq.scale.view(n_pages, PAGE, h_kv, SF_DIM), sf_vec_size=32)
        .view(torch.float8_e8m0fnu)
        .clone()
    )

    page_table = torch.arange(n_pages, dtype=torch.int32, device=DEV).view(
        b, pages_per_seq
    )
    seqused = torch.full((b,), kv, dtype=torch.int32, device=DEV)
    cu_q = torch.arange(0, b + 1, dtype=torch.int32, device=DEV)
    # Row i decodes into its last valid slot.
    loc = (
        torch.arange(b, device=DEV, dtype=torch.int64) * pages_per_seq * PAGE + kv - 1
    )
    # Local (SWA) layers use rel_extent == local_extent (attn.py:209); the FA4
    # sheared-bias path requires window == rel_extent for windowed attention.
    bias = rnd(b, NH, WINDOW if lt == "swa" else REL_EXTENT) * 0.1

    kw = dict(
        page_table=page_table,
        seqused_k=seqused,
        cu_seqlens_q=cu_q,
        max_seqlen_q=1,
        softmax_scale=D**-0.5,
        causal=True,
        num_splits=NUM_SPLITS,
    )
    if lt == "swa":
        kw["window_size"] = (WINDOW - 1, 0)
    if USE_REL:
        kw["rel_bias"] = bias
    return q, k_new, v_new, kc_bf, vc_bf, kc8, vc8, sfk, sfv, loc, kw


def _out(res):
    return res[0] if isinstance(res, tuple) else res


def _make_fns(lt, b, kv):
    """Return {impl: (fn, input_args)} for one case. Cache buffers are captured
    (mutated in place like the real pool); per-token tensors are rotated args."""
    q, k_new, v_new, kc_bf, vc_bf, kc8, vc8, sfk, sfv, loc, kw = _build(lt, b, kv)
    h_kv = k_new.shape[1]

    def bf16_write(k, v):
        _set_kv_buffer_impl(
            k, v, kc_bf.view(-1, h_kv, D), vc_bf.view(-1, h_kv, D), loc,
            row_dim=h_kv * D, store_dtype=torch.bfloat16, device_module=torch.cuda,
            size_limit=kc_bf.numel() // (h_kv * D),
        )

    def mxfp8_write(k, v):
        """to_mxfp8 + pool writes, exactly the prod per-layer sequence
        (attn.py to_mxfp8 + MHATokenToKVPoolMXFP8.set_kv_buffer/_write_scales)."""
        kq, vq = to_mxfp8(k), to_mxfp8(v)
        kc8.view(-1, h_kv, D)[loc] = kq.data
        vc8.view(-1, h_kv, D)[loc] = vq.data
        store_sf_interleaved(kq.scale.view(torch.float8_e8m0fnu), sfk, loc)
        store_sf_interleaved(vq.scale.view(torch.float8_e8m0fnu), sfv, loc)
        return kq, vq

    def chain_bf16(q, k, v, bias):
        bf16_write(k, v)
        return _out(_fn(q, kc_bf, vc_bf, **kw))

    def chain_mxfp8(q, k, v, bias):
        qq = to_mxfp8(q)
        mxfp8_write(k, v)
        return _out(
            _fn(
                qq.data, kc8, vc8,
                sfq=qq.scale.view(torch.float8_e8m0fnu), sfk=sfk, sfv=sfv,
                **kw,
            )
        )

    def fused_write(q, k, v):
        # The prod SGLANG_OPT_INKLING_MXFP8_FUSED_QUANT_STORE pair: q quant in the
        # layer + one fused K/V quant-store in the pool.
        qq = to_mxfp8(q)
        quant_store_kv_mxfp8(
            k, v, loc, kc8.view(-1, h_kv, D), vc8.view(-1, h_kv, D), sfk, sfv
        )
        return qq

    def chain_mxfp8_fused(q, k, v, bias):
        qq = fused_write(q, k, v)
        return _out(
            _fn(
                qq.data, kc8, vc8,
                sfq=qq.scale.view(torch.float8_e8m0fnu), sfk=sfk, sfv=sfv,
                **kw,
            )
        )

    # fa4-only variants: pre-quantized static inputs, attention kernel alone.
    qq_s = to_mxfp8(q)
    q8_s, sfq_s = qq_s.data, qq_s.scale.view(torch.float8_e8m0fnu)

    def fa4_bf16(q, bias):
        return _out(_fn(q, kc_bf, vc_bf, **kw))

    def fa4_mxfp8(q8, bias):
        return _out(_fn(q8, kc8, vc8, sfq=sfq_s, sfk=sfk, sfv=sfv, **kw))

    def ovh_mxfp8(q, k, v):
        to_mxfp8(q)
        mxfp8_write(k, v)

    bias = kw.get("rel_bias", q)  # placeholder arg when REL=0
    return {
        "bf16": (chain_bf16, (q, k_new, v_new, bias)),
        "mxfp8": (chain_mxfp8, (q, k_new, v_new, bias)),
        "mxfp8f": (chain_mxfp8_fused, (q, k_new, v_new, bias)),
        "bf16_fa4": (fa4_bf16, (q, bias)),
        "mxfp8_fa4": (fa4_mxfp8, (q8_s, bias)),
        "bf16_ovh": (bf16_write, (k_new, v_new)),
        "mxfp8_ovh": (ovh_mxfp8, (q, k_new, v_new)),
        "mxfp8f_ovh": (fused_write, (q, k_new, v_new)),
    }


def _check_fused_bitwise(lt, b, kv):
    """Fused kernel must reproduce the unfused path bit-for-bit."""
    q, k_new, v_new, kc_bf, vc_bf, kc8, vc8, sfk, sfv, loc, kw = _build(lt, b, kv)
    h_kv = k_new.shape[1]
    ref_kc, ref_vc = kc8.clone(), vc8.clone()
    ref_sfk, ref_sfv = sfk.clone(), sfv.clone()
    kq, vq = to_mxfp8(k_new), to_mxfp8(v_new)
    qq_ref = to_mxfp8(q)
    ref_kc.view(-1, h_kv, D)[loc] = kq.data
    ref_vc.view(-1, h_kv, D)[loc] = vq.data
    store_sf_interleaved(kq.scale.view(torch.float8_e8m0fnu), ref_sfk, loc)
    store_sf_interleaved(vq.scale.view(torch.float8_e8m0fnu), ref_sfv, loc)
    got_kc, got_vc = kc8.clone(), vc8.clone()
    got_sfk, got_sfv = sfk.clone(), sfv.clone()
    qq = to_mxfp8(q)
    quant_store_kv_mxfp8(
        k_new, v_new, loc, got_kc.view(-1, h_kv, D), got_vc.view(-1, h_kv, D),
        got_sfk, got_sfv,
    )
    ok = all(
        torch.equal(a.view(torch.uint8), e.view(torch.uint8))
        for a, e in [
            (qq.data, qq_ref.data), (qq.scale, qq_ref.scale),
            (got_kc, ref_kc), (got_vc, ref_vc),
            (got_sfk, ref_sfk), (got_sfv, ref_sfv),
        ]
    )
    print(f"[bitwise] fused_quant_store {lt} b={b} kv={kv}: {'PASS' if ok else 'FAIL'}")


@marker.parametrize("lt,b,kv", _parse_cases())
@marker.benchmark("impl", IMPLS)
def bench_mxfp8_decode_chain(lt: str, b: int, kv: int, impl: str):
    fns = _make_fns(lt, b, kv)
    fn, args = fns[impl]

    out = fn(*args)
    case_key = (lt, b, kv)
    if impl == "mxfp8f" and ("bw", case_key) not in _REFS:
        _REFS[("bw", case_key)] = True
        _check_fused_bitwise(lt, b, kv)
    if impl == "bf16":
        _REFS[case_key] = out.detach().float().clone()
    elif impl == "mxfp8" and case_key in _REFS:
        tag = f"mxfp8_decode_chain {lt} b={b} kv={kv} [{NH}h/{args[1].shape[1]}kv d={D}]"
        report(tag, out.detach().float(), _REFS[case_key], tol=REPORT_TOL)

    return marker.do_bench(fn, input_args=args, memory_output=out)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    bench_mxfp8_decode_chain.run()
