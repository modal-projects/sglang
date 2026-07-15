"""Precision audit: sheared-bias vs score-mod, each vs an fp64 reference.

The kvcache bench only checks shear-vs-score_mod at kv%128==0, re=128 and a loose
1e-1 tolerance. This test measures the ACTUAL error of each path against a float64
torch reference implementing the score_mod semantics exactly (bias for
0 <= rel_dist < rel_extent, else 0; causal mask), at prod-like shapes:
rel_extent=1024, unaligned kv lengths, softmax_scale=1/head_dim.

For each case it prints, per impl:
  max/mean abs error vs the fp64 ref, and the (q_row, head) of the max error plus
  that row's kv span relative to the rel_extent boundary -- so a boundary bug
  (bias misalignment at the last partial KV block or at the extent edge) shows up
  as errors clustered at rel_dist ~ 0 / ~ rel_extent instead of uniform bf16 noise.

Run on a GPU:  python benchmark/tml/fusion/test_shear_bias_precision.py

Env overrides: PREC_KV, PREC_QLENS, PREC_REL_EXTENT, PREC_BATCH, PREC_SCALE
(comma lists where plural). PREC_SCALE: "invd" (1/d, prod) or "rsqrt" (d**-0.5).
"""

import os

from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

import torch

from sglang.jit_kernel.flash_attention import flash_attn_with_kvcache as _kvcache_fn
from sglang.srt.models.inkling_common.attn import (
    get_inkling_relative_attention_score_mod,
)

try:
    from flash_attn.cute import flash_attn_varlen_func as _base_fn
except Exception:
    _base_fn = None

DEV = "cuda"
NH, NHK, D = 64, 8, 128
PAGE_SIZE = 128

REL_EXTENT = int(os.environ.get("PREC_REL_EXTENT", "1024"))
KV_SEQLENS = [int(x) for x in os.environ.get("PREC_KV", "1211,8261").split(",")]
QLENS = [int(x) for x in os.environ.get("PREC_QLENS", "1,7,256,2048").split(",")]
BATCH = int(os.environ.get("PREC_BATCH", "2"))
SCALE_KIND = os.environ.get("PREC_SCALE", "invd")
# 1 = local/SWA layer config: window_size=(rel_extent-1, 0) like the backend passes
# for is_local Inkling layers (sliding_window_size = local_extent - 1, rel_extent = local_extent).
WINDOWED = os.environ.get("PREC_WINDOWED", "0") == "1"


def make_ragged_case(q_lens, kv_lens):
    """Mixed per-seq q/kv lengths in one varlen call -- what serving actually does."""
    torch.manual_seed(0)
    b = len(q_lens)
    total_q = sum(q_lens)
    pages_per_seq = max((kv + PAGE_SIZE - 1) // PAGE_SIZE for kv in kv_lens)
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
    cache_seqlens = torch.tensor(kv_lens, dtype=torch.int32, device=DEV)
    q = torch.randn(total_q, NH, D, device=DEV, dtype=torch.bfloat16)
    bias = torch.randn(total_q, NH, REL_EXTENT, device=DEV, dtype=torch.bfloat16)
    cu = (
        torch.tensor([0] + list(q_lens), dtype=torch.int32, device=DEV)
        .cumsum(0)
        .to(torch.int32)
    )
    scale = 1.0 / D if SCALE_KIND == "invd" else D**-0.5
    common = dict(
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu,
        max_seqlen_q=max(q_lens),
        softmax_scale=scale,
        causal=True,
        num_splits=int(os.environ.get("PREC_NUM_SPLITS", "0")),
        ver=4,
    )
    if WINDOWED:
        common["window_size"] = (REL_EXTENT - 1, 0)
    return q, k_cache, v_cache, bias, page_table, common


def ref_fp64_ragged(q, k_cache, v_cache, bias, page_table, q_lens, kv_lens, scale):
    outs = []
    q_off = 0
    for bi, (q_len, s_kv) in enumerate(zip(q_lens, kv_lens)):
        pages = page_table[bi].long()
        k = k_cache[pages].reshape(-1, NHK, D)[:s_kv].double()
        v = v_cache[pages].reshape(-1, NHK, D)[:s_kv].double()
        qb = q[q_off : q_off + q_len].double()
        bb = bias[q_off : q_off + q_len].double()
        q_off += q_len
        k = k.repeat_interleave(NH // NHK, dim=1)
        v = v.repeat_interleave(NH // NHK, dim=1)
        s = torch.einsum("qhd,khd->hqk", qb, k) * scale
        qi = torch.arange(q_len, device=DEV).view(1, -1, 1)
        kj = torch.arange(s_kv, device=DEV).view(1, 1, -1)
        rel = (qi + (s_kv - q_len)) - kj
        in_ext = (rel >= 0) & (rel < REL_EXTENT)
        rel_c = rel.clamp(0, REL_EXTENT - 1)
        bsel = bb.permute(1, 0, 2).gather(2, rel_c.expand(NH, q_len, s_kv))
        s = s + torch.where(
            in_ext.expand_as(bsel),
            bsel,
            torch.zeros((), dtype=torch.float64, device=DEV),
        )
        s = s.masked_fill(rel < 0, float("-inf"))
        if WINDOWED:
            s = s.masked_fill(rel >= REL_EXTENT, float("-inf"))
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", p, v))
    return torch.cat(outs, dim=0)


def run_ragged(sm, q_lens, kv_lens, scale):
    q, kc, vc, bias, pt, common = make_ragged_case(q_lens, kv_lens)
    ref = ref_fp64_ragged(q, kc, vc, bias, pt, q_lens, kv_lens, scale)
    print(f"ragged case q_lens={q_lens} kv_lens={kv_lens}")

    def stats_r(tag, out):
        out = out[0] if isinstance(out, tuple) else out
        err = (out.double() - ref).abs()
        per_seq = []
        q_off = 0
        for q_len in q_lens:
            per_seq.append(err[q_off : q_off + q_len].max().item())
            q_off += q_len
        print(
            f"  {tag:24s} max={err.max().item():.3e} mean={err.mean().item():.3e} "
            f"per_seq_max={['%.1e' % e for e in per_seq]}"
        )
        return err.max().item()

    stats_r(
        "score_mod(vendored)",
        _kvcache_fn(q, kc, vc, score_mod=sm, aux_tensors=[bias], **common),
    )
    e = stats_r("shear(vendored)", _kvcache_fn(q, kc, vc, rel_bias=bias, **common))
    return e


def make_case(b: int, s_kv: int, q_len: int):
    torch.manual_seed(0)
    total_q = b * q_len
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
    q = torch.randn(total_q, NH, D, device=DEV, dtype=torch.bfloat16)
    bias = torch.randn(total_q, NH, REL_EXTENT, device=DEV, dtype=torch.bfloat16)
    cu_seqlens_q = torch.arange(0, total_q + 1, q_len, dtype=torch.int32, device=DEV)
    scale = 1.0 / D if SCALE_KIND == "invd" else D**-0.5
    common = dict(
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=q_len,
        softmax_scale=scale,
        causal=True,
        num_splits=int(os.environ.get("PREC_NUM_SPLITS", "0")),
        ver=4,
    )
    if WINDOWED:
        common["window_size"] = (REL_EXTENT - 1, 0)
    return q, k_cache, v_cache, bias, page_table, common


def ref_fp64(q, k_cache, v_cache, bias, page_table, b, s_kv, q_len, scale):
    """score_mod semantics in float64: bias iff 0 <= rel_dist < REL_EXTENT, causal.
    Chunked over kv-head groups to fit next to a running server."""
    grp = NH // NHK
    qi = torch.arange(q_len, device=DEV).view(-1, 1)
    kj = torch.arange(s_kv, device=DEV).view(1, -1)
    rel = (qi + (s_kv - q_len)) - kj  # (q, kv)
    in_ext = (rel >= 0) & (rel < REL_EXTENT)
    rel_c = rel.clamp(0, REL_EXTENT - 1)
    outs = []
    for bi in range(b):
        pages = page_table[bi].long()
        kb = k_cache[pages].reshape(-1, NHK, D)[:s_kv]  # (s_kv, hk, d) bf16
        vb = v_cache[pages].reshape(-1, NHK, D)[:s_kv]
        out_b = torch.empty(q_len, NH, D, dtype=torch.float64, device=DEV)
        for hk in range(NHK):
            k = kb[:, hk].double()  # (s_kv, d)
            v = vb[:, hk].double()
            for hq in range(hk * grp, (hk + 1) * grp):
                qh = q[bi * q_len : (bi + 1) * q_len, hq].double()  # (q, d)
                bh = bias[bi * q_len : (bi + 1) * q_len, hq].double()  # (q, re)
                s = (qh @ k.T) * scale  # (q, kv)
                bsel = bh.gather(1, rel_c)
                s = s + torch.where(
                    in_ext, bsel, torch.zeros((), dtype=torch.float64, device=DEV)
                )
                s = s.masked_fill(rel < 0, float("-inf"))
                if WINDOWED:
                    s = s.masked_fill(rel >= REL_EXTENT, float("-inf"))
                out_b[:, hq] = torch.softmax(s, dim=-1) @ v
        outs.append(out_b)
    return torch.cat(outs, dim=0)  # (total_q, h, d)


def stats(tag, out, ref, b, s_kv, q_len):
    out = out[0] if isinstance(out, tuple) else out
    err = (out.double() - ref).abs()
    flat = err.max(dim=-1).values  # (total_q, h)
    idx = flat.argmax().item()
    qrow, head = idx // NH, idx % NH
    q_local = qrow % q_len
    kv_pos = q_local + (s_kv - q_len)  # diagonal kv index of that row
    print(
        f"  {tag:24s} max={err.max().item():.3e} mean={err.mean().item():.3e} "
        f"rel_l2={ (err.pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item():.3e} "
        f"argmax q_row={qrow}(local {q_local}) h={head} diag_kv={kv_pos} "
        f"row_kv_span=[{max(0, kv_pos - REL_EXTENT + 1)}..{kv_pos}]"
    )
    return err.max().item()


def main():
    sm = get_inkling_relative_attention_score_mod(REL_EXTENT)
    scale = 1.0 / D if SCALE_KIND == "invd" else D**-0.5
    print(
        f"rel_extent={REL_EXTENT} scale={SCALE_KIND}({scale:.6g}) b={BATCH} "
        f"heads={NH}/{NHK} d={D} windowed={WINDOWED}"
    )
    worst = 0.0
    if os.environ.get("PREC_RAGGED", "0") == "1":
        # decode-like ragged batch, extend-like ragged batch, and a mix crossing tile boundaries
        for q_lens, kv_lens in [
            ([1] * 6, [317, 1211, 129, 4097, 640, 2048]),
            ([1, 7, 33, 256, 130, 100], [1211, 1211, 517, 8261, 130, 1500]),
            ([255, 1, 129, 3], [255, 4097, 129, 1024]),
        ]:
            worst = max(worst, run_ragged(sm, q_lens, kv_lens, scale))
        print(f"done. worst shear-vs-ref max err = {worst:.3e}")
        return
    for s_kv in KV_SEQLENS:
        for q_len in QLENS:
            if q_len > s_kv:
                continue
            q, kc, vc, bias, pt, common = make_case(BATCH, s_kv, q_len)
            ref = ref_fp64(q, kc, vc, bias, pt, BATCH, s_kv, q_len, scale)
            print(f"case b={BATCH} kv={s_kv} q={q_len}")
            out_sm = _kvcache_fn(q, kc, vc, score_mod=sm, aux_tensors=[bias], **common)
            e1 = stats("score_mod(vendored)", out_sm, ref, BATCH, s_kv, q_len)
            out_sh = _kvcache_fn(q, kc, vc, rel_bias=bias, **common)
            e2 = stats("shear(vendored)", out_sh, ref, BATCH, s_kv, q_len)
            if _base_fn is not None:
                out_base = _base_fn(
                    q,
                    kc,
                    vc,
                    cu_seqlens_q=common["cu_seqlens_q"],
                    max_seqlen_q=q_len,
                    seqused_k=common["cache_seqlens"],
                    page_table=common["page_table"],
                    softmax_scale=common["softmax_scale"],
                    causal=True,
                    window_size=common.get("window_size", (None, None)),
                    num_splits=common["num_splits"],
                    score_mod=sm,
                    aux_tensors=[bias],
                )
                stats("score_mod(base)", out_base, ref, BATCH, s_kv, q_len)
                ob = out_base[0] if isinstance(out_base, tuple) else out_base
                a = out_sm[0] if isinstance(out_sm, tuple) else out_sm
                print(
                    f"  {'vendored-vs-base':24s} "
                    f"max={(a.double() - ob.double()).abs().max().item():.3e} "
                    f"bitwise_frac_diff={(a != ob).float().mean().item():.4f}"
                )
            a = out_sm[0] if isinstance(out_sm, tuple) else out_sm
            c = out_sh[0] if isinstance(out_sh, tuple) else out_sh
            print(
                f"  {'shear-vs-score_mod':24s} max={(a.double() - c.double()).abs().max().item():.3e}"
            )
            worst = max(worst, e2)
            del ref
            torch.cuda.empty_cache()
    print(f"done. worst shear-vs-ref max err = {worst:.3e}")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
