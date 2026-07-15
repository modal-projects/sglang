"""Prove/regress the fp8-descale sheared-bias bug.

With fp8 q/k and qk_descale != 1, the kernel folds qk_descale into scale_log2
(flash_fwd_sm100 softmax_loop). The sheared bias enters raw scores as
bias * inv_softmax_scale with inv_softmax_scale = 1/softmax_scale, so its
post-exp2 contribution becomes bias * qk_descale instead of bias. The score_mod
path pre-scales scores by softmax_scale*qk_descale before the mod and is immune.

descale=1 (what the fp8 bench uses) hides the bug. This test uses descale=0.5
(qk_descale=0.25) and compares both paths against an fp64 reference on the
EFFECTIVE (dequantized) values, so fp8 quantization noise is excluded from the
reference and only kernel arithmetic differs.

Expected before fix: shear error >> score_mod error. After fix: comparable.

Run:  python benchmark/tml/fusion/test_shear_bias_fp8_descale.py
"""

import torch

from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

from sglang.jit_kernel.flash_attn.cute import flash_attn_varlen_func as _fn
from sglang.srt.models.inkling_common.attn import (
    get_inkling_relative_attention_score_mod,
)

DEV = "cuda"
NH, NHK, D = 64, 8, 128
PAGE_SIZE = 128
REL_EXTENT = 128
DESCALE = 0.5  # per-head q/k/v descale; qk_descale = 0.25


def ref_fp64(q_eff, k_eff, v_eff, bias, b, s_kv, q_len, scale):
    outs = []
    for bi in range(b):
        k = k_eff[bi]  # (s_kv, hk, d) fp64 effective
        v = v_eff[bi]
        qb = q_eff[bi * q_len : (bi + 1) * q_len]
        bb = bias[bi * q_len : (bi + 1) * q_len].double()
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
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", p, v))
    return torch.cat(outs, dim=0)


def main():
    torch.manual_seed(0)
    b, s_kv, q_len = 2, 512, 256
    scale = D**-0.5
    pages_per_seq = (s_kv + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = b * pages_per_seq
    total_q = b * q_len

    q_bf = torch.randn(total_q, NH, D, device=DEV, dtype=torch.bfloat16)
    kc_bf = torch.randn(
        total_pages, PAGE_SIZE, NHK, D, device=DEV, dtype=torch.bfloat16
    )
    vc_bf = torch.randn(
        total_pages, PAGE_SIZE, NHK, D, device=DEV, dtype=torch.bfloat16
    )
    bias = torch.randn(total_q, NH, REL_EXTENT, device=DEV, dtype=torch.bfloat16)

    # store x/DESCALE in fp8; effective value = fp8 * DESCALE
    q8 = (q_bf.float() / DESCALE).to(torch.float8_e4m3fn)
    k8 = (kc_bf.float() / DESCALE).to(torch.float8_e4m3fn)
    v8 = (vc_bf.float() / DESCALE).to(torch.float8_e4m3fn)
    q_eff = q8.double() * DESCALE
    k_eff = (k8.double() * DESCALE).view(b, -1, NHK, D)[:, :s_kv]
    v_eff = (v8.double() * DESCALE).view(b, -1, NHK, D)[:, :s_kv]

    page_table = torch.arange(total_pages, dtype=torch.int32, device=DEV).view(
        b, pages_per_seq
    )
    cache_seqlens = torch.full((b,), s_kv, dtype=torch.int32, device=DEV)
    cu_seqlens_q = torch.arange(0, total_q + 1, q_len, dtype=torch.int32, device=DEV)
    desc = torch.full((b, NHK), DESCALE, device=DEV, dtype=torch.float32)

    kw = dict(
        page_table=page_table,
        seqused_k=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=q_len,
        softmax_scale=scale,
        causal=True,
        num_splits=1,
        q_descale=desc,
        k_descale=desc,
        v_descale=desc,
    )

    ref = ref_fp64(q_eff, k_eff, v_eff, bias, b, s_kv, q_len, scale)
    sm = get_inkling_relative_attention_score_mod(REL_EXTENT)

    def err(out):
        out = out[0] if isinstance(out, tuple) else out
        return (out.double() - ref).abs().max().item()

    e_sm = err(_fn(q8, k8, v8, score_mod=sm, aux_tensors=[bias], **kw))
    e_sh = err(_fn(q8, k8, v8, rel_bias=bias, **kw))
    print(
        f"fp8 descale={DESCALE} (qk_descale={DESCALE**2}) b={b} kv={s_kv} q={q_len} re={REL_EXTENT}"
    )
    print(f"  score_mod vs fp64 ref: max={e_sm:.3e}")
    print(f"  shear     vs fp64 ref: max={e_sh:.3e}")
    ok = e_sh < 3 * e_sm + 1e-2
    print("PASS" if ok else "FAIL: sheared-bias error is inflated by qk_descale")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
