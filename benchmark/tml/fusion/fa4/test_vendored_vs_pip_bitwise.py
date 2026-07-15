"""Check vendored sheared bias against the pip score_mod baseline.

Runs identical inputs through the vendored FA4 kernel and the pip flash-attn-4
package. The pip package does not expose rel_bias, so MODE=rel_bias compares
vendored rel_bias=bias against pip score_mod=... aux_tensors=[bias]. All modes
are expected BITWISE identical (out and LSE), any split count: the rel_bias
path replicates the score_mod arithmetic (scale-then-add, LOG2_E softmax
domain, per-block ex2-emu gating, runtime-arg scale_log2 operand).

Knobs isolate hypotheses:

  EX2EMU=0    patch ex2_emu_freq=0 into BOTH kernels' _TUNING_CONFIG before any
              compile -> if diffs vanish, the graft changed which fragments get
              FFMA-emulated exp2 vs hardware exp2 (element-mapping shift).
  SPLITS=n    force num_splits (default 1: no split combine in play).
  MODE=rel_bias|score_mod|plain (default rel_bias)
  REL_BIAS_ATOL=... diagnostic max abs tolerance for MODE=rel_bias (default 2e-3)
  REL_EXTENT=n  bias extent (default 1024); small/huge shifts the band coverage
  ZERO_BIAS=1   zero the bias tensor -> isolates structural vs value divergence
  CASES=b,kv,q;b,kv,q;...  override the default case grid

Run:  python benchmark/tml/fusion/test_vendored_vs_pip_bitwise.py
"""

import os

import torch

from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

import flash_attn.cute.flash_fwd_sm100 as pip_fwd
from flash_attn.cute import flash_attn_varlen_func as pip_fn

import sglang.jit_kernel.flash_attn.cute.flash_fwd_sm100 as ven_fwd
from sglang.jit_kernel.flash_attn.cute import flash_attn_varlen_func as ven_fn
from sglang.srt.models.inkling_common.attn import (
    get_inkling_relative_attention_score_mod,
)

DEV = "cuda"
NH, NHK, D = 64, 8, 128
PAGE_SIZE = 128
REL_EXTENT = int(os.environ.get("REL_EXTENT", "1024"))
SPLITS = int(os.environ.get("SPLITS", "1"))
REL_BIAS_ATOL = float(os.environ.get("REL_BIAS_ATOL", "2e-3"))
FAILS = []

if os.environ.get("EX2EMU") == "0":
    for cfg in (pip_fwd._TUNING_CONFIG, ven_fwd._TUNING_CONFIG):
        for v in cfg.values():
            v["ex2_emu_freq"] = 0
    print("[patched] ex2_emu_freq=0 in both kernels")


def make(b, s_kv, q_len):
    torch.manual_seed(0)
    total_q = b * q_len
    pages = b * ((s_kv + PAGE_SIZE - 1) // PAGE_SIZE)
    kc = torch.randn(pages, PAGE_SIZE, NHK, D, device=DEV, dtype=torch.bfloat16)
    vc = torch.randn(pages, PAGE_SIZE, NHK, D, device=DEV, dtype=torch.bfloat16)
    pt = torch.arange(pages, dtype=torch.int32, device=DEV).view(b, -1)
    q = torch.randn(total_q, NH, D, device=DEV, dtype=torch.bfloat16)
    bias = torch.randn(total_q, NH, REL_EXTENT, device=DEV, dtype=torch.bfloat16)
    if os.environ.get("ZERO_BIAS") == "1":
        bias.zero_()
    cu = torch.arange(0, total_q + 1, q_len, dtype=torch.int32, device=DEV)
    sl = torch.full((b,), s_kv, dtype=torch.int32, device=DEV)
    kw = dict(
        cu_seqlens_q=cu,
        seqused_k=sl,
        max_seqlen_q=q_len,
        page_table=pt,
        softmax_scale=1.0 / D,
        causal=True,
        num_splits=SPLITS,
    )
    return q, kc, vc, bias, kw


def diff_report(tag, a, b, q_len):
    a = a[0] if isinstance(a, tuple) else a
    b = b[0] if isinstance(b, tuple) else b
    neq = a != b
    n = int(neq.sum().item())
    print(
        f"  {tag:20s} bitwise_diff={n}/{neq.numel()} ({n / neq.numel():.4%}) "
        f"max={(a.double() - b.double()).abs().max().item():.3e}"
    )
    if n:
        idx = neq.nonzero()
        rows = idx[:, 0]
        heads = idx[:, 1]
        dims = idx[:, 2]
        print(
            f"    rows: min={rows.min().item()} max={rows.max().item()} "
            f"n_uniq={len(rows.unique())}  row%128 uniq={sorted(set((rows % 128).tolist()))[:10]}"
        )
        print(
            f"    heads uniq={len(heads.unique())}  dim%32 uniq={sorted(set((dims % 32).tolist()))[:16]}"
        )
    return n


def tolerance_report(tag, a, b, atol):
    a = a[0] if isinstance(a, tuple) else a
    b = b[0] if isinstance(b, tuple) else b
    diff = (a.float() - b.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ok = max_err <= atol
    print(
        f"  {tag:20s} {'OK' if ok else 'FAIL'} max={max_err:.3e}/{atol:.1e} mean={mean_err:.3e}"
    )
    if not ok:
        FAILS.append(tag)


def main():
    sm = get_inkling_relative_attention_score_mod(REL_EXTENT)
    modes = os.environ.get("MODE", "rel_bias").split(",")
    cases = [(2, 1211, 1), (2, 1211, 256), (2, 8261, 256)]
    if os.environ.get("CASES"):
        cases = [
            tuple(int(x) for x in c.split(",")) for c in os.environ["CASES"].split(";")
        ]
    for b, s_kv, q_len in cases:
        q, kc, vc, bias, kw = make(b, s_kv, q_len)
        print(f"case b={b} kv={s_kv} q={q_len} splits={SPLITS}")
        if "plain" in modes:
            ov = ven_fn(q, kc, vc, **kw)
            op = pip_fn(q, kc, vc, **kw)
            diff_report("plain", ov, op, q_len)
        if "score_mod" in modes:
            ov, ov_lse = ven_fn(
                q, kc, vc, score_mod=sm, aux_tensors=[bias], return_lse=True, **kw
            )
            op, op_lse = pip_fn(
                q, kc, vc, score_mod=sm, aux_tensors=[bias], return_lse=True, **kw
            )
            diff_report("score_mod", ov, op, q_len)
            neq_lse = int((ov_lse != op_lse).sum().item())
            print(
                f"  {'score_mod_lse':20s} bitwise_diff={neq_lse}/{op_lse.numel()} "
                f"max={(ov_lse.double() - op_lse.double()).abs().max().item():.3e}"
            )
        if "rel_bias" in modes:
            ov, ov_lse = ven_fn(q, kc, vc, rel_bias=bias, return_lse=True, **kw)
            op, op_lse = pip_fn(
                q, kc, vc, score_mod=sm, aux_tensors=[bias], return_lse=True, **kw
            )
            if diff_report("rel_bias_vs_pip_sm", ov, op, q_len):
                FAILS.append("rel_bias_vs_pip_sm")
            neq_lse = int((ov_lse != op_lse).sum().item())
            print(
                f"  {'rel_bias_lse':20s} bitwise_diff={neq_lse}/{op_lse.numel()} "
                f"max={(ov_lse.double() - op_lse.double()).abs().max().item():.3e}"
            )
            tolerance_report("rel_bias_close", ov, op, REL_BIAS_ATOL)
    if FAILS:
        raise SystemExit(f"FAILURES: {FAILS}")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
