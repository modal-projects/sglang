"""Opt 4 — fused residual-add + RMSNorm in InklingDecoderLayer (srt/models/inkling.py).

OLD (c262556c2): inkling-custom RMSNorm + explicit residual add, e.g.
    hidden = hidden + attn_out            # standalone add kernel
    mlp_input = mlp_norm(hidden)          # custom-Triton rmsnorm kernel
    (4 global-memory passes per layer for the two norm+add pairs)
NEW: SGLang SRT RMSNorm two-arg fused form
    hidden, residual = mlp_norm(hidden, residual)   # fused_add_rmsnorm (sgl_kernel)
    (2 passes per layer)

Variants:
  old_add+tmlnorm  : explicit add  + custom-Triton rmsnorm        [true old path]
  srt_unfused      : explicit add  + SRT rmsnorm (no residual)    [isolates fusion only]
  new_fused        : SRT rmsnorm(x, residual) -> fused_add_rmsnorm
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import DEVICE, DTYPE, Collector, bench, print_latency_table, rand, set_seed

from sglang.srt.layers.layernorm import RMSNorm as SRTRMSNorm
from sglang.tml.kernels.norm import rmsnorm as OLD_rmsnorm

OPT = "opt4_fused_add_rmsnorm"
col = Collector(OPT, "Decoder residual+norm: add + (custom rmsnorm) -> fused_add_rmsnorm")
EPS = 1e-6


def old_unfused(x, residual, w):
    res = x + residual
    normed = OLD_rmsnorm(res, w, EPS)
    return normed, res


def srt_unfused(x, residual, norm):
    res = x + residual
    normed = norm(res)
    return normed, res


def correctness() -> None:
    print("\n=== opt4 fused add+rmsnorm correctness ===")
    for T, H in [(1024, 2048), (4096, 4096), (16384, 6144)]:
        set_seed(0)
        x0 = rand(T, H)
        r0 = rand(T, H)
        norm = SRTRMSNorm(H, eps=EPS).to(DEVICE).to(DTYPE)
        norm.weight = torch.nn.Parameter(torch.rand(H, device=DEVICE, dtype=DTYPE) * 0.5 + 0.5)
        w = norm.weight.data

        # fp32 reference: rms_norm(x + residual)
        res_ref = (x0.float() + r0.float())
        normed_ref = torch.nn.functional.rms_norm(res_ref, (H,), w.float(), EPS).to(DTYPE)

        normed_old, res_old = old_unfused(x0.clone(), r0.clone(), w)
        # SRT fused: forward mutates copies in place, returns (normed, residual)
        xc, rc = x0.clone(), r0.clone()
        normed_new, res_new = norm(xc, rc)

        col.check(f"normed vs fp32 T={T} H={H}", normed_ref, normed_new, atol=3e-2)
        col.check(f"residual new-vs-old T={T} H={H}", res_old, res_new, atol=3e-2)
        col.check(f"normed new-vs-old T={T} H={H}", normed_old, normed_new, atol=3e-2)


def latency() -> None:
    print("\n=== opt4 fused add+rmsnorm latency ===")
    for T in [1024, 4096, 8192, 16384]:
        for H in [2048, 4096, 6144]:
            set_seed(0)
            x = rand(T, H)
            residual = rand(T, H)
            norm = SRTRMSNorm(H, eps=EPS).to(DEVICE).to(DTYPE)
            norm.weight = torch.nn.Parameter(
                torch.rand(H, device=DEVICE, dtype=DTYPE) * 0.5 + 0.5
            )
            w = norm.weight.data
            xf, rf = x.clone(), residual.clone()  # mutated in place by fused path
            variants = {
                "old_add_tmlnorm": bench(lambda: old_unfused(x, residual, w)),
                "srt_unfused": bench(lambda: srt_unfused(x, residual, norm)),
                "new_fused": bench(lambda: norm(xf, rf)),
            }
            col.latency_row({"T": T, "H": H}, variants, baseline_key="old_add_tmlnorm")
            print(
                f"  T={T:6d} H={H:5d} | old {variants['old_add_tmlnorm']:7.1f}  "
                f"srt_unfused {variants['srt_unfused']:7.1f}  new {variants['new_fused']:7.1f} us"
            )


if __name__ == "__main__":
    print("=" * 72)
    print(f"{OPT}: {col.title}")
    print("=" * 72)
    correctness()
    latency()
    print_latency_table(
        col.latency,
        shape_keys=["T", "H"],
        variant_keys=["old_add_tmlnorm", "srt_unfused", "new_fused"],
        baseline_key="old_add_tmlnorm",
        title="Decoder add+rmsnorm latency (us) — lower is better",
    )
    col.emit()
