"""Opt 3 — fused in-place QK RMSNorm (tml/layers/attn.py).

OLD (c262556c2):
    q = self.q_norm(q.contiguous().view(-1, head_dim))   # copy + custom rmsnorm
    k = self.k_norm(k.contiguous().view(-1, head_dim))   # copy + custom rmsnorm
    (two separate kernels, two .contiguous() copies of the strided qkvr split)
NEW: a single ``apply_qk_norm`` call whose eligible CUDA/bf16 path runs
    ``fused_inplace_qknorm`` over the strided q/k views in place (one kernel, no copy).

Latency compares the OLD two-call+copy pattern against the NEW fused in-place kernel.
Correctness compares the normalized values (and exercises the integrated apply_qk_norm).
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import DEVICE, DTYPE, Collector, bench, print_latency_table, rand, set_seed

from sglang.srt.models.inkling_common.norm import RMSNorm
from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm, fused_inplace_qknorm

OPT = "opt3_qk_norm"
col = Collector(OPT, "QK RMSNorm: 2x(contiguous-copy + custom rmsnorm) -> fused in-place")
EPS = 1e-6


def make_qk(T, Hq, Hk, hd):
    """q, k as non-contiguous slices of a fused qkvr projection output."""
    width = (Hq + 2 * Hk) * hd  # q + k + v widths, contiguous projection
    qkvr = rand(T, width)
    q = qkvr[:, : Hq * hd]
    k = qkvr[:, Hq * hd : (Hq + Hk) * hd]
    assert not q.is_contiguous() and not k.is_contiguous()
    return q, k


def old_pattern(q, k, q_norm, k_norm, hd):
    qn = q_norm(q.contiguous().view(-1, hd))
    kn = k_norm(k.contiguous().view(-1, hd))
    return qn, kn


def correctness() -> None:
    print("\n=== opt3 qk-norm correctness ===")
    print(f"  can_use_fused_inplace_qknorm(128, bf16) = {can_use_fused_inplace_qknorm(128, DTYPE)}")
    for T, Hq, Hk, hd in [(64, 8, 8, 128), (512, 32, 8, 128), (4096, 16, 16, 128)]:
        set_seed(0)
        q, k = make_qk(T, Hq, Hk, hd)
        q_norm = RMSNorm(hd, eps=EPS).to(DEVICE).to(DTYPE)
        k_norm = RMSNorm(hd, eps=EPS).to(DEVICE).to(DTYPE)
        q_norm.weight = torch.nn.Parameter(torch.rand(hd, device=DEVICE, dtype=DTYPE) * 0.5 + 0.5)
        k_norm.weight = torch.nn.Parameter(torch.rand(hd, device=DEVICE, dtype=DTYPE) * 0.5 + 0.5)

        # OLD reference (separate norms), shaped [T, H, hd]
        qn_old, kn_old = old_pattern(q, k, q_norm, k_norm, hd)
        qn_old = qn_old.view(T, Hq, hd)
        kn_old = kn_old.view(T, Hk, hd)

        # NEW fused in-place on fresh copies of the strided views
        q2 = q.clone()
        k2 = k.clone()
        fused_inplace_qknorm(
            q2.view(T, -1, hd), k2.view(T, -1, hd),
            q_norm.weight, k_norm.weight, EPS,
        )
        qn_new = q2.view(T, Hq, hd)
        kn_new = k2.view(T, Hk, hd)

        col.check(f"q fused-vs-old T={T} Hq={Hq} hd={hd}", qn_old, qn_new, atol=2e-2)
        col.check(f"k fused-vs-old T={T} Hk={Hk} hd={hd}", kn_old, kn_new, atol=2e-2)


def latency() -> None:
    print("\n=== opt3 qk-norm latency ===")
    configs = [(512, 32, 8, 128), (2048, 32, 8, 128), (4096, 16, 16, 128), (8192, 32, 8, 128)]
    for T, Hq, Hk, hd in configs:
        set_seed(0)
        q, k = make_qk(T, Hq, Hk, hd)
        q_norm = RMSNorm(hd, eps=EPS).to(DEVICE).to(DTYPE)
        k_norm = RMSNorm(hd, eps=EPS).to(DEVICE).to(DTYPE)
        qw = (torch.rand(hd, device=DEVICE, dtype=DTYPE) * 0.5 + 0.5)
        kw = (torch.rand(hd, device=DEVICE, dtype=DTYPE) * 0.5 + 0.5)
        q_norm.weight = torch.nn.Parameter(qw)
        k_norm.weight = torch.nn.Parameter(kw)
        qv = q.view(T, -1, hd)
        kv = k.view(T, -1, hd)
        variants = {
            "old_2norm_copy": bench(lambda: old_pattern(q, k, q_norm, k_norm, hd)),
            "new_fused_inplace": bench(
                lambda: fused_inplace_qknorm(qv, kv, qw, kw, EPS)
            ),
        }
        col.latency_row(
            {"T": T, "Hq": Hq, "Hk": Hk, "hd": hd}, variants, baseline_key="old_2norm_copy"
        )
        print(
            f"  T={T:5d} Hq={Hq} Hk={Hk} hd={hd} | "
            f"old {variants['old_2norm_copy']:7.1f}  new {variants['new_fused_inplace']:7.1f} us"
        )


if __name__ == "__main__":
    print("=" * 72)
    print(f"{OPT}: {col.title}")
    print("=" * 72)
    correctness()
    latency()
    print_latency_table(
        col.latency,
        shape_keys=["T", "Hq", "Hk", "hd"],
        variant_keys=["old_2norm_copy", "new_fused_inplace"],
        baseline_key="old_2norm_copy",
        title="QK-norm latency (us) — lower is better",
    )
    col.emit()
