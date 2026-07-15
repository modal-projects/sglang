"""Opt 2 — RMSNorm wrapper (tml/layers/norm.py).

OLD (c262556c2): ``from sglang.tml.kernels.norm import rmsnorm`` (custom Triton kernel);
    forward did ``y = rmsnorm(x.contiguous().view(-1, H), weight, eps)``.
NEW: ``from sgl_kernel import rmsnorm`` (FlashInfer); forward does
    ``y = rmsnorm(x.reshape(-1, H), weight.to(x_2d.dtype), eps)``.

Bundles two effects, isolated here:
  * kernel swap     : sgl_kernel vs custom-Triton, both on a contiguous 2D input
  * full wrapper    : new(reshape+sgl) vs old(contiguous-view+custom), same logical input
Checked on contiguous inputs and on the non-contiguous qkvr-split layout.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import DEVICE, DTYPE, Collector, bench, print_latency_table, rand, set_seed

from sglang.tml.kernels.norm import rmsnorm as OLD_rmsnorm  # custom Triton
from sgl_kernel import rmsnorm as NEW_rmsnorm  # FlashInfer

OPT = "opt2_rmsnorm"
col = Collector(OPT, "RMSNorm wrapper: custom-Triton+contiguous.view -> sgl_kernel+reshape")
EPS = 1e-6


def old_wrapper(x: torch.Tensor, w: torch.Tensor, H: int) -> torch.Tensor:
    x_2d = x.contiguous().view(-1, H)
    y = OLD_rmsnorm(x_2d, w, EPS)
    return y.view(x.shape)


def new_wrapper(x: torch.Tensor, w: torch.Tensor, H: int) -> torch.Tensor:
    x_2d = x.reshape(-1, H)
    y = NEW_rmsnorm(x_2d, w.to(x_2d.dtype), EPS)
    return y.view(x.shape)


def _make_input(rows: int, H: int, from_split: bool) -> torch.Tensor:
    if from_split:
        qkvr = rand(rows, H * 4)
        x = qkvr[:, :H].view(rows, 1, H)  # strides (4H, H, 1) -> non-contiguous
        assert not x.is_contiguous()
        return x
    return rand(rows, 1, H)


def correctness() -> None:
    print("\n=== opt2 rmsnorm correctness (vs fp32 ref, and old vs new) ===")
    for rows, H in [(512, 64), (4096, 128), (16384, 128), (8192, 256)]:
        for from_split in (False, True):
            set_seed(0)
            x = _make_input(rows, H, from_split)
            w = torch.rand(H, device=DEVICE, dtype=DTYPE) * 0.5 + 0.5
            ref = torch.nn.functional.rms_norm(
                x.float(), (H,), w.float(), EPS
            ).to(DTYPE)
            old = old_wrapper(x, w, H)
            new = new_wrapper(x, w, H)
            tag = "split" if from_split else "contig"
            col.check(f"old-vs-fp32 rows={rows} H={H} [{tag}]", ref, old, atol=2e-2)
            col.check(f"new-vs-fp32 rows={rows} H={H} [{tag}]", ref, new, atol=2e-2)
            col.check(f"old-vs-new  rows={rows} H={H} [{tag}]", old, new, atol=2e-2)


def latency() -> None:
    print("\n=== opt2 rmsnorm latency ===")
    for rows in [512, 4096, 16384, 65536]:
        for H in [64, 128, 256]:
            set_seed(0)
            # non-contiguous split input == the real q/k layout exercised by the wrapper
            x = _make_input(rows, H, from_split=True)
            w = torch.rand(H, device=DEVICE, dtype=DTYPE) * 0.5 + 0.5
            x_c2d = x.reshape(-1, H).contiguous()  # pure-kernel comparison input
            wc = w.to(DTYPE)
            variants = {
                "old_full": bench(lambda: old_wrapper(x, w, H)),
                "new_full": bench(lambda: new_wrapper(x, w, H)),
                "old_kernel_2d": bench(lambda: OLD_rmsnorm(x_c2d, wc, EPS)),
                "new_kernel_2d": bench(lambda: NEW_rmsnorm(x_c2d, wc, EPS)),
            }
            col.latency_row({"rows": rows, "H": H}, variants, baseline_key="old_full")
            print(
                f"  rows={rows:6d} H={H:3d} | old_full {variants['old_full']:7.1f}  "
                f"new_full {variants['new_full']:7.1f}  "
                f"old_k {variants['old_kernel_2d']:7.1f}  new_k {variants['new_kernel_2d']:7.1f} us"
            )


if __name__ == "__main__":
    print("=" * 72)
    print(f"{OPT}: {col.title}")
    print("=" * 72)
    correctness()
    latency()
    print_latency_table(
        col.latency,
        shape_keys=["rows", "H"],
        variant_keys=["old_full", "new_full", "old_kernel_2d", "new_kernel_2d"],
        baseline_key="old_full",
        title="RMSNorm latency (us) — lower is better",
    )
    col.emit()
