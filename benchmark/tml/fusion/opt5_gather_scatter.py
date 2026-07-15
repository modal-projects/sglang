"""Opt 5 — fused gather->scatter into sconv cache (_prepare_extend_sconv_cache).

OLD (c262556c2): ``track = hidden_states[track_conv_indices].contiguous()`` materialises
    a [B, W-1, D] buffer, then ``copy_if_needed`` scatters masked rows into sconv_cache.
NEW: ``fused_gather_scatter_to_sconv_cache`` writes masked rows straight from
    hidden_states into sconv_cache — no intermediate [B, W-1, D] allocation.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _copy_if_needed import copy_if_needed  # vendored: removed from layers/sconv.py
from _harness import (
    DEVICE,
    Collector,
    bench,
    print_latency_table,
    rand,
    set_seed,
)

from sglang.srt.models.inkling_common.sconv import fused_gather_scatter_to_sconv_cache

try:
    from sglang.jit_kernel.inkling_sconv import (
        fused_gather_scatter_to_sconv_cache as _cuda_gather_scatter,
    )

    HAVE_CUDA = True
except Exception as _e:  # pragma: no cover
    HAVE_CUDA = False
    print(f"[opt5] CUDA-JIT gather_scatter unavailable: {_e}")

OPT = "opt5_gather_scatter"
col = Collector(
    OPT, "sconv-cache gather->scatter: gather.contiguous()+copy_if_needed -> fused"
)


def baseline(hidden_states, sconv_cache, track_conv_indices, mask, dst_indices):
    """OLD path: eager gather + copy_if_needed."""
    B = mask.shape[0]
    track = hidden_states[track_conv_indices].contiguous()  # [B, W-1, D]
    copy_if_needed(
        src_tensor=track,
        dst_tensor=sconv_cache,
        mask=mask,
        src_indices=torch.arange(B, device=DEVICE, dtype=torch.int64),
        dst_indices=dst_indices,
        batch_size=B,
    )


def _make(T, D, W, B, mask_pattern, pool):
    W_1 = W - 1
    hidden = rand(T, D)
    base = torch.randint(0, max(T - W_1, 1), (B,), device=DEVICE)
    track = (
        (base.unsqueeze(1) + torch.arange(W_1, device=DEVICE))
        .clamp(max=T - 1)
        .to(torch.int32)
    )
    if mask_pattern == "all_true":
        mask = torch.ones(B, dtype=torch.bool, device=DEVICE)
    elif mask_pattern == "all_false":
        mask = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    elif mask_pattern == "alternating":
        mask = torch.arange(B, device=DEVICE) % 2 == 0
    else:
        mask = torch.randint(0, 2, (B,), dtype=torch.bool, device=DEVICE)
    dst = torch.randperm(pool, device=DEVICE)[:B].to(torch.int64)
    return hidden, track, mask, dst


def correctness() -> None:
    print("\n=== opt5 gather->scatter correctness ===")
    T = 4096
    for D in [384, 2304, 6144]:
        for W in [3, 4, 5]:
            for B in [1, 8, 64]:
                for mp in ["all_true", "all_false", "alternating", "random"]:
                    set_seed(0)
                    pool = max(B * 4, 16)
                    hidden, track, mask, dst = _make(T, D, W, B, mp, pool)
                    sc_ref = rand(pool, W - 1, D)
                    sc_fused = sc_ref.clone()
                    sc_orig = sc_ref.clone()
                    baseline(hidden, sc_ref, track, mask, dst)
                    fused_gather_scatter_to_sconv_cache(
                        hidden, sc_fused, track, mask, dst
                    )
                    # masked-out slots must be untouched
                    untouched = True
                    for b in range(B):
                        if not mask[b].item():
                            slot = dst[b].item()
                            if not torch.equal(sc_fused[slot], sc_orig[slot]):
                                untouched = False
                                break
                    col.check(
                        f"D={D} W={W} B={B} mask={mp}",
                        sc_ref,
                        sc_fused,
                        atol=1e-5,
                        extra_ok=untouched,
                        extra_msg="" if untouched else "masked slot modified!",
                    )
                    if HAVE_CUDA:
                        sc_cuda = sc_orig.clone()
                        _cuda_gather_scatter(hidden, sc_cuda, track, mask, dst)
                        cu_untouched = all(
                            mask[b].item()
                            or torch.equal(
                                sc_cuda[dst[b].item()], sc_orig[dst[b].item()]
                            )
                            for b in range(B)
                        )
                        col.check(
                            f"CUDA-vs-ref D={D} W={W} B={B} mask={mp}",
                            sc_ref,
                            sc_cuda,
                            atol=1e-5,
                            extra_ok=cu_untouched,
                            extra_msg=(
                                "" if cu_untouched else "CUDA masked slot modified!"
                            ),
                        )


def latency() -> None:
    print("\n=== opt5 gather->scatter latency ===")
    T, pool = 4096, 1024
    configs = [
        (8, 3, 384),
        (8, 4, 2304),
        (8, 5, 6144),
        (64, 4, 2304),
        (64, 5, 6144),
        (256, 4, 2304),
        (256, 4, 6144),
    ]
    for B, W, D in configs:
        set_seed(0)
        hidden, track, mask, dst = _make(T, D, W, B, "all_true", pool)
        sc = rand(pool, W - 1, D)
        variants = {
            "old_gather_copy": bench(lambda: baseline(hidden, sc, track, mask, dst)),
            "new_fused": bench(
                lambda: fused_gather_scatter_to_sconv_cache(
                    hidden, sc, track, mask, dst
                )
            ),
        }
        if HAVE_CUDA:
            variants["cuda_jit"] = bench(
                lambda: _cuda_gather_scatter(hidden, sc, track, mask, dst)
            )
        col.latency_row(
            {"B": B, "W": W, "D": D}, variants, baseline_key="old_gather_copy"
        )
        cuda_str = f"  cuda {variants['cuda_jit']:7.1f}" if HAVE_CUDA else ""
        print(
            f"  B={B:4d} W={W} D={D:5d} | old {variants['old_gather_copy']:7.1f}  "
            f"new {variants['new_fused']:7.1f}{cuda_str} us"
        )


if __name__ == "__main__":
    print("=" * 72)
    print(f"{OPT}: {col.title}")
    print("=" * 72)
    correctness()
    latency()
    print_latency_table(
        col.latency,
        shape_keys=["B", "W", "D"],
        variant_keys=["old_gather_copy", "new_fused", "cuda_jit"],
        baseline_key="old_gather_copy",
        title="gather->scatter latency (us) — lower is better",
    )
    col.emit()
