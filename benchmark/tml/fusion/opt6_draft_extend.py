"""Opt 6 — fused draft-extend sconv cache update (_update_sconv_cache_for_draft_extend).

OLD (c262556c2): initial_state gather + padded cat + windows unfold + (optional tracking
    gather/transpose/contiguous/copy_if_needed) + final gather/transpose + scatter.
NEW: ``fused_draft_extend_sconv_cache`` reads the "virtual padded" sequence directly
    (sconv_cache[ci] for j < W-1, else hidden_states) in one kernel.
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

from sglang.srt.models.inkling_common.sconv import fused_draft_extend_sconv_cache

try:
    from sglang.jit_kernel.inkling_sconv import (
        fused_draft_extend_sconv_cache as _cuda_draft_extend,
    )

    HAVE_CUDA = True
except Exception as _e:  # pragma: no cover
    HAVE_CUDA = False
    print(f"[opt6] CUDA-JIT draft_extend unavailable: {_e}")

OPT = "opt6_draft_extend"
col = Collector(
    OPT, "draft-extend sconv cache: gather+unfold+copy -> fused single kernel"
)


def baseline(
    hidden_states,
    sconv_cache,
    cache_indices,
    num_accepted_tokens,
    draft_token_num,
    do_tracking=False,
    crossed=None,
    track_step=None,
    mamba_track_indices=None,
):
    """OLD path: gather + pad + unfold + (tracking) + scatter."""
    B = cache_indices.shape[0]
    initial_state = sconv_cache[cache_indices]
    input_reshaped = hidden_states.view(B, draft_token_num, -1)
    padded = torch.cat([initial_state, input_reshaped], dim=1)
    windows = padded.unfold(1, sconv_cache.shape[1], 1)
    if do_tracking and crossed is not None:
        track_idx = (
            track_step.long().view(-1, 1, 1, 1).expand(-1, 1, *windows.shape[2:])
        )
        track_sel = windows.gather(1, track_idx).squeeze(1).transpose(1, 2).contiguous()
        copy_if_needed(
            src_tensor=track_sel,
            dst_tensor=sconv_cache,
            mask=crossed,
            src_indices=torch.arange(B, device=DEVICE, dtype=torch.int64),
            dst_indices=mamba_track_indices,
            batch_size=B,
        )
    idx = num_accepted_tokens.long().view(-1, 1, 1, 1).expand(-1, 1, *windows.shape[2:])
    selected = windows.gather(1, idx).squeeze(1).transpose(1, 2).contiguous()
    sconv_cache[cache_indices] = selected


def _make(B, T, D, W, do_tracking, all_crossed, pool):
    W_1 = W - 1
    D_total = D * 4  # non-contiguous hidden_states (qkvr split layout)
    qkvr = rand(B * T, D_total)
    hidden_nc = qkvr[:, :D]
    hidden_c = hidden_nc.contiguous()
    ci = torch.arange(B, dtype=torch.int32, device=DEVICE)
    n_acc = torch.randint(0, T + 1, (B,), dtype=torch.int32, device=DEVICE)
    crossed = track_step = mti = None
    if do_tracking:
        crossed = (
            torch.ones(B, dtype=torch.bool, device=DEVICE)
            if all_crossed
            else torch.randint(0, 2, (B,), dtype=torch.bool, device=DEVICE)
        )
        track_step = torch.randint(0, T + 1, (B,), dtype=torch.int32, device=DEVICE)
        mti = torch.arange(B, dtype=torch.int64, device=DEVICE) + B
    return hidden_nc, hidden_c, ci, n_acc, crossed, track_step, mti


def correctness() -> None:
    print("\n=== opt6 draft-extend correctness ===")
    for D, W in [(384, 3), (2304, 4), (6144, 4)]:
        for B in [1, 4, 32]:
            for T in [1, 4, 8, 32]:
                for do_tracking in [False, True]:
                    for all_crossed in ([False, True] if do_tracking else [False]):
                        set_seed(0)
                        pool = max(B * 4, 32)
                        hidden_nc, hidden_c, ci, n_acc, crossed, ts, mti = _make(
                            B, T, D, W, do_tracking, all_crossed, pool
                        )
                        sc_ref = rand(pool, W - 1, D)
                        sc_fused = sc_ref.clone()
                        baseline(
                            hidden_c,
                            sc_ref,
                            ci,
                            n_acc,
                            T,
                            do_tracking,
                            crossed,
                            ts,
                            mti,
                        )
                        fused_draft_extend_sconv_cache(
                            hidden_nc,
                            sc_fused,
                            ci,
                            n_acc,
                            T,
                            do_tracking,
                            crossed,
                            ts,
                            mti,
                        )
                        col.check(
                            f"B={B} T={T} D={D} W={W} track={do_tracking} all_crossed={all_crossed}",
                            sc_ref,
                            sc_fused,
                            atol=1e-5,
                        )
                        if HAVE_CUDA:
                            # fresh base + same inputs; compare CUDA vs an independent baseline run
                            sc_c = rand(pool, W - 1, D)
                            sc_ref_c = sc_c.clone()
                            baseline(
                                hidden_c,
                                sc_ref_c,
                                ci,
                                n_acc,
                                T,
                                do_tracking,
                                crossed,
                                ts,
                                mti,
                            )
                            _cuda_draft_extend(
                                hidden_nc,
                                sc_c,
                                ci,
                                n_acc,
                                T,
                                do_tracking,
                                crossed,
                                ts,
                                mti,
                            )
                            col.check(
                                f"CUDA-vs-ref B={B} T={T} D={D} W={W} track={do_tracking} all_crossed={all_crossed}",
                                sc_ref_c,
                                sc_c,
                                atol=1e-5,
                            )


def latency() -> None:
    print("\n=== opt6 draft-extend latency (with tracking) ===")
    pool = 512
    configs = [
        (8, 8, 2304, 4),
        (32, 8, 2304, 4),
        (8, 8, 6144, 4),
        (32, 8, 6144, 4),
        (8, 32, 2304, 4),
        (32, 32, 6144, 4),
    ]
    for B, T, D, W in configs:
        set_seed(0)
        hidden_nc, hidden_c, ci, n_acc, crossed, ts, mti = _make(
            B, T, D, W, True, True, pool
        )
        sc = rand(pool, W - 1, D)
        variants = {
            "old_gather_unfold": bench(
                lambda: baseline(hidden_c, sc, ci, n_acc, T, True, crossed, ts, mti)
            ),
            "new_fused": bench(
                lambda: fused_draft_extend_sconv_cache(
                    hidden_nc, sc, ci, n_acc, T, True, crossed, ts, mti
                )
            ),
        }
        if HAVE_CUDA:
            variants["cuda_jit"] = bench(
                lambda: _cuda_draft_extend(
                    hidden_nc, sc, ci, n_acc, T, True, crossed, ts, mti
                )
            )
        col.latency_row(
            {"B": B, "T": T, "D": D, "W": W}, variants, baseline_key="old_gather_unfold"
        )
        cuda_str = f"  cuda {variants['cuda_jit']:7.1f}" if HAVE_CUDA else ""
        print(
            f"  B={B:3d} T={T:3d} D={D:5d} | old {variants['old_gather_unfold']:7.1f}  "
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
        shape_keys=["B", "T", "D", "W"],
        variant_keys=["old_gather_unfold", "new_fused", "cuda_jit"],
        baseline_key="old_gather_unfold",
        title="draft-extend latency (us) — lower is better",
    )
    col.emit()
