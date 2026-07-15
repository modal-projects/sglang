"""Opt 8 — fuse the prefix-cache track-copy into fused_causal_conv1d_update_decode.

Prefix caching with the mamba extra buffer needs the post-update conv window of each
tracked sequence b snapshotted into a persistent ping-pong slot track_indices[b].

OLD (HEAD): fused_causal_conv1d_update_decode(...)  [conv + cache shift-update]
            THEN a separate copy_if_needed launch (_track_conv_state_decode):
                sconv_cache[track_indices[b]] = sconv_cache[cache_indices[b]]  if track_mask[b]
NEW (working tree): fused_causal_conv1d_update_decode(..., track_mask, track_indices)
            writes the post-update window to BOTH cache_indices[b] and track_indices[b]
            in-register (DO_TRACK) — no second kernel launch, no row re-read.

Race-freedom rests on the invariant that working slots (cache_indices) and ping-pong
track slots (track_indices) are pairwise-distinct pool allocations. We test under that
invariant (incl. scattered/disjoint track slots, partial track masks, PAD lanes).

Requires the track-fused kernel (working-tree change to kernels/sconv.py); the old
copy_if_needed (still in layers/sconv.py at this branch) provides the OLD baseline.
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
from opt7_decode_update import PAD, reference_decode

from sglang.srt.models.inkling_common.kernels.sconv import (
    fused_causal_conv1d_update_decode,
)

OPT = "opt8_track_fusion"
col = Collector(
    OPT, "decode track-copy: fused_decode + separate copy_if_needed -> fused in-kernel"
)

if "track_mask" not in fused_causal_conv1d_update_decode.__doc__:
    print(
        "WARNING: fused_causal_conv1d_update_decode lacks track_mask — track fusion not present!"
    )


def make_track(T, D, W, track_pat, pad, pool, scattered=False):
    """Distinct working slots + DISJOINT distinct track slots (the pool invariant)."""
    qkvr = rand(T, D * 4)
    x = qkvr[:, :D]  # non-contiguous k/v split
    weight = rand(D, W)
    sconv_cache = rand(pool, W - 1, D)
    perm = torch.randperm(pool, device=DEVICE)
    cache_indices = perm[:T].to(torch.int32)  # working slots
    track_indices = perm[T : 2 * T].clone().to(torch.int64)  # disjoint track slots
    if (
        scattered
    ):  # shuffle track slots (still disjoint from working) — distinctness, not adjacency
        track_indices = track_indices[torch.randperm(T, device=DEVICE)]
    cache_mask = torch.ones(T, dtype=torch.bool, device=DEVICE)
    if pad and T >= 3:
        cache_indices[1::3] = PAD
        cache_mask = cache_indices != PAD
    valid = cache_indices != PAD
    if track_pat == "all":
        track_mask = valid.clone()
    elif track_pat == "half":
        track_mask = valid & (torch.arange(T, device=DEVICE) % 2 == 0)
    else:  # none
        track_mask = torch.zeros(T, dtype=torch.bool, device=DEVICE)
    return x, weight, sconv_cache, cache_indices, cache_mask, track_mask, track_indices


def reference_track(x, weight, cache, ci, cm, track_mask, track_indices, act, res):
    y, new_cache = reference_decode(x, weight, cache, ci, cm, act, res)
    do = (ci != PAD) & track_mask
    if do.any():
        b = do.nonzero(as_tuple=True)[0]
        new_cache[track_indices[b]] = new_cache[ci[b].clamp(min=0).long()]
    return y, new_cache


def old_path(x, weight, cache, ci, cm, track_mask, track_indices, act, res):
    """OLD: fused decode (no track) then separate copy_if_needed track-copy."""
    y = fused_causal_conv1d_update_decode(
        x, weight, cache, ci, cm, activation=act, use_residual=res
    )
    copy_if_needed(
        src_tensor=cache,
        mask=track_mask,
        src_indices=ci,
        dst_indices=track_indices,
        batch_size=ci.shape[0],
    )
    return y


def correctness() -> None:
    print(
        "\n=== opt8 track-fusion correctness (new vs ref, and new vs old separate-copy) ==="
    )
    for D, W in [(384, 3), (2304, 4), (6144, 4)]:
        for T in [1, 4, 16, 64, 256]:
            for act in (None, "silu"):
                for track_pat in ("all", "half", "none"):
                    for pad in (False, True):
                        for scattered in (False, True):
                            set_seed(0)
                            pool = max(4 * T, 32)
                            x, w, sc0, ci, cm, tm, ti = make_track(
                                T, D, W, track_pat, pad, pool, scattered
                            )
                            ref_y, ref_cache = reference_track(
                                x, w, sc0, ci, cm, tm, ti, act, True
                            )
                            tag = (
                                f"T={T} D={D} W={W} act={act} track={track_pat}"
                                f"{' pad' if pad else ''}{' scat' if scattered else ''}"
                            )
                            # NEW: fused track-copy
                            sc_new = sc0.clone()
                            y_new = fused_causal_conv1d_update_decode(
                                x,
                                w,
                                sc_new,
                                ci,
                                cm,
                                activation=act,
                                use_residual=True,
                                track_mask=tm,
                                track_indices=ti,
                            )
                            # OLD: fused decode + separate copy_if_needed
                            sc_old = sc0.clone()
                            y_old = old_path(x, w, sc_old, ci, cm, tm, ti, act, True)

                            col.check(
                                f"NEW-vs-ref cache {tag}",
                                ref_cache,
                                sc_new,
                                atol=2e-2,
                                rtol=2e-2,
                            )
                            col.check(
                                f"NEW-vs-OLD cache {tag}",
                                sc_old,
                                sc_new,
                                atol=0,
                                rtol=0,
                            )
                            col.check(
                                f"NEW-vs-OLD y     {tag}", y_old, y_new, atol=0, rtol=0
                            )


def latency() -> None:
    print("\n=== opt8 track-fusion latency (track_mask all-True; W=4) ===")
    W = 4
    for D in (2304, 4096, 6144):
        for T in (1, 4, 16, 64, 256):
            set_seed(0)
            pool = max(4 * T, 64)
            x, w, sc0, ci, cm, tm, ti = make_track(T, D, W, "all", False, pool)
            sc_a, sc_b = sc0.clone(), sc0.clone()
            variants = {
                "old_fused+copy_if_needed": bench(
                    lambda: old_path(x, w, sc_a, ci, cm, tm, ti, "silu", True)
                ),
                "new_fused_track": bench(
                    lambda: fused_causal_conv1d_update_decode(
                        x,
                        w,
                        sc_b,
                        ci,
                        cm,
                        activation="silu",
                        use_residual=True,
                        track_mask=tm,
                        track_indices=ti,
                    )
                ),
            }
            col.latency_row(
                {"T": T, "D": D}, variants, baseline_key="old_fused+copy_if_needed"
            )
            print(
                f"  T={T:5d} D={D:5d} | old(fused+copy) {variants['old_fused+copy_if_needed']:7.1f}  "
                f"new(fused track) {variants['new_fused_track']:7.1f} us"
            )


if __name__ == "__main__":
    print("=" * 72)
    print(f"{OPT}: {col.title}")
    print("=" * 72)
    correctness()
    latency()
    print_latency_table(
        col.latency,
        shape_keys=["T", "D"],
        variant_keys=["old_fused+copy_if_needed", "new_fused_track"],
        baseline_key="old_fused+copy_if_needed",
        title="decode track-copy latency (us) — lower is better",
    )
    col.emit()
