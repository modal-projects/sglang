"""Opt 7 — decode path: fused causal_conv1d + sconv-cache shift-update.

Covers the decode-side sconv kernels (the runtime-dominant phase, output_len decode
steps), which the extend-focused opt1 did not touch:

  * fused_causal_conv1d_update_decode (CHANGED on this branch: now allocates a
    contiguous [T,D] output + passes stride_y, so it accepts a non-contiguous k/v
    split and emits contiguous output — the decode analogue of opt1's copy fix).
  * the UNFUSED reference path causal_conv1d(is_decode=True) + update_sconv_cache,
    where update_sconv_cache runs the Helion _update_sconv_cache_helion_kernel.

Checks:
  A. NEW fused vs an independent fp32 reference (output y AND updated cache).
  B. OLD fused (vendored) vs NEW fused — values identical, NEW output contiguous.
  C. NEW fused vs UNFUSED (causal_conv1d is_decode + update_sconv_cache) — exercises
     _update_sconv_cache_helion_kernel; looser tol (independent Helion/Triton impls).
Latency: old-vs-new fused, and fused-vs-unfused.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sconv_baseline as OLD
from _harness import (
    DEVICE,
    Collector,
    bench,
    print_latency_table,
    rand,
    set_seed,
)

from sglang.srt.models.inkling_common.kernels.sconv import (
    causal_conv1d,
    fused_causal_conv1d_update_decode,
    precompute_helion_decode_metadata,
    update_sconv_cache,
)
from sglang.tml.kernels import sconv as NEW

try:
    from sglang.jit_kernel.inkling_sconv import (
        fused_causal_conv1d_update_decode as _cuda_fused_decode,
    )

    HAVE_CUDA = True
except Exception as _e:  # pragma: no cover
    HAVE_CUDA = False
    print(f"[opt7] CUDA-JIT fused_decode unavailable: {_e}")

OPT = "opt7_decode_update"
col = Collector(OPT, "decode fused conv+cache-update: old vs new, fused vs unfused")
PAD = -1  # PAD_SLOT_ID


def make_decode(T, D, W, pattern, pool):
    """x as a non-contiguous k/v split; distinct cache slots; varied mask/PAD."""
    qkvr = rand(T, D * 4)
    x_nc = qkvr[:, :D]
    x_c = x_nc.contiguous()
    weight = rand(D, W)
    sconv_cache = rand(pool, W - 1, D)
    # distinct slots so cache writes never race
    ci = torch.randperm(pool, device=DEVICE)[:T].to(torch.int32)
    if pattern == "all_valid":
        cache_mask = torch.ones(T, dtype=torch.bool, device=DEVICE)
    elif pattern == "half_mask":  # some sequences have no initial state
        cache_mask = torch.arange(T, device=DEVICE) % 2 == 0
    else:  # with_pad: some tokens are CUDA-graph padding (ci == -1)
        cache_mask = torch.ones(T, dtype=torch.bool, device=DEVICE)
        if T >= 2:
            ci[1::3] = PAD
            cache_mask = ci != PAD
    return x_c, x_nc, weight, sconv_cache, ci, cache_mask


def reference_decode(x, weight, sconv_cache, ci, cache_mask, activation, use_residual):
    """fp32 reference: conv over (W-1 cached taps + current token), then shift-update."""
    T, D = x.shape
    W = weight.shape[1]
    W1 = W - 1
    xf, wf, cf = x.float(), weight.float(), sconv_cache.float()
    slot = ci.clamp(min=0).long()
    cm = cache_mask.float()
    valid = ci != PAD
    prefix = cf[slot]  # [T, W1, D]
    acc = torch.zeros(T, D, device=DEVICE, dtype=torch.float32)
    for iw in range(W1):
        acc += (prefix[:, iw, :] * cm[:, None]) * wf[:, iw][None, :]
    acc += xf * wf[:, W1][None, :]
    if activation in ("silu", "swish"):
        acc = acc * torch.sigmoid(acc)
    if use_residual:
        acc += xf
    y = acc.to(x.dtype)
    # shifted cache: new[:, iw] = old[slot, iw+1] * cm (iw<W1-1); new[:, W1-1] = x
    new_cache = sconv_cache.clone()
    new_win = torch.empty(T, W1, D, dtype=sconv_cache.dtype, device=DEVICE)
    for iw in range(W1 - 1):
        new_win[:, iw, :] = (prefix[:, iw + 1, :] * cm[:, None]).to(sconv_cache.dtype)
    new_win[:, W1 - 1, :] = x
    if valid.any():
        v = valid.nonzero(as_tuple=True)[0]
        new_cache[slot[v]] = new_win[v]
    return y, new_cache


def correctness() -> None:
    print("\n=== opt7 decode correctness ===")
    for D, W in [(384, 3), (2304, 4), (4096, 4), (6144, 4), (384, 4)]:
        for T in [1, 4, 16, 64, 256, 1024]:
            for act in (None, "silu"):
                for use_res in (True, False):
                    for pat in ("all_valid", "half_mask", "with_pad"):
                        set_seed(0)
                        pool = max(T * 2, 16)
                        x_c, x_nc, w, sc0, ci, cm = make_decode(T, D, W, pat, pool)
                        ref_y, ref_cache = reference_decode(
                            x_c, w, sc0, ci, cm, act, use_res
                        )
                        tag = f"T={T} D={D} W={W} act={act} res={use_res} {pat}"

                        # A. NEW fused vs reference (non-contiguous input)
                        sc = sc0.clone()
                        y_new = fused_causal_conv1d_update_decode(
                            x_nc, w, sc, ci, cm, activation=act, use_residual=use_res
                        )
                        col.check(
                            f"A NEW-vs-ref y {tag}",
                            ref_y,
                            y_new,
                            atol=2e-2,
                            rtol=2e-2,
                            extra_ok=y_new.is_contiguous(),
                            extra_msg=(
                                "" if y_new.is_contiguous() else "y not contiguous!"
                            ),
                        )
                        col.check(
                            f"A NEW-vs-ref cache {tag}",
                            ref_cache,
                            sc,
                            atol=2e-2,
                            rtol=2e-2,
                        )

                        # B. OLD fused (vendored) vs NEW fused — values identical
                        sc_old = sc0.clone()
                        y_old = OLD.fused_causal_conv1d_update_decode(
                            x_c, w, sc_old, ci, cm, activation=act, use_residual=use_res
                        )
                        col.check(
                            f"B OLD-vs-NEW y {tag}", y_old, y_new, atol=2e-2, rtol=2e-2
                        )
                        col.check(
                            f"B OLD-vs-NEW cache {tag}",
                            sc_old,
                            sc,
                            atol=2e-2,
                            rtol=2e-2,
                        )

                        # CUDA-JIT fused vs reference (output y AND updated cache)
                        if HAVE_CUDA:
                            sc_cuda = sc0.clone()
                            y_cuda = _cuda_fused_decode(
                                x_nc,
                                w,
                                sc_cuda,
                                ci,
                                cm,
                                activation=act,
                                use_residual=use_res,
                            )
                            col.check(
                                f"CUDA-vs-ref y {tag}",
                                ref_y,
                                y_cuda,
                                atol=2e-2,
                                rtol=2e-2,
                                extra_ok=y_cuda.is_contiguous(),
                                extra_msg=(
                                    ""
                                    if y_cuda.is_contiguous()
                                    else "CUDA y not contiguous!"
                                ),
                            )
                            col.check(
                                f"CUDA-vs-ref cache {tag}",
                                ref_cache,
                                sc_cuda,
                                atol=2e-2,
                                rtol=2e-2,
                            )


def correctness_unfused() -> None:
    """C. fused vs unfused causal_conv1d(is_decode) + update_sconv_cache (Helion).

    W=4 only: the Helion update_sconv_cache is AOT-tuned for W_minus_1=3.
    Looser tol: independent Helion vs Triton fp32 accumulation. Compares valid rows.
    """
    print(
        "\n=== opt7 decode fused-vs-unfused (exercises _update_sconv_cache_helion_kernel) ==="
    )
    W = 4
    for D in (2304, 6144):
        for T in (4, 64, 256):
            for act in (None, "silu"):
                set_seed(0)
                pool = max(T * 2, 16)
                x_c, x_nc, w, sc0, ci, cm = make_decode(T, D, W, "all_valid", pool)
                has_init = cm.clone()
                qsl = torch.arange(T + 1, dtype=torch.int64, device=DEVICE)

                sc_f = sc0.clone()
                y_f = fused_causal_conv1d_update_decode(
                    x_c, w, sc_f, ci, cm, activation=act, use_residual=True
                )
                meta = precompute_helion_decode_metadata(T, W, ci, has_init)
                sc_u = sc0.clone()
                y_u = causal_conv1d(
                    x_c,
                    w,
                    sc_u,
                    **meta,
                    activation=act,
                    use_residual=True,
                    is_decode=True,
                )
                update_sconv_cache(x_c, sc_u, ci, has_init, qsl.to(torch.int32))

                valid = (ci != PAD).nonzero(as_tuple=True)[0]
                col.check(
                    f"C fused-vs-unfused y D={D} T={T} act={act}",
                    y_f[valid],
                    y_u[valid],
                    atol=0.1,
                    rtol=1e-2,
                )
                slots = ci[ci != PAD].long()
                col.check(
                    f"C fused-vs-unfused cache D={D} T={T} act={act}",
                    sc_f[slots],
                    sc_u[slots],
                    atol=0.1,
                    rtol=1e-2,
                )


def correctness_track() -> None:
    """D. decode fused with prefix-cache tracking (DO_TRACK): the post-update conv
    window must also be written to the ping-pong slot track_indices[b] wherever
    track_mask[b] & valid. Reference scatters new_win to BOTH the working slot and
    the (disjoint) track slot. Exercises the DO_TRACK path opt7's base sweep skips."""
    print(
        "\n=== opt7 decode DO_TRACK correctness (fused + prefix-cache track-copy) ==="
    )
    for D, W in [(2304, 4), (6144, 4), (384, 3)]:
        for T in (1, 4, 16, 64):
            for pat in ("all_valid", "with_pad"):
                for tmp in ("all", "alt"):
                    set_seed(0)
                    pool = max(T * 4, 32)
                    _, x_nc, w, sc0, ci, cm = make_decode(T, D, W, pat, pool)
                    # ping-pong track slots: disjoint from the working slots (ci)
                    used = torch.zeros(pool, dtype=torch.bool, device=DEVICE)
                    used[ci[ci != PAD].long()] = True
                    tidx = (~used).nonzero(as_tuple=True)[0][:T].to(torch.int64)
                    tmask = (
                        torch.ones(T, dtype=torch.bool, device=DEVICE)
                        if tmp == "all"
                        else (torch.arange(T, device=DEVICE) % 2 == 0)
                    )
                    ref_y, ref_c = reference_decode(x_nc, w, sc0, ci, cm, "silu", True)
                    W1 = W - 1
                    prefix = sc0.float()[ci.clamp(min=0).long()]
                    cmf = cm.float()
                    new_win = torch.empty(T, W1, D, dtype=sc0.dtype, device=DEVICE)
                    for iw in range(W1 - 1):
                        new_win[:, iw, :] = (prefix[:, iw + 1, :] * cmf[:, None]).to(
                            sc0.dtype
                        )
                    new_win[:, W1 - 1, :] = x_nc
                    do = (ci != PAD) & tmask
                    if do.any():
                        d = do.nonzero(as_tuple=True)[0]
                        ref_c[tidx[d].long()] = new_win[d]
                    tag = f"T={T} D={D} W={W} {pat} tmask={tmp}"
                    sc = sc0.clone()
                    y = NEW.fused_causal_conv1d_update_decode(
                        x_nc,
                        w,
                        sc,
                        ci,
                        cm,
                        activation="silu",
                        use_residual=True,
                        track_mask=tmask,
                        track_indices=tidx,
                    )
                    col.check(f"D NEW-track y {tag}", ref_y, y, atol=2e-2, rtol=2e-2)
                    col.check(
                        f"D NEW-track cache {tag}", ref_c, sc, atol=2e-2, rtol=2e-2
                    )
                    if HAVE_CUDA:
                        sc2 = sc0.clone()
                        y2 = _cuda_fused_decode(
                            x_nc,
                            w,
                            sc2,
                            ci,
                            cm,
                            activation="silu",
                            use_residual=True,
                            track_mask=tmask,
                            track_indices=tidx,
                        )
                        col.check(
                            f"D CUDA-track y {tag}", ref_y, y2, atol=2e-2, rtol=2e-2
                        )
                        col.check(
                            f"D CUDA-track cache {tag}",
                            ref_c,
                            sc2,
                            atol=2e-2,
                            rtol=2e-2,
                        )


def latency() -> None:
    print("\n=== opt7 decode latency ===")
    W = 4
    for D in (2304, 4096, 6144):
        for T in (1, 4, 16, 64, 256):
            set_seed(0)
            pool = max(T * 2, 64)
            x_c, x_nc, w, sc0, ci, cm = make_decode(T, D, W, "all_valid", pool)
            has_init = cm.clone()
            qsl = torch.arange(T + 1, dtype=torch.int32, device=DEVICE)
            meta = precompute_helion_decode_metadata(T, W, ci, has_init)
            sc_a, sc_b, sc_c = sc0.clone(), sc0.clone(), sc0.clone()
            variants = {
                "old_fused": bench(
                    lambda: OLD.fused_causal_conv1d_update_decode(
                        x_c, w, sc_a, ci, cm, activation="silu", use_residual=True
                    )
                ),
                "new_fused": bench(
                    lambda: NEW.fused_causal_conv1d_update_decode(
                        x_nc, w, sc_b, ci, cm, activation="silu", use_residual=True
                    )
                ),
                "unfused_conv+update": bench(
                    lambda: (
                        causal_conv1d(
                            x_c,
                            w,
                            sc_c,
                            **meta,
                            activation="silu",
                            use_residual=True,
                            is_decode=True,
                        ),
                        update_sconv_cache(x_c, sc_c, ci, has_init, qsl),
                    )
                ),
            }
            if HAVE_CUDA:
                sc_d = sc0.clone()
                variants["cuda_jit"] = bench(
                    lambda: _cuda_fused_decode(
                        x_nc, w, sc_d, ci, cm, activation="silu", use_residual=True
                    )
                )
            col.latency_row(
                {"T": T, "D": D}, variants, baseline_key="unfused_conv+update"
            )
            cuda_str = f"  cuda {variants['cuda_jit']:7.1f}" if HAVE_CUDA else ""
            print(
                f"  T={T:5d} D={D:5d} | old_fused {variants['old_fused']:7.1f}  "
                f"new_fused {variants['new_fused']:7.1f}  "
                f"unfused {variants['unfused_conv+update']:7.1f}{cuda_str} us"
            )


if __name__ == "__main__":
    print("=" * 72)
    print(f"{OPT}: {col.title}")
    print("=" * 72)
    correctness()
    correctness_unfused()
    correctness_track()
    latency()
    print_latency_table(
        col.latency,
        shape_keys=["T", "D"],
        variant_keys=["old_fused", "new_fused", "unfused_conv+update", "cuda_jit"],
        baseline_key="unfused_conv+update",
        title="decode conv+update latency (us) — lower is better",
    )
    col.emit()
