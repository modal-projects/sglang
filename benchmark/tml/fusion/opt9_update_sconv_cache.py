"""Opt 9 — update_sconv_cache: Helion -> Triton rewrite.

OLD (<= HEAD): Helion `_update_sconv_cache_helion_kernel` (AOT-autotuned, W_minus_1=3).
NEW (working tree): Triton `_update_sconv_cache_kernel` (drops the AOT-autotune dep,
    handles general W_minus_1 and both decode/extend query_len).

Semantics (both): for each sequence b with slot ci=cache_indices[b] and query range
[start,end) (query_start_loc), the new conv state is the last W-1 entries of the virtual
stream [ old_state (W-1, gated by has_initial_state) ++ x[start:end] ]. PAD/empty lanes
are untouched. It's a pure select/copy (no arithmetic) => old vs new must be BIT-EXACT.

OLD baseline = vendored Helion in _sconv_baseline.py. OLD runs only at W_minus_1=3
(its AOT-tuned size); NEW is also checked at W_minus_1 in {2,4} vs the reference.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sconv_baseline as OLD
from _harness import DEVICE, DTYPE, Collector, bench, print_latency_table, rand, set_seed

from sglang.tml.kernels import sconv as NEW

try:
    from sglang.jit_kernel.inkling_sconv import update_sconv_cache as _cuda_update_sconv_cache

    HAVE_CUDA = True
except Exception as _e:  # pragma: no cover - CUDA-JIT optional
    HAVE_CUDA = False
    print(f"[opt9] CUDA-JIT update_sconv_cache unavailable: {_e}")

OPT = "opt9_update_sconv_cache"
col = Collector(OPT, "update_sconv_cache: Helion -> Triton (decode + extend)")
PAD = -1


def make(B, W1, D, mode, lengths, hs_pat, pad, pool):
    """x as a non-contiguous split; distinct slots; decode (len=1) or extend (varied)."""
    if mode == "decode":
        lengths = [1] * B
    T = sum(lengths)
    qkvr = rand(T, D * 4)
    x = qkvr[:, :D]  # non-contiguous [T, D]
    sconv_cache = rand(pool, W1, D)
    qsl = torch.zeros(B + 1, dtype=torch.int32, device=DEVICE)
    qsl[1:] = torch.tensor(lengths, dtype=torch.int32, device=DEVICE).cumsum(0)
    ci = torch.randperm(pool, device=DEVICE)[:B].to(torch.int32)
    if pad and B >= 3:
        ci[1::3] = PAD
    if hs_pat == "all":
        hs = torch.ones(B, dtype=torch.bool, device=DEVICE)
    elif hs_pat == "none":
        hs = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    else:
        hs = torch.arange(B, device=DEVICE) % 2 == 0
    return x, sconv_cache, ci, hs, qsl


def reference(x, sconv_cache, ci, hs, qsl):
    B = ci.shape[0]
    W1 = sconv_cache.shape[1]
    new_cache = sconv_cache.clone()
    q = qsl.tolist()
    for b in range(B):
        slot = int(ci[b])
        start, end = q[b], q[b + 1]
        qlen = end - start
        if slot == PAD or qlen <= 0:
            continue
        old = sconv_cache[slot]
        base = old if bool(hs[b]) else torch.zeros_like(old)
        virtual = torch.cat([base, x[start:end]], dim=0)  # [W1+qlen, D]
        new_cache[slot] = virtual[qlen : qlen + W1]  # last W1
    return new_cache


def _lengths(B, W1, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # mix of qlen < W1, == W1, > W1 to exercise both state-shift and pure-x branches
    return (torch.randint(1, 2 * W1 + 2, (B,), generator=g) + 0).tolist()


def correctness() -> None:
    print("\n=== opt9 update_sconv_cache correctness ===")
    for W1 in (2, 3, 4):
        for D in (384, 2304, 6144):
            for B in (1, 4, 16, 64):
                for mode in ("decode", "extend"):
                    for hs_pat in ("all", "none", "mixed"):
                        for pad in (False, True):
                            set_seed(0)
                            pool = max(4 * B, 32)
                            lengths = _lengths(B, W1, B + W1) if mode == "extend" else None
                            x, sc0, ci, hs, qsl = make(B, W1, D, mode, lengths, hs_pat, pad, pool)
                            ref = reference(x, sc0, ci, hs, qsl)
                            tag = (f"W1={W1} D={D} B={B} {mode} hs={hs_pat}"
                                   f"{' pad' if pad else ''}")
                            # NEW (Triton) vs reference
                            sc_new = sc0.clone()
                            NEW.update_sconv_cache(x, sc_new, ci, hs, qsl)
                            col.check(f"NEW-vs-ref {tag}", ref, sc_new, atol=0, rtol=0)
                            # CUDA-JIT vs reference (bit-exact)
                            if HAVE_CUDA:
                                sc_cuda = sc0.clone()
                                _cuda_update_sconv_cache(x, sc_cuda, ci, hs, qsl)
                                col.check(f"CUDA-vs-ref {tag}", ref, sc_cuda, atol=0, rtol=0)
                            # OLD (Helion) only at its AOT size W1=3
                            if W1 == 3:
                                sc_old = sc0.clone()
                                try:
                                    OLD.update_sconv_cache(x, sc_old, ci, hs, qsl)
                                    col.check(f"OLD-vs-ref {tag}", ref, sc_old, atol=0, rtol=0)
                                    col.check(f"OLD-vs-NEW {tag}", sc_old, sc_new, atol=0, rtol=0)
                                except Exception as e:
                                    col.record_ok(f"OLD-vs-ref {tag}", False,
                                                  note=f"OLD raised {type(e).__name__}: {e}")


def latency() -> None:
    print("\n=== opt9 update_sconv_cache latency (W1=3 = model W=4) ===")
    W1 = 3
    for mode in ("decode", "extend"):
        for D in (2304, 6144):
            for B in (1, 4, 16, 64, 256):
                set_seed(0)
                pool = max(4 * B, 64)
                lengths = [8] * B if mode == "extend" else None
                x, sc0, ci, hs, qsl = make(B, W1, D, mode, lengths, "all", False, pool)
                sc_a, sc_b = sc0.clone(), sc0.clone()
                variants = {}
                try:
                    variants["old_helion"] = bench(
                        lambda: OLD.update_sconv_cache(x, sc_a, ci, hs, qsl))
                except Exception as e:
                    print(f"  OLD failed {mode} B={B} D={D}: {type(e).__name__}: {e}")
                    continue
                variants["new_triton"] = bench(
                    lambda: NEW.update_sconv_cache(x, sc_b, ci, hs, qsl))
                if HAVE_CUDA:
                    sc_c = sc0.clone()
                    variants["cuda_jit"] = bench(
                        lambda: _cuda_update_sconv_cache(x, sc_c, ci, hs, qsl))
                col.latency_row({"mode": mode, "B": B, "D": D}, variants, baseline_key="old_helion")
                cuda_str = f"  cuda_jit {variants['cuda_jit']:7.1f}" if HAVE_CUDA else ""
                print(f"  {mode:6s} B={B:4d} D={D:5d} | old_helion {variants['old_helion']:7.1f}  "
                      f"new_triton {variants['new_triton']:7.1f}{cuda_str} us")


if __name__ == "__main__":
    print("=" * 72)
    print(f"{OPT}: {col.title}")
    print("=" * 72)
    correctness()
    latency()
    print_latency_table(
        col.latency,
        shape_keys=["mode", "B", "D"],
        variant_keys=["old_helion", "new_triton", "cuda_jit"],
        baseline_key="old_helion",
        title="update_sconv_cache latency (us) — lower is better",
    )
    col.emit()
