"""Opt 1 — causal_conv1d (sconv extend prefix kernel).

OLD (commit c262556c2): Helion kernel ``_helion_causal_conv1d_fwd_with_prefix_kernel``;
    the public ``causal_conv1d`` asserts ``x.is_contiguous()`` and materialises a
    ``[B, W-1, D]`` prefix gather (sconv_cache[safe_idx]) before the kernel. The caller
    (attn.py) passed ``k.contiguous()`` / ``v.contiguous()``.
NEW (commit 1eb614dee): Triton kernel ``_causal_conv1d_fwd_with_prefix_kernel``; reads
    arbitrary-stride x directly (no .contiguous() copy), folds the prefix gather and the
    cache-mask multiply into the kernel, emits a contiguous [T, D] output.

This measures the full extend-path win and decomposes it into:
  * kernel rewrite      : new(contig)    vs old(contig)
  * copy elimination    : new(noncontig) vs new(contig)
  * end-to-end          : new(noncontig) vs old(contig)   [what the model actually sees]
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _sconv_baseline as OLD  # vendored Helion baseline @ c262556c2
from _harness import DEVICE, DTYPE, Collector, bench, print_latency_table, rand, set_seed

from sglang.tml.kernels import sconv as NEW  # current Triton impl

try:
    from sglang.jit_kernel.inkling_sconv import causal_conv1d as _cuda_causal_conv1d

    HAVE_CUDA = True
except Exception as _e:  # pragma: no cover - CUDA-JIT optional
    HAVE_CUDA = False
    print(f"[opt1] CUDA-JIT causal_conv1d unavailable: {_e}")

OPT = "opt1_causal_conv1d"
col = Collector(OPT, "causal_conv1d extend prefix: Helion->Triton + copy elimination")


def _make_noncontiguous_k(T: int, D: int, D_total: int) -> torch.Tensor:
    """k as a split-view of qkvr: row-stride D_total, col-stride 1 (non-contiguous)."""
    qkvr = rand(T, D_total)
    k = qkvr[:, :D]
    if T > 1:
        assert not k.is_contiguous()
    return k


def _inputs(T, D, W, B, has_init, pad=False):
    D_total = D * 4
    pool = max(B * 2, 16)
    k_nc = _make_noncontiguous_k(T, D, D_total)
    k_c = k_nc.contiguous()
    weight = rand(D, W)
    sconv_cache = rand(pool, W - 1, D)
    cache_indices = torch.arange(B, dtype=torch.int32, device=DEVICE)
    if pad and B >= 2:
        cache_indices[1::3] = -1  # PAD_SLOT_ID: CUDA-graph padding slots
    if has_init == "all":
        has_initial = torch.ones(B, dtype=torch.bool, device=DEVICE)
    elif has_init == "none":
        has_initial = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    else:
        has_initial = torch.arange(B, device=DEVICE) % 2 == 0
    per = T // B
    qsl = torch.arange(B + 1, dtype=torch.int64, device=DEVICE) * per
    qsl[-1] = T
    old_meta = OLD.precompute_helion_extend_metadata(
        B, T, W, cache_indices, has_initial, qsl
    )
    new_meta = NEW.precompute_helion_extend_metadata(
        B, T, W, cache_indices, has_initial, qsl
    )
    return k_c, k_nc, weight, sconv_cache, old_meta, new_meta


def reference_causal_conv1d(x, weight, sconv_cache, cache_mask, safe_idx, cu, si,
                            activation, use_residual):
    """Independent fp32 PyTorch reference (matches the documented kernel semantics).

    For packed token t in sequence s (bos=cu[s], slot=safe_idx[s]) and tap iw:
        shifted = t - (W-1) + iw
        shifted >= bos              -> tap = x[shifted]                 (in-seq history)
        shifted <  bos              -> tap = sconv_cache[slot, pp] * cache_mask[s]
        out[t] = act(sum_iw tap * weight[:, iw]) (+ x[t] if residual)
    """
    T, D = x.shape
    W = weight.shape[1]
    dev = x.device
    xf, wf, cf = x.float(), weight.float(), sconv_cache.float()
    t_idx = torch.arange(T, device=dev)
    s = si.long()
    bos = cu.long()[s]
    slot = safe_idx.long()[s]
    m = cache_mask.reshape(-1).float()[s]  # [T] per-token prefix mask
    acc = torch.zeros(T, D, device=dev, dtype=torch.float32)
    for iw in range(W):
        shifted = t_idx - (W - 1) + iw
        in_x = (shifted >= bos) & (shifted < T)
        tap_x = xf[shifted.clamp(0, T - 1)] * in_x.unsqueeze(1)
        prefix_pos = shifted - bos + (W - 1)
        in_prefix = (shifted < bos) & (prefix_pos >= 0) & (prefix_pos < (W - 1))
        tap_p = cf[slot, prefix_pos.clamp(0, W - 2)] * in_prefix.unsqueeze(1) * m.unsqueeze(1)
        acc += (tap_x + tap_p) * wf[:, iw].unsqueeze(0)
    if activation in ("silu", "swish"):
        acc = acc * torch.sigmoid(acc)
    if use_residual:
        acc += xf
    return acc.to(x.dtype)


def _check_one(T, D, W, B, act, use_res, has_init, pad, label_extra=""):
    set_seed(0)
    k_c, k_nc, w, cache, om, nm = _inputs(T, D, W, B, has_init, pad=pad)
    ref = reference_causal_conv1d(k_c, w, cache, **nm, activation=act, use_residual=use_res)
    tag = (f"T={T} D={D} W={W} B={B} act={act} res={use_res} init={has_init}"
           f"{' pad' if pad else ''}{label_extra}")
    try:
        old = OLD.causal_conv1d(k_c, w, cache, **om, activation=act, use_residual=use_res)
        col.check(f"OLD-vs-ref {tag}", ref, old, atol=2e-2, rtol=2e-2)
    except Exception as e:
        col.record_ok(f"OLD-vs-ref {tag}", False, note=f"OLD raised {type(e).__name__}: {e}")
    new = NEW.causal_conv1d(k_nc, w, cache, **nm, activation=act, use_residual=use_res)
    col.check(
        f"NEW-vs-ref {tag}", ref, new, atol=2e-2, rtol=2e-2,
        extra_ok=new.is_contiguous(),
        extra_msg="" if new.is_contiguous() else "NEW not contiguous!",
    )
    if HAVE_CUDA:
        cuda = _cuda_causal_conv1d(
            k_nc, w, cache, **nm, activation=act, use_residual=use_res
        )
        col.check(
            f"CUDA-vs-ref {tag}", ref, cuda, atol=2e-2, rtol=2e-2,
            extra_ok=cuda.is_contiguous(),
            extra_msg="" if cuda.is_contiguous() else "CUDA not contiguous!",
        )


def correctness() -> None:
    print("\n=== opt1 causal_conv1d correctness (vs independent fp32 reference) ===")
    # Main sweep: W=3 (real model size) + 4 + 5, all init states, both activations.
    for D, W in [(384, 3), (2304, 3), (2304, 4), (4096, 4), (6144, 4), (2304, 5)]:
        for T, B in [(1, 1), (16, 16), (64, 4), (512, 4), (2048, 16), (4096, 8)]:
            if T < B:
                continue
            for act in (None, "silu"):
                for use_res in (True, False):
                    for has_init in ("all", "none", "mixed"):
                        _check_one(T, D, W, B, act, use_res, has_init, pad=False)

    # PAD-slot sweep (CUDA-graph padding: cache_indices == -1, prefix must zero).
    print("  -- PAD-slot cases --")
    for D, W in [(2304, 4), (6144, 4)]:
        for T, B in [(64, 4), (512, 8), (2048, 16)]:
            for act in (None, "silu"):
                _check_one(T, D, W, B, act, True, "all", pad=True)

    # Edge case: T=0 (idle batch) must return empty without error.
    print("  -- T=0 edge --")
    set_seed(0)
    _, _, w, cache, _, nm = _inputs(4, 2304, 4, 4, "all")
    empty = torch.empty(0, 2304, dtype=DTYPE, device=DEVICE)
    try:
        y0 = NEW.causal_conv1d(empty, w, cache, **nm, activation="silu", use_residual=True)
        col.record_ok("NEW T=0 returns empty", y0.shape[0] == 0, note=f"shape={tuple(y0.shape)}")
    except Exception as e:
        col.record_ok("NEW T=0 returns empty", False, note=f"raised {type(e).__name__}: {e}")


def latency() -> None:
    print("\n=== opt1 causal_conv1d latency (silu+residual, W=4, B=4) ===")
    W, B = 4, 4
    T_values = [512, 2048, 4096, 8192, 16384]
    D_values = [2304, 4096, 6144, 8192]
    for T in T_values:
        for D in D_values:
            set_seed(0)
            k_c, k_nc, w, cache, om, nm = _inputs(T, D, W, B, "all")
            variants: dict[str, float] = {}
            try:
                variants["old_helion_contig"] = bench(
                    lambda: OLD.causal_conv1d(
                        k_c, w, cache, **om, activation="silu", use_residual=True
                    )
                )
            except Exception as e:
                print(f"  OLD failed T={T} D={D}: {type(e).__name__}: {e}")
                continue
            variants["new_triton_contig"] = bench(
                lambda: NEW.causal_conv1d(
                    k_c, w, cache, **nm, activation="silu", use_residual=True
                )
            )
            variants["new_triton_noncontig"] = bench(
                lambda: NEW.causal_conv1d(
                    k_nc, w, cache, **nm, activation="silu", use_residual=True
                )
            )
            if HAVE_CUDA:
                variants["cuda_jit_noncontig"] = bench(
                    lambda: _cuda_causal_conv1d(
                        k_nc, w, cache, **nm, activation="silu", use_residual=True
                    )
                )
            # Baseline ratios on new_triton_noncontig (the correct production kernel).
            # old_helion_contig fails correctness on every multi-token shape here (it is
            # AOT-autotuned for a single fixed B=1/T=8192 shape), so its ~55us flat latency
            # is invalid and must NOT be used as a baseline; it is kept only as a raw column.
            col.latency_row(
                {"T": T, "D": D}, variants, baseline_key="new_triton_noncontig"
            )
            cuda_str = (
                f"  cuda {variants['cuda_jit_noncontig']:8.1f}" if HAVE_CUDA else ""
            )
            print(
                f"  T={T:6d} D={D:5d} | old {variants['old_helion_contig']:8.1f}  "
                f"new_c {variants['new_triton_contig']:8.1f}  "
                f"new_nc {variants['new_triton_noncontig']:8.1f}{cuda_str} us"
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
        variant_keys=[
            "old_helion_contig",
            "new_triton_contig",
            "new_triton_noncontig",
            "cuda_jit_noncontig",
        ],
        baseline_key="new_triton_noncontig",
        title="causal_conv1d extend latency (us) — lower is better",
    )
    col.emit()
