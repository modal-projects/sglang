"""Comprehensive correctness tests and benchmarks for Inkling kernel fusion optimizations.

Tests three optimizations:
  1. sconv kernels (causal_conv1d for extend, fused_causal_conv1d_update_decode for decode)
     accept non-contiguous inputs directly and always emit a contiguous output.
  2. RMSNorm (layers/norm.py) uses sgl_kernel.rmsnorm with x.reshape(-1, hidden_size)
     instead of .contiguous().view().
  3. Fused gather->scatter (fused_gather_scatter_to_sconv_cache) replaces
     hidden_states[track_conv_indices].contiguous() + copy_if_needed.

Run on the container:
  ssh di1
  docker exec -it sgl_cheng python /sgl-workspace/sglang/benchmark/tml/bench_kernel_fusion.py

Output:
  PASS / FAIL for each correctness case.
  Aligned ASCII benchmark tables.
  Summary: total tests, failure count.
"""

from __future__ import annotations

import time
from typing import Callable

import torch

# --------------------------------------------------------------------------- #
# Globals
# --------------------------------------------------------------------------- #

DTYPE = torch.bfloat16
DEVICE = "cuda"

# Accumulate (label, passed) for the final summary.
_results: list[tuple[str, bool]] = []


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _rand(*shape, dtype=DTYPE, device=DEVICE) -> torch.Tensor:
    return torch.randn(*shape, dtype=dtype, device=device)


def _randbool(*shape, device=DEVICE) -> torch.Tensor:
    return torch.randint(0, 2, shape, dtype=torch.bool, device=device)


def bench(fn: Callable, *, warmup: int = 50, reps: int = 200) -> float:
    """Return median execution time in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1e6


def _record(label: str, ok: bool) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {label}")
    _results.append((label, ok))


def _make_noncontiguous_x(T: int, D: int, D_total: int) -> torch.Tensor:
    """Simulate k/v as a non-contiguous split-view (stride-D_total on dim 0).

    Note: PyTorch skips size-1 dimensions in is_contiguous(), so a [1, D]
    tensor is always considered contiguous regardless of stride.  For T>1
    we assert the expected non-contiguous property.
    """
    qkvr = _rand(T, D_total)
    k = qkvr[:, :D]  # strides (D_total, 1)
    if T > 1:
        assert (
            not k.is_contiguous()
        ), f"Expected non-contiguous tensor, got strides {k.stride()}"
    assert k.stride(0) == D_total, "Row stride should equal D_total"
    return k


def _make_query_start_loc(T: int, B: int) -> torch.Tensor:
    """Equal-length sequences: B sequences of T//B tokens each."""
    per_seq = T // B
    locs = torch.arange(B + 1, dtype=torch.int64, device=DEVICE) * per_seq
    locs[-1] = T  # ensure exact total
    return locs


# --------------------------------------------------------------------------- #
# Section 1 – sconv extend: causal_conv1d
# --------------------------------------------------------------------------- #


def _run_causal_conv1d_case(
    T: int,
    D: int,
    W: int,
    B: int,
    activation: str | None,
    use_residual: bool,
    has_initial_all: bool | None,  # None -> mixed (alternating)
) -> None:
    """Single correctness case for causal_conv1d (extend path)."""
    from sglang.srt.models.inkling_common.kernels.sconv import (
        causal_conv1d,
        precompute_helion_extend_metadata,
    )

    torch.manual_seed(0)
    D_total = D * 4
    pool_size = max(B * 2, 16)

    k_nc = _make_noncontiguous_x(T, D, D_total)
    k_c = k_nc.contiguous()

    weight = _rand(D, W)
    sconv_cache = _rand(pool_size, W - 1, D)
    cache_indices = torch.arange(B, dtype=torch.int32, device=DEVICE)

    if has_initial_all is True:
        has_initial_state = torch.ones(B, dtype=torch.bool, device=DEVICE)
    elif has_initial_all is False:
        has_initial_state = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    else:
        # mixed: alternating True/False
        has_initial_state = torch.arange(B, device=DEVICE) % 2 == 0

    query_start_loc = _make_query_start_loc(T, B)

    meta = precompute_helion_extend_metadata(
        B, T, W, cache_indices, has_initial_state, query_start_loc
    )

    ref = causal_conv1d(
        k_c,
        weight,
        sconv_cache,
        cache_mask=meta["cache_mask"],
        safe_idx=meta["safe_idx"],
        cu=meta["cu"],
        si=meta["si"],
        activation=activation,
        use_residual=use_residual,
    )
    opt = causal_conv1d(
        k_nc,
        weight,
        sconv_cache,
        cache_mask=meta["cache_mask"],
        safe_idx=meta["safe_idx"],
        cu=meta["cu"],
        si=meta["si"],
        activation=activation,
        use_residual=use_residual,
    )

    label = (
        f"sconv_extend T={T} D={D} W={W} B={B} "
        f"act={activation} resid={use_residual} init={has_initial_all}"
    )
    ok_val = torch.allclose(ref, opt, atol=1e-3, rtol=1e-3)
    ok_contig = opt.is_contiguous()
    if not ok_val:
        diff = (ref - opt).abs()
        print(
            f"    max_diff={diff.max().item():.4e}  mean_diff={diff.mean().item():.4e}"
        )
    if not ok_contig:
        print(f"    output is NOT contiguous: strides={opt.stride()}")
    _record(label, ok_val and ok_contig)


def test_causal_conv1d_sweep() -> None:
    print("\n=== 1. sconv extend (causal_conv1d) correctness sweep ===")

    D_W_pairs = [
        (384, 3),
        (2304, 3),
        (4032, 4),
        (6144, 5),
        (384, 4),
        (2304, 5),
    ]
    activations: list[str | None] = [None, "silu"]
    use_residuals = [True, False]
    has_initial_options = [True, False, None]  # None = mixed
    T_values = [1, 64, 512, 2048]
    B_values = [1, 4, 32]

    for D, W in D_W_pairs:
        for T in T_values:
            for B in B_values:
                # Skip T < B (can't evenly partition)
                if T < B:
                    continue
                for act in activations:
                    for use_res in use_residuals:
                        for has_init in has_initial_options:
                            _run_causal_conv1d_case(
                                T=T,
                                D=D,
                                W=W,
                                B=B,
                                activation=act,
                                use_residual=use_res,
                                has_initial_all=has_init,
                            )


# --------------------------------------------------------------------------- #
# Section 2 – sconv decode: fused_causal_conv1d_update_decode
# --------------------------------------------------------------------------- #


def _run_fused_decode_case(
    T: int,
    D: int,
    W: int,
    activation: str | None,
    use_residual: bool,
    mask_pattern: str,
) -> None:
    """Single correctness case for fused decode kernel.

    Cross-checks:
      (a) non-contiguous input == contiguous input (output + cache)
      (b) fused decode vs causal_conv1d + update_sconv_cache reference
    """
    from sglang.srt.models.inkling_common.kernels.sconv import (
        causal_conv1d,
        fused_causal_conv1d_update_decode,
        precompute_helion_decode_metadata,
        update_sconv_cache,
    )

    torch.manual_seed(0)
    D_total = D * 4
    pool_size = max(T * 2, 16)

    k_nc = _make_noncontiguous_x(T, D, D_total)
    k_c = k_nc.contiguous()

    weight = _rand(D, W)
    sconv_cache_base = _rand(pool_size, W - 1, D)
    cache_indices = torch.arange(T, dtype=torch.int32, device=DEVICE)

    if mask_pattern == "all_valid":
        cache_mask = torch.ones(T, dtype=torch.bool, device=DEVICE)
    elif mask_pattern == "half_valid":
        cache_mask = torch.arange(T, device=DEVICE) % 2 == 0
    else:  # none_valid
        cache_mask = torch.zeros(T, dtype=torch.bool, device=DEVICE)

    # --- (a) non-contiguous == contiguous ---
    sc_ref = sconv_cache_base.clone()
    sc_opt = sconv_cache_base.clone()
    out_ref = fused_causal_conv1d_update_decode(
        k_c,
        weight,
        sc_ref,
        cache_indices,
        cache_mask,
        activation=activation,
        use_residual=use_residual,
    )
    out_opt = fused_causal_conv1d_update_decode(
        k_nc,
        weight,
        sc_opt,
        cache_indices,
        cache_mask,
        activation=activation,
        use_residual=use_residual,
    )

    ok_val = torch.allclose(out_ref, out_opt, atol=1e-3, rtol=1e-3)
    ok_cache = torch.allclose(sc_ref, sc_opt, atol=1e-3, rtol=1e-3)
    ok_contig = out_opt.is_contiguous()

    if not ok_val:
        diff = (out_ref - out_opt).abs()
        print(
            f"    output max_diff={diff.max().item():.4e}  mean={diff.mean().item():.4e}"
        )
    if not ok_cache:
        diff = (sc_ref - sc_opt).abs()
        print(
            f"    cache max_diff={diff.max().item():.4e}  mean={diff.mean().item():.4e}"
        )

    label_a = (
        f"sconv_decode[nc==c] T={T} D={D} W={W} "
        f"act={activation} resid={use_residual} mask={mask_pattern}"
    )
    _record(label_a, ok_val and ok_cache and ok_contig)

    # --- (b) fused vs causal_conv1d + update_sconv_cache ---
    has_initial_state = cache_mask.clone()
    query_start_loc = torch.arange(T + 1, dtype=torch.int64, device=DEVICE)
    sc_fused = sconv_cache_base.clone()
    sc_ref2 = sconv_cache_base.clone()

    out_fused = fused_causal_conv1d_update_decode(
        k_c,
        weight,
        sc_fused,
        cache_indices,
        cache_mask,
        activation=activation,
        use_residual=use_residual,
    )

    meta = precompute_helion_decode_metadata(T, W, cache_indices, has_initial_state)
    out_ref2 = causal_conv1d(
        k_c,
        weight,
        sc_ref2,
        cache_mask=meta["cache_mask"],
        safe_idx=meta["safe_idx"],
        cu=meta["cu"],
        si=meta["si"],
        activation=activation,
        use_residual=use_residual,
        is_decode=True,
    )
    update_sconv_cache(
        k_c, sc_ref2, cache_indices, has_initial_state, query_start_loc.to(torch.int32)
    )

    # causal_conv1d with is_decode=True intentionally skips cache masking for PAD
    # slots (outputs are never consumed).  Only compare rows where cache_mask=True.
    #
    # atol=0.1: Triton and Helion accumulate the W-tap conv in float32 with different
    # loop-unroll orderings.  For SiLU+residual at large D, this produces up to ~1
    # bfloat16 ULP of rounding difference on O(1) elements out of millions — pure
    # floating-point non-determinism between two independent kernel implementations.
    # The nc==c cross-check (which tests our actual optimization) uses atol=1e-3.
    valid_mask = cache_mask  # [T] bool
    if valid_mask.any():
        ok_fused_val = torch.allclose(
            out_fused[valid_mask], out_ref2[valid_mask], atol=0.1, rtol=1e-2
        )
    else:
        ok_fused_val = True  # nothing to compare

    # Cache: only compare slots actually written (valid cache_indices, mask True)
    valid_slots = cache_indices[valid_mask].long()
    if valid_slots.numel() > 0:
        ok_fused_cache = torch.allclose(
            sc_fused[valid_slots], sc_ref2[valid_slots], atol=0.1, rtol=1e-2
        )
    else:
        ok_fused_cache = True

    if not ok_fused_val:
        diff = (out_fused[valid_mask] - out_ref2[valid_mask]).abs()
        print(
            f"    fused vs ref output max_diff={diff.max().item():.4e}  "
            f"mean={diff.mean().item():.4e}"
        )
    if not ok_fused_cache:
        diff = (sc_fused[valid_slots] - sc_ref2[valid_slots]).abs()
        print(
            f"    fused vs ref cache max_diff={diff.max().item():.4e}  "
            f"mean={diff.mean().item():.4e}"
        )

    label_b = (
        f"sconv_decode[fused==ref] T={T} D={D} W={W} "
        f"act={activation} resid={use_residual} mask={mask_pattern}"
    )
    _record(label_b, ok_fused_val and ok_fused_cache)


def test_fused_decode_sweep() -> None:
    print(
        "\n=== 2. sconv decode (fused_causal_conv1d_update_decode) correctness sweep ==="
    )

    # W=5 (W_minus_1=4) is excluded: update_sconv_cache (Helion) is AOT-tuned for
    # W_minus_1=3 only, and JIT compilation fails for W_minus_1=4.
    # The actual model uses sconv_kernel_size=4 (W_minus_1=3).
    D_W_pairs = [
        (384, 3),
        (2304, 3),
        (4032, 4),
        (6144, 4),
        (384, 4),
        (2304, 3),
    ]
    activations: list[str | None] = [None, "silu"]
    use_residuals = [True, False]
    T_values = [1, 4, 64, 512, 2048]
    mask_patterns = ["all_valid", "half_valid", "none_valid"]

    for D, W in D_W_pairs:
        for T in T_values:
            for act in activations:
                for use_res in use_residuals:
                    for mask_pat in mask_patterns:
                        _run_fused_decode_case(
                            T=T,
                            D=D,
                            W=W,
                            activation=act,
                            use_residual=use_res,
                            mask_pattern=mask_pat,
                        )


# --------------------------------------------------------------------------- #
# Section 3 – RMSNorm
# --------------------------------------------------------------------------- #


def _run_rmsnorm_case(
    T: int,
    num_heads: int,
    head_dim: int,
    from_split: bool,
) -> None:
    """Single correctness case for RMSNorm.

    from_split=True simulates the qkvr non-contiguous path.
    from_split=False uses a plain contiguous 2-D input.
    """
    from sglang.srt.models.inkling_common.norm import RMSNorm

    torch.manual_seed(0)
    hidden_size = head_dim
    norm = RMSNorm(hidden_size=hidden_size).to(DEVICE).to(DTYPE)
    # Use random (non-unit) weight for sensitivity
    norm.weight = torch.nn.Parameter(
        torch.rand(hidden_size, device=DEVICE, dtype=DTYPE) * 0.5 + 0.5
    )

    if from_split:
        # Simulate qkvr projection output; k is a non-contiguous view
        D_total = head_dim * (num_heads * 3 + 2)
        qkvr = _rand(T, D_total)
        q = qkvr[:, : head_dim * num_heads]  # [T, q_size], strides (D_total, 1)
        x = q.view(
            T, num_heads, head_dim
        )  # [T, H, head_dim], strides (D_total, head_dim, 1)
        assert not x.is_contiguous()
    else:
        x = _rand(T, num_heads, head_dim)  # fully contiguous

    # Float32 reference via torch.nn.functional.rms_norm
    x_f32 = x.float()
    w_f32 = norm.weight.float()
    ref = torch.nn.functional.rms_norm(
        x_f32, (hidden_size,), w_f32, norm.variance_epsilon
    ).to(DTYPE)

    opt = norm(x)

    ok_val = torch.allclose(ref, opt, atol=2e-2)
    ok_shape = opt.shape == x.shape
    ok_contig = opt.is_contiguous()

    if not ok_val:
        diff = (ref - opt).abs()
        print(
            f"    max_diff={diff.max().item():.4e}  mean_diff={diff.mean().item():.4e}"
        )
    if not ok_shape:
        print(f"    shape mismatch: got {opt.shape}, expected {x.shape}")

    path = "from_split" if from_split else "contiguous"
    label = f"rmsnorm T={T} H={num_heads} D={head_dim} [{path}]"
    _record(label, ok_val and ok_shape and ok_contig)


def test_rmsnorm_sweep() -> None:
    print("\n=== 3. RMSNorm correctness sweep ===")

    # (T, num_heads, head_dim)
    configs = [
        (64, 8, 128),
        (512, 32, 128),
        (2048, 16, 64),
        (4096, 64, 96),
    ]
    for T, H, D in configs:
        for from_split in [True, False]:
            _run_rmsnorm_case(T=T, num_heads=H, head_dim=D, from_split=from_split)


# --------------------------------------------------------------------------- #
# Section 4 – Fused gather->scatter
# --------------------------------------------------------------------------- #


def _baseline_gather_copy(
    hidden_states: torch.Tensor,
    sconv_cache: torch.Tensor,
    track_conv_indices: torch.Tensor,
    mask: torch.Tensor,
    dst_indices: torch.Tensor,
) -> None:
    """Reference: eager gather + copy_if_needed."""
    from sglang.srt.models.inkling_common.sconv import copy_if_needed

    B = mask.shape[0]
    track_hidden_state = hidden_states[track_conv_indices].contiguous()  # [B, W-1, D]
    copy_if_needed(
        src_tensor=track_hidden_state,
        dst_tensor=sconv_cache,
        mask=mask,
        src_indices=torch.arange(B, device=DEVICE, dtype=torch.int64),
        dst_indices=dst_indices,
        batch_size=B,
    )


def _run_fused_gather_scatter_case(
    T: int,
    D: int,
    W: int,
    B: int,
    mask_pattern: str,
) -> None:
    from sglang.srt.models.inkling_common.sconv import (
        fused_gather_scatter_to_sconv_cache,
    )

    torch.manual_seed(0)
    W_minus_1 = W - 1
    pool = max(B * 4, 16)

    hidden_states = _rand(T, D)
    sconv_cache_ref = _rand(pool, W_minus_1, D)
    sconv_cache_fused = sconv_cache_ref.clone()
    sconv_cache_untouched = sconv_cache_ref.clone()

    # track_conv_indices: [B, W-1], each row is W-1 consecutive positions
    base = torch.randint(0, max(T - W_minus_1, 1), (B,), device=DEVICE)
    track_conv_indices = (
        (base.unsqueeze(1) + torch.arange(W_minus_1, device=DEVICE))
        .clamp(max=T - 1)
        .to(torch.int32)
    )

    if mask_pattern == "all_true":
        mask = torch.ones(B, dtype=torch.bool, device=DEVICE)
    elif mask_pattern == "all_false":
        mask = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    elif mask_pattern == "alternating":
        mask = torch.arange(B, device=DEVICE) % 2 == 0
    else:  # random_50pct
        mask = _randbool(B)

    dst_indices = torch.randperm(pool, device=DEVICE)[:B].to(torch.int64)

    _baseline_gather_copy(
        hidden_states, sconv_cache_ref, track_conv_indices, mask, dst_indices
    )
    fused_gather_scatter_to_sconv_cache(
        hidden_states=hidden_states,
        sconv_cache=sconv_cache_fused,
        track_conv_indices=track_conv_indices,
        mask=mask,
        dst_indices=dst_indices,
    )

    ok_match = torch.allclose(sconv_cache_ref, sconv_cache_fused, atol=1e-5)
    if not ok_match:
        diff = (sconv_cache_ref - sconv_cache_fused).abs()
        print(f"    cache max_diff={diff.max().item():.4e}")

    # Verify mask=False entries are NOT written (compare against untouched clone)
    ok_untouched = True
    for b in range(B):
        if not mask[b].item():
            slot = dst_indices[b].item()
            if not torch.equal(sconv_cache_fused[slot], sconv_cache_untouched[slot]):
                ok_untouched = False
                print(f"    mask=False entry b={b} slot={slot} was modified!")
                break

    label = f"fused_gather_scatter D={D} W={W} B={B} mask={mask_pattern}"
    _record(label, ok_match and ok_untouched)


def test_fused_gather_scatter_sweep() -> None:
    print("\n=== 4. Fused gather->scatter correctness sweep ===")

    D_values = [384, 2304, 4032, 6144]
    W_values = [3, 4, 5]
    B_values = [1, 8, 64]
    mask_patterns = ["all_true", "all_false", "alternating", "random_50pct"]
    T = 4096  # plenty of source tokens

    for D in D_values:
        for W in W_values:
            for B in B_values:
                for mask_pat in mask_patterns:
                    _run_fused_gather_scatter_case(
                        T=T, D=D, W=W, B=B, mask_pattern=mask_pat
                    )


# --------------------------------------------------------------------------- #
# Benchmark helpers
# --------------------------------------------------------------------------- #


def _print_table(
    title: str,
    row_header: str,
    col_header: str,
    row_labels: list,
    col_labels: list,
    data: list[list[str]],
    extra_cols: list[str] | None = None,
    extra_data: list[list[str]] | None = None,
) -> None:
    """Print a readable ASCII table.

    data[r][c] corresponds to row_labels[r], col_labels[c].
    extra_cols/extra_data add trailing columns after the grid.
    """
    # Build full column list
    all_col_labels = [f"{col_header}={c}" for c in col_labels]
    if extra_cols:
        all_col_labels += extra_cols

    # Build full data
    all_data: list[list[str]] = []
    for r, row in enumerate(data):
        full_row = list(row)
        if extra_data:
            full_row += extra_data[r]
        all_data.append(full_row)

    # Column widths
    label_w = max(len(str(row_header)), max(len(str(l)) for l in row_labels)) + 2
    col_ws = [
        max(len(lbl), max(len(all_data[r][c]) for r in range(len(all_data)))) + 2
        for c, lbl in enumerate(all_col_labels)
    ]

    sep = "+" + "-" * label_w + "+" + "+".join("-" * w for w in col_ws) + "+"
    header_row = (
        "|"
        + f" {row_header} ".ljust(label_w)
        + "|"
        + "|".join(f" {lbl} ".ljust(w) for lbl, w in zip(all_col_labels, col_ws))
        + "|"
    )

    print(f"\n{title}")
    print(sep)
    print(header_row)
    print(sep)
    for r, rl in enumerate(row_labels):
        row_str = (
            "|"
            + f" {rl} ".ljust(label_w)
            + "|"
            + "|".join(
                f" {all_data[r][c]} ".ljust(col_ws[c])
                for c in range(len(all_col_labels))
            )
            + "|"
        )
        print(row_str)
    print(sep)


# --------------------------------------------------------------------------- #
# Benchmark 1 – sconv extend
# --------------------------------------------------------------------------- #


def bench_causal_conv1d() -> None:
    from sglang.srt.models.inkling_common.kernels.sconv import (
        causal_conv1d,
        precompute_helion_extend_metadata,
    )

    print("\n=== Benchmark: sconv extend (causal_conv1d) ===")

    T_values = [128, 512, 2048, 4096]
    D_values = [384, 2304, 4032, 6144]
    W, B = 4, 4
    pool_size = B * 2

    # Tables: direct (non-contiguous), copy-then-conv baseline
    table_direct: list[list[str]] = []
    table_copy: list[list[str]] = []

    for T in T_values:
        row_direct: list[str] = []
        row_copy: list[str] = []
        for D in D_values:
            D_total = D * 4
            k_nc = _make_noncontiguous_x(T, D, D_total)
            weight = _rand(D, W)
            sconv_cache = _rand(pool_size, W - 1, D)
            cache_indices = torch.arange(B, dtype=torch.int32, device=DEVICE)
            has_initial_state = torch.ones(B, dtype=torch.bool, device=DEVICE)
            query_start_loc = _make_query_start_loc(T, B)
            meta = precompute_helion_extend_metadata(
                B, T, W, cache_indices, has_initial_state, query_start_loc
            )

            def _fn_direct(x=k_nc, w=weight, sc=sconv_cache, m=meta) -> None:
                causal_conv1d(
                    x,
                    w,
                    sc,
                    cache_mask=m["cache_mask"],
                    safe_idx=m["safe_idx"],
                    cu=m["cu"],
                    si=m["si"],
                )

            def _fn_copy(x=k_nc, w=weight, sc=sconv_cache, m=meta) -> None:
                causal_conv1d(
                    x.contiguous(),
                    w,
                    sc,
                    cache_mask=m["cache_mask"],
                    safe_idx=m["safe_idx"],
                    cu=m["cu"],
                    si=m["si"],
                )

            us_direct = bench(_fn_direct)
            us_copy = bench(_fn_copy)
            row_direct.append(f"{us_direct:.1f}")
            row_copy.append(f"{us_copy:.1f}")

        table_direct.append(row_direct)
        table_copy.append(row_copy)

    _print_table(
        "sconv extend – non-contiguous direct (µs)",
        "T",
        "D",
        T_values,
        D_values,
        table_direct,
    )
    _print_table(
        "sconv extend – copy-then-conv baseline (µs)",
        "T",
        "D",
        T_values,
        D_values,
        table_copy,
    )


# --------------------------------------------------------------------------- #
# Benchmark 2 – sconv decode
# --------------------------------------------------------------------------- #


def bench_fused_decode() -> None:
    from sglang.srt.models.inkling_common.kernels.sconv import (
        fused_causal_conv1d_update_decode,
    )

    print("\n=== Benchmark: sconv decode (fused_causal_conv1d_update_decode) ===")

    T_values = [64, 256, 1024, 4096]
    D_values = [384, 2304, 4032, 6144]
    W = 4

    table_nc: list[list[str]] = []
    table_c: list[list[str]] = []

    for T in T_values:
        row_nc: list[str] = []
        row_c: list[str] = []
        for D in D_values:
            D_total = D * 4
            pool_size = T * 2
            k_nc = _make_noncontiguous_x(T, D, D_total)
            k_c = k_nc.contiguous()
            weight = _rand(D, W)
            sconv_cache = _rand(pool_size, W - 1, D)
            cache_indices = torch.arange(T, dtype=torch.int32, device=DEVICE)
            cache_mask = torch.ones(T, dtype=torch.bool, device=DEVICE)

            def _fn_nc(
                x=k_nc, w=weight, sc=sconv_cache, ci=cache_indices, cm=cache_mask
            ) -> None:
                fused_causal_conv1d_update_decode(x, w, sc, ci, cm)

            def _fn_c(
                x=k_c, w=weight, sc=sconv_cache, ci=cache_indices, cm=cache_mask
            ) -> None:
                fused_causal_conv1d_update_decode(x, w, sc, ci, cm)

            us_nc = bench(_fn_nc)
            us_c = bench(_fn_c)
            row_nc.append(f"{us_nc:.1f}")
            row_c.append(f"{us_c:.1f}")

        table_nc.append(row_nc)
        table_c.append(row_c)

    _print_table(
        "sconv decode – non-contiguous direct (µs)",
        "T",
        "D",
        T_values,
        D_values,
        table_nc,
    )
    _print_table(
        "sconv decode – contiguous input (µs)",
        "T",
        "D",
        T_values,
        D_values,
        table_c,
    )


# --------------------------------------------------------------------------- #
# Benchmark 3 – RMSNorm
# --------------------------------------------------------------------------- #


def bench_rmsnorm() -> None:
    from sglang.srt.models.inkling_common.norm import RMSNorm

    print("\n=== Benchmark: RMSNorm (reshape vs contiguous+view) ===")

    # (T*H, head_dim) combos to explore total sequence * heads product
    TH_values = [512, 4096, 16384, 65536]
    D_values = [64, 128, 256]

    table_reshape: list[list[str]] = []
    table_contiguous_view: list[list[str]] = []

    for TH in TH_values:
        row_reshape: list[str] = []
        row_cv: list[str] = []
        for D in D_values:
            # T*H tokens of head_dim D
            norm = RMSNorm(hidden_size=D).to(DEVICE).to(DTYPE)

            # Non-contiguous 3-D tensor: stride-D_total along dim 0
            D_total = D * 4
            qkvr = _rand(TH, D_total)
            x_nc = qkvr[:, :D].view(TH, 1, D)  # [TH, 1, D], strides (D_total, D, 1)
            assert not x_nc.is_contiguous()

            # reshape path (production code path in norm.forward via x.reshape)
            def _fn_reshape(x=x_nc, n=norm) -> None:
                n(x)

            # legacy path: .contiguous().view() before the reshape inside norm
            def _fn_cv(x=x_nc, n=norm) -> None:
                # Simulate the old wrapper: caller does .contiguous().view(-1, D)
                # then calls rmsnorm, then view back.
                from sgl_kernel import rmsnorm as sgl_rmsnorm

                x2 = x.contiguous().view(-1, D)
                sgl_rmsnorm(x2, n.weight.to(x2.dtype), n.variance_epsilon)

            us_reshape = bench(_fn_reshape)
            us_cv = bench(_fn_cv)
            row_reshape.append(f"{us_reshape:.1f}")
            row_cv.append(f"{us_cv:.1f}")

        table_reshape.append(row_reshape)
        table_contiguous_view.append(row_cv)

    _print_table(
        "RMSNorm – reshape path (production) (µs)",
        "T*H",
        "head_dim",
        TH_values,
        D_values,
        table_reshape,
    )
    _print_table(
        "RMSNorm – contiguous+view legacy path (µs)",
        "T*H",
        "head_dim",
        TH_values,
        D_values,
        table_contiguous_view,
    )


# --------------------------------------------------------------------------- #
# Benchmark 4 – fused gather->scatter
# --------------------------------------------------------------------------- #


def bench_fused_gather_scatter() -> None:
    from sglang.srt.models.inkling_common.sconv import (
        fused_gather_scatter_to_sconv_cache,
    )

    print("\n=== Benchmark: fused gather->scatter ===")

    # Representative (B, W, D) combos
    configs = [
        (8, 3, 384),
        (8, 4, 2304),
        (8, 5, 4032),
        (64, 3, 384),
        (64, 4, 2304),
        (64, 5, 6144),
        (256, 4, 2304),
        (256, 4, 6144),
    ]
    T = 4096
    pool = 1024

    labels: list[str] = []
    us_fused_col: list[str] = []
    us_baseline_col: list[str] = []
    speedup_col: list[str] = []

    for B, W, D in configs:
        W_minus_1 = W - 1
        hidden_states = _rand(T, D)
        sconv_cache = _rand(pool, W_minus_1, D)
        base = torch.randint(0, max(T - W_minus_1, 1), (B,), device=DEVICE)
        track_conv_indices = (
            (base.unsqueeze(1) + torch.arange(W_minus_1, device=DEVICE))
            .clamp(max=T - 1)
            .to(torch.int32)
        )
        mask = torch.ones(B, dtype=torch.bool, device=DEVICE)
        dst_indices = torch.arange(B, device=DEVICE, dtype=torch.int64)

        def _fn_fused(
            hs=hidden_states,
            sc=sconv_cache,
            ti=track_conv_indices,
            m=mask,
            di=dst_indices,
        ) -> None:
            fused_gather_scatter_to_sconv_cache(hs, sc, ti, m, di)

        def _fn_baseline(
            hs=hidden_states,
            sc=sconv_cache,
            ti=track_conv_indices,
            m=mask,
            di=dst_indices,
        ) -> None:
            _baseline_gather_copy(hs, sc, ti, m, di)

        us_f = bench(_fn_fused)
        us_b = bench(_fn_baseline)
        sp = us_b / us_f if us_f > 0 else float("inf")

        labels.append(f"B={B} W={W} D={D}")
        us_fused_col.append(f"{us_f:.1f}")
        us_baseline_col.append(f"{us_b:.1f}")
        speedup_col.append(f"{sp:.2f}x")

    # Print single-column table with 3 data columns
    label_w = max(len("Config"), max(len(l) for l in labels)) + 2
    col_w = 14

    sep = f"+{'-' * label_w}+{'-' * col_w}+{'-' * col_w}+{'-' * col_w}+"
    hdr = (
        f"| {'Config'.ljust(label_w - 1)}"
        f"| {'fused (µs)'.ljust(col_w - 1)}"
        f"| {'baseline (µs)'.ljust(col_w - 1)}"
        f"| {'speedup'.ljust(col_w - 1)}|"
    )
    print(f"\nfused gather->scatter benchmark")
    print(sep)
    print(hdr)
    print(sep)
    for i, lbl in enumerate(labels):
        row = (
            f"| {lbl.ljust(label_w - 1)}"
            f"| {us_fused_col[i].ljust(col_w - 1)}"
            f"| {us_baseline_col[i].ljust(col_w - 1)}"
            f"| {speedup_col[i].ljust(col_w - 1)}|"
        )
        print(row)
    print(sep)


# --------------------------------------------------------------------------- #
# Section 5 – fused_draft_extend_sconv_cache
# --------------------------------------------------------------------------- #


def _baseline_draft_extend(
    hidden_states: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    draft_token_num: int,
    do_tracking: bool = False,
    crossed: torch.Tensor | None = None,
    track_step: torch.Tensor | None = None,
    mamba_track_indices: torch.Tensor | None = None,
) -> None:
    """Reference: original gather+unfold+copy_if_needed logic."""
    from sglang.srt.models.inkling_common.sconv import copy_if_needed

    B = cache_indices.shape[0]
    W_minus_1 = sconv_cache.shape[1]

    initial_state = sconv_cache[cache_indices]
    input_reshaped = hidden_states.view(B, draft_token_num, -1)
    padded = torch.cat([initial_state, input_reshaped], dim=1)
    windows = padded.unfold(1, W_minus_1, 1)

    if (
        do_tracking
        and crossed is not None
        and track_step is not None
        and mamba_track_indices is not None
    ):
        track_idx = (
            track_step.long().view(-1, 1, 1, 1).expand(-1, 1, *windows.shape[2:])
        )
        track_selected = (
            windows.gather(1, track_idx).squeeze(1).transpose(1, 2).contiguous()
        )
        src_indices = torch.arange(B, device=sconv_cache.device, dtype=torch.int64)
        copy_if_needed(
            src_tensor=track_selected,
            dst_tensor=sconv_cache,
            mask=crossed,
            src_indices=src_indices,
            dst_indices=mamba_track_indices,
            batch_size=B,
        )

    idx = num_accepted_tokens.long().view(-1, 1, 1, 1).expand(-1, 1, *windows.shape[2:])
    selected = windows.gather(1, idx).squeeze(1).transpose(1, 2).contiguous()
    sconv_cache[cache_indices] = selected


def _run_draft_extend_case(
    B: int,
    T: int,
    D: int,
    W: int,
    do_tracking: bool,
    all_crossed: bool,
) -> None:
    from sglang.srt.models.inkling_common.sconv import fused_draft_extend_sconv_cache

    torch.manual_seed(0)
    D_total = D * 4  # simulate non-contiguous hidden_states from qkvr split
    pool = max(B * 4, 32)
    W_minus_1 = W - 1

    qkvr = _rand(B * T, D_total)
    hidden_nc = qkvr[:, :D]  # non-contiguous if B*T > 1
    # Also test with 2D hidden_states layout
    hidden_c = hidden_nc.contiguous()

    sconv_cache_ref = _rand(pool, W_minus_1, D)
    sconv_cache_fused = sconv_cache_ref.clone()

    cache_indices = torch.arange(B, dtype=torch.int32, device=DEVICE)
    num_accepted = torch.randint(0, T + 1, (B,), dtype=torch.int32, device=DEVICE)

    crossed = track_step = mamba_track_indices = None
    if do_tracking:
        crossed = (
            torch.ones(B, dtype=torch.bool, device=DEVICE)
            if all_crossed
            else _randbool(B)
        )
        track_step = torch.randint(0, T + 1, (B,), dtype=torch.int32, device=DEVICE)
        mamba_track_indices = (
            torch.arange(B, dtype=torch.int64, device=DEVICE) + B
        )  # distinct from cache_indices

    _baseline_draft_extend(
        hidden_c,
        sconv_cache_ref,
        cache_indices,
        num_accepted,
        T,
        do_tracking,
        crossed,
        track_step,
        mamba_track_indices,
    )
    fused_draft_extend_sconv_cache(
        hidden_nc,
        sconv_cache_fused,
        cache_indices,
        num_accepted,
        T,
        do_tracking,
        crossed,
        track_step,
        mamba_track_indices,
    )

    label = (
        f"draft_extend B={B} T={T} D={D} W={W} "
        f"track={do_tracking} all_crossed={all_crossed}"
    )
    ok = torch.allclose(sconv_cache_ref, sconv_cache_fused, atol=1e-5)
    if not ok:
        diff = (sconv_cache_ref - sconv_cache_fused).abs()
        print(f"    max_diff={diff.max():.4e}  mean={diff.mean():.4e}")
    _record(label, ok)


def test_fused_draft_extend_sweep() -> None:
    print("\n=== 5. fused_draft_extend_sconv_cache correctness sweep ===")

    for D, W in [(384, 3), (2304, 4), (4032, 4), (6144, 4)]:
        for B in [1, 4, 32]:
            for T in [1, 4, 8, 32]:
                for do_tracking in [False, True]:
                    for all_crossed in ([False, True] if do_tracking else [False]):
                        _run_draft_extend_case(B, T, D, W, do_tracking, all_crossed)


def bench_fused_draft_extend() -> None:
    print("\n=== Benchmark: fused_draft_extend_sconv_cache ===")
    from sglang.srt.models.inkling_common.sconv import fused_draft_extend_sconv_cache

    configs = [
        (8, 8, 2304, 4),
        (32, 8, 2304, 4),
        (8, 8, 6144, 4),
        (32, 8, 6144, 4),
        (8, 32, 2304, 4),
        (32, 32, 6144, 4),
    ]
    pool = 512

    labels, us_fused_col, us_base_col, speedup_col = [], [], [], []

    for B, T, D, W in configs:
        W_minus_1 = W - 1
        D_total = D * 4
        qkvr = _rand(B * T, D_total)
        hidden_nc = qkvr[:, :D]
        hidden_c = hidden_nc.contiguous()
        sc = _rand(pool, W_minus_1, D)
        ci = torch.arange(B, dtype=torch.int32, device=DEVICE)
        n_acc = torch.full((B,), T // 2, dtype=torch.int32, device=DEVICE)
        crossed = torch.ones(B, dtype=torch.bool, device=DEVICE)
        ts = torch.full((B,), T // 4, dtype=torch.int32, device=DEVICE)
        mti = (torch.arange(B, device=DEVICE) + B).to(torch.int64)

        us_f = bench(
            lambda: fused_draft_extend_sconv_cache(
                hidden_nc, sc, ci, n_acc, T, True, crossed, ts, mti
            )
        )
        us_b = bench(
            lambda: _baseline_draft_extend(
                hidden_c, sc, ci, n_acc, T, True, crossed, ts, mti
            )
        )
        sp = us_b / us_f if us_f > 0 else float("inf")

        labels.append(f"B={B} T={T} D={D}")
        us_fused_col.append(f"{us_f:.1f}")
        us_base_col.append(f"{us_b:.1f}")
        speedup_col.append(f"{sp:.2f}x")

    label_w = max(len("Config"), max(len(l) for l in labels)) + 2
    col_w = 14
    sep = f"+{'-'*label_w}+{'-'*col_w}+{'-'*col_w}+{'-'*col_w}+"
    hdr = (
        f"| {'Config'.ljust(label_w-1)}"
        f"| {'fused (µs)'.ljust(col_w-1)}"
        f"| {'baseline (µs)'.ljust(col_w-1)}"
        f"| {'speedup'.ljust(col_w-1)}|"
    )
    print("\nfused_draft_extend_sconv_cache benchmark")
    print(sep)
    print(hdr)
    print(sep)
    for i, lbl in enumerate(labels):
        print(
            f"| {lbl.ljust(label_w-1)}"
            f"| {us_fused_col[i].ljust(col_w-1)}"
            f"| {us_base_col[i].ljust(col_w-1)}"
            f"| {speedup_col[i].ljust(col_w-1)}|"
        )
    print(sep)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    print("=" * 72)
    print("Inkling Kernel Fusion Optimization – Correctness Tests + Benchmarks")
    print("=" * 72)

    # ---- Correctness ----
    print("\n" + "=" * 72)
    print("CORRECTNESS TESTS")
    print("=" * 72)

    test_causal_conv1d_sweep()
    test_fused_decode_sweep()
    test_rmsnorm_sweep()
    test_fused_gather_scatter_sweep()
    test_fused_draft_extend_sweep()

    # ---- Summary ----
    total = len(_results)
    failures = [(lbl, ok) for lbl, ok in _results if not ok]
    n_fail = len(failures)
    n_pass = total - n_fail

    print("\n" + "=" * 72)
    print(f"CORRECTNESS SUMMARY: {n_pass}/{total} passed, {n_fail} failed")
    if failures:
        print("FAILURES:")
        for lbl, _ in failures:
            print(f"  FAIL: {lbl}")
    print("=" * 72)

    # ---- Benchmarks ----
    print("\n" + "=" * 72)
    print("BENCHMARKS")
    print("=" * 72)

    bench_causal_conv1d()
    bench_fused_decode()
    bench_rmsnorm()
    bench_fused_gather_scatter()
    bench_fused_draft_extend()

    print("\nDone.")
