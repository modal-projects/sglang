"""Fused target-only sampled verify for DFLASH chains (Triton, ROCm).

Equivalent to: softmax(logits / T) -> top-k/top-p renorm (sorted order, ties by
index) -> speculative rejection sampling of a greedy (one-hot) draft chain ->
residual / bonus sampling. Avoids materializing probabilities and sorting the
vocabulary: the nucleus boundary is found by bisection on order-preserving
integer keys of the scaled logits.
"""

import torch
import triton
import triton.language as tl

_NUM_COARSE = 32
_COARSE_STEP = 0.5
_SCRATCH_CAP = 16384
_KEY_MIN = tl.constexpr(-(2**31))
_TOPK_ALL = 2**30


@triton.jit
def _sort_key(s):
    bits = s.to(tl.int32, bitcast=True)
    # -0.0 and +0.0 compare equal, so they must share a key to tie.
    bits = tl.where(s == 0.0, 0, bits)
    return tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)


@triton.jit
def _mass_count_above(
    src_ptr,
    n,
    x_key,
    lo_key,
    hi_key,
    row_max,
    inv_temp,
    IS_RAW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    mass = tl.zeros([BLOCK], dtype=tl.float32)
    cnt = tl.zeros([BLOCK], dtype=tl.int32)
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        s = tl.load(src_ptr + offs, mask=valid, other=-float("inf")).to(tl.float32)
        if IS_RAW:
            s = s * inv_temp
        k = _sort_key(s)
        sel = valid & (k > x_key) & (k > lo_key) & (k <= hi_key)
        mass += tl.where(sel, tl.exp(s - row_max), 0.0)
        cnt += sel.to(tl.int32)
    return tl.sum(mass, axis=0), tl.sum(cnt, axis=0)


@triton.jit
def _tie_stats(
    src_ptr,
    n,
    v_key,
    lo_key,
    hi_key,
    row_max,
    inv_temp,
    IS_RAW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Mass/count strictly above v_key and count/max-value of keys == v_key."""
    mass = tl.zeros([BLOCK], dtype=tl.float32)
    cnt = tl.zeros([BLOCK], dtype=tl.int32)
    ties = tl.zeros([BLOCK], dtype=tl.int32)
    tie_val = tl.full([BLOCK], -float("inf"), dtype=tl.float32)
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        s = tl.load(src_ptr + offs, mask=valid, other=-float("inf")).to(tl.float32)
        if IS_RAW:
            s = s * inv_temp
        k = _sort_key(s)
        inb = valid & (k > lo_key) & (k <= hi_key)
        above = inb & (k > v_key)
        eq = inb & (k == v_key)
        mass += tl.where(above, tl.exp(s - row_max), 0.0)
        cnt += above.to(tl.int32)
        ties += eq.to(tl.int32)
        tie_val = tl.maximum(tie_val, tl.where(eq, s, -float("inf")))
    return (
        tl.sum(mass, axis=0),
        tl.sum(cnt, axis=0),
        tl.sum(ties, axis=0),
        tl.max(tie_val, axis=0),
    )


@triton.jit
def _nucleus_row_kernel(
    logits_ptr,
    logits_stride,
    temps_ptr,
    top_ks_ptr,
    top_ps_ptr,
    scratch_ptr,
    out_max_ptr,
    out_zkept_ptr,
    out_vkey_ptr,
    out_nkeep_ptr,
    V,
    D: tl.constexpr,
    NUM_COARSE: tl.constexpr,
    COARSE_STEP: tl.constexpr,
    SCRATCH_CAP: tl.constexpr,
    BLOCK: tl.constexpr,
    COARSE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // D
    row_ptr = logits_ptr + row.to(tl.int64) * logits_stride
    inv_temp = 1.0 / tl.load(temps_ptr + b).to(tl.float32)
    top_k = tl.minimum(tl.load(top_ks_ptr + b).to(tl.int32), V)
    top_p = tl.load(top_ps_ptr + b).to(tl.float32)

    # Pass 1: row max and partition function of the scaled logits.
    m_vec = tl.full([BLOCK], -float("inf"), dtype=tl.float32)
    z_vec = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        s = tl.load(row_ptr + offs, mask=offs < V, other=-float("inf")).to(tl.float32)
        s = s * inv_temp
        new_m = tl.maximum(m_vec, s)
        safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
        z_vec = z_vec * tl.exp(m_vec - safe_m) + tl.exp(s - safe_m)
        m_vec = new_m
    row_max = tl.max(m_vec, axis=0)
    z = tl.sum(z_vec * tl.exp(m_vec - row_max), axis=0)

    filtering = (top_p < 1.0) | (top_k < V)
    v_key = tl.full([], _KEY_MIN, dtype=tl.int64)
    n_keep = V
    z_kept = z
    if filtering:
        p_mass = top_p * z
        # Pass 2: mass/count above coarse thresholds max - k*step, k = 1..NUM_COARSE.
        ks = tl.arange(0, NUM_COARSE)
        thr = row_max - (ks + 1).to(tl.float32) * COARSE_STEP
        thr_key = _sort_key(thr)
        c_mass = tl.zeros([NUM_COARSE], dtype=tl.float32)
        c_cnt = tl.zeros([NUM_COARSE], dtype=tl.int32)
        for start in range(0, V, COARSE_BLOCK):
            offs = start + tl.arange(0, COARSE_BLOCK)
            s = tl.load(row_ptr + offs, mask=offs < V, other=-float("inf"))
            s = s.to(tl.float32) * inv_temp
            k = _sort_key(s)
            e = tl.exp(s - row_max)
            above = k[:, None] > thr_key[None, :]
            c_mass += tl.sum(tl.where(above, e[:, None], 0.0), axis=0)
            c_cnt += tl.sum(above.to(tl.int32), axis=0)
        ok = (c_mass <= p_mass) & (c_cnt < top_k)
        # ok is true for a prefix of thresholds; hi is the lowest true one.
        n_ok = tl.sum(ok.to(tl.int32), axis=0)
        hi_key = tl.sum(tl.where(ks == n_ok - 1, thr_key, 0), axis=0).to(tl.int64)
        base_mass = tl.sum(tl.where(ks == n_ok - 1, c_mass, 0.0), axis=0)
        base_cnt = tl.sum(tl.where(ks == n_ok - 1, c_cnt, 0), axis=0)
        if n_ok == 0:
            hi_key = _sort_key(row_max).to(tl.int64)
            base_mass = 0.0
            base_cnt = 0
        lo_key = tl.sum(tl.where(ks == n_ok, thr_key, 0), axis=0).to(tl.int64)
        if n_ok == NUM_COARSE:
            lo_key = tl.full([], _KEY_MIN - 1, dtype=tl.int64)

        # Pass 3: compact the bracket (lo_key, hi_key] into scratch, in index order.
        scratch_row = scratch_ptr + row.to(tl.int64) * SCRATCH_CAP
        n_c = 0
        for start in range(0, V, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            valid = offs < V
            s = tl.load(row_ptr + offs, mask=valid, other=-float("inf"))
            s = s.to(tl.float32) * inv_temp
            k = _sort_key(s).to(tl.int64)
            inb = valid & (k > lo_key) & (k <= hi_key)
            pos = n_c + tl.cumsum(inb.to(tl.int32), axis=0) - 1
            tl.store(scratch_row + pos, s, mask=inb & (pos < SCRATCH_CAP))
            n_c += tl.sum(inb.to(tl.int32), axis=0)
        use_scratch = n_c <= SCRATCH_CAP

        # Bisection for the smallest key v with keep(v) true. keep(lo) is false,
        # keep(hi) is true, and keep only flips at keys present in the row.
        lo = lo_key
        hi = hi_key
        while hi - lo > 1:
            mid = lo + (hi - lo) // 2
            if use_scratch:
                a, c = _mass_count_above(
                    scratch_row, n_c, mid, lo_key, hi_key, row_max, 1.0, False, BLOCK
                )
            else:
                a, c = _mass_count_above(
                    row_ptr, V, mid, lo_key, hi_key, row_max, inv_temp, True, BLOCK
                )
            if ((base_mass + a) <= p_mass) & ((base_cnt + c) < top_k):
                hi = mid
            else:
                lo = mid
        v_key = hi
        if use_scratch:
            a, c, ties, v_val = _tie_stats(
                scratch_row, n_c, v_key, lo_key, hi_key, row_max, 1.0, False, BLOCK
            )
        else:
            a, c, ties, v_val = _tie_stats(
                row_ptr, V, v_key, lo_key, hi_key, row_max, inv_temp, True, BLOCK
            )
        a = base_mass + a
        c = base_cnt + c
        e_v = tl.exp(v_val - row_max)
        n_keep = tl.minimum(ties, top_k - c)
        if e_v > 0:
            room = tl.minimum(tl.floor((p_mass - a) / e_v), 1073741824.0)
            n_keep = tl.minimum(n_keep, room.to(tl.int32) + 1)
        n_keep = tl.maximum(n_keep, 1)
        z_kept = a + n_keep.to(tl.float32) * e_v

    tl.store(out_max_ptr + row, row_max)
    tl.store(out_zkept_ptr + row, z_kept)
    tl.store(out_vkey_ptr + row, v_key)
    tl.store(out_nkeep_ptr + row, n_keep)


@triton.jit
def _kept_prob_unnorm(
    row_ptr, tok, inv_temp, row_max, v_key, n_keep, BLOCK: tl.constexpr
):
    """exp(s_tok - max) if tok is inside the kept set, else 0."""
    s = tl.load(row_ptr + tok).to(tl.float32) * inv_temp
    k = _sort_key(s).to(tl.int64)
    e = tl.exp(s - row_max)
    is_tie = k == v_key
    # Ties keep the first n_keep in index order; only a tied token scans its prefix.
    rank = tl.zeros([], dtype=tl.int32)
    for start in range(0, tl.where(is_tie, tok, 0), BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < tok
        sp = tl.load(row_ptr + offs, mask=m, other=-float("inf")).to(tl.float32)
        kp = _sort_key(sp * inv_temp).to(tl.int64)
        rank += tl.sum((m & (kp == v_key)).to(tl.int32), axis=0)
    kept = (k > v_key) | (is_tie & (rank < n_keep))
    return tl.where(kept, e, 0.0)


@triton.jit
def _chain_verify_kernel(
    logits_ptr,
    logits_stride,
    candidates_ptr,
    temps_ptr,
    row_max_ptr,
    zkept_ptr,
    vkey_ptr,
    nkeep_ptr,
    coins_ptr,
    final_coins_ptr,
    accept_len_ptr,
    bonus_ptr,
    V,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    inv_temp = 1.0 / tl.load(temps_ptr + b).to(tl.float32)
    n_acc = 0
    rej_tok = -1
    for j in tl.static_range(1, D):
        row = b * D + (j - 1)
        row_ptr = logits_ptr + row.to(tl.int64) * logits_stride
        tok = tl.load(candidates_ptr + b * D + j).to(tl.int32)
        e = _kept_prob_unnorm(
            row_ptr,
            tok,
            inv_temp,
            tl.load(row_max_ptr + row),
            tl.load(vkey_ptr + row),
            tl.load(nkeep_ptr + row),
            BLOCK,
        )
        p = e / tl.load(zkept_ptr + row)
        coin = tl.load(coins_ptr + b * D + (j - 1))
        alive = n_acc == j - 1
        rej_tok = tl.where(alive & (coin >= p), tok, rej_tok)
        n_acc = n_acc + (alive & (coin < p)).to(tl.int32)

    row = b * D + n_acc
    row_ptr = logits_ptr + row.to(tl.int64) * logits_stride
    row_max = tl.load(row_max_ptr + row)
    v_key = tl.load(vkey_ptr + row)
    n_keep = tl.load(nkeep_ptr + row)
    z_kept = tl.load(zkept_ptr + row)
    excl = 0.0
    if rej_tok >= 0:
        excl = _kept_prob_unnorm(
            row_ptr, rej_tok, inv_temp, row_max, v_key, n_keep, BLOCK
        )
    target = tl.load(final_coins_ptr + b) * (z_kept - excl)

    cum = 0.0
    ties_seen = 0
    chosen = -1
    last_valid = -1
    for start in range(0, V, BLOCK):
        if chosen < 0:
            offs = start + tl.arange(0, BLOCK)
            valid = offs < V
            s = tl.load(row_ptr + offs, mask=valid, other=-float("inf")).to(tl.float32)
            s = s * inv_temp
            k = _sort_key(s).to(tl.int64)
            eq = valid & (k == v_key)
            tie_rank = ties_seen + tl.cumsum(eq.to(tl.int32), axis=0) - 1
            kept = valid & ((k > v_key) | (eq & (tie_rank < n_keep)))
            kept = kept & (offs != rej_tok)
            val = tl.where(kept, tl.exp(s - row_max), 0.0)
            incl = cum + tl.cumsum(val, axis=0)
            hit = (incl > target) & (val > 0)
            first = tl.min(tl.where(hit, offs, V), axis=0)
            if first < V:
                chosen = first
            last_valid = tl.maximum(
                last_valid, tl.max(tl.where(val > 0, offs, -1), axis=0)
            )
            cum += tl.sum(val, axis=0)
            ties_seen += tl.sum(eq.to(tl.int32), axis=0)
    if chosen < 0:
        chosen = last_valid
    tl.store(accept_len_ptr + b, n_acc)
    tl.store(bonus_ptr + b, chosen.to(tl.int64))


def dflash_sampled_verify_rocm(
    *,
    logits: torch.Tensor,
    candidates: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    coins: torch.Tensor,
    coins_final: torch.Tensor,
    apply_top_k: bool = True,
    apply_top_p: bool = True,
):
    """Returns (accept_len int32 [bs], bonus int64 [bs]).

    logits: [bs * D, V] post-processing target logits (before temperature).
    candidates: [bs, D]; column 0 is the verified root, 1..D-1 the draft chain.
    temperatures/top_ks/top_ps: per request, [bs] (or [bs, 1]).
    coins: [bs, D] uniforms (column j-1 tests draft j); coins_final: [bs].
    """
    bs, D = candidates.shape
    R, V = logits.shape
    assert R == bs * D and logits.stride(-1) == 1
    device = logits.device
    temps = temperatures.reshape(bs).to(torch.float32).contiguous()
    if apply_top_k:
        ks = top_ks.reshape(bs).to(torch.int32).clamp_(min=1).contiguous()
    else:
        ks = torch.full((bs,), _TOPK_ALL, dtype=torch.int32, device=device)
    if apply_top_p:
        ps = top_ps.reshape(bs).to(torch.float32).contiguous()
    else:
        ps = torch.ones((bs,), dtype=torch.float32, device=device)

    row_max = torch.empty((R,), dtype=torch.float32, device=device)
    z_kept = torch.empty((R,), dtype=torch.float32, device=device)
    v_key = torch.empty((R,), dtype=torch.int64, device=device)
    n_keep = torch.empty((R,), dtype=torch.int32, device=device)
    scratch_cap = min(_SCRATCH_CAP, V)
    scratch = torch.empty((R, scratch_cap), dtype=torch.float32, device=device)
    _nucleus_row_kernel[(R,)](
        logits,
        logits.stride(0),
        temps,
        ks,
        ps,
        scratch,
        row_max,
        z_kept,
        v_key,
        n_keep,
        V,
        D=D,
        NUM_COARSE=_NUM_COARSE,
        COARSE_STEP=_COARSE_STEP,
        SCRATCH_CAP=scratch_cap,
        BLOCK=1024,
        COARSE_BLOCK=256,
        num_warps=8,
    )
    accept_len = torch.empty((bs,), dtype=torch.int32, device=device)
    bonus = torch.empty((bs,), dtype=torch.int64, device=device)
    _chain_verify_kernel[(bs,)](
        logits,
        logits.stride(0),
        candidates.contiguous(),
        temps,
        row_max,
        z_kept,
        v_key,
        n_keep,
        coins.to(torch.float32).contiguous(),
        coins_final.reshape(bs).to(torch.float32).contiguous(),
        accept_len,
        bonus,
        V,
        D=D,
        BLOCK=1024,
        num_warps=8,
    )
    return accept_len, bonus
