"""Fused target-only sampled verify for DFLASH chains (Triton, ROCm).

Equivalent to: softmax(logits / T) -> top-k/top-p renorm (sorted order, ties by
index) -> speculative rejection sampling of a greedy (one-hot) draft chain ->
residual / bonus sampling. Never materializes probabilities or sorts the
vocabulary. Every vocab pass is split into chunks across programs:

1. per-chunk max / partition function;
2. per-chunk histogram of (row_max - s) in buckets of width _BW;
3. per row: pick a bucket bracket that must contain the top-k/top-p cutoff;
4. per chunk: exact mass above the bracket, compact bracket tokens (index order);
5. per row: exact cutoff by bisection on order-preserving keys inside the
   bracket (or over the whole row if the bracket is unusable);
6. per chunk: chain accept/reject and kept residual mass;
7. per request: residual / bonus sampling.
"""

import torch
import triton
import triton.language as tl

_NB = 512
_BW = 1.0 / 16
_CHUNK = 4096
_BLOCK = 1024
_SCRATCH_CAP = 16384
_KEY_MIN = tl.constexpr(-(2**31))
_KEY_MAX = tl.constexpr(2**31)
_TOPK_ALL = 2**30


@triton.jit
def _sort_key(s):
    bits = s.to(tl.int32, bitcast=True)
    # -0.0 and +0.0 compare equal, so they must share a key to tie.
    bits = tl.where(s == 0.0, 0, bits)
    return tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)


@triton.jit
def _bucket(s, row_max, INV_BW: tl.constexpr, NB: tl.constexpr):
    d = (row_max - s) * INV_BW
    d = tl.where(d < NB - 1.0, d, NB - 1.0)
    return d.to(tl.int32)


@triton.jit
def _row_params(temps_ptr, top_ks_ptr, top_ps_ptr, b, V):
    inv_temp = 1.0 / tl.load(temps_ptr + b).to(tl.float32)
    top_k = tl.minimum(tl.load(top_ks_ptr + b).to(tl.int32), V)
    top_p = tl.load(top_ps_ptr + b).to(tl.float32)
    return inv_temp, top_k, top_p


@triton.jit
def _row_max(cmax_ptr, row, C, C_PAD: tl.constexpr):
    cs = tl.arange(0, C_PAD)
    m = tl.load(cmax_ptr + row * C + cs, mask=cs < C, other=-float("inf"))
    return tl.max(m, axis=0)


@triton.jit
def _chunk_stats_kernel(
    logits_ptr,
    logits_stride,
    temps_ptr,
    cmax_ptr,
    cz_ptr,
    V,
    C,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.program_id(1)
    row_ptr = logits_ptr + row.to(tl.int64) * logits_stride
    inv_temp = 1.0 / tl.load(temps_ptr + row // D).to(tl.float32)
    m_vec = tl.full([BLOCK], -float("inf"), dtype=tl.float32)
    z_vec = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(c * CHUNK, c * CHUNK + CHUNK, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        s = tl.load(row_ptr + offs, mask=offs < V, other=-float("inf")).to(tl.float32)
        s = s * inv_temp
        new_m = tl.maximum(m_vec, s)
        safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
        z_vec = z_vec * tl.exp(m_vec - safe_m) + tl.exp(s - safe_m)
        m_vec = new_m
    m = tl.max(m_vec, axis=0)
    safe = tl.where(m == -float("inf"), 0.0, m)
    tl.store(cmax_ptr + row * C + c, m)
    tl.store(cz_ptr + row * C + c, tl.sum(z_vec * tl.exp(m_vec - safe), axis=0))


@triton.jit
def _chunk_hist_kernel(
    logits_ptr,
    logits_stride,
    temps_ptr,
    cmax_ptr,
    hist_ptr,
    V,
    C,
    D: tl.constexpr,
    C_PAD: tl.constexpr,
    NB: tl.constexpr,
    INV_BW: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.program_id(1)
    row_ptr = logits_ptr + row.to(tl.int64) * logits_stride
    inv_temp = 1.0 / tl.load(temps_ptr + row // D).to(tl.float32)
    row_max = _row_max(cmax_ptr, row, C, C_PAD)
    h = tl.zeros([NB], dtype=tl.int32)
    for start in range(c * CHUNK, c * CHUNK + CHUNK, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < V
        s = tl.load(row_ptr + offs, mask=valid, other=-float("inf")).to(tl.float32)
        bk = tl.where(valid, _bucket(s * inv_temp, row_max, INV_BW, NB), NB - 1)
        h += tl.histogram(bk, NB)
    tl.store(hist_ptr + (row * C + c).to(tl.int64) * NB + tl.arange(0, NB), h)


@triton.jit
def _select_kernel(
    temps_ptr,
    top_ks_ptr,
    top_ps_ptr,
    cmax_ptr,
    cz_ptr,
    hist_ptr,
    row_max_ptr,
    z_ptr,
    blo_ptr,
    bhi_ptr,
    base_cnt_ptr,
    nbr_ptr,
    off_ptr,
    V,
    C,
    D: tl.constexpr,
    C_PAD: tl.constexpr,
    NB: tl.constexpr,
    BW: tl.constexpr,
):
    row = tl.program_id(0)
    _, top_k, top_p = _row_params(temps_ptr, top_ks_ptr, top_ps_ptr, row // D, V)
    cs = tl.arange(0, C_PAD)
    cm = tl.load(cmax_ptr + row * C + cs, mask=cs < C, other=-float("inf"))
    cz = tl.load(cz_ptr + row * C + cs, mask=cs < C, other=0.0)
    row_max = tl.max(cm, axis=0)
    z = tl.sum(tl.where(cs < C, cz * tl.exp(cm - row_max), 0.0), axis=0)
    tl.store(row_max_ptr + row, row_max)
    tl.store(z_ptr + row, z)
    if (top_p < 1.0) | (top_k < V):
        bidx = tl.arange(0, NB)
        hrow = hist_ptr + row.to(tl.int64) * C * NB
        H = tl.zeros([NB], dtype=tl.int32)
        for c in range(C):
            H += tl.load(hrow + c * NB + bidx)
        Hf = H.to(tl.float32)
        lo_m = Hf * tl.exp(-(bidx + 1).to(tl.float32) * BW)
        hi_m = Hf * tl.exp(-bidx.to(tl.float32) * BW)
        lo_above = tl.cumsum(lo_m, axis=0) - lo_m
        hi_above = tl.cumsum(hi_m, axis=0) - hi_m
        cnt_above = tl.cumsum(H, axis=0) - H
        p_mass = top_p * z
        sure = (hi_above * 1.0001 <= p_mass) & (cnt_above < top_k)
        maybe = (lo_above * 0.9999 <= p_mass) & (cnt_above < top_k)
        b_lo = tl.max(tl.where(sure, bidx, 0), axis=0)
        b_hi = tl.max(tl.where(maybe, bidx, 0), axis=0)
        inb = (bidx >= b_lo) & (bidx <= b_hi)
        tl.store(blo_ptr + row, b_lo)
        tl.store(bhi_ptr + row, b_hi)
        tl.store(base_cnt_ptr + row, tl.sum(tl.where(bidx < b_lo, H, 0), axis=0))
        run = 0
        for c in range(C):
            tl.store(off_ptr + row * C + c, run)
            run += tl.sum(tl.where(inb, tl.load(hrow + c * NB + bidx), 0), axis=0)
        tl.store(nbr_ptr + row, run)


@triton.jit
def _compact_kernel(
    logits_ptr,
    logits_stride,
    temps_ptr,
    top_ks_ptr,
    top_ps_ptr,
    row_max_ptr,
    blo_ptr,
    bhi_ptr,
    nbr_ptr,
    off_ptr,
    amass_ptr,
    sval_ptr,
    sidx_ptr,
    V,
    C,
    D: tl.constexpr,
    NB: tl.constexpr,
    INV_BW: tl.constexpr,
    SCRATCH_CAP: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.program_id(1)
    inv_temp, top_k, top_p = _row_params(temps_ptr, top_ks_ptr, top_ps_ptr, row // D, V)
    if (top_p < 1.0) | (top_k < V):
        row_ptr = logits_ptr + row.to(tl.int64) * logits_stride
        row_max = tl.load(row_max_ptr + row)
        b_lo = tl.load(blo_ptr + row)
        b_hi = tl.load(bhi_ptr + row)
        fits = tl.load(nbr_ptr + row) <= SCRATCH_CAP
        pos0 = tl.load(off_ptr + row * C + c)
        sv = sval_ptr + row.to(tl.int64) * SCRATCH_CAP
        si = sidx_ptr + row.to(tl.int64) * SCRATCH_CAP
        mass = tl.zeros([BLOCK], dtype=tl.float32)
        for start in range(c * CHUNK, c * CHUNK + CHUNK, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            valid = offs < V
            s = tl.load(row_ptr + offs, mask=valid, other=-float("inf")).to(tl.float32)
            s = s * inv_temp
            bk = _bucket(s, row_max, INV_BW, NB)
            mass += tl.where(valid & (bk < b_lo), tl.exp(s - row_max), 0.0)
            inb = valid & (bk >= b_lo) & (bk <= b_hi)
            pos = pos0 + tl.cumsum(inb.to(tl.int32), axis=0) - 1
            tl.store(sv + pos, s, mask=inb & fits)
            tl.store(si + pos, offs, mask=inb & fits)
            pos0 += tl.sum(inb.to(tl.int32), axis=0)
        tl.store(amass_ptr + row * C + c, tl.sum(mass, axis=0))


@triton.jit
def _above(
    src_ptr, n, x_key, row_max, inv_temp, IS_RAW: tl.constexpr, BLOCK: tl.constexpr
):
    """Mass and count of keys > x_key; also count / value of keys == x_key."""
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
        k = _sort_key(s).to(tl.int64)
        gt = valid & (k > x_key)
        eq = valid & (k == x_key)
        mass += tl.where(gt, tl.exp(s - row_max), 0.0)
        cnt += gt.to(tl.int32)
        ties += eq.to(tl.int32)
        tie_val = tl.maximum(tie_val, tl.where(eq, s, -float("inf")))
    return (
        tl.sum(mass, axis=0),
        tl.sum(cnt, axis=0),
        tl.sum(ties, axis=0),
        tl.max(tie_val, axis=0),
    )


@triton.jit
def _cutoff(
    src_ptr,
    idx_ptr,
    n,
    lo,
    hi,
    base_mass,
    base_cnt,
    p_mass,
    top_k,
    row_max,
    inv_temp,
    IS_RAW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Smallest key v kept by the sorted-order filter, given keep(lo) false and
    keep(hi) true; returns (v, n_keep ties of v, kept mass, index of last kept tie)."""
    while hi - lo > 1:
        mid = lo + (hi - lo) // 2
        a, c, _, _ = _above(src_ptr, n, mid, row_max, inv_temp, IS_RAW, BLOCK)
        if ((base_mass + a) <= p_mass) & ((base_cnt + c) < top_k):
            hi = mid
        else:
            lo = mid
    a, c, ties, v_val = _above(src_ptr, n, hi, row_max, inv_temp, IS_RAW, BLOCK)
    a = base_mass + a
    c = base_cnt + c
    e_v = tl.exp(v_val - row_max)
    n_keep = tl.minimum(ties, top_k - c)
    if e_v > 0:
        room = tl.minimum(tl.floor((p_mass - a) / e_v), 1073741824.0)
        n_keep = tl.minimum(n_keep, room.to(tl.int32) + 1)
    n_keep = tl.maximum(n_keep, 1)
    last = -1
    seen = 0
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        s = tl.load(src_ptr + offs, mask=valid, other=-float("inf")).to(tl.float32)
        if IS_RAW:
            s = s * inv_temp
            idx = offs
        else:
            idx = tl.load(idx_ptr + offs, mask=valid, other=-1)
        eq = valid & (_sort_key(s).to(tl.int64) == hi)
        rank = seen + tl.cumsum(eq.to(tl.int32), axis=0) - 1
        last = tl.maximum(
            last, tl.max(tl.where(eq & (rank == n_keep - 1), idx, -1), axis=0)
        )
        seen += tl.sum(eq.to(tl.int32), axis=0)
    return hi, a + n_keep.to(tl.float32) * e_v, last


@triton.jit
def _refine_kernel(
    logits_ptr,
    logits_stride,
    temps_ptr,
    top_ks_ptr,
    top_ps_ptr,
    row_max_ptr,
    z_ptr,
    bhi_ptr,
    base_cnt_ptr,
    nbr_ptr,
    amass_ptr,
    sval_ptr,
    sidx_ptr,
    zkept_ptr,
    vkey_ptr,
    last_ptr,
    V,
    C,
    D: tl.constexpr,
    C_PAD: tl.constexpr,
    NB: tl.constexpr,
    SCRATCH_CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    inv_temp, top_k, top_p = _row_params(temps_ptr, top_ks_ptr, top_ps_ptr, row // D, V)
    row_max = tl.load(row_max_ptr + row)
    z = tl.load(z_ptr + row)
    v_key = tl.full([], _KEY_MIN, dtype=tl.int64)
    z_kept = z
    last = V
    if (top_p < 1.0) | (top_k < V):
        p_mass = top_p * z
        cs = tl.arange(0, C_PAD)
        base = tl.sum(tl.load(amass_ptr + row * C + cs, mask=cs < C, other=0.0), axis=0)
        base_cnt = tl.load(base_cnt_ptr + row)
        n_br = tl.load(nbr_ptr + row)
        sv = sval_ptr + row.to(tl.int64) * SCRATCH_CAP
        si = sidx_ptr + row.to(tl.int64) * SCRATCH_CAP
        ok = (n_br <= SCRATCH_CAP) & (tl.load(bhi_ptr + row) < NB - 1)
        ok = ok & (base <= p_mass) & (base_cnt < top_k)
        key_floor = tl.full([], _KEY_MIN - 1, dtype=tl.int64)
        kmin = tl.full([], _KEY_MAX, dtype=tl.int64)
        kmax = tl.full([], _KEY_MIN, dtype=tl.int64)
        if ok:
            a, _, _, _ = _above(sv, n_br, key_floor, row_max, 1.0, False, BLOCK)
            # The cutoff must lie inside the bracket: the token below it is dropped.
            ok = ((base + a) > p_mass) | ((base_cnt + n_br) >= top_k)
            for start in range(0, n_br, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                m = offs < n_br
                k = _sort_key(tl.load(sv + offs, mask=m, other=0.0)).to(tl.int64)
                kmin = tl.minimum(kmin, tl.min(tl.where(m, k, _KEY_MAX), axis=0))
                kmax = tl.maximum(kmax, tl.max(tl.where(m, k, _KEY_MIN), axis=0))
        if ok:
            v_key, z_kept, last = _cutoff(
                sv,
                si,
                n_br,
                kmin - 1,
                kmax,
                base,
                base_cnt,
                p_mass,
                top_k,
                row_max,
                1.0,
                False,
                BLOCK,
            )
        else:
            row_ptr = logits_ptr + row.to(tl.int64) * logits_stride
            v_key, z_kept, last = _cutoff(
                row_ptr,
                row_ptr,
                V,
                key_floor,
                _sort_key(row_max).to(tl.int64),
                0.0,
                0,
                p_mass,
                top_k,
                row_max,
                inv_temp,
                True,
                BLOCK,
            )
    tl.store(zkept_ptr + row, z_kept)
    tl.store(vkey_ptr + row, v_key)
    tl.store(last_ptr + row, last)


@triton.jit
def _kept(s, idx, v_key, last):
    k = _sort_key(s).to(tl.int64)
    return (k > v_key) | ((k == v_key) & (idx <= last))


@triton.jit
def _accept_kernel(
    logits_ptr,
    logits_stride,
    candidates_ptr,
    temps_ptr,
    row_max_ptr,
    zkept_ptr,
    vkey_ptr,
    last_ptr,
    coins_ptr,
    part_ptr,
    acc_ptr,
    rej_ptr,
    V,
    C,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    inv_temp = 1.0 / tl.load(temps_ptr + b).to(tl.float32)
    n_acc = 0
    rej_tok = -1
    for j in tl.static_range(1, D):
        row = b * D + (j - 1)
        tok = tl.load(candidates_ptr + b * D + j).to(tl.int32)
        st = tl.load(logits_ptr + row.to(tl.int64) * logits_stride + tok).to(tl.float32)
        st = st * inv_temp
        in_set = _kept(st, tok, tl.load(vkey_ptr + row), tl.load(last_ptr + row))
        e = tl.where(in_set, tl.exp(st - tl.load(row_max_ptr + row)), 0.0)
        p = e / tl.load(zkept_ptr + row)
        coin = tl.load(coins_ptr + b * D + (j - 1))
        alive = n_acc == j - 1
        rej_tok = tl.where(alive & (coin >= p), tok, rej_tok)
        n_acc = n_acc + (alive & (coin < p)).to(tl.int32)
    row = b * D + n_acc
    row_ptr = logits_ptr + row.to(tl.int64) * logits_stride
    row_max = tl.load(row_max_ptr + row)
    v_key = tl.load(vkey_ptr + row)
    last = tl.load(last_ptr + row)
    mass = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(c * CHUNK, c * CHUNK + CHUNK, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < V
        s = tl.load(row_ptr + offs, mask=valid, other=-float("inf")).to(tl.float32)
        s = s * inv_temp
        kept = valid & _kept(s, offs, v_key, last) & (offs != rej_tok)
        mass += tl.where(kept, tl.exp(s - row_max), 0.0)
    tl.store(part_ptr + b * C + c, tl.sum(mass, axis=0))
    if c == 0:
        tl.store(acc_ptr + b, n_acc)
        tl.store(rej_ptr + b, rej_tok)


@triton.jit
def _bonus_kernel(
    logits_ptr,
    logits_stride,
    temps_ptr,
    row_max_ptr,
    vkey_ptr,
    last_ptr,
    final_coins_ptr,
    part_ptr,
    acc_ptr,
    rej_ptr,
    bonus_ptr,
    V,
    C,
    D: tl.constexpr,
    C_PAD: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    inv_temp = 1.0 / tl.load(temps_ptr + b).to(tl.float32)
    n_acc = tl.load(acc_ptr + b)
    rej_tok = tl.load(rej_ptr + b)
    cs = tl.arange(0, C_PAD)
    part = tl.load(part_ptr + b * C + cs, mask=cs < C, other=0.0)
    incl = tl.cumsum(part, axis=0)
    target = tl.load(final_coins_ptr + b) * tl.sum(part, axis=0)
    hit = (incl > target) & (part > 0)
    last_pos = tl.max(tl.where(part > 0, cs, -1), axis=0)
    c = tl.min(tl.where(hit, cs, C_PAD), axis=0)
    c = tl.where(c < C_PAD, c, last_pos)
    cum = tl.sum(tl.where(cs == c, incl - part, 0.0), axis=0)
    row = b * D + n_acc
    row_ptr = logits_ptr + row.to(tl.int64) * logits_stride
    row_max = tl.load(row_max_ptr + row)
    v_key = tl.load(vkey_ptr + row)
    last = tl.load(last_ptr + row)
    chosen = -1
    last_valid = -1
    for start in range(c * CHUNK, c * CHUNK + CHUNK, BLOCK):
        if chosen < 0:
            offs = start + tl.arange(0, BLOCK)
            valid = offs < V
            s = tl.load(row_ptr + offs, mask=valid, other=-float("inf")).to(tl.float32)
            s = s * inv_temp
            kept = valid & _kept(s, offs, v_key, last) & (offs != rej_tok)
            val = tl.where(kept, tl.exp(s - row_max), 0.0)
            run = cum + tl.cumsum(val, axis=0)
            first = tl.min(tl.where((run > target) & (val > 0), offs, V), axis=0)
            if first < V:
                chosen = first
            last_valid = tl.maximum(
                last_valid, tl.max(tl.where(val > 0, offs, -1), axis=0)
            )
            cum += tl.sum(val, axis=0)
    if chosen < 0:
        chosen = last_valid
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
    dev = logits.device
    temps = temperatures.reshape(bs).to(torch.float32).contiguous()
    if apply_top_k:
        ks = top_ks.reshape(bs).to(torch.int32).clamp_(min=1).contiguous()
    else:
        ks = torch.full((bs,), _TOPK_ALL, dtype=torch.int32, device=dev)
    if apply_top_p:
        ps = top_ps.reshape(bs).to(torch.float32).contiguous()
    else:
        ps = torch.ones((bs,), dtype=torch.float32, device=dev)
    C = triton.cdiv(V, _CHUNK)
    C_PAD = triton.next_power_of_2(C)
    cap = min(_SCRATCH_CAP, V)
    f32 = dict(dtype=torch.float32, device=dev)
    i32 = dict(dtype=torch.int32, device=dev)
    cmax, cz, amass = (torch.empty((R, C), **f32) for _ in range(3))
    hist = torch.empty((R, C, _NB), **i32)
    off = torch.empty((R, C), **i32)
    row_max, z, z_kept = (torch.empty((R,), **f32) for _ in range(3))
    b_lo, b_hi, base_cnt, n_br, last = (torch.empty((R,), **i32) for _ in range(5))
    v_key = torch.empty((R,), dtype=torch.int64, device=dev)
    sval = torch.empty((R, cap), **f32)
    sidx = torch.empty((R, cap), **i32)
    common = dict(CHUNK=_CHUNK, BLOCK=_BLOCK, num_warps=4)
    st = logits.stride(0)
    _chunk_stats_kernel[(R, C)](logits, st, temps, cmax, cz, V, C, D=D, **common)
    _chunk_hist_kernel[(R, C)](
        logits,
        st,
        temps,
        cmax,
        hist,
        V,
        C,
        D=D,
        C_PAD=C_PAD,
        NB=_NB,
        INV_BW=1.0 / _BW,
        **common,
    )
    _select_kernel[(R,)](
        temps,
        ks,
        ps,
        cmax,
        cz,
        hist,
        row_max,
        z,
        b_lo,
        b_hi,
        base_cnt,
        n_br,
        off,
        V,
        C,
        D=D,
        C_PAD=C_PAD,
        NB=_NB,
        BW=_BW,
        num_warps=4,
    )
    _compact_kernel[(R, C)](
        logits,
        st,
        temps,
        ks,
        ps,
        row_max,
        b_lo,
        b_hi,
        n_br,
        off,
        amass,
        sval,
        sidx,
        V,
        C,
        D=D,
        NB=_NB,
        INV_BW=1.0 / _BW,
        SCRATCH_CAP=cap,
        **common,
    )
    _refine_kernel[(R,)](
        logits,
        st,
        temps,
        ks,
        ps,
        row_max,
        z,
        b_hi,
        base_cnt,
        n_br,
        amass,
        sval,
        sidx,
        z_kept,
        v_key,
        last,
        V,
        C,
        D=D,
        C_PAD=C_PAD,
        NB=_NB,
        SCRATCH_CAP=cap,
        BLOCK=_BLOCK,
        num_warps=8,
    )
    part = torch.empty((bs, C), **f32)
    accept_len = torch.empty((bs,), **i32)
    rej = torch.empty((bs,), **i32)
    bonus = torch.empty((bs,), dtype=torch.int64, device=dev)
    coins = coins.to(torch.float32).contiguous()
    coins_final = coins_final.reshape(bs).to(torch.float32).contiguous()
    cand = candidates.contiguous()
    _accept_kernel[(bs, C)](
        logits,
        st,
        cand,
        temps,
        row_max,
        z_kept,
        v_key,
        last,
        coins,
        part,
        accept_len,
        rej,
        V,
        C,
        D=D,
        **common,
    )
    _bonus_kernel[(bs,)](
        logits,
        st,
        temps,
        row_max,
        v_key,
        last,
        coins_final,
        part,
        accept_len,
        rej,
        bonus,
        V,
        C,
        D=D,
        C_PAD=C_PAD,
        **common,
    )
    return accept_len, bonus
