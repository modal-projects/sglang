"""Routing, activation quantization, gate/up LoRA shrink and finalize around the fused
routed-expert kernels (texperts). All buffers are preallocated: every launch is CUDA-graph safe.

Rows are routed (token, k) pairs owned by this rank, grouped by local expert into 16-row
blocks: expert e's rows start at 16 * (number of blocks of experts < e).
"""

import torch
import triton
import triton.language as tl

NT = tl.constexpr(16)
_SKIP_PAD = __import__("os").environ.get("SGLANG_TEXPERTS_QUANT_SKIP_PAD", "1") != "0"


# BLOCK takes a few fixed sizes (Prep.route_block), and Prep.precompile_route compiles each
# for both divisibility-by-16 classes of the route count that Triton specializes on (the
# alignment hint keeps the route loads vectorized). New variants compiling mid-serve
# stalled every rank for tens of seconds on the first single-request prefill of an unseen
# size. The folded-token count is not specialized on.
@triton.jit(do_not_specialize=["n_tok"])
def _route_kernel(topk_ids, topk_w, slots, scratch, row_of, rtok, rslot, rw, blk_expert,
                  blk_mask_gu, blk_mask_d, nblk, z, a_blk, a_s, n_a, npairs, zcnt,
                  n_routes, top_k, local_off,
                  E: tl.constexpr, ROWS_MAX: tl.constexpr, MB: tl.constexpr,
                  BLOCK: tl.constexpr, EB: tl.constexpr, RB: tl.constexpr, MBB: tl.constexpr,
                  SPT_D: tl.constexpr, ZW: tl.constexpr, ZROWS: tl.constexpr,
                  SL: tl.constexpr, FOLD: tl.constexpr, SHARED: tl.constexpr = False,
                  n_tok=0, E_TOT: tl.constexpr = 0, zd=None, ZERO_ZD: tl.constexpr = False):
    """Program 0: rows for the routes this rank owns. A route's rank inside its expert comes
    from an atomic counter, so rows within a block may be ordered differently from run to run;
    nothing downstream depends on that order (each row is computed and finalized on its own,
    block contents and Z sums are order independent). Programs 1..: clear the gate/up Z."""
    pid = tl.program_id(0)
    if pid > 0:
        zr = (pid - 1) * ZROWS + tl.arange(0, ZROWS)
        zc = tl.arange(0, ZW)
        tl.store(z + zr[:, None] * ZW + zc[None, :], tl.zeros((ZROWS, ZW), tl.int64),
                 mask=zr[:, None] < ROWS_MAX)
        if ZERO_ZD:  # the down-LoRA Z too (no quant kernel on the row-gather path)
            zc2 = tl.arange(0, ZW // 2)
            tl.store(zd + zr[:, None] * (ZW // 2) + zc2[None, :],
                     tl.zeros((ZROWS, ZW // 2), tl.int64), mask=zr[:, None] < ROWS_MAX)
        return
    r = tl.arange(0, RB)
    tl.store(rtok + r, -1, mask=r < ROWS_MAX)
    tl.store(rslot + r, -1, mask=r < ROWS_MAX)
    tl.store(rw + r, 0.0, mask=r < ROWS_MAX)
    b = tl.arange(0, MBB)
    tl.store(blk_expert + b, -1, mask=b < MB)
    tl.store(blk_mask_gu + b, 0, mask=b < MB)
    tl.store(blk_mask_d + b, 0, mask=b < MB)
    e = tl.arange(0, EB)
    tl.store(scratch + e, 0)
    tl.debug_barrier()
    i = tl.arange(0, BLOCK)
    valid_i = i < n_routes
    ids = tl.load(topk_ids + i, mask=valid_i, other=-1).to(tl.int32)
    loc = ids - local_off
    own = valid_i & (loc >= 0) & (loc < E)
    tok = i // top_k
    if SHARED:  # one extra route per token to the folded shared expert (local index E)
        sh = (i >= n_routes) & (i < n_routes + n_tok)
        loc = tl.where(sh, E, loc)
        own = own | sh
        tok = tl.where(sh, i - n_routes, tok)
        valid_i = valid_i | sh
    rank = tl.atomic_add(scratch + loc, 1, mask=own, sem="relaxed")
    tl.debug_barrier()
    cnt = tl.load(scratch + e, mask=e < E_TOT, other=0, volatile=True)
    nblk_e = (cnt + NT - 1) // NT
    blk_start = tl.cumsum(nblk_e, 0) - nblk_e
    tl.store(nblk, tl.sum(nblk_e, 0))
    if SHARED:  # the shared expert's blocks are the last ones
        tl.store(nblk + 1, tl.sum(tl.where(e == E, nblk_e, 0), 0))
    tl.store(scratch + EB + e, blk_start)
    tl.debug_barrier()
    bs = tl.load(scratch + EB + loc, mask=own, other=0, volatile=True)
    row = tl.where(own, bs * NT + rank, -1)
    tl.store(row_of + i, row, mask=valid_i)
    sl = tl.load(slots + tok, mask=own, other=-1)
    w = tl.load(topk_w + i, mask=own & (i < n_routes), other=0.0)
    tl.store(rtok + row, tok, mask=own)
    tl.store(rslot + row, sl, mask=own)
    tl.store(rw + row, w, mask=own)
    tl.store(blk_expert + row // NT, loc, mask=own)
    has = own & (sl >= 0)
    tl.atomic_or(blk_mask_gu + row // NT, 1 << sl, mask=has, sem="relaxed")
    tl.atomic_or(blk_mask_d + row // NT, 1 << (sl // SPT_D), mask=has, sem="relaxed")
    if FOLD:
        # A-items: per block, its slots in pairs (ascending); a_s = -1 pads an odd count
        tl.store(a_s + tl.arange(0, 2 * MBB * 4), -1, mask=tl.arange(0, 2 * MBB * 4) < MB * 8)
        tl.debug_barrier()
        bm = tl.load(blk_mask_gu + b, mask=b < MB, other=0, volatile=True)
        sb = tl.arange(0, SL)
        bit = (bm[:, None] >> sb[None, :]) & 1
        rk = tl.cumsum(bit, 1) - bit
        npr = (tl.sum(bit, 1) + 1) // 2
        aoff = tl.cumsum(npr, 0) - npr
        tl.store(npairs + b, npr, mask=b < MB)
        tl.store(zcnt + b, 0, mask=b < MB)
        tl.store(n_a, tl.sum(npr, 0))
        aid = aoff[:, None] + rk // 2
        on = (bit != 0) & (b[:, None] < MB)
        tl.store(a_s + aid * 2 + rk % 2, sb[None, :] + 0 * aid, mask=on)
        tl.store(a_blk + aid, b[:, None] + 0 * aid, mask=on)


@triton.jit
def _quant_kernel(x, rtok, nblk, xq, x_sf, zd, x_bf, H: tl.constexpr, PART: tl.constexpr,
                  N_SF_KT: tl.constexpr, R: tl.constexpr, BF: tl.constexpr,
                  MX: tl.constexpr = True, SHARED: tl.constexpr = False, blk_expert=None,
                  blk_mask=None, rslot=None, a_s=None, a_ss=0, z=None, E_SH=0,
                  SL: tl.constexpr = 8, SKIP_PAD: tl.constexpr = True):
    """Block b (16 rows), K-part p: MXFP8 of the rows' x[:, p*PART : +PART] into xq / x_sf
    (128x4 swizzle, one 128-row group per block); part 0 also clears the rows' down-LoRA Z.
    Padding rows (no token) are not written: they only feed MMA columns and LoRA rows that are
    never stored, so their stale (finite) contents are harmless, and most of a decode block's
    16 rows are padding."""
    blk = tl.program_id(0)
    p = tl.program_id(1)
    if blk >= tl.load(nblk):
        return
    n = tl.arange(0, 16)
    rows = blk * 16 + n
    tok = tl.load(rtok + rows)
    cols = p * PART + tl.arange(0, PART)
    xv = tl.load(x + tok.to(tl.int64)[:, None] * H + cols[None, :], mask=(tok >= 0)[:, None],
                 other=0.0)
    if SKIP_PAD:
        live = (tok >= 0)[:, None]
    else:
        live = ((tok >= 0) | (tok < 0))[:, None]
    if BF:  # the permuted BF16 rows (folded LoRA-A shrink; the BF16-base GEMM input)
        tl.store(x_bf + rows.to(tl.int64)[:, None] * H + cols[None, :], xv, mask=live)
    if p == 0:
        jd = tl.arange(0, R)
        tl.store(zd + rows[:, None] * R + jd[None, :], tl.zeros((16, R), tl.int64))
    if SHARED:  # folded shared expert: its gate/up LoRA-A (dense buffer, raw) for this part
        if tl.load(blk_expert + blk) == E_SH:
            msk = tl.load(blk_mask + blk)
            sl = tl.load(rslot + rows)
            j = tl.arange(0, 2 * R)
            for si in tl.static_range(SL):
                if ((msk >> si) & 1) != 0:
                    a = tl.load(a_s + si * a_ss + j[:, None] * H + cols[None, :])
                    zz = tl.dot(xv, tl.trans(a))
                    zz = tl.where(zz == zz, zz, 0.0)  # past-rank garbage: masked later
                    zz = tl.minimum(tl.maximum(zz, -1.0e9), 1.0e9)
                    zi = tl.floor(zz * 4294967296.0 + 0.5).to(tl.int64)
                    tl.atomic_add(z + rows[:, None] * (2 * R) + j[None, :], zi,
                                  mask=(sl == si)[:, None], sem="relaxed")
    if not MX:
        return
    g = tl.reshape(xv.to(tl.float32), (16, PART // 32, 32))
    amax = tl.max(tl.abs(g), axis=2)
    sfv = amax * (1.0 / 448.0)
    bits = sfv.to(tl.int32, bitcast=True)
    ex = (bits >> 23) & 255
    mant = bits & 0x7FFFFF
    e8 = tl.where((mant != 0) & (ex < 254), ex + 1, ex)
    inv = tl.where(e8 != 0, ((254 - e8) << 23).to(tl.float32, bitcast=True), 0.0)
    q = (g * inv[:, :, None]).to(tl.float8e4nv)
    q = tl.reshape(q, (16, PART))
    tl.store(xq + rows.to(tl.int64)[:, None] * H + cols[None, :],
             q.to(tl.uint8, bitcast=True), mask=live)
    rr = blk * 128 + n
    c = p * (PART // 32) + tl.arange(0, PART // 32)
    off = (rr[:, None] // 128) * (N_SF_KT * 512) + (c[None, :] // 4) * 512 + \
        (rr[:, None] % 32) * 16 + ((rr[:, None] % 128) // 32) * 4 + c[None, :] % 4
    tl.store(x_sf + off, e8.to(tl.uint8), mask=live)


@triton.jit
def _shrink_kernel(x, rtok, rslot, blk_expert, blk_mask, nblk, a_gu, a_ss, a_es, z,
                   H: tl.constexpr, KS: tl.constexpr, BK: tl.constexpr, R2: tl.constexpr,
                   E_SH: tl.constexpr = -1):
    """(block b, slot s, K quarter q): z[row] += A_gu[s, expert] (2R x H/KS) . x[row] for the
    block's rows on slot s (tensor-core dots over BK chunks; Q32.32 atomics, exact and
    order independent). Programs whose slot is not in the block exit at once."""
    blk = tl.program_id(0)
    s = tl.program_id(1)
    q = tl.program_id(2)
    if blk >= tl.load(nblk):
        return
    if ((tl.load(blk_mask + blk) >> s) & 1) == 0:
        return
    n = tl.arange(0, 16)
    rows = blk * 16 + n
    tok = tl.load(rtok + rows)
    sl = tl.load(rslot + rows)
    mine = (sl == s) & (tok >= 0)
    ex_id = tl.load(blk_expert + blk)
    if ex_id == E_SH:  # the folded shared expert's Z comes from the quant kernel
        return
    base = s.to(tl.int64) * a_ss + ex_id.to(tl.int64) * a_es
    j = tl.arange(0, R2)
    acc = tl.zeros((16, R2), tl.float32)
    k0 = q * (H // KS)
    for kk in range(0, H // KS, BK):
        cols = k0 + kk + tl.arange(0, BK)
        xv = tl.load(x + tok.to(tl.int64)[:, None] * H + cols[None, :], mask=mine[:, None],
                     other=0.0)
        a = tl.load(a_gu + base + j[:, None] * H + cols[None, :])
        acc += tl.dot(xv, tl.trans(a))
    zi = tl.floor(acc * 4294967296.0 + 0.5).to(tl.int64)
    tl.atomic_add(z + rows[:, None] * R2 + j[None, :], zi, mask=mine[:, None], sem="relaxed")


@triton.jit
def _finalize_kernel(y, row_of, topk_w, out, TOPK: tl.constexpr, KP: tl.constexpr,
                     H: tl.constexpr, BH: tl.constexpr, SHARED: tl.constexpr = False, x=None,
                     gw=None, n_routes=0, GATED: tl.constexpr = True, rscale=1.0):
    """out[t] = sum_k w_tk y[row(t, k)] (+ the folded shared expert's row, sigmoid(x . g)
    gated or not): every route row of the token in one [KP, BH] gather."""
    t = tl.program_id(0)
    hb = tl.program_id(1)
    cols = hb * BH + tl.arange(0, BH)
    cm = cols < H
    kk = tl.arange(0, KP)
    km = kk < TOPK
    rows = tl.load(row_of + t * TOPK + kk, mask=km, other=-1)
    ws = tl.load(topk_w + t * TOPK + kk, mask=km, other=0.0)
    ok = rows >= 0
    v = tl.load(y + rows.to(tl.int64)[:, None] * H + cols[None, :],
                mask=ok[:, None] & cm[None, :], other=0.0)
    acc = tl.sum(ws[:, None] * v.to(tl.float32), 0) * rscale  # routed scaling factor
    if SHARED:  # folded shared expert: (sigmoid(x . g) *) its row
        r = tl.load(row_of + n_routes + t)
        vs = tl.load(y + r.to(tl.int64) * H + cols, mask=cm, other=0.0).to(tl.float32)
        if GATED:
            hh = tl.arange(0, H)
            gx = tl.sum(tl.load(x + t.to(tl.int64) * H + hh).to(tl.float32) *
                        tl.load(gw + hh).to(tl.float32), 0)
            vs = vs * tl.sigmoid(gx)
        acc += vs
    tl.store(out + t.to(tl.int64) * H + cols, acc.to(out.dtype.element_ty), mask=cm)


def max_blocks(tokens, top_k, n_local, shared=False):
    routes = tokens * top_k
    return min(routes, n_local) + routes // 16 + (-(-tokens // 16) if shared else 0)


class Prep:
    """Buffers and launches for one (max tokens, top_k, local experts, H, I) configuration."""

    def __init__(self, tokens, top_k, n_local, local_off, H, I, n_slots, rank, device,
                 part=512, shrink_warps=4, mx=True, shared=False):
        self.T, self.K, self.E, self.off = tokens, top_k, n_local, local_off
        self.H, self.I, self.R, self.n_slots = H, I, rank, n_slots
        self.shared = shared
        self.route_warps = int(__import__("os").environ.get("SGLANG_TEXPERTS_ROUTE_WARPS", "8"))
        # route BLOCK sizes: decode-sized (<= 42 tokens at top-6), larger decode / small
        # prefill, and the whole capacity; each compiled once (precompile_route)
        top = max(16, triton.next_power_of_2(tokens * top_k + (tokens if shared else 0)))
        self.route_blocks = tuple(sorted({min(256, top), min(1024, top), top}))
        self._route_compiled = set()
        self.MB = max_blocks(tokens, top_k, n_local, shared)
        self.rows = self.MB * 16
        self.part = min(part, H)
        self.mx = mx
        self.shrink_warps = shrink_warps
        self.n_parts = H // self.part
        dev = device
        i32 = dict(dtype=torch.int32, device=dev)
        self.scratch = torch.zeros(2 * triton.next_power_of_2(n_local + 2), **i32)
        self.row_of = torch.zeros(tokens * top_k + (tokens if shared else 0), **i32)
        self.rtok = torch.zeros(self.rows, **i32)
        self.rslot = torch.zeros(self.rows, **i32)
        self.rw = torch.zeros(self.rows, dtype=torch.float32, device=dev)
        self.blk_expert = torch.zeros(self.MB, **i32)
        self.nblk = torch.zeros(2, **i32)  # blocks, of which the shared expert's (last)
        self.mask_gu = torch.zeros(self.MB, **i32)
        self.mask_d = torch.zeros(self.MB, **i32)
        self.xq = torch.zeros(self.rows, H, dtype=torch.uint8, device=dev)
        self.x_sf = torch.full((self.MB * 128 * (H // 32),), 127, dtype=torch.uint8, device=dev)
        self.z = torch.zeros(self.rows, 2 * rank, dtype=torch.int64, device=dev)  # Q32.32
        self.aq = torch.zeros(self.rows, I, dtype=torch.uint8 if mx else torch.bfloat16,
                              device=dev)
        self.a_sf = torch.full((self.MB * 128 * (I // 32),), 127, dtype=torch.uint8, device=dev)
        self.zd = torch.zeros(self.rows, rank, dtype=torch.int64, device=dev)  # Q32.32
        self.y = torch.zeros(self.rows, H, dtype=torch.bfloat16, device=dev)
        # folded gate/up shrink (texperts fold_a): BF16 rows, FP32 Z, A-item tables
        self.x_bf = torch.zeros(self.rows, H, dtype=torch.bfloat16, device=dev)
        self.zf = torch.zeros(self.rows, 1, 2 * rank, dtype=torch.float32, device=dev)
        self.a_blk = torch.zeros(self.MB * 4, **i32)
        self.a_s = torch.zeros(self.MB * 8, **i32)
        self.n_a = torch.zeros(1, **i32)
        self.npairs = torch.zeros(self.MB, **i32)
        self.zcnt = torch.zeros(self.MB, **i32)
        # split-K workspace / counters of the texperts kernels (sequential on one stream)
        self.ws = torch.zeros(2048 * 128 * 16, dtype=torch.float32, device=dev)
        self.cnt = torch.zeros(self.MB * max(2 * I, H) // 128, dtype=torch.int32, device=dev)

    def route_block(self, routes):
        return next(b for b in self.route_blocks if b >= routes)

    def route(self, topk_ids, topk_w, slots, fold=False, zero_zd=False):
        n = topk_ids.numel()
        assert n <= self.T * self.K
        n_tok = topk_ids.shape[0] if self.shared else 0
        if (fold, zero_zd) not in self._route_compiled:
            self.precompile_route(topk_ids, topk_w, slots, fold, zero_zd)
        self._launch_route(topk_ids, topk_w, slots, fold, zero_zd, n, n_tok,
                           self.route_block(n + n_tok))

    def precompile_route(self, topk_ids, topk_w, slots, fold, zero_zd):
        """Compile every BLOCK size of this (fold, zero_zd) variant now, for a route count
        divisible by 16 (0) and one that is not (3, at most the routes of any call), by
        launching each on those first routes: the real launch that follows rewrites every
        table they touch. Skipped under graph capture, where the launches would be recorded."""
        if torch.cuda.is_current_stream_capturing():
            return
        for block in self.route_blocks:
            for n in (0, min(3, topk_ids.numel())):
                self._launch_route(topk_ids, topk_w, slots, fold, zero_zd, n, 0, block)
        self._route_compiled.add((fold, zero_zd))

    def _launch_route(self, topk_ids, topk_w, slots, fold, zero_zd, n, n_tok, BLOCK):
        zrows = 64
        _route_kernel[(1 + triton.cdiv(self.rows, zrows),)](
            topk_ids.reshape(-1), topk_w.reshape(-1).float(), slots, self.scratch, self.row_of,
            self.rtok, self.rslot, self.rw, self.blk_expert, self.mask_gu, self.mask_d,
            self.nblk, self.z, self.a_blk, self.a_s, self.n_a, self.npairs, self.zcnt, n, self.K,
            self.off, E=self.E, ROWS_MAX=self.rows, MB=self.MB,
            BLOCK=BLOCK, EB=triton.next_power_of_2(self.E + 2),
            RB=triton.next_power_of_2(self.rows), MBB=triton.next_power_of_2(self.MB), SPT_D=2,
            ZW=2 * self.R, ZROWS=zrows, SL=self.n_slots, FOLD=fold, SHARED=self.shared,
            n_tok=n_tok, E_TOT=self.E + (1 if self.shared else 0), zd=self.zd,
            ZERO_ZD=zero_zd, num_warps=self.route_warps)

    def quant_shrink(self, x, a_gu, shrink=True, ks=4, bk=256, keep_bf=False, a_shared=None):
        """x [T, H] BF16; a_gu [slots, E or 1 (shared-outer), 2R, H] BF16 (gate rows first),
        each [2R, H] matrix contiguous."""
        _quant_kernel[(self.MB, self.n_parts)](
            x, self.rtok, self.nblk, self.xq, self.x_sf, self.zd, self.x_bf, H=self.H,
            PART=self.part, N_SF_KT=(self.H // 32 + 3) // 4, R=self.R, BF=keep_bf or not self.mx,
            MX=self.mx, SHARED=a_shared is not None, blk_expert=self.blk_expert,
            blk_mask=self.mask_gu, rslot=self.rslot,
            a_s=a_shared if a_shared is not None else self.x_bf,
            a_ss=a_shared.stride(0) if a_shared is not None else 0, z=self.z, E_SH=self.E,
            SL=self.n_slots, SKIP_PAD=_SKIP_PAD)
        if shrink:
            assert a_gu.stride(3) == 1 and a_gu.stride(2) == a_gu.shape[3]
            a_es = a_gu.stride(1) if a_gu.shape[1] > 1 else 0
            _shrink_kernel[(self.MB, self.n_slots, ks)](
                x, self.rtok, self.rslot, self.blk_expert, self.mask_gu, self.nblk, a_gu,
                a_gu.stride(0), a_es, self.z, H=self.H, KS=ks, BK=bk, R2=2 * self.R,
                E_SH=self.E if a_shared is not None else -1, num_warps=self.shrink_warps)

    def finalize(self, n_tokens, topk_w, out, x=None, gate_w=None, shared_plain=False,
                 rscale=1.0):
        """out[t] = sum_k w_tk y[row(t, k)] (+ sigmoid(x_t . gate_w) or 1 times the shared
        expert's row when it is folded)."""
        BH = 1024
        sh = gate_w is not None or shared_plain
        _finalize_kernel[(n_tokens, triton.cdiv(self.H, BH))](
            self.y, self.row_of, topk_w.reshape(-1), out, TOPK=self.K,
            KP=max(2, triton.next_power_of_2(self.K)), H=self.H, BH=BH,
            SHARED=sh, x=x if x is not None else out, gw=gate_w if gate_w is not None else out,
            n_routes=n_tokens * self.K, GATED=gate_w is not None, rscale=float(rscale))
