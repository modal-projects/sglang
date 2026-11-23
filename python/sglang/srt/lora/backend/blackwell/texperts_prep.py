"""Routing, activation quantization and finalize around the fused routed-expert kernels
(texperts; the gate/up LoRA-A shrink runs inside them). All buffers are preallocated: every launch is CUDA-graph safe.

Rows are routed (token, k) pairs owned by this rank, grouped by local expert into 16-row
blocks: expert e's rows start at 16 * (number of blocks of experts < e).
"""

import torch
import triton
import triton.language as tl

NT = tl.constexpr(16)


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
                  N_SF_KT: tl.constexpr, R: tl.constexpr, BF: tl.constexpr):
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
    live = (tok >= 0)[:, None]
    if BF:  # the permuted BF16 rows (folded LoRA-A shrink; the BF16-base GEMM input)
        tl.store(x_bf + rows.to(tl.int64)[:, None] * H + cols[None, :], xv, mask=live)
    if p == 0:
        jd = tl.arange(0, R)
        tl.store(zd + rows[:, None] * R + jd[None, :], tl.zeros((16, R), tl.int64))
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
def _finalize_kernel(y, row_of, topk_w, out, TOPK: tl.constexpr, KP: tl.constexpr,
                     H: tl.constexpr, BH: tl.constexpr, SHARED: tl.constexpr = False,
                     n_routes=0, rscale=1.0):
    """out[t] = sum_k w_tk y[row(t, k)] (+ the folded shared expert's row): every route row
    of the token in one [KP, BH] gather."""
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
    if SHARED:  # folded shared expert: its row
        r = tl.load(row_of + n_routes + t)
        acc += tl.load(y + r.to(tl.int64) * H + cols, mask=cm, other=0.0).to(tl.float32)
    tl.store(out + t.to(tl.int64) * H + cols, acc.to(out.dtype.element_ty), mask=cm)


def max_blocks(tokens, top_k, n_local, shared=False):
    routes = tokens * top_k
    return min(routes, n_local) + routes // 16 + (-(-tokens // 16) if shared else 0)


class Prep:
    """Buffers and launches for one (max tokens, top_k, local experts, H, I) configuration."""

    def __init__(self, tokens, top_k, n_local, local_off, H, I, n_slots, rank, device,
                 part=512, mx=True, shared=False):
        self.T, self.K, self.E, self.off = tokens, top_k, n_local, local_off
        self.H, self.I, self.R, self.n_slots = H, I, rank, n_slots
        self.shared = shared
        self.route_warps = 8
        # route BLOCK sizes: decode-sized (<= 42 tokens at top-6), larger decode / small
        # prefill, and the whole capacity; each compiled once (precompile_route)
        top = max(16, triton.next_power_of_2(tokens * top_k + (tokens if shared else 0)))
        self.route_blocks = tuple(sorted({min(256, top), min(1024, top), top}))
        self._route_compiled = set()
        self.MB = max_blocks(tokens, top_k, n_local, shared)
        self.rows = self.MB * 16
        self.part = min(part, H)
        self.mx = mx
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
        self.a_blk = torch.zeros(self.MB * 4, **i32)
        self.a_s = torch.zeros(self.MB * 8, **i32)
        self.n_a = torch.zeros(1, **i32)
        self.npairs = torch.zeros(self.MB, **i32)
        self.zcnt = torch.zeros(self.MB, **i32)

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

    def quant(self, x):
        """x [T, H] BF16 -> the routed rows' MXFP8 (xq, x_sf) and BF16 copies (x_bf, the
        in-kernel gate/up LoRA-A shrink's input); clears the rows' down-LoRA Z."""
        assert self.mx
        _quant_kernel[(self.MB, self.n_parts)](
            x, self.rtok, self.nblk, self.xq, self.x_sf, self.zd, self.x_bf, H=self.H,
            PART=self.part, N_SF_KT=(self.H // 32 + 3) // 4, R=self.R, BF=True)

    def finalize(self, n_tokens, topk_w, out, shared=False, rscale=1.0):
        """out[t] = sum_k w_tk y[row(t, k)] (+ the folded shared expert's row)."""
        BH = 1024
        _finalize_kernel[(n_tokens, triton.cdiv(self.H, BH))](
            self.y, self.row_of, topk_w.reshape(-1), out, TOPK=self.K,
            KP=max(2, triton.next_power_of_2(self.K)), H=self.H, BH=BH, SHARED=shared,
            n_routes=n_tokens * self.K, rscale=float(rscale))
