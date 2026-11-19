"""T-tiled fused projection with multi-LoRA for decode: y[T, N] = x W^T + sum scale * B (A x).

One skeleton for a BF16 base (Qwen) and an MXFP8 base (DeepSeek), with per-CTA resources
bounded as T grows (T <= 256):

  * Weights stream once per CTA; the MMA N is the whole (padded) token count n_mma.
  * LoRA-B runs as BF16 K-tiles appended to each M-tile's K sequence. Its N-side operand
    (Zblk, [n_mma, 64] per K-tile) is written into the activation half of the very ring stage
    that holds that K-tile's B weights: a LoRA stage loads no activations, so LoRA-B needs no
    SMEM of its own at any T. Per-token Z rows are prepared once (n_mma x 32 BF16).
  * Epilogue in 32-token chunks: TMEM -> registers -> SMEM staging -> TMA store.
  * Split-K over a cluster, reduced over DSMEM (reduce-scatter by token chunk, rank order).

Grid = [LoRA-A role | base role], launched in clusters of C CTAs (C = ksplit when ksplit
is 3, else 4; base split groups sit inside a cluster). LoRA-A: each cluster holds one
128-row block of A_all and C of its a_splits K-ranges. Every CTA computes its Z^T partial,
parks it in its drained ring, and the cluster reduces the C partials over DSMEM (ranks in
order). Each rank then writes only the compact rows each token needs, its own slot's
z[qi, t, slice*32 + r] (FP32, qi = which group of C K-ranges), and posts a ready count.
Base CTAs that hold LoRA tiles wait for it (wrap-safe, trapping wait), sum the q = a_splits/C
compact planes and build their Zblk. A_all rows are slot * (slices*32) + slice*32 + r
(SGLang's pool order); Bm is [N, slots*32].
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from .ptx import _asm, atomic_add_acq_rel, dsmem_load1, wait_count, global_ns, warp_sum

SF_VEC = 32
SMEM_LIMIT = 232448
CHUNK = 16  # tokens per epilogue chunk; chunks alternate between 2 warpgroups
WGS = 2
TRS = 16  # trace slots per CTA


def n_mma_for(tokens):
    for n in (16, 32, 64, 128, 256):
        if tokens <= n:
            return n
    raise ValueError(f"T={tokens} > 256")


def load_cg4(ptr):
    f32 = ir.F32Type.get()
    rr = llvm.inline_asm(llvm.StructType.get_literal([f32] * 4), [ptr.toint().ir_value()],
                         "ld.global.cg.v4.f32 {$0,$1,$2,$3},[$4];", "=f,=f,=f,=f,l,~{memory}",
                         has_side_effects=True, is_align_stack=False, asm_dialect=0)
    return [cutlass.Float32(llvm.extractvalue(f32, rr, [j])) for j in range(4)]


def fence_proxy_async():
    _asm(None, [], "fence.proxy.async.shared::cta;", "~{memory}")


def st_shared_v4(addr, w):
    _asm(None, [addr.ir_value()] + [v.ir_value() for v in w],
         "st.shared.v4.b32 [$0],{$1,$2,$3,$4};", "r,r,r,r,r,~{memory}")


def ld_shared_v4(addr):
    i32 = ir.IntegerType.get_signless(32)
    rr = llvm.inline_asm(llvm.StructType.get_literal([i32] * 4), [addr.ir_value()],
                         "ld.shared.v4.b32 {$0,$1,$2,$3},[$4];", "=r,=r,=r,=r,r,~{memory}",
                         has_side_effects=True, is_align_stack=False, asm_dialect=0)
    return [cutlass.Int32(llvm.extractvalue(i32, rr, [j])) for j in range(4)]


def ld_global_cg_v4(ptr):
    i32 = ir.IntegerType.get_signless(32)
    rr = llvm.inline_asm(llvm.StructType.get_literal([i32] * 4), [ptr.toint().ir_value()],
                         "ld.global.cg.v4.b32 {$0,$1,$2,$3},[$4];", "=r,=r,=r,=r,l,~{memory}",
                         has_side_effects=True, is_align_stack=False, asm_dialect=0)
    return [cutlass.Int32(llvm.extractvalue(i32, rr, [j])) for j in range(4)]


def st_global_v4(ptr, w):
    _asm(None, [ptr.toint().ir_value()] + [v.ir_value() for v in w],
         "st.global.v4.b32 [$0],{$1,$2,$3,$4};", "l,r,r,r,r,~{memory}")


def dsmem_load4(addr, rank):
    """4 FP32 from CTA ``rank`` of the cluster at this CTA's shared address ``addr``."""
    i32, f32 = ir.IntegerType.get_signless(32), ir.F32Type.get()
    remote = _asm(i32, [addr.ir_value(), rank.ir_value()],
                  "mapa.shared::cluster.u32 $0,$1,$2;", "=r,r,r")
    rr = llvm.inline_asm(llvm.StructType.get_literal([f32] * 4), [remote],
                         "ld.shared::cluster.v4.f32 {$0,$1,$2,$3},[$4];",
                         "=f,=f,=f,=f,r,~{memory}", has_side_effects=True,
                         is_align_stack=False, asm_dialect=0)
    return [cutlass.Float32(llvm.extractvalue(f32, rr, [j])) for j in range(4)]


def st_global_v4f(ptr, v):
    _asm(None, [ptr.toint().ir_value()] + [x.ir_value() for x in v],
         "st.global.v4.f32 [$0],{$1,$2,$3,$4};", "l,f,f,f,f,~{memory}")


def cluster_size_for(ksplit):
    return ksplit if ksplit == 3 else 4


def _pure(ret, args, text, cons):
    """Register-only inline asm (no memory, no side effects): free to schedule. Only for
    warp-uniform call sites when the instruction synchronizes lanes (shfl.sync)."""
    return llvm.inline_asm(ret, args, text, cons, has_side_effects=False, is_align_stack=False,
                           asm_dialect=0)


def rsqrt_approx(x):
    return cutlass.Float32(_pure(ir.F32Type.get(), [x.ir_value()], "rsqrt.approx.ftz.f32 $0,$1;",
                                 "=f,f"))


def st_shared_i32(addr, v):
    _asm(None, [addr.ir_value(), v.ir_value()], "st.shared.b32 [$0],$1;", "r,r,~{memory}")


def ld_shared_i32(addr):
    return cutlass.Int32(_asm(ir.IntegerType.get_signless(32), [addr.ir_value()],
                              "ld.shared.b32 $0,[$1];", "=r,r,~{memory}"))


def st_shared_f32(addr, v):
    _asm(None, [addr.ir_value(), v.ir_value()], "st.shared.f32 [$0],$1;", "r,f,~{memory}")


def ld_shared_v4f(addr):
    f32 = ir.F32Type.get()
    rr = llvm.inline_asm(llvm.StructType.get_literal([f32] * 4), [addr.ir_value()],
                         "ld.shared.v4.f32 {$0,$1,$2,$3},[$4];", "=f,=f,=f,=f,r,~{memory}",
                         has_side_effects=True, is_align_stack=False, asm_dialect=0)
    return [cutlass.Float32(llvm.extractvalue(f32, rr, [j])) for j in range(4)]


def _sync(ret, args, text, cons):
    """Lane-synchronizing register asm (shfl.sync): side-effecting so LLVM never sinks it
    into a lane-divergent branch, but no memory clobber, so loads still move around it."""
    return llvm.inline_asm(ret, args, text, cons, has_side_effects=True, is_align_stack=False,
                           asm_dialect=0)


def shfl_xor_f32(v, lanes):
    return cutlass.Float32(_sync(ir.F32Type.get(), [v.ir_value(), cutlass.Int32(lanes).ir_value()],
                                 "shfl.sync.bfly.b32 $0,$1,$2,31,-1;", "=f,f,r"))


def shfl_idx_i32(v, src):
    return cutlass.Int32(_sync(ir.IntegerType.get_signless(32),
                               [v.ir_value(), cutlass.Int32(src).ir_value()],
                               "shfl.sync.idx.b32 $0,$1,$2,31,-1;", "=r,r,r"))


def load_nc4(ptr):
    f32 = ir.F32Type.get()
    rr = llvm.inline_asm(llvm.StructType.get_literal([f32] * 4), [ptr.toint().ir_value()],
                         "ld.global.nc.v4.f32 {$0,$1,$2,$3},[$4];", "=f,=f,=f,=f,l",
                         has_side_effects=False, is_align_stack=False, asm_dialect=0)
    return [cutlass.Float32(llvm.extractvalue(f32, rr, [j])) for j in range(4)]


def pack_bf16x2(lo, hi):
    return cutlass.Int32(_pure(ir.IntegerType.get_signless(32), [hi.ir_value(), lo.ir_value()],
                               "cvt.rn.bf16x2.f32 $0,$1,$2;", "=r,f,f"))


def st_global_v2(ptr, w):
    _asm(None, [ptr.toint().ir_value()] + [x.ir_value() for x in w],
         "st.global.v2.b32 [$0],{$1,$2};", "l,r,r,~{memory}")


def bulk_push(dst_local, src, nbytes, bar_local, rank):
    """cp.async.bulk from this CTA's SMEM `src` to cluster CTA `rank` at the address that
    corresponds to `dst_local` there, completing `nbytes` on that CTA's mbarrier."""
    i32 = ir.IntegerType.get_signless(32)
    dst = _asm(i32, [dst_local.ir_value(), rank.ir_value()], "mapa.shared::cluster.u32 $0,$1,$2;",
               "=r,r,r")
    bar = _asm(i32, [bar_local.ir_value(), rank.ir_value()], "mapa.shared::cluster.u32 $0,$1,$2;",
               "=r,r,r")
    _asm(None, [dst, src.ir_value(), cutlass.Int32(nbytes).ir_value(), bar],
         "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
         "r,r,r,r,~{memory}")


def named_barrier_n(bar_id, threads):
    _asm(None, [], f"bar.sync {bar_id},{threads};", "~{memory}")


def named_barrier_arrive(bar_id, threads):
    _asm(None, [], f"bar.arrive {bar_id},{threads};", "~{memory}")


def named_barrier_r(bar_reg, threads):
    """bar.sync with the barrier id in a register (per-warpgroup barriers)."""
    _asm(None, [cutlass.Int32(bar_reg).ir_value()], f"bar.sync $0,{threads};", "r,~{memory}")


def named_barrier(bar_id, threads):
    _asm(None, [], f"bar.sync {bar_id},{threads};", "~{memory}")


class TLinear:
    """See the module docstring. Extra layer options:

    pairing   "none": one 128-row accumulator per CTA (tile c).
              "halves": two accumulators, tiles c and c + m_ctas (gate | up halves of a merged
                        gate_up weight; their LoRA slices differ, so each LoRA K-tile is two
                        ring uses, one per accumulator).
              "adjacent": two accumulators, tiles 2c and 2c + 1 (a 256-wide head; one slice,
                        so a LoRA K-tile loads both B tiles and shares one Zblk).
    epi       "linear": y[:, tile] = acc (every accumulator).
              "silu_mul": y[:, c] = silu(acc0) * acc1 ("halves").
              "resnorm": h = res + acc is written back to res, y = h * gamma (the next
                         kernel's activation), and ss[t, c] = sum over this tile of h^2.
    in_rstd   the activation is h * gamma from a "resnorm" kernel: every accumulator column t
              is scaled by rstd[t] = rsqrt(sum_c ss_in[t, c] / K + eps) (exact by linearity,
              including the LoRA-B part).
    """

    def __init__(self, n_out, k, tokens, slice_rows, base="bf16", ksplit=1, a_splits=4,
                 stages=None, n_slots=8, early_trigger=True, trace=False, pairing="none",
                 epi="linear", in_rstd=False, eps=1e-6, cluster=None, qkv=None):
        self.trace = trace
        assert base in ("bf16", "mxfp8")
        self.mx = base == "mxfp8"
        self.N, self.K, self.T = n_out, k, tokens
        self.n_mma = n_mma_for(tokens)
        self.chunk = min(CHUNK, self.n_mma)
        self.n_chunks = self.n_mma // self.chunk
        self.slice_rows = tuple(slice_rows)
        assert self.slice_rows[-1] == n_out and all(b % 128 == 0 for b in self.slice_rows[:-1])
        self.slices = len(self.slice_rows)
        self.n_slots = n_slots
        self.R = self.slices * n_slots * 32
        assert n_out % 128 == 0
        self.row_tiles = n_out // 128
        self.pairing = pairing
        self.nacc = 1 if pairing == "none" else 2
        assert not (self.mx and self.nacc > 1)
        # BF16 K-tile: 64 elements (128-byte rows, SW128). Two accumulators at T=256 would
        # make a 64 KB stage (2 ring stages); they use 32-element tiles (64-byte rows, SW64)
        # so the ring keeps 5-6 stages.
        self.kb = 32 if (not self.mx and self.nacc == 2 and self.n_mma == 256) else 64
        self.row_b = self.kb * 2  # bytes of one activation / Zblk row per stage
        self.lora_tiles = (n_slots * 32 + self.kb - 1) // self.kb
        self.ke = 128 if self.mx else self.kb  # elements per base K-tile
        if pairing == "halves":
            assert self.row_tiles % 2 == 0
            self.m_tiles = self.row_tiles // 2
            self.split_lora = True  # gate and up: different slices
        elif pairing == "adjacent":
            assert self.row_tiles % 2 == 0
            assert all(b % 256 == 0 for b in self.slice_rows[:-1])
            self.m_tiles = self.row_tiles // 2
            self.split_lora = False
        else:
            self.m_tiles = self.row_tiles
            self.split_lora = False
        self.epi = epi
        assert epi in ("linear", "silu_mul", "resnorm", "qkv")
        assert epi != "silu_mul" or pairing == "halves"
        assert epi != "resnorm" or pairing == "none"
        self.units = {"linear": self.nacc, "silu_mul": 1, "resnorm": 2, "qkv": 0}[epi]
        self.mrope = False
        if epi == "qkv":  # qkv = (q heads, kv heads, head dim, rotary dim, gated query[, mrope])
            self.qh, self.kvh, self.hd, self.rot, self.gated = qkv[:5]
            # multi-axis RoPE: positions [3, T]; rotary pair i uses axis axis_map[i]
            self.mrope = len(qkv) > 5 and bool(qkv[5])
            assert self.hd == 128 * self.nacc and self.rot in (64, 128) and self.rot <= self.hd
            self.qheads = self.qh * (2 if self.gated else 1)  # stored q (+gate) heads
            assert (self.qheads + 2 * self.kvh) * self.hd == n_out
        self.in_rstd = in_rstd
        self.eps = eps
        if in_rstd:
            assert k % 512 == 0  # ss_in rows: k/128 partials, read 4 at a time
        self.k_tiles = (k + self.ke - 1) // self.ke
        self.lora_uses = self.lora_tiles * (self.nacc if self.split_lora else 1)
        self.total = self.k_tiles + self.lora_uses
        self.ksplit = ksplit
        assert ksplit in (1, 2, 3, 4)
        self.C = cluster or cluster_size_for(ksplit)
        assert self.C % ksplit == 0
        self.per = (self.total + ksplit - 1) // ksplit
        assert (ksplit - 1) * self.per < self.total
        if self.nacc > 1:  # every split starts with a base tile (initializes both accumulators)
            assert (ksplit - 1) * self.per < self.k_tiles
        self.a_mtiles = (self.R + 127) // 128
        self.a_ktiles = (k + self.kb - 1) // self.kb
        assert a_splits % self.C == 0, (a_splits, self.C)
        self.a_splits = a_splits
        self.q = a_splits // self.C  # compact Z planes the base CTAs sum
        self.a_per = (self.a_ktiles + a_splits - 1) // a_splits
        assert (a_splits - 1) * self.a_per < self.a_ktiles  # every A CTA has K-tiles
        self.inflight = max(1, 8 // self.q)  # Z-build tasks with loads in flight (64 regs)
        self.a_ctas = self.a_mtiles * a_splits
        self.a_real = self.a_ctas
        self.b_ctas = (self.m_tiles * ksplit + self.C - 1) // self.C * self.C
        self.grid = self.a_ctas + self.b_ctas
        self.early_trigger = early_trigger
        self.w_tile = 128 * self.kb * 2  # one 128-row weight K-tile (BF16; FP8 is 128 x 128 B)
        self.stage_bytes = self.nacc * self.w_tile + self.n_mma * self.row_b + \
            (1024 if self.mx else 0)
        # zrow: per-token Z rows for the slices this CTA's LoRA tiles use
        self.zw = self.nacc if self.split_lora else 1
        zrow_bytes = self.n_mma * 64 * self.zw
        fixed0 = zrow_bytes + self.n_mma * 4 + 5120
        self.obuf = 2
        base_fixed = fixed0 + WGS * self.obuf * self.chunk * 128 * 2
        if (SMEM_LIMIT - base_fixed) // self.stage_bytes < 4:
            self.obuf = 1
            base_fixed = fixed0 + WGS * self.chunk * 128 * 2
        # Zblk: a dedicated region (zero-filled early, values scattered when Z is ready) when
        # it still leaves >= 4 ring stages (one accumulator only); else built in the drained
        # LoRA stages.
        self.zbytes = self.n_mma * self.row_b * self.lora_tiles
        fit_ded = (SMEM_LIMIT - base_fixed - self.zbytes) // self.stage_bytes
        self.zded = self.nacc == 1 and self.kb == 64 and (stages is None and fit_ded >= 4 or
                                        (stages is not None and stages <= fit_ded))
        self.fixed_bytes = base_fixed + (self.zbytes if self.zded else 0)
        fit = (SMEM_LIMIT - self.fixed_bytes) // self.stage_bytes
        self.stages = min(8, fit) if stages is None else stages
        assert self.stages >= 2, self.stages
        assert self.stages * self.stage_bytes + self.fixed_bytes <= SMEM_LIMIT
        # FP32 tiles parked in the drained ring for the cluster reductions
        ring = self.stages * (self.nacc * self.w_tile + self.n_mma * self.row_b)
        assert ring >= 128 * self.n_mma * 4  # LoRA-A
        if ksplit > 1:
            assert ring >= self.nacc * 128 * self.n_mma * 4, "split-K tile does not fit"
        # QKV epilogue scratch: one FP32 [chunk, head] tile after the parked split-K tiles
        # split-K reduction: each rank pushes its FP32 partial of chunk c to the chunk's owner
        # (rank c % ksplit) with a bulk DSMEM copy into an inbox after the parked tiles, when
        # it fits; else owners pull with scalar DSMEM loads
        red_bytes = self.nacc * 128 * self.n_mma * 4 if ksplit > 1 else 0
        self.tile_bytes = 128 * self.chunk * 4
        self.cpr = (self.n_chunks + ksplit - 1) // ksplit  # chunks a rank owns (at most)
        inbox = ksplit * self.cpr * self.nacc * self.tile_bytes if ksplit > 1 else 0
        self.bulk = ksplit > 1 and red_bytes + inbox <= ring
        self.inbox_off = red_bytes
        self.qs_off = red_bytes + (inbox if self.bulk else 0)
        if epi == "qkv":
            assert self.qs_off + WGS * self.chunk * self.hd * 4 <= ring, \
                "QKV scratch does not fit"
        acc_cols = self.nacc * self.n_mma + (32 if self.mx else 0)
        assert acc_cols <= 512
        self.tmem_cols = max(32, 1 << (acc_cols - 1).bit_length())
        self.sf_col = self.n_mma
        self.tiler = (128, self.n_mma, self.ke)
        self.tiler16 = (128, self.n_mma, self.kb)
        self.tiler_sfb = (128, max(128, self.n_mma), 128)

    def __repr__(self):
        return (f"TLinear_{'mx' if self.mx else 'bf'}_N{self.N}_K{self.K}_T{self.T}"
                f"_sl{self.slices}_{self.pairing}_{self.epi}{'_rs' if self.in_rstd else ''}"
                f"_S{self.ksplit}_C{self.C}_A{self.a_splits}_st{self.stages}_L{self.n_slots}")

    def tile_of(self, c, a):
        """Weight row tile of accumulator a in CTA column c."""
        if self.pairing == "halves":
            return c + a * self.m_tiles
        if self.pairing == "adjacent":
            return c * 2 + a
        return c

    @cute.jit
    def __call__(self, w: cute.Tensor, w_sf: cute.Tensor, x_q: cute.Tensor, x_sf: cute.Tensor,
                 x_bf: cute.Tensor, a_all: cute.Tensor, bm: cute.Tensor, y: cute.Tensor,
                 z: cute.Tensor, slots: cute.Tensor, ranks: cute.Tensor, scales: cute.Tensor,
                 ready: cute.Tensor, epoch: cute.Tensor, tr: cute.Tensor, res: cute.Tensor,
                 gamma: cute.Tensor, ss_out: cute.Tensor, ss_in: cute.Tensor,
                 qn_w: cute.Tensor, kn_w: cute.Tensor, cs: cute.Tensor, pos: cute.Tensor,
                 kc: cute.Tensor, vc: cute.Tensor, loc: cute.Tensor, qo: cute.Tensor,
                 gate_o: cute.Tensor, axis: cute.Tensor, stream):
        f8, e8, bf = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, cutlass.BFloat16
        major = tcgen05.OperandMajorMode.K
        S = self.stages
        one4 = (1, 1, 1, 1)
        sl = lambda l: cute.slice_(l, (None, None, None, 0))
        mma16 = sm100_utils.make_trivial_tiled_mma(bf, bf, major, major, cutlass.Float32,
                                                   tcgen05.CtaGroup.ONE, self.tiler16[:2])
        sA16_l = sm100_utils.make_smem_layout_a(mma16, self.tiler16, bf, S * self.nacc)
        sB16_l = sm100_utils.make_smem_layout_b(mma16, self.tiler16, bf, S)
        op_a = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), mma16.thr_id)
        op_b = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), mma16.thr_id)
        tma_bm, tbm = cute.nvgpu.make_tiled_tma_atom_A(op_a, bm, sl(sA16_l), self.tiler16, mma16,
                                                       one4)
        tma_aa, taa = cute.nvgpu.make_tiled_tma_atom_A(op_a, a_all, sl(sA16_l), self.tiler16,
                                                       mma16, one4)
        tma_xb, txb = cute.nvgpu.make_tiled_tma_atom_B(op_b, x_bf, sl(sB16_l), self.tiler16,
                                                       mma16, one4)
        if cutlass.const_expr(self.mx):
            a = cute.make_tensor(cute.recast_ptr(w.iterator, dtype=f8), w.layout)
            b = cute.make_tensor(cute.recast_ptr(x_q.iterator, dtype=f8), x_q.layout)
            sfa = cute.make_tensor(cute.recast_ptr(w_sf.iterator, dtype=e8),
                                   blockscaled_utils.tile_atom_to_shape_SF(a.shape, SF_VEC))
            sfb = cute.make_tensor(cute.recast_ptr(x_sf.iterator, dtype=e8),
                                   blockscaled_utils.tile_atom_to_shape_SF(b.shape, SF_VEC))
            mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                f8, major, major, e8, SF_VEC, tcgen05.CtaGroup.ONE, self.tiler[:2])
            mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
                f8, major, major, e8, SF_VEC, tcgen05.CtaGroup.ONE, self.tiler_sfb[:2])
            sA_l = sm100_utils.make_smem_layout_a(mma, self.tiler, f8, S)
            sB_l = sm100_utils.make_smem_layout_b(mma, self.tiler, f8, S)
            sSFA_l = blockscaled_utils.make_smem_layout_sfa(mma, self.tiler, SF_VEC, S)
            sSFB_l = blockscaled_utils.make_smem_layout_sfb(mma, self.tiler, SF_VEC, S)
            op_sfb = sm100_utils.cluster_shape_to_tma_atom_SFB((1, 1), mma.thr_id)
            tma_a, ta = cute.nvgpu.make_tiled_tma_atom_A(op_a, a, sl(sA_l), self.tiler, mma, one4)
            tma_b, tb = cute.nvgpu.make_tiled_tma_atom_B(op_b, b, sl(sB_l), self.tiler, mma, one4)
            tma_sfa, tsfa = cute.nvgpu.make_tiled_tma_atom_A(
                op_a, sfa, sl(sSFA_l), self.tiler, mma, one4, internal_type=cutlass.Int16)
            tma_sfb, tsfb = cute.nvgpu.make_tiled_tma_atom_B(
                op_sfb, sfb, sl(sSFB_l), self.tiler_sfb, mma_sfb, one4,
                internal_type=cutlass.Int16)
            self.w_bytes = cute.size_in_bytes(f8, sl(sA_l)) + cute.size_in_bytes(e8, sl(sSFA_l))
            self.x_bytes = cute.size_in_bytes(f8, sl(sB_l)) + cute.size_in_bytes(e8, sl(sSFB_l))
            sfa_cos, sfb_cos = cute.cosize(sSFA_l), cute.cosize(sSFB_l)
        else:  # BF16 base: the base tiles use the same operands as the LoRA/A tiles
            mma, mma_sfb, sA_l, sB_l = mma16, mma16, sA16_l, sB16_l
            sSFA_l = sSFB_l = cute.make_layout(128)  # unused placeholders
            tma_a, ta = cute.nvgpu.make_tiled_tma_atom_A(op_a, w, sl(sA16_l), self.tiler16,
                                                         mma16, one4)
            tma_b, tb, tma_sfa, tsfa, tma_sfb, tsfb = tma_xb, txb, tma_a, ta, tma_xb, txb
            self.w_bytes = cute.size_in_bytes(bf, sl(sA16_l))
            self.x_bytes = cute.size_in_bytes(bf, sl(sB16_l))
            sfa_cos, sfb_cos = 128, 128
        self.bm_bytes = cute.size_in_bytes(bf, sl(sA16_l))
        self.xb_bytes = cute.size_in_bytes(bf, sl(sB16_l))
        # output staging [chunk tokens x 128 features] BF16, TMA-stored
        sO_l = cute.make_layout((self.chunk, 128, WGS * self.obuf),
                                stride=(128, 1, self.chunk * 128))
        tma_y, ty = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), y,
                                                cute.slice_(sO_l, (None, None, 0)),
                                                (self.chunk, 128))
        tma_r, trr = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), res,
                                                 cute.slice_(sO_l, (None, None, 0)),
                                                 (self.chunk, 128))

        epi_qkv = self.epi == "qkv"

        @cute.struct
        class Storage:
            full: cute.struct.MemRange[cutlass.Int64, S]
            empty: cute.struct.MemRange[cutlass.Int64, S]
            zready: cute.struct.MemRange[cutlass.Int64, S]
            zfree: cute.struct.MemRange[cutlass.Int64, S]
            rbar: cutlass.Int64
            acc_full: cutlass.Int64
            tmem_holder: cutlass.Int32
            sA: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, S * self.nacc * self.w_tile],
                                  1024]
            sB: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, S * self.n_mma * self.row_b],
                                  1024]
            sSFA: cute.struct.Align[cute.struct.MemRange[e8, sfa_cos], 128]
            sSFB: cute.struct.Align[cute.struct.MemRange[e8, sfb_cos], 128]
            zrow: cute.struct.Align[cute.struct.MemRange[bf, self.n_mma * 32 * self.zw], 128]
            tslot: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.n_mma], 16]
            srstd: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.n_mma], 16]
            qconst: cute.struct.Align[cute.struct.MemRange[
                cutlass.Int32, 4 * self.n_mma + 2 * (self.hd if epi_qkv else 0) + 128], 16]
            sZd: cute.struct.Align[cute.struct.MemRange[cutlass.Int8,
                                                        self.zbytes if self.zded else 1024], 1024]
            sO: cute.struct.Align[cute.struct.MemRange[bf, WGS * self.obuf * self.chunk * 128],
                                  1024]

        self.storage = Storage
        self.kernel(mma, mma_sfb, mma16, tma_a, ta, tma_b, tb, tma_sfa, tsfa, tma_sfb, tsfb,
                    tma_bm, tbm, tma_aa, taa, tma_xb, txb, tma_y, ty, tma_r, trr, z, slots,
                    ranks, scales, ready, epoch, tr, res, gamma, ss_out, ss_in, qn_w, kn_w, cs,
                    pos, kc, vc, loc, qo, gate_o, axis, sA_l, sB_l,
                    sSFA_l, sSFB_l, sA16_l, sB16_l, sO_l).launch(
            grid=(self.grid, 1, 1), block=(256, 1, 1), cluster=(self.C, 1, 1),
            smem=Storage.size_in_bytes(), stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(self, mma: cute.TiledMma, mma_sfb: cute.TiledMma, mma16: cute.TiledMma,
               tma_a: cute.CopyAtom, ta: cute.Tensor, tma_b: cute.CopyAtom, tb: cute.Tensor,
               tma_sfa: cute.CopyAtom, tsfa: cute.Tensor, tma_sfb: cute.CopyAtom,
               tsfb: cute.Tensor, tma_bm: cute.CopyAtom, tbm: cute.Tensor,
               tma_aa: cute.CopyAtom, taa: cute.Tensor, tma_xb: cute.CopyAtom, txb: cute.Tensor,
               tma_y: cute.CopyAtom, ty: cute.Tensor, tma_r: cute.CopyAtom, trr: cute.Tensor,
               z: cute.Tensor, slots: cute.Tensor, ranks: cute.Tensor, scales: cute.Tensor,
               ready: cute.Tensor, epoch: cute.Tensor, tr: cute.Tensor, res: cute.Tensor,
               gamma: cute.Tensor, ss_out: cute.Tensor, ss_in: cute.Tensor,
               qn_w: cute.Tensor, kn_w: cute.Tensor, cs: cute.Tensor, pos: cute.Tensor,
               kc: cute.Tensor, vc: cute.Tensor, loc: cute.Tensor, qo: cute.Tensor,
               gate_o: cute.Tensor, axis: cute.Tensor, sA_l: cute.ComposedLayout, sB_l: cute.ComposedLayout, sSFA_l, sSFB_l,
               sA16_l: cute.ComposedLayout, sB16_l: cute.ComposedLayout, sO_l: cute.Layout):
        S = self.stages
        NA = self.nacc
        bf = cutlass.BFloat16
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        et = tidx % 128

        smem = utils.SmemAllocator()
        st = smem.allocate(self.storage)
        sA16 = cute.make_tensor(cute.recast_ptr(st.sA.data_ptr(), swizzle_=sA16_l.inner, dtype=bf),
                                sA16_l.outer)
        sB16 = cute.make_tensor(cute.recast_ptr(st.sB.data_ptr(), swizzle_=sB16_l.inner, dtype=bf),
                                sB16_l.outer)
        if cutlass.const_expr(self.mx):
            f8 = cutlass.Float8E4M3FN
            sA = cute.make_tensor(cute.recast_ptr(st.sA.data_ptr(), swizzle_=sA_l.inner, dtype=f8),
                                  sA_l.outer)
            sB = cute.make_tensor(cute.recast_ptr(st.sB.data_ptr(), swizzle_=sB_l.inner, dtype=f8),
                                  sB_l.outer)
            sSFA = st.sSFA.get_tensor(sSFA_l)
            sSFB = st.sSFB.get_tensor(sSFB_l)
        else:
            sA, sB = sA16, sB16
        zrow = st.zrow.get_tensor(cute.make_layout((self.n_mma, 32 * self.zw),
                                                   stride=(32 * self.zw, 1)))
        tslot = st.tslot.get_tensor(cute.make_layout(self.n_mma))
        srstd = st.srstd.get_tensor(cute.make_layout(self.n_mma))
        sZd = cute.make_tensor(cute.recast_ptr(st.sZd.data_ptr(), swizzle_=sB16_l.inner, dtype=bf),
                               sm100_utils.make_smem_layout_b(
                                   mma16, self.tiler16, bf, max(self.lora_tiles, 1)).outer) \
            if cutlass.const_expr(self.zded) else sB16
        sO = st.sO.get_tensor(sO_l)
        full = st.full.data_ptr()
        empty = st.empty.data_ptr()
        zready = st.zready.data_ptr()
        zfree = st.zfree.data_ptr()
        acc_full = st.acc_full
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(S):
                    cute.arch.mbarrier_init(full + i, 1)
                    cute.arch.mbarrier_init(empty + i, 1)
                    cute.arch.mbarrier_init(zready + i, 1)
                    cute.arch.mbarrier_init(zfree + i, 1)
                cute.arch.mbarrier_init(acc_full, 1)
                cute.arch.mbarrier_init(st.rbar, 1)
        cute.arch.mbarrier_init_fence()
        if warp == 1:
            cute.arch.alloc_tmem(self.tmem_cols, st.tmem_holder)
        cute.arch.barrier()
        acc_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, st.tmem_holder)
        if tidx == 0:
            self._m0(tr, bid)
        acc_fake = mma.make_fragment_C(mma.partition_shape_C(self.tiler[:2]))
        tAcc = cute.make_tensor(acc_ptr, acc_fake.layout)
        acc16_fake = mma16.make_fragment_C(mma16.partition_shape_C(self.tiler16[:2]))
        tAcc16 = cute.make_tensor(acc_ptr, acc16_fake.layout)
        tAcc16b = cute.make_tensor(acc_ptr + self.n_mma, acc16_fake.layout)  # 2nd accumulator

        # chunked epilogue: TMEM columns [c*chunk, (c+1)*chunk) per step
        epi = (128, self.chunk)
        t2r_atom = sm100_utils.get_tmem_load_op(
            (128, self.n_mma, 64), utils.LayoutEnum.ROW_MAJOR, cutlass.Float32,
            cutlass.Float32, epi, False)
        acc_epi = cute.flat_divide(tAcc[((None, None), 0, 0)], epi)
        t2r = tcgen05.make_tmem_copy(t2r_atom, acc_epi[(None, None, 0, 0)])
        thr_t2r = t2r.get_slice(et)
        src = thr_t2r.partition_S(acc_epi)
        src_b = thr_t2r.partition_S(cute.flat_divide(tAcc16b[((None, None), 0, 0)], epi))
        coords = thr_t2r.partition_D(cute.flat_divide(
            cute.make_identity_tensor((128, self.n_mma)), epi))
        r = cute.make_rmem_tensor(coords[(None, None, None, 0, 0)].shape, cutlass.Float32)
        # FP32 tiles parked for the split-K / LoRA-A reductions (column-major: conflict free)
        red = cute.make_tensor(cute.recast_ptr(st.sA.data_ptr(), dtype=cutlass.Float32),
                               cute.make_layout((128, self.n_mma, NA),
                                                stride=(1, 128, 128 * self.n_mma)))
        one = cute.make_layout(1)
        qx = (qn_w, kn_w, cs, pos, kc, vc, loc, qo, gate_o, axis,
              cutlass.Int32(st.sA.data_ptr().toint()) + self.qs_off,
              cutlass.Int32(st.qconst.data_ptr().toint()))

        if bid < self.a_ctas:
            self.a_role(bid, warp, et, mma16, tma_aa, taa, tma_xb, txb, sA16, sB16, full,
                        empty, acc_full, tAcc16, t2r, src, coords, r, red, z, ready, tr,
                        slots)
        else:
            bb = bid - self.a_ctas
            m_tile = cutlass.min(bb // self.ksplit, self.m_tiles - 1)  # CTA column
            valid = bb < self.m_tiles * self.ksplit  # padding CTAs only join cluster barriers
            split = bb % self.ksplit
            rank0 = bb % self.C - split  # cluster rank of this split group's split 0
            j0 = split * self.per
            nj = cutlass.min(self.per, self.total - j0)
            if not valid:
                nj = cutlass.Int32(0)
            has_lora = j0 + nj > self.k_tiles
            t0 = self.tile_of(m_tile, 0)
            t1 = self.tile_of(m_tile, NA - 1)
            thr = mma.get_slice(0)
            gA0 = cute.local_tile(ta, cute.slice_(self.tiler if self.mx else self.tiler16,
                                                  (None, 0, None)), (t0, None, 0))
            gA1 = cute.local_tile(ta, cute.slice_(self.tiler if self.mx else self.tiler16,
                                                  (None, 0, None)), (t1, None, 0))
            gB = cute.local_tile(tb, cute.slice_(self.tiler if self.mx else self.tiler16,
                                                 (0, None, None)), (0, None, 0))
            tAsA, tAgA0 = cpasync.tma_partition(tma_a, 0, one, cute.group_modes(sA, 0, 3),
                                                cute.group_modes(thr.partition_A(gA0), 0, 3))
            _, tAgA1 = cpasync.tma_partition(tma_a, 0, one, cute.group_modes(sA, 0, 3),
                                             cute.group_modes(thr.partition_A(gA1), 0, 3))
            tBsB, tBgB = cpasync.tma_partition(tma_b, 0, one, cute.group_modes(sB, 0, 3),
                                               cute.group_modes(thr.partition_B(gB), 0, 3))
            gBM0 = cute.local_tile(tbm, cute.slice_(self.tiler16, (None, 0, None)), (t0, None, 0))
            gBM1 = cute.local_tile(tbm, cute.slice_(self.tiler16, (None, 0, None)), (t1, None, 0))
            tAsBM, tAgBM0 = cpasync.tma_partition(
                tma_bm, 0, one, cute.group_modes(sA16, 0, 3),
                cute.group_modes(mma16.get_slice(0).partition_A(gBM0), 0, 3))
            _, tAgBM1 = cpasync.tma_partition(
                tma_bm, 0, one, cute.group_modes(sA16, 0, 3),
                cute.group_modes(mma16.get_slice(0).partition_A(gBM1), 0, 3))
            tAsSFA = tAgSFA = tBsSFB = tBgSFB = tAsA  # placeholders (BF16)
            if cutlass.const_expr(self.mx):
                gSFA = cute.local_tile(tsfa, cute.slice_(self.tiler, (None, 0, None)),
                                       (t0, None, 0))
                gSFB = cute.local_tile(tsfb, cute.slice_(self.tiler_sfb, (0, None, None)),
                                       (0, None, 0))
                tAsSFA, tAgSFA = cpasync.tma_partition(
                    tma_sfa, 0, one, cute.group_modes(sSFA, 0, 3),
                    cute.group_modes(thr.partition_A(gSFA), 0, 3))
                tAsSFA = cute.filter_zeros(tAsSFA)
                tAgSFA = cute.filter_zeros(tAgSFA)
                tBsSFB, tBgSFB = cpasync.tma_partition(
                    tma_sfb, 0, one, cute.group_modes(sSFB, 0, 3),
                    cute.group_modes(mma_sfb.get_slice(0).partition_B(gSFB), 0, 3))
                tBsSFB = cute.filter_zeros(tBsSFB)
                tBgSFB = cute.filter_zeros(tBgSFB)
            if valid & (warp == 0):
                with cute.arch.elect_one():
                    cpasync.prefetch_descriptor(tma_a)
                    cpasync.prefetch_descriptor(tma_b)
                    cpasync.prefetch_descriptor(tma_bm)
                pre = cutlass.min(nj, S)
                for i in cutlass.range(pre, unroll=1):  # weights before the dependency wait
                    self.issue_w(j0 + i, i, full, zfree, tma_a, tAgA0, tAgA1, tAsA, tma_sfa,
                                 tAgSFA, tAsSFA, tma_bm, tAgBM0, tAgBM1, tAsBM)
                cute.arch.griddepcontrol_wait()
                if tidx == 0:
                    self._m1(tr, bid)
                for i in cutlass.range(pre, unroll=1):
                    self.issue_x(j0 + i, i, full, tma_b, tBgB, tBsB, tma_sfb, tBgSFB, tBsSFB)
                for kt in cutlass.range(S, nj, unroll=1):
                    s = kt % S
                    cute.arch.mbarrier_wait(empty + s, (kt // S - 1) % 2)
                    self.issue_w(j0 + kt, s, full, zfree, tma_a, tAgA0, tAgA1, tAsA, tma_sfa,
                                 tAgSFA, tAsSFA, tma_bm, tAgBM0, tAgBM1, tAsBM)
                    self.issue_x(j0 + kt, s, full, tma_b, tBgB, tBsB, tma_sfb, tBgSFB, tBsSFB)
                if cutlass.const_expr(self.early_trigger):
                    with cute.arch.elect_one():
                        cute.arch.griddepcontrol_launch_dependents()
            elif valid & (warp == 1):
                self.mma_warp(mma, mma16, sA, sB, sA16, sB16, sSFA_l, sSFB_l,
                              full, empty, zready, acc_full, tAcc, tAcc16, tAcc16b, acc_ptr, j0,
                              nj, st, tr, bid, sZd)
            elif valid & (warp >= 4):
                cute.arch.griddepcontrol_wait()  # inputs of the previous grid (slots, ss, res)
                if cutlass.const_expr(self.epi == "qkv"):
                    self.stage_qconst(qx, m_tile, et)
                if cutlass.const_expr(self.in_rstd):
                    self.compute_rstd(ss_in, srstd, et)  # ends with a warpgroup-1 barrier
                elif cutlass.const_expr(self.epi == "qkv"):
                    named_barrier(2, 128)  # staged QKV constants visible to warpgroup 1
                if cutlass.const_expr(self.in_rstd or self.epi == "qkv"):
                    named_barrier_arrive(7, 256)  # warpgroup 0 may read srstd / qconst
                if cutlass.const_expr(self.zded):
                  if has_lora:
                    self.build_zblk_direct(st, z, slots, ranks, scales, ready, epoch, tslot, bid,
                                           m_tile, et)
                    if et == 0:
                        self._m5(tr, bid)
                        cute.arch.mbarrier_arrive(zready)  # zready[0]: the dedicated Zblk
                elif has_lora:
                    self.build_zrow(z, slots, ranks, scales, ready, epoch, zrow, tslot, bid,
                                    m_tile, et)
                    if et == 0:
                        self._m5(tr, bid)
                    # Zblk for each LoRA use goes into the activation half of its stage once the
                    # producer reports the stage drained (zfree). A stage's LoRA uses come in
                    # order and each needs the previous one consumed, so the phase is the count
                    # of earlier LoRA uses of that stage.
                    first = cutlass.max(j0, self.k_tiles)
                    for idx in cutlass.range(first, j0 + nj, unroll=1):
                        kt = idx - j0
                        s = kt % S
                        cute.arch.mbarrier_wait(zfree + s, ((kt - (first - j0)) // S) % 2)
                        u = idx - self.k_tiles
                        if cutlass.const_expr(self.split_lora):
                            self.write_zblk(st, s, zrow, tslot, u // NA, u % NA, et)
                        else:
                            self.write_zblk(st, s, zrow, tslot, u, 0, et)
                        fence_proxy_async()
                        named_barrier(2, 128)
                        if et == 0:
                            cute.arch.mbarrier_arrive(zready + s)
                    if et == 0:
                        self._m6(tr, bid)
            if cutlass.const_expr(self.in_rstd or self.epi == "qkv"):
                if valid & (warp < 4):
                    named_barrier_n(7, 256)  # srstd / qconst written by warpgroup 1
            if valid:
                wg = warp // 4
                cute.arch.mbarrier_wait(acc_full, 0)
                self.epilogue(m_tile, t2r, src, src_b, coords, r, red, sO, tma_y, ty, tma_r, trr,
                              res, gamma, ss_out, srstd, qx, et, tr, bid, wg)
                if (et == 0) & (wg == 1):
                    self._m7(tr, bid)
            if cutlass.const_expr(self.ksplit > 1):
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                if cutlass.const_expr(self.bulk):
                    if valid & (warp >= 4) & (et == 0):
                        self.push_partials(st, split, rank0)
                if valid:
                    wg = warp // 4
                    if (et == 0) & (wg == 1):
                        self._m6(tr, bid)  # split-K reduction starts (all splits parked)
                    self.reduce_store(m_tile, split, rank0, coords, red, sO, tma_y, ty, tma_r,
                                      trr, res, gamma, ss_out, srstd, qx, et, tr, bid, st, wg)
                    if (et == 0) & (wg == 1):
                        self._m7(tr, bid)  # reduced chunks stored
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
        cute.arch.barrier()
        if warp == 1:
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(acc_ptr, self.tmem_cols)
        if cutlass.const_expr(not self.early_trigger):
            if tidx == 0:
                cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _m0(self, tr: cute.Tensor, bid):
        if cutlass.const_expr(self.trace):
            tr[bid * TRS + 0] = global_ns()

    @cute.jit
    def _m1(self, tr: cute.Tensor, bid):
        if cutlass.const_expr(self.trace):
            tr[bid * TRS + 1] = global_ns()

    @cute.jit
    def _m2(self, tr: cute.Tensor, bid):
        if cutlass.const_expr(self.trace):
            tr[bid * TRS + 2] = global_ns()

    @cute.jit
    def _m3(self, tr: cute.Tensor, bid):
        if cutlass.const_expr(self.trace):
            tr[bid * TRS + 3] = global_ns()

    @cute.jit
    def _m4(self, tr: cute.Tensor, bid):
        if cutlass.const_expr(self.trace):
            tr[bid * TRS + 4] = global_ns()

    @cute.jit
    def _m5(self, tr: cute.Tensor, bid):
        if cutlass.const_expr(self.trace):
            tr[bid * TRS + 5] = global_ns()

    @cute.jit
    def _m6(self, tr: cute.Tensor, bid):
        if cutlass.const_expr(self.trace):
            tr[bid * TRS + 6] = global_ns()

    @cute.jit
    def _m7(self, tr: cute.Tensor, bid):
        if cutlass.const_expr(self.trace):
            tr[bid * TRS + 7] = global_ns()

    @cute.jit
    def _m8(self, tr: cute.Tensor, bid):
        if cutlass.const_expr(self.trace):
            tr[bid * TRS + 8] = global_ns()

    @cute.jit
    def _m9(self, tr: cute.Tensor, bid):
        if cutlass.const_expr(self.trace):
            tr[bid * TRS + 9] = global_ns()

    @cute.jit
    def _m10(self, tr: cute.Tensor, bid):
        if cutlass.const_expr(self.trace):
            tr[bid * TRS + 10] = global_ns()

    @cute.jit
    def _m11(self, tr: cute.Tensor, bid):
        if cutlass.const_expr(self.trace):
            tr[bid * TRS + 11] = global_ns()

    # ---------------------------------------------------------------- base producer
    @cute.jit
    def issue_w(self, idx, s, full, zfree, tma_a, tAgA0, tAgA1, tAsA, tma_sfa, tAgSFA, tAsSFA,
                tma_bm, tAgBM0, tAgBM1, tAsBM):
        """Weight half of ring stage s for K-sequence item idx: base K-tile (every
        accumulator's weight tile, + SF) or a LoRA use (its B mirror tile(s)). Arms full[s]
        for the whole stage, including the activation half issue_x adds."""
        NA = self.nacc
        if idx < self.k_tiles:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(full + s,
                                                        NA * self.w_bytes + self.x_bytes)
            cute.copy(tma_a, tAgA0[(None, idx)], tAsA[(None, s * NA)], tma_bar_ptr=full + s)
            if cutlass.const_expr(NA > 1):
                cute.copy(tma_a, tAgA1[(None, idx)], tAsA[(None, s * NA + 1)],
                          tma_bar_ptr=full + s)
            if cutlass.const_expr(self.mx):
                cute.copy(tma_sfa, tAgSFA[(None, idx)], tAsSFA[(None, s)], tma_bar_ptr=full + s)
        else:
            u = idx - self.k_tiles
            if cutlass.const_expr(self.split_lora):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(zfree + s)  # stage drained (empty was waited)
                    cute.arch.mbarrier_arrive_and_expect_tx(full + s, self.bm_bytes)
                if u % NA == 0:
                    cute.copy(tma_bm, tAgBM0[(None, u // NA)], tAsBM[(None, s * NA)],
                              tma_bar_ptr=full + s)
                else:
                    cute.copy(tma_bm, tAgBM1[(None, u // NA)], tAsBM[(None, s * NA)],
                              tma_bar_ptr=full + s)
            else:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(zfree + s)
                    cute.arch.mbarrier_arrive_and_expect_tx(full + s, NA * self.bm_bytes)
                cute.copy(tma_bm, tAgBM0[(None, u)], tAsBM[(None, s * NA)], tma_bar_ptr=full + s)
                if cutlass.const_expr(NA > 1):
                    cute.copy(tma_bm, tAgBM1[(None, u)], tAsBM[(None, s * NA + 1)],
                              tma_bar_ptr=full + s)

    @cute.jit
    def issue_x(self, idx, s, full, tma_b, tBgB, tBsB, tma_sfb, tBgSFB, tBsSFB):
        if idx < self.k_tiles:
            cute.copy(tma_b, tBgB[(None, idx)], tBsB[(None, s)], tma_bar_ptr=full + s)
            if cutlass.const_expr(self.mx):
                cute.copy(tma_sfb, tBgSFB[(None, idx)], tBsSFB[(None, s)], tma_bar_ptr=full + s)

    # ---------------------------------------------------------------- base MMA warp
    @cute.jit
    def mma_warp(self, mma, mma16, sA, sB, sA16, sB16, sSFA_l, sSFB_l,
                 full, empty, zready, acc_full, tAcc, tAcc16, tAcc16b, acc_ptr, j0, nj, st, tr,
                 bid, sZd):
        S = self.stages
        NA = self.nacc
        tCrA = mma.make_fragment_A(sA)
        tCrB = mma.make_fragment_B(sB)
        tCrBM = mma16.make_fragment_A(sA16)
        tCrZ = mma16.make_fragment_B(sB16)
        tCrZd = mma16.make_fragment_B(sZd)
        nkb = cute.size(tCrBM, mode=[2])
        if cutlass.const_expr(self.mx):
            e8 = cutlass.Float8E8M0FNU
            sSFA = st.sSFA.get_tensor(sSFA_l)
            sSFB = st.sSFB.get_tensor(sSFB_l)
            tCtSFA = cute.make_tensor(
                cute.recast_ptr(acc_ptr + self.sf_col, dtype=e8),
                blockscaled_utils.make_tmem_layout_sfa(mma, self.tiler, SF_VEC,
                                                       cute.slice_(sSFA_l, (None, None, None, 0))))
            tCtSFB = cute.make_tensor(
                cute.recast_ptr(acc_ptr + self.sf_col + 16, dtype=e8),
                blockscaled_utils.make_tmem_layout_sfb(mma, self.tiler, SF_VEC,
                                                       cute.slice_(sSFB_l, (None, None, None, 0))))
            s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), e8)
            cp_sfa = tcgen05.make_s2t_copy(s2t, cute.filter_zeros(tCtSFA))
            cp_sfb = tcgen05.make_s2t_copy(s2t, cute.filter_zeros(tCtSFB))
            src_sfa = tcgen05.get_s2t_smem_desc_tensor(
                cp_sfa, cp_sfa.get_slice(0).partition_S(cute.filter_zeros(sSFA)))
            dst_sfa = cp_sfa.get_slice(0).partition_D(cute.filter_zeros(tCtSFA))
            src_sfb = tcgen05.get_s2t_smem_desc_tensor(
                cp_sfb, cp_sfb.get_slice(0).partition_S(cute.filter_zeros(sSFB)))
            dst_sfb = cp_sfb.get_slice(0).partition_D(cute.filter_zeros(tCtSFB))
        klo = cutlass.max(j0, self.k_tiles) - j0  # first LoRA item of this split
        for kt in cutlass.range(nj, unroll=1):
            s = kt % S
            idx = j0 + kt
            cute.arch.mbarrier_wait(full + s, (kt // S) % 2)
            if kt == 0:
                with cute.arch.elect_one():
                    self._m2(tr, bid)
            if idx == self.k_tiles:
                with cute.arch.elect_one():
                    self._m3(tr, bid)  # base K-tiles issued; first LoRA item next
            if idx < self.k_tiles:
                if cutlass.const_expr(self.mx):
                    cute.copy(cp_sfa, src_sfa[(None, None, None, None, s)], dst_sfa)
                    cute.copy(cp_sfb, src_sfb[(None, None, None, None, s)], dst_sfb)
                    for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        mma.set(tcgen05.Field.ACCUMULATE, (kt != 0) | (kb != 0))
                        mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kb)].iterator)
                        mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kb)].iterator)
                        cute.gemm(mma, tAcc, tCrA[(None, None, kb, s)], tCrB[(None, None, kb, s)],
                                  tAcc)
                else:
                    for kb in cutlass.range_constexpr(nkb):
                        mma16.set(tcgen05.Field.ACCUMULATE, (kt != 0) | (kb != 0))
                        cute.gemm(mma16, tAcc16, tCrBM[(None, None, kb, s * NA)],
                                  tCrZ[(None, None, kb, s)], tAcc16)
                    if cutlass.const_expr(NA > 1):
                        for kb in cutlass.range_constexpr(nkb):
                            mma16.set(tcgen05.Field.ACCUMULATE, (kt != 0) | (kb != 0))
                            cute.gemm(mma16, tAcc16b, tCrBM[(None, None, kb, s * NA + 1)],
                                      tCrZ[(None, None, kb, s)], tAcc16b)
            else:
                u = idx - self.k_tiles
                if cutlass.const_expr(self.zded):
                    cute.arch.mbarrier_wait(zready, 0)  # the dedicated Zblk is complete
                    for kb in cutlass.range_constexpr(nkb):
                        mma16.set(tcgen05.Field.ACCUMULATE, (kt != 0) | (kb != 0))
                        cute.gemm(mma16, tAcc16, tCrBM[(None, None, kb, s)],
                                  tCrZd[(None, None, kb, u)], tAcc16)
                else:
                    cute.arch.mbarrier_wait(zready + s, ((kt - klo) // S) % 2)
                    if cutlass.const_expr(self.split_lora):
                        # every split starts with base tiles (asserted): always accumulate
                        if u % NA == 0:
                            for kb in cutlass.range_constexpr(nkb):
                                mma16.set(tcgen05.Field.ACCUMULATE, True)
                                cute.gemm(mma16, tAcc16, tCrBM[(None, None, kb, s * NA)],
                                          tCrZ[(None, None, kb, s)], tAcc16)
                        else:
                            for kb in cutlass.range_constexpr(nkb):
                                mma16.set(tcgen05.Field.ACCUMULATE, True)
                                cute.gemm(mma16, tAcc16b, tCrBM[(None, None, kb, s * NA)],
                                          tCrZ[(None, None, kb, s)], tAcc16b)
                    else:
                        for kb in cutlass.range_constexpr(nkb):
                            mma16.set(tcgen05.Field.ACCUMULATE, (kt != 0) | (kb != 0))
                            cute.gemm(mma16, tAcc16, tCrBM[(None, None, kb, s * NA)],
                                      tCrZ[(None, None, kb, s)], tAcc16)
                        if cutlass.const_expr(NA > 1):
                            for kb in cutlass.range_constexpr(nkb):
                                mma16.set(tcgen05.Field.ACCUMULATE, (kt != 0) | (kb != 0))
                                cute.gemm(mma16, tAcc16b, tCrBM[(None, None, kb, s * NA + 1)],
                                          tCrZ[(None, None, kb, s)], tAcc16b)
            with cute.arch.elect_one():
                tcgen05.commit(empty + s)
        with cute.arch.elect_one():
            self._m4(tr, bid)
            tcgen05.commit(acc_full)

    # ---------------------------------------------------------------- LoRA-A role
    @cute.jit
    def a_role(self, bid, warp, et, mma16, tma_aa, taa, tma_xb, txb, sA16, sB16, full, empty,
               acc_full, tAcc16, t2r, src, coords, r, red, z, ready, tr, slots):
        S = self.stages
        C = self.C
        rank = bid % C
        cl = bid // C
        a_m = cl // self.q
        qi = cl % self.q
        kk0 = (qi * C + rank) * self.a_per
        nka = cutlass.min(self.a_per, self.a_ktiles - kk0)
        one = cute.make_layout(1)
        thr16 = mma16.get_slice(0)
        gAA = cute.local_tile(taa, cute.slice_(self.tiler16, (None, 0, None)), (a_m, None, 0))
        gXB = cute.local_tile(txb, cute.slice_(self.tiler16, (0, None, None)), (0, None, 0))
        tAsAA, tAgAA = cpasync.tma_partition(tma_aa, 0, one, cute.group_modes(sA16, 0, 3),
                                             cute.group_modes(thr16.partition_A(gAA), 0, 3))
        tBsXB, tBgXB = cpasync.tma_partition(tma_xb, 0, one, cute.group_modes(sB16, 0, 3),
                                             cute.group_modes(thr16.partition_B(gXB), 0, 3))
        if warp == 0:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_aa)
                cpasync.prefetch_descriptor(tma_xb)
            pre = cutlass.min(nka, S)
            for i in cutlass.range(pre, unroll=1):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full + i, self.bm_bytes + self.xb_bytes)
                cute.copy(tma_aa, tAgAA[(None, kk0 + i)], tAsAA[(None, i)], tma_bar_ptr=full + i)
            cute.arch.griddepcontrol_wait()
            for i in cutlass.range(pre, unroll=1):
                cute.copy(tma_xb, tBgXB[(None, kk0 + i)], tBsXB[(None, i)], tma_bar_ptr=full + i)
            for kt in cutlass.range(S, nka, unroll=1):
                s = kt % S
                cute.arch.mbarrier_wait(empty + s, (kt // S - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full + s, self.bm_bytes + self.xb_bytes)
                cute.copy(tma_aa, tAgAA[(None, kk0 + kt)], tAsAA[(None, s)], tma_bar_ptr=full + s)
                cute.copy(tma_xb, tBgXB[(None, kk0 + kt)], tBsXB[(None, s)], tma_bar_ptr=full + s)
            if cutlass.const_expr(self.early_trigger):
                with cute.arch.elect_one():
                    cute.arch.griddepcontrol_launch_dependents()
        elif warp == 1:
            tCrA = mma16.make_fragment_A(sA16)
            tCrB = mma16.make_fragment_B(sB16)
            for kt in cutlass.range(nka, unroll=1):
                s = kt % S
                cute.arch.mbarrier_wait(full + s, (kt // S) % 2)
                for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                    mma16.set(tcgen05.Field.ACCUMULATE, (kt != 0) | (kb != 0))
                    cute.gemm(mma16, tAcc16, tCrA[(None, None, kb, s)], tCrB[(None, None, kb, s)],
                              tAcc16)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + s)
            with cute.arch.elect_one():
                tcgen05.commit(acc_full)
        groups, tasks, per_thread = self.a_tasks()
        sv = cute.make_rmem_tensor(cute.make_layout(per_thread), cutlass.Int32)
        if warp >= 4:
            cute.arch.griddepcontrol_wait()  # slots may come from the previous grid
            self.a_slots(slots, rank, et, sv)
            # all MMAs done => the ring is drained: park the FP32 partial there
            cute.arch.mbarrier_wait(acc_full, 0)
            if et == 0:
                self._m4(tr, bid)
            for c in cutlass.range(self.n_chunks, unroll=1):
                cute.copy(t2r, src[(None, None, None, 0, c)], r)
                cd = coords[(None, None, None, 0, c)]
                for i in cutlass.range_constexpr(cute.size(r)):
                    m, t = cd[i]
                    red[m, t, 0] = r[i]
            cute.arch.fence_view_async_tmem_load()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if warp >= 4:
            if et == 0:
                self._m5(tr, bid)
            self.a_reduce(red, z, sv, a_m, qi, rank, et)
            named_barrier(1, 128)  # all compact-row stores of this CTA precede the release
            if et == 0:
                atomic_add_acq_rel(ready.iterator, cutlass.Int32(1))
                self._m6(tr, bid)
        cute.arch.cluster_arrive()  # peers have finished reading this CTA's partial
        cute.arch.cluster_wait()

    @cute.jit
    def a_tasks(self):
        SW = self.slices * 32
        groups = SW // 4
        per_rank = (self.T + self.C - 1) // self.C
        return groups, per_rank * groups, (per_rank * groups + 127) // 128

    @cute.jit
    def a_slots(self, slots, rank, et, sv):
        """Slot of each of this thread's reduction tasks (-1: nothing to write), read before
        the cluster barrier so the reduction does not wait on it."""
        groups, tasks, per_thread = self.a_tasks()
        for i in cutlass.range_constexpr(per_thread):
            task = et + i * 128
            t = rank + self.C * (task // groups)
            v = cutlass.Int32(-1)
            if (task < tasks) & (t < self.T):
                v = slots[t]
                if (v < 0) | (v >= self.n_slots):
                    v = cutlass.Int32(-1)
            sv[i] = v

    @cute.jit
    def a_reduce(self, red, z, sv, a_m, qi, rank, et):
        """Rank `rank` handles tokens t = rank, rank + C, ...: for each 4-row group of the
        token's own slot rows (slot*SW .. +SW, SW = slices*32) that lies in this 128-row block,
        sum the C parked partials (ranks in order) and store them to z[qi, t, 4g : 4g+4].
        Every task's DSMEM loads are issued before any sum (clamped address when idle)."""
        C = self.C
        SW = self.slices * 32
        groups, tasks, per_thread = self.a_tasks()
        red_base = cutlass.Int32(red.iterator.toint())
        parts = cute.make_rmem_tensor(cute.make_layout((per_thread, C, 4)), cutlass.Float32)
        live = cute.make_rmem_tensor(cute.make_layout(per_thread), cutlass.Int32)
        for i in cutlass.range_constexpr(per_thread):
            task = et + i * 128
            t = rank + C * (task // groups)
            row = sv[i] * SW + task % groups * 4 - a_m * 128
            ok = (sv[i] >= 0) & (row >= 0) & (row < 128)
            live[i] = cutlass.Int32(0)
            addr = red_base
            if ok:
                live[i] = cutlass.Int32(1)
                addr = red_base + (t * 128 + row) * 4
            for p in cutlass.range_constexpr(C):
                v4 = dsmem_load4(addr, cutlass.Int32(p))
                for e in cutlass.range_constexpr(4):
                    parts[i, p, e] = v4[e]
        for i in cutlass.range_constexpr(per_thread):
            if live[i] != 0:
                task = et + i * 128
                t = rank + C * (task // groups)
                acc = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
                for e in cutlass.range_constexpr(4):
                    v = parts[i, 0, e]
                    for p in cutlass.range_constexpr(1, C):
                        v = v + parts[i, p, e]
                    acc[e] = v
                st_global_v4f(z.iterator + cute.crd2idx((qi, t, task % groups * 4), z.layout),
                              [acc[0], acc[1], acc[2], acc[3]])

    # ---------------------------------------------------------------- LoRA-B input
    @cute.jit
    def sid_of(self, tile):
        sid = cutlass.Int32(0)
        for b in cutlass.range_constexpr(self.slices - 1):
            if tile * 128 >= self.slice_rows[b]:
                sid = cutlass.Int32(b + 1)
        return sid

    @cute.jit
    def fill_tslot(self, slots, tslot, et):
        for i in cutlass.range_constexpr((self.n_mma + 127) // 128):
            n = et + i * 128
            if n < self.n_mma:
                v = cutlass.Int32(-1)
                if n < self.T:
                    v = slots[n]
                    if (v < 0) | (v >= self.n_slots):
                        v = cutlass.Int32(-1)
                tslot[n] = v

    @cute.jit
    def wait_ready(self, ready, epoch, bid, et):
        if et == 0:
            want = epoch[bid] + 1
            wait_count(ready.iterator, want * self.a_real)
            epoch[bid] = want

    @cute.jit
    def z_words(self, vals, i, rq, rk, sc):
        """8 FP32 values (sum of the q planes) -> 4 packed BF16x2 words, scaled and rank
        masked (rows rq..rq+7 of the slot)."""
        words = []
        for j in cutlass.range_constexpr(4):
            pair = []
            for h in cutlass.range_constexpr(2):
                v = vals[i, 0, 2 * j + h]
                for p in cutlass.range_constexpr(1, self.q):
                    v = v + vals[i, p, 2 * j + h]
                pv = cutlass.Float32(0)
                if rq + 2 * j + h < rk:
                    pv = v * sc
                pair.append(pv)
            words.append(cutlass.Int32(_asm(
                ir.IntegerType.get_signless(32), [pair[1].ir_value(), pair[0].ir_value()],
                "cvt.rn.bf16x2.f32 $0,$1,$2;", "=r,f,f")))
        return words

    @cute.jit
    def build_zrow(self, z, slots, ranks, scales, ready, epoch, zrow, tslot, bid, m_tile, et):
        """zrow[t, a*32 + r] = bf16(scale[slot] * sum_p z[p, t, sid_a*32 + r]) for r <
        rank(slot), else 0, for tokens with an adapter, for each slice a this CTA's LoRA uses
        (zw of them). Also the per-token slot table (-1 = no adapter: Zblk rows stay zero)."""
        T = self.T
        ZW = self.zw
        self.fill_tslot(slots, tslot, et)
        self.wait_ready(ready, epoch, bid, et)
        named_barrier(2, 128)
        zrow_base = cutlass.Int32(zrow.iterator.toint())
        tasks = (T * 4 * ZW + 127) // 128  # (token, slice, 8-rank group)
        fly = self.inflight
        for i0 in cutlass.range_constexpr(0, tasks, fly):  # loads of `fly` tasks in flight
            nb = min(fly, tasks - i0)
            vals = cute.make_rmem_tensor(cute.make_layout((fly, self.q, 8)), cutlass.Float32)
            for b2 in cutlass.range_constexpr(nb):
                task = et + (i0 + b2) * 128
                n = cutlass.min(task // (4 * ZW), T - 1)
                a = task // 4 % ZW
                sid = self.sid_of(self.tile_of(m_tile, a))
                col = sid * 32 + task % 4 * 8
                for p in cutlass.range_constexpr(self.q):
                    base = z.iterator + cute.crd2idx((p, n, col), z.layout)
                    lo = load_cg4(base)
                    hi = load_cg4(base + 4)
                    for j in cutlass.range_constexpr(4):
                        vals[b2, p, j] = lo[j]
                        vals[b2, p, 4 + j] = hi[j]
            for b2 in cutlass.range_constexpr(nb):
                task = et + (i0 + b2) * 128
                n = task // (4 * ZW)
                a = task // 4 % ZW
                rq = task % 4 * 8
                if n < T:
                    sl = tslot[n]
                    if sl >= 0:
                        words = self.z_words(vals, b2, rq, ranks[sl], scales[sl])
                        st_shared_v4(zrow_base + n * (64 * ZW) + a * 64 + (rq // 8) * 16, words)
        named_barrier(2, 128)

    @cute.jit
    def build_zblk_direct(self, st, z, slots, ranks, scales, ready, epoch, tslot, bid, m_tile,
                          et):
        """Dedicated Zblk (lora_tiles x [n_mma, 64] BF16, SW128 K-major). Zero it and fill the
        slot table before waiting for LoRA-A (zeros do not depend on Z). Then for each token
        with an adapter write its 4 non-zero 16-byte chunks: chunk q of token n holds
        bf16(scale * sum_p z[p, n, sid*32 + 8q : +8]) (rank masked) in tile slot // 2, k-group
        (slot % 2)*4 + q, at n*128 + ((g ^ (n&7)) << 4) in that tile."""
        T = self.T
        sid = self.sid_of(self.tile_of(m_tile, 0))
        base = cutlass.Int32(st.sZd.data_ptr().toint())
        zero = cutlass.Int32(0)
        words16 = self.zbytes // 16
        for i in cutlass.range_constexpr((words16 + 127) // 128):
            w = et + i * 128
            if w < words16:
                st_shared_v4(base + w * 16, [zero, zero, zero, zero])
        self.fill_tslot(slots, tslot, et)
        self.wait_ready(ready, epoch, bid, et)
        named_barrier(2, 128)  # zero fill done before any value lands
        tasks = (T * 4 + 127) // 128
        tile = self.n_mma * 128
        fly = self.inflight
        for i0 in cutlass.range_constexpr(0, tasks, fly):  # loads of `fly` tasks in flight
            nb = min(fly, tasks - i0)
            vals = cute.make_rmem_tensor(cute.make_layout((fly, self.q, 8)), cutlass.Float32)
            for b2 in cutlass.range_constexpr(nb):
                task = et + (i0 + b2) * 128
                n = cutlass.min(task // 4, T - 1)
                col = sid * 32 + task % 4 * 8
                for p in cutlass.range_constexpr(self.q):
                    ptr = z.iterator + cute.crd2idx((p, n, col), z.layout)
                    lo = load_cg4(ptr)
                    hi = load_cg4(ptr + 4)
                    for j in cutlass.range_constexpr(4):
                        vals[b2, p, j] = lo[j]
                        vals[b2, p, 4 + j] = hi[j]
            for b2 in cutlass.range_constexpr(nb):
                task = et + (i0 + b2) * 128
                n = task // 4
                q = task % 4
                if n < T:
                    sl = tslot[n]
                    if sl >= 0:
                        words = self.z_words(vals, b2, q * 8, ranks[sl], scales[sl])
                        g = (sl & 1) * 4 + q
                        st_shared_v4(base + (sl >> 1) * tile + n * 128 + ((g ^ (n & 7)) << 4),
                                     words)
        fence_proxy_async()
        named_barrier(2, 128)

    @cute.jit
    def write_zblk(self, st, s, zrow, tslot, jl, a, et):
        """Stage s, activation half = Zblk for LoRA K-tile jl of zrow slice a, one 16-byte
        store per (token n, 8-wide k-group g). A tile covers kb/32 slots; row n holds
        zrow[n, a*32 + (g%4)*8 : +8] in group g when token n's slot is (kb/32)*jl + g//4, else
        zeros. K-major swizzle: chunk g of row n is at n*rb + ((g ^ sw(n)) << 4), with
        sw(n) = n & 7 for 128-byte rows (SW128) and (n >> 1) & 3 for 64-byte rows (SW64)."""
        rb = self.row_b
        G = self.kb // 8
        spt = self.kb // 32
        stage = cutlass.Int32(st.sB.data_ptr().toint()) + s * (self.n_mma * rb)
        zrow_base = cutlass.Int32(zrow.iterator.toint())
        zero = cutlass.Int32(0)
        for i in cutlass.range_constexpr((self.n_mma * G + 127) // 128):
            task = et + i * 128
            n = task // G
            g = task % G
            if n < self.n_mma:
                w = [zero, zero, zero, zero]
                if tslot[n] == spt * jl + g // 4:
                    w = ld_shared_v4(zrow_base + n * (64 * self.zw) + a * 64 + (g % 4) * 16)
                if cutlass.const_expr(self.kb == 64):
                    st_shared_v4(stage + n * rb + ((g ^ (n & 7)) << 4), w)
                else:
                    st_shared_v4(stage + n * rb + ((g ^ ((n >> 1) & 3)) << 4), w)

    @cute.jit
    def compute_rstd(self, ss_in, srstd, et):
        """srstd[t] = rsqrt(sum_c ss_in[t, c] / K + eps) (partials summed in order)."""
        npart = self.K // 128
        for i in cutlass.range_constexpr((self.n_mma + 127) // 128):
            t = et + i * 128
            if t < self.n_mma:
                acc = cutlass.Float32(0)
                if t < self.T:
                    for p in cutlass.range_constexpr(npart // 4):
                        v4 = load_cg4(ss_in.iterator + cute.crd2idx((t, p * 4), ss_in.layout))
                        for e in cutlass.range_constexpr(4):
                            acc = acc + v4[e]
                srstd[t] = rsqrt_approx(acc * (1.0 / self.K) + self.eps)
        named_barrier(2, 128)

    # ---------------------------------------------------------------- epilogue
    @cute.jit
    def epilogue(self, m_tile, t2r, src, src_b, coords, r, red, sO, tma_y, ty, tma_r, trr, res,
                 gamma, ss_out, srstd, qx, et, tr, bid, wg):
        """TMEM -> registers in 32-token chunks. Without split-K: finish each chunk (scale,
        activation / residual, BF16, TMA store). With split-K: park FP32 for the cluster
        reduction."""
        NA = self.nacc
        m = coords[(None, None, None, 0, 0)][0][0]
        for c in cutlass.range(self.n_chunks, unroll=1):  # runtime loop: bounded code size
            if wg == 1 - c % 2:
              vals = cute.make_rmem_tensor(cute.make_layout((NA, self.chunk)), cutlass.Float32)
              for a in cutlass.range_constexpr(NA):
                  if cutlass.const_expr(a == 0):
                      cute.copy(t2r, src[(None, None, None, 0, c)], r)
                  else:
                      cute.copy(t2r, src_b[(None, None, None, 0, c)], r)
                  cd = coords[(None, None, None, 0, c)]
                  if cutlass.const_expr(self.ksplit > 1):
                      for i in cutlass.range_constexpr(cute.size(r)):
                          mm, t = cd[i]
                          red[mm, t, a] = r[i]
                      if cutlass.const_expr(self.bulk):
                          fence_proxy_async()
                  else:
                      for i in cutlass.range_constexpr(cute.size(r)):
                          vals[a, i] = r[i]
              if cutlass.const_expr(self.ksplit == 1):
                  if (c == 0) & (et == 0):
                      self._m8(tr, bid)  # first chunk in registers
                  self.finish_chunk(c, (c // 2) * self.units, vals, m_tile, m, sO, tma_y, ty, tma_r,
                                    trr, res, gamma, ss_out, srstd, qx, et, wg)
                  if (c == 0) & (et == 0):
                      self._m9(tr, bid)  # first chunk finished
        cute.arch.fence_view_async_tmem_load()
        if cutlass.const_expr(self.ksplit == 1):
            if (et == 0) & (wg == 1):
                self._m10(tr, bid)  # all chunks issued
            self.drain_stores(et, wg)
            if (et == 0) & (wg == 1):
                self._m11(tr, bid)  # stores complete

    @cute.jit
    def push_partials(self, st, split, rank0):
        """Bulk-copy this rank's parked FP32 tile of every chunk it does not own into the
        owner's inbox slot (this rank, chunk index, accumulator); completion is counted on the
        owner's rbar. The parked tiles were written by generic stores of the epilogue threads,
        each of which fenced them to the async proxy before the cluster barrier."""
        KS = self.ksplit
        base = cutlass.Int32(st.sA.data_ptr().toint())
        bar = cutlass.Int32(st.rbar.toint())
        for c in cutlass.range_constexpr(self.n_chunks):
            for a in cutlass.range_constexpr(self.nacc):
                o = c % KS
                if o != split:
                    src = base + (a * 128 * self.n_mma + c * self.chunk * 128) * 4
                    dst = base + self.inbox_off + ((split * self.cpr + c // KS) * self.nacc + a) \
                        * self.tile_bytes
                    bulk_push(dst, src, self.tile_bytes, bar, rank0 + o)

    @cute.jit
    def reduce_store(self, m_tile, split, rank0, coords, red, sO, tma_y, ty, tma_r, trr, res,
                     gamma, ss_out, srstd, qx, et, tr, bid, st, wg):
        """Rank `split` reduces token chunks c = split, split + S, ... over the split group's
        parked FP32 tiles (ranks summed in order: deterministic) and finishes them."""
        NA = self.nacc
        KS = self.ksplit
        m = coords[(None, None, None, 0, 0)][0][0]
        if cutlass.const_expr(self.bulk):
            # every peer tile of every chunk this rank owns has landed in the inbox
            own = self.n_chunks // KS
            if cutlass.const_expr(self.n_chunks % KS != 0):
                if split < self.n_chunks % KS:
                    own = own + 1
            if (et == 0) & (wg == 1):
                cute.arch.mbarrier_arrive_and_expect_tx(
                    st.rbar, (KS - 1) * own * NA * self.tile_bytes)
            cute.arch.mbarrier_wait(st.rbar, 0)
            inbox = cute.make_tensor(
                cute.recast_ptr(st.sA.data_ptr() + self.inbox_off, dtype=cutlass.Float32),
                cute.make_layout((128, self.chunk, NA, self.cpr, KS),
                                 stride=(1, 128, 128 * self.chunk, 128 * self.chunk * NA,
                                         128 * self.chunk * NA * self.cpr)))
        for c in cutlass.range(self.n_chunks, unroll=1):  # runtime loop: bounded code size
            # owned chunks alternate between the warpgroups (first one to warpgroup 1)
            if (c % self.ksplit == split) & (wg == 1 - (c // KS) % 2):
                parts = cute.make_rmem_tensor(cute.make_layout((NA, self.chunk, self.ksplit)),
                                              cutlass.Float32)
                for a in cutlass.range_constexpr(NA):
                    for j in cutlass.range_constexpr(self.chunk):
                        for p in cutlass.range_constexpr(self.ksplit):
                            if cutlass.const_expr(self.bulk):
                                v = inbox[m, j, a, c // KS, p]
                                if p == split:
                                    v = red[m, c * self.chunk + j, a]
                                parts[a, j, p] = v
                            else:
                                parts[a, j, p] = dsmem_load1(
                                    red.iterator + a * 128 * self.n_mma
                                    + (c * self.chunk + j) * 128 + m, rank0 + p)
                if (c == split) & (et == 0) & (wg == 1):
                    self._m8(tr, bid)  # first chunk's partials loaded
                vals = cute.make_rmem_tensor(cute.make_layout((NA, self.chunk)), cutlass.Float32)
                for a in cutlass.range_constexpr(NA):
                    for j in cutlass.range_constexpr(self.chunk):
                        v = parts[a, j, 0]
                        for p in cutlass.range_constexpr(1, self.ksplit):
                            v = v + parts[a, j, p]
                        vals[a, j] = v
                self.finish_chunk(c, (c // KS // 2) * self.units, vals, m_tile, m, sO, tma_y,
                                  ty, tma_r, trr, res, gamma, ss_out, srstd, qx, et, wg)
                if (c == split) & (et == 0) & (wg == 1):
                    self._m9(tr, bid)  # first chunk finished
        if (et == 0) & (wg == 1):
            self._m10(tr, bid)
        self.drain_stores(et, wg)
        if (et == 0) & (wg == 1):
            self._m11(tr, bid)

    @cute.jit
    def finish_chunk(self, c, n0, vals, m_tile, m, sO, tma_y, ty, tma_r, trr, res, gamma,
                     ss_out, srstd, qx, et, wg):
        """vals[a, j]: FP32 accumulator a, feature m of this CTA's tile, token c*chunk + j.
        n0: this chunk's first staging-unit number (selects the buffer, orders reuse)."""
        NA = self.nacc
        CH = self.chunk
        bf = cutlass.BFloat16
        if cutlass.const_expr(self.in_rstd):
            for j in cutlass.range_constexpr(CH):
                sr = srstd[c * CH + j]
                for a in cutlass.range_constexpr(NA):
                    vals[a, j] = vals[a, j] * sr
        if cutlass.const_expr(self.epi == "linear"):
            for a in cutlass.range_constexpr(NA):
                n = n0 + a
                buf = wg * self.obuf + n % self.obuf
                self.stage_chunk(n, et, wg)
                for j in cutlass.range_constexpr(CH):
                    sO[(j, m, buf)] = vals[a, j].to(bf)
                self.store_unit(tma_y, ty, c, self.tile_of(m_tile, a), buf, sO, et, wg)
        elif cutlass.const_expr(self.epi == "qkv"):
            self.qkv_chunk(c, vals, m_tile, m, qx, et, wg)
        elif cutlass.const_expr(self.epi == "silu_mul"):
            buf = wg * self.obuf + n0 % self.obuf
            self.stage_chunk(n0, et, wg)
            for j in cutlass.range_constexpr(CH):
                g = vals[0, j]
                sO[(j, m, buf)] = (g * cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-g, fastmath=True)) * vals[1, j]).to(bf)
            self.store_unit(tma_y, ty, c, m_tile, buf, sO, et, wg)
        else:  # resnorm
            col = m_tile * 128 + m
            gv = gamma[col]
            buf = wg * self.obuf + n0 % self.obuf
            self.stage_chunk(n0, et, wg)
            for j in cutlass.range_constexpr(CH):
                t = c * CH + j
                rv = cutlass.Float32(0)
                if t < self.T:
                    rv = res[t, col].to(cutlass.Float32)
                hb = (rv + vals[0, j]).to(bf)
                sO[(j, m, buf)] = hb
                vals[0, j] = hb.to(cutlass.Float32) * gv
            self.store_unit(tma_r, trr, c, m_tile, buf, sO, et, wg)
            # per-token sum of squares of the stored (BF16) h over this tile's 128 features
            warp = et // 32
            lane = et % 32
            for jj in cutlass.range_constexpr(CH // 4):
                j = warp * (CH // 4) + jj
                s2 = cutlass.Float32(0)
                for e in cutlass.range_constexpr(4):
                    hv = sO[(j, lane * 4 + e, buf)].to(cutlass.Float32)
                    s2 = s2 + hv * hv
                s2 = warp_sum(s2)
                t = c * CH + j
                if (lane == 0) & (t < self.T):
                    ss_out[t, m_tile] = s2
            n1 = n0 + 1
            buf1 = wg * self.obuf + n1 % self.obuf
            self.stage_chunk(n1, et, wg)
            for j in cutlass.range_constexpr(CH):
                sO[(j, m, buf1)] = vals[0, j].to(bf)
            self.store_unit(tma_y, ty, c, m_tile, buf1, sO, et, wg)

    @cute.jit
    def stage_qconst(self, qx, head, et):
        """SMEM copies the QKV epilogue needs per token (position, cache slot) and per feature
        (this head's q or k norm weight), loaded while the main loop runs."""
        qn_w, kn_w, cs, pos, kc, vc, loc, qo, gate_o, axis, qs0, qc = qx
        P1 = 2 * self.n_mma + self.hd
        for i in cutlass.range_constexpr((self.n_mma + 127) // 128):
            t = et + i * 128
            if t < self.n_mma:
                p = cutlass.Int32(0)
                l = cutlass.Int32(-1)
                if t < self.T:
                    l = cutlass.Int32(loc[t])
                    if cutlass.const_expr(self.mrope):
                        p = cutlass.Int32(pos[0, t])
                        st_shared_i32(qc + (P1 + t) * 4, cutlass.Int32(pos[1, t]))
                        st_shared_i32(qc + (P1 + self.n_mma + t) * 4, cutlass.Int32(pos[2, t]))
                    else:
                        p = cutlass.Int32(pos[t])
                st_shared_i32(qc + t * 4, p)
                st_shared_i32(qc + (self.n_mma + t) * 4, l)
        if cutlass.const_expr(self.mrope):
            if et < self.rot // 2:
                st_shared_i32(qc + (P1 + 2 * self.n_mma + et) * 4, cutlass.Int32(axis[et]))
        is_key = (head >= self.qheads) & (head < self.qheads + self.kvh)
        for i in cutlass.range_constexpr((self.hd + 127) // 128):
            n = et + i * 128
            if n < self.hd:
                wv = qn_w[n]
                if is_key:
                    wv = kn_w[n]
                st_shared_f32(qc + (2 * self.n_mma + n) * 4, wv)

    @cute.jit
    def qkv_chunk(self, c, vals, head, m, qx, et, wg):
        """One head (this CTA's tile(s)) for tokens c*chunk ..: stage the BF16-rounded
        projection as FP32 [chunk, head] in SMEM; then each warp finishes chunk/4 tokens, lane l
        holding features g*128 + 4l .. +4 of each 128-group: q/k RMSNorm (weights already
        include Gemma's +1), BF16 rounding, NeoX RoPE on the first `rot` features (partner
        lane l ^ rot/8), then q -> qo, gate (raw) -> gate_o, k / v -> the KV cache at loc[t].
        Loads are batched: norm weights once, positions / cache slots one per lane, and the
        SMEM rows and cos/sin of 4 tokens before any of their math."""
        qn_w, kn_w, cs, pos, kc, vc, loc, qo, gate_o, axis, qs0, qc = qx
        qs = qs0 + wg * (self.chunk * self.hd * 4)
        qbar = 6 - 2 * wg  # warpgroup 1: barrier 4, warpgroup 0: barrier 6
        NA = self.nacc
        CH = self.chunk
        HD = self.hd
        TW = CH // 4  # tokens per warp
        G = min(4, TW)
        bf = cutlass.BFloat16
        for a in cutlass.range_constexpr(NA):
            for j in cutlass.range_constexpr(CH):
                st_shared_f32(qs + (j * HD + a * 128 + m) * 4,
                              vals[a, j].to(bf).to(cutlass.Float32))
        warp = et // 32
        lane = et % 32
        QH = self.qheads
        is_qry = head < QH
        is_gate = cutlass.Boolean(False)
        if cutlass.const_expr(self.gated):
            is_gate = is_qry & (head % 2 == 1)
        is_key = (head >= QH) & (head < QH + self.kvh)
        normed = (is_qry & ~is_gate) | is_key
        half = self.rot // 2
        # per-lane constants: norm weights of this lane's features; one token's pos / loc
        wv = cute.make_rmem_tensor(cute.make_layout(4 * NA), cutlass.Float32)
        for g in cutlass.range_constexpr(NA):
            w4 = ld_shared_v4f(qc + (2 * self.n_mma + g * 128 + lane * 4) * 4)
            for e in cutlass.range_constexpr(4):
                wv[g * 4 + e] = w4[e]
        t_l = c * CH + warp * TW + cutlass.min(lane, TW - 1)
        p_l = ld_shared_i32(qc + t_l * 4)
        l_l = ld_shared_i32(qc + (self.n_mma + t_l) * 4)
        rope_lane = lane * 4 < self.rot
        cidx = (lane * 4) % half
        P1 = 2 * self.n_mma + self.hd
        p_l1 = p_l
        p_l2 = p_l
        axr = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int32)
        if cutlass.const_expr(self.mrope):
            p_l1 = ld_shared_i32(qc + (P1 + t_l) * 4)
            p_l2 = ld_shared_i32(qc + (P1 + self.n_mma + t_l) * 4)
            for e in cutlass.range_constexpr(4):
                axr[e] = ld_shared_i32(qc + (P1 + 2 * self.n_mma + cidx + e) * 4)
        named_barrier_r(qbar, 128)  # staged rows visible
        for g0 in cutlass.range_constexpr(0, TW, G):
            raw = cute.make_rmem_tensor(cute.make_layout((G, 4 * NA)), cutlass.Float32)
            cosv = cute.make_rmem_tensor(cute.make_layout((G, 4)), cutlass.Float32)
            sinv = cute.make_rmem_tensor(cute.make_layout((G, 4)), cutlass.Float32)
            for b in cutlass.range_constexpr(G):
                j = warp * TW + g0 + b
                for g in cutlass.range_constexpr(NA):
                    v4 = ld_shared_v4f(qs + (j * HD + g * 128 + lane * 4) * 4)
                    for e in cutlass.range_constexpr(4):
                        raw[b, g * 4 + e] = v4[e]
                p = shfl_idx_i32(p_l, g0 + b)
                for e in cutlass.range_constexpr(4):
                    cosv[b, e] = cutlass.Float32(1)
                    sinv[b, e] = cutlass.Float32(0)
                if cutlass.const_expr(self.mrope):
                    p1 = shfl_idx_i32(p_l1, g0 + b)
                    p2 = shfl_idx_i32(p_l2, g0 + b)
                    if normed & rope_lane:
                        for e in cutlass.range_constexpr(4):
                            pe = p
                            if axr[e] == 1:
                                pe = p1
                            if axr[e] == 2:
                                pe = p2
                            cosv[b, e] = cs[pe, cidx + e]
                            sinv[b, e] = cs[pe, half + cidx + e]
                elif normed & rope_lane:
                    c4 = load_nc4(cs.iterator + cute.crd2idx((p, cidx), cs.layout))
                    s4 = load_nc4(cs.iterator + cute.crd2idx((p, half + cidx), cs.layout))
                    for e in cutlass.range_constexpr(4):
                        cosv[b, e] = c4[e]
                        sinv[b, e] = s4[e]
            for b in cutlass.range_constexpr(G):
                j = warp * TW + g0 + b
                t = c * CH + j
                lc = shfl_idx_i32(l_l, g0 + b)
                out = cute.make_rmem_tensor(cute.make_layout(4 * NA), cutlass.Float32)
                for i in cutlass.range_constexpr(4 * NA):
                    out[i] = raw[b, i]
                if normed:
                    s2 = cutlass.Float32(0)
                    for i in cutlass.range_constexpr(4 * NA):
                        s2 = s2 + raw[b, i] * raw[b, i]
                    inv = rsqrt_approx(warp_sum(s2) * (1.0 / HD) + self.eps)
                    for g in cutlass.range_constexpr(NA):
                        for e in cutlass.range_constexpr(4):
                            v = (raw[b, g * 4 + e] * inv * wv[g * 4 + e]).to(bf) \
                                .to(cutlass.Float32)
                            peer = shfl_xor_f32(v, self.rot // 8)
                            out[g * 4 + e] = v
                            if cutlass.const_expr(g * 128 < self.rot):
                                if rope_lane:
                                    rp = peer * sinv[b, e]
                                    if lane * 4 < half:
                                        rp = -rp
                                    out[g * 4 + e] = v * cosv[b, e] + rp
                if (t < self.T):
                    for g in cutlass.range_constexpr(NA):
                        n = g * 128 + lane * 4
                        w = [pack_bf16x2(out[g * 4], out[g * 4 + 1]),
                             pack_bf16x2(out[g * 4 + 2], out[g * 4 + 3])]
                        if is_gate:
                            st_global_v2(gate_o.iterator + cute.crd2idx(
                                (t, head // 2 * HD + n), gate_o.layout), w)
                        elif is_qry:
                            qhead = head
                            if cutlass.const_expr(self.gated):
                                qhead = head // 2
                            st_global_v2(qo.iterator + cute.crd2idx((t, qhead * HD + n),
                                                                    qo.layout), w)
                        elif lc >= 0:
                            if is_key:
                                st_global_v2(kc.iterator + cute.crd2idx(
                                    (cutlass.Int64(lc), head - QH, n), kc.layout), w)
                            else:
                                st_global_v2(vc.iterator + cute.crd2idx(
                                    (cutlass.Int64(lc), head - QH - self.kvh, n), vc.layout), w)
        named_barrier_r(qbar, 128)  # scratch reused by the next chunk

    @cute.jit
    def stage_chunk(self, n, et, wg):
        """Before writing staging buffer n % obuf again, the TMA store that read it (obuf
        stores ago) must have finished reading SMEM; and every thread must be done reading it."""
        if n >= self.obuf:
            if et == 0:
                _asm(None, [], f"cp.async.bulk.wait_group.read {self.obuf - 1};", "~{memory}")
            named_barrier_r(5 - 2 * wg, 128)

    @cute.jit
    def store_unit(self, tma, tens, c, tile, buf, sO, et, wg):
        fence_proxy_async()
        named_barrier_r(5 - 2 * wg, 128)
        if et == 0:
            gY = cute.local_tile(tens, (self.chunk, 128), (None, None))
            sPart, gPart = cpasync.tma_partition(tma, 0, cute.make_layout(1),
                                                 cute.group_modes(sO, 0, 2),
                                                 cute.group_modes(gY, 0, 2))
            cute.copy(tma, sPart[(None, buf)], gPart[(None, c, tile)])
            _asm(None, [], "cp.async.bulk.commit_group;", "~{memory}")

    @cute.jit
    def drain_stores(self, et, wg):
        if et == 0:
            _asm(None, [], "cp.async.bulk.wait_group 0;", "~{memory}")
        named_barrier_r(5 - 2 * wg, 128)


def plan(n_out, k, slice_rows, base="bf16", n_slots=8, sms=148, tokens=128, pairing="none",
         epi="linear", in_rstd=False, qkv=None):
    """(ksplit, a_splits) for one wave: the largest split-K that fits next to the LoRA-A CTAs,
    with a_splits a multiple of the cluster size (q = 1-3 compact planes). Configurations the
    kernel rejects (SMEM, split-K tile, LoRA-only splits with two accumulators) are skipped."""
    slices = len(slice_rows)
    a_mtiles = (slices * n_slots * 32 + 127) // 128
    a_ktiles = (k + 63) // 64
    for ks in (4, 3, 2, 1):
        for C in {4: (4,), 3: (3,), 2: (4, 2), 1: (4, 2, 1)}[ks]:
            for a_splits in ((9, 6, 3) if C == 3 else tuple(a for a in (8, 4, 2) if a % C == 0)):
                a_per = (a_ktiles + a_splits - 1) // a_splits
                if (a_splits - 1) * a_per >= a_ktiles:
                    continue
                try:
                    op = TLinear(n_out, k, tokens, slice_rows, base, ks, a_splits, None,
                                 n_slots, pairing=pairing, epi=epi, in_rstd=in_rstd,
                                 cluster=C, qkv=qkv)
                except AssertionError:
                    continue
                if op.grid <= sms and fits_clusters(op):
                    return ks, a_splits, C
    raise ValueError("no one-wave plan")


def fits_clusters(op):
    """All of op's clusters co-resident (driver query; True when it is unavailable)."""
    try:
        from .plans import max_active_clusters
    except Exception:
        return True
    cap = max_active_clusters(op.C, op.stages * op.stage_bytes + op.fixed_bytes)
    return cap is None or op.grid <= cap * op.C


def plans(n_out, k, slice_rows, base="bf16", n_slots=8, sms=148, tokens=128, pairing="none",
          epi="linear", in_rstd=False, per_ks=2, qkv=None):
    """Every valid one-wave (ksplit, a_splits, cluster) for autotuning: per split-K, the
    `per_ks` largest LoRA-A splits that fit."""
    slices = len(slice_rows)
    a_ktiles = (k + 63) // 64
    out = []
    for ks in (1, 2, 3, 4):
        got = 0
        for C in {4: (4,), 3: (3,), 2: (4, 2), 1: (4, 2, 1)}[ks]:
            for a_splits in ((9, 6, 3) if C == 3 else tuple(a for a in (16, 8, 4, 2) if a % C == 0)):
                if got >= per_ks:
                    break
                a_per = (a_ktiles + a_splits - 1) // a_splits
                if (a_splits - 1) * a_per >= a_ktiles:
                    continue
                try:
                    op = TLinear(n_out, k, tokens, slice_rows, base, ks, a_splits, None,
                                 n_slots, pairing=pairing, epi=epi, in_rstd=in_rstd,
                                 cluster=C, qkv=qkv)
                except AssertionError:
                    continue
                if op.grid <= sms and fits_clusters(op) and \
                        (ks, a_splits) not in [(o[0], o[1]) for o in out]:
                    out.append((ks, a_splits, C))
                    got += 1
    return out


def make(w, w_sf, x_bf, x_q, x_sf, a_buf, bm, slots, ranks, scales, slice_rows, base="bf16",
         ksplit=1, a_splits=4, stages=None, compiled=None, trace=False, pairing="none",
         epi="linear", in_rstd=False, res=None, gamma=None, ss_in=None, y=None, ss_out=None,
         cluster=None, qkv=None, qkv_io=None):
    """Test/runner helper. w: [N, K] BF16 (base "bf16") or FP8 (base "mxfp8", with w_sf, x_q,
    x_sf from flashinfer mxfp8_quantize, swizzled); x_bf [T, K] BF16; a_buf [slots, slices*32,
    K]; bm [N, slots*32] (B pool mirror); slots [T] int32; ranks / scales [slots].
    epi "resnorm": res [T, N] BF16 (updated in place), gamma [N] FP32 (already 1 + w for a
    Gemma norm); y = h * gamma; ss_out [n_mma, N/128]. in_rstd: ss_in [n_mma, K/128]."""
    import torch
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack
    from .plans import check_one_wave
    N, K = w.shape
    T = x_bf.shape[0]
    n_slots = a_buf.shape[0]
    op = TLinear(N, K, T, slice_rows, base, ksplit, a_splits, stages, n_slots, trace=trace,
                 pairing=pairing, epi=epi, in_rstd=in_rstd, cluster=cluster, qkv=qkv)
    dev = w.device
    check_one_wave("TLinear", op.grid,
                   op.stages * op.stage_bytes + op.fixed_bytes,
                   torch.cuda.get_device_properties(dev).multi_processor_count,
                   tmem_cols=op.tmem_cols, cluster=op.C)
    if y is None:
        y = torch.empty(T, N // 2 if epi == "silu_mul" else (128 if epi == "qkv" else N),
                        device=dev, dtype=torch.bfloat16)
    z = torch.zeros(op.q, op.n_mma, op.slices * 32, device=dev, dtype=torch.float32)
    ready = torch.zeros(1, device=dev, dtype=torch.int32)
    epoch = torch.zeros(op.grid, device=dev, dtype=torch.int32)
    tr = torch.zeros(op.grid * TRS, device=dev, dtype=torch.int64)
    if epi == "resnorm":
        assert res is not None and gamma is not None
        if ss_out is None:
            ss_out = torch.zeros(op.n_mma, op.m_tiles, device=dev, dtype=torch.float32)
    else:
        res = y
        gamma = torch.zeros(16, device=dev, dtype=torch.float32)
        ss_out = torch.zeros(op.n_mma, 4, device=dev, dtype=torch.float32)
    if not in_rstd:
        ss_in = torch.zeros(op.n_mma, 4, device=dev, dtype=torch.float32)
    dummy = torch.zeros(16, device=dev, dtype=torch.uint8)
    if base == "mxfp8":
        refs = [w.view(torch.uint8).unsqueeze(-1), w_sf.view(torch.uint8).reshape(-1),
                x_q.view(torch.uint8).unsqueeze(-1), x_sf.view(torch.uint8).reshape(-1)]
    else:
        refs = [w.unsqueeze(-1), dummy, dummy, dummy]
    refs += [x_bf.unsqueeze(-1), a_buf.reshape(-1, K).unsqueeze(-1),
             bm.unsqueeze(-1), y, z,
             slots, ranks, scales, ready, epoch, tr, res, gamma, ss_out, ss_in]
    if epi == "qkv":
        # qkv_io: q_norm / k_norm weights (FP32, Gemma +1 applied), cos_sin cache [P, rot] FP32,
        # positions [T] int64, k / v cache [slots, kv heads, head dim] BF16, loc [T] int64,
        # q out [T, q heads * head dim], gate out [T, q heads * head dim] (or None)
        io = dict(qkv_io)
        if io.get("gate") is None:
            io["gate"] = io["q"]
        refs += [io["qn_w"], io["kn_w"], io["cs"], io["pos"], io["kc"], io["vc"], io["loc"],
                 io["q"], io["gate"],
                 io["axis"] if io.get("axis") is not None else torch.zeros(
                     64, device=dev, dtype=torch.int32)]
    else:
        f1 = torch.zeros(16, device=dev, dtype=torch.float32)
        i1 = torch.zeros(16, device=dev, dtype=torch.int64)
        b3 = torch.zeros(1, 1, 16, device=dev, dtype=torch.bfloat16)
        b2 = torch.zeros(1, 16, device=dev, dtype=torch.bfloat16)
        refs += [f1, f1, f1.view(1, 16), i1, b3, b3, i1, b2, b2,
                 torch.zeros(64, device=dev, dtype=torch.int32)]
    args = [from_dlpack(t, assumed_align=16) for t in refs]
    stream = lambda: cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = compiled or cute.compile(op, *args, stream())

    def launch():
        fn(*args, stream())
        return y

    return dict(launch=launch, y=y, op=op, compiled=fn, z=z, trace=tr, ss_out=ss_out, res=res)
