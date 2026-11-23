"""Fused routed-expert projection with multi-LoRA for decode (grouped TLinear).

For every (token block, weight row tile) work item, one CTA computes

    acc[128 rows, 16 tokens] = W_e[rows] . X_blk^T  +  sum_slot Bm_e[rows, slot] . Zblk_slot^T

where the first term is the base expert GEMM (MXFP4 weights x MXFP8 activations, or BF16)
and the LoRA-B part runs as BF16 K-tiles appended to the same TMEM accumulator, one K-tile
per adapter slot used by the block's tokens (as in tlinear). Z comes from the previous
kernel as FP32 partial planes, summed, scaled and rank-masked when Zblk is built.

Rows are routed (token, expert) pairs, grouped by local expert into 16-row blocks (padding
rows have slot -1 and are never stored). Block b covers rows [16b, 16b + 16); its activation
scale factors sit in their own 128-row swizzled group b, so the block's SFB tile is always the
first 16 rows of a 128-row atom.

Weights are the ones the base MoE already holds (stacked [E * rows, K]); the TRT-LLM row
shuffle stays inside 128-row tiles, so the epilogue maps rows back through a lookup table.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05

from .tlinear import (
    fence_proxy_async,
    load_cg4,
    named_barrier,
    st_shared_v4,
    _asm,
)
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from .tlinear import st_global_v2
from .ptx import atomic_add_acq_rel, wait_count


def cp_async16(dst, src, nbytes):
    """16-byte cp.async (L2 only) global -> shared; nbytes = 0 zero-fills."""
    _asm(None, [dst.ir_value(), src.ir_value(), nbytes.ir_value()],
         "cp.async.cg.shared.global [$0], [$1], 16, $2;", "r,l,r,~{memory}")


def cp_async_arrive_noinc(mbar):
    """One arrival on mbar once this thread's earlier cp.async copies have landed."""
    _asm(None, [mbar.ir_value()], "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];",
         "r,~{memory}")

def shfl_xor_f32(v, lanes):
    return cutlass.Float32(llvm.inline_asm(
        ir.F32Type.get(), [v.ir_value(), cutlass.Int32(lanes).ir_value()],
        "shfl.sync.bfly.b32 $0,$1,$2,31,-1;", "=f,f,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=0))


def ue8m0_round_up(x):
    """UE8M0 byte of a non-negative FP32, rounding up (cudaRoundPosInf); 0 for 0."""
    bits = cutlass.Int32(_asm(ir.IntegerType.get_signless(32), [x.ir_value()], "mov.b32 $0,$1;",
                              "=r,f"))
    ex = (bits >> 23) & 255
    mant = bits & 0x7FFFFF
    return ex + cutlass.Int32((mant != 0) & (ex < 254))


def exp2_int(n):
    """2^n as FP32 for an integer -126 <= n <= 127."""
    return cutlass.Float32(_asm(ir.F32Type.get(), [((n + 127) << 23).ir_value()],
                                "mov.b32 $0,$1;", "=f,r"))


def cvt_e4m3x4(a, b, c, d):
    """Four FP32 -> four E4M3 bytes (RN, satfinite), a in the lowest byte."""
    return cutlass.Int32(_asm(
        ir.IntegerType.get_signless(32), [a.ir_value(), b.ir_value(), c.ir_value(), d.ir_value()],
        "{.reg .b16 l, h; cvt.rn.satfinite.e4m3x2.f32 l, $2, $1; "
        "cvt.rn.satfinite.e4m3x2.f32 h, $4, $3; mov.b32 $0, {l, h};}", "=r,f,f,f,f"))


def load_bf16x8(ptr):
    """8 BF16 from global (16 B aligned) as FP32."""
    i32 = ir.IntegerType.get_signless(32)
    rr = llvm.inline_asm(llvm.StructType.get_literal([i32] * 4), [ptr.toint().ir_value()],
                         "ld.global.nc.v4.b32 {$0,$1,$2,$3},[$4];", "=r,=r,=r,=r,l",
                         has_side_effects=True, is_align_stack=False, asm_dialect=0)
    out = []
    for j in range(4):
        w = cutlass.Int32(llvm.extractvalue(i32, rr, [j]))
        out.append(cutlass.Float32(_asm(ir.F32Type.get(), [(w << 16).ir_value()],
                                        "mov.b32 $0,$1;", "=f,r")))
        out.append(cutlass.Float32(_asm(ir.F32Type.get(), [(w & -65536).ir_value()],
                                        "mov.b32 $0,$1;", "=f,r")))
    return out


def load_cg_s64(ptr):
    return cutlass.Int64(_asm(ir.IntegerType.get_signless(64), [ptr.toint().ir_value()],
                              "ld.global.cg.s64 $0,[$1];", "=l,l,~{memory}"))


def fix_to_f32(v):
    """Q32.32 int64 -> FP32."""
    f = cutlass.Float32(_asm(ir.F32Type.get(), [v.ir_value()], "cvt.rn.f32.s64 $0,$1;",
                             "=f,l"))
    return f * cutlass.Float32(2.0 ** -32)


def red_add_fix(ptr, x):
    """Atomically add FP32 x to a Q32.32 int64 in global memory (exact, order independent)."""
    q = cutlass.Int64(_asm(ir.IntegerType.get_signless(64),
                           [(x * cutlass.Float32(2.0 ** 32)).ir_value()],
                           "cvt.rni.s64.f32 $0,$1;", "=l,f"))
    _asm(None, [ptr.toint().ir_value(), q.ir_value()], "red.global.add.u64 [$0],$1;",
         "l,l,~{memory}")


SF_VEC = 32
SMEM_LIMIT = 232448
NT = 16  # rows (routed tokens) per block = MMA N
QD = 8  # unit-id queue depth (producer -> MMA / epilogue warps)


class TExperts:
    """n_rows: weight rows per expert (2I for gate/up, H for down); k: reduction size.
    w_slice: LoRA columns per slot (64 = gate32|up32, 32 = down); n_parts: Z partial planes.
    epi "raw":    out[row, tile*128 + r] = acc (FP32, shuffled row order).
        "swiglu": gate/up tiles (lut: shuffled row -> is_up << 8 | feature in tile). Clamped
                  SwiGLU, BF16-rounded, quantized to MXFP8 into out_q [rows, I] / out_sf (the
                  down kernel's activation layout), and the down-LoRA shrink partials
                  zd[row, tile, r] = A2[slot, expert, r, tile*64 : +64] . act.
        "down":   out[row, lut[tile*128 + r]] = bf16(acc) (lut: shuffled row -> feature)."""

    def __init__(self, n_experts, n_rows, k, max_blocks, base="mxfp4", n_slots=8, w_slice=64,
                 n_parts=1, epi="raw", stages=None, clamp=10.0, fp4_tx="gmem", sms=148,
                 zfix=False, ktile=128, fold_a=False, a_experts=0, zd_fix=None, a_ks=4,
                 pair=False, shared_e=-1, a2_experts=None, lora32=False, m_tiles_sh=None,
                 k_tiles_sh=None, order="m", grid_cap=0, gu32=True):
        assert base in ("mxfp4", "mxfp8", "bf16")
        self.mx = base != "bf16"  # block-scaled MMA with MXFP8 activations
        # mxfp8: FP8 weights with UE8M0 scales (per row, per 32 K), read in place; the gate/up
        # kernel loads each 128-row tile as 64 gate rows + 64 up rows (two TMA boxes) of the
        # expert's [gate; up] weight, so the SwiGLU pairs sit in one tile without a copy
        self.a8 = base == "mxfp8"
        self.halves = self.a8 and epi == "swiglu"
        self.E, self.N, self.K = n_experts, n_rows, k
        assert n_rows % 128 == 0
        self.m_tiles = n_rows // 128 // (2 if pair else 1)
        self.max_blocks = max_blocks
        self.rows_max = max_blocks * NT
        self.n_slots = n_slots
        self.W = w_slice
        assert 64 % w_slice == 0
        # lora32: one 32-wide LoRA-B K-tile per slot (SW64), no zero half when a block uses only
        # one slot of a pair; else 64-wide tiles covering 64 / w_slice slots
        self.l32 = lora32
        assert not lora32 or w_slice == 32
        # gu32 (MXFP4 gate/up, w_slice 64): a slot's LoRA-B is one dense 32-wide tile (each
        # row's own B_gate or B_up row), run as a 32-column MMA against [Z_gate; Z_up] into a
        # separate accumulator; the epilogue adds each row's half. The 64-wide tile it
        # replaces (gate32 | up32 for every row) was half zeros.
        self.gu32 = bool(gu32 and epi == "swiglu" and base == "mxfp4" and w_slice == 64
                         and not lora32 and not pair)
        self.kl = 32 if (lora32 or self.gu32) else 64
        self.spt = 1 if self.gu32 else self.kl // w_slice  # slots per LoRA K-tile
        self.lora_tiles = (n_slots + self.spt - 1) // self.spt
        self.n_parts = n_parts
        # zfix: Z arrives as one Q32.32 int64 vector per row (summed with integer atomics, so
        # order independent and exact), else as n_parts FP32 partial planes
        self.zfix = zfix
        self.zd_fix = zfix if zd_fix is None else zd_fix  # swiglu's down-LoRA Z output
        self.epi = epi
        assert epi in ("raw", "swiglu", "down", "swiglu_bf")
        # pair (BF16 [gate; up] experts, Qwen MoE): an item is gate row tile t and up row tile
        # t + I/128 in two accumulators; LoRA-B runs per half (gate rows x gate Z, up rows x up
        # Z) from a [2, E, I, slots*32] mirror; "swiglu_bf" writes BF16 silu(g) * u.
        self.pair = pair
        self.NA = 2 if pair else 1
        # shared_e >= 0: expert index of a folded shared expert. Its LoRA buffers are the dense
        # pool's (unscaled, garbage past each adapter's rank), so its Z gets the per-slot scale
        # and rank mask; routed experts' B buffers are pre-scaled and zero past the rank.
        self.shared_e = shared_e
        self.a2_experts = n_experts if a2_experts is None else a2_experts
        # shared blocks are the last nblk[1] blocks; their items have their own tile counts
        # (MXFP8 shared expert: 5 gate/up tiles of 64 + 64 features, down K = 384)
        assert not pair or (base == "bf16" and w_slice == 32)
        self.clamp = clamp
        self.fp4_tx = fp4_tx  # bytes an unpacking FP4 TMA completes: "smem" or "gmem" (packed)
        self.inter = n_rows // 2  # swiglu: features per expert
        self.kb = 64  # BF16 K-tile (LoRA-B / BF16 base)
        self.ke = ktile if self.mx else 64  # elements per base K-tile
        assert k % self.ke == 0
        self.k_tiles = k // self.ke
        # A region of a stage: BF16 [128, 64] (LoRA-B / BF16 base) or unpacked FP4 [128, ke]
        self.w_tile = self.NA * max(128 * self.kb * 2, 128 * self.ke if self.mx else 0)
        # A-item K-steps per stage: BF16 [gate; up] pairs have room for two 128 x 64 A tiles
        self.AP = 2 if (pair and fold_a and base == "bf16") else 1
        # BF16 gate/up: the producer warp gathers the block's activation rows straight from
        # the token-ordered hidden states (cp.async by token index), no permuted copy
        self.xg = base == "bf16" and fold_a and pair
        self.x_tile = NT * self.ke if self.mx else NT * self.kb * 2 * self.AP
        # bytes of one Zblk tile (gu32: [2 NT rows (gate rows, then up rows), 32])
        self.ztile = NT * self.kl * 2 * (2 if self.gu32 else 1)
        self.zbytes = self.NA * self.lora_tiles * self.ztile
        self.stage_bytes = self.w_tile + self.x_tile + (2 * 128 * self.ke // 32 if self.mx else 0)
        # epilogue scratch (SwiGLU): gate/up staging [2, 64, NT+1] + activation [NT, 65] FP32,
        # aliased on the item's own Zblk buffer (the MMA finished reading it)
        self.scr_bytes = {"swiglu": (2 * 64 * (NT + 1) + NT * 65) * 4,
                          "swiglu_bf": NT * 129 * 4}.get(epi, 0)
        self.zbuf = max(self.zbytes, self.scr_bytes)
        fixed = 2 * self.zbuf + 2048
        fit = (SMEM_LIMIT - fixed) // self.stage_bytes
        self.stages = min(12, fit) if stages is None else stages
        assert self.stages >= 2
        self.tmem_cols = 128 if self.gu32 else 64
        self.sf_col = 2 * NT  # accumulators at columns 0 and NT; SFA / SFB after them
        self.l_col = 64  # gu32: LoRA accumulators [128, 2 NT] at columns 64 and 64 + 2 NT
        self.tiler = (128, NT, self.ke)
        self.tiler16 = (128, NT, self.kb)
        self.tilerL = (128, NT, self.kl)  # LoRA-B K-tile
        self.tilerZ = (128, 2 * NT, 32)  # gu32 LoRA MMA: N = gate rows | up rows
        self.tiler_sfb = (128, 128, self.ke)
        # persistent: one CTA per SM walks items j = bid, bid + grid, ... over
        # n_blocks * m_tiles (block = j % n_blocks, row tile = j // n_blocks)
        # work order: "m" = row tile major (a wave spans many experts' same tile), "e" = expert
        # (block) major, consecutive units are consecutive row tiles of one block.
        # Expert-major on fewer CTAs than SMs, as SGLang's cutedsl grouped GEMM does, streams
        # the BF16 experts 2-5% faster than row-tile major.
        self.order = order
        self.grid = min(grid_cap or sms, max_blocks * self.m_tiles)
        self.m_tiles_sh = self.m_tiles if m_tiles_sh is None else m_tiles_sh
        self.k_tiles_sh = self.k_tiles if k_tiles_sh is None else k_tiles_sh
        self.rows_sh = self.m_tiles_sh * 128 * self.NA
        self.K_sh = self.k_tiles_sh * self.ke
        self.mx_sh = self.mx and shared_e >= 0  # FP8 shared expert next to FP4 routed ones
        # fold_a (gate/up): the LoRA-A shrink runs in this kernel as "A-items" ahead of the
        # GEMM items. A-item (block, slot pair): Z^T [2 x 64, 16 rows] = A_gu[slot, expert] .
        # x^T over K (BF16, the same tcgen05 pipeline, two 64-row TMA boxes per A tile). Each
        # Z row is written once (FP32, the row's own slot) and the block's counter released;
        # the block's GEMM items wait for it before building their Zblk. A-items precede every
        # GEMM item in each CTA's order and never wait, so the waits cannot deadlock.
        self.fold_a = fold_a
        self.a_experts = a_experts  # experts in the A buffer (1: shared-outer)
        self.a_ks = a_ks  # K parts per A-item (Q32.32 atomics into Z when > 1)
        assert (k // self.kb) % a_ks == 0
        self.a_ktiles = k // self.kb // a_ks // self.AP  # stages per A-item
        assert not fold_a or a_ks == 1 or zfix, "split A-items accumulate fixed-point Z"

    def __repr__(self):
        return (f"TExperts_{('mx8' if self.a8 else 'mx4') if self.mx else 'bf'}_E{self.E}_N{self.N}_K{self.K}"
                f"_B{self.max_blocks}_W{self.W}_P{self.n_parts}_{self.epi}_st{self.stages}"
                f"{'_xg' if self.xg else ''}{'_oe' if self.order == 'e' else ''}_g{self.grid}"
                f"{'_gu32' if self.gu32 else ''}")

    @cute.jit
    def __call__(self, w: cute.Tensor, w_sf: cute.Tensor, x_q: cute.Tensor, x_sf: cute.Tensor,
                 bm: cute.Tensor, z: cute.Tensor, rslot: cute.Tensor, blk_expert: cute.Tensor,
                 blk_mask: cute.Tensor, ranks: cute.Tensor, scales: cute.Tensor,
                 out: cute.Tensor, lut: cute.Tensor, a2: cute.Tensor, out_q: cute.Tensor,
                 out_sf: cute.Tensor, zd: cute.Tensor, nblk: cute.Tensor, x_bf: cute.Tensor, a_gu: cute.Tensor, a_blk: cute.Tensor,
                 a_s: cute.Tensor, n_a: cute.Tensor, npairs: cute.Tensor, zcnt: cute.Tensor,
                 a2s: cute.Tensor, a_gu_s: cute.Tensor, w_sh: cute.Tensor, sf_sh: cute.Tensor,
                 rtok: cute.Tensor, stream):
        f4, f8, e8, bf = (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU,
                          cutlass.BFloat16)
        major = tcgen05.OperandMajorMode.K
        S = self.stages
        one4 = (1, 1, 1, 1)
        sl = lambda l: cute.slice_(l, (None, None, None, 0))
        mma16 = sm100_utils.make_trivial_tiled_mma(bf, bf, major, major, cutlass.Float32,
                                                   tcgen05.CtaGroup.ONE, self.tiler16[:2])
        sA16_l = sm100_utils.make_smem_layout_a(mma16, self.tiler16, bf, S * self.NA)
        sB16_l = sm100_utils.make_smem_layout_b(mma16, self.tiler16, bf, S * self.AP)
        op_a = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), mma16.thr_id)
        op_b = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), mma16.thr_id)
        LI = self.w_tile // (128 * self.kl * 2)  # LoRA tiles per stage A region
        sAL_l = sm100_utils.make_smem_layout_a(mma16, self.tilerL, bf, S * LI)
        tma_bm, tbm = cute.nvgpu.make_tiled_tma_atom_A(op_a, bm, sl(sAL_l), self.tilerL, mma16,
                                                       one4)
        assert self.zbuf % self.ztile == 0
        mma32 = sm100_utils.make_trivial_tiled_mma(bf, bf, major, major, cutlass.Float32,
                                                   tcgen05.CtaGroup.ONE, self.tilerZ[:2])
        if cutlass.const_expr(self.gu32):
            sZL_l = sm100_utils.make_smem_layout_b(mma32, self.tilerZ, bf,
                                                   2 * (self.zbuf // self.ztile))
        else:
            sZL_l = sm100_utils.make_smem_layout_b(mma16, self.tilerL, bf,
                                                   2 * (self.zbuf // self.ztile))
        rows_all = self.E * self.N
        af = f8 if self.a8 else f4
        if cutlass.const_expr(self.mx):
            a = cute.make_tensor(cute.recast_ptr(w.iterator, dtype=af),
                                 cute.make_layout((rows_all, self.K, 1),
                                                  stride=(self.K, 1, rows_all * self.K)))
            b = cute.make_tensor(cute.recast_ptr(x_q.iterator, dtype=f8),
                                 cute.make_layout((self.rows_max, self.K, 1),
                                                  stride=(self.K, 1, self.rows_max * self.K)))
            sfa = cute.make_tensor(cute.recast_ptr(w_sf.iterator, dtype=e8),
                                   blockscaled_utils.tile_atom_to_shape_SF(a.shape, SF_VEC))
            # SF rows: one 128-row group per block
            sfb_shape = (self.max_blocks * 128, self.K, 1)
            sfb = cute.make_tensor(cute.recast_ptr(x_sf.iterator, dtype=e8),
                                   blockscaled_utils.tile_atom_to_shape_SF(sfb_shape, SF_VEC))
            mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                af, f8, major, major, e8, SF_VEC, tcgen05.CtaGroup.ONE, self.tiler[:2])
            mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
                af, f8, major, major, e8, SF_VEC, tcgen05.CtaGroup.ONE, self.tiler_sfb[:2])
            # kind::mxf8f6f4 reads FP4 from SMEM unpacked (16 values in each 16-byte chunk):
            # an 8-bit K-major layout for the MMA descriptors, filled by a U4 -> U8 unpacking TMA.
            sA_l = sm100_utils.make_smem_layout_a(mma, self.tiler, f8, S)
            sB_l = sm100_utils.make_smem_layout_b(mma, self.tiler, f8, S)
            sSFA_l = blockscaled_utils.make_smem_layout_sfa(mma, self.tiler, SF_VEC, S)
            sSFB_l = blockscaled_utils.make_smem_layout_sfb(mma, self.tiler, SF_VEC, S)
            op_sfb = sm100_utils.cluster_shape_to_tma_atom_SFB((1, 1), mma.thr_id)
            if cutlass.const_expr(self.a8):
                tma_a, ta = cute.nvgpu.make_tiled_tma_atom_A(op_a, a, sl(sA_l), self.tiler, mma,
                                                             one4)
            else:
                tma_a, ta = cute.nvgpu.make_tiled_tma_atom_A(op_a, a, sl(sA_l), self.tiler, mma,
                                                             one4, internal_type=cutlass.Uint8)
            tma_b, tb = cute.nvgpu.make_tiled_tma_atom_B(op_b, b, sl(sB_l), self.tiler, mma, one4)
            tma_sfa, tsfa = cute.nvgpu.make_tiled_tma_atom_A(
                op_a, sfa, sl(sSFA_l), self.tiler, mma, one4, internal_type=cutlass.Int16)
            tma_sfb, tsfb = cute.nvgpu.make_tiled_tma_atom_B(
                op_sfb, sfb, sl(sSFB_l), self.tiler_sfb, mma_sfb, one4,
                internal_type=cutlass.Int16)
            mma8 = mma
            tma_as, tas, tma_sfas, tsfas = tma_a, ta, tma_sfa, tsfa
            if cutlass.const_expr(self.mx_sh):
                mma8 = sm100_utils.make_blockscaled_trivial_tiled_mma(
                    f8, f8, major, major, e8, SF_VEC, tcgen05.CtaGroup.ONE, self.tiler[:2])
                a_s8 = cute.make_tensor(cute.recast_ptr(w_sh.iterator, dtype=f8),
                                        cute.make_layout((self.rows_sh, self.K_sh, 1),
                                                         stride=(self.K_sh, 1,
                                                                 self.rows_sh * self.K_sh)))
                sfa_s = cute.make_tensor(cute.recast_ptr(sf_sh.iterator, dtype=e8),
                                         blockscaled_utils.tile_atom_to_shape_SF(a_s8.shape,
                                                                                 SF_VEC))
                tma_as, tas = cute.nvgpu.make_tiled_tma_atom_A(op_a, a_s8, sl(sA_l), self.tiler,
                                                               mma8, one4)
                tma_sfas, tsfas = cute.nvgpu.make_tiled_tma_atom_A(
                    op_a, sfa_s, sl(sSFA_l), self.tiler, mma8, one4, internal_type=cutlass.Int16)
            wa = cute.size_in_bytes(f8, sl(sA_l))
            if cutlass.const_expr(self.fp4_tx == "gmem" and not self.a8):
                wa = wa // 2
            self.w_bytes = wa + cute.size_in_bytes(e8, sl(sSFA_l))
            self.w_bytes_sh = cute.size_in_bytes(f8, sl(sA_l)) + cute.size_in_bytes(
                e8, sl(sSFA_l))  # an FP8 tile is not unpacked: full smem bytes
            self.x_bytes = cute.size_in_bytes(f8, sl(sB_l)) + cute.size_in_bytes(e8, sl(sSFB_l))
            sa_bytes = cute.size_in_bytes(f8, sl(sA_l))
            sb_bytes = cute.size_in_bytes(f8, sl(sB_l))
            sfa_cos, sfb_cos = cute.cosize(sSFA_l), cute.cosize(sSFB_l)
        else:
            mma, mma_sfb, sA_l, sB_l = mma16, mma16, sA16_l, sB16_l
            sSFA_l = sSFB_l = cute.make_layout(128)
            xb = cute.make_tensor(x_q.iterator, cute.make_layout(
                (self.rows_max, self.K, 1), stride=(self.K, 1, self.rows_max * self.K)))
            tma_a, ta = cute.nvgpu.make_tiled_tma_atom_A(op_a, w, sl(sA16_l), self.tiler16,
                                                         mma16, one4)
            tma_b, tb = cute.nvgpu.make_tiled_tma_atom_B(op_b, xb, sl(sB16_l), self.tiler16,
                                                         mma16, one4)
            tma_sfa, tsfa, tma_sfb, tsfb = tma_a, ta, tma_b, tb
            self.w_bytes = cute.size_in_bytes(bf, sl(sA16_l))
            self.x_bytes = cute.size_in_bytes(bf, sl(sB16_l))
            sa_bytes, sb_bytes = self.w_bytes, self.x_bytes
            self.w_bytes_sh = self.w_bytes
            mma8 = mma
            tma_as, tas, tma_sfas, tsfas = tma_a, ta, tma_sfa, tsfa
            sfa_cos, sfb_cos = 128, 128
        assert sa_bytes <= self.w_tile and sb_bytes <= self.x_tile, (sa_bytes, sb_bytes)
        self.bm_bytes = cute.size_in_bytes(bf, sl(sAL_l))
        # fold_a operands: A_gu rows in 64-row TMA boxes (half of a [128, 64] BF16 A tile),
        # permuted BF16 activations as the B operand
        half_l = cute.tile_to_shape(
            tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, bf), (64, 64),
            (0, 1))
        a2d = cute.make_tensor(a_gu.iterator, cute.make_layout(
            (a_gu.shape[0], self.K), stride=(self.K, 1)))
        tma_ah, tah = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), a2d,
                                                  half_l, (64, 64))
        a2d_s = cute.make_tensor(a_gu_s.iterator, cute.make_layout(
            (a_gu_s.shape[0], self.K), stride=(self.K, 1)))
        tma_ahs, tahs = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), a2d_s,
                                                    half_l, (64, 64))
        # mxfp8 gate/up: 64-row boxes of the FP8 weights (gate half, up half of a tile)
        half8_l = cute.tile_to_shape(
            tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, f8),
            (64, self.ke), (0, 1))
        tma_wh, twh = tma_ahs, tahs
        if cutlass.const_expr(self.halves):
            a2d8 = cute.make_tensor(cute.recast_ptr(w.iterator, dtype=f8), cute.make_layout(
                (rows_all, self.K), stride=(self.K, 1)))
            tma_wh, twh = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), a2d8,
                                                      half8_l, (64, self.ke))
        xbf = cute.make_tensor(x_bf.iterator, cute.make_layout(
            (self.rows_max, self.K, 1), stride=(self.K, 1, self.rows_max * self.K)))
        tma_xb, txb = cute.nvgpu.make_tiled_tma_atom_B(op_b, xbf, sl(sB16_l), self.tiler16,
                                                       mma16, one4)
        self.ah_bytes = 64 * 64 * 2
        self.xb_bytes = cute.size_in_bytes(bf, sl(sB16_l))
        self.x_tx = 0 if self.xg else self.x_bytes  # gathered rows complete by cp.async

        @cute.struct
        class Storage:
            full: cute.struct.MemRange[cutlass.Int64, S]
            empty: cute.struct.MemRange[cutlass.Int64, S]
            zready: cute.struct.MemRange[cutlass.Int64, 2]
            zfree: cute.struct.MemRange[cutlass.Int64, 2]
            acc_full: cute.struct.MemRange[cutlass.Int64, 2]
            acc_empty: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holder: cutlass.Int32
            sA: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, S * self.w_tile], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, S * self.x_tile], 1024]
            sSFA: cute.struct.Align[cute.struct.MemRange[e8, sfa_cos], 128]
            sSFB: cute.struct.Align[cute.struct.MemRange[e8, sfb_cos], 128]
            sZd: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, 2 * self.zbuf], 1024]
            tslot: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, NT], 16]
            uq: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, QD], 16]
            uq_full: cute.struct.MemRange[cutlass.Int64, QD]
            uq_empty: cute.struct.MemRange[cutlass.Int64, QD]

        self.storage = Storage
        self.kernel(mma, mma_sfb, mma16, tma_a, ta, tma_b, tb, tma_sfa, tsfa, tma_sfb, tsfb,
                    tma_bm, tbm, z, rslot, blk_expert, blk_mask, ranks, scales, out,
                    lut, a2, out_q, out_sf, zd, nblk, tma_ah, tah, tma_xb, txb, half_l,
                    a_blk, a_s, n_a, npairs, zcnt, a2s, tma_ahs, tahs, mma8, tma_as,
                    tas, tma_sfas, tsfas, x_bf, rtok, tma_wh, twh, half8_l, sA_l, sB_l, sSFA_l,
                    sSFB_l, sA16_l, sB16_l, sAL_l, sZL_l, mma32).launch(
            grid=(self.grid, 1, 1), block=(256, 1, 1), cluster=(1, 1, 1),
            smem=Storage.size_in_bytes(), stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(self, mma: cute.TiledMma, mma_sfb: cute.TiledMma, mma16: cute.TiledMma,
               tma_a: cute.CopyAtom, ta: cute.Tensor, tma_b: cute.CopyAtom, tb: cute.Tensor,
               tma_sfa: cute.CopyAtom, tsfa: cute.Tensor, tma_sfb: cute.CopyAtom,
               tsfb: cute.Tensor, tma_bm: cute.CopyAtom, tbm: cute.Tensor, z: cute.Tensor,
               rslot: cute.Tensor, blk_expert: cute.Tensor, blk_mask: cute.Tensor,
               ranks: cute.Tensor, scales: cute.Tensor, out: cute.Tensor,
               lut: cute.Tensor, a2: cute.Tensor, out_q: cute.Tensor, out_sf: cute.Tensor,
               zd: cute.Tensor, nblk: cute.Tensor, tma_ah: cute.CopyAtom, tah: cute.Tensor, tma_xb: cute.CopyAtom, txb: cute.Tensor,
               half_l: cute.ComposedLayout, a_blk: cute.Tensor, a_s: cute.Tensor,
               n_a: cute.Tensor, npairs: cute.Tensor, zcnt: cute.Tensor, a2s: cute.Tensor,
               tma_ahs: cute.CopyAtom, tahs: cute.Tensor, mma8: cute.TiledMma, tma_as: cute.CopyAtom, tas: cute.Tensor,
               tma_sfas: cute.CopyAtom, tsfas: cute.Tensor, xsrc: cute.Tensor,
               rtok: cute.Tensor, tma_wh: cute.CopyAtom, twh: cute.Tensor, half8_l,
               sA_l: cute.ComposedLayout,
               sB_l: cute.ComposedLayout, sSFA_l, sSFB_l,
               sA16_l: cute.ComposedLayout, sB16_l: cute.ComposedLayout,
               sAL_l: cute.ComposedLayout, sZL_l: cute.ComposedLayout,
               mma32: cute.TiledMma):
        S = self.stages
        LI = self.w_tile // (128 * self.kl * 2)
        bf = cutlass.BFloat16
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        et = tidx % 128

        smem = utils.SmemAllocator()
        st = smem.allocate(self.storage)
        sA16 = cute.make_tensor(cute.recast_ptr(st.sA.data_ptr(), swizzle_=sA16_l.inner,
                                                dtype=bf), sA16_l.outer)
        sB16 = cute.make_tensor(cute.recast_ptr(st.sB.data_ptr(), swizzle_=sB16_l.inner,
                                                dtype=bf), sB16_l.outer)
        if cutlass.const_expr(self.mx):
            sA = cute.make_tensor(cute.recast_ptr(st.sA.data_ptr(), swizzle_=sA_l.inner,
                                                  dtype=cutlass.Float8E4M3FN), sA_l.outer)
            sB = cute.make_tensor(cute.recast_ptr(st.sB.data_ptr(), swizzle_=sB_l.inner,
                                                  dtype=cutlass.Float8E4M3FN), sB_l.outer)
            sSFA = st.sSFA.get_tensor(sSFA_l)
            sSFB = st.sSFB.get_tensor(sSFB_l)
        else:
            sA, sB = sA16, sB16
        sZd = cute.make_tensor(
            cute.recast_ptr(st.sZd.data_ptr(), swizzle_=sZL_l.inner, dtype=bf), sZL_l.outer)
        sAL = cute.make_tensor(cute.recast_ptr(st.sA.data_ptr(), swizzle_=sAL_l.inner, dtype=bf),
                               sAL_l.outer)
        tslot = st.tslot.get_tensor(cute.make_layout(NT))
        full = st.full.data_ptr()
        empty = st.empty.data_ptr()
        zready = st.zready.data_ptr()
        zfree = st.zfree.data_ptr()
        acc_full = st.acc_full.data_ptr()
        acc_empty = st.acc_empty.data_ptr()
        uq = st.uq.get_tensor(cute.make_layout(QD))
        uq_full = st.uq_full.data_ptr()
        uq_empty = st.uq_empty.data_ptr()
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(QD):
                    cute.arch.mbarrier_init(uq_full + i, 1)
                    cute.arch.mbarrier_init(uq_empty + i, 2)  # MMA warp + epilogue warps
                for i in cutlass.range_constexpr(S):
                    cute.arch.mbarrier_init(full + i, 1 + (32 if self.xg else 0))
                    cute.arch.mbarrier_init(empty + i, 1)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(zready + i, 1)
                    cute.arch.mbarrier_init(zfree + i, 1)
                    cute.arch.mbarrier_init(acc_full + i, 1)
                    cute.arch.mbarrier_init(acc_empty + i, 1)
        cute.arch.mbarrier_init_fence()
        if warp == 1:
            cute.arch.alloc_tmem(self.tmem_cols, st.tmem_holder)
        cute.arch.barrier()
        acc_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, st.tmem_holder)
        acc_fake = mma.make_fragment_C(mma.partition_shape_C(self.tiler[:2]))
        acc16_fake = mma16.make_fragment_C(mma16.partition_shape_C(self.tiler16[:2]))
        cute.arch.griddepcontrol_wait()
        nb = nblk[0]
        nbs = cutlass.Int32(0)
        if cutlass.const_expr(self.shared_e >= 0):
            nbs = nblk[1]
        n_items = (nb - nbs) * self.m_tiles + nbs * self.m_tiles_sh
        G = self.grid
        na = cutlass.Int32(0)
        if cutlass.const_expr(self.fold_a):
            na = n_a[0] * self.a_ks
        n_units = na + n_items

        if warp == 0:
            one = cute.make_layout(1)
            thr = mma.get_slice(0)
            til = self.tiler if self.mx else self.tiler16
            c = cutlass.Int32(0)  # ring position, continuous across items
            lane = tidx % 32
            sb_u32 = cutlass.Int32(st.sB.data_ptr().toint())
            qk = cutlass.Int32(0)  # units published to the queue
            cur = cutlass.Int32(n_units)
            if bid < n_units:
                cur = bid
            # the unit after the next one (lane 0 only)
            pend = cutlass.Int32(n_units)
            if lane == 0:
                uq[0] = cur
                cute.arch.mbarrier_arrive(uq_full)
                pend = cur + G
            qk = qk + 1
            while cur < n_units:
              # publish the next unit before loading this one (the epilogue stages its Zblk)
              slot = qk % QD
              if qk >= QD:
                  cute.arch.mbarrier_wait(uq_empty + slot, (qk // QD - 1) % 2)
              if lane == 0:
                  nv = cutlass.min(pend, n_units)
                  uq[slot] = nv
                  cute.arch.mbarrier_arrive(uq_full + slot)
                  if nv < n_units:
                      pend = nv + G
              cute.arch.sync_warp()
              nxt = uq[slot]
              qk = qk + 1
              uu = cur
              if uu < na:  # A-item: LoRA-A shrink of one (block, slot pair)
                self.produce_a(uu, a_blk, a_s, blk_expert, tma_ah, tah, half_l, tma_xb, txb,
                               mma16, st, sB16, full, empty, c, tma_ahs, tahs, xsrc, rtok,
                               lane, sb_u32)
                c = c + self.a_ktiles
              else:
                blk, m_tile = self.decode(uu - na, nb, nbs)
                expert = blk_expert[blk]
                mask = blk_mask[blk]
                tokl = cutlass.Int32(-1)
                if cutlass.const_expr(self.xg):
                    tokl = rtok[blk * NT + lane // 2]
                nbase = cutlass.Int32(self.k_tiles)
                if cutlass.const_expr(self.mx_sh):
                    if expert == self.shared_e:
                        nbase = cutlass.Int32(self.k_tiles_sh)
                row_tile = expert * self.m_tiles * self.NA + m_tile
                nlt = self.NA * self.popc(mask)  # LoRA-B tiles, LI to a stage
                total = nbase + (nlt + LI - 1) // LI
                gA = cute.local_tile(ta, cute.slice_(til, (None, 0, None)), (row_tile, None, 0))
                gA1 = cute.local_tile(ta, cute.slice_(til, (None, 0, None)),
                                      (row_tile + self.m_tiles * (self.NA - 1), None, 0))
                gB = cute.local_tile(tb, cute.slice_(til, (0, None, None)), (blk, None, 0))
                tAsA, tAgA = cpasync.tma_partition(tma_a, 0, one, cute.group_modes(sA, 0, 3),
                                                   cute.group_modes(thr.partition_A(gA), 0, 3))
                _, tAgA1 = cpasync.tma_partition(tma_a, 0, one, cute.group_modes(sA, 0, 3),
                                                 cute.group_modes(thr.partition_A(gA1), 0, 3))
                is_sh = cutlass.Int32(0)
                tBsB, tBgB = cpasync.tma_partition(tma_b, 0, one, cute.group_modes(sB, 0, 3),
                                                   cute.group_modes(thr.partition_B(gB), 0, 3))
                bm_tile = row_tile
                if cutlass.const_expr(self.pair):  # mirror rows: [half, expert, feature]
                    bm_tile = expert * self.m_tiles + m_tile
                gBM = cute.local_tile(tbm, cute.slice_(self.tilerL, (None, 0, None)),
                                      (bm_tile, None, 0))
                tAsBM, tAgBM = cpasync.tma_partition(
                    tma_bm, 0, one, cute.group_modes(sAL, 0, 3),
                    cute.group_modes(mma16.get_slice(0).partition_A(gBM), 0, 3))
                gBM1 = cute.local_tile(tbm, cute.slice_(self.tilerL, (None, 0, None)),
                                       (bm_tile + self.E * self.m_tiles * (self.NA - 1),
                                        None, 0))
                _, tAgBM1 = cpasync.tma_partition(
                    tma_bm, 0, one, cute.group_modes(sAL, 0, 3),
                    cute.group_modes(mma16.get_slice(0).partition_A(gBM1), 0, 3))
                tAsSFA = tAgSFA = tBsSFB = tBgSFB = tAsA
                if cutlass.const_expr(self.mx):
                    gSFA = cute.local_tile(tsfa, cute.slice_(self.tiler, (None, 0, None)),
                                           (row_tile, None, 0))
                    gSFB = cute.local_tile(tsfb, cute.slice_(self.tiler_sfb, (0, None, None)),
                                           (blk, None, 0))
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
                tAgA8 = tAgA
                tAgSFA8 = tAgSFA
                if cutlass.const_expr(self.mx_sh):
                    if expert == self.shared_e:
                        is_sh = cutlass.Int32(1)
                    gA8 = cute.local_tile(tas, cute.slice_(self.tiler, (None, 0, None)),
                                          (m_tile, None, 0))
                    _, tAgA8 = cpasync.tma_partition(tma_as, 0, one, cute.group_modes(sA, 0, 3),
                                                     cute.group_modes(thr.partition_A(gA8), 0, 3))
                    gSFA8 = cute.local_tile(tsfas, cute.slice_(self.tiler, (None, 0, None)),
                                            (m_tile, None, 0))
                    _, tAgSFA8 = cpasync.tma_partition(
                        tma_sfas, 0, one, cute.group_modes(sSFA, 0, 3),
                        cute.group_modes(thr.partition_A(gSFA8), 0, 3))
                    tAgSFA8 = cute.filter_zeros(tAgSFA8)
                j = cutlass.Int32(0)
                for i in cutlass.range(total, unroll=1):
                    kt = i
                    s = c % S
                    if c >= S:
                        cute.arch.mbarrier_wait(empty + s, (c // S - 1) % 2)
                    if i < nbase:
                        with cute.arch.elect_one():
                            if is_sh != 0:
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    full + s, self.NA * self.w_bytes_sh + self.x_tx)
                            else:
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    full + s, self.NA * self.w_bytes + self.x_tx)
                        if cutlass.const_expr(self.mx_sh):
                            if is_sh != 0:
                                cute.copy(tma_as, tAgA8[(None, kt)], tAsA[(None, s)],
                                          tma_bar_ptr=full + s)
                                cute.copy(tma_sfas, tAgSFA8[(None, kt)], tAsSFA[(None, s)],
                                          tma_bar_ptr=full + s)
                            else:
                                self.copy_a(tma_a, tAgA, tAsA, tma_wh, twh, half8_l, st, s, kt,
                                            expert, m_tile, full)
                                cute.copy(tma_sfa, tAgSFA[(None, kt)], tAsSFA[(None, s)],
                                          tma_bar_ptr=full + s)
                        else:
                            self.copy_a(tma_a, tAgA, tAsA, tma_wh, twh, half8_l, st, s, kt,
                                        expert, m_tile, full)
                            if cutlass.const_expr(self.pair):
                                cute.copy(tma_a, tAgA1[(None, kt)], tAsA[(None, s * 2 + 1)],
                                          tma_bar_ptr=full + s)
                            if cutlass.const_expr(self.mx):
                                cute.copy(tma_sfa, tAgSFA[(None, kt)], tAsSFA[(None, s)],
                                          tma_bar_ptr=full + s)
                        if cutlass.const_expr(self.xg):
                            self.gather_x(xsrc, tokl, lane, sb_u32, s * self.AP, kt)
                            cp_async_arrive_noinc(cutlass.Int32((full + s).toint()))
                        else:
                            cute.copy(tma_b, tBgB[(None, kt)], tBsB[(None, s * self.AP)],
                                      tma_bar_ptr=full + s)
                        if cutlass.const_expr(self.mx):
                            cute.copy(tma_sfb, tBgSFB[(None, kt)], tBsSFB[(None, s)],
                                      tma_bar_ptr=full + s)
                    else:
                        # LoRA-B tiles (per slot: gate half, then up half), LI per stage
                        t0 = (i - nbase) * LI
                        nin = cutlass.min(nlt - t0, LI)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(full + s,
                                                                    nin * self.bm_bytes)
                        if cutlass.const_expr(self.xg):
                            cute.arch.mbarrier_arrive(full + s)  # the 32 gather arrivals
                        for pp in cutlass.range_constexpr(LI):
                            if pp < nin:
                                hh = (t0 + pp) % self.NA
                                if hh == 0:
                                    while ((mask >> j) & 1) == 0:
                                        j = j + 1
                                    cute.copy(tma_bm, tAgBM[(None, j)],
                                              tAsBM[(None, s * LI + pp)], tma_bar_ptr=full + s)
                                else:
                                    cute.copy(tma_bm, tAgBM1[(None, j)],
                                              tAsBM[(None, s * LI + pp)], tma_bar_ptr=full + s)
                                if hh == self.NA - 1:
                                    j = j + 1
                    c = c + 1
              cur = nxt
        elif warp == 1:
            self.mma_warp(mma, mma16, sA, sB, sA16, sB16, sSFA_l, sSFB_l, full, empty,
                          zready, zfree, acc_full, acc_empty, acc_fake, acc16_fake, acc_ptr, st,
                          sZd, blk_mask, bid, nb, n_units, na, sAL, mma8, nbs, blk_expert,
                          uq, uq_full, uq_empty, mma32)
        elif warp >= 4:
            # Zblk of unit it+1 is built before the epilogue of unit it, so the MMA warp never
            # waits on an epilogue for its LoRA operand.
            cute.arch.mbarrier_wait(uq_full, 0)
            uu = uq[0]
            if uu < n_units:
                self.stage_z(st, z, rslot, ranks, scales, tslot, uu, nb, nbs, zfree, zready,
                             cutlass.Int32(0), et, na, npairs, zcnt, blk_expert)
            it = cutlass.Int32(0)
            while uu < n_units:
                zb = it % 2
                q1 = (it + 1) % QD
                cute.arch.mbarrier_wait(uq_full + q1, ((it + 1) // QD) % 2)
                un = uq[q1]  # the next unit
                if uu < na:
                    # A-item: write the rows' Z, release the block, then stage the next unit
                    cute.arch.mbarrier_wait(acc_full + zb, (it // 2) % 2)
                    tAcc = cute.make_tensor(acc_ptr + zb * self.NA * NT, acc_fake.layout)
                    self.epilogue_a(tAcc, z, rslot, a_blk, a_s, zcnt, uu, et, acc_empty + zb)
                    if un < n_units:
                        self.stage_z(st, z, rslot, ranks, scales, tslot, un, nb, nbs, zfree,
                                     zready, it + 1, et, na, npairs, zcnt,
                                     blk_expert)
                else:
                    blk, m_tile = self.decode(uu - na, nb, nbs)
                    expert = blk_expert[blk]
                    if un < n_units:
                        self.stage_z(st, z, rslot, ranks, scales, tslot, un, nb, nbs, zfree,
                                     zready, it + 1, et, na, npairs, zcnt,
                                     blk_expert)
                    cute.arch.mbarrier_wait(acc_full + zb, (it // 2) % 2)
                    tAcc = cute.make_tensor(acc_ptr + zb * self.NA * NT, acc_fake.layout)
                    tAccL = cute.make_tensor(acc_ptr + self.l_col + zb * 2 * NT,
                                             acc_fake.layout)
                    has_l = cutlass.Int32(0)
                    if cutlass.const_expr(self.gu32):
                        has_l = cutlass.Int32(blk_mask[blk] != 0)
                    self.epilogue(st, tAcc, out, lut, a2, out_q, out_sf, zd, rslot, expert,
                                  blk, m_tile, et, acc_empty + zb, zb, a2s, tAccL, has_l)
                if et == 0:
                    cute.arch.mbarrier_arrive(uq_empty + it % QD)
                it = it + 1
                uu = un
        cute.arch.barrier()
        if warp == 1:
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(acc_ptr, self.tmem_cols)
        if tidx == 0:
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def stage_z(self, st, z, rslot, ranks, scales, tslot, uu, nb, nbs, zfree, zready, it, et,
                na, npairs, zcnt, blk_expert):
        """Zblk of unit uu (local index it) into buffer it % 2 (A-items have none); every unit
        signals zready so both sides count phases alike. With fold_a, the block's A-items must
        have released its Z first."""
        zb = it % 2
        if it >= 2:
            cute.arch.mbarrier_wait(zfree + zb, (it // 2 - 1) % 2)
        if uu >= na:
            blk, _mt = self.decode(uu - na, nb, nbs)
            if cutlass.const_expr(self.fold_a):
                if et == 0:
                    wait_count(zcnt.iterator + blk, npairs[blk] * self.a_ks)
            self.build_zblk(st, z, rslot, ranks, scales, tslot, blk, zb, et, blk_expert[blk])
        if et == 0:
            cute.arch.mbarrier_arrive(zready + zb)

    @cute.jit
    def epilogue_a(self, tAcc, z, rslot, a_blk, a_s, zcnt, aid, et, acc_empty):
        """A-item epilogue: accumulator lane et is rank row et % 64 of slot a_s[2 aid + et/64];
        each block row on that slot gets its Z value (one writer per value)."""
        epi = (128, NT)
        t2r_atom = sm100_utils.get_tmem_load_op(
            (128, NT, 64), utils.LayoutEnum.ROW_MAJOR, cutlass.Float32, cutlass.Float32, epi,
            False)
        acc_epi = cute.flat_divide(tAcc[((None, None), 0, 0)], epi)
        t2r = tcgen05.make_tmem_copy(t2r_atom, acc_epi[(None, None, 0, 0)])
        thr_t2r = t2r.get_slice(et)
        src = thr_t2r.partition_S(acc_epi)
        coords = thr_t2r.partition_D(cute.flat_divide(cute.make_identity_tensor((128, NT)), epi))
        r = cute.make_rmem_tensor(coords[(None, None, None, 0, 0)].shape, cutlass.Float32)
        cute.copy(t2r, src[(None, None, None, 0, 0)], r)
        cute.arch.fence_view_async_tmem_load()
        named_barrier(4, 128)
        if et == 0:
            cute.arch.mbarrier_arrive(acc_empty)
        cd = coords[(None, None, None, 0, 0)]
        if cutlass.const_expr(self.fold_a):  # (traced only where A-items exist)
            aid = aid // self.a_ks
            blk = a_blk[aid]
            for i in cutlass.range_constexpr(cute.size(r)):
                mm, t = cd[i]
                slot = a_s[2 * aid + mm // 64]
                row = blk * NT + t
                if (slot >= 0) & (rslot[row] == slot):
                    if cutlass.const_expr(self.zfix):
                        red_add_fix(z.iterator + cute.crd2idx((row, mm % 64), z.layout), r[i])
                    else:
                        z[row, 0, mm % 64] = r[i]
            named_barrier(4, 128)
            if et == 0:
                _asm(None, [], "fence.acq_rel.gpu;", "~{memory}")
                atomic_add_acq_rel(zcnt.iterator + blk, cutlass.Int32(1))

    @cute.jit
    def produce_a(self, aid, a_blk, a_s, blk_expert, tma_ah, tah, half_l, tma_xb, txb, mma16,
                  st, sB16, full, empty, c0, tma_ahs, tahs, xsrc, rtok, lane, sb_u32):
        """A-item aid: a_ktiles stages of [A_gu rows of slot s0 | of slot s1] x 64 K (two 64-row
        boxes into the two halves of the stage's A tile) and the block's BF16 activations."""
        if cutlass.const_expr(self.fold_a):
            self.produce_a_body(aid, a_blk, a_s, blk_expert, tma_ah, tah, half_l, tma_xb, txb,
                                mma16, st, sB16, full, empty, c0, tma_ahs, tahs, xsrc, rtok,
                                lane, sb_u32)

    @cute.jit
    def produce_a_body(self, aid, a_blk, a_s, blk_expert, tma_ah, tah, half_l, tma_xb, txb,
                       mma16, st, sB16, full, empty, c0, tma_ahs, tahs, xsrc, rtok, lane,
                       sb_u32):
        S = self.stages
        bf = cutlass.BFloat16
        kp = aid % self.a_ks  # K part of the A-item
        aid = aid // self.a_ks
        blk = a_blk[aid]
        s0 = a_s[2 * aid]
        s1 = a_s[2 * aid + 1]
        ea = cutlass.Int32(0)
        if cutlass.const_expr(self.a_experts > 1):
            ea = blk_expert[blk]
        r0 = s0 * self.a_experts + ea
        r1 = cutlass.max(s1, 0) * self.a_experts + ea
        is_sh = cutlass.Int32(0)
        if cutlass.const_expr(self.shared_e >= 0):
            if blk_expert[blk] == self.shared_e:  # dense [slots, 64, K] buffer, row = slot
                is_sh = cutlass.Int32(1)
                r0 = s0
                r1 = cutlass.max(s1, 0)
        one = cute.make_layout(1)
        gX = cute.local_tile(txb, cute.slice_(self.tiler16, (0, None, None)), (blk, None, 0))
        tXs, tXg = cpasync.tma_partition(tma_xb, 0, one, cute.group_modes(sB16, 0, 3),
                                         cute.group_modes(mma16.get_slice(0).partition_B(gX),
                                                          0, 3))
        g0 = cute.local_tile(tah, (64, 64), (r0, None))
        g1 = cute.local_tile(tah, (64, 64), (r1, None))
        g0s = cute.local_tile(tahs, (64, 64), (r0, None))
        g1s = cute.local_tile(tahs, (64, 64), (r1, None))
        nbytes = self.ah_bytes + (0 if self.xg else self.xb_bytes)
        if s1 >= 0:
            nbytes = nbytes + self.ah_bytes
        tokl = cutlass.Int32(-1)
        if cutlass.const_expr(self.xg):
            tokl = rtok[blk * NT + lane // 2]
        for kt in cutlass.range(self.a_ktiles, unroll=1):
            c = c0 + kt
            s = c % S
            if c >= S:
                cute.arch.mbarrier_wait(empty + s, (c // S - 1) % 2)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(full + s, nbytes * self.AP)
            for q in cutlass.range_constexpr(self.AP):
                ks_ = (kp * self.a_ktiles + kt) * self.AP + q  # 64-wide K step
                base_q = st.sA.data_ptr() + s * self.w_tile + q * 2 * self.ah_bytes
                sA0 = cute.make_tensor(cute.recast_ptr(base_q, swizzle_=half_l.inner, dtype=bf),
                                       half_l.outer)
                tAs0, tAg0 = cpasync.tma_partition(tma_ah, 0, one, cute.group_modes(sA0, 0, 2),
                                                   cute.group_modes(g0, 0, 2))
                if is_sh != 0:
                    tAs0s, tAg0s = cpasync.tma_partition(tma_ahs, 0, one,
                                                         cute.group_modes(sA0, 0, 2),
                                                         cute.group_modes(g0s, 0, 2))
                    cute.copy(tma_ahs, tAg0s[(None, ks_)], tAs0s, tma_bar_ptr=full + s)
                else:
                    cute.copy(tma_ah, tAg0[(None, ks_)], tAs0, tma_bar_ptr=full + s)
                if s1 >= 0:
                    sA1 = cute.make_tensor(
                        cute.recast_ptr(base_q + self.ah_bytes, swizzle_=half_l.inner, dtype=bf),
                        half_l.outer)
                    tAs1, tAg1 = cpasync.tma_partition(tma_ah, 0, one,
                                                       cute.group_modes(sA1, 0, 2),
                                                       cute.group_modes(g1, 0, 2))
                    if is_sh != 0:
                        tAs1s, tAg1s = cpasync.tma_partition(tma_ahs, 0, one,
                                                             cute.group_modes(sA1, 0, 2),
                                                             cute.group_modes(g1s, 0, 2))
                        cute.copy(tma_ahs, tAg1s[(None, ks_)], tAs1s, tma_bar_ptr=full + s)
                    else:
                        cute.copy(tma_ah, tAg1[(None, ks_)], tAs1, tma_bar_ptr=full + s)
                if cutlass.const_expr(self.xg):
                    self.gather_x(xsrc, tokl, lane, sb_u32, s * self.AP + q, ks_)
                else:
                    cute.copy(tma_xb, tXg[(None, ks_)], tXs[(None, s * self.AP + q)],
                              tma_bar_ptr=full + s)
            if cutlass.const_expr(self.xg):
                cp_async_arrive_noinc(cutlass.Int32((full + s).toint()))

    @cute.jit
    def gather_x(self, xsrc, tokl, lane, sb_u32, tile, kstep):
        """Activation tile [16 rows, 64 K] (BF16, SW128 K-major) of 64-wide K step kstep into
        sB tile `tile`: lane l copies four 16-byte chunks of block row l // 2 (token tokl;
        none: zeros) from the token-ordered hidden states."""
        n = lane // 2
        ok = tokl >= 0
        row = cutlass.max(tokl, 0)
        src0 = xsrc.iterator.toint() + (cutlass.Int64(row) * self.K + kstep * 64) * 2
        dst0 = sb_u32 + tile * (NT * 128) + n * 128
        nbytes = cutlass.Int32(16)
        if not ok:
            nbytes = cutlass.Int32(0)
        for q in cutlass.range_constexpr(4):
            cc = (lane % 2) * 4 + q
            cp_async16(dst0 + ((cc ^ (n % 8)) << 4), src0 + cc * 16, nbytes)

    @cute.jit
    def copy_a(self, tma_a, tAgA, tAsA, tma_wh, twh, half8_l, st, s, kt, expert, m_tile, full):
        """Base weight K-tile kt of the unit into stage s: one 128-row box, or (mxfp8 gate/up)
        the tile's 64 gate rows and 64 up rows of the expert's [gate; up] weight."""
        if cutlass.const_expr(self.halves):
            one = cute.make_layout(1)
            for h in cutlass.range_constexpr(2):
                row = expert * self.N + h * (self.N // 2) + m_tile * 64
                g = cute.local_tile(twh, (64, self.ke), (row // 64, None))
                sh = cute.make_tensor(cute.recast_ptr(
                    st.sA.data_ptr() + s * self.w_tile + h * 64 * self.ke,
                    swizzle_=half8_l.inner, dtype=cutlass.Float8E4M3FN), half8_l.outer)
                tS, tG = cpasync.tma_partition(tma_wh, 0, one, cute.group_modes(sh, 0, 2),
                                               cute.group_modes(g, 0, 2))
                cute.copy(tma_wh, tG[(None, kt)], tS, tma_bar_ptr=full + s)
        else:
            cute.copy(tma_a, tAgA[(None, kt)], tAsA[(None, s * self.NA)], tma_bar_ptr=full + s)

    def decode(self, item, nb, nbs):
        """Work item -> (block, row tile): routed blocks' items first, then the shared tail."""
        nr = nb - nbs
        r_items = nr * self.m_tiles
        if cutlass.const_expr(self.order == "e"):
            blk = item // self.m_tiles
            m_tile = item % self.m_tiles
        else:
            blk = item % cutlass.max(nr, 1)
            m_tile = item // cutlass.max(nr, 1)
        if cutlass.const_expr(self.shared_e >= 0):
            j = item - r_items
            sh = cutlass.Int32(item >= r_items)
            if cutlass.const_expr(self.order == "e"):
                blk = sh * (nr + j // self.m_tiles_sh) + (1 - sh) * blk
                m_tile = sh * (j % self.m_tiles_sh) + (1 - sh) * m_tile
            else:
                blk = sh * (nr + j % cutlass.max(nbs, 1)) + (1 - sh) * blk
                m_tile = sh * (j // cutlass.max(nbs, 1)) + (1 - sh) * m_tile
        return blk, m_tile

    def popc(self, mask):
        n = cutlass.Int32(0)
        for j in range(self.lora_tiles):
            n = n + ((mask >> j) & 1)
        return n

    @cute.jit
    def mma_warp(self, mma, mma16, sA, sB, sA16, sB16, sSFA_l, sSFB_l, full, empty, zready,
                 zfree, acc_full, acc_empty, acc_fake, acc16_fake, acc_ptr, st, sZd, blk_mask,
                 bid, nb, n_units, na, sAL, mma8, nbs, blk_expert, uq, uq_full, uq_empty,
                 mma32):
        S = self.stages
        tCrA = mma.make_fragment_A(sA)
        tCrA8 = mma8.make_fragment_A(sA)
        tCrB = mma.make_fragment_B(sB)
        tCrBM = mma16.make_fragment_A(sA16)
        if cutlass.const_expr(self.gu32):
            tCrZd = mma32.make_fragment_B(sZd)
            tCrBL = mma32.make_fragment_A(sAL)
            accL_fake = mma32.make_fragment_C(mma32.partition_shape_C(self.tilerZ[:2]))
        else:
            tCrZd = mma16.make_fragment_B(sZd)
            tCrBL = mma16.make_fragment_A(sAL)
        nkbL = cute.size(tCrBL, mode=[2])
        LI = self.w_tile // (128 * self.kl * 2)
        tCrX16 = mma16.make_fragment_B(sB16)
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
        c = cutlass.Int32(0)
        it = cutlass.Int32(0)
        cute.arch.mbarrier_wait(uq_full, 0)
        uu = uq[0]
        while uu < n_units:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(uq_empty + it % QD)
            blk, _mt = self.decode(uu - na, nb, nbs)
            mask = cutlass.Int32(0)
            nbase = cutlass.Int32(self.a_ktiles)
            is_sh = cutlass.Int32(0)
            if uu >= na:
                mask = blk_mask[blk]
                nbase = cutlass.Int32(self.k_tiles)
                if cutlass.const_expr(self.mx_sh):
                    if blk_expert[blk] == self.shared_e:
                        nbase = cutlass.Int32(self.k_tiles_sh)
                        is_sh = cutlass.Int32(1)
            nlt = self.NA * self.popc(mask)
            total = nbase + (nlt + LI - 1) // LI
            ab = it % 2
            if it >= 2:
                cute.arch.mbarrier_wait(acc_empty + ab, (it // 2 - 1) % 2)
            tAcc = cute.make_tensor(acc_ptr + ab * self.NA * NT, acc_fake.layout)
            tAcc16 = cute.make_tensor(acc_ptr + ab * self.NA * NT, acc16_fake.layout)
            tAcc16b = cute.make_tensor(acc_ptr + ab * self.NA * NT + NT, acc16_fake.layout)
            j = cutlass.Int32(0)
            for kt in cutlass.range(total, unroll=1):
                s = c % S
                cute.arch.mbarrier_wait(full + s, (c // S) % 2)
                if cutlass.const_expr(self.xg):
                    fence_proxy_async()  # rows written by cp.async (generic proxy)
                if uu < na:  # BF16 LoRA-A tiles x BF16 activations (AP K-steps per stage)
                    for q in cutlass.range_constexpr(self.AP):
                        for kb in cutlass.range_constexpr(nkb):
                            mma16.set(tcgen05.Field.ACCUMULATE, (kt != 0) | (q != 0) | (kb != 0))
                            cute.gemm(mma16, tAcc16, tCrBM[(None, None, kb, s * self.NA + q)],
                                      tCrX16[(None, None, kb, s * self.AP + q)], tAcc16)
                elif kt < nbase:
                    if cutlass.const_expr(self.mx):
                        cute.copy(cp_sfa, src_sfa[(None, None, None, None, s)], dst_sfa)
                        cute.copy(cp_sfb, src_sfb[(None, None, None, None, s)], dst_sfb)
                        if is_sh != 0:  # FP8 shared expert (kind::mxf8f6f4, e4m3 A)
                            for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                                mma8.set(tcgen05.Field.ACCUMULATE, (kt != 0) | (kb != 0))
                                mma8.set(tcgen05.Field.SFA, tCtSFA[(None, None, kb)].iterator)
                                mma8.set(tcgen05.Field.SFB, tCtSFB[(None, None, kb)].iterator)
                                cute.gemm(mma8, tAcc, tCrA8[(None, None, kb, s)],
                                          tCrB[(None, None, kb, s)], tAcc)
                        else:
                            for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                                mma.set(tcgen05.Field.ACCUMULATE, (kt != 0) | (kb != 0))
                                mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kb)].iterator)
                                mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kb)].iterator)
                                cute.gemm(mma, tAcc, tCrA[(None, None, kb, s)],
                                          tCrB[(None, None, kb, s)], tAcc)
                    else:
                        for kb in cutlass.range_constexpr(nkb):
                            mma16.set(tcgen05.Field.ACCUMULATE, (kt != 0) | (kb != 0))
                            cute.gemm(mma16, tAcc16, tCrBM[(None, None, kb, s * self.NA)],
                                      tCrX16[(None, None, kb, s * self.AP)], tAcc16)
                        if cutlass.const_expr(self.pair):
                            for kb in cutlass.range_constexpr(nkb):
                                mma16.set(tcgen05.Field.ACCUMULATE, (kt != 0) | (kb != 0))
                                cute.gemm(mma16, tAcc16b, tCrBM[(None, None, kb, s * 2 + 1)],
                                          tCrX16[(None, None, kb, s * self.AP)], tAcc16b)
                else:
                    if kt == nbase:
                        cute.arch.mbarrier_wait(zready + ab, (it // 2) % 2)
                    t0 = (kt - nbase) * LI
                    nin = cutlass.min(nlt - t0, LI)
                    if cutlass.const_expr(self.gu32):
                      tAccL = cute.make_tensor(acc_ptr + self.l_col + ab * 2 * NT,
                                               accL_fake.layout)
                      for pp in cutlass.range_constexpr(LI):
                        if pp < nin:
                            while ((mask >> j) & 1) == 0:
                                j = j + 1
                            zi = ab * (self.zbuf // self.ztile) + j
                            for kb in cutlass.range_constexpr(nkbL):
                                mma32.set(tcgen05.Field.ACCUMULATE,
                                          (kt != nbase) | (pp != 0) | (kb != 0))
                                cute.gemm(mma32, tAccL, tCrBL[(None, None, kb, s * LI + pp)],
                                          tCrZd[(None, None, kb, zi)], tAccL)
                            j = j + 1
                    for pp in cutlass.range_constexpr(0 if self.gu32 else LI):
                        if pp < nin:
                            hh = (t0 + pp) % self.NA
                            if hh == 0:
                                while ((mask >> j) & 1) == 0:
                                    j = j + 1
                            zi = ab * (self.zbuf // self.ztile) + hh * self.lora_tiles + j
                            if hh == 0:
                                for kb in cutlass.range_constexpr(nkbL):
                                    mma16.set(tcgen05.Field.ACCUMULATE, True)
                                    cute.gemm(mma16, tAcc16, tCrBL[(None, None, kb, s * LI + pp)],
                                              tCrZd[(None, None, kb, zi)], tAcc16)
                            else:
                                for kb in cutlass.range_constexpr(nkbL):
                                    mma16.set(tcgen05.Field.ACCUMULATE, True)
                                    cute.gemm(mma16, tAcc16b,
                                              tCrBL[(None, None, kb, s * LI + pp)],
                                              tCrZd[(None, None, kb, zi)], tAcc16b)
                            if hh == self.NA - 1:
                                j = j + 1
                with cute.arch.elect_one():
                    tcgen05.commit(empty + s)
                c = c + 1
            if total == nbase:  # no LoRA tile: the Zblk was not read
                cute.arch.mbarrier_wait(zready + ab, (it // 2) % 2)
            with cute.arch.elect_one():
                tcgen05.commit(acc_full + ab)
                tcgen05.commit(zfree + ab)
            it = it + 1
            cute.arch.mbarrier_wait(uq_full + it % QD, (it // QD) % 2)
            uu = uq[it % QD]

    @cute.jit
    def build_zblk(self, st, z, rslot, ranks, scales, tslot, blk, zb, et, expert):
        """sZd = lora_tiles x [16 rows, 64] BF16 (SW128, K-major). Row n of tile j holds, in
        columns (slot % spt) * W .. + W, bf16(scale * sum_p z[row, p, :]) (rank masked) when
        the row's slot is in tile j; zeros elsewhere."""
        base = cutlass.Int32(st.sZd.data_ptr().toint()) + zb * self.zbuf
        zero = cutlass.Int32(0)
        words16 = self.zbytes // 16
        for i in cutlass.range_constexpr((words16 + 127) // 128):
            wd = et + i * 128
            if wd < words16:
                st_shared_v4(base + wd * 16, [zero, zero, zero, zero])
        if et < NT:
            v = rslot[blk * NT + et]
            if (v < 0) | (v >= self.n_slots):
                v = cutlass.Int32(-1)
            tslot[et] = v
        named_barrier(2, 128)
        G = self.W // 8  # 8-wide groups per row
        tile = self.ztile
        rowb = self.kl * 2  # bytes per Zblk row (128: SW128, 64: SW64)
        for hz in cutlass.range_constexpr(self.NA):  # pair: gate Z tiles, then up Z tiles
          for i in cutlass.range_constexpr((NT * G + 127) // 128):
            task = et + i * 128
            n = task // G
            g = task % G
            if n < NT:
                  sl = tslot[n]
                  if sl >= 0:
                      row = blk * NT + n
                      acc = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Float32)
                      acc.fill(0.0)
                      if cutlass.const_expr(self.zfix):
                          ptr = z.iterator + cute.crd2idx((row, hz * self.W + g * 8), z.layout)
                          for q in cutlass.range_constexpr(8):
                              acc[q] = fix_to_f32(load_cg_s64(ptr + q))
                      else:
                          for p in cutlass.range(self.n_parts, unroll=4):
                              ptr = z.iterator + cute.crd2idx((row, p, g * 8), z.layout)
                              lo = load_cg4(ptr)
                              hi = load_cg4(ptr + 4)
                              for q in cutlass.range_constexpr(4):
                                  acc[q] = acc[q] + lo[q]
                                  acc[4 + q] = acc[4 + q] + hi[q]
                      rk = ranks[sl]
                      sc = scales[sl]
                      if cutlass.const_expr(self.shared_e >= 0):  # routed: pre-scaled B
                          if expert != self.shared_e:
                              rk = cutlass.Int32(32)
                              sc = cutlass.Float32(1.0)
                      words = []
                      for q in cutlass.range_constexpr(4):
                          pr = []
                          for h in cutlass.range_constexpr(2):
                              col = (g * 8 + 2 * q + h) % 32  # rank index within the slice
                              pv = cutlass.Float32(0)
                              if col < rk:
                                  pv = acc[2 * q + h] * sc
                              pr.append(pv)
                          words.append(cutlass.Int32(_asm(
                              ir.IntegerType.get_signless(32), [pr[1].ir_value(), pr[0].ir_value()],
                              "cvt.rn.bf16x2.f32 $0,$1,$2;", "=r,f,f")))
                      jt = sl // self.spt
                      gg = (sl % self.spt) * G + g
                      if cutlass.const_expr(self.gu32):
                          # gate columns -> row n, up columns -> row NT + n (64-byte rows, SW64)
                          zr = n + NT * (g // 4)
                          st_shared_v4(base + jt * tile + zr * 64 +
                                       (((g % 4) ^ ((zr >> 1) & 3)) << 4), words)
                      elif cutlass.const_expr(self.l32):
                          st_shared_v4(base + (hz * self.lora_tiles + jt) * tile + n * rowb +
                                       ((gg ^ ((n >> 1) & 3)) << 4), words)
                      else:
                          st_shared_v4(base + (hz * self.lora_tiles + jt) * tile + n * rowb +
                                       ((gg ^ (n & 7)) << 4), words)
        fence_proxy_async()
        named_barrier(2, 128)

    @cute.jit
    def epilogue(self, st, tAcc, out, lut, a2, out_q, out_sf, zd, rslot, expert, blk, m_tile,
                 et, acc_empty, zb, a2s, tAccL, has_l):
        epi = (128, NT)
        t2r_atom = sm100_utils.get_tmem_load_op(
            (128, NT, 64), utils.LayoutEnum.ROW_MAJOR, cutlass.Float32, cutlass.Float32, epi,
            False)
        acc_epi = cute.flat_divide(tAcc[((None, None), 0, 0)], epi)
        t2r = tcgen05.make_tmem_copy(t2r_atom, acc_epi[(None, None, 0, 0)])
        thr_t2r = t2r.get_slice(et)
        src = thr_t2r.partition_S(acc_epi)
        coords = thr_t2r.partition_D(cute.flat_divide(cute.make_identity_tensor((128, NT)), epi))
        r = cute.make_rmem_tensor(coords[(None, None, None, 0, 0)].shape, cutlass.Float32)
        cute.copy(t2r, src[(None, None, None, 0, 0)], r)
        r1 = cute.make_rmem_tensor(coords[(None, None, None, 0, 0)].shape, cutlass.Float32)
        if cutlass.const_expr(self.pair):  # up half: the accumulator NT columns over
            tAcc1 = cute.make_tensor(tAcc.iterator + NT, tAcc.layout)
            src1 = thr_t2r.partition_S(cute.flat_divide(tAcc1[((None, None), 0, 0)], epi))
            cute.copy(t2r, src1[(None, None, None, 0, 0)], r1)
        rg = cute.make_rmem_tensor(coords[(None, None, None, 0, 0)].shape, cutlass.Float32)
        ru = cute.make_rmem_tensor(coords[(None, None, None, 0, 0)].shape, cutlass.Float32)
        if cutlass.const_expr(self.gu32):  # LoRA accumulator: gate-Z columns | up-Z columns
            if has_l != 0:
                tAccU = cute.make_tensor(tAccL.iterator + NT, tAccL.layout)
                srcg = thr_t2r.partition_S(cute.flat_divide(tAccL[((None, None), 0, 0)], epi))
                srcu = thr_t2r.partition_S(cute.flat_divide(tAccU[((None, None), 0, 0)], epi))
                cute.copy(t2r, srcg[(None, None, None, 0, 0)], rg)
                cute.copy(t2r, srcu[(None, None, None, 0, 0)], ru)
        cute.arch.fence_view_async_tmem_load()
        named_barrier(4, 128)
        if et == 0:
            cute.arch.mbarrier_arrive(acc_empty)  # the MMA warp may reuse this accumulator
        cd = coords[(None, None, None, 0, 0)]
        if cutlass.const_expr(self.gu32):
            if has_l != 0:
                for i in cutlass.range_constexpr(cute.size(r)):
                    mm, t = cd[i]
                    if (lut[m_tile * 128 + mm] >> 8) != 0:
                        r[i] = r[i] + ru[i]
                    else:
                        r[i] = r[i] + rg[i]
        self.finish(st, r, cd, out, lut, a2, out_q, out_sf, zd, rslot, expert, blk, m_tile, et,
                    zb, r1, a2s)

    @cute.jit
    def finish(self, st, r, cd, out, lut, a2, out_q, out_sf, zd, rslot, expert, blk, m_tile,
               et, zb, r1, a2s):
        if cutlass.const_expr(self.epi == "raw"):
            for i in cutlass.range_constexpr(cute.size(r)):
                mm, t = cd[i]
                out[blk * NT + t, m_tile * 128 + mm] = r[i]
        elif cutlass.const_expr(self.epi == "swiglu_bf"):
            self.swiglu_bf_epilogue(st, r, r1, cd, a2, out_q, zd, rslot, expert, blk, m_tile,
                                    et, zb, a2s)
        elif cutlass.const_expr(self.epi == "down"):
            for i in cutlass.range_constexpr(cute.size(r)):
                mm, t = cd[i]
                out[blk * NT + t, lut[m_tile * 128 + mm]] = r[i].to(cutlass.BFloat16)
        else:
            self.swiglu_epilogue(st, r, cd, lut, a2, out_q, out_sf, zd, rslot, expert, blk,
                                 m_tile, et, zb, a2s)

    @cute.jit
    def swiglu_bf_epilogue(self, st, r, r1, cd, a2, out_q, zd, rslot, expert, blk, m_tile, et,
                           zb, a2s):
        """BF16 [gate; up] pair: act = silu(gate) * up (clamped when clamp > 0) -> BF16 into
        out_q [rows, I] (the down kernel's activations), and the down-LoRA shrink partials
        of this tile's 128 features (Q32.32 atomics)."""
        f32 = cutlass.Float32
        sbase = cute.recast_ptr(st.sZd.data_ptr(), dtype=f32) + zb * (self.zbuf // 4)
        act = cute.make_tensor(sbase, cute.make_layout((NT, 128), stride=(129, 1)))
        for i in cutlass.range_constexpr(cute.size(r)):
            mm, t = cd[i]
            g = r[i]
            u = r1[i]
            if cutlass.const_expr(self.clamp > 0):
                g = cute.arch.fmin(g, f32(self.clamp))
                u = cute.arch.fmax(cute.arch.fmin(u, f32(self.clamp)), -f32(self.clamp))
            a = (g * cute.arch.rcp_approx(f32(1.0) + cute.math.exp(-g, fastmath=True)) * u)
            ab = a.to(cutlass.BFloat16)
            out_q[blk * NT + t, m_tile * 128 + mm] = ab
            act[t, mm] = ab.to(f32)
        named_barrier(3, 128)
        t = et // 8
        row = blk * NT + t
        slot = rslot[row]
        if (slot >= 0) & (slot < self.n_slots):
            for jj in cutlass.range_constexpr(4):
                j = (et % 8) * 4 + jj
                arow = (slot * self.a2_experts + expert) * 32 + j
                accz = f32(0)
                abase = a2.iterator + cute.crd2idx((arow, m_tile * 128), a2.layout)
                if cutlass.const_expr(self.shared_e >= 0):
                    if expert == self.shared_e:  # the shared expert's dense down-LoRA A
                        abase = a2s.iterator + cute.crd2idx((slot * 32 + j, m_tile * 128),
                                                            a2s.layout)
                for f8i in cutlass.range_constexpr(16):
                    wv = load_bf16x8(abase + f8i * 8)
                    for q in cutlass.range_constexpr(8):
                        accz = accz + wv[q] * act[t, f8i * 8 + q]
                red_add_fix(zd.iterator + cute.crd2idx((row, j), zd.layout), accz)
        named_barrier(3, 128)

    @cute.jit
    def swiglu_epilogue(self, st, r, cd, lut, a2, out_q, out_sf, zd, rslot, expert, blk, m_tile,
                        et, zb, a2s):
        f32 = cutlass.Float32
        # FP32 staging in the drained ring: scr[is_up, feature, token], act[token, feature]
        sbase = cute.recast_ptr(st.sZd.data_ptr(), dtype=f32) + zb * (self.zbuf // 4)
        scr = cute.make_tensor(sbase, cute.make_layout((2, 64, NT),
                                                       stride=(64 * (NT + 1), NT + 1, 1)))
        act = cute.make_tensor(sbase + 2 * 64 * (NT + 1),
                               cute.make_layout((NT, 64), stride=(65, 1)))
        for i in cutlass.range_constexpr(cute.size(r)):
            mm, t = cd[i]
            info = lut[m_tile * 128 + mm]
            scr[info >> 8, info & 255, t] = r[i]
        named_barrier(3, 128)
        t = et // 8
        fg = et % 8
        L = f32(self.clamp)
        v = cute.make_rmem_tensor(cute.make_layout(8), f32)
        amax = f32(0)
        for q in cutlass.range_constexpr(8):
            f = fg * 8 + q
            g = cute.arch.fmin(scr[0, f, t], L)
            u = cute.arch.fmax(cute.arch.fmin(scr[1, f, t], L), -L)
            a = g * cute.arch.rcp_approx(f32(1.0) + cute.math.exp(-g, fastmath=True)) * u
            a = a.to(cutlass.BFloat16).to(f32)
            v[q] = a
            act[t, f] = a
            amax = cute.arch.fmax(amax, cute.arch.fmax(a, -a))
        amax = cute.arch.fmax(amax, shfl_xor_f32(amax, 1))
        amax = cute.arch.fmax(amax, shfl_xor_f32(amax, 2))
        # UE8M0 scale as TRT-LLM's cvt_warp_fp16_to_mxfp8: e8m0(amax / 448, round up)
        e8 = ue8m0_round_up(amax * f32(1.0 / 448.0))
        inv = f32(0)
        if e8 != 0:
            inv = exp2_int(127 - e8)
        w0 = cvt_e4m3x4(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv)
        w1 = cvt_e4m3x4(v[4] * inv, v[5] * inv, v[6] * inv, v[7] * inv)
        row = blk * NT + t
        I = self.inter
        st_global_v2(out_q.iterator + cute.crd2idx((row, m_tile * 64 + fg * 8), out_q.layout),
                     [w0, w1])
        if fg % 4 == 0:
            rr = blk * 128 + t
            c = m_tile * 2 + fg // 4
            nkt = (I // 32 + 3) // 4
            off = (rr // 128) * (nkt * 512) + (c // 4) * 512 + (rr % 32) * 16 + \
                ((rr % 128) // 32) * 4 + c % 4
            out_sf[off] = cutlass.Uint8(e8)
        named_barrier(3, 128)
        # down-LoRA shrink partial for this tile's 64 features. Only a block's routed rows
        # (often 1-3 of 16) carry a slot: warp w takes rows w, w + 4, ...; lane = rank j loads
        # its A2 row's 64 features at once (one memory round trip per row instead of four)
        # and adds the dot product to zd[row, j] (same products and order per rank as before).
        wq = et // 32
        lane = et % 32
        for rr in cutlass.range_constexpr(NT // 4):
            t2 = wq + rr * 4
            row2 = blk * NT + t2
            slot = rslot[row2]
            if (slot >= 0) & (slot < self.n_slots):
                arow = (slot * self.a2_experts + expert) * 32 + lane
                abase = a2.iterator + cute.crd2idx((arow, m_tile * 64), a2.layout)
                if cutlass.const_expr(self.shared_e >= 0):
                    if expert == self.shared_e:  # the shared expert's dense down-LoRA A
                        abase = a2s.iterator + cute.crd2idx((slot * 32 + lane, m_tile * 64),
                                                            a2s.layout)
                wv = [load_bf16x8(abase + f8i * 8) for f8i in range(8)]
                accz = f32(0)
                for f8i in cutlass.range_constexpr(8):
                    for q in cutlass.range_constexpr(8):
                        accz = accz + wv[f8i][q] * act[t2, f8i * 8 + q]
                if cutlass.const_expr(self.zd_fix):
                    red_add_fix(zd.iterator + cute.crd2idx((row2, lane), zd.layout), accz)
                else:
                    zd[row2, m_tile, lane] = accz
        named_barrier(3, 128)



def fold_refs(op, dev, x_bf=None, a_gu=None, a_blk=None, a_s=None, n_a=None, npairs=None,
              zcnt=None, a2s=None, a_gu_s=None, w_shared=None):
    """fold_a / shared-expert tensors (x_bf [rows_max, K] BF16, a_gu [slots * experts * 64, K]
    BF16, the A-item tables, the shared expert's down A [slots * 32, I] and gate/up A
    [slots * 64, K]) or placeholders shaped for the TMA descriptors."""
    import torch
    i32 = lambda n: torch.zeros(n, dtype=torch.int32, device=dev)
    if x_bf is None:
        x_bf = torch.zeros(op.rows_max, op.K, dtype=torch.bfloat16, device=dev)
        a_gu = torch.zeros(128, op.K, dtype=torch.bfloat16, device=dev)
        a_blk, a_s, n_a, npairs, zcnt = i32(16), i32(32), i32(1), i32(16), i32(16)
    if a2s is None:
        a2s = torch.zeros(32, 16, dtype=torch.bfloat16, device=dev)
    if a_gu_s is None:
        a_gu_s = torch.zeros(128, op.K, dtype=torch.bfloat16, device=dev)
    if op.mx:  # MXFP8 shared expert: (FP8 [rows_sh, K_sh], swizzled UE8M0 SF) or placeholders
        if w_shared is None:
            w_shared = (torch.zeros(op.rows_sh * op.K_sh, dtype=torch.uint8, device=dev),
                        torch.full((op.rows_sh * op.K_sh // 32,), 127, dtype=torch.uint8,
                                   device=dev))
        w8, sf8 = w_shared
        tail = [w8.view(torch.uint8).reshape(-1), sf8.view(torch.uint8).reshape(-1)]
    else:
        # BF16: unused shared-weight operands (placeholders)
        tail = [torch.zeros(128 * op.NA, op.K, dtype=torch.bfloat16, device=dev).unsqueeze(-1),
                torch.zeros(16, dtype=torch.uint8, device=dev)]
    return [x_bf, a_gu.reshape(-1, op.K), a_blk, a_s, n_a, npairs, zcnt, a2s,
            a_gu_s.reshape(-1, op.K)] + tail


def make(op, w, w_sf, x_q, x_sf, bm, z, rslot, blk_expert, blk_mask, ranks, scales, out,
         lut=None, a2=None, out_q=None, out_sf=None, zd=None, nblk=None, x_bf=None, a_gu=None, a_blk=None, a_s=None, n_a=None, npairs=None, zcnt=None,
         a2s=None, a_gu_s=None, w_shared=None, compiled=None, rtok=None):
    """Test/runner helper: w [E, rows, K/2] uint8 (FP4) or [E, rows, K] BF16; w_sf the
    swizzled per-expert UE8M0 scales (flat); x_q [rows_max, K] FP8 (or BF16); x_sf flat
    [max_blocks * 128 * K/32] swizzled; bm [E * rows, slots * W] BF16; z [rows_max, parts, W]
    FP32; rslot [rows_max]; blk_expert / blk_mask [max_blocks]; ranks / scales [slots]."""
    import torch
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack

    if op.mx:
        refs = [w.view(torch.uint8).reshape(-1), w_sf.view(torch.uint8).reshape(-1),
                x_q.view(torch.uint8).reshape(-1), x_sf.view(torch.uint8).reshape(-1)]
    else:
        refs = [w.reshape(-1, w.shape[-1]).unsqueeze(-1), torch.zeros(16, dtype=torch.uint8,
                                                                      device=w.device),
                x_q.reshape(-1), torch.zeros(16, dtype=torch.uint8, device=w.device)]
    dev = w.device
    i32 = torch.zeros(16, dtype=torch.int32, device=dev)
    refs += [bm.unsqueeze(-1), z, rslot, blk_expert, blk_mask, ranks, scales, out,
             lut if lut is not None else i32,
             a2 if a2 is not None else torch.zeros(1, 16, dtype=torch.bfloat16, device=dev),
             (out_q if out_q.dtype == torch.bfloat16 else out_q.view(torch.uint8))
             if out_q is not None else torch.zeros(
                 1, 16, dtype=torch.uint8, device=dev),
             out_sf.view(torch.uint8).reshape(-1) if out_sf is not None else torch.zeros(
                 16, dtype=torch.uint8, device=dev),
             zd if zd is not None else torch.zeros(1, 1, 16, dtype=torch.float32, device=dev),
             nblk]
    refs += fold_refs(op, dev, x_bf, a_gu, a_blk, a_s, n_a, npairs, zcnt, a2s, a_gu_s,
                      w_shared)
    refs.append(rtok if rtok is not None else torch.full((op.rows_max,), -1, dtype=torch.int32,
                                                         device=dev))
    args = [from_dlpack(t, assumed_align=16) for t in refs]
    stream = lambda: cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = compiled or cute.compile(op, *args, stream())

    def launch():
        fn(*args, stream())
        return out

    return dict(launch=launch, compiled=fn)
