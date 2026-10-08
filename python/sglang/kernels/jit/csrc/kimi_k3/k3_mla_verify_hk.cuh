// K3 MLA target-verify (QLEN = 8, 12 heads, latent 576 = 512 nope + 64 rope,
// FP8 e4m3 KV, page size 1) split-KV attention for gfx950, hand-scheduled HIP.
//
// One CTA = 4 waves (1 wave / SIMD, 512 regs each) owns all 96 query rows
// (8 q_pos x 12 heads) of one request and one contiguous KV chunk:
//   waves 0..2  compute: wave w owns query rows 32w..32w+31.
//               Q (FP8, per-row scale) stays in VGPRs as the B operand of
//               S^T = K Q^T (MFMA 32x32x64 f8f6f4), so every lane holds ONE
//               query row and the softmax row max is a lane-local max plus a
//               single permlane32 swap. P (FP8) stays in registers and is the
//               B operand of O^T += V^T P^T; O^T (32 rows x 512 fp32) lives in
//               AGPRs. Online softmax with lazy rescaling (tau = 2).
//   wave 3      loader: streams 64-token KV tiles global -> LDS with
//               buffer_load_dwordx4 ... lds (one kv index per lane, 36
//               instructions per tile), 4-stage ring, one s_barrier per tile.
// LDS tile layout: chunk-major [36 chunks of 16 B][64 tokens][16 B]; odd
// chunks store token t at position t ^ 4 (bank spread for the V transposed
// reads). K is read with ds_read_b128, V with ds_read_b64_tr_b8.
//
// Output: per-split normalized partial O (bf16) + base-2 LSE, merged by the
// v2 Triton reduce (same split formula), or written directly when NSPLIT = 1.

#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace k3hk {

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v16f __attribute__((ext_vector_type(16)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef float v2f __attribute__((ext_vector_type(2)));
#define K3_LDS __attribute__((address_space(3)))

constexpr int kH = 12;
constexpr int kQLen = 8;
constexpr int kRows = kH * kQLen;  // 96
constexpr int kD = 576;
constexpr int kDV = 512;
constexpr int kBN = 64;
constexpr int kNCh = kD / 16;  // 36 chunks of 16 B
constexpr int kTileBytes = kNCh * 1024;
constexpr int kStages = 4;
constexpr int kMboxBytes = 4 * 768;  // DMA voffset mailbox: [tile % 4][tg 0..2][lane] u32
constexpr int kThreads = 256;
constexpr float kTau = 2.0f;
constexpr float kPScale = 64.0f;  // p <= 2^tau = 4 -> 256 < 448

#ifdef K3HK_DEBUG
__device__ int kDbg;
__device__ unsigned long long kTs[4096 * 8];
__device__ int kCnt[4096 * 4];
__device__ unsigned long long kTs2[4096 * 4];
#define K3_TS2(k) do { if (threadIdx.x == 0) kTs2[(blockIdx.x) * 4 + (k)] = __builtin_amdgcn_s_memrealtime(); } while (0)
#define K3_TS(k) do { if (lane == 0) kTs[(blockIdx.x) * 8 + (k)] = __builtin_amdgcn_s_memrealtime(); } while (0)
#define K3_DBG_ARG , int dbg
#define K3_DBG_PASS , dbg
#else
#define K3_DBG_ARG
#define K3_DBG_PASS
#define K3_TS(k) do {} while (0)
#define K3_TS2(k) do {} while (0)
#endif
struct Params {
  const uint16_t* q;         // bf16 [bs*8, 12, 576] (strided rows)
  const uint8_t* kv;         // fp8 e4m3 pool [N, kv_stride]
  const int32_t* kv_indptr;  // [bs+1]
  const int32_t* kv_indices;
  const float* kv_scale;     // [1]
  uint16_t* o_part;          // bf16 [bs, nsplit, 96, 512]
  float* lse_part;           // [bs, nsplit, 96]
  uint16_t* o_final;         // bf16 [bs*8, 12, 512] (strided), used when nsplit == 1
  int64_t stride_q_tok, stride_q_h, stride_o_tok, stride_o_h;
  int32_t kv_stride;
  uint32_t kv_bytes;  // buffer num_records
  int32_t nsplit, min_chunk;
  float sm_scale_log2;
  int32_t q_lds;  // 1: q rows are contiguous ([tok][12][576] dense) -> coalesced LDS staging of Q
  // fused split merge (flags != nullptr): grid = bs * nsplit compute CTAs + nmrg merger CTAs
  int32_t* flags;    // [kFlagCap] zero-initialised, self-resetting: split-done counters, then merger exit counters
  int32_t bs;
  int32_t mrg_rows;  // rows per merger CTA (1, 2, 4, 8); nmrg = bs * 96 / mrg_rows
  int32_t nmrg;
  // auto path (Gluon-or-HK regime switch on the GPU): kv_indptr above is the regime-selected
  // indptr (all zero in the Gluon regime); in the Gluon regime the merger CTAs merge the Gluon
  // splits instead (natural-log LSE, = aiter _mla_softmax_reducev_kernel), so no reduce launch.
  const int32_t* regime;      // nullptr, or the use-HK flag written by the Gluon prologue
  const int32_t* g_indptr;    // true kv_indptr (Gluon split formula)
  const uint16_t* g_logits;   // bf16 [bs, qlen, H, g_ns, 512] (strided)
  const float* g_lse;         // fp32 [bs, qlen, H, g_ns] (strided)
  int64_t g_sl_b, g_sl_qs, g_sl_h, g_sl_s;
  int64_t g_ml_b, g_ml_qs, g_ml_h, g_ml_s;
  int32_t g_ns, g_block_n;
  // tail merge (tail != 0, flags != nullptr, bs * nsplit <= #CUs): every compute CTA merges a
  // share of its request's (row, 256-column) items after a per-request arrival barrier.
  int32_t tail;
  int32_t v_done;
  // fp8 partials: when a request has >= part8_min active splits (0 = never), its partial O is
  // stored as fp8 e4m3 of O / kv_scale (|.| <= max |V| <= 448: no per-row scale needed),
  // in the first 512 B of each 1 KB o_part row; the merge multiplies by kv_scale.
  int32_t part8_min;  // reduce1 (auto path): the HK regime was already merged in-kernel
};
constexpr int kFlagCap = 32768;  // flags[b * nsplit + s] (b * nsplit < kFlagCap), exit counters at flags[kFlagCap + b]

__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_rsrc(const void* p, uint32_t bytes) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), (short)0, (int)bytes, 0x00020000);
}

__device__ __forceinline__ void store_sc1(uint16_t* dst, v4i v) {
  asm volatile("global_store_dwordx4 %0, %1, off sc1" ::"v"(dst), "v"(v) : "memory");
}

__device__ __forceinline__ float bf16_to_f(uint32_t bits16) { return __uint_as_float(bits16 << 16); }

__device__ __forceinline__ uint32_t pack_fp8x4(float a, float b, float c, float d) {
  int r = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
  r = __builtin_amdgcn_cvt_pk_fp8_f32(c, d, r, true);
  return (uint32_t)r;
}

// max / sum of x over lanes l and l ^ 32 (result in every lane)
__device__ __forceinline__ float xhalf_max(float x) {
  auto r = __builtin_amdgcn_permlane32_swap(__float_as_uint(x), __float_as_uint(x), false, false);
  return fmaxf(__uint_as_float(r[0]), __uint_as_float(r[1]));
}
__device__ __forceinline__ float xhalf_sum(float x) {
  auto r = __builtin_amdgcn_permlane32_swap(__float_as_uint(x), __float_as_uint(x), false, false);
  return __uint_as_float(r[0]) + __uint_as_float(r[1]);
}

__device__ __forceinline__ v16f mfma(v8i a, v8i b, v16f c) {
  return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c, 0, 0, 0, 127, 0, 127);
}

typedef uint32_t v4u_ __attribute__((ext_vector_type(4)));
#include "k3_mla_verify_hk_asm.inc"

__device__ __forceinline__ int split_chunk(int L, int nsplit, int min_chunk) {
  int per = (L + nsplit - 1) / nsplit;
  per = (per + kBN - 1) / kBN * kBN;
  return per > min_chunk ? per : min_chunk;
}

// ---------------------------------------------------------------- loader wave
__device__ __forceinline__ void barrier_raw() { asm volatile("s_barrier" ::: "memory"); }

template <int N>
__device__ __forceinline__ void wait_vm() {
  asm volatile("s_waitcnt vmcnt(%0)" ::"n"(N) : "memory");
}

typedef uint32_t v4u __attribute__((ext_vector_type(4)));

__device__ __forceinline__ v4u make_rsrc_s(const void* ptr, uint32_t bytes) {
  const uint64_t a = (uint64_t)ptr;
  v4u r;
  r[0] = __builtin_amdgcn_readfirstlane((uint32_t)a);
  r[1] = __builtin_amdgcn_readfirstlane((uint32_t)(a >> 32) & 0xffffu);
  r[2] = __builtin_amdgcn_readfirstlane(bytes);
  r[3] = 0x00020000u;
  return r;
}

// LDS tile layout ("16 tokens x 4 chunks" blocks): instruction k (0..35) covers token
// group tg = k / 9 (16 tokens) and chunk group cg = k % 9 (4 x 16 B); lane i -> token
// 16 tg + i / 4, LDS slot i % 4 holding chunk 4 cg + ((i % 4) ^ (i >> 4)). So
//   LDS(t, c) = 9216 (t / 16) + 1024 (c / 4) + 64 (t % 16) + 16 ((c % 4) ^ ((t % 16) >> 2)).
// Every instruction touches 16 KV rows x 64 B; the XOR spreads banks for both the
// K row reads and the V transposed reads. Loads are inline asm so the compiler's
// waitcnt pass never inserts a vmcnt(0) (the loader manages vmcnt by hand).
template <int R>
__device__ __forceinline__ void issue_tile(v4u kv_rs, uint32_t slot, uint32_t stride, uint32_t ba, uint32_t co K3_DBG_ARG) {
#ifdef K3HK_DEBUG
  if (dbg & 1) return;
#endif
  issue_tile_asm<R>(kv_rs, slot, stride, ba, co);
}

// kv index of token `lane` of the tile -> hardcoded v[56 + R] (async asm load: the
// destination must not be a compiler-managed register).
template <int R>
__device__ __forceinline__ void load_idx(v4u idx_rs, int pos) {
  load_idx_asm<R>(idx_rs, pos * 4);
}

#ifndef K3HK_BIDX
#define K3HK_BIDX 0  // 1: compute waves start their Q loads after the loader issued the kv-index loads
#endif
#ifndef K3HK_PFA
#define K3HK_PFA 2  // L2 prefetch distance in tiles beyond the LDS-DMA tile (<= 4)
#endif
#ifdef K3HK_PF
constexpr int kPfa = K3HK_PFA;
#else
constexpr int kPfa = 0;  // L2 prefetch off (it was slower): no prefetch loads in the vmcnt bookkeeping
#endif
#ifdef K3HK_PF
constexpr int kPfLoads = 6;  // dword touches per 576 B row (every 128 B line)
#else
constexpr int kPfLoads = 0;
#endif

#define K3_RING_SWITCH(r, F, ...)                 \
  switch ((r) & 7) {                              \
    case 0: F<0>(__VA_ARGS__); break;             \
    case 1: F<1>(__VA_ARGS__); break;             \
    case 2: F<2>(__VA_ARGS__); break;             \
    case 3: F<3>(__VA_ARGS__); break;             \
    case 4: F<4>(__VA_ARGS__); break;             \
    case 5: F<5>(__VA_ARGS__); break;             \
    case 6: F<6>(__VA_ARGS__); break;             \
    default: F<7>(__VA_ARGS__); break;            \
  }

// Loader wave. Per iteration i (after barrier B_i):
//   prefetch tile i+3+P into L2 (6 dword touches per row), load idx(i+4+P),
//   LDS-DMA tile i+3 (an L2 hit by then) into the slot tile i-1 used.
// idx(t) lives in the hardcoded ring v[56 + t % 8]. vmcnt bookkeeping by hand.
__device__ __forceinline__ void loader_wave(const Params& p, uint8_t K3_LDS* smem, int kv_start, int start, int end, int ntiles,
                            int lane K3_DBG_ARG) {
  const v4u kv_rs = make_rsrc_s(p.kv, p.kv_bytes);
  const v4u idx_rs = make_rsrc_s(p.kv_indices + kv_start, (uint32_t)end * 4u);
  const uint32_t stride = (uint32_t)p.kv_stride;
  const uint32_t ba = (uint32_t)(lane >> 2) << 2;                          // bpermute source lane (t % 16) * 4
  const uint32_t co = 16u * (uint32_t)((lane & 3) ^ (lane >> 4));         // chunk-in-group byte offset
  const uint32_t base = (uint32_t)(uintptr_t)smem;
  bool skip = false;
#ifdef K3HK_DEBUG
  skip = (dbg & 1) != 0;
#endif
  auto pos_of = [&](int t) {
    int pos = start + t * kBN + lane;
    return pos < end ? pos : end - 1;  // tail tokens re-read the last valid one (masked)
  };
  auto idx = [&](int t) { K3_RING_SWITCH(t, load_idx_asm, idx_rs, pos_of(t) * 4); };
#ifndef K3HK_BLOCK_LAYOUT
  rl_precompute((uint32_t)lane);
  auto dma = [&](int t) {
    if (!skip) K3_RING_SWITCH(t, issue_tile_rl_asm, kv_rs, base + (t % kStages) * kTileBytes, stride);
  };
#else
  auto dma = [&](int t) {
    if (!skip) K3_RING_SWITCH(t, issue_tile_asm, kv_rs, base + (t % kStages) * kTileBytes, stride, ba, co);
  };
#endif
  auto pf = [&](int t) {
#ifndef K3HK_PF
    return;
#endif
    if (!skip) K3_RING_SWITCH(t, prefetch_asm, kv_rs, stride);
  };
  // prologue: idx(0 .. 3+P), DMA tiles 0..2, prefetch tiles 3 .. 2+P
  for (int t = 0; t < 4 + kPfa && t < ntiles; ++t) idx(t);
#if K3HK_BIDX
  barrier_raw();  // B_idx: the compute waves issue their Q loads behind the kv-index loads
#endif
  wait_vm<0>();
  dma(0);
  K3_TS(6);
  // Q loaded straight into VGPRs (q_lds == 0): slots 1, 2 are free, so DMA tiles 1, 2 now
  const bool early = p.q_lds == 0;
  if (early)
    for (int t = 1; t < 3 && t < ntiles; ++t) dma(t);
  barrier_raw();  // B_q: the compute waves have read their Q out of slots 1..3
  if (!early)
    for (int t = 1; t < 3 && t < ntiles; ++t) dma(t);
  for (int t = 3; t < 3 + kPfa && t < ntiles; ++t) pf(t);
  // B_pre needs tiles 0 and 1 (B_0): younger than DMA(1) = DMA(2) + prefetches
  if (ntiles > 3 + kPfa - 1) {
    wait_vm<36 + kPfLoads * kPfa>();
  } else {
    wait_vm<0>();
  }
  barrier_raw();
  for (int i = 0; i < ntiles; ++i) {
    if (i > 0) {
      // need DMA(i+1) (issued last in iteration i-2). Younger: iteration i-1's idx, pf, DMA(i+2)
      if (i + 3 + kPfa < ntiles) {
        wait_vm<kPfLoads + 1 + 36>();
      } else if (i + 2 < ntiles) {
        wait_vm<36>();
      } else {
        wait_vm<0>();
      }
    }
    barrier_raw();
    // issue order per iteration: idx(i+4+P), pf(i+3+P), DMA(i+3)
    if (i + 4 + kPfa < ntiles) idx(i + 4 + kPfa);
    if (i + 3 + kPfa < ntiles) {
      // idx(i+3+P) was loaded first in iteration i-1 (or in the prologue for i = 0);
      // younger than it: idx(i+4+P), pf(i+2+P), DMA(i+2) of iteration i-1 + idx just issued
      if (i > 0) {
        if (i + 4 + kPfa < ntiles) {
          wait_vm<1 + kPfLoads + 36>();
        } else {
          wait_vm<kPfLoads + 36>();
        }
      }
      pf(i + 3 + kPfa);
    }
    if (i + 3 < ntiles) dma(i + 3);
  }
}


// Shared-DMA loader: tiles 0..2 entirely by the loader (prologue); from tile 3 on,
// compute wave w issues token group w (9 loads) itself and the loader only token
// group 3, so per-wave vmcnt (max 63) no longer caps the bytes in flight.
// Iteration i (after B_i): load idx(i+5); mailbox(i+4) <- voffsets of token groups 0..2;
// DMA tg3 of tile i+3. Compute iteration i: read mailbox(i+3), DMA its group of tile i+3.
__device__ __forceinline__ void loader_wave_sd(const Params& p, uint8_t K3_LDS* smem, int kv_start, int start, int end,
                                               int ntiles, int lane K3_DBG_ARG) {
  const v4u kv_rs = make_rsrc_s(p.kv, p.kv_bytes);
  const v4u idx_rs = make_rsrc_s(p.kv_indices + kv_start, (uint32_t)end * 4u);
  const uint32_t stride = (uint32_t)p.kv_stride;
  const uint32_t ba = (uint32_t)(lane >> 2) << 2;
  const uint32_t co = 16u * (uint32_t)((lane & 3) ^ (lane >> 4));
  const uint32_t base = (uint32_t)(uintptr_t)smem;
  const uint32_t mb = base + kStages * kTileBytes + 4u * lane;
  auto pos_of = [&](int t) {
    int pos = start + t * kBN + lane;
    return pos < end ? pos : end - 1;
  };
  auto idx = [&](int t) { K3_RING_SWITCH(t, load_idx_asm, idx_rs, pos_of(t) * 4); };
  auto dma = [&](int t) { K3_RING_SWITCH(t, issue_tile_asm, kv_rs, base + (t % kStages) * kTileBytes, stride, ba, co); };
  auto tg3 = [&](int t) { K3_RING_SWITCH(t, issue_tg3_asm, kv_rs, base + (t % kStages) * kTileBytes, stride, ba, co); };
  auto mbox = [&](int t) { K3_RING_SWITCH(t, mbox_write_asm, mb + 768u * (t & 3), ba, stride, co); };
  for (int t = 0; t < 5 && t < ntiles; ++t) idx(t);
  wait_vm<0>();
  for (int t = 0; t < 3 && t < ntiles; ++t) dma(t);
  if (ntiles > 3) mbox(3);
  if (ntiles > 2) {
    wait_vm<36>();
  } else {
    wait_vm<0>();
  }
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");  // B_pre
  for (int i = 0; i < ntiles; ++i) {
    if (i > 0) {
      // own part of tile i+1 (DMA(2) in the prologue for i = 1, else tg3 of iteration i-2);
      // younger: iteration i-1's idx(i+4) and tg3(i+2)
      if (i + 2 < ntiles) {
        wait_vm<9>();
      } else {
        wait_vm<0>();
      }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");  // B_i
    if (i + 5 < ntiles) idx(i + 5);
    if (i + 4 < ntiles) {
      if (i > 0) wait_vm<9>();  // idx(i+4) (iteration i-1); younger: tg3(i+2) (+ idx(i+5))
      mbox(i + 4);
    }
    if (i + 3 < ntiles) tg3(i + 3);
  }
}

// ---------------------------------------------------------------- compute waves
// LDS byte offset (within a tile) this lane supplies to ds_read_b64_tr_b8 for
// V column block 0, token group T_m = 0. Tokens needed by output lane
// (col c, half h): T_m + 4h + {0,1,2,3,8,9,10,11}.
// gfx950 ds_read_b64_tr_b8 (probed): within each 16-lane group, output lane i
// byte j = input lane 2j + i/8, byte i%8. So input lane q supplies row j = q/2,
// 8-column half q%2; output lane i holds column i of the group's 16 columns.
// Returns the per-lane LDS offset for V column blocks with cb even (odd: +1 -> ^),
// T_m = 0; the asm adds 64 * (cb / 2) + 576 * T_m as immediates.
#ifndef K3HK_BLOCK_LAYOUT
// token-major 576 B rows; 16-B chunk c of token t at position c ^ ((t >> 2) & 3)
__device__ __forceinline__ uint32_t lds_off(uint32_t t, uint32_t c) {
  return 576u * t + 16u * (c ^ ((t >> 2) & 3u));
}
#else
__device__ __forceinline__ uint32_t lds_off(uint32_t t, uint32_t c) {
  return 9216u * (t >> 4) + 1024u * (c >> 2) + 64u * (t & 15u) + 16u * ((c & 3u) ^ ((t & 15u) >> 2));
}
#endif
// V: per-lane offset for column blocks with cb even (cb_odd = 0) or odd; the asm adds
// 1024 * (cb / 2) + 9216 * (T_m / 16) as immediates (T_m multiple of 16).
__device__ __forceinline__ uint32_t tr_lane_offset(int lane, int cb_odd) {
  const int g = lane >> 4, q = lane & 15;
  const int h = g >> 1;    // lane / 32
  const int par = g & 1;   // 16-col chunk within the 32-col block
  const int j = q >> 1;
  const int tok = 4 * h + (j < 4 ? j : j + 4);
  return lds_off((uint32_t)tok, (uint32_t)(2 * cb_odd + par)) + 8u * (q & 1);
}

// Epilogue staging: O^T block CB (lane = query row l32, cols 32 CB + 8 a + 4 h + 0..3)
// -> bf16 row-major [32][kOPitch] in LDS, so the global stores are full 1 KB rows.
constexpr int kOPitch = 520;  // bf16 elements (1040 B: spreads banks across rows)
template <int CB>
__device__ __forceinline__ void stage_o(uint16_t K3_LDS* buf, int l32, int h, float o_mul) {
  const v16f o = o_read<CB>();
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    const bf16x2 lo = __builtin_convertvector((v2f{o[4 * a] * o_mul, o[4 * a + 1] * o_mul}), bf16x2);
    const bf16x2 hi = __builtin_convertvector((v2f{o[4 * a + 2] * o_mul, o[4 * a + 3] * o_mul}), bf16x2);
    v2i pk = {__builtin_bit_cast(int, lo), __builtin_bit_cast(int, hi)};
    *(v2i K3_LDS*)(buf + l32 * kOPitch + 32 * CB + 8 * a + 4 * h) = pk;
  }
  if constexpr (CB + 1 < 16) stage_o<CB + 1>(buf, l32, h, o_mul);
}

constexpr int kOPitch8 = 544;  // bytes
template <int CB>
__device__ __forceinline__ void stage_o8(uint8_t K3_LDS* buf, int l32, int h, float o_mul) {
  const v16f o = o_read<CB>();
#pragma unroll
  for (int a = 0; a < 4; ++a)
    *(uint32_t K3_LDS*)(buf + l32 * kOPitch8 + 32 * CB + 8 * a + 4 * h) =
        pack_fp8x4(o[4 * a] * o_mul, o[4 * a + 1] * o_mul, o[4 * a + 2] * o_mul, o[4 * a + 3] * o_mul);
  if constexpr (CB + 1 < 16) stage_o8<CB + 1>(buf, l32, h, o_mul);
}

template <int KS>
__device__ __forceinline__ void load_q(const uint16_t* qrow, float q_inv) {
  v4i r[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) r[j] = *(const v4i*)(qrow + 64 * KS + 8 * j);
  v8i qf;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const uint32_t u0 = (uint32_t)r[k >> 1][(k & 1) * 2];
    const uint32_t u1 = (uint32_t)r[k >> 1][(k & 1) * 2 + 1];
    qf[k] = (int)pack_fp8x4(bf16_to_f(u0 & 0xffffu) * q_inv, bf16_to_f(u0 >> 16) * q_inv,
                            bf16_to_f(u1 & 0xffffu) * q_inv, bf16_to_f(u1 >> 16) * q_inv);
  }
  q_set<KS>(qf);
  if constexpr (KS + 1 < 9) load_q<KS + 1>(qrow, q_inv);
}

struct WaveState {
  float m_i;   // running max (log2 units, scaled) used for p
  float nb;    // 6 - m_i (log2 kPScale folded in)
  float al;    // rescale factor of the pending decision (applied to l in the next exp)
  float l_i;   // running sum (x kPScale)
};

// One pipelined iteration on tile i (scores in S buffer D, decision for it already made):
//   QK(i+1) -> S[1-D]  ||  p = exp(S[D]) -> P, l update
//   [mask S[1-D] if tile i+1 needs it]
//   PV(i)  ||  max / rescale decision of tile i+1;  O *= alpha if any row wants it.
struct DmaCtx {
  v8i ones;  // fp8 1.0 x 32 (A operand of the P row-sum MFMA)
  v4u rs;
  uint32_t base, mb;
  int w, i, ntiles;
};

template <int D>
__device__ __forceinline__ void iteration(WaveState& st, uint32_t slot, uint32_t nslot, uint32_t koff_e,
                                          uint32_t koff_o, uint32_t voff, uint32_t voff_o, float s_scale,
                                          bool has_next, int next_thr, bool next_mask, const DmaCtx& dc K3_DBG_ARG) {
  // own DMA part of tile i+1 (issued in iteration i-2) must have landed; younger: part of tile i+2
#ifdef K3HK_SD
  if (dc.i + 2 < dc.ntiles && dc.i >= 1) {
    asm volatile("s_waitcnt vmcnt(9) lgkmcnt(0)\n\ts_barrier" ::: "memory");  // B_i
  } else {
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");  // B_i
  }
#else
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");  // B_i
#endif
#ifdef K3HK_SD
  if (dc.i + 3 < dc.ntiles) {
    const int t = dc.i + 3;
    const uint32_t sl = dc.base + (uint32_t)((t % kStages) * kTileBytes);
    const uint32_t m = dc.mb + 768u * (uint32_t)(t & 3);
    switch (dc.w) {
      case 0: cdma_asm<0>(m, dc.rs, sl); break;
      case 1: cdma_asm<1>(m, dc.rs, sl); break;
      default: cdma_asm<2>(m, dc.rs, sl); break;
    }
  }
#endif
#ifdef K3HK_DEBUG
  if (dbg & 2) return;
#endif
  qk_exp<D>(nslot + koff_e, nslot + koff_o, s_scale, st.nb, st.al, st.l_i);
  if (has_next) {
    if (next_mask) mask_s<1 - D>(next_thr);
    uint64_t rs;
    pv_decide<1 - D>(slot + voff, slot + voff_o, s_scale, st.m_i, st.nb, st.al, rs, dc.ones, st.l_i);
    if (rs != 0) {
      o_scale(st.al);
#ifdef K3HK_DEBUG
      if ((threadIdx.x & 63) == 0) kCnt[blockIdx.x * 4 + dc.w] += 1;
#endif
    }
  } else {
    pv_tile_asm<D>(slot + voff, slot + voff_o, dc.ones, st.al, st.l_i);
  }
}

__device__ __forceinline__ void compute_wave(const Params& p, uint8_t K3_LDS* smem, int b, int split, int L, int start, int end,
                             int ntiles, int w, int lane K3_DBG_ARG) {
  const int h = lane >> 5, l32 = lane & 31;
  const int row = 32 * w + l32;
  const int qpos = row / kH, head = row - qpos * kH;

  // ---- Q: this lane owns d = 64 ks + 32 h + [0, 32) of its row, ks = 0..8.
  // Pass 1: per-row amax (over all 576 dims); pass 2 (L2 hits): quantize into v[48:119].
  const uint16_t* qrow = p.q + (int64_t)(b * kQLen + qpos) * p.stride_q_tok + (int64_t)head * p.stride_q_h + 32 * h;
  // q8 = fp8(q * 448 / amax_row); qscale = amax_row / 448
  float qscale;
  const float kv_scale = *p.kv_scale;
#if K3HK_BIDX
  asm volatile("s_barrier" ::: "memory");  // B_idx
#endif
  if (p.q_lds) {
    // this wave's 32 rows are one contiguous 36 KB block: DMA it into LDS slot 1 + w (tiles 1, 2
    // are issued by the loader only after barrier B_q), read back in the MFMA layout; the slot
    // is released (B_q) before the quantization VALU so the loader's DMA(1, 2) overlap it
    const uint16_t* qblk = p.q + ((int64_t)b * kRows + 32 * w) * kD;
    const uint32_t sl = (uint32_t)(uintptr_t)smem + (uint32_t)((1 + w) * kTileBytes);
    q_load_lds_asm(make_rsrc_s(qblk, 32 * kD * 2), sl, 16u * (uint32_t)lane, sl + 1152u * (uint32_t)l32 + 64u * (uint32_t)h,
                   1024u * (uint32_t)__builtin_amdgcn_readfirstlane((blockIdx.x >> 3) % 36));
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");  // B_q: Q staging slots free
#ifdef K3HK_TSQ
    if (w == 0) K3_TS(5);
#endif
    qscale = q_quant_asm();
  } else {
    qscale = q_prologue_asm(qrow);
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");  // B_q
  }
#if !defined(K3HK_TSQ) && !defined(K3HK_TSE)
  if (w == 0) K3_TS(5);
#endif
  const float s_scale = qscale * (kv_scale * p.sm_scale_log2);

  const uint32_t base = (uint32_t)(uintptr_t)smem;
  // K fragment (ks, tb) of lane (l32, h): token t = 32 tb + l32, chunks 4 ks + 2h (+1)
  // (immediates in the asm: 1024 ks + 18432 tb)
  const uint32_t koff_e = lds_off((uint32_t)l32, 2u * h);
  const uint32_t koff_o = lds_off((uint32_t)l32, 2u * h + 1u);
  const uint32_t voff = tr_lane_offset(lane, 0), voff_o = tr_lane_offset(lane, 1);
  const int lim = min(L - kQLen + qpos, end - 1);  // last visible token of this row (in this split)
  const int mask_from = min(L - kQLen, end - 1);   // tiles ending after this need a mask

  if (!(K3HK_QZO && p.q_lds)) o_zero();  // q_lds: zeroed during the Q loads (q_load_lds_asm)
  WaveState st{-1e30f, 0.f, 0.f, 0.f};
  auto slot_of = [&](int t) { return base + (uint32_t)((t % kStages) * kTileBytes); };
  auto thr_of = [&](int t) { return lim - (start + t * kBN) - 4 * h; };
  auto mask_of = [&](int t) { return start + t * kBN + kBN - 1 > mask_from; };

  if (w == 0) K3_TS(1);
  DmaCtx dc{v8i{0x38383838, 0x38383838, 0x38383838, 0x38383838, 0x38383838, 0x38383838, 0x38383838, 0x38383838},
            make_rsrc_s(p.kv, p.kv_bytes), base, base + kStages * kTileBytes + 256u * (uint32_t)w + 4u * (uint32_t)lane,
            w, 0, ntiles};
  asm volatile("s_waitcnt vmcnt(0)\n\ts_barrier" ::: "memory");  // B_pre: tiles 0, 1 landed
  if (w == 0) K3_TS(2);
  qk_first(base + koff_e, base + koff_o);
  if (mask_of(0)) mask_s<0>(thr_of(0));
  {
    uint64_t rs;
    decide<0>(s_scale, st.m_i, st.nb, st.al, rs);
  }
  int i = 0;
  for (; i + 2 <= ntiles; i += 2) {
    dc.i = i;
    iteration<0>(st, slot_of(i), slot_of(i + 1), koff_e, koff_o, voff, voff_o, s_scale, true, thr_of(i + 1),
                 mask_of(i + 1), dc K3_DBG_PASS);
    const bool nx = i + 2 < ntiles;
    dc.i = i + 1;
    iteration<1>(st, slot_of(i + 1), slot_of(nx ? i + 2 : i + 1), koff_e, koff_o, voff, voff_o, s_scale, nx,
                 thr_of(i + 2), nx && mask_of(i + 2), dc K3_DBG_PASS);
  }
  if (i < ntiles) {
    dc.i = i;
    iteration<0>(st, slot_of(i), slot_of(i), koff_e, koff_o, voff, voff_o, s_scale, false, 0, false, dc K3_DBG_PASS);
  }

  // ---- epilogue
  if (w == 0) K3_TS(3);
#ifdef K3HK_DEBUG
  if (dbg & 4) return;
#endif
  // l: sum of the fp8-rounded P (x kPScale) from the row-sum MFMA -> already the full row
  // in both lane halves (gen_asm LSUM8); otherwise per-half partial sums of the unrounded p.
#ifndef K3HK_NO_LSUM8
  const float l_tot = st.l_i;
#else
  const float l_tot = xhalf_sum(st.l_i);
#endif
  const bool has = l_tot > 0.f;
  const float o_mul = has ? kv_scale / l_tot : 0.f;
  bool part8 = false;
  if (p.part8_min > 0 && p.nsplit != 1 && p.flags == nullptr) {
    const int chunk = split_chunk(L, p.nsplit, p.min_chunk);
    part8 = min((L + chunk - 1) / chunk, p.nsplit) >= p.part8_min;
  }
  if (p.nsplit != 1 && h == 0) {
    const int64_t prow = ((int64_t)b * p.nsplit + split) * kRows + row;
    const float lse = has ? st.m_i + __builtin_amdgcn_logf(l_tot) - 6.0f : -INFINITY;
    if (p.flags != nullptr) {
      __hip_atomic_store(p.lse_part + prow, lse, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);  // sc1
    } else {
      p.lse_part[prow] = lse;
    }
  }
  // After the last barrier only the last tile's slot is still read (by other waves):
  // wave w stages its 32 x 512 bf16 O block in slot (ntiles + w) % 4.
  uint16_t K3_LDS* buf = (uint16_t K3_LDS*)(smem + ((ntiles + w) % kStages) * kTileBytes);
  if (part8) {
    uint8_t K3_LDS* b8 = (uint8_t K3_LDS*)buf;
    stage_o8<0>(b8, l32, h, has ? 1.f / l_tot : 0.f);
    __builtin_amdgcn_s_waitcnt(0xc07f);  // lgkmcnt(0)
    // 1 KB row stride (= the bf16 layout, first half used): no overlap with bf16 requests
    uint8_t* o8 = (uint8_t*)p.o_part + (((int64_t)b * p.nsplit + split) * kRows + 32 * w) * (2 * kDV);
#pragma unroll 4
    for (int r = 0; r < 32; r += 2) {
      const int rr = r + (lane >> 5);
      const v4i v = *(const v4i K3_LDS*)(b8 + rr * kOPitch8 + 16 * (lane & 31));
      *(v4i*)(o8 + rr * (2 * kDV) + 16 * (lane & 31)) = v;
    }
    if (w == 0) {
      __builtin_amdgcn_s_waitcnt(0);
      K3_TS(4);
    }
    return;
  }
  stage_o<0>(buf, l32, h, o_mul);
  __builtin_amdgcn_s_waitcnt(0xc07f);  // lgkmcnt(0): own LDS writes done before the reads below
#ifdef K3HK_TSE
  if (w == 0) K3_TS(5);
#endif
#pragma unroll 4
  for (int r = 0; r < 32; ++r) {
    const int rr = 32 * w + r;
    uint16_t* dst;
    if (p.nsplit == 1) {
      const int qp = rr / kH;
      dst = p.o_final + (int64_t)(b * kQLen + qp) * p.stride_o_tok + (int64_t)(rr - qp * kH) * p.stride_o_h;
    } else {
      dst = p.o_part + (((int64_t)b * p.nsplit + split) * kRows + rr) * kDV;
    }
    const v4i v = *(const v4i K3_LDS*)(buf + r * kOPitch + 8 * lane);
    if (p.flags != nullptr) {
      store_sc1(dst + 8 * lane, v);
    } else {
      *(v4i*)(dst + 8 * lane) = v;
    }
  }
  if (p.flags != nullptr) {
    // publish: this wave's 32 partial rows (+ lse) are at the coherence point -> flag += 1 (3 = split done)
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if (lane == 0 && !p.tail)
      __hip_atomic_fetch_add(p.flags + b * p.nsplit + split, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  if (w == 0) {
    __builtin_amdgcn_s_waitcnt(0);
    K3_TS(4);
  }
}

// ---------------------------------------------------------------- fused split merge
// Merger CTAs come after all compute CTAs in the grid. Merger m owns U = mrg_rows consecutive
// rows of request b = m / (96 / U). Its waves poll the per-split done counters (3 = all three
// compute waves published), pull newly finished splits' partial rows (bf16, written with sc1 =
// agent-coherent write-through) and fold them into an online base-2 LSE merge, so the merge
// overlaps the CTA tail. Deadlock-free: mergers only wait on compute CTAs, which never wait,
// and nmrg (<= 128) < #CUs, so compute CTAs always find a CU even if dispatch were unordered.
// U <= 4: WS = 4 / U waves per row, wave w takes splits s = sub (mod WS); U = 8: 2 rows / wave.
#ifndef K3HK_MRG_SLEEP
#define K3HK_MRG_SLEEP 16  // max back-off: x 2 x 64 clk between empty polls
#endif
constexpr int kMrgAux = 16;  // cache policy sc1 (agent-coherent load)

__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 64));
  return v;
}
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
  return v;
}

// Splits are polled / batched per group of 64 (lane l <-> split 64 j + l). A batch = the newly
// finished splits of one group (<= KB): per-lane lse loads (one instruction per row), one 1 KB
// LDS-DMA per (split, row), then batch max -> rescale -> weighted sum (weights via readlane).
template <int U>
__device__ __forceinline__ void merger_body(const Params& p, uint8_t K3_LDS* smem, int m) {
  constexpr int WS = U <= 4 ? 4 / U : 1;
  constexpr int R = U <= 4 ? 1 : U / 4;
  constexpr int KB = R == 1 ? 32 : 16;  // splits per batch (LDS: KB * R KB per wave)
  const int w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
  const int lane = threadIdx.x & 63;
  constexpr int cpr = kRows / U;
  const int b = m / cpr;
  const int row0 = (m - b * cpr) * U + (U <= 4 ? w / WS : R * w);
  const int sub = U <= 4 ? w % WS : 0;
  const int kv0 = p.kv_indptr[b];
  const int L = p.kv_indptr[b + 1] - kv0;
  const int chunk = split_chunk(L, p.nsplit, p.min_chunk);
  const int n = L > 0 ? min((L + chunk - 1) / chunk, p.nsplit) : 0;
  if (n == 0) return;  // nothing to merge (and nothing was published): no exit count either
  if (w == 0) K3_TS(0);
  int npoll = 0;
  int backoff = 1;
  const int32_t* flg = p.flags + b * p.nsplit;
  const int64_t pbase = (int64_t)b * p.nsplit;
  __amdgpu_buffer_rsrc_t ors = make_rsrc(p.o_part + pbase * kRows * kDV, (uint32_t)(n * kRows * kDV * 2));
  const float* lsep = p.lse_part + pbase * kRows + row0;
  uint8_t K3_LDS* wbuf = smem + w * (KB * R * 1024);
  float acc[R][8];
  float mx[R], ws[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    mx[r] = -INFINITY;
    ws[r] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[r][e] = 0.f;
  }
  uint64_t todo[4];
  int left = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int s = 64 * j + lane;
    todo[j] = __ballot(s < n && (s % WS) == sub);
    left += __builtin_popcountll(todo[j]);
  }
  while (left > 0) {
    // poll the done counters of all groups with work left
    uint64_t take[4];
    int nb = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      bool r = false;
      if ((todo[j] >> lane) & 1) r = __hip_atomic_load(flg + 64 * j + lane, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >= 3;
      uint64_t t = __ballot(r);
      // cap the batch at KB splits (lowest first)
      const int room = KB - nb;
      if (__builtin_popcountll(t) > room) {
        uint64_t t2 = t;
        for (int k = 0; k < room; ++k) t2 &= t2 - 1;
        t &= ~t2;
      }
      take[j] = t;
      nb += __builtin_popcountll(t);
    }
    ++npoll;
    if (nb == 0) {
      // exponential back-off (64 clk .. ~0.85 us): the done counters of a request share a few
      // lines and early mergers would otherwise hammer them while the compute CTAs stream KV
      for (int z = 0; z < backoff; ++z) __builtin_amdgcn_s_sleep(2);
      backoff = backoff < K3HK_MRG_SLEEP ? 2 * backoff : backoff;
      continue;
    }
    backoff = 1;
    left -= nb;
    float lv[4][R];
    {
      int k = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        todo[j] &= ~take[j];
        const bool mine = (take[j] >> lane) & 1;
#pragma unroll
        for (int rr = 0; rr < R; ++rr)
          lv[j][rr] = mine ? __hip_atomic_load(lsep + (int64_t)(64 * j + lane) * kRows + rr, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT)
                           : -INFINITY;
        uint64_t mk = take[j];
        for (; mk != 0; ++k) {
          const int s = 64 * j + __builtin_ctzll(mk);
          mk &= mk - 1;
#pragma unroll
          for (int rr = 0; rr < R; ++rr)
            __builtin_amdgcn_raw_ptr_buffer_load_lds(ors, (void K3_LDS*)(wbuf + (k * R + rr) * 1024), 16,
                                                     (uint32_t)(((s * kRows + row0 + rr) * kDV + 8 * lane) * 2), 0, 0,
                                                     kMrgAux);
        }
      }
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll
    for (int rr = 0; rr < R; ++rr) {
      float bm = lv[0][rr];
#pragma unroll
      for (int j = 1; j < 4; ++j) bm = fmaxf(bm, lv[j][rr]);
      const float mn = fmaxf(mx[rr], wave_max(bm));
      if (mn == -INFINITY) continue;  // all rows empty so far
      const float al = __builtin_amdgcn_exp2f(mx[rr] - mn);
      mx[rr] = mn;
      ws[rr] *= al;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[rr][e] *= al;
      int k = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float wl = ((take[j] >> lane) & 1) ? __builtin_amdgcn_exp2f(lv[j][rr] - mn) : 0.f;
        if (take[j] != 0) ws[rr] += wave_sum(wl);
        uint64_t mk = take[j];
        for (; mk != 0; ++k) {
          const int bit = __builtin_ctzll(mk);
          mk &= mk - 1;
          const float wk = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, wl), bit));
          const v4i x = *(const v4i K3_LDS*)(wbuf + (k * R + rr) * 1024 + 16 * lane);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const uint32_t u = (uint32_t)x[e];
            acc[rr][2 * e] += wk * bf16_to_f(u & 0xffffu);
            acc[rr][2 * e + 1] += wk * bf16_to_f(u >> 16);
          }
        }
      }
    }
  }
  if (w == 0) K3_TS(3);
#ifdef K3HK_DEBUG
  if (w == 0 && lane == 0) kTs[blockIdx.x * 8 + 5] = npoll;
#endif
  // combine the WS waves of a row through LDS (batch buffers are free now)
  if constexpr (WS > 1) {
    __syncthreads();
    float K3_LDS* cb = (float K3_LDS*)smem + w * (64 * 8 + 2);
#pragma unroll
    for (int e = 0; e < 8; ++e) cb[e * 64 + lane] = acc[0][e];
    if (lane == 0) {
      cb[512] = mx[0];
      cb[513] = ws[0];
    }
    __syncthreads();
    if (sub == 0) {
      float M = mx[0];
      for (int t = 1; t < WS; ++t) M = fmaxf(M, ((float K3_LDS*)smem)[(w + t) * (64 * 8 + 2) + 512]);
      const float Mu = M == -INFINITY ? 0.f : M;
      float tw = 0.f;
      float o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      for (int t = 0; t < WS; ++t) {
        const float K3_LDS* c = (const float K3_LDS*)smem + (w + t) * (64 * 8 + 2);
        const float f = __builtin_amdgcn_exp2f(c[512] - Mu);
        tw += f * c[513];
#pragma unroll
        for (int e = 0; e < 8; ++e) o[e] += f * c[e * 64 + lane];
      }
      ws[0] = tw;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[0][e] = o[e];
    }
  }
  if (sub == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int row = row0 + r;
      const float inv = ws[r] > 0.f ? 1.f / ws[r] : 0.f;
      v4i o;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const bf16x2 pk = __builtin_convertvector((v2f{acc[r][2 * e] * inv, acc[r][2 * e + 1] * inv}), bf16x2);
        o[e] = __builtin_bit_cast(int, pk);
      }
      const int qp = row / kH;
      *(v4i*)(p.o_final + (int64_t)(b * kQLen + qp) * p.stride_o_tok + (int64_t)(row - qp * kH) * p.stride_o_h + 8 * lane) = o;
    }
  }
  // last merger CTA of request b resets its done counters (all are 3: every merger saw every split)
  __syncthreads();
  if (w == 0) K3_TS(4);
  if (w == 0) {
    int old = 0;
    if (lane == 0) old = __hip_atomic_fetch_add(p.flags + kFlagCap + b, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    old = __builtin_amdgcn_readfirstlane(old);
    if (old == cpr - 1) {
      for (int s = lane; s < p.nsplit; s += 64)
        __hip_atomic_store(p.flags + b * p.nsplit + s, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (lane == 0) __hip_atomic_store(p.flags + kFlagCap + b, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

// Gluon-regime merge of merger m's rows (Gluon stage 1 has completed: plain loads, no waiting).
// Wave w handles rows row0 + w, w + 4, ... of the CTA's U rows; lane owns columns 8 lane .. + 7.
__device__ __forceinline__ void gluon_merge(const Params& p, int m) {
  const int w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
  const int lane = threadIdx.x & 63;
  const int U = p.mrg_rows;
  const int cpr = kRows / U;
  const int b = m / cpr;
  const int gL = p.g_indptr[b + 1] - p.g_indptr[b];
  const int gper = max(p.g_block_n, gL / p.g_ns);
  const int gnact = gL > 0 ? min((gL + gper - 1) / gper, p.g_ns) : 0;
  for (int row = (m - b * cpr) * U + w; row < (m - b * cpr + 1) * U; row += 4) {
    const int qp = row / kH, h = row - qp * kH;
    const uint16_t* lg = p.g_logits + b * p.g_sl_b + qp * p.g_sl_qs + h * p.g_sl_h + 8 * lane;
    const float* ls = p.g_lse + b * p.g_ml_b + qp * p.g_ml_qs + h * p.g_ml_h;
    float e_max = -INFINITY, e_sum = 0.f;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int s0 = 0; s0 < gnact; s0 += 4) {
      v4i x[4];
      float l[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int s = min(s0 + k, gnact - 1);
        l[k] = s0 + k < gnact ? ls[s * p.g_ml_s] : -INFINITY;
        x[k] = *(const v4i*)(lg + s * p.g_sl_s);
      }
      float tmax = l[0];
#pragma unroll
      for (int k = 1; k < 4; ++k) tmax = fmaxf(tmax, l[k]);
      const float nmax = fmaxf(e_max, tmax);
      if (nmax == -INFINITY) continue;
      const float osc = e_max == -INFINITY ? 0.f : __expf(e_max - nmax);
      e_sum *= osc;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] *= osc;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float wk = l[k] == -INFINITY ? 0.f : __expf(l[k] - nmax);
        e_sum += wk;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const uint32_t u = (uint32_t)x[k][e];
          acc[2 * e] += wk * bf16_to_f(u & 0xffffu);
          acc[2 * e + 1] += wk * bf16_to_f(u >> 16);
        }
      }
      e_max = nmax;
    }
    const float inv = e_sum > 0.f ? 1.f / e_sum : 1.f;
    v4i o;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const bf16x2 pk = __builtin_convertvector((v2f{acc[2 * e] * inv, acc[2 * e + 1] * inv}), bf16x2);
      o[e] = __builtin_bit_cast(int, pk);
    }
    *(v4i*)(p.o_final + (int64_t)(b * kQLen + qp) * p.stride_o_tok + (int64_t)h * p.stride_o_h + 8 * lane) = o;
  }
}

__device__ __forceinline__ void merger_cta(const Params& p, uint8_t K3_LDS* smem, int m) {
#ifdef K3HK_NO_MERGER
  return;
#endif
  if (m >= p.nmrg) return;
  asm volatile(";K3_MERGER_BEGIN" ::: "memory");
  if (p.regime != nullptr && *p.regime == 0) {
    if (p.g_ns > 1) gluon_merge(p, m);
    asm volatile(";K3_MERGER_END" ::: "memory");
    return;
  }
  switch (p.mrg_rows) {
    case 1: merger_body<1>(p, smem, m); break;
    case 2: merger_body<2>(p, smem, m); break;
    case 4: merger_body<4>(p, smem, m); break;
    default: merger_body<8>(p, smem, m); break;
  }
  asm volatile(";K3_MERGER_END" ::: "memory");
}


// ---------------------------------------------------------------- tail merge
// EXPERIMENTAL, compiled only with -DK3HK_TAIL_MERGE (measured slower than the separate reduce:
// the in-kernel merge read is bandwidth-bound and pays a ~2 us arrival barrier on top).
#ifdef K3HK_TAIL_MERGE
// Per request b, flags + kTmStride * b: done[16] / exit[16] sub-counters (one 128 B line each,
// split s counts into s % 16 -- a single counter would serialise ~200 same-address atomics),
// exit_top, abandoned bitmask[8]. All self-resetting (the last CTA to exit zeroes them).
// Item i (0..191) = (row i / 2, columns 256 (i % 2) .. + 255); CTA of split s owns items s, s + nact, ...
// After its own partial is published a CTA waits (bounded) until all nact splits are done, then
// merges its items with one LDS-DMA round trip. Deadlock-free without co-residency guarantees:
// a CTA that times out marks its items abandoned and exits; the last CTA to exit merges them.
constexpr int kTmStride = 2048;
constexpr int kTmSub = 16;
constexpr int kTmDone = 0, kTmExit = 512, kTmTop = 1024, kTmAband = 1056;
#ifndef K3HK_TM_TIMEOUT
#define K3HK_TM_TIMEOUT 3000  // x 10 ns (s_memrealtime = 100 MHz)
#endif
#ifndef K3HK_TM_AUX
#define K3HK_TM_AUX 0  // plain loads: the acquire fence (buffer_inv sc1) already dropped stale L2 lines
#endif
constexpr int kLdsTotal = kStages * kTileBytes + kMboxBytes;

__device__ __forceinline__ int tm_ld(const int32_t* a) { return __hip_atomic_load(a, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
__device__ __forceinline__ void tm_st(int32_t* a, int v) { __hip_atomic_store(a, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
__device__ __forceinline__ int tm_add(int32_t* a, int v) { return __hip_atomic_fetch_add(a, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }

// merge items it0, it0 + istep, ... (count n) of request b; all 256 threads
__device__ __forceinline__ void tm_merge_items(const Params& p, uint8_t K3_LDS* smem, int b, int nact, int it0, int istep, int n) {
  const int t = threadIdx.x, lane = t & 63, w = t >> 6;
  const int g = t >> 5, c = t & 31;
  const int ns = p.nsplit;
  // LDS: [KB items x nact x 512 B partials][KB x 256 lse][256 wts][4 red][4 x 32 x 9 acc]
  // padded to a multiple of 8 splits: the DMA slots of clamped (s >= nact) lanes stay inside the item
  // region (other waves' DMAs may still land after this wave's vmcnt wait)
  const int pbytes = ((nact + 7) & ~7) * 512;
  constexpr int kFixed = 256 * 4 + 16 + 4 * 32 * 9 * 4;
  int KB = (kLdsTotal - kFixed) / (pbytes + 1024);
  KB = KB < 8 ? KB : 8;
  float K3_LDS* lse_l = (float K3_LDS*)(smem + KB * pbytes);
  float K3_LDS* wts = lse_l + KB * 256;
  float K3_LDS* red = wts + 256;
  float K3_LDS* sacc = red + 4;
  __amdgpu_buffer_rsrc_t ors = make_rsrc(p.o_part + (int64_t)b * ns * kRows * kDV, (uint32_t)(nact * kRows * kDV * 2));
  const float* lsep = p.lse_part + (int64_t)b * ns * kRows;
  const int J = (nact + 7) >> 3;
  for (int k0 = 0; k0 < n; k0 += KB) {
    const int kb = min(KB, n - k0);
    float lv[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      if (k < kb) {
        const int it = it0 + (k0 + k) * istep, row = it >> 1, half = it & 1;
        for (int j = 0; j < J; ++j) {
          const int sp = min(2 * w + (lane >> 5) + 8 * j, nact - 1);
          __builtin_amdgcn_raw_ptr_buffer_load_lds(ors, (void K3_LDS*)(smem + k * pbytes + (2 * w + 8 * j) * 512), 16,
                                                   (uint32_t)((sp * kRows + row) * 1024 + half * 512 + 16 * c), 0, 0,
                                                   K3HK_TM_AUX);
        }
        lv[k] = t < nact ? __hip_atomic_load(lsep + t * kRows + row, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) : -INFINITY;
      }
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll
    for (int k = 0; k < 8; ++k)
      if (k < kb) lse_l[k * 256 + t] = lv[k];
    __syncthreads();
#ifdef K3HK_TM_TSDMA
    K3_TS2(3);
#endif
    for (int k = 0; k < kb; ++k) {
      const int it = it0 + (k0 + k) * istep, row = it >> 1, half = it & 1;
      const float l = lse_l[k * 256 + t];
      float m = wave_max(l);
      if (lane == 0) red[w] = m;
      __syncthreads();
      m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
      if (m == -INFINITY) m = 0.f;
      wts[t] = l == -INFINITY ? 0.f : __builtin_amdgcn_exp2f(l - m);
      __syncthreads();
      float acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
      const uint8_t K3_LDS* pb = smem + k * pbytes + 16 * c;
      for (int sp = g; sp < nact; sp += 8) {
        const float wk = wts[sp];
        const v4i x = *(const v4i K3_LDS*)(pb + sp * 512);
        acc[8] += wk;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const uint32_t u = (uint32_t)x[e];
          acc[2 * e] += wk * bf16_to_f(u & 0xffffu);
          acc[2 * e + 1] += wk * bf16_to_f(u >> 16);
        }
      }
#pragma unroll
      for (int e = 0; e < 9; ++e) acc[e] += __shfl_xor(acc[e], 32, 64);
      if (lane < 32) {
#pragma unroll
        for (int e = 0; e < 9; ++e) sacc[(w * 32 + c) * 9 + e] = acc[e];
      }
      __syncthreads();
#ifdef K3HK_TM_DBG
      if (lane == 0 && p.g_lse != nullptr) {
        float* dbg = (float*)p.g_lse + (int64_t)(b * 192 + it) * 16;
        dbg[w] = m;
        dbg[4 + w] = sacc[(w * 32) * 9 + 8];
        dbg[8 + w] = (float)nact;
      }
#endif
      if (t < 32) {
        float o[9];
#pragma unroll
        for (int e = 0; e < 9; ++e)
          o[e] = sacc[c * 9 + e] + sacc[(32 + c) * 9 + e] + sacc[(64 + c) * 9 + e] + sacc[(96 + c) * 9 + e];
        const float inv = o[8] > 0.f ? 1.f / o[8] : 0.f;
        v4i r;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const bf16x2 pk = __builtin_convertvector((v2f{o[2 * e] * inv, o[2 * e + 1] * inv}), bf16x2);
          r[e] = __builtin_bit_cast(int, pk);
        }
        const int qp = row / kH;
        *(v4i*)(p.o_final + (int64_t)(b * kQLen + qp) * p.stride_o_tok + (int64_t)(row - qp * kH) * p.stride_o_h +
                256 * half + 8 * c) = r;
      }
      __syncthreads();
    }
  }
}

__device__ __forceinline__ void tail_merge(const Params& p, uint8_t K3_LDS* smem, int b, int split, int nact) {
  const int t = threadIdx.x, lane = t & 63, w = t >> 6;
  int32_t* f = p.flags + kTmStride * b;
  __shared__ int tm_go;
  // no asm-owned register state is live any more (checker: exempt like the merger blocks)
  asm volatile(";K3_MERGER_BEGIN" ::: "memory");
  // all 4 waves here: the compute waves' partial stores are complete (vmcnt(0) in the epilogue)
  __syncthreads();
  K3_TS2(0);
  if (t == 0) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    tm_add(f + kTmDone + 32 * (split % kTmSub), 1);
  }
  K3_TS2(1);
  const int nitems = split < 2 * kRows ? (2 * kRows - 1 - split) / nact + 1 : 0;
  const int nsub = min(kTmSub, nact);
  if (nitems > 0) {
    if (w == 0) {
      // wait (bounded) until every split of request b has published
      const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
      bool ok = false;
      while (true) {
        int v = lane < nsub ? tm_ld(f + kTmDone + 32 * lane) : 0;
#pragma unroll
        for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
        if (v >= nact) { ok = true; break; }
        if (__builtin_amdgcn_s_memrealtime() - t0 > (uint64_t)K3HK_TM_TIMEOUT) break;
        __builtin_amdgcn_s_sleep(1);
      }
      if (lane == 0) tm_go = ok ? 1 : 0;
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#ifdef K3HK_TM_DELAY
    for (int z = 0; z < K3HK_TM_DELAY; ++z) __builtin_amdgcn_s_sleep(127);
#endif
    K3_TS2(2);
    if (tm_go) {
      tm_merge_items(p, smem, b, nact, split, nact, nitems);
#ifndef K3HK_TM_TSDMA
      K3_TS2(3);
#endif
    } else if (t == 0) {
      __hip_atomic_fetch_or(f + kTmAband + (split >> 5), 1 << (split & 31), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
  // exit: the last CTA of request b merges abandoned items and resets the counters
  if (t == 0) {
    int last = 0;
    const int k = split % kTmSub;
    const int nk = (nact - k + kTmSub - 1) / kTmSub;
    if (tm_add(f + kTmExit + 32 * k, 1) == nk - 1) last = tm_add(f + kTmTop, 1) == nsub - 1;
    tm_go = last;
  }
  __syncthreads();
  if (tm_go) {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    for (int wd = 0; wd < 8; ++wd) {
      int m = tm_ld(f + kTmAband + wd);
      while (m != 0) {
        const int s2 = 32 * wd + __builtin_ctz(m);
        m &= m - 1;
        const int n2 = s2 < 2 * kRows ? (2 * kRows - 1 - s2) / nact + 1 : 0;
        if (n2 > 0) tm_merge_items(p, smem, b, nact, s2, nact, n2);
      }
    }
    __syncthreads();
    if (t < kTmSub) {
      tm_st(f + kTmDone + 32 * t, 0);
      tm_st(f + kTmExit + 32 * t, 0);
    }
    if (t < 8) tm_st(f + kTmAband + t, 0);
    if (t == 0) tm_st(f + kTmTop, 0);
  }
}
#endif  // K3HK_TAIL_MERGE

__global__ __launch_bounds__(kThreads, 1) __attribute__((amdgpu_num_vgpr(48))) void k3_mla_verify_hk_kernel(const Params p) {
  __shared__ __attribute__((aligned(1024))) uint8_t smem_raw[kStages * kTileBytes + kMboxBytes];
  uint8_t K3_LDS* smem = (uint8_t K3_LDS*)smem_raw;
  const int cid = blockIdx.x;
  const int ncomp = p.bs * p.nsplit;
  if (cid >= ncomp) {
    merger_cta(p, smem, cid - ncomp);
    // terminate here so the merger path never joins the compute path's CFG (register-map check)
    asm volatile("s_endpgm" ::: "memory");
    __builtin_unreachable();
  }
  const int b = cid / p.nsplit, split = cid - b * p.nsplit;
#ifdef K3HK_DEBUG
  {
    const int lane = threadIdx.x;
    if (threadIdx.x == 0) K3_TS(7);
  }
#endif
  const int kv_start = p.kv_indptr[b];
  const int L = p.kv_indptr[b + 1] - kv_start;
  const int chunk = split_chunk(L, p.nsplit, p.min_chunk);
  const int start = split * chunk;
  if (start >= L) return;
  const int end = min(start + chunk, L);
  const int ntiles = (end - start + kBN - 1) / kBN;
  const int w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
  const int lane = threadIdx.x & 63;
  if (w == 0) K3_TS(0);
#ifdef K3HK_DEBUG
  const int dbg = __builtin_amdgcn_readfirstlane(kDbg);
#endif
  if (w == 3) {
#ifdef K3HK_LDR_PRIO
    __builtin_amdgcn_s_setprio(3);
#endif
#ifdef K3HK_SD
    loader_wave_sd(p, smem, kv_start, start, end, ntiles, lane K3_DBG_PASS);
#else
    loader_wave(p, smem, kv_start, start, end, ntiles, lane K3_DBG_PASS);
#endif
  } else {
    compute_wave(p, smem, b, split, L, start, end, ntiles, w, lane K3_DBG_PASS);
  }
#ifdef K3HK_TAIL_MERGE
  if (p.tail) {
    const int nact = min((L + chunk - 1) / chunk, p.nsplit);
    tail_merge(p, smem, b, split, nact);
  }
#endif
}

// ---------------------------------------------------------------- split merge
// One workgroup (4 waves) per (row, request); wave w merges splits w, w+4, ...;
// lane L owns columns 8L..8L+7; the 4 partial sums are combined through LDS.
// Same split formula as stage 1; base-2 LSE merge of the active splits.
constexpr int kRedWaves = 4;
// CC: column chunks per row (1: lane owns 8 cols, 16 B loads; 4: lane owns 2 cols of a
// 128-col chunk, for small batches where rows x requests alone cannot fill the GPU).
template <int CC>
__global__ __launch_bounds__(64 * kRedWaves) void k3_mla_verify_hk_reduce(const Params p) {
  constexpr int NC = 8 / CC;  // columns per lane
  __shared__ float red_acc[kRedWaves][NC][64];
  __shared__ float red_w[kRedWaves];
  const int lane = threadIdx.x & 63, w = threadIdx.x >> 6;
  const int row = blockIdx.x / CC, cc = blockIdx.x % CC, b = blockIdx.y;
  const int L = p.kv_indptr[b + 1] - p.kv_indptr[b];
  const int chunk = split_chunk(L, p.nsplit, p.min_chunk);
  const int nact = L > 0 ? min((L + chunk - 1) / chunk, p.nsplit) : 0;
  const float* lse = p.lse_part + (int64_t)b * p.nsplit * kRows + row;
  float m = -INFINITY;
  for (int s = lane; s < nact; s += 64) m = fmaxf(m, lse[(int64_t)s * kRows]);
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 64));
  if (m == -INFINITY) m = 0.f;
  const int col0 = (kDV / CC) * cc + NC * lane;
  const uint16_t* src = p.o_part + ((int64_t)b * p.nsplit * kRows + row) * kDV + col0;
  float acc[NC];
#pragma unroll
  for (int e = 0; e < NC; ++e) acc[e] = 0.f;
  float wsum = 0.f;
  typedef int vec_t __attribute__((ext_vector_type(NC / 2)));
  constexpr int U = 4;
  int s = w;
  for (; s + kRedWaves * (U - 1) < nact; s += kRedWaves * U) {
    float wt[U];
    vec_t x[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int ss = s + kRedWaves * u;
      wt[u] = lse[(int64_t)ss * kRows];
      x[u] = *(const vec_t*)(src + (int64_t)ss * kRows * kDV);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const float wu = __builtin_amdgcn_exp2f(wt[u] - m);
      wsum += wu;
#pragma unroll
      for (int e = 0; e < NC / 2; ++e) {
        const uint32_t v = (uint32_t)x[u][e];
        acc[2 * e] += wu * bf16_to_f(v & 0xffffu);
        acc[2 * e + 1] += wu * bf16_to_f(v >> 16);
      }
    }
  }
  for (; s < nact; s += kRedWaves) {
    const float wu = __builtin_amdgcn_exp2f(lse[(int64_t)s * kRows] - m);
    const vec_t x = *(const vec_t*)(src + (int64_t)s * kRows * kDV);
    wsum += wu;
#pragma unroll
    for (int e = 0; e < NC / 2; ++e) {
      const uint32_t v = (uint32_t)x[e];
      acc[2 * e] += wu * bf16_to_f(v & 0xffffu);
      acc[2 * e + 1] += wu * bf16_to_f(v >> 16);
    }
  }
#pragma unroll
  for (int e = 0; e < NC; ++e) red_acc[w][e][lane] = acc[e];
  if (lane == 0) red_w[w] = wsum;
  __syncthreads();
  if (w != 0) return;
  wsum = red_w[0] + red_w[1] + red_w[2] + red_w[3];
  const float inv = wsum > 0.f ? 1.f / wsum : 0.f;
  vec_t o;
#pragma unroll
  for (int e = 0; e < NC / 2; ++e) {
    const float a0 = red_acc[0][2 * e][lane] + red_acc[1][2 * e][lane] + red_acc[2][2 * e][lane] + red_acc[3][2 * e][lane];
    const float a1 = red_acc[0][2 * e + 1][lane] + red_acc[1][2 * e + 1][lane] + red_acc[2][2 * e + 1][lane] +
                     red_acc[3][2 * e + 1][lane];
    const bf16x2 pk = __builtin_convertvector((v2f{a0 * inv, a1 * inv}), bf16x2);
    o[e] = __builtin_bit_cast(int, pk);
  }
  const int qp = row / kH;
  *(vec_t*)(p.o_final + (int64_t)(b * kQLen + qp) * p.stride_o_tok + (int64_t)(row - qp * kH) * p.stride_o_h + col0) = o;
}

// ---------------------------------------------------------------- split merge, one round trip
// Workgroup (128-column chunk cc, row, request b); thread t = (g = t / 16, c = t % 16): splits
// s = g + 16 j (j < J), columns 128 cc + 8 c .. + 7. All partial / lse loads are issued at once
// (addresses do not depend on kv_indptr: o_part / lse_part are allocated for all nsplit splits;
// inactive splits are masked by selects, so their uninitialised contents never matter), so the
// merge costs one memory round trip instead of indptr -> lse -> max -> partials.
// regime != nullptr (auto path): *regime == 0 -> merge the Gluon splits instead (natural-log
// LSE, = aiter _mla_softmax_reducev_kernel), g_ns == 1 -> nothing to do.

// fp8-partial variant of reduce1_body (o_part rows = 512 fp8 bytes of O / kv_scale); thread loads 8 B
template <int J>
__device__ __forceinline__ void reduce1_body8(const Params& p, float* s_m, float* s_acc, int ip0, int ip1) {
  const int t = threadIdx.x, lane = t & 63, w = t >> 6;
  const int c = t & 15, g = t >> 4;
  const int row = blockIdx.x >> 2, cc = blockIdx.x & 3, b = blockIdx.y;
  const int ns = p.nsplit;
  const int64_t rbase = (int64_t)b * ns * kRows + row;
  const int col = 128 * cc + 8 * c;
  const uint8_t* o8 = (const uint8_t*)p.o_part;
  const int L = ip1 - ip0;
  const int chunk = split_chunk(L, ns, p.min_chunk);
  const int nact = min((L + chunk - 1) / chunk, ns);
  v2i x[J];
  float lv[J];
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const int s = min(g + 16 * j, nact - 1);
    lv[j] = p.lse_part[rbase + (int64_t)s * kRows];
    x[j] = *(const v2i*)(o8 + (rbase + (int64_t)s * kRows) * (2 * kDV) + col);
  }
  const float kvs = *p.kv_scale;
  float m = -INFINITY;
#pragma unroll
  for (int j = 0; j < J; ++j)
    if (g + 16 * j < nact) m = fmaxf(m, lv[j]);
  m = fmaxf(m, __shfl_xor(m, 16, 64));
  m = fmaxf(m, __shfl_xor(m, 32, 64));
  if (lane == 0) s_m[w] = m;
  __syncthreads();
  m = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
  if (m == -INFINITY) m = 0.f;
  float acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const bool ok = g + 16 * j < nact;
    const float wj = ok ? __builtin_amdgcn_exp2f(lv[j] - m) : 0.f;
    acc[8] += wj;
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int u = ok ? x[j][e] : 0;
      const v2f lo = __builtin_amdgcn_cvt_pk_f32_fp8(u, false);
      const v2f hi = __builtin_amdgcn_cvt_pk_f32_fp8(u, true);
      acc[4 * e] += wj * lo[0];
      acc[4 * e + 1] += wj * lo[1];
      acc[4 * e + 2] += wj * hi[0];
      acc[4 * e + 3] += wj * hi[1];
    }
  }
#pragma unroll
  for (int e = 0; e < 9; ++e) {
    acc[e] += __shfl_xor(acc[e], 16, 64);
    acc[e] += __shfl_xor(acc[e], 32, 64);
  }
  if (lane < 16) {
#pragma unroll
    for (int e = 0; e < 9; ++e) s_acc[(w * 16 + c) * 9 + e] = acc[e];
  }
  __syncthreads();
  if (t >= 16) return;
  float o[9];
#pragma unroll
  for (int e = 0; e < 9; ++e)
    o[e] = s_acc[c * 9 + e] + s_acc[(16 + c) * 9 + e] + s_acc[(32 + c) * 9 + e] + s_acc[(48 + c) * 9 + e];
  const float inv = o[8] > 0.f ? kvs / o[8] : 0.f;
  v4i r;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const bf16x2 pk = __builtin_convertvector((v2f{o[2 * e] * inv, o[2 * e + 1] * inv}), bf16x2);
    r[e] = __builtin_bit_cast(int, pk);
  }
  const int qp = row / kH;
  *(v4i*)(p.o_final + (int64_t)(b * kQLen + qp) * p.stride_o_tok + (int64_t)(row - qp * kH) * p.stride_o_h + col) = r;
}

template <int J>
__device__ __forceinline__ void reduce1_body(const Params& p, float* s_m, float* s_acc) {
  const int t = threadIdx.x, lane = t & 63, w = t >> 6;
  const int c = t & 15, g = t >> 4;
  const int row = blockIdx.x >> 2, cc = blockIdx.x & 3, b = blockIdx.y;
  const int ns = p.nsplit;
  const int ip0 = p.kv_indptr[b], ip1 = p.kv_indptr[b + 1];
  const int64_t rbase = (int64_t)b * ns * kRows + row;
  const int col = 128 * cc + 8 * c;
  if (p.part8_min > 0 && ip1 > ip0) {
    const int ch = split_chunk(ip1 - ip0, ns, p.min_chunk);
    if (min((ip1 - ip0 + ch - 1) / ch, ns) >= p.part8_min) {
      reduce1_body8<J>(p, s_m, s_acc, ip0, ip1);
      return;
    }
  }
  v4i x[J];
  float lv[J];
  // splits 0..31 are loaded unconditionally (no wait on kv_indptr), the rest only if active
  constexpr int J0 = J < 2 ? J : 2;
#pragma unroll
  for (int j = 0; j < J0; ++j) {
    const int s = min(g + 16 * j, ns - 1);
    lv[j] = p.lse_part[rbase + (int64_t)s * kRows];
    x[j] = *(const v4i*)(p.o_part + (rbase + (int64_t)s * kRows) * kDV + col);
  }
  const int L = ip1 - ip0;
  const int chunk = split_chunk(L, ns, p.min_chunk);
  const int nact = L > 0 ? min((L + chunk - 1) / chunk, ns) : 0;
#pragma unroll
  for (int j = J0; j < J; ++j) {
    const int s = g + 16 * j;
    lv[j] = -INFINITY;
    x[j] = v4i{0, 0, 0, 0};
    if (s < nact) {
      lv[j] = p.lse_part[rbase + (int64_t)s * kRows];
      x[j] = *(const v4i*)(p.o_part + (rbase + (int64_t)s * kRows) * kDV + col);
    }
  }
  float m = -INFINITY;
#pragma unroll
  for (int j = 0; j < J; ++j)
    if (g + 16 * j < nact) m = fmaxf(m, lv[j]);
  m = fmaxf(m, __shfl_xor(m, 16, 64));
  m = fmaxf(m, __shfl_xor(m, 32, 64));
  if (lane == 0) s_m[w] = m;
  __syncthreads();
  m = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
  if (m == -INFINITY) m = 0.f;
  float acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const bool ok = g + 16 * j < nact;
    const float wj = ok ? __builtin_amdgcn_exp2f(lv[j] - m) : 0.f;
    acc[8] += wj;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const uint32_t u = ok ? (uint32_t)x[j][e] : 0u;
      acc[2 * e] += wj * bf16_to_f(u & 0xffffu);
      acc[2 * e + 1] += wj * bf16_to_f(u >> 16);
    }
  }
#pragma unroll
  for (int e = 0; e < 9; ++e) {
    acc[e] += __shfl_xor(acc[e], 16, 64);
    acc[e] += __shfl_xor(acc[e], 32, 64);
  }
  if (lane < 16) {
#pragma unroll
    for (int e = 0; e < 9; ++e) s_acc[(w * 16 + c) * 9 + e] = acc[e];
  }
  __syncthreads();
  if (t >= 16) return;
  float o[9];
#pragma unroll
  for (int e = 0; e < 9; ++e)
    o[e] = s_acc[c * 9 + e] + s_acc[(16 + c) * 9 + e] + s_acc[(32 + c) * 9 + e] + s_acc[(48 + c) * 9 + e];
  const float inv = o[8] > 0.f ? 1.f / o[8] : 0.f;
  v4i r;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const bf16x2 pk = __builtin_convertvector((v2f{o[2 * e] * inv, o[2 * e + 1] * inv}), bf16x2);
    r[e] = __builtin_bit_cast(int, pk);
  }
  const int qp = row / kH;
  *(v4i*)(p.o_final + (int64_t)(b * kQLen + qp) * p.stride_o_tok + (int64_t)(row - qp * kH) * p.stride_o_h + col) = r;
}

// Gluon-regime merge of (row, cc): two passes over the (few) Gluon splits, thread (g, c) as above.
__device__ __forceinline__ void reduce1_gluon(const Params& p, float* s_m, float* s_acc) {
  const int t = threadIdx.x, lane = t & 63, w = t >> 6;
  const int c = t & 15, g = t >> 4;
  const int row = blockIdx.x >> 2, cc = blockIdx.x & 3, b = blockIdx.y;
  const int qp = row / kH, h = row - qp * kH;
  const int col = 128 * cc + 8 * c;
  const int gL = p.g_indptr[b + 1] - p.g_indptr[b];
  const int gper = max(p.g_block_n, gL / p.g_ns);
  const int gnact = gL > 0 ? min((gL + gper - 1) / gper, p.g_ns) : 0;
  const uint16_t* lg = p.g_logits + b * p.g_sl_b + qp * p.g_sl_qs + h * p.g_sl_h + col;
  const float* ls = p.g_lse + b * p.g_ml_b + qp * p.g_ml_qs + h * p.g_ml_h;
  float acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  float m = -INFINITY;
  if (p.g_ns <= 32) {
    // one round trip: all (<= 2 per thread) splits loaded before kv_indptr is known
    float lv[2];
    v4i x[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int s = min(g + 16 * j, p.g_ns - 1);
      lv[j] = ls[s * p.g_ml_s];
      x[j] = *(const v4i*)(lg + s * p.g_sl_s);
    }
#pragma unroll
    for (int j = 0; j < 2; ++j)
      if (g + 16 * j < gnact) m = fmaxf(m, lv[j]);
    m = fmaxf(m, __shfl_xor(m, 16, 64));
    m = fmaxf(m, __shfl_xor(m, 32, 64));
    if (lane == 0) s_m[w] = m;
    __syncthreads();
    m = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const bool ok = g + 16 * j < gnact && lv[j] != -INFINITY;
      const float wk = ok ? __expf(lv[j] - m) : 0.f;
      acc[8] += wk;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const uint32_t u = ok ? (uint32_t)x[j][e] : 0u;
        acc[2 * e] += wk * bf16_to_f(u & 0xffffu);
        acc[2 * e + 1] += wk * bf16_to_f(u >> 16);
      }
    }
  } else {
  for (int s = g; s < gnact; s += 16) m = fmaxf(m, ls[s * p.g_ml_s]);
  m = fmaxf(m, __shfl_xor(m, 16, 64));
  m = fmaxf(m, __shfl_xor(m, 32, 64));
  if (lane == 0) s_m[w] = m;
  __syncthreads();
  m = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
  for (int s = g; s < gnact; s += 16) {
    const float l = ls[s * p.g_ml_s];
    const v4i x = *(const v4i*)(lg + s * p.g_sl_s);
    if (l == -INFINITY) continue;
    const float wk = __expf(l - m);
    acc[8] += wk;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const uint32_t u = (uint32_t)x[e];
      acc[2 * e] += wk * bf16_to_f(u & 0xffffu);
      acc[2 * e + 1] += wk * bf16_to_f(u >> 16);
    }
  }
  }
#pragma unroll
  for (int e = 0; e < 9; ++e) {
    acc[e] += __shfl_xor(acc[e], 16, 64);
    acc[e] += __shfl_xor(acc[e], 32, 64);
  }
  if (lane < 16) {
#pragma unroll
    for (int e = 0; e < 9; ++e) s_acc[(w * 16 + c) * 9 + e] = acc[e];
  }
  __syncthreads();
  if (t >= 16) return;
  float o[9];
#pragma unroll
  for (int e = 0; e < 9; ++e)
    o[e] = s_acc[c * 9 + e] + s_acc[(16 + c) * 9 + e] + s_acc[(32 + c) * 9 + e] + s_acc[(48 + c) * 9 + e];
  const float inv = o[8] > 0.f ? 1.f / o[8] : 1.f;
  v4i r;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const bf16x2 pk = __builtin_convertvector((v2f{o[2 * e] * inv, o[2 * e + 1] * inv}), bf16x2);
    r[e] = __builtin_bit_cast(int, pk);
  }
  *(v4i*)(p.o_final + (int64_t)(b * kQLen + qp) * p.stride_o_tok + (int64_t)h * p.stride_o_h + col) = r;
}

template <int J>
__global__ __launch_bounds__(256) void k3_mla_verify_hk_reduce1(const Params p) {
  __shared__ float s_m[4];
  __shared__ float s_acc[64 * 9];
  if (p.regime != nullptr && *p.regime == 0) {
    if (p.g_ns > 1) reduce1_gluon(p, s_m, s_acc);
    return;
  }
  reduce1_body<J>(p, s_m, s_acc);
}

// launch: grid (96 * 4, bs), 256 threads; J = ceil(nsplit / 16) rounded up to a power of two
inline void launch_reduce1(const Params& p, int bs, hipStream_t st) {
  const dim3 grid(kRows * 4, bs), blk(256);
  const int J = (p.nsplit + 15) / 16;
  if (J <= 1) hipLaunchKernelGGL(k3_mla_verify_hk_reduce1<1>, grid, blk, 0, st, p);
  else if (J <= 2) hipLaunchKernelGGL(k3_mla_verify_hk_reduce1<2>, grid, blk, 0, st, p);
  else if (J <= 4) hipLaunchKernelGGL(k3_mla_verify_hk_reduce1<4>, grid, blk, 0, st, p);
  else if (J <= 8) hipLaunchKernelGGL(k3_mla_verify_hk_reduce1<8>, grid, blk, 0, st, p);
  else hipLaunchKernelGGL(k3_mla_verify_hk_reduce1<16>, grid, blk, 0, st, p);
}

}  // namespace k3hk

#ifndef K3HK_STANDALONE
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

struct K3MlaVerifyHK {
  // q [bs*8, 12, 576] bf16 (row stride arbitrary, inner contiguous); kv fp8 [N, 576] (any row stride);
  // kv_indptr [>= bs+1] int32; kv_indices int32; kv_scale fp32 [1];
  // o_part bf16 [bs, nsplit, 96, 512]; lse_part fp32 [bs, nsplit, 96]; out bf16 [bs*8, 12, 512].
  static void run(const tvm::ffi::TensorView q,
                  const tvm::ffi::TensorView kv,
                  const tvm::ffi::TensorView kv_indptr,
                  const tvm::ffi::TensorView kv_indices,
                  const tvm::ffi::TensorView kv_scale,
                  const tvm::ffi::TensorView o_part,
                  const tvm::ffi::TensorView lse_part,
                  const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView flags,
                  int64_t nsplit,
                  int64_t min_chunk,
                  double sm_scale_log2,
                  int64_t mrg_rows,
                  int64_t part8_min) {
    using namespace host;
    const int64_t ntok = q.size(0);
    const int64_t bs = ntok / k3hk::kQLen;
    RuntimeCheck(bs * k3hk::kQLen == ntok, "token count must be a multiple of 8");
    RuntimeCheck(q.size(1) == k3hk::kH && q.size(2) == k3hk::kD && q.stride(2) == 1, "q must be [*, 12, 576]");
    RuntimeCheck(kv.size(kv.dim() - 1) == k3hk::kD && kv.stride(kv.dim() - 1) == 1, "kv must be [N, 576]");
    RuntimeCheck(out.size(1) == k3hk::kH && out.size(2) == k3hk::kDV && out.stride(2) == 1, "bad out");
    RuntimeCheck(nsplit >= 1 && min_chunk % k3hk::kBN == 0, "bad split config");
    if (bs == 0) return;
    k3hk::Params p{};
    p.q = static_cast<const uint16_t*>(q.data_ptr());
    p.kv = static_cast<const uint8_t*>(kv.data_ptr());
    p.kv_indptr = static_cast<const int32_t*>(kv_indptr.data_ptr());
    p.kv_indices = static_cast<const int32_t*>(kv_indices.data_ptr());
    p.kv_scale = static_cast<const float*>(kv_scale.data_ptr());
    p.o_part = static_cast<uint16_t*>(o_part.data_ptr());
    p.lse_part = static_cast<float*>(lse_part.data_ptr());
    p.o_final = static_cast<uint16_t*>(out.data_ptr());
    p.stride_q_tok = q.stride(0);
    p.stride_q_h = q.stride(1);
    p.stride_o_tok = out.stride(0);
    p.stride_o_h = out.stride(1);
    const int64_t kv_rows = kv.size(0);
    const int64_t kv_stride = kv.stride(0);
    const int64_t kv_bytes = (kv_rows - 1) * kv_stride + k3hk::kD;
    RuntimeCheck(kv_bytes <= 0xFFFFFFFFll, "kv pool > 4 GiB is not supported by the buffer path");
    p.kv_stride = static_cast<int32_t>(kv_stride);
    p.kv_bytes = static_cast<uint32_t>(kv_bytes);
    p.nsplit = static_cast<int32_t>(nsplit);
    p.min_chunk = static_cast<int32_t>(min_chunk);
    p.sm_scale_log2 = static_cast<float>(sm_scale_log2);
    p.q_lds = (q.stride(1) == k3hk::kD && q.stride(0) == k3hk::kH * k3hk::kD) ? 1 : 0;
    p.bs = static_cast<int32_t>(bs);
    p.flags = nullptr;
    p.mrg_rows = 1;
    p.nmrg = 0;
    p.part8_min = static_cast<int32_t>(part8_min);
    if (mrg_rows > 0 && nsplit > 1) {
      // fused split merge: merger CTAs after the compute CTAs (see merger_body)
      RuntimeCheck(mrg_rows == 1 || mrg_rows == 2 || mrg_rows == 4 || mrg_rows == 8, "mrg_rows must be 1/2/4/8");
      RuntimeCheck(bs * nsplit <= k3hk::kFlagCap && bs <= k3hk::kFlagCap, "fused merge: bs * nsplit too large");
      RuntimeCheck(flags.numel() >= 2 * k3hk::kFlagCap, "fused merge: flags buffer too small");
      RuntimeCheck(bs * k3hk::kRows / mrg_rows <= 192, "fused merge: too many merger CTAs");
      p.flags = static_cast<int32_t*>(flags.data_ptr());
      p.mrg_rows = static_cast<int32_t>(mrg_rows);
      p.nmrg = static_cast<int32_t>(bs * k3hk::kRows / mrg_rows);
      p.part8_min = 0;  // the merger CTAs read bf16 partials
    }
    LaunchKernel(dim3(static_cast<uint32_t>(nsplit * bs + p.nmrg)), k3hk::kThreads, q.device())(
        k3hk::k3_mla_verify_hk_kernel, p);
  }

  // split merge (one memory round trip, k3_mla_verify_hk_reduce1). use_regime != 0 (auto path):
  // regime [1] int32 (0 = Gluon ran: merge g_logits [bs, qlen, H, g_ns, 512] / g_lse
  // [bs, qlen, H, g_ns] with g_indptr's split formula; else merge o_part / lse_part).
  static void reduce(const tvm::ffi::TensorView o_part,
                     const tvm::ffi::TensorView lse_part,
                     const tvm::ffi::TensorView kv_indptr,
                     const tvm::ffi::TensorView out,
                     const tvm::ffi::TensorView regime,
                     const tvm::ffi::TensorView g_logits,
                     const tvm::ffi::TensorView g_lse,
                     const tvm::ffi::TensorView kv_scale,
                     int64_t nsplit,
                     int64_t min_chunk,
                     int64_t use_regime,
                     int64_t g_ns,
                     int64_t g_block_n,
                     int64_t part8_min) {
    using namespace host;
    const int64_t ntok = out.size(0);
    const int64_t bs = ntok / k3hk::kQLen;
    RuntimeCheck(out.size(1) == k3hk::kH && out.size(2) == k3hk::kDV && out.stride(2) == 1, "bad out");
    RuntimeCheck(nsplit >= 1 && nsplit <= 256, "reduce: nsplit must be in [1, 256]");
    if (bs == 0) return;
    k3hk::Params p{};
    p.kv_indptr = static_cast<const int32_t*>(kv_indptr.data_ptr());
    p.o_part = static_cast<uint16_t*>(o_part.data_ptr());
    p.lse_part = static_cast<float*>(lse_part.data_ptr());
    p.o_final = static_cast<uint16_t*>(out.data_ptr());
    p.stride_o_tok = out.stride(0);
    p.stride_o_h = out.stride(1);
    p.nsplit = static_cast<int32_t>(nsplit);
    p.min_chunk = static_cast<int32_t>(min_chunk);
    p.bs = static_cast<int32_t>(bs);
    p.kv_scale = static_cast<const float*>(kv_scale.data_ptr());
    p.part8_min = static_cast<int32_t>(part8_min);
    p.regime = nullptr;
    if (use_regime) {
      p.regime = static_cast<const int32_t*>(regime.data_ptr());
      p.g_indptr = p.kv_indptr;
      p.g_ns = static_cast<int32_t>(g_ns);
      p.g_block_n = static_cast<int32_t>(g_block_n);
      if (g_ns > 1) {
        p.g_logits = static_cast<const uint16_t*>(g_logits.data_ptr());
        p.g_lse = static_cast<const float*>(g_lse.data_ptr());
        p.g_sl_b = g_logits.stride(0);
        p.g_sl_qs = g_logits.stride(1);
        p.g_sl_h = g_logits.stride(2);
        p.g_sl_s = g_logits.stride(3);
        p.g_ml_b = g_lse.stride(0);
        p.g_ml_qs = g_lse.stride(1);
        p.g_ml_h = g_lse.stride(2);
        p.g_ml_s = g_lse.stride(3);
      }
    }
    const dim3 grid(k3hk::kRows * 4, static_cast<uint32_t>(bs));
    const int J = static_cast<int>((nsplit + 15) / 16);
    auto L = LaunchKernel(grid, 256, out.device());
    if (J <= 1) L(k3hk::k3_mla_verify_hk_reduce1<1>, p);
    else if (J <= 2) L(k3hk::k3_mla_verify_hk_reduce1<2>, p);
    else if (J <= 4) L(k3hk::k3_mla_verify_hk_reduce1<4>, p);
    else if (J <= 8) L(k3hk::k3_mla_verify_hk_reduce1<8>, p);
    else L(k3hk::k3_mla_verify_hk_reduce1<16>, p);
  }
};

}  // namespace sglang
#endif  // K3HK_STANDALONE
