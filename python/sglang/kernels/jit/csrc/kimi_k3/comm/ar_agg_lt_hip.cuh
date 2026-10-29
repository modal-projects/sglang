// K3 ROCm TP8, 65..512 tokens: the o_proj all-reduce (or the MoE up_proj
// all-gather + add3) fused with the attention-residual aggregation that
// consumes it, in one launch. Same math and summation order as ar_agg_hip.cuh
// (the <= 80 token variant, whose header explains the statistics scheme);
// what changes is the work mapping, which there is one producer block per
// token (start-barrier slot t) and T x split blocks in the grid.
//
// Here: n_prod = min(T, 80) producer blocks (one AITER start barrier on their
// own slot each, in both modes) reduce-scatter tokens t = b, b + n_prod, ...:
// rank r reduces column slice r of the token, adds the pending prefix (kAR) or
// does the add3 (kAG), stores the bf16 slice and the slice's score / Gram
// partials in its private region and flags the token's 8 x split consumer
// items on every rank. Every block (producers included, after producing) then
// consumes (token, column chunk) items it = b, b + grid, ...: waits for the 8
// producers' flags, sums the statistics in rank order, gathers its columns of
// the row from the peers' slices, mixes with the local bank, RMSNorms and
// writes prefix_out / bank snapshot / out / E4M3 / stream. Producers never
// wait on consumers and have the lowest block ids, so the grid needs no
// co-residency guarantee beyond the producers being scheduled first.
//
// Buffers: a private 4 MiB region below ar_agg_hip's 1 MiB at the tail of
// AITER's tmp area, single-buffered: a producer writes launch n only after
// its start barrier saw every rank's producer of launch n, i.e. every rank's
// launch n - 1 (all of whose consumers read the previous contents) has
// finished in stream order. Flags are reset by their consumer. kAR peer
// inputs are all read before a rank's kernel can end (its consumers wait on
// every rank's producer of every token), so no end barrier.
#pragma once

#ifndef USE_ROCM
#error "ar_agg_lt_hip.cuh is ROCm only"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <custom_all_reduce.cuh>  // AITER: CustomAllreduce, RankData, Signal

namespace sglang {
namespace k3_ar_agg_lt {

constexpr int kNG = 8;
constexpr int kVec = 8;
constexpr int kThreads = 256;
constexpr int kMaxProd = aiter::kMaxBlocks;  // start-barrier slots
constexpr int kMaxT = 512;
constexpr int kDefaultGrid = 512;
constexpr int64_t kRegionBytes = 4 << 20;
constexpr int64_t kDataOff = 0, kDataBytes = 1 << 20;                          // [T][slice] packs
constexpr int64_t kStatOff = 1 << 20, kStatBytes = 256 << 10;                  // [T][64] fp32
constexpr int64_t kFlagOff = (1 << 20) + (256 << 10), kFlagBytes = 256 << 10;  // [T * split][8] u32
constexpr int kStatStride = 64;

using vec_t = device::AlignedVector<bf16_t, kVec>;
using OP = opus::vector_t<opus::bf16_t, kVec>;
using OA = opus::vector_t<opus::fp32_t, kVec>;

enum Mode : int { kAR = 0, kAG = 1 };

struct Params {
  aiter::RankData* dp;  // kAR: peer addresses of the registered partials
  aiter::RankSignals sg;
  aiter::Signal* self_sg;
  int rank, T, split, n_prod, grid;
  int64_t region_off;                 // byte offset of the private region in the tmp area
  const bf16_t* __restrict__ y;       // kAG: this rank's [T, H/8] slice (contiguous)
  const bf16_t* __restrict__ prefix;  // kAR: optional pending prefix [T, H]
  const bf16_t* __restrict__ add_b;   // kAG [T, H]
  const bf16_t* __restrict__ add_c;   // kAG optional [T, H]
  bf16_t* __restrict__ prefix_out;    // [T, H]
  bf16_t* __restrict__ bank;          // [T, NB, H]
  const float* __restrict__ cw;       // [H]
  const bf16_t* __restrict__ ow;      // [H] or null
  bf16_t* __restrict__ out;           // [T, H]
  uint8_t* __restrict__ out8;         // [T, H] or null
  bf16_t* __restrict__ stream;        // [T, H] or null
  int64_t stride_p, stride_b, stride_c, stride_po, stride_bm, stride_bb, stride_o, stride_o8, stride_s;
  int32_t hidden, n_vecs, slice_vecs, chunk_vecs;
  int32_t write_bank;
  float score_eps, out_eps;
};

__device__ __forceinline__ char* region(const Params& p, int q) {
  return reinterpret_cast<char*>(aiter::get_tmp_buf<char>(p.sg.signals[q])) + p.region_off;
}

// AITER start_sync on an explicit slot (only the producer blocks take part),
// split into arrive (post this rank's flag) and wait (all peers posted) so
// local work can overlap the barrier.
__device__ __forceinline__ uint32_t start_arrive(const Params& p, int slot) {
  const uint32_t flag = p.self_sg->_flag[slot] + 1;
  if (threadIdx.x < kNG)
    __scoped_atomic_store_n(
        &p.sg.signals[threadIdx.x]->start[slot][p.rank], flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  return flag;
}

__device__ __forceinline__ void start_wait(const Params& p, int slot, uint32_t flag) {
  if (threadIdx.x < kNG) {
    while (__scoped_atomic_load_n(&p.self_sg->start[slot][threadIdx.x], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < flag)
      ;
  }
  __syncthreads();
  if (threadIdx.x == 0) p.self_sg->_flag[slot] = flag;
}

__device__ __forceinline__ uint8_t to_e4m3(float x) {
  return static_cast<uint8_t>(__hip_cvt_float_to_fp8(x, __HIP_SATFINITE, __HIP_E4M3));
}

typedef float f2_t __attribute__((ext_vector_type(2)));

__device__ __forceinline__ void unpack2(const vec_t& v, f2_t (&f)[kVec / 2]) {
  const uint32_t* u = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
  for (int i = 0; i < kVec / 2; ++i)
    f[i] = f2_t{__uint_as_float(u[i] << 16), __uint_as_float(u[i] & 0xffff0000u)};
}

__device__ __forceinline__ void unpack(const vec_t& v, float (&f)[kVec]) {
  const uint32_t* u = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
  for (int i = 0; i < kVec / 2; ++i) {
    f[2 * i] = __uint_as_float(u[i] << 16);
    f[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
  }
}

template <int kMode, int kNVB>
__device__ __forceinline__ void produce(const Params& p, const int t, const bool valid, float* red_all) {
  using namespace device;
  constexpr int kR = kNVB + 1;
  constexpr int kG = kR * (kR + 1) / 2;
  constexpr int kS = kR + kG;
  const int sv = p.slice_vecs;
  const bf16_t* bank_t = p.bank + static_cast<int64_t>(t) * p.stride_bm;
  // ======================= producer: slice `rank` of token t =======================
  char* mine = region(p, p.rank);
  // two tokens per pass: threads [0, 128) and [128, 256) are independent
  // half-producers (sv <= 128), so one block keeps two slices in flight
  const int i = threadIdx.x & 127;
  const bool pact = valid && i < sv;
  float* red = red_all + (threadIdx.x >> 7) * (kS * 128);
  const int pv_idx = p.rank * sv + i;
  // local operands first (before the start barrier)
  vec_t bk[kNVB];
  float4 cw0, cw1;
  vec_t av, bv, cv;
  if (pact) {
#pragma unroll
    for (int r = 0; r < kNVB; ++r) {
      bk[r].load(bank_t + r * p.stride_bb, pv_idx);
    }
    cw0 = reinterpret_cast<const float4*>(p.cw)[2 * pv_idx];
    cw1 = reinterpret_cast<const float4*>(p.cw)[2 * pv_idx + 1];
    if constexpr (kMode == kAR) {
      if (p.prefix != nullptr) av.load(p.prefix + t * p.stride_p, pv_idx);
    } else {
      av.load(p.y + static_cast<int64_t>(t) * sv * kVec, i);
      bv.load(p.add_b + t * p.stride_b, pv_idx);
      if (p.add_c != nullptr) cv.load(p.add_c + t * p.stride_c, pv_idx);
    }
  }
  // kAR: issue the peer loads first; the bank statistics below overlap them
  OP xin[kNG];
  int flat = 0;
  if constexpr (kMode == kAR) {
    if (pact) {
      flat = t * p.n_vecs + pv_idx;
      const int owner = flat / (p.T * sv);  // AITER 2-stage flat partition
#pragma unroll
      for (int q = 0; q < kNG; ++q)
        xin[q] = reinterpret_cast<const OP*>(p.dp->ptrs[(owner + q) & (kNG - 1)])[flat];
    }
  }
  // packed fp32 math (v_pk_fma_f32): element pairs
  f2_t pp[kS];
#pragma unroll
  for (int s = 0; s < kS; ++s)
    pp[s] = f2_t{0.f, 0.f};
  f2_t x[kR][kVec / 2];
  f2_t cwv[kVec / 2];
  if (pact) {
#pragma unroll
    for (int r = 0; r < kNVB; ++r)
      unpack2(bk[r], x[r]);
    cwv[0] = f2_t{cw0.x, cw0.y};
    cwv[1] = f2_t{cw0.z, cw0.w};
    cwv[2] = f2_t{cw1.x, cw1.y};
    cwv[3] = f2_t{cw1.z, cw1.w};
#pragma unroll
    for (int e = 0; e < kVec / 2; ++e) {
#pragma unroll
      for (int a = 0; a < kNVB; ++a)
        pp[a] = x[a][e] * cwv[e] + pp[a];
      int idx = kR;
#pragma unroll
      for (int a = 0; a < kR; ++a)
#pragma unroll
        for (int b = a; b < kR; ++b) {
          if (b < kNVB) pp[idx] = x[a][e] * x[b][e] + pp[idx];
          ++idx;
        }
    }
  }
  if (pact) {
    vec_t row;
    if constexpr (kMode == kAR) {
      OA acc = aiter::upcast(xin[0]);
#pragma unroll
      for (int q = 1; q < kNG; ++q) {
#pragma unroll
        for (int e = 0; e < kVec; ++e)
          acc[e] += aiter::upcast_s(xin[q][e]);
      }
      const OP ar = aiter::downcast<OP>(acc);
      row = *reinterpret_cast<const vec_t*>(&ar);
      if (p.prefix != nullptr) {
        // attn_res_smallm: bf16(prefix + addend), addend = the AR result
#pragma unroll
        for (int e = 0; e < kVec; ++e)
          row[e] = cast<bf16_t>(cast<fp32_t>(av[e]) + cast<fp32_t>(row[e]));
      }
    } else {
      // AITER all_gather_lastdim_add: bf16(bf16(y + b) [+ c])
      OA acc = aiter::upcast(*reinterpret_cast<const OP*>(&av));
      aiter::packed_assign_add<opus::fp32_t, kVec>(acc, aiter::upcast(*reinterpret_cast<const OP*>(&bv)));
      OP o = aiter::downcast<OP>(acc);
      if (p.add_c != nullptr) {
        OA acc2 = aiter::upcast(o);
        aiter::packed_assign_add<opus::fp32_t, kVec>(acc2, aiter::upcast(*reinterpret_cast<const OP*>(&cv)));
        o = aiter::downcast<OP>(acc2);
      }
      row = *reinterpret_cast<const vec_t*>(&o);
    }
    row.store(mine + kDataOff, static_cast<int64_t>(t) * sv + i);

    // row partials: <row, cw>, <bank_a, row>, <row, row>
    unpack2(row, x[kNVB]);
#pragma unroll
    for (int e = 0; e < kVec / 2; ++e) {
      pp[kNVB] = x[kNVB][e] * cwv[e] + pp[kNVB];
      int idx = kR;
#pragma unroll
      for (int a = 0; a < kR; ++a) {
        idx += kNVB - a;  // (a, kNVB) is the last pair of row a
        pp[idx] = x[a][e] * x[kNVB][e] + pp[idx];
        ++idx;
      }
    }
  }
  float part[kS];
#pragma unroll
  for (int s = 0; s < kS; ++s)
    part[s] = pp[s].x + pp[s].y;
  // half-block reduction of the kS partials (the slice has sv <= kRedT
  // vectors): LDS transpose, 2 lanes per value
  constexpr int kRedT = 128, kLanes = 2;
  static_assert(kS * kLanes <= kRedT, "reduction layout");
  {
#pragma unroll
    for (int s = 0; s < kS; ++s)
      red[s * kRedT + i] = part[s];
  }
  __syncthreads();
  {
    const int s = i / kLanes, l = i % kLanes;
    float sum = 0.f;
    if (s < kS) {
#pragma unroll
      for (int j = 0; j < kRedT / kLanes; ++j)
        sum += red[s * kRedT + j * kLanes + l];
    }
    sum += __shfl_xor(sum, 1, 64);
    if (valid && s < kS && l == 0) reinterpret_cast<float*>(mine + kStatOff)[t * kStatStride + s] = sum;
  }
  __syncthreads();  // red is reused by the next token
}

template <int kMode, int kNVB>
__device__ __forceinline__ void consume(const Params& p, const int it, float* stat) {
  using namespace device;
  constexpr int kR = kNVB + 1;
  constexpr int kG = kR * (kR + 1) / 2;
  constexpr int kS = kR + kG;
  constexpr int kS4 = (kS + 3) / 4;
  const int t = it / p.split;
  const int k = it - t * p.split;
  const int sv = p.slice_vecs;
  const bf16_t* bank_t = p.bank + static_cast<int64_t>(t) * p.stride_bm;
  const int v = k * p.chunk_vecs + threadIdx.x;
  const bool act = threadIdx.x < p.chunk_vecs && v < p.n_vecs;
  vec_t rows[kR];
  vec_t owv;
  if (act) {
#pragma unroll
    for (int r = 0; r < kNVB; ++r) {
      rows[r].load(bank_t + r * p.stride_bb, v);
    }
    if (p.ow != nullptr) owv.load(p.ow, v);
  }

  // ======================= consumer: columns [k*chunk, (k+1)*chunk) of token t =======================
  {
    uint32_t* f = reinterpret_cast<uint32_t*>(region(p, p.rank) + kFlagOff) + it * kNG;
    if (threadIdx.x < kNG) {
      // relaxed polling (an acquire load per spin invalidates the caches every
      // iteration and slows every other block down), one acquire fence after
      while (__scoped_atomic_load_n(f + threadIdx.x, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == 0u)
        ;
      __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
      __scoped_atomic_store_n(f + threadIdx.x, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    }
    __syncthreads();
  }
  // statistics (rank-ordered sum: identical on every rank) and this block's
  // row columns; wave 0 turns the statistics into the mixing weights and the
  // output-norm scale with one value per lane (lane s holds statistic s)
  if (act) {
    const int q = v / sv;
    rows[kNVB].load(region(p, q) + kDataOff, static_cast<int64_t>(t) * sv + (v - q * sv));
  }
  if (threadIdx.x < kS) {
    float sv8[kNG];
#pragma unroll
    for (int q = 0; q < kNG; ++q)
      sv8[q] = reinterpret_cast<const float*>(region(p, q) + kStatOff)[t * kStatStride + threadIdx.x];
    float st = 0.f;
#pragma unroll
    for (int q = 0; q < kNG; ++q)
      st += sv8[q];
    stat[threadIdx.x] = st;
  }
  __syncthreads();

  // scores, global-max softmax, ||mix||^2 = w^T G w: redundant per thread,
  // from registers (one vectorized LDS read of all statistics)
  float sr[kS4 * 4];
#pragma unroll
  for (int i = 0; i < kS4; ++i) {
    const float4 f4 = reinterpret_cast<const float4*>(stat)[i];
    sr[4 * i] = f4.x;
    sr[4 * i + 1] = f4.y;
    sr[4 * i + 2] = f4.z;
    sr[4 * i + 3] = f4.w;
  }
  const float inv_h = 1.0f / static_cast<float>(p.hidden);
  float w[kR];
  {
    float score[kR];
    int d = kR;
#pragma unroll
    for (int a = 0; a < kR; ++a) {
      score[a] = sr[a] * __builtin_amdgcn_rsqf(sr[d] * inv_h + p.score_eps);  // G_aa
      d += kR - a;
    }
    float m = score[kNVB];
#pragma unroll
    for (int r = 0; r < kNVB; ++r)
      m = fmaxf(m, score[r]);
    float den = 0.f;
#pragma unroll
    for (int r = 0; r < kNVB; ++r) {
      w[r] = __builtin_amdgcn_exp2f((score[r] - m) * 1.4426950408889634f);
      den += w[r];
    }
    w[kNVB] = __builtin_amdgcn_exp2f((score[kNVB] - m) * 1.4426950408889634f);
    const float inv = __builtin_amdgcn_rcpf(den + w[kNVB]);
#pragma unroll
    for (int r = 0; r < kR; ++r)
      w[r] *= inv;
  }
  float scale = 1.f;
  if (p.ow != nullptr) {
    float asq = 0.f;
    int idx = kR;
#pragma unroll
    for (int a = 0; a < kR; ++a) {
      asq += w[a] * w[a] * sr[idx++];
      float cross = 0.f;
#pragma unroll
      for (int b = a + 1; b < kR; ++b)
        cross += w[b] * sr[idx++];
      asq += 2.f * w[a] * cross;
    }
    scale = __builtin_amdgcn_rsqf(fmaxf(asq, 0.f) * inv_h + p.out_eps);
  }
  if (!act) return;  // caller syncs before the next item

  rows[kNVB].store(p.prefix_out + t * p.stride_po, v);
  if (p.write_bank) rows[kNVB].store(p.bank + t * p.stride_bm + kNVB * p.stride_bb, v);

  float acc[kVec];
  {
    float xb[kVec];
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      acc[e] = 0.f;
#pragma unroll
    for (int r = 0; r < kNVB; ++r) {
      unpack(rows[r], xb);
#pragma unroll
      for (int e = 0; e < kVec; ++e)
        acc[e] = w[r] * xb[e] + acc[e];
    }
    unpack(rows[kNVB], xb);
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      acc[e] = xb[e] * w[kNVB] + acc[e];
  }
  if (p.stream != nullptr) {
    vec_t o;
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      o[e] = cast<bf16_t>(acc[e]);
    o.store(p.stream + t * p.stride_s, v);
  }
  vec_t o;
  if (p.ow != nullptr) {
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      o[e] = cast<bf16_t>(acc[e] * scale * cast<fp32_t>(owv[e]));
  } else {
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      o[e] = cast<bf16_t>(acc[e]);
  }
  o.store(p.out + t * p.stride_o, v);
  if (p.out8 != nullptr) {
    uint32_t q8[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t packed = 0;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float x = fminf(fmaxf(cast<fp32_t>(o[h * 4 + e]), -448.f), 448.f);
        packed |= static_cast<uint32_t>(to_e4m3(x)) << (8 * e);
      }
      q8[h] = packed;
    }
    *reinterpret_cast<uint2*>(p.out8 + t * p.stride_o8 + v * kVec) = make_uint2(q8[0], q8[1]);
  }
}

template <int kMode, int kNVB>
__global__ void __launch_bounds__(kThreads) ar_agg_lt_kernel(const Params __grid_constant__ p) {
  constexpr int kR = kNVB + 1;
  constexpr int kS = kR + kR * (kR + 1) / 2;
  static_assert(kS <= kStatStride, "statistics layout");
  constexpr int kS4 = (kS + 3) / 4;
  __shared__ float red[2 * kS * 128];
  __shared__ __attribute__((aligned(16))) float stat[kS4 * 4];

  if (static_cast<int>(blockIdx.x) < p.n_prod) {
    const uint32_t sflag = start_arrive(p, blockIdx.x);
    start_wait(p, blockIdx.x, sflag);
    for (int t = blockIdx.x; t < p.T; t += 2 * p.n_prod) {
      const int th = t + (threadIdx.x >> 7) * p.n_prod;
      produce<kMode, kNVB>(p, th, th < p.T, red);
    }
    // one release for all of this block's tokens: the 8 x split consumer
    // items of each token on every rank (a system-scope release per token
    // costs a round of its own)
    const int per = kNG * p.split;
    const int n = (p.T - static_cast<int>(blockIdx.x) + p.n_prod - 1) / p.n_prod * per;
    for (int j = threadIdx.x; j < n; j += kThreads) {
      const int t = blockIdx.x + (j / per) * p.n_prod;
      const int r = j % per;
      const int q = r / p.split, kk = r - q * p.split;
      uint32_t* f = reinterpret_cast<uint32_t*>(region(p, q) + kFlagOff) + (t * p.split + kk) * kNG + p.rank;
      __scoped_atomic_store_n(f, 1u, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }
  }
  const int items = p.T * p.split;
  for (int it = blockIdx.x; it < items; it += gridDim.x) {
    consume<kMode, kNVB>(p, it, stat);
    __syncthreads();  // stat is reused by the next item
  }
}

struct ArAggLt {
  template <int kMode, int kNVB>
  static void launch(const Params& params, hipStream_t stream) {
    hipLaunchKernelGGL((ar_agg_lt_kernel<kMode, kNVB>), dim3(params.grid), dim3(kThreads), 0, stream, params);
  }

  template <int kMode>
  static void dispatch(const Params& params, int64_t nvb, hipStream_t stream) {
    switch (nvb) {
      case 1:
        launch<kMode, 1>(params, stream);
        break;
      case 2:
        launch<kMode, 2>(params, stream);
        break;
      case 3:
        launch<kMode, 3>(params, stream);
        break;
      case 4:
        launch<kMode, 4>(params, stream);
        break;
      case 5:
        launch<kMode, 5>(params, stream);
        break;
      case 6:
        launch<kMode, 6>(params, stream);
        break;
      case 7:
        launch<kMode, 7>(params, stream);
        break;
      case 8:
        launch<kMode, 8>(params, stream);
        break;
      default:
        host::RuntimeCheck(false, "nvb must be in [1, 8]");
    }
  }

  // mode 0 (AR): x = this rank's TP-partial [T, H]; a = optional pending
  //   prefix [T, H]; b, c unused.
  // mode 1 (AG): x = this rank's up_proj column slice y [T, H/8]; b = add_b
  //   [T, H]; c = optional add_c [T, H]; a unused.
  // flags: bit0 a, bit1 c, bit2 write_bank, bit3 out norm, bit4 out8, bit5 stream.
  // reg_inp_ptr != 0 (AR, eager / copy-in capture): x is first staged into
  // AITER's registered input pool at that address, as AITER's all_reduce does.
  static void
  run(int64_t comm,
      int64_t mode,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView a,
      const tvm::ffi::TensorView b,
      const tvm::ffi::TensorView c,
      const tvm::ffi::TensorView prefix_out,
      const tvm::ffi::TensorView bank,
      const tvm::ffi::TensorView cw,
      const tvm::ffi::TensorView ow,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView out8,
      const tvm::ffi::TensorView stream_out,
      int64_t nvb,
      int64_t flags,
      double score_eps,
      double out_eps,
      int64_t reg_inp_ptr,
      int64_t reg_inp_bytes,
      int64_t region_off,
      int64_t parity,
      int64_t split,
      int64_t grid_cap) {
    using namespace host;
    auto* ca = reinterpret_cast<aiter::CustomAllreduce*>(comm);
    RuntimeCheck(ca->world_size_ == kNG, "ar_agg is TP8 only");
    const int64_t T = prefix_out.size(0);
    const int64_t H = prefix_out.size(1);
    RuntimeCheck(T >= 1 && T <= kMaxT, "ar_agg_lt: 1 <= T <= 512");
    RuntimeCheck(H % (kVec * kNG) == 0, "ar_agg: H must be a multiple of 64");
    const int64_t n_vecs = H / kVec, sv = n_vecs / kNG;
    const int64_t chunk = (n_vecs + split - 1) / split;
    RuntimeCheck(sv <= 128 && chunk <= kThreads, "ar_agg: H too large");
    RuntimeCheck(
        T * sv * 16 <= kDataBytes && T * kStatStride * 4 <= kStatBytes && T * split * kNG * 4 <= kFlagBytes,
        "ar_agg: region too small");
    RuntimeCheck(kNG * split <= kThreads, "ar_agg: split too large");
    RuntimeCheck(
        prefix_out.stride(1) == 1 && bank.stride(2) == 1 && out.stride(1) == 1, "inner dim must be contiguous");
    RuntimeCheck(x.size(0) == T && x.stride(1) == 1, "x shape");
    if (mode == kAR)
      RuntimeCheck(x.size(1) == H && x.stride(0) == H, "AR input must be a contiguous [T, H]");
    else
      RuntimeCheck(x.size(1) == H / kNG && x.stride(0) == H / kNG, "AG input must be a contiguous [T, H/8]");
    const bool write_bank = flags & 4;
    RuntimeCheck(bank.size(1) > nvb || !write_bank, "bank has no free row for the snapshot");

    const auto stream = LaunchKernel::resolve_device(prefix_out.device());
    Params p{};
    if (mode == kAR) {
      void* inp = x.data_ptr();
      if (reg_inp_ptr != 0) {
        const int64_t bytes = T * H * 2;
        RuntimeCheck(bytes <= reg_inp_bytes, "registered buffer too small");
        RuntimeDeviceCheck(
            hipMemcpyAsync(reinterpret_cast<void*>(reg_inp_ptr), inp, bytes, hipMemcpyDeviceToDevice, stream));
        inp = reinterpret_cast<void*>(reg_inp_ptr);
      }
      p.dp = ca->get_buffer_RD(stream, inp);
      if (flags & 1) {
        p.prefix = static_cast<const bf16_t*>(a.data_ptr());
        p.stride_p = a.stride(0);
      }
    } else {
      p.y = static_cast<const bf16_t*>(x.data_ptr());
      p.add_b = static_cast<const bf16_t*>(b.data_ptr());
      p.stride_b = b.stride(0);
      if (flags & 2) {
        p.add_c = static_cast<const bf16_t*>(c.data_ptr());
        p.stride_c = c.stride(0);
      }
    }
    p.sg = ca->sg_;
    p.self_sg = ca->self_sg_;
    p.rank = ca->rank_;
    p.T = static_cast<int>(T);
    p.split = static_cast<int>(split);
    (void)parity;
    p.n_prod = static_cast<int>(T < kMaxProd ? T : kMaxProd);
    {
      const int64_t items = T * split;
      int64_t g = grid_cap > 0 ? grid_cap : kDefaultGrid;
      if (g > items) g = items;
      if (g < p.n_prod) g = p.n_prod;
      p.grid = static_cast<int>(g);
    }
    p.region_off = region_off;
    p.prefix_out = static_cast<bf16_t*>(prefix_out.data_ptr());
    p.stride_po = prefix_out.stride(0);
    p.bank = static_cast<bf16_t*>(bank.data_ptr());
    p.stride_bm = bank.stride(0);
    p.stride_bb = bank.stride(1);
    p.cw = static_cast<const float*>(cw.data_ptr());
    if (flags & 8) p.ow = static_cast<const bf16_t*>(ow.data_ptr());
    p.out = static_cast<bf16_t*>(out.data_ptr());
    p.stride_o = out.stride(0);
    if (flags & 16) {
      p.out8 = static_cast<uint8_t*>(out8.data_ptr());
      p.stride_o8 = out8.stride(0);
    }
    if (flags & 32) {
      p.stream = static_cast<bf16_t*>(stream_out.data_ptr());
      p.stride_s = stream_out.stride(0);
    }
    p.hidden = static_cast<int32_t>(H);
    p.n_vecs = static_cast<int32_t>(n_vecs);
    p.slice_vecs = static_cast<int32_t>(sv);
    p.chunk_vecs = static_cast<int32_t>(chunk);
    p.write_bank = write_bank ? 1 : 0;
    p.score_eps = static_cast<float>(score_eps);
    p.out_eps = static_cast<float>(out_eps);
    if (mode == kAR)
      dispatch<kAR>(p, nvb, stream);
    else
      dispatch<kAG>(p, nvb, stream);
    RuntimeDeviceCheck(hipGetLastError());
  }
};

}  // namespace k3_ar_agg_lt
}  // namespace sglang
