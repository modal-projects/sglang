// K3 ROCm (TP8): MoE [latent | shared] all-reduce fused with the latent
// RMSNorm (SGLANG_ROCM_K3_MOE_AR_NORM_FUSED).
//
// Replaces, per MoE layer at decode token counts,
//   aiter cross_device_reduce_{1,2}stage(buf)       buf = [T*L latent | T*H shared]
//   aiter rmsnorm (add_rmsnorm_quant_kernel)(latent) -> normed latent (bf16)
// with one launch in the ar_agg design (see ar_agg_hip.cuh): the reduction is
// spread over many producer blocks, the norm statistics ride the producer ->
// consumer signal, and consumers merge them in a fixed order.
//
// Work split. Rank r owns column slice r of every token: latent packs
// [r*sl, (r+1)*sl) and shared packs [r*sh, (r+1)*sh) (8 bf16 per pack). A
// token's slice is cut into units of kC packs (ul latent + us shared units; at
// K3 sizes kC = 28, ul = 2, us = 4).
//
// PRODUCER blocks (np <= 80, AITER start barrier on slot = block id; up to kKU
// units per block). Remote loads issued by one wave complete ~serially over
// xGMI (~1 us each), so loads are spread: kW groups of 256 threads per block
// work on different units, thread (q, j) of a group loads pack j of its unit
// from the peer at sum position q -- AITER's order: start at the rank owning
// the element in the flat 2-stage partition (rank 0 for the 1-stage kernel).
// LDS transpose, thread j sums in fp32 -> bf16 (bit-identical to AITER).
//   latent unit: stores its packs + its fp32 sum of squares in this rank's
//                private region (every remotely read datum written by another
//                producer sits on its own 128 B line: reading a line someone
//                else is still writing costs ~1-2 us extra);
//   shared unit: writes this rank's shared output slice (+ the region in full
//                mode).
// Then each unit flags every consumer block of its token on every rank.
// CONSUMER blocks (t, k) wait for the unit flags of token t: block (t, 0) (and
// every block in full mode) for all 8 x (ul + us), so when the kernel completes
// on rank r every peer has finished reading r's input (no end barrier, the
// input may be reused at once); the other latent blocks for the 8 x ul latent
// units only. k < lsplit own
// 1/lsplit of the latent columns: one remote load per thread, the 8 x ul
// statistics on the otherwise idle last wave, fixed-order sum -> rstd
// (identical on every rank), normalize. Full mode adds blocks copying the 7
// peer shared slices; scatter-only mode (the ag_agg tail reads add_b only in
// this rank's slice) leaves them out.
//
// Region / ordering: the private 1 MiB region of ar_agg / ag_agg and their
// launch parity. Every region write follows the start barrier of the same
// launch (every rank finished the previous launch on its stream). The region
// is uncached device memory: a producer waits for its stores to complete
// (vmcnt) before the barrier + flag store, no L2 writeback needed.
//
// Numerics: the all-reduced rows are bit-identical to AITER's. The norm is
// bf16((x * rstd) * w), rstd = rsqrtf(ss / L + eps) as in aiter's kernel, ss
// summed in another order: the normed latent matches up to rare 1-ulp flips.
#pragma once

#ifndef USE_ROCM
#error "moe_ar_norm_hip.cuh is ROCm only"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <custom_all_reduce.cuh>  // AITER: CustomAllreduce, RankData, Signal

#include <algorithm>
#include <cstdint>

namespace sglang {
namespace k3_moe_ar_norm {

constexpr int kNG = 8;
constexpr int kVec = 8;
constexpr int kThreads = 512;
constexpr int kW = kThreads / 256;  // unit groups per producer block
constexpr int kC = 28;              // packs per unit
constexpr int kKU = 6;              // units per producer block
constexpr int kMaxP = aiter::kMaxBlocks;  // producer blocks (start-barrier slots)
// same region layout as ar_agg_hip.cuh
constexpr int64_t kDataOff = 0, kDataPar = 256 << 10;
constexpr int64_t kStatOff = 512 << 10, kStatPar = 64 << 10;
constexpr int64_t kFlagOff = 768 << 10, kFlagPar = 64 << 10;
constexpr int kUP = 32;          // packs per unit slot (512 B)
constexpr int kStatStride = 32;  // fp32 per statistics slot (128 B)
constexpr int kMaxUL = 4;        // latent units per token (statistics layout)

using vec_t = device::AlignedVector<bf16_t, kVec>;
using OP = opus::vector_t<opus::bf16_t, kVec>;
using OA = opus::vector_t<opus::fp32_t, kVec>;

struct Params {
  aiter::RankData* dp;  // peer addresses of the registered [T*L | T*H] inputs
  aiter::RankSignals sg;
  aiter::Signal* self_sg;
  int rank, T, np, split, lsplit, parity;
  int ul, nu;  // latent units / units per token
  int64_t region_off;
  int32_t part;            // AITER 2-stage partition (packs); 0 = 1-stage order
  int32_t lv, hv, sl, sh;  // packs per row (latent, shared), per rank slice
  int32_t lchunk, hchunk;  // packs per consumer block
  int32_t full_shared;
  const bf16_t* __restrict__ w;  // [L]
  bf16_t* __restrict__ out;      // [T, L] normed latent
  bf16_t* __restrict__ shared;   // [T, H]
  int64_t stride_o, stride_s;
  double eps;
  int32_t l;
  int64_t* ts;  // optional phase timestamps [grid][4] (instrumentation)
};

#define K3_TS(i)                                                                          \
  do {                                                                                    \
    if (p.ts != nullptr && threadIdx.x == 0) p.ts[blockIdx.x * 4 + (i)] = wall_clock64(); \
  } while (0)

__device__ __forceinline__ char* region(const Params& p, int q) {
  return reinterpret_cast<char*>(aiter::get_tmp_buf<char>(p.sg.signals[q])) + p.region_off;
}

__global__ void __launch_bounds__(kThreads) moe_ar_norm_kernel(const Params __grid_constant__ p) {
  using namespace device;
  const int64_t dpar = kDataOff + p.parity * kDataPar;
  const int64_t spar = kStatOff + p.parity * kStatPar;
  const int64_t fpar = kFlagOff + p.parity * kFlagPar;
  const int nflag = kNG * p.nu;
  K3_TS(0);

  // ======================= producers =======================
  if (static_cast<int>(blockIdx.x) < p.np) {
    __shared__ OP lds[kKU][kNG][kC];
    const int b = blockIdx.x;
    const int nunits = p.T * p.nu;
    char* mine = region(p, p.rank);
    // AITER start barrier on slot b
    const uint32_t flag = p.self_sg->_flag[b] + 1;
    if (threadIdx.x < kNG)
      __scoped_atomic_store_n(
          &p.sg.signals[threadIdx.x]->start[b][p.rank], flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    if (threadIdx.x < kNG) {
      while (__scoped_atomic_load_n(&p.self_sg->start[b][threadIdx.x], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) <
             flag)
        ;
    }
    __syncthreads();
    if (threadIdx.x == 0) p.self_sg->_flag[b] = flag;
    K3_TS(1);

    // loads: thread (g, q, j) fetches pack j of units kk = g, g + kW, ... of
    // this block from the peer at sum position q
    {
      const int g = threadIdx.x / 256, lt = threadIdx.x % 256;
      const int q = lt / kC, j = lt - q * kC;
      if (q < kNG) {
        OP xin[kKU];
#pragma unroll
        for (int kk = 0; kk < kKU; ++kk) {
          const int u = b + kk * p.np;
          if (kk % kW == g && u < nunits) {
            const int t = u / p.nu, w = u - t * p.nu;
            const bool lat = w < p.ul;
            const int i = (lat ? w : w - p.ul) * kC + j;
            const int flat = lat ? t * p.lv + p.rank * p.sl + i : p.T * p.lv + t * p.hv + p.rank * p.sh + i;
            int owner = 0;
            if (p.part > 0) {
              owner = flat / p.part;
              if (owner > kNG - 1) owner = kNG - 1;
            }
            xin[kk] = reinterpret_cast<const OP*>(p.dp->ptrs[(owner + q) & (kNG - 1)])[flat];
          }
        }
#pragma unroll
        for (int kk = 0; kk < kKU; ++kk)
          if (kk % kW == g && b + kk * p.np < nunits) lds[kk][q][j] = xin[kk];
      }
    }
    __syncthreads();
    K3_TS(2);
    // sums: thread (kk, j) (32-lane groups) reduces pack j of unit kk in AITER
    // order; the unit's sum of squares reduces over its group with shuffles
    {
      const int kk = threadIdx.x / 32, j = threadIdx.x % 32;
      const int u = b + kk * p.np;
      const bool live = kk < kKU && u < nunits;
      const int t = live ? u / p.nu : 0, w = live ? u - t * p.nu : 0;
      const bool lat = w < p.ul;
      float ss = 0.f;
      if (live && j < kC) {
        const int i = (lat ? w : w - p.ul) * kC + j;
        OA acc = aiter::upcast(lds[kk][0][j]);
#pragma unroll
        for (int qq = 1; qq < kNG; ++qq) {
          const OP x = lds[kk][qq][j];
#pragma unroll
          for (int e = 0; e < kVec; ++e) acc[e] += aiter::upcast_s(x[e]);
        }
        const OP ar = aiter::downcast<OP>(acc);
        const vec_t row = *reinterpret_cast<const vec_t*>(&ar);
        if (lat) {
          row.store(mine + dpar, static_cast<int64_t>(t * p.nu + w) * kUP + j);
#pragma unroll
          for (int e = 0; e < kVec; ++e) {
            const float x = cast<fp32_t>(row[e]);
            ss += x * x;
          }
        } else {
          row.store(p.shared + t * p.stride_s, p.rank * p.sh + i);
          if (p.full_shared) row.store(mine + dpar, static_cast<int64_t>(t * p.nu + w) * kUP + j);
        }
      }
      if (live && lat) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
        if (j == 0) reinterpret_cast<float*>(mine + spar)[(t * kMaxUL + w) * kStatStride] = ss;
      }
    }
    // the region is uncached memory: once this thread's stores completed they
    // are visible to the peers (no L2 writeback / release fence needed)
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    // flags: unit kk -> every consumer block of its token on every rank
    for (int y = threadIdx.x; y < kKU * kNG * p.split; y += kThreads) {
      const int per = kNG * p.split;
      const int kk = y / per, x = y - kk * per;
      const int u = b + kk * p.np;
      if (u < nunits) {
        const int t = u / p.nu, w = u - t * p.nu;
        const int qq = x / p.split, ks = x - qq * p.split;
        // scatter-only: a shared unit is awaited by block (t, 0) alone (the
        // input-reuse guarantee); latent consumers k > 0 wait for latent units
        if (!p.full_shared && w >= p.ul && ks != 0) continue;
        uint32_t* f =
            reinterpret_cast<uint32_t*>(region(p, qq) + fpar) + (t * p.split + ks) * nflag + p.rank * p.nu + w;
        __scoped_atomic_store_n(f, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
      }
    }
    K3_TS(3);
    return;
  }

  // ======================= consumers =======================
  const int cb = blockIdx.x - p.np;
  const int t = cb / p.split;
  const int k = cb - t * p.split;
  const bool is_lat = k < p.lsplit;
  int v;
  vec_t wv;
  if (is_lat) {
    v = k * p.lchunk + threadIdx.x;
    if (threadIdx.x >= p.lchunk || v >= p.lv) v = -1;
    if (v >= 0) wv.load(p.w, v);
  } else {
    v = (k - p.lsplit) * p.hchunk + threadIdx.x;
    if (threadIdx.x >= p.hchunk || v >= p.hv || v / p.sh == p.rank) v = -1;
  }
  {
    uint32_t* f = reinterpret_cast<uint32_t*>(region(p, p.rank) + fpar) + cb * nflag;
    // flags this block waits for: all 8 x nu, or (scatter-only, k > 0) the
    // 8 x ul latent ones
    const bool all = p.full_shared || k == 0;
    const int nwait = all ? nflag : kNG * p.ul;
    if (threadIdx.x < nwait) {
      const int fi = all ? threadIdx.x : (threadIdx.x / p.ul) * p.nu + threadIdx.x % p.ul;
      // relaxed polling, one acquire fence after (see ar_agg_hip.cuh)
      while (__scoped_atomic_load_n(f + fi, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == 0u)
        ;
      __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
      __scoped_atomic_store_n(f + fi, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    }
    __syncthreads();
  }
  K3_TS(1);
  if (is_lat) {
    __shared__ float st[kNG * kMaxUL];
    __shared__ float rstd_s;
    vec_t x;
    if (v >= 0) {
      const int q = v / p.sl, i = v - q * p.sl;
      x.load(region(p, q) + dpar, static_cast<int64_t>(t * p.nu + i / kC) * kUP + i % kC);
    }
    // the 8 x ul statistics on the last wave (idle in the column loads: a
    // second remote load in the same wave would complete after the first)
    const int sx = threadIdx.x - (kThreads - 64);
    const int nst = kNG * p.ul;
    if (sx >= 0 && sx < nst) {
      const int q = sx / p.ul, w = sx - q * p.ul;
      st[sx] = reinterpret_cast<const float*>(region(p, q) + spar)[(t * kMaxUL + w) * kStatStride];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float ss = 0.f;
      for (int i = 0; i < nst; ++i) ss += st[i];
      // aiter add_rmsnorm_quant: rsqrtf(sum / n + epsilon) (double epsilon)
      rstd_s = rsqrtf(ss / p.l + p.eps);
    }
    __syncthreads();
    K3_TS(2);
    if (v < 0) return;
    const float r = rstd_s;
    vec_t o;
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      const float y = cast<fp32_t>(x[e]) * r;
      o[e] = cast<bf16_t>(y * cast<fp32_t>(wv[e]));
    }
    o.store(p.out + t * p.stride_o, v);
    K3_TS(3);
  } else {
    if (v < 0) return;
    const int q = v / p.sh, i = v - q * p.sh;
    vec_t x;
    x.load(region(p, q) + dpar, static_cast<int64_t>(t * p.nu + p.ul + i / kC) * kUP + i % kC);
    x.store(p.shared + t * p.stride_s, v);
  }
}

struct MoeArNorm {
  // x: this rank's flat TP-partial [T*L + T*H] (bf16); w: [L]; out: [T, L];
  // shared: [T, H]. part: AITER 2-stage partition in packs (0 = 1-stage order).
  static void
  run(int64_t comm,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView shared,
      double eps,
      int64_t part,
      int64_t full_shared,
      int64_t lsplit,
      int64_t hsplit,
      int64_t np,
      int64_t reg_inp_ptr,
      int64_t reg_inp_bytes,
      int64_t region_off,
      int64_t parity,
      int64_t ts_ptr) {
    using namespace host;
    auto* ca = reinterpret_cast<aiter::CustomAllreduce*>(comm);
    RuntimeCheck(ca->world_size_ == kNG, "moe_ar_norm is TP8 only");
    const int64_t T = out.size(0);
    const int64_t L = out.size(1);
    const int64_t H = shared.size(1);
    RuntimeCheck(T >= 1 && shared.size(0) == T, "shared shape");
    RuntimeCheck(L % (kVec * kNG * kC) == 0 && H % (kVec * kNG * kC) == 0, "L, H must be multiples of 8*8*28");
    const int64_t lv = L / kVec, hv = H / kVec, sl = lv / kNG, sh = hv / kNG;
    const int64_t ul = sl / kC, nu = ul + sh / kC;
    RuntimeCheck(ul <= kMaxUL && kNG * ul <= 64 && kNG * nu <= kThreads, "moe_ar_norm: slices too wide");
    RuntimeCheck(x.numel() == T * (L + H) && x.is_contiguous(), "x must be a contiguous [T*(L+H)]");
    RuntimeCheck(out.stride(1) == 1 && shared.stride(1) == 1 && w.is_contiguous() && w.numel() == L, "strides");
    RuntimeCheck(T * nu * kUP * 16 <= kDataPar && T * kMaxUL * kStatStride * 4 <= kStatPar,
                 "moe_ar_norm: region too small");
    const int64_t lchunk = (lv + lsplit - 1) / lsplit;
    const int64_t hs = full_shared ? hsplit : 0;
    const int64_t hchunk = hs > 0 ? (hv + hs - 1) / hs : 1;
    RuntimeCheck(lsplit >= 1 && lchunk <= kThreads - 64 && hchunk <= kThreads, "moe_ar_norm: split too small");
    const int64_t split = lsplit + hs;
    RuntimeCheck(T * split * kNG * nu * 4 <= kFlagPar, "moe_ar_norm: too many flags");
    np = std::min<int64_t>(std::min<int64_t>(np, kMaxP), T * nu);
    RuntimeCheck(np >= 1 && np * kKU >= T * nu, "moe_ar_norm: too many units for the producer grid");

    const auto stream = LaunchKernel::resolve_device(out.device());
    void* inp = x.data_ptr();
    if (reg_inp_ptr != 0) {
      const int64_t bytes = T * (L + H) * 2;
      RuntimeCheck(bytes <= reg_inp_bytes, "registered buffer too small");
      RuntimeDeviceCheck(
          hipMemcpyAsync(reinterpret_cast<void*>(reg_inp_ptr), inp, bytes, hipMemcpyDeviceToDevice, stream));
      inp = reinterpret_cast<void*>(reg_inp_ptr);
    }
    Params p{};
    p.dp = ca->get_buffer_RD(stream, inp);
    p.sg = ca->sg_;
    p.self_sg = ca->self_sg_;
    p.rank = ca->rank_;
    p.T = static_cast<int>(T);
    p.np = static_cast<int>(np);
    p.split = static_cast<int>(split);
    p.lsplit = static_cast<int>(lsplit);
    p.parity = static_cast<int>(parity & 1);
    p.ul = static_cast<int>(ul);
    p.nu = static_cast<int>(nu);
    p.region_off = region_off;
    p.part = static_cast<int32_t>(part);
    p.lv = static_cast<int32_t>(lv);
    p.hv = static_cast<int32_t>(hv);
    p.sl = static_cast<int32_t>(sl);
    p.sh = static_cast<int32_t>(sh);
    p.lchunk = static_cast<int32_t>(lchunk);
    p.hchunk = static_cast<int32_t>(hchunk);
    p.full_shared = full_shared ? 1 : 0;
    p.w = static_cast<const bf16_t*>(w.data_ptr());
    p.out = static_cast<bf16_t*>(out.data_ptr());
    p.shared = static_cast<bf16_t*>(shared.data_ptr());
    p.stride_o = out.stride(0);
    p.stride_s = shared.stride(0);
    p.eps = eps;
    p.l = static_cast<int32_t>(L);
    p.ts = reinterpret_cast<int64_t*>(ts_ptr);
    hipLaunchKernelGGL(moe_ar_norm_kernel, dim3(np + T * split), dim3(kThreads), 0, stream, p);
    RuntimeDeviceCheck(hipGetLastError());
  }
};

}  // namespace k3_moe_ar_norm
}  // namespace sglang
