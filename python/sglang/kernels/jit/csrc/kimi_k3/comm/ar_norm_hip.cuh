// K3 ROCm TP8: the MoE [latent | shared] all-reduce with the latent RMSNorm
// in its epilogue (the ROCm counterpart of k3_ar_fusion.all_reduce_norm).
//
// buf = [T, L] latent then [T, Hs] shared, flat. out = AR(buf) bit-identical
// to AITER's cross_device_reduce_2stage: every element is summed in the order
// AITER's owner rank uses (start at owner o = flat / part, then o+1, ...).
// Work is token-aligned instead: rank r reduces columns [r*S, (r+1)*S) of the
// token's logical (L + Hs)/8-pack row, one block per tpb tokens, and adds its
// latent columns' sum of squares (of the bf16 sums) to a per-(token, rank)
// statistic. After AITER's end barrier the same block gathers its tokens'
// rows from the 8 tmp buffers, sums the 8 statistics in rank order (identical
// on every rank) and writes out and the normed latent. The norm matches the
// standalone RMSNorm up to rare 1-ulp bf16 flips (fp32 sum order).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <custom_all_reduce.cuh>  // AITER: CustomAllreduce, RankData, Signal, start/end_sync

namespace sglang {
namespace k3_ar_norm {

constexpr int kNG = 8;
constexpr int kVec = 8;
constexpr int kThreads = 512;
constexpr int kMaxTPB = 3;
using vec_t = device::AlignedVector<bf16_t, kVec>;
using OP = opus::vector_t<opus::bf16_t, kVec>;
using OA = opus::vector_t<opus::fp32_t, kVec>;

struct Params {
  aiter::RankData* dp;
  aiter::RankSignals sg;
  aiter::Signal* self_sg;
  int rank, T, tpb;
  int L8, H8, S;     // latent / shared packs per token, packs per rank slice
  int64_t part;      // AITER's per-rank partition, packs
  int64_t stat_off;  // bytes into the tmp buffer
  bf16_t* out;
  bf16_t* out_norm;
  const bf16_t* w;
  float eps, inv_l;
};

__device__ __forceinline__ int64_t flat_pack(const Params& p, int t, int j) {
  return j < p.L8 ? static_cast<int64_t>(t) * p.L8 + j
                  : static_cast<int64_t>(p.T) * p.L8 + static_cast<int64_t>(t) * p.H8 + (j - p.L8);
}

__global__ void __launch_bounds__(kThreads, 1) ar_norm_kernel(const Params p) {
  __shared__ float red[kThreads];
  __shared__ float red2[kMaxTPB * 8];
  __shared__ float s_rstd[kMaxTPB];
  const int t0 = blockIdx.x * p.tpb;
  const int nt = min(p.tpb, p.T - t0);
  OP* tmp_self = aiter::get_tmp_buf<OP>(p.sg.signals[p.rank]);
  float* stat_self = reinterpret_cast<float*>(reinterpret_cast<char*>(tmp_self) + p.stat_off);

  aiter::start_sync<kNG>(p.sg, p.self_sg, p.rank);

  // ---- stage 1: reduce this rank's slice of each token (AITER order) ----
  const int i = threadIdx.x;
  float sq = 0.f;
  if (i < nt * p.S) {
    const int s = i / p.S, jj = i - s * p.S, t = t0 + s, j = p.rank * p.S + jj;
    const int64_t f = flat_pack(p, t, j);
    const int o = static_cast<int>(f / p.part);
    OA acc = aiter::upcast(reinterpret_cast<const OP*>(p.dp->ptrs[o])[f]);
#pragma unroll
    for (int q = 1; q < kNG; ++q) {
      const OP v = reinterpret_cast<const OP*>(p.dp->ptrs[(o + q) & (kNG - 1)])[f];
#pragma unroll
      for (int e = 0; e < kVec; ++e)
        acc[e] += aiter::upcast_s(v[e]);
    }
    const OP r = aiter::downcast<OP>(acc);
    tmp_self[static_cast<int64_t>(t) * p.S + jj] = r;
    if (j < p.L8) {
      const vec_t rv = *reinterpret_cast<const vec_t*>(&r);
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        const float x = static_cast<float>(rv[e]);
        sq += x * x;
      }
    }
  }
  red[i] = sq;
  __syncthreads();
  const int seg = p.S / 8;  // fixed-order two-level sum per token
  if (i < nt * 8) {
    const int s = i / 8, k = i - s * 8;
    float a = 0.f;
    for (int m = 0; m < seg; ++m)
      a += red[s * p.S + k * seg + m];
    red2[i] = a;
  }
  __syncthreads();
  if (i < nt) {
    float a = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k)
      a += red2[i * 8 + k];
    stat_self[(t0 + i) * kNG + p.rank] = a;
  }

  aiter::end_sync<kNG>(p.sg, p.self_sg, p.rank);

  // ---- stage 2: gather the tokens' rows, normalize the latent ----
  if (i < nt) {
    float a = 0.f;
#pragma unroll
    for (int q = 0; q < kNG; ++q) {
      const float* st = reinterpret_cast<const float*>(
          reinterpret_cast<const char*>(aiter::get_tmp_buf<OP>(p.sg.signals[q])) + p.stat_off);
      a += st[(t0 + i) * kNG + q];
    }
    s_rstd[i] = rsqrtf(a * p.inv_l + p.eps);
  }
  __syncthreads();
  // all remote loads first (one latency, not one per pass), then the stores
  const int row = p.L8 + p.H8, n = nt * row;
  constexpr int kMaxPass = (kMaxTPB * 1344 + kThreads - 1) / kThreads;
  OP r[kMaxPass];
#pragma unroll
  for (int m = 0; m < kMaxPass; ++m) {
    const int k = i + m * kThreads;
    if (k < n) {
      const int s = k / row, j = k - s * row, q = j / p.S;
      r[m] = aiter::get_tmp_buf<OP>(p.sg.signals[q])[static_cast<int64_t>(t0 + s) * p.S + (j - q * p.S)];
    }
  }
#pragma unroll
  for (int m = 0; m < kMaxPass; ++m) {
    const int k = i + m * kThreads;
    if (k < n) {
      const int s = k / row, j = k - s * row, t = t0 + s;
      reinterpret_cast<OP*>(p.out)[flat_pack(p, t, j)] = r[m];
      if (j < p.L8) {
        const vec_t rv = *reinterpret_cast<const vec_t*>(&r[m]);
        vec_t wv, nv;
        wv.load(p.w, j);
        const float rs = s_rstd[s];
#pragma unroll
        for (int e = 0; e < kVec; ++e)
          nv[e] = static_cast<bf16_t>(static_cast<float>(rv[e]) * rs * static_cast<float>(wv[e]));
        nv.store(p.out_norm + static_cast<int64_t>(t) * p.L8 * kVec, j);
      }
    }
  }
}

struct ArNorm {
  static void
  run(int64_t comm,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView out_norm,
      const tvm::ffi::TensorView w,
      double eps,
      int64_t tpb,
      int64_t reg_inp_ptr,
      int64_t reg_inp_bytes) {
    using namespace host;
    auto* ca = reinterpret_cast<aiter::CustomAllreduce*>(comm);
    RuntimeCheck(ca->world_size_ == kNG, "ar_norm is TP8 only");
    const int64_t T = out_norm.size(0), L = out_norm.size(1);
    const int64_t numel = x.numel();
    RuntimeCheck(T >= 1 && numel % T == 0 && out.numel() == numel, "ar_norm: shapes");
    const int64_t W = numel / T, Hs = W - L;
    RuntimeCheck(L % kVec == 0 && Hs > 0 && Hs % kVec == 0 && W % (kVec * kNG) == 0, "ar_norm: widths");
    const int64_t S = W / kVec / kNG;
    RuntimeCheck(S % 8 == 0 && W / kVec <= 1344 && tpb >= 1 && tpb <= kMaxTPB && tpb * S <= kThreads, "ar_norm: tpb");
    const int64_t grid = (T + tpb - 1) / tpb;
    RuntimeCheck(grid <= aiter::kMaxBlocks, "ar_norm: too many tokens for tpb");
    const int64_t data_bytes = T * S * 16;
    const int64_t stat_off = (data_bytes + 255) / 256 * 256;
    RuntimeCheck(stat_off + T * kNG * 4 <= (1 << 20) * 4, "ar_norm: tmp too small");
    RuntimeCheck(w.numel() == L && out_norm.stride(0) == L, "ar_norm: weight / out_norm");

    const auto stream = LaunchKernel::resolve_device(out.device());
    void* inp = x.data_ptr();
    if (reg_inp_ptr != 0) {
      RuntimeCheck(numel * 2 <= reg_inp_bytes, "registered buffer too small");
      RuntimeDeviceCheck(
          hipMemcpyAsync(reinterpret_cast<void*>(reg_inp_ptr), inp, numel * 2, hipMemcpyDeviceToDevice, stream));
      inp = reinterpret_cast<void*>(reg_inp_ptr);
    }
    Params p{};
    p.dp = ca->get_buffer_RD(stream, inp);
    p.sg = ca->sg_;
    p.self_sg = ca->self_sg_;
    p.rank = ca->rank_;
    p.T = static_cast<int>(T);
    p.tpb = static_cast<int>(tpb);
    p.L8 = static_cast<int>(L / kVec);
    p.H8 = static_cast<int>(Hs / kVec);
    p.S = static_cast<int>(S);
    p.part = numel / kVec / kNG;
    p.stat_off = stat_off;
    p.out = static_cast<bf16_t*>(out.data_ptr());
    p.out_norm = static_cast<bf16_t*>(out_norm.data_ptr());
    p.w = static_cast<const bf16_t*>(w.data_ptr());
    p.eps = static_cast<float>(eps);
    p.inv_l = 1.f / static_cast<float>(L);
    hipLaunchKernelGGL(ar_norm_kernel, dim3(grid), dim3(kThreads), 0, stream, p);
  }
};

}  // namespace k3_ar_norm
}  // namespace sglang
