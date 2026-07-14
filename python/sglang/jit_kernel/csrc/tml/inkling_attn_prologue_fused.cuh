// Fused target-verify attention prologue for Inkling: after the qkvr projection,
// ONE kernel does {k_sconv + v_sconv (verify-mode causal_conv1d) + both
// save_intermediate_conv_windows + per-head q/k RMSNorm + the KV-cache store}.
// Replaces 2x causal_conv1d + 2x save_windows (triton) + fused qk-norm + the
// backend's set_kv_buffer scatter (attention then runs with
// save_kv_cache=False). rel_logits_proj overlaps on the alt stream.
//
// Layout: ONE BLOCK PER TOKEN; one 16B vec (8 channels) per thread. Lane
// roles by vec index: [0, Dq/8) q-norm lanes, then Dkv/8 k lanes, then Dkv/8
// v lanes. head_dim=128 -> a head is 16 CONTIGUOUS lanes, reduced with
// width-16 warp shuffles (Dq/8 and Dkv/8 are multiples of 16, so head groups
// never straddle warps). The convs read cross-token taps directly from the
// (strided) qkvr tensor; per-seq prefixes from the read-only conv caches; the
// per-position windows go to the intermediate buffers exactly like
// save_intermediate_conv_windows (raw copies, no gating). PAD sequences
// (cache_indices == -1) skip prefix/window IO but still emit outputs and the
// KV store (mirroring the unfused path, which stores pad rows too).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <type_traits>

namespace {

constexpr int kPadSlot = -1;
constexpr uint32_t kVecElems = 8;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kHeadLanes = kHeadDim / kVecElems;  // 16

// Per-head (16-lane) sum-reduce -> rsqrt(mean(ss)+eps), broadcast to all lanes
// of the group. A head is 16 CONTIGUOUS lanes on a 16-aligned boundary (Dq/8
// and Dkv/8 are multiples of 16), and the q / k / v roles also start on
// 16-aligned boundaries -- so each 16-lane half-warp is a single (role, head)
// group whose lanes all reach (or all skip) this reduction. The mask names
// EXACTLY that half-warp (not the whole warp), so it stays valid even when the
// k and v roles split a warp (odd num_tp_kv_heads, e.g. 1 KV head/rank) or the
// q/k boundary falls mid-warp -- an xor butterfly then leaves every lane in the
// group with the full sum (no cross-role shuffle, no exited-lane in the mask).
__device__ __forceinline__ float head_rmsnorm_inv(float ss, float eps) {
  const unsigned hmask = 0xFFFFu << (threadIdx.x & 16u);  // this thread's 16-lane group
#pragma unroll
  for (int off = 8; off > 0; off >>= 1) ss += __shfl_xor_sync(hmask, ss, off, 16);
  return rsqrtf(ss / static_cast<float>(kHeadDim) + eps);
}

struct AttnPrologueParams {
  const void* __restrict__ qkvr;  // [T, row_stride] packed projection output
  // sconv (verify) per K/V path
  const void* __restrict__ k_cache;  // [pool, W-1, Dkv]
  const void* __restrict__ v_cache;
  const void* __restrict__ cache_indices;  // int32 [B] (PAD == -1)
  const void* __restrict__ cache_mask;     // bool  [B]
  const void* __restrict__ k_weight;       // [Dkv, W]
  const void* __restrict__ v_weight;
  void* __restrict__ k_inter;  // [max_bs, q, W-1, Dkv]
  void* __restrict__ v_inter;
  // norms
  const void* __restrict__ q_gamma;  // [head_dim]
  const void* __restrict__ k_gamma;  // [head_dim]
  float eps;
  // outputs
  void* __restrict__ q_out;  // [T, Dq]
  void* __restrict__ k_out;  // [T, Dkv]
  void* __restrict__ v_out;  // [T, Dkv]
  const void* __restrict__ loc;  // int64 [T] KV slots
  void* __restrict__ k_buf;      // [slots, Hkv, head_dim]
  void* __restrict__ v_buf;
  int64_t qkvr_stride_t;
  int64_t q_off;  // elem offsets of the q/k/v slices within a qkvr row
  int64_t k_off;
  int64_t v_off;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t weight_stride_d;
  int64_t inter_stride_b;
  int64_t inter_stride_t;
  int64_t inter_stride_w;
  int64_t kv_buf_stride;  // elems per KV slot row (= Hkv * head_dim)
  uint32_t T;
  uint32_t q;  // draft_token_num
  uint32_t dq;
  uint32_t dkv;
};

template <typename DType, int W, bool USE_SILU, bool USE_RESIDUAL, bool DO_STORE>
__global__ __launch_bounds__(1024, 1) void inkling_attn_prologue_kernel(
    const __grid_constant__ AttnPrologueParams p) {
  static_assert(std::is_same_v<DType, __nv_bfloat16>);
  constexpr int W1 = W - 1;
  const uint32_t t = blockIdx.x;
  const uint32_t seq = t / p.q;
  const int bos = static_cast<int>(seq * p.q);
  const uint32_t tq = t - seq * p.q;
  const uint32_t nq = p.dq / kVecElems;
  const uint32_t nkv = p.dkv / kVecElems;
  const uint32_t vi = threadIdx.x;
  const auto* base = static_cast<const __nv_bfloat16*>(p.qkvr);
  const int64_t row = static_cast<int64_t>(t) * p.qkvr_stride_t;

  const int ci = static_cast<const int32_t*>(p.cache_indices)[seq];
  const bool valid = ci != kPadSlot;
  const int slot_id = valid ? ci : 0;
  const float cm =
      (valid && static_cast<const bool*>(p.cache_mask)[seq]) ? 1.0f : 0.0f;

  if (vi < nq) {
    // ---------------- q path: per-head RMSNorm only ----------------
    const uint32_t c = vi * kVecElems;
    const uint4 raw = *reinterpret_cast<const uint4*>(base + row + p.q_off + c);
    float x[kVecElems];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      x[j] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&raw)[j]);
      ss += x[j] * x[j];
    }
    const float inv = head_rmsnorm_inv(ss, p.eps);
    const auto* gq = static_cast<const __nv_bfloat16*>(p.q_gamma) + (c % kHeadDim);
    __nv_bfloat162 o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[j] = __floats2bfloat162_rn(x[2 * j] * inv * __bfloat162float(gq[2 * j]),
                                   x[2 * j + 1] * inv * __bfloat162float(gq[2 * j + 1]));
    }
    *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(p.q_out) +
                              static_cast<int64_t>(t) * p.dq + c) =
        *reinterpret_cast<const uint4*>(o);
    return;
  }
  if (vi >= nq + 2 * nkv) return;

  // ---------------- k / v paths: conv + save_windows (+ k norm) + store ----
  const bool is_k = vi < nq + nkv;
  const uint32_t ch = (is_k ? vi - nq : vi - nq - nkv) * kVecElems;
  const int64_t x_off = is_k ? p.k_off : p.v_off;
  const auto* cp = static_cast<const __nv_bfloat16*>(is_k ? p.k_cache : p.v_cache);
  const auto* wp = static_cast<const __nv_bfloat16*>(is_k ? p.k_weight : p.v_weight);
  auto* ip = static_cast<__nv_bfloat16*>(is_k ? p.k_inter : p.v_inter);
  const int64_t cache_base = static_cast<int64_t>(slot_id) * p.cache_stride_slot + ch;

  uint4 pref[W1];
  __nv_bfloat16 wt[kVecElems][W];
#pragma unroll
  for (int w = 0; w < W1; ++w) {
    pref[w] = *reinterpret_cast<const uint4*>(&cp[cache_base + w * p.cache_stride_w]);
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t wrow = static_cast<int64_t>(ch + j) * p.weight_stride_d;
    if constexpr (W == 4) {
      if (p.weight_stride_d == W) {
        *reinterpret_cast<uint2*>(wt[j]) = *reinterpret_cast<const uint2*>(wp + wrow);
        continue;
      }
    }
#pragma unroll
    for (int w = 0; w < W; ++w) wt[j][w] = wp[wrow + w];
  }

  // In-seq neighbor rows (pre-conv x straight from qkvr) + own row.
  const uint4 xcur = *reinterpret_cast<const uint4*>(base + row + x_off + ch);
  uint4 xn[W1];
#pragma unroll
  for (int j = 1; j <= W1; ++j) {
    const int n = static_cast<int>(t) - j;
    if (n >= bos) {
      xn[j - 1] = *reinterpret_cast<const uint4*>(
          base + static_cast<int64_t>(n) * p.qkvr_stride_t + x_off + ch);
    }
  }

  float y[kVecElems];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float xj = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&xcur)[j]);
    float acc = 0.0f;
#pragma unroll
    for (int iw = 0; iw < W1; ++iw) {
      const int shifted = static_cast<int>(t) - W1 + iw;
      float tap = 0.0f;
      if (shifted >= bos) {
        tap = __bfloat162float(
            reinterpret_cast<const __nv_bfloat16*>(&xn[W1 - 1 - iw])[j]);
      } else {
        const int prefix_pos = shifted - bos + W1;
        if (prefix_pos >= 0) {
          tap = cm * __bfloat162float(
                         reinterpret_cast<const __nv_bfloat16*>(&pref[prefix_pos])[j]);
        }
      }
      acc += tap * __bfloat162float(wt[j][iw]);
    }
    acc += xj * __bfloat162float(wt[j][W1]);
    if constexpr (USE_SILU) acc = __fdividef(acc, 1.0f + __expf(-acc));
    if constexpr (USE_RESIDUAL) acc += xj;
    y[j] = acc;
  }

  if (valid) {  // save_intermediate_conv_windows (raw copies)
    auto* op = ip + static_cast<int64_t>(seq) * p.inter_stride_b +
               static_cast<int64_t>(tq) * p.inter_stride_t + ch;
#pragma unroll
    for (int w = 0; w < W1; ++w) {
      const int position = static_cast<int>(tq) + 1 + w;
      uint4 val;
      if (position < W1) {
        val = pref[position];
      } else {
        const int g = bos + position - W1;
        val = (g == static_cast<int>(t)) ? xcur : xn[t - g - 1];
      }
      *reinterpret_cast<uint4*>(op + w * p.inter_stride_w) = val;
    }
  }

  __nv_bfloat162 o[4];
  if (is_k) {
    // per-head RMSNorm on the conv output (16-lane groups). Round to bf16
    // FIRST: the unfused pipeline writes the conv output to memory as bf16
    // before the norm kernel reads it back.
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      y[j] = __bfloat162float(__float2bfloat16_rn(y[j]));
      ss += y[j] * y[j];
    }
    const float inv = head_rmsnorm_inv(ss, p.eps);
    const auto* gk = static_cast<const __nv_bfloat16*>(p.k_gamma) + (ch % kHeadDim);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[j] = __floats2bfloat162_rn(y[2 * j] * inv * __bfloat162float(gk[2 * j]),
                                   y[2 * j + 1] * inv * __bfloat162float(gk[2 * j + 1]));
    }
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __floats2bfloat162_rn(y[2 * j], y[2 * j + 1]);
  }
  const uint4 ov = *reinterpret_cast<const uint4*>(o);
  auto* out = static_cast<__nv_bfloat16*>(is_k ? p.k_out : p.v_out);
  *reinterpret_cast<uint4*>(out + static_cast<int64_t>(t) * p.dkv + ch) = ov;
  // Fused KV store (DO_STORE). Only used for full-attention layers writing a
  // plain bf16 [slots, Hkv, head_dim] pool indexed directly by out_cache_loc;
  // SWA/local layers keep the backend store (swa_out_cache_loc + its own pool),
  // so the caller passes DO_STORE=false and save_kv_cache=True there.
  if (DO_STORE) {
    const int64_t kv_slot = static_cast<const int64_t*>(p.loc)[t];
    if (kv_slot >= 0) {  // SWA full->swa translation can yield -1 sentinels
      auto* buf = static_cast<__nv_bfloat16*>(is_k ? p.k_buf : p.v_buf);
      *reinterpret_cast<uint4*>(buf + kv_slot * p.kv_buf_stride + ch) = ov;
    }
  }
}

template <typename DType, int W, bool USE_SILU, bool USE_RESIDUAL>
struct AttnPrologueKernel {
  static void
  run(tvm::ffi::TensorView qkvr,
      tvm::ffi::TensorView k_cache,
      tvm::ffi::TensorView v_cache,
      tvm::ffi::TensorView cache_indices,
      tvm::ffi::TensorView cache_mask,
      tvm::ffi::TensorView k_weight,
      tvm::ffi::TensorView v_weight,
      tvm::ffi::TensorView k_inter,
      tvm::ffi::TensorView v_inter,
      tvm::ffi::TensorView q_gamma,
      tvm::ffi::TensorView k_gamma,
      double eps,
      tvm::ffi::TensorView q_out,
      tvm::ffi::TensorView k_out,
      tvm::ffi::TensorView v_out,
      tvm::ffi::TensorView loc,
      tvm::ffi::TensorView k_buf,
      tvm::ffi::TensorView v_buf,
      int64_t q_off,
      int64_t k_off,
      int64_t v_off,
      int64_t q_num,
      int64_t do_store) {
    using namespace host;
    const uint32_t T = static_cast<uint32_t>(qkvr.size(0));
    const uint32_t B = static_cast<uint32_t>(cache_indices.size(0));
    const uint32_t dq = static_cast<uint32_t>(q_out.size(1));
    const uint32_t dkv = static_cast<uint32_t>(k_out.size(1));
    RuntimeCheck(q_num > 0 && T == B * static_cast<uint32_t>(q_num), "T != B*q");
    RuntimeCheck(dq % kHeadDim == 0 && dkv % kHeadDim == 0, "dims % head_dim");
    RuntimeCheck((dq / kVecElems) % kHeadLanes == 0, "q lanes must tile heads");
    RuntimeCheck(qkvr.stride(1) == 1 && qkvr.stride(0) % kVecElems == 0,
                 "qkvr must be row-major with 16B-aligned rows");
    RuntimeCheck(q_off % kVecElems == 0 && k_off % kVecElems == 0 &&
                     v_off % kVecElems == 0, "slice offsets must be 16B aligned");
    RuntimeCheck(k_buf.stride(0) == v_buf.stride(0), "kv buf stride mismatch");
    RuntimeCheck(k_buf.stride(0) % kVecElems == 0, "kv buf rows must be 16B aligned");
    RuntimeCheck(k_cache.stride(2) == 1 && v_cache.stride(2) == 1,
                 "conv caches must be channel-contiguous");
    RuntimeCheck(k_inter.stride(3) == 1 && v_inter.stride(3) == 1 &&
                     k_inter.stride(0) == v_inter.stride(0) &&
                     k_inter.stride(1) == v_inter.stride(1) &&
                     k_inter.stride(2) == v_inter.stride(2),
                 "inter buffers must be channel-contiguous with equal strides");
    const uint32_t lanes = dq / kVecElems + 2 * (dkv / kVecElems);
    RuntimeCheck(lanes <= 1024, "token lanes must fit one block");

    const auto params = AttnPrologueParams{
        .qkvr = qkvr.data_ptr(),
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .cache_indices = cache_indices.data_ptr(),
        .cache_mask = cache_mask.data_ptr(),
        .k_weight = k_weight.data_ptr(),
        .v_weight = v_weight.data_ptr(),
        .k_inter = k_inter.data_ptr(),
        .v_inter = v_inter.data_ptr(),
        .q_gamma = q_gamma.data_ptr(),
        .k_gamma = k_gamma.data_ptr(),
        .eps = static_cast<float>(eps),
        .q_out = q_out.data_ptr(),
        .k_out = k_out.data_ptr(),
        .v_out = v_out.data_ptr(),
        .loc = loc.data_ptr(),
        .k_buf = k_buf.data_ptr(),
        .v_buf = v_buf.data_ptr(),
        .qkvr_stride_t = qkvr.stride(0),
        .q_off = q_off,
        .k_off = k_off,
        .v_off = v_off,
        .cache_stride_slot = k_cache.stride(0),
        .cache_stride_w = k_cache.stride(1),
        .weight_stride_d = k_weight.stride(0),
        .inter_stride_b = k_inter.stride(0),
        .inter_stride_t = k_inter.stride(1),
        .inter_stride_w = k_inter.stride(2),
        .kv_buf_stride = k_buf.stride(0),
        .T = T,
        .q = static_cast<uint32_t>(q_num),
        .dq = dq,
        .dkv = dkv,
    };
    RuntimeCheck(k_cache.stride(0) == v_cache.stride(0) &&
                     k_cache.stride(1) == v_cache.stride(1) &&
                     k_weight.stride(0) == v_weight.stride(0),
                 "k/v cache+weight strides must match");
    const uint32_t block = div_ceil(lanes, 32u) * 32u;
    const auto kernel = do_store
                            ? inkling_attn_prologue_kernel<DType, W, USE_SILU, USE_RESIDUAL, true>
                            : inkling_attn_prologue_kernel<DType, W, USE_SILU, USE_RESIDUAL, false>;
    LaunchKernel(dim3{T}, dim3{block}, qkvr.device())(kernel, params);
  }
};

}  // namespace
