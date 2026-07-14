#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>

#include <cfloat>
#include <climits>
#include <cstdint>

namespace {

// Fixed constants for the Inkling model
static constexpr int kInklingRoutedExperts = 256;
static constexpr int kInklingSharedExperts = 2;
static constexpr int kInklingTotalExperts = kInklingRoutedExperts + kInklingSharedExperts;
static constexpr int kInklingTopK = 6;
static constexpr int kInklingTopPow2 = 8;
static constexpr int kInklingWarpSize = 32;
static constexpr int kInklingValuesPerLane = kInklingRoutedExperts / kInklingWarpSize;

__device__ __forceinline__ float inkling_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ bool inkling_score_better(float score, int idx, float best_score, int best_idx) {
  return score > best_score || (score == best_score && idx < best_idx);
}

// FlashInfer routed-MoE pack: low 16 bits = bf16(weight) bits (round-to-nearest-even,
// same as torch/triton `.to(bfloat16)`), high 16 bits = int16 expert id.
__device__ __forceinline__ int32_t inkling_pack_routed(int32_t id, float w) {
  const uint32_t wbits = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(w)));
  return static_cast<int32_t>((static_cast<uint32_t>(id) << 16) | wbits);
}

template <int WarpsPerBlock, bool ReturnPacked>
__launch_bounds__(kInklingWarpSize* WarpsPerBlock) __global__ void inkling_gate_topk_renorm_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const float* __restrict__ global_scale,
    float* __restrict__ routed_w,
    float* __restrict__ shared_w,
    int64_t* __restrict__ indices,
    int32_t* __restrict__ packed,
    int64_t M,
    int64_t logits_stride_m,
    float route_scale) {
  const int lane = threadIdx.x;
  const int warp_in_block = threadIdx.y;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * WarpsPerBlock + warp_in_block;
  if (row >= M) {
    return;
  }

  const int64_t row_base = row * logits_stride_m;
  float local_scores[kInklingValuesPerLane];
#pragma unroll
  for (int i = 0; i < kInklingValuesPerLane; ++i) {
    const int expert = lane + i * kInklingWarpSize;
    const float raw = logits[row_base + expert];
    local_scores[i] = inkling_sigmoid(raw) + bias[expert];
  }

  int selected_idx[kInklingTopK];
#pragma unroll
  for (int k = 0; k < kInklingTopK; ++k) {
    float best_score = -FLT_MAX;
    int best_idx = INT_MAX;
#pragma unroll
    for (int i = 0; i < kInklingValuesPerLane; ++i) {
      const int expert = lane + i * kInklingWarpSize;
      const float score = local_scores[i];
      if (inkling_score_better(score, expert, best_score, best_idx)) {
        best_score = score;
        best_idx = expert;
      }
    }

#pragma unroll
    for (int offset = kInklingWarpSize / 2; offset > 0; offset >>= 1) {
      const float other_score = __shfl_xor_sync(0xffffffff, best_score, offset);
      const int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
      if (inkling_score_better(other_score, other_idx, best_score, best_idx)) {
        best_score = other_score;
        best_idx = other_idx;
      }
    }

    selected_idx[k] = best_idx;
    if (best_idx % kInklingWarpSize == lane) {
      local_scores[best_idx / kInklingWarpSize] = -FLT_MAX;
    }
    __syncwarp();
  }

  if (lane != 0) {
    return;
  }

  float active[kInklingTopPow2];
#pragma unroll
  for (int i = 0; i < kInklingTopK; ++i) {
    active[i] = inkling_sigmoid(logits[row_base + selected_idx[i]]);
  }
#pragma unroll
  for (int i = 0; i < kInklingSharedExperts; ++i) {
    active[kInklingTopK + i] = inkling_sigmoid(logits[row_base + kInklingRoutedExperts + i]);
  }

  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kInklingTopPow2; ++i) {
    sum += active[i];
  }
  const float scale = route_scale * global_scale[0] / sum;

#pragma unroll
  for (int i = 0; i < kInklingTopK; ++i) {
    const float w = active[i] * scale;
    if constexpr (ReturnPacked) {
      packed[row * kInklingTopK + i] = inkling_pack_routed(selected_idx[i], w);
    } else {
      routed_w[row * kInklingTopK + i] = w;
      indices[row * kInklingTopK + i] = static_cast<int64_t>(selected_idx[i]);
    }
  }
#pragma unroll
  for (int i = 0; i < kInklingSharedExperts; ++i) {
    shared_w[row * kInklingSharedExperts + i] = active[kInklingTopK + i] * scale;
  }
}

template <int WarpsPerBlock, bool ReturnPacked>
void launch_inkling_gate_topk_renorm(
    const float* logits,
    const float* bias,
    const float* global_scale,
    float* routed_w,
    float* shared_w,
    int64_t* indices,
    int32_t* packed,
    int64_t tokens,
    int64_t logits_stride_m,
    float route_scale,
    DLDevice device) {
  using namespace host;
  const dim3 block(kInklingWarpSize, WarpsPerBlock);
  const dim3 grid(static_cast<unsigned int>(div_ceil(tokens, static_cast<int64_t>(WarpsPerBlock))));
  LaunchKernel(grid, block, device)(
      inkling_gate_topk_renorm_kernel<WarpsPerBlock, ReturnPacked>,
      logits,
      bias,
      global_scale,
      routed_w,
      shared_w,
      indices,
      packed,
      tokens,
      logits_stride_m,
      route_scale);
}

template <bool ReturnPacked>
void dispatch_inkling_gate_topk_renorm(
    const float* logits,
    const float* bias,
    const float* global_scale,
    float* routed_w,
    float* shared_w,
    int64_t* indices,
    int32_t* packed,
    int64_t tokens,
    int64_t logits_stride_m,
    float route_scale,
    DLDevice device) {
  if (tokens <= 64) {
    launch_inkling_gate_topk_renorm<1, ReturnPacked>(
        logits, bias, global_scale, routed_w, shared_w, indices, packed, tokens, logits_stride_m, route_scale, device);
  } else if (tokens <= 1024) {
    launch_inkling_gate_topk_renorm<4, ReturnPacked>(
        logits, bias, global_scale, routed_w, shared_w, indices, packed, tokens, logits_stride_m, route_scale, device);
  } else {
    launch_inkling_gate_topk_renorm<8, ReturnPacked>(
        logits, bias, global_scale, routed_w, shared_w, indices, packed, tokens, logits_stride_m, route_scale, device);
  }
}

}  // namespace

void inkling_gate_topk_renorm(
    tvm::ffi::TensorView logits,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView global_scale,
    tvm::ffi::TensorView routed_w,
    tvm::ffi::TensorView shared_w,
    tvm::ffi::TensorView indices,
    double route_scale) {
  using namespace host;

  SymbolicSize M{"tokens"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({M, kInklingTotalExperts}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(logits);
  TensorMatcher({kInklingRoutedExperts}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(bias);
  TensorMatcher({1}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(global_scale);
  TensorMatcher({M, kInklingTopK}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(routed_w);
  TensorMatcher({M, kInklingSharedExperts}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(shared_w);
  TensorMatcher({M, kInklingTopK}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(indices);

  RuntimeCheck(logits.stride(1) == 1, "logits must be contiguous along the expert dimension");
  RuntimeCheck(bias.stride(0) == 1, "bias must be contiguous");
  RuntimeCheck(routed_w.stride(1) == 1, "routed_w must be contiguous along top-k dimension");
  RuntimeCheck(shared_w.stride(1) == 1, "shared_w must be contiguous along shared dimension");
  RuntimeCheck(indices.stride(1) == 1, "indices must be contiguous along top-k dimension");

  const int64_t tokens = M.unwrap();
  if (tokens == 0) {
    return;
  }

  const auto* logits_ptr = static_cast<const float*>(logits.data_ptr());
  const auto* bias_ptr = static_cast<const float*>(bias.data_ptr());
  const auto* global_scale_ptr = static_cast<const float*>(global_scale.data_ptr());
  auto* routed_w_ptr = static_cast<float*>(routed_w.data_ptr());
  auto* shared_w_ptr = static_cast<float*>(shared_w.data_ptr());
  auto* indices_ptr = static_cast<int64_t*>(indices.data_ptr());
  const float route_scale_f = static_cast<float>(route_scale);
  const DLDevice device = device_.unwrap();

  dispatch_inkling_gate_topk_renorm<false>(
      logits_ptr,
      bias_ptr,
      global_scale_ptr,
      routed_w_ptr,
      shared_w_ptr,
      indices_ptr,
      nullptr,
      tokens,
      logits.stride(0),
      route_scale_f,
      device);
}

// Packed variant: emits packed[M, kInklingTopK] int32 ((id<<16)|bf16 weight) instead of
// the routed_w + indices pair; shared_w still written.
void inkling_gate_topk_renorm_packed(
    tvm::ffi::TensorView logits,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView global_scale,
    tvm::ffi::TensorView packed,
    tvm::ffi::TensorView shared_w,
    double route_scale) {
  using namespace host;

  SymbolicSize M{"tokens"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({M, kInklingTotalExperts}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(logits);
  TensorMatcher({kInklingRoutedExperts}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(bias);
  TensorMatcher({1}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(global_scale);
  TensorMatcher({M, kInklingTopK}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(packed);
  TensorMatcher({M, kInklingSharedExperts}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(shared_w);

  RuntimeCheck(logits.stride(1) == 1, "logits must be contiguous along the expert dimension");
  RuntimeCheck(bias.stride(0) == 1, "bias must be contiguous");
  RuntimeCheck(packed.stride(1) == 1, "packed must be contiguous along top-k dimension");
  RuntimeCheck(shared_w.stride(1) == 1, "shared_w must be contiguous along shared dimension");

  const int64_t tokens = M.unwrap();
  if (tokens == 0) {
    return;
  }

  const auto* logits_ptr = static_cast<const float*>(logits.data_ptr());
  const auto* bias_ptr = static_cast<const float*>(bias.data_ptr());
  const auto* global_scale_ptr = static_cast<const float*>(global_scale.data_ptr());
  auto* packed_ptr = static_cast<int32_t*>(packed.data_ptr());
  auto* shared_w_ptr = static_cast<float*>(shared_w.data_ptr());
  const float route_scale_f = static_cast<float>(route_scale);
  const DLDevice device = device_.unwrap();

  dispatch_inkling_gate_topk_renorm<true>(
      logits_ptr,
      bias_ptr,
      global_scale_ptr,
      nullptr,
      shared_w_ptr,
      nullptr,
      packed_ptr,
      tokens,
      logits.stride(0),
      route_scale_f,
      device);
}
