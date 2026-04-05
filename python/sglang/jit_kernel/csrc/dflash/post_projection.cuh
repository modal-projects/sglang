#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr uint32_t kNumWarps = 4;
constexpr uint32_t kThreadsPerBlock = kNumWarps * device::kWarpThreads;

SGL_DEVICE float warp_reduce_sum_broadcast(float value) {
  for (int offset = device::kWarpThreads / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

struct PromptPostProjectionParams {
  const void* raw_k;
  const void* raw_v;
  void* dst_k;
  void* dst_v;
  const void* slot_ids;
  const void* positions;
  const void* k_norm_weight;
  const float* k_norm_eps;
  const float* cos_sin_cache;
  int64_t raw_k_layer_stride;
  int64_t raw_k_token_stride;
  int64_t raw_k_head_stride;
  int64_t dst_k_layer_stride;
  int64_t dst_k_slot_stride;
  int64_t dst_k_head_stride;
  int64_t slot_ids_stride;
  int64_t positions_stride;
  int64_t k_norm_weight_layer_stride;
  int64_t cos_sin_stride;
  uint32_t num_layers;
  uint32_t num_tokens;
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t rotary_dim;
  uint32_t half_rotary_dim;
};

template <bool kUsePDL, typename DType, typename IndexT, typename PosT>
__global__ __launch_bounds__(kThreadsPerBlock) void dflash_prompt_post_projection(
    const __grid_constant__ PromptPostProjectionParams params) {
  using namespace device;
  const uint32_t warp_id = threadIdx.x / device::kWarpThreads;
  const uint32_t lane_id = threadIdx.x % device::kWarpThreads;
  const uint64_t global_warp_id = static_cast<uint64_t>(blockIdx.x) * kNumWarps + warp_id;
  const uint64_t rows_per_layer = static_cast<uint64_t>(params.num_tokens) * static_cast<uint64_t>(params.num_heads);
  const uint64_t total_rows = static_cast<uint64_t>(params.num_layers) * static_cast<uint64_t>(rows_per_layer);
  if (global_warp_id >= total_rows) return;

  const uint32_t layer_idx = global_warp_id / rows_per_layer;
  const uint32_t row_rem = global_warp_id % rows_per_layer;
  const uint32_t token_idx = row_rem / params.num_heads;
  const uint32_t head_idx = row_rem % params.num_heads;

  PDLWaitPrimary<kUsePDL>();

  const auto slot_idx = *(static_cast<const IndexT*>(params.slot_ids) + token_idx * params.slot_ids_stride);
  const auto position =
      static_cast<int64_t>(*(static_cast<const PosT*>(params.positions) + token_idx * params.positions_stride));
  const float eps = params.k_norm_eps[layer_idx];

  const auto raw_k_ptr = static_cast<const DType*>(params.raw_k) + layer_idx * params.raw_k_layer_stride +
                         token_idx * params.raw_k_token_stride + head_idx * params.raw_k_head_stride;
  const auto raw_v_ptr = static_cast<const DType*>(params.raw_v) + layer_idx * params.raw_k_layer_stride +
                         token_idx * params.raw_k_token_stride + head_idx * params.raw_k_head_stride;
  const auto dst_k_ptr = static_cast<DType*>(params.dst_k) + layer_idx * params.dst_k_layer_stride +
                         static_cast<int64_t>(slot_idx) * params.dst_k_slot_stride +
                         head_idx * params.dst_k_head_stride;
  const auto dst_v_ptr = static_cast<DType*>(params.dst_v) + layer_idx * params.dst_k_layer_stride +
                         static_cast<int64_t>(slot_idx) * params.dst_k_slot_stride +
                         head_idx * params.dst_k_head_stride;
  const auto norm_weight_ptr =
      static_cast<const DType*>(params.k_norm_weight) + layer_idx * params.k_norm_weight_layer_stride;

  float sum_sq = 0.0f;
  for (uint32_t d = lane_id; d < params.head_dim; d += device::kWarpThreads) {
    const float k = device::cast<float>(raw_k_ptr[d]);
    sum_sq += k * k;
  }
  sum_sq = warp_reduce_sum_broadcast(sum_sq);
  const float inv_rms = rsqrtf(sum_sq / static_cast<float>(params.head_dim) + eps);

  for (uint32_t d = lane_id; d < params.head_dim; d += device::kWarpThreads) {
    const float weight = device::cast<float>(norm_weight_ptr[d]);
    const float k = device::cast<float>(raw_k_ptr[d]) * inv_rms * weight;
    const float v = device::cast<float>(raw_v_ptr[d]);
    float out = k;
    if (d < params.rotary_dim) {
      const uint32_t base_idx = d < params.half_rotary_dim ? d : d - params.half_rotary_dim;
      const uint32_t pair_idx = d < params.half_rotary_dim ? d + params.half_rotary_dim : d - params.half_rotary_dim;
      const float pair_weight = device::cast<float>(norm_weight_ptr[pair_idx]);
      const float pair_k = device::cast<float>(raw_k_ptr[pair_idx]) * inv_rms * pair_weight;
      const float cos_v = params.cos_sin_cache[position * params.cos_sin_stride + base_idx];
      const float sin_v = params.cos_sin_cache[position * params.cos_sin_stride + params.half_rotary_dim + base_idx];
      out = d < params.half_rotary_dim ? (k * cos_v - pair_k * sin_v) : (k * cos_v + pair_k * sin_v);
    }
    dst_k_ptr[d] = device::cast<DType>(out);
    dst_v_ptr[d] = device::cast<DType>(v);
  }

  PDLTriggerSecondary<kUsePDL>();
}

struct CommitPostProjectionParams {
  const void* raw_k;
  const void* raw_v;
  void* dst_k;
  void* dst_v;
  const void* slot_ids_2d;
  const void* commit_lens;
  const void* positions;
  const void* k_norm_weight;
  const float* k_norm_eps;
  const float* cos_sin_cache;
  int64_t raw_k_layer_stride;
  int64_t raw_k_batch_stride;
  int64_t raw_k_block_stride;
  int64_t raw_k_head_stride;
  int64_t dst_k_layer_stride;
  int64_t dst_k_slot_stride;
  int64_t dst_k_head_stride;
  int64_t slot_ids_batch_stride;
  int64_t slot_ids_block_stride;
  int64_t commit_lens_stride;
  int64_t positions_batch_stride;
  int64_t positions_block_stride;
  int64_t k_norm_weight_layer_stride;
  int64_t cos_sin_stride;
  uint32_t num_layers;
  uint32_t batch_size;
  uint32_t block_size;
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t rotary_dim;
  uint32_t half_rotary_dim;
};

template <bool kUsePDL, typename DType, typename IndexT, typename LenT, typename PosT>
__global__ __launch_bounds__(kThreadsPerBlock) void dflash_commit_post_projection(
    const __grid_constant__ CommitPostProjectionParams params) {
  using namespace device;
  const uint32_t warp_id = threadIdx.x / device::kWarpThreads;
  const uint32_t lane_id = threadIdx.x % device::kWarpThreads;
  const uint64_t global_warp_id = static_cast<uint64_t>(blockIdx.x) * kNumWarps + warp_id;
  const uint64_t rows_per_layer = static_cast<uint64_t>(params.batch_size) * static_cast<uint64_t>(params.block_size) *
                                  static_cast<uint64_t>(params.num_heads);
  const uint64_t total_rows = static_cast<uint64_t>(params.num_layers) * static_cast<uint64_t>(rows_per_layer);
  if (global_warp_id >= total_rows) return;

  const uint32_t layer_idx = global_warp_id / rows_per_layer;
  const uint32_t row_rem = global_warp_id % rows_per_layer;
  const uint32_t batch_idx = row_rem / (params.block_size * params.num_heads);
  const uint32_t rem = row_rem % (params.block_size * params.num_heads);
  const uint32_t block_idx = rem / params.num_heads;
  const uint32_t head_idx = rem % params.num_heads;

  PDLWaitPrimary<kUsePDL>();

  const auto keep = *(static_cast<const LenT*>(params.commit_lens) + batch_idx * params.commit_lens_stride);
  if (block_idx >= static_cast<uint32_t>(keep)) return;

  const auto slot_idx =
      *(static_cast<const IndexT*>(params.slot_ids_2d) + batch_idx * params.slot_ids_batch_stride +
        block_idx * params.slot_ids_block_stride);
  const auto position = static_cast<int64_t>(
      *(static_cast<const PosT*>(params.positions) + batch_idx * params.positions_batch_stride +
        block_idx * params.positions_block_stride));
  const float eps = params.k_norm_eps[layer_idx];

  const auto raw_k_ptr = static_cast<const DType*>(params.raw_k) + layer_idx * params.raw_k_layer_stride +
                         batch_idx * params.raw_k_batch_stride + block_idx * params.raw_k_block_stride +
                         head_idx * params.raw_k_head_stride;
  const auto raw_v_ptr = static_cast<const DType*>(params.raw_v) + layer_idx * params.raw_k_layer_stride +
                         batch_idx * params.raw_k_batch_stride + block_idx * params.raw_k_block_stride +
                         head_idx * params.raw_k_head_stride;
  const auto dst_k_ptr = static_cast<DType*>(params.dst_k) + layer_idx * params.dst_k_layer_stride +
                         static_cast<int64_t>(slot_idx) * params.dst_k_slot_stride +
                         head_idx * params.dst_k_head_stride;
  const auto dst_v_ptr = static_cast<DType*>(params.dst_v) + layer_idx * params.dst_k_layer_stride +
                         static_cast<int64_t>(slot_idx) * params.dst_k_slot_stride +
                         head_idx * params.dst_k_head_stride;
  const auto norm_weight_ptr =
      static_cast<const DType*>(params.k_norm_weight) + layer_idx * params.k_norm_weight_layer_stride;

  float sum_sq = 0.0f;
  for (uint32_t d = lane_id; d < params.head_dim; d += device::kWarpThreads) {
    const float k = device::cast<float>(raw_k_ptr[d]);
    sum_sq += k * k;
  }
  sum_sq = warp_reduce_sum_broadcast(sum_sq);
  const float inv_rms = rsqrtf(sum_sq / static_cast<float>(params.head_dim) + eps);

  for (uint32_t d = lane_id; d < params.head_dim; d += device::kWarpThreads) {
    const float weight = device::cast<float>(norm_weight_ptr[d]);
    const float k = device::cast<float>(raw_k_ptr[d]) * inv_rms * weight;
    const float v = device::cast<float>(raw_v_ptr[d]);
    float out = k;
    if (d < params.rotary_dim) {
      const uint32_t base_idx = d < params.half_rotary_dim ? d : d - params.half_rotary_dim;
      const uint32_t pair_idx = d < params.half_rotary_dim ? d + params.half_rotary_dim : d - params.half_rotary_dim;
      const float pair_weight = device::cast<float>(norm_weight_ptr[pair_idx]);
      const float pair_k = device::cast<float>(raw_k_ptr[pair_idx]) * inv_rms * pair_weight;
      const float cos_v = params.cos_sin_cache[position * params.cos_sin_stride + base_idx];
      const float sin_v = params.cos_sin_cache[position * params.cos_sin_stride + params.half_rotary_dim + base_idx];
      out = d < params.half_rotary_dim ? (k * cos_v - pair_k * sin_v) : (k * cos_v + pair_k * sin_v);
    }
    dst_k_ptr[d] = device::cast<DType>(out);
    dst_v_ptr[d] = device::cast<DType>(v);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL, typename DType>
struct DFlashPromptPostProjectionKernel {
  template <typename IndexT, typename PosT>
  static constexpr auto kernel = dflash_prompt_post_projection<kUsePDL, DType, IndexT, PosT>;

  static void
  run(const tvm::ffi::TensorView raw_k,
      const tvm::ffi::TensorView raw_v,
      const tvm::ffi::TensorView dst_k,
      const tvm::ffi::TensorView dst_v,
      const tvm::ffi::TensorView slot_ids,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView k_norm_weight,
      const tvm::ffi::TensorView k_norm_eps,
      const tvm::ffi::TensorView cos_sin_cache) {
    using namespace host;
    auto L = SymbolicSize{"num_layers"};
    auto T = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"num_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto S = SymbolicSize{"num_slots"};
    auto RKL = SymbolicSize{"raw_k_layer_stride"};
    auto RKT = SymbolicSize{"raw_k_token_stride"};
    auto RKH = SymbolicSize{"raw_k_head_stride"};
    auto DKL = SymbolicSize{"dst_k_layer_stride"};
    auto DKS = SymbolicSize{"dst_k_slot_stride"};
    auto DKH = SymbolicSize{"dst_k_head_stride"};
    auto SS = SymbolicSize{"slot_ids_stride"};
    auto PS = SymbolicSize{"positions_stride"};
    auto NWL = SymbolicSize{"k_norm_weight_layer_stride"};
    auto C = SymbolicSize{"cos_sin_rows"};
    auto R = SymbolicSize{"rotary_dim"};
    auto CS = SymbolicSize{"cos_sin_stride"};
    auto device = SymbolicDevice{};
    auto index_dtype = SymbolicDType{};
    auto pos_dtype = SymbolicDType{};
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({L, T, H, D})
        .with_strides({RKL, RKT, RKH, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(raw_k)
        .verify(raw_v);
    TensorMatcher({L, S, H, D})
        .with_strides({DKL, DKS, DKH, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(dst_k)
        .verify(dst_v);
    TensorMatcher({T})
        .with_strides({SS})
        .with_dtype<int32_t, int64_t>(index_dtype)
        .with_device(device)
        .verify(slot_ids);
    TensorMatcher({T}).with_strides({PS}).with_dtype<int32_t, int64_t>(pos_dtype).with_device(device).verify(positions);
    TensorMatcher({L, D}).with_strides({NWL, 1}).with_dtype<DType>().with_device(device).verify(k_norm_weight);
    TensorMatcher({L}).with_dtype<float>().with_device(device).verify(k_norm_eps);
    TensorMatcher({C, R}).with_strides({CS, 1}).with_dtype<float>().with_device(device).verify(cos_sin_cache);

    RuntimeCheck(R.unwrap() > 0 && R.unwrap() <= D.unwrap() && R.unwrap() % 2 == 0);

    const auto params = PromptPostProjectionParams{
        .raw_k = raw_k.data_ptr(),
        .raw_v = raw_v.data_ptr(),
        .dst_k = dst_k.data_ptr(),
        .dst_v = dst_v.data_ptr(),
        .slot_ids = slot_ids.data_ptr(),
        .positions = positions.data_ptr(),
        .k_norm_weight = k_norm_weight.data_ptr(),
        .k_norm_eps = static_cast<const float*>(k_norm_eps.data_ptr()),
        .cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr()),
        .raw_k_layer_stride = RKL.unwrap(),
        .raw_k_token_stride = RKT.unwrap(),
        .raw_k_head_stride = RKH.unwrap(),
        .dst_k_layer_stride = DKL.unwrap(),
        .dst_k_slot_stride = DKS.unwrap(),
        .dst_k_head_stride = DKH.unwrap(),
        .slot_ids_stride = SS.unwrap(),
        .positions_stride = PS.unwrap(),
        .k_norm_weight_layer_stride = NWL.unwrap(),
        .cos_sin_stride = CS.unwrap(),
        .num_layers = static_cast<uint32_t>(L.unwrap()),
        .num_tokens = static_cast<uint32_t>(T.unwrap()),
        .num_heads = static_cast<uint32_t>(H.unwrap()),
        .head_dim = static_cast<uint32_t>(D.unwrap()),
        .rotary_dim = static_cast<uint32_t>(R.unwrap()),
        .half_rotary_dim = static_cast<uint32_t>(R.unwrap() / 2),
    };
    const auto num_rows = static_cast<uint64_t>(params.num_layers) * static_cast<uint64_t>(params.num_tokens) *
                          static_cast<uint64_t>(params.num_heads);
    const auto num_blocks = div_ceil(num_rows, static_cast<uint64_t>(kNumWarps));

    if (index_dtype.is_type<int32_t>()) {
      if (pos_dtype.is_type<int32_t>()) {
        LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())
            .enable_pdl(kUsePDL)(kernel<int32_t, int32_t>, params);
      } else {
        LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())
            .enable_pdl(kUsePDL)(kernel<int32_t, int64_t>, params);
      }
    } else {
      if (pos_dtype.is_type<int32_t>()) {
        LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())
            .enable_pdl(kUsePDL)(kernel<int64_t, int32_t>, params);
      } else {
        LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())
            .enable_pdl(kUsePDL)(kernel<int64_t, int64_t>, params);
      }
    }
  }
};

template <bool kUsePDL, typename DType>
struct DFlashCommitPostProjectionKernel {
  template <typename IndexT, typename LenT, typename PosT>
  static constexpr auto kernel = dflash_commit_post_projection<kUsePDL, DType, IndexT, LenT, PosT>;

  static void
  run(const tvm::ffi::TensorView raw_k,
      const tvm::ffi::TensorView raw_v,
      const tvm::ffi::TensorView dst_k,
      const tvm::ffi::TensorView dst_v,
      const tvm::ffi::TensorView slot_ids_2d,
      const tvm::ffi::TensorView commit_lens,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView k_norm_weight,
      const tvm::ffi::TensorView k_norm_eps,
      const tvm::ffi::TensorView cos_sin_cache) {
    using namespace host;
    auto L = SymbolicSize{"num_layers"};
    auto B = SymbolicSize{"batch_size"};
    auto T = SymbolicSize{"block_size"};
    auto H = SymbolicSize{"num_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto S = SymbolicSize{"num_slots"};
    auto RKL = SymbolicSize{"raw_k_layer_stride"};
    auto RKB = SymbolicSize{"raw_k_batch_stride"};
    auto RKT = SymbolicSize{"raw_k_block_stride"};
    auto RKH = SymbolicSize{"raw_k_head_stride"};
    auto DKL = SymbolicSize{"dst_k_layer_stride"};
    auto DKS = SymbolicSize{"dst_k_slot_stride"};
    auto DKH = SymbolicSize{"dst_k_head_stride"};
    auto SB = SymbolicSize{"slot_ids_batch_stride"};
    auto ST = SymbolicSize{"slot_ids_block_stride"};
    auto CLS = SymbolicSize{"commit_lens_stride"};
    auto PB = SymbolicSize{"positions_batch_stride"};
    auto PT = SymbolicSize{"positions_block_stride"};
    auto NWL = SymbolicSize{"k_norm_weight_layer_stride"};
    auto C = SymbolicSize{"cos_sin_rows"};
    auto R = SymbolicSize{"rotary_dim"};
    auto CS = SymbolicSize{"cos_sin_stride"};
    auto device = SymbolicDevice{};
    auto index_dtype = SymbolicDType{};
    auto len_dtype = SymbolicDType{};
    auto pos_dtype = SymbolicDType{};
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({L, B, T, H, D})
        .with_strides({RKL, RKB, RKT, RKH, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(raw_k)
        .verify(raw_v);
    TensorMatcher({L, S, H, D})
        .with_strides({DKL, DKS, DKH, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(dst_k)
        .verify(dst_v);
    TensorMatcher({B, T})
        .with_strides({SB, ST})
        .with_dtype<int32_t, int64_t>(index_dtype)
        .with_device(device)
        .verify(slot_ids_2d);
    TensorMatcher({B}).with_strides({CLS}).with_dtype<int32_t, int64_t>(len_dtype).with_device(device).verify(
        commit_lens);
    TensorMatcher({B, T}).with_strides({PB, PT}).with_dtype<int32_t, int64_t>(pos_dtype).with_device(device).verify(
        positions);
    TensorMatcher({L, D}).with_strides({NWL, 1}).with_dtype<DType>().with_device(device).verify(k_norm_weight);
    TensorMatcher({L}).with_dtype<float>().with_device(device).verify(k_norm_eps);
    TensorMatcher({C, R}).with_strides({CS, 1}).with_dtype<float>().with_device(device).verify(cos_sin_cache);

    RuntimeCheck(R.unwrap() > 0 && R.unwrap() <= D.unwrap() && R.unwrap() % 2 == 0);

    const auto params = CommitPostProjectionParams{
        .raw_k = raw_k.data_ptr(),
        .raw_v = raw_v.data_ptr(),
        .dst_k = dst_k.data_ptr(),
        .dst_v = dst_v.data_ptr(),
        .slot_ids_2d = slot_ids_2d.data_ptr(),
        .commit_lens = commit_lens.data_ptr(),
        .positions = positions.data_ptr(),
        .k_norm_weight = k_norm_weight.data_ptr(),
        .k_norm_eps = static_cast<const float*>(k_norm_eps.data_ptr()),
        .cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr()),
        .raw_k_layer_stride = RKL.unwrap(),
        .raw_k_batch_stride = RKB.unwrap(),
        .raw_k_block_stride = RKT.unwrap(),
        .raw_k_head_stride = RKH.unwrap(),
        .dst_k_layer_stride = DKL.unwrap(),
        .dst_k_slot_stride = DKS.unwrap(),
        .dst_k_head_stride = DKH.unwrap(),
        .slot_ids_batch_stride = SB.unwrap(),
        .slot_ids_block_stride = ST.unwrap(),
        .commit_lens_stride = CLS.unwrap(),
        .positions_batch_stride = PB.unwrap(),
        .positions_block_stride = PT.unwrap(),
        .k_norm_weight_layer_stride = NWL.unwrap(),
        .cos_sin_stride = CS.unwrap(),
        .num_layers = static_cast<uint32_t>(L.unwrap()),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .block_size = static_cast<uint32_t>(T.unwrap()),
        .num_heads = static_cast<uint32_t>(H.unwrap()),
        .head_dim = static_cast<uint32_t>(D.unwrap()),
        .rotary_dim = static_cast<uint32_t>(R.unwrap()),
        .half_rotary_dim = static_cast<uint32_t>(R.unwrap() / 2),
    };
    const auto num_rows = static_cast<uint64_t>(params.num_layers) * static_cast<uint64_t>(params.batch_size) *
                          static_cast<uint64_t>(params.block_size) * static_cast<uint64_t>(params.num_heads);
    const auto num_blocks = div_ceil(num_rows, static_cast<uint64_t>(kNumWarps));

    auto launch = [&](auto k) {
      LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap()).enable_pdl(kUsePDL)(k, params);
    };
    if (index_dtype.is_type<int32_t>()) {
      if (len_dtype.is_type<int32_t>()) {
        if (pos_dtype.is_type<int32_t>()) {
          launch(kernel<int32_t, int32_t, int32_t>);
        } else {
          launch(kernel<int32_t, int32_t, int64_t>);
        }
      } else {
        if (pos_dtype.is_type<int32_t>()) {
          launch(kernel<int32_t, int64_t, int32_t>);
        } else {
          launch(kernel<int32_t, int64_t, int64_t>);
        }
      }
    } else {
      if (len_dtype.is_type<int32_t>()) {
        if (pos_dtype.is_type<int32_t>()) {
          launch(kernel<int64_t, int32_t, int32_t>);
        } else {
          launch(kernel<int64_t, int32_t, int64_t>);
        }
      } else {
        if (pos_dtype.is_type<int32_t>()) {
          launch(kernel<int64_t, int64_t, int32_t>);
        } else {
          launch(kernel<int64_t, int64_t, int64_t>);
        }
      }
    }
  }
};

}  // namespace
