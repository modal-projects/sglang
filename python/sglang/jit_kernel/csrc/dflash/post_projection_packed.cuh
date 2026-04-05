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
constexpr uint32_t kFastLocalElems = 8;

SGL_DEVICE float warp_reduce_sum_broadcast(float value) {
  for (int offset = device::kWarpThreads / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

struct PromptPackedPostProjectionParams {
  const void* packed_kv;
  void* dst_k;
  void* dst_v;
  const void* slot_ids;
  const void* positions;
  const void* k_norm_weight;
  const float* k_norm_eps;
  const float* cos_sin_cache;
  int64_t packed_token_stride;
  int64_t packed_group_stride;
  int64_t packed_pair_stride;
  int64_t packed_head_stride;
  int64_t dst_k_layer_stride;
  int64_t dst_k_slot_stride;
  int64_t dst_k_head_stride;
  int64_t slot_ids_stride;
  int64_t positions_stride;
  int64_t k_norm_weight_layer_stride;
  int64_t cos_sin_stride;
  uint32_t layer_start;
  uint32_t group_count;
  uint32_t num_tokens;
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t rotary_dim;
  uint32_t half_rotary_dim;
};

template <bool kUsePDL, typename DType, typename IndexT, typename PosT>
__global__ __launch_bounds__(kThreadsPerBlock) void dflash_prompt_packed_post_projection(
    const __grid_constant__ PromptPackedPostProjectionParams params) {
  using namespace device;
  const uint32_t warp_id = threadIdx.x / device::kWarpThreads;
  const uint32_t lane_id = threadIdx.x % device::kWarpThreads;
  const uint64_t global_warp_id = static_cast<uint64_t>(blockIdx.x) * kNumWarps + warp_id;
  const uint64_t rows_per_group = static_cast<uint64_t>(params.num_tokens) * static_cast<uint64_t>(params.num_heads);
  const uint64_t total_rows = static_cast<uint64_t>(params.group_count) * static_cast<uint64_t>(rows_per_group);
  if (global_warp_id >= total_rows) return;

  const uint32_t row_base = global_warp_id / params.group_count;
  const uint32_t group_idx = global_warp_id % params.group_count;
  const uint32_t token_idx = row_base / params.num_heads;
  const uint32_t head_idx = row_base % params.num_heads;
  const uint32_t layer_idx = params.layer_start + group_idx;

  PDLWaitPrimary<kUsePDL>();

  const auto slot_idx = *(static_cast<const IndexT*>(params.slot_ids) + token_idx * params.slot_ids_stride);
  const auto position =
      static_cast<int64_t>(*(static_cast<const PosT*>(params.positions) + token_idx * params.positions_stride));
  const float eps = params.k_norm_eps[layer_idx];

  const auto packed_base = static_cast<const DType*>(params.packed_kv) + token_idx * params.packed_token_stride +
                           group_idx * params.packed_group_stride + head_idx * params.packed_head_stride;
  const auto raw_k_ptr = packed_base;
  const auto raw_v_ptr = packed_base + params.packed_pair_stride;
  const auto dst_k_ptr = static_cast<DType*>(params.dst_k) + layer_idx * params.dst_k_layer_stride +
                         static_cast<int64_t>(slot_idx) * params.dst_k_slot_stride +
                         head_idx * params.dst_k_head_stride;
  const auto dst_v_ptr = static_cast<DType*>(params.dst_v) + layer_idx * params.dst_k_layer_stride +
                         static_cast<int64_t>(slot_idx) * params.dst_k_slot_stride +
                         head_idx * params.dst_k_head_stride;
  const auto norm_weight_ptr =
      static_cast<const DType*>(params.k_norm_weight) + layer_idx * params.k_norm_weight_layer_stride;

  float local_raw_k[kFastLocalElems];
  float local_weight[kFastLocalElems];
  uint32_t local_count = 0;
  float sum_sq = 0.0f;
  for (uint32_t d = lane_id; d < params.head_dim; d += device::kWarpThreads) {
    const float k_val = device::cast<float>(raw_k_ptr[d]);
    if (local_count < kFastLocalElems) {
      local_raw_k[local_count] = k_val;
      local_weight[local_count] = device::cast<float>(norm_weight_ptr[d]);
    }
    sum_sq += k_val * k_val;
    ++local_count;
  }
  sum_sq = warp_reduce_sum_broadcast(sum_sq);
  const float inv_rms = rsqrtf(sum_sq / static_cast<float>(params.head_dim) + eps);
  const bool use_local_pair_reuse =
      (params.half_rotary_dim % device::kWarpThreads) == 0 && local_count <= kFastLocalElems;
  const uint32_t pair_delta = params.half_rotary_dim / device::kWarpThreads;

  for (uint32_t i = 0; i < local_count; ++i) {
    const uint32_t d = lane_id + i * device::kWarpThreads;
    const float k = local_raw_k[i] * inv_rms * local_weight[i];
    const float v = device::cast<float>(raw_v_ptr[d]);
    float out = k;
    if (d < params.rotary_dim) {
      const uint32_t base_idx = d < params.half_rotary_dim ? d : d - params.half_rotary_dim;
      float pair_k;
      if (use_local_pair_reuse) {
        const uint32_t pair_i = d < params.half_rotary_dim ? i + pair_delta : i - pair_delta;
        pair_k = local_raw_k[pair_i] * inv_rms * local_weight[pair_i];
      } else {
        const uint32_t pair_idx = d < params.half_rotary_dim ? d + params.half_rotary_dim : d - params.half_rotary_dim;
        const float pair_weight = device::cast<float>(norm_weight_ptr[pair_idx]);
        pair_k = device::cast<float>(raw_k_ptr[pair_idx]) * inv_rms * pair_weight;
      }
      const float cos_v = params.cos_sin_cache[position * params.cos_sin_stride + base_idx];
      const float sin_v = params.cos_sin_cache[position * params.cos_sin_stride + params.half_rotary_dim + base_idx];
      out = d < params.half_rotary_dim ? (k * cos_v - pair_k * sin_v) : (k * cos_v + pair_k * sin_v);
    }
    dst_k_ptr[d] = device::cast<DType>(out);
    dst_v_ptr[d] = device::cast<DType>(v);
  }

  PDLTriggerSecondary<kUsePDL>();
}

struct CommitPackedPostProjectionParams {
  const void* packed_kv;
  void* dst_k;
  void* dst_v;
  const void* slot_ids_2d;
  const void* commit_lens;
  const void* positions;
  const void* k_norm_weight;
  const float* k_norm_eps;
  const float* cos_sin_cache;
  int64_t packed_batch_stride;
  int64_t packed_block_stride;
  int64_t packed_group_stride;
  int64_t packed_pair_stride;
  int64_t packed_head_stride;
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
  uint32_t layer_start;
  uint32_t batch_size;
  uint32_t block_size;
  uint32_t group_count;
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t rotary_dim;
  uint32_t half_rotary_dim;
};

template <bool kUsePDL, typename DType, typename IndexT, typename LenT, typename PosT>
__global__ __launch_bounds__(kThreadsPerBlock) void dflash_commit_packed_post_projection(
    const __grid_constant__ CommitPackedPostProjectionParams params) {
  using namespace device;
  const uint32_t warp_id = threadIdx.x / device::kWarpThreads;
  const uint32_t lane_id = threadIdx.x % device::kWarpThreads;
  const uint64_t global_warp_id = static_cast<uint64_t>(blockIdx.x) * kNumWarps + warp_id;
  const uint64_t rows_per_group = static_cast<uint64_t>(params.batch_size) * static_cast<uint64_t>(params.block_size) *
                                  static_cast<uint64_t>(params.num_heads);
  const uint64_t total_rows = static_cast<uint64_t>(params.group_count) * static_cast<uint64_t>(rows_per_group);
  if (global_warp_id >= total_rows) return;

  const uint32_t row_base = global_warp_id / params.group_count;
  const uint32_t group_idx = global_warp_id % params.group_count;
  const uint32_t batch_idx = row_base / (params.block_size * params.num_heads);
  const uint32_t rem = row_base % (params.block_size * params.num_heads);
  const uint32_t block_idx = rem / params.num_heads;
  const uint32_t head_idx = rem % params.num_heads;
  const uint32_t layer_idx = params.layer_start + group_idx;

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

  const auto packed_base = static_cast<const DType*>(params.packed_kv) + batch_idx * params.packed_batch_stride +
                           block_idx * params.packed_block_stride + group_idx * params.packed_group_stride +
                           head_idx * params.packed_head_stride;
  const auto raw_k_ptr = packed_base;
  const auto raw_v_ptr = packed_base + params.packed_pair_stride;
  const auto dst_k_ptr = static_cast<DType*>(params.dst_k) + layer_idx * params.dst_k_layer_stride +
                         static_cast<int64_t>(slot_idx) * params.dst_k_slot_stride +
                         head_idx * params.dst_k_head_stride;
  const auto dst_v_ptr = static_cast<DType*>(params.dst_v) + layer_idx * params.dst_k_layer_stride +
                         static_cast<int64_t>(slot_idx) * params.dst_k_slot_stride +
                         head_idx * params.dst_k_head_stride;
  const auto norm_weight_ptr =
      static_cast<const DType*>(params.k_norm_weight) + layer_idx * params.k_norm_weight_layer_stride;

  float local_raw_k[kFastLocalElems];
  float local_weight[kFastLocalElems];
  uint32_t local_count = 0;
  float sum_sq = 0.0f;
  for (uint32_t d = lane_id; d < params.head_dim; d += device::kWarpThreads) {
    const float k_val = device::cast<float>(raw_k_ptr[d]);
    if (local_count < kFastLocalElems) {
      local_raw_k[local_count] = k_val;
      local_weight[local_count] = device::cast<float>(norm_weight_ptr[d]);
    }
    sum_sq += k_val * k_val;
    ++local_count;
  }
  sum_sq = warp_reduce_sum_broadcast(sum_sq);
  const float inv_rms = rsqrtf(sum_sq / static_cast<float>(params.head_dim) + eps);
  const bool use_local_pair_reuse =
      (params.half_rotary_dim % device::kWarpThreads) == 0 && local_count <= kFastLocalElems;
  const uint32_t pair_delta = params.half_rotary_dim / device::kWarpThreads;

  for (uint32_t i = 0; i < local_count; ++i) {
    const uint32_t d = lane_id + i * device::kWarpThreads;
    const float k = local_raw_k[i] * inv_rms * local_weight[i];
    const float v = device::cast<float>(raw_v_ptr[d]);
    float out = k;
    if (d < params.rotary_dim) {
      const uint32_t base_idx = d < params.half_rotary_dim ? d : d - params.half_rotary_dim;
      float pair_k;
      if (use_local_pair_reuse) {
        const uint32_t pair_i = d < params.half_rotary_dim ? i + pair_delta : i - pair_delta;
        pair_k = local_raw_k[pair_i] * inv_rms * local_weight[pair_i];
      } else {
        const uint32_t pair_idx = d < params.half_rotary_dim ? d + params.half_rotary_dim : d - params.half_rotary_dim;
        const float pair_weight = device::cast<float>(norm_weight_ptr[pair_idx]);
        pair_k = device::cast<float>(raw_k_ptr[pair_idx]) * inv_rms * pair_weight;
      }
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
struct DFlashPromptPackedPostProjectionKernel {
  template <typename IndexT, typename PosT>
  static constexpr auto kernel = dflash_prompt_packed_post_projection<kUsePDL, DType, IndexT, PosT>;

  static void
  run(const tvm::ffi::TensorView packed_kv,
      const tvm::ffi::TensorView dst_k,
      const tvm::ffi::TensorView dst_v,
      const tvm::ffi::TensorView slot_ids,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView k_norm_weight,
      const tvm::ffi::TensorView k_norm_eps,
      const tvm::ffi::TensorView cos_sin_cache,
      int layer_start) {
    using namespace host;
    auto T = SymbolicSize{"num_tokens"};
    auto G = SymbolicSize{"group_count"};
    auto P = SymbolicSize{"kv_pair"};
    auto H = SymbolicSize{"num_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto L = SymbolicSize{"num_layers"};
    auto S = SymbolicSize{"num_slots"};
    auto C = SymbolicSize{"cos_cache_len"};
    auto R = SymbolicSize{"rotary_dim"};
    auto PKT = SymbolicSize{"packed_token_stride"};
    auto PKG = SymbolicSize{"packed_group_stride"};
    auto PKP = SymbolicSize{"packed_pair_stride"};
    auto PKH = SymbolicSize{"packed_head_stride"};
    auto DKL = SymbolicSize{"dst_k_layer_stride"};
    auto DKS = SymbolicSize{"dst_k_slot_stride"};
    auto DKH = SymbolicSize{"dst_k_head_stride"};
    auto SLOT = SymbolicSize{"slot_ids_stride"};
    auto POS = SymbolicSize{"positions_stride"};
    auto NWL = SymbolicSize{"k_norm_weight_layer_stride"};
    auto CSS = SymbolicSize{"cos_sin_stride"};
    auto index_dtype = SymbolicDType{};
    auto pos_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({T, G, P, H, D})  //
        .with_strides({PKT, PKG, PKP, PKH, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(packed_kv);
    TensorMatcher({L, S, H, D})  //
        .with_strides({DKL, DKS, DKH, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(dst_k)
        .verify(dst_v);
    TensorMatcher({T})  //
        .with_strides({SLOT})
        .with_dtype<int32_t, int64_t>(index_dtype)
        .with_device(device)
        .verify(slot_ids);
    TensorMatcher({T})  //
        .with_strides({POS})
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device)
        .verify(positions);
    TensorMatcher({L, D})  //
        .with_strides({NWL, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k_norm_weight);
    TensorMatcher({L})  //
        .with_strides({1})
        .with_dtype<float>()
        .with_device(device)
        .verify(k_norm_eps);
    TensorMatcher({C, R})  //
        .with_strides({CSS, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(cos_sin_cache);

    RuntimeCheck(P.unwrap() == 2, "packed_kv pair dimension must be 2, got ", P.unwrap());
    RuntimeCheck(R.unwrap() > 0 && R.unwrap() <= D.unwrap(), "rotary_dim must be in (0, head_dim].");
    RuntimeCheck(R.unwrap() % 2 == 0, "rotary_dim must be even.");
    RuntimeCheck(layer_start >= 0, "layer_start must be non-negative.");
    RuntimeCheck(
        static_cast<int64_t>(layer_start) + G.unwrap() <= L.unwrap(),
        "layer_start + group_count must not exceed num_layers.");

    const auto params = PromptPackedPostProjectionParams{
        .packed_kv = packed_kv.data_ptr(),
        .dst_k = dst_k.data_ptr(),
        .dst_v = dst_v.data_ptr(),
        .slot_ids = slot_ids.data_ptr(),
        .positions = positions.data_ptr(),
        .k_norm_weight = k_norm_weight.data_ptr(),
        .k_norm_eps = static_cast<const float*>(k_norm_eps.data_ptr()),
        .cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr()),
        .packed_token_stride = PKT.unwrap(),
        .packed_group_stride = PKG.unwrap(),
        .packed_pair_stride = PKP.unwrap(),
        .packed_head_stride = PKH.unwrap(),
        .dst_k_layer_stride = DKL.unwrap(),
        .dst_k_slot_stride = DKS.unwrap(),
        .dst_k_head_stride = DKH.unwrap(),
        .slot_ids_stride = SLOT.unwrap(),
        .positions_stride = POS.unwrap(),
        .k_norm_weight_layer_stride = NWL.unwrap(),
        .cos_sin_stride = CSS.unwrap(),
        .layer_start = static_cast<uint32_t>(layer_start),
        .group_count = static_cast<uint32_t>(G.unwrap()),
        .num_tokens = static_cast<uint32_t>(T.unwrap()),
        .num_heads = static_cast<uint32_t>(H.unwrap()),
        .head_dim = static_cast<uint32_t>(D.unwrap()),
        .rotary_dim = static_cast<uint32_t>(R.unwrap()),
        .half_rotary_dim = static_cast<uint32_t>(R.unwrap() / 2),
    };

    const auto total_rows =
        static_cast<uint64_t>(G.unwrap()) * static_cast<uint64_t>(T.unwrap()) * static_cast<uint64_t>(H.unwrap());
    const auto num_blocks = div_ceil(total_rows, static_cast<uint64_t>(kNumWarps));
    if (index_dtype.is_type<int32_t>()) {
      if (pos_dtype.is_type<int32_t>()) {
        LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
            .enable_pdl(kUsePDL)(kernel<int32_t, int32_t>, params);
      } else {
        LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
            .enable_pdl(kUsePDL)(kernel<int32_t, int64_t>, params);
      }
    } else {
      if (pos_dtype.is_type<int32_t>()) {
        LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
            .enable_pdl(kUsePDL)(kernel<int64_t, int32_t>, params);
      } else {
        LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
            .enable_pdl(kUsePDL)(kernel<int64_t, int64_t>, params);
      }
    }
  }
};

template <bool kUsePDL, typename DType>
struct DFlashCommitPackedPostProjectionKernel {
  template <typename IndexT, typename LenT, typename PosT>
  static constexpr auto kernel = dflash_commit_packed_post_projection<kUsePDL, DType, IndexT, LenT, PosT>;

  static void
  run(const tvm::ffi::TensorView packed_kv,
      const tvm::ffi::TensorView dst_k,
      const tvm::ffi::TensorView dst_v,
      const tvm::ffi::TensorView slot_ids_2d,
      const tvm::ffi::TensorView commit_lens,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView k_norm_weight,
      const tvm::ffi::TensorView k_norm_eps,
      const tvm::ffi::TensorView cos_sin_cache,
      int layer_start) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto BLK = SymbolicSize{"block_size"};
    auto G = SymbolicSize{"group_count"};
    auto P = SymbolicSize{"kv_pair"};
    auto H = SymbolicSize{"num_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto L = SymbolicSize{"num_layers"};
    auto S = SymbolicSize{"num_slots"};
    auto C = SymbolicSize{"cos_cache_len"};
    auto R = SymbolicSize{"rotary_dim"};
    auto PKB = SymbolicSize{"packed_batch_stride"};
    auto PKBL = SymbolicSize{"packed_block_stride"};
    auto PKG = SymbolicSize{"packed_group_stride"};
    auto PKP = SymbolicSize{"packed_pair_stride"};
    auto PKH = SymbolicSize{"packed_head_stride"};
    auto DKL = SymbolicSize{"dst_k_layer_stride"};
    auto DKS = SymbolicSize{"dst_k_slot_stride"};
    auto DKH = SymbolicSize{"dst_k_head_stride"};
    auto SIB = SymbolicSize{"slot_ids_batch_stride"};
    auto SIBLK = SymbolicSize{"slot_ids_block_stride"};
    auto CLS = SymbolicSize{"commit_lens_stride"};
    auto POSB = SymbolicSize{"positions_batch_stride"};
    auto POSBLK = SymbolicSize{"positions_block_stride"};
    auto NWL = SymbolicSize{"k_norm_weight_layer_stride"};
    auto CSS = SymbolicSize{"cos_sin_stride"};
    auto index_dtype = SymbolicDType{};
    auto len_dtype = SymbolicDType{};
    auto pos_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({B, BLK, G, P, H, D})  //
        .with_strides({PKB, PKBL, PKG, PKP, PKH, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(packed_kv);
    TensorMatcher({L, S, H, D})  //
        .with_strides({DKL, DKS, DKH, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(dst_k)
        .verify(dst_v);
    TensorMatcher({B, BLK})  //
        .with_strides({SIB, SIBLK})
        .with_dtype<int32_t, int64_t>(index_dtype)
        .with_device(device)
        .verify(slot_ids_2d);
    TensorMatcher({B})  //
        .with_strides({CLS})
        .with_dtype<int32_t, int64_t>(len_dtype)
        .with_device(device)
        .verify(commit_lens);
    TensorMatcher({B, BLK})  //
        .with_strides({POSB, POSBLK})
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device)
        .verify(positions);
    TensorMatcher({L, D})  //
        .with_strides({NWL, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k_norm_weight);
    TensorMatcher({L})  //
        .with_strides({1})
        .with_dtype<float>()
        .with_device(device)
        .verify(k_norm_eps);
    TensorMatcher({C, R})  //
        .with_strides({CSS, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(cos_sin_cache);

    RuntimeCheck(P.unwrap() == 2, "packed_kv pair dimension must be 2, got ", P.unwrap());
    RuntimeCheck(R.unwrap() > 0 && R.unwrap() <= D.unwrap(), "rotary_dim must be in (0, head_dim].");
    RuntimeCheck(R.unwrap() % 2 == 0, "rotary_dim must be even.");
    RuntimeCheck(layer_start >= 0, "layer_start must be non-negative.");
    RuntimeCheck(
        static_cast<int64_t>(layer_start) + G.unwrap() <= L.unwrap(),
        "layer_start + group_count must not exceed num_layers.");

    const auto params = CommitPackedPostProjectionParams{
        .packed_kv = packed_kv.data_ptr(),
        .dst_k = dst_k.data_ptr(),
        .dst_v = dst_v.data_ptr(),
        .slot_ids_2d = slot_ids_2d.data_ptr(),
        .commit_lens = commit_lens.data_ptr(),
        .positions = positions.data_ptr(),
        .k_norm_weight = k_norm_weight.data_ptr(),
        .k_norm_eps = static_cast<const float*>(k_norm_eps.data_ptr()),
        .cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr()),
        .packed_batch_stride = PKB.unwrap(),
        .packed_block_stride = PKBL.unwrap(),
        .packed_group_stride = PKG.unwrap(),
        .packed_pair_stride = PKP.unwrap(),
        .packed_head_stride = PKH.unwrap(),
        .dst_k_layer_stride = DKL.unwrap(),
        .dst_k_slot_stride = DKS.unwrap(),
        .dst_k_head_stride = DKH.unwrap(),
        .slot_ids_batch_stride = SIB.unwrap(),
        .slot_ids_block_stride = SIBLK.unwrap(),
        .commit_lens_stride = CLS.unwrap(),
        .positions_batch_stride = POSB.unwrap(),
        .positions_block_stride = POSBLK.unwrap(),
        .k_norm_weight_layer_stride = NWL.unwrap(),
        .cos_sin_stride = CSS.unwrap(),
        .layer_start = static_cast<uint32_t>(layer_start),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .block_size = static_cast<uint32_t>(BLK.unwrap()),
        .group_count = static_cast<uint32_t>(G.unwrap()),
        .num_heads = static_cast<uint32_t>(H.unwrap()),
        .head_dim = static_cast<uint32_t>(D.unwrap()),
        .rotary_dim = static_cast<uint32_t>(R.unwrap()),
        .half_rotary_dim = static_cast<uint32_t>(R.unwrap() / 2),
    };

    const auto total_rows = static_cast<uint64_t>(G.unwrap()) * static_cast<uint64_t>(B.unwrap()) *
                            static_cast<uint64_t>(BLK.unwrap()) * static_cast<uint64_t>(H.unwrap());
    const auto num_blocks = div_ceil(total_rows, static_cast<uint64_t>(kNumWarps));
    if (index_dtype.is_type<int32_t>()) {
      if (len_dtype.is_type<int32_t>()) {
        if (pos_dtype.is_type<int32_t>()) {
          LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int32_t, int32_t, int32_t>, params);
        } else {
          LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int32_t, int32_t, int64_t>, params);
        }
      } else {
        if (pos_dtype.is_type<int32_t>()) {
          LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int32_t, int64_t, int32_t>, params);
        } else {
          LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int32_t, int64_t, int64_t>, params);
        }
      }
    } else {
      if (len_dtype.is_type<int32_t>()) {
        if (pos_dtype.is_type<int32_t>()) {
          LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int64_t, int32_t, int32_t>, params);
        } else {
          LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int64_t, int32_t, int64_t>, params);
        }
      } else {
        if (pos_dtype.is_type<int32_t>()) {
          LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int64_t, int64_t, int32_t>, params);
        } else {
          LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int64_t, int64_t, int64_t>, params);
        }
      }
    }
  }
};

}  // namespace
