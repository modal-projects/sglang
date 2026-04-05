#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr uint32_t kNumWarps = 4;
constexpr uint32_t kThreadsPerBlock = kNumWarps * device::kWarpThreads;
constexpr int32_t kStatusActive = 1 << 0;
constexpr int32_t kGpuStopMask = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);

struct DFlashPrepareBlockParams {
  const void* __restrict__ committed_len;
  const void* __restrict__ reserved_len;
  const void* __restrict__ next_verified_id;
  const void* __restrict__ generation;
  const void* __restrict__ status_flags;
  const void* __restrict__ req_pool_indices;
  const void* __restrict__ req_generation;
  const void* __restrict__ req_to_token;

  void* __restrict__ active_mask_out;
  void* __restrict__ oob_flags_out;
  void* __restrict__ query_positions_out;
  void* __restrict__ query_slot_ids_out;
  void* __restrict__ query_input_ids_out;
  void* __restrict__ emit_ids_out;
  void* __restrict__ sample_indices_out;

  int64_t state_stride;
  int64_t status_stride;
  int64_t req_pool_stride;
  int64_t req_generation_stride;
  int64_t req_to_token_row_stride;
  int64_t req_to_token_col_stride;
  int64_t active_mask_stride;
  int64_t oob_flags_stride;
  int64_t query_positions_row_stride;
  int64_t query_positions_col_stride;
  int64_t query_slot_ids_row_stride;
  int64_t query_slot_ids_col_stride;
  int64_t query_input_ids_row_stride;
  int64_t emit_ids_row_stride;
  int64_t sample_indices_row_stride;
  int64_t sample_indices_col_stride;

  uint32_t bucket_bs;
  uint32_t block_size;
  uint32_t req_to_token_width;
  uint32_t sample_col_count;
  int64_t mask_token_id;
  int64_t dummy_slot_id;
};

template <typename ReqIndexT, typename StateT, typename TokenT>
__global__ void dflash_prepare_block_kernel(const DFlashPrepareBlockParams __grid_constant__ params) {
  using namespace device;
  const uint32_t warp_id = blockIdx.x * kNumWarps + threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  if (warp_id >= params.bucket_bs) return;

  ReqIndexT req_idx = 0;
  StateT req_generation = 0;
  if (lane_id == 0) {
    req_idx = *(static_cast<const ReqIndexT*>(params.req_pool_indices) + warp_id * params.req_pool_stride);
    req_generation = *(static_cast<const StateT*>(params.req_generation) + warp_id * params.req_generation_stride);
  }
  req_idx = __shfl_sync(0xFFFFFFFFu, req_idx, 0);
  req_generation = __shfl_sync(0xFFFFFFFFu, req_generation, 0);

  const auto state_offset = static_cast<int64_t>(req_idx) * params.state_stride;
  const auto status_offset = static_cast<int64_t>(req_idx) * params.status_stride;
  const auto committed_len = *(static_cast<const StateT*>(params.committed_len) + state_offset);
  const auto reserved_len = *(static_cast<const StateT*>(params.reserved_len) + state_offset);
  const auto next_verified_id = *(static_cast<const StateT*>(params.next_verified_id) + state_offset);
  const auto generation = *(static_cast<const StateT*>(params.generation) + state_offset);
  const auto status_flags = *(static_cast<const StateT*>(params.status_flags) + status_offset);

  const bool active = ((static_cast<int32_t>(status_flags) & kStatusActive) != 0) &&
                      ((static_cast<int32_t>(status_flags) & kGpuStopMask) == 0) && (generation == req_generation);

  if (lane_id == 0) {
    *(static_cast<int32_t*>(params.active_mask_out) + warp_id * params.active_mask_stride) = active ? 1 : 0;
  }
  if (!active) return;

  const int64_t last_pos = static_cast<int64_t>(committed_len) + static_cast<int64_t>(params.block_size) - 1;
  const bool oob =
      last_pos >= static_cast<int64_t>(reserved_len) || last_pos >= static_cast<int64_t>(params.req_to_token_width);
  if (lane_id == 0) {
    *(static_cast<int32_t*>(params.oob_flags_out) + warp_id * params.oob_flags_stride) = oob ? 1 : 0;
    if (!oob) {
      *(static_cast<StateT*>(params.query_input_ids_out) + warp_id * params.query_input_ids_row_stride) =
          next_verified_id;
      *(static_cast<StateT*>(params.emit_ids_out) + warp_id * params.emit_ids_row_stride) = next_verified_id;
    }
  }
  if (oob) return;

  for (uint32_t col = lane_id; col < params.block_size; col += kWarpThreads) {
    const auto logical_pos = static_cast<int64_t>(committed_len) + static_cast<int64_t>(col);
    const auto token = *(
        static_cast<const TokenT*>(params.req_to_token) +
        static_cast<int64_t>(req_idx) * params.req_to_token_row_stride + logical_pos * params.req_to_token_col_stride);
    *(static_cast<int64_t*>(params.query_positions_out) + warp_id * params.query_positions_row_stride +
      col * params.query_positions_col_stride) = logical_pos;
    *(static_cast<TokenT*>(params.query_slot_ids_out) + warp_id * params.query_slot_ids_row_stride +
      col * params.query_slot_ids_col_stride) = token;
  }
}

template <typename ReqIndexT, typename StateT, typename TokenT>
__global__ void dflash_prepare_block_fused_sample_kernel(const DFlashPrepareBlockParams __grid_constant__ params) {
  using namespace device;
  const uint32_t warp_id = blockIdx.x * kNumWarps + threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  if (warp_id >= params.bucket_bs) return;

  ReqIndexT req_idx = 0;
  StateT req_generation = 0;
  if (lane_id == 0) {
    req_idx = *(static_cast<const ReqIndexT*>(params.req_pool_indices) + warp_id * params.req_pool_stride);
    req_generation = *(static_cast<const StateT*>(params.req_generation) + warp_id * params.req_generation_stride);
  }
  req_idx = __shfl_sync(0xFFFFFFFFu, req_idx, 0);
  req_generation = __shfl_sync(0xFFFFFFFFu, req_generation, 0);

  const auto state_offset = static_cast<int64_t>(req_idx) * params.state_stride;
  const auto status_offset = static_cast<int64_t>(req_idx) * params.status_stride;
  const auto committed_len = *(static_cast<const StateT*>(params.committed_len) + state_offset);
  const auto reserved_len = *(static_cast<const StateT*>(params.reserved_len) + state_offset);
  const auto next_verified_id = *(static_cast<const StateT*>(params.next_verified_id) + state_offset);
  const auto generation = *(static_cast<const StateT*>(params.generation) + state_offset);
  const auto status_flags = *(static_cast<const StateT*>(params.status_flags) + status_offset);

  const bool active = ((static_cast<int32_t>(status_flags) & kStatusActive) != 0) &&
                      ((static_cast<int32_t>(status_flags) & kGpuStopMask) == 0) && (generation == req_generation);
  const int64_t last_pos = static_cast<int64_t>(committed_len) + static_cast<int64_t>(params.block_size) - 1;
  const bool oob = active && (last_pos >= static_cast<int64_t>(reserved_len) ||
                              last_pos >= static_cast<int64_t>(params.req_to_token_width));
  const bool valid = active && !oob;

  if (lane_id == 0) {
    *(static_cast<int32_t*>(params.active_mask_out) + warp_id * params.active_mask_stride) = active ? 1 : 0;
    *(static_cast<int32_t*>(params.oob_flags_out) + warp_id * params.oob_flags_stride) = oob ? 1 : 0;
  }

  const auto mask_token = static_cast<StateT>(params.mask_token_id);
  const auto zero_token = static_cast<StateT>(0);
  const auto dummy_slot = static_cast<TokenT>(params.dummy_slot_id);

  for (uint32_t col = lane_id; col < params.block_size; col += kWarpThreads) {
    const bool is_first = col == 0;
    const int64_t logical_pos = static_cast<int64_t>(committed_len) + static_cast<int64_t>(col);
    TokenT slot_id = dummy_slot;
    if (valid) {
      slot_id =
          *(static_cast<const TokenT*>(params.req_to_token) +
            static_cast<int64_t>(req_idx) * params.req_to_token_row_stride +
            logical_pos * params.req_to_token_col_stride);
    }
    *(static_cast<int64_t*>(params.query_positions_out) + warp_id * params.query_positions_row_stride +
      col * params.query_positions_col_stride) = valid ? logical_pos : 0;
    *(static_cast<TokenT*>(params.query_slot_ids_out) + warp_id * params.query_slot_ids_row_stride +
      col * params.query_slot_ids_col_stride) = slot_id;
    *(static_cast<StateT*>(params.query_input_ids_out) + warp_id * params.query_input_ids_row_stride +
      static_cast<int64_t>(col)) = (valid && is_first) ? next_verified_id : mask_token;
    *(static_cast<StateT*>(params.emit_ids_out) + warp_id * params.emit_ids_row_stride + static_cast<int64_t>(col)) =
        (valid && is_first) ? next_verified_id : zero_token;
  }

  for (uint32_t col = lane_id; col < params.sample_col_count; col += kWarpThreads) {
    *(static_cast<int32_t*>(params.sample_indices_out) + warp_id * params.sample_indices_row_stride +
      col * params.sample_indices_col_stride) = static_cast<int32_t>(warp_id * params.block_size + (col + 1));
  }
}

template <bool kUsePDL>
struct DFlashPrepareBlockKernel {
  template <typename ReqIndexT, typename StateT, typename TokenT>
  static constexpr auto kernel = dflash_prepare_block_kernel<ReqIndexT, StateT, TokenT>;

  template <typename ReqIndexT, typename TokenT>
  static auto get_kernel(const bool use_int32_state) {
    return use_int32_state ? kernel<ReqIndexT, int32_t, TokenT> : kernel<ReqIndexT, int64_t, TokenT>;
  }

  template <typename ReqIndexT>
  static auto get_kernel_with_token(const bool use_int32_state, const bool use_int32_token) {
    if (use_int32_token) {
      return get_kernel<ReqIndexT, int32_t>(use_int32_state);
    }
    return get_kernel<ReqIndexT, int64_t>(use_int32_state);
  }

  static void
  run(const tvm::ffi::TensorView committed_len,
      const tvm::ffi::TensorView reserved_len,
      const tvm::ffi::TensorView next_verified_id,
      const tvm::ffi::TensorView generation,
      const tvm::ffi::TensorView status_flags,
      const tvm::ffi::TensorView req_pool_indices,
      const tvm::ffi::TensorView req_generation,
      const tvm::ffi::TensorView req_to_token,
      const tvm::ffi::TensorView active_mask_out,
      const tvm::ffi::TensorView oob_flags_out,
      const tvm::ffi::TensorView query_positions_out,
      const tvm::ffi::TensorView query_slot_ids_out,
      const tvm::ffi::TensorView query_input_ids_out,
      const tvm::ffi::TensorView emit_ids_out) {
    using namespace host;

    auto R = SymbolicSize{"num_req_slots"};
    auto W = SymbolicSize{"req_to_token_width"};
    auto B = SymbolicSize{"bucket_bs"};
    auto T = SymbolicSize{"block_size"};

    auto state_stride = SymbolicSize{"state_stride"};
    auto req_pool_stride = SymbolicSize{"req_pool_stride"};
    auto req_generation_stride = SymbolicSize{"req_generation_stride"};
    auto req_to_token_row_stride = SymbolicSize{"req_to_token_row_stride"};
    auto req_to_token_col_stride = SymbolicSize{"req_to_token_col_stride"};
    auto active_mask_stride = SymbolicSize{"active_mask_stride"};
    auto oob_flags_stride = SymbolicSize{"oob_flags_stride"};
    auto query_positions_row_stride = SymbolicSize{"query_positions_row_stride"};
    auto query_positions_col_stride = SymbolicSize{"query_positions_col_stride"};
    auto query_slot_ids_row_stride = SymbolicSize{"query_slot_ids_row_stride"};
    auto query_slot_ids_col_stride = SymbolicSize{"query_slot_ids_col_stride"};
    auto query_input_ids_row_stride = SymbolicSize{"query_input_ids_row_stride"};
    auto emit_ids_row_stride = SymbolicSize{"emit_ids_row_stride"};

    auto state_dtype = SymbolicDType{};
    auto req_index_dtype = SymbolicDType{};
    auto token_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({R})  //
        .with_strides({state_stride})
        .with_dtype<int32_t, int64_t>(state_dtype)
        .with_device(device)
        .verify(committed_len)
        .verify(reserved_len)
        .verify(next_verified_id)
        .verify(generation)
        .verify(status_flags);
    TensorMatcher({B})  //
        .with_strides({req_pool_stride})
        .with_dtype<int32_t, int64_t>(req_index_dtype)
        .with_device(device)
        .verify(req_pool_indices);
    TensorMatcher({B})  //
        .with_strides({req_generation_stride})
        .with_dtype<int32_t, int64_t>(state_dtype)
        .with_device(device)
        .verify(req_generation);
    TensorMatcher({R, W})  //
        .with_strides({req_to_token_row_stride, req_to_token_col_stride})
        .with_dtype<int32_t, int64_t>(token_dtype)
        .with_device(device)
        .verify(req_to_token);
    TensorMatcher({B})  //
        .with_strides({active_mask_stride})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(active_mask_out);
    TensorMatcher({B})  //
        .with_strides({oob_flags_stride})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(oob_flags_out);
    TensorMatcher({B, T})  //
        .with_strides({query_positions_row_stride, query_positions_col_stride})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(query_positions_out);
    TensorMatcher({B, T})  //
        .with_strides({query_slot_ids_row_stride, query_slot_ids_col_stride})
        .with_dtype<int32_t, int64_t>(token_dtype)
        .with_device(device)
        .verify(query_slot_ids_out);
    TensorMatcher({B, T})  //
        .with_strides({query_input_ids_row_stride, 1})
        .with_dtype<int32_t, int64_t>(state_dtype)
        .with_device(device)
        .verify(query_input_ids_out);
    TensorMatcher({B, T})  //
        .with_strides({emit_ids_row_stride, 1})
        .with_dtype<int32_t, int64_t>(state_dtype)
        .with_device(device)
        .verify(emit_ids_out);

    RuntimeCheck(T.unwrap() > 0, "block_size must be positive, got ", T.unwrap());

    const auto params = DFlashPrepareBlockParams{
        .committed_len = committed_len.data_ptr(),
        .reserved_len = reserved_len.data_ptr(),
        .next_verified_id = next_verified_id.data_ptr(),
        .generation = generation.data_ptr(),
        .status_flags = status_flags.data_ptr(),
        .req_pool_indices = req_pool_indices.data_ptr(),
        .req_generation = req_generation.data_ptr(),
        .req_to_token = req_to_token.data_ptr(),
        .active_mask_out = active_mask_out.data_ptr(),
        .oob_flags_out = oob_flags_out.data_ptr(),
        .query_positions_out = query_positions_out.data_ptr(),
        .query_slot_ids_out = query_slot_ids_out.data_ptr(),
        .query_input_ids_out = query_input_ids_out.data_ptr(),
        .emit_ids_out = emit_ids_out.data_ptr(),
        .state_stride = state_stride.unwrap(),
        .status_stride = state_stride.unwrap(),
        .req_pool_stride = req_pool_stride.unwrap(),
        .req_generation_stride = req_generation_stride.unwrap(),
        .req_to_token_row_stride = req_to_token_row_stride.unwrap(),
        .req_to_token_col_stride = req_to_token_col_stride.unwrap(),
        .active_mask_stride = active_mask_stride.unwrap(),
        .oob_flags_stride = oob_flags_stride.unwrap(),
        .query_positions_row_stride = query_positions_row_stride.unwrap(),
        .query_positions_col_stride = query_positions_col_stride.unwrap(),
        .query_slot_ids_row_stride = query_slot_ids_row_stride.unwrap(),
        .query_slot_ids_col_stride = query_slot_ids_col_stride.unwrap(),
        .query_input_ids_row_stride = query_input_ids_row_stride.unwrap(),
        .emit_ids_row_stride = emit_ids_row_stride.unwrap(),
        .bucket_bs = static_cast<uint32_t>(B.unwrap()),
        .block_size = static_cast<uint32_t>(T.unwrap()),
        .req_to_token_width = static_cast<uint32_t>(W.unwrap()),
    };

    const bool use_int32_state = state_dtype.is_type<int32_t>();
    const bool use_int32_req_index = req_index_dtype.is_type<int32_t>();
    const bool use_int32_token = token_dtype.is_type<int32_t>();

    const auto selected_kernel = [&]() {
      if (use_int32_req_index) {
        return get_kernel_with_token<int32_t>(use_int32_state, use_int32_token);
      }
      return get_kernel_with_token<int64_t>(use_int32_state, use_int32_token);
    }();

    const auto num_blocks = div_ceil(static_cast<uint64_t>(B.unwrap()), static_cast<uint64_t>(kNumWarps));
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
        .enable_pdl(kUsePDL)(selected_kernel, params);
  }
};

template <bool kUsePDL>
struct DFlashPrepareBlockFusedSampleKernel {
  template <typename ReqIndexT, typename StateT, typename TokenT>
  static constexpr auto kernel = dflash_prepare_block_fused_sample_kernel<ReqIndexT, StateT, TokenT>;

  template <typename ReqIndexT, typename TokenT>
  static auto get_kernel(const bool use_int32_state) {
    return use_int32_state ? kernel<ReqIndexT, int32_t, TokenT> : kernel<ReqIndexT, int64_t, TokenT>;
  }

  template <typename ReqIndexT>
  static auto get_kernel_with_token(const bool use_int32_state, const bool use_int32_token) {
    if (use_int32_token) {
      return get_kernel<ReqIndexT, int32_t>(use_int32_state);
    }
    return get_kernel<ReqIndexT, int64_t>(use_int32_state);
  }

  static void
  run(const tvm::ffi::TensorView committed_len,
      const tvm::ffi::TensorView reserved_len,
      const tvm::ffi::TensorView next_verified_id,
      const tvm::ffi::TensorView generation,
      const tvm::ffi::TensorView status_flags,
      const tvm::ffi::TensorView req_pool_indices,
      const tvm::ffi::TensorView req_generation,
      const tvm::ffi::TensorView req_to_token,
      const tvm::ffi::TensorView active_mask_out,
      const tvm::ffi::TensorView oob_flags_out,
      const tvm::ffi::TensorView query_positions_out,
      const tvm::ffi::TensorView query_slot_ids_out,
      const tvm::ffi::TensorView query_input_ids_out,
      const tvm::ffi::TensorView emit_ids_out,
      const tvm::ffi::TensorView sample_indices_out,
      int64_t mask_token_id,
      int64_t dummy_slot_id) {
    using namespace host;

    auto R = SymbolicSize{"num_req_slots"};
    auto W = SymbolicSize{"req_to_token_width"};
    auto B = SymbolicSize{"bucket_bs"};
    auto T = SymbolicSize{"block_size"};
    auto S = SymbolicSize{"sample_cols"};

    auto state_stride = SymbolicSize{"state_stride"};
    auto req_pool_stride = SymbolicSize{"req_pool_stride"};
    auto req_generation_stride = SymbolicSize{"req_generation_stride"};
    auto req_to_token_row_stride = SymbolicSize{"req_to_token_row_stride"};
    auto req_to_token_col_stride = SymbolicSize{"req_to_token_col_stride"};
    auto active_mask_stride = SymbolicSize{"active_mask_stride"};
    auto oob_flags_stride = SymbolicSize{"oob_flags_stride"};
    auto query_positions_row_stride = SymbolicSize{"query_positions_row_stride"};
    auto query_positions_col_stride = SymbolicSize{"query_positions_col_stride"};
    auto query_slot_ids_row_stride = SymbolicSize{"query_slot_ids_row_stride"};
    auto query_slot_ids_col_stride = SymbolicSize{"query_slot_ids_col_stride"};
    auto query_input_ids_row_stride = SymbolicSize{"query_input_ids_row_stride"};
    auto emit_ids_row_stride = SymbolicSize{"emit_ids_row_stride"};
    auto sample_indices_row_stride = SymbolicSize{"sample_indices_row_stride"};
    auto sample_indices_col_stride = SymbolicSize{"sample_indices_col_stride"};

    auto state_dtype = SymbolicDType{};
    auto req_index_dtype = SymbolicDType{};
    auto token_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({R})  //
        .with_strides({state_stride})
        .with_dtype<int32_t, int64_t>(state_dtype)
        .with_device(device)
        .verify(committed_len)
        .verify(reserved_len)
        .verify(next_verified_id)
        .verify(generation)
        .verify(status_flags);
    TensorMatcher({B})  //
        .with_strides({req_pool_stride})
        .with_dtype<int32_t, int64_t>(req_index_dtype)
        .with_device(device)
        .verify(req_pool_indices);
    TensorMatcher({B})  //
        .with_strides({req_generation_stride})
        .with_dtype<int32_t, int64_t>(state_dtype)
        .with_device(device)
        .verify(req_generation);
    TensorMatcher({R, W})  //
        .with_strides({req_to_token_row_stride, req_to_token_col_stride})
        .with_dtype<int32_t, int64_t>(token_dtype)
        .with_device(device)
        .verify(req_to_token);
    TensorMatcher({B})  //
        .with_strides({active_mask_stride})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(active_mask_out);
    TensorMatcher({B})  //
        .with_strides({oob_flags_stride})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(oob_flags_out);
    TensorMatcher({B, T})  //
        .with_strides({query_positions_row_stride, query_positions_col_stride})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(query_positions_out);
    TensorMatcher({B, T})  //
        .with_strides({query_slot_ids_row_stride, query_slot_ids_col_stride})
        .with_dtype<int32_t, int64_t>(token_dtype)
        .with_device(device)
        .verify(query_slot_ids_out);
    TensorMatcher({B, T})  //
        .with_strides({query_input_ids_row_stride, 1})
        .with_dtype<int32_t, int64_t>(state_dtype)
        .with_device(device)
        .verify(query_input_ids_out);
    TensorMatcher({B, T})  //
        .with_strides({emit_ids_row_stride, 1})
        .with_dtype<int32_t, int64_t>(state_dtype)
        .with_device(device)
        .verify(emit_ids_out);
    TensorMatcher({B, S})  //
        .with_strides({sample_indices_row_stride, sample_indices_col_stride})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(sample_indices_out);

    RuntimeCheck(T.unwrap() > 0, "block_size must be positive, got ", T.unwrap());
    RuntimeCheck(
        S.unwrap() == std::max<int64_t>(T.unwrap() - 1, 0),
        "sample_indices_out second dimension must equal block_size - 1.");

    const auto params = DFlashPrepareBlockParams{
        .committed_len = committed_len.data_ptr(),
        .reserved_len = reserved_len.data_ptr(),
        .next_verified_id = next_verified_id.data_ptr(),
        .generation = generation.data_ptr(),
        .status_flags = status_flags.data_ptr(),
        .req_pool_indices = req_pool_indices.data_ptr(),
        .req_generation = req_generation.data_ptr(),
        .req_to_token = req_to_token.data_ptr(),
        .active_mask_out = active_mask_out.data_ptr(),
        .oob_flags_out = oob_flags_out.data_ptr(),
        .query_positions_out = query_positions_out.data_ptr(),
        .query_slot_ids_out = query_slot_ids_out.data_ptr(),
        .query_input_ids_out = query_input_ids_out.data_ptr(),
        .emit_ids_out = emit_ids_out.data_ptr(),
        .sample_indices_out = sample_indices_out.data_ptr(),
        .state_stride = state_stride.unwrap(),
        .status_stride = state_stride.unwrap(),
        .req_pool_stride = req_pool_stride.unwrap(),
        .req_generation_stride = req_generation_stride.unwrap(),
        .req_to_token_row_stride = req_to_token_row_stride.unwrap(),
        .req_to_token_col_stride = req_to_token_col_stride.unwrap(),
        .active_mask_stride = active_mask_stride.unwrap(),
        .oob_flags_stride = oob_flags_stride.unwrap(),
        .query_positions_row_stride = query_positions_row_stride.unwrap(),
        .query_positions_col_stride = query_positions_col_stride.unwrap(),
        .query_slot_ids_row_stride = query_slot_ids_row_stride.unwrap(),
        .query_slot_ids_col_stride = query_slot_ids_col_stride.unwrap(),
        .query_input_ids_row_stride = query_input_ids_row_stride.unwrap(),
        .emit_ids_row_stride = emit_ids_row_stride.unwrap(),
        .sample_indices_row_stride = sample_indices_row_stride.unwrap(),
        .sample_indices_col_stride = sample_indices_col_stride.unwrap(),
        .bucket_bs = static_cast<uint32_t>(B.unwrap()),
        .block_size = static_cast<uint32_t>(T.unwrap()),
        .req_to_token_width = static_cast<uint32_t>(W.unwrap()),
        .sample_col_count = static_cast<uint32_t>(S.unwrap()),
        .mask_token_id = mask_token_id,
        .dummy_slot_id = dummy_slot_id,
    };

    const bool use_int32_state = state_dtype.is_type<int32_t>();
    const bool use_int32_req_index = req_index_dtype.is_type<int32_t>();
    const bool use_int32_token = token_dtype.is_type<int32_t>();

    const auto selected_kernel = [&]() {
      if (use_int32_req_index) {
        return get_kernel_with_token<int32_t>(use_int32_state, use_int32_token);
      }
      return get_kernel_with_token<int64_t>(use_int32_state, use_int32_token);
    }();

    const auto num_blocks = div_ceil(static_cast<uint64_t>(B.unwrap()), static_cast<uint64_t>(kNumWarps));
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
        .enable_pdl(kUsePDL)(selected_kernel, params);
  }
};

}  // namespace
