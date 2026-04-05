#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr uint32_t kThreadsPerBlock = 256;
constexpr int32_t kStatusActive = 1 << 0;
constexpr int32_t kEosFinishedFlags = (1 << 1) | (1 << 2);
constexpr int32_t kStopFinishedFlags = (1 << 4) | (1 << 2);
constexpr int32_t kGpuStopMask = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);

struct DFlashAcceptPublishParams {
  void* __restrict__ committed_len;
  const void* __restrict__ reserved_len;
  void* __restrict__ next_verified_id;
  const void* __restrict__ generation;
  void* __restrict__ status_flags;

  const void* __restrict__ req_pool_indices;
  const void* __restrict__ req_generation;
  const void* __restrict__ emit_ids;
  const void* __restrict__ target_top1;
  const int32_t* __restrict__ active_mask;
  const void* __restrict__ eos_ids;
  const void* __restrict__ stop_ids;

  int32_t* __restrict__ accept_lens_out;
  int32_t* __restrict__ commit_lens_out;
  void* __restrict__ bonus_ids_out;
  int32_t* __restrict__ gpu_stop_flags_out;
  int32_t* __restrict__ oob_flags_out;

  int64_t state_stride;
  int64_t req_pool_stride;
  int64_t req_generation_stride;
  int64_t emit_row_stride;
  int64_t emit_col_stride;
  int64_t target_row_stride;
  int64_t target_col_stride;
  int64_t active_stride;
  int64_t eos_stride;
  int64_t stop_stride;
  int64_t accept_stride;
  int64_t commit_stride;
  int64_t bonus_stride;
  int64_t flags_stride;
  int64_t oob_stride;

  uint32_t batch_size;
  uint32_t block_size;
  uint32_t eos_count;
  uint32_t stop_count;
};

template <typename ReqIndexT, typename StateT, typename TokenT>
__global__ void dflash_accept_publish_kernel(const DFlashAcceptPublishParams __grid_constant__ params) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= params.batch_size) return;

  const bool active = *(params.active_mask + row * params.active_stride) != 0;
  if (!active) {
    *(params.accept_lens_out + row * params.accept_stride) = 0;
    *(params.commit_lens_out + row * params.commit_stride) = 0;
    *(static_cast<TokenT*>(params.bonus_ids_out) + row * params.bonus_stride) = 0;
    *(params.gpu_stop_flags_out + row * params.flags_stride) = 0;
    *(params.oob_flags_out + row * params.oob_stride) = 0;
    return;
  }

  int32_t accept_len = 0;
  for (uint32_t col = 0; col + 1 < params.block_size; ++col) {
    const auto emit_id =
        *(static_cast<const TokenT*>(params.emit_ids) + row * params.emit_row_stride +
          (col + 1) * params.emit_col_stride);
    const auto target_id =
        *(static_cast<const TokenT*>(params.target_top1) + row * params.target_row_stride +
          col * params.target_col_stride);
    if (emit_id != target_id) break;
    ++accept_len;
  }

  const int32_t commit_len = accept_len + 1;
  const auto bonus_id =
      *(static_cast<const TokenT*>(params.target_top1) + row * params.target_row_stride +
        static_cast<int64_t>(accept_len) * params.target_col_stride);

  int32_t flags = 0;
  for (int32_t col = 0; col < commit_len; ++col) {
    const auto token =
        *(static_cast<const TokenT*>(params.emit_ids) + row * params.emit_row_stride +
          static_cast<int64_t>(col) * params.emit_col_stride);
    for (uint32_t i = 0; i < params.eos_count; ++i) {
      const auto eos_id = *(static_cast<const TokenT*>(params.eos_ids) + static_cast<int64_t>(i) * params.eos_stride);
      if (token == eos_id) {
        flags |= kEosFinishedFlags;
        break;
      }
    }
    for (uint32_t i = 0; i < params.stop_count; ++i) {
      const auto stop_id =
          *(static_cast<const TokenT*>(params.stop_ids) + static_cast<int64_t>(i) * params.stop_stride);
      if (token == stop_id) {
        flags |= kStopFinishedFlags;
        break;
      }
    }
  }

  *(params.accept_lens_out + row * params.accept_stride) = accept_len;
  *(params.commit_lens_out + row * params.commit_stride) = commit_len;
  *(static_cast<TokenT*>(params.bonus_ids_out) + row * params.bonus_stride) = bonus_id;
  *(params.gpu_stop_flags_out + row * params.flags_stride) = flags;

  const auto req_idx = *(static_cast<const ReqIndexT*>(params.req_pool_indices) + row * params.req_pool_stride);
  const auto req_generation = *(static_cast<const StateT*>(params.req_generation) + row * params.req_generation_stride);
  const auto generation =
      *(static_cast<const StateT*>(params.generation) + static_cast<int64_t>(req_idx) * params.state_stride);
  if (generation != req_generation) {
    *(params.oob_flags_out + row * params.oob_stride) = 0;
    return;
  }

  auto committed_ptr = static_cast<StateT*>(params.committed_len) + static_cast<int64_t>(req_idx) * params.state_stride;
  const auto reserved_len =
      *(static_cast<const StateT*>(params.reserved_len) + static_cast<int64_t>(req_idx) * params.state_stride);
  const auto new_committed_len = *committed_ptr + static_cast<StateT>(commit_len);
  const bool oob = new_committed_len > reserved_len;
  *(params.oob_flags_out + row * params.oob_stride) = oob ? 1 : 0;
  if (oob) return;

  *committed_ptr = new_committed_len;
  *(static_cast<StateT*>(params.next_verified_id) + static_cast<int64_t>(req_idx) * params.state_stride) =
      static_cast<StateT>(bonus_id);

  auto status_ptr = static_cast<StateT*>(params.status_flags) + static_cast<int64_t>(req_idx) * params.state_stride;
  auto new_status = static_cast<int32_t>(*status_ptr) | flags;
  if ((new_status & kGpuStopMask) != 0) {
    new_status &= ~kStatusActive;
  }
  *status_ptr = static_cast<StateT>(new_status);
}

template <bool kUsePDL>
struct DFlashAcceptPublishKernel {
  template <typename ReqIndexT, typename StateT, typename TokenT>
  static constexpr auto kernel = dflash_accept_publish_kernel<ReqIndexT, StateT, TokenT>;

  static void
  run(const tvm::ffi::TensorView committed_len,
      const tvm::ffi::TensorView reserved_len,
      const tvm::ffi::TensorView next_verified_id,
      const tvm::ffi::TensorView generation,
      const tvm::ffi::TensorView status_flags,
      const tvm::ffi::TensorView req_pool_indices,
      const tvm::ffi::TensorView req_generation,
      const tvm::ffi::TensorView emit_ids,
      const tvm::ffi::TensorView target_top1,
      const tvm::ffi::TensorView active_mask,
      const tvm::ffi::TensorView eos_ids,
      const tvm::ffi::TensorView stop_ids,
      const tvm::ffi::TensorView accept_lens_out,
      const tvm::ffi::TensorView commit_lens_out,
      const tvm::ffi::TensorView bonus_ids_out,
      const tvm::ffi::TensorView gpu_stop_flags_out,
      const tvm::ffi::TensorView oob_flags_out) {
    using namespace host;

    auto R = SymbolicSize{"num_req_slots"};
    auto B = SymbolicSize{"batch_size"};
    auto T = SymbolicSize{"block_size"};
    auto E = SymbolicSize{"eos_count"};
    auto S = SymbolicSize{"stop_count"};
    auto state_stride = SymbolicSize{"state_stride"};
    auto req_pool_stride = SymbolicSize{"req_pool_stride"};
    auto req_generation_stride = SymbolicSize{"req_generation_stride"};
    auto emit_row_stride = SymbolicSize{"emit_row_stride"};
    auto emit_col_stride = SymbolicSize{"emit_col_stride"};
    auto target_row_stride = SymbolicSize{"target_row_stride"};
    auto target_col_stride = SymbolicSize{"target_col_stride"};
    auto active_stride = SymbolicSize{"active_stride"};
    auto eos_stride = SymbolicSize{"eos_stride"};
    auto stop_stride = SymbolicSize{"stop_stride"};
    auto accept_stride = SymbolicSize{"accept_stride"};
    auto commit_stride = SymbolicSize{"commit_stride"};
    auto bonus_stride = SymbolicSize{"bonus_stride"};
    auto flags_stride = SymbolicSize{"flags_stride"};
    auto oob_stride = SymbolicSize{"oob_stride"};
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
    TensorMatcher({B, T})  //
        .with_strides({emit_row_stride, emit_col_stride})
        .with_dtype<int32_t, int64_t>(token_dtype)
        .with_device(device)
        .verify(emit_ids);
    TensorMatcher({B, T})  //
        .with_strides({target_row_stride, target_col_stride})
        .with_dtype<int32_t, int64_t>(token_dtype)
        .with_device(device)
        .verify(target_top1);
    TensorMatcher({B})  //
        .with_strides({active_stride})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(active_mask)
        .verify(accept_lens_out)
        .verify(commit_lens_out)
        .verify(gpu_stop_flags_out)
        .verify(oob_flags_out);
    TensorMatcher({E})  //
        .with_strides({eos_stride})
        .with_dtype<int32_t, int64_t>(token_dtype)
        .with_device(device)
        .verify(eos_ids);
    TensorMatcher({S})  //
        .with_strides({stop_stride})
        .with_dtype<int32_t, int64_t>(token_dtype)
        .with_device(device)
        .verify(stop_ids);
    TensorMatcher({B})  //
        .with_strides({bonus_stride})
        .with_dtype<int32_t, int64_t>(token_dtype)
        .with_device(device)
        .verify(bonus_ids_out);

    RuntimeCheck(T.unwrap() > 0, "block_size must be positive, got ", T.unwrap());

    const auto params = DFlashAcceptPublishParams{
        .committed_len = committed_len.data_ptr(),
        .reserved_len = reserved_len.data_ptr(),
        .next_verified_id = next_verified_id.data_ptr(),
        .generation = generation.data_ptr(),
        .status_flags = status_flags.data_ptr(),
        .req_pool_indices = req_pool_indices.data_ptr(),
        .req_generation = req_generation.data_ptr(),
        .emit_ids = emit_ids.data_ptr(),
        .target_top1 = target_top1.data_ptr(),
        .active_mask = static_cast<const int32_t*>(active_mask.data_ptr()),
        .eos_ids = eos_ids.data_ptr(),
        .stop_ids = stop_ids.data_ptr(),
        .accept_lens_out = static_cast<int32_t*>(accept_lens_out.data_ptr()),
        .commit_lens_out = static_cast<int32_t*>(commit_lens_out.data_ptr()),
        .bonus_ids_out = bonus_ids_out.data_ptr(),
        .gpu_stop_flags_out = static_cast<int32_t*>(gpu_stop_flags_out.data_ptr()),
        .oob_flags_out = static_cast<int32_t*>(oob_flags_out.data_ptr()),
        .state_stride = state_stride.unwrap(),
        .req_pool_stride = req_pool_stride.unwrap(),
        .req_generation_stride = req_generation_stride.unwrap(),
        .emit_row_stride = emit_row_stride.unwrap(),
        .emit_col_stride = emit_col_stride.unwrap(),
        .target_row_stride = target_row_stride.unwrap(),
        .target_col_stride = target_col_stride.unwrap(),
        .active_stride = active_stride.unwrap(),
        .eos_stride = eos_stride.unwrap(),
        .stop_stride = stop_stride.unwrap(),
        .accept_stride = active_stride.unwrap(),
        .commit_stride = active_stride.unwrap(),
        .bonus_stride = bonus_stride.unwrap(),
        .flags_stride = active_stride.unwrap(),
        .oob_stride = active_stride.unwrap(),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .block_size = static_cast<uint32_t>(T.unwrap()),
        .eos_count = static_cast<uint32_t>(E.unwrap()),
        .stop_count = static_cast<uint32_t>(S.unwrap()),
    };

    const auto num_blocks = div_ceil(static_cast<uint64_t>(B.unwrap()), static_cast<uint64_t>(kThreadsPerBlock));
    const bool use_int32_state = state_dtype.is_type<int32_t>();
    const bool use_int32_token = token_dtype.is_type<int32_t>();
    if (req_index_dtype.is_type<int32_t>()) {
      if (use_int32_state) {
        if (use_int32_token) {
          LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int32_t, int32_t, int32_t>, params);
        } else {
          LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int32_t, int32_t, int64_t>, params);
        }
      } else {
        if (use_int32_token) {
          LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int32_t, int64_t, int32_t>, params);
        } else {
          LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int32_t, int64_t, int64_t>, params);
        }
      }
    } else {
      if (use_int32_state) {
        if (use_int32_token) {
          LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int64_t, int32_t, int32_t>, params);
        } else {
          LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int64_t, int32_t, int64_t>, params);
        }
      } else {
        if (use_int32_token) {
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
