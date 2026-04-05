#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr uint32_t kThreadsPerBlock = 256;
constexpr int32_t kStatusActive = 1 << 0;
constexpr int32_t kGpuStopMask = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);

struct DFlashPublishStateParams {
  void* __restrict__ committed_len;
  const void* __restrict__ reserved_len;
  void* __restrict__ next_verified_id;
  const void* __restrict__ generation;
  void* __restrict__ status_flags;
  const void* __restrict__ req_pool_indices;
  const void* __restrict__ req_generation;
  const void* __restrict__ commit_lens;
  const void* __restrict__ bonus_ids;
  const void* __restrict__ gpu_stop_flags;
  int32_t* __restrict__ oob_flags_out;

  int64_t state_stride;
  int64_t req_pool_stride;
  int64_t row_stride;
  int64_t oob_stride;
  uint32_t batch_size;
};

template <typename ReqIndexT, typename StateT>
__global__ void dflash_publish_state_kernel(const DFlashPublishStateParams __grid_constant__ params) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= params.batch_size) return;

  const auto req_idx = *(static_cast<const ReqIndexT*>(params.req_pool_indices) + row * params.req_pool_stride);
  const auto req_generation = *(static_cast<const StateT*>(params.req_generation) + row * params.row_stride);
  const auto commit_len = *(static_cast<const StateT*>(params.commit_lens) + row * params.row_stride);
  if (commit_len <= 0) {
    *(params.oob_flags_out + row * params.oob_stride) = 0;
    return;
  }

  const auto generation =
      *(static_cast<const StateT*>(params.generation) + static_cast<int64_t>(req_idx) * params.state_stride);
  if (generation != req_generation) {
    *(params.oob_flags_out + row * params.oob_stride) = 0;
    return;
  }

  auto committed_ptr = static_cast<StateT*>(params.committed_len) + static_cast<int64_t>(req_idx) * params.state_stride;
  const auto reserved_len =
      *(static_cast<const StateT*>(params.reserved_len) + static_cast<int64_t>(req_idx) * params.state_stride);
  const auto new_committed_len = *committed_ptr + commit_len;
  const bool oob = new_committed_len > reserved_len;
  *(params.oob_flags_out + row * params.oob_stride) = oob ? 1 : 0;
  if (oob) return;

  *committed_ptr = new_committed_len;
  *(static_cast<StateT*>(params.next_verified_id) + static_cast<int64_t>(req_idx) * params.state_stride) =
      *(static_cast<const StateT*>(params.bonus_ids) + row * params.row_stride);

  auto status_ptr = static_cast<StateT*>(params.status_flags) + static_cast<int64_t>(req_idx) * params.state_stride;
  auto new_status =
      static_cast<int32_t>(*status_ptr) |
      static_cast<int32_t>(*(static_cast<const StateT*>(params.gpu_stop_flags) + row * params.row_stride));
  if ((new_status & kGpuStopMask) != 0) {
    new_status &= ~kStatusActive;
  }
  *status_ptr = static_cast<StateT>(new_status);
}

template <bool kUsePDL>
struct DFlashPublishStateKernel {
  template <typename ReqIndexT, typename StateT>
  static constexpr auto kernel = dflash_publish_state_kernel<ReqIndexT, StateT>;

  static void
  run(const tvm::ffi::TensorView committed_len,
      const tvm::ffi::TensorView reserved_len,
      const tvm::ffi::TensorView next_verified_id,
      const tvm::ffi::TensorView generation,
      const tvm::ffi::TensorView status_flags,
      const tvm::ffi::TensorView req_pool_indices,
      const tvm::ffi::TensorView req_generation,
      const tvm::ffi::TensorView commit_lens,
      const tvm::ffi::TensorView bonus_ids,
      const tvm::ffi::TensorView gpu_stop_flags,
      const tvm::ffi::TensorView oob_flags_out) {
    using namespace host;

    auto R = SymbolicSize{"num_req_slots"};
    auto B = SymbolicSize{"batch_size"};
    auto state_stride = SymbolicSize{"state_stride"};
    auto req_pool_stride = SymbolicSize{"req_pool_stride"};
    auto row_stride = SymbolicSize{"row_stride"};
    auto oob_stride = SymbolicSize{"oob_stride"};
    auto state_dtype = SymbolicDType{};
    auto req_index_dtype = SymbolicDType{};
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
        .with_strides({row_stride})
        .with_dtype<int32_t, int64_t>(state_dtype)
        .with_device(device)
        .verify(req_generation)
        .verify(commit_lens)
        .verify(bonus_ids)
        .verify(gpu_stop_flags);
    TensorMatcher({B})  //
        .with_strides({oob_stride})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(oob_flags_out);

    const auto params = DFlashPublishStateParams{
        .committed_len = committed_len.data_ptr(),
        .reserved_len = reserved_len.data_ptr(),
        .next_verified_id = next_verified_id.data_ptr(),
        .generation = generation.data_ptr(),
        .status_flags = status_flags.data_ptr(),
        .req_pool_indices = req_pool_indices.data_ptr(),
        .req_generation = req_generation.data_ptr(),
        .commit_lens = commit_lens.data_ptr(),
        .bonus_ids = bonus_ids.data_ptr(),
        .gpu_stop_flags = gpu_stop_flags.data_ptr(),
        .oob_flags_out = static_cast<int32_t*>(oob_flags_out.data_ptr()),
        .state_stride = state_stride.unwrap(),
        .req_pool_stride = req_pool_stride.unwrap(),
        .row_stride = row_stride.unwrap(),
        .oob_stride = oob_stride.unwrap(),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
    };

    const auto num_blocks = div_ceil(static_cast<uint64_t>(B.unwrap()), static_cast<uint64_t>(kThreadsPerBlock));
    const bool use_int32_state = state_dtype.is_type<int32_t>();
    if (req_index_dtype.is_type<int32_t>()) {
      if (use_int32_state) {
        LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
            .enable_pdl(kUsePDL)(kernel<int32_t, int32_t>, params);
      } else {
        LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
            .enable_pdl(kUsePDL)(kernel<int32_t, int64_t>, params);
      }
    } else {
      if (use_int32_state) {
        LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
            .enable_pdl(kUsePDL)(kernel<int64_t, int32_t>, params);
      } else {
        LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
            .enable_pdl(kUsePDL)(kernel<int64_t, int64_t>, params);
      }
    }
  }
};

}  // namespace
