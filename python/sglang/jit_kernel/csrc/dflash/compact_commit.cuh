#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr uint32_t kCompactThreadsPerBlock = 256;

struct DFlashCompactCommitParams {
  const void* verify_hidden;
  const void* positions;
  const void* slot_ids_2d;
  const void* commit_lens;
  const int32_t* row_offsets;

  void* hidden_out;
  void* positions_out;
  void* slot_ids_out;

  int64_t hidden_batch_stride;
  int64_t hidden_block_stride;
  int64_t hidden_col_stride;
  int64_t positions_batch_stride;
  int64_t positions_block_stride;
  int64_t slot_ids_batch_stride;
  int64_t slot_ids_block_stride;
  int64_t commit_lens_stride;
  int64_t row_offsets_stride;
  int64_t hidden_out_row_stride;
  int64_t positions_out_stride;
  int64_t slot_ids_out_stride;

  uint32_t batch_size;
  uint32_t block_size;
  uint32_t hidden_size;
  uint32_t total_rows;
};

template <typename DType, typename PosT, typename IndexT, typename LenT>
__global__ void dflash_compact_commit_kernel(const DFlashCompactCommitParams __grid_constant__ params) {
  const uint32_t row_idx = blockIdx.x;
  if (row_idx >= params.total_rows) return;

  const uint32_t batch_idx = row_idx / params.block_size;
  const uint32_t block_idx = row_idx % params.block_size;
  const auto keep = *(static_cast<const LenT*>(params.commit_lens) + batch_idx * params.commit_lens_stride);
  if (block_idx >= static_cast<uint32_t>(keep)) return;

  const int32_t out_row =
      *(params.row_offsets + batch_idx * params.row_offsets_stride) + static_cast<int32_t>(block_idx);
  const auto src_hidden = static_cast<const DType*>(params.verify_hidden) + batch_idx * params.hidden_batch_stride +
                          block_idx * params.hidden_block_stride;
  auto dst_hidden =
      static_cast<DType*>(params.hidden_out) + static_cast<int64_t>(out_row) * params.hidden_out_row_stride;

  for (uint32_t d = threadIdx.x; d < params.hidden_size; d += blockDim.x) {
    dst_hidden[d] = src_hidden[d * params.hidden_col_stride];
  }

  if (threadIdx.x == 0) {
    const auto position =
        *(static_cast<const PosT*>(params.positions) + batch_idx * params.positions_batch_stride +
          block_idx * params.positions_block_stride);
    const auto slot_id =
        *(static_cast<const IndexT*>(params.slot_ids_2d) + batch_idx * params.slot_ids_batch_stride +
          block_idx * params.slot_ids_block_stride);
    *(static_cast<PosT*>(params.positions_out) + static_cast<int64_t>(out_row) * params.positions_out_stride) =
        position;
    *(static_cast<IndexT*>(params.slot_ids_out) + static_cast<int64_t>(out_row) * params.slot_ids_out_stride) = slot_id;
  }
}

template <bool kUsePDL, typename DType>
struct DFlashCompactCommitKernel {
  template <typename PosT, typename IndexT, typename LenT>
  static constexpr auto kernel = dflash_compact_commit_kernel<DType, PosT, IndexT, LenT>;

  static void
  run(const tvm::ffi::TensorView verify_hidden,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView slot_ids_2d,
      const tvm::ffi::TensorView commit_lens,
      const tvm::ffi::TensorView row_offsets,
      const tvm::ffi::TensorView hidden_out,
      const tvm::ffi::TensorView positions_out,
      const tvm::ffi::TensorView slot_ids_out) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto BLK = SymbolicSize{"block_size"};
    auto H = SymbolicSize{"hidden_size"};
    auto MAXT = SymbolicSize{"max_rows"};
    auto HBS = SymbolicSize{"hidden_batch_stride"};
    auto HBLS = SymbolicSize{"hidden_block_stride"};
    auto HCS = SymbolicSize{"hidden_col_stride"};
    auto PBS = SymbolicSize{"positions_batch_stride"};
    auto PBLS = SymbolicSize{"positions_block_stride"};
    auto SIBS = SymbolicSize{"slot_ids_batch_stride"};
    auto SIBLS = SymbolicSize{"slot_ids_block_stride"};
    auto CLS = SymbolicSize{"commit_lens_stride"};
    auto ROS = SymbolicSize{"row_offsets_stride"};
    auto HORS = SymbolicSize{"hidden_out_row_stride"};
    auto POSS = SymbolicSize{"positions_out_stride"};
    auto SLOTS = SymbolicSize{"slot_ids_out_stride"};
    auto pos_dtype = SymbolicDType{};
    auto index_dtype = SymbolicDType{};
    auto len_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({B, BLK, H})  //
        .with_strides({HBS, HBLS, HCS})
        .with_dtype<DType>()
        .with_device(device)
        .verify(verify_hidden);
    TensorMatcher({B, BLK})  //
        .with_strides({PBS, PBLS})
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device)
        .verify(positions);
    TensorMatcher({B, BLK})  //
        .with_strides({SIBS, SIBLS})
        .with_dtype<int32_t, int64_t>(index_dtype)
        .with_device(device)
        .verify(slot_ids_2d);
    TensorMatcher({B})  //
        .with_strides({CLS})
        .with_dtype<int32_t, int64_t>(len_dtype)
        .with_device(device)
        .verify(commit_lens);
    TensorMatcher({B})  //
        .with_strides({ROS})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(row_offsets);
    TensorMatcher({MAXT, H})  //
        .with_strides({HORS, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(hidden_out);
    TensorMatcher({MAXT})  //
        .with_strides({POSS})
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device)
        .verify(positions_out);
    TensorMatcher({MAXT})  //
        .with_strides({SLOTS})
        .with_dtype<int32_t, int64_t>(index_dtype)
        .with_device(device)
        .verify(slot_ids_out);

    RuntimeCheck(BLK.unwrap() > 0, "block_size must be positive.");
    RuntimeCheck(H.unwrap() > 0, "hidden_size must be positive.");
    RuntimeCheck(MAXT.unwrap() >= B.unwrap() * BLK.unwrap(), "compact output buffers are too small.");

    const auto params = DFlashCompactCommitParams{
        .verify_hidden = verify_hidden.data_ptr(),
        .positions = positions.data_ptr(),
        .slot_ids_2d = slot_ids_2d.data_ptr(),
        .commit_lens = commit_lens.data_ptr(),
        .row_offsets = static_cast<const int32_t*>(row_offsets.data_ptr()),
        .hidden_out = hidden_out.data_ptr(),
        .positions_out = positions_out.data_ptr(),
        .slot_ids_out = slot_ids_out.data_ptr(),
        .hidden_batch_stride = HBS.unwrap(),
        .hidden_block_stride = HBLS.unwrap(),
        .hidden_col_stride = HCS.unwrap(),
        .positions_batch_stride = PBS.unwrap(),
        .positions_block_stride = PBLS.unwrap(),
        .slot_ids_batch_stride = SIBS.unwrap(),
        .slot_ids_block_stride = SIBLS.unwrap(),
        .commit_lens_stride = CLS.unwrap(),
        .row_offsets_stride = ROS.unwrap(),
        .hidden_out_row_stride = HORS.unwrap(),
        .positions_out_stride = POSS.unwrap(),
        .slot_ids_out_stride = SLOTS.unwrap(),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .block_size = static_cast<uint32_t>(BLK.unwrap()),
        .hidden_size = static_cast<uint32_t>(H.unwrap()),
        .total_rows = static_cast<uint32_t>(B.unwrap() * BLK.unwrap()),
    };

    if (pos_dtype.is_type<int32_t>()) {
      if (index_dtype.is_type<int32_t>()) {
        if (len_dtype.is_type<int32_t>()) {
          LaunchKernel(params.total_rows, kCompactThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int32_t, int32_t, int32_t>, params);
        } else {
          LaunchKernel(params.total_rows, kCompactThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int32_t, int32_t, int64_t>, params);
        }
      } else {
        if (len_dtype.is_type<int32_t>()) {
          LaunchKernel(params.total_rows, kCompactThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int32_t, int64_t, int32_t>, params);
        } else {
          LaunchKernel(params.total_rows, kCompactThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int32_t, int64_t, int64_t>, params);
        }
      }
    } else {
      if (index_dtype.is_type<int32_t>()) {
        if (len_dtype.is_type<int32_t>()) {
          LaunchKernel(params.total_rows, kCompactThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int64_t, int32_t, int32_t>, params);
        } else {
          LaunchKernel(params.total_rows, kCompactThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int64_t, int32_t, int64_t>, params);
        }
      } else {
        if (len_dtype.is_type<int32_t>()) {
          LaunchKernel(params.total_rows, kCompactThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int64_t, int64_t, int32_t>, params);
        } else {
          LaunchKernel(params.total_rows, kCompactThreadsPerBlock, device.unwrap())  //
              .enable_pdl(kUsePDL)(kernel<int64_t, int64_t, int64_t>, params);
        }
      }
    }
  }
};

}  // namespace
