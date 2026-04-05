#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr uint32_t kNumWarps = 4;
constexpr uint32_t kThreadsPerBlock = kNumWarps * device::kWarpThreads;

template <int64_t kElementBytes>
SGL_DEVICE void copy_row_kv_warp(
    const void* __restrict__ src_k,
    const void* __restrict__ src_v,
    void* __restrict__ dst_k,
    void* __restrict__ dst_v) {
  using namespace device;
  constexpr int64_t kAlignment = (kElementBytes % (16 * kWarpThreads) == 0) ? 16
                                 : kElementBytes % (8 * kWarpThreads) == 0  ? 8
                                 : kElementBytes % (4 * kWarpThreads) == 0  ? 4
                                 : kElementBytes % 4 == 0                   ? 4
                                                                            : 0;
  static_assert(kAlignment > 0, "Element size must be a multiple of 4 bytes");

  using vec_t = AlignedStorage<uint32_t, kAlignment / 4>;
  constexpr int64_t kLoopBytes = sizeof(vec_t) * kWarpThreads;
  constexpr int64_t kLoopCount = kElementBytes / kLoopBytes;
  const auto gmem = tile::Memory<vec_t>::warp();

#pragma unroll kLoopCount
  for (int64_t i = 0; i < kLoopCount; ++i) {
    const auto k_val = gmem.load(src_k, i);
    const auto v_val = gmem.load(src_v, i);
    gmem.store(dst_k, k_val, i);
    gmem.store(dst_v, v_val, i);
  }

  if constexpr (kLoopCount * kLoopBytes < kElementBytes) {
    if (gmem.in_bound(kElementBytes / sizeof(vec_t), kLoopCount)) {
      const auto k_val = gmem.load(src_k, kLoopCount);
      const auto v_val = gmem.load(src_v, kLoopCount);
      gmem.store(dst_k, k_val, kLoopCount);
      gmem.store(dst_v, v_val, kLoopCount);
    }
  }
}

struct PromptWriteParams {
  const void* __restrict__ src_k;
  const void* __restrict__ src_v;
  void* __restrict__ dst_k;
  void* __restrict__ dst_v;
  const void* __restrict__ slot_ids;
  int64_t src_layer_stride_bytes;
  int64_t src_token_stride_bytes;
  int64_t dst_layer_stride_bytes;
  int64_t dst_slot_stride_bytes;
  int64_t slot_ids_stride;
  uint32_t num_layers;
  uint32_t num_tokens;
};

template <int64_t kElementBytes, int kSplit, bool kUsePDL, typename IndexT>
__global__ void dflash_prompt_kv_prefix_write(const __grid_constant__ PromptWriteParams params) {
  using namespace device;
  constexpr auto kSplitBytes = kElementBytes / kSplit;
  const uint32_t warp_id = blockIdx.x * kNumWarps + threadIdx.x / kWarpThreads;
  const uint64_t total_rows = static_cast<uint64_t>(params.num_layers) * static_cast<uint64_t>(params.num_tokens);
  const uint64_t row_id = warp_id / kSplit;
  const uint32_t split_id = warp_id % kSplit;
  if (row_id >= total_rows) return;

  const uint32_t layer_idx = row_id / params.num_tokens;
  const uint32_t token_idx = row_id % params.num_tokens;
  const auto slot_ptr = static_cast<const IndexT*>(params.slot_ids) + token_idx * params.slot_ids_stride;
  PDLWaitPrimary<kUsePDL>();
  const auto slot_idx = *slot_ptr;

  const auto src_k_ptr = pointer::offset(
      params.src_k,
      layer_idx * params.src_layer_stride_bytes,
      token_idx * params.src_token_stride_bytes + split_id * kSplitBytes);
  const auto src_v_ptr = pointer::offset(
      params.src_v,
      layer_idx * params.src_layer_stride_bytes,
      token_idx * params.src_token_stride_bytes + split_id * kSplitBytes);
  const auto dst_k_ptr = pointer::offset(
      params.dst_k,
      layer_idx * params.dst_layer_stride_bytes,
      slot_idx * params.dst_slot_stride_bytes + split_id * kSplitBytes);
  const auto dst_v_ptr = pointer::offset(
      params.dst_v,
      layer_idx * params.dst_layer_stride_bytes,
      slot_idx * params.dst_slot_stride_bytes + split_id * kSplitBytes);

  copy_row_kv_warp<kSplitBytes>(src_k_ptr, src_v_ptr, dst_k_ptr, dst_v_ptr);
  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kElementBytes, bool kUsePDL>
struct DFlashPromptKVPrefixWriteKernel {
  template <typename IndexT>
  static constexpr auto kernel = dflash_prompt_kv_prefix_write<kElementBytes, 1, kUsePDL, IndexT>;
  template <int kSplit, typename IndexT>
  static constexpr auto split_kernel = dflash_prompt_kv_prefix_write<kElementBytes, kSplit, kUsePDL, IndexT>;

  template <typename IndexT>
  static auto get_kernel(const int num_split) {
    using namespace host;
    if constexpr (kElementBytes % (4 * 128) == 0) {
      if (num_split == 4) return split_kernel<4, IndexT>;
    }
    if constexpr (kElementBytes % (2 * 128) == 0) {
      if (num_split == 2) return split_kernel<2, IndexT>;
    }
    if (num_split == 1) return kernel<IndexT>;
    Panic("Unsupported num_split {} for DFlash prompt row_bytes {}", num_split, kElementBytes);
  }

  static void
  run(const tvm::ffi::TensorView src_k,
      const tvm::ffi::TensorView src_v,
      const tvm::ffi::TensorView dst_k,
      const tvm::ffi::TensorView dst_v,
      const tvm::ffi::TensorView slot_ids,
      const int num_split) {
    using namespace host;
    auto L = SymbolicSize{"num_layers"};
    auto T = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"feature_dim"};
    auto S = SymbolicSize{"num_slots"};
    auto SKL = SymbolicSize{"src_layer_stride"};
    auto SKT = SymbolicSize{"src_token_stride"};
    auto DKL = SymbolicSize{"dst_layer_stride"};
    auto DKS = SymbolicSize{"dst_slot_stride"};
    auto IS = SymbolicSize{"slot_stride"};
    auto dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    auto index_dtype = SymbolicDType{};
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({L, T, D})  //
        .with_strides({SKL, SKT, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(src_k)
        .verify(src_v);
    TensorMatcher({L, S, D})  //
        .with_strides({DKL, DKS, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(dst_k)
        .verify(dst_v);
    TensorMatcher({T})  //
        .with_strides({IS})
        .with_dtype<int32_t, int64_t>(index_dtype)
        .with_device(device)
        .verify(slot_ids);

    const int64_t dtype_size = dtype_bytes(dtype.unwrap());
    RuntimeCheck(kElementBytes == dtype_size * D.unwrap());

    const auto params = PromptWriteParams{
        .src_k = src_k.data_ptr(),
        .src_v = src_v.data_ptr(),
        .dst_k = dst_k.data_ptr(),
        .dst_v = dst_v.data_ptr(),
        .slot_ids = slot_ids.data_ptr(),
        .src_layer_stride_bytes = SKL.unwrap() * dtype_size,
        .src_token_stride_bytes = SKT.unwrap() * dtype_size,
        .dst_layer_stride_bytes = DKL.unwrap() * dtype_size,
        .dst_slot_stride_bytes = DKS.unwrap() * dtype_size,
        .slot_ids_stride = IS.unwrap(),
        .num_layers = static_cast<uint32_t>(L.unwrap()),
        .num_tokens = static_cast<uint32_t>(T.unwrap()),
    };
    const auto use_int32 = index_dtype.is_type<int32_t>();
    const auto selected_kernel = use_int32 ? get_kernel<int32_t>(num_split) : get_kernel<int64_t>(num_split);
    const auto num_rows = static_cast<uint64_t>(params.num_layers) * static_cast<uint64_t>(params.num_tokens);
    const auto num_blocks = div_ceil(num_rows * static_cast<uint64_t>(num_split), static_cast<uint64_t>(kNumWarps));
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
        .enable_pdl(kUsePDL)(selected_kernel, params);
  }
};

struct CommitWriteParams {
  const void* __restrict__ src_k;
  const void* __restrict__ src_v;
  void* __restrict__ dst_k;
  void* __restrict__ dst_v;
  const void* __restrict__ slot_ids_2d;
  const void* __restrict__ commit_lens;
  int64_t src_layer_stride_bytes;
  int64_t src_batch_stride_bytes;
  int64_t src_block_stride_bytes;
  int64_t dst_layer_stride_bytes;
  int64_t dst_slot_stride_bytes;
  int64_t slot_batch_stride;
  int64_t slot_block_stride;
  int64_t commit_len_stride;
  uint32_t num_layers;
  uint32_t batch_size;
  uint32_t block_size;
};

template <int64_t kElementBytes, int kSplit, bool kUsePDL, typename IndexT, typename LenT>
__global__ void dflash_commit_kv_prefix_write(const __grid_constant__ CommitWriteParams params) {
  using namespace device;
  constexpr auto kSplitBytes = kElementBytes / kSplit;
  const uint32_t warp_id = blockIdx.x * kNumWarps + threadIdx.x / kWarpThreads;
  const uint64_t rows_per_layer = static_cast<uint64_t>(params.batch_size) * static_cast<uint64_t>(params.block_size);
  const uint64_t total_rows = static_cast<uint64_t>(params.num_layers) * static_cast<uint64_t>(rows_per_layer);
  const uint64_t row_id = warp_id / kSplit;
  const uint32_t split_id = warp_id % kSplit;
  if (row_id >= total_rows) return;

  const uint32_t layer_idx = row_id / rows_per_layer;
  const uint32_t row_offset = row_id % rows_per_layer;
  const uint32_t batch_idx = row_offset / params.block_size;
  const uint32_t block_idx = row_offset % params.block_size;

  const auto keep_ptr = static_cast<const LenT*>(params.commit_lens) + batch_idx * params.commit_len_stride;
  PDLWaitPrimary<kUsePDL>();
  const auto keep = *keep_ptr;
  if (block_idx >= static_cast<uint32_t>(keep)) return;

  const auto slot_ptr = static_cast<const IndexT*>(params.slot_ids_2d) + batch_idx * params.slot_batch_stride +
                        block_idx * params.slot_block_stride;
  const auto slot_idx = *slot_ptr;

  const auto src_k_ptr = pointer::offset(
      params.src_k,
      layer_idx * params.src_layer_stride_bytes,
      batch_idx * params.src_batch_stride_bytes + block_idx * params.src_block_stride_bytes + split_id * kSplitBytes);
  const auto src_v_ptr = pointer::offset(
      params.src_v,
      layer_idx * params.src_layer_stride_bytes,
      batch_idx * params.src_batch_stride_bytes + block_idx * params.src_block_stride_bytes + split_id * kSplitBytes);
  const auto dst_k_ptr = pointer::offset(
      params.dst_k,
      layer_idx * params.dst_layer_stride_bytes,
      slot_idx * params.dst_slot_stride_bytes + split_id * kSplitBytes);
  const auto dst_v_ptr = pointer::offset(
      params.dst_v,
      layer_idx * params.dst_layer_stride_bytes,
      slot_idx * params.dst_slot_stride_bytes + split_id * kSplitBytes);

  copy_row_kv_warp<kSplitBytes>(src_k_ptr, src_v_ptr, dst_k_ptr, dst_v_ptr);
  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kElementBytes, bool kUsePDL>
struct DFlashCommitKVPrefixWriteKernel {
  template <typename IndexT, typename LenT>
  static constexpr auto kernel = dflash_commit_kv_prefix_write<kElementBytes, 1, kUsePDL, IndexT, LenT>;

  template <int kSplit, typename IndexT, typename LenT>
  static constexpr auto split_kernel = dflash_commit_kv_prefix_write<kElementBytes, kSplit, kUsePDL, IndexT, LenT>;

  template <typename IndexT, typename LenT>
  static auto get_kernel_for_split(const int num_split) {
    using namespace host;
    if constexpr (kElementBytes % (4 * 128) == 0) {
      if (num_split == 4) return split_kernel<4, IndexT, LenT>;
    }
    if constexpr (kElementBytes % (2 * 128) == 0) {
      if (num_split == 2) return split_kernel<2, IndexT, LenT>;
    }
    if (num_split == 1) return kernel<IndexT, LenT>;
    Panic("Unsupported num_split {} for DFlash commit row_bytes {}", num_split, kElementBytes);
  }

  static void
  run(const tvm::ffi::TensorView src_k,
      const tvm::ffi::TensorView src_v,
      const tvm::ffi::TensorView dst_k,
      const tvm::ffi::TensorView dst_v,
      const tvm::ffi::TensorView slot_ids_2d,
      const tvm::ffi::TensorView commit_lens,
      const int num_split) {
    using namespace host;
    auto L = SymbolicSize{"num_layers"};
    auto B = SymbolicSize{"batch_size"};
    auto T = SymbolicSize{"block_size"};
    auto D = SymbolicSize{"feature_dim"};
    auto S = SymbolicSize{"num_slots"};
    auto SKL = SymbolicSize{"src_layer_stride"};
    auto SKB = SymbolicSize{"src_batch_stride"};
    auto SKT = SymbolicSize{"src_block_stride"};
    auto DKL = SymbolicSize{"dst_layer_stride"};
    auto DKS = SymbolicSize{"dst_slot_stride"};
    auto I0 = SymbolicSize{"slot_batch_stride"};
    auto I1 = SymbolicSize{"slot_block_stride"};
    auto CL = SymbolicSize{"commit_stride"};
    auto dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    auto index_dtype = SymbolicDType{};
    auto len_dtype = SymbolicDType{};
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({L, B, T, D})  //
        .with_strides({SKL, SKB, SKT, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(src_k)
        .verify(src_v);
    TensorMatcher({L, S, D})  //
        .with_strides({DKL, DKS, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(dst_k)
        .verify(dst_v);
    TensorMatcher({B, T})  //
        .with_strides({I0, I1})
        .with_dtype<int32_t, int64_t>(index_dtype)
        .with_device(device)
        .verify(slot_ids_2d);
    TensorMatcher({B})  //
        .with_strides({CL})
        .with_dtype<int32_t, int64_t>(len_dtype)
        .with_device(device)
        .verify(commit_lens);

    const int64_t dtype_size = dtype_bytes(dtype.unwrap());
    RuntimeCheck(kElementBytes == dtype_size * D.unwrap());

    const auto params = CommitWriteParams{
        .src_k = src_k.data_ptr(),
        .src_v = src_v.data_ptr(),
        .dst_k = dst_k.data_ptr(),
        .dst_v = dst_v.data_ptr(),
        .slot_ids_2d = slot_ids_2d.data_ptr(),
        .commit_lens = commit_lens.data_ptr(),
        .src_layer_stride_bytes = SKL.unwrap() * dtype_size,
        .src_batch_stride_bytes = SKB.unwrap() * dtype_size,
        .src_block_stride_bytes = SKT.unwrap() * dtype_size,
        .dst_layer_stride_bytes = DKL.unwrap() * dtype_size,
        .dst_slot_stride_bytes = DKS.unwrap() * dtype_size,
        .slot_batch_stride = I0.unwrap(),
        .slot_block_stride = I1.unwrap(),
        .commit_len_stride = CL.unwrap(),
        .num_layers = static_cast<uint32_t>(L.unwrap()),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .block_size = static_cast<uint32_t>(T.unwrap()),
    };
    const bool use_int32_index = index_dtype.is_type<int32_t>();
    const bool use_int32_len = len_dtype.is_type<int32_t>();
    const auto selected_kernel_split = [&]() {
      if (use_int32_index) {
        return use_int32_len ? get_kernel_for_split<int32_t, int32_t>(num_split)
                             : get_kernel_for_split<int32_t, int64_t>(num_split);
      }
      return use_int32_len ? get_kernel_for_split<int64_t, int32_t>(num_split)
                           : get_kernel_for_split<int64_t, int64_t>(num_split);
    }();
    const auto rows_per_layer = static_cast<uint64_t>(params.batch_size) * static_cast<uint64_t>(params.block_size);
    const auto total_rows = static_cast<uint64_t>(params.num_layers) * static_cast<uint64_t>(rows_per_layer);
    const auto num_blocks = div_ceil(total_rows * static_cast<uint64_t>(num_split), static_cast<uint64_t>(kNumWarps));
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
        .enable_pdl(kUsePDL)(selected_kernel_split, params);
  }
};

}  // namespace
