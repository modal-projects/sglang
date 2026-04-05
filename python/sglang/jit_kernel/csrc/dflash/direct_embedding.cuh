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

template <int64_t kRowBytes>
SGL_DEVICE void copy_row_warp(const void* __restrict__ src, void* __restrict__ dst) {
  using namespace device;
  constexpr int64_t kAlignment = (kRowBytes % (16 * kWarpThreads) == 0) ? 16
                                 : kRowBytes % (8 * kWarpThreads) == 0  ? 8
                                 : kRowBytes % (4 * kWarpThreads) == 0  ? 4
                                 : kRowBytes % 4 == 0                   ? 4
                                                                        : 0;
  static_assert(kAlignment > 0, "Row size must be a multiple of 4 bytes");

  using vec_t = AlignedStorage<uint32_t, kAlignment / 4>;
  constexpr int64_t kLoopBytes = sizeof(vec_t) * kWarpThreads;
  constexpr int64_t kLoopCount = kRowBytes / kLoopBytes;
  const auto gmem = tile::Memory<vec_t>::warp();

#pragma unroll kLoopCount
  for (int64_t i = 0; i < kLoopCount; ++i) {
    const auto val = gmem.load(src, i);
    gmem.store(dst, val, i);
  }

  if constexpr (kLoopCount * kLoopBytes < kRowBytes) {
    if (gmem.in_bound(kRowBytes / sizeof(vec_t), kLoopCount)) {
      const auto val = gmem.load(src, kLoopCount);
      gmem.store(dst, val, kLoopCount);
    }
  }
}

struct DirectEmbeddingParams {
  const void* __restrict__ embedding_table;
  const void* __restrict__ first_token_ids;
  void* __restrict__ output;
  int64_t table_row_stride_bytes;
  int64_t first_token_stride;
  int64_t output_row_stride_bytes;
  int64_t mask_token_id;
  uint32_t batch_size;
  uint32_t block_size;
};

template <int64_t kRowBytes, bool kUsePDL, typename IndexT>
__global__ void dflash_direct_embedding(const __grid_constant__ DirectEmbeddingParams params) {
  using namespace device;
  const uint32_t warp_id = blockIdx.x * kNumWarps + threadIdx.x / kWarpThreads;
  const uint64_t total_rows = static_cast<uint64_t>(params.batch_size) * static_cast<uint64_t>(params.block_size);
  if (warp_id >= total_rows) return;

  const uint32_t batch_idx = warp_id / params.block_size;
  const uint32_t block_col = warp_id % params.block_size;

  const auto first_token_ptr =
      static_cast<const IndexT*>(params.first_token_ids) + batch_idx * params.first_token_stride;
  PDLWaitPrimary<kUsePDL>();
  const auto token_id = (block_col == 0) ? *first_token_ptr : static_cast<IndexT>(params.mask_token_id);

  const auto src_ptr =
      pointer::offset(params.embedding_table, static_cast<int64_t>(token_id) * params.table_row_stride_bytes);
  auto dst_ptr = pointer::offset(params.output, warp_id * params.output_row_stride_bytes);
  copy_row_warp<kRowBytes>(src_ptr, dst_ptr);
  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kRowBytes, bool kUsePDL>
struct DFlashDirectEmbeddingKernel {
  template <typename IndexT>
  static constexpr auto kernel = dflash_direct_embedding<kRowBytes, kUsePDL, IndexT>;

  static void
  run(const tvm::ffi::TensorView embedding_table,
      const tvm::ffi::TensorView first_token_ids,
      const tvm::ffi::TensorView output,
      const int block_size,
      const int64_t mask_token_id) {
    using namespace host;
    auto V = SymbolicSize{"vocab_size"};
    auto H = SymbolicSize{"hidden_size"};
    auto B = SymbolicSize{"batch_size"};
    auto K = SymbolicSize{"block_size_out"};
    auto ER = SymbolicSize{"embedding_row_stride"};
    auto IS = SymbolicSize{"index_stride"};
    auto OB = SymbolicSize{"output_batch_stride"};
    auto OK = SymbolicSize{"output_block_stride"};
    auto dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    auto index_dtype = SymbolicDType{};
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({V, H})  //
        .with_strides({ER, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(embedding_table);
    TensorMatcher({B})  //
        .with_strides({IS})
        .with_dtype<int32_t, int64_t>(index_dtype)
        .with_device(device)
        .verify(first_token_ids);
    TensorMatcher({B, K, H})  //
        .with_strides({OB, OK, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(output);

    RuntimeCheck(K.unwrap() == block_size);
    RuntimeCheck(OK.unwrap() == H.unwrap());
    RuntimeCheck(OB.unwrap() == K.unwrap() * H.unwrap());
    RuntimeCheck(mask_token_id >= 0 && mask_token_id < V.unwrap());
    const int64_t dtype_size = dtype_bytes(dtype.unwrap());
    RuntimeCheck(kRowBytes == dtype_size * H.unwrap());

    const auto params = DirectEmbeddingParams{
        .embedding_table = embedding_table.data_ptr(),
        .first_token_ids = first_token_ids.data_ptr(),
        .output = output.data_ptr(),
        .table_row_stride_bytes = ER.unwrap() * dtype_size,
        .first_token_stride = IS.unwrap(),
        .output_row_stride_bytes = OK.unwrap() * dtype_size,
        .mask_token_id = mask_token_id,
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .block_size = static_cast<uint32_t>(block_size),
    };
    const auto selected_kernel = index_dtype.is_type<int32_t>() ? kernel<int32_t> : kernel<int64_t>;
    const auto total_rows = static_cast<uint64_t>(params.batch_size) * static_cast<uint64_t>(params.block_size);
    const auto num_blocks = div_ceil(total_rows, static_cast<uint64_t>(kNumWarps));
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
        .enable_pdl(kUsePDL)(selected_kernel, params);
  }
};

}  // namespace
