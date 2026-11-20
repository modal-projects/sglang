#pragma once

// MXFP8 producers for the MXFP4-expert (DeepSeek-V4.1) decomposed MoE-LoRA op.
//
// Both kernels write only the valid (expanded -> permuted) rows; the grouped
// GEMMs never read padding rows (they are bounded by the routing's CTA maps).
//
//   permuteQuantMxfp8Kernel   bf16 hidden [T, H] -> permuted MXFP8 [P, H] + SF
//   loraActQuantMxfp8Kernel   raw gate/up GEMM output (gated-interleaved
//                             columns: even = up, odd = gate) + BF16 LoRA delta
//                             ([gate | up] per expanded row) -> clamped SwiGLU
//                             -> down-LoRA input (bf16, per expanded row) and
//                             permuted MXFP8 [P, I] + SF for the down GEMM.
//
// Scale factors are UE8M0 per 32 elements in the swizzled layout the TRT-LLM
// grouped GEMM reads for its token-side operand (8x4 for token tiles < 128,
// 128x4 otherwise).

#include "nv_internal/tensorrt_llm/kernels/quantization_utils.cuh"
#include <cstdint>
#include <cuda_bf16.h>

namespace flashinfer {
namespace sgl_mx_lora {

namespace tk = tensorrt_llm::kernels;

constexpr int kEltsPerThread = 8;  // one uint64_t of e4m3 per thread
constexpr int kSfVec = 32;         // MX block
constexpr int kThreads = 128;
using Vec8 = tk::PackedVec<__nv_bfloat16, kEltsPerThread>;

template <bool Swizzled128>
inline __device__ int64_t sfOffset(int row, int sfCol, int numRows,
                                   int numSfCols) {
  if constexpr (Swizzled128) {
    return tk::get_sf_out_offset_128x4(0, row, sfCol, numRows, numSfCols);
  } else {
    return tk::get_sf_out_offset_8x4(0, row, sfCol, numRows, numSfCols);
  }
}

// grid.x = numTokens * topK; one CTA per expanded row. hidden % 256 == 0 keeps
// every warp either fully active or fully idle (the SF reduction shuffles over
// the full warp).
template <bool Swizzled128>
__global__ void __launch_bounds__(kThreads)
    permuteQuantMxfp8Kernel(__nv_bfloat16 const *__restrict__ hidden,
                            int32_t const *__restrict__ expandedIdxToPermutedIdx,
                            int topK, int hiddenDim, int numSfRows,
                            uint8_t *__restrict__ outFp8,
                            uint8_t *__restrict__ outSf) {
  int const expandedIdx = blockIdx.x;
  int const permutedIdx = expandedIdxToPermutedIdx[expandedIdx];
  if (permutedIdx < 0) {
    return;
  }
  int const token = expandedIdx / topK;
  int const numChunks = hiddenDim / kEltsPerThread;
  int const numSfCols = hiddenDim / kSfVec;
  for (int chunk = threadIdx.x; chunk < numChunks; chunk += kThreads) {
    Vec8 v = *reinterpret_cast<Vec8 const *>(
        hidden + (int64_t)token * hiddenDim + (int64_t)chunk * kEltsPerThread);
    uint8_t sf;
    uint8_t *sfPtr = (chunk % (kSfVec / kEltsPerThread) == 0) ? &sf : nullptr;
    uint64_t q =
        tk::cvt_warp_fp16_to_mxfp8<__nv_bfloat16, kSfVec, kEltsPerThread>(
            v, sfPtr);
    reinterpret_cast<uint64_t *>(outFp8 + (int64_t)permutedIdx * hiddenDim)
        [chunk] = q;
    if (sfPtr != nullptr) {
      outSf[sfOffset<Swizzled128>(permutedIdx, chunk / (kSfVec / kEltsPerThread),
                                  numSfRows, numSfCols)] = sf;
    }
  }
}

inline __device__ float siluf(float x) { return x / (1.0f + __expf(-x)); }

// grid.x = numTokens * topK. inter % 256 == 0.
template <bool Swizzled128>
__global__ void __launch_bounds__(kThreads)
    loraActQuantMxfp8Kernel(__nv_bfloat16 const *__restrict__ gateUp,
                            __nv_bfloat16 const *__restrict__ loraDelta,
                            __nv_bfloat16 *__restrict__ loraInputOut,
                            int32_t const *__restrict__ expandedIdxToPermutedIdx,
                            float const *__restrict__ clampLimit, int inter,
                            int numSfRows, uint8_t *__restrict__ outFp8,
                            uint8_t *__restrict__ outSf) {
  int const expandedIdx = blockIdx.x;
  int const permutedIdx = expandedIdxToPermutedIdx[expandedIdx];
  int const numChunks = inter / kEltsPerThread;
  int64_t const liBase = (int64_t)expandedIdx * inter;
  if (permutedIdx < 0) {
    if (loraInputOut != nullptr) {
      for (int chunk = threadIdx.x; chunk < numChunks; chunk += kThreads) {
        *reinterpret_cast<uint4 *>(loraInputOut + liBase +
                                   (int64_t)chunk * kEltsPerThread) =
            make_uint4(0, 0, 0, 0);
      }
    }
    return;
  }
  float const limit = clampLimit != nullptr ? clampLimit[0] : 0.f;
  int const numSfCols = inter / kSfVec;
  int64_t const guBase = (int64_t)permutedIdx * 2 * inter;
  int64_t const dBase = (int64_t)expandedIdx * 2 * inter;
  for (int chunk = threadIdx.x; chunk < numChunks; chunk += kThreads) {
    int const h0 = chunk * kEltsPerThread;
    union {
      uint4 u[2];
      __nv_bfloat16 b[16];
    } gu;
    gu.u[0] = *reinterpret_cast<uint4 const *>(gateUp + guBase + 2 * h0);
    gu.u[1] = *reinterpret_cast<uint4 const *>(gateUp + guBase + 2 * h0 + 8);
    union {
      uint4 u;
      __nv_bfloat16 b[8];
    } dg, du;
    if (loraDelta != nullptr) {
      dg.u = *reinterpret_cast<uint4 const *>(loraDelta + dBase + h0);
      du.u = *reinterpret_cast<uint4 const *>(loraDelta + dBase + inter + h0);
    }
    Vec8 v;
#pragma unroll
    for (int i = 0; i < kEltsPerThread / 2; ++i) {
      float a[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int const e = 2 * i + j;
        float up = (float)gu.b[2 * e];
        float gate = (float)gu.b[2 * e + 1];
        if (loraDelta != nullptr) {
          gate += (float)dg.b[e];
          up += (float)du.b[e];
        }
        if (clampLimit != nullptr) {
          gate = fminf(gate, limit);
          up = fminf(fmaxf(up, -limit), limit);
        }
        a[j] = siluf(gate) * up;
      }
      v.elts[i] = __float22bfloat162_rn(make_float2(a[0], a[1]));
    }
    if (loraInputOut != nullptr) {
      *reinterpret_cast<Vec8 *>(loraInputOut + liBase + h0) = v;
    }
    uint8_t sf;
    uint8_t *sfPtr = (chunk % (kSfVec / kEltsPerThread) == 0) ? &sf : nullptr;
    uint64_t q =
        tk::cvt_warp_fp16_to_mxfp8<__nv_bfloat16, kSfVec, kEltsPerThread>(
            v, sfPtr);
    reinterpret_cast<uint64_t *>(outFp8 + (int64_t)permutedIdx * inter)[chunk] =
        q;
    if (sfPtr != nullptr) {
      outSf[sfOffset<Swizzled128>(permutedIdx, chunk / (kSfVec / kEltsPerThread),
                                  numSfRows, numSfCols)] = sf;
    }
  }
}

inline void launchPermuteQuantMxfp8(__nv_bfloat16 const *hidden,
                                    int32_t const *expandedIdxToPermutedIdx,
                                    int numTokens, int topK, int hiddenDim,
                                    int numSfRows, bool swizzled128,
                                    uint8_t *outFp8, uint8_t *outSf,
                                    cudaStream_t stream) {
  dim3 grid(numTokens * topK);
  if (grid.x == 0) {
    return;
  }
  if (swizzled128) {
    permuteQuantMxfp8Kernel<true><<<grid, kThreads, 0, stream>>>(
        hidden, expandedIdxToPermutedIdx, topK, hiddenDim, numSfRows, outFp8,
        outSf);
  } else {
    permuteQuantMxfp8Kernel<false><<<grid, kThreads, 0, stream>>>(
        hidden, expandedIdxToPermutedIdx, topK, hiddenDim, numSfRows, outFp8,
        outSf);
  }
}

inline void launchLoraActQuantMxfp8(
    __nv_bfloat16 const *gateUp, __nv_bfloat16 const *loraDelta,
    __nv_bfloat16 *loraInputOut, int32_t const *expandedIdxToPermutedIdx,
    float const *clampLimit, int numExpandedRows, int inter, int numSfRows,
    bool swizzled128, uint8_t *outFp8, uint8_t *outSf, cudaStream_t stream) {
  dim3 grid(numExpandedRows);
  if (grid.x == 0) {
    return;
  }
  if (swizzled128) {
    loraActQuantMxfp8Kernel<true><<<grid, kThreads, 0, stream>>>(
        gateUp, loraDelta, loraInputOut, expandedIdxToPermutedIdx, clampLimit,
        inter, numSfRows, outFp8, outSf);
  } else {
    loraActQuantMxfp8Kernel<false><<<grid, kThreads, 0, stream>>>(
        gateUp, loraDelta, loraInputOut, expandedIdxToPermutedIdx, clampLimit,
        inter, numSfRows, outFp8, outSf);
  }
}

} // namespace sgl_mx_lora
} // namespace flashinfer
