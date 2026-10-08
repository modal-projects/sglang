# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 ROCm decode/verify: a row normalization fused into the prologue of
the skinny bf16 GEMM that consumes it (M = tokens <= 16, graph replay). HIP
kernel: jit/csrc/kimi_k3/norm_gemm_hip.cuh.

``kda_onorm_oproj_hip``  (SGLANG_ROCM_K3_KDA_ONORM_OPROJ)
    y[M, N] = o @ w_o^T,  o = bf16(((x * rstd) * norm_w) * sigmoid(gate)) per
    (token, head of 128) -- _kda_onorm_gated_strided_kernel + the o_proj GEMM
    in one launch. The operand o is bit-identical to the unfused kernel's.

``rmsnorm_gemm_hip``  (experimental, not wired: slower than aiter rmsnorm +
    the tuned hgemm at N = 896 because 56 blocks cannot pull the weights fast
    enough without a cross-block split-K)
    y[M, N] = bf16(x * rstd) @ w^T  (the MoE latent RMSNorm with its weight
    folded into w + the column-shard up_proj).

Outputs differ from the unfused pair only by GEMM summation order (<= 1 bf16
ulp vs the tuned hgemm; equal error vs a float64 reference).

Triton versions of both were tried first and lose on gfx950: the normalized
operand cannot feed Triton's pipelined dot loads (5-15 us slower).
"""

from __future__ import annotations

import functools
from typing import Optional

import torch


# The HIP kernel issues this lane's operand loads, then its whole weight
# stream, and normalizes only its own MFMA A fragment while the weights are in
# flight (every block still recomputes the M x K prologue: that VALU work, not
# bandwidth, is what the fusion costs on top of the plain GEMM).

_MODE_GATED_HEAD = 0
_MODE_ROW_RMS = 1
_MODE_GATED_HEAD_FAST = 3
HIP_MAX_M = 16

# (waves, n_tiles_per_block) per mode; K fixes the per-wave K blocks
HIP_ONORM_CFG = (12, 2)
HIP_RMS_CFG = (16, 1)


@functools.cache
def _hip_module(mode: int, waves: int, nt: int, kb: int):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(mode, waves, nt, kb)
    return load_jit(
        f"k3_norm_gemm_{mode}_{waves}_{nt}_{kb}",
        *args,
        cuda_files=["kimi_k3/norm_gemm_hip.cuh"],
        cuda_wrappers=[("run", f"k3_norm_gemm::NormGemm<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def hip_onorm_supported(M: int, K: int, N: int, D: int, cfg=None) -> bool:
    waves, nt = cfg or HIP_ONORM_CFG
    return 1 <= M <= HIP_MAX_M and D == 128 and K == waves * 128 and N % (16 * nt) == 0


def hip_rms_supported(M: int, K: int, N: int, cfg=None) -> bool:
    waves, nt = cfg or HIP_RMS_CFG
    return 1 <= M <= HIP_MAX_M and K % (waves * 32) == 0 and N % (16 * nt) == 0


def kda_onorm_oproj_hip(
    x: torch.Tensor,  # [M, HV*D] contiguous rows
    gate: torch.Tensor,  # [M, HV*D] unit inner stride, row stride % 8 == 0
    norm_weight: torch.Tensor,  # [D] bf16 or fp32
    eps: float,
    w: torch.Tensor,  # [N, HV*D] bf16 contiguous
    out: Optional[torch.Tensor] = None,
    cfg: Optional[tuple] = None,
    fast_sigmoid: bool = False,
) -> torch.Tensor:
    x = x.reshape(-1, w.shape[1])
    M, K = x.shape
    N = w.shape[0]
    waves, nt = cfg or HIP_ONORM_CFG
    assert hip_onorm_supported(M, K, N, norm_weight.shape[0], (waves, nt))
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if norm_weight.dtype != torch.float32:  # callers on the hot path pass fp32
        norm_weight = norm_weight.float()
    flags = 1 | 4
    assert norm_weight.is_contiguous()
    mode = _MODE_GATED_HEAD_FAST if fast_sigmoid else _MODE_GATED_HEAD
    _hip_module(mode, waves, nt, 4).run(x, gate, norm_weight, w, out, flags, float(eps))
    return out


def rmsnorm_gemm_hip(
    x: torch.Tensor,  # [M, K] bf16
    norm_weight: Optional[torch.Tensor],  # must be None: fold the norm weight into w
    eps: float,
    w: torch.Tensor,  # [N, K] bf16 contiguous
    out: Optional[torch.Tensor] = None,
    cfg: Optional[tuple] = None,
) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    assert norm_weight is None, "fold the RMSNorm weight into w"
    waves, nt = cfg or HIP_RMS_CFG
    assert hip_rms_supported(M, K, N, (waves, nt))
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    _hip_module(_MODE_ROW_RMS, waves, nt, K // (waves * 32)).run(x, x, x, w, out, 0, float(eps))
    return out
