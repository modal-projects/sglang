"""K3 MLA target-verify (QLEN=8, 12 heads, fp8 KV latent 576) hand-scheduled HIP kernel (gfx950).

Same contract and split/merge scheme as k3_mla_verify_v2 (one CTA per (request,
KV split) covering all 8 q_pos x 12 heads, so the KV is read once; per-split
normalized partial O + base-2 LSE, merged by v2's Triton reduce), but stage 1
is jit/csrc/kimi_k3/k3_mla_verify_hk.cuh:

  * 4 waves = 3 compute waves (32 query rows each, the 96 rows exactly, no
    padding) + 1 loader wave that streams 64-token KV tiles global -> LDS with
    buffer_load_dwordx4 ... lds (4-stage ring, one s_barrier per tile).
  * S^T = K Q^T (FP8 MFMA 32x32x64, Q quantized per row like v2 and held in
    VGPRs) so each lane owns one query row; O^T += V^T P^T with P (FP8) kept
    in registers and O^T (32 x 512 fp32 per wave) in AGPRs; V read with
    ds_read_b64_tr_b8.
  * The inner loop is hand-scheduled inline asm (gen script next to the .cuh):
    the 18 QK MFMAs of tile i+1 are interleaved with the softmax VALU of tile
    i, then the 16 PV MFMAs of tile i stream the transposed V reads.

No host syncs; the grid depends only on bs (CUDA-graph capturable).
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import torch
import triton

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.attention.k3_mla_verify_v2 import (
    _k3_mla_verify_reduce,
    _ones,
)

_LOG2E = 1.4426950408889634
_BLOCK_N = 64
_MIN_CHUNK = 256
_TARGET_CTAS = 256  # one CTA per CU (1 wave / SIMD, 144 KB LDS)
_MAX_SPLITS = 256

_CSRC = os.path.join(os.path.dirname(__file__), "..", "..", "jit", "csrc", "kimi_k3")


def _src_hash() -> str:
    h = hashlib.sha1()
    for name in ("k3_mla_verify_hk.cuh", "k3_mla_verify_hk_asm.inc"):
        with open(os.path.join(_CSRC, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


@cache_once
def _module():
    return load_jit(
        "kimi_k3_mla_verify_hk",
        _src_hash(),
        cuda_files=["kimi_k3/k3_mla_verify_hk.cuh"],
        cuda_wrappers=[("run", "K3MlaVerifyHK::run")],
        extra_cuda_cflags=["-O3", "-mllvm", "-amdgpu-spill-vgpr-to-agpr=0"],
    )


def hk_stage1(q, kv, kv_indptr, kv_indices, kv_scale, o_part, lse_part, out, nsplit: int, min_chunk: int,
              sm_scale: float) -> None:
    """Stage 1 only (partials into o_part / lse_part with the v2 split formula)."""
    _module().run(q, kv.view(torch.uint8), kv_indptr, kv_indices, kv_scale, o_part, lse_part, out,
                  nsplit, min_chunk, float(sm_scale) * _LOG2E)


def default_num_splits(bs: int) -> int:
    """Power-of-two splits per request so that bs * NSPLIT ~ one CTA per CU."""
    n = max(1, -(-_TARGET_CTAS // bs))
    p = 1
    while p < n:
        p *= 2
    return min(p, _MAX_SPLITS)


def k3_mla_verify_hk(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    kv_scale=None,
    *,
    num_heads: int = 12,
    qlen: int = 8,
    v_head_dim: int = 512,
    out: Optional[torch.Tensor] = None,
    num_splits: Optional[int] = None,
) -> torch.Tensor:
    """Drop-in for k3_mla_verify_v2 (same arguments / semantics)."""
    D = kv_buffer.shape[-1]
    assert num_heads == 12 and qlen == 8 and v_head_dim == 512 and D == 576
    kv = kv_buffer.view(-1, D)
    q = q.view(-1, num_heads, D)
    ntok = q.shape[0]
    bs = ntok // qlen
    assert bs * qlen == ntok
    if out is None:
        out = q.new_empty((ntok, num_heads, v_head_dim))
    if kv_scale is None:
        kv_scale = _ones(q.device)
    elif not isinstance(kv_scale, torch.Tensor):
        kv_scale = torch.full((1,), float(kv_scale), dtype=torch.float32, device=q.device)
    if bs == 0:
        return out
    nsplit = num_splits or default_num_splits(bs)
    rows = num_heads * qlen
    if nsplit > 1:
        o_part = torch.empty((bs, nsplit, rows, v_head_dim), dtype=torch.bfloat16, device=q.device)
        lse_part = torch.empty((bs, nsplit, rows), dtype=torch.float32, device=q.device)
    else:
        o_part = out
        lse_part = kv_scale
    _module().run(
        q, kv.view(torch.uint8), kv_indptr, kv_indices, kv_scale.float(), o_part, lse_part, out,
        nsplit, _MIN_CHUNK, float(sm_scale) * _LOG2E,
    )
    if nsplit > 1:
        _k3_mla_verify_reduce[(bs, rows, v_head_dim // 128)](
            o_part, lse_part, kv_indptr, out, out.stride(0), out.stride(1),
            H=num_heads, QLEN=qlen, D_V=v_head_dim, NSPLIT=nsplit,
            NSPLIT_P2=triton.next_power_of_2(nsplit), BLOCK_N=_BLOCK_N, MIN_CHUNK=_MIN_CHUNK,
            S_BLK=min(64, triton.next_power_of_2(nsplit)), D_BLK=128,
            num_warps=4 if nsplit >= 64 else 2,
        )
    return out
