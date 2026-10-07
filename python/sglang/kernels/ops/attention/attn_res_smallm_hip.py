"""Attention-residual aggregation for decode/verify token counts on ROCm.

Same contract as attn_res_hip.attn_res_hip (score the nvb bank rows and the
prefix, global-max softmax, mix, optional output RMSNorm / E4M3 copy, folded
residual add and bank snapshot), as one HIP kernel
(jit/csrc/kimi_k3/attn_res_smallm.cuh) shaped for T = 1..64.

Why: the Triton _agg_kernel keeps the bank as a [next_pow2(nvb), 8192] fp32
register tile and pays ~0.65-1 us per bank row in LDS layout round trips
(45 s_barrier at nvb = 8), so at M = 8 it takes 5.3 us (nvb = 1) to 10.7 us
(nvb = 8) although it moves < 1.2 MB. One CU streams those rows in well under a
microsecond, so the HIP kernel keeps one block per token but holds each row as
a few 16-byte vectors per thread: all loads in flight at once, per-thread
partial sums, one block reduction for all 2 * (nvb + 1) score partials, one for
||acc||^2. (Splitting H over CTAs was tried: the cross-CTA reduction costs
either a second launch, ~7.8 us total, or an atomic barrier, +3-10 us.)

Numerics: identical formulas and bf16 rounding points; fp32 reductions run in
a different order than the Triton lane tree, so the bf16 outputs match
_agg_kernel except rare 1-ulp round-half flips (~1e-4 of elements; equal
error vs a float64 reference). prefix_out and the bank snapshot are
bit-identical.

stream_out additionally receives the bf16 pre-norm mixture (what attn_res_hip
returns with ow=None): the K3 dspark aux capture of the previous layer is that
value, so the capture's own launch and the packer copy fold in here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

MAX_NVB = 8


def _launch_shape(hidden_size: int) -> tuple[int, int]:
    """(vectors per thread, threads per block) for H = hidden_size."""
    n_vecs = hidden_size // 8
    for threads in (448, 512, 384, 256, 320, 576, 640, 704, 768, 832, 896, 960, 1024):
        if n_vecs % threads == 0 and n_vecs // threads <= 4:
            return n_vecs // threads, threads
    threads = 512
    return -(-n_vecs // threads), threads


@cache_once
def _module(vpt: int, threads: int) -> Module:
    args = make_cpp_args(vpt, threads)
    return load_jit(
        f"kimi_k3_attn_res_smallm_{vpt}_{threads}",
        *args,
        cuda_files=["kimi_k3/attn_res_smallm.cuh"],
        cuda_wrappers=[("run", f"AttnResSmallMKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def supported(hidden_size: int, nvb: int) -> bool:
    return 1 <= nvb <= MAX_NVB and hidden_size % 8 == 0 and _launch_shape(hidden_size)[0] <= 4


def attn_res_smallm_hip(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: Optional[torch.Tensor],
    out: torch.Tensor,
    nvb: int,
    score_eps: float,
    out_eps: float,
    *,
    addend: Optional[torch.Tensor] = None,
    prefix_out: Optional[torch.Tensor] = None,
    write_prefix: bool = False,
    out_fp8: Optional[torch.Tensor] = None,
    stream_out: Optional[torch.Tensor] = None,
) -> None:
    """Drop-in for attn_res_hip at small T (see module docstring).

    stream_out : optional [T, H] (row-strided view ok) receiving the bf16
                 pre-norm mixture.
    """
    T, H = prefix_sum.shape
    assert supported(H, nvb), (H, nvb)
    assert addend is None or prefix_out is not None, "addend requires prefix_out"
    assert cw.dtype == torch.float32 and prefix_sum.dtype == torch.bfloat16
    vpt, threads = _launch_shape(H)
    flags = (
        (1 if addend is not None else 0)
        | (2 if write_prefix else 0)
        | (4 if ow is not None else 0)
        | (8 if out_fp8 is not None else 0)
        | (16 if stream_out is not None else 0)
    )
    # Unused operands still need a real tensor; the flags gate every access.
    _module(vpt, threads).run(
        prefix_sum,
        addend if addend is not None else prefix_sum,
        prefix_out if addend is not None else prefix_sum,
        bank,
        cw,
        ow if ow is not None else prefix_sum,
        out,
        out_fp8.view(torch.uint8) if out_fp8 is not None else out,
        stream_out if stream_out is not None else out,
        nvb,
        flags,
        float(score_eps),
        float(out_eps),
    )
