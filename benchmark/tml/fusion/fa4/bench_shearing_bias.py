"""Per-kernel micro-bench for the ShearingBias kernel (bias staging for FA4 rel_bias).

Times ONLY the compiled ShearingBias launch — the CuSeqlensToBlocks prep tensors are
built once outside the timed region — at the exact shapes the kvcache path feeds it
(varlen q, seqused_k = cache_seqlens, pack_gqa, TML full-attn head config). The
compile/call sequence mirrors interface.py::_flash_attn_fwd's rel_bias block.

Impls:
  orig      -> clamp_subtiles=False (grid z always tile_m/rows_per_cta = 32 subtiles)
  clamp     -> clamp_subtiles=True  (grid z = ceil(min(tile_m, eff_seqlen_q)/rows_per_cta))
  clamp_r8  -> clamp + rows_per_cta=8  (256 threads/CTA)
  clamp_r16 -> clamp + rows_per_cta=16 (512 threads/CTA)

`clamp` asserts its output buffer is bitwise identical to `orig` (both buffers are
zeroed first, so unwritten rows must match too). End-to-end semantic correctness of
the kernel itself is covered by test_shear_bias_precision.py / the bitwise suite.

Run:  python benchmark/tml/fusion/fa4/bench_shearing_bias.py
"""

import os

import torch

import cutlass.cute as cute

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.flash_attn.cute.cute_dsl_utils import to_cute_tensor
from sglang.jit_kernel.flash_attn.cute.cu_blocks_kernels import CuSeqlensToBlocksKernel
from sglang.jit_kernel.flash_attn.cute.shearing_bias import ShearingBias

DEV = "cuda"
# TML full-attn: 64 q heads, 8 kv heads (TP1 view). Per-rank: TP4 -> 16/2,
# TP8 -> 8/1. Override to bench at prod per-rank head counts.
NH = int(os.environ.get("NH", "64"))
NHK = int(os.environ.get("NHK", "8"))
QHPK = NH // NHK  # pack_gqa qhead_per_kvhead
REL_EXTENT = 128
REL_EXTENT_PADDED = REL_EXTENT + 256
TILE_M = 128  # group_tile_bias
ROWS_PER_CTA = 4
# (batch, q_len, kv_seqlen): decode sweep incl. large batch / long context, plus
# one chunked-prefill row. kv_seqlen only moves each row's n_idx window math.
SHAPES = [
    (1, 1, 512), (8, 1, 512), (64, 1, 512), (128, 1, 512), (256, 1, 512),
    (8, 256, 512),
]

_compile_cache = {}
_REFS = {}


VARIANTS = {
    "orig": dict(rows_per_cta=ROWS_PER_CTA, clamp_subtiles=False),
    "clamp": dict(rows_per_cta=ROWS_PER_CTA, clamp_subtiles=True),
    "clamp_r8": dict(rows_per_cta=8, clamp_subtiles=True),
    "clamp_r16": dict(rows_per_cta=16, clamp_subtiles=True),
}


def _compiled_shear(variant: str, example):
    if variant in _compile_cache:
        return _compile_cache[variant]
    rel_bias, bias, max_q, max_k, cu_q, seqused_k, cu_blocks, b2b = example
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        ShearingBias(
            REL_EXTENT,
            is_causal=True,
            is_local=False,
            pack_gqa=True,
            qhead_per_kvhead=QHPK,
            tile_m=TILE_M,
            max_m_blocks_leq_one=False,
            **VARIANTS[variant],
        ),
        to_cute_tensor(rel_bias),
        to_cute_tensor(bias),
        max_q,
        max_k,
        to_cute_tensor(cu_q, assumed_align=4, leading_dim=0),
        None,  # cu_seqlens_k
        None,  # seqused_q
        to_cute_tensor(seqused_k, assumed_align=4, leading_dim=0),
        to_cute_tensor(cu_blocks, assumed_align=4, leading_dim=0),
        to_cute_tensor(b2b, assumed_align=4, leading_dim=0),
        None,  # window_size_left
        None,  # window_size_right
        stream,
        options="--enable-tvm-ffi",
    )
    _compile_cache[variant] = compiled
    return compiled


def _compiled_prep():
    if "prep" in _compile_cache:
        return _compile_cache["prep"]
    cu_blocks = torch.empty(2, dtype=torch.int32, device=DEV)
    cu_q = torch.zeros(2, dtype=torch.int32, device=DEV)
    b2b = torch.empty(1, dtype=torch.int32, device=DEV)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        CuSeqlensToBlocksKernel(tile=TILE_M, seqlen_multiple=QHPK),
        to_cute_tensor(cu_blocks, assumed_align=4, leading_dim=0),
        to_cute_tensor(cu_q, assumed_align=4, leading_dim=0),
        to_cute_tensor(b2b, assumed_align=4, leading_dim=0),
        stream,
        options="--enable-tvm-ffi",
    )
    _compile_cache["prep"] = compiled
    return compiled


def _make_case(b: int, q_len: int, kv_seqlen: int):
    torch.manual_seed(0)
    total_q = b * q_len
    rel_bias = torch.randn(total_q, NH, REL_EXTENT, device=DEV, dtype=torch.bfloat16) * 0.1
    bias = torch.zeros(
        total_q + TILE_M, NH, REL_EXTENT_PADDED, device=DEV, dtype=torch.bfloat16
    )
    cu_q = torch.arange(0, total_q + 1, q_len, dtype=torch.int32, device=DEV)
    seqused_k = torch.full((b,), kv_seqlen, dtype=torch.int32, device=DEV)
    cu_blocks = torch.empty(b + 1, dtype=torch.int32, device=DEV)
    total_blocks_max = (total_q * QHPK + b * (TILE_M - 1)) // TILE_M
    b2b = torch.empty(total_blocks_max, dtype=torch.int32, device=DEV)
    _compiled_prep()(cu_blocks, cu_q, b2b)
    torch.cuda.synchronize()
    return rel_bias, bias, q_len, kv_seqlen, cu_q, seqused_k, cu_blocks, b2b


@marker.parametrize("b, q_len, kv_seqlen", SHAPES)
@marker.benchmark("impl", list(VARIANTS))
def bench_shearing_bias(b: int, q_len: int, kv_seqlen: int, impl: str):
    case = _make_case(b, q_len, kv_seqlen)
    rel_bias, bias, max_q, max_k, cu_q, seqused_k, cu_blocks, b2b = case
    compiled = _compiled_shear(impl, case)

    def fn(rel_bias, bias, cu_q, seqused_k, cu_blocks, b2b):
        compiled(
            rel_bias, bias, max_q, max_k, cu_q, None, None, seqused_k,
            cu_blocks, b2b, None, None,
        )

    bias.zero_()
    fn(rel_bias, bias, cu_q, seqused_k, cu_blocks, b2b)
    torch.cuda.synchronize()
    key = (b, q_len, kv_seqlen)
    if impl == "orig":
        _REFS[key] = bias.clone()
    else:
        ref = _REFS[key]
        n_diff = (bias.view(torch.int16) != ref.view(torch.int16)).sum().item()
        status = "PASS" if n_diff == 0 else f"FAIL ({n_diff} elems differ)"
        print(f"[bitwise clamp-vs-orig] b={b} q={q_len} kv={kv_seqlen}: {status}", flush=True)
        assert n_diff == 0

    return marker.do_bench(
        fn,
        input_args=(rel_bias, bias, cu_q, seqused_k, cu_blocks, b2b),
        memory_output=(bias,),
    )


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    bench_shearing_bias.run()
