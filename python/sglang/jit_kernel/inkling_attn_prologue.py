"""Fused target-verify attention prologue: {k/v sconv + save_windows + qk-norm
+ KV-cache store} in one kernel (csrc/tml/inkling_attn_prologue_fused.cuh)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_attn_prologue_module(
    dtype: torch.dtype, w: int, use_silu: bool, use_residual: bool
) -> Module:
    args = make_cpp_args(dtype, w, use_silu, use_residual)
    return load_jit(
        "inkling_attn_prologue_fused",
        *args,
        cuda_files=["tml/inkling_attn_prologue_fused.cuh"],
        cuda_wrappers=[("attn_prologue", f"AttnPrologueKernel<{args}>::run")],
    )


def compile_inkling_attn_prologue(
    dtype: torch.dtype, w: int, use_silu: bool, use_residual: bool
) -> None:
    _jit_attn_prologue_module(dtype, w, use_silu, use_residual)


def inkling_attn_prologue_verify(
    qkvr: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    k_inter: torch.Tensor,
    v_inter: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    eps: float,
    loc: torch.Tensor,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    q_off: int,
    k_off: int,
    v_off: int,
    dq: int,
    dkv: int,
    draft_token_num: int,
    activation: str | None = None,
    use_residual: bool = True,
    do_store: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns fresh contiguous (q_normed, k_normed, v_conv) [T, dq/dkv];
    KV rows are also scattered into k_buf/v_buf at ``loc`` (the attention call
    should pass save_kv_cache=False)."""
    t = qkvr.shape[0]
    q_out = torch.empty(t, dq, dtype=qkvr.dtype, device=qkvr.device)
    k_out = torch.empty(t, dkv, dtype=qkvr.dtype, device=qkvr.device)
    v_out = torch.empty(t, dkv, dtype=qkvr.dtype, device=qkvr.device)
    if activation == "swish":
        activation = "silu"
    use_silu = activation in ("silu", "swish")
    w = k_weight.shape[1]
    module = _jit_attn_prologue_module(qkvr.dtype, w, use_silu, use_residual)
    hkv = dkv // 128
    module.attn_prologue(
        qkvr, k_cache, v_cache, cache_indices.to(torch.int32),
        cache_mask, k_weight, v_weight, k_inter, v_inter,
        q_gamma, k_gamma, float(eps), q_out, k_out, v_out,
        loc, k_buf.view(-1, hkv * 128), v_buf.view(-1, hkv * 128),
        int(q_off), int(k_off), int(v_off), int(draft_token_num), int(do_store),
    )
    return q_out, k_out, v_out
