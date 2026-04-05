from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_POST_PROJECTION_JIT_VERSION = "v2"


@cache_once
def _jit_dflash_prompt_post_projection_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(is_arch_support_pdl(), dtype)
    return load_jit(
        f"dflash_prompt_post_projection_{_POST_PROJECTION_JIT_VERSION}",
        *args,
        cuda_files=["dflash/post_projection.cuh"],
        cuda_wrappers=[
            (
                "dflash_prompt_post_projection",
                f"DFlashPromptPostProjectionKernel<{args}>::run",
            )
        ],
    )


@cache_once
def _jit_dflash_commit_post_projection_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(is_arch_support_pdl(), dtype)
    return load_jit(
        f"dflash_commit_post_projection_{_POST_PROJECTION_JIT_VERSION}",
        *args,
        cuda_files=["dflash/post_projection.cuh"],
        cuda_wrappers=[
            (
                "dflash_commit_post_projection",
                f"DFlashCommitPostProjectionKernel<{args}>::run",
            )
        ],
    )


def prompt_post_projection_cuda(
    raw_k: torch.Tensor,
    raw_v: torch.Tensor,
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    slot_ids: torch.Tensor,
    positions: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_eps: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> None:
    module = _jit_dflash_prompt_post_projection_module(raw_k.dtype)
    module.dflash_prompt_post_projection(
        raw_k,
        raw_v,
        dst_k,
        dst_v,
        slot_ids,
        positions,
        k_norm_weight,
        k_norm_eps,
        cos_sin_cache,
    )


def commit_post_projection_cuda(
    raw_k: torch.Tensor,
    raw_v: torch.Tensor,
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    slot_ids_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    positions: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_eps: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> None:
    module = _jit_dflash_commit_post_projection_module(raw_k.dtype)
    module.dflash_commit_post_projection(
        raw_k,
        raw_v,
        dst_k,
        dst_v,
        slot_ids_2d,
        commit_lens,
        positions,
        k_norm_weight,
        k_norm_eps,
        cos_sin_cache,
    )
