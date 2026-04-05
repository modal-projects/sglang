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


@cache_once
def _jit_dflash_publish_state_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "dflash_publish_state",
        *args,
        cuda_files=["dflash/publish_state.cuh"],
        cuda_wrappers=[
            (
                "dflash_publish_state",
                f"DFlashPublishStateKernel<{args}>::run",
            )
        ],
    )


def publish_state_cuda(
    committed_len: torch.Tensor,
    reserved_len: torch.Tensor,
    next_verified_id: torch.Tensor,
    generation: torch.Tensor,
    status_flags: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_generation: torch.Tensor,
    commit_lens: torch.Tensor,
    bonus_ids: torch.Tensor,
    gpu_stop_flags: torch.Tensor,
    oob_flags_out: torch.Tensor,
) -> None:
    module = _jit_dflash_publish_state_module()
    module.dflash_publish_state(
        committed_len,
        reserved_len,
        next_verified_id,
        generation,
        status_flags,
        req_pool_indices,
        req_generation,
        commit_lens,
        bonus_ids,
        gpu_stop_flags,
        oob_flags_out,
    )
