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

_ACCEPT_PUBLISH_JIT_VERSION = "v2"


@cache_once
def _jit_dflash_accept_publish_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        f"dflash_accept_publish_{_ACCEPT_PUBLISH_JIT_VERSION}",
        *args,
        cuda_files=["dflash/accept_publish.cuh"],
        cuda_wrappers=[
            (
                "dflash_accept_publish",
                f"DFlashAcceptPublishKernel<{args}>::run",
            )
        ],
    )


def accept_publish_cuda(
    committed_len: torch.Tensor,
    reserved_len: torch.Tensor,
    next_verified_id: torch.Tensor,
    generation: torch.Tensor,
    status_flags: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_generation: torch.Tensor,
    emit_ids: torch.Tensor,
    target_top1: torch.Tensor,
    active_mask: torch.Tensor,
    eos_ids: torch.Tensor,
    stop_ids: torch.Tensor,
    accept_lens_out: torch.Tensor,
    commit_lens_out: torch.Tensor,
    bonus_ids_out: torch.Tensor,
    gpu_stop_flags_out: torch.Tensor,
    oob_flags_out: torch.Tensor,
) -> None:
    module = _jit_dflash_accept_publish_module()
    module.dflash_accept_publish(
        committed_len,
        reserved_len,
        next_verified_id,
        generation,
        status_flags,
        req_pool_indices,
        req_generation,
        emit_ids,
        target_top1,
        active_mask,
        eos_ids,
        stop_ids,
        accept_lens_out,
        commit_lens_out,
        bonus_ids_out,
        gpu_stop_flags_out,
        oob_flags_out,
    )
