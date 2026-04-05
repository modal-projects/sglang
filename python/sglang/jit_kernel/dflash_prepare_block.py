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


_PREPARE_BLOCK_JIT_VERSION = "v2"


@cache_once
def _jit_dflash_prepare_block_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        f"dflash_prepare_block_{_PREPARE_BLOCK_JIT_VERSION}",
        *args,
        cuda_files=["dflash/prepare_block.cuh"],
        cuda_wrappers=[
            (
                "dflash_prepare_block",
                f"DFlashPrepareBlockKernel<{args}>::run",
            ),
            (
                "dflash_prepare_block_fused_sample",
                f"DFlashPrepareBlockFusedSampleKernel<{args}>::run",
            ),
        ],
    )


def prepare_block_cuda(
    committed_len: torch.Tensor,
    reserved_len: torch.Tensor,
    next_verified_id: torch.Tensor,
    generation: torch.Tensor,
    status_flags: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_generation: torch.Tensor,
    req_to_token: torch.Tensor,
    active_mask_out: torch.Tensor,
    oob_flags_out: torch.Tensor,
    query_positions_out: torch.Tensor,
    query_slot_ids_out: torch.Tensor,
    query_input_ids_out: torch.Tensor,
    emit_ids_out: torch.Tensor,
) -> None:
    module = _jit_dflash_prepare_block_module()
    module.dflash_prepare_block(
        committed_len,
        reserved_len,
        next_verified_id,
        generation,
        status_flags,
        req_pool_indices,
        req_generation,
        req_to_token,
        active_mask_out,
        oob_flags_out,
        query_positions_out,
        query_slot_ids_out,
        query_input_ids_out,
        emit_ids_out,
    )


def prepare_block_fused_sample_cuda(
    committed_len: torch.Tensor,
    reserved_len: torch.Tensor,
    next_verified_id: torch.Tensor,
    generation: torch.Tensor,
    status_flags: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_generation: torch.Tensor,
    req_to_token: torch.Tensor,
    active_mask_out: torch.Tensor,
    oob_flags_out: torch.Tensor,
    query_positions_out: torch.Tensor,
    query_slot_ids_out: torch.Tensor,
    query_input_ids_out: torch.Tensor,
    emit_ids_out: torch.Tensor,
    sample_indices_out: torch.Tensor,
    mask_token_id: int,
    dummy_slot_id: int,
) -> None:
    module = _jit_dflash_prepare_block_module()
    module.dflash_prepare_block_fused_sample(
        committed_len,
        reserved_len,
        next_verified_id,
        generation,
        status_flags,
        req_pool_indices,
        req_generation,
        req_to_token,
        active_mask_out,
        oob_flags_out,
        query_positions_out,
        query_slot_ids_out,
        query_input_ids_out,
        emit_ids_out,
        sample_indices_out,
        int(mask_token_id),
        int(dummy_slot_id),
    )
