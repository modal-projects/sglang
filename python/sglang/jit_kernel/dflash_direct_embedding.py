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


_DIRECT_EMBEDDING_JIT_VERSION = "v1"


def _require_supported_row_bytes(row_bytes: int) -> None:
    if row_bytes <= 0 or row_bytes % 4 != 0:
        raise ValueError(
            "DFlash direct embedding expects row_bytes to be positive and divisible by 4, "
            f"got {row_bytes}."
        )


@cache_once
def _jit_dflash_direct_embedding_module(row_bytes: int) -> Module:
    _require_supported_row_bytes(row_bytes)
    args = make_cpp_args(row_bytes, is_arch_support_pdl())
    return load_jit(
        f"dflash_direct_embedding_{_DIRECT_EMBEDDING_JIT_VERSION}",
        *args,
        cuda_files=["dflash/direct_embedding.cuh"],
        cuda_wrappers=[
            (
                "dflash_direct_embedding",
                f"DFlashDirectEmbeddingKernel<{args}>::run",
            )
        ],
    )


def direct_embedding_cuda(
    embedding_table: torch.Tensor,
    first_token_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    block_size: int,
    mask_token_id: int,
) -> None:
    row_bytes = int(output.shape[-1]) * int(output.element_size())
    module = _jit_dflash_direct_embedding_module(row_bytes)
    module.dflash_direct_embedding(
        embedding_table,
        first_token_ids,
        output,
        int(block_size),
        int(mask_token_id),
    )
