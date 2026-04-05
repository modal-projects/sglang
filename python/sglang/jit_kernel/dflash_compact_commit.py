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


_COMPACT_COMMIT_JIT_VERSION = "v3"


@cache_once
def _jit_dflash_compact_commit_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(is_arch_support_pdl(), dtype)
    return load_jit(
        f"dflash_compact_commit_{_COMPACT_COMMIT_JIT_VERSION}",
        *args,
        cuda_files=["dflash/compact_commit.cuh"],
        cuda_wrappers=[
            (
                "dflash_compact_commit",
                f"DFlashCompactCommitKernel<{args}>::run",
            )
        ],
    )


def compact_commit_cuda(
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    row_offsets: torch.Tensor,
    hidden_out: torch.Tensor,
    positions_out: torch.Tensor,
    slot_ids_out: torch.Tensor,
) -> None:
    module = _jit_dflash_compact_commit_module(verify_hidden.dtype)
    module.dflash_compact_commit(
        verify_hidden,
        positions,
        slot_ids_2d,
        commit_lens,
        row_offsets,
        hidden_out,
        positions_out,
        slot_ids_out,
    )
