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
def _jit_dflash_accept_bonus_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "dflash_accept_bonus",
        *args,
        cuda_files=["dflash/accept_bonus.cuh"],
        cuda_wrappers=[
            (
                "dflash_accept_bonus",
                f"DFlashAcceptBonusKernel<{args}>::run",
            )
        ],
    )


def accept_bonus_cuda(
    emit_ids: torch.Tensor,
    target_top1: torch.Tensor,
    active_mask: torch.Tensor,
    eos_ids: torch.Tensor,
    stop_ids: torch.Tensor,
    accept_lens_out: torch.Tensor,
    commit_lens_out: torch.Tensor,
    bonus_ids_out: torch.Tensor,
    gpu_stop_flags_out: torch.Tensor,
) -> None:
    module = _jit_dflash_accept_bonus_module()
    module.dflash_accept_bonus(
        emit_ids,
        target_top1,
        active_mask,
        eos_ids,
        stop_ids,
        accept_lens_out,
        commit_lens_out,
        bonus_ids_out,
        gpu_stop_flags_out,
    )
