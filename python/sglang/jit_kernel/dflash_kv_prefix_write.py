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


def _require_supported_row_bytes(row_bytes: int) -> None:
    if row_bytes <= 0 or row_bytes % 4 != 0:
        raise ValueError(
            f"DFlash KV prefix write expects row_bytes to be positive and divisible by 4, got {row_bytes}."
        )


def _resolve_explicit_num_split(row_bytes: int, num_split: int) -> int:
    _require_supported_row_bytes(row_bytes)
    if num_split not in (1, 2, 4):
        raise ValueError(f"Unsupported num_split={num_split}. Expected one of 1, 2, 4.")
    if row_bytes % (num_split * 128) != 0:
        raise ValueError(
            f"row_bytes={row_bytes} is not compatible with num_split={num_split}."
        )
    return num_split


def _resolve_prompt_num_split(row_bytes: int, num_split: int) -> int:
    if num_split != 0:
        return _resolve_explicit_num_split(row_bytes, num_split)
    _require_supported_row_bytes(row_bytes)
    # Prompt write favored split=1 across the first B200 sweeps.
    return 1


def _resolve_commit_num_split(row_bytes: int, num_split: int) -> int:
    if num_split != 0:
        return _resolve_explicit_num_split(row_bytes, num_split)
    _require_supported_row_bytes(row_bytes)
    # Commit write behaved differently from prompt write on B200. The first
    # sweeps favored split=2 for the 512-byte and 2048-byte row sizes we expect
    # to matter most, while 1024-byte and larger rows stayed closer to split=1.
    if row_bytes == 512 and row_bytes % (2 * 128) == 0:
        return 2
    if row_bytes == 2048 and row_bytes % (2 * 128) == 0:
        return 2
    return 1


@cache_once
def _jit_dflash_prompt_kv_prefix_write_module(row_bytes: int) -> Module:
    _require_supported_row_bytes(row_bytes)
    args = make_cpp_args(row_bytes, is_arch_support_pdl())
    return load_jit(
        "dflash_prompt_kv_prefix_write",
        *args,
        cuda_files=["dflash/kv_prefix_write.cuh"],
        cuda_wrappers=[
            (
                "dflash_prompt_kv_prefix_write",
                f"DFlashPromptKVPrefixWriteKernel<{args}>::run",
            )
        ],
    )


@cache_once
def _jit_dflash_commit_kv_prefix_write_module(row_bytes: int) -> Module:
    _require_supported_row_bytes(row_bytes)
    args = make_cpp_args(row_bytes, is_arch_support_pdl())
    return load_jit(
        "dflash_commit_kv_prefix_write",
        *args,
        cuda_files=["dflash/kv_prefix_write.cuh"],
        cuda_wrappers=[
            (
                "dflash_commit_kv_prefix_write",
                f"DFlashCommitKVPrefixWriteKernel<{args}>::run",
            )
        ],
    )


def prompt_kv_prefix_write(
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    slot_ids: torch.Tensor,
    *,
    num_split: int = 0,
) -> None:
    row_bytes = int(src_k.shape[-1]) * int(src_k.element_size())
    resolved_num_split = _resolve_prompt_num_split(row_bytes, num_split)
    module = _jit_dflash_prompt_kv_prefix_write_module(row_bytes)
    module.dflash_prompt_kv_prefix_write(
        src_k, src_v, dst_k, dst_v, slot_ids, resolved_num_split
    )


def commit_kv_prefix_write(
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    slot_ids_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    *,
    num_split: int = 0,
) -> None:
    row_bytes = int(src_k.shape[-1]) * int(src_k.element_size())
    resolved_num_split = _resolve_commit_num_split(row_bytes, num_split)
    module = _jit_dflash_commit_kv_prefix_write_module(row_bytes)
    module.dflash_commit_kv_prefix_write(
        src_k,
        src_v,
        dst_k,
        dst_v,
        slot_ids_2d,
        commit_lens,
        resolved_num_split,
    )
