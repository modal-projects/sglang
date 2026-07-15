"""VENDORED BASELINE — `copy_if_needed` + `copy_if_needed_kernel`, verbatim from
python/sglang/tml/layers/sconv.py before the decode-track-fusion working-tree change
removed them. Provides the OLD separate-copy baseline for opt5/opt6/opt8 so those
benchmarks keep working after `copy_if_needed` is deleted from the layer. DO NOT EDIT.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def copy_if_needed_kernel(
    src_tensor_ptr,
    dst_tensor_ptr,
    mask_ptr,
    src_indices_ptr,
    dst_indices_ptr,
    stride_0,  # stride for first dimension (batch/pool index)
    numel_per_row: tl.constexpr,  # total elements per row
    BLOCK_SIZE: tl.constexpr,
):
    """For each batch element, if the track mask is True, copy the entire row from
    src_indices[i] to dst_indices[i]. Grid: (batch_size,)."""
    batch_idx = tl.program_id(0)
    track_mask = tl.load(mask_ptr + batch_idx)
    if not track_mask:
        return
    src_idx = tl.load(src_indices_ptr + batch_idx)
    dst_idx = tl.load(dst_indices_ptr + batch_idx)
    for offset in range(0, numel_per_row, BLOCK_SIZE):
        element_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = element_indices < numel_per_row
        src_ptr = src_tensor_ptr + src_idx * stride_0 + element_indices
        dst_ptr = dst_tensor_ptr + dst_idx * stride_0 + element_indices
        data = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_ptr, data, mask=mask)


def copy_if_needed(
    src_tensor: torch.Tensor,
    mask: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    batch_size: int,
    dst_tensor: torch.Tensor | None = None,
):
    """Copy `src_tensor[src_indices[i]] -> dst_tensor[dst_indices[i]]` where mask[i]."""
    numel_per_row = src_tensor[0].numel()
    assert dst_tensor is None or src_tensor.stride(0) == dst_tensor.stride(0), (
        "Src and dst tensors must have the same outter stride"
    )
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    copy_if_needed_kernel[grid](
        src_tensor,
        dst_tensor if dst_tensor is not None else src_tensor,
        mask,
        src_indices,
        dst_indices,
        src_tensor.stride(0),
        numel_per_row,
        BLOCK_SIZE,
    )
