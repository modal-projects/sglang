from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dflash.contracts import DFlashDirectEmbeddingResult
from sglang.srt.speculative.dflash.kernels.direct_embedding import (
    DFlashDirectEmbeddingWorkspace,
    create_direct_embedding_workspace,
    direct_embedding_result_from_workspace,
)
from sglang.srt.speculative.dflash.reference.direct_embedding import (
    _validate_direct_embedding_inputs,
)


def _pick_launch_config(hidden_size: int) -> tuple[int, int]:
    if hidden_size <= 64:
        return 64, 2
    if hidden_size <= 128:
        return 128, 4
    if hidden_size <= 256:
        return 256, 4
    return 512, 8


@triton.jit
def _direct_embedding_kernel(
    embedding_table_ptr,
    first_token_ids_ptr,
    output_ptr,
    table_row_stride,
    first_token_stride,
    output_row_stride,
    block_size,
    hidden_size,
    mask_token_id,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    batch_idx = row_idx // block_size
    block_col = row_idx % block_size
    first_token_id = tl.load(first_token_ids_ptr + batch_idx * first_token_stride)
    token_id = tl.where(block_col == 0, first_token_id, mask_token_id)

    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size
    src_ptr = embedding_table_ptr + token_id * table_row_stride + offs
    dst_ptr = output_ptr + row_idx * output_row_stride + offs
    tl.store(dst_ptr, tl.load(src_ptr, mask=mask, other=0), mask=mask)


def direct_embedding_triton_fast(
    *,
    embedding_table: torch.Tensor,
    first_token_ids: torch.Tensor,
    block_size: int,
    mask_token_id: int,
    workspace: DFlashDirectEmbeddingWorkspace | None = None,
) -> DFlashDirectEmbeddingResult:
    embedding_table, first_token_ids = _validate_direct_embedding_inputs(
        embedding_table=embedding_table,
        first_token_ids=first_token_ids,
        block_size=block_size,
        mask_token_id=mask_token_id,
    )
    batch_size = int(first_token_ids.numel())
    hidden_size = int(embedding_table.shape[1])
    if workspace is None:
        workspace = create_direct_embedding_workspace(
            bucket_bs=batch_size,
            block_size=block_size,
            hidden_size=hidden_size,
            dtype=embedding_table.dtype,
            device=embedding_table.device,
        )
    if tuple(workspace.query_embeds.shape) != (batch_size, block_size, hidden_size):
        raise ValueError(
            "direct embedding workspace shape mismatch. "
            f"Expected {(batch_size, block_size, hidden_size)}, "
            f"got {tuple(workspace.query_embeds.shape)}."
        )
    total_rows = batch_size * block_size
    if total_rows == 0:
        return direct_embedding_result_from_workspace(workspace)

    output_flat = workspace.query_embeds.view(total_rows, hidden_size)
    vec_block, num_warps = _pick_launch_config(hidden_size)
    grid = (total_rows, triton.cdiv(hidden_size, vec_block))
    _direct_embedding_kernel[grid](
        embedding_table.contiguous(),
        first_token_ids.contiguous(),
        output_flat,
        embedding_table.stride(0),
        first_token_ids.stride(0),
        output_flat.stride(0),
        block_size,
        hidden_size,
        int(mask_token_id),
        BLOCK_SIZE=vec_block,
        num_warps=num_warps,
    )
    return direct_embedding_result_from_workspace(workspace)


def direct_embedding_triton(
    *,
    embedding_table: torch.Tensor,
    first_token_ids: torch.Tensor,
    block_size: int,
    mask_token_id: int,
) -> DFlashDirectEmbeddingResult:
    return direct_embedding_triton_fast(
        embedding_table=embedding_table,
        first_token_ids=first_token_ids,
        block_size=block_size,
        mask_token_id=mask_token_id,
        workspace=None,
    )
