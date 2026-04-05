from __future__ import annotations

import torch

from sglang.jit_kernel.dflash_direct_embedding import direct_embedding_cuda
from sglang.srt.speculative.dflash.contracts import DFlashDirectEmbeddingResult
from sglang.srt.speculative.dflash.kernels.direct_embedding import (
    DFlashDirectEmbeddingWorkspace,
    create_direct_embedding_workspace,
    direct_embedding_result_from_workspace,
)
from sglang.srt.speculative.dflash.reference.direct_embedding import (
    _validate_direct_embedding_inputs,
)


def direct_embedding_jit_fast(
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
    if batch_size == 0:
        return direct_embedding_result_from_workspace(workspace)
    direct_embedding_cuda(
        embedding_table.contiguous(),
        first_token_ids.contiguous(),
        workspace.query_embeds,
        block_size=block_size,
        mask_token_id=mask_token_id,
    )
    return direct_embedding_result_from_workspace(workspace)


def direct_embedding_jit(
    *,
    embedding_table: torch.Tensor,
    first_token_ids: torch.Tensor,
    block_size: int,
    mask_token_id: int,
) -> DFlashDirectEmbeddingResult:
    return direct_embedding_jit_fast(
        embedding_table=embedding_table,
        first_token_ids=first_token_ids,
        block_size=block_size,
        mask_token_id=mask_token_id,
        workspace=None,
    )
