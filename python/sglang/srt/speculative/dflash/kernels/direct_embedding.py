from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.speculative.dflash.contracts import DFlashDirectEmbeddingResult
from sglang.srt.speculative.dflash.reference.direct_embedding import (
    _validate_direct_embedding_inputs,
)


@dataclass
class DFlashDirectEmbeddingWorkspace:
    query_embeds: torch.Tensor


def create_direct_embedding_workspace(
    *,
    bucket_bs: int,
    block_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> DFlashDirectEmbeddingWorkspace:
    if bucket_bs < 0:
        raise ValueError(f"bucket_bs must be non-negative, got {bucket_bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
    device = torch.device(device)
    return DFlashDirectEmbeddingWorkspace(
        query_embeds=torch.empty(
            (bucket_bs, block_size, hidden_size),
            dtype=dtype,
            device=device,
        )
    )


def direct_embedding_result_from_workspace(
    workspace: DFlashDirectEmbeddingWorkspace,
) -> DFlashDirectEmbeddingResult:
    return DFlashDirectEmbeddingResult(query_embeds=workspace.query_embeds)


def direct_embedding_control(
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
    mask_embedding = embedding_table.narrow(0, int(mask_token_id), 1).view(
        1, 1, hidden_size
    )
    workspace.query_embeds.copy_(mask_embedding.expand_as(workspace.query_embeds))
    if batch_size > 0:
        workspace.query_embeds[:, 0, :].copy_(
            embedding_table.index_select(0, first_token_ids.to(dtype=torch.int64))
        )
    return direct_embedding_result_from_workspace(workspace)
