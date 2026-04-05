from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.speculative.dflash.contracts import DFlashDirectEmbeddingResult


def _validate_embedding_lookup_inputs(
    *,
    embedding_table: torch.Tensor,
    query_input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if embedding_table.ndim != 2:
        raise ValueError(
            "embedding_table must be rank-2 [vocab_size, hidden_size]. "
            f"Got {tuple(embedding_table.shape)}."
        )
    if query_input_ids.ndim != 2:
        raise ValueError(
            "query_input_ids must be rank-2 [batch_size, block_size]. "
            f"Got {tuple(query_input_ids.shape)}."
        )
    if query_input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "query_input_ids must use int32 or int64 dtype. "
            f"Got {query_input_ids.dtype}."
        )
    if query_input_ids.device != embedding_table.device:
        raise ValueError(
            "query_input_ids and embedding_table must be on the same device. "
            f"Got {query_input_ids.device} and {embedding_table.device}."
        )
    vocab_size = int(embedding_table.shape[0])
    if int(query_input_ids.numel()) > 0:
        min_id = int(query_input_ids.min().item())
        max_id = int(query_input_ids.max().item())
        if min_id < 0 or max_id >= vocab_size:
            raise ValueError(
                "query_input_ids contain out-of-range token ids. "
                f"Expected [0, {vocab_size}), got [{min_id}, {max_id}]."
            )
    return embedding_table, query_input_ids


def _validate_direct_embedding_inputs(
    *,
    embedding_table: torch.Tensor,
    first_token_ids: torch.Tensor,
    block_size: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if embedding_table.ndim != 2:
        raise ValueError(
            "embedding_table must be rank-2 [vocab_size, hidden_size]. "
            f"Got {tuple(embedding_table.shape)}."
        )
    if first_token_ids.ndim != 1:
        raise ValueError(
            "first_token_ids must be rank-1 [batch_size]. "
            f"Got {tuple(first_token_ids.shape)}."
        )
    if first_token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "first_token_ids must use int32 or int64 dtype. "
            f"Got {first_token_ids.dtype}."
        )
    if first_token_ids.device != embedding_table.device:
        raise ValueError(
            "first_token_ids and embedding_table must be on the same device. "
            f"Got {first_token_ids.device} and {embedding_table.device}."
        )
    vocab_size = int(embedding_table.shape[0])
    if mask_token_id < 0 or mask_token_id >= vocab_size:
        raise ValueError(
            f"mask_token_id must be in [0, {vocab_size}), got {mask_token_id}."
        )
    if int(first_token_ids.numel()) > 0:
        min_id = int(first_token_ids.min().item())
        max_id = int(first_token_ids.max().item())
        if min_id < 0 or max_id >= vocab_size:
            raise ValueError(
                "first_token_ids contain out-of-range token ids. "
                f"Expected [0, {vocab_size}), got [{min_id}, {max_id}]."
            )
    return embedding_table, first_token_ids


def embedding_lookup_reference(
    *,
    embedding_table: torch.Tensor,
    query_input_ids: torch.Tensor,
) -> DFlashDirectEmbeddingResult:
    embedding_table, query_input_ids = _validate_embedding_lookup_inputs(
        embedding_table=embedding_table,
        query_input_ids=query_input_ids,
    )
    return DFlashDirectEmbeddingResult(
        query_embeds=F.embedding(query_input_ids, embedding_table),
    )


def direct_embedding_reference(
    *,
    embedding_table: torch.Tensor,
    first_token_ids: torch.Tensor,
    block_size: int,
    mask_token_id: int,
) -> DFlashDirectEmbeddingResult:
    embedding_table, first_token_ids = _validate_direct_embedding_inputs(
        embedding_table=embedding_table,
        first_token_ids=first_token_ids,
        block_size=block_size,
        mask_token_id=mask_token_id,
    )
    batch_size = int(first_token_ids.numel())
    hidden_size = int(embedding_table.shape[1])
    mask_embedding = embedding_table.narrow(0, int(mask_token_id), 1).view(
        1, 1, hidden_size
    )
    output = mask_embedding.expand(batch_size, block_size, hidden_size).clone()
    if batch_size > 0:
        first_rows = embedding_table.index_select(
            0, first_token_ids.to(dtype=torch.int64)
        )
        output[:, 0, :].copy_(first_rows)
    return DFlashDirectEmbeddingResult(query_embeds=output)
