from __future__ import annotations

import torch

from sglang.srt.speculative.dflash.contracts import (
    DFlashKVCache,
    DFlashMaterializerConfig,
)
from sglang.srt.speculative.dflash.reference.kv_prefix_write import (
    _validate_commit_write_inputs,
    _validate_prompt_write_inputs,
)


def write_prompt_index_copy_control(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    slot_ids: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    inplace: bool = False,
) -> DFlashKVCache:
    slot_ids, cache_k, cache_v = _validate_prompt_write_inputs(
        cache=cache,
        config=config,
        slot_ids=slot_ids,
        cache_k=cache_k,
        cache_v=cache_v,
    )
    updated = cache if inplace else cache.clone()
    index_slot_ids = slot_ids.to(dtype=torch.int64)
    for layer_idx in range(config.num_layers):
        updated.k_cache[layer_idx].index_copy_(0, index_slot_ids, cache_k[layer_idx])
        updated.v_cache[layer_idx].index_copy_(0, index_slot_ids, cache_v[layer_idx])
    return updated


def write_commit_prefix_rowwise_control(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    slot_ids_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    inplace: bool = False,
) -> DFlashKVCache:
    slot_ids_2d, commit_lens, cache_k, cache_v = _validate_commit_write_inputs(
        cache=cache,
        config=config,
        slot_ids_2d=slot_ids_2d,
        commit_lens=commit_lens,
        cache_k=cache_k,
        cache_v=cache_v,
    )
    updated = cache if inplace else cache.clone()
    _, bs, _, _, _ = cache_k.shape
    for layer_idx in range(config.num_layers):
        for row in range(bs):
            keep = int(commit_lens[row].item())
            if keep <= 0:
                continue
            slot_slice = slot_ids_2d[row, :keep].to(dtype=torch.int64)
            updated.k_cache[layer_idx].index_copy_(
                0,
                slot_slice,
                cache_k[layer_idx, row, :keep],
            )
            updated.v_cache[layer_idx].index_copy_(
                0,
                slot_slice,
                cache_v[layer_idx, row, :keep],
            )
    return updated


def write_commit_prefix_flatten_control(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    slot_ids_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    inplace: bool = False,
) -> DFlashKVCache:
    slot_ids_2d, commit_lens, cache_k, cache_v = _validate_commit_write_inputs(
        cache=cache,
        config=config,
        slot_ids_2d=slot_ids_2d,
        commit_lens=commit_lens,
        cache_k=cache_k,
        cache_v=cache_v,
    )
    updated = cache if inplace else cache.clone()
    _, _, block_size, _, _ = cache_k.shape
    offsets = torch.arange(
        block_size,
        device=slot_ids_2d.device,
        dtype=torch.int32,
    )[None, :]
    valid_mask = offsets < commit_lens[:, None]
    valid_slot_ids = slot_ids_2d[valid_mask]
    if int(valid_slot_ids.numel()) == 0:
        return updated
    valid_slot_ids = valid_slot_ids.to(dtype=torch.int64)
    for layer_idx in range(config.num_layers):
        updated.k_cache[layer_idx].index_copy_(
            0,
            valid_slot_ids,
            cache_k[layer_idx][valid_mask],
        )
        updated.v_cache[layer_idx].index_copy_(
            0,
            valid_slot_ids,
            cache_v[layer_idx][valid_mask],
        )
    return updated


def write_commit_masked_dummy_control(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    slot_ids_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    dummy_slot_id: int,
    inplace: bool = False,
) -> DFlashKVCache:
    slot_ids_2d, commit_lens, cache_k, cache_v = _validate_commit_write_inputs(
        cache=cache,
        config=config,
        slot_ids_2d=slot_ids_2d,
        commit_lens=commit_lens,
        cache_k=cache_k,
        cache_v=cache_v,
    )
    if dummy_slot_id < 0 or dummy_slot_id >= cache.num_slots:
        raise ValueError(
            f"dummy_slot_id must be in [0, num_slots), got {dummy_slot_id}."
        )
    updated = cache if inplace else cache.clone()
    _, bs, block_size, _, _ = cache_k.shape
    offsets = torch.arange(
        block_size,
        device=slot_ids_2d.device,
        dtype=torch.int32,
    )[None, :]
    valid_mask = offsets < commit_lens[:, None]
    safe_slot_ids = (
        torch.where(
            valid_mask,
            slot_ids_2d,
            torch.full_like(slot_ids_2d, int(dummy_slot_id)),
        )
        .view(bs * block_size)
        .to(dtype=torch.int64)
    )
    flat_k = cache_k.view(
        config.num_layers, bs * block_size, config.num_kv_heads, config.head_dim
    )
    flat_v = cache_v.view(
        config.num_layers, bs * block_size, config.num_kv_heads, config.head_dim
    )
    for layer_idx in range(config.num_layers):
        updated.k_cache[layer_idx].index_copy_(0, safe_slot_ids, flat_k[layer_idx])
        updated.v_cache[layer_idx].index_copy_(0, safe_slot_ids, flat_v[layer_idx])
    return updated
