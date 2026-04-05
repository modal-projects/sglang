from __future__ import annotations

import torch

from sglang.srt.speculative.dflash.contracts import (
    DFlashKVCache,
    DFlashMaterializerConfig,
)


def _validate_prompt_write_inputs(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    slot_ids: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache.validate(config)
    slot_ids = slot_ids.view(-1).to(device=cache.k_cache.device)
    if slot_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "slot_ids must have dtype int32 or int64 for prompt write. "
            f"Got {slot_ids.dtype}."
        )
    expected_shape = (
        config.num_layers,
        int(slot_ids.numel()),
        config.num_kv_heads,
        config.head_dim,
    )
    if tuple(cache_k.shape) != expected_shape:
        raise ValueError(
            "cache_k shape mismatch for prompt write. "
            f"Expected {expected_shape}, got {tuple(cache_k.shape)}."
        )
    if tuple(cache_v.shape) != expected_shape:
        raise ValueError(
            "cache_v shape mismatch for prompt write. "
            f"Expected {expected_shape}, got {tuple(cache_v.shape)}."
        )
    if int(slot_ids.numel()) > 0:
        if (
            int(slot_ids.min().item()) < 0
            or int(slot_ids.max().item()) >= cache.num_slots
        ):
            raise ValueError(
                "slot_ids must be in [0, num_slots). "
                f"Got min={int(slot_ids.min().item())}, max={int(slot_ids.max().item())}, "
                f"num_slots={cache.num_slots}."
            )
    cache_k = cache_k.to(device=cache.k_cache.device, dtype=cache.k_cache.dtype)
    cache_v = cache_v.to(device=cache.v_cache.device, dtype=cache.v_cache.dtype)
    return slot_ids, cache_k, cache_v


def _validate_commit_write_inputs(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    slot_ids_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache.validate(config)
    if slot_ids_2d.ndim != 2:
        raise ValueError(
            "slot_ids_2d must be rank-2 for commit write. "
            f"Got {tuple(slot_ids_2d.shape)}."
        )
    bs, block_size = slot_ids_2d.shape
    expected_shape = (
        config.num_layers,
        bs,
        block_size,
        config.num_kv_heads,
        config.head_dim,
    )
    if tuple(cache_k.shape) != expected_shape:
        raise ValueError(
            "cache_k shape mismatch for commit write. "
            f"Expected {expected_shape}, got {tuple(cache_k.shape)}."
        )
    if tuple(cache_v.shape) != expected_shape:
        raise ValueError(
            "cache_v shape mismatch for commit write. "
            f"Expected {expected_shape}, got {tuple(cache_v.shape)}."
        )
    if tuple(commit_lens.shape) != (bs,):
        raise ValueError(
            "commit_lens shape mismatch for commit write. "
            f"Expected {(bs,)}, got {tuple(commit_lens.shape)}."
        )
    slot_ids_2d = slot_ids_2d.to(device=cache.k_cache.device)
    commit_lens = commit_lens.to(device=cache.k_cache.device)
    if slot_ids_2d.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "slot_ids_2d must have dtype int32 or int64 for commit write. "
            f"Got {slot_ids_2d.dtype}."
        )
    if commit_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "commit_lens must have dtype int32 or int64 for commit write. "
            f"Got {commit_lens.dtype}."
        )
    cache_k = cache_k.to(device=cache.k_cache.device, dtype=cache.k_cache.dtype)
    cache_v = cache_v.to(device=cache.v_cache.device, dtype=cache.v_cache.dtype)
    if bs > 0 and block_size > 0:
        if (
            int(slot_ids_2d.min().item()) < 0
            or int(slot_ids_2d.max().item()) >= cache.num_slots
        ):
            raise ValueError(
                "slot_ids_2d must be in [0, num_slots). "
                f"Got min={int(slot_ids_2d.min().item())}, max={int(slot_ids_2d.max().item())}, "
                f"num_slots={cache.num_slots}."
            )
        if (
            int(commit_lens.min().item()) < 0
            or int(commit_lens.max().item()) > block_size
        ):
            raise ValueError(
                "commit_lens must be in [0, block_size]. "
                f"Got min={int(commit_lens.min().item())}, max={int(commit_lens.max().item())}, "
                f"block_size={block_size}."
            )
    return slot_ids_2d, commit_lens, cache_k, cache_v


def write_prompt_reference(
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
    num_tokens = int(slot_ids.numel())
    for layer_idx in range(config.num_layers):
        for tok_idx in range(num_tokens):
            slot = int(slot_ids[tok_idx].item())
            updated.k_cache[layer_idx, slot] = cache_k[layer_idx, tok_idx]
            updated.v_cache[layer_idx, slot] = cache_v[layer_idx, tok_idx]
    return updated


def write_commit_prefix_reference(
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
            for col in range(keep):
                slot = int(slot_ids_2d[row, col].item())
                updated.k_cache[layer_idx, slot] = cache_k[layer_idx, row, col]
                updated.v_cache[layer_idx, slot] = cache_v[layer_idx, row, col]
    return updated
