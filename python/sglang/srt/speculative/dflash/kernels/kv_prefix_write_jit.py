from __future__ import annotations

import torch

from sglang.jit_kernel.dflash_kv_prefix_write import (
    commit_kv_prefix_write,
    prompt_kv_prefix_write,
)
from sglang.srt.speculative.dflash.contracts import (
    DFlashKVCache,
    DFlashMaterializerConfig,
)
from sglang.srt.speculative.dflash.reference.kv_prefix_write import (
    _validate_commit_write_inputs,
    _validate_prompt_write_inputs,
)


def write_prompt_jit(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    slot_ids: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    num_split: int = 0,
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
    if num_tokens == 0:
        return updated

    feature_dim = config.num_kv_heads * config.head_dim
    src_k = cache_k.reshape(config.num_layers, num_tokens, feature_dim).contiguous()
    src_v = cache_v.reshape(config.num_layers, num_tokens, feature_dim).contiguous()
    dst_k = updated.k_cache.reshape(config.num_layers, updated.num_slots, feature_dim)
    dst_v = updated.v_cache.reshape(config.num_layers, updated.num_slots, feature_dim)
    prompt_kv_prefix_write(
        src_k,
        src_v,
        dst_k,
        dst_v,
        slot_ids.contiguous(),
        num_split=num_split,
    )
    return updated


def write_commit_prefix_jit(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    slot_ids_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    num_split: int = 0,
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
    batch_size, block_size_tokens = slot_ids_2d.shape
    if batch_size == 0 or int(commit_lens.max().item()) == 0:
        return updated

    feature_dim = config.num_kv_heads * config.head_dim
    src_k = cache_k.reshape(
        config.num_layers, batch_size, block_size_tokens, feature_dim
    ).contiguous()
    src_v = cache_v.reshape(
        config.num_layers, batch_size, block_size_tokens, feature_dim
    ).contiguous()
    dst_k = updated.k_cache.reshape(config.num_layers, updated.num_slots, feature_dim)
    dst_v = updated.v_cache.reshape(config.num_layers, updated.num_slots, feature_dim)
    commit_kv_prefix_write(
        src_k,
        src_v,
        dst_k,
        dst_v,
        slot_ids_2d.contiguous(),
        commit_lens.contiguous(),
        num_split=num_split,
    )
    return updated
