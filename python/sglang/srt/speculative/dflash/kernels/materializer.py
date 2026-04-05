from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.speculative.dflash.contracts import (
    DFlashKVCache,
    DFlashMaterializerConfig,
    DFlashMaterializerWeights,
)
from sglang.srt.speculative.dflash.reference.materializer import (
    _apply_k_rms_norm,
    _apply_neox_rope,
    _validate_commit_inputs,
    _validate_prompt_inputs,
)


def _grouped_project(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    projected = torch.einsum("nh,goh->gno", hidden, weight)
    if bias is not None:
        projected = projected + bias[:, None, :]
    return projected


def _write_layer_kv(
    *,
    cache: DFlashKVCache,
    layer_idx: int,
    slot_ids: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    slot_ids_index = slot_ids.to(dtype=torch.int64)
    cache.k_cache[layer_idx].index_copy_(0, slot_ids_index, k)
    cache.v_cache[layer_idx].index_copy_(0, slot_ids_index, v)


def materialize_prompt_per_layer_control(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    chunk_size: int | None = None,
    inplace: bool = False,
) -> DFlashKVCache:
    hidden, positions, slot_ids = _validate_prompt_inputs(
        cache=cache,
        config=config,
        weights=weights,
        hidden=hidden,
        positions=positions,
        slot_ids=slot_ids,
    )
    updated = cache if inplace else cache.clone()
    num_tokens = int(hidden.shape[0])
    if num_tokens == 0:
        return updated

    chunk = num_tokens if chunk_size is None else int(chunk_size)
    if chunk <= 0:
        raise ValueError(f"chunk_size must be positive when set, got {chunk_size}.")

    for start in range(0, num_tokens, chunk):
        end = min(start + chunk, num_tokens)
        hidden_chunk = hidden[start:end]
        positions_chunk = positions[start:end]
        slot_chunk = slot_ids[start:end]
        for layer_idx in range(config.num_layers):
            kv = F.linear(
                hidden_chunk,
                weights.kv_proj_weight[layer_idx],
                (
                    None
                    if weights.kv_proj_bias is None
                    else weights.kv_proj_bias[layer_idx]
                ),
            )
            k_flat, v_flat = kv.split([config.kv_size, config.kv_size], dim=-1)
            k = k_flat.view(-1, config.num_kv_heads, config.head_dim)
            v = v_flat.view(-1, config.num_kv_heads, config.head_dim)
            k = _apply_k_rms_norm(
                k,
                weights.k_norm_weight[layer_idx],
                weights.k_norm_eps[layer_idx],
            )
            k = _apply_neox_rope(k, positions_chunk, config)
            _write_layer_kv(
                cache=updated,
                layer_idx=layer_idx,
                slot_ids=slot_chunk,
                k=k,
                v=v,
            )
    return updated


def materialize_prompt_grouped_control(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    group_size: int,
    chunk_size: int | None = None,
    inplace: bool = False,
) -> DFlashKVCache:
    hidden, positions, slot_ids = _validate_prompt_inputs(
        cache=cache,
        config=config,
        weights=weights,
        hidden=hidden,
        positions=positions,
        slot_ids=slot_ids,
    )
    updated = cache if inplace else cache.clone()
    num_tokens = int(hidden.shape[0])
    if num_tokens == 0:
        return updated
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")

    chunk = num_tokens if chunk_size is None else int(chunk_size)
    if chunk <= 0:
        raise ValueError(f"chunk_size must be positive when set, got {chunk_size}.")

    for start in range(0, num_tokens, chunk):
        end = min(start + chunk, num_tokens)
        hidden_chunk = hidden[start:end]
        positions_chunk = positions[start:end]
        slot_chunk = slot_ids[start:end]
        for layer_start in range(0, config.num_layers, group_size):
            layer_end = min(layer_start + group_size, config.num_layers)
            proj = _grouped_project(
                hidden_chunk,
                weights.kv_proj_weight[layer_start:layer_end],
                (
                    None
                    if weights.kv_proj_bias is None
                    else weights.kv_proj_bias[layer_start:layer_end]
                ),
            )
            group_count = layer_end - layer_start
            for offset in range(group_count):
                layer_idx = layer_start + offset
                k_flat, v_flat = proj[offset].split(
                    [config.kv_size, config.kv_size], dim=-1
                )
                k = k_flat.view(-1, config.num_kv_heads, config.head_dim)
                v = v_flat.view(-1, config.num_kv_heads, config.head_dim)
                k = _apply_k_rms_norm(
                    k,
                    weights.k_norm_weight[layer_idx],
                    weights.k_norm_eps[layer_idx],
                )
                k = _apply_neox_rope(k, positions_chunk, config)
                _write_layer_kv(
                    cache=updated,
                    layer_idx=layer_idx,
                    slot_ids=slot_chunk,
                    k=k,
                    v=v,
                )
    return updated


def materialize_commit_per_layer_control(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    commit_lens: torch.Tensor,
    inplace: bool = False,
) -> DFlashKVCache:
    verify_hidden, positions, slot_ids, commit_lens = _validate_commit_inputs(
        cache=cache,
        config=config,
        weights=weights,
        verify_hidden=verify_hidden,
        positions=positions,
        slot_ids=slot_ids,
        commit_lens=commit_lens,
    )
    updated = cache if inplace else cache.clone()
    bs, block_size, hidden_size = verify_hidden.shape
    if bs == 0 or int(commit_lens.max().item()) == 0:
        return updated

    flat_hidden = verify_hidden.view(bs * block_size, hidden_size)
    flat_positions = positions.view(bs * block_size)
    flat_slot_ids = slot_ids.view(bs * block_size)
    offsets = torch.arange(block_size, device=verify_hidden.device, dtype=torch.int32)[
        None, :
    ]
    valid_mask = offsets < commit_lens[:, None]
    valid_idx = torch.nonzero(valid_mask.view(-1), as_tuple=False).flatten()
    valid_slot_ids = flat_slot_ids.index_select(0, valid_idx)

    for layer_idx in range(config.num_layers):
        kv = F.linear(
            flat_hidden,
            weights.kv_proj_weight[layer_idx],
            None if weights.kv_proj_bias is None else weights.kv_proj_bias[layer_idx],
        )
        k_flat, v_flat = kv.split([config.kv_size, config.kv_size], dim=-1)
        k = k_flat.view(-1, config.num_kv_heads, config.head_dim)
        v = v_flat.view(-1, config.num_kv_heads, config.head_dim)
        k = _apply_k_rms_norm(
            k,
            weights.k_norm_weight[layer_idx],
            weights.k_norm_eps[layer_idx],
        )
        k = _apply_neox_rope(k, flat_positions, config)
        _write_layer_kv(
            cache=updated,
            layer_idx=layer_idx,
            slot_ids=valid_slot_ids,
            k=k.index_select(0, valid_idx),
            v=v.index_select(0, valid_idx),
        )
    return updated


def materialize_commit_grouped_control(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    commit_lens: torch.Tensor,
    group_size: int,
    inplace: bool = False,
) -> DFlashKVCache:
    verify_hidden, positions, slot_ids, commit_lens = _validate_commit_inputs(
        cache=cache,
        config=config,
        weights=weights,
        verify_hidden=verify_hidden,
        positions=positions,
        slot_ids=slot_ids,
        commit_lens=commit_lens,
    )
    updated = cache if inplace else cache.clone()
    bs, block_size, hidden_size = verify_hidden.shape
    if bs == 0 or int(commit_lens.max().item()) == 0:
        return updated
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")

    flat_hidden = verify_hidden.view(bs * block_size, hidden_size)
    flat_positions = positions.view(bs * block_size)
    flat_slot_ids = slot_ids.view(bs * block_size)
    offsets = torch.arange(block_size, device=verify_hidden.device, dtype=torch.int32)[
        None, :
    ]
    valid_mask = offsets < commit_lens[:, None]
    valid_idx = torch.nonzero(valid_mask.view(-1), as_tuple=False).flatten()
    valid_slot_ids = flat_slot_ids.index_select(0, valid_idx)

    for layer_start in range(0, config.num_layers, group_size):
        layer_end = min(layer_start + group_size, config.num_layers)
        proj = _grouped_project(
            flat_hidden,
            weights.kv_proj_weight[layer_start:layer_end],
            (
                None
                if weights.kv_proj_bias is None
                else weights.kv_proj_bias[layer_start:layer_end]
            ),
        )
        group_count = layer_end - layer_start
        for offset in range(group_count):
            layer_idx = layer_start + offset
            k_flat, v_flat = proj[offset].split(
                [config.kv_size, config.kv_size], dim=-1
            )
            k = k_flat.view(-1, config.num_kv_heads, config.head_dim)
            v = v_flat.view(-1, config.num_kv_heads, config.head_dim)
            k = _apply_k_rms_norm(
                k,
                weights.k_norm_weight[layer_idx],
                weights.k_norm_eps[layer_idx],
            )
            k = _apply_neox_rope(k, flat_positions, config)
            _write_layer_kv(
                cache=updated,
                layer_idx=layer_idx,
                slot_ids=valid_slot_ids,
                k=k.index_select(0, valid_idx),
                v=v.index_select(0, valid_idx),
            )
    return updated
