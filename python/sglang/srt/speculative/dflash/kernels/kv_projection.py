from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.speculative.dflash.contracts import (
    DFlashMaterializerConfig,
    DFlashMaterializerWeights,
    DFlashProjectedKV,
)
from sglang.srt.speculative.dflash.reference.kv_projection import (
    _validate_commit_projection_inputs,
    _validate_prompt_projection_inputs,
)
from sglang.srt.speculative.dflash.reference.materializer import (
    _apply_k_rms_norm,
    _apply_neox_rope,
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


def project_prompt_per_layer_control(
    *,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    chunk_size: int | None = None,
) -> DFlashProjectedKV:
    hidden, positions = _validate_prompt_projection_inputs(
        config=config,
        weights=weights,
        hidden=hidden,
        positions=positions,
    )
    num_tokens = int(hidden.shape[0])
    if num_tokens == 0:
        projected = DFlashProjectedKV(
            cache_k=torch.empty(
                (config.num_layers, 0, config.num_kv_heads, config.head_dim),
                dtype=hidden.dtype,
                device=hidden.device,
            ),
            cache_v=torch.empty(
                (config.num_layers, 0, config.num_kv_heads, config.head_dim),
                dtype=hidden.dtype,
                device=hidden.device,
            ),
        )
        projected.validate(config, prefix_shape=(config.num_layers, 0))
        return projected
    chunk = num_tokens if chunk_size is None else int(chunk_size)
    if chunk <= 0:
        raise ValueError(f"chunk_size must be positive when set, got {chunk_size}.")

    cache_k = torch.empty(
        (config.num_layers, num_tokens, config.num_kv_heads, config.head_dim),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    cache_v = torch.empty_like(cache_k)
    for start in range(0, num_tokens, chunk):
        end = min(start + chunk, num_tokens)
        hidden_chunk = hidden[start:end]
        positions_chunk = positions[start:end]
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
            cache_k[layer_idx, start:end] = k
            cache_v[layer_idx, start:end] = v

    projected = DFlashProjectedKV(cache_k=cache_k, cache_v=cache_v)
    projected.validate(config, prefix_shape=(config.num_layers, num_tokens))
    return projected


def project_prompt_grouped_control(
    *,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    group_size: int,
    chunk_size: int | None = None,
) -> DFlashProjectedKV:
    hidden, positions = _validate_prompt_projection_inputs(
        config=config,
        weights=weights,
        hidden=hidden,
        positions=positions,
    )
    num_tokens = int(hidden.shape[0])
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    if num_tokens == 0:
        projected = DFlashProjectedKV(
            cache_k=torch.empty(
                (config.num_layers, 0, config.num_kv_heads, config.head_dim),
                dtype=hidden.dtype,
                device=hidden.device,
            ),
            cache_v=torch.empty(
                (config.num_layers, 0, config.num_kv_heads, config.head_dim),
                dtype=hidden.dtype,
                device=hidden.device,
            ),
        )
        projected.validate(config, prefix_shape=(config.num_layers, 0))
        return projected

    chunk = num_tokens if chunk_size is None else int(chunk_size)
    if chunk <= 0:
        raise ValueError(f"chunk_size must be positive when set, got {chunk_size}.")

    cache_k = torch.empty(
        (config.num_layers, num_tokens, config.num_kv_heads, config.head_dim),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    cache_v = torch.empty_like(cache_k)
    for start in range(0, num_tokens, chunk):
        end = min(start + chunk, num_tokens)
        hidden_chunk = hidden[start:end]
        positions_chunk = positions[start:end]
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
                cache_k[layer_idx, start:end] = k
                cache_v[layer_idx, start:end] = v

    projected = DFlashProjectedKV(cache_k=cache_k, cache_v=cache_v)
    projected.validate(config, prefix_shape=(config.num_layers, num_tokens))
    return projected


def project_commit_per_layer_control(
    *,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
) -> DFlashProjectedKV:
    verify_hidden, positions = _validate_commit_projection_inputs(
        config=config,
        weights=weights,
        verify_hidden=verify_hidden,
        positions=positions,
    )
    bs, block_size, hidden_size = verify_hidden.shape
    flattened = project_prompt_per_layer_control(
        config=config,
        weights=weights,
        hidden=verify_hidden.reshape(bs * block_size, hidden_size),
        positions=positions.reshape(bs * block_size),
    )
    projected = DFlashProjectedKV(
        cache_k=flattened.cache_k.reshape(
            config.num_layers,
            bs,
            block_size,
            config.num_kv_heads,
            config.head_dim,
        ),
        cache_v=flattened.cache_v.reshape(
            config.num_layers,
            bs,
            block_size,
            config.num_kv_heads,
            config.head_dim,
        ),
    )
    projected.validate(config, prefix_shape=(config.num_layers, bs, block_size))
    return projected


def project_commit_grouped_control(
    *,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    group_size: int,
) -> DFlashProjectedKV:
    verify_hidden, positions = _validate_commit_projection_inputs(
        config=config,
        weights=weights,
        verify_hidden=verify_hidden,
        positions=positions,
    )
    bs, block_size, hidden_size = verify_hidden.shape
    flattened = project_prompt_grouped_control(
        config=config,
        weights=weights,
        hidden=verify_hidden.reshape(bs * block_size, hidden_size),
        positions=positions.reshape(bs * block_size),
        group_size=group_size,
    )
    projected = DFlashProjectedKV(
        cache_k=flattened.cache_k.reshape(
            config.num_layers,
            bs,
            block_size,
            config.num_kv_heads,
            config.head_dim,
        ),
        cache_v=flattened.cache_v.reshape(
            config.num_layers,
            bs,
            block_size,
            config.num_kv_heads,
            config.head_dim,
        ),
    )
    projected.validate(config, prefix_shape=(config.num_layers, bs, block_size))
    return projected
