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


def _project_group(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    config: DFlashMaterializerConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_size = int(weight.shape[0])
    flat_weight = weight.reshape(group_size * 2 * config.kv_size, config.hidden_size)
    flat_bias = None if bias is None else bias.reshape(group_size * 2 * config.kv_size)
    kv = F.linear(hidden, flat_weight, flat_bias)
    kv = kv.view(
        int(hidden.shape[0]),
        group_size,
        2,
        config.num_kv_heads,
        config.head_dim,
    ).permute(1, 0, 2, 3, 4)
    return kv[:, :, 0], kv[:, :, 1]


def project_raw_prompt_reference(
    *,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    group_size: int = 1,
    chunk_size: int | None = None,
) -> DFlashProjectedKV:
    hidden, _ = _validate_prompt_projection_inputs(
        config=config,
        weights=weights,
        hidden=hidden,
        positions=positions,
    )
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    num_tokens = int(hidden.shape[0])
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
        for layer_start in range(0, config.num_layers, group_size):
            layer_end = min(layer_start + group_size, config.num_layers)
            group_k, group_v = _project_group(
                hidden_chunk,
                weights.kv_proj_weight[layer_start:layer_end],
                (
                    None
                    if weights.kv_proj_bias is None
                    else weights.kv_proj_bias[layer_start:layer_end]
                ),
                config=config,
            )
            cache_k[layer_start:layer_end, start:end] = group_k
            cache_v[layer_start:layer_end, start:end] = group_v

    projected = DFlashProjectedKV(cache_k=cache_k, cache_v=cache_v)
    projected.validate(config, prefix_shape=(config.num_layers, num_tokens))
    return projected


def project_raw_commit_reference(
    *,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    group_size: int = 1,
) -> DFlashProjectedKV:
    verify_hidden, _ = _validate_commit_projection_inputs(
        config=config,
        weights=weights,
        verify_hidden=verify_hidden,
        positions=positions,
    )
    bs, block_size, hidden_size = verify_hidden.shape
    flat = project_raw_prompt_reference(
        config=config,
        weights=weights,
        hidden=verify_hidden.reshape(bs * block_size, hidden_size),
        positions=positions.reshape(bs * block_size),
        group_size=group_size,
    )
    projected = DFlashProjectedKV(
        cache_k=flat.cache_k.reshape(
            config.num_layers,
            bs,
            block_size,
            config.num_kv_heads,
            config.head_dim,
        ),
        cache_v=flat.cache_v.reshape(
            config.num_layers,
            bs,
            block_size,
            config.num_kv_heads,
            config.head_dim,
        ),
    )
    projected.validate(config, prefix_shape=(config.num_layers, bs, block_size))
    return projected
