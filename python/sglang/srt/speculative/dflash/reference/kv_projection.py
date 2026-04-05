from __future__ import annotations

import torch

from sglang.srt.speculative.dflash.contracts import (
    DFlashMaterializerConfig,
    DFlashMaterializerWeights,
    DFlashProjectedKV,
)
from sglang.srt.speculative.dflash.reference.materializer import (
    _project_kv_per_layer,
)


def _validate_prompt_projection_inputs(
    *,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    hidden: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    config.validate()
    weights.validate(config)
    if hidden.ndim != 2 or int(hidden.shape[1]) != config.hidden_size:
        raise ValueError(
            "hidden must have shape [N, hidden_size]. "
            f"Expected hidden_size={config.hidden_size}, got {tuple(hidden.shape)}."
        )
    positions = positions.view(-1).to(device=hidden.device, dtype=torch.int64)
    if int(positions.numel()) != int(hidden.shape[0]):
        raise ValueError(
            "positions length mismatch for prompt projection. "
            f"Expected {int(hidden.shape[0])}, got {int(positions.numel())}."
        )
    return hidden, positions


def _validate_commit_projection_inputs(
    *,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    config.validate()
    weights.validate(config)
    if verify_hidden.ndim != 3 or int(verify_hidden.shape[-1]) != config.hidden_size:
        raise ValueError(
            "verify_hidden must have shape [bs, B, hidden_size]. "
            f"Expected hidden_size={config.hidden_size}, got {tuple(verify_hidden.shape)}."
        )
    bs, block_size, _ = verify_hidden.shape
    if tuple(positions.shape) != (bs, block_size):
        raise ValueError(
            "positions shape mismatch for commit projection. "
            f"Expected {(bs, block_size)}, got {tuple(positions.shape)}."
        )
    positions = positions.to(device=verify_hidden.device, dtype=torch.int64)
    return verify_hidden, positions


def project_prompt_reference(
    *,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    hidden: torch.Tensor,
    positions: torch.Tensor,
) -> DFlashProjectedKV:
    hidden, positions = _validate_prompt_projection_inputs(
        config=config,
        weights=weights,
        hidden=hidden,
        positions=positions,
    )
    num_tokens = int(hidden.shape[0])
    cache_k = torch.empty(
        (config.num_layers, num_tokens, config.num_kv_heads, config.head_dim),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    cache_v = torch.empty_like(cache_k)
    for layer_idx in range(config.num_layers):
        k, v = _project_kv_per_layer(
            hidden=hidden,
            positions=positions,
            config=config,
            weights=weights,
            layer_idx=layer_idx,
        )
        cache_k[layer_idx] = k
        cache_v[layer_idx] = v
    projected = DFlashProjectedKV(cache_k=cache_k, cache_v=cache_v)
    projected.validate(config, prefix_shape=(config.num_layers, num_tokens))
    return projected


def project_commit_reference(
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
    flattened = project_prompt_reference(
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
