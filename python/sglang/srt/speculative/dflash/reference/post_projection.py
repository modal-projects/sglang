from __future__ import annotations

import torch

from sglang.srt.speculative.dflash.contracts import (
    DFlashKVCache,
    DFlashMaterializerConfig,
    DFlashMaterializerWeights,
    DFlashProjectedKV,
)
from sglang.srt.speculative.dflash.reference.kv_prefix_write import (
    _validate_commit_write_inputs,
    _validate_prompt_write_inputs,
    write_commit_prefix_reference,
    write_prompt_reference,
)
from sglang.srt.speculative.dflash.reference.materializer import (
    _apply_k_rms_norm,
    _apply_neox_rope,
)


def build_neox_cos_sin_cache(
    *,
    rotary_dim: int,
    rope_theta: float,
    max_position: int,
    device: torch.device | str,
) -> torch.Tensor:
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be positive and even, got {rotary_dim}.")
    if max_position < 0:
        raise ValueError(f"max_position must be >= 0, got {max_position}.")
    device = torch.device(device)
    half = rotary_dim // 2
    positions = torch.arange(max_position + 1, device=device, dtype=torch.float32).view(
        -1, 1
    )
    inv_idx = torch.arange(half, device=device, dtype=torch.float32)
    inv_freq = torch.pow(
        torch.tensor(float(rope_theta), device=device, dtype=torch.float32),
        -(2.0 * inv_idx / float(rotary_dim)),
    ).view(1, half)
    freqs = positions * inv_freq
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1).contiguous()


def _validate_prompt_inputs(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    projected: DFlashProjectedKV,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    config.validate()
    weights.validate(config)
    slot_ids, raw_k, raw_v = _validate_prompt_write_inputs(
        cache=cache,
        config=config,
        slot_ids=slot_ids,
        cache_k=projected.cache_k,
        cache_v=projected.cache_v,
    )
    positions = positions.view(-1).to(device=cache.k_cache.device, dtype=torch.int64)
    if int(positions.numel()) != int(slot_ids.numel()):
        raise ValueError(
            "positions length mismatch for prompt post-projection. "
            f"Expected {int(slot_ids.numel())}, got {int(positions.numel())}."
        )
    if cos_sin_cache.ndim != 2 or int(cos_sin_cache.shape[1]) != config.rotary_dim:
        raise ValueError(
            "cos_sin_cache shape mismatch for prompt post-projection. "
            f"Expected [P, {config.rotary_dim}], got {tuple(cos_sin_cache.shape)}."
        )
    cos_sin_cache = cos_sin_cache.to(device=cache.k_cache.device, dtype=torch.float32)
    if int(positions.numel()) > 0 and int(positions.max().item()) >= int(
        cos_sin_cache.shape[0]
    ):
        raise ValueError(
            "cos_sin_cache too short for prompt post-projection positions. "
            f"max_position={int(positions.max().item())}, cache_len={int(cos_sin_cache.shape[0])}."
        )
    return raw_k, raw_v, positions, slot_ids, cos_sin_cache


def _validate_commit_inputs(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    projected: DFlashProjectedKV,
    positions: torch.Tensor,
    slot_ids_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    config.validate()
    weights.validate(config)
    slot_ids_2d, commit_lens, raw_k, raw_v = _validate_commit_write_inputs(
        cache=cache,
        config=config,
        slot_ids_2d=slot_ids_2d,
        commit_lens=commit_lens,
        cache_k=projected.cache_k,
        cache_v=projected.cache_v,
    )
    if tuple(positions.shape) != tuple(slot_ids_2d.shape):
        raise ValueError(
            "positions shape mismatch for commit post-projection. "
            f"Expected {tuple(slot_ids_2d.shape)}, got {tuple(positions.shape)}."
        )
    positions = positions.to(device=cache.k_cache.device, dtype=torch.int64)
    if cos_sin_cache.ndim != 2 or int(cos_sin_cache.shape[1]) != config.rotary_dim:
        raise ValueError(
            "cos_sin_cache shape mismatch for commit post-projection. "
            f"Expected [P, {config.rotary_dim}], got {tuple(cos_sin_cache.shape)}."
        )
    if int(positions.numel()) > 0 and int(positions.max().item()) >= int(
        cos_sin_cache.shape[0]
    ):
        raise ValueError(
            "cos_sin_cache too short for commit post-projection positions. "
            f"max_position={int(positions.max().item())}, cache_len={int(cos_sin_cache.shape[0])}."
        )
    cos_sin_cache = cos_sin_cache.to(device=cache.k_cache.device, dtype=torch.float32)
    return raw_k, raw_v, positions, slot_ids_2d, commit_lens, cos_sin_cache


def postprocess_prompt_reference(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    projected: DFlashProjectedKV,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    inplace: bool = False,
) -> DFlashKVCache:
    raw_k, raw_v, positions, slot_ids, _ = _validate_prompt_inputs(
        cache=cache,
        config=config,
        weights=weights,
        projected=projected,
        positions=positions,
        slot_ids=slot_ids,
        cos_sin_cache=cos_sin_cache,
    )
    proc_k = torch.empty_like(raw_k)
    for layer_idx in range(config.num_layers):
        k = _apply_k_rms_norm(
            raw_k[layer_idx],
            weights.k_norm_weight[layer_idx],
            weights.k_norm_eps[layer_idx],
        )
        proc_k[layer_idx] = _apply_neox_rope(k, positions, config)
    return write_prompt_reference(
        cache=cache,
        config=config,
        slot_ids=slot_ids,
        cache_k=proc_k,
        cache_v=raw_v,
        inplace=inplace,
    )


def postprocess_commit_reference(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    projected: DFlashProjectedKV,
    positions: torch.Tensor,
    slot_ids_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    inplace: bool = False,
) -> DFlashKVCache:
    raw_k, raw_v, positions, slot_ids_2d, commit_lens, _ = _validate_commit_inputs(
        cache=cache,
        config=config,
        weights=weights,
        projected=projected,
        positions=positions,
        slot_ids_2d=slot_ids_2d,
        commit_lens=commit_lens,
        cos_sin_cache=cos_sin_cache,
    )
    proc_k = torch.empty_like(raw_k)
    for layer_idx in range(config.num_layers):
        k = _apply_k_rms_norm(
            raw_k[layer_idx].reshape(-1, config.num_kv_heads, config.head_dim),
            weights.k_norm_weight[layer_idx],
            weights.k_norm_eps[layer_idx],
        ).reshape_as(raw_k[layer_idx])
        proc_k[layer_idx] = _apply_neox_rope(
            k.reshape(-1, config.num_kv_heads, config.head_dim),
            positions.reshape(-1),
            config,
        ).reshape_as(raw_k[layer_idx])
    return write_commit_prefix_reference(
        cache=cache,
        config=config,
        slot_ids_2d=slot_ids_2d,
        commit_lens=commit_lens,
        cache_k=proc_k,
        cache_v=raw_v,
        inplace=inplace,
    )
