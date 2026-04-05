from __future__ import annotations

import torch

from sglang.srt.speculative.dflash.contracts import (
    DFlashKVCache,
    DFlashMaterializerConfig,
    DFlashMaterializerWeights,
)
from sglang.srt.speculative.dflash.reference.materializer import (
    _apply_k_rms_norm,
    _apply_neox_rope,
)


def _validate_prompt_packed_inputs(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    packed_kv: torch.Tensor,
    layer_start: int,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor]:
    config.validate()
    weights.validate(config)
    cache.validate(config)
    if packed_kv.ndim != 5:
        raise ValueError(
            "packed_kv must have shape [N, G, 2, num_kv_heads, head_dim]. "
            f"Got {tuple(packed_kv.shape)}."
        )
    num_tokens, group_count, pair_dim, num_kv_heads, head_dim = packed_kv.shape
    if pair_dim != 2:
        raise ValueError(f"packed_kv pair dimension must be 2, got {pair_dim}.")
    if num_kv_heads != config.num_kv_heads or head_dim != config.head_dim:
        raise ValueError(
            "packed_kv head shape mismatch. "
            f"Expected (*, *, 2, {config.num_kv_heads}, {config.head_dim}), got {tuple(packed_kv.shape)}."
        )
    if layer_start < 0 or layer_start + group_count > config.num_layers:
        raise ValueError(
            "packed_kv layer range is out of bounds. "
            f"layer_start={layer_start}, group_count={group_count}, num_layers={config.num_layers}."
        )
    positions = positions.view(-1).to(device=cache.k_cache.device)
    slot_ids = slot_ids.view(-1).to(device=cache.k_cache.device)
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "positions must have dtype int32 or int64 for packed prompt post-projection. "
            f"Got {positions.dtype}."
        )
    if slot_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "slot_ids must have dtype int32 or int64 for packed prompt post-projection. "
            f"Got {slot_ids.dtype}."
        )
    if int(positions.numel()) != num_tokens or int(slot_ids.numel()) != num_tokens:
        raise ValueError(
            "positions and slot_ids must have one entry per packed token. "
            f"Got positions={int(positions.numel())}, slot_ids={int(slot_ids.numel())}, tokens={num_tokens}."
        )
    if num_tokens > 0:
        if (
            int(slot_ids.min().item()) < 0
            or int(slot_ids.max().item()) >= cache.num_slots
        ):
            raise ValueError(
                "slot_ids must be in [0, num_slots). "
                f"Got min={int(slot_ids.min().item())}, max={int(slot_ids.max().item())}, num_slots={cache.num_slots}."
            )
    if cos_sin_cache.ndim != 2 or int(cos_sin_cache.shape[1]) != config.rotary_dim:
        raise ValueError(
            "cos_sin_cache shape mismatch for packed prompt post-projection. "
            f"Expected [P, {config.rotary_dim}], got {tuple(cos_sin_cache.shape)}."
        )
    if num_tokens > 0 and int(positions.max().item()) >= int(cos_sin_cache.shape[0]):
        raise ValueError(
            "cos_sin_cache too short for packed prompt positions. "
            f"max_position={int(positions.max().item())}, cache_len={int(cos_sin_cache.shape[0])}."
        )
    return (
        packed_kv,
        int(group_count),
        positions,
        slot_ids,
        cos_sin_cache.to(device=cache.k_cache.device, dtype=torch.float32),
    )


def _validate_commit_packed_inputs(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    packed_kv: torch.Tensor,
    layer_start: int,
    positions: torch.Tensor,
    slot_ids_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    config.validate()
    weights.validate(config)
    cache.validate(config)
    if packed_kv.ndim != 6:
        raise ValueError(
            "packed_kv must have shape [bs, B, G, 2, num_kv_heads, head_dim]. "
            f"Got {tuple(packed_kv.shape)}."
        )
    bs, block_size, group_count, pair_dim, num_kv_heads, head_dim = packed_kv.shape
    if pair_dim != 2:
        raise ValueError(f"packed_kv pair dimension must be 2, got {pair_dim}.")
    if num_kv_heads != config.num_kv_heads or head_dim != config.head_dim:
        raise ValueError(
            "packed_kv head shape mismatch. "
            f"Expected (*, *, *, 2, {config.num_kv_heads}, {config.head_dim}), got {tuple(packed_kv.shape)}."
        )
    if layer_start < 0 or layer_start + group_count > config.num_layers:
        raise ValueError(
            "packed_kv layer range is out of bounds. "
            f"layer_start={layer_start}, group_count={group_count}, num_layers={config.num_layers}."
        )
    if tuple(positions.shape) != (bs, block_size):
        raise ValueError(
            "positions shape mismatch for packed commit post-projection. "
            f"Expected {(bs, block_size)}, got {tuple(positions.shape)}."
        )
    if tuple(slot_ids_2d.shape) != (bs, block_size):
        raise ValueError(
            "slot_ids_2d shape mismatch for packed commit post-projection. "
            f"Expected {(bs, block_size)}, got {tuple(slot_ids_2d.shape)}."
        )
    if tuple(commit_lens.shape) != (bs,):
        raise ValueError(
            "commit_lens shape mismatch for packed commit post-projection. "
            f"Expected {(bs,)}, got {tuple(commit_lens.shape)}."
        )
    positions = positions.to(device=cache.k_cache.device)
    slot_ids_2d = slot_ids_2d.to(device=cache.k_cache.device)
    commit_lens = commit_lens.to(device=cache.k_cache.device)
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "positions must have dtype int32 or int64 for packed commit post-projection. "
            f"Got {positions.dtype}."
        )
    if slot_ids_2d.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "slot_ids_2d must have dtype int32 or int64 for packed commit post-projection. "
            f"Got {slot_ids_2d.dtype}."
        )
    if commit_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "commit_lens must have dtype int32 or int64 for packed commit post-projection. "
            f"Got {commit_lens.dtype}."
        )
    if int(commit_lens.numel()) > 0:
        if (
            int(commit_lens.min().item()) < 0
            or int(commit_lens.max().item()) > block_size
        ):
            raise ValueError(
                "commit_lens must be in [0, block_size]. "
                f"Got min={int(commit_lens.min().item())}, max={int(commit_lens.max().item())}, block_size={block_size}."
            )
    if cos_sin_cache.ndim != 2 or int(cos_sin_cache.shape[1]) != config.rotary_dim:
        raise ValueError(
            "cos_sin_cache shape mismatch for packed commit post-projection. "
            f"Expected [P, {config.rotary_dim}], got {tuple(cos_sin_cache.shape)}."
        )
    if (
        bs > 0
        and block_size > 0
        and int(positions.max().item()) >= int(cos_sin_cache.shape[0])
    ):
        raise ValueError(
            "cos_sin_cache too short for packed commit positions. "
            f"max_position={int(positions.max().item())}, cache_len={int(cos_sin_cache.shape[0])}."
        )
    return (
        packed_kv,
        int(group_count),
        positions,
        slot_ids_2d,
        commit_lens,
        cos_sin_cache.to(device=cache.k_cache.device, dtype=torch.float32),
    )


def postprocess_prompt_packed_reference(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    packed_kv: torch.Tensor,
    layer_start: int,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    inplace: bool = False,
) -> DFlashKVCache:
    packed_kv, group_count, positions, slot_ids, _ = _validate_prompt_packed_inputs(
        cache=cache,
        config=config,
        weights=weights,
        packed_kv=packed_kv,
        layer_start=layer_start,
        positions=positions,
        slot_ids=slot_ids,
        cos_sin_cache=cos_sin_cache,
    )
    updated = cache if inplace else cache.clone()
    if int(slot_ids.numel()) == 0:
        return updated
    slot_ids_index = slot_ids.to(dtype=torch.int64)
    for group_idx in range(group_count):
        layer_idx = layer_start + group_idx
        k = packed_kv[:, group_idx, 0]
        v = packed_kv[:, group_idx, 1]
        k = _apply_k_rms_norm(
            k,
            weights.k_norm_weight[layer_idx],
            weights.k_norm_eps[layer_idx],
        )
        k = _apply_neox_rope(k, positions, config)
        updated.k_cache[layer_idx].index_copy_(0, slot_ids_index, k)
        updated.v_cache[layer_idx].index_copy_(0, slot_ids_index, v)
    return updated


def postprocess_commit_packed_reference(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    packed_kv: torch.Tensor,
    layer_start: int,
    positions: torch.Tensor,
    slot_ids_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    inplace: bool = False,
) -> DFlashKVCache:
    packed_kv, group_count, positions, slot_ids_2d, commit_lens, _ = (
        _validate_commit_packed_inputs(
            cache=cache,
            config=config,
            weights=weights,
            packed_kv=packed_kv,
            layer_start=layer_start,
            positions=positions,
            slot_ids_2d=slot_ids_2d,
            commit_lens=commit_lens,
            cos_sin_cache=cos_sin_cache,
        )
    )
    updated = cache if inplace else cache.clone()
    if int(commit_lens.max().item()) == 0:
        return updated
    slot_ids_index = slot_ids_2d.to(dtype=torch.int64)
    for group_idx in range(group_count):
        layer_idx = layer_start + group_idx
        k = packed_kv[:, :, group_idx, 0]
        v = packed_kv[:, :, group_idx, 1]
        k = _apply_k_rms_norm(
            k.reshape(-1, config.num_kv_heads, config.head_dim),
            weights.k_norm_weight[layer_idx],
            weights.k_norm_eps[layer_idx],
        ).reshape_as(k)
        k = _apply_neox_rope(
            k.reshape(-1, config.num_kv_heads, config.head_dim),
            positions.reshape(-1),
            config,
        ).reshape_as(k)
        for row in range(int(commit_lens.shape[0])):
            keep = int(commit_lens[row].item())
            if keep <= 0:
                continue
            updated.k_cache[layer_idx].index_copy_(
                0, slot_ids_index[row, :keep], k[row, :keep]
            )
            updated.v_cache[layer_idx].index_copy_(
                0, slot_ids_index[row, :keep], v[row, :keep]
            )
    return updated
