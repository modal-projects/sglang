from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.speculative.dflash.contracts import (
    DFlashKVCache,
    DFlashMaterializerConfig,
    DFlashMaterializerWeights,
)


def _validate_prompt_inputs(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    config.validate()
    weights.validate(config)
    cache.validate(config)
    if hidden.ndim != 2 or int(hidden.shape[1]) != config.hidden_size:
        raise ValueError(
            "hidden must have shape [N, hidden_size]. "
            f"Expected hidden_size={config.hidden_size}, got {tuple(hidden.shape)}."
        )
    num_tokens = int(hidden.shape[0])
    positions = positions.view(-1).to(device=hidden.device)
    slot_ids = slot_ids.view(-1).to(device=hidden.device)
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "positions must have dtype int32 or int64. " f"Got {positions.dtype}."
        )
    if slot_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "slot_ids must have dtype int32 or int64. " f"Got {slot_ids.dtype}."
        )
    if int(positions.numel()) != num_tokens:
        raise ValueError(
            "positions length mismatch for prompt materialization. "
            f"Expected {num_tokens}, got {int(positions.numel())}."
        )
    if int(slot_ids.numel()) != num_tokens:
        raise ValueError(
            "slot_ids length mismatch for prompt materialization. "
            f"Expected {num_tokens}, got {int(slot_ids.numel())}."
        )
    if num_tokens > 0:
        if (
            int(slot_ids.min().item()) < 0
            or int(slot_ids.max().item()) >= cache.num_slots
        ):
            raise ValueError(
                "slot_ids must be in [0, num_slots). "
                f"Got min={int(slot_ids.min().item())}, max={int(slot_ids.max().item())}, "
                f"num_slots={cache.num_slots}."
            )
    return hidden, positions, slot_ids


def _validate_commit_inputs(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    commit_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    config.validate()
    weights.validate(config)
    cache.validate(config)
    if verify_hidden.ndim != 3 or int(verify_hidden.shape[-1]) != config.hidden_size:
        raise ValueError(
            "verify_hidden must have shape [bs, B, hidden_size]. "
            f"Expected hidden_size={config.hidden_size}, got {tuple(verify_hidden.shape)}."
        )
    bs, block_size, _ = verify_hidden.shape
    if tuple(positions.shape) != (bs, block_size):
        raise ValueError(
            "positions shape mismatch for commit materialization. "
            f"Expected {(bs, block_size)}, got {tuple(positions.shape)}."
        )
    if tuple(slot_ids.shape) != (bs, block_size):
        raise ValueError(
            "slot_ids shape mismatch for commit materialization. "
            f"Expected {(bs, block_size)}, got {tuple(slot_ids.shape)}."
        )
    if tuple(commit_lens.shape) != (bs,):
        raise ValueError(
            "commit_lens shape mismatch for commit materialization. "
            f"Expected {(bs,)}, got {tuple(commit_lens.shape)}."
        )
    positions = positions.to(device=verify_hidden.device)
    slot_ids = slot_ids.to(device=verify_hidden.device)
    commit_lens = commit_lens.to(device=verify_hidden.device)
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "positions must have dtype int32 or int64. " f"Got {positions.dtype}."
        )
    if slot_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "slot_ids must have dtype int32 or int64. " f"Got {slot_ids.dtype}."
        )
    if commit_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "commit_lens must have dtype int32 or int64. " f"Got {commit_lens.dtype}."
        )
    if int(commit_lens.numel()) > 0:
        if (
            int(commit_lens.min().item()) < 0
            or int(commit_lens.max().item()) > block_size
        ):
            raise ValueError(
                "commit_lens must be in [0, block_size]. "
                f"Got min={int(commit_lens.min().item())}, max={int(commit_lens.max().item())}, "
                f"block_size={block_size}."
            )
    return verify_hidden, positions, slot_ids, commit_lens


def _apply_k_rms_norm(
    k: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: torch.Tensor | float,
) -> torch.Tensor:
    eps_value = float(eps.item()) if isinstance(eps, torch.Tensor) else float(eps)
    norm_weight = norm_weight.to(device=k.device, dtype=k.dtype).view(1, 1, -1)
    inv_rms = torch.rsqrt(
        k.to(torch.float32).pow(2).mean(dim=-1, keepdim=True) + eps_value
    )
    return (k.to(torch.float32) * inv_rms * norm_weight.to(torch.float32)).to(
        dtype=k.dtype
    )


def _apply_neox_rope(
    k: torch.Tensor,
    positions: torch.Tensor,
    config: DFlashMaterializerConfig,
) -> torch.Tensor:
    rotary_dim = int(config.rotary_dim)
    half = rotary_dim // 2
    pos = positions.to(device=k.device, dtype=torch.float32).view(-1, 1, 1)
    inv_idx = torch.arange(half, device=k.device, dtype=torch.float32)
    inv_freq = torch.pow(
        torch.tensor(float(config.rope_theta), device=k.device, dtype=torch.float32),
        -(2.0 * inv_idx / float(rotary_dim)),
    ).view(1, 1, half)
    freqs = pos * inv_freq
    cos = torch.cos(freqs).to(dtype=k.dtype)
    sin = torch.sin(freqs).to(dtype=k.dtype)

    k_out = k.clone()
    first = k[..., :half]
    second = k[..., half:rotary_dim]
    k_out[..., :half] = first * cos - second * sin
    k_out[..., half:rotary_dim] = second * cos + first * sin
    return k_out


def _project_kv_per_layer(
    *,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv = F.linear(
        hidden,
        weights.kv_proj_weight[layer_idx],
        None if weights.kv_proj_bias is None else weights.kv_proj_bias[layer_idx],
    )
    k_flat, v_flat = kv.split([config.kv_size, config.kv_size], dim=-1)
    k = k_flat.view(-1, config.num_kv_heads, config.head_dim)
    v = v_flat.view(-1, config.num_kv_heads, config.head_dim)
    k = _apply_k_rms_norm(
        k, weights.k_norm_weight[layer_idx], weights.k_norm_eps[layer_idx]
    )
    k = _apply_neox_rope(k, positions, config)
    return k, v


def materialize_prompt_reference(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
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
    if int(hidden.shape[0]) == 0:
        return updated
    slot_ids_index = slot_ids.to(dtype=torch.int64)

    for layer_idx in range(config.num_layers):
        k, v = _project_kv_per_layer(
            hidden=hidden,
            positions=positions,
            config=config,
            weights=weights,
            layer_idx=layer_idx,
        )
        updated.k_cache[layer_idx].index_copy_(0, slot_ids_index, k)
        updated.v_cache[layer_idx].index_copy_(0, slot_ids_index, v)
    return updated


def materialize_commit_reference(
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
    bs, block_size, hidden_size = verify_hidden.shape
    if bs == 0 or int(commit_lens.max().item()) == 0:
        return cache if inplace else cache.clone()

    valid_hidden = []
    valid_positions = []
    valid_slot_ids = []
    for row in range(bs):
        keep = int(commit_lens[row].item())
        if keep <= 0:
            continue
        valid_hidden.append(verify_hidden[row, :keep, :])
        valid_positions.append(positions[row, :keep])
        valid_slot_ids.append(slot_ids[row, :keep])

    if not valid_hidden:
        return cache if inplace else cache.clone()

    hidden_flat = torch.cat(valid_hidden, dim=0).view(-1, hidden_size)
    positions_flat = torch.cat(valid_positions, dim=0)
    slot_ids_flat = torch.cat(valid_slot_ids, dim=0)
    return materialize_prompt_reference(
        cache=cache,
        config=config,
        weights=weights,
        hidden=hidden_flat,
        positions=positions_flat,
        slot_ids=slot_ids_flat,
        inplace=inplace,
    )
