from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dflash.contracts import (
    DFlashKVCache,
    DFlashMaterializerConfig,
    DFlashMaterializerWeights,
    DFlashProjectedKV,
)
from sglang.srt.speculative.dflash.reference.post_projection import (
    _validate_commit_inputs,
    _validate_prompt_inputs,
)


def _pick_num_warps(head_dim: int) -> int:
    if head_dim <= 64:
        return 2
    if head_dim <= 128:
        return 4
    return 8


@triton.jit
def _prompt_postproj_kernel(
    raw_k_ptr,
    raw_v_ptr,
    dst_k_ptr,
    dst_v_ptr,
    slot_ids_ptr,
    positions_ptr,
    norm_weight_ptr,
    eps_ptr,
    cos_sin_cache_ptr,
    raw_k_layer_stride,
    raw_k_token_stride,
    raw_k_head_stride,
    dst_k_layer_stride,
    dst_k_slot_stride,
    dst_k_head_stride,
    slot_stride,
    pos_stride,
    norm_weight_layer_stride,
    cos_sin_stride,
    num_tokens,
    num_heads,
    head_dim,
    rotary_dim,
    half_rotary_dim,
    BLOCK_HD: tl.constexpr,
):
    row_idx = tl.program_id(0)
    heads_per_layer = num_tokens * num_heads
    layer_idx = row_idx // heads_per_layer
    row_rem = row_idx % heads_per_layer
    token_idx = row_rem // num_heads
    head_idx = row_rem % num_heads

    slot_idx = tl.load(slot_ids_ptr + token_idx * slot_stride)
    position = tl.load(positions_ptr + token_idx * pos_stride)
    eps = tl.load(eps_ptr + layer_idx).to(tl.float32)

    offs = tl.arange(0, BLOCK_HD)
    mask_hd = offs < head_dim
    raw_k = tl.load(
        raw_k_ptr
        + layer_idx * raw_k_layer_stride
        + token_idx * raw_k_token_stride
        + head_idx * raw_k_head_stride
        + offs,
        mask=mask_hd,
        other=0.0,
    ).to(tl.float32)
    raw_v = tl.load(
        raw_v_ptr
        + layer_idx * raw_k_layer_stride
        + token_idx * raw_k_token_stride
        + head_idx * raw_k_head_stride
        + offs,
        mask=mask_hd,
        other=0.0,
    )
    norm_weight = tl.load(
        norm_weight_ptr + layer_idx * norm_weight_layer_stride + offs,
        mask=mask_hd,
        other=1.0,
    ).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(raw_k * raw_k) / head_dim + eps)
    k_normed = raw_k * inv_rms * norm_weight

    is_first_half = offs < half_rotary_dim
    is_rotary = offs < rotary_dim
    pair_idx = tl.where(is_first_half, offs + half_rotary_dim, offs - half_rotary_dim)
    pair_mask = is_rotary & (pair_idx < head_dim)
    pair_raw_k = tl.load(
        raw_k_ptr
        + layer_idx * raw_k_layer_stride
        + token_idx * raw_k_token_stride
        + head_idx * raw_k_head_stride
        + pair_idx,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    pair_norm_weight = tl.load(
        norm_weight_ptr + layer_idx * norm_weight_layer_stride + pair_idx,
        mask=pair_mask,
        other=1.0,
    ).to(tl.float32)
    pair_normed = pair_raw_k * inv_rms * pair_norm_weight

    base_idx = tl.where(is_first_half, offs, offs - half_rotary_dim)
    cos = tl.load(
        cos_sin_cache_ptr + position * cos_sin_stride + base_idx,
        mask=is_rotary,
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + position * cos_sin_stride + half_rotary_dim + base_idx,
        mask=is_rotary,
        other=0.0,
    ).to(tl.float32)
    k_rot = tl.where(
        is_first_half,
        k_normed * cos - pair_normed * sin,
        k_normed * cos + pair_normed * sin,
    )
    k_out = tl.where(is_rotary, k_rot, k_normed)

    tl.store(
        dst_k_ptr
        + layer_idx * dst_k_layer_stride
        + slot_idx * dst_k_slot_stride
        + head_idx * dst_k_head_stride
        + offs,
        k_out.to(raw_v.dtype),
        mask=mask_hd,
    )
    tl.store(
        dst_v_ptr
        + layer_idx * dst_k_layer_stride
        + slot_idx * dst_k_slot_stride
        + head_idx * dst_k_head_stride
        + offs,
        raw_v,
        mask=mask_hd,
    )


@triton.jit
def _commit_postproj_kernel(
    raw_k_ptr,
    raw_v_ptr,
    dst_k_ptr,
    dst_v_ptr,
    slot_ids_ptr,
    commit_lens_ptr,
    positions_ptr,
    norm_weight_ptr,
    eps_ptr,
    cos_sin_cache_ptr,
    raw_k_layer_stride,
    raw_k_batch_stride,
    raw_k_block_stride,
    raw_k_head_stride,
    dst_k_layer_stride,
    dst_k_slot_stride,
    dst_k_head_stride,
    slot_batch_stride,
    slot_block_stride,
    commit_len_stride,
    pos_batch_stride,
    pos_block_stride,
    norm_weight_layer_stride,
    cos_sin_stride,
    batch_size,
    block_size,
    num_heads,
    head_dim,
    rotary_dim,
    half_rotary_dim,
    BLOCK_HD: tl.constexpr,
):
    row_idx = tl.program_id(0)
    rows_per_layer = batch_size * block_size * num_heads
    layer_idx = row_idx // rows_per_layer
    row_rem = row_idx % rows_per_layer
    batch_idx = row_rem // (block_size * num_heads)
    row_rem = row_rem % (block_size * num_heads)
    block_idx = row_rem // num_heads
    head_idx = row_rem % num_heads

    keep = tl.load(commit_lens_ptr + batch_idx * commit_len_stride)
    if block_idx >= keep:
        return

    slot_idx = tl.load(
        slot_ids_ptr + batch_idx * slot_batch_stride + block_idx * slot_block_stride
    )
    position = tl.load(
        positions_ptr + batch_idx * pos_batch_stride + block_idx * pos_block_stride
    )
    eps = tl.load(eps_ptr + layer_idx).to(tl.float32)

    offs = tl.arange(0, BLOCK_HD)
    mask_hd = offs < head_dim
    raw_k = tl.load(
        raw_k_ptr
        + layer_idx * raw_k_layer_stride
        + batch_idx * raw_k_batch_stride
        + block_idx * raw_k_block_stride
        + head_idx * raw_k_head_stride
        + offs,
        mask=mask_hd,
        other=0.0,
    ).to(tl.float32)
    raw_v = tl.load(
        raw_v_ptr
        + layer_idx * raw_k_layer_stride
        + batch_idx * raw_k_batch_stride
        + block_idx * raw_k_block_stride
        + head_idx * raw_k_head_stride
        + offs,
        mask=mask_hd,
        other=0.0,
    )
    norm_weight = tl.load(
        norm_weight_ptr + layer_idx * norm_weight_layer_stride + offs,
        mask=mask_hd,
        other=1.0,
    ).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(raw_k * raw_k) / head_dim + eps)
    k_normed = raw_k * inv_rms * norm_weight

    is_first_half = offs < half_rotary_dim
    is_rotary = offs < rotary_dim
    pair_idx = tl.where(is_first_half, offs + half_rotary_dim, offs - half_rotary_dim)
    pair_mask = is_rotary & (pair_idx < head_dim)
    pair_raw_k = tl.load(
        raw_k_ptr
        + layer_idx * raw_k_layer_stride
        + batch_idx * raw_k_batch_stride
        + block_idx * raw_k_block_stride
        + head_idx * raw_k_head_stride
        + pair_idx,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    pair_norm_weight = tl.load(
        norm_weight_ptr + layer_idx * norm_weight_layer_stride + pair_idx,
        mask=pair_mask,
        other=1.0,
    ).to(tl.float32)
    pair_normed = pair_raw_k * inv_rms * pair_norm_weight
    base_idx = tl.where(is_first_half, offs, offs - half_rotary_dim)
    cos = tl.load(
        cos_sin_cache_ptr + position * cos_sin_stride + base_idx,
        mask=is_rotary,
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + position * cos_sin_stride + half_rotary_dim + base_idx,
        mask=is_rotary,
        other=0.0,
    ).to(tl.float32)
    k_rot = tl.where(
        is_first_half,
        k_normed * cos - pair_normed * sin,
        k_normed * cos + pair_normed * sin,
    )
    k_out = tl.where(is_rotary, k_rot, k_normed)

    tl.store(
        dst_k_ptr
        + layer_idx * dst_k_layer_stride
        + slot_idx * dst_k_slot_stride
        + head_idx * dst_k_head_stride
        + offs,
        k_out.to(raw_v.dtype),
        mask=mask_hd,
    )
    tl.store(
        dst_v_ptr
        + layer_idx * dst_k_layer_stride
        + slot_idx * dst_k_slot_stride
        + head_idx * dst_k_head_stride
        + offs,
        raw_v,
        mask=mask_hd,
    )


def postprocess_prompt_triton(
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
    raw_k, raw_v, positions, slot_ids, cos_sin_cache = _validate_prompt_inputs(
        cache=cache,
        config=config,
        weights=weights,
        projected=projected,
        positions=positions,
        slot_ids=slot_ids,
        cos_sin_cache=cos_sin_cache,
    )
    updated = cache if inplace else cache.clone()
    num_tokens = int(slot_ids.numel())
    if num_tokens == 0:
        return updated

    raw_k = raw_k.contiguous()
    raw_v = raw_v.contiguous()
    norm_weight = weights.k_norm_weight.to(
        device=raw_k.device, dtype=raw_k.dtype
    ).contiguous()
    eps = weights.k_norm_eps.to(device=raw_k.device, dtype=torch.float32).contiguous()
    cos_sin_cache = cos_sin_cache.contiguous()
    block_hd = triton.next_power_of_2(config.head_dim)
    grid = (config.num_layers * num_tokens * config.num_kv_heads,)
    _prompt_postproj_kernel[grid](
        raw_k,
        raw_v,
        updated.k_cache,
        updated.v_cache,
        slot_ids,
        positions,
        norm_weight,
        eps,
        cos_sin_cache,
        raw_k.stride(0),
        raw_k.stride(1),
        raw_k.stride(2),
        updated.k_cache.stride(0),
        updated.k_cache.stride(1),
        updated.k_cache.stride(2),
        slot_ids.stride(0),
        positions.stride(0),
        norm_weight.stride(0),
        cos_sin_cache.stride(0),
        num_tokens,
        config.num_kv_heads,
        config.head_dim,
        config.rotary_dim,
        config.rotary_dim // 2,
        BLOCK_HD=block_hd,
        num_warps=_pick_num_warps(config.head_dim),
    )
    return updated


def postprocess_commit_triton(
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
    raw_k, raw_v, positions, slot_ids_2d, commit_lens, cos_sin_cache = (
        _validate_commit_inputs(
            cache=cache,
            config=config,
            weights=weights,
            projected=projected,
            positions=positions,
            slot_ids_2d=slot_ids_2d,
            commit_lens=commit_lens,
            cos_sin_cache=cos_sin_cache,
        )
    )
    updated = cache if inplace else cache.clone()
    batch_size, block_size = slot_ids_2d.shape
    if batch_size == 0 or int(commit_lens.max().item()) == 0:
        return updated

    raw_k = raw_k.contiguous()
    raw_v = raw_v.contiguous()
    norm_weight = weights.k_norm_weight.to(
        device=raw_k.device, dtype=raw_k.dtype
    ).contiguous()
    eps = weights.k_norm_eps.to(device=raw_k.device, dtype=torch.float32).contiguous()
    cos_sin_cache = cos_sin_cache.contiguous()
    block_hd = triton.next_power_of_2(config.head_dim)
    grid = (config.num_layers * batch_size * block_size * config.num_kv_heads,)
    _commit_postproj_kernel[grid](
        raw_k,
        raw_v,
        updated.k_cache,
        updated.v_cache,
        slot_ids_2d,
        commit_lens,
        positions,
        norm_weight,
        eps,
        cos_sin_cache,
        raw_k.stride(0),
        raw_k.stride(1),
        raw_k.stride(2),
        raw_k.stride(3),
        updated.k_cache.stride(0),
        updated.k_cache.stride(1),
        updated.k_cache.stride(2),
        slot_ids_2d.stride(0),
        slot_ids_2d.stride(1),
        commit_lens.stride(0),
        positions.stride(0),
        positions.stride(1),
        norm_weight.stride(0),
        cos_sin_cache.stride(0),
        batch_size,
        block_size,
        config.num_kv_heads,
        config.head_dim,
        config.rotary_dim,
        config.rotary_dim // 2,
        BLOCK_HD=block_hd,
        num_warps=_pick_num_warps(config.head_dim),
    )
    return updated
