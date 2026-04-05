from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dflash.contracts import (
    DFlashKVCache,
    DFlashMaterializerConfig,
    DFlashMaterializerWeights,
)
from sglang.srt.speculative.dflash.reference.post_projection_packed import (
    _validate_commit_packed_inputs,
    _validate_prompt_packed_inputs,
)


def _pick_num_warps(head_dim: int) -> int:
    if head_dim <= 64:
        return 2
    if head_dim <= 128:
        return 4
    return 8


@triton.jit
def _prompt_packed_postproj_kernel(
    packed_kv_ptr,
    dst_k_ptr,
    dst_v_ptr,
    slot_ids_ptr,
    positions_ptr,
    norm_weight_ptr,
    eps_ptr,
    cos_sin_cache_ptr,
    packed_token_stride,
    packed_group_stride,
    packed_pair_stride,
    packed_head_stride,
    dst_k_layer_stride,
    dst_k_slot_stride,
    dst_k_head_stride,
    slot_stride,
    pos_stride,
    norm_weight_layer_stride,
    cos_sin_stride,
    layer_start,
    group_count,
    num_tokens,
    num_heads,
    head_dim,
    rotary_dim,
    half_rotary_dim,
    BLOCK_HD: tl.constexpr,
):
    row_idx = tl.program_id(0)
    rows_per_group = num_tokens * num_heads
    group_idx = row_idx // rows_per_group
    row_rem = row_idx % rows_per_group
    token_idx = row_rem // num_heads
    head_idx = row_rem % num_heads
    if group_idx >= group_count:
        return

    layer_idx = layer_start + group_idx
    slot_idx = tl.load(slot_ids_ptr + token_idx * slot_stride)
    position = tl.load(positions_ptr + token_idx * pos_stride)
    eps = tl.load(eps_ptr + layer_idx).to(tl.float32)

    offs = tl.arange(0, BLOCK_HD)
    mask_hd = offs < head_dim
    packed_base = (
        token_idx * packed_token_stride
        + group_idx * packed_group_stride
        + head_idx * packed_head_stride
        + offs
    )
    raw_k = tl.load(
        packed_kv_ptr + packed_base,
        mask=mask_hd,
        other=0.0,
    ).to(tl.float32)
    raw_v = tl.load(
        packed_kv_ptr + packed_base + packed_pair_stride,
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
        packed_kv_ptr
        + token_idx * packed_token_stride
        + group_idx * packed_group_stride
        + head_idx * packed_head_stride
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

    dst_base = (
        layer_idx * dst_k_layer_stride
        + slot_idx * dst_k_slot_stride
        + head_idx * dst_k_head_stride
        + offs
    )
    tl.store(dst_k_ptr + dst_base, k_out.to(raw_v.dtype), mask=mask_hd)
    tl.store(dst_v_ptr + dst_base, raw_v, mask=mask_hd)


@triton.jit
def _commit_packed_postproj_kernel(
    packed_kv_ptr,
    dst_k_ptr,
    dst_v_ptr,
    slot_ids_ptr,
    commit_lens_ptr,
    positions_ptr,
    norm_weight_ptr,
    eps_ptr,
    cos_sin_cache_ptr,
    packed_batch_stride,
    packed_block_stride,
    packed_group_stride,
    packed_pair_stride,
    packed_head_stride,
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
    layer_start,
    batch_size,
    block_size,
    group_count,
    num_heads,
    head_dim,
    rotary_dim,
    half_rotary_dim,
    BLOCK_HD: tl.constexpr,
):
    row_idx = tl.program_id(0)
    rows_per_group = batch_size * block_size * num_heads
    group_idx = row_idx // rows_per_group
    row_rem = row_idx % rows_per_group
    batch_idx = row_rem // (block_size * num_heads)
    row_rem = row_rem % (block_size * num_heads)
    block_idx = row_rem // num_heads
    head_idx = row_rem % num_heads
    if group_idx >= group_count:
        return

    keep = tl.load(commit_lens_ptr + batch_idx * commit_len_stride)
    if block_idx >= keep:
        return

    layer_idx = layer_start + group_idx
    slot_idx = tl.load(
        slot_ids_ptr + batch_idx * slot_batch_stride + block_idx * slot_block_stride
    )
    position = tl.load(
        positions_ptr + batch_idx * pos_batch_stride + block_idx * pos_block_stride
    )
    eps = tl.load(eps_ptr + layer_idx).to(tl.float32)

    offs = tl.arange(0, BLOCK_HD)
    mask_hd = offs < head_dim
    packed_base = (
        batch_idx * packed_batch_stride
        + block_idx * packed_block_stride
        + group_idx * packed_group_stride
        + head_idx * packed_head_stride
        + offs
    )
    raw_k = tl.load(
        packed_kv_ptr + packed_base,
        mask=mask_hd,
        other=0.0,
    ).to(tl.float32)
    raw_v = tl.load(
        packed_kv_ptr + packed_base + packed_pair_stride,
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
        packed_kv_ptr
        + batch_idx * packed_batch_stride
        + block_idx * packed_block_stride
        + group_idx * packed_group_stride
        + head_idx * packed_head_stride
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

    dst_base = (
        layer_idx * dst_k_layer_stride
        + slot_idx * dst_k_slot_stride
        + head_idx * dst_k_head_stride
        + offs
    )
    tl.store(dst_k_ptr + dst_base, k_out.to(raw_v.dtype), mask=mask_hd)
    tl.store(dst_v_ptr + dst_base, raw_v, mask=mask_hd)


def postprocess_prompt_packed_triton(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    packed_kv: torch.Tensor,
    layer_start: int,
    positions,
    slot_ids,
    cos_sin_cache,
    inplace: bool = False,
) -> DFlashKVCache:
    packed_kv, group_count, positions, slot_ids, cos_sin_cache = (
        _validate_prompt_packed_inputs(
            cache=cache,
            config=config,
            weights=weights,
            packed_kv=packed_kv,
            layer_start=layer_start,
            positions=positions,
            slot_ids=slot_ids,
            cos_sin_cache=cos_sin_cache,
        )
    )
    updated = cache if inplace else cache.clone()
    if int(slot_ids.numel()) == 0:
        return updated
    block_hd = triton.next_power_of_2(config.head_dim)
    num_warps = _pick_num_warps(config.head_dim)
    total_rows = int(group_count) * int(slot_ids.numel()) * config.num_kv_heads
    _prompt_packed_postproj_kernel[(total_rows,)](
        packed_kv.contiguous(),
        updated.k_cache,
        updated.v_cache,
        slot_ids.contiguous(),
        positions.contiguous(),
        weights.k_norm_weight.to(
            device=packed_kv.device, dtype=packed_kv.dtype
        ).contiguous(),
        weights.k_norm_eps.to(
            device=packed_kv.device, dtype=cos_sin_cache.dtype
        ).contiguous(),
        cos_sin_cache.contiguous(),
        packed_kv.stride(0),
        packed_kv.stride(1),
        packed_kv.stride(2),
        packed_kv.stride(3),
        updated.k_cache.stride(0),
        updated.k_cache.stride(1),
        updated.k_cache.stride(2),
        slot_ids.stride(0),
        positions.stride(0),
        weights.k_norm_weight.stride(0),
        cos_sin_cache.stride(0),
        layer_start,
        group_count,
        int(slot_ids.numel()),
        config.num_kv_heads,
        config.head_dim,
        config.rotary_dim,
        config.rotary_dim // 2,
        BLOCK_HD=block_hd,
        num_warps=num_warps,
    )
    return updated


def postprocess_commit_packed_triton(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    packed_kv: torch.Tensor,
    layer_start: int,
    positions,
    slot_ids_2d,
    commit_lens,
    cos_sin_cache,
    inplace: bool = False,
) -> DFlashKVCache:
    packed_kv, group_count, positions, slot_ids_2d, commit_lens, cos_sin_cache = (
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
    block_hd = triton.next_power_of_2(config.head_dim)
    num_warps = _pick_num_warps(config.head_dim)
    total_rows = (
        group_count
        * int(slot_ids_2d.shape[0])
        * int(slot_ids_2d.shape[1])
        * config.num_kv_heads
    )
    _commit_packed_postproj_kernel[(total_rows,)](
        packed_kv.contiguous(),
        updated.k_cache,
        updated.v_cache,
        slot_ids_2d.contiguous(),
        commit_lens.contiguous(),
        positions.contiguous(),
        weights.k_norm_weight.to(
            device=packed_kv.device, dtype=packed_kv.dtype
        ).contiguous(),
        weights.k_norm_eps.to(
            device=packed_kv.device, dtype=cos_sin_cache.dtype
        ).contiguous(),
        cos_sin_cache.contiguous(),
        packed_kv.stride(0),
        packed_kv.stride(1),
        packed_kv.stride(2),
        packed_kv.stride(3),
        packed_kv.stride(4),
        updated.k_cache.stride(0),
        updated.k_cache.stride(1),
        updated.k_cache.stride(2),
        slot_ids_2d.stride(0),
        slot_ids_2d.stride(1),
        commit_lens.stride(0),
        positions.stride(0),
        positions.stride(1),
        weights.k_norm_weight.stride(0),
        cos_sin_cache.stride(0),
        layer_start,
        int(slot_ids_2d.shape[0]),
        int(slot_ids_2d.shape[1]),
        group_count,
        config.num_kv_heads,
        config.head_dim,
        config.rotary_dim,
        config.rotary_dim // 2,
        BLOCK_HD=block_hd,
        num_warps=num_warps,
    )
    return updated
