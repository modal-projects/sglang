from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dflash.contracts import (
    DFlashKVCache,
    DFlashMaterializerConfig,
)
from sglang.srt.speculative.dflash.reference.kv_prefix_write import (
    _validate_commit_write_inputs,
    _validate_prompt_write_inputs,
)


def _pick_launch_config(feature_dim: int) -> tuple[int, int]:
    if feature_dim <= 64:
        return 64, 2
    if feature_dim <= 128:
        return 128, 4
    if feature_dim <= 256:
        return 256, 4
    return 512, 8


@triton.jit
def _prompt_write_kernel(
    src_k_ptr,
    src_v_ptr,
    dst_k_ptr,
    dst_v_ptr,
    slot_ids_ptr,
    src_layer_stride,
    src_token_stride,
    dst_layer_stride,
    dst_slot_stride,
    slot_ids_stride,
    num_tokens,
    feature_dim,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    token_idx = row_idx % num_tokens
    layer_idx = row_idx // num_tokens
    slot_idx = tl.load(slot_ids_ptr + token_idx * slot_ids_stride)
    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < feature_dim

    src_k = (
        src_k_ptr + layer_idx * src_layer_stride + token_idx * src_token_stride + offs
    )
    src_v = (
        src_v_ptr + layer_idx * src_layer_stride + token_idx * src_token_stride + offs
    )
    dst_k = dst_k_ptr + layer_idx * dst_layer_stride + slot_idx * dst_slot_stride + offs
    dst_v = dst_v_ptr + layer_idx * dst_layer_stride + slot_idx * dst_slot_stride + offs

    tl.store(dst_k, tl.load(src_k, mask=mask, other=0), mask=mask)
    tl.store(dst_v, tl.load(src_v, mask=mask, other=0), mask=mask)


@triton.jit
def _commit_write_kernel(
    src_k_ptr,
    src_v_ptr,
    dst_k_ptr,
    dst_v_ptr,
    slot_ids_ptr,
    commit_lens_ptr,
    src_layer_stride,
    src_batch_stride,
    src_block_stride,
    dst_layer_stride,
    dst_slot_stride,
    slot_batch_stride,
    slot_block_stride,
    commit_lens_stride,
    batch_size,
    block_size,
    feature_dim,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    feat_block_idx = tl.program_id(1)
    rows_per_layer = batch_size * block_size
    layer_idx = row_idx // rows_per_layer
    row_offset = row_idx % rows_per_layer
    batch_idx = row_offset // block_size
    block_idx = row_offset % block_size

    keep = tl.load(commit_lens_ptr + batch_idx * commit_lens_stride)
    valid = block_idx < keep
    slot_idx = tl.load(
        slot_ids_ptr + batch_idx * slot_batch_stride + block_idx * slot_block_stride,
        mask=valid,
        other=0,
    )
    offs = feat_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = valid & (offs < feature_dim)

    src_k = (
        src_k_ptr
        + layer_idx * src_layer_stride
        + batch_idx * src_batch_stride
        + block_idx * src_block_stride
        + offs
    )
    src_v = (
        src_v_ptr
        + layer_idx * src_layer_stride
        + batch_idx * src_batch_stride
        + block_idx * src_block_stride
        + offs
    )
    dst_k = dst_k_ptr + layer_idx * dst_layer_stride + slot_idx * dst_slot_stride + offs
    dst_v = dst_v_ptr + layer_idx * dst_layer_stride + slot_idx * dst_slot_stride + offs

    tl.store(dst_k, tl.load(src_k, mask=mask, other=0), mask=mask)
    tl.store(dst_v, tl.load(src_v, mask=mask, other=0), mask=mask)


def write_prompt_triton(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    slot_ids: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
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

    block_size, num_warps = _pick_launch_config(feature_dim)
    grid = (config.num_layers * num_tokens, triton.cdiv(feature_dim, block_size))
    _prompt_write_kernel[grid](
        src_k,
        src_v,
        dst_k,
        dst_v,
        slot_ids,
        src_k.stride(0),
        src_k.stride(1),
        dst_k.stride(0),
        dst_k.stride(1),
        slot_ids.stride(0),
        num_tokens,
        feature_dim,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return updated


def write_commit_prefix_triton(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    slot_ids_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
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

    vec_block, num_warps = _pick_launch_config(feature_dim)
    grid = (
        config.num_layers * batch_size * block_size_tokens,
        triton.cdiv(feature_dim, vec_block),
    )
    _commit_write_kernel[grid](
        src_k,
        src_v,
        dst_k,
        dst_v,
        slot_ids_2d,
        commit_lens,
        src_k.stride(0),
        src_k.stride(1),
        src_k.stride(2),
        dst_k.stride(0),
        dst_k.stride(1),
        slot_ids_2d.stride(0),
        slot_ids_2d.stride(1),
        commit_lens.stride(0),
        batch_size,
        block_size_tokens,
        feature_dim,
        BLOCK_SIZE=vec_block,
        num_warps=num_warps,
    )
    return updated
