from __future__ import annotations

from sglang.jit_kernel.dflash_post_projection_packed import (
    commit_packed_post_projection_cuda,
    prompt_packed_post_projection_cuda,
)
from sglang.srt.speculative.dflash.contracts import (
    DFlashKVCache,
    DFlashMaterializerConfig,
    DFlashMaterializerWeights,
)
from sglang.srt.speculative.dflash.reference.post_projection_packed import (
    _validate_commit_packed_inputs,
    _validate_prompt_packed_inputs,
)


def _prepare_jit_norm_tensors(
    *,
    packed_kv,
    weights: DFlashMaterializerWeights,
    cos_sin_cache,
    k_norm_weight=None,
    k_norm_eps=None,
):
    if k_norm_weight is None:
        k_norm_weight = weights.k_norm_weight.to(
            device=packed_kv.device,
            dtype=packed_kv.dtype,
        ).contiguous()
    if k_norm_eps is None:
        k_norm_eps = weights.k_norm_eps.to(
            device=packed_kv.device,
            dtype=cos_sin_cache.dtype,
        ).contiguous()
    return k_norm_weight, k_norm_eps


def postprocess_prompt_packed_jit_unchecked(
    *,
    cache: DFlashKVCache,
    packed_kv,
    layer_start: int,
    positions,
    slot_ids,
    cos_sin_cache,
    k_norm_weight,
    k_norm_eps,
    inplace: bool = False,
) -> DFlashKVCache:
    updated = cache if inplace else cache.clone()
    if int(slot_ids.numel()) == 0:
        return updated
    prompt_packed_post_projection_cuda(
        packed_kv,
        updated.k_cache,
        updated.v_cache,
        slot_ids,
        positions,
        k_norm_weight,
        k_norm_eps,
        cos_sin_cache,
        int(layer_start),
    )
    return updated


def postprocess_prompt_packed_jit(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    packed_kv,
    layer_start: int,
    positions,
    slot_ids,
    cos_sin_cache,
    inplace: bool = False,
) -> DFlashKVCache:
    packed_kv, _, positions, slot_ids, cos_sin_cache = _validate_prompt_packed_inputs(
        cache=cache,
        config=config,
        weights=weights,
        packed_kv=packed_kv,
        layer_start=layer_start,
        positions=positions,
        slot_ids=slot_ids,
        cos_sin_cache=cos_sin_cache,
    )
    k_norm_weight, k_norm_eps = _prepare_jit_norm_tensors(
        packed_kv=packed_kv,
        weights=weights,
        cos_sin_cache=cos_sin_cache,
    )
    return postprocess_prompt_packed_jit_unchecked(
        cache=cache,
        packed_kv=packed_kv.contiguous(),
        layer_start=layer_start,
        positions=positions.contiguous(),
        slot_ids=slot_ids.contiguous(),
        cos_sin_cache=cos_sin_cache.contiguous(),
        k_norm_weight=k_norm_weight,
        k_norm_eps=k_norm_eps,
        inplace=inplace,
    )


def postprocess_commit_packed_jit_unchecked(
    *,
    cache: DFlashKVCache,
    packed_kv,
    layer_start: int,
    positions,
    slot_ids_2d,
    commit_lens,
    cos_sin_cache,
    k_norm_weight,
    k_norm_eps,
    inplace: bool = False,
) -> DFlashKVCache:
    updated = cache if inplace else cache.clone()
    if int(commit_lens.max().item()) == 0:
        return updated
    commit_packed_post_projection_cuda(
        packed_kv,
        updated.k_cache,
        updated.v_cache,
        slot_ids_2d,
        commit_lens,
        positions,
        k_norm_weight,
        k_norm_eps,
        cos_sin_cache,
        int(layer_start),
    )
    return updated


def postprocess_commit_packed_jit(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    packed_kv,
    layer_start: int,
    positions,
    slot_ids_2d,
    commit_lens,
    cos_sin_cache,
    inplace: bool = False,
) -> DFlashKVCache:
    packed_kv, _, positions, slot_ids_2d, commit_lens, cos_sin_cache = (
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
    k_norm_weight, k_norm_eps = _prepare_jit_norm_tensors(
        packed_kv=packed_kv,
        weights=weights,
        cos_sin_cache=cos_sin_cache,
    )
    return postprocess_commit_packed_jit_unchecked(
        cache=cache,
        packed_kv=packed_kv.contiguous(),
        layer_start=layer_start,
        positions=positions.contiguous(),
        slot_ids_2d=slot_ids_2d.contiguous(),
        commit_lens=commit_lens.contiguous(),
        cos_sin_cache=cos_sin_cache.contiguous(),
        k_norm_weight=k_norm_weight,
        k_norm_eps=k_norm_eps,
        inplace=inplace,
    )
