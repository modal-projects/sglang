from __future__ import annotations

from sglang.jit_kernel.dflash_post_projection import (
    commit_post_projection_cuda,
    prompt_post_projection_cuda,
)
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


def postprocess_prompt_jit(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    projected: DFlashProjectedKV,
    positions,
    slot_ids,
    cos_sin_cache,
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
    if int(slot_ids.numel()) == 0:
        return updated
    prompt_post_projection_cuda(
        raw_k.contiguous(),
        raw_v.contiguous(),
        updated.k_cache,
        updated.v_cache,
        slot_ids.contiguous(),
        positions.contiguous(),
        weights.k_norm_weight.to(device=raw_k.device, dtype=raw_k.dtype).contiguous(),
        weights.k_norm_eps.to(
            device=raw_k.device, dtype=cos_sin_cache.dtype
        ).contiguous(),
        cos_sin_cache.contiguous(),
    )
    return updated


def postprocess_commit_jit(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    projected: DFlashProjectedKV,
    positions,
    slot_ids_2d,
    commit_lens,
    cos_sin_cache,
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
    if int(slot_ids_2d.shape[0]) == 0 or int(commit_lens.max().item()) == 0:
        return updated
    commit_post_projection_cuda(
        raw_k.contiguous(),
        raw_v.contiguous(),
        updated.k_cache,
        updated.v_cache,
        slot_ids_2d.contiguous(),
        commit_lens.contiguous(),
        positions.contiguous(),
        weights.k_norm_weight.to(device=raw_k.device, dtype=raw_k.dtype).contiguous(),
        weights.k_norm_eps.to(
            device=raw_k.device, dtype=cos_sin_cache.dtype
        ).contiguous(),
        cos_sin_cache.contiguous(),
    )
    return updated
