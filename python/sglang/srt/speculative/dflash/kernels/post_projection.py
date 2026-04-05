from __future__ import annotations

from sglang.srt.speculative.dflash.contracts import (
    DFlashKVCache,
    DFlashMaterializerConfig,
    DFlashMaterializerWeights,
    DFlashProjectedKV,
)
from sglang.srt.speculative.dflash.reference.post_projection import (
    postprocess_commit_reference,
    postprocess_prompt_reference,
)


def postprocess_prompt_control(
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
    return postprocess_prompt_reference(
        cache=cache,
        config=config,
        weights=weights,
        projected=projected,
        positions=positions,
        slot_ids=slot_ids,
        cos_sin_cache=cos_sin_cache,
        inplace=inplace,
    )


def postprocess_commit_control(
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
    return postprocess_commit_reference(
        cache=cache,
        config=config,
        weights=weights,
        projected=projected,
        positions=positions,
        slot_ids_2d=slot_ids_2d,
        commit_lens=commit_lens,
        cos_sin_cache=cos_sin_cache,
        inplace=inplace,
    )
