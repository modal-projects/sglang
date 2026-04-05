from __future__ import annotations

from sglang.srt.speculative.dflash.contracts import (
    DFlashMaterializerConfig,
    DFlashMaterializerWeights,
    DFlashProjectedKV,
)
from sglang.srt.speculative.dflash.reference.raw_kv_projection import (
    project_raw_commit_reference,
    project_raw_prompt_reference,
)


def project_raw_prompt_control(
    *,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    hidden,
    positions,
    group_size: int = 1,
    chunk_size: int | None = None,
) -> DFlashProjectedKV:
    return project_raw_prompt_reference(
        config=config,
        weights=weights,
        hidden=hidden,
        positions=positions,
        group_size=group_size,
        chunk_size=chunk_size,
    )


def project_raw_commit_control(
    *,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    verify_hidden,
    positions,
    group_size: int = 1,
) -> DFlashProjectedKV:
    return project_raw_commit_reference(
        config=config,
        weights=weights,
        verify_hidden=verify_hidden,
        positions=positions,
        group_size=group_size,
    )
