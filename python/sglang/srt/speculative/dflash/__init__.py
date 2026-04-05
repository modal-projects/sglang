"""DFlash kernel experiment lab.

This package currently contains the isolated kernel/reference/benchmark work
for the clean-room DFlash rewrite. Runtime bring-up layers were intentionally
removed so the remaining tree stays focused on kernel experiments only.
"""

from sglang.srt.speculative.dflash.contracts import (
    GPU_STOP_MASK,
    STATUS_ACTIVE,
    STATUS_CANCELED,
    STATUS_EOS_SEEN,
    STATUS_FINISHED,
    STATUS_STOPPED_BY_TOKEN,
    DFlashAcceptBonusResult,
    DFlashDirectEmbeddingResult,
    DFlashKVCache,
    DFlashMaterializerConfig,
    DFlashMaterializerWeights,
    DFlashPrepareBlockResult,
    DFlashProjectedKV,
    DFlashRequestStateTable,
    compute_live_row_mask,
)

__all__ = [
    "DFlashAcceptBonusResult",
    "DFlashDirectEmbeddingResult",
    "DFlashKVCache",
    "DFlashMaterializerConfig",
    "DFlashMaterializerWeights",
    "DFlashPrepareBlockResult",
    "DFlashProjectedKV",
    "DFlashRequestStateTable",
    "GPU_STOP_MASK",
    "STATUS_ACTIVE",
    "STATUS_CANCELED",
    "STATUS_EOS_SEEN",
    "STATUS_FINISHED",
    "STATUS_STOPPED_BY_TOKEN",
    "compute_live_row_mask",
]
