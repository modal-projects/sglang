"""Owned DFlash feature captures and their commit metadata."""

import torch
import triton
import triton.language as tl


@triton.jit
def _capture_dflash_features_kernel(
    Hidden,
    Packed,
    Starts,
    CommitLens,
    CheckpointSteps,
    CheckpointIndices,
    Metadata,
    ROW_SIZE: tl.constexpr,
    FEATURE_SIZE: tl.constexpr,
    STRIDE_BATCH: tl.constexpr,
    STRIDE_ROW: tl.constexpr,
    STRIDE_FEATURE: tl.constexpr,
    HAS_CHECKPOINT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    source = (
        req * STRIDE_BATCH
        + offsets // FEATURE_SIZE * STRIDE_ROW
        + offsets % FEATURE_SIZE * STRIDE_FEATURE
    )
    values = tl.load(Hidden + source, offsets < ROW_SIZE, other=0)
    tl.store(Packed + req * ROW_SIZE + offsets, values, offsets < ROW_SIZE)
    if tile == 0:
        start = tl.load(Starts + req).to(tl.int64)
        count = tl.load(CommitLens + req).to(tl.int64)
        end = tl.full((), -1, tl.int64)
        if HAS_CHECKPOINT:
            step = tl.load(CheckpointSteps + req)
            index = tl.load(CheckpointIndices + req)
            end = tl.where((step >= 0) & (index >= 0), start + step + 1, end)
        tl.store(Metadata + req * 3, start)
        tl.store(Metadata + req * 3 + 1, count)
        tl.store(Metadata + req * 3 + 2, end)


def capture_dflash_features(
    hidden: torch.Tensor,
    starts: torch.Tensor,
    commit_lens: torch.Tensor,
    checkpoint_steps: torch.Tensor | None,
    checkpoint_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Own a [batch, block, features] copy and build [batch, 3] metadata.

    The caller owns the returned copy independently of the reusable model output.
    Position/count inputs are contiguous vectors. A valid checkpoint's end is
    start + step + 1; absent or disabled checkpoints have end -1.
    """
    packed = torch.empty(hidden.shape, dtype=hidden.dtype, device=hidden.device)
    metadata = torch.empty(
        (hidden.shape[0], 3), dtype=torch.int64, device=hidden.device
    )
    row_size = hidden.shape[1] * hidden.shape[2]
    _capture_dflash_features_kernel[(hidden.shape[0], triton.cdiv(row_size, 4096))](
        hidden,
        packed,
        starts,
        commit_lens,
        checkpoint_steps,
        checkpoint_indices,
        metadata,
        row_size,
        hidden.shape[2],
        *hidden.stride(),
        checkpoint_steps is not None,
        4096,
    )
    return packed, metadata
