"""Fused DFlash post-verify bookkeeping for the ROCm DFlash worker."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.row_argmax import _medium_row_argmax


def verify_argmax(x: torch.Tensor) -> torch.Tensor:
    """``torch.argmax(x, -1)`` for fp32 verify logits (NaNs win, first index on
    ties), split over the vocab."""
    assert x.dim() == 2 and x.dtype == torch.float32 and x.stride(1) == 1
    return _medium_row_argmax(x)


@triton.jit
def _floordiv(a, b):
    # torch floor division for b > 0; Triton's // truncates toward zero.
    q = a // b
    return tl.where((a % b != 0) & (a < 0), q - 1, q)


@triton.jit
def _mamba_track_kernel(
    commit_ptr,
    pre_ptr,
    post_ptr,
    last_out,
    steps_out,
    n,
    interval,
    HAS_TRACK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    commit = tl.load(commit_ptr + offs, mask=m, other=0).to(tl.int64)
    tl.store(last_out + offs, commit - 1, mask=m)
    if HAS_TRACK:
        pre = tl.load(pre_ptr + offs, mask=m, other=0).to(tl.int64)
        post = tl.load(post_ptr + offs, mask=m, other=0).to(tl.int64)
        qpre = _floordiv(pre, interval)
        qpost = _floordiv(post, interval)
        ith = tl.maximum(qpost * interval - pre - 1, 0)
        can = (qpre != qpost) & (ith < commit)
        tl.store(steps_out + offs, tl.where(can, ith, -1), mask=m)


def mamba_track_steps(commit_lens, seq_lens_pre, seq_lens_post, interval, has_track):
    """(last_correct_step_indices, mamba_steps_to_track or None), int64, as
    ``DFlashWorkerV2._update_target_mamba_state_after_verify`` derives them."""
    n = commit_lens.shape[0]
    last = torch.empty((n,), dtype=torch.int64, device=commit_lens.device)
    steps = torch.empty_like(last) if has_track else last
    if n:
        BLOCK = 128
        _mamba_track_kernel[(triton.cdiv(n, BLOCK),)](
            commit_lens,
            seq_lens_pre if has_track else commit_lens,
            seq_lens_post if has_track else commit_lens,
            last,
            steps,
            n,
            int(interval) if has_track else 1,
            HAS_TRACK=bool(has_track),
            BLOCK=BLOCK,
            num_warps=1,
        )
    return last, (steps if has_track else None)


@triton.jit
def _compact_lens_kernel(seq_ptr, dpl_out, suffix_out, n, W, PAGE, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    s = tl.load(seq_ptr + offs, mask=m, other=0).to(tl.int64)
    s32 = s.to(tl.int32)
    vis = tl.where(s32 > W, W, s32).to(tl.int64)
    if PAGE > 1:
        vs = s - vis
        r = vs % PAGE
        r = tl.where(r < 0, r + PAGE, r)
        dpl = (s - (vs - r)).to(tl.int32)
    else:
        dpl = vis.to(tl.int32)
    tl.store(dpl_out + offs, dpl, mask=m)
    tl.store(suffix_out + offs, s - dpl.to(tl.int64), mask=m)


def compact_draft_lens(seq_lens, window, page_size):
    """(draft_prefix_lens int32, suffix_start int64) as
    ``DFlashWorkerV2._compute_compact_draft_seq_lens`` and
    ``_rebuild_compact_draft_cache`` derive them, in one launch."""
    n = seq_lens.shape[0]
    dpl = torch.empty((n,), dtype=torch.int32, device=seq_lens.device)
    suffix = torch.empty((n,), dtype=torch.int64, device=seq_lens.device)
    if n:
        BLOCK = 128
        _compact_lens_kernel[(triton.cdiv(n, BLOCK),)](
            seq_lens,
            dpl,
            suffix,
            n,
            int(window),
            int(page_size),
            BLOCK=BLOCK,
            num_warps=1,
        )
    return dpl, suffix
