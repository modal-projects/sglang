from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _verify_rows_validity_kernel(
    values,
    valid_rows,
    num_rows: tl.constexpr,
    vocab: tl.constexpr,
    stride_batch: tl.constexpr,
    stride_row: tl.constexpr,
    stride_vocab: tl.constexpr,
    IS_LOGITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = values + row // num_rows * stride_batch + row % num_rows * stride_row
    valid = True
    has_finite = False
    mass = 0.0
    for start in range(0, vocab, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < vocab
        x = tl.load(base + cols * stride_vocab, mask=mask, other=0.0).to(tl.float32)
        if IS_LOGITS:
            allowed = (x == x) & (x != float("inf"))
            has_finite |= (
                tl.sum((mask & (x > -float("inf")) & (x < float("inf"))).to(tl.int32))
                > 0
            )
        else:
            allowed = (x >= 0.0) & (x < float("inf"))
            mass += tl.sum(x, 0)
        valid &= tl.sum((mask & ~allowed).to(tl.int32)) == 0
    if IS_LOGITS:
        valid &= has_finite
    else:
        valid &= (mass > 0.0) & (mass < float("inf"))
    tl.store(valid_rows + row, valid)
    if not valid:
        for start in range(0, vocab, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            if IS_LOGITS:
                x = tl.where(cols == 0, 0.0, -float("inf"))
            else:
                x = tl.where(cols == 0, 1.0, 0.0)
            tl.store(base + cols * stride_vocab, x, mask=cols < vocab)


def prepare_verify_rows_(values: torch.Tensor, *, is_logits: bool) -> torch.Tensor:
    """Preserve pre-repair validity for [batch, rows, vocab] verification values."""
    bs, num_rows, vocab = values.shape
    if values.is_cuda:
        valid_rows = torch.empty((bs, num_rows), dtype=torch.bool, device=values.device)
        args = (values, valid_rows, num_rows, vocab, *values.stride())
        _verify_rows_validity_kernel[(bs * num_rows,)](
            *args, IS_LOGITS=is_logits, BLOCK=1024
        )
        return valid_rows

    finite = torch.isfinite(values)
    if is_logits:
        valid_rows = (finite | values.isneginf()).all(dim=-1) & finite.any(dim=-1)
    else:
        mass = values.sum(dim=-1, dtype=torch.float32)
        valid_rows = (finite & (values >= 0)).all(dim=-1) & mass.isfinite() & (mass > 0)
    # Invalid rows produce a bounded token while the scheduler retires the attempt.
    values.masked_fill_(~valid_rows[..., None], -float("inf") if is_logits else 0.0)
    values[..., 0].copy_(
        torch.where(valid_rows, values[..., 0], 0.0 if is_logits else 1.0)
    )
    return valid_rows


@triton.jit
def _first_invalid_rows_kernel(
    valid_rows,
    correct_lens,
    out,
    num_rows: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    correct_len = tl.load(correct_lens + batch)
    valid = tl.load(valid_rows + batch * num_rows + rows, rows < num_rows, other=True)
    first = tl.min(tl.where((rows <= correct_len) & ~valid, rows, num_rows), 0)
    tl.store(out + batch, tl.where(first < num_rows, first, -1).to(tl.int32))


def write_first_invalid_rows(
    *, valid_rows: torch.Tensor, correct_lens: torch.Tensor, out: torch.Tensor
) -> None:
    """Only rows through the bonus position were reached by this verification."""
    bs, num_rows = valid_rows.shape
    if valid_rows.is_cuda:
        _first_invalid_rows_kernel[(bs,)](
            valid_rows,
            correct_lens,
            out,
            num_rows,
            BLOCK=triton.next_power_of_2(num_rows),
        )
        return
    rows = torch.arange(num_rows, device=valid_rows.device)
    first = torch.where(
        ~valid_rows & (rows <= correct_lens[:, None]), rows, num_rows
    ).amin(dim=1)
    out.copy_(torch.where(first < num_rows, first, -1))
