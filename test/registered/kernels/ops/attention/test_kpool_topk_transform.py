from __future__ import annotations

from typing import Optional

import pytest
import torch

from sglang.kernels.ops.moe.kpool_topk_transform import (
    fast_kpool_topk_transform_fused,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _reference(
    score: torch.Tensor,
    lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    *,
    page_table: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    rows = score.shape[0]
    group_topk = topk // pool_size
    offsets = torch.arange(pool_size, dtype=torch.int32, device=score.device)
    out_cols = topk + (pool_size - 1 if seq_lens is not None else 0)
    out = torch.full((rows, out_cols), -1, dtype=torch.int32, device=score.device)

    for row in range(rows):
        start = 0 if row_starts is None else int(row_starts[row])
        length = int(lengths[row])
        valid_count = min(length, group_topk)
        if length <= group_topk:
            selected = torch.arange(length, dtype=torch.int32, device=score.device)
        else:
            selected = torch.topk(
                score[row, start : start + length],
                group_topk,
                sorted=False,
            ).indices.to(torch.int32)
        token_ids = (selected.unsqueeze(1) * pool_size + offsets).reshape(-1)

        table_row = (
            row if page_table_row_index is None else int(page_table_row_index[row])
        )
        if page_table is not None:
            token_ids = page_table[table_row, token_ids.long()].to(torch.int32)
        elif topk_indices_offset is not None:
            token_ids = token_ids + topk_indices_offset[row]
        write_pos = valid_count * pool_size
        out[row, :write_pos] = token_ids[:write_pos]

        if seq_lens is not None:
            tail_count = int(seq_lens[row]) % pool_size
            raw_tail = length * pool_size + torch.arange(
                tail_count, dtype=torch.int32, device=score.device
            )
            if page_table is not None:
                raw_tail = page_table[table_row, raw_tail.long()].to(torch.int32)
            elif topk_indices_offset is not None:
                raw_tail = raw_tail + topk_indices_offset[row]
            out[row, write_pos : write_pos + tail_count] = raw_tail
    return out


@pytest.mark.parametrize(
    "pool_size,group_topk",
    [(16, 128), (16, 160), (16, 192), (16, 224), (8, 256), (4, 512)],
)
@pytest.mark.parametrize("mode", ["raw", "paged", "ragged"])
@pytest.mark.parametrize("append_tail", [False, True])
@pytest.mark.parametrize("use_row_starts", [False, True])
@torch.inference_mode()
def test_kpool_topk_transform_matches_reference(
    pool_size: int,
    group_topk: int,
    mode: str,
    append_tail: bool,
    use_row_starts: bool,
) -> None:
    torch.manual_seed(42)
    rows = 5
    cols = 1024
    topk = pool_size * group_topk
    storage = torch.randn(rows, cols + 37, dtype=torch.float32, device="cuda")
    score = storage[:, :cols]
    row_starts = (
        torch.randint(0, 128, (rows,), dtype=torch.int32, device="cuda")
        if use_row_starts
        else None
    )
    max_lengths = (
        cols - row_starts
        if row_starts is not None
        else torch.full((rows,), cols, dtype=torch.int32, device="cuda")
    )
    min_length = min(group_topk + 1, cols // 2)
    lengths = torch.stack(
        [
            torch.randint(
                min_length,
                int(max_lengths[row]) + 1,
                (),
                dtype=torch.int32,
                device="cuda",
            )
            for row in range(rows)
        ]
    )

    page_table = None
    page_table_row_index = None
    topk_indices_offset = None
    if mode == "paged":
        table_cols = cols * pool_size + pool_size
        page_table = torch.arange(
            rows * table_cols, dtype=torch.int32, device="cuda"
        ).view(rows, table_cols)
        page_table_row_index = torch.arange(
            rows - 1, -1, -1, dtype=torch.int32, device="cuda"
        )
    elif mode == "ragged":
        topk_indices_offset = torch.randint(
            0, 2048, (rows,), dtype=torch.int32, device="cuda"
        )

    seq_lens = None
    if append_tail:
        tail_counts = torch.randint(
            0, pool_size, (rows,), dtype=torch.int32, device="cuda"
        )
        seq_lens = lengths * pool_size + tail_counts

    expected = _reference(
        score,
        lengths,
        pool_size,
        topk,
        page_table=page_table,
        topk_indices_offset=topk_indices_offset,
        row_starts=row_starts,
        seq_lens=seq_lens,
        page_table_row_index=page_table_row_index,
    )
    actual = fast_kpool_topk_transform_fused(
        score=score,
        lengths=lengths,
        pool_size=pool_size,
        topk=topk,
        page_table=page_table,
        topk_indices_offset=topk_indices_offset,
        row_starts=row_starts,
        seq_lens=seq_lens,
        page_table_row_index=page_table_row_index,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        torch.sort(actual, dim=-1).values,
        torch.sort(expected, dim=-1).values,
        atol=0,
        rtol=0,
    )


@torch.inference_mode()
def test_kpool_topk_uses_allocated_stride_for_packed_ragged_logits() -> None:
    """DeepGEMM may pack live ragged logits beyond the tensor view's columns."""
    torch.manual_seed(7)
    rows = 2
    visible_width = 1024
    allocated_width = 1328
    pool_size = 16
    topk = 2048
    storage = torch.randn(rows, allocated_width, dtype=torch.float32, device="cuda")
    score = storage[:, :visible_width]
    row_starts = torch.tensor([1024, 1000], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([304, 328], dtype=torch.int32, device="cuda")

    expected = _reference(
        storage,
        lengths,
        pool_size,
        topk,
        row_starts=row_starts,
    )
    actual = fast_kpool_topk_transform_fused(
        score=score,
        lengths=lengths,
        pool_size=pool_size,
        topk=topk,
        row_starts=row_starts,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        torch.sort(actual, dim=-1).values,
        torch.sort(expected, dim=-1).values,
        atol=0,
        rtol=0,
    )


@torch.inference_mode()
def test_kpool_topk_clamps_lengths_to_allocated_logits_row() -> None:
    """Invalid runtime metadata must neither read nor translate out of bounds."""
    torch.manual_seed(0)
    width = 1024
    pool_size = 4
    topk = 2048
    score = torch.randn(3, width, dtype=torch.float32, device="cuda")
    row_starts = torch.tensor([0, 200, 900], dtype=torch.int32, device="cuda")
    lengths = torch.full((3,), 4096, dtype=torch.int32, device="cuda")
    seq_lens = lengths * pool_size + 1

    result = fast_kpool_topk_transform_fused(
        score=score,
        lengths=lengths,
        pool_size=pool_size,
        topk=topk,
        row_starts=row_starts,
        seq_lens=seq_lens,
    )
    torch.cuda.synchronize()

    for row, start in enumerate(row_starts.cpu().tolist()):
        available_groups = width - start
        history_tokens = min(available_groups * pool_size, topk)
        history = result[row, :history_tokens]
        assert torch.all(history >= 0)
        assert torch.all(history < available_groups * pool_size)
        # The logical tail cannot be translated safely when the history had to
        # be clamped to the materialized logits row.
        assert torch.all(result[row, history_tokens:] == -1)
