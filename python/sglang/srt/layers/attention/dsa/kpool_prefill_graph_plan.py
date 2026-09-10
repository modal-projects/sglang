"""Bounded persistent metadata prototype for pooled-indexer prefill capture."""

import dataclasses

import torch

from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    BLOCK_SIZE_K,
    INDEX_HEAD_DIM,
    gather_index_k_scale_prefix_into,
    kpool_assemble_softmax_rotate_write_cache,
    scatter_kpool_tail_updates,
)
from sglang.srt.layers.attention.dsa.kpool_plan import (
    KPoolExtendPlan,
    PoolWriteRows,
    TailWriteRows,
)


class KPoolPrefillGraphPlan:
    def __init__(self, max_tokens, max_requests, max_history_rows, device, pool_size=4,
                 max_context=None, mapping_mode=None):
        if min(max_tokens, max_requests, max_history_rows, pool_size) <= 0:
            raise ValueError("Capture capacities must be positive")
        if max_history_rows % BLOCK_SIZE_K:
            raise ValueError("History capacity must contain complete packed pages")
        self.max_tokens = max_tokens
        self.max_requests = max_requests
        self.max_history_rows = max_history_rows
        self.max_writes = (max_tokens + (pool_size - 1) * max_requests) // pool_size
        self.pool_size = pool_size
        self.layout = None
        self.ragged_paged_page_table = None
        self.live_tokens = self.token_capacity = max_tokens
        self.max_context = max_context
        self.max_logits_rows = max(1, (max_context or max_history_rows * pool_size) // pool_size)
        self.skip_logits = max_context is not None and max_context <= 2048
        self.mapping_mode = mapping_mode
        self._owns_page_table = max_context is not None and mapping_mode == "paged"

        def zeros(n, dtype=torch.int32):
            return torch.zeros(n, dtype=dtype, device=device)

        self.writes = PoolWriteRows(
            req=zeros(self.max_writes, torch.int64),
            pool_id=zeros(self.max_writes, torch.int64),
            n_from_tail=zeros(self.max_writes),
            chunk_src=zeros(self.max_writes, torch.int64),
            tail_logical_base=zeros(self.max_writes),
            write_loc=zeros(self.max_writes, torch.int64),
        )
        self.tails = TailWriteRows(
            req=zeros(max_requests, torch.int64),
            dst_logical_start=zeros(max_requests),
            chunk_src=zeros(max_requests, torch.int64),
            n_write=zeros(max_requests),
        )
        self.write_mask = zeros(self.max_writes, torch.bool)
        self.num_history_rows = zeros(1)
        self.num_tokens = zeros(1)
        self.seq_lens_expanded = zeros(max_tokens)
        self.pooled_seq_lens_expanded = zeros(max_tokens)
        self.ragged_q_ks = zeros(max_tokens)
        self.ragged_q_ke = zeros(max_tokens)
        self.ragged_concat_page_table = zeros(max_history_rows // BLOCK_SIZE_K)
        self.ragged_paged_page_table_row_index = zeros(max_tokens)
        self.topk_offsets = zeros(max_tokens)
        self.ragged_k_u8 = zeros((max_history_rows, INDEX_HEAD_DIM), torch.uint8)
        self.ragged_k_scale = zeros(max_history_rows, torch.float32)
        if self._owns_page_table:
            self.ragged_paged_page_table = zeros((max_requests, max_context))
            self.layout = "paged"

    def clear(self):
        for rows in (self.writes, self.tails):
            for field in dataclasses.fields(rows):
                getattr(rows, field.name).zero_()
        for name in (
            "write_mask", "num_history_rows", "num_tokens", "seq_lens_expanded",
            "pooled_seq_lens_expanded", "ragged_q_ks", "ragged_q_ke",
            "ragged_concat_page_table", "ragged_paged_page_table_row_index", "topk_offsets",
        ):
            getattr(self, name).zero_()
        if self._owns_page_table:
            self.ragged_paged_page_table.zero_()

    def can_fit(self, plan: KPoolExtendPlan):
        if plan is None:
            return False
        layout = "paged" if plan.ragged_paged_page_table is not None else "ragged"
        if plan.cp is not None or self.layout not in (None, layout):
            return False
        if not (
            plan.seq_lens_expanded.numel() <= self.max_tokens
            and plan.writes.req.numel() <= self.max_writes
            and plan.tails.req.numel() <= self.max_requests
            and 0 <= plan.ragged_total_k_rows <= self.max_history_rows
            and plan.ragged_total_k_rows % BLOCK_SIZE_K == 0
            and plan.ragged_concat_page_table.numel()
            == plan.ragged_total_k_rows // BLOCK_SIZE_K
        ):
            return False
        if self.ragged_paged_page_table is not None and not self._owns_page_table:
            return (
                plan.ragged_paged_page_table.data_ptr()
                == self.ragged_paged_page_table.data_ptr()
                and plan.ragged_paged_page_table.shape
                == self.ragged_paged_page_table.shape
            )
        return True

    def update(self, plan: KPoolExtendPlan, topk_offsets=None, *, req_pool_indices=None,
               extend_seq_lens=None, max_seq_len=None):
        if not self.can_fit(plan):
            raise ValueError("Pooled-indexer plan exceeds or changes the capture contract")
        n = plan.seq_lens_expanded.numel()
        if self.max_context is not None and (
            not 0 < n <= self.live_tokens or max_seq_len is None or max_seq_len > self.max_context
            or (max_seq_len <= 2048) != self.skip_logits
        ):
            raise ValueError("KPool batch changes captured query or context dispatch")
        copies = []
        for target, source in ((self.writes, plan.writes), (self.tails, plan.tails)):
            count = source.req.numel()
            for field in dataclasses.fields(source):
                value = getattr(source, field.name)
                if value.ndim != 1 or value.numel() != count:
                    raise ValueError("Inconsistent pooled-indexer write metadata")
                copies.append((getattr(target, field.name), value))
        for name in ("seq_lens_expanded", "pooled_seq_lens_expanded", "ragged_q_ks", "ragged_q_ke"):
            value = getattr(plan, name)
            if value.ndim != 1 or value.numel() != n:
                raise ValueError("Inconsistent pooled-indexer query metadata")
            copies.append((getattr(self, name), value))
        if plan.ragged_concat_page_table.ndim != 1:
            raise ValueError("Invalid pooled-indexer page list")
        copies.append((self.ragged_concat_page_table, plan.ragged_concat_page_table))
        table_copy = None
        if plan.ragged_paged_page_table is not None:
            rows = plan.ragged_paged_page_table_row_index
            if rows is None or rows.ndim != 1 or rows.numel() != n:
                raise ValueError("Missing pooled-indexer page-table row mapping")
            if self._owns_page_table:
                if (req_pool_indices is None or extend_seq_lens is None
                    or req_pool_indices.numel() != len(extend_seq_lens)
                    or not 0 < len(extend_seq_lens) <= self.max_requests
                    or sum(extend_seq_lens) != n
                    or plan.ragged_paged_page_table.ndim != 2
                    or plan.ragged_paged_page_table.shape[1] < max_seq_len):
                    raise ValueError("Invalid pooled-indexer request page mapping")
                table_copy = plan.ragged_paged_page_table[:, :max_seq_len].index_select(
                    0, req_pool_indices.to(torch.int64)
                )
                rows = torch.tensor(
                    [i for i, count in enumerate(extend_seq_lens) for _ in range(count)],
                    dtype=torch.int32, device=self.num_tokens.device,
                )
            copies.append((self.ragged_paged_page_table_row_index, rows))
        if topk_offsets is not None:
            if topk_offsets.numel() != n:
                raise ValueError("Inconsistent pooled-indexer top-k offsets")
            copies.append((self.topk_offsets, topk_offsets.reshape(-1)))
        self.layout = "paged" if plan.ragged_paged_page_table is not None else "ragged"
        if not self._owns_page_table:
            self.ragged_paged_page_table = plan.ragged_paged_page_table
        else:
            self.ragged_paged_page_table.zero_()
            self.ragged_paged_page_table[:table_copy.shape[0], :table_copy.shape[1]].copy_(table_copy)
        for target, source in copies:
            target.zero_()
            target[: source.numel()].copy_(source)
        if topk_offsets is None:
            self.topk_offsets.zero_()
        self.write_mask.zero_()
        self.write_mask[: plan.writes.req.numel()].fill_(True)
        self.num_tokens.fill_(n)
        self.num_history_rows.fill_(plan.ragged_total_k_rows)

    def write_cache(self, pool, buf, chunk_k, chunk_score, tail_k, tail_score, ape, round_scale=False):
        if pool.index_kpool != self.pool_size or pool.slots_per_page != BLOCK_SIZE_K:
            raise ValueError("Unsupported pooled cache layout")
        kpool_assemble_softmax_rotate_write_cache(
            pool=pool, buf=buf, chunk_k=chunk_k, chunk_score=chunk_score,
            tail_k=tail_k, tail_score=tail_score, req_pool_idx=self.writes.req,
            n_from_tail=self.writes.n_from_tail, chunk_src_start=self.writes.chunk_src,
            tail_logical_base=self.writes.tail_logical_base, ape=ape,
            loc=self.writes.write_loc, write_mask=self.write_mask, round_scale=round_scale,
        )
        scatter_kpool_tail_updates(
            pool=pool, chunk_k=chunk_k, chunk_score=chunk_score,
            tail_k=tail_k, tail_score=tail_score, req_pool_idx=self.tails.req,
            dst_logical_start=self.tails.dst_logical_start,
            chunk_src_start=self.tails.chunk_src, n_write=self.tails.n_write,
        )

    def gather(self, pool, buf):
        gather_index_k_scale_prefix_into(
            pool=pool, buf=buf, page_indices=self.ragged_concat_page_table,
            seq_len=self.max_history_rows, k_out=self.ragged_k_u8,
            scale_out=self.ragged_k_scale, active_rows=self.num_history_rows,
        )

    def write_and_gather(self, pool, buf, chunk_k, chunk_score, tail_k, tail_score, ape, round_scale=False):
        self.write_cache(pool, buf, chunk_k, chunk_score, tail_k, tail_score, ape, round_scale)
        self.gather(pool, buf)
