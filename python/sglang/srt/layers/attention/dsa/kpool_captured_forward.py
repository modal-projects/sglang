"""Pooled-indexer prefill with fixed-address metadata and bounded graph shapes."""

import torch
import triton
import triton.language as tl

from sglang.srt.model_executor.forward_context import get_token_to_kv_pool


@triton.jit
def _select_all_kpool_indices(
    seq_lens,
    page_table,
    page_rows,
    topk_offsets,
    output,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    PAGE_COL_STRIDE: tl.constexpr,
    PAGE_COLS: tl.constexpr,
    HAS_PAGE_TABLE: tl.constexpr,
    HAS_PAGE_ROWS: tl.constexpr,
    HAS_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    length = tl.load(seq_lens + row, mask=row < N, other=0)
    valid = (row < N) & (col < length) & (col < TOPK)
    value = col
    if HAS_PAGE_TABLE:
        page_row = row
        if HAS_PAGE_ROWS:
            page_row = tl.load(page_rows + row, mask=row < N, other=0)
        value = tl.load(
            page_table
            + page_row.to(tl.int64) * PAGE_STRIDE
            + col * PAGE_COL_STRIDE,
            mask=valid & (col < PAGE_COLS),
            other=-1,
        )
    elif HAS_OFFSETS:
        offset = tl.load(topk_offsets + row, mask=row < N, other=0)
        value += offset
    tl.store(
        output + row * WIDTH + col,
        tl.where(valid, value, -1),
        mask=col < WIDTH,
    )


def select_all_kpool_indices(
    seq_lens,
    topk,
    pool_size,
    out_rows,
    page_table=None,
    page_table_row_index=None,
    topk_offsets=None,
):
    """Preserve short-path token order, including its final unused tail columns."""
    assert seq_lens.ndim == 1 and out_rows >= seq_lens.numel()
    assert page_table is None or topk_offsets is None
    assert page_table_row_index is None or page_table is not None
    if page_table_row_index is not None:
        assert page_table_row_index.shape == seq_lens.shape
    if topk_offsets is not None:
        assert topk_offsets.shape == seq_lens.shape
    width = topk + pool_size - 1
    output = torch.empty(
        (out_rows, width), dtype=torch.int32, device=seq_lens.device
    )
    _select_all_kpool_indices[(out_rows,)](
        seq_lens,
        page_table if page_table is not None else seq_lens,
        page_table_row_index if page_table_row_index is not None else seq_lens,
        topk_offsets if topk_offsets is not None else seq_lens,
        output,
        N=seq_lens.numel(),
        TOPK=topk,
        WIDTH=width,
        PAGE_STRIDE=page_table.stride(0) if page_table is not None else 0,
        PAGE_COL_STRIDE=page_table.stride(1) if page_table is not None else 0,
        PAGE_COLS=page_table.shape[1] if page_table is not None else 0,
        HAS_PAGE_TABLE=page_table is not None,
        HAS_PAGE_ROWS=page_table_row_index is not None,
        HAS_OFFSETS=topk_offsets is not None,
        BLOCK=triton.next_power_of_2(width),
    )
    return output


def forward_captured_kpool(
    indexer,
    x,
    q_lora,
    positions,
    forward_batch,
    layer_id,
    return_indices,
    plan,
):
    import deep_gemm

    from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant

    n = plan.live_tokens
    out_rows = x.shape[0]
    if not 0 < n <= min(out_rows, q_lora.shape[0], positions.shape[0]):
        raise ValueError("Captured pooled-indexer inputs do not fit the logical shape")
    x = x[:n]
    q_lora = q_lora[:n]
    positions = positions[:n]

    if plan.skip_logits or not return_indices:
        key = indexer._get_k_bf16(x, positions)
    else:
        query, key, _ = indexer._get_q_k_bf16(
            q_lora,
            x,
            positions,
            enable_dual_stream=False,
            forward_batch=forward_batch,
        )
        q_fp8, q_scale = act_quant(query, indexer.block_size, indexer.scale_fmt)

    gate_score = indexer._project_compress_gate(x, indexer.prefill_stable_projection)
    pool = get_token_to_kv_pool()
    if hasattr(pool, "invalidate_index_buffer_for_layer"):
        pool.invalidate_index_buffer_for_layer(layer_id)
    buf = pool.get_index_k_with_scale_buffer(layer_id=layer_id)
    tail_k, tail_score = pool.get_compress_tail_buffers(layer_id)
    plan.write_cache(
        pool=pool,
        buf=buf,
        chunk_k=key,
        chunk_score=gate_score,
        tail_k=tail_k,
        tail_score=tail_score,
        ape=indexer.index_kpool_compress_ape,
        round_scale=indexer.scale_fmt is not None,
    )
    if not return_indices:
        return None

    page_table = None
    page_rows = None
    topk_offsets = None
    if plan.mapping_mode == "paged":
        page_table = plan.ragged_paged_page_table
        page_rows = plan.ragged_paged_page_table_row_index
        if page_table is None:
            raise ValueError("Captured pooled-indexer is missing its page table")
    elif plan.mapping_mode == "ragged":
        topk_offsets = plan.topk_offsets
    elif plan.mapping_mode is not None:
        raise ValueError("Unsupported captured pooled-indexer top-k mapping")

    if plan.skip_logits:
        return select_all_kpool_indices(
            seq_lens=plan.seq_lens_expanded,
            topk=indexer.index_topk,
            pool_size=indexer.index_kpool,
            out_rows=out_rows,
            page_table=page_table,
            page_table_row_index=page_rows,
            topk_offsets=topk_offsets,
        )

    weights = indexer._get_logits_head_gate(
        x, q_scale, stable_projection=indexer.prefill_stable_projection
    )
    plan.gather(pool, indexer._get_index_k_read_buffer(pool, layer_id))
    logits = deep_gemm.fp8_mqa_logits(
        q_fp8.contiguous(),
        (
            plan.ragged_k_u8.view(torch.float8_e4m3fn),
            plan.ragged_k_scale,
        ),
        weights.squeeze(-1).contiguous(),
        plan.ragged_q_ks,
        plan.ragged_q_ke,
        clean_logits=False,
        max_seqlen_k=plan.max_logits_rows,
    )
    return indexer._topk_from_kpool_logits(
        logits,
        plan.pooled_seq_lens_expanded,
        seq_lens=plan.seq_lens_expanded,
        page_table=page_table,
        topk_offsets=topk_offsets,
        out_rows=out_rows,
        page_table_row_index=page_rows,
    )
