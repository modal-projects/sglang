import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsa.transform_index import (
    prepare_trtllm_nope_sparse_metadata,
)


class DSAPrefillGraphMetadata:
    def __init__(self, max_tokens, device, capture_sizes, num_heads):
        self.max_tokens = max_tokens
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        previous = 0
        self.capture_sizes = set()
        for size in sorted(capture_sizes):
            if num_heads in (8, 16, 32) and previous >= sm_count // 2:
                self.capture_sizes.add(size)
            previous = size
        self.seq_lens = torch.ones(max_tokens, dtype=torch.int32, device=device)
        self.cache_locs = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self.num_tokens = torch.zeros(1, dtype=torch.int32, device=device)
        self.variant = None

    def can_capture(self, num_tokens):
        return self.variant is not None or num_tokens in self.capture_sizes

    def update(self, forward_batch, metadata):
        tokens = sum(forward_batch.extend_seq_lens_cpu)
        if not 0 < tokens <= self.max_tokens:
            raise ValueError(f"Invalid captured DSA prefill size: {tokens}")
        self.variant = forward_batch.dsa_prefill_graph_variant
        if self.variant is not None and self.variant[0] != tokens:
            raise ValueError("DSA capture variant does not match its live token count")
        self.num_tokens.fill_(tokens)
        self.seq_lens.fill_(1)
        self.seq_lens[:tokens].copy_(metadata.dsa_cache_seqlens_int32[:tokens])
        self.cache_locs.zero_()
        self.cache_locs[:tokens].copy_(forward_batch.out_cache_loc[:tokens])


@triton.jit
def _zero_padded_output(
    Output, NumTokens, SIZE: tl.constexpr, DIM: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    real = tl.load(NumTokens)
    tl.store(Output + offsets, 0, (offsets < SIZE) & (offsets >= real * DIM))


def trtllm_prefill_graph_attention(
    q,
    kv_cache,
    topk_indices,
    seq_lens,
    num_tokens,
    workspace,
    counter_buffer,
    scaling,
    qk_nope_head_dim,
    max_seq_len,
):
    import flashinfer.decode

    tokens, heads, rank = q.shape
    topk_width = triton.cdiv(topk_indices.shape[1], 4) * 4
    page_table = torch.full(
        (tokens, topk_width), -1, device=q.device, dtype=torch.int32
    )
    page_table[:, : topk_indices.shape[1]].copy_(topk_indices)
    topk_lens = prepare_trtllm_nope_sparse_metadata(page_table)
    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=q.to(torch.float8_e4m3fn).contiguous().view(tokens, 1, heads, rank),
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=rank,
        qk_rope_head_dim=0,
        block_tables=page_table.unsqueeze(1),
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        sparse_mla_top_k=topk_width,
        bmm1_scale=scaling,
        backend="trtllm-gen",
        sparse_mla_top_k_lens=topk_lens,
        multi_ctas_kv_counter_buffer=counter_buffer,
    )
    _zero_padded_output[(triton.cdiv(output.numel(), 1024),)](
        output, num_tokens, output.numel(), heads * rank, 1024
    )
    return output


def dsa_prefill_graph_forward(backend, layer, q, k, k_rope, topk_indices, metadata):
    from sglang.srt.layers.attention.trtllm_mla_backend import (
        grow_multi_ctas_kv_counter_buffer_if_needed,
    )

    tokens = q.shape[0]
    live_tokens, max_seq_len = metadata.variant or (
        tokens,
        backend.dsa_index_topk + backend.dsa_index_kpool,
    )
    pool = backend.token_to_kv_pool
    pool.set_mla_kv_buffer(
        layer,
        metadata.cache_locs[:live_tokens],
        k[:live_tokens].squeeze(1).to(torch.float8_e4m3fn),
        k_rope[:live_tokens].squeeze(1).to(torch.float8_e4m3fn),
    )
    kv = pool.get_key_buffer(layer.layer_id).view(
        -1, 1, backend.real_page_size, backend.kv_cache_dim
    )
    backend._multi_ctas_kv_counter_buffer = grow_multi_ctas_kv_counter_buffer_if_needed(
        backend._multi_ctas_kv_counter_buffer,
        torch.device(backend.device),
        backend.num_q_heads,
        tokens,
    )
    k_scale = getattr(layer, "k_scale_float", None)
    result = trtllm_prefill_graph_attention(
        q.view(tokens, layer.tp_q_head_num, layer.v_head_dim)[:live_tokens],
        kv,
        topk_indices[:live_tokens],
        metadata.seq_lens[:live_tokens],
        metadata.num_tokens,
        backend.workspace_buffer,
        backend._multi_ctas_kv_counter_buffer,
        layer.scaling * (k_scale if k_scale is not None else 1.0),
        backend.qk_nope_head_dim,
        max_seq_len,
    )
    if live_tokens == tokens:
        return result
    output = result.new_zeros((tokens, *result.shape[1:]))
    output[:live_tokens].copy_(result)
    return output
