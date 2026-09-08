import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsa.transform_index import (
    prepare_trtllm_nope_sparse_metadata,
)


class DSAPrefillGraphMetadata:
    def __init__(self, max_tokens, device):
        self.max_tokens = max_tokens
        self.seq_lens = torch.ones(max_tokens, dtype=torch.int32, device=device)
        self.cache_locs = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self.num_tokens = torch.zeros(1, dtype=torch.int32, device=device)

    def update(self, forward_batch, metadata):
        tokens = sum(forward_batch.extend_seq_lens_cpu)
        if not 0 < tokens <= self.max_tokens:
            raise ValueError(f"Invalid captured DSA prefill size: {tokens}")
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
    pool = backend.token_to_kv_pool
    pool.set_mla_kv_buffer(
        layer,
        metadata.cache_locs[:tokens],
        k.squeeze(1).to(torch.float8_e4m3fn),
        k_rope.squeeze(1).to(torch.float8_e4m3fn),
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
    return trtllm_prefill_graph_attention(
        q.view(tokens, layer.tp_q_head_num, layer.v_head_dim),
        kv,
        topk_indices,
        metadata.seq_lens[:tokens],
        metadata.num_tokens,
        backend.workspace_buffer,
        backend._multi_ctas_kv_counter_buffer,
        layer.scaling * (k_scale if k_scale is not None else 1.0),
        backend.qk_nope_head_dim,
        backend.dsa_index_topk + backend.dsa_index_kpool,
    )
