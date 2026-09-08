import itertools

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn


def chunk_capacity(tokens, requests, size):
    return (tokens + (size - 1) * min(tokens, requests)) // size


class KDAPrefillGraphMetadata:
    def __init__(self, max_tokens, max_requests, device):
        self.max_tokens = max_tokens
        self.max_requests = max_requests
        self.device = device
        self.cu_seqlens = torch.zeros(
            max_requests + 2, dtype=torch.int32, device=device
        )
        self.cache_indices = torch.full(
            (max_requests + 1,), -1, dtype=torch.int64, device=device
        )
        self.has_initial_state = torch.zeros(
            max_requests + 1, dtype=torch.bool, device=device
        )
        self.chunk_offsets = torch.zeros(
            max_requests + 2, dtype=torch.int32, device=device
        )
        self.chunk_indices = torch.empty(
            (chunk_capacity(max_tokens, max_requests, 64), 2),
            dtype=torch.int32,
            device=device,
        )
        self.conv_indices = torch.empty(
            (chunk_capacity(max_tokens, max_requests, 8), 2),
            dtype=torch.int32,
            device=device,
        )
        self.track_conv_indices = torch.zeros(
            (max_requests, 3), dtype=torch.int64, device=device
        )
        self.track_conv_dst = torch.full(
            (max_requests,), -1, dtype=torch.int64, device=device
        )
        self.track_h_src = torch.zeros(max_requests, dtype=torch.int64, device=device)
        self.track_h_dst = torch.full(
            (max_requests,), -1, dtype=torch.int64, device=device
        )
        self.track_final_src = torch.zeros(
            max_requests, dtype=torch.int64, device=device
        )
        self.track_final_dst = torch.full(
            (max_requests,), -1, dtype=torch.int64, device=device
        )

    def update(self, lengths, cache_indices, prefix_lens, metadata):
        lengths = [int(value) for value in lengths]
        total = sum(lengths)
        if (
            len(lengths) > self.max_requests
            or total > self.max_tokens
            or any(n <= 0 for n in lengths)
        ):
            raise ValueError(f"Unsupported KDA graph layout: {lengths}")
        offsets = list(itertools.accumulate(lengths, initial=0))
        self.cu_seqlens.fill_(total)
        self.cu_seqlens[: len(offsets)].copy_(torch.tensor(offsets, dtype=torch.int32))
        self.cache_indices.fill_(-1)
        self.cache_indices[: len(lengths)].copy_(cache_indices)
        self.has_initial_state.zero_()
        self.has_initial_state[: len(lengths)].copy_(prefix_lens > 0)
        for size, target in ((64, self.chunk_indices), (8, self.conv_indices)):
            counts = [(n + size - 1) // size for n in lengths]
            pairs = [
                (seq, chunk)
                for seq, count in enumerate(counts)
                for chunk in range(count)
            ]
            target[:, 0].fill_(self.max_requests)
            target[:, 1].zero_()
            target[: len(pairs)].copy_(torch.tensor(pairs, dtype=torch.int32))
            if size == 64:
                offsets = list(itertools.accumulate(counts, initial=0))
                self.chunk_offsets.fill_(len(pairs))
                self.chunk_offsets[: len(offsets)].copy_(
                    torch.tensor(offsets, dtype=torch.int32)
                )
        for target, name in (
            (self.track_conv_dst, "conv_states_mask_indices"),
            (self.track_conv_indices, "track_conv_indices"),
            (self.track_h_src, "track_ssm_h_src"),
            (self.track_h_dst, "track_ssm_h_dst"),
            (self.track_final_src, "track_ssm_final_src"),
            (self.track_final_dst, "track_ssm_final_dst"),
        ):
            target.fill_(-1 if target.ndim == 1 else 0)
            source = (
                getattr(metadata, name, None) if metadata.has_mamba_track_mask else None
            )
            if source is not None:
                target[: len(source)].copy_(source)


@triton.jit
def _snapshot_conv(
    X,
    Indices,
    Dst,
    Pool,
    DIM: tl.constexpr,
    STRIDE: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request, block = tl.program_id(0), tl.program_id(1)
    dst = tl.load(Dst + request).to(tl.int64)
    if dst < 0:
        return
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < WINDOW * DIM
    rows = tl.load(Indices + request * WINDOW + offsets // DIM, mask, 0).to(tl.int64)
    values = tl.load(X + rows * DIM + offsets % DIM, mask, 0)
    tl.store(Pool + dst * STRIDE + offsets, values, mask)


@triton.jit
def _snapshot_state(
    Source,
    Src,
    Dst,
    Pool,
    SIZE: tl.constexpr,
    SRC_STRIDE: tl.constexpr,
    DST_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request, block = tl.program_id(0), tl.program_id(1)
    dst = tl.load(Dst + request).to(tl.int64)
    if dst < 0:
        return
    src = tl.load(Src + request).to(tl.int64)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(Source + src * SRC_STRIDE + offsets, offsets < SIZE, 0)
    tl.store(Pool + dst * DST_STRIDE + offsets, values, offsets < SIZE)


@triton.jit
def _zero_padding(
    Output,
    Cu,
    END: tl.constexpr,
    TOKENS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    real = tl.load(Cu + END)
    tl.store(Output + offsets, 0, (offsets < TOKENS * DIM) & (offsets >= real * DIM))


def kda_prefill_graph_forward(layer, mixed_qkv, a, b, conv_pool, ssm_states, metadata):
    tokens, dim = mixed_qkv.shape
    requests = metadata.max_requests
    _snapshot_conv[(requests, triton.cdiv(3 * dim, 1024))](
        mixed_qkv,
        metadata.track_conv_indices,
        metadata.track_conv_dst,
        conv_pool,
        dim,
        conv_pool.stride(0),
        3,
        1024,
    )
    conv_indices = metadata.conv_indices[: chunk_capacity(tokens, requests, 8)]
    chunk_indices = metadata.chunk_indices[: chunk_capacity(tokens, requests, 64)]
    qkv = causal_conv1d_fn(
        mixed_qkv.transpose(0, 1),
        layer.conv_weights,
        layer.bias,
        conv_states=conv_pool.transpose(-1, -2),
        query_start_loc=metadata.cu_seqlens,
        seq_lens_cpu=None,
        cache_indices=metadata.cache_indices,
        has_initial_state=metadata.has_initial_state,
        chunk_indices=conv_indices,
        activation="silu",
    ).transpose(0, 1)
    q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
    q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
    k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
    v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)
    gate_was_flat = a.ndim == 3
    if gate_was_flat:
        a = a.unflatten(-1, (-1, layer.head_k_dim))
    output, h = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=a,
        beta=b,
        initial_state=ssm_states,
        initial_state_indices=metadata.cache_indices,
        cu_seqlens=metadata.cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=metadata.chunk_offsets,
        use_qk_l2norm_in_kernel=True,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        lower_bound=layer.lower_bound,
        beta_is_raw=gate_was_flat,
        output_intermediate_states=True,
    )
    state_size = ssm_states[0].numel()
    grid = (requests, triton.cdiv(state_size, 1024))
    _snapshot_state[grid](
        h,
        metadata.track_h_src,
        metadata.track_h_dst,
        ssm_states,
        state_size,
        h.stride(1),
        ssm_states.stride(0),
        1024,
    )
    _snapshot_state[grid](
        ssm_states,
        metadata.track_final_src,
        metadata.track_final_dst,
        ssm_states,
        state_size,
        ssm_states.stride(0),
        ssm_states.stride(0),
        1024,
    )
    _zero_padding[(triton.cdiv(output.numel(), 1024),)](
        output, metadata.cu_seqlens, requests + 1, tokens, layer.v_dim, 1024
    )
    return output
