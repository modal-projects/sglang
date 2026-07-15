"""VENDORED BASELINE — verbatim copy of python/sglang/tml/kernels/sconv.py at
commit c262556c2 (parent of the Inkling kernel-fusion commits). Provides the OLD
Helion causal_conv1d for old-vs-new microbenchmarks. DO NOT EDIT by hand.
"""
from functools import partial
from typing import TypedDict

import helion
import helion.language as hl
import torch
import triton
import triton.language as tl
from sglang.srt.kernels.helion_utils import helion_aot_autotune

PAD_SLOT_ID = -1


class SconvDecodeMetadata(TypedDict):
    cache_mask: torch.Tensor
    safe_idx: torch.Tensor
    cu: torch.Tensor
    si: torch.Tensor


class SconvExtendMetadata(TypedDict):
    cache_mask: torch.Tensor
    safe_idx: torch.Tensor
    cu: torch.Tensor
    si: torch.Tensor


CHUNK_SIZE = 64

_D_SIZES = [4096, 8192]


def _sconv_key(*args: object):
    """All sconv kernels share the same key structure: find the [D, W] weight tensor."""
    weight: torch.Tensor | None = None
    for a in args:
        if isinstance(a, torch.Tensor) and a.ndim == 2:
            weight = a
            break
    assert weight is not None
    x = args[0]
    assert isinstance(x, torch.Tensor)
    return weight.shape[0], (x.dtype, weight.dtype), (weight.shape[1],)


def _sconv_with_prefix_inputs(*tensor_specs: object, d_sizes: list[int]):
    """Generate autotuning inputs for the with-prefix kernel."""
    W, T = 4, 8192
    with torch.device("cuda"):
        result = []
        for D in d_sizes:
            row = []
            for spec in tensor_specs:
                if spec == "btd":
                    row.append(torch.randn(1, T, D, dtype=torch.bfloat16))
                elif spec == "prefix":
                    row.append(torch.randn(1, W - 1, D, dtype=torch.bfloat16))
                elif spec == "dw":
                    row.append(torch.randn(D, W, dtype=torch.bfloat16))
                elif spec == "cu_seqlens":
                    row.append(torch.tensor([0, T], dtype=torch.int64))
                elif spec == "seq_idx":
                    row.append(torch.zeros(T, dtype=torch.int32))
                else:
                    row.append(spec)
            result.append(tuple(row))
        return result


@helion_aot_autotune(
    "causal_conv1d_fwd_with_prefix",
    kernel_key=_sconv_key,
    primary_inputs=partial(
        _sconv_with_prefix_inputs,
        "btd",
        "prefix",
        "dw",
        "silu",
        True,
        "cu_seqlens",
        "seq_idx",
        d_sizes=_D_SIZES,
    ),
)
@helion.kernel(static_shapes=False, ignore_warnings=[helion.exc.TensorOperationInWrapper])
def _helion_causal_conv1d_fwd_with_prefix_kernel(
    x: torch.Tensor,  # [1, T, D]
    prefix: torch.Tensor,  # [num_seqs, W-1, D]
    weight: torch.Tensor,  # [D, W]
    activation: str,
    use_residual: bool,
    cu_seqlens: torch.Tensor,  # [num_seqs + 1]
    seq_idx: torch.Tensor,  # [T]
) -> torch.Tensor:
    B, T, D = x.shape
    W = hl.specialize(weight.shape[1])
    y = torch.empty_like(x)

    for tile_b, tile_t, tile_d in hl.tile([B, T, D], block_size=[1, None, None]):
        acc = hl.zeros([tile_b, tile_t, tile_d], dtype=torch.float32)
        b_idx = tile_b.index
        t_idx = tile_t.index
        d_idx = tile_d.index

        si = hl.load(seq_idx, [t_idx])
        bos = hl.load(cu_seqlens, [si.to(torch.int64)])

        for iw in hl.static_range(W):
            shifted_t = t_idx - (W - 1) + iw

            # Load from x for positions within the sequence
            in_x = (shifted_t >= bos) & (shifted_t < T)
            x_val = hl.load(
                x,
                [b_idx[:, None, None], shifted_t[None, :, None], d_idx[None, None, :]],
                extra_mask=in_x[None, :, None],
            )

            # Load from prefix for positions before sequence start
            prefix_pos = shifted_t - bos + (W - 1)
            in_prefix = (shifted_t < bos) & (prefix_pos >= 0)
            p_val = hl.load(
                prefix,
                [
                    si[None, :, None].to(torch.int64),
                    prefix_pos[None, :, None],
                    d_idx[None, None, :],
                ],
                extra_mask=in_prefix[None, :, None],
            )

            w_val = hl.load(weight, [d_idx, iw])
            combined = (x_val + p_val).to(torch.float32) * w_val.to(torch.float32)[None, None, :]
            acc = acc + combined

        if activation == "silu" or activation == "swish":
            acc = acc * torch.sigmoid(acc)

        if use_residual:
            x_res = x[tile_b, tile_t, tile_d].to(torch.float32)
            acc = acc + x_res

        y[tile_b, tile_t, tile_d] = acc.to(y.dtype)

    return y


# todo(horace): Shift this to be precomputed data
def _seq_idx_from_cu_seqlens(cu_seqlens: torch.Tensor, T: int) -> torch.Tensor:
    """Compute seq_idx from cu_seqlens: for each position, which sequence it belongs to."""
    t = torch.arange(T, dtype=torch.int64, device=cu_seqlens.device)
    # Clamp to [0, num_seqs-1] to prevent OOB when cu_seqlens doesn't span all T
    # tokens (e.g. during CUDA graph capture warmup with dummy zero-length sequences).
    num_seqs = cu_seqlens.shape[0] - 1
    return (
        (torch.searchsorted(cu_seqlens, t, side="right") - 1)
        .clamp(max=num_seqs - 1)
        .to(torch.int32)
    )


def precompute_helion_decode_metadata(
    B: int,
    W: int,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> SconvDecodeMetadata:
    """Precompute metadata for the helion decode path. Call once, reuse across layers."""
    device = cache_indices.device
    valid = cache_indices != PAD_SLOT_ID
    cache_mask = (has_initial_state & valid)[:, None, None]  # [B, 1, 1]
    safe_idx = cache_indices.clamp(min=0).long()
    # Each sequence has exactly 1 token in the packed [1, B, D] layout
    cu = torch.arange(B + 1, dtype=torch.int64, device=device)
    si = torch.arange(B, dtype=torch.int32, device=device)
    return SconvDecodeMetadata(
        cache_mask=cache_mask,
        safe_idx=safe_idx,
        cu=cu,
        si=si,
    )


def precompute_helion_extend_metadata(
    B: int,
    T: int,
    W: int,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> SconvExtendMetadata:
    """Precompute metadata for the helion extend path. Call once, reuse across layers."""
    device = cache_indices.device

    valid = cache_indices != PAD_SLOT_ID
    cache_mask = (has_initial_state & valid)[:, None, None]  # [B, 1, 1]
    safe_idx = cache_indices.clamp(min=0).long()

    cu = query_start_loc.to(torch.int64)
    si = _seq_idx_from_cu_seqlens(cu, T)

    return SconvExtendMetadata(
        cache_mask=cache_mask,
        safe_idx=safe_idx,
        cu=cu,
        si=si,
    )


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_mask: torch.Tensor,
    safe_idx: torch.Tensor,
    cu: torch.Tensor,
    si: torch.Tensor,
    activation: str | None = None,
    use_residual: bool = True,
    is_decode: bool = False,
) -> torch.Tensor:
    """Inference sconv with prefix loaded directly from cache.

    Metadata args (cache_mask, safe_idx, cu, si) should be precomputed once
    per forward pass via precompute_helion_{decode,extend}_metadata and reused
    across layers.
    """
    assert x.is_contiguous()
    if activation == "swish":
        activation = "silu"

    T = x.shape[0]

    if T == 0:
        return torch.empty_like(x)

    # Build [B, W-1, D] prefix from cache (zeros for no-cache sequences).
    # During decode every real sequence has a valid cache; only CUDA-graph
    # padding slots have cache_mask=False.  Their outputs are never consumed
    # and the Helion kernel processes sequences independently (via cu/si),
    # so stale prefix data cannot contaminate valid sequences.
    if is_decode:
        prefix = sconv_cache[safe_idx]
    else:
        prefix = sconv_cache[safe_idx] * cache_mask  # [B, W-1, D]

    # Pack x as [1, T, D] — unified for both decode (T=B) and extend (T=total_tokens).
    x_packed = x.unsqueeze(0)

    y = _helion_causal_conv1d_fwd_with_prefix_kernel(
        x_packed, prefix, weight, activation or "", use_residual, cu, si
    )
    return y.squeeze(0)


def _update_sconv_cache_helion_key(
    x,
    sconv_cache,
    cache_indices,
    has_initial_state,
    query_start_loc,
):
    """(numeric=D, hash=dtype, exact=W-1)"""
    return x.shape[-1], (x.dtype,), (sconv_cache.shape[1],)


def _update_sconv_cache_helion_inputs(d_sizes, B=4096):
    W_minus_1 = 3
    max_slots = max(B * 2, 4096)
    with torch.device("cuda"):
        return [
            (
                torch.randn(B, D, dtype=torch.bfloat16),  # x [T=B, D]
                torch.randn(max_slots, W_minus_1, D, dtype=torch.bfloat16),  # sconv_cache
                torch.arange(B, dtype=torch.int32),  # cache_indices
                torch.ones(B, dtype=torch.bool),  # has_initial_state
                torch.arange(0, B + 1, dtype=torch.int32),  # query_start_loc
            )
            for D in d_sizes
        ]


_UPDATE_SCONV_CACHE_PRIMARY_D = [384, 2304, 4032, 6144]
_UPDATE_SCONV_CACHE_SECONDARY_D = [i * 96 for i in range(8, 65)]


@helion_aot_autotune(
    "update_sconv_cache",
    kernel_key=_update_sconv_cache_helion_key,
    primary_inputs=partial(
        _update_sconv_cache_helion_inputs, d_sizes=_UPDATE_SCONV_CACHE_PRIMARY_D
    ),
    secondary_inputs=partial(
        _update_sconv_cache_helion_inputs, d_sizes=_UPDATE_SCONV_CACHE_SECONDARY_D
    ),
)
@helion.kernel(
    static_shapes=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _update_sconv_cache_helion_kernel(
    x: torch.Tensor,  # [T, D]
    sconv_cache: torch.Tensor,  # [max_slots, W-1, D]
    cache_indices: torch.Tensor,  # [B] int32
    has_initial_state: torch.Tensor,  # [B] bool
    query_start_loc: torch.Tensor,  # [B+1] int32
) -> torch.Tensor:
    """General update_sconv_cache: handles both decode (ql=1) and extend (ql>=1).

    For each output cache position w, the new value is either:
    - From x (when query_len >= W_minus_1 - w): x[end - W_minus_1 + w]
    - From shifted old cache (when query_len < W_minus_1 - w and has_state): old_cache[w + query_len]
    - Zero (when query_len < W_minus_1 - w and not has_state)

    Uses nested static_range to select the shift source column at runtime
    without data-dependent subscripts on the W dimension.
    """
    B = cache_indices.shape[0]
    D = x.shape[-1]
    W_minus_1 = hl.specialize(sconv_cache.shape[1])

    for tile_b, tile_d in hl.tile([B, D]):
        ci = torch.clamp(cache_indices[tile_b], min=0).to(torch.int64)
        end = query_start_loc[tile_b + 1].to(torch.int64)
        start = query_start_loc[tile_b].to(torch.int64)
        query_len = end - start
        has_state = has_initial_state[tile_b].to(x.dtype)
        valid = (cache_indices[tile_b] != PAD_SLOT_ID) & (query_len > 0)

        for w in hl.static_range(W_minus_1):
            old_val = sconv_cache[ci, w, tile_d]

            # Does this position get a token from x?
            gets_x = query_len >= (W_minus_1 - w)

            # x token value (clamped for safety when gets_x is False)
            x_idx = torch.clamp(end - W_minus_1 + w, min=0)
            x_val = x[x_idx, tile_d]

            # Shifted cache value: select old_cache[w + query_len] via static loop
            shift_val = old_val * 0  # zeros with correct shape
            for src_w in hl.static_range(W_minus_1):
                match = query_len == (src_w - w)
                src_val = sconv_cache[ci, src_w, tile_d]
                shift_val = torch.where(match[:, None], src_val, shift_val)
            shift_val = shift_val * has_state[:, None]

            new_val = torch.where(gets_x[:, None], x_val, shift_val)
            final_val = torch.where(valid[:, None], new_val, old_val)
            sconv_cache[ci, w, tile_d] = final_val

    return sconv_cache


def update_sconv_cache(
    x: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    _update_sconv_cache_helion_kernel(
        x, sconv_cache, cache_indices, has_initial_state, query_start_loc
    )


# ---------------------------------------------------------------------------
# Fused decode kernel: causal_conv1d + update_sconv_cache in one launch
# ---------------------------------------------------------------------------


@triton.jit
def _fused_causal_conv1d_update_decode_kernel(
    x,  # [T, D]
    sconv_cache,  # [max_slots, W-1, D]
    cache_indices,  # [B] int32
    cache_mask,  # [B] bool (cache_indices != PAD_SLOT_ID)
    weight,  # [D, W]
    y,  # [T, D]
    stride_x_t,
    stride_x_d,
    stride_cache_slot,
    stride_cache_d,
    stride_cache_w,
    stride_weight_d,
    stride_weight_w,
    T,
    D,
    USE_SILU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    W: tl.constexpr,
):
    """Fused depthwise causal conv1d + cache shift-update for decode.

    Decode invariant: each token t belongs to sequence t, with bos=t.
      - iw = 0..W-2: always reads from sconv_cache (the conv state history)
      - iw = W-1:    always reads from x (the current token)

    General for any W. Uses tl.static_range for both conv and update.
    Cache values are re-read for the update shift (trades W-1 extra loads
    for generality and lower register pressure vs manual unroll).

    The track-copy is NOT fused because track_indices[t] can alias
    cache_indices[s] for a different token, causing a cross-block race.
    """
    t_off = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < D
    td_mask = t_mask[:, None] & d_mask[None, :]

    ci = tl.load(cache_indices + t_off, mask=t_mask, other=-1)
    safe_idx = tl.maximum(ci, 0).to(tl.int64)
    valid = ci != -1  # PAD_SLOT_ID = -1 (can't reference Python global in @jit)
    cm = tl.load(cache_mask + t_off, mask=t_mask, other=0).to(sconv_cache.dtype.element_ty)

    cache_base = (
        sconv_cache + safe_idx[:, None] * stride_cache_slot + d_off[None, :] * stride_cache_d
    )
    weight_base = weight + d_off * stride_weight_d

    # ---- CONV ----
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

    # Cache taps: iw = 0..W-2
    for iw in tl.static_range(W - 1):
        pv = tl.load(
            cache_base + iw * stride_cache_w, mask=td_mask, other=0, eviction_policy="evict_last"
        )
        w = tl.load(weight_base + iw * stride_weight_w, mask=d_mask, other=0).to(tl.float32)
        acc += (pv * cm[:, None]).to(tl.float32) * w[None, :]

    # Current token: iw = W-1
    xv = tl.load(
        x + t_off[:, None] * stride_x_t + d_off[None, :] * stride_x_d, mask=td_mask, other=0
    )
    xv_f32 = xv.to(tl.float32)
    w_last = tl.load(weight_base + (W - 1) * stride_weight_w, mask=d_mask, other=0).to(tl.float32)
    acc += xv_f32 * w_last[None, :]

    if USE_SILU:
        acc = acc * tl.sigmoid(acc)

    if USE_RESIDUAL:
        acc += xv_f32

    tl.store(
        y + t_off[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
        acc.to(xv.dtype),
        mask=td_mask,
    )

    # ---- UPDATE: shift cache left, write new token ----
    # cache[slot, w, d] = cache[slot, w+1, d] * cm  for w = 0..W-3
    # cache[slot, W-2, d] = xv
    write_mask = td_mask & valid[:, None]
    for iw in tl.static_range(W - 2):
        # Re-read cache[slot, d, iw+1] for the shift
        shifted = tl.load(cache_base + (iw + 1) * stride_cache_w, mask=td_mask, other=0)
        tl.store(cache_base + iw * stride_cache_w, shifted * cm[:, None], mask=write_mask)
    # Last position gets the new token
    tl.store(cache_base + (W - 2) * stride_cache_w, xv, mask=write_mask)


def _select_fused_decode_config(T: int, D: int) -> tuple[int, int, int, int]:
    """Select (BLOCK_T, BLOCK_D, num_warps, num_stages) for the fused decode kernel.

    Heuristic: keep BLOCK_T small (1-2) for decode since T=B is moderate.
    Scale BLOCK_D so that grid has enough blocks to fill the GPU.
    """
    if T <= 2048:
        block_t = 2
    else:
        # Round down to power of 2; Triton requires tl.arange size to be power of 2.
        raw = min(T // 1024, 8)
        block_t = 1 << (raw.bit_length() - 1)

    target_blocks = 1024
    t_blocks = max(T // block_t, 1)
    needed_d_blocks = max(target_blocks // t_blocks, 1)
    block_d = max(D // needed_d_blocks, 64)
    block_d = 1 << max(min((block_d).bit_length() - 1, 9), 6)

    tile_elems = block_t * block_d
    if tile_elems <= 128:
        num_warps = 1
    elif tile_elems <= 512:
        num_warps = 2
    else:
        num_warps = 4

    return block_t, block_d, num_warps, 3


def fused_causal_conv1d_update_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    activation: str | None = None,
    use_residual: bool = True,
) -> torch.Tensor:
    """Fused causal_conv1d + update_sconv_cache for decode.

    Replaces the sequence: prefix construction -> conv -> cache update
    with a single kernel launch. The track-copy (copy_if_needed) must
    be called separately after this function.
    """
    T, D = x.shape
    W = weight.shape[1]
    y = torch.empty_like(x)
    cm = cache_mask.view(-1)
    use_silu = activation in ("silu", "swish")

    bt, bd, nw, ns = _select_fused_decode_config(T, D)

    grid = (triton.cdiv(T, bt), triton.cdiv(D, bd))
    _fused_causal_conv1d_update_decode_kernel[grid](
        x,
        sconv_cache,
        cache_indices,
        cm,
        weight,
        y,
        x.stride(0),
        x.stride(1),
        sconv_cache.stride(0),
        sconv_cache.stride(2),  # stride_cache_d
        sconv_cache.stride(1),  # stride_cache_w
        weight.stride(0),
        weight.stride(1),
        T,
        D,
        USE_SILU=use_silu,
        USE_RESIDUAL=use_residual,
        BLOCK_T=bt,
        BLOCK_D=bd,
        W=W,
        num_warps=nw,
        num_stages=ns,
    )
    return y


@triton.jit
def _save_intermediate_conv_windows_kernel(
    sconv_cache_ptr,  # [cache_size, W-1, D]
    hidden_states_ptr,  # [B, T_max, D]
    cache_indices_ptr,  # [B] int32
    out_ptr,  # [max_bs, T, W-1, D]
    cache_slot_stride,
    cache_pos_stride,
    hidden_b_stride,
    hidden_t_stride,
    out_b_stride,
    out_t_stride,
    out_w_stride,
    D,
    W_MINUS_1: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    cache_idx = tl.load(cache_indices_ptr + pid_b).to(tl.int64)

    # PAD_SLOT_ID guard: skip padded batch slots. Mirrors
    # fused_mamba_state_scatter_with_mask's early-exit. Avoids the OOB
    # negative-stride load that would result from `cache_idx == PAD_SLOT_ID`.
    if cache_idx == PAD_SLOT_ID:
        return

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    for w in tl.static_range(W_MINUS_1):
        position = pid_t + 1 + w
        if position < W_MINUS_1:
            src_offset = cache_idx * cache_slot_stride + position * cache_pos_stride + d_off
            val = tl.load(sconv_cache_ptr + src_offset, mask=d_mask, other=0.0)
        else:
            t_in_hidden = position - W_MINUS_1
            src_offset = (
                pid_b.to(tl.int64) * hidden_b_stride
                + t_in_hidden.to(tl.int64) * hidden_t_stride
                + d_off
            )
            val = tl.load(hidden_states_ptr + src_offset, mask=d_mask, other=0.0)

        dst_offset = (
            pid_b.to(tl.int64) * out_b_stride
            + pid_t.to(tl.int64) * out_t_stride
            + w * out_w_stride
            + d_off
        )
        tl.store(out_ptr + dst_offset, val, mask=d_mask)


def save_intermediate_conv_windows(
    sconv_cache: torch.Tensor,  # [cache_size, W-1, D]
    hidden_states: torch.Tensor,  # [B, T_max, D] or [B*T_max, D]
    cache_indices: torch.Tensor,  # [B], int32 or int64
    intermediate_out: torch.Tensor,  # [max_bs, T, W-1, D]
    batch_size: int,
    draft_token_num: int,
) -> None:
    """Fused unfold-and-write into intermediate_out[:batch_size].

    Equivalent to:
        initial = sconv_cache[cache_indices[:batch_size]]
        padded  = torch.cat([initial, hidden_states[:batch_size, :draft_token_num]], dim=1)
        windows = padded.unfold(1, W-1, 1)[:, 1:draft_token_num+1].transpose(-2,-1).contiguous()
        intermediate_out[:batch_size] = windows
    """
    if batch_size == 0 or draft_token_num == 0:
        return

    W_minus_1, D = sconv_cache.shape[1], sconv_cache.shape[2]
    if W_minus_1 == 0:
        return

    if hidden_states.dim() == 2:
        hidden_states = hidden_states.view(batch_size, -1, hidden_states.shape[-1])
    assert hidden_states.dim() == 3, f"unexpected hidden_states shape {hidden_states.shape}"
    assert hidden_states.shape[0] == batch_size
    assert hidden_states.shape[2] == D
    assert intermediate_out.shape[1] == draft_token_num
    assert intermediate_out.shape[2] == W_minus_1
    assert intermediate_out.shape[3] == D

    # kernel assumption
    assert sconv_cache.stride(-1) == 1, "sconv_cache must be D-contiguous"
    assert hidden_states.stride(-1) == 1, "hidden_states must be D-contiguous"
    assert intermediate_out.stride(-1) == 1, "intermediate_out must be D-contiguous"

    cache_indices = cache_indices[:batch_size].to(torch.int32).contiguous()

    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    grid = (batch_size, draft_token_num, triton.cdiv(D, BLOCK_D))

    _save_intermediate_conv_windows_kernel[grid](
        sconv_cache,
        hidden_states,
        cache_indices,
        intermediate_out,
        sconv_cache.stride(0),
        sconv_cache.stride(1),
        hidden_states.stride(0),
        hidden_states.stride(1),
        intermediate_out.stride(0),
        intermediate_out.stride(1),
        intermediate_out.stride(2),
        D,
        W_MINUS_1=W_minus_1,
        BLOCK_D=BLOCK_D,
        PAD_SLOT_ID=PAD_SLOT_ID,
    )
