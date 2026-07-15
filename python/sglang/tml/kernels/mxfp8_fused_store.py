"""Fused MXFP8 quantize + paged-KV scatter store (tier-2 KV-write path).

One kernel launch replaces the legacy three-step store of a K or V tensor
into the MXFP8 KV pool:

    1. ``to_mxfp8`` (``_mxfp8_quant_kernel``)     — bf16 -> e4m3 + e8m0 scales
    2. ``k_buffer[loc] = data``                   — aten fancy-index scatter
    3. ``store_sf_interleaved``                   — interleaved scale scatter

Bit-identical quantization recipe (per 32-element group along head_dim,
fp32 math — must match ``mxfp8_quant._mxfp8_quant_kernel`` exactly):

    amax         = max(|x|, 1e-30)
    scale_biased = clamp(ceil(log2(amax / 448)) + 127, 0, 254)
    descale      = 2^(scale_biased - 127)
    data         = clamp(x / descale, -448, 448) -> e4m3
    scale byte   = uint8(scale_biased)            (e8m0 biased exponent)

Scale layout (page_size == 128, head_dim == 128 only): the FA4
BlockScaledBasicChunk atom layout ``[num_pages, nheads, 32, 4, sf_dim=4]``
e8m0. The 4 scale bytes of a (token, head) are contiguous, so they are
written as one packed little-endian u32 at u32-offset

    page * nheads * 128 + h * 128 + (t % 32) * 4 + t // 32

for a token at page offset ``t`` (must match
``mxfp8_interleave_sf._store_sf_interleaved_kernel`` byte-for-byte).

CUDA-graph-capture safe: fixed grid derived from tensor shapes, no host
syncs, no ``.item()``. Negative ``loc`` entries are skipped (the legacy
aten fancy-index would instead wrap negative indices Python-style; real
callers never pass negative slots).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.tml.kernels.mxfp8_quant import MXFP8_BLOCK_SIZE


@triton.jit
def _mxfp8_quant_store_kernel(
    x_ptr,  # [N, H, D] bf16 (arbitrary strides)
    data_ptr,  # [slots, H, D] e4m3, NHD contiguous
    sf_ptr,  # u32 view of [num_pages, H, 32, 4, 4] e8m0, contiguous
    loc_ptr,  # [N] int slot ids
    N,
    sxn,
    sxh,
    sxd,
    H: tl.constexpr,
    NUM_GROUPS: tl.constexpr,  # D // 32 (== 4)
    PAGE_SIZE: tl.constexpr,  # 128
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    h = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    loc = tl.load(loc_ptr + offs_n, mask=n_mask, other=0).to(tl.int64)
    valid = n_mask & (loc >= 0)

    g = tl.arange(0, NUM_GROUPS)
    e = tl.arange(0, 32)
    d = g[:, None] * 32 + e[None, :]  # (G, 32) head_dim offsets

    x = tl.load(
        x_ptr + offs_n[:, None, None] * sxn + h * sxh + d[None, :, :] * sxd,
        mask=n_mask[:, None, None],
        other=0.0,
    ).to(tl.float32)  # (BLOCK_N, G, 32)

    # Quant recipe — identical fp32 op sequence to _mxfp8_quant_kernel.
    amax = tl.maximum(tl.max(tl.abs(x), axis=2), 1e-30)
    scale_biased = tl.ceil(tl.log2(amax / 448.0)) + 127.0
    scale_biased = tl.minimum(tl.maximum(scale_biased, 0.0), 254.0)
    descale = tl.exp2(scale_biased - 127.0)
    xq = tl.clamp(x / descale[:, :, None], -448.0, 448.0).to(
        data_ptr.dtype.element_ty
    )

    # Data scatter: token rows are H*D contiguous e4m3 in the NHD pool.
    D: tl.constexpr = NUM_GROUPS * 32
    tl.store(
        data_ptr + (loc * H + h)[:, None, None] * D + d[None, :, :],
        xq,
        mask=valid[:, None, None],
    )

    # Scale scatter: pack the 4 e8m0 bytes little-endian into one u32 and
    # write at the interleaved BlockScaledBasicChunk position.
    sb = scale_biased.to(tl.uint32)  # (BLOCK_N, G), values in [0, 254]
    packed = tl.sum(sb << (g * 8)[None, :].to(tl.uint32), axis=1)
    page = loc // PAGE_SIZE
    t = loc % PAGE_SIZE
    CHUNK: tl.constexpr = PAGE_SIZE // 32
    ipos = (t % 32) * CHUNK + t // 32
    U32_PER_HEAD: tl.constexpr = 32 * CHUNK  # == 128 for page_size 128
    tl.store(
        sf_ptr + (page * H + h) * U32_PER_HEAD + ipos,
        packed.to(tl.int32, bitcast=True),
        mask=valid,
    )


def mxfp8_quant_store(
    x: torch.Tensor,  # [N, H, D] bf16 (strided views OK)
    data_buf: torch.Tensor,  # [slots, H, D] float8_e4m3fn, contiguous
    sf_buf: torch.Tensor,  # [num_pages, H, 32, 4, 4] e8m0/uint8, contiguous
    loc: torch.Tensor,  # [N] int32/int64 slot ids
    page_size: int = 128,
) -> None:
    """Quantize bf16 rows to MXFP8 and scatter data + interleaved scales
    into the paged KV pool in a single launch."""
    assert x.dim() == 3, f"expected [N, H, D] input, got {tuple(x.shape)}"
    N, H, D = x.shape
    assert page_size == 128, (
        f"fused MXFP8 store requires page_size=128 (interleaved SF layout), "
        f"got {page_size}"
    )
    assert D == 128, f"fused MXFP8 store requires head_dim=128, got {D}"
    num_groups = D // MXFP8_BLOCK_SIZE  # 4 => one u32 of scales per token-head
    assert data_buf.dtype == torch.float8_e4m3fn and data_buf.is_contiguous()
    assert data_buf.shape[1:] == (H, D), (
        f"data buffer {tuple(data_buf.shape)} does not match input heads/dim "
        f"({H}, {D})"
    )
    assert sf_buf.is_contiguous()
    assert sf_buf.shape == (
        data_buf.shape[0] // page_size,
        H,
        32,
        page_size // 32,
        num_groups,
    ), f"unexpected sf buffer shape {tuple(sf_buf.shape)}"
    if N == 0:
        return

    sf_u8 = (
        sf_buf.view(torch.uint8)
        if sf_buf.dtype == torch.float8_e8m0fnu
        else sf_buf
    )
    sf_u32 = sf_u8.reshape(-1, 4).view(torch.int32).reshape(-1)

    block_n = min(64, triton.next_power_of_2(N))
    grid = (triton.cdiv(N, block_n), H)
    _mxfp8_quant_store_kernel[grid](
        x,
        data_buf,
        sf_u32,
        loc,
        N,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        H=H,
        NUM_GROUPS=num_groups,
        PAGE_SIZE=page_size,
        BLOCK_N=block_n,
    )
