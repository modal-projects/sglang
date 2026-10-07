"""Kimi-K3 MLA extend-with-prefix K/V assembly without the concat (ROCm).

An extend batch with a cached prefix runs MHA over the whole sequence, so the
aiter backend rebuilds per-head K/V for every prefix token of every MLA layer
(aiter_backend.forward_extend, ``qk_head_dim != kv_lora + rope`` branch):

    K_Buffer = index_select(cache, kv_indices)          vectorized_gather
    kvc, k_pe = split(K_Buffer); kvc.to(bf16); k_pe.to  2x strided fp8 cast
    kv = kv_b_proj(kvc.contiguous())                    [N, H*(dn+dv)] GEMM
    k = cat([kv[..., :dn], broadcast(k_pe, H)])         CatArrayBatchedCopy
    v = kv[..., dn:]                                    (view)

At 99k prefix tokens x 3 requests the cast + concat cost ~1.7 ms per layer,
more than the GEMM. Under SGLANG_ROCM_K3_MLA_PREFIX_NOCAT this module instead:

1. one Triton launch gathers the latent rows from the paged cache by
   kv_indices, casts them to bf16 into a dense [N, kv_lora] GEMM input, and
   writes k_pe (cast, broadcast to all H heads) straight into its final slot
   of the K buffer;
2. mode "bmm": one strided-batched GEMM (batch = head, A broadcast with batch
   stride 0) writes [v_h | k_nope_h] of every head straight into a
   [N, H, dv + dn + rope] buffer whose per-head row is [v | k_nope | k_pe],
   so K = buf[..., dv:] and V = buf[..., :dv] are views and nothing is copied.
   The weight is kv_b_proj's rows permuted per head to [v; k_nope] (cached,
   3 MB per layer).
   mode "copy": the original kv_b_proj GEMM, then one Triton copy of k_nope
   into a dense [N, H, dn + rope] K (k_pe already in place); V stays a view.

The attention kernel (aiter opus gqa_d192_v128, varlen) takes arbitrary row /
head strides for K and V, so the strided views go in as-is.

Measured on MI355X (one layer, 12 heads, K/V build only, attention excluded):
3x(90k+9k) rows 2.67 ms old -> 1.45 ms copy (GEMM 0.96 + copy 0.32 + gather
0.17) / 1.41 ms bmm (batched GEMM 1.25 + gather 0.16); 1x(90k+9k) 0.90 -> 0.50.
hipBLASLt picks a slower tile for the batched GEMM (and a different K split,
so bf16 results differ by 1 ulp in ~1e-5 of elements), and the attention
kernel is ~1% slower on the wider-strided K, so "copy" is the default.

Numerics: the gather / cast / broadcast are exact (e4m3fn -> bf16 is exact,
no scale, as the original ``.to``). Mode "copy" is bit-identical to the
original path. Mode "bmm" computes the same dot products with a different
GEMM call (per head instead of one wide GEMM); whether hipBLASLt returns
bit-identical bf16 values is checked by the unit test, not assumed.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _k3_prefix_gather_kernel(
    cache_ptr,  # [P, LORA + ROPE] (fp8 or bf16), row stride s_cache
    idx_ptr,  # [N] int32/int64
    kvc_ptr,  # [N, LORA] bf16 dense out (GEMM input)
    kpe_ptr,  # k_pe destination: element (n, h, d) at n*s_kpe_n + h*s_kpe_h + d
    N,
    s_cache,
    s_kpe_n,
    s_kpe_h,
    LORA: tl.constexpr,
    ROPE: tl.constexpr,
    H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    idx = tl.load(idx_ptr + rows, mask=rmask, other=0).to(tl.int64)
    src = cache_ptr + idx[:, None] * s_cache
    rows64 = rows.to(tl.int64)[:, None]

    offs_l = tl.arange(0, LORA)[None, :]
    lat = tl.load(src + offs_l, mask=rmask[:, None], other=0.0)
    tl.store(
        kvc_ptr + rows64 * LORA + offs_l,
        lat.to(kvc_ptr.dtype.element_ty),
        mask=rmask[:, None],
    )

    offs_r = tl.arange(0, ROPE)[None, :]
    pe = tl.load(src + LORA + offs_r, mask=rmask[:, None], other=0.0)
    pe = pe.to(kpe_ptr.dtype.element_ty)
    dst = kpe_ptr + rows64 * s_kpe_n + offs_r
    for h in tl.static_range(H):
        tl.store(dst + h * s_kpe_h, pe, mask=rmask[:, None])


@triton.jit
def _k3_prefix_knope_copy_kernel(
    kv_ptr,  # [R, *] rows of the GEMM output, row stride s_kv
    k_ptr,  # [R, *] rows of K, row stride s_k
    R,
    s_kv,
    s_k,
    DN: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = (pid * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
    rmask = rows < R
    offs = tl.arange(0, DN)[None, :]
    x = tl.load(kv_ptr + rows[:, None] * s_kv + offs, mask=rmask[:, None])
    tl.store(k_ptr + rows[:, None] * s_k + offs, x, mask=rmask[:, None])


def _gather_dequant(
    cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kvc: torch.Tensor,
    kpe_dst: torch.Tensor,
    kv_lora_rank: int,
    rope_dim: int,
) -> None:
    n = kv_indices.shape[0]
    if n == 0:
        return
    cache2d = cache.view(cache.shape[0], -1)
    assert cache2d.shape[1] == kv_lora_rank + rope_dim and cache2d.stride(1) == 1
    assert kpe_dst.dim() == 3 and kpe_dst.stride(2) == 1
    block_n = 8
    _k3_prefix_gather_kernel[(triton.cdiv(n, block_n),)](
        cache2d,
        kv_indices,
        kvc,
        kpe_dst,
        n,
        cache2d.stride(0),
        kpe_dst.stride(0),
        kpe_dst.stride(1),
        LORA=kv_lora_rank,
        ROPE=rope_dim,
        H=kpe_dst.shape[1],
        BLOCK_N=block_n,
        num_warps=4,
    )


def _vk_weight(
    kv_b_proj: torch.nn.Module, num_heads: int, nope_dim: int, v_dim: int
) -> torch.Tensor:
    """kv_b_proj weight [H*(dn+dv), L] -> per-head [v; k_nope] rows, [H, dv+dn, L].
    Cached on the module (keyed by the weight storage)."""
    w = kv_b_proj.weight
    key = (w.data_ptr(), w._version, num_heads, nope_dim, v_dim)
    cached = getattr(kv_b_proj, "_k3_nocat_vk_weight", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    w3 = w.detach().view(num_heads, nope_dim + v_dim, -1)
    wvk = torch.cat([w3[:, nope_dim:], w3[:, :nope_dim]], dim=1).contiguous()
    kv_b_proj._k3_nocat_vk_weight = (key, wvk)
    return wvk


def bmm_eligible(
    kv_b_proj: torch.nn.Module,
    num_heads: int,
    nope_dim: int,
    v_dim: int,
    kv_lora_rank: int,
) -> bool:
    """The "bmm" mode reads kv_b_proj.weight directly, so it only applies to a
    plain dense bf16 weight without bias (UnquantizedLinearMethod)."""
    w = getattr(kv_b_proj, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dtype != torch.bfloat16 or w.dim() != 2:
        return False
    if tuple(w.shape) != (num_heads * (nope_dim + v_dim), kv_lora_rank):
        return False
    if not w.is_contiguous() or getattr(kv_b_proj, "bias", None) is not None:
        return False
    qm = getattr(kv_b_proj, "quant_method", None)
    return qm is None or type(qm).__name__ == "UnquantizedLinearMethod"


def k3_mla_prefix_kv(
    cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_b_proj: torch.nn.Module,
    num_heads: int,
    nope_dim: int,
    v_dim: int,
    kv_lora_rank: int,
    rope_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    mode: str = "copy",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (k [N, H, dn+rope], v [N, H, dv]) for the prefix rows
    ``cache[kv_indices]``; both may be strided views (unit last stride)."""
    n = kv_indices.shape[0]
    dev = cache.device
    kvc = torch.empty((n, kv_lora_rank), dtype=dtype, device=dev)
    if mode == "bmm":
        row = v_dim + nope_dim + rope_dim
        buf = torch.empty((n, num_heads, row), dtype=dtype, device=dev)
        k = buf[:, :, v_dim:]
        v = buf[:, :, :v_dim]
        _gather_dequant(
            cache, kv_indices, kvc, buf[:, :, v_dim + nope_dim :], kv_lora_rank, rope_dim
        )
        if n > 0:
            wvk = _vk_weight(kv_b_proj, num_heads, nope_dim, v_dim)
            torch.bmm(
                kvc.unsqueeze(0).expand(num_heads, n, kv_lora_rank),
                wvk.transpose(1, 2),
                out=buf[:, :, : v_dim + nope_dim].transpose(0, 1),
            )
        return k, v
    if mode != "copy":
        raise ValueError(f"unknown k3_mla_prefix_kv mode {mode!r}")
    k = torch.empty((n, num_heads, nope_dim + rope_dim), dtype=dtype, device=dev)
    _gather_dequant(cache, kv_indices, kvc, k[:, :, nope_dim:], kv_lora_rank, rope_dim)
    kv = kv_b_proj(kvc)[0]
    assert kv.is_contiguous()
    kv = kv.view(n, num_heads, nope_dim + v_dim)
    v = kv[:, :, nope_dim:]
    rows = n * num_heads
    if rows > 0:
        block_r = 32
        _k3_prefix_knope_copy_kernel[(triton.cdiv(rows, block_r),)](
            kv,
            k,
            rows,
            nope_dim + v_dim,
            nope_dim + rope_dim,
            DN=nope_dim,
            BLOCK_R=block_r,
            num_warps=4,
        )
    return k, v
