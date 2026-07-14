from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl


_MAX_FUSED_SIZE = 65536


def _get_num_warps_from_block_size(block_size: int) -> int:
    if block_size >= 32768:
        return 32
    if block_size >= 8192:
        return 16
    if block_size >= 2048:
        return 8
    return 4


def _largest_power_of_2(n: int) -> int:
    assert n > 0, f"{n=}"
    return 1 << (n.bit_length() - 1)


@lru_cache(maxsize=128)
def _get_grid_size_for_mem_bw_kernel(device: torch.device, factor: int = 8) -> int:
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    return _largest_power_of_2(num_sms) * factor


@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    rstd_ptr,
    eps,
    x_stride_0,
    y_stride_0,
    n_cols,
    block_size_n: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, block_size_n)
    mask_n = offs_n < n_cols

    weight = tl.load(weight_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + pid_m * x_stride_0 + offs_n, mask=mask_n, other=0.0).to(
        tl.float32
    )
    row_var = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.math.rsqrt(row_var + eps)
    tl.store(rstd_ptr + pid_m, rstd)

    y = x * rstd * weight
    tl.store(y_ptr + pid_m * y_stride_0 + offs_n, y, mask=mask_n)


@triton.jit(do_not_specialize=["n_rows"])
def _rmsnorm_fwd_kernel_block_m(
    x_ptr,
    weight_ptr,
    y_ptr,
    rstd_ptr,
    eps,
    x_stride_0,
    y_stride_0,
    n_rows,
    n_cols,
    block_size_m: tl.constexpr,
    block_size_n: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, block_size_n)
    mask_n = offs_n < n_cols

    num_blocks_m = tl.cdiv(n_rows, block_size_m)
    blocks_per_pid = tl.cdiv(num_blocks_m, tl.num_programs(0))
    block_id_start = pid_m * blocks_per_pid
    block_id_end = min(block_id_start + blocks_per_pid, num_blocks_m)

    weight = tl.load(weight_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    for block_id in range(block_id_start, block_id_end):
        offs_m = block_id * block_size_m + tl.arange(0, block_size_m)
        mask_m = offs_m < n_rows
        mask_mn = mask_m[:, None] & mask_n[None, :]
        x = tl.load(
            x_ptr + offs_m[:, None] * x_stride_0 + offs_n[None, :],
            mask=mask_mn,
            other=0.0,
        ).to(tl.float32)
        row_var = tl.sum(x * x, axis=1) / n_cols
        rstd = tl.math.rsqrt(row_var + eps)
        tl.store(rstd_ptr + offs_m, rstd, mask=mask_m)

        y = x * rstd[:, None] * weight
        tl.store(
            y_ptr + offs_m[:, None] * y_stride_0 + offs_n[None, :],
            y,
            mask=mask_mn,
        )


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.ndim == 2, f"{x.shape=}"
    assert weight.ndim == 1, f"{weight.shape=}"
    n_rows, n_cols = x.shape
    assert weight.shape[0] == n_cols, f"{weight.shape=} {x.shape=}"
    y = torch.empty_like(x)
    rstd = torch.empty((n_rows,), dtype=torch.float32, device=x.device)

    block_size_n = triton.next_power_of_2(n_cols)
    max_block_size_n = _MAX_FUSED_SIZE // x.element_size()
    if max_block_size_n < block_size_n:
        raise RuntimeError(f"Large {n_cols=} is not supported")
    block_size_m = max(1, 4096 // block_size_n)
    num_warps = _get_num_warps_from_block_size(block_size_n)

    if block_size_m == 1:
        _rmsnorm_fwd_kernel[(n_rows,)](
            x,
            weight,
            y,
            rstd,
            eps,
            x.stride(0),
            y.stride(0),
            n_cols,
            block_size_n,
            num_warps=num_warps,
        )
    else:
        grid_size = _get_grid_size_for_mem_bw_kernel(x.device)
        _rmsnorm_fwd_kernel_block_m[(grid_size,)](
            x,
            weight,
            y,
            rstd,
            eps,
            x.stride(0),
            y.stride(0),
            n_rows,
            n_cols,
            block_size_m,
            block_size_n,
            num_warps=num_warps,
        )
    return y
