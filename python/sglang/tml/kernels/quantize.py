# type: ignore

import torch
import triton
import triton.language as tl

NVFP4_BLOCK_SIZE = 16

E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
E2M1_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
E2M1_MAX = 6.0
E4M3_MAX = 448.0
E2M1_MAX_VALUE = tl.constexpr(6.0)
E4M3_MAX_VALUE = tl.constexpr(448.0)


@triton.jit
def _sm86_max_nan_xorsign_abs_f32(a, b):
    """
    Computes the maximum of the absolute values of the two inputs and sets its
    sign to the XOR of the signs of the inputs.

    NaN inputs propagate to the output.
    """
    tl.static_assert(
        a.dtype == tl.float32, "max.NaN.xorsign.abs.f32 requires float32 inputs"
    )
    tl.static_assert(
        b.dtype == tl.float32, "max.NaN.xorsign.abs.f32 requires float32 inputs"
    )
    return tl.inline_asm_elementwise(
        """{
    max.NaN.xorsign.abs.f32 $0, $1, $2;
    }""",
        "=r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _nan_propagating_absmax_reduce(x, axis=None, other_axis: tl.constexpr = None):
    """Return the float32 maximum of abs(x) along `axis`, propagating NaNs.

    The input `x` is expected to be float32, and the output is also float32 but
    bitcasted to uint32 and with sign bit removed.
    """
    x_absmax = tl.reduce(x, axis, _sm86_max_nan_xorsign_abs_f32, keep_dims=True)
    if other_axis is not None:
        x_absmax = tl.reduce(
            x_absmax, other_axis, _sm86_max_nan_xorsign_abs_f32, keep_dims=True
        )
    # NOTE: The reduction result's sign is the XOR of all input signs. Clear
    # the sign bit to make it represent absolute max.
    x_absmax = x_absmax.to(tl.uint32, bitcast=True) & 0x7FFFFFFF
    return x_absmax


@triton.jit
def _to_nvfp4_kernel(
    x_ptr,
    x_out_ptr,
    x_scales_ptr,
    x_amax,
    x_ptr_stride_m: tl.constexpr,
    x_ptr_stride_n: tl.constexpr,
    x_out_ptr_stride_m: tl.constexpr,
    x_out_ptr_stride_n: tl.constexpr,
    x_scales_ptr_stride_m: tl.constexpr,
    x_scales_ptr_stride_n: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    N_BLOCKS_PER_GRID: tl.constexpr,
    NVFP4_BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr,
    SEED: "tl.uint32 | None" = None,
    ENABLE_UNBIASED_SR_ADJUSTMENT: tl.constexpr = False,
) -> None:
    m_id = tl.program_id(0)
    n_id = tl.program_id(1)
    if INT64_INDEX:
        m_id = m_id.to(tl.int64)
        n_id = n_id.to(tl.int64)

    # Base offsets (scalars)
    offs_m_base = m_id * BLOCK_SIZE_M
    offs_n_base = n_id * BLOCK_SIZE_N

    offs_m = offs_m_base + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offs_n_full = offs_n_base + tl.arange(0, BLOCK_SIZE_N)[None, :]
    offs_n_half = offs_n_base // 2 + tl.arange(0, BLOCK_SIZE_N // 2)[None, :]
    offs_n_scales = (
        offs_n_base // NVFP4_BLOCK_SIZE + tl.arange(0, N_BLOCKS_PER_GRID)[None, :]
    )

    # When SR is on, optionally inflate the scales by 17/16 so the
    # post-FP8-cast block scale is guaranteed >= the unrounded target,
    # leaving headroom for `cvt.rs` to round up without saturating —
    # i.e. it makes SR truly unbiased at the cost of an output that
    # averages to (input * 16/17). Off by default; the caller is
    # expected to apply its own 17/16 correction at dequant if it
    # opts in.
    apply_sr_adjustment: tl.constexpr = (SEED is not None) and (
        ENABLE_UNBIASED_SR_ADJUSTMENT
    )

    # global scale computation
    amax = tl.load(x_amax)
    amax = amax.to(tl.float32)
    if apply_sr_adjustment:
        global_scale = amax / (E2M1_MAX_VALUE * E4M3_MAX_VALUE * 16 / 17)
    else:
        global_scale = amax / (E2M1_MAX_VALUE * E4M3_MAX_VALUE)

    # local scale computation
    x = tl.load(x_ptr + offs_m * x_ptr_stride_m + offs_n_full * x_ptr_stride_n)
    x_scale_blocks = x.reshape(BLOCK_SIZE_M, N_BLOCKS_PER_GRID, NVFP4_BLOCK_SIZE)
    block_scales = _nan_propagating_absmax_reduce(
        x_scale_blocks.to(tl.float32), axis=2
    ).to(tl.float32, bitcast=True)
    block_scales = block_scales.reshape(BLOCK_SIZE_M, N_BLOCKS_PER_GRID)
    if apply_sr_adjustment:
        block_scales = block_scales / (E2M1_MAX_VALUE * global_scale * 16 / 17)
    else:
        block_scales = block_scales / (E2M1_MAX_VALUE * global_scale)
    block_scales = tl.where(block_scales == 0, 1.0, block_scales)
    block_scales = block_scales.to(tl.float8e4nv)

    # store scales - build 2D offset grid matching v1's pattern
    offs_scale_mn = (
        offs_m * x_scales_ptr_stride_m + offs_n_scales * x_scales_ptr_stride_n
    )
    tl.store(x_scales_ptr + offs_scale_mn, block_scales)

    x_block_scaled = (
        x_scale_blocks / (block_scales.to(tl.float32) * global_scale)[:, :, None]
    )
    # Match v1 reshape sequence exactly to preserve element ordering
    x_block_scaled = x_block_scaled.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)
    # Reshape to [M, N // 2, 2] for PTX
    x_block_scaled = x_block_scaled.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)
    (x_block_scaled_b1, x_block_scaled_b2) = x_block_scaled.split()

    # PTX inline assembly: processes entire [M, N // 2] tensor
    # Interleaves values from b1 and b2 for correct byte packing
    # Rounding triton is adapted from
    # https://github.com/mit-han-lab/fouroversix/blob/main/src/fouroversix/quantize/triton_kernel.py
    if SEED is not None:
        rbits = tl.randint(
            SEED,
            tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N // 2 + offs_n_half,
        ).cast(tl.uint32)

        x_e2m1 = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 tmp0, tmp1;
            cvt.rs.satfinite.e2m1x4.f32 tmp0, {$6, $2, $5, $1}, $9;
            cvt.rs.satfinite.e2m1x4.f32 tmp1, {$8, $4, $7, $3}, $10;
            mov.b32 $0, {tmp0, tmp1};
            }
            """,
            constraints="=r,r,r,r,r,r,r,r,r,r,r,r,r",
            args=[x_block_scaled_b1, x_block_scaled_b2, rbits],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )
    else:
        # Deterministic rounding (nearest-even)
        x_e2m1 = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b8 byte0, byte1, byte2, byte3;
            cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
            cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
            cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
            cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
            mov.b32 $0, {byte0, byte1, byte2, byte3};
            }
            """,
            constraints="=r,r,r,r,r,r,r,r,r",
            args=[x_block_scaled_b1, x_block_scaled_b2],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )

    tl.store(
        x_out_ptr + offs_m * x_out_ptr_stride_m + offs_n_half * x_out_ptr_stride_n,
        x_e2m1,
    )


def quantize_to_fp4(
    x: torch.Tensor,
    x_amax: torch.Tensor | None = None,
    seed: int | None = None,
    enable_unbiased_sr_adjustment: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize `x` to NVFP4. `seed=None` selects RTN; an int enables SR.

    `enable_unbiased_sr_adjustment` toggles a kernel-side 16/17
    inflation of the FP8 block scale that prevents `cvt.rs` from
    saturating during SR. Off by default; turn on if you want truly
    unbiased SR at the cost of having to apply a `* 17/16` correction
    at dequant.
    """
    N = x.shape[-1]
    original_shape = list(x.shape)
    if x.dim() == 1:
        x = x.view(1, N)
    else:
        x = x.view(-1, N)
    M = x.shape[0]

    NVFP4_BLOCK_SIZE = 16
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = (
        4 * NVFP4_BLOCK_SIZE
    )  # Process 4 NVFP4 blocks per grid block (64 elements)

    # Require aligned dimensions for performance
    assert N % NVFP4_BLOCK_SIZE == 0, (
        f"N must be divisible by {NVFP4_BLOCK_SIZE}, got N={N}"
    )
    assert M % BLOCK_SIZE_M == 0, f"M must be divisible by {BLOCK_SIZE_M}, got M={M}"
    assert N % BLOCK_SIZE_N == 0, f"N must be divisible by {BLOCK_SIZE_N}, got N={N}"

    if x_amax is None:
        x_amax = x.abs().max().float()

    x = x.contiguous()

    x_out = torch.empty((M, N // 2), device=x.device, dtype=torch.uint8)
    n_blocks = N // NVFP4_BLOCK_SIZE
    x_scales = torch.empty(
        (M, n_blocks),
        device=x.device,
        dtype=torch.float8_e4m3fn,
    )

    grid = lambda _: (  # noqa: E731
        M // BLOCK_SIZE_M,
        N // BLOCK_SIZE_N,
    )

    _to_nvfp4_kernel[grid](
        x_ptr=x,
        x_out_ptr=x_out,
        x_scales_ptr=x_scales,
        x_amax=x_amax,
        x_ptr_stride_m=x.stride(0),
        x_ptr_stride_n=x.stride(1),
        x_out_ptr_stride_m=x_out.stride(0),
        x_out_ptr_stride_n=x_out.stride(1),
        x_scales_ptr_stride_m=x_scales.stride(0),
        x_scales_ptr_stride_n=x_scales.stride(1),
        SEED=seed,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NVFP4_BLOCK_SIZE=NVFP4_BLOCK_SIZE,
        N_BLOCKS_PER_GRID=BLOCK_SIZE_N // NVFP4_BLOCK_SIZE,
        INT64_INDEX=x.nbytes >= 2**31,
        ENABLE_UNBIASED_SR_ADJUSTMENT=enable_unbiased_sr_adjustment,
    )

    original_shape[-1] = original_shape[-1] // 2
    return (
        x_out.reshape(tuple(original_shape)),
        x_scales.reshape(tuple(original_shape[:-1] + [n_blocks])),
        x_amax / (E2M1_MAX * E4M3_MAX),
    )
