import torch
import triton
import triton.language as tl

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0
NVFP4_GLOBAL_SCALE_NUMERATOR = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX


@triton.jit
def _partial_absmax_kernel(
    x_ptr,
    partial_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    partial = tl.max(tl.abs(x), axis=0)
    tl.store(partial_ptr + pid, partial)


@triton.jit
def _finish_global_scale_kernel(
    partial_ptr,
    scale_ptr,
    num_partials,
    NUMERATOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_partials
    partials = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    amax = tl.max(partials, axis=0)
    amax = tl.maximum(amax, 1.0e-12)
    scale = NUMERATOR / amax
    tl.store(scale_ptr, scale)


def compute_nvfp4_global_scale(hidden_states: torch.Tensor) -> torch.Tensor:
    """Compute FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / absmax(hidden_states)."""
    assert hidden_states.is_cuda, "NVFP4 global-scale kernel requires CUDA"
    if not hidden_states.is_contiguous():
        x_amax = hidden_states.abs().max().clamp(min=1e-12).float()
        return (NVFP4_GLOBAL_SCALE_NUMERATOR / x_amax).float()

    numel = hidden_states.numel()
    scale = torch.empty((), dtype=torch.float32, device=hidden_states.device)
    if numel == 0:
        scale.fill_(NVFP4_GLOBAL_SCALE_NUMERATOR / 1.0e-12)
        return scale

    # Keep the second-stage reduction to a single Triton program for normal Inkling
    # token/hidden sizes.
    block_size = max(4096, triton.next_power_of_2(triton.cdiv(numel, 65536)))
    num_partials = triton.cdiv(numel, block_size)
    partials = torch.empty(
        (num_partials,), dtype=torch.float32, device=hidden_states.device
    )

    _partial_absmax_kernel[(num_partials,)](
        hidden_states,
        partials,
        numel,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )

    finish_block = triton.next_power_of_2(num_partials)
    _finish_global_scale_kernel[(1,)](
        partials,
        scale,
        num_partials,
        NUMERATOR=NVFP4_GLOBAL_SCALE_NUMERATOR,
        BLOCK_SIZE=finish_block,
        num_warps=8,
    )
    return scale
