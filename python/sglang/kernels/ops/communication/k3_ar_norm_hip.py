"""K3 ROCm TP8: the MoE [latent | shared] all-reduce with the latent RMSNorm
fused into its epilogue (see ``jit/csrc/kimi_k3/comm/ar_norm_hip.cuh``).
``out`` is bit-identical to AITER's 2-stage all-reduce; ``out_norm`` matches
the standalone RMSNorm up to rare 1-ulp bf16 flips (fp32 sum order)."""

from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

WORLD_SIZE = 8
MAX_BLOCKS = 80
MAX_TPB = 3
THREADS = 512


@cache_once
def _module():
    from aiter.jit.core import AITER_CSRC_DIR

    return load_jit(
        "k3_ar_norm_hip",
        cuda_files=["kimi_k3/comm/ar_norm_hip.cuh"],
        cuda_wrappers=[("run", "k3_ar_norm::ArNorm::run")],
        extra_include_paths=[f"{AITER_CSRC_DIR}/include"],
        extra_cuda_cflags=["-O3"],
    )


def _slice_packs(width: int) -> int:
    return width // 8 // WORLD_SIZE


def tokens_per_block(num_tokens: int, width: int) -> int:
    return max(1, -(-num_tokens // MAX_BLOCKS))


def supported(num_tokens: int, latent: int, width: int) -> bool:
    s = _slice_packs(width)
    tpb = tokens_per_block(num_tokens, width)
    return (
        num_tokens >= 1
        and latent % 8 == 0
        and width % (8 * WORLD_SIZE) == 0
        and s % 8 == 0
        and tpb <= MAX_TPB
        and tpb * s <= THREADS
    )


def usable(ca, num_tokens: int, latent: int, width: int) -> bool:
    return (
        ca is not None
        and not getattr(ca, "disabled", True)
        and hasattr(ca, "_pool")
        and getattr(ca, "world_size", 0) == WORLD_SIZE
        and supported(num_tokens, latent, width)
        and not (ca._IS_CAPTURING and not torch.cuda.is_current_stream_capturing())
    )


def ar_norm(
    ca,
    buf: torch.Tensor,
    num_tokens: int,
    latent: int,
    weight: torch.Tensor,
    eps: float,
    *,
    tpb: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(AR(buf), rmsnorm(AR(buf)[: T * latent].view(T, latent)))."""
    assert buf.is_contiguous() and buf.dtype == torch.bfloat16
    width = buf.numel() // num_tokens
    tpb = tpb or tokens_per_block(num_tokens, width)
    if torch.cuda.is_current_stream_capturing() and getattr(
        ca, "enable_register_for_capturing", True
    ):
        assert ca._IS_CAPTURING
        reg_ptr, reg_bytes = 0, 0
    else:
        reg_ptr, reg_bytes = ca._pool["input"].data_ptr, ca._pool["input"].max_size
    out = torch.empty_like(buf)
    out_norm = torch.empty(num_tokens, latent, dtype=buf.dtype, device=buf.device)
    _module().run(
        ca._ptr, buf, out, out_norm, weight, float(eps), tpb, reg_ptr, reg_bytes
    )
    return out, out_norm
