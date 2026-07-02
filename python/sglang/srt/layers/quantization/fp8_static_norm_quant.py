"""Fused RMSNorm + static-scale FP8 quantization (flashinfer >= 0.6.12, CUDA).

Env-gated by ``SGLANG_FP8_STATIC_NORM_QUANT=1`` (default off; byte-identical
behavior when unset). Producer-side fusion for online-FP8 linears that carry a
STATIC scalar ``input_scale`` (e.g. the online-FP8 patch used for
MuonClip-trained Kimi-K2.6, where a static activation scale of 1.0 is a bare
saturating e4m3 cast): the RMSNorm feeding such a linear emits FP8 directly
and the consumer receives a pre-quantized ``(fp8_tensor, scale)`` tuple
(handled in ``fp8_utils.apply_fp8_linear``), skipping the standalone
activation-quant kernel AND halving the norm's store traffic.

Measured B200 [16384, 7168]: flashinfer ``rmsnorm_quant`` 56.5us < plain bf16
rmsnorm 70.2us < rmsnorm + static cast 124.9us. Decode M=8: 1.87us == plain
rmsnorm.

NOTE: only wire this with static scale 1.0 (the online-FP8 patch default).
The flashinfer quant-scale convention (multiply vs divide) is not validated
here against sglang's ``static_quant_fp8`` (divide) for scales != 1.0.
"""

from typing import Optional, Tuple

import torch

from sglang.srt.utils import get_bool_env_var, is_cuda, is_flashinfer_available

_FP8_DTYPE = torch.float8_e4m3fn

ENABLE_FP8_STATIC_NORM_QUANT = (
    get_bool_env_var("SGLANG_FP8_STATIC_NORM_QUANT")
    and is_cuda()
    and is_flashinfer_available()
)


def static_fp8_input_scale_of(linear) -> Optional[torch.Tensor]:
    """The static scalar ``input_scale`` of an FP8 linear, or None.

    Returns a tensor only when the consumer linear actually holds an FP8
    weight and a static scalar activation scale (set after
    ``process_weights_after_loading``), i.e. when feeding it a pre-quantized
    ``(fp8, scale)`` tuple is valid.
    """
    if not ENABLE_FP8_STATIC_NORM_QUANT or linear is None:
        return None
    # Piecewise CUDA graph: flashinfer's rmsnorm_quant falls back to the
    # CuteDSL rmsnorm_quant_cute launcher for strided rows, which dynamo
    # cannot trace (SymInt __imul__ graph break under fullgraph=True; PCG
    # compile crash on kimi-bench-v2pcg 2026-07-02). PCG only captures
    # small-M extends where the fused norm+quant saves ~nothing (56.5us at
    # M=16384); disengage under PCG and let the linear quantize its own
    # input. The eager >max-tokens path (16k chunks) keeps SNQ.
    from sglang.srt.compilation.piecewise_context_manager import (
        is_in_piecewise_cuda_graph,
    )

    if is_in_piecewise_cuda_graph():
        return None
    weight = getattr(linear, "weight", None)
    if weight is None or weight.dtype != _FP8_DTYPE:
        return None
    input_scale = getattr(linear, "input_scale", None)
    if input_scale is None or input_scale.numel() != 1:
        return None
    return input_scale.reshape(1)


def rmsnorm_quant_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm(x) quantized straight to fp8 with a static scalar scale."""
    from flashinfer import norm

    out = torch.empty(x.shape, dtype=_FP8_DTYPE, device=x.device)
    if x.is_contiguous():
        norm.rmsnorm_quant(out, x, weight, scale, eps)
    else:
        # e.g. q_a slice of the fused qkv_a latent: strided rows, stride[-1]==1
        norm.rmsnorm_quant_cute(out, x, weight, scale, eps)
    return out, scale


def fused_add_rmsnorm_quant_fp8(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """residual += x; out_fp8 = quant(RMSNorm(residual)). Residual in-place."""
    from flashinfer import norm

    out = torch.empty(x.shape, dtype=_FP8_DTYPE, device=x.device)
    norm.fused_add_rmsnorm_quant(out, x, residual, weight, scale, eps)
    return (out, scale), residual
