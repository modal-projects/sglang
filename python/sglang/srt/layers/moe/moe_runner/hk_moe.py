"""HipKittens block-FP8 grouped MoE (aiter.hk_fp8_moe) for large MoE batches on MI355 (gfx950).

Specialized to GLM-5.3-Flash's TP2 shard: hidden 4096, per-GPU moe_intermediate 1024, top-8/9 (shared expert fused),
SiLU experts with optional swiglu_limit clamp. Reads the tensors the aiter runner already holds: weights in aiter's
shuffle_weight(w, (16, 16)) layout with 128x128 power-of-two (UE8M0) fp32 block scales (SGLANG_HK_MOE requantizes
them at load), so no second weight copy is needed.
"""

import functools
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Measured end to end on MI355 TP2 (GLM-5.3-Flash, tuned aiter tables): HK wins from ~1.5k tokens (3072: 180 -> 172
# ms), aiter's fused_moe below (1024: 76 vs 79 ms); 2048 keeps a margin over the crossover.
_MIN_TOKENS = 2048
_HIDDEN, _INTERMEDIATE, _MAX_EXPERTS = 4096, 1024, 1024


@functools.cache
def _hk_fp8_moe():
    """aiter's hk_fp8_moe on gfx950 with an aiter that has it, else None (the aiter runner keeps the call)."""
    from aiter import get_gfx

    try:
        from aiter.ops.hk_fp8_moe import hk_fp8_moe
    except ImportError:
        hk_fp8_moe = None
    if hk_fp8_moe is None or get_gfx() != "gfx950":
        logger.warning(
            "SGLANG_HK_MOE is set but aiter.hk_fp8_moe is unavailable (needs gfx950 and an aiter with it); "
            "using aiter's fused_moe"
        )
        return None
    logger.info(
        "Using HipKittens block-FP8 MoE for MoE batches >= %d tokens", _MIN_TOKENS
    )
    return hk_fp8_moe


def maybe_forward(runner_input, quant_info, config) -> Optional[torch.Tensor]:
    """Run the HK MoE when the call matches its contract, else return None (the caller runs aiter)."""
    hk_fp8_moe = _hk_fp8_moe()
    if hk_fp8_moe is None or not _supported(runner_input, quant_info, config):
        return None
    return hk_fp8_moe(
        runner_input.hidden_states,
        runner_input.topk_ids,
        runner_input.topk_weights,
        quant_info.w13_weight,
        quant_info.w13_scale,
        quant_info.w2_weight,
        quant_info.w2_scale,
        swiglu_limit=quant_info.swiglu_limit,
    )


def _supported(runner_input, quant_info, config) -> bool:
    from sglang.srt.layers.moe.moe_runner.aiter import AiterQuantType

    x, w13, w2 = runner_input.hidden_states, quant_info.w13_weight, quant_info.w2_weight
    num_experts = w13.shape[0]
    return (
        x.shape[0] >= _MIN_TOKENS
        and config.activation == "silu"
        and not config.no_combine
        and runner_input.quant_type == AiterQuantType.PER_128X128
        and runner_input.a1_scale is None
        and runner_input.num_local_tokens is None
        and runner_input.output_dtype in (None, torch.bfloat16)
        and quant_info.a13_scale is None
        and quant_info.b13 is None
        and quant_info.b2 is None
        and quant_info.expert_mask is None
        and not quant_info.doweight_stage1
        and not quant_info.hidden_pad
        and not quant_info.intermediate_pad
        and _gate_up_separated(quant_info.fused_moe_kwargs)
        and x.dtype == torch.bfloat16
        and tuple(x.shape[1:]) == (_HIDDEN,)
        and tuple(w13.shape[1:]) == (2 * _INTERMEDIATE, _HIDDEN)
        and tuple(w2.shape[1:]) == (_HIDDEN, _INTERMEDIATE)
        and num_experts <= _MAX_EXPERTS
        and tuple(quant_info.w13_scale.shape)
        == (num_experts, 2 * _INTERMEDIATE // 128, _HIDDEN // 128)
        and tuple(quant_info.w2_scale.shape)
        == (num_experts, _HIDDEN // 128, _INTERMEDIATE // 128)
        and quant_info.w13_scale.dtype == torch.float32
        and quant_info.w2_scale.dtype == torch.float32
        and runner_input.topk_ids.shape[1] in (8, 9)
    )


def _gate_up_separated(fused_moe_kwargs: Optional[dict]) -> bool:
    """The kernel reads w13 as [gate; up]; aiter's gate/up-interleaved layout is not supported."""
    if not fused_moe_kwargs:
        return True
    from aiter.ops.flydsl.moe_common import GateMode

    return fused_moe_kwargs == {"gate_mode": GateMode.SEPARATED.value}
