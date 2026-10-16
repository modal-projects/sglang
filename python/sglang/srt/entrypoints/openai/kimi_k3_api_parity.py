"""Kimi-K3 request handling matching Modal's production (B300) K3 endpoint.

Sampling parameters are passed through unchanged. Effort tiers and thinking
switches K3 cannot honor fall back to the encoder default (K3 always reasons)
with a warning instead of failing the request.
"""

import logging
from typing import Any, Dict, Optional

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

logger = logging.getLogger(__name__)

KIMI_K3_THINKING_EFFORTS = ("low", "high", "max")
# Moonshot fixes top_p at 0.95 and temperature at 1.0 for K3 (prod parity).
KIMI_K3_DEFAULT_TOP_P = 0.95
KIMI_K3_DEFAULT_TEMPERATURE = 1.0


def apply_sampling_defaults(
    sampling_params: Dict[str, Any],
    model_defaults: Dict[str, Any],
    requested_top_p: Optional[float],
    requested_temperature: Optional[float],
) -> None:
    """Supply Moonshot's fixed sampling values when nothing else does."""
    if requested_top_p is None and "top_p" not in model_defaults:
        sampling_params["top_p"] = KIMI_K3_DEFAULT_TOP_P
    if requested_temperature is None and "temperature" not in model_defaults:
        sampling_params["temperature"] = KIMI_K3_DEFAULT_TEMPERATURE


def drop_unsupported_reasoning_controls(request: ChatCompletionRequest) -> None:
    """Remove thinking-disable switches and unsupported effort tiers in place.

    normalize_reasoning_inputs funnels every disable spelling (including
    thinking={"type":"disabled"} and reasoning_effort="none") into the
    thinking/enable_thinking template kwargs.
    """
    kwargs = request.chat_template_kwargs or {}
    for key in ("thinking", "enable_thinking"):
        if kwargs.get(key) is False:
            logger.warning("Kimi K3 always reasons; ignoring %s=False.", key)
            del kwargs[key]
    for key in ("reasoning_effort", "thinking_effort"):
        if kwargs.get(key) not in (None, *KIMI_K3_THINKING_EFFORTS):
            logger.warning(
                "Kimi K3 does not support %s=%r; using the default.", key, kwargs[key]
            )
            del kwargs[key]
    if request.reasoning_effort not in (None, *KIMI_K3_THINKING_EFFORTS):
        logger.warning(
            "Kimi K3 does not support reasoning_effort=%r; using the default.",
            request.reasoning_effort,
        )
        request.reasoning_effort = None
