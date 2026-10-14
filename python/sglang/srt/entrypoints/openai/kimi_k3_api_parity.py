"""Kimi-K3 request validation and sampling defaults matching Modal's
production K3 endpoint.

Requests outside the sampling envelope target-only verification supports, or
asking for an effort tier K3 cannot honor, are rejected rather than silently
adjusted.
"""

from typing import Any, Dict, Optional

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

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


def validate_sampling_envelope(
    *,
    temperature: Optional[float],
    top_p: Optional[float],
    presence_penalty: float,
    frequency_penalty: float,
    n: int,
) -> Optional[str]:
    if temperature is not None and not (0.0 <= temperature <= 1.0):
        return "Kimi K3 temperature must be within [0.0, 1.0]."
    if top_p is not None and not (0.95 <= top_p <= 1.0):
        return "Kimi K3 top_p must be within [0.95, 1.0]."
    if presence_penalty != 0:
        return "Kimi K3 presence_penalty must be 0."
    if frequency_penalty != 0:
        return "Kimi K3 frequency_penalty must be 0."
    if n != 1:
        return "Kimi K3 n must be 1."
    return None


def validate_reasoning_effort(request: ChatCompletionRequest) -> Optional[str]:
    """Omitting the field stays valid; Moonshot documents it as optional and
    defaulting to "max"."""
    template_kwargs = request.chat_template_kwargs or {}
    supported = ", ".join(repr(tier) for tier in KIMI_K3_THINKING_EFFORTS)

    # normalize_reasoning_inputs funnels every disable spelling into these
    # keys, including thinking={"type":"disabled"} and reasoning_effort="none".
    if False in (
        template_kwargs.get("thinking"),
        template_kwargs.get("enable_thinking"),
    ):
        return (
            "Kimi K3 always reasons and cannot disable thinking. Omit the "
            f"thinking controls, or pass a reasoning_effort of {supported}."
        )

    # chat_template_kwargs.reasoning_effort is promoted onto the request
    # later, in _convert_to_internal_request.
    candidates = (
        ("reasoning_effort", template_kwargs.get("reasoning_effort")),
        ("reasoning_effort", request.reasoning_effort),
        ("thinking_effort", template_kwargs.get("thinking_effort")),
    )
    for field, value in candidates:
        if value is None:
            continue
        if value not in KIMI_K3_THINKING_EFFORTS:
            return (
                f"Kimi K3 does not support {field}={value!r}; "
                f"supported values are {supported}."
            )
    return None


def validate_chat_request(request: ChatCompletionRequest) -> Optional[str]:
    return validate_reasoning_effort(request) or validate_sampling_envelope(
        temperature=request.temperature,
        top_p=request.top_p,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        n=request.n,
    )
