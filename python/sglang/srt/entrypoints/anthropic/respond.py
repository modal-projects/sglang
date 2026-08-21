"""Response shaping for the Anthropic Messages dialect.

Pure module: stop-reason resolution, usage translation, error envelopes
and whole-response conversion. No router, no engine objects — everything
feeds off OpenAI-side ``Usage``/``ChatCompletionResponse`` artifacts and
the Anthropic request lab values (spec §3/§5).
"""

import json
import logging
import uuid
from typing import Any, Optional, Union

from fastapi.responses import JSONResponse

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicError,
    AnthropicErrorResponse,
    AnthropicMessagesResponse,
    AnthropicOutputTokensDetails,
    AnthropicUsage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse
from sglang.srt.utils.anthropic_http import (
    ANTHROPIC_REQUEST_ID_SCOPE_KEY,
    _ANTHROPIC_STATUS_TO_ERROR_TYPE,
)

logger = logging.getLogger(__name__)

# Map OpenAI finish reasons to Anthropic stop reasons (spec §3.1).
#
# * ``stop``/``length``/``tool_calls`` map directly. (A ``stop`` finish
#   that carries a string ``matched_stop`` CONFIRMED as a request
#   ``stop_sequences`` member is upgraded to ``stop_sequence`` by
#   ``_resolve_stop_reason`` — see there for the ordering contract.)
# * ``abort`` stays unmapped: Anthropic's enum has no abort signal, so it
#   falls through to the ``end_turn`` default with a WARNING — matching
#   the original behaviour so operators don't lose the abort in logs.
# * ``content_filter`` is deliberately NOT mapped: no sglang code path
#   produces it (producer census: it exists only in OpenAI Literal
#   declarations), so a mapping would be dead, unverifiable code. The
#   ``refusal`` value stays in the response Literal as a shape contract.
#
# SGLang's scheduler only ever emits ``stop``/``length``/``abort``
# (``schedule_batch.py: FINISH_MATCHED_* / FINISH_LENGTH / FINISH_ABORT``)
# plus the ``tool_calls`` rewrite in ``serving_chat.py``; context-window
# exhaustion arrives as an ordinary ``length`` finish, so Anthropic's
# ``model_context_window_exceeded`` cannot be distinguished today.
STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def _resolve_stop_reason(
    finish_reason: Optional[str],
    matched_stop: Union[None, int, str],
    requested_stops: frozenset[str],
) -> tuple[str, Optional[str]]:
    """Translate an OpenAI ``finish_reason``/``matched_stop`` pair to the
    Anthropic ``(stop_reason, stop_sequence)`` wire pair.

    ORDERING CONTRACT: the ``finish_reason`` mapping OWNS the base
    semantics; a string ``matched_stop`` may only UPGRADE a plain
    ``finish_reason=="stop"`` (never override
    ``length``/``tool_calls``/…). This guards two real failures:

    * stream tool-call turns: ``serving_chat.py`` passes ``matched``
      through un-nulled on tool-call finish chunks, so naive precedence
      would confuse ``stop_sequence`` with ``tool_calls`` — the client
      needs ``tool_use``.
    * scheduler-internal strings: ``FINISH_MATCHED_STR`` variants with
      non-request payload (e.g. the ``"NaN happened"`` sentinel in
      ``schedule_batch.py``) MUST NOT leak to the wire; the matched
      string only earns the upgrade when it is verifiably a member of the
      request's own ``stop_sequences``.

    An int ``matched_stop`` is a stop TOKEN id (``FINISH_MATCHED_TOKEN``,
    e.g. EOS) — never a stop_sequence, keeps the plain mapping.
    """
    effective_finish = finish_reason or "stop"
    stop_reason = STOP_REASON_MAP.get(effective_finish)
    if stop_reason is None:
        logger.warning(
            "Unmapped OpenAI finish_reason %r; defaulting to end_turn",
            effective_finish,
        )
        stop_reason = "end_turn"

    if effective_finish == "stop" and isinstance(matched_stop, str):
        if matched_stop in requested_stops:
            return "stop_sequence", matched_stop
        # Verified non-member: scheduler-internal datapoint (e.g.
        # schedule_batch.py's ``"NaN happened"`` FINISH_MATCHED_STR
        # sentinel) or a stale matched string from another finish path —
        # NEVER trust it onto the wire.
        logger.warning(
            "matched_stop string %r not among request stop_sequences %r; "
            "treating the finish as plain end_turn",
            matched_stop,
            sorted(requested_stops),
        )
    # An int ``matched_stop`` (FINISH_MATCHED_TOKEN id, e.g. EOS) or a
    # mapped non-stop finish (length/tool_calls) keep the plain mapping.
    return stop_reason, None


# Status → error.type (spec §5.2) is an ALIAS of the route layer's
# canonical table — ``_ANTHROPIC_STATUS_TO_ERROR_TYPE`` in
# utils/anthropic_http.py is the single owner of that mapping: no
# parallel maps, and only spec-enum values are produced (a 408 entry was
# deliberately dropped: its ``request_timeout_error`` type is not part
# of the Anthropic §5.2 enum and nothing emits it). Unknown statuses
# fall back to ``api_error`` in ``_convert_openai_error_response``.
ERROR_TYPE_MAP = _ANTHROPIC_STATUS_TO_ERROR_TYPE


def _scrub_error_message(message: str, status_code: int) -> str:
    """Cap and sanitize an upstream error string before it reaches the wire.

    * 5xx: always collapse to a generic sentence — raw internals (paths,
      tracebacks, prompt fragments) never leave the server.
    * 4xx: keep the client-actionable content, but strip control chars so
      the JSON envelope can't be poisoned and cap at a generous length.
    """
    if status_code >= 500:
        return "Internal server error"
    cleaned = "".join(ch for ch in message if ch >= " " or ch in "\n\t").strip()
    if len(cleaned) > 2000:
        cleaned = cleaned[:2000] + "… (truncated)"
    return cleaned or "Request failed"


def _build_error_response(
    status_code: int,
    error_type: str,
    message: str,
    exception_name: Optional[str] = None,
    raw_request=None,
) -> JSONResponse:
    """Create an Anthropic-format error response.

    ``error.type`` is restricted to Anthropic's documented enum so strict
    SDK clients (anthropic-sdk-python / -typescript) keep parsing the
    response into their typed error classes. ``exception_name`` — when
    provided — is logged at WARNING level so operators can still grep
    server-side, but it never reaches the wire. The ``request_id``
    (spec §5.1, audit G-02) is taken SINGLE-SOURCE from the ASGI scope
    key published by ``AnthropicRequestIdMiddleware``
    (``ANTHROPIC_REQUEST_ID_SCOPE_KEY``) so the body id ALWAYS equals
    the ``request-id`` response header; when the scope is absent (unit
    paths, no middleware) a fresh ``req_…`` is minted. The id is also
    echoed in the WARNING so wire body and log line cross-reference.
    NOTE (G-24, spec §5.2): the 503 → 529 WIRE-status translation is
    owned solely by the route layer —
    ``AnthropicOverloadedStatusMiddleware`` in utils/anthropic_http.py
    (529 is Anthropic-specific, not IANA, and SDKs key overload retries
    on it). This function emits the status it is GIVEN; the
    ``overloaded_error`` error TYPE for 503s comes from
    ``ERROR_TYPE_MAP``.
    """
    scope = getattr(raw_request, "scope", None) or {}
    request_id = scope.get(ANTHROPIC_REQUEST_ID_SCOPE_KEY) or (
        f"req_{uuid.uuid4().hex}"
    )
    if exception_name:
        logger.warning(
            "Anthropic error response %s (exception=%s, request_id=%s): %s",
            error_type,
            exception_name,
            request_id,
            message,
        )
    error_resp = AnthropicErrorResponse(
        error=AnthropicError(type=error_type, message=message),
        request_id=request_id,
    )
    return JSONResponse(
        status_code=status_code,
        content=error_resp.model_dump(),
    )


def _convert_openai_error_response(response, raw_request=None) -> JSONResponse:
    """Forward an upstream OpenAI-handler error as an Anthropic error.

    The original error message is preserved for 4xx (after light
    sanitization) so callers see the real validation failure. For 5xx
    we always return a generic ``"Internal server error"`` to avoid
    leaking ``str(e)`` payloads that the OpenAI handler builds from
    raw exceptions (paths, tracebacks, prompt fragments, etc.).
    """
    status_code = getattr(response, "status_code", 500)
    body = getattr(response, "body", b"") or b""
    error_type = ERROR_TYPE_MAP.get(status_code, "api_error")

    upstream_message: Optional[str] = None
    try:
        payload = json.loads(body.decode("utf-8")) if body else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Non-JSON body (HTML gateway error, plain text, ...). Use a
        # bounded slice of the raw body so the client still has a
        # useful hint instead of a generic placeholder.
        try:
            upstream_message = body.decode("utf-8", errors="replace")[:500]
        except Exception:
            upstream_message = None
    else:
        if isinstance(payload, dict):
            error_payload = payload.get("error", payload)
            if isinstance(error_payload, dict):
                upstream_message = error_payload.get("message") or payload.get(
                    "message"
                )
                # Honor the upstream error.type only for 4xx; 5xx is
                # normalized below.
                if status_code < 500:
                    upstream_type = error_payload.get("type")
                    if isinstance(upstream_type, str) and upstream_type:
                        error_type = upstream_type
            elif isinstance(error_payload, str):
                upstream_message = error_payload
            elif isinstance(payload.get("message"), str):
                upstream_message = payload["message"]

    message = _scrub_error_message(upstream_message or "", status_code)
    return _build_error_response(
        status_code=status_code,
        error_type=error_type,
        message=message,
        raw_request=raw_request,
    )


def _cached_prompt_tokens(usage) -> int:
    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
    return getattr(prompt_tokens_details, "cached_tokens", 0) or 0


def _anthropic_input_tokens(usage) -> int:
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    cached = _cached_prompt_tokens(usage)
    if cached > prompt:
        # Upstream telemetry bug: cached cannot exceed the prompt it caches.
        # Clamping silently here would hide the discrepancy from billing
        # dashboards, so make it visible at WARNING level.
        logger.warning(
            "Cached tokens (%d) exceed prompt tokens (%d); clamping "
            "input_tokens to 0. This usually indicates an upstream "
            "telemetry bug.",
            cached,
            prompt,
        )
    return max(prompt - cached, 0)


def _anthropic_usage_from_openai(
    usage,
    *,
    include_input: bool,
    include_output: bool,
    force_zero_output: bool = False,
) -> AnthropicUsage:
    """Map an OpenAI ``Usage`` onto Anthropic's spec §3.3 usage object.

    ``cache_read_input_tokens`` surfaces whenever the OpenAI side carried
    it (the Anthropic adapter always opts each request into the
    request-scoped ``report_cached_tokens`` gate — G-25). The G-18
    ``output_tokens_details.thinking_tokens`` detail surfaces OpenAI
    ``reasoning_tokens`` (omitted entirely unless reasoning actually
    ran). ``service_tier`` is always ``"standard"`` — the only tier a
    local server has (G-10).
    """
    if usage is None:
        return AnthropicUsage(
            input_tokens=0 if include_input else None,
            output_tokens=0 if include_output else None,
        )

    usage_fields: dict[str, Any] = {}
    cached_tokens = _cached_prompt_tokens(usage)
    if include_input:
        usage_fields["input_tokens"] = _anthropic_input_tokens(usage)
        if cached_tokens:
            usage_fields["cache_read_input_tokens"] = cached_tokens
    if include_output:
        usage_fields["output_tokens"] = (
            0 if force_zero_output else (getattr(usage, "completion_tokens", 0) or 0)
        )
        # Spec §3.3 ``output_tokens_details.thinking_tokens`` — how many
        # output tokens were internal reasoning. The local backend reports
        # it as ``reasoning_tokens`` (defaults 0 off-reasoning, so this is
        # omitted entirely unless a reasoning run actually spent tokens).
        reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0
        if reasoning_tokens > 0:
            usage_fields["output_tokens_details"] = AnthropicOutputTokensDetails(
                thinking_tokens=reasoning_tokens
            )
    # G-10 (spec §3.3): local serving has exactly one service tier.
    usage_fields["service_tier"] = "standard"
    return AnthropicUsage(**usage_fields)


def convert_response(
    response: ChatCompletionResponse,
    stop_sequences: Optional[list[str]] = None,
) -> AnthropicMessagesResponse:
    """Convert an OpenAI ChatCompletionResponse to an Anthropic Messages response.

    ``stop_sequences`` is the request's own stop list — needed by
    ``_resolve_stop_reason`` to verify a matched-stop string before it
    may surface as ``stop_sequence`` (spec §3.1); omitting it keeps
    every matched string at plain ``end_turn`` (defensive default).
    """
    if not response.choices:
        return AnthropicMessagesResponse(
            content=[TextBlock(text="")],
            model=response.model,
            stop_reason="end_turn",
            usage=AnthropicUsage(input_tokens=0, output_tokens=0),
        )

    choice = response.choices[0]
    content: list[AnthropicContentBlock] = []

    # Add reasoning content as a thinking block. signature is omitted
    # entirely when the backend doesn't provide one — empty strings
    # would fail downstream Anthropic signature verifiers.
    if choice.message.reasoning_content:
        content.append(ThinkingBlock(thinking=choice.message.reasoning_content))

    # Add text content
    if choice.message.content:
        content.append(TextBlock(text=choice.message.content))

    # Add tool calls
    if choice.message.tool_calls:
        for tool_call in choice.message.tool_calls:
            raw_args = tool_call.function.arguments
            try:
                tool_input = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                # Surface invalid tool arguments so an empty-dict
                # tool call is never indistinguishable from a real
                # one when something downstream goes wrong.
                logger.warning(
                    "Tool %r emitted invalid JSON arguments: %r — "
                    "defaulting to empty input",
                    tool_call.function.name,
                    (raw_args or "")[:200],
                )
                tool_input = {}

            content.append(
                ToolUseBlock(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=tool_input,
                )
            )

    # Map stop reason. ``choice.matched_stop`` carries the scheduler's
    # stop signal: a VERIFIED str (member of the request's
    # ``stop_sequences``) means a stop string matched (→ Anthropic
    # stop_reason="stop_sequence" plus the matched string, spec §3.1);
    # an int is a stop token id (→ plain end_turn, no stop_sequence).
    # The helper owns the ordering + unmapped WARNING paths.
    stop_reason, stop_sequence = _resolve_stop_reason(
        choice.finish_reason,
        choice.matched_stop,
        frozenset(stop_sequences or ()),
    )

    # Anthropic requires ``content`` to contain at least one block.
    # Empty string completions (max_tokens=1 stop, content filter, etc.)
    # would otherwise ship ``content=[]`` and break strict SDK parsers.
    if not content:
        content.append(TextBlock(text=""))

    return AnthropicMessagesResponse(
        id=f"msg_{uuid.uuid4().hex}",
        content=content,
        model=response.model,
        stop_reason=stop_reason,
        stop_sequence=stop_sequence,
        usage=_anthropic_usage_from_openai(
            response.usage,
            include_input=True,
            include_output=True,
        ),
    )
