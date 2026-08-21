"""SSE translator for the Anthropic Messages dialect (spec §4).

Pure OpenAI-stream → Anthropic-event translation loop with layered G-21
keep-alive pings (static ping right after ``message_start``; shielded
idle watchdog on the upstream ``__anext__``) and G-22 terminal-frame
guarantees.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncGenerator, Optional, Union

from pydantic import ValidationError

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicError,
    AnthropicMessageEndDelta,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    ErrorEvent,
    InputJsonDelta,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    PingEvent,
    SignatureDelta,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
)
from sglang.srt.entrypoints.anthropic.respond import (
    ERROR_TYPE_MAP,
    _anthropic_usage_from_openai,
    _resolve_stop_reason,
)
from sglang.srt.entrypoints.openai.protocol import ChatCompletionStreamResponse

logger = logging.getLogger(__name__)


def _wrap_sse_event(data: str, event_type: str) -> str:
    """Format an Anthropic SSE event with event type and data lines."""
    return f"event: {event_type}\ndata: {data}\n\n"


class StreamTranslator:
    """OpenAI-stream → Anthropic-SSE translator (spec §4 event loop).

    Config is constructor-level; ``generate`` accepts a per-call
    ``ping_interval_seconds`` override so the serving layer keeps
    honoring ``serving._ping_interval_seconds`` overrides without
    constructor plumbing.
    """

    def __init__(self, ping_interval_seconds: float = 15.0):
        self.ping_interval_seconds = ping_interval_seconds

    async def generate(
        self,
        stream_iter,
        first_sse_line: Optional[str],
        anthropic_request: AnthropicMessagesRequest,
        ping_interval_seconds: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """Convert OpenAI chat stream to Anthropic event stream.

        The FIRST SSE line is pre-loaded by ``_handle_streaming``'s
        kick-start (G-22); ``None`` means the OpenAI stream exhausted with
        zero lines and only the terminal error frames should be emitted.
        """

        interval = (
            ping_interval_seconds
            if ping_interval_seconds is not None
            else self.ping_interval_seconds
        )
        content_block_index = 0
        content_block_open = False
        content_block_type: Optional[str] = None
        captured_thinking_signature: str = ""
        finish_reason: Optional[str] = None
        # ``matched_stop`` travels on the finish_reason chunk: str when a
        # stop STRING matched (→ Anthropic stop_sequence), int for a stop
        # token id (stays end_turn). See ``_resolve_stop_reason``.
        matched_stop: Union[None, int, str] = None
        final_usage: Optional[AnthropicUsage] = None
        message_started = False
        had_content_delta = False
        message_id = f"msg_{uuid.uuid4().hex}"
        model = anthropic_request.model

        def _message_start_event(usage) -> MessageStartEvent:
            return MessageStartEvent(
                message=AnthropicMessagesResponse(
                    id=message_id,
                    content=[],
                    model=model,
                    usage=_anthropic_usage_from_openai(
                        usage,
                        include_input=True,
                        include_output=True,
                        force_zero_output=True,
                    ),
                ),
            )

        def _emit(event: AnthropicStreamEvent) -> str:
            return _wrap_sse_event(
                event.model_dump_json(exclude_none=True),
                event.type,
            )

        def _close_content_block_events() -> list[AnthropicStreamEvent]:
            nonlocal content_block_index, content_block_open
            nonlocal content_block_type, captured_thinking_signature

            events: list[AnthropicStreamEvent] = []
            if not content_block_open:
                return events

            # Only emit signature_delta when a real signature is available.
            # Anthropic's spec treats absence as "unsigned thinking"; an
            # empty-string signature would fail downstream verifiers.
            if content_block_type == "thinking" and captured_thinking_signature:
                events.append(
                    ContentBlockDeltaEvent(
                        index=content_block_index,
                        delta=SignatureDelta(
                            signature=captured_thinking_signature,
                        ),
                    )
                )

            events.append(ContentBlockStopEvent(index=content_block_index))
            content_block_open = False
            content_block_type = None
            content_block_index += 1
            captured_thinking_signature = ""
            return events

        def _ensure_content_block_events(
            block_type: str,
            content_block: AnthropicContentBlock,
            force_new: bool = False,
        ) -> list[AnthropicStreamEvent]:
            """Open a content_block, closing the prior one if needed.

            ``force_new=True`` closes an existing block even when its type
            matches — required when a stream emits two consecutive
            ``tool_use`` blocks: each tool needs its own
            ``content_block_start``/``stop`` pair and its own
            ``content_block_index``, otherwise the second tool's
            ``input_json_delta`` chunks would append to the first tool's
            JSON arguments and corrupt both tool calls.
            """
            nonlocal content_block_open, content_block_type

            events: list[AnthropicStreamEvent] = []
            if content_block_open and (force_new or content_block_type != block_type):
                events.extend(_close_content_block_events())
            if not content_block_open:
                events.append(
                    ContentBlockStartEvent(
                        index=content_block_index,
                        content_block=content_block,
                    )
                )
                content_block_open = True
                content_block_type = block_type
            return events

        def _ensure_message_started(usage) -> list[str]:
            """Emit message_start exactly once. Returns SSE frames to yield."""
            nonlocal message_started
            if message_started:
                return []
            message_started = True
            return [_emit(_message_start_event(usage))]

        def _build_error_event(error_type: str, message: str) -> ErrorEvent:
            return ErrorEvent(
                error=AnthropicError(type=error_type, message=message),
            )

        def _flush_on_error(error_type: str, message: str) -> list[str]:
            """Build a self-contained terminal SSE sequence on error.

            Guarantees that whatever events we emit on the failure path
            leave the wire in a valid state: message_start (if not yet
            sent), close any open content block, then ErrorEvent and
            MessageStopEvent. Strict SDK clients reject streams whose
            content_block_start has no matching content_block_stop, so
            the close step is mandatory even on the error path.
            """
            frames: list[str] = []
            frames.extend(_ensure_message_started(None))
            for event in _close_content_block_events():
                frames.append(_emit(event))
            frames.append(_emit(_build_error_event(error_type, message)))
            frames.append(_emit(MessageStopEvent()))
            return frames

        def _parse_upstream_error(data_str: str) -> Optional[tuple[str, str]]:
            """Detect an OpenAI handler streaming-error envelope.

            ``OpenAIServingChat.create_streaming_error_response`` emits
            ``data: {"error": {"object":"error","message":"...",
            "type":"BadRequestError","code":400}}``; the regular
            ChatCompletionStreamResponse validator rejects it. Pull the
            type/message out so the Anthropic client sees the real
            failure instead of a generic 'Stream processing error'.
            """
            try:
                payload = json.loads(data_str)
            except (json.JSONDecodeError, ValueError):
                return None
            if not isinstance(payload, dict):
                return None
            err = payload.get("error")
            if not isinstance(err, dict):
                return None
            upstream_message = err.get("message") or "Upstream error"
            code = err.get("code")
            error_type = (
                ERROR_TYPE_MAP.get(code, "api_error")
                if isinstance(code, int)
                else "api_error"
            )
            return error_type, str(upstream_message)

        # The OpenAI stream exhausted with zero lines (kick-start found a
        # StopAsyncIteration): emit the same self-contained terminal error
        # sequence a [DONE]-without-content would produce, then stop.
        if first_sse_line is None:
            for frame in _flush_on_error("api_error", "Backend produced no content"):
                yield frame
            return

        pending = first_sse_line
        # G-21 watchdog: the pending ``__anext__`` task must survive the
        # timeout (cancelling it would inject GeneratorExit into the
        # upstream async generator and DESTROY the stream after the first
        # ping). One long-lived task, shielded per wait, is created once
        # and reused until it yields a line.
        anext_task: Optional[asyncio.Task] = None
        while True:
            if pending is not None:
                sse_line = pending
                pending = None
            else:
                try:
                    if anext_task is None:
                        anext_task = asyncio.ensure_future(stream_iter.__anext__())
                    sse_line = await asyncio.wait_for(
                        asyncio.shield(anext_task),
                        timeout=interval,
                    )
                    anext_task = None
                except asyncio.TimeoutError:
                    # No OpenAI chunk inside the watchdog interval — emit
                    # a keep-alive ping (spec §4.2: pings may be
                    # interleaved anywhere) so intermediaries with short
                    # idle timeouts don't drop long-prefill requests.
                    yield _emit(PingEvent())
                    continue
                except StopAsyncIteration:
                    anext_task = None
                    break
                except asyncio.CancelledError:
                    # Client disconnect / task shutdown: also stop the
                    # upstream read instead of leaving a pending task.
                    if anext_task is not None:
                        anext_task.cancel()
                    raise
                except ValueError as e:
                    # Mid-flight ValueError (pre-first-chunk ones are caught
                    # earlier by the G-22 kick-start) — surface as a proper
                    # Anthropic error event rather than aborting the
                    # StreamingResponse generator.
                    logger.warning("OpenAI stream raised ValueError: %s", e)
                    for frame in _flush_on_error(
                        "invalid_request_error", str(e) or "Request failed"
                    ):
                        yield frame
                    return
                except Exception as e:
                    logger.exception("OpenAI stream raised mid-flight: %s", e)
                    for frame in _flush_on_error(
                        "api_error", "Internal server error"
                    ):
                        yield frame
                    return

            if not sse_line.startswith("data: "):
                continue

            data_str = sse_line[6:].strip()

            if data_str == "[DONE]":
                for frame in _ensure_message_started(None):
                    yield frame

                # No content AND no finish_reason: the backend dropped the
                # stream silently. Surface as api_error so clients see the
                # failure instead of a fake empty success. If finish_reason
                # IS set we trust the backend's signal — a legitimate empty
                # completion (max_tokens=1 stop, content filter, etc.)
                # deserves a normal message_delta/message_stop pair, not
                # an error that triggers SDK retry loops.
                if not had_content_delta and finish_reason is None:
                    logger.warning(
                        "Stream produced no content and no finish_reason "
                        "before [DONE]; emitting api_error event"
                    )
                    yield _emit(
                        _build_error_event("api_error", "Backend produced no content")
                    )
                    yield _emit(MessageStopEvent())
                    continue

                # Close any open content block
                for event in _close_content_block_events():
                    yield _emit(event)

                # Emit message_delta with stop_reason and usage. A string
                # ``matched_stop`` upgrades the finish to Anthropic's
                # ``stop_sequence`` pair (spec §3.1); token-id matches and
                # the plain finish map are handled inside the helper.
                stop_reason, stop_sequence = _resolve_stop_reason(
                    finish_reason,
                    matched_stop,
                    frozenset(anthropic_request.stop_sequences or ()),
                )
                yield _emit(
                    MessageDeltaEvent(
                        delta=AnthropicMessageEndDelta(
                            stop_reason=stop_reason,
                            stop_sequence=stop_sequence,
                        ),
                        usage=final_usage or AnthropicUsage(output_tokens=0),
                    )
                )

                yield _emit(MessageStopEvent())
                continue

            # Parse the OpenAI chunk
            try:
                chunk = ChatCompletionStreamResponse.model_validate_json(data_str)
            except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
                # First check whether this is the OpenAI handler's
                # streaming error envelope (validator rejects it because
                # it lacks id/choices/created/model). Forwarding the real
                # type/message keeps the failure debuggable instead of
                # collapsing every backend error into "Stream processing
                # error".
                upstream = _parse_upstream_error(data_str)
                if upstream is not None:
                    error_type, error_message = upstream
                    logger.warning(
                        "Forwarding upstream stream error (%s): %s",
                        error_type,
                        error_message,
                    )
                    for frame in _flush_on_error(error_type, error_message):
                        yield frame
                    return

                logger.warning(
                    "Failed to parse Anthropic stream chunk (%s): %s",
                    type(e).__name__,
                    data_str[:200],
                )
                for frame in _flush_on_error("api_error", "Stream processing error"):
                    yield frame
                return

            if chunk.usage is not None:
                final_usage = _anthropic_usage_from_openai(
                    chunk.usage,
                    include_input=False,
                    include_output=True,
                )

            # Usage-only chunk (empty choices with usage info)
            if not chunk.choices and chunk.usage:
                continue

            if not chunk.choices:
                continue

            choice = chunk.choices[0]

            # Capture finish_reason on this chunk but DO NOT short-circuit:
            # some OpenAI-compatible backends pack the final content token
            # (or last tool-args fragment) into the same chunk as
            # finish_reason. Skipping delta processing would silently drop
            # that payload — sometimes the whole completion if it was a
            # one-token reply. Fall through to the delta handlers below.
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
                # ``matched_stop`` ships on the same finish chunk; hold it
                # so the [DONE] finalization can report a stop_sequence.
                matched_stop = choice.matched_stop

            delta = choice.delta

            # Defer message_start until the first chunk carrying real prompt
            # usage or content. OpenAI streams emit a role-only chunk before
            # usage is available; emitting message_start there would ship
            # input_tokens=0 to the client.
            has_delta_payload = bool(
                delta.reasoning_content
                or delta.tool_calls
                or (delta.content is not None and delta.content != "")
                or chunk.usage
            )
            # The finish_reason chunk should also flip message_started so a
            # zero-content completion (the path that previously fired the
            # 'Backend produced no content' error) emits the standard
            # message_start before [DONE] closes the stream.
            if (
                has_delta_payload or choice.finish_reason is not None
            ) and not message_started:
                yield _emit(_message_start_event(chunk.usage))
                message_started = True
                # Static start-of-stream ping (G-21 layer 1): an early
                # keep-alive for idle-timeout proxies. Stall-time pings
                # come from the watchdog in the read loop.
                yield _emit(PingEvent())

            if (
                not has_delta_payload
                and delta.role == "assistant"
                and (delta.content is None or delta.content == "")
            ):
                continue

            # Handle reasoning content deltas
            if delta.reasoning_content:
                for event in _ensure_content_block_events(
                    "thinking",
                    ThinkingBlock(thinking=""),
                ):
                    yield _emit(event)

                yield _emit(
                    ContentBlockDeltaEvent(
                        index=content_block_index,
                        delta=ThinkingDelta(thinking=delta.reasoning_content),
                    )
                )
                had_content_delta = True

            # Handle tool call deltas
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    tc_id = tc.id
                    tc_func = tc.function

                    # New tool call: always close the previous block (even if
                    # it was also tool_use — each tool needs its own index)
                    # and start a fresh one.
                    if tc_func and tc_func.name:
                        for event in _ensure_content_block_events(
                            "tool_use",
                            ToolUseBlock(
                                id=tc_id or f"toolu_{uuid.uuid4().hex}",
                                name=tc_func.name,
                                input={},
                            ),
                            force_new=True,
                        ):
                            yield _emit(event)
                        # A zero-argument tool call may never emit an
                        # input_json_delta; the tool_use start block itself is
                        # still meaningful content because it carries id/name.
                        had_content_delta = True

                        if tc_func.arguments:
                            yield _emit(
                                ContentBlockDeltaEvent(
                                    index=content_block_index,
                                    delta=InputJsonDelta(
                                        partial_json=tc_func.arguments,
                                    ),
                                )
                            )
                            had_content_delta = True

                    elif tc_func and tc_func.arguments:
                        # Continuing arguments for current tool call
                        if content_block_type != "tool_use":
                            logger.warning(
                                "Dropping tool_call argument delta with no "
                                "open tool_use block: %r",
                                (tc_func.arguments or "")[:100],
                            )
                            continue
                        yield _emit(
                            ContentBlockDeltaEvent(
                                index=content_block_index,
                                delta=InputJsonDelta(
                                    partial_json=tc_func.arguments,
                                ),
                            )
                        )
                        had_content_delta = True

            # Handle text content deltas
            if delta.content is not None and delta.content != "":
                for event in _ensure_content_block_events(
                    "text",
                    TextBlock(text=""),
                ):
                    yield _emit(event)

                yield _emit(
                    ContentBlockDeltaEvent(
                        index=content_block_index,
                        delta=TextDelta(text=delta.content),
                    )
                )
                had_content_delta = True
