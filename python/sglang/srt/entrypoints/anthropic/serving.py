"""Handler for Anthropic Messages API requests.

Converts Anthropic requests to OpenAI ChatCompletion format, delegates to
OpenAIServingChat for processing, and converts responses back to Anthropic format.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional, Union

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
)
from sglang.srt.entrypoints.anthropic.convert import (
    ConversionContext,
    convert_to_chat_completion_request,
)
from sglang.srt.entrypoints.anthropic.respond import (
    _build_error_response,
    _convert_openai_error_response,
    _scrub_error_message,
    convert_response,
)

from sglang.srt.entrypoints.anthropic.streaming import StreamTranslator
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from sglang.srt.observability.req_time_stats import monotonic_time
from sglang.srt.parser.template_detection import detect_inline_system_support

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

logger = logging.getLogger(__name__)

class AnthropicServing:
    """Handler for Anthropic Messages API requests.

    Acts as a translation layer between Anthropic's Messages API and SGLang's
    OpenAI-compatible chat completion infrastructure.
    """

    def __init__(self, openai_serving_chat: OpenAIServingChat):
        self.openai_serving_chat = openai_serving_chat
        self._merge_inline_system = not detect_inline_system_support(
            self._chat_template()
        )
        # G-21 (spec §4.2, spec §0 item "ping"): Anthropic streams ping
        # keep-alives anywhere in the stream. Two layers: (1) a static
        # ping right after the first message_start so proxies see early
        # bytes; (2) an idle watchdog that emits a ping whenever the OpenAI
        # stream stalls past this interval (long prefill, scheduling
        # pressure). Kept as an attribute so tests can shrink it.
        self._ping_interval_seconds = 15.0
        self._translator = StreamTranslator(self._ping_interval_seconds)

    def _chat_template(self) -> Optional[str]:
        tokenizer_manager = getattr(self.openai_serving_chat, "tokenizer_manager", None)
        if tokenizer_manager is None:
            return None
        tokenizer = getattr(tokenizer_manager, "tokenizer", None)
        if tokenizer is None:
            return None
        return getattr(tokenizer, "chat_template", None)

    async def handle_messages(
        self,
        request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> Union[JSONResponse, StreamingResponse]:
        """Main entry point for /v1/messages endpoint."""
        try:
            chat_request = self._convert_to_chat_completion_request(request)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error converting Anthropic request: %s", e)
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=str(e),
                raw_request=raw_request,
            )

        if request.stream:
            return await self._handle_streaming(chat_request, request, raw_request)
        else:
            return await self._handle_non_streaming(chat_request, request, raw_request)

    def _convert_to_chat_completion_request(
        self, anthropic_request: AnthropicMessagesRequest
    ) -> ChatCompletionRequest:
        """OpenAI-wire conversion pipeline (lives in ``anthropic/convert.py``)."""
        return convert_to_chat_completion_request(
            anthropic_request,
            ConversionContext(
                merge_inline_system=self._merge_inline_system,
                wrap_reasoning_history=self.openai_serving_chat.wrap_reasoning_history,
                apply_reasoning_enabled=self.openai_serving_chat.apply_reasoning_enabled,
            ),
        )

    async def _handle_non_streaming(
        self,
        chat_request: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> JSONResponse:
        """Handle non-streaming Anthropic request by delegating to OpenAI handler."""
        # ``monotonic_time`` is ``time.perf_counter`` under the hood; the
        # downstream stats layer subtracts other ``perf_counter`` samples
        # from this, so they must come from the same clock.
        received_time = monotonic_time()

        # Validate
        error_msg = self.openai_serving_chat._validate_request(chat_request)
        if error_msg:
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=error_msg,
                raw_request=raw_request,
            )

        try:
            # Convert to internal request
            adapted_request, processed_request = (
                self.openai_serving_chat._convert_to_internal_request(
                    chat_request, raw_request
                )
            )
            adapted_request.received_time = received_time

            # Get response from OpenAI handler
            response = await self.openai_serving_chat._handle_non_streaming_request(
                adapted_request, processed_request, raw_request
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error processing Anthropic request: %s", e)
            return self._error_response(
                status_code=500,
                error_type="api_error",
                message="Internal server error",
                exception_name=type(e).__name__,
                raw_request=raw_request,
            )

        # Check for error responses from OpenAI handler
        if not isinstance(response, ChatCompletionResponse):
            # It's an error response (ORJSONResponse)
            return self._convert_openai_error_response(response, raw_request)

        # Convert to Anthropic response
        anthropic_response = self._convert_response(
            response, anthropic_request.stop_sequences
        )
        return JSONResponse(content=anthropic_response.model_dump(exclude_none=True))

    async def _handle_streaming(
        self,
        chat_request: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, JSONResponse]:
        """Handle streaming Anthropic request."""
        received_time = monotonic_time()

        # Validate
        error_msg = self.openai_serving_chat._validate_request(chat_request)
        if error_msg:
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=error_msg,
                raw_request=raw_request,
            )

        try:
            adapted_request, processed_request = (
                self.openai_serving_chat._convert_to_internal_request(
                    chat_request, raw_request
                )
            )
            adapted_request.received_time = received_time
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error converting streaming request: %s", e)
            return self._error_response(
                status_code=500,
                error_type="api_error",
                message="Internal server error",
                exception_name=type(e).__name__,
                raw_request=raw_request,
            )

        # G-22: pre-200 validation. Kick-start the OpenAI stream BEFORE
        # committing to HTTP 200: errors that surface on the FIRST
        # generator step (invalid-token ValueError from tokenization,
        # multi-modal count mismatches, context-length overflow, …)
        # become a real 400 envelope instead of arriving as an in-band SSE
        # error on an already-200 response — SDK clients must see the
        # correct status for retry/routing logic (notes A.2).
        openai_stream = self.openai_serving_chat._generate_chat_stream(
            adapted_request, processed_request, raw_request
        )
        try:
            stream_iter = openai_stream.__aiter__()
            first_sse_line = await stream_iter.__anext__()
        except ValueError as e:
            logger.warning(
                "Pre-stream validation failed for Anthropic request: %s", e
            )
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=_scrub_error_message(str(e), 400),
                raw_request=raw_request,
            )
        except StopAsyncIteration:
            # The OpenAI generator completed with zero lines — nothing
            # would ever reach the client. Degrade to a terminal error
            # stream rather than a half-open SSE.
            first_sse_line = None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error opening OpenAI stream: %s", e)
            return self._error_response(
                status_code=500,
                error_type="api_error",
                message="Internal server error",
                exception_name=type(e).__name__,
                raw_request=raw_request,
            )

        return StreamingResponse(
            self._generate_anthropic_stream(
                stream_iter,
                first_sse_line,
                anthropic_request,
            ),
            media_type="text/event-stream",
            background=self.openai_serving_chat.tokenizer_manager.create_abort_task(
                adapted_request
            ),
        )

    def _generate_anthropic_stream(
        self,
        stream_iter,
        first_sse_line: Optional[str],
        anthropic_request: AnthropicMessagesRequest,
    ) -> AsyncGenerator[str, None]:
        """Factory returning the translated SSE async-generator (loop in
        ``anthropic/streaming.StreamTranslator``). NOT a def-async wrap so
        call sites keep receiving the async generator itself. The ping
        interval is read HERE so post-construction overrides
        (``serving._ping_interval_seconds = ...``) keep working."""
        return self._translator.generate(
            stream_iter,
            first_sse_line,
            anthropic_request,
            ping_interval_seconds=self._ping_interval_seconds,
        )

    def _convert_response(
        self,
        response: ChatCompletionResponse,
        stop_sequences: Optional[list[str]] = None,
    ) -> AnthropicMessagesResponse:
        """Delegate: response shaping lives in ``anthropic/respond.py``."""
        return convert_response(response, stop_sequences=stop_sequences)

    def _convert_openai_error_response(self, response, raw_request=None) -> JSONResponse:
        """Delegate: upstream error forwarding lives in ``anthropic/respond.py``."""
        return _convert_openai_error_response(response, raw_request)

    def _error_response(
        self,
        status_code: int,
        error_type: str,
        message: str,
        exception_name: Optional[str] = None,
        raw_request=None,
    ) -> JSONResponse:
        """Delegate: Anthropic error envelopes live in ``anthropic/respond.py``."""
        return _build_error_response(
            status_code=status_code,
            error_type=error_type,
            message=message,
            exception_name=exception_name,
            raw_request=raw_request,
        )

    async def handle_count_tokens(
        self,
        request: AnthropicCountTokensRequest,
        raw_request: Request,
    ) -> JSONResponse:
        """Handle /v1/messages/count_tokens endpoint.

        Converts the request to a ChatCompletionRequest, applies the chat
        template via the OpenAI handler to tokenize, and returns the count.
        """
        try:
            # Build a minimal AnthropicMessagesRequest so we can reuse conversion
            messages_request = AnthropicMessagesRequest(
                model=request.model,
                messages=request.messages,
                max_tokens=1,  # dummy, not used for counting
                system=request.system,
                thinking=request.thinking,
                tools=request.tools,
                tool_choice=request.tool_choice,
            )
            chat_request = self._convert_to_chat_completion_request(messages_request)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error converting count_tokens request: %s", e)
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=str(e),
                raw_request=raw_request,
            )

        try:
            is_multimodal = (
                self.openai_serving_chat.tokenizer_manager.model_config.is_multimodal
            )
            processed = self.openai_serving_chat._process_messages(
                chat_request, is_multimodal
            )

            if isinstance(processed.prompt_ids, list):
                input_tokens = len(processed.prompt_ids)
            else:
                # prompt_ids is a string (multimodal case) — tokenize it
                tokenizer = self.openai_serving_chat.tokenizer_manager.tokenizer
                input_tokens = len(tokenizer.encode(processed.prompt_ids))

            return JSONResponse(
                content=AnthropicCountTokensResponse(
                    input_tokens=input_tokens
                ).model_dump()
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error counting tokens: %s", e)
            return self._error_response(
                status_code=500,
                error_type="api_error",
                message="Internal server error",
                exception_name=type(e).__name__,
                raw_request=raw_request,
            )
