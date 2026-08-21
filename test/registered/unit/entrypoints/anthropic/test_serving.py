import asyncio
import json
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from fastapi.responses import JSONResponse  # noqa: E402
from jinja2 import Environment  # noqa: E402

from sglang.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicMessage,
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import (  # noqa: E402
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from sglang.srt.parser.template_detection import (  # noqa: E402
    detect_inline_system_support,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeOpenAIServingChat:
    def __init__(self, stream_lines=None, chat_template=None, stall_after_first_line=0):
        self.stream_lines = stream_lines or []
        # G-21 test hook: sleep AFTER the first yielded line so the
        # adapter's ping watchdog can fire mid-stream.
        self.stall_after_first_line = stall_after_first_line
        self.apply_reasoning_calls: list[bool] = []
        self.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template=chat_template)
        )

    def _generate_chat_stream(self, adapted_request, processed_request, raw_request):
        async def _gen():
            for i, line in enumerate(self.stream_lines):
                yield line
                if i == 0 and self.stall_after_first_line:
                    await asyncio.sleep(self.stall_after_first_line)

        return _gen()

    def apply_reasoning_enabled(self, chat_request, enabled):
        self.apply_reasoning_calls.append(enabled)

    def wrap_reasoning_history(self, text):
        return f"<think>\n{text}\n</think>"


class _FakeNonStreamingErrorOpenAI:
    """Returns a configurable error response from the OpenAI handler."""

    def __init__(self, status_code=400, body=None, content=None):
        self._status_code = status_code
        self._body = body
        self._content = content

    def _validate_request(self, chat_request):
        return None

    def _convert_to_internal_request(self, chat_request, raw_request):
        return SimpleNamespace(), chat_request

    async def _handle_non_streaming_request(
        self, adapted_request, processed_request, raw_request
    ):
        if self._body is not None:
            # Build a response object exposing raw bytes via `.body`.
            return SimpleNamespace(status_code=self._status_code, body=self._body)
        return JSONResponse(
            status_code=self._status_code,
            content=self._content
            or {
                "error": {
                    "type": "invalid_request_error",
                    "message": "context length exceeded",
                }
            },
        )


class _FakeNonStreamingOpenAI:
    """Returns a configurable ChatCompletionResponse from the OpenAI handler."""

    def __init__(self, response):
        self._response = response
        self.apply_reasoning_calls = []

    def _validate_request(self, chat_request):
        return None

    def _convert_to_internal_request(self, chat_request, raw_request):
        return SimpleNamespace(), chat_request

    def apply_reasoning_enabled(self, chat_request, enabled):
        self.apply_reasoning_calls.append(enabled)

    def wrap_reasoning_history(self, text):
        return f"<think>\n{text}\n</think>"

    async def _handle_non_streaming_request(
        self, adapted_request, processed_request, raw_request
    ):
        return self._response


def _chunk(choices=None, usage=None):
    data = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": choices or [],
    }
    if usage is not None:
        data["usage"] = usage
    return f"data: {json.dumps(data)}\n\n"


def _choice(delta, finish_reason=None, matched_stop=None):
    return {
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason,
        # ``matched_stop`` mirrors the OpenAI stream choice field: str for a
        # matched stop string, int for a matched stop token id, None if the
        # finish was not stop-matched. ``None`` round-trips through JSON as
        # null and is a valid value for the protocol's Union type.
        "matched_stop": matched_stop,
    }


async def _collect_anthropic_events(serving, anthropic_request):
    """Drive the Anthropic stream generator the same way production does
    (G-22): the OpenAI stream is kick-started OUTSIDE the generator and
    the first SSE line is handed in pre-loaded."""
    openai_stream = serving.openai_serving_chat._generate_chat_stream(
        object(), object(), object()
    )
    stream_iter = openai_stream.__aiter__()
    try:
        first_sse_line = await stream_iter.__anext__()
    except StopAsyncIteration:
        first_sse_line = None
    events = []
    async for sse in serving._generate_anthropic_stream(
        stream_iter,
        first_sse_line,
        anthropic_request,
    ):
        for line in sse.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


class TestAnthropicServing(unittest.TestCase):
    # Renders system at any position (GLM/Kimi/Qwen3) → can pass through.
    INLINE_SYSTEM_TEMPLATE = (
        "{%- for message in messages %}"
        "{{- message.role }}: {{ message.content }}\n"
        "{%- endfor %}"
    )
    GLM_TOOL_RESULT_TEMPLATE = """
{%- for message in messages if message.role == "tool" -%}
{%- if loop.first -%}<|observation|>{%- endif -%}
{%- if message.content is string -%}
<tool_response>{{ message.content }}</tool_response>
{%- elif message.content.0.type == "tool_reference" -%}
<tool_response><tools>
{%- for reference in message.content -%}
{%- for tool in tools if tool.function.name == reference.name -%}
{{ tool.function.name }}
{%- endfor -%}
{%- endfor -%}
</tools></tool_response>
{%- endif -%}
{%- endfor -%}
"""

    def _serving(self, stream_lines=None, chat_template=None, stall_after_first_line=0):
        return AnthropicServing(
            _FakeOpenAIServingChat(stream_lines, chat_template, stall_after_first_line)
        )

    def _anthropic_request(self, **overrides):
        data = {
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
        data.update(overrides)
        return AnthropicMessagesRequest.model_validate(data)

    def _tool_result_request(self, content, tools=None):
        overrides = {
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": content,
                        }
                    ],
                }
            ],
        }
        if tools is not None:
            overrides["tools"] = tools
        return self._anthropic_request(**overrides)

    def test_stream_closes_tool_block_before_text_delta(self):
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk(
                    [
                        _choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": '{"query"',
                                        },
                                    }
                                ]
                            }
                        )
                    ]
                ),
                _chunk(
                    [
                        _choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "type": "function",
                                        "function": {"arguments": ': "sglang"}'},
                                    }
                                ]
                            }
                        )
                    ]
                ),
                _chunk([_choice({"content": "done"})]),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )

        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        block_events = [
            (event["type"], event.get("content_block", {}).get("type"))
            for event in events
            if event["type"].startswith("content_block")
        ]

        self.assertEqual(
            block_events,
            [
                ("content_block_start", "tool_use"),
                ("content_block_delta", None),
                ("content_block_delta", None),
                ("content_block_stop", None),
                ("content_block_start", "text"),
                ("content_block_delta", None),
                ("content_block_stop", None),
            ],
        )

        text_delta = [
            event
            for event in events
            if event["type"] == "content_block_delta"
            and event["delta"].get("type") == "text_delta"
        ][0]
        self.assertEqual(text_delta["index"], 1)

    def test_stream_reasoning_content_uses_thinking_block(self):
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk([_choice({"reasoning_content": "think first"})]),
                _chunk([_choice({"content": "answer"})]),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )

        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        content_events = [
            event for event in events if event["type"].startswith("content_block")
        ]

        self.assertEqual(content_events[0]["content_block"]["type"], "thinking")
        # Signature is absent (None and excluded) — never emit empty
        # string, which would fail downstream Anthropic signature verifiers.
        self.assertNotIn("signature", content_events[0]["content_block"])
        self.assertEqual(content_events[1]["delta"]["type"], "thinking_delta")
        self.assertEqual(content_events[1]["delta"]["thinking"], "think first")
        # No empty signature_delta event between thinking_delta and content_block_stop.
        self.assertEqual(content_events[2]["type"], "content_block_stop")
        self.assertEqual(content_events[3]["content_block"]["type"], "text")
        # Confirm no signature_delta event was emitted in the entire stream.
        sig_deltas = [
            event
            for event in events
            if event["type"] == "content_block_delta"
            and event.get("delta", {}).get("type") == "signature_delta"
        ]
        self.assertEqual(sig_deltas, [])

    def test_stream_usage_subtracts_cache_read_and_omits_final_input_tokens(self):
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 0,
            "total_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
        final_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk([_choice({"content": "hi"})], usage=usage),
                _chunk([], usage=final_usage),
                "data: [DONE]\n\n",
            ]
        )

        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        message_start = [event for event in events if event["type"] == "message_start"][
            0
        ]
        message_delta = [event for event in events if event["type"] == "message_delta"][
            0
        ]

        self.assertEqual(message_start["message"]["usage"]["input_tokens"], 6)
        self.assertEqual(
            message_start["message"]["usage"]["cache_read_input_tokens"], 4
        )
        self.assertNotIn("input_tokens", message_delta["usage"])
        self.assertEqual(message_delta["usage"]["output_tokens"], 2)

    def test_non_streaming_usage_subtracts_cache_read_tokens(self):
        response = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            }
        )

        anthropic_response = self._serving()._convert_response(response)

        self.assertEqual(anthropic_response.usage.input_tokens, 6)
        self.assertEqual(anthropic_response.usage.output_tokens, 2)
        self.assertEqual(anthropic_response.usage.cache_read_input_tokens, 4)

    def test_tool_result_search_result_content_is_flattened(self):
        request = AnthropicMessagesRequest.model_validate(
            {
                "model": "test-model",
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call_1",
                                "content": [
                                    {
                                        "type": "search_result",
                                        "title": "SGLang docs",
                                        "source": "https://docs.sglang.ai",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": "Anthropic API notes",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )

        chat_request = self._serving()._convert_to_chat_completion_request(request)
        tool_message = [
            msg
            for msg in chat_request.model_dump()["messages"]
            if msg["role"] == "tool"
        ][0]

        self.assertIn("SGLang docs", tool_message["content"])
        self.assertIn("https://docs.sglang.ai", tool_message["content"])
        self.assertIn("Anthropic API notes", tool_message["content"])

    def test_mixed_tool_reference_content_preserves_part_order(self):
        request = self._tool_result_request(
            [
                {"type": "text", "text": "Tool loaded: Bash"},
                {"type": "tool_reference", "tool_name": "Bash"},
                {"type": "text", "text": "Ready"},
            ]
        )

        chat_request = self._serving()._convert_to_chat_completion_request(request)
        messages = chat_request.model_dump(exclude_none=True)["messages"]

        self.assertEqual([message["role"] for message in messages], ["tool"] * 3)
        self.assertEqual(
            [message["tool_call_id"] for message in messages], ["call_1"] * 3
        )
        self.assertEqual(messages[0]["content"], "Tool loaded: Bash")
        self.assertEqual(
            messages[1]["content"],
            [{"type": "tool_reference", "name": "Bash"}],
        )
        self.assertEqual(messages[2]["content"], "Ready")

    def test_mixed_tool_reference_content_renders_text_and_schema(self):
        template = Environment().from_string(self.GLM_TOOL_RESULT_TEMPLATE)
        request = self._tool_result_request(
            [
                {"type": "text", "text": "Tool loaded: Bash"},
                {"type": "tool_reference", "tool_name": "Bash"},
            ],
            tools=[
                {
                    "name": "Bash",
                    "description": "Run a shell command",
                    "input_schema": {"type": "object", "properties": {}},
                    "defer_loading": True,
                }
            ],
        )

        chat_request = self._serving()._convert_to_chat_completion_request(request)
        payload = chat_request.model_dump(exclude_none=True)
        prompt = template.render(messages=payload["messages"], tools=payload["tools"])

        self.assertIn("<tool_response>Tool loaded: Bash</tool_response>", prompt)
        self.assertIn("<tools>Bash</tools>", prompt)
        self.assertEqual(prompt.count("<|observation|>"), 1)

    def test_reference_only_tool_result_remains_one_message(self):
        request = self._tool_result_request(
            [
                {"type": "tool_reference", "tool_name": "Bash"},
                {"type": "tool_reference", "tool_name": "Read"},
            ]
        )

        chat_request = self._serving()._convert_to_chat_completion_request(request)
        messages = chat_request.model_dump(exclude_none=True)["messages"]

        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0]["content"],
            [
                {"type": "tool_reference", "name": "Bash"},
                {"type": "tool_reference", "name": "Read"},
            ],
        )

    def test_builtin_web_search_tool_without_schema_is_skipped(self):
        request = AnthropicMessagesRequest.model_validate(
            {
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "search sglang"}],
                "tools": [{"name": "web_search", "type": "web_search_20250305"}],
                "tool_choice": {"type": "auto"},
            }
        )

        chat_request = self._serving()._convert_to_chat_completion_request(request)

        self.assertIsNone(chat_request.tools)
        self.assertEqual(chat_request.tool_choice, "none")

    def test_custom_tool_without_schema_is_rejected(self):
        # With the discriminated union, an AnthropicCustomTool variant must
        # carry an input_schema. The check fires at request-parse time
        # (Pydantic raises ValidationError, a subclass of ValueError).
        with self.assertRaisesRegex(ValueError, "input_schema"):
            AnthropicMessagesRequest.model_validate(
                {
                    "model": "test-model",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "call a tool"}],
                    "tools": [{"name": "custom_without_schema"}],
                }
            )

    def test_non_streaming_openai_error_response_is_forwarded(self):
        serving = AnthropicServing(_FakeNonStreamingErrorOpenAI())
        chat_request = ChatCompletionRequest(
            model="test-model",
            max_tokens=16,
            messages=[{"role": "user", "content": "hello"}],
        )
        anthropic_request = self._anthropic_request(stream=False)

        response = asyncio.run(
            serving._handle_non_streaming(chat_request, anthropic_request, object())
        )
        payload = json.loads(response.body)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(payload["error"]["message"], "context length exceeded")

    # ------------------------------------------------------------------
    # Edge-case coverage added in the review-fix pass
    # ------------------------------------------------------------------

    def test_stream_text_then_tool_use_closes_text_block(self):
        """Text deltas followed by tool_use must close the text block before opening tool_use index."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": "Hello"})]),
                _chunk([_choice({"content": " world"})]),
                _chunk(
                    [
                        _choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "search",
                                            "arguments": '{"q":"hello"}',
                                        },
                                    }
                                ]
                            }
                        )
                    ]
                ),
                _chunk([_choice({}, finish_reason="tool_calls")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        block_events = [
            (event["type"], event.get("content_block", {}).get("type"))
            for event in events
            if event["type"].startswith("content_block")
        ]
        # text block (start, delta, delta, stop) then tool_use (start, delta, stop)
        self.assertEqual(block_events[0], ("content_block_start", "text"))
        text_stop_idx = next(
            i for i, ev in enumerate(block_events) if ev == ("content_block_stop", None)
        )
        tool_start_idx = next(
            i
            for i, ev in enumerate(block_events)
            if ev == ("content_block_start", "tool_use")
        )
        self.assertLess(text_stop_idx, tool_start_idx)

    def test_stream_tool_use_without_arguments_is_not_empty_completion(self):
        """A zero-argument tool call is valid content even without input_json_delta."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk(
                    [
                        _choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "ping", "arguments": ""},
                                    }
                                ]
                            }
                        )
                    ]
                ),
                _chunk([_choice({}, finish_reason="tool_calls")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )

        self.assertFalse(any(event["type"] == "error" for event in events))
        tool_start = next(
            event
            for event in events
            if event["type"] == "content_block_start"
            and event["content_block"]["type"] == "tool_use"
        )
        self.assertEqual(tool_start["content_block"]["name"], "ping")
        message_delta = next(
            event for event in events if event["type"] == "message_delta"
        )
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")

    def test_stream_no_usage_chunk_emits_error_event(self):
        """Stream that yields only [DONE] (no content delta) must surface as an error event."""
        serving = self._serving(["data: [DONE]\n\n"])
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [event["type"] for event in events]
        # Sequence: message_start, error, message_stop
        self.assertEqual(types, ["message_start", "error", "message_stop"])
        error_event = events[1]
        self.assertEqual(error_event["error"]["type"], "api_error")
        self.assertIn("no content", error_event["error"]["message"].lower())

    def test_cache_read_exceeds_prompt_tokens_clamps_to_zero(self):
        """When cached_tokens > prompt_tokens, input_tokens clamps to 0 instead of going negative."""
        usage = {
            "prompt_tokens": 4,
            "completion_tokens": 0,
            "total_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 10},
        }
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk([_choice({"content": "ok"})], usage=usage),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        message_start = next(e for e in events if e["type"] == "message_start")
        usage_out = message_start["message"]["usage"]
        self.assertEqual(usage_out["input_tokens"], 0)
        self.assertEqual(usage_out["cache_read_input_tokens"], 10)

    def test_usage_without_prompt_tokens_details(self):
        """Usage object without prompt_tokens_details must omit cache_read_input_tokens cleanly."""
        usage = {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5}
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk([_choice({"content": "ok"})], usage=usage),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        message_start = next(e for e in events if e["type"] == "message_start")
        usage_out = message_start["message"]["usage"]
        self.assertEqual(usage_out["input_tokens"], 5)
        self.assertNotIn("cache_read_input_tokens", usage_out)

    def test_non_streaming_error_with_non_json_body(self):
        """Non-JSON upstream error body falls back to body[:500] as the message (for 4xx)."""
        serving = AnthropicServing(
            _FakeNonStreamingErrorOpenAI(
                status_code=400,
                body=b"<html>upstream gateway rejected: bad payload</html>",
            )
        )
        chat_request = ChatCompletionRequest(
            model="test-model",
            max_tokens=16,
            messages=[{"role": "user", "content": "hello"}],
        )
        anthropic_request = self._anthropic_request(stream=False)
        response = asyncio.run(
            serving._handle_non_streaming(chat_request, anthropic_request, object())
        )
        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertIn("upstream gateway rejected", payload["error"]["message"])

    def test_non_streaming_error_5xx_scrubs_message(self):
        """5xx errors always return a generic message regardless of upstream payload.

        Type mapping follows spec §5.2, including the dedicated
        ``timeout_error`` for 504. G-24: the 503 → 529 WIRE translation
        lives in the route-layer ``AnthropicOverloadedStatusMiddleware``
        (covered by test_http_contract.py) — unit tests bypass middleware,
        so the serving layer must emit status 503 + type
        ``overloaded_error`` here and nothing more.
        """
        for status_code, expected_type, expected_status in [
            (500, "api_error", 500),
            (502, "api_error", 502),
            (503, "overloaded_error", 503),
            (504, "timeout_error", 504),
        ]:
            serving = AnthropicServing(
                _FakeNonStreamingErrorOpenAI(
                    status_code=status_code,
                    body=b'{"error":{"message":"sensitive internals: /opt/secret","type":"internal"}}',
                )
            )
            chat_request = ChatCompletionRequest(
                model="test-model",
                max_tokens=16,
                messages=[{"role": "user", "content": "hello"}],
            )
            anthropic_request = self._anthropic_request(stream=False)
            response = asyncio.run(
                serving._handle_non_streaming(chat_request, anthropic_request, object())
            )
            payload = json.loads(response.body)
            self.assertEqual(
                response.status_code,
                expected_status,
                f"upstream {status_code} should surface as {expected_status}",
            )
            self.assertEqual(
                payload["error"]["type"],
                expected_type,
                f"status {status_code} should map to {expected_type}",
            )
            self.assertEqual(
                payload["error"]["message"],
                "Internal server error",
                f"status {status_code} must scrub the message; got {payload['error']['message']!r}",
            )

    def test_non_streaming_response_includes_thinking_block(self):
        """When the OpenAI response carries reasoning_content, the Anthropic response has a thinking block first."""
        response = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "the answer is 4",
                            "reasoning_content": "2 + 2 = 4",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                    "total_tokens": 8,
                },
            }
        )
        anthropic_response = self._serving()._convert_response(response)
        # thinking block first, then text block
        self.assertEqual(anthropic_response.content[0].type, "thinking")
        self.assertEqual(anthropic_response.content[0].thinking, "2 + 2 = 4")
        self.assertEqual(anthropic_response.content[1].type, "text")
        self.assertEqual(anthropic_response.content[1].text, "the answer is 4")

    def test_request_thinking_disabled_invokes_apply_reasoning_enabled(self):
        """``thinking={"type": "disabled"}`` must flip the reasoning toggle off."""
        serving = self._serving()
        request = self._anthropic_request(thinking={"type": "disabled"}, stream=False)
        serving._convert_to_chat_completion_request(request)
        self.assertEqual(serving.openai_serving_chat.apply_reasoning_calls, [False])

    def test_request_thinking_enabled_with_budget_tokens_logs_warning(self):
        """SDK shape: ``enabled`` requires ``budget_tokens``. We accept it
        (the SDK would), but log a WARNING because the local backend has
        no equivalent hard-cap knob — the budget is not enforced.

        ``max_tokens`` is deliberately above ``budget_tokens`` because
        G-11 (spec §7.2) now 400s ``budget_tokens >= max_tokens``."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(
            thinking={"type": "enabled", "budget_tokens": 2048},
            max_tokens=4096,
            stream=False,
        )
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic", level=logging.WARNING
        ) as log:
            serving._convert_to_chat_completion_request(request)
        self.assertEqual(serving.openai_serving_chat.apply_reasoning_calls, [True])
        self.assertTrue(
            any("budget_tokens=2048" in r and "not enforced" in r for r in log.output),
            f"expected unenforced-budget warning: {log.output}",
        )

    def test_request_thinking_enabled_requires_budget_tokens(self):
        """SDK requires ``budget_tokens`` for ``type=enabled`` — Pydantic 400."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            self._anthropic_request(thinking={"type": "enabled"}, stream=False)
        self.assertIn("budget_tokens", str(ctx.exception))

    def test_request_thinking_enabled_budget_below_min_is_rejected(self):
        """SDK doc: ``budget_tokens`` must be >= 1024."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            self._anthropic_request(
                thinking={"type": "enabled", "budget_tokens": 512}, stream=False
            )
        self.assertIn("1024", str(ctx.exception))

    def test_request_thinking_disabled_with_display_is_rejected(self):
        """SDK ``ThinkingConfigDisabledParam`` has no ``display`` field."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            self._anthropic_request(
                thinking={"type": "disabled", "display": "omitted"}, stream=False
            )
        self.assertIn("display", str(ctx.exception))

    def test_request_thinking_disabled_with_budget_is_rejected(self):
        """SDK ``ThinkingConfigDisabledParam`` has no ``budget_tokens`` field."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            self._anthropic_request(
                thinking={"type": "disabled", "budget_tokens": 2048}, stream=False
            )
        self.assertIn("budget_tokens", str(ctx.exception))

    def test_request_thinking_adaptive_with_budget_is_rejected(self):
        """SDK ``ThinkingConfigAdaptiveParam`` has no ``budget_tokens`` field."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            self._anthropic_request(
                thinking={"type": "adaptive", "budget_tokens": 2048}, stream=False
            )
        self.assertIn("budget_tokens", str(ctx.exception))

    def test_request_thinking_adaptive_is_treated_as_enabled(self):
        """Claude 4.7 ``thinking.type='adaptive'`` (the SDK default for
        unknown models) must be accepted and routed to ``apply_reasoning_enabled(True)``.
        """
        serving = self._serving()
        request = self._anthropic_request(thinking={"type": "adaptive"}, stream=False)
        serving._convert_to_chat_completion_request(request)
        self.assertEqual(serving.openai_serving_chat.apply_reasoning_calls, [True])

    def test_request_max_tokens_zero_is_accepted_as_prewarm_and_clamped(self):
        """Spec §2.1: ``max_tokens=0`` is a legal cache pre-warm request
        ("don't generate"). sglang's radix cache only fills from a REAL
        engine pass, so a synthesized 200 with no engine call would report
        "warmed" while warming nothing. Contract: the converter clamps 0→1
        (whose forced ``length`` finish maps back to ``max_tokens`` on the
        wire) and logs the clamp for operators."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(max_tokens=0, stream=False)
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic", level=logging.INFO
        ) as log:
            chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(chat_request.max_tokens, 1)
        self.assertTrue(
            any("pre-warm" in r for r in log.output),
            f"expected a clamp notice mentioning the pre-warm: {log.output}",
        )

    def test_request_max_tokens_zero_prewarm_reaches_engine_and_returns_max_tokens(self):
        """R1-G05 adjudication, round-trip proof: a pre-warm request goes
        through the REAL engine handler (``_handle_non_streaming`` is
        exercised — no early-exit synth), and the forced single-token
        ``length`` finish surfaces as ``stop_reason="max_tokens"``."""
        response = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-prewarm",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
            }
        )
        serving = AnthropicServing(_FakeNonStreamingOpenAI(response))
        request = self._anthropic_request(max_tokens=0, stream=False)
        reply = asyncio.run(serving.handle_messages(request, object()))
        self.assertEqual(reply.status_code, 200)
        payload = json.loads(reply.body)
        self.assertEqual(payload["type"], "message")
        self.assertEqual(payload["stop_reason"], "max_tokens")
        self.assertEqual(payload["usage"]["output_tokens"], 1)

    def test_request_max_tokens_negative_is_rejected(self):
        """Negative ``max_tokens`` remains an invalid_request (spec §2.1)."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self._anthropic_request(max_tokens=-1, stream=False)

    def test_request_thinking_display_omitted_logs_warning_but_still_enables(self):
        """``thinking.display='omitted'`` is accepted; reasoning stays on
        because we cannot suppress reasoning text from the OpenAI stream.
        ``enabled`` requires ``budget_tokens`` per SDK shape."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(
            thinking={
                "type": "enabled",
                "budget_tokens": 1024,
                "display": "omitted",
            },
            max_tokens=4096,  # G-11: budget_tokens >= max_tokens now 400s
            stream=False,
        )
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic", level=logging.WARNING
        ) as log:
            serving._convert_to_chat_completion_request(request)
        self.assertEqual(serving.openai_serving_chat.apply_reasoning_calls, [True])
        self.assertTrue(any("omitted" in r for r in log.output))

    def test_request_output_config_effort_maps_to_reasoning_effort(self):
        """``output_config.effort`` rows map onto ``reasoning_effort``."""
        for anthropic_effort, openai_effort in [
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "max"),  # OpenAI Literal has no xhigh
            ("max", "max"),
        ]:
            with self.subTest(anthropic_effort=anthropic_effort):
                serving = self._serving()
                request = self._anthropic_request(
                    output_config={"effort": anthropic_effort}, stream=False
                )
                chat_request = serving._convert_to_chat_completion_request(request)
                self.assertEqual(chat_request.reasoning_effort, openai_effort)

    def test_request_output_config_task_budget_is_logged_not_enforced(self):
        """``task_budget`` is a soft hint; ``max_tokens`` is the hard cap."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(
            output_config={"task_budget": {"type": "tokens", "total": 32768}},
            stream=False,
        )
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic", level=logging.INFO
        ) as log:
            chat_request = serving._convert_to_chat_completion_request(request)
        # max_tokens is untouched
        self.assertEqual(chat_request.max_tokens, 16)
        self.assertTrue(any("task_budget" in r and "32768" in r for r in log.output))

    def test_request_output_config_format_json_schema_maps_to_response_format(self):
        """Anthropic structured outputs (spec §2.1):
        ``output_config.format={"type": "json_schema", "schema": {...}}``
        bridges to an OpenAI ``response_format`` of type ``json_schema``.
        Anthropic's shape has no ``name`` — the bridge synthesises the
        neutral label "response"; ``strict`` is left unset."""
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        serving = self._serving()
        request = self._anthropic_request(
            output_config={"format": {"type": "json_schema", "schema": schema}},
            stream=False,
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        response_format = chat_request.response_format
        self.assertIsNotNone(response_format)
        self.assertEqual(response_format.type, "json_schema")
        self.assertEqual(response_format.json_schema.name, "response")
        self.assertIsNone(response_format.json_schema.description)
        self.assertEqual(response_format.json_schema.schema_, schema)
        self.assertIsNone(response_format.json_schema.strict)

    def test_request_output_config_format_accepts_json_schema_alias(self):
        """The OpenAI-flavoured key ``json_schema`` is accepted alongside
        Anthropic's canonical ``schema`` key (validation_alias) so
        payloads from either ecosystem parse."""
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        serving = self._serving()
        request = self._anthropic_request(
            output_config={
                "format": {"type": "json_schema", "json_schema": schema}
            },
            stream=False,
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(chat_request.response_format.json_schema.schema_, schema)

    def test_request_betas_is_accepted_and_logged(self):
        """``betas`` is accepted and logged; the local backend has no beta system."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(betas=["thinking-2025-08-04"], stream=False)
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic", level=logging.INFO
        ) as log:
            serving._convert_to_chat_completion_request(request)
        self.assertTrue(any("thinking-2025-08-04" in r for r in log.output))

    def test_assistant_thinking_history_is_rewrapped_for_chat_template(self):
        """Past-turn thinking blocks get re-emitted via wrap_reasoning_history."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "ponder"},
                        {"type": "text", "text": "hello"},
                    ],
                },
                {"role": "user", "content": "again"},
            ],
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        # ``ChatCompletionRequest.messages`` is a list of Pydantic
        # ChatCompletionMessage*Param instances; access via attributes.
        assistant_msg = next(m for m in chat_request.messages if m.role == "assistant")
        content = assistant_msg.content
        # Reasoning history sits in front; the thinking block itself is dropped
        # from the prompt so its text is not duplicated.
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict):
                    texts.append(part.get("text", ""))
                else:
                    texts.append(getattr(part, "text", "") or "")
        else:
            texts = [content]
        joined = "\n".join(texts)
        self.assertIn("<think>", joined)
        self.assertIn("ponder", joined)
        self.assertNotIn("<think>\nponder\n</think>\nponder", joined)

    def test_redacted_thinking_history_is_rejected(self):
        """``redacted_thinking`` cannot be rendered by local parsers."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "redacted_thinking", "data": "opaque"},
                    ],
                },
            ],
        )
        with self.assertRaises(ValueError):
            serving._convert_to_chat_completion_request(request)

    def test_stream_text_then_thinking_closes_text_block(self):
        """Text deltas followed by reasoning_content must close the text block before opening thinking."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": "Direct"})]),
                _chunk([_choice({"reasoning_content": "but wait, let me think"})]),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        block_events = [
            (event["type"], event.get("content_block", {}).get("type"))
            for event in events
            if event["type"].startswith("content_block")
        ]
        # text (start, delta, stop) then thinking (start, delta, stop)
        self.assertEqual(block_events[0], ("content_block_start", "text"))
        text_stop_idx = next(
            i for i, ev in enumerate(block_events) if ev == ("content_block_stop", None)
        )
        thinking_start_idx = next(
            i
            for i, ev in enumerate(block_events)
            if ev == ("content_block_start", "thinking")
        )
        self.assertLess(text_stop_idx, thinking_start_idx)

    def test_stream_consecutive_tool_calls_get_separate_blocks(self):
        """Two tool_use calls in sequence must occupy distinct content_block indices."""
        serving = self._serving(
            [
                _chunk(
                    [
                        _choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_a",
                                        "function": {
                                            "name": "alpha",
                                            "arguments": '{"x":1}',
                                        },
                                    }
                                ]
                            }
                        )
                    ]
                ),
                _chunk(
                    [
                        _choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 1,
                                        "id": "call_b",
                                        "function": {
                                            "name": "beta",
                                            "arguments": '{"y":2}',
                                        },
                                    }
                                ]
                            }
                        )
                    ]
                ),
                _chunk([_choice({}, finish_reason="tool_calls")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        starts = [
            (e["index"], e["content_block"]["name"])
            for e in events
            if e["type"] == "content_block_start"
        ]
        stops = [e["index"] for e in events if e["type"] == "content_block_stop"]
        deltas = [
            (e["index"], e["delta"].get("partial_json"))
            for e in events
            if e["type"] == "content_block_delta"
            and e["delta"].get("type") == "input_json_delta"
        ]
        # Each tool gets its own start, its own stop, and its own
        # argument delta — without the fix, beta's args were appended
        # to alpha's index 0 block.
        self.assertEqual(starts, [(0, "alpha"), (1, "beta")])
        self.assertEqual(stops, [0, 1])
        self.assertEqual(deltas, [(0, '{"x":1}'), (1, '{"y":2}')])

    def test_stream_finish_chunk_with_payload_emits_delta(self):
        """A chunk carrying both finish_reason and content must not drop the content."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant"})]),
                _chunk([_choice({"content": "last token"}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        text_deltas = [
            e["delta"]["text"]
            for e in events
            if e["type"] == "content_block_delta"
            and e["delta"].get("type") == "text_delta"
        ]
        self.assertEqual(text_deltas, ["last token"])
        # And stop_reason still travels via message_delta
        message_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "end_turn")

    def test_stream_stop_sequence_from_matched_stop_string(self):
        """A str ``matched_stop`` on the finish chunk means one of the
        request's stop_sequences matched (spec §3.1): message_delta must
        carry stop_reason='stop_sequence' plus the matched string."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": "Hi"})]),
                _chunk(
                    [
                        _choice(
                            {},
                            finish_reason="stop",
                            matched_stop="\n\nHuman:",
                        )
                    ]
                ),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(
                serving,
                self._anthropic_request(stop_sequences=["\n\nHuman:", "###"]),
            )
        )
        message_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "stop_sequence")
        self.assertEqual(message_delta["delta"]["stop_sequence"], "\n\nHuman:")

    def test_stream_tool_calls_finish_with_matched_str_stays_tool_use(self):
        """Round-1 defect (a): the OpenAI stream passes ``matched``
        through un-nulled on TOOL-CALL finish chunks — the matched string
        must NOT override the tool_calls finish (client needs tool_use
        semantics)."""
        serving = self._serving(
            [
                _chunk(
                    [
                        _choice(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "fn",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            }
                        )
                    ]
                ),
                _chunk(
                    [
                        _choice(
                            {}, finish_reason="tool_calls", matched_stop="\n\nHuman:"
                        )
                    ]
                ),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        message_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")
        self.assertNotIn("stop_sequence", message_delta["delta"])

    def test_stream_foreign_matched_str_falls_back_to_end_turn(self):
        """Round-1 defect (b): a matched string that is NOT a member of
        the request's stop_sequences (scheduler-internal sentinel like
        FINISH_MATCHED_STR's 'NaN happened') must NOT leak to the wire —
        fall back to end_turn."""
        import logging

        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": "Hi"})]),
                _chunk(
                    [
                        _choice(
                            {}, finish_reason="stop", matched_stop="NaN happened"
                        )
                    ]
                ),
                "data: [DONE]\n\n",
            ]
        )
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic", level=logging.WARNING
        ) as log:
            events = asyncio.run(
                _collect_anthropic_events(serving, self._anthropic_request())
            )
        message_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "end_turn")
        self.assertNotIn("stop_sequence", message_delta["delta"])
        self.assertTrue(
            any("not among request stop_sequences" in rec for rec in log.output)
        )

    def test_stream_matched_stop_token_id_keeps_end_turn(self):
        """An int ``matched_stop`` is a stop TOKEN id (e.g. EOS), not a
        user-provided stop sequence — the wire must stay 'end_turn' with
        NO stop_sequence field (``exclude_none`` drops it)."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": "Hi"})]),
                _chunk([_choice({}, finish_reason="stop", matched_stop=151643)]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        message_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "end_turn")
        self.assertNotIn("stop_sequence", message_delta["delta"])

    def test_stream_ping_watchdog_emits_on_stall(self):
        """G-21 (spec §4.2): when the OpenAI stream stalls past the ping
        interval, a keep-alive ping is emitted instead of silence, so
        proxies with idle timeouts don't drop the connection."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": "Hi"})]),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ],
            stall_after_first_line=0.15,
        )
        serving._ping_interval_seconds = 0.05
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [e["type"] for e in events]
        # Layer 1 (static, right after message_start) + layer 2 (watchdog
        # during the stall between chunk 1 and chunk 2).
        self.assertEqual(types[0], "message_start")
        self.assertEqual(types[1], "ping")
        self.assertGreaterEqual(types.count("ping"), 2)
        self.assertEqual(types[-1], "message_stop")
        self.assertLess(types.index("ping"), types.index("message_stop"))

    def test_stream_fast_stream_carries_single_static_ping(self):
        """G-21 layer 1: exactly one static ping, right after the FIRST
        message_start — keep-alive for idle-timeout proxies without spamming
        strict SDK clients. No stalls → no watchdog pings on top."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": "Hi"})]),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )
        serving._ping_interval_seconds = 0.05
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "message_start")
        self.assertEqual(types[1], "ping")
        self.assertEqual(types.count("ping"), 1)

    def test_stream_empty_completion_with_finish_reason_emits_message_delta(self):
        """An empty stream with a finish_reason is a legitimate stop, not api_error."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant"})]),
                _chunk([_choice({}, finish_reason="length")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [e["type"] for e in events]
        self.assertIn("message_start", types)
        self.assertIn("message_delta", types)
        self.assertIn("message_stop", types)
        self.assertNotIn("error", types)
        message_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "max_tokens")

    def test_stream_no_finish_no_content_still_emits_api_error(self):
        """Backend that drops both content and finish_reason is genuinely broken."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant"})]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [e["type"] for e in events]
        self.assertIn("error", types)
        err = next(e for e in events if e["type"] == "error")
        self.assertEqual(err["error"]["type"], "api_error")

    def test_stream_upstream_error_envelope_is_forwarded(self):
        """OpenAI handler streaming-error JSON must surface real type/message."""
        upstream_error = {
            "error": {
                "object": "error",
                "message": "context length exceeded",
                "type": "BadRequestError",
                "code": 400,
            }
        }
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant"})]),
                f"data: {json.dumps(upstream_error)}\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        err = next(e for e in events if e["type"] == "error")
        self.assertEqual(err["error"]["type"], "invalid_request_error")
        self.assertEqual(err["error"]["message"], "context length exceeded")
        # message_stop must still close the stream
        self.assertEqual(events[-1]["type"], "message_stop")

    def test_stream_parse_failure_closes_open_content_block(self):
        """Unparsable mid-stream chunk must still close any open content_block."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": "first"})]),
                "data: {not-json\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [e["type"] for e in events]
        # Sequence: message_start, content_block_start, content_block_delta,
        # content_block_stop, error, message_stop
        self.assertIn("content_block_start", types)
        self.assertEqual(
            types.count("content_block_stop"),
            types.count("content_block_start"),
            f"unbalanced block events: {types}",
        )
        self.assertIn("error", types)
        self.assertEqual(types[-1], "message_stop")

    def test_stream_pre_first_chunk_value_error_returns_400(self):
        """G-22: a ValueError surfaced on the FIRST OpenAI-stream step
        (tokenization failure etc.) must abort BEFORE the HTTP 200 is
        committed — the client receives a real 400 envelope, not an
        in-band SSE error."""

        class _RaisingOpenAI(_FakeOpenAIServingChat):
            def _validate_request(self, chat_request):
                return None

            def _convert_to_internal_request(self, chat_request, raw_request):
                return SimpleNamespace(), chat_request

            def _generate_chat_stream(
                self, adapted_request, processed_request, raw_request
            ):
                async def _gen():
                    raise ValueError("tokenization failed")
                    yield  # pragma: no cover

                return _gen()

        serving = AnthropicServing(_RaisingOpenAI())
        chat_request = serving._convert_to_chat_completion_request(
            self._anthropic_request()
        )
        response = asyncio.run(
            serving._handle_streaming(chat_request, self._anthropic_request(), object())
        )
        self.assertEqual(response.status_code, 400)
        payload = json.loads(response.body)
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertIn("tokenization failed", payload["error"]["message"])

    def test_stream_mid_flight_value_error_still_flushes_envelope(self):
        """A ValueError raised AFTER the stream is committed (mid-flight)
        keeps the pre-G-22 behaviour: an in-band Anthropic error sequence
        so the client sees a clean failure instead of a half-open SSE."""

        class _MidFlightRaisingOpenAI(_FakeOpenAIServingChat):
            def _generate_chat_stream(
                self, adapted_request, processed_request, raw_request
            ):
                async def _gen():
                    yield _chunk([_choice({"role": "assistant", "content": "Hi"})])
                    raise ValueError("detokenization failed")

                return _gen()

        serving = AnthropicServing(_MidFlightRaisingOpenAI())
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "message_start")
        self.assertIn("error", types)
        err = next(e for e in events if e["type"] == "error")
        self.assertEqual(err["error"]["type"], "invalid_request_error")
        self.assertIn("detokenization failed", err["error"]["message"])
        self.assertEqual(types[-1], "message_stop")

    def test_server_tool_only_with_tool_choice_any_raises_400(self):
        """A request with only server-side tools cannot honor tool_choice=any."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            tool_choice={"type": "any"},
        )
        with self.assertRaises(ValueError) as ctx:
            serving._convert_to_chat_completion_request(request)
        self.assertIn("tool_choice", str(ctx.exception))

    def test_tool_choice_named_custom_tool_is_resolved(self):
        """tool_choice={type:'tool', name:'X'} where X is a custom tool wires through."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            tools=[
                {
                    "type": "custom",
                    "name": "lookup",
                    "input_schema": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "lookup"},
        )
        # Must not AttributeError: Tool.function is a Pydantic model, not a
        # dict — access must be via .name, never .get("name").
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(chat_request.tool_choice.type, "function")
        self.assertEqual(chat_request.tool_choice.function.name, "lookup")

    def test_disable_parallel_tool_use_maps_to_parallel_tool_calls_false(self):
        """``disable_parallel_tool_use: true`` (spec §2.6) caps the model
        at a single tool_use block; OpenAI's ``parallel_tool_calls=False``
        carries the identical "at most one tool" semantics. It must apply
        across the auto/any/tool choice modes."""
        for tool_choice in [
            {"type": "auto", "disable_parallel_tool_use": True},
            {"type": "any", "disable_parallel_tool_use": True},
            {"type": "tool", "name": "lookup", "disable_parallel_tool_use": True},
        ]:
            with self.subTest(tool_choice=tool_choice):
                serving = self._serving()
                request = self._anthropic_request(
                    stream=False,
                    tools=[
                        {
                            "type": "custom",
                            "name": "lookup",
                            "input_schema": {"type": "object", "properties": {}},
                        }
                    ],
                    tool_choice=tool_choice,
                )
                chat_request = serving._convert_to_chat_completion_request(request)
                self.assertFalse(chat_request.parallel_tool_calls)

    def test_parallel_tool_calls_stays_true_without_flag(self):
        """Without ``disable_parallel_tool_use`` the OpenAI default
        (``parallel_tool_calls=True``) is preserved for tool-bearing
        requests — parallel tool use stays enabled."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            tools=[
                {
                    "type": "custom",
                    "name": "lookup",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            tool_choice={"type": "auto"},
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertTrue(chat_request.parallel_tool_calls)

    def test_tool_choice_named_unknown_tool_raises_400(self):
        """tool_choice={type:'tool', name:'X'} where X is missing must raise."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            tools=[
                {
                    "type": "custom",
                    "name": "lookup",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            tool_choice={"type": "tool", "name": "nonexistent"},
        )
        with self.assertRaises(ValueError) as ctx:
            serving._convert_to_chat_completion_request(request)
        self.assertIn("nonexistent", str(ctx.exception))

    def test_convert_response_non_streaming_empty_content_keeps_block(self):
        """Empty-string completion must still produce a content list of len 1."""
        response = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 0,
                    "total_tokens": 5,
                },
            }
        )
        serving = self._serving()
        anthropic_response = serving._convert_response(response)
        self.assertEqual(len(anthropic_response.content), 1)
        self.assertEqual(anthropic_response.content[0].type, "text")
        self.assertEqual(anthropic_response.content[0].text, "")

    def test_convert_response_matched_stop_string_becomes_stop_sequence(self):
        """Non-stream: the scheduler's matched stop STRING (a configured
        stop_sequences entry) must surface as stop_reason='stop_sequence'
        with the matched string (spec §3.1), on both the model and the
        exclude_none wire dump."""
        response = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                        "matched_stop": "###",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
            }
        )
        anthropic_response = self._serving()._convert_response(
            response, stop_sequences=["###"]
        )
        self.assertEqual(anthropic_response.stop_reason, "stop_sequence")
        self.assertEqual(anthropic_response.stop_sequence, "###")

    def test_convert_response_foreign_matched_str_not_stop_sequence(self):
        """Non-stream variant of the sentinel guard: matched string NOT
        among the request's stop_sequences → end_turn, no stop_sequence."""
        import logging

        response = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                        "matched_stop": "NaN happened",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
            }
        )
        serving = self._serving()
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic", level=logging.WARNING
        ):
            anthropic_response = serving._convert_response(
                response, stop_sequences=["###"]
            )
        self.assertEqual(anthropic_response.stop_reason, "end_turn")
        self.assertIsNone(anthropic_response.stop_sequence)
        dumped = anthropic_response.model_dump(exclude_none=True)
        self.assertEqual(dumped["stop_reason"], "end_turn")
        self.assertNotIn("stop_sequence", dumped)

    def test_convert_response_matched_stop_token_id_keeps_end_turn(self):
        """An int ``matched_stop`` is a stop token id (e.g. EOS matched),
        NOT one of the request's stop_sequences — stop_reason stays
        'end_turn' and no stop_sequence reaches the wire."""
        response = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                        "matched_stop": 0,
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
            }
        )
        anthropic_response = self._serving()._convert_response(response)
        self.assertEqual(anthropic_response.stop_reason, "end_turn")
        self.assertIsNone(anthropic_response.stop_sequence)
        self.assertNotIn(
            "stop_sequence", anthropic_response.model_dump(exclude_none=True)
        )

    def test_error_response_does_not_leak_exception_name(self):
        """``error.type`` must stay in Anthropic's documented literal set."""
        serving = self._serving()
        response = serving._error_response(
            status_code=500,
            error_type="api_error",
            message="Internal server error",
            exception_name="KeyError",
        )
        body = json.loads(bytes(response.body).decode())
        self.assertEqual(body["error"]["type"], "api_error")

    def test_error_response_includes_request_id(self):
        """Spec §5.1: the error envelope carries a top-level ``request_id``
        with the ``req_...`` prefix for client-side quoting/correlation."""
        serving = self._serving()
        response = serving._error_response(
            status_code=400,
            error_type="invalid_request_error",
            message="bad request",
        )
        body = json.loads(bytes(response.body).decode())
        self.assertEqual(response.status_code, 400)
        request_id = body.get("request_id")
        self.assertIsInstance(request_id, str)
        self.assertTrue(
            request_id.startswith("req_"),
            f"request_id must use Anthropic's req_... prefix: {request_id!r}",
        )
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "invalid_request_error")

    def test_non_streaming_usage_includes_thinking_tokens_detail(self):
        """Spec §3.3: when the backend spent reasoning tokens, usage gains
        ``output_tokens_details.thinking_tokens``; absent otherwise."""
        base = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }
        reasoning_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "reasoning_tokens": 7,
        }
        response = ChatCompletionResponse.model_validate(
            {**base, "usage": reasoning_usage}
        )
        anthropic_response = self._serving()._convert_response(response)
        self.assertEqual(
            anthropic_response.usage.output_tokens_details.thinking_tokens, 7
        )
        dumped = anthropic_response.model_dump(exclude_none=True)
        self.assertEqual(
            dumped["usage"]["output_tokens_details"]["thinking_tokens"], 7
        )

        # Non-reasoning run: the details object must not appear at all.
        plain_response = ChatCompletionResponse.model_validate(
            {
                **base,
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }
        )
        anthropic_response = self._serving()._convert_response(plain_response)
        self.assertIsNone(anthropic_response.usage.output_tokens_details)
        self.assertNotIn(
            "output_tokens_details",
            anthropic_response.model_dump(exclude_none=True)["usage"],
        )

    def test_user_message_text_tool_text_preserves_order(self):
        """User message [text, tool_result, text] must stay user→tool→user on the wire."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_x",
                            "content": "ok",
                        },
                        {"type": "text", "text": "second"},
                    ],
                }
            ],
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        # chat_request.messages items are Pydantic ChatCompletionMessage*Param
        # variants — use attribute access, not subscripts.
        roles = [m.role for m in chat_request.messages]
        self.assertEqual(roles, ["user", "tool", "user"])
        self.assertEqual(chat_request.messages[0].content, "first")
        self.assertEqual(chat_request.messages[2].content, "second")

    def test_empty_text_assistant_turn_preserves_role_alternation(self):
        """Assistant turn with only empty text must NOT vanish from the wire."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            messages=[
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
                {"role": "user", "content": "u2"},
            ],
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        roles = [m.role for m in chat_request.messages]
        # Without the fix this collapses to ['user', 'user'] and breaks
        # strict role-alternation chat templates (qwen, llama, mistral).
        self.assertEqual(roles, ["user", "assistant", "user"])

    def test_in_messages_system_merged_when_template_requires_first(self):
        """When the chat template rejects mid-conversation ``role: "system"``
        (e.g. Qwen's system-first guard), the converter folds the inline
        system turn into the leading system block so the template doesn't
        400. The request object itself is no longer mutated — detection runs
        in the serving layer on conversion."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "Reply with exactly: OK"},
                {"role": "user", "content": "go"},
            ],
        )
        self.assertIsNone(request.system)
        self.assertEqual([m.role for m in request.messages], ["user", "system", "user"])
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(
            [m.role for m in chat_request.messages], ["system", "user", "user"]
        )
        self.assertEqual(chat_request.messages[0].content, "Reply with exactly: OK")

    def test_in_messages_system_merged_with_top_level_when_merge(self):
        """On the merge path, a top-level ``system`` field and a mid-conversation
        system turn are joined into the leading system block; top-level text
        comes first."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            system="You are terse.",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "One word only."},
                {"role": "user", "content": "go"},
            ],
        )
        self.assertEqual(request.system, "You are terse.")
        self.assertEqual([m.role for m in request.messages], ["user", "system", "user"])
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(
            [m.role for m in chat_request.messages], ["system", "user", "user"]
        )
        self.assertEqual(
            chat_request.messages[0].content, "You are terse.\nOne word only."
        )

    def test_in_messages_system_passed_through_when_template_allows_inline(self):
        """When the chat template renders ``role: "system"`` at any position
        (GLM / Kimi / Qwen3), the inline system turn stays at its original
        position — preserving the prefix cache and the request's structure."""
        serving = self._serving(chat_template=self.INLINE_SYSTEM_TEMPLATE)
        self.assertFalse(serving._merge_inline_system)
        request = self._anthropic_request(
            stream=False,
            system="You are terse.",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "Reply with exactly: OK"},
                {"role": "user", "content": "go"},
            ],
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(
            [m.role for m in chat_request.messages],
            ["system", "user", "system", "user"],
        )
        self.assertEqual(chat_request.messages[0].content, "You are terse.")
        self.assertEqual(chat_request.messages[2].content, "Reply with exactly: OK")

    def test_top_level_system_only_is_unchanged(self):
        """A request with only the top-level ``system`` field (no in-messages
        system turn) is unaffected on both detection paths: the system field is
        preserved verbatim and the dialogue order is untouched. Guards the
        common multi-turn path against regressions."""
        for template in (None, self.INLINE_SYSTEM_TEMPLATE):
            serving = self._serving(chat_template=template)
            request = self._anthropic_request(
                stream=False,
                system="You are a helpful assistant.",
                messages=[
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Hello Alice!"},
                    {"role": "user", "content": "What is my name?"},
                ],
            )
            self.assertEqual(request.system, "You are a helpful assistant.")
            self.assertEqual(
                [m.role for m in request.messages], ["user", "assistant", "user"]
            )
            chat_request = serving._convert_to_chat_completion_request(request)
            self.assertEqual(
                [m.role for m in chat_request.messages],
                ["system", "user", "assistant", "user"],
            )
            self.assertEqual(
                chat_request.messages[0].content, "You are a helpful assistant."
            )

    def test_constructed_message_objects_merged_on_merge_path(self):
        """Requests built programmatically with ``AnthropicMessage`` objects
        (e.g. ``handle_count_tokens``) also get inline system folded into the
        leading block on the merge path."""
        serving = self._serving()
        request = AnthropicMessagesRequest(
            model="m",
            max_tokens=8,
            messages=[
                AnthropicMessage(role="user", content="hi"),
                AnthropicMessage(role="system", content="be terse"),
                AnthropicMessage(role="user", content="go"),
            ],
        )
        self.assertEqual([m.role for m in request.messages], ["user", "system", "user"])
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(
            [m.role for m in chat_request.messages], ["system", "user", "user"]
        )
        self.assertEqual(chat_request.messages[0].content, "be terse")

    def test_thinking_history_drop_on_missing_detector(self):
        """Replaying a thinking block on a non-reasoning model should not 400."""

        class _NoDetectorOpenAI(_FakeOpenAIServingChat):
            def wrap_reasoning_history(self, text):
                raise ValueError("no reasoning detector is configured")

        serving = AnthropicServing(_NoDetectorOpenAI())
        request = self._anthropic_request(
            stream=False,
            messages=[
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "I think..."}],
                },
                {"role": "user", "content": "follow-up"},
            ],
        )
        # Must convert successfully; the thinking block is silently dropped.
        chat_request = serving._convert_to_chat_completion_request(request)
        roles = [m.role for m in chat_request.messages]
        self.assertIn("user", roles)
        # The assistant turn was rendered (as empty placeholder) so
        # alternation is preserved.
        self.assertIn("assistant", roles)

    def test_stop_reason_content_filter_falls_back_unmapped(self):
        """REVIEWER ORDER: ``content_filter`` has NO producer anywhere in
        sglang (producer census: it exists only in OpenAI Literal
        declarations), so the old ``content_filter → refusal`` mapping was
        unverifiable dead code and is REMOVED. An OpenAI-side
        ``content_filter`` finish, like any other unknown finish, degrades
        to ``end_turn`` with the generic unmapped-reason WARNING.
        ``refusal`` stays in the response Literal as a shape contract."""
        import logging

        response = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "content_filter",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )
        serving = self._serving()
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic", level=logging.WARNING
        ) as log:
            anthropic_response = serving._convert_response(response)
        self.assertEqual(anthropic_response.stop_reason, "end_turn")
        self.assertIsNone(anthropic_response.stop_sequence)
        self.assertTrue(
            any("Unmapped" in rec for rec in log.output),
            f"expected the generic unmapped-finish WARNING: {log.output}",
        )

    def test_stop_reason_abort_falls_back_with_warning(self):
        """``abort`` has no Anthropic equivalent (spec §3.1's enum carries
        no abort signal) — the response degrades to 'end_turn' and a
        WARNING so operators don't lose the abort signal in logs."""
        import logging

        response = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "abort",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )
        serving = self._serving()
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic", level=logging.WARNING
        ) as log:
            anthropic_response = serving._convert_response(response)
        self.assertEqual(anthropic_response.stop_reason, "end_turn")
        self.assertTrue(
            any("abort" in rec for rec in log.output),
            f"expected a warning mentioning the unmapped finish_reason: {log.output}",
        )

    # ---------- G-08: tool_use/tool_result pairing (spec §2.2.1/§7.1) ----------

    def test_tool_use_without_immediate_tool_result_raises_400(self):
        """G-08 (spec §2.2.1): every ``tool_use`` id must be followed
        IMMEDIATELY by matching ``tool_result`` blocks — otherwise the
        engine would generate with corrupt tool context."""
        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tu_1", "name": "fn", "input": {}}
                    ],
                },
                {"role": "user", "content": "no results here"},
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            serving._convert_to_chat_completion_request(request)
        self.assertIn("`tool_use` ids were found without `tool_result`", str(ctx.exception))
        self.assertIn("tu_1", str(ctx.exception))

    def test_tool_use_followed_by_wrong_tool_result_id_raises_400(self):
        """G-08: a tool_result whose id does not match the preceding
        tool_use is also a pairing failure."""
        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tu_1", "name": "fn", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu_2", "content": "x"}
                    ],
                },
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            serving._convert_to_chat_completion_request(request)
        self.assertIn("tu_1", str(ctx.exception))

    def test_tool_result_with_empty_ids_raises_400(self):
        """G-08: tool_result must identify the tool call it answers."""
        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "", "id": "", "content": "x"}
                    ],
                }
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            serving._convert_to_chat_completion_request(request)
        self.assertIn("tool_use_id", str(ctx.exception))

    def test_valid_tool_pairing_converts_to_wire(self):
        """G-08 happy path: paired tool_use/tool_result convert to
        assistant tool_calls + tool messages in OpenAI form."""
        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "sf"}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu_1", "content": "sunny"}
                    ],
                },
            ]
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        roles = [m.role for m in chat_request.messages]
        self.assertEqual(roles, ["user", "assistant", "tool"])
        tool_msg = chat_request.messages[2]
        self.assertEqual(tool_msg.tool_call_id, "tu_1")
        self.assertEqual(tool_msg.content, "sunny")

    # ---------- G-12: document blocks degrade to text --------------------------

    def test_document_pdf_blocks_degrade_to_placeholder(self):
        """G-12 (spec §2.2.2): binary PDF blocks are accepted structurally
        (SDK shape) but degrade to an explicit text placeholder — the local
        backend cannot run the PDF pipeline, and a skip would fabricate
        context without the document."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize this:"},
                        {
                            "type": "document",
                            "source": {"type": "url", "url": "https://x/f.pdf"},
                        },
                    ],
                }
            ]
        )
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic", level=logging.WARNING
        ) as log:
            chat_request = serving._convert_to_chat_completion_request(request)
        texts = [part.text for part in chat_request.messages[0].content]
        self.assertEqual(texts[0], "Summarize this:")
        self.assertIn(
            "PDF document omitted: backend lacks PDF support", texts[1]
        )
        self.assertTrue(any("PDF" in rec or "pdf" in rec for rec in log.output))

    def test_document_pdf_with_title_keeps_title_in_placeholder(self):
        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "title": "quarterly.pdf",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "AAAA",
                            },
                        }
                    ],
                }
            ]
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        content = chat_request.messages[0].content
        self.assertEqual(
            content,
            '[PDF document "quarterly.pdf" omitted: backend lacks PDF support]',
        )

    def test_document_text_source_converts_to_text(self):
        """G-12: text documents go through as plain text."""
        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "text", "media_type": "text/plain", "data": "Body text"},
                        }
                    ],
                }
            ]
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        content = chat_request.messages[0].content
        self.assertEqual(content, "Body text")

    def test_document_file_source_raises_400(self):
        """G-12: the hosted Files-API source cannot be consulted locally."""
        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "document", "source": {"type": "file", "file_id": "file_1"}}
                    ],
                }
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            serving._convert_to_chat_completion_request(request)
        self.assertIn("Files API", str(ctx.exception))

    def test_document_in_tool_result_content_degrades(self):
        """G-12: documents EMBEDDED in tool_result.content degrade the
        same way — real tool loops pipe pages through the feed (audit
        §1 table, document row)."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tu_1", "name": "fetch", "input": {"url": "https://x/f.pdf"}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {"type": "base64", "media_type": "application/pdf", "data": "AAAA"},
                                }
                            ],
                        }
                    ],
                },
            ]
        )
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic", level=logging.WARNING
        ):
            chat_request = serving._convert_to_chat_completion_request(request)
        tool_msg = chat_request.messages[-1]
        self.assertEqual(tool_msg.role, "tool")
        self.assertIn(
            "PDF document omitted: backend lacks PDF support", tool_msg.content
        )

    # ---------- G-14/G-15: server-tool + unknown blocks ------------------------

    def test_unknown_block_type_in_user_content_degrades(self):
        """G-15 (spec §0 tolerance clause): an unknown future block type
        must parse and degrade to a placeholder — never a 400."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "future_block_v3", "payload": 1},
                        {"type": "text", "text": "hi"},
                    ],
                }
            ]
        )
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic", level=logging.WARNING
        ) as log:
            chat_request = serving._convert_to_chat_completion_request(request)
        texts = [part.text for part in chat_request.messages[0].content]
        self.assertIn(
            '[Unsupported content block "future_block_v3" omitted',
            texts[0],
        )
        self.assertEqual(texts[1], "hi")
        self.assertTrue(any("future_block_v3" in rec for rec in log.output))

    def test_server_tool_use_block_in_assistant_turn_degrades(self):
        """G-14 (spec §7.1/notes A.3): 'server tool use must pass through
        the conversation history' — keep the block visible as a text
        placeholder instead of inventing a client tool call."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "srvtoolu_1",
                            "name": "web_search",
                            "input": {"query": "sf weather"},
                        },
                        {"type": "text", "text": "Let me look."},
                    ],
                },
                {"role": "user", "content": "go on"},
            ]
        )
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic", level=logging.WARNING
        ):
            chat_request = serving._convert_to_chat_completion_request(request)
        roles = [m.role for m in chat_request.messages]
        self.assertEqual(roles, ["user", "assistant", "user"])
        texts = [part.text for part in chat_request.messages[1].content]
        self.assertIn(
            'Unsupported content block "server_tool_use" omitted',
            texts[0],
        )

    def test_web_fetch_server_tool_skipped_with_log(self):
        """G-14b (notes A.3): dated server-tool families (web_fetch)
        carry no input_schema for versioned prompt templates; skip them
        (no local server-side tooling) with a WARNING, do not invent
        client tool calls, and keep client (function) tools registered."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            tools=[
                {"type": "web_fetch_20250910", "name": "web_fetch"},
                {
                    "type": "custom",
                    "name": "local_fn",
                    "input_schema": {"type": "object", "properties": {}},
                },
            ],
        )
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic", level=logging.INFO
        ) as log:
            chat_request = serving._convert_to_chat_completion_request(request)
        self.assertTrue(any("web_fetch" in rec for rec in log.output))
        # The function tool is kept; the server tool is not faked.
        self.assertIsNotNone(chat_request.tools)
        self.assertEqual(len(chat_request.tools), 1)
        self.assertEqual(chat_request.tools[0].function.name, "local_fn")

    def test_is_server_tool_recognizes_web_fetch_and_dated_families(self):
        from sglang.srt.entrypoints.anthropic.protocol import (
            is_server_tool,
        )

        def _tool(t):
            return AnthropicMessagesRequest.model_validate(
                {
                    "model": "m",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [t],
                }
            ).tools[0]

        self.assertTrue(
            is_server_tool(_tool({"type": "web_fetch_20250910", "name": "web_fetch"}))
        )
        self.assertTrue(
            is_server_tool(
                _tool({"type": "code_execution_20260120", "name": "code_execution"})
            )
        )
        # Dated tool_search_tool_/memory_ families route to the generic
        # server-tool model and skip-with-log the same way (work order G-14).
        self.assertTrue(
            is_server_tool(
                _tool({"type": "tool_search_tool_20251112", "name": "tool_search_tool"})
            )
        )
        self.assertTrue(
            is_server_tool(_tool({"type": "memory_20251112", "name": "memory"}))
        )
        self.assertFalse(
            is_server_tool(
                _tool(
                    {
                        "type": "custom",
                        "name": "fn",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                )
            )
        )

    # ---------- G-06: consecutive same-role messages merge ---------------------

    def test_consecutive_user_messages_merge(self):
        """G-06 (spec §2.2): consecutive same-role messages combine into
        one, so one OpenAI user message carries both text parts."""
        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {"role": "user", "content": "part one"},
                {"role": "user", "content": [{"type": "text", "text": "part two"}]},
            ]
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(len(chat_request.messages), 1)
        user_msg = chat_request.messages[-1]
        content = user_msg.content
        texts = (
            [content]
            if isinstance(content, str)
            else [part.text for part in content if part.type == "text"]
        )
        self.assertEqual(texts, ["part one", "part two"])

    def test_consecutive_assistant_messages_merge(self):
        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {"role": "assistant", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "question"},
            ]
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        roles = [m.role for m in chat_request.messages]
        self.assertEqual(roles, ["assistant", "user"])

    def test_consecutive_system_messages_merge_to_single_system(self):
        """G-06: with an inline-system-capable template, consecutive
        system messages become ONE system message whose content is the
        joined text (existing single-system message semantics preserved)."""
        serving = self._serving(chat_template=self.INLINE_SYSTEM_TEMPLATE)
        request = self._anthropic_request(
            messages=[
                {"role": "system", "content": "Rule A"},
                {"role": "system", "content": "Rule B"},
                {"role": "user", "content": "hi"},
            ]
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        system_msgs = [m for m in chat_request.messages if m.role == "system"]
        self.assertEqual(len(system_msgs), 1)
        content = system_msgs[0].content
        if isinstance(content, list):
            content = "".join(part.text or "" for part in content if part.type == "text")
        self.assertEqual(content, "Rule A\nRule B")

    def test_tool_result_boundary_not_merged(self):
        """G-06 + G-08: a tool_result block forges the required
        tool-result turn boundary — the following user turn must start a
        NEW message rather than merging INTO the tool results."""
        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tu_1", "name": "fn", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"}],
                },
                {"role": "user", "content": "and now?"},
            ]
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        roles = [m.role for m in chat_request.messages]
        self.assertEqual(roles, ["assistant", "tool", "user"])

    # ---------- G-07: trailing assistant prefill -------------------------------

    def test_trailing_assistant_becomes_prefill(self):
        """G-07 (spec §2.2.1): a trailing assistant message continues the
        generation (``continue_final_message``), never demotes to user."""
        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {"role": "user", "content": "Say hi"},
                {"role": "assistant", "content": "Hi there,"},
            ]
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertTrue(chat_request.continue_final_message)
        tail = chat_request.messages[-1]
        self.assertEqual((tail.role, tail.content), ("assistant", "Hi there,"))

    def test_trailing_assistant_with_tool_use_raises_400(self):
        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tu_1", "name": "fn", "input": {}}
                    ],
                },
            ]
        )
        # The pairing rule fires before prefill coercion — either way
        # a clear 400, never a demoted user message.
        with self.assertRaises(ValueError) as ctx:
            serving._convert_to_chat_completion_request(request)
        self.assertIn("tool_result", str(ctx.exception))

    def test_trailing_assistant_thinking_block_raises_prefill_400(self):
        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "hmm", "signature": ""}],
                },
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            serving._convert_to_chat_completion_request(request)
        self.assertIn("must not contain thinking blocks", str(ctx.exception))

    def test_trailing_assistant_trailing_whitespace_raises_400(self):
        """G-07 (spec §2.2.1): the API rejects prefills with trailing
        whitespace; do the same so clients get identical behavior."""
        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "partial answer "},
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            serving._convert_to_chat_completion_request(request)
        self.assertIn("trailing whitespace", str(ctx.exception))

    # ---------- G-09: temperature clamp ---------------------------------------

    def test_temperature_above_one_clamps_with_warning(self):
        import logging

        serving = self._serving()
        request = self._anthropic_request(temperature=1.5, stream=False)
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic", level=logging.WARNING
        ) as log:
            chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(chat_request.temperature, 1.0)
        self.assertTrue(any("clamp" in rec.lower() for rec in log.output))

    def test_temperature_below_zero_raises(self):
        serving = self._serving()
        request = self._anthropic_request(temperature=-0.1, stream=False)
        with self.assertRaises(ValueError):
            serving._convert_to_chat_completion_request(request)

    # ---------- G-10: service_tier ---------------------------------------------

    def test_service_tier_request_accepted_and_echoed_standard(self):
        serving = self._serving()
        request = self._anthropic_request(service_tier="auto", stream=False)
        # Request-side acceptance.
        self.assertEqual(request.service_tier, "auto")
        serving._convert_to_chat_completion_request(request)  # no raise
        # Response-side echo (spec §3.3).
        response = ChatCompletionResponse(
            id="chatcmpl-test",
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        )
        anthropic_response = serving._convert_response(response)
        body = anthropic_response.model_dump(exclude_none=True)
        self.assertEqual(body["usage"]["service_tier"], "standard")

    def test_service_tier_invalid_value_rejected(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            AnthropicMessagesRequest(
                model="test-model",
                max_tokens=16,
                messages=[{"role": "user", "content": "hi"}],
                service_tier="priority_only",
            )

    # ---------- G-13: tool_result is_error preserved ---------------------------

    def test_tool_result_is_error_prefixes_failure_marker(self):
        """G-13 (spec §2.2.2): ``is_error: true`` means a FAILED tool call
        — prefix the tool message so the model cannot mistake the error
        payload for a successful result."""
        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tu_1", "name": "fn", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "is_error": True,
                            "content": "connection refused",
                        }
                    ],
                },
            ]
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        tool_msg = chat_request.messages[-1]
        self.assertEqual(
            tool_msg.content, "[Tool execution failed] connection refused"
        )

    def test_tool_result_is_error_false_no_prefix(self):
        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tu_1", "name": "fn", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu_1", "content": "fine"}
                    ],
                },
            ]
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(chat_request.messages[-1].content, "fine")

    # ---------- G-25: cache reporting opt-in -----------------------------------

    def test_request_opts_into_per_request_cache_reporting(self):
        """G-25 (spec §3.3): Anthropic usage carries
        cache_read_input_tokens unconditionally — the adapter opts each
        request into cache reporting without flipping the OpenAI surface
        (no --enable-cache-report required)."""
        serving = self._serving()
        chat_request = serving._convert_to_chat_completion_request(
            self._anthropic_request()
        )
        self.assertTrue(chat_request.report_cached_tokens)

    # ---------- G-26: cache_control accept-and-log -----------------------------

    def test_cache_control_blocks_accepted_and_logged_once(self):
        import logging

        # The once-per-process compat flags live in convert.py; the test
        # resets the canonical owner.
        import sglang.srt.entrypoints.anthropic.convert as serving_module

        serving = self._serving()
        request = self._anthropic_request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "long context",
                            "cache_control": {"type": "ephemeral", "ttl": "5m"},
                        }
                    ],
                }
            ]
        )
        serving_module.ConversionContext.cache_control_logged = False
        try:
            with self.assertLogs(
                "sglang.srt.entrypoints.anthropic", level=logging.INFO
            ) as log:
                chat_request = serving._convert_to_chat_completion_request(request)
            self.assertTrue(any("cache_control" in rec for rec in log.output))
            # cache_control does not reach the OpenAI wire.
            self.assertNotIn(
                "cache_control", json.dumps(chat_request.model_dump(exclude_none=True))
            )
        finally:
            serving_module.ConversionContext.cache_control_logged = False

    # ---------- G-03 (B21): 100k message cap ------------------------------------

    def test_message_array_limit_validated(self):
        """G-03/B21 (spec §2.1): the 100,000-message cap is a client
        validation — > 100k rejects before the engine is engaged."""
        from pydantic import ValidationError

        huge = [{"role": "user", "content": "x"}] * (100_000 + 1)
        with self.assertRaises(ValidationError):
            AnthropicMessagesRequest(
                model="test-model",
                max_tokens=1,
                messages=huge,
            )
        # Exactly at cap passes.
        req = AnthropicMessagesRequest(
            model="test-model",
            max_tokens=1,
            messages=[{"role": "user", "content": "x"}] * 100_000,
        )
        self.assertEqual(len(req.messages), 100_000)

    # ---------- G-11: thinking cross-field validation ---------------------------

    def test_thinking_budget_below_max_tokens_without_beta_raises(self):
        """G-11 (spec §7.2): thinking budget must fit under max_tokens…"""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            AnthropicMessagesRequest(
                model="test-model",
                max_tokens=512,
                messages=[{"role": "user", "content": "hi"}],
                thinking={"type": "enabled", "budget_tokens": 1024},
            )
        self.assertIn("budget_tokens", str(ctx.exception))

    def test_thinking_budget_allowed_with_interleaved_beta(self):
        """G-11: the documented beta header lifts the max_tokens ≤ budget
        bound for interleaved thinking."""
        req = AnthropicMessagesRequest(
            model="test-model",
            max_tokens=256,
            betas=["interleaved-thinking-2025-05-14"],
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "enabled", "budget_tokens": 65536},
        )
        self.assertIsNotNone(req)

    def test_thinking_enabled_forced_tool_choice_raises(self):
        """G-11 (spec §7.2): thinking + forced tool issuance is a 400."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            AnthropicMessagesRequest(
                model="test-model",
                max_tokens=2048,
                messages=[{"role": "user", "content": "hi"}],
                thinking={"type": "enabled", "budget_tokens": 1024},
                tool_choice={"type": "any"},
            )

    def test_trailing_assistant_with_thinking_enabled_raises(self):
        """G-11 (spec §7.2): prefill + thinking is a 400."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            AnthropicMessagesRequest(
                model="test-model",
                max_tokens=2048,
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "let me think"},
                ],
                thinking={"type": "enabled", "budget_tokens": 1024},
            )

    # ---------- G-15: unknown blocks keep their extra fields -------------------

    def test_unknown_block_preserves_extra_fields(self):
        """G-15: GenericContentBlock round-trips unknown fields
        (extra="allow") AND still degrades to a placeholder."""
        from sglang.srt.entrypoints.anthropic.protocol import GenericContentBlock

        block = GenericContentBlock(type="halo_widget_v1", glow="blue", rank=2)
        self.assertEqual(block.glow, "blue")
        self.assertEqual(block.rank, 2)
        dumped = block.model_dump()
        self.assertEqual(dumped["glow"], "blue")

    # ---------- G-10: service_tier on count_tokens ----------------------------

    def test_count_tokens_accepts_service_tier(self):
        from pydantic import ValidationError

        from sglang.srt.entrypoints.anthropic.protocol import (
            AnthropicCountTokensRequest,
        )

        good = AnthropicCountTokensRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            service_tier="standard_only",
        )
        self.assertEqual(good.service_tier, "standard_only")
        with self.assertRaises(ValidationError):
            AnthropicCountTokensRequest(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                service_tier="priority_only",
            )

    # ---------- Claude Code acceptance corpus (notes §3-§5) --------------------

    def test_claude_code_request_envelope_parses_and_converts(self):
        """§3's real Claude Code envelope: adaptive thinking + omitted
        display, system list-blocks with cache_control, a system-role entry
        INSIDE messages, context_management, metadata user_id, betas —
        all accepted without a 400 (driving §5 checklist item 3)."""
        request = self._anthropic_request(
            max_tokens=32000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<system-reminder>ctx</system-reminder>",
                        },
                        {
                            "type": "text",
                            "text": "say hi",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
                {"role": "system", "content": "Agent docs injected mid-conversation"},
            ],
            system=[
                {
                    "type": "text",
                    "text": "You are Claude Code…",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            thinking={"type": "adaptive", "display": "omitted"},
            output_config={"effort": "high"},
            context_management={
                "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
            },
            metadata={"user_id": '{"device_id": "x", "session_id": "y"}'},
            betas=["claude-code-20250219", "interleaved-thinking-2025-05-14"],
        )
        serving = self._serving()
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertIsNotNone(chat_request)
        roles = [m.role for m in chat_request.messages]
        self.assertIn("user", roles)
        # The mid-conversation system entry folded per template handling.
        self.assertFalse(
            "unknown" in json.dumps(chat_request.model_dump()).lower()
        )
        # Usage channel opted in for the wire (§5 item 1: usage in
        # message_start always present in our stream frames).
        self.assertTrue(chat_request.report_cached_tokens)

    def test_echoed_thinking_empty_signature_tolerated(self):
        """CC notes §4/§5.5: the CLI normalizes a missing generation
        signature to ``signature: \"\"`` when echoing thinking blocks back
        in history. Empty and missing must BOTH parse and convert (never a
        fabricated signature on the wire)."""
        for sig in ("", None):
            payload = [{"type": "thinking", "thinking": "hmm"}]
            if sig is not None:
                payload[0]["signature"] = sig
            request = self._anthropic_request(
                messages=[
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": payload + [{"type": "text", "text": "answer"}],
                    },
                    {"role": "user", "content": "next"},
                ]
            )
            serving = self._serving()
            chat_request = serving._convert_to_chat_completion_request(request)
            self.assertIsNotNone(chat_request)
            self.assertNotIn("signature", json.dumps(chat_request.model_dump()))

    def test_tool_result_followup_cc_shape_converts(self):
        """CC §3 tool-result follow-up: user turn starts with a tool_result
        carrying string content, is_error:false and cache_control
        (well-formed pair stays green)."""
        serving = self._serving()
        request = self._anthropic_request(
            tools=[
                {
                    "type": "custom",
                    "name": "Bash",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                }
            ],
            messages=[
                {"role": "user", "content": "run echo hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01ABC",
                            "name": "Bash",
                            "input": {"command": "echo hi"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_01ABC",
                            "content": "hi",
                            "is_error": False,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ]
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        roles = [m.role for m in chat_request.messages]
        self.assertEqual(roles, ["user", "assistant", "tool"])
        # is_error=false ⇒ NO failure marker.
        self.assertEqual(chat_request.messages[-1].content, "hi")
        tool_msg = chat_request.messages[-1]
        self.assertEqual(tool_msg.tool_call_id, "toolu_01ABC")
        # Declared custom tool converted (NOT classified as a server tool).
        self.assertIsNotNone(chat_request.tools)
        self.assertEqual(chat_request.tools[0].function.name, "Bash")

    # ---------- G-22: zero-line OpenAI stream ----------------------------------

    def test_stream_backend_zero_lines_degrades_to_error_frames(self):
        """G-22: an OpenAI stream that exhausts with zero lines yields a
        self-contained terminal error sequence (not a half-open SSE)."""
        serving = self._serving(stream_lines=[])
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "message_start")
        self.assertIn("error", types)
        err = next(e for e in events if e["type"] == "error")
        self.assertEqual(err["error"]["type"], "api_error")
        self.assertEqual(types[-1], "message_stop")

    # ---------- G-02: error envelope request id ---------------------------------

    def test_error_response_request_id_single_source_scope(self):
        """G-02 (spec §5.1): the error BODY's ``request_id`` takes the id
        published by ``AnthropicRequestIdMiddleware`` on the ASGI scope
        — single source, so body id ALWAYS equals the ``request-id``
        response header; without a middleware stamp it mints a fresh
        ``req_…`` (unit paths / missing scope key)."""
        serving = self._serving()
        fake_raw = SimpleNamespace(
            scope={"anthropic.request_id": "req_from_scope"}
        )
        response = serving._error_response(
            400, "invalid_request_error", "bad", raw_request=fake_raw
        )
        payload = json.loads(response.body)
        self.assertEqual(payload["request_id"], "req_from_scope")

        # Scope present but key absent → mint.
        fake_raw2 = SimpleNamespace(scope={})
        body_id = json.loads(
            serving._error_response(500, "api_error", "bad", raw_request=fake_raw2).body
        )["request_id"]
        self.assertTrue(body_id.startswith("req_"))
        self.assertNotEqual(body_id, "req_from_scope")
        # No raw_request at all → mint.
        body_id2 = json.loads(
            serving._error_response(500, "api_error", "bad").body
        )["request_id"]
        self.assertTrue(body_id2.startswith("req_"))


class TestDetectInlineSystemSupport(unittest.TestCase):
    """Chat-template detection for mid-conversation system messages (#28883)."""

    def test_guarded_template_not_supported(self):
        guarded = (
            "{%- for message in messages %}"
            "{%- if message.role == 'system' and not loop.first %}"
            "{{- raise_exception('system must be first') }}"
            "{%- endif %}"
            "{%- endfor %}"
        )
        self.assertFalse(detect_inline_system_support(guarded))

    def test_inline_template_supported(self):
        inline = (
            "{%- for message in messages %}"
            "{{- message.role }}: {{ message.content }}\n"
            "{%- endfor %}"
        )
        self.assertTrue(detect_inline_system_support(inline))

    def test_silent_drop_template_not_supported(self):
        # Renders only the leading system; silently ignores later system turns.
        silent_drop = (
            "{%- if messages[0].role == 'system' %}"
            "{{ messages[0].content }}\n"
            "{%- endif %}"
            "{%- for message in messages %}"
            "{%- if message.role in ('user', 'assistant') %}"
            "{{ message.role }}: {{ message.content }}\n"
            "{%- endif %}"
            "{%- endfor %}"
        )
        self.assertFalse(detect_inline_system_support(silent_drop))

    def test_no_template_not_supported(self):
        self.assertFalse(detect_inline_system_support(None))
        self.assertFalse(detect_inline_system_support(""))


if __name__ == "__main__":
    unittest.main()
