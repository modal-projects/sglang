"""Engine failures must terminate OpenAI responses without final token usage."""

import asyncio
import json
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException, Request
from openai_harmony import Role
from test_serving_chat import _MockTemplateManager, _MockTokenizerManager
from utils import collect_stream_events, event_payloads, make_serving

from sglang.srt.entrypoints.context import (
    HarmonyContext,
    SimpleContext,
    StreamingHarmonyContext,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    RequestResponseMetadata,
    ResponsesRequest,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_MATCHED_STR
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def fault_reasons():
    for status in (HTTPStatus.INTERNAL_SERVER_ERROR, HTTPStatus.BAD_GATEWAY, 599):
        reason = FINISH_ABORT("engine failed", status).to_json()
        yield reason, int(status)
        # JSON/MessagePack transport turns HTTPStatus into an ordinary integer.
        if isinstance(status, HTTPStatus):
            yield json.loads(json.dumps(reason)), int(status)
        yield {**reason, "status_code": str(int(status))}, int(status)
    yield FINISH_ABORT("encoder failed", 408, "encoder_timeout").to_json(), 408
    yield FINISH_MATCHED_STR("invalid output", err_type="invalid_token").to_json(), 500


def engine_chunk(reason=None, *, index=0):
    return {
        "text": "Partial",
        "index": index,
        "prompt_token_ids": [1, 2, 3],
        "output_ids": [4, 5],
        "meta_info": {
            "id": "cmpl-fault-test",
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "cached_tokens": 1,
            "reasoning_tokens": 0,
            "hidden_states": [[0.5], [0.25]],
            "finish_reason": reason,
        },
    }


class GenerationErrorsTestCase(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", enable_cache_report=True), role="tokenizer"
        )

    def fixture(self, chat, *, stream=True, continuous=False, optional_fields=False):
        manager = _MockTokenizerManager()
        manager.request_logger = Mock(log_requests=False)
        template = _MockTemplateManager()
        options = {"include_usage": True, "continuous_usage_stats": continuous}
        kwargs = {
            "model": "x",
            "stream": stream,
            "stream_options": options if stream else None,
            "return_hidden_states": True,
        }
        if chat:
            serving = OpenAIServingChat(manager, template)
            request = ChatCompletionRequest(
                messages=[{"role": "user", "content": "Hi"}],
                return_input_ids_in_sglext=True,
                return_output_ids_in_sglext=True,
                logprobs=optional_fields,
                **kwargs,
            )
        else:
            serving = OpenAIServingCompletion(manager, template)
            request = CompletionRequest(
                prompt="Hi",
                return_token_ids=optional_fields,
                logprobs=1 if optional_fields else None,
                **kwargs,
            )
        raw = Mock(spec=Request)
        raw.headers = {}
        return serving, request, raw

    def run_request(self, serving, request, raw, chunks):
        async def generate(*args, **kwargs):
            for chunk in chunks:
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk

        serving.tokenizer_manager.generate_request = generate

        async def collect():
            with patch.object(
                serving,
                "_convert_to_internal_request",
                return_value=(GenerateReqInput(text="Hi"), request),
            ):
                response = await serving.handle_request(request, raw)
            if request.stream:
                self.assertEqual(response.status_code, 200)
                return [chunk async for chunk in response.body_iterator]
            return response

        return asyncio.run(collect())

    def test_fault_stream_ends_at_error_without_usage_or_buffered_metadata(self):
        """A fault ends all choices; only earlier continuous usage can remain."""
        for chat in (True, False):
            for reason, status in fault_reasons():
                for partial in (False, True):
                    for continuous in (False, True):
                        with self.subTest(
                            chat=chat,
                            reason=reason,
                            partial=partial,
                            continuous=continuous,
                        ):
                            serving, request, raw = self.fixture(
                                chat, continuous=continuous
                            )
                            request.n = 2
                            chunks = [engine_chunk({"type": "stop"})] if partial else []
                            chunks.append(engine_chunk(reason, index=1))
                            chunks.append(engine_chunk({"type": "stop"}, index=1))
                            wire = self.run_request(serving, request, raw, chunks)
                            self.assertEqual(wire[-1], "data: [DONE]\n\n")
                            events = [
                                json.loads(x.removeprefix("data: ")) for x in wire[:-1]
                            ]
                            self.assertEqual(events[-1]["error"]["code"], status)
                            self.assertEqual(sum("error" in e for e in events), 1)
                            self.assertFalse(
                                any(e.get("choices") == [] for e in events)
                            )
                            usages = [e["usage"] for e in events if e.get("usage")]
                            self.assertEqual(bool(usages), partial and continuous)

    def test_fault_before_optional_output_fields_has_only_error_and_done(self):
        """Terminal faults do not require token IDs or logprob payloads."""
        for chat in (True, False):
            with self.subTest(chat=chat):
                serving, request, raw = self.fixture(chat, optional_fields=True)
                chunk = engine_chunk(FINISH_ABORT("engine failed", 500).to_json())
                del chunk["output_ids"]
                wire = self.run_request(serving, request, raw, [chunk])
                self.assertEqual(len(wire), 2)
                self.assertEqual(json.loads(wire[0][6:])["error"]["code"], 500)
                self.assertEqual(wire[1], "data: [DONE]\n\n")

    def test_rejection_cancellation_and_natural_finish_keep_usage(self):
        """Usage policy remains distinct from failure and cancellation status."""
        reasons = [
            FINISH_ABORT("request stopped", status).to_json()
            for status in (400, 408, 429, 503, None)
        ]
        reasons += [
            FINISH_ABORT("cancelled", 400, "cancelled").to_json(),
            {"type": "stop"},
            {"type": "length", "length": 2},
        ]
        for chat in (True, False):
            for reason in reasons:
                with self.subTest(chat=chat, reason=reason):
                    serving, request, raw = self.fixture(chat)
                    wire = self.run_request(
                        serving, request, raw, [engine_chunk(reason)]
                    )
                    self.assertEqual(wire[-1], "data: [DONE]\n\n")
                    footer = json.loads(wire[-2][6:])
                    self.assertEqual(footer["choices"], [])
                    self.assertEqual(footer["usage"]["total_tokens"], 7)

    def test_nonstream_fault_result_cannot_be_serialized_as_success(self):
        """Fault metadata that does not raise in the tokenizer still returns an error."""
        for chat in (True, False):
            for reason, status in fault_reasons():
                with self.subTest(chat=chat, reason=reason):
                    serving, request, raw = self.fixture(
                        chat, stream=False, optional_fields=True
                    )
                    result = self.run_request(
                        serving,
                        request,
                        raw,
                        [
                            [
                                engine_chunk({"type": "stop"}),
                                engine_chunk(reason, index=1),
                            ]
                        ],
                    )
                    self.assertEqual(result.status_code, status)
                    body = json.loads(result.body)
                    self.assertEqual(body["object"], "error")
                    self.assertNotIn("usage", body)
                    self.assertNotIn("choices", body)

    def test_nonstream_transport_errors_keep_status_without_usage(self):
        """Errors raised by the tokenizer keep their HTTP status and error body."""
        for chat in (True, False):
            for status in (400, 408, 429, 500, 502, 503):
                with self.subTest(chat=chat, status=status):
                    serving, request, raw = self.fixture(chat, stream=False)
                    response = self.run_request(
                        serving,
                        request,
                        raw,
                        [HTTPException(status_code=status, detail="generation failed")],
                    )
                    self.assertEqual(response.status_code, status)
                    self.assertNotIn("usage", json.loads(response.body))


class ResponsesErrorsTestCase(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", enable_response_store=True), role="tokenizer"
        )

    def test_create_responses_preserves_transport_error_status(self):
        """The outer request handler must not relabel engine errors as HTTP 400."""
        for status in (408, 429, 500, 502, 503):
            with self.subTest(status=status):
                serving = make_serving()
                serving.use_harmony = False
                serving.default_chat_template_kwargs = {}
                serving.template_manager.chat_template_name = None
                serving.template_manager.jinja_template_content_format = "string"
                serving.tokenizer_manager.tokenizer.apply_chat_template.return_value = [
                    1,
                    2,
                    3,
                ]
                serving.tokenizer_manager.abort_request = Mock()
                serving.reasoning_parser = None
                serving.tool_call_parser = None

                async def generate(*args, **kwargs):
                    raise HTTPException(status_code=status, detail="generation failed")
                    yield

                serving.tokenizer_manager.generate_request = generate
                response = asyncio.run(
                    serving.create_responses(
                        ResponsesRequest(model="x", input="hi", store=False)
                    )
                )
                self.assertEqual(response.status_code, status)
                self.assertNotIn("usage", json.loads(response.body))

    def test_terminal_outcomes_usage_and_storage_agree(self):
        """Faults fail without usage while release cancellation/length semantics survive."""
        cases = [
            (reason, "failed", "server_error", False) for reason, _ in fault_reasons()
        ]
        for code in (400, 408, 429, "429", 503, "503", None):
            cases.append(
                (
                    FINISH_ABORT("stopped", code).to_json(),
                    "failed",
                    "rate_limit_exceeded"
                    if str(code) == "429"
                    else "invalid_prompt"
                    if code in (400, 408)
                    else "server_error",
                    True,
                )
            )
        cases += [
            (
                FINISH_ABORT("cancelled", 400, "cancelled").to_json(),
                "failed",
                "server_error",
                True,
            ),
            ({"type": "stop"}, "completed", None, True),
            ({"type": "length", "length": 2}, "incomplete", None, True),
        ]
        for reason, status, error_code, has_usage in cases:
            for harmony in (False, True):
                for stream in (False, True):
                    with self.subTest(reason=reason, harmony=harmony, stream=stream):
                        serving = make_serving()
                        serving.reasoning_parser = None
                        serving.tool_call_parser = None
                        serving.use_harmony = harmony
                        request = ResponsesRequest(
                            model="x", input="hi", stream=stream, store=True
                        )
                        metadata = RequestResponseMetadata(
                            request_id=request.request_id
                        )
                        chunk = engine_chunk(reason)
                        context = SimpleContext()
                        context.last_output = chunk
                        if harmony:
                            context = Mock(
                                spec=StreamingHarmonyContext
                                if stream
                                else HarmonyContext
                            )
                            context.finish_reason = reason
                            context.num_prompt_tokens = 5
                            context.num_output_tokens = 2
                            context.num_cached_tokens = 1
                            context.num_reasoning_tokens = 0
                            context.messages = []
                            context.num_init_messages = 0
                            context.parser = SimpleNamespace(
                                current_content="Partial",
                                current_role=Role.ASSISTANT,
                                current_channel="final",
                                current_recipient=None,
                            )

                        async def generate():
                            yield context if harmony else chunk

                        kwargs = {
                            "request": request,
                            "sampling_params": {},
                            "result_generator": generate(),
                            "model_name": "x",
                            "tokenizer": Mock(),
                            "request_metadata": metadata,
                            "require_reasoning": False,
                        }
                        if not stream:
                            response = asyncio.run(
                                serving.responses_full_generator(
                                    context=context, **kwargs
                                )
                            ).model_dump()
                        else:
                            generator = (
                                serving.responses_stream_generator(
                                    context=context, **kwargs
                                )
                                if harmony
                                else serving.responses_stream_generator_non_harmony(
                                    **kwargs
                                )
                            )
                            events = event_payloads(
                                asyncio.run(collect_stream_events(generator))
                            )
                            terminal = events[-1]
                            self.assertEqual(terminal["type"], f"response.{status}")
                            self.assertEqual(
                                [e["sequence_number"] for e in events],
                                list(range(len(events))),
                            )
                            self.assertEqual(
                                sum(
                                    e["type"]
                                    in (
                                        "response.failed",
                                        "response.completed",
                                        "response.incomplete",
                                    )
                                    for e in events
                                ),
                                1,
                            )
                            response = terminal["response"]
                        self.assertEqual(response["status"], status)
                        self.assertTrue(response["output"])
                        self.assertEqual(response["usage"] is not None, has_usage)
                        self.assertEqual(
                            metadata.final_usage_info is not None, has_usage
                        )
                        if error_code:
                            self.assertEqual(response["error"]["code"], error_code)
                        else:
                            self.assertIsNone(response["error"])
                        stored = serving.response_store[request.request_id]
                        self.assertEqual(stored.status, status)
                        self.assertEqual(stored.usage is not None, has_usage)


if __name__ == "__main__":
    unittest.main()
