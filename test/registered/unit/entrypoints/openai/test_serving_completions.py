"""
Unit-tests for the refactored completions-serving handler (no pytest).
Run with:
    python -m unittest tests.test_serving_completions_unit -v
"""

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import asyncio
import json
import unittest
from http import HTTPStatus
from typing import Optional
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException, Request
from test_serving_chat import _MockTemplateManager as _MockChatTemplateManager
from test_serving_chat import _MockTokenizerManager
from utils import generation_error_chunk, generation_fault_reasons

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_MATCHED_STR
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _spec_result(index):
    return {
        "text": f"choice-{index}",
        "meta_info": {
            "id": "cmpl-spec-test",
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "cached_tokens": 0,
            "finish_reason": {"type": "stop"},
            "weight_version": "default",
            "spec_accept_rate": 0.5,
            "spec_accept_length": 2.0,
            "spec_cap_length": index + 1.0,
            "spec_block_accept_length": index + 0.5,
            "spec_num_correct_drafts": 1,
            "spec_num_proposed_drafts": 2,
            "spec_verify_ct": 1,
            "spec_correct_drafts_histogram": [0, 1],
            "spec_cap_lens_histogram": [index, 1],
        },
        "index": index,
    }


class _MockTemplateManager:
    """Minimal mock for TemplateManager."""

    def __init__(self):
        self.chat_template_name: Optional[str] = None
        self.jinja_template_content_format: Optional[str] = None
        self.completion_template_name: Optional[str] = (
            None  # Set to None to avoid template processing
        )
        self.jinja_template_may_reorder_tool_results = False


class ServingCompletionTestCase(unittest.TestCase):
    """Bundle all prompt/echo tests in one TestCase."""

    # ---------- shared test fixtures ----------
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        # build the mock TokenizerManager once for every test
        tm = Mock(spec=TokenizerManager)

        tm.tokenizer = Mock()
        tm.tokenizer.encode.return_value = [1, 2, 3, 4]
        tm.tokenizer.decode.return_value = "decoded text"
        tm.tokenizer.bos_token_id = 1

        tm.model_config = Mock(is_multimodal=False)
        tm.server_args = Mock(enable_cache_report=False)

        tm.generate_request = AsyncMock()
        tm.create_abort_task = Mock()

        self.template_manager = _MockTemplateManager()
        self.sc = OpenAIServingCompletion(tm, self.template_manager)
        self.fastapi_request = Mock(spec=Request)

    # ---------- prompt-handling ----------
    def test_single_token_ids_prompt(self):
        req = CompletionRequest(model="x", prompt=[1, 2, 3, 4], max_tokens=100)
        internal, _ = self.sc._convert_to_internal_request(req)
        self.assertEqual(internal.input_ids, [1, 2, 3, 4])

    def test_cache_salt_and_extra_key_remain_distinct(self):
        req = CompletionRequest(
            model="x",
            prompt=[1, 2, 3, 4],
            max_tokens=1,
            cache_salt="tenant-a",
            extra_key="classification",
        )
        internal, _ = self.sc._convert_to_internal_request(req)
        self.assertEqual(internal.cache_salt, "tenant-a")
        self.assertEqual(internal.extra_key, "classification")

    def test_single_request_rejects_batched_cache_salt(self):
        req = CompletionRequest(
            model="x",
            prompt=[1, 2, 3, 4],
            max_tokens=1,
            cache_salt=["tenant-a"],
        )
        internal, _ = self.sc._convert_to_internal_request(req)
        with self.assertRaisesRegex(ValueError, "single request"):
            internal.normalize_batch_and_arguments()

    # ---------- echo-handling ----------
    def test_echo_with_list_of_strings_streaming(self):
        req = CompletionRequest(
            model="x", prompt=["A", "B"], max_tokens=1, echo=True, n=1
        )
        self.assertEqual(self.sc._get_echo_text(req, 0), "A")
        self.assertEqual(self.sc._get_echo_text(req, 1), "B")

    def test_echo_with_token_ids_streaming(self):
        req = CompletionRequest(model="x", prompt=[1, 2, 3], max_tokens=1, echo=True)
        self.sc.tokenizer_manager.tokenizer.decode.return_value = "decoded_prompt"
        self.assertEqual(self.sc._get_echo_text(req, 0), "decoded_prompt")

    def test_echo_with_multiple_token_ids_streaming(self):
        req = CompletionRequest(
            model="x", prompt=[[1, 2], [3, 4]], max_tokens=1, echo=True, n=1
        )
        self.sc.tokenizer_manager.tokenizer.decode.return_value = "decoded"
        self.assertEqual(self.sc._get_echo_text(req, 0), "decoded")

    def test_prepare_echo_prompts_non_streaming(self):
        # single string
        req = CompletionRequest(model="x", prompt="Hi", echo=True)
        self.assertEqual(self.sc._prepare_echo_prompts(req), ["Hi"])

        # list of strings
        req = CompletionRequest(model="x", prompt=["Hi", "Yo"], echo=True)
        self.assertEqual(self.sc._prepare_echo_prompts(req), ["Hi", "Yo"])

        # token IDs
        req = CompletionRequest(model="x", prompt=[1, 2, 3], echo=True)
        self.sc.tokenizer_manager.tokenizer.decode.return_value = "decoded"
        self.assertEqual(self.sc._prepare_echo_prompts(req), ["decoded"])

    # ---------- response_format handling ----------
    def test_response_format_json_object(self):
        """Test that response_format json_object is correctly processed in sampling params."""
        req = CompletionRequest(
            model="x",
            prompt="Generate a JSON object:",
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        sampling_params = self.sc._build_sampling_params(req)
        self.assertEqual(sampling_params["json_schema"], '{"type": "object"}')

    def test_response_format_json_schema(self):
        """Test that response_format json_schema is correctly processed in sampling params."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        req = CompletionRequest(
            model="x",
            prompt="Generate a JSON object:",
            max_tokens=100,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "person", "schema": schema},
            },
        )
        sampling_params = self.sc._build_sampling_params(req)
        # The schema should be converted to string by convert_json_schema_to_str
        self.assertIn("json_schema", sampling_params)
        self.assertIsInstance(sampling_params["json_schema"], str)

    def test_response_format_json_schema_missing_schema(self):
        """Test that json_schema response_format without a schema raises a ValueError."""
        req = CompletionRequest(
            model="x",
            prompt="Generate a JSON object:",
            max_tokens=100,
            response_format={"type": "json_schema"},
        )
        with self.assertRaises(ValueError):
            self.sc._build_sampling_params(req)

    def test_response_format_structural_tag(self):
        """Test that response_format structural_tag is correctly processed in sampling params."""
        req = CompletionRequest(
            model="x",
            prompt="Generate structured output:",
            max_tokens=100,
            response_format={
                "type": "structural_tag",
                "structures": [{"begin": "<data>", "end": "</data>"}],
                "triggers": ["<data>"],
            },
        )
        sampling_params = self.sc._build_sampling_params(req)
        # The structural_tag should be processed
        self.assertIn("structural_tag", sampling_params)
        self.assertIsInstance(sampling_params["structural_tag"], str)

    def test_response_format_none(self):
        """Test that no response_format doesn't add extra constraints."""
        req = CompletionRequest(model="x", prompt="Generate text:", max_tokens=100)
        sampling_params = self.sc._build_sampling_params(req)
        # Should not have json_schema or structural_tag from response_format
        # (but might have json_schema from the legacy json_schema field)
        self.assertIsNone(sampling_params.get("structural_tag"))

    def test_non_streaming_response(self):
        req = CompletionRequest(
            model="x",
            prompt="Hello",
            max_tokens=10,
            logprobs=False,
            return_token_ids=True,
        )

        mock_ret = [
            {
                "text": " world",
                "output_ids": [3, 4],
                "prompt_token_ids": [1, 2],
                "meta_info": {
                    "id": "test-id",
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "finish_reason": {"type": "stop"},
                    "weight_version": "v1",
                },
            }
        ]

        response = self.sc._build_completion_response(req, mock_ret, 1234567890)

        self.assertEqual(len(response.choices), 1)
        self.assertEqual(response.choices[0].text, " world")
        self.assertEqual(len(response.choices[0].logprobs.top_logprobs), 0)
        self.assertEqual(response.choices[0].token_ids, [3, 4])
        self.assertEqual(response.choices[0].prompt_token_ids, [1, 2])

    def test_streaming_abort_yields_error(self):
        """Test that an abort finish reason during streaming correctly yields an error and stops."""
        err_msg = "Aborted by scheduler"
        err_code = HTTPStatus.INTERNAL_SERVER_ERROR

        async def _mock_generate_abort(*args, **kwargs):
            yield {
                "text": "Partial ",
                "meta_info": {
                    "id": "cmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {
                        "type": "abort",
                        "status_code": err_code,
                        "message": err_msg,
                    },
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.sc.tokenizer_manager.generate_request = _mock_generate_abort

        req = CompletionRequest(
            model="x",
            prompt="Hello world",
            max_tokens=100,
            stream=True,
        )

        adapted_request, _ = self.sc._convert_to_internal_request(req)

        async def run_stream():
            chunks = []
            try:
                async for chunk in self.sc._generate_completion_stream(
                    adapted_request, req, self.fastapi_request
                ):
                    chunks.append(chunk)
            except Exception as e:
                print(f"Error during stream iteration: {e}")
            return chunks

        loop = get_or_create_event_loop()
        chunks = loop.run_until_complete(run_stream())

        error_chunk_data = None
        for c in chunks:
            if "error" in c:
                error_chunk_data = json.loads(c[len("data: ") :])
                break
        self.assertIsNotNone(error_chunk_data, "Error chunk not found in stream")
        self.assertEqual(error_chunk_data["error"]["message"], err_msg)
        self.assertEqual(error_chunk_data["error"]["code"], err_code.value)

        # Ensure the stream stops after the abort error
        # The last chunk should be "data: [DONE]\n\n"
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")

        # Check that there is an error chunk and a DONE chunk, and possibly a role chunk
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("error", chunks[0])

    def test_streaming_token_ids_deltas_cover_output_exactly(self):
        req = CompletionRequest(
            model="x",
            prompt="Hi",
            max_tokens=10,
            stream=True,
            return_token_ids=True,
        )
        adapted_request, _ = self.sc._convert_to_internal_request(req)

        for incremental in (False, True):
            # Both of these are read through `get_serving()` now, so assigning
            # them on the mock manager's record has no effect on what the code
            # under test sees. State them where the code reads them.
            with (
                self.subTest(incremental_streaming_output=incremental),
                get_context().override_server_args(
                    stream_response_default_include_usage=False,
                    incremental_streaming_output=incremental,
                ),
            ):
                texts = ("a", "b", "c") if incremental else ("a", "ab", "abc")
                output_ids = (
                    ([5], [6], [7]) if incremental else ([5], [5, 6], [5, 6, 7])
                )
                chunks = [
                    {
                        "text": text,
                        "output_ids": ids,
                        "prompt_token_ids": [1, 2],
                        "meta_info": {
                            "id": "cmpl-test",
                            "prompt_tokens": 2,
                            "completion_tokens": i + 1,
                            "finish_reason": {"type": "stop"} if i == 2 else None,
                        },
                        "index": 0,
                    }
                    for i, (text, ids) in enumerate(zip(texts, output_ids))
                ]

                async def _mock_generate(*args, _chunks=chunks, **kwargs):
                    for chunk in _chunks:
                        yield chunk

                self.sc.tokenizer_manager.generate_request = _mock_generate

                async def run_stream():
                    return [
                        chunk
                        async for chunk in self.sc._generate_completion_stream(
                            adapted_request, req, self.fastapi_request
                        )
                    ]

                loop = get_or_create_event_loop()
                raw_chunks = loop.run_until_complete(run_stream())

                choices = []
                for raw in raw_chunks:
                    if not raw.startswith("data: ") or raw.strip() == "data: [DONE]":
                        continue
                    data = json.loads(raw[len("data: ") :])
                    choices.extend(data.get("choices", []))

                token_ids = [tid for c in choices for tid in c.get("token_ids", [])]
                text = "".join(c["text"] for c in choices)
                self.assertEqual(text, "abc")
                self.assertEqual(token_ids, [5, 6, 7])
                self.assertEqual(choices[0]["prompt_token_ids"], [1, 2])
                for choice in choices[1:]:
                    self.assertNotIn("prompt_token_ids", choice)

    def test_non_streaming_cached_tokens_details_emits_sglext(self):
        """Test that non-streaming completion responses emit cached token details in sglext."""

        req = CompletionRequest(
            model="x",
            prompt="Hello world",
            max_tokens=100,
            return_cached_tokens_details=True,
        )
        ret = [
            {
                "text": "Cached response",
                "meta_info": {
                    "id": "cmpl-cache-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 6,
                    "cached_tokens_details": {
                        "device": 4,
                        "host": 1,
                        "storage": 1,
                        "storage_backend": "file",
                    },
                    "finish_reason": {"type": "stop", "matched": None},
                    "weight_version": "default",
                },
            }
        ]

        response = self.sc._build_completion_response(req, ret, 1234567890)

        self.assertIsNotNone(response.sglext)
        self.assertEqual(
            response.sglext.cached_tokens_details.model_dump(exclude_none=True),
            {
                "device": 4,
                "host": 1,
                "storage": 1,
                "storage_backend": "file",
            },
        )

    def test_parallel_sampling_returns_spec_details_per_choice(self):
        req = CompletionRequest(
            model="x",
            prompt="Hello world",
            max_tokens=100,
            n=2,
            return_spec_tokens_details=True,
        )
        ret = [_spec_result(index) for index in range(2)]

        response = self.sc._build_completion_response(req, ret, 1234567890)

        details = response.sglext.spec_tokens_details
        self.assertEqual(len(details), 2)
        self.assertEqual(details[0].spec_cap_length, 1.0)
        self.assertEqual(details[0].spec_block_accept_length, 0.5)
        self.assertEqual(details[0].spec_cap_lens_histogram, [0, 1])
        self.assertEqual(details[1].spec_cap_length, 2.0)
        self.assertEqual(details[1].spec_block_accept_length, 1.5)
        self.assertEqual(details[1].spec_cap_lens_histogram, [1, 1])

        single_req = req.model_copy(update={"n": 1})
        single_response = self.sc._build_completion_response(
            single_req, ret[:1], 1234567890
        )
        self.assertEqual(
            single_response.sglext.spec_tokens_details.spec_cap_length,
            1.0,
        )

        disabled_req = single_req.model_copy(
            update={"return_spec_tokens_details": False}
        )
        disabled_response = self.sc._build_completion_response(
            disabled_req, ret[:1], 1234567890
        )
        self.assertIsNone(disabled_response.sglext)

    def test_streaming_parallel_sampling_orders_spec_details_by_choice(self):
        async def mock_generate(*args, **kwargs):
            for index in (1, 0):
                yield _spec_result(index)

        self.sc.tokenizer_manager.generate_request = mock_generate
        req = CompletionRequest(
            model="x",
            prompt="Hello world",
            max_tokens=100,
            n=2,
            stream=True,
            return_spec_tokens_details=True,
        )
        adapted_request, _ = self.sc._convert_to_internal_request(req)

        async def run_stream(request):
            return [
                chunk
                async for chunk in self.sc._generate_completion_stream(
                    adapted_request, request, self.fastapi_request
                )
            ]

        chunks = get_or_create_event_loop().run_until_complete(run_stream(req))
        parsed = [
            json.loads(chunk[len("data: ") :])
            for chunk in chunks
            if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]"
        ]
        details = next(chunk["sglext"] for chunk in parsed if "sglext" in chunk)[
            "spec_tokens_details"
        ]
        self.assertEqual([item["spec_cap_length"] for item in details], [1.0, 2.0])
        self.assertEqual(
            [item["spec_cap_lens_histogram"] for item in details],
            [[0, 1], [1, 1]],
        )

        async def mock_single_generate(*args, **kwargs):
            async for content in mock_generate():
                if content["index"] == 0:
                    yield content

        self.sc.tokenizer_manager.generate_request = mock_single_generate
        single_req = req.model_copy(update={"n": 1})
        single_chunks = get_or_create_event_loop().run_until_complete(
            run_stream(single_req)
        )
        single_parsed = [
            json.loads(chunk[len("data: ") :])
            for chunk in single_chunks
            if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]"
        ]
        single_details = next(
            chunk["sglext"] for chunk in single_parsed if "sglext" in chunk
        )["spec_tokens_details"]
        self.assertIsInstance(single_details, dict)

    def test_streaming_cached_tokens_details_emits_sglext(self):
        """Test that streaming completion responses emit cached token details in sglext."""

        async def _mock_generate_with_cached_tokens_details(*args, **kwargs):
            yield {
                "text": "Cached response",
                "meta_info": {
                    "id": "cmpl-cache-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 6,
                    "cached_tokens_details": {
                        "device": 4,
                        "host": 1,
                        "storage": 1,
                        "storage_backend": "file",
                    },
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.sc.tokenizer_manager.generate_request = (
            _mock_generate_with_cached_tokens_details
        )

        req = CompletionRequest(
            model="x",
            prompt="Hello world",
            max_tokens=100,
            stream=True,
            return_cached_tokens_details=True,
        )

        adapted_request, _ = self.sc._convert_to_internal_request(req)

        async def run_stream():
            chunks = []
            async for chunk in self.sc._generate_completion_stream(
                adapted_request, req, self.fastapi_request
            ):
                chunks.append(chunk)
            return chunks

        loop = get_or_create_event_loop()
        chunks = loop.run_until_complete(run_stream())

        sglext_chunks = []
        for chunk in chunks:
            if not chunk.startswith("data: ") or chunk.strip() == "data: [DONE]":
                continue
            data = json.loads(chunk[len("data: ") :])
            if "sglext" in data:
                sglext_chunks.append(data)

        self.assertEqual(len(sglext_chunks), 1)
        self.assertEqual(sglext_chunks[0]["choices"], [])
        self.assertEqual(
            sglext_chunks[0]["sglext"]["cached_tokens_details"],
            {
                "device": 4,
                "host": 1,
                "storage": 1,
                "storage_backend": "file",
            },
        )


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
        template = _MockChatTemplateManager()
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
            for reason, status in generation_fault_reasons():
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
                            chunks = (
                                [generation_error_chunk({"type": "stop"})]
                                if partial
                                else []
                            )
                            chunks.append(generation_error_chunk(reason, index=1))
                            chunks.append(
                                generation_error_chunk({"type": "stop"}, index=1)
                            )
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
                chunk = generation_error_chunk(
                    FINISH_ABORT("engine failed", 500).to_json()
                )
                del chunk["output_ids"]
                wire = self.run_request(serving, request, raw, [chunk])
                self.assertEqual(len(wire), 2)
                self.assertEqual(json.loads(wire[0][6:])["error"]["code"], 500)
                self.assertEqual(wire[1], "data: [DONE]\n\n")

    def test_fault_error_preserves_producer_message(self):
        """Invalid-token stops retain their cause in streamed and full errors."""
        cases = (
            (
                FINISH_MATCHED_STR("NaN happened", err_type="invalid_token").to_json(),
                "NaN happened",
            ),
            (
                {"type": "stop", "err_type": "invalid_token", "matched": 7},
                "Generation aborted.",
            ),
            (FINISH_ABORT("engine failed", 500).to_json(), "engine failed"),
        )
        for chat in (True, False):
            for stream in (True, False):
                for reason, message in cases:
                    with self.subTest(chat=chat, stream=stream, reason=reason):
                        serving, request, raw = self.fixture(chat, stream=stream)
                        response = self.run_request(
                            serving, request, raw, [generation_error_chunk(reason)]
                        )
                        if stream:
                            self.assertEqual(len(response), 2)
                            self.assertEqual(response[-1], "data: [DONE]\n\n")
                            error = json.loads(response[0][6:])["error"]
                        else:
                            self.assertEqual(response.status_code, 500)
                            error = json.loads(response.body)
                        self.assertEqual(error["message"], message)

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
                        serving, request, raw, [generation_error_chunk(reason)]
                    )
                    self.assertEqual(wire[-1], "data: [DONE]\n\n")
                    footer = json.loads(wire[-2][6:])
                    self.assertEqual(footer["choices"], [])
                    self.assertEqual(footer["usage"]["total_tokens"], 7)

    def test_nonstream_fault_result_cannot_be_serialized_as_success(self):
        """Fault metadata that does not raise in the tokenizer still returns an error."""
        for chat in (True, False):
            for reason, status in generation_fault_reasons():
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
                                generation_error_chunk({"type": "stop"}),
                                generation_error_chunk(reason, index=1),
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
