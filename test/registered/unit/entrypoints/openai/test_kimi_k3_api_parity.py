"""Unit tests for srt/entrypoints/openai/kimi_k3_api_parity.py"""

import asyncio
import unittest

from utils import make_serving

from sglang.srt.entrypoints.openai import kimi_k3_api_parity
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ResponsesRequest,
)
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _chat_request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="x", messages=[{"role": "user", "content": "hi"}], **kwargs
    )


class TestKimiK3SamplingEnvelope(CustomTestCase):
    def _validate(self, **overrides):
        kwargs = dict(
            temperature=None,
            top_p=None,
            presence_penalty=0,
            frequency_penalty=0,
            n=1,
        )
        kwargs.update(overrides)
        return kimi_k3_api_parity.validate_sampling_envelope(**kwargs)

    def test_envelope_bounds(self):
        self.assertIsNone(self._validate())
        self.assertIsNone(self._validate(temperature=1.0, top_p=0.95))
        cases = {
            "temperature": dict(temperature=1.5),
            "top_p": dict(top_p=0.9),
            "presence_penalty": dict(presence_penalty=0.5),
            "frequency_penalty": dict(frequency_penalty=0.5),
            "n must be 1": dict(n=2),
        }
        for needle, overrides in cases.items():
            with self.subTest(needle=needle):
                self.assertIn(needle, self._validate(**overrides))


class TestKimiK3ReasoningEffort(CustomTestCase):
    def test_supported_and_omitted_efforts_pass(self):
        self.assertIsNone(kimi_k3_api_parity.validate_reasoning_effort(_chat_request()))
        for effort in kimi_k3_api_parity.KIMI_K3_THINKING_EFFORTS:
            with self.subTest(effort=effort):
                request = _chat_request(reasoning_effort=effort)
                self.assertIsNone(kimi_k3_api_parity.validate_reasoning_effort(request))

    def test_unsupported_effort_and_disabled_thinking_are_rejected(self):
        request = _chat_request(chat_template_kwargs={"thinking_effort": "medium"})
        self.assertIn(
            "thinking_effort='medium'",
            kimi_k3_api_parity.validate_reasoning_effort(request),
        )
        request = _chat_request(chat_template_kwargs={"thinking": False})
        self.assertIn(
            "cannot disable thinking",
            kimi_k3_api_parity.validate_reasoning_effort(request),
        )


class TestKimiK3ServingGates(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.serving = make_serving()

    def test_chat_validation_only_applies_to_kimi_k3(self):
        request = _chat_request(top_p=0.5)
        self.assertIsNone(self.serving._validate_request(request))
        self.serving.chat_encoding_spec = "kimi_k3"
        self.assertIn("top_p", self.serving._validate_request(request))

    def test_responses_model_name_is_only_free_form_for_kimi_k3(self):
        # tool_choice="required" without tools fails right after the model
        # check, so the error shows whether the model check ran.
        request = ResponsesRequest(
            model="unknown-model", input="hi", store=False, tool_choice="required"
        )
        response = asyncio.run(self.serving.create_responses(request))
        self.assertEqual(response.status_code, 404)

        self.serving.chat_encoding_spec = "kimi_k3"
        response = asyncio.run(self.serving.create_responses(request))
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"tool_choice", response.body)


if __name__ == "__main__":
    unittest.main()
