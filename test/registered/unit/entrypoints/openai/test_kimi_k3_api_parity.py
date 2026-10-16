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


class TestKimiK3ReasoningControls(CustomTestCase):
    def _drop(self, **kwargs):
        request = _chat_request(**kwargs)
        kimi_k3_api_parity.drop_unsupported_reasoning_controls(request)
        return request

    def test_supported_and_omitted_efforts_are_kept(self):
        self.assertIsNone(self._drop().reasoning_effort)
        for effort in kimi_k3_api_parity.KIMI_K3_THINKING_EFFORTS:
            with self.subTest(effort=effort):
                self.assertEqual(
                    self._drop(reasoning_effort=effort).reasoning_effort, effort
                )
                kwargs = self._drop(chat_template_kwargs={"thinking_effort": effort})
                self.assertEqual(
                    kwargs.chat_template_kwargs, {"thinking_effort": effort}
                )

    def test_unsupported_effort_and_disabled_thinking_are_dropped(self):
        self.assertIsNone(self._drop(reasoning_effort="medium").reasoning_effort)
        request = self._drop(
            chat_template_kwargs={
                "thinking": False,
                "enable_thinking": False,
                "reasoning_effort": "none",
                "thinking_effort": "medium",
                "other": 1,
            }
        )
        self.assertEqual(request.chat_template_kwargs, {"other": 1})


class TestKimiK3SamplingDefaults(CustomTestCase):
    def _apply(self, model_defaults, top_p=None, temperature=None):
        params = {"top_p": 1.0, "temperature": 0.7}
        kimi_k3_api_parity.apply_sampling_defaults(
            params, model_defaults, top_p, temperature
        )
        return params

    def test_omitted_values_get_moonshot_defaults(self):
        self.assertEqual(self._apply({}), {"top_p": 0.95, "temperature": 1.0})

    def test_request_and_generation_config_values_win(self):
        unchanged = {"top_p": 1.0, "temperature": 0.7}
        self.assertEqual(self._apply({}, top_p=1.0, temperature=0.7), unchanged)
        self.assertEqual(self._apply({"top_p": 1.0, "temperature": 0.7}), unchanged)


class TestKimiK3ServingGates(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.serving = make_serving()

    def test_kimi_k3_accepts_prod_b300_sampling_and_reasoning_params(self):
        self.serving.chat_encoding_spec = "kimi_k3"
        request = _chat_request(
            temperature=1.3,
            top_p=0.5,
            presence_penalty=0.5,
            frequency_penalty=0.3,
            n=2,
            reasoning_effort="medium",
            chat_template_kwargs={"enable_thinking": False},
        )
        self.assertIsNone(self.serving._validate_request(request))
        self.assertIsNone(request.reasoning_effort)
        self.assertNotIn(False, request.chat_template_kwargs.values())

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
