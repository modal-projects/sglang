#!/usr/bin/env python3
"""E2E harness: REAL sglang Anthropic Messages API code over real HTTP,
backed by a FAKE OpenAIServingChat (no model, no engine, no GPU).

Architecture:
    Claude Code CLI --(HTTP/SSE)--> THIS server
        FastAPI app (scratch) whose /v1/messages routes are the REAL sglang
        route functions from sglang.srt.entrypoints.http_server, with
        app.state.anthropic_serving = AnthropicServing(<fake chat>) where
        AnthropicServing is the REAL sglang class under test.

The fake chat implements the exact duck-type surface AnthropicServing uses
(enumerated from serving.py):
  tokenizer_manager (tokenizer.chat_template, tokenizer.model_config.is_multimodal)
  apply_reasoning_enabled(chat_request, enabled)
  wrap_reasoning_history(text)
  _validate_request(chat_request)
  _convert_to_internal_request(chat_request, raw_request)
  _handle_non_streaming_request(adapted, processed, raw_request)
  _generate_chat_stream(adapted, processed, raw_request)  -> async gen of "data: ..\\n\\n" SSE lines
  _process_messages(chat_request, is_multimodal)          -> for count_tokens

Behavior is scripted off the CONVERTED (OpenAI-shaped) request, so the test
exercises sglang's real Anthropic->OpenAI conversion, streaming translation,
and error envelopes with the real Claude Code CLI as the client.

Run:  ../.venv312/bin/python sglang_anthropic_harness.py --port 8078
Env:  PYTHONPATH=/home/ec2-user/sglang/python SGLANG_CACHE_DIR=/tmp/sglang-e2e-cache
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
TRANSCRIPTS = HERE / "transcripts_sglang"
TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
JSONL = TRANSCRIPTS / "requests.jsonl"

sys.path.insert(0, "/home/ec2-user/sglang/python")
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import sglang.srt.entrypoints.http_server as hs
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse

MARKER = "MOCK_SGLANG_E2E_42"


def _jlog(payload: dict):
    payload["_ts"] = time.time()
    with JSONL.open("a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for c in content or []:
        t = getattr(c, "text", None)
        if t:
            parts.append(t)
        elif isinstance(c, dict) and c.get("type") == "text":
            parts.append(c.get("text", ""))
    return " ".join(parts)


class E2EFakeOpenAIServingChat:
    """Duck-typed stand-in for OpenAIServingChat over a scripted generator."""

    def __init__(self):
        self.apply_reasoning_calls = []
        from starlette.background import BackgroundTasks

        self.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template=None),
            model_config=SimpleNamespace(is_multimodal=False),
            create_abort_task=lambda obj: BackgroundTasks(),  # no-op abort plumbing
        )

    # ---- conversion-time hooks ----------------------------------------
    def apply_reasoning_enabled(self, chat_request, enabled):
        self.apply_reasoning_calls.append(bool(enabled))

    def wrap_reasoning_history(self, text):
        return f"<think>\n{text}\n</think>"

    # ---- shared plumbing ----------------------------------------------
    def _validate_request(self, chat_request):
        return None

    def _convert_to_internal_request(self, chat_request, raw_request):
        return SimpleNamespace(), chat_request

    def _process_messages(self, chat_request, is_multimodal):
        # count_tokens needs .prompt_ids; 1 fake token per rendered char /4
        rendered = json.dumps(chat_request.model_dump(exclude_none=True), default=str)
        return SimpleNamespace(prompt_ids=[0] * max(1, len(rendered) // 4))

    # ---- scripted behavior --------------------------------------------
    def _plan(self, chat_request):
        """Decide a plan dict from the CONVERTED OpenAI request."""
        msgs = chat_request.messages or []
        dump = chat_request.model_dump(exclude_none=True)
        roles = [getattr(m, "role", None) for m in msgs]
        n_tool_msgs = sum(1 for r in roles if r == "tool")
        n_system_msgs = sum(1 for r in roles if r == "system")
        has_tools = bool(getattr(chat_request, "tools", None))
        last_user = ""
        for m in reversed(msgs):
            if getattr(m, "role", None) == "user":
                last_user = _text_of(getattr(m, "content", None))
                break
        lowered = last_user.lower()
        _jlog(
            {
                "stage": "converted_chat_request",
                "roles": roles,
                "n_messages": len(msgs),
                "has_tools": has_tools,
                "tool_choice": str(getattr(chat_request, "tool_choice", None)),
                "stream": getattr(chat_request, "stream", None),
                "max_tokens": getattr(chat_request, "max_tokens", None),
                "reasoning_effort": getattr(chat_request, "reasoning_effort", None),
                "last_user_tail": last_user[-200:],
                "dump_head": json.dumps(dump, default=str)[:3000],
            }
        )
        if n_tool_msgs > 0:
            tool_msg = [m for m in msgs if getattr(m, "role", None) == "tool"][0]
            content = getattr(tool_msg, "content", None)
            return {
                "kind": "text",
                "text": (
                    f"E2E-OK: sglang delivered tool_result as role=tool "
                    f"(tool_call_id={getattr(tool_msg, 'tool_call_id', None)}; "
                    f"content={str(content)[:80]!r}). Round trip through REAL "
                    f"sglang anthropic code complete. Marker={MARKER}"
                ),
            }
        if "refuse" in lowered:
            return {"kind": "text", "text": "E2E: refusing.", "finish": "content_filter"}
        if getattr(chat_request, "max_tokens", None) == 1:
            # G-05 clamp wire-visibility: a max_tokens=0 prewarm clamped to 1
            # reaches the fake engine here (chat_request.max_tokens == 1) and
            # must wire back as a one-token "length" finish.
            return {"kind": "text", "text": "WARM", "finish": "length"}
        if "stopword" in lowered:
            # Simulate a scheduler stop-sequence match: the finish chunk
            # carries matched_stop=<first requested stop (else "STOP")>. The
            # anthropic layer must surface stop_reason="stop_sequence" (S1:
            # only when the matched string is a member of stop_sequences).
            stop = chat_request.stop
            matched = stop[0] if isinstance(stop, list) and stop else (stop if isinstance(stop, str) else None)
            return {
                "kind": "text",
                "text": "E2E: body before stop. STOP",
                "matched_stop": matched,
            }
        if "think please" in lowered:
            return {"kind": "think", "thinking": "pondering inside sglang e2e", "text": "E2E: thought complete."}
        if has_tools and ("add" in lowered or "tool" in lowered or n_system_msgs > 0):
            # n_system_msgs>0 covers the claude envelope whose reminder text lists tools
            return {
                "kind": "tool_calls",
                "name": "Bash",
                "arguments": {"command": f"echo {MARKER}", "description": "Echo e2e marker"},
            }
        return {
            "kind": "text",
            "text": (
                f"E2E-OK from sglang AnthropicServing. roles={roles} "
                f"n_system={n_system_msgs} has_tools={has_tools} "
                f"tail={last_user[-60:]!r}"
            ),
        }

    # ---- serving entry points used by AnthropicServing -----------------
    async def _handle_non_streaming_request(self, adapted_request, processed_request, raw_request):
        plan = self._plan(processed_request)
        message = {"role": "assistant"}
        finish = "stop"
        if plan["kind"] == "tool_calls":
            message["content"] = ""
            message["tool_calls"] = [
                {
                    "id": "call_" + uuid.uuid4().hex[:12],
                    "type": "function",
                    "function": {"name": plan["name"], "arguments": json.dumps(plan["arguments"])},
                }
            ]
            finish = "tool_calls"
        elif plan["kind"] == "think":
            message["reasoning_content"] = plan["thinking"]
            message["content"] = plan["text"]
        else:
            message["content"] = plan["text"]
            finish = plan.get("finish", "stop")
        choice_obj = {"index": 0, "message": message, "finish_reason": finish}
        if plan.get("matched_stop") is not None:
            choice_obj["matched_stop"] = plan["matched_stop"]
        return ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-e2e",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": getattr(processed_request, "model", "e2e-fake"),
                "choices": [choice_obj],
                "usage": {"prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59},
            }
        )

    def _generate_chat_stream(self, adapted_request, processed_request, raw_request):
        plan = self._plan(processed_request)

        def chunk(choices=None, usage=None):
            data = {
                "id": "chatcmpl-e2e",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": getattr(processed_request, "model", "e2e-fake"),
                "choices": choices or [],
            }
            if usage is not None:
                data["usage"] = usage
            return f"data: {json.dumps(data)}\n\n"

        def choice(delta, finish=None):
            return {"index": 0, "delta": delta, "finish_reason": finish}

        usage = {"prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59}
        lines = [chunk([choice({"role": "assistant", "content": ""})])]
        if plan["kind"] == "tool_calls":
            lines.append(
                chunk(
                    [
                        choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_" + uuid.uuid4().hex[:12],
                                        "type": "function",
                                        "function": {
                                            "name": plan["name"],
                                            "arguments": json.dumps(plan["arguments"]),
                                        },
                                    }
                                ]
                            }
                        )
                    ]
                )
            )
            lines.append(chunk([choice({}, "tool_calls")], usage=usage))
        elif plan["kind"] == "think":
            for piece in [plan["thinking"]]:
                lines.append(chunk([choice({"reasoning_content": piece})]))
            for piece in [plan["text"]]:
                lines.append(chunk([choice({"content": piece})]))
            lines.append(chunk([choice({}, "stop")], usage=usage))
        else:
            text = plan["text"]
            words = text.split(" ")
            for i, w in enumerate(words):
                piece = w + (" " if i < len(words) - 1 else "")
                lines.append(chunk([choice({"content": piece})]))
            # matched_stop lives on the CHOICE object (OpenAI schema:457/497/1197),
            # NOT inside delta — mimics what serving_chat's finish chunks emit.
            final_choice = choice({}, plan.get("finish", "stop"))
            if plan.get("matched_stop") is not None:
                final_choice["matched_stop"] = plan["matched_stop"]
            lines.append(chunk([final_choice], usage=usage))
        lines.append("data: [DONE]\n\n")

        async def _gen():
            for line in lines:
                yield line
            await asyncio.sleep(0)

        return _gen()


def build_app() -> FastAPI:
    app = FastAPI(openapi_url=None)

    # REAL sglang route functions + REAL validation dependency + REAL
    # Anthropic-style validation-error handler — only the engine/lifespan
    # is replaced.
    app.add_exception_handler(RequestValidationError, hs.validation_exception_handler)
    app.post("/v1/messages", dependencies=[Depends(hs.validate_json_request)])(
        hs.anthropic_v1_messages
    )
    app.post("/v1/messages/count_tokens", dependencies=[Depends(hs.validate_json_request)])(
        hs.anthropic_v1_count_tokens
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        if request.url.path.startswith("/v1/"):
            body = await request.body()
            try:
                parsed = json.loads(body)
            except Exception:
                parsed = {"_raw": body.decode(errors="replace")[:2000]}
            _jlog(
                {
                    "stage": "http_request",
                    "method": request.method,
                    "path": request.url.path,
                    "headers": dict(request.headers),
                    "body": parsed,
                }
            )
            # body was consumed; rebuild receive channel
            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}

            request = Request(request.scope, receive)
        response = await call_next(request)
        return response

    @app.get("/health")
    async def health():
        return {"ok": True}

    app.state.anthropic_serving = AnthropicServing(E2EFakeOpenAIServingChat())
    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8078)
    args = parser.parse_args()
    JSONL.write_text("")
    from fastapi.exceptions import RequestValidationError as _rv
    _ = _rv
    uvicorn.run(build_app(), host="127.0.0.1", port=args.port, log_level="warning")
