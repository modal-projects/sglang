"""Unit tests for srt/entrypoints/sse_keepalive.py"""

import asyncio
import unittest

import orjson

from sglang.srt.entrypoints.sse_keepalive import (
    KEEPALIVE_COMMENT,
    SSEKeepaliveMiddleware,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

INTERVAL = 0.02
SSE_HEADERS = [(b"content-type", b"text/event-stream")]
JSON_HEADERS = [(b"content-type", b"application/json")]


def _app(delay, status=200, headers=SSE_HEADERS, body=b"data: hi\n\n"):
    async def app(scope, receive, send):
        request = await receive()
        assert request["type"] == "http.request"
        await asyncio.sleep(delay)
        await send(
            {"type": "http.response.start", "status": status, "headers": headers}
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    return app


def _run(app, stream=True, path="/v1/chat/completions", interval=INTERVAL):
    body = orjson.dumps({"stream": stream})
    pending = [
        {"type": "http.request", "body": body[:4], "more_body": True},
        {"type": "http.request", "body": body[4:], "more_body": False},
    ]

    async def receive():
        return pending.pop(0)

    sent = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": path}
    asyncio.run(SSEKeepaliveMiddleware(app, interval=interval)(scope, receive, send))
    return sent


def _bodies(sent):
    return [m["body"] for m in sent if m["type"] == "http.response.body"]


class TestSSEKeepaliveMiddleware(CustomTestCase):
    def test_queued_stream_gets_sse_headers_and_keepalives(self):
        sent = _run(_app(delay=INTERVAL * 6))
        self.assertEqual(sent[0]["type"], "http.response.start")
        self.assertEqual(sent[0]["status"], 200)
        self.assertIn(
            (b"content-type", b"text/event-stream; charset=utf-8"), sent[0]["headers"]
        )
        self.assertEqual(sum(m["type"] == "http.response.start" for m in sent), 1)
        bodies = _bodies(sent)
        self.assertIn(KEEPALIVE_COMMENT, bodies)
        self.assertEqual(bodies[-1], b"data: hi\n\n")

    def test_fast_stream_is_untouched(self):
        sent = _run(_app(delay=0))
        self.assertEqual(sent[0]["headers"], SSE_HEADERS)
        self.assertEqual(_bodies(sent), [b"data: hi\n\n"])

    def test_late_error_becomes_sse_error_event(self):
        error = b'{"error": {"message": "boom"}}'
        sent = _run(
            _app(delay=INTERVAL * 6, status=400, headers=JSON_HEADERS, body=error)
        )
        self.assertEqual(sent[0]["status"], 200)
        self.assertEqual(_bodies(sent)[-1], b"data: " + error + b"\n\ndata: [DONE]\n\n")

    def test_early_error_keeps_http_status(self):
        sent = _run(_app(delay=0, status=400, headers=JSON_HEADERS, body=b"{}"))
        self.assertEqual(sent[0]["status"], 400)

    def test_non_stream_and_disabled_requests_pass_through(self):
        for kwargs in (
            dict(stream=False),
            dict(path="/generate"),
            dict(interval=0),
        ):
            with self.subTest(**kwargs):
                sent = _run(_app(delay=INTERVAL * 4, headers=JSON_HEADERS), **kwargs)
                self.assertEqual(sent[0]["headers"], JSON_HEADERS)
                self.assertNotIn(KEEPALIVE_COMMENT, _bodies(sent))

    def test_interval_defaults_to_env(self):
        with envs.SGLANG_K3_SSE_KEEPALIVE_S.override(15):
            self.assertEqual(SSEKeepaliveMiddleware(_app(0)).interval, 15)
        self.assertEqual(SSEKeepaliveMiddleware(_app(0)).interval, 0)


if __name__ == "__main__":
    unittest.main()
