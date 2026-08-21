"""Unit tests for srt/utils/anthropic_http.py — pure-ASGI, no server needed.

Covers audit items G-02 (request-id), G-03 (413 cap) and the G-01/G-24 helper
primitives. A tiny fake-ASGI harness keeps these tests dependency-light.
"""

import asyncio
import json
import re
import unittest
from datetime import datetime, timezone

from sglang.srt.utils.anthropic_http import (
    ANTHROPIC_MAX_BODY_BYTES,
    ANTHROPIC_REQUEST_ID_SCOPE_KEY,
    AnthropicBodySizeLimitMiddleware,
    AnthropicOverloadedStatusMiddleware,
    AnthropicRequestIdMiddleware,
    anthropic_error_body,
    anthropic_error_spec,
    anthropic_model_entry,
    anthropic_model_list,
    is_anthropic_messages_path,
    new_anthropic_request_id,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

REQUEST_ID_RE = re.compile(rb"^req_[0-9a-f]{32}$")


def _scope(path="/v1/messages", method="POST", headers=None):
    return {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "server": ("testserver", 80),
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers or [],
    }


def _run_asgi(app, scope, receive_messages=()):
    """Drive an ASGI callable; return the list of messages it sent."""
    incoming = iter(receive_messages)
    sent = []

    async def receive():
        try:
            return next(incoming)
        except StopIteration:
            return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return sent


def _headers_of(start_message):
    return {
        name.lower(): value for name, value in start_message.get("headers", [])
    }


async def _json_app(scope, receive, send):
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": b'{"ok":true}'})


async def _sse_app(scope, receive, send):
    """Fake SSE stream: header flush, then several body chunks."""
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": b"event: message_start\ndata: {}\n\n",
            "more_body": True,
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": b"event: message_stop\ndata: {}\n\n",
            "more_body": False,
        }
    )


class TestHelpers(CustomTestCase):
    def test_is_anthropic_messages_path(self):
        self.assertTrue(is_anthropic_messages_path("/v1/messages"))
        self.assertTrue(is_anthropic_messages_path("/v1/messages/count_tokens"))
        self.assertFalse(is_anthropic_messages_path("/v1/chat/completions"))
        self.assertFalse(is_anthropic_messages_path("/v1/models"))

    def test_new_request_id_format(self):
        rid = new_anthropic_request_id()
        self.assertRegex(rid, r"^req_[0-9a-f]{32}$")
        self.assertNotEqual(rid, new_anthropic_request_id())

    def test_error_body_shape(self):
        body = anthropic_error_body(error_type="overloaded_error", message="Overloaded")
        self.assertEqual(
            body,
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
            },
        )
        with_id = anthropic_error_body(
            error_type="authentication_error", message="m", request_id="req_abc"
        )
        self.assertEqual(with_id["request_id"], "req_abc")


class TestG02RequestIdMiddleware(CustomTestCase):
    """G-02 (spec §1.3): request-id header on every /v1/messages* response."""

    def test_json_response_gets_header(self):
        scope = _scope("/v1/messages")
        sent = _run_asgi(AnthropicRequestIdMiddleware(_json_app), scope)
        self.assertEqual(sent[0]["type"], "http.response.start")
        rid = _headers_of(sent[0])[b"request-id"]
        self.assertRegex(rid, REQUEST_ID_RE)
        # Body forwarded untouched.
        self.assertEqual(sent[1]["body"], b'{"ok":true}')

    def test_streaming_response_gets_header_and_unchanged_body(self):
        sent = _run_asgi(AnthropicRequestIdMiddleware(_sse_app), _scope())
        self.assertEqual(len(sent), 3)
        rid = _headers_of(sent[0])[b"request-id"]
        self.assertRegex(rid, REQUEST_ID_RE)
        # SSE frames forwarded verbatim (no buffering/rewriting).
        self.assertEqual(sent[1]["body"], b"event: message_start\ndata: {}\n\n")
        self.assertTrue(sent[1]["more_body"])
        self.assertEqual(sent[2]["body"], b"event: message_stop\ndata: {}\n\n")

    def test_scope_key_published_for_inner_layers(self):
        """serving.py reads this key to embed the same request_id in error bodies."""
        observed = {}

        async def probe_app(scope, receive, send):
            observed["rid"] = scope.get(ANTHROPIC_REQUEST_ID_SCOPE_KEY)
            await _json_app(scope, receive, send)

        sent = _run_asgi(AnthropicRequestIdMiddleware(probe_app), _scope())
        header_rid = _headers_of(sent[0])[b"request-id"].decode()
        self.assertEqual(observed["rid"], header_rid)

    def test_error_response_gets_header(self):
        async def error_app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 400,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"type":"error","error":{"type":"invalid_request_error","message":"x"}}',
                }
            )

        sent = _run_asgi(AnthropicRequestIdMiddleware(error_app), _scope())
        self.assertEqual(sent[0]["status"], 400)
        self.assertRegex(_headers_of(sent[0])[b"request-id"], REQUEST_ID_RE)

    def test_no_duplicate_header_when_inner_sets_one(self):
        async def preset_app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"request-id", b"req_inner")],
                }
            )
            await send({"type": "http.response.body", "body": b"{}"})

        sent = _run_asgi(AnthropicRequestIdMiddleware(preset_app), _scope())
        rids = [
            value
            for name, value in sent[0]["headers"]
            if name.lower() == b"request-id"
        ]
        self.assertEqual(rids, [b"req_inner"])

    def test_non_messages_paths_pass_through_byte_identical(self):
        for path in ("/v1/chat/completions", "/v1/models", "/generate"):
            scope = _scope(path)
            sent = _run_asgi(AnthropicRequestIdMiddleware(_json_app), scope)
            self.assertNotIn(b"request-id", _headers_of(sent[0]), msg=path)
            self.assertNotIn(ANTHROPIC_REQUEST_ID_SCOPE_KEY, scope, msg=path)

    def test_non_http_scopes_pass_through(self):
        async def lifespan_app(scope, receive, send):
            await send({"type": "lifespan.startup.complete"})

        sent = _run_asgi(
            AnthropicRequestIdMiddleware(lifespan_app), {"type": "lifespan"}
        )
        self.assertEqual(sent, [{"type": "lifespan.startup.complete"}])


class TestAnthropicErrorSpec(CustomTestCase):
    """spec §5.2 canonical status→(type,message) owner (G-04/G-24 + 5xx scrub)."""

    def test_503_is_overloaded_with_canonical_message(self):
        spec = anthropic_error_spec(status_code=503)
        self.assertEqual(spec["error_type"], "overloaded_error")
        self.assertEqual(spec["message"], "Overloaded")
        # Wire status is NOT translated here — the middleware owns 503→529.

    def test_504_is_timeout_error(self):
        self.assertEqual(anthropic_error_spec(status_code=504)["error_type"], "timeout_error")

    def test_status_type_mapping(self):
        for code, expected in (
            (400, "invalid_request_error"),
            (401, "authentication_error"),
            (403, "permission_error"),
            (404, "not_found_error"),
            (413, "request_too_large"),
            (422, "invalid_request_error"),
            (429, "rate_limit_error"),
            (500, "api_error"),
            (502, "api_error"),
        ):
            self.assertEqual(
                anthropic_error_spec(status_code=code)["error_type"], expected, msg=code
            )

    def test_unlisted_status_falls_back_to_api_error(self):
        self.assertEqual(
            anthropic_error_spec(status_code=418)["error_type"], "api_error"
        )

    def test_5xx_detail_is_scrubbed(self):
        spec = anthropic_error_spec(status_code=502, detail="stacktrace with /paths")
        self.assertEqual(spec["message"], "Internal server error")

    def test_4xx_detail_echoed_or_defaulted(self):
        self.assertEqual(
            anthropic_error_spec(status_code=400, detail="bad field")["message"],
            "bad field",
        )
        self.assertEqual(anthropic_error_spec(status_code=404)["message"], "Request failed")


class TestG24OverloadedStatusMiddleware(CustomTestCase):
    """G-24 (spec §5.2): single owner of 503→529 (+ canonical body) on /v1/messages*."""

    async def _make_503_envelope_app(self, scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 503,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", b"66"),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b'{"type":"error","error":{"type":"api_error","message":"INTERNALS LEAK"}}',
            }
        )

    async def _make_503_plain_app(self, scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 503,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"engine busy"})

    def test_503_translated_to_529_and_body_normalized(self):
        scope = _scope()
        scope[ANTHROPIC_REQUEST_ID_SCOPE_KEY] = "req_test123"
        sent = _run_asgi(
            AnthropicOverloadedStatusMiddleware(self._make_503_envelope_app), scope
        )
        self.assertEqual(sent[0]["status"], 529)
        body = json.loads(sent[1]["body"])
        self.assertEqual(
            body,
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
                "request_id": "req_test123",
            },
        )
        # Content-Length fixed up for the rewritten body.
        self.assertEqual(
            _headers_of(sent[0])[b"content-length"], str(len(sent[1]["body"])).encode()
        )

    def test_503_body_id_generated_without_scope_key(self):
        sent = _run_asgi(
            AnthropicOverloadedStatusMiddleware(self._make_503_envelope_app), _scope()
        )
        body = json.loads(sent[1]["body"])
        self.assertRegex(body["request_id"], r"^req_[0-9a-f]{32}$")

    def test_non_envelope_503_body_untouched(self):
        sent = _run_asgi(
            AnthropicOverloadedStatusMiddleware(self._make_503_plain_app), _scope()
        )
        self.assertEqual(sent[0]["status"], 529)
        self.assertEqual(sent[1]["body"], b"engine busy")

    def test_non_503_untouched(self):
        sent = _run_asgi(AnthropicOverloadedStatusMiddleware(_json_app), _scope())
        self.assertEqual(sent[0]["status"], 200)

    def test_other_paths_keep_503(self):
        sent = _run_asgi(
            AnthropicOverloadedStatusMiddleware(self._make_503_envelope_app),
            _scope("/v1/chat/completions"),
        )
        self.assertEqual(sent[0]["status"], 503)


class TestG03BodySizeLimitMiddleware(CustomTestCase):
    """G-03 (spec §1.4): POST /v1/messages* bodies > 32 MB → 413 request_too_large."""

    def _assert_413_envelope(self, sent):
        self.assertEqual(sent[0]["status"], 413)
        body = json.loads(sent[1]["body"])
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "request_too_large")
        return body

    def test_oversize_content_length_rejected(self):
        scope = _scope(headers=[(b"content-length", b"33554433")])  # 32MB + 1
        sent = _run_asgi(AnthropicBodySizeLimitMiddleware(_json_app), scope)
        body = self._assert_413_envelope(sent)
        self.assertRegex(body["request_id"], r"^req_[0-9a-f]{32}$")
        self.assertEqual(
            _headers_of(sent[0])[b"request-id"].decode(), body["request_id"]
        )

    def test_at_limit_content_length_passes(self):
        scope = _scope(headers=[(b"content-length", str(ANTHROPIC_MAX_BODY_BYTES).encode())])
        sent = _run_asgi(AnthropicBodySizeLimitMiddleware(_json_app), scope)
        self.assertEqual(sent[0]["status"], 200)

    def test_chunked_over_limit_rejected(self):
        scope = _scope(headers=[(b"transfer-encoding", b"chunked")])
        frames = [
            {"type": "http.request", "body": b"x" * (16 * 1024 * 1024), "more_body": True},
            {"type": "http.request", "body": b"y" * (17 * 1024 * 1024), "more_body": False},
        ]
        sent = _run_asgi(
            AnthropicBodySizeLimitMiddleware(_json_app), scope, frames
        )
        self._assert_413_envelope(sent)

    def test_chunked_under_limit_replayed_verbatim(self):
        frames_in = [
            {"type": "http.request", "body": b'{"a":', "more_body": True},
            {"type": "http.request", "body": b"1}", "more_body": False},
        ]
        received_bodies = []

        async def sink_app(scope, receive, send):
            while True:
                msg = await receive()
                if msg["type"] != "http.request":
                    break
                received_bodies.append(msg["body"])
                if not msg.get("more_body"):
                    break
            await _json_app(scope, receive, send)

        scope = _scope(headers=[(b"transfer-encoding", b"chunked")])
        sent = _run_asgi(AnthropicBodySizeLimitMiddleware(sink_app), scope, frames_in)
        self.assertEqual(received_bodies, [b'{"a":', b"1}"])
        self.assertEqual(sent[0]["status"], 200)

    def test_413_reuses_outer_request_id(self):
        """Composed as in production: request-id middleware outside → same id everywhere."""
        app = AnthropicRequestIdMiddleware(AnthropicBodySizeLimitMiddleware(_json_app))
        scope = _scope(headers=[(b"content-length", b"33554433")])
        sent = _run_asgi(app, scope)
        body = self._assert_413_envelope(sent)
        self.assertEqual(
            _headers_of(sent[0])[b"request-id"].decode(), body["request_id"]
        )
        self.assertEqual(body["request_id"], scope[ANTHROPIC_REQUEST_ID_SCOPE_KEY])

    def test_non_post_and_other_paths_pass_through(self):
        for scope in (
            _scope("/v1/messages", method="GET", headers=[(b"content-length", b"99999999")]),
            _scope("/v1/chat/completions", headers=[(b"content-length", b"99999999")]),
            _scope("/v1/models", headers=[(b"content-length", b"99999999")]),
        ):
            sent = _run_asgi(AnthropicBodySizeLimitMiddleware(_json_app), scope)
            self.assertEqual(sent[0]["status"], 200, msg=scope["path"])


class TestG27ModelPayloads(CustomTestCase):
    """G-27 (spec §12.2): Anthropic-shaped model objects."""

    def test_entry_shape(self):
        entry = anthropic_model_entry(
            model_id="claude-ish", created=1_700_000_000, max_model_len=4096
        )
        self.assertEqual(entry["type"], "model")
        self.assertEqual(entry["id"], "claude-ish")
        self.assertEqual(entry["display_name"], "claude-ish")
        self.assertEqual(
            entry["created_at"],
            datetime.fromtimestamp(1_700_000_000, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        )
        self.assertTrue(entry["created_at"].endswith("Z"))
        self.assertEqual(entry["max_input_tokens"], 4096)

    def test_entry_omits_unknown_context_len(self):
        entry = anthropic_model_entry(model_id="lora", created=0, max_model_len=None)
        self.assertNotIn("max_input_tokens", entry)
        self.assertEqual(entry["created_at"], "1970-01-01T00:00:00Z")

    def test_list_envelope(self):
        entries = [
            anthropic_model_entry(model_id="a", created=0, max_model_len=1),
            anthropic_model_entry(model_id="b", created=0, max_model_len=None),
        ]
        payload = anthropic_model_list(entries)
        self.assertEqual(payload["first_id"], "a")
        self.assertEqual(payload["last_id"], "b")
        self.assertIs(payload["has_more"], False)
        self.assertEqual(len(payload["data"]), 2)

    def test_list_envelope_empty(self):
        payload = anthropic_model_list([])
        self.assertIsNone(payload["first_id"])
        self.assertIsNone(payload["last_id"])
        self.assertEqual(payload["data"], [])


if __name__ == "__main__":
    unittest.main()
