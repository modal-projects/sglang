"""Route-layer unit tests for the Anthropic HTTP contract in http_server.py.

Covers the http_server-owned slices of docs_new/anthropic_gap_audit.md:
G-24 (503→529), G-03 caps and G-27 models content-negotiation. G-01/G-02 are
tested with their own modules (utils/auth.py, utils/anthropic_http.py). G-02's
scope-key contract for serving.py is asserted here as well.
"""

import asyncio
import json
import unittest

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from fastapi import HTTPException  # noqa: E402
from starlette.requests import Request  # noqa: E402

from sglang.srt.entrypoints.http_server import app  # noqa: E402
from sglang.srt.utils.anthropic_http import ANTHROPIC_REQUEST_ID_SCOPE_KEY  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _http_exception_handler():
    """The @app.exception_handler(HTTPException) function (its module-global name
    is shadowed by the RequestValidationError handler, so look it up in the app)."""
    return app.exception_handlers[HTTPException]


def _fake_request(path: str, method: str = "POST") -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "server": ("testserver", 80),
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [],
        }
    )


class TestG24OverloadedStatus(unittest.TestCase):
    """G-24 (spec §5.2): the HTTPException handler builds the canonical
    overloaded_error BODY (via anthropic_error_spec) while the *wire* 503→529
    translation is owned solely by AnthropicOverloadedStatusMiddleware."""

    def _handle_503(self, path="/v1/messages"):
        return asyncio.run(
            _http_exception_handler()(
                _fake_request(path),
                HTTPException(status_code=503, detail="engine busy"),
            )
        )

    def test_handler_503_body_is_canonical_overload_shape(self):
        resp = self._handle_503()
        # No 529 inside the handler: the middleware owns that decision.
        self.assertEqual(resp.status_code, 503)
        body = json.loads(resp.body)
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "overloaded_error")
        self.assertEqual(body["error"]["message"], "Overloaded")

    def test_529_full_chain_body_request_id_matches_header(self):
        """Full production chain: request-id middleware → overload middleware
        → HTTPException handler envelope. Asserts spec §5.1 mirror:
        body request_id == response request-id header, status 529."""
        from sglang.srt.utils.anthropic_http import (
            AnthropicOverloadedStatusMiddleware,
            AnthropicRequestIdMiddleware,
        )

        scope = _fake_request("/v1/messages").scope
        sent = []

        async def handler_app(s, r, send):
            # Run the real handler against the SHARED scope so it sees the
            # request-id key stamped by the outer middleware.
            resp = await _http_exception_handler()(
                Request(s), HTTPException(status_code=503, detail="engine busy")
            )
            await resp(s, r, send)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        chain = AnthropicRequestIdMiddleware(
            AnthropicOverloadedStatusMiddleware(handler_app)
        )
        asyncio.run(chain(scope, receive, send))
        self.assertEqual(sent[0]["status"], 529)
        body = json.loads(sent[1]["body"])
        self.assertEqual(body["error"]["type"], "overloaded_error")
        self.assertEqual(body["error"]["message"], "Overloaded")
        header_rid = next(
            value.decode()
            for name, value in sent[0]["headers"]
            if name.lower() == b"request-id"
        )
        self.assertEqual(body["request_id"], header_rid)

    def test_503_count_tokens_body_shape(self):
        resp = self._handle_503("/v1/messages/count_tokens")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(json.loads(resp.body)["error"]["type"], "overloaded_error")

    def test_openai_dialect_keeps_503(self):
        resp = asyncio.run(
            _http_exception_handler()(
                _fake_request("/v1/chat/completions"),
                HTTPException(status_code=503, detail="engine busy"),
            )
        )
        self.assertEqual(resp.status_code, 503)

    def test_other_statuses_untouched(self):
        resp = asyncio.run(
            _http_exception_handler()(
                _fake_request("/v1/messages"),
                HTTPException(status_code=500, detail="boom"),
            )
        )
        self.assertEqual(resp.status_code, 500)
        body = json.loads(resp.body)
        self.assertEqual(body["error"]["type"], "api_error")
        self.assertEqual(body["error"]["message"], "Internal server error")

    def test_handler_body_request_id_from_scope_key(self):
        """Round-2 S2 item: handler envelopes embed the middleware-stamped scope
        request id (spec §5.1 mirror) — e.g. the 429-overload path."""
        req = _fake_request("/v1/messages")
        req.scope[ANTHROPIC_REQUEST_ID_SCOPE_KEY] = "req_from_scope"
        resp = asyncio.run(
            _http_exception_handler()(
                req, HTTPException(status_code=429, detail="slow down")
            )
        )
        body = json.loads(resp.body)
        self.assertEqual(resp.status_code, 429)
        self.assertEqual(body["error"]["type"], "rate_limit_error")
        self.assertEqual(body["request_id"], "req_from_scope")


class TestValidationHandlerRequestId(unittest.TestCase):
    """Round-2 S2 item: the RequestValidationError (pydantic 400) envelope also
    mirrors the middleware-stamped request id in its body."""

    def _validation_handler(self):
        from fastapi.exceptions import RequestValidationError

        return app.exception_handlers[RequestValidationError]

    def test_validation_400_body_request_id_from_scope(self):
        from fastapi.exceptions import RequestValidationError

        req = _fake_request("/v1/messages")
        req.scope[ANTHROPIC_REQUEST_ID_SCOPE_KEY] = "req_val_scope"
        exc = RequestValidationError(
            errors=[{"loc": ["body", "model"], "msg": "Field required", "type": "missing"}]
        )
        resp = asyncio.run(self._validation_handler()(req, exc))
        body = json.loads(resp.body)
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertEqual(body["request_id"], "req_val_scope")

    def test_handler_omits_request_id_when_scope_key_absent(self):
        """No middleware (or non-stamped scope) ⇒ body has no request_id."""
        resp = asyncio.run(
            _http_exception_handler()(
                _fake_request("/v1/messages"), HTTPException(status_code=404, detail="x")
            )
        )
        self.assertNotIn("request_id", json.loads(resp.body))


class TestG02RequestIdEndToEnd(unittest.TestCase):
    """G-02 on the real app: the request-id middleware stamps even error
    envelopes produced by FastAPI exception handlers (middleware wraps them)."""

    def _client(self):
        from fastapi.testclient import TestClient

        return TestClient(app)

    def test_request_id_body_matches_header_on_validation_error(self):
        resp = self._client().post(
            "/v1/messages",
            content=b"not-json",
            headers={"content-type": "application/json"},
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertRegex(resp.headers["request-id"], r"^req_[0-9a-f]{32}$")
        # Spec §5.1: body request_id mirrors the request-id header.
        self.assertEqual(body["request_id"], resp.headers["request-id"])

    def test_request_id_body_matches_header_on_count_tokens_errors(self):
        resp = self._client().post(
            "/v1/messages/count_tokens",
            content=b"not-json",
            headers={"content-type": "application/json"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertRegex(resp.headers["request-id"], r"^req_[0-9a-f]{32}$")
        self.assertEqual(resp.json()["request_id"], resp.headers["request-id"])

    def test_no_request_id_on_openai_paths(self):
        resp = self._client().post(
            "/v1/chat/completions",
            content=b"not-json",
            headers={"content-type": "application/json"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertNotIn("request-id", resp.headers)
        self.assertNotIn("request_id", resp.text)


class TestG03CapsEndToEnd(unittest.TestCase):
    """G-03 on the real app: 32 MB body cap → 413 request_too_large envelope."""

    def _client(self):
        from fastapi.testclient import TestClient

        return TestClient(app)

    def test_oversize_body_413_envelope(self):
        # Declared Content-Length over 32 MiB (httpx honors the explicit header;
        # the middleware short-circuits before any body parsing).
        resp = self._client().post(
            "/v1/messages",
            content=b"{}",
            headers={
                "content-type": "application/json",
                "content-length": str(32 * 1024 * 1024 + 1),
            },
        )
        self.assertEqual(resp.status_code, 413)
        body = resp.json()
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "request_too_large")
        self.assertRegex(body["request_id"], r"^req_[0-9a-f]{32}$")
        self.assertEqual(resp.headers["request-id"], body["request_id"])

    def test_oversize_body_413_on_count_tokens(self):
        resp = self._client().post(
            "/v1/messages/count_tokens",
            content=b"{}",
            headers={
                "content-type": "application/json",
                "content-length": str(32 * 1024 * 1024 + 1),
            },
        )
        self.assertEqual(resp.status_code, 413)
        self.assertEqual(resp.json()["error"]["type"], "request_too_large")

    # NB: G-03's 100k-message cap (spec §2.1) is enforced by a protocol-level
    # pydantic validator on the request models (anthropic/protocol.py); a
    # route-layer duplicate would be unreachable.


class TestG27ModelsNegotiation(unittest.TestCase):
    """G-27 (spec §12.2): GET /v1/models{,/{id}} answer the Anthropic shape only
    when the request carries an `anthropic-version` header."""

    def setUp(self):
        from types import SimpleNamespace

        import sglang.srt.entrypoints.http_server as hs

        self._hs = hs
        self._prev_state = hs._global_state
        tokenizer_manager = SimpleNamespace(
            served_model_name="test-model",
            model_config=SimpleNamespace(context_len=4096),
            server_args=SimpleNamespace(enable_lora=False),
        )
        hs.set_global_state(
            hs._GlobalState(
                tokenizer_manager=tokenizer_manager,
                template_manager=None,
                scheduler_info=None,
            )
        )

    def tearDown(self):
        self._hs.set_global_state(self._prev_state)

    def _client(self):
        from fastapi.testclient import TestClient

        return TestClient(app)

    def test_openai_shape_byte_identical_without_header(self):
        resp = self._client().get("/v1/models")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["object"], "list")
        self.assertEqual(len(body["data"]), 1)
        card = body["data"][0]
        # OpenAI dialect keys — and none of the Anthropic ones.
        self.assertEqual(card["object"], "model")
        self.assertEqual(card["id"], "test-model")
        self.assertEqual(card["owned_by"], "sglang")
        self.assertEqual(card["max_model_len"], 4096)
        self.assertNotIn("display_name", card)
        self.assertNotIn("created_at", card)
        self.assertNotIn("first_id", body)

    def test_anthropic_shape_with_version_header(self):
        resp = self._client().get(
            "/v1/models", headers={"anthropic-version": "2023-06-01"}
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(set(body.keys()), {"data", "first_id", "last_id", "has_more"})
        self.assertIs(body["has_more"], False)
        self.assertEqual(body["first_id"], "test-model")
        self.assertEqual(body["last_id"], "test-model")
        entry = body["data"][0]
        self.assertEqual(entry["type"], "model")
        self.assertEqual(entry["id"], "test-model")
        self.assertEqual(entry["display_name"], "test-model")
        self.assertTrue(entry["created_at"].endswith("Z"))
        self.assertEqual(entry["max_input_tokens"], 4096)

    def test_retrieve_model_anthropic_shape(self):
        resp = self._client().get(
            "/v1/models/test-model", headers={"anthropic-version": "2023-06-01"}
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["type"], "model")
        self.assertEqual(body["id"], "test-model")
        self.assertEqual(body["max_input_tokens"], 4096)

    def test_retrieve_model_404_anthropic_envelope(self):
        resp = self._client().get(
            "/v1/models/no-such-model", headers={"anthropic-version": "2023-06-01"}
        )
        self.assertEqual(resp.status_code, 404)
        body = resp.json()
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "not_found_error")
        self.assertIn("no-such-model", body["error"]["message"])

    def test_retrieve_model_404_openai_shape_untouched(self):
        resp = self._client().get("/v1/models/no-such-model")
        self.assertEqual(resp.status_code, 404)
        body = resp.json()
        self.assertEqual(body["error"]["code"], "model_not_found")
        self.assertEqual(body["error"]["type"], "invalid_request_error")

    def test_retrieve_model_openai_shape_untouched(self):
        resp = self._client().get("/v1/models/test-model")
        self.assertEqual(resp.status_code, 200)
        card = resp.json()
        self.assertEqual(card["object"], "model")
        self.assertEqual(card["id"], "test-model")
        self.assertNotIn("display_name", card)


if __name__ == "__main__":
    unittest.main()
