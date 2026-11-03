import asyncio
import unittest

import zstandard

from sglang.srt.entrypoints.http_response_compression import (
    _MIN_NBYTES,
    ResponseCompressionMiddleware,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")

LARGE = b'{"meta_info":{"routed_experts":"' + b"QUFBQUFB" * _MIN_NBYTES + b'"}}'
SMALL = b'{"text":"hello world"}'


def _start(headers=()):
    return {
        "type": "http.response.start",
        "status": 200,
        "headers": [(b"content-type", b"application/json"), *headers],
    }


def _body(body, more_body=False):
    return {"type": "http.response.body", "body": body, "more_body": more_body}


def _drive(accept_encoding, messages, scope_type="http"):
    """Run the middleware over an app that sends `messages`; returns what reached the client."""
    headers = [] if accept_encoding is None else [(b"accept-encoding", accept_encoding)]
    scope = {"type": scope_type, "headers": headers}
    sent = []

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    async def app(inner_scope, inner_receive, inner_send):
        for message in messages:
            await inner_send(message)

    asyncio.run(ResponseCompressionMiddleware(app)(scope, receive, send))
    return sent


def _header(message, name):
    values = [v for k, v in message["headers"] if k == name]
    return values[0] if values else None


class TestResponseCompressionMiddleware(CustomTestCase):
    def test_passthrough_without_accept_encoding(self):
        messages = [
            _start([(b"content-length", str(len(LARGE)).encode())]),
            _body(LARGE),
        ]
        self.assertEqual(_drive(None, messages), messages)

    def test_compresses_large_body_when_zstd_accepted(self):
        sent = _drive(
            b"gzip, zstd",
            [_start([(b"content-length", str(len(LARGE)).encode())]), _body(LARGE)],
        )
        self.assertEqual(len(sent), 2)
        start, body = sent
        self.assertEqual(_header(start, b"content-encoding"), b"zstd")
        self.assertEqual(
            _header(start, b"content-length"), str(len(body["body"])).encode()
        )
        self.assertIn(b"Accept-Encoding", _header(start, b"vary"))
        self.assertEqual(_header(start, b"content-type"), b"application/json")
        self.assertLess(len(body["body"]), len(LARGE))
        self.assertEqual(
            zstandard.ZstdDecompressor().decompress(
                body["body"], max_output_size=len(LARGE)
            ),
            LARGE,
        )

    def test_weights_decide_acceptance(self):
        for accept, compressed in (
            (b"zstd;q=0", False),
            (b"zstd; q=0.0, gzip", False),
            (b"zstd;q=0.5", True),
            (b"ZSTD", True),
            (b"gzip, br", False),
            (b"zstd;q=bogus", False),
        ):
            with self.subTest(accept=accept):
                start, _ = _drive(accept, [_start(), _body(LARGE)])
                self.assertEqual(
                    _header(start, b"content-encoding"), b"zstd" if compressed else None
                )

    def test_small_body_passes_through(self):
        messages = [_start(), _body(SMALL)]
        self.assertEqual(_drive(b"zstd", messages), messages)

    def test_streamed_body_passes_through(self):
        messages = [_start(), _body(LARGE, more_body=True), _body(LARGE)]
        self.assertEqual(_drive(b"zstd", messages), messages)

    def test_encoded_body_passes_through(self):
        messages = [_start([(b"content-encoding", b"gzip")]), _body(LARGE)]
        self.assertEqual(_drive(b"zstd", messages), messages)

    def test_non_http_scope_passes_through(self):
        messages = [{"type": "websocket.send", "bytes": LARGE}]
        self.assertEqual(_drive(b"zstd", messages, scope_type="websocket"), messages)


if __name__ == "__main__":
    unittest.main()
