"""Unit tests for srt/entrypoints/early_reject.py"""

import asyncio
import unittest
from types import SimpleNamespace

import orjson

from sglang.srt.entrypoints.early_reject import (
    APIEarlyRejectGate,
    APIEarlyRejectMiddleware,
    configure_api_early_reject,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _scope(app, path="/v1/chat/completions", method="POST"):
    return {"type": "http", "method": method, "path": path, "app": app}


async def _empty_receive():
    return {"type": "http.request", "body": b"", "more_body": False}


class _BlockingApp:
    """Sends a 200 once ``release`` is set, so a permit stays held until then."""

    def __init__(self):
        self.release = asyncio.Event()
        self.entered = asyncio.Event()

    async def __call__(self, scope, receive, send):
        self.entered.set()
        await self.release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


async def _call(middleware, scope):
    sent = []

    async def send(message):
        sent.append(message)

    await middleware(scope, _empty_receive, send)
    return sent


class TestAPIEarlyRejectGate(CustomTestCase):
    def test_permits_are_bounded_and_released(self):
        async def run():
            gate = APIEarlyRejectGate(2)
            self.assertTrue(await gate.try_acquire())
            self.assertTrue(await gate.try_acquire())
            self.assertFalse(await gate.try_acquire())
            await gate.release()
            self.assertEqual(gate.active_requests, 1)
            self.assertTrue(await gate.try_acquire())

        asyncio.run(run())

    def test_invalid_usage_raises(self):
        with self.assertRaises(ValueError):
            APIEarlyRejectGate(0)
        with self.assertRaises(RuntimeError):
            asyncio.run(APIEarlyRejectGate(1).release())


class TestAPIEarlyRejectMiddleware(CustomTestCase):
    def _run_with_held_permit(self, path):
        """Return the response of a second request while a first one is in flight."""

        async def run():
            fastapi_app = SimpleNamespace(state=SimpleNamespace())
            configure_api_early_reject(fastapi_app, 1)
            inner = _BlockingApp()
            middleware = APIEarlyRejectMiddleware(inner)
            first = asyncio.ensure_future(_call(middleware, _scope(fastapi_app)))
            await inner.entered.wait()
            rejected = await _call(middleware, _scope(fastapi_app, path=path))
            inner.release.set()
            accepted = await first
            after = asyncio.ensure_future(_call(middleware, _scope(fastapi_app)))
            await asyncio.sleep(0)
            return accepted, rejected, await after

        return asyncio.run(run())

    def test_over_capacity_request_gets_openai_503(self):
        accepted, rejected, after = self._run_with_held_permit("/v1/completions")
        self.assertEqual(accepted[0]["status"], 200)
        self.assertEqual(after[0]["status"], 200)
        self.assertEqual(rejected[0]["status"], 503)
        self.assertIn((b"retry-after", b"1"), rejected[0]["headers"])
        body = orjson.loads(rejected[1]["body"])
        self.assertEqual(body["error"]["code"], "service_unavailable")

    def test_over_capacity_messages_request_gets_anthropic_error(self):
        _, rejected, _ = self._run_with_held_permit("/v1/messages")
        self.assertEqual(rejected[0]["status"], 503)
        body = orjson.loads(rejected[1]["body"])
        self.assertEqual(body["error"]["type"], "overloaded_error")

    def test_unconfigured_or_non_generation_requests_pass_through(self):
        async def run():
            fastapi_app = SimpleNamespace(state=SimpleNamespace())
            configure_api_early_reject(fastapi_app, None)
            inner = _BlockingApp()
            inner.release.set()
            middleware = APIEarlyRejectMiddleware(inner)
            results = [await _call(middleware, _scope(fastapi_app))]
            configure_api_early_reject(fastapi_app, 1)
            await fastapi_app.state.api_early_reject_gate.try_acquire()
            for scope in (
                _scope(fastapi_app, path="/health"),
                _scope(fastapi_app, method="GET"),
            ):
                results.append(await _call(middleware, scope))
            return results

        for sent in asyncio.run(run()):
            self.assertEqual(sent[0]["status"], 200)


if __name__ == "__main__":
    unittest.main()
