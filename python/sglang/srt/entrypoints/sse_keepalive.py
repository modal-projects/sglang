"""Keep idle SSE streams alive while a request waits for its first token.

SGLang sends response headers only with the first generated chunk, so a
streaming request that waits in the scheduler queue sends no bytes at all
until then. Modal's ingress silently drops HTTP connections that carry no
bytes for about 350 s, so at saturation such requests hang client-side
forever. If the application has not started its response within
``interval`` seconds, this middleware sends 200 + ``text/event-stream``
headers and an SSE comment line every ``interval`` seconds until it does.
Once an SSE response is streaming, the same comment is sent at event
boundaries whenever the application is silent for ``interval`` seconds
(``/v1/responses`` sends ``response.created`` immediately and then nothing
while queued). SSE clients ignore comment lines. Requests that fail before the first
keepalive keep their real HTTP status; a later non-200 response is delivered
as a ``data:`` error event. Keepalives stop once the application's
response body ends, and are only inserted between complete SSE events.
"""

import asyncio

import orjson
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sglang.srt.entrypoints.early_reject import GENERATION_API_PATHS
from sglang.srt.environ import envs

KEEPALIVE_COMMENT = b": keep-alive\n\n"
_SSE_HEADERS = [
    (b"content-type", b"text/event-stream; charset=utf-8"),
    (b"cache-control", b"no-cache"),
    (b"x-accel-buffering", b"no"),
]


def _is_stream_request(body: bytes) -> bool:
    try:
        payload = orjson.loads(body)
    except orjson.JSONDecodeError:
        return False
    return isinstance(payload, dict) and payload.get("stream") is True


class SSEKeepaliveMiddleware:
    def __init__(self, app: ASGIApp, interval: float | None = None):
        self.app = app
        if interval is None:
            interval = envs.SGLANG_K3_SSE_KEEPALIVE_S.get()
        self.interval = interval

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            self.interval <= 0
            or scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in GENERATION_API_PATHS
        ):
            await self.app(scope, receive, send)
            return

        chunks = []
        while True:
            message = await receive()
            if message["type"] != "http.request":
                await self.app(scope, _replay([message], receive), send)
                return
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        app_receive = _replay(
            [{"type": "http.request", "body": body, "more_body": False}], receive
        )
        if not _is_stream_request(body):
            await self.app(scope, app_receive, send)
            return

        lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        state = {
            "app_started": False,
            "keepalive_started": False,
            "error_status": None,
            "error_body": b"",
            "sse": False,
            "at_boundary": False,
            "done": False,
            "last_send": loop.time(),
        }

        async def app_send(message: Message) -> None:
            async with lock:
                if message["type"] == "http.response.start":
                    state["app_started"] = True
                    headers = dict(message.get("headers") or [])
                    state["sse"] = headers.get(b"content-type", b"").startswith(
                        b"text/event-stream"
                    )
                elif message["type"] == "http.response.body":
                    body = message.get("body", b"")
                    if body:
                        state["at_boundary"] = body.endswith(
                            b"\n\n"
                        ) and not body.endswith(b"[DONE]\n\n")
                    if not message.get("more_body", False):
                        state["done"] = True
                state["last_send"] = loop.time()
                if not state["keepalive_started"]:
                    await send(message)
                    return
                if message["type"] == "http.response.start":
                    if message["status"] != 200:
                        state["error_status"] = message["status"]
                    return
                if (
                    message["type"] != "http.response.body"
                    or state["error_status"] is None
                ):
                    await send(message)
                    return
                state["error_body"] += message.get("body", b"")
                if not message.get("more_body", False):
                    await send(
                        {
                            "type": "http.response.body",
                            "body": _error_event(state),
                            "more_body": False,
                        }
                    )

        task = asyncio.ensure_future(self.app(scope, app_receive, app_send))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=self.interval)
                if done:
                    break
                async with lock:
                    if state["done"]:
                        break
                    if not state["app_started"]:
                        if not state["keepalive_started"]:
                            state["keepalive_started"] = True
                            await send(
                                {
                                    "type": "http.response.start",
                                    "status": 200,
                                    "headers": _SSE_HEADERS,
                                }
                            )
                    elif not (
                        (state["sse"] or state["keepalive_started"])
                        and state["at_boundary"]
                        and state["error_status"] is None
                        and loop.time() - state["last_send"] >= self.interval
                    ):
                        continue
                    await send(
                        {
                            "type": "http.response.body",
                            "body": KEEPALIVE_COMMENT,
                            "more_body": True,
                        }
                    )
                    state["last_send"] = loop.time()
        except OSError:
            pass
        await task


def _error_event(state: dict) -> bytes:
    body = state["error_body"].strip()
    try:
        orjson.loads(body)
    except orjson.JSONDecodeError:
        body = orjson.dumps(
            {
                "error": {
                    "message": body.decode(errors="replace"),
                    "code": state["error_status"],
                }
            }
        )
    return b"data: " + body + b"\n\ndata: [DONE]\n\n"


def _replay(messages: list, receive: Receive) -> Receive:
    async def replay() -> Message:
        if messages:
            return messages.pop(0)
        return await receive()

    return replay
