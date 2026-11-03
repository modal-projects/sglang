"""Pure-ASGI middleware that zstd-compresses whole response bodies for clients that accept zstd."""

import asyncio
import logging

import zstandard
from starlette.datastructures import Headers, MutableHeaders

logger = logging.getLogger(__name__)

# Smaller bodies gain too little to pay for the trip to the executor.
_MIN_NBYTES = 1 << 16


def _accepts_zstd(scope) -> bool:
    for coding in Headers(scope=scope).get("accept-encoding", "").split(","):
        name, _, params = coding.partition(";")
        if name.strip().lower() != "zstd":
            continue
        weight = params.strip().lower().removeprefix("q=")
        try:
            return not weight or float(weight) > 0
        except ValueError:
            return False
    return False


def _zstd_compress(body: bytes) -> bytes:
    return zstandard.ZstdCompressor().compress(body)


class ResponseCompressionMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not _accepts_zstd(scope):
            return await self.app(scope, receive, send)

        start = None
        decided = False

        async def wrapped_send(message):
            nonlocal start, decided
            if decided:
                return await send(message)
            if message["type"] == "http.response.start":
                start = message
                return
            decided = True
            body = message.get("body", b"")
            headers = MutableHeaders(raw=list(start["headers"]))
            if (
                message["type"] != "http.response.body"
                or message.get("more_body", False)
                or len(body) < _MIN_NBYTES
                or "content-encoding" in headers
            ):
                await send(start)
                return await send(message)
            try:
                body = await asyncio.get_running_loop().run_in_executor(
                    None, _zstd_compress, body
                )
            except Exception as e:
                logger.warning("response body compress failed: %s", e)
                await send(start)
                return await send(message)
            headers["content-encoding"] = "zstd"
            headers["content-length"] = str(len(body))
            headers.add_vary_header("Accept-Encoding")
            await send({**start, "headers": headers.raw})
            await send({"type": "http.response.body", "body": body, "more_body": False})

        await self.app(scope, receive, wrapped_send)
