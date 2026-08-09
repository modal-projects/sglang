"""Pure-ASGI middleware that derives the sglext id opt-in from `modal-log-exclude`.

Gated on `SGLANG_RETURN_IDS_FROM_LOG_EXCLUDE`. Callers tag requests that must
stay out of logs with `modal-log-exclude`; the remaining requests are loggable,
so this rewrites their headers to request the sglext input/output token ids that
the logging pipeline consumes.

The middleware is authoritative: it drops any caller-supplied
`x-sglext-return-{input,output}-ids` first, so a log-excluded request cannot opt
itself back into returning ids. Requests without the header are treated as
excluded.
"""

LOG_EXCLUDE_HEADER = b"modal-log-exclude"
ID_HEADERS = (b"x-sglext-return-input-ids", b"x-sglext-return-output-ids")

# Only chat completions honors the sglext id headers today.
ID_PATHS = ("/v1/chat/completions",)

_FALSEY = frozenset({b"0", b"false", b"no", b"off"})


def is_log_excluded(headers) -> bool:
    for key, value in headers:
        if key.lower() == LOG_EXCLUDE_HEADER:
            return value.strip().lower() not in _FALSEY
    return True


class LogExcludeIdsMiddleware:
    """Request sglext input/output ids for requests that may be logged."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith(ID_PATHS):
            return await self.app(scope, receive, send)

        headers = [(k, v) for k, v in scope["headers"] if k.lower() not in ID_HEADERS]
        if not is_log_excluded(scope["headers"]):
            headers.extend((header, b"1") for header in ID_HEADERS)

        scope = dict(scope)
        scope["headers"] = headers
        await self.app(scope, receive, send)
