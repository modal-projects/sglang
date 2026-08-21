"""Shared HTTP-contract helpers for sglang's Anthropic-compatible API surface.

Route/middleware-layer pieces of docs_new/anthropic_gap_audit.md:
- G-01: Anthropic error envelope for 401/403 auth denials on /v1/messages*
        (spec §5.1/§5.2) — consumed by ``sglang.srt.utils.auth``.
- G-02: ``request-id: req_<uuid4hex>`` header on every /v1/messages* response
        (spec §1.3), including streaming and error responses.
- G-03: 32 MB request-body cap → 413 ``request_too_large`` (spec §1.4).
- G-24: HTTP 503 → 529 ``overloaded_error`` translation on /v1/messages*
        (spec §5.2 — 529 is Anthropic-specific, not an IANA code).
- G-27: Anthropic-shaped /v1/models entries (spec §12.2) used by http_server's
        content-negotiated models endpoints.

Kept dependency-light (stdlib only — no starlette, no torch) so both
``sglang.srt.utils.auth`` and ``sglang.srt.entrypoints.http_server`` can import
it without cycles, and unit tests can exercise every piece directly. The
middleware classes are deliberately pure-ASGI so they also wrap streaming (SSE)
responses and responses synthesized by exception handlers.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    MutableMapping,
    Optional,
    Protocol,
    TypedDict,
)

# ASGI types spelled out locally (keeps this module import-free of starlette).
Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]
HeaderList = List[Any]  # ASGI raw headers: list of (bytes, bytes) pairs

# --- Path surface (spec §1.1) -------------------------------------------------

ANTHROPIC_MESSAGES_PATH_PREFIX = "/v1/messages"


def is_anthropic_messages_path(path: str) -> bool:
    """True for the Anthropic Messages API surface (/v1/messages + /v1/messages/count_tokens)."""
    return path.startswith(ANTHROPIC_MESSAGES_PATH_PREFIX)


# --- request-id (spec §1.3/§5.1; audit G-01/G-02) ---------------------------

# ASGI scope key under which AnthropicRequestIdMiddleware publishes the request
# id so inner layers (serving.py error bodies, our own 413/529 responses) can
# reuse the SAME id. Spec §5.1: body "request_id" mirrors the "request-id" header.
ANTHROPIC_REQUEST_ID_SCOPE_KEY = "anthropic.request_id"


def new_anthropic_request_id() -> str:
    """``req_...`` id (spec §1.7); uuid4hex keeps it URL/JSON-safe."""
    return f"req_{uuid.uuid4().hex}"


def anthropic_error_body(
    *, error_type: str, message: str, request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Anthropic error envelope body (spec §5.1)."""
    body: Dict[str, Any] = {
        "type": "error",
        "error": {"type": error_type, "message": message},
    }
    if request_id is not None:
        body["request_id"] = request_id
    return body


def encode_json_body(body: Dict[str, Any]) -> bytes:
    """Serialize an envelope for hand-rolled ASGI responses."""
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


# --- Canonical status → error-spec mapping (spec §5.2) ------------------------

# HTTP status → Anthropic error.type; unlisted codes fall back to api_error.
# G-04 (spec §5.2): gateway timeouts surface as Anthropic's "timeout_error"
# (mirrors serving.py's ERROR_TYPE_MAP), not the generic api_error bucket.
_ANTHROPIC_STATUS_TO_ERROR_TYPE: Dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "overloaded_error",
    504: "timeout_error",
}


class AnthropicErrorSpec(TypedDict):
    """Canonical (type, message) pair for a /v1/messages* error response."""

    error_type: str
    message: str


def anthropic_error_spec(
    *, status_code: int, detail: Optional[str] = None
) -> AnthropicErrorSpec:
    """Canonical status → {"error_type","message"} for /v1/messages* (spec §5.2).

    Single owner of three rules (the /v1/messages branch of http_server's
    HTTPException handler and AnthropicOverloadedStatusMiddleware both build
    bodies from this — no parallel maps):
    1. status → error.type mapping (incl. 503→overloaded_error / G-24 and
       504→timeout_error / G-04), falling back to api_error;
    2. 5xx scrub: NEVER echo upstream ``detail`` (may contain stack/PII) — and
       503 in particular gets Anthropic's canonical message "Overloaded";
    3. 4xx/known statuses echo the (already client-safe) detail text.

    The *wire* status is the caller's concern: 503→529 translation is owned by
    AnthropicOverloadedStatusMiddleware alone.
    """
    error_type = _ANTHROPIC_STATUS_TO_ERROR_TYPE.get(status_code, "api_error")
    if status_code >= 500:
        message = "Overloaded" if status_code == 503 else "Internal server error"
    else:
        message = detail or "Request failed"
    return {"error_type": error_type, "message": message}


# --- G-02: request-id response header (spec §1.3) ----------------------------


class AnthropicRequestIdMiddleware:
    """Stamp ``request-id: req_<uuid4hex>`` on every /v1/messages* response.

    Spec §1.3 requires the header on *every* response — normal JSON, SSE streams
    and error envelopes alike. Implemented as pure ASGI (rather than per-route)
    so it also covers responses synthesized by FastAPI exception handlers and by
    the G-03/G-24 middlewares below; ``http.response.start`` interception is
    header-only, so streams are never buffered.

    The id is published in the ASGI scope before calling the inner app, so
    serving.py can embed the SAME value into AnthropicErrorResponse.request_id
    bodies (spec §5.1: ``request_id`` mirrors the ``request-id`` header).

    Non-/v1/messages* paths pass through untouched (byte-identical responses).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not is_anthropic_messages_path(
            scope.get("path", "")
        ):
            await self.app(scope, receive, send)
            return

        request_id = new_anthropic_request_id()
        scope[ANTHROPIC_REQUEST_ID_SCOPE_KEY] = request_id
        encoded = request_id.encode("latin-1")

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers: HeaderList = message.setdefault("headers", [])
                if not any(name.lower() == b"request-id" for name, _ in headers):
                    headers.append((b"request-id", encoded))
            await send(message)

        await self.app(scope, receive, send_with_request_id)


class _MiddlewareStack(Protocol):
    """Structural type for anything with Starlette's add_middleware API."""

    def add_middleware(
        self, middleware_class: type, *args: Any, **kwargs: Any
    ) -> None: ...


def add_anthropic_http_contract_middlewares(app: _MiddlewareStack) -> None:
    """Register the /v1/messages* HTTP-contract middlewares (audit G-02/G-03/G-24).

    Ordering: Starlette runs the LAST-added user middleware first (outermost).
    The API-key middleware is only added later, at server launch, so it stays
    outermost overall and stamps its own request id on 401/403 denials (see
    utils/auth.py). Among these, AnthropicRequestIdMiddleware is outermost so
    its header also lands on 413/529 responses produced by the inner two.
    """
    app.add_middleware(AnthropicOverloadedStatusMiddleware)
    app.add_middleware(AnthropicBodySizeLimitMiddleware)
    app.add_middleware(AnthropicRequestIdMiddleware)


# --- G-24: 503 → 529 overloaded (spec §5.2) ----------------------------------


class AnthropicOverloadedStatusMiddleware:
    """Single owner of HTTP 503 → 529 translation on /v1/messages* (G-24, spec §5.2).

    529 is Anthropic-specific (not an IANA code) and Anthropic SDKs key their
    overload retries on it. Because FastAPI's ExceptionMiddleware lives *inside*
    the user middleware stack, this middleware sees BOTH HTTPException-handler
    envelopes (http_server.py builds the body from anthropic_error_spec) and raw
    503 Responses from engine overload paths — one translation point covers all.

    Behavior on a 503 start message: status → 529, and when the body is a JSON
    error envelope it is normalized to the canonical ``overloaded_error`` /
    "Overloaded" (retry-contract stable, never an upstream string). 503 bodies
    are tiny terminal responses, so the buffering done to rewrite them is
    bounded and never touches streaming/SSE traffic. Non-503 responses and
    non-/v1/messages* paths pass through byte-identical (plain 503 elsewhere).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not is_anthropic_messages_path(
            scope.get("path", "")
        ):
            await self.app(scope, receive, send)
            return

        start_message: Optional[Message] = None
        body_chunks: List[bytes] = []

        async def send_buffered(message: Message) -> None:
            nonlocal start_message
            if message["type"] == "http.response.start":
                if message.get("status") == 503:
                    # Hold the (status-rewritten) start until the body is known.
                    start_message = dict(message)
                    start_message["status"] = 529
                    return
                await send(message)
                return
            if message["type"] == "http.response.body" and start_message is not None:
                body_chunks.append(message.get("body", b"") or b"")
                if not message.get("more_body", False):
                    await self._flush(scope, start_message, body_chunks, send)
                    start_message = None  # terminal
                return
            # start not held (non-503) or trailers/other message types
            await send(message)

        await self.app(scope, receive, send_buffered)

    async def _flush(
        self,
        scope: Scope,
        start_message: Message,
        body_chunks: List[bytes],
        send: Send,
    ) -> None:
        body = b"".join(body_chunks)
        rewritten = self._overloaded_body(scope, start_message, body)
        headers = list(start_message.get("headers", []))
        headers = [
            (name, value)
            for name, value in headers
            if name.lower() != b"content-length"
        ]
        headers.append((b"content-length", str(len(rewritten)).encode("ascii")))
        start_message["headers"] = headers
        await send(start_message)
        await send({"type": "http.response.body", "body": rewritten})

    def _overloaded_body(
        self, scope: Scope, start_message: Message, body: bytes
    ) -> bytes:
        """Normalize JSON error envelopes to the canonical overload shape.

        Non-JSON or non-envelope bodies (not produced by sglang's own handlers,
        but possible from proxies) are left byte-identical — only the status moves.
        """
        parsed: Any = None
        try:
            parsed = json.loads(body) if body else None
        except ValueError:
            return body
        if not (isinstance(parsed, dict) and isinstance(parsed.get("error"), dict)):
            return body
        spec = anthropic_error_spec(status_code=503)
        request_id = scope.get(ANTHROPIC_REQUEST_ID_SCOPE_KEY)
        if request_id is None:
            request_id = new_anthropic_request_id()
        return encode_json_body(
            anthropic_error_body(
                error_type=spec["error_type"],
                message=spec["message"],
                request_id=request_id,
            )
        )


# --- G-03: 32 MB body cap (spec §1.4) ----------------------------------------

ANTHROPIC_MAX_BODY_BYTES = 32 * 1024 * 1024  # spec §1.4: Messages + count_tokens

# NB: spec §2.1's 100k-message cap is enforced by a protocol-level pydantic
# validator on AnthropicMessagesRequest — NOT duplicated here (a route-layer
# dependency would be dead code: pydantic body parsing fires before it runs).


class AnthropicBodySizeLimitMiddleware:
    """Reject /v1/messages* POST bodies > 32 MB with 413 ``request_too_large``.

    Spec §1.4: the Messages API limits request bodies to 32 MB (stat
    Content-Length fast path; chunked uploads are counted and aborted as soon
    as the running total crosses the cap). The envelope is Anthropic-shaped
    (spec §5.1/§5.2) and reuses the request id published by the outer
    AnthropicRequestIdMiddleware so body ``request_id`` == header.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int = ANTHROPIC_MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def _send_too_large(self, scope: Scope, send: Send) -> None:
        request_id = scope.get(ANTHROPIC_REQUEST_ID_SCOPE_KEY)
        if request_id is None:
            # Standalone use (no outer request-id middleware): still emit the header.
            request_id = new_anthropic_request_id()
        payload = encode_json_body(
            anthropic_error_body(
                error_type="request_too_large",
                message=(
                    f"Request exceeds the maximum size of {self.max_bytes} bytes"
                ),
                request_id=request_id,
            )
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode("ascii")),
                    (b"request-id", request_id.encode("latin-1")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or not is_anthropic_messages_path(scope.get("path", ""))
        ):
            await self.app(scope, receive, send)
            return

        declared: Optional[int] = None
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = None  # garbage header: let the route layer answer
                break

        if declared is not None:
            if declared > self.max_bytes:
                await self._send_too_large(scope, send)
                return
            await self.app(scope, receive, send)
            return

        # No Content-Length (chunked transfer): buffer body frames while
        # counting; replay them to the inner app only if the total stays legal.
        buffered: List[Message] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                buffered.append(message)
                break
            total += len(message.get("body", b"") or b"")
            if total > self.max_bytes:
                await self._send_too_large(scope, send)
                return
            buffered.append(message)
            if not message.get("more_body", False):
                break

        replay = iter(buffered)

        async def replay_receive() -> Message:
            try:
                return next(replay)
            except StopIteration:
                return await receive()

        await self.app(scope, replay_receive, send)


# --- G-27: Anthropic-shaped /v1/models (spec §12.2) --------------------------


class AnthropicModelEntry(TypedDict, total=False):
    """One Anthropic Model object (spec §12.2)."""

    type: str
    id: str
    display_name: str
    created_at: str
    max_input_tokens: int


class AnthropicModelListPayload(TypedDict):
    """Anthropic Models list envelope (spec §12.2)."""

    data: List[AnthropicModelEntry]
    first_id: Optional[str]
    last_id: Optional[str]
    has_more: bool


def anthropic_model_entry(
    *, model_id: str, created: int, max_model_len: Optional[int]
) -> AnthropicModelEntry:
    """Build one Anthropic Model object (spec §12.2; audit G-27).

    ``display_name`` mirrors ``id`` (we have no friendlier label); ``created_at``
    is the ISO-8601 form of the OpenAI ModelCard epoch; ``max_input_tokens`` is
    only emitted when the backend knows the context length (LoRA adapter cards
    report None). ``max_tokens`` is deliberately omitted rather than fabricated.
    """
    entry: AnthropicModelEntry = {
        "type": "model",
        "id": model_id,
        "display_name": model_id,
        "created_at": (
            datetime.fromtimestamp(created, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        ),
    }
    if max_model_len is not None:
        entry["max_input_tokens"] = max_model_len
    return entry


def anthropic_model_list(
    entries: List[AnthropicModelEntry],
) -> AnthropicModelListPayload:
    """Anthropic Models list envelope (spec §12.2): sglang serves a single page."""
    return {
        "data": entries,
        "first_id": entries[0]["id"] if entries else None,
        "last_id": entries[-1]["id"] if entries else None,
        "has_more": False,
    }
