#!/usr/bin/env python3
"""Self-contained MOCK Anthropic Messages API server.

No sglang dependency. FastAPI + uvicorn. Purpose: end-to-end probe target for
the REAL Claude Code CLI (`claude -p ...`) pointed at ANTHROPIC_BASE_URL.

Implements:
  POST /v1/messages                  non-streaming + SSE-streaming responses
  POST /v1/messages/count_tokens     trivial token count
  ANY  <other path>                  logged + 404  (so we can see every probe)

Behaviors (driven by request content):
  * last user msg contains a tool_result block -> final text echo (round trip)
  * last user text contains "add" or "tool"   -> tool_use block: add_numbers
  * otherwise                                 -> plain text reply

Every request (method, path, headers, raw body) is appended to
transcripts/requests.jsonl AND written to transcripts/NNN_<path>.json
(pretty) with the full response we returned (SSE text included).

Run:  .venv312/bin/python mock_anthropic_server.py --port 8077
"""

import argparse
import json
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

HERE = Path(__file__).resolve().parent
TRANSCRIPTS = HERE / "transcripts"
TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
JSONL = TRANSCRIPTS / "requests.jsonl"

app = FastAPI()
_seq = {"n": 0}

ADD_NUMBERS_SCHEMA = {
    "type": "object",
    "properties": {
        "a": {"type": "number", "description": "first addend"},
        "b": {"type": "number", "description": "second addend"},
    },
    "required": ["a", "b"],
}


def _log(direction: str, payload: dict):
    payload["_ts"] = time.time()
    payload["_dir"] = direction
    with JSONL.open("a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def _dump_pair(req: Request, headers: dict, body: dict, resp_payload: dict):
    _seq["n"] += 1
    name = f"{_seq['n']:03d}_{req.url.path.strip('/').replace('/', '_') or 'root'}.json"
    (TRANSCRIPTS / name).write_text(
        json.dumps(
            {
                "request": {
                    "method": req.method,
                    "path": req.url.path,
                    "query": str(req.url.query),
                    "headers": headers,
                    "body": body,
                },
                "response": resp_payload,
            },
            indent=2,
            default=str,
        )
    )


def _last_user_text(messages):
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content, None
        texts, tool_results = [], None
        for block in content or []:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    tool_results = block
        return " ".join(texts), tool_results
    return "", None


def _msg_id():
    return "msg_" + uuid.uuid4().hex[:24]


def _base_message(model, content, stop_reason, in_toks, out_toks):
    return {
        "id": _msg_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": in_toks, "output_tokens": out_toks},
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_gen(model, blocks, stop_reason, in_toks, out_toks):
    """blocks: list of ('text', str) or ('tool_use', {id,name,input}) tuples.

    Yields the canonical Anthropic SSE event sequence and RECORDS everything
    we emit so the transcript shows exactly what the client consumed.
    """
    emitted = []
    msg = _base_message(model, [], None, in_toks, 1)
    chunk = _sse("message_start", {"type": "message_start", "message": msg})
    emitted.append(("message_start", msg))
    yield chunk
    for index, (kind, payload) in enumerate(blocks):
        if kind == "text":
            start = {"type": "text", "text": ""}
            yield _sse(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": start},
            )
            emitted.append(("content_block_start", start))
            # word-ish chunks to exercise streaming reassembly
            words = payload.split(" ")
            for i, w in enumerate(words):
                piece = w + (" " if i < len(words) - 1 else "")
                delta = {"type": "text_delta", "text": piece}
                yield _sse(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": index, "delta": delta},
                )
                emitted.append(("content_block_delta", delta))
        elif kind == "thinking":
            start = {"type": "thinking", "thinking": ""}  # NO signature field: probe acceptance
            yield _sse(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": start},
            )
            emitted.append(("content_block_start", start))
            delta = {"type": "thinking_delta", "thinking": payload}
            yield _sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": index, "delta": delta},
            )
            emitted.append(("content_block_delta", delta))
        else:  # tool_use
            start = {"type": "tool_use", "id": payload["id"], "name": payload["name"], "input": {}}
            yield _sse(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": start},
            )
            emitted.append(("content_block_start", start))
            delta = {"type": "input_json_delta", "partial_json": json.dumps(payload["input"])}
            yield _sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": index, "delta": delta},
            )
            emitted.append(("content_block_delta", delta))
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": index})
        emitted.append(("content_block_stop", {"index": index}))
    mdelta = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": out_toks},
    }
    yield _sse("message_delta", mdelta)
    emitted.append(("message_delta", mdelta))
    yield _sse("message_stop", {"type": "message_stop"})
    emitted.append(("message_stop", {}))
    STREAM_LOG["last_events"] = emitted


STREAM_LOG = {"last_events": []}


def _plan(body: dict):
    """Decide (blocks, stop_reason) from the request body."""
    messages = body.get("messages", [])
    text, tool_result = _last_user_text(messages)
    if tool_result is not None:
        content = tool_result.get("content")
        if isinstance(content, list):
            rt = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            rt = str(content)
        reply = (
            f"MOCK: got tool_result for tool_use_id={tool_result.get('tool_use_id')}: {rt!r}. "
            "add_numbers(2, 3) = 5. Done."
        )
        return [("text", reply)], "end_turn"
    lowered = text.lower()
    if "thinktool" in lowered:
        return [
            ("thinking", "reasoning first..."),
            ("text", "MOCK: calling a tool after thinking."),
            (
                "tool_use",
                {
                    "id": "toolu_" + uuid.uuid4().hex[:20],
                    "name": "Bash",
                    "input": {"command": "echo MOCK_TOOL_WORKS_42", "description": "Echo a marker string"},
                },
            ),
        ], "tool_use"
    if "think please" in lowered:
        return [
            ("thinking", "let me reason about this carefully..."),
            ("text", "MOCK: done thinking; the answer is 42."),
        ], "end_turn"
    if "refuse" in lowered:
        return [("text", "MOCK: I must decline.")], "refusal"
    if "go long" in lowered:
        return [("text", "MOCK: truncated mid-sentence because the")], "max_tokens"
    if "add_numbers" in lowered or "add" in lowered or "tool" in lowered:
        # A client can only execute tools it declared. Claude Code ships
        # built-ins (Bash, Read, ...); use Bash with a harmless echo so the
        # client can really run it and POST the tool_result back. Generic
        # clients (curl tests) that declare add_numbers get the toy tool.
        client_tools = {t.get("name") for t in body.get("tools") or [] if isinstance(t, dict)}
        if "add_numbers" in client_tools and "Bash" not in client_tools:
            return [
                (
                    "tool_use",
                    {"id": "toolu_" + uuid.uuid4().hex[:20], "name": "add_numbers", "input": {"a": 2, "b": 3}},
                )
            ], "tool_use"
        return [
            ("text", "MOCK: calling a tool now."),
            (
                "tool_use",
                {
                    "id": "toolu_" + uuid.uuid4().hex[:20],
                    "name": "Bash",
                    "input": {"command": "echo MOCK_TOOL_WORKS_42", "description": "Echo a marker string"},
                },
            ),
        ], "tool_use"
    return [("text", f"Hello from mock-anthropic! You said: {text[:80]!r}")], "end_turn"


@app.post("/v1/messages")
async def messages(request: Request):
    raw = await request.body()
    headers = {k: v for k, v in request.headers.items()}
    try:
        body = json.loads(raw)
    except Exception:
        body = {"_unparsable": raw.decode(errors="replace")}
    _log("request", {"method": "POST", "path": "/v1/messages", "headers": headers, "body": body})

    model = body.get("model", "mock-model")
    stream = bool(body.get("stream"))
    last_text, _ = _last_user_text(body.get("messages", []))
    lowered = last_text.lower()

    # ---- error-injection experiments -------------------------------------
    if "overload503" in lowered:
        resp = {"type": "error", "error": {"type": "overloaded_error", "message": "mock is overloaded"}}
        _dump_pair(request, headers, body, {"json": resp, "status": 503})
        return JSONResponse(resp, status_code=503)
    if "hit stop word" in lowered and stream:
        def stopseq_gen():
            msg = _base_message(model, [], None, 42, 1)
            yield _sse("message_start", {"type": "message_start", "message": msg})
            yield _sse(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            )
            yield _sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "MOCK: line one.\n\nSTOP"}},
            )
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
            yield _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "stop_sequence", "stop_sequence": "\n\nSTOP"},
                    "usage": {"output_tokens": 9},
                },
            )
            yield _sse("message_stop", {"type": "message_stop"})

        _dump_pair(request, headers, body, {"note": "STOPSEQ: stop_reason=stop_sequence with literal stop_sequence"})
        return StreamingResponse(stopseq_gen(), media_type="text/event-stream")
    if "nousage" in lowered and stream:
        msg = _base_message(model, [], None, 0, 0)
        msg.pop("usage", None)  # message_start without usage

        def nousg_gen():
            yield _sse("message_start", {"type": "message_start", "message": msg})
            yield _sse(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            )
            yield _sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "MOCK: no usage anywhere"}},
            )
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
            # message_delta with NO usage key at all
            yield _sse(
                "message_delta",
                {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}},
            )
            yield _sse("message_stop", {"type": "message_stop"})

        _dump_pair(request, headers, body, {"note": "NOUSAGE: usage stripped from all events"})
        return StreamingResponse(nousg_gen(), media_type="text/event-stream")
    if "error500" in lowered:
        resp = {"type": "error", "error": {"type": "api_error", "message": "boom from mock"}}
        _dump_pair(request, headers, body, {"json": resp, "status": 500})
        return JSONResponse(resp, status_code=500)
    if "breakjson" in lowered and stream:
        # Client asked for SSE; we answer with a plain JSON message instead.
        resp = _base_message(model, [{"type": "text", "text": "MOCK: wrong content-type reply"}], "end_turn", 42, 17)
        _dump_pair(request, headers, body, {"json": resp, "note": "BREAKJSON: JSON instead of SSE"})
        return JSONResponse(resp)
    if "breaksse" in lowered and stream:
        def bad_gen():
            yield _sse("message_start", {"type": "message_start", "message": _base_message(model, [], None, 42, 1)})
            yield _sse(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            )
            yield _sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial"}},
            )
            # generator ends: no content_block_stop / message_delta / message_stop

        _dump_pair(request, headers, body, {"note": "BREAKSSE: truncated stream"})
        return StreamingResponse(bad_gen(), media_type="text/event-stream")
    # -----------------------------------------------------------------------

    blocks, stop_reason = _plan(body if isinstance(body, dict) else {})

    if not stream:
        content = []
        for kind, payload in blocks:
            if kind == "text":
                content.append({"type": "text", "text": payload})
            elif kind == "thinking":
                content.append({"type": "thinking", "thinking": payload})  # no signature
            else:
                content.append(
                    {"type": "tool_use", "id": payload["id"], "name": payload["name"], "input": payload["input"]}
                )
        resp = _base_message(model, content, stop_reason, in_toks=42, out_toks=17)
        _log("response", {"path": "/v1/messages", "json": resp})
        _dump_pair(request, headers, body, {"json": resp})
        return JSONResponse(resp)

    def gen():
        yield from _stream_gen(model, blocks, stop_reason, in_toks=42, out_toks=17)

    resp_headers = {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache",
        "connection": "keep-alive",
    }
    response = StreamingResponse(gen(), headers=resp_headers, media_type="text/event-stream")
    _dump_pair(request, headers, body, {"sse_blocks": blocks, "stop_reason": stop_reason})
    return response


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    raw = await request.body()
    headers = {k: v for k, v in request.headers.items()}
    try:
        body = json.loads(raw)
    except Exception:
        body = {"_unparsable": raw.decode(errors="replace")}
    _log("request", {"method": "POST", "path": "/v1/messages/count_tokens", "headers": headers, "body": body})
    resp = {"input_tokens": 123}
    _dump_pair(request, headers, body, {"json": resp})
    return JSONResponse(resp)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def catch_all(request: Request, path: str):
    raw = await request.body()
    headers = {k: v for k, v in request.headers.items()}
    _log(
        "request",
        {
            "method": request.method,
            "path": "/" + path,
            "query": str(request.url.query),
            "headers": headers,
            "body": raw.decode(errors="replace")[:4000],
        },
    )
    if path == "health" or path == "":
        return JSONResponse({"ok": True})
    return JSONResponse(
        {"type": "error", "error": {"type": "not_found_error", "message": f"mock has no route /{path}"}},
        status_code=404,
    )


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8077)
    args = parser.parse_args()
    JSONL.write_text("")  # fresh transcript per server start
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
