# Anthropic Messages API — Implementation Specification

**Purpose:** authoritative wire-format spec for implementing Anthropic-compatible serving (e.g., in sglang). It covers the complete request/response/streaming/error surface of the Anthropic **Messages API**, plus an Anthropic↔OpenAI translation reference.

**Sources (all official, fetched August 2026):**

- Create a Message: <https://platform.claude.com/docs/en/api/messages/create> (formerly docs.anthropic.com/en/api/messages)
- Count tokens: <https://platform.claude.com/docs/en/api/messages/count_tokens>
- Models API: <https://platform.claude.com/docs/en/api/models/list>
- Message Batches: <https://platform.claude.com/docs/en/api/messages/batches/create>, <https://platform.claude.com/docs/en/build-with-claude/batch-processing>
- Errors: <https://platform.claude.com/docs/en/api/errors>
- API overview / auth / headers: <https://platform.claude.com/docs/en/api/overview>
- Versioning: <https://platform.claude.com/docs/en/api/versioning>
- Beta headers: <https://platform.claude.com/docs/en/api/beta-headers>
- Rate limits: <https://platform.claude.com/docs/en/api/rate-limits>
- Service tiers: <https://platform.claude.com/docs/en/api/service-tiers>
- Streaming: <https://platform.claude.com/docs/en/build-with-claude/streaming>
- Tool use: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview>, <https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools>, <https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls>, <https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use>, <https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use>, <https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools>, <https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool>, <https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming>
- Thinking: <https://platform.claude.com/docs/en/build-with-claude/thinking>, <https://platform.claude.com/docs/en/build-with-claude/extended-thinking>, <https://platform.claude.com/docs/en/build-with-claude/thinking-tool-workflows>
- Vision: <https://platform.claude.com/docs/en/build-with-claude/vision>
- PDF support: <https://platform.claude.com/docs/en/build-with-claude/pdf-support>
- Prompt caching: <https://platform.claude.com/docs/en/build-with-claude/prompt-caching>
- Context windows: <https://platform.claude.com/docs/en/build-with-claude/context-windows>
- Stop reasons: <https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons>
- Mid-conversation system messages: <https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages>
- Context editing (context management): <https://platform.claude.com/docs/en/build-with-claude/context-editing>
- Token counting guide: <https://platform.claude.com/docs/en/build-with-claude/token-counting>
- Effort: <https://platform.claude.com/docs/en/build-with-claude/effort>
- Structured outputs: <https://platform.claude.com/docs/en/build-with-claude/structured-outputs>
- Models overview: <https://platform.claude.com/docs/en/about-claude/models/overview>
- OpenAI SDK compatibility: <https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk>
- Python SDK type definitions: <https://github.com/anthropics/anthropic-sdk-python> (canonical model classes)

> **Time note.** This document reflects the API as of **API version `2023-06-01` + all GA features through 2026** (Claude 4.x/5.x era). The wire contract of `/v1/messages` has been stable since 2023; Anthropic's versioning policy (§1.6) is strictly additive. Items that are beta or model-gated are flagged as such.

---

## 0. Executive summary — the 12 things an implementer must not get wrong

1. **`max_tokens` is REQUIRED** on every `POST /v1/messages` call (unlike OpenAI chat completions, where it is optional). `0` is legal and means "cache pre-warm, don't generate".
2. There is **no `"system"` role inside `messages[]`** for the classic protocol — system prompts go in the top-level `system` string-or-block-array parameter. (Newest models additionally allow *mid-conversation* `{"role":"system"}` messages; see §2.2.4.) Anthropic's API **merges consecutive same-role turns** server-side.
3. Message `content` is either a **string** (shorthand for one text block) or an **array of typed content blocks**. Request and response use *different* block types (param vs returned variants).
4. Assistant responses are `content[]` arrays that **interleave `text`, `tool_use`, `thinking`, `redacted_thinking`, `server_tool_use`, `web_search_tool_result`, …** blocks. A response is never just a string, even when there's a single text block.
5. `stop_reason` is Anthropic-specific: `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`, `pause_turn`, `refusal`, `model_context_window_exceeded`. Map these carefully when bridging to OpenAI `finish_reason`.
6. Streaming is **named-event SSE** with a block-indexed protocol: `message_start` → per-block `content_block_start` → `content_block_delta` (…+`ping`) → `content_block_stop` → `message_delta` (stop_reason + *cumulative* usage) → `message_stop`. No `[DONE]`. Tool-call arguments stream as **partial JSON strings** (`input_json_delta.partial_json`) that the client concatenates and parses at block end.
7. Errors are always `{"type":"error","error":{"type":..., "message":...}, "request_id":...}` — including mid-stream (an SSE `error` event can arrive *after* a `200 OK`). HTTP code 529 (`overloaded_error`) is Anthropic-specific.
8. Every `tool_use` block has an `id` (`toolu_...`); the client answers with `tool_result` blocks (`tool_use_id` matching) as the **first** blocks of the **next `user` message** — no intervening messages allowed, all results together, text only after results. Violations are 400s.
9. The `usage` object is not two fields — it includes `cache_creation_input_tokens`, `cache_read_input_tokens`, `cache_creation.ephemeral_{5m,1h}_input_tokens`, `server_tool_use.{web_search_requests,web_fetch_requests}`, `output_tokens_details.thinking_tokens`, `service_tier`, `inference_geo`. **`input_tokens` only counts tokens *after the last cache breakpoint*** — total input = `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`. In streaming, usage in `message_delta` is **cumulative**.
10. Prompt caching is **opt-in and explicit**: `cache_control: {"type":"ephemeral", "ttl":"5m"|"1h"}` on up to **4 blocks** (or a single top-level `cache_control` for automatic caching). Cache lookup covers `tools` → `system` → `messages` *prefix* up to and including the marked block. Per-model minimum prompt sizes (2.7/§9).
11. Thinking (extended/adaptive) returns `thinking` blocks with cryptographic **`signature`** fields (and possibly `redacted_thinking` blocks) that **must be echoed back unmodified** in tool-use continuations, or the API returns 400. Manual thinking forbids forced `tool_choice` (`any`/`tool`) and assistant prefills; `budget_tokens` ≥ 1024 and < `max_tokens` (except interleaved).
12. Versioned headers matter: **`anthropic-version: 2023-06-01` is mandatory on every request** (including `GET /v1/models`); `x-api-key` **or** `Authorization: Bearer` authenticates; **`anthropic-beta: a,b,c`** gates beta features (comma-separated, not repeated headers).

---

## 1. Endpoints, transport, headers, auth

### 1.1 Base URL and endpoint inventory

Base URL: `https://api.anthropic.com` (the direct Claude API). <https://platform.claude.com/docs/en/api/overview>

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/messages` | POST | Create a Message (the core endpoint; `stream: true` switches to SSE) |
| `/v1/messages/count_tokens` | POST | Count input tokens of a would-be message request without creating it |
| `/v1/messages/batches` | POST | Create a Message Batch (async, 50% cheaper) |
| `/v1/messages/batches` | GET | List message batches |
| `/v1/messages/batches/{message_batch_id}` | GET | Retrieve a batch |
| `/v1/messages/batches/{message_batch_id}/results` | GET | Stream/fetch batch results (.jsonl) |
| `/v1/messages/batches/{message_batch_id}/cancel` | POST | Cancel a batch |
| `/v1/messages/batches/{message_batch_id}` | DELETE | Archive a batch |
| `/v1/models` | GET | List models |
| `/v1/models/{model_id}` | GET | Get one model |
| `/v1/complete` | POST | **Legacy** Text Completions API (deprecated; ignore for new work) |
| `/v1/files`, `/v1/skills`, `/v1/experimental/...` | various | Files/Skills/Admin APIs (out of scope but coexist under the same headers) |

Only `/v1/messages`, `/v1/messages/count_tokens`, and `/v1/models` are required for a minimally conformant Anthropic-compatible serving surface; batches are §12.3.

### 1.2 Required request headers

From <https://platform.claude.com/docs/en/api/overview> and <https://platform.claude.com/docs/en/api/beta-headers>:

| Header | Value | Required |
|---|---|---|
| `x-api-key` | Anthropic API key (`sk-ant-...`) | **Yes, unless** `Authorization` is used |
| `Authorization` | `Bearer <token>` (OAuth / workload-identity federation access token from `POST /v1/oauth/token`) | **Yes, unless** `x-api-key` is used |
| `anthropic-version` | API version date, canonical value `2023-06-01` | **Yes — on every request, including `GET /v1/models`** |
| `content-type` | `application/json` | **Yes** for POST bodies |
| `anthropic-beta` | Comma-separated beta feature names, e.g. `interleaved-thinking-2025-05-14,prompt-caching-2024-07-31` | Optional; only for beta features |
| `anthropic-user-profile-id` | User profile ID for acting on behalf of another party | Optional; requires `user-profiles` beta |

Behavioral notes:

- **Both auth schemes** are accepted on the same endpoints; SDKs use `x-api-key` by default. Missing/invalid credentials → `401 authentication_error`; a valid key without rights to a resource → `403 permission_error`.
- Missing `anthropic-version` → `400 invalid_request_error`. Only two versions exist: `2023-01-01` (initial) and `2023-06-01` (current; changed SSE to named events, removed `data: [DONE]`, removed legacy `exception`/`truncated` values). <https://platform.claude.com/docs/en/api/versioning>
- `anthropic-beta` is a *single* header whose value is a **comma-separated list** (`anthropic-beta: feature1,feature2,feature3`). Invalid or unauthorized beta names → `400 invalid_request_error` naming the bad value(s). Officially known beta header strings include: `message-batches-2024-09-24`, `prompt-caching-2024-07-31`, `computer-use-2024-10-22`, `computer-use-2025-01-24`, `pdfs-2024-09-25`, `token-counting-2024-11-01`, `token-efficient-tools-2025-02-19`, `output-128k-2025-02-19`, `files-api-2025-04-14`, `mcp-client-2025-04-04`, `dev-full-thinking-2025-05-14`, `interleaved-thinking-2025-05-14`, `code-execution-2025-05-22`, `extended-cache-ttl-2025-04-11`, `context-1m-2025-08-07`, `context-management-2025-06-27`, `model-context-window-exceeded-2025-08-26`, `skills-2025-10-02`, `fast-mode-2026-02-01`, `output-300k-2026-03-24`, `fine-grained-tool-streaming-2025-05-14` (legacy; now per-tool `eager_input_streaming`), and others listed in the API reference/models list. Many features that began as betas (PDFs, token counting, prompt caching, web tools, 1M context) are now GA and need **no** header.
- `user-agent`: the API does not enforce a value; official SDKs send e.g. `anthropic-python/0.x.y`. Servers should treat it as opaque.
- CORS: official browser SDKs need `anthropic-dangerous-direct-browser-access: true` (SDK option); servers should accept the header and reply with permissive CORS preflights if browser clients matter.

### 1.3 Response headers (every response)

<https://platform.claude.com/docs/en/api/overview>, <https://platform.claude.com/docs/en/api/rate-limits>, <https://platform.claude.com/docs/en/api/errors>:

| Header | Meaning |
|---|---|
| `request-id` | Globally unique `req_...` ID; also echoed as `request_id` in error bodies |
| `anthropic-organization-id` | Org of the credential |
| `anthropic-workspace-id` | `wrkspc_...` workspace ID of the credential (when applicable) |
| `anthropic-ratelimit-requests-limit` / `-remaining` / `-reset` | Request rate-limit window info (reset in RFC 3339) |
| `anthropic-ratelimit-tokens-limit` / `-remaining` / `-reset` | Aggregate token limit info (most restrictive limit shown; remaining rounded to nearest 1000) |
| `anthropic-ratelimit-input-tokens-*` / `anthropic-ratelimit-output-tokens-*` | Separate ITPM/OTPM windows |
| `retry-after` | Seconds to wait before retry (on 429; **absent** on spend-cap 429s) |
| `anthropic-priority-input-tokens-*`, `anthropic-priority-output-tokens-*` (`-limit/-remaining/-reset`) | Present when a Priority Tier evaluation happened (`service_tier: auto`) |

### 1.4 Request size limits

<https://platform.claude.com/docs/en/api/errors#request-size-limits>:

| Endpoint type | Max body size |
|---|---|
| Messages API | **32 MB** |
| Token Counting API | 32 MB |
| Batch API | 256 MB |
| Files API | 500 MB |

Oversize → `413 request_too_large` (returned by Cloudflare before hitting the model servers).

### 1.5 Timeouts and long requests

<https://platform.claude.com/docs/en/api/errors#long-requests>: networks may drop idle connections; Anthropic recommends streaming (or batches) for anything long-running, and the official SDKs *refuse* non-streaming Messages requests whose expected duration exceeds ~10 minutes (large `max_tokens`). Suggested: TCP keep-alive, connection pooling, and streaming for large outputs. Gateway timeout behavior: `504 timeout_error` is possible server-side.

### 1.6 Versioning policy (why servers should be tolerant)

<https://platform.claude.com/docs/en/api/versioning>: for a given `anthropic-version`, existing inputs/outputs won't change, but Anthropic **may** add optional inputs, add output fields, add **new enum variants** (e.g., new `stop_reason` values, new SSE event types, new content-block types), and change conditions for specific error types. **Implementations must pass through unknown content-block types, event types, and fields gracefully.**

### 1.7 ID formats (prefixes)

| Prefix | Object |
|---|---|
| `msg_...` | Message id |
| `msgbatch_...` | Message Batch id |
| `toolu_...` | client tool_use block id |
| `srvtoolu_...` | server_tool_use block id |
| `req_...` | request id (header/body) |
| `file_...` | Files API file id |
| `container_...` | code-execution container id |
| `wrkspc_...` | workspace id |

ID shapes can change over time — treat as opaque strings.

---

## 2. `POST /v1/messages` — request schema

Canonical reference: <https://platform.claude.com/docs/en/api/messages/create>. `stream` boolean selects JSON vs SSE response. Unless marked required, parameters are optional.

### 2.1 Top-level scalar / structural parameters

| Field | Type | Req | Default | Constraints / semantics |
|---|---|---|---|---|
| `model` | string | **Req** | — | Model ID (e.g. `claude-opus-5`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`, older dated IDs) or API alias. Unknown model → `404 not_found_error`-style failure. |
| `messages` | array of `MessageParam` | **Req** | — | Max **100,000 messages**. Only `user`, `assistant` (and, on newest models, mid-conversation `system`) roles. See §?2.2. |
| `max_tokens` | number (int) | **Req** | — | Max tokens to generate; model stops earlier if finished. `0` = cache pre-warm only (empty `content`, `stop_reason:"max_tokens"`, populated `usage`). Per-model maximums in §12.2. Counts **include** thinking tokens when thinking is enabled. |
| `system` | string **or** array of `TextBlockParam` | Opt | none | System prompt. Block form allows per-block `cache_control` and `citations` params. |
| `stop_sequences` | array of string | Opt | none | Custom stop sequences; generation stops when one is produced → `stop_reason:"stop_sequence"`, `stop_sequence:"<matched>"`. No documented count cap. |
| `stream` | boolean | Opt | `false` | `true` → SSE event stream (§4). |
| `temperature` | number | Opt | `1.0` | Range `0.0–1.0`. Not fully deterministic even at 0. **Model rules:** on the newest models (Claude Fable 5/Mythos 5/Mythos Preview/Opus 5/Opus 4.8/Opus 4.7/Sonnet 5) any non-default sampling params → 400; on older thinking-capable models, while thinking is enabled, `temperature` and `top_k` are incompatible and `top_p` must be ≥ 0.95. |
| `top_p` | number | Opt | none | Nucleus sampling. "Advanced use cases only." Same model restrictions as above. |
| `top_k` | number (int) | Opt | none | Only sample from top-K options. Same restrictions. |
| `metadata` | object `{user_id?}` | Opt | — | `metadata.user_id`: opaque external user identifier (uuid/hash; **no PII**) used for abuse detection. |
| `service_tier` | `"auto"` \| `"standard_only"` | Opt | `"auto"` | Use Priority Tier capacity when possible vs force standard tier. Response reports actual tier in `usage.service_tier` (`standard`/`priority`/`batch`). |
| `thinking` | `ThinkingConfigParam` | Opt | model-dependent (adaptive default on newest) | `{type:"enabled", budget_tokens:int, display?: "summarized"|"omitted"}` \| `{type:"disabled"}` \| `{type:"adaptive", display?}`. §7. |
| `tool_choice` | object | Opt | `{"type":"auto"}` | `{"type":"auto"}` \| `{"type":"any"}` \| `{"type":"tool","name":<tool name>}` \| `{"type":"none"}`; first three accept `disable_parallel_tool_use?: boolean` (default `false`). §2.6. |
| `tools` | array of tool definitions | Opt | none | Custom tools with `input_schema` (JSON Schema) or Anthropic-hosted tools. §2.5. |
| `cache_control` | `{"type":"ephemeral", "ttl"?: "5m"|"1h"}` | Opt | — | **Top-level automatic prompt caching**: API applies a cache breakpoint at the last cacheable block. Incompatible with a conflicting block-level marker on that same last block (400) and unavailable if 4 explicit breakpoints already exist (400). §9. |
| `container` | `{id?: string, skills?: [{skill_id, type:"anthropic"|"custom", version?}]}` **or** string | Opt | — | Container (code-execution sandbox) reuse across requests. |
| `inference_geo` | string | Opt | workspace default | Geographic region for inference (e.g. `"us"`). Echoed in `usage.inference_geo`. |
| `output_config` | `{effort?: "low"|"medium"|"high"|"xhigh"|"max", format?: {type:"json_schema", schema: object}}` | Opt | `effort: "high"` | Effort levels (§12.4) and **structured outputs** (JSON-schema-constrained responses). |
| `mcp_servers` | array | Opt (beta) | — | MCP connector servers (`mcp-client` beta). Out of scope for base implementations. |
| `context_management` / context-editing strategies | object(s) | Opt (beta, per docs) | — | Server-side tool-result/thinking clearing (`clear_tool_uses_20250919`, `clear_thinking_20251015`) under `context-management-2025-06-27` beta; model capabilities expose it via Models API. §12.5. |

### 2.2 `messages[]`

Ref: <https://platform.claude.com/docs/en/api/messages/create>.

#### 2.2.1 Roles

- **`"user"`** — end-user input, tool results (`tool_result` blocks), images, documents.
- **`"assistant"`** — prior model output: `text`, `tool_use`, `thinking`, `redacted_thinking`, server tool blocks. The model generates the *next* turn.
- **`"system"`** *(current API reference; GA only on Claude Fable 5, Mythos 5, Opus 4.8, Opus 5, Sonnet 5)* — *mid-conversation system message*: inserts/updates system instructions partway through a conversation without invalidating the cached prefix. No beta header. With the `mid-conversation-tool-changes-2026-07-01` beta the content may also contain `tool_addition`/`tool_removal` blocks referencing tools by name (`tool_reference`) to change tool availability mid-conversation (Opus 5/4.8/Fable 5/Mythos 5; not Sonnet 5). <https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages>

There is **no** conventional in-band `system` role otherwise — the top-level `system` parameter is the canonical place (current token-counting reference still states "there is no `"system"` role for input messages in the Messages API", describing the classic contract).

#### 2.2.2 Turn structure rules

- Models are trained for **alternating `user`/`assistant` turns**; the API **combines consecutive same-role turns into a single turn** (it does not error). Empty conversations are invalid.
- The conversation normally **starts with a `user` message** (mid-conversation `system` messages are inserted between turns but not as the first conversational turn).
- If the **final message has role `assistant`**, the response **continues directly from the end of that message's content** — this is Anthropic's *prefill* mechanism (`[{"role":"user","content":"...?"},{"role":"assistant","content":"The best answer is ("}]`). Rules/caveats:
  - Prefills are blocked while thinking is enabled ("you can't pre-fill the assistant response while thinking is on"). <https://platform.claude.com/docs/en/build-with-claude/thinking#limitations>
  - With forced tool use (`tool_choice: any|tool`), the API prefills internally and the model emits **no leading natural-language text** before `tool_use` blocks. <https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools#forcing-tool-use>
  - Classic validation: a prefilled assistant message whose text ends with **trailing whitespace** is rejected (`invalid_request_error`); keep prefill content clean.
- **Tool-pairing rules** (enforced, 400 on violation): every `tool_use` block in the final assistant turn must be immediately followed by a `user` message whose content **begins with** matching `tool_result` blocks — one per `tool_use` `id` — before any other content. If the assistant turn also contains a **server tool call with no result yet**, the user message may contain **only** `tool_result` blocks (no trailing text, same `tools` array must be resent). Error text to expect: `` `tool_use` ids were found without `tool_result` blocks immediately after ``. <https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls>
- **Thinking round-trip rules:** within the latest assistant message, consecutive `thinking`/`redacted_thinking` blocks must match what the model generated, **unmodified and in original order** (modified → 400). Text placed into an *omitted*-display thinking block's empty `thinking` field is ignored, not rejected. <https://platform.claude.com/docs/en/build-with-claude/thinking#preserving-thinking-blocks>

#### 2.2.3 `content` — string shorthand

Each message's `content` is either:

```json
{"role": "user", "content": "Hello, Claude"}
```

or an array of content blocks; the string is exactly equivalent to a single text block:

```json
{"role": "user", "content": [{"type": "text", "text": "Hello, Claude"}]}
```

#### 2.2.4 Empty/blank content

Empty text blocks cannot be cached; a text block must contain text. (Warm-up placeholder user messages for cache pre-warming need *non-whitespace* content.)

### 2.3 Input content block types (`ContentBlockParam` union)

Every block has a `type` discriminator. **`cache_control`** (`{"type":"ephemeral","ttl"?: "5m"|"1h"}`, default TTL 5m) may attach to most top-level blocks to create a cache breakpoint (§9).

| `type` | Fields | Notes |
|---|---|---|
| **`text`** | `text: string`; optional `cache_control`, `citations: TextCitationParam[]` | The basic block. `citations` param array exists for round-tripping cited text (types: `char_location`, `page_location`, `content_block_location`, `web_search_result_location`, `search_result_location`). |
| **`image`** | `source`, optional `cache_control`, `transformations` | `source` is one of: `{"type":"base64","media_type":"image/jpeg"\|"image/png"\|"image/gif"\|"image/webp","data":"<b64>"}`; `{"type":"url","url":"https://..."}`; `{"type":"file","file_id":"file_..."}`. `transformations.oversized_image: "downsize"(default)|"error"` — with `"error"`, over-limit images are rejected with a 400 naming dimensions instead of silent downscale. Limits in §8. |
| **`document`** | `source`, optional `cache_control`, `citations: {enabled?: bool}`, `title?: string`, `context?: string` | `source` one of: base64 PDF `{"type":"base64","media_type":"application/pdf","data":...}`; plain text `{"type":"text","media_type":"text/plain","data":...}`; `{"type":"url","url":...}`; `{"type":"file","file_id":...}`; or `{"type":"content","content": string \| (TextBlockParam|ImageBlockParam)[]}` (a document made of blocks). §8.3. |
| **`search_result`** | `content: TextBlockParam[]`, `source: string`, `title: string`, optional `cache_control`, `citations` | Client-supplied search results for citations workflows. |
| **`tool_use`** | `id: string`, `name: string`, `input: object`, optional `cache_control`, `caller`, `toolset_name` | Appears in *assistant* role (round-tripped history). `caller`: `{type:"direct"}` (default) or server-tool caller `{type:"code_execution_20250825"|"code_execution_20260120", tool_id}` (programmatic tool calling). `toolset_name` for computer/browser toolset member calls. |
| **`tool_result`** | `tool_use_id: string`, optional `content`, `is_error: bool`, `cache_control`, `toolset_name` | Appears in *user* role. `content`: string **or** array of `text` / `image` / `document` / `search_result` / `tool_reference` / `browser_state` blocks; omitted `content` = empty result. `is_error: true` marks failed executions. |
| **`thinking`** | `signature: string`, `thinking: string` | Round-trip of an earlier response's thinking block; `signature` verified server-side. Must pass back unmodified in original order. |
| **`redacted_thinking`** | `data: string` | Opaque encrypted redacted thinking; pass back unchanged. |
| **`server_tool_use`** | `id`, `name: "web_search"\|"web_fetch"\|"code_execution"\|"bash_code_execution"\|"text_editor_code_execution"\|"tool_search_tool_regex"\|"tool_search_tool_bm25"`, `input: object`, optional `caller` | Round-tripped history of server tool calls. |
| **`web_search_tool_result`** | `tool_use_id`, `content`: `WebSearchResultBlock[]` (`{type:"web_search_result", title, url, encrypted_content, page_age?}`) **or** `{type:"web_search_tool_result_error", error_code: "invalid_tool_input"\|"unavailable"\|"max_uses_exceeded"\|"too_many_requests"\|"query_too_long"\|"request_too_large"}`; optional `cache_control`, `caller` | Round-tripped history of server-side web search results. |
| **`web_fetch_tool_result`** | `tool_use_id`, `content`: `{type:"web_fetch_result", url, retrieved_at?|null, content: DocumentBlockParam}` **or** `{type:"web_fetch_tool_result_error", error_code: "invalid_tool_input"\|"url_too_long"\|"url_not_allowed"\|"url_not_in_prior_context"\|"url_not_accessible"\|"unsupported_content_type"\|"too_many_requests"\|"max_uses_exceeded"\|"unavailable"}` | Same idea for web fetch. |
| **`code_execution_tool_result`** | `tool_use_id`, `content`: `{type:"code_execution_tool_result_error", error_code: "invalid_tool_input"\|"unavailable"\|"too_many_requests"\|"execution_time_exceeded"}` \| `{type:"code_execution_result", return_code, stderr, stdout, content: [{type:"code_execution_output", file_id}]}` \| `{type:"encrypted_code_execution_result", encrypted_stdout, return_code, stderr, content: [...]}` | Server-side code execution results. |
| **`bash_code_execution_tool_result`** | `tool_use_id`, `content`: error (`invalid_tool_input`/`unavailable`/`too_many_requests`/`execution_time_exceeded`/`output_file_too_large`) or `{type:"bash_code_execution_result", return_code, stderr, stdout, content:[{type:"bash_code_execution_output", file_id}]}` | Server-side bash tool results. |
| **`text_editor_code_execution_tool_result`** | `tool_use_id`, `content`: error (`+file_not_found`, optional `error_message`) \| `{type:"text_editor_code_execution_view_result", content, file_type:"text"|"image"|"pdf", num_lines?, start_line?, total_lines?}` \| `{type:"text_editor_code_execution_create_result", is_file_update}` \| `{type:"text_editor_code_execution_str_replace_result", ...}` | Server-side text-editor tool results. |
| **`tool_search_tool_result`** | `tool_use_id`, `content`: error \| `{type:"tool_search_tool_search_result", tool_references: [{type:"tool_reference", tool_name}]}` | Result of tool-search tools. |
| **`container_upload`** | `file_id: string` | File to upload into the code-execution container. |
| **`tool_reference`** | `tool_name: string` (+ optional `cache_control`) | Inside `tool_result.content` — references a deferred tool (tool search). |
| **`browser_state`** | `tabs: [{tab_id,title,url,active?}]`, optional `state_changes` ([`tab_opened`, `download_started`, `download_completed`, `download_failed` entries]) | Browser toolset state; at most one per tool_result; non-error results only. |

Implementation advice: implement `text`, `image`, `document`, `thinking`, `redacted_thinking`, `tool_use`, `tool_result` fully; treat server-tool blocks as opaque round-trippable values unless you emulate them.

### 2.4 `system`

- `system: "You are a helpful assistant"` — plain string, or
- `system: [{"type":"text","text":"...","cache_control":{"type":"ephemeral"}}]`.

Multi-block system arrays work; order matters (cache prefix). `citations` params may be present on system text blocks for round-trip completeness.

### 2.5 `tools[]`

Ref: <https://platform.claude.com/docs/en/api/messages/create>, <https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools>, and tool reference pages.

#### 2.5.1 Custom (client) tool definition

```json
{
  "type": "custom",                       // optional; default
  "name": "get_stock_price",              // REQUIRED; regex ^[a-zA-Z0-9_-]{1,64}$
  "description": "Get the current stock price for a given ticker symbol.",
  "input_schema": {                       // REQUIRED; JSON Schema (draft 2020-12), root "type":"object"
    "type": "object",
    "properties": {"ticker": {"type":"string","description":"The stock ticker symbol, e.g. AAPL"}},
    "required": ["ticker"]
  },
  "strict": true,                          // optional: grammar-constrained sampling → input guaranteed to match schema (not on computer/browser toolsets)
  "input_examples": [{"ticker":"AAPL"}],   // optional: example inputs (must validate against schema)
  "eager_input_streaming": true,           // optional: stream partial JSON of this tool's input without buffering (fine-grained streaming)
  "defer_loading": false,                  // optional: withhold from prompt until surfaced via tool_search
  "allowed_callers": ["direct", "code_execution_20250825", "code_execution_20260120", "code_execution_20260521"], // optional
  "cache_control": {"type":"ephemeral"}    // optional: cache breakpoint covers prefix up to and incl. this tool
}
```

The model answers with `tool_use` blocks `{id:"toolu_01D7FLrfh4GYq7yT1ULFeyMV","name":"get_stock_price","input":{"ticker":"^GSPC"}}`.

#### 2.5.2 Built-in (Anthropic-defined) tools in `tools[]`

Declared with `"type": "<name>_<yyyymmdd>"` entries rather than `input_schema`. Common versions (from the current API reference):

| Tool entry `type` | `name` | Role |
|---|---|---|
| `bash_20250124` | `bash` | client-side bash tool |
| `text_editor_20250124` / `text_editor_20250429` / `text_editor_20250728` | `str_replace_editor` / `str_replace_based_edit_tool` | client-side file editor |
| `computer_20241022` / `computer_20250124` (or toolset `computer_toolset_20260801`) | `computer` | computer use |
| `browser_toolset_20260801` | (toolset members, e.g. `navigate`, `screenshot`) | browser use toolset |
| `memory_20250818` | `memory` | client-side memory |
| `code_execution_20250522` / `code_execution_20250825` / `code_execution_20260120` / `code_execution_20260521` | `code_execution` | **server-side** code execution |
| `web_search_20250305` / `web_search_20260209` / `web_search_20260318` | `web_search` | **server-side** web search; params: `max_uses`, `allowed_domains` xor `blocked_domains`, `user_location {type:"approximate", city?, country?, region?, timezone?}` |
| `web_fetch_20250910` / `web_fetch_20260209` / `web_fetch_20260309` / `web_fetch_20260318` | `web_fetch` | **server-side** web fetch; params: `allowed_domains`, `blocked_domains`, `max_uses`, `max_content_tokens`, `citations:{enabled}` |
| `tool_search_tool_regex_20251119` / `tool_search_tool_bm25_*` | `tool_search_tool_regex` / `tool_search_tool_bm25` | defer-loading tool search |

Server tools accept the same optional `cache_control`, `defer_loading`, `strict`, `allowed_callers`, `input_examples` fields. Toolset entries (computer/browser) accept `cache_control`, `allowed_callers` and reject `strict`. For compatibility serving, builtin tool entries should be validated & passed through (or 400-rejected explicitly) — they trigger Anthropic-side execution you cannot emulate locally.

### 2.6 `tool_choice`

<https://platform.claude.com/docs/en/api/messages/create>, <https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools>:

| Variant | Shape | Semantics |
|---|---|---|
| `auto` (default) | `{"type":"auto","disable_parallel_tool_use":false}` | Model decides; text and/or tool_use allowed. `disable_parallel_tool_use:true` ⇒ *at most one* tool use. |
| `any` | `{"type":"any",...}` | Model **must** call at least one tool; no natural-language preface. `disable_parallel_tool_use:true` ⇒ *exactly one*. |
| `tool` | `{"type":"tool","name":"get_weather",...}` | Force a *specific* tool; exactly one call. |
| `none` | `{"type":"none"}` | Tools forbidden (`tools[]` may still be present, e.g. to preserve a cached prefix). |

Forced use (`any`/`tool`) is **incompatible with manual extended thinking** (`thinking.type:"enabled"` → 400) but works with adaptive thinking.

### 2.7 `thinking` parameter

```json
"thinking": {"type": "enabled", "budget_tokens": 10000, "display": "summarized"}
"thinking": {"type": "disabled"}
"thinking": {"type": "adaptive", "display": "omitted"}
```

- `budget_tokens`: REQUIRED when `type:"enabled"`; must be **≥ 1024** and **< `max_tokens`** (thinking counts toward `max_tokens`). Exception: with **interleaved thinking**, `budget_tokens` may exceed `max_tokens` because the budget spans all thinking blocks of the turn. Incompatible with `max_tokens: 0` cache pre-warming.
- `display`: `"summarized"` (default on most models) streams/returns a summarized chain-of-thought; `"omitted"` returns thinking blocks with empty `thinking` text but a valid `signature` (default on newest models: Opus 5/Sonnet 5/Fable 5/Mythos 5/Mythos Preview/Opus 4.8/Opus 4.7).
- Model availability matrix is in §7.1.

### 2.8 Reference body (non-streaming)

```json
{
  "model": "claude-opus-5",
  "max_tokens": 1024,
  "system": [{"type":"text","text":"Today's date is 2024-06-01."}],
  "messages": [{"role":"user","content":"Hello, world"}],
  "temperature": 1,
  "stop_sequences": ["\n\nHuman:"],
  "stream": false,
  "metadata": {"user_id": "13803d75-b56b-4675-a5b7-544f4b6a1234"},
  "tools": [{"name":"get_weather","input_schema":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}],
  "tool_choice": {"type":"auto","disable_parallel_tool_use":false},
  "thinking": {"type":"adaptive"},
  "service_tier": "auto",
  "top_k": 5,
  "top_p": 0.7
}
```

---

## 3. `POST /v1/messages` — response schema (non-streaming)

HTTP `200`, `content-type: application/json` (or `text/event-stream` when streaming). Canonical schema from the API reference "Returns" section of <https://platform.claude.com/docs/en/api/messages/create>:

```json
{
  "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
  "type": "message",
  "role": "assistant",
  "model": "claude-opus-5",
  "content": [{"type": "text", "text": "Hi! My name is Claude.", "citations": null}],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "stop_details": null,
  "container": null,
  "usage": {
    "input_tokens": 2095,
    "output_tokens": 503,
    "cache_creation_input_tokens": 2051,
    "cache_read_input_tokens": 2051,
    "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
    "server_tool_use": {"web_fetch_requests": 2, "web_search_requests": 0},
    "output_tokens_details": {"thinking_tokens": 0},
    "service_tier": "standard",
    "inference_geo": "global"
  }
}
```

### 3.1 Field-by-field

| Field | Type | Notes |
|---|---|---|
| `id` | string | `msg_...` unique ID. Format/length may change. |
| `type` | `"message"` | Constant. |
| `role` | `"assistant"` | Constant. |
| `model` | string | The model that generated the response (resolved ID, not alias). |
| `content` | array of `ContentBlock` | See §3.2. Even a plain text answer is an array of one `text` block. If the request's final message was an assistant prefill, response content **continues directly after** it (response contains only the continuation, e.g. `[{"type":"text","text":"B)"}]`). May be `[]` (e.g. cache pre-warm with `max_tokens:0`, or an "empty response" `end_turn` edge case of 2–3 no-content tokens, mostly seen after tool results — handle it gracefully). |
| `stop_reason` | enum \| `null` | `"end_turn"` (natural completion) · `"max_tokens"` (hit request `max_tokens` or model max; **response is truncated — check whether the last block is an incomplete `tool_use`** and retry with a larger budget) · `"stop_sequence"` (matched one of `stop_sequences`; see `stop_sequence`) · `"tool_use"` (model called ≥1 tool) · `"pause_turn"` (server-tool sampling loop reached its iteration limit, default 10; resend the assistant response as-is to continue) · `"refusal"` (safety classifier intervened; see `stop_details`) · `"model_context_window_exceeded"` (generation hit the context window, Claude 4.5+; older models 400-error instead unless `model-context-window-exceeded-2025-08-26` beta). **Non-streaming: always non-null.** Streaming: null in `message_start`, set in `message_delta`. |
| `stop_sequence` | string \| `null` | Which `stop_sequences` entry matched, if any. |
| `stop_details` | `{type:"refusal", category, explanation} \| null` | Non-null only for `refusal`. `category`: `"cyber"`, `"bio"`, `"frontier_llm"`, `"reasoning_extraction"` (request tried to extract raw CoT), `"general_harms"`. `explanation`: human-readable rationale (unstable text, may be null). |
| `container` | `{id, expires_at, skills: [{skill_id, type, version}] \| null} \| null` | Info about the code-execution container used. Pass `container.id` back to reuse. |
| `usage` | object | Billing/rate-limit usage. See §3.3. |

### 3.2 Response `content` block types (`ContentBlock` union)

| `type` | Fields | Notes |
|---|---|---|
| **`text`** | `text: string`, `citations: TextCitation[] \| null` | Citations appear when document/PDF citations or web search results are enabled and used. Citation variants: `char_location`, `page_location`, `content_block_location`, `web_search_result_location` (`{cited_text, encrypted_index, title, url}`), `search_result_location`. |
| **`thinking`** | `thinking: string`, `signature: string` | Summarized thinking text (empty string when `display:"omitted"`). `signature` is an **opaque encrypted blob** authenticating the block; pass back unmodified. |
| **`redacted_thinking`** | `data: string` | Safety-redacted thinking; opaque+encrypted. Pass back unchanged. |
| **`tool_use`** | `id: string` (`toolu_...`), `name: string`, `input: object`, optional `caller`, `toolset_name` | Client tool call. `caller.type:"direct"` unless produced by programmatic (server-orchestrated) tool calling. |
| **`server_tool_use`** | `id: string` (`srvtoolu_...`), `name: "web_search"\|"web_fetch"\|"code_execution"\|"bash_code_execution"\|"text_editor_code_execution"\|"tool_search_tool_regex"\|"tool_search_tool_bm25"`, `input: object`, optional `caller` | A server-side tool invocation. Its result usually arrives inline in the same response as the matching result block. |
| **`web_search_tool_result`** | `tool_use_id`, `content: WebSearchResultBlock[] \| error`, `caller?` | `WebSearchResultBlock {type:"web_search_result", title, url, encrypted_content, page_age}` — the **result payload is server-encrypted** (`encrypted_content`); clients pass it back verbatim but cannot read it. Error variant as §2.3. |
| **`web_fetch_tool_result`** | `tool_use_id`, `content: {type:"web_fetch_result", url, retrieved_at, content: DocumentBlock} \| error` | Fetched page as an embedded document. |
| **`code_execution_tool_result`** | `tool_use_id`, `content: error \| code_execution_result \| encrypted_code_execution_result` | `code_execution_result: {return_code, stderr, stdout, content: [{type:"code_execution_output", file_id}]}` (output files via Files API ids). |
| **`bash_code_execution_tool_result`** | same shape as code execution w/ `bash_` prefixes | |
| **`text_editor_code_execution_tool_result`** | view/create/str_replace result variants | |
| **`tool_search_tool_result`** | `tool_references` list | |
| **`container_upload`** | `file_id` | Echo of uploaded container file. |

### 3.3 `usage` object

From the API reference and SDK `Usage` type:

| Field | Type | Meaning |
|---|---|---|
| `input_tokens` | int | Input tokens **after the last cache breakpoint** (i.e., not read from / written to cache). |
| `cache_creation_input_tokens` | int \| null | Input tokens written to cache this request. |
| `cache_read_input_tokens` | int \| null | Input tokens read from cache. |
| `cache_creation` | `{ephemeral_5m_input_tokens: int, ephemeral_1h_input_tokens: int} \| null` | TTL breakdown of `cache_creation_input_tokens` (its parts sum to it). |
| `output_tokens` | int | Output tokens generated (thinking included; non-zero even on empty visible output). |
| `output_tokens_details` | `{thinking_tokens: int} \| null` | Read-only decomposition: how many output tokens were internal reasoning (raw, before summarization). ≤ `output_tokens`. |
| `server_tool_use` | `{web_search_requests: int, web_fetch_requests: int} \| null` | Server tool request counts for billing. |
| `service_tier` | `"standard" \| "priority" \| "batch" \| null` | Which tier served the request. |
| `inference_geo` | string \| null | Where inference ran. |

**Total input = `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`.** Token counts need not match visible content one-to-one (server-side prompt transformation/parsing). <https://platform.claude.com/docs/en/build-with-claude/prompt-caching#tracking-cache-performance>, <https://platform.claude.com/docs/en/api/messages/create>.

---

## 4. Streaming protocol (SSE)

Ref: <https://platform.claude.com/docs/en/build-with-claude/streaming>, SDK event types at <https://github.com/anthropics/anthropic-sdk-python/tree/main/src/anthropic/types>.

### 4.1 Transport conventions

- Enable with `"stream": true`. Response: `200 OK`, `content-type: text/event-stream`.
- Every event is a **named** SSE event:
  ```
  event: <event_type>\n
  data: <single-line JSON whose "type" equals the event name>\n
  \n
  ```
- Plexing: `event:` line carries the same name as the JSON `type`. No `data: [DONE]` sentinel (removed in API version `2023-06-01`).
- Deltas are **incremental** (`"Hello"`, `"!"` — not cumulative prefixes).
- Clients must ignore **unknown event types / unknown delta types** (versioning policy).

### 4.2 Event lifecycle (ordering guarantees)

1. **`message_start`** — exactly once, first event.
2. For each content block *i* (index aligned with the final Message `content[]`): **`content_block_start`** → **1+ `content_block_delta`** → **`content_block_stop`**. Blocks arrive in order; a new block begins only after the previous block's `_stop`. (Exception: server-side *fallback* responses can emit a `fallback` content block as a bare start+stop pair with no deltas.)
3. **one or more `message_delta`** events — top-level mutations of the Message object (stop_reason, stop_sequence, container, stop_details) with **cumulative usage**.
4. **`message_stop`** — exactly once, last event of a successful stream.
- **`ping`** events (`{"type":"ping"}`) may be interleaved **anywhere** for keep-alive.
- An **`error`** event may occur **anywhere**, including after `message_start` (then the stream typically terminates without `message_stop`). See §5.4.

### 4.3 Exact event shapes

**`message_start`** — message skeleton with **empty** `content[]`, `stop_reason: null`; `usage` already carries the input accounting and a small initial `output_tokens` (<5):

```sse
event: message_start
data: {"type": "message_start", "message": {"id": "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY", "type": "message", "role": "assistant", "content": [], "model": "claude-opus-5", "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 25, "output_tokens": 1}}}
```

When caching/web tools are in play, the `usage` in `message_start` can already include `cache_creation_input_tokens` / `cache_read_input_tokens`.

**`content_block_start`** — opens block at `index`; `content_block` is the *skeleton* of the block (`text` empty, `tool_use.input` empty object, `thinking.thinking`/`signature` empty):

```sse
event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}

event: content_block_start
data: {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_01T1x1fJ34qAmk2tNTrN7Up6", "name": "get_weather", "input": {}}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"server_tool_use","id":"srvtoolu_014hJH82Qum7Td6UV8gDXThB","name":"web_search","input":{}}}
```

Server-tool *result* blocks may appear as **`content_block_start` events carrying the complete result** (no deltas in between), e.g. a full `web_search_tool_result` with its `encrypted_content` result list.

**`content_block_delta`** — updates block at `index`. `delta` variants:

```sse
event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"location\":"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": " \"San Francisc"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "I need to find the GCD of 1071 and 462..."}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds..."}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "citations_delta", "citation": {"type": "web_search_result_location", "cited_text": "...", "encrypted_index": "...", "title": "...", "url": "..."}}}
```

Delta-type rules:

- `text_delta.text` — incremental text; concatenate to assemble the block.
- `input_json_delta.partial_json` — **string fragments of the final `tool_use`/`server_tool_use` `input` JSON**. Concatenate all fragments; parse the JSON **after** `content_block_stop` (use partial-JSON parsers for preview). The first delta is commonly the empty string `""`. Current models emit one complete key/value at a time, distributed across several chunked deltas — expect *gaps* of no events mid-tool-call while the model works (unless `eager_input_streaming`/fine-grained tool streaming is on).
- `thinking_delta.thinking` — incremental summarized thinking text. Absent entirely when `display:"omitted"` (block opens, then only a `signature_delta` arrives, then closes).
- `signature_delta.signature` — sent **exactly once, just before `content_block_stop`** for each `thinking` block. Concatenate/finalize into the block's `signature`.
- `citations_delta.citation` — incremental citations for a text block (`citations: [...]` on the final block).

**`content_block_stop`**:

```sse
event: content_block_stop
data: {"type": "content_block_stop", "index": 0}
```

**`message_delta`** — top-level updates + **cumulative** usage; `delta` carries whichever of `stop_reason`, `stop_sequence`, `stop_details`, `container` changed:

```sse
event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": 15}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":89}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":10682,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":510,"server_tool_use":{"web_search_requests":1}}}
```

`usage` here is `MessageDeltaUsage`: `output_tokens` (int, cumulative, required) plus optional cumulative `input_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `server_tool_use`, `output_tokens_details`. **Do not sum across `message_delta` events — take the last.**

**`message_stop`**:

```sse
event: message_stop
data: {"type": "message_stop"}
```

**`ping`**:

```sse
event: ping
data: {"type": "ping"}
```

**`error`** (mid-stream):

```sse
event: error
data: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
```

### 4.4 Canonical examples (from the docs)

Simple text:

```sse
event: message_start
data: {"type": "message_start", "message": {"id": "msg_1nZd...", "type": "message", "role": "assistant", "content": [], "model": "claude-opus-5", "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 25, "output_tokens": 1}}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}

event: ping
data: {"type": "ping"}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "!"}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence":null}, "usage": {"output_tokens": 15}}

event: message_stop
data: {"type": "message_stop"}
```

Tool use (note empty first `partial_json`):

```sse
event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01T1x1fJ34qAmk2tNTrN7Up6","name":"get_weather","input":{}}}
event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}
event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}}
...
event: content_block_stop
data: {"type":"content_block_stop","index":1}
event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":89}}
```

Thinking (display summarized):

```sse
event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}
event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "I need to find the GCD..."}}
event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "EqQBCgIYAhIMMb3LzNrMu..."}}
event: content_block_stop
data: {"type": "content_block_stop", "index": 0}
```

### 4.5 Fine-grained tool streaming

`eager_input_streaming: true` on a tool (or the legacy `fine-grained-tool-streaming-2025-05-14` beta) removes server-side buffering of tool input JSON so `input_json_delta` fragments flow token-by-token — at the cost of the API no longer validating the accumulated input. <https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming>

### 4.6 Stream error recovery

If the connection dies mid-stream: capture the partial assistant response and start a new request. Claude 4.5-and-earlier: put the partial text into a new assistant prefill message and continue. Claude 4.6+: send a user message like `Your previous response was interrupted and ended with [previous_response]. Continue from where you left off.` Tool-use and thinking blocks **cannot** be partially resumed — only text blocks. <https://platform.claude.com/docs/en/build-with-claude/streaming#error-recovery>

---

## 5. Error format

Ref: <https://platform.claude.com/docs/en/api/errors>.

### 5.1 Body shape

```json
{
  "type": "error",
  "error": {"type": "not_found_error", "message": "The requested resource could not be found."},
  "request_id": "req_011CSHoEeqs5C35K2UUqR7Fy"
}
```

- `type` is always `"error"` at top level; `error.type` is the machine-readable class, `error.message` human-readable (unstable — don't string-match).
- `request_id` mirrors the `request-id` response header.
- New error `type` values may be added over time (versioning policy).

### 5.2 HTTP status ↔ error type

| HTTP | `error.type` | Meaning / triggers |
|---|---|---|
| 400 | `invalid_request_error` | Format/content problems of the request; also used for other unlisted 4XX; also returned when usage hits an org/workspace spend limit (except Claude Code workspace limits, which can be 429). |
| 401 | `authentication_error` | API key malformed/revoked/expired; (AWS: bad SigV4). |
| 402 | `billing_error` | Billing/payment problem. |
| 403 | `permission_error` | Key lacks permission for the resource/workspace. |
| 404 | `not_found_error` | Unknown resource — also **unknown `model` strings** and bad endpoint paths. |
| 409 | `conflict_error` | State conflicts (concurrent modification, unique-violation). |
| 413 | `request_too_large` | Body exceeds the endpoint's byte limit (see §1.4). |
| 429 | `rate_limit_error` | RPM/ITPM/OTPM exceeded, monthly spend cap reached, or Claude-Code-workspace spend limit. Regular 429s carry `retry-after`; **spend-cap 429s have no `retry-after`** and keep failing until access resumes. |
| 500 | `api_error` | Internal Anthropic error. Retry w/ exponential backoff; include request ID in support tickets. |
| 504 | `timeout_error` | Processing timed out (long requests; prefer streaming/batches). |
| **529** | `overloaded_error` | **API temporarily overloaded (Anthropic-specific code, not an IANA code).** Sharp traffic ramps may instead yield 429 via acceleration limits; ramp gradually. |

### 5.3 Retry guidance

Official SDKs retry connection errors, rate limits, and 5xx **twice by default** with exponential backoff, honoring `retry-after`. A shim server should implement idempotent, back-off-aware behavior identically on the client-facing side.

### 5.4 Mid-stream errors

After a streaming request has returned `200 OK`, a failure surfaces as an SSE event, not a new HTTP status:

```sse
event: error
data: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
```

Typical mid-stream types: `overloaded_error`, `api_error`. The stream then ends (usually without `message_stop`). Clients should keep the partial content and offer resume (§4.6).

---

## 6. Tool use — end-to-end flow

Refs: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview>, </handle-tool-calls>, </parallel-tool-use>.

### 6.1 The client-tool loop

1. **Request** declares `tools` (+ optional `tool_choice`). See §2.5.
2. **Response** with `stop_reason: "tool_use"` contains one or more `tool_use` content blocks:
   ```json
   {
     "id": "msg_01Aq9w938a90dw8q",
     "type": "message", "role": "assistant", "model": "claude-opus-5",
     "stop_reason": "tool_use", "stop_sequence": null,
     "content": [
       {"type": "text", "text": "I'll check the current weather in San Francisco for you."},
       {"type": "tool_use", "id": "toolu_01A09q90qw90lq917835lq9", "name": "get_weather", "input": {"location": "San Francisco, CA", "unit": "celsius"}}
     ]
   }
   ```
3. **Client executes** each tool with `input` (validated vs `input_schema` only when `strict:true`; never assume shape otherwise).
4. **Continue conversation** by appending:
   - the assistant message *exactly as returned* (including `tool_use` blocks, and `thinking`/`redacted_thinking` blocks when thinking is on — unmodified, original order), then
   - **one** new `user` message whose `content` **begins with** one `tool_result` block **per `tool_use` block**:
   ```json
   {"role": "user", "content": [
     {"type": "tool_result", "tool_use_id": "toolu_01A09q90qw90lq917835lq9", "content": "15 degrees"}
   ]}
   ```
   `content` may be a string, an array of nested blocks (`text`, `image`, `document`, `search_result`; `tool_reference`/`browser_state` for toolsets), or omitted entirely (empty result). `"is_error": true` signals execution failure (write *instructive* error content — Claude retries bad calls 2–3 times before apologizing).
5. Repeat until a response with `stop_reason: "end_turn"` (or another terminal reason) arrives.

### 6.2 ID matching & placement rules (enforced by 400 errors)

- `tool_result.tool_use_id` must equal a `tool_use.id` from the **immediately preceding** assistant turn (`toolu_...` ids; server tools use `srvtoolu_...`).
- The `user` message must **immediately follow** the assistant `tool_use` message — **no messages in between**.
- Inside that user message, **all `tool_result` blocks come first**; any text/images come **after** all results. A text-before-result body is rejected.
- Every `tool_use` needs a result — even skipped calls (`{"type":"tool_result","tool_use_id":"...","is_error":true,"content":"Not executed: ..."}`).
- Result blocks for computer/browser toolset members must echo the same `toolset_name` and may contain only `text`/`image` (+ one `browser_state` for browser tools); batch-actions run **sequentially in order, stop at first failure**.
- Mixed tool turns: if the assistant turn also contains a `server_tool_use` whose result has not arrived yet, the continuation user message may contain **only** `tool_result` blocks and the request must carry the **same `tools` array**; otherwise 400 (e.g. "... but no `web_search` tool was provided"). The API runs the deferred server tool on resubmission.

### 6.3 Parallel tool use

Default behavior: a single response may contain **multiple `tool_use` blocks**. Any execution order is fine; return all results together in the one follow-up user message as above. Disable via `tool_choice.disable_parallel_tool_use: true` (auto: at most one call; any/tool: exactly one). <https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use>

### 6.4 Server tools (web_search / web_fetch / code_execution / …)

- The model emits `server_tool_use` blocks; Anthropic executes and **injects the result block into the same response** (e.g. `web_search_tool_result` with `encrypted_content` rows, or an error-code payload). No client round trip required.
- Round-trip **both** the `server_tool_use` and result blocks verbatim when continuing.
- `pause_turn` stop reason: server-side loop hit its iteration limit (default 10) — resend the assistant response as-is to continue the loop. Only server-tool flows produce `pause_turn`; client-tool waits always use `stop_reason: "tool_use"`. <https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons#pause-turn>
- Billing: counted in `usage.server_tool_use.{web_search_requests, web_fetch_requests}`; web search is priced per 1,000 searches plus tokens.
- An Anthropic-compatible local server should either implement these tools or **reject/passthrough** accordingly — but always preserve these foreign blocks on round-trip instead of dropping them.

### 6.5 Strict tool use / structured outputs

`"strict": true` on a custom tool → grammar-constrained sampling **guarantees** `tool_use.input` validates against `input_schema`, and names are always valid. Same pipeline backs `output_config.format: {type:"json_schema", schema}` for schema-constrained free-text responses. JSON Schema subset limits (no `uniqueItems`, `$ref` recursion limits, etc.) are in <https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-schema-limitations>.

---

## 7. Thinking (extended & adaptive)

Refs: <https://platform.claude.com/docs/en/build-with-claude/thinking>, <https://platform.claude.com/docs/en/build-with-claude/extended-thinking>, <https://platform.claude.com/docs/en/build-with-claude/thinking-tool-workflows>.

### 7.1 Modes and per-model availability

- **`thinking: {"type":"adaptive", "display"?}`** — Claude decides when/how deeply to think; interleaved thinking is automatic. On Claude Fable 5/Mythos 5/Mythos Preview/Opus 5/Sonnet 5 adaptive thinking is **always on** (and `display` defaults to `"omitted"` — pass `"summarized"` to see thinking text).
- **`thinking: {"type":"enabled", "budget_tokens":N, "display"?}`** — legacy *manual extended thinking*. Only mode for Claude 4.5-era and older thinking models; accepted (deprecated) on 4.6; **rejected (400) on 4.7+ / 5.x**.
- **`thinking: {"type":"disabled"}`** — off. Claude Haiku 4.5 does not support interleaved thinking and ignores the interleaved-beta header. On Claude Opus 5, `thinking: disabled` at `xhigh`/`max` effort → 400.

### 7.2 Budget rules (manual mode)

- `budget_tokens ≥ 1024` and `< max_tokens`; it's a **target**, not a cap (model may finish early); `max_tokens` stays the hard ceiling on total output for the turn (thinking + visible output).
- **Interleaved thinking** (reasoning between tool calls within one assistant turn): manual mode needs beta header `interleaved-thinking-2025-05-14` on Claude 4.5/4.1/4 (Opus/Sonnet; not Haiku 4.5, ignored). With interleaving, `budget_tokens` may exceed `max_tokens` (budget spans all thinking blocks).

### 7.3 Response & round-trip rules

- Thinking yields `thinking` blocks `{type:"thinking", thinking, signature}` **before** other blocks, possibly multiple (interleaved: between tool calls); some `redacted_thinking` blocks `{type:"redacted_thinking", data}` for safety-redacted portions.
- `signature`: opaque, encrypted, **verified** when echoed back — pass thinking blocks back **unmodified and in original order** (400 otherwise). Signatures are platform-portable (Anthropic API/Bedrock/Vertex) and significantly longer on Claude 4+.
- **Required:** in a tool-use continuation, include the thinking/redacted blocks of that assistant turn. **Recommended to always** pass all prior thinking blocks back; "keep-all" models (Opus 4.5+, Sonnet 4.6+, Fable 5, Mythos*) retain & bill them as input; strip-models (earlier Opus/Sonnet, all Haiku) auto-strip what they don't need. **Don't filter by `type=="thinking"` alone — include `redacted_thinking`.**
- **Switching models mid-conversation:** strip thinking/redacted blocks from earlier turns (other models silently ignore them but still bill input tokens).

### 7.4 Compatibility constraints

- No assistant **prefill** while thinking is on.
- Manual thinking (`type:"enabled"`) only allows `tool_choice: auto|none` — `any`/`tool` → 400. Adaptive thinking allows forced tool use.
- Sampling: newest models (Opus 5/Sonnet 5/Fable 5/Mythos*/Opus 4.7/4.8) reject **any** non-default `temperature`/`top_p`/`top_k` (400). Older models while thinking: no `temperature`/`top_k` changes; `top_p` restricted to `[0.95, 1]`.
- Billing/context: thinking tokens bill as output and count toward `max_tokens`/context window; preserved thinking from earlier turns bills as input on keep-all models. `usage.output_tokens_details.thinking_tokens` exposes the raw reasoning spend.
- Streaming: `thinking_delta` events then a single `signature_delta` right before `content_block_stop` (see §4.3); with `display:"omitted"` there are **no** thinking deltas — only the signature.

---

## 8. Vision & multimodal

Refs: <https://platform.claude.com/docs/en/build-with-claude/vision>, <https://platform.claude.com/docs/en/build-with-claude/pdf-support>.

### 8.1 Image blocks

```json
{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "<b64>"}}
{"type": "image", "source": {"type": "url", "url": "https://example.com/pic.png"}}
{"type": "image", "source": {"type": "file", "file_id": "file_011CNha8..."}}
```

- Media types: `image/jpeg`, `image/png`, `image/gif`, `image/webp` (animations unsupported; **first frame only**).
- Optional `transformations: {"oversized_image": "downsize"(default) | "error"}` — with `"error"` over-limit images are 400-rejected with dimensions named.

### 8.2 Image limits

| Limit | Value |
|---|---|
| Images per request (API) | **100** for 200k-context models; **600** otherwise (claude.ai: 20/message) |
| Max dimensions per image | **8000 × 8000 px** |
| Many-image rule | **>20 images in one request** ⇒ stricter per-image dimension cap on *every* image (incl. images in resent history and inside `tool_result`; on Bedrock/Vertex document/PDF blocks also count). Exceeding → `invalid_request_error` mentioning "many-image requests". Safe default: keep each image ≤ **2000 px** per side or ≤20 image/document blocks. |
| Max size per image | **10 MB base64-encoded** on the direct API (5 MB on Bedrock/Vertex); request-body cap of §1.4 may bite first. |
| Visual token cost | `⌈width/28⌉ × ⌈height/28⌉` visual tokens (28×28 patches), after model-tier downscaling. |
| Resolution tiers | **Standard** (all models ≤4.6): max long edge 1568 px, max 1568 visual tokens. **High-resolution** (Claude 4.7+ / 5.x): long edge 2576 px, max 4784 visual tokens (~3× cost). Oversized images are downscaled preserving aspect ratio (except computer/browser tool_result images, which are **rejected**, not scaled). |

### 8.3 Document (PDF) blocks

```json
{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "<b64>"}, "title": "My PDF", "citations": {"enabled": true}}
{"type": "document", "source": {"type": "url", "url": "https://example.com/doc.pdf"}}
{"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "plain text here"}}
{"type": "document", "source": {"type": "content", "content": [{"type":"text","text":"..."},{"type":"image","source": {...}}]}}
{"type": "document", "source": {"type": "file", "file_id": "file_..."}}
```

- Limits: **600 pages/request** (100 pages when the request's context window < 1M), within the 32 MB request cap. Dense PDFs can exhaust the token window first.
- Cost model: each PDF page is converted to an image **plus** extracted text; ~1,500–3,000 tokens/page of text + per-page image tokens (§8.2 formula).
- `citations: {"enabled": true}` enables per-block citations of the document; responses then cite via `page_location` (PDF), `char_location` (plain text), `content_block_location` (content documents), `search_result_location`, or `web_search_result_location` in `text` block `citations` arrays.
- Binary Office formats (.xlsx/.docx) unsupported — convert to text/PDF first.

---

## 9. Prompt caching

Ref: <https://platform.claude.com/docs/en/build-with-claude/prompt-caching>.

### 9.1 `cache_control` breakpoints

- Block-level: add `"cache_control": {"type": "ephemeral", "ttl": "5m"|"1h"}` (default `5m`) to blocks in `tools[]`, `system[]`, or `messages[].content[]` (images & documents included).
- **At most 4 breakpoints per request.** Writes happen **only at breakpoints**; reads search backward from each breakpoint for a prefix-hash hit.
- Cacheable region: the ordered prefix `tools` → `system` → `messages`, up to and including the marked block. Changes at an earlier level **invalidate that level and everything after** (tools edits bust system+messages; system edits bust messages; `tool_choice` and images bust only the message-level cache; thinking/effort configuration is rendered into the prompt and always busts at least message-level).
- **20-block lookback:** reads examine at most 20 block positions per breakpoint (breakpoint included); longer-growing conversations need additional breakpoints.
- Concurrency: a cache entry becomes available only **after the first response begins** — fire the first request, wait, then fan out.
- Automatic caching: top-level `"cache_control": {"type":"ephemeral"}` puts the breakpoint on the last cacheable block and moves it forward as the conversation grows. It consumes one of the 4 slots; conflicts (explicit different-TTL marker on the same last block, or 4 existing breakpoints) → 400. Not supported on legacy-Bedrock.
- Pre-warming: `max_tokens: 0` writes cache at the breakpoint without generating (empty `content`, `stop_reason:"max_tokens"`, usage populated). Use an explicit breakpoint on the shared prefix (system/tools), mirroring the thinking/effort config of follow-up requests.

### 9.2 Minimum cacheable prompt length (per request prefix at the breakpoint)

| Model(s) | Min tokens |
|---|---|
| Claude Opus 5, Fable 5, Mythos 5 | 512 |
| Claude Opus 4.8, Sonnet 5, Sonnet 4.6, Sonnet 4.5, Opus 4.1, Opus 4, Sonnet 4 | 1024 |
| Claude Mythos Preview, Opus 4.7 | 2048 |
| Claude Haiku 3.5 | 2048 |
| Claude Opus 4.6, Opus 4.5, Haiku 4.5 | 4096 |

Sub-minimum prefixes are processed **uncached, silently (no error)** — detect via zero `cache_creation_input_tokens` + `cache_read_input_tokens`.

### 9.3 What can/cannot be cached

Cacheable: tool definitions, system blocks, user/assistant text, images, documents, tool_use/tool_result blocks. **Not directly cacheable:** thinking blocks (they *are* cached as part of assistant turn content around tool results — and count as input when cache-read), sub-content (e.g. citations) on their own, and **empty text blocks**.

### 9.4 TTL and pricing structure

- TTLs: `5m` (default; refreshed on hit) and `1h` (extended, higher write price; previously beta `extended-cache-ttl-2025-04-11`). Mixing allowed with the constraint: **1h breakpoints must precede 5m ones**; billing segments `A` (highest cache hit) → `B` (highest 1h block after A) → `C` (last cache_control block): charge cache-read for A, 1h-write for B−A, 5m-write for C−B.
- Cache reads cost ~10% of base input price; 5m writes 1.25×; 1h writes 2×. `usage.cache_creation.ephemeral_5m_input_tokens` / `.ephemeral_1h_input_tokens` break down writes by TTL.

### 9.5 Usage accounting (restated because shims get this wrong)

`input_tokens` = tokens **after the last breakpoint** only. **Total processed input = input_tokens + cache_creation_input_tokens + cache_read_input_tokens.** All three count toward the context window and rate limits (cache reads count 0.1× against Priority Tier drawdown).

---

## 10. Anthropic ↔ OpenAI chat-completions translation spec

This is the precise field-level mapping used by every OpenAI-compat front-end (`api.openai.com/v1/chat/completions` ⇄ `/v1/messages`). "Ant" = Anthropic native, "OAI" = OpenAI chat completions.

### 10.1 Request mapping (OAI → Ant)

| OpenAI request field | → Anthropic field | Notes / approximations |
|---|---|---|
| `model` | `model` | Direct string copy; map org-specific aliases. |
| `messages[]` | `messages[]` + top-level `system` | **Split out `role:"system"` and `role:"developer"` messages** into the top-level `system` param. Concatenate multiple system/developer texts with `"\n"` — Anthropic supports only (a) one leading system prompt and (b) newest-model mid-conversation system messages. If order matters and you'll cache, hoist as one block or synthesize blocks. |
| `messages[].role: user/assistant` | same roles | Consecutive same-role messages may be merged by the server, but keeping a clean alternation preserves cache behavior. |
| `messages[].role: tool` (`{tool_call_id, content}`) | Merge consecutive tool results of one turn into **one `user` message** whose content is `[tool_result{tool_use_id=tool_call_id, content}, ...]` blocks **first**, other content after | OAI had one message per tool call; Anthropic wants one user message with all results of the turn. |
| `messages[].role: function` (legacy) | same as `tool` | Same treatment. |
| assistant `tool_calls[] = [{id, function:{name, arguments}}]` | assistant `content[]` text blocks + `tool_use` blocks `{id, name, input: JSON.parse(arguments)}` | Parse `arguments` string into the `input` object; malformed → either best-effort parse or reject. Keep block order (text before/after tool calls as present). |
| assistant `content: null` w/ tool_calls | no text block | Legal. |
| user content parts: `{type:"text",text}` | `{"type":"text","text":...}` | 1:1. |
| user parts: `{type:"image_url", image_url:{url}}` | `image` block | `data:image/...;base64,xxx` URLs → base64 `source`; `http(s)` URLs → `{"type":"url","url":...}` (or download+inline). OAI `detail` field (low/high/auto) has **no Anthropic equivalent** — drop. |
| user parts: `{type:"file"}`, `input_audio` | — | No base equivalent (drop or 400). Anthropic uses `document` blocks outside the OAI surface. |
| `max_tokens` / `max_completion_tokens` | `max_tokens` | **Anthropic REQUIRES it.** If the client omitted it, synthesize a value (common choices: model's max, or a generous default like the remaining context or 4096/8192). This is the single biggest source of OAI-client breakage. |
| `stop` (string \| string[≤4]) | `stop_sequences` (string[]) | 1:1; note Anthropic's own OAI-shim says only *non-whitespace* stop sequences work. |
| `stream` | `stream` | 1:1. |
| `stream_options.include_usage` | (nothing) | Anthropic **always** includes usage (message_start carries input; message_delta carries cumulative output). Honor the flag when translating *back* to OAI chunks. |
| `temperature` (0–2) | `temperature` (0–1) | **Clamp >1 to 1.0** or 400. (Anthropic's own shim: values >1 capped at 1; newest models reject non-default values entirely — pass through and surface the 400.) |
| `top_p` | `top_p` | 1:1 (0–1). |
| — | `top_k` | Anthropic-only; expose via `extra_body`. |
| `seed` | — | Ignored (Anthropic never deterministic). |
| `n` | — | Anthropic always `n=1`; reject `n>1` (or 400). |
| `presence_penalty`, `frequency_penalty` | — | No equivalent. Ignore (silently, like Anthropic's own shim) or approximate via prompts; do not 400. |
| `logit_bias` | — | Ignore. |
| `logprobs`, `top_logprobs` | — | No Anthropic equivalent at all. Ignore / reject per policy. |
| `user` | `metadata.user_id` | Opaque id semantics align (uuid/hash). |
| `metadata` | — | Ignore (Anthropic `metadata` only has `user_id`). |
| `response_format: {type:"text"}` | — | default. |
| `response_format: {type:"json_object"}` | approx | Best practice: add an assistant prefill `{"role":"assistant","content":"{"}` (and force-arg "{"), or a tool you demand via `tool_choice`, or `output_config.format`. Not equivalent. |
| `response_format: {type:"json_schema",..."strict":true}` | `output_config: {"format": {"type":"json_schema", "schema": ...}}` (native structured outputs) | Full equivalent on supported models. |
| `tools = [{type:"function", function:{name,description,parameters,strict}}]` | `tools = [{name, description, input_schema: parameters, strict}]` | 1:1 field renames; OAI `strict:true` ≈ Ant `strict:true`. Legacy `functions[]` array maps identically. |
| `tool_choice: "none"` | `{"type":"none"}` | |
| `tool_choice: "auto"` | `{"type":"auto"}` | |
| `tool_choice: "required"` | `{"type":"any"}` | |
| `tool_choice: {"type":"function","function":{"name":"f"}}` | `{"type":"tool","name":"f"}` | |
| `parallel_tool_calls: false` | `disable_parallel_tool_use: true` on the `tool_choice` object | `parallel_tool_calls:true`/absent → `false`/omit. |
| `reasoning_effort` (o-series) | `thinking`/`output_config.effort` | Not 1:1. Approx: `output_config.effort: low|medium|high`; thinking via `extra_body.thinking`. Anthropic's shim ignores `reasoning_effort`. |
| `service_tier` (OAI) | `service_tier: "auto"\|"standard_only"` (Ant semantics differ) | Anthropic's shim ignores OAI `service_tier`. |
| `store`, `prediction`, `audio`, `modalities` | — | Ignore. |

**Anthropic-only request fields with no OAI source**: `system` block arrays w/ `cache_control`, `top_k`, `thinking`, `container`, `inference_geo`, `cache_control` top-level, `mcp_servers`, `context_management`, document/search_result blocks. Typical shims let clients pass them through `extra_body`.

### 10.2 Response mapping (Ant → OAI)

| Anthropic response | → OpenAI `chat.completion` | Notes |
|---|---|---|
| `id: "msg_..."` | `id` (any string; many shims emit `chatcmpl-...`) | Opaque. |
| `type:"message"` | `object:"chat.completion"` | |
| — | `created` (unix seconds) | Synthesize (now). |
| `model` | `model` | Echo the requested string if you want OAI-client fidelity; Anthropic echoes the resolved model ID. |
| `role:"assistant"` | `choices[0].message.role:"assistant"` | |
| `content[] text blocks` | `choices[0].message.content` (single concatenated string; `null` if none) | Join all text blocks. |
| `content[] tool_use blocks` | `choices[0].message.tool_calls = [{index?, id, type:"function", function:{name, arguments: JSON.stringify(input)}}]` | Serialize `input` object to a JSON string. |
| `thinking` / `redacted_thinking` blocks | No OAI standard. Common practice: non-standard `delta/message.reasoning_content` (DeepSeek-R1 convention) or drop from message and surface via extension field; redacted blocks usually dropped. | Anthropic's own OAI shim does not return thinking text. |
| `stop_reason` | `choices[0].finish_reason` | `end_turn→"stop"` · `stop_sequence→"stop"` · `max_tokens→"length"` · `model_context_window_exceeded→"length"` · `tool_use→"tool_calls"` · `refusal→"content_filter"` (alternatively `stop` + `message.refusal` text; Anthropic's shim leaves `refusal` empty) · `pause_turn→` no OAI concept; usually map to `"stop"` (or `"tool_calls"` when blocks are pending) and let the client re-POST the conversation. |
| `stop_sequence` | (nothing) | OAI doesn't identify which stop fired. |
| `usage.input_tokens` (post-breakpoint only!) | `usage.prompt_tokens` | **Must add the cache fields**: `prompt_tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens`. Optionally set `prompt_tokens_details.cached_tokens = cache_read_input_tokens`. |
| `usage.output_tokens` | `usage.completion_tokens` | Thinking included. Optionally `completion_tokens_details.reasoning_tokens = output_tokens_details.thinking_tokens`. |
| sum | `usage.total_tokens` | prompt + completion. |
| `container`, `stop_details`, `inference_geo`, `server_tool_use`, `service_tier` | — | No OAI equivalent; drop or mirror as extension fields. |

### 10.3 Streaming mapping (Ant SSE → OAI `chat.completion.chunk`)

| Anthropic event | → OpenAI chunk(s) |
|---|---|
| `message_start` | First chunk: `{choices:[{index:0, delta:{role:"assistant", content:""}, finish_reason:null}]}` (+ optionally capture usage). |
| `content_block_start` (text) | no output (or empty-role delta already sent) |
| `content_block_delta` `text_delta` | `delta: {content: text}` |
| `content_block_start` (`tool_use`) | `delta: {tool_calls: [{index: <n>, id, type:"function", function:{name, arguments:""}}]}` (allocate your own OAI tool_calls index per block) |
| `content_block_delta` `input_json_delta` | `delta: {tool_calls: [{index: <n>, function: {arguments: partial_json}}]}` — **this maps cleanly**: both sides stream partial JSON *strings*; concatenate `partial_json` fragments = OAI client-side `arguments` accumulation. |
| `content_block_delta` `thinking_delta` | Non-standard `delta:{reasoning_content: thinking}` (widely compatible w/ R1-style clients) or drop. |
| `content_block_delta` `signature_delta` | nothing (or extension field). |
| `content_block_stop` | nothing (just advance your block→tool_calls index map). |
| `content_block_start` of server-tool result blocks | nothing / passthrough extension (they're provider-internal). |
| `message_delta` | Final chunk: `choices:[{index:0, delta:{}, finish_reason: mapped(stop_reason)}]`; usage chunk (`{usage:{...}, choices:[]}`) honoring the client's `stream_options.include_usage`. **Only send the usage chunk at the end**; don't forward every cumulative message_delta usage. |
| `message_stop` | `data: [DONE]` |
| `ping` | nothing (or SSE comment `: ping`). |
| `error` | OAI-style error object in-band, then close (`{"error":{"message":...,"type":...}}`); clients tolerate. |

**Structural caveat (no 1:1):** Anthropic interleaves *multiple* text blocks and tool blocks at distinct `content[]` indices; OAI chat completions have one `content` string + flat `tool_calls` per choice. Concatenation + index bookkeeping is the standard approximation (map each `content_block_start.index` → `tool_calls.index` on first sight).

**Other impossible 1:1s:** logprobs, multiple `n` completions, deterministic seeds, penalties, `logit_bias`, Anthropic prompt-cache TTLs (approximate with `prompt_cache_key` on the OAI side only if your backend exposes it), `pause_turn`-style resumability, Anthropic prefill continuations (OAI has no prefill concept; OAI-side shims fake it via `messages[].assistant` partial + `add_generation_prompt` semantics — underscores why Anthropic-native serves better for prefill workflows).

---

## 11. Anthropic's own OpenAI-SDK compatibility shim (reference implementation)

Docs: <https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk>. Anthropic itself exposes an OpenAI-compatible surface on the direct API so the stock OpenAI SDK works by changing three things:

```python
client = OpenAI(api_key=os.environ["ANTHROPIC_API_KEY"], base_url="https://api.anthropic.com/v1/")
client.chat.completions.create(model="claude-opus-5", messages=[...])
```

Behavior of this reference shim (useful as an oracle when implementing your own):

- **System/developer messages are hoisted & concatenated** (`"\n"`-joined) into one leading system prompt.
- **Supported fully:** `model`, `max_tokens`, `max_completion_tokens`, `stream`, `stream_options`, `top_p`, `parallel_tool_calls`, `temperature` (capped at 1), `stop` (non-whitespace sequences), `tools[n].function.{name,description,parameters}`, legacy `functions`, `tool_calls`, `function_call`, tool-role `{content, tool_call_id}`, `image_url.url`.
- **Rejected/limited:** `n` must be exactly 1; prompt caching **not available** through this surface; audio input stripped; response_format ignored (use Anthropic structured outputs); `strict` on tools **ignored** (no schema guarantee).
- **Silently ignored:** `logprobs`, `top_logprobs`, `metadata`, `prediction`, `presence_penalty`, `frequency_penalty`, `seed`, `service_tier`, `audio`, `logit_bias`, `store`, `user`, `modalities`, `reasoning_effort`, per-message `name`, assistant `refusal`/`audio`, tool/function `name`, `image_url.detail`.
- **Response shape:** standard chat.completion with `choices` length 1; `usage.{prompt_tokens, completion_tokens, total_tokens}` populated; `completion_tokens_details`, `prompt_tokens_details`, `choices[].message.refusal`, `.audio`, `logprobs`, `service_tier`, `system_fingerprint` always empty.
- **Headers:** responds with `openai-version: 2020-10-01`, `openai-processing-ms` empty, plus the usual `request-id`, `retry-after`, and `x-ratelimit-*`-style compatibility headers; auth via `x-api-key` or `authorization`.
- **Errors** are emitted in OpenAI error format; message text differs — log only, don't parse.
- **Thinking** can be turned on via `extra_body: {"thinking": {"type":"enabled","budget_tokens":N}}`; process reasoning is **not** returned over the OAI surface.
- Rate limits are the same ones as `/v1/messages`.
- Anthropic positions the shim for **testing/comparison, not production**; native API recommended for full features (PDFs, citations, prompt caching, thinking visibility).

**Implementation takeaway for sglang-style servers:** Anthropic's shim choice of *silently ignoring* unsupported fields (rather than erroring) is the dominant client-friendliness pattern; mirror it, but log dropped fields.

---

## 12. Everything else an implementer must not miss

### 12.1 Token counting endpoint — `POST /v1/messages/count_tokens`

<https://platform.claude.com/docs/en/api/messages/count_tokens>, <https://platform.claude.com/docs/en/build-with-claude/token-counting>

- Accepts the **same body minus output-side params**: `model` (required), `messages` (required), `system`, `tools`, `tool_choice`, `thinking`; **no** `max_tokens`, `stream`, sampling needed.
- Response:
  ```json
  {"input_tokens": 2095}
  ```
  (`MessageTokensCount` — input tokens across messages + system + tools.)
- **Free** but rate-limited per usage tier. 32 MB body limit.
- Counts are model-specific: **Claude 4.7+ and Mythos/Fable use a newer tokenizer (~30% more tokens for identical text)** — count with the target `model`, don't reuse older counts.
- Anthropic-side counts include server-side system-prompt transformations (tools rendering, etc.), so a local tokenizer will underestimate by overhead tokens; treat local counts as approximations.

### 12.2 Models API — `GET /v1/models`, `GET /v1/models/{id}`

<https://platform.claude.com/docs/en/api/models/list>

- Query params: `limit` (default 20, 1–1000), `after_id`, `before_id` (old-style cursor pagination, not `page`). Same auth/version headers.
- Response list:
  ```json
  {
    "data": [{
      "type": "model", "id": "claude-opus-5", "display_name": "Claude Opus 5",
      "created_at": "2026-07-24T00:00:00Z",
      "max_input_tokens": 1000000, "max_tokens": 128000,
      "capabilities": {
        "batch": {"supported": true}, "citations": {"supported": true},
        "code_execution": {"supported": true},
        "context_management": {"supported": true, "clear_thinking_20251015": {"supported": true}, "clear_tool_uses_20250919": {"supported": true}, "compact_20260112": {"supported": true}},
        "effort": {"supported": true, "high": {"supported": true}, "low": {"supported": true}, "max": {"supported": true}, "medium": {"supported": true}, "xhigh": {"supported": true}},
        "image_input": {"supported": true}, "pdf_input": {"supported": true},
        "structured_outputs": {"supported": true},
        "thinking": {"supported": true, "types": {"adaptive": {"supported": true}, "enabled": {"supported": true}}}
      }
    }],
    "first_id": "...", "last_id": "...", "has_more": true
  }
  ```
- `max_input_tokens` / `max_tokens` per model is the programmatic source for §12.4 numbers. Newer models listed first. Deprecation/retirement statuses live in <https://platform.claude.com/docs/en/about-claude/model-deprecations>.

### 12.3 Message Batches API

<https://platform.claude.com/docs/en/api/messages/batches/create>, </batch-processing>

- **Create:** `POST /v1/messages/batches` with
  ```json
  {"requests": [
    {"custom_id": "my-custom-id-1", "params": {"model": "...", "max_tokens": 1024, "messages": [...]}},
    {"custom_id": "my-custom-id-2", "params": {...}}
  ]}
  ```
  `params` is a **complete non-streaming `/v1/messages` request body**. Limits: ≤ **100,000 requests** or **256 MB**, whichever first. Older `message-batches-2024-09-24` beta header now GA (no header needed).
- **Batch object:**
  ```json
  {
    "id": "msgbatch_013Zva2CMHLNnXjNJJKqJ2EF", "type": "message_batch",
    "processing_status": "in_progress | canceling | ended",
    "request_counts": {"processing": 100, "succeeded": 50, "errored": 30, "canceled": 10, "expired": 10},
    "ended_at": "...", "created_at": "...", "expires_at": "...", "cancel_initiated_at": null, "archived_at": null,
    "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_.../results"
  }
  ```
- Processing: most batches < 1 h; **max 24 h** then unfinished requests `expire`. 50% price discount; `usage.service_tier: "batch"`.
- **Results:** `GET /v1/messages/batches/{id}/results` → `.jsonl`, **order not guaranteed**, keyed by `custom_id`, one line per request:
  ```jsonl
  {"custom_id":"my-first-request","result":{"type":"succeeded","message":{...full Message...}}}
  {"custom_id":"my-second-request","result":{"type":"errored","error":{"type":"error","error":{"type":"invalid_request_error","message":"..."},"request_id":"req_..."}}}
  {"custom_id":"my-third-request","result":{"type":"canceled"}}
  {"custom_id":"my-fourth-request","result":{"type":"expired"}}
  ```
- Other ops: list (`GET /v1/messages/batches` w/ `after_id`/`before_id`/`limit`), retrieve (`GET /v1/messages/batches/{id}`), cancel (`POST .../cancel`), archive (`DELETE /v1/messages/batches/{id}`).
- Server tools (web search) work in batches at standard prices; extended output `output-300k-2026-03-24` beta (up to 300k tokens) exists *only* on batches for Opus 5/4.8/4.7/4.6, Sonnet 5/4.6.

### 12.4 Context windows & max output per model

Sources: <https://platform.claude.com/docs/en/build-with-claude/context-windows>, <https://platform.claude.com/docs/en/about-claude/models/overview>.

| Generation | Context | Max output (`max_tokens` ceiling) |
|---|---|---|
| Claude Opus 5, Sonnet 5, Fable 5, Mythos 5/Preview, Opus 4.8/4.7/4.6, Sonnet 4.6 | **1M tokens** (default, no beta header) | **128K** (batches + `output-300k-2026-03-24` → 300K; Claude 3.7-era `output-128k-2025-02-19` history) |
| Claude Sonnet 4.5, Opus 4.5, Haiku 4.5 | 200K | 64K |
| Claude Sonnet 4 | 200K (1M was gated by `context-1m-2025-08-07` beta) | 64K |
| Claude Opus 4 / 4.1 | 200K | 32K |
| Claude 3.7 Sonnet | 200K | 64K (128K with `output-128k-2025-02-19`) |
| Claude 3.5 Sonnet / 3.5 Haiku | 200K | 8192 |
| Claude 3 Opus / Sonnet / Haiku | 200K | 4096 |

- Via the Models API these are `max_input_tokens` / `max_tokens` fields — prefer reading them programmatically.
- **Overflow behavior:** input alone too long → 400 `invalid_request_error` ("prompt is too long"). On Claude 4.5+, `input + max_tokens` overflow is accepted, generation stops with `stop_reason: "model_context_window_exceeded"`; on older models the API 400s unless you opt in via `model-context-window-exceeded-2025-08-26`.
- Context-awareness: Sonnet 5/4.6/4.5/Haiku 4.5 get automatic injected `<budget:token_budget>` + `<system_warning>` tags; newer Opus/Fable/Mythos use task budgets (beta) instead. Servers that emulate the API contract don't need to emulate the tags, but requests round-tripping through such models must pass them through if present.

### 12.5 Context management / context editing (beta)

`context-management-2025-06-27` enables server-side `context_management` strategies — `clear_tool_uses_20250919` (clear old tool results), `clear_thinking_20251015` (override per-model thinking preservation), `compact_20260112` (server compaction). Also: `mid-conversation-tool-changes-2026-07-01` (§2.2.1). Implementations may safely 400/ignore unsupported strategies; pass through the header field names if proxying.

### 12.6 Effort parameter

`output_config.effort: "low"|"medium"|"high"|"xhigh"|"max"` (default `high`≡unset), GA on Opus 4.5/4.6 and 4.7/4.8/5, Sonnet 4.6/5, Fable/Mythos 5-class models; controls *all* output token spend incl. tool calls & thinking. Newest-model rule: `thinking:"disabled"` at `xhigh`/`max` on Claude Opus 5 → 400. <https://platform.claude.com/docs/en/build-with-claude/effort>

### 12.7 Rate limits (operational)

- Measured per model class in **RPM / input-TPM / output-TPM**; 429 with `retry-after` on breach (spend-cap 429 has none).
- Transport: `anthropic-ratelimit-*` headers (§1.3) give live windows; implement these headers in shims for compatibility with Anthropic-aware clients.

### 12.8 Other contract details worth testing

- **100,000 messages/request cap** (413/400-style rejection beyond).
- Body must be UTF-8 JSON; unknown/extra request fields → typically 400 `invalid_request_error` on the native API (Anthropic validates strictly; unlike its OpenAI shim, which ignores unknowns). Be explicit about which behavior your shim implements.
- `usage.output_tokens` is non-zero even for empty strings; `stop_reason` non-null for non-streaming.
- Assistant-prefill content cannot end with trailing whitespace (400).
- A response `content` may be `[]` (empty) in edge cases (`end_turn` after tool results; `max_tokens: 0` prewarms). Handle gracefully.
- `stop_sequences` matched ⇒ that exact string returned in `stop_sequence`.
- Web-search-side cached-token surprise: ephemeral-5m writes can appear under server-tool flows due to automatic server-tool-result caching (`5m` writes you didn't explicitly mark — documented behavior).
- For streaming clients: **never** rely on receiving `message_stop`; always handle truncation/timeouts (error recovery §4.6).
- For model aliasing: pre-4.6 aliases (`claude-sonnet-4-5`) resolve to dated snapshots; 4.6+ IDs are already pinned snapshots (e.g. `claude-sonnet-4-6`).

---

## Appendix A. Beta header registry (as documented)

| Header | Feature |
|---|---|
| `prompt-caching-2024-07-31` | prompt caching (now GA) |
| `pdfs-2024-09-25` | PDF support (now GA) |
| `message-batches-2024-09-24` | batches (now GA) |
| `token-counting-2024-11-01` | count_tokens (now GA) |
| `computer-use-2024-10-22`, `computer-use-2025-01-24` | computer use versions |
| `token-efficient-tools-2025-02-19` | token-efficient tool calling |
| `output-128k-2025-02-19` | 128k outputs on Sonnet 3.7 |
| `fine-grained-tool-streaming-2025-05-14` | global fine-grained tool streaming (superseded by per-tool `eager_input_streaming`) |
| `interleaved-thinking-2025-05-14` | interleaved thinking on manual-mode 4.x models (automatic w/ adaptive) |
| `dev-full-thinking-2025-05-14` | full (unsummarized) thinking for dev |
| `files-api-2025-04-14` | Files API |
| `mcp-client-2025-04-04`, `mcp-client-2025-11-20` | MCP connector |
| `code-execution-2025-05-22` | code execution tool |
| `extended-cache-ttl-2025-04-11` | 1h cache TTL (now GA) |
| `context-1m-2025-08-07` | 1M context on Sonnet-4.x (1M is default on 4.6+/5.x) |
| `context-management-2025-06-27` | context editing strategies |
| `model-context-window-exceeded-2025-08-26` | graceful context-overflow stop on pre-4.5 models |
| `skills-2025-10-02` | Agent Skills / container skills |
| `fast-mode-2026-02-01` | `speed:"fast"` (Opus 5 / Opus 4.8 research preview) |
| `output-300k-2026-03-24` | 300k outputs on batches (128k-capable models) |
| `mid-conversation-tool-changes-2026-07-01` | mid-conversation tool add/remove |
| `mcp-tunnels-2026-06-22`, `managed-agents-2026-04-01`, `agent-memory-2026-07-22`, `user-profiles-2026-03-24` / `-2026-08-18` | endpoint-scoped agent-platform betas |

## Appendix B. Minimal conformance checklist for an Anthropic-compatible server

1. `POST /v1/messages`: accept and validate the full request schema; honor required `max_tokens`, and 400 cleanly on protocol violations (role/structure/tool pairing/thinking tampering).
2. Emit canonical response §3 with stop_reason mapping and full usage accounting (`input_tokens` post-breakpoint vs cache fields).
3. SSE §4 event order, cumulative usage, pings, error events; no `[DONE]`.
4. Error envelope §5 incl. `request_id` + HTTP codes 400/401/403/404/413/429/500/529 (+402/409/504 as applicable).
5. `POST /v1/messages/count_tokens` → `{"input_tokens": N}`.
6. `GET /v1/models` (+ `{id}`) with `data/first_id/last_id/has_more` and `max_input_tokens`/`max_tokens` per model.
7. Header contract: require `anthropic-version`, accept `x-api-key`/`Authorization`, tolerate `anthropic-beta`, echo `request-id` (+ rate-limit headers if you have them).
8. Behave under versioning policy: unknown block/event/enum extensions must not crash clients — and your *client-facing* emitter should keep unknown round-tripped blocks intact.
9. If exposing OpenAI-compat too, mirror §10 mappings, honoring Anthropic's own shim behavior (§11) for field support and silent-ignore decisions.

