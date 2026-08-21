# sglang × vLLM — API-serving architecture notes: Anthropic Messages API (`/v1/messages`)

Date: 2026-09. sglang checkout: `/home/ec2-user/sglang`, branch `main` @ `d90ef6980`.
vLLM research: `vllm-project/vllm` `main` (fetched from GitHub raw/API, Sept 2026 state).

> **Headline finding:** sglang **already ships** a complete Anthropic Messages API
> implementation on `main` — `POST /v1/messages` and `POST /v1/messages/count_tokens`
> are registered on every server with no flag. The question "where/how would one add
> it" is therefore answered by *this tree itself*, and Part C below is written both
> as (i) the canonical blueprint that sglang actually follows, and (ii) a concrete
> gap-closure plan diffed against vLLM's parallel implementation.

---

## Part A — sglang architecture map (authoritative, file:line referenced)

### A.1 Process & serving-layer layout

```
sglang serve
└── python/sglang/srt/entrypoints/http_server.py        FastAPI app, all HTTP routes
    ├── _GlobalState { tokenizer_manager, template_manager, scheduler_info }
    │     (http_server.py:195-213, set in lifespan @ http_server.py:269-339)
    ├── fast_api_app.state.openai_serving_chat  = OpenAIServingChat(tm, tpl)   (L306-310)
    ├── fast_api_app.state.openai_serving_{completion,embedding,classify,score,
    │      rerank,tokenize,detokenize,transcription,responses}               (L303-367)
    ├── fast_api_app.state.ollama_serving                                   (L334)
    └── fast_api_app.state.anthropic_serving = AnthropicServing(openai_serving_chat)  (L337-339)
```

Engine startup spawns the scheduler and detokenizer as separate OS processes
(`sglang/srt/entrypoints/engine.py`: `init_tokenizer_manager`,
`run_scheduler_process`, `run_detokenizer_process`, imported at http_server.py:74-79).
The HTTP process holds a `TokenizerManager`
(`sglang/srt/managers/tokenizer_manager.py:386`), which tokenizes and forwards
`GenerateReqInput` objects to the scheduler over ZMQ/IPC and streams results back.

### A.2 Request flow: HTTP → TokenizerManager → Scheduler

1. **Route** (e.g. http_server.py:1722 `@app.post("/v1/chat/completions")`) →
   pydantic-validated `ChatCompletionRequest` + `raw_request: Request`.
2. **Serving handler** `OpenAIServingBase.handle_request`
   (`entrypoints/openai/serving_base.py:73-133`):
   `_validate_request` → `_convert_to_internal_request` → dispatch to
   `_handle_streaming_request` / `_handle_non_streaming_request` based on `request.stream`;
   uniform exception → `create_error_response` mapping (ValueError→400, Exception→500).
3. **`OpenAIServingChat._convert_to_internal_request`** (`serving_chat.py:930-1061`):
   `_process_messages` (chat-template rendering, tool constraint construction —
   `serving_chat.py:1063-1167`) → `request.to_sampling_params(...)`
   (`protocol.py:1072`) → builds `GenerateReqInput` with `text`/`input_ids`,
   `image_data`, `audio_data`, `video_data`, `sampling_params`, `rid`, etc.
4. **`TokenizerManager.generate_request`** (`tokenizer_manager.py:765`):
   normalizes, tokenizes (`_tokenize_one_request`), `_send_one_request` to the
   scheduler over ZMQ, then `async for response in self._wait_one_response(...)`
   yields per-request outputs (text deltas + `meta_info`) back to the serving layer.
5. **Non-streaming**: `serving_chat.py:1772-1795` takes the first item and builds a
   `ChatCompletionResponse` via `_build_chat_response` (`serving_chat.py:1797`;
   usage via `UsageProcessor.calculate_response_usage` @ L1957).
6. **Streaming**: `serving_chat.py:1478-1504` kick-starts the generator (so a
   pre-first-chunk ValueError still returns HTTP 400), wraps in
   `StreamingResponse(..., media_type="text/event-stream",
   background=tokenizer_manager.create_abort_task(adapted_request))`
   (abort task: `tokenizer_manager.py:2113` — aborts the engine request when the
   HTTP client disconnects).

`MessageProcessingResult` (prompt, prompt_ids, image/audio/video data, stop,
tool_call_constraint, require_reasoning…) is defined in `protocol.py:2016`.

### A.3 Route inventory (http_server.py)

| Surface | Route(s) | Lines |
|---|---|---|
| OpenAI completions | `POST /v1/completions` | 1714 |
| OpenAI chat | `POST /v1/chat/completions` | 1722 |
| OpenAI embeddings | `POST /v1/embeddings` | 1732 |
| classify / tokenize(+detokenize) | `/v1/classify`, `/v1/tokenize`, … | 1744-1773 |
| audio transcriptions | `POST /v1/audio/transcriptions` | 1792 |
| models | `GET /v1/models`, `GET /v1/models/{model}` | 1843, 1875 |
| score / rerank | `POST /v1/score`, `/v1/rerank` | 1900 |
| **OpenAI Responses API** | `POST /v1/responses`, `GET /v1/responses/{id}`, `POST /v1/responses/{id}/cancel` | 1908-1941 |
| Ollama | `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, root-only routes | 1958-1996 |
| **Anthropic** | `POST /v1/messages`, `POST /v1/messages/count_tokens` | **2002, 2012** |
| SageMaker | `/ping`, `/invocations` | 2023-2036 |
| Vertex AI | `AIP_PREDICT_ROUTE` (`/vertex_generate` default) | 2040 |

Error-envelope dispatch by path prefix happens in the two exception handlers:
`validation_exception_handler(HTTPException)` (http_server.py:532-585) and
`validation_exception_handler(RequestValidationError)` (L589-620) — both have a
`/v1/messages` branch producing the Anthropic `{"type":"error","error":{...}}`
shape (`_anthropic_error_response`, L521-529; `_anthropic_validation_message`,
L496-518, which scrubs file paths from pydantic errors; 422 is re-mapped to 400).

### A.4 `protocol.py` (OpenAI dialect)

`entrypoints/openai/protocol.py` (2115 lines) is a flat module of pydantic models:
infra (`ErrorResponse` L96, `UsageInfo` L205, `PromptTokensDetails` L185,
`StreamOptions` L214), completion DTOs (L328-530), chat content parts
(`ChatCompletionMessageContentTextPart` L531, `...ImagePart` L575, video/audio
parts L565-636, `...ToolReferenceBlock` L637), tool DTOs (`Function` L742,
`Tool` L759 — note it carries a sglang-specific `defer_loading` field used by
Anthropic tool_search flows, `ToolChoice` L784),
`ChatCompletionRequest` L823 (incl. `reasoning_effort` L869,
`chat_template_kwargs` L910, `separate_reasoning`, `to_sampling_params` L1072),
response DTOs L1175-1271 (`ChatCompletionResponse`, `DeltaMessage` L1229,
`ChatCompletionStreamResponse` L1256), embeddings/classify/score/rerank, tokenize
L1443, and the **Responses API** DTOs (`ResponsesRequest` L1576,
`ResponsesResponse` L1851).

### A.5 Chat templates, reasoning, tool calls, multimodal

- **Templates**: `TemplateManager` (`sglang/srt/parser/template_manager.py:54`)
  resolves HF/jinja templates; `serving_chat._apply_jinja_template` (L1169) vs
  `_apply_conversation_template` (L1412) which uses
  `sglang/srt/parser/conversation.py` when a named legacy template is chosen.
- **Reasoning**: `ReasoningParser` (`sglang/srt/parser/reasoning_parser.py`)
  wraps a per-model `BaseReasoningFormatDetector`; configured via
  `--reasoning-parser`, instantiated per stream in
  `_process_reasoning_stream` (`serving_chat.py:2198`), honoring
  `request.separate_reasoning` (L714/1867). The detector's
  `think_start_token`/`think_end_token` are re-used by
  `wrap_reasoning_history` (`serving_chat.py:2275-2294`) to re-wrap prior-turn
  thinking. `apply_reasoning_enabled` (`serving_chat.py:2314-2384`) is the
  write-side toggle used by the Anthropic adapter (raises when the parser is
  absent/always-on — see errors at L2326, L2355, L2377).
- **Tool calls**: `FunctionCallParser` registry
  (`sglang/srt/function_call/function_call_parser.py:55`; `ToolCallParserEnum`
  L64 maps ~40 parser names to detector classes under `sglang/srt/function_call/*_detector.py`).
  `parse_non_stream` (L132) / `parse_stream_chunk` (L154) return
  `(normal_text, list[ToolCallItem])`. Constraint generation:
  `parser.get_structure_constraint(...)` in `_process_messages` L1104-1135
  (grammar/xgrammar JSON-schema forcing for `tool_choice="required"`/named).
- **Multi-modal**: `image_url` parts keep `data:` URIs/URLs flowing through
  `_process_messages` into `GenerateReqInput.image_data`
  (`serving_chat.py:1016-1020`).

### A.6 SSE streaming & usage accounting

- OpenAI SSE chunks are serialized with msgspec in
  `entrypoints/openai/sse_utils.py:52` (`build_sse_content` → `"data: {...}\n\n"`),
  terminated by `"data: [DONE]\n\n"` at `serving_chat.py:1770`.
- Errors mid-stream: `create_streaming_error_response`
  (`serving_base.py:214-228`) emits `data: {"error": {...}}` frames
  (usage at `serving_chat.py:1600-1606` for aborts, L1764-1768 for ValueErrors).
- Usage bookkeeping is centralized in
  `entrypoints/openai/usage_processor.py` (`UsageProcessor.calculate_response_usage`
  L18, `calculate_streaming_usage` L58), fed from per-chunk
  `meta_info` fields (`prompt_tokens`, `completion_tokens`, `reasoning_tokens`,
  `cached_tokens` — `serving_chat.py:1549-1570`); final usage chunk emitted as an
  empty-choices chunk when `include_usage` (L1731-1762).

### A.7 Other API dialects (extension-point precedents)

| Dialect | Files | Layering |
|---|---|---|
| **Anthropic** | `entrypoints/anthropic/protocol.py` (518 lines), `entrypoints/anthropic/serving.py` (1465), routes in http_server.py | **Wrapper**: `AnthropicServing(openai_serving_chat)` composes the chat handler; converts request→`ChatCompletionRequest`, calls private `_validate_request`/`_convert_to_internal_request`/`_handle_non_streaming_request`/`_generate_chat_stream`, converts back. |
| Ollama | `entrypoints/ollama/{protocol,serving,smart_router}.py` | `OllamaServing(tokenizer_manager)` — talks to TokenizerManager directly. |
| Responses API | `entrypoints/openai/serving_responses.py` | **Subclass**: `OpenAIServingResponses(OpenAIServingChat)` (L137), optional init (http_server.py:357-378 degrades gracefully), plus response store + tool-server loop. Init failure is logged and endpoint omitted (L368-378). |
| gRPC | `entrypoints/grpc_server.py` | Thin shim delegating to the external `smg-grpc-servicer` package (L156+). |
| Vertex/SageMaker | inline in http_server.py | Ad-hoc translation to `CompletionRequest`/`ChatCompletionRequest`. |

### A.8 The existing Anthropic implementation (what's already in-tree)

**`entrypoints/anthropic/protocol.py`** — Anthropic-faithful pydantic models
(documented as mirroring `anthropic-sdk-python` shapes, L1-7):

- Errors: `AnthropicError` L23, `AnthropicErrorResponse` L30.
- Usage: `AnthropicUsage` L37 — `input_tokens`/`output_tokens` **Optional**
  because streaming `message_delta` omits `input_tokens`.
- **Discriminated-union content blocks** on `type`: `TextBlock` L54, `ImageBlock`
  L59, `ToolUseBlock` L66, `ToolResultBlock` L73 (accepts both `tool_use_id` and
  legacy `id`), `ToolReferenceBlock` L82 (sglang extension for deferred tools),
  `SearchResultBlock` L92, `ThinkingBlock` L100, `RedactedThinkingBlock` L106;
  `AnthropicContentBlock` union L111-123.
- **Tools as a discriminated union with a function-tag discriminator**:
  `AnthropicCustomTool` L134 (validates `input_schema`, defaults `type:"object"`
  L143-150), `AnthropicWebSearchTool` L153 / `AnthropicComputerTool` L169 /
  `AnthropicBashTool` L181 / `AnthropicTextEditorTool` L190 (server-side
  built-ins matched by `web_search_\d{8}`-style patterns); `_tool_discriminator`
  L199-220 + `AnthropicTool` L223; `is_server_tool` L235.
- `AnthropicToolChoice` L248 (`auto|any|tool|none` — note: no
  `disable_parallel_tool_use`), `AnthropicThinkingParam` L255 (SDK-faithful
  discriminated validation: `enabled` requires `budget_tokens>=1024`, `disabled`
  forbids both fields, `adaptive` forbids `budget_tokens`; L281-312),
  `AnthropicTaskBudget` L315 / `AnthropicOutputConfig` L329 (`effort` in
  `minimal..xhigh|max`, `task_budget`), `AnthropicCountTokensRequest/Response`
  L341/355, `AnthropicMessagesRequest` L361 (`max_tokens` **required**, validated
  positive L389-394; `betas` accepted L380).
- Streaming DTOs: delta models `TextDelta`/`InputJsonDelta`/`ThinkingDelta`/
  `SignatureDelta` L402-425, `AnthropicMessageEndDelta` L428 (stop_reason ∈
  `end_turn|max_tokens|stop_sequence|tool_use`), event models L445-498
  (`MessageStartEvent` carries a full `AnthropicMessagesResponse`, L445) and the
  response DTO `AnthropicMessagesResponse` L501 (`msg_<uuid>` id L504).

**`entrypoints/anthropic/serving.py`** — `AnthropicServing` (L184) translation
layer. Docstring L1-5 states the strategy: convert to `ChatCompletionRequest`,
delegate to `OpenAIServingChat`, convert back.

Request path —
- `handle_messages` L206: convert → stream/non-stream dispatch; conversion errors
  → `invalid_request_error` 400.
- `_convert_to_chat_completion_request` L229-721:
  - **system**: top-level `system` string/block-list flattened, plus *inline*
    `role:"system"` messages merged when the chat template can't render them —
    `self._merge_inline_system` decided at init by
    `parser.template_detection.detect_inline_system_support` (L193;
    the probe renders `[system,user,system,user]` with a sentinel through the
    template, `template_detection.py:599-627`).
  - **messages**: role pass-through L443-550; assistant thinking history re-wrapped
    into parser tokens via `OpenAIServingChat.wrap_reasoning_history`
    (`_convert_assistant_thinking_blocks` L369-402; `redacted_thinking` →
    **400 ValueError** L382); tool_use → OpenAI `tool_calls` with
    `json.dumps(input)` L487-496; **tool_result** (user turn) becomes separate
    OpenAI `role:"tool"` messages with wire-order preservation via
    `_emit_user_message` flush L427-441; nested tool_result content supports
    text/image/`tool_reference`(name vs tool_name)/`search_result`
    L298-367, and groups runs of `tool_reference` parts for GLM-style templates
    L347-363; empty assistant turns emit `""` to preserve role alternation L545-550.
  - **images**: base64/URL Anthropic sources → OpenAI `image_url` parts
    (`data:{mime};base64,{data}`), L235-267.
  - **sampling**: `max_tokens` (required → OpenAI `max_tokens`),
    `stop_sequences→stop`, `temperature/top_p/top_k` L552-567; `stream_options`
    forced `{include_usage, continuous_usage_stats}` when streaming L570-574.
  - **thinking**: `apply_reasoning_enabled` L598-609 — `adaptive`≡`enabled`;
    `budget_tokens` and `display:"omitted"` accepted but logged as unenforceable
    warnings L588-608.
  - **output_config**: `effort→reasoning_effort` (`xhigh→max`) L617-622,
    `task_budget` logged-as-hint L623-631.
  - **tools**: server tools (web_search/computer/bash/text_editor) skipped with a
    log L648-660; custom tools → `Tool(type="function", defer_loading=...)`
    L664-674.
  - **tool_choice**: `auto→"auto"`, `any→"required"`, `tool→ToolChoice(function
    named)` with 400 if the named tool was skipped/absent, and 400 when
    `any|tool` is requested but all tools were server types L684-719.
- `_handle_non_streaming` L723-775: reuses `_validate_request` (L736),
  `_convert_to_internal_request` (L746), `_handle_non_streaming_request` (L754);
  non-`ChatCompletionResponse` results → `_convert_openai_error_response`.
- `_handle_streaming` L777-824: same pipeline, wraps
  `_generate_anthropic_stream` in a FastAPI `StreamingResponse` with the abort
  `BackgroundTask` from `tokenizer_manager.create_abort_task` (L821).
- `_generate_anthropic_stream` L826-1245: **event state machine** translating
  OpenAI `data:` frames:
  - SSE framing `event: {type}\ndata: {json}\n\n` via `_wrap_sse_event` L156.
  - Block helpers: `_close_content_block_events` L870 (emits trailing
    `signature_delta` only when a real signature was captured, then
    `content_block_stop`), `_ensure_content_block_events` L898 (opens a block,
    `force_new=True` for consecutive tool_use blocks — L903-911 comment),
    `_ensure_message_started` L929 (deferred until first content/usage/finish
    chunk so `input_tokens` isn't 0, L1133-1158).
  - Usage: `_anthropic_usage_from_openai` L110 — `input_tokens =
    prompt_tokens - cached_tokens` with clamp-and-warn on telemetry anomalies
    L88-107; `cache_read_input_tokens` populated L127-128; `output_tokens` zeroed
    on `message_start` (`force_zero_output` L113).
  - Delta mapping: `reasoning_content→thinking` block + `thinking_delta`
    L1160-1174; `tool_calls` → `tool_use` block with `toolu_<uuid>` fallback id
    and `input_json_delta(partial_json)` L1176-1229; text → `text_delta`
    L1231-1245.
  - Terminal: `[DONE]` → close open block → `message_delta(stop_reason)` via
    `STOP_REASON_MAP {stop→end_turn, length→max_tokens, tool_calls→tool_use}`
    L68-72 (unmapped finish reasons warn + default `end_turn` L1056-1064) → `message_stop` L1030-1073.
    *Finish-reason chunks may still carry final content — handled before
    short-circuit (L1122-1130).* Silent-empty streams (no content, no finish
    reason) emit an `api_error` event instead of a fake success (L1034-1050).
  - Error resilience: pre-first-chunk ValueError →
    `invalid_request_error` envelope L1008-1018; upstream OpenAI streaming-error
    envelopes detected (`_parse_upstream_error` L960-986) and forwarded with the
    real type via `_flush_on_error` L942-958 (always closes open content blocks so
    strict SDKs don't reject the stream L948-951); mid-flight exceptions →
    `api_error` (message scrubbed: `_scrub_error_message` L161-181 — 5xx is
    always generic "Internal server error").
- `_convert_response` (non-stream) L1247-1323: reasoning→`ThinkingBlock`
  (signature omitted — L1262-1264: no backend signature exists, and empty strings
  would fail downstream verifiers), text→`TextBlock`, `tool_calls`→`ToolUseBlock`
  with safe `json.loads` fallback L1275-1288, empty-content → single empty
  `TextBlock` L1307-1311, `stop_reason` via `STOP_REASON_MAP` L1298-1305.
- Errors: `ERROR_TYPE_MAP` (HTTP→Anthropic types, inkl. 429→`rate_limit_error`,
  503→`overloaded_error`) L74-85; `_error_response` L1374-1402 (never leaks
  exception names — logged server-side only).
- `handle_count_tokens` L1404-1465: builds a dummy `max_tokens=1` request, reuses
  the same conversion, then calls `_process_messages` directly to measure
  `prompt_ids` length (tokenizer fallback for the multimodal string path
  L1444-1449).

**Registration details**: routes http_server.py:1999-2019 (under the
`##### Anthropic-compatible API endpoints #####` banner), handler init
http_server.py:336-339, and dedicated error-envelope branches for
`/v1/messages*` in both exception handlers (http_server.py:540-565, 598-603).

### A.9 Tests & docs

- **Unit (CPU, no server)**: `test/registered/unit/entrypoints/anthropic/test_serving.py`
  (1576 lines, ~55 tests) — fakes `_FakeOpenAIServingChat`
  (test_serving.py:30-49) and drives `_generate_anthropic_stream` /
  `_convert_to_chat_completion_request` directly; covers stream block
  open/close ordering, thinking, tool_use streaming, usage/cache math, error
  envelopes, thinking-param validation, output_config/betas handling, system
  merging, tool_reference, and `_flush_on_error` paths. Registered via
  `register_cpu_ci(est_time=1, suite="base-a-test-cpu")` (L27).
- **GPU server tests**: `test/registered/openai_server/function_call/test_anthropic_tool_use.py`
  (launches a real server with `popen_launch_server` from
  `sglang.test.test_utils`, `CustomTestCase`, CI-registered for CUDA & AMD — see
  file head L1-35). General API tests via the mixin
  `python/sglang/test/kits/anthropic_messages_kit.py` (`AnthropicMessagesMixin`,
  uses the real `anthropic` SDK + raw requests), consumed by
  `test/registered/openai_server/basic/test_openai_server.py:38`
  (`class TestOpenAIServer(CustomTestCase, AnthropicMessagesMixin)`).
- **Manual VLM test**: `test/manual/vlm/test_anthropic_vision.py` (image blocks).
- **Docs**: `docs/docs/basic_usage/anthropic_api.mdx` (registered in
  `docs/docs.json` L914) — launch instructions, streaming, tools,
  count_tokens, and a Claude Code section (env vars incl.
  `CLAUDE_CODE_ATTRIBUTION_HEADER=0` for prefix-cache reuse).
- **Other (unrelated) anthropic references**: the sglang frontend-language
  Anthropic *client* backend `python/sglang/lang/backend/anthropic.py`
  (+ `python/sglang/lang/ir.py`, `examples/frontend_language/quick_start/anthropic_example_*.py`);
  `anthropic>=0.20.0` test/runtime dep in `python/pyproject.toml:20`; the Rust
  router `sgl-model-gateway/src/routers/header_utils.rs:130-171` rewrites auth to
  `x-api-key` + adds `anthropic-version: 2023-06-01` when *proxying to Anthropic's
  own API* (provider header passthrough, not a local endpoint).
- **`anthropic-version` header validation: none in the Python server** (grep
  across the tree only finds the gateway passthrough above) — the header is
  accepted/ignored, same posture as vLLM.

---

## Part B — vLLM's Anthropic implementation (web research)

### B.1 Yes — vLLM has it. PR history

Retrieved via GitHub search API (`repo:vllm-project/vllm anthropic messages in:title type:pr`):

| PR | Merged | What |
|---|---|---|
| #22627 | 2025-10-22 | **Original**: `vllm/entrypoints/anthropic/{api_server.py, protocol.py, serving_messages.py}` + `tests/entrypoints/anthropic/test_messages.py` (`anthropic` SDK client added to `tests/utils.py`) |
| #27882 | 2025-11-01 | "**Adds** anthropic /v1/messages endpoint **to openai api_server**" — collapsed the standalone api_server into the shared OpenAI server |
| #27792 | 2025-11-06 | chore: avoid duplicated serialization |
| #29971 | 2025-12-04 | fix: stream's first chunk must carry `input_tokens` |
| #34745 / #34887 | 2026-02 | fixes: empty `tool_call_id`; tool-call arguments streaming |
| #35588 | 2026-03-02 | **`/v1/messages/count_tokens`** (via `render_chat_request`) |
| #35557 | 2026-02-28 | fix: base64 image handling |
| #36992 | 2026-03-16 | accept `redacted_thinking` blocks |
| #40125 | 2026-04-20 | `chat_template_kwargs` passthrough |
| #40912 | 2026-06-20 | report cache usage (`prompt_tokens_details` → cache_read/creation) |
| #42396 | 2026-06-02 | structured-output `output_config.format.json_schema` + `effort` |
| #44283 / #46025 | 2026-06 | in-messages `role:"system"` support + auto-detect template for mid-conversation system messages (same trick as sglang) |
| #45807 | 2026-08-20 | report `stop_sequence` stop_reason (matched stop string) |
| open: #34053 (protocol compliance), #35035 (thinking request param), #46321, #47613, #48308 … | — | not merged |

### B.2 Current file layout (`main`, Sept 2026)

```
vllm/entrypoints/anthropic/
├── protocol.py     (289 lines — flat models)
├── serving.py      (1048 lines — AnthropicServingMessages)
└── api_router.py   (123 lines — FastAPI APIRouter)
```

- **Route registration**: NOT in `vllm/entrypoints/openai/api_server.py` (that file
  is now a 59-line facade → `vllm/entrypoints/launchers/api_server/*`). Real wiring:
  `vllm/entrypoints/generate/api_router.py` — `attach_router` imported as
  `register_anthropic_api_router` and called at L40-44 (gated on `"generate" in
  supported_tasks`); the handler object `state.anthropic_serving_messages =
  AnthropicServingMessages(engine_client, state.openai_serving_models,
  args.response_role, online_renderer=..., request_logger=..., chat_template=...,
  ...)` constructed in the same file (L195-210).
- **api_router.py**: `APIRouter` with `POST /v1/messages` (L51-86) and
  `POST /v1/messages/count_tokens` (L89-119); decorators `@with_cancellation`
  `@load_aware_call` (vLLM serve-utils concerns sglang handles elsewhere);
  `translate_error_response` (L39-48) re-wraps OpenAI `ErrorResponse` into
  Anthropic's envelope.
- **Layering — subclass, not wrapper**: `AnthropicServingMessages(OpenAIServingChat)`
  (serving.py:99). `create_messages` (L591-618): convert request →
  `self.create_chat_completion(chat_req, raw_request)` (inherited) →
  `messages_full_converter` (non-stream) or `message_stream_converter` (stream).

### B.3 Key translation functions (serving.py, with lines)

- `_build_anthropic_usage` L56-92: `input_tokens = prompt_tokens − cache_read −
  cache_creation` (both from `UsageInfo.prompt_tokens_details`), `exclude_unset`
  serialization so absent cache fields disappear from the wire.
- `_convert_anthropic_to_openai_request` L193-218 → `_convert_system_message`
  L221-256, `_convert_messages` L274-305, `_convert_message_content` L307-341,
  `_convert_block` L343-374, `_convert_tool_use_block` L376-387
  (`json.dumps(input)`), `_convert_tool_result_block` /
  `_convert_user_tool_result` L389-464 (tool role-message + *separate* user message
  for images L446-455 and separate tool message for `tool_reference` L457-464),
  `_build_base_request` L466-493 (`max_tokens` → **both** `max_tokens` and
  `max_completion_tokens`; `stop_sequences→stop`; plus vLLM-only
  `cache_salt`/`kv_transfer_params`/`ec_transfer_params`/`chat_template_kwargs`),
  `_handle_streaming_options` L516-529 (force include_usage),
  `_handle_output_config` L495-514 (`format.json_schema`→`ResponseFormat`,
  `effort`→`reasoning_effort`), `_convert_tool_choice` L531-558 (`any→required`,
  named→`ChatCompletionNamedToolChoiceParam`, **`disable_parallel_tool_use`→
  `parallel_tool_calls` L542-544**), `_convert_tools` L560-589 (with `strict` and
  `defer_loading` passthrough).
- **Claude Code billing-header stripping**: system blocks starting with
  `x-anthropic-billing-header` are *dropped server-side* (L238-241, L261-271) —
  sglang instead documents the client-side `CLAUDE_CODE_ATTRIBUTION_HEADER=0`.
- **Template system-first auto-detect**: `_detect_merge_inline_system`
  L143-170 renders a [s,u,s,u] conversation through the chat template in jinja's
  ImmutableSandboxedEnvironment — same probe as sglang's
  `detect_inline_system_support` (who-copied-whom: vLLM #46025 merged
  2026-06-18; sglang's unit test references GLM quirks similarly).
- `messages_full_converter` L620-680: `reasoning→thinking` block with a **fake
  `signature=uuid.uuid4().hex`** (L647-654); finish mapping incl.
  `stop_reason="stop_sequence"` when vLLM's choice carries a matched stop *string*
  (L633-640); `tool_calls`→`tool_use` with `json.loads(arguments)`; empty
  completion → single empty text block (L672-676, "strict clients assume
  content[0]").
- `message_stream_converter` L682-1017: `_ActiveBlockState` (index, type,
  signature, tool_use_id, `pending_content` — text arriving while a tool_use block
  is open is *buffered* and flushed as a new text block after the tool block
  closes, L891-897 + L772-788); `start_block`/`stop_active_block` with
  `signature_delta` emitted just before `content_block_stop` (L731-759, fake uuid
  signature per thinking block L709-711); `tool_index_to_id` map so argument-only
  deltas (no id) attach to the right block (L921-996); on `[DONE]`:
  `stop_and_flush` + `message_stop` (L790-802); usage-only final chunk →
  `message_delta` with stop_reason/usage incl. `stop_sequence` string propagation
  (L832-854); whole generator wrapped in try/except → terminal `error` event with
  `sanitize_message` (L1008-1017). Every frame via
  `wrap_data_with_event` = `event: …\ndata: …\n\n` (L95-96).
- `count_tokens` L1019-1048: converts + `self.render_chat_request(chat_req)`
  (inherited renderer/processor pipeline — the analog of sglang's
  `_process_messages`) and sums `prompt_token_ids`; response includes
  `context_management.original_input_tokens`.
- **protocol flatness**: vLLM uses one `AnthropicContentBlock` model with all
  fields optional (protocol.py:36-64) — pragmatic but loses validation; sglang's
  discriminated union is stricter. vLLM's request model has **no `thinking`
  field** (request-side thinking only via `output_config.effort`) — PR #35035 for
  that is still open; sglang is ahead here (validated `AnthropicThinkingParam`,
  wrapped thinking history, `apply_reasoning_enabled` integration). vLLM handles
  `redacted_thinking` by silently skipping (serving.py:361-366); sglang rejects it
  with a 400 (serving.py:381-382) — different policy choice.
- **`anthropic-version` / `anthropic-beta` headers**: *not validated anywhere in
  vLLM* (no match in entrypoints; tests don't exercise it). Request-level `betas`:
  no field either. sglang likewise ignores the header, but **accepts** a `betas`
  request field (logged no-op, serving.py:633-640) — closer to the SDK's wire
  shape.
- **Tests**: `tests/entrypoints/anthropic/test_messages.py` (real
  `anthropic.AsyncAnthropic` SDK against a `RemoteOpenAIServer` with
  `--tool-call-parser hermes`, `--enable-auto-tool-choice`, served model name
  `claude-3-7-sonnet-latest`), `test_anthropic_messages_conversion.py` (converter
  unit tests), `test_protocol_exports.py`.

### B.4 Alternatives (for the mapping-table canon)

- **LiteLLM** keeps OpenAI→Anthropic translation in
  `litellm/llms/anthropic/chat/transformation.py` (`AnthropicConfig`:
  `map_openai_params` L1396, `transform_request` L1783, `transform_response`
  L2526) and a proxy-level Anthropic endpoint in
  `litellm/proxy/anthropic_endpoints/endpoints.py` (`route_type="anthropic_messages"`,
  plus `count_tokens` and passthrough streaming). Useful as a second reference for
  field mapping; server-side translation direction (Anthropic→OpenAI) is
  implemented ad-hoc there.

### B.5 Canonical field mapping (consensus of sglang/vLLM/LiteLLM)

| Anthropic (request) | OpenAI chat.completions | Notes |
|---|---|---|
| `system` (str \| blocks) | leading `role:"system"` message | sglang joins parts with `\n`; vLLM concatenates and strips `x-anthropic-billing-header` |
| `messages[*] role user/assistant` (+ inline `system`) | `messages` | inline system merged into leading block if template can't render (both servers, jinja probe) |
| content block `text` | `content: str` or `{"type":"text"}` | empty assistant turn → `""` (sglang) |
| `image` (base64/url) | `image_url` part, `data:` URI for base64 | both |
| `tool_use {id,name,input}` | `tool_calls[{id,type:"function",function:{name,arguments=json.dumps(input)}}]` | id fallback `call_*` (OpenAI) / `toolu_*` (Anthropic) |
| `tool_result` (user turn) | one or more `role:"tool"` messages (`tool_call_id=block.tool_use_id`) | sglang preserves interleaved part order; vLLM splits images into a separate user msg |
| `thinking` history | re-wrapped in parser think-tokens (sglang) / `reasoning` field on msg (vLLM) | `redacted_thinking`: sglang=400, vLLM=skip |
| `tools[] {name,description,input_schema}` | `tools[{type:"function",function:{name,description,parameters}}]` | server tools skipped (sglang logs; vLLM's protocol has no server-tool variants) |
| `tool_choice auto/any/none/tool` | `"auto"/"required"/"none"/named ToolChoice` | vLLM also maps `disable_parallel_tool_use→parallel_tool_calls` |
| `max_tokens` (**required**) | `max_tokens` (+ `max_completion_tokens` in vLLM) | sglang validates >0 |
| `stop_sequences` | `stop` | |
| `temperature`/`top_p`/`top_k` | same names | |
| `stream=true` | `stream=true` + `stream_options{include_usage,continuous_usage_stats}` | forced by both |
| `thinking.enable{,d}` + `budget_tokens` | parser toggle (`reasoning_effort` / `chat_template_kwargs`) | sglang: full param; vLLM: only `output_config.effort` |
| `metadata`, `betas` | — | accepted, no-op (sglang logs) |

| Anthropic (response) | mapped from | Notes |
|---|---|---|
| `id: msg_*` | new uuid (sglang) / reused completion id (vLLM) | |
| `content`: `thinking`/`text`/`tool_use` blocks | `reasoning_content`, `content`, `tool_calls` (json.loads args) | empty completion → one empty text block (both) |
| `stop_reason`: `end_turn`/`max_tokens`/`stop_sequence`/`tool_use` | finish_reason `stop`/`length`/matched-stop-string/`tool_calls` | **`stop_sequence` propagation: vLLM yes (#45807), sglang no — see gap G1** |
| `usage.input_tokens` | `prompt_tokens − cached_tokens` | both; sglang clamps >prompt anomalies w/ warning |
| `usage.cache_read_input_tokens` | `prompt_tokens_details.cached_tokens` | both; vLLM also `cache_creation` |
| SSE events | `chat.completion.chunk` stream | message_start → content_block_{start,delta,stop} (+`thinking_delta`,`input_json_delta`,`signature_delta`) → message_delta(usage+stop_reason) → message_stop; `event:`+`data:` framing; **no literal `[DONE]`** on the wire (SDK tolerates) |

---

## Part C — Implementation plan for sglang

### C.0 Status: baseline already exists

`entrypoints/anthropic/{protocol.py,serving.py}` + http_server wiring + docs +
tests are merged (A.8/A.9). The plan below is therefore: **(1)** what to touch
when maintaining/extending it, and **(2)** the delta to close vs vLLM and the
Anthropic spec. If this tree were to lose the feature, the same plan recreates it:
the wrapper-over-`OpenAIServingChat` design is the one validated here (Responses
API subclassing is the alternative pattern; the wrapper was chosen because the
Anthropic adapter needs `ChatCompletionRequest` *construction* plus
stream-translation, not chat-behavior overrides).

### C.1 Files & extension points (existing + to-modify)

| File | Action |
|---|---|
| `python/sglang/srt/entrypoints/anthropic/protocol.py` | extend (gaps G2-G4 below) |
| `python/sglang/srt/entrypoints/anthropic/serving.py` | extend (G1-G6) |
| `python/sglang/srt/entrypoints/http_server.py` | no change needed for listed gaps; touch only for new routes (e.g. `/v1/messages/render`) or header validation middleware (G7) |
| `test/registered/unit/entrypoints/anthropic/test_serving.py` | add CPU unit tests per gap |
| `python/sglang/test/kits/anthropic_messages_kit.py` | add SDK-level e2e coverage |
| `test/registered/openai_server/function_call/test_anthropic_tool_use.py` | extend GPU tool tests |
| `docs/docs/basic_usage/anthropic_api.mdx` | document new behavior |

### C.2 Concrete gap-closure plan (diffed vs vLLM + spec)

- **G1 — `stop_sequence` propagation (vLLM #45807 parity).** Today
  `STOP_REASON_MAP` (serving.py:68) collapses every natural stop to `end_turn`,
  and `AnthropicMessageEndDelta.stop_sequence` (protocol.py:436-439) is never
  populated. Change `_convert_response` (serving.py:1298-1305) and the `[DONE]`
  terminal branch (serving.py:1056-1072) to read the matched stop:
  non-stream path exposes it on `ChatCompletionResponseChoice.matched_stop`
  (serving_chat.py:1931); stream path packs it into the finish chunk
  (`build_sse_content(..., matched_stop=...)`, serving_chat.py:1650-1657) — note
  the Anthropic converter currently parses chunks with
  `ChatCompletionStreamResponse.model_validate_json` (serving.py:1077), which
  *does* carry the field when present (protocol.py:1244
  `ChatCompletionResponseStreamChoice`). Emit `stop_reason="stop_sequence"` +
  `stop_sequence=<str>` only when the matched value is a string.
- **G2 — `tool_choice.disable_parallel_tool_use`.** Add
  `disable_parallel_tool_use: Optional[bool]` to `AnthropicToolChoice`
  (protocol.py:248-252) and set `chat_request.parallel_tool_calls = not
  disable_parallel_tool_use` in the tool_choice branch (serving.py:684-719).
  vLLM: serving.py:542-544. Also add the SDK's `tool_choice.name required when
  type=="tool"` model validator (vLLM protocol.py:100-104).
- **G3 — `output_config.format` (structured output).** vLLM #42396 maps
  `output_config.format.json_schema`→`ResponseFormat`. Add
  `AnthropicJsonOutputFormat` next to `AnthropicOutputConfig` (protocol.py:329)
  and set `chat_request.response_format` in the `output_config` block
  (serving.py:617-631).
- **G4 — vLLM-forward-compat request fields (optional, low priority):**
  `chat_template_kwargs` passthrough (vLLM #40125), `cache_salt` (natural fit —
  `ChatCompletionRequest`/`GenerateReqInput` already support it,
  serving_chat.py:1041), `kv_transfer_params`/`ec_transfer_params`. These are
  vLLM-proprietary, not Anthropic spec; add only if cross-implementation client
  code needs it. Claude Code sends none of them.
- **G5 — Claude Code billing-header stripping (policy decision).** sglang
  documents `CLAUDE_CODE_ATTRIBUTION_HEADER=0`; vLLM additionally strips
  `x-anthropic-billing-header*` system blocks server-side (vLLM serving.py:238-241).
  If sglang adopts this, do it in `_convert_system_text`/`_extract_system_text`
  (serving.py:136-153, 404-425). Keeping the doc-only approach is defensible —
  server-side stripping silently changes token counts vs. the client.
- **G6 — Usage completeness.** Surface `cache_creation_input_tokens` when the
  backend reports it (today only `cache_read_input_tokens`, serving.py:127-128),
  guard with the same clamp warning (serving.py:93-107). Extend
  `AnthropicCountTokensResponse` (protocol.py:355) with optional
  `context_management.original_input_tokens` if desired (vLLM parity, its
  protocol.py:253-256).
- **G7 — `anthropic-version` header.** Neither sglang nor vLLM validates it.
  Recommended: keep ignoring (any-version tolerance is what Anthropic's own docs
  imply for third-party backends); at most log first-seen value per process.
  Do NOT reject unknown versions — Claude Code pins headers client-side.
  (`anthropic-beta` header: same posture; the request-body `betas` field is
  already accepted-and-logged, serving.py:633-640.)
- **G8 — `ping` events & keepalive.** `PingEvent` exists (protocol.py:477) but is
  never emitted; Anthropic uses `ping` for long stalls. Optional: emit on slow
  first-token (needs a watchdog around `_generate_anthropic_stream`); low value
  for local serving.
- **G9 — Thinking signatures.** The stream pipeline already has the
  `signature_delta` hook (serving.py:870-896) but `captured_thinking_signature`
  is never assigned — no backend produces one. Deliberately leave unsigned
  (spec-compliant); do NOT copy vLLM's random-uuid signatures (they fail
  signature verification and mislead clients).
- **G10 — Route cosmetics (optional).** Anthropic SDK only ever calls
  `/v1/messages`; some proxies also expose `/anthropic/v1/messages` aliases —
  not needed; don't add.

### C.3 Streaming translation strategy (as implemented — keep)

OpenAI `data:` frames → parse-or-forward-error (upstream error envelopes stay
legible, serving.py:960-986) → deferred `message_start` (so `input_tokens` is
real, serving.py:1133-1158) → block state machine with `force_new` per tool
call (serving.py:898-927) → finish-chunk payload harvesting before terminal
events (serving.py:1122-1130) → `message_delta` + `message_stop`; `[DONE]`
consumed internally, never forwarded. This is *more* defensive than vLLM's
converter (which emits `message_start` on the first chunk unconditionally and
buffers text-after-tool_use in `pending_content`). One behavioral difference to
note when debugging cross-impl: vLLM immediate message_start vs sglang deferred.

### C.4 Edge-case checklist (verified against both codebases)

1. **tool_use streaming**: each tool call = own block + index (`force_new`);
   arg-only deltas without an open tool_use block are dropped with a warning
   (serving.py:1212-1229 — vLLM instead maps by `index`, its L971-995; sglang
   relies on the chat handler's tool-call id continuity). Zero-arg calls must
   still mark content (serving.py:1196-1199). Consecutive tool calls: separate
   blocks (test test_serving.py:1010).
2. **thinking/reasoning separation**: `reasoning_content` → thinking block at
   *start* of content; text-after-thinking closes the thinking block
   (test L980); history re-wrap needs parser tokens, else warn-and-drop /
   `redacted_thinking`→400 (serving.py:369-402).
3. **stop_reason mapping**: `stop→end_turn`, `length→max_tokens`,
   `tool_calls→tool_use`, abort/content-filter → warn+`end_turn`
   (serving.py:68-72, 1056-1064); matched stop string → `stop_sequence` (G1).
4. **max_tokens required**: enforced by pydantic validator (protocol.py:389-394);
   `count_tokens` synthesizes `max_tokens=1` (serving.py:1419).
5. **system prompt**: top-level str/blocks + inline-system merging gated on the
   jinja probe (serving.py:193, template_detection.py:599); empty/whitespace
   system parts dropped.
6. **image blocks**: base64→`data:` URI, URL passthrough, in user content and in
   nested `tool_result.content` (serving.py:235-267, 322-327); unknown source
   shapes dropped silently (returns None — consider a warning).
7. **usage fields**: `input_tokens = prompt − cache_read` (clamp+warn),
   `cache_read_input_tokens` only when >0, `output_tokens` forced 0 on
   `message_start`, streaming final usage on `message_delta` (no `input_tokens`
   there — spec), non-stream both (serving.py:88-133, 304-test).
8. **error envelopes**: HTTP-path 4xx/422→Anthropic shape via http_server
   handlers (L496-620); in-stream errors self-contained terminal sequence
   (`_flush_on_error`, serving.py:942-958); 5xx messages always scrubbed;
   exception class names never on the wire.
9. **role alternation**: empty assistant turn → `content:""`; user turn that is
   only tool_results emits **no** user message (serving.py:541-550).
10. **server tools**: web_search/computer/bash/text_editor skipped;
    `tool_choice=any|tool` with only server tools → 400 (serving.py:648-717).
11. **abort/disconnect**: `create_abort_task` BackgroundTask on
    StreamingResponse (serving.py:821-823) — same mechanism as chat.
12. **n>1 / logprobs**: Anthropic spec has no `n`/`logprobs`; the adapter only
    ever renders choice[0] — multi-`n` requests are not expressible (fine).
13. **max_tokens=1 silent-empty streams**: `[DONE]` with finish_reason but no
    content still yields a normal terminal pair; truly silent streams yield
    `api_error` (serving.py:1034-1050).

### C.5 Test plan additions

- Unit (extend `test/registered/unit/entrypoints/anthropic/test_serving.py`):
  G1 matched-stop→`stop_sequence` (stream + non-stream), G2
  disable_parallel_tool_use, G3 format mapping, G6 cache_creation, system-header
  stripping (if G5 adopted).
- e2e (extend `anthropic_messages_kit.py` + `test_anthropic_tool_use.py`):
  Claude-Code-shaped payloads (multi-block system, `betas`,
  `output_config.effort`), stop_sequences assertion, image-in-tool_result via
  the VLM manual test.

---

## Appendix — citations

- sglang (this checkout): all `python/sglang/srt/...` and `test/...` paths above,
  line numbers from `main @ d90ef6980`.
- vLLM: `github.com/vllm-project/vllm` — `vllm/entrypoints/anthropic/{api_router.py,
  serving.py, protocol.py}`, `vllm/entrypoints/generate/api_router.py` (L40-44,
  L195-210), `tests/entrypoints/anthropic/*`; PRs #22627/#27882/#29971/#34745/
  #34887/#35557/#35588/#36992/#40125/#40912/#42396/#44283/#45807/#46025
  (github.com/vllm-project/vllm/pull/<n>); open: #34053, #35035.
- LiteLLM: `github.com/BerriAI/litellm` —
  `litellm/llms/anthropic/chat/transformation.py`,
  `litellm/proxy/anthropic_endpoints/endpoints.py`.
- Note: harness `web_search` was unavailable (missing DEEPSEEK_API_KEY); vLLM/
  LiteLLM data was gathered directly from the GitHub REST API + raw.githubusercontent.com.
