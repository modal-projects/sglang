# Anthropic Messages API — Definitive Gap Audit for sglang

Date: 2026-09. Tree: `/home/ec2-user/sglang` (`main @ d90ef6980` lineage; audit performed against the live checkout).

**Authoritative inputs** (read in full):
- Spec: `docs_new/anthropic_messages_api_spec.md` — §0 (12 highest-risk contract points), §1–§12, Appendix A/B.
- Architecture/map: `docs_new/sglang_vllm_api_architecture_notes.md` — Part A (architecture), Part B (vLLM), Part C (preliminary gaps G1–G10).

**Code read in full**: `python/sglang/srt/entrypoints/anthropic/protocol.py` (518), `.../anthropic/serving.py` (1465), anthropic sections of `.../http_server.py`, `test/registered/unit/entrypoints/anthropic/test_serving.py`, `python/sglang/test/kits/anthropic_messages_kit.py`, plus OpenAI-side integration points: `openai/serving_chat.py` (matched_stop L1649/1931, prefill L307–343, `_validate_request` L829), `openai/protocol.py` (`ResponseFormat` L230, `parallel_tool_calls` L859, `continue_final_message` L903, `UsageInfo` L205), `openai/usage_processor.py`, `parser/template_detection.py` L599–627, `utils/auth.py`.

## 0. Legend

**Verdicts:** `IMPLEMENTED` · `PARTIAL` (works, contract deviates) · `MISSING` · `OUT` (deliberately out of scope, reason given).

**Severity (missions-scoped):**
- **BLOCKER-CC** — required for `ANTHROPIC_BASE_URL=<sglang> claude` (Claude Code) to work *well* with streaming + tool use + thinking. THE deliverable.
- **SPEC-FIDELITY** — needed for Anthropic conformance (spec §0 / Appendix B checklist), not needed by Claude Code.
- **OPTIONAL-OUT** — batches/files/containers/MCP/full PDF imaging/real TTL prompt caching; record decision, do not build unless strategy changes.

**Effort:** S (< ~50 LOC touched, one layer) / M (multi-file or new state machine logic) / L (engine/scheduler/infra work).

---

## 1. Conformance matrix (every spec requirement → verdict)

### A. Endpoints, transport, headers, auth (spec §1)

| # | Requirement (spec ref) | Code ref (current) | Verdict | Severity if gap |
|---|---|---|---|---|
| A1 | `POST /v1/messages` exists | http_server.py:2002–2009 | IMPLEMENTED | — |
| A2 | `POST /v1/messages/count_tokens` | http_server.py:2012–2019; serving.py:1404–1465 | IMPLEMENTED | — |
| A3 | Batches API §12.3 | — | OUT (no async queue infra; vLLM also lacks it) | OPTIONAL-OUT |
| A4 | `x-api-key` **or** `Authorization: Bearer` both authenticate (§1.2) | `utils/auth.py` `decide_request_auth` checks **Bearer only** (`_check_bearer_token`, auth.py:~106–113); middleware wired http_server.py:2559–2571 | **MISSING** → G-01 | **BLOCKER-CC** (only when server runs with `--api-key`) |
| A5 | 401/403 in Anthropic envelope | middleware returns `{"error":"Unauthorized"}` ORJSON, bypassing FastAPI exception handlers (auth.py:~188–198) | **MISSING** → G-01 | BLOCKER-CC (same condition) |
| A6 | `anthropic-version` required on every request | Not validated anywhere (notes A.9 L318–320); vLLM identical | IMPLEMENTED-as-decision (tolerate; see §4) | — |
| A7 | `anthropic-beta` comma-separated tolerated | Header ignored; request-body `betas` accepted+logged serving.py:636–640 | IMPLEMENTED (tolerate) | — |
| A8 | `request-id` response header + `request_id` in error body (§1.3, §5.1) | `AnthropicErrorResponse` has no `request_id` (protocol.py:30–34); no header set | **MISSING** → G-02 | SPEC-FIDELITY |
| A9 | `anthropic-ratelimit-*` response headers | Not emitted | OUT (no quota infra; fabricating values misleads clients) | OPTIONAL-OUT |
| A10 | 32 MB body cap → `413 request_too_large` (§1.4); type mapped http_server.py:547 | No size limit enforced | **MISSING** → G-03 | SPEC-FIDELITY |
| A11 | Timeouts / 504 `timeout_error` | ERROR_TYPE_MAP maps 504→`api_error` (serving.py:84) and http_server.py:553 same — spec §5.2 says `timeout_error` | PARTIAL → G-04 | SPEC-FIDELITY |
| A12 | CORS preflight tolerance | CORSMiddleware present (http_server.py:62; server-args driven) | IMPLEMENTED | — |

### B. Request schema (spec §2)

| # | Requirement | Code ref | Verdict | Severity |
|---|---|---|---|---|
| B1 | `max_tokens` REQUIRED | protocol.py:366 (no default) | IMPLEMENTED | — |
| B2 | `max_tokens: 0` = cache pre-warm legal (§0.1, §2.1, §9.1) | Validator rejects `v <= 0` → 400 (protocol.py:389–394) | **MISSING** → G-05 | SPEC-FIDELITY |
| B3 | `system`: string or block array | serving.py:404–425; multi-block joined `\n` | IMPLEMENTED (block `cache_control`/`citations` params ignored — see D6/H-items) | — |
| B4 | Mid-conversation `role:"system"` messages | `AnthropicMessage.role` includes `"system"` (protocol.py:127); merge-or-inline via jinja probe (serving.py:193–195, 414–420; template_detection.py:599–627) | IMPLEMENTED | — |
| B5 | **Merge consecutive same-role turns** server-side (§0.2, §2.2.2) | Adapter passes roles through unchanged (serving.py:443–550) | PARTIAL → G-06 | SPEC-FIDELITY (most chat templates tolerate `user,user`; GLM-family may not) |
| B6 | Conversation starts with user; empty conversations invalid | `_validate_request` "Messages cannot be empty" (serving_chat.py:831); first-role not enforced | PARTIAL (lenient; harmless) | SPEC-FIDELITY (minor) |
| B7 | **Assistant prefill** (last msg = assistant ⇒ continue from its end) (§0/§2.2.2) | Adapter emits assistant turn then generation prompt; worse, OpenAI layer **converts trailing assistant→user** when `continue_final_message=False` (serving_chat.py:340–342). `continue_final_message` EXISTS (openai/protocol.py:903; serving_chat.py:307–343, 456–497, 1246–1424) but the adapter never sets it | **MISSING** → G-07 | SPEC-FIDELITY (high value; prefills widely used by agent harnesses & JSON-mode tricks) |
| B8 | Prefill trailing-whitespace → 400 (§12.8) | Not validated | **MISSING** (fold into G-07) | SPEC-FIDELITY |
| B9 | **Tool pairing rules** (§0.8, §6.2): every `tool_use` id followed immediately by matching `tool_result` blocks first in next user turn → 400 `` `tool_use` ids were found without `tool_result`... `` | No validation; missing id degrades to `tool_call_id=""` (serving.py:503–519; vLLM fixed same bug #34745) | **MISSING** → G-08 | SPEC-FIDELITY |
| B10 | String shorthand for content (§2.2.3) | serving.py:447–449 | IMPLEMENTED | — |
| B11 | `metadata.user_id` pass/accept (§2.1) | Field exists (protocol.py:367); silently dropped, never read/forwarded/logged | PARTIAL (accept-and-ignore is shim-standard; spec uses it for abuse detection — N/A locally). Optionally map → OpenAI `user`. | SPEC-FIDELITY (trivial) |
| B12 | `stop_sequences` | serving.py:566–567 → `stop` | IMPLEMENTED (propagation of the *matched* sequence is the gap — see F2) | — |
| B13 | `temperature` 0–1 range (§2.1; shim clamps or 400s >1) | Passed through verbatim (serving.py:560–561); sglang accepts 0–2 → `temperature: 2` silently honored | PARTIAL → G-09 | SPEC-FIDELITY |
| B14 | `top_p`, `top_k` | serving.py:562–565 | IMPLEMENTED | — |
| B15 | `stream` | serving.py:557, 570–574 (forces `include_usage`) | IMPLEMENTED | — |
| B16 | `service_tier` request param (§2.1) + `usage.service_tier` response | Not in protocol.py (extra-key ignored silently); `AnthropicUsage` lacks field (protocol.py:37–48) | **MISSING** → G-10 | SPEC-FIDELITY |
| B17 | `container` | Not in protocol (extra ignored) | OUT (no code-exec containers) | OPTIONAL-OUT |
| B18 | `inference_geo` | Not in protocol (ignored) | OUT | OPTIONAL-OUT |
| B19 | `mcp_servers` | Not in protocol (ignored) | OUT | OPTIONAL-OUT |
| B20 | `context_management` (beta) | Not in protocol (ignored) | PARTIAL-silent-ignore. Claude Code ≥2.x **does send** `context_management` edits; ignoring is *functional* (context just grows until the 400 "prompt is too long"), but log once per session. Fold into G-06-family logging | SPEC-FIDELITY (log-only fix) |
| B21 | 100k-message cap (§2.1) | Not validated | **MISSING** (fold into G-03) | SPEC-FIDELITY |
| B22 | Thinking cross-field rules: `budget_tokens < max_tokens` (§7.2); forced `tool_choice any|tool` + manual thinking → 400 (§2.6); prefill + thinking → 400 (§7.4) | Only SDK-shape rules enforced (protocol.py:281–312); none of the cross-field 400s | **MISSING** → G-11 | SPEC-FIDELITY |
| B23 | Unknown/extra request fields | Silently ignored (pydantic v2 default `extra="ignore"`) | IMPLEMENTED-as-decision (matches Anthropic's own OAI-shim posture §11; native API 400s — document the choice) | — |
| B24 | `output_config.effort` | serving.py:617–622 (→ `reasoning_effort`; `xhigh→max` documented) | IMPLEMENTED | — |
| B25 | **`output_config.format.json_schema`** (structured outputs §2.1, §6.5) | `AnthropicOutputConfig` has only effort/task_budget (protocol.py:329–338) → `format` silently dropped; backend exists (`ResponseFormat`/`JsonSchemaResponseFormat` openai/protocol.py:219–232, request field L847, sampling wiring L1124–1133) | **MISSING** → G-28 | SPEC-FIDELITY |
| B26 | `output_config.task_budget` | Accepted, logged hint (protocol.py:315–327; serving.py:623–631) | IMPLEMENTED-as-decision (soft hint; never hard-enforced) | — |

### C. Content blocks (spec §2.3, §8)

| # | Requirement | Code ref | Verdict | Severity |
|---|---|---|---|---|
| C1 | `text` blocks | protocol.py:54–56 | IMPLEMENTED (`citations` param on incoming text silently dropped — extra-ignore) | — |
| C2 | `image` blocks (base64, url) | serving.py:235–267, 475–480; nested tool_result images L322–327; e2e kit test L55–117 | IMPLEMENTED. `file_id` source → silently dropped (returns None L267) | PARTIAL (minor: file source unimplementable w/o Files API → log the drop) |
| C3 | **`document` blocks** (PDF/text/url/content) (§2.3, §8.3) | **No `DocumentBlock` in the union** (protocol.py:111–123) → pydantic rejects with 400. Note: Claude Code's Read tool on a PDF returns a `document` block inside `tool_result.content` — **the whole request 400s and the tool loop dies** | **MISSING** → G-12 | **BLOCKER-CC** (tool-use path: CC Read-on-PDF). Full PDF→image rendering is OUT; accept-and-degrade is the fix |
| C4 | `search_result` blocks | Flattened to text (serving.py:269–296, 482–485) | PARTIAL (functional; `citations` intent lost — acceptable) | SPEC-FIDELITY (minor) |
| C5 | `tool_use` in history | serving.py:487–496 (`json.dumps(input)`) | IMPLEMENTED | — |
| C6 | `tool_result`: `tool_use_id` + legacy `id` fallback | protocol.py:73–79; serving.py:504 | IMPLEMENTED | — |
| C7 | `tool_result.is_error` honored | Field parsed (protocol.py:79) but never read by conversion — error results rendered as success text | PARTIAL → G-13 | SPEC-FIDELITY (model-quality: Anthropic models change behavior on is_error) |
| C8 | `thinking` round-trip with signature | Re-wrapped into parser think tokens (serving.py:369–402, via serving_chat.py:2275); signature accepted but not verified (nothing to verify against) | IMPLEMENTED (local-model semantics; see §4) | — |
| C9 | `redacted_thinking` | 400 (serving.py:381–382) — deliberate (vLLM silently skips) | IMPLEMENTED-as-decision (§4); PR-level alternative exists | — |
| C10 | Server-tool result blocks in history (`server_tool_use`, `web_search_tool_result`, `web_fetch_tool_result`, …) — spec §6.4: "always preserve foreign blocks on round-trip" | **Not in `AnthropicContentBlock` union (protocol.py:111–123) → request 400s.** `web_fetch_YYYYMMDD` *tool* entries also fall through `_tool_discriminator` (protocol.py:199–220) to `custom` → `input_schema` required → 400 | **MISSING** → G-14 | SPEC-FIDELITY (only hit by cross-backend conversation histories) |
| C11 | Versioning-policy tolerance (§1.6): *unknown* block/event/field types must not 400 | Discriminated union + pydantic → **unknown block types 400 the whole request** | PARTIAL → G-15 | SPEC-FIDELITY |
| C12 | Image limits (100/600 images, 8000px, many-image rule, §8.2) | Not enforced (backend does its own thing) | OUT | OPTIONAL-OUT |

### D. Tools & tool_choice (spec §2.5, §2.6, §6)

| # | Requirement | Code ref | Verdict | Severity |
|---|---|---|---|---|
| D1 | Custom tools → `Tool(type="function")` incl. `defer_loading` | serving.py:662–674; `input_schema` validated/defaulted protocol.py:143–150 | IMPLEMENTED | — |
| D2 | `strict: true` on tools (§2.5.1, §6.5 — grammar-guaranteed input) | **Field not declared → silently ignored** (extra-ignore). Backend has grammar constraints (`get_structure_constraint` for required/named, notes A.5) | PARTIAL → G-16 | SPEC-FIDELITY |
| D3 | `eager_input_streaming` (fine-grained tool streaming §4.5) | Not declared → ignored. Adapter streams whatever the backend emits without buffering anyway; grammar constraints can make args arrive in one big chunk | PARTIAL (acceptable: no client-visible promise; declare field + doc) | SPEC-FIDELITY (minor) |
| D4 | `tool_choice` auto/any/tool/none | serving.py:684–719 (`any`→`required`, named→`ToolChoice`), 400 on named-missing / all-server-tools | IMPLEMENTED | — |
| D5 | `tool_choice.name` required when `type=="tool"` | Currently a None-name falls into the "not in tools list" 400 (serving.py:699–706) — fine UX, but add explicit validator (vLLM protocol.py:100–104 parity) | PARTIAL (minor) | SPEC-FIDELITY (trivial) |
| D6 | **`disable_parallel_tool_use`** (§2.6) — auto⇒≤1 call, any/tool⇒exactly 1 | **Absent from `AnthropicToolChoice`** (protocol.py:248–252); backend supports `parallel_tool_calls` (openai/protocol.py:859) | **MISSING** → G-17 | SPEC-FIDELITY (harnesses that serialize tool calls use it) |
| D7 | Parallel tool use default (multiple `tool_use` blocks in one response) | Stream `force_new` per call (serving.py:898–927, 1185–1199); non-stream list (L1273–1296) | IMPLEMENTED | — |
| D8 | Server tools: accept-and-ignore vs explicit 400 policy (§2.5.2, §6.4; notes C.4-10) | web_search/computer/bash/text_editor skipped with log (serving.py:648–660); forced-choice with only server tools → explicit 400 (L711–717) | IMPLEMENTED-as-decision (§4). Gap: `memory_*` tool type unrecognized→custom-400; `web_fetch_*` 400 (folds into G-14) | — |
| D9 | `tool_reference` blocks | IMPLEMENTED (protocol.py:82–89; serving.py:328–335, GLM grouping L347–363) | IMPLEMENTED (sglang extension, spec-aligned) | — |

### E. Thinking (spec §7)

| # | Requirement | Code ref | Verdict | Severity |
|---|---|---|---|---|
| E1 | `thinking.type` enabled/disabled/adaptive; SDK-shape validation | protocol.py:255–312 (enabled⇒budget≥1024; disabled/adaptive forbid fields) | IMPLEMENTED (adapter treats adaptive≡enabled, logged docs) | — |
| E2 | `budget_tokens` honored as budget | **Not enforceable** — accepted w/ WARNING (serving.py:588–594) | PARTIAL (deliberate; engine has no knob. §4 keep) | — |
| E3 | `display: "omitted"` | Accepted w/ WARNING, still streams (serving.py:599–608) | PARTIAL (deliberate; §4 keep) | — |
| E4 | Thinking blocks before other blocks; round-trip rules (§7.3) | `_convert_assistant_thinking_blocks` prepends reasoning history (serving.py:456–459, 461–466) | IMPLEMENTED | — |
| E5 | `signature` round-trip verified | No crypto possible locally; signature never fabricated (serving.py:878–889; unlike vLLM's fake-uuid sigs) | IMPLEMENTED-as-decision (§4) | — |
| E6 | Interleaved thinking in streamed output (thinking → text/tool → thinking) | State machine handles block-type transitions generically (serving.py:1160–1245) | IMPLEMENTED (parser-dependent: reasoning parsers usually emit reasoning prefix-only, mechanically still correct) | — |
| E7 | `usage.output_tokens_details.thinking_tokens` (§3.3, §7.4) | `UsageInfo.reasoning_tokens` populated from `meta_info` (usage_processor.py:32; openai/protocol.py:211) but the adapter never maps it; `AnthropicUsage` lacks the field | **MISSING** → G-18 | SPEC-FIDELITY |

### F. Response schema (spec §3) & stop reasons

| # | Requirement | Code ref | Verdict | Severity |
|---|---|---|---|---|
| F1 | Response envelope (`id msg_*`, `type`, `role`, `model`, `content[]`, `usage`) | protocol.py:501–513; serving.py:1247–1323 | IMPLEMENTED | — |
| F2 | **`stop_reason: "stop_sequence"` + `stop_sequence: <matched string>`** (§0.5, §3.1, §12.8) | `STOP_REASON_MAP` collapses `stop→end_turn` (serving.py:68–72, 1056–1073, 1298–1305); `matched_stop` is *available* on choices (openai/protocol.py:1191, 1253; serving_chat.py:1649–1657 stream, 1931–1935 non-stream) — never read by adapter | **MISSING** → G-19 (≡ notes G1) | SPEC-FIDELITY (cheap, exact) |
| F3 | `stop_reason` 7-value coverage (§0.5): `pause_turn`, `refusal`(+`stop_details`), `model_context_window_exceeded` missing | Literal restricted to 4 values (protocol.py:436–438, 509–511) | PARTIAL → split: F3a `model_context_window_exceeded` G-20 (needs scheduler distinguishing *context-full* length from *max_tokens* length — see scheduler `schedule_batch.py:273` `"type":"length"` only); F3b `refusal`/`pause_turn` OUT (no safety classifier stack / no server-tool loop — structurally not producible; document) | SPEC-FIDELITY / OPTIONAL-OUT |
| F4 | `content: []` edge (§3.1, §12.8 — SDK must not crash) | Adapter injects one empty `TextBlock` (serving.py:1307–1311) — deliberate dev (§4) | IMPLEMENTED-as-decision | — |
| F5 | `stop_sequence`, `stop_details`, `container` response fields | `stop_sequence` on model but never set (G-19); `stop_details`/`container` absent (OUT w/ F3b/B17) | PARTIAL (covered) | — |
| F6 | Non-stream `stop_reason` always non-null | Defaults `end_turn` (serving.py:1299, 1305) | IMPLEMENTED | — |

### G. Streaming SSE (spec §4)

| # | Requirement | Code ref | Verdict | Severity |
|---|---|---|---|---|
| G-s1 | Named events, `event:` line == JSON `type`, single-line data, no `[DONE]` | `_wrap_sse_event` serving.py:156–158; `[DONE]` consumed internally (L1030) | IMPLEMENTED (kit test_raw_http_streaming asserts ordering) | — |
| G-s2 | Event lifecycle ordering (message_start → blocks → message_delta → message_stop; block start/deltas/stop containment) | State machine serving.py:826–1245; unit tests cover ordering | IMPLEMENTED | — |
| G-s3 | `message_start` carries real input usage (spec: skeleton w/ usage; Anthropic emits immediately) | **Deferred** until first payload/usage chunk (serving.py:1133–1158) — deviation, *more* accurate than Anthropic/vLLM (§4) | IMPLEMENTED-as-decision | — |
| G-s4 | `content_block_start` skeletons (empty text/input/thinking) | L1188–1192 (`input:{}`), L1234 (`text:""`), L1164 (`thinking:""`) | IMPLEMENTED. Nit: Anthropic commonly sends an empty first `partial_json ""`; sglang only sends real fragments | PARTIAL (cosmetic; SDK-tolerated) |
| G-s5 | `input_json_delta` partial-JSON streaming; one tool per block/index | L1176–1229; force_new per call L1185–1194; zero-arg call start still marks content L1196–1199 | IMPLEMENTED | — |
| G-s6 | `thinking_delta` + single `signature_delta` right before `content_block_stop` | Hook present (serving.py:870–896); `captured_thinking_signature` never assigned → no signature event (correct locally) | IMPLEMENTED | — |
| G-s7 | **`ping` keepalive events** (§4.2) | `PingEvent` defined (protocol.py:477–479), **never emitted** (not even imported in serving.py) | **MISSING** → G-21 | SPEC-FIDELITY (real value when sglang sits behind idle-killing proxies/LBs during long thinking streams; Claude Code direct = fine) |
| G-s8 | `message_delta` cumulative usage, output_tokens required; optional input_tokens echo | serving.py:1106–1115, 1065–1070 (`output_tokens` only) | IMPLEMENTED (input echo optional per spec) | — |
| G-s9 | Mid-stream `error` events, stream ends w/o message_stop on failure | `_flush_on_error` (serving.py:942–958) + upstream envelope detection (L960–986) + message scrubbing | IMPLEMENTED (deliberately also closes open blocks + emits message_stop for SDK strictness — §4) | — |
| G-s10 | Silent-empty stream → error, not fake success (spec §12.8 spirit) | serving.py:1034–1050 | IMPLEMENTED | — |
| G-s11 | Pre-first-chunk failures should be HTTP 400, not post-200 SSE error (OpenAI path kick-starts generator notes A.2-6) | Anthropic `_handle_streaming` returns StreamingResponse without kick-start (serving.py:813–824); ValueError path → SSE error event after HTTP 200 (L1008–1018) | PARTIAL → G-22 | SPEC-FIDELITY |
| G-s12 | Abort on client disconnect | `create_abort_task` BackgroundTask (serving.py:821–823); graceful abort → terminal end_turn | IMPLEMENTED | — |
| G-s13 | Fine-grained (token-by-token) tool arg streaming when `eager_input_streaming` | see D3 | PARTIAL | SPEC-FIDELITY (minor) |

### H. Errors (spec §5) & status taxonomy

| # | Requirement | Code ref | Verdict | Severity |
|---|---|---|---|---|
| H1 | Envelope `{"type":"error","error":{type,message},"request_id":...}` | serving.py:1374–1402; http_server.py:521–529 | PARTIAL — missing `request_id` (G-02) | SPEC-FIDELITY |
| H2 | 400 invalid_request_error | http_server.py:543, 598–603; serving.py:218 | IMPLEMENTED | — |
| H3 | 401 authentication_error / 403 permission_error | Types mapped (serving.py:76–77, http_server.py:544–545) **but unreachable**: middleware short-circuits with `{"error":"Unauthorized"}` (G-01) | PARTIAL → G-01 | BLOCKER-CC |
| H4 | 402 billing_error | No billing | OUT | OPTIONAL-OUT |
| H5 | 404 not_found_error (unknown model) | `GET /v1/models/{id}` 404s with **OpenAI** envelope (http_server.py:1881–1891); unknown `model` on /v1/messages is *not validated at all* (sglang serves its one model regardless) | PARTIAL → G-23 | SPEC-FIDELITY (careful: aliases — validate only if `model` clearly mismatches served names, or skip) |
| H6 | 409 conflict_error | No stateful resources | OUT | OPTIONAL-OUT |
| H7 | 429 rate_limit_error (+`retry-after`) | Type mapped (serving.py:80); sglang emits 429 rarely; no `retry-after` | PARTIAL (acceptable) | SPEC-FIDELITY (minor) |
| H8 | 500 api_error, message scrubbed | serving.py:161–181, 759–766 | IMPLEMENTED | — |
| H9 | 504 `timeout_error` (≠ api_error) | Mapped to `api_error` (A11) | PARTIAL → G-04 | SPEC-FIDELITY |
| H10 | **529 `overloaded_error`** (HTTP 529, Anthropic-specific; SDK retry keyed on it) | 503→`overloaded_error` *type* but **status stays 503** (serving.py:83, http_server.py:552) | PARTIAL → G-24 | SPEC-FIDELITY (Claude Code parses the type + message; mostly cosmetic for CC, exact for SDKs) |
| H11 | 422→400 remap for /v1/messages | http_server.py:598–603 | IMPLEMENTED | — |
| H12 | Error path never leaks internals/paths | `_scrub_error_message` (serving.py:161–181); `_anthropic_validation_message` (http_server.py:496–518) | IMPLEMENTED | — |

### I. Usage & caching accounting (spec §3.3, §9)

| # | Requirement | Code ref | Verdict | Severity |
|---|---|---|---|---|
| I1 | `input_tokens` = post-breakpoint; total = input+cache_creation+cache_read | Adapter: `input_tokens = prompt_tokens − cached_tokens` w/ clamp+warn (serving.py:88–133). Semantics align with Anthropic *shape* given radix cache = implicit shared prefix | IMPLEMENTED | — |
| I2 | `cache_read_input_tokens` populated | **Only when server started `--enable-cache-report`** (usage_processor.py:36–43, 71–74; serving_chat.py:1960) — default-off ⇒ field almost never emitted | PARTIAL → G-25 | SPEC-FIDELITY (Claude Code shows cache stats; no functional break) |
| I3 | `cache_creation_input_tokens` | sglang reports no cache-write attribution | **MISSING** (needs scheduler/radix-cache write stats) | SPEC-FIDELITY — effort **L**, mark optional |
| I4 | `cache_creation.ephemeral_{5m,1h}_input_tokens` | No TTL concept | OUT (no TTL infra) | OPTIONAL-OUT |
| I5 | `cache_control` breakpoints on tools/system/messages + top-level auto-caching (§9) | **No `cache_control` field anywhere in anthropic/**; today saved by accident (pydantic extra-ignore ⇒ Claude Code's breakpoint-laden bodies parse fine) | PARTIAL → G-26 | SPEC-FIDELITY (make ignore *explicit* so SDK round-trips and future fields stay safe; sglang's radix cache already gives the perf) |
| I6 | `server_tool_use` usage counts | No server tools | OUT | OPTIONAL-OUT |
| I7 | `output_tokens_details.thinking_tokens` | Missing (E7/G-18) | **MISSING** | SPEC-FIDELITY |
| I8 | `service_tier` response field | Missing (B16/G-10) | **MISSING** | SPEC-FIDELITY |

### J. count_tokens & models (spec §12.1, §12.2)

| # | Requirement | Code ref | Verdict | Severity |
|---|---|---|---|---|
| J1 | count_tokens accepts same body minus output params; returns `{"input_tokens": N}` | serving.py:1404–1465 (renders via `_process_messages`; multimodal tokenizer fallback L1444–1449) | IMPLEMENTED (local counts ≈, document underestimate caveat §12.1) | — |
| J2 | count_tokens validation errors in envelope | L1428–1434 | IMPLEMENTED | — |
| J3 | `GET /v1/models` Anthropic shape (`data[{type:"model",display_name,created_at,max_input_tokens,max_tokens,capabilities}]`, first_id/last_id/has_more) | Route exists but emits **OpenAI** `ModelList/ModelCard` (http_server.py:1843–1897) — same path namespace, can't naively change without breaking OpenAI clients | **MISSING** → G-27 | SPEC-FIDELITY (Claude Code tolerates the foreign 200 shape; treat as polish) |
| J4 | `GET /v1/models/{id}` Anthropic shape + Anthropic-envelope 404 | OpenAI shape; OpenAI 404 envelope (http_server.py:1880–1891) | PARTIAL (same fix as G-27) | SPEC-FIDELITY |

---

## 2. Gap list (actionable items only)

| ID | Area | Spec ref | Current code ref | Verdict | Severity | Recommended fix sketch | Effort |
|---|---|---|---|---|---|---|---|
| **G-01** | Auth: `x-api-key` + Anthropic 401/403 envelope | §1.2, §5.2 | `utils/auth.py` `decide_request_auth` (Bearer-only); middleware ORJSON `{"error":"Unauthorized"}`; wiring http_server.py:2559–2571 | MISSING | **BLOCKER-CC** | (a) In `decide_request_auth`, treat `x-api-key: <k>` like a Bearer token when `path.startswith("/v1/messages")` (Anthropic SDKs default to `x-api-key`; Claude Code w/ `ANTHROPIC_API_KEY` sends it). (b) In `add_api_key_middleware._ApiKeyASGIMiddleware.__call__`, on denial branch for `/v1/messages*` return `{"type":"error","error":{"type":"authentication_error"\|"permission_error","message":...}, "request_id": req_*}` instead of `{"error":...}`. — Skip entirely when `--api-key` unset (no-op). | M |
| **G-02** | `request-id` header + `request_id` in error bodies | §1.3, §5.1 | serving.py:1374–1402; protocol.py:30–34; http_server.py:521–529 | MISSING | SPEC-FIDELITY | Add `request_id: Optional[str]` to `AnthropicErrorResponse`; generate `req_<uuid4hex>` in `_error_response`/`_anthropic_error_response`; set matching `request-id` response header (also on 200s via tiny middleware for `/v1/messages*`). | S |
| **G-03** | Body size + message-count caps → 413 `request_too_large` | §1.4, §12.8 | none | MISSING | SPEC-FIDELITY | Dependency/middleware on `/v1/messages*`: reject `content-length > 32MB` (or measured body) with 413 + `request_too_large`; validate `len(messages) ≤ 100_000` (400). Type already mapped (http_server.py:547). | S |
| **G-04** | 504 type string | §5.2 | serving.py:84; http_server.py:553 | PARTIAL | SPEC-FIDELITY | Map `504 → "timeout_error"` in both maps (spec enum; SDK has `APITimeoutError` on connection timeouts anyway, but keep wire enum exact). | S |
| **G-05** | `max_tokens: 0` (pre-warm) | §0.1, §2.1, §9.1 | protocol.py:389–394 | MISSING | SPEC-FIDELITY | Change validator to `v < 0` → error; in `handle_messages`, when `max_tokens == 0`: skip engine call, synthesize response `{content: [], stop_reason: "max_tokens", stop_sequence: null, usage: {input_tokens: <count via the count_tokens path>, output_tokens: 0}}` (stream: message_start+delta+stop). Also relax/thinking-incompatibility note §2.7 (`thinking.enabled` + mt:0 → 400). | M |
| **G-06** | Merge consecutive same-role messages | §0.2, §2.2.2 | serving.py:443–550 pass-through | PARTIAL | SPEC-FIDELITY | Pre-pass over converted openai_messages: merge adjacent same-role `user`/`assistant` (concat text w/ `\n`, concat block lists; merge `tool_calls`). Do it on the **Anthropic** side pre-conversion so tool-result flushing stays per-block. Guard: skip merge across a `tool_use → tool_result` pairing boundary. | M |
| **G-07** | Assistant prefill semantics | §2.2.2, §7.4, §12.8 | adapter lacks; openai `continue_final_message` unused (openai/protocol.py:903; serving_chat.py:307–343) | MISSING | SPEC-FIDELITY | In `_convert_to_chat_completion_request`: if last message role is `assistant`, set `chat_request.continue_final_message = True`; coerce the trailing assistant content to a string (join text blocks; refuse — 400 — if it contains tool_use/thinking blocks per §7.4 + pairing rules); 400 on prefill + `thinking.enabled` (§7.4); 400 on trailing whitespace in the prefill text (§12.8). Note continue_final_message only handles string content (serving_chat.py:320, 335) so coercion must happen on our side. | M |
| **G-08** | Tool pairing validation | §0.8, §6.2 | serving.py:503–519 (`tool_call_id = ... or ""`) | MISSING | SPEC-FIDELITY | In `_convert_to_chat_completion_request` (or a `_validate_pairing` pre-pass on the Anthropic messages): (a) collect `tool_use` ids of final assistant turn; require next message is `user` whose blocks *begin* with exactly-matching `tool_result`s (one per id, any order); (b) require `tool_result.tool_use_id` (or legacy `id`) non-empty — 400 with spec message `` `tool_use` ids were found without `tool_result` blocks immediately after ``. Keep behavior for well-formed CC traffic unchanged. | M |
| **G-09** | `temperature` range fidelity | §2.1, §10.1 | serving.py:560–561 | PARTIAL | SPEC-FIDELITY | Clamp `>1.0` to `1.0` with a WARNING (Anthropic-shim choice §10.1), or 400. Clamp `<0` → 400. | S |
| **G-10** | `service_tier` param + response echo | §2.1, §3.3 | absent (protocol.py:37–48, 361–380) | MISSING | SPEC-FIDELITY | Add `service_tier: Optional[Literal["auto","standard_only"]]` to request (accept/no-op + debug log); add `service_tier: Optional[str]` to `AnthropicUsage`; emit `"standard"` (the only tier a local server has). | S |
| **G-11** | Thinking cross-field 400s | §2.6, §7.2, §7.4 | protocol.py:281–312 (shape only) | MISSING | SPEC-FIDELITY | After shape validation: `enabled && budget_tokens >= max_tokens` → 400 (spec: must be `< max_tokens`); `enabled && tool_choice in (any, tool)` → 400 (§2.6/§7.4); sampled-params-with-thinking restrictions are model-specific → document, don't enforce. | S |
| **G-12** | `document` blocks (PDF/text/url/content) | §2.3, §8.3 | union lacks it (protocol.py:111–123) | MISSING | **BLOCKER-CC** (CC `Read` on PDF → `document` block in `tool_result.content` → whole request 400; tool loop dies) | Add `DocumentBlock{type, source(content/b64/url/text/file), title?, context?, citations?}` to the union + nested handling in `_convert_tool_result_content` and user-content loop: `text/plain` → text part; `content` documents → flatten child blocks (text/image); `base64 application/pdf`/`url` → if model multimodal-capable and sglang gains PDF plumbing later, wire it; **otherwise degrade to an explicit text placeholder** `[PDF document "title" omitted: backend lacks PDF support]` + WARNING (do **not** 400 — a failed tool loop is worse than degraded context), or 400 with a *clear* message as a policy flag. Files-API `file_id` → 400 with explicit "Files API unsupported". | M |
| **G-13** | `tool_result.is_error` honored | §2.3, §6.1 | parsed (protocol.py:79), never read | PARTIAL | SPEC-FIDELITY | When `is_error: true`, wrap content: prefix `[Tool execution failed] ` (or template-native error marker) in converted tool message(s) so the model retries intelligently instead of treating it as success. | S |
| **G-14** | Opaque round-trip of server-tool blocks + `web_fetch_*`/`memory_*` tool entries | §2.3, §2.5.2, §6.4, §1.6 | protocol.py:111–123, 134–232; serving.py:648–660 | MISSING | SPEC-FIDELITY | (a) Add catch-all block models to `AnthropicContentBlock` for `server_tool_use`, `web_search_tool_result`, `web_fetch_tool_result`, `code_execution_tool_result`(family) with `model_config = ConfigDict(extra="allow")`, converted to a short text placeholder in prompts (never dropped silently; log). (b) Extend `_tool_discriminator` to recognize `web_fetch_`, `code_execution_`, `tool_search_tool_`, `memory_`, toolset patterns → new server-tool models → same skip-with-log path. | M |
| **G-15** | Unknown future block types tolerated (versioning §1.6) | §1.6, App-B.8 | discriminated union 400s on unknown `type` | PARTIAL | SPEC-FIDELITY | Add a final fallback variant `GenericBlock(type: str, extra=allow)` (or a custom discriminator default) to the content-block union; conversion: log + degrade to text placeholder. Keeps tomorrow's clients working. | M |
| **G-16** | Tool `strict: true` | §2.5.1, §6.5 | field absent → ignored | PARTIAL | SPEC-FIDELITY | Declare `strict: Optional[bool]` on `AnthropicCustomTool`; when `strict` and tool is force-chosen, reuse sglang's JSON-schema constraint path (`json_schema` sampling param — the same machinery `ResponseFormat` uses, openai/protocol.py:1124–1133); otherwise log that per-tool strictness is best-effort. | M |
| **G-17** | `disable_parallel_tool_use` | §2.6, §6.3 | `AnthropicToolChoice` protocol.py:248–252; backend `parallel_tool_calls` openai/protocol.py:859 | MISSING | SPEC-FIDELITY | Add `disable_parallel_tool_use: Optional[bool]`; in tool_choice branch set `chat_request.parallel_tool_calls = not tc.disable_parallel_tool_use` (default True). (≡ vLLM serving.py:542–544; notes G2.) | S |
| **G-18** | `usage.output_tokens_details.thinking_tokens` | §3.3, §7.4 | data exists upstream (`UsageInfo.reasoning_tokens`, openai/protocol.py:211) | MISSING | SPEC-FIDELITY | Extend `AnthropicUsage` with `output_tokens_details: Optional[dict]`; in `_anthropic_usage_from_openai` map `usage.reasoning_tokens` → `output_tokens_details.thinking_tokens` when >0 (stream + non-stream). | S |
| **G-19** | `stop_sequence` propagation | §0.5, §3.1, §12.8 (≡ notes G1) | serving.py:68–72, 1056–1073, 1298–1305 | MISSING | SPEC-FIDELITY | Non-stream: read `choice.matched_stop` (serving_chat.py:1931); when `finish_reason=="stop"` and `matched_stop` is a *string* → `stop_reason="stop_sequence"`, `stop_sequence=<str>` (int = matched stop-token id → keep `end_turn`). Stream: finish chunk carries `matched_stop` (serving_chat.py:1649–1657 → ChatCompletionResponseStreamChoice.matched_stop, openai/protocol.py:1253); capture alongside `finish_reason` (serving.py:1128–1129) and emit in terminal `message_delta`. | S |
| **G-20** | `stop_reason: "model_context_window_exceeded"` | §3.1, §12.4 | scheduler only emits `{"type":"length"}` (schedule_batch.py:273) | MISSING | SPEC-FIDELITY (M) | Requires distinguishing context-window exhaustion from max_new_tokens exhaust in scheduler `meta_info.finish_reason` (e.g. add `"reason": "context"` flag). Then map in both STOP_REASON_MAP + Literal expansion (protocol.py:436–438, 509–511). Engine work — split into its own PR. | M–L |
| **G-21** | `ping` keepalive | §4.2 | PingEvent protocol.py:477–479, never emitted | MISSING | SPEC-FIDELITY | Wrap the OpenAI stream iterator with an idle watchdog (e.g. `asyncio.wait_for(stream_iter.__anext__(), timeout=10s)`) yielding `PingEvent` frames on timeout until first content; reset per chunk. No engine changes. | M |
| **G-22** | Pre-first-chunk errors should be HTTP 400 (not post-200 SSE error) | §4.2/§5.4 spirit; OpenAI parity (notes A.2) | serving.py:813–824 (no kick-start; ValueError handled at L1008–1018 inside stream) | PARTIAL | SPEC-FIDELITY | Adopt the chat handler's kick-start pattern: pull first item of `_generate_chat_stream` inside `_handle_streaming` before constructing `StreamingResponse`; on ValueError return HTTP 400 envelope (`_error_response`), else re-chain first item into generator. | S–M |
| **G-23** | 404 for unknown `model` on /v1/messages | §5.2 | not validated; GET model 404 envelope is OpenAI-shaped (http_server.py:1881–1891) | PARTIAL | SPEC-FIDELITY | Optional: if `request.model` ∉ served names (+ aliases), return 404 `not_found_error` — behind a flag, since serve-time aliases make name-matching opinionated. Fix GET envelope under G-27. | S |
| **G-24** | HTTP 529 for overload | §5.2, App-B.4 | 503 status kept (serving.py:83, http_server.py:552) | PARTIAL | SPEC-FIDELITY | For `/v1/messages*`: translate 503 → status **529**, type `overloaded_error` (Starlette/httpx accept non-IANA codes; SDKs retry 529 by default). Keep 503 for other dialects. | S |
| **G-25** | `cache_read_input_tokens` reporting gate | §3.3, §9.5 | usage_processor.py:36–43 (only under `--enable-cache-report`) | PARTIAL | SPEC-FIDELITY | Stop requiring `--enable-cache-report` for the Anthropic path: either always populate `prompt_tokens_details.cached_tokens` in UsageProcessor (it is gated *only* by the flag — decide deliberately: exposure on OpenAI API is the historical reason for the flag), or have the adapter read raw `meta_info.cached_tokens`. Cheapest correct: make the flag default-on only affects nothing else — prefer adapter-side read. | S–M |
| **G-26** | `cache_control` explicit accept-and-ignore | §9 | zero occurrences in anthropic/*.py (extra-ignore accident) | MISSING(explicit) | SPEC-FIDELITY | Add `cache_control: Optional[CacheControl]` (type `ephemeral`, ttl `5m\|1h`) to TextBlock/ImageBlock/DocumentBlock/ToolUse/ToolResult/system blocks + tools + top-level request field; log first-seen at INFO "mapped to engine radix cache automatically". Do not attempt TTL semantics. | S–M |
| **G-27** | Anthropic-shaped `GET /v1/models` (+`/{id}`) | §12.2, App-B.6 | OpenAI `ModelList` (http_server.py:1843–1897) | MISSING | SPEC-FIDELITY | Content-negotiate: if request carries `anthropic-version` header (or User-Agent `anthropic-*`/Claude-Code), emit `{data:[{type:"model", id, display_name: id, created_at, max_input_tokens: context_len, max_tokens: min(context_len,128k)?— read from ModelCard.max_model_len}], first_id, last_id, has_more:false}`; else keep OpenAI shape. Same for `/{id}` incl. Anthropic-envelope 404. | M |
| **G-28** | `output_config.format.json_schema` → `ResponseFormat` | §2.1, §6.5 (≡ notes G3) | `AnthropicOutputConfig` protocol.py:329–338; backend openai/protocol.py:219–232, 847, 1124–1133 | MISSING | SPEC-FIDELITY | Add `format: Optional[AnthropicOutputFormat]` (`{type:"json_schema", schema: dict}`) to `AnthropicOutputConfig`; in serving.py:617–631 set `chat_request.response_format = ResponseFormat(type="json_schema", json_schema=JsonSchemaResponseFormat(name="anthropic_structured_output", schema=schema))` (alias `schema_`). Unknown `format.type` → 400. | S |

### Non-actionable / recorded OUT decisions

| Item | Reason |
|---|---|
| Batches API §12.3 | No async queue/store infra; not required for minimal conformance (§1.1); vLLM lacks it too |
| Files/Skills/Containers/MCP/`container_upload` | Whole separate APIs; `container`/`mcp_servers` silently ignored by extra-ignore — acceptable + documented |
| `refusal` + `stop_details` | No safety-classifier stack in sglang; the value is structurally unproducible. Re-expand the Literals only when a producer exists |
| `pause_turn` | No server-side tool loop; unproducible |
| Full PDF→per-page image+text rendering (§8.3) | Huge lift, model-dependent; covered by G-12 degrade path |
| Real prompt-caching TTLs + `cache_creation*` write attribution | No radix-cache write telemetry today (I3, effort L). Revisit if scheduler adds write stats |
| Image count/size limits (§8.2) | Backend-opaque; enforce only if tokenizer manager exposes dims |
| Rate-limit/priority headers (§1.3/§12.7) | No quota infra; fabricating headers is worse than omitting |
| Fine-grained tool streaming *guarantee* | Adapter never buffers; guarantee belongs to the tool-call parser layer (D3/G-s13) |
| `seed`, `n>1`, logprobs, penalties | Not in Anthropic spec for inbound; `n` silently stays 1 (documented notes C.4-12) |

---

## 3. Implementation order (tranches)

Ordering principle: Claude Code correctness first, then cheap spec wins, then structural work. Every item lands with unit tests in `test/registered/unit/entrypoints/anthropic/test_serving.py` (+ kit/GPU tests where noted).

**Tranche 0 — unify the prior G1–G10 ledger** (mapping in §6): G1→G-19, G2→G-17, G3→G-28, G6→G-18+G-25, G8→G-21; G4 (=OUT), G5/G7/G9/G10 are standing *decisions* (see §4). Everything else in this audit is new.

**Tranche 1 — Claude Code hardening (BLOCKER-CC + CC-visible).**
1. **G-01** x-api-key + Anthropic 401/403 envelope (M) — *top of list; only thing that fully blocks login behind `--api-key`*
2. **G-12** document-block acceptance/degrade (M)
3. **G-25** cache-read usage always available (S–M)
4. **G-08** tool pairing 400s + non-empty `tool_call_id` (M) — protects CC sessions from silent history corruption
5. **G-15** unknown-block tolerance (M) + **G-14** server-tool block round-trip (M) — future CC versions send newer blocks
6. **G-19** stop_sequence propagation (S) + kit e2e assertion

**Tranche 2 — Cheap spec-fidelity sweep (all S).**
7. **G-17** disable_parallel_tool_use (S)
8. **G-28** `output_config.format.json_schema → ResponseFormat` (S; add `AnthropicOutputFormat{type:"json_schema",schema}` to `AnthropicOutputConfig`, protocol.py:329–338; set `chat_request.response_format = ResponseFormat(type="json_schema", json_schema=JsonSchemaResponseFormat(name="anthropic_output", schema_=schema))` in serving.py:617–631 — backend machinery already exists, openai/protocol.py:230–232, 847, 1124–1133) *(≡ notes G3 — keep as its own PR)*
9. **G-05** max_tokens=0 prewarm (M)
10. **G-11** thinking cross-field 400s (S)
11. **G-09** temperature clamp/400 (S)
12. **G-02** request-id + request_id (S)
13. **G-04** 504→timeout_error; **G-24** 503→HTTP 529 (S)
14. **G-10** service_tier accept + `"standard"` echo (S)
15. **G-18** thinking_tokens usage detail (S)
16. **G-13** is_error surfacing (S)
17. **G-26** explicit cache_control accept-ignore (S–M)
18. **G-06** same-role merge (M) + log ignored `context_management`
19. **G-03** 413 cap (S); **D5** tool_choice name validator (trivial)

**Tranche 3 — Structural fidelity.**
20. **G-07** prefill via `continue_final_message` (+ trailing-whitespace 400)
21. **G-21** ping keepalive watchdog
22. **G-22** streaming pre-200 validation kick-start
23. **G-27** Anthropic-shaped models endpoints (header-negotiated)
24. **G-23** optional 404-on-unknown-model (flagged)
25. **G-16** tool `strict` grammar wiring
26. **G-20** model_context_window_exceeded (needs scheduler finish-reason enrichment; coordinate w/ engine owners)

**Tranche 4 — Explicitly NOT scheduled (OPTIONAL-OUT).** Batches, Files, containers/MCP, refusal/pause_turn producers, real prompt-cache TTL + cache_creation_input_tokens (I3; revisit only when scheduler exposes radix-write attribution), rate-limit headers, image dimension limits, 402/409.

---

## 4. What NOT to change — deliberate deviations (comments already in code; keep them)

0. **G-05 max_tokens=0 prewarm → clamp-to-1 is the ADJUDICATED FINAL (§5 R1-G05):** a synthesized end_turn response without a real engine pass canNOT prewarm (radix cache is populated only by real engine calls); clamp 0→1 + INFO + `stop_reason="max_tokens"` is semantically closest. Do NOT re-litigate toward synthesized responses.

   Verbatim chair rationale (also embedded at `convert.py:789-799`):
   > "promoting `max_tokens:0` to a successful-but-empty response without engine contact canNOT prewarm — radix cache fills ONLY from a real engine pass; the only sane purposes a client has for max_tokens=0 in a serving stack are (i) cache priming or (ii) accidental input, and (c-empty-response) dishonestly reports done-for-(i). Clamp 0→1 + INFO + forced max_tokens stop_reason costs one token and is HONEST."

1. **Deferred `message_start`** (serving.py:1133–1158). Anthropic emits it immediately with estimated usage; sglang waits for the first payload/usage chunk so `input_tokens` is real. Strictly better for billing-fidelity; SDK-legal (event order unchanged). Do not "fix" to vLLM's immediate emit.
2. **No fabricated thinking signatures** (serving.py:870–896, 1262–1264; protocol.py:100–104 `signature: Optional`). vLLM emits `uuid4().hex` signatures — those *fail* real verification and mislead clients into trusting round-tripped thinking. Absent signature = spec-legal "unsigned thinking". Keep G9 resolved as "never assign".
3. **`redacted_thinking` history → 400** (serving.py:381–382). vLLM silently skips (loses context silently); 400 is loud and honest. Only revisit if Claude-Code-in-the-wild proves it must be tolerated — then skip-with-warning, never fake.
4. **`budget_tokens` / `display:"omitted"` accepted-with-WARNING, not enforced** (serving.py:588–608). No engine knob; rejecting would break lawful SDK callers. Keep.
5. **Server tools skipped + forced-choice-with-only-server-tools → explicit 400** (serving.py:679–717). Silent downgrade (`tool_choice=any` → no tools) would deceive callers; 400 is the correct refusal.
6. **Empty completion → one empty `TextBlock`** (serving.py:1307–1311) instead of spec-legal `content: []`. Strict SDKs index `content[0]`; empty array breaks them. Spec permits both. Keep.
7. **`_flush_on_error` closes open blocks and emits `message_stop` after the error event** (serving.py:942–958). Anthropic's real streams often end *without* message_stop post-error, but strict SDK parsers hang on unclosed blocks. Deviation is intentional and tested.
8. **`anthropic-version` header not validated** (nowhere in http_server; vLLM identical). Tolerance is the documented posture for third-party backends; rejecting unknown versions only breaks pinned clients. At most, log first-seen value.
9. **Billing-header stripping: documentation-only** (`CLAUDE_CODE_ATTRIBUTION_HEADER=0`, docs `anthropic_api.mdx`). Do *not* copy vLLM's server-side strip of `x-anthropic-billing-header` system blocks — it silently desyncs client/server token accounting and confuses count_tokens.
10. **5xx error messages always generic** (serving.py:161–181, 761–766; http_server.py:555–560). Never echo upstream exception text on 5xx. Keep; extend rather than relax.
11. **User turn consisting solely of tool_results emits no user message** (serving.py:541–544); empty assistant turn → `""` placeholder (L545–550). These preserve strict-template role alternation; do not "simplify".
12. **OpenAI shape for `/v1/models` today** — only change via the header-negotiated G-27, never unconditionally (OpenAI clients share the route).

---

## 5. Test plan deltas

- Unit (`test_serving.py`): G-19 matched-stop stream+non-stream; G-17 parallel flag; G-28 format mapping; G-01 middleware branch (auth.py is already unit-test friendly); G-08 pairing 400 matrix; G-12 document degrade; G-05 prewarm; G-07 prefill coercion incl. trailing-ws 400; G-15 generic block; G-25 cache-read w/o flag; G-18 thinking_tokens; G-11 cross-field 400s; G-21 ping via fake-stall stream.
- e2e (`anthropic_messages_kit.py` + `test_anthropic_tool_use.py`): stop_sequences assertion on real server; disable_parallel_tool_use; json_schema format; x-api-key auth against `--api-key` server; Claude-Code-shaped megarequest (system blocks + cache_control + betas + output_config.effort + tool use) round-trip.
- Docs: update `docs/docs/basic_usage/anthropic_api.mdx` per tranche landing (auth notes, prefill, document degrade policy, cache-report default).

## 6. Relation to prior gap ledgers

| Notes (Part C) | This audit |
|---|---|
| G1 stop_sequence | G-19 |
| G2 disable_parallel_tool_use | G-17 |
| G3 output_config.format | G-28 |
| G4 vLLM-compat passthrough fields | OUT (kept OUT; harmless) |
| G5 billing-header strip | §4.9 (do-not-change) |
| G6 usage completeness | G-18 + G-25 (+ I3 recorded OUT-L) |
| G7 anthropic-version | §4.8 (do-not-change) |
| G8 ping | G-21 (upgraded: planned) |
| G9 thinking signatures | §4.2 (do-not-change) |
| G10 route cosmetics | OUT |
| — new in this audit | G-01 auth envelope (BLOCKER), G-12 documents (BLOCKER), G-07 prefill, G-08 pairing, G-15 unknown-block tolerance, G-20 context-window stop reason, G-22/23/24/27, G-05 prewarm, G-03 413, G-02 request_id, G-09 temperature, G-10 service_tier, G-11 thinking 400s, G-13 is_error, G-14 server-tool round-trip, G-16 strict, G-26 cache_control fields |

---

## 5. Post-implementation review deltas (thermonuclear round 1, 2026-08-21)

Recorded after `docs_new/reviews/thermonuclear_round1.md`; supersedes tranche rows
where noted. Each finding was independently verified by the parent session before adoption.

| # | Delta | Resolution |
|---|-------|-----------|
| R1-S1 | **G-19 wiring defect (live):** `_resolve_stop_reason` checked `matched_stop` before the finish_reason map; stream tool-call turns (`openai/serving_chat.py:1653-1660` ships `matched` un-nulled, non-stream 2110/2130 nulls it) could emit `stop_sequence` instead of `tool_use`, and the scheduler-internal sentinel `FINISH_MATCHED_STR(matched="NaN happened")` (`managers/schedule_batch.py:1612`) could leak as a client-visible stop_sequence. | Fix: map finish_reason first; upgrade only `stop`; membership-guard `matched ∈ request.stop_sequences`. Regression tests for both failure modes. |
| R1-S2 | **G-02 split-brain:** serving `_error_response` minted its own id (header≠body); `_scope_request_id` committed dead. | Scope id (middleware-stamped, `ANTHROPIC_REQUEST_ID_SCOPE_KEY`) threaded into `_error_response`; generate-fallback only when scope absent. |
| R1-S3 | **G-24 doubled:** handler 503→529 special case + `AnthropicOverloadedStatusMiddleware`. | Handler special case deleted; middleware single-owner. |
| R1-S4 | **G-03 100k cap doubled and unreachable:** pydantic validation fires during FastAPI dependency solving, so route-level `Depends(...)` cap checks never ran. | Route deps deleted; protocol validator authoritative. |
| R1-S5 | **Error-spec ownership triplicated** (handler map, serving map, middleware bodies); 504 had already diverged. | One canonical `anthropic_error_spec(status)` in `sglang.srt/utils/anthropic_http.py` consumed by handler, middlewares, auth (incl. G-04 `timeout_error` fix). |
| R1-G17 | **Producer census falsified the G-17 gate:** `content_filter` occurs only in `openai/protocol.py` Literal declarations; no sglang path produces it. Gate (a) fails. | REVERT DISPATCHED (turn-3): map branch + 2 tests deleted; `"refusal"` stays in the stop_reason `Literal` as shape-contract only; gate (a) stands for ANY future literal re-expansion. |
| R1-G05 | **FINAL after two-way adjudication: KEEP THE CLAMP.** A synthesized end_turn response without engine contact canNOT prewarm — radix cache fills only from a real engine pass; the only sane purposes for `max_tokens=0` in a serving stack are cache priming or accidental input, and an empty-success response dishonestly reports the former. Clamp 0→1 + INFO + `stop_reason="max_tokens"` costs one token and is honest. (Round-1's early-exit remedy superseded here after the implementer surfaced the authority conflict explicitly — corrected by chair override, 2026-08-21.) |
| R1-SPLIT | **File-size blocker actualized:** serving.py ≈1,989 lines, converter alone 678. | Split before remaining tranche: `anthropic/convert.py` (~750 pure), `anthropic/streaming.py` (~450 translator+watchdog), `anthropic/respond.py` (~350), `serving.py` ~500 orchestration. |
| R1-MIXIN | 21× duplicated `cache_control` field declarations in protocol.py → single pydantic mixin inherited by all block classes; wire schema identical. |
