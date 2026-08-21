# CLAUDE_CODE_NOTES — pointing the real Claude Code CLI at a mock Anthropic server

Harness: `docs_new/harness/mock_anthropic_server.py` (FastAPI+uvicorn, run by
`docs_new/harness/.venv312/bin/python mock_anthropic_server.py --port 8077`).
Transcripts: `docs_new/harness/transcripts/` (`requests.jsonl` +
per-request `NNN_v1_messages.json` pretty dumps incl. the response we sent).
Probe driver: `docs_new/harness/probe.sh`.

**Did `claude -p` succeed against the mock? YES** — repeated runs, exit 0,
clean assistant results, including full tool-use round trips (Claude Code
executed the declared `Bash` tool locally and POSTed the `tool_result` back).
CLI under test: `claude` 2.1.238 (Claude Code), user-agent
`claude-cli/2.1.238 (external, sdk-cli)` (Bun-compiled binary:
a `Bun/1.4.0` agent makes the startup probe).

## 1. Exact env-var recipe (works!)

```bash
export HOME=/tmp/claude-home           # our sandbox $HOME is read-only; pick any writable dir
export ANTHROPIC_BASE_URL=http://127.0.0.1:8077   # plain http on loopback is accepted, no TLS needed
export ANTHROPIC_AUTH_TOKEN=test       # -> header  authorization: Bearer test
# (alternative: ANTHROPIC_API_KEY=x    # -> header  x-api-key: x   — see §3)
export ANTHROPIC_MODEL=mock-claude     # any string; the mock must accept it; unvalidated
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
claude -p 'say hi' --output-format json
```

Notes: `-p` (print mode) skips the workspace trust dialog; fresh HOME avoids
touching host state. `--debug-file` did not materialize a log for me — raw
wire capture via server-side transcripts worked better. A
`[claude-code:unrecognized_model] {"model":"mock-claude",...}` warning line is
printed to **stdout** before the JSON result (parse the LAST stdout line).

## 2. What Claude Code sends — headers (every /v1/messages call)

| header | value observed |
|---|---|
| `anthropic-version` | `2023-06-01` (always) |
| `anthropic-beta` | `claude-code-20250219,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24` |
| `authorization` | `Bearer test` (from ANTHROPIC_AUTH_TOKEN) |
| `anthropic-dangerous-direct-browser-access` | `true` |
| `x-app` | `cli` |
| `x-claude-code-session-id` | per-run UUID |
| `x-stainless-*` | SDK telemetry: `lang: js`, `runtime-version: v26.3.0`, `package-version: 0.112.1`, `retry-count: 0`, `timeout: 600`, `os: Linux`, `arch: x64` |
| `accept` | `application/json`; `accept-encoding: gzip, deflate, br, zstd` |
| `user-agent` | `claude-cli/2.1.238 (external, sdk-cli)` |

At startup it also issues **`HEAD {base}/api/hello`** (user-agent `Bun/1.4.0`).
Our 404 answer was perfectly tolerated (one probe per CLI launch). No
`/v1/messages/count_tokens` call was ever observed. No `/v1/models` call.

## 3. What Claude Code sends — body (POST /v1/messages)

* `model`: the `ANTHROPIC_MODEL` string verbatim — never validated.
* **`stream: true` — ALWAYS**, for main-loop AND auxiliary calls
  (session-title generation too). A non-streaming request only appears as a
  recovery retry (see §5).
* `max_tokens`: **32000** regardless of prompt.
* `system`: **array of text blocks**, each carrying `cache_control: {type:"ephemeral"}`.
* `messages`: user content is a **list of blocks** (system-reminder context
  block + the actual prompt as `{"type":"text","text":...,"cache_control":{"type":"ephemeral"}}`).
  **CRITICAL / non-standard:** `role: "system"` messages appear INSIDE
  `messages[]` (agent-type docs and a `<total_tokens>…</total_tokens>`
  remaining-context notice) — gated by its `mid-conversation-system-2026-04-07`
  beta. Servers with strict role enums will choke on this.
* `thinking`: `{"type":"adaptive","display":"omitted"}`.
* `output_config`: `{"effort":"high"}`.
* `context_management`: `{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}`.
* `metadata.user_id`: JSON string `{device_id, account_uuid:"", session_id}`.
* `tools`: 25 built-ins: Agent, Bash, CronCreate/Delete/List, DesignSync,
  Edit, EnterWorktree/ExitWorktree, NotebookEdit, Read, ReportFindings,
  ScheduleWakeup, SendMessage, Skill, TaskCreate/Get/List/Output/Stop/Update,
  WebFetch, WebSearch, Workflow, Write.
* Default permission mode executes a bare `Bash: echo ...` with **no prompt**
  (safe-command allowlist) — our tool round trip ran fully autonomously.

Tool-result follow-up call: assistant's `tool_use` is mirrored, then a
**role=user message whose content is `[{"type":"tool_result",
"tool_use_id":"toolu_…","content":"<plain string>","is_error":false,
"cache_control":{"type":"ephemeral"}}]`**; `cache_control` also rides on
tool_result blocks.

## 4. Response behaviors observed — acceptance & tolerance (empirical!)

All against `POST /v1/messages` with `stream:true`:

| Server behavior | CLI reaction | Verdict |
|---|---|---|
| Canonical SSE: `message_start→content_block_start/delta/stop→message_delta→message_stop` | consumed, rc=0 | ✅ REQUIRED shape works |
| `tool_use` block + `stop_reason:"tool_use"` (declared tool name) | executes tool, POSTs follow-up with `tool_result` | ✅ round trip works |
| **`thinking` block with NO `signature` field** | accepted; in the NEXT request's history the client re-emits it as `{"type":"thinking","thinking":…,"signature":""}` | ✅ accepted, ⚠️ **client normalizes missing signature to `signature:""` in echoed history — servers MUST tolerate `signature:""` on request-side thinking blocks** |
| `stop_reason:"max_tokens"` | CLI **auto-resumes** with synthetic user msg "Output token limit hit. Resume directly — no apology …" (extra turn) | ✅ handled |
| `stop_reason:"stop_sequence"` + literal `stop_sequence` string | consumed cleanly (rc=0; stop reason surfaced in result envelope) | ✅ handled |
| HTTP 503 `overloaded_error` envelope | **12 identical attempts** (1 + 11 retries, same as 500), then terminal `API Error: 503 …` rc=1 | ⚠️ expect repeated identical POSTs |
| `stop_reason:"refusal"` | terminal: `API Error: <model> can't help with this …` rc=1, `is_error:true` | ✅ handled |
| **No `usage` anywhere** (no key in `message_start.message`, none in `message_delta`) | accepted, rc=0 | ✅ usage optional |
| HTTP 500 `{"type":"error","error":{"type":"api_error",…}}` | **retries the identical request 11×** (client-level loop; `x-stainless-retry-count` stays 0), then terminal `API Error: 500 …` rc=1 | ⚠️ expect repeated identical POSTs |
| `stream:true` answered with plain JSON body (no SSE) | **falls back to a `stream:false` retry** of the same conversation, then continues normally | ⚠️ tolerant, but don't rely on it |
| SSE truncated mid-block (no stop events, clean EOF) | **falls back to a `stream:false` retry**, continues | ⚠️ tolerant |
| `model` echoed ≠ requested (any string) | accepted ("unrecognized_model" stdout warning only) | ✅ unvalidated |
| 404 on `HEAD /api/hello` | ignored | ✅ |

## 5. Checklist — what a server MUST do to keep Claude Code happy
(acceptance criteria for e.g. sglang's Anthropic endpoint)

1. `POST {base}/v1/messages` with **SSE streaming that emits the canonical
   event sequence**: `message_start` (full message skeleton incl.
   `id`,`type:"message"`,`role:"assistant"`,`model`,`content:[]`,`usage`),
   then per block `content_block_start` (indexed) → `content_block_delta`
   (indexed, `text_delta` | `input_json_delta` | `thinking_delta`) →
   `content_block_stop`, closing with `message_delta`
   (`delta.stop_reason` set; `usage.output_tokens`) and `message_stop`.
   Correct **indexed block interleaving** (text/tool_use/thinking in any
   order) is essential.
2. Honor **`stop_reason` semantics**: `end_turn`, `tool_use`, `max_tokens`,
   `refusal` all drive distinct client control flow. `tool_use` must pair
   with at least one `tool_use` content block whose `name` is one of the
   **client-declared tools**.
3. Accept the request envelope **without strict validation surprises**:
   `anthropic-version: 2023-06-01`; long `anthropic-beta` list; auth EITHER
   `authorization: Bearer …` (ANTHROPIC_AUTH_TOKEN) OR `x-api-key`
   (ANTHROPIC_API_KEY); fields `thinking` (`adaptive`/`display"),
   `output_config.effort`, `context_management`, `metadata`, `cache_control`
   on content blocks; **`role:"system"` entries inside `messages[]`**.
4. Accept **list-form** `system` (blocks, possibly with `cache_control`) and
   list-or-string user/assistant `content`, incl. `tool_result` blocks with
   string or block-list `content` and `is_error`.
5. Tolerate **`signature:""` (empty string) on `thinking` blocks in echoed
   history** — Claude Code writes that when generation lacked a signature.
   (Conversely, generation-time thinking blocks MAY omit `signature`;
   the client accepts it.)
6. Make responses robust to **retries**: identical POSTs may repeat up to
   ~11× after 5xx; after malformed SSE the client re-issues with
   `stream:false` — so **non-streaming JSON responses must also be
   implemented correctly** (same content shape, no event stream).
7. Emit parseable JSON envelope `{"type":"error","error":{"type","message"}}`
   with a sensible HTTP status for failures — the CLI surfaces
   `API Error: <status> <message>` to the user.
8. Don't require `anthropic-version`/`anthropic-beta` to gate behavior from
   the server side if you only target the CLI; but DO NOT reject unknown
   headers/fields.
9. Model name: echo or don't — unvalidated. Any `ANTHROPIC_MODEL` string
   flows through.

## 6. Repro commands

```bash
cd /home/ec2-user/sglang/docs_new/harness
../harness/.venv312/bin/python mock_anthropic_server.py --port 8077 &   # or nohup
bash probe.sh 'say hi' 'thinktool' 'refuse' 'go long' 'error500' 'breakjson' 'breaksse' 'nousage'
# inspect: transcripts/requests.jsonl, transcripts/NNN_v1_messages.json
```

Triggers implemented by the mock: plain text echo (default) · "add"/"tool" →
`tool_use` (client-declared `Bash` echo, else toy `add_numbers` when the
client declares it) · tool_result in history → confirmation text ·
"think please" → thinking block w/o signature · "thinktool" → thinking +
tool_use · "refuse" → stop_reason refusal · "go long" → stop_reason
max_tokens · "hit stop word" → stop_reason stop_sequence + literal
stop_sequence string · "nousage" · "error500" · "overload503" · "breakjson" ·
"breaksse".
