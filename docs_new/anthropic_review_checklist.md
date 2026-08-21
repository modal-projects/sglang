# Anthropic Messages API — Diff Acceptance Criteria

Author: orchestrator. Used to review implementation-agent diffs against
`docs_new/anthropic_messages_api_spec.md` and the goal's gap list.
Each item: what MUST hold + how to verify (unit test / code inspection).

## G1 stop_sequence propagation
- [ ] `FINISH_MATCHED_STR`-style chunk (finish_reason="stop", matched_stop="<str>")
  → streaming `message_delta.delta.stop_reason == "stop_sequence"` AND
  `stop_sequence == <str>`; non-streaming response same fields.
- [ ] `FINISH_MATCHED_TOKEN`-style (matched_stop=int) → `end_turn`,
  `stop_sequence` ABSENT (exclude_none).
- [ ] matched_stop=None with "stop" → `end_turn`.
- [ ] "tool_calls" finish closest to correctness: tool_use blocks closed,
  stop_reason "tool_use" (existing behavior preserved).
- [ ] "length" → "max_tokens" (unchanged). Do NOT synthesize
  `model_context_window_exceeded` — backend has no distinct signal today.

## G-stopreasons protocol literals
- [ ] `AnthropicMessagesResponse.stop_reason` Literal ⊇ {end_turn, max_tokens,
  stop_sequence, tool_use, model_context_window_exceeded, refusal, pause_turn}.
- [ ] Same for `AnthropicMessageEndDelta.stop_reason`.
- [ ] serving STOP_REASON_MAP or helper: "content_filter" → "refusal"
  (was: end_turn + warning); existing test expectation must be UPDATED
  (test_stop_reason_content_filter_falls_back_with_warning → renamed/rewritten).
- [ ] unknown/abort → end_turn + WARNING (unchanged fallback).
- [ ] pause_turn NEVER emitted by current code paths (server-tool loops only).

## G2 disable_parallel_tool_use
- [ ] AnthropicToolChoice gains `disable_parallel_tool_use: Optional[bool]`.
- [ ] True → `chat_request.parallel_tool_calls = False`; False/None → untouched
  (default True). Applies on every tool_choice branch incl. implicit auto.

## G3 output_config.format.json_schema
- [ ] New `AnthropicOutputFormat` model: {type: "json_schema", schema: {...}};
  accepts SDK alias `json_schema` key (populate_by_name or validation AliasChoices).
- [ ] serving maps to `chat_request.response_format = ResponseFormat(
  type="json_schema", json_schema=JsonSchemaResponseFormat(name=..., schema=<dict>))`.
- [ ] Only type=="json_schema" today; unknown format types → 400 invalid_request.

## G5 SSE ping
- [ ] Exactly one PingEvent emitted immediately after message_start on every
  streaming response (incl. deferred message_start path and error flush path:
  NO ping needed on errors — only on successful start; spec-conformant either
  way, but must not break event ordering).

## G6 max_tokens==0 (prewarm)
- [ ] Validator accepts 0; conversion clamps to 1 with a comment+log line.
  (Skip if spec says min is 1 — confirm against spec §2 in implementation.)

## Existing behavior invariants (must not regress)
- [ ] All pre-existing ~55 tests pass (content_filter test exempt, see above).
- [ ] Usage math untouched: input = prompt − cached; cache_read fields omission
  when absent; streaming message_delta omits input_tokens.
- [ ] No thinking signatures fabricated; signature_delta only when real.
- [ ] Error path: 5xx bodies scrubbed; error.type restricted to Anthropic enum.
- [ ] event: lines present on every SSE frame; [DONE] consumed internally,
  never forwarded.

## Tests
- [ ] One test per new behavior above (both stream + non-stream for G1).
- [ ] Full file passes on .venv-tests (python 3.11):
  `PYTHONPATH=python .venv-tests/bin/python -m pytest test/registered/unit/entrypoints/anthropic/test_serving.py -q`

## E2E acceptance (the deliverable)
- [ ] Claude Code CLI `claude -p` round-trip against sglang serving the
  Anthropic API (mode per SERVING_FEASIBILITY verdict), streaming + at least
  one tool_use/tool_result cycle observed in transcripts.
