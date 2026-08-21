# E2E acceptance runbook — Claude Code × sglang /v1/messages

Goal: prove the gap-fixed Anthropic Messages API with the REAL `claude` CLI
(claude-cli/2.1.238 verified). Two serving modes; pick per
`SERVING_FEASIBILITY.md` verdict (or run both, MODE=A then MODE=B).

## Env recipe (from CLAUDE_CODE_NOTES.md §1 — empirically verified)

```bash
export HOME=/tmp/claude-home                 # host $HOME is read-only
export ANTHROPIC_BASE_URL=http://127.0.0.1:9077
export ANTHROPIC_AUTH_TOKEN=test             # exercises Bearer path
export ANTHROPIC_MODEL=<served-model-name>
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
# G-01 check variant: use ANTHROPIC_API_KEY=test instead (exercises x-api-key)
```

## MODE A — real model, real sglang server

⛔ VERDICT (SERVING_FEASIBILITY.md): infeasible on THIS box. Host is Xeon
8375C Ice Lake (no AMX); sglang's CPU path requires AMX (4th-gen Xeon+) +
source-built sgl-kernel CPU (oneDNN); pip sgl-kernel wheels are CUDA-only;
`--device cpu` fails at ServerArgs resolution (`No module named 'sgl_kernel'`,
memory_pool→cache_move). Re-attempt on an AMX/GPU box with the protocol
unchanged; until then MODE B is authoritative.

```bash
cd /home/ec2-user/sglang
source <ServingFeasibility-recipe>            # venv + cache redirects (SGLANG_CACHE_DIR=/tmp/sgl-cache ...)
.venv/bin/python -m sglang.launch_server --model-path <tiny-instruct-model> \
  --device cpu --host 127.0.0.1 --port 9077 --attention-backend torch_native \
  --disable-radix-cache --skip-tokenizer-init=false
# wait for /health 200
```

## MODE B — real AnthropicServing, scripted OpenAIServingChat (no model) ✅ BUILT + PROVEN

`docs_new/harness/sglang_anthropic_harness.py` (b87b55cb): REAL http_server
route functions + validate_json_request + validation-error envelope + REAL
AnthropicServing over a scripted fake chat; transcripts under
`docs_new/harness/transcripts_sglang/`. Running instance: port 8078
(**restart it after ANY serving.py/protocol.py change — imports bind at
server start**). Probes: `docs_new/harness/probe_sglang.sh`.
Verified live so far: streaming canon, autonomous 2-turn tool round trip
(marker MOCK_SGLANG_E2E_42 executed by CLI via Bash), refusal surface.
Matrix items 1,2,3,5,6,10 ≈ passing per agent note — REMAINING sections
verified post-batch-2 (4,7,8,9,11,12,13 + signature-echo explicit).

## Acceptance matrix (assert ALL from wire transcripts — MockServer-style)

| # | Scenario | Expected |
|---|---|---|
| 1 | `claude -p 'say hi'` | rc=0, SSE canonical order message_start→blocks→message_delta(stop_reason)+message_stop; ping right after message_start (G-21) |
| 2 | tool round trip ("run echo") | CLI executes declared tool, POSTs tool_result with cache_control tolerated, final answer rc=0 |
| 3 | thinking stream (reasoning model or scripted) | thinking blocks WITHOUT signature accepted; echoed history w/ `signature:""` tolerated (no 400) |
| 4 | stop-sequence finish | message_delta carries stop_reason="stop_sequence" + stop_sequence=<str> (G-19) |
| 5 | max_tokens=32000 ceiling scripted to cut off | CLI auto-resume turn works (CC retry semantics) |
| 6 | refusal mapped from content_filter | CLI surfaces graceful refusal (rc=1, is_error:true) |
| 7 | x-api-key auth (server --api-key test) | request passes; denial returns Anthropic envelope 401 (G-01) |
| 8 | request-id response header present on /v1/messages* | G-02 (hdr part) |
| 9 | 529 on overload-simulated 503 | G-24 |
| 10 | system-in-messages roles (CC sends them) + cache_control blocks | 200, no validation error (G-26/G-15 tolerance) |
| 11 | prefill: seed trailing assistant turn | continues coherently; no trailing-assistant→user inversion (G-07) |
| 12 | document block in tool_result.content | accepted & degraded, tool loop survives (G-12) |
| 13 | output_config.format json_schema (tool-free Q) | constrained JSON response parses per schema (G-28) |

Method: run probes through a recording shim (reuse mock_anthropic_server.py's
transcript logging as a PROXY in front of the real server, or tcpdump-lite via
a logging FastAPI middleware). Save transcripts under docs_new/harness/transcripts_e2e/.

## Sign-off

- [ ] matrix 1–13 pass (or each failure root-caused to engine-vs-adapter)
- [ ] unit suite green: test_serving.py full run
- [ ] diff review signed off vs anthropic_review_checklist.md + audit §2/§4
