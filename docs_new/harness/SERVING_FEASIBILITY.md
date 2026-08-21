# SERVING_FEASIBILITY — can this host serve a real model on CPU?

**Verdict: NO for real weights on this box — but we don't need it. Use the
"real AnthropicServing over real HTTP + fake OpenAIServingChat" mode, which
is BUILT and PROVEN end-to-end with the real Claude Code CLI** (see §4).

## 1. Host facts

| fact | value | implication |
|---|---|---|
| GPU | none (`nvidia-smi` fails) | CUDA path dead |
| CPU | Intel Xeon Platinum 8375C (**Ice Lake**, 4 vCPU) | **no Intel AMX** — flags only `avx512f`; no `amx_*`, not even `avx512_bf16` |
| RAM | 4 cores, small box | 0.5B-class fp32 fits, irrelevant given below |

sglang's CPU support (`docs/docs/hardware-platforms/cpu_server.mdx`) is
explicitly "enabled and optimized on CPUs equipped with Intel® AMX®
instructions, 4th generation or newer Intel® Xeon®" — Ice Lake is 3rd gen.
The CPU engine exists (`SGLANG_USE_CPU_ENGINE=1`, `--device cpu`,
`python/sglang/srt/hardware_backend/cpu/` — currently only a `quantization/`
subdir for w8a8_int8), but its kernels come from an **sgl-kernel CPU build
compiled from source** (`python/sglang/kernels/aot/pyproject_cpu.toml`,
gcc-13 toolchain, oneDNN, libiomp5/tcmalloc/tbb). Prebuilt wheels on PyPI
(`sgl-kernel 0.3.21`, abi3 manylinux x86_64) are **CUDA-oriented**, and the
repo pins `sglang-kernel==0.4.6.post1` (internal naming) — no CPU wheel you
can `pip install`.

## 2. Empirical failure when launching for real (documented, unpatched)

```
PYTHONPATH=python SGLANG_USE_CPU_ENGINE=1 \
  .venv312/bin/python -m sglang.launch_server --device cpu \
  --model-path Qwen/Qwen2.5-0.5B-Instruct --port 8099
```

dies **during ServerArgs resolution**, before any engine/model work:

```
sglang/srt/server_args.py:6562 _handle_context_parallelism
→ sglang/srt/layers/cp/zigzag.py:54
→ sglang/srt/mem_cache/memory_pool.py:45
→ sglang/srt/kernels/ops/kvcache/cache_move.py:10
→ from sgl_kernel import copy_all_layer_kv_cache_cpu
ModuleNotFoundError: No module named 'sgl_kernel'
```

- `maybe_stub_sgl_kernel()` cannot rescue a real launch: the stub is a
  tests-only MagicMock meta-path hook, and sglang spawns fresh
  scheduler/tokenizer **subprocesses** that re-import everything; kernel
  symbols like `copy_all_layer_kv_cache_cpu` are actually *called* there.
- Even if args parsed, model execution needs real sgl_kernel/flashinfer CPU
  kernels; `torch.compile` triton path does not cover sglang custom ops
  (`layers/quantization`, attention backends are `flashinfer`/`fa`/`trtllm*`
  GPU-first; CPU backend expects the AMX kernel build).
- Note on `--attention-backend torch_native`: the backend IS registered
  (`attention_registry.py:188`, `server_args.py:184`) but selecting it cannot
  rescue a CPU launch — the fatal `sgl_kernel` import fires in
  `ServerArgs.__post_init__._handle_context_parallelism` →
  `layers/cp/zigzag.py` → `mem_cache/memory_pool.py` →
  `sglang/kernels/ops/kvcache/cache_move.py`, which is unconditional and runs
  before any attention-backend selection.
- For an AMX-capable box use a 4th-gen+ Xeon EC2 family (Sapphire Rapids:
  m7i/c7i/r7i; Emerald Rapids: c8i/m8i/r8i) and the `xeon` Docker image or
  the source build from the doc (gcc-13, oneDNN, libiomp5/tcmalloc/tbb
  LD_PRELOAD).

**Conclusion: a from-pip real serve on this host is impossible today;
a from-source CPU build is possible in principle but needs an AMX machine
(Spring: use a 4th-gen+ Xeon instance type, e.g. m7i/c7i on EC2) + ~hour-long
sgl-kernel build.**

## 3. Import-level feasibility (cheap checks, all PASS)

With the TASK1 venv (`docs_new/harness/.venv312`, py3.12, torch 2.13.0+cpu):

```
PYTHONPATH=python SGLANG_CACHE_DIR=/tmp/sgl-cache \
  .venv312/bin/python -c "import sglang.srt.entrypoints.http_server"   # OK
```

- `sglang.srt.entrypoints.anthropic.serving` — OK
- `sglang.srt.entrypoints.http_server` — OK (only extra pkg needed: `jsonschema`)
- `python -m sglang.launch_server --help`-equivalent — FAILS at sgl_kernel above

## 4. VERDICT — e2e mode to standardize on

### ✅ Chosen: "sglang-harness" mode (real Anthropic code + fake engine + real HTTP)
Implemented in `docs_new/harness/sglang_anthropic_harness.py` (port 8078):
- REAL `sglang.srt.entrypoints.http_server.anthropic_v1_messages` /
  `anthropic_v1_count_tokens` route functions and `validate_json_request`
  dependency and `RequestValidationError` handler (Anthropic error envelope).
- REAL `AnthropicServing` doing request validation, Anthropic→OpenAI
  conversion, SSE event translation, error envelopes.
- Scratch `E2EFakeOpenAIServingChat` (duck-type surface enumerated from
  `serving.py`: `tokenizer_manager` (with `create_abort_task`),
  `apply_reasoning_enabled`, `wrap_reasoning_history`, `_validate_request`,
  `_convert_to_internal_request`, `_handle_non_streaming_request`,
  `_generate_chat_stream`, `_process_messages`).
- Only the engine + FastAPI `lifespan` (which needs the tokenizer-manager
  process) are bypassed; no sglang file modified.

**Validation results (real Claude Code CLI → this harness):**
- `claude -p 'hello from claude'` → rc=0; full **tool round trip**: sglang
  emitted tool_use(Bash echo) → CLI executed → tool_result returned via sglang
  conversion to `role=tool` (roles seen after conversion:
  `['system','user','assistant','tool']`) → final answer `E2E-OK …
  Marker=MOCK_SGLANG_E2E_42`. transcripts: `docs_new/harness/transcripts_sglang/`.
- `claude -p 'think please'` → real sglang **thinking blocks** consumed by the
  CLI (rc=0). Claude Code's `thinking:{type:adaptive,display:omitted}` is
  accepted; real code logs ``Anthropic thinking.display='omitted' is accepted …``.
- `output_config.effort=high` (sent by the CLI) → converted to
  `reasoning_effort='high'` by REAL conversion code. Embedded
  `role:'system'` messages inside `messages[]` pass protocol validation +
  conversion (protocol.py on this branch was extended for them by the
  concurrent session).
- `/v1/messages/count_tokens` real handler works when the fake supplies
  `prompt_ids` (returns `{"input_tokens":N}`); Claude Code never calls it.

**Why this is the right acceptance harness:** it exercises the exact code
paths the unit tests cover (`test_serving.py`) — protocol models, conversion,
streaming translation, error envelopes, stop-reason mapping — with the REAL
wire client (Claude Code), without needing GPU/AMX/kernels. When a GPU or
AMX box becomes available, the same probes (`probe.sh`, `probe_sglang.sh`)
can be re-pointed at a real `sglang serve` unchanged.

### 🟡 Supporting mode: standalone mock (`mock_anthropic_server.py`, port 8077)
Pure-FastAPI reference server from TASK2 — keep it as (a) a *known-good*
contrast target when debugging sglang behavior, (b) the place where malformed
envelopes (breakjson/breaksse/error500) probe CLIENT tolerance, and
(c) documentation of exact CLI wire format (`transcripts/`).

### ❌ Rejected: real-weight CPU serve on this host — see §1–2.
Fallback if a real model is ever mandatory on x86 without AMX: none clean;
llama.cpp/vLLM-CPU could serve OpenAI but would bypass all sglang anthropic
code being validated.

## 5. Commands

```bash
# harness (chosen mode)
cd /home/ec2-user/sglang/docs_new/harness
nohup env PYTHONPATH=/home/ec2-user/sglang/python SGLANG_CACHE_DIR=/tmp/sgl-cache-e2e \
  ../harness/.venv312/bin/python sglang_anthropic_harness.py --port 8078 \
  > transcripts_sglang/server.log 2>&1 &
bash probe_sglang.sh 'hello from claude' 'think please'

# reference mock
nohup ../harness/.venv312/bin/python mock_anthropic_server.py --port 8077 \
  > transcripts/server.log 2>&1 &
bash probe.sh 'say hi' 'thinktool' 'refuse' 'go long' 'error500' 'breakjson'
```
Note: each sandbox shell gets a fresh /tmp; kill patterns must avoid matching
the launching shell (`pkill -f "sglang_anthropic_ha[r]ness"` in a *separate*
call from any command mentioning the server path).
