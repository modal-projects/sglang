# TASK 1 — CPU unit-test baseline for the Anthropic entrypoint tests

## FINAL snapshot (2026-08-21 05:10 UTC, tree incl. concurrent-session edits up to 05:02):

**73 passed + 8 subtests passed, 0 failed, ~7.3 s.**

## Result (earlier snapshot: 2026-08-21 ~04:55 UTC)

```
PYTHONPATH=/home/ec2-user/sglang/python \
SGLANG_CACHE_DIR=/tmp/sgl-cache XDG_CACHE_HOME=/tmp/sgl-cache \
HF_HOME=/tmp/sgl-cache/hf TORCHINDUCTOR_CACHE_DIR=/tmp/sgl-cache/inductor \
TRITON_CACHE_DIR=/tmp/sgl-cache/triton \
/home/ec2-user/sglang/docs_new/harness/.venv312/bin/python -m pytest \
  /home/ec2-user/sglang/test/registered/unit/entrypoints/anthropic/test_serving.py
```

**Outcome: 71 passed + 8 subtests passed, 0 failed, ~7.4 s** (collection + run both work).

⚠️ CONCURRENCY WARNING: this repo is being edited live by *another* Claude
session (PIDs visible: interactive `claude --dangerously-skip-permissions`
processes; it created `/.venv-tests/`, `/.cache-local/`, `/.pip-cache/` and
modified, within minutes of my run: `python/sglang/srt/entrypoints/anthropic/protocol.py`
(04:52), `.../serving.py` (04:53), `test/.../anthropic/test_serving.py`
(04:54, grew 1576→1861 lines)). An earlier run of mine on the mid-edit tree
gave 61 passed / 1 failed (`test_stop_reason_content_filter_falls_back_with_warning`,
`content_filter`→`refusal` mapping); after that session's `serving.py` edit
landed, the full file passes. Treat counts as a time-stamped snapshot, not a
stable baseline, until the other session settles.
Git state at snapshot: HEAD = `6127d1dae` + uncommitted modifications to the
three files above (not made by me; I made zero source edits).

## Interpreter choice — why NOT the system python

- `/usr/bin/python` and `/usr/bin/python3` are **3.9.25**; `python/pyproject.toml`
  declares `requires-python = ">=3.10"` and this is a *hard* runtime
  requirement, not just metadata: `sglang/srt/environ.py:124` evaluates
  `str | None` at class-body execution → `TypeError: unsupported operand
  type(s) for |: 'type' and 'NoneType'` on 3.9 (verified empirically).
  `python3.10` is NOT installed; `python3.11` (3.11.13) and `python3.12`
  (3.12.12) are at `/usr/bin/python3.11|3.12` but have **no pip**.
- `~/.local` and `~/.cache` are **READ-ONLY** on this host (pip `--user`
  installs and the default uv cache both fail with EROFS). The pre-provisioned
  py3.9 user-site (torch 2.8.0+cu128, fastapi, pydantic, uvicorn) is therefore
  unusable for installing missing pieces, and it is py3.9-only anyway.

## Chosen solution: uv venv, CPython 3.12, CPU-only torch

Env: `export UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy`

```
~/.local/bin/uv venv --python /usr/bin/python3.12 docs_new/harness/.venv312
# CUDA-free torch pair from the pytorch CPU index (saves ~4 GB vs PyPI CUDA wheels):
~/.local/bin/uv pip install --python docs_new/harness/.venv312/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch torchvision
~/.local/bin/uv pip install --python docs_new/harness/.venv312/bin/python \
  pytest pytest-asyncio triton orjson psutil pybase64 aiohttp prometheus-client \
  anthropic "openai>=1.66" fastapi uvicorn jinja2 tqdm requests IPython \
  transformers==5.12.1 msgspec pyzmq dill xgrammar==0.2.1 compressed-tensors \
  gguf datasets einops partial_json_parser sentencepiece soundfile tiktoken \
  xxhash zstandard blobfile distro python-multipart setproctitle interegular \
  uvloop nvidia-ml-py
```

Resolved key versions: torch 2.13.0+cpu, torchvision 0.28.0+cpu, triton 3.7.1,
transformers 5.12.1 (pinned by sglang; **5.15.1 breaks** — sglang's
`configs/__init__.py` re-registers `qwen3_asr`, which transformers ≥5.15
already ships → `ValueError: 'qwen3_asr' is already used`), xgrammar 0.2.1
(pinned; needed unconditionally by `function_call/inkling_detector.py`),
pytest 9.1.1, openai 3.3.1, anthropic 1.0.0, fastapi 0.141.1.

### Import blockers hit and resolved (in order)
1. Torch JIT/inductor caches target `~/.cache/sglang` (read-only) → redirect via
   `SGLANG_CACHE_DIR` + friends (see invocation above). This is needed because
   `sglang.srt.environ.redirect_third_party_caches()` points triton/inductor/
   HF caches under it.
2. `IPython` (unconditional, `sglang/utils.py`) → installed.
3. `transformers` (via `srt/configs`) → installed, pinned 5.12.1.
4. `msgspec` (via `srt/kernels`) → installed.
5. `zmq` (via `srt/utils/network.py` ← `parallel_state`) → pyzmq installed.
6. `dill` (via `sampling/custom_logit_processor.py`) → installed.
7. `xgrammar` (via `function_call/inkling_detector.py`, NOT guarded) → installed 0.2.1.
8. `compressed_tensors` (via `layers/quantization`) → installed.
9. `gguf` (via `layers/quantization/gguf.py`) → installed.
10. `datasets` (via `sglang/test/test_utils.py` → benchmark chain) **at pytest
    collection time** → installed.

No unresolved import blockers remain. `sgl_kernel` is absent by design
(`maybe_stub_sgl_kernel()` in `sglang/test/test_utils.py` stubs it on GPU-less
hosts — no patching required; the test file calls it before the heavy imports).

Benign noise: `Only CUDA, HIP and XPU support AWQ` / `Only CUDA, MUSA and NPU
support GGUF` UserWarnings on a CPU-only box; torch 2.13 `torch.jit` and
sglang `max_tokens` DeprecationWarnings.

## Repro check (fresh shell)

```
cd /home/ec2-user/sglang
docs_new/harness/.venv312/bin/python -V   # Python 3.12.12
PYTHONPATH=python SGLANG_CACHE_DIR=/tmp/sgl-cache \
  docs_new/harness/.venv312/bin/python -c \
  "from sglang.test.test_utils import maybe_stub_sgl_kernel; maybe_stub_sgl_kernel();
   import sglang.srt.entrypoints.anthropic.serving as m; print('OK', m.AnthropicServing)"
```
