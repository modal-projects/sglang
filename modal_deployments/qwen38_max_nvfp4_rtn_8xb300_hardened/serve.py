"""Qwen3.8-Max NVFP4-RTN on one 8xB300 node, TP8: hardened static-scales build.

Hardened variant of ../qwen38_max_nvfp4_rtn_8xb300 (same checkpoint, same
server shape): three python-only sglang engine patches (see patches/ -- mamba
non-finite radix-cache guard, mamba pool sanitize on flush/reset and extend
admission, GDN per-sequence initial-state gate), NVMe weight staging, and a
warmup contract that gates readiness on response CONTENT (HTTP 200 alone does
not prove health here).

Validated 2026-08-08: GPQA-Diamond 179/198 (90.4%) and 177/198 (89.4%) across
two complete runs, zero request errors (198 questions, zero-shot CoT
"Answer: <letter>", temperature 0.6, top_p 0.95, max_tokens 32768,
concurrency 8).

NEVER enable per-token activation quant on this build:
SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION is hard-set to 0. Per-token
requires flashinfer kernel patches this config does not carry (without them
it NaN-collapses 100% of outputs) and has an open /flush_cache poisoning
issue even patched. That experimental stack, and the full investigation
writeup, live in the modal-share bundle qwen38-nvfp4-zero-guard/
(http://modal-share.tail5292b.ts.net/files/qwen38-nvfp4-zero-guard/).

To deploy: set APP_NAME below (it ships unset so a checked-in copy can never
redeploy someone else's app), then run (runc is required for sane
weight-staging and load times):

  MODAL_FUNCTION_RUNTIME=runc MODAL_PROFILE=modal-labs MODAL_ENVIRONMENT=qwen-bringup \
    uv run modal deploy modal_deployments/qwen38_max_nvfp4_rtn_8xb300_hardened/serve.py
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

MINUTES = 60
PORT = 8000

TP_SIZE = 8
GPU = "B300:8"

# NVFP4 routed experts (92 main layers), BF16 everything else incl. the
# mtp.* head; needs 8xB300 (2.3 TB).
MODEL_MOUNT = "/model"
MODEL_PATH = f"{MODEL_MOUNT}/Qwen3.8-Max-NVFP4-RTN-v2"
SERVED_MODEL_NAME = "qwen3.8-max-nvfp4-rtn"

# 4h: the happy path (NVMe-staged, runc) boots in ~20-30 min, but volume
# throughput degrades under concurrent readers -- volume-direct reads have
# been observed at ~70 s/shard, which blows a 90-min budget mid-load -- and
# gVisor roughly halves staging throughput. 4h covers the degraded cases.
STARTUP_TIMEOUT = 240 * MINUTES

# Set your own app name, e.g. "qwen38-max-nvfp4-rtn-8xb300-hardened-<you>".
# Ships unset so deploying this file as-is can never redeploy a live app.
APP_NAME = None
if not APP_NAME:
    raise SystemExit("Edit APP_NAME in this file before deploying.")

PATCH_DIR = Path(__file__).parent / "patches"

# Three-patch engine hardening (python-only, applied to the image's
# /sgl-workspace/sglang tree). Each patch is grep-verified so a
# silently-failed hunk fails the image build instead of shipping an
# unhardened engine.
serving_image = (
    modal.Image.from_registry("lmsysorg/sglang:nightly-dev-cu13-20260806-ae5f8c94")
    .add_local_file(
        str(PATCH_DIR / "sglang_mamba_nonfinite_guard.patch"),
        "/opt/sglang_mamba_nonfinite_guard.patch",
        copy=True,
    )
    .add_local_file(
        str(PATCH_DIR / "sglang_mamba_pool_sanitize.patch"),
        "/opt/sglang_mamba_pool_sanitize.patch",
        copy=True,
    )
    .add_local_file(
        str(PATCH_DIR / "sglang_gdn_initial_state_gate.patch"),
        "/opt/sglang_gdn_initial_state_gate.patch",
        copy=True,
    )
    .run_commands(
        "bash -c 'command -v patch >/dev/null || (apt-get update && apt-get install -y patch)'",
        # never cache non-finite mamba state into the radix tree
        "cd /sgl-workspace/sglang && "
        "patch -p1 --forward < /opt/sglang_mamba_nonfinite_guard.patch",
        "grep -q 'slots_state_is_finite' "
        "/sgl-workspace/sglang/python/sglang/srt/mem_cache/memory_pool.py",
        "grep -q '_MAMBA_NONFINITE_CACHE_GUARD' "
        "/sgl-workspace/sglang/python/sglang/srt/mem_cache/mamba_radix_cache.py",
        # sanitize mamba state slots on flush/reset + zero non-finite slots at
        # extend admission (the /flush_cache -> poisoned-fresh-request carrier;
        # a cold boot is an implicit flush)
        "cd /sgl-workspace/sglang && "
        "patch -p1 --forward < /opt/sglang_mamba_pool_sanitize.patch",
        "grep -q 'def sanitize_all' "
        "/sgl-workspace/sglang/python/sglang/srt/mem_cache/memory_pool.py",
        "grep -q maybe_sanitize_slots_nonfinite "
        "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py",
        # structural fix: fresh sequences never read slot memory (per-seq
        # has_initial_state mask down to the fla chunk kernel)
        "cd /sgl-workspace/sglang && "
        "patch -p1 --forward < /opt/sglang_gdn_initial_state_gate.patch",
        "grep -rq 'USE_H0_MASK' /sgl-workspace/sglang/python/sglang/kernels/",
    )
    .uv_pip_install("autoinference-utils==0.2.2")
    .env({
        "HF_HUB_OFFLINE": "1",
        # STATIC activation scales, hard-set: per-token dynamic quant must NOT
        # be enabled on this build (see the module docstring).
        "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION": "0",
        "SGLANG_TIMEOUT_KEEP_ALIVE": "300",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    })
)

SERVER_ARGS = {
    "--served-model-name": SERVED_MODEL_NAME,
    "--attention-backend": "trtllm_mha",
    "--chunked-prefill-size": "16384",
    "--context-length": "131072",
    "--cuda-graph-max-bs": "16",
    "--dist-timeout": "3600",
    "--dtype": "bfloat16",
    "--max-running-requests": "16",
    "--mem-fraction-static": "0.85",
    "--moe-runner-backend": "flashinfer_trtllm",
    "--quantization": "modelopt_fp4",
    "--trust-remote-code": "",
}


def stage_weights_to_disk(src: str, dst_root: str = "/tmp/model-weights") -> str:
    """Copy the checkpoint from the volume to local NVMe before loading.

    The model loader reads shards with many small random accesses, which is
    slow through the volume FUSE mount (~70 s/shard observed on degraded
    volume-direct loads). Large parallel sequential copies pull the same
    bytes at full volume throughput (3-8 GB/s observed on runc; gVisor runs
    ~2-3x slower); loading then runs at local-disk speed. Idempotent per
    file (size check), so a retried container resumes where the last one
    stopped.
    """
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor

    threads = 32
    dst = os.path.join(dst_root, os.path.basename(src))
    entries = []
    for root, _, files in os.walk(src):
        for f in files:
            s = os.path.join(root, f)
            d = os.path.join(dst, os.path.relpath(s, src))
            entries.append((s, d, os.path.getsize(s)))
    total = sum(sz for _, _, sz in entries)
    t0 = time.time()
    progress = {"done": 0, "last_print": 0.0}

    def copy_one(entry):
        s, d, sz = entry
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if not (os.path.exists(d) and os.path.getsize(d) == sz):
            tmp = d + ".part"
            shutil.copyfile(s, tmp)
            os.rename(tmp, d)
        progress["done"] += sz
        now = time.time()
        if now - progress["last_print"] > 30:
            progress["last_print"] = now
            gb = progress["done"] / 1e9
            print(
                f"[stage] {gb:.0f}/{total / 1e9:.0f} GB"
                f" ({gb / max(now - t0, 1):.2f} GB/s)",
                flush=True,
            )

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(copy_one, entries))
    dt = time.time() - t0
    print(
        f"[stage] staged {total / 1e9:.0f} GB in {dt:.0f}s"
        f" ({total / 1e9 / max(dt, 1):.2f} GB/s)",
        flush=True,
    )
    return dst


def assert_warmup_content(port: int, tries: int = 4) -> None:
    """Gate readiness on CONTENT: the server must produce coherent text on the
    historically-poisoned shapes. An HTTP 200 alone proves nothing here -- a
    NaN-poisoned server happily returns 200 with "!!!!" bodies, so a
    status-code warmup marks a broken replica ready. Design constraints
    learned from failed boots:
    - initial singles are EXPECTED poison victims (observe, never assert);
    - identical prompts get radix-cached, so a poisoned generation replays on
      every retry of the same wording -> every request uses a unique nonce;
    - never scrub via /flush_cache: on unpatched builds flushing returns pool
      memory to the free list unsanitized and re-triggers the poisoning
      (reproduced 2/2); scrub with fresh generation bursts instead;
    - requests must be genuinely sequential (pacing sleeps), and connection
      errors must not consume assertion budget (gate on a completed request).
    """
    import concurrent.futures
    import json
    import time
    import urllib.request
    import uuid

    def one(prompt: str, max_tokens: int = 24) -> str:
        payload = {
            "model": SERVED_MODEL_NAME,
            "messages": [
                {"role": "user", "content": f"[{uuid.uuid4().hex[:8]}] {prompt}"}
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            body = json.loads(r.read())
        text = body["choices"][0]["message"]["content"] or ""
        print(f"[warmup] resp: {text.strip()[:20]!r}", flush=True)
        return text

    def healthy(text: str) -> bool:
        stripped = text.strip()
        return bool(stripped) and set(stripped) != {"!"}

    # Phase 0: wait until the server actually completes one request (503s
    # here never count against anything).
    t0 = time.time()
    while True:
        try:
            one("Reply with a single word.")
            break
        except Exception as e:  # noqa: BLE001
            if time.time() - t0 > 900:
                raise RuntimeError(f"server never served a completion: {e!r}") from e
            print(f"[warmup] not accepting requests yet ({e!r}); retrying", flush=True)
            time.sleep(5)

    # Phase 1: diagnostic singles (expected victims) then a scrub burst.
    observed_poison = False
    pre = [one("Say the number one.") for _ in range(2)]
    if any(not healthy(r) for r in pre):
        observed_poison = True
        print(
            "[warmup] scrub: poisoned singles observed (expected on fresh boot)",
            flush=True,
        )
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        list(ex.map(one, ["Name one color."] * 4))
    time.sleep(2)
    print(f"[warmup] scrub phase done (poison observed: {observed_poison})", flush=True)

    # Phase 2: asserted. Unique nonces per attempt; re-burst between attempts
    # so a cached poisoned generation can never replay.
    last_err = None
    for attempt in range(tries):
        try:
            singles = []
            for _ in range(2):
                singles.append(one("Say the number two."))
                time.sleep(1)
            with concurrent.futures.ThreadPoolExecutor(4) as ex:
                burst = list(ex.map(one, ["Name one animal."] * 4))
            time.sleep(1)
            post = [one("Reply with exactly OK.")]
            results = singles + burst + post
            bad = [r for r in results if not healthy(r)]
            if not bad:
                print(
                    f"[warmup] content OK across {len(results)} post-scrub responses"
                    f" (boot poison observed: {observed_poison})",
                    flush=True,
                )
                return
            last_err = f"degenerate POST-SCRUB responses ({len(bad)}/{len(results)})"
            print(f"[warmup] attempt {attempt + 1}: {last_err}", flush=True)
            with concurrent.futures.ThreadPoolExecutor(8) as ex:
                list(ex.map(one, ["Count to three."] * 8))
            time.sleep(2)
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)
            print(f"[warmup] attempt {attempt + 1} failed: {last_err}", flush=True)
            time.sleep(5)
    raise RuntimeError(f"warmup content assertion failed: {last_err}")


app = modal.App(name=APP_NAME)


@app.server(
    image=serving_image,
    gpu=GPU,
    cpu=32,
    memory=262144,
    # NVMe scratch for the staged checkpoint (~1.5 TB) with headroom
    ephemeral_disk=2 * 1024 * 1024,
    min_containers=1,
    max_containers=1,
    scaledown_window=10 * MINUTES,
    port=PORT,
    routing_region="us-west",
    unauthenticated=True,
    # 120s: in-flight long generations get a chance to finish on scaledown
    exit_grace_period=120,
    startup_timeout=STARTUP_TIMEOUT,
    target_concurrency=16,
    volumes={
        MODEL_MOUNT: modal.Volume.from_name("qwen38-max-nvfp4-rtn"),
    },
)
class Server:
    @modal.enter()
    def startup(self):
        from autoinference_utils.endpoint import SGLangEndpoint

        local_model_path = stage_weights_to_disk(MODEL_PATH)
        print(f"Starting SGLang with server args: {SERVER_ARGS}")
        self.endpoint = SGLangEndpoint(
            model_path=local_model_path,
            worker_port=PORT,
            tp=TP_SIZE,
            extra_server_args=SERVER_ARGS,
            health_timeout=STARTUP_TIMEOUT,
            health_poll_interval=10.0,
        )
        self.endpoint.start()
        # Warmup contract: assert CONTENT on the worst-case shapes --
        # sequential singles (bs=1 decode + tiny prefill graphs) AND a
        # concurrent burst (batched graphs) -- before serving.
        assert_warmup_content(port=PORT)
        print(f"{SERVED_MODEL_NAME} ({GPU}) is ready.")

    @modal.exit()
    def stop(self):
        if hasattr(self, "endpoint"):
            self.endpoint.stop()
