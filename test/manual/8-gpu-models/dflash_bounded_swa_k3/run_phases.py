"""Run the bounded-SWA DFLASH benchmark phases against a Kimi K3 server.

Runs inside the Modal container. Launches one sglang server for the given
source tree and --max-running-requests, then runs the requested phases against
it and writes one JSON file per phase under --out:

  phase0.json  memory: startup log lines (KV pool sizes, bounded draft pool,
               max_total_num_tokens, free memory) and /get_server_info
  phase1.json  ideal-case hit cost: the same prompt three times at 8k/32k/128k
               tokens; TTFT, cached tokens and prefill tokens per request
  phase2.json  adverse cases: chat with short turns, agentic loop with long
               outputs; TTFT and prefill tokens per turn
  phase3.json  throughput sweep with sglang.benchmark.serving on a multi-turn
               shared-prefix workload at several concurrencies
  phase4.json  the same sweep on an Instinct-shaped workload

Every request is sent with temperature 0. TTFT is measured client side as
time to the first streamed chunk. Prefill work is attributed from the server's
"Prefill batch" log lines emitted while the request was in flight (the client
is single-stream in phases 1 and 2, so the attribution is exact).
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

SERVER_PORT = 30000
SERVER = f"http://127.0.0.1:{SERVER_PORT}"
TARGET_MODEL = "moonshotai/Kimi-K3"
DRAFT_MODEL = "modal-labs/Kimi-K3-DFlash"
# The weights volume is an HF hub cache without refs/ for the target, so offline
# name resolution fails; point the server at the snapshot directories directly.
HF_HUB = (
    os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME", "/hf-cache") + "/hub"
)
TARGET_MODEL_PATH = (
    f"{HF_HUB}/models--moonshotai--Kimi-K3/snapshots/"
    "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
)
DRAFT_MODEL_PATH = (
    f"{HF_HUB}/models--modal-labs--Kimi-K3-DFlash/snapshots/"
    "c192d15a43407bf758b5ae0880d5c72052fef1de"
)
TARGET_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
DRAFT_REVISION = "c192d15a43407bf758b5ae0880d5c72052fef1de"
LOCAL_MODELS = "/tmp/k3_models"


def materialize_model_dir(snapshot: str, repo: str, revision: str, dest: str) -> str:
    """Build a complete model directory from a possibly incomplete hub snapshot.

    Every entry of the snapshot that resolves (weights, configs, code) is
    linked into `dest`; entries whose blob is missing from the volume are
    fetched from the hub at the pinned revision. The volume itself is never
    written.
    """
    # Small files are copied, not linked: transformers resolves a remote-code
    # module's symlink and then looks for its relative imports next to the
    # resolved blob, so code files must be real siblings. Weight shards are
    # linked to the volume.
    import shutil

    os.makedirs(dest, exist_ok=True)
    missing = []
    for name in sorted(os.listdir(snapshot)):
        src = os.path.join(snapshot, name)
        dst = os.path.join(dest, name)
        if os.path.lexists(dst):
            continue
        if os.path.isdir(src) and not os.path.islink(src):
            shutil.copytree(src, dst, symlinks=False, ignore_dangling_symlinks=True)
        elif os.path.exists(src):
            real = os.path.realpath(src)
            if name.endswith(".safetensors") or os.path.getsize(real) > 64 << 20:
                os.symlink(real, dst)
            else:
                shutil.copyfile(real, dst)
        else:
            missing.append(name)
    log(f"{repo}: materialized {len(os.listdir(dest))} entries into {dest}")
    if missing:
        from huggingface_hub import hf_hub_download

        log(f"{repo}: fetching {len(missing)} files missing from the volume: {missing}")
        for name in missing:
            hf_hub_download(
                repo_id=repo, filename=name, revision=revision, local_dir=dest
            )
    return dest


DRAFT_WINDOW = 4096
DRAFT_BLOCK = 16

# Phase 1 prompt lengths (tokens of context before the question).
PHASE1_LENGTHS = (8_192, 32_768, 131_072)
# Phase 2 shapes.
CHAT_CONTEXT = 32_768
CHAT_TURNS = 6
CHAT_MSG_TOKENS = 300
CHAT_ANSWER_TOKENS = 800
AGENT_CONTEXT = 32_768
AGENT_STEPS = 4
AGENT_OUTPUT_TOKENS = 20_000
AGENT_TOOL_TOKENS = 300


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #


def server_command(max_running_requests: int, extra: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        TARGET_MODEL_PATH,
        "--served-model-name",
        TARGET_MODEL,
        "--trust-remote-code",
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        DRAFT_MODEL_PATH,
        "--speculative-dflash-block-size",
        str(DRAFT_BLOCK),
        "--speculative-draft-attention-backend",
        "trtllm_mha",
        "--attention-backend",
        "trtllm_mla",
        "--linear-attn-prefill-backend",
        "ptx_kda",
        "--linear-attn-decode-backend",
        "triton",
        "--linear-attn-verify-backend",
        "triton",
        "--enable-linear-replayssm-spec",
        "--linear-replayssm-cache-len",
        "32",
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--cuda-graph-backend-prefill",
        "breakable",
        "--cuda-graph-max-bs-prefill",
        "16384",
        "--tp-size",
        "8",
        "--mem-fraction-static",
        "0.88",
        "--max-running-requests",
        str(max_running_requests),
        "--enable-metrics",
        "--log-level",
        "info",
        "--host",
        "0.0.0.0",
        "--port",
        str(SERVER_PORT),
        *extra,
    ]


INTERESTING = (
    "Load weight",
    "Loading safetensors",
    "Capture",
    "KV Cache",
    "Cache is allocated",
    "Memory pool",
    "max_total_num_tokens",
    "DFLASH",
    "bounded",
    "Traceback",
    "Error",
    "error",
    "The server is fired up",
)


class Server:
    def __init__(self, log_path: Path, cmd: list[str]):
        self.log_path = log_path
        self.cmd = cmd
        self.proc = None
        self._log_fh = None
        self._tail_pos = 0

    def start(self) -> None:
        env = dict(
            os.environ,
            HF_HUB_OFFLINE="1",
            SGLANG_ENABLE_OVERLAP_PLAN_STREAM="1",
            PYTHONUNBUFFERED="1",
        )
        log(f"launching server: {' '.join(self.cmd)}")
        self._log_fh = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            self.cmd, stdout=self._log_fh, stderr=subprocess.STDOUT, env=env
        )

    def read_new_log(self) -> str:
        with open(self.log_path, errors="replace") as f:
            f.seek(self._tail_pos)
            text = f.read()
            self._tail_pos = f.tell()
        return text

    def log_offset(self) -> int:
        return self.log_path.stat().st_size

    def log_slice(self, start: int, end: int | None = None) -> str:
        with open(self.log_path, errors="replace") as f:
            f.seek(start)
            if end is None:
                return f.read()
            return f.read(max(0, end - start))

    def wait_healthy(self, timeout_s: float) -> float:
        t0 = time.time()
        last_print = 0.0
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                tail = self.log_slice(max(0, self.log_offset() - 20000))
                raise RuntimeError(
                    f"server exited with {self.proc.returncode}; log tail:\n{tail}"
                )
            new = self.read_new_log()
            for line in new.splitlines():
                if any(k in line for k in INTERESTING):
                    print("  server| " + line[-400:], flush=True)
            try:
                if requests.get(SERVER + "/health", timeout=5).status_code == 200:
                    took = time.time() - t0
                    log(f"server healthy after {took:.0f}s")
                    return took
            except requests.RequestException:
                pass
            if time.time() - last_print > 120:
                log(f"waiting for server ({time.time() - t0:.0f}s)")
                last_print = time.time()
            time.sleep(10)
        raise TimeoutError("server did not become healthy")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            log("stopping server")
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        # Scheduler/TP workers are named sglang::...; the runner itself is not.
        subprocess.run(
            ["pkill", "-9", "-f", "sglang::|sglang.launch_server"], check=False
        )
        if self._log_fh:
            self._log_fh.close()


# --------------------------------------------------------------------------- #
# Log parsing
# --------------------------------------------------------------------------- #

PREFILL_RE = re.compile(
    r"Prefill batch.*?#new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)"
)
USAGE_RE = re.compile(r"(?:full )?token usage: ([0-9.]+)")
MAMBA_USAGE_RE = re.compile(r"mamba usage: ([0-9.]+)")
DECODE_RE = re.compile(r"Decode batch.*?#running-req: (\d+)")
ACCEPT_RE = re.compile(r"accept len: ([0-9.]+)")
RETRACT_RE = re.compile(r"#retracted_reqs: (\d+)")


def prefill_summary(text: str) -> dict:
    new_tokens = 0
    cached = []
    n = 0
    for m in PREFILL_RE.finditer(text):
        n += 1
        new_tokens += int(m.group(2))
        cached.append(int(m.group(3)))
    return {
        "prefill_batches": n,
        "prefill_new_tokens": new_tokens,
        "prefill_cached_tokens_first": cached[0] if cached else None,
        "prefill_cached_tokens_max": max(cached) if cached else None,
    }


def load_summary(text: str) -> dict:
    usages = [float(m.group(1)) for m in USAGE_RE.finditer(text)]
    mamba = [float(m.group(1)) for m in MAMBA_USAGE_RE.finditer(text)]
    running = [int(m.group(1)) for m in DECODE_RE.finditer(text)]
    accept = [float(m.group(1)) for m in ACCEPT_RE.finditer(text)]
    retracted = sum(int(m.group(1)) for m in RETRACT_RE.finditer(text))
    return {
        "token_usage_max": max(usages) if usages else None,
        "token_usage_p50": sorted(usages)[len(usages) // 2] if usages else None,
        "mamba_usage_max": max(mamba) if mamba else None,
        "running_req_max": max(running) if running else None,
        "decode_accept_len_mean": sum(accept) / len(accept) if accept else None,
        "retracted_reqs": retracted,
        "prefill": prefill_summary(text),
    }


STARTUP_KEYS = (
    "is allocated",
    "VA upper bound",
    "max_total_num_tokens=",
    "Memory pool end",
    "DFLASH bounded",
    "bounded draft KV cache is disabled",
    "DFLASH all-SWA",
    "Capture cuda graph",
    "Capture prefill",
    "cuda graph",
    "sliding window memory pool",
    "SWA ring",
    "swa_max_total_num_tokens",
    "Mamba Cache",
    "mamba",
    "Load weight end",
    "Load weight begin",
    "draft",
)


def startup_lines(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        if any(k in line for k in STARTUP_KEYS) and "Prefill batch" not in line:
            # strip the timestamp/pid prefix for readability, keep the rank
            out.append(line.strip()[-600:])
    return out


# --------------------------------------------------------------------------- #
# Client helpers
# --------------------------------------------------------------------------- #


def server_info() -> dict:
    return requests.get(SERVER + "/get_server_info", timeout=60).json()


def flush_cache() -> None:
    requests.post(SERVER + "/flush_cache", timeout=120).raise_for_status()
    time.sleep(1)


def generate(text: str, max_new_tokens: int, ignore_eos: bool = False) -> dict:
    body = {
        "text": text,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
            "ignore_eos": ignore_eos,
        },
        "stream": True,
    }
    t0 = time.perf_counter()
    first = None
    acc = ""
    final = None
    with requests.post(SERVER + "/generate", json=body, stream=True, timeout=3600) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            if not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                break
            obj = json.loads(payload)
            if first is None:
                first = time.perf_counter()
            chunk = obj.get("text", "")
            # /generate streams cumulative text by default; tolerate deltas.
            acc = chunk if chunk.startswith(acc) else acc + chunk
            final = obj
    t1 = time.perf_counter()
    meta = final["meta_info"] if final else {}
    return {
        "ttft_s": (first - t0) if first else None,
        "e2e_s": t1 - t0,
        "text": acc,
        "prompt_tokens": meta.get("prompt_tokens"),
        "cached_tokens": meta.get("cached_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "spec_accept_length": meta.get("spec_accept_length"),
        "spec_verify_ct": meta.get("spec_verify_ct"),
        "finish_reason": (meta.get("finish_reason") or {}).get("type"),
    }


class Prompts:
    """Deterministic natural-language filler with exact token budgets."""

    TOPICS = [
        "the maintenance schedule of the coastal rail network",
        "how the municipal water treatment plant handles seasonal demand",
        "the procurement policy for laboratory equipment",
        "the history of the regional observatory",
        "guidelines for the volunteer translation program",
        "the architecture of the inventory reconciliation service",
        "a field guide to migratory birds of the northern wetlands",
        "the onboarding handbook for new lighthouse keepers",
        "a review of the community orchard's harvest records",
        "the incident report format used by the ferry operator",
    ]
    VERBS = ["describes", "summarizes", "documents", "explains", "records", "outlines"]
    ADJ = ["quarterly", "provisional", "revised", "detailed", "preliminary", "final"]

    def __init__(self, tokenizer, seed: int = 7):
        self.tok = tokenizer
        rng = random.Random(seed)
        paragraphs = []
        for i in range(12_000):
            topic = rng.choice(self.TOPICS)
            sents = []
            for j in range(rng.randint(3, 6)):
                sents.append(
                    f"Section {i}.{j} {rng.choice(self.VERBS)} {topic} in its "
                    f"{rng.choice(self.ADJ)} form, noting {rng.randint(2, 900)} "
                    f"items reviewed over {rng.randint(1, 52)} weeks and a "
                    f"variance of {rng.randint(0, 99)} percent against the plan."
                )
            paragraphs.append(" ".join(sents))
        corpus = "\n\n".join(paragraphs)
        self.ids = self.tok.encode(corpus, add_special_tokens=False)
        log(f"filler corpus: {len(self.ids)} tokens")

    def filler(self, n_tokens: int, offset: int = 0) -> str:
        assert offset + n_tokens <= len(self.ids), "filler corpus too small"
        return self.tok.decode(self.ids[offset : offset + n_tokens])

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def chat(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #


def phase0(server: Server, startup_s: float) -> dict:
    text = server.log_slice(0)
    info = server_info()
    keep = {
        k: info.get(k)
        for k in (
            "max_total_num_tokens",
            "max_running_requests",
            "chunked_prefill_size",
            "context_len",
            "page_size",
            "mem_fraction_static",
            "speculative_num_draft_tokens",
            "speculative_algorithm",
            "cuda_graph_max_bs",
            "attention_backend",
        )
    }
    internal = info.get("internal_states") or []
    return {
        "startup_s": startup_s,
        "server_info": keep,
        "internal_states": internal[:1],
        "startup_lines": startup_lines(text),
    }


def timed_request(server: Server, text: str, max_new: int, ignore_eos=False) -> dict:
    start = server.log_offset()
    out = generate(text, max_new, ignore_eos=ignore_eos)
    time.sleep(0.5)
    sl = server.log_slice(start)
    out.update(prefill_summary(sl))
    accept = [float(m.group(1)) for m in ACCEPT_RE.finditer(sl)]
    out["decode_accept_len_mean"] = sum(accept) / len(accept) if accept else None
    out.pop("text_full", None)
    return out


def phase1(server: Server, prompts: Prompts) -> dict:
    results = {}
    for n in PHASE1_LENGTHS:
        flush_cache()
        msgs = [
            {"role": "system", "content": "You are a careful technical writer."},
            {
                "role": "user",
                "content": prompts.filler(n)
                + "\n\nIn two sentences, what kind of document is this?",
            },
        ]
        text = prompts.chat(msgs)
        runs = []
        for rep in range(3):
            r = timed_request(server, text, 64)
            r.pop("text")
            r["rep"] = rep
            runs.append(r)
            log(
                f"phase1 n={n} rep={rep}: ttft={r['ttft_s']:.3f}s prompt={r['prompt_tokens']} "
                f"cached={r['cached_tokens']} prefill_new={r['prefill_new_tokens']} "
                f"accept={r['spec_accept_length']}"
            )
            time.sleep(2)  # let the finished request land in the radix tree
        results[str(n)] = runs
    return results


def phase2(server: Server, prompts: Prompts) -> dict:
    out = {}

    # Chat: short user turns, ~800-token answers, over 6 turns.
    flush_cache()
    msgs = [
        {"role": "system", "content": "You are a careful technical writer."},
        {
            "role": "user",
            "content": prompts.filler(CHAT_CONTEXT, offset=200_000)
            + "\n\nWrite a detailed commentary on the sections above.",
        },
    ]
    turns = []
    for t in range(CHAT_TURNS):
        text = prompts.chat(msgs)
        r = timed_request(server, text, CHAT_ANSWER_TOKENS, ignore_eos=True)
        answer = r.pop("text")
        r["turn"] = t
        turns.append(r)
        log(
            f"phase2 chat turn={t}: ttft={r['ttft_s']:.3f}s prompt={r['prompt_tokens']} "
            f"cached={r['cached_tokens']} prefill_new={r['prefill_new_tokens']} "
            f"completion={r['completion_tokens']} accept={r['spec_accept_length']}"
        )
        msgs.append({"role": "assistant", "content": answer})
        msgs.append(
            {
                "role": "user",
                "content": "Continue with the next part, considering this note: "
                + prompts.filler(CHAT_MSG_TOKENS, offset=400_000 + t * 1000),
            }
        )
        time.sleep(2)
    out["chat"] = turns

    # Agentic: long outputs, short tool results, over 4 steps.
    flush_cache()
    msgs = [
        {"role": "system", "content": "You are an autonomous engineering agent."},
        {
            "role": "user",
            "content": prompts.filler(AGENT_CONTEXT, offset=600_000)
            + "\n\nProduce an exhaustive implementation plan for the system above.",
        },
    ]
    steps = []
    for s in range(AGENT_STEPS):
        text = prompts.chat(msgs)
        r = timed_request(server, text, AGENT_OUTPUT_TOKENS, ignore_eos=True)
        answer = r.pop("text")
        r["step"] = s
        steps.append(r)
        log(
            f"phase2 agent step={s}: ttft={r['ttft_s']:.3f}s prompt={r['prompt_tokens']} "
            f"cached={r['cached_tokens']} prefill_new={r['prefill_new_tokens']} "
            f"completion={r['completion_tokens']} accept={r['spec_accept_length']} "
            f"e2e={r['e2e_s']:.1f}s"
        )
        msgs.append({"role": "assistant", "content": answer})
        msgs.append(
            {
                "role": "user",
                "content": "Tool result: "
                + prompts.filler(AGENT_TOOL_TOKENS, offset=900_000 + s * 1000),
            }
        )
        time.sleep(2)
    out["agentic"] = steps
    return out


def bench_serving(
    server: Server,
    out_dir: Path,
    tag: str,
    *,
    groups: int,
    prompts_per_group: int,
    system_len: int,
    question_len: int,
    output_len: int,
    turns: int,
    concurrency: int,
) -> dict:
    flush_cache()
    info_before = server_info()
    start = server.log_offset()
    result_file = out_dir / f"bench_{tag}.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang-oai-chat",
        "--base-url",
        SERVER,
        "--model",
        TARGET_MODEL,
        "--tokenizer",
        TARGET_MODEL_PATH,
        "--dataset-name",
        "generated-shared-prefix",
        "--gsp-num-groups",
        str(groups),
        "--gsp-prompts-per-group",
        str(prompts_per_group),
        "--gsp-system-prompt-len",
        str(system_len),
        "--gsp-question-len",
        str(question_len),
        "--gsp-output-len",
        str(output_len),
        "--gsp-num-turns",
        str(turns),
        "--gsp-range-ratio",
        "1.0",
        "--num-prompts",
        str(groups * prompts_per_group),
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        "inf",
        "--warmup-requests",
        "1",
        "--disable-tqdm",
        "--output-file",
        str(result_file),
    ]
    log(f"bench {tag}: {' '.join(cmd[3:])}")
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=dict(os.environ, HF_HUB_OFFLINE="1"),
        timeout=7200,
    )
    took = time.time() - t0
    (out_dir / f"bench_{tag}.stdout").write_text(
        proc.stdout + "\n--- stderr ---\n" + proc.stderr
    )
    summary = {}
    if result_file.exists():
        lines = [l for l in result_file.read_text().splitlines() if l.strip()]
        if lines:
            summary = json.loads(lines[-1])
    info_after = server_info()
    sl = server.log_slice(start)
    keep_keys = (
        "completed",
        "total_input_tokens",
        "total_output_tokens",
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_e2e_latency_ms",
        "mean_e2e_latency_ms",
        "p99_e2e_latency_ms",
        "duration",
        "max_concurrency",
        "accept_length",
    )
    res = {k: summary.get(k) for k in keep_keys if k in summary}
    res.update(
        {
            "tag": tag,
            "returncode": proc.returncode,
            "wall_s": took,
            "config": dict(
                groups=groups,
                prompts_per_group=prompts_per_group,
                system_len=system_len,
                question_len=question_len,
                output_len=output_len,
                turns=turns,
                concurrency=concurrency,
            ),
            "server_load": load_summary(sl),
            "avg_spec_accept_length_before": _accept(info_before),
            "avg_spec_accept_length_after": _accept(info_after),
        }
    )
    if proc.returncode != 0:
        res["stderr_tail"] = proc.stderr[-3000:]
    log(
        f"bench {tag}: rc={proc.returncode} out_tok/s={res.get('output_throughput')} "
        f"ttft_p50={res.get('median_ttft_ms')}ms ttft_p99={res.get('p99_ttft_ms')}ms "
        f"tpot_p50={res.get('median_tpot_ms')}ms usage_max={res['server_load']['token_usage_max']} "
        f"retracted={res['server_load']['retracted_reqs']}"
    )
    return res


def _accept(info: dict):
    states = info.get("internal_states") or [{}]
    return (states[0] or {}).get("avg_spec_accept_length")


def phase3(
    server: Server, out_dir: Path, concurrencies: list[int], system_len: int
) -> dict:
    runs = []
    for c in concurrencies:
        groups = max(2, c // 2)
        runs.append(
            bench_serving(
                server,
                out_dir,
                f"p3_c{c}",
                groups=groups,
                prompts_per_group=2,
                system_len=system_len,
                question_len=256,
                output_len=512,
                turns=4,
                concurrency=c,
            )
        )
    return {"system_len": system_len, "runs": runs}


def phase4(server: Server, out_dir: Path, concurrencies: list[int]) -> dict:
    # Instinct shape from ClickHouse (endpoint_token_metrics, Sep 2026):
    # prompt p50 ~100k tokens, new tokens per turn p50 ~500-1500, answer p50 ~300,
    # 99% of requests are prefix hits, per-replica concurrency p50 1, p90 3.
    runs = []
    for c in concurrencies:
        groups = max(2, c)
        runs.append(
            bench_serving(
                server,
                out_dir,
                f"p4_c{c}",
                groups=groups,
                prompts_per_group=1,
                system_len=100_000,
                question_len=600,
                output_len=300,
                turns=6,
                concurrency=c,
            )
        )
    return {"runs": runs}


# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-label", required=True)
    ap.add_argument("--mrr", type=int, required=True, help="--max-running-requests")
    ap.add_argument("--phases", default="0,1,2,3,4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--health-timeout", type=float, default=6000)
    ap.add_argument("--p3-concurrency", default="8,24,48")
    ap.add_argument("--p3-system-len", type=int, default=32768)
    ap.add_argument("--p4-concurrency", default="4,16")
    ap.add_argument("--server-extra", default="")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    phases = {p.strip() for p in args.phases.split(",") if p.strip()}
    extra = args.server_extra.split() if args.server_extra else []

    global TARGET_MODEL_PATH, DRAFT_MODEL_PATH
    for path in (TARGET_MODEL_PATH, DRAFT_MODEL_PATH):
        if not os.path.exists(os.path.join(path, "config.json")):
            raise SystemExit(f"missing model snapshot: {path}")
    TARGET_MODEL_PATH = materialize_model_dir(
        TARGET_MODEL_PATH, TARGET_MODEL, TARGET_REVISION, f"{LOCAL_MODELS}/target"
    )
    DRAFT_MODEL_PATH = materialize_model_dir(
        DRAFT_MODEL_PATH, DRAFT_MODEL, DRAFT_REVISION, f"{LOCAL_MODELS}/draft"
    )
    server = Server(out_dir / "server.log", server_command(args.mrr, extra))
    status = {"ref": args.ref_label, "mrr": args.mrr, "phases": sorted(phases)}
    (out_dir / "status.json").write_text(json.dumps(status, indent=2))
    try:
        server.start()
        startup_s = server.wait_healthy(args.health_timeout)
        # Warm up cuda graphs etc. with one tiny request.
        generate("Say hello.", 8)

        if "0" in phases:
            r = phase0(server, startup_s)
            (out_dir / "phase0.json").write_text(json.dumps(r, indent=2))
            for line in r["startup_lines"]:
                print("  p0| " + line, flush=True)

        prompts = None
        if phases & {"1", "2"}:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                TARGET_MODEL_PATH, trust_remote_code=True
            )
            prompts = Prompts(tok)

        if "1" in phases:
            r = phase1(server, prompts)
            (out_dir / "phase1.json").write_text(json.dumps(r, indent=2))
        if "2" in phases:
            r = phase2(server, prompts)
            (out_dir / "phase2.json").write_text(json.dumps(r, indent=2))
        if "3" in phases:
            cs = [int(x) for x in args.p3_concurrency.split(",") if x]
            r = phase3(server, out_dir, cs, args.p3_system_len)
            (out_dir / "phase3.json").write_text(json.dumps(r, indent=2))
        if "4" in phases:
            cs = [int(x) for x in args.p4_concurrency.split(",") if x]
            r = phase4(server, out_dir, cs)
            (out_dir / "phase4.json").write_text(json.dumps(r, indent=2))
        status["ok"] = True
    except Exception as e:  # noqa: BLE001
        status["ok"] = False
        status["error"] = repr(e)[-4000:]
        log(f"FAILED: {e!r}")
        import traceback

        traceback.print_exc()
    finally:
        server.stop()
        (out_dir / "status.json").write_text(json.dumps(status, indent=2))
    return 0 if status.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
