"""Modal app for running sglang.bench_serving.

Usage:
  uvx modal run scripts/modal/bench_serving_app.py --config bench_config.toml
"""

import os
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent.parent

HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

server_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    ).entrypoint([])
    .apt_install("git", "curl", "libnuma-dev")
    .pip_install("sglang[all]==0.5.13.post1")
    .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
)

HF_SECRET = modal.Secret.from_name("huggingface-secret")

app = modal.App("sglang-bench-serve")


def _tee_pipe(pipe, *files):
    """Read lines from a pipe and write to each file (real-time tee)."""
    try:
        for line in iter(pipe.readline, b""):
            for f in files:
                f.write(line)
                f.flush()
    finally:
        pipe.close()


def _bench_cmd(**kwargs) -> list:
    """Build a ``python -m sglang.bench_serving`` argument list from kwargs."""
    cmd = [sys.executable, "-m", "sglang.bench_serving"]
    for k, v in kwargs.items():
        if v is None or (isinstance(v, bool) and not v):
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, list):
            cmd.extend([flag] + [str(x) for x in v])
        elif isinstance(v, bool) and v:
            cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])
    return cmd


def _build_server_cmd(cfg: dict) -> list[str]:
    """Build sglang launch_server command from config dict."""
    srv = cfg["server"]
    cmd = [
        sys.executable, "-u", "-m", "sglang.launch_server",
        "--model-path", srv["model"],
        "--host", srv.get("host", "0.0.0.0"),
        "--port", str(srv.get("port", 30000)),
    ]
    for k, v in srv.get("args", {}).items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool) and v:
            cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])
    return cmd


@app.local_entrypoint()
def main(config: str):
    cfg_path = Path(config)
    cfg_text = cfg_path.read_text()
    cfg = tomllib.loads(cfg_text)
    model = cfg["server"]["model"]
    spec = cfg["server"].get("args", {}).get("speculative_algorithm", "none")
    print(f"Benchmarking {model} on B200 (spec={spec})...")
    serve_and_bench.remote(config_text=cfg_text)


@app.function(
    image=server_image,
    gpu="B200:2",
    secrets=[HF_SECRET],
    timeout=3600,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
)
def serve_and_bench(config_text: str) -> str:
    import requests

    cfg = tomllib.loads(config_text)
    port = cfg["server"].get("port", 30000)
    name = cfg["server"]["model"].replace("/", "_")
    server_cmd = _build_server_cmd(cfg)

    extra_env = {k: v for k, v in cfg.get("env", {}).items()}
    proc_env = os.environ.copy()
    proc_env.update(extra_env)

    startup_timeout = int(cfg.get("server", {}).get("startup_timeout", 600))

    stdout_f = open(f"{name}.stdout", "wb")
    stderr_f = open(f"{name}.stderr", "wb")

    proc = subprocess.Popen(
        server_cmd,
        env=proc_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    t_out = threading.Thread(target=_tee_pipe, args=(proc.stdout, sys.stdout.buffer, stdout_f), daemon=True)
    t_err = threading.Thread(target=_tee_pipe, args=(proc.stderr, sys.stderr.buffer, stderr_f), daemon=True)
    t_out.start()
    t_err.start()

    try:
        deadline = time.time() + startup_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError()
            try:
                if requests.get(f"http://0.0.0.0:{port}/health", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise RuntimeError(
                f"Server did not become ready within {startup_timeout}s."
            )

        print("Server ready. Running benchmark...")
        bench = cfg.get("bench", {})
        cmd = _bench_cmd(
            host="0.0.0.0",
            port=port,
            backend="sglang",
            dataset_name=bench.get("dataset_name", "random"),
            num_prompts=bench.get("num_prompts", 30),
            random_input_len=bench.get("random_input_len", 512),
            random_output_len=bench.get("random_output_len", 512),
            request_rate=bench.get("request_rate", 5),
            warmup_requests=bench.get("warmup_requests", 1),
        )
        result = subprocess.run(cmd, check=True, capture_output=True)
        sys.stdout.buffer.write(result.stdout)
        sys.stdout.buffer.flush()
        sys.stderr.buffer.write(result.stderr)
        sys.stderr.buffer.flush()
        stdout_f.write(result.stdout)
        stderr_f.write(result.stderr)
        stdout_f.flush()
        stderr_f.flush()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        stdout_f.close()
        stderr_f.close()
