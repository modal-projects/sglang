from __future__ import annotations

import json
import os
import pathlib
import subprocess
from typing import Any

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOTE_REPO_ROOT = pathlib.Path("/sgl-workspace/sglang")
TEST_FILE = REMOTE_REPO_ROOT / "test/manual/test_qwen35_merged_lora_live_update.py"

APP_NAME = "sglang-qwen35-merged-lora-live-update"
HF_CACHE_PATH = "/root/.cache/huggingface"
HF_CACHE_VOLUME_NAME = os.getenv("HF_CACHE_VOLUME_NAME", "huggingface-cache")
SGLANG_IMAGE_TAG = os.getenv(
    "SGLANG_MODAL_IMAGE_TAG",
    "lmsysorg/sglang:nightly-dev-cu13-20260407-5cc246e0",
)
GPU = os.getenv("QWEN35_LIVE_UPDATE_MODAL_GPU", "B200")
MEMORY_MB = int(os.getenv("QWEN35_LIVE_UPDATE_MODAL_MEMORY_MB", "131072"))

BASE_MODEL = os.getenv("QWEN35_BASE_MODEL", "Qwen/Qwen3.5-35B-A3B")
PREFILL_ATTENTION_BACKEND = os.getenv(
    "QWEN35_PREFILL_ATTENTION_BACKEND", "trtllm_mha"
)
DECODE_ATTENTION_BACKEND = os.getenv(
    "QWEN35_DECODE_ATTENTION_BACKEND", "trtllm_mha"
)
MOE_RUNNER_BACKEND = os.getenv("QWEN35_MOE_RUNNER_BACKEND", "flashinfer_trtllm")
MEM_FRACTION_STATIC = os.getenv("QWEN35_LIVE_UPDATE_MEM_FRACTION_STATIC", "0.85")
CHUNKED_PREFILL_SIZE = os.getenv("QWEN35_LIVE_UPDATE_CHUNKED_PREFILL_SIZE", "8192")
MAX_PREFILL_TOKENS = os.getenv("QWEN35_LIVE_UPDATE_MAX_PREFILL_TOKENS", "8192")
PAGE_SIZE = os.getenv("QWEN35_LIVE_UPDATE_PAGE_SIZE", "64")
MAX_RUNNING_REQUESTS = os.getenv("QWEN35_LIVE_UPDATE_MAX_RUNNING_REQUESTS", "32")
CUDA_GRAPH_MAX_BS = os.getenv("QWEN35_LIVE_UPDATE_CUDA_GRAPH_MAX_BS", "32")
KV_CACHE_DTYPE = os.getenv("QWEN35_LIVE_UPDATE_KV_CACHE_DTYPE", "bf16")
MAMBA_SCHEDULER_STRATEGY = os.getenv(
    "QWEN35_LIVE_UPDATE_MAMBA_SCHEDULER_STRATEGY", "extra_buffer"
)
MAMBA_SSM_DTYPE = os.getenv("QWEN35_LIVE_UPDATE_MAMBA_SSM_DTYPE", "bfloat16")
ATOMIC_PAUSE_MODE = os.getenv("QWEN35_LIVE_UPDATE_ATOMIC_PAUSE_MODE", "in_place")
FLUSH_CACHE = os.getenv("QWEN35_LIVE_UPDATE_FLUSH_CACHE", "0")
WEIGHT_VERSION = os.getenv(
    "QWEN35_LIVE_UPDATE_WEIGHT_VERSION", "qwen35-merged-lora-live-update"
)

LOCAL_ADAPTER_DIR = pathlib.Path(os.getenv("QWEN35_LORA_DIR", str(REPO_ROOT)))
LOCAL_ADAPTER_CONFIG_PATH = pathlib.Path(
    os.getenv("QWEN35_LORA_CONFIG", str(LOCAL_ADAPTER_DIR / "adapter_config.json"))
)
LOCAL_ADAPTER_WEIGHTS_PATH = pathlib.Path(
    os.getenv(
        "QWEN35_LORA_WEIGHTS",
        str(LOCAL_ADAPTER_DIR / "sampler_weights_init.safetensors"),
    )
)
ADAPTER_VOLUME_SUBDIR = pathlib.PurePosixPath(
    "local-adapters/qwen35-merged-lora-live-update"
)
ADAPTER_VOLUME_CONFIG_REL = ADAPTER_VOLUME_SUBDIR / "adapter_config.json"
ADAPTER_VOLUME_WEIGHTS_REL = ADAPTER_VOLUME_SUBDIR / "sampler_weights_init.safetensors"

HF_IMAGE_ENV = {
    "HF_HUB_CACHE": HF_CACHE_PATH,
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

SOURCE_DIRS = [
    (
        REPO_ROOT / "python/sglang",
        str(REMOTE_REPO_ROOT / "python/sglang"),
    ),
    (
        REPO_ROOT / "test/manual",
        str(REMOTE_REPO_ROOT / "test/manual"),
    ),
]

RUNTIME_CONFIG_SECRET = modal.Secret.from_dict(
    {
        "SGLANG_MODAL_IMAGE_TAG": SGLANG_IMAGE_TAG,
        "QWEN35_LIVE_UPDATE_MODAL_GPU": GPU,
        "QWEN35_BASE_MODEL": BASE_MODEL,
        "QWEN35_PREFILL_ATTENTION_BACKEND": PREFILL_ATTENTION_BACKEND,
        "QWEN35_DECODE_ATTENTION_BACKEND": DECODE_ATTENTION_BACKEND,
        "QWEN35_MOE_RUNNER_BACKEND": MOE_RUNNER_BACKEND,
        "QWEN35_LIVE_UPDATE_MEM_FRACTION_STATIC": MEM_FRACTION_STATIC,
        "QWEN35_LIVE_UPDATE_CHUNKED_PREFILL_SIZE": CHUNKED_PREFILL_SIZE,
        "QWEN35_LIVE_UPDATE_MAX_PREFILL_TOKENS": MAX_PREFILL_TOKENS,
        "QWEN35_LIVE_UPDATE_PAGE_SIZE": PAGE_SIZE,
        "QWEN35_LIVE_UPDATE_MAX_RUNNING_REQUESTS": MAX_RUNNING_REQUESTS,
        "QWEN35_LIVE_UPDATE_CUDA_GRAPH_MAX_BS": CUDA_GRAPH_MAX_BS,
        "QWEN35_LIVE_UPDATE_KV_CACHE_DTYPE": KV_CACHE_DTYPE,
        "QWEN35_LIVE_UPDATE_MAMBA_SCHEDULER_STRATEGY": MAMBA_SCHEDULER_STRATEGY,
        "QWEN35_LIVE_UPDATE_MAMBA_SSM_DTYPE": MAMBA_SSM_DTYPE,
        "QWEN35_LIVE_UPDATE_ATOMIC_PAUSE_MODE": ATOMIC_PAUSE_MODE,
        "QWEN35_LIVE_UPDATE_FLUSH_CACHE": FLUSH_CACHE,
        "QWEN35_LIVE_UPDATE_WEIGHT_VERSION": WEIGHT_VERSION,
    }
)

app = modal.App(name=APP_NAME)
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

image = modal.Image.from_registry(SGLANG_IMAGE_TAG).env(HF_IMAGE_ENV)
if modal.is_local():
    for local_path, remote_path in SOURCE_DIRS:
        image = image.add_local_dir(local_path, remote_path, copy=False)


def _tail(text: str, limit: int = 20000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _remote_pythonpath() -> str:
    return f"{REMOTE_REPO_ROOT / 'python'}:{REMOTE_REPO_ROOT}"


def _remote_volume_path(relative_path: pathlib.PurePosixPath) -> pathlib.Path:
    return pathlib.Path(HF_CACHE_PATH) / pathlib.Path(str(relative_path))


def _py_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(HF_IMAGE_ENV)
    env["PYTHONPATH"] = _remote_pythonpath()
    env["PYTHONPYCACHEPREFIX"] = "/tmp/pycache"
    env["QWEN35_BASE_MODEL"] = BASE_MODEL
    env["QWEN35_PREFILL_ATTENTION_BACKEND"] = PREFILL_ATTENTION_BACKEND
    env["QWEN35_DECODE_ATTENTION_BACKEND"] = DECODE_ATTENTION_BACKEND
    env["QWEN35_MOE_RUNNER_BACKEND"] = MOE_RUNNER_BACKEND
    env["QWEN35_LIVE_UPDATE_MEM_FRACTION_STATIC"] = MEM_FRACTION_STATIC
    env["QWEN35_LIVE_UPDATE_CHUNKED_PREFILL_SIZE"] = CHUNKED_PREFILL_SIZE
    env["QWEN35_LIVE_UPDATE_MAX_PREFILL_TOKENS"] = MAX_PREFILL_TOKENS
    env["QWEN35_LIVE_UPDATE_PAGE_SIZE"] = PAGE_SIZE
    env["QWEN35_LIVE_UPDATE_MAX_RUNNING_REQUESTS"] = MAX_RUNNING_REQUESTS
    env["QWEN35_LIVE_UPDATE_CUDA_GRAPH_MAX_BS"] = CUDA_GRAPH_MAX_BS
    env["QWEN35_LIVE_UPDATE_KV_CACHE_DTYPE"] = KV_CACHE_DTYPE
    env["QWEN35_LIVE_UPDATE_MAMBA_SCHEDULER_STRATEGY"] = MAMBA_SCHEDULER_STRATEGY
    env["QWEN35_LIVE_UPDATE_MAMBA_SSM_DTYPE"] = MAMBA_SSM_DTYPE
    env["QWEN35_LIVE_UPDATE_ATOMIC_PAUSE_MODE"] = ATOMIC_PAUSE_MODE
    env["QWEN35_LIVE_UPDATE_FLUSH_CACHE"] = FLUSH_CACHE
    env["QWEN35_LIVE_UPDATE_WEIGHT_VERSION"] = WEIGHT_VERSION
    env["QWEN35_LORA_CONFIG"] = str(_remote_volume_path(ADAPTER_VOLUME_CONFIG_REL))
    env["QWEN35_LORA_WEIGHTS"] = str(_remote_volume_path(ADAPTER_VOLUME_WEIGHTS_REL))
    return env


def _upload_adapter_assets_to_volume() -> dict[str, Any]:
    if not LOCAL_ADAPTER_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing adapter config for Modal upload: {LOCAL_ADAPTER_CONFIG_PATH}"
        )
    if not LOCAL_ADAPTER_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing adapter weights for Modal upload: {LOCAL_ADAPTER_WEIGHTS_PATH}"
        )

    with hf_cache_vol.batch_upload(force=True) as batch:
        batch.put_file(str(LOCAL_ADAPTER_CONFIG_PATH), str(ADAPTER_VOLUME_CONFIG_REL))
        batch.put_file(str(LOCAL_ADAPTER_WEIGHTS_PATH), str(ADAPTER_VOLUME_WEIGHTS_REL))

    return {
        "local_adapter_config": str(LOCAL_ADAPTER_CONFIG_PATH),
        "local_adapter_weights": str(LOCAL_ADAPTER_WEIGHTS_PATH),
        "local_adapter_weights_size_bytes": LOCAL_ADAPTER_WEIGHTS_PATH.stat().st_size,
        "volume_adapter_config": str(_remote_volume_path(ADAPTER_VOLUME_CONFIG_REL)),
        "volume_adapter_weights": str(_remote_volume_path(ADAPTER_VOLUME_WEIGHTS_REL)),
    }


@app.function(
    image=image,
    gpu=GPU,
    memory=MEMORY_MB,
    timeout=4 * 60 * 60,
    retries=0,
    secrets=[RUNTIME_CONFIG_SECRET],
    volumes={HF_CACHE_PATH: hf_cache_vol},
)
def run_validation() -> dict[str, Any]:
    proc = subprocess.run(
        ["python", str(TEST_FILE)],
        env=_py_env(),
        capture_output=True,
        text=True,
        timeout=4 * 60 * 60,
    )
    return {
        "cmd": ["python", str(TEST_FILE)],
        "returncode": proc.returncode,
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
        "base_model": BASE_MODEL,
        "gpu": GPU,
        "image_tag": SGLANG_IMAGE_TAG,
        "prefill_attention_backend": PREFILL_ATTENTION_BACKEND,
        "decode_attention_backend": DECODE_ATTENTION_BACKEND,
        "moe_runner_backend": MOE_RUNNER_BACKEND,
        "flush_cache": FLUSH_CACHE.lower() in ("1", "true", "yes", "on"),
        "atomic_pause_mode": ATOMIC_PAUSE_MODE,
    }


@app.local_entrypoint()
def main():
    upload_info = _upload_adapter_assets_to_volume()
    result = run_validation.remote()
    payload = {
        "upload_info": upload_info,
        "validation_result": result,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])
