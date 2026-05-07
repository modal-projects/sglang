from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOTE_REPO_ROOT = pathlib.Path("/sgl-workspace/sglang")

APP_NAME = "sglang-qwen35-lora-merge-latency"
HF_CACHE_PATH = "/root/.cache/huggingface"
HF_CACHE_VOLUME_NAME = os.getenv("HF_CACHE_VOLUME_NAME", "huggingface-cache")
SGLANG_IMAGE_TAG = os.getenv(
    "SGLANG_MODAL_IMAGE_TAG",
    "lmsysorg/sglang:nightly-dev-cu13-20260407-5cc246e0",
)
GPU = os.getenv("QWEN35_LIVE_UPDATE_MODAL_GPU", "B200")
MEMORY_MB = int(os.getenv("QWEN35_LIVE_UPDATE_MODAL_MEMORY_MB", "131072"))

BASE_MODEL = os.getenv("QWEN35_BASE_MODEL", "Qwen/Qwen3.5-35B-A3B")
PREFILL_ATTENTION_BACKEND = os.getenv("QWEN35_PREFILL_ATTENTION_BACKEND", "trtllm_mha")
DECODE_ATTENTION_BACKEND = os.getenv("QWEN35_DECODE_ATTENTION_BACKEND", "trtllm_mha")
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
WEIGHT_VERSION = os.getenv("QWEN35_LIVE_UPDATE_WEIGHT_VERSION", "latency-harness")

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
ADAPTER_VOLUME_SUBDIR = pathlib.PurePosixPath("local-adapters/qwen35-latency")
ADAPTER_VOLUME_CONFIG_REL = ADAPTER_VOLUME_SUBDIR / "adapter_config.json"
ADAPTER_VOLUME_WEIGHTS_REL = ADAPTER_VOLUME_SUBDIR / "sampler_weights_init.safetensors"
MERGE_LOADER = "sglang.srt.model_loader.lora_merge_loader.merge_lora_tensors_inplace"

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
]

RUNTIME_CONFIG_SECRET = modal.Secret.from_dict(
    {
        "SGLANG_MODAL_IMAGE_TAG": SGLANG_IMAGE_TAG,
        "QWEN35_LIVE_UPDATE_MODAL_GPU": GPU,
        "QWEN35_BASE_MODEL": BASE_MODEL,
    }
)

app = modal.App(name=APP_NAME)
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

image = modal.Image.from_registry(SGLANG_IMAGE_TAG).run_commands(
    f"rm -rf {HF_CACHE_PATH}"
).env(HF_IMAGE_ENV)
if modal.is_local():
    for local_path, remote_path in SOURCE_DIRS:
        image = image.add_local_dir(local_path, remote_path, copy=False)


def _remote_volume_path(relative_path: pathlib.PurePosixPath) -> pathlib.Path:
    return pathlib.Path(HF_CACHE_PATH) / pathlib.Path(str(relative_path))


def _py_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(HF_IMAGE_ENV)
    env["PYTHONPATH"] = f"{REMOTE_REPO_ROOT / 'python'}:{REMOTE_REPO_ROOT}"
    env["PYTHONPYCACHEPREFIX"] = "/tmp/pycache"
    env["SGLANG_EXTRA_ROUTERS"] = (
        "sglang.srt.entrypoints.local_lora_merge_router:router"
    )
    return env


def _upload_adapter_assets_to_volume() -> dict[str, Any]:
    missing = [
        str(path)
        for path in (LOCAL_ADAPTER_CONFIG_PATH, LOCAL_ADAPTER_WEIGHTS_PATH)
        if not path.exists()
    ]
    if missing:
        return {
            "skipped_upload": True,
            "missing_local_files": missing,
            "volume_adapter_config": str(_remote_volume_path(ADAPTER_VOLUME_CONFIG_REL)),
            "volume_adapter_weights": str(
                _remote_volume_path(ADAPTER_VOLUME_WEIGHTS_REL)
            ),
            "note": "Using adapter files already present in the Modal HF cache volume.",
        }

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


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 3)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min_ms": round(min(values), 3) if values else None,
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
        "p99_ms": _percentile(values, 99),
        "max_ms": round(max(values), 3) if values else None,
    }


def _summarize_streams(
    records: list[dict[str, Any]],
    *,
    update_start: float,
    update_end: float,
) -> dict[str, Any]:
    intervals_by_window: dict[str, list[float]] = {"pre": [], "during": [], "post": []}
    overlapping_update_gaps: list[float] = []
    first_chunk_latencies = []
    e2e_latencies = []
    chunk_times = []

    for record in records:
        events = record["events"]
        if events:
            first_chunk_latencies.append((events[0]["t"] - record["start"]) * 1000)
            chunk_times.extend(event["t"] for event in events)
        e2e_latencies.append((record["end"] - record["start"]) * 1000)

        previous = record["start"]
        for event in events:
            current = event["t"]
            interval_ms = (current - previous) * 1000
            if current < update_start:
                bucket = "pre"
            elif current <= update_end:
                bucket = "during"
            else:
                bucket = "post"
            intervals_by_window[bucket].append(interval_ms)
            if previous <= update_end and current >= update_start:
                overlapping_update_gaps.append(interval_ms)
            previous = current

    update_duration = max(update_end - update_start, 1e-9)
    pre_duration = max(
        update_start - min((r["start"] for r in records), default=update_start), 1e-9
    )
    post_duration = max(
        max((r["end"] for r in records), default=update_end) - update_end, 1e-9
    )
    pre_chunks = len([t for t in chunk_times if t < update_start])
    during_chunks = len([t for t in chunk_times if update_start <= t <= update_end])
    post_chunks = len([t for t in chunk_times if t > update_end])

    return {
        "chunk_interval_ms": {
            key: _summary(value) for key, value in intervals_by_window.items()
        },
        "first_chunk_latency_ms": _summary(first_chunk_latencies),
        "request_e2e_latency_ms": _summary(e2e_latencies),
        "max_update_overlapping_gap_ms": round(max(overlapping_update_gaps), 3)
        if overlapping_update_gaps
        else None,
        "chunk_throughput_per_s": {
            "pre": round(pre_chunks / pre_duration, 3),
            "during": round(during_chunks / update_duration, 3),
            "post": round(post_chunks / post_duration, 3),
        },
        "update_window_ms": round(update_duration * 1000, 3),
    }


def _compact_meta_info(meta_info: Any) -> Any:
    if not isinstance(meta_info, dict):
        return meta_info
    compact = dict(meta_info)
    epochs = compact.pop("output_token_weight_epochs", None)
    if isinstance(epochs, list):
        compact["output_token_weight_epochs_len"] = len(epochs)
        compact["output_token_weight_epochs_head"] = epochs[:8]
        compact["output_token_weight_epochs_tail"] = epochs[-8:]
    return compact


def _compact_pair_trace(pair: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "index",
        "base_name",
        "prestaged",
        "pair_ms",
        "resolve_delta_specs_ms",
        "apply_loaded_delta_ms",
        "gpu_add_calls",
        "gpu_add_bytes",
        "gpu_delta_to_device_ms",
        "gpu_dst_to_fp32_ms",
        "gpu_add_ms",
        "gpu_copy_back_ms",
        "touched_flashinfer_layer_count",
        "dense_delta_shape",
        "gpu_stage_lora_ms",
        "gpu_select_lora_ms",
        "gpu_matmul_ms",
        "gpu_apply_ms",
        "gpu_bucket_apply_ms",
        "gpu_vocab_tile_ms",
        "gpu_dense_full_ms",
        "gpu_bucket_count",
        "gpu_matmul_count",
        "gpu_bucket_bytes_estimate_max",
        "spec_param_names",
    )
    return {key: pair[key] for key in keys if key in pair}


def _extract_update_trace_summary(update_result: dict[str, Any]) -> dict[str, Any]:
    body = update_result.get("body") if isinstance(update_result, dict) else None
    trace = ((body or {}).get("trace") or {}).get("sglang_update_trace") or {}
    scheduler_results = trace.get("scheduler_results") or []
    scheduler_trace = scheduler_results[0] if scheduler_results else {}
    pause_trace = scheduler_trace.get("scheduler_pause_trace") or {}

    return {
        "request_id": trace.get("request_id"),
        "weight_epoch": trace.get("weight_epoch"),
        "weight_version": trace.get("weight_version"),
        "tokenizer_paused_total_ms": trace.get("tokenizer_paused_total_ms"),
        "tokenizer_scheduler_communicator_ms": trace.get(
            "tokenizer_scheduler_communicator_ms"
        ),
        "scheduler_pause_trace": pause_trace,
        "tp_worker_deserialize_ms": scheduler_trace.get("tp_worker_deserialize_ms"),
        "lora_loader_pair_count": scheduler_trace.get("lora_loader_pair_count"),
        "lora_loader_pair_apply_total_ms": scheduler_trace.get(
            "lora_loader_pair_apply_total_ms"
        ),
        "lora_loader_pair_apply_max_ms": scheduler_trace.get(
            "lora_loader_pair_apply_max_ms"
        ),
        "lora_loader_finalize_flashinfer_ms": scheduler_trace.get(
            "lora_loader_finalize_flashinfer_ms"
        ),
        "lora_loader_empty_cache_ms": scheduler_trace.get(
            "lora_loader_empty_cache_ms"
        ),
        "model_runner_rebuild_device_graphs_ms": scheduler_trace.get(
            "model_runner_rebuild_device_graphs_ms"
        ),
        "model_runner_init_device_graphs_ms": scheduler_trace.get(
            "model_runner_init_device_graphs_ms"
        ),
        "scheduler_worker_update_ms": scheduler_trace.get("scheduler_worker_update_ms"),
        "scheduler_update_total_ms": scheduler_trace.get("scheduler_update_total_ms"),
        "lora_loader_merge_impl": scheduler_trace.get("lora_loader_merge_impl"),
        "lora_loader_peak_device_budget_bytes": scheduler_trace.get(
            "lora_loader_peak_device_budget_bytes"
        ),
        "lora_loader_memory_budget_source": scheduler_trace.get(
            "lora_loader_memory_budget_source"
        ),
        "lora_loader_gpu_bucket_bytes": scheduler_trace.get(
            "lora_loader_gpu_bucket_bytes"
        ),
        "lora_loader_gpu_bucket_count": scheduler_trace.get(
            "lora_loader_gpu_bucket_count"
        ),
        "lora_loader_gpu_matmul_count": scheduler_trace.get(
            "lora_loader_gpu_matmul_count"
        ),
        "lora_loader_gpu_bucket_bytes_estimate_max": scheduler_trace.get(
            "lora_loader_gpu_bucket_bytes_estimate_max"
        ),
        "lora_loader_gpu_stage_lora_ms": scheduler_trace.get(
            "lora_loader_gpu_stage_lora_ms"
        ),
        "lora_loader_gpu_select_lora_ms": scheduler_trace.get(
            "lora_loader_gpu_select_lora_ms"
        ),
        "lora_loader_gpu_matmul_ms": scheduler_trace.get(
            "lora_loader_gpu_matmul_ms"
        ),
        "lora_loader_gpu_apply_ms": scheduler_trace.get("lora_loader_gpu_apply_ms"),
        "tokenizer_pre_pause_prepare_ms": trace.get("tokenizer_pre_pause_prepare_ms"),
        "tokenizer_lora_prestage_scheduler_communicator_ms": trace.get(
            "tokenizer_lora_prestage_scheduler_communicator_ms"
        ),
        "lora_prestage_scheduler_results": trace.get("lora_prestage_scheduler_results"),
        "lora_loader_prestage_consumed": scheduler_trace.get(
            "lora_loader_prestage_consumed"
        ),
        "lora_loader_prestage_hit_count": scheduler_trace.get(
            "lora_loader_prestage_hit_count"
        ),
        "lora_loader_prestage_miss_count": scheduler_trace.get(
            "lora_loader_prestage_miss_count"
        ),
        "lora_loader_prestage_staged_pair_count": scheduler_trace.get(
            "lora_loader_prestage_staged_pair_count"
        ),
        "lora_loader_prestage_unstaged_pair_count": scheduler_trace.get(
            "lora_loader_prestage_unstaged_pair_count"
        ),
        "lora_loader_prestage_staged_bytes": scheduler_trace.get(
            "lora_loader_prestage_staged_bytes"
        ),
        "lora_loader_prestage_unstaged_bytes": scheduler_trace.get(
            "lora_loader_prestage_unstaged_bytes"
        ),
        "lora_loader_prestage_max_apply_temp_bytes": scheduler_trace.get(
            "lora_loader_prestage_max_apply_temp_bytes"
        ),
        "lora_loader_prestage_capacity_bytes": scheduler_trace.get(
            "lora_loader_prestage_capacity_bytes"
        ),
        "lora_loader_prestage_apply_budget_fraction": scheduler_trace.get(
            "lora_loader_prestage_apply_budget_fraction"
        ),
        "lora_loader_prestage_complete": scheduler_trace.get(
            "lora_loader_prestage_complete"
        ),
        "lora_loader_memory_start": scheduler_trace.get("lora_loader_memory_start"),
        "lora_loader_memory_end": scheduler_trace.get("lora_loader_memory_end"),
        "lora_loader_first_pairs": [
            _compact_pair_trace(pair)
            for pair in scheduler_trace.get("lora_loader_first_pairs", [])
        ],
        "lora_loader_top_pairs": [
            _compact_pair_trace(pair)
            for pair in scheduler_trace.get("lora_loader_top_pairs", [])
        ],
    }


def _compact_update_result(update_result: dict[str, Any]) -> dict[str, Any]:
    body = update_result.get("body") if isinstance(update_result, dict) else None
    compact = {
        "status_code": update_result.get("status_code"),
        "elapsed_ms": update_result.get("elapsed_ms"),
    }
    if "error" in update_result:
        compact["error"] = update_result["error"]
    if isinstance(body, dict):
        route_trace = body.get("trace") or {}
        compact["body"] = {
            "success": body.get("success"),
            "message": body.get("message"),
            "request_id": body.get("request_id"),
            "weight_version": body.get("weight_version"),
            "tensor_count": body.get("tensor_count"),
            "adapter_load_ms": route_trace.get("adapter_load_ms"),
            "serialize_ms": route_trace.get("serialize_ms"),
            "update_weights_ms": route_trace.get("update_weights_ms"),
            "total_ms": route_trace.get("total_ms"),
            "trace_summary": _extract_update_trace_summary(update_result),
        }
    return compact


def _compact_load_report(load: Any) -> Any:
    if not isinstance(load, dict):
        return load
    compact = {
        "timestamp": load.get("timestamp"),
        "dp_rank_count": load.get("dp_rank_count"),
        "aggregate": load.get("aggregate"),
    }
    loads = load.get("loads")
    if isinstance(loads, list):
        compact["loads"] = [
            {
                "num_running_reqs": item.get("num_running_reqs"),
                "num_waiting_reqs": item.get("num_waiting_reqs"),
                "num_used_tokens": item.get("num_used_tokens"),
                "token_usage": item.get("token_usage"),
                "gen_throughput": item.get("gen_throughput"),
                "max_running_requests": item.get("max_running_requests"),
                "memory": item.get("memory"),
            }
            for item in loads
            if isinstance(item, dict)
        ]
    return compact


def _compact_active_gate(active_gate: dict[str, Any]) -> dict[str, Any]:
    compact = dict(active_gate)
    if "first_active_load" in compact:
        compact["first_active_load"] = _compact_load_report(compact["first_active_load"])
    if "load" in compact:
        compact["load"] = _compact_load_report(compact["load"])
    return compact


def _server_args() -> dict[str, Any]:
    return {
        "model_path": BASE_MODEL,
        "served_model_name": BASE_MODEL,
        "enable_multimodal": True,
        "reasoning_parser": "qwen3",
        "mem_fraction_static": float(MEM_FRACTION_STATIC),
        "chunked_prefill_size": int(CHUNKED_PREFILL_SIZE),
        "max_prefill_tokens": int(MAX_PREFILL_TOKENS),
        "kv_cache_dtype": KV_CACHE_DTYPE,
        "prefill_attention_backend": PREFILL_ATTENTION_BACKEND,
        "decode_attention_backend": DECODE_ATTENTION_BACKEND,
        "page_size": int(PAGE_SIZE),
        "moe_runner_backend": MOE_RUNNER_BACKEND,
        "mamba_scheduler_strategy": MAMBA_SCHEDULER_STRATEGY,
        "mamba_ssm_dtype": MAMBA_SSM_DTYPE,
        "cuda_graph_bs": list(range(1, int(CUDA_GRAPH_MAX_BS) + 1)),
        "cuda_graph_max_bs": int(CUDA_GRAPH_MAX_BS),
        "max_running_requests": int(MAX_RUNNING_REQUESTS),
        "custom_weight_loader": [MERGE_LOADER],
        "weight_version": "baseline",
        "device": "cuda",
    }


def _append_cli_arg(cmd: list[str], name: str, value: Any) -> None:
    flag = "--" + name.replace("_", "-")
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if isinstance(value, list):
        if not value:
            return
        cmd.append(flag)
        cmd.extend(str(item) for item in value)
        return
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _server_cmd() -> list[str]:
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
    ]
    for key, value in _server_args().items():
        if key in {"host", "port"}:
            continue
        _append_cli_arg(cmd, key, value)
    return cmd


def _runtime_fingerprint() -> dict[str, Any]:
    env_keys = (
        "HOSTNAME",
        "CUDA_VISIBLE_DEVICES",
        "MODAL_APP_ID",
        "MODAL_CLOUD_PROVIDER",
        "MODAL_CONTAINER_ID",
        "MODAL_ENVIRONMENT",
        "MODAL_FUNCTION_ID",
        "MODAL_REGION",
        "MODAL_TASK_ID",
    )
    result: dict[str, Any] = {
        "env": {key: os.environ[key] for key in env_keys if key in os.environ}
    }
    try:
        result["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,pci.bus_id,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
    except Exception as exc:
        result["nvidia_smi_error"] = repr(exc)
    return result


def _wait_ready(
    proc: subprocess.Popen[str],
    *,
    base_url: str,
    logs: list[str],
    timeout: float = 1200.0,
) -> None:
    import requests

    deadline = time.monotonic() + timeout
    last_error: str | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                "Server exited before becoming ready.\n" + "".join(logs[-200:])
            )
        try:
            response = requests.get(f"{base_url}/get_model_info", timeout=5)
            if response.ok:
                return
            last_error = f"{response.status_code}: {response.text[:500]}"
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for server readiness: {last_error}")


def _terminate_server(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=30)


@app.function(
    image=image,
    gpu=GPU,
    memory=MEMORY_MB,
    cpu=16,
    timeout=4 * 60 * 60,
    retries=0,
    secrets=[RUNTIME_CONFIG_SECRET],
    volumes={HF_CACHE_PATH: hf_cache_vol},
)
def run_latency_harness(
    concurrency: int = 16,
    max_new_tokens: int = 512,
    pre_update_delay_s: float = 3.0,
    prompt_tokens: int = 16,
    active_load_timeout_s: float = 120.0,
    active_load_min_running_reqs: int = 0,
    active_load_min_streams: int = 0,
    trace_top_k: int = 8,
    trace_sync: bool = False,
    sync_on_pause: bool = False,
    peak_device_bytes: str = "",
    gpu_bucket_bytes: str = "",
    vram_headroom_gb: float = 8.0,
    prestage_before_pause: bool = False,
    stream_server_logs: bool = False,
    include_server_log_tail: bool = False,
    full_trace: bool = False,
) -> dict[str, Any]:
    import requests

    os.environ.update(_py_env())
    runtime = _runtime_fingerprint()
    adapter_config_path = _remote_volume_path(ADAPTER_VOLUME_CONFIG_REL)
    adapter_weights_path = _remote_volume_path(ADAPTER_VOLUME_WEIGHTS_REL)

    env = _py_env()
    env["SGLANG_LORA_MERGE_TRACE_TOPK"] = str(trace_top_k)
    env["SGLANG_LORA_MERGE_TRACE_FIRST_N"] = str(trace_top_k)
    env["SGLANG_LORA_MERGE_TRACE_SYNC"] = "1" if trace_sync else "0"
    env["SGLANG_WEIGHT_UPDATE_SYNC_ON_PAUSE"] = "1" if sync_on_pause else "0"
    env["SGLANG_LORA_MERGE_VRAM_HEADROOM_GB"] = str(vram_headroom_gb)
    if peak_device_bytes:
        env["SGLANG_LORA_MERGE_PEAK_DEVICE_BYTES"] = peak_device_bytes
    elif gpu_bucket_bytes:
        env["SGLANG_LORA_MERGE_GPU_BUCKET_BYTES"] = gpu_bucket_bytes
    base_url = "http://127.0.0.1:30000"
    logs: list[str] = []
    proc = subprocess.Popen(
        _server_cmd(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert proc.stdout is not None

    def _read_logs() -> None:
        for line in proc.stdout:
            logs.append(line)
            if stream_server_logs:
                print(line, end="", flush=True)

    threading.Thread(target=_read_logs, daemon=True).start()
    _wait_ready(proc, base_url=base_url, logs=logs)
    records_lock = threading.Lock()
    started_streams = 0
    streams_with_events: set[int] = set()
    total_stream_events = 0
    completed_streams = 0

    def _stream_one(index: int) -> dict[str, Any]:
        nonlocal completed_streams, started_streams, total_stream_events
        prompt = " ".join([f"latency-{index}"] * prompt_tokens)
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "stream": True,
            "extra_key": f"latency-harness-{index}",
        }
        start = time.monotonic()
        events = []
        final_payload = None
        with records_lock:
            started_streams += 1
        try:
            with requests.post(
                f"{base_url}/generate",
                json=payload,
                stream=True,
                timeout=(30, 1800),
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line or not raw_line.startswith("data: "):
                        continue
                    data = raw_line[len("data: ") :]
                    now = time.monotonic()
                    if data == "[DONE]":
                        break
                    parsed = json.loads(data)
                    if "error" in parsed:
                        raise RuntimeError(parsed["error"])
                    final_payload = parsed
                    with records_lock:
                        total_stream_events += 1
                        streams_with_events.add(index)
                    events.append(
                        {
                            "t": now,
                            "text_len": len(parsed.get("text") or ""),
                            "has_meta_info": "meta_info" in parsed,
                        }
                    )
        except Exception as exc:
            with records_lock:
                completed_streams += 1
            return {
                "index": index,
                "start": start,
                "end": time.monotonic(),
                "events": events,
                "error": repr(exc),
                "final_meta_info": None,
            }

        with records_lock:
            completed_streams += 1
        return {
            "index": index,
            "start": start,
            "end": time.monotonic(),
            "events": events,
            "error": None,
            "final_meta_info": (final_payload or {}).get("meta_info"),
        }

    def _aggregate_running_reqs(load: Any) -> int | None:
        if not isinstance(load, dict):
            return None
        aggregate = load.get("aggregate")
        if isinstance(aggregate, dict) and "total_running_reqs" in aggregate:
            return int(aggregate["total_running_reqs"])
        loads = load.get("loads")
        if not isinstance(loads, list):
            return None
        total = 0
        for item in loads:
            if isinstance(item, dict):
                total += int(item.get("num_running_reqs") or 0)
        return total

    def _snapshot_stream_state() -> dict[str, Any]:
        with records_lock:
            return {
                "started_streams": started_streams,
                "streams_with_events": len(streams_with_events),
                "total_stream_events": total_stream_events,
                "completed_streams": completed_streams,
            }

    def _wait_for_active_decode() -> dict[str, Any]:
        min_running = active_load_min_running_reqs or max(1, concurrency // 2)
        min_streams = active_load_min_streams or max(1, min(concurrency, concurrency // 2))
        deadline = time.monotonic() + active_load_timeout_s
        first_active_load = None
        last_load = None
        last_error = None

        while time.monotonic() < deadline:
            stream_state = _snapshot_stream_state()
            try:
                load = requests.get(f"{base_url}/v1/loads", timeout=5).json()
                last_load = load
                running_reqs = _aggregate_running_reqs(load)
            except Exception as exc:
                last_error = repr(exc)
                running_reqs = None

            active = (
                running_reqs is not None
                and running_reqs >= min_running
                and stream_state["streams_with_events"] >= min_streams
                and stream_state["completed_streams"] < concurrency
            )
            if active:
                if first_active_load is None:
                    first_active_load = last_load
                if pre_update_delay_s > 0:
                    time.sleep(pre_update_delay_s)
                    stream_state = _snapshot_stream_state()
                    try:
                        load = requests.get(f"{base_url}/v1/loads", timeout=5).json()
                        last_load = load
                        running_reqs = _aggregate_running_reqs(load)
                    except Exception as exc:
                        last_error = repr(exc)
                        running_reqs = None
                    active = (
                        running_reqs is not None
                        and running_reqs >= min_running
                        and stream_state["streams_with_events"] >= min_streams
                        and stream_state["completed_streams"] < concurrency
                    )
                    if not active:
                        time.sleep(0.05)
                        continue
                return {
                    "success": True,
                    "min_running_reqs": min_running,
                    "min_streams_with_events": min_streams,
                    "running_reqs": running_reqs,
                    "stream_state": stream_state,
                    "first_active_load": first_active_load,
                    "load": last_load,
                    "last_error": last_error,
                }
            time.sleep(0.05)

        return {
            "success": False,
            "min_running_reqs": min_running,
            "min_streams_with_events": min_streams,
            "running_reqs": _aggregate_running_reqs(last_load),
            "stream_state": _snapshot_stream_state(),
            "load": last_load,
            "last_error": last_error,
        }

    def _merge_adapter() -> dict[str, Any]:
        payload = {
            "adapter_config_path": str(adapter_config_path),
            "adapter_weights_path": str(adapter_weights_path),
            "strict": True,
            "flush_cache": FLUSH_CACHE.lower() in ("1", "true", "yes", "on"),
            "atomic_pause_mode": ATOMIC_PAUSE_MODE,
            "weight_version": WEIGHT_VERSION,
            "prestage_before_pause": prestage_before_pause,
            "vram_headroom_gb": vram_headroom_gb,
        }
        if peak_device_bytes:
            payload["peak_device_bytes"] = peak_device_bytes
        started_at = time.monotonic()
        try:
            response = requests.post(
                f"{base_url}/admin/update_merged_lora_from_local",
                json=payload,
                timeout=1800,
            )
        except Exception as exc:
            ended_at = time.monotonic()
            return {
                "status_code": None,
                "elapsed_ms": round((ended_at - started_at) * 1000, 3),
                "error": repr(exc),
            }
        ended_at = time.monotonic()
        return {
            "status_code": response.status_code,
            "elapsed_ms": round((ended_at - started_at) * 1000, 3),
            "body": response.json(),
        }

    try:
        warmup = requests.post(
            f"{base_url}/generate",
            json={
                "text": "warmup",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=600,
        )
        warmup.raise_for_status()

        with ThreadPoolExecutor(max_workers=concurrency + 1) as executor:
            stream_futures = [
                executor.submit(_stream_one, index) for index in range(concurrency)
            ]
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                with records_lock:
                    if started_streams >= concurrency:
                        break
                time.sleep(0.05)
            active_gate = _wait_for_active_decode()
            load_before_update = active_gate.get("load") or {
                "error": active_gate.get("last_error") or "active decode gate failed"
            }

            update_start = time.monotonic()
            update_future = executor.submit(_merge_adapter)
            update_result = update_future.result()
            update_end = time.monotonic()

            records = [future.result() for future in as_completed(stream_futures)]

        errors = [record for record in records if record["error"]]
        if update_result["status_code"] is None or update_result["status_code"] >= 400:
            errors.append({"update_error": update_result})

        meta_infos = [
            record["final_meta_info"]
            for record in records
            if record.get("final_meta_info") is not None
        ]
        mixed_meta_infos = [
            meta for meta in meta_infos if meta.get("mixed_weight_epochs")
        ]

        returned_update_result = (
            update_result if full_trace else _compact_update_result(update_result)
        )
        result = {
            "base_model": BASE_MODEL,
            "gpu": GPU,
            "image_tag": SGLANG_IMAGE_TAG,
            "runtime": runtime,
            "concurrency": concurrency,
            "max_new_tokens": max_new_tokens,
            "pre_update_delay_s": pre_update_delay_s,
            "active_load_timeout_s": active_load_timeout_s,
            "active_load_min_running_reqs": active_load_min_running_reqs,
            "active_load_min_streams": active_load_min_streams,
            "trace_top_k": trace_top_k,
            "trace_sync": trace_sync,
            "sync_on_pause": sync_on_pause,
            "peak_device_bytes": peak_device_bytes,
            "gpu_bucket_bytes": gpu_bucket_bytes,
            "vram_headroom_gb": vram_headroom_gb,
            "prestage_before_pause": prestage_before_pause,
            "stream_server_logs": stream_server_logs,
            "atomic_pause_mode": ATOMIC_PAUSE_MODE,
            "flush_cache": FLUSH_CACHE.lower() in ("1", "true", "yes", "on"),
            "active_gate": _compact_active_gate(active_gate),
            "load_before_update": _compact_load_report(load_before_update),
            "update_result": returned_update_result,
            "update_trace_summary": _extract_update_trace_summary(update_result),
            "summary": _summarize_streams(
                records, update_start=update_start, update_end=update_end
            ),
            "stream_error_count": len(errors),
            "stream_errors": errors[:4],
            "mixed_response_count": len(mixed_meta_infos),
            "sample_mixed_meta_info": (
                _compact_meta_info(mixed_meta_infos[0]) if mixed_meta_infos else None
            ),
            "sample_final_meta_info": (
                _compact_meta_info(meta_infos[0]) if meta_infos else None
            ),
        }
        if include_server_log_tail:
            result["server_log_tail"] = "".join(logs[-200:])
        return result
    finally:
        _terminate_server(proc)


@app.local_entrypoint()
def main(
    concurrency: int = 16,
    max_new_tokens: int = 512,
    pre_update_delay_s: float = 3.0,
    prompt_tokens: int = 16,
    active_load_timeout_s: float = 120.0,
    active_load_min_running_reqs: int = 0,
    active_load_min_streams: int = 0,
    trace_top_k: int = 8,
    trace_sync: bool = False,
    sync_on_pause: bool = False,
    peak_device_bytes: str = "",
    gpu_bucket_bytes: str = "",
    vram_headroom_gb: float = 8.0,
    prestage_before_pause: bool = False,
    stream_server_logs: bool = False,
    include_server_log_tail: bool = False,
    full_trace: bool = False,
) -> None:
    upload_info = _upload_adapter_assets_to_volume()
    with modal.enable_output():
        result = run_latency_harness.remote(
            concurrency=concurrency,
            max_new_tokens=max_new_tokens,
            pre_update_delay_s=pre_update_delay_s,
            prompt_tokens=prompt_tokens,
            active_load_timeout_s=active_load_timeout_s,
            active_load_min_running_reqs=active_load_min_running_reqs,
            active_load_min_streams=active_load_min_streams,
            trace_top_k=trace_top_k,
            trace_sync=trace_sync,
            sync_on_pause=sync_on_pause,
            peak_device_bytes=peak_device_bytes,
            gpu_bucket_bytes=gpu_bucket_bytes,
            vram_headroom_gb=vram_headroom_gb,
            prestage_before_pause=prestage_before_pause,
            stream_server_logs=stream_server_logs,
            include_server_log_tail=include_server_log_tail,
            full_trace=full_trace,
        )
    print(json.dumps({"upload_info": upload_info, "result": result}, indent=2))
    if result["stream_error_count"]:
        raise SystemExit(1)
