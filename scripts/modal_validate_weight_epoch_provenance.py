from __future__ import annotations

import json
import os
import pathlib
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

APP_NAME = "sglang-weight-epoch-validation"
PORT = 30000
HF_CACHE_PATH = "/root/.cache/huggingface"
HF_CACHE_VOLUME_NAME = os.getenv("HF_CACHE_VOLUME_NAME", "huggingface-cache")
SGLANG_IMAGE_TAG = os.getenv(
    "SGLANG_MODAL_IMAGE_TAG",
    "lmsysorg/sglang:nightly-dev-cu13-20260401-b6fe0cca",
)
GPU = os.getenv("SGLANG_MODAL_GPU", "A10G")
RUNTIME_CONFIG_SECRET = modal.Secret.from_dict(
    {
        "SGLANG_MODAL_IMAGE_TAG": SGLANG_IMAGE_TAG,
        "SGLANG_MODAL_GPU": GPU,
    }
)

HF_IMAGE_ENV = {
    "HF_HUB_CACHE": HF_CACHE_PATH,
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

SOURCE_DIRS = [
    (
        REPO_ROOT / "python/sglang",
        "/sgl-workspace/sglang/python/sglang",
    ),
    (
        REPO_ROOT / "test/registered/unit/managers",
        "/sgl-workspace/sglang/test/registered/unit/managers",
    ),
]

UNIT_TESTS = [
    "/sgl-workspace/sglang/test/registered/unit/managers/test_weight_epoch_provenance.py",
    "/sgl-workspace/sglang/test/registered/unit/managers/test_scheduler_pause_generation.py",
]

app = modal.App(name=APP_NAME)
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

image = modal.Image.from_registry(SGLANG_IMAGE_TAG).env(HF_IMAGE_ENV)
if modal.is_local():
    for local_path, remote_path in SOURCE_DIRS:
        image = image.add_local_dir(
            local_path,
            remote_path,
            copy=False,
        )


def _py_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(HF_IMAGE_ENV)
    env["PYTHONPATH"] = "/sgl-workspace/sglang/python:/sgl-workspace/sglang"
    return env


def _tail(lines: list[str], limit: int = 120) -> str:
    return "".join(lines[-limit:])


def _run_checked(
    cmd: list[str],
    *,
    env: dict[str, str],
    timeout: int | None = None,
) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    result = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def _wait_ready(
    proc: subprocess.Popen[str],
    *,
    base_url: str,
    logs: list[str],
    timeout: float = 1200.0,
    poll_interval: float = 2.0,
) -> None:
    import requests

    deadline = time.time() + timeout
    last_error: str | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Server exited before becoming ready.\n{_tail(logs)}"
            )
        try:
            response = requests.get(f"{base_url}/get_model_info", timeout=5)
            if response.ok:
                return
            last_error = f"{response.status_code}: {response.text}"
        except Exception as exc:  # pragma: no cover - diagnostic path
            last_error = repr(exc)
        time.sleep(poll_interval)

    raise TimeoutError(
        f"Timed out waiting for server readiness. last_error={last_error}\n{_tail(logs)}"
    )


def _server_cmd(model_path: str, weight_version: str) -> list[str]:
    return [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--weight-version",
        weight_version,
        "--mem-fraction-static",
        "0.72",
        "--chunked-prefill-size",
        "2048",
        "--max-prefill-tokens",
        "2048",
        "--cuda-graph-max-bs",
        "4",
        "--max-running-requests",
        "8",
        "--log-level-http",
        "warning",
    ]


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> Any:
    import requests

    try:
        response = requests.request(method, url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        raise RuntimeError(
            f"HTTP request failed: method={method} url={url} payload={payload!r} error={exc!r}"
        ) from None


def _wait_for_running_requests(
    base_url: str,
    *,
    min_running: int = 1,
    timeout: float = 30.0,
    poll_interval: float = 0.2,
) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_load = None
    while time.time() < deadline:
        last_load = _request_json("get", f"{base_url}/v1/loads", timeout=30)
        aggregate = last_load.get("aggregate", {})
        if aggregate.get("total_running_reqs", 0) >= min_running:
            return last_load
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for running requests. last_load={last_load}")


def _require_keys(mapping: dict[str, Any], keys: list[str], label: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise AssertionError(f"{label} missing keys: {missing}; payload={mapping}")


def _extract_finish_type(output: dict[str, Any]) -> str | None:
    finish_reason = output.get("meta_info", {}).get("finish_reason")
    if isinstance(finish_reason, dict):
        return finish_reason.get("type")
    if isinstance(finish_reason, str):
        return finish_reason
    return None


def _generate_request(
    base_url: str,
    *,
    text: str,
    extra_key: str | None = None,
    max_new_tokens: int = 8,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": ignore_eos,
        },
    }
    if extra_key is not None:
        payload["extra_key"] = extra_key
    return _request_json("post", f"{base_url}/generate", payload=payload, timeout=600)


def _run_non_blocking_update(
    base_url: str,
    *,
    pause_mode: str | None,
    atomic_pause_mode: str | None,
    next_model: str,
    next_weight_version: str,
    flush_cache: bool,
    request_count: int,
    max_new_tokens: int,
    logs: list[str],
) -> dict[str, Any]:
    if (pause_mode is None) == (atomic_pause_mode is None):
        raise ValueError(
            "Exactly one of pause_mode or atomic_pause_mode must be provided."
        )

    def _decode(idx: int) -> dict[str, Any]:
        return _generate_request(
            base_url,
            text=f"Question {idx}: The capital of France is",
            extra_key=f"tenant-{idx}",
            max_new_tokens=max_new_tokens,
            ignore_eos=True,
        )

    with ThreadPoolExecutor(max_workers=request_count) as executor:
        futures = [executor.submit(_decode, idx) for idx in range(request_count)]
        load_before_pause = _wait_for_running_requests(base_url, min_running=1)
        pause_ret = None
        if pause_mode is not None:
            pause_ret = _request_json(
                "post",
                f"{base_url}/pause_generation",
                payload={"mode": pause_mode},
                timeout=120,
            )
            time.sleep(0.5)

        update_payload = {
            "model_path": next_model,
            "flush_cache": flush_cache,
            "weight_version": next_weight_version,
        }
        if atomic_pause_mode is not None:
            update_payload["atomic_pause_mode"] = atomic_pause_mode
        update_ret = _request_json(
            "post",
            f"{base_url}/update_weights_from_disk",
            payload=update_payload,
            timeout=1200,
        )
        continue_ret = None
        if pause_mode is not None:
            continue_ret = _request_json(
                "post",
                f"{base_url}/continue_generation",
                payload={},
                timeout=120,
            )
        outputs = []
        request_errors = []
        for future in as_completed(futures):
            try:
                outputs.append(future.result())
            except Exception as exc:
                request_errors.append(repr(exc))

    if request_errors:
        raise RuntimeError(
            f"{pause_mode} request errors: {request_errors}\n{_tail(logs)}"
        )

    finish_types = [_extract_finish_type(output) for output in outputs]
    aborted = [finish_type for finish_type in finish_types if finish_type == "abort"]
    if aborted:
        raise AssertionError(
            f"{pause_mode} update aborted requests: finish_types={finish_types}"
        )

    return {
        "pause_mode": pause_mode,
        "atomic_pause_mode": atomic_pause_mode,
        "load_before_pause": load_before_pause,
        "pause_response": pause_ret,
        "update_response": update_ret,
        "continue_response": continue_ret,
        "finish_types": finish_types,
        "outputs": outputs,
    }


@app.function(
    image=image,
    gpu=GPU,
    timeout=90 * 60,
    volumes={HF_CACHE_PATH: hf_cache_vol},
    secrets=[RUNTIME_CONFIG_SECRET],
)
def run_validation(
    instruct_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    base_model: str = "Qwen/Qwen2.5-0.5B",
    request_count: int = 8,
    max_new_tokens: int = 512,
    require_mixed_in_place: bool = True,
    run_retract: bool = True,
) -> dict[str, Any]:
    runtime_image_tag = os.getenv("SGLANG_MODAL_IMAGE_TAG", SGLANG_IMAGE_TAG)
    runtime_gpu = os.getenv("SGLANG_MODAL_GPU", GPU)
    expected_meta_keys = [
        "weight_version",
        "weight_version_start",
        "weight_version_end",
        "weight_epoch_start",
        "weight_epoch_end",
        "cache_epoch",
        "mixed_weight_epochs",
        "resume_from_stale_kv",
        "output_token_weight_epochs",
    ]

    env = _py_env()
    unit_test_results = []
    for test_path in UNIT_TESTS:
        unit_test_results.append(
            _run_checked(["python", test_path], env=env, timeout=600)
        )

    proc: subprocess.Popen[str] | None = None
    logs: list[str] = []
    try:
        proc = subprocess.Popen(
            _server_cmd(instruct_model, "modal-instruct"),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None

        def _reader() -> None:
            for line in proc.stdout:
                logs.append(line)

        threading.Thread(target=_reader, daemon=True).start()

        base_url = f"http://127.0.0.1:{PORT}"
        _wait_ready(proc, base_url=base_url, logs=logs)

        model_info_start = _request_json("get", f"{base_url}/get_model_info")
        if model_info_start["weight_epoch"] != 0:
            raise AssertionError(f"Unexpected initial model_info: {model_info_start}")
        if model_info_start["weight_version"] != "modal-instruct":
            raise AssertionError(f"Unexpected initial model_info: {model_info_start}")

        raw_generate = _generate_request(
            base_url,
            text="hello",
            extra_key="tenant-a",
            max_new_tokens=8,
        )
        _require_keys(raw_generate["meta_info"], expected_meta_keys, "raw generate")
        if raw_generate["meta_info"]["weight_epoch_start"] != 0:
            raise AssertionError(raw_generate["meta_info"])
        if raw_generate["meta_info"]["weight_epoch_end"] != 0:
            raise AssertionError(raw_generate["meta_info"])

        chat_completion = _request_json(
            "post",
            f"{base_url}/v1/chat/completions",
            payload={
                "model": instruct_model,
                "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                "temperature": 0,
                "max_tokens": 8,
            },
            timeout=300,
        )
        _require_keys(chat_completion["metadata"], expected_meta_keys, "chat metadata")

        completion = _request_json(
            "post",
            f"{base_url}/v1/completions",
            payload={
                "model": instruct_model,
                "prompt": "hello",
                "temperature": 0,
                "max_tokens": 8,
            },
            timeout=300,
        )
        _require_keys(completion["metadata"], expected_meta_keys, "completion metadata")

        in_place_run = _run_non_blocking_update(
            base_url,
            pause_mode=None,
            atomic_pause_mode="in_place",
            next_model=base_model,
            next_weight_version="modal-base",
            flush_cache=False,
            request_count=request_count,
            max_new_tokens=max_new_tokens,
            logs=logs,
        )
        print(
            json.dumps(
                {
                    "stage": "after_in_place",
                    "load_before_pause": in_place_run["load_before_pause"],
                    "pause_response": in_place_run["pause_response"],
                    "update_response": in_place_run["update_response"],
                    "continue_response": in_place_run["continue_response"],
                    "finish_types": in_place_run["finish_types"],
                    "all_meta_info": [
                        output["meta_info"] for output in in_place_run["outputs"]
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        model_info_after_in_place = _request_json("get", f"{base_url}/get_model_info")
        if model_info_after_in_place["weight_epoch"] != 1:
            raise AssertionError(model_info_after_in_place)
        if model_info_after_in_place["weight_version"] != "modal-base":
            raise AssertionError(model_info_after_in_place)

        mixed_outputs = [
            output
            for output in in_place_run["outputs"]
            if output["meta_info"].get("mixed_weight_epochs")
        ]
        if require_mixed_in_place and not mixed_outputs:
            raise AssertionError(
                f"Expected at least one mixed-epoch response.\n{json.dumps(in_place_run, indent=2)[:4000]}"
            )
        mixed_meta = mixed_outputs[0]["meta_info"] if mixed_outputs else None
        if mixed_meta is not None:
            if mixed_meta["weight_epoch_start"] != 0:
                raise AssertionError(mixed_meta)
            if mixed_meta["weight_epoch_end"] != 1:
                raise AssertionError(mixed_meta)
            if mixed_meta["cache_epoch"] != 0:
                raise AssertionError(mixed_meta)
            if not mixed_meta["resume_from_stale_kv"]:
                raise AssertionError(mixed_meta)
            if mixed_meta["weight_version_start"] != "modal-instruct":
                raise AssertionError(mixed_meta)
            if mixed_meta["weight_version_end"] != "modal-base":
                raise AssertionError(mixed_meta)
            if set(mixed_meta["output_token_weight_epochs"]) != {0, 1}:
                raise AssertionError(mixed_meta)

        retract_run = None
        model_info_after_retract = None
        final_generate = None
        if run_retract:
            retract_run = _run_non_blocking_update(
                base_url,
                pause_mode=None,
                atomic_pause_mode="retract",
                next_model=instruct_model,
                next_weight_version="modal-instruct-return",
                flush_cache=True,
                request_count=request_count,
                max_new_tokens=max_new_tokens,
                logs=logs,
            )
            model_info_after_retract = _request_json("get", f"{base_url}/get_model_info")
            if model_info_after_retract["weight_epoch"] != 2:
                raise AssertionError(model_info_after_retract)
            if model_info_after_retract["weight_version"] != "modal-instruct-return":
                raise AssertionError(model_info_after_retract)

            final_generate = _generate_request(
                base_url,
                text="hello again",
                extra_key="tenant-a",
                max_new_tokens=8,
            )
            _require_keys(final_generate["meta_info"], expected_meta_keys, "final generate")
            if final_generate["meta_info"]["weight_epoch_start"] != 2:
                raise AssertionError(final_generate["meta_info"])
            if final_generate["meta_info"]["weight_epoch_end"] != 2:
                raise AssertionError(final_generate["meta_info"])

        return {
            "image_tag": runtime_image_tag,
            "gpu": runtime_gpu,
            "instruct_model": instruct_model,
            "base_model": base_model,
            "unit_tests": [
                {
                    "cmd": result["cmd"],
                    "returncode": result["returncode"],
                    "stdout_tail": result["stdout"][-2000:],
                    "stderr_tail": result["stderr"][-2000:],
                }
                for result in unit_test_results
            ],
            "model_info_start": model_info_start,
            "raw_generate_meta_info": raw_generate["meta_info"],
            "chat_metadata": chat_completion["metadata"],
            "completion_metadata": completion["metadata"],
            "model_info_after_in_place": model_info_after_in_place,
            "in_place_all_meta_info": [
                output["meta_info"] for output in in_place_run["outputs"]
            ],
            "in_place_sample_meta_info": mixed_meta,
            "in_place_finish_types": in_place_run["finish_types"],
            "model_info_after_retract": model_info_after_retract,
            "retract_all_meta_info": [
                output["meta_info"] for output in retract_run["outputs"]
            ]
            if retract_run is not None
            else None,
            "retract_finish_types": retract_run["finish_types"]
            if retract_run is not None
            else None,
            "final_generate_meta_info": final_generate["meta_info"]
            if final_generate is not None
            else None,
            "server_log_tail": _tail(logs),
        }
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)


@app.local_entrypoint()
def main(
    instruct_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    base_model: str = "Qwen/Qwen2.5-0.5B",
    request_count: int = 8,
    max_new_tokens: int = 512,
    require_mixed_in_place: bool = True,
    run_retract: bool = True,
) -> None:
    with modal.enable_output():
        result = run_validation.remote(
            instruct_model=instruct_model,
            base_model=base_model,
            request_count=request_count,
            max_new_tokens=max_new_tokens,
            require_mixed_in_place=require_mixed_in_place,
            run_retract=run_retract,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
