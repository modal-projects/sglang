from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import pathlib
import re
import shutil
import sys
import unittest
from typing import Any

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOTE_REPO_ROOT = pathlib.Path("/sgl-workspace/sglang")
REMOTE_TMP_DIR = pathlib.Path("/tmp/qwen35-merged-lora-logprob-diff")

APP_NAME = "sglang-qwen35-merged-lora-logprob-diff"
HF_CACHE_PATH = "/root/.cache/huggingface"
HF_CACHE_VOLUME_NAME = os.getenv("HF_CACHE_VOLUME_NAME", "huggingface-cache")
SGLANG_IMAGE_TAG = os.getenv(
    "SGLANG_MODAL_IMAGE_TAG",
    "lmsysorg/sglang:nightly-dev-cu13-20260407-5cc246e0",
)
GPU = os.getenv("SGLANG_MODAL_GPU", "H100")

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

BASE_MODEL = os.getenv("QWEN35_BASE_MODEL", "Qwen/Qwen3.5-35B-A3B")
MAX_NEW_TOKENS = os.getenv("QWEN35_MERGE_MAX_NEW_TOKENS", "48")
MERGE_MAX_ABS_THRESHOLD = os.getenv("QWEN35_MERGE_MAX_ABS_THRESHOLD", "5e-2")
MERGE_MEAN_ABS_THRESHOLD = os.getenv("QWEN35_MERGE_MEAN_ABS_THRESHOLD", "5e-3")
MERGE_DTYPE = os.getenv("QWEN35_MERGE_DTYPE", "bfloat16")
FORCE_ATTENTION_BACKEND = os.getenv("QWEN35_FORCE_ATTENTION_BACKEND", "")
SGLANG_ENABLE_DETERMINISTIC_INFERENCE = os.getenv(
    "QWEN35_SGLANG_ENABLE_DETERMINISTIC_INFERENCE", "1"
)

ADAPTER_VOLUME_SUBDIR = pathlib.PurePosixPath(
    "local-adapters/qwen35-merged-lora-logprob-diff"
)
ADAPTER_VOLUME_CONFIG_REL = ADAPTER_VOLUME_SUBDIR / "adapter_config.json"
ADAPTER_VOLUME_WEIGHTS_REL = ADAPTER_VOLUME_SUBDIR / "sampler_weights_init.safetensors"

RUNTIME_CONFIG_SECRET = modal.Secret.from_dict(
    {
        "SGLANG_MODAL_IMAGE_TAG": SGLANG_IMAGE_TAG,
        "SGLANG_MODAL_GPU": GPU,
        "QWEN35_BASE_MODEL": BASE_MODEL,
        "QWEN35_MERGE_MAX_NEW_TOKENS": MAX_NEW_TOKENS,
        "QWEN35_MERGE_MAX_ABS_THRESHOLD": MERGE_MAX_ABS_THRESHOLD,
        "QWEN35_MERGE_MEAN_ABS_THRESHOLD": MERGE_MEAN_ABS_THRESHOLD,
        "QWEN35_MERGE_DTYPE": MERGE_DTYPE,
        "QWEN35_FORCE_ATTENTION_BACKEND": FORCE_ATTENTION_BACKEND,
        "QWEN35_SGLANG_ENABLE_DETERMINISTIC_INFERENCE": (
            SGLANG_ENABLE_DETERMINISTIC_INFERENCE
        ),
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
        str(REMOTE_REPO_ROOT / "python/sglang"),
    ),
    (
        REPO_ROOT / "test/manual",
        str(REMOTE_REPO_ROOT / "test/manual"),
    ),
]

REQUESTED_TEST_CMD = [
    "python3",
    "-m",
    "unittest",
    "test.manual.test_qwen35_merged_lora_logprob_diff",
    "-v",
]
TEST_FILE = REMOTE_REPO_ROOT / "test/manual/test_qwen35_merged_lora_logprob_diff.py"

app = modal.App(name=APP_NAME)
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(SGLANG_IMAGE_TAG)
    .env(HF_IMAGE_ENV)
    .pip_install("accelerate", "peft>=0.18.0")
)
if modal.is_local():
    for local_path, remote_path in SOURCE_DIRS:
        image = image.add_local_dir(local_path, remote_path, copy=False)


def _tail(text: str, limit: int = 16000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _remote_pythonpath() -> str:
    return f"{REMOTE_REPO_ROOT / 'python'}:{REMOTE_REPO_ROOT}"


def _remote_volume_path(relative_path: pathlib.PurePosixPath) -> pathlib.Path:
    return pathlib.Path(HF_CACHE_PATH) / pathlib.Path(str(relative_path))


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
        batch.put_file(
            str(LOCAL_ADAPTER_CONFIG_PATH),
            str(ADAPTER_VOLUME_CONFIG_REL),
        )
        batch.put_file(
            str(LOCAL_ADAPTER_WEIGHTS_PATH),
            str(ADAPTER_VOLUME_WEIGHTS_REL),
        )

    return {
        "local_adapter_dir": str(LOCAL_ADAPTER_DIR),
        "local_adapter_config": str(LOCAL_ADAPTER_CONFIG_PATH),
        "local_adapter_weights": str(LOCAL_ADAPTER_WEIGHTS_PATH),
        "local_adapter_weights_size_bytes": LOCAL_ADAPTER_WEIGHTS_PATH.stat().st_size,
        "volume_adapter_config": str(_remote_volume_path(ADAPTER_VOLUME_CONFIG_REL)),
        "volume_adapter_weights": str(_remote_volume_path(ADAPTER_VOLUME_WEIGHTS_REL)),
    }


def _extract_metric(output: str, name: str) -> float | None:
    match = re.search(rf"{re.escape(name)}\s*=\s*([0-9.eE+-]+)", output)
    if match is None:
        return None
    return float(match.group(1))


def _load_test_module():
    module_name = "modal_test_qwen35_merged_lora_logprob_diff"
    spec = importlib.util.spec_from_file_location(module_name, TEST_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load test module from {TEST_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _run_test_module(*, force_attention_backend: str | None) -> dict[str, Any]:
    module = _load_test_module()
    if force_attention_backend:
        original_engine = module.sgl.Engine

        def _engine_with_attention_backend(*args, **kwargs):
            kwargs.setdefault("attention_backend", force_attention_backend)
            return original_engine(*args, **kwargs)

        module.sgl.Engine = _engine_with_attention_backend

    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    output = stream.getvalue()
    return {
        "successful": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "output": output,
    }


@app.function(
    image=image,
    gpu=GPU,
    timeout=4 * 60 * 60,
    secrets=[RUNTIME_CONFIG_SECRET],
    volumes={HF_CACHE_PATH: hf_cache_vol},
)
def run_qwen35_logprob_validation() -> dict[str, Any]:
    volume_config_path = _remote_volume_path(ADAPTER_VOLUME_CONFIG_REL)
    volume_weights_path = _remote_volume_path(ADAPTER_VOLUME_WEIGHTS_REL)
    runtime_adapter_dir = REMOTE_TMP_DIR / "adapter_dir"
    runtime_adapter_config_path = runtime_adapter_dir / "adapter_config.json"
    runtime_peft_weights_path = runtime_adapter_dir / "adapter_model.safetensors"

    runtime_adapter_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(volume_config_path, runtime_adapter_config_path)
    if runtime_peft_weights_path.exists() or runtime_peft_weights_path.is_symlink():
        runtime_peft_weights_path.unlink()
    runtime_peft_weights_path.symlink_to(volume_weights_path)

    env = os.environ.copy()
    env.update(HF_IMAGE_ENV)
    env["PYTHONPATH"] = _remote_pythonpath()
    env["QWEN35_BASE_MODEL"] = os.getenv("QWEN35_BASE_MODEL", BASE_MODEL)
    env["QWEN35_LORA_DIR"] = str(runtime_adapter_dir)
    env["QWEN35_LORA_CONFIG"] = str(runtime_adapter_config_path)
    env["QWEN35_LORA_WEIGHTS"] = str(volume_weights_path)
    env["QWEN35_MERGE_MAX_NEW_TOKENS"] = os.getenv(
        "QWEN35_MERGE_MAX_NEW_TOKENS", MAX_NEW_TOKENS
    )
    env["QWEN35_MERGE_MAX_ABS_THRESHOLD"] = os.getenv(
        "QWEN35_MERGE_MAX_ABS_THRESHOLD", MERGE_MAX_ABS_THRESHOLD
    )
    env["QWEN35_MERGE_MEAN_ABS_THRESHOLD"] = os.getenv(
        "QWEN35_MERGE_MEAN_ABS_THRESHOLD", MERGE_MEAN_ABS_THRESHOLD
    )
    env["QWEN35_MERGE_DTYPE"] = os.getenv("QWEN35_MERGE_DTYPE", MERGE_DTYPE)
    env["QWEN35_SGLANG_ENABLE_DETERMINISTIC_INFERENCE"] = os.getenv(
        "QWEN35_SGLANG_ENABLE_DETERMINISTIC_INFERENCE",
        SGLANG_ENABLE_DETERMINISTIC_INFERENCE,
    )
    force_attention_backend = os.getenv(
        "QWEN35_FORCE_ATTENTION_BACKEND", FORCE_ATTENTION_BACKEND
    ).strip()

    os.environ.update(env)
    test_result = _run_test_module(
        force_attention_backend=force_attention_backend or None
    )
    combined_output = test_result["output"]

    return {
        "image_tag": os.getenv("SGLANG_MODAL_IMAGE_TAG", SGLANG_IMAGE_TAG),
        "gpu": os.getenv("SGLANG_MODAL_GPU", GPU),
        "base_model": env["QWEN35_BASE_MODEL"],
        "pythonpath": env["PYTHONPATH"],
        "requested_test_cmd": REQUESTED_TEST_CMD,
        "effective_runner": "in_process_unittest_module",
        "test_file": str(TEST_FILE),
        "forced_attention_backend": force_attention_backend or None,
        "enable_deterministic_inference": env[
            "QWEN35_SGLANG_ENABLE_DETERMINISTIC_INFERENCE"
        ].strip().lower()
        in ("1", "true", "yes", "on"),
        "returncode": 0 if test_result["successful"] else 1,
        "successful": test_result["successful"],
        "tests_run": test_result["tests_run"],
        "failures": test_result["failures"],
        "errors": test_result["errors"],
        "runtime_adapter_dir": str(runtime_adapter_dir),
        "runtime_adapter_config": str(runtime_adapter_config_path),
        "runtime_peft_weights": str(runtime_peft_weights_path),
        "volume_adapter_config_exists": volume_config_path.exists(),
        "volume_adapter_weights_exists": volume_weights_path.exists(),
        "volume_adapter_weights_size_bytes": volume_weights_path.stat().st_size,
        "runtime_peft_weights_is_symlink": runtime_peft_weights_path.is_symlink(),
        "overall_max_abs": _extract_metric(combined_output, "max_abs"),
        "overall_mean_abs": _extract_metric(combined_output, "mean_abs"),
        "output_tail": _tail(combined_output),
    }


@app.local_entrypoint()
def main() -> None:
    upload_result = _upload_adapter_assets_to_volume()
    with modal.enable_output():
        result = run_qwen35_logprob_validation.remote()
    print(
        json.dumps(
            {
                "upload": upload_result,
                "validation": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
