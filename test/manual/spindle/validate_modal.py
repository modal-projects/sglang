"""Bounded validation of the local SGLang upgrade on Modal."""

import json
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[3]
app = modal.App("spindle-sglang-v0523-validation")
image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.23")
    .entrypoint([])
    .env({"SGLANG_RUST_BUILD_MODE": "auto", "SGLANG_DISABLE_CUDNN_CHECK": "1"})
    .add_local_dir(
        ROOT / "python",
        "/sgl-workspace/sglang/python",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc", "**/.pytest_cache/**"],
    )
    .add_local_dir(ROOT / "rust", "/sgl-workspace/sglang/rust", copy=True)
    .add_local_dir(
        ROOT / "test/registered",
        "/sgl-workspace/sglang/test/registered",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc", "**/.pytest_cache/**"],
    )
    .pip_install("pytest", "pytest-asyncio", "pytest-timeout", "xxhash", "zstandard")
)

UNIT_FILES = [
    "/sgl-workspace/sglang/python/sglang/srt/weight_sync",
    "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/weight_updater_test.py",
    "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/weight_updater_test.py",
    "/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer_test.py",
    "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4_test.py",
    "/sgl-workspace/sglang/python/sglang/srt/models/kimi_k3_test.py",
    "/sgl-workspace/sglang/python/sglang/srt/arg_groups/validation_hook_test.py",
    "/sgl-workspace/sglang/test/registered/unit/lora/test_request_lifetime.py",
    "/sgl-workspace/sglang/test/registered/unit/lora/test_spindle_adapter_preparation.py",
    "/sgl-workspace/sglang/test/registered/unit/lora/test_deepseek_v4_lora_unit.py",
    "/sgl-workspace/sglang/test/registered/unit/lora/test_mem_pool_ep_unit.py",
    "/sgl-workspace/sglang/test/registered/unit/managers/test_data_parallel_controller.py",
    "/sgl-workspace/sglang/test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py",
]


@app.function(image=image, cpu=8, memory=32768, timeout=1800)
def units():
    import sglang
    import torch

    print(
        json.dumps({"sglang": sglang.__file__, "torch": torch.__version__}), flush=True
    )
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--timeout=90", *UNIT_FILES], check=True
    )
    return "unit tests passed"


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=1800)
def sampler():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            "/sgl-workspace/sglang/test/registered/lora/test_lora_qwen3_5_4b_logprob_diff.py",
        ],
        check=True,
    )
    return "Qwen3.5-4B sampler logprob parity passed"


@app.function(
    image=image, gpu=["B300:4", "B200:4"], cpu=24, memory=196608, timeout=3600
)
def moe():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            "/sgl-workspace/sglang/test/registered/lora/test_lora_qwen3_5_35b_a3b_logprob_diff.py",
        ],
        check=True,
    )
    return "Qwen3.5-35B-A3B TP4 MoE logprob parity passed"


@app.function(image=image, gpu="B200", cpu=8, memory=32768, timeout=1200)
def marlin():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            "/sgl-workspace/sglang/test/registered/unit/lora/test_mxfp4_marlin_lora.py",
        ],
        check=True,
    )
    return "GPT-OSS MXFP4 Marlin kernel parity passed on B200"


@app.local_entrypoint()
def main(gpu: bool = False, mode: str = "all"):
    functions = {"units": units, "dense": sampler, "moe": moe, "marlin": marlin}
    selected = (
        list(functions)
        if mode == "all" and gpu
        else [mode if mode != "all" else "units"]
    )
    calls = [(name, functions[name].spawn()) for name in selected]
    errors = []
    for name, call in calls:
        print("CHECK", name, call.object_id, flush=True)
    for name, call in calls:
        try:
            print("RESULT", name, call.get(), flush=True)
        except Exception as exc:
            print("FAILED", name, str(exc), flush=True)
            errors.append(name)
    if errors:
        raise RuntimeError("Failed checks: " + ", ".join(errors))
