from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import os
import pathlib
import re
import shutil
import sys
from typing import Any

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOTE_REPO_ROOT = pathlib.Path("/sgl-workspace/sglang")
REMOTE_TMP_DIR = pathlib.Path("/tmp/qwen35-merged-lora-logprob-diff")
TEST_FILE = REMOTE_REPO_ROOT / "test/manual/test_qwen35_merged_lora_logprob_diff.py"

APP_NAME = "sglang-qwen35-merged-lora-logprob-diff-split"
HF_CACHE_PATH = "/root/.cache/huggingface"
HF_CACHE_VOLUME_NAME = os.getenv("HF_CACHE_VOLUME_NAME", "huggingface-cache")
SGLANG_IMAGE_TAG = os.getenv(
    "SGLANG_MODAL_IMAGE_TAG",
    "lmsysorg/sglang:nightly-dev-cu13-20260407-5cc246e0",
)
HF_GPU = os.getenv("QWEN35_HF_MODAL_GPU", "H100")
SGLANG_GPU = os.getenv("QWEN35_SGLANG_MODAL_GPU", "H200")
SGLANG_MEMORY_MB = int(os.getenv("QWEN35_SGLANG_MODAL_MEMORY_MB", "131072"))
HF_INSTALL_CAUSAL_CONV1D = os.getenv(
    "QWEN35_HF_INSTALL_CAUSAL_CONV1D", "1"
).strip().lower() in ("1", "true", "yes", "on")
SGLANG_DISABLE_CUDA_GRAPH = os.getenv("QWEN35_SGLANG_DISABLE_CUDA_GRAPH", "")
SGLANG_MEM_FRACTION_STATIC = os.getenv("QWEN35_SGLANG_MEM_FRACTION_STATIC", "")
SGLANG_MOE_RUNNER_BACKEND = os.getenv("QWEN35_SGLANG_MOE_RUNNER_BACKEND", "")
HF_ATTN_IMPLEMENTATION = os.getenv("QWEN35_HF_ATTN_IMPLEMENTATION", "")
SGLANG_ENABLE_DETERMINISTIC_INFERENCE = os.getenv(
    "QWEN35_SGLANG_ENABLE_DETERMINISTIC_INFERENCE", "1"
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

BASE_MODEL = os.getenv("QWEN35_BASE_MODEL", "Qwen/Qwen3.5-35B-A3B")
MAX_NEW_TOKENS = int(os.getenv("QWEN35_MERGE_MAX_NEW_TOKENS", "48"))
MERGE_MAX_ABS_THRESHOLD = float(os.getenv("QWEN35_MERGE_MAX_ABS_THRESHOLD", "5e-2"))
MERGE_MEAN_ABS_THRESHOLD = float(os.getenv("QWEN35_MERGE_MEAN_ABS_THRESHOLD", "5e-3"))
MERGE_DTYPE = os.getenv("QWEN35_MERGE_DTYPE", "bfloat16")
SGLANG_ATTENTION_BACKEND = os.getenv("QWEN35_SGLANG_ATTENTION_BACKEND", "")
ADAPTER_SUBSET = os.getenv("QWEN35_ADAPTER_SUBSET", "all")
TRACE_PROMPT_INDEX = int(os.getenv("QWEN35_TRACE_PROMPT_INDEX", "2"))
TRACE_TOP_K = int(os.getenv("QWEN35_TRACE_TOP_K", "5"))
PROMPT_INDEXES = os.getenv("QWEN35_PROMPT_INDEXES", "").strip()

ADAPTER_VOLUME_SUBDIR = pathlib.PurePosixPath(
    "local-adapters/qwen35-merged-lora-logprob-diff"
)
ADAPTER_VOLUME_CONFIG_REL = ADAPTER_VOLUME_SUBDIR / "adapter_config.json"
ADAPTER_VOLUME_WEIGHTS_REL = ADAPTER_VOLUME_SUBDIR / "sampler_weights_init.safetensors"

RUNTIME_CONFIG_SECRET = modal.Secret.from_dict(
    {
        "SGLANG_MODAL_IMAGE_TAG": SGLANG_IMAGE_TAG,
        "QWEN35_HF_MODAL_GPU": HF_GPU,
        "QWEN35_SGLANG_MODAL_GPU": SGLANG_GPU,
        "QWEN35_BASE_MODEL": BASE_MODEL,
        "QWEN35_MERGE_MAX_NEW_TOKENS": str(MAX_NEW_TOKENS),
        "QWEN35_MERGE_MAX_ABS_THRESHOLD": str(MERGE_MAX_ABS_THRESHOLD),
        "QWEN35_MERGE_MEAN_ABS_THRESHOLD": str(MERGE_MEAN_ABS_THRESHOLD),
        "QWEN35_MERGE_DTYPE": MERGE_DTYPE,
        "QWEN35_ADAPTER_SUBSET": ADAPTER_SUBSET,
        "QWEN35_TRACE_PROMPT_INDEX": str(TRACE_PROMPT_INDEX),
        "QWEN35_TRACE_TOP_K": str(TRACE_TOP_K),
        "QWEN35_PROMPT_INDEXES": PROMPT_INDEXES,
        "QWEN35_HF_ATTN_IMPLEMENTATION": HF_ATTN_IMPLEMENTATION,
        "QWEN35_SGLANG_ATTENTION_BACKEND": SGLANG_ATTENTION_BACKEND,
        "QWEN35_SGLANG_DISABLE_CUDA_GRAPH": SGLANG_DISABLE_CUDA_GRAPH,
        "QWEN35_SGLANG_MEM_FRACTION_STATIC": SGLANG_MEM_FRACTION_STATIC,
        "QWEN35_SGLANG_MOE_RUNNER_BACKEND": SGLANG_MOE_RUNNER_BACKEND,
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

app = modal.App(name=APP_NAME)
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(SGLANG_IMAGE_TAG)
    .env(HF_IMAGE_ENV)
    # PEFT 0.19 raises if an incompatible torchao from the base SGLang image
    # is importable, even when torchao is not needed for this validation.
    .run_commands("python -m pip uninstall -y torchao")
)
_image_pip_packages = [
    "accelerate",
    "peft>=0.18.0",
    "einops",
    "datasets",
    "ninja",
    "fla-core",
    "flash-linear-attention",
]
image = image.pip_install(*_image_pip_packages, extra_options="--no-deps")
if HF_INSTALL_CAUSAL_CONV1D:
    image = image.pip_install(
        "causal-conv1d>=1.5.0.post8",
        extra_options="--no-deps --no-build-isolation",
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


def _ensure_import_paths() -> None:
    for path in (str(REMOTE_REPO_ROOT), str(REMOTE_REPO_ROOT / "python")):
        if path not in sys.path:
            sys.path.insert(0, path)
    importlib.invalidate_caches()


def _remote_volume_path(relative_path: pathlib.PurePosixPath) -> pathlib.Path:
    return pathlib.Path(HF_CACHE_PATH) / pathlib.Path(str(relative_path))


def _prepare_runtime_adapter_dir() -> tuple[pathlib.Path, pathlib.Path]:
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
    return runtime_adapter_dir, volume_weights_path


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
        "local_adapter_dir": str(LOCAL_ADAPTER_DIR),
        "local_adapter_config": str(LOCAL_ADAPTER_CONFIG_PATH),
        "local_adapter_weights": str(LOCAL_ADAPTER_WEIGHTS_PATH),
        "local_adapter_weights_size_bytes": LOCAL_ADAPTER_WEIGHTS_PATH.stat().st_size,
        "volume_adapter_config": str(_remote_volume_path(ADAPTER_VOLUME_CONFIG_REL)),
        "volume_adapter_weights": str(_remote_volume_path(ADAPTER_VOLUME_WEIGHTS_REL)),
    }


def _load_test_module():
    _ensure_import_paths()
    module_name = "modal_test_qwen35_merged_lora_logprob_diff_split"
    spec = importlib.util.spec_from_file_location(module_name, TEST_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load test module from {TEST_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _serialize_hf_results(hf_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for item in hf_results:
        serialized.append(
            {
                "prompt": item["prompt"],
                "prompt_len": item["prompt_len"],
                "full_ids": item["full_ids"],
                "completion_ids": item["completion_ids"],
                "completion_text": item["completion_text"],
                "hf_prefill_logprobs": item["hf_prefill_logprobs"].tolist(),
                "hf_completion_logprobs": item["hf_completion_logprobs"].tolist(),
            }
        )
    return serialized


def _serialize_hf_base_results(
    hf_base_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for item in hf_base_results:
        serialized.append(
            {
                "prompt": item["prompt"],
                "hf_base_prefill_logprobs": item["hf_base_prefill_logprobs"].tolist(),
                "hf_base_completion_logprobs": item[
                    "hf_base_completion_logprobs"
                ].tolist(),
            }
        )
    return serialized


def _serialize_hf_materialized_results(
    hf_materialized_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for item in hf_materialized_results:
        serialized.append(
            {
                "prompt": item["prompt"],
                "hf_materialized_prefill_logprobs": item[
                    "hf_materialized_prefill_logprobs"
                ].tolist(),
                "hf_materialized_completion_logprobs": item[
                    "hf_materialized_completion_logprobs"
                ].tolist(),
            }
        )
    return serialized


def _serialize_sglang_results(
    sglang_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for item in sglang_results:
        serialized.append(
            {
                "prompt": item["prompt"],
                "prompt_len": item["prompt_len"],
                "hf_completion_ids": item["hf_completion_ids"],
                "hf_completion_text": item["hf_completion_text"],
                "prefill_none_positions": item["prefill_none_positions"],
                "free_run_completion_ids": item["free_run_completion_ids"],
                "free_run_completion_text": item["free_run_completion_text"],
                "teacher_forced_prefill_logprobs": list(
                    item["teacher_forced_prefill_logprobs"]
                ),
                "teacher_forced_completion_logprobs": list(
                    item["teacher_forced_completion_logprobs"]
                ),
                "free_run_completion_logprobs": list(
                    item["free_run_completion_logprobs"]
                ),
            }
        )
    return serialized


def _decode_token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _serialize_sglang_top_logprobs(
    tokenizer,
    top_logprobs: list[list[Any]] | None,
) -> list[list[dict[str, Any]]] | None:
    if top_logprobs is None:
        return None
    serialized: list[list[dict[str, Any]]] = []
    for row in top_logprobs:
        serialized.append(
            [
                {
                    "logprob": float(entry[0]),
                    "token_id": int(entry[1]),
                    "token_text": _decode_token_text(tokenizer, int(entry[1])),
                }
                for entry in row
            ]
        )
    return serialized


def _extract_trace_payload(
    items: list[dict[str, Any]], key: str
) -> dict[str, Any] | None:
    for item in items:
        payload = item.get(key)
        if payload is not None:
            return payload
    return None


def _filtered_adapter_metadata(
    adapter_tensors: list[tuple[str, Any]],
    subset_spec: str,
) -> dict[str, Any]:
    def _base_name(name: str) -> str:
        for suffix in (".lora_A.default.weight", ".lora_B.default.weight"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    base_names = sorted({_base_name(name) for name, _ in adapter_tensors})
    return {
        "adapter_subset": subset_spec,
        "filtered_adapter_tensor_count": len(adapter_tensors),
        "filtered_adapter_base_count": len(base_names),
        "filtered_adapter_base_examples": base_names[:8],
    }


def _extract_metric(output: str, name: str) -> float | None:
    match = re.search(rf"{re.escape(name)}\s*=\s*([0-9.eE+-]+)", output)
    if match is None:
        return None
    return float(match.group(1))


def _parse_bool_env(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _parse_prompt_indexes_env(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    indexes: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        indexes.append(int(token))
    return indexes or None


@app.function(
    image=image,
    gpu=HF_GPU,
    timeout=4 * 60 * 60,
    retries=0,
    secrets=[RUNTIME_CONFIG_SECRET],
    volumes={HF_CACHE_PATH: hf_cache_vol},
)
def run_hf_reference() -> dict[str, Any]:
    import torch

    os.environ.update(HF_IMAGE_ENV)
    os.environ["PYTHONPATH"] = _remote_pythonpath()

    module = _load_test_module()
    runtime_adapter_dir, volume_weights_path = _prepare_runtime_adapter_dir()
    adapter_subset = os.getenv("QWEN35_ADAPTER_SUBSET", ADAPTER_SUBSET)
    with open(runtime_adapter_dir / "adapter_config.json", "r") as f:
        adapter_config = json.load(f)
    filtered_adapter_tensors = module._filter_adapter_tensors(
        list(module.load_file(str(volume_weights_path)).items()),
        adapter_subset,
    )
    filtered_adapter_dir = runtime_adapter_dir
    if module._normalize_adapter_subset_tokens(adapter_subset) != ["all"]:
        filtered_adapter_dir = REMOTE_TMP_DIR / f"adapter_dir_{adapter_subset.replace(',', '_')}"
        module._write_adapter_dir(
            adapter_config,
            filtered_adapter_tensors,
            filtered_adapter_dir,
        )

    stream = io.StringIO()
    selected_prompt_indices = _parse_prompt_indexes_env(
        os.getenv("QWEN35_PROMPT_INDEXES", PROMPT_INDEXES)
    )
    prompts = module.PROMPTS
    if selected_prompt_indices is not None:
        prompts = [module.PROMPTS[index] for index in selected_prompt_indices]
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        (
            hf_results,
            hf_materialized_results,
            hf_runtime_backend,
            hf_materialized_backend,
        ) = module._hf_generate_and_score_with_materialized_comparison(
            model_path=os.getenv("QWEN35_BASE_MODEL", BASE_MODEL),
            adapter_dir=filtered_adapter_dir,
            prompts=prompts,
            max_new_tokens=int(
                os.getenv("QWEN35_MERGE_MAX_NEW_TOKENS", str(MAX_NEW_TOKENS))
            ),
            torch_dtype=getattr(torch, os.getenv("QWEN35_MERGE_DTYPE", MERGE_DTYPE)),
        )
        hf_base_results, hf_base_backend = module._hf_score_sequences_base(
            model_path=os.getenv("QWEN35_BASE_MODEL", BASE_MODEL),
            prompt_results=hf_results,
            torch_dtype=getattr(torch, os.getenv("QWEN35_MERGE_DTYPE", MERGE_DTYPE)),
        )

    hf_fast_path_fallback_detected = any(
        report.get("fast_path_fallback_detected", False)
        for report in (
            hf_runtime_backend,
            hf_base_backend,
            hf_materialized_backend,
        )
    )
    hf_trace_prompt = _extract_trace_payload(hf_results, "trace_prompt_data") or {}
    hf_trace_prompt.update(
        _extract_trace_payload(hf_base_results, "trace_prompt_base_data") or {}
    )
    hf_trace_prompt.update(
        _extract_trace_payload(
            hf_materialized_results, "trace_prompt_materialized_data"
        )
        or {}
    )
    if hf_trace_prompt and selected_prompt_indices is not None:
        local_prompt_index = hf_trace_prompt.get("prompt_index")
        if isinstance(local_prompt_index, int) and 0 <= local_prompt_index < len(
            selected_prompt_indices
        ):
            hf_trace_prompt["source_prompt_index"] = selected_prompt_indices[
                local_prompt_index
            ]

    return {
        "gpu": os.getenv("QWEN35_HF_MODAL_GPU", HF_GPU),
        "base_model": os.getenv("QWEN35_BASE_MODEL", BASE_MODEL),
        "requested_attn_implementation": (
            os.getenv("QWEN35_HF_ATTN_IMPLEMENTATION", HF_ATTN_IMPLEMENTATION).strip()
            or None
        ),
        "selected_prompt_indices": selected_prompt_indices,
        "prompt_count": len(hf_results),
        "runtime_adapter_dir": str(filtered_adapter_dir),
        "volume_adapter_weights": str(volume_weights_path),
        "hf_results": _serialize_hf_results(hf_results),
        "hf_base_results": _serialize_hf_base_results(hf_base_results),
        "hf_materialized_results": _serialize_hf_materialized_results(
            hf_materialized_results
        ),
        "hf_runtime_backend": hf_runtime_backend,
        "hf_base_backend": hf_base_backend,
        "hf_materialized_backend": hf_materialized_backend,
        "hf_fast_path_fallback_detected": hf_fast_path_fallback_detected,
        "hf_trace_prompt": hf_trace_prompt or None,
        "output_tail": _tail(stream.getvalue()),
        **_filtered_adapter_metadata(filtered_adapter_tensors, adapter_subset),
    }


@app.function(
    image=image,
    gpu=SGLANG_GPU,
    memory=SGLANG_MEMORY_MB,
    timeout=4 * 60 * 60,
    retries=0,
    secrets=[RUNTIME_CONFIG_SECRET],
    volumes={HF_CACHE_PATH: hf_cache_vol},
)
def run_sglang_reference(
    hf_results: list[dict[str, Any]],
) -> dict[str, Any]:
    import torch

    os.environ.update(HF_IMAGE_ENV)
    os.environ["PYTHONPATH"] = _remote_pythonpath()

    module = _load_test_module()
    _, volume_weights_path = _prepare_runtime_adapter_dir()
    volume_config_path = _remote_volume_path(ADAPTER_VOLUME_CONFIG_REL)
    tokenizer = module.AutoTokenizer.from_pretrained(
        os.getenv("QWEN35_BASE_MODEL", BASE_MODEL)
    )

    with open(volume_config_path, "r") as f:
        adapter_config = json.load(f)
    adapter_subset = os.getenv("QWEN35_ADAPTER_SUBSET", ADAPTER_SUBSET)
    adapter_tensors = module._filter_adapter_tensors(
        list(module.load_file(str(volume_weights_path)).items()),
        adapter_subset,
    )

    force_attention_backend = os.getenv(
        "QWEN35_SGLANG_ATTENTION_BACKEND", SGLANG_ATTENTION_BACKEND
    ).strip()
    disable_cuda_graph = os.getenv(
        "QWEN35_SGLANG_DISABLE_CUDA_GRAPH", SGLANG_DISABLE_CUDA_GRAPH
    ).strip()
    mem_fraction_static = os.getenv(
        "QWEN35_SGLANG_MEM_FRACTION_STATIC", SGLANG_MEM_FRACTION_STATIC
    ).strip()
    moe_runner_backend = os.getenv(
        "QWEN35_SGLANG_MOE_RUNNER_BACKEND", SGLANG_MOE_RUNNER_BACKEND
    ).strip()
    enable_deterministic_inference = _parse_bool_env(
        os.getenv(
            "QWEN35_SGLANG_ENABLE_DETERMINISTIC_INFERENCE",
            SGLANG_ENABLE_DETERMINISTIC_INFERENCE,
        ),
        default=True,
    )
    if (
        force_attention_backend
        or disable_cuda_graph
        or mem_fraction_static
        or moe_runner_backend
    ):
        original_engine = module.sgl.Engine

        def _engine_with_overrides(*args, **kwargs):
            if force_attention_backend:
                kwargs.setdefault("attention_backend", force_attention_backend)
            if disable_cuda_graph:
                kwargs.setdefault(
                    "disable_cuda_graph",
                    disable_cuda_graph.lower() in ("1", "true", "yes", "on"),
                )
            if mem_fraction_static:
                kwargs.setdefault("mem_fraction_static", float(mem_fraction_static))
            if moe_runner_backend:
                kwargs.setdefault("moe_runner_backend", moe_runner_backend)
            return original_engine(*args, **kwargs)

        module.sgl.Engine = _engine_with_overrides

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        engine = module.sgl.Engine(
            model_path=os.getenv("QWEN35_BASE_MODEL", BASE_MODEL),
            dtype=os.getenv("QWEN35_MERGE_DTYPE", MERGE_DTYPE),
            custom_weight_loader=[module.MERGE_LOADER],
            disable_radix_cache=True,
            enable_deterministic_inference=enable_deterministic_inference,
            log_level="error",
        )
        try:
            sglang_base_results = []
            sglang_trace_prompt: dict[str, Any] | None = None
            trace_top_k = int(os.getenv("QWEN35_TRACE_TOP_K", str(TRACE_TOP_K)))
            trace_prompt_index = int(
                os.getenv("QWEN35_TRACE_PROMPT_INDEX", str(TRACE_PROMPT_INDEX))
            )
            for prompt_index, item in enumerate(hf_results):
                trace_this_prompt = (
                    trace_top_k > 0 and prompt_index == trace_prompt_index
                )
                score_out = module._unwrap_single_response(
                    engine.generate(
                        input_ids=item["full_ids"],
                        sampling_params={"max_new_tokens": 0, "temperature": 0.0},
                        return_logprob=True,
                        logprob_start_len=0,
                        top_logprobs_num=trace_top_k if trace_this_prompt else 0,
                    )
                )
                meta = score_out["meta_info"]
                input_token_logprobs = meta["input_token_logprobs"]
                teacher_forced_base_prefill_logprobs = [
                    None if logprob is None else float(logprob)
                    for logprob, _, _ in input_token_logprobs[1 : item["prompt_len"]]
                ]
                teacher_forced_base_completion_logprobs = [
                    float(logprob)
                    for logprob, _, _ in input_token_logprobs[item["prompt_len"] :]
                ]
                sglang_base_results.append(
                    {
                        "prompt": item["prompt"],
                        "teacher_forced_base_prefill_logprobs": (
                            teacher_forced_base_prefill_logprobs
                        ),
                        "teacher_forced_base_completion_logprobs": (
                            teacher_forced_base_completion_logprobs
                        ),
                    }
                )
                if trace_this_prompt:
                    input_top_logprobs = meta.get("input_top_logprobs")
                    sglang_trace_prompt = {
                        "prompt_index": prompt_index,
                        "prompt": item["prompt"],
                        "prompt_len": item["prompt_len"],
                        "full_ids": item["full_ids"],
                        "full_token_texts": [
                            _decode_token_text(tokenizer, token_id)
                            for token_id in item["full_ids"]
                        ],
                        "completion_ids": item["completion_ids"],
                        "completion_token_texts": [
                            _decode_token_text(tokenizer, token_id)
                            for token_id in item["completion_ids"]
                        ],
                        "sglang_base_prefill_top_logprobs": _serialize_sglang_top_logprobs(
                            tokenizer,
                            input_top_logprobs[1 : item["prompt_len"]]
                            if input_top_logprobs is not None
                            else None,
                        ),
                        "sglang_base_completion_top_logprobs": _serialize_sglang_top_logprobs(
                            tokenizer,
                            input_top_logprobs[item["prompt_len"] :]
                            if input_top_logprobs is not None
                            else None,
                        ),
                    }

            success, message = engine.update_weights_from_tensor(
                named_tensors=adapter_tensors,
                manifest={"adapter_config": adapter_config},
                load_format=module.MERGE_LOADER,
            )
            if not success:
                raise RuntimeError(f"Merged LoRA update failed: {message}")

            sglang_results: list[dict[str, Any]] = []
            max_new_tokens = int(
                os.getenv("QWEN35_MERGE_MAX_NEW_TOKENS", str(MAX_NEW_TOKENS))
            )
            for prompt_index, item in enumerate(hf_results):
                trace_this_prompt = (
                    trace_top_k > 0 and prompt_index == trace_prompt_index
                )
                score_out = module._unwrap_single_response(
                    engine.generate(
                        input_ids=item["full_ids"],
                        sampling_params={"max_new_tokens": 0, "temperature": 0.0},
                        return_logprob=True,
                        logprob_start_len=0,
                        top_logprobs_num=trace_top_k if trace_this_prompt else 0,
                    )
                )
                meta = score_out["meta_info"]
                input_token_logprobs = meta["input_token_logprobs"]
                prefill_input_token_logprobs = input_token_logprobs[1 : item["prompt_len"]]
                teacher_forced_prefill_logprobs = [
                    None if logprob is None else float(logprob)
                    for logprob, _, _ in prefill_input_token_logprobs
                ]
                prefill_none_positions = [
                    relative_idx + 1
                    for relative_idx, logprob in enumerate(
                        teacher_forced_prefill_logprobs
                    )
                    if logprob is None
                ]
                completion_input_token_logprobs = input_token_logprobs[item["prompt_len"] :]
                teacher_forced_logprobs: list[float] = []
                for relative_idx, (logprob, _, _) in enumerate(
                    completion_input_token_logprobs
                ):
                    if logprob is None:
                        raise RuntimeError(
                            "SGLang returned None input_token_logprob for completion "
                            f"token {relative_idx} of prompt: {item['prompt']}"
                        )
                    teacher_forced_logprobs.append(float(logprob))

                gen_out = module._unwrap_single_response(
                    engine.generate(
                        prompt=item["prompt"],
                        sampling_params={
                            "max_new_tokens": max_new_tokens,
                            "temperature": 0.0,
                        },
                        return_logprob=True,
                        logprob_start_len=0,
                        top_logprobs_num=trace_top_k if trace_this_prompt else 0,
                    )
                )
                meta = gen_out["meta_info"]
                output_token_logprobs = meta["output_token_logprobs"]
                sglang_results.append(
                    {
                        "prompt": item["prompt"],
                        "prompt_len": item["prompt_len"],
                        "hf_completion_ids": item["completion_ids"],
                        "hf_completion_text": item["completion_text"],
                        "teacher_forced_prefill_logprobs": (
                            teacher_forced_prefill_logprobs
                        ),
                        "prefill_none_positions": prefill_none_positions,
                        "teacher_forced_completion_logprobs": teacher_forced_logprobs,
                        "free_run_completion_ids": [
                            int(x[1]) for x in output_token_logprobs
                        ],
                        "free_run_completion_logprobs": [
                            float(x[0]) for x in output_token_logprobs
                        ],
                        "free_run_completion_text": gen_out["text"],
                    }
                )
                if trace_this_prompt and sglang_trace_prompt is not None:
                    input_top_logprobs = score_out["meta_info"].get("input_top_logprobs")
                    sglang_trace_prompt.update(
                        {
                            "sglang_prefill_top_logprobs": _serialize_sglang_top_logprobs(
                                tokenizer,
                                input_top_logprobs[1 : item["prompt_len"]]
                                if input_top_logprobs is not None
                                else None,
                            ),
                            "sglang_completion_top_logprobs": _serialize_sglang_top_logprobs(
                                tokenizer,
                                input_top_logprobs[item["prompt_len"] :]
                                if input_top_logprobs is not None
                                else None,
                            ),
                            "sglang_free_run_top_logprobs": _serialize_sglang_top_logprobs(
                                tokenizer,
                                meta.get("output_top_logprobs"),
                            ),
                            "sglang_free_run_completion_token_texts": [
                                _decode_token_text(tokenizer, token_id)
                                for token_id, _ in zip(
                                    [int(x[1]) for x in output_token_logprobs],
                                    output_token_logprobs,
                                )
                            ],
                        }
                    )
        finally:
            engine.shutdown()
            module._cleanup_torch()

    selected_prompt_indices = _parse_prompt_indexes_env(
        os.getenv("QWEN35_PROMPT_INDEXES", PROMPT_INDEXES)
    )
    if sglang_trace_prompt and selected_prompt_indices is not None:
        local_prompt_index = sglang_trace_prompt.get("prompt_index")
        if isinstance(local_prompt_index, int) and 0 <= local_prompt_index < len(
            selected_prompt_indices
        ):
            sglang_trace_prompt["source_prompt_index"] = selected_prompt_indices[
                local_prompt_index
            ]

    return {
        "gpu": os.getenv("QWEN35_SGLANG_MODAL_GPU", SGLANG_GPU),
        "host_memory_mb": SGLANG_MEMORY_MB,
        "base_model": os.getenv("QWEN35_BASE_MODEL", BASE_MODEL),
        "forced_attention_backend": force_attention_backend or None,
        "disable_cuda_graph": (
            disable_cuda_graph.lower() in ("1", "true", "yes", "on")
            if disable_cuda_graph
            else None
        ),
        "enable_deterministic_inference": enable_deterministic_inference,
        "mem_fraction_static": (
            float(mem_fraction_static) if mem_fraction_static else None
        ),
        "moe_runner_backend": moe_runner_backend or None,
        "volume_adapter_config": str(volume_config_path),
        "volume_adapter_weights": str(volume_weights_path),
        "selected_prompt_indices": selected_prompt_indices,
        "prompt_count": len(sglang_results),
        "sglang_base_results": sglang_base_results,
        "sglang_results": _serialize_sglang_results(sglang_results),
        "sglang_trace_prompt": sglang_trace_prompt,
        "output_tail": _tail(stream.getvalue()),
        **_filtered_adapter_metadata(adapter_tensors, adapter_subset),
    }


def _compare_references(
    hf_reference: dict[str, Any], sglang_reference: dict[str, Any]
) -> dict[str, Any]:
    hf_results = hf_reference["hf_results"]
    hf_base_results = hf_reference["hf_base_results"]
    sglang_base_results = sglang_reference["sglang_base_results"]
    sglang_results = sglang_reference["sglang_results"]
    if len(hf_results) != len(sglang_results):
        raise AssertionError(
            f"Prompt count mismatch: hf={len(hf_results)} sglang={len(sglang_results)}"
        )
    if len(hf_base_results) != len(hf_results) or len(sglang_base_results) != len(hf_results):
        raise AssertionError("Base reference prompt count mismatch.")

    prompt_metrics: list[dict[str, Any]] = []
    completion_mismatches: list[dict[str, Any]] = []
    base_prefill_max_abs_values: list[float] = []
    base_prefill_mean_abs_values: list[float] = []
    base_max_abs_values: list[float] = []
    base_mean_abs_values: list[float] = []
    delta_prefill_max_abs_values: list[float] = []
    delta_prefill_mean_abs_values: list[float] = []
    delta_max_abs_values: list[float] = []
    delta_mean_abs_values: list[float] = []
    prefill_max_abs_values: list[float] = []
    prefill_mean_abs_values: list[float] = []
    max_abs_values: list[float] = []
    mean_abs_values: list[float] = []
    for hf_base_item, hf_item, sglang_base_item, sglang_item in zip(
        hf_base_results, hf_results, sglang_base_results, sglang_results
    ):
        if hf_item["prompt"] != sglang_item["prompt"]:
            raise AssertionError(
                "Prompt ordering mismatch between HF and SGLang references."
            )
        if hf_base_item["prompt"] != hf_item["prompt"] or sglang_base_item["prompt"] != hf_item["prompt"]:
            raise AssertionError("Base/merged prompt alignment mismatch.")

        hf_completion_ids = hf_item["completion_ids"]
        free_run_completion_ids = sglang_item["free_run_completion_ids"]
        prefix_match_len = 0
        while (
            prefix_match_len < len(hf_completion_ids)
            and prefix_match_len < len(free_run_completion_ids)
            and hf_completion_ids[prefix_match_len]
            == free_run_completion_ids[prefix_match_len]
        ):
            prefix_match_len += 1
        first_divergence_index = (
            None
            if len(hf_completion_ids) == len(free_run_completion_ids)
            and prefix_match_len == len(hf_completion_ids)
            else prefix_match_len
        )

        if hf_completion_ids != free_run_completion_ids:
            hf_token_at_divergence = (
                hf_completion_ids[first_divergence_index]
                if first_divergence_index is not None
                and first_divergence_index < len(hf_completion_ids)
                else None
            )
            sglang_token_at_divergence = (
                free_run_completion_ids[first_divergence_index]
                if first_divergence_index is not None
                and first_divergence_index < len(free_run_completion_ids)
                else None
            )
            completion_mismatches.append(
                {
                    "prompt": hf_item["prompt"],
                    "prefix_match_len": prefix_match_len,
                    "first_divergence_index": first_divergence_index,
                    "hf_token_at_divergence": hf_token_at_divergence,
                    "sglang_token_at_divergence": sglang_token_at_divergence,
                    "hf_completion_ids": hf_completion_ids,
                    "sglang_completion_ids": free_run_completion_ids,
                    "hf_completion_text": hf_item["completion_text"],
                    "sglang_completion_text": sglang_item["free_run_completion_text"],
                }
            )

        hf_prefill_logprobs = hf_item["hf_prefill_logprobs"]
        sglang_prefill_logprobs = sglang_item["teacher_forced_prefill_logprobs"]
        if len(hf_prefill_logprobs) != len(sglang_prefill_logprobs):
            raise AssertionError(
                f"Prefill length mismatch for prompt: {hf_item['prompt']}"
            )
        prefill_none_positions = sglang_item["prefill_none_positions"]
        comparable_prefill_diffs = [
            abs(float(hf) - float(sglang))
            for hf, sglang in zip(hf_prefill_logprobs, sglang_prefill_logprobs)
            if sglang is not None
        ]
        prefill_max_abs = (
            max(comparable_prefill_diffs) if comparable_prefill_diffs else None
        )
        prefill_mean_abs = (
            sum(comparable_prefill_diffs) / len(comparable_prefill_diffs)
            if comparable_prefill_diffs
            else None
        )
        if prefill_max_abs is not None:
            prefill_max_abs_values.append(prefill_max_abs)
        if prefill_mean_abs is not None:
            prefill_mean_abs_values.append(prefill_mean_abs)

        hf_base_prefill_logprobs = hf_base_item["hf_base_prefill_logprobs"]
        sglang_base_prefill_logprobs = sglang_base_item[
            "teacher_forced_base_prefill_logprobs"
        ]
        if len(hf_base_prefill_logprobs) != len(sglang_base_prefill_logprobs):
            raise AssertionError(
                f"Base prefill length mismatch for prompt: {hf_item['prompt']}"
            )
        comparable_base_prefill_diffs = [
            abs(float(hf) - float(sglang))
            for hf, sglang in zip(
                hf_base_prefill_logprobs, sglang_base_prefill_logprobs
            )
            if sglang is not None
        ]
        base_prefill_max_abs = (
            max(comparable_base_prefill_diffs)
            if comparable_base_prefill_diffs
            else None
        )
        base_prefill_mean_abs = (
            sum(comparable_base_prefill_diffs) / len(comparable_base_prefill_diffs)
            if comparable_base_prefill_diffs
            else None
        )
        if base_prefill_max_abs is not None:
            base_prefill_max_abs_values.append(base_prefill_max_abs)
        if base_prefill_mean_abs is not None:
            base_prefill_mean_abs_values.append(base_prefill_mean_abs)

        prefill_first_large_diff_position = None
        prefill_token_id_at_first_large_diff = None
        for relative_idx, (hf, sglang) in enumerate(
            zip(hf_prefill_logprobs, sglang_prefill_logprobs)
        ):
            if sglang is None:
                continue
            if abs(float(hf) - float(sglang)) > MERGE_MAX_ABS_THRESHOLD:
                prefill_first_large_diff_position = relative_idx + 1
                prefill_token_id_at_first_large_diff = hf_item["full_ids"][
                    prefill_first_large_diff_position
                ]
                break

        hf_logprobs = hf_item["hf_completion_logprobs"]
        sglang_logprobs = sglang_item["teacher_forced_completion_logprobs"]
        if len(hf_logprobs) != len(sglang_logprobs):
            raise AssertionError(
                f"Completion length mismatch for prompt: {hf_item['prompt']}"
            )

        diff = [
            abs(float(hf) - float(sglang))
            for hf, sglang in zip(hf_logprobs, sglang_logprobs)
        ]
        max_abs = max(diff) if diff else 0.0
        mean_abs = (sum(diff) / len(diff)) if diff else 0.0
        max_abs_values.append(max_abs)
        mean_abs_values.append(mean_abs)

        hf_base_logprobs = hf_base_item["hf_base_completion_logprobs"]
        sglang_base_logprobs = sglang_base_item[
            "teacher_forced_base_completion_logprobs"
        ]
        if len(hf_base_logprobs) != len(sglang_base_logprobs):
            raise AssertionError(
                f"Base completion length mismatch for prompt: {hf_item['prompt']}"
            )
        base_diff = [
            abs(float(hf) - float(sglang))
            for hf, sglang in zip(hf_base_logprobs, sglang_base_logprobs)
        ]
        base_max_abs = max(base_diff) if base_diff else 0.0
        base_mean_abs = (sum(base_diff) / len(base_diff)) if base_diff else 0.0
        base_max_abs_values.append(base_max_abs)
        base_mean_abs_values.append(base_mean_abs)

        hf_prefill_delta = [
            float(merged) - float(base)
            for base, merged in zip(hf_base_prefill_logprobs, hf_prefill_logprobs)
        ]
        sglang_prefill_delta = [
            None if merged is None or base is None else float(merged) - float(base)
            for base, merged in zip(
                sglang_base_prefill_logprobs, sglang_prefill_logprobs
            )
        ]
        comparable_prefill_delta_diffs = [
            abs(hf_delta - sglang_delta)
            for hf_delta, sglang_delta in zip(hf_prefill_delta, sglang_prefill_delta)
            if sglang_delta is not None
        ]
        delta_prefill_max_abs = (
            max(comparable_prefill_delta_diffs)
            if comparable_prefill_delta_diffs
            else None
        )
        delta_prefill_mean_abs = (
            sum(comparable_prefill_delta_diffs) / len(comparable_prefill_delta_diffs)
            if comparable_prefill_delta_diffs
            else None
        )
        if delta_prefill_max_abs is not None:
            delta_prefill_max_abs_values.append(delta_prefill_max_abs)
        if delta_prefill_mean_abs is not None:
            delta_prefill_mean_abs_values.append(delta_prefill_mean_abs)

        hf_completion_delta = [
            float(merged) - float(base)
            for base, merged in zip(hf_base_logprobs, hf_logprobs)
        ]
        sglang_completion_delta = [
            float(merged) - float(base)
            for base, merged in zip(sglang_base_logprobs, sglang_logprobs)
        ]
        delta_diff = [
            abs(hf_delta - sglang_delta)
            for hf_delta, sglang_delta in zip(
                hf_completion_delta, sglang_completion_delta
            )
        ]
        delta_max_abs = max(delta_diff) if delta_diff else 0.0
        delta_mean_abs = (sum(delta_diff) / len(delta_diff)) if delta_diff else 0.0
        delta_max_abs_values.append(delta_max_abs)
        delta_mean_abs_values.append(delta_mean_abs)
        prompt_metrics.append(
            {
                "prompt": hf_item["prompt"],
                "hf_completion_text": hf_item["completion_text"],
                "sglang_completion_text": sglang_item["free_run_completion_text"],
                "completion_tokens": len(hf_item["completion_ids"]),
                "prefix_match_len": prefix_match_len,
                "first_divergence_index": first_divergence_index,
                "prefill_tokens": len(hf_prefill_logprobs),
                "prefill_comparable_tokens": len(comparable_prefill_diffs),
                "prefill_none_positions": prefill_none_positions,
                "prefill_first_large_diff_position": (
                    prefill_first_large_diff_position
                ),
                "prefill_token_id_at_first_large_diff": (
                    prefill_token_id_at_first_large_diff
                ),
                "base_prefill_max_abs": base_prefill_max_abs,
                "base_prefill_mean_abs": base_prefill_mean_abs,
                "prefill_max_abs": prefill_max_abs,
                "prefill_mean_abs": prefill_mean_abs,
                "base_max_abs": base_max_abs,
                "base_mean_abs": base_mean_abs,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "delta_prefill_max_abs": delta_prefill_max_abs,
                "delta_prefill_mean_abs": delta_prefill_mean_abs,
                "delta_max_abs": delta_max_abs,
                "delta_mean_abs": delta_mean_abs,
            }
        )

    overall_base_prefill_max_abs = (
        max(base_prefill_max_abs_values) if base_prefill_max_abs_values else None
    )
    overall_base_prefill_mean_abs = (
        sum(base_prefill_mean_abs_values) / len(base_prefill_mean_abs_values)
        if base_prefill_mean_abs_values
        else None
    )
    overall_prefill_max_abs = (
        max(prefill_max_abs_values) if prefill_max_abs_values else None
    )
    overall_prefill_mean_abs = (
        sum(prefill_mean_abs_values) / len(prefill_mean_abs_values)
        if prefill_mean_abs_values
        else None
    )
    overall_base_max_abs = max(base_max_abs_values) if base_max_abs_values else None
    overall_base_mean_abs = (
        sum(base_mean_abs_values) / len(base_mean_abs_values)
        if base_mean_abs_values
        else None
    )
    overall_max_abs = max(max_abs_values) if max_abs_values else None
    overall_mean_abs = (
        sum(mean_abs_values) / len(mean_abs_values) if mean_abs_values else None
    )
    overall_delta_prefill_max_abs = (
        max(delta_prefill_max_abs_values) if delta_prefill_max_abs_values else None
    )
    overall_delta_prefill_mean_abs = (
        sum(delta_prefill_mean_abs_values) / len(delta_prefill_mean_abs_values)
        if delta_prefill_mean_abs_values
        else None
    )
    overall_delta_max_abs = max(delta_max_abs_values) if delta_max_abs_values else None
    overall_delta_mean_abs = (
        sum(delta_mean_abs_values) / len(delta_mean_abs_values)
        if delta_mean_abs_values
        else None
    )
    passed = (
        not completion_mismatches
        and overall_max_abs is not None
        and overall_mean_abs is not None
        and overall_max_abs <= MERGE_MAX_ABS_THRESHOLD
        and overall_mean_abs <= MERGE_MEAN_ABS_THRESHOLD
    )

    return {
        "base_prefill_logprob_comparison_basis": "teacher_forced_base_on_prompt_token_ids[1:prompt_len]",
        "prefill_logprob_comparison_basis": "teacher_forced_on_prompt_token_ids[1:prompt_len]",
        "base_logprob_comparison_basis": "teacher_forced_base_on_hf_completion_ids",
        "logprob_comparison_basis": "teacher_forced_on_hf_completion_ids",
        "delta_logprob_comparison_basis": "adapter_delta=(merged-base) on identical token ids",
        "overall_base_prefill_max_abs": overall_base_prefill_max_abs,
        "overall_base_prefill_mean_abs": overall_base_prefill_mean_abs,
        "overall_prefill_max_abs": overall_prefill_max_abs,
        "overall_prefill_mean_abs": overall_prefill_mean_abs,
        "overall_base_max_abs": overall_base_max_abs,
        "overall_base_mean_abs": overall_base_mean_abs,
        "overall_delta_prefill_max_abs": overall_delta_prefill_max_abs,
        "overall_delta_prefill_mean_abs": overall_delta_prefill_mean_abs,
        "overall_delta_max_abs": overall_delta_max_abs,
        "overall_delta_mean_abs": overall_delta_mean_abs,
        "prompt_metrics": prompt_metrics,
        "completion_mismatches": completion_mismatches,
        "overall_max_abs": overall_max_abs,
        "overall_mean_abs": overall_mean_abs,
        "max_abs_threshold": MERGE_MAX_ABS_THRESHOLD,
        "mean_abs_threshold": MERGE_MEAN_ABS_THRESHOLD,
        "successful": passed,
    }


def _compare_hf_self_consistency(hf_reference: dict[str, Any]) -> dict[str, Any]:
    hf_results = hf_reference["hf_results"]
    hf_materialized_results = hf_reference["hf_materialized_results"]
    if len(hf_results) != len(hf_materialized_results):
        raise AssertionError(
            "HF prompt count mismatch between runtime-PEFT and materialized merged refs."
        )

    prompt_metrics: list[dict[str, Any]] = []
    prefill_max_abs_values: list[float] = []
    prefill_mean_abs_values: list[float] = []
    max_abs_values: list[float] = []
    mean_abs_values: list[float] = []
    for hf_item, materialized_item in zip(hf_results, hf_materialized_results):
        if hf_item["prompt"] != materialized_item["prompt"]:
            raise AssertionError(
                "HF prompt ordering mismatch between runtime-PEFT and materialized refs."
            )

        hf_prefill_logprobs = hf_item["hf_prefill_logprobs"]
        materialized_prefill_logprobs = materialized_item[
            "hf_materialized_prefill_logprobs"
        ]
        if len(hf_prefill_logprobs) != len(materialized_prefill_logprobs):
            raise AssertionError(
                f"HF self-check prefill length mismatch for prompt: {hf_item['prompt']}"
            )
        prefill_diff = [
            abs(float(runtime_lp) - float(materialized_lp))
            for runtime_lp, materialized_lp in zip(
                hf_prefill_logprobs, materialized_prefill_logprobs
            )
        ]
        prefill_max_abs = max(prefill_diff) if prefill_diff else 0.0
        prefill_mean_abs = (sum(prefill_diff) / len(prefill_diff)) if prefill_diff else 0.0
        prefill_max_abs_values.append(prefill_max_abs)
        prefill_mean_abs_values.append(prefill_mean_abs)

        hf_completion_logprobs = hf_item["hf_completion_logprobs"]
        materialized_completion_logprobs = materialized_item[
            "hf_materialized_completion_logprobs"
        ]
        if len(hf_completion_logprobs) != len(materialized_completion_logprobs):
            raise AssertionError(
                f"HF self-check completion length mismatch for prompt: {hf_item['prompt']}"
            )
        diff = [
            abs(float(runtime_lp) - float(materialized_lp))
            for runtime_lp, materialized_lp in zip(
                hf_completion_logprobs, materialized_completion_logprobs
            )
        ]
        max_abs = max(diff) if diff else 0.0
        mean_abs = (sum(diff) / len(diff)) if diff else 0.0
        max_abs_values.append(max_abs)
        mean_abs_values.append(mean_abs)
        prompt_metrics.append(
            {
                "prompt": hf_item["prompt"],
                "prefill_tokens": len(hf_prefill_logprobs),
                "completion_tokens": len(hf_completion_logprobs),
                "prefill_max_abs": prefill_max_abs,
                "prefill_mean_abs": prefill_mean_abs,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
            }
        )

    return {
        "comparison_basis": "hf_peft_runtime_vs_hf_materialized_merged on identical token ids",
        "overall_prefill_max_abs": max(prefill_max_abs_values) if prefill_max_abs_values else None,
        "overall_prefill_mean_abs": (
            sum(prefill_mean_abs_values) / len(prefill_mean_abs_values)
            if prefill_mean_abs_values
            else None
        ),
        "overall_max_abs": max(max_abs_values) if max_abs_values else None,
        "overall_mean_abs": (
            sum(mean_abs_values) / len(mean_abs_values) if mean_abs_values else None
        ),
        "prompt_metrics": prompt_metrics,
    }


def _build_prompt_trace(
    hf_reference: dict[str, Any],
    sglang_reference: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any] | None:
    hf_trace = hf_reference.get("hf_trace_prompt")
    sglang_trace = sglang_reference.get("sglang_trace_prompt")
    if not hf_trace or not sglang_trace:
        return None

    prompt = hf_trace["prompt"]
    prompt_metrics = next(
        item for item in comparison["prompt_metrics"] if item["prompt"] == prompt
    )
    completion_mismatch = next(
        (
            item
            for item in comparison["completion_mismatches"]
            if item["prompt"] == prompt
        ),
        None,
    )

    prefill_rows: list[dict[str, Any]] = []
    for idx in range(hf_trace["prompt_len"] - 1):
        token_id = hf_trace["full_ids"][idx + 1]
        token_text = hf_trace["full_token_texts"][idx + 1]
        hf_base_lp = hf_reference["hf_base_results"][TRACE_PROMPT_INDEX][
            "hf_base_prefill_logprobs"
        ][idx]
        hf_merged_lp = hf_reference["hf_results"][TRACE_PROMPT_INDEX][
            "hf_prefill_logprobs"
        ][idx]
        hf_materialized_lp = hf_reference["hf_materialized_results"][TRACE_PROMPT_INDEX][
            "hf_materialized_prefill_logprobs"
        ][idx]
        sglang_base_lp = sglang_reference["sglang_base_results"][TRACE_PROMPT_INDEX][
            "teacher_forced_base_prefill_logprobs"
        ][idx]
        sglang_merged_lp = sglang_reference["sglang_results"][TRACE_PROMPT_INDEX][
            "teacher_forced_prefill_logprobs"
        ][idx]
        prefill_rows.append(
            {
                "position": idx + 1,
                "token_id": token_id,
                "token_text": token_text,
                "hf_base_logprob": hf_base_lp,
                "hf_merged_logprob": hf_merged_lp,
                "hf_materialized_logprob": hf_materialized_lp,
                "sglang_base_logprob": sglang_base_lp,
                "sglang_merged_logprob": sglang_merged_lp,
                "base_abs_diff": abs(hf_base_lp - sglang_base_lp),
                "merged_abs_diff": abs(hf_merged_lp - sglang_merged_lp),
                "delta_abs_diff": abs(
                    (hf_merged_lp - hf_base_lp) - (sglang_merged_lp - sglang_base_lp)
                ),
                "hf_runtime_vs_materialized_abs_diff": abs(
                    hf_merged_lp - hf_materialized_lp
                ),
            }
        )

    completion_rows: list[dict[str, Any]] = []
    for idx, token_id in enumerate(hf_trace["completion_ids"]):
        token_text = hf_trace["completion_token_texts"][idx]
        hf_base_lp = hf_reference["hf_base_results"][TRACE_PROMPT_INDEX][
            "hf_base_completion_logprobs"
        ][idx]
        hf_merged_lp = hf_reference["hf_results"][TRACE_PROMPT_INDEX][
            "hf_completion_logprobs"
        ][idx]
        hf_materialized_lp = hf_reference["hf_materialized_results"][TRACE_PROMPT_INDEX][
            "hf_materialized_completion_logprobs"
        ][idx]
        sglang_base_lp = sglang_reference["sglang_base_results"][TRACE_PROMPT_INDEX][
            "teacher_forced_base_completion_logprobs"
        ][idx]
        sglang_merged_lp = sglang_reference["sglang_results"][TRACE_PROMPT_INDEX][
            "teacher_forced_completion_logprobs"
        ][idx]
        completion_rows.append(
            {
                "position": idx,
                "token_id": token_id,
                "token_text": token_text,
                "hf_base_logprob": hf_base_lp,
                "hf_merged_logprob": hf_merged_lp,
                "hf_materialized_logprob": hf_materialized_lp,
                "sglang_base_logprob": sglang_base_lp,
                "sglang_merged_logprob": sglang_merged_lp,
                "base_abs_diff": abs(hf_base_lp - sglang_base_lp),
                "merged_abs_diff": abs(hf_merged_lp - sglang_merged_lp),
                "delta_abs_diff": abs(
                    (hf_merged_lp - hf_base_lp) - (sglang_merged_lp - sglang_base_lp)
                ),
                "hf_runtime_vs_materialized_abs_diff": abs(
                    hf_merged_lp - hf_materialized_lp
                ),
            }
        )

    def _top_positions(rows: list[dict[str, Any]], key: str, count: int = 5) -> list[int]:
        sorted_rows = sorted(rows, key=lambda row: row[key], reverse=True)
        return [int(row["position"]) for row in sorted_rows[:count]]

    interesting_prefill_positions = sorted(
        {
            1,
            *(pos for pos in _top_positions(prefill_rows, "delta_abs_diff") if pos >= 1),
            *(pos for pos in _top_positions(prefill_rows, "merged_abs_diff") if pos >= 1),
            *(
                [prompt_metrics["prefill_first_large_diff_position"]]
                if prompt_metrics["prefill_first_large_diff_position"] is not None
                else []
            ),
        }
    )
    interesting_completion_positions = sorted(
        {
            0,
            *(pos for pos in _top_positions(completion_rows, "delta_abs_diff")),
            *(pos for pos in _top_positions(completion_rows, "merged_abs_diff")),
            *(
                [completion_mismatch["first_divergence_index"]]
                if completion_mismatch is not None
                and completion_mismatch["first_divergence_index"] is not None
                else []
            ),
        }
    )

    def _snapshot_prefill(position: int) -> dict[str, Any]:
        idx = position - 1
        row = prefill_rows[idx]
        return {
            **row,
            "hf_base_topk": hf_trace["hf_base_prefill_top_logprobs"][idx],
            "hf_runtime_topk": hf_trace["hf_runtime_prefill_top_logprobs"][idx],
            "hf_materialized_topk": hf_trace["hf_materialized_prefill_top_logprobs"][idx],
            "sglang_base_topk": sglang_trace["sglang_base_prefill_top_logprobs"][idx],
            "sglang_merged_topk": sglang_trace["sglang_prefill_top_logprobs"][idx],
        }

    def _snapshot_completion(position: int) -> dict[str, Any]:
        row = completion_rows[position]
        return {
            **row,
            "hf_base_topk": hf_trace["hf_base_completion_top_logprobs"][position],
            "hf_runtime_topk": hf_trace["hf_runtime_completion_top_logprobs"][position],
            "hf_materialized_topk": hf_trace["hf_materialized_completion_top_logprobs"][
                position
            ],
            "sglang_base_topk": sglang_trace["sglang_base_completion_top_logprobs"][
                position
            ],
            "sglang_merged_topk": sglang_trace["sglang_completion_top_logprobs"][position],
        }

    free_run_steps: list[dict[str, Any]] = []
    sglang_completion_ids = sglang_reference["sglang_results"][TRACE_PROMPT_INDEX][
        "free_run_completion_ids"
    ]
    sglang_completion_logprobs = sglang_reference["sglang_results"][TRACE_PROMPT_INDEX][
        "free_run_completion_logprobs"
    ]
    for idx, hf_token_id in enumerate(hf_trace["completion_ids"]):
        if idx >= len(sglang_completion_ids):
            break
        free_run_steps.append(
            {
                "position": idx,
                "hf_token_id": hf_token_id,
                "hf_token_text": hf_trace["completion_token_texts"][idx],
                "hf_logprob": sglang_reference.get("unused") if False else hf_reference[
                    "hf_results"
                ][TRACE_PROMPT_INDEX]["hf_completion_logprobs"][idx],
                "sglang_token_id": sglang_completion_ids[idx],
                "sglang_token_text": sglang_trace[
                    "sglang_free_run_completion_token_texts"
                ][idx],
                "sglang_logprob": sglang_completion_logprobs[idx],
                "match": hf_token_id == sglang_completion_ids[idx],
            }
        )

    divergence_step = (
        completion_mismatch["first_divergence_index"]
        if completion_mismatch is not None
        and completion_mismatch["first_divergence_index"] is not None
        else 0
    )

    return {
        "prompt_index": TRACE_PROMPT_INDEX,
        "trace_top_k": TRACE_TOP_K,
        "prompt": prompt,
        "comparison_prompt_metrics": prompt_metrics,
        "completion_mismatch": completion_mismatch,
        "prefill_rows": prefill_rows,
        "completion_rows": completion_rows,
        "interesting_prefill_snapshots": [
            _snapshot_prefill(position)
            for position in interesting_prefill_positions
            if 1 <= position <= len(prefill_rows)
        ],
        "interesting_completion_snapshots": [
            _snapshot_completion(position)
            for position in interesting_completion_positions
            if 0 <= position < len(completion_rows)
        ],
        "free_run_steps": free_run_steps[:16],
        "free_run_divergence_topk": {
            "position": divergence_step,
            "hf_runtime_topk": hf_trace["hf_runtime_free_run_top_logprobs"][
                divergence_step
            ],
            "sglang_topk": sglang_trace["sglang_free_run_top_logprobs"][divergence_step],
        },
    }


@app.local_entrypoint()
def main() -> None:
    upload_result = _upload_adapter_assets_to_volume()
    enable_modal_output = os.getenv("QWEN35_MODAL_ENABLE_OUTPUT", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    modal_output_ctx = modal.enable_output() if enable_modal_output else contextlib.nullcontext()
    with modal_output_ctx:
        hf_reference = run_hf_reference.remote()
        sglang_reference = run_sglang_reference.remote(hf_reference["hf_results"])
    comparison = _compare_references(hf_reference, sglang_reference)
    hf_self_comparison = _compare_hf_self_consistency(hf_reference)
    prompt_trace = _build_prompt_trace(hf_reference, sglang_reference, comparison)
    print(
        json.dumps(
            {
                "upload": upload_result,
                "hf_reference": {
                    "gpu": hf_reference["gpu"],
                    "base_model": hf_reference["base_model"],
                    "selected_prompt_indices": hf_reference["selected_prompt_indices"],
                    "adapter_subset": hf_reference["adapter_subset"],
                    "filtered_adapter_tensor_count": hf_reference[
                        "filtered_adapter_tensor_count"
                    ],
                    "filtered_adapter_base_count": hf_reference[
                        "filtered_adapter_base_count"
                    ],
                    "prompt_count": hf_reference["prompt_count"],
                    "requested_attn_implementation": hf_reference[
                        "requested_attn_implementation"
                    ],
                    "hf_fast_path_fallback_detected": hf_reference[
                        "hf_fast_path_fallback_detected"
                    ],
                    "hf_runtime_backend": hf_reference["hf_runtime_backend"],
                    "hf_base_backend": hf_reference["hf_base_backend"],
                    "hf_materialized_backend": hf_reference[
                        "hf_materialized_backend"
                    ],
                    "output_tail": hf_reference["output_tail"],
                },
                "sglang_reference": {
                    "gpu": sglang_reference["gpu"],
                    "host_memory_mb": sglang_reference["host_memory_mb"],
                    "base_model": sglang_reference["base_model"],
                    "selected_prompt_indices": sglang_reference[
                        "selected_prompt_indices"
                    ],
                    "adapter_subset": sglang_reference["adapter_subset"],
                    "forced_attention_backend": sglang_reference[
                        "forced_attention_backend"
                    ],
                    "disable_cuda_graph": sglang_reference["disable_cuda_graph"],
                    "mem_fraction_static": sglang_reference["mem_fraction_static"],
                    "prompt_count": sglang_reference["prompt_count"],
                    "output_tail": sglang_reference["output_tail"],
                },
                "hf_self_comparison": hf_self_comparison,
                "prompt_trace": prompt_trace,
                "comparison": comparison,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def hf_only() -> None:
    upload_result = _upload_adapter_assets_to_volume()
    enable_modal_output = os.getenv("QWEN35_MODAL_ENABLE_OUTPUT", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    modal_output_ctx = modal.enable_output() if enable_modal_output else contextlib.nullcontext()
    with modal_output_ctx:
        hf_reference = run_hf_reference.remote()
    hf_self_comparison = _compare_hf_self_consistency(hf_reference)
    print(
        json.dumps(
            {
                "upload": upload_result,
                "hf_reference": hf_reference,
                "hf_self_comparison": hf_self_comparison,
            },
            indent=2,
            sort_keys=True,
        )
    )
