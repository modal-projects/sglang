"""Modal validation harness for Qwen3.5 live LoRA sync.

Usage:
  MODAL_ENVIRONMENT=jason-dev N_GPUS=1 \
    modal run scripts/modal/qwen35_live_sync_validation.py --mode real_text
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import modal


def _find_local_repo_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    for candidate in candidates:
        for root in (candidate, *candidate.parents):
            if (root / "python/sglang").is_dir() and (root / "scripts/modal").is_dir():
                return root
    raise FileNotFoundError("Could not locate the sglang-qwen-3.5-vl-lora-sync repo root.")


LOCAL_REPO_ROOT = _find_local_repo_root() if modal.is_local() else None
LOCAL_FLASH_FDE_SRC = Path("/home/jm/flash-fde/src/autoinference")

MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
REAL_ADAPTER_REPO = "Chenzhiz/Qwen3.5-35B-A3B-SWE-LoRA"
MERGE_FROM_TENSORS_LOAD_FORMAT = (
    "sglang.srt.weight_sync.lora_merge_loader.apply_lora_merge_from_tensors"
)
REFERENCE_LORA_NAME = "__reference_runtime_lora__"

APP_NAME = "qwen35-live-sync-validation"
DEFAULT_PORT = 8000
MINUTES = 60

GPU_TYPE = os.getenv("GPU_TYPE", "B200")
N_GPUS = int(os.getenv("N_GPUS", "1"))
GPU = f"{GPU_TYPE}:{N_GPUS}"
HF_CACHE_PATH = "/root/.cache/huggingface"
HF_CACHE_VOLUME_NAME = os.getenv("HF_CACHE_VOLUME_NAME", "huggingface-cache")
MM_ATTENTION_BACKEND = os.getenv("MM_ATTENTION_BACKEND", "triton_attn")
REFERENCE_LORA_BACKEND = os.getenv("REFERENCE_LORA_BACKEND", "triton")
REFERENCE_BASE_MAX_ABS_TOL = float(os.getenv("REFERENCE_BASE_MAX_ABS_TOL", "1e-7"))
# Runtime LoRA and merged-weight paths are mathematically aligned, but they do
# not necessarily produce bit-identical logits because they route the low-rank
# contribution through different numerical kernels. Keep this as a diagnostic
# unless the caller explicitly opts into strict parity.
REFERENCE_LORA_MAX_ABS_TOL = float(os.getenv("REFERENCE_LORA_MAX_ABS_TOL", "2e-1"))
ENFORCE_REFERENCE_LORA_PARITY = (
    os.getenv("ENFORCE_REFERENCE_LORA_PARITY", "0") == "1"
)
REFERENCE_UNLOAD_MAX_ABS_TOL = float(
    os.getenv("REFERENCE_UNLOAD_MAX_ABS_TOL", "1e-7")
)
REFERENCE_CONTEXT_LENGTH = int(os.getenv("REFERENCE_CONTEXT_LENGTH", "8192"))
REFERENCE_CHUNKED_PREFILL_SIZE = int(
    os.getenv("REFERENCE_CHUNKED_PREFILL_SIZE", "1024")
)
REFERENCE_MAX_PREFILL_TOKENS = int(
    os.getenv("REFERENCE_MAX_PREFILL_TOKENS", "1024")
)
REFERENCE_MEM_FRACTION_STATIC = os.getenv("REFERENCE_MEM_FRACTION_STATIC", "0.72")
REFERENCE_MAX_LORAS_PER_BATCH = os.getenv("REFERENCE_MAX_LORAS_PER_BATCH", "1")
REFERENCE_MAX_LOADED_LORAS = os.getenv("REFERENCE_MAX_LOADED_LORAS", "1")
# The strongest oracle we currently have inside the live server is to read back
# touched parameter slices before/after the update and compare the resulting
# stored bf16 weights against the expected merged values under the live merge
# contract: accumulate the LoRA delta in fp32 against an fp32 scratch view of
# the target weight, then cast back once to the stored dtype.
PARAMETER_ORACLE_TRUNCATE_SIZE = int(
    os.getenv("PARAMETER_ORACLE_TRUNCATE_SIZE", "4")
)
PARAMETER_ORACLE_MAX_TARGETS = int(os.getenv("PARAMETER_ORACLE_MAX_TARGETS", "0"))
PARAMETER_ORACLE_WEIGHT_DTYPE = os.getenv("PARAMETER_ORACLE_WEIGHT_DTYPE", "bf16")
PARAMETER_ORACLE_MAX_ABS_TOL = float(
    os.getenv("PARAMETER_ORACLE_MAX_ABS_TOL", "5e-4")
)

SGLANG_IMAGE_TAG = os.getenv(
    "SGLANG_IMAGE_TAG",
    "lmsysorg/sglang:nightly-dev-cu13-20260311-dc4380e3",
)
CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", "32768"))

ENV_VARS = {
    "HF_HUB_CACHE": HF_CACHE_PATH,
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
    "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
    "SGLANG_USE_CUTEDSL_GDN_DECODE": "1",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
}

COMMON_SERVER_ARGS = {
    "--served-model-name": MODEL_NAME,
    "--context-length": str(CONTEXT_LENGTH),
    "--disable-cuda-graph": "",
    "--skip-server-warmup": "",
    "--chunked-prefill-size": "4096",
    "--max-prefill-tokens": "4096",
    "--mem-fraction-static": "0.85",
    "--kv-cache-dtype": "bf16",
    "--page-size": "64",
    "--prefill-attention-backend": os.getenv("PREFILL_ATTENTION_BACKEND", "trtllm_mha"),
    "--decode-attention-backend": os.getenv("DECODE_ATTENTION_BACKEND", "trtllm_mha"),
    "--moe-runner-backend": os.getenv("MOE_RUNNER_BACKEND", "flashinfer_trtllm"),
    "--mamba-scheduler-strategy": "extra_buffer",
}

app = modal.App(name=APP_NAME)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME)
image = modal.Image.from_registry(SGLANG_IMAGE_TAG).env(ENV_VARS)
if modal.is_local():
    image = image.add_local_dir(
        LOCAL_REPO_ROOT / "python/sglang",
        remote_path="/sgl-workspace/sglang/python/sglang",
        copy=True,
    ).add_local_dir(
        LOCAL_FLASH_FDE_SRC,
        remote_path="/root/autoinference",
        copy=True,
    )

with image.imports():
    import requests
    from autoinference.runtime.endpoint import SGLangEndpoint
    from huggingface_hub import snapshot_download
    from PIL import Image
    from safetensors.torch import load_file, save as save_safetensors
    from transformers import AutoTokenizer

    from sglang.srt.lora.peft import compile_peft_lora_payload


def _phase_names(mode: str) -> list[str]:
    if mode == "all":
        return ["real_text", "synthetic_linear_text", "real_multimodal"]
    if mode in {"real_text", "synthetic_linear_text", "real_multimodal"}:
        return [mode]
    raise ValueError(f"Unsupported mode={mode!r}.")


def _build_server_args(
    *,
    enable_multimodal: bool,
    enable_reference_lora: bool = False,
    adapter_config: dict[str, Any] | None = None,
    reference_lora_path: str | None = None,
) -> dict[str, str]:
    server_args = dict(COMMON_SERVER_ARGS)
    if enable_reference_lora:
        if reference_lora_path is None:
            raise ValueError(
                "reference_lora_path is required for the reference LoRA server."
            )
        server_args["--context-length"] = str(REFERENCE_CONTEXT_LENGTH)
        server_args["--chunked-prefill-size"] = str(REFERENCE_CHUNKED_PREFILL_SIZE)
        server_args["--max-prefill-tokens"] = str(REFERENCE_MAX_PREFILL_TOKENS)
        server_args["--mem-fraction-static"] = REFERENCE_MEM_FRACTION_STATIC
        server_args["--lora-paths"] = f"{REFERENCE_LORA_NAME}={reference_lora_path}"
        server_args["--lora-backend"] = REFERENCE_LORA_BACKEND
        server_args["--max-loras-per-batch"] = REFERENCE_MAX_LORAS_PER_BATCH
        server_args["--max-loaded-loras"] = REFERENCE_MAX_LOADED_LORAS
        if adapter_config is not None and isinstance(
            adapter_config.get("target_modules"), list
        ):
            server_args["--lora-target-modules"] = " ".join(
                adapter_config["target_modules"]
            )
    else:
        server_args["--custom-weight-loader"] = MERGE_FROM_TENSORS_LOAD_FORMAT
    if enable_multimodal:
        server_args["--enable-multimodal"] = ""
        server_args["--mm-attention-backend"] = MM_ATTENTION_BACKEND
    return server_args


def _start_endpoint(
    tp_size: int,
    *,
    enable_multimodal: bool,
    enable_reference_lora: bool = False,
    adapter_config: dict[str, Any] | None = None,
    reference_lora_path: str | None = None,
) -> Any:
    endpoint = SGLangEndpoint(
        model_path=MODEL_NAME,
        worker_port=DEFAULT_PORT,
        tp=tp_size,
        extra_server_args=_build_server_args(
            enable_multimodal=enable_multimodal,
            enable_reference_lora=enable_reference_lora,
            adapter_config=adapter_config,
            reference_lora_path=reference_lora_path,
        ),
        health_timeout=40 * MINUTES,
        health_poll_interval=10.0,
    )
    try:
        endpoint.start()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "SGLang endpoint failed to start "
            f"(returncode={exc.returncode}, cmd={exc.cmd!r})."
        ) from exc
    return endpoint


def _request_generate(base_url: str, payload: dict[str, Any]) -> Any:
    response = requests.post(f"{base_url}/generate", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


def _model_info(base_url: str) -> dict[str, Any]:
    response = requests.get(f"{base_url}/model_info", timeout=30)
    response.raise_for_status()
    return response.json()


def _extract_input_logprobs(result: Any) -> list[float]:
    if isinstance(result, list):
        logprobs: list[float] = []
        for item in result:
            logprobs.extend(_extract_input_logprobs(item))
        return logprobs
    return [entry[0] for entry in result["meta_info"]["input_token_logprobs"]][1:]


def _mean_abs_diff(lhs: list[float], rhs: list[float]) -> float:
    if len(lhs) != len(rhs):
        raise ValueError(f"Logprob lengths differ: {len(lhs)} vs {len(rhs)}.")
    if not lhs:
        raise ValueError("Expected non-empty logprob sequences.")
    return sum(abs(a - b) for a, b in zip(lhs, rhs)) / len(lhs)


def _max_abs_diff(lhs: list[float], rhs: list[float]) -> float:
    if len(lhs) != len(rhs):
        raise ValueError(f"Logprob lengths differ: {len(lhs)} vs {len(rhs)}.")
    if not lhs:
        raise ValueError("Expected non-empty logprob sequences.")
    return max(abs(a - b) for a, b in zip(lhs, rhs))


def _serialize_payload(named_tensors: list[tuple[str, Any]]) -> tuple[bytes, str]:
    weights_bytes = save_safetensors(
        {name: tensor.detach().cpu().contiguous() for name, tensor in named_tensors}
    )
    digest = hashlib.sha256(weights_bytes).hexdigest()
    return weights_bytes, digest


def _post_update_weights(
    base_url: str,
    named_tensors: list[tuple[str, Any]],
    loader_metadata: dict[str, Any],
    *,
    weight_version: str,
    base_weight_version: str = "default",
) -> dict[str, Any]:
    weights_bytes, digest = _serialize_payload(named_tensors)
    metadata = {
        "tensor_format": "safetensors",
        "load_format": MERGE_FROM_TENSORS_LOAD_FORMAT,
        "flush_cache": True,
        "base_weight_version": base_weight_version,
        "weight_version": weight_version,
        "payload_digest": digest,
        "loader_metadata": loader_metadata,
        "crash_on_error": True,
    }
    response = requests.post(
        f"{base_url}/update_weights_from_bytes",
        data={"metadata": json.dumps(metadata)},
        files={
            "weights_file": (
                "weights.safetensors",
                weights_bytes,
                "application/octet-stream",
            )
        },
        timeout=300,
    )
    response.raise_for_status()
    body = response.json()
    body["payload_digest"] = digest
    return body


def _get_weights_by_name(
    base_url: str,
    name: str,
    *,
    truncate_size: int = 2,
) -> Any:
    import torch

    response = requests.post(
        f"{base_url}/get_weights_by_name",
        json={"name": name, "truncate_size": truncate_size},
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()
    if body is None:
        raise AssertionError(f"Server returned no weights for parameter {name!r}.")
    return torch.tensor(body, dtype=torch.float32)


def _resolve_component_scaling(
    component_spec: dict[str, Any],
    target_spec: dict[str, Any],
    loader_metadata: dict[str, Any],
) -> float:
    scaling = component_spec.get("scaling")
    if scaling is None:
        scaling = target_spec.get("scaling")
    if scaling is not None:
        return float(scaling)

    rank = (
        component_spec.get("rank")
        or component_spec.get("r")
        or target_spec.get("rank")
        or target_spec.get("r")
        or loader_metadata.get("rank")
        or loader_metadata.get("r")
    )
    alpha = (
        component_spec.get("lora_alpha")
        or target_spec.get("lora_alpha")
        or loader_metadata.get("lora_alpha")
    )
    if alpha is not None and rank is not None:
        return float(alpha) / float(rank)
    return 1.0


def _resolve_parameter_oracle_weight_dtype() -> Any:
    import torch

    if PARAMETER_ORACLE_WEIGHT_DTYPE == "bf16":
        return torch.bfloat16
    if PARAMETER_ORACLE_WEIGHT_DTYPE == "fp16":
        return torch.float16
    if PARAMETER_ORACLE_WEIGHT_DTYPE == "fp32":
        return torch.float32
    raise ValueError(
        f"Unsupported PARAMETER_ORACLE_WEIGHT_DTYPE={PARAMETER_ORACLE_WEIGHT_DTYPE!r}."
    )


def _normalize_parameter_oracle_shard_id(shard_id: Any) -> tuple[int, ...]:
    if shard_id is None:
        raise ValueError("Parameter oracle shard id cannot be None.")
    if isinstance(shard_id, (list, tuple)):
        return tuple(_normalize_parameter_oracle_shard_token(token) for token in shard_id)
    return (_normalize_parameter_oracle_shard_token(shard_id),)


def _normalize_parameter_oracle_shard_token(token: Any) -> int:
    if isinstance(token, int):
        return token
    if token == "q":
        return 0
    if token == "k":
        return 1
    if token == "v":
        return 2
    raise ValueError(f"Unsupported parameter-oracle shard token {token!r}.")


def _is_parameter_oracle_supported_target(
    target_spec: dict[str, Any],
    *,
    tensor_dict: dict[str, Any],
) -> bool:
    components = target_spec.get("components", [target_spec])
    for component in components:
        if component.get("fused_experts"):
            return False
        lora_a = tensor_dict.get(component["lora_a_name"])
        lora_b = tensor_dict.get(component["lora_b_name"])
        if lora_a is None or lora_b is None:
            return False
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            return False
    return True


def _select_parameter_oracle_targets(
    named_tensors: list[tuple[str, Any]],
    loader_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    tensor_dict = dict(named_tensors)
    selected: list[dict[str, Any]] = []
    for target_spec in loader_metadata.get("targets", []):
        if not _is_parameter_oracle_supported_target(
            target_spec,
            tensor_dict=tensor_dict,
        ):
            continue
        selected.append(target_spec)
        if (
            PARAMETER_ORACLE_MAX_TARGETS > 0
            and len(selected) >= PARAMETER_ORACLE_MAX_TARGETS
        ):
            break
    return selected


def _compose_parameter_oracle_delta(
    target_spec: dict[str, Any],
    *,
    tensor_dict: dict[str, Any],
    loader_metadata: dict[str, Any],
) -> Any:
    import torch

    direct_total = None
    sharded_components: list[tuple[tuple[int, ...], Any]] = []
    for component in target_spec.get("components", [target_spec]):
        scaling = _resolve_component_scaling(component, target_spec, loader_metadata)
        lora_a = tensor_dict[component["lora_a_name"]].to(dtype=torch.float32)
        lora_b = tensor_dict[component["lora_b_name"]].to(dtype=torch.float32)
        component_delta = torch.matmul(lora_b, lora_a) * scaling
        shard_id = component.get("shard_id")
        if shard_id is None:
            if sharded_components:
                raise ValueError(
                    f"Cannot mix direct and sharded components for {target_spec['target_name']!r}."
                )
            direct_total = (
                component_delta
                if direct_total is None
                else direct_total + component_delta
            )
            continue

        if direct_total is not None:
            raise ValueError(
                f"Cannot mix direct and sharded components for {target_spec['target_name']!r}."
            )
        sharded_components.append(
            (_normalize_parameter_oracle_shard_id(shard_id), component_delta)
        )

    if direct_total is not None:
        return direct_total
    if not sharded_components:
        raise ValueError(
            f"No parameter-oracle-compatible components for {target_spec['target_name']!r}."
        )

    sharded_components.sort(key=lambda item: item[0])
    return torch.cat([component_delta for _, component_delta in sharded_components], dim=0)


def _build_parameter_oracle_expected_after_slices(
    named_tensors: list[tuple[str, Any]],
    loader_metadata: dict[str, Any],
    *,
    before_slices: dict[str, Any],
    target_specs: list[dict[str, Any]],
    truncate_size: int = PARAMETER_ORACLE_TRUNCATE_SIZE,
) -> dict[str, Any]:
    import torch

    tensor_dict = dict(named_tensors)
    target_dtype = _resolve_parameter_oracle_weight_dtype()
    expected_after: dict[str, Any] = {}
    for target_spec in target_specs:
        target_name = target_spec["target_name"]
        target_delta = _compose_parameter_oracle_delta(
            target_spec,
            tensor_dict=tensor_dict,
            loader_metadata=loader_metadata,
        )
        before_slice = before_slices[target_name]
        row_count = min(truncate_size, before_slice.shape[0], target_delta.shape[0])
        expected_after[target_name] = (
            before_slice[:row_count]
            .to(dtype=torch.float32)
            .add_(target_delta[:row_count].to(dtype=torch.float32))
            .to(dtype=target_dtype)
            .to(dtype=torch.float32)
            .cpu()
        )
    return expected_after


def _collect_parameter_oracle(
    *,
    base_url: str,
    expected_after_slices: dict[str, Any],
    truncate_size: int = PARAMETER_ORACLE_TRUNCATE_SIZE,
) -> dict[str, Any]:
    if not expected_after_slices:
        return {}

    per_target: dict[str, Any] = {}
    overall_max_abs_after_diff = 0.0
    for target_name, expected_after in expected_after_slices.items():
        after_weight = _get_weights_by_name(
            base_url,
            target_name,
            truncate_size=truncate_size,
        )
        compared_rows = min(after_weight.shape[0], expected_after.shape[0])
        max_abs_diff = float(
            (after_weight[:compared_rows] - expected_after[:compared_rows]).abs().max().item()
        )
        mean_abs_diff = float(
            (after_weight[:compared_rows] - expected_after[:compared_rows]).abs().mean().item()
        )
        overall_max_abs_after_diff = max(overall_max_abs_after_diff, max_abs_diff)
        per_target[target_name] = {
            "max_abs_after_diff": max_abs_diff,
            "mean_abs_after_diff": mean_abs_diff,
        }

    return {
        "truncate_size": truncate_size,
        "target_count": len(per_target),
        "weight_dtype": PARAMETER_ORACLE_WEIGHT_DTYPE,
        "overall_max_abs_after_diff": overall_max_abs_after_diff,
        "per_target": per_target,
    }


def _load_real_adapter_payload() -> dict[str, Any]:
    adapter_path = snapshot_download(
        REAL_ADAPTER_REPO,
        allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
    )
    with open(os.path.join(adapter_path, "adapter_config.json")) as config_file:
        adapter_config = json.load(config_file)
    tensors = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))
    named_tensors = list(tensors.items())
    compiled_tensors, compiled_loader_metadata = compile_peft_lora_payload(
        named_tensors,
        target_resolver="qwen3_5",
        loader_metadata={"adapter_config": adapter_config},
    )
    return {
        "adapter_path": adapter_path,
        "adapter_config": adapter_config,
        "raw_named_tensors": named_tensors,
        "compiled_named_tensors": compiled_tensors,
        "compiled_loader_metadata": compiled_loader_metadata,
    }


def _classify_raw_adapter_tensor_family(tensor_name: str) -> str | None:
    if ".mlp.shared_expert." in tensor_name:
        return "shared_expert"
    if ".mlp.experts." in tensor_name:
        return "routed_expert"
    if ".linear_attn." in tensor_name:
        return "linear_attn"
    if ".self_attn." in tensor_name:
        return "dense_attention"
    if (
        ".mlp." in tensor_name
        and ".mlp.experts." not in tensor_name
        and ".mlp.shared_expert." not in tensor_name
    ):
        return "dense_mlp"
    if "embed_tokens" in tensor_name or "lm_head" in tensor_name:
        return "dense_embeddings"
    return None


def _filter_real_adapter_named_tensors(
    named_tensors: list[tuple[str, Any]],
    *,
    family: str,
) -> list[tuple[str, Any]]:
    if family == "all":
        return list(named_tensors)
    special_family_predicates = {
        "dense_attention_qkv": lambda name: any(
            f".self_attn.{module_name}." in name
            for module_name in ("q_proj", "k_proj", "v_proj")
        ),
        "dense_attention_o_proj": lambda name: ".self_attn.o_proj." in name,
    }
    predicate = special_family_predicates.get(family)
    if predicate is not None:
        filtered = [
            (name, tensor) for name, tensor in named_tensors if predicate(name)
        ]
        if not filtered:
            raise ValueError(f"No adapter tensors found for family={family!r}.")
        return filtered
    family_groups = {
        "dense_language": {
            "dense_attention",
            "dense_mlp",
            "dense_embeddings",
        },
    }
    accepted_families = family_groups.get(family, {family})
    filtered = [
        (name, tensor)
        for name, tensor in named_tensors
        if _classify_raw_adapter_tensor_family(name) in accepted_families
    ]
    if not filtered:
        raise ValueError(f"No adapter tensors found for family={family!r}.")
    return filtered


def _compile_real_adapter_payload(
    *,
    adapter_config: dict[str, Any],
    raw_named_tensors: list[tuple[str, Any]],
) -> dict[str, Any]:
    compiled_tensors, compiled_loader_metadata = compile_peft_lora_payload(
        raw_named_tensors,
        target_resolver="qwen3_5",
        loader_metadata={"adapter_config": adapter_config},
    )
    return {
        "adapter_config": adapter_config,
        "raw_named_tensors": raw_named_tensors,
        "compiled_named_tensors": compiled_tensors,
        "compiled_loader_metadata": compiled_loader_metadata,
    }


def _materialize_reference_adapter(
    *,
    adapter_config: dict[str, Any],
    raw_named_tensors: list[tuple[str, Any]],
    adapter_name: str,
) -> str:
    adapter_dir = Path(tempfile.mkdtemp(prefix=f"{adapter_name}-", dir="/tmp"))
    adapter_dir.joinpath("adapter_model.safetensors").write_bytes(
        save_safetensors(
            {name: tensor.detach().cpu().contiguous() for name, tensor in raw_named_tensors}
        )
    )
    adapter_dir.joinpath("adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2, sort_keys=True)
    )
    return str(adapter_dir)


def _load_probe_text_config() -> Any:
    config_path = snapshot_download(
        MODEL_NAME,
        allow_patterns=["config.json"],
    )
    with open(os.path.join(config_path, "config.json")) as config_file:
        config_dict = json.load(config_file)
    return SimpleNamespace(**config_dict.get("text_config", config_dict))


def _text_logprob_payloads(tokenizer: Any) -> list[list[int]]:
    prompts = [
        "Write a two-line Python docstring for a function that hashes request payloads for dedupe validation.",
        "List three concrete failure modes when replaying a duplicate weight update into a live server.",
    ]
    return [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]


def _text_logprobs(
    base_url: str,
    tokenizer: Any,
    *,
    lora_path: str | None = None,
) -> list[float]:
    payload = {
        "input_ids": _text_logprob_payloads(tokenizer),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 0,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if lora_path is not None:
        payload["lora_path"] = [lora_path] * len(payload["input_ids"])
    return _extract_input_logprobs(_request_generate(base_url, payload))


def _multimodal_logprobs(base_url: str) -> list[float]:
    payload = {
        "text": (
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            "Answer with one lowercase word: what is the dominant color?"
        ),
        "image_data": _red_square_data_uri(),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 0,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    return _extract_input_logprobs(_request_generate(base_url, payload))


def _red_square_data_uri() -> str:
    image = Image.new("RGB", (16, 16), color=(220, 20, 60))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_linear_attn_probe(
    text_config: Any,
    *,
    layer_id: int = 0,
    rank: int = 4,
    seed: int = 17,
) -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    hidden_size = int(text_config.hidden_size)
    key_dim = int(text_config.linear_key_head_dim) * int(text_config.linear_num_key_heads)
    value_dim = int(text_config.linear_value_head_dim) * int(
        text_config.linear_num_value_heads
    )
    num_v_heads = int(text_config.linear_num_value_heads)

    target_prefix = f"model.layers.{layer_id}.linear_attn"
    payload_tensors = [
        (
            f"{target_prefix}.in_proj_qkv_probe.lora_A.weight",
            _make_probe_factor((rank, hidden_size), seed=seed + 1),
        ),
        (
            f"{target_prefix}.in_proj_qkv_probe.lora_B.weight",
            _make_probe_factor((2 * key_dim + value_dim, rank), seed=seed + 2),
        ),
        (
            f"{target_prefix}.in_proj_z_probe.lora_A.weight",
            _make_probe_factor((rank, hidden_size), seed=seed + 3),
        ),
        (
            f"{target_prefix}.in_proj_z_probe.lora_B.weight",
            _make_probe_factor((value_dim, rank), seed=seed + 4),
        ),
        (
            f"{target_prefix}.in_proj_b_probe.lora_A.weight",
            _make_probe_factor((rank, hidden_size), seed=seed + 5),
        ),
        (
            f"{target_prefix}.in_proj_b_probe.lora_B.weight",
            _make_probe_factor((num_v_heads, rank), seed=seed + 6),
        ),
        (
            f"{target_prefix}.in_proj_a_probe.lora_A.weight",
            _make_probe_factor((rank, hidden_size), seed=seed + 7),
        ),
        (
            f"{target_prefix}.in_proj_a_probe.lora_B.weight",
            _make_probe_factor((num_v_heads, rank), seed=seed + 8),
        ),
    ]
    loader_metadata = {
        "rank": rank,
        "lora_alpha": rank,
        "targets": [
            {
                "target_name": f"{target_prefix}.in_proj_qkvz.weight",
                "components": [
                    {
                        "component_id": "in_proj_qkv",
                        "lora_a_name": f"{target_prefix}.in_proj_qkv_probe.lora_A.weight",
                        "lora_b_name": f"{target_prefix}.in_proj_qkv_probe.lora_B.weight",
                        "shard_id": [0, 1, 2],
                    },
                    {
                        "component_id": "in_proj_z",
                        "lora_a_name": f"{target_prefix}.in_proj_z_probe.lora_A.weight",
                        "lora_b_name": f"{target_prefix}.in_proj_z_probe.lora_B.weight",
                        "shard_id": 3,
                    },
                ],
            },
            {
                "target_name": f"{target_prefix}.in_proj_ba.weight",
                "components": [
                    {
                        "component_id": "in_proj_b",
                        "lora_a_name": f"{target_prefix}.in_proj_b_probe.lora_A.weight",
                        "lora_b_name": f"{target_prefix}.in_proj_b_probe.lora_B.weight",
                        "shard_id": 0,
                    },
                    {
                        "component_id": "in_proj_a",
                        "lora_a_name": f"{target_prefix}.in_proj_a_probe.lora_A.weight",
                        "lora_b_name": f"{target_prefix}.in_proj_a_probe.lora_B.weight",
                        "shard_id": 1,
                    },
                ],
            },
        ],
    }
    return payload_tensors, loader_metadata


def _make_probe_factor(shape: tuple[int, ...], *, seed: int) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.02


def _summarize_raw_adapter_families(named_tensors: list[tuple[str, Any]]) -> dict[str, bool]:
    tensor_names = [name for name, _ in named_tensors]
    return {
        "has_dense_language": any(
            ".self_attn." in name
            or (
                ".mlp." in name
                and ".mlp.experts." not in name
                and ".mlp.shared_expert." not in name
            )
            or "embed_tokens" in name
            or "lm_head" in name
            for name in tensor_names
        ),
        "has_shared_expert": any(".mlp.shared_expert." in name for name in tensor_names),
        "has_routed_expert": any(".mlp.experts." in name for name in tensor_names),
        "has_linear_attn": any(".linear_attn." in name for name in tensor_names),
    }


def _summarize_compiled_targets(loader_metadata: dict[str, Any]) -> dict[str, int]:
    stats = {
        "target_count": 0,
        "dense_language_targets": 0,
        "shared_expert_targets": 0,
        "routed_expert_components": 0,
        "linear_attn_targets": 0,
    }
    for target_spec in loader_metadata.get("targets", []):
        target_name = target_spec["target_name"]
        stats["target_count"] += 1
        if ".linear_attn." in target_name:
            stats["linear_attn_targets"] += 1
        elif ".mlp.shared_expert." in target_name:
            stats["shared_expert_targets"] += 1
        elif ".mlp.experts." not in target_name:
            stats["dense_language_targets"] += 1

        for component in target_spec.get("components", [target_spec]):
            if component.get("fused_experts"):
                stats["routed_expert_components"] += 1
    return stats


def _assert_compiled_family_coverage(
    raw_family_stats: dict[str, bool],
    compiled_target_stats: dict[str, int],
) -> None:
    if raw_family_stats["has_dense_language"] and not compiled_target_stats[
        "dense_language_targets"
    ]:
        raise AssertionError("Compiled payload lost dense language target coverage.")
    if raw_family_stats["has_shared_expert"] and not compiled_target_stats[
        "shared_expert_targets"
    ]:
        raise AssertionError("Compiled payload lost shared-expert target coverage.")
    if raw_family_stats["has_routed_expert"] and not compiled_target_stats[
        "routed_expert_components"
    ]:
        raise AssertionError("Compiled payload lost routed-expert target coverage.")
    if raw_family_stats["has_linear_attn"] and not compiled_target_stats[
        "linear_attn_targets"
    ]:
        raise AssertionError("Compiled payload lost linear-attention target coverage.")


def _unload_lora_adapter(base_url: str, *, lora_name: str) -> dict[str, Any]:
    response = requests.post(
        f"{base_url}/unload_lora_adapter",
        json={"lora_name": lora_name},
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()
    if not body.get("success"):
        raise AssertionError(
            f"Failed to unload reference LoRA adapter {lora_name!r}: {body!r}."
        )
    return body


def _run_reference_text_lora_oracle(
    tp_size: int,
    *,
    tokenizer: Any,
    adapter_config: dict[str, Any],
    adapter_path: str,
) -> dict[str, Any]:
    base_url = f"http://127.0.0.1:{DEFAULT_PORT}"
    endpoint = _start_endpoint(
        tp_size,
        enable_multimodal=False,
        enable_reference_lora=True,
        adapter_config=adapter_config,
        reference_lora_path=adapter_path,
    )
    try:
        base_before = _text_logprobs(base_url, tokenizer)
        lora_logprobs = _text_logprobs(
            base_url,
            tokenizer,
            lora_path=REFERENCE_LORA_NAME,
        )
        unload_result = _unload_lora_adapter(
            base_url,
            lora_name=REFERENCE_LORA_NAME,
        )
        base_after_unload = _text_logprobs(base_url, tokenizer)
    finally:
        endpoint.stop()

    lora_effect_mean_abs_diff = _mean_abs_diff(base_before, lora_logprobs)
    unload_restore_max_abs_diff = _max_abs_diff(base_before, base_after_unload)
    if lora_effect_mean_abs_diff <= 1e-6:
        raise AssertionError("Reference runtime LoRA path did not change prompt logprobs.")
    if unload_restore_max_abs_diff > REFERENCE_UNLOAD_MAX_ABS_TOL:
        raise AssertionError(
            "Reference runtime LoRA unload did not restore base logprobs within "
            f"tolerance {REFERENCE_UNLOAD_MAX_ABS_TOL}: max_diff={unload_restore_max_abs_diff}."
        )

    return {
        "base_logprobs": base_before,
        "lora_logprobs": lora_logprobs,
        "unload_result": unload_result,
        "lora_effect_mean_abs_logprob_diff": lora_effect_mean_abs_diff,
        "unload_restore_max_abs_logprob_diff": unload_restore_max_abs_diff,
    }


def _run_phase(
    base_url: str,
    *,
    name: str,
    named_tensors: list[tuple[str, Any]],
    loader_metadata: dict[str, Any],
    get_logprobs: Any,
    reference_oracle: dict[str, Any] | None = None,
    raw_family_stats: dict[str, bool] | None = None,
    compiled_target_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    expected_weight_version = f"{name}.v1"
    baseline_model_info = _model_info(base_url)
    if baseline_model_info["weight_version"] != "default":
        raise AssertionError(
            f"{name}: expected fresh server with weight_version='default', "
            f"got {baseline_model_info['weight_version']!r}."
        )

    if raw_family_stats is not None and compiled_target_stats is not None:
        _assert_compiled_family_coverage(raw_family_stats, compiled_target_stats)

    parameter_oracle_target_specs = _select_parameter_oracle_targets(
        named_tensors,
        loader_metadata,
    )
    parameter_oracle_before_slices = {
        target_spec["target_name"]: _get_weights_by_name(
            base_url,
            target_spec["target_name"],
            truncate_size=PARAMETER_ORACLE_TRUNCATE_SIZE,
        )
        for target_spec in parameter_oracle_target_specs
    }
    parameter_oracle_expected_after_slices = _build_parameter_oracle_expected_after_slices(
        named_tensors,
        loader_metadata,
        before_slices=parameter_oracle_before_slices,
        target_specs=parameter_oracle_target_specs,
    )

    baseline = get_logprobs()
    update_result = _post_update_weights(
        base_url,
        named_tensors,
        loader_metadata,
        weight_version=expected_weight_version,
    )
    if expected_weight_version not in update_result["message"]:
        raise AssertionError(
            f"{name}: update response did not report the new weight version: "
            f"{update_result['message']!r}."
        )

    after_apply_model_info = _model_info(base_url)
    if after_apply_model_info["weight_version"] != expected_weight_version:
        raise AssertionError(
            f"{name}: expected active weight_version={expected_weight_version!r}, "
            f"got {after_apply_model_info['weight_version']!r}."
        )

    after_apply = get_logprobs()
    parameter_oracle = _collect_parameter_oracle(
        base_url=base_url,
        expected_after_slices=parameter_oracle_expected_after_slices,
    )
    replay_result = _post_update_weights(
        base_url,
        named_tensors,
        loader_metadata,
        weight_version=expected_weight_version,
    )
    if "already applied" not in replay_result["message"].lower():
        raise AssertionError(
            f"{name}: replay response did not report duplicate suppression: "
            f"{replay_result['message']!r}."
        )

    after_replay_model_info = _model_info(base_url)
    if after_replay_model_info["weight_version"] != expected_weight_version:
        raise AssertionError(
            f"{name}: replay changed weight_version unexpectedly to "
            f"{after_replay_model_info['weight_version']!r}."
        )

    after_replay = get_logprobs()
    mean_abs_diff = _mean_abs_diff(baseline, after_apply)
    replay_diff = _mean_abs_diff(after_apply, after_replay)
    if mean_abs_diff <= 1e-6:
        raise AssertionError(f"{name}: update did not change prompt logprobs.")
    if replay_diff > 1e-9:
        raise AssertionError(f"{name}: duplicate replay changed prompt logprobs.")
    if (
        parameter_oracle
        and parameter_oracle["overall_max_abs_after_diff"] > PARAMETER_ORACLE_MAX_ABS_TOL
    ):
        raise AssertionError(
            f"{name}: parameter oracle mismatch "
            f"(max_after_diff={parameter_oracle['overall_max_abs_after_diff']}, "
            f"tol={PARAMETER_ORACLE_MAX_ABS_TOL}). "
            f"parameter_oracle={json.dumps(parameter_oracle, sort_keys=True)}"
        )

    result = {
        "phase": name,
        "num_tensors": len(named_tensors),
        "weight_version_before": baseline_model_info["weight_version"],
        "weight_version_after_apply": after_apply_model_info["weight_version"],
        "weight_version_after_replay": after_replay_model_info["weight_version"],
        "mean_abs_logprob_diff": mean_abs_diff,
        "replay_abs_logprob_diff": replay_diff,
        "update_result": update_result,
        "replay_result": replay_result,
    }
    if parameter_oracle:
        result["parameter_oracle"] = parameter_oracle
    if reference_oracle is not None:
        reference_base_mean_abs_diff = _mean_abs_diff(
            baseline, reference_oracle["base_logprobs"]
        )
        reference_base_max_abs_diff = _max_abs_diff(
            baseline, reference_oracle["base_logprobs"]
        )
        reference_lora_mean_abs_diff = _mean_abs_diff(
            after_apply, reference_oracle["lora_logprobs"]
        )
        reference_lora_max_abs_diff = _max_abs_diff(
            after_apply, reference_oracle["lora_logprobs"]
        )
        if reference_base_max_abs_diff > REFERENCE_BASE_MAX_ABS_TOL:
            raise AssertionError(
                f"{name}: base logprobs differ between reference and target server shapes "
                f"(max_diff={reference_base_max_abs_diff}, tol={REFERENCE_BASE_MAX_ABS_TOL})."
            )
        if (
            ENFORCE_REFERENCE_LORA_PARITY
            and reference_lora_max_abs_diff > REFERENCE_LORA_MAX_ABS_TOL
        ):
            oracle_suffix = ""
            if parameter_oracle:
                oracle_suffix = (
                    " parameter_oracle="
                    + json.dumps(
                        {
                            "overall_max_abs_after_diff": parameter_oracle[
                                "overall_max_abs_after_diff"
                            ],
                            "target_count": parameter_oracle["target_count"],
                            "per_target": parameter_oracle["per_target"],
                        },
                        sort_keys=True,
                    )
                )
            raise AssertionError(
                f"{name}: live-sync logprobs do not match reference runtime LoRA "
                f"(max_diff={reference_lora_max_abs_diff}, tol={REFERENCE_LORA_MAX_ABS_TOL})."
                f"{oracle_suffix}"
            )
        result["reference_oracle"] = {
            "reference_base_mean_abs_logprob_diff": reference_base_mean_abs_diff,
            "reference_base_max_abs_logprob_diff": reference_base_max_abs_diff,
            "reference_lora_mean_abs_logprob_diff": reference_lora_mean_abs_diff,
            "reference_lora_max_abs_logprob_diff": reference_lora_max_abs_diff,
            "reference_lora_max_abs_logprob_tol": REFERENCE_LORA_MAX_ABS_TOL,
            "reference_lora_parity_enforced": ENFORCE_REFERENCE_LORA_PARITY,
            "reference_lora_within_tol": (
                reference_lora_max_abs_diff <= REFERENCE_LORA_MAX_ABS_TOL
            ),
            "reference_lora_effect_mean_abs_logprob_diff": reference_oracle[
                "lora_effect_mean_abs_logprob_diff"
            ],
            "reference_unload_restore_max_abs_logprob_diff": reference_oracle[
                "unload_restore_max_abs_logprob_diff"
            ],
        }
    if compiled_target_stats is not None:
        result["compiled_target_stats"] = compiled_target_stats
    if raw_family_stats is not None:
        result["raw_family_stats"] = raw_family_stats
    return result


@app.function(
    image=image,
    gpu=GPU,
    timeout=90 * MINUTES,
    volumes={HF_CACHE_PATH: hf_cache_volume},
)
def run_validation(
    mode: str = "all",
    tp_size: int = 1,
    family: str = "all",
) -> list[dict[str, Any]]:
    base_url = f"http://127.0.0.1:{DEFAULT_PORT}"
    phase_names = _phase_names(mode)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    text_config = (
        _load_probe_text_config()
        if "synthetic_linear_text" in phase_names
        else None
    )
    real_payload = (
        _load_real_adapter_payload()
        if any(phase_name.startswith("real_") for phase_name in phase_names)
        else None
    )

    results = []
    for phase_name in phase_names:
        phase_payload = real_payload
        reference_adapter_path = None
        if phase_name.startswith("real_"):
            if real_payload is None:
                raise ValueError(f"real_payload is required for {phase_name}.")
            filtered_raw_named_tensors = _filter_real_adapter_named_tensors(
                real_payload["raw_named_tensors"],
                family=family,
            )
            phase_payload = _compile_real_adapter_payload(
                adapter_config=real_payload["adapter_config"],
                raw_named_tensors=filtered_raw_named_tensors,
            )
            reference_adapter_path = (
                real_payload["adapter_path"]
                if family == "all"
                else _materialize_reference_adapter(
                    adapter_config=phase_payload["adapter_config"],
                    raw_named_tensors=phase_payload["raw_named_tensors"],
                    adapter_name=f"qwen35-{phase_name}-{family}",
                )
            )

        reference_oracle = None
        if phase_name == "real_text":
            reference_oracle = _run_reference_text_lora_oracle(
                tp_size,
                tokenizer=tokenizer,
                adapter_config=phase_payload["adapter_config"],
                adapter_path=reference_adapter_path,
            )

        endpoint = _start_endpoint(
            tp_size,
            enable_multimodal=(phase_name == "real_multimodal"),
        )
        try:
            if phase_name == "real_text":
                results.append(
                    _run_phase(
                        base_url,
                        name=f"{phase_name}.{family}",
                        named_tensors=phase_payload["compiled_named_tensors"],
                        loader_metadata=phase_payload["compiled_loader_metadata"],
                        get_logprobs=lambda: _text_logprobs(base_url, tokenizer),
                        reference_oracle=reference_oracle,
                        raw_family_stats=_summarize_raw_adapter_families(
                            phase_payload["raw_named_tensors"]
                        ),
                        compiled_target_stats=_summarize_compiled_targets(
                            phase_payload["compiled_loader_metadata"]
                        ),
                    )
                )
            elif phase_name == "synthetic_linear_text":
                if text_config is None:
                    raise ValueError("text_config is required for synthetic_linear_text.")
                named_tensors, loader_metadata = _build_linear_attn_probe(text_config)
                results.append(
                    _run_phase(
                        base_url,
                        name=phase_name,
                        named_tensors=named_tensors,
                        loader_metadata=loader_metadata,
                        get_logprobs=lambda: _text_logprobs(base_url, tokenizer),
                        compiled_target_stats=_summarize_compiled_targets(loader_metadata),
                    )
                )
            elif phase_name == "real_multimodal":
                results.append(
                    _run_phase(
                        base_url,
                        name=f"{phase_name}.{family}",
                        named_tensors=phase_payload["compiled_named_tensors"],
                        loader_metadata=phase_payload["compiled_loader_metadata"],
                        get_logprobs=lambda: _multimodal_logprobs(base_url),
                        raw_family_stats=_summarize_raw_adapter_families(
                            phase_payload["raw_named_tensors"]
                        ),
                        compiled_target_stats=_summarize_compiled_targets(
                            phase_payload["compiled_loader_metadata"]
                        ),
                    )
                )
            else:
                raise ValueError(f"Unsupported phase_name={phase_name!r}.")
        finally:
            endpoint.stop()

    return results


@app.local_entrypoint()
def main(mode: str = "all", tp_size: int = 1, family: str = "all"):
    results = run_validation.remote(mode=mode, tp_size=tp_size, family=family)
    print(json.dumps(results, indent=2, sort_keys=True))
