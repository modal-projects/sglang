from __future__ import annotations

import json
import math
import os
import pathlib
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOTE_REPO_ROOT = pathlib.Path("/sgl-workspace/sglang")
HF_CACHE_PATH = "/root/.cache/huggingface"
HF_CACHE_VOLUME_NAME = os.getenv("HF_CACHE_VOLUME_NAME", "huggingface-cache")
SGLANG_IMAGE_TAG = os.getenv(
    "SGLANG_MODAL_IMAGE_TAG",
    "lmsysorg/sglang:nightly-dev-cu13-20260407-5cc246e0",
)
SGLANG_GPU = os.getenv("QWEN35_SGLANG_MODAL_GPU", "H200")
SGLANG_MEMORY_MB = int(os.getenv("QWEN35_SGLANG_MODAL_MEMORY_MB", "32768"))

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
    "local-adapters/qwen35-merged-lora-logprob-diff"
)
ADAPTER_VOLUME_CONFIG_REL = ADAPTER_VOLUME_SUBDIR / "adapter_config.json"
ADAPTER_VOLUME_WEIGHTS_REL = ADAPTER_VOLUME_SUBDIR / "sampler_weights_init.safetensors"

HF_IMAGE_ENV = {
    "HF_HUB_CACHE": HF_CACHE_PATH,
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

app = modal.App(name="sglang-flashinfer-trtllm-moe-layout-probe")
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
image = modal.Image.from_registry(SGLANG_IMAGE_TAG).env(HF_IMAGE_ENV)
if modal.is_local():
    image = image.add_local_dir(
        REPO_ROOT / "python/sglang",
        str(REMOTE_REPO_ROOT / "python/sglang"),
        copy=False,
    )

ROUTED_TENSOR_RE = re.compile(
    r"^(?P<layer_prefix>.+\.mlp\.experts)\.(?P<target>w1|w2|w3|gate_proj|down_proj|up_proj)"
    r"\.lora_(?P<kind>A|B)(?:\.default)?\.weight$"
)


def _ensure_import_paths() -> None:
    for path in (str(REMOTE_REPO_ROOT), str(REMOTE_REPO_ROOT / "python")):
        if path not in sys.path:
            sys.path.insert(0, path)


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
        batch.put_file(str(LOCAL_ADAPTER_CONFIG_PATH), str(ADAPTER_VOLUME_CONFIG_REL))
        batch.put_file(str(LOCAL_ADAPTER_WEIGHTS_PATH), str(ADAPTER_VOLUME_WEIGHTS_REL))

    return {
        "local_adapter_config": str(LOCAL_ADAPTER_CONFIG_PATH),
        "local_adapter_weights": str(LOCAL_ADAPTER_WEIGHTS_PATH),
        "local_adapter_weights_size_bytes": LOCAL_ADAPTER_WEIGHTS_PATH.stat().st_size,
        "volume_adapter_config": str(_remote_volume_path(ADAPTER_VOLUME_CONFIG_REL)),
        "volume_adapter_weights": str(_remote_volume_path(ADAPTER_VOLUME_WEIGHTS_REL)),
    }


def _resolve_scaling(config: dict[str, Any]) -> float:
    if "scaling" in config:
        return float(config["scaling"])
    rank = int(config["r"])
    if rank <= 0:
        raise ValueError(f"Invalid LoRA rank: {rank}")
    return float(config["lora_alpha"]) / rank


def _canonical_target(target: str) -> str:
    aliases = {
        "gate_proj": "w1",
        "down_proj": "w2",
        "up_proj": "w3",
    }
    return aliases.get(target, target)


@dataclass
class ProbeSelection:
    layer_prefix: str
    expert_index: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    w1_name: str
    w2_name: str
    w3_name: str
    w2_delta: Any


def _select_routed_probe(adapter_config: dict[str, Any], adapter_tensors: dict[str, Any]):
    import torch

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    routed_examples: list[str] = []
    for name, tensor in adapter_tensors.items():
        if ".mlp.experts" in name and len(routed_examples) < 24:
            routed_examples.append(f"{name} shape={tuple(tensor.shape)}")
        match = ROUTED_TENSOR_RE.match(name)
        if match is None:
            continue
        target = _canonical_target(match.group("target"))
        layer_prefix = match.group("layer_prefix")
        base_group = grouped.setdefault(layer_prefix, {})
        target_group = base_group.setdefault(target, {})
        target_group[match.group("kind")] = tensor

    candidate_bases = sorted(
        layer_prefix
        for layer_prefix, targets in grouped.items()
        if {"w1", "w2", "w3"} <= set(targets.keys())
        and {"A", "B"} <= set(targets["w1"].keys())
        and {"A", "B"} <= set(targets["w2"].keys())
        and {"A", "B"} <= set(targets["w3"].keys())
    )
    if not candidate_bases:
        raise RuntimeError(
            "Could not find a routed expert with complete w1/w2/w3 LoRA pairs. "
            f"First routed tensor examples: {routed_examples}"
        )

    layer_prefix = candidate_bases[0]
    routed = grouped[layer_prefix]
    w1_a = routed["w1"]["A"]
    w1_b = routed["w1"]["B"]
    w2_a = routed["w2"]["A"]
    w2_b = routed["w2"]["B"]
    w3_a = routed["w3"]["A"]
    w3_b = routed["w3"]["B"]

    expert_index = 0

    def pick_expert_slice(tensor: Any, *, expert_dim_value: int | None = None) -> Any:
        if tensor.dim() == 2:
            return tensor
        if tensor.dim() != 3:
            raise RuntimeError(f"Unsupported routed tensor shape: {tuple(tensor.shape)}")
        if expert_dim_value is not None and tensor.shape[0] == expert_dim_value:
            return tensor[expert_index]
        if tensor.shape[0] == 1:
            return tensor[0]
        return tensor[expert_index]

    inferred_num_experts = max(
        int(tensor.shape[0])
        for tensor in (w1_a, w1_b, w2_a, w2_b, w3_a, w3_b)
        if tensor.dim() == 3
    )

    w1_a_expert = pick_expert_slice(w1_a, expert_dim_value=inferred_num_experts)
    w1_b_expert = pick_expert_slice(w1_b, expert_dim_value=inferred_num_experts)
    w2_a_expert = pick_expert_slice(w2_a, expert_dim_value=inferred_num_experts)
    w2_b_expert = pick_expert_slice(w2_b, expert_dim_value=inferred_num_experts)
    w3_a_expert = pick_expert_slice(w3_a, expert_dim_value=inferred_num_experts)
    w3_b_expert = pick_expert_slice(w3_b, expert_dim_value=inferred_num_experts)

    hidden_size = int(w1_a_expert.shape[1])
    intermediate_size = int(w1_b_expert.shape[0])
    if int(w2_b_expert.shape[0]) != hidden_size:
        raise RuntimeError(f"Unexpected w2 B shape: {tuple(w2_b.shape)}")
    if int(w2_a_expert.shape[1]) != intermediate_size:
        raise RuntimeError(
            f"Inconsistent routed shapes: w1 intermediate={intermediate_size}, w2 A={tuple(w2_a.shape)}"
        )
    if (
        int(w3_a_expert.shape[1]) != hidden_size
        or int(w3_b_expert.shape[0]) != intermediate_size
    ):
        raise RuntimeError(
            "Inconsistent routed shapes between w1 and w3: "
            f"w1 A={tuple(w1_a.shape)} B={tuple(w1_b.shape)}, "
            f"w3 A={tuple(w3_a.shape)} B={tuple(w3_b.shape)}"
        )

    scaling = _resolve_scaling(adapter_config)
    w2_delta = (w2_b_expert.float() @ w2_a_expert.float()).mul_(scaling).to(
        torch.bfloat16
    )

    return ProbeSelection(
        layer_prefix=layer_prefix,
        expert_index=expert_index,
        num_experts=inferred_num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        w1_name=f"{layer_prefix}.w1",
        w2_name=f"{layer_prefix}.w2",
        w3_name=f"{layer_prefix}.w3",
        w2_delta=w2_delta,
    )


def _make_canonical_weight(shape: tuple[int, int], device: Any):
    import torch

    numel = math.prod(shape)
    base = torch.arange(numel, device=device, dtype=torch.float32)
    base = ((base % 1021) - 510.0) / 97.0
    return base.reshape(shape).to(torch.bfloat16).contiguous()


def _diff_stats(actual: Any, expected: Any) -> dict[str, Any]:
    diff = (actual.float() - expected.float()).abs()
    return {
        "shape": list(actual.shape),
        "numel": int(diff.numel()),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _current_restore(live_bf16: Any, canonical_shape: tuple[int, int]):
    return live_bf16.reshape(canonical_shape).contiguous()


def _inverse_block_layout(block_layout_u8: Any) -> Any:
    block_rows, rows, block_k = block_layout_u8.shape
    return (
        block_layout_u8.permute(1, 0, 2)
        .contiguous()
        .reshape(rows, block_rows * block_k)
    )


@app.function(
    image=image,
    gpu=SGLANG_GPU,
    memory=SGLANG_MEMORY_MB,
    timeout=60 * 60,
    retries=0,
    volumes={HF_CACHE_PATH: hf_cache_vol},
)
def probe_flashinfer_trtllm_layout() -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    _ensure_import_paths()

    volume_config_path = _remote_volume_path(ADAPTER_VOLUME_CONFIG_REL)
    volume_weights_path = _remote_volume_path(ADAPTER_VOLUME_WEIGHTS_REL)
    with open(volume_config_path, "r") as f:
        adapter_config = json.load(f)
    adapter_tensors = load_file(str(volume_weights_path))
    selection = _select_routed_probe(adapter_config, adapter_tensors)

    device = torch.device("cuda")
    w13_shape = (2 * selection.intermediate_size, selection.hidden_size)
    w2_shape = (selection.hidden_size, selection.intermediate_size)

    w13_canonical = _make_canonical_weight(w13_shape, device)
    w2_canonical = _make_canonical_weight(w2_shape, device)
    w2_delta = selection.w2_delta.to(device=device, dtype=torch.bfloat16).contiguous()

    epilogue_tile_m = 128
    block_k = 128
    cache: dict[tuple[str, tuple[int, ...]], Any] = {}

    def forward_w13(canonical_bf16: Any) -> Any:
        u8 = canonical_bf16.contiguous().view(torch.uint8)
        permute = _maybe_get_cached_w3_w1_permute_indices(cache, u8, epilogue_tile_m)
        shuffled = u8.index_select(0, permute.to(u8.device)).contiguous()
        return convert_to_block_layout(shuffled, block_k).view(torch.bfloat16).contiguous()

    def forward_w2(canonical_bf16: Any) -> Any:
        u8 = canonical_bf16.contiguous().view(torch.uint8)
        permute = get_w2_permute_indices_with_cache(cache, u8, epilogue_tile_m)
        shuffled = u8.index_select(0, permute.to(u8.device)).contiguous()
        return convert_to_block_layout(shuffled, block_k).view(torch.bfloat16).contiguous()

    def exact_inverse_w13(live_bf16: Any) -> Any:
        canonical_u8 = torch.empty(w13_shape, device=device, dtype=torch.bfloat16).view(
            torch.uint8
        )
        permute = _maybe_get_cached_w3_w1_permute_indices(
            cache, canonical_u8, epilogue_tile_m
        )
        inverse_permute = torch.argsort(permute)
        live_u8 = live_bf16.contiguous().view(torch.uint8)
        unblocked = _inverse_block_layout(live_u8)
        return (
            unblocked.index_select(0, inverse_permute.to(unblocked.device))
            .contiguous()
            .view(torch.bfloat16)
        )

    def exact_inverse_w2(live_bf16: Any) -> Any:
        canonical_u8 = torch.empty(w2_shape, device=device, dtype=torch.bfloat16).view(
            torch.uint8
        )
        permute = get_w2_permute_indices_with_cache(cache, canonical_u8, epilogue_tile_m)
        inverse_permute = torch.argsort(permute)
        live_u8 = live_bf16.contiguous().view(torch.uint8)
        unblocked = _inverse_block_layout(live_u8)
        return (
            unblocked.index_select(0, inverse_permute.to(unblocked.device))
            .contiguous()
            .view(torch.bfloat16)
        )

    w13_live = forward_w13(w13_canonical)
    w2_live = forward_w2(w2_canonical)

    w13_restored_current = _current_restore(w13_live, w13_shape)
    w2_restored_current = _current_restore(w2_live, w2_shape)
    w13_restored_exact = exact_inverse_w13(w13_live)
    w2_restored_exact = exact_inverse_w2(w2_live)

    expected_live_w2 = forward_w2((w2_canonical + w2_delta).contiguous())
    current_updated_live_w2 = forward_w2((w2_restored_current + w2_delta).contiguous())
    exact_updated_live_w2 = forward_w2((w2_restored_exact + w2_delta).contiguous())

    return {
        "gpu": SGLANG_GPU,
        "image_tag": SGLANG_IMAGE_TAG,
        "adapter_config_path": str(volume_config_path),
        "adapter_weights_path": str(volume_weights_path),
        "selected_routed_expert": {
            "layer_prefix": selection.layer_prefix,
            "expert_index": selection.expert_index,
            "num_experts": selection.num_experts,
            "hidden_size": selection.hidden_size,
            "intermediate_size": selection.intermediate_size,
            "w1_name": selection.w1_name,
            "w2_name": selection.w2_name,
            "w3_name": selection.w3_name,
            "w2_delta_shape": list(selection.w2_delta.shape),
        },
        "w13_probe": {
            "canonical_shape_bf16": list(w13_shape),
            "live_shape_bf16": list(w13_live.shape),
            "current_restore_vs_original": _diff_stats(
                w13_restored_current, w13_canonical
            ),
            "exact_inverse_vs_original": _diff_stats(w13_restored_exact, w13_canonical),
        },
        "w2_probe": {
            "canonical_shape_bf16": list(w2_shape),
            "live_shape_bf16": list(w2_live.shape),
            "current_restore_vs_original": _diff_stats(
                w2_restored_current, w2_canonical
            ),
            "exact_inverse_vs_original": _diff_stats(w2_restored_exact, w2_canonical),
            "hot_update_expected_vs_current_live": _diff_stats(
                current_updated_live_w2, expected_live_w2
            ),
            "hot_update_expected_vs_exact_inverse_control_live": _diff_stats(
                exact_updated_live_w2, expected_live_w2
            ),
            "w2_delta_stats": {
                "shape": list(w2_delta.shape),
                "max_abs": float(w2_delta.float().abs().max().item()),
                "mean_abs": float(w2_delta.float().abs().mean().item()),
            },
        },
    }


@app.local_entrypoint()
def main() -> None:
    upload = _upload_adapter_assets_to_volume()
    with modal.enable_output():
        result = probe_flashinfer_trtllm_layout.remote()
    print(json.dumps({"upload": upload, "probe": result}, indent=2, sort_keys=True))
