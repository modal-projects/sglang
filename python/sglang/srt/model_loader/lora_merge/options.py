import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class MergeMemoryBudget:
    peak_bytes: int
    bucket_bytes: int


@dataclass(frozen=True)
class LoraMergeOptions:
    manifest: Dict[str, Any]
    scaling: float
    strict: bool
    memory_budget: MergeMemoryBudget
    apply_bucket_bytes: int
    consume_prestaged: bool
    prestage_request_id: Optional[str]
    empty_cache_after_merge: bool
    added_tokens_config: Any = None


def manifest_bool(manifest: Optional[Dict[str, Any]], key: str) -> bool:
    if not manifest:
        return False
    value = manifest.get(key)
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def prestage_request_id(manifest: Dict[str, Any]) -> str:
    request_id = (
        manifest.get("lora_merge_prestage_request_id")
        or manifest.get("prestage_request_id")
    )
    if request_id is None:
        raise ValueError("LoRA merge prestage requires a request_id.")
    return str(request_id)


def parse_bytes_value(raw: Any) -> int:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    value = str(raw).strip().lower()
    multipliers = {
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
    }
    for suffix, multiplier in multipliers.items():
        if value.endswith(suffix):
            return int(float(value[: -len(suffix)]) * multiplier)
    return int(float(value))


def resolve_lora_merge_options(
    manifest: Optional[Dict[str, Any]],
    *,
    device: torch.device,
    include_scaling: bool,
) -> LoraMergeOptions:
    manifest = dict(manifest or {})
    memory_budget = resolve_merge_memory_budget(manifest, device)
    apply_bucket_bytes = resolve_apply_bucket_bytes(manifest, memory_budget)
    return LoraMergeOptions(
        manifest=manifest,
        scaling=resolve_scaling(manifest) if include_scaling else 1.0,
        strict=bool(manifest.get("strict", True)),
        memory_budget=memory_budget,
        apply_bucket_bytes=apply_bucket_bytes,
        consume_prestaged=manifest_bool(manifest, "lora_merge_consume_prestaged"),
        prestage_request_id=(
            prestage_request_id(manifest)
            if (
                manifest_bool(manifest, "lora_merge_consume_prestaged")
                or "lora_merge_prestage_request_id" in manifest
                or "prestage_request_id" in manifest
            )
            else None
        ),
        empty_cache_after_merge=_env_bool("SGLANG_LORA_MERGE_EMPTY_CACHE", True),
        added_tokens_config=manifest.get("added_tokens_config"),
    )


def resolve_merge_memory_budget(
    manifest: Dict[str, Any], device: torch.device
) -> MergeMemoryBudget:
    default = 512 * 1024**2
    raw_budget, source = _manifest_first(
        manifest,
        (
            "peak_device_bytes",
            "lora_merge_peak_device_bytes",
            "gpu_peak_device_bytes",
            "lora_merge_gpu_peak_device_bytes",
            "gpu_bucket_bytes",
            "lora_merge_gpu_bucket_bytes",
        ),
    )
    if raw_budget is not None:
        budget = parse_bytes_value(raw_budget)
    elif os.environ.get("SGLANG_LORA_MERGE_PEAK_DEVICE_BYTES") is not None:
        budget = _parse_bytes_env("SGLANG_LORA_MERGE_PEAK_DEVICE_BYTES", default)
        source = "env:SGLANG_LORA_MERGE_PEAK_DEVICE_BYTES"
    elif os.environ.get("SGLANG_LORA_MERGE_GPU_BUCKET_BYTES") is not None:
        budget = _parse_bytes_env("SGLANG_LORA_MERGE_GPU_BUCKET_BYTES", default)
        source = "env:SGLANG_LORA_MERGE_GPU_BUCKET_BYTES"
    else:
        budget = default
        source = "default"

    headroom_gb = _float_env("SGLANG_LORA_MERGE_VRAM_HEADROOM_GB", 8.0)
    if "vram_headroom_gb" in manifest:
        headroom_gb = float(manifest["vram_headroom_gb"])
    elif "lora_merge_vram_headroom_gb" in manifest:
        headroom_gb = float(manifest["lora_merge_vram_headroom_gb"])

    if device.type == "cuda" and torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info(device)
        available = int(free_bytes - headroom_gb * 1024**3)
        if available > 0:
            budget = min(budget, available)

    explicit_budget = source != "default"
    min_budget = 1024**2 if explicit_budget else 64 * 1024**2
    peak_bytes = max(min_budget, int(budget))
    return MergeMemoryBudget(
        peak_bytes=peak_bytes,
        bucket_bytes=peak_bytes,
    )


def resolve_apply_bucket_bytes(
    manifest: Dict[str, Any], budget: MergeMemoryBudget
) -> int:
    raw_bucket, _ = _manifest_first(
        manifest, ("apply_bucket_bytes", "lora_merge_apply_bucket_bytes")
    )
    if raw_bucket is not None:
        return max(1, min(budget.peak_bytes, parse_bytes_value(raw_bucket)))

    if os.environ.get("SGLANG_LORA_MERGE_APPLY_BUCKET_BYTES") is not None:
        return max(
            1,
            min(
                budget.peak_bytes,
                _parse_bytes_env(
                    "SGLANG_LORA_MERGE_APPLY_BUCKET_BYTES", budget.peak_bytes
                ),
            ),
        )

    fraction = _float_env("SGLANG_LORA_MERGE_APPLY_BUDGET_FRACTION", 0.5)
    if "apply_budget_fraction" in manifest:
        fraction = float(manifest["apply_budget_fraction"])
    elif "lora_merge_apply_budget_fraction" in manifest:
        fraction = float(manifest["lora_merge_apply_budget_fraction"])
    fraction = min(1.0, max(0.0, fraction))
    return max(1, int(budget.peak_bytes * fraction))


def resolve_scaling(manifest: Dict[str, Any]) -> float:
    if "scaling" in manifest:
        return float(manifest["scaling"])

    config = manifest.get("config_dict") or manifest.get("adapter_config")
    if config is None:
        raise ValueError(
            "Merged LoRA update requires manifest['config_dict'] or manifest['scaling']."
        )

    lora_alpha = float(config["lora_alpha"])
    rank = int(config["r"])
    if rank <= 0:
        raise ValueError(f"Invalid LoRA rank: {rank}")
    return lora_alpha / rank


def _manifest_first(manifest: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[Any, str]:
    for key in keys:
        if key in manifest:
            return manifest[key], f"manifest:{key}"
    return None, ""


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in ("0", "false", "no", "off")


def _parse_bytes_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return parse_bytes_value(raw)
