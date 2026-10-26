from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

# Every dtype in our converted tier-1 checkpoints (NVFP4 and FP8).
TORCH_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "U8": torch.uint8,
}


@dataclass(frozen=True)
class CanonicalTensor:
    name: str
    dtype: str  # safetensors dtype name, e.g. "BF16", "U8", "F8_E4M3"
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Place:
    """Element i of the canonical region `src` (row-major) is element i of `dst`.
    A wider `dst` dtype means a cast, so the tensor must arrive raw."""

    src: tuple[slice, ...]
    dst: torch.Tensor


@dataclass(frozen=True)
class Copy:
    """A serving tensor holding another serving tensor's values, refreshed after apply."""

    src: torch.Tensor
    dst: torch.Tensor


class OtherGpu:
    pass


@dataclass(frozen=True)
class Ignored:
    reason: str


OTHER_GPU = OtherGpu()
Entry = tuple[Place, ...] | OtherGpu | Ignored


@dataclass
class DeltaMap:
    inventory: dict[str, CanonicalTensor]
    entries: dict[str, Entry]
    copies: list[Copy]

    def check_manifest(self, manifest) -> None:
        """Every delta tensor is one the map covers, with the checkpoint's dtype and shape;
        a tensor whose places cast must arrive raw. Tensors left out are not touched."""
        for name, entry in manifest.tensors.items():
            tensor = self.inventory.get(name)
            if tensor is None:
                raise ValueError(f"{name} is not in the loaded checkpoint")
            if entry["dtype"] != tensor.dtype or tuple(entry["shape"]) != tensor.shape:
                raise ValueError(
                    f"{name}: {entry['dtype']}{entry['shape']} in the delta, "
                    f"{tensor.dtype}{list(tensor.shape)} loaded"
                )
            places = self.entries[name]
            if isinstance(places, Ignored):
                raise ValueError(
                    f"{name} is in the delta but not applied: {places.reason}"
                )
            casts = isinstance(places, tuple) and any(
                p.dst.dtype != TORCH_DTYPES[tensor.dtype] for p in places
            )
            if casts and entry["encoding"] != "raw_bytes":
                raise ValueError(
                    f"{name} is stored wider than the checkpoint, so it must arrive raw"
                )


def read_inventory(runner) -> dict[str, CanonicalTensor]:
    """Names, dtypes and shapes from the loaded checkpoint's safetensors headers."""
    from sglang.srt.model_loader.weight_utils import (
        filter_duplicate_safetensors_files,
        maybe_add_mtp_safetensors,
    )

    folder = Path(runner.model_config.model_path)
    if not folder.is_dir():
        raise ValueError(f"GPU delta needs a local checkpoint directory: {folder}")
    index = "model.safetensors.index.json"
    files = filter_duplicate_safetensors_files(
        [str(path) for path in folder.glob("*.safetensors")], str(folder), index
    )
    files = maybe_add_mtp_safetensors(
        files, str(folder), index, runner.model_config.hf_config
    )
    inventory: dict[str, CanonicalTensor] = {}
    for path in sorted(files):
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in source.keys():
                if name in inventory:
                    raise ValueError(f"tensor {name} appears in two checkpoint shards")
                view = source.get_slice(name)
                if view.get_dtype() not in TORCH_DTYPES:
                    raise ValueError(f"{name}: unsupported dtype {view.get_dtype()}")
                inventory[name] = CanonicalTensor(
                    name, view.get_dtype(), tuple(view.get_shape())
                )
    if not inventory:
        raise ValueError(f"no checkpoint tensors found in {folder}")
    return inventory


def model_rule(model):
    """The rule for the target's model class; the draft is part of the same family."""
    from sglang.srt.weight_sync.stitch_delta.rules.deepseek_mla import DeepSeekMla

    architecture = model.config.architectures[0]
    for rule in (DeepSeekMla,):
        if architecture in rule.architectures:
            return rule(model.config)
    raise ValueError(f"GPU delta has no rules for {architecture}")


def build_map(runners, inventory: dict[str, CanonicalTensor]) -> DeltaMap:
    from sglang.srt.weight_sync.stitch_delta.rules import placement

    models = {role: runner.model for role, runner in runners}
    rule = model_rule(models["target"])
    # Every path, so aliases (e.g. attn_mha.kv_b_proj) don't hide the loader's name.
    modules = {
        role: dict(model.named_modules(remove_duplicate=False))
        for role, model in models.items()
    }
    params = {
        role: dict(model.named_parameters(remove_duplicate=False))
        for role, model in models.items()
    }
    entries: dict[str, Entry] = {}
    for name, tensor in inventory.items():
        target = rule.target(name)
        if isinstance(target, Ignored):
            entries[name] = target
        elif target.expert is not None:
            layer = modules[target.runner][target.module]
            entries[name] = placement.expert_places(layer, target, tensor)
        else:
            param = params[target.runner].get(target.param)
            if param is None:
                raise ValueError(f"{name}: no serving parameter {target.param}")
            module = modules[target.runner][target.param.rsplit(".", 1)[0]]
            entries[name] = placement.linear_places(
                module, param, target, tensor, inventory
            )
    copies = [
        copy
        for model in models.values()
        for module in model.modules()
        for copy in (*rule.copies(module), *placement.expert_copies(module))
    ]
    return DeltaMap(inventory, entries, copies)
