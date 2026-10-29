"""Where each checkpoint tensor lives in this GPU's serving tensors, built once at startup."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_lora,
    get_model,
    get_parallel,
)

logger = logging.getLogger(__name__)

# Checkpoint dtypes the rules handle (NVFP4 and block FP8).
TORCH_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "U8": torch.uint8,
}


@dataclass(frozen=True)
class CanonicalTensor:
    name: str
    dtype: str  # safetensors dtype name, e.g. "BF16", "U8", "F8_E4M3"
    shape: tuple[int, ...]
    file: Path


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


@dataclass(frozen=True)
class Derived:
    """Serving tensors SGLang computes from checkpoint tensors at load, recomputed after apply
    with SGLang's own function: `values()` returns (value, destination) pairs."""

    values: Callable[[], list[tuple[torch.Tensor, torch.Tensor]]]


@dataclass(frozen=True)
class Destination:
    """Where a model rule sends a checkpoint tensor: a parameter (or one shard of a fused
    one), or one projection of a routed expert."""

    role: str
    param: str = ""
    shards: tuple[str, ...] = ()
    shard: int = 0
    module: str = ""
    expert: int | None = None
    projection: str = ""
    kind: str = ""
    # Applied to a GPU copy of the checkpoint tensor (DeltaMap.shadows), from which
    # a Derived recomputes what SGLang serves.
    shadow: bool = False


class OtherGpu:
    pass


@dataclass(frozen=True)
class Ignored:
    reason: str


OTHER_GPU = OtherGpu()
Entry = tuple[Place, ...] | OtherGpu | Ignored


@dataclass(frozen=True)
class MapTable:
    """The inventory as arrays sorted by name, so a version's tensors join it in one search."""

    names: np.ndarray  # fixed-width bytes
    dtypes: np.ndarray  # safetensors dtype names
    shape: np.ndarray
    shape_end: np.ndarray
    nbytes: np.ndarray
    held: np.ndarray  # this GPU applies it
    ignored: np.ndarray
    raw_only: np.ndarray  # a place is wider than the checkpoint dtype: must arrive raw
    entries: list[Entry]


@dataclass
class DeltaMap:
    inventory: dict[str, CanonicalTensor]
    entries: dict[str, Entry]
    copies: list[Copy]
    table: MapTable
    derived: list[Derived] = field(default_factory=list)
    # Checkpoint tensors the loader reads but does not keep: a GPU copy the deltas XOR into,
    # by (role, serving path), from which a Derived recomputes what SGLang serves.
    shadows: dict[tuple[str, str], torch.Tensor] = field(default_factory=dict)


def load_map(runners) -> DeltaMap:
    roles = [role for role, _ in runners]
    if roles.count("target") != 1:
        raise ValueError(f"expected one target runner, got {roles}")
    started = time.perf_counter()
    _check_scope(runners)
    inventory = _read_inventory(dict(runners)["target"])
    headers_read = time.perf_counter()
    delta_map = _build_map(runners, inventory)
    logger.info(
        f"Weight delta map built for {len(inventory)} checkpoint tensors: "
        f"headers {headers_read - started:.2f} s, map {time.perf_counter() - headers_read:.2f} s"
    )
    return delta_map


def _check_scope(runners) -> None:
    from sglang.srt.model_executor.model_runner_components.weight_updater import (
        _unsupported_derived_weight_cache_error,
    )

    parallel, moe = get_parallel(), get_exec().moe
    target = dict(runners)["target"]
    refused = {
        "multi-node engines": parallel.nnodes > 1,
        "data-parallel replicas": parallel.dp_size > 1,
        "pipeline parallelism": parallel.pp_size > 1,
        "context parallelism": parallel.attn_cp_size > 1 or parallel.dcp_size > 1,
        "attention TP above 1 (DP attention only, for now)": parallel.attn_tp_size > 1,
        "PD disaggregation": get_disagg().disaggregation_mode != "null",
        "LoRA": bool(get_lora().enable_lora),
        "the memory saver (it moves weights)": get_exec().features.enable_memory_saver,
        "EPLB or redundant experts": moe.enable_eplb
        or moe.ep_num_redundant_experts > 0,
        "DP-attention local control broadcast": (
            parallel.enable_dp_attention_local_control_broadcast
        ),
        "the zero-copy weight cache": get_model().weight_cache_mode != "off",
        "MoE gate/up rows permuted in place (SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED)": (
            envs.SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED.get()
        ),
        "a draft model from another checkpoint": any(
            runner.model_config.model_path != target.model_config.model_path
            for role, runner in runners
            if role != "target"
        ),
    }
    problems = [name for name, hit in refused.items() if hit]
    problems += filter(
        None, (_unsupported_derived_weight_cache_error(r.model) for _, r in runners)
    )
    if problems:
        raise ValueError(
            "--enable-weight-delta does not support " + "; ".join(problems)
        )


def _read_inventory(runner) -> dict[str, CanonicalTensor]:
    """Names, dtypes and shapes from the loaded checkpoint's safetensors headers."""
    from sglang.srt.model_loader.weight_utils import (
        filter_duplicate_safetensors_files,
        maybe_add_mtp_safetensors,
    )

    folder = Path(runner.model_config.model_path)
    if not folder.is_dir():
        raise ValueError(f"Weight delta needs a local checkpoint directory: {folder}")
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
                    name, view.get_dtype(), tuple(view.get_shape()), Path(path)
                )
    if not inventory:
        raise ValueError(f"no checkpoint tensors found in {folder}")
    return inventory


def _build_map(runners, inventory: dict[str, CanonicalTensor]) -> DeltaMap:
    from sglang.srt.weight_sync.weight_delta.rules import placement

    models = {role: runner.model for role, runner in runners}
    rule = _model_rule(dict(runners)["target"], models, inventory)
    shadows: dict[tuple[str, str], torch.Tensor] = {}
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
        destination = rule.destination(name)
        if isinstance(destination, Ignored):
            entries[name] = destination
        elif destination.shadow:
            device = next(models[destination.role].parameters()).device
            shadow = _read_checkpoint(tensor).to(device)
            shadows[(destination.role, destination.param)] = shadow
            entries[name] = (Place((slice(None),) * shadow.dim(), shadow),)
        elif destination.expert is not None:
            layer = modules[destination.role][destination.module]
            entries[name] = placement.expert_places(layer, destination, tensor)
        else:
            param = params[destination.role].get(destination.param)
            if param is None:
                raise ValueError(f"{name}: no serving parameter {destination.param}")
            module = modules[destination.role][destination.param.rsplit(".", 1)[0]]
            entries[name] = placement.linear_places(
                module, param, destination, tensor, inventory
            )
    copies = [
        copy
        for model in models.values()
        for module in model.modules()
        for copy in (*rule.copies(module), *placement.expert_copies(module))
    ]
    derived = [
        item
        for role, model in models.items()
        for path, module in model.named_modules()
        for item in (
            *rule.derived(role, path, module, shadows),
            *placement.layer_derived(module),
        )
    ]
    return DeltaMap(
        inventory, entries, copies, _table(inventory, entries), derived, shadows
    )


def _table(
    inventory: dict[str, CanonicalTensor], entries: dict[str, Entry]
) -> MapTable:
    names = sorted(inventory)
    tensors = [inventory[name] for name in names]
    rows = [entries[name] for name in names]
    return MapTable(
        names=np.array([name.encode() for name in names]),
        dtypes=np.array([tensor.dtype for tensor in tensors]),
        shape=np.array([dim for tensor in tensors for dim in tensor.shape], np.int64),
        shape_end=np.cumsum([len(tensor.shape) for tensor in tensors], dtype=np.int64),
        nbytes=np.array(
            [
                math.prod(tensor.shape) * TORCH_DTYPES[tensor.dtype].itemsize
                for tensor in tensors
            ],
            np.int64,
        ),
        held=np.array([isinstance(entry, tuple) for entry in rows]),
        ignored=np.array([isinstance(entry, Ignored) for entry in rows]),
        raw_only=np.array(
            [
                isinstance(entry, tuple)
                and any(
                    place.dst.dtype != TORCH_DTYPES[tensor.dtype] for place in entry
                )
                for tensor, entry in zip(tensors, rows)
            ]
        ),
        entries=rows,
    )


def _model_rule(runner, models, inventory):
    from sglang.srt.weight_sync.weight_delta.rules.deepseek_mla import DeepSeekMla
    from sglang.srt.weight_sync.weight_delta.rules.qwen3_5 import Qwen3_5

    # The checkpoint's config: a multimodal model's own config may be its text part's.
    architecture = runner.model_config.hf_config.architectures[0]
    for rule in (DeepSeekMla, Qwen3_5):
        if architecture in rule.architectures:
            return rule(models, inventory)
    raise ValueError(f"Weight delta has no rules for {architecture}")


def _read_checkpoint(tensor: CanonicalTensor) -> torch.Tensor:
    with safe_open(str(tensor.file), framework="pt") as source:
        return source.get_tensor(tensor.name)
