"""Prepared host runtime images for short-pause dense weight commits.

The ordinary disk loader is the correctness fallback.  This module implements
the final commit primitive for a faster path: all checkpoint parsing, sharding,
fusion and quantization must already have produced a byte-exact host image of
the model's live CUDA storages before :meth:`commit` is called.

The implementation is intentionally dense-update safe. It inventories runtime
storage rather than changed tensor names, copies every byte in that inventory,
and preserves aliases by overwriting existing storage instead of rebinding
Parameters. The legacy model-specific preparer excludes checkpoint-invariant
CUDA state; architecture-neutral grouping includes every discovered CUDA
storage so it does not need such a classification. Bounded scratch modules and
staging buffers keep the design within hosts that cannot hold a second model in
HBM.
"""

from __future__ import annotations

import copy
import concurrent.futures
import ctypes
import gc
import itertools
import json
import logging
import os
import re
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

logger = logging.getLogger(__name__)

_GIB = 1 << 30
_MIB = 1 << 20
_ALIGNMENT = 4096
_DECODER_LAYER = re.compile(r"^(language_model\.model\.layers\.\d+)(?:\.|$)")
_VISION_BLOCK = re.compile(r"^(vision_tower\.encoder\.blocks\.\d+)(?:\.|$)")
_LOGICAL_STORAGE_RANGE_ATTR = "_sglang_prepared_logical_storage_range"


@dataclass(frozen=True)
class RuntimeStorageSegment:
    """One unique live CUDA storage and its range in a host runtime image."""

    name: str
    image_offset: int
    nbytes: int
    device_bytes: torch.Tensor


@dataclass
class RuntimeStateImage:
    """A byte-exact pageable host image plus identity metadata."""

    bytes: torch.Tensor
    identity: str


@dataclass
class HostLoadTarget:
    """One registered scratch tensor temporarily assembled in pinned CPU RAM."""

    name: str
    parameter: torch.nn.Parameter
    host_data: torch.Tensor


@dataclass(frozen=True)
class RuntimeModuleGroup:
    """A bounded module subtree used as one scratch preparation unit."""

    path: str
    nbytes: int


class HostImageCapacityError(RuntimeError):
    """A restored checkpoint group does not fit the remaining host image."""


@dataclass
class HostImageArena:
    """Allocate aligned restored CPU storages from a shared host image."""

    bytes: torch.Tensor
    begin: int
    end: int
    cursor: int

    @classmethod
    def from_range(
        cls,
        image_bytes: torch.Tensor,
        begin: int,
        end: int,
    ) -> HostImageArena:
        if image_bytes.device.type != "cpu":
            raise ValueError("host image backing must be a CPU tensor")
        if begin < 0 or end < begin or end > image_bytes.numel():
            raise ValueError(
                f"invalid host image arena range [{begin}, {end}) for "
                f"{image_bytes.numel()} bytes"
            )
        return cls(bytes=image_bytes, begin=begin, end=end, cursor=begin)

    @property
    def used_bytes(self) -> int:
        return self.cursor - self.begin

    @property
    def capacity_bytes(self) -> int:
        return self.end - self.begin

    def allocate(self, source_bytes: torch.Tensor) -> torch.Tensor:
        begin = _align_up(self.cursor)
        end = begin + source_bytes.numel()
        if end > self.end:
            raise HostImageCapacityError(
                "restored checkpoint storage exceeds remaining runtime-image "
                f"capacity: requested={source_bytes.numel()} used={self.used_bytes} "
                f"capacity={self.capacity_bytes}"
            )
        self.cursor = end
        return self.bytes[begin:end]


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _storage_bytes(tensor: torch.Tensor) -> torch.Tensor:
    storage = tensor.untyped_storage()
    logical_range = getattr(tensor, _LOGICAL_STORAGE_RANGE_ATTR, None)
    if logical_range is None:
        byte_offset, nbytes = 0, storage.nbytes()
    else:
        byte_offset, nbytes = logical_range
        if byte_offset < 0 or nbytes < 0 or byte_offset + nbytes > storage.nbytes():
            raise ValueError(
                "invalid prepared logical storage range: "
                f"offset={byte_offset} nbytes={nbytes} storage={storage.nbytes()}"
            )
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        storage, byte_offset, (nbytes,), (1,)
    )


def _walk_attribute(value: Any, name: str, depth: int = 0):
    if isinstance(value, torch.Tensor):
        yield name, value
    elif depth < 2 and isinstance(value, dict):
        for child_name, child in value.items():
            yield from _walk_attribute(child, f"{name}[{child_name!r}]", depth + 1)
    elif depth < 2 and isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_attribute(child, f"{name}[{index}]", depth + 1)


def iter_model_tensors(model: torch.nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield registered and shallow unregistered model tensors.

    Quantized kernels sometimes keep derived weights as ordinary attributes.
    They are part of the observable runtime state even though state_dict omits
    them, so use the same shallow attribute policy as the inventory diagnostic.
    """

    yield from model.named_parameters(remove_duplicate=False)
    yield from model.named_buffers(remove_duplicate=False)
    reserved = {"_parameters", "_buffers", "_modules"}
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for attribute_name, value in vars(module).items():
            if attribute_name in reserved:
                continue
            yield from _walk_attribute(value, f"{prefix}{attribute_name}")


def runtime_module_path(name: str) -> str | None:
    """Map a live runtime tensor to the checkpoint module that produces it."""

    for pattern in (_DECODER_LAYER, _VISION_BLOCK):
        match = pattern.match(name)
        if match is not None:
            return match.group(1)
    if name.startswith("mm_projector."):
        return "mm_projector"
    for path in (
        "language_model.lm_head",
        "language_model.model.embed_tokens",
        "language_model.model.norm",
        "vision_tower.encoder.final_layernorm",
        "vision_tower.patch_embed",
    ):
        if name == path or name.startswith(f"{path}."):
            return path
    return None


def build_runtime_storage_plan(
    model: torch.nn.Module,
    *,
    include_all_cuda_tensors: bool = False,
) -> tuple[list[RuntimeStorageSegment], int]:
    """Build a deterministic checkpoint-derived plan without changing addresses."""

    unique: dict[tuple[int | None, int, int], tuple[str, torch.Tensor]] = {}
    for name, tensor in iter_model_tensors(model):
        if tensor.device.type != "cuda" or (
            not include_all_cuda_tensors and runtime_module_path(name) is None
        ):
            continue
        storage = tensor.untyped_storage()
        nbytes = storage.nbytes()
        key = (tensor.device.index, storage.data_ptr(), nbytes)
        current = unique.get(key)
        if current is None or name < current[0]:
            unique[key] = (name, tensor)

    offset = 0
    segments = []
    for name, tensor in sorted(unique.values(), key=lambda item: item[0]):
        offset = _align_up(offset)
        device_bytes = _storage_bytes(tensor)
        segments.append(
            RuntimeStorageSegment(
                name=name,
                image_offset=offset,
                nbytes=device_bytes.numel(),
                device_bytes=device_bytes,
            )
        )
        offset += device_bytes.numel()
    return segments, _align_up(offset)


def _tensor_storage_key(tensor: torch.Tensor) -> tuple[int | None, int, int]:
    storage_bytes = _storage_bytes(tensor)
    return tensor.device.index, storage_bytes.data_ptr(), storage_bytes.numel()


def _iter_direct_module_tensors(
    module: torch.nn.Module,
) -> Iterable[torch.Tensor]:
    """Yield tensors owned directly by one module, excluding child subtrees."""

    yield from (item for item in module._parameters.values() if item is not None)
    yield from (item for item in module._buffers.values() if item is not None)
    reserved = {"_parameters", "_buffers", "_modules"}
    for name, value in vars(module).items():
        if name in reserved:
            continue
        yield from (tensor for _, tensor in _walk_attribute(value, name))


def build_runtime_module_groups(
    model: torch.nn.Module,
    *,
    max_group_bytes: int,
    device_type: str = "cuda",
) -> list[RuntimeModuleGroup]:
    """Partition a model tree into bounded, storage-complete scratch groups.

    The partition is derived only from the loaded runtime structure. A subtree
    becomes a group when its unique CUDA storage fits the configured scratch
    budget; larger containers recurse into their children. This naturally
    selects transformer blocks for large models without recognizing model
    classes or layer-name patterns.

    A large container that directly owns CUDA tensors as well as child modules
    cannot be split safely by the current proxy cloner. Fail explicitly so the
    caller can retain the ordinary full-loader fallback.
    """

    if max_group_bytes <= 0:
        raise ValueError("runtime module group budget must be positive")

    modules = dict(model.named_modules())
    subtree_keys: dict[str, set[tuple[int | None, int, int]]] = {}
    direct_keys: dict[str, set[tuple[int | None, int, int]]] = {}
    storage_bytes: dict[tuple[int | None, int, int], int] = {}

    def collect(path: str, module: torch.nn.Module):
        direct: set[tuple[int | None, int, int]] = set()
        for tensor in _iter_direct_module_tensors(module):
            if tensor.device.type != device_type:
                continue
            key = _tensor_storage_key(tensor)
            direct.add(key)
            storage_bytes[key] = key[2]
        direct_keys[path] = direct
        subtree = set(direct)
        prefix = f"{path}." if path else ""
        for child_name, child in module.named_children():
            child_path = f"{prefix}{child_name}"
            subtree.update(collect(child_path, child))
        subtree_keys[path] = subtree
        return subtree

    collect("", model)
    groups: list[RuntimeModuleGroup] = []

    def visit(path: str):
        module = modules[path]
        keys = subtree_keys[path]
        if not keys:
            return
        nbytes = sum(storage_bytes[key] for key in keys)
        children = [
            f"{path}.{name}" if path else name
            for name, child in module.named_children()
            if subtree_keys.get(f"{path}.{name}" if path else name)
        ]
        if path and (nbytes <= max_group_bytes or not children):
            groups.append(RuntimeModuleGroup(path=path, nbytes=nbytes))
            return
        if direct_keys[path]:
            raise ValueError(
                "cannot bound prepared reload module with direct CUDA tensors "
                f"and child subtrees: path={path or '<root>'!r} "
                f"bytes={nbytes} budget={max_group_bytes}"
            )
        for child_path in children:
            visit(child_path)

    visit("")
    if not groups:
        raise ValueError("loaded model has no bounded CUDA runtime groups")
    return groups


def _checkpoint_name_candidates(
    model: torch.nn.Module,
    name: str,
    root_prefixes: set[str],
) -> list[str]:
    """Return architecture-neutral name candidates for module attribution."""

    candidates = [name]
    mapper = getattr(model, "hf_to_sglang_mapper", None)
    if mapper is not None:
        try:
            mapped_name = mapper._map_name(name)
            if mapped_name is None:
                return []
            candidates.append(mapped_name)
        except Exception:
            logger.debug(
                "checkpoint name mapper could not map %s for prepared grouping",
                name,
                exc_info=True,
            )
    for candidate in list(candidates):
        first = candidate.split(".", 1)[0]
        for prefix in root_prefixes:
            if first != prefix:
                candidates.append(f"{prefix}.{candidate}")
    return list(dict.fromkeys(candidates))


def map_checkpoint_names_to_runtime_groups(
    model: torch.nn.Module,
    names: Iterable[str],
    groups: list[RuntimeModuleGroup],
) -> dict[str, str | None]:
    """Map every checkpoint tensor to a bounded live module subtree."""

    paths = sorted((group.path for group in groups), key=len, reverse=True)
    root_prefixes = {path.split(".", 1)[0] for path in paths}
    result: dict[str, str | None] = {}
    for name in names:
        candidates = _checkpoint_name_candidates(
            model,
            name,
            root_prefixes,
        )
        if not candidates:
            result[name] = None
            continue
        matches = {
            path
            for candidate in candidates
            for path in paths
            if candidate == path or candidate.startswith(f"{path}.")
        }
        if not matches:
            raise ValueError(
                "checkpoint tensor cannot be attributed to a bounded runtime "
                f"module group: {name!r}"
            )
        # Groups form a disjoint tree frontier, so multiple matches indicate a
        # bad partition rather than an ambiguity to guess through.
        if len(matches) != 1:
            raise ValueError(
                f"checkpoint tensor maps to multiple runtime groups: "
                f"{name!r} -> {sorted(matches)}"
            )
        result[name] = matches.pop()
    return result


def runtime_group_image_ranges(
    segments: list[RuntimeStorageSegment],
    groups: list[RuntimeModuleGroup],
) -> dict[str, tuple[int, int]]:
    """Return the aligned final-image span owned by every runtime group."""

    spans: dict[str, list[int]] = {}
    paths = [group.path for group in groups]
    for segment in segments:
        matches = [
            path
            for path in paths
            if segment.name == path or segment.name.startswith(f"{path}.")
        ]
        if len(matches) != 1:
            raise ValueError(
                "runtime storage is not covered by exactly one module group: "
                f"{segment.name!r} -> {matches}"
            )
        span = spans.setdefault(matches[0], [segment.image_offset, segment.image_offset])
        span[0] = min(span[0], segment.image_offset)
        span[1] = max(span[1], segment.image_offset + segment.nbytes)

    missing = [path for path in paths if path not in spans]
    if missing:
        raise ValueError(f"runtime module groups have no image storage: {missing}")
    ranges = {path: (span[0], span[1]) for path, span in spans.items()}
    ordered = sorted((begin, end, path) for path, (begin, end) in ranges.items())
    for (_, previous_end, previous_path), (begin, _, path) in zip(
        ordered,
        ordered[1:],
    ):
        if begin < previous_end:
            raise ValueError(
                "runtime module image spans overlap: "
                f"{previous_path!r} and {path!r}"
            )
    return ranges


def runtime_group_finalization_order(
    groups: list[RuntimeModuleGroup],
    segments: list[RuntimeStorageSegment],
    raw_image_ranges: dict[str, tuple[int, int]],
) -> list[str]:
    """Order finalization so final bytes never overwrite unconsumed raw bytes.

    Restored checkpoint layouts and finalized kernel layouts can have different
    per-group sizes even when their full-model totals fit the same host image.
    Raw groups are packed in final-image order. A group can be finalized once
    its final span overlaps no other remaining raw group.
    """

    final_ranges = runtime_group_image_ranges(segments, groups)
    ordered_paths = sorted(final_ranges, key=lambda path: final_ranges[path][0])
    remaining = set(ordered_paths)
    result: list[str] = []

    def overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
        return left[0] < right[1] and right[0] < left[1]

    while remaining:
        ready = [
            path
            for path in ordered_paths
            if path in remaining
            and not any(
                other != path
                and other in raw_image_ranges
                and overlaps(final_ranges[path], raw_image_ranges[other])
                for other in remaining
            )
        ]
        if not ready:
            dependencies = {
                path: sorted(
                    other
                    for other in remaining
                    if other != path
                    and other in raw_image_ranges
                    and overlaps(final_ranges[path], raw_image_ranges[other])
                )
                for path in sorted(remaining)
            }
            raise RuntimeError(
                "restored and finalized runtime-image layouts form an unsafe "
                f"overwrite cycle: {dependencies}"
            )
        for path in ready:
            remaining.remove(path)
            result.append(path)
    return result


def checkpoint_module_path(name: str) -> str:
    """Return a bounded scratch-module path for a checkpoint tensor.

    Kimi checkpoints are physically ordered by these groups, which lets the
    preparer consume every source tensor exactly once while keeping only one
    decoder/vision block in spare HBM.
    """

    path = runtime_module_path(name)
    if path is None:
        raise ValueError(f"no bounded scratch module for checkpoint tensor {name!r}")
    return path


def ordered_mmap_weights_iterator(
    model_path: str,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Read a sharded safetensors checkpoint in tensor-name order.

    The normal CPU safetensors iterator walks one file at a time.  Large layers
    can span checkpoint shards, so that order cannot satisfy the one-scratch-
    module bound used by prepared reloads.  The index gives us a cheap global
    ordering while every rank maps the same page-cache-backed files.  Weight
    loaders then fault only the TP slices they actually consume instead of
    forcing an O_DIRECT GPU copy of every source tensor.
    """

    from safetensors import safe_open

    index_path = Path(model_path) / "model.safetensors.index.json"
    with index_path.open() as file:
        index = json.load(file)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid or empty safetensors weight map: {index_path}")

    filenames = sorted(set(weight_map.values()))
    with ExitStack() as stack:
        handles = {
            filename: stack.enter_context(
                safe_open(
                    Path(model_path) / filename,
                    framework="pt",
                    device="cpu",
                )
            )
            for filename in filenames
        }
        for name in sorted(weight_map):
            yield name, handles[weight_map[name]].get_tensor(name)


def streaming_mmap_weights_iterator(
    model_path: str,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Read a sharded safetensors checkpoint once in physical file order."""

    from safetensors import safe_open

    index_path = Path(model_path) / "model.safetensors.index.json"
    with index_path.open() as file:
        index = json.load(file)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid or empty safetensors weight map: {index_path}")

    for filename in sorted(set(weight_map.values())):
        with safe_open(
            Path(model_path) / filename,
            framework="pt",
            device="cpu",
        ) as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name)


def grouped_mmap_weights_iterator(
    model_path: str,
    model: torch.nn.Module,
    groups: list[RuntimeModuleGroup],
) -> Iterable[tuple[str, Iterable[tuple[str, torch.Tensor]]]]:
    """Yield mmap checkpoint tensors grouped by bounded runtime subtree."""

    from safetensors import safe_open

    index_path = Path(model_path) / "model.safetensors.index.json"
    with index_path.open() as file:
        index = json.load(file)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid or empty safetensors weight map: {index_path}")

    group_for_name = map_checkpoint_names_to_runtime_groups(
        model,
        weight_map,
        groups,
    )
    names_by_group: dict[str, list[str]] = {}
    for name in sorted(weight_map):
        path = group_for_name[name]
        if path is not None:
            names_by_group.setdefault(path, []).append(name)

    filenames = sorted(set(weight_map.values()))
    with ExitStack() as stack:
        handles = {
            filename: stack.enter_context(
                safe_open(
                    Path(model_path) / filename,
                    framework="pt",
                    device="cpu",
                )
            )
            for filename in filenames
        }
        # Yield the complete runtime frontier, including groups with no direct
        # checkpoint tensor. Such groups can contain tied or synthesized state
        # that must be cloned and postprocessed into the dense runtime image.
        for group in groups:
            path = group.path
            names = names_by_group.get(path, ())
            yield (
                path,
                ((name, handles[weight_map[name]].get_tensor(name)) for name in names),
            )


def _clone_tensor_storage(
    tensor: torch.Tensor,
    tensor_memo: dict[int, torch.Tensor],
    storage_memo: dict[tuple[int | None, int, int], torch.Tensor],
    *,
    target_device: torch.device | None = None,
    pin_memory: bool = False,
    storage_allocator: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Clone a tensor while retaining subclasses, views, and storage aliases."""

    cached = tensor_memo.get(id(tensor))
    if cached is not None:
        return cached

    source_bytes = _storage_bytes(tensor)
    storage_key = (
        tensor.device.index,
        source_bytes.data_ptr(),
        source_bytes.numel(),
    )
    cloned_storage = storage_memo.get(storage_key)
    if cloned_storage is None:
        if storage_allocator is not None:
            cloned_storage = storage_allocator(source_bytes)
            if cloned_storage.dtype != torch.uint8:
                raise TypeError("storage allocator must return a uint8 tensor")
            if cloned_storage.numel() != source_bytes.numel():
                raise ValueError(
                    "storage allocator returned the wrong size: "
                    f"{cloned_storage.numel()} != {source_bytes.numel()}"
                )
        elif target_device is None:
            cloned_storage = source_bytes.clone()
        else:
            cloned_storage = torch.empty(
                source_bytes.numel(),
                dtype=torch.uint8,
                device=target_device,
                pin_memory=pin_memory,
            )
        if storage_allocator is not None:
            destination_device = cloned_storage.device
        else:
            destination_device = (
                tensor.device if target_device is None else target_device
            )
        if storage_allocator is not None or target_device is not None:
            non_blocking = (
                source_bytes.device.type == "cuda" and cloned_storage.is_pinned()
            ) or (
                source_bytes.is_pinned() and destination_device.type == "cuda"
            )
            cloned_storage.copy_(source_bytes, non_blocking=non_blocking)
        storage_memo[storage_key] = cloned_storage
    cloned_device = (
        cloned_storage.device
        if storage_allocator is not None
        else tensor.device if target_device is None else target_device
    )
    cloned_byte_offset = cloned_storage.storage_offset() * cloned_storage.element_size()
    source_byte_offset = source_bytes.storage_offset() * source_bytes.element_size()
    tensor_byte_offset = tensor.storage_offset() * tensor.element_size()
    relative_tensor_byte_offset = tensor_byte_offset - source_byte_offset
    if relative_tensor_byte_offset < 0:
        raise ValueError(
            "tensor view begins before its prepared logical storage range: "
            f"tensor_offset={tensor_byte_offset} storage_offset={source_byte_offset}"
        )
    view_byte_offset = cloned_byte_offset + relative_tensor_byte_offset
    if view_byte_offset % tensor.element_size():
        raise ValueError(
            "cloned storage is not aligned for tensor dtype: "
            f"offset={view_byte_offset} element_size={tensor.element_size()}"
        )
    view = torch.empty(0, dtype=tensor.dtype, device=cloned_device).set_(
        cloned_storage.untyped_storage(),
        view_byte_offset // tensor.element_size(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )
    if isinstance(tensor, torch.nn.Parameter):
        cloned = type(tensor)._make_subclass(type(tensor), view, tensor.requires_grad)
        if hasattr(tensor, "__dict__"):
            cloned.__dict__.update(tensor.__dict__)
    else:
        cloned = view.requires_grad_(tensor.requires_grad)
        if hasattr(tensor, "__dict__"):
            cloned.__dict__.update(tensor.__dict__)
    setattr(
        cloned,
        _LOGICAL_STORAGE_RANGE_ATTR,
        (cloned_byte_offset, cloned_storage.numel()),
    )
    tensor_memo[id(tensor)] = cloned
    return cloned


def _clone_attribute_tensors(
    value: Any,
    tensor_memo: dict[int, torch.Tensor],
    storage_memo: dict[tuple[int | None, int, int], torch.Tensor],
    *,
    target_device: torch.device | None = None,
    pin_memory: bool = False,
    storage_allocator: Callable[[torch.Tensor], torch.Tensor] | None = None,
    depth: int = 0,
) -> Any:
    """Clone tensor leaves but share immutable and non-tensor runtime objects."""

    if isinstance(value, torch.Tensor):
        return _clone_tensor_storage(
            value,
            tensor_memo,
            storage_memo,
            target_device=target_device,
            pin_memory=pin_memory,
            storage_allocator=storage_allocator,
        )
    if depth >= 2:
        return value
    if isinstance(value, dict):
        return {
            key: _clone_attribute_tensors(
                child,
                tensor_memo,
                storage_memo,
                target_device=target_device,
                pin_memory=pin_memory,
                storage_allocator=storage_allocator,
                depth=depth + 1,
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _clone_attribute_tensors(
                child,
                tensor_memo,
                storage_memo,
                target_device=target_device,
                pin_memory=pin_memory,
                storage_allocator=storage_allocator,
                depth=depth + 1,
            )
            for child in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _clone_attribute_tensors(
                child,
                tensor_memo,
                storage_memo,
                target_device=target_device,
                pin_memory=pin_memory,
                storage_allocator=storage_allocator,
                depth=depth + 1,
            )
            for child in value
        )
    return value


def clone_module_tensors(
    module: torch.nn.Module,
    *,
    target_device: torch.device | None = None,
    pin_memory: bool = False,
    storage_allocator: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.nn.Module:
    """Clone a module tree's tensor state without copying CUDA runtime objects."""

    if pin_memory and (target_device is None or target_device.type != "cpu"):
        raise ValueError("pin_memory requires an explicit CPU target device")
    if storage_allocator is not None and target_device is None:
        raise ValueError("storage_allocator requires an explicit target device")
    tensor_memo: dict[int, torch.Tensor] = {}
    storage_memo: dict[tuple[int | None, int, int], torch.Tensor] = {}

    def clone(current: torch.nn.Module) -> torch.nn.Module:
        result = copy.copy(current)
        result._parameters = {
            name: (
                None
                if parameter is None
                else _clone_tensor_storage(
                    parameter,
                    tensor_memo,
                    storage_memo,
                    target_device=target_device,
                    pin_memory=pin_memory,
                    storage_allocator=storage_allocator,
                )
            )
            for name, parameter in current._parameters.items()
        }
        result._buffers = {
            name: (
                None
                if buffer is None
                else _clone_tensor_storage(
                    buffer,
                    tensor_memo,
                    storage_memo,
                    target_device=target_device,
                    pin_memory=pin_memory,
                    storage_allocator=storage_allocator,
                )
            )
            for name, buffer in current._buffers.items()
        }
        result._modules = {
            name: None if child is None else clone(child)
            for name, child in current._modules.items()
        }
        for name, value in vars(current).items():
            if name in {"_parameters", "_buffers", "_modules"}:
                continue
            result.__dict__[name] = _clone_attribute_tensors(
                value,
                tensor_memo,
                storage_memo,
                target_device=target_device,
                pin_memory=pin_memory,
                storage_allocator=storage_allocator,
            )
        return result

    return clone(module)


def module_at_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Return one registered module subtree by dotted path."""

    current = model
    for part in path.split("."):
        child = current._modules.get(part)
        if child is None:
            raise KeyError(f"module path {path!r} is missing component {part!r}")
        current = child
    return current


def replace_proxy_module(
    proxy: torch.nn.Module,
    live_model: torch.nn.Module,
    path: str,
    replacement: torch.nn.Module,
) -> None:
    """Replace a proxy subtree while cloning each shared ancestor at most once."""

    proxy_cursor = proxy
    live_cursor = live_model
    parts = path.split(".")
    for part in parts[:-1]:
        live_child = live_cursor._modules.get(part)
        proxy_child = proxy_cursor._modules.get(part)
        if live_child is None or proxy_child is None:
            raise KeyError(f"module path {path!r} is missing component {part!r}")
        if proxy_child is live_child:
            proxy_child = copy.copy(live_child)
            proxy_child._modules = live_child._modules.copy()
            proxy_cursor._modules[part] = proxy_child
        proxy_cursor = proxy_child
        live_cursor = live_child
    proxy_cursor._modules[parts[-1]] = replacement


def release_module_tensors(module: torch.nn.Module) -> None:
    """Drop tensor references from a disposable cloned module tree."""

    def drop(value: Any, depth: int = 0) -> Any:
        if isinstance(value, torch.Tensor):
            return None
        if depth >= 2:
            return value
        if isinstance(value, dict):
            return {key: drop(child, depth + 1) for key, child in value.items()}
        if isinstance(value, list):
            return [drop(child, depth + 1) for child in value]
        if isinstance(value, tuple):
            return tuple(drop(child, depth + 1) for child in value)
        return value

    reserved = {"_parameters", "_buffers", "_modules"}
    for current in module.modules():
        current._parameters = dict.fromkeys(current._parameters)
        current._buffers = dict.fromkeys(current._buffers)
        for name, value in vars(current).items():
            if name not in reserved:
                current.__dict__[name] = drop(value)


def build_host_load_proxy(
    model: torch.nn.Module,
    groups: list[RuntimeModuleGroup],
    target_device: torch.device,
    restore_weights_before_loading,
    *,
    image_bytes: torch.Tensor | None = None,
    segments: list[RuntimeStorageSegment] | None = None,
) -> tuple[
    torch.nn.Module,
    dict[str, float | int],
    dict[str, tuple[int, int]],
]:
    """Clone a restored model frontier into pageable rank-local CPU memory.

    The resulting proxy owns every tensor in the runtime-derived group frontier
    but shares tensorless runtime objects. The model's ordinary ``load_weights``
    can therefore run once against CPU tensors, amortizing architecture-specific
    name mapping and expert dispatch across the entire checkpoint. When a host
    image is supplied, restored storages are packed directly into that full
    image in final-group order. A restored layout that exceeds the remaining
    full-image capacity falls back to bounded pageable storage for only that
    group.
    """

    if (image_bytes is None) != (segments is None):
        raise ValueError("image_bytes and segments must be supplied together")
    final_image_ranges = (
        runtime_group_image_ranges(segments, groups)
        if image_bytes is not None and segments is not None
        else {}
    )
    ordered_groups = (
        sorted(groups, key=lambda group: final_image_ranges[group.path][0])
        if final_image_ranges
        else groups
    )
    arena = (
        HostImageArena.from_range(image_bytes, 0, image_bytes.numel())
        if image_bytes is not None
        else None
    )
    proxy = copy.copy(model)
    proxy._modules = model._modules.copy()
    raw_image_ranges: dict[str, tuple[int, int]] = {}
    stats: dict[str, float | int] = {
        "groups": 0,
        "clone_s": 0.0,
        "restore_s": 0.0,
        "d2h_s": 0.0,
        "direct_image_groups": 0,
        "direct_image_capacity_bytes": 0,
        "direct_image_used_bytes": 0,
        "fallback_groups": 0,
    }
    for group in ordered_groups:
        phase_started = time.perf_counter()
        _, device_shadow = clone_module_proxy(model, group.path)
        stats["clone_s"] += time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        restore_weights_before_loading(device_shadow, target_device)
        stats["restore_s"] += time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        arena_cursor = arena.cursor if arena is not None else 0
        raw_begin = _align_up(arena_cursor)
        try:
            host_shadow = clone_module_tensors(
                device_shadow,
                target_device=torch.device("cpu"),
                storage_allocator=arena.allocate if arena is not None else None,
            )
            if target_device.type == "cuda":
                # Direct-image D2H copies can be asynchronous because the image
                # is pinned. The bounded CUDA scratch must remain live until all
                # copies from this group have completed.
                torch.cuda.synchronize(target_device)
        except HostImageCapacityError:
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            if arena is not None:
                # A partial group cannot remain in the packed layout. Reclaim
                # its bytes and keep only this bounded group in pageable RAM.
                arena.cursor = arena_cursor
            logger.warning(
                "[RL_PREPARED_STATE] restored group %s exceeds remaining full "
                "image capacity; using bounded pageable fallback",
                group.path,
            )
            host_shadow = clone_module_tensors(
                device_shadow,
                target_device=torch.device("cpu"),
            )
            stats["fallback_groups"] += 1
        else:
            if arena is not None:
                raw_image_ranges[group.path] = (raw_begin, arena.cursor)
                stats["direct_image_groups"] += 1
                stats["direct_image_used_bytes"] = arena.used_bytes
        stats["d2h_s"] += time.perf_counter() - phase_started
        replace_proxy_module(proxy, model, group.path, host_shadow)
        stats["groups"] += 1
        del device_shadow, host_shadow

    if arena is not None:
        stats["direct_image_capacity_bytes"] = arena.capacity_bytes
    return proxy, stats, raw_image_ranges


def clone_module_proxy(
    model: torch.nn.Module, path: str
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Shallow-clone ancestors and tensor-clone only the selected CUDA module."""

    parts = path.split(".")
    live = model
    proxy = copy.copy(model)
    proxy._modules = model._modules.copy()
    proxy_cursor = proxy
    for index, part in enumerate(parts):
        live_child = live._modules.get(part)
        if live_child is None:
            raise KeyError(f"module path {path!r} is missing component {part!r}")
        if index == len(parts) - 1:
            shadow = clone_module_tensors(live_child)
            proxy_cursor._modules[part] = shadow
            return proxy, shadow
        proxy_child = copy.copy(live_child)
        proxy_child._modules = live_child._modules.copy()
        proxy_cursor._modules[part] = proxy_child
        proxy_cursor = proxy_child
        live = live_child
    raise AssertionError("empty module path")


def move_load_targets_to_pinned_host(module: torch.nn.Module) -> list[HostLoadTarget]:
    """Move unique registered parameters to initialized pinned host tensors.

    Model ``weight_loader`` functions mostly slice and fuse checkpoint tensors
    into a much smaller set of rank-local Parameters.  Letting those functions
    write CPU mmap tensors directly into pinned CPU Parameters avoids issuing
    thousands of tiny pageable H2D copies.  The completed Parameters are copied
    to CUDA in a few large transfers before quantization post-processing.

    This is deliberately a storage transport primitive, not a model-specific
    load plan: the model's ordinary ``load_weights`` implementation still owns
    all name mapping, sharding, expert placement, and fusion semantics.
    """

    targets: list[HostLoadTarget] = []
    seen: set[int] = set()
    device_sources: list[torch.Tensor] = []
    for name, parameter in module.named_parameters(remove_duplicate=False):
        if id(parameter) in seen or parameter.device.type != "cuda":
            continue
        seen.add(id(parameter))
        host_data = torch.empty_strided(
            tuple(parameter.shape),
            tuple(parameter.stride()),
            dtype=parameter.dtype,
            device="cpu",
            pin_memory=True,
        )
        # A model may intentionally omit tied, synthesized, or checkpoint-
        # invariant Parameters from load_weights. Preserve their current value
        # so CPU assembly remains a full-state operation even when only a
        # subset of this scratch module is filled by checkpoint tensors.
        device_source = parameter.data
        host_data.copy_(device_source, non_blocking=True)
        device_sources.append(device_source)
        parameter.data = host_data
        targets.append(
            HostLoadTarget(
                name=name,
                parameter=parameter,
                host_data=host_data,
            )
        )
    if device_sources:
        torch.cuda.synchronize()
    return targets


def move_load_targets_to_device(
    targets: list[HostLoadTarget],
    target_device: torch.device,
    stream: torch.cuda.Stream,
) -> tuple[int, int]:
    """Move assembled Parameters back to CUDA with large nonblocking copies."""

    copies = 0
    copied_bytes = 0
    with torch.cuda.stream(stream):
        for target in targets:
            if target.parameter.device.type != "cpu":
                raise RuntimeError(
                    "CPU assembly changed a load target's device unexpectedly: "
                    f"{target.name} is on {target.parameter.device}"
                )
            device_data = torch.empty_strided(
                tuple(target.host_data.shape),
                tuple(target.host_data.stride()),
                dtype=target.host_data.dtype,
                device=target_device,
            )
            device_data.copy_(target.host_data, non_blocking=True)
            target.parameter.data = device_data
            copies += 1
            copied_bytes += target.host_data.untyped_storage().nbytes()
    stream.synchronize()
    return copies, copied_bytes


class PreparedRuntimeState:
    """Own active/prepared host images and commit prepared bytes in-place."""

    def __init__(self, model: torch.nn.Module):
        self._auto_groups = (
            os.environ.get("SGLANG_PREPARED_AUTO_MODULE_GROUPS", "0") == "1"
        )
        self._module_groups: list[RuntimeModuleGroup] = []
        if self._auto_groups:
            max_group_bytes = (
                int(os.environ.get("SGLANG_PREPARED_MAX_GROUP_GB", "8")) * _GIB
            )
            self._module_groups = build_runtime_module_groups(
                model,
                max_group_bytes=max_group_bytes,
            )
        self.segments, self.image_nbytes = build_runtime_storage_plan(
            model,
            include_all_cuda_tensors=self._auto_groups,
        )
        if self._auto_groups:
            runtime_group_image_ranges(self.segments, self._module_groups)
            logger.info(
                "[RL_PREPARED_STATE] auto module groups=%d max_bytes=%d",
                len(self._module_groups),
                max(group.nbytes for group in self._module_groups),
            )
        self.active: RuntimeStateImage | None = None
        self.prepared: RuntimeStateImage | None = None
        self._pinned_prefix: torch.Tensor | None = None
        self._tail_buffers: list[torch.Tensor] = []
        self._stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self._prefix_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self._gpu_stage_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self._tail_chunk_bytes = (
            int(os.environ.get("SGLANG_PREPARED_TAIL_CHUNK_MIB", "1024")) * _MIB
        )
        self._tail_buffer_count = int(
            os.environ.get("SGLANG_PREPARED_TAIL_BUFFER_COUNT", "8")
        )
        if self._tail_buffer_count < 2:
            raise ValueError("prepared reload needs at least two tail buffers")
        self._staged_image: RuntimeStateImage | None = None
        self._staged_tail: list[tuple[int, int, float] | None] = []
        self._gpu_stage: torch.Tensor | None = None
        self._gpu_stage_image_offset = 0
        self._checkpoint_device_buffer: torch.Tensor | None = None
        self._full_pinned = (
            os.environ.get("SGLANG_PREPARED_FULL_PINNED_IMAGE", "0") == "1"
        )
        self._preallocated_image_bytes: torch.Tensor | None = None

    @property
    def prepared_identity(self) -> str | None:
        return self.prepared.identity if self.prepared is not None else None

    def allocate_image(self, identity: str) -> RuntimeStateImage:
        if self._preallocated_image_bytes is not None:
            image_bytes = self._preallocated_image_bytes
            self._preallocated_image_bytes = None
        else:
            try:
                image_bytes = torch.empty(
                    self.image_nbytes,
                    dtype=torch.uint8,
                    pin_memory=self._full_pinned,
                )
            except RuntimeError:
                if not self._full_pinned:
                    raise
                logger.exception(
                    "[RL_PREPARED_STATE] full pinned image allocation failed; "
                    "falling back to bounded staging"
                )
                self._full_pinned = False
                image_bytes = torch.empty(self.image_nbytes, dtype=torch.uint8)
        return RuntimeStateImage(bytes=image_bytes, identity=identity)

    def capture_active(self, identity: str) -> dict[str, float | int]:
        """Snapshot the currently serving model; intended for background setup."""

        started = time.perf_counter()
        image = self.allocate_image(identity)
        for segment in self.segments:
            begin = segment.image_offset
            image.bytes[begin : begin + segment.nbytes].copy_(segment.device_bytes)
        torch.cuda.synchronize()
        self.active = image
        return {
            "bytes": self.image_nbytes,
            "storages": len(self.segments),
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def begin_preparation(self, identity: str) -> RuntimeStateImage:
        """Allocate a fresh dense image for all checkpoint-derived storages."""

        if self.prepared is None or self.prepared is self.active:
            self.prepared = self.allocate_image(identity)
        else:
            # After the second successful commit ``prepared`` is the previous
            # active image. It is no longer needed for serving and is exactly
            # the right-sized scratch buffer for the next target. Reuse it
            # instead of transiently allocating a third full image before the
            # old reference can be released.
            self.prepared.identity = identity
        self._staged_image = None
        self._staged_tail = []
        self._gpu_stage = None
        self._gpu_stage_image_offset = 0
        return self.prepared

    @staticmethod
    def _parallel_memcpy(
        destination_ptr: int,
        copies: list[tuple[int, int, int]],
        *,
        max_workers: int,
    ) -> float:
        """Copy independent CPU ranges with a bounded set of memcpy workers."""

        if not copies:
            return 0.0
        if max_workers < 1:
            raise ValueError("parallel memcpy needs at least one worker")
        bins: list[list[tuple[int, int, int]]] = [
            [] for _ in range(min(max_workers, len(copies)))
        ]
        bin_bytes = [0] * len(bins)
        for copy_item in sorted(copies, key=lambda item: item[2], reverse=True):
            bin_index = min(range(len(bins)), key=bin_bytes.__getitem__)
            bins[bin_index].append(copy_item)
            bin_bytes[bin_index] += copy_item[2]

        def copy_bin(items: list[tuple[int, int, int]]) -> None:
            for destination_offset, source_ptr, nbytes in items:
                ctypes.memmove(
                    destination_ptr + destination_offset,
                    source_ptr,
                    nbytes,
                )

        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(bins)) as executor:
            list(executor.map(copy_bin, bins))
        return time.perf_counter() - started

    def _iter_batched_cuda_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        target_device: torch.device,
        stats: dict[str, float | int],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Batch mmap tensors into large pinned H2D transfers.

        The ordinary model loader still sees the exact same named CUDA tensors.
        A batch remains live until the loader asks for the next one, at which
        point the current stream is synchronized before either reusable buffer
        is overwritten.
        """

        self._ensure_transfer_buffers()
        host_buffer = self._tail_buffers[0]
        capacity = host_buffer.numel()
        if (
            self._checkpoint_device_buffer is None
            or self._checkpoint_device_buffer.numel() != capacity
        ):
            self._checkpoint_device_buffer = torch.empty(
                capacity,
                dtype=torch.uint8,
                device=target_device,
            )
        device_buffer = self._checkpoint_device_buffer
        stream = torch.cuda.current_stream(target_device)
        max_workers = int(
            os.environ.get(
                "SGLANG_PREPARED_CHECKPOINT_MEMCPY_WORKERS",
                str(self._tail_buffer_count),
            )
        )
        broadcast_checkpoint = (
            os.environ.get(
                "SGLANG_PREPARED_BROADCAST_CHECKPOINT",
                "0",
            )
            == "1"
        )
        tp_group = None
        is_source_rank = True
        if broadcast_checkpoint:
            from sglang.srt.distributed.parallel_state import get_tp_group

            tp_group = get_tp_group()
            is_source_rank = tp_group.rank_in_group == 0

        batch: list[tuple[str, torch.Tensor, int, int]] = []
        batch_bytes = 0

        def drain_batch():
            nonlocal batch, batch_bytes
            if not batch:
                return
            stream.synchronize()
            if is_source_rank:
                copies = [
                    (offset, tensor.data_ptr(), nbytes)
                    for _, tensor, offset, nbytes in batch
                ]
                stats["pack_s"] += self._parallel_memcpy(
                    host_buffer.data_ptr(),
                    copies,
                    max_workers=max_workers,
                )
                started = time.perf_counter()
                device_buffer[:batch_bytes].copy_(
                    host_buffer[:batch_bytes],
                    non_blocking=True,
                )
                stream.synchronize()
                stats["h2d_s"] += time.perf_counter() - started
            if tp_group is not None:
                started = time.perf_counter()
                tp_group.broadcast(device_buffer[:batch_bytes], src=0)
                stream.synchronize()
                stats["broadcast_s"] += time.perf_counter() - started
            stats["batches"] += 1
            stats["source_bytes"] += sum(item[3] for item in batch)
            for name, tensor, offset, _ in batch:
                element_size = tensor.element_size()
                loaded = torch.empty(
                    0,
                    dtype=tensor.dtype,
                    device=target_device,
                ).set_(
                    device_buffer.untyped_storage(),
                    offset // element_size,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                )
                yield name, loaded
            batch = []
            batch_bytes = 0

        for name, tensor in weights:
            if tensor.device.type != "cpu":
                raise ValueError(
                    "batched mmap checkpoint expected CPU tensors, got "
                    f"{tensor.device} for {name}"
                )
            if not tensor.is_contiguous():
                raise ValueError(
                    f"batched mmap checkpoint tensor is not contiguous: {name}"
                )
            nbytes = tensor.numel() * tensor.element_size()
            if nbytes > capacity:
                yield from drain_batch()
                loaded = torch.empty_like(tensor, device=target_device)
                loaded_bytes = _storage_bytes(loaded)
                source_offset = 0
                while source_offset < nbytes:
                    size = min(capacity, nbytes - source_offset)
                    stream.synchronize()
                    if is_source_rank:
                        started = time.perf_counter()
                        ctypes.memmove(
                            host_buffer.data_ptr(),
                            tensor.data_ptr() + source_offset,
                            size,
                        )
                        stats["pack_s"] += time.perf_counter() - started
                        started = time.perf_counter()
                        loaded_bytes[source_offset : source_offset + size].copy_(
                            host_buffer[:size],
                            non_blocking=True,
                        )
                        stream.synchronize()
                        stats["h2d_s"] += time.perf_counter() - started
                    if tp_group is not None:
                        started = time.perf_counter()
                        tp_group.broadcast(
                            loaded_bytes[source_offset : source_offset + size],
                            src=0,
                        )
                        stream.synchronize()
                        stats["broadcast_s"] += time.perf_counter() - started
                    stats["batches"] += 1
                    stats["source_bytes"] += size
                    source_offset += size
                yield name, loaded
                stream.synchronize()
                continue

            offset = _align_up(batch_bytes, max(_ALIGNMENT, tensor.element_size()))
            if batch and offset + nbytes > capacity:
                yield from drain_batch()
                offset = 0
            batch.append((name, tensor, offset, nbytes))
            batch_bytes = offset + nbytes

        yield from drain_batch()
        stream.synchronize()

    def _copy_shadow_module(self, path: str, shadow: torch.nn.Module) -> int:
        if self.prepared is None:
            raise RuntimeError("begin_preparation must run first")
        shadow_tensors = dict(iter_model_tensors(shadow))
        prefix = f"{path}."

        if self.prepared.bytes.is_pinned():
            copied = 0
            with torch.cuda.stream(self._stream):
                for segment in self.segments:
                    if not segment.name.startswith(prefix):
                        continue
                    relative_name = segment.name[len(prefix) :]
                    tensor = shadow_tensors.get(relative_name)
                    if tensor is None:
                        raise RuntimeError(
                            f"prepared shadow {path!r} is missing runtime tensor "
                            f"{relative_name!r}"
                        )
                    shadow_bytes = _storage_bytes(tensor)
                    if shadow_bytes.numel() != segment.nbytes:
                        raise RuntimeError(
                            f"prepared storage changed shape for {segment.name}: "
                            f"live={segment.nbytes} bytes, "
                            f"shadow={shadow_bytes.numel()} bytes"
                        )
                    begin = segment.image_offset
                    self.prepared.bytes[begin : begin + segment.nbytes].copy_(
                        shadow_bytes,
                        non_blocking=True,
                    )
                    copied += segment.nbytes
            self._stream.synchronize()
            return copied

        self._ensure_transfer_buffers()
        copied = 0

        def chunks():
            nonlocal copied
            for segment in self.segments:
                if not segment.name.startswith(prefix):
                    continue
                relative_name = segment.name[len(prefix) :]
                tensor = shadow_tensors.get(relative_name)
                if tensor is None:
                    raise RuntimeError(
                        f"prepared shadow {path!r} is missing runtime tensor "
                        f"{relative_name!r}"
                    )
                shadow_bytes = _storage_bytes(tensor)
                if shadow_bytes.numel() != segment.nbytes:
                    raise RuntimeError(
                        f"prepared storage changed shape for {segment.name}: "
                        f"live={segment.nbytes} bytes, "
                        f"shadow={shadow_bytes.numel()} bytes"
                    )
                source_offset = 0
                while source_offset < segment.nbytes:
                    size = min(
                        segment.nbytes - source_offset,
                        self._tail_chunk_bytes,
                    )
                    copied += size
                    yield (
                        shadow_bytes[source_offset : source_offset + size],
                        segment.image_offset + source_offset,
                        size,
                    )
                    source_offset += size

        chunk_iterator = iter(chunks())
        events = [torch.cuda.Event() for _ in self._tail_buffers]

        def stage(slot: int):
            try:
                source, image_offset, size = next(chunk_iterator)
            except StopIteration:
                return None
            buffer = self._tail_buffers[slot]
            with torch.cuda.stream(self._stream):
                buffer[:size].copy_(source, non_blocking=True)
                events[slot].record(self._stream)
            return image_offset, size

        def copy_to_image(
            slot: int,
            image_offset: int,
            size: int,
        ) -> None:
            events[slot].synchronize()
            ctypes.memmove(
                self.prepared.bytes.data_ptr() + image_offset,
                self._tail_buffers[slot].data_ptr(),
                size,
            )

        pending: list[concurrent.futures.Future[None] | None] = [None] * len(
            self._tail_buffers
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self._tail_buffers)
        ) as executor:
            for slot in range(len(self._tail_buffers)):
                item = stage(slot)
                if item is not None:
                    pending[slot] = executor.submit(
                        copy_to_image,
                        slot,
                        item[0],
                        item[1],
                    )
            while any(future is not None for future in pending):
                for slot, future in enumerate(pending):
                    if future is None:
                        continue
                    future.result()
                    item = stage(slot)
                    pending[slot] = (
                        None
                        if item is None
                        else executor.submit(
                            copy_to_image,
                            slot,
                            item[0],
                            item[1],
                        )
                    )
        self._stream.synchronize()
        return copied

    def prepare_from_disk(
        self,
        *,
        model: torch.nn.Module,
        model_config: Any,
        model_path: str,
        load_format: str,
        target_device: torch.device,
        identity: str,
    ) -> dict[str, float | int | str]:
        """Build a next runtime image one scratch module at a time.

        The live module is never rebound or overwritten.  Each shadow reuses the
        ordinary model weight loader and quantization hooks, making arbitrary
        dense checkpoint changes correct without relying on tensor sparsity.
        """

        from sglang.srt.model_loader.loader import (
            DefaultModelLoader,
            LoadConfig,
            get_model_loader,
            restore_weights_before_loading,
        )

        started = time.perf_counter()
        self.begin_preparation(identity)
        original_model_path = model_config.model_path
        model_config.model_path = model_path
        loader = get_model_loader(LoadConfig(load_format=load_format), model_config)
        if not isinstance(loader, DefaultModelLoader):
            model_config.model_path = original_model_path
            raise TypeError(
                f"prepared reload requires DefaultModelLoader, got {loader}"
            )

        copied_bytes = 0
        group_count = 0
        seen_paths: set[str] = set()
        totals: dict[str, float | int] = {
            "clone_s": 0.0,
            "restore_s": 0.0,
            "load_s": 0.0,
            "postprocess_s": 0.0,
            "synchronize_s": 0.0,
            "d2h_s": 0.0,
            "source_bytes": 0,
            "batches": 0,
            "pack_s": 0.0,
            "h2d_s": 0.0,
            "broadcast_s": 0.0,
            "cleanup_gc_s": 0.0,
            "host_target_alloc_s": 0.0,
            "raw_h2d_s": 0.0,
            "raw_h2d_bytes": 0,
            "raw_h2d_copies": 0,
            "host_proxy_clone_s": 0.0,
            "host_proxy_restore_s": 0.0,
            "host_proxy_d2h_s": 0.0,
            "host_proxy_direct_image_groups": 0,
            "host_proxy_direct_image_capacity_bytes": 0,
            "host_proxy_direct_image_used_bytes": 0,
            "host_proxy_fallback_groups": 0,
            "single_pass_load_s": 0.0,
            "load_plan_hits": 0,
            "load_plan_fallback": 0,
            "load_plan_unknown": 0,
            "cuda_allocated_bytes_peak": 0,
            "cuda_allocated_after_release_bytes_peak": 0,
            "cuda_free_bytes_min": 1 << 63,
        }
        load_proxy: torch.nn.Module | None = None
        raw_image_ranges: dict[str, tuple[int, int]] = {}
        try:
            batched_mmap = (
                os.environ.get(
                    "SGLANG_PREPARED_BATCHED_MMAP_CHECKPOINT",
                    "0",
                )
                == "1"
            )
            cpu_assemble = (
                os.environ.get(
                    "SGLANG_PREPARED_CPU_ASSEMBLY",
                    "0",
                )
                == "1"
            )
            single_pass_cpu = (
                os.environ.get(
                    "SGLANG_PREPARED_SINGLE_PASS_CPU_ASSEMBLY",
                    "0",
                )
                == "1"
            )
            if single_pass_cpu and not cpu_assemble:
                raise ValueError(
                    "single-pass CPU assembly requires SGLANG_PREPARED_CPU_ASSEMBLY=1"
                )
            if single_pass_cpu and not self._auto_groups:
                raise ValueError(
                    "single-pass CPU assembly requires automatic runtime "
                    "module grouping"
                )
            if cpu_assemble and batched_mmap:
                raise ValueError(
                    "CPU checkpoint assembly and batched CUDA mmap loading "
                    "are mutually exclusive"
                )
            mmap_checkpoint = (
                batched_mmap
                or os.environ.get("SGLANG_PREPARED_MMAP_CHECKPOINT", "0") == "1"
            )
            if self._auto_groups and not mmap_checkpoint:
                raise ValueError(
                    "automatic runtime module grouping requires the mmap "
                    "checkpoint iterator"
                )
            if single_pass_cpu:
                phase_started = time.perf_counter()
                if self.prepared is None:
                    raise AssertionError("prepared runtime image is missing")
                (
                    load_proxy,
                    host_proxy_stats,
                    raw_image_ranges,
                ) = build_host_load_proxy(
                    model,
                    self._module_groups,
                    target_device,
                    restore_weights_before_loading,
                    image_bytes=self.prepared.bytes,
                    segments=self.segments,
                )
                logger.info(
                    "[RL_PREPARED_STATE] built rank-local host load proxy "
                    "groups=%d wall_s=%.3f clone_s=%.3f restore_s=%.3f "
                    "d2h_s=%.3f direct_image_groups=%d "
                    "direct_image_used_bytes=%d fallback_groups=%d",
                    host_proxy_stats["groups"],
                    time.perf_counter() - phase_started,
                    host_proxy_stats["clone_s"],
                    host_proxy_stats["restore_s"],
                    host_proxy_stats["d2h_s"],
                    host_proxy_stats["direct_image_groups"],
                    host_proxy_stats["direct_image_used_bytes"],
                    host_proxy_stats["fallback_groups"],
                )
                totals["host_proxy_clone_s"] = host_proxy_stats["clone_s"]
                totals["host_proxy_restore_s"] = host_proxy_stats["restore_s"]
                totals["host_proxy_d2h_s"] = host_proxy_stats["d2h_s"]
                for name in (
                    "direct_image_groups",
                    "direct_image_capacity_bytes",
                    "direct_image_used_bytes",
                    "fallback_groups",
                ):
                    totals[f"host_proxy_{name}"] = host_proxy_stats[name]

                phase_started = time.perf_counter()
                from sglang.srt.model_loader.prepared_load_plan import (
                    get_or_create_prepared_load_plan,
                )

                prepared_load_plan = get_or_create_prepared_load_plan(model)
                if prepared_load_plan is not None and prepared_load_plan.recorded:
                    load_plan_stats = prepared_load_plan.replay(
                        load_proxy,
                        streaming_mmap_weights_iterator(model_path),
                        max_workers=int(
                            os.environ.get(
                                "SGLANG_PREPARED_LOAD_PLAN_WORKERS",
                                "16",
                            )
                        ),
                    )
                    totals["load_plan_hits"] = load_plan_stats["hits"]
                    totals["load_plan_fallback"] = load_plan_stats["fallback"]
                    totals["load_plan_unknown"] = load_plan_stats["unknown"]
                else:
                    logger.info(
                        "[RL_PREPARED_LOAD_PLAN] unavailable; using ordinary "
                        "full checkpoint router"
                    )
                    load_proxy.load_weights(
                        streaming_mmap_weights_iterator(model_path)
                    )
                totals["single_pass_load_s"] = time.perf_counter() - phase_started
                logger.info(
                    "[RL_PREPARED_STATE] single-pass rank-local CPU load wall_s=%.3f",
                    totals["single_pass_load_s"],
                )
                finalization_order = runtime_group_finalization_order(
                    self._module_groups,
                    self.segments,
                    raw_image_ranges,
                )
                grouped = ((path, None) for path in finalization_order)
            elif mmap_checkpoint:
                logger.info(
                    "[RL_PREPARED_STATE] using shared page-cache mmap checkpoint%s",
                    " with pinned H2D batching" if batched_mmap else "",
                )
                if self._auto_groups:
                    grouped = grouped_mmap_weights_iterator(
                        model_path,
                        model,
                        self._module_groups,
                    )
                else:
                    weights = ordered_mmap_weights_iterator(model_path)
            else:
                weights = loader._get_weights_iterator(
                    DefaultModelLoader.Source.init_new(model_config, model)
                )
            if not self._auto_groups:
                grouped = itertools.groupby(
                    weights, key=lambda item: checkpoint_module_path(item[0])
                )
            for path, group in grouped:
                if path in seen_paths:
                    raise RuntimeError(
                        f"checkpoint iterator revisited {path!r}; bounded scratch "
                        "preparation requires each module's tensors to be contiguous"
                    )
                seen_paths.add(path)
                group_started = time.perf_counter()
                phase_started = group_started
                host_targets: list[HostLoadTarget] = []
                host_target_alloc_s = 0.0
                raw_h2d_s = 0.0
                raw_h2d_bytes = 0
                raw_h2d_copies = 0
                if single_pass_cpu:
                    if load_proxy is None:
                        raise AssertionError("single-pass load proxy is missing")
                    host_shadow = module_at_path(load_proxy, path)
                    direct_image = path in raw_image_ranges
                    if direct_image:
                        pinned_shadow = host_shadow
                    else:
                        pinned_shadow = clone_module_tensors(
                            host_shadow,
                            target_device=torch.device("cpu"),
                            pin_memory=True,
                        )
                    host_target_alloc_s = time.perf_counter() - phase_started
                    replace_proxy_module(
                        load_proxy,
                        model,
                        path,
                        module_at_path(model, path),
                    )

                    phase_started = time.perf_counter()
                    shadow = clone_module_tensors(
                        pinned_shadow,
                        target_device=target_device,
                    )
                    torch.cuda.synchronize(target_device)
                    raw_h2d_s = time.perf_counter() - phase_started
                    shadow_storages = {
                        _tensor_storage_key(tensor)
                        for _, tensor in iter_model_tensors(shadow)
                        if tensor.device.type == "cuda"
                    }
                    raw_h2d_copies = len(shadow_storages)
                    raw_h2d_bytes = sum(key[2] for key in shadow_storages)
                    if not direct_image:
                        del pinned_shadow
                    release_module_tensors(host_shadow)
                    del host_shadow
                    clone_s = 0.0
                    restore_s = 0.0
                    load_s = 0.0
                else:
                    proxy, shadow = clone_module_proxy(model, path)
                    clone_s = time.perf_counter() - phase_started
                    phase_started = time.perf_counter()
                    restore_weights_before_loading(shadow, target_device)
                    restore_s = time.perf_counter() - phase_started
                if cpu_assemble and not single_pass_cpu:
                    phase_started = time.perf_counter()
                    host_targets = move_load_targets_to_pinned_host(shadow)
                    host_target_alloc_s = time.perf_counter() - phase_started
                batch_stats: dict[str, float | int] = {
                    "source_bytes": 0,
                    "batches": 0,
                    "pack_s": 0.0,
                    "h2d_s": 0.0,
                    "broadcast_s": 0.0,
                }
                if not single_pass_cpu:
                    group_weights: Iterable[tuple[str, torch.Tensor]] = group
                    if batched_mmap:
                        group_weights = self._iter_batched_cuda_weights(
                            group,
                            target_device,
                            batch_stats,
                        )
                    phase_started = time.perf_counter()
                    proxy.load_weights(group_weights)
                    load_s = time.perf_counter() - phase_started
                    if cpu_assemble:
                        phase_started = time.perf_counter()
                        raw_h2d_copies, raw_h2d_bytes = move_load_targets_to_device(
                            host_targets,
                            target_device,
                            self._stream,
                        )
                        raw_h2d_s = time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                for _, module in shadow.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        quant_method.process_weights_after_loading(module)
                postprocess_s = time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                torch.cuda.synchronize(target_device)
                synchronize_s = time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                group_bytes = self._copy_shadow_module(path, shadow)
                d2h_s = time.perf_counter() - phase_started
                copied_bytes += group_bytes
                group_count += 1
                totals["clone_s"] += clone_s
                totals["restore_s"] += restore_s
                totals["load_s"] += load_s
                totals["postprocess_s"] += postprocess_s
                totals["synchronize_s"] += synchronize_s
                totals["d2h_s"] += d2h_s
                totals["host_target_alloc_s"] += host_target_alloc_s
                totals["raw_h2d_s"] += raw_h2d_s
                totals["raw_h2d_bytes"] += raw_h2d_bytes
                totals["raw_h2d_copies"] += raw_h2d_copies
                for name in (
                    "source_bytes",
                    "batches",
                    "pack_s",
                    "h2d_s",
                    "broadcast_s",
                ):
                    totals[name] += batch_stats[name]
                logger.info(
                    "[RL_PREPARED_STATE] group=%s bytes=%d wall_s=%.3f "
                    "clone_s=%.3f restore_s=%.3f load_s=%.3f "
                    "postprocess_s=%.3f synchronize_s=%.3f d2h_s=%.3f "
                    "source_bytes=%d batches=%d pack_s=%.3f h2d_s=%.3f "
                    "broadcast_s=%.3f host_target_alloc_s=%.3f "
                    "raw_h2d_s=%.3f raw_h2d_bytes=%d raw_h2d_copies=%d",
                    path,
                    group_bytes,
                    time.perf_counter() - group_started,
                    clone_s,
                    restore_s,
                    load_s,
                    postprocess_s,
                    synchronize_s,
                    d2h_s,
                    batch_stats["source_bytes"],
                    batch_stats["batches"],
                    batch_stats["pack_s"],
                    batch_stats["h2d_s"],
                    batch_stats["broadcast_s"],
                    host_target_alloc_s,
                    raw_h2d_s,
                    raw_h2d_bytes,
                    raw_h2d_copies,
                )
                cuda_allocated_before_release = torch.cuda.memory_allocated(
                    target_device
                )
                # Module/quantization objects can contain reference cycles. Drop
                # every tensor leaf explicitly once its finalized bytes are in
                # the host image so bounded preparation never accumulates one
                # CUDA scratch group per layer while waiting for cyclic GC.
                release_module_tensors(shadow)
                del shadow
                # Parameter subclasses and quantization helpers can form
                # short-lived reference cycles. Collect the young generation
                # while the scratch group is bounded; waiting until the full
                # frontier is finalized can retain one CUDA layer per cycle.
                del module, quant_method
                cleanup_gc_started = time.perf_counter()
                gc.collect(0)
                cleanup_gc_s = time.perf_counter() - cleanup_gc_started
                totals["cleanup_gc_s"] += cleanup_gc_s
                cuda_allocated_after_release = torch.cuda.memory_allocated(
                    target_device
                )
                cuda_free_bytes, _ = torch.cuda.mem_get_info(target_device)
                totals["cuda_allocated_bytes_peak"] = max(
                    totals["cuda_allocated_bytes_peak"],
                    cuda_allocated_before_release,
                )
                totals["cuda_allocated_after_release_bytes_peak"] = max(
                    totals["cuda_allocated_after_release_bytes_peak"],
                    cuda_allocated_after_release,
                )
                totals["cuda_free_bytes_min"] = min(
                    totals["cuda_free_bytes_min"],
                    cuda_free_bytes,
                )
                logger.info(
                    "[RL_PREPARED_HBM] group=%s allocated_before_release=%d "
                    "allocated_after_release=%d free_after_release=%d",
                    path,
                    cuda_allocated_before_release,
                    cuda_allocated_after_release,
                    cuda_free_bytes,
                )
                if not single_pass_cpu:
                    del proxy
                    cleanup_gc_started = time.perf_counter()
                    gc.collect()
                    cleanup_gc_s = time.perf_counter() - cleanup_gc_started
                    totals["cleanup_gc_s"] += cleanup_gc_s
                    logger.info(
                        "[RL_PREPARED_STATE] group=%s cleanup_gc_s=%.3f",
                        path,
                        cleanup_gc_s,
                    )
            if single_pass_cpu:
                cleanup_gc_started = time.perf_counter()
                gc.collect()
                cleanup_gc_s = time.perf_counter() - cleanup_gc_started
                totals["cleanup_gc_s"] += cleanup_gc_s
                logger.info(
                    "[RL_PREPARED_STATE] single-pass cleanup_gc_s=%.3f",
                    cleanup_gc_s,
                )
        finally:
            load_proxy = None
            self._checkpoint_device_buffer = None
            torch.cuda.empty_cache()
            model_config.model_path = original_model_path

        expected_bytes = sum(segment.nbytes for segment in self.segments)
        if copied_bytes != expected_bytes:
            raise RuntimeError(
                "prepared runtime image is incomplete: "
                f"copied={copied_bytes} expected={expected_bytes} bytes"
            )

        stage_stats = self.stage_prepared()
        return {
            "identity": identity,
            "groups": group_count,
            "copied_bytes": copied_bytes,
            "image_bytes": self.image_nbytes,
            "pinned_prefix_bytes": stage_stats["pinned_prefix_bytes"],
            "tail_buffer_bytes": stage_stats["tail_buffer_bytes"],
            "gpu_stage_bytes": stage_stats["gpu_stage_bytes"],
            "gpu_stage_wall_s": stage_stats["gpu_stage_wall_s"],
            "stage_wall_s": stage_stats["wall_s"],
            "clone_s": round(float(totals["clone_s"]), 6),
            "restore_s": round(float(totals["restore_s"]), 6),
            "load_s": round(float(totals["load_s"]), 6),
            "postprocess_s": round(float(totals["postprocess_s"]), 6),
            "synchronize_s": round(float(totals["synchronize_s"]), 6),
            "d2h_s": round(float(totals["d2h_s"]), 6),
            "source_bytes": int(totals["source_bytes"]),
            "batches": int(totals["batches"]),
            "pack_s": round(float(totals["pack_s"]), 6),
            "h2d_s": round(float(totals["h2d_s"]), 6),
            "broadcast_s": round(float(totals["broadcast_s"]), 6),
            "cleanup_gc_s": round(float(totals["cleanup_gc_s"]), 6),
            "host_target_alloc_s": round(float(totals["host_target_alloc_s"]), 6),
            "raw_h2d_s": round(float(totals["raw_h2d_s"]), 6),
            "raw_h2d_bytes": int(totals["raw_h2d_bytes"]),
            "raw_h2d_copies": int(totals["raw_h2d_copies"]),
            "host_proxy_clone_s": round(float(totals["host_proxy_clone_s"]), 6),
            "host_proxy_restore_s": round(float(totals["host_proxy_restore_s"]), 6),
            "host_proxy_d2h_s": round(float(totals["host_proxy_d2h_s"]), 6),
            "host_proxy_direct_image_groups": int(
                totals["host_proxy_direct_image_groups"]
            ),
            "host_proxy_direct_image_capacity_bytes": int(
                totals["host_proxy_direct_image_capacity_bytes"]
            ),
            "host_proxy_direct_image_used_bytes": int(
                totals["host_proxy_direct_image_used_bytes"]
            ),
            "host_proxy_fallback_groups": int(
                totals["host_proxy_fallback_groups"]
            ),
            "single_pass_load_s": round(float(totals["single_pass_load_s"]), 6),
            "load_plan_hits": int(totals["load_plan_hits"]),
            "load_plan_fallback": int(totals["load_plan_fallback"]),
            "load_plan_unknown": int(totals["load_plan_unknown"]),
            "cuda_allocated_bytes_peak": int(totals["cuda_allocated_bytes_peak"]),
            "cuda_allocated_after_release_bytes_peak": int(
                totals["cuda_allocated_after_release_bytes_peak"]
            ),
            "cuda_free_bytes_min": int(totals["cuda_free_bytes_min"]),
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def _ensure_transfer_buffers(self) -> None:
        if not self._tail_buffers:
            self._tail_buffers = [
                torch.empty(self._tail_chunk_bytes, dtype=torch.uint8, pin_memory=True)
                for _ in range(self._tail_buffer_count)
            ]

    def _ensure_pinned_prefix(self) -> int:
        requested = min(
            self.image_nbytes,
            int(os.environ.get("SGLANG_PREPARED_PINNED_GB", "56")) * _GIB,
        )
        if self._pinned_prefix is None or self._pinned_prefix.numel() != requested:
            self._pinned_prefix = torch.empty(
                requested,
                dtype=torch.uint8,
                pin_memory=True,
            )
        return requested

    def preallocate_transfer_buffers(self) -> dict[str, float | int]:
        """Pay page-locking/allocation costs once during engine startup."""

        started = time.perf_counter()
        if self._full_pinned:
            if self.prepared is None and self._preallocated_image_bytes is None:
                try:
                    self._preallocated_image_bytes = torch.empty(
                        self.image_nbytes,
                        dtype=torch.uint8,
                        pin_memory=True,
                    )
                except RuntimeError:
                    logger.exception(
                        "[RL_PREPARED_STATE] full pinned image preallocation "
                        "failed; falling back to bounded staging"
                    )
                    self._full_pinned = False
            if self._full_pinned:
                return {
                    "full_pinned_image_bytes": self.image_nbytes,
                    "pinned_prefix_bytes": 0,
                    "tail_buffer_bytes": 0,
                    "wall_s": round(time.perf_counter() - started, 6),
                }
        pinned_prefix_bytes = self._ensure_pinned_prefix()
        self._ensure_transfer_buffers()
        return {
            "full_pinned_image_bytes": 0,
            "pinned_prefix_bytes": pinned_prefix_bytes,
            "tail_buffer_bytes": sum(item.numel() for item in self._tail_buffers),
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def _copy_tail_chunk(
        self,
        image: RuntimeStateImage,
        slot: int,
        image_offset: int,
        wait_event: torch.cuda.Event | None = None,
        image_end: int | None = None,
    ) -> tuple[int, int, float]:
        if wait_event is not None:
            wait_event.synchronize()
        if image_end is None:
            image_end = self.image_nbytes
        size = min(self._tail_chunk_bytes, image_end - image_offset)
        started = time.perf_counter()
        ctypes.memmove(
            self._tail_buffers[slot].data_ptr(),
            image.bytes.data_ptr() + image_offset,
            size,
        )
        return image_offset, size, time.perf_counter() - started

    def _prefill_tail(
        self,
        image: RuntimeStateImage,
        image_offset: int,
        image_end: int | None = None,
    ) -> list[tuple[int, int, float] | None]:
        if image_end is None:
            image_end = self.image_nbytes
        ready: list[tuple[int, int, float] | None] = [None] * self._tail_buffer_count
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._tail_buffer_count
        ) as executor:
            futures = {}
            for slot in range(self._tail_buffer_count):
                if image_offset >= image_end:
                    break
                futures[slot] = executor.submit(
                    self._copy_tail_chunk,
                    image,
                    slot,
                    image_offset,
                    None,
                    image_end,
                )
                image_offset += min(self._tail_chunk_bytes, image_end - image_offset)
            for slot, future in futures.items():
                image_offset, size, _ = future.result()
                ready[slot] = (image_offset, size, 0.0)
        return ready

    def _stage_gpu_range(
        self,
        image: RuntimeStateImage,
        image_offset: int,
        nbytes: int,
    ) -> dict[str, float | int]:
        if nbytes <= 0:
            self._gpu_stage = None
            self._gpu_stage_image_offset = image_offset
            return {"bytes": 0, "wall_s": 0.0}

        started = time.perf_counter()
        image_end = image_offset + nbytes
        self._gpu_stage = torch.empty(
            nbytes,
            dtype=torch.uint8,
            device=torch.cuda.current_device(),
        )
        self._gpu_stage_image_offset = image_offset
        ready: list[
            tuple[int, int, float]
            | concurrent.futures.Future[tuple[int, int, float]]
            | None
        ] = list(self._prefill_tail(image, image_offset, image_end))
        next_offset = image_offset + sum(item[1] for item in ready if item is not None)
        events = [torch.cuda.Event() for _ in self._tail_buffers]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._tail_buffer_count
        ) as executor:
            while any(item is not None for item in ready):
                for slot in range(self._tail_buffer_count):
                    item = ready[slot]
                    if item is None:
                        continue
                    if isinstance(item, concurrent.futures.Future):
                        item = item.result()
                    source_offset, size, _ = item
                    destination_offset = source_offset - image_offset
                    with torch.cuda.stream(self._gpu_stage_stream):
                        self._gpu_stage[
                            destination_offset : destination_offset + size
                        ].copy_(self._tail_buffers[slot][:size], non_blocking=True)
                        events[slot].record(self._gpu_stage_stream)
                    if next_offset < image_end:
                        ready[slot] = executor.submit(
                            self._copy_tail_chunk,
                            image,
                            slot,
                            next_offset,
                            events[slot],
                            image_end,
                        )
                        next_offset += min(
                            self._tail_chunk_bytes, image_end - next_offset
                        )
                    else:
                        ready[slot] = None
        self._gpu_stage_stream.synchronize()
        return {
            "bytes": nbytes,
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def stage_prepared(self) -> dict[str, float | int]:
        """Pre-pin the largest safe prefix; this runs before inference pauses."""

        if self.prepared is None:
            raise RuntimeError("no prepared runtime image")
        started = time.perf_counter()
        if self.prepared.bytes.is_pinned():
            self._staged_image = self.prepared
            self._staged_tail = []
            self._gpu_stage = None
            return {
                "pinned_prefix_bytes": self.image_nbytes,
                "tail_buffer_bytes": 0,
                "gpu_stage_bytes": 0,
                "gpu_stage_wall_s": 0.0,
                "wall_s": round(time.perf_counter() - started, 6),
            }
        requested = self._ensure_pinned_prefix()
        self._ensure_transfer_buffers()
        self._pinned_prefix.copy_(self.prepared.bytes[:requested])
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info()
        reserve_bytes = (
            int(os.environ.get("SGLANG_PREPARED_GPU_RESERVE_GB", "12")) * _GIB
        )
        requested_gpu_bytes = (
            int(os.environ.get("SGLANG_PREPARED_GPU_STAGING_GB", "30")) * _GIB
        )
        gpu_stage_bytes = min(
            requested_gpu_bytes,
            max(0, free_bytes - reserve_bytes),
            self.image_nbytes - requested,
        )
        gpu_stage_stats = self._stage_gpu_range(
            self.prepared,
            requested,
            gpu_stage_bytes,
        )
        tail_offset = requested + gpu_stage_bytes
        self._staged_tail = self._prefill_tail(self.prepared, tail_offset)
        self._staged_image = self.prepared
        return {
            "pinned_prefix_bytes": requested,
            "tail_buffer_bytes": sum(item.numel() for item in self._tail_buffers),
            "gpu_stage_bytes": gpu_stage_stats["bytes"],
            "gpu_stage_wall_s": gpu_stage_stats["wall_s"],
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def _scatter_range(
        self,
        image_start: int,
        source: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> int:
        image_end = image_start + source.numel()
        copies = 0
        for segment in self.segments:
            segment_start = segment.image_offset
            segment_end = segment_start + segment.nbytes
            begin = max(image_start, segment_start)
            end = min(image_end, segment_end)
            if begin >= end:
                continue
            segment.device_bytes[begin - segment_start : end - segment_start].copy_(
                source[begin - image_start : end - image_start], non_blocking=True
            )
            copies += 1
        return copies

    def _copy_to_device(self, image: RuntimeStateImage) -> dict[str, float | int]:
        if image.bytes.is_pinned():
            if self._staged_image is not image:
                raise RuntimeError("requested runtime image is not staged")
            wall_started = time.perf_counter()
            with torch.cuda.stream(self._stream):
                copies = self._scatter_range(0, image.bytes, self._stream)
            self._stream.synchronize()
            wall_s = time.perf_counter() - wall_started
            return {
                "bytes": self.image_nbytes,
                "copies": copies,
                "cpu_tail_copy_thread_s": 0.0,
                "gpu_stage_bytes": 0,
                "wall_s": round(wall_s, 6),
                "gbps": round(
                    self.image_nbytes / max(wall_s, 1e-9) / 1e9,
                    3,
                ),
            }
        if self._pinned_prefix is None or not self._tail_buffers:
            raise RuntimeError("stage_prepared must run before commit")
        if self._staged_image is not image:
            raise RuntimeError("requested runtime image is not staged")

        stream = self._stream
        events = [torch.cuda.Event() for _ in self._tail_buffers]
        cpu_copy_thread_s = 0.0
        copies = 0
        wall_started = time.perf_counter()
        with torch.cuda.stream(self._prefix_stream):
            copies += self._scatter_range(0, self._pinned_prefix, self._prefix_stream)
        gpu_stage_bytes = self._gpu_stage.numel() if self._gpu_stage is not None else 0
        if self._gpu_stage is not None:
            with torch.cuda.stream(self._gpu_stage_stream):
                copies += self._scatter_range(
                    self._gpu_stage_image_offset,
                    self._gpu_stage,
                    self._gpu_stage_stream,
                )

        ready: list[
            tuple[int, int, float]
            | concurrent.futures.Future[tuple[int, int, float]]
            | None
        ] = list(self._staged_tail)
        next_offset = (
            self._pinned_prefix.numel()
            + gpu_stage_bytes
            + sum(item[1] for item in self._staged_tail if item is not None)
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._tail_buffer_count
        ) as executor:
            while any(item is not None for item in ready):
                for slot in range(self._tail_buffer_count):
                    item = ready[slot]
                    if item is None:
                        continue
                    if isinstance(item, concurrent.futures.Future):
                        item = item.result()
                    image_offset, size, copy_s = item
                    cpu_copy_thread_s += copy_s
                    with torch.cuda.stream(stream):
                        copies += self._scatter_range(
                            image_offset,
                            self._tail_buffers[slot][:size],
                            stream,
                        )
                        events[slot].record(stream)
                    if next_offset < self.image_nbytes:
                        ready[slot] = executor.submit(
                            self._copy_tail_chunk,
                            image,
                            slot,
                            next_offset,
                            events[slot],
                        )
                        next_offset += min(
                            self._tail_chunk_bytes,
                            self.image_nbytes - next_offset,
                        )
                    else:
                        ready[slot] = None

        self._prefix_stream.synchronize()
        self._gpu_stage_stream.synchronize()
        stream.synchronize()
        wall_s = time.perf_counter() - wall_started
        return {
            "bytes": self.image_nbytes,
            "copies": copies,
            "cpu_tail_copy_thread_s": round(cpu_copy_thread_s, 6),
            "gpu_stage_bytes": gpu_stage_bytes,
            "wall_s": round(wall_s, 6),
            "gbps": round(self.image_nbytes / max(wall_s, 1e-9) / 1e9, 3),
        }

    def commit(self) -> dict[str, float | int | str]:
        """Overwrite every checkpoint-derived storage and retain the old image.

        A previous successfully committed image is available for rollback.  The
        first prepared commit intentionally has no rollback snapshot: capturing
        the serving model would itself take several minutes on this host and is
        outside the paused-path contract.
        """

        if self.prepared is None:
            raise RuntimeError("no prepared runtime image")
        prepared = self.prepared
        try:
            stats = self._copy_to_device(prepared)
        except Exception:
            if self.active is not None:
                logger.exception(
                    "prepared commit failed; restoring active runtime image"
                )
                self.prepared = self.active
                self.stage_prepared()
                self._copy_to_device(self.active)
                self._gpu_stage = None
            raise
        self._gpu_stage = None
        self.active, self.prepared = prepared, self.active
        stats["identity"] = prepared.identity
        return stats
