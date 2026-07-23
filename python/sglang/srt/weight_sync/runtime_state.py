"""Prepared host runtime images for short-pause dense weight commits.

The ordinary disk loader is the correctness fallback.  This module implements
the final commit primitive for a faster path: all checkpoint parsing, sharding,
fusion and quantization must already have produced a byte-exact host image of
the model's live CUDA storages before :meth:`commit` is called.

The implementation is intentionally dense-update safe.  It inventories unique
storages rather than tensor names, copies every byte, and preserves aliases by
overwriting existing storage instead of rebinding Parameters.  A bounded pinned
prefix and two pinned streaming buffers keep the design within hosts that cannot
pin an entire 600+ GB tensor-parallel runtime image.
"""

from __future__ import annotations

import copy
import gc
import itertools
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch

logger = logging.getLogger(__name__)

_GIB = 1 << 30
_MIB = 1 << 20
_ALIGNMENT = 4096
_DECODER_LAYER = re.compile(r"^(language_model\.model\.layers\.\d+)(?:\.|$)")
_VISION_BLOCK = re.compile(r"^(vision_tower\.encoder\.blocks\.\d+)(?:\.|$)")


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


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _storage_bytes(tensor: torch.Tensor) -> torch.Tensor:
    storage = tensor.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
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


def build_runtime_storage_plan(
    model: torch.nn.Module,
) -> tuple[list[RuntimeStorageSegment], int]:
    """Build a deterministic unique-storage plan without changing GPU addresses."""

    unique: dict[tuple[int | None, int, int], tuple[str, torch.Tensor]] = {}
    for name, tensor in iter_model_tensors(model):
        if tensor.device.type != "cuda":
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


def checkpoint_module_path(name: str) -> str:
    """Return a bounded scratch-module path for a checkpoint tensor.

    Kimi checkpoints are physically ordered by these groups, which lets the
    preparer consume every source tensor exactly once while keeping only one
    decoder/vision block in spare HBM.
    """

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
    raise ValueError(f"no bounded scratch module for checkpoint tensor {name!r}")


def clone_module_proxy(
    model: torch.nn.Module, path: str
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Shallow-clone ancestors and deep-clone only the selected CUDA module."""

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
            shadow = copy.deepcopy(live_child)
            proxy_cursor._modules[part] = shadow
            return proxy, shadow
        proxy_child = copy.copy(live_child)
        proxy_child._modules = live_child._modules.copy()
        proxy_cursor._modules[part] = proxy_child
        proxy_cursor = proxy_child
        live = live_child
    raise AssertionError("empty module path")


class PreparedRuntimeState:
    """Own active/prepared host images and commit prepared bytes in-place."""

    def __init__(self, model: torch.nn.Module):
        self.segments, self.image_nbytes = build_runtime_storage_plan(model)
        self.active: RuntimeStateImage | None = None
        self.prepared: RuntimeStateImage | None = None
        self._pinned_prefix: torch.Tensor | None = None
        self._tail_buffers: list[torch.Tensor] = []
        self._stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self._tail_chunk_bytes = int(
            os.environ.get("SGLANG_PREPARED_TAIL_CHUNK_MIB", "1024")
        ) * _MIB

    @property
    def prepared_identity(self) -> str | None:
        return self.prepared.identity if self.prepared is not None else None

    def allocate_image(self, identity: str) -> RuntimeStateImage:
        return RuntimeStateImage(
            bytes=torch.empty(self.image_nbytes, dtype=torch.uint8), identity=identity
        )

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
        """Start from the active image so unchanged/static storages stay exact."""

        if self.active is None:
            raise RuntimeError("capture_active must run before preparing an update")
        self.prepared = RuntimeStateImage(
            bytes=self.active.bytes.clone(), identity=identity
        )
        return self.prepared

    def _copy_shadow_module(self, path: str, shadow: torch.nn.Module) -> int:
        if self.prepared is None:
            raise RuntimeError("begin_preparation must run first")
        shadow_tensors = dict(iter_model_tensors(shadow))
        prefix = f"{path}."
        copied = 0
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
                    f"live={segment.nbytes} bytes, shadow={shadow_bytes.numel()} bytes"
                )
            begin = segment.image_offset
            self.prepared.bytes[begin : begin + segment.nbytes].copy_(shadow_bytes)
            copied += segment.nbytes
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
            raise TypeError(f"prepared reload requires DefaultModelLoader, got {loader}")

        copied_bytes = 0
        group_count = 0
        seen_paths: set[str] = set()
        try:
            weights = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(model_config, model)
            )
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
                proxy, shadow = clone_module_proxy(model, path)
                restore_weights_before_loading(shadow, target_device)
                proxy.load_weights(group)
                for _, module in shadow.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        quant_method.process_weights_after_loading(module)
                torch.cuda.synchronize(target_device)
                group_bytes = self._copy_shadow_module(path, shadow)
                copied_bytes += group_bytes
                group_count += 1
                logger.info(
                    "[RL_PREPARED_STATE] group=%s bytes=%d wall_s=%.3f",
                    path,
                    group_bytes,
                    time.perf_counter() - group_started,
                )
                del shadow, proxy
                gc.collect()
        finally:
            model_config.model_path = original_model_path

        stage_stats = self.stage_prepared()
        return {
            "identity": identity,
            "groups": group_count,
            "copied_bytes": copied_bytes,
            "image_bytes": self.image_nbytes,
            "stage_wall_s": stage_stats["wall_s"],
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def stage_prepared(self) -> dict[str, float | int]:
        """Pre-pin the largest safe prefix; this runs before inference pauses."""

        if self.prepared is None:
            raise RuntimeError("no prepared runtime image")
        requested = min(
            self.image_nbytes,
            int(os.environ.get("SGLANG_PREPARED_PINNED_GB", "60")) * _GIB,
        )
        if self._pinned_prefix is None or self._pinned_prefix.numel() != requested:
            self._pinned_prefix = torch.empty(
                requested, dtype=torch.uint8, pin_memory=True
            )
        if not self._tail_buffers:
            self._tail_buffers = [
                torch.empty(
                    self._tail_chunk_bytes, dtype=torch.uint8, pin_memory=True
                )
                for _ in range(2)
            ]
        started = time.perf_counter()
        self._pinned_prefix.copy_(self.prepared.bytes[:requested])
        return {
            "pinned_prefix_bytes": requested,
            "tail_buffer_bytes": sum(item.numel() for item in self._tail_buffers),
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
            segment.device_bytes[
                begin - segment_start : end - segment_start
            ].copy_(source[begin - image_start : end - image_start], non_blocking=True)
            copies += 1
        return copies

    def _copy_to_device(self, image: RuntimeStateImage) -> dict[str, float | int]:
        if self._pinned_prefix is None or not self._tail_buffers:
            raise RuntimeError("stage_prepared must run before commit")

        stream = self._stream
        events = [torch.cuda.Event() for _ in self._tail_buffers]
        pending = [False] * len(self._tail_buffers)
        cpu_copy_s = 0.0
        copies = 0
        wall_started = time.perf_counter()
        with torch.cuda.stream(stream):
            copies += self._scatter_range(0, self._pinned_prefix, stream)

        offset = self._pinned_prefix.numel()
        tail_index = 0
        while offset < self.image_nbytes:
            slot = tail_index % len(self._tail_buffers)
            if pending[slot]:
                events[slot].synchronize()
            size = min(self._tail_chunk_bytes, self.image_nbytes - offset)
            cpu_started = time.perf_counter()
            self._tail_buffers[slot][:size].copy_(image.bytes[offset : offset + size])
            cpu_copy_s += time.perf_counter() - cpu_started
            with torch.cuda.stream(stream):
                copies += self._scatter_range(
                    offset, self._tail_buffers[slot][:size], stream
                )
                events[slot].record(stream)
            pending[slot] = True
            offset += size
            tail_index += 1

        stream.synchronize()
        wall_s = time.perf_counter() - wall_started
        return {
            "bytes": self.image_nbytes,
            "copies": copies,
            "cpu_tail_copy_s": round(cpu_copy_s, 6),
            "wall_s": round(wall_s, 6),
            "gbps": round(self.image_nbytes / max(wall_s, 1e-9) / 1e9, 3),
        }

    def commit(self) -> dict[str, float | int | str]:
        """Overwrite every inventoried storage, retaining the old host image."""

        if self.prepared is None:
            raise RuntimeError("no prepared runtime image")
        prepared = self.prepared
        try:
            stats = self._copy_to_device(prepared)
        except Exception:
            if self.active is not None:
                logger.exception("prepared commit failed; restoring active runtime image")
                self._pinned_prefix.copy_(
                    self.active.bytes[: self._pinned_prefix.numel()]
                )
                self._copy_to_device(self.active)
            raise
        self.active, self.prepared = prepared, self.active
        stats["identity"] = prepared.identity
        return stats
