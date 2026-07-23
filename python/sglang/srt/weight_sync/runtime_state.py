"""Prepared host runtime images for short-pause dense weight commits.

The ordinary disk loader is the correctness fallback.  This module implements
the final commit primitive for a faster path: all checkpoint parsing, sharding,
fusion and quantization must already have produced a byte-exact host image of
the model's live CUDA storages before :meth:`commit` is called.

The implementation is intentionally dense-update safe.  It inventories every
unique checkpoint-derived runtime storage rather than changed tensor names,
copies every byte in that inventory, and preserves aliases by overwriting
existing storage instead of rebinding Parameters.  Checkpoint-invariant CUDA
buffers are deliberately excluded: copying them would add hundreds of gigabytes
of needless GPU-to-host traffic without changing the result.  A bounded pinned
prefix and two pinned streaming buffers keep the design within hosts that cannot
pin an entire 600+ GB tensor-parallel runtime image.
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


@dataclass
class HostLoadTarget:
    """One registered scratch tensor temporarily assembled in pinned CPU RAM."""

    name: str
    parameter: torch.nn.Parameter
    host_data: torch.Tensor


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
) -> tuple[list[RuntimeStorageSegment], int]:
    """Build a deterministic checkpoint-derived plan without changing addresses."""

    unique: dict[tuple[int | None, int, int], tuple[str, torch.Tensor]] = {}
    for name, tensor in iter_model_tensors(model):
        if tensor.device.type != "cuda" or runtime_module_path(name) is None:
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


def _clone_tensor_storage(
    tensor: torch.Tensor,
    tensor_memo: dict[int, torch.Tensor],
    storage_memo: dict[tuple[int | None, int, int], torch.Tensor],
) -> torch.Tensor:
    """Clone a tensor while retaining subclasses, views, and storage aliases."""

    cached = tensor_memo.get(id(tensor))
    if cached is not None:
        return cached

    storage = tensor.untyped_storage()
    storage_key = (tensor.device.index, storage.data_ptr(), storage.nbytes())
    cloned_storage = storage_memo.get(storage_key)
    if cloned_storage is None:
        cloned_storage = _storage_bytes(tensor).clone()
        storage_memo[storage_key] = cloned_storage
    view = torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
        cloned_storage.untyped_storage(),
        tensor.storage_offset(),
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
    tensor_memo[id(tensor)] = cloned
    return cloned


def _clone_attribute_tensors(
    value: Any,
    tensor_memo: dict[int, torch.Tensor],
    storage_memo: dict[tuple[int | None, int, int], torch.Tensor],
    depth: int = 0,
) -> Any:
    """Clone tensor leaves but share immutable and non-tensor runtime objects."""

    if isinstance(value, torch.Tensor):
        return _clone_tensor_storage(value, tensor_memo, storage_memo)
    if depth >= 2:
        return value
    if isinstance(value, dict):
        return {
            key: _clone_attribute_tensors(child, tensor_memo, storage_memo, depth + 1)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _clone_attribute_tensors(child, tensor_memo, storage_memo, depth + 1)
            for child in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _clone_attribute_tensors(child, tensor_memo, storage_memo, depth + 1)
            for child in value
        )
    return value


def clone_module_tensors(module: torch.nn.Module) -> torch.nn.Module:
    """Clone a module tree's tensor state without copying CUDA runtime objects."""

    tensor_memo: dict[int, torch.Tensor] = {}
    storage_memo: dict[tuple[int | None, int, int], torch.Tensor] = {}

    def clone(current: torch.nn.Module) -> torch.nn.Module:
        result = copy.copy(current)
        result._parameters = {
            name: (
                None
                if parameter is None
                else _clone_tensor_storage(parameter, tensor_memo, storage_memo)
            )
            for name, parameter in current._parameters.items()
        }
        result._buffers = {
            name: (
                None
                if buffer is None
                else _clone_tensor_storage(buffer, tensor_memo, storage_memo)
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
                value, tensor_memo, storage_memo
            )
        return result

    return clone(module)


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
        self.segments, self.image_nbytes = build_runtime_storage_plan(model)
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
        }
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
            if cpu_assemble and batched_mmap:
                raise ValueError(
                    "CPU checkpoint assembly and batched CUDA mmap loading "
                    "are mutually exclusive"
                )
            if (
                batched_mmap
                or os.environ.get("SGLANG_PREPARED_MMAP_CHECKPOINT", "0") == "1"
            ):
                logger.info(
                    "[RL_PREPARED_STATE] using shared page-cache mmap checkpoint%s",
                    " with pinned H2D batching" if batched_mmap else "",
                )
                weights = ordered_mmap_weights_iterator(model_path)
            else:
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
                phase_started = group_started
                proxy, shadow = clone_module_proxy(model, path)
                clone_s = time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                restore_weights_before_loading(shadow, target_device)
                restore_s = time.perf_counter() - phase_started
                host_targets: list[HostLoadTarget] = []
                host_target_alloc_s = 0.0
                raw_h2d_s = 0.0
                raw_h2d_bytes = 0
                raw_h2d_copies = 0
                if cpu_assemble:
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
                del shadow, proxy
                cleanup_gc_started = time.perf_counter()
                gc.collect()
                cleanup_gc_s = time.perf_counter() - cleanup_gc_started
                totals["cleanup_gc_s"] += cleanup_gc_s
                logger.info(
                    "[RL_PREPARED_STATE] group=%s cleanup_gc_s=%.3f",
                    path,
                    cleanup_gc_s,
                )
        finally:
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
