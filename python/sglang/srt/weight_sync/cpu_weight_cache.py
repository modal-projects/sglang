"""Cache a canonical checkpoint and rank-ready CPU weight image.

Preparation deliberately reuses the model's ordinary loader and quantization
hooks. CPU-rich delta preparation retains one canonical node-shared CPU
snapshot. That source is not CUDA-registered because it only feeds CPU layout
work. Preparation writes TP-sharded tensors into the persistent rank image,
then runs reload-safe post-load transforms on CPU. Quantization methods that
require device kernels retain the bounded GPU-staging path. The resulting
runtime bytes remain in
:class:`~sglang.srt.weight_sync.prepared_weights.PreparedWeightImage`; the live
model is never rebound or overwritten during preparation.

This is a full-target cache. It does not inspect which checkpoint tensors
changed and has no delta-format or sparsity-dependent path.
"""

from __future__ import annotations

import copy
import functools
import gc
import json
import logging
import math
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.utils import STABLE_CPU_WEIGHT_SOURCE_ATTR
from sglang.srt.weight_sync.host_shared_memory import HostSharedMemoryBuffer
from sglang.srt.weight_sync.prepared_weights import (
    PreparedWeightImage,
    iter_weight_tensors,
)

logger = logging.getLogger(__name__)

_POSITIONAL_IO_CHUNK_BYTES = 64 << 20


def _safetensors_dtypes() -> dict[str, torch.dtype]:
    result = {
        "BOOL": torch.bool,
        "I8": torch.int8,
        "U8": torch.uint8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
    }
    optional = {
        "U16": "uint16",
        "U32": "uint32",
        "U64": "uint64",
        "F8_E4M3": "float8_e4m3fn",
        "F8_E4M3FNUZ": "float8_e4m3fnuz",
        "F8_E5M2": "float8_e5m2",
        "F8_E5M2FNUZ": "float8_e5m2fnuz",
        "F4": "float4_e2m1fn_x2",
    }
    for code, name in optional.items():
        dtype = getattr(torch, name, None)
        if dtype is not None:
            result[code] = dtype
    return result


_SAFETENSORS_DTYPES = _safetensors_dtypes()


@dataclass(frozen=True)
class SafeTensorEntry:
    dtype: torch.dtype
    dtype_code: str
    shape: tuple[int, ...]
    relative_begin: int
    relative_end: int


@dataclass(frozen=True)
class SafeTensorsLayout:
    data_offset: int
    file_nbytes: int
    tensors: dict[str, SafeTensorEntry]


@dataclass
class CpuImageGroupLoad:
    """One CPU-loaded group awaiting its post-load transforms."""

    group_index: int
    group: WeightModuleGroup
    checkpoint_tensors: int
    transport: str
    cpu_shadow: torch.nn.Module
    gpu_stage_cpu_storages: set[tuple[int, int]]
    group_started: float
    cpu_clone_s: float
    restore_s: float
    cpu_load_s: float


class InMemorySafeTensorsFile:
    """Expose safetensors views from one bounded CPU or CUDA byte tensor.

    The header is parsed once from the CPU source. The same validated layout is
    then used for views into the one bulk CUDA copy, avoiding thousands of
    small CPU-to-GPU tensor transfers.
    """

    def __init__(
        self,
        buffer: torch.Tensor,
        *,
        layout: SafeTensorsLayout | None = None,
    ):
        if (
            buffer.device.type not in {"cpu", "cuda"}
            or buffer.dtype != torch.uint8
            or buffer.ndim != 1
            or not buffer.is_contiguous()
        ):
            raise ValueError("safetensors source must be contiguous CPU or CUDA bytes")
        if layout is None:
            if buffer.device.type != "cpu":
                raise ValueError("safetensors headers must be parsed from CPU bytes")
            layout = self._parse_layout(buffer)
        if layout.file_nbytes != buffer.numel():
            raise ValueError(
                "safetensors layout file size differs from source buffer: "
                f"layout={layout.file_nbytes} buffer={buffer.numel()}"
            )
        if layout.data_offset > buffer.numel():
            raise ValueError(
                "safetensors layout exceeds source buffer: "
                f"data_offset={layout.data_offset} file={buffer.numel()}"
            )
        self.buffer = buffer
        self.layout = layout
        self.data_offset = layout.data_offset
        self.tensors = layout.tensors

    @staticmethod
    def _parse_layout(buffer: torch.Tensor) -> SafeTensorsLayout:
        if buffer.numel() < 8:
            raise ValueError("safetensors source is shorter than its header prefix")
        prefix = buffer[:8].numpy().tobytes()
        header_nbytes = int.from_bytes(prefix, "little")
        data_offset = 8 + header_nbytes
        if header_nbytes <= 0 or data_offset > buffer.numel():
            raise ValueError(
                "invalid safetensors header length: "
                f"header={header_nbytes} file={buffer.numel()}"
            )
        try:
            header = json.loads(buffer[8:data_offset].numpy().tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid safetensors JSON header") from exc
        if not isinstance(header, dict):
            raise ValueError("safetensors header is not an object")
        header.pop("__metadata__", None)
        tensors = {}
        for name, metadata in header.items():
            if not isinstance(name, str) or not isinstance(metadata, dict):
                raise ValueError("invalid safetensors tensor metadata")
            dtype_code = metadata.get("dtype")
            dtype = _SAFETENSORS_DTYPES.get(dtype_code)
            if dtype is None:
                raise TypeError(f"unsupported safetensors dtype {dtype_code!r}")
            shape = metadata.get("shape")
            offsets = metadata.get("data_offsets")
            if (
                not isinstance(shape, list)
                or not all(isinstance(value, int) and value >= 0 for value in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
            ):
                raise ValueError(f"invalid safetensors metadata for {name!r}")
            relative_begin, relative_end = offsets
            begin = data_offset + relative_begin
            end = data_offset + relative_end
            if relative_begin < 0 or begin > end or end > buffer.numel():
                raise ValueError(
                    f"safetensors offsets are out of bounds for {name!r}: {offsets}"
                )
            shape_tuple = tuple(shape)
            expected_numel = math.prod(shape_tuple)
            expected_bytes = (
                (expected_numel + 1) // 2
                if dtype_code == "F4"
                else expected_numel * torch.empty((), dtype=dtype).element_size()
            )
            actual_bytes = relative_end - relative_begin
            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"tensor byte size mismatch for {name!r}: "
                    f"expected={expected_bytes} actual={actual_bytes}"
                )
            tensors[name] = SafeTensorEntry(
                dtype=dtype,
                dtype_code=dtype_code,
                shape=shape_tuple,
                relative_begin=relative_begin,
                relative_end=relative_end,
            )
        return SafeTensorsLayout(
            data_offset=data_offset,
            file_nbytes=buffer.numel(),
            tensors=tensors,
        )

    def get_tensor(self, name: str) -> torch.Tensor:
        entry = self.tensors.get(name)
        if entry is None:
            raise KeyError(f"safetensors source has no tensor {name!r}")
        begin = self.data_offset + entry.relative_begin
        end = self.data_offset + entry.relative_end
        source = self.buffer[begin:end]
        try:
            tensor = source.view(entry.dtype).reshape(entry.shape)
            # The node-shared source remains immutable until every local rank
            # crosses the reuse barrier. Native loaders may therefore batch
            # independent CPU copies without retaining a reused stream buffer.
            setattr(tensor, STABLE_CPU_WEIGHT_SOURCE_ATTR, True)
            return tensor
        except RuntimeError as exc:
            raise ValueError(f"cannot construct safetensors view for {name!r}") from exc

    def get_tensor_bytes(self, name: str) -> torch.Tensor:
        """Return the canonical encoded bytes for an in-place source transform."""

        entry = self.tensors.get(name)
        if entry is None:
            raise KeyError(f"safetensors source has no tensor {name!r}")
        begin = self.data_offset + entry.relative_begin
        end = self.data_offset + entry.relative_end
        return self.buffer[begin:end]


def _pread_file_to_tensor(
    path: Path,
    target: torch.Tensor,
    *,
    drop_cache_after_read: bool = False,
) -> float:
    if target.numel() != path.stat().st_size:
        raise ValueError(
            f"source buffer size mismatch for {path}: "
            f"buffer={target.numel()} file={path.stat().st_size}"
        )
    started = time.perf_counter()
    array = target.numpy()
    view = memoryview(array).cast("B")
    fd = os.open(path, os.O_RDONLY)
    offset = 0
    try:
        while offset < target.numel():
            end = min(offset + _POSITIONAL_IO_CHUNK_BYTES, target.numel())
            nread = os.preadv(fd, [view[offset:end]], offset)
            if nread <= 0:
                raise EOFError(
                    f"unexpected EOF reading {path}: "
                    f"offset={offset} size={target.numel()}"
                )
            offset += nread
    finally:
        os.close(fd)
        view.release()
    wall_s = time.perf_counter() - started
    if drop_cache_after_read and hasattr(os, "posix_fadvise"):
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            # Cache eviction is a memory-pressure optimization. Some remote or
            # sandboxed filesystems do not implement fadvise; correctness does
            # not depend on it.
            pass
    return wall_s


class HostSharedCheckpoint(HostSharedMemoryBuffer):
    """One canonical checkpoint mapping shared by local TP ranks."""

    def __init__(
        self,
        *,
        capacity: int,
        cpu_group: Any,
    ):
        setup_started = time.perf_counter()
        super().__init__(
            nbytes=capacity,
            cpu_group=cpu_group,
            name="weight-checkpoint",
        )
        self.capacity = self.nbytes
        self.setup_wall_s = time.perf_counter() - setup_started

    def stats(self) -> dict[str, Any]:
        return {
            "transport": "host_shared_checkpoint",
            "bytes": self.capacity,
            "physical_host_copies": 1,
            "wall_s": round(self.setup_wall_s, 6),
        }


@dataclass(frozen=True)
class WeightModuleGroup:
    """A model subtree small enough to compile without a second full model."""

    path: str
    nbytes: int


def _storage_key(tensor: torch.Tensor) -> tuple[int | None, int, int]:
    storage = tensor.untyped_storage()
    return tensor.device.index, storage.data_ptr(), storage.nbytes()


def _direct_weight_tensors(
    module: torch.nn.Module,
) -> Iterable[torch.Tensor]:
    yield from (value for value in module._parameters.values() if value is not None)
    yield from (value for value in module._buffers.values() if value is not None)
    get_extra = getattr(module, "prepared_weight_tensors", None)
    if get_extra is not None:
        for _, tensor in get_extra():
            yield tensor


def build_weight_module_groups(
    model: torch.nn.Module,
    *,
    max_group_bytes: int,
    device_type: str = "cuda",
) -> list[WeightModuleGroup]:
    """Partition the runtime module tree into bounded, storage-complete groups."""

    if max_group_bytes <= 0:
        raise ValueError("weight preparation group budget must be positive")

    subtree_keys: dict[str, set[tuple[int | None, int, int]]] = {}
    direct_keys: dict[str, set[tuple[int | None, int, int]]] = {}
    storage_nbytes: dict[tuple[int | None, int, int], int] = {}

    def collect(path: str, module: torch.nn.Module):
        direct: set[tuple[int | None, int, int]] = set()
        for tensor in _direct_weight_tensors(module):
            if tensor.device.type != device_type:
                continue
            key = _storage_key(tensor)
            direct.add(key)
            storage_nbytes[key] = key[2]
        direct_keys[path] = direct
        subtree = set(direct)
        prefix = f"{path}." if path else ""
        for child_name, child in module.named_children():
            subtree.update(collect(f"{prefix}{child_name}", child))
        subtree_keys[path] = subtree
        return subtree

    collect("", model)
    groups: list[WeightModuleGroup] = []

    def visit(path: str, module: torch.nn.Module) -> None:
        keys = subtree_keys[path]
        if not keys:
            return
        nbytes = sum(storage_nbytes[key] for key in keys)
        prefix = f"{path}." if path else ""
        children = [
            (f"{prefix}{name}", child)
            for name, child in module.named_children()
            if subtree_keys[f"{prefix}{name}"]
        ]
        if path and (nbytes <= max_group_bytes or not children):
            if nbytes > max_group_bytes:
                logger.warning(
                    "indivisible weight module exceeds preparation budget: "
                    "path=%s bytes=%d budget=%d",
                    path,
                    nbytes,
                    max_group_bytes,
                )
            groups.append(WeightModuleGroup(path=path, nbytes=nbytes))
            return
        if direct_keys[path]:
            raise ValueError(
                "cannot split a weight module that owns direct tensors and "
                f"child subtrees: path={path or '<root>'!r} bytes={nbytes} "
                f"budget={max_group_bytes}"
            )
        for child_path, child in children:
            visit(child_path, child)

    visit("", model)
    if not groups:
        raise ValueError(f"model has no {device_type} weight module groups")
    return groups


def _clone_tensor(
    tensor: torch.Tensor,
    tensor_memo: dict[int, torch.Tensor],
    storage_memo: dict[tuple[int | None, int, int], torch.Tensor],
    *,
    target_device: torch.device | None = None,
    copy_data: bool = True,
    storage_factory: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    cached = tensor_memo.get(id(tensor))
    if cached is not None:
        return cached

    key = _storage_key(tensor)
    storage_bytes = storage_memo.get(key)
    if storage_bytes is None:
        source = torch.empty(
            0,
            dtype=torch.uint8,
            device=tensor.device,
        ).set_(
            tensor.untyped_storage(),
            0,
            (tensor.untyped_storage().nbytes(),),
            (1,),
        )
        if storage_factory is not None:
            storage_bytes = storage_factory(tensor, source)
        else:
            device = tensor.device if target_device is None else target_device
            storage_bytes = torch.empty(
                source.numel(),
                dtype=torch.uint8,
                device=device,
            )
            if copy_data:
                storage_bytes.copy_(source, non_blocking=True)
        if (
            storage_bytes.dtype != torch.uint8
            or storage_bytes.ndim != 1
            or storage_bytes.numel() != source.numel()
        ):
            raise ValueError(
                "cloned storage must be a flat byte tensor with the source size"
            )
        storage_memo[key] = storage_bytes
    view = torch.empty(0, dtype=tensor.dtype, device=storage_bytes.device).set_(
        storage_bytes.untyped_storage(),
        storage_bytes.storage_offset() + tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )
    if isinstance(tensor, torch.nn.Parameter):
        cloned = type(tensor)._make_subclass(
            type(tensor),
            view,
            tensor.requires_grad,
        )
        if hasattr(tensor, "__dict__"):
            cloned.__dict__.update(tensor.__dict__)
    else:
        cloned = view.requires_grad_(tensor.requires_grad)
        if hasattr(tensor, "__dict__"):
            cloned.__dict__.update(tensor.__dict__)
    tensor_memo[id(tensor)] = cloned
    return cloned


def _clone_attribute(
    value: Any,
    tensor_memo: dict[int, torch.Tensor],
    storage_memo: dict[tuple[int | None, int, int], torch.Tensor],
    container_memo: dict[int, Any],
    *,
    target_device: torch.device | None,
    copy_data: bool,
    storage_factory: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
) -> Any:
    if isinstance(value, torch.Tensor):
        return _clone_tensor(
            value,
            tensor_memo,
            storage_memo,
            target_device=target_device,
            copy_data=copy_data,
            storage_factory=storage_factory,
        )
    cached = container_memo.get(id(value))
    if cached is _CONTAINER_CLONE_IN_PROGRESS:
        raise ValueError("cyclic immutable loader state cannot be cloned safely")
    if cached is not None:
        return cached
    if isinstance(value, dict):
        cloned = copy.copy(value)
        cloned.clear()
        container_memo[id(value)] = cloned
        cloned.update(
            (
                key,
                _clone_attribute(
                    child,
                    tensor_memo,
                    storage_memo,
                    container_memo,
                    target_device=target_device,
                    copy_data=copy_data,
                    storage_factory=storage_factory,
                ),
            )
            for key, child in value.items()
        )
        return cloned
    if isinstance(value, list):
        cloned = copy.copy(value)
        cloned.clear()
        container_memo[id(value)] = cloned
        cloned.extend(
            _clone_attribute(
                child,
                tensor_memo,
                storage_memo,
                container_memo,
                target_device=target_device,
                copy_data=copy_data,
                storage_factory=storage_factory,
            )
            for child in value
        )
        return cloned
    if isinstance(value, set):
        cloned = copy.copy(value)
        cloned.clear()
        container_memo[id(value)] = cloned
        cloned.update(
            _clone_attribute(
                child,
                tensor_memo,
                storage_memo,
                container_memo,
                target_device=target_device,
                copy_data=copy_data,
                storage_factory=storage_factory,
            )
            for child in value
        )
        return cloned
    if isinstance(value, (tuple, frozenset)):
        container_memo[id(value)] = _CONTAINER_CLONE_IN_PROGRESS
        children = [
            _clone_attribute(
                child,
                tensor_memo,
                storage_memo,
                container_memo,
                target_device=target_device,
                copy_data=copy_data,
                storage_factory=storage_factory,
            )
            for child in value
        ]
        if isinstance(value, frozenset):
            cloned = frozenset(children)
        elif hasattr(value, "_fields"):
            cloned = type(value)(*children)
        else:
            cloned = tuple(children)
        container_memo[id(value)] = cloned
        return cloned
    return value


_CONTAINER_CLONE_IN_PROGRESS = object()


def _rebind_cloned_method_owners(
    value: Any,
    owner_memo: dict[int, Any],
    value_memo: dict[int, Any],
) -> Any:
    cloned_value = owner_memo.get(id(value))
    if cloned_value is not None:
        return cloned_value
    owner = getattr(value, "__self__", None)
    function = getattr(value, "__func__", None)
    if owner is not None and function is not None:
        cloned_owner = owner_memo.get(id(owner))
        return value if cloned_owner is None else function.__get__(cloned_owner)

    cached = value_memo.get(id(value))
    if cached is _CONTAINER_CLONE_IN_PROGRESS:
        raise ValueError("cyclic immutable loader state cannot be rebound safely")
    if cached is not None:
        return cached
    if isinstance(value, functools.partial):
        cloned = functools.partial(
            _rebind_cloned_method_owners(value.func, owner_memo, value_memo),
            *(
                _rebind_cloned_method_owners(item, owner_memo, value_memo)
                for item in value.args
            ),
            **{
                key: _rebind_cloned_method_owners(item, owner_memo, value_memo)
                for key, item in (value.keywords or {}).items()
            },
        )
        value_memo[id(value)] = cloned
        cloned.__dict__.update(
            {
                key: _rebind_cloned_method_owners(item, owner_memo, value_memo)
                for key, item in value.__dict__.items()
            }
        )
        return cloned
    if isinstance(value, dict):
        cloned = copy.copy(value)
        cloned.clear()
        value_memo[id(value)] = cloned
        cloned.update(
            (
                key,
                _rebind_cloned_method_owners(item, owner_memo, value_memo),
            )
            for key, item in value.items()
        )
        return cloned
    if isinstance(value, list):
        cloned = copy.copy(value)
        cloned.clear()
        value_memo[id(value)] = cloned
        cloned.extend(
            _rebind_cloned_method_owners(item, owner_memo, value_memo) for item in value
        )
        return cloned
    if isinstance(value, set):
        cloned = copy.copy(value)
        cloned.clear()
        value_memo[id(value)] = cloned
        cloned.update(
            _rebind_cloned_method_owners(item, owner_memo, value_memo) for item in value
        )
        return cloned
    if isinstance(value, (tuple, frozenset)):
        value_memo[id(value)] = _CONTAINER_CLONE_IN_PROGRESS
        items = [
            _rebind_cloned_method_owners(item, owner_memo, value_memo) for item in value
        ]
        if isinstance(value, frozenset):
            cloned = frozenset(items)
        elif hasattr(value, "_fields"):
            cloned = type(value)(*items)
        else:
            cloned = tuple(items)
        value_memo[id(value)] = cloned
        return cloned
    return value


def clone_module_tensors(
    module: torch.nn.Module,
    *,
    target_device: torch.device | None = None,
    copy_data: bool = True,
    storage_factory: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.nn.Module:
    """Clone tensor state and mutable objects used by weight-loading hooks."""

    tensor_memo: dict[int, torch.Tensor] = {}
    storage_memo: dict[tuple[int | None, int, int], torch.Tensor] = {}
    container_memo: dict[int, Any] = {}
    loader_object_memo: dict[int, Any] = {}
    module_memo: dict[int, torch.nn.Module] = {}

    def clone(current: torch.nn.Module) -> torch.nn.Module:
        result = copy.copy(current)
        module_memo[id(current)] = result
        result._parameters = {
            name: (
                None
                if parameter is None
                else _clone_tensor(
                    parameter,
                    tensor_memo,
                    storage_memo,
                    target_device=target_device,
                    copy_data=copy_data,
                    storage_factory=storage_factory,
                )
            )
            for name, parameter in current._parameters.items()
        }
        result._buffers = {
            name: (
                None
                if buffer is None
                else _clone_tensor(
                    buffer,
                    tensor_memo,
                    storage_memo,
                    target_device=target_device,
                    copy_data=copy_data,
                    storage_factory=storage_factory,
                )
            )
            for name, buffer in current._buffers.items()
        }
        result._modules = {
            name: None if child is None else clone(child)
            for name, child in current._modules.items()
        }
        result._non_persistent_buffers_set = current._non_persistent_buffers_set.copy()
        for name, value in vars(current).items():
            if name in {"_parameters", "_buffers", "_modules"}:
                continue
            clone_for_preparation = getattr(
                value,
                "clone_for_weight_preparation",
                None,
            )
            if callable(clone_for_preparation):
                if id(value) not in loader_object_memo:
                    cloned_value = clone_for_preparation()
                    if cloned_value is value:
                        raise RuntimeError(
                            "clone_for_weight_preparation() returned the live object"
                        )
                    loader_object_memo[id(value)] = cloned_value
                result.__dict__[name] = loader_object_memo[id(value)]
                continue
            if name in {"quant_method", "scheme"} and value is not None:
                if id(value) not in loader_object_memo:
                    loader_object_memo[id(value)] = copy.copy(value)
                result.__dict__[name] = loader_object_memo[id(value)]
                continue
            result.__dict__[name] = _clone_attribute(
                value,
                tensor_memo,
                storage_memo,
                container_memo,
                target_device=target_device,
                copy_data=copy_data,
                storage_factory=storage_factory,
            )
        return result

    result = clone(module)
    owner_memo = {**module_memo, **loader_object_memo}
    value_memo: dict[int, Any] = {}
    for cloned_module in module_memo.values():
        for name, value in vars(cloned_module).items():
            if name in {"_parameters", "_buffers", "_modules"}:
                continue
            cloned_module.__dict__[name] = _rebind_cloned_method_owners(
                value,
                owner_memo,
                value_memo,
            )
    for cloned_object in loader_object_memo.values():
        if not hasattr(cloned_object, "__dict__"):
            continue
        for name, value in vars(cloned_object).items():
            cloned_object.__dict__[name] = _rebind_cloned_method_owners(
                value,
                owner_memo,
                value_memo,
            )
    for cloned_tensor in tensor_memo.values():
        for name, value in vars(cloned_tensor).items():
            cloned_tensor.__dict__[name] = _rebind_cloned_method_owners(
                value,
                owner_memo,
                value_memo,
            )
    return result


def clone_module_proxy(
    model: torch.nn.Module,
    path: str,
    *,
    target_device: torch.device | None = None,
    copy_data: bool = True,
    storage_factory: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Shallow-clone ancestors and tensor-clone only one selected subtree."""

    def clone_shell(module: torch.nn.Module) -> torch.nn.Module:
        result = copy.copy(module)
        result._parameters = module._parameters.copy()
        result._buffers = module._buffers.copy()
        result._modules = module._modules.copy()
        result._non_persistent_buffers_set = module._non_persistent_buffers_set.copy()
        for name in ("quant_method", "scheme"):
            value = getattr(module, name, None)
            if value is not None:
                result.__dict__[name] = copy.copy(value)
        for name, value in vars(module).items():
            clone_for_preparation = getattr(
                value,
                "clone_for_weight_preparation",
                None,
            )
            if callable(clone_for_preparation):
                result.__dict__[name] = clone_for_preparation()
        return result

    parts = path.split(".")
    live = model
    proxy = clone_shell(model)
    proxy_cursor = proxy
    for index, part in enumerate(parts):
        live_child = live._modules.get(part)
        if live_child is None:
            raise KeyError(f"module path {path!r} is missing component {part!r}")
        if index == len(parts) - 1:
            shadow = clone_module_tensors(
                live_child,
                target_device=target_device,
                copy_data=copy_data,
                storage_factory=storage_factory,
            )
            proxy_cursor._modules[part] = shadow
            return proxy, shadow
        proxy_child = clone_shell(live_child)
        proxy_cursor._modules[part] = proxy_child
        proxy_cursor = proxy_child
        live = live_child
    raise AssertionError("empty module path")


def _map_checkpoint_name(
    model: torch.nn.Module,
    name: str,
) -> str | None:
    """Apply the same authoritative name mapper as the ordinary loader."""

    mapper = getattr(model, "hf_to_sglang_mapper", None)
    return name if mapper is None else mapper._map_name(name)


def _longest_group_prefix(
    name: str,
    paths: set[str],
) -> str | None:
    parts = name.split(".")
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        if candidate in paths:
            return candidate
    return None


def map_checkpoint_names_to_groups(
    model: torch.nn.Module,
    names: Iterable[str],
    groups: list[WeightModuleGroup],
) -> dict[str, str | None]:
    """Map checkpoint tensors to the bounded runtime subtree that loads them."""

    paths = {group.path for group in groups}
    root_prefixes = {path.split(".", 1)[0] for path in paths}
    result: dict[str, str | None] = {}
    for name in names:
        mapped = _map_checkpoint_name(model, name)
        if mapped is None:
            result[name] = None
            continue
        direct = _longest_group_prefix(mapped, paths)
        if direct is not None:
            result[name] = direct
            continue

        # Some wrapper models delegate unprefixed checkpoint names into one
        # named runtime child. Only infer that wrapper prefix when the
        # authoritative mapped name has no direct runtime match.
        matches = {
            match
            for root in root_prefixes
            if (match := _longest_group_prefix(f"{root}.{mapped}", paths)) is not None
        }
        if len(matches) > 1:
            raise ValueError(f"ambiguous checkpoint group for {name!r}: {matches}")
        result[name] = next(iter(matches), None)
    return result


def _checkpoint_index(model_path: str) -> tuple[dict[str, str], Path]:
    root = Path(model_path)
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise ValueError(
            f"expected one safetensors index in {model_path!r}, found {indexes}"
        )
    payload = json.loads(indexes[0].read_text())
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid safetensors weight map: {indexes[0]}")
    return weight_map, root


def _canonical_checkpoint_layout(
    root: Path,
    filenames: list[str],
) -> tuple[
    dict[str, int],
    dict[str, int],
    int,
    tuple[str, tuple[tuple[str, int], ...]],
]:
    """Lay complete checkpoint files out once in a page-aligned CPU image."""

    file_sizes = {filename: (root / filename).stat().st_size for filename in filenames}
    offsets = {}
    capacity = 0
    for filename in filenames:
        capacity = (capacity + 4095) // 4096 * 4096
        offsets[filename] = capacity
        capacity += file_sizes[filename]
    capacity = (capacity + 4095) // 4096 * 4096
    signature = (
        os.path.realpath(root),
        tuple((filename, file_sizes[filename]) for filename in filenames),
    )
    return file_sizes, offsets, capacity, signature


class _IdentityCheckpointTransform:
    """Validate the persistent baseline without mutating its bytes."""

    base_version = 0

    @staticmethod
    def transform_file(
        filename: str,
        _tensor_file: InMemorySafeTensorsFile,
    ) -> dict[str, Any]:
        return {
            "operation": "canonical_baseline",
            "filename": filename,
            "changed_tensors": 0,
            "target_tensor_bytes": 0,
            "compressed_bytes": 0,
            "wall_s": 0.0,
        }


def _preapply_resident_checkpoint_transform(
    *,
    filenames: list[str],
    file_sizes: dict[str, int],
    offsets: dict[str, int],
    checkpoint: HostSharedCheckpoint,
    source_transform: Any,
    rank: int,
    world_size: int,
    cpu_group: Any,
) -> dict[str, Any]:
    """Advance and verify the complete canonical image before compiling it.

    The canonical files are disjoint, so local TP ranks can own a strided
    subset without locking. One collective publishes either complete success
    or every rank's error after all owned tensor checksums have run. Runtime
    layout compilation then consumes an immutable, fully verified snapshot;
    it does not need a collective at every checkpoint-file batch.
    """

    started = time.perf_counter()
    owned_transforms = []
    local_error = None
    try:
        for file_index, filename in enumerate(filenames):
            if file_index % world_size != rank:
                continue
            file_nbytes = file_sizes[filename]
            source_view = checkpoint.view(
                file_nbytes,
                offset=offsets[filename],
            )
            try:
                transform = source_transform.transform_file(
                    filename,
                    InMemorySafeTensorsFile(source_view),
                )
            finally:
                del source_view
            owned_transforms.append(transform)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    verify_started = time.perf_counter()
    if world_size > 1:
        errors: list[str | None] = [None] * world_size
        torch.distributed.all_gather_object(
            errors,
            local_error,
            group=cpu_group,
        )
    else:
        errors = [local_error]
    verify_barrier_s = time.perf_counter() - verify_started
    errors = [value for value in errors if value is not None]
    if errors:
        raise RuntimeError(
            "canonical checkpoint transform failed before runtime compilation: "
            + "; ".join(errors)
        )

    stats = {
        "operation": "preapply_resident_checkpoint_transform",
        "files": len(filenames),
        "owner_rank": rank,
        "owned_files": len(owned_transforms),
        "owned_transforms": owned_transforms,
        "transform_wall_s": round(
            sum(value.get("wall_s", 0.0) for value in owned_transforms),
            6,
        ),
        "verify_barrier_s": round(verify_barrier_s, 6),
        "wall_s": round(time.perf_counter() - started, 6),
    }
    logger.info(
        "Prepared canonical checkpoint on rank %d: files=%d "
        "transform_time=%.3fs synchronization_time=%.3fs wall_time=%.3fs",
        rank,
        len(owned_transforms),
        stats["transform_wall_s"],
        stats["verify_barrier_s"],
        stats["wall_s"],
    )
    return stats


class CPUWeightCache:
    """Compile indexed safetensors targets into complete pinned host images."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        max_group_bytes: int,
        host_cpu_group: Any = None,
    ):
        if getattr(model, "secondary_weights", None):
            raise NotImplementedError(
                "CPU weight preparation does not support models with secondary "
                "checkpoint sources; use the disk update path for this model"
            )
        self.model = model
        self.target_device = torch.device("cuda", torch.cuda.current_device())
        self.max_group_bytes = max_group_bytes
        self.host_cpu_group = host_cpu_group
        self.groups = build_weight_module_groups(
            model,
            max_group_bytes=max_group_bytes,
        )
        self._weight_preparation_device(model)
        self.image = PreparedWeightImage(model)
        self._prepare_stream = torch.cuda.Stream(device=self.target_device)
        self._canonical_checkpoint: HostSharedCheckpoint | None = None
        self._canonical_checkpoint_signature: (
            tuple[
                str,
                tuple[tuple[str, int], ...],
            ]
            | None
        ) = None
        self._canonical_lineage: tuple[str, str] | None = None
        self._canonical_checkpoint_version: int | None = None
        logger.info(
            "CPU weight cache layout: groups=%d storages=%d bytes=%d "
            "max_staging_bytes=%d",
            len(self.groups),
            len(self.image.segments),
            self.image.image_nbytes,
            self.max_group_bytes,
        )

    def _run_on_all_host_ranks(self, description: str, function: Callable[[], Any]):
        result = None
        error = None
        try:
            result = function()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        distributed = torch.distributed.is_initialized()
        world_size = (
            torch.distributed.get_world_size(group=self.host_cpu_group)
            if distributed
            else 1
        )
        if world_size > 1:
            errors: list[str | None] = [None] * world_size
            torch.distributed.all_gather_object(
                errors,
                error,
                group=self.host_cpu_group,
            )
        else:
            errors = [error]
        errors = [value for value in errors if value is not None]
        if errors:
            raise RuntimeError(f"{description} failed: " + "; ".join(errors))
        return result

    def initialize_host_cache(self, *, model_path: str) -> dict[str, Any]:
        """Allocate and populate persistent host state before replica readiness."""

        started = time.perf_counter()
        registration = self.image.register_host_memory()
        seed = self.image.capture_active(identity="")

        def inspect_checkpoint():
            weight_map, root = _checkpoint_index(model_path)
            filenames = sorted(set(weight_map.values()))
            return (root, filenames) + _canonical_checkpoint_layout(root, filenames)

        (
            root,
            filenames,
            file_sizes,
            offsets,
            capacity,
            signature,
        ) = self._run_on_all_host_ranks(
            "CPU weight cache checkpoint inspection",
            inspect_checkpoint,
        )
        checkpoint, created = self._get_canonical_checkpoint(
            capacity=capacity,
            signature=signature,
        )
        if not created:
            raise RuntimeError(
                "canonical checkpoint snapshot already exists during "
                "cache initialization"
            )
        canonical_snapshot = self._populate_canonical_checkpoint(
            root=root,
            filenames=filenames,
            file_sizes=file_sizes,
            offsets=offsets,
            checkpoint=checkpoint,
        )
        self._canonical_checkpoint_version = 0
        layout_warmup = self.prepare_from_checkpoint(
            model_path=model_path,
            identity="0",
            source_transform=_IdentityCheckpointTransform(),
        )
        validation = self.image.validate_against_active()
        self.image.accept_prepared_baseline()
        stats = {
            "operation": "initialize_host_cache",
            "canonical_checkpoint_bytes": sum(file_sizes.values()),
            "rank_image_bytes": self.image.image_nbytes,
            "registration_wall_s": registration["wall_s"],
            "capture_wall_s": seed["wall_s"],
            "canonical_load_wall_s": canonical_snapshot["wall_s"],
            "initial_compile_wall_s": layout_warmup["wall_s"],
            "validation_wall_s": validation["wall_s"],
            "wall_s": round(time.perf_counter() - started, 6),
        }
        logger.info(
            "CPU weight cache ready: rank_image_bytes=%d "
            "canonical_checkpoint_bytes=%d compile_time=%.3fs wall_time=%.3fs",
            self.image.image_nbytes,
            stats["canonical_checkpoint_bytes"],
            stats["initial_compile_wall_s"],
            stats["wall_s"],
        )
        return stats

    def _copy_shadow_to_image(
        self,
        path: str,
        shadow: torch.nn.Module,
    ) -> tuple[set[int], int]:
        updated: set[int] = set()
        copied_bytes = 0
        seen_shadow_storages: set[tuple[int | None, int, int]] = set()
        copies = []
        for relative_name, tensor in iter_weight_tensors(shadow):
            if tensor.device.type != "cuda":
                continue
            shadow_key = _storage_key(tensor)
            if shadow_key in seen_shadow_storages:
                continue
            seen_shadow_storages.add(shadow_key)
            full_name = f"{path}.{relative_name}" if relative_name else path
            segment = self.image.segments_by_name.get(full_name)
            if segment is None:
                raise RuntimeError(
                    f"prepared shadow produced unknown weight {full_name!r}"
                )
            source = torch.empty(
                0,
                dtype=torch.uint8,
                device=tensor.device,
            ).set_(
                tensor.untyped_storage(),
                0,
                (tensor.untyped_storage().nbytes(),),
                (1,),
            )
            copies.append((segment, source))
            updated.add(id(segment))
            copied_bytes += segment.nbytes
        self.image.copy_device_segments_to_image(copies)
        return updated, copied_bytes

    def _copy_cpu_shadow_to_image(
        self,
        path: str,
        shadow: torch.nn.Module,
    ) -> tuple[set[int], int, int]:
        """Publish rebound CPU tensors while retaining image-backed writes."""

        updated: set[int] = set()
        runtime_bytes = 0
        copied_bytes = 0
        seen_shadow_storages: set[tuple[int | None, int, int]] = set()
        for relative_name, tensor in iter_weight_tensors(shadow):
            if tensor.device.type != "cpu":
                continue
            shadow_key = _storage_key(tensor)
            if shadow_key in seen_shadow_storages:
                continue
            seen_shadow_storages.add(shadow_key)
            full_name = f"{path}.{relative_name}" if relative_name else path
            segment = self.image.segments_by_name.get(full_name)
            if segment is None:
                raise RuntimeError(
                    f"prepared shadow produced unknown weight {full_name!r}"
                )
            source = torch.empty(0, dtype=torch.uint8).set_(
                tensor.untyped_storage(),
                0,
                (tensor.untyped_storage().nbytes(),),
                (1,),
            )
            if source.numel() != segment.nbytes:
                raise RuntimeError(
                    "prepared shadow storage size changed: "
                    f"name={full_name!r} source={source.numel()} "
                    f"target={segment.nbytes}"
                )
            target = self.image.image[
                segment.image_offset : segment.image_offset + segment.nbytes
            ]
            if source.data_ptr() != target.data_ptr():
                target.copy_(source)
                copied_bytes += segment.nbytes
            updated.add(id(segment))
            runtime_bytes += segment.nbytes
        return updated, runtime_bytes, copied_bytes

    @staticmethod
    def _weight_preparation_device(shadow: torch.nn.Module) -> str:
        device = "cpu"
        for module_name, module in shadow.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is None:
                continue
            get_device = getattr(
                quant_method,
                "weight_preparation_device",
                None,
            )
            method_device = get_device(module) if callable(get_device) else None
            if method_device not in {"cpu", "cuda"}:
                raise NotImplementedError(
                    "CPU weight preparation is unsupported for quantization "
                    f"method {type(quant_method).__name__} at "
                    f"{module_name or '<root>'}"
                )
            if method_device == "cuda":
                device = "cuda"
        return device

    def _image_backed_cpu_storage(
        self,
        tensor: torch.Tensor,
        source_bytes: torch.Tensor,
    ) -> torch.Tensor:
        try:
            return self.image.storage_image_bytes(tensor)
        except KeyError:
            # Tensor attributes that are neither registered state nor declared
            # prepared weights may still carry loader metadata. Preserve those
            # in bounded CPU scratch; they are not copied into the final image.
            return source_bytes.to("cpu").clone()

    def _load_group_into_cpu_image(
        self,
        *,
        group_index: int,
        group: WeightModuleGroup,
        names: list[str],
        get_tensor: Callable[[str], torch.Tensor],
        transport: str,
    ) -> CpuImageGroupLoad:
        """Run the authoritative loader directly into one host-image range."""

        group_started = time.perf_counter()
        logger.debug(
            "Preparing CPU weight group %d/%d: path=%s "
            "estimated_bytes=%d checkpoint_tensors=%d transport=%s",
            group_index,
            len(self.groups),
            group.path,
            group.nbytes,
            len(names),
            transport,
        )

        gpu_stage_cpu_storages: set[tuple[int, int]] = set()

        def cpu_storage_factory(
            tensor: torch.Tensor,
            source_bytes: torch.Tensor,
        ) -> torch.Tensor:
            storage_bytes = self._image_backed_cpu_storage(tensor, source_bytes)
            if tensor.device.type == "cuda":
                storage = storage_bytes.untyped_storage()
                gpu_stage_cpu_storages.add(
                    (storage.data_ptr(), storage.nbytes()),
                )
            return storage_bytes

        phase_started = time.perf_counter()
        proxy, cpu_shadow = clone_module_proxy(
            self.model,
            group.path,
            target_device=torch.device("cpu"),
            copy_data=False,
            storage_factory=cpu_storage_factory,
        )
        cpu_clone_s = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        DefaultModelLoader.restore_weights_before_loading(
            cpu_shadow, torch.device("cpu")
        )
        restore_s = time.perf_counter() - phase_started

        weights = ((name, get_tensor(name)) for name in names)
        phase_started = time.perf_counter()
        proxy.load_weights(weights)
        cpu_load_s = time.perf_counter() - phase_started
        del proxy, weights

        return CpuImageGroupLoad(
            group_index=group_index,
            group=group,
            checkpoint_tensors=len(names),
            transport=transport,
            cpu_shadow=cpu_shadow,
            gpu_stage_cpu_storages=gpu_stage_cpu_storages,
            group_started=group_started,
            cpu_clone_s=cpu_clone_s,
            restore_s=restore_s,
            cpu_load_s=cpu_load_s,
        )

    def _finalize_cpu_image_group(
        self,
        loaded: CpuImageGroupLoad,
    ) -> tuple[set[int], int, dict[str, Any]]:
        """Run post-load transforms and write final runtime bytes to the image."""

        group = loaded.group
        cpu_shadow = loaded.cpu_shadow
        background_h2d_bytes = sum(
            nbytes for _, nbytes in loaded.gpu_stage_cpu_storages
        )
        postprocess_device = self._weight_preparation_device(cpu_shadow)
        if postprocess_device == "cpu":
            phase_started = time.perf_counter()
            for _, module in cpu_shadow.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)
            quant_submit_s = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            group_updated, group_bytes, cpu_image_copy_bytes = (
                self._copy_cpu_shadow_to_image(
                    group.path,
                    cpu_shadow,
                )
            )
            image_copy_s = time.perf_counter() - phase_started
            h2d_submit_s = 0.0
            device_sync_s = 0.0
            background_h2d_bytes = 0
            background_d2h_bytes = 0
            gpu_shadow = None
        else:

            def gpu_storage_factory(
                tensor: torch.Tensor,
                source_bytes: torch.Tensor,
            ) -> torch.Tensor:
                storage = tensor.untyped_storage()
                source_key = (storage.data_ptr(), storage.nbytes())
                target_device = (
                    self.target_device
                    if source_key in loaded.gpu_stage_cpu_storages
                    else tensor.device
                )
                storage_bytes = torch.empty(
                    source_bytes.numel(),
                    dtype=torch.uint8,
                    device=target_device,
                )
                storage_bytes.copy_(source_bytes, non_blocking=True)
                return storage_bytes

            with torch.cuda.stream(self._prepare_stream):
                phase_started = time.perf_counter()
                gpu_shadow = clone_module_tensors(
                    cpu_shadow,
                    target_device=self.target_device,
                    copy_data=True,
                    storage_factory=gpu_storage_factory,
                )
                h2d_submit_s = time.perf_counter() - phase_started

                phase_started = time.perf_counter()
                for _, module in gpu_shadow.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        quant_method.process_weights_after_loading(module)
                quant_submit_s = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            self._prepare_stream.synchronize()
            device_sync_s = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            group_updated, group_bytes = self._copy_shadow_to_image(
                group.path,
                gpu_shadow,
            )
            image_copy_s = time.perf_counter() - phase_started
            cpu_image_copy_bytes = 0
            background_d2h_bytes = group_bytes
        stats = {
            "path": group.path,
            "transport": loaded.transport,
            "checkpoint_tensors": loaded.checkpoint_tensors,
            "bytes": group_bytes,
            "postprocess_device": postprocess_device,
            "background_h2d_bytes": background_h2d_bytes,
            "background_d2h_bytes": background_d2h_bytes,
            "cpu_image_copy_bytes": cpu_image_copy_bytes,
            "cpu_clone_s": round(loaded.cpu_clone_s, 6),
            "restore_s": round(loaded.restore_s, 6),
            "cpu_load_s": round(loaded.cpu_load_s, 6),
            "h2d_submit_s": round(h2d_submit_s, 6),
            "quant_submit_s": round(quant_submit_s, 6),
            "device_sync_s": round(device_sync_s, 6),
            "image_copy_s": round(image_copy_s, 6),
            "wall_s": round(time.perf_counter() - loaded.group_started, 6),
        }
        logger.debug(
            "Prepared CPU weight group %d/%d: path=%s "
            "bytes=%d postprocess_device=%s wall_s=%.6f "
            "cpu_load_s=%.6f h2d_submit_s=%.6f "
            "device_sync_s=%.6f image_copy_s=%.6f transport=%s",
            loaded.group_index,
            len(self.groups),
            group.path,
            group_bytes,
            stats["postprocess_device"],
            stats["wall_s"],
            stats["cpu_load_s"],
            stats["h2d_submit_s"],
            stats["device_sync_s"],
            stats["image_copy_s"],
            loaded.transport,
        )
        del cpu_shadow, gpu_shadow
        gc.collect(0)
        return group_updated, group_bytes, stats

    def _compile_group_via_cpu_image(
        self,
        *,
        group_index: int,
        group: WeightModuleGroup,
        names: list[str],
        get_tensor: Callable[[str], torch.Tensor],
        transport: str,
    ) -> tuple[set[int], int, dict[str, Any]]:
        """Compile one group sequentially through the rank-local host image."""

        loaded = self._load_group_into_cpu_image(
            group_index=group_index,
            group=group,
            names=names,
            get_tensor=get_tensor,
            transport=transport,
        )
        return self._finalize_cpu_image_group(loaded)

    def _get_canonical_checkpoint(
        self,
        *,
        capacity: int,
        signature: tuple[str, tuple[tuple[str, int], ...]],
    ) -> tuple[HostSharedCheckpoint, bool]:
        if self._canonical_checkpoint is not None:
            if (
                self._canonical_checkpoint.capacity < capacity
                or self._canonical_checkpoint_signature != signature
            ):
                self._discard_canonical_checkpoint(
                    "canonical checkpoint layout changed",
                )
            else:
                return self._canonical_checkpoint, False
        self._canonical_checkpoint = HostSharedCheckpoint(
            capacity=capacity,
            cpu_group=self.host_cpu_group,
        )
        self._canonical_checkpoint_signature = signature
        self._canonical_checkpoint_version = None
        return self._canonical_checkpoint, True

    def _populate_canonical_checkpoint(
        self,
        *,
        root: Path,
        filenames: list[str],
        file_sizes: dict[str, int],
        offsets: dict[str, int],
        checkpoint: HostSharedCheckpoint,
    ) -> dict[str, Any]:
        """Read one complete canonical checkpoint into node-shared CPU memory."""

        started = time.perf_counter()
        distributed = torch.distributed.is_initialized()
        cpu_group = self.host_cpu_group
        world_size = (
            torch.distributed.get_world_size(group=cpu_group) if distributed else 1
        )
        rank = torch.distributed.get_rank(group=cpu_group) if distributed else 0
        owned_reads = []
        local_error = None
        try:
            for file_index, filename in enumerate(filenames):
                if file_index % world_size != rank:
                    continue
                file_nbytes = file_sizes[filename]
                wall_s = _pread_file_to_tensor(
                    root / filename,
                    checkpoint.view(
                        file_nbytes,
                        offset=offsets[filename],
                    ),
                    drop_cache_after_read=True,
                )
                owned_reads.append(
                    {
                        "filename": filename,
                        "bytes": file_nbytes,
                        "wall_s": round(wall_s, 6),
                    }
                )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"

        if world_size > 1:
            errors: list[str | None] = [None] * world_size
            torch.distributed.all_gather_object(
                errors,
                local_error,
                group=cpu_group,
            )
        else:
            errors = [local_error]
        errors = [error for error in errors if error is not None]
        if errors:
            raise RuntimeError(
                "failed to populate canonical CPU snapshot: " + "; ".join(errors)
            )

        stats = checkpoint.stats()
        stats.update(
            {
                "operation": "populate_canonical_cpu_snapshot",
                "persistent_canonical_snapshot": True,
                "base_version": 0,
                "files": len(filenames),
                "checkpoint_bytes": sum(file_sizes.values()),
                "owner_rank": rank,
                "owned_reads": owned_reads,
                "owned_bytes": sum(item["bytes"] for item in owned_reads),
                "wall_s": round(time.perf_counter() - started, 6),
            }
        )
        return stats

    def _discard_canonical_checkpoint(self, reason: str) -> None:
        checkpoint = self._canonical_checkpoint
        self._canonical_checkpoint = None
        self._canonical_checkpoint_signature = None
        self._canonical_lineage = None
        self._canonical_checkpoint_version = None
        gc.collect()
        if checkpoint is not None:
            checkpoint.close()
        logger.warning("Discarded canonical CPU checkpoint snapshot: %s", reason)

    def canonical_checkpoint_version(
        self,
        *,
        base_checkpoint_dir: str,
        source_dir: str,
    ) -> int:
        """Return a reusable canonical version, or reset another lineage."""

        if self._canonical_checkpoint is None:
            return 0
        requested_lineage = (
            os.path.realpath(base_checkpoint_dir),
            os.path.realpath(source_dir),
        )
        if (
            self._canonical_checkpoint_version == 0
            and self._canonical_checkpoint_signature is not None
            and self._canonical_checkpoint_signature[0] == requested_lineage[0]
        ):
            # The same immutable v0 checkpoint may anchor a new publisher run.
            self._canonical_lineage = requested_lineage
            return 0
        if self._canonical_lineage != requested_lineage:
            self._discard_canonical_checkpoint(
                "checkpoint lineage changed",
            )
            return 0
        return self._canonical_checkpoint_version or 0

    def canonical_checkpoint_root(self) -> str | None:
        if self._canonical_checkpoint_signature is None:
            return None
        return self._canonical_checkpoint_signature[0]

    def _compile_from_resident_checkpoint(
        self,
        *,
        root: Path,
        weight_map: dict[str, str],
        names_by_group: dict[str, list[str]],
        source_stats: list[dict[str, Any]],
        source_transform: Any,
    ):
        filenames = sorted(set(weight_map.values()))
        (
            file_sizes,
            offsets,
            capacity,
            signature,
        ) = self._run_on_all_host_ranks(
            "canonical checkpoint layout inspection",
            lambda: _canonical_checkpoint_layout(root, filenames),
        )
        checkpoint, created = self._get_canonical_checkpoint(
            capacity=capacity,
            signature=signature,
        )
        expected_version = int(getattr(source_transform, "base_version", 0))
        if created:
            source_stats.append(
                self._populate_canonical_checkpoint(
                    root=root,
                    filenames=filenames,
                    file_sizes=file_sizes,
                    offsets=offsets,
                    checkpoint=checkpoint,
                )
            )
            self._canonical_checkpoint_version = expected_version
        elif self._canonical_checkpoint_version != expected_version:
            raise RuntimeError(
                "canonical CPU checkpoint version mismatch: "
                f"snapshot={self._canonical_checkpoint_version} "
                f"delta_base={expected_version}"
            )

        distributed = torch.distributed.is_initialized()
        cpu_group = self.host_cpu_group
        world_size = (
            torch.distributed.get_world_size(group=cpu_group) if distributed else 1
        )
        rank = torch.distributed.get_rank(group=cpu_group) if distributed else 0
        source_stats.append(
            _preapply_resident_checkpoint_transform(
                filenames=filenames,
                file_sizes=file_sizes,
                offsets=offsets,
                checkpoint=checkpoint,
                source_transform=source_transform,
                rank=rank,
                world_size=world_size,
                cpu_group=cpu_group,
            )
        )

        handles = {
            filename: InMemorySafeTensorsFile(
                checkpoint.view(file_sizes[filename], offset=offsets[filename])
            )
            for filename in filenames
        }
        try:
            for group_index, group in enumerate(self.groups, start=1):
                yield self._compile_group_via_cpu_image(
                    group_index=group_index,
                    group=group,
                    names=names_by_group[group.path],
                    get_tensor=lambda name: handles[weight_map[name]].get_tensor(name),
                    transport="resident_canonical_cpu_checkpoint",
                )
        finally:
            handles.clear()
            gc.collect()

    def prepare_from_checkpoint(
        self,
        *,
        model_path: str,
        identity: str,
        source_transform: Any,
    ) -> dict[str, Any]:
        """Compile every bounded module from a complete canonical checkpoint."""

        started = time.perf_counter()

        def inspect_checkpoint():
            weight_map, root = _checkpoint_index(model_path)
            return (
                weight_map,
                root,
                map_checkpoint_names_to_groups(
                    self.model,
                    weight_map,
                    self.groups,
                ),
            )

        weight_map, root, group_for_name = self._run_on_all_host_ranks(
            "checkpoint inspection",
            inspect_checkpoint,
        )
        names_by_group: dict[str, list[str]] = {group.path: [] for group in self.groups}
        unmapped = []
        for name, group_path in group_for_name.items():
            if group_path is None:
                unmapped.append(name)
            else:
                names_by_group[group_path].append(name)
        if unmapped:
            raise RuntimeError(
                "CPU weight preparation cannot map every checkpoint tensor "
                f"to a runtime weight group; unmapped={unmapped[:20]}"
            )

        if not self.image.preparing and not self.image.valid:
            # Failed preparation may have partially overwritten the sole host
            # image, while the active CUDA model is still unchanged. Restore
            # the ordinary reload starting state before allowing a retry so
            # checkpoint-optional weights cannot inherit bytes from a failed
            # target.
            self.image.capture_active(identity="")
        self.image.begin_preparation(identity)
        updated_segments: set[int] = set()
        copied_bytes = 0
        group_stats = []
        source_stats = []
        try:
            self.image.register_host_memory()
            progress_interval = max(1, math.ceil(len(self.groups) / 10))
            for (
                group_updated,
                group_bytes,
                stats,
            ) in self._compile_from_resident_checkpoint(
                root=root,
                weight_map=weight_map,
                names_by_group=names_by_group,
                source_stats=source_stats,
                source_transform=source_transform,
            ):
                updated_segments.update(group_updated)
                copied_bytes += group_bytes
                group_stats.append(stats)
                completed_groups = len(group_stats)
                if (
                    completed_groups == 1
                    or completed_groups % progress_interval == 0
                    or completed_groups == len(self.groups)
                ):
                    logger.info(
                        "CPU weight image %s progress: groups=%d/%d "
                        "bytes=%d elapsed=%.3fs",
                        identity,
                        completed_groups,
                        len(self.groups),
                        copied_bytes,
                        time.perf_counter() - started,
                    )

            expected_segments = {id(value) for value in self.image.segments}
            missing = expected_segments - updated_segments
            if missing:
                missing_names = [
                    segment.name
                    for segment in self.image.segments
                    if id(segment) in missing
                ]
                raise RuntimeError(
                    "checkpoint did not produce every runtime weight "
                    f"storage; missing={missing_names[:20]}"
                )
            self.image.finish_preparation(identity)
        except Exception as exc:
            self.image.invalidate(
                f"checkpoint compilation of {identity!r} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            raise

        phase_totals = {
            phase: round(
                sum(value.get(phase, 0.0) for value in group_stats),
                6,
            )
            for phase in (
                "cpu_clone_s",
                "restore_s",
                "cpu_load_s",
                "h2d_submit_s",
                "quant_submit_s",
                "device_sync_s",
                "image_copy_s",
            )
        }
        traffic = {
            name: sum(value.get(name, 0) for value in group_stats)
            for name in (
                "background_h2d_bytes",
                "background_d2h_bytes",
                "cpu_image_copy_bytes",
            )
        }
        postprocess_devices = {
            device: sum(
                value["bytes"]
                for value in group_stats
                if value.get("postprocess_device") == device
            )
            for device in ("cpu", "cuda")
        }
        load_stats = [
            value
            for value in source_stats
            if value.get("operation") == "populate_canonical_cpu_snapshot"
        ]
        transform_stats = next(
            value
            for value in source_stats
            if value.get("operation") == "preapply_resident_checkpoint_transform"
        )
        source_summary = {
            "checkpoint_bytes": (
                self._canonical_checkpoint.capacity
                if self._canonical_checkpoint is not None
                else 0
            ),
            "loaded_from_disk": bool(load_stats),
            "load_wall_s": round(
                sum(value["wall_s"] for value in load_stats),
                6,
            ),
            "transform_wall_s": transform_stats["transform_wall_s"],
            "synchronization_wall_s": transform_stats["verify_barrier_s"],
        }
        wall_s = round(time.perf_counter() - started, 6)
        logger.info(
            "Prepared CPU weight image %s: bytes=%d wall_time=%.3fs "
            "source=%s phases=%s",
            identity,
            copied_bytes,
            wall_s,
            source_summary,
            phase_totals,
        )
        return {
            "operation": "prepare_cpu_weights",
            "target": identity,
            "groups": len(self.groups),
            "checkpoint_tensors": len(weight_map),
            "unmapped_checkpoint_tensors": len(unmapped),
            "runtime_storages": len(updated_segments),
            "bytes": copied_bytes,
            "transport": "resident_canonical_cpu_checkpoint",
            "wall_s": wall_s,
            "source": source_summary,
            "phases": phase_totals,
            "postprocess_bytes": postprocess_devices,
            "traffic": traffic,
        }

    def prepare_from_delta_lineage(
        self,
        *,
        base_checkpoint_dir: str,
        source_dir: str,
        target_version: int,
        identity: str,
    ) -> dict[str, Any]:
        """Reconstruct and compile a target without materializing it on disk."""

        from sglang.srt.weight_sync.delta_checkpoint import (
            DeltaCheckpointOverlay,
            validate_delta_target,
        )

        validate_delta_target(source_dir, target_version)
        base_version = self.canonical_checkpoint_version(
            base_checkpoint_dir=base_checkpoint_dir,
            source_dir=source_dir,
        )
        if target_version < base_version:
            self._discard_canonical_checkpoint(
                f"requested rollback from v{base_version} to v{target_version}",
            )
            base_version = 0
        try:
            with DeltaCheckpointOverlay(
                base_checkpoint_dir=base_checkpoint_dir,
                source_dir=source_dir,
                target_version=target_version,
                cpu_group=self.host_cpu_group,
                base_version=base_version,
                current_checkpoint_dir=self.canonical_checkpoint_root(),
            ) as source:
                stage_stats = source.stage_stats
                stats = self.prepare_from_checkpoint(
                    model_path=str(source.checkpoint_root),
                    identity=identity,
                    source_transform=source,
                )
            self._canonical_checkpoint_version = (
                target_version if self._canonical_checkpoint is not None else None
            )
            if self._canonical_checkpoint is not None:
                self._canonical_lineage = (
                    os.path.realpath(base_checkpoint_dir),
                    os.path.realpath(source_dir),
                )
        except Exception:
            # A transform or loader failure may leave an unknown subset of the
            # canonical bytes advanced. Fail closed: the live GPU model remains
            # untouched, and a retry reconstructs from the immutable local base.
            self._discard_canonical_checkpoint(
                f"preparation of v{target_version} did not complete",
            )
            raise
        stats["delta_checkpoint"] = stage_stats
        stats["resident_canonical_snapshot"] = {
            "version": self._canonical_checkpoint_version,
            "bytes": (
                0
                if self._canonical_checkpoint is None
                else self._canonical_checkpoint.capacity
            ),
            "physical_host_copies": 1,
        }
        stats["wall_s_including_delta_stage"] = round(
            stats["wall_s"] + stage_stats["wall_s"],
            6,
        )
        return stats

    def commit(
        self,
        expected_identity: str,
    ) -> dict[str, float | int | str]:
        return self.image.commit(expected_identity)
