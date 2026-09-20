"""Bounded, isolated module views for native weight loaders."""

from __future__ import annotations

import copy
import functools
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch

from sglang.srt.weight_sync.rank_weight_image import iter_derived_weight_tensors

logger = logging.getLogger(__name__)

_COPY_IN_PROGRESS = object()


@dataclass(frozen=True)
class WeightLoadGroup:
    """One storage-complete model subtree compiled as a bounded unit."""

    path: str
    nbytes: int


def _storage_key(tensor: torch.Tensor) -> tuple[int | None, int, int]:
    storage = tensor.untyped_storage()
    return tensor.device.index, storage.data_ptr(), storage.nbytes()


def _direct_weight_tensors(module: torch.nn.Module) -> Iterable[torch.Tensor]:
    yield from (value for value in module._parameters.values() if value is not None)
    yield from (
        value
        for name, value in module._buffers.items()
        if value is not None and name not in module._non_persistent_buffers_set
    )
    yield from (tensor for _, tensor in iter_derived_weight_tensors(module))


def build_weight_load_groups(
    model: torch.nn.Module,
    *,
    max_group_bytes: int,
    device_type: str = "cuda",
) -> list[WeightLoadGroup]:
    """Partition a model into storage-complete bounded loader units."""

    if max_group_bytes <= 0:
        raise ValueError("weight load group budget must be positive")

    direct_weight_keys: dict[str, set[tuple[int | None, int, int]]] = {}
    subtree_weight_keys: dict[str, set[tuple[int | None, int, int]]] = {}
    storage_nbytes: dict[tuple[int | None, int, int], int] = {}

    def collect(path: str, module: torch.nn.Module):
        direct_weights = set()
        for tensor in _direct_weight_tensors(module):
            if tensor.device.type != device_type:
                continue
            key = _storage_key(tensor)
            if key[2] == 0:
                continue
            direct_weights.add(key)
            storage_nbytes[key] = key[2]
        direct_weight_keys[path] = direct_weights

        subtree_weights = set(direct_weights)
        prefix = f"{path}." if path else ""
        for child_name, child in module._modules.items():
            if child is not None:
                child_weights = collect(f"{prefix}{child_name}", child)
                subtree_weights.update(child_weights)
        subtree_weight_keys[path] = subtree_weights
        return subtree_weights

    collect("", model)
    groups: list[WeightLoadGroup] = []

    def partition(path: str, module: torch.nn.Module) -> None:
        weight_keys = subtree_weight_keys[path]
        if not weight_keys:
            return
        nbytes = sum(storage_nbytes[key] for key in weight_keys)
        prefix = f"{path}." if path else ""
        children = [
            (f"{prefix}{name}", child)
            for name, child in module._modules.items()
            if child is not None and subtree_weight_keys[f"{prefix}{name}"]
        ]
        indivisible = bool(getattr(module, "weight_load_indivisible", False))
        if indivisible or nbytes <= max_group_bytes or not children:
            if nbytes > max_group_bytes:
                logger.warning(
                    "Indivisible weight load group exceeds its byte budget: "
                    "path=%s bytes=%d budget=%d",
                    path or "<root>",
                    nbytes,
                    max_group_bytes,
                )
            groups.append(WeightLoadGroup(path=path, nbytes=nbytes))
            return
        if direct_weight_keys[path]:
            raise ValueError(
                "cannot split a module that owns tensor state and child weight "
                f"subtrees: path={path or '<root>'!r} bytes={nbytes} "
                f"budget={max_group_bytes}"
            )
        for child_path, child in children:
            partition(child_path, child)

    partition("", model)
    if not groups:
        raise ValueError(f"model has no {device_type} weight load groups")

    owners: dict[tuple[int | None, int, int], str] = {}
    for group in groups:
        for key in subtree_weight_keys[group.path]:
            previous = owners.setdefault(key, group.path)
            if previous != group.path:
                raise ValueError(
                    "weight storage spans independent load groups: "
                    f"{previous!r} and {group.path!r}"
                )
    return groups


class _ModuleCopy:
    def __init__(
        self,
        *,
        target_device: torch.device | None,
        copy_data: bool,
        storage_factory: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
    ):
        self.target_device = target_device
        self.copy_data = copy_data
        self.storage_factory = storage_factory
        self.tensors: dict[int, torch.Tensor] = {}
        self.storages: dict[tuple[int | None, int, int], torch.Tensor] = {}
        self.modules: dict[int, torch.nn.Module] = {}
        self.objects: dict[int, Any] = {}
        self.containers: dict[int, Any] = {}
        self.tensor_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def tensor(self, source: torch.Tensor) -> torch.Tensor:
        cached = self.tensors.get(id(source))
        if cached is not None:
            return cached

        key = _storage_key(source)
        storage_bytes = self.storages.get(key)
        if storage_bytes is None:
            source_bytes = torch.empty(0, dtype=torch.uint8, device=source.device).set_(
                source.untyped_storage(),
                0,
                (source.untyped_storage().nbytes(),),
                (1,),
            )
            if self.storage_factory is not None:
                storage_bytes = self.storage_factory(source, source_bytes)
            else:
                device = (
                    source.device if self.target_device is None else self.target_device
                )
                storage_bytes = torch.empty(
                    source_bytes.numel(), dtype=torch.uint8, device=device
                )
                if self.copy_data:
                    storage_bytes.copy_(source_bytes, non_blocking=True)
            if (
                storage_bytes.dtype != torch.uint8
                or storage_bytes.ndim != 1
                or storage_bytes.numel() != source_bytes.numel()
            ):
                raise ValueError(
                    "replacement storage must be a flat byte tensor with the "
                    "source storage size"
                )
            self.storages[key] = storage_bytes

        byte_offset = storage_bytes.storage_offset()
        if byte_offset % source.element_size():
            raise ValueError(
                "replacement storage offset is not aligned for tensor dtype: "
                f"offset={byte_offset} dtype={source.dtype}"
            )
        view = torch.empty(0, dtype=source.dtype, device=storage_bytes.device).set_(
            storage_bytes.untyped_storage(),
            byte_offset // source.element_size() + source.storage_offset(),
            tuple(source.shape),
            tuple(source.stride()),
        )
        if isinstance(source, torch.nn.Parameter):
            copied = type(source)._make_subclass(
                type(source), view, source.requires_grad
            )
        else:
            copied = view.requires_grad_(source.requires_grad)
        self.tensors[id(source)] = copied
        self.tensor_pairs.append((source, copied))
        return copied

    @staticmethod
    def _is_mutable_loader_state(value: Any) -> bool:
        return any(
            callable(getattr(value, name, None))
            for name in (
                "process_weights_after_loading",
                "restore_weights_before_loading",
                "set_quant_config",
            )
        )

    def value(self, source: Any) -> Any:
        if isinstance(source, torch.Tensor):
            return self.tensor(source)
        if isinstance(source, torch.nn.Module):
            return self.module(source)

        cached = self.objects.get(id(source))
        if cached is not None:
            return cached
        cached = self.containers.get(id(source))
        if cached is _COPY_IN_PROGRESS:
            raise ValueError("cyclic immutable loader state cannot be copied safely")
        if cached is not None:
            return cached

        if self._is_mutable_loader_state(source):
            copied = copy.copy(source)
            self.objects[id(source)] = copied
            if hasattr(source, "__dict__"):
                copied.__dict__.update(
                    {name: self.value(value) for name, value in vars(source).items()}
                )
            return copied
        if isinstance(source, dict):
            copied = copy.copy(source)
            copied.clear()
            self.containers[id(source)] = copied
            copied.update((key, self.value(value)) for key, value in source.items())
            return copied
        if isinstance(source, list):
            copied = copy.copy(source)
            copied.clear()
            self.containers[id(source)] = copied
            copied.extend(self.value(value) for value in source)
            return copied
        if isinstance(source, set):
            copied = copy.copy(source)
            copied.clear()
            self.containers[id(source)] = copied
            copied.update(self.value(value) for value in source)
            return copied
        if isinstance(source, (tuple, frozenset)):
            self.containers[id(source)] = _COPY_IN_PROGRESS
            values = [self.value(value) for value in source]
            if isinstance(source, frozenset):
                copied = frozenset(values)
            elif hasattr(source, "_fields"):
                copied = type(source)(*values)
            else:
                copied = tuple(values)
            self.containers[id(source)] = copied
            return copied
        return source

    def module(self, source: torch.nn.Module) -> torch.nn.Module:
        cached = self.modules.get(id(source))
        if cached is not None:
            return cached

        copied = copy.copy(source)
        self.modules[id(source)] = copied
        copied._parameters = {
            name: None if tensor is None else self.tensor(tensor)
            for name, tensor in source._parameters.items()
        }
        # Non-persistent buffers are runtime state, not checkpoint state. A
        # model or quantization method must explicitly expose any buffer whose
        # value is derived from weights and therefore belongs in the image.
        derived_tensor_ids = {
            id(tensor) for _, tensor in iter_derived_weight_tensors(source)
        }
        copied._buffers = {}
        for name, tensor in source._buffers.items():
            if tensor is None:
                copied._buffers[name] = None
            elif (
                name in source._non_persistent_buffers_set
                and id(tensor) not in derived_tensor_ids
            ):
                copied._buffers[name] = tensor
            else:
                copied._buffers[name] = self.tensor(tensor)
        copied._modules = {
            name: None if child is None else self.module(child)
            for name, child in source._modules.items()
        }
        copied._non_persistent_buffers_set = source._non_persistent_buffers_set.copy()
        for name, value in vars(source).items():
            if name not in {"_parameters", "_buffers", "_modules"}:
                copied.__dict__[name] = self.value(value)
        return copied

    def finish(self, source: torch.nn.Module) -> torch.nn.Module:
        copied = self.module(source)

        # Parameter loaders often retain bound methods and tensor aliases in
        # ordinary tensor attributes. Copy those after the module graph exists
        # so every owner can be rebound to its isolated counterpart.
        index = 0
        while index < len(self.tensor_pairs):
            source_tensor, copied_tensor = self.tensor_pairs[index]
            index += 1
            if hasattr(source_tensor, "__dict__"):
                copied_tensor.__dict__.update(
                    {
                        name: self.value(value)
                        for name, value in vars(source_tensor).items()
                    }
                )

        owners = {**self.modules, **self.objects}
        memo: dict[int, Any] = {}
        for owner in [
            *self.modules.values(),
            *self.objects.values(),
            *self.tensors.values(),
        ]:
            if not hasattr(owner, "__dict__"):
                continue
            for name, value in vars(owner).items():
                owner.__dict__[name] = _rebind_methods(value, owners, memo)
        return copied


def _rebind_methods(value: Any, owners: dict[int, Any], memo: dict[int, Any]) -> Any:
    copied_owner = owners.get(id(value))
    if copied_owner is not None:
        return copied_owner

    owner = getattr(value, "__self__", None)
    function = getattr(value, "__func__", None)
    if owner is not None and function is not None:
        copied_owner = owners.get(id(owner))
        return value if copied_owner is None else function.__get__(copied_owner)

    cached = memo.get(id(value))
    if cached is _COPY_IN_PROGRESS:
        raise ValueError("cyclic immutable loader state cannot be rebound safely")
    if cached is not None:
        return cached
    if isinstance(value, functools.partial):
        copied = functools.partial(
            _rebind_methods(value.func, owners, memo),
            *(_rebind_methods(item, owners, memo) for item in value.args),
            **{
                key: _rebind_methods(item, owners, memo)
                for key, item in (value.keywords or {}).items()
            },
        )
        memo[id(value)] = copied
        copied.__dict__.update(
            {
                key: _rebind_methods(item, owners, memo)
                for key, item in value.__dict__.items()
            }
        )
        return copied
    if isinstance(value, dict):
        copied = copy.copy(value)
        copied.clear()
        memo[id(value)] = copied
        copied.update(
            (key, _rebind_methods(item, owners, memo)) for key, item in value.items()
        )
        return copied
    if isinstance(value, list):
        copied = copy.copy(value)
        copied.clear()
        memo[id(value)] = copied
        copied.extend(_rebind_methods(item, owners, memo) for item in value)
        return copied
    if isinstance(value, set):
        copied = copy.copy(value)
        copied.clear()
        memo[id(value)] = copied
        copied.update(_rebind_methods(item, owners, memo) for item in value)
        return copied
    if isinstance(value, (tuple, frozenset)):
        memo[id(value)] = _COPY_IN_PROGRESS
        values = [_rebind_methods(item, owners, memo) for item in value]
        if isinstance(value, frozenset):
            copied = frozenset(values)
        elif hasattr(value, "_fields"):
            copied = type(value)(*values)
        else:
            copied = tuple(values)
        memo[id(value)] = copied
        return copied
    return value


def clone_module_for_weight_loading(
    module: torch.nn.Module,
    *,
    target_device: torch.device | None = None,
    copy_data: bool = True,
    storage_factory: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.nn.Module:
    """Clone module state that native load and post-load code may mutate."""

    return _ModuleCopy(
        target_device=target_device,
        copy_data=copy_data,
        storage_factory=storage_factory,
    ).finish(module)


def build_weight_loader_view(
    model: torch.nn.Module,
    path: str,
    *,
    target_device: torch.device | None = None,
    copy_data: bool = True,
    storage_factory: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Build a model-shaped loader view with one isolated target subtree."""

    if not path:
        shadow = clone_module_for_weight_loading(
            model,
            target_device=target_device,
            copy_data=copy_data,
            storage_factory=storage_factory,
        )
        return shadow, shadow

    def copy_shell(module: torch.nn.Module) -> torch.nn.Module:
        copied = copy.copy(module)
        copied._parameters = module._parameters.copy()
        copied._buffers = module._buffers.copy()
        copied._modules = module._modules.copy()
        copied._non_persistent_buffers_set = module._non_persistent_buffers_set.copy()
        for name, value in vars(module).items():
            if _ModuleCopy._is_mutable_loader_state(value):
                copied.__dict__[name] = copy.copy(value)
        return copied

    parts = path.split(".")
    live = model
    view = copy_shell(model)
    cursor = view
    for index, part in enumerate(parts):
        child = live._modules.get(part)
        if child is None:
            raise KeyError(f"module path {path!r} is missing component {part!r}")
        if index == len(parts) - 1:
            shadow = clone_module_for_weight_loading(
                child,
                target_device=target_device,
                copy_data=copy_data,
                storage_factory=storage_factory,
            )
            cursor._modules[part] = shadow
            return view, shadow
        shell = copy_shell(child)
        cursor._modules[part] = shell
        cursor = shell
        live = child
    raise AssertionError("empty module path")
