"""Bounded staging clones for SGLang model-specific weight loaders."""

from __future__ import annotations

import copy
import functools
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)

_CLONE_IN_PROGRESS = object()


@dataclass(frozen=True)
class WeightModuleGroup:
    """A model subtree small enough to compile without a second full model."""

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
    get_extra = getattr(module, "get_additional_weight_tensors", None)
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
        raise ValueError("weight compilation group budget must be positive")

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
        for child_name, child in module._modules.items():
            if child is not None:
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
            for name, child in module._modules.items()
            if child is not None and subtree_keys[f"{prefix}{name}"]
        ]
        indivisible = bool(getattr(module, "weight_staging_indivisible", False))
        if indivisible or nbytes <= max_group_bytes or not children:
            if nbytes > max_group_bytes:
                logger.warning(
                    "Indivisible weight module exceeds compilation budget: "
                    "path=%s bytes=%d budget=%d",
                    path or "<root>",
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

    owners: dict[tuple[int | None, int, int], str] = {}
    for group in groups:
        for key in subtree_keys[group.path]:
            previous = owners.setdefault(key, group.path)
            if previous != group.path:
                raise ValueError(
                    "weight storage spans independent compilation groups: "
                    f"{previous!r} and {group.path!r}; use the disk update path "
                    "for this model"
                )
    return groups


def _clone_tensor(
    tensor: torch.Tensor,
    tensor_memo: dict[int, torch.Tensor],
    storage_memo: dict[tuple[int | None, int, int], torch.Tensor],
    tensor_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    target_device: torch.device | None,
    copy_data: bool,
    storage_factory: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor:
    cached = tensor_memo.get(id(tensor))
    if cached is not None:
        return cached

    key = _storage_key(tensor)
    storage_bytes = storage_memo.get(key)
    if storage_bytes is None:
        source = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
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

    storage_byte_offset = storage_bytes.storage_offset()
    if storage_byte_offset % tensor.element_size():
        raise ValueError(
            "cloned byte storage offset is not aligned for the tensor dtype: "
            f"offset={storage_byte_offset} dtype={tensor.dtype}"
        )
    view = torch.empty(0, dtype=tensor.dtype, device=storage_bytes.device).set_(
        storage_bytes.untyped_storage(),
        storage_byte_offset // tensor.element_size() + tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )
    if isinstance(tensor, torch.nn.Parameter):
        cloned = type(tensor)._make_subclass(
            type(tensor),
            view,
            tensor.requires_grad,
        )
    else:
        cloned = view.requires_grad_(tensor.requires_grad)
    tensor_memo[id(tensor)] = cloned
    tensor_pairs.append((tensor, cloned))
    return cloned


def _clone_attribute(
    value: Any,
    tensor_memo: dict[int, torch.Tensor],
    storage_memo: dict[tuple[int | None, int, int], torch.Tensor],
    tensor_pairs: list[tuple[torch.Tensor, torch.Tensor]],
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
            tensor_pairs,
            target_device=target_device,
            copy_data=copy_data,
            storage_factory=storage_factory,
        )

    cached = container_memo.get(id(value))
    if cached is _CLONE_IN_PROGRESS:
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
                    tensor_pairs,
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
                tensor_pairs,
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
                tensor_pairs,
                container_memo,
                target_device=target_device,
                copy_data=copy_data,
                storage_factory=storage_factory,
            )
            for child in value
        )
        return cloned
    if isinstance(value, (tuple, frozenset)):
        container_memo[id(value)] = _CLONE_IN_PROGRESS
        children = [
            _clone_attribute(
                child,
                tensor_memo,
                storage_memo,
                tensor_pairs,
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
    if cached is _CLONE_IN_PROGRESS:
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
        value_memo[id(value)] = _CLONE_IN_PROGRESS
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


def clone_weight_module(
    module: torch.nn.Module,
    *,
    target_device: torch.device | None = None,
    copy_data: bool = True,
    storage_factory: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.nn.Module:
    """Clone tensor state and mutable objects used by weight-loading hooks."""

    tensor_memo: dict[int, torch.Tensor] = {}
    storage_memo: dict[tuple[int | None, int, int], torch.Tensor] = {}
    tensor_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    container_memo: dict[int, Any] = {}
    loader_object_memo: dict[int, Any] = {}
    module_memo: dict[int, torch.nn.Module] = {}

    def clone(current: torch.nn.Module) -> torch.nn.Module:
        cached = module_memo.get(id(current))
        if cached is not None:
            return cached
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
                    tensor_pairs,
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
                    tensor_pairs,
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
            clone_for_staging = getattr(value, "clone_for_weight_staging", None)
            if callable(clone_for_staging):
                if id(value) not in loader_object_memo:
                    cloned_value = clone_for_staging()
                    if cloned_value is value:
                        raise RuntimeError(
                            "clone_for_weight_staging() returned the live object"
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
                tensor_pairs,
                container_memo,
                target_device=target_device,
                copy_data=copy_data,
                storage_factory=storage_factory,
            )
        return result

    result = clone(module)
    # Parameter loaders and quantization methods keep tensors in custom tensor
    # attributes. Clone them after registered state has populated the tensor
    # memo so aliases remain intact without sharing live storage.
    pair_index = 0
    while pair_index < len(tensor_pairs):
        source, cloned = tensor_pairs[pair_index]
        pair_index += 1
        if not hasattr(source, "__dict__"):
            continue
        cloned.__dict__.update(
            {
                name: _clone_attribute(
                    value,
                    tensor_memo,
                    storage_memo,
                    tensor_pairs,
                    container_memo,
                    target_device=target_device,
                    copy_data=copy_data,
                    storage_factory=storage_factory,
                )
                for name, value in vars(source).items()
            }
        )

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


def build_weight_loader_proxy(
    model: torch.nn.Module,
    path: str,
    *,
    target_device: torch.device | None = None,
    copy_data: bool = True,
    storage_factory: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Build a loader proxy with one isolated, tensor-cloned subtree."""

    if not path:
        shadow = clone_weight_module(
            model,
            target_device=target_device,
            copy_data=copy_data,
            storage_factory=storage_factory,
        )
        return shadow, shadow

    def clone_shell(module: torch.nn.Module) -> torch.nn.Module:
        result = copy.copy(module)
        result._parameters = module._parameters.copy()
        result._buffers = module._buffers.copy()
        result._modules = module._modules.copy()
        result._non_persistent_buffers_set = module._non_persistent_buffers_set.copy()
        object_memo: dict[int, Any] = {}
        for name, value in vars(module).items():
            clone_for_staging = getattr(value, "clone_for_weight_staging", None)
            if callable(clone_for_staging):
                if id(value) not in object_memo:
                    object_memo[id(value)] = clone_for_staging()
                result.__dict__[name] = object_memo[id(value)]
            elif name in {"quant_method", "scheme"} and value is not None:
                if id(value) not in object_memo:
                    object_memo[id(value)] = copy.copy(value)
                result.__dict__[name] = object_memo[id(value)]
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
            shadow = clone_weight_module(
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


def _map_checkpoint_name(model: torch.nn.Module, name: str) -> str | None:
    """Map one checkpoint name, returning ``None`` for ignored weights."""

    mapper = getattr(model, "weight_update_name_mapper", None)
    if mapper is None:
        mapper = getattr(model, "hf_to_sglang_mapper", None)
    return name if mapper is None else mapper._map_name(name)


def filter_ignored_checkpoint_weights(
    model: torch.nn.Module,
    weight_map: dict[str, str],
) -> dict[str, str]:
    """Remove tensors that the model's authoritative loader contract ignores."""

    return {
        name: filename
        for name, filename in weight_map.items()
        if _map_checkpoint_name(model, name) is not None
    }


def _longest_group_prefix(name: str, paths: set[str]) -> str | None:
    parts = name.split(".")
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        if candidate in paths:
            return candidate
    return "" if "" in paths else None


def _groups_after_eliding_model_wrapper(
    name: str,
    paths: set[str],
) -> set[str]:
    """Find groups after removing one nested ``*_model`` wrapper segment."""

    parts = name.split(".")
    matches = set()
    for index, part in enumerate(parts[1:-1], start=1):
        if part != "model" and not part.endswith("_model"):
            continue
        candidate = ".".join(parts[:index] + parts[index + 1 :])
        match = _longest_group_prefix(candidate, paths)
        if match is not None:
            matches.add(match)
    return matches


def map_checkpoint_names_to_groups(
    model: torch.nn.Module,
    names: Iterable[str],
    groups: list[WeightModuleGroup],
) -> dict[str, str | None]:
    """Map checkpoint tensors to the bounded runtime subtree that loads them."""

    paths = {group.path for group in groups}
    root_prefixes = {path.split(".", 1)[0] for path in paths if path}
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

        wrapper_matches = _groups_after_eliding_model_wrapper(mapped, paths)
        if len(wrapper_matches) > 1:
            raise ValueError(
                "ambiguous checkpoint group after model-wrapper normalization "
                f"for {name!r}: {wrapper_matches}"
            )
        if wrapper_matches:
            result[name] = next(iter(wrapper_matches))
            continue

        # Wrapper models may delegate unprefixed checkpoint names into one
        # named runtime child. Infer that prefix only when it is unambiguous.
        matches = {
            match
            for root in root_prefixes
            if (match := _longest_group_prefix(f"{root}.{mapped}", paths)) is not None
            and (match != root or len(root_prefixes) == 1)
        }
        if len(matches) > 1:
            raise ValueError(f"ambiguous checkpoint group for {name!r}: {matches}")
        result[name] = next(iter(matches), None)
    return result
