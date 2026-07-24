"""Incrementally advance a pinned runtime image from published XOR deltas.

During the initial, ordinary model load, :class:`RuntimeDeltaPlan` observes
only source-view to parameter-view ``copy_`` operations. It does not trace or
replay arbitrary computation. After quantization postprocessing, direct copies
whose destination layout is unchanged can be applied to the host runtime image
as XORs. Dense XOR and position/value sparse XOR share the same proven mapping;
the sparse form avoids materializing unchanged zero bytes. Model/quantization
hooks own derived or repacked destinations.

Every changed source tensor must be accounted for. Unsupported operations fail
preparation; they never preserve stale bytes or silently select a disk reload.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import threading
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
import zstandard
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.model_loader.weight_utils import default_weight_loader

logger = logging.getLogger(__name__)

_SOURCE_TAG = "_sglang_runtime_delta_source"


class RuntimeDeltaCoverageError(RuntimeError):
    """The recorded direct plan and explicit hooks do not cover a delta."""


@dataclass(frozen=True)
class SparseXorDelta:
    """Byte positions and XOR values for one logical checkpoint tensor."""

    encoded: bytes
    positions: torch.Tensor
    values: torch.Tensor
    logical_nbytes: int

    @property
    def encoded_nbytes(self) -> int:
        return len(self.encoded)

    def materialize(self) -> bytearray:
        dense = bytearray(self.logical_nbytes)
        target = torch.frombuffer(dense, dtype=torch.uint8)
        target[self.positions] = self.values
        return dense


RuntimeDelta = bytes | SparseXorDelta


@dataclass(frozen=True)
class TensorSignature:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    storage_nbytes: int
    dtype: torch.dtype


@dataclass(frozen=True)
class TensorViewSpec:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int


@dataclass(frozen=True)
class ParameterSignature:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    storage_nbytes: int
    dtype: torch.dtype


@dataclass(frozen=True)
class DirectCopy:
    parameter_name: str
    parameter_signature: ParameterSignature
    source_view: TensorViewSpec
    destination_view: TensorViewSpec


@dataclass(frozen=True)
class RuntimeCopy:
    image_storage_offset: int
    dtype: torch.dtype
    source_view: TensorViewSpec
    destination_view: TensorViewSpec


@dataclass(frozen=True)
class RuntimeTensorSpec:
    image_storage_offset: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int


def _view_numel(view: TensorViewSpec) -> int:
    result = 1
    for size in view.shape:
        result *= size
    return result


def _view_is_contiguous(view: TensorViewSpec) -> bool:
    expected_stride = 1
    for size, stride in zip(reversed(view.shape), reversed(view.stride)):
        if size > 1 and stride != expected_stride:
            return False
        expected_stride *= size
    return True


def _tensor_signature(tensor: torch.Tensor) -> TensorSignature:
    return TensorSignature(
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        storage_offset=tensor.storage_offset(),
        storage_nbytes=tensor.untyped_storage().nbytes(),
        dtype=tensor.dtype,
    )


def _parameter_signature(parameter: torch.nn.Parameter) -> ParameterSignature:
    return ParameterSignature(
        shape=tuple(parameter.shape),
        stride=tuple(parameter.stride()),
        storage_offset=parameter.storage_offset(),
        storage_nbytes=parameter.untyped_storage().nbytes(),
        dtype=parameter.dtype,
    )


def _iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


class _CopyRecorder(TorchDispatchMode):
    """Record direct copies while rejecting other mutations of source/dest."""

    def __init__(
        self,
        parameter: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.parameter_ptr = parameter.untyped_storage().data_ptr()
        self.parameter_nbytes = parameter.untyped_storage().nbytes()
        self.source_ptr = loaded_weight.untyped_storage().data_ptr()
        self.source_nbytes = loaded_weight.untyped_storage().nbytes()
        self.source_dtype = loaded_weight.dtype
        self.copies: list[tuple[TensorViewSpec, TensorViewSpec]] = []
        self.unsafe = False

    def _uses_storage(self, value, pointer: int) -> bool:
        return any(
            tensor.untyped_storage().data_ptr() == pointer
            for tensor in _iter_tensors(value)
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is torch.ops.aten.copy_.default and len(args) >= 2:
            destination, source = args[:2]
            if (
                isinstance(destination, torch.Tensor)
                and destination.untyped_storage().data_ptr() == self.parameter_ptr
            ):
                if (
                    destination.untyped_storage().nbytes() != self.parameter_nbytes
                    or not destination.is_contiguous()
                    or not isinstance(source, torch.Tensor)
                    or source.dtype != self.source_dtype
                    or source.untyped_storage().data_ptr() != self.source_ptr
                    or source.untyped_storage().nbytes() != self.source_nbytes
                ):
                    self.unsafe = True
                else:
                    self.copies.append(
                        (
                            TensorViewSpec(
                                shape=tuple(source.shape),
                                stride=tuple(source.stride()),
                                storage_offset=source.storage_offset(),
                            ),
                            TensorViewSpec(
                                shape=tuple(destination.shape),
                                stride=tuple(destination.stride()),
                                storage_offset=destination.storage_offset(),
                            ),
                        )
                    )
            if (
                isinstance(destination, torch.Tensor)
                and destination.untyped_storage().data_ptr() == self.source_ptr
            ):
                self.unsafe = True
        elif (
            func in (torch.ops.aten.fill_.Scalar, torch.ops.aten.zero_.default)
            and len(args) >= 1
            and isinstance(args[0], torch.Tensor)
            and args[0].untyped_storage().data_ptr() == self.parameter_ptr
            and not self._uses_storage((args[1:], kwargs), self.source_ptr)
            and (
                func is torch.ops.aten.zero_.default
                or (len(args) >= 2 and args[1] == 0)
            )
        ):
            # Vocabulary-parallel loaders zero their padding after copying the
            # source shard. Padding is invariant across versions, so it is not
            # part of the source-to-runtime delta mapping.
            pass
        elif func._schema.is_mutable:
            if self._uses_storage((args, kwargs), self.parameter_ptr) or self._uses_storage(
                (args, kwargs), self.source_ptr
            ):
                self.unsafe = True
        return func(*args, **kwargs)


class RuntimeDeltaPlan:
    """Proven initial-loader copies plus explicit derived-layout hooks."""

    def __init__(self) -> None:
        self.source_signatures: dict[str, TensorSignature] = {}
        self.direct_copies: dict[str, list[DirectCopy]] = {}
        self.unsafe_sources: set[str] = set()
        self.runtime_copies: dict[str, list[RuntimeCopy]] = {}
        self.resolved_copies: dict[str, list[RuntimeCopy]] = {}
        self.hook_sources: set[str] = set()
        self.quant_adapter_sources: dict[str, str] = {}
        self.dtype_conversion_sources: set[str] = set()
        self.unsupported_quant_sources: dict[str, str] = {}
        self.runtime_tensor_specs: dict[str, RuntimeTensorSpec] = {}
        self.finalized = False

    def record(
        self,
        model: torch.nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> dict[str, Any]:
        """Run the normal loader once and observe its direct storage copies."""

        started = time.perf_counter()
        seen: set[str] = set()
        copies: dict[str, list[DirectCopy]] = {}
        unsafe: set[str] = set()
        loader_kinds: Counter[str] = Counter()
        lock = threading.Lock()

        def tagged():
            for name, tensor in weights:
                seen.add(name)
                self.source_signatures[name] = _tensor_signature(tensor)
                try:
                    setattr(tensor, _SOURCE_TAG, name)
                except (AttributeError, RuntimeError):
                    unsafe.add(name)
                yield name, tensor

        wrapped: list[tuple[torch.nn.Parameter, str, Any]] = []

        def install_loader(parameter: torch.nn.Parameter, loader: Any) -> str:
            try:
                parameter.weight_loader = loader
                return "weight_loader"
            except AttributeError:
                parameter._weight_loader = loader
                return "_weight_loader"

        try:
            for parameter_name, parameter in model.named_parameters():
                original_loader = getattr(parameter, "weight_loader", None)
                created = original_loader is None
                if created:
                    original_loader = default_weight_loader

                def make_wrapper(
                    parameter_name: str = parameter_name,
                    original_loader: Any = original_loader,
                ):
                    loader_kind = (
                        f"{getattr(original_loader, '__module__', '')}."
                        f"{getattr(original_loader, '__qualname__', type(original_loader).__name__)}"
                    )

                    def recording_loader(
                        parameter_arg,
                        loaded_weight,
                        *args,
                        **kwargs,
                    ):
                        source_name = getattr(loaded_weight, _SOURCE_TAG, None)
                        if source_name is None:
                            return original_loader(
                                parameter_arg, loaded_weight, *args, **kwargs
                            )
                        signature = _parameter_signature(parameter_arg)
                        recorder = _CopyRecorder(parameter_arg, loaded_weight)
                        with recorder:
                            result = original_loader(
                                parameter_arg, loaded_weight, *args, **kwargs
                            )
                        with lock:
                            loader_kinds[loader_kind] += 1
                            if (
                                recorder.unsafe
                                or not recorder.copies
                                or parameter_arg.untyped_storage().data_ptr()
                                != recorder.parameter_ptr
                            ):
                                unsafe.add(source_name)
                            else:
                                copies.setdefault(source_name, []).extend(
                                    DirectCopy(
                                        parameter_name=parameter_name,
                                        parameter_signature=signature,
                                        source_view=source_view,
                                        destination_view=destination_view,
                                    )
                                    for source_view, destination_view in recorder.copies
                                )
                        return result

                    return recording_loader

                slot = install_loader(parameter, make_wrapper())
                wrapped.append(
                    (parameter, slot, None if created else original_loader)
                )

            model.load_weights(tagged())
        finally:
            for parameter, slot, original_loader in wrapped:
                if original_loader is None:
                    delattr(parameter, slot)
                else:
                    setattr(parameter, slot, original_loader)

        self.unsafe_sources = unsafe | (seen - copies.keys())
        self.direct_copies = {
            name: source_copies
            for name, source_copies in copies.items()
            if name not in self.unsafe_sources
        }
        stats = {
            "sources": len(seen),
            "direct_sources": len(self.direct_copies),
            "direct_operations": sum(map(len, self.direct_copies.values())),
            "unsafe_sources": len(self.unsafe_sources),
            "loader_kinds": dict(loader_kinds),
            "wall_s": round(time.perf_counter() - started, 6),
        }
        logger.info("[RL_RUNTIME_DELTA_RECORD] %s", stats)
        return stats

    def finalize(self, model: torch.nn.Module, segments) -> dict[str, Any]:
        """Resolve direct destinations against final postprocessed storages."""

        parameters = dict(model.named_parameters())
        segment_by_storage = {
            (
                segment.device_bytes.device.index,
                segment.device_bytes.untyped_storage().data_ptr(),
                segment.device_bytes.untyped_storage().nbytes(),
            ): segment
            for segment in segments
        }
        quant_adapter_parameters: dict[str, str] = {}
        unsupported_quant_parameters: dict[str, str] = {}
        for module_name, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is None:
                continue
            get_adapter_parameters = getattr(
                quant_method,
                "host_runtime_delta_parameter_names",
                None,
            )
            if get_adapter_parameters is None:
                if getattr(
                    quant_method,
                    "host_runtime_delta_direct_safe",
                    False,
                ):
                    continue
                prefix = f"{module_name}." if module_name else ""
                method_name = type(quant_method).__name__
                unsupported_quant_parameters.update(
                    {
                        f"{prefix}{name}": method_name
                        for name, _ in module.named_parameters(recurse=False)
                    }
                )
                continue
            prefix = f"{module_name}." if module_name else ""
            for local_name in get_adapter_parameters(module):
                full_name = f"{prefix}{local_name}"
                if full_name in quant_adapter_parameters:
                    raise RuntimeError(
                        f"duplicate host runtime quant adapter for {full_name!r}"
                    )
                quant_adapter_parameters[full_name] = module_name
        forced_patterns = tuple(
            getattr(model, "host_runtime_delta_fallback_patterns", ())
        )

        from sglang.srt.weight_sync.host_runtime import iter_model_tensors

        runtime_tensor_specs: dict[str, RuntimeTensorSpec] = {}
        for name, tensor in iter_model_tensors(model):
            if tensor.device.type != "cuda":
                continue
            storage = tensor.untyped_storage()
            segment = segment_by_storage.get(
                (tensor.device.index, storage.data_ptr(), storage.nbytes())
            )
            if segment is None:
                continue
            runtime_tensor_specs[name] = RuntimeTensorSpec(
                image_storage_offset=segment.image_offset,
                dtype=tensor.dtype,
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                storage_offset=tensor.storage_offset(),
            )

        runtime_copies: dict[str, list[RuntimeCopy]] = {}
        resolved_copies: dict[str, list[RuntimeCopy]] = {}
        hook_sources: set[str] = set(self.unsafe_sources)
        quant_adapter_sources: dict[str, str] = {}
        dtype_conversion_sources: set[str] = set()
        unsupported_quant_sources: dict[str, str] = {}
        reasons: Counter[str] = Counter()
        reason_examples: dict[str, list[dict[str, str]]] = {}

        def reject(source_name: str, reason: str, parameter_name: str = "") -> None:
            reasons[reason] += 1
            examples = reason_examples.setdefault(reason, [])
            if len(examples) < 12:
                examples.append(
                    {
                        "source": source_name,
                        "parameter": parameter_name,
                    }
                )

        for source_name in sorted(self.unsafe_sources):
            reject(source_name, "initial_loader_unsupported")
        for source_name, source_copies in self.direct_copies.items():
            resolved: list[RuntimeCopy] = []
            for copy in source_copies:
                parameter = parameters.get(copy.parameter_name)
                if parameter is None:
                    reject(source_name, "missing_parameter", copy.parameter_name)
                    break
                if _parameter_signature(parameter) != copy.parameter_signature:
                    reject(source_name, "changed_layout", copy.parameter_name)
                    break
                storage = parameter.untyped_storage()
                segment = segment_by_storage.get(
                    (parameter.device.index, storage.data_ptr(), storage.nbytes())
                )
                if segment is None:
                    reject(
                        source_name,
                        "storage_not_mirrored",
                        copy.parameter_name,
                    )
                    break
                resolved.append(
                    RuntimeCopy(
                        image_storage_offset=segment.image_offset,
                        dtype=parameter.dtype,
                        source_view=copy.source_view,
                        destination_view=copy.destination_view,
                    )
                )
            else:
                resolved_copies[source_name] = resolved
                if any(pattern in source_name for pattern in forced_patterns):
                    hook_sources.add(source_name)
                    reject(source_name, "model_fallback")
                    continue
                unsupported_methods = {
                    unsupported_quant_parameters[copy.parameter_name]
                    for copy in source_copies
                    if copy.parameter_name in unsupported_quant_parameters
                }
                if unsupported_methods:
                    hook_sources.add(source_name)
                    unsupported_quant_sources[source_name] = ",".join(
                        sorted(unsupported_methods)
                    )
                    reject(
                        source_name,
                        "unsupported_quant_method",
                        source_copies[0].parameter_name,
                    )
                    continue
                adapter_module = {
                    quant_adapter_parameters[copy.parameter_name]
                    for copy in source_copies
                    if copy.parameter_name in quant_adapter_parameters
                }
                if adapter_module:
                    if len(adapter_module) != 1:
                        raise RuntimeError(
                            "one checkpoint source maps to multiple host runtime "
                            f"quant adapters: {source_name!r} -> "
                            f"{sorted(adapter_module)}"
                        )
                    module_name = next(iter(adapter_module))
                    quant_adapter_sources[source_name] = module_name
                    hook_sources.add(source_name)
                    reject(
                        source_name,
                        "quant_adapter",
                        source_copies[0].parameter_name,
                    )
                    continue
                source_signature = self.source_signatures[source_name]
                if any(
                    copy.parameter_signature.dtype != source_signature.dtype
                    for copy in source_copies
                ):
                    hook_sources.add(source_name)
                    dtype_conversion_sources.add(source_name)
                    reject(
                        source_name,
                        "dtype_conversion",
                        source_copies[0].parameter_name,
                    )
                    continue
                runtime_copies[source_name] = resolved
                continue
            hook_sources.add(source_name)

        self.runtime_copies = runtime_copies
        self.resolved_copies = resolved_copies
        self.hook_sources = hook_sources
        self.quant_adapter_sources = quant_adapter_sources
        self.dtype_conversion_sources = dtype_conversion_sources
        self.unsupported_quant_sources = unsupported_quant_sources
        self.runtime_tensor_specs = runtime_tensor_specs
        self.finalized = True
        stats = {
            "direct_sources": len(runtime_copies),
            "direct_operations": sum(map(len, runtime_copies.values())),
            "hook_sources": len(hook_sources),
            "direct_source_bytes": sum(
                self.source_signatures[name].storage_nbytes for name in runtime_copies
            ),
            "hook_source_bytes": sum(
                self.source_signatures[name].storage_nbytes for name in hook_sources
            ),
            "reasons": dict(reasons),
            "reason_examples": reason_examples,
        }
        logger.info("[RL_RUNTIME_DELTA_FINALIZE] %s", stats)
        return stats

    def apply_versions(
        self,
        *,
        model: torch.nn.Module,
        host_image: torch.Tensor,
        source_dir: str,
        base_version: int,
        target_version: int,
    ) -> dict[str, Any]:
        if not self.finalized:
            raise RuntimeError("runtime delta plan was not finalized")
        if target_version <= base_version:
            raise ValueError(
                f"runtime delta target {target_version} must exceed base {base_version}"
            )

        started = time.perf_counter()
        stage_wall_s: Counter[str] = Counter()
        changed_sources = 0
        logical_bytes = 0
        max_raw_bytes = 0
        versions: list[tuple[int, str, str]] = []
        changed_names: set[str] = set()
        for version in range(base_version + 1, target_version + 1):
            version_dir = os.path.join(source_dir, f"weight_v{version:06d}")
            with open(
                os.path.join(version_dir, "model.safetensors.index.json")
            ) as file:
                index = json.load(file)
            metadata = index["metadata"]
            if int(metadata["base_version"]) != version - 1:
                raise RuntimeError(
                    f"out-of-order runtime delta v{version}: "
                    f"base={metadata['base_version']}"
                )
            encoding = metadata.get("delta_encoding")
            if encoding not in ("xor", "xor_sparse"):
                raise NotImplementedError(
                    "host runtime preparation currently requires dense or sparse "
                    f"XOR deltas; v{version} uses {encoding!r}"
                )
            changed_names.update(index.get("weight_map", {}))
            versions.append((version, version_dir, encoding))

        unknown = changed_names - self.source_signatures.keys()
        if unknown:
            raise RuntimeDeltaCoverageError(
                "runtime delta contains sources absent from initial load: "
                f"{sorted(unknown)[:20]}"
            )
        unsupported_quant = changed_names & self.unsupported_quant_sources.keys()
        if unsupported_quant:
            examples = [
                (name, self.unsupported_quant_sources[name])
                for name in sorted(unsupported_quant)[:20]
            ]
            raise RuntimeDeltaCoverageError(
                "runtime delta reaches quantization methods without an explicit "
                f"host adapter; examples={examples}"
            )
        needed_model_hooks = (
            changed_names
            - self.runtime_copies.keys()
            - self.quant_adapter_sources.keys()
            - self.dtype_conversion_sources
        )
        model_hook = getattr(model, "apply_host_runtime_delta", None)
        if needed_model_hooks and model_hook is None:
            raise RuntimeDeltaCoverageError(
                "runtime delta needs explicit model coverage "
                f"for {len(needed_model_hooks)} sources; examples="
                f"{sorted(needed_model_hooks)[:20]}"
            )
        validate_model_hook = getattr(
            model,
            "validate_host_runtime_delta_sources",
            None,
        )
        if needed_model_hooks and validate_model_hook is not None:
            validate_model_hook(needed_model_hooks)

        modules = dict(model.named_modules())
        quant_sources_by_module: dict[str, set[str]] = {}
        for name in changed_names & self.quant_adapter_sources.keys():
            quant_sources_by_module.setdefault(
                self.quant_adapter_sources[name],
                set(),
            ).add(name)
        for module_name, names in quant_sources_by_module.items():
            quant_method = getattr(modules[module_name], "quant_method", None)
            validate_adapter = getattr(
                quant_method,
                "validate_host_runtime_delta_sources",
                None,
            )
            if validate_adapter is not None:
                validate_adapter(
                    layer=modules[module_name],
                    module_name=module_name,
                    plan=self,
                    source_names=names,
                )

        decoded_bytes = 0
        sparse_sources = 0
        for _, version_dir, encoding in versions:
            stage_started = time.perf_counter()
            _, payloads = _read_delta_payloads(version_dir)
            stage_wall_s["read_compressed"] += time.perf_counter() - stage_started
            for name, compressed in payloads.items():
                signature = self.source_signatures.get(name)
                if signature is None:
                    raise RuntimeError(
                        f"runtime delta source was absent from initial load: {name!r}"
                    )
                stage_started = time.perf_counter()
                decompressor = zstandard.ZstdDecompressor()
                encoded = (
                    decompressor.decompress(
                        compressed,
                        max_output_size=signature.storage_nbytes,
                    )
                    if encoding == "xor"
                    else decompressor.decompress(compressed)
                )
                stage_wall_s["decompress"] += time.perf_counter() - stage_started
                if encoding == "xor":
                    delta: RuntimeDelta = encoded
                    if len(encoded) != signature.storage_nbytes:
                        raise RuntimeError(
                            f"runtime delta size mismatch for {name!r}: "
                            f"expected={signature.storage_nbytes} "
                            f"actual={len(encoded)}"
                        )
                else:
                    delta = _decode_sparse_xor(
                        encoded,
                        signature.storage_nbytes,
                    )
                    sparse_sources += 1
                changed_sources += 1
                logical_bytes += signature.storage_nbytes
                decoded_nbytes = (
                    len(delta)
                    if isinstance(delta, bytes)
                    else delta.encoded_nbytes
                )
                decoded_bytes += decoded_nbytes
                max_raw_bytes = max(max_raw_bytes, decoded_nbytes)
                stage_started = time.perf_counter()
                if name in self.runtime_copies:
                    self._apply_direct_xor(host_image, name, delta)
                    stage_wall_s["direct"] += (
                        time.perf_counter() - stage_started
                    )
                    continue

                context = RuntimeDeltaUpdateContext(
                    plan=self,
                    host_image=host_image,
                    source_deltas={name: delta},
                )
                module_name = self.quant_adapter_sources.get(name)
                if module_name is not None:
                    module = modules[module_name]
                    quant_method = getattr(module, "quant_method", None)
                    apply_adapter = getattr(
                        quant_method,
                        "apply_host_runtime_delta",
                        None,
                    )
                    if apply_adapter is None:
                        raise RuntimeDeltaCoverageError(
                            "quantization method declared host runtime sources but "
                            "does not implement apply_host_runtime_delta: "
                            f"module={module_name!r} "
                            f"method={type(quant_method).__name__}"
                        )
                    handled = set(
                        apply_adapter(
                            layer=module,
                            module_name=module_name,
                            context=context,
                            source_names={name},
                        )
                    )
                    if handled != {name}:
                        raise RuntimeError(
                            "runtime delta quant adapter coverage mismatch: "
                            f"source={name!r} handled={sorted(handled)}"
                        )
                    stage_wall_s["quant_adapter"] += (
                        time.perf_counter() - stage_started
                    )
                    continue

                if name in self.dtype_conversion_sources:
                    context.apply_dtype_conversion(name)
                    stage_wall_s["dtype_conversion"] += (
                        time.perf_counter() - stage_started
                    )
                    continue

                handled = set(
                    model_hook(
                        context=context,
                        source_names={name},
                    )
                )
                if handled != {name}:
                    raise RuntimeError(
                        "runtime delta model hook coverage mismatch: "
                        f"source={name!r} handled={sorted(handled)}"
                    )
                stage_wall_s["model_hook"] += (
                    time.perf_counter() - stage_started
                )

        finalize_started = time.perf_counter()
        empty_context = RuntimeDeltaUpdateContext(
            plan=self,
            host_image=host_image,
            source_deltas={},
        )
        for module_name, names in quant_sources_by_module.items():
            module = modules[module_name]
            quant_method = getattr(module, "quant_method", None)
            finalize_adapter = getattr(
                quant_method,
                "finalize_host_runtime_delta",
                None,
            )
            if finalize_adapter is not None:
                finalize_adapter(
                    layer=module,
                    module_name=module_name,
                    context=empty_context,
                    source_names=names,
                )
        finalize_model_hook = getattr(
            model,
            "finalize_host_runtime_delta",
            None,
        )
        if needed_model_hooks and finalize_model_hook is not None:
            finalize_model_hook(
                context=empty_context,
                source_names=needed_model_hooks,
            )
        stage_wall_s["finalize"] += time.perf_counter() - finalize_started

        accounted = (
            changed_names & self.runtime_copies.keys()
            | changed_names & self.quant_adapter_sources.keys()
            | changed_names & self.dtype_conversion_sources
            | needed_model_hooks
        )
        if accounted != changed_names:
            missing = changed_names - accounted
            extra = accounted - changed_names
            if missing or extra:
                raise RuntimeError(
                    "runtime delta hook coverage mismatch: "
                    f"missing={sorted(missing)[:20]} extra={sorted(extra)[:20]}"
                )

        return {
            "base_version": base_version,
            "target_version": target_version,
            "changed_sources": changed_sources,
            "logical_bytes": logical_bytes,
            "decoded_bytes": decoded_bytes,
            "sparse_sources": sparse_sources,
            "hook_sources": len(
                changed_names - self.runtime_copies.keys()
            ),
            "max_raw_bytes": max_raw_bytes,
            "stage_wall_s": {
                name: round(value, 6)
                for name, value in sorted(stage_wall_s.items())
            },
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def _apply_direct_xor(
        self,
        host_image: torch.Tensor,
        source_name: str,
        raw_delta: RuntimeDelta,
        *,
        copies: Optional[list[RuntimeCopy]] = None,
    ) -> None:
        selected_copies = (
            self.runtime_copies[source_name] if copies is None else copies
        )
        if isinstance(raw_delta, SparseXorDelta):
            mapped_copies = []
            for copy in selected_copies:
                mapped = self._sparse_copy_bytes(
                    source_name=source_name,
                    delta=raw_delta,
                    source_view=copy.source_view,
                    destination_view=copy.destination_view,
                    destination_dtype=copy.dtype,
                )
                if mapped is None:
                    self._apply_direct_xor(
                        host_image,
                        source_name,
                        raw_delta.materialize(),
                        copies=selected_copies,
                    )
                    return
                mapped_copies.append((copy, mapped))
            for copy, (positions, values) in mapped_copies:
                if positions.numel():
                    positions = positions + copy.image_storage_offset
                    host_image[positions] = host_image[positions].bitwise_xor(
                        values
                    )
            return

        signature = self.source_signatures[source_name]
        source_bytes = _buffer_as_tensor(raw_delta)
        source = torch.empty(0, dtype=signature.dtype).set_(
            source_bytes.untyped_storage(),
            signature.storage_offset,
            signature.shape,
            signature.stride,
        )
        for copy in selected_copies:
            element_size = torch.empty((), dtype=copy.dtype).element_size()
            destination_element_offset = (
                copy.image_storage_offset // element_size
                + copy.destination_view.storage_offset
            )
            destination = torch.empty(0, dtype=copy.dtype).set_(
                host_image.untyped_storage(),
                destination_element_offset,
                copy.destination_view.shape,
                copy.destination_view.stride,
            )
            source_view = source.as_strided(
                copy.source_view.shape,
                copy.source_view.stride,
                copy.source_view.storage_offset,
            )
            destination.view(torch.uint8).bitwise_xor_(
                source_view.contiguous().view(torch.uint8)
            )

    def _sparse_copy_bytes(
        self,
        *,
        source_name: str,
        delta: SparseXorDelta,
        source_view: TensorViewSpec,
        destination_view: TensorViewSpec,
        destination_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Map contiguous source-storage byte positions to destination storage."""

        signature = self.source_signatures[source_name]
        if (
            signature.dtype != destination_dtype
            or source_view.shape != destination_view.shape
            or not _view_is_contiguous(source_view)
            or not _view_is_contiguous(destination_view)
        ):
            return None
        element_size = torch.empty((), dtype=signature.dtype).element_size()
        view_nbytes = _view_numel(source_view) * element_size
        source_begin = source_view.storage_offset * element_size
        source_end = source_begin + view_nbytes
        mask = (delta.positions >= source_begin) & (
            delta.positions < source_end
        )
        relative = delta.positions[mask] - source_begin
        destination_begin = destination_view.storage_offset * element_size
        return destination_begin + relative, delta.values[mask]


class RuntimeDeltaUpdateContext:
    """Bounded access to one delta batch and the pinned final-layout image."""

    def __init__(
        self,
        *,
        plan: RuntimeDeltaPlan,
        host_image: torch.Tensor,
        source_deltas: dict[str, RuntimeDelta],
    ) -> None:
        self.plan = plan
        self.host_image = host_image
        self.source_deltas = source_deltas

    def source_tensor(self, source_name: str) -> torch.Tensor:
        signature = self.plan.source_signatures[source_name]
        delta = self.source_deltas[source_name]
        source_bytes = _buffer_as_tensor(
            delta.materialize()
            if isinstance(delta, SparseXorDelta)
            else delta
        )
        return torch.empty(0, dtype=signature.dtype).set_(
            source_bytes.untyped_storage(),
            signature.storage_offset,
            signature.shape,
            signature.stride,
        )

    def source_view(
        self,
        source_name: str,
        view: TensorViewSpec,
    ) -> torch.Tensor:
        return self.source_tensor(source_name).as_strided(
            view.shape,
            view.stride,
            view.storage_offset,
        )

    def host_tensor(self, tensor_name: str) -> torch.Tensor:
        try:
            spec = self.plan.runtime_tensor_specs[tensor_name]
        except KeyError as exc:
            raise RuntimeDeltaCoverageError(
                f"runtime tensor {tensor_name!r} is not mirrored in the host image"
            ) from exc
        element_size = torch.empty((), dtype=spec.dtype).element_size()
        return torch.empty(0, dtype=spec.dtype).set_(
            self.host_image.untyped_storage(),
            spec.image_storage_offset // element_size + spec.storage_offset,
            spec.shape,
            spec.stride,
        )

    def direct_copies(self, source_name: str) -> list[DirectCopy]:
        return self.plan.direct_copies[source_name]

    def sparse_delta(self, source_name: str) -> SparseXorDelta | None:
        delta = self.source_deltas[source_name]
        return delta if isinstance(delta, SparseXorDelta) else None

    def sparse_copy_bytes(
        self,
        source_name: str,
        copy: DirectCopy,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        delta = self.sparse_delta(source_name)
        if delta is None:
            return None
        return self.plan._sparse_copy_bytes(
            source_name=source_name,
            delta=delta,
            source_view=copy.source_view,
            destination_view=copy.destination_view,
            destination_dtype=copy.parameter_signature.dtype,
        )

    def xor_contiguous_target(
        self,
        source_name: str,
        target: torch.Tensor,
    ) -> None:
        """Apply an entire contiguous source XOR to an equal-shaped target."""

        signature = self.plan.source_signatures[source_name]
        delta = self.source_deltas[source_name]
        if (
            target.dtype != signature.dtype
            or tuple(target.shape) != signature.shape
            or not target.is_contiguous()
        ):
            raise RuntimeDeltaCoverageError(
                "contiguous runtime target does not match checkpoint source: "
                f"{source_name!r} source={signature.shape}/{signature.dtype} "
                f"target={tuple(target.shape)}/{target.dtype}"
            )
        target_bytes = target.view(torch.uint8).reshape(-1)
        if isinstance(delta, SparseXorDelta):
            element_size = target.element_size()
            source_begin = signature.storage_offset * element_size
            positions = delta.positions - source_begin
            if (
                positions.numel()
                and (
                    positions[0].item() < 0
                    or positions[-1].item() >= target_bytes.numel()
                )
            ):
                raise RuntimeDeltaCoverageError(
                    f"sparse XOR is outside source view for {source_name!r}"
                )
            target_bytes[positions] = target_bytes[positions].bitwise_xor(
                delta.values
            )
            return
        target_bytes.bitwise_xor_(
            self.source_tensor(source_name).contiguous().view(torch.uint8)
        )

    def xor_direct(self, source_name: str) -> None:
        try:
            copies = self.plan.resolved_copies[source_name]
        except KeyError as exc:
            raise RuntimeDeltaCoverageError(
                f"{source_name!r} has no safe direct runtime destination"
            ) from exc
        self.plan._apply_direct_xor(
            self.host_image,
            source_name,
            self.source_deltas[source_name],
            copies=copies,
        )

    def apply_dtype_conversion(self, source_name: str) -> None:
        """Apply a direct loader copy whose only transform is dtype conversion."""

        delta = self.sparse_delta(source_name)
        if delta is not None and self._apply_sparse_dtype_conversion(
            source_name,
            delta,
        ):
            return
        source = self.source_tensor(source_name)
        source_signature = self.plan.source_signatures[source_name]
        for direct, runtime in zip(
            self.plan.direct_copies[source_name],
            self.plan.resolved_copies[source_name],
        ):
            element_size = torch.empty((), dtype=runtime.dtype).element_size()
            destination = torch.empty(0, dtype=runtime.dtype).set_(
                self.host_image.untyped_storage(),
                runtime.image_storage_offset // element_size
                + runtime.destination_view.storage_offset,
                runtime.destination_view.shape,
                runtime.destination_view.stride,
            )
            source_delta = source.as_strided(
                direct.source_view.shape,
                direct.source_view.stride,
                direct.source_view.storage_offset,
            )
            if destination.shape != source_delta.shape:
                raise RuntimeDeltaCoverageError(
                    "dtype-converting runtime copy changed shape: "
                    f"{source_name!r} source={tuple(source_delta.shape)} "
                    f"destination={tuple(destination.shape)}"
                )
            reconstructed = destination.to(source_signature.dtype).contiguous()
            reconstructed.view(torch.uint8).bitwise_xor_(
                source_delta.contiguous().view(torch.uint8)
            )
            destination.copy_(reconstructed)

    def _apply_sparse_dtype_conversion(
        self,
        source_name: str,
        delta: SparseXorDelta,
    ) -> bool:
        source_signature = self.plan.source_signatures[source_name]
        source_element_size = torch.empty(
            (),
            dtype=source_signature.dtype,
        ).element_size()
        pending = []
        for direct, runtime in zip(
            self.plan.direct_copies[source_name],
            self.plan.resolved_copies[source_name],
        ):
            if (
                direct.source_view.shape != runtime.destination_view.shape
                or not _view_is_contiguous(direct.source_view)
                or not _view_is_contiguous(runtime.destination_view)
            ):
                return False
            source_begin = (
                direct.source_view.storage_offset * source_element_size
            )
            source_end = source_begin + (
                _view_numel(direct.source_view) * source_element_size
            )
            mask = (delta.positions >= source_begin) & (
                delta.positions < source_end
            )
            relative = delta.positions[mask] - source_begin
            pending.append((runtime, relative, delta.values[mask]))

        for runtime, relative, values in pending:
            destination_element_size = torch.empty(
                (),
                dtype=runtime.dtype,
            ).element_size()
            destination = torch.empty(0, dtype=runtime.dtype).set_(
                self.host_image.untyped_storage(),
                runtime.image_storage_offset // destination_element_size
                + runtime.destination_view.storage_offset,
                runtime.destination_view.shape,
                runtime.destination_view.stride,
            )
            flat_destination = destination.reshape(-1)
            source_elements = torch.div(
                relative,
                source_element_size,
                rounding_mode="floor",
            )
            lanes = relative.remainder(source_element_size)
            unique_elements, inverse = torch.unique(
                source_elements,
                sorted=True,
                return_inverse=True,
            )
            reconstructed = (
                flat_destination[unique_elements]
                .to(source_signature.dtype)
                .contiguous()
            )
            reconstructed_bytes = reconstructed.view(torch.uint8).reshape(
                -1,
                source_element_size,
            )
            reconstructed_bytes[inverse, lanes] = reconstructed_bytes[
                inverse,
                lanes,
            ].bitwise_xor(values)
            flat_destination[unique_elements] = reconstructed.to(runtime.dtype)
        return True


def _buffer_as_tensor(
    value,
    *,
    dtype: torch.dtype = torch.uint8,
    count: int = -1,
    offset: int = 0,
) -> torch.Tensor:
    """Create a zero-copy buffer view without PyTorch's read-only warning."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given buffer is not writable",
            category=UserWarning,
        )
        return torch.frombuffer(
            value,
            dtype=dtype,
            count=count,
            offset=offset,
        )


def _decode_sparse_xor(encoded: bytes, logical_nbytes: int) -> SparseXorDelta:
    if len(encoded) < 8:
        raise RuntimeError(
            f"sparse XOR payload is shorter than its count: {len(encoded)}B"
        )
    (count,) = struct.unpack_from("<Q", encoded)
    expected = 8 + count * 8 + count
    if len(encoded) != expected:
        raise RuntimeError(
            "sparse XOR payload size mismatch: "
            f"count={count} actual={len(encoded)} expected={expected}"
        )
    positions = _buffer_as_tensor(
        encoded,
        dtype=torch.int64,
        count=count,
        offset=8,
    )
    values = _buffer_as_tensor(
        encoded,
        count=count,
        offset=8 + count * 8,
    )
    if count:
        if positions[0].item() < 0 or positions[-1].item() >= logical_nbytes:
            raise RuntimeError(
                "sparse XOR position out of bounds: "
                f"first={positions[0].item()} last={positions[-1].item()} "
                f"logical_nbytes={logical_nbytes}"
            )
        if count > 1 and torch.any(positions[1:] <= positions[:-1]).item():
            raise RuntimeError(
                "sparse XOR positions must be strictly increasing"
            )
    return SparseXorDelta(
        encoded=encoded,
        positions=positions,
        values=values,
        logical_nbytes=logical_nbytes,
    )


def _safetensors_size(blob: bytes) -> Optional[int]:
    if len(blob) < 8:
        return None
    (header_len,) = struct.unpack("<Q", blob[:8])
    if len(blob) < 8 + header_len:
        return None
    try:
        header = json.loads(blob[8 : 8 + header_len])
    except ValueError:
        return None
    end = max(
        (
            info["data_offsets"][1]
            for name, info in header.items()
            if name != "__metadata__"
        ),
        default=0,
    )
    return 8 + header_len + end


def _read_delta_payloads(
    version_dir: str,
) -> tuple[dict[str, Any], dict[str, memoryview]]:
    with open(os.path.join(version_dir, "model.safetensors.index.json")) as file:
        index = json.load(file)
    expected_files = sorted(set(index.get("weight_map", {}).values()))
    payloads: dict[str, memoryview] = {}
    blobs: list[bytes] = []
    for filename in expected_files:
        path = os.path.join(version_dir, filename)
        with open(path, "rb") as file:
            blob = file.read()
        expected_size = _safetensors_size(blob)
        if expected_size is None or len(blob) != expected_size:
            raise FileNotFoundError(
                f"incomplete runtime delta blob {path}: "
                f"actual={len(blob)} expected={expected_size}"
            )
        blobs.append(blob)
        (header_len,) = struct.unpack("<Q", blob[:8])
        header = json.loads(blob[8 : 8 + header_len])
        data_start = 8 + header_len
        view = memoryview(blob)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            if name in payloads:
                raise RuntimeError(f"duplicate runtime delta tensor {name!r}")
            payloads[name] = view[data_start + begin : data_start + end]
    if set(payloads) != set(index.get("weight_map", {})):
        raise RuntimeError(
            "runtime delta index/payload mismatch: "
            f"index_only={sorted(set(index.get('weight_map', {})) - set(payloads))[:20]} "
            f"payload_only={sorted(set(payloads) - set(index.get('weight_map', {})))[:20]}"
        )
    # memoryviews retain their exporting bytes, so the local list only makes
    # that lifetime explicit while constructing the result.
    del blobs
    return index, payloads


def record_runtime_delta_plan(
    model: torch.nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> dict[str, Any]:
    plan = RuntimeDeltaPlan()
    model._runtime_delta_plan = plan
    return plan.record(model, weights)
