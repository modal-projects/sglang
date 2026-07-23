"""Record and replay checkpoint-to-parameter dispatch for dense preparation.

Prepared runtime reloads consume the complete checkpoint and produce a complete
rank-local runtime image.  Re-running a model's Python ``load_weights`` router
for every background preparation is unnecessary: for a fixed model instance,
checkpoint name, target parameter name, and loader arguments are invariant.

This module records those calls during the ordinary initial model load and can
later replay them against an isomorphic host proxy.  It is deliberately not a
partial-reload mechanism:

* every checkpoint tensor is consumed on every replay;
* a recorded tensor is always loaded, irrespective of whether its bytes changed;
* names whose behavior cannot be represented as direct parameter-loader calls
  are streamed through the model's ordinary ``load_weights`` implementation.

Models opt in because some loaders have name-dependent post-load tails.  Such
models declare fallback name patterns for the tensors that must continue through
their ordinary router.  An unsupported model remains correct and simply uses the
ordinary full loader.
"""

from __future__ import annotations

import concurrent.futures
import logging
import resource
import threading
import time
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.model_loader.utils import should_async_load
from sglang.srt.model_loader.weight_utils import default_weight_loader

logger = logging.getLogger(__name__)

_SOURCE_TAG = "_sglang_prepared_load_source"
_BATCH_NAMES = 256
_BATCH_BYTES = 256 << 20
_MIN_INFLIGHT_BATCHES = 32
_INFLIGHT_BATCHES_PER_WORKER = 2
_LOGICAL_STORAGE_RANGE_ATTR = "_sglang_prepared_logical_storage_range"


@dataclass(frozen=True)
class PreparedLoadCall:
    """One direct call made by a model's ordinary checkpoint router."""

    parameter_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class ParameterStorageSignature:
    """The restored storage layout whose complete writes were observed."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    storage_nbytes: int


@dataclass(frozen=True)
class TensorViewSpec:
    """One strided view into a source or destination tensor storage."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int


@dataclass(frozen=True)
class SourceTensorSignature:
    """Checkpoint tensor layout required by a compiled direct copy."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    storage_nbytes: int
    dtype: torch.dtype


@dataclass(frozen=True)
class PreparedDirectCopy:
    """A direct source-view to parameter-view copy observed at initial load."""

    parameter_name: str
    parameter_signature: ParameterStorageSignature
    source_signature: SourceTensorSignature
    source_view: TensorViewSpec
    destination_view: TensorViewSpec


def _logical_storage_range(tensor: torch.Tensor) -> tuple[int, int]:
    storage_nbytes = tensor.untyped_storage().nbytes()
    logical_range = getattr(tensor, _LOGICAL_STORAGE_RANGE_ATTR, None)
    if logical_range is None:
        return 0, storage_nbytes
    byte_offset, nbytes = logical_range
    if byte_offset < 0 or nbytes < 0 or byte_offset + nbytes > storage_nbytes:
        raise ValueError(
            "invalid prepared logical storage range: "
            f"offset={byte_offset} nbytes={nbytes} storage={storage_nbytes}"
        )
    return byte_offset, nbytes


def _parameter_storage_signature(
    parameter: torch.nn.Parameter,
) -> ParameterStorageSignature:
    logical_byte_offset, logical_nbytes = _logical_storage_range(parameter)
    tensor_byte_offset = parameter.storage_offset() * parameter.element_size()
    relative_byte_offset = tensor_byte_offset - logical_byte_offset
    if relative_byte_offset < 0 or relative_byte_offset % parameter.element_size():
        raise ValueError(
            "parameter view is not aligned within its logical storage: "
            f"tensor_offset={tensor_byte_offset} logical_offset={logical_byte_offset}"
        )
    return ParameterStorageSignature(
        shape=tuple(parameter.shape),
        stride=tuple(parameter.stride()),
        storage_offset=relative_byte_offset // parameter.element_size(),
        storage_nbytes=logical_nbytes,
    )


class _DestinationWriteRecorder(TorchDispatchMode):
    """Observe contiguous ``copy_`` destinations inside one weight loader."""

    def __init__(
        self,
        parameter: torch.nn.Parameter,
        intervals: list[tuple[int, int]],
        loaded_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self._storage_ptr = parameter.untyped_storage().data_ptr()
        self._storage_nbytes = parameter.untyped_storage().nbytes()
        self._logical_byte_offset, _ = _logical_storage_range(parameter)
        self._loaded_storage_ptr = loaded_weight.untyped_storage().data_ptr()
        self._loaded_storage_nbytes = loaded_weight.untyped_storage().nbytes()
        self._loaded_dtype = loaded_weight.dtype
        self._loaded_signature = SourceTensorSignature(
            shape=tuple(loaded_weight.shape),
            stride=tuple(loaded_weight.stride()),
            storage_offset=loaded_weight.storage_offset(),
            storage_nbytes=self._loaded_storage_nbytes,
            dtype=loaded_weight.dtype,
        )
        self._intervals = intervals
        self.direct_copies: list[
            tuple[SourceTensorSignature, TensorViewSpec, TensorViewSpec]
        ] = []
        self.has_unsupported_destination_write = False
        self.has_unsupported_source_mutation = False

    @staticmethod
    def _iter_tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from _DestinationWriteRecorder._iter_tensors(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from _DestinationWriteRecorder._iter_tensors(item)

    def _uses_destination_storage(self, value) -> bool:
        return any(
            tensor.untyped_storage().data_ptr() == self._storage_ptr
            for tensor in self._iter_tensors(value)
        )

    def _uses_loaded_storage(self, value) -> bool:
        return any(
            tensor.untyped_storage().data_ptr() == self._loaded_storage_ptr
            for tensor in self._iter_tensors(value)
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is torch.ops.aten.copy_.default and args:
            destination = args[0]
            if (
                isinstance(destination, torch.Tensor)
                and destination.untyped_storage().data_ptr()
                == self._loaded_storage_ptr
            ):
                self.has_unsupported_source_mutation = True
            if (
                isinstance(destination, torch.Tensor)
                and destination.untyped_storage().data_ptr() == self._storage_ptr
                and destination.untyped_storage().nbytes() == self._storage_nbytes
            ):
                if destination.is_contiguous():
                    begin = (
                        destination.storage_offset() * destination.element_size()
                        - self._logical_byte_offset
                    )
                    end = begin + destination.numel() * destination.element_size()
                    self._intervals.append((begin, end))
                source = args[1] if len(args) > 1 else None
                if (
                    destination.is_contiguous()
                    and isinstance(source, torch.Tensor)
                    and source.dtype == self._loaded_dtype
                    and source.untyped_storage().data_ptr()
                    == self._loaded_storage_ptr
                    and source.untyped_storage().nbytes()
                    == self._loaded_storage_nbytes
                ):
                    self.direct_copies.append(
                        (
                            self._loaded_signature,
                            TensorViewSpec(
                                shape=tuple(source.shape),
                                stride=tuple(source.stride()),
                                storage_offset=source.storage_offset(),
                            ),
                            TensorViewSpec(
                                shape=tuple(destination.shape),
                                stride=tuple(destination.stride()),
                                storage_offset=(
                                    destination.storage_offset()
                                    - (
                                        self._logical_byte_offset
                                        // destination.element_size()
                                    )
                                ),
                            ),
                        )
                    )
                else:
                    self.has_unsupported_destination_write = True
        elif func._schema.is_mutable:
            if self._uses_destination_storage((args, kwargs)):
                self.has_unsupported_destination_write = True
            if self._uses_loaded_storage((args, kwargs)):
                self.has_unsupported_source_mutation = True
        return func(*args, **kwargs)


def _covers_storage(intervals: Iterable[tuple[int, int]], nbytes: int) -> bool:
    """Return whether a set of byte intervals covers one entire storage."""

    cursor = 0
    for begin, end in sorted(intervals):
        if end <= cursor:
            continue
        if begin > cursor:
            return False
        cursor = end
        if cursor >= nbytes:
            return True
    return cursor >= nbytes


class PreparedLoadPlan:
    """A full-checkpoint dispatch plan recorded from an ordinary model load."""

    def __init__(self, fallback_patterns: Iterable[str] = ()) -> None:
        self.entries: dict[str, list[PreparedLoadCall]] = {}
        self.direct_copies: dict[str, list[PreparedDirectCopy]] = {}
        self.fallback: set[str] = set()
        self.fallback_patterns = tuple(fallback_patterns)
        self.loader_kinds: dict[str, int] = {}
        self.fully_overwritten_parameters: dict[
            str, ParameterStorageSignature
        ] = {}
        self.recorded = False

    def _forced_fallback(self, name: str) -> bool:
        return any(pattern in name for pattern in self.fallback_patterns)

    def record(
        self,
        model: torch.nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> dict[str, Any]:
        """Run the ordinary loader once while observing direct loader calls."""

        started = time.perf_counter()
        recorded: dict[str, list[PreparedLoadCall]] = {}
        recorded_direct_copies: dict[str, list[PreparedDirectCopy]] = {}
        uncompilable_sources: set[str] = set()
        seen: set[str] = set()
        loader_kinds: Counter[str] = Counter()
        write_intervals: dict[str, list[tuple[int, int]]] = {}
        parameter_signatures: dict[str, ParameterStorageSignature] = {}
        unsafe_parameters: set[str] = set()
        lock = threading.Lock()

        def tagged() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                seen.add(name)
                try:
                    setattr(tensor, _SOURCE_TAG, name)
                except (AttributeError, RuntimeError):
                    # Tensor wrappers that reject Python attributes remain on
                    # the ordinary full-loader path.
                    pass
                yield name, tensor

        wrapped: list[tuple[torch.nn.Parameter, str, Any]] = []

        def install_loader(parameter: torch.nn.Parameter, value: Any) -> str:
            try:
                parameter.weight_loader = value
                return "weight_loader"
            except AttributeError:
                # BasevLLMParameter exposes weight_loader through a read-only
                # property backed by _weight_loader.
                parameter._weight_loader = value
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
                    module_name = getattr(original_loader, "__module__", "")
                    qualname = getattr(
                        original_loader,
                        "__qualname__",
                        type(original_loader).__name__,
                    )
                    loader_kind = f"{module_name}.{qualname}"

                    def recording_loader(
                        parameter_arg,
                        loaded_weight,
                        *args,
                        **kwargs,
                    ):
                        source_name = getattr(loaded_weight, _SOURCE_TAG, None)
                        if source_name is not None:
                            call = PreparedLoadCall(
                                parameter_name=parameter_name,
                                args=args,
                                kwargs=kwargs.copy(),
                            )
                            with lock:
                                recorded.setdefault(source_name, []).append(call)
                                loader_kinds[loader_kind] += 1
                        signature = _parameter_storage_signature(parameter_arg)
                        with lock:
                            intervals = write_intervals.setdefault(
                                parameter_name, []
                            )
                            previous_signature = parameter_signatures.setdefault(
                                parameter_name, signature
                            )
                        if (
                            parameter_name in unsafe_parameters
                            or previous_signature != signature
                        ):
                            # A loader that rebinds or reshapes its destination
                            # cannot be proven safe for copy elision.
                            with lock:
                                unsafe_parameters.add(parameter_name)
                                parameter_signatures.pop(parameter_name, None)
                                intervals.clear()
                                if source_name is not None:
                                    uncompilable_sources.add(source_name)
                            return original_loader(
                                parameter_arg,
                                loaded_weight,
                                *args,
                                **kwargs,
                            )
                        recorder = _DestinationWriteRecorder(
                            parameter_arg,
                            intervals,
                            loaded_weight,
                        )
                        with recorder:
                            result = original_loader(
                                parameter_arg,
                                loaded_weight,
                                *args,
                                **kwargs,
                            )
                        if source_name is not None:
                            with lock:
                                if (
                                    recorder.has_unsupported_destination_write
                                    or recorder.has_unsupported_source_mutation
                                    or not recorder.direct_copies
                                ):
                                    uncompilable_sources.add(source_name)
                                else:
                                    recorded_direct_copies.setdefault(
                                        source_name, []
                                    ).extend(
                                        PreparedDirectCopy(
                                            parameter_name=parameter_name,
                                            parameter_signature=signature,
                                            source_signature=source_signature,
                                            source_view=source_view,
                                            destination_view=destination_view,
                                        )
                                        for (
                                            source_signature,
                                            source_view,
                                            destination_view,
                                        ) in recorder.direct_copies
                                    )
                        return result

                    return recording_loader

                slot = install_loader(parameter, make_wrapper())
                wrapped.append(
                    (
                        parameter,
                        slot,
                        None if created else original_loader,
                    )
                )

            model.load_weights(tagged())
        finally:
            for parameter, slot, original_loader in wrapped:
                if original_loader is None:
                    delattr(parameter, slot)
                else:
                    setattr(parameter, slot, original_loader)

        self.entries = recorded
        self.fallback = {
            name
            for name in seen
            if name not in recorded or self._forced_fallback(name)
        }
        self.loader_kinds = dict(loader_kinds)
        self.fully_overwritten_parameters = {
            name: signature
            for name, signature in parameter_signatures.items()
            if name not in unsafe_parameters
            if _covers_storage(
                write_intervals.get(name, ()),
                signature.storage_nbytes,
            )
        }
        for name in self.fallback:
            self.entries.pop(name, None)
        self.direct_copies = {
            name: copies
            for name, copies in recorded_direct_copies.items()
            if name in self.entries
            if name not in uncompilable_sources
            if all(
                copy.parameter_name not in unsafe_parameters for copy in copies
            )
        }
        self.recorded = True
        stats: dict[str, Any] = {
            "plan": "record",
            "entries": len(self.entries),
            "fallback": len(self.fallback),
            "seen": len(seen),
            "loader_kinds": self.loader_kinds,
            "direct_copy_entries": len(self.direct_copies),
            "direct_copy_operations": sum(
                len(copies) for copies in self.direct_copies.values()
            ),
            "fully_overwritten_parameters": len(
                self.fully_overwritten_parameters
            ),
            "wall_s": round(time.perf_counter() - started, 6),
        }
        logger.info("[RL_PREPARED_LOAD_PLAN] %s", stats)
        return stats

    def fully_overwritten_parameter_names(
        self,
        module: torch.nn.Module,
        *,
        parameter_prefix: str = "",
    ) -> set[str]:
        """Return restored Parameters proven fully determined by checkpoint load.

        The caller still groups aliases by storage and requires every Parameter
        view backed by a storage to appear in this result before eliding its
        initial copy. Unrecognized, partial, and reshaped destinations stay on
        the conservative initialization path.
        """

        parameters = dict(module.named_parameters(remove_duplicate=False))
        result: set[str] = set()
        for relative_name, parameter in parameters.items():
            full_name = (
                f"{parameter_prefix}.{relative_name}"
                if parameter_prefix
                else relative_name
            )
            signature = self.fully_overwritten_parameters.get(full_name)
            if signature is not None and signature == _parameter_storage_signature(
                parameter
            ):
                result.add(relative_name)
        return result

    @staticmethod
    def _parameter_signature(
        parameter: torch.nn.Parameter,
    ) -> ParameterStorageSignature:
        return _parameter_storage_signature(parameter)

    @staticmethod
    def _source_signature(tensor: torch.Tensor) -> SourceTensorSignature:
        return SourceTensorSignature(
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            storage_offset=tensor.storage_offset(),
            storage_nbytes=tensor.untyped_storage().nbytes(),
            dtype=tensor.dtype,
        )

    def _resolve_direct_copies(
        self,
        parameters: dict[str, torch.nn.Parameter],
        loaded_weight: torch.Tensor,
        copies: list[PreparedDirectCopy],
    ) -> list[tuple[torch.Tensor, torch.Tensor, int, int, int]] | None:
        """Resolve validated copy views against one isomorphic host proxy."""

        resolved: list[tuple[torch.Tensor, torch.Tensor, int, int, int]] = []
        actual_source_signature = self._source_signature(loaded_weight)
        for copy in copies:
            if actual_source_signature != copy.source_signature:
                return None
            parameter = parameters.get(copy.parameter_name)
            if (
                parameter is None
                or self._parameter_signature(parameter) != copy.parameter_signature
            ):
                return None
            try:
                logical_byte_offset, _ = _logical_storage_range(parameter)
                if logical_byte_offset % parameter.element_size():
                    return None
                destination_storage_offset = (
                    logical_byte_offset // parameter.element_size()
                    + copy.destination_view.storage_offset
                )
                source = (
                    loaded_weight
                    if (
                        copy.source_view.shape == tuple(loaded_weight.shape)
                        and copy.source_view.stride == tuple(loaded_weight.stride())
                        and copy.source_view.storage_offset
                        == loaded_weight.storage_offset()
                    )
                    else loaded_weight.as_strided(
                        copy.source_view.shape,
                        copy.source_view.stride,
                        copy.source_view.storage_offset,
                    )
                )
                destination = (
                    parameter
                    if (
                        copy.destination_view.shape == tuple(parameter.shape)
                        and copy.destination_view.stride == tuple(parameter.stride())
                        and destination_storage_offset
                        == parameter.storage_offset()
                    )
                    else parameter.as_strided(
                        copy.destination_view.shape,
                        copy.destination_view.stride,
                        destination_storage_offset,
                    )
                )
            except RuntimeError:
                return None
            if (
                source.dtype != loaded_weight.dtype
                or destination.dtype != parameter.dtype
                or source.shape != destination.shape
                or not destination.is_contiguous()
            ):
                return None
            begin = destination.storage_offset() * destination.element_size()
            end = begin + destination.numel() * destination.element_size()
            resolved.append(
                (
                    destination,
                    source,
                    destination.untyped_storage().data_ptr(),
                    begin,
                    end,
                )
            )
        return resolved

    def replay(
        self,
        model: torch.nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        max_workers: int,
    ) -> dict[str, Any]:
        """Consume the full checkpoint and replay safe direct dispatch calls.

        Direct batches may execute in parallel, but the fallback iterator does
        not report exhaustion until every direct batch has completed.  This
        preserves the ordinary loader's guarantee that name-gated post-load
        work runs only after all direct parameter writes are visible.
        """

        if not self.recorded:
            raise RuntimeError("prepared load plan has not been recorded")
        if max_workers <= 0:
            raise ValueError("prepared load plan needs at least one worker")

        started = time.perf_counter()
        parameters = dict(model.named_parameters())
        counts = {"hit": 0, "fallback": 0, "unknown": 0}
        futures: list[concurrent.futures.Future[None]] = []
        batch: list[tuple[str, torch.Tensor, list[PreparedLoadCall]]] = []
        batch_bytes = 0
        source_next_s = 0.0
        source_next_cpu_s = 0.0
        source_next_minor_faults = 0
        source_next_major_faults = 0
        source_tensors = 0
        source_bytes = 0
        direct_source_bytes = 0
        fallback_source_bytes = 0
        submitted_batches = 0
        max_inflight_batches = 0
        drain_wait_s = 0.0
        synchronous_dispatch_s = 0.0
        worker_wall_s = 0.0
        worker_cpu_s = 0.0
        worker_minor_faults = 0
        worker_major_faults = 0
        worker_calls = 0
        worker_bytes = 0
        worker_direct_entries = 0
        worker_direct_operations = 0
        worker_foreach_calls = 0
        worker_foreach_fallback_calls = 0
        worker_lock = threading.Lock()
        inflight_limit = max(
            _MIN_INFLIGHT_BATCHES,
            max_workers * _INFLIGHT_BATCHES_PER_WORKER,
        )

        def dispatch(
            loaded_weight: torch.Tensor,
            calls: list[PreparedLoadCall],
        ) -> None:
            for call in calls:
                parameter = parameters[call.parameter_name]
                loader = (
                    getattr(parameter, "weight_loader", None)
                    or default_weight_loader
                )
                loader(
                    parameter,
                    loaded_weight,
                    *call.args,
                    **call.kwargs,
                )

        @torch.inference_mode()
        def dispatch_batch(
            items: list[tuple[str, torch.Tensor, list[PreparedLoadCall]]],
        ) -> None:
            nonlocal worker_wall_s
            nonlocal worker_cpu_s
            nonlocal worker_minor_faults
            nonlocal worker_major_faults
            nonlocal worker_calls
            nonlocal worker_bytes
            nonlocal worker_direct_entries
            nonlocal worker_direct_operations
            nonlocal worker_foreach_calls
            nonlocal worker_foreach_fallback_calls
            batch_started = time.perf_counter()
            batch_cpu_started = time.thread_time()
            batch_usage_started = resource.getrusage(resource.RUSAGE_THREAD)
            call_count = 0
            copied_bytes = 0
            direct_entries = 0
            direct_operations = 0
            foreach_calls = 0
            foreach_fallback_calls = 0
            destinations: list[torch.Tensor] = []
            sources: list[torch.Tensor] = []
            destination_ranges: dict[int, list[tuple[int, int]]] = {}

            def add_nonoverlapping_range(
                ranges: list[tuple[int, int]],
                begin: int,
                end: int,
            ) -> bool:
                """Insert one sorted interval, or report an existing overlap."""

                index = bisect_left(ranges, (begin, end))
                if index > 0 and ranges[index - 1][1] > begin:
                    return False
                if index < len(ranges) and ranges[index][0] < end:
                    return False
                ranges.insert(index, (begin, end))
                return True

            def flush_direct() -> None:
                nonlocal foreach_calls, foreach_fallback_calls
                if not destinations:
                    return
                foreach_calls += 1
                try:
                    torch._foreach_copy_(destinations, sources)
                except RuntimeError:
                    # Repeating a possibly completed copy is idempotent. Keep a
                    # conservative per-view fallback for dtype/backend variants
                    # unsupported by foreach.
                    foreach_fallback_calls += 1
                    for destination, source in zip(destinations, sources):
                        destination.copy_(source)
                destinations.clear()
                sources.clear()
                destination_ranges.clear()

            for name, loaded_weight, load_calls in items:
                call_count += len(load_calls)
                copied_bytes += loaded_weight.numel() * loaded_weight.element_size()
                direct = self.direct_copies.get(name)
                resolved = (
                    self._resolve_direct_copies(
                        parameters,
                        loaded_weight,
                        direct,
                    )
                    if direct is not None
                    else None
                )
                if resolved is None:
                    flush_direct()
                    dispatch(loaded_weight, load_calls)
                    continue

                direct_entries += 1
                direct_operations += len(resolved)
                for destination, source, storage_ptr, begin, end in resolved:
                    ranges = destination_ranges.setdefault(storage_ptr, [])
                    if not add_nonoverlapping_range(ranges, begin, end):
                        flush_direct()
                        ranges = destination_ranges.setdefault(storage_ptr, [])
                        if not add_nonoverlapping_range(ranges, begin, end):
                            raise AssertionError(
                                "empty direct-copy batch contains an overlap"
                            )
                    destinations.append(destination)
                    sources.append(source)
            flush_direct()
            batch_usage_done = resource.getrusage(resource.RUSAGE_THREAD)
            with worker_lock:
                worker_wall_s += time.perf_counter() - batch_started
                worker_cpu_s += time.thread_time() - batch_cpu_started
                worker_minor_faults += (
                    batch_usage_done.ru_minflt - batch_usage_started.ru_minflt
                )
                worker_major_faults += (
                    batch_usage_done.ru_majflt - batch_usage_started.ru_majflt
                )
                worker_calls += call_count
                worker_bytes += copied_bytes
                worker_direct_entries += direct_entries
                worker_direct_operations += direct_operations
                worker_foreach_calls += foreach_calls
                worker_foreach_fallback_calls += foreach_fallback_calls

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:

            def drain() -> None:
                nonlocal drain_wait_s
                wait_started = time.perf_counter()
                for future in futures:
                    future.result()
                drain_wait_s += time.perf_counter() - wait_started
                futures.clear()

            def flush() -> None:
                nonlocal batch_bytes, submitted_batches, max_inflight_batches
                if not batch:
                    return
                if len(futures) >= inflight_limit:
                    drain()
                futures.append(executor.submit(dispatch_batch, batch.copy()))
                submitted_batches += 1
                max_inflight_batches = max(max_inflight_batches, len(futures))
                batch.clear()
                batch_bytes = 0

            def fallback_weights() -> Iterable[tuple[str, torch.Tensor]]:
                nonlocal batch_bytes
                nonlocal direct_source_bytes
                nonlocal fallback_source_bytes
                nonlocal source_bytes
                nonlocal source_next_s
                nonlocal source_next_cpu_s
                nonlocal source_next_minor_faults
                nonlocal source_next_major_faults
                nonlocal source_tensors
                nonlocal synchronous_dispatch_s
                iterator = iter(weights)
                while True:
                    next_started = time.perf_counter()
                    next_cpu_started = time.thread_time()
                    next_usage_started = resource.getrusage(resource.RUSAGE_THREAD)
                    try:
                        item = next(iterator)
                    except StopIteration:
                        item = None
                    next_usage_done = resource.getrusage(resource.RUSAGE_THREAD)
                    source_next_s += time.perf_counter() - next_started
                    source_next_cpu_s += time.thread_time() - next_cpu_started
                    source_next_minor_faults += (
                        next_usage_done.ru_minflt - next_usage_started.ru_minflt
                    )
                    source_next_major_faults += (
                        next_usage_done.ru_majflt - next_usage_started.ru_majflt
                    )
                    if item is None:
                        break
                    name, tensor = item
                    tensor_bytes = tensor.numel() * tensor.element_size()
                    source_tensors += 1
                    source_bytes += tensor_bytes
                    calls = self.entries.get(name)
                    representable = calls is not None and all(
                        call.parameter_name in parameters for call in calls
                    )
                    if not representable:
                        if name in self.fallback:
                            counts["fallback"] += 1
                        else:
                            counts["unknown"] += 1
                        fallback_source_bytes += tensor_bytes
                        yield name, tensor
                        continue

                    counts["hit"] += 1
                    direct_source_bytes += tensor_bytes
                    if should_async_load(tensor):
                        batch.append((name, tensor, calls))
                        batch_bytes += tensor_bytes
                        if (
                            len(batch) >= _BATCH_NAMES
                            or batch_bytes >= _BATCH_BYTES
                        ):
                            flush()
                    else:
                        dispatch_started = time.perf_counter()
                        dispatch(tensor, calls)
                        synchronous_dispatch_s += (
                            time.perf_counter() - dispatch_started
                        )
                flush()
                # model.load_weights commonly executes a name-dependent tail
                # immediately after its input iterator is exhausted.
                drain()

            model_load_started = time.perf_counter()
            model.load_weights(fallback_weights())
            model_load_s = time.perf_counter() - model_load_started
            drain()

        stats: dict[str, Any] = {
            "plan": "replay",
            "hits": counts["hit"],
            "fallback": counts["fallback"],
            "unknown": counts["unknown"],
            "workers": max_workers,
            "source_tensors": source_tensors,
            "source_bytes": source_bytes,
            "direct_source_bytes": direct_source_bytes,
            "fallback_source_bytes": fallback_source_bytes,
            "source_next_s": round(source_next_s, 6),
            "source_next_cpu_s": round(source_next_cpu_s, 6),
            "source_next_minor_faults": source_next_minor_faults,
            "source_next_major_faults": source_next_major_faults,
            "submitted_batches": submitted_batches,
            "max_inflight_batches": max_inflight_batches,
            "inflight_limit": inflight_limit,
            "drain_wait_s": round(drain_wait_s, 6),
            "synchronous_dispatch_s": round(synchronous_dispatch_s, 6),
            "worker_wall_s": round(worker_wall_s, 6),
            "worker_cpu_s": round(worker_cpu_s, 6),
            "worker_minor_faults": worker_minor_faults,
            "worker_major_faults": worker_major_faults,
            "worker_calls": worker_calls,
            "worker_bytes": worker_bytes,
            "worker_direct_entries": worker_direct_entries,
            "worker_direct_operations": worker_direct_operations,
            "worker_foreach_calls": worker_foreach_calls,
            "worker_foreach_fallback_calls": worker_foreach_fallback_calls,
            "model_load_s": round(model_load_s, 6),
            "wall_s": round(time.perf_counter() - started, 6),
        }
        logger.info("[RL_PREPARED_LOAD_PLAN] %s", stats)
        return stats


def get_or_create_prepared_load_plan(
    model: torch.nn.Module,
) -> PreparedLoadPlan | None:
    """Return a model's opt-in dense preparation plan."""

    if not getattr(model, "supports_prepared_load_plan", False):
        return None
    plan = getattr(model, "_prepared_load_plan", None)
    if plan is None:
        plan = PreparedLoadPlan(
            getattr(model, "prepared_load_plan_fallback_patterns", ())
        )
        model._prepared_load_plan = plan
    return plan
