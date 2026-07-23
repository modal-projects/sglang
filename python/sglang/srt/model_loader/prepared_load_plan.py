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
import threading
import time
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


class _DestinationWriteRecorder(TorchDispatchMode):
    """Observe contiguous ``copy_`` destinations inside one weight loader."""

    def __init__(
        self,
        parameter: torch.nn.Parameter,
        intervals: list[tuple[int, int]],
    ) -> None:
        super().__init__()
        self._storage_ptr = parameter.untyped_storage().data_ptr()
        self._storage_nbytes = parameter.untyped_storage().nbytes()
        self._intervals = intervals

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.copy_.default and args:
            destination = args[0]
            if (
                isinstance(destination, torch.Tensor)
                and destination.untyped_storage().data_ptr() == self._storage_ptr
                and destination.untyped_storage().nbytes() == self._storage_nbytes
                and destination.is_contiguous()
            ):
                begin = destination.storage_offset() * destination.element_size()
                end = begin + destination.numel() * destination.element_size()
                self._intervals.append((begin, end))
        return func(*args, **({} if kwargs is None else kwargs))


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
                        signature = ParameterStorageSignature(
                            shape=tuple(parameter_arg.shape),
                            stride=tuple(parameter_arg.stride()),
                            storage_offset=parameter_arg.storage_offset(),
                            storage_nbytes=parameter_arg.untyped_storage().nbytes(),
                        )
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
                            return original_loader(
                                parameter_arg,
                                loaded_weight,
                                *args,
                                **kwargs,
                            )
                        with _DestinationWriteRecorder(parameter_arg, intervals):
                            return original_loader(
                                parameter_arg,
                                loaded_weight,
                                *args,
                                **kwargs,
                            )

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
        self.recorded = True
        stats: dict[str, Any] = {
            "plan": "record",
            "entries": len(self.entries),
            "fallback": len(self.fallback),
            "seen": len(seen),
            "loader_kinds": self.loader_kinds,
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
            if signature is not None and signature == ParameterStorageSignature(
                shape=tuple(parameter.shape),
                stride=tuple(parameter.stride()),
                storage_offset=parameter.storage_offset(),
                storage_nbytes=parameter.untyped_storage().nbytes(),
            ):
                result.add(relative_name)
        return result

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
        batch: list[tuple[torch.Tensor, list[PreparedLoadCall]]] = []
        batch_bytes = 0
        source_next_s = 0.0
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
        worker_calls = 0
        worker_bytes = 0
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

        def dispatch_batch(
            items: list[tuple[torch.Tensor, list[PreparedLoadCall]]],
        ) -> None:
            nonlocal worker_wall_s, worker_cpu_s, worker_calls, worker_bytes
            batch_started = time.perf_counter()
            batch_cpu_started = time.thread_time()
            call_count = 0
            copied_bytes = 0
            for loaded_weight, load_calls in items:
                dispatch(loaded_weight, load_calls)
                call_count += len(load_calls)
                copied_bytes += loaded_weight.numel() * loaded_weight.element_size()
            with worker_lock:
                worker_wall_s += time.perf_counter() - batch_started
                worker_cpu_s += time.thread_time() - batch_cpu_started
                worker_calls += call_count
                worker_bytes += copied_bytes

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
                nonlocal source_tensors
                nonlocal synchronous_dispatch_s
                iterator = iter(weights)
                while True:
                    next_started = time.perf_counter()
                    try:
                        name, tensor = next(iterator)
                    except StopIteration:
                        source_next_s += time.perf_counter() - next_started
                        break
                    source_next_s += time.perf_counter() - next_started
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
                        batch.append((tensor, calls))
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
            "submitted_batches": submitted_batches,
            "max_inflight_batches": max_inflight_batches,
            "inflight_limit": inflight_limit,
            "drain_wait_s": round(drain_wait_s, 6),
            "synchronous_dispatch_s": round(synchronous_dispatch_s, 6),
            "worker_wall_s": round(worker_wall_s, 6),
            "worker_cpu_s": round(worker_cpu_s, 6),
            "worker_calls": worker_calls,
            "worker_bytes": worker_bytes,
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
