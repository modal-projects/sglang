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
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from sglang.srt.model_loader.utils import should_async_load
from sglang.srt.model_loader.weight_utils import default_weight_loader

logger = logging.getLogger(__name__)

_SOURCE_TAG = "_sglang_prepared_load_source"
_BATCH_NAMES = 256
_BATCH_BYTES = 256 << 20
_MAX_INFLIGHT_BATCHES = 32


@dataclass(frozen=True)
class PreparedLoadCall:
    """One direct call made by a model's ordinary checkpoint router."""

    parameter_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class PreparedLoadPlan:
    """A full-checkpoint dispatch plan recorded from an ordinary model load."""

    def __init__(self, fallback_patterns: Iterable[str] = ()) -> None:
        self.entries: dict[str, list[PreparedLoadCall]] = {}
        self.fallback: set[str] = set()
        self.fallback_patterns = tuple(fallback_patterns)
        self.recorded = False

    def _forced_fallback(self, name: str) -> bool:
        return any(pattern in name for pattern in self.fallback_patterns)

    def record(
        self,
        model: torch.nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> dict[str, int | float | str]:
        """Run the ordinary loader once while observing direct loader calls."""

        started = time.perf_counter()
        recorded: dict[str, list[PreparedLoadCall]] = {}
        seen: set[str] = set()
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
        for name in self.fallback:
            self.entries.pop(name, None)
        self.recorded = True
        stats: dict[str, int | float | str] = {
            "plan": "record",
            "entries": len(self.entries),
            "fallback": len(self.fallback),
            "seen": len(seen),
            "wall_s": round(time.perf_counter() - started, 6),
        }
        logger.info("[RL_PREPARED_LOAD_PLAN] %s", stats)
        return stats

    def replay(
        self,
        model: torch.nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        max_workers: int,
    ) -> dict[str, int | float | str]:
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
            for loaded_weight, calls in items:
                dispatch(loaded_weight, calls)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:

            def drain() -> None:
                for future in futures:
                    future.result()
                futures.clear()

            def flush() -> None:
                nonlocal batch_bytes
                if not batch:
                    return
                if len(futures) >= _MAX_INFLIGHT_BATCHES:
                    drain()
                futures.append(executor.submit(dispatch_batch, batch.copy()))
                batch.clear()
                batch_bytes = 0

            def fallback_weights() -> Iterable[tuple[str, torch.Tensor]]:
                nonlocal batch_bytes
                for name, tensor in weights:
                    calls = self.entries.get(name)
                    representable = calls is not None and all(
                        call.parameter_name in parameters for call in calls
                    )
                    if not representable:
                        if name in self.fallback:
                            counts["fallback"] += 1
                        else:
                            counts["unknown"] += 1
                        yield name, tensor
                        continue

                    counts["hit"] += 1
                    if should_async_load(tensor):
                        batch.append((tensor, calls))
                        batch_bytes += tensor.numel() * tensor.element_size()
                        if (
                            len(batch) >= _BATCH_NAMES
                            or batch_bytes >= _BATCH_BYTES
                        ):
                            flush()
                    else:
                        dispatch(tensor, calls)
                flush()
                # model.load_weights commonly executes a name-dependent tail
                # immediately after its input iterator is exhausted.
                drain()

            model.load_weights(fallback_weights())
            drain()

        stats: dict[str, int | float | str] = {
            "plan": "replay",
            "hits": counts["hit"],
            "fallback": counts["fallback"],
            "unknown": counts["unknown"],
            "workers": max_workers,
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
