"""Record/replay load plans for repeated in-place weight reloads.

A weight reload (update_weights_from_disk) executes the same dispatch every
time: the mapping *checkpoint tensor name -> (target param, weight_loader,
shard/expert args)* is a pure function of the model architecture and the
checkpoint layout, both fixed for the life of a server. Yet each model's
``load_weights`` re-derives it per tensor per reload through substring scans
over stacked/expert mapping lists and per-model rewrite logic — measured at
81-99% of reload wall time on large MoE checkpoints.

``LoadPlan`` removes that recurring cost:

- The FIRST reload runs the model's own ``load_weights`` unchanged, with every
  parameter's ``weight_loader`` transparently wrapped so the effects of the
  dispatch are recorded: source checkpoint name -> [(param fqn, extra loader
  args)]. Recording is keyed by a tag planted on each yielded tensor, so a
  loader call whose tensor was derived in flight (fused, re-quantized, ...)
  simply has no tag and is never recorded — such names permanently fall back.
- Subsequent reloads REPLAY: O(1) plan lookup per tensor, direct weight_loader
  invocation (params re-resolved by fqn, robust to postprocess rebinds), with
  host-tensor loads dispatched to a thread pool so H2D copies overlap for every
  architecture. Names without a recorded plan — derived inputs, names the model
  skipped, and anything matching the model's declared
  ``load_plan_fallback_patterns`` — are streamed through the model's own
  ``load_weights`` unchanged, which also preserves per-model post-load tails
  (e.g. the DeepSeek MLA ``post_load_weights(weight_names=...)`` re-derivation,
  which is why those models declare ``("kv_b_proj",)`` as a fallback pattern).

Models opt in by setting ``supports_load_plan_replay = True``; the feature is
additionally gated behind ``SGLANG_ENABLE_RELOAD_LOAD_PLAN=1``.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from typing import Any, Dict, Iterable, List, Set, Tuple

import torch

from sglang.srt.model_loader.utils import should_async_load

logger = logging.getLogger(__name__)

_SOURCE_TAG = "_load_plan_source_name"
# Soft bound on outstanding async loader calls, so replay cannot materialize
# unboundedly many CPU tensors ahead of the H2D copies that release them.
_MAX_INFLIGHT = 4096


class LoadPlan:
    """One model instance's recorded reload dispatch (see module docstring)."""

    def __init__(self, fallback_patterns: Iterable[str] = ()) -> None:
        # source checkpoint name -> [(param fqn, loader args, loader kwargs)]
        self.entries: Dict[str, List[Tuple[str, tuple, dict]]] = {}
        # Names that must always go through the model's own load_weights.
        self.fallback: Set[str] = set()
        self.fallback_patterns = tuple(fallback_patterns)
        self.recorded = False

    def _forced_fallback(self, name: str) -> bool:
        return any(pattern in name for pattern in self.fallback_patterns)

    def record(
        self, model: torch.nn.Module, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Run the model's own load_weights (a fully normal load) while
        recording the dispatch it performs."""
        start = time.perf_counter()
        recorded: Dict[str, List[Tuple[str, tuple, dict]]] = {}
        seen: Set[str] = set()
        lock = threading.Lock()

        def tagged() -> Iterable[Tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                seen.add(name)
                try:
                    setattr(tensor, _SOURCE_TAG, name)
                except AttributeError:
                    pass  # untaggable tensor: its loads stay unrecorded -> fallback
                yield name, tensor

        # weight_loader is a plain attribute on vanilla Parameters but a
        # read-only property over `_weight_loader` on BasevLLMParameter
        # subclasses — wrap whichever slot actually holds the callable.
        wrapped: List[Tuple[torch.nn.Parameter, str, Any]] = []

        def swap_loader(param: torch.nn.Parameter, value: Any) -> str:
            try:
                param.weight_loader = value
                return "weight_loader"
            except AttributeError:
                param._weight_loader = value
                return "_weight_loader"

        try:
            for fqn, param in model.named_parameters():
                loader = getattr(param, "weight_loader", None)
                if loader is None:
                    continue  # default-loaded params fall back to load_weights

                def make_wrapper(fqn: str = fqn, loader: Any = loader):
                    def recording_loader(param_arg, tensor, *args, **kwargs):
                        source = getattr(tensor, _SOURCE_TAG, None)
                        if source is not None:
                            with lock:
                                recorded.setdefault(source, []).append((fqn, args, kwargs))
                        return loader(param_arg, tensor, *args, **kwargs)

                    return recording_loader

                slot = swap_loader(param, make_wrapper())
                wrapped.append((param, slot, loader))

            model.load_weights(tagged())
        finally:
            for param, slot, loader in wrapped:
                setattr(param, slot, loader)

        self.entries = recorded
        # Seen-but-unrecorded names have effects the boundary cannot represent
        # (derived tensors, buffered fusions) or none at all (skipped names);
        # both are only correct through the model's own loader.
        self.fallback = {name for name in seen if name not in recorded}
        self.fallback.update(name for name in recorded if self._forced_fallback(name))
        for name in self.fallback:
            self.entries.pop(name, None)
        self.recorded = True
        stats = {
            "plan": "record",
            "plan_entries": len(self.entries),
            "plan_fallback": len(self.fallback),
            "plan_record_s": round(time.perf_counter() - start, 2),
        }
        logger.info(
            f"[load plan] recorded {stats['plan_entries']} entries "
            f"({stats['plan_fallback']} fallback names) in {stats['plan_record_s']}s"
        )
        return stats

    def replay(
        self,
        model: torch.nn.Module,
        weights: Iterable[Tuple[str, torch.Tensor]],
        max_workers: int = 8,
    ) -> Dict[str, Any]:
        """Dispatch recorded names directly (host tensors via a thread pool);
        stream everything else through the model's own load_weights, which
        also runs any per-model post-load tail on exactly those names."""
        start = time.perf_counter()
        params_dict = dict(model.named_parameters())
        counts = {"hit": 0, "fallback": 0, "unknown": 0}
        futures: List[concurrent.futures.Future] = []

        def dispatch(fqn: str, tensor: torch.Tensor, args: tuple, kwargs: dict) -> None:
            param = params_dict[fqn]
            param.weight_loader(param, tensor, *args, **kwargs)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

            def filtered() -> Iterable[Tuple[str, torch.Tensor]]:
                for name, tensor in weights:
                    calls = self.entries.get(name)
                    if calls is None:
                        counts["fallback" if name in self.fallback else "unknown"] += 1
                        yield name, tensor
                        continue
                    counts["hit"] += 1
                    if should_async_load(tensor):
                        if len(futures) >= _MAX_INFLIGHT:
                            for future in futures:
                                future.result()
                            futures.clear()
                        for fqn, args, kwargs in calls:
                            futures.append(executor.submit(dispatch, fqn, tensor, args, kwargs))
                    else:
                        for fqn, args, kwargs in calls:
                            dispatch(fqn, tensor, args, kwargs)

            model.load_weights(filtered())
            for future in futures:
                future.result()

        stats = {
            "plan": "replay",
            "plan_hits": counts["hit"],
            "plan_fallback": counts["fallback"],
            "plan_unknown": counts["unknown"],
            "plan_replay_s": round(time.perf_counter() - start, 2),
        }
        logger.info(
            f"[load plan] replayed {stats['plan_hits']} names "
            f"({stats['plan_fallback']} fallback, {stats['plan_unknown']} unknown) "
            f"in {stats['plan_replay_s']}s"
        )
        return stats


def get_or_create_plan(model: torch.nn.Module) -> "LoadPlan | None":
    """The model instance's reload plan, if it opted in (else None)."""
    if not getattr(model, "supports_load_plan_replay", False):
        return None
    plan = getattr(model, "_reload_load_plan", None)
    if plan is None:
        plan = LoadPlan(getattr(model, "load_plan_fallback_patterns", ()))
        model._reload_load_plan = plan
    return plan
