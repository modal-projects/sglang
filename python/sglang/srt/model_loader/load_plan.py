"""Record/replay load plans for repeated in-place weight reloads.

``update_weights_from_disk`` executes the same dispatch every time: the
mapping *checkpoint tensor name -> (target param, weight_loader, shard/expert
args)* is a pure function of the model architecture and the checkpoint
layout, both fixed for the life of a server. Yet each model's
``load_weights`` re-derives it per tensor per reload through substring scans
over stacked/expert mapping lists and per-model rewrite logic — measured at
81-99% of reload wall time on large MoE checkpoints.

``LoadPlan`` provides two reload fast paths on top of one recorded mapping:

- **Replay** (full reloads): the first reload runs the model's own
  ``load_weights`` unchanged while every parameter's ``weight_loader`` is
  transparently wrapped, recording *source name -> [(param fqn, loader
  args)]*. Later reloads dispatch each name's recorded calls directly —
  O(1) lookup, no per-model scan — with host-tensor work batched onto a
  thread pool so H2D copies overlap for every architecture.
Invariants the implementation depends on (each one was earned by a real
failure — treat them as load-bearing):

1. *Attribution rides the tensor, not control flow.* Each yielded tensor is
   tagged with its source name; a loader call whose tensor lost the tag was
   derived in flight (fused, re-quantized) and is never recorded as a
   replayable entry. Names whose loads the boundary cannot represent stream
   through the model's own ``load_weights`` on every replay.
2. *Fallback streaming preserves per-model tails.* Models like DeepSeek run
   ``post_load_weights(weight_names=...)`` keyed on the names their
   ``load_weights`` consumed; because fallback names are streamed through the
   model's own loader, those tails see exactly the names they must act on.
   ``load_plan_fallback_patterns`` forces names into this path (e.g.
   ``kv_b_proj``, whose MLA re-derivation is name-gated).
3. *Loader-less params are intercepted by installing the wrapper as the
   attribute.* Norms and plain biases load via ``getattr(param,
   "weight_loader", default_weight_loader)``; during record the wrapper IS
   the attribute (removed on restore), so their loads are attributed too.

Models opt in with ``supports_load_plan_replay = True``; the feature is
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
from sglang.srt.model_loader.weight_utils import default_weight_loader

logger = logging.getLogger(__name__)

_SOURCE_TAG = "_load_plan_source_name"
# Dispatch in batches: one executor submit per batch, so per-name queue and
# future churn amortizes away on 100k+-tensor checkpoints. The byte cap
# bounds how many CPU tensors a batch keeps alive.
_BATCH_NAMES = 256
_BATCH_BYTES = 256 << 20
# Soft bound on outstanding batches, so replay cannot materialize unboundedly
# many CPU tensors ahead of the H2D copies that release them.
_MAX_INFLIGHT = 64


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

    # ------------------------------------------------------------------ record

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
                # Loader-less params (norms, plain biases) load through
                # default_weight_loader via getattr-with-default in the model
                # loops — installing the wrapper AS the attribute intercepts
                # them identically (removed on restore).
                created = loader is None
                if created:
                    loader = default_weight_loader

                def make_wrapper(fqn: str = fqn, loader: Any = loader):
                    def recording_loader(param_arg, tensor, *args, **kwargs):
                        source = getattr(tensor, _SOURCE_TAG, None)
                        if source is not None:
                            with lock:
                                recorded.setdefault(source, []).append(
                                    (fqn, args, kwargs)
                                )
                        # A loader call whose tensor lost the tag was derived in
                        # flight (fused/re-quantized); it stays unrecorded, so
                        # its source name falls back to the model's own loader.
                        return loader(param_arg, tensor, *args, **kwargs)

                    return recording_loader

                slot = swap_loader(param, make_wrapper())
                wrapped.append((param, slot, None if created else loader))

            model.load_weights(tagged())
        finally:
            for param, slot, loader in wrapped:
                if loader is None:
                    delattr(param, slot)
                else:
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

    # ------------------------------------------------------------------ replay

    def replay(
        self,
        model: torch.nn.Module,
        weights: Iterable[Tuple[str, torch.Tensor]],
        max_workers: int = 8,
    ) -> Dict[str, Any]:
        """Dispatch recorded names directly (batched onto a thread pool);
        stream everything else through the model's own load_weights, which
        also runs any per-model post-load tail on exactly those names."""
        start = time.perf_counter()
        params_dict = dict(model.named_parameters())
        counts = {"hit": 0, "fallback": 0, "unknown": 0}
        futures: List[concurrent.futures.Future] = []
        batch: List[Tuple[torch.Tensor, List[Tuple[str, tuple, dict]]]] = []
        batch_bytes = [0]

        def dispatch(
            tensor: torch.Tensor, calls: List[Tuple[str, tuple, dict]]
        ) -> None:
            for fqn, args, kwargs in calls:
                param = params_dict[fqn]
                loader = getattr(param, "weight_loader", None) or default_weight_loader
                loader(param, tensor, *args, **kwargs)

        def dispatch_batch(
            items: List[Tuple[torch.Tensor, List[Tuple[str, tuple, dict]]]],
        ) -> None:
            for tensor, calls in items:
                dispatch(tensor, calls)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

            def flush() -> None:
                if not batch:
                    return
                if len(futures) >= _MAX_INFLIGHT:
                    for future in futures:
                        future.result()
                    futures.clear()
                futures.append(executor.submit(dispatch_batch, batch[:]))
                batch.clear()
                batch_bytes[0] = 0

            def filtered() -> Iterable[Tuple[str, torch.Tensor]]:
                for name, tensor in weights:
                    calls = self.entries.get(name)
                    if calls is None:
                        counts["fallback" if name in self.fallback else "unknown"] += 1
                        yield name, tensor
                        continue
                    counts["hit"] += 1
                    if should_async_load(tensor):
                        batch.append((tensor, calls))
                        batch_bytes[0] += tensor.numel() * tensor.element_size()
                        if len(batch) >= _BATCH_NAMES or batch_bytes[0] >= _BATCH_BYTES:
                            flush()
                    else:
                        dispatch(tensor, calls)
                flush()

            model.load_weights(filtered())
            for future in futures:
                future.result()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

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
