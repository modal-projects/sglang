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
- **Partial reload** (O(delta)): given the touched checkpoint names of a
  weight update, reload only those tensors (plus the closures below), then
  incrementally re-run post-loading for exactly the touched modules (see
  ``process_weights_after_partial_loading`` on quant methods). At QAT-style
  update densities this replaces an O(model) reload with seconds of work.

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
4. *Derived calls on the feeder's thread attribute to the name being
   consumed.* Models dispatch fused loads either inline (attributable) or on
   executor threads with no tag (not attributable); the thread check keeps
   the inline attribution unambiguous. Executor-thread fusions must be
   declared via ``load_plan_fused_aliases`` (validated against real param
   fqns) or their names fall back to full reloads.
5. *Partial reloads use expert-granular closure.* Post-loading transforms
   consume an expert's weights AND scales together, and the full pass may
   destroy raw forms in place — so every checkpoint tensor of a touched
   expert reloads raw before the incremental transform re-runs.
6. *Anything unresolvable declines to a full reload*, which is also the
   recovery path for a partially-applied pass: it rewrites every weight and
   re-runs the global post-loading.

Models opt in with ``supports_load_plan_replay = True``; the feature is
additionally gated behind ``SGLANG_ENABLE_RELOAD_LOAD_PLAN=1``.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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

    def __init__(
        self,
        fallback_patterns: Iterable[str] = (),
        fused_aliases: Iterable[Tuple[str, str]] = (),
    ) -> None:
        # source checkpoint name -> [(param fqn, loader args, loader kwargs)]
        self.entries: Dict[str, List[Tuple[str, tuple, dict]]] = {}
        # Names that must always go through the model's own load_weights.
        self.fallback: Set[str] = set()
        # source name -> param fqns it (eventually) wrote, kept for EVERY name
        # incl. fallback ones — partial reloads use this to find the modules a
        # touched checkpoint tensor feeds.
        self.effects: Dict[str, List[str]] = {}
        self.fallback_patterns = tuple(fallback_patterns)
        # Declared (name substring -> param substring) rewrites for buffered
        # fusions whose loader call fires on a worker thread with no source
        # tag (e.g. DeepSeek q/kv_a_proj -> fused_qkv_a_proj_with_mqa).
        # Candidates are validated against real param fqns before use.
        self.fused_aliases = tuple(fused_aliases)
        # module fqn -> expert id -> the checkpoint names feeding that expert.
        self.expert_index: Dict[str, Dict[int, Set[str]]] = {}
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
        derived_effects: Dict[str, List[str]] = {}
        seen: Set[str] = set()
        lock = threading.Lock()
        # For loader calls whose tensor lost the tag (fused/derived), attribute
        # the touched params to the name being consumed — but only for calls on
        # the feeder's own thread, where "current name" is unambiguous (models
        # dispatch derived fusions inline; async dispatch is tag-pure).
        feeder = {"name": None, "thread": threading.get_ident()}

        def tagged() -> Iterable[Tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                seen.add(name)
                feeder["name"] = name
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
                                recorded.setdefault(source, []).append((fqn, args, kwargs))
                        elif threading.get_ident() == feeder["thread"] and feeder["name"]:
                            with lock:
                                derived_effects.setdefault(feeder["name"], []).append(fqn)
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
        self.effects = {name: [fqn for fqn, _, _ in calls] for name, calls in recorded.items()}
        for name, fqns in derived_effects.items():
            self.effects.setdefault(name, []).extend(fqns)
        # Seen-but-unrecorded names have effects the boundary cannot represent
        # (derived tensors, buffered fusions) or none at all (skipped names);
        # both are only correct through the model's own loader. Their observed
        # param effects are still kept in `effects` for module attribution.
        self.fallback = {name for name in seen if name not in recorded}
        self.fallback.update(name for name in recorded if self._forced_fallback(name))
        for name in self.fallback:
            self.entries.pop(name, None)
        # (module fqn -> expert id -> checkpoint names): per-expert transforms
        # (fp4 shuffle/swizzle) consume an expert's weights AND scales together,
        # so a partial reload must refresh every tensor of a touched expert.
        self.expert_index = {}
        for name, calls in self.entries.items():
            for fqn, args, kwargs in calls:
                expert_id = kwargs.get("expert_id")
                if expert_id is None:
                    continue
                module_fqn = fqn.rsplit(".", 1)[0]
                self.expert_index.setdefault(module_fqn, {}).setdefault(int(expert_id), set()).add(name)
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

        def dispatch(tensor: torch.Tensor, calls: List[Tuple[str, tuple, dict]]) -> None:
            for fqn, args, kwargs in calls:
                param = params_dict[fqn]
                loader = getattr(param, "weight_loader", None) or default_weight_loader
                loader(param, tensor, *args, **kwargs)

        def dispatch_batch(items: List[Tuple[torch.Tensor, List[Tuple[str, tuple, dict]]]]) -> None:
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

    # ----------------------------------------------------------------- partial

    def _alias_fqn(self, name: str, param_fqns: Set[str]) -> Optional[str]:
        for src_sub, dst_sub in self.fused_aliases:
            if src_sub in name:
                candidate = name.replace(src_sub, dst_sub)
                if candidate in param_fqns:
                    return candidate
        return None

    def touched_plan(
        self, touched: Iterable[str], param_fqns: Set[str]
    ) -> "Optional[Tuple[Dict[str, Dict[str, Set[int]]], Set[str]]]":
        """Resolve touched checkpoint names to (per-module touched detail,
        the names to reload).

        detail: module fqn -> param attr name -> expert ids whose slices the
        touched names rewrite (empty set for whole-param / non-expert
        touches). Reload names are the touched names themselves plus two
        closures: every sibling input of a touched fused param (the model
        re-fuses only from a complete part set) and every tensor of a touched
        expert (per-expert transforms consume weights AND scales together).
        Inert names (seen at record, zero effects, no alias) are skipped.
        None when any touched name cannot be attributed — the caller must
        full-reload.
        """
        detail: Dict[str, Dict[str, Set[int]]] = {}
        reload_names: Set[str] = set()
        touched_fused: Set[str] = set()
        inert = 0

        def note(fqn: str, expert_id: Optional[int]) -> None:
            module_fqn, param_name = fqn.rsplit(".", 1)
            experts = detail.setdefault(module_fqn, {}).setdefault(param_name, set())
            if expert_id is not None:
                experts.add(int(expert_id))

        for name in touched:
            calls = self.entries.get(name)
            if calls:
                reload_names.add(name)
                for fqn, args, kwargs in calls:
                    note(fqn, kwargs.get("expert_id"))
                continue
            fqns = self.effects.get(name)
            if fqns:
                reload_names.add(name)
                for fqn in fqns:
                    note(fqn, None)
                continue
            alias = self._alias_fqn(name, param_fqns)
            if alias is not None:
                reload_names.add(name)
                touched_fused.add(alias)
                note(alias, None)
                continue
            if name in self.fallback:
                inert += 1  # seen at record, provably no effect on params
                continue
            logger.info(
                f"[load plan] partial reload falling back to full: "
                f"no recorded param attribution for touched name {name!r}"
            )
            return None
        if inert:
            logger.info(f"[load plan] partial reload skipping {inert} inert touched names")
        # Expert closure: refresh every tensor of a touched expert so per-expert
        # re-transforms consume a complete raw slot.
        for module_fqn, param_experts in list(detail.items()):
            experts = set().union(*param_experts.values()) if param_experts else set()
            for expert_id in experts:
                for name in self.expert_index.get(module_fqn, {}).get(expert_id, ()):
                    if name not in reload_names:
                        reload_names.add(name)
                        for fqn, args, kwargs in self.entries.get(name, ()):
                            note(fqn, kwargs.get("expert_id"))
        # A touched fused param needs ALL its declared inputs streamed.
        if touched_fused:
            for name in self.fallback:
                alias = self._alias_fqn(name, param_fqns)
                if alias in touched_fused:
                    reload_names.add(name)
        return detail, reload_names

    def partial_replay(
        self,
        model: torch.nn.Module,
        checkpoint_dir: str,
        touched: List[str],
        max_workers: int = 8,
        resolved: "Optional[Tuple[Dict[str, Dict[str, Set[int]]], Set[str]]]" = None,
    ) -> "Optional[Tuple[Dict[str, Any], Dict[str, Dict[str, Set[int]]]]]":
        """Reload ONLY the touched checkpoint names (plus their closures),
        reading just their tensors from disk. Returns (stats, per-module
        touched detail) for the caller's incremental postprocess, or None when
        a full reload is required (unattributable names, or a name missing
        from the index). Pass ``resolved`` (a prior ``touched_plan`` result)
        to skip re-resolving — callers use it to pre-flight quant-method
        support before any tensor is read."""
        if not self.recorded:
            return None
        if resolved is None:
            resolved = self.touched_plan(touched, {fqn for fqn, _ in model.named_parameters()})
        if resolved is None:
            return None
        detail, reload_names = resolved
        start = time.perf_counter()
        weights = partial_weights_iterator(checkpoint_dir, sorted(reload_names))
        if weights is None:
            logger.info(
                "[load plan] partial reload falling back to full: "
                "checkpoint index missing or a closure name absent from it"
            )
            return None
        stats = self.replay(model, weights, max_workers=max_workers)
        stats.update(
            plan="partial",
            plan_touched=len(touched),
            plan_modules=len(detail),
            plan_closure=len(reload_names),
            plan_replay_s=round(time.perf_counter() - start, 2),
        )
        logger.info(
            f"[load plan] partial pass: {stats['plan_touched']} touched names -> "
            f"{stats['plan_modules']} modules, {stats['plan_closure']} reloaded names "
            f"in {stats['plan_replay_s']}s"
        )
        return stats, detail


def partial_weights_iterator(
    checkpoint_dir: str, names: List[str]
) -> "Optional[Iterable[Tuple[str, torch.Tensor]]]":
    """Random-access reads of exactly `names` from a safetensors checkpoint,
    grouped by shard so each needed file opens once. None when the index is
    missing or a name is absent (caller falls back to a full reload)."""
    import json
    import os

    from safetensors import safe_open

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    try:
        with open(index_path) as f:
            weight_map: Dict[str, str] = json.load(f)["weight_map"]
    except (OSError, KeyError, ValueError):
        return None
    by_file: Dict[str, List[str]] = {}
    for name in names:
        shard = weight_map.get(name)
        if shard is None:
            return None
        by_file.setdefault(shard, []).append(name)

    def read() -> Iterable[Tuple[str, torch.Tensor]]:
        for shard, shard_names in sorted(by_file.items()):
            with safe_open(os.path.join(checkpoint_dir, shard), framework="pt", device="cpu") as f:
                for name in shard_names:
                    yield name, f.get_tensor(name)

    return read()


def get_or_create_plan(model: torch.nn.Module) -> "LoadPlan | None":
    """The model instance's reload plan, if it opted in (else None)."""
    if not getattr(model, "supports_load_plan_replay", False):
        return None
    plan = getattr(model, "_reload_load_plan", None)
    if plan is None:
        plan = LoadPlan(
            getattr(model, "load_plan_fallback_patterns", ()),
            fused_aliases=getattr(model, "load_plan_fused_aliases", ()),
        )
        model._reload_load_plan = plan
    return plan
