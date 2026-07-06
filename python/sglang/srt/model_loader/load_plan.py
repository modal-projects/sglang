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
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.model_loader.utils import should_async_load
from sglang.srt.model_loader.weight_utils import default_weight_loader

logger = logging.getLogger(__name__)

_SOURCE_TAG = "_load_plan_source_name"
# Soft bound on outstanding async loader calls, so replay cannot materialize
# unboundedly many CPU tensors ahead of the H2D copies that release them.
_MAX_INFLIGHT = 4096
# Compiled programs are executed in batches: one executor submit per batch, so
# per-name queue/future churn amortizes away on 100k+-tensor checkpoints. A
# byte cap keeps a batch's referenced CPU tensors bounded.
_BATCH_NAMES = 256
_BATCH_BYTES = 256 << 20
# Per-worker pinned staging arena. A pageable H2D copy_ is a synchronous
# cudaMemcpy (~0.5ms each — the dominant fast-pass cost at 276k copies);
# staging the source through pinned memory makes every device copy an async
# launch. Tensors larger than the arena fall back to the pageable path.
_ARENA_BYTES = 512 << 20
# Pinned staging is OFF by default: measured across whole-tensor, per-copy,
# size-gated, and contiguity-gated variants, the per-copy Python + memcpy it
# adds always lost to the direct pageable path at both copy-size regimes
# (GLM 55->71s, K2.6 390->685-860s). The machinery stays for experiments:
# set SGLANG_LOAD_PLAN_STAGE_MAX_BYTES to a byte threshold to enable.
_STAGE_MAX_BYTES = int(os.environ.get("SGLANG_LOAD_PLAN_STAGE_MAX_BYTES", "0"))


class _PinnedStager(threading.local):
    """Thread-local pinned arena + CUDA stream for async H2D program copies.

    Sources are staged whole into the arena (one CPU memcpy), then each
    program copy launches async from the arena view on this thread's stream.
    When the arena wraps, the stream is synchronized before reuse.
    """

    def __init__(self) -> None:
        self.arena: Optional[torch.Tensor] = None
        self.stream: Optional[torch.cuda.Stream] = None
        self.offset = 0

    def ready(self) -> bool:
        if not torch.cuda.is_available():
            return False
        if self.arena is None:
            try:
                self.arena = torch.empty(_ARENA_BYTES, dtype=torch.uint8, pin_memory=True)
            except RuntimeError:  # pinning refused (memlock limits): stay pageable
                return False
            self.stream = torch.cuda.Stream()
        return True

    def stage(self, view: torch.Tensor) -> Optional[torch.Tensor]:
        """A pinned, contiguous copy of exactly `view`'s elements, valid until
        the next arena wrap. Staging per copy (not per source tensor) matters:
        TP-sliced copies read a fraction of their source, and staging the whole
        tensor amplifies CPU memcpy traffic by the TP factor."""
        nbytes = view.numel() * view.element_size()
        if nbytes > _ARENA_BYTES:
            return None
        self.offset = (self.offset + 15) & ~15  # dtype-view alignment
        if self.offset + nbytes > _ARENA_BYTES:
            self.stream.synchronize()  # in-flight copies still read the arena
            self.offset = 0
        staged = (
            self.arena[self.offset : self.offset + nbytes]
            .view(view.dtype)
            .reshape(view.shape)
        )
        staged.copy_(view)  # strided-source CPU memcpy of only the read bytes
        self.offset += nbytes
        return staged

# One compiled copy: dst/src expressed relative to the param's / input
# tensor's own storage offset, so both survive storage rebinds and per-reload
# CPU buffers. (fqn, dst_off_delta, dst_size, dst_stride, src_off_delta,
# src_size, src_stride)
_Copy = Tuple[str, int, tuple, tuple, int, tuple, tuple]


class _CallObserver(TorchDispatchMode):
    """Watches one weight_loader call and extracts its effect on the model
    params as raw strided copies — or taints the call when its effect cannot
    be expressed that way.

    Loaders frequently materialize an input region before copying it into the
    param (``narrow().contiguous()``, ``.to(dtype)``, ``empty().copy_(view)``
    staging). Those temporaries are tracked by provenance: a temp whose entire
    contents are a row-major materialization of one strided input region maps
    back to that region, and a whole-temp copy into a param compiles to a
    direct copy from the original input view (``copy_`` reads strided sources
    and casts dtypes itself, so the temp is skippable). Anything the
    provenance can't express taints the call and it keeps dispatching."""

    def __init__(self, param_storage_to_fqn: Dict[int, str], params: Dict[str, torch.nn.Parameter]):
        super().__init__()
        self.param_storage_to_fqn = param_storage_to_fqn
        self.params = params
        self.input_tensor: Optional[torch.Tensor] = None
        self.copies: List[_Copy] = []
        self.tainted = False
        # temp storage ptr -> (src_off_delta, size, stride) of the ONE input
        # region the temp's full contents materialize, in row-major order.
        self.provenance: Dict[int, Tuple[int, tuple, tuple]] = {}

    def begin_call(self, input_tensor: torch.Tensor) -> None:
        self.input_tensor = input_tensor
        self.copies = []
        self.tainted = False
        self.provenance = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        try:
            self._observe(func, args, kwargs, out)
        except Exception:  # noqa: BLE001 — observation must never break the load
            self.tainted = True
        return out

    def _input_view_spec(self, view: torch.Tensor) -> Optional[Tuple[int, tuple, tuple]]:
        """The input region a tensor reads: directly (a view of the input) or
        transitively (the whole contents of a tracked temp)."""
        ptr = view.untyped_storage().data_ptr()
        if ptr == self.input_tensor.untyped_storage().data_ptr():
            return (
                view.storage_offset() - self.input_tensor.storage_offset(),
                tuple(view.shape),
                tuple(view.stride()),
            )
        spec = self.provenance.get(ptr)
        if spec is not None and _reads_whole_storage(view, spec):
            return spec
        return None

    def _observe(self, func, args, kwargs, out) -> None:
        aten = torch.ops.aten
        if func is aten.copy_.default:
            dst, src = args[0], args[1]
            dst_ptr = dst.untyped_storage().data_ptr()
            fqn = self.param_storage_to_fqn.get(dst_ptr)
            src_spec = self._input_view_spec(src)
            if fqn is not None:
                if src_spec is None:
                    self.tainted = True  # source underivable from the input
                    return
                base = self.params[fqn].data
                self.copies.append(
                    (
                        fqn,
                        dst.storage_offset() - base.storage_offset(),
                        tuple(dst.shape),
                        tuple(dst.stride()),
                        *src_spec,
                    )
                )
                return
            # Staging copy into a temp (`empty().copy_(view)`): track it when
            # the temp is written whole, in row-major order.
            if src_spec is not None and dst.is_contiguous() and dst.storage_offset() == 0:
                self.provenance[dst_ptr] = src_spec
            else:
                self.provenance.pop(dst_ptr, None)  # partially/oddly written: distrust
            return
        # Materializing producers: clone/contiguous/cast of an input region.
        if func in (aten.clone.default, aten._to_copy.default, aten.contiguous.default):
            source = args[0]
            if isinstance(out, torch.Tensor) and isinstance(source, torch.Tensor):
                spec = self._input_view_spec(source)
                if spec is not None and out.is_contiguous() and out.storage_offset() == 0:
                    self.provenance[out.untyped_storage().data_ptr()] = spec
            return
        # Any other mutation of a param storage is inexpressible: taint. A
        # mutation of a tracked temp invalidates its provenance.
        schema = getattr(func, "_schema", None)
        if schema is None:
            return
        for i, arg in enumerate(schema.arguments):
            if arg.alias_info is None or not arg.alias_info.is_write:
                continue
            value = args[i] if i < len(args) else kwargs.get(arg.name)
            if not isinstance(value, torch.Tensor):
                continue
            ptr = value.untyped_storage().data_ptr()
            if ptr in self.param_storage_to_fqn:
                self.tainted = True
                return
            self.provenance.pop(ptr, None)


def _reads_whole_storage(view: torch.Tensor, spec: Tuple[int, tuple, tuple]) -> bool:
    """True when `view` reads its temp's full materialized contents in
    row-major order (so the temp's provenance spec substitutes exactly).
    Shape may differ from the spec's (loaders reshape); numel must match."""
    numel = 1
    for dim in spec[1]:
        numel *= dim
    return view.is_contiguous() and view.storage_offset() == 0 and view.numel() == numel


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
        self.recorded = False
        # source name -> (input shape, input dtype, [compiled copies]) for
        # names whose loader effect reduced to raw strided copies; built by the
        # first (observed) replay. Names that observed as inexpressible map to
        # None and keep dispatching through their weight_loader.
        self.programs: Dict[str, Optional[Tuple[tuple, torch.dtype, List[_Copy]]]] = {}
        self.compiled = False

    def _forced_fallback(self, name: str) -> bool:
        return any(pattern in name for pattern in self.fallback_patterns)

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
        allow_compile: bool = True,
    ) -> Dict[str, Any]:
        """Dispatch recorded names directly; stream everything else through the
        model's own load_weights, which also runs any per-model post-load tail
        on exactly those names.

        The first replay runs each dispatched weight_loader call under a
        TorchDispatchMode observer and compiles calls whose entire effect is
        strided copies of the input into param storages down to raw copy
        programs; later replays execute those programs directly — no
        weight_loader Python at all — and only dispatch the inexpressible
        remainder. Host-tensor work goes through a thread pool either way."""
        start = time.perf_counter()
        params_dict = dict(model.named_parameters())
        counts = {"hit": 0, "fallback": 0, "unknown": 0, "compiled": 0, "dispatched": 0}
        futures: List[concurrent.futures.Future] = []
        batch: List[Tuple[torch.Tensor, List[_Copy]]] = []
        batch_bytes = [0]
        stager = _PinnedStager()  # thread-local arenas/streams, per replay pass
        # Partial passes never compile: a program map built from a partial
        # stream would mark the plan compiled while covering few names.
        compiling = allow_compile and not self.compiled

        def dispatch(fqn: str, tensor: torch.Tensor, args: tuple, kwargs: dict) -> None:
            param = params_dict[fqn]
            loader = getattr(param, "weight_loader", None) or default_weight_loader
            loader(param, tensor, *args, **kwargs)

        def run_program(tensor: torch.Tensor, copies: List[_Copy]) -> None:
            # Stage each copy's source view through this thread's pinned arena:
            # pageable H2D copies are synchronous cudaMemcpys, and at hundreds
            # of thousands of copies their per-call stalls dominate.
            use_stager = stager.ready()
            for fqn, dst_off, dst_size, dst_stride, src_off, src_size, src_stride in copies:
                base = params_dict[fqn].data
                dst = torch.as_strided(base, dst_size, dst_stride, base.storage_offset() + dst_off)
                src = torch.as_strided(tensor, src_size, src_stride, tensor.storage_offset() + src_off)
                if src.shape != dst.shape:
                    # Provenance-composed sources keep the input region's shape;
                    # the observed copy went through a reshaped temp. Equal numel
                    # was implied by the observed copy_ succeeding.
                    src = src.reshape(dst.shape)
                staged = (
                    stager.stage(src)
                    if use_stager
                    # Contiguous only: staging a strided view is an elementwise
                    # CPU gather (~2ms per 2MB copy — measured 785s of K2.6
                    # fast-pass load), not a memcpy. Strided sources stay on
                    # the direct pageable path, which gathers once on device
                    # transfer anyway.
                    and src.is_contiguous()
                    and src.numel() * src.element_size() <= _STAGE_MAX_BYTES
                    else None
                )
                if staged is not None:
                    with torch.cuda.stream(stager.stream):
                        dst.copy_(staged, non_blocking=True)
                else:
                    dst.copy_(src, non_blocking=True)

        def run_program_batch(items: List[Tuple[torch.Tensor, List[_Copy]]]) -> None:
            for tensor, copies in items:
                run_program(tensor, copies)

        observer = (
            _CallObserver({p.data.untyped_storage().data_ptr(): fqn for fqn, p in params_dict.items()}, params_dict)
            if compiling
            else None
        )

        def submit(fn, *args) -> None:
            if len(futures) >= _MAX_INFLIGHT:
                for future in futures:
                    future.result()
                futures.clear()
            futures.append(executor.submit(fn, *args))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

            def filtered() -> Iterable[Tuple[str, torch.Tensor]]:
                for name, tensor in weights:
                    calls = self.entries.get(name)
                    if calls is None:
                        counts["fallback" if name in self.fallback else "unknown"] += 1
                        yield name, tensor
                        continue
                    counts["hit"] += 1
                    if compiling:
                        # Observe synchronously (dispatch modes are thread-local).
                        observer.begin_call(tensor)
                        with observer:
                            for fqn, args, kwargs in calls:
                                dispatch(fqn, tensor, args, kwargs)
                        self.programs[name] = (
                            None
                            if observer.tainted or not observer.copies
                            else (tuple(tensor.shape), tensor.dtype, observer.copies)
                        )
                        continue
                    program = self.programs.get(name)
                    if program is not None and program[0] == tuple(tensor.shape) and program[1] == tensor.dtype:
                        counts["compiled"] += 1
                        if should_async_load(tensor):
                            # Batch program executions: per-name submits cost
                            # ~1ms of executor/GIL churn, which dominates a
                            # multi-hundred-thousand-tensor checkpoint.
                            batch.append((tensor, program[2]))
                            batch_bytes[0] += tensor.numel() * tensor.element_size()
                            if len(batch) >= _BATCH_NAMES or batch_bytes[0] >= _BATCH_BYTES:
                                submit(run_program_batch, batch[:])
                                batch.clear()
                                batch_bytes[0] = 0
                        else:
                            run_program(tensor, program[2])
                    else:
                        counts["dispatched"] += 1
                        if should_async_load(tensor):
                            for fqn, args, kwargs in calls:
                                submit(dispatch, fqn, tensor, args, kwargs)
                        else:
                            for fqn, args, kwargs in calls:
                                dispatch(fqn, tensor, args, kwargs)
                if batch:
                    submit(run_program_batch, batch[:])
                    batch.clear()

            model.load_weights(filtered())
            for future in futures:
                future.result()

        if compiling:
            self.compiled = True
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # non_blocking program copies must land before postprocess

        stats = {
            "plan": "compile" if compiling else "fast",
            "plan_hits": counts["hit"],
            "plan_fallback": counts["fallback"],
            "plan_unknown": counts["unknown"],
            "plan_replay_s": round(time.perf_counter() - start, 2),
        }
        if compiling:
            stats["plan_compiled"] = sum(1 for p in self.programs.values() if p is not None)
        else:
            stats["plan_compiled"] = counts["compiled"]
            stats["plan_dispatched"] = counts["dispatched"]
        logger.info(
            f"[load plan] {stats['plan']} pass: {stats['plan_hits']} names "
            f"({stats['plan_compiled']} compiled, {stats['plan_fallback']} fallback, "
            f"{stats['plan_unknown']} unknown) in {stats['plan_replay_s']}s"
        )
        return stats


    def _alias_fqn(self, name: str, param_fqns: Set[str]) -> Optional[str]:
        for src_sub, dst_sub in self.fused_aliases:
            if src_sub in name:
                candidate = name.replace(src_sub, dst_sub)
                if candidate in param_fqns:
                    return candidate
        return None

    def module_closure(
        self, touched: Iterable[str], param_fqns: Set[str]
    ) -> "Optional[Tuple[Set[str], Set[str]]]":
        """Expand touched checkpoint names to (touched module fqns, the full
        set of names feeding those modules).

        Postprocess consumes a module's params in their RAW loaded state, and
        a prior postprocess may have transformed them in place — so every name
        feeding a touched module must be reloaded, not just the changed ones.
        Returns None when any touched name has no recorded param attribution
        (derived-only fusions, never-seen names): the caller must fall back to
        a full reload.
        """
        touched_modules: Set[str] = set()
        inert = 0
        for name in touched:
            fqns = self.effects.get(name)
            if not fqns:
                alias = self._alias_fqn(name, param_fqns)
                if alias is not None:
                    fqns = [alias]
                elif name in self.fallback:
                    # Seen at record but produced no param effect and matches
                    # no declared fusion: the model ignores this tensor (e.g.
                    # MTP weights on a non-speculative deployment) — a change
                    # to it cannot affect engine state.
                    inert += 1
                    continue
                else:
                    logger.info(
                        f"[load plan] partial reload falling back to full: "
                        f"no recorded param attribution for touched name {name!r}"
                    )
                    return None
            for fqn in fqns:
                touched_modules.add(fqn.rsplit(".", 1)[0])
        if inert:
            logger.info(f"[load plan] partial reload skipping {inert} inert touched names")
        module_names: Set[str] = set()
        for name, fqns in self.effects.items():
            if any(fqn.rsplit(".", 1)[0] in touched_modules for fqn in fqns):
                module_names.add(name)
        # Fusion inputs have no effects entries; collect them by alias so the
        # model re-fuses from a complete part set.
        if self.fused_aliases:
            for name in self.fallback:
                alias = self._alias_fqn(name, param_fqns)
                if alias is not None and alias.rsplit(".", 1)[0] in touched_modules:
                    module_names.add(name)
        return touched_modules, module_names

    def partial_replay(
        self,
        model: torch.nn.Module,
        checkpoint_dir: str,
        touched: List[str],
        max_workers: int = 8,
    ) -> "Optional[Tuple[Dict[str, Any], Set[str]]]":
        """Reload ONLY the modules the touched checkpoint names feed, reading
        just their tensors from disk. Returns (stats, touched module fqns) for
        the caller's filtered postprocess, or None when a full reload is
        required (unattributable names, or a name missing from the index)."""
        if not self.recorded:
            return None
        closure = self.module_closure(touched, {fqn for fqn, _ in model.named_parameters()})
        if closure is None:
            return None
        touched_modules, closure_names = closure
        start = time.perf_counter()
        weights = partial_weights_iterator(checkpoint_dir, sorted(closure_names))
        if weights is None:
            logger.info(
                "[load plan] partial reload falling back to full: "
                "checkpoint index missing or a closure name absent from it"
            )
            return None
        stats = self.replay(model, weights, max_workers=max_workers, allow_compile=False)
        stats.update(
            plan="partial",
            plan_touched=len(touched),
            plan_modules=len(touched_modules),
            plan_closure=len(closure_names),
            plan_replay_s=round(time.perf_counter() - start, 2),
        )
        logger.info(
            f"[load plan] partial pass: {stats['plan_touched']} touched names -> "
            f"{stats['plan_modules']} modules, {stats['plan_closure']} reloaded names "
            f"in {stats['plan_replay_s']}s"
        )
        return stats, touched_modules


def partial_weights_iterator(
    checkpoint_dir: str, names: List[str]
) -> "Optional[Iterable[Tuple[str, torch.Tensor]]]":
    """Random-access reads of exactly `names` from a safetensors checkpoint,
    grouped by shard so each needed file opens once. None when the index is
    missing or a name is absent (caller falls back to a full reload)."""
    import json

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
