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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.model_loader.utils import should_async_load

logger = logging.getLogger(__name__)

_SOURCE_TAG = "_load_plan_source_name"
# Soft bound on outstanding async loader calls, so replay cannot materialize
# unboundedly many CPU tensors ahead of the H2D copies that release them.
_MAX_INFLIGHT = 4096

# One compiled copy: dst/src expressed relative to the param's / input
# tensor's own storage offset, so both survive storage rebinds and per-reload
# CPU buffers. (fqn, dst_off_delta, dst_size, dst_stride, src_off_delta,
# src_size, src_stride)
_Copy = Tuple[str, int, tuple, tuple, int, tuple, tuple]


class _CallObserver(TorchDispatchMode):
    """Watches one weight_loader call and extracts its effect on the model
    params as raw strided copies — or taints the call when its effect cannot
    be expressed that way (derived temporaries, non-copy mutations)."""

    def __init__(self, param_storage_to_fqn: Dict[int, str], params: Dict[str, torch.nn.Parameter]):
        super().__init__()
        self.param_storage_to_fqn = param_storage_to_fqn
        self.params = params
        self.input_tensor: Optional[torch.Tensor] = None
        self.copies: List[_Copy] = []
        self.tainted = False

    def begin_call(self, input_tensor: torch.Tensor) -> None:
        self.input_tensor = input_tensor
        self.copies = []
        self.tainted = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        try:
            self._observe(func, args, kwargs)
        except Exception:  # noqa: BLE001 — observation must never break the load
            self.tainted = True
        return out

    def _observe(self, func, args, kwargs) -> None:
        if func is torch.ops.aten.copy_.default:
            dst, src = args[0], args[1]
            fqn = self.param_storage_to_fqn.get(dst.untyped_storage().data_ptr())
            if fqn is None:
                return  # write to a temporary: not a param effect
            inp = self.input_tensor
            if src.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr():
                self.tainted = True  # source is derived, not a view of the input
                return
            base = self.params[fqn].data
            self.copies.append(
                (
                    fqn,
                    dst.storage_offset() - base.storage_offset(),
                    tuple(dst.shape),
                    tuple(dst.stride()),
                    src.storage_offset() - inp.storage_offset(),
                    tuple(src.shape),
                    tuple(src.stride()),
                )
            )
            return
        # Any other mutation of a param storage is inexpressible: taint.
        schema = getattr(func, "_schema", None)
        if schema is None:
            return
        for i, arg in enumerate(schema.arguments):
            if arg.alias_info is None or not arg.alias_info.is_write:
                continue
            value = args[i] if i < len(args) else kwargs.get(arg.name)
            if (
                isinstance(value, torch.Tensor)
                and value.untyped_storage().data_ptr() in self.param_storage_to_fqn
            ):
                self.tainted = True
                return


class LoadPlan:
    """One model instance's recorded reload dispatch (see module docstring)."""

    def __init__(self, fallback_patterns: Iterable[str] = ()) -> None:
        # source checkpoint name -> [(param fqn, loader args, loader kwargs)]
        self.entries: Dict[str, List[Tuple[str, tuple, dict]]] = {}
        # Names that must always go through the model's own load_weights.
        self.fallback: Set[str] = set()
        self.fallback_patterns = tuple(fallback_patterns)
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
        compiling = not self.compiled

        def dispatch(fqn: str, tensor: torch.Tensor, args: tuple, kwargs: dict) -> None:
            param = params_dict[fqn]
            param.weight_loader(param, tensor, *args, **kwargs)

        def run_program(tensor: torch.Tensor, copies: List[_Copy]) -> None:
            for fqn, dst_off, dst_size, dst_stride, src_off, src_size, src_stride in copies:
                base = params_dict[fqn].data
                dst = torch.as_strided(base, dst_size, dst_stride, base.storage_offset() + dst_off)
                src = torch.as_strided(tensor, src_size, src_stride, tensor.storage_offset() + src_off)
                dst.copy_(src, non_blocking=True)

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
                            submit(run_program, tensor, program[2])
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


def get_or_create_plan(model: torch.nn.Module) -> "LoadPlan | None":
    """The model instance's reload plan, if it opted in (else None)."""
    if not getattr(model, "supports_load_plan_replay", False):
        return None
    plan = getattr(model, "_reload_load_plan", None)
    if plan is None:
        plan = LoadPlan(getattr(model, "load_plan_fallback_patterns", ()))
        model._reload_load_plan = plan
    return plan
