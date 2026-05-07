import asyncio
import time
from typing import Awaitable, Callable, Optional, TypeVar

from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
)
from sglang.srt.managers.weight_update.tracing import elapsed_ms, ensure_update_trace

T = TypeVar("T")

ATOMIC_PAUSE_MODES = {"abort", "retract", "in_place"}


class WeightUpdateOrchestrator:
    """Runs optional prestage/pause/update/resume lifecycle for weight updates."""

    def __init__(self, tokenizer_manager):
        self.manager = tokenizer_manager

    async def run(
        self,
        obj,
        update_fn: Callable[[], Awaitable[T]],
        *,
        prepare_fn: Optional[Callable[[], Awaitable[None]]] = None,
        discard_prepare_fn: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> T:
        self._ensure_state()
        trace = ensure_update_trace(obj)
        trace["atomic_pause_mode"] = getattr(obj, "atomic_pause_mode", None)

        async with self.manager.weight_update_orchestration_lock:
            pause_mode = getattr(obj, "atomic_pause_mode", None)
            if pause_mode is None:
                if prepare_fn is not None:
                    prepare_started_at = time.monotonic()
                    await prepare_fn()
                    trace["tokenizer_pre_update_prepare_ms"] = elapsed_ms(
                        prepare_started_at
                    )
                update_started_at = time.monotonic()
                try:
                    return await update_fn()
                finally:
                    trace["tokenizer_update_fn_ms"] = elapsed_ms(update_started_at)

            self._validate_pause_mode(obj, pause_mode)

            async with self.manager.is_pause_cond:
                already_paused = self.manager.is_pause
            trace["tokenizer_already_paused_before_update"] = already_paused

            if prepare_fn is not None:
                prepare_started_at = time.monotonic()
                await prepare_fn()
                trace["tokenizer_pre_pause_prepare_ms"] = elapsed_ms(prepare_started_at)

            paused_here = False
            if not already_paused:
                try:
                    pause_trace = {
                        "request_id": trace.get("request_id"),
                        "parent": "weight_update",
                    }
                    pause_started_at = time.monotonic()
                    await self.manager.pause_generation(
                        PauseGenerationReqInput(mode=pause_mode, trace=pause_trace)
                    )
                    trace["tokenizer_pause_generation_ms"] = elapsed_ms(
                        pause_started_at
                    )
                    trace["scheduler_pause_trace"] = pause_trace
                except Exception:
                    if discard_prepare_fn is not None:
                        await discard_prepare_fn()
                    raise
                paused_here = True

            paused_started_at = time.monotonic()
            try:
                update_started_at = time.monotonic()
                return await update_fn()
            finally:
                trace["tokenizer_update_fn_ms"] = elapsed_ms(update_started_at)
                if paused_here:
                    continue_trace = {
                        "request_id": trace.get("request_id"),
                        "parent": "weight_update",
                    }
                    continue_started_at = time.monotonic()
                    await self.manager.continue_generation(
                        ContinueGenerationReqInput(trace=continue_trace)
                    )
                    trace["tokenizer_continue_generation_ms"] = elapsed_ms(
                        continue_started_at
                    )
                    trace["scheduler_continue_trace"] = continue_trace
                    trace["tokenizer_paused_total_ms"] = elapsed_ms(paused_started_at)

    def _ensure_state(self) -> None:
        if not hasattr(self.manager, "weight_update_orchestration_lock"):
            self.manager.weight_update_orchestration_lock = asyncio.Lock()
        if not hasattr(self.manager, "is_pause_cond"):
            self.manager.is_pause_cond = asyncio.Condition()
        if not hasattr(self.manager, "is_pause"):
            self.manager.is_pause = False

    @staticmethod
    def _validate_pause_mode(obj, pause_mode: str) -> None:
        if pause_mode not in ATOMIC_PAUSE_MODES:
            raise ValueError(
                f"Invalid atomic_pause_mode: {pause_mode!r}. "
                f"Expected one of {sorted(ATOMIC_PAUSE_MODES)}."
            )
        if pause_mode == "in_place" and getattr(obj, "flush_cache", False):
            raise ValueError(
                "flush_cache must be false when atomic_pause_mode='in_place'."
            )
