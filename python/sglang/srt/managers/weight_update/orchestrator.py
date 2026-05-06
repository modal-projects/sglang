import asyncio
from typing import Awaitable, Callable, Optional, TypeVar

from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
)

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

        async with self.manager.weight_update_orchestration_lock:
            pause_mode = getattr(obj, "atomic_pause_mode", None)
            if pause_mode is None:
                if prepare_fn is not None:
                    await prepare_fn()
                return await update_fn()

            self._validate_pause_mode(obj, pause_mode)

            async with self.manager.is_pause_cond:
                already_paused = self.manager.is_pause

            if prepare_fn is not None:
                await prepare_fn()

            paused_here = False
            if not already_paused:
                try:
                    await self.manager.pause_generation(
                        PauseGenerationReqInput(mode=pause_mode)
                    )
                except Exception:
                    if discard_prepare_fn is not None:
                        await discard_prepare_fn()
                    raise
                paused_here = True

            try:
                return await update_fn()
            finally:
                if paused_here:
                    await self.manager.continue_generation(ContinueGenerationReqInput())

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
