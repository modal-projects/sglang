"""The scheduler's side of weight delta: prepare a version while serving, apply it in the pause."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import torch

from sglang.srt.managers.io_struct import (
    PrepareWeightsFromDeltaReqInput,
    PrepareWeightsFromDeltaReqOutput,
    UpdateWeightsFromDeltaReqInput,
    UpdateWeightsFromDeltaReqOutput,
)
from sglang.srt.runtime_context import get_parallel, get_serving
from sglang.srt.weight_sync.weight_delta.apply import Applier, Plan
from sglang.srt.weight_sync.weight_delta.map import DeltaMap
from sglang.srt.weight_sync.weight_delta.prepare import Prepared, PrepareProcess

logger = logging.getLogger(__name__)


@dataclass
class _Ready:
    manifest: str
    prepared: Prepared
    staged: torch.Tensor
    plan: Plan


class Receiver:
    """One version at a time: a prepare runs in this GPU's prepare process, and the scheduler
    loop replies once it is done; the update then applies it inside the caller's pause."""

    def __init__(
        self, delta_map: DeltaMap, codec: str, device: torch.device, send_reply
    ) -> None:
        self.codec = codec
        self.process = PrepareProcess(delta_map.table, _threads())
        self.applier = Applier(delta_map, codec, device)
        self._send_reply = send_reply
        self._pending: PrepareWeightsFromDeltaReqInput | None = None
        self._ready: _Ready | None = None

    def prepare(
        self, req: PrepareWeightsFromDeltaReqInput
    ) -> PrepareWeightsFromDeltaReqOutput | None:
        if self._pending is not None:
            return PrepareWeightsFromDeltaReqOutput(
                success=False, message="a weight-delta prepare is already running"
            )
        self._ready = None
        self.process.start(req.manifest, get_serving().weight_version)
        self._pending = req
        return None

    def check_pending(self) -> None:
        """Once per scheduler loop: reply to the running prepare if it is done."""
        if self._pending is None:
            return
        req = self._pending
        try:
            result = self.process.poll()
            if result is None:
                return
            prepared, staged = result
            if prepared.codec != self.codec:
                raise ValueError(
                    f"the delta is {prepared.codec}, the engine applies {self.codec}"
                )
            started = time.perf_counter()
            plan = self.applier.plan(prepared)
            stats = {
                **prepared.stats,
                "staging_capacity": prepared.staging_capacity,
                "plan_s": time.perf_counter() - started,
            }
            self._ready = _Ready(req.manifest, prepared, staged, plan)
            output = PrepareWeightsFromDeltaReqOutput(
                success=True, message="Prepared.", stats=stats
            )
            logger.info(
                f"Weight delta {prepared.base_version} -> {prepared.target_version} "
                f"prepared: {stats}"
            )
        except Exception as error:
            output = PrepareWeightsFromDeltaReqOutput(success=False, message=str(error))
        self._pending = None
        self._send_reply(output, req)

    def update(
        self, req: UpdateWeightsFromDeltaReqInput, weight_updater
    ) -> UpdateWeightsFromDeltaReqOutput:
        """Refusals leave the weights and the prepared version as they are; a failure while
        applying raises, with the weights partly updated, and takes the engine down."""
        ready, version = self._ready, get_serving().weight_version
        refusal = None
        if self._pending is not None:
            refusal = "a weight-delta prepare is still running"
        elif ready is None or ready.manifest != req.manifest:
            refusal = f"{req.manifest} is not prepared"
        elif str(ready.prepared.target_version) != req.weight_version:
            refusal = (
                f"{req.manifest} reaches version {ready.prepared.target_version}, "
                f"not {req.weight_version}"
            )
        elif str(ready.prepared.base_version) != version:
            refusal = f"the engine moved to version {version} after the prepare"
        if refusal is not None:
            return UpdateWeightsFromDeltaReqOutput(success=False, message=refusal)
        self._ready = None
        stats = self.applier.apply(ready.plan, ready.staged)
        logger.info(f"Weight delta applied up to version {req.weight_version}: {stats}")
        weight_updater.record_weight_version_after_update(req.weight_version)
        success, message = weight_updater.flush_cache_after_weight_update(
            req, True, "Succeeded to apply the weight delta."
        )
        return UpdateWeightsFromDeltaReqOutput(
            success=success, message=message, stats=stats
        )


def _threads() -> int:
    """This GPU's share of the container's CPUs: its quota if it sets one, else its cores."""
    cpus = len(os.sched_getaffinity(0))
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            cpus = int(quota) // int(period)
    except (OSError, ValueError):
        pass
    return max(1, cpus // get_parallel().tp_size)
