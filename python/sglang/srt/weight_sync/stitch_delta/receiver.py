from __future__ import annotations

import json
import logging
import os
import threading
import time

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    GpuDeltaReqInput,
    GpuDeltaReqOutput,
    GpuDeltaTestReqInput,
    GpuDeltaTestReqOutput,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_lora,
    get_model,
    get_parallel,
    get_serving,
)
from sglang.srt.weight_sync.stitch_delta.map import build_map, read_inventory
from sglang.srt.weight_sync.stitch_delta.memory import HostArena
from sglang.srt.weight_sync.stitch_delta.xor_kernel import warm_up

logger = logging.getLogger(__name__)


def gather_attn_tp(entry: dict) -> list[dict]:
    """Only attention-TP rank 0 replies, so a reply carries its whole group's entries."""
    group = get_parallel().attn_tp_group
    if group.world_size == 1:
        return [entry]
    entries = [None] * group.world_size
    torch.distributed.all_gather_object(entries, entry, group=group.cpu_group)
    return entries


def _cpu_budget() -> int:
    """CPUs this process may use: the container's quota if it sets one, else its cores."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0))


def check_scope(runners) -> None:
    """Refuse, at startup, every configuration GPU delta does not cover."""
    from sglang.srt.model_executor.model_runner_components.weight_updater import (
        _unsupported_derived_weight_cache_error,
    )

    parallel, moe = get_parallel(), get_exec().moe
    target = next(runner for role, runner in runners if role == "target")
    refused = {
        "multi-node engines": parallel.nnodes > 1,
        "data-parallel replicas": parallel.dp_size > 1,
        "pipeline parallelism": parallel.pp_size > 1,
        "context parallelism": parallel.attn_cp_size > 1 or parallel.dcp_size > 1,
        "attention TP above 1 (DP attention only, for now)": parallel.attn_tp_size > 1,
        "expert TP above 1": parallel.moe_tp_size > 1,
        "PD disaggregation": get_disagg().disaggregation_mode != "null",
        "LoRA": bool(get_lora().enable_lora),
        "the memory saver (it moves weights)": get_exec().features.enable_memory_saver,
        "EPLB or redundant experts": moe.enable_eplb
        or moe.ep_num_redundant_experts > 0,
        "DP-attention local control broadcast": (
            parallel.enable_dp_attention_local_control_broadcast
        ),
        "the zero-copy weight cache": get_model().weight_cache_mode != "off",
        "MoE gate/up rows permuted in place (SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED)": (
            envs.SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED.get()
        ),
        "a draft model from another checkpoint": any(
            runner.model_config.model_path != target.model_config.model_path
            for role, runner in runners
            if role != "target"
        ),
    }
    problems = [name for name, hit in refused.items() if hit]
    problems += filter(
        None,
        (_unsupported_derived_weight_cache_error(r.model) for _, r in runners),
    )
    if problems:
        raise ValueError("--enable-gpu-delta does not support " + "; ".join(problems))


class GpuDeltaReceiver:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.runners = list(scheduler.tp_worker.weight_update_runners())
        if scheduler.draft_worker is not None:
            self.runners += scheduler.draft_worker.weight_update_runners()
        target = [runner for role, runner in self.runners if role == "target"]
        if len(target) != 1:
            raise ValueError(
                f"expected one target runner, got {[r for r, _ in self.runners]}"
            )
        started = time.perf_counter()
        check_scope(self.runners)
        self.inventory = read_inventory(target[0])
        read = time.perf_counter()
        if envs.SGLANG_GPU_DELTA_INVENTORY_ONLY.get():
            self.delta_map = None
            logger.warning("GPU delta map not built (inventory only); updates are refused")
        else:
            self.delta_map = build_map(self.runners, self.inventory)
            logger.info(
                f"GPU delta map built for {len(self.inventory)} checkpoint tensors: "
                f"headers {read - started:.2f} s, map {time.perf_counter() - read:.2f} s"
            )
        self.device = next(target[0].model.parameters()).device
        # The container's CPUs, split across its GPUs, undo the outer Zstd.
        self.workers = max(1, _cpu_budget() // get_parallel().tp_size)
        # Compiled now, so the first update does not pay for it.
        compiled = time.perf_counter()
        warm_up(self.device.index)
        logger.info(
            f"GPU delta kernels compiled in {time.perf_counter() - compiled:.2f} s; "
            f"{self.workers} CPU workers for the outer Zstd"
        )
        # Kept across updates: the decompression engine's host memory, the XOR plans, and the
        # stream prepare runs on (PyTorch caches freed GPU memory per stream, so a new stream
        # per prepare would strand what the last one used).
        self.arena = HostArena(self.device.index)
        self._plans = {}
        self._stream = torch.cuda.Stream(self.device)
        self._decoder = None
        self._lock = threading.Lock()
        self._state, self._error, self._fetched = "idle", None, None

    @property
    def decoder(self):
        if self._decoder is None:
            from sglang.srt.weight_sync.stitch_delta.nvcomp import SnappyDecoder

            self._decoder = SnappyDecoder(self.device)
        return self._decoder

    def handle(self, req: GpuDeltaReqInput) -> GpuDeltaReqOutput:
        if self.delta_map is None and req.action in ("fetch", "apply"):
            return GpuDeltaReqOutput(
                success=False, message="inventory-only engine", state="idle"
            )
        if req.action == "fetch":
            return self._start_fetch(req)
        if req.action == "apply":
            return self._apply(req)
        if req.action == "discard":
            with self._lock:
                self._close()
        if req.action == "release":
            with self._lock:
                if self._state == "fetching":
                    return GpuDeltaReqOutput(
                        success=False, message="a prepare is running", state="fetching"
                    )
                self._close()
                self.arena.release()
                self._plans = {}
        return self._status()

    def _status(self) -> GpuDeltaReqOutput:
        with self._lock:
            state, error, fetched = self._state, self._error, self._fetched
        return GpuDeltaReqOutput(
            success=error is None,
            message=error or "",
            state=state,
            target_version=None if fetched is None else str(fetched.target_version),
        )

    def _close(self) -> None:
        self._state, self._error, self._fetched = "idle", None, None

    def _start_fetch(self, req: GpuDeltaReqInput) -> GpuDeltaReqOutput:
        with self._lock:
            if self._state == "fetching":
                return GpuDeltaReqOutput(
                    success=False,
                    message="a fetch is already running",
                    state="fetching",
                )
            self._close()
            self._state = "fetching"
        threading.Thread(
            target=self._fetch,
            args=(req.manifests, get_serving().weight_version, req.verify_sha256),
            name="gpu-delta-fetch",
            daemon=True,
        ).start()
        return self._status()

    def _fetch(self, manifests, version, verify_sha256) -> None:
        from sglang.srt.weight_sync.stitch_delta.fetch import fetch

        try:
            # The engine keeps serving: this thread's GPU work goes on its own stream.
            torch.cuda.set_device(self.device)
            stream = self._stream
            with torch.cuda.stream(stream):
                fetched = fetch(
                    self.delta_map,
                    manifests,
                    version,
                    self.decoder,
                    self.workers,
                    self.arena,
                    self._plans,
                    verify_sha256,
                )
            stream.synchronize()
        except Exception as error:
            with self._lock:
                self._state, self._error = "failed", repr(error)
            return
        logger.info(
            f"GPU deltas prepared up to version {fetched.target_version}: {fetched.stats}"
        )
        with self._lock:
            self._state, self._fetched, self._plans = "fetched", fetched, fetched.plans

    def _apply(self, req: GpuDeltaReqInput) -> GpuDeltaReqOutput:
        from sglang.srt.weight_sync.stitch_delta.apply import apply

        # Refusals leave the weights and the prepared deltas as they are.
        with self._lock:
            state, fetched = self._state, self._fetched
            if state != "fetched":
                return GpuDeltaReqOutput(
                    success=False,
                    message=f"nothing prepared to apply (prepare is {state})",
                    state=state,
                )
            target = str(fetched.target_version)
            if req.weight_version != target:
                return GpuDeltaReqOutput(
                    success=False,
                    message=f"the prepared deltas reach version {target}, "
                    f"not {req.weight_version}",
                    state=state,
                )
            version = get_serving().weight_version
            if str(fetched.base_version) != version:
                self._close()
                return GpuDeltaReqOutput(
                    success=False,
                    message=f"the engine moved to version {version} after prepare",
                    state="idle",
                )
            self._state, self._fetched = "idle", None
        # From here on a failure propagates and takes the engine down.
        timings = apply(fetched, self.delta_map, self.device)
        del fetched
        logger.info(f"GPU deltas applied up to version {target}: {timings}")
        updater = self.scheduler.weight_updater
        updater.record_weight_version_after_update(target)
        success, message = updater.flush_cache_after_weight_update(
            req, True, "Succeeded to apply GPU deltas."
        )
        return GpuDeltaReqOutput(
            success=success, message=message, state="idle", target_version=target
        )

    def test(self, req: GpuDeltaTestReqInput) -> GpuDeltaTestReqOutput:
        from sglang.srt.weight_sync.stitch_delta import testing

        tp_rank = get_parallel().tp_rank
        # Every GPU must reach the gather, so a failure becomes an entry.
        try:
            if req.action == "dump":
                result = testing.dump(self, include_inventory=tp_rank == 0)
            elif req.action == "check_map":
                result = testing.check_map(self)
            elif req.action == "check_fetch":
                result = testing.check_fetch(self, req.manifests, req.version)
            elif req.action == "apply":
                result = testing.check_apply(self, req.manifests, req.version)
            elif req.action == "check_version":
                result = testing.check_version(self, req.digests, req.manifests)
            else:
                result = testing.state_hashes(self)
            entry = {"tp_rank": tp_rank, **result}
        except Exception as error:
            entry = {"tp_rank": tp_rank, "error": repr(error)}
        entries = gather_attn_tp(entry)
        errors = [
            f"tp_rank {e['tp_rank']}: {e['error']}" for e in entries if "error" in e
        ]
        return GpuDeltaTestReqOutput(
            success=not errors, message=" | ".join(errors), payload=json.dumps(entries)
        )
