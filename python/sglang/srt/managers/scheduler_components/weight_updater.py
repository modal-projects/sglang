from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import msgspec
import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    ChecksumInfo,
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    StageWeightUpdateReqInput,
    StageWeightUpdateReqOutput,
    UpdateWeightFromCPUReqInput,
    UpdateWeightFromCPUReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)

logger = logging.getLogger(__name__)


def _get_draft_model_runner(draft_worker):
    # DFlash / FrozenKVMTP workers expose draft_model_runner directly
    runner = getattr(draft_worker, "draft_model_runner", None)
    if runner is not None:
        return runner
    # EAGLEWorkerV2: _draft_worker.draft_runner
    inner = getattr(draft_worker, "_draft_worker", None)
    if inner is not None:
        runner = getattr(inner, "draft_runner", None)
        if runner is not None:
            return runner
    return None


def _merge_checksum_payloads(target: Dict, draft: Dict) -> Dict:
    merged_checksums = dict(target["checksums"])
    for name, chk in draft["checksums"].items():
        merged_checksums[f"draft.{name}"] = chk
    h = hashlib.sha256()
    for name in sorted(merged_checksums):
        h.update(name.encode())
        h.update(merged_checksums[name].encode())
    target["checksums"] = merged_checksums
    target["per_gpu_checksum"] = h.hexdigest()
    return target


@dataclass(kw_only=True, slots=True)
class SchedulerWeightUpdaterManager:
    tp_worker: Any
    draft_worker: Any
    tp_cpu_group: Any
    weight_update_stage_cpu_group: Any
    host_cpu_group: Any
    boot_model_path: str
    memory_saver_adapter: Any
    flush_cache: Callable[..., bool]
    is_fully_idle: Callable[..., bool]
    send_control_output: Callable[[Any, Any], None]
    scheduler: Optional[Any] = None
    metrics_collector: Optional[Any] = None
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None
    _pending_weight_update_stage: Optional[
        Tuple[StageWeightUpdateReqInput, threading.Thread]
    ] = field(default=None, init=False)
    _weight_update_stage_result: Optional[StageWeightUpdateReqOutput] = field(
        default=None,
        init=False,
    )
    _cpu_weight_cache_checkpoint_dir: Optional[str] = field(
        default=None,
        init=False,
    )
    _cpu_weight_cache_base_version: Optional[int] = field(
        default=None,
        init=False,
    )
    _cpu_weight_cache_initialization_stats: Optional[Dict[str, Any]] = field(
        default=None,
        init=False,
    )
    _cpu_weight_cache_initialization_error: Optional[str] = field(
        default=None,
        init=False,
    )

    @contextmanager
    def _observe_weight_load(self, source: str) -> Iterator[None]:
        # Edge-trigger weight_load_duration_seconds at the end of each
        # update_weights_from_* call. Engine is paused during the update so
        # the periodic log_stats path can't carry this.
        # `source` distinguishes disk, CPU, distributed, tensor, and IPC loads.
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.metrics_collector is not None:
                self.metrics_collector.observe_weight_load(
                    time.perf_counter() - t0, source
                )

    def flush_cache_after_weight_update(self, recv_req) -> None:
        if recv_req.flush_cache:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            assert flush_cache_success, "Cache flush failed after updating weights"

    @staticmethod
    def _all_gather(value: Any, group: Any) -> list[Any]:
        if not torch.distributed.is_initialized():
            return [value]
        world_size = torch.distributed.get_world_size(group=group)
        if world_size == 1:
            return [value]
        values = [None] * world_size
        torch.distributed.all_gather_object(values, value, group=group)
        return values

    def _initialize_cpu_weight_cache(
        self, checkpoint_dir: str, base_version: int
    ) -> Dict[str, Any]:
        checkpoint_dir = os.path.realpath(checkpoint_dir)
        started = time.perf_counter()
        if self._cpu_weight_cache_initialization_stats is not None:
            if (
                self._cpu_weight_cache_checkpoint_dir != checkpoint_dir
                or self._cpu_weight_cache_base_version != base_version
            ):
                raise RuntimeError(
                    "CPU weight cache was initialized from a different base: "
                    f"({self._cpu_weight_cache_checkpoint_dir!r}, "
                    f"v{self._cpu_weight_cache_base_version}) != "
                    f"({checkpoint_dir!r}, v{base_version})"
                )
            return {
                "operation": "initialize_cpu_weight_cache",
                "checkpoint_dir": checkpoint_dir,
                "initialized": False,
                "wall_s": round(time.perf_counter() - started, 6),
            }

        if self._cpu_weight_cache_initialization_error is not None:
            logger.warning("Retrying CPU weight cache initialization")
            self._cpu_weight_cache_initialization_error = None
        self._cpu_weight_cache_checkpoint_dir = checkpoint_dir
        self._cpu_weight_cache_base_version = base_version
        server_args = self.tp_worker.model_runner.server_args
        try:
            stats = self.tp_worker.initialize_cpu_weight_cache(
                checkpoint_dir=checkpoint_dir,
                seed_from_active_weights=self._is_boot_checkpoint(checkpoint_dir),
                base_version=base_version,
                host_group=self.host_cpu_group,
                max_compile_group_bytes=int(
                    server_args.cpu_weight_cache_max_compile_group_gb * (1 << 30)
                ),
                canonical_checkpoint_dir=(
                    server_args.cpu_weight_cache_canonical_checkpoint_dir
                ),
            )
        except Exception:
            self._cpu_weight_cache_initialization_error = traceback.format_exc()
            logger.exception("CPU weight cache initialization failed")
            raise RuntimeError(
                "CPU weight cache initialization failed:\n"
                + self._cpu_weight_cache_initialization_error
            ) from None
        self._cpu_weight_cache_initialization_stats = stats
        return {
            "operation": "initialize_cpu_weight_cache",
            "checkpoint_dir": checkpoint_dir,
            "initialized": True,
            "initialization": stats,
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def _stage_cpu_weight_checkpoint(
        self, checkpoint_dir: str, target_version: int
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        success, message, stats = (
            self.tp_worker.stage_cpu_weight_update_from_checkpoint(
                checkpoint_dir=checkpoint_dir,
                target_version=target_version,
                host_group=self.host_cpu_group,
            )
        )
        if not success or stats is None:
            raise RuntimeError(message)
        return {
            "operation": "stage_cpu_weight_checkpoint",
            "checkpoint_dir": checkpoint_dir,
            "checkpoint": stats,
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def _is_boot_checkpoint(self, checkpoint_dir: str) -> bool:
        return os.path.realpath(checkpoint_dir) == os.path.realpath(
            self.boot_model_path
        )

    def _pending_weight_update_stage_message(self) -> str | None:
        pending = self._pending_weight_update_stage is not None
        if not any(self._all_gather(pending, self.tp_cpu_group)):
            return None
        return (
            "A background weight update stage is running; live weights cannot "
            "change until it finishes."
        )

    def _cpu_weight_cache_conflict(self, operation: str) -> str | None:
        if not self.tp_worker.model_runner.server_args.enable_cpu_weight_cache:
            return None
        return (
            f"{operation} is unavailable while the CPU weight cache is enabled. "
            "Stage and commit CPU weight updates through their dedicated APIs, "
            "or launch without --enable-cpu-weight-cache."
        )

    def _stage_weight_update_preflight_message(
        self,
        recv_req: StageWeightUpdateReqInput,
    ) -> str | None:
        server_args = self.tp_worker.model_runner.server_args
        message = None
        if recv_req.target_version < 0:
            message = "target_version must be non-negative."
        elif recv_req.base_version < 0:
            message = "base_version must be non-negative."
        elif recv_req.target_version < recv_req.base_version:
            message = "target_version must not precede base_version."
        elif (
            recv_req.target_version > recv_req.base_version
            and recv_req.checkpoint_source_dir is None
        ):
            message = "checkpoint_source_dir is required after the base version."
        elif recv_req.destination == "disk":
            if recv_req.local_checkpoint_dir is None:
                message = "local_checkpoint_dir is required for disk staging."
            elif server_args.enable_cpu_weight_cache:
                message = (
                    "Disk staging is unavailable while the CPU weight cache is "
                    "enabled."
                )
        elif recv_req.destination == "cpu":
            if not server_args.enable_cpu_weight_cache:
                message = "CPU staging requires --enable-cpu-weight-cache."
            elif GPU_MEMORY_TYPE_WEIGHTS in self.offload_tags:
                message = "CPU staging requires live GPU weights to be resident."
        else:
            message = (
                f"unsupported weight staging destination: {recv_req.destination!r}"
            )

        messages = sorted(
            {
                value
                for value in self._all_gather(message, self.tp_cpu_group)
                if value is not None
            }
        )
        return "; ".join(messages) if messages else None

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        if message := self._pending_weight_update_stage_message():
            return UpdateWeightFromDiskReqOutput(success=False, message=message)
        if message := self._cpu_weight_cache_conflict("update_weights_from_disk"):
            return UpdateWeightFromDiskReqOutput(success=False, message=message)
        with self._observe_weight_load("disk"):
            success, message = self.tp_worker.update_weights_from_disk(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_disk(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            return UpdateWeightFromDiskReqOutput(
                success=success, message=message, num_paused_requests=0
            )

    def stage_weight_update(self, recv_req: StageWeightUpdateReqInput):
        """Start staging a verified target while inference continues."""

        if self._pending_weight_update_stage_message() is not None:
            return StageWeightUpdateReqOutput(
                success=False,
                message="Another weight update stage is already running.",
            )
        if message := self._stage_weight_update_preflight_message(recv_req):
            return StageWeightUpdateReqOutput(success=False, message=message)

        def stage() -> None:
            try:
                output = self._stage_weight_update_sync(recv_req)
            except Exception:
                logger.exception("Background weight update staging failed")
                output = StageWeightUpdateReqOutput(
                    success=False,
                    message=traceback.format_exc(),
                )
            self._weight_update_stage_result = output

        thread = threading.Thread(
            target=stage,
            name="weight-update-stage",
            daemon=True,
        )
        self._pending_weight_update_stage = (recv_req, thread)
        thread.start()
        return None

    def _refresh_checkpoint_source(
        self,
        checkpoint_source_dir: str,
        target_version: int,
    ) -> float:
        from sglang.srt.weight_sync.disk_checkpoint import refresh_checkpoint_source

        started = time.perf_counter()
        host_rank = (
            torch.distributed.get_rank(group=self.host_cpu_group)
            if torch.distributed.is_initialized()
            else 0
        )
        local_error = None
        if host_rank == 0:
            try:
                refresh_checkpoint_source(
                    checkpoint_source_dir,
                    target_version,
                    self.tp_worker.model_runner.server_args.checkpoint_source_refresh_hook,
                )
            except Exception:
                local_error = traceback.format_exc()
        errors = [
            error
            for error in self._all_gather(local_error, self.host_cpu_group)
            if error is not None
        ]
        if errors:
            raise RuntimeError("checkpoint source refresh failed: " + "; ".join(errors))
        return time.perf_counter() - started

    def _stage_weight_update_sync(
        self,
        recv_req: StageWeightUpdateReqInput,
    ) -> StageWeightUpdateReqOutput:
        """Stage one target on disk or in a rank-ready CPU image."""

        from sglang.srt.weight_sync import disk_checkpoint

        started = time.perf_counter()
        rank = (
            torch.distributed.get_rank(group=self.weight_update_stage_cpu_group)
            if torch.distributed.is_initialized()
            else 0
        )
        local_stats: Dict[str, Any] = {
            "rank": rank,
            "base_version": recv_req.base_version,
            "target_version": recv_req.target_version,
            "destination": recv_req.destination,
        }
        cpu_cache_was_initialized = (
            self._cpu_weight_cache_initialization_stats is not None
        )
        try:
            checkpoint_dir = recv_req.base_checkpoint_dir or self.boot_model_path
            if recv_req.destination == "cpu":
                if recv_req.target_version == recv_req.base_version:
                    if self._cpu_weight_cache_initialization_stats is None:
                        local_stats["stage"] = self._initialize_cpu_weight_cache(
                            checkpoint_dir,
                            recv_req.base_version,
                        )
                    elif (
                        self._cpu_weight_cache_checkpoint_dir
                        == os.path.realpath(checkpoint_dir)
                        and self._cpu_weight_cache_base_version == recv_req.base_version
                    ):
                        local_stats["stage"] = self._initialize_cpu_weight_cache(
                            checkpoint_dir,
                            recv_req.base_version,
                        )
                    else:
                        local_stats["stage"] = self._stage_cpu_weight_checkpoint(
                            checkpoint_dir,
                            recv_req.target_version,
                        )
                else:
                    if self._cpu_weight_cache_initialization_error is not None:
                        raise RuntimeError(
                            "CPU weight cache initialization failed:\n"
                            + self._cpu_weight_cache_initialization_error
                        )
                    if self._cpu_weight_cache_initialization_stats is None:
                        raise RuntimeError(
                            "CPU weight cache is not initialized. Stage the base "
                            "version to CPU before staging a delta target."
                        )
                    if (
                        self._cpu_weight_cache_checkpoint_dir
                        != os.path.realpath(checkpoint_dir)
                        or self._cpu_weight_cache_base_version != recv_req.base_version
                    ):
                        raise RuntimeError(
                            "CPU weight cache base does not match the initialized "
                            f"cache: ({checkpoint_dir!r}, v{recv_req.base_version}) "
                            f"!= ({self._cpu_weight_cache_checkpoint_dir!r}, "
                            f"v{self._cpu_weight_cache_base_version})"
                        )
                    checkpoint_source_dir = recv_req.checkpoint_source_dir
                    if checkpoint_source_dir is None:
                        raise ValueError(
                            "checkpoint_source_dir is required after the base version"
                        )
                    source_refresh_wall_s = self._refresh_checkpoint_source(
                        checkpoint_source_dir,
                        recv_req.target_version,
                    )
                    success, message, stage_stats = (
                        self.tp_worker.stage_cpu_weight_update_from_delta_lineage(
                            checkpoint_source_dir=checkpoint_source_dir,
                            target_version=recv_req.target_version,
                            host_group=self.host_cpu_group,
                        )
                    )
                    if not success or stage_stats is None:
                        raise RuntimeError(message)
                    stage_stats["source_refresh_wall_s"] = round(
                        source_refresh_wall_s,
                        6,
                    )
                    local_stats["stage"] = stage_stats
            else:
                local_checkpoint_dir = recv_req.local_checkpoint_dir
                if local_checkpoint_dir is None:
                    raise ValueError(
                        "local_checkpoint_dir is required for disk staging"
                    )
                local_stats["stage"] = disk_checkpoint.materialize(
                    local_checkpoint_dir=local_checkpoint_dir,
                    base_checkpoint_dir=checkpoint_dir,
                    checkpoint_source_dir=(
                        recv_req.checkpoint_source_dir or checkpoint_dir
                    ),
                    target_version=recv_req.target_version,
                    base_version=recv_req.base_version,
                    checkpoint_source_refresh_hook=(
                        self.tp_worker.model_runner.server_args.checkpoint_source_refresh_hook
                    ),
                )
            success, message = True, "Success."
        except Exception:
            success, message = False, traceback.format_exc()
            logger.error(message)
        local_stats["wall_s"] = round(time.perf_counter() - started, 6)

        results = self._all_gather(
            (success, message, local_stats),
            self.weight_update_stage_cpu_group,
        )
        success = all(result[0] for result in results)
        if not success:
            messages = list(
                dict.fromkeys(result[1] for result in results if not result[0])
            )
            message = "\n".join(messages)
        if (
            recv_req.destination == "cpu"
            and recv_req.target_version == recv_req.base_version
            and success
        ):
            self._cpu_weight_cache_checkpoint_dir = os.path.realpath(checkpoint_dir)
            self._cpu_weight_cache_base_version = recv_req.base_version
        if recv_req.destination == "cpu" and not success:
            if recv_req.target_version == recv_req.base_version:
                if not cpu_cache_was_initialized:
                    self.tp_worker.discard_cpu_weight_cache(
                        "distributed CPU weight cache initialization failed"
                    )
                    self._cpu_weight_cache_checkpoint_dir = None
                    self._cpu_weight_cache_base_version = None
                    self._cpu_weight_cache_initialization_stats = None
                    self._cpu_weight_cache_initialization_error = message
                else:
                    self.tp_worker.invalidate_staged_cpu_weight_update(
                        "distributed CPU checkpoint staging failed"
                    )
            else:
                self.tp_worker.invalidate_staged_cpu_weight_update(
                    "distributed CPU weight update staging failed"
                )
        return StageWeightUpdateReqOutput(
            success=success,
            message=message,
            # The tokenizer manager already fans out to every scheduler. The
            # all-gather above is only the fail-closed synchronization point;
            # returning it from every scheduler would duplicate rank stats.
            rank_stats=[local_stats],
        )

    def check_pending_weight_update_stage(self) -> None:
        pending = self._pending_weight_update_stage
        if pending is None:
            return
        recv_req, thread = pending
        if thread.is_alive():
            return
        self._pending_weight_update_stage = None
        thread.join()
        output = self._weight_update_stage_result
        self._weight_update_stage_result = None
        if output is None:
            output = StageWeightUpdateReqOutput(
                success=False,
                message="Weight update staging ended without a result.",
            )
        self.send_control_output(recv_req, output)

    def update_weights_from_cpu(
        self,
        recv_req: UpdateWeightFromCPUReqInput,
    ) -> UpdateWeightFromCPUReqOutput:
        """Commit a complete target image to the target model."""

        with self._observe_weight_load("cpu"):
            if message := self._pending_weight_update_stage_message():
                return UpdateWeightFromCPUReqOutput(success=False, message=message)
            if self._cpu_weight_cache_initialization_error is not None:
                preflight = (False, "CPU weight cache initialization failed.")
            elif self._cpu_weight_cache_initialization_stats is None:
                preflight = (False, "CPU weight cache is not initialized.")
            else:
                preflight = self.tp_worker.validate_staged_cpu_weight_update(
                    recv_req.target_version
                )
            preflight_results = self._all_gather(preflight, self.tp_cpu_group)
            if not all(result[0] for result in preflight_results):
                return UpdateWeightFromCPUReqOutput(
                    success=False,
                    message="; ".join(
                        result[1] for result in preflight_results if not result[0]
                    ),
                )

            if recv_req.flush_cache:
                cache_flushed = self.flush_cache(empty_cache=recv_req.torch_empty_cache)
                if not all(self._all_gather(cache_flushed, self.tp_cpu_group)):
                    return UpdateWeightFromCPUReqOutput(
                        success=False,
                        message="Cache flush failed before CPU weight commit.",
                    )

            try:
                success, message, stats = self.tp_worker.update_weights_from_cpu(
                    recv_req.target_version
                )
            except Exception:
                success, message, stats = False, traceback.format_exc(), None

            rank = (
                torch.distributed.get_rank(group=self.tp_cpu_group)
                if torch.distributed.is_initialized()
                else 0
            )
            rank_results = self._all_gather(
                (success, message, {"rank": rank, **(stats or {})}),
                self.tp_cpu_group,
            )
            success = all(result[0] for result in rank_results)
            if not success:
                message = "; ".join(
                    result[1] for result in rank_results if not result[0]
                )
                logger.critical(
                    "CPU weight commit failed after distributed preflight. "
                    "The engine may contain partially committed weights: %s",
                    message,
                )
                raise RuntimeError(
                    "CPU weight commit failed; terminating the engine to avoid "
                    "serving a mixed model. " + message
                )
            return UpdateWeightFromCPUReqOutput(
                success=True,
                message=message,
                # The tokenizer manager collects one response per scheduler.
                # Keep the all-gather local to the commit protocol so response
                # metadata remains linear in the number of workers.
                rank_stats=[{"rank": rank, **(stats or {})}],
            )

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success=success, message=message)

    def destroy_weights_update_group(
        self,
        recv_req: DestroyWeightsUpdateGroupReqInput,
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success=success, message=message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        if message := self._pending_weight_update_stage_message():
            return UpdateWeightsFromDistributedReqOutput(
                success=False,
                message=message,
            )
        if message := self._cpu_weight_cache_conflict(
            "update_weights_from_distributed"
        ):
            return UpdateWeightsFromDistributedReqOutput(
                success=False,
                message=message,
            )
        with self._observe_weight_load("distributed"):
            success, message = self.tp_worker.update_weights_from_distributed(recv_req)
            if success:
                self.flush_cache_after_weight_update(recv_req)
            else:
                logger.error(message)
            return UpdateWeightsFromDistributedReqOutput(
                success=success, message=message
            )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors."""
        if message := self._pending_weight_update_stage_message():
            return UpdateWeightsFromTensorReqOutput(success=False, message=message)
        if message := self._cpu_weight_cache_conflict("update_weights_from_tensor"):
            return UpdateWeightsFromTensorReqOutput(success=False, message=message)
        with self._observe_weight_load("tensor"):
            if recv_req.disable_draft_model:
                worker = self.tp_worker
            else:
                worker = self.draft_worker or self.tp_worker
            success, message = worker.update_weights_from_tensor(recv_req)
            if success:
                self.flush_cache_after_weight_update(recv_req)
            else:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromTensorReqOutput(success=success, message=message)

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        if message := self._pending_weight_update_stage_message():
            return UpdateWeightsFromIPCReqOutput(success=False, message=message)
        if message := self._cpu_weight_cache_conflict("update_weights_from_ipc"):
            return UpdateWeightsFromIPCReqOutput(success=False, message=message)
        with self._observe_weight_load("ipc"):
            success, message = self.tp_worker.update_weights_from_ipc(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_ipc(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromIPCReqOutput(success=success, message=message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter=parameter)

    def _assert_weight_cache_inactive(self, op: str) -> None:
        """Reject freeing/restoring model weights while the CUDA IPC weight
        cache is active: the weights are shared with the daemon via CUDA IPC, so
        freeing them would leave the daemon and every peer pointing at released
        memory.
        """
        mode = self.tp_worker.model_runner.server_args.weight_cache_mode
        if mode != "off":
            raise RuntimeError(
                f"[weight_cache] {op} of model weights is not supported while the "
                f"weight cache is active (--weight-cache-mode {mode}): the weights "
                f"are shared with the daemon via CUDA IPC, so freeing them would "
                f"corrupt the daemon's master copy and every co-attached engine. "
                f"Restart with --weight-cache-mode off to use this operation."
            )

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        if message := self._pending_weight_update_stage_message():
            raise RuntimeError(message)
        assert (
            self.is_fully_idle()
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES
        if GPU_MEMORY_TYPE_WEIGHTS in tags and (
            message := self._cpu_weight_cache_conflict("releasing model-weight memory")
        ):
            raise RuntimeError(message)

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            scheduler = self.scheduler
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.release_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.release_memory_occupation()
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self._assert_weight_cache_inactive("release_memory_occupation")
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        if message := self._pending_weight_update_stage_message():
            raise RuntimeError(message)
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES
        if GPU_MEMORY_TYPE_WEIGHTS in tags and (
            message := self._cpu_weight_cache_conflict("resuming model-weight memory")
        ):
            raise RuntimeError(message)

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self._assert_weight_cache_inactive("resume_memory_occupation")
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
            scheduler = self.scheduler
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.resume_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.resume_memory_occupation()

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self, recv_req: CheckWeightsReqInput):
        try:
            payload = self.tp_worker.model_runner.check_weights(
                action=recv_req.action, allow_quant_error=recv_req.allow_quant_error
            )

            if self.draft_worker is not None:
                draft_runner = _get_draft_model_runner(self.draft_worker)
                if draft_runner is not None:
                    draft_payload = draft_runner.check_weights(
                        action=recv_req.action,
                        allow_quant_error=recv_req.allow_quant_error,
                    )
                    if payload is not None and draft_payload is not None:
                        payload = _merge_checksum_payloads(payload, draft_payload)

            tp_size = torch.distributed.get_world_size(group=self.tp_cpu_group)
            if tp_size > 1 and payload is not None:
                all_payloads = [None] * tp_size
                torch.distributed.all_gather_object(
                    all_payloads, payload, group=self.tp_cpu_group
                )
                payload = all_payloads
            if payload is not None:
                # Normalize to one ChecksumInfo per rank so the wire shape is a
                # uniform List[ChecksumInfo] (tp==1 becomes a single-element list).
                per_rank = payload if isinstance(payload, list) else [payload]
                payload = [msgspec.convert(p, ChecksumInfo) for p in per_rank]
            return CheckWeightsReqOutput(
                success=True, message="Success.", payload=payload
            )
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self, params):
        url = params["url"]

        self.tp_worker.model_runner.weight_exporter.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.weight_exporter.save_remote_model(draft_url)

    def save_sharded_model(self, params):
        self.tp_worker.model_runner.weight_exporter.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    with torch.inference_mode():
        self_named_buffers = dict(model.named_buffers())
        for name, tensor in static_params["buffers"]:
            self_named_buffers[name][...] = tensor
