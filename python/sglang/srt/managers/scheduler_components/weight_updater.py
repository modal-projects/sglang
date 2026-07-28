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
    scheduler: Optional[Any] = None
    metrics_collector: Optional[Any] = None
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None
    _pending_weight_update_stage: Optional[
        Tuple[StageWeightUpdateReqInput, threading.Thread]
    ] = field(
        default=None,
        init=False,
    )
    _weight_update_stage_result: Optional[StageWeightUpdateReqOutput] = field(
        default=None,
        init=False,
    )
    _cpu_weight_cache_base_checkpoint_dir: Optional[str] = field(
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

    def _initialize_cpu_weight_cache(
        self,
        base_checkpoint_dir: str,
    ) -> Dict[str, Any]:
        if not self.tp_worker.model_runner.server_args.enable_cpu_weight_cache:
            raise RuntimeError(
                "CPU weight cache initialization requires --enable-cpu-weight-cache."
            )

        base_checkpoint_dir = os.path.realpath(base_checkpoint_dir)
        started = time.perf_counter()
        if self._cpu_weight_cache_initialization_stats is not None:
            if self._cpu_weight_cache_base_checkpoint_dir != base_checkpoint_dir:
                raise RuntimeError(
                    "CPU weight cache was initialized from a different checkpoint: "
                    f"{self._cpu_weight_cache_base_checkpoint_dir!r} != "
                    f"{base_checkpoint_dir!r}."
                )
            return {
                "operation": "initialize_cpu_weight_cache",
                "base_checkpoint_dir": base_checkpoint_dir,
                "initialized": False,
                "wall_s": round(time.perf_counter() - started, 6),
            }

        if self._cpu_weight_cache_initialization_error is not None:
            logger.warning("Retrying failed CPU weight cache initialization")
            self._cpu_weight_cache_initialization_error = None
            self._cpu_weight_cache_base_checkpoint_dir = None
        self._cpu_weight_cache_base_checkpoint_dir = base_checkpoint_dir
        try:
            stats = self.tp_worker.initialize_cpu_weight_cache(
                self.host_cpu_group,
                base_checkpoint_dir=base_checkpoint_dir,
            )
        except Exception:
            self._cpu_weight_cache_initialization_error = traceback.format_exc()
            logger.exception("CPU weight cache initialization failed")
            raise RuntimeError(
                "CPU weight cache initialization failed:\n"
                + self._cpu_weight_cache_initialization_error
            )
        self._cpu_weight_cache_initialization_stats = stats
        return {
            "operation": "initialize_cpu_weight_cache",
            "base_checkpoint_dir": base_checkpoint_dir,
            "initialized": True,
            "initialization": stats,
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def _pending_weight_update_stage_message(self) -> str | None:
        stage_pending = self._pending_weight_update_stage is not None
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size(group=self.tp_cpu_group)
            pending_by_rank = [None] * world_size
            torch.distributed.all_gather_object(
                pending_by_rank,
                stage_pending,
                group=self.tp_cpu_group,
            )
        else:
            pending_by_rank = [stage_pending]
        if not any(pending_by_rank):
            return None
        return (
            "A background weight update stage is running; "
            "live weights cannot be changed until it finishes."
        )

    def _cpu_weight_cache_conflict(self, operation: str) -> str | None:
        if not self.tp_worker.model_runner.server_args.enable_cpu_weight_cache:
            return None
        return (
            f"{operation} is unavailable while the CPU weight cache is enabled. "
            "Use /stage_weight_update with destination=cpu followed by "
            "/update_weights_from_cpu, or launch without "
            "--enable-cpu-weight-cache."
        )

    def _stage_weight_update_preflight_message(
        self, recv_req: StageWeightUpdateReqInput
    ) -> str | None:
        message = None
        cpu_weight_cache_enabled = (
            self.tp_worker.model_runner.server_args.enable_cpu_weight_cache
        )
        if recv_req.target_version < 0:
            message = "target_version must be non-negative."
        elif recv_req.target_version > 0 and recv_req.checkpoint_source_dir is None:
            message = "checkpoint_source_dir is required for targets after version 0."
        elif recv_req.destination == "disk" and cpu_weight_cache_enabled:
            message = (
                "Disk weight update staging is unavailable while the CPU weight "
                "cache is enabled. Launch without --enable-cpu-weight-cache to "
                "use disk staging and /update_weights_from_disk."
            )
        elif recv_req.destination == "cpu":
            if not cpu_weight_cache_enabled:
                message = (
                    "CPU weight update staging requires --enable-cpu-weight-cache."
                )
            elif GPU_MEMORY_TYPE_WEIGHTS in self.offload_tags:
                message = (
                    "CPU weight update staging requires the live model weights "
                    "to be resident on the GPU."
                )

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size(group=self.tp_cpu_group)
            messages = [None] * world_size
            torch.distributed.all_gather_object(
                messages,
                message,
                group=self.tp_cpu_group,
            )
        else:
            messages = [message]
        messages = sorted({value for value in messages if value is not None})
        return "; ".join(messages) if messages else None

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        if message := self._pending_weight_update_stage_message():
            return UpdateWeightFromDiskReqOutput(success=False, message=message)
        if message := self._cpu_weight_cache_conflict("/update_weights_from_disk"):
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
        """Stage a weight update while inference continues."""

        if self._pending_weight_update_stage_message() is not None:
            return StageWeightUpdateReqOutput(
                success=False,
                message="Another weight update stage is already running.",
            )
        if message := self._stage_weight_update_preflight_message(recv_req):
            return StageWeightUpdateReqOutput(
                success=False,
                message=message,
            )

        def stage():
            self._weight_update_stage_result = self._stage_weight_update_sync(recv_req)

        thread = threading.Thread(
            target=stage,
            name="weight-update-stage",
            daemon=True,
        )
        self._pending_weight_update_stage = (recv_req, thread)
        thread.start()
        return None

    def _stage_weight_update_sync(self, recv_req: StageWeightUpdateReqInput):
        """Stage one verified target in disk or rank-ready CPU memory."""
        from sglang.srt.weight_sync import disk_checkpoint

        server_args = self.tp_worker.model_runner.server_args
        started = time.perf_counter()
        local_stats: Dict[str, Any] = {
            "rank": (
                torch.distributed.get_rank(group=self.tp_cpu_group)
                if torch.distributed.is_initialized()
                else 0
            ),
            "target_version": recv_req.target_version,
            "destination": recv_req.destination,
        }
        try:
            base_checkpoint_dir = recv_req.base_checkpoint_dir or self.boot_model_path
            if recv_req.destination == "cpu":
                canonical_checkpoint_dir = (
                    server_args.cpu_weight_cache_canonical_checkpoint_dir
                )
                if recv_req.target_version == 0:
                    if canonical_checkpoint_dir is not None:
                        local_stats["canonical_checkpoint_materialization"] = (
                            disk_checkpoint.materialize(
                                local_checkpoint_dir=canonical_checkpoint_dir,
                                base_checkpoint_dir=base_checkpoint_dir,
                                checkpoint_source_dir=base_checkpoint_dir,
                                target_version=0,
                            )
                        )
                    local_stats["stage"] = self._initialize_cpu_weight_cache(
                        base_checkpoint_dir
                    )
                else:
                    if self._cpu_weight_cache_initialization_error is not None:
                        raise RuntimeError(
                            "CPU weight cache initialization failed:\n"
                            + self._cpu_weight_cache_initialization_error
                        )
                    if self._cpu_weight_cache_initialization_stats is None:
                        raise RuntimeError(
                            "CPU weight cache is not initialized. Stage version 0 "
                            "to the CPU destination before staging a delta target."
                        )
                    if self._cpu_weight_cache_base_checkpoint_dir != os.path.realpath(
                        base_checkpoint_dir
                    ):
                        raise RuntimeError(
                            "CPU weight cache base checkpoint does not match "
                            "the initialized cache: "
                            f"{base_checkpoint_dir!r} != "
                            f"{self._cpu_weight_cache_base_checkpoint_dir!r}."
                        )
                    assert recv_req.checkpoint_source_dir is not None
                    if canonical_checkpoint_dir is not None:
                        from sglang.srt.weight_sync.cpu_delta_checkpoint import (
                            validate_delta_target,
                        )

                        validate_delta_target(
                            checkpoint_source_dir=recv_req.checkpoint_source_dir,
                            target_version=recv_req.target_version,
                        )
                        materialization = disk_checkpoint.materialize(
                            local_checkpoint_dir=canonical_checkpoint_dir,
                            base_checkpoint_dir=base_checkpoint_dir,
                            checkpoint_source_dir=recv_req.checkpoint_source_dir,
                            target_version=recv_req.target_version,
                            checkpoint_source_refresh_hook=(
                                server_args.checkpoint_source_refresh_hook
                            ),
                        )
                        success, message, stage_stats = (
                            self.tp_worker.stage_cpu_weight_update_from_checkpoint(
                                checkpoint_dir=canonical_checkpoint_dir,
                                target_version=recv_req.target_version,
                                host_cpu_group=self.host_cpu_group,
                            )
                        )
                        if not success:
                            raise RuntimeError(message)
                        stage_stats["canonical_checkpoint_materialization"] = (
                            materialization
                        )
                    else:
                        refresh_started = time.perf_counter()
                        refresh_error = None
                        host_rank = (
                            torch.distributed.get_rank(group=self.host_cpu_group)
                            if torch.distributed.is_initialized()
                            else 0
                        )
                        if (
                            host_rank == 0
                            and server_args.checkpoint_source_refresh_hook
                        ):
                            try:
                                disk_checkpoint.refresh_checkpoint_source(
                                    recv_req.checkpoint_source_dir,
                                    recv_req.target_version,
                                    server_args.checkpoint_source_refresh_hook,
                                )
                            except Exception:
                                refresh_error = traceback.format_exc()
                        if torch.distributed.is_initialized():
                            refresh_errors = [None] * torch.distributed.get_world_size(
                                group=self.host_cpu_group
                            )
                            torch.distributed.all_gather_object(
                                refresh_errors,
                                refresh_error,
                                group=self.host_cpu_group,
                            )
                        else:
                            refresh_errors = [refresh_error]
                        refresh_errors = [
                            value for value in refresh_errors if value is not None
                        ]
                        if refresh_errors:
                            raise RuntimeError(
                                "checkpoint source refresh failed: "
                                + "; ".join(refresh_errors)
                            )
                        source_refresh_wall_s = round(
                            time.perf_counter() - refresh_started,
                            6,
                        )
                        success, message, stage_stats = (
                            self.tp_worker.stage_cpu_weight_update_from_delta_lineage(
                                base_checkpoint_dir=base_checkpoint_dir,
                                checkpoint_source_dir=recv_req.checkpoint_source_dir,
                                target_version=recv_req.target_version,
                                host_cpu_group=self.host_cpu_group,
                            )
                        )
                        if not success:
                            raise RuntimeError(message)
                        stage_stats["source_refresh_wall_s"] = source_refresh_wall_s
                    local_stats["stage"] = stage_stats
            else:
                if recv_req.local_checkpoint_dir is None:
                    raise ValueError(
                        "local_checkpoint_dir is required for the disk destination"
                    )
                local_stats["stage"] = disk_checkpoint.materialize(
                    local_checkpoint_dir=recv_req.local_checkpoint_dir,
                    base_checkpoint_dir=base_checkpoint_dir,
                    checkpoint_source_dir=(
                        recv_req.checkpoint_source_dir or base_checkpoint_dir
                    ),
                    target_version=recv_req.target_version,
                    checkpoint_source_refresh_hook=(
                        server_args.checkpoint_source_refresh_hook
                    ),
                )
            success, message = True, "Success."
        except Exception:
            success, message = False, traceback.format_exc()
            logger.error(message)
        local_stats["wall_s"] = round(time.perf_counter() - started, 6)

        tp_size = (
            torch.distributed.get_world_size(group=self.weight_update_stage_cpu_group)
            if torch.distributed.is_initialized()
            else 1
        )
        rank_stats = [local_stats]
        if tp_size > 1:
            results = [None] * tp_size
            torch.distributed.all_gather_object(
                results,
                (success, message, local_stats),
                group=self.weight_update_stage_cpu_group,
            )
            success = all(ok for ok, _, _ in results)
            message = "; ".join(msg for ok, msg, _ in results if not ok) or message
            rank_stats = [stats for _, _, stats in results]
        if recv_req.destination == "cpu" and recv_req.target_version == 0:
            if not success:
                failed_before_stage = (
                    self._cpu_weight_cache_initialization_error is not None
                )
                self._cpu_weight_cache_initialization_error = message
                self._cpu_weight_cache_initialization_stats = None
                if not failed_before_stage:
                    self.tp_worker.discard_cpu_weight_cache(
                        "distributed CPU weight cache initialization failed",
                    )
        elif recv_req.destination == "cpu" and not success:
            self.tp_worker.invalidate_staged_cpu_weight_update(
                "distributed CPU weight update staging failed",
            )
        return StageWeightUpdateReqOutput(
            success=success,
            message=message,
            rank_stats=rank_stats,
        )

    def check_pending_weight_update_stage(self) -> None:
        if self._pending_weight_update_stage is None:
            return
        recv_req, thread = self._pending_weight_update_stage
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
        self.scheduler.ipc_channels.send_to_tokenizer.send_output(output, recv_req)

    def update_weights_from_cpu(
        self,
        recv_req: UpdateWeightFromCPUReqInput,
    ):
        """Update the target model from a complete, verified rank-ready CPU image."""

        with self._observe_weight_load("cpu"):
            if message := self._pending_weight_update_stage_message():
                return UpdateWeightFromCPUReqOutput(
                    success=False,
                    message=message,
                )
            if self._cpu_weight_cache_initialization_error is not None:
                preflight = (
                    False,
                    "CPU weight cache initialization failed.",
                )
            elif self._cpu_weight_cache_initialization_stats is None:
                preflight = (
                    False,
                    "CPU weight cache is not initialized.",
                )
            else:
                preflight = self.tp_worker.validate_staged_cpu_weight_update(
                    recv_req.target_version
                )
            tp_size = (
                torch.distributed.get_world_size(group=self.tp_cpu_group)
                if torch.distributed.is_initialized()
                else 1
            )
            preflight_results = [preflight]
            if tp_size > 1:
                preflight_results = [None] * tp_size
                torch.distributed.all_gather_object(
                    preflight_results,
                    preflight,
                    group=self.tp_cpu_group,
                )
            if not all(success for success, _ in preflight_results):
                return UpdateWeightFromCPUReqOutput(
                    success=False,
                    message="; ".join(
                        message for success, message in preflight_results if not success
                    ),
                )

            if recv_req.flush_cache:
                cache_flushed = self.flush_cache(empty_cache=recv_req.torch_empty_cache)
                cache_flush_results = [cache_flushed]
                if tp_size > 1:
                    cache_flush_results = [None] * tp_size
                    torch.distributed.all_gather_object(
                        cache_flush_results,
                        cache_flushed,
                        group=self.tp_cpu_group,
                    )
                if not all(cache_flush_results):
                    return UpdateWeightFromCPUReqOutput(
                        success=False,
                        message="Cache flush failed before CPU weight commit.",
                    )

            try:
                success, message, stats = self.tp_worker.update_weights_from_cpu(
                    recv_req
                )
            except Exception:
                success, message, stats = False, traceback.format_exc(), None

            rank = (
                torch.distributed.get_rank(group=self.tp_cpu_group)
                if torch.distributed.is_initialized()
                else 0
            )
            local_stats = {"rank": rank, **(stats or {})}
            rank_results = [(success, message, local_stats)]
            if tp_size > 1:
                rank_results = [None] * tp_size
                torch.distributed.all_gather_object(
                    rank_results,
                    (success, message, local_stats),
                    group=self.tp_cpu_group,
                )
                success = all(ok for ok, _, _ in rank_results)
                message = (
                    "; ".join(msg for ok, msg, _ in rank_results if not ok) or message
                )
            if not success:
                logger.critical(
                    "CPU weight commit failed after distributed preflight. "
                    "The engine cannot safely continue because one or more "
                    "ranks may contain a partially committed model: %s",
                    message,
                )
                raise RuntimeError(
                    "CPU weight commit failed; terminating the engine to avoid "
                    "serving a mixed or partially committed model. " + message
                )
            return UpdateWeightFromCPUReqOutput(
                success=success,
                message=message,
                rank_stats=[value for _, _, value in rank_results],
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
            "/update_weights_from_distributed"
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
        if message := self._cpu_weight_cache_conflict("/update_weights_from_tensor"):
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
        if message := self._cpu_weight_cache_conflict("/update_weights_from_ipc"):
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
