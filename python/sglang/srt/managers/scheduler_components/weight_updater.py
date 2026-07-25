from __future__ import annotations

import hashlib
import logging
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
    PullWeightsReqInput,
    PullWeightsReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromCPUReqInput,
    UpdateWeightsFromCPUReqOutput,
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
    background_cpu_group: Any
    host_cpu_group: Any
    memory_saver_adapter: Any
    flush_cache: Callable[..., bool]
    is_fully_idle: Callable[..., bool]
    scheduler: Optional[Any] = None
    metrics_collector: Optional[Any] = None
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None
    _pending_pull: Optional[Tuple[PullWeightsReqInput, threading.Thread]] = field(
        default=None,
        init=False,
    )
    _pull_result: Optional[PullWeightsReqOutput] = field(
        default=None,
        init=False,
    )

    @contextmanager
    def _observe_weight_load(self, source: str) -> Iterator[None]:
        # Edge-trigger weight_load_duration_seconds at the end of each
        # update_weights_from_* call. Engine is paused during the update so
        # the periodic log_stats path can't carry this.
        # `source` distinguishes disk vs distributed vs tensor vs ipc.
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

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
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

    def pull_weights(self, recv_req: PullWeightsReqInput):
        """Pull weights in a background thread while inference continues."""

        if self._pending_pull is not None:
            return PullWeightsReqOutput(
                success=False,
                message="Another weight pull or preparation is already running.",
            )

        def pull():
            self._pull_result = self._pull_weights_sync(recv_req)

        thread = threading.Thread(
            target=pull,
            name="weight-pull",
            daemon=True,
        )
        self._pending_pull = (recv_req, thread)
        thread.start()
        return None

    def _pull_weights_sync(self, recv_req: PullWeightsReqInput):
        """Pull one verified target to disk or rank-ready CPU memory."""
        from sglang.srt.weight_sync import local_checkpoint

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
            base_checkpoint_dir = recv_req.base_checkpoint_dir or server_args.model_path
            if recv_req.destination == "cpu":
                if self.draft_worker is not None:
                    raise NotImplementedError(
                        "CPU preparation does not yet support a speculative "
                        "draft worker; refusing to update only the target model"
                    )
                refresh_started = time.perf_counter()
                refresh_error = None
                host_rank = (
                    torch.distributed.get_rank(group=self.host_cpu_group)
                    if torch.distributed.is_initialized()
                    else 0
                )
                if (
                    host_rank == 0
                    and recv_req.target_version > 0
                    and server_args.weight_source_refresh_hook
                ):
                    try:
                        local_checkpoint.refresh_source(
                            recv_req.source_dir,
                            recv_req.target_version,
                            server_args.weight_source_refresh_hook,
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
                        "weight source refresh failed: " + "; ".join(refresh_errors)
                    )
                local_stats["checkpoint_pull"] = {
                    "operation": "cpu_prepare",
                    "materialized_target_checkpoint": False,
                    "source_refresh_wall_s": round(
                        time.perf_counter() - refresh_started,
                        6,
                    ),
                }
                local_stats["checkpoint_pull_wall_s"] = 0.0
                prepare_started = time.perf_counter()
                success, message, prepare_stats = self.tp_worker.prepare_cpu_weights(
                    base_checkpoint_dir=base_checkpoint_dir,
                    source_dir=recv_req.source_dir,
                    target_version=recv_req.target_version,
                    host_cpu_group=self.host_cpu_group,
                )
                if not success:
                    raise RuntimeError(message)
                local_stats["runtime_prepare_wall_s"] = round(
                    time.perf_counter() - prepare_started,
                    6,
                )
                local_stats["runtime_prepare"] = prepare_stats
            else:
                if recv_req.local_checkpoint_dir is None:
                    raise ValueError(
                        "local_checkpoint_dir is required for the disk destination"
                    )
                pull_started = time.perf_counter()
                pull_stats = local_checkpoint.pull(
                    local_checkpoint_dir=recv_req.local_checkpoint_dir,
                    base_dir=base_checkpoint_dir,
                    source_dir=recv_req.source_dir,
                    target_version=recv_req.target_version,
                    source_refresh_hook=server_args.weight_source_refresh_hook,
                )
                local_stats["checkpoint_pull_wall_s"] = round(
                    time.perf_counter() - pull_started,
                    6,
                )
                local_stats["checkpoint_pull"] = pull_stats
            success, message = True, "Success."
        except Exception:
            success, message = False, traceback.format_exc()
            logger.error(message)
        local_stats["total_wall_s"] = round(time.perf_counter() - started, 6)

        tp_size = (
            torch.distributed.get_world_size(group=self.background_cpu_group)
            if torch.distributed.is_initialized()
            else 1
        )
        rank_stats = [local_stats]
        if tp_size > 1:
            results = [None] * tp_size
            torch.distributed.all_gather_object(
                results,
                (success, message, local_stats),
                group=self.background_cpu_group,
            )
            success = all(ok for ok, _, _ in results)
            message = "; ".join(msg for ok, msg, _ in results if not ok) or message
            rank_stats = [stats for _, _, stats in results]
        return PullWeightsReqOutput(
            success=success,
            message=message,
            rank_stats=rank_stats,
        )

    def check_pending_pull(self) -> None:
        if self._pending_pull is None:
            return
        recv_req, thread = self._pending_pull
        if thread.is_alive():
            return
        self._pending_pull = None
        thread.join()
        output = self._pull_result
        self._pull_result = None
        if output is None:
            output = PullWeightsReqOutput(
                success=False, message="Weight pull ended without a result."
            )
        self.scheduler.ipc_channels.send_to_tokenizer.send_output(output, recv_req)

    def update_weights_from_cpu(
        self,
        recv_req: UpdateWeightsFromCPUReqInput,
    ):
        """Update from one complete, verified rank-ready CPU image."""

        with self._observe_weight_load("cpu"):
            if self._pending_pull is not None:
                return UpdateWeightsFromCPUReqOutput(
                    success=False,
                    message="CPU weight preparation is still running.",
                )
            if self.draft_worker is not None:
                return UpdateWeightsFromCPUReqOutput(
                    success=False,
                    message=(
                        "CPU updates do not yet support a speculative draft "
                        "worker; refusing to update only the target model."
                    ),
                )
            preflight = self.tp_worker.validate_cpu_weight_update(
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
                return UpdateWeightsFromCPUReqOutput(
                    success=False,
                    message="; ".join(
                        message for success, message in preflight_results if not success
                    ),
                )

            success, message, stats = self.tp_worker.update_weights_from_cpu(recv_req)
            if success:
                self.flush_cache_after_weight_update(recv_req)
            else:
                logger.error(message)

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
            return UpdateWeightsFromCPUReqOutput(
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
        assert (
            self.is_fully_idle()
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

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
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

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
