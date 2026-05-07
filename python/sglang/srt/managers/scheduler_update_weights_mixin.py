from __future__ import annotations

import logging
import time
import traceback
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.managers.io_struct import (
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DiscardPreparedWeightsFromTensorReqInput,
    DiscardPreparedWeightsFromTensorReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    PrepareWeightsFromTensorReqInput,
    PrepareWeightsFromTensorReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from sglang.srt.managers.weight_update.tracing import elapsed_ms, ensure_update_trace

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerUpdateWeightsMixin:
    def _record_successful_weight_update(self: Scheduler, recv_req) -> None:
        if getattr(recv_req, "weight_epoch", None) is not None:
            self.current_weight_epoch = recv_req.weight_epoch

    def update_weights_from_disk(
        self: Scheduler, recv_req: UpdateWeightFromDiskReqInput
    ):
        """In-place update of the weights from disk."""
        success, message = self.tp_worker.update_weights_from_disk(recv_req)
        tp_success = success
        if success and self.draft_worker is not None:
            success, message = self.draft_worker.update_weights_from_disk(recv_req)
        if tp_success and recv_req.flush_cache:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            assert flush_cache_success, "Cache flush failed after updating weights"
        if not success:
            logger.error(message)
        return UpdateWeightFromDiskReqOutput(success, message, 0)

    def init_weights_update_group(
        self: Scheduler, recv_req: InitWeightsUpdateGroupReqInput
    ):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success, message)

    def destroy_weights_update_group(
        self: Scheduler, recv_req: DestroyWeightsUpdateGroupReqInput
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success, message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        success, message = self.tp_worker.update_weights_from_distributed(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache(
                    empty_cache=recv_req.torch_empty_cache
                )
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightsFromDistributedReqOutput(success, message)

    def update_weights_from_tensor(
        self: Scheduler, recv_req: UpdateWeightsFromTensorReqInput
    ):
        """Update the online model parameter from tensors."""
        trace = ensure_update_trace(recv_req)
        update_started_at = time.monotonic()
        trace["scheduler_update_start_monotonic"] = update_started_at
        trace["scheduler_tp_rank"] = getattr(self.tp_worker, "tp_rank", None)
        pause_trace = getattr(self, "_last_generation_pause_trace", None)
        request_id = trace.get("request_id")
        if (
            pause_trace is not None
            and request_id is not None
            and pause_trace.get("request_id") == request_id
        ):
            trace["scheduler_pause_trace"] = dict(pause_trace)
            trace["scheduler_pause_to_update_start_ms"] = round(
                (
                    update_started_at
                    - pause_trace["scheduler_pause_generation_start_monotonic"]
                )
                * 1000,
                3,
            )
        if recv_req.disable_draft_model:
            worker = self.tp_worker
        else:
            worker = self.draft_worker or self.tp_worker
        trace["scheduler_worker_kind"] = (
            "draft_worker" if worker is self.draft_worker else "tp_worker"
        )
        worker_started_at = time.monotonic()
        success, message = worker.update_weights_from_tensor(recv_req)
        trace["scheduler_worker_update_ms"] = elapsed_ms(worker_started_at)
        # TODO extract common code b/t update_weights_from_distributed and update_weights_from_tensor later
        if success:
            if recv_req.flush_cache:
                flush_started_at = time.monotonic()
                flush_cache_success = self.flush_cache(
                    empty_cache=recv_req.torch_empty_cache
                )
                trace["scheduler_flush_cache_ms"] = elapsed_ms(flush_started_at)
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        barrier_started_at = time.monotonic()
        torch.distributed.barrier(group=self.tp_cpu_group)
        trace["scheduler_tp_cpu_barrier_ms"] = elapsed_ms(barrier_started_at)
        if success:
            self._record_successful_weight_update(recv_req)
        trace["scheduler_success"] = success
        trace["scheduler_update_total_ms"] = elapsed_ms(update_started_at)
        return UpdateWeightsFromTensorReqOutput(success, message, trace=trace)

    def prepare_weights_from_tensor(
        self: Scheduler, recv_req: PrepareWeightsFromTensorReqInput
    ):
        trace = ensure_update_trace(recv_req)
        prepare_started_at = time.monotonic()
        trace["scheduler_prestage_start_monotonic"] = prepare_started_at
        trace["scheduler_tp_rank"] = getattr(self.tp_worker, "tp_rank", None)
        if recv_req.disable_draft_model:
            worker = self.tp_worker
        else:
            worker = self.draft_worker or self.tp_worker
        trace["scheduler_worker_kind"] = (
            "draft_worker" if worker is self.draft_worker else "tp_worker"
        )
        worker_started_at = time.monotonic()
        success, message = worker.prepare_weights_from_tensor(recv_req)
        trace["scheduler_prepare_worker_ms"] = elapsed_ms(worker_started_at)
        barrier_started_at = time.monotonic()
        torch.distributed.barrier(group=self.tp_cpu_group)
        trace["scheduler_prepare_tp_cpu_barrier_ms"] = elapsed_ms(barrier_started_at)
        if not success:
            logger.error(message)
        trace["scheduler_prestage_success"] = success
        trace["scheduler_prepare_total_ms"] = elapsed_ms(prepare_started_at)
        return PrepareWeightsFromTensorReqOutput(success, message, trace=trace)

    def discard_prepared_weights_from_tensor(
        self: Scheduler, recv_req: DiscardPreparedWeightsFromTensorReqInput
    ):
        trace = ensure_update_trace(recv_req)
        discard_started_at = time.monotonic()
        if recv_req.disable_draft_model:
            worker = self.tp_worker
        else:
            worker = self.draft_worker or self.tp_worker
        worker_started_at = time.monotonic()
        success, message = worker.discard_prepared_weights_from_tensor(recv_req)
        trace["scheduler_discard_worker_ms"] = elapsed_ms(worker_started_at)
        barrier_started_at = time.monotonic()
        torch.distributed.barrier(group=self.tp_cpu_group)
        trace["scheduler_discard_tp_cpu_barrier_ms"] = elapsed_ms(barrier_started_at)
        if not success:
            logger.error(message)
        trace["scheduler_discard_total_ms"] = elapsed_ms(discard_started_at)
        return DiscardPreparedWeightsFromTensorReqOutput(success, message, trace=trace)

    def update_weights_from_ipc(
        self: Scheduler, recv_req: UpdateWeightsFromIPCReqInput
    ):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        success, message = self.tp_worker.update_weights_from_ipc(recv_req)
        tp_success = success
        if success and self.draft_worker is not None:
            success, message = self.draft_worker.update_weights_from_ipc(recv_req)
        if tp_success and recv_req.flush_cache:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            assert flush_cache_success, "Cache flush failed after updating weights"
        if not success:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromIPCReqOutput(success, message)

    def get_weights_by_name(self: Scheduler, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter)

    def release_memory_occupation(
        self: Scheduler, recv_req: ReleaseMemoryOccupationReqInput
    ):
        assert (
            self.is_fully_idle()
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
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

    def resume_memory_occupation(
        self: Scheduler, recv_req: ResumeMemoryOccupationReqInput
    ):
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

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self: Scheduler, recv_req: CheckWeightsReqInput):
        try:
            payload = self.tp_worker.model_runner.check_weights(action=recv_req.action)
            return CheckWeightsReqOutput(
                success=True, message="Success.", payload=payload
            )
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self: Scheduler, params):
        url = params["url"]

        self.tp_worker.model_runner.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.save_remote_model(draft_url)

    def save_sharded_model(self: Scheduler, params):
        self.tp_worker.model_runner.save_sharded_model(
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
