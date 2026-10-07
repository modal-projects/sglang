from __future__ import annotations

import logging
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

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
    BeginWeightUpdateReqInput,
    BeginWeightUpdateReqOutput,
    ChecksumInfo,
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    CommitWeightUpdateReqInput,
    CommitWeightUpdateReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    EndWeightUpdateReqInput,
    EndWeightUpdateReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    PrepareWeightUpdateReqInput,
    PrepareWeightUpdateReqOutput,
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
from sglang.srt.runtime_context import get_model
from sglang.srt.utils.weight_checker import overall_checksum
from sglang.srt.weight_sync.rank_weight_image import iter_weight_tensors

logger = logging.getLogger(__name__)


def _merge_checksum_payloads(role_payloads: List[Tuple[str, Dict]]) -> Dict:
    merged: Dict[str, str] = {}
    parallelism_infos = []
    for role, p in role_payloads:
        for name, chk in p["checksums"].items():
            # only draft roles are prefixed, so target keys stay stable
            key = name if role == "target" else f"{role}.{name}"
            if key in merged:
                raise ValueError(f"checksum key collision: {key}")
            merged[key] = chk
        parallelism_infos.append(p["parallelism_info"])
    return {
        "checksums": merged,
        "per_gpu_checksum": overall_checksum(merged),
        "parallelism_info": parallelism_infos,
    }


def _parse_runner_selector(selector: str) -> Set[str]:
    if selector == "all":
        return {"target", "draft"}
    if selector in ("target", "draft"):
        return {selector}
    raise ValueError(
        f"invalid selector {selector!r}; expected 'target', 'draft', or 'all'"
    )


class _WeightUpdateSession(msgspec.Struct, frozen=True):
    # recorded at begin so end finalizes the same runners
    selector: str
    loaded_weights: bool = False


def _weight_tensor_layout(model: torch.nn.Module) -> dict[str, tuple]:
    """Describe the tensor views whose addresses may be captured at runtime."""
    layout = {}
    for name, tensor in iter_weight_tensors(model):
        storage = tensor.untyped_storage()
        layout[name] = (
            str(tensor.device),
            tensor.data_ptr(),
            storage.data_ptr(),
            storage.nbytes(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
        )
    return layout


def _assert_weight_tensor_layout_unchanged(
    before: dict[str, tuple], model: torch.nn.Module
) -> None:
    after = _weight_tensor_layout(model)
    changed = sorted(
        name
        for name in before.keys() | after.keys()
        if before.get(name) != after.get(name)
    )
    if changed:
        raise RuntimeError(
            "in-place disk reload changed live weight tensor layouts; existing "
            f"runtime references are unsafe (changed={changed[:20]})"
        )


@dataclass(kw_only=True, slots=True)
class SchedulerWeightUpdaterManager:
    tp_worker: Any
    draft_worker: Any
    tp_cpu_group: Any
    memory_saver_adapter: Any
    flush_cache: Callable[..., bool]
    is_fully_idle: Callable[..., bool]
    scheduler: Optional[Any] = None
    weight_stage_cpu_group: Any = None
    host_cpu_group: Any = None
    weight_update_staging: Optional[str] = None
    weight_update_local_checkpoint_dir: Optional[str] = None
    weight_update_base_checkpoint_dir: str = ""
    weight_update_base_version: int = 0
    weight_update_max_compile_group_bytes: int = 8 << 30
    send_control_output: Optional[Callable[[Any, Any], None]] = None
    metrics_collector: Optional[Any] = None
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None
    # replicated on every TP rank, so a rejected call returns on all ranks before any barrier
    _session: Optional[_WeightUpdateSession] = None
    _pending_weight_preparation: Optional[
        Tuple[PrepareWeightUpdateReqInput, threading.Thread]
    ] = field(default=None, init=False)
    _weight_preparation_result: Optional[PrepareWeightUpdateReqOutput] = field(
        default=None,
        init=False,
    )
    _prepared_weight_version: Optional[int] = field(default=None, init=False)
    _served_weight_version: int = field(init=False)

    def __post_init__(self) -> None:
        self._served_weight_version = self.weight_update_base_version

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

    def initialize_weight_staging(self) -> None:
        """Construct the configured inactive destination before serving."""

        backend = self.weight_update_staging
        if backend is None:
            return
        started = time.perf_counter()
        local_error = None
        try:
            if backend == "cpu":
                stats = self._target_runner().weight_updater.initialize_rank_weight_stager(
                    checkpoint_dir=self.weight_update_base_checkpoint_dir,
                    version=self.weight_update_base_version,
                    host_group=self.host_cpu_group,
                    max_compile_group_bytes=self.weight_update_max_compile_group_bytes,
                    canonical_checkpoint_dir=self.weight_update_local_checkpoint_dir,
                )
            else:
                from sglang.srt.weight_sync.disk_checkpoint import materialize

                if self.weight_update_local_checkpoint_dir is None:
                    raise RuntimeError(
                        "disk weight staging requires a local checkpoint directory"
                    )
                stats = materialize(
                    local_checkpoint_dir=self.weight_update_local_checkpoint_dir,
                    base_checkpoint_dir=self.weight_update_base_checkpoint_dir,
                    checkpoint_source_dir=self.weight_update_base_checkpoint_dir,
                    target_version=self.weight_update_base_version,
                    base_version=self.weight_update_base_version,
                )
        except Exception:
            stats = None
            local_error = traceback.format_exc()

        errors = [
            error
            for error in self._all_gather(local_error, self.tp_cpu_group)
            if error is not None
        ]
        if errors:
            if backend == "cpu":
                self._target_runner().weight_updater.close_rank_weight_stager()
            raise RuntimeError(
                "failed to initialize staged weight updates:\n" + "\n".join(errors)
            )
        logger.info(
            "Initialized %s weight staging at v%d in %.3fs: %s",
            backend,
            self.weight_update_base_version,
            time.perf_counter() - started,
            stats,
        )

    def _staged_update_conflict(self, operation: str) -> str | None:
        if self.weight_update_staging is None:
            return None
        return (
            f"{operation} is unavailable while staged weight updates are enabled; "
            "use prepare_weight_update and commit_weight_update"
        )

    def _pending_preparation_message(self) -> str | None:
        if self.weight_update_staging is None:
            return None
        pending = self._pending_weight_preparation is not None
        if not any(self._all_gather(pending, self.tp_cpu_group)):
            return None
        return "A background weight preparation is still running."

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

    def record_weight_version_after_update(self, weight_version: Optional[str]) -> None:
        self.scheduler.record_weight_version_change(new_version=weight_version)

    def prepare_weight_update(self, recv_req: PrepareWeightUpdateReqInput):
        """Start preparation without blocking inference scheduling."""

        local_state = {
            "backend": self.weight_update_staging,
            "pending": self._pending_weight_preparation is not None,
            "prepared": self._prepared_weight_version,
            "served": self._served_weight_version,
        }
        states = self._all_gather(local_state, self.tp_cpu_group)
        if any(state != states[0] for state in states[1:]):
            return PrepareWeightUpdateReqOutput(
                success=False,
                message=f"Weight preparation state differs across ranks: {states}",
            )
        state = states[0]
        if state["backend"] is None:
            return PrepareWeightUpdateReqOutput(
                success=False,
                message="Staged weight updates are not enabled.",
            )
        if state["pending"]:
            return PrepareWeightUpdateReqOutput(
                success=False,
                message="Another weight preparation is already running.",
            )
        if not recv_req.checkpoint_source_dir:
            return PrepareWeightUpdateReqOutput(
                success=False,
                message="checkpoint_source_dir must not be empty.",
            )
        if state["prepared"] is not None:
            if state["prepared"] == recv_req.target_version:
                return PrepareWeightUpdateReqOutput(
                    success=True,
                    message=f"Weight version {recv_req.target_version} is prepared.",
                )
            if recv_req.target_version < state["prepared"]:
                return PrepareWeightUpdateReqOutput(
                    success=False,
                    message=(
                        f"Target version {recv_req.target_version} must not precede "
                        f"prepared version {state['prepared']}."
                    ),
                )
            self._prepared_weight_version = None
            logger.info(
                "Discarded prepared weight version %d in favor of v%d.",
                state["prepared"],
                recv_req.target_version,
            )
        if recv_req.target_version <= state["served"]:
            return PrepareWeightUpdateReqOutput(
                success=False,
                message=(
                    f"Target version {recv_req.target_version} must follow served "
                    f"version {state['served']}."
                ),
            )

        def prepare() -> None:
            try:
                output = self._prepare_weight_update_sync(recv_req)
            except Exception:
                logger.exception("Background weight preparation failed")
                output = PrepareWeightUpdateReqOutput(
                    success=False,
                    message=traceback.format_exc(),
                )
            self._weight_preparation_result = output

        thread = threading.Thread(
            target=prepare,
            name="weight-preparation",
            daemon=True,
        )
        self._pending_weight_preparation = (recv_req, thread)
        thread.start()
        return None

    def _prepare_weight_update_sync(
        self,
        recv_req: PrepareWeightUpdateReqInput,
    ) -> PrepareWeightUpdateReqOutput:
        started = time.perf_counter()
        local_error = None
        local_stats = None
        try:
            if self.weight_update_staging == "cpu":
                local_stats = (
                    self._target_runner().weight_updater.stage_rank_weight_update(
                        checkpoint_source_dir=recv_req.checkpoint_source_dir,
                        target_version=recv_req.target_version,
                    )
                )
            else:
                from sglang.srt.weight_sync.disk_checkpoint import materialize

                if self.weight_update_local_checkpoint_dir is None:
                    raise RuntimeError(
                        "disk weight staging requires a local checkpoint directory"
                    )
                local_stats = materialize(
                    local_checkpoint_dir=self.weight_update_local_checkpoint_dir,
                    base_checkpoint_dir=self.weight_update_base_checkpoint_dir,
                    checkpoint_source_dir=recv_req.checkpoint_source_dir,
                    target_version=recv_req.target_version,
                    base_version=self.weight_update_base_version,
                )
        except Exception:
            local_error = traceback.format_exc()

        rank = (
            torch.distributed.get_rank(group=self.weight_stage_cpu_group)
            if torch.distributed.is_initialized()
            else 0
        )
        result = {
            "rank": rank,
            "backend": self.weight_update_staging,
            "target_version": recv_req.target_version,
            "stage": local_stats,
            "wall_s": round(time.perf_counter() - started, 6),
        }
        gathered = self._all_gather(
            (local_error, result),
            self.weight_stage_cpu_group,
        )
        errors = [error for error, _ in gathered if error is not None]
        if errors:
            if self.weight_update_staging == "cpu":
                self._target_runner().weight_updater.discard_prepared_rank_weights(
                    "distributed weight preparation failed"
                )
            return PrepareWeightUpdateReqOutput(
                success=False,
                message="\n".join(dict.fromkeys(errors)),
                rank_stats=[rank_result for _, rank_result in gathered],
            )

        self._prepared_weight_version = recv_req.target_version
        return PrepareWeightUpdateReqOutput(
            success=True,
            message=f"Prepared weight version {recv_req.target_version}.",
            rank_stats=[rank_result for _, rank_result in gathered],
        )

    def check_pending_weight_preparation(self) -> None:
        pending = self._pending_weight_preparation
        if pending is None:
            return
        recv_req, thread = pending
        if thread.is_alive():
            time.sleep(0)
            return
        thread.join()
        self._pending_weight_preparation = None
        output = self._weight_preparation_result
        self._weight_preparation_result = None
        if output is None:
            output = PrepareWeightUpdateReqOutput(
                success=False,
                message="Weight preparation ended without a result.",
            )
        if self.send_control_output is None:
            raise RuntimeError("weight preparation has no control-output channel")
        self.send_control_output(recv_req, output)

    def commit_weight_update(
        self,
        recv_req: CommitWeightUpdateReqInput,
    ) -> CommitWeightUpdateReqOutput:
        """Commit one collectively prepared target while scheduling is paused."""

        with self._observe_weight_load(
            f"staged_{self.weight_update_staging or 'disabled'}"
        ):
            if self.weight_update_staging is None:
                return CommitWeightUpdateReqOutput(
                    success=False,
                    message="Staged weight updates are not enabled.",
                )
            if message := self._pending_preparation_message():
                return CommitWeightUpdateReqOutput(success=False, message=message)

            local_preflight_error = None
            try:
                if self._prepared_weight_version != recv_req.target_version:
                    raise RuntimeError(
                        f"Weight version {recv_req.target_version} is not prepared; "
                        f"prepared={self._prepared_weight_version}."
                    )
                if self.weight_update_staging == "cpu":
                    self._target_runner().weight_updater.validate_rank_weight_commit(
                        recv_req.target_version
                    )
            except Exception:
                local_preflight_error = traceback.format_exc()
            preflight_errors = [
                error
                for error in self._all_gather(
                    local_preflight_error,
                    self.tp_cpu_group,
                )
                if error is not None
            ]
            if preflight_errors:
                return CommitWeightUpdateReqOutput(
                    success=False,
                    message="\n".join(dict.fromkeys(preflight_errors)),
                )

            if recv_req.flush_cache:
                cache_flushed = self.flush_cache(empty_cache=recv_req.torch_empty_cache)
                if not all(self._all_gather(cache_flushed, self.tp_cpu_group)):
                    return CommitWeightUpdateReqOutput(
                        success=False,
                        message="Cache flush failed before weight commit.",
                    )

            started = time.perf_counter()
            local_error = None
            local_stats = None
            try:
                if self.weight_update_staging == "cpu":
                    local_stats = (
                        self._target_runner().weight_updater.commit_rank_weight_update(
                            recv_req.target_version
                        )
                    )
                else:
                    runner = self._target_runner()
                    model = runner.model
                    weight_layout = _weight_tensor_layout(model)
                    success, message = runner.weight_updater.update_weights_from_disk(
                        self.weight_update_local_checkpoint_dir,
                        runner.load_config.load_format,
                        recapture_cuda_graph=False,
                    )
                    if not success:
                        raise RuntimeError(message)
                    _assert_weight_tensor_layout_unchanged(weight_layout, model)
                    local_stats = {
                        "operation": "commit_disk_weight_update",
                        "wall_s": round(time.perf_counter() - started, 6),
                    }
            except Exception:
                local_error = traceback.format_exc()

            rank = (
                torch.distributed.get_rank(group=self.tp_cpu_group)
                if torch.distributed.is_initialized()
                else 0
            )
            result = {
                "rank": rank,
                "backend": self.weight_update_staging,
                "target_version": recv_req.target_version,
                "commit": local_stats,
                "wall_s": round(time.perf_counter() - started, 6),
            }
            gathered = self._all_gather((local_error, result), self.tp_cpu_group)
            errors = [error for error, _ in gathered if error is not None]
            if errors:
                logger.critical(
                    "Weight commit failed after live weights may have changed: %s",
                    "\n".join(errors),
                )
                raise RuntimeError(
                    "weight commit failed; terminating the engine to avoid serving "
                    "mixed weights:\n" + "\n".join(dict.fromkeys(errors))
                )

            self._served_weight_version = recv_req.target_version
            self._prepared_weight_version = None
            self.record_weight_version_after_update(str(recv_req.target_version))
            return CommitWeightUpdateReqOutput(
                success=True,
                message=f"Committed weight version {recv_req.target_version}.",
                rank_stats=[rank_result for _, rank_result in gathered],
            )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        if message := self._pending_preparation_message():
            return UpdateWeightFromDiskReqOutput(success=False, message=message)
        if message := self._staged_update_conflict("update_weights_from_disk"):
            return UpdateWeightFromDiskReqOutput(success=False, message=message)
        with self._observe_weight_load("disk"):
            success, message = True, "Succeeded to update model weights."
            target_updated = False
            for role, runner in self._select_runners():
                success, message = runner.weight_updater.update_weights_from_disk(
                    recv_req.model_path,
                    recv_req.load_format,
                    recapture_cuda_graph=recv_req.recapture_cuda_graph,
                )
                if not success:
                    break
                target_updated |= role == "target"
            # the served weights changed even if a draft runner failed afterwards
            if target_updated:
                self.flush_cache_after_weight_update(recv_req)
            if success:
                self.record_weight_version_after_update(recv_req.weight_version)
            else:
                logger.error(message)
            return UpdateWeightFromDiskReqOutput(
                success=success, message=message, num_paused_requests=0
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

    def _target_runner(self):
        runners = self._select_runners("target")
        if len(runners) != 1 or runners[0][0] != "target":
            raise RuntimeError(
                "staged weight updates require one target runner per rank"
            )
        return runners[0][1]

    def _select_runners(self, selector: str = "all") -> List[Tuple[str, Any]]:
        roles = _parse_runner_selector(selector)
        runners: List[Tuple[str, Any]] = []
        if "target" in roles:
            runners += self.tp_worker.weight_update_runners()
        if "draft" in roles and self.draft_worker is not None:
            runners += self.draft_worker.weight_update_runners()
        return runners

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        if message := self._pending_preparation_message():
            return UpdateWeightsFromDistributedReqOutput(success=False, message=message)
        if message := self._staged_update_conflict("update_weights_from_distributed"):
            return UpdateWeightsFromDistributedReqOutput(success=False, message=message)
        if self._session is None:
            return UpdateWeightsFromDistributedReqOutput(
                success=False,
                message="update_weights_from_distributed must run between "
                "begin_weight_update() and end_weight_update()",
            )
        with self._observe_weight_load("distributed"):
            # only the target runner joined the update group; drafts load its receive
            target = self.tp_worker.model_runner.weight_updater
            try:
                weights = target.receive_weights_from_distributed(
                    names=recv_req.names,
                    dtypes=recv_req.dtypes,
                    shapes=recv_req.shapes,
                    group_name=recv_req.group_name,
                    load_format=recv_req.load_format,
                )
            except Exception as e:
                success, message = False, f"Failed to receive weights: {e}"
                logger.error(message)
            else:
                success, message = True, "Succeeded to update parameter online."
                for _, runner in self._select_runners(recv_req.selector):
                    success, message = (
                        runner.weight_updater.load_weights_from_distributed(weights)
                    )
                    if not success:
                        break
            if success:
                self._session = msgspec.structs.replace(
                    self._session, loaded_weights=True
                )
                self.flush_cache_after_weight_update(recv_req)
                self.record_weight_version_after_update(recv_req.weight_version)
            return UpdateWeightsFromDistributedReqOutput(
                success=success, message=message
            )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors."""
        if message := self._pending_preparation_message():
            return UpdateWeightsFromTensorReqOutput(success=False, message=message)
        if message := self._staged_update_conflict("update_weights_from_tensor"):
            return UpdateWeightsFromTensorReqOutput(success=False, message=message)
        if self._session is None:
            return UpdateWeightsFromTensorReqOutput(
                success=False,
                message="update_weights_from_tensor must run between "
                "begin_weight_update() and end_weight_update()",
            )
        with self._observe_weight_load("tensor"):
            named_tensors = self.tp_worker.deserialize_own_rank(
                recv_req.serialized_named_tensors
            )
            success, message = True, "Success"
            for _, runner in self._select_runners(recv_req.selector):
                success, message = runner.weight_updater.update_weights_from_tensor(
                    named_tensors=named_tensors,
                    load_format=recv_req.load_format,
                )
                if not success:
                    break
            if success:
                self._session = msgspec.structs.replace(
                    self._session, loaded_weights=True
                )
                self.flush_cache_after_weight_update(recv_req)
                self.record_weight_version_after_update(recv_req.weight_version)
            else:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromTensorReqOutput(success=success, message=message)

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        if message := self._pending_preparation_message():
            return UpdateWeightsFromIPCReqOutput(success=False, message=message)
        if message := self._staged_update_conflict("update_weights_from_ipc"):
            return UpdateWeightsFromIPCReqOutput(success=False, message=message)
        with self._observe_weight_load("ipc"):
            success, message = True, "Succeeded to update model weights."
            target_updated = False
            for role, runner in self._select_runners():
                success, message = runner.weight_updater.update_weights_from_ipc(
                    recv_req
                )
                if not success:
                    break
                target_updated |= role == "target"
            # the served weights changed even if a draft runner failed afterwards
            if target_updated:
                self.flush_cache_after_weight_update(recv_req)
            if success:
                self.record_weight_version_after_update(recv_req.weight_version)
            else:
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
        mode = get_model().weight_cache_mode
        if mode != "off":
            raise RuntimeError(
                f"[weight_cache] {op} of model weights is not supported while the "
                f"weight cache is active (--weight-cache-mode {mode}): the weights "
                f"are shared with the daemon via CUDA IPC, so freeing them would "
                f"corrupt the daemon's master copy and every co-attached engine. "
                f"Restart with --weight-cache-mode off to use this operation."
            )

    def begin_weight_update(self, recv_req: BeginWeightUpdateReqInput):
        """Open the session: restore in-place-packed weights on the selected runners."""
        if message := self._staged_update_conflict("begin_weight_update"):
            return BeginWeightUpdateReqOutput(success=False, message=message)
        if self._session is not None:
            return BeginWeightUpdateReqOutput(
                success=False,
                message="a weight-update session is already open; "
                "call end_weight_update() first",
            )
        for _, runner in self._select_runners(recv_req.selector):
            runner.weight_updater.begin_weight_update()
        self._session = _WeightUpdateSession(selector=recv_req.selector)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return BeginWeightUpdateReqOutput(success=True, message="Success")

    def end_weight_update(self, recv_req: EndWeightUpdateReqInput):
        """Finalize the runners begin opened; post_load_weights only if no load ran (P2P/RDMA)."""
        if message := self._staged_update_conflict("end_weight_update"):
            return EndWeightUpdateReqOutput(success=False, message=message)
        if self._session is None:
            return EndWeightUpdateReqOutput(
                success=False,
                message="no weight-update session is open; call begin_weight_update() first",
            )
        run_post_load = not self._session.loaded_weights
        for _, runner in self._select_runners(self._session.selector):
            runner.weight_updater.end_weight_update(run_post_load=run_post_load)
        self._session = None
        torch.distributed.barrier(group=self.tp_cpu_group)
        return EndWeightUpdateReqOutput(success=True, message="Success")

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        scheduler = self.scheduler
        assert self.is_fully_idle(
            ignore_waiting=scheduler is not None and scheduler._engine_paused
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            if message := self._pending_preparation_message():
                raise RuntimeError(message)
            if message := self._staged_update_conflict("releasing model-weight memory"):
                raise RuntimeError(message)
            self._assert_weight_cache_inactive("release_memory_occupation")

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
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

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            if message := self._pending_preparation_message():
                raise RuntimeError(message)
            if message := self._staged_update_conflict("resuming model-weight memory"):
                raise RuntimeError(message)
            self._assert_weight_cache_inactive("resume_memory_occupation")

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
            role_payloads = []
            for role, runner in self._select_runners(recv_req.selector):
                p = runner.check_weights(
                    action=recv_req.action,
                    allow_quant_error=recv_req.allow_quant_error,
                    skip_tensor_list=recv_req.skip_tensor_list,
                    role=role,
                )
                if p is not None:
                    role_payloads.append((role, p))
            payload = _merge_checksum_payloads(role_payloads) if role_payloads else None

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
            assert draft_url is not None, (
                "draft_url must be provided when draft model is enabled"
            )
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
