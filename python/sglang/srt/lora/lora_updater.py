"""Adapter-scoped replacement without stopping unrelated decode requests."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from weakref import WeakSet

import torch

if TYPE_CHECKING:
    from sglang.srt.lora.lora import LoRAAdapter
    from sglang.srt.lora.lora_config import LoRAConfig
    from sglang.srt.managers.io_struct import UpdateLoRAAdapterReqInput
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler


@dataclass
class PendingUpdate:
    request: UpdateLoRAAdapterReqInput
    future: Future[Any]
    phase: Literal["prepare", "upload", "rollback"] = "prepare"
    prepared: tuple[LoRAConfig, LoRAAdapter] | None = None
    error: str | None = None


class LoRAUpdater:
    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        self.manager = scheduler.tp_worker.model_runner.lora_manager
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lora-update"
        )
        self.pending: PendingUpdate | None = None
        self.paused: set[str] = set()
        self.retained: WeakSet[Req] = WeakSet()
        self.failed: dict[str, str] = {}

    def submit(self, request: UpdateLoRAAdapterReqInput) -> None:
        error = None
        unchanged = False
        try:
            if self.pending is not None:
                raise ValueError("Another LoRA replacement is in progress")
            uid = request.lora_id
            current = self.manager.lora_refs[uid]
            version = self.manager.lora_weight_versions.get(uid, current.lora_name)
            unchanged = uid not in self.failed and request.version == version
            if unchanged and request.lora_path != current.lora_path:
                raise ValueError("LoRA version already refers to a different path")
            if (
                not unchanged
                and request.expected_version is not None
                and request.expected_version != version
            ):
                raise ValueError(
                    f"Expected LoRA version {request.expected_version}, found {version}"
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        error = self._errors(error)
        if error or self._all_ready(unchanged):
            self._reply(request, error)
            return
        self.pending = PendingUpdate(
            request,
            self.executor.submit(
                copy_context().run, self.manager.prepare_lora_adapter, request.to_ref()
            ),
        )

    def is_paused(self, uid: str | None) -> bool:
        return uid in self.paused

    def stamp(self, req: Req) -> None:
        uid = req.lora_id
        if uid in self.manager.lora_cache_epochs:
            req.set_lora_cache_epoch(self.manager.lora_cache_epochs[uid])
            req.lora_weight_version = self.manager.lora_weight_versions[uid]

    def reject_failed(self, req: Req) -> bool:
        if req.lora_id not in self.failed:
            return False
        self.scheduler.fail_lora_request(req, self.failed[req.lora_id])
        return True

    def _all_ready(self, ready: bool) -> bool:
        if not torch.distributed.is_initialized():
            return ready
        value = torch.tensor([ready], dtype=torch.int32)
        torch.distributed.all_reduce(
            value, op=torch.distributed.ReduceOp.MIN, group=self.scheduler.tp_cpu_group
        )
        return bool(value.item())

    def _errors(self, error: str | None) -> str | None:
        if not torch.distributed.is_initialized():
            return error
        errors = [None] * torch.distributed.get_world_size(self.scheduler.tp_cpu_group)
        torch.distributed.all_gather_object(
            errors, error, group=self.scheduler.tp_cpu_group
        )
        return "; ".join(dict.fromkeys(e for e in errors if e)) or None

    def poll(self) -> None:
        pending = self.pending
        if pending is None or not self._all_ready(pending.future.done()):
            return
        request = pending.request
        uid = request.lora_id
        error = None
        try:
            value = pending.future.result()
            if pending.phase == "prepare":
                self.manager.validate_lora_replacement(request.to_ref(), value)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        error = self._errors(error)
        if pending.phase == "prepare":
            if error:
                self._finish(error)
                return
            pending.prepared = value
            try:
                self.scheduler.quiesce_lora_requests(uid)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            error = self._errors(error)
            if error:
                self.paused.discard(uid)
                self._finish(error)
                return
            self.paused.add(uid)
            if uid in self.manager.memory_pool.uid_to_buffer_id:
                self.manager.updating_loras.add(uid)
            pending.phase = "upload"
            pending.future = self.executor.submit(
                copy_context().run, self.manager.copy_lora_replacement, uid, value[1]
            )
        elif pending.phase == "upload":
            if error is None:
                try:
                    slot = self.manager.memory_pool.uid_to_buffer_id.get(uid)
                    if slot is not None:
                        self.manager._notify_lora_slots_updated({slot})
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            error = self._errors(error)
            if error:
                pending.error = error
                pending.phase = "rollback"
                pending.future = self.executor.submit(
                    copy_context().run,
                    self.manager.copy_lora_replacement,
                    uid,
                    self.manager.loras[uid],
                )
                return
            config, adapter = pending.prepared
            self.manager.configs[uid] = config
            self.manager.loras[uid] = adapter
            self.manager.lora_refs[uid] = request.to_ref()
            self.manager.lora_weight_versions[uid] = request.version
            self.manager.lora_cache_epochs[uid] = request.cache_epoch
            for req in self.scheduler.waiting_queue:
                if req.lora_id == uid:
                    self.stamp(req)
            self.failed.pop(uid, None)
            self.paused.discard(uid)
            self._finish(None)
        else:
            if error is None:
                try:
                    slot = self.manager.memory_pool.uid_to_buffer_id.get(uid)
                    if slot is not None:
                        self.manager._notify_lora_slots_updated({slot})
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            error = self._errors(error)
            if error:
                self.failed[uid] = (
                    f"LoRA replacement and rollback failed: {pending.error}; {error}"
                )
                for req in list(self.scheduler.waiting_queue):
                    if req.lora_id == uid:
                        self.scheduler.waiting_queue.remove(req)
                        self.reject_failed(req)
            else:
                self.paused.discard(uid)
            self._finish(pending.error)

    def _reply(self, request: UpdateLoRAAdapterReqInput, error: str | None) -> None:
        output = self.manager.create_lora_update_result(error is None, error or "")
        self.scheduler.send_control_output(output, request)

    def _finish(self, error: str | None) -> None:
        pending, self.pending = self.pending, None
        self.manager.updating_loras.discard(pending.request.lora_id)
        self._reply(pending.request, error)

    def forget(self, uid: str) -> None:
        self.paused.discard(uid)
        self.failed.pop(uid, None)
        for req in list(self.retained):
            if req.lora_id == uid:
                self.retained.discard(req)

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.pending is not None:
            self._finish("Server shut down during LoRA replacement")
