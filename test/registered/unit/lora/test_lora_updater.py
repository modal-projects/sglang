"""Adapter replacement isolates requests, slots and cache generations."""

from concurrent.futures import Future
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang.srt.lora import lora_updater
from sglang.srt.lora.lora_updater import LoRAUpdater
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Executor:
    def __init__(self, **kwargs: Any) -> None:
        self.jobs: list[tuple[Future, Any, tuple]] = []
        self.closed = False

    def submit(self, function: Any, *args: Any) -> Future:
        future = Future()
        self.jobs.append((future, function, args))
        return future

    def run(self) -> None:
        future, function, args = self.jobs.pop(0)
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(function(*args))
            except Exception as error:
                future.set_exception(error)

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.closed = True
        for future, _, _ in self.jobs:
            future.cancel()


@dataclass
class Request:
    lora_id: str = "a"
    lora_name: str = "A"
    lora_path: str = "/new"
    version: str = "v2"
    expected_version: str | None = None
    cache_epoch: str = "epoch2"

    def to_ref(self) -> SimpleNamespace:
        return SimpleNamespace(
            lora_id=self.lora_id, lora_name=self.lora_name, lora_path=self.lora_path
        )


class Req:
    def __init__(self, uid: str) -> None:
        self.lora_id = uid
        self.output_ids = [1, 2]
        self.lora_weight_version = None
        self.cache_epoch = "original"
        self.kv = torch.ones(2)
        self.error = None

    def set_lora_cache_epoch(self, epoch: str) -> None:
        self.cache_epoch = epoch


class Manager:
    def __init__(self) -> None:
        self.memory_pool = SimpleNamespace(uid_to_buffer_id={"a": 0, "b": 1})
        self.slots = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
        self.loras = {
            uid: SimpleNamespace(weights=self.slots[index].clone())
            for uid, index in self.memory_pool.uid_to_buffer_id.items()
        }
        self.configs = {"a": object(), "b": object()}
        self.lora_refs = {"a": Request(lora_path="/old").to_ref()}
        self.lora_weight_versions: dict[str, str] = {}
        self.lora_cache_epochs: dict[str, str] = {}
        self.updating_loras: set[str] = set()
        self.prepare_error = False
        self.upload_error = False
        self.rollback_error = False
        self.validation_error = False

    def prepare_lora_adapter(
        self, ref: SimpleNamespace
    ) -> tuple[object, SimpleNamespace]:
        if self.prepare_error:
            raise OSError("adapter read failed")
        return object(), SimpleNamespace(weights=torch.tensor([2.0, 2.0]))

    def validate_lora_replacement(self, ref: SimpleNamespace, prepared: tuple) -> None:
        if self.validation_error:
            raise ValueError("replacement shape differs")

    def copy_lora_replacement(self, uid: str, adapter: SimpleNamespace) -> None:
        slot = self.memory_pool.uid_to_buffer_id[uid]
        self.slots[slot, 0] = adapter.weights[0]
        if (self.upload_error and adapter.weights[0] == 2) or (
            self.rollback_error and adapter.weights[0] == 1
        ):
            raise RuntimeError("copy interrupted")
        self.slots[slot].copy_(adapter.weights)

    def _notify_lora_slots_updated(self, slots: set[int]) -> None:
        pass

    def create_lora_update_result(self, success: bool, message: str) -> SimpleNamespace:
        return SimpleNamespace(success=success, error_message=message)


class Scheduler:
    def __init__(self) -> None:
        self.manager = Manager()
        self.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(lora_manager=self.manager)
        )
        self.a, self.b = Req("a"), Req("b")
        self.waiting_queue = [self.a, self.b]
        self.replies: list[SimpleNamespace] = []

    def send_control_output(self, output: SimpleNamespace, request: Request) -> None:
        self.replies.append(output)

    def quiesce_lora_requests(self, uid: str) -> None:
        for req in self.waiting_queue:
            if req.lora_id == uid:
                req.kv = None

    def fail_lora_request(self, req: Req, message: str) -> None:
        req.error = message


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> tuple[Scheduler, LoRAUpdater]:
    monkeypatch.setattr(lora_updater, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    scheduler = Scheduler()
    updater = LoRAUpdater(scheduler)
    yield scheduler, updater
    updater.close()


def test_replacement_preserves_other_adapter_and_reuses_slot(runtime: tuple) -> None:
    scheduler, updater = runtime
    manager = scheduler.manager
    b_kv = scheduler.b.kv
    b_slot = manager.slots[1].data_ptr()
    updater.submit(Request())
    assert not updater.is_paused("a")
    torch.testing.assert_close(manager.slots[0], torch.ones(2))
    updater.executor.run()
    updater.poll()
    assert updater.is_paused("a") and not updater.is_paused("b")
    assert scheduler.a.kv is None
    assert manager.updating_loras == {"a"}
    for _ in range(4):
        scheduler.b.output_ids.append(int(manager.slots[1].sum()))
        updater.poll()
    assert not scheduler.replies
    assert scheduler.b.output_ids == [1, 2, 6, 6, 6, 6]
    updater.executor.run()
    updater.poll()
    assert scheduler.replies[-1].success
    assert not updater.is_paused("a") and not manager.updating_loras
    assert scheduler.a.output_ids == [1, 2]
    assert scheduler.a.lora_weight_version == "v2"
    assert scheduler.a.cache_epoch == "epoch2"
    assert scheduler.b.cache_epoch == "original"
    assert scheduler.b.kv is b_kv
    assert manager.slots[1].data_ptr() == b_slot
    torch.testing.assert_close(manager.slots, torch.tensor([[2.0, 2.0], [3.0, 3.0]]))
    assert manager.memory_pool.uid_to_buffer_id == {"a": 0, "b": 1}
    assert manager.lora_refs["a"].lora_id == "a"


@pytest.mark.parametrize("failure", ["prepare_error", "validation_error"])
def test_invalid_preparation_does_not_pause_or_invalidate_requests(
    runtime: tuple, failure: str
) -> None:
    scheduler, updater = runtime
    kv = scheduler.a.kv
    setattr(scheduler.manager, failure, True)
    updater.submit(Request())
    updater.executor.run()
    updater.poll()
    assert not scheduler.replies[-1].success
    assert not updater.paused
    assert scheduler.a.kv is kv
    assert scheduler.a.cache_epoch == "original"
    torch.testing.assert_close(scheduler.manager.slots[0], torch.ones(2))


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_partial_upload_rolls_back_or_blocks_only_the_failed_adapter(
    runtime: tuple, rollback_fails: bool
) -> None:
    scheduler, updater = runtime
    manager = scheduler.manager
    manager.upload_error = True
    manager.rollback_error = rollback_fails
    updater.submit(Request())
    updater.executor.run()
    updater.poll()
    updater.executor.run()
    updater.poll()
    assert updater.is_paused("a")
    updater.executor.run()
    updater.poll()
    assert not scheduler.replies[-1].success
    assert not updater.is_paused("b")
    torch.testing.assert_close(manager.slots[1], torch.full((2,), 3.0))
    assert manager.lora_refs["a"].lora_path == "/old"
    if rollback_fails:
        assert updater.is_paused("a")
        assert scheduler.a.error
        new = Req("a")
        assert updater.reject_failed(new) and new.error
        manager.upload_error = manager.rollback_error = False
        updater.submit(Request(version="v3", cache_epoch="epoch3"))
        updater.executor.run()
        updater.poll()
        updater.executor.run()
        updater.poll()
        assert scheduler.replies[-1].success
        assert not updater.is_paused("a") and not updater.failed
    else:
        assert not updater.is_paused("a")
        torch.testing.assert_close(manager.slots[0], torch.ones(2))


def test_versions_reject_stale_updates_and_make_retries_idempotent(
    runtime: tuple,
) -> None:
    scheduler, updater = runtime
    updater.submit(Request(expected_version="missing"))
    assert not scheduler.replies[-1].success and updater.pending is None
    updater.submit(Request())
    updater.executor.run()
    updater.poll()
    updater.executor.run()
    updater.poll()
    updater.submit(Request(cache_epoch="must-not-change"))
    assert scheduler.replies[-1].success
    assert updater.pending is None
    assert scheduler.manager.lora_cache_epochs["a"] == "epoch2"
    updater.submit(Request(lora_path="/different"))
    assert not scheduler.replies[-1].success


def test_shutdown_finishes_pending_rpc_and_releases_executor(runtime: tuple) -> None:
    scheduler, updater = runtime
    updater.submit(Request())
    updater.close()
    assert updater.pending is None
    assert updater.executor.closed
    assert not scheduler.replies[-1].success


def test_preparation_captures_runtime_context(runtime: tuple) -> None:
    scheduler, updater = runtime
    context = ContextVar("lora-test-context", default="missing")
    original = scheduler.manager.prepare_lora_adapter

    def prepare(ref: SimpleNamespace) -> tuple:
        assert context.get() == "scheduler-context"
        return original(ref)

    scheduler.manager.prepare_lora_adapter = prepare
    token = context.set("scheduler-context")
    try:
        updater.submit(Request())
    finally:
        context.reset(token)
    updater.executor.run()
    updater.poll()
    assert updater.is_paused("a")
