from types import SimpleNamespace
from unittest import mock

import pytest

from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _manager():
    return SchedulerWeightUpdaterManager(
        tp_worker=mock.Mock(),
        draft_worker=None,
        tp_cpu_group=object(),
        background_cpu_group=object(),
        host_cpu_group=object(),
        memory_saver_adapter=mock.Mock(),
        flush_cache=mock.Mock(return_value=True),
        is_fully_idle=mock.Mock(return_value=True),
    )


@pytest.mark.parametrize(
    ("method_name", "worker_method"),
    [
        ("update_weights_from_disk", "update_weights_from_disk"),
        ("update_weights_from_distributed", "update_weights_from_distributed"),
        ("update_weights_from_tensor", "update_weights_from_tensor"),
        ("update_weights_from_ipc", "update_weights_from_ipc"),
    ],
)
def test_background_pull_blocks_other_weight_mutations(method_name, worker_method):
    manager = _manager()
    manager._pending_pull = (mock.Mock(), mock.Mock())

    result = getattr(manager, method_name)(SimpleNamespace())

    assert not result.success
    assert "background weight pull" in result.message
    getattr(manager.tp_worker, worker_method).assert_not_called()


@pytest.mark.parametrize(
    "method_name",
    ["release_memory_occupation", "resume_memory_occupation"],
)
def test_background_pull_blocks_weight_memory_transitions(method_name):
    manager = _manager()
    manager._pending_pull = (mock.Mock(), mock.Mock())

    with pytest.raises(RuntimeError, match="background weight pull"):
        getattr(manager, method_name)(SimpleNamespace())


def test_background_pull_guard_is_collectively_consistent():
    manager = _manager()

    def all_gather_object(output, value, *, group):
        output[:] = [value, True]

    with (
        mock.patch(
            "sglang.srt.managers.scheduler_components.weight_updater."
            "torch.distributed.is_initialized",
            return_value=True,
        ),
        mock.patch(
            "sglang.srt.managers.scheduler_components.weight_updater."
            "torch.distributed.get_world_size",
            return_value=2,
        ),
        mock.patch(
            "sglang.srt.managers.scheduler_components.weight_updater."
            "torch.distributed.all_gather_object",
            side_effect=all_gather_object,
        ),
    ):
        result = manager.update_weights_from_disk(SimpleNamespace())

    assert not result.success
    manager.tp_worker.update_weights_from_disk.assert_not_called()


def test_second_pull_uses_collective_pending_guard():
    manager = _manager()

    def all_gather_object(output, value, *, group):
        output[:] = [value, True]

    with (
        mock.patch(
            "sglang.srt.managers.scheduler_components.weight_updater."
            "torch.distributed.is_initialized",
            return_value=True,
        ),
        mock.patch(
            "sglang.srt.managers.scheduler_components.weight_updater."
            "torch.distributed.get_world_size",
            return_value=2,
        ),
        mock.patch(
            "sglang.srt.managers.scheduler_components.weight_updater."
            "torch.distributed.all_gather_object",
            side_effect=all_gather_object,
        ),
    ):
        result = manager.pull_weights(SimpleNamespace(destination="disk"))

    assert not result.success
    assert manager._pending_pull is None


def test_cpu_preparation_rejects_offloaded_live_weights():
    manager = _manager()
    manager.offload_tags.add("weights")

    result = manager.pull_weights(SimpleNamespace(destination="cpu"))

    assert not result.success
    assert "resident on the GPU" in result.message


def test_cpu_commit_failure_reaches_collective_before_failing_closed():
    manager = _manager()
    manager.tp_worker.validate_cpu_weight_update.return_value = (True, "ready")
    manager.tp_worker.update_weights_from_cpu.side_effect = RuntimeError(
        "commit failed"
    )

    collective_calls = 0

    def all_gather_object(output, value, *, group):
        nonlocal collective_calls
        collective_calls += 1
        if isinstance(value, bool):
            output[:] = [value, value]
        elif len(value) == 2:
            output[:] = [value, value]
        else:
            output[:] = [value, (True, "committed", {"rank": 1})]

    request = SimpleNamespace(
        target_version=1,
        flush_cache=True,
        torch_empty_cache=False,
    )
    with (
        mock.patch(
            "sglang.srt.managers.scheduler_components.weight_updater."
            "torch.distributed.is_initialized",
            return_value=True,
        ),
        mock.patch(
            "sglang.srt.managers.scheduler_components.weight_updater."
            "torch.distributed.get_world_size",
            return_value=2,
        ),
        mock.patch(
            "sglang.srt.managers.scheduler_components.weight_updater."
            "torch.distributed.get_rank",
            return_value=0,
        ),
        mock.patch(
            "sglang.srt.managers.scheduler_components.weight_updater."
            "torch.distributed.all_gather_object",
            side_effect=all_gather_object,
        ),
        pytest.raises(RuntimeError, match="terminating the engine"),
    ):
        manager.update_weights_from_cpu(request)

    assert collective_calls == 3
    manager.flush_cache.assert_not_called()
