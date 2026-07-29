import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _manager():
    manager = SchedulerWeightUpdaterManager(
        tp_worker=mock.Mock(),
        draft_worker=None,
        tp_cpu_group=object(),
        weight_update_stage_cpu_group=object(),
        host_cpu_group=object(),
        boot_model_path="/base",
        memory_saver_adapter=mock.Mock(),
        flush_cache=mock.Mock(return_value=True),
        is_fully_idle=mock.Mock(return_value=True),
    )
    manager.tp_worker.model_runner.cpu_weight_cache = None
    manager.tp_worker.model_runner.server_args = SimpleNamespace(
        enable_cpu_weight_cache=False,
        cpu_weight_cache_canonical_checkpoint_dir=None,
    )
    return manager


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cpu_offload_gb", 1, "CPU and layer offloading"),
        ("offload_group_size", 1, "CPU and layer offloading"),
        ("cpu_weight_cache_max_compile_group_gb", 0, "must be positive"),
        ("pp_size", 2, "pipeline parallelism"),
        ("enable_lora", True, "dynamic LoRA"),
        ("lora_paths", ["/adapter"], "dynamic LoRA"),
        ("elastic_ep_backend", "nixl", "elastic expert"),
        ("enable_elastic_expert_backup", True, "elastic expert"),
    ],
)
def test_cpu_weight_cache_rejects_incompatible_launch_modes(
    field,
    value,
    message,
):
    args = ServerArgs(model_path="dummy")
    args.enable_cpu_weight_cache = True
    setattr(args, field, value)

    with pytest.raises(ValueError, match=message):
        args._validate_cpu_weight_cache_compatibility()


def test_cpu_weight_cache_rejects_automatic_eplb():
    args = ServerArgs(model_path="dummy")
    args.enable_cpu_weight_cache = True
    args.enable_eplb = True

    with pytest.raises(ValueError, match="automatic EPLB"):
        args._validate_cpu_weight_cache_compatibility()


def test_cpu_weight_cache_supports_speculative_decoding():
    args = ServerArgs(model_path="dummy")
    args.enable_cpu_weight_cache = True
    args.speculative_algorithm = "EAGLE"

    args._validate_cpu_weight_cache_compatibility()


def test_cpu_weight_cache_canonical_checkpoint_dir_requires_cache():
    args = ServerArgs(model_path="dummy")
    args.cpu_weight_cache_canonical_checkpoint_dir = "/local-checkpoint"

    with pytest.raises(ValueError, match="requires --enable-cpu-weight-cache"):
        args._validate_cpu_weight_cache_compatibility()


def test_cpu_weight_cache_canonical_checkpoint_dir_must_not_be_empty():
    args = ServerArgs(model_path="dummy")
    args.enable_cpu_weight_cache = True
    args.cpu_weight_cache_canonical_checkpoint_dir = ""

    with pytest.raises(ValueError, match="must not be empty"):
        args._validate_cpu_weight_cache_compatibility()


@pytest.mark.parametrize(
    ("method_name", "worker_method"),
    [
        ("update_weights_from_disk", "update_weights_from_disk"),
        ("update_weights_from_distributed", "update_weights_from_distributed"),
        ("update_weights_from_tensor", "update_weights_from_tensor"),
        ("update_weights_from_ipc", "update_weights_from_ipc"),
    ],
)
def test_background_weight_stage_blocks_other_weight_mutations(
    method_name, worker_method
):
    manager = _manager()
    manager._pending_weight_update_stage = (mock.Mock(), mock.Mock())

    result = getattr(manager, method_name)(SimpleNamespace())

    assert not result.success
    assert "background weight update stage" in result.message
    getattr(manager.tp_worker, worker_method).assert_not_called()


@pytest.mark.parametrize(
    "method_name",
    ["release_memory_occupation", "resume_memory_occupation"],
)
def test_background_weight_stage_blocks_weight_memory_transitions(method_name):
    manager = _manager()
    manager._pending_weight_update_stage = (mock.Mock(), mock.Mock())

    with pytest.raises(RuntimeError, match="background weight update stage"):
        getattr(manager, method_name)(SimpleNamespace())


def test_background_weight_stage_guard_is_collectively_consistent():
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


def test_second_weight_stage_uses_collective_pending_guard():
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
        result = manager.stage_weight_update(SimpleNamespace(destination="disk"))

    assert not result.success
    assert manager._pending_weight_update_stage is None


def test_cpu_staging_rejects_offloaded_live_weights():
    manager = _manager()
    manager.tp_worker.model_runner.server_args.enable_cpu_weight_cache = True
    manager.offload_tags.add("weights")

    result = manager.stage_weight_update(
        SimpleNamespace(
            destination="cpu",
            checkpoint_source_dir="/source",
            target_version=1,
        )
    )

    assert not result.success
    assert "resident on the GPU" in result.message


def test_cpu_staging_requires_enabled_cache():
    manager = _manager()

    result = manager.stage_weight_update(
        SimpleNamespace(
            destination="cpu",
            checkpoint_source_dir="/source",
            target_version=1,
        )
    )

    assert not result.success
    assert "--enable-cpu-weight-cache" in result.message
    assert manager._pending_weight_update_stage is None


def test_disk_staging_is_rejected_in_cpu_cache_mode():
    manager = _manager()
    manager.tp_worker.model_runner.server_args.enable_cpu_weight_cache = True

    request = SimpleNamespace(
        destination="disk",
        checkpoint_source_dir=None,
        target_version=0,
    )
    message = manager._stage_weight_update_preflight_message(request)
    assert "Disk weight update staging is unavailable" in message


def test_weight_stage_requires_source_after_base_version():
    manager = _manager()

    request = SimpleNamespace(
        destination="disk",
        checkpoint_source_dir=None,
        target_version=1,
    )

    assert (
        manager._stage_weight_update_preflight_message(request)
        == "checkpoint_source_dir is required for targets after version 0."
    )


def test_disk_staging_defaults_to_immutable_boot_model():
    manager = _manager()
    manager.tp_worker.model_runner.server_args = SimpleNamespace(
        enable_cpu_weight_cache=False,
        model_path="/latest/local/checkpoint",
        checkpoint_source_refresh_hook=None,
        cpu_weight_cache_canonical_checkpoint_dir=None,
    )
    request = SimpleNamespace(
        base_checkpoint_dir=None,
        destination="disk",
        local_checkpoint_dir="/local/checkpoint",
        checkpoint_source_dir="/published",
        target_version=1,
    )

    with mock.patch(
        "sglang.srt.weight_sync.disk_checkpoint.materialize",
        return_value={},
    ) as materialize:
        result = manager._stage_weight_update_sync(request)

    assert result.success
    materialize.assert_called_once_with(
        local_checkpoint_dir="/local/checkpoint",
        base_checkpoint_dir="/base",
        checkpoint_source_dir="/published",
        target_version=1,
        checkpoint_source_refresh_hook=None,
    )


def test_cpu_base_uses_cache_initialization():
    manager = _manager()
    manager.tp_worker.model_runner.server_args = SimpleNamespace(
        enable_cpu_weight_cache=True,
        model_path="/base",
        checkpoint_source_refresh_hook=None,
        cpu_weight_cache_canonical_checkpoint_dir=None,
    )
    manager.tp_worker.initialize_cpu_weight_cache.return_value = {
        "operation": "initialize_cpu_weight_cache"
    }

    request = SimpleNamespace(
        base_checkpoint_dir=None,
        destination="cpu",
        local_checkpoint_dir=None,
        checkpoint_source_dir=None,
        target_version=0,
    )
    result = manager._stage_weight_update_sync(request)

    assert result.success
    assert (
        result.rank_stats[0]["stage"]["initialization"]["operation"]
        == "initialize_cpu_weight_cache"
    )
    manager.tp_worker.initialize_cpu_weight_cache.assert_called_once_with(
        manager.host_cpu_group,
        base_checkpoint_dir="/base",
        seed_from_active_weights=True,
    )


@pytest.mark.parametrize("drop_cache", [False, True])
def test_cpu_base_materializes_disk_backed_canonical_checkpoint(drop_cache):
    manager = _manager()
    manager.tp_worker.model_runner.server_args = SimpleNamespace(
        enable_cpu_weight_cache=True,
        model_path="/base",
        checkpoint_source_refresh_hook=None,
        cpu_weight_cache_canonical_checkpoint_dir="/canonical",
        weight_loader_drop_cache_after_load=drop_cache,
    )
    manager.tp_worker.initialize_cpu_weight_cache.return_value = {
        "operation": "initialize_cpu_weight_cache"
    }
    request = SimpleNamespace(
        base_checkpoint_dir=None,
        destination="cpu",
        local_checkpoint_dir=None,
        checkpoint_source_dir=None,
        target_version=0,
    )

    with (
        mock.patch(
            "sglang.srt.weight_sync.disk_checkpoint.materialize",
            return_value={"operation": "materialize"},
        ) as materialize,
        mock.patch(
            "sglang.srt.weight_sync.disk_checkpoint.drop_checkpoint_page_cache"
        ) as drop_page_cache,
    ):
        result = manager._stage_weight_update_sync(request)

    assert result.success
    assert (
        result.rank_stats[0]["stage"]["canonical_checkpoint_materialization"][
            "operation"
        ]
        == "materialize"
    )
    materialize.assert_called_once_with(
        local_checkpoint_dir="/canonical",
        base_checkpoint_dir="/base",
        checkpoint_source_dir="/base",
        target_version=0,
    )
    manager.tp_worker.initialize_cpu_weight_cache.assert_called_once_with(
        manager.host_cpu_group,
        base_checkpoint_dir="/base",
        seed_from_active_weights=True,
    )
    if drop_cache:
        drop_page_cache.assert_called_once_with("/canonical")
    else:
        drop_page_cache.assert_not_called()


def test_cpu_delta_materializes_then_compiles_disk_backed_canonical_checkpoint():
    manager = _manager()
    manager.tp_worker.model_runner.server_args = SimpleNamespace(
        enable_cpu_weight_cache=True,
        model_path="/base",
        checkpoint_source_refresh_hook="package.refresh",
        cpu_weight_cache_canonical_checkpoint_dir="/canonical",
        weight_loader_drop_cache_after_load=True,
    )
    manager._cpu_weight_cache_base_checkpoint_dir = "/base"
    manager._cpu_weight_cache_initialization_stats = {}
    manager.tp_worker.stage_cpu_weight_update_from_checkpoint.return_value = (
        True,
        "staged",
        {},
    )
    request = SimpleNamespace(
        base_checkpoint_dir=None,
        destination="cpu",
        local_checkpoint_dir=None,
        checkpoint_source_dir="/published",
        target_version=3,
    )
    refreshed = False

    def refresh(*_args):
        nonlocal refreshed
        refreshed = True

    def validate_after_refresh(**_kwargs):
        assert refreshed

    with (
        mock.patch(
            "sglang.srt.weight_sync.disk_checkpoint.refresh_checkpoint_source",
            side_effect=refresh,
        ) as refresh_source,
        mock.patch(
            "sglang.srt.weight_sync.cpu_delta_checkpoint.validate_delta_target",
            side_effect=validate_after_refresh,
        ) as validate,
        mock.patch(
            "sglang.srt.weight_sync.disk_checkpoint.materialize",
            return_value={"operation": "materialize", "target_version": 3},
        ) as materialize,
        mock.patch(
            "sglang.srt.weight_sync.disk_checkpoint.drop_checkpoint_page_cache"
        ) as drop_page_cache,
    ):
        result = manager._stage_weight_update_sync(request)

    assert result.success
    validate.assert_called_once_with(
        checkpoint_source_dir="/published",
        target_version=3,
    )
    refresh_source.assert_called_once_with("/published", 3, "package.refresh")
    materialize.assert_called_once_with(
        local_checkpoint_dir="/canonical",
        base_checkpoint_dir="/base",
        checkpoint_source_dir="/published",
        target_version=3,
    )
    manager.tp_worker.stage_cpu_weight_update_from_checkpoint.assert_called_once_with(
        checkpoint_dir="/canonical",
        target_version=3,
        host_cpu_group=manager.host_cpu_group,
    )
    manager.tp_worker.stage_cpu_weight_update_from_delta_lineage.assert_not_called()
    drop_page_cache.assert_not_called()
    assert (
        result.rank_stats[0]["stage"]["canonical_checkpoint_materialization"][
            "target_version"
        ]
        == 3
    )


def test_cpu_weight_cache_initialization_runs_in_background():
    manager = _manager()
    manager.tp_worker.model_runner.server_args = SimpleNamespace(
        enable_cpu_weight_cache=True,
        model_path="/base",
        checkpoint_source_refresh_hook=None,
        cpu_weight_cache_canonical_checkpoint_dir=None,
    )
    started = threading.Event()
    release = threading.Event()

    def initialize(
        host_cpu_group,
        *,
        base_checkpoint_dir,
        seed_from_active_weights,
    ):
        assert host_cpu_group is manager.host_cpu_group
        assert base_checkpoint_dir == "/base"
        assert seed_from_active_weights
        started.set()
        assert release.wait(timeout=5)
        return {"operation": "initialize_cpu_weight_cache"}

    manager.tp_worker.initialize_cpu_weight_cache.side_effect = initialize
    request = SimpleNamespace(
        base_checkpoint_dir=None,
        destination="cpu",
        local_checkpoint_dir=None,
        checkpoint_source_dir=None,
        target_version=0,
    )
    result = manager.stage_weight_update(request)
    assert result is None
    assert started.wait(timeout=5)
    assert manager._pending_weight_update_stage[1].is_alive()

    release.set()
    manager._pending_weight_update_stage[1].join()
    result = manager._weight_update_stage_result

    assert result.success
    assert (
        result.rank_stats[0]["stage"]["initialization"]["operation"]
        == "initialize_cpu_weight_cache"
    )


def test_cpu_weight_cache_initialization_can_retry_after_cleanup():
    manager = _manager()
    manager.tp_worker.model_runner.server_args = SimpleNamespace(
        enable_cpu_weight_cache=True,
        model_path="/base",
        checkpoint_source_refresh_hook=None,
        cpu_weight_cache_canonical_checkpoint_dir=None,
    )
    manager.tp_worker.initialize_cpu_weight_cache.side_effect = [
        RuntimeError("broken"),
        {"operation": "initialize_cpu_weight_cache"},
    ]

    request = SimpleNamespace(
        base_checkpoint_dir=None,
        destination="cpu",
        local_checkpoint_dir=None,
        checkpoint_source_dir=None,
        target_version=0,
    )
    first = manager._stage_weight_update_sync(request)
    second = manager._stage_weight_update_sync(request)

    assert not first.success
    assert "broken" in first.message
    assert second.success
    assert second.rank_stats[0]["stage"]["initialized"]
    assert manager.tp_worker.initialize_cpu_weight_cache.call_count == 2


def test_cpu_weight_cache_initialization_requires_enabled_cache():
    manager = _manager()

    result = manager.stage_weight_update(
        SimpleNamespace(
            destination="cpu",
            checkpoint_source_dir=None,
            target_version=0,
        )
    )

    assert not result.success
    assert "--enable-cpu-weight-cache" in result.message
    manager.tp_worker.initialize_cpu_weight_cache.assert_not_called()


def test_cpu_weight_cache_rejects_a_different_checkpoint():
    manager = _manager()
    manager.tp_worker.model_runner.server_args = SimpleNamespace(
        enable_cpu_weight_cache=True,
        model_path="/base",
        checkpoint_source_refresh_hook=None,
        cpu_weight_cache_canonical_checkpoint_dir=None,
    )
    manager.tp_worker.initialize_cpu_weight_cache.return_value = {}

    first = manager._stage_weight_update_sync(
        SimpleNamespace(
            base_checkpoint_dir="/base",
            destination="cpu",
            local_checkpoint_dir=None,
            checkpoint_source_dir=None,
            target_version=0,
        )
    )
    second = manager._stage_weight_update_sync(
        SimpleNamespace(
            base_checkpoint_dir="/other",
            destination="cpu",
            local_checkpoint_dir=None,
            checkpoint_source_dir=None,
            target_version=0,
        )
    )

    assert first.success
    assert not second.success
    assert "different checkpoint" in second.message
    manager.tp_worker.initialize_cpu_weight_cache.assert_called_once()


def test_cpu_weight_cache_initialization_failure_is_collective_and_retryable():
    manager = _manager()
    manager.tp_worker.model_runner.server_args = SimpleNamespace(
        enable_cpu_weight_cache=True,
        model_path="/base",
        checkpoint_source_refresh_hook=None,
        cpu_weight_cache_canonical_checkpoint_dir=None,
    )
    manager.tp_worker.initialize_cpu_weight_cache.return_value = {"rank": 0}

    def all_gather_object(output, value, *, group):
        assert group is manager.weight_update_stage_cpu_group
        output[:] = [
            value,
            (False, "rank 1 failed", {"rank": 1}),
        ]

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
    ):
        first = manager._stage_weight_update_sync(
            SimpleNamespace(
                base_checkpoint_dir=None,
                destination="cpu",
                local_checkpoint_dir=None,
                checkpoint_source_dir=None,
                target_version=0,
            )
        )

    second = manager._stage_weight_update_sync(
        SimpleNamespace(
            base_checkpoint_dir=None,
            destination="cpu",
            local_checkpoint_dir=None,
            checkpoint_source_dir=None,
            target_version=0,
        )
    )

    assert not first.success
    assert "rank 1 failed" in first.message
    assert second.success
    assert manager.tp_worker.initialize_cpu_weight_cache.call_count == 2
    manager.tp_worker.discard_cpu_weight_cache.assert_called_once()


def test_cpu_commit_rejects_initializing_cache():
    manager = _manager()
    manager.tp_worker.model_runner.server_args = SimpleNamespace(
        enable_cpu_weight_cache=True,
        model_path="/base",
        checkpoint_source_refresh_hook=None,
        cpu_weight_cache_canonical_checkpoint_dir=None,
    )
    release = threading.Event()

    def initialize(_, *, base_checkpoint_dir):
        assert base_checkpoint_dir == "/base"
        assert release.wait(timeout=5)
        return {}

    manager.tp_worker.initialize_cpu_weight_cache.side_effect = initialize
    manager.stage_weight_update(
        SimpleNamespace(
            base_checkpoint_dir=None,
            destination="cpu",
            local_checkpoint_dir=None,
            checkpoint_source_dir=None,
            target_version=0,
        )
    )
    try:
        result = manager.update_weights_from_cpu(
            SimpleNamespace(target_version=1),
        )
    finally:
        release.set()
        manager._pending_weight_update_stage[1].join()

    assert not result.success
    assert "background weight update stage" in result.message
    manager.tp_worker.validate_staged_cpu_weight_update.assert_not_called()


def test_cpu_staging_preflight_is_collectively_consistent():
    manager = _manager()
    manager.tp_worker.model_runner.server_args.enable_cpu_weight_cache = True

    def all_gather_object(output, value, *, group):
        if isinstance(value, bool):
            output[:] = [value, value]
        else:
            output[:] = [
                value,
                "CPU weight update staging requires GPU-resident weights.",
            ]

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
        result = manager.stage_weight_update(
            SimpleNamespace(
                destination="cpu",
                checkpoint_source_dir="/source",
                target_version=1,
            )
        )

    assert not result.success
    assert "GPU-resident weights" in result.message
    assert manager._pending_weight_update_stage is None


def test_cpu_staging_failure_invalidates_every_rank():
    manager = _manager()
    manager.tp_worker.model_runner.server_args = SimpleNamespace(
        enable_cpu_weight_cache=True,
        model_path="/base",
        checkpoint_source_refresh_hook=None,
        cpu_weight_cache_canonical_checkpoint_dir=None,
    )
    manager._cpu_weight_cache_initialization_stats = {}
    manager.tp_worker.stage_cpu_weight_update_from_delta_lineage.return_value = (
        True,
        "staged",
        {},
    )

    def all_gather_object(output, value, *, group):
        if value is None:
            output[:] = [None, None]
        else:
            output[:] = [
                value,
                (False, "remote staging failed", {"rank": 1}),
            ]

    request = SimpleNamespace(
        base_checkpoint_dir=None,
        destination="cpu",
        local_checkpoint_dir=None,
        checkpoint_source_dir="/source",
        target_version=1,
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
    ):
        result = manager._stage_weight_update_sync(request)

    assert not result.success
    assert "remote staging failed" in result.message
    manager.tp_worker.invalidate_staged_cpu_weight_update.assert_called_once()


def test_cpu_staging_requires_initialized_cache():
    manager = _manager()
    manager.tp_worker.model_runner.server_args = SimpleNamespace(
        enable_cpu_weight_cache=True,
        model_path="/base",
        checkpoint_source_refresh_hook=None,
        cpu_weight_cache_canonical_checkpoint_dir=None,
    )
    request = SimpleNamespace(
        base_checkpoint_dir="/base",
        destination="cpu",
        local_checkpoint_dir=None,
        checkpoint_source_dir="/source",
        target_version=1,
    )

    result = manager._stage_weight_update_sync(request)

    assert not result.success
    assert "is not initialized" in result.message
    manager.tp_worker.stage_cpu_weight_update_from_delta_lineage.assert_not_called()
    manager.tp_worker.invalidate_staged_cpu_weight_update.assert_called_once()


def test_cpu_staging_requires_initialized_base_checkpoint():
    manager = _manager()
    manager.tp_worker.model_runner.server_args = SimpleNamespace(
        enable_cpu_weight_cache=True,
        model_path="/boot",
        checkpoint_source_refresh_hook=None,
        cpu_weight_cache_canonical_checkpoint_dir=None,
    )
    manager._cpu_weight_cache_base_checkpoint_dir = "/local/base"
    manager._cpu_weight_cache_initialization_stats = {}
    request = SimpleNamespace(
        base_checkpoint_dir="/other/base",
        destination="cpu",
        local_checkpoint_dir=None,
        checkpoint_source_dir="/source",
        target_version=1,
    )

    result = manager._stage_weight_update_sync(request)

    assert not result.success
    assert "does not match the initialized cache" in result.message
    manager.tp_worker.stage_cpu_weight_update_from_delta_lineage.assert_not_called()
    manager.tp_worker.invalidate_staged_cpu_weight_update.assert_called_once()


@pytest.mark.parametrize(
    ("method_name", "worker_method"),
    [
        ("update_weights_from_disk", "update_weights_from_disk"),
        ("update_weights_from_distributed", "update_weights_from_distributed"),
        ("update_weights_from_tensor", "update_weights_from_tensor"),
        ("update_weights_from_ipc", "update_weights_from_ipc"),
    ],
)
def test_cpu_weight_cache_rejects_other_live_weight_mutations(
    method_name,
    worker_method,
):
    manager = _manager()
    manager.tp_worker.model_runner.server_args.enable_cpu_weight_cache = True

    result = getattr(manager, method_name)(SimpleNamespace())

    assert not result.success
    assert "CPU weight cache is enabled" in result.message
    getattr(manager.tp_worker, worker_method).assert_not_called()


@pytest.mark.parametrize(
    "method_name",
    ["release_memory_occupation", "resume_memory_occupation"],
)
def test_cpu_weight_cache_rejects_weight_memory_transitions(method_name):
    manager = _manager()
    manager.tp_worker.model_runner.server_args.enable_cpu_weight_cache = True

    with pytest.raises(RuntimeError, match="CPU weight cache is enabled"):
        getattr(manager, method_name)(SimpleNamespace(tags=["weights"]))


def test_cpu_commit_failure_reaches_collective_before_failing_closed():
    manager = _manager()
    manager._cpu_weight_cache_initialization_stats = {}
    manager.tp_worker.validate_staged_cpu_weight_update.return_value = (True, "ready")
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

    assert collective_calls == 4
    manager.flush_cache.assert_called_once_with(empty_cache=False)


def test_cpu_commit_rejects_failed_cache_flush_before_mutating_weights():
    manager = _manager()
    manager._cpu_weight_cache_initialization_stats = {}
    manager.tp_worker.validate_staged_cpu_weight_update.return_value = (True, "ready")
    manager.flush_cache.return_value = False

    result = manager.update_weights_from_cpu(
        SimpleNamespace(
            target_version=1,
            flush_cache=True,
            torch_empty_cache=False,
        )
    )

    assert not result.success
    assert result.message == "Cache flush failed before CPU weight commit."
    manager.tp_worker.update_weights_from_cpu.assert_not_called()


def test_cpu_commit_flushes_cache_before_mutating_weights():
    manager = _manager()
    manager._cpu_weight_cache_initialization_stats = {}
    manager.tp_worker.validate_staged_cpu_weight_update.return_value = (True, "ready")
    order = []

    def flush_cache(*, empty_cache):
        order.append(("flush", empty_cache))
        return True

    def update_weights(_):
        order.append(("commit", None))
        return True, "committed", {}

    manager.flush_cache.side_effect = flush_cache
    manager.tp_worker.update_weights_from_cpu.side_effect = update_weights

    result = manager.update_weights_from_cpu(
        SimpleNamespace(
            target_version=1,
            flush_cache=True,
            torch_empty_cache=False,
        )
    )

    assert result.success
    assert order == [("flush", False), ("commit", None)]


def test_cpu_commit_updates_only_target_model():
    manager = _manager()
    manager.draft_worker = mock.Mock()
    manager._cpu_weight_cache_initialization_stats = {}
    manager.tp_worker.validate_staged_cpu_weight_update.return_value = (True, "ready")
    manager.tp_worker.update_weights_from_cpu.return_value = (
        True,
        "committed",
        {},
    )

    result = manager.update_weights_from_cpu(
        SimpleNamespace(
            target_version=1,
            flush_cache=False,
            torch_empty_cache=False,
        )
    )

    assert result.success
    manager.tp_worker.update_weights_from_cpu.assert_called_once()
    manager.draft_worker.update_weights_from_cpu.assert_not_called()
