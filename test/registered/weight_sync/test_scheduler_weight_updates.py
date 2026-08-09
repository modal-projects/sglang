from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _manager(*, cpu_cache=False):
    tp_worker = mock.Mock()
    tp_worker.model_runner.cpu_weight_cache = None
    tp_worker.model_runner.server_args = SimpleNamespace(
        checkpoint_source_refresh_hook=None,
        cpu_weight_cache_canonical_checkpoint_dir=None,
        cpu_weight_cache_max_compile_group_gb=8.0,
        enable_cpu_weight_cache=cpu_cache,
        weight_cache_mode="off",
    )
    return SchedulerWeightUpdaterManager(
        tp_worker=tp_worker,
        draft_worker=None,
        tp_cpu_group=object(),
        weight_update_stage_cpu_group=object(),
        host_cpu_group=object(),
        boot_model_path="/base",
        memory_saver_adapter=mock.Mock(),
        flush_cache=mock.Mock(return_value=True),
        is_fully_idle=mock.Mock(return_value=True),
        send_control_output=mock.Mock(),
    )


def _stage_request(
    *,
    destination,
    target_version,
    base_checkpoint_dir=None,
    checkpoint_source_dir=None,
    local_checkpoint_dir=None,
):
    return SimpleNamespace(
        destination=destination,
        target_version=target_version,
        base_checkpoint_dir=base_checkpoint_dir,
        checkpoint_source_dir=checkpoint_source_dir,
        local_checkpoint_dir=local_checkpoint_dir,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cpu_weight_cache_max_compile_group_gb", 0, "must be positive"),
        ("weight_cache_mode", "daemon", "weight-cache-mode"),
        ("cpu_offload_gb", 1, "resident on the GPU"),
        ("offload_group_size", 1, "resident on the GPU"),
        ("pp_size", 2, "pipeline parallelism"),
        ("dcp_replicate_q_proj", True, "dcp-replicate-q-proj"),
        ("enable_eplb", True, "automatic EPLB"),
        ("enable_lora", True, "dynamic LoRA"),
        ("lora_paths", ["/adapter"], "dynamic LoRA"),
        ("elastic_ep_backend", "nixl", "elastic expert"),
        ("enable_elastic_expert_backup", True, "elastic expert"),
    ],
)
def test_cpu_weight_cache_rejects_incompatible_server_args(field, value, message):
    server_args = ServerArgs(model_path="dummy")
    server_args.enable_cpu_weight_cache = True
    setattr(server_args, field, value)

    with (
        mock.patch("sglang.srt.server_args.is_cuda", return_value=True),
        pytest.raises(ValueError, match=message),
    ):
        server_args._handle_cpu_weight_cache()


def test_cpu_weight_cache_keeps_speculative_draft_weights_fixed():
    server_args = ServerArgs(model_path="dummy")
    server_args.enable_cpu_weight_cache = True
    server_args.speculative_algorithm = "EAGLE"

    with mock.patch("sglang.srt.server_args.is_cuda", return_value=True):
        server_args._handle_cpu_weight_cache()


def test_disk_canonical_checkpoint_requires_cpu_weight_cache():
    server_args = ServerArgs(model_path="dummy")
    server_args.cpu_weight_cache_canonical_checkpoint_dir = "/local/checkpoint"

    with pytest.raises(ValueError, match="requires --enable-cpu-weight-cache"):
        server_args._handle_cpu_weight_cache()


def test_disk_canonical_checkpoint_rejects_empty_path():
    server_args = ServerArgs(model_path="dummy")
    server_args.enable_cpu_weight_cache = True
    server_args.cpu_weight_cache_canonical_checkpoint_dir = ""

    with (
        mock.patch("sglang.srt.server_args.is_cuda", return_value=True),
        pytest.raises(ValueError, match="must not be empty"),
    ):
        server_args._handle_cpu_weight_cache()


def test_cpu_cache_always_creates_a_dedicated_host_group():
    scheduler = SimpleNamespace(
        tp_cpu_group=object(),
        tp_group=SimpleNamespace(ranks=[4, 5]),
        server_args=SimpleNamespace(
            enable_cpu_weight_cache=True,
            model_path="/base",
        ),
        tp_worker=mock.Mock(),
        draft_worker=None,
        memory_saver_adapter=mock.Mock(),
        flush_cache=mock.Mock(),
        is_fully_idle=mock.Mock(),
        send_control_output=mock.Mock(),
        metrics_collector=None,
    )

    def gather_hosts(output, _hostname, *, group):
        assert group is scheduler.tp_cpu_group
        output[:] = ["host-a", "host-a"]

    with (
        mock.patch(
            "sglang.srt.managers.scheduler.torch.distributed.get_world_size",
            return_value=2,
        ),
        mock.patch(
            "sglang.srt.managers.scheduler.torch.distributed.all_gather_object",
            side_effect=gather_hosts,
        ),
        mock.patch(
            "sglang.srt.managers.scheduler.socket.gethostname",
            return_value="host-a",
        ),
        mock.patch(
            "sglang.srt.managers.scheduler.create_custom_parallel_group",
            side_effect=["stage-group", "host-group"],
        ) as create_group,
        mock.patch(
            "sglang.srt.managers.scheduler.SchedulerWeightUpdaterManager"
        ) as manager_type,
    ):
        Scheduler.init_weight_updater(scheduler)

    assert create_group.call_args_list == [
        mock.call(group_ranks=[4, 5]),
        mock.call(group_ranks=[4, 5]),
    ]
    assert manager_type.call_args.kwargs["weight_update_stage_cpu_group"] == (
        "stage-group"
    )
    assert manager_type.call_args.kwargs["host_cpu_group"] == "host-group"


def test_disk_stage_defaults_to_boot_checkpoint():
    manager = _manager()
    request = _stage_request(
        destination="disk",
        target_version=1,
        checkpoint_source_dir="/updates",
        local_checkpoint_dir="/local",
    )

    with mock.patch(
        "sglang.srt.weight_sync.disk_checkpoint.materialize",
        return_value={},
    ) as materialize:
        result = manager._stage_weight_update_sync(request)

    assert result.success
    materialize.assert_called_once_with(
        local_checkpoint_dir="/local",
        base_checkpoint_dir="/base",
        checkpoint_source_dir="/updates",
        target_version=1,
        checkpoint_source_refresh_hook=None,
    )


def test_stage_response_reports_only_the_local_scheduler_stats():
    manager = _manager()
    request = _stage_request(
        destination="disk",
        target_version=1,
        checkpoint_source_dir="/updates",
        local_checkpoint_dir="/local",
    )

    def gather_with_remote_success(value, group):
        assert group is manager.weight_update_stage_cpu_group
        return [value, (True, "Success.", {"rank": 1, "wall_s": 2.0})]

    with (
        mock.patch(
            "sglang.srt.weight_sync.disk_checkpoint.materialize",
            return_value={},
        ),
        mock.patch.object(
            SchedulerWeightUpdaterManager,
            "_all_gather",
            autospec=True,
            side_effect=gather_with_remote_success,
        ),
    ):
        result = manager._stage_weight_update_sync(request)

    assert result.success
    assert len(result.rank_stats) == 1
    assert result.rank_stats[0]["rank"] == 0


def test_cpu_stage_zero_initializes_rank_cache():
    manager = _manager(cpu_cache=True)
    manager.tp_worker.initialize_cpu_weight_cache.return_value = {"cache": "ready"}

    result = manager._stage_weight_update_sync(
        _stage_request(destination="cpu", target_version=0)
    )

    assert result.success
    manager.tp_worker.initialize_cpu_weight_cache.assert_called_once_with(
        checkpoint_dir="/base",
        seed_from_active_weights=True,
        host_group=manager.host_cpu_group,
        max_compile_group_bytes=8 << 30,
        canonical_checkpoint_dir=None,
    )
    assert result.rank_stats[0]["stage"]["initialization"] == {"cache": "ready"}


def test_cpu_stage_zero_compiles_a_non_boot_checkpoint():
    manager = _manager(cpu_cache=True)
    manager.tp_worker.initialize_cpu_weight_cache.return_value = {"cache": "ready"}

    result = manager._stage_weight_update_sync(
        _stage_request(
            destination="cpu",
            target_version=0,
            base_checkpoint_dir="/other-base",
        )
    )

    assert result.success
    manager.tp_worker.initialize_cpu_weight_cache.assert_called_once_with(
        checkpoint_dir="/other-base",
        seed_from_active_weights=False,
        host_group=manager.host_cpu_group,
        max_compile_group_bytes=8 << 30,
        canonical_checkpoint_dir=None,
    )


def test_cpu_delta_stage_requires_initialized_base():
    manager = _manager(cpu_cache=True)

    result = manager._stage_weight_update_sync(
        _stage_request(
            destination="cpu",
            target_version=1,
            checkpoint_source_dir="/updates",
        )
    )

    assert not result.success
    assert "not initialized" in result.message
    manager.tp_worker.stage_cpu_weight_update_from_delta_lineage.assert_not_called()
    manager.tp_worker.invalidate_staged_cpu_weight_update.assert_called_once()


def test_cpu_delta_stage_refreshes_then_compiles():
    manager = _manager(cpu_cache=True)
    manager._cpu_weight_cache_checkpoint_dir = "/base"
    manager._cpu_weight_cache_initialization_stats = {}
    manager.tp_worker.stage_cpu_weight_update_from_delta_lineage.return_value = (
        True,
        "staged",
        {},
    )
    request = _stage_request(
        destination="cpu",
        target_version=1,
        checkpoint_source_dir="/updates",
    )

    with mock.patch.object(
        SchedulerWeightUpdaterManager,
        "_refresh_checkpoint_source",
        return_value=1.25,
    ):
        result = manager._stage_weight_update_sync(request)

    assert result.success
    assert result.rank_stats[0]["stage"]["source_refresh_wall_s"] == 1.25
    manager.tp_worker.stage_cpu_weight_update_from_delta_lineage.assert_called_once_with(
        checkpoint_source_dir="/updates",
        target_version=1,
        host_group=manager.host_cpu_group,
    )


def test_distributed_cpu_initialization_failure_discards_every_rank_and_retries():
    manager = _manager(cpu_cache=True)
    manager.tp_worker.initialize_cpu_weight_cache.return_value = {"rank": 0}
    request = _stage_request(destination="cpu", target_version=0)

    def gather_with_remote_failure(value, group):
        assert group is manager.weight_update_stage_cpu_group
        return [value, (False, "rank 1 failed", {"rank": 1})]

    with mock.patch.object(
        SchedulerWeightUpdaterManager,
        "_all_gather",
        autospec=True,
        side_effect=gather_with_remote_failure,
    ):
        first = manager._stage_weight_update_sync(request)
    second = manager._stage_weight_update_sync(request)

    assert not first.success
    assert "rank 1 failed" in first.message
    assert second.success
    assert manager.tp_worker.initialize_cpu_weight_cache.call_count == 2
    manager.tp_worker.discard_cpu_weight_cache.assert_called_once()


def test_distributed_cpu_delta_failure_invalidates_every_rank():
    manager = _manager(cpu_cache=True)
    manager._cpu_weight_cache_checkpoint_dir = "/base"
    manager._cpu_weight_cache_initialization_stats = {}
    manager.tp_worker.stage_cpu_weight_update_from_delta_lineage.return_value = (
        True,
        "staged",
        {},
    )
    request = _stage_request(
        destination="cpu",
        target_version=1,
        checkpoint_source_dir="/updates",
    )

    def gather_with_remote_failure(value, group):
        assert group is manager.weight_update_stage_cpu_group
        return [value, (False, "rank 1 failed", {"rank": 1})]

    with (
        mock.patch.object(
            SchedulerWeightUpdaterManager,
            "_refresh_checkpoint_source",
            return_value=0.0,
        ),
        mock.patch.object(
            SchedulerWeightUpdaterManager,
            "_all_gather",
            autospec=True,
            side_effect=gather_with_remote_failure,
        ),
    ):
        result = manager._stage_weight_update_sync(request)

    assert not result.success
    assert "rank 1 failed" in result.message
    manager.tp_worker.invalidate_staged_cpu_weight_update.assert_called_once()


def test_cache_initialization_runs_in_background_and_returns_deferred_output():
    manager = _manager(cpu_cache=True)
    started = threading.Event()
    release = threading.Event()

    def initialize(**_kwargs):
        started.set()
        assert release.wait(timeout=5)
        return {}

    manager.tp_worker.initialize_cpu_weight_cache.side_effect = initialize
    request = _stage_request(destination="cpu", target_version=0)

    assert manager.stage_weight_update(request) is None
    assert started.wait(timeout=5)
    assert manager._pending_weight_update_stage is not None

    release.set()
    manager._pending_weight_update_stage[1].join(timeout=5)
    manager.check_pending_weight_update_stage()

    manager.send_control_output.assert_called_once()
    sent_request, output = manager.send_control_output.call_args.args
    assert sent_request is request
    assert output.success


@pytest.mark.parametrize(
    ("method_name", "worker_method"),
    [
        ("update_weights_from_disk", "update_weights_from_disk"),
        ("update_weights_from_distributed", "update_weights_from_distributed"),
        ("update_weights_from_tensor", "update_weights_from_tensor"),
        ("update_weights_from_ipc", "update_weights_from_ipc"),
    ],
)
def test_background_stage_blocks_live_weight_mutations(method_name, worker_method):
    manager = _manager()
    manager._pending_weight_update_stage = (mock.Mock(), mock.Mock())

    result = getattr(manager, method_name)(SimpleNamespace())

    assert not result.success
    assert "background weight update stage" in result.message
    getattr(manager.tp_worker, worker_method).assert_not_called()


@pytest.mark.parametrize(
    ("method_name", "worker_method"),
    [
        ("update_weights_from_disk", "update_weights_from_disk"),
        ("update_weights_from_distributed", "update_weights_from_distributed"),
        ("update_weights_from_tensor", "update_weights_from_tensor"),
        ("update_weights_from_ipc", "update_weights_from_ipc"),
    ],
)
def test_cpu_cache_mode_blocks_other_weight_mutations(method_name, worker_method):
    manager = _manager(cpu_cache=True)

    result = getattr(manager, method_name)(SimpleNamespace())

    assert not result.success
    assert "CPU weight cache is enabled" in result.message
    getattr(manager.tp_worker, worker_method).assert_not_called()


def test_cpu_cache_blocks_weight_offloading():
    manager = _manager(cpu_cache=True)

    with pytest.raises(RuntimeError, match="CPU weight cache is enabled"):
        manager.release_memory_occupation(
            SimpleNamespace(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        )


def test_cpu_commit_flushes_before_updating_only_the_target():
    manager = _manager(cpu_cache=True)
    manager.draft_worker = mock.Mock()
    manager._cpu_weight_cache_initialization_stats = {}
    manager.tp_worker.validate_staged_cpu_weight_update.return_value = (True, "ready")
    order = []

    def flush_cache(*, empty_cache):
        order.append(("flush", empty_cache))
        return True

    def commit(target_version):
        order.append(("commit", target_version))
        return True, "committed", {}

    manager.flush_cache.side_effect = flush_cache
    manager.tp_worker.update_weights_from_cpu.side_effect = commit

    result = manager.update_weights_from_cpu(
        SimpleNamespace(
            target_version=2,
            flush_cache=True,
            torch_empty_cache=False,
        )
    )

    assert result.success
    assert order == [("flush", False), ("commit", 2)]
    manager.draft_worker.update_weights_from_cpu.assert_not_called()


def test_cpu_commit_response_reports_only_the_local_scheduler_stats():
    manager = _manager(cpu_cache=True)
    manager._cpu_weight_cache_initialization_stats = {}
    manager.tp_worker.validate_staged_cpu_weight_update.return_value = (True, "ready")
    manager.tp_worker.update_weights_from_cpu.return_value = (
        True,
        "committed",
        {"wall_s": 1.0},
    )

    def gather_with_remote_success(value, group):
        assert group is manager.tp_cpu_group
        if isinstance(value, bool):
            return [value, False]
        if len(value) == 2:
            return [value, (True, "ready")]
        return [value, (True, "committed", {"rank": 1, "wall_s": 2.0})]

    with mock.patch.object(
        SchedulerWeightUpdaterManager,
        "_all_gather",
        autospec=True,
        side_effect=gather_with_remote_success,
    ):
        result = manager.update_weights_from_cpu(
            SimpleNamespace(
                target_version=2,
                flush_cache=False,
                torch_empty_cache=False,
            )
        )

    assert result.success
    assert result.rank_stats == [{"rank": 0, "wall_s": 1.0}]


def test_cpu_commit_rejects_failed_flush_before_mutating_weights():
    manager = _manager(cpu_cache=True)
    manager._cpu_weight_cache_initialization_stats = {}
    manager.tp_worker.validate_staged_cpu_weight_update.return_value = (True, "ready")
    manager.flush_cache.return_value = False

    result = manager.update_weights_from_cpu(
        SimpleNamespace(
            target_version=2,
            flush_cache=True,
            torch_empty_cache=False,
        )
    )

    assert not result.success
    assert "Cache flush failed" in result.message
    manager.tp_worker.update_weights_from_cpu.assert_not_called()


def test_cpu_commit_fails_closed_after_mutation_error():
    manager = _manager(cpu_cache=True)
    manager._cpu_weight_cache_initialization_stats = {}
    manager.tp_worker.validate_staged_cpu_weight_update.return_value = (True, "ready")
    manager.tp_worker.update_weights_from_cpu.side_effect = RuntimeError("broken")

    with pytest.raises(RuntimeError, match="terminating the engine"):
        manager.update_weights_from_cpu(
            SimpleNamespace(
                target_version=1,
                flush_cache=False,
                torch_empty_cache=False,
            )
        )


def test_deferred_control_output_uses_rust_server_when_present():
    scheduler = SimpleNamespace(
        rust_server=mock.Mock(),
        ipc_channels=mock.Mock(),
    )
    request = object()
    output = object()

    Scheduler.send_control_output(scheduler, request, output)

    scheduler.rust_server.push_control_output.assert_called_once_with(request, output)
    scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()


def test_deferred_control_output_uses_tokenizer_ipc_without_rust():
    scheduler = SimpleNamespace(
        rust_server=None,
        ipc_channels=mock.Mock(),
    )
    request = object()
    output = object()

    Scheduler.send_control_output(scheduler, request, output)

    scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once_with(
        output,
        request,
    )
