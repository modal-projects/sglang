from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sglang.srt.weight_sync.cpu_weight_cache as cpu_weight_cache
from sglang.srt.model_executor.model_runner_components import (
    weight_updater as weight_updater_module,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeImage:
    def __init__(self):
        self.validated = []

    def validate_commit(self, target_version):
        self.validated.append(target_version)


class _FakeCache:
    def __init__(
        self,
        model,
        *,
        max_compile_group_bytes,
        host_group,
        canonical_checkpoint_dir=None,
    ):
        self.model = model
        self.max_compile_group_bytes = max_compile_group_bytes
        self.host_group = host_group
        self.canonical_checkpoint_dir = canonical_checkpoint_dir
        self.image = _FakeImage()
        self.staged = []
        self.staged_checkpoints = []
        self.committed = []
        self.invalidated = []
        self.closed = []

    def initialize_from_checkpoint(
        self, checkpoint_dir, *, seed_from_active_weights, base_version=0
    ):
        return {
            "checkpoint_dir": checkpoint_dir,
            "seed_from_active_weights": seed_from_active_weights,
            "base_version": base_version,
            "wall_s": 2.0,
        }

    def stage_delta_lineage(self, *, checkpoint_source_dir, target_version):
        self.staged.append((checkpoint_source_dir, target_version))
        return {
            "compile": {"bytes": 32, "groups": 2},
            "wall_s": 3.0,
        }

    def stage_checkpoint(self, checkpoint_dir, *, target_version):
        self.staged_checkpoints.append((checkpoint_dir, target_version))
        return {"wall_s": 4.0}

    def commit(self, target_version):
        self.committed.append(target_version)
        return {"bytes": 32, "wall_s": 1.0, "gbps": 0.000000032}

    def invalidate_stage(self, reason):
        self.invalidated.append(reason)

    def close(self, reason):
        self.closed.append(reason)


def _make_weight_updater(*, is_draft_worker=False):
    runner = SimpleNamespace(
        cpu_weight_cache=None,
        is_draft_worker=is_draft_worker,
        server_args=SimpleNamespace(
            dcp_replicate_q_proj=False,
            weight_cache_mode="off",
        ),
    )
    model = torch.nn.Linear(2, 2)
    updater = WeightUpdater(
        tp_rank=0,
        device="cuda",
        gpu_id=0,
        model_config=SimpleNamespace(),
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=lambda *_args, **_kwargs: None,
        recapture_cuda_graph=lambda: None,
        get_model_runner=lambda: runner,
    )
    return updater, runner


def test_cpu_weight_cache_worker_lifecycle():
    updater, runner = _make_weight_updater()
    host_group = object()
    with (
        patch.object(cpu_weight_cache, "CPUWeightCache", _FakeCache),
        patch.object(
            weight_updater_module,
            "_unsupported_derived_weight_cache_error",
            return_value=None,
        ),
        patch.object(torch.cuda, "device", return_value=nullcontext()),
        patch.object(torch.cuda, "synchronize"),
    ):
        initialization = updater.initialize_cpu_weight_cache(
            checkpoint_dir="/checkpoint",
            seed_from_active_weights=True,
            base_version=119,
            host_group=host_group,
            max_compile_group_bytes=1024,
            canonical_checkpoint_dir=None,
        )

        cache = runner.cpu_weight_cache
        assert isinstance(cache, _FakeCache)
        assert cache.host_group is host_group
        assert cache.max_compile_group_bytes == 1024
        assert initialization["cache_population_wall_s"] == 2.0
        assert initialization["seed_from_active_weights"] is True
        assert initialization["base_version"] == 119

        success, _, stage = updater.stage_cpu_weight_update_from_delta_lineage(
            checkpoint_source_dir="/updates",
            target_version=1,
            host_group=host_group,
        )
        assert success
        assert stage is not None
        assert cache.staged == [("/updates", 1)]
        assert updater.validate_staged_cpu_weight_update(1)[0]

        success, _, stage = updater.stage_cpu_weight_update_from_checkpoint(
            checkpoint_dir="/saved",
            target_version=119,
            host_group=host_group,
        )
        assert success
        assert stage is not None
        assert cache.staged_checkpoints == [("/saved", 119)]

        success, _, commit = updater.update_weights_from_cpu(1)
        assert success
        assert commit is not None
        assert cache.committed == [1]

        updater.invalidate_staged_cpu_weight_update("peer failed")
        assert cache.invalidated == ["peer failed"]
        updater.discard_cpu_weight_cache("done")
        assert runner.cpu_weight_cache is None
        assert cache.closed == ["done"]


def test_cpu_weight_cache_rejects_draft_models():
    updater, _ = _make_weight_updater(is_draft_worker=True)

    with pytest.raises(RuntimeError, match="only supported for target models"):
        updater.initialize_cpu_weight_cache(
            checkpoint_dir="/checkpoint",
            seed_from_active_weights=True,
            base_version=0,
            host_group=None,
            max_compile_group_bytes=1024,
            canonical_checkpoint_dir=None,
        )


def test_other_weight_updates_reject_an_active_cpu_cache():
    updater, runner = _make_weight_updater()
    runner.cpu_weight_cache = object()

    with pytest.raises(RuntimeError, match="canonical checkpoint"):
        updater.update_weights_from_disk("/checkpoint", "safetensors")


def test_cpu_staging_rejects_a_different_host_group():
    updater, runner = _make_weight_updater()
    runner.cpu_weight_cache = _FakeCache(
        torch.nn.Module(),
        max_compile_group_bytes=1024,
        host_group=object(),
    )

    success, message, stats = updater.stage_cpu_weight_update_from_delta_lineage(
        checkpoint_source_dir="/updates",
        target_version=1,
        host_group=object(),
    )

    assert not success
    assert "cannot change" in message
    assert stats is None
