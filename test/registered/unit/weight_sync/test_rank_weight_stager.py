from __future__ import annotations

from types import SimpleNamespace

import pytest

import sglang.srt.weight_sync.rank_weight_stager as stager_module
from sglang.srt.weight_sync.rank_weight_stager import RankWeightStager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Image:
    def __init__(self):
        self.image_nbytes = 64
        self.staged = False
        self.target_version = None
        self.valid = True
        self.invalid_reason = None
        self.commits = []

    def validate_commit(self, version):
        if not self.valid or not self.staged or self.target_version != version:
            raise RuntimeError("image is not prepared")

    def commit(self, version):
        self.validate_commit(version)
        self.staged = False
        self.commits.append(version)
        return {"operation": "commit", "target_version": version}

    def invalidate(self, reason):
        self.valid = False
        self.staged = False
        self.invalid_reason = reason


class _Compiler:
    def __init__(self, *_args, **_kwargs):
        self.image = _Image()
        self.compiles = []
        self.validated_names = []

    def initialize_from_active(self):
        return {"operation": "initialize"}

    def prepare_loader_views(self, weight_map):
        return {"operation": "prepare_views", "tensors": len(weight_map)}

    def validate_delta_names(self, names):
        self.validated_names.append(set(names))

    def compile(self, checkpoint, *, target_version):
        assert checkpoint.version == target_version
        self.image.valid = True
        self.image.staged = True
        self.image.target_version = target_version
        self.compiles.append(target_version)
        return {"operation": "compile", "target_version": target_version}

    def close(self):
        pass


class _Checkpoint:
    def __init__(self, root, *, host_group, version, storage):
        self.root = root
        self.host_group = host_group
        self.version = version
        self.storage = storage
        self.valid = True
        self.weight_map = {"weight": "model.safetensors"}
        self.checkpoint_bytes = 32
        self.closed = False

    def stats(self):
        if not self.valid:
            raise RuntimeError("invalid checkpoint")
        return {"version": self.version, "storage": self.storage}

    def release_cached_pages(self):
        return {"files": 0}

    def close(self):
        self.closed = True


class _Transform:
    def __init__(
        self,
        checkpoint,
        *,
        checkpoint_source_dir,
        target_version,
        host_group,
    ):
        self.checkpoint = checkpoint
        self.target_version = target_version
        self.operations_by_name = {"weight": [target_version]}

    def apply(self):
        previous = self.checkpoint.version
        self.checkpoint.version = self.target_version
        return {"from": previous, "to": self.target_version}


def _new_stager(monkeypatch, *, canonical_checkpoint_dir=None):
    materializations = []

    def fake_materialize(**kwargs):
        materializations.append(kwargs)
        return {"operation": "materialize", "target": kwargs["target_version"]}

    monkeypatch.setattr(stager_module, "RankWeightCompiler", _Compiler)
    monkeypatch.setattr(stager_module, "CanonicalCheckpoint", _Checkpoint)
    monkeypatch.setattr(stager_module, "CanonicalDeltaTransform", _Transform)
    monkeypatch.setattr(stager_module, "materialize", fake_materialize)
    stager = RankWeightStager(
        SimpleNamespace(),
        max_compile_group_bytes=16,
        host_group=None,
        canonical_checkpoint_dir=canonical_checkpoint_dir,
    )
    return stager, materializations


def test_memory_canonical_stage_and_commit(monkeypatch, tmp_path):
    stager, materializations = _new_stager(monkeypatch)
    stats = stager.initialize(tmp_path / "base", version=4)

    assert stats["version"] == 4
    assert stager.served_version == 4
    assert stager.canonical_version == 4
    assert materializations == []

    stats = stager.stage(
        checkpoint_source_dir=tmp_path / "updates",
        target_version=7,
    )
    assert stats["canonical_transform"] == {"from": 4, "to": 7}
    assert stager.prepared_version == 7
    assert stager.canonical_version == 7

    assert stager.commit(7)["target_version"] == 7
    assert stager.served_version == 7
    assert stager.prepared_version is None


def test_repeated_stage_is_idempotent(monkeypatch, tmp_path):
    stager, _ = _new_stager(monkeypatch)
    stager.initialize(tmp_path / "base", version=0)
    stager.stage(checkpoint_source_dir=tmp_path / "updates", target_version=2)

    stats = stager.stage(
        checkpoint_source_dir=tmp_path / "updates",
        target_version=2,
    )

    assert stats["reused"] is True
    assert stager.compiler.compiles == [2]
    with pytest.raises(RuntimeError, match="already prepared"):
        stager.stage(
            checkpoint_source_dir=tmp_path / "updates",
            target_version=3,
        )


def test_discard_allows_canonical_to_advance(monkeypatch, tmp_path):
    stager, _ = _new_stager(monkeypatch)
    stager.initialize(tmp_path / "base", version=0)
    stager.stage(checkpoint_source_dir=tmp_path / "updates", target_version=2)
    stager.discard_prepared("not committed")

    stats = stager.stage(
        checkpoint_source_dir=tmp_path / "updates",
        target_version=5,
    )

    assert stats["canonical_transform"] == {"from": 2, "to": 5}
    assert stager.prepared_version == 5


def test_rollback_target_reseeds_canonical(monkeypatch, tmp_path):
    stager, _ = _new_stager(monkeypatch)
    stager.initialize(tmp_path / "base", version=0)
    stager.stage(checkpoint_source_dir=tmp_path / "updates", target_version=5)
    stager.discard_prepared("prepare a lower target")

    stats = stager.stage(
        checkpoint_source_dir=tmp_path / "updates",
        target_version=3,
    )

    assert stats["canonical_reset"] is True
    assert stats["canonical_transform"] == {"from": 0, "to": 3}


def test_changing_source_lineage_reseeds_canonical(monkeypatch, tmp_path):
    stager, _ = _new_stager(monkeypatch)
    stager.initialize(tmp_path / "base", version=0)
    stager.stage(checkpoint_source_dir=tmp_path / "updates-a", target_version=2)
    stager.discard_prepared("switch source lineage")

    stats = stager.stage(
        checkpoint_source_dir=tmp_path / "updates-b",
        target_version=3,
    )

    assert stats["canonical_reset"] is True
    assert stats["canonical_transform"] == {"from": 0, "to": 3}


def test_disk_canonical_reopens_after_materialization(monkeypatch, tmp_path):
    local = tmp_path / "canonical"
    stager, materializations = _new_stager(
        monkeypatch,
        canonical_checkpoint_dir=local,
    )
    stager.initialize(tmp_path / "base", version=0)

    stats = stager.stage(
        checkpoint_source_dir=tmp_path / "updates",
        target_version=4,
    )

    assert [call["target_version"] for call in materializations] == [0, 4]
    assert stats["canonical_checkpoint"] == {"version": 4, "storage": "disk"}
    assert stager.canonical_version == 4


def test_failed_stage_invalidates_image(monkeypatch, tmp_path):
    stager, _ = _new_stager(monkeypatch)
    stager.initialize(tmp_path / "base", version=0)

    def fail_compile(*_args, **_kwargs):
        raise RuntimeError("compile failed")

    stager.compiler.compile = fail_compile
    with pytest.raises(RuntimeError, match="compile failed"):
        stager.stage(
            checkpoint_source_dir=tmp_path / "updates",
            target_version=1,
        )

    assert stager.prepared_version is None
    assert not stager.compiler.image.valid
    assert "compile failed" in stager.compiler.image.invalid_reason


def test_commit_requires_exact_prepared_version(monkeypatch, tmp_path):
    stager, _ = _new_stager(monkeypatch)
    stager.initialize(tmp_path / "base", version=0)
    stager.stage(checkpoint_source_dir=tmp_path / "updates", target_version=1)

    with pytest.raises(RuntimeError, match="not prepared"):
        stager.commit(2)
    assert stager.served_version == 0
