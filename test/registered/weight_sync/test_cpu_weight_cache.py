from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import zstandard
from safetensors.torch import save_file

import sglang.srt.weight_sync.cpu_weight_cache as weight_cache
import sglang.srt.weight_sync.host_local_buffer as host_memory
from sglang.srt.weight_sync.canonical_checkpoint import (
    CanonicalCheckpoint,
    DiskCanonicalCheckpointUpdate,
)
from sglang.srt.weight_sync.checksum import calculate_checksum
from sglang.srt.weight_sync.cpu_delta_checkpoint import DeltaCheckpointTransform
from sglang.srt.weight_sync.disk_checkpoint import materialize
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeImage:
    def __init__(self):
        self.image_nbytes = 4096
        self.weight_nbytes = 32
        self.valid = False
        self.staged = False
        self.target_version = None
        self.invalid_reason = None
        self.committed = []

    def validate_against_active(self):
        return {"operation": "validate", "wall_s": 0.0}

    def accept_staged_baseline(self):
        if not self.valid or not self.staged:
            raise RuntimeError("no staged baseline")
        self.staged = False
        self.target_version = None

    def invalidate(self, reason):
        self.valid = False
        self.staged = False
        self.invalid_reason = reason

    def commit(self, target_version):
        if not self.valid or not self.staged or self.target_version != target_version:
            raise RuntimeError("no matching staged image")
        self.staged = False
        self.committed.append(target_version)
        return {"operation": "commit", "target_version": target_version}


class _FakeCompiler:
    fail_versions = set()

    def __init__(self, _model, *, max_group_bytes):
        self.max_group_bytes = max_group_bytes
        self.image = _FakeImage()
        self.compiled = []
        self.closed = False

    def initialize_from_active(self):
        self.image.valid = True
        return {"operation": "initialize", "wall_s": 0.0}

    def checkpoint_groups(self, weight_map):
        return {"model": list(weight_map)}

    def compile(self, checkpoint, *, target_version):
        if target_version in self.fail_versions:
            self.image.invalidate("injected compilation failure")
            raise RuntimeError("injected compilation failure")
        with checkpoint.tensor_group("model", ["a"]) as group:
            value = group.get_tensor("a").clone()
        self.compiled.append((target_version, value))
        self.image.valid = True
        self.image.staged = True
        self.image.target_version = target_version
        return {"operation": "compile", "target_version": target_version}

    def close(self):
        self.closed = True


def _write_delta(
    root: Path,
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    checksum: str | None = None,
) -> None:
    version_dir = root / "weight_v000001"
    version_dir.mkdir(parents=True)
    before_bytes = before.contiguous().view(torch.uint8).numpy()
    after_bytes = after.contiguous().view(torch.uint8).numpy()
    payload = np.bitwise_xor(before_bytes, after_bytes)
    compressed = zstandard.ZstdCompressor(level=1).compress(payload)
    filename = "model-00000-of-00001.safetensors"
    save_file(
        {"a": torch.frombuffer(bytearray(compressed), dtype=torch.uint8)},
        version_dir / filename,
        metadata={
            "a": checksum or calculate_checksum("xxh3-128", after_bytes),
        },
    )
    (version_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "version": "000001",
                    "base_version": "000000",
                    "delta_encoding": "xor",
                    "compression_format": "zstd",
                    "checksum_format": "xxh3-128",
                },
                "weight_map": {"a": filename},
            }
        )
    )


def _create_cache(
    tmp_path,
    monkeypatch,
    base_value,
    *,
    canonical_checkpoint_dir=None,
    seed_from_active_weights=False,
):
    base = tmp_path / "base"
    shared_memory = tmp_path / "shared-memory"
    base.mkdir()
    shared_memory.mkdir()
    save_file({"a": base_value}, base / "model.safetensors")
    monkeypatch.setattr(weight_cache, "CPUWeightCompiler", _FakeCompiler)
    _FakeCompiler.fail_versions = set()
    cache = weight_cache.CPUWeightCache(
        torch.nn.Module(),
        max_compile_group_bytes=1024,
        host_group=None,
        canonical_checkpoint_dir=canonical_checkpoint_dir,
    )
    with patch.object(host_memory, "_SHARED_MEMORY_ROOT", shared_memory):
        initialization = cache.initialize_from_checkpoint(
            base,
            seed_from_active_weights=seed_from_active_weights,
        )
    return cache, initialization, shared_memory


def _distributed_disk_update_worker(
    rank,
    world_size,
    rendezvous,
    canonical,
    updates,
    shared_memory,
    expected,
):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    checkpoint = None
    try:
        checkpoint = CanonicalCheckpoint(
            canonical,
            host_group=torch.distributed.group.WORLD,
            version=0,
            storage="disk",
        )
        transform = DeltaCheckpointTransform(
            checkpoint,
            checkpoint_source_dir=updates,
            target_version=1,
            host_group=torch.distributed.group.WORLD,
        )
        with patch.object(host_memory, "_SHARED_MEMORY_ROOT", Path(shared_memory)):
            update = DiskCanonicalCheckpointUpdate(
                checkpoint,
                names_by_group={"model": ["a"]},
                transform=transform,
                target_version=1,
                host_group=torch.distributed.group.WORLD,
            )
        checkpoint.close()
        checkpoint = None
        with update:
            with update.tensor_group("model", ["a"]) as group:
                actual = group.get_tensor("a").clone()
            update.finish()
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.uint8))
        torch.distributed.barrier()
    finally:
        if checkpoint is not None:
            checkpoint.close()
        torch.distributed.destroy_process_group()


def test_cache_initializes_stages_and_commits(tmp_path, monkeypatch):
    base = torch.arange(8, dtype=torch.uint8)
    target = base.roll(1)
    updates = tmp_path / "updates"
    _write_delta(updates, base, target)
    cache, initialization, _ = _create_cache(tmp_path, monkeypatch, base)
    try:
        assert initialization["canonical_checkpoint"]["version"] == 0
        torch.testing.assert_close(cache.compiler.compiled[0][1], base)

        stats = cache.stage_delta_lineage(
            checkpoint_source_dir=updates,
            target_version=1,
        )
        assert stats["canonical_reset"] is False
        assert cache.canonical_version == 1
        torch.testing.assert_close(cache.compiler.compiled[-1][1], target)

        cache.commit(1)
        assert cache.image.committed == [1]
    finally:
        cache.close()


def test_boot_checkpoint_uses_captured_active_image(tmp_path, monkeypatch):
    base = torch.arange(8, dtype=torch.uint8)
    cache, initialization, _ = _create_cache(
        tmp_path,
        monkeypatch,
        base,
        seed_from_active_weights=True,
    )
    try:
        assert initialization["rank_image_source"] == "active_model"
        assert initialization["baseline_compile"] is None
        assert initialization["validation"] is None
        assert cache.compiler.compiled == []
        assert cache.image.valid
        assert not cache.image.staged
    finally:
        cache.close()


def test_disk_canonical_checkpoint_materializes_and_compiles(tmp_path, monkeypatch):
    base = torch.arange(8, dtype=torch.uint8)
    target = base.roll(1)
    updates = tmp_path / "updates"
    canonical = tmp_path / "canonical"
    _write_delta(updates, base, target)
    cache, initialization, _ = _create_cache(
        tmp_path,
        monkeypatch,
        base,
        canonical_checkpoint_dir=canonical,
    )
    try:
        assert initialization["canonical_checkpoint"]["storage"] == "host_local_disk"
        assert initialization["canonical_checkpoint"]["allocated_bytes"] == 0
        assert initialization["canonical_materialization"]["target_version"] == 0

        stats = cache.stage_delta_lineage(
            checkpoint_source_dir=updates,
            target_version=1,
        )
        assert stats["delta_setup"]["delta_versions"] == [1]
        assert stats["delta_transform"]["delta_tensors"] == 1
        assert (
            stats["canonical_materialization"]["operation"]
            == "stream_canonical_checkpoint_update"
        )
        assert stats["canonical_materialization"]["canonical_read_bytes"] == 8
        assert stats["canonical_materialization"]["persistence"]["physical_bytes"] == 8
        assert stats["canonical_materialization"]["target_version"] == 1
        assert cache.canonical_version == 1
        torch.testing.assert_close(cache.compiler.compiled[-1][1], target)
    finally:
        cache.close()


def test_disk_canonical_update_is_shared_and_published_once(tmp_path):
    base = tmp_path / "base"
    canonical = tmp_path / "canonical"
    updates = tmp_path / "updates"
    shared_memory = tmp_path / "shared-memory"
    base.mkdir()
    shared_memory.mkdir()
    before = torch.arange(32, dtype=torch.uint8)
    after = before.roll(1)
    save_file({"a": before}, base / "model.safetensors")
    _write_delta(updates, before, after)
    materialize(
        local_checkpoint_dir=str(canonical),
        base_checkpoint_dir=str(base),
        checkpoint_source_dir=str(base),
        target_version=0,
    )

    torch.multiprocessing.start_processes(
        _distributed_disk_update_worker,
        args=(
            2,
            str(tmp_path / "gloo-rendezvous"),
            str(canonical),
            str(updates),
            str(shared_memory),
            after.tolist(),
        ),
        nprocs=2,
        join=True,
        start_method="fork",
    )

    checkpoint = CanonicalCheckpoint(
        canonical,
        host_group=None,
        version=1,
        storage="disk",
    )
    try:
        torch.testing.assert_close(checkpoint.get_tensor("a"), after)
        state = json.loads((canonical / ".weight_sync" / "state.json").read_text())
        assert int(state["version"]) == 1
    finally:
        checkpoint.close()


def test_changing_lineage_reseeds_the_canonical_checkpoint(tmp_path, monkeypatch):
    base = torch.arange(8, dtype=torch.uint8)
    first = base.roll(1)
    second = base.flip(0)
    first_updates = tmp_path / "first-updates"
    second_updates = tmp_path / "second-updates"
    _write_delta(first_updates, base, first)
    _write_delta(second_updates, base, second)
    cache, _, shared_memory = _create_cache(tmp_path, monkeypatch, base)
    try:
        cache.stage_delta_lineage(
            checkpoint_source_dir=first_updates,
            target_version=1,
        )
        with patch.object(host_memory, "_SHARED_MEMORY_ROOT", shared_memory):
            stats = cache.stage_delta_lineage(
                checkpoint_source_dir=second_updates,
                target_version=1,
            )
        assert stats["canonical_reset"] is True
        torch.testing.assert_close(cache.compiler.compiled[-1][1], second)
    finally:
        cache.close()


def test_failed_transform_is_discarded_and_can_be_retried(tmp_path, monkeypatch):
    base = torch.arange(8, dtype=torch.uint8)
    target = base.roll(1)
    updates = tmp_path / "updates"
    _write_delta(updates, base, target, checksum="0" * 32)
    cache, _, shared_memory = _create_cache(tmp_path, monkeypatch, base)
    try:
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            cache.stage_delta_lineage(
                checkpoint_source_dir=updates,
                target_version=1,
            )
        assert cache.canonical_version is None
        assert cache.image.valid

        shutil.rmtree(updates)
        _write_delta(updates, base, target)
        with patch.object(host_memory, "_SHARED_MEMORY_ROOT", shared_memory):
            stats = cache.stage_delta_lineage(
                checkpoint_source_dir=updates,
                target_version=1,
            )
        assert stats["canonical_reset"] is True
        torch.testing.assert_close(cache.compiler.compiled[-1][1], target)
    finally:
        cache.close()


def test_compilation_retry_reuses_verified_canonical_target(tmp_path, monkeypatch):
    base = torch.arange(8, dtype=torch.uint8)
    target = base.roll(1)
    updates = tmp_path / "updates"
    _write_delta(updates, base, target)
    cache, _, _ = _create_cache(tmp_path, monkeypatch, base)
    try:
        _FakeCompiler.fail_versions = {1}
        with pytest.raises(RuntimeError, match="injected compilation failure"):
            cache.stage_delta_lineage(
                checkpoint_source_dir=updates,
                target_version=1,
            )
        assert cache.canonical_version == 1

        _FakeCompiler.fail_versions = set()
        stats = cache.stage_delta_lineage(
            checkpoint_source_dir=updates,
            target_version=1,
        )
        assert stats["delta_setup"]["delta_versions"] == []
        torch.testing.assert_close(cache.compiler.compiled[-1][1], target)
    finally:
        _FakeCompiler.fail_versions = set()
        cache.close()


def test_failed_disk_compilation_leaves_no_published_canonical_version(
    tmp_path, monkeypatch
):
    base = torch.arange(8, dtype=torch.uint8)
    target = base.roll(1)
    updates = tmp_path / "updates"
    canonical = tmp_path / "canonical"
    _write_delta(updates, base, target)
    cache, _, _ = _create_cache(
        tmp_path,
        monkeypatch,
        base,
        canonical_checkpoint_dir=canonical,
    )
    try:
        _FakeCompiler.fail_versions = {1}
        with pytest.raises(RuntimeError, match="injected compilation failure"):
            cache.stage_delta_lineage(
                checkpoint_source_dir=updates,
                target_version=1,
            )
        assert cache.canonical_version is None
        assert not (canonical / ".weight_sync" / "state.json").exists()

        _FakeCompiler.fail_versions = set()
        stats = cache.stage_delta_lineage(
            checkpoint_source_dir=updates,
            target_version=1,
        )
        assert stats["canonical_reset"] is True
        torch.testing.assert_close(cache.compiler.compiled[-1][1], target)
    finally:
        _FakeCompiler.fail_versions = set()
        cache.close()


def test_failed_disk_checksum_leaves_no_published_canonical_version(
    tmp_path, monkeypatch
):
    base = torch.arange(8, dtype=torch.uint8)
    target = base.roll(1)
    updates = tmp_path / "updates"
    canonical = tmp_path / "canonical"
    _write_delta(updates, base, target, checksum="0" * 32)
    cache, _, _ = _create_cache(
        tmp_path,
        monkeypatch,
        base,
        canonical_checkpoint_dir=canonical,
    )
    try:
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            cache.stage_delta_lineage(
                checkpoint_source_dir=updates,
                target_version=1,
            )
        assert cache.canonical_version is None
        assert not (canonical / ".weight_sync" / "state.json").exists()
    finally:
        cache.close()
