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
from sglang.srt.weight_sync.checksum import calculate_checksum
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

    def compile(self, checkpoint, *, target_version):
        if target_version in self.fail_versions:
            self.image.invalidate("injected compilation failure")
            raise RuntimeError("injected compilation failure")
        value = checkpoint.get_tensor("a").clone()
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
    )
    with patch.object(host_memory, "_SHARED_MEMORY_ROOT", shared_memory):
        initialization = cache.initialize_from_checkpoint(
            base,
            seed_from_active_weights=seed_from_active_weights,
        )
    return cache, initialization, shared_memory


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
