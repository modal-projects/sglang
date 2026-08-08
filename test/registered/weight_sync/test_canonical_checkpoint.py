from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

import sglang.srt.weight_sync.host_local_buffer as host_memory
from sglang.srt.weight_sync.canonical_checkpoint import CanonicalCheckpoint
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _write_index(root, weight_map):
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


def test_indexed_checkpoint_is_cached_once_with_zero_copy_views(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    shared_memory = tmp_path / "shared-memory"
    checkpoint.mkdir()
    shared_memory.mkdir()
    expected = {
        "model.a": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "model.b": torch.arange(7, dtype=torch.uint8),
    }
    save_file({"model.a": expected["model.a"]}, checkpoint / "model-1.safetensors")
    save_file({"model.b": expected["model.b"]}, checkpoint / "model-2.safetensors")
    _write_index(
        checkpoint,
        {
            "model.a": "model-1.safetensors",
            "model.b": "model-2.safetensors",
        },
    )

    with patch.object(host_memory, "_SHARED_MEMORY_ROOT", shared_memory):
        cached = CanonicalCheckpoint(checkpoint, host_group=None)
    try:
        for name, tensor in expected.items():
            torch.testing.assert_close(cached.get_tensor(name), tensor)
        stats = cached.stats()
        assert stats["files"] == 2
        assert stats["tensors"] == 2
        assert stats["physical_host_copies"] == 1
        assert stats["allocated_bytes"] >= stats["checkpoint_bytes"]
        assert stats["version"] == 0
        assert list(shared_memory.iterdir()) == []
    finally:
        cached.close()


def test_unindexed_checkpoint_discovers_tensor_locations(tmp_path):
    shared_memory = tmp_path / "shared-memory"
    shared_memory.mkdir()
    save_file({"a": torch.arange(3)}, tmp_path / "a.safetensors")
    save_file({"b": torch.arange(5)}, tmp_path / "b.safetensors")

    with patch.object(host_memory, "_SHARED_MEMORY_ROOT", shared_memory):
        cached = CanonicalCheckpoint(tmp_path, host_group=None)
    try:
        assert cached.weight_map == {
            "a": "a.safetensors",
            "b": "b.safetensors",
        }
        torch.testing.assert_close(cached.get_tensor("b"), torch.arange(5))
    finally:
        cached.close()


def test_duplicate_unindexed_tensor_fails_loudly(tmp_path):
    save_file({"a": torch.arange(3)}, tmp_path / "a.safetensors")
    save_file({"a": torch.arange(3)}, tmp_path / "b.safetensors")

    with pytest.raises(ValueError, match="appears in both"):
        CanonicalCheckpoint(tmp_path, host_group=None)


def test_index_must_match_shard_contents(tmp_path):
    save_file({"a": torch.arange(3)}, tmp_path / "model.safetensors")
    _write_index(tmp_path, {"b": "model.safetensors"})

    with pytest.raises(ValueError, match="does not match checkpoint shards"):
        CanonicalCheckpoint(tmp_path, host_group=None)


def test_update_lifecycle_hides_partial_checkpoint(tmp_path):
    shared_memory = tmp_path / "shared-memory"
    shared_memory.mkdir()
    save_file({"a": torch.arange(3)}, tmp_path / "model.safetensors")

    with patch.object(host_memory, "_SHARED_MEMORY_ROOT", shared_memory):
        cached = CanonicalCheckpoint(tmp_path, host_group=None, version=4)
    try:
        cached.begin_update(5)
        with pytest.raises(RuntimeError, match="invalid"):
            cached.get_tensor("a")
        cached.finish_update(5)
        assert cached.version == 5
        torch.testing.assert_close(cached.get_tensor("a"), torch.arange(3))

        cached.begin_update(6)
        cached.fail_update("checksum mismatch")
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            cached.get_tensor("a")
    finally:
        cached.close()
