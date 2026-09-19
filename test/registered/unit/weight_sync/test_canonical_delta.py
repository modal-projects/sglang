from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import zstandard
from safetensors.torch import save_file

import sglang.srt.weight_sync.canonical_delta as canonical_delta
import sglang.srt.weight_sync.host_local_buffer as host_memory
from sglang.srt.weight_sync.canonical_checkpoint import CanonicalCheckpoint
from sglang.srt.weight_sync.canonical_delta import CanonicalDeltaTransform
from sglang.srt.weight_sync.checksum import calculate_checksum
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _bytes(tensor: torch.Tensor) -> np.ndarray:
    return tensor.contiguous().view(torch.uint8).numpy()


def _write_delta(
    root: Path,
    version: int,
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    *,
    encoding: str,
    checksum_overrides: dict[str, str] | None = None,
) -> None:
    version_dir = root / f"weight_v{version:06d}"
    version_dir.mkdir(parents=True)
    payloads = {}
    checksums = {}
    for name, new_tensor in after.items():
        old = _bytes(before[name])
        new = _bytes(new_tensor)
        changed = new != old
        if not np.any(changed):
            continue
        if encoding == "xor":
            payload = np.bitwise_xor(new, old)
        else:
            positions = np.flatnonzero(changed).astype("<u4")
            payload = np.concatenate(
                [
                    np.array([positions.size], dtype="<u4").view(np.uint8),
                    positions.view(np.uint8),
                    new[changed],
                ]
            )
        compressed = zstandard.ZstdCompressor(level=1).compress(payload)
        payloads[name] = torch.frombuffer(bytearray(compressed), dtype=torch.uint8)
        checksums[name] = calculate_checksum("xxh3-128", new)

    checksums.update(checksum_overrides or {})
    filename = "model-00000-of-00001.safetensors"
    if payloads:
        save_file(payloads, version_dir / filename, metadata=checksums)
    index = {
        "metadata": {
            "version": f"{version:06d}",
            "base_version": f"{version - 1:06d}",
            "delta_encoding": encoding,
            "compression_format": "zstd",
            "checksum_format": "xxh3-128",
        },
        "weight_map": {name: filename for name in payloads},
    }
    (version_dir / "model.safetensors.index.json").write_text(json.dumps(index))


def _checkpoint(tmp_path: Path, tensors: dict[str, torch.Tensor]):
    base = tmp_path / "base"
    shared_memory = tmp_path / "shared-memory"
    base.mkdir()
    shared_memory.mkdir()
    save_file(tensors, base / "model.safetensors")
    with patch.object(host_memory, "_SHARED_MEMORY_ROOT", shared_memory):
        return CanonicalCheckpoint(base, host_group=None)


def _distributed_transform_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    base: str,
    versions: str,
    shared_memory: str,
) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    cached = None
    try:
        with patch.object(host_memory, "_SHARED_MEMORY_ROOT", Path(shared_memory)):
            cached = CanonicalCheckpoint(
                base,
                host_group=torch.distributed.group.WORLD,
            )
        CanonicalDeltaTransform(
            cached,
            checkpoint_source_dir=versions,
            target_version=1,
            host_group=torch.distributed.group.WORLD,
        ).apply()
        actual = cached.get_tensor("a")
        torch.testing.assert_close(
            actual,
            torch.arange(32, dtype=torch.uint8).roll(1),
        )
        del actual
        torch.distributed.barrier()
    finally:
        if cached is not None:
            cached.close()
        torch.distributed.destroy_process_group()


def test_xor_lineage_advances_canonical_checkpoint(tmp_path):
    versions = tmp_path / "updates"
    v0 = {
        "a": torch.arange(12, dtype=torch.float32),
        "b": torch.arange(9, dtype=torch.uint8),
    }
    v1 = {
        "a": v0["a"] + 0.5,
        "b": v0["b"].clone(),
    }
    v2 = {
        "a": v1["a"] * 2,
        "b": v1["b"].roll(2),
    }
    _write_delta(versions, 1, v0, v1, encoding="xor")
    _write_delta(versions, 2, v1, v2, encoding="xor")
    cached = _checkpoint(tmp_path, v0)
    try:
        transform = CanonicalDeltaTransform(
            cached,
            checkpoint_source_dir=versions,
            target_version=2,
            host_group=None,
            max_working_memory_bytes=1 << 20,
        )
        assert transform.setup_stats["delta_versions"] == [1, 2]
        stats = transform.apply()

        assert cached.version == 2
        assert stats["delta_tensors"] == 2
        assert stats["delta_fragments"] == 3
        for name, expected in v2.items():
            torch.testing.assert_close(cached.get_tensor(name), expected)
    finally:
        cached.close()


def test_xor_lineage_is_folded_before_canonical_mutation(tmp_path):
    versions = tmp_path / "updates"
    v0 = {"a": torch.arange(1024, dtype=torch.uint8)}
    v1 = {"a": v0["a"].roll(1)}
    v2 = {"a": v1["a"].roll(2)}
    _write_delta(versions, 1, v0, v1, encoding="xor")
    _write_delta(versions, 2, v1, v2, encoding="xor")
    cached = _checkpoint(tmp_path, v0)
    try:
        stats = CanonicalDeltaTransform(
            cached,
            checkpoint_source_dir=versions,
            target_version=2,
            host_group=None,
        ).apply()

        assert stats["folded_tensors"] == 1
        torch.testing.assert_close(cached.get_tensor("a"), v2["a"])
    finally:
        cached.close()


def test_mixed_lineage_is_folded_before_canonical_mutation(tmp_path):
    versions = tmp_path / "updates"
    v0 = {"a": torch.arange(1024, dtype=torch.uint8)}
    v1 = {"a": v0["a"].roll(1)}
    v2 = {"a": v1["a"].clone()}
    v2["a"][[0, 17, 511, 1023]] = torch.tensor([9, 8, 7, 6], dtype=torch.uint8)
    v3 = {"a": v2["a"].roll(2)}
    _write_delta(versions, 1, v0, v1, encoding="xor")
    _write_delta(versions, 2, v1, v2, encoding="overwrite")
    _write_delta(versions, 3, v2, v3, encoding="xor")
    cached = _checkpoint(tmp_path, v0)
    try:
        with patch.object(
            canonical_delta.np,
            "copyto",
            wraps=np.copyto,
        ) as copyto:
            CanonicalDeltaTransform(
                cached,
                checkpoint_source_dir=versions,
                target_version=3,
                host_group=None,
            ).apply()

        assert copyto.call_count == 1
        torch.testing.assert_close(cached.get_tensor("a"), v3["a"])
    finally:
        cached.close()


def test_folded_xor_cancellation_reconstructs_original_bytes(tmp_path):
    versions = tmp_path / "updates"
    v0 = {"a": torch.arange(256, dtype=torch.uint8)}
    v1 = {"a": v0["a"].roll(1)}
    v2 = {"a": v0["a"].clone()}
    _write_delta(versions, 1, v0, v1, encoding="xor")
    _write_delta(versions, 2, v1, v2, encoding="xor")
    cached = _checkpoint(tmp_path, v0)
    target = v0["a"].clone()
    try:
        CanonicalDeltaTransform(
            cached,
            checkpoint_source_dir=versions,
            target_version=2,
            host_group=None,
        )._transform_tensors(
            {"a": target},
            description="test",
        )

        torch.testing.assert_close(target, v2["a"])
    finally:
        cached.close()


def test_overwrite_delta_advances_canonical_checkpoint(tmp_path):
    versions = tmp_path / "updates"
    v0 = {"a": torch.arange(256, dtype=torch.uint8)}
    v1 = {"a": v0["a"].clone()}
    v1["a"][[0, 17, 128, 255]] = torch.tensor([9, 8, 7, 6], dtype=torch.uint8)
    _write_delta(versions, 1, v0, v1, encoding="overwrite")
    cached = _checkpoint(tmp_path, v0)
    try:
        CanonicalDeltaTransform(
            cached,
            checkpoint_source_dir=versions,
            target_version=1,
            host_group=None,
        ).apply()
        torch.testing.assert_close(cached.get_tensor("a"), v1["a"])
    finally:
        cached.close()


def test_caller_owned_transform_does_not_advance_checkpoint(tmp_path):
    versions = tmp_path / "updates"
    v0 = {"a": torch.arange(256, dtype=torch.uint8)}
    v1 = {"a": v0["a"].clone()}
    v1["a"][[0, 17, 128, 255]] = torch.tensor([9, 8, 7, 6], dtype=torch.uint8)
    _write_delta(versions, 1, v0, v1, encoding="overwrite")
    cached = _checkpoint(tmp_path, v0)
    target_bytes = _bytes(v0["a"]).copy()
    target = torch.from_numpy(target_bytes)
    try:
        transform = CanonicalDeltaTransform(
            cached,
            checkpoint_source_dir=versions,
            target_version=1,
            host_group=None,
        )
        transform._transform_tensors(
            {"a": target},
            description="test",
        )
        torch.testing.assert_close(target, v1["a"])
        assert cached.version == 0
    finally:
        cached.close()


def test_checksum_failure_invalidates_mutated_checkpoint(tmp_path):
    versions = tmp_path / "updates"
    v0 = {"a": torch.arange(16, dtype=torch.uint8)}
    v1 = {"a": v0["a"].roll(1)}
    _write_delta(
        versions,
        1,
        v0,
        v1,
        encoding="xor",
        checksum_overrides={"a": "0" * 32},
    )
    cached = _checkpoint(tmp_path, v0)
    try:
        transform = CanonicalDeltaTransform(
            cached,
            checkpoint_source_dir=versions,
            target_version=1,
            host_group=None,
        )
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            transform.apply()
        assert not cached.valid
        assert cached.version == 0
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            cached.get_tensor("a")
    finally:
        cached.close()


def test_invalid_lineage_fails_before_mutating_checkpoint(tmp_path):
    versions = tmp_path / "updates"
    v0 = {"a": torch.arange(8, dtype=torch.uint8)}
    v1 = {"a": v0["a"].roll(1)}
    _write_delta(versions, 1, v0, v1, encoding="xor")
    index_path = versions / "weight_v000001" / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["metadata"]["base_version"] = "000007"
    index_path.write_text(json.dumps(index))
    cached = _checkpoint(tmp_path, v0)
    try:
        with pytest.raises(ValueError, match="builds on v7"):
            CanonicalDeltaTransform(
                cached,
                checkpoint_source_dir=versions,
                target_version=1,
                host_group=None,
            )
        assert cached.valid
        assert cached.version == 0
        torch.testing.assert_close(cached.get_tensor("a"), v0["a"])
    finally:
        cached.close()


def test_current_target_is_a_noop(tmp_path):
    cached = _checkpoint(tmp_path, {"a": torch.arange(3)})
    try:
        stats = CanonicalDeltaTransform(
            cached,
            checkpoint_source_dir=tmp_path / "missing",
            target_version=0,
            host_group=None,
        ).apply()
        assert stats["delta_tensors"] == 0
        assert cached.version == 0
    finally:
        cached.close()


def test_empty_delta_version_is_folded_into_later_target(tmp_path):
    versions = tmp_path / "updates"
    v0 = {"a": torch.arange(8, dtype=torch.uint8)}
    v1 = {"a": v0["a"].clone()}
    v2 = {"a": v1["a"].roll(1)}
    _write_delta(versions, 1, v0, v1, encoding="xor")
    _write_delta(versions, 2, v1, v2, encoding="xor")
    cached = _checkpoint(tmp_path, v0)
    try:
        transform = CanonicalDeltaTransform(
            cached,
            checkpoint_source_dir=versions,
            target_version=2,
            host_group=None,
        )
        assert transform.setup_stats["delta_versions"] == [1, 2]
        transform.apply()
        assert cached.version == 2
        torch.testing.assert_close(cached.get_tensor("a"), v2["a"])
    finally:
        cached.close()


def test_host_workers_apply_shared_delta_exactly_once(tmp_path):
    base = tmp_path / "base"
    versions = tmp_path / "updates"
    shared_memory = tmp_path / "shared-memory"
    base.mkdir()
    shared_memory.mkdir()
    v0 = {"a": torch.arange(32, dtype=torch.uint8)}
    v1 = {"a": v0["a"].roll(1)}
    save_file(v0, base / "model.safetensors")
    _write_delta(versions, 1, v0, v1, encoding="xor")

    torch.multiprocessing.start_processes(
        _distributed_transform_worker,
        args=(
            2,
            str(tmp_path / "gloo-rendezvous"),
            str(base),
            str(versions),
            str(shared_memory),
        ),
        nprocs=2,
        join=True,
        start_method="fork",
    )
