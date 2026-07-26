"""CPU tests for host-cached weight preparation."""

import concurrent.futures
import json
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import zstandard
from safetensors.torch import save_file

from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
    _prepared_weight_copy_workers,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.model_loader.utils import (
    STABLE_CPU_WEIGHT_SOURCE_ATTR,
)
from sglang.srt.weight_sync import cpu_weight_cache
from sglang.srt.weight_sync.cpu_weight_cache import (
    CpuImageGroupLoad,
    CPUWeightCache,
    HostSharedCheckpoint,
    InMemorySafeTensorsFile,
    WeightModuleGroup,
    _canonical_checkpoint_layout,
    _pread_file_to_tensor,
    _preapply_resident_checkpoint_transform,
    clone_module_tensors,
)
from sglang.srt.weight_sync.delta_checkpoint import (
    DeltaCheckpointOverlay,
    HostSharedDeltaBuffer,
    _resolve_lineage,
    validate_delta_target,
)
from sglang.srt.weight_sync.prepared_weights import PreparedWeightSegment
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class _CopyOnlyModelOptMoE(FusedMoE):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.quant_method = object.__new__(ModelOptNvFp4FusedMoEMethod)

    def weight_loader(self, target, source):
        self._copy_loaded_weight(target, source)


def _shared_checkpoint_worker(rank: int, world_size: int, rendezvous: str):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    source = None
    try:
        source = HostSharedCheckpoint(
            capacity=4097,
            cpu_group=torch.distributed.group.WORLD,
        )
        view = source.view(4097)
        if rank == 2:
            view.fill_(37)
        torch.distributed.barrier()
        if not torch.all(view == 37):
            raise AssertionError(f"rank {rank} did not observe the shared write")
        tail = source.view(17, offset=4097)
        if rank == 0:
            tail.fill_(91)
        torch.distributed.barrier()
        if not torch.all(tail == 91):
            raise AssertionError(f"rank {rank} did not observe the offset write")
        del tail
        del view
        torch.distributed.barrier()
    finally:
        if source is not None:
            source.close()
        torch.distributed.destroy_process_group()


def _shared_delta_worker(rank: int, world_size: int, rendezvous: str):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    arena = None
    try:
        arena = HostSharedDeltaBuffer(
            nbytes=5003,
            cpu_group=torch.distributed.group.WORLD,
        )
        view = arena.view(5003, offset=0)
        if rank == 1:
            view.fill_(73)
        torch.distributed.barrier()
        if not torch.all(view == 73):
            raise AssertionError(f"rank {rank} did not observe shared delta bytes")
        del view
        torch.distributed.barrier()
    finally:
        if arena is not None:
            arena.close()
        torch.distributed.destroy_process_group()


class TestInMemorySafeTensorsFile(unittest.TestCase):
    def test_cpu_cache_rejects_secondary_checkpoint_sources(self):
        model = torch.nn.Linear(2, 2)
        model.secondary_weights = [object()]

        with self.assertRaisesRegex(NotImplementedError, "secondary checkpoint"):
            CPUWeightCache(model, max_group_bytes=1024)

    def test_positional_read_and_tensor_views(self):
        tensors = {
            "bf16": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
            "f32": torch.arange(10, dtype=torch.float32).reshape(2, 5),
            "u8": torch.arange(17, dtype=torch.uint8),
        }
        if hasattr(torch, "float8_e4m3fn"):
            tensors["f8"] = torch.arange(8, dtype=torch.float32).to(torch.float8_e4m3fn)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.safetensors"
            save_file(tensors, path)
            source = torch.empty(path.stat().st_size, dtype=torch.uint8)
            with mock.patch.object(
                cpu_weight_cache,
                "_POSITIONAL_IO_CHUNK_BYTES",
                16,
            ):
                wall_s = _pread_file_to_tensor(
                    path,
                    source,
                )
            self.assertGreaterEqual(wall_s, 0.0)

            parsed = InMemorySafeTensorsFile(source)
            copied = InMemorySafeTensorsFile(
                source.clone(),
                layout=parsed.layout,
            )
            for name, expected in tensors.items():
                for actual in (parsed.get_tensor(name), copied.get_tensor(name)):
                    self.assertTrue(
                        getattr(
                            actual,
                            STABLE_CPU_WEIGHT_SOURCE_ATTR,
                            False,
                        )
                    )
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertTrue(
                        torch.equal(
                            actual.view(torch.uint8),
                            expected.view(torch.uint8),
                        )
                    )

    def test_invalid_header_fails_loudly(self):
        source = torch.tensor(
            list((1024).to_bytes(8, "little")) + [0] * 8,
            dtype=torch.uint8,
        )
        with self.assertRaisesRegex(ValueError, "header length"):
            InMemorySafeTensorsFile(source)

    def test_runtime_delta_reconstructs_and_verifies_in_source_buffer(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            base = root / "base"
            source_root = root / "published"
            version = source_root / "weight_v000001"
            version_2 = source_root / "weight_v000002"
            base.mkdir()
            version.mkdir(parents=True)
            version_2.mkdir(parents=True)
            shard = "model-00001-of-00001.safetensors"

            original = {
                "layer.a": torch.arange(64, dtype=torch.uint8),
                "layer.b": torch.arange(31, dtype=torch.uint8),
            }
            target_a = torch.arange(64, dtype=torch.uint8).flip(0)
            target_a_2 = torch.arange(64, dtype=torch.uint8).roll(11)
            save_file(original, base / shard)
            (base / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": {name: shard for name in original},
                    }
                )
            )

            difference = torch.bitwise_xor(original["layer.a"], target_a)
            compressed = zstandard.ZstdCompressor().compress(
                difference.numpy().tobytes()
            )
            checksum = f"{zlib.adler32(target_a.numpy(), 1):08x}"
            save_file(
                {"layer.a": torch.tensor(list(compressed), dtype=torch.uint8)},
                version / shard,
                metadata={"layer.a": checksum},
            )
            (version / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "version": "000001",
                            "base_version": "000000",
                            "delta_encoding": "xor",
                            "compression_format": "zstd",
                            "checksum_format": "adler32",
                        },
                        "weight_map": {"layer.a": shard},
                    }
                )
            )
            difference_2 = torch.bitwise_xor(target_a, target_a_2)
            compressed_2 = zstandard.ZstdCompressor().compress(
                difference_2.numpy().tobytes()
            )
            checksum_2 = f"{zlib.adler32(target_a_2.numpy(), 1):08x}"
            save_file(
                {
                    "layer.a": torch.tensor(
                        list(compressed_2),
                        dtype=torch.uint8,
                    )
                },
                version_2 / shard,
                metadata={"layer.a": checksum_2},
            )
            (version_2 / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "version": "000002",
                            "base_version": "000001",
                            "delta_encoding": "xor",
                            "compression_format": "zstd",
                            "checksum_format": "adler32",
                        },
                        "weight_map": {"layer.a": shard},
                    }
                )
            )

            source = DeltaCheckpointOverlay(
                base_checkpoint_dir=str(base),
                source_dir=str(source_root),
                target_version=1,
                cpu_group=None,
            )
            try:
                path = base / shard
                encoded = torch.empty(path.stat().st_size, dtype=torch.uint8)
                _pread_file_to_tensor(path, encoded)
                tensor_file = InMemorySafeTensorsFile(encoded)
                stats = source.transform_file(shard, tensor_file)
                torch.testing.assert_close(
                    tensor_file.get_tensor("layer.a"),
                    target_a,
                )
                torch.testing.assert_close(
                    tensor_file.get_tensor("layer.b"),
                    original["layer.b"],
                )
                self.assertEqual(stats["changed_tensors"], 1)
                self.assertEqual(stats["target_tensor_bytes"], 64)
                self.assertEqual(source.stage_stats["physical_host_copies"], 1)
            finally:
                source.close()

            source_2 = DeltaCheckpointOverlay(
                base_checkpoint_dir=str(base),
                source_dir=str(source_root),
                target_version=2,
                cpu_group=None,
                base_version=1,
                current_checkpoint_dir=str(base),
            )
            try:
                stats_2 = source_2.transform_file(shard, tensor_file)
                torch.testing.assert_close(
                    tensor_file.get_tensor("layer.a"),
                    target_a_2,
                )
                torch.testing.assert_close(
                    tensor_file.get_tensor("layer.b"),
                    original["layer.b"],
                )
                self.assertEqual(source_2.stage_stats["delta_versions"], [2])
                self.assertEqual(stats_2["changed_tensors"], 1)
            finally:
                source_2.close()

    def test_resident_full_checkpoint_can_anchor_a_later_delta(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            base = root / "base"
            published = root / "published"
            full = published / "weight_v000001"
            delta = published / "weight_v000002"
            base.mkdir()
            full.mkdir(parents=True)
            delta.mkdir()
            shard = "model.safetensors"

            save_file({"weight": torch.tensor([0], dtype=torch.uint8)}, base / shard)
            target_1 = torch.arange(32, dtype=torch.uint8)
            target_2 = target_1.flip(0)
            save_file({"weight": target_1}, full / shard)
            for checkpoint in (base, full):
                (checkpoint / "model.safetensors.index.json").write_text(
                    json.dumps(
                        {
                            "metadata": {},
                            "weight_map": {"weight": shard},
                        }
                    )
                )

            difference = torch.bitwise_xor(target_1, target_2)
            compressed = zstandard.ZstdCompressor().compress(
                difference.numpy().tobytes()
            )
            save_file(
                {"weight": torch.tensor(list(compressed), dtype=torch.uint8)},
                delta / shard,
                metadata={
                    "weight": f"{zlib.adler32(target_2.numpy(), 1):08x}",
                },
            )
            (delta / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "version": "000002",
                            "base_version": "000001",
                            "delta_encoding": "xor",
                            "compression_format": "zstd",
                            "checksum_format": "adler32",
                        },
                        "weight_map": {"weight": shard},
                    }
                )
            )

            source = DeltaCheckpointOverlay(
                base_checkpoint_dir=str(base),
                current_checkpoint_dir=str(full),
                source_dir=str(published),
                target_version=2,
                base_version=1,
                cpu_group=None,
            )
            try:
                encoded = torch.empty((full / shard).stat().st_size, dtype=torch.uint8)
                _pread_file_to_tensor(full / shard, encoded)
                tensor_file = InMemorySafeTensorsFile(encoded)
                source.transform_file(shard, tensor_file)
                torch.testing.assert_close(
                    tensor_file.get_tensor("weight"),
                    target_2,
                )
            finally:
                source.close()

    def test_cpu_preparation_rejects_full_checkpoint_targets(self):
        with tempfile.TemporaryDirectory() as root_value:
            source = Path(root_value)
            full = source / "weight_v000001"
            full.mkdir()
            (full / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"version": "000001"},
                        "weight_map": {"weight": "model.safetensors"},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "delta targets only"):
                validate_delta_target(str(source), 1)
            with self.assertRaisesRegex(ValueError, "requires a delta target"):
                validate_delta_target(str(source), 0)

    def test_delta_after_full_anchor_reseeds_resident_snapshot(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            base = root / "base"
            source = root / "published"
            resident = root / "resident"
            base.mkdir()
            resident.mkdir()
            (base / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": {"weight": "base.safetensors"},
                    }
                )
            )
            (resident / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": {"weight": "resident.safetensors"},
                    }
                )
            )

            full = source / "weight_v000002"
            delta = source / "weight_v000003"
            full.mkdir(parents=True)
            delta.mkdir()
            (full / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"version": "000002"},
                        "weight_map": {"weight": "full.safetensors"},
                    }
                )
            )
            (delta / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "version": "000003",
                            "base_version": "000002",
                            "delta_encoding": "xor",
                            "compression_format": "zstd",
                            "checksum_format": "adler32",
                        },
                        "weight_map": {"weight": "delta.safetensors"},
                    }
                )
            )

            checkpoint_root, anchor_version, deltas = _resolve_lineage(
                base_checkpoint_dir=str(base),
                current_checkpoint_dir=str(resident),
                source_dir=str(source),
                target_version=3,
                base_version=1,
            )
            self.assertEqual(checkpoint_root, full)
            self.assertEqual(anchor_version, 2)
            self.assertEqual([version for version, _, _ in deltas], [3])

    def test_lineage_accepts_empty_delta_before_changed_target(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            base = root / "base"
            source = root / "published"
            base.mkdir()
            (base / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": {"weight": "base.safetensors"},
                    }
                )
            )
            for version, weight_map in (
                (1, {}),
                (2, {"weight": "delta.safetensors"}),
            ):
                version_dir = source / f"weight_v{version:06d}"
                version_dir.mkdir(parents=True)
                (version_dir / "model.safetensors.index.json").write_text(
                    json.dumps(
                        {
                            "metadata": {
                                "version": f"{version:06d}",
                                "base_version": f"{version - 1:06d}",
                                "delta_encoding": "xor",
                                "compression_format": "zstd",
                                "checksum_format": "adler32",
                            },
                            "weight_map": weight_map,
                        }
                    )
                )

            checkpoint_root, anchor_version, deltas = _resolve_lineage(
                base_checkpoint_dir=str(base),
                current_checkpoint_dir=None,
                source_dir=str(source),
                target_version=2,
            )
            self.assertEqual(checkpoint_root, base)
            self.assertEqual(anchor_version, 0)
            self.assertEqual([version for version, _, _ in deltas], [1, 2])

    def test_empty_delta_target_needs_no_shared_blob_arena(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            base = root / "base"
            source = root / "published"
            target = source / "weight_v000001"
            base.mkdir()
            target.mkdir(parents=True)
            (base / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": {"weight": "base.safetensors"},
                    }
                )
            )
            (target / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "version": "000001",
                            "base_version": "000000",
                            "delta_encoding": "xor",
                            "compression_format": "zstd",
                            "checksum_format": "adler32",
                        },
                        "weight_map": {},
                    }
                )
            )

            overlay = DeltaCheckpointOverlay(
                base_checkpoint_dir=str(base),
                source_dir=str(source),
                target_version=1,
                cpu_group=None,
            )
            try:
                self.assertIsNone(overlay.arena)
                self.assertEqual(overlay.stage_stats["delta_versions"], [1])
                self.assertEqual(overlay.stage_stats["changed_tensors"], 0)
            finally:
                overlay.close()

    def test_host_shared_checkpoint_is_visible_to_all_local_ranks(self):
        with tempfile.TemporaryDirectory() as root:
            rendezvous = str(Path(root) / "gloo-rendezvous")
            torch.multiprocessing.start_processes(
                _shared_checkpoint_worker,
                args=(4, rendezvous),
                nprocs=4,
                join=True,
                start_method="fork",
            )

    def test_canonical_snapshot_population_reads_each_file_once(self):
        source = None
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            payloads = {
                "a.safetensors": bytes(range(251)) * 19,
                "b.safetensors": bytes(range(127)) * 37,
            }
            for filename, payload in payloads.items():
                (root / filename).write_bytes(payload)
            filenames = sorted(payloads)
            file_sizes, offsets, capacity, signature = _canonical_checkpoint_layout(
                root, filenames
            )
            self.assertEqual(signature[0], str(root.resolve()))

            try:
                source = HostSharedCheckpoint(
                    capacity=capacity,
                    cpu_group=None,
                )
                compiler = object.__new__(CPUWeightCache)
                compiler.host_cpu_group = None
                stats = compiler._populate_canonical_checkpoint(
                    root=root,
                    filenames=filenames,
                    file_sizes=file_sizes,
                    offsets=offsets,
                    checkpoint=source,
                )
                self.assertEqual(
                    stats["checkpoint_bytes"],
                    sum(len(payload) for payload in payloads.values()),
                )
                self.assertEqual(stats["owned_bytes"], stats["checkpoint_bytes"])
                for filename, expected in payloads.items():
                    actual = source.view(
                        len(expected),
                        offset=offsets[filename],
                    )
                    self.assertEqual(actual.numpy().tobytes(), expected)
                    del actual
            finally:
                if source is not None:
                    source.close()

    def test_resident_transform_finishes_before_runtime_compilation(self):
        class RecordingTransform:
            def __init__(self):
                self.filenames = []

            def transform_file(self, filename, tensor_file):
                self.filenames.append(filename)
                tensor = tensor_file.get_tensor("value")
                tensor.add_(1)
                return {
                    "filename": filename,
                    "wall_s": 0.25,
                }

        source = None
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            for index in range(3):
                save_file(
                    {"value": torch.tensor([index], dtype=torch.int64)},
                    root / f"{index}.safetensors",
                )
            filenames = sorted(path.name for path in root.glob("*.safetensors"))
            file_sizes, offsets, capacity, _ = _canonical_checkpoint_layout(
                root,
                filenames,
            )
            try:
                source = HostSharedCheckpoint(
                    capacity=capacity,
                    cpu_group=None,
                )
                for filename in filenames:
                    target = source.view(
                        file_sizes[filename],
                        offset=offsets[filename],
                    )
                    _pread_file_to_tensor(root / filename, target)
                    del target

                transform = RecordingTransform()
                stats = _preapply_resident_checkpoint_transform(
                    filenames=filenames,
                    file_sizes=file_sizes,
                    offsets=offsets,
                    checkpoint=source,
                    source_transform=transform,
                    rank=0,
                    world_size=1,
                    cpu_group=None,
                )

                self.assertEqual(transform.filenames, filenames)
                self.assertEqual(stats["owned_files"], len(filenames))
                self.assertEqual(stats["transform_wall_s"], 0.75)
                for index, filename in enumerate(filenames):
                    parsed = InMemorySafeTensorsFile(
                        source.view(
                            file_sizes[filename],
                            offset=offsets[filename],
                        )
                    )
                    self.assertEqual(parsed.get_tensor("value").item(), index + 1)
            finally:
                if source is not None:
                    source.close()

    def test_initialized_v0_snapshot_binds_to_first_matching_lineage(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            base = root / "base"
            source = root / "source"
            next_source = root / "next-source"
            base.mkdir()
            source.mkdir()
            next_source.mkdir()
            compiler = object.__new__(CPUWeightCache)
            compiler._canonical_checkpoint = mock.Mock()
            compiler._canonical_checkpoint_signature = (
                str(base.resolve()),
                (),
            )
            compiler._canonical_checkpoint_version = 0
            compiler._canonical_lineage = None

            version = compiler.canonical_checkpoint_version(
                base_checkpoint_dir=str(base),
                source_dir=str(source),
            )

            self.assertEqual(version, 0)
            self.assertEqual(
                compiler._canonical_lineage,
                (str(base.resolve()), str(source.resolve())),
            )
            self.assertEqual(
                compiler.canonical_checkpoint_version(
                    base_checkpoint_dir=str(base),
                    source_dir=str(next_source),
                ),
                0,
            )
            self.assertEqual(
                compiler._canonical_lineage,
                (str(base.resolve()), str(next_source.resolve())),
            )
            compiler._canonical_checkpoint.close.assert_not_called()

    def test_node_shared_delta_arena_is_visible_to_all_local_ranks(self):
        with tempfile.TemporaryDirectory() as root:
            rendezvous = str(Path(root) / "gloo-delta-rendezvous")
            torch.multiprocessing.start_processes(
                _shared_delta_worker,
                args=(4, rendezvous),
                nprocs=4,
                join=True,
                start_method="fork",
            )

    def test_clone_can_write_shared_storage_into_external_cpu_image(self):
        class SharedWeights(torch.nn.Module):
            def __init__(self):
                super().__init__()
                base = torch.arange(12, dtype=torch.float32)
                self.weight = torch.nn.Parameter(base.reshape(3, 4))
                self.register_buffer("alias", self.weight.data[1:].reshape(-1))
                self.metadata = torch.tensor([37], dtype=torch.int64)

        module = SharedWeights()
        original_weight = module.weight.detach().clone()
        image = torch.zeros(
            module.weight.untyped_storage().nbytes() + 32,
            dtype=torch.uint8,
        )
        image_buffer = memoryview(image.numpy()).cast("B")
        weight_key = (
            module.weight.untyped_storage().data_ptr(),
            module.weight.untyped_storage().nbytes(),
        )

        def storage_factory(tensor, source_bytes):
            key = (
                tensor.untyped_storage().data_ptr(),
                tensor.untyped_storage().nbytes(),
            )
            if key == weight_key:
                return torch.frombuffer(
                    image_buffer[16 : 16 + source_bytes.numel()],
                    dtype=torch.uint8,
                )
            return source_bytes.clone()

        shadow = clone_module_tensors(
            module,
            target_device=torch.device("cpu"),
            copy_data=False,
            storage_factory=storage_factory,
        )
        with torch.no_grad():
            shadow.weight.copy_(
                torch.full_like(shadow.weight, 5),
            )

        self.assertEqual(shadow.weight.untyped_storage().nbytes(), 48)
        self.assertEqual(
            shadow.alias.untyped_storage().data_ptr(),
            shadow.weight.untyped_storage().data_ptr(),
        )
        self.assertTrue(torch.all(shadow.alias == 5))
        torch.testing.assert_close(module.weight, original_weight)
        self.assertTrue(torch.all(image[16:64].view(torch.float32) == 5))
        self.assertEqual(shadow.metadata.item(), 37)

    def test_cpu_postprocessing_avoids_background_device_traffic(self):
        class CpuPostprocessor:
            @staticmethod
            def weight_preparation_device(layer):
                return "cpu"

            @staticmethod
            def process_weights_after_loading(layer):
                layer.weight = torch.nn.Parameter(
                    layer.weight + 1,
                    requires_grad=False,
                )

        image = torch.arange(8, dtype=torch.float32).view(torch.uint8).clone()
        image_buffer = memoryview(image.numpy()).cast("B")
        shadow = torch.nn.Module()
        shadow.weight = torch.nn.Parameter(
            torch.frombuffer(image_buffer, dtype=torch.float32),
            requires_grad=False,
        )
        shadow.quant_method = CpuPostprocessor()
        segment = PreparedWeightSegment(
            name="layer.weight",
            image_offset=0,
            nbytes=image.numel(),
            device_bytes=torch.empty(0, dtype=torch.uint8),
        )
        group = WeightModuleGroup(path="layer", nbytes=image.numel())
        compiler = object.__new__(CPUWeightCache)
        compiler.groups = [group]
        compiler.image = SimpleNamespace(
            image=image,
            segments_by_name={"layer.weight": segment},
        )
        loaded = CpuImageGroupLoad(
            group_index=1,
            group=group,
            checkpoint_tensors=1,
            transport="test",
            cpu_shadow=shadow,
            gpu_stage_cpu_storages={(1234, image.numel())},
            group_started=0.0,
            cpu_clone_s=0.0,
            restore_s=0.0,
            cpu_load_s=0.0,
        )

        updated, copied_bytes, stats = compiler._finalize_cpu_image_group(loaded)

        self.assertEqual(updated, {id(segment)})
        self.assertEqual(copied_bytes, image.numel())
        self.assertEqual(stats["postprocess_device"], "cpu")
        self.assertEqual(stats["background_h2d_bytes"], 0)
        self.assertEqual(stats["background_d2h_bytes"], 0)
        self.assertEqual(stats["cpu_image_copy_bytes"], image.numel())
        torch.testing.assert_close(
            image.view(torch.float32),
            torch.arange(1, 9, dtype=torch.float32),
        )

    def test_unknown_quantization_method_is_not_assumed_safe(self):
        shadow = torch.nn.Module()
        shadow.quant_method = object()

        with self.assertRaisesRegex(
            NotImplementedError,
            "unsupported for quantization method object",
        ):
            CPUWeightCache._weight_preparation_device(shadow)

    def test_modelopt_moe_batch_preserves_native_copy_views(self):
        layer = _CopyOnlyModelOptMoE()
        float_storage = torch.zeros((6, 5), dtype=torch.float32)
        float_target_1 = float_storage[1:3]
        float_target_2 = float_storage[3:6]
        float_source_1 = torch.arange(10, dtype=torch.float32).reshape(2, 5)
        float_source_2 = torch.arange(15, dtype=torch.float32).reshape(5, 3).t()
        int_target = torch.zeros(4, dtype=torch.int32)
        int_source = torch.tensor(9, dtype=torch.int32)

        layer.weight_loader_batch(
            [
                ((float_target_1, float_source_1), {}),
                ((float_target_2, float_source_2), {}),
                ((int_target, int_source), {}),
            ]
        )

        torch.testing.assert_close(float_target_1, float_source_1)
        torch.testing.assert_close(float_target_2, float_source_2)
        torch.testing.assert_close(int_target, torch.full_like(int_target, 9))

    def test_prepared_copy_workers_use_equal_local_cpu_share(self):
        with (
            mock.patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer." "os.sched_getaffinity",
                return_value=set(range(80)),
            ),
            mock.patch.object(torch.distributed, "is_initialized", return_value=True),
            mock.patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer.get_world_group",
                return_value=mock.Mock(local_size=4, world_size=4),
            ),
            mock.patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer.get_server_args",
                return_value=mock.Mock(
                    nnodes=1,
                    dp_size=1,
                    enable_dp_attention=False,
                ),
            ),
            mock.patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer."
                "envs.SGLANG_SET_CPU_AFFINITY.get",
                return_value=False,
            ),
        ):
            self.assertEqual(_prepared_weight_copy_workers(), 20)

    def test_prepared_copy_workers_derive_multinode_local_share(self):
        with (
            mock.patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer." "os.sched_getaffinity",
                return_value=set(range(80)),
            ),
            mock.patch.object(torch.distributed, "is_initialized", return_value=True),
            mock.patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer.get_world_group",
                return_value=mock.Mock(local_size=0, world_size=8),
            ),
            mock.patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer.get_server_args",
                return_value=mock.Mock(
                    nnodes=2,
                    dp_size=1,
                    enable_dp_attention=False,
                ),
            ),
            mock.patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer."
                "envs.SGLANG_SET_CPU_AFFINITY.get",
                return_value=False,
            ),
        ):
            self.assertEqual(_prepared_weight_copy_workers(), 20)

    def test_prepared_copy_workers_use_process_affinity_when_sglang_pins_cpus(self):
        with (
            mock.patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer." "os.sched_getaffinity",
                return_value=set(range(20)),
            ),
            mock.patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer."
                "envs.SGLANG_SET_CPU_AFFINITY.get",
                return_value=True,
            ),
        ):
            self.assertEqual(_prepared_weight_copy_workers(), 20)

    def test_modelopt_moe_batch_parallel_copy_preserves_native_views(self):
        layer = _CopyOnlyModelOptMoE()
        targets = [torch.zeros(256, dtype=torch.float32) for _ in range(8)]
        sources = [
            torch.full_like(target, index + 1) for index, target in enumerate(targets)
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            layer.weight_loader_batch(
                [((target, source), {}) for target, source in zip(targets, sources)],
                executor=executor,
            )

        for index, (target, source) in enumerate(zip(targets, sources)):
            with self.subTest(index=index):
                torch.testing.assert_close(target, source)

    def test_modelopt_moe_batch_orders_overlapping_destinations(self):
        layer = _CopyOnlyModelOptMoE()
        target = torch.zeros(8, dtype=torch.float32)
        first = torch.ones_like(target)
        second = torch.full_like(target, 2)
        independent_target = torch.zeros(1024, dtype=torch.float32)
        independent_source = torch.full_like(independent_target, 3)
        other_target = torch.zeros(1024, dtype=torch.float32)
        other_source = torch.full_like(other_target, 4)

        class RecordingExecutor:
            def __init__(self, delegate):
                self.delegate = delegate
                self.submissions = 0

            def submit(self, function, *args, **kwargs):
                self.submissions += 1
                return self.delegate.submit(function, *args, **kwargs)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            recording_executor = RecordingExecutor(executor)
            layer.weight_loader_batch(
                [
                    ((independent_target, independent_source), {}),
                    ((other_target, other_source), {}),
                    ((target, first), {}),
                    ((target, second), {}),
                ],
                executor=recording_executor,
            )

        self.assertGreater(recording_executor.submissions, 0)
        torch.testing.assert_close(independent_target, independent_source)
        torch.testing.assert_close(other_target, other_source)
        torch.testing.assert_close(target, second)


if __name__ == "__main__":
    unittest.main()
