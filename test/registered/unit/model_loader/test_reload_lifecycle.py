import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

import sglang.srt.model_executor.model_runner_components.weight_updater as updater_mod
import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _RecordingQuantMethod(QuantizeMethodBase):
    def __init__(self, events):
        self.events = events

    def apply(self, layer, x):
        return x

    def restore_weights_before_loading(self, layer):
        self.events.append("restore")

    def process_weights_after_loading(self, layer):
        self.events.append("postprocess")


class _LifecycleModel(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.layer = nn.Linear(1, 1, bias=False)
        self.layer.quant_method = _RecordingQuantMethod(events)
        self.events = events

    def load_weights(self, weights):
        list(weights)
        self.events.append("load")


class _RecordingLoader(DefaultModelLoader):
    def __init__(self, load_config, events, name, *, fail=False):
        super().__init__(load_config)
        self.events = events
        self.name = name
        self.fail = fail

    def restore_weights_before_loading(self, model, target_device):
        self.events.append(f"{self.name}.restore")

    def _get_all_weights(self, model_config, model):
        self.events.append(f"{self.name}.weights:{model_config.model_path}")
        return iter(())

    def load_weights_and_postprocess(self, model, weights, target_device):
        self.events.append(f"{self.name}.load")
        if self.fail:
            raise RuntimeError(f"{self.name} load failed")


class _RecordingStager:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.events = []

    def initialize(self, checkpoint_dir, *, version):
        self.events.append(("initialize", checkpoint_dir, version))
        return {"operation": "initialize", "version": version}

    def stage(self, *, checkpoint_source_dir, target_version):
        self.events.append(("stage", checkpoint_source_dir, target_version))
        return {"operation": "stage", "target_version": target_version}

    def validate_commit(self, target_version):
        self.events.append(("validate", target_version))

    def commit(self, target_version):
        self.events.append(("commit", target_version))
        return {"operation": "commit", "target_version": target_version}

    def discard_prepared(self, reason):
        self.events.append(("discard", reason))

    def close(self):
        self.events.append(("close",))


def _make_weight_updater(model, model_config, runner, update_model_fields=None):
    return WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=model_config,
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=update_model_fields or Mock(),
        recapture_cuda_graph=Mock(),
        get_model_runner=lambda: runner,
    )


class TestReloadLifecycle(CustomTestCase):
    def setUp(self):
        super().setUp()
        weight_cache = patch.object(
            updater_mod,
            "get_model",
            return_value=SimpleNamespace(weight_cache_mode="off"),
        )
        weight_cache.start()
        self.addCleanup(weight_cache.stop)

    def test_restore_load_postprocess_order(self):
        events = []
        model = _LifecycleModel(events)

        with patch.object(loader_mod, "is_cuda_alike", return_value=False):
            DefaultModelLoader.restore_weights_before_loading(
                model, torch.device("cpu")
            )
            DefaultModelLoader.load_weights_and_postprocess(
                model, iter(()), torch.device("cpu")
            )

        self.assertEqual(events, ["restore", "load", "postprocess"])

    def test_compressed_tensor_method_delegates_restore(self):
        layer = nn.Module()
        layer.scheme = Mock()
        method = CompressedTensorsLinearMethod(None)

        method.restore_weights_before_loading(layer)

        layer.scheme.restore_weights_before_loading.assert_called_once_with(layer)

    def test_update_reuses_initial_load_config(self):
        events = []
        original_load_config = LoadConfig(
            load_format=LoadFormat.FASTSAFETENSORS,
            download_dir="/cache",
            model_loader_extra_config={"enable_gds": False},
            ignore_patterns=["unused/**"],
        )
        runner = SimpleNamespace(load_config=original_load_config)
        model_config = SimpleNamespace(
            model_path="/original", dtype=torch.float32, quantization=None
        )
        update_model_fields = Mock()
        updater = _make_weight_updater(
            nn.Module(), model_config, runner, update_model_fields
        )
        target_loader = _RecordingLoader(original_load_config, events, "target")

        with (
            patch.object(
                updater_mod,
                "_unsupported_derived_weight_cache_error",
                return_value=None,
            ),
            patch.object(updater_mod, "get_available_gpu_memory", return_value=1.0),
            patch.object(
                updater_mod, "get_model_loader", return_value=target_loader
            ) as get_model_loader,
        ):
            success, _ = updater.update_weights_from_disk(
                "/target", LoadFormat.FASTSAFETENSORS
            )

        self.assertTrue(success)
        self.assertEqual(
            events,
            ["target.restore", "target.weights:/target", "target.load"],
        )
        load_config = get_model_loader.call_args.args[0]
        self.assertIsNot(load_config, original_load_config)
        self.assertEqual(load_config.load_format, LoadFormat.FASTSAFETENSORS)
        self.assertEqual(load_config.download_dir, "/cache")
        self.assertEqual(load_config.model_loader_extra_config, {"enable_gds": False})
        self.assertEqual(load_config.ignore_patterns, ["unused/**"])
        self.assertEqual(model_config.model_path, "/target")
        update_model_fields.assert_called_once()

    def test_failed_update_restores_original_checkpoint(self):
        events = []
        original_load_config = LoadConfig(load_format=LoadFormat.SAFETENSORS)
        runner = SimpleNamespace(load_config=original_load_config)
        model_config = SimpleNamespace(
            model_path="/original", dtype=torch.float32, quantization=None
        )
        update_model_fields = Mock()
        updater = _make_weight_updater(
            nn.Module(), model_config, runner, update_model_fields
        )
        target_loader = _RecordingLoader(
            original_load_config, events, "target", fail=True
        )
        original_loader = _RecordingLoader(original_load_config, events, "original")

        def get_model_loader(load_config, active_model_config):
            if active_model_config.model_path == "/target":
                return target_loader
            return original_loader

        with (
            patch.object(
                updater_mod,
                "_unsupported_derived_weight_cache_error",
                return_value=None,
            ),
            patch.object(updater_mod, "get_available_gpu_memory", return_value=1.0),
            patch.object(updater_mod, "get_model_loader", get_model_loader),
        ):
            success, message = updater.update_weights_from_disk(
                "/target", LoadFormat.SAFETENSORS
            )

        self.assertFalse(success)
        self.assertIn("Rolled back to the original weights", message)
        self.assertEqual(
            events,
            [
                "target.restore",
                "target.weights:/target",
                "target.load",
                "original.restore",
                "original.weights:/original",
                "original.load",
            ],
        )
        self.assertEqual(model_config.model_path, "/original")
        update_model_fields.assert_not_called()

    def test_failed_update_and_rollback_terminate_the_engine(self):
        events = []
        original_load_config = LoadConfig(load_format=LoadFormat.SAFETENSORS)
        runner = SimpleNamespace(load_config=original_load_config)
        model_config = SimpleNamespace(
            model_path="/original", dtype=torch.float32, quantization=None
        )
        updater = _make_weight_updater(nn.Module(), model_config, runner)
        target_loader = _RecordingLoader(
            original_load_config, events, "target", fail=True
        )
        original_loader = _RecordingLoader(
            original_load_config, events, "original", fail=True
        )

        def get_model_loader(load_config, active_model_config):
            if active_model_config.model_path == "/target":
                return target_loader
            return original_loader

        with (
            patch.object(
                updater_mod,
                "_unsupported_derived_weight_cache_error",
                return_value=None,
            ),
            patch.object(updater_mod, "get_available_gpu_memory", return_value=1.0),
            patch.object(updater_mod, "get_model_loader", get_model_loader),
            self.assertRaisesRegex(
                RuntimeError,
                "terminating the engine to avoid serving a partially updated model",
            ),
        ):
            updater.update_weights_from_disk("/target", LoadFormat.SAFETENSORS)

        self.assertEqual(model_config.model_path, "/original")

    def test_failed_update_from_active_path_terminates_without_rollback(self):
        events = []
        original_load_config = LoadConfig(load_format=LoadFormat.SAFETENSORS)
        runner = SimpleNamespace(load_config=original_load_config)
        model_config = SimpleNamespace(
            model_path="/mutable", dtype=torch.float32, quantization=None
        )
        updater = _make_weight_updater(nn.Module(), model_config, runner)
        loader = _RecordingLoader(original_load_config, events, "target", fail=True)

        with (
            patch.object(
                updater_mod,
                "_unsupported_derived_weight_cache_error",
                return_value=None,
            ),
            patch.object(updater_mod, "get_available_gpu_memory", return_value=1.0),
            patch.object(updater_mod, "get_model_loader", return_value=loader),
            self.assertRaisesRegex(RuntimeError, "unavailable for rollback"),
        ):
            updater.update_weights_from_disk("/mutable", LoadFormat.SAFETENSORS)

        self.assertEqual(
            events,
            ["target.restore", "target.weights:/mutable", "target.load"],
        )

    def test_partial_quantized_reload_is_rejected_before_mutation(self):
        original_load_config = LoadConfig(load_format=LoadFormat.SAFETENSORS)
        runner = SimpleNamespace(load_config=original_load_config)
        model_config = SimpleNamespace(
            model_path="/original", dtype=torch.float32, quantization="fp8"
        )
        updater = _make_weight_updater(nn.Module(), model_config, runner)

        with (
            patch.object(
                updater_mod,
                "_unsupported_derived_weight_cache_error",
                return_value=None,
            ),
            patch.object(updater_mod, "get_available_gpu_memory", return_value=1.0),
            patch.object(updater_mod, "get_model_loader") as get_model_loader,
        ):
            success, message = updater.update_weights_from_disk(
                "/target",
                LoadFormat.SAFETENSORS,
                weight_name_filter=lambda _: True,
            )

        self.assertFalse(success)
        self.assertIn("weight_name_filter is not supported", message)
        self.assertEqual(model_config.model_path, "/original")
        get_model_loader.assert_not_called()

    def test_rank_weight_stager_owns_prepare_and_commit(self):
        model = nn.Module()
        runner = SimpleNamespace(
            is_draft_worker=False,
            rank_weight_stager=None,
            forward_stream=object(),
        )
        model_config = SimpleNamespace(model_path="/checkpoint")
        updater = WeightUpdater(
            tp_rank=0,
            device="cuda",
            gpu_id=0,
            model_config=model_config,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=Mock(),
            recapture_cuda_graph=Mock(),
            get_model_runner=lambda: runner,
        )

        with (
            patch.object(updater_mod, "RankWeightStager", _RecordingStager),
            patch.object(
                updater_mod,
                "_unsupported_derived_weight_cache_error",
                return_value=None,
            ),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
            patch.object(torch.cuda, "synchronize"),
        ):
            stats = updater.initialize_rank_weight_stager(
                checkpoint_dir="/checkpoint",
                version=3,
                host_group=None,
                max_compile_group_bytes=1024,
                canonical_checkpoint_dir="/canonical",
            )
            self.assertEqual(stats["version"], 3)

            stager = runner.rank_weight_stager
            self.assertEqual(
                stager.kwargs,
                {
                    "max_compile_group_bytes": 1024,
                    "host_group": None,
                    "cuda_stream": runner.forward_stream,
                    "canonical_checkpoint_dir": "/canonical",
                },
            )
            self.assertEqual(
                updater.stage_rank_weight_update(
                    checkpoint_source_dir="/updates",
                    target_version=5,
                )["target_version"],
                5,
            )
            updater.validate_rank_weight_commit(5)
            self.assertEqual(updater.commit_rank_weight_update(5)["target_version"], 5)
            updater.discard_prepared_rank_weights("distributed failure")
            updater.close_rank_weight_stager()

        self.assertIsNone(runner.rank_weight_stager)
        self.assertEqual(
            stager.events,
            [
                ("initialize", "/checkpoint", 3),
                ("stage", "/updates", 5),
                ("validate", 5),
                ("commit", 5),
                ("discard", "distributed failure"),
                ("close",),
            ],
        )

    def test_stager_rejects_out_of_band_weight_mutation(self):
        runner = SimpleNamespace(rank_weight_stager=object())
        updater = _make_weight_updater(
            nn.Module(),
            SimpleNamespace(model_path="/checkpoint"),
            runner,
        )

        with self.assertRaisesRegex(RuntimeError, "out-of-band mutation"):
            updater._assert_direct_update_allowed("update_weights_from_tensor")


if __name__ == "__main__":
    unittest.main()
