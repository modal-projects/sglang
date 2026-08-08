import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

import sglang.srt.model_executor.model_runner_components.weight_updater as updater_mod
import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
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
            raise RuntimeError("target load failed")


def _make_weight_updater(model, model_config, runner, update_model_fields):
    return WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=model_config,
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=update_model_fields,
        recapture_cuda_graph=Mock(),
        get_model_runner=lambda: runner,
    )


class TestReloadLifecycle(CustomTestCase):
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

    def test_update_reuses_initial_load_config(self):
        events = []
        original_load_config = LoadConfig(
            load_format=LoadFormat.FASTSAFETENSORS,
            download_dir="/cache",
            model_loader_extra_config={"enable_gds": False},
            ignore_patterns=["unused/**"],
        )
        runner = SimpleNamespace(
            load_config=original_load_config,
            server_args=SimpleNamespace(weight_cache_mode="off"),
        )
        model = nn.Module()
        model_config = SimpleNamespace(
            model_path="/original", dtype=torch.float32, quantization=None
        )
        update_model_fields = Mock()
        updater = _make_weight_updater(model, model_config, runner, update_model_fields)
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
        runner = SimpleNamespace(
            load_config=original_load_config,
            server_args=SimpleNamespace(weight_cache_mode="off"),
        )
        model = nn.Module()
        model_config = SimpleNamespace(
            model_path="/original", dtype=torch.float32, quantization=None
        )
        update_model_fields = Mock()
        updater = _make_weight_updater(model, model_config, runner, update_model_fields)
        target_loader = _RecordingLoader(
            original_load_config, events, "target", fail=True
        )
        original_loader = _RecordingLoader(original_load_config, events, "original")

        def get_model_loader(load_config, active_model_config):
            return (
                target_loader
                if active_model_config.model_path == "/target"
                else original_loader
            )

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

    def test_partial_quantized_reload_is_rejected_before_mutation(self):
        original_load_config = LoadConfig(load_format=LoadFormat.SAFETENSORS)
        runner = SimpleNamespace(
            load_config=original_load_config,
            server_args=SimpleNamespace(weight_cache_mode="off"),
        )
        model = nn.Module()
        model_config = SimpleNamespace(
            model_path="/original", dtype=torch.float32, quantization="fp8"
        )
        updater = _make_weight_updater(model, model_config, runner, Mock())

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


if __name__ == "__main__":
    unittest.main()
