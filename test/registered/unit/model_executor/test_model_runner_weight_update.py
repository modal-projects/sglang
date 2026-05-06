import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.srt.model_loader.lora_merge_loader import merge_lora_tensors_inplace
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_executor.model_runner import ModelRunner


class DummyWeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)

    def load_weights(self, named_tensors):
        named_tensor_map = dict(named_tensors)
        if "weight" in named_tensor_map:
            self.weight.data = named_tensor_map["weight"].clone()


class TestModelRunnerWeightUpdate(unittest.TestCase):
    def _new_runner(self) -> ModelRunner:
        runner = ModelRunner.__new__(ModelRunner)
        runner.device = "cuda"
        runner.tp_rank = 0
        runner.server_args = SimpleNamespace(custom_weight_loader=["dummy.loader"])
        runner.model = object()
        runner.graph_runner = None
        runner.piecewise_cuda_graph_runner = None
        runner._call_custom_weight_loader = MagicMock()
        runner.rebuild_device_graphs_after_weight_update = MagicMock()
        return runner

    def test_custom_loader_value_error_propagates_without_legacy_retry(self):
        runner = self._new_runner()
        runner.model = object()
        calls = []

        def custom_loader(model, named_tensors, *, load_context=None):
            calls.append(load_context)
            raise ValueError("loader failed")

        with self.assertRaisesRegex(ValueError, "loader failed"):
            ModelRunner._call_custom_weight_loader(
                runner,
                custom_loader,
                named_tensors=[],
                infered_device="cuda:0",
                manifest={"x": 1},
            )

        self.assertEqual(len(calls), 1)

    def test_lora_merge_loader_does_not_force_recapture_for_value_only_updates(self):
        self.assertFalse(
            getattr(
                merge_lora_tensors_inplace,
                "sglang_requires_cuda_graph_recapture",
                False,
            )
        )

    @patch(
        "sglang.srt.model_executor.model_runner._unwrap_tensor",
        side_effect=lambda tensor, tp_rank, device=None: tensor,
    )
    @patch("sglang.srt.model_executor.model_runner.dynamic_import")
    @patch("sglang.srt.model_executor.model_runner.torch.get_device_module")
    def test_custom_loader_auto_recaptures_when_declared(
        self, mock_get_device_module, mock_dynamic_import, mock_unwrap_tensor
    ):
        custom_loader = MagicMock()
        custom_loader.sglang_supports_host_tensors = True
        custom_loader.sglang_requires_cuda_graph_recapture = True
        mock_dynamic_import.return_value = custom_loader
        mock_get_device_module.return_value = SimpleNamespace(
            current_device=lambda: "cuda:0"
        )

        runner = self._new_runner()
        tensor = torch.ones(1)

        success, message = ModelRunner.update_weights_from_tensor(
            runner,
            named_tensors=[("weight", tensor)],
            load_format="dummy.loader",
        )

        self.assertTrue(success)
        self.assertEqual(message, "Success")
        runner._call_custom_weight_loader.assert_called_once()
        runner.rebuild_device_graphs_after_weight_update.assert_called_once()
        self.assertEqual(mock_unwrap_tensor.call_count, 1)

    @patch(
        "sglang.srt.model_executor.model_runner._unwrap_tensor",
        side_effect=lambda tensor, tp_rank, device=None: tensor,
    )
    @patch("sglang.srt.model_executor.model_runner.dynamic_import")
    @patch("sglang.srt.model_executor.model_runner.torch.get_device_module")
    def test_request_flag_recaptures_even_without_loader_opt_in(
        self, mock_get_device_module, mock_dynamic_import, mock_unwrap_tensor
    ):
        custom_loader = MagicMock()
        custom_loader.sglang_supports_host_tensors = True
        custom_loader.sglang_requires_cuda_graph_recapture = False
        mock_dynamic_import.return_value = custom_loader
        mock_get_device_module.return_value = SimpleNamespace(
            current_device=lambda: "cuda:0"
        )

        runner = self._new_runner()
        tensor = torch.ones(1)

        success, message = ModelRunner.update_weights_from_tensor(
            runner,
            named_tensors=[("weight", tensor)],
            load_format="dummy.loader",
            recapture_cuda_graph=True,
        )

        self.assertTrue(success)
        self.assertEqual(message, "Success")
        runner._call_custom_weight_loader.assert_called_once()
        runner.rebuild_device_graphs_after_weight_update.assert_called_once()
        self.assertEqual(mock_unwrap_tensor.call_count, 1)

    @patch(
        "sglang.srt.model_executor.model_runner._unwrap_tensor",
        side_effect=lambda tensor, tp_rank, device=None: tensor,
    )
    @patch("sglang.srt.model_executor.model_runner.torch.get_device_module")
    def test_tensor_update_auto_recaptures_when_storage_changes(
        self, mock_get_device_module, mock_unwrap_tensor
    ):
        mock_get_device_module.return_value = SimpleNamespace(
            current_device=lambda: "cpu"
        )

        runner = self._new_runner()
        runner.device = "cpu"
        runner.server_args = SimpleNamespace(custom_weight_loader=[])
        runner.model = DummyWeightModel()
        runner.graph_runner = object()
        runner.piecewise_cuda_graph_runner = None

        success, message = ModelRunner.update_weights_from_tensor(
            runner,
            named_tensors=[("weight", torch.full((1,), 2.0))],
            load_format=None,
        )

        self.assertTrue(success)
        self.assertEqual(message, "Success")
        runner.rebuild_device_graphs_after_weight_update.assert_called_once()
        self.assertEqual(mock_unwrap_tensor.call_count, 1)

    @patch("sglang.srt.model_executor.model_runner.get_available_gpu_memory")
    @patch("sglang.srt.model_executor.model_runner.get_model_loader")
    @patch("sglang.srt.model_executor.model_runner.set_default_torch_dtype")
    def test_disk_update_skips_recapture_when_storage_stays_stable(
        self,
        mock_set_default_torch_dtype,
        mock_get_model_loader,
        mock_get_available_gpu_memory,
    ):
        mock_get_available_gpu_memory.return_value = 1.0
        mock_set_default_torch_dtype.return_value = nullcontext()

        loader = DefaultModelLoader.__new__(DefaultModelLoader)
        loader._get_weights_iterator = MagicMock(return_value=iter([]))

        def _load_weights_and_postprocess(model, weights, target_device):
            model.weight.data.copy_(torch.full_like(model.weight.data, 3.0))

        loader.load_weights_and_postprocess = MagicMock(
            side_effect=_load_weights_and_postprocess
        )
        mock_get_model_loader.return_value = loader

        runner = self._new_runner()
        runner.device = "cpu"
        runner.graph_runner = object()
        runner.piecewise_cuda_graph_runner = None
        runner.gpu_id = 0
        runner.model = DummyWeightModel()
        runner.model._sglang_cuda_graph_recapture_required = True
        runner.model_config = SimpleNamespace(
            model_path="old",
            revision=None,
            dtype=torch.float32,
        )
        runner.server_args = SimpleNamespace(
            model_path="old",
            load_format="auto",
            custom_weight_loader=[],
        )

        success, message = ModelRunner.update_weights_from_disk(
            runner,
            model_path="new",
            load_format="auto",
        )

        self.assertTrue(success)
        self.assertEqual(message, "Succeeded to update model weights.")
        runner.rebuild_device_graphs_after_weight_update.assert_not_called()
        self.assertFalse(runner.model._sglang_cuda_graph_recapture_required)

    @patch("sglang.srt.model_executor.model_runner.get_available_gpu_memory")
    @patch("sglang.srt.model_executor.model_runner.get_model_loader")
    @patch("sglang.srt.model_executor.model_runner.set_default_torch_dtype")
    def test_disk_update_auto_recaptures_when_loader_marks_graph_unsafe_module(
        self,
        mock_set_default_torch_dtype,
        mock_get_model_loader,
        mock_get_available_gpu_memory,
    ):
        mock_get_available_gpu_memory.return_value = 1.0
        mock_set_default_torch_dtype.return_value = nullcontext()

        loader = DefaultModelLoader.__new__(DefaultModelLoader)
        loader._get_weights_iterator = MagicMock(return_value=iter([]))

        def _load_weights_and_postprocess(model, weights, target_device):
            model._sglang_cuda_graph_recapture_required = True
            model.weight.data.copy_(torch.full_like(model.weight.data, 4.0))

        loader.load_weights_and_postprocess = MagicMock(
            side_effect=_load_weights_and_postprocess
        )
        mock_get_model_loader.return_value = loader

        runner = self._new_runner()
        runner.device = "cpu"
        runner.graph_runner = object()
        runner.piecewise_cuda_graph_runner = None
        runner.gpu_id = 0
        runner.model = DummyWeightModel()
        runner.model_config = SimpleNamespace(
            model_path="old",
            revision=None,
            dtype=torch.float32,
        )
        runner.server_args = SimpleNamespace(
            model_path="old",
            load_format="auto",
            custom_weight_loader=[],
        )

        success, message = ModelRunner.update_weights_from_disk(
            runner,
            model_path="new",
            load_format="auto",
        )

        self.assertTrue(success)
        self.assertEqual(message, "Succeeded to update model weights.")
        runner.rebuild_device_graphs_after_weight_update.assert_called_once()

    @patch("sglang.srt.model_executor.model_runner.gc.collect")
    @patch("sglang.srt.model_executor.model_runner.torch.get_device_module")
    def test_rebuild_device_graphs_after_weight_update_reinitializes_both_graphs(
        self, mock_get_device_module, mock_gc_collect
    ):
        empty_cache = MagicMock()
        mock_get_device_module.return_value = SimpleNamespace(empty_cache=empty_cache)

        runner = ModelRunner.__new__(ModelRunner)
        runner.device = "cuda"
        runner.graph_runner = object()
        runner.piecewise_cuda_graph_runner = object()
        runner.graph_mem_usage = 123
        runner.init_device_graphs = MagicMock()
        runner.init_piecewise_cuda_graphs = MagicMock()

        ModelRunner.rebuild_device_graphs_after_weight_update(runner)

        self.assertIsNone(runner.graph_runner)
        self.assertIsNone(runner.piecewise_cuda_graph_runner)
        self.assertEqual(runner.graph_mem_usage, 0)
        mock_gc_collect.assert_called_once()
        empty_cache.assert_called_once()
        runner.init_device_graphs.assert_called_once()
        runner.init_piecewise_cuda_graphs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
