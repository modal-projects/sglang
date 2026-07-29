import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    record_ue8m0_scale_checkpoint_layout,
    restore_ue8m0_scale_checkpoint_layout,
)
from sglang.srt.layers.quantization.mxfp4 import (
    Mxfp4MoEMethod,
    _compose_trtllm_gate_up_permutation,
)
from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    Mxfp4FlashinferTrtllmMoEMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_method() -> Mxfp4MoEMethod:
    method = object.__new__(Mxfp4MoEMethod)
    method.use_marlin = False
    method.use_flashinfer = True
    method.use_deep_gemm = False
    method._fi_kernel = "trtllm_sm100"
    method.num_experts = 2
    method.intermediate_size_per_partition = 128
    method.hidden_size = 128
    return method


def _make_layer(seed: int) -> torch.nn.Module:
    generator = torch.Generator().manual_seed(seed)
    layer = torch.nn.Module()
    layer.num_local_experts = 2
    layer.moe_runner_config = SimpleNamespace(gate_up_interleaved=True)

    def parameter(shape, dtype):
        if dtype == torch.bfloat16:
            value = torch.randn(shape, generator=generator, dtype=dtype)
        else:
            value = torch.randint(
                0,
                256,
                shape,
                generator=generator,
                dtype=dtype,
            )
        return torch.nn.Parameter(value, requires_grad=False)

    layer.w13_weight = parameter((2, 256, 64), torch.uint8)
    layer.w13_weight_scale = parameter((2, 256, 4), torch.uint8)
    layer.w13_weight_bias = parameter((2, 256), torch.bfloat16)
    layer.w2_weight = parameter((2, 128, 64), torch.uint8)
    layer.w2_weight_scale = parameter((2, 128, 4), torch.uint8)
    layer.w2_weight_bias = parameter((2, 128), torch.bfloat16)
    return layer


def _runtime_state(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone() for name, parameter in layer.named_parameters()
    }


class _CheckpointFp4Method:
    @staticmethod
    def restore_weights_before_loading(layer):
        restore_ue8m0_scale_checkpoint_layout(layer.w13_weight_scale_inv)
        restore_ue8m0_scale_checkpoint_layout(layer.w2_weight_scale_inv)

    @staticmethod
    def process_weights_after_loading(layer):
        record_ue8m0_scale_checkpoint_layout(layer.w13_weight_scale_inv)
        record_ue8m0_scale_checkpoint_layout(layer.w2_weight_scale_inv)
        layer.w13_weight.data = layer.w13_weight.data.view(torch.int8)
        layer.w2_weight.data = layer.w2_weight.data.view(torch.int8)


def _make_hybrid_method() -> Mxfp4FlashinferTrtllmMoEMethod:
    method = object.__new__(Mxfp4FlashinferTrtllmMoEMethod)
    method._fp8 = _CheckpointFp4Method()
    method.prefix = "test.experts"
    return method


def _make_hybrid_layer(seed: int) -> torch.nn.Module:
    generator = torch.Generator().manual_seed(seed)
    layer = torch.nn.Module()
    layer.num_local_experts = 2

    def parameter(shape, dtype):
        if dtype == torch.float32:
            value = torch.rand(shape, generator=generator, dtype=dtype) * 1.75
        else:
            value = torch.randint(
                -128,
                128,
                shape,
                generator=generator,
                dtype=dtype,
            )
        result = torch.nn.Parameter(value, requires_grad=False)
        result.weight_loader = object()
        if dtype == torch.float32:
            result.format_ue8m0 = False
        return result

    layer.w13_weight = parameter((2, 256, 64), torch.int8)
    layer.w13_weight_scale_inv = parameter((2, 256, 4), torch.float32)
    layer.w2_weight = parameter((2, 128, 64), torch.int8)
    layer.w2_weight_scale_inv = parameter((2, 128, 4), torch.float32)
    return layer


class TestMxfp4Reload(unittest.TestCase):
    def test_gate_up_reordering_is_composed_with_runtime_permutation(self):
        indices = torch.tensor([3, 0, 2, 1])
        rows = torch.arange(4)

        interleaved = _compose_trtllm_gate_up_permutation(
            indices,
            gate_up_interleaved=True,
        )
        torch.testing.assert_close(
            rows.index_select(0, interleaved),
            rows.view(2, 2).flip(1).reshape(-1).index_select(0, indices),
        )

        blocked_gate_up = torch.tensor([2, 0, 3, 1])
        blocked = _compose_trtllm_gate_up_permutation(
            indices,
            gate_up_interleaved=False,
        )
        torch.testing.assert_close(
            rows.index_select(0, blocked),
            rows.index_select(0, blocked_gate_up).index_select(0, indices),
        )

    def test_repeated_reload_restores_checkpoint_scale_bytes(self):
        method = _make_method()
        layer = _make_layer(seed=1)
        target = _make_layer(seed=2)
        reference = _make_layer(seed=2)

        method.process_weights_after_loading(layer)
        runtime_pointers = {
            name: parameter.data_ptr() for name, parameter in layer.named_parameters()
        }

        method.restore_weights_before_loading(layer)
        self.assertEqual(layer.w13_weight_scale.dtype, torch.uint8)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.uint8)
        for name in (
            "w13_weight",
            "w13_weight_scale",
            "w13_weight_bias",
            "w2_weight",
            "w2_weight_scale",
            "w2_weight_bias",
        ):
            self.assertEqual(torch.count_nonzero(getattr(layer, name)), 0)
            getattr(layer, name).data.copy_(getattr(target, name))

        method.process_weights_after_loading(layer)
        method.process_weights_after_loading(reference)

        for name, expected in _runtime_state(reference).items():
            with self.subTest(name=name):
                actual = getattr(layer, name)
                if actual.dtype == torch.float8_e4m3fn:
                    torch.testing.assert_close(
                        actual.view(torch.uint8),
                        expected.view(torch.uint8),
                    )
                else:
                    torch.testing.assert_close(actual, expected)
                self.assertEqual(
                    actual.data_ptr(),
                    runtime_pointers[name],
                )

    def test_cpu_postprocessing_capability_is_backend_specific(self):
        method = _make_method()
        self.assertEqual(
            method.weight_update_postprocess_device(torch.nn.Module()),
            "cpu",
        )

        method._fi_kernel = "cutlass_sm90"
        self.assertIsNone(method.weight_update_postprocess_device(torch.nn.Module()))

    def test_hybrid_mxfp4_repeated_reload_restores_checkpoint_layout(self):
        method = _make_hybrid_method()
        layer = _make_hybrid_layer(seed=3)
        target = _make_hybrid_layer(seed=4)
        reference = _make_hybrid_layer(seed=4)

        method.process_weights_after_loading(layer)
        runtime_pointers = {
            name: parameter.data_ptr() for name, parameter in layer.named_parameters()
        }

        method.restore_weights_before_loading(layer)
        self.assertEqual(layer.w13_weight.dtype, torch.int8)
        self.assertEqual(layer.w2_weight.dtype, torch.int8)
        self.assertEqual(layer.w13_weight_scale_inv.dtype, torch.float32)
        self.assertEqual(layer.w2_weight_scale_inv.dtype, torch.float32)
        for name, parameter in target.named_parameters():
            getattr(layer, name).data.copy_(parameter)

        method.process_weights_after_loading(layer)
        method.process_weights_after_loading(reference)

        for name, expected in _runtime_state(reference).items():
            with self.subTest(name=name):
                actual = getattr(layer, name)
                if actual.dtype == torch.float8_e4m3fn:
                    torch.testing.assert_close(
                        actual.view(torch.uint8),
                        expected.view(torch.uint8),
                    )
                else:
                    torch.testing.assert_close(actual, expected)
                self.assertEqual(actual.data_ptr(), runtime_pointers[name])
                self.assertTrue(hasattr(actual, "weight_loader"))

    def test_hybrid_mxfp4_rejects_incompatible_runtime_buffer(self):
        layer = torch.nn.Module()
        parameter = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
        parameter._weight_update_runtime_buffer = torch.zeros(5)
        layer.register_parameter("weight", parameter)

        with self.assertRaisesRegex(RuntimeError, "differs from its runtime buffer"):
            Mxfp4FlashinferTrtllmMoEMethod._publish_runtime_parameter(
                layer,
                "weight",
                torch.zeros(4),
            )


if __name__ == "__main__":
    unittest.main()
