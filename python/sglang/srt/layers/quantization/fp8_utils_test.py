import unittest
from unittest.mock import patch

import torch
from compressed_tensors.quantization import QuantizationStrategy

import sglang.srt.layers.quantization.fp8_utils as fp8_utils
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_fp8 as compressed_fp8,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8,
)

BLOCK_SIZE = [128, 128]


def _make_params(n: int = 64, k: int = 128):
    weight = torch.nn.Parameter(torch.zeros((n, k)), requires_grad=False)
    weight_scale = torch.nn.Parameter(torch.ones((1, 1)), requires_grad=False)
    weight_scale.format_ue8m0 = False
    return weight, weight_scale


class TestDeepGemmUE8M0Requant(unittest.TestCase):
    def _enabled_deepgemm_ue8m0(self):
        return patch.multiple(
            deep_gemm_wrapper,
            ENABLE_JIT_DEEPGEMM=True,
            DEEPGEMM_SCALE_UE8M0=True,
        )

    def test_compressed_tensors_block_processing_preserves_ue8m0_marker(self):
        scheme = CompressedTensorsW8A8Fp8.__new__(CompressedTensorsW8A8Fp8)
        scheme.strategy = QuantizationStrategy.BLOCK
        scheme.is_static_input_scheme = False
        scheme.weight_block_size = BLOCK_SIZE
        scheme.w8a8_block_fp8_linear = (
            fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
        )

        layer = torch.nn.Module()
        layer.weight, layer.weight_scale = _make_params()
        layer.orig_dtype = torch.bfloat16

        with (
            self._enabled_deepgemm_ue8m0(),
            patch.object(fp8_utils, "requant_weight_ue8m0_inplace") as requant,
        ):
            scheme.process_weights_after_loading(layer)
            scheme.process_weights_after_loading(layer)

            self.assertTrue(layer.weight_scale.format_ue8m0)
            requant.assert_called_once()

            scheme.restore_weights_before_loading(layer)
            self.assertFalse(layer.weight_scale.format_ue8m0)
            scheme.process_weights_after_loading(layer)
            self.assertEqual(requant.call_count, 2)

    def test_reload_reuses_fp8_runtime_buffers(self):
        weight = torch.nn.Parameter(torch.zeros((2, 2)), requires_grad=False)
        scale = torch.nn.Parameter(torch.ones((1, 1)), requires_grad=False)
        scale.format_ue8m0 = False
        fp8_utils.record_ue8m0_scale_checkpoint_layout(scale)

        runtime_scale = torch.zeros((4, 2), dtype=torch.int32)
        scale.data = runtime_scale
        scale.format_ue8m0 = True
        fp8_utils.restore_ue8m0_scale_checkpoint_layout(scale)

        self.assertEqual(scale.shape, (1, 1))
        self.assertEqual(scale.dtype, torch.float32)
        self.assertFalse(scale.format_ue8m0)

        weight_ptr = weight.data_ptr()
        runtime_scale_ptr = runtime_scale.data_ptr()
        new_weight = torch.full_like(weight, 3)
        new_runtime_scale = torch.full_like(runtime_scale, 7)
        with patch.object(
            fp8_utils,
            "requant_weight_ue8m0",
            return_value=(new_weight, new_runtime_scale),
        ):
            fp8_utils.requant_weight_ue8m0_inplace(weight, scale, BLOCK_SIZE)

        self.assertEqual(weight.data_ptr(), weight_ptr)
        self.assertEqual(scale.data_ptr(), runtime_scale_ptr)
        torch.testing.assert_close(weight, new_weight)
        torch.testing.assert_close(scale, new_runtime_scale)
        self.assertFalse(hasattr(scale, "_reload_runtime_scale"))

    def test_reload_can_restart_after_incomplete_scale_restore(self):
        scale = torch.nn.Parameter(torch.ones((1, 1)), requires_grad=False)
        scale.format_ue8m0 = False
        fp8_utils.record_ue8m0_scale_checkpoint_layout(scale)

        runtime_scale = torch.zeros((4, 2), dtype=torch.int32)
        scale.data = runtime_scale
        scale.format_ue8m0 = True

        fp8_utils.restore_ue8m0_scale_checkpoint_layout(scale)
        scale.data.fill_(7)
        fp8_utils.restore_ue8m0_scale_checkpoint_layout(scale)

        self.assertEqual(scale.shape, (1, 1))
        self.assertEqual(scale.dtype, torch.float32)
        self.assertFalse(scale.format_ue8m0)
        self.assertEqual(
            scale._reload_runtime_scale.data_ptr(), runtime_scale.data_ptr()
        )
        torch.testing.assert_close(runtime_scale, torch.zeros_like(runtime_scale))

    def test_compressed_tensors_channel_weight_restores_checkpoint_layout(self):
        scheme = CompressedTensorsW8A8Fp8.__new__(CompressedTensorsW8A8Fp8)
        scheme.strategy = QuantizationStrategy.CHANNEL
        scheme.is_static_input_scheme = False
        scheme.weight_block_size = None

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.arange(6, dtype=torch.float32).reshape(2, 3),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(torch.ones((2, 1)), requires_grad=False)
        checkpoint_weight = layer.weight

        with (
            patch.object(compressed_fp8, "is_fp8_fnuz", return_value=False),
            patch.object(compressed_fp8, "_use_aiter", False),
        ):
            scheme.process_weights_after_loading(layer)

        self.assertIs(layer.weight, checkpoint_weight)
        self.assertEqual(layer.weight.data_ptr(), checkpoint_weight.data_ptr())
        self.assertEqual(layer.weight.shape, (3, 2))
        torch.testing.assert_close(
            layer.weight,
            torch.arange(6, dtype=torch.float32).reshape(2, 3).t(),
        )

        layer.weight.data.fill_(5)

        with patch.object(
            compressed_fp8, "apply_fp8_linear", return_value=torch.empty(0)
        ) as apply_fp8:
            scheme.apply_weights(layer, torch.ones(1, 3))
            self.assertIs(apply_fp8.call_args.kwargs["weight"], layer.weight)

            scheme.apply_weights(
                layer,
                (torch.ones(1, 3), torch.ones(1), torch.bfloat16),
            )
            self.assertIs(apply_fp8.call_args.kwargs["weight"], layer.weight)

        with (
            patch.object(compressed_fp8, "is_fp8_fnuz", return_value=False),
            patch.object(compressed_fp8, "_use_aiter", False),
        ):
            scheme.restore_weights_before_loading(layer)
            self.assertEqual(layer.weight.shape, (2, 3))
            self.assertEqual(layer.weight.data_ptr(), checkpoint_weight.data_ptr())

            layer.weight.data.copy_(
                torch.arange(6, dtype=torch.float32).reshape(2, 3) + 10
            )
            scheme.process_weights_after_loading(layer)
            scheme.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.shape, (3, 2))
        self.assertEqual(layer.weight.data_ptr(), checkpoint_weight.data_ptr())
        torch.testing.assert_close(
            layer.weight,
            (torch.arange(6, dtype=torch.float32).reshape(2, 3) + 10).t(),
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestInverseTransformScaleUe8m0(unittest.TestCase):
    def test_round_trip_with_partial_final_block(self):
        mn = 2624
        k_blocks = 48
        exponents = torch.randint(
            1,
            255,
            ((mn + 127) // 128, k_blocks),
            dtype=torch.int32,
            device="cuda",
        )
        sf_fp32_original = (exponents << 23).view(torch.float32)

        sf_packed_original = fp8_utils.transform_scale_ue8m0(sf_fp32_original, mn=mn)
        sf_fp32_recreated = fp8_utils.inverse_transform_scale_ue8m0(
            sf_packed_original, mn=mn
        )

        self.assertTrue(torch.equal(sf_fp32_original, sf_fp32_recreated))
