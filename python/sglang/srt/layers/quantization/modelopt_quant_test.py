import unittest

import torch
import torch.nn as nn

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader


class TestModelOptNvfp4(unittest.TestCase):
    def test_swiglu_reload_restores_checkpoint_buffers(self):
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
        )
        method = ModelOptFp4LinearMethod(config)
        layer = nn.Module()
        method.create_weights(
            layer,
            input_size_per_partition=32,
            output_partition_sizes=[64],
            input_size=32,
            output_size=64,
            params_dtype=torch.bfloat16,
            weight_loader=default_weight_loader,
        )
        layer._interleave_for_swiglu_fusion = True
        layer.weight.data = torch.empty(0, dtype=layer.weight.dtype)
        layer.weight_scale.data = torch.empty(0, dtype=layer.weight_scale.dtype)

        method.restore_weights_before_loading(layer)

        self.assertEqual(layer.weight.shape, (64, 16))
        self.assertEqual(layer.weight_scale.shape, (64, 2))

    def test_moe_reload_reapplies_checkpoint_deinterleave(self):
        method = ModelOptNvFp4FusedMoEMethod.__new__(ModelOptNvFp4FusedMoEMethod)
        layer = nn.Module()
        layer.inference_moe_w13_interleaved = True
        layer._w13_deinterleaved = True

        method.restore_weights_before_loading(layer)

        self.assertFalse(layer._w13_deinterleaved)
