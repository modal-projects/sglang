import unittest

import torch

from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsFusedMoEMethod,
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4LinearMethod,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _UnsupportedMethod(QuantizeMethodBase):
    def create_weights(self, layer, *args, **kwargs):
        pass

    def apply(self, layer, *args, **kwargs):
        return torch.empty(0)


class _StagingScheme:
    def weight_staging_postprocess_device(self, layer):
        return "cpu"


class WeightStagingCapabilitiesTest(unittest.TestCase):
    def test_methods_are_unsupported_by_default(self):
        method = _UnsupportedMethod()
        layer = torch.nn.Module()

        self.assertIsNone(method.weight_staging_postprocess_device(layer))

    def test_compressed_tensor_methods_delegate_to_their_scheme(self):
        layer = torch.nn.Module()
        layer.scheme = _StagingScheme()

        for method in (
            CompressedTensorsLinearMethod(None),
            CompressedTensorsFusedMoEMethod(None),
        ):
            self.assertEqual(method.weight_staging_postprocess_device(layer), "cpu")

    def test_modelopt_fp4_uses_cuda_postprocessing(self):
        layer = torch.nn.Module()
        linear = ModelOptFp4LinearMethod(None)
        moe = ModelOptNvFp4FusedMoEMethod.__new__(ModelOptNvFp4FusedMoEMethod)

        self.assertEqual(linear.weight_staging_postprocess_device(layer), "cuda")
        self.assertEqual(moe.weight_staging_postprocess_device(layer), "cuda")

        layer.inference_moe_w13_interleaved = True
        self.assertIsNone(moe.weight_staging_postprocess_device(layer))

    def test_kv_cache_commit_refreshes_python_scales(self):
        layer = torch.nn.Module()
        layer.k_scale = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
        layer.v_scale = torch.nn.Parameter(torch.tensor(3.0), requires_grad=False)
        method = BaseKVCacheMethod(None)

        self.assertEqual(method.weight_staging_postprocess_device(layer), "cpu")
        method.process_weights_after_weight_commit(layer)

        self.assertEqual(layer.k_scale_float, 2.0)
        self.assertEqual(layer.v_scale_float, 3.0)


if __name__ == "__main__":
    unittest.main()
