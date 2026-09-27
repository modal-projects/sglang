"""TRT prefix packing must preserve scales through the later QKV preparation."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    _quantize_fp8_qkv,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestTRTPrefixPack(CustomTestCase):
    def test_pack_matches_unfused_with_strides_and_checkpoint_scales(self):
        for scales in ((1.0, 1.0), (2.0, 4.0)):
            with self.subTest(scales=scales):
                torch.manual_seed(7)
                shape = (257, 12, 128)
                full = torch.randn((257, 12, 256), device="cuda", dtype=torch.bfloat16)
                k_nope, v = full[..., :128], full[..., 128:]
                k_pe = torch.randn((257, 1, 64), device="cuda", dtype=torch.bfloat16)
                q = torch.randn((257, 12, 192), device="cuda", dtype=torch.bfloat16)
                layer = SimpleNamespace(
                    k_scale_float=scales[0],
                    v_scale_float=scales[1],
                    k_scale=torch.tensor([scales[0]], device="cuda"),
                    v_scale=torch.tensor([scales[1]], device="cuda"),
                )
                backend = TRTLLMMLABackend.__new__(TRTLLMMLABackend)
                packed_k, packed_v = backend.pack_prefix_chunk_kv(
                    k_nope, k_pe, v, layer=layer
                )
                k = torch.cat((k_nope, k_pe.expand(-1, shape[1], -1)), dim=-1)
                expected = _quantize_fp8_qkv(q, k, v, layer)
                actual = _quantize_fp8_qkv(q, packed_k, packed_v, layer)
                assert actual[1] is packed_k and actual[2] is packed_v
                assert actual[3:] == expected[3:] == scales
                for left, right in zip(actual[:3], expected[:3]):
                    torch.testing.assert_close(
                        left.float(), right.float(), rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
