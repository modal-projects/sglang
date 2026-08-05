import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.mm_utils import _can_skip_pre_embed_feature_move
from sglang.srt.multimodal.processors.kimi_k25 import (
    KimiGPUProcessorWrapper,
    _gpu_preprocess_images,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_wrapper(
    preprocess_device: str, keep_feature_on_device: bool = False
) -> KimiGPUProcessorWrapper:
    hf_processor = MagicMock()
    return KimiGPUProcessorWrapper(
        hf_processor,
        image_token="<image>",
        patch_size=14,
        merge_kernel_size=2,
        in_patch_limit=16384,
        patch_limit_on_one_side=128,
        fixed_output_tokens=4096,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        preprocess_device=preprocess_device,
        preprocess_microbatch_size=1,
        keep_feature_on_device=keep_feature_on_device,
    )


class TestKimiProcessorRouting(unittest.TestCase):
    def test_gpu_preprocess_route(self):
        wrapper = _make_wrapper("gpu")
        images = [object()]
        with patch.object(wrapper, "_gpu_call", return_value={"mode": "gpu"}) as call:
            self.assertEqual(wrapper(text="prompt", images=images), {"mode": "gpu"})
            call.assert_called_once_with("prompt", images)

    def test_cpu_preprocess_route_removes_device_override(self):
        wrapper = _make_wrapper("cpu")
        images = [object()]
        with patch.object(wrapper, "_cpu_call", return_value={"mode": "cpu"}) as call:
            self.assertEqual(
                wrapper(text="prompt", images=images, device="cuda"),
                {"mode": "cpu"},
            )
            call.assert_called_once_with("prompt", images)

    def test_cpu_preprocess_can_place_final_features_on_gpu(self):
        wrapper = _make_wrapper("cpu", keep_feature_on_device=True)
        wrapper._hf_processor.return_value = {
            "input_ids": torch.tensor([[1]]),
            "pixel_values": torch.ones(1, 3, 14, 14),
        }
        with patch.object(torch.Tensor, "to", return_value="gpu-features") as move:
            result = wrapper._cpu_call("prompt", None)
        self.assertEqual(result["pixel_values"], "gpu-features")
        move.assert_called_once_with("cuda")

    def test_kimi_embedder_handles_cpu_features(self):
        class CpuFeatureOwner:
            handles_cpu_mm_features = True

            def get_image_feature(self, items):
                return items

        self.assertTrue(
            _can_skip_pre_embed_feature_move(CpuFeatureOwner().get_image_feature)
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestKimiGpuPreprocessing(unittest.TestCase):
    def test_cpu_offload_matches_gpu_output(self):
        config = {
            "new_height": 28,
            "new_width": 28,
            "pad_height": 0,
            "pad_width": 0,
        }
        source = torch.randint(0, 256, (3, 28, 28), dtype=torch.uint8, device="cuda")
        mean = torch.zeros((1, 3, 1, 1), dtype=torch.float32, device="cuda")
        std_inv = torch.ones((1, 3, 1, 1), dtype=torch.float32, device="cuda")

        gpu_output, gpu_grid = _gpu_preprocess_images(
            [source.clone()],
            [config],
            mean,
            std_inv,
            patch_size=14,
            microbatch_size=1,
            offload_to_cpu=False,
        )
        cpu_output, cpu_grid = _gpu_preprocess_images(
            [source.clone()],
            [config],
            mean,
            std_inv,
            patch_size=14,
            microbatch_size=1,
            offload_to_cpu=True,
        )

        self.assertTrue(gpu_output.is_cuda)
        self.assertFalse(cpu_output.is_cuda)
        self.assertEqual(cpu_output.shape, (4, 3, 14, 14))
        torch.testing.assert_close(cpu_output, gpu_output.cpu())
        torch.testing.assert_close(cpu_grid, gpu_grid)


if __name__ == "__main__":
    unittest.main()
