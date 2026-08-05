import math
import re
from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sglang.srt.managers.schedule_batch import (
    MultimodalProcessorOutput,
)
from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin

# ---------------------------------------------------------------------------
# GPU image preprocessing utilities (resize, pad, normalize, patchify on CUDA)
# ---------------------------------------------------------------------------


def navit_resize_config(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None = None,
) -> dict:
    """Compute NaViT resize target dimensions and token count.

    Pure math -- no image data needed, only (width, height).
    """
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w = min(max(1, int(width * scale)), patch_limit_on_one_side * patch_size)
    new_h = min(max(1, int(height * scale)), patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor
        num_tokens = token_height * token_width

    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
    }


def _get_image_dimensions(image: Union[torch.Tensor, Image.Image]) -> tuple[int, int]:
    """Get (width, height) from a CUDA tensor or PIL Image."""
    if isinstance(image, torch.Tensor):
        # nvJPEG returns (C, H, W) uint8
        return image.shape[2], image.shape[1]
    return image.size  # PIL returns (width, height)


def _pil_to_cuda_chw(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to (C, H, W) uint8 CUDA tensor."""
    arr = np.asarray(image.convert("RGB"))
    return torch.from_numpy(arr).permute(2, 0, 1).cuda()


def _process_single_image(
    image: Union[torch.Tensor, Image.Image],
    config: dict,
    image_mean: torch.Tensor,
    image_std_inv: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process a single image on GPU: resize -> pad -> normalize -> patchify."""
    if isinstance(image, Image.Image):
        image = _pil_to_cuda_chw(image)

    new_h, new_w = config["new_height"], config["new_width"]
    pad_h, pad_w = config["pad_height"], config["pad_width"]

    x = image.unsqueeze(0).float()
    x = F.interpolate(x, size=(new_h, new_w), mode="bicubic", align_corners=False)

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)

    x = x / 255.0
    x = (x - image_mean) * image_std_inv

    _, C, H, W = x.shape
    T = 1
    gh, gw = H // patch_size, W // patch_size
    x = x.view(T, C, gh, patch_size, gw, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, C, patch_size, patch_size)

    grid_thw = torch.tensor([T, gh, gw], dtype=torch.int64, device=x.device)
    return x, grid_thw


def _gpu_preprocess_images(
    images: list[Union[torch.Tensor, Image.Image]],
    resize_configs: list[dict],
    image_mean: torch.Tensor,
    image_std_inv: torch.Tensor,
    patch_size: int,
    microbatch_size: int,
    offload_to_cpu: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess bounded image microbatches on GPU.

    When ``offload_to_cpu`` is enabled, processed patches are copied directly
    into a CPU output tensor so the complete request does not remain resident
    on GPU while chunked language-model prefill runs.
    """
    n = len(images)
    output_device = torch.device("cpu") if offload_to_cpu else image_mean.device
    if n == 0:
        return (
            torch.empty(
                0,
                3,
                patch_size,
                patch_size,
                dtype=torch.float32,
                device=output_device,
            ),
            torch.empty(0, 3, dtype=torch.int64),
        )

    offsets = [0]
    grid_rows = []
    groups = defaultdict(list)
    for idx, config in enumerate(resize_configs):
        padded_h = config["new_height"] + config["pad_height"]
        padded_w = config["new_width"] + config["pad_width"]
        gh, gw = padded_h // patch_size, padded_w // patch_size
        offsets.append(offsets[-1] + gh * gw)
        grid_rows.append((1, gh, gw))
        groups[
            (
                config["new_height"],
                config["new_width"],
                padded_h,
                padded_w,
            )
        ].append(idx)

    pixel_values = torch.empty(
        offsets[-1],
        3,
        patch_size,
        patch_size,
        dtype=torch.float32,
        device=output_device,
    )
    grid_thws = torch.tensor(grid_rows, dtype=torch.int64)

    for (target_h, target_w, padded_h, padded_w), group in groups.items():
        for begin in range(0, len(group), microbatch_size):
            image_indices = group[begin : begin + microbatch_size]
            full_size_tensors = []
            for image_idx in image_indices:
                image = images[image_idx]
                if isinstance(image, Image.Image):
                    image = _pil_to_cuda_chw(image)
                full_size_tensors.append(image.unsqueeze(0).float())
                # The processed patches are the only representation needed
                # after this point. Release nvJPEG CUDA tensors incrementally.
                images[image_idx] = None
                image = None

            resized = [
                F.interpolate(
                    tensor,
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False,
                )
                for tensor in full_size_tensors
            ]
            batch = resized[0] if len(resized) == 1 else torch.cat(resized, dim=0)
            del resized, full_size_tensors

            pad_h = padded_h - target_h
            pad_w = padded_w - target_w
            if pad_h > 0 or pad_w > 0:
                batch = F.pad(batch, (0, pad_w, 0, pad_h), value=0.0)

            batch.div_(255.0).sub_(image_mean).mul_(image_std_inv)
            batch_size, channels, height, width = batch.shape
            gh, gw = height // patch_size, width // patch_size
            patches = (
                batch.view(
                    batch_size,
                    channels,
                    gh,
                    patch_size,
                    gw,
                    patch_size,
                )
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(batch_size, -1, channels, patch_size, patch_size)
            )

            for local_idx, image_idx in enumerate(image_indices):
                output = pixel_values[offsets[image_idx] : offsets[image_idx + 1]]
                output.copy_(patches[local_idx], non_blocking=False)

            del patches, batch

    if offload_to_cpu:
        # The tokenizer/processor is a separate CUDA-using process. Return its
        # temporary allocator reservations to the model workers.
        torch.cuda.empty_cache()
    return pixel_values, grid_thws


# ---------------------------------------------------------------------------
# Kimi K2.5 GPU processor wrapper
# ---------------------------------------------------------------------------


class KimiGPUProcessorWrapper:
    """Wraps Kimi's HF processor with configurable image preprocessing.

    GPU path: nvJPEG CUDA tensor / PIL -> _gpu_preprocess_images()
    CPU path: PIL -> medias kwarg -> original HF KimiK25Processor.__call__

    Exposes attributes that base class's process_mm_data needs so it behaves
    like a normal HF processor from the outside.
    """

    def __init__(
        self,
        hf_processor,
        image_token,
        patch_size,
        merge_kernel_size,
        in_patch_limit,
        patch_limit_on_one_side,
        fixed_output_tokens,
        image_mean,
        image_std,
        preprocess_device,
        preprocess_microbatch_size,
        keep_feature_on_device,
    ):
        self._hf_processor = hf_processor
        self._image_token = image_token
        self._patch_size = patch_size
        self._merge_kernel_size = merge_kernel_size
        self._in_patch_limit = in_patch_limit
        self._patch_limit_on_one_side = patch_limit_on_one_side
        self._fixed_output_tokens = fixed_output_tokens
        self._image_mean = image_mean
        self._image_std = image_std
        self._preprocess_device = preprocess_device
        self._preprocess_microbatch_size = preprocess_microbatch_size
        self._offload_to_cpu = not keep_feature_on_device
        self._gpu_norm_tensors = None

        # Explicitly expose attributes that base class process_mm_data needs:
        # - image_processor: checked via isinstance(..., BaseImageProcessor)
        # - tokenizer: used for tokenization
        # - media_processor: used by CPU fallback path
        self.image_processor = hf_processor.image_processor
        self.tokenizer = hf_processor.tokenizer
        self.media_processor = hf_processor.media_processor

    def __call__(self, text=None, images=None, **kwargs):
        # process_mm_data passes images via kwargs["images"]
        images = images or kwargs.pop("images", None)

        if images and self._preprocess_device == "gpu":
            return self._gpu_call(text, images)
        # BaseMultiModalProcessor may select a CUDA device for fast image
        # processors. Explicit CPU preprocessing must override that choice.
        kwargs.pop("device", None)
        return self._cpu_call(text, images, **kwargs)

    def _gpu_call(self, text, images):
        """Bypass HF KimiK25VisionProcessor.preprocess entirely -- use GPU ops."""
        input_text = text[0] if isinstance(text, list) else text

        # 1. Compute resize configs (CPU math)
        resize_configs = []
        for image in images:
            w, h = _get_image_dimensions(image)
            resize_configs.append(
                navit_resize_config(
                    w,
                    h,
                    self._patch_size,
                    self._merge_kernel_size,
                    self._in_patch_limit,
                    self._patch_limit_on_one_side,
                    self._fixed_output_tokens,
                )
            )

        # 2. Expand image tokens
        parts = input_text.split(self._image_token)
        result = [parts[0]]
        for config, part in zip(resize_configs, parts[1:]):
            result.append(self._image_token * config["num_tokens"] + part)
        input_text = "".join(result)

        # 3. Tokenize
        text_inputs = self._hf_processor.tokenizer(input_text, return_tensors="pt")

        # 4. GPU image preprocessing
        image_mean, image_std_inv = self._get_gpu_norm_tensors()
        pixel_values, grid_thws = _gpu_preprocess_images(
            images,
            resize_configs,
            image_mean,
            image_std_inv,
            self._patch_size,
            self._preprocess_microbatch_size,
            self._offload_to_cpu,
        )

        return {
            "input_ids": text_inputs["input_ids"],
            "pixel_values": pixel_values,
            # Use SGL-standard key so get_new_expanded_mm_items() can split
            # per-image for cache granularity (it looks up 'image_grid_thw').
            "image_grid_thw": grid_thws,
        }

    def _cpu_call(self, text, images, **kwargs):
        """Fallback: token expansion + medias kwarg -> original HF processor."""
        input_text = text[0] if isinstance(text, list) else text

        if images:
            # Token expansion via media_tokens_calculator
            parts = input_text.split(self._image_token)
            result = [parts[0]]
            for image, part in zip(images, parts[1:]):
                num_tokens = self._hf_processor.media_processor.media_tokens_calculator(
                    {"type": "image", "image": image}
                )
                result.append(self._image_token * num_tokens + part)
            input_text = "".join(result)

            # Convert to medias format for Kimi's HF processor
            kwargs["medias"] = [{"type": "image", "image": img} for img in images]

        out = self._hf_processor(text=[input_text], **kwargs)
        grid_thws = out.pop("grid_thws", None)
        if grid_thws is not None:
            out["image_grid_thw"] = grid_thws
        if not self._offload_to_cpu and isinstance(
            out.get("pixel_values"), torch.Tensor
        ):
            out["pixel_values"] = out["pixel_values"].to("cuda")
        return out

    def _get_gpu_norm_tensors(self, device="cuda"):
        if self._gpu_norm_tensors is None:
            image_mean = torch.tensor(
                self._image_mean, device=device, dtype=torch.float32
            ).view(1, 3, 1, 1)
            image_std_inv = (
                1.0 / torch.tensor(self._image_std, device=device, dtype=torch.float32)
            ).view(1, 3, 1, 1)
            self._gpu_norm_tensors = (image_mean, image_std_inv)
        return self._gpu_norm_tensors


# ---------------------------------------------------------------------------
# Kimi K2.5 SGLang multimodal processor
# ---------------------------------------------------------------------------


# Compatible with KimiVLForConditionalGeneration
class KimiK2_5VLImageProcessor(KimiGridMMDataMixin, SGLangBaseProcessor):
    models = [KimiK25ForConditionalGeneration]
    gpu_image_decode = True  # nvJPEG for JPEG, PIL fallback for others

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        preprocess_device = server_args.mm_preprocess_device
        if preprocess_device == "auto":
            preprocess_device = "gpu" if torch.cuda.is_available() else "cpu"
        elif preprocess_device == "gpu" and not torch.cuda.is_available():
            raise RuntimeError(
                "--mm-preprocess-device=gpu requires CUDA to be available"
            )
        # CPU preprocessing requires PIL/CPU inputs rather than nvJPEG CUDA
        # tensors. This instance setting is consumed by load_mm_data().
        self.gpu_image_decode = preprocess_device == "gpu"
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|media_pad|>",
            # TODO: could we convert in MultimodalSpecialTokens?
            image_token_id=hf_config.media_placeholder_token_id,
            image_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),
        ).build(_processor)

        # Extract media processing config from HF processor
        media_proc_cfg = _processor.media_processor.media_proc_cfg

        # Replace with GPU-capable wrapper
        self._processor = KimiGPUProcessorWrapper(
            _processor,
            image_token=self.mm_tokens.image_token,
            patch_size=media_proc_cfg["patch_size"],
            merge_kernel_size=media_proc_cfg["merge_kernel_size"],
            in_patch_limit=media_proc_cfg["in_patch_limit"],
            patch_limit_on_one_side=media_proc_cfg["patch_limit_on_one_side"],
            fixed_output_tokens=media_proc_cfg.get("fixed_output_tokens"),
            image_mean=media_proc_cfg["image_mean"],
            image_std=media_proc_cfg["image_std"],
            preprocess_device=preprocess_device,
            preprocess_microbatch_size=server_args.mm_preprocess_microbatch_size,
            keep_feature_on_device=server_args.keep_mm_feature_on_device,
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
        )

    def get_mm_data(self, prompt, embeddings, **kwargs):
        img_grid_thw = kwargs.get("img_grid_thw", None)
        return self._build_kimi_mm_data_from_grids(
            prompt=prompt,
            embeddings=embeddings,
            image_token_id=self.mm_tokens.image_token_id,
            img_grid_thw=img_grid_thw,
        )
