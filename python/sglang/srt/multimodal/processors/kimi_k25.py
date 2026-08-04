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
from sglang.srt.utils import envs

# ---------------------------------------------------------------------------
# GPU image preprocessing utilities (resize, pad, normalize, patchify on CUDA)
# ---------------------------------------------------------------------------

# Preprocessing runs in the tokenizer process, whose CUDA context shares GPU 0
# with TP rank 0 of the model. Only whatever `--mem-fraction-static` left unused
# is available there -- on a 4xB200 box at 0.935 that is a few hundred MiB, so
# preprocessing must not scale its GPU footprint with the client's input
# resolution.
#
# Two properties make that achievable:
#   * `navit_resize_config` bounds the *output* to `in_patch_limit` patches
#     regardless of input size, so anything larger than the resize target is
#     transient by construction.
#   * `get_image_feature` casts pixel_values to the vision tower's dtype
#     (bfloat16 under `--dtype bfloat16`) anyway, so computing in float32 costs
#     2x the memory for precision that is discarded downstream.
#
# Hence: compute in _PREPROC_DTYPE, normalize in place, skip the resize when the
# image already matches its target, and build multi-image batches into a single
# pre-allocated buffer instead of concatenating per-image tensors.
#
# The resampling math is deliberately left untouched, so pixel_values differ from
# the float32 pipeline only by bfloat16 rounding -- the same rounding
# get_image_feature applies on the way into the vision tower.
_PREPROC_DTYPE = torch.bfloat16

# Optional hard cap on the resolution handed to the GPU resize, in pixels.
#
# Peak footprint still scales with input resolution (3 bytes/pixel for the
# decoded uint8 image plus 6 for its bfloat16 copy), so a sufficiently large
# image can exhaust the preprocessing budget no matter how tight the pipeline
# is. Setting this trades image quality for a bound: inputs above the cap are
# decimated by an integer factor in uint8 before being widened.
#
# Off by default because the cost is not negligible -- nearest-neighbour
# subsampling aliases in a way the subsequent bicubic pass cannot undo, measured
# at ~25-27 dB PSNR against the unclamped pipeline (max deviation ~0.27 on the
# normalized [-1, 1] scale) once the factor reaches 2. Prefer bounding request
# size upstream; reach for this only if untrusted inputs must be tolerated, and
# validate model accuracy before relying on it.
_MAX_GPU_RESIZE_PIXELS = envs.SGLANG_KIMI_MM_MAX_GPU_RESIZE_PIXELS.get()

# Retain this multiple of the target resolution on each axis when decimating, so
# the bicubic resize that follows still has detail to work with.
_PRESCALE_HEADROOM = 2


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
    # Make the CHW view contiguous before the transfer: the copy has to happen
    # regardless, and doing it host-side keeps the H2D transfer a single DMA.
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().cuda()


def _decimate_uint8(x: torch.Tensor, new_h: int, new_w: int) -> torch.Tensor:
    """Subsample a (1, C, H, W) uint8 tensor by an integer factor.

    Only applied when `_MAX_GPU_RESIZE_PIXELS` is configured and the image
    exceeds it; see that constant for the quality trade-off. Returns `x`
    unchanged when disabled or when no factor of 2 or more is available.
    """
    _, _, h, w = x.shape
    if not _MAX_GPU_RESIZE_PIXELS or h * w <= _MAX_GPU_RESIZE_PIXELS:
        return x
    factor = min(h // (_PRESCALE_HEADROOM * new_h), w // (_PRESCALE_HEADROOM * new_w))
    if factor < 2:
        return x
    return x[:, :, ::factor, ::factor].contiguous()


def _resize_to_target(
    image: Union[torch.Tensor, Image.Image],
    new_h: int,
    new_w: int,
) -> torch.Tensor:
    """Return a (1, C, new_h, new_w) `_PREPROC_DTYPE` tensor on GPU.

    Images already at their target -- anything within `in_patch_limit`, which
    covers ordinary screenshots -- skip the interpolate entirely and are only
    widened. Anything else is resized with the same bicubic call as before, so
    the resampling is unchanged; `_decimate_uint8` is a no-op unless a cap has
    been configured.
    """
    if isinstance(image, Image.Image):
        image = _pil_to_cuda_chw(image)

    x = image.unsqueeze(0)
    if x.shape[2] == new_h and x.shape[3] == new_w:
        # Widening from uint8 always copies, so the result is safe to mutate in
        # place. `.contiguous()` is required because the interpolate that would
        # otherwise have normalized the layout is skipped here.
        return x.to(_PREPROC_DTYPE).contiguous()

    x = _decimate_uint8(x, new_h, new_w)
    return F.interpolate(
        x.to(_PREPROC_DTYPE),
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
    )


def _normalize_and_patchify(
    x: torch.Tensor,
    image_mean: torch.Tensor,
    image_std_inv: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize in place, then split (B, C, H, W) into flat patches.

    `x` must be exclusively owned by the caller; it is mutated.
    """
    x.div_(255.0).sub_(image_mean).mul_(image_std_inv)

    B, C, H, W = x.shape
    T = 1
    gh, gw = H // patch_size, W // patch_size
    x = x.view(B, C, gh, patch_size, gw, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C, patch_size, patch_size)

    grid_thw = torch.tensor([T, gh, gw], dtype=torch.int64, device=x.device)
    return x, grid_thw


def _process_single_image(
    image: Union[torch.Tensor, Image.Image],
    config: dict,
    image_mean: torch.Tensor,
    image_std_inv: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process a single image on GPU: resize -> pad -> normalize -> patchify."""
    new_h, new_w = config["new_height"], config["new_width"]
    pad_h, pad_w = config["pad_height"], config["pad_width"]

    x = _resize_to_target(image, new_h, new_w)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)

    patches, grid_thw = _normalize_and_patchify(
        x, image_mean, image_std_inv, patch_size
    )
    return patches.squeeze(0), grid_thw


def _gpu_preprocess_images(
    images: list[Union[torch.Tensor, Image.Image]],
    resize_configs: list[dict],
    image_mean: torch.Tensor,
    image_std_inv: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU preprocessing pipeline for a batch of images.

    Groups images with the same target padded size for batch processing.
    """
    n = len(images)
    if n == 0:
        device = image_mean.device
        return (
            torch.empty(
                0, 3, patch_size, patch_size, dtype=_PREPROC_DTYPE, device=device
            ),
            torch.empty(0, 3, dtype=torch.int64, device=device),
        )

    groups = defaultdict(list)
    for idx, (image, config) in enumerate(zip(images, resize_configs)):
        padded_h = config["new_height"] + config["pad_height"]
        padded_w = config["new_width"] + config["pad_width"]
        target_h = config["new_height"]
        target_w = config["new_width"]
        groups[(target_h, target_w, padded_h, padded_w)].append((idx, image, config))

    all_patches = [None] * n
    all_grids = [None] * n

    for (target_h, target_w, padded_h, padded_w), group in groups.items():
        if len(group) == 1:
            idx, image, config = group[0]
            patches, grid = _process_single_image(
                image, config, image_mean, image_std_inv, patch_size
            )
            all_patches[idx] = patches
            all_grids[idx] = grid
        else:
            # Resize into a pre-allocated padded batch one image at a time, so
            # only a single per-image temporary is live instead of the whole
            # group at input resolution. The buffer is zeroed, which also
            # supplies the padding the previous F.pad call added.
            first = group[0][1]
            channels = first.shape[0] if isinstance(first, torch.Tensor) else 3
            batch = torch.zeros(
                (len(group), channels, padded_h, padded_w),
                dtype=_PREPROC_DTYPE,
                device="cuda",
            )
            for i, (_, image, _) in enumerate(group):
                resized = _resize_to_target(image, target_h, target_w)
                batch[i : i + 1, :, :target_h, :target_w].copy_(resized)
                del resized

            batch, grid = _normalize_and_patchify(
                batch, image_mean, image_std_inv, patch_size
            )
            for i, (idx, _, _) in enumerate(group):
                all_patches[idx] = batch[i]
                all_grids[idx] = grid

    pixel_values = torch.cat(all_patches, dim=0)
    grid_thws = torch.stack(all_grids, dim=0)
    return pixel_values, grid_thws


# ---------------------------------------------------------------------------
# Kimi K2.5 GPU processor wrapper
# ---------------------------------------------------------------------------


class KimiGPUProcessorWrapper:
    """Wraps Kimi's HF processor to do GPU image preprocessing.

    GPU path: nvJPEG CUDA tensor / PIL -> _gpu_preprocess_images()
    CPU fallback: PIL -> medias kwarg -> original HF KimiK25Processor.__call__

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

        if images and torch.cuda.is_available():
            return self._gpu_call(text, images)
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
            images, resize_configs, image_mean, image_std_inv, self._patch_size
        )

        grid_thws = grid_thws.cpu()

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
        return out

    def _get_gpu_norm_tensors(self, device="cuda"):
        if self._gpu_norm_tensors is None:
            # Invert in float32 for accuracy, then match the preprocessing dtype
            # so the normalization can run as in-place ops (an in-place op whose
            # operand is wider than the output is a hard error in torch).
            image_mean = (
                torch.tensor(self._image_mean, device=device, dtype=torch.float32)
                .view(1, 3, 1, 1)
                .to(_PREPROC_DTYPE)
            )
            image_std_inv = (
                (
                    1.0
                    / torch.tensor(
                        self._image_std, device=device, dtype=torch.float32
                    )
                )
                .view(1, 3, 1, 1)
                .to(_PREPROC_DTYPE)
            )
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
