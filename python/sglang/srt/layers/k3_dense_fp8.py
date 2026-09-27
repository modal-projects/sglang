"""Static E4M3 for K3's post-load merged linears; routed experts stay MXFP4."""

import torch
from torch import nn

from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear, input_to_float8


class K3DenseFP8Linear(nn.Module):
    def __init__(self, *, weight: torch.Tensor, tuple_output: bool = False):
        super().__init__()
        if weight.dtype != torch.bfloat16 or not weight.is_cuda or weight.ndim != 2:
            raise ValueError("K3 dense FP8 requires a CUDA BF16 weight matrix")
        if any(size % 16 for size in weight.shape):
            raise ValueError("K3 dense FP8 requires weight dimensions divisible by 16")
        quantized, scale = input_to_float8(weight.contiguous())
        if (
            not torch.isfinite(scale).all()
            or not torch.isfinite(quantized.float()).all()
        ):
            raise ValueError("Nonfinite K3 dense FP8 weights or scale")
        self.register_buffer("weight", quantized.t())
        self.register_buffer("weight_scale", scale)
        self.register_buffer(
            "input_scale", torch.ones(1, device=weight.device, dtype=torch.float32)
        )
        self.tuple_output = tuple_output

    def forward(self, x: torch.Tensor):
        if x.numel() == 0:
            output = x.new_empty((*x.shape[:-1], self.weight.shape[1]))
        else:
            output = apply_fp8_linear(
                x,
                self.weight,
                self.weight_scale,
                input_scale=self.input_scale,
                cutlass_fp8_supported=False,
                pad_output=False,
            )
        return (output, None) if self.tuple_output else output
