from __future__ import annotations

import abc

import torch
from torch import nn

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.server_args import get_global_server_args


def lora_compatible_layout_enabled() -> bool:
    """Serve stock contiguous [gate||up] (not Inkling-interleaved) so LoRA slicing lines up;
    construction and loading must agree. --enable-lora-gated: non-LoRA is byte-identical.
    """
    return get_global_server_args().enable_lora


def deinterleave_gate_up(weight: torch.Tensor, dim: int) -> torch.Tensor:
    """Convert Inkling [gate0, up0, ...] interleaved layout to stock [gate..., up...]."""
    dim = dim % weight.dim()
    if weight.shape[dim] % 2 != 0:
        raise ValueError(
            f"Cannot deinterleave odd gate/up dimension {dim}: {tuple(weight.shape)}"
        )
    shape = list(weight.shape)
    half = shape[dim] // 2
    view_shape = shape[:dim] + [half, 2] + shape[dim + 1 :]
    return (
        weight.reshape(view_shape)
        .transpose(dim, dim + 1)
        .reshape_as(weight)
        .contiguous()
    )


class FusedMoELoadingMixin(abc.ABC):
    def __init__(
        self,
        quant_config: QuantizationConfig | None,
        quant_method: UnquantizedFusedMoEMethod,
        moe_runner_config: MoeRunnerConfig,
        moe_tp_rank: int,
    ) -> None:
        super().__init__()
        helper = FusedMoE.__new__(FusedMoE)
        nn.Module.__init__(helper)
        helper.quant_config = quant_config
        helper.quant_method = quant_method
        helper.moe_runner_config = moe_runner_config
        helper.use_triton_kernels = False
        helper.moe_tp_rank = moe_tp_rank
        helper.use_presharded_weights = False
        helper.use_flashinfer_trtllm_moe = False
        # Keep the loading helper OUT of the parent's module tree: it is a
        # method-borrowing object with no parameters/buffers. If it were a
        # registered submodule, the dummy/post-load path (which walks
        # named_modules() and calls quant_method.process_weights_after_loading on
        # every module that has one) would process this weightless helper as if it
        # were a real quantized layer — the upstream ModelOpt MoE method then dies
        # on a missing w13_input_scale. object.__setattr__ stores it as a plain
        # attribute, so named_modules() never sees it; weight_loader_fused still
        # reaches it via self.helper.
        object.__setattr__(self, "helper", helper)

    def weight_loader_fused(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
    ) -> None:
        return self.helper.weight_loader_fused(
            param, loaded_weight, weight_name, shard_id
        )
