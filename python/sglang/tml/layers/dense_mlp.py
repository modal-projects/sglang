from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.llama import LlamaMLP
from sglang.srt.server_args import get_global_server_args
from sglang.tml.kernels.comm import symm_mem_all_reduce
from sglang.tml.layers.util import FusedMoELoadingMixin, lora_compatible_layout_enabled


class SharedExpertFp4Strategy(Enum):
    """How a shared-expert dense MLP materializes its weights for serving."""

    #: Checkpoint BF16 -> serve bf16.
    BF16 = "bf16"
    #: Checkpoint FP4 -> serve FP4, repacked for flashinfer-trtllm (no dequant).
    FP4 = "fp4"

    @property
    def loads_fp4_checkpoint(self) -> bool:
        """Shared experts are NVFP4 in the checkpoint (-> allocate FP4 params to load)."""
        return self is SharedExpertFp4Strategy.FP4

    @property
    def serves_fp4(self) -> bool:
        """Shared experts are served as FP4 (-> the _forward_fp4 path)."""
        return self is SharedExpertFp4Strategy.FP4


@torch.compile(fullgraph=True)
def swiglu(z_btn: torch.Tensor) -> torch.Tensor:
    # Run swiglu in float32: torch.compile will fuse the cast to fp32 and will
    # not materialize the fp32 intermediate.
    dtype = z_btn.dtype
    z_btn = z_btn.float()
    # NOTE: Interleave gate and up for TP.
    y_btn = F.silu(z_btn[..., ::2]) * z_btn[..., 1::2]
    return y_btn.to(dtype)


@torch.compile(fullgraph=True)
def swiglu_contiguous(z_btn: torch.Tensor) -> torch.Tensor:
    # Run swiglu in float32: torch.compile will fuse the cast to fp32 and will
    # not materialize the fp32 intermediate.
    dtype = z_btn.dtype
    z_btn = z_btn.float()
    # NOTE: strided gate and up for inference TP.
    y_btn = (
        F.silu(z_btn[..., : z_btn.shape[-1] // 2]) * z_btn[..., z_btn.shape[-1] // 2 :]
    )
    return y_btn.to(dtype)


class InklingSwiglu(nn.Module):
    def __init__(self, interleaved: bool = True):
        super().__init__()
        self.interleaved = interleaved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swiglu(x) if self.interleaved else swiglu_contiguous(x)


class InklingDenseMLP(LlamaMLP):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        use_global_scale: bool,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        fused: bool = False,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_group: torch.distributed.ProcessGroup | None = None,
        use_dp_attention_reduce: bool = False,
    ) -> None:
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group

        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act="silu",
            quant_config=quant_config,
            prefix=prefix,
            reduce_results=False,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            use_dp_attention_reduce=use_dp_attention_reduce,
        )

        if use_global_scale:
            self.global_scale = nn.Parameter(torch.empty(1), requires_grad=False)
        else:
            self.global_scale = None

        # TODO we need to fix the helion kernel so that we can have contiguous swiglu.
        # Under --enable-lora gate/up is de-interleaved to [gate||up] at load, so run
        # contiguous swiglu (see lora_compatible_layout_enabled for the invariant).
        fused = fused and not lora_compatible_layout_enabled()
        self.layer_id = layer_id
        self.act_fn = InklingSwiglu(interleaved=fused)

    def forward(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch | None = None,
        use_reduce_scatter: bool = False,
    ):
        x = super().forward(x, forward_batch, use_reduce_scatter)
        if self.global_scale is not None:
            x = x * self.global_scale
        if not use_reduce_scatter and self.tp_group is not None:
            x = symm_mem_all_reduce(x, self.tp_group)
        return x


# Compile since eager achieves low mem bw for shared experts shapes (e.g.
# ~700 GB/s on GB300)
@torch.compile(fullgraph=True)
def _sum_dim0(x: torch.Tensor) -> torch.Tensor:
    return x.float().sum(dim=0).to(x.dtype)


# we borrow the FusedMoE layer to implement the batch dense mlp loading, it is easier to use the same quant method and kernel.
class InklingBatchDenseMLP(nn.Module, FusedMoELoadingMixin):
    def __init__(
        self,
        n_shared_experts: int,
        d_model: int,
        shared_d_mlp: int,
        layer_id: int,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        inference_moe_w13_interleaved: bool = True,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_group: torch.distributed.ProcessGroup | None = None,
    ):
        nn.Module.__init__(self)
        self.n_shared_experts = n_shared_experts
        self.inference_moe_w13_interleaved = inference_moe_w13_interleaved
        self.quant_config = quant_config
        self._fp4_strategy = self._resolve_fp4_strategy(quant_config, prefix)
        if self._fp4_strategy.loads_fp4_checkpoint:
            from sglang.tml.layers.quantization.quant import InklingNvfp4MoEMethod

            self.quant_method = InklingNvfp4MoEMethod(quant_config=quant_config)
        else:
            self.quant_method = UnquantizedFusedMoEMethod(False)
        self.moe_ep_size = 1
        self.moe_ep_rank = 0
        self.num_experts = n_shared_experts
        self.num_local_experts = n_shared_experts
        self.hidden_size = d_model
        self.layer_id = layer_id

        self.moe_tp_rank = tp_rank
        self.moe_tp_size = tp_size
        self.tp_group = tp_group

        local_intermediate_size = shared_d_mlp // self.moe_tp_size
        self.intermediate_size_per_partition = local_intermediate_size
        self.moe_runner_config = MoeRunnerConfig(
            num_experts=n_shared_experts,
            num_local_experts=n_shared_experts,
            hidden_size=d_model,
            intermediate_size_per_partition=local_intermediate_size,
            layer_id=layer_id,
            top_k=None,
            num_fused_shared_experts=n_shared_experts,
            params_dtype=torch.get_default_dtype(),
            activation="silu",
            apply_router_weight_on_input=False,
            inplace=True,
            no_combine=False,
            routed_scaling_factor=None,
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
            is_gated=True,
        )

        FusedMoELoadingMixin.__init__(
            self,
            quant_config,
            self.quant_method,
            self.moe_runner_config,
            self.moe_tp_rank,
        )
        self.quant_method.create_weights(
            layer=self,
            num_experts=n_shared_experts,
            hidden_size=d_model,
            intermediate_size_per_partition=local_intermediate_size,
            params_dtype=torch.get_default_dtype(),
            weight_loader=self.weight_loader_fused,
            with_bias=False,
        )
        self._fp4_shared_processed = False

    @staticmethod
    def _resolve_fp4_strategy(
        quant_config: QuantizationConfig | None, prefix: str
    ) -> SharedExpertFp4Strategy:
        from sglang.tml.layers.quantization.config import InklingModelOptNvfp4Config

        S = SharedExpertFp4Strategy
        # Plain bf16 models, and EP-replicated shared experts (no TP), serve bf16.
        if not isinstance(quant_config, InklingModelOptNvfp4Config):
            return S.BF16
        ckpt_prefix = prefix.replace(".experts.shared_experts", ".shared_experts")
        shared_excluded = quant_config.exclude_layer(
            f"{ckpt_prefix}.shared_w13_weight"
        ) or quant_config.exclude_layer(f"{ckpt_prefix}.shared_w2_weight")
        if shared_excluded:
            return S.BF16
        # Backend (marlin / flashinfer-trtllm / cutlass) is chosen and validated by
        # the OSS ModelOptFp4LinearMethod at process_weights_after_loading time.
        return S.FP4

    def forward(
        self, x: torch.Tensor, gammas: torch.Tensor, use_reduce_scatter: bool = False
    ) -> torch.Tensor:
        """
        t: number of tokens (batch size * sequence length)
        s: n_shared_experts
        d: d_model
        f: shared_d_mlp
        """
        assert x.ndim in (2, 3), f"{x.shape=}"
        assert gammas.ndim in (2, 3), f"{gammas.shape=}"
        assert (
            gammas.size(-1) == self.n_shared_experts
        ), f"{gammas.shape=} {self.n_shared_experts=}"
        if self._fp4_strategy.serves_fp4:
            return self._forward_fp4(x, gammas, use_reduce_scatter)

        x_td = x.view(-1, x.size(-1)) if x.ndim != 2 else x
        gammas_ts = gammas.view(-1, gammas.size(-1)) if gammas.ndim != 2 else gammas
        gammas_st = gammas_ts.transpose(0, 1)

        # these are for precision aligning w/ torchtitan computation
        _bmm = torch.bmm

        x_std = x_td.unsqueeze(0).expand(self.n_shared_experts, -1, -1).contiguous()
        # Batch the shared experts on dim-0
        y_st2f = _bmm(x_std, self.w13_weight.mT)
        y_stf = self._swiglu(y_st2f, gammas_st)
        z_std = _bmm(y_stf, self.w2_weight.mT)

        # Cast to float32 for sum then back to original dtype (matches TorchTitan _sum_dim0).
        out_td = _sum_dim0(z_std)
        if not use_reduce_scatter and self.tp_group is not None:
            out_td = symm_mem_all_reduce(out_td, self.tp_group)
        return out_td.view_as(x) if x.ndim == 2 else out_td

    def _swiglu(self, y_st2f: torch.Tensor, gammas_st: torch.Tensor) -> torch.Tensor:
        # Use the fused triton kernel instead of the helion silu_and_mul which
        # produces NaN for small shared-expert batches.
        from sglang.tml.kernels.inkling_moe import silu_and_mul_triton

        assert (
            self.inference_moe_w13_interleaved
        ), "silu_and_mul_triton requires interleaved w13"
        y_st_2f = y_st2f.view(-1, y_st2f.size(-1))
        y_st_f = silu_and_mul_triton(y_st_2f, gammas_st.reshape(-1))
        return y_st_f.view(*y_st2f.shape[:-1], y_st2f.size(-1) // 2)

    # ---- NVFP4 shared-expert serving via OSS ModelOptFp4LinearMethod ----
    # The two dense shared-sink GEMMs (w13, w2) are served as plain NVFP4 linears
    # through the generic ModelOptFp4LinearMethod, which owns weight prep and the
    # GEMM for every backend (marlin on SM80-90, flashinfer-trtllm / cutlass on
    # SM100+). Per-expert checkpoint loading is unchanged; here we only stack the
    # loaded experts into the two 2D linear weights and hand them off.

    def process_weights_after_loading(self) -> None:
        if self._fp4_shared_processed:
            return
        self._fp4_shared_processed = True
        if self._fp4_strategy is SharedExpertFp4Strategy.FP4:
            self._build_fp4_linears()

    def _build_fp4_linears(self) -> None:
        n = self.n_shared_experts
        # w13: stack experts on output rows [n, 2*inter, K/2] -> [n*2*inter, K/2].
        two_inter, k_half = self.w13_weight.shape[1], self.w13_weight.shape[2]
        self._w13_linear = self._make_fp4_linear(
            self.w13_weight.data.reshape(n * two_inter, k_half),
            self.w13_scale.data.reshape(n * two_inter, self.w13_scale.shape[2]),
            self.w13_scale2.data,
            self.w13_input_amax.data,
            in_features=k_half * 2,
            out_features=n * two_inter,
        )
        # w2: concat experts on K [n, d_model, inter/2] -> [d_model, n*inter/2].
        d_model, inter_half = self.w2_weight.shape[1], self.w2_weight.shape[2]
        self._fp4_shared_intermediate = inter_half * 2
        self._w2_linear = self._make_fp4_linear(
            self.w2_weight.data.permute(1, 0, 2).reshape(d_model, n * inter_half),
            self.w2_scale.data.permute(1, 0, 2).reshape(
                d_model, n * self.w2_scale.shape[2]
            ),
            self.w2_scale2.data,
            self.w2_input_amax.data,
            in_features=n * inter_half * 2,
            out_features=d_model,
        )
        for nm in (
            "w13_weight",
            "w2_weight",
            "w13_scale",
            "w2_scale",
            "w13_scale2",
            "w2_scale2",
            "w13_original_shape",
            "w2_original_shape",
            "w13_input_amax",
            "w2_input_amax",
        ):
            self._parameters.pop(nm, None)

    def _make_fp4_linear(
        self,
        packed: torch.Tensor,
        block_scale: torch.Tensor,
        scale2: torch.Tensor,
        input_amax: torch.Tensor,
        in_features: int,
        out_features: int,
    ) -> nn.Module:
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptFp4LinearMethod,
        )

        method = ModelOptFp4LinearMethod(self.quant_config)
        layer = nn.Module()
        method.create_weights(
            layer,
            input_size_per_partition=in_features,
            output_partition_sizes=[out_features],
            input_size=in_features,
            output_size=out_features,
            params_dtype=torch.get_default_dtype(),
            weight_loader=lambda *a, **k: None,
        )
        # create_weights allocates on CPU; the holder is never module-placed, so move
        # it to the loaded weights' device before prep (marlin workspace needs cuda).
        layer.to(packed.device)
        global_scale, input_scale = self._shared_scales(scale2, input_amax)
        layer.weight.data.copy_(packed.contiguous())
        layer.weight_scale.data.copy_(block_scale.contiguous())
        layer.weight_scale_2.data.copy_(global_scale.reshape(1))
        layer.input_scale.data.copy_(input_scale.reshape(1))
        method.process_weights_after_loading(layer)
        layer._fp4_method = method
        return layer

    def _shared_scales(
        self, scale2: torch.Tensor, input_amax: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # All shared experts must share one global weight scale (reshard with
        # single_global_scale=True). ModelOpt's input_scale = amax / (6 * 448).
        flat2 = scale2.reshape(-1).float()
        if get_global_server_args().load_format == "dummy" and not bool(
            torch.all(flat2 == flat2[0])
        ):
            # dummy load fills per-element noise; collapse so the pipeline runs.
            flat2 = torch.ones_like(flat2)
        assert bool(torch.all(flat2 == flat2[0])), (
            f"shared-expert scale2 not constant across experts ({flat2.tolist()}); "
            "reshard with single_global_scale=True"
        )
        global_scale = flat2[0]
        input_scale = input_amax.float().reshape(()) / (6.0 * 448.0)
        if input_scale.item() <= 0.0:
            # Uncalibrated input quantizer (amax==0) would poison alpha/input_scale_inv;
            # fall back to neutral (per-block dynamic scales absorb the activation).
            input_scale = torch.ones_like(input_scale)
        return global_scale, input_scale

    def _forward_fp4(
        self, x: torch.Tensor, gammas: torch.Tensor, use_reduce_scatter: bool = False
    ) -> torch.Tensor:
        x_td = x.view(-1, x.size(-1)) if x.ndim != 2 else x
        gammas_ts = gammas.view(-1, gammas.size(-1)) if gammas.ndim != 2 else gammas
        gammas_st = gammas_ts.transpose(0, 1)

        y_t_s2f = self._w13_linear._fp4_method.apply(self._w13_linear, x_td)
        y_st2f = y_t_s2f.view(
            x_td.shape[0], self.n_shared_experts, 2 * self._fp4_shared_intermediate
        )
        y_st2f = y_st2f.transpose(0, 1).contiguous()
        y_stf = self._swiglu(y_st2f, gammas_st)
        y_t_sf = y_stf.transpose(0, 1).reshape(x_td.shape[0], -1).contiguous()
        out_td = self._w2_linear._fp4_method.apply(self._w2_linear, y_t_sf)

        if not use_reduce_scatter and self.tp_group is not None:
            out_td = symm_mem_all_reduce(out_td, self.tp_group)
        return out_td.view_as(x) if x.ndim == 2 else out_td
