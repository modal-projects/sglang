"""Step-5 (StepFun) configuration.

Text decoder: 92-layer hybrid of sliding-window (512) and full-attention GQA
layers in a 3:1 pattern, MoE on layers 3-90 (352 experts, top-8, one shared
expert), per-head sigmoid attention gate, per-layer RoPE theta / partial rotary,
and a region-sparse indexer (CSA block compression, top-512 regions of 8) on the
full-attention layers. Per-layer arrays in the checkpoint carry trailing entries
for the ``num_nextn_predict_layers`` MTP layers.

VL wrapper: the Step-3.7 perception encoder and projector geometry, unchanged.
"""

from typing import Any, Optional, Union

from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.step3p7 import Step3p7VisionEncoderConfig


class Step5SparseConfig:
    """The ``sparse_config`` block, kept nested as the checkpoint stores it."""

    def __init__(
        self,
        enabled: bool = True,
        proxy_dim: int = 256,
        sparse_indexer_rope_dim: int = 32,
        sparse_indexer_use_rope: bool = True,
        sparse_indexer_num_heads: int = 16,
        sparse_indexer_num_k_heads: int = 1,
        sparse_indexer_q_norm_type: str = "rmsnorm",
        sparse_indexer_k_norm_type: str = "layernorm",
        sparse_indexer_csa_z_norm_type: str = "none",
        sparse_indexer_softmax_variant: str = "ssmax",
        sparse_indexer_ssmax_s_granularity: str = "q_head",
        num_provider_groups: int = 4,
        topk: int = 512,
        region_block_size: int = 8,
        compression_method: str = "csa_block_compress",
        attention_impl: str = "sparse_gqa",
        apply_to_layer_types: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        self.enabled = enabled
        self.proxy_dim = proxy_dim
        self.sparse_indexer_rope_dim = sparse_indexer_rope_dim
        self.sparse_indexer_use_rope = sparse_indexer_use_rope
        self.sparse_indexer_num_heads = sparse_indexer_num_heads
        self.sparse_indexer_num_k_heads = sparse_indexer_num_k_heads
        self.sparse_indexer_q_norm_type = sparse_indexer_q_norm_type
        self.sparse_indexer_k_norm_type = sparse_indexer_k_norm_type
        self.sparse_indexer_csa_z_norm_type = sparse_indexer_csa_z_norm_type
        self.sparse_indexer_softmax_variant = sparse_indexer_softmax_variant
        self.sparse_indexer_ssmax_s_granularity = sparse_indexer_ssmax_s_granularity
        self.num_provider_groups = num_provider_groups
        self.topk = topk
        self.region_block_size = region_block_size
        self.compression_method = compression_method
        self.attention_impl = attention_impl
        self.apply_to_layer_types = list(apply_to_layer_types or ["full_attention"])
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def budget_tokens(self) -> int:
        """Tokens attended per query row: selected regions times region size."""
        return self.topk * self.region_block_size

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class Step5Config(PretrainedConfig):
    model_type = "step5"
    architectures = ["Step5ForCausalLM"]
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 13824,
        num_hidden_layers: int = 92,
        vocab_size: int = 128896,
        num_attention_heads: int = 64,
        num_attention_groups: int = 4,
        head_dim: int = 192,
        rms_norm_eps: float = 1e-5,
        norm_dtype: str = "float32",
        fp32_residual_connection: bool = True,
        use_head_wise_attn_gate: bool = True,
        rope_theta: Union[float, list] = 10000.0,
        partial_rotary_factors: Optional[list] = None,
        layer_types: Optional[list] = None,
        sliding_window: Optional[int] = 512,
        rope_scaling: Optional[dict] = None,
        yarn_only_types: Optional[list] = None,
        max_position_embeddings: int = 1048576,
        use_moe: bool = True,
        moe_intermediate_size: int = 1536,
        share_expert_dim: int = 1536,
        moe_num_experts: int = 352,
        moe_top_k: int = 8,
        moe_layer_list: Optional[list] = None,
        use_moe_router_bias: bool = True,
        need_fp32_gate: bool = True,
        norm_expert_weight: bool = True,
        moe_router_scaling_factor: float = 3.0,
        swiglu_limits: Optional[list] = None,
        swiglu_limits_shared: Optional[list] = None,
        num_nextn_predict_layers: int = 3,
        sparse_config: Optional[Union[dict, Step5SparseConfig]] = None,
        tie_word_embeddings: bool = False,
        bos_token_id: int = 0,
        eos_token_id: Optional[Union[int, list]] = None,
        **kwargs: Any,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.norm_dtype = norm_dtype
        self.fp32_residual_connection = fp32_residual_connection
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        self.max_position_embeddings = max_position_embeddings
        self.sliding_window = sliding_window
        self.rope_scaling = rope_scaling
        # transformers rewrites rope_scaling in super().__init__; keep the checkpoint's form.
        self._rope_scaling_raw = dict(rope_scaling) if rope_scaling else None
        self.yarn_only_types = list(yarn_only_types or [])
        self.num_nextn_predict_layers = num_nextn_predict_layers

        total_layers = num_hidden_layers + num_nextn_predict_layers
        self.layer_types = (
            list(layer_types) if layer_types is not None else ["full_attention"] * total_layers
        )
        if len(self.layer_types) != total_layers:
            raise ValueError(
                f"layer_types must have {total_layers} entries (main + MTP), got {len(self.layer_types)}"
            )
        self.rope_theta = (
            list(rope_theta)
            if isinstance(rope_theta, (list, tuple))
            else [float(rope_theta)] * total_layers
        )
        self.partial_rotary_factors = list(partial_rotary_factors or [1.0] * total_layers)

        self.use_moe = use_moe
        self.moe_intermediate_size = moe_intermediate_size
        self.share_expert_dim = share_expert_dim
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_layer_list = list(moe_layer_list or [])
        self.use_moe_router_bias = use_moe_router_bias
        self.need_fp32_gate = need_fp32_gate
        self.norm_expert_weight = norm_expert_weight
        self.moe_router_scaling_factor = moe_router_scaling_factor
        self.swiglu_limits = list(swiglu_limits or [])
        self.swiglu_limits_shared = list(swiglu_limits_shared or [])

        if isinstance(sparse_config, Step5SparseConfig):
            self.sparse_config = sparse_config
        elif sparse_config:
            self.sparse_config = Step5SparseConfig(**sparse_config)
        else:
            self.sparse_config = None

        # step3p5's decoder reads the comma-separated form; derive it if absent.
        if not kwargs.get("moe_layers_enum"):
            kwargs["moe_layers_enum"] = ",".join(str(i) for i in self.moe_layer_list)
        # The checkpoint carries both names for top-k; keep one.
        kwargs.pop("num_experts_per_tok", None)
        kwargs.pop("architectures", None)

        # Transformers validates layer_types against num_hidden_layers; the checkpoint's
        # list includes the MTP layers. Align for the validator, then restore.
        real_num_hidden_layers = self.num_hidden_layers
        self.num_hidden_layers = len(self.layer_types)
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id if eos_token_id is not None else [1, 2],
            **kwargs,
        )
        self.num_hidden_layers = real_num_hidden_layers

    # --- names other code expects -------------------------------------------------
    @property
    def num_key_value_heads(self) -> int:
        return self.num_attention_groups

    @property
    def num_experts_per_tok(self) -> int:
        return self.moe_top_k

    # --- per-layer accessors (valid for main and MTP layer indices) ---------------
    def layer_type(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]

    def is_sliding_layer(self, layer_idx: int) -> bool:
        return self.layer_type(layer_idx) == "sliding_attention"

    def sliding_window_for(self, layer_idx: int) -> Optional[int]:
        return self.sliding_window if self.is_sliding_layer(layer_idx) else None

    def is_sparse_layer(self, layer_idx: int) -> bool:
        return (
            self.sparse_config is not None
            and self.sparse_config.enabled
            and self.layer_type(layer_idx) in self.sparse_config.apply_to_layer_types
        )

    def rope_theta_for(self, layer_idx: int) -> float:
        return float(self.rope_theta[layer_idx])

    def rotary_dim_for(self, layer_idx: int) -> int:
        span = int(round(self.head_dim * self.partial_rotary_factors[layer_idx]))
        return span - span % 2

    def rope_scaling_for(self, layer_idx: int) -> Optional[dict]:
        if self._rope_scaling_raw and self.layer_type(layer_idx) in self.yarn_only_types:
            return dict(self._rope_scaling_raw)
        return None

    def is_moe_layer(self, layer_idx: int) -> bool:
        return self.use_moe and layer_idx in self.moe_layer_list

    def swiglu_limit_for(self, layer_idx: int) -> Optional[float]:
        return self.swiglu_limits[layer_idx] if layer_idx < len(self.swiglu_limits) else None

    def swiglu_limit_shared_for(self, layer_idx: int) -> Optional[float]:
        return (
            self.swiglu_limits_shared[layer_idx]
            if layer_idx < len(self.swiglu_limits_shared)
            else None
        )

    def to_dict(self) -> dict:
        output = super().to_dict()
        if self.sparse_config is not None:
            output["sparse_config"] = self.sparse_config.to_dict()
        return output


class Step5VLConfig(PretrainedConfig):
    """Step-5 vision-language wrapper: Step-3.7 perception encoder + projector."""

    model_type = "step5_vl"
    architectures = ["Step5ForConditionalGeneration"]

    def __init__(
        self,
        vision_config: Optional[Union[dict, Step3p7VisionEncoderConfig]] = None,
        text_config: Optional[Union[dict, Step5Config]] = None,
        understand_projector_stride: int = 2,
        projector_bias: bool = False,
        image_token_id: int = 128001,
        image_token_len: int = 169,
        patch_token_len: int = 81,
        im_start_token: str = "<im_start>",
        im_end_token: str = "<im_end>",
        im_patch_token: str = "<im_patch>",
        use_im_start_end: bool = True,
        vision_select_layer: int = -1,
        **kwargs: Any,
    ) -> None:
        if vision_config is None:
            vision_config = Step3p7VisionEncoderConfig()
        elif isinstance(vision_config, dict):
            vision_config = Step3p7VisionEncoderConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = Step5Config()
        elif isinstance(text_config, dict):
            text_config = Step5Config(**text_config)
        self.text_config = text_config

        self.understand_projector_stride = understand_projector_stride
        self.projector_bias = projector_bias
        self.hidden_size = text_config.hidden_size
        self.image_token_id = image_token_id
        self.image_token_len = image_token_len
        self.patch_token_len = patch_token_len
        self.im_start_token = im_start_token
        self.im_end_token = im_end_token
        self.im_patch_token = im_patch_token
        self.use_im_start_end = use_im_start_end
        self.vision_select_layer = vision_select_layer

        kwargs.pop("architectures", None)
        super().__init__(**kwargs)
