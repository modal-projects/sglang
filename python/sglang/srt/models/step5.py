"""Step-5 (StepFun): Step5ForCausalLM / Step5ForConditionalGeneration.

The decoder is the Step-3.5 trunk (hybrid sliding/full GQA, QK-norm, per-layer
partial RoPE, head-wise attention gate, 352-expert MoE with clamped SwiGLU,
shared expert, biased sigmoid router) at 92 layers, so it reuses step3p5's
modules under Step5Config. The vision tower and projector are Step-3.7's.

Phase 1 (this file): the full-attention layers run dense. Step-5's sparse
indexer selects 512 regions of 8 tokens, so dense attention is exactly
equivalent for contexts up to 4096 tokens; the indexer weights are skipped at
load. Phase 2 adds the region-sparse indexer on the QSA backend.
"""

import logging
from typing import Iterable, Optional, Tuple

import torch
from transformers.activations import ACT2FN

from sglang.srt.configs.step5 import Step5Config, Step5VLConfig
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.step3_vl_10b import PerceptionEncoder
from sglang.srt.models.step3p5 import Step3p5ForCausalLM
from sglang.srt.models.step3p7 import Step3p7ForConditionalGeneration
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

# Indexer parameters live directly under self_attn in the checkpoint.
_SPARSE_INDEXER_MARKERS = (".self_attn.sparse_indexer_", ".self_attn.ssmax_s")


def _drop_sparse_indexer_weights(weights: Iterable[Tuple[str, torch.Tensor]]):
    dropped = 0
    for name, loaded_weight in weights:
        if any(marker in name for marker in _SPARSE_INDEXER_MARKERS):
            dropped += 1
            continue
        yield name, loaded_weight
    if dropped:
        logger.info("Step5: skipped %d sparse-indexer tensors (dense phase)", dropped)


_PARAM_SUFFIXES = (".w13_weight", ".w2_weight", ".weight", ".bias")


def _normalize_ignored_layers(quant_config: Optional[QuantizationConfig]) -> None:
    """The checkpoint's modules_to_not_convert names parameters
    ("...moe.experts.w13_weight"); sglang skips quantization by module prefix
    ("...moe.experts"). Add the module form of every parameter-style entry."""
    ignored = getattr(quant_config, "ignored_layers", None)
    if not ignored:
        return
    extra = []
    for name in ignored:
        for suffix in _PARAM_SUFFIXES:
            if name.endswith(suffix):
                extra.append(name[: -len(suffix)])
                break
    merged = list(dict.fromkeys([*ignored, *extra]))
    if len(merged) != len(ignored):
        quant_config.ignored_layers = merged
        logger.info("Step5: expanded %d parameter-style ignore entries to module prefixes", len(merged) - len(ignored))


class Step5ForCausalLM(Step3p5ForCausalLM):
    def __init__(
        self,
        config: Step5Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        _normalize_ignored_layers(quant_config)
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        return super().load_weights(_drop_sparse_indexer_weights(weights), is_nextn=is_nextn)


class Step5ForConditionalGeneration(Step3p7ForConditionalGeneration):
    # FP8 checkpoints name text weights "model.*" and the quantization ignore
    # list uses the same prefix with sglang's fused expert names
    # ("model.layers.88.moe.experts.w13_weight"); sglang parameters live under
    # "language_model.model.*". Map both prefix styles.
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.layers.": "language_model.model.layers.",
            "model.embed_tokens": "language_model.model.embed_tokens",
            "model.norm": "language_model.model.norm",
            "model.vision_model": "vision_model",
            "model.vit_large_projector": "vit_large_projector",
            "lm_head": "language_model.lm_head",
        }
    )

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return Step5ForCausalLM.get_model_config_for_expert_location(config.text_config)

    def __init__(
        self,
        config: Step5VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        # Rebuild step3p7's constructor with the Step-5 language model; calling
        # super().__init__ would allocate a Step3p5ForCausalLM we then discard.
        torch.nn.Module.__init__(self)
        self.config = config

        self.vision_model = PerceptionEncoder(
            config.vision_config,
            ACT2FN[config.vision_config.hidden_act],
            quant_config=None,  # vision weights stay BF16
            prefix=add_prefix("vision_model", prefix),
        )
        self.vit_large_projector = ColumnParallelLinear(
            config.vision_config.width * 4,
            config.text_config.hidden_size,
            bias=config.projector_bias,
            gather_output=True,
            quant_config=None,  # projector is BF16
            prefix=add_prefix("vit_large_projector", prefix),
        )
        self.language_model = Step5ForCausalLM(
            config=config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )


EntryClass = [Step5ForCausalLM, Step5ForConditionalGeneration]
