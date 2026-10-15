"""Step-5 MTP (NEXTN) draft model.

Step-5 ships three chained MTP layers (model.layers.92..94): dense MLP, gated
sliding-window attention, rope 1e4, SwiGLU limit 7 -- all read from the trailing
entries of the per-layer config lists. The Step-3.5 draft hardcodes the block's
layer index (45), which for Step-5 would select a MoE layer; this builds the
block at its real index (num_hidden_layers + draft_model_idx) so the per-layer
lookups resolve to the MTP entries. draft_model_idx picks which chained layer to
load (0 for single-layer NEXTN, 0..2 under multi-layer EAGLE).

The target architecture is Step5ForConditionalGeneration; the draft receives the
same (VL) config and unwraps text_config here.
"""

from typing import Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.layer_boundary import layer_stack
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.models.step3p5 import Step3p5DecoderLayer
from sglang.srt.models.step3p5_mtp import (
    SharedHead,
    Step3p5AMultiTokenPredictor,
    Step3p5MTP,
)
from sglang.srt.utils import add_prefix


def _text_config(config: PretrainedConfig) -> PretrainedConfig:
    return getattr(config, "text_config", None) or config


class Step5MultiTokenPredictor(Step3p5AMultiTokenPredictor):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        mtp_layer_idx: int = 0,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        layer_id = config.num_hidden_layers + mtp_layer_idx
        self.enorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = SharedHead(config=config, quant_config=quant_config)
        with layer_stack():
            # No quant_config: the MTP block is BF16 in the checkpoint.
            self.mtp_block = Step3p5DecoderLayer(
                config=config, layer_id=layer_id, prefix=f"{prefix}.mtp_block"
            )
        self.lm_head = self.shared_head.head


class Step5MTP(Step3p5MTP):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        draft_model_idx: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        config = _text_config(config)
        self.config = config
        self.quant_config = quant_config
        self.draft_model_idx = 0 if draft_model_idx is None else draft_model_idx
        self.model = Step5MultiTokenPredictor(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
            mtp_layer_idx=self.draft_model_idx,
        )
        self.logits_processor = LogitsProcessor(config)
        self.lm_head = self.model.lm_head

    def get_attention_sliding_window_size(self) -> Optional[int]:
        # The MTP block is a sliding-window layer (config.layer_types[92..94]);
        # the draft ModelRunner needs the window so the attention backend
        # builds window_kv_indptr for it.
        layer_id = self.config.num_hidden_layers + self.draft_model_idx
        if self.config.layer_types[layer_id] == "sliding_attention":
            return self.config.sliding_window
        return None


EntryClass = [Step5MTP]
