from types import SimpleNamespace

import torch

from sglang.srt.model_loader.utils import STABLE_WEIGHT_SOURCE_ATTR
from sglang.srt.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _BatchOwner:
    def __init__(self):
        self.immediate_calls = []
        self.batched_calls = []
        self.call_order = []

    def supports_deferred_weight_copies(self):
        return True

    def weight_loader(self, *args, **kwargs):
        self.immediate_calls.append((args, kwargs))
        self.call_order.append("immediate")

    def load_weights_batched(self, calls, *, executor):
        self.batched_calls.extend(calls)
        self.call_order.append("batched")


def _weight_loading_model(owner):
    param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    param.weight_loader = owner.weight_loader
    return SimpleNamespace(
        config=SimpleNamespace(
            num_experts=1,
            tie_word_embeddings=False,
            encoder_only=False,
        ),
        enable_shared_expert_fusion=False,
        num_fused_shared_experts=0,
        pp_group=SimpleNamespace(is_last_rank=False),
        start_layer=0,
        end_layer=1,
        model=SimpleNamespace(start_layer=0, end_layer=1, layers=[]),
        named_parameters=lambda remove_duplicate=False: [
            ("model.layers.0.mlp.experts.w13_weight", param)
        ],
    )


def test_load_weights_only_defers_explicitly_stable_sources():
    owner = _BatchOwner()
    model = _weight_loading_model(owner)
    stable = torch.ones(1)
    setattr(stable, STABLE_WEIGHT_SOURCE_ATTR, True)

    Qwen3_5MoeForConditionalGeneration.load_weights(
        model,
        [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", stable),
            ("model.layers.0.mlp.experts.0.up_proj.weight", torch.ones(1)),
        ],
    )

    assert len(owner.batched_calls) == 1
    assert owner.batched_calls[0][0][1] is stable
    assert len(owner.immediate_calls) == 1
    assert owner.call_order == ["batched", "immediate"]
