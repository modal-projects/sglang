from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.models import inkling as inkling_module


class _Embedding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((input_ids.shape[0], self.hidden_size))


class _InputProjection(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.new_zeros((inputs.shape[0], self.hidden_size))


class _RecordingSconv(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, positions, forward_batch):
        self.calls += 1
        return hidden_states + 2


class _ScatteredTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, tp_size: int, residual_value: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.residual_value = residual_value
        self.mlp_sconv = _RecordingSconv()
        self.scattered_sconv = True
        self.attn_tp_group = object()

    def forward(self, hidden_states, positions, forward_batch, residual, **kwargs):
        num_tokens = hidden_states.shape[0]
        shard = hidden_states.new_ones((num_tokens, self.hidden_size // self.tp_size))
        residual = hidden_states.new_full(
            (num_tokens, self.hidden_size), self.residual_value
        )
        return shard, residual


def test_inkling_mtp_finishes_deferred_scattered_mlp_sconv(monkeypatch):
    hidden_size = 16
    tp_size = 4
    block = _ScatteredTransformerBlock(hidden_size, tp_size, residual_value=5)
    layer = inkling_module.InklingMTPLayer.__new__(inkling_module.InklingMTPLayer)
    nn.Module.__init__(layer)
    layer.embed_tokens = _Embedding(hidden_size)
    layer.main_model_embed_norm = None
    layer.hidden_norm = nn.Identity()
    layer.embed_norm = nn.Identity()
    layer.input_proj = _InputProjection(hidden_size)
    layer.transformer_block = block
    layer.log_scaling_n_floor = None
    layer.log_scaling_alpha = 0.1
    layer.chain_norm = None
    layer.mup_width_multiplier = None

    gathered = []

    def _all_gather_hidden(shard, group):
        gathered.append((shard.shape, group))
        return shard.repeat(1, tp_size)

    monkeypatch.setattr(inkling_module, "all_gather_hidden", _all_gather_hidden)
    forward_batch = SimpleNamespace(
        spec_info=SimpleNamespace(hidden_states=torch.zeros((2, hidden_size))),
        forward_mode=SimpleNamespace(is_idle=lambda: False),
    )

    output, chain_hidden = layer(
        torch.tensor([1, 2]), torch.tensor([0, 1]), forward_batch
    )

    # block output 1 -> deferred sconv output 3 -> gathered to H -> residual +5.
    assert output.shape == (2, hidden_size)
    torch.testing.assert_close(output, torch.full((2, hidden_size), 8.0))
    torch.testing.assert_close(chain_hidden, output)
    assert block.mlp_sconv.calls == 1
    assert gathered == [((2, hidden_size // tp_size), block.attn_tp_group)]
