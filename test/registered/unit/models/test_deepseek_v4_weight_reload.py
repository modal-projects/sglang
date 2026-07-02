from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM, MQALayer


def _make_attention(*, tp_size: int = 2, tp_rank: int = 1) -> MQALayer:
    attention = object.__new__(MQALayer)
    torch.nn.Module.__init__(attention)
    attention.n_heads = 8
    attention.attn_tp_size = tp_size
    attention.attn_tp_rank = tp_rank
    attention.n_local_heads = attention.n_heads // tp_size
    attention.attn_sink = torch.nn.Parameter(
        torch.arange(attention.n_heads, dtype=torch.float32)
    )
    attention._attn_sink_local = None
    return attention


def test_refresh_attn_sink_cache_updates_stable_local_storage():
    attention = _make_attention()

    attention.refresh_attn_sink_cache()
    local_sink = attention._attn_sink_local
    assert local_sink is not None
    assert local_sink.shape == (64,)
    torch.testing.assert_close(local_sink[:4], torch.arange(4, 8, dtype=torch.float32))
    torch.testing.assert_close(local_sink[4:], torch.zeros(60))
    attention.attn_sink.data.add_(10)
    attention.refresh_attn_sink_cache()

    assert attention._attn_sink_local is local_sink
    torch.testing.assert_close(
        local_sink[:4], torch.arange(14, 18, dtype=torch.float32)
    )


def test_refresh_attn_sink_cache_rejects_layout_changes():
    attention = _make_attention()
    attention._attn_sink_local = torch.empty(32)

    with pytest.raises(RuntimeError, match="layout changed"):
        attention.refresh_attn_sink_cache()


def test_single_rank_uses_parameter_directly(monkeypatch):
    monkeypatch.setattr(
        deepseek_v4,
        "get_cp_decode_attn_tp_ctx",
        lambda: SimpleNamespace(is_enabled=False),
    )
    attention = _make_attention(tp_size=1, tp_rank=0)

    attention.refresh_attn_sink_cache()

    assert attention._attn_sink_local is None
    assert attention._local_attn_sink() is attention.attn_sink


def test_cp_decode_attention_sink_is_ready_before_first_forward(monkeypatch):
    monkeypatch.setattr(
        deepseek_v4,
        "get_cp_decode_attn_tp_ctx",
        lambda: SimpleNamespace(is_enabled=True, decode_tp_rank=1, decode_tp_size=2),
    )
    attention = _make_attention(tp_size=1, tp_rank=0)

    attention.refresh_attn_sink_cache()

    local_sink = attention._attn_sink_local
    assert local_sink is not None
    torch.testing.assert_close(local_sink[:4], torch.arange(4, 8, dtype=torch.float32))
    attention.attn_sink.data.add_(10)
    attention.refresh_attn_sink_cache()
    assert attention._attn_sink_local is local_sink
    torch.testing.assert_close(
        local_sink[:4], torch.arange(14, 18, dtype=torch.float32)
    )


def test_post_load_refreshes_attention_sink_cache(monkeypatch):
    monkeypatch.setattr(deepseek_v4, "_FP8_WO_A_GEMM", False)
    attention = SimpleNamespace(
        compress_ratio=0,
        refresh_attn_sink_cache=Mock(),
    )
    layer = SimpleNamespace(
        self_attn=attention,
        refresh_mhc_norm_weight_cache=lambda: None,
    )
    model = object.__new__(DeepseekV4ForCausalLM)
    object.__setattr__(
        model,
        "model",
        SimpleNamespace(start_layer=0, end_layer=1, layers=[layer]),
    )

    model.post_load_weights()

    attention.refresh_attn_sink_cache.assert_called_once_with()


def test_nextn_post_load_refreshes_attention_sink_cache(monkeypatch):
    monkeypatch.setattr(deepseek_v4, "_FP8_WO_A_GEMM", False)
    attention = SimpleNamespace(refresh_attn_sink_cache=Mock())
    model = object.__new__(DeepseekV4ForCausalLM)
    object.__setattr__(
        model,
        "model",
        SimpleNamespace(decoder=SimpleNamespace(self_attn=attention)),
    )

    model.post_load_weights(is_nextn=True)

    attention.refresh_attn_sink_cache.assert_called_once_with()
