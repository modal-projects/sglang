import sys
from types import SimpleNamespace

import pytest
from sglang.srt.model_executor.model_runner_components import spec_aux_hidden_state
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    _map_muse_target_layer_ids,
    _resolve_dflash_draft_cell_size,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    ("target_model_type", "draft_architecture", "expected"),
    [
        ("muse_glimmer", "MuseGlimmerAssistantModel", [2, 14, 26, 38, 50]),
        ("muse_glimmer", "DFlash2DraftModel", [2, 14, 26, 38, 50]),
        ("muse_glimmer", "DFlashDraftModel", [1, 13, 25, 37, 49]),
        ("qwen3", "DFlash2DraftModel", [1, 13, 25, 37, 49]),
        ("qwen3", "MuseGlimmerAssistantModel", [1, 13, 25, 37, 49]),
    ],
)
def test_muse_target_layer_id_mapping(target_model_type, draft_architecture, expected):
    """The +1 belongs to Muse targets, which report layer outputs where the rest
    report layer inputs. The draft architecture alone does not earn it."""
    assert (
        _map_muse_target_layer_ids(
            target_hf_config=SimpleNamespace(model_type=target_model_type),
            draft_hf_config=SimpleNamespace(architectures=[draft_architecture]),
            layer_ids=[1, 13, 25, 37, 49],
        )
        == expected
    )


def test_dflash_draft_cell_uses_attention_tp_size(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        spec_aux_hidden_state,
        "get_model",
        lambda: SimpleNamespace(kv_cache_dtype="auto"),
    )
    monkeypatch.setattr(
        spec_aux_hidden_state,
        "get_spec",
        lambda: SimpleNamespace(
            speculative_draft_kv_cache_dtype=None,
            speculative_draft_attention_backend="flashinfer",
        ),
    )
    monkeypatch.setattr(
        spec_aux_hidden_state,
        "get_parallel",
        lambda: SimpleNamespace(tp_size=4, attn_tp_size=2),
    )

    from sglang.srt.mem_cache import kv_cache_dtype
    from sglang.srt.speculative import dflash_utils

    monkeypatch.setattr(
        kv_cache_dtype,
        "configure_kv_cache_dtype",
        lambda **kwargs: (None, "bf16"),
    )

    def fake_cell_size(**kwargs):
        captured.update(kwargs)
        return 123

    monkeypatch.setattr(
        dflash_utils,
        "dflash_draft_cell_size_per_token",
        fake_cell_size,
    )

    assert (
        _resolve_dflash_draft_cell_size(
            draft_model_config=SimpleNamespace(dtype="bf16"),
            draft_num_layers=3,
        )
        == 123
    )
    assert captured["tp_size"] == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
