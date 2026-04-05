from __future__ import annotations

import pytest
import torch

from sglang.srt.speculative.dflash.experiments.fixtures import (
    make_commit_materializer_fixture,
    make_prompt_materializer_fixture,
)
from sglang.srt.speculative.dflash.kernels.post_projection_jit import (
    postprocess_commit_jit,
    postprocess_prompt_jit,
)
from sglang.srt.speculative.dflash.kernels.post_projection_triton import (
    postprocess_commit_triton,
    postprocess_prompt_triton,
)
from sglang.srt.speculative.dflash.kernels.raw_kv_projection import (
    project_raw_commit_control,
    project_raw_prompt_control,
)
from sglang.srt.speculative.dflash.reference.materializer import (
    materialize_commit_reference,
    materialize_prompt_reference,
)


def _assert_cache_close(
    expected, actual, *, atol: float = 1e-1, rtol: float = 1e-1
) -> None:
    assert torch.allclose(expected.k_cache, actual.k_cache, atol=atol, rtol=rtol)
    assert torch.allclose(expected.v_cache, actual.v_cache, atol=atol, rtol=rtol)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "provider", [postprocess_prompt_triton, postprocess_prompt_jit]
)
@pytest.mark.parametrize("group_size", [1, 4])
def test_prompt_post_projection_gpu_matches_reference(
    provider, group_size: int
) -> None:
    fixture = make_prompt_materializer_fixture(
        num_layers=4,
        hidden_size=256,
        num_kv_heads=4,
        head_dim=64,
        rotary_dim=64,
        num_slots=256,
        num_tokens=64,
        device="cuda",
        dtype=torch.bfloat16,
        seed=0,
    )
    projected = project_raw_prompt_control(
        config=fixture.config,
        weights=fixture.weights,
        hidden=fixture.hidden,
        positions=fixture.positions,
        group_size=group_size,
        chunk_size=32,
    )
    reference = materialize_prompt_reference(
        cache=fixture.cache.clone(),
        config=fixture.config,
        weights=fixture.weights,
        hidden=fixture.hidden,
        positions=fixture.positions,
        slot_ids=fixture.slot_ids,
    )
    actual = provider(
        cache=fixture.cache.clone(),
        config=fixture.config,
        weights=fixture.weights,
        projected=projected,
        positions=fixture.positions,
        slot_ids=fixture.slot_ids,
        cos_sin_cache=fixture.cos_sin_cache,
    )
    _assert_cache_close(reference, actual)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "provider", [postprocess_commit_triton, postprocess_commit_jit]
)
@pytest.mark.parametrize("group_size", [1, 4])
def test_commit_post_projection_gpu_matches_reference(
    provider, group_size: int
) -> None:
    fixture = make_commit_materializer_fixture(
        num_layers=4,
        hidden_size=256,
        num_kv_heads=4,
        head_dim=64,
        rotary_dim=64,
        num_slots=512,
        batch_size=8,
        block_size=8,
        device="cuda",
        dtype=torch.bfloat16,
        seed=0,
    )
    projected = project_raw_commit_control(
        config=fixture.config,
        weights=fixture.weights,
        verify_hidden=fixture.verify_hidden,
        positions=fixture.positions,
        group_size=group_size,
    )
    reference = materialize_commit_reference(
        cache=fixture.cache.clone(),
        config=fixture.config,
        weights=fixture.weights,
        verify_hidden=fixture.verify_hidden,
        positions=fixture.positions,
        slot_ids=fixture.slot_ids,
        commit_lens=fixture.commit_lens,
    )
    actual = provider(
        cache=fixture.cache.clone(),
        config=fixture.config,
        weights=fixture.weights,
        projected=projected,
        positions=fixture.positions,
        slot_ids_2d=fixture.slot_ids,
        commit_lens=fixture.commit_lens,
        cos_sin_cache=fixture.cos_sin_cache,
    )
    _assert_cache_close(reference, actual)
