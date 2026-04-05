import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, suite="stage-b-kernel-unit-1-gpu")

from sglang.srt.speculative.dflash.experiments.fixtures import (
    make_commit_materializer_fixture,
    make_prompt_materializer_fixture,
)
from sglang.srt.speculative.dflash.kernels.materializer_packed import (
    create_packed_materializer_workspace,
    materialize_commit_packed_compact_jit,
    materialize_commit_packed_compact_triton,
    materialize_commit_packed_jit,
    materialize_commit_packed_jit_fast,
    materialize_commit_packed_jit_workspace_fast,
    materialize_commit_packed_triton,
    materialize_prompt_packed_jit,
    materialize_prompt_packed_jit_fast,
    materialize_prompt_packed_jit_workspace_fast,
    materialize_prompt_packed_triton,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.parametrize(
    "provider",
    [
        materialize_prompt_packed_triton,
        materialize_prompt_packed_jit,
        materialize_prompt_packed_jit_fast,
        materialize_prompt_packed_jit_workspace_fast,
    ],
)
@pytest.mark.parametrize("group_size", [1, 4])
def test_prompt_materializer_packed_gpu_matches_reference(
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
    fixture = type(fixture)(
        cache=fixture.cache,
        config=fixture.config,
        weights=fixture.weights,
        hidden=fixture.hidden,
        positions=fixture.positions.to(torch.int32),
        slot_ids=fixture.slot_ids.to(torch.int32),
        cos_sin_cache=fixture.cos_sin_cache,
    )
    reference = materialize_prompt_reference(
        cache=fixture.cache.clone(),
        config=fixture.config,
        weights=fixture.weights,
        hidden=fixture.hidden,
        positions=fixture.positions,
        slot_ids=fixture.slot_ids,
    )
    kwargs = {}
    if provider is materialize_prompt_packed_jit_workspace_fast:
        kwargs["workspace"] = create_packed_materializer_workspace(
            config=fixture.config,
            weights=fixture.weights,
            group_size=group_size,
            max_rows=32,
            dtype=fixture.hidden.dtype,
            device=fixture.hidden.device,
            cos_sin_cache=fixture.cos_sin_cache,
        )
    actual = provider(
        cache=fixture.cache.clone(),
        config=fixture.config,
        weights=fixture.weights,
        hidden=fixture.hidden,
        positions=fixture.positions,
        slot_ids=fixture.slot_ids,
        group_size=group_size,
        chunk_size=32,
        cos_sin_cache=fixture.cos_sin_cache,
        **kwargs,
    )
    _assert_cache_close(reference, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.parametrize(
    "provider",
    [
        materialize_commit_packed_triton,
        materialize_commit_packed_jit,
        materialize_commit_packed_compact_triton,
        materialize_commit_packed_compact_jit,
        materialize_commit_packed_jit_fast,
        materialize_commit_packed_jit_workspace_fast,
    ],
)
@pytest.mark.parametrize("group_size", [1, 4])
def test_commit_materializer_packed_gpu_matches_reference(
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
    fixture = type(fixture)(
        cache=fixture.cache,
        config=fixture.config,
        weights=fixture.weights,
        verify_hidden=fixture.verify_hidden,
        positions=fixture.positions.to(torch.int32),
        slot_ids=fixture.slot_ids.to(torch.int32),
        commit_lens=fixture.commit_lens.to(torch.int32),
        cos_sin_cache=fixture.cos_sin_cache,
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
    kwargs = {}
    if provider is materialize_commit_packed_jit_workspace_fast:
        kwargs["workspace"] = create_packed_materializer_workspace(
            config=fixture.config,
            weights=fixture.weights,
            group_size=group_size,
            max_rows=int(
                fixture.verify_hidden.shape[0] * fixture.verify_hidden.shape[1]
            ),
            dtype=fixture.verify_hidden.dtype,
            device=fixture.verify_hidden.device,
            cos_sin_cache=fixture.cos_sin_cache,
        )
    actual = provider(
        cache=fixture.cache.clone(),
        config=fixture.config,
        weights=fixture.weights,
        verify_hidden=fixture.verify_hidden,
        positions=fixture.positions,
        slot_ids=fixture.slot_ids,
        commit_lens=fixture.commit_lens,
        group_size=group_size,
        cos_sin_cache=fixture.cos_sin_cache,
        **kwargs,
    )
    _assert_cache_close(reference, actual)
