import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, suite="stage-b-kernel-unit-1-gpu")

from sglang.srt.speculative.dflash.experiments.fixtures import (
    make_commit_write_fixture,
    make_prompt_write_fixture,
)
from sglang.srt.speculative.dflash.kernels.kv_prefix_write_jit import (
    write_commit_prefix_jit,
    write_prompt_jit,
)
from sglang.srt.speculative.dflash.kernels.kv_prefix_write_triton import (
    write_commit_prefix_triton,
    write_prompt_triton,
)
from sglang.srt.speculative.dflash.reference.kv_prefix_write import (
    write_commit_prefix_reference,
    write_prompt_reference,
)


def _assert_cache_equal(expected, actual) -> None:
    assert torch.equal(expected.k_cache, actual.k_cache)
    assert torch.equal(expected.v_cache, actual.v_cache)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.parametrize("writer", [write_prompt_triton, write_prompt_jit])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_prompt_write_gpu_matches_reference(writer, slot_dtype: torch.dtype) -> None:
    fixture = make_prompt_write_fixture(
        num_layers=4,
        num_kv_heads=2,
        head_dim=64,
        num_slots=1024,
        num_tokens=96,
        device="cuda",
        seed=7,
    )
    fixture = type(fixture)(
        cache=type(fixture.cache)(
            k_cache=fixture.cache.k_cache.to(dtype=torch.bfloat16),
            v_cache=fixture.cache.v_cache.to(dtype=torch.bfloat16),
        ),
        config=fixture.config,
        slot_ids=fixture.slot_ids.to(dtype=slot_dtype),
        cache_k=fixture.cache_k.to(dtype=torch.bfloat16),
        cache_v=fixture.cache_v.to(dtype=torch.bfloat16),
        dummy_slot_id=fixture.dummy_slot_id,
    )
    reference = write_prompt_reference(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids=fixture.slot_ids,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
    )
    actual = writer(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids=fixture.slot_ids,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
    )
    _assert_cache_equal(reference, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.parametrize(
    "writer", [write_commit_prefix_triton, write_commit_prefix_jit]
)
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("commit_len_dtype", [torch.int32, torch.int64])
def test_commit_write_gpu_matches_reference(
    writer,
    slot_dtype: torch.dtype,
    commit_len_dtype: torch.dtype,
) -> None:
    fixture = make_commit_write_fixture(
        num_layers=4,
        num_kv_heads=2,
        head_dim=64,
        num_slots=2048,
        batch_size=12,
        block_size=8,
        device="cuda",
        seed=11,
    )
    fixture = type(fixture)(
        cache=type(fixture.cache)(
            k_cache=fixture.cache.k_cache.to(dtype=torch.bfloat16),
            v_cache=fixture.cache.v_cache.to(dtype=torch.bfloat16),
        ),
        config=fixture.config,
        slot_ids_2d=fixture.slot_ids_2d.to(dtype=slot_dtype),
        commit_lens=fixture.commit_lens.to(dtype=commit_len_dtype),
        cache_k=fixture.cache_k.to(dtype=torch.bfloat16),
        cache_v=fixture.cache_v.to(dtype=torch.bfloat16),
        dummy_slot_id=fixture.dummy_slot_id,
    )
    reference = write_commit_prefix_reference(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids_2d=fixture.slot_ids_2d,
        commit_lens=fixture.commit_lens,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
    )
    actual = writer(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids_2d=fixture.slot_ids_2d,
        commit_lens=fixture.commit_lens,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
    )
    _assert_cache_equal(reference, actual)
