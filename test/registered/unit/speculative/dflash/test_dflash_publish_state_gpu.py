import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, suite="stage-b-kernel-unit-1-gpu")

from sglang.srt.speculative.dflash.experiments.common import (
    assert_dataclass_tensors_equal,
)
from sglang.srt.speculative.dflash.experiments.fixtures import (
    make_publish_state_fixture,
)
from sglang.srt.speculative.dflash.kernels.publish_state_jit import (
    publish_state_jit,
)
from sglang.srt.speculative.dflash.kernels.publish_state_triton import (
    publish_state_triton,
)
from sglang.srt.speculative.dflash.reference.core import publish_state_reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.parametrize("req_pool_dtype", [torch.int32, torch.int64])
def test_publish_state_gpu_matches_reference(req_pool_dtype: torch.dtype) -> None:
    fixture = make_publish_state_fixture(
        bucket_bs=16,
        num_req_slots=32,
        req_to_token_width=256,
        block_size=16,
        device="cuda",
        seed=11,
    )
    fixture = type(fixture)(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices.to(dtype=req_pool_dtype),
        req_generation=fixture.req_generation,
        commit_lens=fixture.commit_lens,
        bonus_ids=fixture.bonus_ids,
        gpu_stop_flags=fixture.gpu_stop_flags,
    )
    reference = publish_state_reference(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        commit_lens=fixture.commit_lens,
        bonus_ids=fixture.bonus_ids,
        gpu_stop_flags=fixture.gpu_stop_flags,
    )
    triton_out = publish_state_triton(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        commit_lens=fixture.commit_lens,
        bonus_ids=fixture.bonus_ids,
        gpu_stop_flags=fixture.gpu_stop_flags,
    )
    jit_out = publish_state_jit(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        commit_lens=fixture.commit_lens,
        bonus_ids=fixture.bonus_ids,
        gpu_stop_flags=fixture.gpu_stop_flags,
    )
    assert_dataclass_tensors_equal(reference, triton_out)
    assert_dataclass_tensors_equal(reference, jit_out)
