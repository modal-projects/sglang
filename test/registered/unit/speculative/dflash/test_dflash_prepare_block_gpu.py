import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, suite="stage-b-kernel-unit-1-gpu")

from sglang.srt.speculative.dflash.experiments.common import (
    assert_dataclass_tensors_equal,
)
from sglang.srt.speculative.dflash.experiments.fixtures import (
    make_prepare_block_fixture,
)
from sglang.srt.speculative.dflash.kernels.prepare_block import (
    create_prepare_block_workspace,
)
from sglang.srt.speculative.dflash.kernels.prepare_block_jit import (
    prepare_block_jit,
    prepare_block_jit_fast,
)
from sglang.srt.speculative.dflash.kernels.prepare_block_triton import (
    prepare_block_triton,
    prepare_block_triton_fast,
)
from sglang.srt.speculative.dflash.reference.core import prepare_block_reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.parametrize("req_pool_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("req_to_token_dtype", [torch.int32, torch.int64])
def test_prepare_block_gpu_matches_reference(
    req_pool_dtype: torch.dtype,
    req_to_token_dtype: torch.dtype,
) -> None:
    fixture = make_prepare_block_fixture(
        bucket_bs=16,
        num_req_slots=32,
        req_to_token_width=256,
        block_size=16,
        device="cuda",
        seed=13,
    )
    fixture = type(fixture)(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices.to(dtype=req_pool_dtype),
        req_generation=fixture.req_generation,
        req_to_token=fixture.req_to_token.to(dtype=req_to_token_dtype),
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
    )
    reference = prepare_block_reference(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        req_to_token=fixture.req_to_token,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
    )
    triton_out = prepare_block_triton(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        req_to_token=fixture.req_to_token,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
    )
    jit_out = prepare_block_jit(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        req_to_token=fixture.req_to_token,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
    )
    triton_fast_out = prepare_block_triton_fast(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        req_to_token=fixture.req_to_token,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
        workspace=create_prepare_block_workspace(
            bucket_bs=int(fixture.req_pool_indices.numel()),
            block_size=fixture.block_size,
            state_dtype=fixture.state.next_verified_id.dtype,
            token_dtype=fixture.req_to_token.dtype,
            device=fixture.req_to_token.device,
        ),
    )
    jit_fast_out = prepare_block_jit_fast(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        req_to_token=fixture.req_to_token,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
        workspace=create_prepare_block_workspace(
            bucket_bs=int(fixture.req_pool_indices.numel()),
            block_size=fixture.block_size,
            state_dtype=fixture.state.next_verified_id.dtype,
            token_dtype=fixture.req_to_token.dtype,
            device=fixture.req_to_token.device,
        ),
    )
    assert_dataclass_tensors_equal(reference, triton_out)
    assert_dataclass_tensors_equal(reference, jit_out)
    assert_dataclass_tensors_equal(reference, triton_fast_out)
    assert_dataclass_tensors_equal(reference, jit_fast_out)
