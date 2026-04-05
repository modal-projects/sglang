import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-kernel-unit-1-gpu")

from sglang.srt.speculative.dflash.experiments.common import (
    assert_dataclass_tensors_equal,
)
from sglang.srt.speculative.dflash.experiments.fixtures import (
    make_accept_publish_fixture,
)
from sglang.srt.speculative.dflash.kernels.accept_bonus_jit import (
    accept_bonus_jit,
)
from sglang.srt.speculative.dflash.kernels.accept_bonus_triton import (
    accept_bonus_triton,
)
from sglang.srt.speculative.dflash.kernels.accept_publish_jit import (
    accept_publish_jit,
)
from sglang.srt.speculative.dflash.kernels.accept_publish_triton import (
    accept_publish_triton,
)
from sglang.srt.speculative.dflash.kernels.publish_state_jit import (
    publish_state_jit,
)
from sglang.srt.speculative.dflash.kernels.publish_state_triton import (
    publish_state_triton,
)
from sglang.srt.speculative.dflash.reference.core import accept_publish_reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.parametrize("token_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("req_pool_dtype", [torch.int32, torch.int64])
def test_accept_publish_gpu_matches_reference(
    token_dtype: torch.dtype,
    req_pool_dtype: torch.dtype,
) -> None:
    fixture = make_accept_publish_fixture(
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
        emit_ids=fixture.emit_ids.to(dtype=token_dtype),
        target_top1=fixture.target_top1.to(dtype=token_dtype),
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids.to(dtype=token_dtype),
        stop_token_ids=fixture.stop_token_ids.to(dtype=token_dtype),
    )

    reference = accept_publish_reference(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )

    separate_triton_accept = accept_bonus_triton(
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )
    separate_triton_state = publish_state_triton(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        commit_lens=separate_triton_accept.commit_lens,
        bonus_ids=separate_triton_accept.bonus_ids,
        gpu_stop_flags=separate_triton_accept.gpu_stop_flags,
    )
    fused_triton = accept_publish_triton(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )

    separate_jit_accept = accept_bonus_jit(
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )
    separate_jit_state = publish_state_jit(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        commit_lens=separate_jit_accept.commit_lens,
        bonus_ids=separate_jit_accept.bonus_ids,
        gpu_stop_flags=separate_jit_accept.gpu_stop_flags,
    )
    fused_jit = accept_publish_jit(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )

    separate_triton = type(reference)(
        accept=separate_triton_accept,
        state=separate_triton_state,
    )
    separate_jit = type(reference)(
        accept=separate_jit_accept,
        state=separate_jit_state,
    )

    assert_dataclass_tensors_equal(reference, separate_triton)
    assert_dataclass_tensors_equal(reference, fused_triton)
    assert_dataclass_tensors_equal(reference, separate_jit)
    assert_dataclass_tensors_equal(reference, fused_jit)
