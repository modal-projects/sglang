import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, suite="stage-b-kernel-unit-1-gpu")

from sglang.srt.speculative.dflash.experiments.common import (
    assert_dataclass_tensors_equal,
)
from sglang.srt.speculative.dflash.experiments.fixtures import (
    make_accept_bonus_fixture,
)
from sglang.srt.speculative.dflash.kernels.accept_bonus_jit import (
    accept_bonus_jit,
)
from sglang.srt.speculative.dflash.kernels.accept_bonus_triton import (
    accept_bonus_triton,
)
from sglang.srt.speculative.dflash.reference.core import accept_bonus_reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.parametrize("token_dtype", [torch.int32, torch.int64])
def test_accept_bonus_gpu_matches_reference(token_dtype: torch.dtype) -> None:
    fixture = make_accept_bonus_fixture(
        bucket_bs=16,
        block_size=16,
        device="cuda",
        seed=7,
    )
    fixture = type(fixture)(
        emit_ids=fixture.emit_ids.to(dtype=token_dtype),
        target_top1=fixture.target_top1.to(dtype=token_dtype),
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids.to(dtype=token_dtype),
        stop_token_ids=fixture.stop_token_ids.to(dtype=token_dtype),
    )
    reference = accept_bonus_reference(
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )
    triton_out = accept_bonus_triton(
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )
    jit_out = accept_bonus_jit(
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )
    assert_dataclass_tensors_equal(reference, triton_out)
    assert_dataclass_tensors_equal(reference, jit_out)
