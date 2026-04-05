import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, suite="stage-b-kernel-unit-1-gpu")

from sglang.srt.speculative.dflash.experiments.common import (
    assert_dataclass_tensors_equal,
)
from sglang.srt.speculative.dflash.experiments.fixtures import (
    DirectEmbeddingFixture,
    make_direct_embedding_fixture,
)
from sglang.srt.speculative.dflash.kernels.direct_embedding import (
    create_direct_embedding_workspace,
    direct_embedding_control,
)
from sglang.srt.speculative.dflash.kernels.direct_embedding_jit import (
    direct_embedding_jit,
    direct_embedding_jit_fast,
)
from sglang.srt.speculative.dflash.kernels.direct_embedding_triton import (
    direct_embedding_triton,
    direct_embedding_triton_fast,
)
from sglang.srt.speculative.dflash.reference.direct_embedding import (
    embedding_lookup_reference,
)


def _cast_fixture(
    fixture: DirectEmbeddingFixture,
    *,
    embedding_dtype: torch.dtype,
    token_dtype: torch.dtype,
) -> DirectEmbeddingFixture:
    return DirectEmbeddingFixture(
        embedding_table=fixture.embedding_table.to(dtype=embedding_dtype),
        query_input_ids=fixture.query_input_ids.to(dtype=token_dtype),
        first_token_ids=fixture.first_token_ids.to(dtype=token_dtype),
        mask_token_id=fixture.mask_token_id,
        block_size=fixture.block_size,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.parametrize("embedding_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("token_dtype", [torch.int32, torch.int64])
def test_direct_embedding_gpu_matches_reference(
    embedding_dtype: torch.dtype,
    token_dtype: torch.dtype,
) -> None:
    fixture = make_direct_embedding_fixture(
        bucket_bs=16,
        block_size=16,
        vocab_size=32768,
        hidden_size=1024,
        device="cuda",
        dtype=embedding_dtype,
        seed=11,
    )
    fixture = _cast_fixture(
        fixture,
        embedding_dtype=embedding_dtype,
        token_dtype=token_dtype,
    )

    reference = embedding_lookup_reference(
        embedding_table=fixture.embedding_table,
        query_input_ids=fixture.query_input_ids,
    )
    control = direct_embedding_control(
        embedding_table=fixture.embedding_table,
        first_token_ids=fixture.first_token_ids,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
    )
    triton_out = direct_embedding_triton(
        embedding_table=fixture.embedding_table,
        first_token_ids=fixture.first_token_ids,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
    )
    jit_out = direct_embedding_jit(
        embedding_table=fixture.embedding_table,
        first_token_ids=fixture.first_token_ids,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
    )
    triton_fast_out = direct_embedding_triton_fast(
        embedding_table=fixture.embedding_table,
        first_token_ids=fixture.first_token_ids,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
        workspace=create_direct_embedding_workspace(
            bucket_bs=int(fixture.first_token_ids.numel()),
            block_size=fixture.block_size,
            hidden_size=int(fixture.embedding_table.shape[1]),
            dtype=fixture.embedding_table.dtype,
            device=fixture.embedding_table.device,
        ),
    )
    jit_fast_out = direct_embedding_jit_fast(
        embedding_table=fixture.embedding_table,
        first_token_ids=fixture.first_token_ids,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
        workspace=create_direct_embedding_workspace(
            bucket_bs=int(fixture.first_token_ids.numel()),
            block_size=fixture.block_size,
            hidden_size=int(fixture.embedding_table.shape[1]),
            dtype=fixture.embedding_table.dtype,
            device=fixture.embedding_table.device,
        ),
    )

    assert_dataclass_tensors_equal(reference, control)
    assert_dataclass_tensors_equal(reference, triton_out)
    assert_dataclass_tensors_equal(reference, jit_out)
    assert_dataclass_tensors_equal(reference, triton_fast_out)
    assert_dataclass_tensors_equal(reference, jit_fast_out)
