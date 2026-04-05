from __future__ import annotations

import argparse

import torch

from sglang.srt.speculative.dflash.bench.common import time_callable
from sglang.srt.speculative.dflash.experiments.common import (
    assert_dataclass_tensors_equal,
    print_stats_block,
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
    direct_embedding_reference,
    embedding_lookup_reference,
)


def _resolve_int_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    mapping = {
        "int32": torch.int32,
        "i32": torch.int32,
        "int64": torch.int64,
        "i64": torch.int64,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported integer dtype '{name}'.")
    return mapping[normalized]


def _resolve_float_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "f32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported floating dtype '{name}'.")
    return mapping[normalized]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


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


def _run_embedding_reference(fixture: DirectEmbeddingFixture):
    return embedding_lookup_reference(
        embedding_table=fixture.embedding_table,
        query_input_ids=fixture.query_input_ids,
    )


def _run_direct_reference(fixture: DirectEmbeddingFixture):
    return direct_embedding_reference(
        embedding_table=fixture.embedding_table,
        first_token_ids=fixture.first_token_ids,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
    )


def _run_control(fixture: DirectEmbeddingFixture):
    return direct_embedding_control(
        embedding_table=fixture.embedding_table,
        first_token_ids=fixture.first_token_ids,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
    )


def _run_triton(fixture: DirectEmbeddingFixture):
    return direct_embedding_triton(
        embedding_table=fixture.embedding_table,
        first_token_ids=fixture.first_token_ids,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
    )


def _run_jit(fixture: DirectEmbeddingFixture):
    return direct_embedding_jit(
        embedding_table=fixture.embedding_table,
        first_token_ids=fixture.first_token_ids,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
    )


def _run_triton_fast(fixture: DirectEmbeddingFixture, workspace):
    return direct_embedding_triton_fast(
        embedding_table=fixture.embedding_table,
        first_token_ids=fixture.first_token_ids,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
        workspace=workspace,
    )


def _run_jit_fast(fixture: DirectEmbeddingFixture, workspace):
    return direct_embedding_jit_fast(
        embedding_table=fixture.embedding_table,
        first_token_ids=fixture.first_token_ids,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
        workspace=workspace,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA DFlash direct embedding kernels."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bucket-bs", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--embedding-dtypes", default="bf16")
    parser.add_argument("--token-dtypes", default="int32,int64")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if torch.device(args.device).type != "cuda":
        raise ValueError("This benchmark is CUDA-only.")

    embedding_dtypes = [
        _resolve_float_dtype(x) for x in args.embedding_dtypes.split(",") if x.strip()
    ]
    token_dtypes = [
        _resolve_int_dtype(x) for x in args.token_dtypes.split(",") if x.strip()
    ]

    for embedding_dtype in embedding_dtypes:
        base = make_direct_embedding_fixture(
            bucket_bs=args.bucket_bs,
            block_size=args.block_size,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            device=args.device,
            dtype=embedding_dtype,
            seed=args.seed,
        )
        for token_dtype in token_dtypes:
            fixture = _cast_fixture(
                base,
                embedding_dtype=embedding_dtype,
                token_dtype=token_dtype,
            )
            embedding_reference = _run_embedding_reference(fixture)
            direct_reference = _run_direct_reference(fixture)
            control = _run_control(fixture)
            triton_out = _run_triton(fixture)
            jit_out = _run_jit(fixture)
            triton_workspace = create_direct_embedding_workspace(
                bucket_bs=int(fixture.first_token_ids.numel()),
                block_size=fixture.block_size,
                hidden_size=int(fixture.embedding_table.shape[1]),
                dtype=fixture.embedding_table.dtype,
                device=fixture.embedding_table.device,
            )
            jit_workspace = create_direct_embedding_workspace(
                bucket_bs=int(fixture.first_token_ids.numel()),
                block_size=fixture.block_size,
                hidden_size=int(fixture.embedding_table.shape[1]),
                dtype=fixture.embedding_table.dtype,
                device=fixture.embedding_table.device,
            )
            triton_fast_out = _run_triton_fast(fixture, triton_workspace)
            jit_fast_out = _run_jit_fast(fixture, jit_workspace)

            assert_dataclass_tensors_equal(embedding_reference, direct_reference)
            assert_dataclass_tensors_equal(embedding_reference, control)
            assert_dataclass_tensors_equal(embedding_reference, triton_out)
            assert_dataclass_tensors_equal(embedding_reference, jit_out)
            assert_dataclass_tensors_equal(embedding_reference, triton_fast_out)
            assert_dataclass_tensors_equal(embedding_reference, jit_fast_out)

            stats_by_variant = {
                "torch_embedding": time_callable(
                    lambda: _run_embedding_reference(fixture),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=args.device,
                ),
                "torch_direct": time_callable(
                    lambda: _run_control(fixture),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=args.device,
                ),
                "triton": time_callable(
                    lambda: _run_triton(fixture),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=args.device,
                ),
                "jit": time_callable(
                    lambda: _run_jit(fixture),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=args.device,
                ),
                "triton_fast": time_callable(
                    lambda ws=triton_workspace: _run_triton_fast(fixture, ws),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=args.device,
                ),
                "jit_fast": time_callable(
                    lambda ws=jit_workspace: _run_jit_fast(fixture, ws),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=args.device,
                ),
            }
            print_stats_block(
                "direct_embedding_gpu"
                f"[embedding={_dtype_name(embedding_dtype)},token={_dtype_name(token_dtype)}]",
                stats_by_variant,
            )


if __name__ == "__main__":
    main()
