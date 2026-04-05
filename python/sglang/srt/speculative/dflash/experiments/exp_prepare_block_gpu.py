from __future__ import annotations

import argparse

import torch

from sglang.srt.speculative.dflash.bench.common import time_callable
from sglang.srt.speculative.dflash.experiments.common import (
    assert_dataclass_tensors_equal,
    print_stats_block,
)
from sglang.srt.speculative.dflash.experiments.fixtures import (
    PrepareBlockFixture,
    make_prepare_block_fixture,
)
from sglang.srt.speculative.dflash.kernels.prepare_block import (
    create_prepare_block_workspace,
    prepare_block_control,
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


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _cast_fixture(
    fixture: PrepareBlockFixture,
    *,
    req_pool_dtype: torch.dtype,
    req_to_token_dtype: torch.dtype,
) -> PrepareBlockFixture:
    return PrepareBlockFixture(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices.to(dtype=req_pool_dtype),
        req_generation=fixture.req_generation.to(dtype=fixture.state.generation.dtype),
        req_to_token=fixture.req_to_token.to(dtype=req_to_token_dtype),
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
    )


def _run_reference(fixture: PrepareBlockFixture):
    return prepare_block_reference(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        req_to_token=fixture.req_to_token,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
    )


def _run_control(fixture: PrepareBlockFixture):
    return prepare_block_control(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        req_to_token=fixture.req_to_token,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
    )


def _run_triton(fixture: PrepareBlockFixture):
    return prepare_block_triton(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        req_to_token=fixture.req_to_token,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
    )


def _run_jit(fixture: PrepareBlockFixture):
    return prepare_block_jit(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        req_to_token=fixture.req_to_token,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
    )


def _run_triton_fast(fixture: PrepareBlockFixture, workspace):
    return prepare_block_triton_fast(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        req_to_token=fixture.req_to_token,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
        workspace=workspace,
    )


def _run_jit_fast(fixture: PrepareBlockFixture, workspace):
    return prepare_block_jit_fast(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        req_to_token=fixture.req_to_token,
        block_size=fixture.block_size,
        mask_token_id=fixture.mask_token_id,
        workspace=workspace,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA DFlash prepare_block kernels."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bucket-bs", type=int, default=32)
    parser.add_argument("--num-req-slots", type=int, default=64)
    parser.add_argument("--req-to-token-width", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--req-pool-dtypes", default="int32,int64")
    parser.add_argument("--req-to-token-dtypes", default="int64")
    args = parser.parse_args()

    if torch.device(args.device).type != "cuda":
        raise ValueError("This benchmark is CUDA-only.")

    base = make_prepare_block_fixture(
        bucket_bs=args.bucket_bs,
        num_req_slots=args.num_req_slots,
        req_to_token_width=args.req_to_token_width,
        block_size=args.block_size,
        device=args.device,
        seed=args.seed,
    )

    req_pool_dtypes = [
        _resolve_int_dtype(x) for x in args.req_pool_dtypes.split(",") if x.strip()
    ]
    req_to_token_dtypes = [
        _resolve_int_dtype(x) for x in args.req_to_token_dtypes.split(",") if x.strip()
    ]

    for req_pool_dtype in req_pool_dtypes:
        for req_to_token_dtype in req_to_token_dtypes:
            fixture = _cast_fixture(
                base,
                req_pool_dtype=req_pool_dtype,
                req_to_token_dtype=req_to_token_dtype,
            )
            reference = _run_reference(fixture)
            control = _run_control(fixture)
            triton_out = _run_triton(fixture)
            jit_out = _run_jit(fixture)
            fast_workspace_triton = create_prepare_block_workspace(
                bucket_bs=int(fixture.req_pool_indices.numel()),
                block_size=fixture.block_size,
                state_dtype=fixture.state.next_verified_id.dtype,
                token_dtype=fixture.req_to_token.dtype,
                device=fixture.req_to_token.device,
            )
            fast_workspace_jit = create_prepare_block_workspace(
                bucket_bs=int(fixture.req_pool_indices.numel()),
                block_size=fixture.block_size,
                state_dtype=fixture.state.next_verified_id.dtype,
                token_dtype=fixture.req_to_token.dtype,
                device=fixture.req_to_token.device,
            )
            triton_fast_out = _run_triton_fast(fixture, fast_workspace_triton)
            jit_fast_out = _run_jit_fast(fixture, fast_workspace_jit)
            assert_dataclass_tensors_equal(reference, control)
            assert_dataclass_tensors_equal(reference, triton_out)
            assert_dataclass_tensors_equal(reference, jit_out)
            assert_dataclass_tensors_equal(reference, triton_fast_out)
            assert_dataclass_tensors_equal(reference, jit_fast_out)

            stats_by_variant = {
                "torch_control": time_callable(
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
                    lambda ws=fast_workspace_triton: _run_triton_fast(fixture, ws),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=args.device,
                ),
                "jit_fast": time_callable(
                    lambda ws=fast_workspace_jit: _run_jit_fast(fixture, ws),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=args.device,
                ),
            }
            print_stats_block(
                "prepare_block_gpu"
                f"[req_pool={_dtype_name(req_pool_dtype)},req_to_token={_dtype_name(req_to_token_dtype)}]",
                stats_by_variant,
            )


if __name__ == "__main__":
    main()
