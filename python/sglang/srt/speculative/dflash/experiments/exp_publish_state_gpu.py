from __future__ import annotations

import argparse

import torch

from sglang.srt.speculative.dflash.bench.common import time_callable
from sglang.srt.speculative.dflash.experiments.common import (
    assert_dataclass_tensors_equal,
    print_stats_block,
)
from sglang.srt.speculative.dflash.experiments.fixtures import (
    PublishStateFixture,
    make_publish_state_fixture,
)
from sglang.srt.speculative.dflash.kernels.publish_state import publish_state_control
from sglang.srt.speculative.dflash.kernels.publish_state_jit import publish_state_jit
from sglang.srt.speculative.dflash.kernels.publish_state_triton import (
    publish_state_triton,
)
from sglang.srt.speculative.dflash.reference.core import publish_state_reference


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
    fixture: PublishStateFixture, *, req_pool_dtype: torch.dtype
) -> PublishStateFixture:
    return PublishStateFixture(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices.to(dtype=req_pool_dtype),
        req_generation=fixture.req_generation,
        commit_lens=fixture.commit_lens,
        bonus_ids=fixture.bonus_ids,
        gpu_stop_flags=fixture.gpu_stop_flags,
    )


def _run_reference(fixture: PublishStateFixture):
    return publish_state_reference(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        commit_lens=fixture.commit_lens,
        bonus_ids=fixture.bonus_ids,
        gpu_stop_flags=fixture.gpu_stop_flags,
    )


def _run_control(fixture: PublishStateFixture):
    return publish_state_control(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        commit_lens=fixture.commit_lens,
        bonus_ids=fixture.bonus_ids,
        gpu_stop_flags=fixture.gpu_stop_flags,
    )


def _run_triton(fixture: PublishStateFixture):
    return publish_state_triton(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        commit_lens=fixture.commit_lens,
        bonus_ids=fixture.bonus_ids,
        gpu_stop_flags=fixture.gpu_stop_flags,
    )


def _run_jit(fixture: PublishStateFixture):
    return publish_state_jit(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        commit_lens=fixture.commit_lens,
        bonus_ids=fixture.bonus_ids,
        gpu_stop_flags=fixture.gpu_stop_flags,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA DFlash publish_state kernels."
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
    args = parser.parse_args()

    if torch.device(args.device).type != "cuda":
        raise ValueError("This benchmark is CUDA-only.")

    base = make_publish_state_fixture(
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

    for req_pool_dtype in req_pool_dtypes:
        fixture = _cast_fixture(base, req_pool_dtype=req_pool_dtype)
        reference = _run_reference(fixture)
        control = _run_control(fixture)
        triton_out = _run_triton(fixture)
        jit_out = _run_jit(fixture)
        assert_dataclass_tensors_equal(reference, control)
        assert_dataclass_tensors_equal(reference, triton_out)
        assert_dataclass_tensors_equal(reference, jit_out)

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
        }
        print_stats_block(
            f"publish_state_gpu[req_pool={_dtype_name(req_pool_dtype)}]",
            stats_by_variant,
        )


if __name__ == "__main__":
    main()
