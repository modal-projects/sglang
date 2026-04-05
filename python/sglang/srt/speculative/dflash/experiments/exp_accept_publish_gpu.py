from __future__ import annotations

import argparse

import torch

from sglang.srt.speculative.dflash.bench.common import time_callable
from sglang.srt.speculative.dflash.contracts import DFlashAcceptPublishResult
from sglang.srt.speculative.dflash.experiments.common import (
    assert_dataclass_tensors_equal,
    print_stats_block,
)
from sglang.srt.speculative.dflash.experiments.fixtures import (
    AcceptPublishFixture,
    make_accept_publish_fixture,
)
from sglang.srt.speculative.dflash.kernels.accept_bonus_jit import accept_bonus_jit
from sglang.srt.speculative.dflash.kernels.accept_bonus_triton import (
    accept_bonus_triton,
)
from sglang.srt.speculative.dflash.kernels.accept_publish import (
    accept_publish_control,
)
from sglang.srt.speculative.dflash.kernels.accept_publish_jit import (
    accept_publish_jit,
)
from sglang.srt.speculative.dflash.kernels.accept_publish_triton import (
    accept_publish_triton,
)
from sglang.srt.speculative.dflash.kernels.publish_state_jit import publish_state_jit
from sglang.srt.speculative.dflash.kernels.publish_state_triton import (
    publish_state_triton,
)
from sglang.srt.speculative.dflash.reference.core import accept_publish_reference


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
    fixture: AcceptPublishFixture,
    *,
    token_dtype: torch.dtype,
    req_pool_dtype: torch.dtype,
) -> AcceptPublishFixture:
    return AcceptPublishFixture(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices.to(dtype=req_pool_dtype),
        req_generation=fixture.req_generation,
        emit_ids=fixture.emit_ids.to(dtype=token_dtype),
        target_top1=fixture.target_top1.to(dtype=token_dtype),
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids.to(dtype=token_dtype),
        stop_token_ids=fixture.stop_token_ids.to(dtype=token_dtype),
    )


def _run_reference(fixture: AcceptPublishFixture):
    return accept_publish_reference(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )


def _run_control(fixture: AcceptPublishFixture):
    return accept_publish_control(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )


def _run_separate_triton(fixture: AcceptPublishFixture):
    accept = accept_bonus_triton(
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )
    state = publish_state_triton(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        commit_lens=accept.commit_lens,
        bonus_ids=accept.bonus_ids,
        gpu_stop_flags=accept.gpu_stop_flags,
    )
    return DFlashAcceptPublishResult(accept=accept, state=state)


def _run_fused_triton(fixture: AcceptPublishFixture):
    return accept_publish_triton(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )


def _run_separate_jit(fixture: AcceptPublishFixture):
    accept = accept_bonus_jit(
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )
    state = publish_state_jit(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        commit_lens=accept.commit_lens,
        bonus_ids=accept.bonus_ids,
        gpu_stop_flags=accept.gpu_stop_flags,
    )
    return DFlashAcceptPublishResult(accept=accept, state=state)


def _run_fused_jit(fixture: AcceptPublishFixture):
    return accept_publish_jit(
        state=fixture.state,
        req_pool_indices=fixture.req_pool_indices,
        req_generation=fixture.req_generation,
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA DFlash fused accept_publish kernels."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bucket-bs", type=int, default=32)
    parser.add_argument("--num-req-slots", type=int, default=64)
    parser.add_argument("--req-to-token-width", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--token-dtypes", default="int32,int64")
    parser.add_argument("--req-pool-dtypes", default="int32,int64")
    args = parser.parse_args()

    if torch.device(args.device).type != "cuda":
        raise ValueError("This benchmark is CUDA-only.")

    base = make_accept_publish_fixture(
        bucket_bs=args.bucket_bs,
        num_req_slots=args.num_req_slots,
        req_to_token_width=args.req_to_token_width,
        block_size=args.block_size,
        device=args.device,
        seed=args.seed,
    )
    token_dtypes = [
        _resolve_int_dtype(x) for x in args.token_dtypes.split(",") if x.strip()
    ]
    req_pool_dtypes = [
        _resolve_int_dtype(x) for x in args.req_pool_dtypes.split(",") if x.strip()
    ]

    for token_dtype in token_dtypes:
        for req_pool_dtype in req_pool_dtypes:
            fixture = _cast_fixture(
                base,
                token_dtype=token_dtype,
                req_pool_dtype=req_pool_dtype,
            )
            reference = _run_reference(fixture)
            control = _run_control(fixture)
            separate_triton = _run_separate_triton(fixture)
            fused_triton = _run_fused_triton(fixture)
            separate_jit = _run_separate_jit(fixture)
            fused_jit = _run_fused_jit(fixture)

            assert_dataclass_tensors_equal(reference, control)
            assert_dataclass_tensors_equal(reference, separate_triton)
            assert_dataclass_tensors_equal(reference, fused_triton)
            assert_dataclass_tensors_equal(reference, separate_jit)
            assert_dataclass_tensors_equal(reference, fused_jit)

            stats_by_variant = {
                "torch_control": time_callable(
                    lambda: _run_control(fixture),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=args.device,
                ),
                "separate_triton": time_callable(
                    lambda: _run_separate_triton(fixture),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=args.device,
                ),
                "fused_triton": time_callable(
                    lambda: _run_fused_triton(fixture),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=args.device,
                ),
                "separate_jit": time_callable(
                    lambda: _run_separate_jit(fixture),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=args.device,
                ),
                "fused_jit": time_callable(
                    lambda: _run_fused_jit(fixture),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=args.device,
                ),
            }
            print_stats_block(
                f"accept_publish_gpu[token={_dtype_name(token_dtype)},req_pool={_dtype_name(req_pool_dtype)}]",
                stats_by_variant,
            )


if __name__ == "__main__":
    main()
