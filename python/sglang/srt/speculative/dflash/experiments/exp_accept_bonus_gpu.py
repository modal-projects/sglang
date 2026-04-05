from __future__ import annotations

import argparse

import torch

from sglang.srt.speculative.dflash.bench.common import time_callable
from sglang.srt.speculative.dflash.experiments.common import (
    assert_dataclass_tensors_equal,
    print_stats_block,
)
from sglang.srt.speculative.dflash.experiments.fixtures import (
    AcceptBonusFixture,
    make_accept_bonus_fixture,
)
from sglang.srt.speculative.dflash.kernels.accept_bonus import accept_bonus_control
from sglang.srt.speculative.dflash.kernels.accept_bonus_jit import accept_bonus_jit
from sglang.srt.speculative.dflash.kernels.accept_bonus_triton import (
    accept_bonus_triton,
)
from sglang.srt.speculative.dflash.reference.core import accept_bonus_reference


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
    fixture: AcceptBonusFixture, *, token_dtype: torch.dtype
) -> AcceptBonusFixture:
    return AcceptBonusFixture(
        emit_ids=fixture.emit_ids.to(dtype=token_dtype),
        target_top1=fixture.target_top1.to(dtype=token_dtype),
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids.to(dtype=token_dtype),
        stop_token_ids=fixture.stop_token_ids.to(dtype=token_dtype),
    )


def _run_reference(fixture: AcceptBonusFixture):
    return accept_bonus_reference(
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )


def _run_control(fixture: AcceptBonusFixture):
    return accept_bonus_control(
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )


def _run_triton(fixture: AcceptBonusFixture):
    return accept_bonus_triton(
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )


def _run_jit(fixture: AcceptBonusFixture):
    return accept_bonus_jit(
        emit_ids=fixture.emit_ids,
        target_top1=fixture.target_top1,
        active_mask=fixture.active_mask,
        eos_token_ids=fixture.eos_token_ids,
        stop_token_ids=fixture.stop_token_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA DFlash accept_bonus kernels."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bucket-bs", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--token-dtypes", default="int32,int64")
    args = parser.parse_args()

    if torch.device(args.device).type != "cuda":
        raise ValueError("This benchmark is CUDA-only.")

    base = make_accept_bonus_fixture(
        bucket_bs=args.bucket_bs,
        block_size=args.block_size,
        device=args.device,
        seed=args.seed,
    )
    token_dtypes = [
        _resolve_int_dtype(x) for x in args.token_dtypes.split(",") if x.strip()
    ]

    for token_dtype in token_dtypes:
        fixture = _cast_fixture(base, token_dtype=token_dtype)
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
            f"accept_bonus_gpu[token={_dtype_name(token_dtype)}]",
            stats_by_variant,
        )


if __name__ == "__main__":
    main()
