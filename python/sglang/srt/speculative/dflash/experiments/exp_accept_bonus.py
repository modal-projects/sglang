from __future__ import annotations

import argparse

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
from sglang.srt.speculative.dflash.reference.core import accept_bonus_reference


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DFlash accept_bonus variants."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bucket-bs", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fixture = make_accept_bonus_fixture(
        bucket_bs=args.bucket_bs,
        block_size=args.block_size,
        device=args.device,
        seed=args.seed,
    )
    reference = _run_reference(fixture)
    control = _run_control(fixture)
    assert_dataclass_tensors_equal(reference, control)

    stats_by_variant = {
        "control": time_callable(
            lambda: _run_control(fixture),
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
    }
    print_stats_block("accept_bonus", stats_by_variant)


if __name__ == "__main__":
    main()
