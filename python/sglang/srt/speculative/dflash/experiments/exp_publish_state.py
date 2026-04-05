from __future__ import annotations

import argparse

from sglang.srt.speculative.dflash.bench.common import time_callable
from sglang.srt.speculative.dflash.experiments.common import print_stats_block
from sglang.srt.speculative.dflash.experiments.fixtures import (
    PublishStateFixture,
    make_publish_state_fixture,
)
from sglang.srt.speculative.dflash.kernels.publish_state import publish_state_control
from sglang.srt.speculative.dflash.reference.core import publish_state_reference


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


def _assert_state_tables_equal(expected, actual) -> None:
    for attr in (
        "committed_len",
        "reserved_len",
        "next_verified_id",
        "generation",
        "status_flags",
    ):
        lhs = getattr(expected, attr)
        rhs = getattr(actual, attr)
        if not lhs.equal(rhs):
            raise AssertionError(f"State tensor '{attr}' does not match.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DFlash publish_state variants."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bucket-bs", type=int, default=32)
    parser.add_argument("--num-req-slots", type=int, default=64)
    parser.add_argument("--req-to-token-width", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fixture = make_publish_state_fixture(
        bucket_bs=args.bucket_bs,
        num_req_slots=args.num_req_slots,
        req_to_token_width=args.req_to_token_width,
        block_size=args.block_size,
        device=args.device,
        seed=args.seed,
    )
    reference = _run_reference(fixture)
    control = _run_control(fixture)
    _assert_state_tables_equal(reference, control)

    stats_by_variant = {
        "control": time_callable(
            lambda: _run_control(fixture),
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
    }
    print_stats_block("publish_state", stats_by_variant)


if __name__ == "__main__":
    main()
