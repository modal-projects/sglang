from __future__ import annotations

import argparse

from sglang.srt.speculative.dflash.bench.common import time_callable
from sglang.srt.speculative.dflash.experiments.common import (
    assert_dataclass_tensors_equal,
    print_stats_block,
)
from sglang.srt.speculative.dflash.experiments.fixtures import (
    PrepareBlockFixture,
    make_prepare_block_fixture,
)
from sglang.srt.speculative.dflash.kernels.prepare_block import prepare_block_control
from sglang.srt.speculative.dflash.reference.core import prepare_block_reference


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DFlash prepare_block variants."
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

    fixture = make_prepare_block_fixture(
        bucket_bs=args.bucket_bs,
        num_req_slots=args.num_req_slots,
        req_to_token_width=args.req_to_token_width,
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
    print_stats_block("prepare_block", stats_by_variant)


if __name__ == "__main__":
    main()
