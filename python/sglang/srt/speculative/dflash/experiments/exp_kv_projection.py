from __future__ import annotations

import argparse

from sglang.srt.speculative.dflash.bench.common import time_callable
from sglang.srt.speculative.dflash.experiments.common import (
    assert_dataclass_tensors_equal,
    print_stats_block,
)
from sglang.srt.speculative.dflash.experiments.fixtures import (
    CommitProjectionFixture,
    PromptProjectionFixture,
    make_commit_projection_fixture,
    make_prompt_projection_fixture,
)
from sglang.srt.speculative.dflash.kernels.kv_projection import (
    project_commit_grouped_control,
    project_commit_per_layer_control,
    project_prompt_grouped_control,
    project_prompt_per_layer_control,
)
from sglang.srt.speculative.dflash.reference.kv_projection import (
    project_commit_reference,
    project_prompt_reference,
)


def _parse_sizes(raw: str, *, fallback_all: int) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        item = token.strip().lower()
        if not item:
            continue
        if item == "all":
            out.append(fallback_all)
        else:
            out.append(int(item))
    return out


def _run_prompt(
    fixture: PromptProjectionFixture,
    *,
    chunk_sizes: list[int],
    group_sizes: list[int],
    warmup: int,
    iters: int,
    device: str,
) -> None:
    reference = project_prompt_reference(
        config=fixture.config,
        weights=fixture.weights,
        hidden=fixture.hidden,
        positions=fixture.positions,
    )
    stats_by_variant = {}
    for chunk_size in chunk_sizes:
        per_layer = project_prompt_per_layer_control(
            config=fixture.config,
            weights=fixture.weights,
            hidden=fixture.hidden,
            positions=fixture.positions,
            chunk_size=chunk_size,
        )
        assert_dataclass_tensors_equal(reference, per_layer, atol=1e-5, rtol=1e-5)
        stats_by_variant[f"per_layer_c{chunk_size}"] = time_callable(
            lambda cs=chunk_size: project_prompt_per_layer_control(
                config=fixture.config,
                weights=fixture.weights,
                hidden=fixture.hidden,
                positions=fixture.positions,
                chunk_size=cs,
            ),
            warmup=warmup,
            iters=iters,
            device=device,
        )
        for group_size in group_sizes:
            grouped = project_prompt_grouped_control(
                config=fixture.config,
                weights=fixture.weights,
                hidden=fixture.hidden,
                positions=fixture.positions,
                group_size=group_size,
                chunk_size=chunk_size,
            )
            assert_dataclass_tensors_equal(reference, grouped, atol=1e-5, rtol=1e-5)
            stats_by_variant[f"grouped_g{group_size}_c{chunk_size}"] = time_callable(
                lambda gs=group_size, cs=chunk_size: project_prompt_grouped_control(
                    config=fixture.config,
                    weights=fixture.weights,
                    hidden=fixture.hidden,
                    positions=fixture.positions,
                    group_size=gs,
                    chunk_size=cs,
                ),
                warmup=warmup,
                iters=iters,
                device=device,
            )
    print_stats_block("kv_projection_prompt", stats_by_variant)


def _run_commit(
    fixture: CommitProjectionFixture,
    *,
    group_sizes: list[int],
    warmup: int,
    iters: int,
    device: str,
) -> None:
    reference = project_commit_reference(
        config=fixture.config,
        weights=fixture.weights,
        verify_hidden=fixture.verify_hidden,
        positions=fixture.positions,
    )
    per_layer = project_commit_per_layer_control(
        config=fixture.config,
        weights=fixture.weights,
        verify_hidden=fixture.verify_hidden,
        positions=fixture.positions,
    )
    assert_dataclass_tensors_equal(reference, per_layer, atol=1e-5, rtol=1e-5)
    stats_by_variant = {
        "per_layer": time_callable(
            lambda: project_commit_per_layer_control(
                config=fixture.config,
                weights=fixture.weights,
                verify_hidden=fixture.verify_hidden,
                positions=fixture.positions,
            ),
            warmup=warmup,
            iters=iters,
            device=device,
        )
    }
    for group_size in group_sizes:
        grouped = project_commit_grouped_control(
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
            group_size=group_size,
        )
        assert_dataclass_tensors_equal(reference, grouped, atol=1e-5, rtol=1e-5)
        stats_by_variant[f"grouped_g{group_size}"] = time_callable(
            lambda gs=group_size: project_commit_grouped_control(
                config=fixture.config,
                weights=fixture.weights,
                verify_hidden=fixture.verify_hidden,
                positions=fixture.positions,
                group_size=gs,
            ),
            warmup=warmup,
            iters=iters,
            device=device,
        )
    print_stats_block("kv_projection_commit", stats_by_variant)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark isolated DFlash KV projection variants."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--rotary-dim", type=int, default=16)
    parser.add_argument("--num-slots", type=int, default=1024)
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--group-sizes", default="2,4,all")
    parser.add_argument("--chunk-sizes", default="128,all")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    chunk_sizes = _parse_sizes(args.chunk_sizes, fallback_all=args.num_tokens)
    group_sizes = _parse_sizes(args.group_sizes, fallback_all=args.num_layers)

    prompt_fixture = make_prompt_projection_fixture(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        rotary_dim=args.rotary_dim,
        num_slots=args.num_slots,
        num_tokens=args.num_tokens,
        device=args.device,
        seed=args.seed,
    )
    _run_prompt(
        prompt_fixture,
        chunk_sizes=chunk_sizes,
        group_sizes=group_sizes,
        warmup=args.warmup,
        iters=args.iters,
        device=args.device,
    )

    commit_fixture = make_commit_projection_fixture(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        rotary_dim=args.rotary_dim,
        num_slots=args.num_slots,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=args.device,
        seed=args.seed,
    )
    _run_commit(
        commit_fixture,
        group_sizes=group_sizes,
        warmup=args.warmup,
        iters=args.iters,
        device=args.device,
    )


if __name__ == "__main__":
    main()
