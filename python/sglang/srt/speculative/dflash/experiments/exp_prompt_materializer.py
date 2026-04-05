from __future__ import annotations

import argparse

import torch

from sglang.srt.speculative.dflash.bench.common import time_callable
from sglang.srt.speculative.dflash.experiments.common import print_stats_block
from sglang.srt.speculative.dflash.experiments.fixtures import (
    PromptMaterializerFixture,
    make_prompt_materializer_fixture,
)
from sglang.srt.speculative.dflash.kernels.materializer import (
    materialize_prompt_grouped_control,
    materialize_prompt_per_layer_control,
)
from sglang.srt.speculative.dflash.reference.materializer import (
    materialize_prompt_reference,
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


def _assert_cache_close(
    expected, actual, *, atol: float = 1e-5, rtol: float = 1e-5
) -> None:
    if not torch.allclose(expected.k_cache, actual.k_cache, atol=atol, rtol=rtol):
        raise AssertionError(
            "k_cache mismatch between reference and control materializer."
        )
    if not torch.allclose(expected.v_cache, actual.v_cache, atol=atol, rtol=rtol):
        raise AssertionError(
            "v_cache mismatch between reference and control materializer."
        )


def _run_reference(fixture: PromptMaterializerFixture):
    return materialize_prompt_reference(
        cache=fixture.cache.clone(),
        config=fixture.config,
        weights=fixture.weights,
        hidden=fixture.hidden,
        positions=fixture.positions,
        slot_ids=fixture.slot_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DFlash prompt materializer variants."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--rotary-dim", type=int, default=16)
    parser.add_argument("--num-slots", type=int, default=1024)
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--group-sizes", default="2,4,all")
    parser.add_argument("--chunk-sizes", default="128,all")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fixture = make_prompt_materializer_fixture(
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
    reference = _run_reference(fixture)
    chunk_sizes = _parse_sizes(args.chunk_sizes, fallback_all=args.num_tokens)
    group_sizes = _parse_sizes(args.group_sizes, fallback_all=args.num_layers)

    stats_by_variant: dict[str, object] = {}
    for chunk_size in chunk_sizes:
        per_layer = materialize_prompt_per_layer_control(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            hidden=fixture.hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
            chunk_size=chunk_size,
        )
        _assert_cache_close(reference, per_layer)
        stats_by_variant[f"per_layer_c{chunk_size}"] = time_callable(
            lambda cs=chunk_size: materialize_prompt_per_layer_control(
                cache=fixture.cache,
                config=fixture.config,
                weights=fixture.weights,
                hidden=fixture.hidden,
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                chunk_size=cs,
                inplace=True,
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        for group_size in group_sizes:
            grouped = materialize_prompt_grouped_control(
                cache=fixture.cache.clone(),
                config=fixture.config,
                weights=fixture.weights,
                hidden=fixture.hidden,
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                group_size=group_size,
                chunk_size=chunk_size,
            )
            _assert_cache_close(reference, grouped)
            stats_by_variant[f"grouped_g{group_size}_c{chunk_size}"] = time_callable(
                lambda gs=group_size, cs=chunk_size: materialize_prompt_grouped_control(
                    cache=fixture.cache,
                    config=fixture.config,
                    weights=fixture.weights,
                    hidden=fixture.hidden,
                    positions=fixture.positions,
                    slot_ids=fixture.slot_ids,
                    group_size=gs,
                    chunk_size=cs,
                    inplace=True,
                ),
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
            )

    print_stats_block("prompt_materializer", stats_by_variant)


if __name__ == "__main__":
    main()
