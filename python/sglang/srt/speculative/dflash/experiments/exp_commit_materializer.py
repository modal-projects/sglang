from __future__ import annotations

import argparse

import torch

from sglang.srt.speculative.dflash.bench.common import time_callable
from sglang.srt.speculative.dflash.experiments.common import print_stats_block
from sglang.srt.speculative.dflash.experiments.fixtures import (
    CommitMaterializerFixture,
    make_commit_materializer_fixture,
)
from sglang.srt.speculative.dflash.kernels.materializer import (
    materialize_commit_grouped_control,
    materialize_commit_per_layer_control,
)
from sglang.srt.speculative.dflash.reference.materializer import (
    materialize_commit_reference,
)


def _parse_group_sizes(raw: str, *, num_layers: int) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        item = token.strip().lower()
        if not item:
            continue
        if item == "all":
            out.append(num_layers)
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


def _run_reference(fixture: CommitMaterializerFixture):
    return materialize_commit_reference(
        cache=fixture.cache.clone(),
        config=fixture.config,
        weights=fixture.weights,
        verify_hidden=fixture.verify_hidden,
        positions=fixture.positions,
        slot_ids=fixture.slot_ids,
        commit_lens=fixture.commit_lens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DFlash commit materializer variants."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--rotary-dim", type=int, default=16)
    parser.add_argument("--num-slots", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--group-sizes", default="2,4,all")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fixture = make_commit_materializer_fixture(
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
    reference = _run_reference(fixture)

    per_layer = materialize_commit_per_layer_control(
        cache=fixture.cache.clone(),
        config=fixture.config,
        weights=fixture.weights,
        verify_hidden=fixture.verify_hidden,
        positions=fixture.positions,
        slot_ids=fixture.slot_ids,
        commit_lens=fixture.commit_lens,
    )
    _assert_cache_close(reference, per_layer)

    stats_by_variant = {
        "per_layer": time_callable(
            lambda: materialize_commit_per_layer_control(
                cache=fixture.cache,
                config=fixture.config,
                weights=fixture.weights,
                verify_hidden=fixture.verify_hidden,
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                commit_lens=fixture.commit_lens,
                inplace=True,
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
    }

    for group_size in _parse_group_sizes(args.group_sizes, num_layers=args.num_layers):
        grouped = materialize_commit_grouped_control(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
            commit_lens=fixture.commit_lens,
            group_size=group_size,
        )
        _assert_cache_close(reference, grouped)
        stats_by_variant[f"grouped_g{group_size}"] = time_callable(
            lambda gs=group_size: materialize_commit_grouped_control(
                cache=fixture.cache,
                config=fixture.config,
                weights=fixture.weights,
                verify_hidden=fixture.verify_hidden,
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                commit_lens=fixture.commit_lens,
                group_size=gs,
                inplace=True,
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )

    print_stats_block("commit_materializer", stats_by_variant)


if __name__ == "__main__":
    main()
