from __future__ import annotations

import argparse

import torch

from sglang.srt.speculative.dflash.bench.common import time_callable
from sglang.srt.speculative.dflash.experiments.common import print_stats_block
from sglang.srt.speculative.dflash.experiments.fixtures import (
    make_commit_materializer_fixture,
    make_prompt_materializer_fixture,
)
from sglang.srt.speculative.dflash.kernels.materializer import (
    materialize_commit_grouped_control,
    materialize_prompt_grouped_control,
)
from sglang.srt.speculative.dflash.kernels.materializer_packed import (
    create_packed_materializer_workspace,
    materialize_commit_packed_compact_jit,
    materialize_commit_packed_compact_triton,
    materialize_commit_packed_jit,
    materialize_commit_packed_jit_fast,
    materialize_commit_packed_jit_workspace_fast,
    materialize_commit_packed_triton,
    materialize_prompt_packed_jit,
    materialize_prompt_packed_jit_fast,
    materialize_prompt_packed_jit_workspace_fast,
    materialize_prompt_packed_triton,
)
from sglang.srt.speculative.dflash.kernels.post_projection_jit import (
    postprocess_commit_jit,
    postprocess_prompt_jit,
)
from sglang.srt.speculative.dflash.kernels.post_projection_triton import (
    postprocess_commit_triton,
    postprocess_prompt_triton,
)
from sglang.srt.speculative.dflash.kernels.raw_kv_projection import (
    project_raw_commit_control,
    project_raw_prompt_control,
)
from sglang.srt.speculative.dflash.reference.materializer import (
    materialize_commit_reference,
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


def _assert_cache_close(expected, actual, *, atol: float, rtol: float) -> None:
    if not torch.allclose(expected.k_cache, actual.k_cache, atol=atol, rtol=rtol):
        raise AssertionError("k_cache mismatch in packed materializer benchmark.")
    if not torch.allclose(expected.v_cache, actual.v_cache, atol=atol, rtol=rtol):
        raise AssertionError("v_cache mismatch in packed materializer benchmark.")


def _run_prompt(
    args: argparse.Namespace, *, group_sizes: list[int], chunk_sizes: list[int]
) -> None:
    fixture = make_prompt_materializer_fixture(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        rotary_dim=args.rotary_dim,
        num_slots=args.num_slots,
        num_tokens=args.num_tokens,
        device=args.device,
        dtype=torch.bfloat16,
        seed=args.seed,
    )
    fixture = type(fixture)(
        cache=fixture.cache,
        config=fixture.config,
        weights=fixture.weights,
        hidden=fixture.hidden,
        positions=fixture.positions.to(torch.int32),
        slot_ids=fixture.slot_ids.to(torch.int32),
        cos_sin_cache=fixture.cos_sin_cache,
    )
    reference = materialize_prompt_reference(
        cache=fixture.cache.clone(),
        config=fixture.config,
        weights=fixture.weights,
        hidden=fixture.hidden,
        positions=fixture.positions,
        slot_ids=fixture.slot_ids,
    )
    stats_by_variant: dict[str, object] = {}
    for chunk_size in chunk_sizes:
        for group_size in group_sizes:
            label = f"g{group_size}"
            prompt_workspace = create_packed_materializer_workspace(
                config=fixture.config,
                weights=fixture.weights,
                group_size=group_size,
                max_rows=min(int(chunk_size), int(fixture.hidden.shape[0])),
                dtype=fixture.hidden.dtype,
                device=fixture.hidden.device,
                cos_sin_cache=fixture.cos_sin_cache,
            )
            full = materialize_prompt_grouped_control(
                cache=fixture.cache.clone(),
                config=fixture.config,
                weights=fixture.weights,
                hidden=fixture.hidden,
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                group_size=group_size,
                chunk_size=chunk_size,
            )
            split_triton = postprocess_prompt_triton(
                cache=fixture.cache.clone(),
                config=fixture.config,
                weights=fixture.weights,
                projected=project_raw_prompt_control(
                    config=fixture.config,
                    weights=fixture.weights,
                    hidden=fixture.hidden,
                    positions=fixture.positions,
                    group_size=group_size,
                    chunk_size=chunk_size,
                ),
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                cos_sin_cache=fixture.cos_sin_cache,
            )
            split_jit = postprocess_prompt_jit(
                cache=fixture.cache.clone(),
                config=fixture.config,
                weights=fixture.weights,
                projected=project_raw_prompt_control(
                    config=fixture.config,
                    weights=fixture.weights,
                    hidden=fixture.hidden,
                    positions=fixture.positions,
                    group_size=group_size,
                    chunk_size=chunk_size,
                ),
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                cos_sin_cache=fixture.cos_sin_cache,
            )
            packed_triton = materialize_prompt_packed_triton(
                cache=fixture.cache.clone(),
                config=fixture.config,
                weights=fixture.weights,
                hidden=fixture.hidden,
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                group_size=group_size,
                chunk_size=chunk_size,
                cos_sin_cache=fixture.cos_sin_cache,
            )
            packed_jit = materialize_prompt_packed_jit(
                cache=fixture.cache.clone(),
                config=fixture.config,
                weights=fixture.weights,
                hidden=fixture.hidden,
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                group_size=group_size,
                chunk_size=chunk_size,
                cos_sin_cache=fixture.cos_sin_cache,
            )
            packed_jit_fast = materialize_prompt_packed_jit_fast(
                cache=fixture.cache.clone(),
                config=fixture.config,
                weights=fixture.weights,
                hidden=fixture.hidden,
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                group_size=group_size,
                chunk_size=chunk_size,
                cos_sin_cache=fixture.cos_sin_cache,
            )
            packed_jit_workspace = materialize_prompt_packed_jit_workspace_fast(
                cache=fixture.cache.clone(),
                config=fixture.config,
                weights=fixture.weights,
                hidden=fixture.hidden,
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                group_size=group_size,
                chunk_size=chunk_size,
                workspace=prompt_workspace,
            )
            for actual in (
                full,
                split_triton,
                split_jit,
                packed_triton,
                packed_jit,
                packed_jit_fast,
                packed_jit_workspace,
            ):
                _assert_cache_close(reference, actual, atol=1e-1, rtol=1e-1)

            stats_by_variant[f"full_{label}_c{chunk_size}"] = time_callable(
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
            stats_by_variant[f"split_triton_{label}_c{chunk_size}"] = time_callable(
                lambda gs=group_size, cs=chunk_size: postprocess_prompt_triton(
                    cache=fixture.cache,
                    config=fixture.config,
                    weights=fixture.weights,
                    projected=project_raw_prompt_control(
                        config=fixture.config,
                        weights=fixture.weights,
                        hidden=fixture.hidden,
                        positions=fixture.positions,
                        group_size=gs,
                        chunk_size=cs,
                    ),
                    positions=fixture.positions,
                    slot_ids=fixture.slot_ids,
                    cos_sin_cache=fixture.cos_sin_cache,
                    inplace=True,
                ),
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
            )
            stats_by_variant[f"split_jit_{label}_c{chunk_size}"] = time_callable(
                lambda gs=group_size, cs=chunk_size: postprocess_prompt_jit(
                    cache=fixture.cache,
                    config=fixture.config,
                    weights=fixture.weights,
                    projected=project_raw_prompt_control(
                        config=fixture.config,
                        weights=fixture.weights,
                        hidden=fixture.hidden,
                        positions=fixture.positions,
                        group_size=gs,
                        chunk_size=cs,
                    ),
                    positions=fixture.positions,
                    slot_ids=fixture.slot_ids,
                    cos_sin_cache=fixture.cos_sin_cache,
                    inplace=True,
                ),
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
            )
            stats_by_variant[f"packed_triton_{label}_c{chunk_size}"] = time_callable(
                lambda gs=group_size, cs=chunk_size: materialize_prompt_packed_triton(
                    cache=fixture.cache,
                    config=fixture.config,
                    weights=fixture.weights,
                    hidden=fixture.hidden,
                    positions=fixture.positions,
                    slot_ids=fixture.slot_ids,
                    group_size=gs,
                    chunk_size=cs,
                    cos_sin_cache=fixture.cos_sin_cache,
                    inplace=True,
                ),
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
            )
            stats_by_variant[f"packed_jit_{label}_c{chunk_size}"] = time_callable(
                lambda gs=group_size, cs=chunk_size: materialize_prompt_packed_jit(
                    cache=fixture.cache,
                    config=fixture.config,
                    weights=fixture.weights,
                    hidden=fixture.hidden,
                    positions=fixture.positions,
                    slot_ids=fixture.slot_ids,
                    group_size=gs,
                    chunk_size=cs,
                    cos_sin_cache=fixture.cos_sin_cache,
                    inplace=True,
                ),
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
            )
            stats_by_variant[f"packed_jit_fast_{label}_c{chunk_size}"] = time_callable(
                lambda gs=group_size, cs=chunk_size: materialize_prompt_packed_jit_fast(
                    cache=fixture.cache,
                    config=fixture.config,
                    weights=fixture.weights,
                    hidden=fixture.hidden,
                    positions=fixture.positions,
                    slot_ids=fixture.slot_ids,
                    group_size=gs,
                    chunk_size=cs,
                    cos_sin_cache=fixture.cos_sin_cache,
                    inplace=True,
                ),
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
            )
            stats_by_variant[f"packed_jit_ws_{label}_c{chunk_size}"] = time_callable(
                lambda gs=group_size, cs=chunk_size, ws=prompt_workspace: materialize_prompt_packed_jit_workspace_fast(
                    cache=fixture.cache,
                    config=fixture.config,
                    weights=fixture.weights,
                    hidden=fixture.hidden,
                    positions=fixture.positions,
                    slot_ids=fixture.slot_ids,
                    group_size=gs,
                    chunk_size=cs,
                    workspace=ws,
                    inplace=True,
                ),
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
            )

    print_stats_block("materializer_packed_prompt", stats_by_variant)


def _run_commit(args: argparse.Namespace, *, group_sizes: list[int]) -> None:
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
        dtype=torch.bfloat16,
        seed=args.seed,
    )
    fixture = type(fixture)(
        cache=fixture.cache,
        config=fixture.config,
        weights=fixture.weights,
        verify_hidden=fixture.verify_hidden,
        positions=fixture.positions.to(torch.int32),
        slot_ids=fixture.slot_ids.to(torch.int32),
        commit_lens=fixture.commit_lens.to(torch.int32),
        cos_sin_cache=fixture.cos_sin_cache,
    )
    reference = materialize_commit_reference(
        cache=fixture.cache.clone(),
        config=fixture.config,
        weights=fixture.weights,
        verify_hidden=fixture.verify_hidden,
        positions=fixture.positions,
        slot_ids=fixture.slot_ids,
        commit_lens=fixture.commit_lens,
    )
    compact_hidden_scratch = torch.empty(
        (args.batch_size * args.block_size, args.hidden_size),
        dtype=fixture.verify_hidden.dtype,
        device=fixture.verify_hidden.device,
    )
    compact_positions_scratch = torch.empty(
        (args.batch_size * args.block_size,),
        dtype=fixture.positions.dtype,
        device=fixture.positions.device,
    )
    compact_slots_scratch = torch.empty(
        (args.batch_size * args.block_size,),
        dtype=fixture.slot_ids.dtype,
        device=fixture.slot_ids.device,
    )
    stats_by_variant: dict[str, object] = {}
    for group_size in group_sizes:
        label = f"g{group_size}"
        commit_workspace = create_packed_materializer_workspace(
            config=fixture.config,
            weights=fixture.weights,
            group_size=group_size,
            max_rows=int(
                fixture.verify_hidden.shape[0] * fixture.verify_hidden.shape[1]
            ),
            dtype=fixture.verify_hidden.dtype,
            device=fixture.verify_hidden.device,
            cos_sin_cache=fixture.cos_sin_cache,
        )
        full = materialize_commit_grouped_control(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
            commit_lens=fixture.commit_lens,
            group_size=group_size,
        )
        split_triton = postprocess_commit_triton(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            projected=project_raw_commit_control(
                config=fixture.config,
                weights=fixture.weights,
                verify_hidden=fixture.verify_hidden,
                positions=fixture.positions,
                group_size=group_size,
            ),
            positions=fixture.positions,
            slot_ids_2d=fixture.slot_ids,
            commit_lens=fixture.commit_lens,
            cos_sin_cache=fixture.cos_sin_cache,
        )
        split_jit = postprocess_commit_jit(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            projected=project_raw_commit_control(
                config=fixture.config,
                weights=fixture.weights,
                verify_hidden=fixture.verify_hidden,
                positions=fixture.positions,
                group_size=group_size,
            ),
            positions=fixture.positions,
            slot_ids_2d=fixture.slot_ids,
            commit_lens=fixture.commit_lens,
            cos_sin_cache=fixture.cos_sin_cache,
        )
        packed_triton = materialize_commit_packed_triton(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
            commit_lens=fixture.commit_lens,
            group_size=group_size,
            cos_sin_cache=fixture.cos_sin_cache,
        )
        packed_jit = materialize_commit_packed_jit(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
            commit_lens=fixture.commit_lens,
            group_size=group_size,
            cos_sin_cache=fixture.cos_sin_cache,
        )
        packed_jit_fast = materialize_commit_packed_jit_fast(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
            commit_lens=fixture.commit_lens,
            group_size=group_size,
            cos_sin_cache=fixture.cos_sin_cache,
        )
        packed_jit_workspace = materialize_commit_packed_jit_workspace_fast(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
            commit_lens=fixture.commit_lens,
            group_size=group_size,
            workspace=commit_workspace,
        )
        compact_triton = materialize_commit_packed_compact_triton(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
            commit_lens=fixture.commit_lens,
            group_size=group_size,
            cos_sin_cache=fixture.cos_sin_cache,
            hidden_scratch=compact_hidden_scratch,
            positions_scratch=compact_positions_scratch,
            slot_ids_scratch=compact_slots_scratch,
        )
        compact_jit = materialize_commit_packed_compact_jit(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
            commit_lens=fixture.commit_lens,
            group_size=group_size,
            cos_sin_cache=fixture.cos_sin_cache,
            hidden_scratch=compact_hidden_scratch,
            positions_scratch=compact_positions_scratch,
            slot_ids_scratch=compact_slots_scratch,
        )
        for actual in (
            full,
            split_triton,
            split_jit,
            packed_triton,
            packed_jit,
            packed_jit_fast,
            packed_jit_workspace,
            compact_triton,
            compact_jit,
        ):
            _assert_cache_close(reference, actual, atol=1e-1, rtol=1e-1)

        stats_by_variant[f"full_{label}"] = time_callable(
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
        stats_by_variant[f"split_triton_{label}"] = time_callable(
            lambda gs=group_size: postprocess_commit_triton(
                cache=fixture.cache,
                config=fixture.config,
                weights=fixture.weights,
                projected=project_raw_commit_control(
                    config=fixture.config,
                    weights=fixture.weights,
                    verify_hidden=fixture.verify_hidden,
                    positions=fixture.positions,
                    group_size=gs,
                ),
                positions=fixture.positions,
                slot_ids_2d=fixture.slot_ids,
                commit_lens=fixture.commit_lens,
                cos_sin_cache=fixture.cos_sin_cache,
                inplace=True,
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        stats_by_variant[f"split_jit_{label}"] = time_callable(
            lambda gs=group_size: postprocess_commit_jit(
                cache=fixture.cache,
                config=fixture.config,
                weights=fixture.weights,
                projected=project_raw_commit_control(
                    config=fixture.config,
                    weights=fixture.weights,
                    verify_hidden=fixture.verify_hidden,
                    positions=fixture.positions,
                    group_size=gs,
                ),
                positions=fixture.positions,
                slot_ids_2d=fixture.slot_ids,
                commit_lens=fixture.commit_lens,
                cos_sin_cache=fixture.cos_sin_cache,
                inplace=True,
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        stats_by_variant[f"packed_triton_{label}"] = time_callable(
            lambda gs=group_size: materialize_commit_packed_triton(
                cache=fixture.cache,
                config=fixture.config,
                weights=fixture.weights,
                verify_hidden=fixture.verify_hidden,
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                commit_lens=fixture.commit_lens,
                group_size=gs,
                cos_sin_cache=fixture.cos_sin_cache,
                inplace=True,
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        stats_by_variant[f"packed_jit_{label}"] = time_callable(
            lambda gs=group_size: materialize_commit_packed_jit(
                cache=fixture.cache,
                config=fixture.config,
                weights=fixture.weights,
                verify_hidden=fixture.verify_hidden,
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                commit_lens=fixture.commit_lens,
                group_size=gs,
                cos_sin_cache=fixture.cos_sin_cache,
                inplace=True,
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        stats_by_variant[f"packed_jit_fast_{label}"] = time_callable(
            lambda gs=group_size: materialize_commit_packed_jit_fast(
                cache=fixture.cache,
                config=fixture.config,
                weights=fixture.weights,
                verify_hidden=fixture.verify_hidden,
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                commit_lens=fixture.commit_lens,
                group_size=gs,
                cos_sin_cache=fixture.cos_sin_cache,
                inplace=True,
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        stats_by_variant[f"packed_jit_ws_{label}"] = time_callable(
            lambda gs=group_size, ws=commit_workspace: materialize_commit_packed_jit_workspace_fast(
                cache=fixture.cache,
                config=fixture.config,
                weights=fixture.weights,
                verify_hidden=fixture.verify_hidden,
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                commit_lens=fixture.commit_lens,
                group_size=gs,
                workspace=ws,
                inplace=True,
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        stats_by_variant[f"compact_triton_{label}"] = time_callable(
            lambda gs=group_size: materialize_commit_packed_compact_triton(
                cache=fixture.cache,
                config=fixture.config,
                weights=fixture.weights,
                verify_hidden=fixture.verify_hidden,
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                commit_lens=fixture.commit_lens,
                group_size=gs,
                cos_sin_cache=fixture.cos_sin_cache,
                hidden_scratch=compact_hidden_scratch,
                positions_scratch=compact_positions_scratch,
                slot_ids_scratch=compact_slots_scratch,
                inplace=True,
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        stats_by_variant[f"compact_jit_{label}"] = time_callable(
            lambda gs=group_size: materialize_commit_packed_compact_jit(
                cache=fixture.cache,
                config=fixture.config,
                weights=fixture.weights,
                verify_hidden=fixture.verify_hidden,
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                commit_lens=fixture.commit_lens,
                group_size=gs,
                cos_sin_cache=fixture.cos_sin_cache,
                hidden_scratch=compact_hidden_scratch,
                positions_scratch=compact_positions_scratch,
                slot_ids_scratch=compact_slots_scratch,
                inplace=True,
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )

    print_stats_block("materializer_packed_commit", stats_by_variant)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark packed grouped DFlash materializer kernels."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rotary-dim", type=int, default=128)
    parser.add_argument("--num-slots", type=int, default=4096)
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--group-sizes", default="2,4,all")
    parser.add_argument("--chunk-sizes", default="128,all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    group_sizes = _parse_sizes(args.group_sizes, fallback_all=args.num_layers)
    chunk_sizes = _parse_sizes(args.chunk_sizes, fallback_all=args.num_tokens)
    _run_prompt(args, group_sizes=group_sizes, chunk_sizes=chunk_sizes)
    _run_commit(args, group_sizes=group_sizes)


if __name__ == "__main__":
    main()
