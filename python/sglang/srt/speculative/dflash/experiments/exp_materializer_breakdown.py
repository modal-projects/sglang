from __future__ import annotations

import argparse
from collections.abc import Callable

from sglang.srt.speculative.dflash.bench.common import time_callable
from sglang.srt.speculative.dflash.experiments.common import (
    assert_dataclass_tensors_equal,
    print_stats_block,
)
from sglang.srt.speculative.dflash.experiments.fixtures import (
    CommitMaterializerFixture,
    PromptMaterializerFixture,
    make_commit_materializer_fixture,
    make_prompt_materializer_fixture,
)
from sglang.srt.speculative.dflash.kernels.kv_prefix_write import (
    write_commit_prefix_flatten_control,
    write_prompt_index_copy_control,
)
from sglang.srt.speculative.dflash.kernels.kv_prefix_write_jit import (
    write_commit_prefix_jit,
    write_prompt_jit,
)
from sglang.srt.speculative.dflash.kernels.kv_prefix_write_triton import (
    write_commit_prefix_triton,
    write_prompt_triton,
)
from sglang.srt.speculative.dflash.kernels.kv_projection import (
    project_commit_grouped_control,
    project_commit_per_layer_control,
    project_prompt_grouped_control,
    project_prompt_per_layer_control,
)
from sglang.srt.speculative.dflash.kernels.materializer import (
    materialize_commit_grouped_control,
    materialize_commit_per_layer_control,
    materialize_prompt_grouped_control,
    materialize_prompt_per_layer_control,
)
from sglang.srt.speculative.dflash.reference.kv_projection import (
    project_commit_reference,
    project_prompt_reference,
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


def _prompt_writer_variants(device: str) -> list[tuple[str, Callable]]:
    writers: list[tuple[str, Callable]] = [("torch", write_prompt_index_copy_control)]
    if str(device).startswith("cuda"):
        writers.extend([("triton", write_prompt_triton), ("jit", write_prompt_jit)])
    return writers


def _commit_writer_variants(device: str) -> list[tuple[str, Callable]]:
    writers: list[tuple[str, Callable]] = [
        ("torch", write_commit_prefix_flatten_control)
    ]
    if str(device).startswith("cuda"):
        writers.extend(
            [("triton", write_commit_prefix_triton), ("jit", write_commit_prefix_jit)]
        )
    return writers


def _project_prompt(
    fixture: PromptMaterializerFixture,
    *,
    chunk_size: int,
    group_size: int | None,
):
    if group_size is None:
        return project_prompt_per_layer_control(
            config=fixture.config,
            weights=fixture.weights,
            hidden=fixture.hidden,
            positions=fixture.positions,
            chunk_size=chunk_size,
        )
    return project_prompt_grouped_control(
        config=fixture.config,
        weights=fixture.weights,
        hidden=fixture.hidden,
        positions=fixture.positions,
        group_size=group_size,
        chunk_size=chunk_size,
    )


def _materialize_prompt(
    fixture: PromptMaterializerFixture,
    *,
    chunk_size: int,
    group_size: int | None,
    inplace: bool,
):
    if group_size is None:
        return materialize_prompt_per_layer_control(
            cache=fixture.cache if inplace else fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            hidden=fixture.hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
            chunk_size=chunk_size,
            inplace=inplace,
        )
    return materialize_prompt_grouped_control(
        cache=fixture.cache if inplace else fixture.cache.clone(),
        config=fixture.config,
        weights=fixture.weights,
        hidden=fixture.hidden,
        positions=fixture.positions,
        slot_ids=fixture.slot_ids,
        group_size=group_size,
        chunk_size=chunk_size,
        inplace=inplace,
    )


def _write_prompt(
    writer: Callable,
    fixture: PromptMaterializerFixture,
    projection,
    *,
    inplace: bool,
):
    return writer(
        cache=fixture.cache if inplace else fixture.cache.clone(),
        config=fixture.config,
        slot_ids=fixture.slot_ids,
        cache_k=projection.cache_k,
        cache_v=projection.cache_v,
        inplace=inplace,
    )


def _project_commit(
    fixture: CommitMaterializerFixture,
    *,
    group_size: int | None,
):
    if group_size is None:
        return project_commit_per_layer_control(
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
        )
    return project_commit_grouped_control(
        config=fixture.config,
        weights=fixture.weights,
        verify_hidden=fixture.verify_hidden,
        positions=fixture.positions,
        group_size=group_size,
    )


def _materialize_commit(
    fixture: CommitMaterializerFixture,
    *,
    group_size: int | None,
    inplace: bool,
):
    if group_size is None:
        return materialize_commit_per_layer_control(
            cache=fixture.cache if inplace else fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
            commit_lens=fixture.commit_lens,
            inplace=inplace,
        )
    return materialize_commit_grouped_control(
        cache=fixture.cache if inplace else fixture.cache.clone(),
        config=fixture.config,
        weights=fixture.weights,
        verify_hidden=fixture.verify_hidden,
        positions=fixture.positions,
        slot_ids=fixture.slot_ids,
        commit_lens=fixture.commit_lens,
        group_size=group_size,
        inplace=inplace,
    )


def _write_commit(
    writer: Callable,
    fixture: CommitMaterializerFixture,
    projection,
    *,
    inplace: bool,
):
    return writer(
        cache=fixture.cache if inplace else fixture.cache.clone(),
        config=fixture.config,
        slot_ids_2d=fixture.slot_ids,
        commit_lens=fixture.commit_lens,
        cache_k=projection.cache_k,
        cache_v=projection.cache_v,
        inplace=inplace,
    )


def _run_prompt(
    args: argparse.Namespace,
    *,
    chunk_sizes: list[int],
    group_sizes: list[int],
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
        seed=args.seed,
    )
    full_reference = materialize_prompt_reference(
        cache=fixture.cache.clone(),
        config=fixture.config,
        weights=fixture.weights,
        hidden=fixture.hidden,
        positions=fixture.positions,
        slot_ids=fixture.slot_ids,
    )
    projection_reference = project_prompt_reference(
        config=fixture.config,
        weights=fixture.weights,
        hidden=fixture.hidden,
        positions=fixture.positions,
    )

    stats_by_variant = {}
    writer_variants = _prompt_writer_variants(args.device)
    for chunk_size in chunk_sizes:
        for label, group_size in [
            ("per_layer", None),
            *[(f"g{g}", g) for g in group_sizes],
        ]:
            projection = _project_prompt(
                fixture,
                chunk_size=chunk_size,
                group_size=group_size,
            )
            full = _materialize_prompt(
                fixture,
                chunk_size=chunk_size,
                group_size=group_size,
                inplace=False,
            )
            assert_dataclass_tensors_equal(
                projection_reference, projection, atol=1e-5, rtol=1e-5
            )
            assert_dataclass_tensors_equal(full_reference, full, atol=1e-5, rtol=1e-5)

            stats_by_variant[f"project_{label}_c{chunk_size}"] = time_callable(
                lambda cs=chunk_size, gs=group_size: _project_prompt(
                    fixture,
                    chunk_size=cs,
                    group_size=gs,
                ),
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
            )
            stats_by_variant[f"full_{label}_c{chunk_size}"] = time_callable(
                lambda cs=chunk_size, gs=group_size: _materialize_prompt(
                    fixture,
                    chunk_size=cs,
                    group_size=gs,
                    inplace=True,
                ),
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
            )

            for writer_name, writer in writer_variants:
                write = _write_prompt(writer, fixture, projection, inplace=False)
                assert_dataclass_tensors_equal(
                    full_reference, write, atol=1e-5, rtol=1e-5
                )
                stats_by_variant[f"write_{writer_name}_{label}_c{chunk_size}"] = (
                    time_callable(
                        lambda w=writer, proj=projection: _write_prompt(
                            w,
                            fixture,
                            proj,
                            inplace=True,
                        ),
                        warmup=args.warmup,
                        iters=args.iters,
                        device=args.device,
                    )
                )
                stats_by_variant[f"split_{writer_name}_{label}_c{chunk_size}"] = (
                    time_callable(
                        lambda cs=chunk_size, gs=group_size, w=writer: _write_prompt(
                            w,
                            fixture,
                            _project_prompt(
                                fixture,
                                chunk_size=cs,
                                group_size=gs,
                            ),
                            inplace=True,
                        ),
                        warmup=args.warmup,
                        iters=args.iters,
                        device=args.device,
                    )
                )

    print_stats_block("materializer_breakdown_prompt", stats_by_variant)


def _run_commit(
    args: argparse.Namespace,
    *,
    group_sizes: list[int],
) -> None:
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
    full_reference = materialize_commit_reference(
        cache=fixture.cache.clone(),
        config=fixture.config,
        weights=fixture.weights,
        verify_hidden=fixture.verify_hidden,
        positions=fixture.positions,
        slot_ids=fixture.slot_ids,
        commit_lens=fixture.commit_lens,
    )
    projection_reference = project_commit_reference(
        config=fixture.config,
        weights=fixture.weights,
        verify_hidden=fixture.verify_hidden,
        positions=fixture.positions,
    )

    stats_by_variant = {}
    writer_variants = _commit_writer_variants(args.device)
    for label, group_size in [
        ("per_layer", None),
        *[(f"g{g}", g) for g in group_sizes],
    ]:
        projection = _project_commit(fixture, group_size=group_size)
        full = _materialize_commit(
            fixture,
            group_size=group_size,
            inplace=False,
        )
        assert_dataclass_tensors_equal(
            projection_reference, projection, atol=1e-5, rtol=1e-5
        )
        assert_dataclass_tensors_equal(full_reference, full, atol=1e-5, rtol=1e-5)

        stats_by_variant[f"project_{label}"] = time_callable(
            lambda gs=group_size: _project_commit(fixture, group_size=gs),
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        stats_by_variant[f"full_{label}"] = time_callable(
            lambda gs=group_size: _materialize_commit(
                fixture,
                group_size=gs,
                inplace=True,
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )

        for writer_name, writer in writer_variants:
            write = _write_commit(writer, fixture, projection, inplace=False)
            assert_dataclass_tensors_equal(full_reference, write, atol=1e-5, rtol=1e-5)
            stats_by_variant[f"write_{writer_name}_{label}"] = time_callable(
                lambda w=writer, proj=projection: _write_commit(
                    w,
                    fixture,
                    proj,
                    inplace=True,
                ),
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
            )
            stats_by_variant[f"split_{writer_name}_{label}"] = time_callable(
                lambda gs=group_size, w=writer: _write_commit(
                    w,
                    fixture,
                    _project_commit(fixture, group_size=gs),
                    inplace=True,
                ),
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
            )

    print_stats_block("materializer_breakdown_commit", stats_by_variant)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DFlash materializer ablations on matched inputs."
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

    group_sizes = _parse_sizes(args.group_sizes, fallback_all=args.num_layers)
    chunk_sizes = _parse_sizes(args.chunk_sizes, fallback_all=args.num_tokens)
    _run_prompt(args, chunk_sizes=chunk_sizes, group_sizes=group_sizes)
    _run_commit(args, group_sizes=group_sizes)


if __name__ == "__main__":
    main()
