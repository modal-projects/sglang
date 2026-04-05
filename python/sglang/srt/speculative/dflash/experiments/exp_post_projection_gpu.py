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
    materialize_commit_per_layer_control,
    materialize_prompt_grouped_control,
    materialize_prompt_per_layer_control,
)
from sglang.srt.speculative.dflash.kernels.post_projection import (
    postprocess_commit_control,
    postprocess_prompt_control,
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


def _parse_dtype(raw: str) -> torch.dtype:
    item = raw.strip().lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if item not in mapping:
        raise ValueError(f"Unsupported dtype {raw!r}.")
    return mapping[item]


def _assert_cache_close(expected, actual, *, atol: float, rtol: float) -> None:
    if not torch.allclose(expected.k_cache, actual.k_cache, atol=atol, rtol=rtol):
        raise AssertionError("k_cache mismatch in post-projection benchmark.")
    if not torch.allclose(expected.v_cache, actual.v_cache, atol=atol, rtol=rtol):
        raise AssertionError("v_cache mismatch in post-projection benchmark.")


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
        dtype=_parse_dtype(args.dtype),
        seed=args.seed,
    )
    reference = materialize_prompt_reference(
        cache=fixture.cache.clone(),
        config=fixture.config,
        weights=fixture.weights,
        hidden=fixture.hidden,
        positions=fixture.positions,
        slot_ids=fixture.slot_ids,
    )
    providers = [("torch", postprocess_prompt_control)]
    if str(args.device).startswith("cuda"):
        providers.extend(
            [("triton", postprocess_prompt_triton), ("jit", postprocess_prompt_jit)]
        )

    stats_by_variant: dict[str, object] = {}
    atol = 1e-1 if fixture.hidden.dtype in (torch.float16, torch.bfloat16) else 1e-5
    rtol = atol
    for chunk_size in chunk_sizes:
        for label, group_size in [
            ("per_layer", 1),
            *[(f"g{g}", g) for g in group_sizes],
        ]:
            projected = project_raw_prompt_control(
                config=fixture.config,
                weights=fixture.weights,
                hidden=fixture.hidden,
                positions=fixture.positions,
                group_size=group_size,
                chunk_size=chunk_size,
            )
            stats_by_variant[f"raw_project_{label}_c{chunk_size}"] = time_callable(
                lambda gs=group_size, cs=chunk_size: project_raw_prompt_control(
                    config=fixture.config,
                    weights=fixture.weights,
                    hidden=fixture.hidden,
                    positions=fixture.positions,
                    group_size=gs,
                    chunk_size=cs,
                ),
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
            )
            if group_size == 1:
                full_fn = lambda cs=chunk_size: materialize_prompt_per_layer_control(
                    cache=fixture.cache,
                    config=fixture.config,
                    weights=fixture.weights,
                    hidden=fixture.hidden,
                    positions=fixture.positions,
                    slot_ids=fixture.slot_ids,
                    chunk_size=cs,
                    inplace=True,
                )
            else:
                full_fn = lambda gs=group_size, cs=chunk_size: materialize_prompt_grouped_control(
                    cache=fixture.cache,
                    config=fixture.config,
                    weights=fixture.weights,
                    hidden=fixture.hidden,
                    positions=fixture.positions,
                    slot_ids=fixture.slot_ids,
                    group_size=gs,
                    chunk_size=cs,
                    inplace=True,
                )
            full_stats = time_callable(
                full_fn,
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
            )
            stats_by_variant[f"full_{label}_c{chunk_size}"] = full_stats
            for provider_name, provider in providers:
                actual = provider(
                    cache=fixture.cache.clone(),
                    config=fixture.config,
                    weights=fixture.weights,
                    projected=projected,
                    positions=fixture.positions,
                    slot_ids=fixture.slot_ids,
                    cos_sin_cache=fixture.cos_sin_cache,
                )
                _assert_cache_close(reference, actual, atol=atol, rtol=rtol)
                stats_by_variant[f"postproj_{provider_name}_{label}_c{chunk_size}"] = (
                    time_callable(
                        lambda p=provider, proj=projected: p(
                            cache=fixture.cache,
                            config=fixture.config,
                            weights=fixture.weights,
                            projected=proj,
                            positions=fixture.positions,
                            slot_ids=fixture.slot_ids,
                            cos_sin_cache=fixture.cos_sin_cache,
                            inplace=True,
                        ),
                        warmup=args.warmup,
                        iters=args.iters,
                        device=args.device,
                    )
                )
                stats_by_variant[f"split_{provider_name}_{label}_c{chunk_size}"] = (
                    time_callable(
                        lambda p=provider, gs=group_size, cs=chunk_size: p(
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
                )

    print_stats_block("post_projection_prompt", stats_by_variant)


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
        dtype=_parse_dtype(args.dtype),
        seed=args.seed,
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
    providers = [("torch", postprocess_commit_control)]
    if str(args.device).startswith("cuda"):
        providers.extend(
            [("triton", postprocess_commit_triton), ("jit", postprocess_commit_jit)]
        )

    stats_by_variant: dict[str, object] = {}
    atol = (
        1e-1 if fixture.verify_hidden.dtype in (torch.float16, torch.bfloat16) else 1e-5
    )
    rtol = atol
    for label, group_size in [("per_layer", 1), *[(f"g{g}", g) for g in group_sizes]]:
        projected = project_raw_commit_control(
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
            group_size=group_size,
        )
        stats_by_variant[f"raw_project_{label}"] = time_callable(
            lambda gs=group_size: project_raw_commit_control(
                config=fixture.config,
                weights=fixture.weights,
                verify_hidden=fixture.verify_hidden,
                positions=fixture.positions,
                group_size=gs,
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        if group_size == 1:
            full_fn = lambda: materialize_commit_per_layer_control(
                cache=fixture.cache,
                config=fixture.config,
                weights=fixture.weights,
                verify_hidden=fixture.verify_hidden,
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                commit_lens=fixture.commit_lens,
                inplace=True,
            )
        else:
            full_fn = lambda gs=group_size: materialize_commit_grouped_control(
                cache=fixture.cache,
                config=fixture.config,
                weights=fixture.weights,
                verify_hidden=fixture.verify_hidden,
                positions=fixture.positions,
                slot_ids=fixture.slot_ids,
                commit_lens=fixture.commit_lens,
                group_size=gs,
                inplace=True,
            )
        full_stats = time_callable(
            full_fn,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        stats_by_variant[f"full_{label}"] = full_stats
        for provider_name, provider in providers:
            actual = provider(
                cache=fixture.cache.clone(),
                config=fixture.config,
                weights=fixture.weights,
                projected=projected,
                positions=fixture.positions,
                slot_ids_2d=fixture.slot_ids,
                commit_lens=fixture.commit_lens,
                cos_sin_cache=fixture.cos_sin_cache,
            )
            _assert_cache_close(reference, actual, atol=atol, rtol=rtol)
            stats_by_variant[f"postproj_{provider_name}_{label}"] = time_callable(
                lambda p=provider, proj=projected: p(
                    cache=fixture.cache,
                    config=fixture.config,
                    weights=fixture.weights,
                    projected=proj,
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
            stats_by_variant[f"split_{provider_name}_{label}"] = time_callable(
                lambda p=provider, gs=group_size: p(
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

    print_stats_block("post_projection_commit", stats_by_variant)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DFlash post-projection GPU variants."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
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
