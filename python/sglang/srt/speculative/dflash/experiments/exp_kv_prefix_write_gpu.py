from __future__ import annotations

import argparse

import torch

from sglang.srt.speculative.dflash.bench.common import time_callable
from sglang.srt.speculative.dflash.experiments.common import print_stats_block
from sglang.srt.speculative.dflash.experiments.fixtures import (
    CommitWriteFixture,
    PromptWriteFixture,
    make_commit_write_fixture,
    make_prompt_write_fixture,
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
from sglang.srt.speculative.dflash.reference.kv_prefix_write import (
    write_commit_prefix_reference,
    write_prompt_reference,
)


def _resolve_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'.")
    return mapping[normalized]


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


def _parse_int_dtype_list(raw: str) -> list[torch.dtype]:
    out: list[torch.dtype] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        out.append(_resolve_int_dtype(item))
    return out


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _cast_prompt_fixture(
    fixture: PromptWriteFixture,
    dtype: torch.dtype,
    slot_id_dtype: torch.dtype,
) -> PromptWriteFixture:
    return PromptWriteFixture(
        cache=type(fixture.cache)(
            k_cache=fixture.cache.k_cache.to(dtype=dtype),
            v_cache=fixture.cache.v_cache.to(dtype=dtype),
        ),
        config=fixture.config,
        slot_ids=fixture.slot_ids.to(dtype=slot_id_dtype),
        cache_k=fixture.cache_k.to(dtype=dtype),
        cache_v=fixture.cache_v.to(dtype=dtype),
        dummy_slot_id=fixture.dummy_slot_id,
    )


def _cast_commit_fixture(
    fixture: CommitWriteFixture,
    dtype: torch.dtype,
    slot_id_dtype: torch.dtype,
    commit_len_dtype: torch.dtype,
) -> CommitWriteFixture:
    return CommitWriteFixture(
        cache=type(fixture.cache)(
            k_cache=fixture.cache.k_cache.to(dtype=dtype),
            v_cache=fixture.cache.v_cache.to(dtype=dtype),
        ),
        config=fixture.config,
        slot_ids_2d=fixture.slot_ids_2d.to(dtype=slot_id_dtype),
        commit_lens=fixture.commit_lens.to(dtype=commit_len_dtype),
        cache_k=fixture.cache_k.to(dtype=dtype),
        cache_v=fixture.cache_v.to(dtype=dtype),
        dummy_slot_id=fixture.dummy_slot_id,
    )


def _assert_cache_equal(expected, actual) -> None:
    if not torch.equal(expected.k_cache, actual.k_cache):
        raise AssertionError("k_cache mismatch.")
    if not torch.equal(expected.v_cache, actual.v_cache):
        raise AssertionError("v_cache mismatch.")


def _valid_splits(fixture) -> list[int]:
    row_bytes = (
        int(fixture.config.num_kv_heads)
        * int(fixture.config.head_dim)
        * int(fixture.cache_k.element_size())
    )
    return [split for split in (1, 2, 4) if row_bytes % (split * 128) == 0]


def _run_prompt(
    fixture: PromptWriteFixture,
    *,
    warmup: int,
    iters: int,
    device: str,
) -> None:
    reference = write_prompt_reference(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids=fixture.slot_ids,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
    )
    torch_control = write_prompt_index_copy_control(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids=fixture.slot_ids,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
    )
    triton_out = write_prompt_triton(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids=fixture.slot_ids,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
    )
    jit_out = write_prompt_jit(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids=fixture.slot_ids,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
    )
    _assert_cache_equal(reference, torch_control)
    _assert_cache_equal(reference, triton_out)
    _assert_cache_equal(reference, jit_out)
    for split in _valid_splits(fixture):
        jit_split_out = write_prompt_jit(
            cache=fixture.cache.clone(),
            config=fixture.config,
            slot_ids=fixture.slot_ids,
            cache_k=fixture.cache_k,
            cache_v=fixture.cache_v,
            num_split=split,
        )
        _assert_cache_equal(reference, jit_split_out)

    stats = {
        "torch_index_copy": time_callable(
            lambda: write_prompt_index_copy_control(
                cache=fixture.cache,
                config=fixture.config,
                slot_ids=fixture.slot_ids,
                cache_k=fixture.cache_k,
                cache_v=fixture.cache_v,
                inplace=True,
            ),
            warmup=warmup,
            iters=iters,
            device=device,
        ),
        "triton": time_callable(
            lambda: write_prompt_triton(
                cache=fixture.cache,
                config=fixture.config,
                slot_ids=fixture.slot_ids,
                cache_k=fixture.cache_k,
                cache_v=fixture.cache_v,
                inplace=True,
            ),
            warmup=warmup,
            iters=iters,
            device=device,
        ),
        "jit_auto": time_callable(
            lambda: write_prompt_jit(
                cache=fixture.cache,
                config=fixture.config,
                slot_ids=fixture.slot_ids,
                cache_k=fixture.cache_k,
                cache_v=fixture.cache_v,
                inplace=True,
            ),
            warmup=warmup,
            iters=iters,
            device=device,
        ),
    }
    for split in _valid_splits(fixture):
        stats[f"jit_split{split}"] = time_callable(
            lambda s=split: write_prompt_jit(
                cache=fixture.cache,
                config=fixture.config,
                slot_ids=fixture.slot_ids,
                cache_k=fixture.cache_k,
                cache_v=fixture.cache_v,
                num_split=s,
                inplace=True,
            ),
            warmup=warmup,
            iters=iters,
            device=device,
        )
    print_stats_block(
        f"kv_prefix_write_prompt_gpu[slot={_dtype_name(fixture.slot_ids.dtype)}]",
        stats,
    )


def _run_commit(
    fixture: CommitWriteFixture,
    *,
    warmup: int,
    iters: int,
    device: str,
) -> None:
    reference = write_commit_prefix_reference(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids_2d=fixture.slot_ids_2d,
        commit_lens=fixture.commit_lens,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
    )
    torch_control = write_commit_prefix_flatten_control(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids_2d=fixture.slot_ids_2d,
        commit_lens=fixture.commit_lens,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
    )
    triton_out = write_commit_prefix_triton(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids_2d=fixture.slot_ids_2d,
        commit_lens=fixture.commit_lens,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
    )
    jit_out = write_commit_prefix_jit(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids_2d=fixture.slot_ids_2d,
        commit_lens=fixture.commit_lens,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
    )
    _assert_cache_equal(reference, torch_control)
    _assert_cache_equal(reference, triton_out)
    _assert_cache_equal(reference, jit_out)
    for split in _valid_splits(fixture):
        jit_split_out = write_commit_prefix_jit(
            cache=fixture.cache.clone(),
            config=fixture.config,
            slot_ids_2d=fixture.slot_ids_2d,
            commit_lens=fixture.commit_lens,
            cache_k=fixture.cache_k,
            cache_v=fixture.cache_v,
            num_split=split,
        )
        _assert_cache_equal(reference, jit_split_out)

    stats = {
        "torch_flatten": time_callable(
            lambda: write_commit_prefix_flatten_control(
                cache=fixture.cache,
                config=fixture.config,
                slot_ids_2d=fixture.slot_ids_2d,
                commit_lens=fixture.commit_lens,
                cache_k=fixture.cache_k,
                cache_v=fixture.cache_v,
                inplace=True,
            ),
            warmup=warmup,
            iters=iters,
            device=device,
        ),
        "triton": time_callable(
            lambda: write_commit_prefix_triton(
                cache=fixture.cache,
                config=fixture.config,
                slot_ids_2d=fixture.slot_ids_2d,
                commit_lens=fixture.commit_lens,
                cache_k=fixture.cache_k,
                cache_v=fixture.cache_v,
                inplace=True,
            ),
            warmup=warmup,
            iters=iters,
            device=device,
        ),
        "jit_auto": time_callable(
            lambda: write_commit_prefix_jit(
                cache=fixture.cache,
                config=fixture.config,
                slot_ids_2d=fixture.slot_ids_2d,
                commit_lens=fixture.commit_lens,
                cache_k=fixture.cache_k,
                cache_v=fixture.cache_v,
                inplace=True,
            ),
            warmup=warmup,
            iters=iters,
            device=device,
        ),
    }
    for split in _valid_splits(fixture):
        stats[f"jit_split{split}"] = time_callable(
            lambda s=split: write_commit_prefix_jit(
                cache=fixture.cache,
                config=fixture.config,
                slot_ids_2d=fixture.slot_ids_2d,
                commit_lens=fixture.commit_lens,
                cache_k=fixture.cache_k,
                cache_v=fixture.cache_v,
                num_split=s,
                inplace=True,
            ),
            warmup=warmup,
            iters=iters,
            device=device,
        )
    print_stats_block(
        "kv_prefix_write_commit_gpu"
        f"[slot={_dtype_name(fixture.slot_ids_2d.dtype)},len={_dtype_name(fixture.commit_lens.dtype)}]",
        stats,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA DFlash KV prefix write kernels."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-slots", type=int, default=4096)
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-slot-id-dtypes", default="int32,int64")
    parser.add_argument("--commit-slot-id-dtypes", default="int32,int64")
    parser.add_argument("--commit-len-dtypes", default="int32,int64")
    args = parser.parse_args()

    if torch.device(args.device).type != "cuda":
        raise ValueError("This benchmark is CUDA-only.")
    dtype = _resolve_dtype(args.dtype)

    prompt_base = make_prompt_write_fixture(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        num_slots=args.num_slots,
        num_tokens=args.num_tokens,
        device=args.device,
        seed=args.seed,
    )
    for slot_id_dtype in _parse_int_dtype_list(args.prompt_slot_id_dtypes):
        prompt_fixture = _cast_prompt_fixture(prompt_base, dtype, slot_id_dtype)
        _run_prompt(
            prompt_fixture,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )

    commit_base = make_commit_write_fixture(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        num_slots=args.num_slots,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=args.device,
        seed=args.seed,
    )
    for slot_id_dtype in _parse_int_dtype_list(args.commit_slot_id_dtypes):
        for commit_len_dtype in _parse_int_dtype_list(args.commit_len_dtypes):
            commit_fixture = _cast_commit_fixture(
                commit_base,
                dtype,
                slot_id_dtype,
                commit_len_dtype,
            )
            _run_commit(
                commit_fixture,
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
            )


if __name__ == "__main__":
    main()
