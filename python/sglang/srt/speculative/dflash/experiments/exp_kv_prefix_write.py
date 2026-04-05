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
    write_commit_masked_dummy_control,
    write_commit_prefix_flatten_control,
    write_commit_prefix_rowwise_control,
    write_prompt_index_copy_control,
)
from sglang.srt.speculative.dflash.reference.kv_prefix_write import (
    write_commit_prefix_reference,
    write_prompt_reference,
)


def _assert_cache_equal(
    expected,
    actual,
    *,
    ignore_slot_id: int | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    if ignore_slot_id is None:
        if atol == 0.0 and rtol == 0.0:
            if not torch.equal(expected.k_cache, actual.k_cache):
                raise AssertionError("k_cache mismatch.")
            if not torch.equal(expected.v_cache, actual.v_cache):
                raise AssertionError("v_cache mismatch.")
        else:
            if not torch.allclose(
                expected.k_cache, actual.k_cache, atol=atol, rtol=rtol
            ):
                raise AssertionError("k_cache mismatch.")
            if not torch.allclose(
                expected.v_cache, actual.v_cache, atol=atol, rtol=rtol
            ):
                raise AssertionError("v_cache mismatch.")
        return

    mask = torch.ones(
        (expected.num_slots,), dtype=torch.bool, device=expected.k_cache.device
    )
    mask[ignore_slot_id] = False
    if atol == 0.0 and rtol == 0.0:
        if not torch.equal(expected.k_cache[:, mask], actual.k_cache[:, mask]):
            raise AssertionError("k_cache mismatch outside ignored slot.")
        if not torch.equal(expected.v_cache[:, mask], actual.v_cache[:, mask]):
            raise AssertionError("v_cache mismatch outside ignored slot.")
    else:
        if not torch.allclose(
            expected.k_cache[:, mask], actual.k_cache[:, mask], atol=atol, rtol=rtol
        ):
            raise AssertionError("k_cache mismatch outside ignored slot.")
        if not torch.allclose(
            expected.v_cache[:, mask], actual.v_cache[:, mask], atol=atol, rtol=rtol
        ):
            raise AssertionError("v_cache mismatch outside ignored slot.")


def _run_prompt(
    fixture: PromptWriteFixture, *, warmup: int, iters: int, device: str
) -> None:
    reference = write_prompt_reference(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids=fixture.slot_ids,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
    )
    control = write_prompt_index_copy_control(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids=fixture.slot_ids,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
    )
    _assert_cache_equal(reference, control)
    stats = {
        "prompt_index_copy": time_callable(
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
        )
    }
    print_stats_block("kv_prefix_write_prompt", stats)


def _run_commit(
    fixture: CommitWriteFixture, *, warmup: int, iters: int, device: str
) -> None:
    reference = write_commit_prefix_reference(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids_2d=fixture.slot_ids_2d,
        commit_lens=fixture.commit_lens,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
    )
    rowwise = write_commit_prefix_rowwise_control(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids_2d=fixture.slot_ids_2d,
        commit_lens=fixture.commit_lens,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
    )
    flatten = write_commit_prefix_flatten_control(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids_2d=fixture.slot_ids_2d,
        commit_lens=fixture.commit_lens,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
    )
    masked = write_commit_masked_dummy_control(
        cache=fixture.cache.clone(),
        config=fixture.config,
        slot_ids_2d=fixture.slot_ids_2d,
        commit_lens=fixture.commit_lens,
        cache_k=fixture.cache_k,
        cache_v=fixture.cache_v,
        dummy_slot_id=fixture.dummy_slot_id,
    )
    _assert_cache_equal(reference, rowwise)
    _assert_cache_equal(reference, flatten)
    _assert_cache_equal(reference, masked, ignore_slot_id=fixture.dummy_slot_id)
    stats = {
        "commit_rowwise": time_callable(
            lambda: write_commit_prefix_rowwise_control(
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
        "commit_flatten": time_callable(
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
        "commit_masked_dummy": time_callable(
            lambda: write_commit_masked_dummy_control(
                cache=fixture.cache,
                config=fixture.config,
                slot_ids_2d=fixture.slot_ids_2d,
                commit_lens=fixture.commit_lens,
                cache_k=fixture.cache_k,
                cache_v=fixture.cache_v,
                dummy_slot_id=fixture.dummy_slot_id,
                inplace=True,
            ),
            warmup=warmup,
            iters=iters,
            device=device,
        ),
    }
    print_stats_block("kv_prefix_write_commit", stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark isolated DFlash KV prefix write variants."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--num-slots", type=int, default=1024)
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prompt_fixture = make_prompt_write_fixture(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        num_slots=args.num_slots,
        num_tokens=args.num_tokens,
        device=args.device,
        seed=args.seed,
    )
    _run_prompt(
        prompt_fixture,
        warmup=args.warmup,
        iters=args.iters,
        device=args.device,
    )

    commit_fixture = make_commit_write_fixture(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        num_slots=args.num_slots,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=args.device,
        seed=args.seed,
    )
    _run_commit(
        commit_fixture,
        warmup=args.warmup,
        iters=args.iters,
        device=args.device,
    )


if __name__ == "__main__":
    main()
