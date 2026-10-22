"""Measure shared raw-tail storage with exact BF16 rows, without loading a model."""

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.speculative.dflash_cache import DFlashCacheSizing
from sglang.srt.speculative.dflash_feature_tail import DFlashContextCapture


def measure(requests=32, window=2048, width=12288, prefill_tokens=4096):
    if prefill_tokens < window or window % 256:
        raise ValueError(
            "Use whole 256-row windows and a prefill budget of at least one window"
        )
    block, gap, suffix = 16, 512, 64
    dtype, device = torch.bfloat16, "cuda"
    rows = torch.arange(window + gap + suffix, device=device)
    source = torch.empty((prefill_tokens, width), dtype=dtype, device=device)
    verify = torch.empty((requests, block, width), dtype=dtype, device=device)
    reqs = [
        SimpleNamespace(
            kv=ReqKvInfo(req_pool_idx=i + 1),
            finished=lambda: False,
            is_retracted=False,
        )
        for i in range(requests)
    ]
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    per_batch = prefill_tokens // window
    for offset in range(0, requests, per_batch):
        batch = reqs[offset : offset + per_batch]
        for i in range(len(batch)):
            values = ((rows[:window] + 13 * (offset + i)) % 127).to(dtype)
            source[i * window : (i + 1) * window].copy_(values[:, None])
        capture = DFlashContextCapture.prefill(
            reqs=batch,
            target_hidden=source[: len(batch) * window],
            prefix_lens=[0] * len(batch),
            extend_lens=[window] * len(batch),
            checkpoints=[(0, window)] * len(batch),
            window_size=window,
        )
        source.fill_(-1)
        capture.consume(batch)

    for start in range(window, window + gap + suffix, block):
        for i in range(requests):
            verify[i].copy_(
                ((rows[start : start + block] + 13 * i) % 127).to(dtype)[:, None]
            )
        checkpoint = start + block == window + gap
        capture = DFlashContextCapture.decode(
            reqs=reqs,
            target_hidden=verify,
            start_positions=torch.full((requests,), start, device=device),
            commit_lens=torch.full((requests,), block, device=device),
            checkpoint_steps=torch.full((requests,), block - 1, device=device)
            if checkpoint
            else None,
            checkpoint_indices=torch.ones(requests, dtype=torch.int64, device=device)
            if checkpoint
            else None,
            checkpoint_slots=[1] * requests if checkpoint else None,
            window_size=window,
        )
        verify.fill_(-2)
        capture.metadata = capture.metadata.cpu()
        capture.consume(reqs)

    tails = [
        (
            req.kv.dflash_context_state.checkpoint(0, window),
            req.kv.dflash_context_state.checkpoint(1, window + gap),
            req.kv.dflash_context_state.snapshot(),
        )
        for req in reqs
    ]
    storages = {
        part.untyped_storage().data_ptr(): part.untyped_storage().nbytes()
        for windows in tails
        for tail in windows
        for part in tail.parts
    }
    gc.collect()
    torch.cuda.synchronize()
    observed = torch.cuda.memory_allocated() - baseline
    peak = torch.cuda.max_memory_allocated() - baseline
    sizing = DFlashCacheSizing(
        window_size=window,
        block_size=block,
        max_requests=requests,
        feature_size=width,
        feature_dtype=dtype,
        kv_heads=4,
        head_dim=128,
        num_layers=6,
        kv_dtype=dtype,
        page_size=64,
        prefill_tokens=prefill_tokens,
        overlap=True,
        tap_cache=True,
    )
    assert peak <= sizing.context_gpu_reserve_bytes
    for i, windows in enumerate(tails):
        for tail in windows:
            pos = tail.start_pos
            for part in tail.parts:
                expected = ((rows[pos : pos + part.shape[0]] + 13 * i) % 127).to(dtype)
                assert torch.equal(part, expected[:, None].expand_as(part))
                pos += part.shape[0]
    return {
        "requests": requests,
        "window": window,
        "feature_size": width,
        "prefill_tokens": prefill_tokens,
        "dtype": str(dtype),
        "checkpoint_gap": gap,
        "live_suffix": suffix,
        "retained_feature_bytes": sum(storages.values()),
        "old_ring_and_two_snapshot_bytes": 3
        * requests
        * window
        * width
        * dtype.itemsize,
        "observed_allocation_bytes_above_sources": observed,
        "observed_peak_bytes_above_sources": peak,
        "context_admission_reserve_bytes": sizing.context_gpu_reserve_bytes,
        "exact_row_checks": True,
        "scope": "Raw-feature lifecycle on one CUDA stream; sources excluded. Admission reserve also includes materialized KV. Not a model-serving peak measurement.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = measure()
    encoded = json.dumps(result, indent=2) + "\n"
    print(encoded, end="")
    if args.output:
        args.output.write_text(encoded)
