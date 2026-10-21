"""Measure DFlash context extension/reprojection with a local BF16 checkpoint.

Uses the production target-feature projection, fused six-layer materializer,
and unified-pool writes. TP cases measure one rank's sharded K/V geometry;
the feature projection is replicated. No target forward or attention runs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from torch import nn

from sglang.kernels.ops.speculative.fused_kv_materialize import (
    FusedKVMaterializeHelper,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.unified_memory_pool import (
    MHASubPoolSpec,
    UnifiedKVPool,
    UnifiedMHATokenToKVPool,
)
from sglang.srt.models.dflash import DFlashDraftModel


def load_context_model(model_path, tp_size, max_rows):
    config = SimpleNamespace(**json.loads((model_path / "config.json").read_text()))
    assert config.architectures == ["DFlash2DraftModel"]
    assert not config.attention_bias
    assert config.num_key_value_heads % tp_size == 0
    hidden = config.hidden_size
    taps = len(config.dflash_config["target_layer_ids"])
    heads = config.num_key_value_heads // tp_size
    device = torch.device("cuda", torch.cuda.current_device())
    rope = get_rope(
        config.head_dim,
        config.head_dim,
        config.max_position_embeddings,
        config.rope_parameters["rope_theta"],
        dtype=torch.bfloat16,
    ).to(device)
    fc = nn.Linear(
        taps * hidden, hidden, bias=False, device=device, dtype=torch.bfloat16
    )
    hidden_norm = RMSNorm(hidden, eps=config.rms_norm_eps).to(device, torch.bfloat16)
    layers = []
    with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as f:
        fc.weight.copy_(f.get_tensor("fc.weight"))
        hidden_norm.weight.copy_(f.get_tensor("hidden_norm.weight"))
        for i in range(config.num_hidden_layers):
            prefix = f"layers.{i}.self_attn."
            # Rank zero's contiguous Q/K/V head shards, in fused QKV order.
            weights = [
                f.get_tensor(prefix + name + "_proj.weight").chunk(tp_size, dim=0)[0]
                for name in ("q", "k", "v")
            ]
            norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps).to(
                device, torch.bfloat16
            )
            norm.weight.copy_(f.get_tensor(prefix + "k_norm.weight"))
            attn = SimpleNamespace(
                qkv_proj=SimpleNamespace(weight=torch.cat(weights).to(device)),
                q_size=weights[0].shape[0],
                kv_size=heads * config.head_dim,
                num_kv_heads=heads,
                head_dim=config.head_dim,
                k_norm=norm,
                rotary_emb=rope,
                attn=SimpleNamespace(layer_id=i),
            )
            layers.append(SimpleNamespace(self_attn=attn))
    model = SimpleNamespace(
        config=config,
        fc=fc,
        hidden_norm=hidden_norm,
        num_context_features=taps,
        quantizable=False,
    )
    helper = FusedKVMaterializeHelper(
        layers,
        rope,
        heads,
        config.head_dim,
        device,
        max_position_hint=config.max_position_embeddings,
    )
    helper._ensure_workspace(max_rows, torch.bfloat16)
    return model, layers, helper


def make_pool(config, tp_size, batch_size):
    spec = MHASubPoolSpec(
        name="swa",
        layer_num=config.num_hidden_layers,
        head_num=config.num_key_value_heads // tp_size,
        head_dim=config.head_dim,
        store_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        grow_direction="up",
    )
    page_size = 16
    # Unified arenas require another sub-pool; no target work runs here.
    full = MHASubPoolSpec(
        name="full",
        layer_num=1,
        head_num=1,
        head_dim=128,
        store_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        grow_direction="down",
    )
    arena = UnifiedKVPool(
        total_bytes=(batch_size * config.sliding_window + page_size)
        * spec.entry_bytes()
        + page_size * full.entry_bytes(),
        sub_pool_specs=[spec, full],
        device="cuda",
        enable_memory_saver=False,
        page_size=page_size,
    )
    pool = UnifiedMHATokenToKVPool(
        unified_buffer=arena,
        sub_pool_name="swa",
        page_size=page_size,
        start_layer=0,
        end_layer=config.num_hidden_layers - 1,
        enable_alt_stream=False,
    )
    return pool


def validate_materializer(model, layers, helper, pool):
    """Independent per-layer GEMMs and FP32 norm/RoPE check the measured path."""
    rows = 7
    raw = torch.randn(rows, model.fc.in_features, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(4096, 4096 + rows, device="cuda")
    locs = torch.arange(16, 16 + rows, device="cuda")
    projected = DFlashDraftModel.project_target_hidden(model, raw)

    def write(i, k, v):
        pool.set_kv_buffer(
            layers[i].self_attn.attn, KVWriteLoc(locs, physical=True), k, v
        )

    helper.materialize(projected, positions, write)
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        weight = attn.qkv_proj.weight[attn.q_size :]
        kv = torch.nn.functional.linear(projected, weight)
        k, v = kv.reshape(rows, 2, attn.num_kv_heads, attn.head_dim).unbind(1)
        k = k.float()
        k = k * torch.rsqrt(
            k.square().mean(-1, keepdim=True) + model.config.rms_norm_eps
        )
        k = k * attn.k_norm.weight.float()
        cos, sin = attn.rotary_emb.cos_sin_cache[positions].float().chunk(2, dim=-1)
        first, second = k.chunk(2, dim=-1)
        expected_k = torch.cat(
            (
                first * cos[:, None] - second * sin[:, None],
                second * cos[:, None] + first * sin[:, None],
            ),
            dim=-1,
        ).to(torch.bfloat16)
        torch.testing.assert_close(
            pool.get_key_buffer(i)[locs], expected_k, rtol=0.02, atol=0.02
        )
        torch.testing.assert_close(
            pool.get_value_buffer(i)[locs], v, rtol=0.02, atol=0.02
        )


def measure_graph(fn, samples, sample_ms):
    # Four calls per replay avoid Python replay overhead dominating small cases.
    calls_per_replay = 4
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls_per_replay):
            fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        graph.replay()
    end.record()
    end.synchronize()
    replay_ms = start.elapsed_time(end) / 5
    repeats = max(1, math.ceil(sample_ms / replay_ms))
    times = []
    for _ in range(samples):
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (repeats * calls_per_replay))
    return {
        "median_us": statistics.median(times),
        "samples_us": times,
        "calls_per_sample": repeats * calls_per_replay,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", type=Path, default=Path("/tmp/autospec/baseline")
    )
    parser.add_argument("--tp-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[1, 8, 16, 64, 128, 256, 512, 1024, 2048],
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--sample-ms", type=float, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_path": str(args.model_path),
        "timing": "CUDA graph, 4 calls/replay, warm buffers",
        "tp_note": "rank-local TP-shaped projection only; no distributed inference",
        "samples": args.samples,
        "sample_ms": args.sample_ms,
        "results": [],
    }
    for tp_size in args.tp_sizes:
        model, layers, helper = load_context_model(
            args.model_path, tp_size, max(args.batch_sizes) * max(args.lengths)
        )
        assert all(0 < n <= model.config.sliding_window for n in args.lengths)
        for batch_size in args.batch_sizes:
            pool = make_pool(model.config, tp_size, batch_size)
            validate_materializer(model, layers, helper, pool)
            for length in args.lengths:
                rows = batch_size * length
                raw = torch.randn(
                    rows, model.fc.in_features, device="cuda", dtype=torch.bfloat16
                )
                positions = torch.arange(4096, 4096 + length, device="cuda").repeat(
                    batch_size
                )
                locs = (
                    16
                    + torch.arange(batch_size, device="cuda")[:, None]
                    * model.config.sliding_window
                    + torch.arange(
                        model.config.sliding_window - length,
                        model.config.sliding_window,
                        device="cuda",
                    )[None]
                ).flatten()

                def write(i, k, v):
                    pool.set_kv_buffer(
                        layers[i].self_attn.attn, KVWriteLoc(locs, physical=True), k, v
                    )

                def project_features():
                    return DFlashDraftModel.project_target_hidden(model, raw)

                def project_kv():
                    helper.materialize(project_features(), positions, lambda *_: None)

                def extend_write():
                    helper.materialize(project_features(), positions, write)

                # Decode projects a dense block, then stores only commit prefixes.
                commit_lens = torch.full(
                    (batch_size,), min(8, length), device="cuda", dtype=torch.int32
                )

                def write_prefix(i, k, v):
                    pool.set_kv_buffer_prefix_valid(
                        layers[i].self_attn.attn,
                        KVWriteLoc(locs.view(batch_size, length), physical=True),
                        commit_lens,
                        k,
                        v,
                    )

                def extend_prefix():
                    helper.materialize(project_features(), positions, write_prefix)

                row = {
                    "tp_size": tp_size,
                    "batch_size": batch_size,
                    "rows_per_request": length,
                    "total_rows": rows,
                }
                for name, fn in (
                    ("fc_norm", project_features),
                    ("reproject_without_pool_write", project_kv),
                    ("extend_with_pool_write", extend_write),
                ):
                    row[name] = measure_graph(fn, args.samples, args.sample_ms)
                if length == model.config.dflash_config["block_size"]:
                    row["decode_block_commit_8"] = measure_graph(
                        extend_prefix, args.samples, args.sample_ms
                    )
                result["results"].append(row)
                print(
                    json.dumps(
                        {
                            k: v["median_us"] if isinstance(v, dict) else v
                            for k, v in row.items()
                        }
                    ),
                    flush=True,
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
