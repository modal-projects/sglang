"""Muse Spark (models/muse_spark.py, muse_spark_omni.py), its language model only, with BF16 experts."""

from __future__ import annotations

import re

import torch

from sglang.srt.weight_sync.weight_delta.map import (
    OTHER_GPU,
    Derived,
    Destination,
    Ignored,
    Place,
)

EXPERT = re.compile(
    r"(?P<module>.+\.mlp\.experts)\.(?P<expert>\d+)\.(?P<projection>gate|up|down)_proj\.weight$"
)
# The checkpoint's q/k/v/output-gate projections, in the order the loader stacks them into qkv_proj.
QKVG = ("q_proj", "k_proj", "v_proj", "gate_proj")
TOWERS = (
    "model.vision_tower.",
    "model.vision_adapter.",
    "model.vision_projection.",
    "model.audio_tower.",
)
# TRT-LLM's BF16 MoE layout (MuseSpark1xBf16Experts._shuffle_weights): rows permuted in tiles of 32 for the
# epilogue (and gate/up interleaved), each row's K in blocks of 64 elements, block-major.
ROW_TILE, K_BLOCK = 32, 64


class MuseSpark:
    architectures = {"MuseSparkForConditionalGeneration"}

    def __init__(self, models, inventory):
        self.model = models["target"]
        self.inventory = inventory
        self.params = dict(self.model.named_parameters())
        self.modules = dict(self.model.named_modules())
        self.checked_layouts: set[tuple[int, int]] = set()

    def destination(self, name: str) -> Destination | Ignored:
        if name.startswith(TOWERS):
            if not getattr(self.model, "language_only", False):
                raise ValueError(
                    "Weight delta serves Muse Spark's language model only (--language-model-only)"
                )
            return Ignored("the engine serves the language model only")
        expert = EXPERT.match(name)
        if expert:
            return Destination(
                "target", build=lambda tensor: self._expert_places(expert, tensor)
            )
        for shard, leaf in enumerate(QKVG):
            suffix = f".self_attn.{leaf}.weight"
            if name.endswith(suffix):
                prefix = name[: -len(suffix)]
                return Destination(
                    "target",
                    build=lambda tensor: self._qkvg_places(prefix, shard, tensor),
                )
        return Destination("target", param=name)

    def copies(self, module) -> list:
        return []

    def derived(self, role, path, module, shadows) -> list[Derived]:
        """The model's own post-load: it refreshes every cache derived from the weights in place
        (residual gate coefficients, effective norm weights, the fused boundaries' copies, the
        router's split, the experts' finalize weight)."""
        if path != "":
            return []

        def refresh():
            module.post_load_weights()
            return []

        return [Derived(refresh)]

    def _qkvg_places(self, prefix: str, shard: int, tensor) -> tuple[Place]:
        param = self.params[f"{prefix}.self_attn.qkv_proj.weight"]
        rows = [
            self.inventory[f"{prefix}.self_attn.{leaf}.weight"].shape[0]
            for leaf in QKVG
        ]
        if sum(rows) != param.shape[0]:
            raise ValueError(f"{prefix}: qkv_proj does not hold every head (attention TP 1)")
        region = param.narrow(0, sum(rows[:shard]), rows[shard])
        if QKVG[shard] in ("q_proj", "k_proj"):
            config = self.model.config
            heads = (
                config.num_attention_heads
                if QKVG[shard] == "q_proj"
                else config.num_key_value_heads
            )
            # The loader interleaves each head's rotary halves: checkpoint row (head, half, i) is
            # served at row (head, i, half).
            region = region.view(heads, rows[shard] // heads // 2, 2, -1).transpose(1, 2)
        return (Place((slice(None),) * len(tensor.shape), region),)

    def _expert_places(self, match, tensor):
        experts = self.modules[match["module"]]
        if type(experts).__name__ != "MuseSpark1xBf16Experts":
            raise ValueError(f"{tensor.name}: no layout for {type(experts).__name__}")
        slot = int(match["expert"]) - experts.local_expert_offset
        if not 0 <= slot < experts.num_local_experts:
            return OTHER_GPU
        projection = match["projection"]
        if experts.use_trtllm:
            dst = self._trtllm_view(experts, slot, projection)
        else:
            if projection == "down":
                dst = experts.down_proj.data[slot]
            else:
                half = experts.intermediate_size
                rows = slice(0, half) if projection == "gate" else slice(half, 2 * half)
                dst = experts.gate_up_proj.data[slot][rows]
        if dst.numel() != tensor.shape[0] * tensor.shape[1]:
            raise ValueError(f"{tensor.name}: expert layout does not match the checkpoint")
        return (Place((slice(None), slice(None)), dst),)

    def _trtllm_view(self, experts, slot: int, projection: str) -> torch.Tensor:
        """The expert's projection as a view of the shuffled parameter, in checkpoint order."""
        half = experts.intermediate_size
        hidden = experts.gate_up_proj.numel() // experts.num_local_experts // (2 * half)
        if (half, hidden) not in self.checked_layouts:
            _check_trtllm_layout(half, hidden, experts.gate_up_proj.device)
            self.checked_layouts.add((half, hidden))
        tile = ROW_TILE * K_BLOCK
        if projection == "down":
            down = experts.down_proj.data
            out_rows, k = hidden, half
            return torch.as_strided(
                down,
                (out_rows // ROW_TILE, 8, 4, k // K_BLOCK, K_BLOCK),
                (tile, K_BLOCK, 8 * K_BLOCK, out_rows * K_BLOCK, 1),
                down.storage_offset() + slot * out_rows * k,
            )
        gate_up = experts.gate_up_proj.data
        # The kernel stacks [up; gate] and interleaves them in groups of 8 rows.
        stack = 1 if projection == "gate" else 0
        return torch.as_strided(
            gate_up,
            (half // 16, 8, 2, hidden // K_BLOCK, K_BLOCK),
            (tile, K_BLOCK, 16 * K_BLOCK, 2 * half * K_BLOCK, 1),
            gate_up.storage_offset() + slot * 2 * half * hidden + stack * 8 * K_BLOCK,
        )


def _check_trtllm_layout(half: int, hidden: int, device: torch.device) -> None:
    """The views above against FlashInfer's own permutations for these shapes; refused if they differ."""
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    from sglang.srt.layers.muse_spark.muse_spark_bf16_moe import _BLOCK_K_BYTES, _EPILOGUE_TILE_M

    # Row n of a shuffled matrix holds row order[n] of the matrix the kernel was given.
    cache = {}
    w13 = torch.empty(2 * half, 2 * hidden, dtype=torch.uint8, device=device)
    w13_order = _maybe_get_cached_w3_w1_permute_indices(cache, w13, _EPILOGUE_TILE_M)
    w2 = torch.empty(hidden, 2 * half, dtype=torch.uint8, device=device)
    w2_order = get_w2_permute_indices_with_cache(cache, w2, _EPILOGUE_TILE_M)
    # [up; gate] as the kernel is given it: checkpoint rows [gate; up] = [0, half) and [half, 2 half).
    given = torch.cat([torch.arange(half, 2 * half), torch.arange(0, half)])
    expected_w13 = given[w13_order.to("cpu", torch.int64)]
    n = torch.arange(2 * half)
    q, r = n // ROW_TILE, n % ROW_TILE
    # The view: checkpoint row t = (q, a, b) of gate (stack 1) or up (stack 0) at 32 q + 16 b + 8 stack + a.
    a, b, stack = r % 8, r // 16, (r // 8) % 2
    ours_w13 = torch.where(stack == 1, 0, half) + q * 16 + a * 2 + b
    n2 = torch.arange(hidden)
    q2, r2 = n2 // ROW_TILE, n2 % ROW_TILE
    ours_w2 = q2 * ROW_TILE + (r2 % 8) * 4 + r2 // 8
    # Two rows of two K blocks each: block-major is block 0 of both rows, then block 1.
    labels = torch.arange(4 * _BLOCK_K_BYTES).view(2, 2 * _BLOCK_K_BYTES) % 256
    blocks = convert_to_block_layout(labels.to(torch.uint8).to(device), _BLOCK_K_BYTES)
    block_major = labels.view(2, 2, _BLOCK_K_BYTES).transpose(0, 1).flatten()
    if (
        not torch.equal(expected_w13, ours_w13)
        or not torch.equal(w2_order.to("cpu", torch.int64), ours_w2)
        or not torch.equal(blocks.flatten().to("cpu", torch.int64), block_major)
        or _BLOCK_K_BYTES != 2 * K_BLOCK
    ):
        raise ValueError(
            f"Weight delta has no layout for this FlashInfer's TRT-LLM BF16 MoE shuffle "
            f"(intermediate {half}, hidden {hidden})"
        )
