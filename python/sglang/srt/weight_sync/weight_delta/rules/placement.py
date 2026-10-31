"""Which part of a canonical tensor this GPU holds, and where in the parameter."""

from __future__ import annotations

import math

import torch

from sglang.srt.distributed.utils import get_group_rank_size
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.weight_sync.weight_delta.map import (
    OTHER_GPU,
    TORCH_DTYPES,
    CanonicalTensor,
    Derived,
    Place,
)
from sglang.srt.weight_sync.weight_delta.rules import (
    cutedsl_nvfp4,
    fp8_block,
    unquantized,
)

BACKENDS = (cutedsl_nvfp4, fp8_block, unquantized)


def linear_places(
    module, param, destination, tensor: CanonicalTensor, inventory
) -> tuple[Place]:
    """Whole, or this GPU's TP slice, at the destination's offset inside a fused parameter."""
    rank, size = get_group_rank_size(getattr(module, "tp_group", None))
    src = [slice(None)] * len(tensor.shape)
    split_dim = None
    if size > 1:
        if isinstance(module, RowParallelLinear):
            split_dim = 1
        elif isinstance(module, ColumnParallelLinear) and not isinstance(
            module, QKVParallelLinear
        ):
            split_dim = 0
        else:
            raise ValueError(f"{tensor.name}: no TP rule for {type(module).__name__}")
        local = tensor.shape[split_dim] // size
        if local * size != tensor.shape[split_dim]:
            raise ValueError(
                f"{tensor.name}: dim {split_dim} does not split {size} ways"
            )
        src[split_dim] = slice(rank * local, (rank + 1) * local)
    dst = param
    if destination.shards:
        rows = [
            inventory[shard].shape[0] // (size if split_dim == 0 else 1)
            for shard in destination.shards
        ]
        if sum(rows) != param.shape[0]:
            raise ValueError(
                f"{tensor.name}: fused shards do not fill {destination.param}"
            )
        dst = param.narrow(0, sum(rows[: destination.shard]), rows[destination.shard])
    elements = math.prod(len(range(*s.indices(n))) for s, n in zip(src, tensor.shape))
    if dst.numel() != elements:
        raise ValueError(
            f"{tensor.name}: {elements} elements, {destination.param} holds {dst.numel()}"
        )
    if dst.dtype != TORCH_DTYPES[tensor.dtype] and len(tensor.shape) > 1:
        raise ValueError(f"{tensor.name}: loader converts a matrix to {dst.dtype}")
    return (Place(tuple(src), dst, module.tp_group if size > 1 else None),)


def _backend_for(layer):
    for backend in BACKENDS:
        if backend.handles(layer):
            return backend
    runner_backend = getattr(
        getattr(layer.quant_method, "runner", None), "runner_backend", None
    )
    raise ValueError(
        f"Weight delta has no layout for {type(layer.quant_method).__name__}"
        + (f" on the {runner_backend} MoE runner" if runner_backend is not None else "")
    )


def expert_places(layer, destination, tensor: CanonicalTensor):
    """A routed expert lives whole on the GPU that owns it."""
    if layer.moe_tp_size != 1:
        raise ValueError("Weight delta needs expert TP 1")
    slot = layer._map_global_expert_id_to_local_expert_id(destination.expert)
    if slot < 0:
        return OTHER_GPU
    place = _backend_for(layer).expert_place(
        layer, slot, destination.projection, destination.kind, tensor
    )
    if place.dst.numel() != math.prod(tensor.shape):
        raise ValueError(f"{tensor.name}: expert layout does not match the checkpoint")
    return (place,)


def expert_copies(module):
    if not hasattr(module, "w13_weight") or not hasattr(module, "quant_method"):
        return []
    return _backend_for(module).copies(module)


def layer_derived(module) -> list[Derived]:
    """GemmaRMSNorm serves gemma_weight = weight + 1 (refresh_gemma_weight)."""
    if not hasattr(module, "refresh_gemma_weight"):
        return []
    return [Derived(lambda: [(torch.add(module.weight, 1.0), module.gemma_weight)])]
