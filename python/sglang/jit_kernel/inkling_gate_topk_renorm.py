"""Shape-specialized Inkling MoE gate top-k + renorm JIT kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "inkling_gate_topk_renorm",
        "fast_math",
        cuda_files=["moe/inkling_gate_topk_renorm.cuh"],
        cuda_wrappers=[
            ("inkling_gate_topk_renorm", "inkling_gate_topk_renorm"),
            ("inkling_gate_topk_renorm_packed", "inkling_gate_topk_renorm_packed"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


def _launch_inkling_gate_topk_renorm(
    logits: torch.Tensor,
    bias: torch.Tensor,
    global_scale: torch.Tensor,
    routed_w: torch.Tensor,
    shared_w: torch.Tensor,
    indices: torch.Tensor,
    route_scale: float,
) -> None:
    module = _jit_module()
    module.inkling_gate_topk_renorm(
        logits, bias, global_scale, routed_w, shared_w, indices, float(route_scale)
    )


def inkling_gate_topk_renorm(
    logits: torch.Tensor,
    bias: torch.Tensor,
    global_scale: torch.Tensor,
    route_scale: float,
    *,
    return_packed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
    """Select top-6 routed experts from 256 and renorm with 2 shared experts.

    This is specialized for the Inkling fused gate layout:
    ``logits`` is ``[tokens, 258]`` fp32, where columns ``0:256`` are routed
    experts and columns ``256:258`` are shared experts. The top-k selection key is
    ``sigmoid(logits[:, :256]) + bias``; renorm is over sigmoid(raw logits) for
    the selected routed experts plus both shared experts.

    ``return_packed=True`` emits the FlashInfer routed-MoE pack instead of the
    routed_w + indices pair: ``packed[t,6]`` int32 = ``(expert_id << 16) | bf16
    weight bits``. Returns ``(packed, shared_w)``.
    """
    assert logits.is_cuda and logits.dtype == torch.float32 and logits.dim() == 2
    assert logits.shape[1] == 258 and logits.stride(1) == 1
    assert bias.is_cuda and bias.dtype == torch.float32 and bias.shape == (256,)
    assert global_scale.is_cuda and global_scale.dtype == torch.float32
    assert global_scale.numel() == 1

    tokens = logits.shape[0]
    shared_w = torch.empty((tokens, 2), dtype=torch.float32, device=logits.device)
    if return_packed:
        packed = torch.empty((tokens, 6), dtype=torch.int32, device=logits.device)
        if tokens == 0:
            return packed, shared_w
        _jit_module().inkling_gate_topk_renorm_packed(
            logits,
            bias.contiguous(),
            global_scale.contiguous(),
            packed,
            shared_w,
            float(route_scale),
        )
        return packed, shared_w

    routed_w = torch.empty((tokens, 6), dtype=torch.float32, device=logits.device)
    indices = torch.empty((tokens, 6), dtype=torch.int64, device=logits.device)
    if tokens == 0:
        return routed_w, shared_w, indices

    _launch_inkling_gate_topk_renorm(
        logits,
        bias.contiguous(),
        global_scale.contiguous(),
        routed_w,
        shared_w,
        indices,
        route_scale,
    )
    return routed_w, shared_w, indices
