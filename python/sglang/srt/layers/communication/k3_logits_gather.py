# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 ROCm logits gather, installed by models/kimi_k3.py on its
LogitsProcessor via ``install``."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.environ import envs


def logits_all_gather_lastdim(logits: torch.Tensor) -> Optional[torch.Tensor]:
    """Gather vocab-parallel logits [rows, vocab/tp] -> [rows, vocab] with
    AITER's last-dim all-gather, which writes the final layout directly instead
    of a dim-0 gather plus a movedim copy (bit-identical). None when it does not
    apply."""
    if (
        not logits.is_cuda
        or logits.dim() != 2
        or logits.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or not logits.is_contiguous()
        or logits.shape[0] == 0
        or logits.shape[-1] % (16 // logits.element_size()) != 0
    ):
        return None
    from sglang.srt.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    ca = getattr(group, "ca_comm", None)
    if (
        ca is None
        or getattr(ca, "disabled", True)
        or not hasattr(ca, "all_gather_reg")
        or not hasattr(ca, "all_gather_unreg")
        or not ca.should_custom_ag(logits)
    ):
        return None
    out = logits.new_empty((logits.shape[0], logits.shape[1] * group.world_size))
    if getattr(ca, "_IS_CAPTURING", False):
        if torch.cuda.is_current_stream_capturing():
            if envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get():
                ca.all_gather_unreg(logits, out=out, dim=-1)
            else:
                ca.all_gather_reg(logits, out=out, dim=-1)
        else:
            # graph warm-up: no host collective (as _all_gather_into_tensor)
            out.zero_()
    else:
        ca.all_gather_unreg(logits, out=out, dim=-1)
    return out


def install(logits_processor) -> None:
    fallback = logits_processor._logits_gatherer

    def gather(logits: torch.Tensor) -> torch.Tensor:
        out = logits_all_gather_lastdim(logits)
        return out if out is not None else fallback(logits)

    logits_processor._logits_gatherer = gather
