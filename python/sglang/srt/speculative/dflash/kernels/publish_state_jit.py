from __future__ import annotations

import torch

from sglang.jit_kernel.dflash_publish_state import publish_state_cuda
from sglang.srt.speculative.dflash.contracts import DFlashRequestStateTable
from sglang.srt.speculative.dflash.kernels.publish_state_triton import (
    _validate_publish_state_inputs,
)


def publish_state_jit(
    *,
    state: DFlashRequestStateTable,
    req_pool_indices: torch.Tensor,
    req_generation: torch.Tensor,
    commit_lens: torch.Tensor,
    bonus_ids: torch.Tensor,
    gpu_stop_flags: torch.Tensor,
    inplace: bool = False,
) -> DFlashRequestStateTable:
    req_pool_indices, req_generation, commit_lens, bonus_ids, gpu_stop_flags = (
        _validate_publish_state_inputs(
            state=state,
            req_pool_indices=req_pool_indices,
            req_generation=req_generation,
            commit_lens=commit_lens,
            bonus_ids=bonus_ids,
            gpu_stop_flags=gpu_stop_flags,
        )
    )
    updated = state if inplace else state.clone()
    oob_flags = torch.zeros(
        (int(req_pool_indices.numel()),),
        dtype=torch.int32,
        device=updated.generation.device,
    )

    publish_state_cuda(
        updated.committed_len,
        updated.reserved_len,
        updated.next_verified_id,
        updated.generation,
        updated.status_flags,
        req_pool_indices.contiguous(),
        req_generation.contiguous(),
        commit_lens.contiguous(),
        bonus_ids.contiguous(),
        gpu_stop_flags.contiguous(),
        oob_flags,
    )

    if bool(oob_flags.any().item()):
        raise ValueError(
            "publish_state_jit would advance committed_len beyond reserved_len."
        )
    updated.validate()
    return updated
