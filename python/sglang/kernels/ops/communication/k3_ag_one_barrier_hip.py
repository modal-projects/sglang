"""K3 ROCm TP8: AITER's all_gather_lastdim_add without its trailing barrier
(bit-identical). Graph capture only; see
``jit/csrc/kimi_k3/comm/ag_one_barrier_hip.cuh``.
"""

from __future__ import annotations

import collections

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

WORLD_SIZE = 8

# Keeps each input alive until two calls later, past the next collective's
# start barrier, so the graph allocator cannot reuse it while peers read it.
_KEEPALIVE: collections.deque = collections.deque(maxlen=2)
_EMPTY: dict = {}


@cache_once
def _module():
    from aiter.jit.core import AITER_CSRC_DIR

    return load_jit(
        "k3_ag_one_barrier_hip",
        cuda_files=["kimi_k3/comm/ag_one_barrier_hip.cuh"],
        cuda_wrappers=[("run", "k3_ag_one_barrier::AgOneBarrier::run")],
        extra_include_paths=[f"{AITER_CSRC_DIR}/include"],
        # AITER's custom_all_reduce module flushes denormals; match its adds
        extra_cuda_cflags=["-O3", "-fgpu-flush-denormals-to-zero"],
    )


def all_gather_lastdim_add_1sync(
    ca,
    y: torch.Tensor,
    add_b: torch.Tensor,
    add_c: torch.Tensor | None,
) -> torch.Tensor:
    """bf16(bf16(all_gather(y, dim=-1) + add_b) [+ add_c])."""
    assert torch.cuda.is_current_stream_capturing() and ca._IS_CAPTURING
    rows, ld = y.shape
    out = torch.empty((rows, ld * WORLD_SIZE), dtype=y.dtype, device=y.device)
    if add_c is None:
        add_c = _EMPTY.get(y.device)
        if add_c is None:
            add_c = _EMPTY[y.device] = torch.empty(0, dtype=y.dtype, device=y.device)
    _module().run(ca._ptr, y, out, add_b, add_c)
    _KEEPALIVE.append(y)
    return out
