"""K3 ROCm TP8: MoE [latent | shared] all-reduce fused with the latent RMSNorm
(SGLANG_ROCM_K3_MOE_AR_NORM_FUSED).

One launch instead of AITER's cross_device_reduce_{1,2}stage over the flat
TP-partial ``[T*L latent | T*H shared]`` buffer plus aiter's rmsnorm on the
latent half, in the ``k3_ar_agg_hip`` design: many producer blocks each reduce
a unit of this rank's column slice of one token in AITER's summation order
(the reduced rows are bit-identical), publish it with the unit's sum of
squares in the private tmp region and flag the consumers, which merge the
8 x 2 statistics in a fixed order (identical rstd on every rank) and
normalize their columns. The normed latent matches the unfused path up to
rare 1-ulp bf16 flips.

``full_shared=False`` (scatter-only) leaves the shared output valid only in
this rank's H/8 column slice: enough for the ``ag_agg`` tail, which reads
add_b there only. Shares the private region and the launch parity with
``k3_ar_agg_hip``. See ``jit/csrc/kimi_k3/comm/moe_ar_norm_hip.cuh``.
"""

from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.communication import k3_ar_agg_hip as _agg

WORLD_SIZE = 8
MAX_T = 80
_UNIT = 28  # packs (8 bf16) per producer unit
# AITER CustomAllreduce: 1-stage kernel (rank-0-first sums) below this many
# bytes at world size 8 with full xGMI, 2-stage (owner-first) above
_ONE_STAGE_BYTES = 80 * 1024
_LSPLIT = 2  # latent consumer blocks per token
_HSPLIT = 4  # shared consumer blocks per token (full_shared only)
_NP = 80  # producer blocks (capped at T * units)
_TS = [0]  # instrumentation: device pointer of an int64 [grid, 4] timestamp buffer


@cache_once
def _module():
    from aiter.jit.core import AITER_CSRC_DIR

    return load_jit(
        "k3_moe_ar_norm_hip",
        cuda_files=["kimi_k3/comm/moe_ar_norm_hip.cuh"],
        cuda_wrappers=[("run", "k3_moe_ar_norm::MoeArNorm::run")],
        extra_include_paths=[f"{AITER_CSRC_DIR}/include"],
        extra_cuda_cflags=["-O3"],
    )


def supported(num_tokens: int, latent: int, hidden: int, full_shared: bool = False) -> bool:
    q = 8 * WORLD_SIZE * _UNIT
    if not (1 <= num_tokens <= MAX_T and latent % q == 0 and hidden % q == 0):
        return False
    ul = latent // q
    nu = ul + hidden // q
    split = _LSPLIT + (_HSPLIT if full_shared else 0)
    return (
        ul <= 4
        and num_tokens * nu <= 6 * 80  # producer grid
        and num_tokens * nu * 32 * 16 <= 256 << 10  # region data (per parity)
        and num_tokens * split * WORLD_SIZE * nu * 4 <= 64 << 10  # flags
    )


def usable(ca, num_tokens: int, latent: int, hidden: int, full_shared: bool = False) -> bool:
    """TP8 AITER communicator, enabled, not in the collective-free graph
    warm-up pass, and the AITER all-reduce the fused kernel replaces is its
    1/2-stage one (full xGMI, buffer under AITER's custom-AR ceiling)."""
    return (
        ca is not None
        and not getattr(ca, "disabled", True)
        and hasattr(ca, "_pool")
        and getattr(ca, "world_size", 0) == WORLD_SIZE
        and getattr(ca, "max_size", 0) >= 32 << 20
        and getattr(ca, "fully_connected", True)
        and supported(num_tokens, latent, hidden, full_shared)
        and num_tokens * (latent + hidden) * 2 <= ca.max_size
        and not (ca._IS_CAPTURING and not torch.cuda.is_current_stream_capturing())
    )


def moe_ar_norm(
    ca,
    buf: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_tokens: int,
    latent: int,
    *,
    full_shared: bool = True,
    out: torch.Tensor | None = None,
    shared_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """buf: this rank's flat TP-partial [T*L | T*H] (bf16, contiguous).
    Returns (rmsnorm(AR(latent)) [T, L], AR(shared) [T, H]); with
    full_shared=False only shared[:, rank*H/8:(rank+1)*H/8] is written.
    Graph capture (inside AITER's capture()) registers ``buf``; eager calls
    stage it through AITER's input pool like AITER's all_reduce."""
    T, L = num_tokens, latent
    H = buf.numel() // T - L
    assert buf.is_contiguous() and buf.dtype == torch.bfloat16 and buf.numel() == T * (L + H)
    assert supported(T, L, H, full_shared), (T, L, H, full_shared)
    if out is None:
        out = buf.new_empty(T, L)
    if shared_out is None:
        shared_out = buf.new_empty(T, H)
    if torch.cuda.is_current_stream_capturing() and getattr(
        ca, "enable_register_for_capturing", True
    ):
        assert ca._IS_CAPTURING
        reg_ptr, reg_bytes = 0, 0
    else:
        reg_ptr, reg_bytes = ca._pool["input"].data_ptr, ca._pool["input"].max_size
    nbytes = T * (L + H) * 2
    part = 0 if nbytes < _ONE_STAGE_BYTES else T * (L + H) // 8 // WORLD_SIZE
    _module().run(
        ca._ptr,
        buf,
        weight,
        out,
        shared_out,
        float(eps),
        part,
        1 if full_shared else 0,
        _LSPLIT,
        _HSPLIT,
        _NP,
        reg_ptr,
        reg_bytes,
        _agg._region_off(ca),
        _agg._next_parity(),
        _TS[0],
    )
    return out, shared_out
