# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fused row-parallel GEMM + two-shot multimem all-reduce (Blackwell/CuteDSL).

Wraps flashinfer's ``PersistentDenseGemmKernel(all_reduce="two_shot")`` which
fuses a dense bf16 GEMM with a per-tile multimem two-shot all-reduce over NVLink
multicast (torch symmetric memory). Used to replace the (o_proj matmul ->
communicator all-reduce) pair on the DeepseekV2/Kimi attention prefill path,
overlapping the TP all-reduce with the GEMM epilogue.

Enable with ``SGLANG_FUSED_OPROJ_AR=1``. The fused path only engages for
eager EXTEND/prefill batches (decode / target-verify / draft-extend run inside
CUDA graphs and are untouched) with ``num_tokens`` in
``[SGLANG_FUSED_OPROJ_AR_MIN_TOKENS, SGLANG_FUSED_OPROJ_AR_MAX_TOKENS]``
(default [1024, 16384]), plain TP (no DP attention, no scattered attn input),
attention-TP world size in {2, 4, 8}, and an unquantized bf16 o_proj.

The wrapper output is already reduced across the attention-TP group; the
returned tensor carries the ``_sglang_fused_oproj_ar_done`` marker attribute
which ``CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual``
checks to skip its all-reduce (mirroring the ``_sglang_needs_allreduce_fusion``
marker pattern).

Implementation notes:

- One shared instance per (N, K) shape serves all layers; per-layer weights are
  passed as runtime tensor args (the compiled artifact has fully dynamic
  layouts, so a single ``cute.compile`` artifact serves varying M).
- Input activations are staged into a persistent buffer padded up to the CTA
  tile M (128) so per-M cute tensor views can be cached (bounded by
  max_m / 128 entries) and ragged M is handled without recompiles. The GEMM
  output lands in a symmetric-memory buffer sized [max_m, N]; padded tail rows
  receive garbage partial sums but stay inside the buffer and are never read.
- Barrier flags are double-buffered (parity-toggled per call). All flags return
  to zero after each launch, but a rank that is one kernel ahead may arrive on
  flags of the *next* launch before a lagging rank has finished the CAS-reset
  of its final barrier when the flag ranges of two consecutive launches with
  different M overlap; alternating buffers puts one full kernel completion on
  every rank between reuses, which closes that race.
- By default the reduced output is copied out of the symmetric buffer
  (~10us at M=16K vs ~600us kernel). ``SGLANG_FUSED_OPROJ_AR_VIEW=1`` returns
  a view of the symmetric buffer instead, which is only safe when nothing
  (e.g. two-batch-overlap's interleaved microbatch) can enqueue the next fused
  call before all consumers of the previous output.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# CTA tile shape of the (256, 256) MMA tiler with 2-CTA instructions.
_TILE_M = 128
_TILE_N = 256

_ENABLED = os.getenv("SGLANG_FUSED_OPROJ_AR", "0") == "1"
_MIN_TOKENS = int(os.getenv("SGLANG_FUSED_OPROJ_AR_MIN_TOKENS", "1024"))
_MAX_TOKENS = int(os.getenv("SGLANG_FUSED_OPROJ_AR_MAX_TOKENS", "16384"))
_RETURN_VIEW = os.getenv("SGLANG_FUSED_OPROJ_AR_VIEW", "0") == "1"
_SUPPORTED_WORLD_SIZES = (2, 4, 8)

# Marker attribute set on the returned (already reduced) tensor; checked by
# the layer communicator to skip its attention-TP all-reduce.
MARKER_ATTR = "_sglang_fused_oproj_ar_done"


def _round_up(x: int, mult: int) -> int:
    return (x + mult - 1) // mult * mult


class FusedGemmAllReduce:
    """Fused ``allreduce(x @ weight.T)`` over one TP group at fixed (N, K).

    Construction is cheap; the first ``__call__`` performs collective setup
    (symmetric-memory rendezvous + ``cute.compile``) and therefore must be
    reached by all ranks of ``group`` in lockstep — callers must gate on
    rank-symmetric conditions only.
    """

    def __init__(
        self,
        group,
        n: int,
        k: int,
        dtype: torch.dtype = torch.bfloat16,
        max_m: int = None,
    ):
        assert n % _TILE_N == 0, f"N={n} must be a multiple of {_TILE_N}"
        assert k % 64 == 0, f"K={k} must be a multiple of 64"
        self.group = group
        self.world_size = group.world_size
        self.rank = group.rank_in_group
        assert self.world_size in _SUPPORTED_WORLD_SIZES
        self.n = n
        self.k = k
        self.dtype = dtype
        self.max_m = _round_up(max_m or _MAX_TOKENS, _TILE_M)

        self.initialized = False
        self._compiled = None
        self._b_cache = {}  # weight data_ptr -> (cute tensor, torch 3d view)
        self._bundles = {}  # padded M -> (a, c, c_mc cute tensors + torch refs)
        self._parity = 0

    # ------------------------------------------------------------------
    # lazy (collective) initialization
    # ------------------------------------------------------------------

    def _lazy_init(self, device: torch.device) -> None:
        import cutlass
        import cutlass.torch as cutlass_torch
        import cutlass.utils as cutlass_utils
        import torch.distributed._symmetric_memory as symm_mem
        from cutlass import cute
        from cutlass.cute.runtime import from_dlpack
        from flashinfer.cute_dsl.gemm_allreduce_two_shot import (
            PersistentDenseGemmKernel,
        )

        try:
            from cuda.bindings import driver as cuda_driver
        except ImportError:
            from cuda import cuda as cuda_driver

        self._cutlass_torch = cutlass_torch
        self._from_dlpack = from_dlpack
        self._cuda_driver = cuda_driver
        pg = self.group.device_group

        # Persistent input staging buffer (lets us cache per-M cute tensors and
        # pad ragged M up to the tile boundary; the copy is ~10us at M=16K).
        self._a_staging = torch.zeros(
            self.max_m, self.k, dtype=self.dtype, device=device
        )
        # Symmetric-memory output buffer + multicast handle.
        self._c_symm = symm_mem.empty(
            (self.max_m, self.n), dtype=self.dtype, device=device
        )
        self._c_symm.zero_()
        c_handle = symm_mem.rendezvous(self._c_symm, group=pg.group_name)
        self._c_mc_ptr = c_handle.multicast_ptr

        # Double-buffered barrier flags: one i32 per CTA tile + one per SM for
        # the kernel's final SM-wise inter-GPU barrier.
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        num_tiles_max = (self.max_m // _TILE_M) * (self.n // _TILE_N)
        self._flag_refs = []  # keep torch tensors alive
        self._flag_pairs = []
        for _ in range(2):
            bf = symm_mem.empty(
                (num_tiles_max + num_sms,), dtype=torch.int32, device=device
            )
            bf.zero_()
            handle = symm_mem.rendezvous(bf, group=pg.group_name)
            bf_mc_torch = cutlass_torch.as_tensor(
                handle.multicast_ptr, bf.shape, bf.dtype
            )
            bf_cute = from_dlpack(bf).mark_layout_dynamic()
            bf_mc_cute = from_dlpack(bf_mc_torch).mark_layout_dynamic()
            self._flag_refs.append((bf, bf_mc_torch))
            self._flag_pairs.append((bf_cute, bf_mc_cute))

        major, minor = torch.cuda.get_device_capability(device)
        try:
            gemm = PersistentDenseGemmKernel(
                cutlass.Float32,
                True,  # use_2cta_instrs
                (256, 256),  # mma_tiler_mn
                (2, 1),  # cluster_shape_mn
                True,  # use_tma_store
                all_reduce="two_shot",
                sm_version=f"sm_{major}{minor}",
            )
        except TypeError:  # older flashinfer without sm_version kwarg
            gemm = PersistentDenseGemmKernel(
                cutlass.Float32,
                True,
                (256, 256),
                (2, 1),
                True,
                all_reduce="two_shot",
            )
        # The kernel reads rank/world from the default process group; override
        # with the TP group's values (identical on single-node plain TP).
        gemm.num_ranks = self.world_size
        gemm.rank_id = self.rank

        max_active_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(2)
        self._gemm = gemm
        self._max_active_clusters = max_active_clusters
        self.initialized = True

    def _get_b_cute(self, weight: torch.Tensor):
        entry = self._b_cache.get(weight.data_ptr())
        if entry is None:
            assert weight.shape == (self.n, self.k) and weight.stride(1) == 1
            # (N, K, L=1) view, K-major, mirroring cutlass_torch.matrix().
            w3 = torch.as_strided(
                weight, (self.n, self.k, 1), (weight.stride(0), 1, self.n * self.k)
            )
            b_cute = self._from_dlpack(w3, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            entry = (b_cute, w3)
            self._b_cache[weight.data_ptr()] = entry
        return entry[0]

    def _get_bundle(self, mpad: int):
        bundle = self._bundles.get(mpad)
        if bundle is None:
            # A: (M, K, L=1) K-major view of the staging buffer.
            a3 = torch.as_strided(
                self._a_staging, (mpad, self.k, 1), (self.k, 1, mpad * self.k)
            )
            a_cute = self._from_dlpack(a3, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            # C: (M, N, L=1) N-major view of the symmetric buffer (+ multicast).
            c3 = torch.as_strided(self._c_symm, (mpad, self.n, 1), (self.n, 1, 1))
            c_cute = self._from_dlpack(c3, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            c_mc_torch = self._cutlass_torch.as_tensor(
                self._c_mc_ptr, (mpad, self.n, 1), self.dtype
            )
            c_mc_cute = self._from_dlpack(
                c_mc_torch, assumed_align=16
            ).mark_layout_dynamic(leading_dim=1)
            bundle = (a_cute, c_cute, c_mc_cute, a3, c3, c_mc_torch)
            self._bundles[mpad] = bundle
        return bundle

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        return_view: Optional[bool] = None,
    ) -> torch.Tensor:
        """Return ``allreduce(x @ weight.T)`` (bf16, shape [M, N])."""
        assert x.dim() == 2 and x.shape[1] == self.k
        m = x.shape[0]
        assert 0 < m <= self.max_m
        mpad = _round_up(m, _TILE_M)

        if not self.initialized:
            self._lazy_init(x.device)

        b_cute = self._get_b_cute(weight)
        a_cute, c_cute, c_mc_cute = self._get_bundle(mpad)[:3]
        bf_cute, bf_mc_cute = self._flag_pairs[self._parity]
        self._parity ^= 1

        self._a_staging[:m].copy_(x)
        stream = self._cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
        if self._compiled is None:
            from cutlass import cute

            self._compiled = cute.compile(
                self._gemm,
                a_cute,
                b_cute,
                c_cute,
                self._max_active_clusters,
                stream,
                c_mc=c_mc_cute,
                barrier_flag=bf_cute,
                barrier_flag_mc=bf_mc_cute,
            )
        self._compiled(
            a_cute,
            b_cute,
            c_cute,
            stream,
            c_mc=c_mc_cute,
            barrier_flag=bf_cute,
            barrier_flag_mc=bf_mc_cute,
        )

        out2d = self._c_symm[:m]
        if return_view if return_view is not None else _RETURN_VIEW:
            return out2d
        out = torch.empty((m, self.n), dtype=self.dtype, device=x.device)
        out.copy_(out2d)
        return out


# ----------------------------------------------------------------------
# o_proj glue (model-side entry point)
# ----------------------------------------------------------------------

_WRAPPERS = {}  # (n, k) -> FusedGemmAllReduce | None (None = permanently off)


def fused_oproj_ar_enabled() -> bool:
    return _ENABLED


def _get_wrapper(n: int, k: int) -> Optional[FusedGemmAllReduce]:
    if (n, k) in _WRAPPERS:
        return _WRAPPERS[(n, k)]
    wrapper = None
    try:
        from sglang.srt.layers.communicator import get_attn_tp_context
        from sglang.srt.layers.dp_attention import (
            get_attention_dp_size,
            get_attention_tp_group,
        )

        major, _ = torch.cuda.get_device_capability()
        group = get_attention_tp_group()
        if (
            major == 10
            and get_attention_dp_size() == 1
            and not get_attn_tp_context().input_scattered
            and group.world_size in _SUPPORTED_WORLD_SIZES
            and n % _TILE_N == 0
            and k % 64 == 0
        ):
            wrapper = FusedGemmAllReduce(group, n, k)
        else:
            logger.info(
                "SGLANG_FUSED_OPROJ_AR=1 but the fused o_proj GEMM+allreduce "
                "path is unsupported for this configuration; falling back "
                "(sm major=%s, attn_dp=%s, attn_tp=%s, n=%s, k=%s)",
                major,
                get_attention_dp_size(),
                group.world_size,
                n,
                k,
            )
    except Exception:
        logger.exception(
            "Failed to set up fused o_proj GEMM+allreduce; falling back."
        )
    _WRAPPERS[(n, k)] = wrapper
    return wrapper


def maybe_fused_oproj_allreduce(
    o_proj, x: torch.Tensor, forward_batch
) -> Optional[torch.Tensor]:
    """Fused replacement for ``o_proj(x)`` + attention-TP all-reduce.

    Returns the TP-reduced output (with the skip-allreduce marker attribute
    set) when the fused path applies, else None (caller falls back to the
    regular o_proj matmul + communicator all-reduce).

    All gates below must be rank-symmetric across the attention-TP group:
    the first successful call performs a collective initialization, and every
    call launches a collective kernel.
    """
    if not _ENABLED:
        return None
    fm = forward_batch.forward_mode
    if (
        not fm.is_extend()
        or fm.is_target_verify()
        or fm.is_draft_extend(include_v2=True)
    ):
        return None
    if not isinstance(x, torch.Tensor) or x.dim() != 2:
        return None
    m = x.shape[0]
    if m < _MIN_TOKENS or m > _MAX_TOKENS:
        return None
    weight = getattr(o_proj, "weight", None)
    if (
        weight is None
        or not isinstance(weight, torch.Tensor)
        or weight.dim() != 2
        or not weight.is_contiguous()
        or weight.dtype != torch.bfloat16
        or x.dtype != torch.bfloat16
        or getattr(o_proj, "bias", None) is not None
        or o_proj.reduce_results
    ):
        return None
    n, k = weight.shape
    if x.shape[1] != k:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None

    wrapper = _get_wrapper(n, k)
    if wrapper is None:
        return None
    try:
        out = wrapper(x, weight)
    except Exception:
        if wrapper.initialized:
            # Post-init failures are not safely recoverable (peers may already
            # be inside the collective kernel); surface the error.
            raise
        logger.exception(
            "Fused o_proj GEMM+allreduce initialization failed; disabling."
        )
        _WRAPPERS[(n, k)] = None
        return None
    setattr(out, MARKER_ATTR, True)
    return out
