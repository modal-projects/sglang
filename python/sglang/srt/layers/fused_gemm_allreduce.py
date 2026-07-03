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
# Dense-layer MLP down_proj (layer 0 on Kimi) fused GEMM+AR; shares the symm C
# buffer with the o_proj wrapper (zero marginal pinned HBM besides flags).
_MLP_ENABLED = os.getenv("SGLANG_FUSED_MLP_AR", "0") == "1"
_SUPPORTED_WORLD_SIZES = (2, 4, 8)

# Marker attribute set on the returned (already reduced) tensor; checked by
# the layer communicator to skip its attention-TP all-reduce.
MARKER_ATTR = "_sglang_fused_oproj_ar_done"


def _round_up(x: int, mult: int) -> int:
    return (x + mult - 1) // mult * mult


# Shared symmetric-memory C buffer pool: one [max_m, n] buffer per
# (group_name, max_m, n, dtype). Safe to time-share across fused-GEMM+AR
# wrappers on the same stream: the kernel's final SM-wise inter-GPU barrier
# guarantees peers' reads/writes of local C for launch N finish before launch
# N ends on this rank, and a peer's launch N+1 sweep spins on ITS OWN flag
# buffer (separate per wrapper) until this rank arrives. Flag buffers must NOT
# be shared (parity/reset protocols are per wrapper).
_SHARED_C = {}


def _get_shared_c_symm(pg_group_name: str, max_m: int, n: int, dtype, device):
    import torch.distributed._symmetric_memory as symm_mem

    key = (pg_group_name, max_m, n, dtype)
    entry = _SHARED_C.get(key)
    if entry is None:
        # inference_mode(False): the first eager forward can run inside
        # torch.inference_mode() (flashinfer autotune dummy runs). Buffers
        # created there would be inference tensors, and downstream torch-level
        # in-place ops on views of them (e.g. the MoE routed-scale+shared add
        # on a returned hidden view) then fail outside InferenceMode
        # ("Inplace update to inference tensor"). Old-campaign lesson
        # (kimi-bench ffe49fe318).
        with torch.inference_mode(False):
            c_symm = symm_mem.empty((max_m, n), dtype=dtype, device=device)
            c_symm.zero_()
            handle = symm_mem.rendezvous(c_symm, group=pg_group_name)
        entry = (c_symm, handle.multicast_ptr)
        _SHARED_C[key] = entry
    return entry


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
        in_dtype: Optional[torch.dtype] = None,
        stage_a: bool = True,
    ):
        """``dtype`` is the output (C) dtype; ``in_dtype`` the A/B operand dtype.

        ``in_dtype=torch.float8_e4m3fn`` selects the fp8 path (per-tensor static
        scales): pass the per-rank ``alpha = input_scale * weight_scale`` to
        ``__call__`` — it is applied to the fp32 accumulator in the epilogue
        BEFORE the bf16 partial enters the two-shot multimem reduce (scales
        differ per rank, so post-AR scaling would be wrong).
        """
        assert n % _TILE_N == 0, f"N={n} must be a multiple of {_TILE_N}"
        assert k % 64 == 0, f"K={k} must be a multiple of 64"
        self.group = group
        self.world_size = group.world_size
        self.rank = group.rank_in_group
        assert self.world_size in _SUPPORTED_WORLD_SIZES
        self.n = n
        self.k = k
        self.dtype = dtype
        self.in_dtype = in_dtype if in_dtype is not None else dtype
        assert self.in_dtype in (torch.bfloat16, torch.float8_e4m3fn)
        self.max_m = _round_up(max_m or _MAX_TOKENS, _TILE_M)
        # stage_a=False skips the persistent A staging buffer (zero pinned A
        # memory): per-call cute views over the caller's tensor instead, which
        # requires M % _TILE_M == 0 (no padding possible) and costs a per-call
        # from_dlpack. Used by the (once-per-forward) dense-MLP site.
        self.stage_a = stage_a

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

        # Vendored flashinfer kernel + runtime per-rank alpha (fp8 static scales).
        from sglang.srt.layers.gemm_allreduce_two_shot_alpha import (
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

        # inference_mode(False) around ALL persistent state allocation: the
        # first eager forward can run inside torch.inference_mode()
        # (flashinfer autotune dummy runs); tensors created there would be
        # inference tensors and later torch-level in-place updates (staging
        # copy_, or MoE in-place adds on returned views) fail outside
        # InferenceMode. Old-campaign lesson (kimi-bench ffe49fe318).
        with torch.inference_mode(False):
            # Persistent input staging buffer (lets us cache per-M cute
            # tensors and pad ragged M up to the tile boundary; the copy is
            # ~10us at M=16K). In fp8 mode this doubles as the activation
            # quant target (saturating cast happens in the staging copy).
            if self.stage_a:
                self._a_staging = torch.zeros(
                    self.max_m, self.k, dtype=self.in_dtype, device=device
                )
            else:
                self._a_staging = None
            # Symmetric-memory output buffer + multicast handle (SHARED across
            # wrappers of the same group/shape — see _get_shared_c_symm).
            self._c_symm, self._c_mc_ptr = _get_shared_c_symm(
                pg.group_name, self.max_m, self.n, self.dtype, device
            )

            # Double-buffered barrier flags: one i32 per CTA tile + one per SM
            # for the kernel's final SM-wise inter-GPU barrier.
            num_sms = torch.cuda.get_device_properties(
                device
            ).multi_processor_count
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
        gemm = PersistentDenseGemmKernel(
            cutlass.Float32,
            True,  # use_2cta_instrs
            (256, 256),  # mma_tiler_mn
            (2, 1),  # cluster_shape_mn
            True,  # use_tma_store
            all_reduce="two_shot",
            sm_version=f"sm_{major}{minor}",
        )
        # The kernel reads rank/world from the default process group; override
        # with the TP group's values (identical on single-node plain TP).
        gemm.num_ranks = self.world_size
        gemm.rank_id = self.rank

        max_active_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(2)
        self._gemm = gemm
        self._max_active_clusters = max_active_clusters
        self.initialized = True
        # Anti-silent-no-op: one explicit engagement line per wrapper (the
        # fallback paths log their reasons; this is the positive signal).
        logger.info(
            "fused GEMM+two-shot-AR ENGAGED: n=%d k=%d in_dtype=%s stage_a=%s "
            "world=%d rank=%d max_m=%d (symm C shared pool, %d MB pinned here)",
            self.n,
            self.k,
            self.in_dtype,
            self.stage_a,
            self.world_size,
            self.rank,
            self.max_m,
            (self._a_staging.nbytes if self._a_staging is not None else 0)
            // (1024 * 1024),
        )

    def _get_b_cute(self, weight: torch.Tensor):
        entry = self._b_cache.get(weight.data_ptr())
        if entry is None:
            assert weight.dtype == self.in_dtype
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

    def _make_a_cute(self, src: torch.Tensor, mpad: int):
        # A: (M, K, L=1) K-major view over `src` (staging buffer or caller
        # tensor in stage_a=False mode).
        a3 = torch.as_strided(src, (mpad, self.k, 1), (self.k, 1, mpad * self.k))
        a_cute = self._from_dlpack(a3, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        )
        return a_cute, a3

    def _get_bundle(self, mpad: int):
        bundle = self._bundles.get(mpad)
        if bundle is None:
            if self.stage_a:
                a_cute, a3 = self._make_a_cute(self._a_staging, mpad)
            else:
                a_cute, a3 = None, None  # built per call from the input tensor
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
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Return ``allreduce(alpha * (x @ weight.T))`` (bf16, shape [M, N]).

        ``alpha`` is this rank's per-tensor static scale product
        (input_scale * weight_scale) on the fp8 path; 1.0 for bf16.
        """
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

        if self.stage_a:
            if x.dtype == self.in_dtype:
                self._a_staging[:m].copy_(x)
            else:
                # fp8 path fed bf16 activations: saturating per-tensor
                # static-1.0 quant folded into the staging copy.
                assert self.in_dtype == torch.float8_e4m3fn
                self._a_staging[:m].copy_(torch.clamp(x, -448.0, 448.0))
        else:
            # No staging: operate on the caller's tensor directly (padding is
            # impossible, so M must already be tile-aligned).
            assert m == mpad, f"stage_a=False needs M % {_TILE_M} == 0, got {m}"
            if x.dtype != self.in_dtype:
                assert self.in_dtype == torch.float8_e4m3fn
                x = torch.clamp(x, -448.0, 448.0).to(self.in_dtype)
            if not x.is_contiguous():
                x = x.contiguous()
            a_cute, _a3 = self._make_a_cute(x, mpad)
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
                alpha,
                c_mc=c_mc_cute,
                barrier_flag=bf_cute,
                barrier_flag_mc=bf_mc_cute,
            )
        self._compiled(
            a_cute,
            b_cute,
            c_cute,
            stream,
            alpha,
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

_WRAPPERS = {}  # (n, k, in_dtype) -> FusedGemmAllReduce | None (None = off)


def fused_oproj_ar_enabled() -> bool:
    return _ENABLED


def _fp8_static_params(linear) -> Optional[tuple]:
    """Detect a per-tensor-static fp8 linear (fp8-static workstream interface).

    Returns ``(weight_nk, alpha)`` where ``weight_nk`` is the e4m3 weight as a
    (N, K) row-major view and ``alpha = float(input_scale * weight_scale)`` is
    this rank's scalar scale product, or None if the layer is not fp8-static.
    """
    try:
        from sglang.srt.layers.quantization.fp8_static_norm_quant import (
            static_fp8_input_scale_of,
        )

        input_scale = static_fp8_input_scale_of(linear)
    except ImportError:
        input_scale = getattr(linear, "input_scale", None)
        weight = getattr(linear, "weight", None)
        if (
            input_scale is None
            or weight is None
            or weight.dtype != torch.float8_e4m3fn
            or input_scale.numel() != 1
        ):
            input_scale = None
    if input_scale is None:
        return None
    weight = linear.weight
    weight_scale = getattr(linear, "weight_scale", None)
    if weight_scale is None or weight_scale.numel() != 1:
        return None
    # fp8-static stores the weight transposed ([K, N], col-major for cublasLt);
    # recover the (N, K) row-major view the kernel wants.
    if weight.dim() != 2:
        return None
    if weight.stride(1) == 1 and weight.is_contiguous():
        weight_nk = weight  # already (N, K) row-major
    elif weight.stride(0) == 1:
        weight_nk = weight.t()  # (K, N) col-major -> (N, K) row-major view
    else:
        return None
    if not weight_nk.is_contiguous():
        return None
    alpha = float(input_scale.item()) * float(weight_scale.item())
    return weight_nk, alpha


_FP8_ALPHA_CACHE = {}  # id(linear) -> (weight_nk, alpha)


def _get_wrapper(
    n: int,
    k: int,
    in_dtype: torch.dtype = torch.bfloat16,
    stage_a: bool = True,
) -> Optional[FusedGemmAllReduce]:
    key = (n, k, in_dtype, stage_a)
    if key in _WRAPPERS:
        return _WRAPPERS[key]
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
            wrapper = FusedGemmAllReduce(
                group, n, k, in_dtype=in_dtype, stage_a=stage_a
            )
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
    _WRAPPERS[key] = wrapper
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
    # Piecewise CUDA graph: the CuteDSL fused GEMM+AR launcher (and the
    # marker-attribute handshake with the communicator) cannot cross a dynamo
    # trace — tracing the engaged branch crashes PCG compile, and a marker set
    # inside a compiled region would be invisible to the traced consumer
    # (silent double all-reduce). Captured extends are <= the PCG token
    # ceiling where fusion is ~breakeven anyway; the true-eager >max-tokens
    # path (16k chunks) keeps the fused win. Trace-stable on this fork:
    # compiled callables run only under is_in_piecewise_cuda_graph()=True.
    from sglang.srt.compilation.piecewise_context_manager import (
        is_in_piecewise_cuda_graph,
    )

    if is_in_piecewise_cuda_graph():
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
        or x.dtype != torch.bfloat16
        or getattr(o_proj, "bias", None) is not None
        or o_proj.reduce_results
    ):
        return None
    # Pick the operand path: bf16 weights, or per-tensor-static fp8 weights
    # (fp8-static workstream; alpha folded into the epilogue pre-reduce).
    alpha = 1.0
    if weight.dtype == torch.bfloat16 and weight.is_contiguous():
        weight_nk = weight
        in_dtype = torch.bfloat16
    elif weight.dtype == torch.float8_e4m3fn:
        cached = _FP8_ALPHA_CACHE.get(id(o_proj))
        if cached is None:
            fp8 = _fp8_static_params(o_proj)
            if fp8 is None:
                return None
            _FP8_ALPHA_CACHE[id(o_proj)] = fp8
            cached = fp8
        weight_nk, alpha = cached
        in_dtype = torch.float8_e4m3fn
    else:
        return None
    n, k = weight_nk.shape
    if x.shape[1] != k:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None

    wrapper = _get_wrapper(n, k, in_dtype)
    if wrapper is None:
        return None
    try:
        out = wrapper(x, weight_nk, alpha=alpha)
    except Exception:
        if wrapper.initialized:
            # Post-init failures are not safely recoverable (peers may already
            # be inside the collective kernel); surface the error.
            raise
        logger.exception(
            "Fused o_proj GEMM+allreduce initialization failed; disabling."
        )
        _WRAPPERS[(n, k, in_dtype, True)] = None
        return None
    setattr(out, MARKER_ATTR, True)
    return out


# ----------------------------------------------------------------------
# dense-MLP down_proj glue (RowParallelLinear with reduce_results=True)
# ----------------------------------------------------------------------


def maybe_fused_mlp_allreduce(down_proj, x: torch.Tensor) -> Optional[torch.Tensor]:
    """Fused replacement for ``down_proj(x)`` INCLUDING its internal TP
    all-reduce (RowParallelLinear.reduce_results path, e.g. the Kimi layer-0
    dense MLP). Returns the reduced [M, N] bf16 output or None to fall back.

    Uses the attention-TP group (gated to equal the full TP group) so the
    symmetric C buffer is shared with the o_proj wrapper — the site costs no
    extra pinned HBM (stage_a=False: per-call A views, M must be tile-aligned).

    Gates must stay rank-symmetric (M is identical across ranks at plain TP).
    """
    if not _MLP_ENABLED:
        return None
    # See maybe_fused_oproj_allreduce: CuteDSL launcher + reduced-output
    # semantics cannot cross a dynamo trace; disengage for compiled forwards.
    from sglang.srt.compilation.piecewise_context_manager import (
        is_in_piecewise_cuda_graph,
    )

    if is_in_piecewise_cuda_graph():
        return None
    if not isinstance(x, torch.Tensor) or x.dim() != 2 or x.dtype != torch.bfloat16:
        return None
    m = x.shape[0]
    if m < _MIN_TOKENS or m > _MAX_TOKENS or m % _TILE_M != 0:
        return None
    if not getattr(down_proj, "reduce_results", False):
        return None
    if getattr(down_proj, "bias", None) is not None:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_world_size
        from sglang.srt.layers.dp_attention import get_attention_tp_group

        # The MLP all-reduce runs over the FULL TP group; reuse the attn-TP
        # group (and its shared symm buffer) only when they coincide.
        if get_tensor_model_parallel_world_size() != get_attention_tp_group().world_size:
            return None
    except Exception:
        return None

    weight = getattr(down_proj, "weight", None)
    if weight is None or not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        return None
    alpha = 1.0
    if weight.dtype == torch.bfloat16 and weight.is_contiguous():
        weight_nk = weight
        in_dtype = torch.bfloat16
    elif weight.dtype == torch.float8_e4m3fn:
        cached = _FP8_ALPHA_CACHE.get(id(down_proj))
        if cached is None:
            fp8 = _fp8_static_params(down_proj)
            if fp8 is None:
                return None
            _FP8_ALPHA_CACHE[id(down_proj)] = fp8
            cached = fp8
        weight_nk, alpha = cached
        in_dtype = torch.float8_e4m3fn
    else:
        return None
    n, k = weight_nk.shape
    if x.shape[1] != k:
        return None

    wrapper = _get_wrapper(n, k, in_dtype, stage_a=False)
    if wrapper is None:
        return None
    try:
        return wrapper(x, weight_nk, alpha=alpha)
    except Exception:
        if wrapper.initialized:
            raise
        logger.exception("Fused MLP GEMM+allreduce initialization failed; disabling.")
        _WRAPPERS[(n, k, in_dtype, False)] = None
        return None
