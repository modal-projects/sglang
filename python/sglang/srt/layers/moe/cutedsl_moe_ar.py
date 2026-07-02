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
"""Fused CuteDSL MoE fc2+finalize+two-shot-multimem-all-reduce glue (Blackwell).

Drives ``Sm100BlockScaledContiguousGroupedGemmFinalizeAllReduceKernel``
(``cutedsl_finalize_allreduce.py``, a copy of flashinfer's CuteDSL fc2
finalize-fusion kernel extended with an overlapped per-N-column two-shot
multimem all-reduce) so that the post-MoE TP all-reduce (~600us/layer at
16k-token prefill on 4xB200) is hidden inside the fc2 kernel.

Enable with ``SGLANG_CUTEDSL_FUSED_AR=1`` (requires
``--moe-runner-backend flashinfer_cutedsl``). The fused path only engages for
eager prefill-scale batches (``num_tokens >= SGLANG_CUTEDSL_FUSED_AR_MIN_TOKENS``,
default 1024 — decode-scale launches are gated off: with us-scale kernels a
rank can run a full launch ahead and the kernel's exact-match-CAS DONE flags
can deadlock; prefill-scale is immune in practice and validated for 600+
hang-free launches), plain TP (no DP attention), TP world size in {2, 4, 8},
a2a backend ``none``, and no TBO/SBO.

Handshake with the model (DeepseekV2MoE.forward_normal):

1. The model checks ``moe_ar_should_engage(num_tokens, ...)`` (all inputs
   rank-symmetric), folds the TP-sharded shared-expert partial output into a
   "seed" and calls ``moe_ar_stash_seed(seed, num_tokens)`` right before
   ``self.experts(...)``.
2. The cutedsl moe_runner fused func pops the stash and, when present, runs
   ``run_fused_ar`` instead of ``CuteDslMoEWrapper.run``: the fc2 output lands
   in a torch symmetric-memory buffer that is pre-loaded with the seed
   (instead of the stock zero-memset, both overlapped with GEMM1 on the aux
   stream), so the in-kernel all-reduce produces
   ``sum_ranks(routed_partial + shared_partial)`` — exactly what the model's
   (shared-add -> tensor_model_parallel_all_reduce) would have produced.
3. The returned tensor carries the ``MARKER_ATTR`` attribute; the model skips
   its shared-add and post-MoE all-reduce when it is set. If the runner fell
   back (no marker), the model uses the seed as the ordinary shared output —
   correctness is preserved in both directions.

Initialization is EAGER and collective: ``maybe_init_fused_ar`` is called from
``ensure_cutedsl_wrapper`` during the first (dummy-run) forward, allocates the
symmetric output + flag buffers, compiles the kernel and performs one real
warmup launch on all TP ranks — so an OOM/compile failure crashes (or cleanly
disables, rank-symmetrically via an all_reduce vote) at startup instead of
mid-serving. Local allocation happens before any collective so a single-rank
OOM cannot strand peers inside a rendezvous.

Per-rank extra memory (Kimi-K2.6 TP=4, max_num_tokens=16384): symmetric fc2
output 16384 x 7168 bf16 = 224MB + flag buffer (1 page) + multicast mappings,
all outside the torch caching allocator. The CuteDslMoEWrapper preallocs
(~292MB: 224MB moe_output + ~66MB gemm1 out/scale + sort buffers) are stock
cutedsl costs, unchanged.

Grid-size invariant: the AR kernel's flag-buffer layout is derived from the
launched grid size. Every fused launch uses the wrapper's constant-size
preallocated gemm1 buffer as fc2's A operand (m = max_permuted rows), so the
persistent grid is always the full device and the layout never moves. The
warmup launch goes through the same path. ``run_fused_ar`` asserts this.

Pinned stock-path tactics (see moe_tactics sweep, 2026-07-01): for
``num_tokens >= 512`` the stock ``CuteDslMoEWrapper.run`` gets
``tactic=(256, ((256,256),(2,1),False), ((256,256),(2,1),True))`` — fc1
(256,256) is 1.26-1.42x over the autotuner's (256,128) pick and fc2 (256,256)
1.26x (the sweep's rasterM winner is not reachable through the stock wrapper,
which drops the raster flag; the fused-AR kernel builds rasterM in directly).
Below 512 tokens ``tactic=None`` keeps the autotuner (fc1 (256,256) regresses
~3us at M=128).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# Marker attribute set on the returned (already reduced) tensor; checked by
# DeepseekV2MoE.forward_normal / FusedMoE.forward to skip the post-MoE TP
# all-reduce (same pattern as _sglang_fused_oproj_ar_done).
MARKER_ATTR = "_sglang_cutedsl_moe_ar_done"

_ENABLED = os.getenv("SGLANG_CUTEDSL_FUSED_AR", "0") == "1"
_MIN_TOKENS = int(os.getenv("SGLANG_CUTEDSL_FUSED_AR_MIN_TOKENS", "1024"))
# Return a view of the symmetric buffer instead of copying out (~37us at 16k).
# Only safe when nothing can enqueue the next MoE layer's buffer zeroing
# before all consumers of this output ran (true for the plain eager prefill
# path; NOT validated with TBO).
_RETURN_VIEW = os.getenv("SGLANG_CUTEDSL_FUSED_AR_VIEW", "0") == "1"
_SUPPORTED_WORLD_SIZES = (2, 4, 8)

# Fused-path kernel config (validated in moe_ar_notes runs 5-10 at mma_n=128;
# mma_n=256 is the tactic-sweep fc2 winner and is validated end-to-end by
# human_tools/moe_ar_integration_bench.py — env knob kept for fallback).
_TILE_SIZE = 256
_SF_VEC_SIZE = 16
_GEMM1_MMA = (256, 256)
_GEMM1_CLUSTER = (2, 1)
_GEMM2_MMA = (256, int(os.getenv("SGLANG_CUTEDSL_FUSED_AR_MMA_N", "256")))
_GEMM2_CLUSTER = (2, 1)
_NUM_AR_WARPS = 4

# Pinned stock-path tactic: (tile_size, gemm1, gemm2) with
# (mma_tiler_mn, cluster_shape_mn, raster_along_m) per GEMM. The stock
# flashinfer runner applies tile/mma/cluster and ignores the raster flags.
_PINNED_TACTIC = (256, ((256, 256), (2, 1), False), ((256, 256), (2, 1), True))
_PIN_ENABLED = os.getenv("SGLANG_CUTEDSL_PIN_TACTIC", "1" if _ENABLED else "0") == "1"
_PIN_MIN_TOKENS = int(os.getenv("SGLANG_CUTEDSL_PIN_TACTIC_MIN_TOKENS", "512"))

_state: Optional["_FusedMoeArState"] = None
_state_failed = False
# (seed_tensor_or_None, num_tokens) stashed by the model right before the
# experts call; popped by the runner fused func. Single-slot: engagement is
# gated off under TBO/SBO so stash/pop cannot interleave across microbatches.
_pending_stash: Optional[tuple] = None
_warned_stash_mismatch = False


def _round_up(x: int, mult: int) -> int:
    return (x + mult - 1) // mult * mult


def pinned_moe_tactic(num_tokens: int) -> Optional[tuple]:
    """Pinned CuteDslMoEWrapper.run tactic for prefill-scale batches."""
    if _PIN_ENABLED and num_tokens >= _PIN_MIN_TOKENS:
        return _PINNED_TACTIC
    return None


def fused_moe_ar_enabled() -> bool:
    return _ENABLED


# ---------------------------------------------------------------------------
# model <-> runner handshake
# ---------------------------------------------------------------------------


def moe_ar_should_engage(
    num_tokens: int,
    should_allreduce_fusion: bool = False,
    use_reduce_scatter: bool = False,
) -> bool:
    """Rank-symmetric gate, evaluated by the model before stashing a seed.

    Callers must only pass rank-symmetric inputs: every rank of the TP group
    must reach the collective fused kernel for the same layers the same
    number of times.
    """
    if _state is None:
        return False
    if should_allreduce_fusion or use_reduce_scatter:
        # Downstream components expect an unreduced output to fuse/replace
        # the all-reduce themselves.
        return False
    if num_tokens < _MIN_TOKENS or num_tokens > _state.max_tokens:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    return True


def moe_ar_stash_seed(seed: Optional[torch.Tensor], num_tokens: int) -> None:
    global _pending_stash
    _pending_stash = (seed, num_tokens)


def moe_ar_take_stash(num_tokens: int) -> Optional[tuple]:
    """Pop the model's stash. Returns (seed,) when the fused path should run."""
    global _pending_stash, _warned_stash_mismatch
    stash = _pending_stash
    _pending_stash = None
    if stash is None or _state is None:
        return None
    seed, stash_tokens = stash
    if stash_tokens != num_tokens:
        if not _warned_stash_mismatch:
            _warned_stash_mismatch = True
            logger.warning(
                "cutedsl fused MoE+AR: discarding stale seed stash "
                "(stash tokens=%s, runner tokens=%s)",
                stash_tokens,
                num_tokens,
            )
        return None
    return (seed,)


def moe_ar_consume_marker(t: Any) -> bool:
    """Read-and-clear the already-reduced marker on a tensor."""
    if getattr(t, MARKER_ATTR, False):
        setattr(t, MARKER_ATTR, False)
        return True
    return False


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


class _FusedMoeArState:
    """Per-TP-group persistent buffers + compiled fused fc2+AR kernel.

    __init__ performs LOCAL work only (allocations, kernel object); the
    collective parts (symmetric-memory rendezvous, compile, warmup launch)
    live in collective_init() so a single-rank local failure can be voted on
    before any rank enters a collective.
    """

    def __init__(self, group, hidden_size: int, max_num_tokens: int, device):
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group
        self.world_size = group.world_size
        self.rank = group.rank_in_group
        self.hidden = hidden_size
        self.device = device
        # padded so any num_tokens <= max_num_tokens can be padded up to a
        # multiple of world_size while staying in-buffer
        self.max_tokens = _round_up(max_num_tokens, 128)

        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.num_sms = num_sms
        # Flag layout (see kernel): [0..7] v0 slots, go [8, 8+nCTA),
        # DONE [8+nCTA, 8+2nCTA), v1 column counters [8+2nCTA, +n_blocks),
        # v1 column flags [+n_blocks), per-CTA epoch slots [+nCTA).
        # nCTA <= num_sms, n_blocks <= hidden/128. +64 slack.
        self.num_flags = 8 + 3 * num_sms + 2 * (hidden_size // 128) + 64

        # ---- local allocations (may OOM -> caught before any collective) ----
        self.out_symm = symm_mem.empty(
            (self.max_tokens, hidden_size), dtype=torch.bfloat16, device=device
        )
        self.out_symm.zero_()
        self.flags = symm_mem.empty(
            (self.num_flags,), dtype=torch.int32, device=device
        )
        self.flags.zero_()

        self.aux_stream = torch.cuda.Stream(device=device)
        self.main_event = torch.cuda.Event()
        self.memset_event = torch.cuda.Event()

        self._out_mc_ptr: Optional[int] = None
        self._flags_mc_ptr: Optional[int] = None
        self._kernel = None
        self._compiled = None
        self._max_active_clusters = None
        self._m_expected: Optional[int] = None

    # ---- collective phase ----

    def collective_init(self) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        pg = self.group.device_group
        out_handle = symm_mem.rendezvous(self.out_symm, group=pg.group_name)
        self._out_mc_ptr = out_handle.multicast_ptr
        flag_handle = symm_mem.rendezvous(self.flags, group=pg.group_name)
        self._flags_mc_ptr = flag_handle.multicast_ptr
        if not self._out_mc_ptr or not self._flags_mc_ptr:
            raise RuntimeError(
                "symmetric memory multicast unavailable "
                f"(out_mc={self._out_mc_ptr}, flags_mc={self._flags_mc_ptr})"
            )

        from flashinfer.cute_dsl.utils import get_max_active_clusters

        from sglang.srt.layers.moe.cutedsl_finalize_allreduce import (
            Sm100BlockScaledContiguousGroupedGemmFinalizeAllReduceKernel,
        )

        kernel = Sm100BlockScaledContiguousGroupedGemmFinalizeAllReduceKernel(
            sf_vec_size=_SF_VEC_SIZE,
            mma_tiler_mn=_GEMM2_MMA,
            cluster_shape_mn=_GEMM2_CLUSTER,
            use_blkred=True,
            raster_along_m=True,
            enable_pdl=True,
            all_reduce="two_shot_overlap",
            num_ar_warps=_NUM_AR_WARPS,
            v1_reg_alloc=True,
        )
        # the ctor reads the default process group; pin to the TP group
        kernel.num_ranks = self.world_size
        kernel.rank_id = self.rank
        self._kernel = kernel
        self._max_active_clusters = get_max_active_clusters(
            _GEMM2_CLUSTER[0] * _GEMM2_CLUSTER[1]
        )

    # ---- fc2 + finalize + all-reduce launch ----

    def launch_fc2_ar(
        self,
        a: torch.Tensor,
        a_sf: torch.Tensor,
        b: torch.Tensor,
        b_sf: torch.Tensor,
        alpha: torch.Tensor,
        tile_idx_to_expert_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        token_final_scales: torch.Tensor,
        num_tokens_pad: int,
    ) -> None:
        import cutlass
        import cutlass.cute as cute
        from flashinfer.cute_dsl.utils import make_ptr

        try:
            from cuda.bindings import driver as cuda_driver
        except ImportError:
            from cuda import cuda as cuda_driver

        m = a.shape[0]
        k = a.shape[1] * 2  # fp4 packed
        n = b.shape[1]
        num_experts = b.shape[0]
        top_k = token_final_scales.shape[1]
        assert n == self.hidden
        assert num_tokens_pad % self.world_size == 0
        assert num_tokens_pad <= self.max_tokens
        # The flag-buffer layout is a function of the launched grid size,
        # which is a function of m (see module docstring). All launches on
        # one flag buffer must use the same m.
        if self._m_expected is None:
            self._m_expected = m
        assert m == self._m_expected, (
            f"fused MoE+AR requires a constant fc2 A-buffer size "
            f"(got m={m}, expected {self._m_expected})"
        )

        def _i32ptr(t):
            return make_ptr(
                cutlass.Int32, t.data_ptr(), cute.AddressSpace.gmem
            )

        args = [
            make_ptr(
                cutlass.Float4E2M1FN, a.data_ptr(), cute.AddressSpace.gmem,
                assumed_align=32,
            ),
            make_ptr(
                cutlass.Float4E2M1FN, b.data_ptr(), cute.AddressSpace.gmem,
                assumed_align=32,
            ),
            make_ptr(
                cutlass.Float8E4M3FN, a_sf.data_ptr(), cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float8E4M3FN, b_sf.data_ptr(), cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.BFloat16, self.out_symm.data_ptr(),
                cute.AddressSpace.gmem, assumed_align=32,
            ),
            make_ptr(cutlass.Float32, alpha.data_ptr(), cute.AddressSpace.gmem),
            _i32ptr(tile_idx_to_expert_idx),
            _i32ptr(tile_idx_to_mn_limit),
            _i32ptr(permuted_idx_to_expanded_idx),
            _i32ptr(num_non_exiting_tiles),
            make_ptr(
                cutlass.Float32, token_final_scales.data_ptr(),
                cute.AddressSpace.gmem, assumed_align=16,
            ),
            m,
            n,
            k,
            num_experts,
            num_tokens_pad,
            top_k,
        ]
        extra = dict(
            c_mc_ptr=make_ptr(
                cutlass.BFloat16, self._out_mc_ptr, cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            barrier_flag_ptr=make_ptr(
                cutlass.Int32, self.flags.data_ptr(), cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            barrier_flag_mc_ptr=make_ptr(
                cutlass.Int32, self._flags_mc_ptr, cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            num_flags=self.num_flags,
        )
        stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
        if self._compiled is None:
            self._compiled = cute.compile(
                self._kernel.wrapper,
                *args,
                tile_size=_TILE_SIZE,
                scaling_vector_size=_SF_VEC_SIZE,
                max_active_clusters=self._max_active_clusters,
                stream=stream,
                **extra,
            )
        self._compiled(*args, stream=stream, **extra)


# ---------------------------------------------------------------------------
# initialization (called from ensure_cutedsl_wrapper)
# ---------------------------------------------------------------------------


def _vote_all_ranks_ok(ok: bool, group) -> bool:
    """AND-reduce a local success flag across the TP group.

    Keeps enable/disable decisions rank-symmetric: a fused kernel launched by
    a subset of ranks would spin forever.
    """
    import torch.distributed as dist

    flag = torch.tensor(
        [1 if ok else 0], dtype=torch.int32, device=torch.cuda.current_device()
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group.device_group)
    return bool(int(flag.item()) == 1)


def maybe_init_fused_ar(layer: torch.nn.Module) -> None:
    """Eagerly set up the fused MoE+AR path (buffers, compile, warmup launch).

    Called from ensure_cutedsl_wrapper on every rank during the first
    (dummy-run) forward — i.e. in TP lockstep. Any failure disables the fused
    path permanently and rank-symmetrically; the stock cutedsl path is
    untouched.
    """
    global _state, _state_failed
    if not _ENABLED or _state is not None or _state_failed:
        return
    if torch.cuda.is_current_stream_capturing():
        return  # defer to the next eager forward (rank-symmetric)

    # --- deterministic (identical on all ranks) config gates: no vote needed
    try:
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.layers.moe.utils import (
            get_moe_a2a_backend,
            is_sbo_enabled,
            is_tbo_enabled,
        )

        def get_attention_dp_size() -> int:
            # DP attention is initialized before the first forward in a real
            # server; if it was never initialized (e.g. layer-level benches),
            # there is no DP.
            try:
                from sglang.srt.layers.dp_attention import (
                    get_attention_dp_size as _dp_size,
                )

                return _dp_size()
            except Exception:
                return 1

        wrapper = getattr(layer, "_cutedsl_wrapper", None)
        group = get_tp_group()
        major, _ = torch.cuda.get_device_capability()
        reason = None
        if major != 10:
            reason = f"sm major {major} != 10"
        elif group.world_size not in _SUPPORTED_WORLD_SIZES:
            reason = f"tp={group.world_size} unsupported"
        elif get_attention_dp_size() != 1:
            reason = "DP attention enabled"
        elif not get_moe_a2a_backend().is_none():
            reason = "a2a backend is not none"
        elif is_tbo_enabled() or is_sbo_enabled():
            reason = "TBO/SBO enabled"
        elif wrapper is None or not wrapper.use_cuda_graph:
            # the constant-size preallocated gemm1 buffer is what keeps the
            # fused kernel's grid (and thus flag layout) constant
            reason = "CuteDslMoEWrapper prealloc buffers disabled"
        elif layer.hidden_size % 128 != 0:
            reason = f"hidden_size={layer.hidden_size} not a multiple of 128"
        elif layer.moe_runner_config.params_dtype != torch.bfloat16:
            reason = f"output dtype {layer.moe_runner_config.params_dtype}"
        if reason is not None:
            _state_failed = True
            logger.info(
                "SGLANG_CUTEDSL_FUSED_AR=1 but unsupported (%s); "
                "using the stock cutedsl path.",
                reason,
            )
            return
    except Exception:
        _state_failed = True
        logger.exception("fused MoE+AR gate evaluation failed; disabling.")
        return

    # --- phase 1: local allocations (vote before any collective) ---
    state = None
    ok = False
    try:
        # inference_mode(False): the first eager forward can run inside
        # torch.inference_mode() (flashinfer autotune / PCG warmup dummy
        # runs). State tensors created there would be inference tensors and
        # the per-forward active.copy_(seed) in _run_fused_ar_impl then
        # fails outside InferenceMode ("Inplace update to inference tensor").
        with torch.inference_mode(False):
            state = _FusedMoeArState(
                group=group,
                hidden_size=layer.hidden_size,
                max_num_tokens=wrapper.max_num_tokens,
                device=torch.device(torch.cuda.current_device()),
            )
        ok = True
    except Exception:
        logger.exception("fused MoE+AR local allocation failed; disabling.")
    if not _vote_all_ranks_ok(ok, group):
        _state_failed = True
        if ok:
            logger.warning(
                "fused MoE+AR disabled: local allocation failed on a peer rank."
            )
        _free_state(state)
        return

    # --- phase 2: collective init + compile + warmup launch ---
    ok = False
    try:
        state.collective_init()
        _warmup(state, layer, wrapper)
        ok = True
    except Exception:
        logger.exception("fused MoE+AR collective init/warmup failed; disabling.")
    if not _vote_all_ranks_ok(ok, group):
        _state_failed = True
        if ok:
            logger.warning(
                "fused MoE+AR disabled: collective init failed on a peer rank."
            )
        # NOTE: buffers that completed a rendezvous are leaked on purpose —
        # peers may hold imported mappings.
        return

    _state = state
    logger.info(
        "fused MoE+AR enabled: tp=%d hidden=%d max_tokens=%d "
        "gemm2_mma=%s symm_out=%.0fMB flags=%d ints",
        state.world_size,
        state.hidden,
        state.max_tokens,
        _GEMM2_MMA,
        state.max_tokens * state.hidden * 2 / 1e6,
        state.num_flags,
    )


def _free_state(state: Optional[_FusedMoeArState]) -> None:
    if state is not None:
        state.out_symm = None
        state.flags = None


def _warmup(state: _FusedMoeArState, layer, wrapper) -> None:
    """One real fused launch through run_fused_ar on synthetic inputs.

    Exercises rendezvous'd buffers, the compile, and the full cross-rank flag
    protocol at startup. Inputs (routing especially) are seeded identically on
    all ranks; the actual values are irrelevant.
    """
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    device = state.device
    num_tokens = max(_MIN_TOKENS, 1024)
    cpu_gen = torch.Generator().manual_seed(20260701)
    x = (
        0.01 * torch.randn(num_tokens, layer.hidden_size, generator=cpu_gen)
    ).to(device=device, dtype=torch.bfloat16)
    topk_ids = torch.randint(
        0,
        layer.num_experts,
        (num_tokens, wrapper.top_k),
        dtype=torch.int32,
        generator=cpu_gen,
    ).to(device)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, wrapper.top_k, generator=cpu_gen), dim=-1
    ).to(device=device, dtype=torch.float32)

    w1_alpha, fc2_input_scale, w2_alpha = layer._cutedsl_scales
    x_fp4, x_sf = fp4_quantize(
        x, layer._cutedsl_input_scale, sf_vec_size=_SF_VEC_SIZE,
        is_sf_swizzled_layout=False,
    )
    out = _run_fused_ar_impl(
        state=state,
        wrapper=wrapper,
        x_fp4=x_fp4,
        x_sf=x_sf,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_weight=layer.w13_weight,
        w13_weight_sf=getattr(
            layer, "w13_blockscale_mma", layer.w13_blockscale_swizzled
        ),
        w1_alpha=w1_alpha,
        a2_scale=fc2_input_scale,
        w2_weight=layer.w2_weight,
        w2_weight_sf=getattr(
            layer, "w2_blockscale_mma", layer.w2_blockscale_swizzled
        ),
        w2_alpha=w2_alpha,
        seed=None,
    )
    torch.cuda.synchronize()
    if not torch.isfinite(out.float().sum()):
        raise RuntimeError("fused MoE+AR warmup produced non-finite output")


# ---------------------------------------------------------------------------
# fused execution (runner side)
# ---------------------------------------------------------------------------


def run_fused_ar(
    quant_info,
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    seed: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fused MoE forward (moe_sort + GEMM1 + fc2/finalize/all-reduce).

    Mirrors flashinfer's _moe_core_impl with the pinned prefill tactics and
    the AR kernel as fc2. The returned tensor is already reduced across the
    TP group and carries MARKER_ATTR.

    Post-init failures are raised (peers may already be inside the collective
    kernel; silently falling back would desynchronize the TP group).
    """
    assert _state is not None
    out = _run_fused_ar_impl(
        state=_state,
        wrapper=quant_info.wrapper,
        x_fp4=x_fp4,
        x_sf=x_sf,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_weight=quant_info.w13_weight,
        w13_weight_sf=quant_info.w13_weight_sf,
        w1_alpha=quant_info.w1_alpha,
        a2_scale=quant_info.a2_scale,
        w2_weight=quant_info.w2_weight,
        w2_weight_sf=quant_info.w2_weight_sf,
        w2_alpha=quant_info.w2_alpha,
        seed=seed,
    )
    setattr(out, MARKER_ATTR, True)
    return out


def _run_fused_ar_impl(
    state: _FusedMoeArState,
    wrapper,
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    a2_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    seed: Optional[torch.Tensor],
) -> torch.Tensor:
    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion import (  # noqa: E501
        blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_nvfp4,
    )
    from flashinfer.fused_moe.cute_dsl.moe_utils import (
        moe_output_memset_inplace,
        moe_sort,
    )

    num_tokens = topk_ids.shape[0]
    pad = _round_up(num_tokens, state.world_size)
    assert pad <= state.max_tokens
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.to(torch.float32)

    # Step 1: moe_sort at the pinned tile size, into the wrapper's
    # constant-size preallocated buffers (=> constant fc2 m / grid size).
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        _expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        _total_num_padded_tokens,
        num_non_exiting_tiles,
    ) = moe_sort(
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        num_experts=wrapper.num_experts,
        top_k=wrapper.top_k,
        local_expert_offset=wrapper.local_expert_offset,
        num_local_experts=wrapper.num_local_experts,
        tile_tokens_dim=_TILE_SIZE,
        **(wrapper._moe_sort_buffers or {}),
    )

    active = state.out_symm[:num_tokens]
    state.main_event.record()
    active.record_stream(state.aux_stream)
    if seed is not None:
        seed.record_stream(state.aux_stream)

    # Step 2: GEMM1 + SwiGLU (pinned tactic), overlapped with the aux-stream
    # seed-copy/zero of the output slice below.
    intermediate, intermediate_sf = (
        blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_nvfp4(
            a=x_fp4,
            b=w13_weight,
            a_scale=x_sf,
            b_scale=w13_weight_sf,
            alpha=w1_alpha,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            token_id_mapping=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            out=wrapper._gemm1_output,
            out_scale=wrapper._gemm1_output_scale,
            global_scale=a2_scale,
            topk=wrapper.top_k,
            c_dtype="float4_e2m1fn",
            mma_tiler_mn=_GEMM1_MMA,
            cluster_shape_mn=_GEMM1_CLUSTER,
            enable_pdl=wrapper.enable_pdl,
        )
    )

    # Step 3: pre-load the active output slice with the shared-expert partial
    # (or zero it): the fc2 epilogue scatter-adds routed contributions on top
    # and the in-kernel all-reduce then sums (routed_r + shared_r) over ranks.
    # Padded tail rows [num_tokens, pad) are never scattered to and never
    # read; the AR sweep reduces garbage into them, which is harmless.
    with torch.cuda.stream(state.aux_stream):
        state.main_event.wait()
        if seed is not None:
            active.copy_(seed)
        else:
            moe_output_memset_inplace(active)
        state.memset_event.record()
    state.memset_event.wait()

    # Step 4: fc2 + finalize + fused two-shot all-reduce.
    state.launch_fc2_ar(
        a=intermediate,
        a_sf=intermediate_sf,
        b=w2_weight,
        b_sf=w2_weight_sf,
        alpha=w2_alpha,
        tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        token_final_scales=topk_weights,
        num_tokens_pad=pad,
    )

    if _RETURN_VIEW:
        return active
    out = torch.empty(
        (num_tokens, state.hidden), dtype=torch.bfloat16, device=x_fp4.device
    )
    out.copy_(active)
    return out
