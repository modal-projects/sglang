from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


# v4 (full one-shot) is out-of-place and drops the exit barrier, so it needs a
# double-buffered input. We carve three regions out of the tail of the enlarged
# comm.buffer -- two rotating input buffers (A/B) + one output -- so a region is
# only reused two ARs later, separated by the intervening AR's entry barrier
# (capture-safe: the A/B alternation bakes into the graph). Region size covers a
# few decode rows at hidden=6144; v4 only fires for num_tokens <= 2.
#
# SAFETY INVARIANT: the reuse-distance-2 argument requires the v4 AR sequence to
# alternate A,B,A,B *globally* -- across forwards, and across graph replays. That
# holds because every forward issues an even number of v4 ARs (attn + MLP per
# layer), so each captured graph starts and ends on opposite regions and replays
# stay aligned with each other and with eager forwards. If a forward could ever
# issue an ODD number of v4 ARs (e.g. a layer taking a reduce-scatter path at
# num_tokens <= 2), a replay boundary would put the same region in consecutive
# ARs and a lagging peer could still be multicast-reading it -- audit this before
# changing which layers all-reduce at decode.
_INKLING_AR_V4_REGION = 16 * 6144  # elems; 16B-aligned (mult of 8)

# v5 (push one-shot) needs a per-rank staging slot on every GPU: two rotating
# staging buffers of world*_INKLING_AR_V5_REGION elems (A/B, same reuse-distance-2
# argument and SAFETY INVARIANT as v4 above -- v5's single barrier plays the
# entry barrier's role) plus one local output region. Sized to the band where
# v5 beats torch multimem (<=96 rows at hidden=6144 on B200/TP4), plus the
# fused target-verify chain (bs*draft_token_num <= 144 rows at bs=16, Q=9).
_INKLING_AR_V5_REGION = 160 * 6144  # elems; 16B-aligned (mult of 8)

# The custom kernels reduce one 16B vector (8 bf16 elems) at a time and their
# validate() rejects a num_items that isn't a multiple of this. torch symm-mem
# only enforces 4B alignment, so a non-vector bf16 size (e.g. [1, 2]) must fall
# back to torch multimem instead of hitting the kernel's hard check. (For Inkling
# proper this never bites -- hidden=6144 is a multiple of 8 -- but the utility
# is general.)
_INKLING_AR_VEC = 8

# World sizes with torch-multimem NVLink support; the symm-mem fast path (and
# with it the custom kernels) is only taken for these.
_INKLING_AR_WORLD_SIZES = (4, 6, 8)


class _InklingArResources(msgspec.Struct):
    """Per-group custom-AR resources: barrier flags/state + comm.buffer peer and
    multicast pointers + v4 double-buffer region offsets + rotation index."""

    rank: int
    world: int
    buffer_ptrs_dev: int
    multicast_ptr: int
    flag_ptrs_dev: int
    state_ptr: int
    v4_in: tuple[int, int]  # (A, B) input region starts (elems)
    v4_out: int  # output region start (elems)
    v5_in: tuple[int, int]  # (A, B) push-staging region starts (elems)
    v5_out: int  # v5 output region start (elems)
    v4_cur: int = 0  # rotation index, flips per v4 AR
    v5_cur: int = 0  # rotation index, flips per v5 AR
    refs: tuple = ()  # keep-alive: (flags, state, hdl, hflags)


# Lazily-built per-group resources, keyed by group name. Built once on the
# first eager call (before any capture). comm.buffer itself is enlarged at
# communicator init (a normal, non-inference tensor) so producer GEMMs can
# write into it -- including v4's input regions.
_INKLING_AR_CACHE: dict[str, _InklingArResources] = {}


@functools.cache
def _ar_jit():
    """The inkling_all_reduce JIT wrapper module, imported once on first use (kept
    lazy so importing comm.py doesn't pull in the JIT machinery)."""
    from sglang.jit_kernel import inkling_all_reduce

    return inkling_all_reduce


@functools.cache
def _ar_fused_jit():
    from sglang.jit_kernel import inkling_ar_fused

    return inkling_ar_fused


def _get_inkling_ar_resources(comm) -> _InklingArResources | None:
    """Return the cached custom-AR resources for ``comm``, or ``None`` if they
    can't be built now (a CUDA-graph capture is active). The first eager call
    populates the cache before capture."""
    key = comm.group.group_name
    cached = _INKLING_AR_CACHE.get(key)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        return None
    import torch.distributed._symmetric_memory as torch_symm_mem

    jit = _ar_jit()
    world = comm.world_size
    # The JIT kernels static_assert a power-of-two world (std::has_single_bit).
    # TP=6 is torch-multimem-eligible but would trip that assertion at compile
    # time; return None so it stays on the plain-multimem fallback path.
    if world & (world - 1) != 0:
        return None
    dev = comm.buffer.device
    hdl = torch_symm_mem.rendezvous(comm.buffer, key)  # idempotent (done at init)
    flags = torch_symm_mem.empty(jit.flags_numel(world), device=dev, dtype=torch.uint32)
    flags.zero_()
    hflags = torch_symm_mem.rendezvous(flags, key)
    # Device-side barrier so no peer's first fused-AR kernel can write an epoch
    # into our flags while our zero_ is still pending on the stream (the zero
    # would clobber the signal; the protocol self-heals, but don't rely on it).
    hflags.barrier()
    state = torch.zeros(jit.STATE_SIZE, device=dev, dtype=torch.uint32)
    jit.compile_inkling_all_reduce(comm.dtype, world)
    # v4 + v5 regions at the tail of comm.buffer. A huge in-place v3 AR's [0:n]
    # may reach into these regions; that is safe -- every fused AR's entry
    # barrier proves all peers finished the previous AR before any broadcast
    # touches the buffer, and the staging regions hold no cross-AR state.
    total = comm.buffer.numel()
    v4reg = _INKLING_AR_V4_REGION
    v5stage = world * _INKLING_AR_V5_REGION
    v5_base = total - 3 * v4reg - 2 * v5stage - _INKLING_AR_V5_REGION
    res = _InklingArResources(
        rank=hdl.rank,
        world=world,
        buffer_ptrs_dev=hdl.buffer_ptrs_dev,
        multicast_ptr=hdl.multicast_ptr,
        flag_ptrs_dev=hflags.buffer_ptrs_dev,
        state_ptr=state.data_ptr(),
        v4_in=(total - 3 * v4reg, total - 2 * v4reg),
        v4_out=total - v4reg,
        v5_in=(v5_base, v5_base + v5stage),
        v5_out=v5_base + 2 * v5stage,
        refs=(flags, state, hdl, hflags),
    )
    _INKLING_AR_CACHE[key] = res
    return res


def ensure_inkling_ar_resources(group: GroupCoordinator) -> None:
    """Eagerly build the custom-AR resources for ``group`` (idempotent).

    Call at model init: the lazy first-call build only works when an eager
    forward runs before CUDA-graph capture (historically guaranteed by the
    prefill BCG capture's eager breaks). With the prefill graph disabled and
    --skip-server-warmup, decode capture would otherwise see no resources and
    silently bake the non-custom fallback into the decode graphs."""
    comm = group.torch_symm_mem_comm
    if (
        comm is not None
        and not comm.disabled
        and group.world_size in _INKLING_AR_WORLD_SIZES
    ):
        _get_inkling_ar_resources(comm)


def _v4_enabled(comm, num_tokens: int) -> bool:
    return _ar_jit().select_ar_config(num_tokens, comm.world_size)[0] == "v4"


# Fused decode {MoE AR -> mlp_sconv -> attn_norm} band: bounded by the v5
# staging region size and by one per-block-barrier slot per token row.
_INKLING_AR_FUSED_MAX_TOKENS = 96
# Target-verify band: T = batch * draft_token_num (144 at bs=16, Q=9), bounded
# by the (enlarged) staging region rows and the per-block barrier slots.
_INKLING_AR_FUSED_MAX_TOKENS_VERIFY = 160


def ar_sconv_norm_fusable(
    group: GroupCoordinator,
    forward_batch,
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
) -> bool:
    """True when a decode {all-reduce -> sconv -> add+RMSNorm} chain
    (attn-side: wo_ud AR -> attn_sconv -> mlp_norm; MoE-side: MoE AR ->
    mlp_sconv -> next attn_norm)
    can run as the single fused kernel (jit_kernel/inkling_ar_fused.py). Must be
    evaluated identically by the producing layer (MoE ``reduce=False``) and the
    consuming layer/tail -- it is a pure function of per-forward state."""
    if not (
        envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR.get()
        and envs.SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV_NORM.get()
    ):
        return False
    fm = forward_batch.forward_mode
    if fm.is_decode():
        max_tokens = _INKLING_AR_FUSED_MAX_TOKENS
    elif fm.is_target_verify():
        max_tokens = _INKLING_AR_FUSED_MAX_TOKENS_VERIFY
    else:
        return False
    comm = group.torch_symm_mem_comm
    if (
        comm is None
        or comm.disabled
        or group.world_size not in _INKLING_AR_WORLD_SIZES
        or dtype != comm.dtype
    ):
        return False
    if (
        num_tokens > max_tokens
        or num_tokens > _ar_jit().MAX_BARRIER_BLOCKS
        or hidden % _INKLING_AR_VEC != 0
        or hidden // _INKLING_AR_VEC > 1024  # one 16B vec per thread, one block/row
        or num_tokens * hidden > _INKLING_AR_V5_REGION
    ):
        return False
    return _get_inkling_ar_resources(comm) is not None


def ar_sconv_norm_fused(
    input: torch.Tensor,
    residual: torch.Tensor,
    sconv,
    norm,
    forward_batch,
    group: GroupCoordinator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused decode {all-reduce -> sconv -> residual-add + RMSNorm}: one kernel
    replacing ``symm_mem_all_reduce`` + ``fused_causal_conv1d_update_decode`` +
    the fused-add RMSNorm. ``input`` holds the UNREDUCED MoE partial sums
    (``InklingMoE.forward(reduce=False)``); returns ``(hs, residual)`` exactly like
    the unfused ``sconv -> norm(hs, res)`` chain. The caller must have checked
    ``ar_sconv_norm_fusable``. Occupies one v5 staging rotation slot (this
    IS a v5 AR with the epilogue seam filled in; same reuse-distance rule)."""
    comm = group.torch_symm_mem_comm
    res = _get_inkling_ar_resources(comm)
    hs_out = torch.empty_like(input)
    residual_out = torch.empty_like(residual)
    cur = res.v5_cur
    stage_off = res.v5_in[cur]
    esz = comm.buffer.element_size()
    mc = res.multicast_ptr + stage_off * esz
    local = comm.buffer.data_ptr() + stage_off * esz
    if forward_batch.forward_mode.is_target_verify():
        sconv_cache, cache_indices, cache_mask, conv_weight, inter_out = (
            sconv.verify_fused_ar_inputs(forward_batch)
        )
        _ar_fused_jit().inkling_ar_sconv_norm_verify(
            input, residual, residual_out, hs_out,
            norm.weight, norm.variance_epsilon,
            sconv_cache, cache_indices, cache_mask, conv_weight,
            inter_out, forward_batch.spec_info.draft_token_num,
            mc, local, res.flag_ptrs_dev, res.state_ptr, res.rank, res.world,
            activation=sconv.activation,
            use_residual=sconv.use_residual,
        )
    else:
        sconv_cache, cache_indices, cache_mask, conv_weight = (
            sconv.decode_fused_ar_inputs(forward_batch)
        )
        _ar_fused_jit().inkling_ar_sconv_norm(
            input, residual, residual_out, hs_out,
            norm.weight, norm.variance_epsilon,
            sconv_cache, cache_indices, cache_mask, conv_weight,
            mc, local, res.flag_ptrs_dev, res.state_ptr, res.rank, res.world,
            activation=sconv.activation,
            use_residual=sconv.use_residual,
            track_mask=forward_batch.mamba_track_mask,
            track_indices=forward_batch.mamba_track_indices,
        )
    res.v5_cur = 1 - cur
    return hs_out, residual_out


def get_ar_buffer(
    group: GroupCoordinator,
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Return a ``[num_tokens, hidden]`` view of a rendezvous'd symm buffer for a
    producer to write into, or ``None`` when the symm-mem fast path is ineligible.

    With ``SGLANG_OPT_USE_INKLING_CUSTOM_AR`` the communicator buffer is enlarged at
    init (256 MiB) so big prefill ARs fit; for the v4 (num_tokens<=2) bucket this
    returns the current rotating input region so ``symm_mem_all_reduce`` can run
    the out-of-place full one-shot.
    """
    comm = group.torch_symm_mem_comm
    if (
        comm is None
        or comm.disabled
        or group.world_size not in _INKLING_AR_WORLD_SIZES
        or dtype != comm.dtype
    ):
        return None
    n = num_tokens * hidden
    nbytes = n * dtype.itemsize
    if nbytes % 4 != 0:
        return None
    if envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR.get():
        res = _get_inkling_ar_resources(comm)
        if (
            res is not None
            and _v4_enabled(comm, num_tokens)
            and n <= _INKLING_AR_V4_REGION
            and n % _INKLING_AR_VEC == 0
        ):
            off = res.v4_in[res.v4_cur]
            return comm.buffer[off : off + n].view(num_tokens, hidden)
    if nbytes >= comm.max_size:
        return None
    return comm.buffer[:n].view(num_tokens, hidden)


def symm_mem_all_reduce(
    input: torch.Tensor,
    group: GroupCoordinator,
    *,
    output: torch.Tensor | None = None,
    input_is_ar_buffer: bool = False,
    num_sms: int = 32,
) -> torch.Tensor:
    """All-reduce ``input`` across ``group`` in the communicator's symm buffer.

    Default (``--enable-torch-symm-mem``): one-shot NVLink ``multimem_all_reduce_``.
    With ``SGLANG_OPT_USE_INKLING_CUSTOM_AR``: dispatch on shape to the autotuned
    custom kernels -- v5 push one-shot (out-of-place, double-buffered staging)
    for the latency band, v3/v3b two-shot multimem for medium/large -- with
    torch multimem for the remaining small ("mm") bucket. The buffer is enlarged
    at init so large prefill ARs take this path instead of NCCL.
    """
    _ = num_sms
    if group.world_size == 1:
        if output is None:
            return input
        output.copy_(input)
        return output

    comm = group.torch_symm_mem_comm
    if (
        output is None
        and comm is not None
        and not comm.disabled
        and group.world_size in _INKLING_AR_WORLD_SIZES
        and comm.should_torch_symm_mem_allreduce(input)
    ):
        n = input.numel()
        num_tokens = input.shape[0] if input.dim() >= 2 else n
        res = (
            _get_inkling_ar_resources(comm)
            if envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR.get()
            else None
        )
        # Custom kernels need a 16B-vector-multiple size (validate() enforces it);
        # a non-vector size falls through to plain multimem below. The "mm" bucket
        # inside this block also uses torch multimem, but it still requires the
        # kernel-eligible size to reach here, so gate the whole block on it.
        if res is not None and n % _INKLING_AR_VEC == 0:
            jit = _ar_jit()
            kernel, nb, bs = jit.select_ar_config(num_tokens, res.world)
            if (
                kernel == "v5"
                and n <= _INKLING_AR_V5_REGION
                and input.data_ptr() % 16 == 0
            ):
                # Push one-shot: multicast-push input into the rotating staging
                # buffer, one per-block barrier, local reduce into the out
                # region. Input is read locally, so it needs NO stage-in copy
                # even when it isn't an AR buffer.
                cur = res.v5_cur
                stage_off = res.v5_in[cur]
                out_view = comm.buffer[res.v5_out : res.v5_out + n]
                esz = comm.buffer.element_size()
                jit.inkling_multimem_push_oneshot(
                    input.view(-1), out_view,
                    res.multicast_ptr + stage_off * esz,
                    comm.buffer.data_ptr() + stage_off * esz,
                    res.flag_ptrs_dev, res.state_ptr, res.rank, res.world,
                    n, nb, bs, per_block_barrier=True,
                )
                res.v5_cur = 1 - cur
                return out_view.view(input.shape)
            if kernel == "v4" and n <= _INKLING_AR_V4_REGION:
                # out-of-place full one-shot in the double-buffered tail regions.
                cur = res.v4_cur
                in_off = res.v4_in[cur]
                out_off = res.v4_out
                in_view = comm.buffer[in_off : in_off + n]
                out_view = comm.buffer[out_off : out_off + n]
                if not input_is_ar_buffer:
                    in_view.copy_(input.view(-1))
                mc = res.multicast_ptr + in_off * comm.buffer.element_size()
                jit.inkling_multimem_full_oneshot(
                    in_view, out_view, mc, res.flag_ptrs_dev,
                    res.state_ptr, res.rank, res.world, n, nb, bs,
                )
                res.v4_cur = 1 - cur
                return out_view.view(input.shape)

            buf = comm.buffer[:n]
            if not input_is_ar_buffer:
                buf.copy_(input.view(-1))
            if kernel in ("v3", "v3b"):
                jit.inkling_multimem_one_shot_fused(
                    buf, res.multicast_ptr, res.flag_ptrs_dev,
                    res.state_ptr, res.rank, res.world, n, nb, bs,
                    per_block_barrier=(kernel == "v3b"),
                )
                return buf.view(input.shape)
            if kernel == "v2":
                jit.inkling_two_shot_all_reduce_fused(
                    buf, res.buffer_ptrs_dev, res.flag_ptrs_dev,
                    res.state_ptr, res.rank, res.world, n, nb, bs,
                )
                return buf.view(input.shape)
            # "mm" bucket: torch multimem on comm.buffer.
            torch.ops.symm_mem.multimem_all_reduce_(buf, "sum", comm.group.group_name)
            return buf.view(input.shape)

        # flag off, resources unavailable (in capture / non-power-of-two world),
        # or a non-vector size: plain multimem.
        buf = comm.buffer[:n]
        if not input_is_ar_buffer:
            buf.copy_(input.view(-1))
        torch.ops.symm_mem.multimem_all_reduce_(buf, "sum", comm.group.group_name)
        return buf.view(input.shape)

    result = group.all_reduce(input)
    if output is None:
        return result
    output.copy_(result)
    return output
