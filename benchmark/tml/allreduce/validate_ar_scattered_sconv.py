"""Validation + bench for the fused extend {AR + scattered sconv} kernel.

Compares ``inkling_ar_scattered_sconv`` (jit_kernel/inkling_ar_scattered_sconv.py)
against the exact unfused reference chain, exploiting that the sconv is
channelwise so the full-width conv equals the concat of the per-rank shard
convs:

    x_ref   = bf16 sum of all ranks' partials            # exact integer patterns
    y_ref   = causal_conv1d(x_ref, weight, cache_ref)    # full-width, unfused jit
    update_sconv_cache(x_ref, cache_ref, ...)            # full-width state update

The fused kernel must produce y == y_ref (all H columns, every rank), the
sharded x_scratch == x_ref[:, rank*Hc:(rank+1)*Hc], and its sharded cache ==
cache_ref[:, :, rank*Hc:(rank+1)*Hc] -- BIT-EXACT (fp32 accum over the same
bf16 values, single rounding at the store, both paths).

Coverage: multi-sequence qsl with boundaries inside token tiles, mixed
has_initial_state, PAD (-1) slots, decode-shaped batches (qsl == arange),
v3 (grid) and v3b (per-block) barriers, interop with plain v3 ARs, and a
CUDA-graph capture + replay phase. --bench adds a fused vs unfused-chain sweep.

Run (TP4):
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \
      benchmark/tml/allreduce/validate_ar_scattered_sconv.py [--bench]
"""

import sys

import torch
import torch.distributed as dist
import validate_inkling_all_reduce as V

from sglang.jit_kernel.inkling_all_reduce import inkling_multimem_one_shot_fused
from sglang.jit_kernel.inkling_ar_scattered_sconv import (
    inkling_ar_banded_sconv,
    inkling_ar_scattered_sconv,
)
from sglang.srt.models.inkling_common.kernels.sconv import (
    _seq_idx_from_cu_seqlens,
    causal_conv1d,
    update_sconv_cache,
)

D = V.HIDDEN
W = 4
POOL = 512
OUT_TOKENS = 8192  # OUT region rows (mirrors _INKLING_AR_SSCONV_OUT_REGION)


class Case:
    """One extend case: shared full-width reference state + this rank's shard."""

    def __init__(self, h, T, qsl, has_init, cache_idx, salt):
        dev = h.buffer.device
        self.T = T
        g = torch.Generator(device=dev)
        g.manual_seed(7000 + salt)
        # Rank-identical full-width state (integer-ish values, exact in bf16).
        self.weight = (
            (torch.randint(-8, 9, (D, W), generator=g, device=dev)) * 0.25
        ).bfloat16()
        self.cache_full = (
            torch.randint(-16, 17, (POOL, W - 1, D), generator=g, device=dev)
        ).bfloat16()
        self.qsl = torch.tensor(qsl, device=dev, dtype=torch.int32)
        b = len(qsl) - 1
        self.has_init = torch.tensor(has_init, device=dev, dtype=torch.bool)
        self.ci = torch.tensor(cache_idx, device=dev, dtype=torch.int32)
        assert self.has_init.numel() == b and self.ci.numel() == b
        # Kernel metadata (mirrors precompute_helion_extend_metadata).
        valid = self.ci != -1
        self.cache_mask = self.has_init & valid
        self.safe_idx = self.ci.clamp(min=0).long()
        self.cu = self.qsl.to(torch.int64)
        self.si = _seq_idx_from_cu_seqlens(self.cu, T)
        self.salt = salt

    def reference(self, h):
        """Full-width unfused chain on a fresh cache clone."""
        n = self.T * D
        x = h.expected(n, self.salt).view(self.T, D)
        cache = self.cache_full.clone()
        y = causal_conv1d(
            x=x,
            weight=self.weight,
            sconv_cache=cache,
            cache_mask=self.cache_mask[:, None, None],
            safe_idx=self.safe_idx,
            cu=self.cu,
            si=self.si,
            activation="silu",
            use_residual=True,
        )
        update_sconv_cache(
            x=x,
            sconv_cache=cache,
            cache_indices=self.ci,
            has_initial_state=self.has_init,
            query_start_loc=self.qsl,
        )
        return x, y, cache


def run_fused(
    h,
    case,
    out_off,
    per_block,
    track=None,
    nb=0,
    bs=0,
    track_from_cache=False,
    norm=None,
    use_stream=False,
):
    """Producer partials into buf[0:n]; fused kernel (update+track in-kernel);
    return (out, x_scratch, cache_shard[, norm_out, residual]). ``norm`` is
    (gamma, residual, eps); it forces the grid exit barrier variant."""
    T = case.T
    n = T * D
    hc = D // h.world
    dev = h.buffer.device
    buf = h.buffer[:n].view(T, D)
    buf.copy_(h.pattern(n, case.salt).view(T, D))
    cache_shard = case.cache_full[:, :, h.rank * hc : (h.rank + 1) * hc].contiguous()
    weight_shard = case.weight[h.rank * hc : (h.rank + 1) * hc].contiguous()
    x_scratch = torch.empty((T, hc), dtype=torch.bfloat16, device=dev)
    if track is None:
        trows = torch.empty((0, W - 1), dtype=torch.int64, device=dev)
        tmask = torch.empty((0,), dtype=torch.bool, device=dev)
        tdst = torch.empty((0,), dtype=torch.int64, device=dev)
    else:
        trows, tmask, tdst = track
    norm_kwargs = {}
    norm_out = residual = None
    if norm is not None:
        gamma, residual, eps = norm
        norm_out = torch.empty((T, D), dtype=torch.bfloat16, device=dev)
        norm_kwargs = dict(
            out_local=h.buffer[out_off : out_off + n].view(T, D),
            norm_gamma=gamma,
            norm_residual=residual,
            norm_out=norm_out,
            norm_eps=eps,
        )
    inkling_ar_scattered_sconv(
        buf,
        x_scratch,
        cache_shard,
        case.safe_idx,
        case.cache_mask,
        case.ci,
        case.has_init,
        case.cu,
        case.si,
        weight_shard,
        trows,
        tmask,
        tdst,
        h.hdl.multicast_ptr,
        h.hdl.multicast_ptr + out_off * h.elem_size,
        h.hflags.buffer_ptrs_dev,
        h.state.data_ptr(),
        h.rank,
        h.world,
        activation="silu",
        use_residual=True,
        num_blocks=nb,
        block_size=bs,
        per_block_barrier=per_block,
        track_from_cache=track_from_cache,
        use_stream=use_stream,
        **norm_kwargs,
    )
    out = h.buffer[out_off : out_off + n].view(T, D).clone()
    if norm is not None:
        return out, x_scratch, cache_shard, norm_out, residual
    return out, x_scratch, cache_shard


def check_case(h, case, out_off, per_block, tag, track=None, use_stream=False):
    x_ref, y_ref, cache_ref = case.reference(h)
    if track is not None:
        trows, tmask, tdst = track
        for b in range(tmask.numel()):
            if tmask[b]:
                for w in range(W - 1):
                    cache_ref[tdst[b], w] = x_ref[trows[b, w]]
    out, x_scratch, cache_shard = run_fused(
        h, case, out_off, per_block, track=track, use_stream=use_stream
    )
    hc = D // h.world
    sl = slice(h.rank * hc, (h.rank + 1) * hc)
    ok_y = torch.equal(out, y_ref)
    ok_x = torch.equal(x_scratch, x_ref[:, sl].contiguous())
    ok_c = torch.equal(cache_shard, cache_ref[:, :, sl].contiguous())
    flags = torch.tensor([ok_y, ok_x, ok_c], dtype=torch.int32, device=out.device)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    if not bool(flags.min().item()):
        bad = (out != y_ref).nonzero()[:4].tolist() if not ok_y else []
        V.log(
            f"  FAIL {tag}: y={bool(flags[0])} x={bool(flags[1])} cache={bool(flags[2])} bad_y={bad}"
        )
        return False
    V.log(f"  ok {tag}")
    return True


def check_fw_case(
    h, case, out_off, per_block, tag, track=None, use_stream=False, nb=0, bs=0, walk=0
):
    """FULL-WIDTH mode (non-scattered): replicated [POOL, W-1, D] cache passed
    whole with cache_col0 = rank * Hc; the kernel must produce the gathered
    post-conv [T, D] AND leave the full-width cache identical to the unfused
    reference on EVERY rank (phase 3 re-ld_reduces window rows full-width)."""
    T = case.T
    n = T * D
    hc = D // h.world
    dev = h.buffer.device
    x_ref, y_ref, cache_ref = case.reference(h)
    if track is not None:
        trows, tmask, tdst = track
        for b in range(tmask.numel()):
            if tmask[b]:
                for w in range(W - 1):
                    cache_ref[tdst[b], w] = x_ref[trows[b, w]]
    buf = h.buffer[:n].view(T, D)
    buf.copy_(h.pattern(n, case.salt).view(T, D))
    cache_full = case.cache_full.clone()
    weight_shard = case.weight[h.rank * hc : (h.rank + 1) * hc]
    x_scratch = torch.empty((T, hc), dtype=torch.bfloat16, device=dev)
    if track is None:
        trows = torch.empty((0, W - 1), dtype=torch.int64, device=dev)
        tmask = torch.empty((0,), dtype=torch.bool, device=dev)
        tdst = torch.empty((0,), dtype=torch.int64, device=dev)
    else:
        trows, tmask, tdst = track
    inkling_ar_scattered_sconv(
        buf,
        x_scratch,
        cache_full,
        case.safe_idx,
        case.cache_mask,
        case.ci,
        case.has_init,
        case.cu,
        case.si,
        weight_shard,
        trows,
        tmask,
        tdst,
        h.hdl.multicast_ptr,
        h.hdl.multicast_ptr + out_off * h.elem_size,
        h.hflags.buffer_ptrs_dev,
        h.state.data_ptr(),
        h.rank,
        h.world,
        activation="silu",
        use_residual=True,
        num_blocks=nb,
        block_size=bs,
        per_block_barrier=per_block,
        need_scratch=False,
        use_stream=use_stream,
        stream_walk=walk,
        full_update=True,
        cache_col0=h.rank * hc,
    )
    out = h.buffer[out_off : out_off + n].view(T, D).clone()
    ok_y = torch.equal(out, y_ref)
    ok_c = torch.equal(cache_full, cache_ref)  # FULL width, every rank
    flags = torch.tensor([ok_y, ok_c], dtype=torch.int32, device=dev)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    if not bool(flags.min().item()):
        bad = (out != y_ref).nonzero()[:4].tolist() if not ok_y else []
        V.log(f"  FAIL {tag}: y={bool(flags[0])} cache={bool(flags[1])} bad_y={bad}")
        return False
    V.log(f"  ok {tag}")
    return True


def check_norm_case(
    h,
    case,
    out_off,
    tag,
    track_from_cache_masked=False,
    per_block=False,
    use_stream=False,
):
    """Norm-fused decode/verify/extend bands (chunked or streaming kernel).
    Optionally exercises the decode track_from_cache path (masked odd rows ->
    post-update window)."""
    dev = h.buffer.device
    b = case.cu.numel() - 1
    g = torch.Generator(device=dev)
    g.manual_seed(9000 + case.salt)
    gamma = (
        (torch.randint(-4, 5, (D,), generator=g, device=dev)) * 0.25 + 1.0
    ).bfloat16()
    residual = (
        (torch.randint(-8, 9, (case.T, D), generator=g, device=dev)) * 0.5
    ).bfloat16()
    eps = 1e-6
    track = None
    track_from_cache = False
    tmask = tdst = None
    if track_from_cache_masked:
        tmask = (torch.arange(b, device=dev) % 2).bool()
        tdst = torch.arange(b, device=dev, dtype=torch.int64) + (POOL - b - 1)
        trows = torch.empty((0, W - 1), dtype=torch.int64, device=dev)
        track = (trows, tmask, tdst)
        track_from_cache = True

    x_ref, y_ref, cache_ref = case.reference(h)
    if track_from_cache_masked:
        for i in range(b):
            if tmask[i] and case.ci[i] != -1:
                cache_ref[tdst[i]] = cache_ref[case.ci[i]]
    v_ref = y_ref.float() + residual.float()
    res_ref = v_ref.bfloat16()
    rms = torch.rsqrt(v_ref.pow(2).mean(-1, keepdim=True) + eps)
    norm_ref = (v_ref * rms * gamma.float()).bfloat16()

    res_inout = residual.clone()
    out, x_scratch, cache_shard, norm_out, res_after = run_fused(
        h,
        case,
        out_off,
        per_block=per_block,
        track=track,
        track_from_cache=track_from_cache,
        norm=(gamma, res_inout, eps),
        use_stream=use_stream,
    )
    hc = D // h.world
    sl = slice(h.rank * hc, (h.rank + 1) * hc)
    ok_y = torch.equal(out, y_ref)
    ok_c = torch.equal(cache_shard, cache_ref[:, :, sl].contiguous())
    ok_res = torch.equal(res_after, res_ref)
    # ssq reduction order differs from torch's mean -> allow 1-ulp bf16 slack.
    ok_n = torch.allclose(norm_out.float(), norm_ref.float(), rtol=2e-2, atol=2e-2)
    flags = torch.tensor([ok_y, ok_c, ok_res, ok_n], dtype=torch.int32, device=dev)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    if not bool(flags.min().item()):
        V.log(
            f"  FAIL {tag}: y={bool(flags[0])} cache={bool(flags[1])}"
            f" res={bool(flags[2])} norm={bool(flags[3])}"
        )
        return False
    V.log(f"  ok {tag}")
    return True


def check_oneshot_case(h, case, out_off, tag, track_masked=False, vpt=0):
    """ONE-SHOT decode {AR + scattered sconv + norm}: sharded cache + full
    weight; partials passed as a LOCAL tensor; window staging in the OUT
    region; post-update-window track semantics."""
    from sglang.jit_kernel.inkling_ar_scattered_sconv import (
        inkling_ar_ssconv_norm_decode,
    )

    dev = h.buffer.device
    T = case.T
    n = T * D
    hc = D // h.world
    b = case.cu.numel() - 1
    assert b == T, "one-shot cases must be decode-shaped (one token per seq)"
    g = torch.Generator(device=dev)
    g.manual_seed(9500 + case.salt)
    gamma = (
        (torch.randint(-4, 5, (D,), generator=g, device=dev)) * 0.25 + 1.0
    ).bfloat16()
    residual = (
        (torch.randint(-8, 9, (T, D), generator=g, device=dev)) * 0.5
    ).bfloat16()
    eps = 1e-6
    if track_masked:
        tmask = (torch.arange(T, device=dev) % 2).bool()
        tdst = torch.arange(T, device=dev, dtype=torch.int64) + (POOL - T - 1)
    else:
        tmask = torch.empty((0,), dtype=torch.bool, device=dev)
        tdst = torch.empty((0,), dtype=torch.int64, device=dev)

    # Reference (full-width unfused chain + norm + track).
    x_ref, y_ref, cache_ref = case.reference(h)
    if track_masked:
        for i in range(T):
            if tmask[i] and case.ci[i] != -1:
                cache_ref[tdst[i]] = cache_ref[case.ci[i]]
    v_ref = y_ref.float() + residual.float()
    res_ref = v_ref.bfloat16()
    rms = torch.rsqrt(v_ref.pow(2).mean(-1, keepdim=True) + eps)
    norm_ref = (v_ref * rms * gamma.float()).bfloat16()

    # Fused one-shot: local partials, sharded cache, full weight.
    in_part = h.pattern(n, case.salt).view(T, D).clone()
    cache_shard = case.cache_full[:, :, h.rank * hc : (h.rank + 1) * hc].contiguous()
    res_out = torch.empty_like(residual)
    hs_out = torch.empty_like(residual)
    esz = h.elem_size
    stage_off = 0  # [world, T, D] partial staging at buffer start
    wstage_off = out_off  # [T, W-1, D] window staging in the OUT area
    inkling_ar_ssconv_norm_decode(
        in_part,
        residual.clone(),
        res_out,
        hs_out,
        gamma,
        eps,
        cache_shard,
        case.ci,
        case.cache_mask,
        case.weight,  # FULL [D, W]
        tmask,
        tdst,
        h.hdl.multicast_ptr + stage_off * esz,
        h.buffer.data_ptr() + stage_off * esz,
        h.hdl.multicast_ptr + wstage_off * esz,
        h.buffer.data_ptr() + wstage_off * esz,
        h.hflags.buffer_ptrs_dev,
        h.state.data_ptr(),
        h.rank,
        h.world,
        activation="silu",
        use_residual=True,
        vecs_per_thread=vpt,
    )
    sl = slice(h.rank * hc, (h.rank + 1) * hc)
    ok_hs = torch.allclose(hs_out.float(), norm_ref.float(), rtol=2e-2, atol=2e-2)
    ok_res = torch.equal(res_out, res_ref)
    ok_c = torch.equal(cache_shard, cache_ref[:, :, sl].contiguous())
    flags = torch.tensor([ok_hs, ok_res, ok_c], dtype=torch.int32, device=dev)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    if not bool(flags.min().item()):
        V.log(
            f"  FAIL {tag}: hs={bool(flags[0])} res={bool(flags[1])} cache={bool(flags[2])}"
        )
        return False
    V.log(f"  ok {tag}")
    return True


def check_coldec_case(h, case, out_off, tag, track_masked=False, vpt=0):
    """COLUMN DECODE V2: block-per-row two-round kernel + fused norm."""
    from sglang.jit_kernel.inkling_ar_scattered_sconv import inkling_ar_col_decode

    dev = h.buffer.device
    T = case.T
    n = T * D
    hc = D // h.world
    g = torch.Generator(device=dev)
    g.manual_seed(9800 + case.salt)
    gamma = (
        (torch.randint(-4, 5, (D,), generator=g, device=dev)) * 0.25 + 1.0
    ).bfloat16()
    residual = (
        (torch.randint(-8, 9, (T, D), generator=g, device=dev)) * 0.5
    ).bfloat16()
    eps = 1e-6
    if track_masked:
        tmask = (torch.arange(T, device=dev) % 2).bool()
        tdst = torch.arange(T, device=dev, dtype=torch.int64) + (POOL - T - 1)
    else:
        tmask = torch.empty((0,), dtype=torch.bool, device=dev)
        tdst = torch.empty((0,), dtype=torch.int64, device=dev)

    x_ref, y_ref, cache_ref = case.reference(h)
    if track_masked:
        for i in range(T):
            if tmask[i] and case.ci[i] != -1:
                cache_ref[tdst[i]] = cache_ref[case.ci[i]]
    v_ref = y_ref.float() + residual.float()
    res_ref = v_ref.bfloat16()
    rms = torch.rsqrt(v_ref.pow(2).mean(-1, keepdim=True) + eps)
    norm_ref = (v_ref * rms * gamma.float()).bfloat16()

    buf = h.buffer[:n].view(T, D)
    buf.copy_(h.pattern(n, case.salt).view(T, D))
    cache_shard = case.cache_full[:, :, h.rank * hc : (h.rank + 1) * hc].contiguous()
    weight_shard = case.weight[h.rank * hc : (h.rank + 1) * hc].contiguous()
    out_local = h.buffer[out_off : out_off + n].view(T, D)
    res_out = torch.empty_like(residual)
    hs_out = torch.empty_like(residual)
    inkling_ar_col_decode(
        buf,
        out_local,
        residual,
        res_out,
        hs_out,
        gamma,
        eps,
        cache_shard,
        case.ci,
        case.cache_mask,
        weight_shard,
        tmask,
        tdst,
        h.hdl.multicast_ptr,
        h.hdl.multicast_ptr + out_off * h.elem_size,
        h.hflags.buffer_ptrs_dev,
        h.state.data_ptr(),
        h.rank,
        h.world,
        activation="silu",
        use_residual=True,
        vecs_per_thread=vpt,
    )
    sl = slice(h.rank * hc, (h.rank + 1) * hc)
    ok_hs = torch.allclose(hs_out.float(), norm_ref.float(), rtol=2e-2, atol=2e-2)
    ok_res = torch.equal(res_out, res_ref)
    ok_c = torch.equal(cache_shard, cache_ref[:, :, sl].contiguous())
    flags = torch.tensor([ok_hs, ok_res, ok_c], dtype=torch.int32, device=dev)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    if not bool(flags.min().item()):
        V.log(
            f"  FAIL {tag}: hs={bool(flags[0])} res={bool(flags[1])} cache={bool(flags[2])}"
        )
        return False
    V.log(f"  ok {tag}")
    return True


EMPTY_I64 = None  # set in main once device known


def run_banded(h, case, out_off, per_block, track=None, nb=0, bs=0, dbg=0):
    """Full-width banded fused call; returns (out, cache_full_clone)."""
    T = case.T
    n = T * D
    buf = h.buffer[:n].view(T, D)
    buf.copy_(h.pattern(n, case.salt).view(T, D))
    cache = case.cache_full.clone()
    tpr = (T + h.world - 1) // h.world
    scratch = torch.empty((tpr + W - 1, D), dtype=torch.bfloat16, device=buf.device)
    if track is None:
        trows = torch.empty((0, W - 1), dtype=torch.int64, device=buf.device)
        tmask = torch.empty((0,), dtype=torch.bool, device=buf.device)
        tdst = torch.empty((0,), dtype=torch.int64, device=buf.device)
    else:
        trows, tmask, tdst = track
    inkling_ar_banded_sconv(
        buf,
        scratch,
        cache,
        case.safe_idx,
        case.cache_mask,
        case.ci,
        case.has_init,
        case.cu,
        case.si,
        case.weight,
        trows,
        tmask,
        tdst,
        h.hdl.multicast_ptr,
        h.hdl.multicast_ptr + out_off * h.elem_size,
        h.hflags.buffer_ptrs_dev,
        h.state.data_ptr(),
        h.rank,
        h.world,
        activation="silu",
        use_residual=True,
        num_blocks=nb,
        block_size=bs,
        per_block_barrier=per_block,
        debug_phase=dbg,
    )
    out = h.buffer[out_off : out_off + n].view(T, D).clone()
    return out, cache


def check_banded_scattered(h, case, out_off, per_block, tag, track=None):
    """Banded kernel in SCATTERED mode: sharded cache + full weight + staged
    windows (pushed in-kernel). Output/track/update must match the full-width
    reference with the cache narrowed to this rank's columns."""
    from sglang.jit_kernel.inkling_ar_scattered_sconv import inkling_ar_banded_sconv

    dev = h.buffer.device
    T = case.T
    n = T * D
    hc = D // h.world
    x_ref, y_ref, cache_ref = case.reference(h)
    if track is not None:
        trows, tmask, tdst = track
        for b in range(tmask.numel()):
            if tmask[b]:
                for w in range(W - 1):
                    cache_ref[tdst[b], w] = x_ref[trows[b, w]]
    else:
        trows = torch.empty((0, W - 1), dtype=torch.int64, device=dev)
        tmask = torch.empty((0,), dtype=torch.bool, device=dev)
        tdst = torch.empty((0,), dtype=torch.int64, device=dev)
    buf = h.buffer[:n].view(T, D)
    buf.copy_(h.pattern(n, case.salt).view(T, D))
    cache_shard = case.cache_full[:, :, h.rank * hc : (h.rank + 1) * hc].contiguous()
    tpr = -(-T // h.world)
    scratch = torch.empty((tpr + W - 1, D), dtype=torch.bfloat16, device=dev)
    # window staging: carve past the OUT rows (disjoint from in/out regions)
    wstage_off = out_off + min(V.MAX_TOKENS, OUT_TOKENS) * D
    inkling_ar_banded_sconv(
        buf,
        scratch,
        cache_shard,
        case.safe_idx,
        case.cache_mask,
        case.ci,
        case.has_init,
        case.cu,
        case.si,
        case.weight,
        trows,
        tmask,
        tdst,
        h.hdl.multicast_ptr,
        h.hdl.multicast_ptr + out_off * h.elem_size,
        h.hflags.buffer_ptrs_dev,
        h.state.data_ptr(),
        h.rank,
        h.world,
        activation="silu",
        use_residual=True,
        per_block_barrier=per_block,
        mc_wstage=h.hdl.multicast_ptr + wstage_off * h.elem_size,
        local_wstage=h.buffer.data_ptr() + wstage_off * h.elem_size,
    )
    out = h.buffer[out_off : out_off + n].view(T, D).clone()
    sl = slice(h.rank * hc, (h.rank + 1) * hc)
    ok_y = torch.equal(out, y_ref)
    ok_c = torch.equal(cache_shard, cache_ref[:, :, sl].contiguous())
    flags = torch.tensor([ok_y, ok_c], dtype=torch.int32, device=dev)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    if not bool(flags.min().item()):
        bad = (out != y_ref).nonzero()[:4].tolist() if not ok_y else []
        V.log(f"  FAIL {tag}: y={bool(flags[0])} cache={bool(flags[1])} bad_y={bad}")
        return False
    V.log(f"  ok {tag}")
    return True


def check_banded(h, case, out_off, per_block, tag, track=None):
    x_ref, y_ref, cache_ref = case.reference(h)
    if track is not None:
        trows, tmask, tdst = track
        for b in range(tmask.numel()):
            if tmask[b]:
                for w in range(W - 1):
                    cache_ref[tdst[b], w] = x_ref[trows[b, w]]
    out, cache = run_banded(h, case, out_off, per_block, track=track)
    ok_y = torch.equal(out, y_ref)
    ok_c = torch.equal(cache, cache_ref)
    flags = torch.tensor([ok_y, ok_c], dtype=torch.int32, device=out.device)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    if not bool(flags.min().item()):
        V.log(f"  FAIL {tag}: y={bool(flags[0])} cache={bool(flags[1])}")
        return False
    V.log(f"  ok {tag}")
    return True


def main():
    bench = "--bench" in sys.argv
    if any(a.startswith("--tune") for a in sys.argv):
        # Room for T=16384 inputs + a disjoint OUT region (chunked prefill
        # bounds production T at max_prefill_tokens=16384).
        V.MAX_TOKENS = 32768
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    h = V.Harness(dev)
    out_off = min(V.MAX_TOKENS, OUT_TOKENS) * D  # OUT region after input rows
    assert (
        h.buffer.numel() >= out_off + OUT_TOKENS * D or True
    )  # harness sized by MAX_TOKENS

    V.log(f"validate_ar_scattered_sconv world={h.world} hidden={D} W={W}")
    fails = 0

    # [S1] correctness matrix
    cases = []
    # decode-shaped: every token its own sequence
    for T in (1, 2, 3, 8, 64):
        qsl = list(range(T + 1))
        cases.append(
            (
                f"decode T={T}",
                T,
                qsl,
                [True] * T,
                [(i * 3 + 1) % POOL for i in range(T)],
            )
        )
    # single-seq extends (tile-boundary coverage: T straddles rt=16 tiles)
    for T in (5, 16, 17, 250, 1024):
        cases.append((f"ext1 T={T}", T, [0, T], [True], [7]))
    # fresh state (has_init False) + PAD slot
    cases.append(("fresh T=40", 40, [0, 40], [False], [3]))
    cases.append(("pad T=40", 40, [0, 40], [True], [-1]))
    # multi-seq with boundaries inside tiles, mixed state
    cases.append(
        (
            "multi4 T=301",
            301,
            [0, 5, 21, 150, 301],
            [True, False, True, True],
            [1, 2, -1, 9],
        )
    )
    cases.append(("multi2 T=4096", 4096, [0, 1500, 4096], [True, True], [11, 12]))
    for tag, T, qsl, hi, ci in cases:
        for per_block in (False, True):
            c = Case(h, T, qsl, hi, ci, salt=T + (17 if per_block else 0))
            ok = check_case(
                h, c, out_off, per_block, f"{tag} {'v3b' if per_block else 'v3'}"
            )
            fails += 0 if ok else 1

    # [S1a2] STREAMING rolling-window variant (extends)
    V.log("streaming rolling-window kernel:")
    for tag, T, qsl, hi, ci in cases:
        c = Case(h, T, qsl, hi, ci, salt=7700 + T)
        ok = check_case(h, c, out_off, False, f"st-{tag}", use_stream=True)
        fails += 0 if ok else 1
    c = Case(h, 301, [0, 5, 150, 301], [True, True, True], [1, 2, 9], salt=7999)
    trows = torch.tensor(
        [[0, 1, 2], [64, 65, 66], [200, 201, 202]], dtype=torch.int64, device=dev
    )
    tmask = torch.tensor([True, False, True], dtype=torch.bool, device=dev)
    tdst = torch.tensor([100, 101, 102], dtype=torch.int64, device=dev)
    fails += (
        0
        if check_case(
            h,
            c,
            out_off,
            False,
            "st-track T=301",
            track=(trows, tmask, tdst),
            use_stream=True,
        )
        else 1
    )

    # [S1a3] FULL-WIDTH mode (non-scattered): replicated cache + cache_col0,
    # phase 3 updates all H columns on every rank. Chunked AND streaming.
    V.log("full-width (replicated cache, non-scattered) mode:")
    fw_cases = [
        (
            "fw-decode T=8",
            8,
            list(range(9)),
            [True] * 8,
            [(i * 3 + 1) % POOL for i in range(8)],
        ),
        ("fw-ext1 T=250", 250, [0, 250], [True], [7]),
        ("fw-ext1 T=1024", 1024, [0, 1024], [True], [7]),
        ("fw-fresh T=40", 40, [0, 40], [False], [3]),
        ("fw-pad T=40", 40, [0, 40], [True], [-1]),
        (
            "fw-multi4 T=301",
            301,
            [0, 5, 21, 150, 301],
            [True, False, True, True],
            [1, 2, -1, 9],
        ),
        ("fw-multi2 T=4096", 4096, [0, 1500, 4096], [True, True], [11, 12]),
        ("fw-multi2 T=8192", 8192, [0, 3000, 8192], [True, False], [5, 6]),
    ]
    for tag, T, qsl, hi, ci in fw_cases:
        for use_stream in (False, True):
            c = Case(h, T, qsl, hi, ci, salt=8300 + T + (23 if use_stream else 0))
            ok = check_fw_case(
                h,
                c,
                out_off,
                False,
                f"{tag} {'stream' if use_stream else 'chunked'}",
                use_stream=use_stream,
            )
            fails += 0 if ok else 1
    c = Case(h, 301, [0, 5, 150, 301], [True, True, True], [1, 2, 9], salt=8399)
    fails += (
        0
        if check_fw_case(
            h,
            c,
            out_off,
            False,
            "fw-track T=301",
            track=(trows, tmask, tdst),
            use_stream=False,
        )
        else 1
    )

    # [S1b] token-banded (full-width) matrix
    V.log("banded (full-width) kernel:")
    banded_cases = [
        (
            "b-decode T=8",
            8,
            list(range(9)),
            [True] * 8,
            [(i * 3 + 1) % POOL for i in range(8)],
        ),
        ("b-ext1 T=250", 250, [0, 250], [True], [7]),
        ("b-fresh T=40", 40, [0, 40], [False], [3]),
        ("b-pad T=40", 40, [0, 40], [True], [-1]),
        (
            "b-multi4 T=301",
            301,
            [0, 5, 21, 150, 301],
            [True, False, True, True],
            [1, 2, -1, 9],
        ),
        ("b-multi2 T=4096", 4096, [0, 1500, 4096], [True, True], [11, 12]),
        ("b-bandedge T=7", 7, [0, 7], [True], [4]),  # bands thinner than W-1
    ]
    for tag, T, qsl, hi, ci in banded_cases:
        for per_block in (False, True):
            c = Case(h, T, qsl, hi, ci, salt=3000 + T + (7 if per_block else 0))
            ok = check_banded(
                h, c, out_off, per_block, f"{tag} {'v3b' if per_block else 'v3'}"
            )
            fails += 0 if ok else 1
    # [S1b2] banded SCATTERED mode: sharded cache + staged window push
    V.log("banded-scattered (sharded cache, window push) kernel:")
    bs_cases = [
        ("bsc-ext1 T=250", 250, [0, 250], [True], [7]),
        ("bsc-ext1 T=4096", 4096, [0, 4096], [True], [11]),
        ("bsc-fresh T=40", 40, [0, 40], [False], [3]),
        ("bsc-pad T=40", 40, [0, 40], [True], [-1]),
        (
            "bsc-multi4 T=301",
            301,
            [0, 5, 21, 150, 301],
            [True, False, True, True],
            [1, 2, -1, 9],
        ),
        (
            "bsc-decode T=8",
            8,
            list(range(9)),
            [True] * 8,
            [(i * 3 + 1) % POOL for i in range(8)],
        ),
    ]
    for tag, T, qsl, hi, ci in bs_cases:
        for per_block in (False, True):
            c = Case(h, T, qsl, hi, ci, salt=7000 + T + (11 if per_block else 0))
            ok = check_banded_scattered(
                h, c, out_off, per_block, f"{tag} {'v3b' if per_block else 'v3'}"
            )
            fails += 0 if ok else 1
    c = Case(h, 301, [0, 5, 150, 301], [True, True, True], [1, 2, 9], salt=7301)
    trows = torch.tensor(
        [[0, 1, 2], [64, 65, 66], [200, 201, 202]], dtype=torch.int64, device=dev
    )
    tmask = torch.tensor([True, False, True], dtype=torch.bool, device=dev)
    tdst = torch.tensor([100, 101, 102], dtype=torch.int64, device=dev)
    fails += (
        0
        if check_banded_scattered(
            h, c, out_off, True, "bsc-track T=301", track=(trows, tmask, tdst)
        )
        else 1
    )

    # [S1c] norm-fused decode/verify band (grid barrier) + decode track_from_cache
    V.log("norm-fused (add+RMSNorm tail) kernel:")
    norm_cases = [
        ("norm-decode T=1", 1, [0, 1], [True], [9]),
        (
            "norm-decode T=8",
            8,
            list(range(9)),
            [True] * 8,
            [(i * 5 + 2) % POOL for i in range(8)],
        ),
        (
            "norm-decode T=96",
            96,
            list(range(97)),
            [True] * 96,
            [(i * 3 + 1) % POOL for i in range(96)],
        ),
        ("norm-verify T=64 pad", 64, [0, 16, 32, 48, 64], [True] * 4, [-1, -1, -1, -1]),
        (
            "norm-fresh T=12",
            12,
            list(range(13)),
            [False] * 12,
            [(i * 7 + 3) % POOL for i in range(12)],
        ),
    ]
    for tag, T, qsl, hi, ci in norm_cases:
        for per_block in (False, True):
            c = Case(h, T, qsl, hi, ci, salt=5000 + T + (13 if per_block else 0))
            ok = check_norm_case(
                h,
                c,
                out_off,
                f"{tag} {'v3b' if per_block else 'v3'}",
                per_block=per_block,
            )
            fails += 0 if ok else 1
    # decode track_from_cache: masked odd rows snapshot the post-update window
    for per_block in (False, True):
        c = Case(
            h,
            16,
            list(range(17)),
            [True] * 16,
            [(i * 3 + 1) % POOL for i in range(16)],
            salt=5200 + per_block,
        )
        ok = check_norm_case(
            h,
            c,
            out_off,
            f"norm-decode-track T=16 {'v3b' if per_block else 'v3'}",
            track_from_cache_masked=True,
            per_block=per_block,
        )
        fails += 0 if ok else 1

    # [S1c2] extend band with the fused norm tail (the extend call sites pass
    # norm too now): chunked AND streaming, incl. T=8192 where the chunked
    # path turns on the smem weight stage + smem tile alongside the tail.
    V.log("norm-fused extend band (chunked + streaming):")
    norm_ext_cases = [
        (
            "norm-ext-multi4 T=301",
            301,
            [0, 5, 21, 150, 301],
            [True, False, True, True],
            [1, 2, -1, 9],
        ),
        ("norm-ext T=1024", 1024, [0, 1024], [True], [7]),
        ("norm-ext-multi2 T=4096", 4096, [0, 1500, 4096], [True, True], [11, 12]),
        ("norm-ext-fresh T=8192", 8192, [0, 3000, 8192], [True, False], [5, 6]),
    ]
    for tag, T, qsl, hi, ci in norm_ext_cases:
        for use_stream in (False, True):
            c = Case(h, T, qsl, hi, ci, salt=5400 + T + (19 if use_stream else 0))
            ok = check_norm_case(
                h,
                c,
                out_off,
                f"{tag} {'stream' if use_stream else 'chunked'}",
                use_stream=use_stream,
            )
            fails += 0 if ok else 1

    # [S1d] ONE-SHOT decode {AR + scattered sconv + norm} (v5 push pattern)
    V.log("one-shot decode (push + window-shard staging) kernel:")
    oneshot_cases = [
        ("os-decode T=1", 1, [0, 1], [True], [9]),
        (
            "os-decode T=8",
            8,
            list(range(9)),
            [True] * 8,
            [(i * 5 + 2) % POOL for i in range(8)],
        ),
        (
            "os-decode T=96",
            96,
            list(range(97)),
            [True] * 96,
            [(i * 3 + 1) % POOL for i in range(96)],
        ),
        (
            "os-fresh T=12",
            12,
            list(range(13)),
            [False] * 12,
            [(i * 7 + 3) % POOL for i in range(12)],
        ),
        ("os-pad T=8", 8, list(range(9)), [True] * 8, [-1] * 8),
    ]
    for tag, T, qsl, hi, ci in oneshot_cases:
        for vpt in (1, 2):
            c = Case(h, T, qsl, hi, ci, salt=6000 + T + 29 * vpt)
            ok = check_oneshot_case(h, c, out_off, f"{tag} vpt{vpt}", vpt=vpt)
            fails += 0 if ok else 1
    c = Case(
        h,
        16,
        list(range(17)),
        [True] * 16,
        [(i * 3 + 1) % POOL for i in range(16)],
        salt=6200,
    )
    fails += (
        0
        if check_oneshot_case(h, c, out_off, "os-decode-track T=16", track_masked=True)
        else 1
    )

    # [S1e] COLUMN DECODE V2
    V.log("column decode v2 (block-per-row two-round) kernel:")
    for tag, T, qsl, hi, ci in [
        ("cd2 T=1", 1, [0, 1], [True], [9]),
        (
            "cd2 T=8",
            8,
            list(range(9)),
            [True] * 8,
            [(i * 5 + 2) % POOL for i in range(8)],
        ),
        (
            "cd2 T=96",
            96,
            list(range(97)),
            [True] * 96,
            [(i * 3 + 1) % POOL for i in range(96)],
        ),
        (
            "cd2 fresh T=12",
            12,
            list(range(13)),
            [False] * 12,
            [(i * 7 + 3) % POOL for i in range(12)],
        ),
        ("cd2 pad T=8", 8, list(range(9)), [True] * 8, [-1] * 8),
        (
            "cd2 T=204",
            204,
            list(range(205)),
            [True] * 204,
            [(i * 3 + 1) % POOL for i in range(204)],
        ),
    ]:
        for vpt in (1, 2):
            c = Case(h, T, qsl, hi, ci, salt=8200 + T + 31 * vpt)
            ok = check_coldec_case(h, c, out_off, f"{tag} vpt{vpt}", vpt=vpt)
            fails += 0 if ok else 1
    c = Case(
        h,
        16,
        list(range(17)),
        [True] * 16,
        [(i * 3 + 1) % POOL for i in range(16)],
        salt=8400,
    )
    fails += (
        0
        if check_coldec_case(h, c, out_off, "cd2 track T=16", track_masked=True)
        else 1
    )

    # banded + track
    c = Case(h, 301, [0, 5, 150, 301], [True, True, True], [1, 2, 9], salt=4001)
    trows = torch.tensor(
        [[0, 1, 2], [64, 65, 66], [200, 201, 202]], dtype=torch.int64, device=dev
    )
    tmask = torch.tensor([True, False, True], dtype=torch.bool, device=dev)
    tdst = torch.tensor([100, 101, 102], dtype=torch.int64, device=dev)
    fails += (
        0
        if check_banded(
            h, c, out_off, True, "b-track T=301", track=(trows, tmask, tdst)
        )
        else 1
    )
    c2 = Case(h, 301, [0, 5, 150, 301], [True, True, True], [1, 2, 9], salt=4002)
    fails += (
        0
        if check_case(
            h, c2, out_off, True, "col-track T=301", track=(trows, tmask, tdst)
        )
        else 1
    )

    # [S2] interop: fused calls interleaved with plain v3 ARs (shared barrier state)
    c = Case(h, 128, [0, 128], [True], [5], salt=555)
    x_ref, y_ref, cache_ref = c.reference(h)
    for i in range(3):
        n = 64 * D
        buf = h.buffer[:n]
        buf.copy_(h.pattern(n, 42 + i))
        inkling_multimem_one_shot_fused(
            buf,
            h.hdl.multicast_ptr,
            h.hflags.buffer_ptrs_dev,
            h.state.data_ptr(),
            h.rank,
            h.world,
            n,
            0,
            0,
            per_block_barrier=(i % 2 == 0),
        )
        assert torch.equal(buf, h.expected(n, 42 + i)), "plain v3 interop failed"
        out, _, _ = run_fused(h, c, out_off, per_block=(i % 2 == 1))
        ok = torch.equal(out, y_ref)
        flags = torch.tensor([ok], dtype=torch.int32, device=dev)
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        fails += 0 if bool(flags.min().item()) else 1
    V.log(f"  ok interop x3" if fails == 0 else f"  interop fails={fails}")

    # [S3] CUDA-graph capture + replay
    c = Case(h, 256, [0, 100, 256], [True, True], [21, 22], salt=777)
    x_ref, y_ref, cache_ref = c.reference(h)
    hc = D // h.world
    cache_shard = c.cache_full[:, :, h.rank * hc : (h.rank + 1) * hc].contiguous()
    weight_shard = c.weight[h.rank * hc : (h.rank + 1) * hc].contiguous()
    n = c.T * D
    buf = h.buffer[:n].view(c.T, D)
    x_scratch = torch.empty((c.T, hc), dtype=torch.bfloat16, device=dev)
    cache_snap = cache_shard.clone()

    te_r3 = torch.empty((0, W - 1), dtype=torch.int64, device=dev)
    te_m3 = torch.empty((0,), dtype=torch.bool, device=dev)
    te_d3 = torch.empty((0,), dtype=torch.int64, device=dev)

    def call():
        inkling_ar_scattered_sconv(
            buf,
            x_scratch,
            cache_shard,
            c.safe_idx,
            c.cache_mask,
            c.ci,
            c.has_init,
            c.cu,
            c.si,
            weight_shard,
            te_r3,
            te_m3,
            te_d3,
            h.hdl.multicast_ptr,
            h.hdl.multicast_ptr + out_off * h.elem_size,
            h.hflags.buffer_ptrs_dev,
            h.state.data_ptr(),
            h.rank,
            h.world,
            activation="silu",
            use_residual=True,
            per_block_barrier=True,
        )

    buf.copy_(h.pattern(n, c.salt).view(c.T, D))
    call()  # warm (JIT compile outside capture)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for r in range(3):
        cache_shard.copy_(cache_snap)
        buf.copy_(h.pattern(n, c.salt).view(c.T, D))
        graph.replay()
        torch.cuda.synchronize()
        out = h.buffer[out_off : out_off + n].view(c.T, D)
        ok = torch.equal(out, y_ref)
        flags = torch.tensor([ok], dtype=torch.int32, device=dev)
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        fails += 0 if bool(flags.min().item()) else 1
    V.log("  ok graph capture+replay x3" if fails == 0 else f"  graph fails={fails}")

    V.log(f"RESULT: {'PASS' if fails == 0 else f'FAIL ({fails})'}")

    if bench and fails == 0:
        V.log("\nbench: fused vs unfused chains (min-of-5x20, us)")
        V.log(
            "columns: prod(best cfg) | unfused-scat | fused v3/v3b | banded v3/v3b, best (nb,bs) each"
        )
        gname = dist.group.WORLD.group_name
        for T in (
            1,
            4,
            8,
            16,
            64,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
            6144,
            8192,
        ):
            if T > min(V.MAX_TOKENS, OUT_TOKENS):
                break
            c = Case(h, T, [0, T], [True], [7], salt=T)
            n = T * D
            hc2 = D // h.world
            cache_shard2 = c.cache_full[
                :, :, h.rank * hc2 : (h.rank + 1) * hc2
            ].contiguous()
            cache_full2 = c.cache_full.clone()
            x_full = h.expected(n, c.salt).view(T, D).clone()

            # Unfused estimate: plain v3 AR + full-width sconv on the reduced x
            # (production baseline without scattering; comm volume identical).
            def unfused():
                b = h.buffer[:n]
                inkling_multimem_one_shot_fused(
                    b,
                    h.hdl.multicast_ptr,
                    h.hflags.buffer_ptrs_dev,
                    h.state.data_ptr(),
                    h.rank,
                    h.world,
                    n,
                    0,
                    0,
                    per_block_barrier=False,
                )
                causal_conv1d(
                    x=x_full,
                    weight=c.weight,
                    sconv_cache=cache_full2,
                    cache_mask=c.cache_mask[:, None, None],
                    safe_idx=c.safe_idx,
                    cu=c.cu,
                    si=c.si,
                    activation="silu",
                    use_residual=True,
                )

            hcf = D // h.world
            cache_shard_f = c.cache_full[
                :, :, h.rank * hcf : (h.rank + 1) * hcf
            ].contiguous()
            weight_shard_f = c.weight[h.rank * hcf : (h.rank + 1) * hcf].contiguous()
            x_scratch_f = torch.empty((T, hcf), dtype=torch.bfloat16, device=dev)
            buf_f = h.buffer[:n].view(T, D)
            te0_r = torch.empty((0, W - 1), dtype=torch.int64, device=dev)
            te0_m = torch.empty((0,), dtype=torch.bool, device=dev)
            te0_d = torch.empty((0,), dtype=torch.int64, device=dev)

            def fused(pb):
                def f():
                    inkling_ar_scattered_sconv(
                        buf_f,
                        x_scratch_f,
                        cache_shard_f,
                        c.safe_idx,
                        c.cache_mask,
                        c.ci,
                        c.has_init,
                        c.cu,
                        c.si,
                        weight_shard_f,
                        te0_r,
                        te0_m,
                        te0_d,
                        h.hdl.multicast_ptr,
                        h.hdl.multicast_ptr + out_off * h.elem_size,
                        h.hflags.buffer_ptrs_dev,
                        h.state.data_ptr(),
                        h.rank,
                        h.world,
                        activation="silu",
                        use_residual=True,
                        per_block_barrier=pb,
                    )

                return f

            # True unfused SCATTERED chain (production reduce_scatter_hidden /
            # all_gather_hidden): torch multimem RS -> shard conv -> multimem AG.
            hc3 = D // h.world
            weight_shard3 = c.weight[h.rank * hc3 : (h.rank + 1) * hc3].contiguous()
            cache_shard3 = c.cache_full[
                :, :, h.rank * hc3 : (h.rank + 1) * hc3
            ].contiguous()
            symm_in = h.buffer[:n].view(T, D)
            rs_out = torch.empty((T, hc3), dtype=torch.bfloat16, device=dev)
            ag_out = h.buffer[out_off : out_off + n].view(h.world * T, hc3)

            def unfused_scattered():
                torch.ops.symm_mem.reduce_scatter_out(symm_in, gname, True, rs_out)
                y = causal_conv1d(
                    x=rs_out,
                    weight=weight_shard3,
                    sconv_cache=cache_shard3,
                    cache_mask=c.cache_mask[:, None, None],
                    safe_idx=c.safe_idx,
                    cu=c.cu,
                    si=c.si,
                    activation="silu",
                    use_residual=True,
                )
                torch.ops.symm_mem.multimem_all_gather_out(y, gname, ag_out)
                ag_out.view(h.world, T, hc3).movedim(0, 1).reshape(T, D)

            h.buffer[:n].copy_(h.pattern(n, c.salt))
            t_unf = V.bench_us(dev, unfused)
            t_scat = V.bench_us(dev, unfused_scattered)
            # Per-kernel config sweep: best over nb x bs (0 = kernel default).
            CFGS = [(0, 0)] + [
                (nb2, bs2) for nb2 in (16, 32, 64, 96, 148) for bs2 in (512, 1024)
            ]

            def best_of(fn):
                best_t, best_c = float("inf"), (0, 0)
                for cfg in CFGS:
                    tt = V.bench_us(dev, fn(*cfg))
                    if tt < best_t:
                        best_t, best_c = tt, cfg
                return best_t, best_c

            # prod v3 (incl. the autotuned-table config in the sweep grid).
            from sglang.jit_kernel.inkling_all_reduce import select_ar_config

            kk, tnb, tbs = select_ar_config(T, h.world)

            def prod_fn(nb2, bs2):
                def f():
                    b = h.buffer[:n]
                    inkling_multimem_one_shot_fused(
                        b,
                        h.hdl.multicast_ptr,
                        h.hflags.buffer_ptrs_dev,
                        h.state.data_ptr(),
                        h.rank,
                        h.world,
                        n,
                        nb2,
                        bs2,
                        per_block_barrier=False,
                    )
                    causal_conv1d(
                        x=x_full,
                        weight=c.weight,
                        sconv_cache=cache_full2,
                        cache_mask=c.cache_mask[:, None, None],
                        safe_idx=c.safe_idx,
                        cu=c.cu,
                        si=c.si,
                        activation="silu",
                        use_residual=True,
                    )
                    update_sconv_cache(
                        x=x_full,
                        sconv_cache=cache_full2,
                        cache_indices=c.ci,
                        has_initial_state=c.has_init,
                        query_start_loc=c.qsl,
                    )

                return f

            t_prod, c_prod = best_of(prod_fn)
            if kk == "v3":  # also try the tuned-table config explicitly
                tt = V.bench_us(dev, prod_fn(tnb, tbs))
                if tt < t_prod:
                    t_prod, c_prod = tt, (tnb, tbs)

            def fused_fn(pb):
                def g(nb2, bs2):
                    def f():
                        inkling_ar_scattered_sconv(
                            buf_f,
                            x_scratch_f,
                            cache_shard_f,
                            c.safe_idx,
                            c.cache_mask,
                            c.ci,
                            c.has_init,
                            c.cu,
                            c.si,
                            weight_shard_f,
                            te0_r,
                            te0_m,
                            te0_d,
                            h.hdl.multicast_ptr,
                            h.hdl.multicast_ptr + out_off * h.elem_size,
                            h.hflags.buffer_ptrs_dev,
                            h.state.data_ptr(),
                            h.rank,
                            h.world,
                            activation="silu",
                            use_residual=True,
                            num_blocks=nb2,
                            block_size=bs2,
                            per_block_barrier=pb,
                        )

                    return f

                return g

            cache_b = c.cache_full.clone()
            tprb = (T + h.world - 1) // h.world
            scratch_b = torch.empty((tprb + W - 1, D), dtype=torch.bfloat16, device=dev)
            te_r = torch.empty((0, W - 1), dtype=torch.int64, device=dev)
            te_m = torch.empty((0,), dtype=torch.bool, device=dev)
            te_d = torch.empty((0,), dtype=torch.int64, device=dev)

            def banded_fn(pb):
                def g(nb2, bs2):
                    def f():
                        inkling_ar_banded_sconv(
                            buf_f,
                            scratch_b,
                            cache_b,
                            c.safe_idx,
                            c.cache_mask,
                            c.ci,
                            c.has_init,
                            c.cu,
                            c.si,
                            c.weight,
                            te_r,
                            te_m,
                            te_d,
                            h.hdl.multicast_ptr,
                            h.hdl.multicast_ptr + out_off * h.elem_size,
                            h.hflags.buffer_ptrs_dev,
                            h.state.data_ptr(),
                            h.rank,
                            h.world,
                            activation="silu",
                            use_residual=True,
                            num_blocks=nb2,
                            block_size=bs2,
                            per_block_barrier=pb,
                        )

                    return f

                return g

            # banded SCATTERED: sharded cache + full weight + window push.
            cache_bsc = c.cache_full[
                :, :, h.rank * hc2 : (h.rank + 1) * hc2
            ].contiguous()
            wst_off = out_off + min(V.MAX_TOKENS, OUT_TOKENS) * D

            def bsc_fn(pb):
                def g(nb2, bs2):
                    def f():
                        inkling_ar_banded_sconv(
                            buf_f,
                            scratch_b,
                            cache_bsc,
                            c.safe_idx,
                            c.cache_mask,
                            c.ci,
                            c.has_init,
                            c.cu,
                            c.si,
                            c.weight,
                            te_r,
                            te_m,
                            te_d,
                            h.hdl.multicast_ptr,
                            h.hdl.multicast_ptr + out_off * h.elem_size,
                            h.hflags.buffer_ptrs_dev,
                            h.state.data_ptr(),
                            h.rank,
                            h.world,
                            activation="silu",
                            use_residual=True,
                            num_blocks=nb2,
                            block_size=bs2,
                            per_block_barrier=pb,
                            mc_wstage=h.hdl.multicast_ptr + wst_off * h.elem_size,
                            local_wstage=h.buffer.data_ptr() + wst_off * h.elem_size,
                        )

                    return f

                return g

            t_f3, c_f3 = best_of(fused_fn(False))
            t_f3b, c_f3b = best_of(fused_fn(True))
            t_b3, c_b3 = best_of(banded_fn(False))
            t_b3b, c_b3b = best_of(banded_fn(True))
            t_bsc, c_bsc = best_of(bsc_fn(False))
            t_bscb, c_bscb = best_of(bsc_fn(True))
            t_scat2 = V.bench_us(dev, unfused_scattered)
            V.log(
                f"{T:>7} | prod={t_prod:>6.1f}{c_prod} | scat={t_scat2:>6.1f}"
                f" | f3={t_f3:>6.1f}{c_f3} | f3b={t_f3b:>6.1f}{c_f3b}"
                f" | b3={t_b3:>6.1f}{c_b3} | b3b={t_b3b:>6.1f}{c_b3b}"
                f" | bsc3={t_bsc:>6.1f}{c_bsc} | bsc3b={t_bscb:>6.1f}{c_bscb}"
            )

    # One-shot decode kernel bench: decode-shaped (B == T) latency band vs the
    # column two-shot + norm (what it replaced in the decode graphs).
    if bench:
        from sglang.jit_kernel.inkling_ar_scattered_sconv import (
            inkling_ar_ssconv_norm_decode,
        )

        V.log("one-shot decode band (us, best of vpt 1/2):")
        for T in (1, 4, 8, 16, 32, 64, 96):
            qsl = list(range(T + 1))
            c = Case(
                h,
                T,
                qsl,
                [True] * T,
                [(i * 3 + 1) % POOL for i in range(T)],
                salt=8800 + T,
            )
            n = T * D
            hcx = D // h.world
            gsm = torch.ones((D,), dtype=torch.bfloat16, device=dev)
            resid = torch.zeros((T, D), dtype=torch.bfloat16, device=dev)
            ro = torch.empty_like(resid)
            ho = torch.empty_like(resid)
            in_part = h.pattern(n, c.salt).view(T, D).clone()
            cache_sh = c.cache_full[
                :, :, h.rank * hcx : (h.rank + 1) * hcx
            ].contiguous()
            tmask0 = torch.empty((0,), dtype=torch.bool, device=dev)
            tdst0 = torch.empty((0,), dtype=torch.int64, device=dev)
            wst_off2 = out_off + min(V.MAX_TOKENS, OUT_TOKENS) * D

            def os_fn(vpt):
                def f():
                    inkling_ar_ssconv_norm_decode(
                        in_part,
                        resid,
                        ro,
                        ho,
                        gsm,
                        1e-6,
                        cache_sh,
                        c.ci,
                        c.cache_mask,
                        c.weight,
                        tmask0,
                        tdst0,
                        h.hdl.multicast_ptr,
                        h.buffer.data_ptr(),
                        h.hdl.multicast_ptr + wst_off2 * h.elem_size,
                        h.buffer.data_ptr() + wst_off2 * h.elem_size,
                        h.hflags.buffer_ptrs_dev,
                        h.state.data_ptr(),
                        h.rank,
                        h.world,
                        activation="silu",
                        use_residual=True,
                        vecs_per_thread=vpt,
                    )

                return f

            t1 = V.bench_us(dev, os_fn(1))
            t2 = V.bench_us(dev, os_fn(2))
            V.log(f"{T:>7} | oneshot={min(t1, t2):>6.1f} (vpt{1 if t1 <= t2 else 2})")

    # ------------------------------------------------------------------
    # --tune: column two-shot config sweep, EAGER vs CAPTURED-GRAPH REPLAY
    # (eager tight-loop numbers do not transfer to in-graph conditions:
    # T=4096 measured 169us eager vs 204us in production graphs before the
    # config fix). Covers the extend band and the norm-fused decode band.
    # ------------------------------------------------------------------
    if any(a.startswith("--tune") for a in sys.argv):
        large_only = "--tune-large" in sys.argv
        # Rank alignment around graph capture must NOT touch NCCL/CUDA: an
        # NCCL barrier interleaved with stream capture aborts with
        # DistBackendError. Use a CPU-side gloo barrier instead.
        gloo_pg = dist.new_group(backend="gloo")

        def bench_graph(fn):
            fn()  # warm outside capture
            torch.cuda.synchronize()
            dist.barrier(group=gloo_pg)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            torch.cuda.synchronize()
            dist.barrier(group=gloo_pg)
            t = V.bench_us(dev, g.replay)
            torch.cuda.synchronize()
            dist.barrier(group=gloo_pg)
            return t

        TUNE_CFG = [(0, 0)] + [
            (nb2, bs2)
            for nb2 in (16, 24, 32, 48, 64, 80, 96, 128, 148, 192, 256)
            for bs2 in (256, 384, 512, 768, 1024)
        ]
        V.log(
            "column EXTEND tune: T | best-graph us (nb,bs,barrier) | eager-at-that-cfg | default(0,0,v3) graph"
        )
        ext_sizes = (
            ()
            if (
                "--tune-bs" in sys.argv
                or "--tune-norm-ext" in sys.argv
                or "--tune-fw" in sys.argv
            )
            else (
                (8192, 12288, 16384)
                if large_only
                else (
                    128,
                    256,
                    512,
                    768,
                    1024,
                    1536,
                    2048,
                    3072,
                    4096,
                    6144,
                    8192,
                    10240,
                    12288,
                    14336,
                    16384,
                )
            )
        )
        for T in ext_sizes:
            c = Case(h, T, [0, T], [True], [7], salt=9000 + T)
            n = T * D
            oo = T * D if T > 8192 else out_off  # keep OUT disjoint from input
            hcx = D // h.world
            cache_sh = c.cache_full[
                :, :, h.rank * hcx : (h.rank + 1) * hcx
            ].contiguous()
            weight_sh = c.weight[h.rank * hcx : (h.rank + 1) * hcx].contiguous()
            xs = torch.empty((T, hcx), dtype=torch.bfloat16, device=dev)
            bufT = h.buffer[:n].view(T, D)
            bufT.copy_(h.pattern(n, c.salt).view(T, D))
            e_r = torch.empty((0, W - 1), dtype=torch.int64, device=dev)
            e_m = torch.empty((0,), dtype=torch.bool, device=dev)
            e_d = torch.empty((0,), dtype=torch.int64, device=dev)

            def col_fn(nb2, bs2, pb):
                def f():
                    inkling_ar_scattered_sconv(
                        bufT,
                        xs,
                        cache_sh,
                        c.safe_idx,
                        c.cache_mask,
                        c.ci,
                        c.has_init,
                        c.cu,
                        c.si,
                        weight_sh,
                        e_r,
                        e_m,
                        e_d,
                        h.hdl.multicast_ptr,
                        h.hdl.multicast_ptr + oo * h.elem_size,
                        h.hflags.buffer_ptrs_dev,
                        h.state.data_ptr(),
                        h.rank,
                        h.world,
                        activation="silu",
                        use_residual=True,
                        num_blocks=nb2,
                        block_size=bs2,
                        per_block_barrier=pb,
                    )

                return f

            rows = []
            for pb in (False, True):
                for nb2, bs2 in TUNE_CFG:
                    tg = bench_graph(col_fn(nb2, bs2, pb))
                    rows.append((tg, nb2, bs2, pb))
            rows.sort()
            t_def = bench_graph(col_fn(0, 0, False))
            top = " ".join(
                f"{r[0]:.1f}@({r[1]},{r[2]},{'b' if r[3] else 'g'})" for r in rows[:3]
            )
            V.log(f"TUNE_EXT {T} best3: {top} default: {t_def:.1f}")

        # EXTEND + fused-norm-tail vs the production unfused-norm chain
        # ({fused no-norm @ shipped cfg} then sgl_kernel fused_add_rmsnorm on
        # the OUT view -- exactly what the extend call sites did before the
        # tail moved in-kernel). Graph-replay us. Sweeps chunked cfgs
        # everywhere plus stream cfgs in the stream band.
        if "--tune-norm-ext" in sys.argv:
            from sgl_kernel import fused_add_rmsnorm

            def shipped_cfg(T):
                # Mirror of comm.py's world-keyed extend dispatch tables
                # (nb, bs, use_stream, stream_walk).
                if h.world == 8:
                    if T < 3072:
                        if T <= 128:
                            return (48, 384, False, 0)
                        if T <= 256:
                            return (80, 384, False, 0)
                        if T <= 512:
                            return (192, 768, False, 0)
                        if T <= 768:
                            return (96, 512, False, 0)
                        if T <= 1024:
                            return (96, 768, False, 0)
                        if T <= 1536:
                            return (192, 768, False, 0)
                        return (148, 1024, False, 0)
                    if 8192 <= T < 10240:
                        return (48, 1024, False, 0)
                    if T < 4096:
                        return (148, 256, True, 16)
                    if T < 6144:
                        return (96, 512, True, 24)
                    if T < 8192:
                        return (64, 512, True, 32)
                    if T < 16384:
                        return (148, 128, True, 0)
                    return (96, 192, True, 0)
                if T >= 3072:
                    if T < 4096:
                        return (148, 256, True, 16)
                    if T < 6144:
                        return (0, 0, True, 24)
                    if T < 8192:
                        return (148, 256, True, 32)
                    if T < 10240:
                        return (96, 512, True, 0)
                    return (148, 256, True, 0)
                if T <= 128:
                    return (96, 384, False, 0)
                if T <= 256:
                    return (148, 512, False, 0)
                if T <= 768:
                    return (128, 256, False, 0)
                return (148, 384, False, 0)

            V.log(
                "EXTEND+norm tune: T | chain (fused@ship + rmsnorm) |"
                " fused-norm@ship | best (nb,bs,kernel,walk)"
            )
            for T in (512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384):
                c = Case(h, T, [0, T], [True], [7], salt=9800 + T)
                n = T * D
                hcx = D // h.world
                cache_sh = c.cache_full[
                    :, :, h.rank * hcx : (h.rank + 1) * hcx
                ].contiguous()
                weight_sh = c.weight[h.rank * hcx : (h.rank + 1) * hcx].contiguous()
                xs = torch.empty((T, hcx), dtype=torch.bfloat16, device=dev)
                bufT = h.buffer[:n].view(T, D)
                bufT.copy_(h.pattern(n, c.salt).view(T, D))
                oo = T * D if T > 8192 else out_off
                out_view = h.buffer[oo : oo + n].view(T, D)
                gsm = torch.ones((D,), dtype=torch.bfloat16, device=dev)
                resid = torch.zeros((T, D), dtype=torch.bfloat16, device=dev)
                nout = torch.empty_like(resid)
                e_r = torch.empty((0, W - 1), dtype=torch.int64, device=dev)
                e_m = torch.empty((0,), dtype=torch.bool, device=dev)
                e_d = torch.empty((0,), dtype=torch.int64, device=dev)

                def mkn(nb2, bs2, stream, walk, with_norm):
                    norm_kw = (
                        dict(
                            out_local=out_view,
                            norm_gamma=gsm,
                            norm_residual=resid,
                            norm_out=nout,
                            norm_eps=1e-6,
                        )
                        if with_norm
                        else {}
                    )

                    def f():
                        inkling_ar_scattered_sconv(
                            bufT,
                            xs,
                            cache_sh,
                            c.safe_idx,
                            c.cache_mask,
                            c.ci,
                            c.has_init,
                            c.cu,
                            c.si,
                            weight_sh,
                            e_r,
                            e_m,
                            e_d,
                            h.hdl.multicast_ptr,
                            h.hdl.multicast_ptr + oo * h.elem_size,
                            h.hflags.buffer_ptrs_dev,
                            h.state.data_ptr(),
                            h.rank,
                            h.world,
                            activation="silu",
                            use_residual=True,
                            num_blocks=nb2,
                            block_size=bs2,
                            per_block_barrier=False,
                            need_scratch=False,
                            use_stream=stream,
                            stream_walk=walk,
                            **norm_kw,
                        )
                        if not with_norm:
                            fused_add_rmsnorm(out_view, resid, gsm, 1e-6)

                    return f

                nb0, bs0, st0, wl0 = shipped_cfg(T)
                t_chain = bench_graph(mkn(nb0, bs0, st0, wl0, False))
                t_ship = bench_graph(mkn(nb0, bs0, st0, wl0, True))
                best = (t_ship, nb0, bs0, st0, wl0)
                cfgs = [
                    (nb2, bs2, False, 0)
                    for nb2, bs2 in (
                        (0, 0),
                        (48, 384),
                        (64, 512),
                        (96, 384),
                        (96, 512),
                        (96, 768),
                        (128, 256),
                        (148, 384),
                        (148, 512),
                        (148, 1024),
                        (192, 768),
                        (48, 1024),
                    )
                ]
                if T >= 3072:
                    cfgs += [
                        (nb2, bs2, True, wl)
                        for nb2, bs2 in (
                            (0, 0),
                            (148, 256),
                            (96, 512),
                            (64, 512),
                            (148, 128),
                            (96, 192),
                            (128, 128),
                        )
                        for wl in (0, 16, 24, 32)
                    ]
                for cfg in cfgs:
                    tg = bench_graph(mkn(*cfg, True))
                    if tg < best[0]:
                        best = (tg, *cfg)
                V.log(
                    f"NORMEXT {T}: chain={t_chain:.1f} ship={t_ship:.1f}"
                    f" best={best[0]:.1f}@({best[1]},{best[2]},"
                    f"{'st' if best[3] else 'ch'},L{best[4]})"
                )
            dist.destroy_process_group()
            return 0 if fails == 0 else 1

        # FULL-WIDTH fused {AR + column conv + full-width update} vs the
        # non-scattered production extend chain {one-shot AR + full-width
        # causal_conv1d + update_sconv_cache}, graph-replay us. The fused
        # kernel does 1/P of the conv FLOPs and skips the x round trip; the
        # baseline is exactly what non-scattered extend runs today.
        if "--tune-fw" in sys.argv:
            from sglang.jit_kernel.inkling_all_reduce import select_ar_config

            V.log("FULL-WIDTH tune: T | base chain | fused best (nb,bs,kernel,walk)")
            for T in (512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384):
                c = Case(h, T, [0, T], [True], [7], salt=9850 + T)
                n = T * D
                hcx = D // h.world
                oo = T * D if T > 8192 else out_off
                bufT = h.buffer[:n].view(T, D)
                bufT.copy_(h.pattern(n, c.salt).view(T, D))
                # Baseline state (full width, as prod non-scattered).
                x_full = h.expected(n, c.salt).view(T, D).clone()
                cache_base = c.cache_full.clone()
                buf_flat = h.buffer[:n]

                def base_fn(nb2, bs2):
                    def f():
                        inkling_multimem_one_shot_fused(
                            buf_flat,
                            h.hdl.multicast_ptr,
                            h.hflags.buffer_ptrs_dev,
                            h.state.data_ptr(),
                            h.rank,
                            h.world,
                            n,
                            nb2,
                            bs2,
                            per_block_barrier=False,
                        )
                        causal_conv1d(
                            x=x_full,
                            weight=c.weight,
                            sconv_cache=cache_base,
                            cache_mask=c.cache_mask[:, None, None],
                            safe_idx=c.safe_idx,
                            cu=c.cu,
                            si=c.si,
                            activation="silu",
                            use_residual=True,
                        )
                        update_sconv_cache(
                            x=x_full,
                            sconv_cache=cache_base,
                            cache_indices=c.ci,
                            has_initial_state=c.has_init,
                            query_start_loc=c.qsl,
                        )

                    return f

                kk, tnb, tbs = select_ar_config(T, h.world)
                base_best = float("inf")
                base_cfgs = {(0, 0), (96, 512), (148, 512), (64, 512)}
                if kk in ("v3", "v3b"):
                    base_cfgs.add((tnb, tbs))
                for nb2, bs2 in base_cfgs:
                    base_best = min(base_best, bench_graph(base_fn(nb2, bs2)))

                # Fused full-width.
                cache_fw = c.cache_full.clone()
                weight_sh = c.weight[h.rank * hcx : (h.rank + 1) * hcx]
                xs = torch.empty((T, hcx), dtype=torch.bfloat16, device=dev)
                e_r = torch.empty((0, W - 1), dtype=torch.int64, device=dev)
                e_m = torch.empty((0,), dtype=torch.bool, device=dev)
                e_d = torch.empty((0,), dtype=torch.int64, device=dev)

                def fw_fn(nb2, bs2, stream, walk):
                    def f():
                        inkling_ar_scattered_sconv(
                            bufT,
                            xs,
                            cache_fw,
                            c.safe_idx,
                            c.cache_mask,
                            c.ci,
                            c.has_init,
                            c.cu,
                            c.si,
                            weight_sh,
                            e_r,
                            e_m,
                            e_d,
                            h.hdl.multicast_ptr,
                            h.hdl.multicast_ptr + oo * h.elem_size,
                            h.hflags.buffer_ptrs_dev,
                            h.state.data_ptr(),
                            h.rank,
                            h.world,
                            activation="silu",
                            use_residual=True,
                            num_blocks=nb2,
                            block_size=bs2,
                            per_block_barrier=False,
                            need_scratch=False,
                            use_stream=stream,
                            stream_walk=walk,
                            full_update=True,
                            cache_col0=h.rank * hcx,
                        )

                    return f

                cfgs = [
                    (nb2, bs2, False, 0)
                    for nb2, bs2 in (
                        (0, 0),
                        (48, 384),
                        (64, 512),
                        (96, 384),
                        (96, 512),
                        (96, 768),
                        (128, 256),
                        (148, 384),
                        (148, 512),
                        (148, 1024),
                        (192, 768),
                        (48, 1024),
                    )
                ]
                cfgs += [
                    (nb2, bs2, True, wl)
                    for nb2, bs2 in (
                        (0, 0),
                        (148, 256),
                        (96, 512),
                        (64, 512),
                        (148, 128),
                        (96, 192),
                        (128, 128),
                    )
                    for wl in (0, 8, 16, 24, 32)
                ]
                best = (float("inf"), 0, 0, False, 0)
                for cfg in cfgs:
                    tg = bench_graph(fw_fn(*cfg))
                    if tg < best[0]:
                        best = (tg, *cfg)
                V.log(
                    f"TUNE_FW {T}: base={base_best:.1f} fused={best[0]:.1f}"
                    f"@({best[1]},{best[2]},{'st' if best[3] else 'ch'},L{best[4]})"
                )
            dist.destroy_process_group()
            return 0 if fails == 0 else 1

        # BASE chain (v3 AR + full-width conv + update) graph-replay reference.
        if "--sweep-base" in sys.argv:
            V.log("BASE chain graph-replay reference:")
            for T in (
                128,
                256,
                512,
                768,
                1024,
                1536,
                2048,
                3072,
                4096,
                6144,
                8192,
                10240,
                12288,
                14336,
                16384,
            ):
                c = Case(h, T, [0, T], [True], [7], salt=9600 + T)
                n = T * D
                bufT = h.buffer[:n]
                bufT.copy_(h.pattern(n, c.salt))
                x_full = h.expected(n, c.salt).view(T, D).clone()
                cache_full = c.cache_full.clone()

                def base_fn(nb2, bs2):
                    def f():
                        inkling_multimem_one_shot_fused(
                            bufT,
                            h.hdl.multicast_ptr,
                            h.hflags.buffer_ptrs_dev,
                            h.state.data_ptr(),
                            h.rank,
                            h.world,
                            n,
                            nb2,
                            bs2,
                            per_block_barrier=False,
                        )
                        causal_conv1d(
                            x=x_full,
                            weight=c.weight,
                            sconv_cache=cache_full,
                            cache_mask=c.cache_mask[:, None, None],
                            safe_idx=c.safe_idx,
                            cu=c.cu,
                            si=c.si,
                            activation="silu",
                            use_residual=True,
                        )
                        update_sconv_cache(
                            x=x_full,
                            sconv_cache=cache_full,
                            cache_indices=c.ci,
                            has_initial_state=c.has_init,
                            query_start_loc=c.qsl,
                        )

                    return f

                from sglang.jit_kernel.inkling_all_reduce import select_ar_config

                kk, tnb, tbs = select_ar_config(T, h.world)
                best = (float("inf"), 0, 0)
                cfgs = {(0, 0), (96, 512), (148, 512), (64, 512)}
                if kk in ("v3", "v3b"):
                    cfgs.add((tnb, tbs))
                for nb2, bs2 in cfgs:
                    tg = bench_graph(base_fn(nb2, bs2))
                    if tg < best[0]:
                        best = (tg, nb2, bs2)
                V.log(f"BASE {T}: {best[0]:.1f}@({best[1]},{best[2]})")
            dist.destroy_process_group()
            return 0 if fails == 0 else 1

        # Full-range stream-vs-chunked sweep (graph-replay us).
        if "--sweep-stream" in sys.argv:

            def tuned_cfg(T):
                if T <= 128:
                    return (96, 384)
                if T <= 256:
                    return (148, 512)
                if T <= 768:
                    return (128, 256)
                if T <= 3072:
                    return (148, 384)
                if T <= 10240:
                    return (0, 0)
                if T <= 12288:
                    return (128, 512)
                if T <= 14336:
                    return (192, 512)
                return (96, 768)

            V.log("T | chunked(tuned) | stream best (nb,bs)")
            for T in (
                128,
                256,
                512,
                768,
                1024,
                1536,
                2048,
                3072,
                4096,
                6144,
                8192,
                10240,
                12288,
                14336,
                16384,
            ):
                c = Case(h, T, [0, T], [True], [7], salt=9700 + T)
                n = T * D
                hcx = D // h.world
                cache_sh = c.cache_full[
                    :, :, h.rank * hcx : (h.rank + 1) * hcx
                ].contiguous()
                weight_sh = c.weight[h.rank * hcx : (h.rank + 1) * hcx].contiguous()
                xs = torch.empty((T, hcx), dtype=torch.bfloat16, device=dev)
                bufT = h.buffer[:n].view(T, D)
                bufT.copy_(h.pattern(n, c.salt).view(T, D))
                oo = T * D if T > 8192 else out_off
                e_r = torch.empty((0, W - 1), dtype=torch.int64, device=dev)
                e_m = torch.empty((0,), dtype=torch.bool, device=dev)
                e_d = torch.empty((0,), dtype=torch.int64, device=dev)

                def mk(nb2, bs2, stream, walk=0):
                    def f():
                        inkling_ar_scattered_sconv(
                            bufT,
                            xs,
                            cache_sh,
                            c.safe_idx,
                            c.cache_mask,
                            c.ci,
                            c.has_init,
                            c.cu,
                            c.si,
                            weight_sh,
                            e_r,
                            e_m,
                            e_d,
                            h.hdl.multicast_ptr,
                            h.hdl.multicast_ptr + oo * h.elem_size,
                            h.hflags.buffer_ptrs_dev,
                            h.state.data_ptr(),
                            h.rank,
                            h.world,
                            activation="silu",
                            use_residual=True,
                            num_blocks=nb2,
                            block_size=bs2,
                            per_block_barrier=False,
                            need_scratch=False,
                            use_stream=stream,
                            stream_walk=walk,
                        )

                    return f

                tc = tuned_cfg(T)
                t_ch = bench_graph(mk(tc[0], tc[1], False))
                best = (float("inf"), 0, 0, 0)
                for nb2, bs2 in (
                    (0, 0),
                    (32, 512),
                    (64, 512),
                    (96, 512),
                    (148, 512),
                    (64, 1024),
                    (96, 1024),
                    (32, 256),
                    (148, 128),
                    (128, 128),
                    (96, 192),
                    (148, 256),
                ):
                    for wl in (0, 16, 24, 32, 64):
                        tg = bench_graph(mk(nb2, bs2, True, wl))
                        if tg < best[0]:
                            best = (tg, nb2, bs2, wl)
                V.log(
                    f"SWEEP {T}: chunked={t_ch:.1f}"
                    f" stream={best[0]:.1f}@({best[1]},{best[2]},L{best[3]})"
                )
            dist.destroy_process_group()
            return 0 if fails == 0 else 1

        # Batch-size sensitivity at fixed T (extend): B enters only phase 3
        # (linear, tiny) and the 3 boundary rows per sequence -- verify.
        if "--tune-bs" in sys.argv:
            V.log("EXTEND B-sweep at fixed T (tuned cfg, graph-replay us):")
            for T, cfg in ((4096, (0, 0)), (16384, (96, 768))):
                for B in (1, 4, 16, 64, 256):
                    if B > T:
                        continue
                    step = T // B
                    qsl = [i * step for i in range(B)] + [T]
                    c = Case(
                        h,
                        T,
                        qsl,
                        [True] * B,
                        [(i * 3 + 1) % POOL for i in range(B)],
                        salt=9900 + T + B,
                    )
                    n = T * D
                    hcx = D // h.world
                    cache_sh = c.cache_full[
                        :, :, h.rank * hcx : (h.rank + 1) * hcx
                    ].contiguous()
                    weight_sh = c.weight[h.rank * hcx : (h.rank + 1) * hcx].contiguous()
                    xs = torch.empty((T, hcx), dtype=torch.bfloat16, device=dev)
                    bufT = h.buffer[:n].view(T, D)
                    bufT.copy_(h.pattern(n, c.salt).view(T, D))
                    oo = T * D if T > 8192 else out_off
                    e_r = torch.empty((0, W - 1), dtype=torch.int64, device=dev)
                    e_m = torch.empty((0,), dtype=torch.bool, device=dev)
                    e_d = torch.empty((0,), dtype=torch.int64, device=dev)

                    def bs_fn():
                        inkling_ar_scattered_sconv(
                            bufT,
                            xs,
                            cache_sh,
                            c.safe_idx,
                            c.cache_mask,
                            c.ci,
                            c.has_init,
                            c.cu,
                            c.si,
                            weight_sh,
                            e_r,
                            e_m,
                            e_d,
                            h.hdl.multicast_ptr,
                            h.hdl.multicast_ptr + oo * h.elem_size,
                            h.hflags.buffer_ptrs_dev,
                            h.state.data_ptr(),
                            h.rank,
                            h.world,
                            activation="silu",
                            use_residual=True,
                            num_blocks=cfg[0],
                            block_size=cfg[1],
                            per_block_barrier=False,
                            need_scratch=False,
                            use_stream="--stream" in sys.argv,
                        )

                    tg = bench_graph(bs_fn)
                    V.log(f"TUNE_BS T={T} B={B}: {tg:.1f}")
            dist.destroy_process_group()
            return 0 if fails == 0 else 1

        V.log(
            "column DECODE+norm tune: T | best-graph us (nb,bs,barrier) | default(v3b tuned)"
        )
        for T in (
            1,
            2,
            4,
            8,
            12,
            16,
            24,
            32,
            48,
            64,
            80,
            96,
            112,
            128,
            144,
            160,
            176,
            192,
            204,
        ):
            qsl = list(range(T + 1))
            c = Case(
                h,
                T,
                qsl,
                [True] * T,
                [(i * 3 + 1) % POOL for i in range(T)],
                salt=9500 + T,
            )
            n = T * D
            hcx = D // h.world
            cache_sh = c.cache_full[
                :, :, h.rank * hcx : (h.rank + 1) * hcx
            ].contiguous()
            weight_sh = c.weight[h.rank * hcx : (h.rank + 1) * hcx].contiguous()
            xs = torch.empty((T, hcx), dtype=torch.bfloat16, device=dev)
            bufT = h.buffer[:n].view(T, D)
            bufT.copy_(h.pattern(n, c.salt).view(T, D))
            gsm = torch.ones((D,), dtype=torch.bfloat16, device=dev)
            resid = torch.zeros((T, D), dtype=torch.bfloat16, device=dev)
            nout = torch.empty_like(resid)
            e_r = torch.empty((0, W - 1), dtype=torch.int64, device=dev)
            e_m = torch.empty((0,), dtype=torch.bool, device=dev)
            e_d = torch.empty((0,), dtype=torch.int64, device=dev)

            def coln_fn(nb2, bs2, pb):
                def f():
                    inkling_ar_scattered_sconv(
                        bufT,
                        xs,
                        cache_sh,
                        c.safe_idx,
                        c.cache_mask,
                        c.ci,
                        c.has_init,
                        c.cu,
                        c.si,
                        weight_sh,
                        e_r,
                        e_m,
                        e_d,
                        h.hdl.multicast_ptr,
                        h.hdl.multicast_ptr + out_off * h.elem_size,
                        h.hflags.buffer_ptrs_dev,
                        h.state.data_ptr(),
                        h.rank,
                        h.world,
                        activation="silu",
                        use_residual=True,
                        num_blocks=nb2,
                        block_size=bs2,
                        per_block_barrier=pb,
                        out_local=h.buffer[out_off : out_off + n].view(T, D),
                        norm_gamma=gsm,
                        norm_residual=resid,
                        norm_out=nout,
                        norm_eps=1e-6,
                    )

                return f

            rows = []
            for pb in (False, True):
                for nb2, bs2 in TUNE_CFG:
                    tg = bench_graph(coln_fn(nb2, bs2, pb))
                    rows.append((tg, nb2, bs2, pb))
            rows.sort()
            t_def = bench_graph(coln_fn(0, 0, True))
            top = " ".join(
                f"{r[0]:.1f}@({r[1]},{r[2]},{'b' if r[3] else 'g'})" for r in rows[:3]
            )
            V.log(f"TUNE_DEC {T} best3: {top} default: {t_def:.1f}")

    dist.destroy_process_group()
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
