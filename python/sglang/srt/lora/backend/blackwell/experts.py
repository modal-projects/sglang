"""Fused routed-expert multi-LoRA for MXFP4 experts (DeepSeek-V4.1) on the TRT-LLM layout.

Per decode batch and MoE layer (``--lora-backend blackwell`` with routed LoRA targets):

    route            local (token, k) routes -> 16-row expert blocks (texperts_prep)
    quant + shrink   MXFP8 of the routed rows; gate/up LoRA-A for each (expert, slot) in a block
    gate/up kernel   MXFP4 GEMM + LoRA-B K-tiles, clamped SwiGLU, MXFP8 for the down GEMM,
                     down LoRA-A partials (texperts, epi "swiglu")
    down kernel      MXFP4 GEMM + LoRA-B K-tiles (texperts, epi "down")
    finalize         per token: sum of its routes weighted by the top-k weights

Adapters stay BF16. The base weights are the ones the TRT-LLM MoE already holds; the LoRA-B
buffers are mirrored once per adapter load into the shuffled row order of those weights.
Batches it does not take (prefill, more than MAX_TOKENS tokens, unexpected layouts) return
None, and the caller runs the existing path.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

MAX_TOKENS = int(os.environ.get("SGLANG_BLACKWELL_MOE_LORA_MAX_TOKENS", "256"))
NT = 16

_LUTS = {}
_KERNELS = {}
_PREPS = {}
_MIRRORED = []  # layer states holding LoRA-B mirrors
_VERSION = [None]
_REJECTED = set()
# True while lora_dispatch warms the decomposed path: forward() then declines every batch
FORCE_DECOMPOSED = [False]
# The MoE block's shared expert while the block hands it to the fused path (one forward; GLM's
# FP8 path), else None.
SHARED_BLOCK = [None]


def _fold_shared():
    """Fold GLM-5.3 Flash's small TP8 FP8 shared expert into the routed kernels (as expert E);
    SGLANG_BLACKWELL_MOE_SHARED_FOLD=0 runs it separately."""
    return os.environ.get("SGLANG_BLACKWELL_MOE_SHARED_FOLD", "1") == "1"


def enabled():
    flag = os.environ.get("SGLANG_BLACKWELL_MOE_LORA", "1")
    if flag == "0":
        return False
    if flag == "force":  # tests without a server context
        return True
    try:
        from sglang.srt.runtime_context import get_lora
        return get_lora().lora_backend == "blackwell"
    except Exception:
        return False


_TRACED = set()


def trace_once(*key):
    if key not in _TRACED:
        _TRACED.add(key)
        logger.info("blackwell MoE LoRA trace: %s", key)


def _reject(reason):
    if reason not in _REJECTED:
        _REJECTED.add(reason)
        logger.info("blackwell MoE LoRA: using the decomposed path (%s)", reason)
    return None


def _luts(I, H, device):
    key = (I, H, str(device))
    if key not in _LUTS:
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )
        orig = torch.cat([torch.arange(I, 2 * I), torch.arange(I)])
        p13 = _maybe_get_cached_w3_w1_permute_indices(
            {}, torch.zeros(2 * I, H // 2, dtype=torch.uint8), 128).cpu().long()
        full13 = orig[p13]
        feat = full13 % I
        tile = torch.arange(2 * I) // 128
        local = feat - 64 * tile
        assert int(local.min()) >= 0 and int(local.max()) < 64, "gate/up shuffle left its tile"
        lut13 = (((full13 >= I).long() << 8) | local).int()
        p2 = get_w2_permute_indices_with_cache(
            {}, torch.zeros(H, I // 2, dtype=torch.uint8), 128).cpu().long()
        assert bool(((p2 // 128) == (torch.arange(H) // 128)).all()), "down shuffle left its tile"
        _LUTS[key] = (lut13.to(device), full13.to(device), p2.int().to(device))
    return _LUTS[key]


def _row_e8(base, rows, cols):
    """[rows, cols] per-row, per-32-K UE8M0 exponents of a 128x128 block FP8 linear with
    UE8M0 scales."""
    from ..blackwell_backend import block_fp8_e8
    e8 = block_fp8_e8(base)
    if e8 is None or tuple(e8.shape) != (rows, cols):
        raise ValueError("shared expert scales are not UE8M0 block FP8")
    return e8


class _SharedMX:
    """GLM's TP FP8 shared expert (TP slice of Is features) as texperts expert E: gate/up
    weights in the routed experts' tile order (tile t = features [64t, 64t + 64) of gate and
    of up, rows as lut13 of tile t), down weights K-padded to 128 and rows in the routed down
    order (p2), with their UE8M0 scales re-swizzled. Built once per layer from the model's own
    tensors (which stay in place for the stock path)."""

    def __init__(self, block, lut13, p2, I, H, device):
        from ..blackwell_backend import _swizzle_sf
        se = block.shared_experts
        self.gu, self.dn = se.gate_up_proj, se.down_proj
        wg = self.gu.base_layer.weight
        wd = self.dn.base_layer.weight
        Is = wg.shape[0] // 2
        self.Is = Is
        self.mts = -(-Is // 64)  # gate/up row tiles
        self.Kd = -(-Is // 128) * 128  # down K, padded
        assert self.mts * 128 <= 2 * I and tuple(wd.shape) == (H, Is)
        info = lut13[: self.mts * 128].long()
        f = 64 * (torch.arange(self.mts * 128, device=device) // 128) + (info & 255)
        valid = f < Is
        src = torch.where(info >= 256, Is + f, f)[valid]
        u8 = torch.uint8
        sfg = _row_e8(self.gu.base_layer, 2 * Is, H // 32)
        w = torch.zeros(self.mts * 128, H, dtype=u8, device=device)
        sf = torch.full((self.mts * 128, H // 32), 127, dtype=u8, device=device)
        w[valid] = wg.view(u8)[src]
        sf[valid] = sfg[src]
        self.w_gu, self.sf_gu = w.reshape(-1), _swizzle_sf(sf)
        sfd = _row_e8(self.dn.base_layer, H, Is // 32)
        w = torch.zeros(H, self.Kd, dtype=u8, device=device)
        sf = torch.full((H, self.Kd // 32), 127, dtype=u8, device=device)
        w[:, :Is] = wd.view(u8)
        sf[:, : Is // 32] = sfd
        pl = p2.long()
        self.w_d, self.sf_d = w[pl].reshape(-1).contiguous(), _swizzle_sf(sf[pl].contiguous())
        self.rows_valid, self.src = valid, src
        self.a2s = None
        self.limit = float(se.swiglu_limit) if se.swiglu_limit is not None else 1e30

    def refresh(self, bm_gu, bm_d, E, I, H, SL, p2):
        """Shared rows of the LoRA-B mirrors and the K-padded down LoRA-A (dense buffers are
        unscaled and may hold garbage past each adapter's rank: made finite here, masked
        by rank in the kernels)."""
        Is = self.Is
        nan0 = lambda t: torch.nan_to_num(t.float(), 0.0, 0.0, 0.0)
        bs = nan0(self.gu.lora_b)  # [SL, 2Is, R]
        rows = self.mts * 128
        dst = bm_gu.view(-1, SL, 64)[E * 2 * I: E * 2 * I + rows]
        b = bs[:, self.src].permute(1, 0, 2).to(torch.bfloat16)  # [rows valid, SL, R]
        up = (self.src >= Is).view(-1, 1, 1)
        full = torch.zeros(rows, SL, 64, dtype=torch.bfloat16, device=bm_gu.device)
        full[self.rows_valid] = torch.cat([torch.where(up, 0, b), torch.where(up, b, 0)], -1)
        dst.copy_(full)
        bds = nan0(self.dn.lora_b)  # [SL, H, R]
        bm_d.view(-1, SL, 32)[E * H:(E + 1) * H] = \
            bds[:, p2.long()].permute(1, 0, 2).to(torch.bfloat16)
        a = nan0(self.dn.lora_a).to(torch.bfloat16)  # [SL, R, Is]
        if self.a2s is None:
            self.a2s = torch.zeros(SL * 32, self.Kd, dtype=torch.bfloat16, device=a.device)
        self.a2s[:, :Is] = a.reshape(SL * 32, Is)


class _LayerState:
    """LoRA-B mirrors of one MoE layer in the shuffled row order of its MXFP4 weights: gate/up
    dense 32 wide (each row's own B_gate or B_up row, texperts gu32), down [H, slots * 32]."""

    def __init__(self, lora_info, E, off, I, H, full13, p2, device):
        self.lora_info = lora_info
        self.E, self.off, self.I, self.H = E, off, I, H
        self.full13, self.p2 = full13, p2
        SL = lora_info.gate_up_lora_b_weights.shape[0]
        self.SL = SL
        self.bm_gu = torch.empty(E * 2 * I, SL * 32, dtype=torch.bfloat16, device=device)
        self.bm_d = torch.empty(E * H, SL * 32, dtype=torch.bfloat16, device=device)
        self.refresh()

        self.clamp = None  # set by forward() on the eager call
        self.ranks = torch.full((SL,), 32, dtype=torch.int32, device=device)
        self.scales = torch.ones(SL, dtype=torch.float32, device=device)

    def _local(self, t):
        """[slots, experts, ...] -> this rank's experts (EP-local or global buffers)."""
        if t.shape[1] == 1:  # shared-outer: one tensor for every expert
            return t.expand(t.shape[0], self.E, *t.shape[2:])
        if t.shape[1] != self.E:
            return t[:, self.off:self.off + self.E]
        return t

    def refresh(self):
        li = self.lora_info
        b_gu = self._local(li.gate_up_lora_b_weights)  # [SL, E, 2I, R], rows [gate; up]
        b_d = self._local(li.down_lora_b_weights)  # [SL, E, H, R]
        I, E, SL = self.I, self.E, self.SL
        for e0 in range(0, E, 8):  # bounded temporaries
            e1 = min(E, e0 + 8)
            # each row's own B row (gate rows: B_gate, up rows: B_up)
            self.bm_gu.view(-1, 2 * I, SL, 32)[e0:e1] = \
                b_gu[:, e0:e1].permute(1, 2, 0, 3)[:, self.full13]
            self.bm_d.view(-1, self.H, SL, 32)[e0:e1] = \
                b_d[:, e0:e1].permute(1, 2, 0, 3)[:, self.p2.long()]


def refresh_mirrors(version):
    """Called by the blackwell backend whenever the memory pool loaded or cleared a slot."""
    if version == _VERSION[0]:
        return
    _VERSION[0] = version
    for st in _MIRRORED:
        st.refresh()


def _kernels(E, I, H, MB, SL, clamp, device, EA):
    key = (E, I, H, MB, SL, clamp, EA)
    if key not in _KERNELS:
        from . import texperts
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        gu = texperts.TExperts(E, 2 * I, H, MB, n_slots=SL, w_slice=64, epi="swiglu",
                               clamp=clamp, zfix=True, sms=sms, fold_a=True, a_experts=EA,
                               zd_fix=True)
        assert gu.gu32
        dn = texperts.TExperts(E, H, I, MB, n_slots=SL, w_slice=32, epi="down", zfix=True,
                               sms=sms, lora32=True)
        _KERNELS[key] = dict(gu=gu, dn=dn, gu_fn=None, dn_fn=None)
    return _KERNELS[key]


def _prep(T_max, K, E, off, H, I, SL, R, device, mx=True, shared=False):
    key = (T_max, K, E, off, H, I, SL, R, mx, shared)
    if key not in _PREPS:
        from .texperts_prep import Prep
        _PREPS[key] = Prep(T_max, K, E, off, H, I, SL, R, device, mx=mx, shared=shared)
    return _PREPS[key]


def _launch(k, name, op, refs):
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    args = [from_dlpack(t, assumed_align=16) for t in refs]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if k[name] is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("blackwell MoE LoRA: kernel compile requested during capture")
        import time
        t0 = time.perf_counter()
        k[name] = cute.compile(op, *args, stream)
        logger.info("blackwell MoE LoRA: compiled %r in %.1fs", op, time.perf_counter() - t0)
    k[name](*args, stream)


def forward(hidden_states, topk_ids, topk_weights, quant_info, lora_info, base_layer, output):
    """Fused routed-expert LoRA into ``output`` [T, H]; returns it, or None to fall back."""
    T, H = hidden_states.shape
    if FORCE_DECOMPOSED[0]:
        return None
    if T == 0 or T > MAX_TOKENS:
        return _reject(f"T={T}")
    if hidden_states.dtype != torch.bfloat16:
        return _reject(f"hidden {hidden_states.dtype}")
    R = lora_info.max_lora_rank
    if R != 32:
        return _reject(f"rank {R}")
    a_gu = lora_info.gate_up_lora_a_weights
    SL = a_gu.shape[0]
    if SL > 8:
        return _reject(f"{SL} slots")
    I = quant_info.intermediate_size_per_partition
    E = quant_info.local_num_experts
    off = quant_info.local_expert_offset
    K = topk_ids.shape[1]
    if (2 * I) % 128 or H % 512 or I % 128:
        return _reject(f"H={H} I={I}")
    w13 = quant_info.w13_weight
    if w13.dtype != torch.uint8 or tuple(w13.shape) != (E, 2 * I, H // 2):
        return _reject(f"w13 {w13.dtype} {tuple(w13.shape)}")
    dev = hidden_states.device
    lut13, full13, p2 = _luts(I, H, dev)
    st = getattr(base_layer, "_blackwell_moe_lora", None)
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            return _reject("first use during graph capture")
        if lora_info.down_lora_a_weights.shape[1] != E:
            return _reject("routed LoRA buffers span every expert (expected EP-local)")
        cl = quant_info.gemm1_clamp_limit
        clamp = float(cl.flatten()[0]) if cl is not None else 1e30
        st = _LayerState(lora_info, E, off, I, H, full13, p2, dev)
        st.clamp = clamp
        st.dummies = (torch.zeros(16, device=dev),
                      torch.zeros(1, 16, dtype=torch.bfloat16, device=dev),
                      torch.zeros(1, 16, dtype=torch.uint8, device=dev),
                      torch.zeros(16, dtype=torch.uint8, device=dev),
                      torch.zeros(1, 16, dtype=torch.int64, device=dev))
        base_layer._blackwell_moe_lora = st
        _MIRRORED.append(st)
    st.lora_info = lora_info  # buffers are stable; keep the latest handle
    prep = _prep(MAX_TOKENS, K, E, off, H, I, SL, R, dev)
    EA = a_gu.shape[1]  # experts in the gate/up A buffer (1: shared-outer)
    k = _kernels(E, I, H, prep.MB, SL, st.clamp, dev, EA)
    if k.get("dn_fold") is None:  # placeholders for the down kernel's fold_a inputs
        if torch.cuda.is_current_stream_capturing():
            return _reject("first use during graph capture")
        from .texperts import fold_refs
        k["dn_fold"] = fold_refs(k["dn"], dev)
    a_d = lora_info.down_lora_a_weights.reshape(SL * E * R, I)
    # LoRA-B buffers are pre-scaled and zero past each adapter's rank: unit scales, full rank
    ranks, scales = st.ranks, st.scales
    slots = lora_info.token_lora_mapping[:T].to(torch.int32)
    prep.route(topk_ids, topk_weights, slots, fold=True)
    # MXFP8 + BF16 rows; the gate/up LoRA-A shrink runs inside the gate/up kernel (fold_a)
    prep.quant(hidden_states)
    w2 = quant_info.w2_weight
    gs13 = quant_info.w13_weight_scale
    gs2 = quant_info.w2_weight_scale
    u8 = lambda t: t.view(torch.uint8).reshape(-1)
    dummy_f, d_bf, d_u8, d_u8f, d_i64 = st.dummies
    _launch(k, "gu_fn", k["gu"], [
        u8(w13), u8(gs13), u8(prep.xq), u8(prep.x_sf), st.bm_gu.unsqueeze(-1), prep.z,
        prep.rslot, prep.blk_expert, prep.mask_gu, ranks, scales, dummy_f, lut13, a_d,
        prep.aq, u8(prep.a_sf), prep.zd, prep.nblk, prep.x_bf,
        a_gu.reshape(-1, H), prep.a_blk, prep.a_s, prep.n_a, prep.npairs, prep.zcnt]
        + k["dn_fold"][7:] + [prep.rtok])
    _launch(k, "dn_fn", k["dn"], [
        u8(w2), u8(gs2), u8(prep.aq), u8(prep.a_sf), st.bm_d.unsqueeze(-1), prep.zd,
        prep.rslot, prep.blk_expert, prep.mask_gu, ranks, scales, prep.y, p2, d_bf, d_u8, d_u8f,
        d_i64, prep.nblk] + k["dn_fold"] + [prep.rtok])
    prep.finalize(T, topk_weights, output)
    return output


# ----------------------------------------------------------------------------- BF16 (Qwen MoE)


class _LayerStateBF:
    """One BF16 MoE layer ([gate; up] experts) on the fused path: LoRA-B mirrors gate/up rows
    [2, E, I] and down rows [E, H], slots side by side."""

    def __init__(self, layer, E, I, H, device):
        self.layer = layer
        self.E, self.I, self.H = E, I, H
        SL = layer.gate_up_lora_b_weights.shape[0]
        self.SL = SL
        self.bm_pair = torch.zeros(2 * E * I, SL * 32, dtype=torch.bfloat16, device=device)
        self.bm_d = torch.zeros(E * H, SL * 32, dtype=torch.bfloat16, device=device)
        self.lut = torch.arange(H, dtype=torch.int32, device=device)
        self.ones = torch.ones(SL, dtype=torch.float32, device=device)
        self.full = torch.full((SL,), 32, dtype=torch.int32, device=device)
        self.refresh()

    def _experts(self, t):
        return t.expand(t.shape[0], self.E, *t.shape[2:]) if t.shape[1] == 1 else t

    def refresh(self):
        E, I, SL = self.E, self.I, self.SL
        b_gu = self._experts(self.layer.gate_up_lora_b_weights)  # [SL, E, 2I, R]
        b_d = self._experts(self.layer.down_lora_b_weights)  # [SL, E, H, R]
        pair = self.bm_pair.view(2, E, I, SL, 32)
        dmir = self.bm_d.view(E, self.H, SL, 32)
        for e0 in range(0, E, 32):
            e1 = min(E, e0 + 32)
            src = b_gu[:, e0:e1].reshape(SL, e1 - e0, 2, I, 32).permute(2, 1, 3, 0, 4)
            pair[:, e0:e1] = src.reshape(2, e1 - e0, I, SL, 32)
            dmir[e0:e1] = b_d[:, e0:e1].permute(1, 2, 0, 3)


# persistent CTAs of the BF16 expert kernels (B300, 148 SMs: 104-128 all beat 148)
BF16_GRID = 116


def _routed_scale(base):
    """The Triton runner multiplies its summed expert output by the runner config's
    routed_scaling_factor (unless the factor was folded into the top-k weights)."""
    rc = getattr(base, "moe_runner_config", None)
    f = getattr(rc, "routed_scaling_factor", None)
    return 1.0 if f is None else float(f)


def _kernels_bf(E, I, H, MB, SL, EA, device, clamp=0.0):
    """clamp: the experts' swiglu limit (gate <= L, |up| <= L before SiLU(gate) * up), 0 = none."""
    key = ("bf", E, I, H, MB, SL, EA, clamp)
    if key not in _KERNELS:
        from . import texperts
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        gu = texperts.TExperts(E, 2 * I, H, MB, base="bf16", n_slots=SL, w_slice=32,
                               epi="swiglu_bf", clamp=clamp, zfix=True, fold_a=True,
                               a_experts=EA, zd_fix=True, pair=True, sms=sms, lora32=True,
                               a_ks=1, order="e", grid_cap=BF16_GRID)
        assert gu.xg  # gate/up gathers its rows from x by token index
        dn = texperts.TExperts(E, H, I, MB, base="bf16", n_slots=SL, w_slice=32, epi="down",
                               zfix=True, sms=sms, lora32=True, order="e", grid_cap=BF16_GRID)
        _KERNELS[key] = dict(gu=gu, dn=dn, gu_fn=None, dn_fn=None, dn_fold=None, gu_fold=None)
    return _KERNELS[key]


def forward_bf16(dispatch_output, lora_layer, token_slots):
    """Fused routed-expert LoRA for a BF16 [gate; up] MoE (FusedMoEWithLoRA, TP1/EP1): the
    routed [T, H] output, or None to fall back."""
    base = lora_layer.base_layer
    x = dispatch_output.hidden_states
    topk = dispatch_output.topk_output
    T, H = x.shape
    if T == 0 or T > MAX_TOKENS:
        return _reject(f"T={T}")
    w13 = base.w13_weight
    if w13.dtype != torch.bfloat16 or w13.dim() != 3:
        return _reject(f"w13 {w13.dtype}")
    E, I2, Hw = w13.shape
    I = I2 // 2
    if Hw != H or I % 128 or H % 128 or E != getattr(base, "num_local_experts", E):
        return _reject(f"E={E} I={I} H={H}")
    # interleaved gate/up only applies to gpt-oss style layers (gemm1_alpha); the wrapper knows
    if getattr(lora_layer, "_uses_interleaved_gate_up", False):
        return _reject("interleaved gate/up")
    a_gu = lora_layer.gate_up_lora_a_weights
    R = a_gu.shape[2] // 2
    SL = a_gu.shape[0]
    if R != 32 or SL > 8 or lora_layer.down_lora_a_weights.shape[1] != E:
        return _reject(f"rank {R} slots {SL}")
    dev = x.device
    st = getattr(lora_layer, "_blackwell_moe_lora", None)
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            return _reject("first use during graph capture")
        st = _LayerStateBF(lora_layer, E, I, H, dev)
        lora_layer._blackwell_moe_lora = st
        _MIRRORED.append(st)
    K = topk.topk_ids.shape[1]
    prep = _prep(MAX_TOKENS, K, E, 0, H, I, SL, R, dev, mx=False)
    EA = a_gu.shape[1]
    rc = getattr(base, "moe_runner_config", None)
    if rc is not None and (getattr(rc, "activation", "silu") != "silu"
                           or not getattr(rc, "is_gated", True)
                           or getattr(rc, "gemm1_alpha", None) is not None):
        return _reject(f"activation {getattr(rc, 'activation', None)}")
    lim = getattr(rc, "swiglu_limit", None) or getattr(rc, "gemm1_clamp_limit", None)
    if isinstance(lim, torch.Tensor):
        lim = float(lim.flatten()[0])
    k = _kernels_bf(E, I, H, prep.MB, SL, EA, dev, float(lim or 0.0))
    if k["dn_fold"] is None:
        if torch.cuda.is_current_stream_capturing():
            return _reject("first use during graph capture")
        from .texperts import fold_refs
        k["dn_fold"] = fold_refs(k["dn"], dev)
        k["gu_fold"] = fold_refs(k["gu"], dev)
    ranks, scales = st.full, st.ones
    slots = token_slots[:T].to(torch.int32)
    if not x.is_contiguous():  # gate/up gathers its rows from x itself (no permuted copy)
        x = x.contiguous()
    prep.route(topk.topk_ids, topk.topk_weights, slots, fold=True, zero_zd=True)
    a_d = lora_layer.down_lora_a_weights.reshape(SL * E * R, I)
    f32d = torch.zeros(16, device=dev) if not hasattr(st, "dummy") else st.dummy
    st.dummy = f32d
    w13a, w2a = base.w13_weight, base.w2_weight
    _launch(k, "gu_fn", k["gu"], [
        w13a.reshape(-1, H).unsqueeze(-1), prep.x_sf, x.reshape(-1), prep.x_sf,
        st.bm_pair.unsqueeze(-1), prep.z, prep.rslot, prep.blk_expert, prep.mask_gu, ranks,
        scales, f32d, st.lut, a_d, prep.aq, prep.a_sf, prep.zd, prep.nblk, x,
        a_gu.reshape(-1, H), prep.a_blk, prep.a_s, prep.n_a, prep.npairs, prep.zcnt]
        + k["gu_fold"][7:] + [prep.rtok])
    _launch(k, "dn_fn", k["dn"], [
        w2a.reshape(-1, I).unsqueeze(-1), prep.x_sf, prep.aq.reshape(-1), prep.x_sf,
        st.bm_d.unsqueeze(-1), prep.zd, prep.rslot, prep.blk_expert, prep.mask_gu, ranks,
        scales, prep.y, st.lut, a_d, prep.aq, prep.a_sf, prep.zd, prep.nblk]
        + k["dn_fold"] + [prep.rtok])
    out = torch.empty_like(x)
    prep.finalize(T, topk.topk_weights, out, rscale=_routed_scale(base))
    return out


# ------------------------------------------------------------ FP8 block-scaled (GLM-5.3 Flash)
def _ue8m0_requant_(w, s):
    """In place: FP8 block weights [E, N, K] with FP32 128x128 scales -> power-of-two (UE8M0)
    scales, values requantized, as SGLang does for DeepGEMM on Blackwell. The tensors stay the
    layer's own, so the Triton fallback (prefill) computes with the same weights."""
    E, N, K = w.shape
    for e in range(E):
        q = w[e].view(N // 128, 128, K // 128, 128).float()
        sc = s[e].float()
        pow2 = torch.frexp(sc)[0] == 0.5
        if bool(pow2.all()):
            continue
        dq = (q * sc[:, None, :, None]).to(torch.bfloat16).float()
        amax = dq.abs().amax(dim=(1, 3)).clamp(min=1e-10)
        new = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
        w[e].copy_((dq / new[:, None, :, None]).to(torch.float8_e4m3fn).view(N, K))
        s[e].copy_(new.to(s.dtype))


def shared_fold_ok_fp8b(block, x):
    """Whether a DeepSeek-style MoE block on the FP8-block fused path (GLM-5.3 Flash) can
    hand its TP shared expert (UE8M0 block FP8, dense LoRA) to the routed kernels for x."""
    if not enabled() or not _fold_shared():
        return False
    T = x.shape[0]
    if T == 0 or T > MAX_TOKENS or x.dtype != torch.bfloat16:
        return False
    se = getattr(block, "shared_experts", None)
    if se is None or getattr(block, "_shared_expert_tp1", True) \
            or getattr(block, "num_fused_shared_experts", 1):
        return False
    ok = getattr(block, "_blackwell_shared_ok", None)
    if ok is None:
        from ..blackwell_backend import block_fp8_e8
        try:
            gu, dn = se.gate_up_proj, se.down_proj
            ok = (getattr(gu, "lora_a", None) is not None
                  and getattr(dn, "lora_a", None) is not None
                  and gu.lora_a.shape[1] == 64 and dn.lora_a.shape[1] == 32
                  and block_fp8_e8(gu.base_layer) is not None
                  and block_fp8_e8(dn.base_layer) is not None
                  and gu.base_layer.weight.shape[0] % 128 == 0
                  and dn.base_layer.weight.shape[1] % 128 == 0
                  and getattr(gu, "lora_backend", None) is not None)
        except Exception:
            ok = False
        block._blackwell_shared_ok = bool(ok)
    return block._blackwell_shared_ok


class _LayerStateFP8:
    """One FP8 block-scaled MoE layer on the fused path: UE8M0 scale mirrors per row and 32 K
    (gate/up rows in the kernel's tile order: 64 gate rows then 64 up rows per tile), and the
    LoRA-B mirrors in the same row order (EP-local LoRA buffers, pre-scaled B)."""

    def __init__(self, layer, E, I, H, device, block=None):
        from ..blackwell_backend import _swizzle_sf
        base = layer.base_layer
        self.layer = layer
        self.E, self.I, self.H = E, I, H
        _ue8m0_requant_(base.w13_weight.data, base.w13_weight_scale_inv.data)
        _ue8m0_requant_(base.w2_weight.data, base.w2_weight_scale_inv.data)
        r = torch.arange(2 * I, device=device)
        t, mm = r // 128, r % 128
        self.perm = torch.where(mm < 64, 64 * t + mm, I + 64 * t + mm - 64)
        self.lut13 = (((self.perm >= I).long() << 8) | (mm % 64)).int()
        self.p2 = torch.arange(H, dtype=torch.int32, device=device)

        def sf_rows(sc, rows, kdim):
            e8 = (torch.log2(sc.float()).round() + 127).to(torch.uint8)
            return e8[:, rows // 128].repeat_interleave(4, dim=2).reshape(-1, kdim // 32)
        self.sf13 = _swizzle_sf(sf_rows(base.w13_weight_scale_inv, self.perm, H).contiguous())
        self.sf2 = _swizzle_sf(sf_rows(base.w2_weight_scale_inv,
                                       torch.arange(H, device=device), I).contiguous())
        SL = layer.gate_up_lora_b_weights.shape[0]
        self.SL = SL
        # the folded shared expert (expert E): weights / scales in the tile order, its rows of
        # the LoRA-B mirrors after the routed experts'
        self.shared = _SharedMX(block, self.lut13, self.p2, I, H, device) \
            if block is not None else None
        ES = E + (1 if block is not None else 0)
        self.bm_gu = torch.empty(ES * 2 * I, SL * 64, dtype=torch.bfloat16, device=device)
        self.bm_d = torch.empty(ES * H, SL * 32, dtype=torch.bfloat16, device=device)
        self.ranks = torch.full((SL,), 32, dtype=torch.int32, device=device)
        self.scales = torch.ones(SL, dtype=torch.float32, device=device)
        self.refresh()

    def _experts(self, t):
        return t.expand(t.shape[0], self.E, *t.shape[2:]) if t.shape[1] == 1 else t

    def refresh(self):
        E, I, SL = self.E, self.I, self.SL
        b_gu = self._experts(self.layer.gate_up_lora_b_weights)  # [SL, E, 2I, R]
        b_d = self._experts(self.layer.down_lora_b_weights)  # [SL, E, H, R]
        is_up = (self.perm >= I).view(1, -1, 1, 1)
        for e0 in range(0, E, 8):
            e1 = min(E, e0 + 8)
            src = b_gu[:, e0:e1].permute(1, 2, 0, 3)[:, self.perm]  # [e, 2I, SL, R]
            dst = self.bm_gu.view(-1, 2 * I, SL, 64)[e0:e1]
            dst[..., :32] = torch.where(is_up, 0, src)
            dst[..., 32:] = torch.where(is_up, src, 0)
            self.bm_d.view(-1, self.H, SL, 32)[e0:e1] = b_d[:, e0:e1].permute(1, 2, 0, 3)
        if getattr(self, "shared", None) is not None:
            self.shared.refresh(self.bm_gu, self.bm_d, E, I, self.H, SL, self.p2)


def _kernels_fp8(E, I, H, MB, SL, EA, clamp, device, sh=None):
    """sh: (gate/up row tiles, down K) of a folded shared expert (expert E), or None."""
    key = ("fp8b", E, I, H, MB, SL, EA, clamp, sh)
    if key not in _KERNELS:
        from . import texperts
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        ES, se = (E + 1, E) if sh else (E, -1)
        # gate/up LoRA-A shrink inside the kernel (A-items). A separate Triton shrink is faster
        # in isolation, but with the folded shared expert its dense LoRA-A costs ~1.5 ms/step
        # at batch 64 (serving), vs ~0.15 ms for the A-items
        gu = texperts.TExperts(ES, 2 * I, H, MB, base="mxfp8", n_slots=SL, w_slice=64,
                               epi="swiglu", clamp=clamp, zfix=True, sms=sms, fold_a=True,
                               a_experts=EA, zd_fix=True, a_ks=2, shared_e=se, a2_experts=E,
                               m_tiles_sh=sh[0] if sh else None)
        dn = texperts.TExperts(ES, H, I, MB, base="mxfp8", n_slots=SL, w_slice=32, epi="down",
                               zfix=True, sms=sms, lora32=True, shared_e=se,
                               k_tiles_sh=sh[1] // 128 if sh else None)
        _KERNELS[key] = dict(gu=gu, dn=dn, gu_fn=None, dn_fn=None, dn_fold=None)
    return _KERNELS[key]


def forward_fp8b(dispatch_output, lora_layer, token_slots):
    """Fused routed-expert LoRA for FP8 [gate; up] experts with 128x128 block scales (Triton
    runner, EP-local expert ids): [T, H] partial output of this rank's experts, or None."""
    base = lora_layer.base_layer
    x = dispatch_output.hidden_states
    topk = dispatch_output.topk_output
    T, H = x.shape
    if T == 0 or T > MAX_TOKENS:
        return _reject(f"T={T}")
    w13 = base.w13_weight
    s13 = getattr(base, "w13_weight_scale_inv", None)
    if w13.dtype != torch.float8_e4m3fn or s13 is None or s13.dtype != torch.float32:
        return _reject(f"w13 {w13.dtype}")
    E, I2, Hw = w13.shape
    I = I2 // 2
    if Hw != H or I % 128 or H % 512 or tuple(s13.shape) != (E, I2 // 128, H // 128):
        return _reject(f"fp8 block layout E={E} I={I} H={H} s={tuple(s13.shape)}")
    rc = getattr(base, "moe_runner_config", None)
    if rc is not None and (getattr(rc, "activation", "silu") != "silu"
                           or not getattr(rc, "is_gated", True)
                           or getattr(rc, "gemm1_alpha", None) is not None):
        return _reject(f"activation {getattr(rc, 'activation', None)}")
    if getattr(lora_layer, "_uses_interleaved_gate_up", False):
        return _reject("interleaved gate/up")
    a_gu = lora_layer.gate_up_lora_a_weights
    R = a_gu.shape[2] // 2
    SL = a_gu.shape[0]
    if R != 32 or SL > 8 or lora_layer.down_lora_a_weights.shape[1] != E:
        return _reject(f"rank {R} slots {SL} experts {lora_layer.down_lora_a_weights.shape}")
    dev = x.device
    block = SHARED_BLOCK[0]
    st = getattr(lora_layer, "_blackwell_moe_lora", None)
    if st is not None and (st.shared is None) != (block is None):
        if torch.cuda.is_current_stream_capturing():
            return _reject("shared-expert fold changed during graph capture")
        _MIRRORED.remove(st)
        st = None
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            return _reject("first use during graph capture")
        st = _LayerStateFP8(lora_layer, E, I, H, dev, block)
        lora_layer._blackwell_moe_lora = st
        _MIRRORED.append(st)
    lim = getattr(rc, "swiglu_limit", None) or getattr(rc, "gemm1_clamp_limit", None)
    if isinstance(lim, torch.Tensor):
        lim = float(lim.flatten()[0])
    clamp = float(lim) if lim else 1e30
    sh = st.shared
    if sh is not None and abs(sh.limit - clamp) > 1e-6 * max(1.0, abs(clamp)):
        return _reject(f"shared swiglu limit {sh.limit} != routed {clamp}")
    K = topk.topk_ids.shape[1]
    prep = _prep(MAX_TOKENS, K, E, 0, H, I, SL, R, dev, shared=sh is not None)
    EA = a_gu.shape[1]
    k = _kernels_fp8(E, I, H, prep.MB, SL, EA, clamp, dev, (sh.mts, sh.Kd) if sh else None)
    if k["dn_fold"] is None:
        if torch.cuda.is_current_stream_capturing():
            return _reject("first use during graph capture")
        from .texperts import fold_refs
        k["dn_fold"] = fold_refs(k["dn"], dev)
    if not x.is_contiguous():
        x = x.contiguous()
    slots = token_slots[:T].to(torch.int32)
    prep.route(topk.topk_ids, topk.topk_weights, slots, fold=True)
    prep.quant(x)
    a_d = lora_layer.down_lora_a_weights.reshape(SL * E * R, I)
    u8 = lambda t: t.view(torch.uint8).reshape(-1)
    if not hasattr(st, "dummies"):
        st.dummies = (torch.zeros(16, device=dev),
                      torch.zeros(1, 16, dtype=torch.bfloat16, device=dev),
                      torch.zeros(1, 16, dtype=torch.uint8, device=dev),
                      torch.zeros(16, dtype=torch.uint8, device=dev),
                      torch.zeros(1, 16, dtype=torch.int64, device=dev))
    dummy_f, d_bf, d_u8, d_u8f, d_i64 = st.dummies
    ranks, scales = st.ranks, st.scales
    tail_gu, tail_d = k["dn_fold"][7:], k["dn_fold"]
    if sh is not None:  # dense LoRA of the shared expert: this batch's per-slot rank / scaling
        bi = sh.gu.lora_backend.batch_info
        ranks, scales = bi.lora_ranks.to(torch.int32), bi.scalings.float()
        tail_gu = [sh.a2s, sh.gu.lora_a.reshape(-1, H), sh.w_gu, sh.sf_gu]
        tail_d = k["dn_fold"][:9] + [sh.w_d, sh.sf_d]
    _launch(k, "gu_fn", k["gu"], [
        u8(w13), st.sf13, u8(prep.xq), u8(prep.x_sf), st.bm_gu.unsqueeze(-1), prep.z,
        prep.rslot, prep.blk_expert, prep.mask_gu, ranks, scales, dummy_f, st.lut13, a_d,
        prep.aq, u8(prep.a_sf), prep.zd, prep.nblk, prep.x_bf,
        a_gu.reshape(-1, H), prep.a_blk, prep.a_s, prep.n_a, prep.npairs, prep.zcnt]
        + tail_gu + [prep.rtok])
    _launch(k, "dn_fn", k["dn"], [
        u8(base.w2_weight), st.sf2, u8(prep.aq), u8(prep.a_sf), st.bm_d.unsqueeze(-1), prep.zd,
        prep.rslot, prep.blk_expert, prep.mask_gu, ranks, scales, prep.y, st.p2, d_bf,
        d_u8, d_u8f, d_i64, prep.nblk] + tail_d + [prep.rtok])
    out = torch.empty_like(x)
    # routed experts scaled as the Triton runner does; the shared expert's row added as is
    prep.finalize(T, topk.topk_weights, out, shared=sh is not None, rscale=_routed_scale(base))
    return out
