"""Blackwell (SM100/SM103) fused multi-LoRA backend: ``--lora-backend blackwell``.

For decode-shaped batches (one token per request, up to 256 tokens) every dense LoRA linear
runs as one CuTe DSL kernel (``blackwell/tlinear.py``) that computes the base GEMM and the
per-token LoRA delta together:

  * LoRA-A CTAs compute Z = A_slot x for all adapter slots, reduce their K-splits over a
    cluster and publish each token's own-slot rows;
  * base CTAs stream the weight once with N = all tokens and append the LoRA-B K-tiles
    (B_slot against a per-token Z block) to the same accumulator;
  * split-K partials are reduced over DSMEM inside a cluster, in a fixed order (results are
    bitwise reproducible for a given batch size).

BF16 bases and MXFP8 bases (FP8 weight + swizzled UE8M0 scales, FlashInfer layout) are
supported; gate/up (two equal slices) pairs both halves in one CTA. Everything else - prefill,
larger batches, other quantizations, ranks above 32, layers with bias - runs through the
inherited ChunkedSGMV path unchanged.

Two fused layer entry points extend the same kernel's epilogue (BF16 bases):
  * ``forward_gate_up_swiglu``: gate/up + LoRA, then silu(gate) * up;
  * ``forward_attention_inputs``: QKV + LoRA, then q/k RMSNorm, NeoX RoPE (partial rotary and
    multi-axis RoPE included), the gated-query split, and the K/V write into the KV cache.

LoRA-B is read from a per-layer [N, slots*32] copy of the pool buffer, refreshed when the
memory pool reports new weights (``sync_lora_weights``).
"""

import logging
import os

import torch

from sglang.srt.lora.backend.chunked_backend import MIN_CHUNK_SIZE, ChunkedSgmvLoRABackend
from sglang.srt.lora.dense.plan import DenseLoraKind
from sglang.srt.lora.utils import LoRABatchInfo, MoELoRABatchInfo, generate_sequence_lengths
from sglang.srt.runtime_context import LoRABatchLayout, get_forward

logger = logging.getLogger(__name__)

MAX_TOKENS = int(os.environ.get("SGLANG_BLACKWELL_LORA_MAX_TOKENS", "256"))
RANK = 32


def _is_blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _mxfp8_base(base) -> bool:
    from sglang.srt.layers.quantization.mxfp8_input import accepts_mxfp8_swizzled_input

    return (
        accepts_mxfp8_swizzled_input(base)
        and getattr(base, "weight_scale_inv_swizzled", None) is not None
        and base.weight.dtype == torch.float8_e4m3fn
    )


def block_fp8_e8(base):
    """Per row, per 32 K UE8M0 exponents [N, K/32] (uint8) of a 128x128 block-FP8 linear whose
    scales are UE8M0: SGLang's DeepGEMM requant on Blackwell (packed int32, 4 K-groups per
    word, MN major) or power-of-two FP32 block scales. None when they are not UE8M0."""
    w = getattr(base, "weight", None)
    s = getattr(base, "weight_scale_inv", None)
    if w is None or s is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2:
        return None
    N, K = w.shape
    kg = -(-K // 128)
    if s.dtype == torch.int32 and getattr(s, "format_ue8m0", False) and s.dim() == 2:
        # a fresh row-major copy: DeepGEMM keeps a TMA-aligned stride even on a size-1 dim
        e8 = s.new_empty(s.shape).copy_(s).view(torch.uint8).reshape(s.shape[0], -1)
        if e8.shape[0] < N or e8.shape[1] < kg:
            return None
        e8 = e8[:N, :kg]
    elif s.dtype == torch.float32 and tuple(s.shape) == (-(-N // 128), kg):
        if not bool((torch.frexp(s)[0] == 0.5).all()):
            return None
        e8 = ((s.contiguous().view(torch.int32) >> 23) & 255).to(torch.uint8)
        e8 = e8.repeat_interleave(128, dim=0)[:N]
    else:
        return None
    return e8.repeat_interleave(4, dim=1)[:, : K // 32].contiguous()


def _block_fp8_sf(base):
    """Swizzled MX scales (per row, per 32 K) of a UE8M0 block-FP8 linear (block_fp8_e8),
    cached on the layer (the weights are read in place); None when not UE8M0."""
    cached = getattr(base, "_bw_mx_sf", None)
    if cached is not None:
        return cached
    e8 = block_fp8_e8(base)
    if e8 is None:
        return None
    N, K = base.weight.shape
    rows, cols = -(-N // 128) * 128, -(-(K // 32) // 4) * 4
    sf = torch.full((rows, cols), 127, dtype=torch.uint8, device=e8.device)
    sf[:N, : K // 32] = e8
    base._bw_mx_sf = _swizzle_sf(sf)
    return base._bw_mx_sf


def _swizzle_sf(sf):
    """[R, C] UE8M0 (R % 128 == 0, C % 4 == 0) -> FlashInfer's 128x4 swizzled layout (flat)."""
    R, C = sf.shape
    return sf.view(R // 128, 4, 32, C // 4, 4).permute(0, 3, 2, 1, 4).reshape(-1).contiguous()


def _unswizzle_sf(sf, rows, cols):
    """Inverse of _swizzle_sf for a [rows, cols] matrix stored padded to 128 x 4 atoms."""
    R, C = -(-rows // 128) * 128, -(-cols // 4) * 4
    flat = sf.reshape(-1).view(torch.uint8)[: R * C]
    return flat.view(R // 128, C // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(R, C)[:rows, :cols]


class _SplitPiece:
    pass


class _SplitState:
    """Per-layer copies for _fused_split: slice weights (K padded), re-swizzled scales, and
    the slice's LoRA A [slots, 32, Kp] / B mirror [rows, slots * 32] (refreshed per adapter
    load)."""

    def __init__(self, base, rows):
        w = base.weight
        self.N, self.K = w.shape
        self.Kp = -(-self.K // 128) * 128
        kc, kcp = self.K // 32, self.Kp // 32
        sf_lin = _unswizzle_sf(base.weight_scale_inv_swizzled, self.N, kc)
        self.pieces = []
        sizes = [b - a for a, b in zip((0,) + tuple(rows[:-1]), rows)]
        # merged: every slice padded to a 128-row multiple inside one weight copy, so a single
        # standard multi-slice launch covers them; the output columns are gathered back
        self.merged = len(rows) > 1 and len(set(sizes)) == 1 and sizes[0] % 128 != 0
        if self.merged:
            n = sizes[0]
            npad = -(-n // 128) * 128
            pc = _SplitPiece()
            pc.r0, pc.r1, pc.n_pad = 0, self.N, npad * len(rows)
            pc.slice_n, pc.slice_pad = n, npad
            wp = torch.zeros(pc.n_pad, self.Kp, dtype=torch.uint8, device=w.device)
            sfp = torch.full((pc.n_pad, kcp), 127, dtype=torch.uint8, device=w.device)
            for i in range(len(rows)):
                wp[i * npad:i * npad + n, : self.K] = w[i * n:(i + 1) * n].view(torch.uint8)
                sfp[i * npad:i * npad + n, :kc] = sf_lin[i * n:(i + 1) * n]
            pc.w = wp.view(torch.float8_e4m3fn)
            pc.sf = _swizzle_sf(sfp)
            pc.a = pc.b = None
            self.pieces.append(pc)
            self.version = None
            return
        r0 = 0
        for r1 in rows:
            pc = _SplitPiece()
            pc.r0, pc.r1, n = r0, r1, r1 - r0
            pc.n_pad = -(-n // 128) * 128
            wp = torch.zeros(n, self.Kp, dtype=torch.uint8, device=w.device)
            wp[:, : self.K] = w[r0:r1].view(torch.uint8)
            pc.w = wp.view(torch.float8_e4m3fn)
            sfp = torch.full((pc.n_pad, kcp), 127, dtype=torch.uint8, device=w.device)
            sfp[:n, :kc] = sf_lin[r0:r1]
            pc.sf = _swizzle_sf(sfp)
            pc.a = pc.b = None
            self.pieces.append(pc)
            r0 = r1
        self.lora_a = self.lora_b = None

    def bind(self, lora_a, lora_b):
        """First use: allocate the slice copies once (CUDA graphs keep these addresses)."""
        self.lora_a, self.lora_b = lora_a, lora_b
        if self.merged:
            pc = self.pieces[0]
            pc.a = torch.zeros(lora_a.shape[0], lora_a.shape[1], self.Kp, dtype=lora_a.dtype,
                               device=lora_a.device)
            pc.b = torch.zeros(pc.n_pad, lora_b.shape[0] * RANK, dtype=lora_b.dtype,
                               device=lora_b.device)
            self.refresh()
            return
        for pc in self.pieces:
            pc.a = torch.zeros(lora_a.shape[0], RANK, self.Kp, dtype=lora_a.dtype,
                               device=lora_a.device)
            pc.b = torch.empty(pc.r1 - pc.r0, lora_b.shape[0] * RANK, dtype=lora_b.dtype,
                               device=lora_b.device)
        self.refresh()

    def refresh(self):
        """In place, from the pool buffers (after an adapter load or clear)."""
        if self.merged:
            pc = self.pieces[0]
            n, npad = pc.slice_n, pc.slice_pad
            pc.a[:, :, : self.K].copy_(self.lora_a[:, :, : self.K])
            for i in range(self.N // n):
                pc.b[i * npad:i * npad + n].copy_(
                    self.lora_b[:, i * n:(i + 1) * n].permute(1, 0, 2).reshape(n, -1))
            return
        for i, pc in enumerate(self.pieces):
            pc.a[:, :, : self.K].copy_(self.lora_a[:, i * RANK:(i + 1) * RANK, : self.K])
            pc.b.copy_(self.lora_b[:, pc.r0:pc.r1].permute(1, 0, 2)
                       .reshape(pc.r1 - pc.r0, -1))


class BlackwellLoRABackend(ChunkedSgmvLoRABackend):
    name = "blackwell"

    def __init__(self, max_loras_per_batch, device, server_args):
        super().__init__(max_loras_per_batch, device, server_args)
        if not _is_blackwell():
            raise ValueError("--lora-backend blackwell requires an SM100/SM103 GPU")
        self._kernels = {}
        self._mirrored = []  # layers holding a LoRA-B mirror
        self._weights_version = None
        self._rejected = set()
        self._split_layers = []
        self._bw_global_info = None
        self._bw_graph_global = None
        self._bw_req = None

    # ------------------------------------------------------------------ DP attention
    # Under DP attention (attention TP1; LoRAManager enforces it) each rank's attention runs
    # its own requests (DP_LOCAL) and the FFN / MoE section runs every rank's tokens gathered
    # in rank order (TP_GLOBAL). ``batch_info`` resolves to the rows of the section being run,
    # so the fused kernels, the inherited csgmv path and the MoE paths read the right batch.
    supports_dp_attention = True

    @property
    def batch_info(self):
        d = self.__dict__
        g = d.get("_bw_global_info")
        if g is not None and get_forward().lora_batch_layout is LoRABatchLayout.TP_GLOBAL:
            return g
        return d.get("_bw_local_info")

    @batch_info.setter
    def batch_info(self, info):
        self.__dict__["_bw_local_info"] = info

    def get_batch_info(self, layout=None):
        if layout is None:
            return self.batch_info
        g = self.__dict__.get("_bw_global_info")
        if layout is LoRABatchLayout.TP_GLOBAL and g is not None:
            return g
        return self.__dict__.get("_bw_local_info")

    def reset_batch_state(self):
        super().reset_batch_state()
        self._bw_global_info = None

    def prepare_lora_batch(self, forward_batch, weight_indices, lora_ranks, scalings,
                           use_cuda_graph, use_prefill_cuda_graph=False):
        self._bw_global_info = None
        super().prepare_lora_batch(forward_batch, weight_indices, lora_ranks, scalings,
                                   use_cuda_graph, use_prefill_cuda_graph=use_prefill_cuda_graph)
        self._bw_req = (list(weight_indices), list(lora_ranks))

    def init_dp_attention_cuda_graph_batch_info(self, max_num_tokens: int) -> None:
        """Static TP_GLOBAL batch for decode graphs: one segment per gathered token."""
        from sglang.srt.runtime_context import get_parallel

        local = self.cuda_graph_batch_info
        T = max_num_tokens * get_parallel().attn_dp_size
        with torch.device(self.device):
            self._bw_arange = torch.arange(T + 1, dtype=torch.int32)
            self._bw_graph_global = LoRABatchInfo(
                bs=T, use_cuda_graph=True, num_segments=T, max_len=MIN_CHUNK_SIZE,
                seg_lens=None, seg_indptr=self._bw_arange.clone(),
                weight_indices=torch.zeros(T, dtype=torch.int32),
                permutation=self._bw_arange[:T].clone(), lora_ranks=local.lora_ranks,
                scalings=local.scalings, req_seg_indptr=self._bw_arange.clone(),
                req_weight_indices=torch.zeros(T, dtype=torch.int32), expected_tokens=T)

    def prepare_global_lora_batch(self, forward_batch) -> None:
        """Every DP rank's per-token adapter slots in rank order (the FFN's row order) and the
        TP_GLOBAL batch built from them: tokens sorted by adapter and chunked like csgmv's own
        batches, written into static buffers for decode graphs (slots from the manager's CPU
        gather, no GPU sync) and into fresh tensors otherwise (slots gathered on the GPU)."""
        from sglang.srt.layers.dp_attention import dp_gather_replicate, dp_slot_in

        self._bw_global_info = None
        local = self.__dict__.get("_bw_local_info")
        g_num = forward_batch.global_num_tokens_cpu
        if local is None or self._bw_req is None or g_num is None or len(g_num) <= 1:
            return
        weight_indices, lora_ranks = self._bw_req
        no_lora = next((i for i, r in enumerate(lora_ranks) if r == 0), -1)
        n_local = int(g_num[dp_slot_in(g_num)])
        if forward_batch.forward_mode.is_idle() or not weight_indices:
            tok = torch.empty(0, dtype=torch.int32)
        else:
            tok = torch.repeat_interleave(torch.tensor(weight_indices, dtype=torch.int32),
                                          generate_sequence_lengths(forward_batch, "cpu"))
        tok = tok[:n_local]
        if tok.numel() < n_local:  # graph padding rows: no adapter
            tok = torch.cat([tok, torch.full((n_local - tok.numel(),), no_lora,
                                             dtype=torch.int32)])
        Tg = int(sum(g_num))
        gg = self._bw_graph_global
        graph = local.use_cuda_graph and gg is not None and Tg <= gg.weight_indices.shape[0]
        w = gg.req_weight_indices[:Tg] if graph else torch.empty(Tg, dtype=torch.int32,
                                                                 device=self.device)
        # Decode graphs: one row per request, each rank's block padded to its token count, so
        # the gathered slots come from the manager's CPU gather of the requests' adapters (a
        # GPU gather would need a sync here to build the csgmv segments). Else gather rows.
        gs = getattr(self, "dp_gathered_slots", None)
        w_cpu = None
        if (graph and gs is not None and len(gs) == len(g_num)
                and forward_batch.forward_mode.is_decode()
                and all(len(s_) <= int(n_) for s_, n_ in zip(gs, g_num))):
            rows = []
            for s_, n_ in zip(gs, g_num):
                rows += s_ + [no_lora] * (int(n_) - len(s_))
            w_cpu = torch.tensor(rows, dtype=torch.int32).clamp_(min=0)
            w.copy_(w_cpu.pin_memory(), non_blocking=True)
        else:
            local_w = tok.pin_memory().to(self.device, non_blocking=True)
            dp_gather_replicate(w, local_w, forward_batch)
            w.clamp_(min=0)  # padding with every slot holding an adapter: any slot (unused rows)
        ranks = local.lora_ranks
        tmap = torch.where(ranks[w.long()] > 0, w, torch.full_like(w, -1))
        if graph:
            # csgmv segments: tokens sorted by adapter, chunked (one segment per token would make
            # every token reload its adapter's LoRA-A). The chunk size depends only on Tg, which
            # a captured graph fixes, so replays match the BLOCK_M the capture compiled.
            info = gg
            Tm = info.weight_indices.shape[0]
            if w_cpu is None:
                w_cpu = w.cpu()
            perm = torch.argsort(w_cpu, stable=True)
            chunk = self._determine_chunk_size_for_tokens(Tg)
            seg_w, seg_indptr = self._get_segments_info(w_cpu[perm], chunk)
            ns = len(seg_w)
            info.weight_indices[:ns].copy_(seg_w, non_blocking=True)
            info.weight_indices[ns:].zero_()
            info.seg_indptr[:ns + 1].copy_(seg_indptr, non_blocking=True)
            info.seg_indptr[ns + 1:].fill_(Tg)
            info.permutation[:Tg].copy_(perm.to(torch.int32), non_blocking=True)
            info.req_weight_indices[Tg:].zero_()
            info.req_seg_indptr.copy_(self._bw_arange)
            info.req_seg_indptr[Tg + 1:].fill_(Tg)
            info.bs = info.expected_tokens = Tg
            info.num_segments, info.max_len = ns, chunk
            if self.is_moe_lora:
                bufs = self.moe_cg_buffers
                bufs["token_lora_mapping"][:Tg].copy_(tmap)
                bufs["token_lora_mapping"][Tg:].fill_(-1)
                bufs["adapter_enabled"][:ranks.shape[0]].copy_((ranks > 0).to(torch.int32))
                info.moe_lora_info = MoELoRABatchInfo(
                    seg_indptr=info.req_seg_indptr[:Tg + 1], req_to_lora=w,
                    adapter_enabled=bufs["adapter_enabled"],
                    token_lora_mapping=bufs["token_lora_mapping"][:Tm])
        else:
            if w_cpu is None:
                w_cpu = w.cpu()
            perm = torch.argsort(w_cpu, stable=True)
            chunk = self._determine_chunk_size_for_tokens(Tg)
            seg_w, seg_indptr = self._get_segments_info(w_cpu[perm], chunk)
            dev = self.device
            info = LoRABatchInfo(
                bs=Tg, use_cuda_graph=False, num_segments=len(seg_w), max_len=chunk,
                seg_lens=None, seg_indptr=seg_indptr.to(dev, non_blocking=True),
                weight_indices=seg_w.to(dev, non_blocking=True),
                permutation=perm.to(torch.int32).to(dev, non_blocking=True),
                lora_ranks=ranks, scalings=local.scalings,
                req_seg_indptr=torch.arange(Tg + 1, dtype=torch.int32, device=dev),
                req_weight_indices=w, expected_tokens=Tg)
            if self.is_moe_lora:
                info.moe_lora_info = MoELoRABatchInfo(
                    seg_indptr=info.req_seg_indptr, req_to_lora=w,
                    adapter_enabled=(ranks > 0).to(torch.int32), token_lora_mapping=tmap)
        info.has_active_lora = any(r > 0 for r in lora_ranks)
        # graph replays pass a batch view without is_extend_in_batch (always decode)
        info.is_prefill = not graph and bool(getattr(forward_batch, "is_extend_in_batch", True))
        self._bw_global_info = info

    # ------------------------------------------------------------------ pool sync
    def sync_lora_weights(self, version) -> None:
        """Refresh every LoRA-B mirror after the memory pool loaded or cleared a slot."""
        if version == self._weights_version:
            return
        self._weights_version = version
        from sglang.srt.lora.backend.blackwell import experts as bw_experts

        bw_experts.refresh_mirrors(version)
        for st in self._split_layers:
            st.refresh()
        for layer in self._mirrored:
            b = layer.lora_b
            layer._blackwell_b_mirror.copy_(b.permute(1, 0, 2).reshape(b.shape[1], -1))

    def _b_mirror(self, layer, lora_b):
        m = getattr(layer, "_blackwell_b_mirror", None)
        if m is None:
            m = lora_b.permute(1, 0, 2).reshape(lora_b.shape[1], -1).contiguous()
            layer._blackwell_b_mirror = m
            self._mirrored.append(layer)
        return m

    # ------------------------------------------------------------------ dispatch
    def forward_with_base(self, layer, x, base_fn, lora_a, lora_b, output_offset, offsets,
                          *, all_reduce=None, kind=DenseLoraKind.LINEAR,
                          pruned_batch_info=None):
        if kind == DenseLoraKind.LINEAR and pruned_batch_info is None:
            y = self._fused(layer, x, lora_a, lora_b, offsets)
            if y is not None:
                return all_reduce(y) if all_reduce is not None else y
        return super().forward_with_base(
            layer, x, base_fn, lora_a, lora_b, output_offset, offsets,
            all_reduce=all_reduce, kind=kind, pruned_batch_info=pruned_batch_info)

    def _reject(self, layer, reason):
        key = (type(layer).__name__, reason)
        if key not in self._rejected:
            self._rejected.add(key)
            logger.info("blackwell LoRA: %s uses csgmv (%s)", type(layer).__name__, reason)
        return None

    def _decode_tokens(self, x):
        """T when ``x`` is a decode-shaped batch this backend fuses, else None."""
        info = self.batch_info
        if info is None or getattr(info, "is_prefill", False):
            return None
        if not isinstance(x, torch.Tensor) or x.dim() != 2 or x.dtype != torch.bfloat16:
            return None
        T = x.shape[0]
        if info.bs != T or not (1 <= T <= MAX_TOKENS) or info.req_weight_indices is None:
            return None
        return T

    def _slots(self, T):
        """Per-token adapter slots (int32), padded to two rows for a single token: a one-row
        TMA tensor is degenerate."""
        slots = self.batch_info.req_weight_indices[:T].to(torch.int32)
        if T == 1:
            slots = torch.cat([slots, torch.full_like(slots, -1)])
        return slots

    @staticmethod
    def _pad(x, T):
        return torch.cat([x, torch.zeros_like(x)]) if T == 1 else x

    def _fused(self, layer, x, lora_a, lora_b, offsets):
        info = self.batch_info
        T = self._decode_tokens(x)
        if T is None:
            return None
        base = layer.base_layer
        w = base.weight
        if getattr(base, "bias", None) is not None:
            return self._reject(layer, "bias")
        N, K = w.shape
        if lora_b.shape[-1] != RANK or lora_a.shape[0] > 8 or lora_a.shape[1] % RANK:
            return self._reject(layer, f"A {tuple(lora_a.shape)} B {tuple(lora_b.shape)}")
        slices = lora_a.shape[1] // RANK
        rows = tuple(int(o) for o in offsets[1:])
        if len(rows) != slices or rows[-1] != N:
            return self._reject(layer, f"slices {rows} N={N}")
        mx = w.dtype == torch.float8_e4m3fn
        if any(r % 128 for r in rows[:-1]) or N % 128 or K % 128:
            if mx and _mxfp8_base(base):
                return self._fused_split(layer, x, T, lora_a, lora_b, rows)
            return self._reject(layer, f"slices {rows} N={N} K={K}")
        w_sf = None
        if mx:
            if _mxfp8_base(base):
                w_sf = base.weight_scale_inv_swizzled
            else:
                if torch.cuda.is_current_stream_capturing() and \
                        getattr(base, "_bw_mx_sf", None) is None:
                    return self._reject(layer, "first block-FP8 use during graph capture")
                w_sf = _block_fp8_sf(base)  # UE8M0 block FP8 (DeepGEMM requant)
                if w_sf is None:
                    return self._reject(layer, "unsupported FP8 layout")
        elif w.dtype != torch.bfloat16 or not w.is_contiguous():
            return self._reject(layer, f"base {w.dtype}")
        if K % 128:
            return self._reject(layer, f"K={K}")
        kern = self._kernel(N, K, T, rows, mx, lora_a.shape[0])
        if kern is None:
            return self._reject(layer, f"no one-wave plan for N={N} K={K} T={T}")
        if not kern.compiled and torch.cuda.is_current_stream_capturing():
            return self._reject(layer, "first use during graph capture")

        x = self._pad(x.contiguous(), T)
        slots = self._slots(T)
        n_rows = x.shape[0]
        y = torch.empty(n_rows, N, device=x.device, dtype=torch.bfloat16)
        extra = {}
        if mx:
            from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

            x_q, x_sf = flashinfer_mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
            extra = dict(w_sf=w_sf, x_q=x_q, x_sf=x_sf)
        kern(w, x, lora_a, self._b_mirror(layer, lora_b), slots,
             info.lora_ranks.to(torch.int32), info.scalings.float(), y, **extra)
        return y[:T] if T != n_rows else y

    # ---------------------------------------------------- unaligned MXFP8 shapes (split / pad)
    def _fused_split(self, layer, x, T, lora_a, lora_b, rows):
        """MXFP8 projections whose slices or K are not 128-aligned (DeepSeek-V4.1's TP8
        shared expert: gate/up slices of 288 rows, down K = 288): one fused launch per slice,
        each slice's weight rows and scales re-swizzled on their own (rows past the slice are
        TMA zero fill, stores past it are clipped), and K zero-padded to a 128 multiple."""
        from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

        info = self.batch_info
        st = self._split_state(layer, lora_a, lora_b, rows)
        if st is None:
            return self._reject(layer, "first split use during graph capture")
        K, Kp = st.K, st.Kp
        if Kp != K:
            x = torch.nn.functional.pad(x, (0, Kp - K))
        x = self._pad(x.contiguous(), T)
        slots = self._slots(T)
        n_rows = x.shape[0]
        x_q, x_sf = flashinfer_mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
        y = torch.empty(n_rows, st.N, device=x.device, dtype=torch.bfloat16)
        if st.merged:
            pc = st.pieces[0]
            ns = len(rows)
            srows = tuple(pc.slice_pad * (i + 1) for i in range(ns))
            kern = self._kernel(pc.n_pad, Kp, T, srows, True, lora_a.shape[0])
            if kern is None:
                return self._reject(layer, f"no plan for merged N={pc.n_pad} K={Kp} T={T}")
            if not kern.compiled and torch.cuda.is_current_stream_capturing():
                return self._reject(layer, "first use during graph capture")
            yp = torch.empty(n_rows, pc.n_pad, device=x.device, dtype=torch.bfloat16)
            kern(pc.w, x, pc.a, pc.b, slots, info.lora_ranks.to(torch.int32),
                 info.scalings.float(), yp, w_sf=pc.sf, x_q=x_q, x_sf=x_sf)
            y = yp.view(n_rows, ns, pc.slice_pad)[:, :, : pc.slice_n].reshape(n_rows, st.N)
            return y[:T] if T != n_rows else y
        kerns = []
        for pc in st.pieces:
            kern = self._kernel(pc.n_pad, Kp, T, (pc.n_pad,), True, lora_a.shape[0])
            if kern is None:
                return self._reject(layer, f"no plan for split N={pc.n_pad} K={Kp} T={T}")
            if not kern.compiled and torch.cuda.is_current_stream_capturing():
                return self._reject(layer, "first use during graph capture")
            kerns.append(kern)
        for pc, kern in zip(st.pieces, kerns):
            kern(pc.w, x, pc.a, pc.b, slots, info.lora_ranks.to(torch.int32),
                 info.scalings.float(), y[:, pc.r0:pc.r1], w_sf=pc.sf, x_q=x_q, x_sf=x_sf)
        return y[:T] if T != n_rows else y

    def _split_state(self, layer, lora_a, lora_b, rows):
        st = getattr(layer, "_blackwell_split", None)
        if st is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            st = _SplitState(layer.base_layer, rows)
            st.bind(lora_a, lora_b)
            layer._blackwell_split = st
            self._split_layers.append(st)
        return st

    # A model's MLP may hand both projections to forward_mlp_mx (DeepSeek-V4.1's shared expert)
    takes_mlp_mx = True

    def forward_mlp_mx(self, gu, dn, x, limit):
        """down(clamped_swiglu(gate_up(x))) for an MLP whose MXFP8 gate/up slices and down K
        are not 128-aligned (DeepSeek-V4.1's TP8 shared expert: 288 features): the merged
        gate/up launch, one kernel for the activation (padded K, BF16 + MXFP8: mx_act), and
        the down launch. None when the layers or the batch do not fit (the caller runs the
        unfused path)."""
        T = self._decode_tokens(x)
        if T is None:
            return None
        bg, bd = gu.base_layer, dn.base_layer
        if not (_mxfp8_base(bg) and _mxfp8_base(bd)):
            return None
        if getattr(bg, "bias", None) is not None or getattr(bd, "bias", None) is not None:
            return None
        if getattr(bd, "reduce_results", True) or not getattr(bd, "input_is_parallel", False) \
                or hasattr(bd, "lora_input_transform"):
            return None
        a1, b1, a2, b2 = gu.lora_a, gu.lora_b, dn.lora_a, dn.lora_b
        for a, b, sl in ((a1, b1, 2), (a2, b2, 1)):
            if b.shape[-1] != RANK or a.shape[0] > 8 or a.shape[1] != sl * RANK:
                return None
        N, K = bg.weight.shape
        Nd, Kd = bd.weight.shape
        rows = tuple(int(o) for o in gu.lora_offsets[1:])
        rows_d = tuple(int(o) for o in dn.lora_offsets[1:])
        Is = N // 2
        if rows != (Is, N) or Is % 128 == 0 or Kd != Is or rows_d != (Nd,) or Nd % 128 \
                or K % 128:
            return None
        st = self._split_state(gu, a1, b1, rows)
        std = self._split_state(dn, a2, b2, rows_d)
        if st is None or std is None or not st.merged or std.merged or len(std.pieces) != 1:
            return None
        pc, pd = st.pieces[0], std.pieces[0]
        Kp = std.Kp
        srows = tuple(pc.slice_pad * (i + 1) for i in range(2))
        kg = self._kernel(pc.n_pad, K, T, srows, True, a1.shape[0])
        kd = self._kernel(pd.n_pad, Kp, T, (pd.n_pad,), True, a2.shape[0])
        if kg is None or kd is None:
            return None
        if (not kg.compiled or not kd.compiled) and torch.cuda.is_current_stream_capturing():
            return None
        from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

        from .blackwell.mx_act import swiglu_pad_mxfp8

        info = self.batch_info
        slots = self._slots(T)
        ranks, scales = info.lora_ranks.to(torch.int32), info.scalings.float()
        x = self._pad(x.contiguous(), T)
        n_rows = x.shape[0]
        x_q, x_sf = flashinfer_mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
        yp = torch.empty(n_rows, pc.n_pad, device=x.device, dtype=torch.bfloat16)
        kg(pc.w, x, pc.a, pc.b, slots, ranks, scales, yp, w_sf=pc.sf, x_q=x_q, x_sf=x_sf)
        act, act_q, act_sf = swiglu_pad_mxfp8(yp, n_rows, Is, pc.slice_pad, Kp, limit)
        y = torch.empty(n_rows, Nd, device=x.device, dtype=torch.bfloat16)
        kd(pd.w, act, pd.a, pd.b, slots, ranks, scales, y, w_sf=pd.sf, x_q=act_q, x_sf=act_sf)
        return y[:T] if T != n_rows else y

    def _kernel(self, N, K, T, rows, mx, n_slots, epi="linear", pairings=None, qkv=None,
                eps=1e-6):
        from sglang.srt.lora.backend.blackwell import plans, tlinear
        from sglang.srt.lora.backend.blackwell.runner import FusedProjection

        T_run = max(T, 2)
        key = (N, K, T_run, rows, mx, n_slots, epi, tuple(pairings or ()), qkv, eps)
        if key in self._kernels:
            return self._kernels[key]
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        slices = len(rows)
        if pairings is None:
            pairings = ["none"]
            if not mx and slices == 2 and rows[0] * 2 == N:
                pairings = ["halves", "none"]  # gate/up: both halves in one CTA
            elif not mx and N % 256 == 0 and all(r % 256 == 0 for r in rows[:-1]):
                pairings = ["none", "adjacent"]
        op = None
        for pairing in pairings:
            cands = []
            t_ = plans.tuned(N, K, slices, pairing, T_run, mx)
            if t_:
                cands.append(t_)
            try:
                cands.append(tlinear.plan(N, K, rows, "mxfp8" if mx else "bf16", n_slots, sms,
                                          T_run, pairing, epi, False, qkv=qkv))
            except ValueError:
                pass
            for ks, a_splits, cluster in cands:
                try:
                    cand = tlinear.TLinear(N, K, T_run, rows, "mxfp8" if mx else "bf16", ks,
                                           a_splits, None, n_slots, pairing=pairing,
                                           cluster=cluster, epi=epi, qkv=qkv, eps=eps)
                except AssertionError:
                    continue
                smem = cand.stages * cand.stage_bytes + cand.fixed_bytes
                try:
                    plans.check_one_wave("blackwell LoRA", cand.grid, smem, sms,
                                         tmem_cols=cand.tmem_cols, cluster=cand.C)
                except ValueError:
                    continue
                op = cand
                break
            if op is not None:
                break
        kern = FusedProjection(op, torch.device("cuda")) if op is not None else None
        self._kernels[key] = kern
        if op is not None:
            logger.info("blackwell LoRA: %s N=%d K=%d T=%d -> %r", epi, N, K, T_run, op)
        return kern

    # ------------------------------------------------------------------ fused layer kinds
    def _bf16_lora(self, layer, slices):
        """(w, N, K, rows) when ``layer`` is a BF16 LoRA linear with ``slices`` rank-32
        slices on 128-row boundaries, else None."""
        base = layer.base_layer
        w = base.weight
        a, b = layer.lora_a, layer.lora_b
        if getattr(base, "bias", None) is not None or w.dtype != torch.bfloat16:
            return None
        if not w.is_contiguous() or b.shape[-1] != RANK or a.shape[0] > 8:
            return None
        if a.shape[1] != slices * RANK:
            return None
        N, K = w.shape
        rows = tuple(int(o) for o in layer.lora_offsets[1:])
        if len(rows) != slices or rows[-1] != N or any(r % 128 for r in rows[:-1]) or K % 128:
            return None
        return w, N, K, rows

    def forward_gate_up_swiglu(self, layer, x):
        T = self._decode_tokens(x)
        if T is None:
            return None
        got = self._bf16_lora(layer, 2)
        if got is None:
            return None
        w, N, K, rows = got
        if rows[0] * 2 != N:
            return None
        kern = self._kernel(N, K, T, rows, False, layer.lora_a.shape[0], epi="silu_mul",
                            pairings=["halves"])
        if kern is None or (not kern.compiled and torch.cuda.is_current_stream_capturing()):
            return None
        info = self.batch_info
        x = self._pad(x.contiguous(), T)
        y = torch.empty(x.shape[0], N // 2, device=x.device, dtype=torch.bfloat16)
        kern(w, x, layer.lora_a, self._b_mirror(layer, layer.lora_b), self._slots(T),
             info.lora_ranks.to(torch.int32), info.scalings.float(), y)
        return y[:T] if T == 1 else y

    def forward_attention_inputs(self, layer, x, spec):
        T = self._decode_tokens(x)
        if T is None:
            return None
        got = self._bf16_lora(layer, 3)
        if got is None:
            return None
        w, N, K, rows = got
        hd, qh, kvh = spec.head_dim, spec.num_heads, spec.num_kv_heads
        rope = spec.rotary_emb
        rot = getattr(rope, "rotary_dim", None)
        qw = qh * hd * (2 if spec.gated else 1)
        if hd not in (128, 256) or rot not in (64, 128) or rot > hd:
            return None
        if not getattr(rope, "is_neox_style", False) or rows != (qw, qw + kvh * hd, N) \
                or N != qw + 2 * kvh * hd:
            return None
        positions = spec.positions
        axis_map = getattr(rope, "axis_map", None)
        mrope = positions.dim() == 2
        if mrope and (axis_map is None or positions.shape[0] != 3):
            return None
        eps = getattr(spec.q_norm, "variance_epsilon", None)
        if eps is None or eps != getattr(spec.k_norm, "variance_epsilon", None):
            return None
        pool = getattr(spec.forward_batch, "token_to_kv_pool", None)
        if pool is None:
            from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

            pool = get_token_to_kv_pool()
        k_cache, v_cache = pool.get_kv_buffer(spec.layer_id)
        if k_cache.dtype != torch.bfloat16 or k_cache.dim() != 3 \
                or tuple(k_cache.shape[1:]) != (kvh, hd):
            return None
        geo = (qh, kvh, hd, rot, bool(spec.gated), mrope)
        kern = self._kernel(N, K, T, rows, False, layer.lora_a.shape[0], epi="qkv",
                            pairings=["adjacent"] if hd == 256 else ["none"], qkv=geo,
                            eps=float(eps))
        if kern is None or (not kern.compiled and torch.cuda.is_current_stream_capturing()):
            return None
        info = self.batch_info
        loc = spec.forward_batch.out_cache_loc[:T].to(torch.int64)
        pos = positions[..., :T].to(torch.int64)
        if T == 1:
            loc = torch.cat([loc, torch.full_like(loc, -1)])
            pos = torch.cat([pos, torch.zeros_like(pos)], dim=-1)
        x = self._pad(x.contiguous(), T)
        n_rows = x.shape[0]
        q = torch.empty(n_rows, qh * hd, device=x.device, dtype=torch.bfloat16)
        gate = torch.empty_like(q) if spec.gated else None
        io = dict(qn_w=self._norm_gamma(spec.q_norm), kn_w=self._norm_gamma(spec.k_norm),
                  cs=self._cos_sin(rope), pos=pos, kc=k_cache, vc=v_cache, loc=loc, q=q,
                  gate=gate, axis=self._axis_map(rope) if mrope else None)
        dummy = torch.empty(n_rows, 128, device=x.device, dtype=torch.bfloat16)
        kern(w, x, layer.lora_a, self._b_mirror(layer, layer.lora_b), self._slots(T),
             info.lora_ranks.to(torch.int32), info.scalings.float(), dummy, io=io)
        if T == 1:
            q, gate = q[:1], (gate[:1] if gate is not None else None)
        return q, gate

    @staticmethod
    def _norm_gamma(norm):
        """FP32 effective RMSNorm weight (Gemma-style norms scale by 1 + w)."""
        g = getattr(norm, "_blackwell_gamma", None)
        if g is None:
            w = norm.weight.detach().float()
            if type(norm).__name__ == "GemmaRMSNorm":
                w = w + 1.0
            g = norm._blackwell_gamma = w.contiguous()
        return g

    @staticmethod
    def _cos_sin(rope):
        cs = getattr(rope, "_blackwell_cos_sin", None)
        if cs is None:
            cs = rope._blackwell_cos_sin = rope.cos_sin_cache.detach().float().contiguous()
        return cs

    @staticmethod
    def _axis_map(rope):
        ax = getattr(rope, "_blackwell_axis_map", None)
        if ax is None:
            ax = rope._blackwell_axis_map = rope.axis_map.to(torch.int32).contiguous()
        return ax
