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

LoRA-B is read from a per-layer [N, slots*32] copy of the pool buffer, refreshed when the
memory pool reports new weights (``sync_lora_weights``).
"""

import logging
import os

import torch

from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend
from sglang.srt.lora.dense.plan import DenseLoraKind

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

    # ------------------------------------------------------------------ pool sync
    def sync_lora_weights(self, version) -> None:
        """Refresh every LoRA-B mirror after the memory pool loaded or cleared a slot."""
        if version == self._weights_version:
            return
        self._weights_version = version
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

    def _fused(self, layer, x, lora_a, lora_b, offsets):
        info = self.batch_info
        if info is None or getattr(info, "is_prefill", False):
            return None
        if not isinstance(x, torch.Tensor) or x.dim() != 2 or x.dtype != torch.bfloat16:
            return None
        T = x.shape[0]
        if info.bs != T or not (1 <= T <= MAX_TOKENS) or info.req_weight_indices is None:
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
        if len(rows) != slices or rows[-1] != N or any(r % 128 for r in rows[:-1]) or N % 128:
            return self._reject(layer, f"slices {rows} N={N}")
        mx = w.dtype == torch.float8_e4m3fn
        if mx:
            if not _mxfp8_base(base):
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

        x = x.contiguous()
        slots = info.req_weight_indices[:T].to(torch.int32)
        if T == 1:  # a single-row TMA tensor is degenerate: run two rows, keep the first
            x = torch.cat([x, torch.zeros_like(x)])
            slots = torch.cat([slots, torch.full_like(slots, -1)])
        n_rows = x.shape[0]
        y = torch.empty(n_rows, N, device=x.device, dtype=torch.bfloat16)
        extra = {}
        if mx:
            from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

            x_q, x_sf = flashinfer_mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
            extra = dict(w_sf=base.weight_scale_inv_swizzled, x_q=x_q, x_sf=x_sf)
        kern(w, x, lora_a, self._b_mirror(layer, lora_b), slots,
             info.lora_ranks.to(torch.int32), info.scalings.float(), y, **extra)
        return y[:T] if T != n_rows else y

    def _kernel(self, N, K, T, rows, mx, n_slots):
        from sglang.srt.lora.backend.blackwell import plans, tlinear
        from sglang.srt.lora.backend.blackwell.runner import FusedProjection

        T_run = max(T, 2)
        key = (N, K, T_run, rows, mx, n_slots)
        if key in self._kernels:
            return self._kernels[key]
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        slices = len(rows)
        pairings = ["none"]
        if not mx and slices == 2 and rows[0] * 2 == N:
            pairings = ["halves", "none"]  # gate/up: both halves in one CTA
        elif not mx and N % 256 == 0 and all(r % 256 == 0 for r in rows[:-1]):
            pairings = ["none", "adjacent"]
        op = None
        for pairing in pairings:
            cands = []
            t_ = plans.tuned(N, K, slices, pairing, T_run)
            if t_:
                cands.append(t_)
            try:
                cands.append(tlinear.plan(N, K, rows, "mxfp8" if mx else "bf16", n_slots, sms,
                                          T_run, pairing))
            except ValueError:
                pass
            for ks, a_splits, cluster in cands:
                try:
                    cand = tlinear.TLinear(N, K, T_run, rows, "mxfp8" if mx else "bf16", ks,
                                           a_splits, None, n_slots, pairing=pairing,
                                           cluster=cluster)
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
            logger.info("blackwell LoRA: N=%d K=%d T=%d -> %r", N, K, T_run, op)
        return kern
