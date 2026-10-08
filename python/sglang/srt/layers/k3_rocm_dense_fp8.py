"""Static per-tensor FP8 for selected Kimi-K3 dense linears on ROCm.

ROCm port of the ``SGLANG_K3_TARGET_DENSE_FP8`` ``tensor_static`` representation
used by the B300 production engine: K3 checkpoints keep attention and shared
experts in BF16, and at decode those skinny GEMMs are weight-bandwidth bound,
so serving them from E4M3 halves the bytes read per step.

Representation (identical to the CUDA one): one fp32 weight scale per merged
linear (amax / 448) and a unit activation scale, i.e. activations are
saturating-cast to E4M3 (aiter static quant) and the GEMM is aiter's pre-shuffled FP8 GEMM.

Scopes (``SGLANG_K3_TARGET_DENSE_FP8``): ``off`` | ``front`` (merged MoE front:
shared gate_up + router + latent down) | ``wide`` (front + KDA q/k/v/g).
"""

from __future__ import annotations

import logging
import math

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_FP8 = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8).max


def scope() -> str:
    value = envs.SGLANG_K3_TARGET_DENSE_FP8.get().lower()
    if value not in ("off", "front", "wide"):
        raise ValueError(f"SGLANG_K3_TARGET_DENSE_FP8={value!r}; expected off|front|wide")
    return value


def front_enabled() -> bool:
    return scope() in ("front", "wide")


def qkvg_enabled() -> bool:
    return scope() == "wide"


@torch.no_grad()
def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] bf16/fp16 -> ([N, K] E4M3, scalar fp32 scale) with scale = amax / 448."""
    w = weight.float()
    amax = w.abs().amax().clamp(min=1e-12)
    scale = (amax / _FP8_MAX).reshape(())
    q = (w / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8).contiguous()
    return q, scale.to(torch.float32)


# One-slot producer -> consumer handoff for a pre-quantized activation. A kernel
# that already holds the normalized row (attn_res aggregation) can emit its E4M3
# copy; the next StaticFP8Weight call consumes it only if it receives that very
# tensor (or a same-sized view of it; see _same_data), so a stale or reused buffer
# can never be picked up. Anything else falls back to quantizing in the GEMM wrapper.
_prequant_slot: tuple[torch.Tensor, torch.Tensor] | None = None


def offer_prequantized(x: torch.Tensor, xq: torch.Tensor) -> None:
    global _prequant_slot
    _prequant_slot = (x, xq)


def _same_data(src: torch.Tensor, x: torch.Tensor) -> bool:
    """x is src, or a same-sized view of it. The slot holds a strong reference
    to src, so its storage cannot be freed and reused while offered: any tensor
    with the same address, element count and dtype must alias src's data."""
    return x is src or (
        x.dtype == src.dtype
        and x.numel() == src.numel()
        and x.data_ptr() == src.data_ptr()
        and x.is_contiguous()
        and src.is_contiguous()
    )


def _take_prequantized(x: torch.Tensor) -> torch.Tensor | None:
    global _prequant_slot
    slot, _prequant_slot = _prequant_slot, None
    if slot is not None and _same_data(slot[0], x):
        return slot[1]
    return None


class StaticFP8Weight:
    """An E4M3 [N, K] weight with one scale; ``__call__`` computes x @ W^T.

    ROCm kernels: aiter.static_per_tensor_quant (unit activation scale) and the
    FlyDSL/CK pre-shuffled FP8 GEMM aiter.gemm_a8w8_bpreshuffle, which at decode
    sizes is ~1.4-2.3x faster than the bf16 GEMM it replaces on gfx950. The
    per-tensor weight scale is broadcast to the per-channel scale the kernel takes.
    """

    _MAX_TOKENS = 1 << 15

    def __init__(self, weight: torch.Tensor):
        from aiter.ops.shuffle import shuffle_weight

        q, scale = quantize_weight(weight)
        self.n, self.k = q.shape
        # Only the pre-shuffled copy is kept; module-weight views that alias it are
        # placeholders (their layers are routed here and never read them directly).
        self.weight = shuffle_weight(q, layout=(16, 16))
        del q
        self.scale = scale
        self.w_scale = scale.reshape(1, 1).expand(1, self.n).contiguous()
        self.unit = torch.ones(1, dtype=torch.float32, device=weight.device)
        self.x_scale = torch.ones(
            self._MAX_TOKENS, 1, dtype=torch.float32, device=weight.device
        )

    @property
    def shape(self) -> torch.Size:
        return self.weight.shape

    def __call__(
        self,
        x: torch.Tensor,
        out_dtype: torch.dtype | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import aiter

        x2 = x.reshape(-1, x.shape[-1])
        m = x2.shape[0]
        dtype = out_dtype or x.dtype
        if dtype not in (torch.bfloat16, torch.float16) or m > self._MAX_TOKENS:
            raise ValueError(f"K3 ROCm FP8 linear: unsupported out dtype {dtype} or M={m}")
        if m == 0:
            y = x2.new_empty((0, self.n), dtype=dtype)
        else:
            xq = _take_prequantized(x)
            if xq is None:
                xq = torch.empty(x2.shape, dtype=_FP8, device=x2.device)
                aiter.static_per_tensor_quant(xq, x2.contiguous(), self.unit)
            else:
                xq = xq.reshape(x2.shape)
            y = aiter.gemm_a8w8_bpreshuffle(
                xq, self.weight, self.x_scale[:m], self.w_scale, dtype=dtype
            )
        y = y.reshape(*x.shape[:-1], self.n)
        if out is not None:
            out.copy_(y)
            return out
        return y


class StaticFP8LinearMethod:
    """Drop-in ``quant_method`` for an already-loaded unquantized linear layer."""

    def __init__(self, fp8: StaticFP8Weight):
        self.fp8 = fp8

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        y = self.fp8(x)
        return y if bias is None else y + bias

    def process_weights_after_loading(self, layer) -> None:  # already converted
        return None


@torch.no_grad()
def convert_linear(layer: torch.nn.Module) -> StaticFP8Weight:
    """Convert ``layer.weight`` in place and route the layer through FP8."""
    fp8 = StaticFP8Weight(layer.weight.data)
    layer.weight.data = fp8.weight  # drop the bf16 tensor; nothing else reads it
    layer.quant_method = StaticFP8LinearMethod(fp8)
    return fp8


def _merged_variant(name: str) -> str:
    """The tuned base-shape FlyDSL kernel with its 3rd scheduling field zeroed.

    The tuned table picks a nonzero value there for N=6144 at M=1-2; at the
    merged widths (6304 / 7840, a partial last CTA wave) it is 0.4-0.7 us
    slower than 0 on gfx950, and the variants are bit-identical (same tile
    shape and K order; checked at M=1..16)."""
    import re

    mt = re.match(r"^(flydsl_bpreshuflle_\d+x\d+x\d+_F8_F8_B16_)(\d+)x(\d+)x(\d+)x(\d+)(_default)$", name)
    if mt is None:
        return name
    g = mt.groups()
    return f"{g[0]}{g[1]}x{g[2]}x0x{g[4]}{g[5]}"


class MergedTailFP8Weight:
    """A StaticFP8Weight ``base`` [N, K] with extra E4M3 rows appended in the same
    pre-shuffled buffer: one GEMM computes ``x @ [base | tail]^T`` at small M.

    The base rows keep their per-tensor scale and their exact bytes (the base
    weight becomes a view of the merged buffer, so no extra memory), and the GEMM
    runs the very FlyDSL kernel AITER's tuned table picks for the *base* shape at
    each M; the GEMM writes each output column from its own weight row, so the
    base columns are bit-identical to ``base(x)``. The tail rows get one scale
    per row (the kernel takes a per-channel weight scale). Only M with a FlyDSL
    tuned config for the base shape are covered (``covers``); the caller keeps
    its unmerged path for the rest.
    """

    def __init__(self, base: StaticFP8Weight, tail: torch.Tensor, max_m: int):
        import aiter
        from aiter import dtypes
        from aiter.jit.core import AITER_CONFIGS
        from aiter.ops.gemm_op_a8w8 import (
            _parse_flydsl_kernel_name,
            get_GEMM_config_with_quant_type,
        )
        from aiter.ops.shuffle import shuffle_weight

        assert base.n % 16 == 0 and tail.dim() == 2 and tail.shape[1] == base.k
        self.base_n, self.k, self.tail_n = base.n, base.k, tail.shape[0]
        self.configs: dict[int, dict] = {}
        tile_ns = {16}
        for m in range(1, max_m + 1):
            cfg = get_GEMM_config_with_quant_type(
                m, base.n, base.k, dtypes.fp8,
                AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE,
            )
            if cfg is None or cfg.get("libtype") != "flydsl":
                continue
            name = str(cfg.get("kernelName", ""))
            parsed = _parse_flydsl_kernel_name(name)
            if parsed is None or int(cfg.get("splitK", 0) or 0) != 0:
                continue
            tn = int(parsed[1])
            if base.n % tn:
                continue
            tile_ns.add(tn)
            self.configs[m] = {"kernelName": _merged_variant(name)}
        lcm = 1
        for tn in tile_ns:
            lcm = lcm * tn // math.gcd(lcm, tn)
        n = base.n + self.tail_n
        self.n = n + (-n) % lcm
        pad = self.n - n

        t = tail.float()
        t_scale = t.abs().amax(dim=1).clamp(min=1e-12) / _FP8_MAX
        tq = (t / t_scale[:, None]).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
        if pad:
            tq = torch.cat([tq, tq.new_zeros(pad, self.k)])
        # shuffle (16, 16) permutes within 16-row groups, so shuffling the tail
        # alone and appending it equals shuffling the concatenation.
        tail_sh = shuffle_weight(tq.contiguous(), layout=(16, 16))
        merged = torch.cat([base.weight, tail_sh]).contiguous()
        self.weight = merged
        base.weight = merged[: base.n]  # same bytes; frees the old buffer
        self.w_scale = torch.cat(
            [
                base.w_scale.reshape(-1),
                t_scale.to(torch.float32),
                torch.ones(pad, dtype=torch.float32, device=merged.device),
            ]
        ).reshape(1, self.n).contiguous()
        self.x_scale = base.x_scale
        self.unit = base.unit
        self.tail_scale = t_scale
        self._aiter = aiter

    def covers(self, m: int) -> bool:
        return m in self.configs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle_flydsl

        x2 = x.reshape(-1, x.shape[-1])
        m = x2.shape[0]
        xq = _take_prequantized(x)
        if xq is None:
            xq = torch.empty(x2.shape, dtype=_FP8, device=x2.device)
            self._aiter.static_per_tensor_quant(xq, x2.contiguous(), self.unit)
        else:
            xq = xq.reshape(x2.shape)
        y = torch.empty((m, self.n), dtype=torch.bfloat16, device=x2.device)
        gemm_a8w8_bpreshuffle_flydsl(
            xq, self.weight, self.x_scale[:m], self.w_scale, y, self.configs[m]
        )
        return y.reshape(*x.shape[:-1], self.n)
