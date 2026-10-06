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
            xq = torch.empty(x2.shape, dtype=_FP8, device=x2.device)
            aiter.static_per_tensor_quant(xq, x2.contiguous(), self.unit)
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
