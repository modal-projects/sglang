"""Online FP8 (W8A8) upgrade of the BF16 dense linears of an NVFP4 checkpoint.

Env-gated by ``SGLANG_ONLINE_FP8_STATIC=1`` (default off; byte-identical
dispatch when unset). Serving a stock NVFP4 MoE checkpoint (e.g.
``nvidia/Kimi-K2.6-NVFP4`` — NVFP4 routed experts + BF16 everything else) with
``--quantization modelopt_fp4`` leaves every non-MoE Linear in BF16. This
lever quantizes those linears to FP8 *online at engine load*:

- weight: PER-TENSOR scale (``input_to_float8`` in
  ``Fp8LinearMethod.process_weights_after_loading``; ``cutlass_fp8_supported``
  is forced False so the per-tensor branch is taken and ``apply`` routes to
  the ``torch._scaled_mm`` cublasLt per-tensor kernel — measured 38.2ms vs
  52.6ms cutlass per-token GEMM-only per 16k chunk on B200, vs 69.5ms BF16
  nvjet).
- activation: STATIC scale 1.0 — a bare saturating e4m3 cast. Validated for
  MuonClip-trained Kimi-K2.6 (rel-err ~= amax-calibrated ~= per-token
  dynamic); NOT safe for arbitrary models. Any quality gate must be re-run
  when pointing this at a different checkpoint.

Targeting is structural: any Linear the fp4 dispatch leaves unquantized
(``UnquantizedLinearMethod``) inside the decoder stack is rerouted, EXCEPT
modules with BF16-only custom kernels / that must stay full precision:
the MoE router gate (dsv3_router_gemm), the DSA indexer, vision towers, and
anything outside ``.layers.`` (lm_head / embeddings). Additional exclusions
compose via ``SGLANG_FP8_IGNORED_LAYERS`` (comma list, handled by
``Fp8Config``), e.g. ``SGLANG_FP8_IGNORED_LAYERS=o_proj`` keeps o_proj BF16
so the fused o_proj GEMM+allreduce lever stays engaged.

REQUIRED at serve time: ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``.
The load-time BF16->FP8 churn fragments the caching allocator; without VMM
coalescing SGLang measures less free memory after load and sizes the KV pool
*smaller* than the BF16 baseline, erasing the win.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.utils import get_bool_env_var, log_info_on_rank0

logger = logging.getLogger(__name__)

ENABLE_ONLINE_FP8_STATIC = get_bool_env_var("SGLANG_ONLINE_FP8_STATIC")

_rerouted_prefixes: list[str] = []


def _eligible(prefix: str) -> bool:
    return (
        ".layers." in prefix
        and "vision" not in prefix
        and "visual" not in prefix
        and ".indexer." not in prefix
        and not prefix.endswith(".gate")
    )


def maybe_online_fp8_static_linear_method(layer: torch.nn.Module, prefix: str):
    """Online-FP8-static quant method for a BF16 decoder Linear, or None.

    Called by NVFP4 configs for layers whose own dispatch produced
    ``UnquantizedLinearMethod``. Returns a configured ``Fp8LinearMethod``
    (online, static-1.0 activations, cublasLt per-tensor route) or None to
    keep the original method (env off, ineligible module, or module listed in
    ``SGLANG_FP8_IGNORED_LAYERS``).
    """
    if not ENABLE_ONLINE_FP8_STATIC or not _eligible(prefix):
        return None

    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

    method = Fp8Config(
        is_checkpoint_fp8_serialized=False,
        activation_scheme="dynamic",
    ).get_quant_method(layer, prefix)
    if not isinstance(method, Fp8LinearMethod):
        # e.g. UnquantizedLinearMethod via SGLANG_FP8_IGNORED_LAYERS
        return None

    # Route away from the cutlass GemmUniversal per-token path: with
    # cutlass_fp8_supported=False, process_weights_after_loading quantizes the
    # weight PER-TENSOR (input_to_float8) and apply() takes the per-tensor
    # torch._scaled_mm (cublasLt) branch.
    method.cutlass_fp8_supported = False
    _orig_pwal = method.process_weights_after_loading

    def _pwal_static_scale(layer: torch.nn.Module, _orig=_orig_pwal) -> None:
        _orig(layer)
        weight = getattr(layer, "weight", None)
        if weight is not None and weight.dtype == torch.float8_e4m3fn:
            # Static activation scale 1.0: bare saturating e4m3 cast.
            layer.input_scale = torch.nn.Parameter(
                torch.ones(1, dtype=torch.float32, device=weight.device),
                requires_grad=False,
            )

    method.process_weights_after_loading = _pwal_static_scale

    if not _rerouted_prefixes:
        # Startup assert (anti-silent-no-op): exactly one loud line per rank-0
        # process the first time the lever actually engages.
        log_info_on_rank0(
            logger,
            "SGLANG_ONLINE_FP8_STATIC=1: rerouting BF16 dense linears to "
            f"online FP8 (static-1.0 acts, cublasLt per-tensor); first={prefix}",
        )
    _rerouted_prefixes.append(prefix)
    return method
