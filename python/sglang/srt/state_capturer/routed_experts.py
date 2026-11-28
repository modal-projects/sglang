from typing import Any, Optional, Tuple

import numpy as np
import pybase64
import torch
import torch.nn.functional as F

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    get_dp_local_slice_cpu,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_resources,
    get_schedule,
)
from sglang.srt.state_capturer.base import BaseTopkCapturer

ROUTED_EXPERTS_WIRE_DTYPES: dict[str, np.dtype] = {
    "int32": np.dtype("<i4"),
    "int16": np.dtype("<i2"),
    "uint16": np.dtype("<u2"),
    "uint8": np.dtype("u1"),
}

_INT16_MAX = 32767
_UINT16_MAX = 65535
_UINT8_MAX = 255


def _infer_num_routed_experts(model_config: ModelConfig) -> Optional[int]:
    hf_cfg = getattr(model_config, "hf_text_config", None)
    if hf_cfg is None:
        return None
    for attr in ("n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts"):
        val = getattr(hf_cfg, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    return None


def encode_routed_experts_tensor(
    tensor: torch.Tensor,
    wire_dtype: Optional[str] = None,
) -> Tuple[str, str]:
    """Encode captured routed-experts tensor as base64 in the requested wire dtype.

    Defaults to ``"int32"`` when ``wire_dtype`` is omitted for backwards
    compatibility. If values do not fit a requested narrower unsigned dtype
    (e.g. ``"uint8"`` with expert IDs > 255 or ``-1`` padding), falls back to
    ``"int16"`` (or ``"int32"``) and returns the actual emitted dtype name.
    """
    requested = wire_dtype or "int32"
    if requested not in ROUTED_EXPERTS_WIRE_DTYPES:
        raise ValueError(
            f"unsupported routed_experts_dtype {requested!r}; "
            f"choose from {sorted(ROUTED_EXPERTS_WIRE_DTYPES)}"
        )
    arr = tensor.numpy()
    actual = requested
    if arr.size > 0 and requested != "int32":
        low = int(arr.min())
        high = int(arr.max())
        if requested == "uint8" and not (0 <= low and high <= _UINT8_MAX):
            actual = "int16" if (-32768 <= low and high <= _INT16_MAX) else "int32"
        elif requested == "uint16" and not (0 <= low and high <= _UINT16_MAX):
            actual = "int16" if (-32768 <= low and high <= _INT16_MAX) else "int32"
        elif requested == "int16" and not (-32768 <= low and high <= _INT16_MAX):
            actual = "int32"
    target_dtype = ROUTED_EXPERTS_WIRE_DTYPES[actual]
    if arr.dtype != target_dtype:
        arr = arr.astype(target_dtype, copy=False)
    encoded = pybase64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("utf-8")
    return encoded, actual


def _is_scattered_a2a_backend() -> bool:
    """Return whether routed tokens are scattered across attention-TP ranks."""
    backend = get_moe_a2a_backend()
    return backend.is_deepep() or backend.is_deepep_v2()


class RoutedExpertsCapturer(BaseTopkCapturer):
    """Capturer for routed experts with host buffer.

    Routed experts share a global device buffer across DP ranks (indexed by
    dp_rank), so `_get_local_slice` overrides the default to apply DP-rank-aware
    slicing. The device cache also holds extra columns for any fused shared
    experts; the host cache and user-facing return drop them via the
    [:topk_size] truncation.
    """

    @staticmethod
    def create(
        *,
        model: torch.nn.Module,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        device: str,
    ) -> Optional["RoutedExpertsCapturer"]:
        if not get_exec().features.enable_return_routed_experts:
            return None
        # The loader attaches this to every entry model, including wrappers.
        # Size from the owner, independent of which runner is currently active.
        num_fused_shared_experts = model.shared_experts_config.num_fused_shared_experts
        return RoutedExpertsCapturer(
            model_config,
            num_tokens=num_tokens,
            max_running_requests=max_running_requests,
            num_fused_shared_experts=num_fused_shared_experts,
            device=device,
        )

    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        num_fused_shared_experts: int,
        device: str,
    ):
        self.num_fused_shared_experts = num_fused_shared_experts
        topk_size = model_config.hf_text_config.num_experts_per_tok
        num_layers = model_config.hf_text_config.num_hidden_layers

        # Scale by num_dp_ranks so the buffer covers the full DP-concatenated batch.
        # _get_local_slice indexes into [attention_dp_rank * cuda_graph_batch, ...)
        # and otherwise overflows on dp_rank > 0 when max_running_requests >
        # chunked_prefill_size.
        # FIXME: spec decoding's num_verify_tokens is still not accounted for.
        max_batch_size = max(
            get_schedule().chunked_prefill_size * get_parallel().num_dp_ranks,
            max_running_requests * get_parallel().num_dp_ranks,
        )

        num_experts = _infer_num_routed_experts(model_config)
        cache_dtype = (
            torch.int16
            if num_experts is not None
            and num_experts + num_fused_shared_experts <= _INT16_MAX
            else torch.int32
        )

        super().__init__(
            num_tokens=num_tokens,
            max_batch_size=max_batch_size,
            num_layers=num_layers,
            topk_size=topk_size,
            device=device,
            name="routed_experts",
            device_topk_size=topk_size + num_fused_shared_experts,
            dtype=cache_dtype,
        )

        # Rebuild the full token batch before routed-expert readback.
        # Keep gather_buffer int32 because NCCL has no ncclInt16; super().capture
        # narrows into the int16 device_cache after the all-gather.
        if _is_scattered_a2a_backend():
            attn_tp_size = (
                get_parallel().attn_tp_size if is_dp_attention_enabled() else 1
            )
            self.gather_buffer = torch.empty(
                (
                    self.device_cache.buffer.shape[0] * attn_tp_size,
                    self.device_cache.buffer.shape[2],
                ),
                dtype=torch.int32,
                device=device,
            )

    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        device_width = self.device_cache.buffer.shape[2]
        if topk_indices.shape[1] != device_width:
            # Routes whose shared columns are appended after capture pass routed ids only.
            if topk_indices.shape[1] != device_width - self.num_fused_shared_experts:
                raise ValueError(
                    f"Captured top-k width {topk_indices.shape[1]} is neither the "
                    f"device width {device_width} nor its routed part "
                    f"{device_width - self.num_fused_shared_experts}"
                )
            topk_indices = F.pad(
                topk_indices, (0, self.num_fused_shared_experts), value=-1
            )
        if _is_scattered_a2a_backend():
            local_topk = topk_indices.to(torch.int32)
            topk_indices = self.gather_buffer[
                : local_topk.size(0) * get_parallel().attn_tp_size
            ]
            attn_tp_all_gather_into_tensor(topk_indices, local_topk)
        super().capture(layer_id, topk_indices)

    def _get_local_slice(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ) -> torch.Tensor:
        # Gathered rows start at buffer offset zero on every DP rank.
        if is_dp_attention_enabled() and not _is_scattered_a2a_backend():
            # GPU->CPU sync would break overlap; operate on CPU directly.
            local_start_pos, local_num_tokens = get_dp_local_slice_cpu(
                forward_batch, can_run_graph, cuda_graph_batch
            )
            local_end_pos = local_start_pos + local_num_tokens
        else:
            local_start_pos, local_end_pos = 0, forward_batch.out_cache_loc.shape[0]
        return self.device_cache.buffer[
            local_start_pos:local_end_pos, :, : self.topk_size
        ]


def get_global_experts_capturer() -> Optional[RoutedExpertsCapturer]:

    return get_resources().experts_capturer


def set_global_experts_capturer(capturer: Optional[RoutedExpertsCapturer]):

    get_resources().experts_capturer = capturer


def destroy_global_experts_capturer():
    if (capturer := get_resources().experts_capturer) is not None:
        capturer.destroy()
    get_resources().experts_capturer = None


def extract_routed_experts_from_meta_info(data):
    # Decode base64 routed_experts honoring optional routed_experts_dtype (default int32).
    meta = data["meta_info"]
    routed_experts_base64 = meta.get("routed_experts", None)
    dtype_name = meta.get("routed_experts_dtype", "int32")
    dtype = ROUTED_EXPERTS_WIRE_DTYPES[dtype_name]
    routed_experts = np.frombuffer(
        pybase64.b64decode(routed_experts_base64.encode("utf-8")), dtype=dtype
    )
    return routed_experts


def disable_routed_experts_capture_for_draft(model: Any) -> None:
    """Opt every draft MoE ``TopK`` out of routed-experts (R3) capture.

    Capture is target-only; a draft ``TopK`` must never write the target's
    process-global buffer. ``HashTopK`` has no ``topk_config`` and never
    captures, so it is left untouched.
    """
    # Lazy import: ``layers.moe.topk`` imports ``get_global_experts_capturer``
    # from this module, so a top-level import here would be circular.
    from sglang.srt.layers.moe.topk import TopK

    for module in model.modules():
        if isinstance(module, TopK):
            module.topk_config.allow_routed_experts_capture = False
