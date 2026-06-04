from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.utils import is_cuda, is_musa

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

DEFAULT_DFLASH_MASK_TOKEN = "<|MASK|>"
DFLASH_PREFILL_REFILL_TARGET_ENV = "SGLANG_DFLASH_PREFILL_REFILL_TARGET"

logger = logging.getLogger(__name__)

_DFLASH_SAMPLING_VERIFY_AVAILABLE = False
_DFLASH_CHAIN_VERIFY_BUFFERS: dict[tuple[Optional[int], int], dict[str, Any]] = {}
_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS = frozenset(
    {
        "FlashInferAttnBackend",
        "FlashInferMLAAttnBackend",
        "FlashAttentionBackend",
        "TritonAttnBackend",
        "TRTLLMHAAttnBackend",
        "TRTLLMMLABackend",
    }
)


if is_cuda() or is_musa():
    try:
        from sgl_kernel import (
            top_k_renorm_prob,
            top_p_renorm_prob,
            tree_speculative_sampling_target_only,
        )

        _DFLASH_SAMPLING_VERIFY_AVAILABLE = True
    except Exception:
        top_k_renorm_prob = None
        top_p_renorm_prob = None
        tree_speculative_sampling_target_only = None
else:
    top_k_renorm_prob = None
    top_p_renorm_prob = None
    tree_speculative_sampling_target_only = None


def is_dflash_sampling_verify_available() -> bool:
    return _DFLASH_SAMPLING_VERIFY_AVAILABLE


def get_default_dflash_prefill_refill_target(max_running_requests: int) -> int:
    """Choose how many free running-request slots DFlash waits for before refill."""
    max_running_requests = max(0, int(max_running_requests))
    if max_running_requests < 8:
        return 1
    return min(4, max(2, (max_running_requests + 5) // 6))


def get_dflash_prefill_refill_target_override() -> Optional[int]:
    env_value = os.getenv(DFLASH_PREFILL_REFILL_TARGET_ENV)
    if env_value is None or not env_value.strip():
        return None
    try:
        return int(env_value)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; using DFlash prefill refill heuristic.",
            DFLASH_PREFILL_REFILL_TARGET_ENV,
            env_value,
        )
        return None


def resolve_dflash_prefill_refill_target(max_running_requests: int) -> int:
    override = get_dflash_prefill_refill_target_override()
    if override is not None:
        return override
    return get_default_dflash_prefill_refill_target(max_running_requests)


def should_delay_dflash_prefill_for_batching(
    *,
    running_bs: int,
    num_allocatable_reqs: int,
    max_running_requests: int,
    prefill_refill_target: int,
) -> bool:
    if running_bs <= 0:
        return False

    target_prefill_bs = int(prefill_refill_target)
    if target_prefill_bs <= 1:
        return False

    target_prefill_bs = min(target_prefill_bs, int(max_running_requests))
    return int(num_allocatable_reqs) < target_prefill_bs


def scale_kv_cell_size_per_token_for_dflash(
    *,
    target_cell_size_per_token: int,
    target_num_layers: int,
    draft_num_layers: int,
    draft_cell_size_per_token: Optional[int] = None,
) -> int:
    """Compute bytes/token budget for combined target+draft KV pools (DFLASH).

    DFLASH runs a separate draft runner with its own KV pool. The target runner's
    token capacity must fit both pools in aggregate.

    Returns:
        Approximate per-token bytes for (target KV + draft KV), expressed as a
        scaled version of `target_cell_size_per_token`, unless an explicit
        `draft_cell_size_per_token` is provided (in which case we sum them).
    """
    if target_cell_size_per_token <= 0:
        raise ValueError(
            "target_cell_size_per_token must be positive, "
            f"got {target_cell_size_per_token}."
        )

    if draft_cell_size_per_token is not None:
        draft_cell_size_per_token = int(draft_cell_size_per_token)
        if draft_cell_size_per_token <= 0:
            raise ValueError(
                "draft_cell_size_per_token must be positive when provided, "
                f"got {draft_cell_size_per_token}."
            )
        return int(target_cell_size_per_token) + int(draft_cell_size_per_token)

    if target_num_layers <= 0 or draft_num_layers <= 0:
        return int(target_cell_size_per_token)

    total_layers = int(target_num_layers) + int(draft_num_layers)
    return (
        int(target_cell_size_per_token) * int(total_layers) + int(target_num_layers) - 1
    ) // int(target_num_layers)


def resolve_dflash_verify_mask_policy(attn_backend: Any) -> tuple[str, bool]:
    backend = attn_backend
    for _ in range(4):
        full_backend = getattr(backend, "full_attn_backend", None)
        if full_backend is None:
            break
        backend = full_backend
    backend_name = type(backend).__name__
    return backend_name, (backend_name not in _DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS)


def apply_dflash_verify_logits_adjustments(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    draft_token_num: int,
) -> None:
    """Apply sampling-time logit adjustments for DFlash verify in place.

    This keeps v1 and v2 verify semantics aligned while letting overlap scheduling
    use the cheaper precomputed `acc_linear_penalties` path instead of allocating a
    repeated `[bs * draft_token_num, vocab]` penalty tensor every step.
    """
    if sampling_info is None:
        return
    if next_token_logits.ndim != 2:
        raise ValueError(
            "next_token_logits must be 2D, "
            f"got shape={tuple(next_token_logits.shape)}."
        )
    if draft_token_num <= 0:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}.")

    bs = len(sampling_info)
    if next_token_logits.shape[0] != bs * draft_token_num:
        raise ValueError(
            "next_token_logits row count mismatch for DFlash verify adjustments. "
            f"Expected {bs * draft_token_num}, got {next_token_logits.shape[0]}."
        )

    if sampling_info.has_custom_logit_processor:
        apply_custom_logit_processor(
            next_token_logits,
            sampling_info,
            num_tokens_in_batch=draft_token_num,
        )

    acc_linear_penalties = getattr(sampling_info, "acc_linear_penalties", None)
    penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
    vocab_mask = getattr(sampling_info, "vocab_mask", None)
    logit_bias = getattr(sampling_info, "logit_bias", None)

    logits_3d: Optional[torch.Tensor] = None

    def get_logits_3d() -> torch.Tensor:
        nonlocal logits_3d
        if logits_3d is None:
            logits_3d = next_token_logits.reshape(bs, draft_token_num, -1)
        return logits_3d

    # Dense fallback only when we need live penalizer application or a vocab mask.
    # In overlap scheduling the common path is `acc_linear_penalties`, which can be
    # broadcast over the verify block without materializing a repeated buffer.
    if (
        penalizer is not None and penalizer.is_required and acc_linear_penalties is None
    ) or vocab_mask is not None:
        linear_penalty = torch.zeros(
            (bs, next_token_logits.shape[1]),
            dtype=torch.float32,
            device=next_token_logits.device,
        )
        sampling_info.apply_logits_bias(linear_penalty)
        get_logits_3d().add_(
            linear_penalty[:, None, :].to(dtype=next_token_logits.dtype)
        )
        return

    if acc_linear_penalties is not None:
        if (
            acc_linear_penalties.device != next_token_logits.device
            or acc_linear_penalties.dtype != next_token_logits.dtype
        ):
            acc_linear_penalties = acc_linear_penalties.to(
                device=next_token_logits.device,
                dtype=next_token_logits.dtype,
            )
        get_logits_3d().add_(acc_linear_penalties[:, None, :])

    if logit_bias is not None:
        if (
            logit_bias.device != next_token_logits.device
            or logit_bias.dtype != next_token_logits.dtype
        ):
            logit_bias = logit_bias.to(
                device=next_token_logits.device,
                dtype=next_token_logits.dtype,
            )
        get_logits_3d().add_(logit_bias[:, None, :])


def _normalize_dflash_mrope_inputs(
    *,
    positions: torch.Tensor,
    lengths: Optional[torch.Tensor | Sequence[int]],
    mask: Optional[torch.Tensor],
    multimodal_inputs: Optional[Sequence[Any]],
) -> list[torch.Tensor]:
    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)
    if positions.ndim == 2:
        if lengths is not None:
            raise ValueError(
                "DFLASH mRoPE positions cannot use both 2D rows and lengths."
            )
        if mask is not None:
            if mask.shape != positions.shape:
                raise ValueError(
                    "DFLASH mRoPE mask/positions shape mismatch: "
                    f"{tuple(mask.shape)} vs {tuple(positions.shape)}."
                )
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
        chunks = []
        for batch_idx in range(int(positions.shape[0])):
            row = positions[batch_idx]
            chunks.append(row if mask is None else row[mask[batch_idx]])
        return chunks

    if positions.ndim != 1:
        raise ValueError(
            "DFLASH mRoPE positions must be 1D packed positions or 2D rows, "
            f"got shape={tuple(positions.shape)}."
        )
    if mask is not None:
        raise ValueError("DFLASH mRoPE mask is only supported with 2D positions.")

    if lengths is None:
        if multimodal_inputs is not None and len(multimodal_inputs) > 1:
            raise ValueError(
                "DFLASH mRoPE packed positions for multiple requests require lengths."
            )
        return [positions]

    if isinstance(lengths, torch.Tensor):
        lengths_list = [int(length) for length in lengths.to("cpu").tolist()]
    else:
        lengths_list = [int(length) for length in lengths]
    if any(length < 0 for length in lengths_list):
        raise ValueError(
            f"DFLASH mRoPE lengths must be non-negative, got {lengths_list}."
        )
    if sum(lengths_list) != int(positions.numel()):
        raise ValueError(
            "DFLASH mRoPE packed positions/lengths mismatch: "
            f"positions={int(positions.numel())}, lengths_sum={sum(lengths_list)}."
        )
    return list(torch.split(positions, lengths_list))


def _dflash_mrope_delta(mm_input: Any, *, device: torch.device) -> torch.Tensor:
    delta = getattr(mm_input, "mrope_position_delta", None)
    if delta is None:
        raise ValueError(
            "DFLASH mRoPE needs mrope_position_delta for generated positions."
        )
    if not isinstance(delta, torch.Tensor):
        delta = torch.as_tensor(delta, dtype=torch.int64)
    if delta.numel() != 1:
        raise ValueError(
            "DFLASH mRoPE expects one mrope_position_delta per request, "
            f"got shape={tuple(delta.shape)}."
        )
    return delta.reshape(-1)[0].to(device=device, dtype=torch.int64)


def _build_dflash_mrope_positions_for_request(
    scalar_positions: torch.Tensor,
    mm_input: Any,
) -> torch.Tensor:
    scalar_positions = scalar_positions.to(torch.int64)
    if scalar_positions.numel() == 0:
        return scalar_positions.new_empty((3, 0))

    mm_positions = (
        None if mm_input is None else getattr(mm_input, "mrope_positions", None)
    )
    if mm_positions is None:
        return scalar_positions.unsqueeze(0).repeat(3, 1)
    if not isinstance(mm_positions, torch.Tensor):
        mm_positions = torch.as_tensor(mm_positions, dtype=torch.int64)
    if mm_positions.ndim != 2 or int(mm_positions.shape[0]) != 3:
        raise ValueError(
            "DFLASH mRoPE expects mm_input.mrope_positions with shape [3, prompt_len], "
            f"got shape={tuple(mm_positions.shape)}."
        )

    device = scalar_positions.device
    mm_positions = mm_positions.to(device=device, dtype=torch.int64, non_blocking=True)
    prompt_len = int(mm_positions.shape[1])
    delta = _dflash_mrope_delta(mm_input, device=device)
    request_positions = (scalar_positions + delta).unsqueeze(0).repeat(3, 1)
    if prompt_len == 0:
        return request_positions

    prompt_mask = (scalar_positions >= 0) & (scalar_positions < prompt_len)
    prompt_indices = torch.clamp(scalar_positions, min=0, max=prompt_len - 1)
    prompt_positions = mm_positions.index_select(1, prompt_indices)
    return torch.where(prompt_mask.unsqueeze(0), prompt_positions, request_positions)


def _build_dflash_mrope_positions(
    positions: torch.Tensor,
    *,
    multimodal_inputs: Optional[Sequence[Any]],
    lengths: Optional[torch.Tensor | Sequence[int]] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build draft mRoPE positions shaped [3, N] from scalar absolute positions."""
    chunks = _normalize_dflash_mrope_inputs(
        positions=positions,
        lengths=lengths,
        mask=mask,
        multimodal_inputs=multimodal_inputs,
    )
    if multimodal_inputs is None:
        multimodal_inputs = [None] * len(chunks)
    if len(multimodal_inputs) != len(chunks):
        raise ValueError(
            "DFLASH mRoPE multimodal input count mismatch: "
            f"got {len(multimodal_inputs)} inputs for {len(chunks)} position chunks."
        )
    if not chunks:
        return positions.new_empty((3, 0), dtype=torch.int64)

    mrope_chunks = [
        _build_dflash_mrope_positions_for_request(chunk, mm_input)
        for chunk, mm_input in zip(chunks, multimodal_inputs)
    ]
    return torch.cat(mrope_chunks, dim=1).to(dtype=torch.int64)


def _get_or_create_chain_verify_buffers(
    *,
    bs: int,
    draft_token_num: int,
    device: torch.device,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    key = (device.index, int(draft_token_num))
    cached = _DFLASH_CHAIN_VERIFY_BUFFERS.get(key)
    cap_bs = 0 if cached is None else int(cached["cap_bs"])
    if cap_bs < bs:
        new_cap = max(int(bs), cap_bs * 2 if cap_bs > 0 else int(bs))
        retrieve_index = torch.arange(
            new_cap * draft_token_num, dtype=torch.int64, device=device
        ).view(new_cap, draft_token_num)
        row_next = torch.arange(
            1, draft_token_num + 1, dtype=torch.int64, device=device
        )
        row_next[-1] = -1
        retrieve_next_token = row_next.unsqueeze(0).expand(new_cap, -1).clone()
        retrieve_next_sibling = torch.full(
            (new_cap, draft_token_num), -1, dtype=torch.int64, device=device
        )
        predicts = torch.empty(
            (new_cap * draft_token_num,), dtype=torch.int32, device=device
        )
        accept_index = torch.empty(
            (new_cap, draft_token_num), dtype=torch.int32, device=device
        )
        accept_token_num = torch.empty((new_cap,), dtype=torch.int32, device=device)
        cached = {
            "cap_bs": int(new_cap),
            "retrieve_index": retrieve_index,
            "retrieve_next_token": retrieve_next_token,
            "retrieve_next_sibling": retrieve_next_sibling,
            "predicts": predicts,
            "accept_index": accept_index,
            "accept_token_num": accept_token_num,
        }
        _DFLASH_CHAIN_VERIFY_BUFFERS[key] = cached

    assert cached is not None
    retrieve_index = cached["retrieve_index"][:bs]
    retrieve_next_token = cached["retrieve_next_token"][:bs]
    retrieve_next_sibling = cached["retrieve_next_sibling"][:bs]
    predicts = cached["predicts"][: bs * draft_token_num]
    accept_index = cached["accept_index"][:bs]
    accept_token_num = cached["accept_token_num"][:bs]
    return (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    )


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    """Select target layer indices used to build DFlash context features.

    Args:
        num_target_layers: Number of transformer layers in the runtime target model.
        num_draft_layers: Number of layers in the DFlash draft model.

    Returns:
        A list of 0-based target layer indices of length `num_draft_layers`.

    Notes:
        - DFlash uses hidden states after each selected target layer (HF-style).
        - SGLang captures "before layer i", so the model hook will typically add +1
          when mapping to capture points.
    """
    if num_target_layers <= 0:
        raise ValueError(
            f"num_target_layers must be positive, got {num_target_layers}."
        )
    if num_draft_layers <= 0:
        raise ValueError(f"num_draft_layers must be positive, got {num_draft_layers}.")

    if num_draft_layers == 1:
        return [num_target_layers // 2]

    start = 1
    end = num_target_layers - 3
    if end < start:
        raise ValueError(
            "DFlash layer selection requires num_target_layers >= 4. "
            f"Got num_target_layers={num_target_layers}."
        )

    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def get_dflash_layer_types(config: Any) -> Optional[Sequence[str]]:
    text_config = _get_text_config(config)
    layer_types = _cfg_get(text_config, "layer_types", _cfg_get(config, "layer_types"))
    if layer_types is None:
        return None
    if isinstance(layer_types, str) or not isinstance(layer_types, Sequence):
        raise ValueError(
            "DFLASH config.layer_types must be a sequence of attention type strings."
        )
    return layer_types


def get_dflash_attention_sliding_window_size(config: Any) -> Optional[int]:
    layer_types = get_dflash_layer_types(config)
    if layer_types is None or "sliding_attention" not in layer_types:
        return None

    text_config = _get_text_config(config)
    sliding_window = _cfg_get(
        text_config, "sliding_window", _cfg_get(config, "sliding_window")
    )
    if sliding_window is None:
        raise ValueError(
            "DFLASH sliding_attention layers require config.sliding_window."
        )

    # HF sliding windows include the current token; SGLang stores window_left.
    return int(sliding_window) - 1


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_text_config(config: Any) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("text_config", config)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            resolved = get_text_config()
            if resolved is not None:
                return resolved
        except TypeError:
            pass
    return config


def _get_dflash_config(config: Any) -> dict:
    if isinstance(config, dict):
        cfg = config.get("dflash_config", None)
    else:
        cfg = getattr(config, "dflash_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg

    try:
        return dict(cfg)
    except Exception:
        return {}


def _parse_optional_int(
    value: Any,
    *,
    field_name: str,
    min_value: Optional[int] = None,
) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception as e:
        raise ValueError(f"Invalid {field_name}={value!r}.") from e
    if min_value is not None and parsed < int(min_value):
        comparator = "positive" if int(min_value) == 1 else f">= {int(min_value)}"
        raise ValueError(f"{field_name} must be {comparator}, got {parsed}.")
    return parsed


@dataclass(frozen=True)
class DFlashDraftConfig:
    num_hidden_layers: Optional[int]
    num_target_layers: Optional[int]
    block_size: Optional[int]
    target_layer_ids: Optional[List[int]]
    mask_token: str
    mask_token_id: Optional[int]

    def require_num_layers(self) -> int:
        if self.num_hidden_layers is None:
            raise ValueError(
                "DFLASH requires draft num_hidden_layers in config. "
                "Got config without num_hidden_layers."
            )
        return int(self.num_hidden_layers)

    def resolve_block_size(self, *, default: Optional[int] = None) -> Optional[int]:
        return self.block_size if self.block_size is not None else default

    def resolve_target_layer_ids(
        self,
        *,
        target_num_layers: int,
        draft_num_layers: Optional[int] = None,
    ) -> List[int]:
        target_num_layers = int(target_num_layers)
        if target_num_layers <= 0:
            raise ValueError(
                f"target_num_layers must be positive, got {target_num_layers}."
            )

        if self.target_layer_ids is None:
            if draft_num_layers is None:
                draft_num_layers = self.require_num_layers()
            return build_target_layer_ids(target_num_layers, int(draft_num_layers))

        resolved = list(self.target_layer_ids)
        if len(resolved) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(resolved)}."
            )
        for idx, val in enumerate(resolved):
            if val < 0 or val >= target_num_layers:
                raise ValueError(
                    "DFLASH target_layer_ids contains an out-of-range layer id. "
                    f"target_layer_ids[{idx}]={val}, target_num_layers={target_num_layers}."
                )
        return resolved


def parse_dflash_draft_config(*, draft_hf_config: Any) -> DFlashDraftConfig:
    """Parse and validate DFLASH draft config fields from HF config/dict."""
    dflash_cfg = _get_dflash_config(draft_hf_config)
    draft_text_config = _get_text_config(draft_hf_config)

    num_hidden_layers = _parse_optional_int(
        _cfg_get(draft_text_config, "num_hidden_layers", None),
        field_name="DFLASH draft num_hidden_layers",
        min_value=1,
    )
    raw_num_target_layers = dflash_cfg.get(
        "num_target_layers",
        _cfg_get(draft_hf_config, "num_target_layers", None),
    )
    num_target_layers = _parse_optional_int(
        raw_num_target_layers,
        field_name="DFLASH draft num_target_layers",
        min_value=1,
    )

    # Keep support for current checkpoints where block_size is top-level.
    raw_block_size = dflash_cfg.get(
        "block_size",
        _cfg_get(draft_hf_config, "block_size", None),
    )
    block_size = _parse_optional_int(
        raw_block_size,
        field_name="DFLASH block_size",
        min_value=1,
    )

    layer_ids = dflash_cfg.get(
        "target_layer_ids",
        _cfg_get(draft_hf_config, "target_layer_ids", None),
    )
    parsed_target_layer_ids: Optional[List[int]]
    if layer_ids is None:
        parsed_target_layer_ids = None
    else:
        if not isinstance(layer_ids, (list, tuple)):
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be a list of ints, "
                f"got type={type(layer_ids).__name__}."
            )
        parsed_target_layer_ids = [int(x) for x in layer_ids]
        if len(parsed_target_layer_ids) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(parsed_target_layer_ids)}."
            )

    mask_token = dflash_cfg.get("mask_token", None)
    if mask_token is None:
        mask_token = DEFAULT_DFLASH_MASK_TOKEN
    if not isinstance(mask_token, str) or not mask_token:
        raise ValueError(
            "DFLASH dflash_config.mask_token must be a non-empty string, "
            f"got {mask_token!r}."
        )

    mask_token_id = dflash_cfg.get("mask_token_id", None)
    if mask_token_id is not None:
        if not isinstance(mask_token_id, Integral) or isinstance(mask_token_id, bool):
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be an integer, "
                f"got {mask_token_id!r} (type={type(mask_token_id).__name__})."
            )
        mask_token_id = int(mask_token_id)
        if mask_token_id < 0:
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be non-negative, "
                f"got {mask_token_id}."
            )

    return DFlashDraftConfig(
        num_hidden_layers=num_hidden_layers,
        num_target_layers=num_target_layers,
        block_size=block_size,
        target_layer_ids=parsed_target_layer_ids,
        mask_token=mask_token,
        mask_token_id=mask_token_id,
    )


def can_dflash_slice_qkv_weight(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether DFlash can slice KV weights from a fused QKV linear layer."""
    quant_method = getattr(qkv_proj, "quant_method", None)
    if not isinstance(quant_method, UnquantizedLinearMethod):
        return (
            False,
            "quantized qkv_proj is not supported for this path "
            f"(quant_method={type(quant_method).__name__})",
        )
    if not hasattr(qkv_proj, "weight"):
        return False, "qkv weight tensor is missing"
    return True, ""


def can_dflash_use_fused_qkv_proj(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether a QKV layer is eligible for DFlash fused KV materialization."""
    eligible, reason = can_dflash_slice_qkv_weight(qkv_proj)
    if not eligible:
        return False, reason
    if getattr(qkv_proj, "bias", None) is not None:
        return False, "qkv bias is not supported for fused KV path"
    return True, ""


def compute_dflash_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens (greedy verify rule).

    Args:
        candidates: Token ids proposed by the DFlash draft, including the current token.
            Shape: [bs, block_size]. candidates[:, 0] is the current token.
        target_predict: Token ids predicted by the target model for each position in the block.
            Shape: [bs, block_size]. target_predict[:, t] corresponds to argmax at position t.

    Returns:
        correct_len: int32 tensor [bs], number of accepted *draft* tokens (excluding current token and bonus token).
        bonus: int64 tensor [bs], the target-predicted token at index correct_len (the "bonus" token to append).

    Notes:
        Matches the reference implementation rule:
          accept while candidates[:, 1:] == target_predict[:, :-1] consecutively.
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates. "
            f"candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )

    bs, block_size = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    matches = candidates[:, 1:] == target_predict[:, :-1]
    correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    bonus = target_predict[torch.arange(bs, device=target_predict.device), correct_len]
    return correct_len, bonus.to(torch.int64)


def compute_dflash_sampling_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    max_top_k: Optional[int] = None,
    uniform_top_k_value: Optional[int] = None,
    threshold_single: Optional[float] = None,
    threshold_acc: Optional[float] = None,
    uniform_samples: Optional[torch.Tensor] = None,
    uniform_samples_for_final_sampling: Optional[torch.Tensor] = None,
    use_sparse_topk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens for non-greedy sampling.

    This is a chain-specialized variant of speculative target-only verification:
      - DFlash proposals are linear (topk == 1), so each verify level has at most one candidate.
      - When a candidate is rejected at a level, the final token is sampled from
        `relu(q - p)` where `p` has only the rejected candidate mass.
    """
    if not _DFLASH_SAMPLING_VERIFY_AVAILABLE:
        raise RuntimeError(
            "DFLASH non-greedy verification is unavailable on this build/device."
        )
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if next_token_logits.ndim != 2:
        raise ValueError(
            "next_token_logits must be 2D, "
            f"got shape={tuple(next_token_logits.shape)}."
        )

    bs, draft_token_num = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if draft_token_num <= 0:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}.")
    if next_token_logits.shape[0] != bs * draft_token_num:
        raise ValueError(
            "next_token_logits row count mismatch. "
            f"Expected {bs * draft_token_num}, got {next_token_logits.shape[0]}."
        )
    if candidates.device != next_token_logits.device:
        raise ValueError(
            "candidates and next_token_logits must be on the same device, "
            f"got {candidates.device} and {next_token_logits.device}."
        )

    if threshold_single is None:
        from sglang.srt.server_args import get_global_server_args

        threshold_single = get_global_server_args().speculative_accept_threshold_single
    if threshold_acc is None:
        from sglang.srt.server_args import get_global_server_args

        threshold_acc = get_global_server_args().speculative_accept_threshold_acc
    threshold_single = float(threshold_single)
    threshold_acc = max(float(threshold_acc), 1e-9)

    device = next_token_logits.device

    if uniform_samples is None:
        uniform_samples = torch.rand(
            (bs, draft_token_num), dtype=torch.float32, device=device
        )
    else:
        if uniform_samples.shape != (bs, draft_token_num):
            raise ValueError(
                "uniform_samples shape mismatch. "
                f"Expected {(bs, draft_token_num)}, got {tuple(uniform_samples.shape)}."
            )
        uniform_samples = uniform_samples.to(device=device, dtype=torch.float32)

    if uniform_samples_for_final_sampling is None:
        uniform_samples_for_final_sampling = torch.rand(
            (bs,), dtype=torch.float32, device=device
        )
    else:
        if uniform_samples_for_final_sampling.shape != (bs,):
            raise ValueError(
                "uniform_samples_for_final_sampling shape mismatch. "
                f"Expected {(bs,)}, got {tuple(uniform_samples_for_final_sampling.shape)}."
            )
        uniform_samples_for_final_sampling = uniform_samples_for_final_sampling.to(
            device=device,
            dtype=torch.float32,
        )

    need_top_k = bool(getattr(sampling_info, "need_top_k_sampling", True))
    need_top_p = bool(getattr(sampling_info, "need_top_p_sampling", False))
    # Build target distribution once over all verify rows.
    expanded_temperature = torch.repeat_interleave(
        sampling_info.temperatures, draft_token_num, dim=0
    )
    scaled_logits = next_token_logits / expanded_temperature
    sparse_topk_applied = False

    if use_sparse_topk and need_top_k:
        repeated_top_ks = torch.repeat_interleave(
            sampling_info.top_ks, draft_token_num, dim=0
        ).to(dtype=torch.int64)
        vocab_size = int(scaled_logits.shape[-1])
        repeated_top_ks.clamp_(min=1, max=vocab_size)
        if max_top_k is None:
            max_top_k = int(repeated_top_ks.max().item())
        else:
            max_top_k = int(max_top_k)
        if max_top_k < 1:
            max_top_k = 1
        elif max_top_k > vocab_size:
            max_top_k = vocab_size

        # Sparse exact path for top-k/top-p (top-k-first semantics), then scatter to dense.
        if 0 < max_top_k < vocab_size:
            topk_logits, topk_indices = torch.topk(scaled_logits, k=max_top_k, dim=-1)
            if uniform_top_k_value is None or int(uniform_top_k_value) != max_top_k:
                ranks = torch.arange(max_top_k, device=device, dtype=torch.int64)[
                    None, :
                ]
                valid = ranks < repeated_top_ks.unsqueeze(1)
                topk_logits = topk_logits.masked_fill(~valid, float("-inf"))

            topk_probs = F.softmax(topk_logits, dim=-1)
            if need_top_p:
                repeated_top_ps = torch.repeat_interleave(
                    sampling_info.top_ps, draft_token_num, dim=0
                )
                topk_probs = top_p_renorm_prob(topk_probs, repeated_top_ps)

            target_probs = torch.zeros_like(scaled_logits, dtype=topk_probs.dtype)
            target_probs.scatter_(1, topk_indices, topk_probs)
            sparse_topk_applied = True

    if not sparse_topk_applied:
        target_probs = F.softmax(scaled_logits, dim=-1)
        if need_top_k:
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ks, draft_token_num, dim=0),
            )
        if need_top_p:
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ps, draft_token_num, dim=0),
            )
    target_probs = target_probs.view(bs, draft_token_num, -1).contiguous()
    draft_probs = torch.zeros_like(target_probs)

    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _get_or_create_chain_verify_buffers(
        bs=bs,
        draft_token_num=draft_token_num,
        device=device,
    )
    candidates_i64 = (
        candidates if candidates.dtype == torch.int64 else candidates.to(torch.int64)
    )
    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates_i64,
        # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    correct_len = accept_token_num
    row_ids = torch.arange(bs, dtype=torch.long, device=device)
    accept_pos = accept_index[row_ids, correct_len.to(torch.long)].to(torch.long)
    bonus = predicts[accept_pos].to(torch.int64)
    return correct_len, bonus


def validate_dflash_request(req: Req, enable_overlap: bool) -> Optional[str]:
    if req.return_logprob:
        return "DFLASH speculative decoding does not support return_logprob yet."

    if enable_overlap and req.return_hidden_states:
        return "DFLASH speculative decoding does not support return_hidden_states yet."

    if (
        req.sampling_params.json_schema is not None
        or req.sampling_params.regex is not None
        or req.sampling_params.ebnf is not None
        or req.sampling_params.structural_tag is not None
    ):
        return (
            "DFLASH speculative decoding does not support "
            "grammar-constrained decoding yet."
        )

    return None
