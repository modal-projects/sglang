from typing import Optional

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.model_loader.weight_utils import get_actual_shard_size

_FLASHINFER_TRTLLM_BLOCK_K_BYTES = 128
_FLASHINFER_TRTLLM_BF16_BLOCK_COLS = _FLASHINFER_TRTLLM_BLOCK_K_BYTES // 2


def try_apply_flashinfer_trtllm_moe_lora_op_cuda_bucketed(
    *,
    param_name: str,
    param: torch.nn.Parameter,
    owner_module: torch.nn.Module,
    shard_id: str,
    local_expert_id: int,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    bucket_bytes: int,
) -> bool:
    quant_method = getattr(owner_module, "quant_method", None)
    if not isinstance(quant_method, UnquantizedFusedMoEMethod):
        return False
    if not getattr(quant_method, "use_flashinfer_trtllm_moe", False):
        return False
    if param.data.dim() != 4:
        return False
    if param.dtype != torch.bfloat16:
        raise ValueError(
            "Direct FlashInfer TRT-LLM MoE LoRA update currently supports only "
            f"BF16 weights. Parameter {param_name} has dtype {param.dtype}."
        )
    if lora_a.dim() != 2 or lora_b.dim() != 2:
        raise ValueError(
            f"Direct FlashInfer TRT-LLM MoE LoRA update expects 2D LoRA tensors for {param_name}; "
            f"got A={tuple(lora_a.shape)} B={tuple(lora_b.shape)}."
        )

    device = param.device
    tp_rank = getattr(owner_module, "moe_tp_rank", 0)
    use_presharded_weights = getattr(owner_module, "use_presharded_weights", False)
    is_gated = getattr(
        getattr(owner_module, "moe_runner_config", None), "is_gated", False
    )

    if shard_id in ("w1", "w3"):
        expected_param = getattr(owner_module, "w13_weight", None)
        if param is not expected_param:
            raise ValueError(
                f"FlashInfer TRT-LLM target {shard_id} must update w13_weight for {param_name}."
            )
        half_rows = int(owner_module.intermediate_size_per_partition)
        canonical_rows = half_rows * 2 if is_gated else half_rows
        canonical_cols = int(owner_module.hidden_size)
        if canonical_cols % _FLASHINFER_TRTLLM_BF16_BLOCK_COLS != 0:
            raise ValueError(
                f"FlashInfer TRT-LLM BF16 blocked columns must be divisible by "
                f"{_FLASHINFER_TRTLLM_BF16_BLOCK_COLS}; got {canonical_cols} for {param_name}."
            )
        expected_shape = (
            int(owner_module.num_local_experts),
            canonical_cols // _FLASHINFER_TRTLLM_BF16_BLOCK_COLS,
            canonical_rows,
            _FLASHINFER_TRTLLM_BF16_BLOCK_COLS,
        )
        switch_w13 = getattr(quant_method, "load_up_proj_weight_first", False)
        dst_row_start = (
            half_rows
            if (
                (switch_w13 and shard_id == "w1")
                or (not switch_w13 and shard_id == "w3")
            )
            and is_gated
            else 0
        )
        weight_start = half_rows * tp_rank
        lora_b_start = 0 if use_presharded_weights else weight_start
        lora_b_rows = get_actual_shard_size(
            half_rows,
            lora_b_start,
            lora_b.shape[0],
        )
        lora_a_start = 0
        lora_a_cols = canonical_cols
        dst_col_start = 0
        permute_fn = "w13"
    elif shard_id == "w2":
        expected_param = getattr(owner_module, "w2_weight", None)
        if param is not expected_param:
            raise ValueError(
                f"FlashInfer TRT-LLM target w2 must update w2_weight for {param_name}."
            )
        canonical_rows = int(owner_module.hidden_size)
        canonical_cols = int(owner_module.intermediate_size_per_partition)
        if canonical_cols % _FLASHINFER_TRTLLM_BF16_BLOCK_COLS != 0:
            raise ValueError(
                f"FlashInfer TRT-LLM BF16 blocked columns must be divisible by "
                f"{_FLASHINFER_TRTLLM_BF16_BLOCK_COLS}; got {canonical_cols} for {param_name}."
            )
        expected_shape = (
            int(owner_module.num_local_experts),
            canonical_cols // _FLASHINFER_TRTLLM_BF16_BLOCK_COLS,
            canonical_rows,
            _FLASHINFER_TRTLLM_BF16_BLOCK_COLS,
        )
        shard_size = canonical_cols
        weight_start = shard_size * tp_rank
        lora_a_start = 0 if use_presharded_weights else weight_start
        lora_a_cols = get_actual_shard_size(
            shard_size,
            lora_a_start,
            lora_a.shape[1],
        )
        lora_b_start = 0
        lora_b_rows = canonical_rows
        dst_row_start = 0
        dst_col_start = 0
        permute_fn = "w2"
    else:
        raise ValueError(
            "Unsupported MoE shard id for FlashInfer TRT-LLM direct update on "
            f"{param_name}: {shard_id}"
        )

    if tuple(param.data.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected FlashInfer TRT-LLM BF16 blocked shape for {param_name}: "
            f"got {tuple(param.data.shape)}, expected {expected_shape}."
        )
    if local_expert_id < 0 or local_expert_id >= param.data.shape[0]:
        raise ValueError(
            f"Local expert id {local_expert_id} is out of bounds for {param_name} "
            f"with {param.data.shape[0]} local experts."
        )
    if lora_b_rows <= 0 or lora_a_cols <= 0:
        return True
    if lora_b_start < 0 or lora_b_start + lora_b_rows > lora_b.shape[0]:
        raise ValueError(
            f"LoRA B row slice is out of bounds for {param_name}: "
            f"start={lora_b_start} rows={lora_b_rows} shape={tuple(lora_b.shape)}."
        )
    if lora_a_start < 0 or lora_a_start + lora_a_cols > lora_a.shape[1]:
        raise ValueError(
            f"LoRA A column slice is out of bounds for {param_name}: "
            f"start={lora_a_start} cols={lora_a_cols} shape={tuple(lora_a.shape)}."
        )

    inverse_permute = _flashinfer_trtllm_inverse_permute_indices(
        quant_method,
        permute_fn=permute_fn,
        rows=canonical_rows,
        cols=canonical_cols,
        device=device,
        dtype=param.dtype,
    )
    live_expert = param.data[local_expert_id]

    rank = int(lora_a.shape[0])
    a = _stage_cuda_fp32(lora_a.narrow(1, lora_a_start, lora_a_cols), device)
    rows_per_bucket = _dense_rows_per_bucket(
        rank=rank,
        cols=lora_a_cols,
        total_rows=lora_b_rows,
        bucket_bytes=bucket_bytes,
    )

    row = 0
    while row < lora_b_rows:
        rows = min(rows_per_bucket, lora_b_rows - row)
        b = _stage_cuda_fp32(lora_b.narrow(0, lora_b_start + row, rows), device)
        delta = torch.matmul(b, a).mul_(scaling)

        _add_flashinfer_trtllm_blocked_delta_fp32_(
            live_expert,
            inverse_permute,
            row_start=dst_row_start + row,
            col_start=dst_col_start,
            delta=delta,
        )
        row += rows

    return True


def get_flashinfer_moe_layer_for_postprocess(
    param: torch.nn.Parameter,
    owner_module: Optional[torch.nn.Module] = None,
) -> Optional[torch.nn.Module]:
    layer = owner_module
    if layer is None:
        weight_loader = getattr(param, "weight_loader", None)
        layer = getattr(weight_loader, "__self__", None)
    if layer is None:
        return None

    quant_method = getattr(layer, "quant_method", None)
    if not isinstance(quant_method, UnquantizedFusedMoEMethod):
        return None
    if not getattr(quant_method, "use_flashinfer_trtllm_moe", False):
        return None
    if not hasattr(layer, "w13_weight") or not hasattr(layer, "w2_weight"):
        return None
    return layer


def finalize_flashinfer_moe_layer_after_merge(layer: torch.nn.Module) -> None:
    quant_method = getattr(layer, "quant_method", None)
    if not isinstance(quant_method, UnquantizedFusedMoEMethod):
        return
    if getattr(quant_method, "use_flashinfer_trtllm_moe", False):
        # Partial routed-expert merges only restore the weight tensor that the
        # active loader touched. The untouched sibling can still be in blocked
        # FlashInfer TRT-LLM layout, so restore both canonical load shapes
        # before re-running the shared postprocess.
        for weight_name in ("w13_weight", "w2_weight"):
            param = getattr(layer, weight_name, None)
            if param is None:
                continue
            quant_method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
                layer,
                param,
                f"merged_update.experts.{weight_name}",
            )
    quant_method.process_weights_after_loading(layer)


def _stage_cuda_fp32(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device == device and tensor.dtype == torch.float32:
        return tensor
    return tensor.to(device=device, dtype=torch.float32, non_blocking=True)


def _dense_rows_per_bucket(
    *, rank: int, cols: int, total_rows: int, bucket_bytes: int
) -> int:
    a_bytes = 4 * rank * cols
    bytes_per_row = 4 * (rank + 2 * cols)
    available_for_rows = max(0, bucket_bytes - a_bytes)
    return max(
        1,
        min(total_rows, available_for_rows // max(bytes_per_row, 1)),
    )


def _flashinfer_trtllm_inverse_permute_indices(
    quant_method: UnquantizedFusedMoEMethod,
    *,
    permute_fn: str,
    rows: int,
    cols: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    cache = getattr(quant_method, "_cache_permute_indices", None)
    if cache is None:
        cache = {}
        quant_method._cache_permute_indices = cache

    cache_key = (
        "sglang_lora_inverse_permute",
        permute_fn,
        str(device),
        str(dtype),
        int(rows),
        int(cols),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    template_u8 = torch.empty((rows, cols), device=device, dtype=dtype).view(torch.uint8)
    if permute_fn == "w13":
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache,
            template_u8,
            128,
        )
    elif permute_fn == "w2":
        permute_indices = get_w2_permute_indices_with_cache(
            cache,
            template_u8,
            128,
        )
    else:
        raise ValueError(f"Unsupported FlashInfer TRT-LLM MoE permute: {permute_fn}")

    inverse = torch.argsort(permute_indices).to(device=device)
    cache[cache_key] = inverse
    return inverse


def _add_flashinfer_trtllm_blocked_delta_fp32_(
    live_expert: torch.Tensor,
    inverse_permute: torch.Tensor,
    *,
    row_start: int,
    col_start: int,
    delta: torch.Tensor,
) -> None:
    if delta.numel() == 0:
        return
    if live_expert.dim() != 3:
        raise ValueError(
            "FlashInfer TRT-LLM direct LoRA update expects one expert in blocked "
            f"layout [col_blocks, rows, 64], got shape={tuple(live_expert.shape)}."
        )

    rows = int(delta.shape[0])
    cols = int(delta.shape[1])
    if row_start < 0 or row_start + rows > inverse_permute.numel():
        raise ValueError(
            "FlashInfer TRT-LLM direct LoRA row slice is out of bounds: "
            f"row_start={row_start} rows={rows} canonical_rows={inverse_permute.numel()}."
        )
    max_cols = int(live_expert.shape[0]) * _FLASHINFER_TRTLLM_BF16_BLOCK_COLS
    if col_start < 0 or col_start + cols > max_cols:
        raise ValueError(
            "FlashInfer TRT-LLM direct LoRA column slice is out of bounds: "
            f"col_start={col_start} cols={cols} canonical_cols={max_cols}."
        )
    if (
        col_start % _FLASHINFER_TRTLLM_BF16_BLOCK_COLS != 0
        or cols % _FLASHINFER_TRTLLM_BF16_BLOCK_COLS != 0
    ):
        raise ValueError(
            "FlashInfer TRT-LLM direct LoRA column slice must be BF16-block aligned: "
            f"col_start={col_start} cols={cols} block_cols={_FLASHINFER_TRTLLM_BF16_BLOCK_COLS}."
        )

    delta_fp32 = delta.to(device=live_expert.device, dtype=torch.float32)
    row_positions = inverse_permute.narrow(0, int(row_start), rows).to(
        device=live_expert.device
    )
    block_start = int(col_start) // _FLASHINFER_TRTLLM_BF16_BLOCK_COLS
    block_count = cols // _FLASHINFER_TRTLLM_BF16_BLOCK_COLS
    live_blocks = live_expert.narrow(0, block_start, block_count)
    blocked_delta = (
        delta_fp32.reshape(rows, block_count, _FLASHINFER_TRTLLM_BF16_BLOCK_COLS)
        .permute(1, 0, 2)
        .contiguous()
    )

    updated_rows = live_blocks.index_select(1, row_positions).to(dtype=torch.float32)
    updated_rows.add_(blocked_delta)
    live_blocks.index_copy_(
        1, row_positions, updated_rows.to(dtype=live_expert.dtype)
    )
