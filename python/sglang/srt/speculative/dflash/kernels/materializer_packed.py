from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang.srt.speculative.dflash.contracts import (
    DFlashKVCache,
    DFlashMaterializerConfig,
    DFlashMaterializerWeights,
)
from sglang.srt.speculative.dflash.kernels.compact_commit_jit import (
    compact_commit_jit,
)
from sglang.srt.speculative.dflash.kernels.compact_commit_triton import (
    compact_commit_triton,
)
from sglang.srt.speculative.dflash.kernels.post_projection_packed_jit import (
    postprocess_commit_packed_jit,
    postprocess_commit_packed_jit_unchecked,
    postprocess_prompt_packed_jit,
    postprocess_prompt_packed_jit_unchecked,
)
from sglang.srt.speculative.dflash.kernels.post_projection_packed_triton import (
    postprocess_commit_packed_triton,
    postprocess_prompt_packed_triton,
)
from sglang.srt.speculative.dflash.reference.materializer import (
    _validate_commit_inputs,
    _validate_prompt_inputs,
)
from sglang.srt.speculative.dflash.reference.post_projection import (
    build_neox_cos_sin_cache,
)


@dataclass(frozen=True)
class DFlashPackedProjectionGroup:
    layer_start: int
    group_count: int
    flat_weight_t: torch.Tensor
    flat_bias: torch.Tensor | None
    out_cols: int


@dataclass(frozen=True)
class DFlashPackedMaterializerWorkspace:
    group_size: int
    max_rows: int
    packed_flat_scratch: torch.Tensor
    k_norm_weight: torch.Tensor
    k_norm_eps: torch.Tensor
    cos_sin_cache: torch.Tensor
    groups: tuple[DFlashPackedProjectionGroup, ...]


def _project_packed_group(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    config: DFlashMaterializerConfig,
) -> torch.Tensor:
    group_count = int(weight.shape[0])
    flat_weight = weight.reshape(group_count * 2 * config.kv_size, config.hidden_size)
    flat_bias = None if bias is None else bias.reshape(group_count * 2 * config.kv_size)
    return F.linear(hidden, flat_weight, flat_bias).view(
        int(hidden.shape[0]),
        group_count,
        2,
        config.num_kv_heads,
        config.head_dim,
    )


def _project_packed_group_mm_out(
    hidden: torch.Tensor,
    group: DFlashPackedProjectionGroup,
    *,
    workspace: DFlashPackedMaterializerWorkspace,
    config: DFlashMaterializerConfig,
) -> torch.Tensor:
    row_count = int(hidden.shape[0])
    if row_count > workspace.max_rows:
        raise ValueError(
            f"workspace max_rows={workspace.max_rows} is smaller than requested row_count={row_count}."
        )
    out_flat = workspace.packed_flat_scratch.narrow(0, 0, row_count).narrow(
        1, 0, group.out_cols
    )
    torch.mm(hidden, group.flat_weight_t, out=out_flat)
    if group.flat_bias is not None:
        out_flat.add_(group.flat_bias)
    return out_flat.view(
        row_count,
        group.group_count,
        2,
        config.num_kv_heads,
        config.head_dim,
    )


def _ensure_cos_sin_cache(
    *,
    config: DFlashMaterializerConfig,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    if cos_sin_cache is not None:
        return cos_sin_cache.to(device=device, dtype=torch.float32)
    max_position = int(positions.max().item()) if int(positions.numel()) > 0 else 0
    return build_neox_cos_sin_cache(
        rotary_dim=config.rotary_dim,
        rope_theta=config.rope_theta,
        max_position=max_position,
        device=device,
    )


def _prepare_jit_postprocess_constants(
    *,
    weights: DFlashMaterializerWeights,
    dtype: torch.dtype,
    device: torch.device,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        weights.k_norm_weight.to(device=device, dtype=dtype).contiguous(),
        weights.k_norm_eps.to(device=device, dtype=cos_sin_cache.dtype).contiguous(),
    )


def _validate_workspace(
    *,
    workspace: DFlashPackedMaterializerWorkspace,
    config: DFlashMaterializerConfig,
    group_size: int,
    row_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if workspace.group_size != group_size:
        raise ValueError(
            f"workspace group_size={workspace.group_size} does not match requested group_size={group_size}."
        )
    if row_count > workspace.max_rows:
        raise ValueError(
            f"workspace max_rows={workspace.max_rows} is smaller than requested row_count={row_count}."
        )
    if workspace.packed_flat_scratch.device != device:
        raise ValueError(
            f"workspace device={workspace.packed_flat_scratch.device} does not match requested device={device}."
        )
    if workspace.packed_flat_scratch.dtype != dtype:
        raise ValueError(
            f"workspace dtype={workspace.packed_flat_scratch.dtype} does not match requested dtype={dtype}."
        )
    if len(workspace.groups) != (config.num_layers + group_size - 1) // group_size:
        raise ValueError(
            "workspace group count does not match config/group_size. "
            f"Got {len(workspace.groups)} groups for num_layers={config.num_layers}, group_size={group_size}."
        )


def create_packed_materializer_workspace(
    *,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    group_size: int,
    max_rows: int,
    dtype: torch.dtype,
    device: torch.device | str,
    cos_sin_cache: torch.Tensor,
) -> DFlashPackedMaterializerWorkspace:
    config.validate()
    weights.validate(config)
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    if max_rows < 0:
        raise ValueError(f"max_rows must be non-negative, got {max_rows}.")
    device = torch.device(device)
    staged_cos_sin = cos_sin_cache.to(device=device, dtype=torch.float32).contiguous()
    k_norm_weight, k_norm_eps = _prepare_jit_postprocess_constants(
        weights=weights,
        dtype=dtype,
        device=device,
        cos_sin_cache=staged_cos_sin,
    )
    groups: list[DFlashPackedProjectionGroup] = []
    max_out_cols = 0
    for layer_start in range(0, config.num_layers, group_size):
        layer_end = min(layer_start + group_size, config.num_layers)
        group_count = layer_end - layer_start
        out_cols = group_count * 2 * config.kv_size
        flat_weight = (
            weights.kv_proj_weight[layer_start:layer_end]
            .to(device=device, dtype=dtype)
            .contiguous()
            .view(out_cols, config.hidden_size)
        )
        flat_bias = None
        if weights.kv_proj_bias is not None:
            flat_bias = (
                weights.kv_proj_bias[layer_start:layer_end]
                .to(device=device, dtype=dtype)
                .contiguous()
                .view(out_cols)
            )
        groups.append(
            DFlashPackedProjectionGroup(
                layer_start=layer_start,
                group_count=group_count,
                flat_weight_t=flat_weight.transpose(0, 1).contiguous(),
                flat_bias=flat_bias,
                out_cols=out_cols,
            )
        )
        max_out_cols = max(max_out_cols, out_cols)
    packed_flat_scratch = torch.empty(
        (max_rows, max_out_cols),
        dtype=dtype,
        device=device,
    )
    return DFlashPackedMaterializerWorkspace(
        group_size=group_size,
        max_rows=max_rows,
        packed_flat_scratch=packed_flat_scratch,
        k_norm_weight=k_norm_weight,
        k_norm_eps=k_norm_eps,
        cos_sin_cache=staged_cos_sin,
        groups=tuple(groups),
    )


def _materialize_prompt_packed(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    group_size: int,
    chunk_size: int | None,
    cos_sin_cache: torch.Tensor | None,
    inplace: bool,
    provider,
) -> DFlashKVCache:
    hidden, positions, slot_ids = _validate_prompt_inputs(
        cache=cache,
        config=config,
        weights=weights,
        hidden=hidden,
        positions=positions,
        slot_ids=slot_ids,
    )
    updated = cache if inplace else cache.clone()
    num_tokens = int(hidden.shape[0])
    if num_tokens == 0:
        return updated
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    chunk = num_tokens if chunk_size is None else int(chunk_size)
    if chunk <= 0:
        raise ValueError(f"chunk_size must be positive when set, got {chunk_size}.")
    cos_sin = _ensure_cos_sin_cache(
        config=config,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        device=hidden.device,
    )
    for start in range(0, num_tokens, chunk):
        end = min(start + chunk, num_tokens)
        hidden_chunk = hidden[start:end]
        positions_chunk = positions[start:end]
        slot_chunk = slot_ids[start:end]
        for layer_start in range(0, config.num_layers, group_size):
            layer_end = min(layer_start + group_size, config.num_layers)
            packed = _project_packed_group(
                hidden_chunk,
                weights.kv_proj_weight[layer_start:layer_end],
                (
                    None
                    if weights.kv_proj_bias is None
                    else weights.kv_proj_bias[layer_start:layer_end]
                ),
                config=config,
            )
            updated = provider(
                cache=updated,
                config=config,
                weights=weights,
                packed_kv=packed,
                layer_start=layer_start,
                positions=positions_chunk,
                slot_ids=slot_chunk,
                cos_sin_cache=cos_sin,
                inplace=True,
            )
    return updated


def _materialize_commit_packed(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    commit_lens: torch.Tensor,
    group_size: int,
    cos_sin_cache: torch.Tensor | None,
    inplace: bool,
    provider,
) -> DFlashKVCache:
    verify_hidden, positions, slot_ids, commit_lens = _validate_commit_inputs(
        cache=cache,
        config=config,
        weights=weights,
        verify_hidden=verify_hidden,
        positions=positions,
        slot_ids=slot_ids,
        commit_lens=commit_lens,
    )
    updated = cache if inplace else cache.clone()
    bs, block_size, hidden_size = verify_hidden.shape
    if bs == 0 or int(commit_lens.max().item()) == 0:
        return updated
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    cos_sin = _ensure_cos_sin_cache(
        config=config,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        device=verify_hidden.device,
    )
    flat_hidden = verify_hidden.view(bs * block_size, hidden_size)
    for layer_start in range(0, config.num_layers, group_size):
        layer_end = min(layer_start + group_size, config.num_layers)
        packed = _project_packed_group(
            flat_hidden,
            weights.kv_proj_weight[layer_start:layer_end],
            (
                None
                if weights.kv_proj_bias is None
                else weights.kv_proj_bias[layer_start:layer_end]
            ),
            config=config,
        ).view(
            bs,
            block_size,
            layer_end - layer_start,
            2,
            config.num_kv_heads,
            config.head_dim,
        )
        updated = provider(
            cache=updated,
            config=config,
            weights=weights,
            packed_kv=packed,
            layer_start=layer_start,
            positions=positions,
            slot_ids_2d=slot_ids,
            commit_lens=commit_lens,
            cos_sin_cache=cos_sin,
            inplace=True,
        )
    return updated


def _materialize_commit_packed_jit_fast(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    commit_lens: torch.Tensor,
    group_size: int,
    cos_sin_cache: torch.Tensor | None,
    inplace: bool,
    k_norm_weight: torch.Tensor | None,
    k_norm_eps: torch.Tensor | None,
) -> DFlashKVCache:
    updated = cache if inplace else cache.clone()
    bs, block_size, hidden_size = verify_hidden.shape
    if bs == 0 or int(commit_lens.max().item()) == 0:
        return updated
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    cos_sin = _ensure_cos_sin_cache(
        config=config,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        device=verify_hidden.device,
    ).contiguous()
    k_norm_weight, k_norm_eps = (
        _prepare_jit_postprocess_constants(
            weights=weights,
            dtype=verify_hidden.dtype,
            device=verify_hidden.device,
            cos_sin_cache=cos_sin,
        )
        if k_norm_weight is None or k_norm_eps is None
        else (k_norm_weight, k_norm_eps)
    )
    flat_hidden = verify_hidden.reshape(bs * block_size, hidden_size)
    positions = positions.contiguous()
    slot_ids = slot_ids.contiguous()
    commit_lens = commit_lens.contiguous()
    for layer_start in range(0, config.num_layers, group_size):
        layer_end = min(layer_start + group_size, config.num_layers)
        packed = (
            _project_packed_group(
                flat_hidden,
                weights.kv_proj_weight[layer_start:layer_end],
                (
                    None
                    if weights.kv_proj_bias is None
                    else weights.kv_proj_bias[layer_start:layer_end]
                ),
                config=config,
            )
            .view(
                bs,
                block_size,
                layer_end - layer_start,
                2,
                config.num_kv_heads,
                config.head_dim,
            )
            .contiguous()
        )
        updated = postprocess_commit_packed_jit_unchecked(
            cache=updated,
            packed_kv=packed,
            layer_start=layer_start,
            positions=positions,
            slot_ids_2d=slot_ids,
            commit_lens=commit_lens,
            cos_sin_cache=cos_sin,
            k_norm_weight=k_norm_weight,
            k_norm_eps=k_norm_eps,
            inplace=True,
        )
    return updated


def _materialize_prompt_packed_jit_workspace_fast(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    chunk_size: int | None,
    workspace: DFlashPackedMaterializerWorkspace,
    inplace: bool,
) -> DFlashKVCache:
    updated = cache if inplace else cache.clone()
    num_tokens = int(hidden.shape[0])
    if num_tokens == 0:
        return updated
    chunk = num_tokens if chunk_size is None else int(chunk_size)
    if chunk <= 0:
        raise ValueError(f"chunk_size must be positive when set, got {chunk_size}.")
    _validate_workspace(
        workspace=workspace,
        config=config,
        group_size=workspace.group_size,
        row_count=min(num_tokens, chunk),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    positions = positions.contiguous()
    slot_ids = slot_ids.contiguous()
    for start in range(0, num_tokens, chunk):
        end = min(start + chunk, num_tokens)
        hidden_chunk = hidden[start:end].contiguous()
        positions_chunk = positions[start:end]
        slot_chunk = slot_ids[start:end]
        for group in workspace.groups:
            packed = _project_packed_group_mm_out(
                hidden_chunk,
                group,
                workspace=workspace,
                config=config,
            )
            updated = postprocess_prompt_packed_jit_unchecked(
                cache=updated,
                packed_kv=packed,
                layer_start=group.layer_start,
                positions=positions_chunk,
                slot_ids=slot_chunk,
                cos_sin_cache=workspace.cos_sin_cache,
                k_norm_weight=workspace.k_norm_weight,
                k_norm_eps=workspace.k_norm_eps,
                inplace=True,
            )
    return updated


def _materialize_commit_packed_jit_workspace_fast(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    commit_lens: torch.Tensor,
    workspace: DFlashPackedMaterializerWorkspace,
    inplace: bool,
) -> DFlashKVCache:
    updated = cache if inplace else cache.clone()
    bs, block_size, hidden_size = verify_hidden.shape
    if bs == 0 or int(commit_lens.max().item()) == 0:
        return updated
    _validate_workspace(
        workspace=workspace,
        config=config,
        group_size=workspace.group_size,
        row_count=bs * block_size,
        device=verify_hidden.device,
        dtype=verify_hidden.dtype,
    )
    flat_hidden = verify_hidden.reshape(bs * block_size, hidden_size).contiguous()
    positions = positions.contiguous()
    slot_ids = slot_ids.contiguous()
    commit_lens = commit_lens.contiguous()
    for group in workspace.groups:
        packed = _project_packed_group_mm_out(
            flat_hidden,
            group,
            workspace=workspace,
            config=config,
        ).view(
            bs,
            block_size,
            group.group_count,
            2,
            config.num_kv_heads,
            config.head_dim,
        )
        updated = postprocess_commit_packed_jit_unchecked(
            cache=updated,
            packed_kv=packed,
            layer_start=group.layer_start,
            positions=positions,
            slot_ids_2d=slot_ids,
            commit_lens=commit_lens,
            cos_sin_cache=workspace.cos_sin_cache,
            k_norm_weight=workspace.k_norm_weight,
            k_norm_eps=workspace.k_norm_eps,
            inplace=True,
        )
    return updated


def _materialize_commit_packed_compact(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    commit_lens: torch.Tensor,
    group_size: int,
    cos_sin_cache: torch.Tensor | None,
    inplace: bool,
    compact_provider,
    postprocess_provider,
    hidden_scratch: torch.Tensor | None,
    positions_scratch: torch.Tensor | None,
    slot_ids_scratch: torch.Tensor | None,
) -> DFlashKVCache:
    verify_hidden, positions, slot_ids, commit_lens = _validate_commit_inputs(
        cache=cache,
        config=config,
        weights=weights,
        verify_hidden=verify_hidden,
        positions=positions,
        slot_ids=slot_ids,
        commit_lens=commit_lens,
    )
    updated = cache if inplace else cache.clone()
    bs, block_size, _ = verify_hidden.shape
    if bs == 0 or int(commit_lens.max().item()) == 0:
        return updated
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    cos_sin = _ensure_cos_sin_cache(
        config=config,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        device=verify_hidden.device,
    )
    compact_hidden, compact_positions, compact_slot_ids = compact_provider(
        verify_hidden=verify_hidden,
        positions=positions,
        slot_ids_2d=slot_ids,
        commit_lens=commit_lens,
        hidden_out=hidden_scratch,
        positions_out=positions_scratch,
        slot_ids_out=slot_ids_scratch,
    )
    if int(compact_hidden.shape[0]) == 0:
        return updated
    for layer_start in range(0, config.num_layers, group_size):
        layer_end = min(layer_start + group_size, config.num_layers)
        packed = _project_packed_group(
            compact_hidden,
            weights.kv_proj_weight[layer_start:layer_end],
            (
                None
                if weights.kv_proj_bias is None
                else weights.kv_proj_bias[layer_start:layer_end]
            ),
            config=config,
        )
        updated = postprocess_provider(
            cache=updated,
            config=config,
            weights=weights,
            packed_kv=packed,
            layer_start=layer_start,
            positions=compact_positions,
            slot_ids=compact_slot_ids,
            cos_sin_cache=cos_sin,
            inplace=True,
        )
    return updated


def materialize_prompt_packed_triton(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    group_size: int,
    chunk_size: int | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    inplace: bool = False,
) -> DFlashKVCache:
    return _materialize_prompt_packed(
        cache=cache,
        config=config,
        weights=weights,
        hidden=hidden,
        positions=positions,
        slot_ids=slot_ids,
        group_size=group_size,
        chunk_size=chunk_size,
        cos_sin_cache=cos_sin_cache,
        inplace=inplace,
        provider=postprocess_prompt_packed_triton,
    )


def materialize_prompt_packed_jit(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    group_size: int,
    chunk_size: int | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    inplace: bool = False,
) -> DFlashKVCache:
    return _materialize_prompt_packed(
        cache=cache,
        config=config,
        weights=weights,
        hidden=hidden,
        positions=positions,
        slot_ids=slot_ids,
        group_size=group_size,
        chunk_size=chunk_size,
        cos_sin_cache=cos_sin_cache,
        inplace=inplace,
        provider=postprocess_prompt_packed_jit,
    )


def materialize_prompt_packed_jit_fast(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    group_size: int,
    chunk_size: int | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    inplace: bool = False,
    k_norm_weight: torch.Tensor | None = None,
    k_norm_eps: torch.Tensor | None = None,
) -> DFlashKVCache:
    updated = cache if inplace else cache.clone()
    num_tokens = int(hidden.shape[0])
    if num_tokens == 0:
        return updated
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    chunk = num_tokens if chunk_size is None else int(chunk_size)
    if chunk <= 0:
        raise ValueError(f"chunk_size must be positive when set, got {chunk_size}.")
    cos_sin = _ensure_cos_sin_cache(
        config=config,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        device=hidden.device,
    ).contiguous()
    k_norm_weight, k_norm_eps = (
        _prepare_jit_postprocess_constants(
            weights=weights,
            dtype=hidden.dtype,
            device=hidden.device,
            cos_sin_cache=cos_sin,
        )
        if k_norm_weight is None or k_norm_eps is None
        else (k_norm_weight, k_norm_eps)
    )
    positions = positions.contiguous()
    slot_ids = slot_ids.contiguous()
    for start in range(0, num_tokens, chunk):
        end = min(start + chunk, num_tokens)
        hidden_chunk = hidden[start:end]
        positions_chunk = positions[start:end]
        slot_chunk = slot_ids[start:end]
        for layer_start in range(0, config.num_layers, group_size):
            layer_end = min(layer_start + group_size, config.num_layers)
            packed = _project_packed_group(
                hidden_chunk,
                weights.kv_proj_weight[layer_start:layer_end],
                (
                    None
                    if weights.kv_proj_bias is None
                    else weights.kv_proj_bias[layer_start:layer_end]
                ),
                config=config,
            ).contiguous()
            updated = postprocess_prompt_packed_jit_unchecked(
                cache=updated,
                packed_kv=packed,
                layer_start=layer_start,
                positions=positions_chunk,
                slot_ids=slot_chunk,
                cos_sin_cache=cos_sin,
                k_norm_weight=k_norm_weight,
                k_norm_eps=k_norm_eps,
                inplace=True,
            )
    return updated


def materialize_prompt_packed_jit_workspace_fast(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    group_size: int,
    chunk_size: int | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    workspace: DFlashPackedMaterializerWorkspace | None = None,
    inplace: bool = False,
) -> DFlashKVCache:
    hidden, positions, slot_ids = _validate_prompt_inputs(
        cache=cache,
        config=config,
        weights=weights,
        hidden=hidden,
        positions=positions,
        slot_ids=slot_ids,
    )
    if workspace is None:
        if cos_sin_cache is None:
            raise ValueError(
                "cos_sin_cache is required when workspace is not provided."
            )
        row_limit = int(hidden.shape[0]) if chunk_size is None else int(chunk_size)
        workspace = create_packed_materializer_workspace(
            config=config,
            weights=weights,
            group_size=group_size,
            max_rows=row_limit,
            dtype=hidden.dtype,
            device=hidden.device,
            cos_sin_cache=cos_sin_cache,
        )
    return _materialize_prompt_packed_jit_workspace_fast(
        cache=cache,
        config=config,
        hidden=hidden,
        positions=positions,
        slot_ids=slot_ids,
        chunk_size=chunk_size,
        workspace=workspace,
        inplace=inplace,
    )


def materialize_commit_packed_triton(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    commit_lens: torch.Tensor,
    group_size: int,
    cos_sin_cache: torch.Tensor | None = None,
    inplace: bool = False,
) -> DFlashKVCache:
    return _materialize_commit_packed(
        cache=cache,
        config=config,
        weights=weights,
        verify_hidden=verify_hidden,
        positions=positions,
        slot_ids=slot_ids,
        commit_lens=commit_lens,
        group_size=group_size,
        cos_sin_cache=cos_sin_cache,
        inplace=inplace,
        provider=postprocess_commit_packed_triton,
    )


def materialize_commit_packed_jit(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    commit_lens: torch.Tensor,
    group_size: int,
    cos_sin_cache: torch.Tensor | None = None,
    inplace: bool = False,
) -> DFlashKVCache:
    return _materialize_commit_packed(
        cache=cache,
        config=config,
        weights=weights,
        verify_hidden=verify_hidden,
        positions=positions,
        slot_ids=slot_ids,
        commit_lens=commit_lens,
        group_size=group_size,
        cos_sin_cache=cos_sin_cache,
        inplace=inplace,
        provider=postprocess_commit_packed_jit,
    )


def materialize_commit_packed_jit_fast(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    commit_lens: torch.Tensor,
    group_size: int,
    cos_sin_cache: torch.Tensor | None = None,
    inplace: bool = False,
    k_norm_weight: torch.Tensor | None = None,
    k_norm_eps: torch.Tensor | None = None,
) -> DFlashKVCache:
    return _materialize_commit_packed_jit_fast(
        cache=cache,
        config=config,
        weights=weights,
        verify_hidden=verify_hidden,
        positions=positions,
        slot_ids=slot_ids,
        commit_lens=commit_lens,
        group_size=group_size,
        cos_sin_cache=cos_sin_cache,
        inplace=inplace,
        k_norm_weight=k_norm_weight,
        k_norm_eps=k_norm_eps,
    )


def materialize_commit_packed_jit_workspace_fast(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    commit_lens: torch.Tensor,
    group_size: int,
    cos_sin_cache: torch.Tensor | None = None,
    workspace: DFlashPackedMaterializerWorkspace | None = None,
    inplace: bool = False,
) -> DFlashKVCache:
    verify_hidden, positions, slot_ids, commit_lens = _validate_commit_inputs(
        cache=cache,
        config=config,
        weights=weights,
        verify_hidden=verify_hidden,
        positions=positions,
        slot_ids=slot_ids,
        commit_lens=commit_lens,
    )
    if workspace is None:
        if cos_sin_cache is None:
            raise ValueError(
                "cos_sin_cache is required when workspace is not provided."
            )
        workspace = create_packed_materializer_workspace(
            config=config,
            weights=weights,
            group_size=group_size,
            max_rows=int(verify_hidden.shape[0] * verify_hidden.shape[1]),
            dtype=verify_hidden.dtype,
            device=verify_hidden.device,
            cos_sin_cache=cos_sin_cache,
        )
    return _materialize_commit_packed_jit_workspace_fast(
        cache=cache,
        config=config,
        verify_hidden=verify_hidden,
        positions=positions,
        slot_ids=slot_ids,
        commit_lens=commit_lens,
        workspace=workspace,
        inplace=inplace,
    )


def materialize_commit_packed_compact_triton(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    commit_lens: torch.Tensor,
    group_size: int,
    cos_sin_cache: torch.Tensor | None = None,
    inplace: bool = False,
    hidden_scratch: torch.Tensor | None = None,
    positions_scratch: torch.Tensor | None = None,
    slot_ids_scratch: torch.Tensor | None = None,
) -> DFlashKVCache:
    return _materialize_commit_packed_compact(
        cache=cache,
        config=config,
        weights=weights,
        verify_hidden=verify_hidden,
        positions=positions,
        slot_ids=slot_ids,
        commit_lens=commit_lens,
        group_size=group_size,
        cos_sin_cache=cos_sin_cache,
        inplace=inplace,
        compact_provider=compact_commit_triton,
        postprocess_provider=postprocess_prompt_packed_triton,
        hidden_scratch=hidden_scratch,
        positions_scratch=positions_scratch,
        slot_ids_scratch=slot_ids_scratch,
    )


def materialize_commit_packed_compact_jit(
    *,
    cache: DFlashKVCache,
    config: DFlashMaterializerConfig,
    weights: DFlashMaterializerWeights,
    verify_hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_ids: torch.Tensor,
    commit_lens: torch.Tensor,
    group_size: int,
    cos_sin_cache: torch.Tensor | None = None,
    inplace: bool = False,
    hidden_scratch: torch.Tensor | None = None,
    positions_scratch: torch.Tensor | None = None,
    slot_ids_scratch: torch.Tensor | None = None,
) -> DFlashKVCache:
    return _materialize_commit_packed_compact(
        cache=cache,
        config=config,
        weights=weights,
        verify_hidden=verify_hidden,
        positions=positions,
        slot_ids=slot_ids,
        commit_lens=commit_lens,
        group_size=group_size,
        cos_sin_cache=cos_sin_cache,
        inplace=inplace,
        compact_provider=compact_commit_jit,
        postprocess_provider=postprocess_prompt_packed_jit,
        hidden_scratch=hidden_scratch,
        positions_scratch=positions_scratch,
        slot_ids_scratch=slot_ids_scratch,
    )
