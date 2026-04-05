from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.speculative.dflash.contracts import (
    STATUS_ACTIVE,
    STATUS_FINISHED,
    DFlashKVCache,
    DFlashMaterializerConfig,
    DFlashMaterializerWeights,
    DFlashRequestStateTable,
)


@dataclass(frozen=True)
class PrepareBlockFixture:
    state: DFlashRequestStateTable
    req_pool_indices: torch.Tensor
    req_generation: torch.Tensor
    req_to_token: torch.Tensor
    block_size: int
    mask_token_id: int


@dataclass(frozen=True)
class AcceptBonusFixture:
    emit_ids: torch.Tensor
    target_top1: torch.Tensor
    active_mask: torch.Tensor
    eos_token_ids: torch.Tensor
    stop_token_ids: torch.Tensor


@dataclass(frozen=True)
class PublishStateFixture:
    state: DFlashRequestStateTable
    req_pool_indices: torch.Tensor
    req_generation: torch.Tensor
    commit_lens: torch.Tensor
    bonus_ids: torch.Tensor
    gpu_stop_flags: torch.Tensor


@dataclass(frozen=True)
class AcceptPublishFixture:
    state: DFlashRequestStateTable
    req_pool_indices: torch.Tensor
    req_generation: torch.Tensor
    emit_ids: torch.Tensor
    target_top1: torch.Tensor
    active_mask: torch.Tensor
    eos_token_ids: torch.Tensor
    stop_token_ids: torch.Tensor


@dataclass(frozen=True)
class PromptMaterializerFixture:
    cache: DFlashKVCache
    config: DFlashMaterializerConfig
    weights: DFlashMaterializerWeights
    hidden: torch.Tensor
    positions: torch.Tensor
    slot_ids: torch.Tensor
    cos_sin_cache: torch.Tensor


@dataclass(frozen=True)
class CommitMaterializerFixture:
    cache: DFlashKVCache
    config: DFlashMaterializerConfig
    weights: DFlashMaterializerWeights
    verify_hidden: torch.Tensor
    positions: torch.Tensor
    slot_ids: torch.Tensor
    commit_lens: torch.Tensor
    cos_sin_cache: torch.Tensor


@dataclass(frozen=True)
class PromptProjectionFixture:
    config: DFlashMaterializerConfig
    weights: DFlashMaterializerWeights
    hidden: torch.Tensor
    positions: torch.Tensor


@dataclass(frozen=True)
class CommitProjectionFixture:
    config: DFlashMaterializerConfig
    weights: DFlashMaterializerWeights
    verify_hidden: torch.Tensor
    positions: torch.Tensor


@dataclass(frozen=True)
class PromptWriteFixture:
    cache: DFlashKVCache
    config: DFlashMaterializerConfig
    slot_ids: torch.Tensor
    cache_k: torch.Tensor
    cache_v: torch.Tensor
    dummy_slot_id: int


@dataclass(frozen=True)
class CommitWriteFixture:
    cache: DFlashKVCache
    config: DFlashMaterializerConfig
    slot_ids_2d: torch.Tensor
    commit_lens: torch.Tensor
    cache_k: torch.Tensor
    cache_v: torch.Tensor
    dummy_slot_id: int


@dataclass(frozen=True)
class DirectEmbeddingFixture:
    embedding_table: torch.Tensor
    query_input_ids: torch.Tensor
    first_token_ids: torch.Tensor
    mask_token_id: int
    block_size: int


def make_prepare_block_fixture(
    *,
    bucket_bs: int = 8,
    num_req_slots: int = 16,
    req_to_token_width: int = 256,
    block_size: int = 16,
    mask_token_id: int = 151643,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> PrepareBlockFixture:
    if bucket_bs <= 0 or num_req_slots <= 0 or req_to_token_width <= 0:
        raise ValueError(
            "bucket_bs, num_req_slots, and req_to_token_width must be > 0."
        )

    device = torch.device(device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)

    committed_len = torch.randint(
        low=4,
        high=max(req_to_token_width - max(block_size * 2, 8), 5),
        size=(num_req_slots,),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    headroom = torch.randint(
        low=max(block_size * 2, 8),
        high=max(block_size * 4, 16),
        size=(num_req_slots,),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    reserved_len = torch.minimum(
        committed_len + headroom,
        torch.full_like(committed_len, req_to_token_width),
    )
    next_verified_id = torch.randint(
        low=100,
        high=50000,
        size=(num_req_slots,),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    generation = torch.randint(
        low=1,
        high=1000,
        size=(num_req_slots,),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    status_flags = torch.full(
        (num_req_slots,),
        STATUS_ACTIVE,
        dtype=torch.int32,
        device=device,
    )
    if num_req_slots > 2:
        status_flags[2] = STATUS_ACTIVE | STATUS_FINISHED

    req_to_token = torch.arange(
        num_req_slots * req_to_token_width, device=device, dtype=torch.int64
    ).view(num_req_slots, req_to_token_width)

    req_pool_indices = (
        torch.arange(bucket_bs, device=device, dtype=torch.int32) % num_req_slots
    )
    req_generation = generation.index_select(
        0, req_pool_indices.to(torch.int64)
    ).clone()
    if bucket_bs > 1:
        req_generation[1] += 1
    if bucket_bs > 2:
        req_pool_indices[2] = 2
        req_generation[2] = generation[2]

    state = DFlashRequestStateTable(
        committed_len=committed_len,
        reserved_len=reserved_len,
        next_verified_id=next_verified_id,
        generation=generation,
        status_flags=status_flags,
    )
    return PrepareBlockFixture(
        state=state,
        req_pool_indices=req_pool_indices,
        req_generation=req_generation,
        req_to_token=req_to_token,
        block_size=block_size,
        mask_token_id=mask_token_id,
    )


def make_direct_embedding_fixture(
    *,
    bucket_bs: int = 32,
    block_size: int = 16,
    vocab_size: int = 32768,
    hidden_size: int = 1024,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> DirectEmbeddingFixture:
    if bucket_bs <= 0 or block_size <= 0:
        raise ValueError(
            f"bucket_bs and block_size must be positive, got {bucket_bs}, {block_size}."
        )
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be at least 2, got {vocab_size}.")
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}.")

    device = torch.device(device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed + 53)
    mask_token_id = vocab_size - 1
    embedding_table = torch.randn(
        vocab_size,
        hidden_size,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    first_token_ids = torch.randint(
        low=0,
        high=mask_token_id,
        size=(bucket_bs,),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    if bucket_bs > 1:
        first_token_ids[1] = mask_token_id
    if bucket_bs > 3:
        first_token_ids[3] = mask_token_id
    query_input_ids = torch.full(
        (bucket_bs, block_size),
        mask_token_id,
        dtype=torch.int32,
        device=device,
    )
    query_input_ids[:, 0] = first_token_ids
    return DirectEmbeddingFixture(
        embedding_table=embedding_table,
        query_input_ids=query_input_ids,
        first_token_ids=first_token_ids,
        mask_token_id=mask_token_id,
        block_size=block_size,
    )


def make_accept_bonus_fixture(
    *,
    bucket_bs: int = 8,
    block_size: int = 16,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> AcceptBonusFixture:
    if bucket_bs <= 0 or block_size <= 0:
        raise ValueError("bucket_bs and block_size must be > 0.")

    device = torch.device(device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)

    emit_ids = torch.randint(
        low=10,
        high=1000,
        size=(bucket_bs, block_size),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    target_top1 = torch.randint(
        low=10,
        high=1000,
        size=(bucket_bs, block_size),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    active_mask = torch.ones((bucket_bs,), dtype=torch.bool, device=device)
    if bucket_bs > 0:
        target_top1[0, :-1] = emit_ids[0, 1:]
    if bucket_bs > 1:
        target_top1[1, 0] = emit_ids[1, 1]
        target_top1[1, 1] = emit_ids[1, 2]
    if bucket_bs > 2:
        active_mask[2] = False

    eos_token_ids = emit_ids[0, :1].clone()
    stop_token_ids = (
        emit_ids[1, 1:2].clone() if bucket_bs > 1 else emit_ids[0, 1:2].clone()
    )
    return AcceptBonusFixture(
        emit_ids=emit_ids,
        target_top1=target_top1,
        active_mask=active_mask,
        eos_token_ids=eos_token_ids,
        stop_token_ids=stop_token_ids,
    )


def make_publish_state_fixture(
    *,
    bucket_bs: int = 8,
    num_req_slots: int = 16,
    req_to_token_width: int = 256,
    block_size: int = 16,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> PublishStateFixture:
    prepare = make_prepare_block_fixture(
        bucket_bs=bucket_bs,
        num_req_slots=num_req_slots,
        req_to_token_width=req_to_token_width,
        block_size=block_size,
        device=device,
        seed=seed,
    )
    device = torch.device(device)
    commit_lens = torch.ones((bucket_bs,), dtype=torch.int32, device=device)
    bonus_ids = torch.arange(7000, 7000 + bucket_bs, dtype=torch.int32, device=device)
    gpu_stop_flags = torch.zeros((bucket_bs,), dtype=torch.int32, device=device)
    if bucket_bs > 0:
        commit_lens[0] = min(block_size, 3)
    if bucket_bs > 1:
        gpu_stop_flags[1] = STATUS_FINISHED
    if bucket_bs > 2:
        commit_lens[2] = 0

    return PublishStateFixture(
        state=prepare.state,
        req_pool_indices=prepare.req_pool_indices,
        req_generation=prepare.req_generation,
        commit_lens=commit_lens,
        bonus_ids=bonus_ids,
        gpu_stop_flags=gpu_stop_flags,
    )


def make_accept_publish_fixture(
    *,
    bucket_bs: int = 8,
    num_req_slots: int = 16,
    req_to_token_width: int = 256,
    block_size: int = 16,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> AcceptPublishFixture:
    prepare = make_prepare_block_fixture(
        bucket_bs=bucket_bs,
        num_req_slots=num_req_slots,
        req_to_token_width=req_to_token_width,
        block_size=block_size,
        device=device,
        seed=seed,
    )
    accept = make_accept_bonus_fixture(
        bucket_bs=bucket_bs,
        block_size=block_size,
        device=device,
        seed=seed + 1,
    )
    return AcceptPublishFixture(
        state=prepare.state,
        req_pool_indices=prepare.req_pool_indices,
        req_generation=prepare.req_generation,
        emit_ids=accept.emit_ids,
        target_top1=accept.target_top1,
        active_mask=accept.active_mask,
        eos_token_ids=accept.eos_token_ids,
        stop_token_ids=accept.stop_token_ids,
    )


def _make_materializer_components(
    *,
    num_layers: int,
    hidden_size: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    num_slots: int,
    device: torch.device | str,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[DFlashKVCache, DFlashMaterializerConfig, DFlashMaterializerWeights]:
    device = torch.device(device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)

    config = DFlashMaterializerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
    )
    kv_size = config.kv_size
    weights = DFlashMaterializerWeights(
        kv_proj_weight=torch.randn(
            num_layers,
            2 * kv_size,
            hidden_size,
            dtype=dtype,
            device=device,
            generator=generator,
        ),
        kv_proj_bias=torch.randn(
            num_layers,
            2 * kv_size,
            dtype=dtype,
            device=device,
            generator=generator,
        ),
        k_norm_weight=torch.randn(
            num_layers,
            head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        ).abs_()
        + 0.1,
        k_norm_eps=torch.full(
            (num_layers,),
            1e-6,
            dtype=torch.float32,
            device=device,
        ),
    )
    cache = DFlashKVCache(
        k_cache=torch.randn(
            num_layers,
            num_slots,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        ),
        v_cache=torch.randn(
            num_layers,
            num_slots,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        ),
    )
    return cache, config, weights


def _make_rope_cos_sin_cache(
    *,
    rotary_dim: int,
    rope_theta: float,
    max_position: int,
    device: torch.device | str,
) -> torch.Tensor:
    device = torch.device(device)
    half = rotary_dim // 2
    positions = torch.arange(max_position + 1, device=device, dtype=torch.float32).view(
        -1, 1
    )
    inv_idx = torch.arange(half, device=device, dtype=torch.float32)
    inv_freq = torch.pow(
        torch.tensor(float(rope_theta), device=device, dtype=torch.float32),
        -(2.0 * inv_idx / float(rotary_dim)),
    ).view(1, half)
    freqs = positions * inv_freq
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1).contiguous()


def make_prompt_materializer_fixture(
    *,
    num_layers: int = 8,
    hidden_size: int = 64,
    num_kv_heads: int = 4,
    head_dim: int = 16,
    rotary_dim: int = 16,
    num_slots: int = 1024,
    num_tokens: int = 128,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> PromptMaterializerFixture:
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}.")
    cache, config, weights = _make_materializer_components(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        num_slots=num_slots,
        device=device,
        seed=seed,
        dtype=dtype,
    )
    device = torch.device(device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed + 17)
    hidden = torch.randn(
        num_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    positions = torch.randint(
        low=0,
        high=max(num_slots // 2, 1),
        size=(num_tokens,),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    slot_ids = (
        torch.randperm(num_slots, device=device, generator=generator)[:num_tokens]
        .to(torch.int64)
        .contiguous()
    )
    max_position = int(positions.max().item()) if num_tokens > 0 else 0
    return PromptMaterializerFixture(
        cache=cache,
        config=config,
        weights=weights,
        hidden=hidden,
        positions=positions,
        slot_ids=slot_ids,
        cos_sin_cache=_make_rope_cos_sin_cache(
            rotary_dim=rotary_dim,
            rope_theta=config.rope_theta,
            max_position=max_position,
            device=device,
        ),
    )


def make_prompt_projection_fixture(
    *,
    num_layers: int = 8,
    hidden_size: int = 64,
    num_kv_heads: int = 4,
    head_dim: int = 16,
    rotary_dim: int = 16,
    num_slots: int = 1024,
    num_tokens: int = 128,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> PromptProjectionFixture:
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}.")
    _, config, weights = _make_materializer_components(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        num_slots=num_slots,
        device=device,
        seed=seed,
        dtype=dtype,
    )
    device = torch.device(device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed + 23)
    hidden = torch.randn(
        num_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    positions = torch.randint(
        low=0,
        high=max(num_slots // 2, 1),
        size=(num_tokens,),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    return PromptProjectionFixture(
        config=config,
        weights=weights,
        hidden=hidden,
        positions=positions,
    )


def make_commit_materializer_fixture(
    *,
    num_layers: int = 8,
    hidden_size: int = 64,
    num_kv_heads: int = 4,
    head_dim: int = 16,
    rotary_dim: int = 16,
    num_slots: int = 1024,
    batch_size: int = 16,
    block_size: int = 16,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> CommitMaterializerFixture:
    if batch_size <= 0 or block_size <= 0:
        raise ValueError(
            f"batch_size and block_size must be positive, got {batch_size}, {block_size}."
        )
    cache, config, weights = _make_materializer_components(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        num_slots=num_slots,
        device=device,
        seed=seed,
        dtype=dtype,
    )
    device = torch.device(device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed + 29)
    verify_hidden = torch.randn(
        batch_size,
        block_size,
        hidden_size,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    base_positions = torch.randint(
        low=0,
        high=max(num_slots // 2, 1),
        size=(batch_size,),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    offsets = torch.arange(block_size, device=device, dtype=torch.int64)
    positions = base_positions[:, None] + offsets[None, :]
    slot_ids = torch.randperm(
        num_slots,
        device=device,
        generator=generator,
    )[
        : batch_size * block_size
    ].view(batch_size, block_size)
    commit_lens = torch.randint(
        low=0,
        high=block_size + 1,
        size=(batch_size,),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    if batch_size > 0:
        commit_lens[0] = block_size
    if batch_size > 1:
        commit_lens[1] = 0
    max_position = int(positions.max().item()) if batch_size > 0 else 0
    return CommitMaterializerFixture(
        cache=cache,
        config=config,
        weights=weights,
        verify_hidden=verify_hidden,
        positions=positions,
        slot_ids=slot_ids.to(torch.int64),
        commit_lens=commit_lens,
        cos_sin_cache=_make_rope_cos_sin_cache(
            rotary_dim=rotary_dim,
            rope_theta=config.rope_theta,
            max_position=max_position,
            device=device,
        ),
    )


def make_commit_projection_fixture(
    *,
    num_layers: int = 8,
    hidden_size: int = 64,
    num_kv_heads: int = 4,
    head_dim: int = 16,
    rotary_dim: int = 16,
    num_slots: int = 1024,
    batch_size: int = 16,
    block_size: int = 16,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> CommitProjectionFixture:
    if batch_size <= 0 or block_size <= 0:
        raise ValueError(
            f"batch_size and block_size must be positive, got {batch_size}, {block_size}."
        )
    _, config, weights = _make_materializer_components(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        num_slots=num_slots,
        device=device,
        seed=seed,
        dtype=dtype,
    )
    device = torch.device(device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed + 31)
    verify_hidden = torch.randn(
        batch_size,
        block_size,
        hidden_size,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    base_positions = torch.randint(
        low=0,
        high=max(num_slots // 2, 1),
        size=(batch_size,),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    offsets = torch.arange(block_size, device=device, dtype=torch.int64)
    positions = base_positions[:, None] + offsets[None, :]
    return CommitProjectionFixture(
        config=config,
        weights=weights,
        verify_hidden=verify_hidden,
        positions=positions,
    )


def make_prompt_write_fixture(
    *,
    num_layers: int = 8,
    num_kv_heads: int = 4,
    head_dim: int = 16,
    num_slots: int = 1024,
    num_tokens: int = 128,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> PromptWriteFixture:
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}.")
    cache, config, _ = _make_materializer_components(
        num_layers=num_layers,
        hidden_size=head_dim * num_kv_heads * 2,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=head_dim,
        num_slots=num_slots,
        device=device,
        seed=seed,
    )
    device = torch.device(device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed + 41)
    slot_ids = (
        torch.randperm(num_slots - 1, device=device, generator=generator)[:num_tokens]
        .to(torch.int64)
        .contiguous()
    )
    cache_k = torch.randn(
        num_layers,
        num_tokens,
        num_kv_heads,
        head_dim,
        dtype=cache.k_cache.dtype,
        device=device,
        generator=generator,
    )
    cache_v = torch.randn(
        num_layers,
        num_tokens,
        num_kv_heads,
        head_dim,
        dtype=cache.v_cache.dtype,
        device=device,
        generator=generator,
    )
    return PromptWriteFixture(
        cache=cache,
        config=config,
        slot_ids=slot_ids,
        cache_k=cache_k,
        cache_v=cache_v,
        dummy_slot_id=num_slots - 1,
    )


def make_commit_write_fixture(
    *,
    num_layers: int = 8,
    num_kv_heads: int = 4,
    head_dim: int = 16,
    num_slots: int = 1024,
    batch_size: int = 16,
    block_size: int = 16,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> CommitWriteFixture:
    if batch_size <= 0 or block_size <= 0:
        raise ValueError(
            f"batch_size and block_size must be positive, got {batch_size}, {block_size}."
        )
    cache, config, _ = _make_materializer_components(
        num_layers=num_layers,
        hidden_size=head_dim * num_kv_heads * 2,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=head_dim,
        num_slots=num_slots,
        device=device,
        seed=seed,
    )
    device = torch.device(device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed + 53)
    slot_ids_2d = (
        torch.randperm(
            num_slots - 1,
            device=device,
            generator=generator,
        )[: batch_size * block_size]
        .view(batch_size, block_size)
        .to(torch.int64)
    )
    commit_lens = torch.randint(
        low=0,
        high=block_size + 1,
        size=(batch_size,),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    if batch_size > 0:
        commit_lens[0] = block_size
    if batch_size > 1:
        commit_lens[1] = 0
    cache_k = torch.randn(
        num_layers,
        batch_size,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=cache.k_cache.dtype,
        device=device,
        generator=generator,
    )
    cache_v = torch.randn(
        num_layers,
        batch_size,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=cache.v_cache.dtype,
        device=device,
        generator=generator,
    )
    return CommitWriteFixture(
        cache=cache,
        config=config,
        slot_ids_2d=slot_ids_2d,
        commit_lens=commit_lens,
        cache_k=cache_k,
        cache_v=cache_v,
        dummy_slot_id=num_slots - 1,
    )
