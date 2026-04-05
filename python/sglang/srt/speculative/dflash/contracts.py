from __future__ import annotations

from dataclasses import dataclass

import torch

STATUS_ACTIVE = 1 << 0
STATUS_EOS_SEEN = 1 << 1
STATUS_FINISHED = 1 << 2
STATUS_CANCELED = 1 << 3
STATUS_STOPPED_BY_TOKEN = 1 << 4

GPU_STOP_MASK = (
    STATUS_EOS_SEEN | STATUS_FINISHED | STATUS_CANCELED | STATUS_STOPPED_BY_TOKEN
)


@dataclass
class DFlashRequestStateTable:
    committed_len: torch.Tensor
    reserved_len: torch.Tensor
    next_verified_id: torch.Tensor
    generation: torch.Tensor
    status_flags: torch.Tensor

    def clone(self) -> "DFlashRequestStateTable":
        return DFlashRequestStateTable(
            committed_len=self.committed_len.clone(),
            reserved_len=self.reserved_len.clone(),
            next_verified_id=self.next_verified_id.clone(),
            generation=self.generation.clone(),
            status_flags=self.status_flags.clone(),
        )

    def validate(self) -> None:
        tensors = (
            self.committed_len,
            self.reserved_len,
            self.next_verified_id,
            self.generation,
            self.status_flags,
        )
        first_shape = tensors[0].shape
        for tensor in tensors[1:]:
            if tensor.shape != first_shape:
                raise ValueError(
                    "All request-state tensors must have the same shape. "
                    f"Expected {tuple(first_shape)}, got {tuple(tensor.shape)}."
                )
        if self.committed_len.ndim != 1:
            raise ValueError(
                "Request-state tensors must be 1D. "
                f"Got committed_len.ndim={self.committed_len.ndim}."
            )
        if torch.any(self.committed_len > self.reserved_len):
            raise ValueError(
                "committed_len must not exceed reserved_len in the request state table."
            )


@dataclass
class DFlashPrepareBlockResult:
    query_input_ids: torch.Tensor
    query_positions: torch.Tensor
    query_slot_ids: torch.Tensor
    emit_ids: torch.Tensor
    sample_indices: torch.Tensor | None
    active_mask: torch.Tensor


@dataclass
class DFlashDirectEmbeddingResult:
    query_embeds: torch.Tensor


@dataclass
class DFlashAcceptBonusResult:
    accept_lens: torch.Tensor
    commit_lens: torch.Tensor
    bonus_ids: torch.Tensor
    gpu_stop_flags: torch.Tensor


@dataclass
class DFlashAcceptPublishResult:
    accept: DFlashAcceptBonusResult
    state: DFlashRequestStateTable


@dataclass(frozen=True)
class DFlashMaterializerConfig:
    num_layers: int
    hidden_size: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float = 1000000.0

    @property
    def kv_size(self) -> int:
        return int(self.num_kv_heads) * int(self.head_dim)

    def validate(self) -> None:
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}.")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}.")
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {self.num_kv_heads}.")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}.")
        if self.rotary_dim <= 0 or self.rotary_dim > self.head_dim:
            raise ValueError(
                "rotary_dim must be in (0, head_dim]. "
                f"Got rotary_dim={self.rotary_dim}, head_dim={self.head_dim}."
            )
        if self.rotary_dim % 2 != 0:
            raise ValueError(
                f"rotary_dim must be even for neox-style RoPE, got {self.rotary_dim}."
            )
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}.")


@dataclass
class DFlashMaterializerWeights:
    kv_proj_weight: torch.Tensor
    kv_proj_bias: torch.Tensor | None
    k_norm_weight: torch.Tensor
    k_norm_eps: torch.Tensor

    def validate(self, config: DFlashMaterializerConfig) -> None:
        config.validate()
        expected_kv_shape = (
            config.num_layers,
            2 * config.kv_size,
            config.hidden_size,
        )
        if tuple(self.kv_proj_weight.shape) != expected_kv_shape:
            raise ValueError(
                "kv_proj_weight shape mismatch. "
                f"Expected {expected_kv_shape}, got {tuple(self.kv_proj_weight.shape)}."
            )
        if self.kv_proj_bias is not None:
            expected_bias_shape = (config.num_layers, 2 * config.kv_size)
            if tuple(self.kv_proj_bias.shape) != expected_bias_shape:
                raise ValueError(
                    "kv_proj_bias shape mismatch. "
                    f"Expected {expected_bias_shape}, got {tuple(self.kv_proj_bias.shape)}."
                )
        expected_norm_shape = (config.num_layers, config.head_dim)
        if tuple(self.k_norm_weight.shape) != expected_norm_shape:
            raise ValueError(
                "k_norm_weight shape mismatch. "
                f"Expected {expected_norm_shape}, got {tuple(self.k_norm_weight.shape)}."
            )
        expected_eps_shape = (config.num_layers,)
        if tuple(self.k_norm_eps.shape) != expected_eps_shape:
            raise ValueError(
                "k_norm_eps shape mismatch. "
                f"Expected {expected_eps_shape}, got {tuple(self.k_norm_eps.shape)}."
            )


@dataclass
class DFlashKVCache:
    k_cache: torch.Tensor
    v_cache: torch.Tensor

    @property
    def num_slots(self) -> int:
        return int(self.k_cache.shape[1])

    def clone(self) -> "DFlashKVCache":
        return DFlashKVCache(
            k_cache=self.k_cache.clone(),
            v_cache=self.v_cache.clone(),
        )

    def validate(self, config: DFlashMaterializerConfig) -> None:
        config.validate()
        expected_prefix = (
            config.num_layers,
            self.num_slots,
            config.num_kv_heads,
            config.head_dim,
        )
        if tuple(self.k_cache.shape) != expected_prefix:
            raise ValueError(
                "k_cache shape mismatch. "
                f"Expected {expected_prefix}, got {tuple(self.k_cache.shape)}."
            )
        if tuple(self.v_cache.shape) != expected_prefix:
            raise ValueError(
                "v_cache shape mismatch. "
                f"Expected {expected_prefix}, got {tuple(self.v_cache.shape)}."
            )
        if self.k_cache.dtype != self.v_cache.dtype:
            raise ValueError(
                "k_cache and v_cache must have the same dtype. "
                f"Got {self.k_cache.dtype} and {self.v_cache.dtype}."
            )
        if self.k_cache.device != self.v_cache.device:
            raise ValueError(
                "k_cache and v_cache must be on the same device. "
                f"Got {self.k_cache.device} and {self.v_cache.device}."
            )


@dataclass
class DFlashProjectedKV:
    cache_k: torch.Tensor
    cache_v: torch.Tensor

    def validate(
        self,
        config: DFlashMaterializerConfig,
        *,
        prefix_shape: tuple[int, ...],
    ) -> None:
        config.validate()
        expected_shape = prefix_shape + (config.num_kv_heads, config.head_dim)
        if tuple(self.cache_k.shape) != expected_shape:
            raise ValueError(
                "cache_k shape mismatch. "
                f"Expected {expected_shape}, got {tuple(self.cache_k.shape)}."
            )
        if tuple(self.cache_v.shape) != expected_shape:
            raise ValueError(
                "cache_v shape mismatch. "
                f"Expected {expected_shape}, got {tuple(self.cache_v.shape)}."
            )
        if self.cache_k.dtype != self.cache_v.dtype:
            raise ValueError(
                "cache_k and cache_v must have the same dtype. "
                f"Got {self.cache_k.dtype} and {self.cache_v.dtype}."
            )
        if self.cache_k.device != self.cache_v.device:
            raise ValueError(
                "cache_k and cache_v must be on the same device. "
                f"Got {self.cache_k.device} and {self.cache_v.device}."
            )


def compute_live_row_mask(
    state: DFlashRequestStateTable,
    req_pool_indices: torch.Tensor,
    req_generation: torch.Tensor,
) -> torch.Tensor:
    state.validate()
    req_idx = req_pool_indices.to(dtype=torch.int64)
    req_generation = req_generation.to(
        device=state.generation.device,
        dtype=state.generation.dtype,
    )
    status = state.status_flags.index_select(0, req_idx)
    live = (status & STATUS_ACTIVE) != 0
    live &= (status & GPU_STOP_MASK) == 0
    live &= state.generation.index_select(0, req_idx) == req_generation
    return live
