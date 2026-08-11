"""Host-memory dedup for replicated-KV HiCache across attention-TP ranks.

MLA/DSA KV is identical on every attn-TP rank, so only the src rank (attn-TP
rank 0) keeps a real host pool; the other ranks run allocator-only "dummy"
pools and receive loaded pages via an NCCL broadcast on the load stream.

GQA generalization: when num_kv_heads < attn_tp_size, QKVParallelLinear
replicates each KV-head shard on ``R = attn_tp_size // num_kv_heads``
consecutive ranks (shard_id = tp_rank // R), so the SAME dedup applies per
replica group: the group's first rank owns the host pool and broadcasts
loads to its R-1 peers over a per-group NCCL subgroup. The replica factor is
stamped on the device pool as ``hicache_dedup_kv_replicas`` at pool build
(kv_cache_configurator) — pools without the stamp never dedup.

Single source of truth for the dedup gating and the broadcast machinery —
every dedup decision elsewhere must derive from these helpers.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.distributed import (
    get_attn_tensor_model_parallel_rank,
    get_attn_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
)
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)


# Backends that never touch the host KV buffer directly, so they tolerate
# the buffer-less dummy pools. RDMA/registered backends (mooncake/eic/simm/
# hf3fs/nixl/aibrix) pin or register the buffer — dedup stays off for them.
_DEDUP_COMPATIBLE_STORAGE = frozenset({None, "", "file"})


def storage_supports_host_dedup(storage_backend: Optional[str]) -> bool:
    """Whether MLA/DSA host-memory dedup can engage with this storage backend."""
    return storage_backend in _DEDUP_COMPATIBLE_STORAGE


def _aligned_host_tokens(
    *,
    device_tokens: int,
    size_per_token: int,
    host_to_device_ratio: float,
    host_size_gb: int,
    page_size: int,
) -> int:
    requested = (
        int(host_size_gb * 1e9 // size_per_token)
        if host_size_gb > 0
        else int(device_tokens * host_to_device_ratio)
    )
    return (requested // page_size + 1) * page_size


def estimate_mla_host_pool_bytes(
    device_pool: MLATokenToKVPool,
    *,
    host_to_device_ratio: float,
    host_size_gb: int,
    page_size: int,
) -> tuple[int, int]:
    """Return ``(bytes, tokens)`` without allocating the MLA host pool."""
    kv_cache_dim = getattr(device_pool, "kv_cache_dim", None)
    if kv_cache_dim is None:
        kv_cache_dim = device_pool.kv_lora_rank + device_pool.qk_rope_head_dim
    size_per_token = (
        kv_cache_dim * device_pool.store_dtype.itemsize * device_pool.layer_num
    )
    tokens = _aligned_host_tokens(
        device_tokens=device_pool.size,
        size_per_token=size_per_token,
        host_to_device_ratio=host_to_device_ratio,
        host_size_gb=host_size_gb,
        page_size=page_size,
    )
    return tokens * size_per_token, tokens


def estimate_target_host_pool_bytes(
    device_pool,
    *,
    host_to_device_ratio: float,
    host_size_gb: int,
    page_size: int,
) -> tuple[int, int]:
    """``(bytes, tokens)`` for one physical copy of the target host pool
    (MLA or GQA/MHA), without allocating it."""
    inner = _unwrap_kv_pool(device_pool)
    if isinstance(inner, MLATokenToKVPool):
        return estimate_mla_host_pool_bytes(
            inner,
            host_to_device_ratio=host_to_device_ratio,
            host_size_gb=host_size_gb,
            page_size=page_size,
        )
    # MHA host pools store K and V (factor 2) for every host layer.
    size_per_token = (
        inner.head_num
        * inner.head_dim
        * inner.store_dtype.itemsize
        * 2
        * inner.layer_num
    )
    tokens = _aligned_host_tokens(
        device_tokens=inner.size,
        size_per_token=size_per_token,
        host_to_device_ratio=host_to_device_ratio,
        host_size_gb=host_size_gb,
        page_size=page_size,
    )
    return tokens * size_per_token, tokens


def estimate_mamba_host_pool_bytes(
    device_pool,
    *,
    host_to_device_ratio: float,
    host_size_gb: int,
) -> tuple[int, int]:
    """Return ``(bytes, tokens)`` for one rank-local Mamba/KDA host pool."""
    conv_bytes = sum(
        math.prod(state.shape[2:]) * state.element_size()
        for state in device_pool.mamba_cache.conv
    )
    temporal = device_pool.mamba_cache.temporal
    temporal_bytes = math.prod(temporal.shape[2:]) * temporal.element_size()
    size_per_token = (conv_bytes + temporal_bytes) * device_pool.num_mamba_layers
    tokens = _aligned_host_tokens(
        device_tokens=device_pool.size,
        size_per_token=size_per_token,
        host_to_device_ratio=host_to_device_ratio,
        host_size_gb=host_size_gb,
        page_size=1,
    )
    return tokens * size_per_token, tokens


def estimate_draft_host_pool_bytes(
    device_pool, *, host_tokens: int, page_size: int
) -> tuple[int, int]:
    """Return ``(bytes, tokens)`` for one rank-local draft L2 pool."""
    if isinstance(device_pool, MLATokenToKVPool):
        kv_cache_dim = getattr(device_pool, "kv_cache_dim", None)
        if kv_cache_dim is None:
            kv_cache_dim = device_pool.kv_lora_rank + device_pool.qk_rope_head_dim
        size_per_token = (
            kv_cache_dim * device_pool.store_dtype.itemsize * device_pool.layer_num
        )
    elif isinstance(device_pool, MHATokenToKVPool):
        size_per_token = (
            2
            * device_pool.head_num
            * device_pool.head_dim
            * device_pool.layer_num
            * device_pool.store_dtype.itemsize
        )
    else:
        raise ValueError(
            "Cannot estimate HiCache draft host memory for "
            f"{type(device_pool).__name__}."
        )
    tokens = (host_tokens // page_size + 1) * page_size
    return tokens * size_per_token, tokens


def enforce_hicache_host_budget(
    *,
    target_bytes: int,
    rank_local_bytes: dict[str, int],
    tp_size: int,
    context: str,
    target_copies: int = 1,
) -> int:
    """Log and validate one deterministic, node-aggregate HiCache plan.

    ``target_copies`` is the number of physical target host pools across the
    attn-TP group: 1 for MLA (fully replicated KV), attn_tp // replica_group
    for GQA replica-group dedup.
    """
    from sglang.srt.environ import envs

    budget_gib = envs.SGLANG_HICACHE_HOST_BUDGET_GIB.get()
    if budget_gib <= 0:
        raise ValueError("SGLANG_HICACHE_HOST_BUDGET_GIB must be positive.")
    aggregate = target_bytes * target_copies + tp_size * sum(
        rank_local_bytes.values()
    )
    parts = ", ".join(
        [
            f"target={target_bytes / (1024**3):.2f} GiB x{target_copies}",
            *[
                f"{name}={num_bytes / (1024**3):.2f} GiB x{tp_size}"
                for name, num_bytes in sorted(rank_local_bytes.items())
            ],
        ]
    )
    logger.info(
        "HiCache aggregate host budget (%s): %s; total=%.2f GiB, cap=%d GiB",
        context,
        parts,
        aggregate / (1024**3),
        budget_gib,
    )
    if aggregate > budget_gib * (1024**3):
        raise ValueError(
            f"HiCache aggregate host plan for {context} requires "
            f"{aggregate / (1024**3):.2f} GiB ({parts}), exceeding "
            f"SGLANG_HICACHE_HOST_BUDGET_GIB={budget_gib}. Reduce "
            "--hicache-ratio/--hicache-size or raise the explicit cap."
        )
    return aggregate


def enforce_dedup_draft_host_budget(
    controller, draft_device_pool, *, page_size: int
) -> int:
    """Revalidate the aggregate plan before allocating a rank-local draft L2."""
    group = controller.mem_pool_host
    anchor = getattr(group, "anchor_entry", None)
    target = group if anchor is None else anchor.host_pool
    target_bytes = target.size * target.size_per_token
    target_tokens = target.size

    mamba_bytes = 0
    mamba_tokens = 0
    entry_map = getattr(group, "entry_map", {})
    try:
        from sglang.srt.mem_cache.hicache_storage import PoolName

        mamba_entry = entry_map.get(PoolName.MAMBA)
    except Exception:
        mamba_entry = None
    if mamba_entry is not None:
        mamba = mamba_entry.host_pool
        mamba_bytes = mamba.size * mamba.size_per_token
        mamba_tokens = mamba.size

    draft_bytes, draft_tokens = estimate_draft_host_pool_bytes(
        draft_device_pool,
        host_tokens=target_tokens,
        page_size=page_size,
    )
    _, tp_size = dedup_rank_and_size()
    target_device_pool = getattr(target, "device_pool", None)
    target_copies = (
        num_target_host_copies(target_device_pool)
        if target_device_pool is not None
        else 1
    )
    # All three allocator implementations preallocate CPU bookkeeping:
    # target/draft use uint8+int64+bool (10 B/slot), Mamba uint8+int64.
    allocator_metadata_bytes = (
        target_tokens * 10 + mamba_tokens * 9 + draft_tokens * 10
    )
    rank_local = {
        "allocator_metadata": allocator_metadata_bytes,
        "draft": draft_bytes,
    }
    if mamba_entry is not None:
        rank_local["mamba"] = mamba_bytes
    return enforce_hicache_host_budget(
        target_bytes=target_bytes,
        rank_local_bytes=rank_local,
        tp_size=tp_size,
        target_copies=target_copies,
        context=(
            f"MLA dedup with draft L2 "
            f"(target_tokens={target_tokens}, draft_tokens={draft_tokens})"
        ),
    )


def dedup_rank_and_size() -> tuple[int, int]:
    """Attn-TP rank/size when DP attention is enabled, model-TP otherwise."""
    if is_dp_attention_enabled():
        return (
            get_attn_tensor_model_parallel_rank(),
            get_attn_tensor_model_parallel_world_size(),
        )
    return (
        get_tensor_model_parallel_rank(),
        get_tensor_model_parallel_world_size(),
    )


def _unwrap_kv_pool(kv_cache):
    """Hybrid linear models wrap the full-attention pool; dedup what it wraps."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    if isinstance(kv_cache, HybridLinearKVPool):
        return kv_cache.full_kv_pool
    return kv_cache


def _mha_scale_sidecar(inner) -> bool:
    """MHA pools whose quantization keeps per-token scale sidecar buffers the
    broadcast would not cover (MXFP8 block scales). Plain fp8_e4m3 KV stores
    scales per layer on the attention modules, not in the pool -- safe."""
    try:
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolMXFP8

        return isinstance(inner, MHATokenToKVPoolMXFP8)
    except ImportError:
        return False


def kv_replica_group_size(kv_cache) -> int:
    """Ranks per identical-KV replica group inside the attn-TP group.

    MLA: the whole attn-TP group (KV fully replicated). MHA/GQA: the
    ``hicache_dedup_kv_replicas`` stamp from pool build (1 = not replicated,
    or an unstamped pool -- never dedup those).
    """
    inner = _unwrap_kv_pool(kv_cache)
    if isinstance(inner, MLATokenToKVPool):
        return dedup_rank_and_size()[1]
    return int(getattr(inner, "hicache_dedup_kv_replicas", 1) or 1)


def dedup_group_rank_and_size(kv_cache) -> tuple[int, int]:
    """(rank within this rank's replica group, group size).

    Replica groups are consecutive rank blocks (shard_id = tp_rank // R), so
    the position within the group is attn_tp_rank % R and the group src is
    the block's first rank.
    """
    rank, size = dedup_rank_and_size()
    group = min(kv_replica_group_size(kv_cache), size)
    if group <= 1:
        return rank, 1
    return rank % group, group


def num_target_host_copies(kv_cache) -> int:
    """Physical copies of the target host pool across the attn-TP group
    (= number of replica groups; 1 for MLA)."""
    _, size = dedup_rank_and_size()
    group = min(kv_replica_group_size(kv_cache), size)
    if group <= 1:
        return size
    return size // group


def host_dedup_eligible(
    kv_cache, storage_backend: Optional[str], enabled: bool = False
) -> bool:
    """Rank-independent gate. CUDA only; FP4/MXFP8 excluded (their per-rank
    scale buffers are not covered by the broadcast)."""
    if not (enabled and is_cuda() and storage_supports_host_dedup(storage_backend)):
        return False
    inner = _unwrap_kv_pool(kv_cache)
    if isinstance(inner, MLATokenToKVPoolFP4):
        return False
    if isinstance(inner, MLATokenToKVPool):
        return True
    return (
        isinstance(inner, MHATokenToKVPool)
        and not _mha_scale_sidecar(inner)
        and kv_replica_group_size(kv_cache) > 1
    )


def require_host_dedup_supported(
    kv_cache, storage_backend: Optional[str], enabled: bool = False
) -> None:
    """Fail closed when the opt-in cannot preserve target-cache semantics."""
    if not enabled:
        return
    inner = _unwrap_kv_pool(kv_cache)
    if not isinstance(inner, (MLATokenToKVPool, MHATokenToKVPool)):
        raise ValueError(
            "--enable-hicache-host-dedup requires an MLA or replicated "
            f"GQA/MHA target KV pool, got {type(inner).__name__}."
        )
    if isinstance(inner, MLATokenToKVPoolFP4):
        raise ValueError(
            "--enable-hicache-host-dedup does not support FP4 MLA KV: "
            "the per-rank scale buffer is not broadcast."
        )
    if isinstance(inner, MHATokenToKVPool):
        if _mha_scale_sidecar(inner):
            raise ValueError(
                "--enable-hicache-host-dedup does not support MXFP8 MHA "
                "KV: the per-token scale sidecar is not broadcast."
            )
        if kv_replica_group_size(kv_cache) <= 1:
            raise ValueError(
                "--enable-hicache-host-dedup on a GQA/MHA pool requires "
                "replicated KV heads (num_kv_heads < attn_tp_size, stamped as "
                "hicache_dedup_kv_replicas at pool build). This pool has no "
                "replication -- every rank holds distinct KV; nothing to dedup."
            )
        _, size = dedup_rank_and_size()
        if size % kv_replica_group_size(kv_cache) != 0:
            raise ValueError(
                "KV replica group size must divide the attention-TP size; got "
                f"group={kv_replica_group_size(kv_cache)}, attn_tp={size}."
            )
    if getattr(inner, "layer_shard_enabled", False):
        raise ValueError(
            "--enable-hicache-host-dedup requires every attention-TP "
            "rank to own every target attention layer."
        )
    if not is_cuda():
        raise ValueError(
            "--enable-hicache-host-dedup currently requires CUDA/NCCL."
        )
    if not storage_supports_host_dedup(storage_backend):
        raise ValueError(
            "--enable-hicache-host-dedup does not support storage backend "
            f"{storage_backend!r}; only L2-only mode or the file backend is "
            "supported."
        )
    _, size = dedup_rank_and_size()
    if size <= 1:
        raise ValueError(
            "--enable-hicache-host-dedup requires attention TP > 1."
        )


def is_host_dedup_dummy_rank(
    kv_cache, storage_backend: Optional[str], enabled: bool = False
) -> bool:
    """Whether this rank must construct an allocator-only (dummy) host pool."""
    require_host_dedup_supported(kv_cache, storage_backend, enabled)
    if not host_dedup_eligible(kv_cache, storage_backend, enabled):
        return False
    group_rank, group_size = dedup_group_rank_and_size(kv_cache)
    return group_size > 1 and group_rank != 0


class HostDedupBroadcaster:
    """Broadcasts host-loaded MLA KV (and DSA indexer) device pages from the
    src rank to its attn-TP peers over a dedicated NCCL group.

    Layers are broadcast one at a time so the controller can release each
    layer to the model as soon as its H2D + broadcast finishes.  The staging
    allocation retains the old all-layer size and is reinterpreted as a larger
    per-layer token chunk.  Consequently, layerwise mode does not multiply the
    NCCL call count by ``layer_num`` for large loads.
    """

    # Tokens (or DSA indexer pages) staged per broadcast chunk.
    CHUNK_TOKENS = 512

    def __init__(
        self,
        device_pool,
        group: torch.distributed.ProcessGroup,
        src_global_rank: int,
    ):
        self.device_pool = device_pool
        inner = _unwrap_kv_pool(device_pool)
        self.group = group
        self.src_global_rank = src_global_rank
        self.is_src = dedup_group_rank_and_size(device_pool)[0] == 0
        self.layer_num = inner.layer_num
        self.device = inner.device
        # Per-layer buffer streams to broadcast: MLA has one packed kv
        # buffer; MHA/GQA keeps K and V separately.
        if isinstance(inner, MLATokenToKVPool):
            self._streams = [("kv", inner.kv_buffer, inner.kv_cache_dim)]
        else:
            k_elem = math.prod(inner.k_buffer[0].shape[1:]) or 1
            v_elem = math.prod(inner.v_buffer[0].shape[1:]) or 1
            self._streams = [
                ("k", inner.k_buffer, k_elem),
                ("v", inner.v_buffer, v_elem),
            ]
        max_elem = max(elem for _, _, elem in self._streams)
        self.kv_staging = torch.empty(
            self.layer_num * self.CHUNK_TOKENS * max_elem,
            dtype=self._streams[0][1][0].dtype,
            device=self.device,
        )
        # DSA keeps a per-page indexer buffer that must be broadcast too.
        self.idx_bufs = None
        self.idx_elem = None
        self.idx_staging = None
        if isinstance(inner, DSATokenToKVPool):
            self.idx_bufs = inner.index_k_with_scale_buffer
            self.idx_elem = math.prod(self.idx_bufs[0].shape[1:]) or 1
            self.idx_staging = torch.empty(
                self.layer_num * self.CHUNK_TOKENS * self.idx_elem,
                dtype=self.idx_bufs[0].dtype,
                device=self.device,
            )

    @classmethod
    def build(
        cls,
        device_pool,
        tp_group: torch.distributed.ProcessGroup,
        attn_tp_group: Optional[torch.distributed.ProcessGroup],
    ) -> HostDedupBroadcaster:
        """Build the NCCL group (a world collective — all dedup participants
        must call in lockstep) and the staging buffers."""
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        base_group = tp_group
        if is_dp_attention_enabled() and attn_tp_group is not None:
            base_group = attn_tp_group
        group_ranks = list(torch.distributed.get_process_group_ranks(base_group))
        world = len(group_ranks)
        replica = min(kv_replica_group_size(device_pool), world)
        if replica >= world:
            subgroup_rank_lists = [group_ranks]
        else:
            # Replica groups are consecutive rank blocks (see module
            # docstring). new_group is a world collective: every rank must
            # create EVERY subgroup, in the same order, keeping only its own.
            assert world % replica == 0, (world, replica)
            subgroup_rank_lists = [
                group_ranks[i : i + replica] for i in range(0, world, replica)
            ]
        my_rank = torch.distributed.get_rank()
        group = None
        my_ranks = None
        for ranks in subgroup_rank_lists:
            candidate = create_custom_parallel_group(
                group_ranks=ranks, backend="nccl"
            )
            if my_rank in ranks:
                group = candidate
                my_ranks = ranks
        assert group is not None and my_ranks is not None, (
            f"rank {my_rank} not in any dedup subgroup of {group_ranks}"
        )
        broadcaster = cls(device_pool, group, src_global_rank=my_ranks[0])
        # NCCL allocates this communicator's device buffers lazily at its
        # FIRST collective — which for this group is the first host-tier
        # loadback, potentially hours into serving on a ~full device. That
        # ~10 MiB cudaCalloc then fails (ncclUnhandledCudaError), poisoning
        # the group and every later loadback. Pay the allocation here, at
        # build time, while boot memory is still free — with a production-
        # sized payload so every size-dependent NCCL path allocates now.
        # All participants reach build() in lockstep (world collective).
        for _, _, elem in broadcaster._streams:
            warm = broadcaster.kv_staging[: cls.CHUNK_TOKENS * elem]
            torch.distributed.broadcast(warm, src=my_ranks[0], group=group)
        if broadcaster.idx_staging is not None:
            torch.distributed.broadcast(
                broadcaster.idx_staging, src=my_ranks[0], group=group
            )
        torch.cuda.synchronize(broadcaster.device)
        return broadcaster

    def prepare_broadcast(
        self, device_indices: torch.Tensor, load_stream
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare reusable KV/indexer indices for one layerwise load."""
        indices = device_indices
        if not indices.is_cuda:
            indices = indices.to(self.device)
        if indices.is_cuda:
            indices.record_stream(load_stream)

        page_idx = None
        if self.idx_bufs is not None:
            page_size = _unwrap_kv_pool(self.device_pool).page_size
            page_idx = (
                torch.unique(torch.div(indices, page_size, rounding_mode="floor"))
                if page_size > 1
                else indices
            )
            if page_idx.is_cuda:
                page_idx.record_stream(load_stream)
        return indices, page_idx

    def broadcast_loaded_layer(
        self,
        layer_id: int,
        prepared: tuple[torch.Tensor, Optional[torch.Tensor]],
        trace=None,
    ) -> None:
        """Broadcast one loaded KV layer and its optional DSA indexer layer."""
        indices, page_idx = prepared
        for name, bufs, elem in self._streams:
            self._bcast_layer(
                bufs,
                self.kv_staging,
                indices,
                elem,
                layer_id,
                trace=trace,
                trace_prefix=name,
            )
        if self.idx_bufs is not None:
            assert page_idx is not None
            self._bcast_layer(
                self.idx_bufs,
                self.idx_staging,
                page_idx,
                self.idx_elem,
                layer_id,
                trace=trace,
                trace_prefix="indexer",
            )

    def _bcast_layer(
        self,
        buf_list,
        staging,
        target,
        elem,
        layer_id: int,
        trace=None,
        trace_prefix: str = "kv",
    ) -> None:
        """Chunked in-place broadcast for one layer.

        ``staging`` is sized for ``layer_num * CHUNK_TOKENS`` rows.  Reusing
        the full allocation for one layer preserves the previous maximum NCCL
        payload size while enabling per-layer completion events.  ``index_select``
        with an output tensor and ``index_copy_`` avoid the temporary tensors
        created by advanced indexing in the original all-layer implementation.
        """
        n = target.shape[0]
        rows_per_chunk = staging.numel() // elem
        assert rows_per_chunk > 0
        layer_buf = buf_list[layer_id]
        row_shape = layer_buf.shape[1:]

        for start in range(0, n, rows_per_chunk):
            cur = min(rows_per_chunk, n - start)
            idx = target[start : start + cur]
            chunk = staging[: cur * elem]
            chunk_rows = chunk.view(cur, *row_shape)
            if self.is_src:
                pack_start = self._trace_event(trace)
                torch.index_select(layer_buf, 0, idx, out=chunk_rows)
                self._finish_trace_phase(
                    trace,
                    f"{trace_prefix}_pack",
                    pack_start,
                    chunk.numel() * chunk.element_size(),
                )
            nccl_start = self._trace_event(trace)
            torch.distributed.broadcast(
                chunk, src=self.src_global_rank, group=self.group
            )
            self._finish_trace_phase(
                trace,
                f"{trace_prefix}_nccl",
                nccl_start,
                chunk.numel() * chunk.element_size(),
            )
            if not self.is_src:
                scatter_start = self._trace_event(trace)
                layer_buf.index_copy_(0, idx, chunk_rows)
                self._finish_trace_phase(
                    trace,
                    f"{trace_prefix}_scatter",
                    scatter_start,
                    chunk.numel() * chunk.element_size(),
                )

    @staticmethod
    def _trace_event(trace):
        if trace is None:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    @staticmethod
    def _finish_trace_phase(trace, name: str, start, num_bytes: int) -> None:
        if trace is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        trace["events"].append((name, start, end, num_bytes))

    def destroy(self) -> None:
        if self.group is None:
            return
        try:
            torch.distributed.destroy_process_group(self.group)
        except Exception:
            pass
        self.group = None


def maybe_build_host_dedup_broadcaster(
    device_pool,
    tp_group: torch.distributed.ProcessGroup,
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
    storage_backend: Optional[str],
    enabled: bool = False,
) -> Optional[HostDedupBroadcaster]:
    """None when dedup does not engage (gate fails or single attn-TP rank)."""
    require_host_dedup_supported(device_pool, storage_backend, enabled)
    if not host_dedup_eligible(device_pool, storage_backend, enabled):
        return None
    if dedup_group_rank_and_size(device_pool)[1] <= 1:
        return None
    return HostDedupBroadcaster.build(device_pool, tp_group, attn_tp_group)


@dataclass
class HostDedupPrebuild:
    """Groups/buffers rendezvoused ahead of the slow host KV allocation."""

    broadcaster: HostDedupBroadcaster
    # None without a storage backend, so a later runtime attach still builds
    # its gloo groups inline.
    prefetch_sync_groups: Optional[List[torch.distributed.ProcessGroup]]


def maybe_prebuild_host_dedup(
    kv_cache,
    tp_group: torch.distributed.ProcessGroup,
    attn_cp_group: Optional[torch.distributed.ProcessGroup],
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
    storage_backend: Optional[str],
    enabled: bool = False,
) -> Optional[HostDedupPrebuild]:
    """Issue the controller's init-time world collectives BEFORE the host KV
    pool is allocated.

    The src rank can spend many minutes pinning host KV while the dummy
    ranks race ahead into create_custom_parallel_group (NCCL bcast group +
    gloo prefetch groups) and trip the 600s NCCL watchdog; prebuilding
    completes the rendezvouses in lockstep first. Returns None when dedup
    does not engage — same gating as the controller, so groups are never
    built on ranks that would ignore them.
    """
    broadcaster = maybe_build_host_dedup_broadcaster(
        kv_cache, tp_group, attn_tp_group, storage_backend, enabled
    )
    if broadcaster is None:
        return None
    rank, size = dedup_rank_and_size()
    group_rank, group_size = dedup_group_rank_and_size(kv_cache)
    logger.info(
        "HiCache host dedup active: attn_tp_rank=%d/%d, "
        "replica_group_rank=%d/%d, role=%s, target_host_pool=%s",
        rank,
        size,
        group_rank,
        group_size,
        "owner" if broadcaster.is_src else "receiver",
        "physical" if broadcaster.is_src else "allocator-only",
    )
    prefetch_sync_groups = None
    if storage_backend is not None:
        prefetch_sync_groups = _prebuild_prefetch_sync_groups(
            tp_group, attn_cp_group, attn_tp_group
        )
    return HostDedupPrebuild(broadcaster, prefetch_sync_groups)


def _prebuild_prefetch_sync_groups(
    tp_group: torch.distributed.ProcessGroup,
    attn_cp_group: Optional[torch.distributed.ProcessGroup],
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
) -> List[torch.distributed.ProcessGroup]:
    """Same construction as HiCacheController._create_prefetch_sync_groups."""
    from sglang.srt.distributed.parallel_state import create_custom_parallel_group

    groups: List[torch.distributed.ProcessGroup] = []
    seen_rank_sets = set()
    if attn_cp_group is not None or attn_tp_group is not None:
        base_groups = [attn_cp_group, attn_tp_group]
    else:
        base_groups = [tp_group]
    for group in base_groups:
        if group is None or torch.distributed.get_world_size(group=group) == 1:
            continue
        ranks = tuple(torch.distributed.get_process_group_ranks(group))
        if ranks in seen_rank_sets:
            continue
        seen_rank_sets.add(ranks)
        groups.append(
            create_custom_parallel_group(group_ranks=list(ranks), backend="gloo")
        )
    return groups
