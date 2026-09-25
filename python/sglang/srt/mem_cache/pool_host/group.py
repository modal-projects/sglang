from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer


@dataclass
class PoolEntry:
    name: PoolName
    host_pool: Any
    device_pool: Any
    layer_mapper: Callable[[int], int | None]
    is_primary_index_anchor: bool = False
    # Reclaim callbacks receive the absolute allocation size n. The host
    # callback evicts n slots; the device callback makes alloc(n) feasible.
    host_evict_fn: Callable[[int], Any] | None = None
    device_evict_fn: Callable[[int], Any] | None = None
    device_alloc_fn: Callable[[int], Any] | None = None
    device_free_fn: Callable[[Any], Any] | None = None
    packed_draft_device_pools: tuple[Any, ...] = ()


class HostPoolGroup:
    """Allocation facade for an anchor host pool and its side pools."""

    def __init__(self, entries: list[PoolEntry]):
        if not entries:
            raise ValueError("HostPoolGroup requires at least one pool entry.")
        if len({entry.name for entry in entries}) != len(entries):
            raise ValueError("HostPoolGroup pool names must be unique.")

        anchors = [entry for entry in entries if entry.is_primary_index_anchor]
        if len(anchors) > 1:
            raise ValueError("HostPoolGroup requires at most one anchor pool.")

        self.entries = list(entries)
        self.entry_map = {entry.name: entry for entry in entries}
        self.anchor_entry = anchors[0] if anchors else entries[0]

        self.layout = self.anchor_entry.host_pool.layout
        self.page_size = self.anchor_entry.host_pool.page_size
        self.device = self.anchor_entry.host_pool.device
        self.size = self.anchor_entry.host_pool.size
        self.logical_size = self.anchor_entry.host_pool.logical_size
        self._refresh_transfer_capabilities()

    def _refresh_transfer_capabilities(self) -> None:
        child_write_back_jit = [
            entry.host_pool.can_use_write_back_jit for entry in self.entries
        ]
        self.can_use_write_back_jit = all(child_write_back_jit)
        self.supports_per_pool_backup_indices = any(child_write_back_jit)

    def add_entry(self, entry: PoolEntry) -> None:
        if entry.name in self.entry_map:
            raise ValueError(f"Host pool {entry.name} is already registered.")
        if entry.is_primary_index_anchor:
            raise ValueError("Cannot replace the anchor of an existing HostPoolGroup.")
        self.entries.append(entry)
        self.entry_map[entry.name] = entry
        self._refresh_transfer_capabilities()

    def get_entry(self, name: PoolName | None = None) -> PoolEntry:
        return self.anchor_entry if name is None else self.entry_map[name]

    def get_pool(self, name: PoolName):
        return self.get_entry(name).host_pool

    def alloc(
        self,
        need_size: int,
        *,
        pool: PoolName | None = None,
        reclaim: Callable[[int], Any] | None = None,
    ) -> torch.Tensor | None:
        """Allocate from one pool, optionally reclaiming once before retrying."""
        host_pool = self.get_entry(pool).host_pool
        indices = host_pool.alloc(need_size)
        if indices is None and reclaim is not None:
            reclaim(need_size)
            indices = host_pool.alloc(need_size)
        return indices

    def free(self, indices: torch.Tensor, *, pool: PoolName | None = None) -> int:
        return self.get_entry(pool).host_pool.free(indices)

    def resolve_host_transfers(
        self,
        transfers: list[PoolTransfer] | None,
        *,
        primary_device_indices: torch.Tensor | None = None,
        primary_host_indices: torch.Tensor | None = None,
    ) -> list[PoolTransfer] | None:
        """Allocate unresolved side-pool host indices atomically.

        On failure, every allocation made by this call is released and the
        corresponding transfer is restored to its unresolved state.
        """
        if not transfers:
            return None

        allocated: list[tuple[PoolTransfer, torch.Tensor]] = []
        derived_transfers: list[PoolTransfer] = []

        def rollback() -> None:
            for transfer, indices in allocated:
                self.free(indices, pool=transfer.name)
                transfer.host_indices = None

        for transfer in transfers:
            if transfer.indices_from_pool is not None:
                derived_transfers.append(transfer)
                continue
            if transfer.host_indices is not None or transfer.device_indices is None:
                continue
            entry = self.entry_map.get(transfer.name)
            if entry is None:
                continue
            indices = self.alloc(
                len(transfer.device_indices),
                pool=transfer.name,
                reclaim=entry.host_evict_fn,
            )
            if indices is None:
                rollback()
                return None
            transfer.host_indices = indices
            allocated.append((transfer, indices))

        for transfer in derived_transfers:
            if transfer.indices_from_pool == self.anchor_entry.name:
                transfer.host_indices = primary_host_indices
                transfer.device_indices = primary_device_indices
                continue

            source = next(
                (
                    candidate
                    for candidate in transfers
                    if candidate.indices_from_pool is None
                    and candidate.name == transfer.indices_from_pool
                ),
                None,
            )
            if source is None:
                rollback()
                return None
            transfer.host_indices = source.host_indices
            transfer.device_indices = source.device_indices
        return transfers

    def release_transfers(self, transfers: list[PoolTransfer] | None) -> int:
        """Release independently allocated side-pool indices.

        Derived transfers share another pool's indices and are deliberately
        skipped so each allocation is released exactly once.
        """
        released = 0
        for transfer in transfers or []:
            if transfer.indices_from_pool is not None or transfer.host_indices is None:
                continue
            released += self.free(transfer.host_indices, pool=transfer.name)
        return released

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id: int,
        io_backend: str,
        pool_transfers: list[PoolTransfer] | None = None,
    ) -> None:
        # Group-level duck-typed transfer surface. The L2 transfer engine
        # currently only drives per-entry host pools (L2Transfer.host_pool);
        # these four methods are exercised by tests and reserved for engine
        # paths that move several pools of one op together.
        # 1. Anchor (KV) transfer
        anchor = self.anchor_entry
        local_layer_id = anchor.layer_mapper(layer_id)
        if local_layer_id is not None and host_indices.numel() > 0:
            anchor.host_pool.load_to_device_per_layer(
                anchor.device_pool,
                host_indices,
                device_indices,
                local_layer_id,
                io_backend,
            )

        # 2. Extra pool transfers
        self.load_extra_to_device_per_layer(
            layer_id, io_backend, pool_transfers=pool_transfers
        )

    def load_extra_to_device_per_layer(
        self,
        layer_id: int,
        io_backend: str,
        pool_transfers: list[PoolTransfer] | None = None,
        pool_names: set[PoolName] | None = None,
    ) -> None:
        """Load selected non-anchor pools for one global transfer layer."""
        for transfer in pool_transfers or []:
            if pool_names is not None and transfer.name not in pool_names:
                continue
            entry = self.entry_map.get(transfer.name)
            if entry is None or transfer.host_indices is None:
                continue
            local_layer_id = entry.layer_mapper(layer_id)
            if local_layer_id is None:
                continue
            entry.host_pool.load_to_device_per_layer(
                entry.device_pool,
                transfer.host_indices,
                transfer.device_indices,
                local_layer_id,
                io_backend,
            )

    def backup_from_device_all_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend: str,
        pool_transfers: list[PoolTransfer] | None = None,
    ) -> None:
        # 1. Anchor (KV) backup
        self.anchor_entry.host_pool.backup_from_device_all_layer(
            self.anchor_entry.device_pool,
            host_indices,
            device_indices,
            io_backend,
        )
        # 2. Extra pool backup
        self.backup_extra_from_device_all_layer(
            io_backend, pool_transfers=pool_transfers
        )

    def backup_extra_from_device_all_layer(
        self,
        io_backend: str,
        pool_transfers: list[PoolTransfer] | None = None,
        pool_names: set[PoolName] | None = None,
    ) -> None:
        """Back up selected non-anchor pools without touching the anchor."""
        for transfer in pool_transfers or []:
            if pool_names is not None and transfer.name not in pool_names:
                continue
            entry = self.entry_map.get(transfer.name)
            if entry is None or transfer.host_indices is None:
                continue
            entry.host_pool.backup_from_device_all_layer(
                entry.device_pool,
                transfer.host_indices,
                transfer.device_indices,
                io_backend,
            )

    @property
    def kv_buffer(self):
        return self.anchor_entry.host_pool.kv_buffer

    @property
    def v_buffer(self):
        return getattr(self.anchor_entry.host_pool, "v_buffer", None)

    @property
    def index_k_buffer(self):
        return getattr(self.anchor_entry.host_pool, "index_k_buffer", None)

    @property
    def index_k_scale_buffer(self):
        # Delegate to the anchor pool so NpuMemcacheStore sees the same
        # buffer set as get_page_buffer_meta (which also delegates), keeping
        # the per-page component-key count consistent (k, v, index_k, scale).
        return getattr(self.anchor_entry.host_pool, "index_k_scale_buffer", None)

    @property
    def dsa_kv_cache_store_fp8(self):
        # Delegate so the L3 store skips the dead v component exactly when
        # get_page_buffer_meta (which also delegates) skips it.
        return getattr(self.anchor_entry.host_pool, "dsa_kv_cache_store_fp8", False)

    @property
    def size_per_token(self):
        return self.anchor_entry.host_pool.size_per_token

    def clear(self) -> None:
        for entry in self.entries:
            entry.host_pool.clear()

    def destroy(self) -> None:
        for entry in self.entries:
            entry.host_pool.destroy()

    def available_size(self, pool: PoolName | None = None):
        return self.get_entry(pool).host_pool.available_size()
