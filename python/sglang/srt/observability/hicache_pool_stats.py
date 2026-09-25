# SPDX-License-Identifier: Apache-2.0
"""Logical host slot occupancy, including sidecars that share an index space."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import msgspec

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_storage import SidecarPoolSpec
    from sglang.srt.mem_cache.pool_host.group import HostPoolGroup


class HostPoolStats(msgspec.Struct, frozen=True):
    pool: str
    used_slots: int
    total_slots: int
    indices_from_pool: str = ""


def collect_host_pool_stats(
    host_pool_group: HostPoolGroup,
    sidecar_pool_specs: Sequence[SidecarPoolSpec],
) -> list[HostPoolStats]:
    derived_pools = {spec.pool_name for spec in sidecar_pool_specs}
    independent = {}
    for entry in host_pool_group.entries:
        if entry.name in derived_pools or entry.host_pool is None:
            continue
        try:
            available = entry.host_pool.available_size()
        except NotImplementedError:
            # Some host pools provide transfers without an allocator.
            continue
        total = entry.host_pool.logical_size
        independent[entry.name] = HostPoolStats(
            pool=str(entry.name),
            used_slots=max(total - available, 0),
            total_slots=total,
        )

    stats = list(independent.values())
    for spec in sidecar_pool_specs:
        entry = host_pool_group.entry_map.get(spec.pool_name)
        source = independent.get(spec.indices_from_pool)
        if entry is None or entry.host_pool is None or source is None:
            continue
        # Sidecars share logical indices; their own free lists never consume them.
        stats.append(
            HostPoolStats(
                pool=str(spec.pool_name),
                used_slots=source.used_slots,
                total_slots=source.total_slots,
                indices_from_pool=str(spec.indices_from_pool),
            )
        )
    return stats
