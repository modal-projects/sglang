"""Bounded prefix history for measuring recomputation after local KV eviction.

The digests identify logical pages, independently of tree splits. They contain
no KV data and cannot restore an evicted prefix. A hit measures only a recent
local capacity eviction; TTL and capacity truncation make it a lower bound.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Iterable, Protocol, runtime_checkable

import msgspec

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode


@runtime_checkable
class GhostMetrics(Protocol):
    def increment_kv_inserted_tokens(self, num_tokens: int) -> None: ...

    def increment_kv_ghost_list_recorded(self, num_pages: int) -> None: ...

    def increment_kv_ghost_list_discarded(
        self, reason: str, num_pages: int
    ) -> None: ...

    def observe_kv_reinsert_after_evict(
        self, delay_seconds: float, num_tokens: int
    ) -> None: ...


@runtime_checkable
class InitializableGhostMetrics(Protocol):
    def initialize_kv_ghost_metrics(self) -> None: ...


def _node_digests(node: UnifiedTreeNode, page_size: int) -> list[bytes]:
    pending = []
    current = node
    while current.parent is not None and current.ghost_digests is None:
        pending.append(current)
        current = current.parent
    previous = current.ghost_digests
    for current in reversed(pending):
        key = current.key
        if previous:
            prior = previous[-1]
        else:
            namespace = hashlib.blake2b(digest_size=16)
            namespace.update(bytes([key.is_bigram]))
            for part in (key.extra_key, key.cache_salt):
                if part is None:
                    namespace.update(b"\x00")
                else:
                    encoded = part.encode("utf-8")
                    namespace.update(b"\x01" + len(encoded).to_bytes(8, "little"))
                    namespace.update(encoded)
            prior = namespace.digest()
        digests = []
        for start in range(0, len(key) // page_size * page_size, page_size):
            # Use the lookup identity, including any content-aware key fields.
            page_key = key.child_key_at(start, page_size)
            prior = hashlib.blake2b(
                prior + msgspec.msgpack.encode(page_key), digest_size=16
            ).digest()
            digests.append(prior)
        current.ghost_digests = previous = digests
        current.ghost_end = current.parent.ghost_end + len(key)
    return node.ghost_digests


class _GhostEntry(msgspec.Struct):
    serial: int
    evicted_at: float
    anchor_id: int
    page_end: int
    cutoffs: dict[int, int] | None = None


class KVGhostTracker:
    def __init__(
        self,
        *,
        capacity: int,
        ttl_seconds: float,
        page_size: int,
        metrics: GhostMetrics,
    ):
        if (
            capacity <= 0
            or page_size <= 0
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
        ):
            raise ValueError("Ghost capacity and TTL must be positive and finite")
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self.page_size = page_size
        self.metrics = metrics
        self._entries: OrderedDict[bytes, _GhostEntry] = OrderedDict()
        self._by_serial: dict[int, bytes] = {}
        self._by_anchor: dict[int, set[int]] = {}
        self._by_ref: dict[int, dict[int, int]] = {}
        self._cutoff_count = 0
        self._next_serial = 0
        self._deleted_slots = 0

    def reset(self) -> None:
        self._discard("flush", len(self._entries))
        self._entries.clear()
        self._by_serial.clear()
        self._by_anchor.clear()
        self._by_ref.clear()
        self._cutoff_count = 0
        self._deleted_slots = 0

    def _discard(self, reason: str, count: int) -> None:
        if count:
            self.metrics.increment_kv_ghost_list_discarded(reason, count)

    def _remove(self, digest: bytes) -> _GhostEntry | None:
        entry = self._entries.pop(digest, None)
        if entry is None:
            return None
        del self._by_serial[entry.serial]
        anchors = self._by_anchor[entry.anchor_id]
        anchors.remove(entry.serial)
        if not anchors:
            del self._by_anchor[entry.anchor_id]
        self._deleted_slots += 4 + 3 * len(entry.cutoffs or ())
        if entry.cutoffs:
            for ref_id in entry.cutoffs:
                refs = self._by_ref[ref_id]
                del refs[entry.serial]
                self._cutoff_count -= 1
                if not refs:
                    del self._by_ref[ref_id]
        self._maybe_compact()
        return entry

    def _maybe_compact(self) -> None:
        # Python buckets retain deleted slots; live-link limits alone do not
        # bound their backing storage after repeated partial removals.
        if self._deleted_slots < self.capacity:
            return
        self._entries = OrderedDict(self._entries)
        self._by_serial = {}
        self._by_anchor = {}
        self._by_ref = {}
        for digest, entry in self._entries.items():
            self._by_serial[entry.serial] = digest
            self._by_anchor.setdefault(entry.anchor_id, set()).add(entry.serial)
            entry.cutoffs = (
                {ref_id: cutoff for ref_id, cutoff in entry.cutoffs.items()}
                if entry.cutoffs
                else None
            )
            if entry.cutoffs:
                for ref_id, cutoff in entry.cutoffs.items():
                    self._by_ref.setdefault(ref_id, {})[entry.serial] = cutoff
        self._deleted_slots = 0

    def _expire(self, now: float) -> None:
        expired = 0
        while self._entries:
            digest, entry = next(iter(self._entries.items()))
            if now - entry.evicted_at < self.ttl_seconds:
                break
            self._remove(digest)
            expired += 1
        self._discard("ttl", expired)

    def on_dropped(self, node: UnifiedTreeNode) -> None:
        now = time.monotonic()
        self._expire(now)
        recorded = discarded = 0
        digests = _node_digests(node, self.page_size)
        start = node.ghost_end - len(node.key)
        for index, digest in enumerate(digests):
            replaced = self._remove(digest)
            recorded += replaced is None
            self._next_serial += 1
            entry = _GhostEntry(
                serial=self._next_serial,
                evicted_at=now,
                anchor_id=node.id,
                page_end=start + (index + 1) * self.page_size,
            )
            self._entries[digest] = entry
            self._by_serial[entry.serial] = digest
            self._by_anchor.setdefault(node.id, set()).add(entry.serial)
            if len(self._entries) > self.capacity:
                self._remove(next(iter(self._entries)))
                discarded += 1
        if recorded:
            self.metrics.increment_kv_ghost_list_recorded(recorded)
        self._discard("capacity", discarded)

    def on_inserted(self, node: UnifiedTreeNode, *, restored: bool = False) -> None:
        now = time.monotonic()
        self._expire(now)
        digests = _node_digests(node, self.page_size)
        if not restored:
            self.metrics.increment_kv_inserted_tokens(len(digests) * self.page_size)
        matched = 0
        runs: list[tuple[float, int]] = []
        previous = None
        for digest in digests:
            entry = self._remove(digest)
            stamp = entry.evicted_at if entry is not None else None
            if stamp is not None:
                matched += 1
                if stamp == previous:
                    runs[-1] = (stamp, runs[-1][1] + 1)
                else:
                    runs.append((stamp, 1))
            previous = stamp
        if not restored:
            for stamp, pages in runs:
                self.metrics.observe_kv_reinsert_after_evict(
                    now - stamp, pages * self.page_size
                )
        self._discard("restored" if restored else "reinsert", matched)

    def on_split(self, child_id: int, prefix_id: int, split_end: int) -> None:
        for serial in tuple(self._by_anchor.get(child_id, ())):
            entry = self._entries[self._by_serial[serial]]
            if entry.page_end <= split_end:
                self._move_anchor(entry, prefix_id)

    def _move_anchor(self, entry: _GhostEntry, anchor_id: int) -> None:
        old = self._by_anchor[entry.anchor_id]
        old.remove(entry.serial)
        if not old:
            del self._by_anchor[entry.anchor_id]
        entry.anchor_id = anchor_id
        self._by_anchor.setdefault(anchor_id, set()).add(entry.serial)
        self._deleted_slots += 2
        self._maybe_compact()

    def on_deleted(
        self,
        node_id: int,
        parent_id: int,
        references: Iterable[tuple[int, int]],
    ) -> None:
        # Preserve only receipt links backed by retained ghosts, not dead nodes.
        serials = self._by_anchor.get(node_id)
        if not serials:
            return
        discarded = 0
        for ref_id, span_end in references:
            for serial in tuple(self._by_anchor.get(node_id, ())):
                entry = self._entries[self._by_serial[serial]]
                cutoff = min(entry.page_end, span_end)
                prior = entry.cutoffs.get(ref_id) if entry.cutoffs else None
                if prior is None and self._cutoff_count >= self.capacity:
                    self._remove(self._by_serial[serial])
                    discarded += 1
                    continue
                if entry.cutoffs is None:
                    entry.cutoffs = {}
                if prior is None:
                    self._cutoff_count += 1
                cutoff = max(cutoff, prior or 0)
                entry.cutoffs[ref_id] = cutoff
                self._by_ref.setdefault(ref_id, {})[serial] = cutoff
            if node_id not in self._by_anchor:
                break
        for serial in tuple(self._by_anchor.get(node_id, ())):
            self._move_anchor(self._entries[self._by_serial[serial]], parent_id)
        self._discard("capacity", discarded)

    def invalidate_ref(self, ref_id: int, start: int) -> None:
        discarded = 0
        for serial, cutoff in tuple(self._by_ref.get(ref_id, {}).items()):
            if start < cutoff:
                self._remove(self._by_serial[serial])
                discarded += 1
        self._discard("invalidate", discarded)

    def retire_anchor(self, node_id: int) -> None:
        serials = tuple(self._by_anchor.get(node_id, ()))
        for serial in serials:
            self._remove(self._by_serial[serial])
        self._discard("invalidate", len(serials))

    def release_ref(self, ref_id: int) -> None:
        refs = self._by_ref.pop(ref_id, None)
        if refs is None:
            return
        for serial in refs:
            entry = self._entries[self._by_serial[serial]]
            del entry.cutoffs[ref_id]
            if not entry.cutoffs:
                entry.cutoffs = None
        self._cutoff_count -= len(refs)
        self._deleted_slots += 1 + 2 * len(refs)
        self._maybe_compact()
