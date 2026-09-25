"""Live receipt metadata, bounded by active receipts and their captured pages.

For R receipts with aligned ends E_r and page size P, there are at most
sum(E_r / P) fragment records and the same number of reverse-index entries.
Splits partition records; deletion prunes them; release removes the receipt.
Neither nodes nor historical ancestry are retained here.
"""

from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    NodeId,
    PrefixRef,
)


class PrefixRefRegistry:
    def __init__(self):
        self._owner = object()
        self._next_id = 0
        self._refs: dict[int, dict[NodeId, tuple[int, int]]] = {}
        self._by_node: dict[NodeId, set[int]] = {}

    def clear(self):
        self._refs.clear()
        self._by_node.clear()

    def capture(self, spans: dict[NodeId, tuple[int, int]]) -> PrefixRef:
        self._next_id += 1
        ref_id = self._next_id
        self._refs[ref_id] = spans
        for node_id in spans:
            self._by_node.setdefault(node_id, set()).add(ref_id)
        return PrefixRef(self._owner, ref_id)

    def _validate_owner(self, ref: PrefixRef):
        if ref._owner is not self._owner:
            raise ValueError("prefix receipt belongs to another tree core")

    def is_active(self, ref: PrefixRef) -> bool:
        self._validate_owner(ref)
        return ref._id in self._refs

    def overlapping(self, ref: PrefixRef, start: int):
        self._validate_owner(ref)
        return sorted(
            (span_start, node_id)
            for node_id, (span_start, end) in self._refs.get(ref._id, {}).items()
            if end > start
        )

    def release(self, ref: PrefixRef):
        self._validate_owner(ref)
        for node_id in self._refs.pop(ref._id, {}):
            refs = self._by_node[node_id]
            refs.remove(ref._id)
            if not refs:
                del self._by_node[node_id]

    def split(self, child_id: NodeId, prefix_id: NodeId, split_len: int):
        for ref_id in tuple(self._by_node.get(child_id, ())):
            spans = self._refs[ref_id]
            start, end = spans[child_id]
            split = start + split_len
            spans[prefix_id] = (start, min(end, split))
            self._by_node.setdefault(prefix_id, set()).add(ref_id)
            if end > split:
                spans[child_id] = (split, end)
            else:
                del spans[child_id]
                self._by_node[child_id].remove(ref_id)
        if child_id in self._by_node and not self._by_node[child_id]:
            del self._by_node[child_id]

    def references(self, node_id: NodeId):
        for ref_id in self._by_node.get(node_id, ()):
            yield ref_id, self._refs[ref_id][node_id][1]

    def remove(self, node_id: NodeId):
        for ref_id in self._by_node.pop(node_id, ()):
            del self._refs[ref_id][node_id]

    def counts(self) -> tuple[int, int]:
        assert all(self._by_node.values())
        assert {
            (node_id, ref_id)
            for ref_id, spans in self._refs.items()
            for node_id in spans
        } == {
            (node_id, ref_id)
            for node_id, refs in self._by_node.items()
            for ref_id in refs
        }
        return len(self._refs), sum(map(len, self._refs.values()))
