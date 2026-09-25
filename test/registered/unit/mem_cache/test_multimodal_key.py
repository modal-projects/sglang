"""Cache keys retain media identity at every prefix and page boundary."""

import hashlib
import unittest
from array import array

import msgspec
import torch

from sglang.srt.disaggregation.kv_events import BlockStored
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.multimodal_key import MultimodalKeySpan, slice_mm_spans
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.utils import get_hash_str
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _identity(content):
    return "sha256:" + hashlib.sha256(content.encode()).hexdigest()


def _key(last_media="second", *, placeholder=1000001, bigram=False):
    return RadixKey(
        array("q", [1, 2, placeholder, placeholder, 3, placeholder, 4]),
        is_bigram=bigram,
        mm_spans=(
            MultimodalKeySpan(2, 4, _identity("first")),
            MultimodalKeySpan(5, 6, _identity(last_media)),
        ),
    )


class TestMultimodalKey(unittest.TestCase):
    def test_later_media_change_preserves_the_earlier_prefix(self):
        left, right = _key(), _key("third")
        self.assertEqual(left.match(right), 5)
        self.assertEqual(left.match(right, page_size=2), 4)
        self.assertEqual(left.child_key(page_size=4), right.child_key(page_size=4))
        self.assertNotEqual(left.child_key_at(5), right.child_key_at(5))

    def test_routing_placeholders_do_not_change_content_equality(self):
        left, right = _key(), _key(placeholder=1000002)
        self.assertEqual(left.match(right), len(left))
        self.assertEqual(left.child_key(page_size=6), right.child_key(page_size=6))
        self.assertEqual(
            get_hash_str(left, page_size=1), get_hash_str(right, page_size=1)
        )

    def test_slices_preserve_within_media_positions_and_empty_ranges(self):
        key = _key()
        sliced = key[3:6]
        self.assertEqual(
            sliced.mm_spans[0], MultimodalKeySpan(0, 1, _identity("first"), 1)
        )
        self.assertEqual(sliced.match_at(key, 3), 3)
        self.assertEqual(key[3:3].mm_spans, ())
        self.assertEqual(slice_mm_spans(key.mm_spans, 3, 3), ())
        self.assertNotEqual(key[2:3].child_key(), key[3:4].child_key())

    def test_one_token_span_hashes_all_identity_bytes(self):
        first, second = _key(), _key("third")
        first_hashes = get_hash_str(first, page_size=1)
        second_hashes = get_hash_str(second, page_size=1)
        self.assertEqual(first_hashes[:5], second_hashes[:5])
        self.assertNotEqual(first_hashes[5:], second_hashes[5:])
        self.assertEqual(
            get_hash_str(first[5:], first_hashes[4], page_size=1), first_hashes[5:]
        )

    def test_bigram_boundaries_include_both_token_identities(self):
        first, second = _key(bigram=True), _key("third", bigram=True)
        self.assertEqual(first.match(second), 4)
        self.assertEqual(first[1:4].match_at(first, 1), 3)
        self.assertEqual(first[3:3].mm_spans, ())
        hashes = get_hash_str(first, page_size=2)
        self.assertEqual(get_hash_str(first[2:4], hashes[0]), hashes[1])
        other_routing = _key(placeholder=1000002, bigram=True)
        self.assertEqual(first.match(other_routing), len(first))
        self.assertEqual(hashes, get_hash_str(other_routing, page_size=2))

    def test_text_only_storage_bytes_are_unchanged(self):
        tokens = array("q", [1, 2, 3, 4])
        key = RadixKey(tokens)
        expected = hashlib.sha256(
            b"".join(value.to_bytes(4, "little") for value in tokens)
        ).hexdigest()
        self.assertEqual(get_hash_str(key), expected)
        self.assertEqual(key.child_key(page_size=2), (1, 2))

    def test_media_child_edges_have_a_distinct_structural_tag(self):
        edge = _key().child_key_at(2)
        self.assertEqual(edge[0], "mm")
        self.assertEqual(len(edge), 3)

    def test_capped_prefix_omits_future_media(self):
        key = _key()
        key.limit = 2
        plain = RadixKey(array("q", [1, 2]))
        self.assertEqual(key.match(plain), 2)
        self.assertEqual(key.child_key(page_size=2), plain.child_key(page_size=2))
        self.assertEqual(get_hash_str(key), get_hash_str(plain))

    def test_event_digests_survive_node_splits_and_legacy_decoding(self):
        class LegacyStored(msgspec.Struct, tag="BlockStored"):
            block_hashes: list[int]
            parent_block_hash: int | None
            token_ids: list[int]
            block_size: int
            lora_id: int | None

        cache = RadixCache.create_simulated(page_size=1, enable_kv_cache_events=True)
        cache.kv_events.take()
        key = _key()
        cache.insert(InsertParams(key=key, value=torch.arange(len(key))))
        stored = cache.kv_events.take()[0]
        self.assertIsInstance(stored, BlockStored)
        self.assertEqual(stored.block_hashes_sha256, get_hash_str(key, page_size=1))
        legacy = msgspec.msgpack.decode(
            msgspec.msgpack.encode(stored), type=LegacyStored
        )
        self.assertEqual(legacy.block_hashes, stored.block_hashes)
        self.assertEqual(legacy.token_ids, list(key.token_ids))

        match = cache.match_prefix(MatchPrefixParams(key=key[:4]))
        cache.kv_events.record_remove(match.last_device_node)
        removed = cache.kv_events.take()[0]
        self.assertEqual(removed.block_hashes_sha256, stored.block_hashes_sha256[:4])

    def test_legacy_router_hint_does_not_authorize_a_different_media_prefix(self):
        cache = RadixCache.create_simulated(page_size=1)
        cache.insert(InsertParams(key=_key(), value=torch.arange(7)))
        match = cache.match_prefix(MatchPrefixParams(key=_key("third")))
        self.assertEqual(match.device_indices.tolist(), list(range(5)))


if __name__ == "__main__":
    unittest.main()
