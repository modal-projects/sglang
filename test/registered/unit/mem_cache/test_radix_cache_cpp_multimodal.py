"""Native radix prefixes retain media identity through splits and request caching."""

import hashlib
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.multimodal_key import MultimodalKeySpan
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _identity(name):
    return "sha256:" + hashlib.sha256(name.encode()).hexdigest()


def _matched(tree, tokens, spans=()):
    chunks, _, _, _ = tree.match_prefix(tokens, spans)
    return torch.cat(chunks).tolist() if chunks else []


class TestRadixCacheCppMultimodal(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.mem_cache.cpp_radix_tree.radix_tree import RadixTreeCpp
        from sglang.srt.mem_cache.radix_cache_cpp import RadixCacheCpp

        cls.tree_type = RadixTreeCpp
        cls.cache_type = RadixCacheCpp

    def _tree(self, page_size):
        return self.tree_type(False, None, page_size, 1)

    def test_later_media_preserves_the_earlier_prefix(self):
        """A later image branches at its page without invalidating earlier media."""
        tokens = [10, 11, 90, 90, 20, 21, 91, 91, 30, 31, 32, 33]
        alternate = [10, 11, 80, 80, 20, 21, 81, 81, 30, 31, 32, 33]
        first, second, replacement = map(_identity, ("image-a", "image-b", "image-c"))
        original_spans = [(2, 4, first, 0), (6, 8, second, 0)]
        next_spans = [(2, 4, first, 0), (6, 8, replacement, 0)]
        for page_size in (1, 2, 4):
            with self.subTest(page_size=page_size):
                tree = self._tree(page_size)
                tree.writing_through(tokens, torch.arange(12), original_spans)
                shared = 6 // page_size * page_size
                self.assertEqual(
                    _matched(tree, alternate, next_spans), list(range(shared))
                )
                _, reused = tree.writing_through(
                    alternate, torch.arange(100, 112), next_spans
                )
                self.assertEqual(reused, shared)
                self.assertEqual(
                    _matched(tree, tokens, original_spans), list(range(12))
                )
                self.assertEqual(
                    _matched(tree, alternate, next_spans),
                    list(range(shared)) + list(range(100 + shared, 112)),
                )
                tree.debug_print()

    def test_same_media_ignores_routing_tokens_and_span_partition(self):
        """Equivalent media positions match even when their spans were sliced."""
        identity = _identity("wide-image")
        tokens = [10, 90, 90, 90, 90, 90, 20, 21]
        alternate = [10, 80, 81, 82, 83, 84, 20, 21]
        spans = [(1, 6, identity, 7)]
        partitioned = [(1, 3, identity, 7), (3, 6, identity, 9)]
        for page_size in (1, 2, 4):
            with self.subTest(page_size=page_size):
                tree = self._tree(page_size)
                tree.writing_through(tokens, torch.arange(8), spans)
                self.assertEqual(_matched(tree, alternate, partitioned), list(range(8)))
                _, reused = tree.writing_through(
                    alternate, torch.arange(100, 108), partitioned
                )
                self.assertEqual(reused, 8)
                self.assertEqual(tree.total_size(), 8)
                # A text token at a media position belongs to a different branch.
                shared = 1 // page_size * page_size
                self.assertEqual(_matched(tree, tokens), list(range(shared)))
                tree.writing_through(tokens, torch.arange(200, 208))
                self.assertEqual(
                    _matched(tree, tokens),
                    list(range(shared)) + list(range(200 + shared, 208)),
                )
                self.assertEqual(_matched(tree, alternate, partitioned), list(range(8)))

    def test_split_inside_media_preserves_item_position(self):
        """Splitting an edge cannot restart its suffix at media offset zero."""
        identity = _identity("image-with-several-pages")
        tokens = [10, 11, 90, 90, 90, 90, 20, 21]
        spans = [(2, 6, identity, 5)]
        tree = self._tree(2)
        tree.writing_through(tokens, torch.arange(8), spans)
        self.assertEqual(
            _matched(tree, tokens[:4], [(2, 4, identity, 5)]), list(range(4))
        )
        alternate = [10, 11, 80, 80, 80, 80, 20, 21]
        self.assertEqual(_matched(tree, alternate, spans), list(range(8)))
        self.assertEqual(_matched(tree, alternate, [(2, 6, identity, 6)]), [0, 1])
        chunks, _, node, _ = tree.match_prefix(alternate, spans)
        tree.lock_ref(node, True)
        self.assertEqual(tree.protected_size(), 8)
        self.assertEqual(tree.evict(8), [])
        tree.lock_ref(node, False)
        self.assertEqual(sum(len(part) for part in tree.evict(8)), 8)
        self.assertEqual(tree.total_size(), 0)
        tree.debug_print()

    def test_short_media_and_unaligned_tail(self):
        """A one-token image at a page boundary remains part of that page's key."""
        identity = _identity("single-token-image")
        other = _identity("another-single-token-image")
        tokens = [10, 11, 12, 90, 20, 21, 22, 23, 24]
        tree = self._tree(4)
        tree.writing_through(tokens, torch.arange(9), [(3, 4, identity, 0)])
        alternate = [10, 11, 12, 80, 20, 21, 22, 23, 24]
        self.assertEqual(
            _matched(tree, alternate, [(3, 4, identity, 0)]), list(range(8))
        )
        self.assertEqual(_matched(tree, alternate, [(3, 4, other, 0)]), [])
        tree.reset()
        # Media wholly in the uncacheable tail cannot change a preceding page.
        tree.writing_through(tokens, torch.arange(9), [(8, 9, identity, 0)])
        self.assertEqual(_matched(tree, tokens, [(8, 9, other, 0)]), list(range(8)))

    def test_request_insert_and_rematch_keep_media_spans(self):
        """Finished and chunked requests must use the same identity on native calls."""
        cache = self.cache_type.__new__(self.cache_type)
        cache.tree = self._tree(2)
        cache.cache_controller = None
        cache.device = torch.device("cpu")
        cache.page_size = 2
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.stack([torch.arange(16), torch.arange(20, 36)])
        )
        freed = []
        cache.token_to_kv_pool_allocator = SimpleNamespace(
            free=lambda values: freed.append(values.clone())
        )
        spans = (MultimodalKeySpan(2, 6, _identity("retained-image")),)

        def request(tokens, index):
            return SimpleNamespace(
                cache_salt=None,
                kv=SimpleNamespace(holds_kv=True, req_pool_idx=index),
                origin_input_ids=array("q", tokens),
                output_ids=array("q"),
                extra_key=None,
                mm_cache_spans=spans,
                prefix_indices=torch.empty(0, dtype=torch.int64),
                last_node=0,
                get_fill_ids=lambda: array("q", tokens),
            )

        first = request([10, 11, 90, 90, 90, 90, 20, 21], 0)
        cache.cache_finished_req(first, kv_len_to_handle=8)
        second = request([10, 11, 80, 80, 80, 80, 20, 21, 22, 23], 1)
        cache.cache_unfinished_req(second)
        self.assertEqual(second.prefix_indices.tolist(), list(range(8)) + [28, 29])
        self.assertEqual(freed[0].tolist(), list(range(20, 28)))
        self.assertEqual(cache.tree.protected_size(), 10)
        cache.cache_finished_req(second, kv_len_to_handle=10)
        self.assertEqual(cache.tree.protected_size(), 0)

        capped = RadixKey(second.origin_input_ids, limit=4, mm_spans=spans)
        matched = cache.match_prefix(MatchPrefixParams(key=capped))
        self.assertEqual(matched.device_indices.tolist(), list(range(4)))
        sliced = RadixKey(second.origin_input_ids, mm_spans=spans)[2:8]
        another = self.cache_type.__new__(self.cache_type)
        another.tree = self._tree(2)
        another.cache_controller = None
        another.device = torch.device("cpu")
        another._insert(sliced, torch.arange(40, 46))
        self.assertEqual(
            another.match_prefix(MatchPrefixParams(key=sliced)).device_indices.tolist(),
            list(range(40, 46)),
        )


if __name__ == "__main__":
    unittest.main()
