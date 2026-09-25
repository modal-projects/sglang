"""Token-only external stores cannot authenticate media-conditioned KV suffixes."""

import hashlib
import importlib
import sys
import threading
import unittest
from array import array
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.multimodal_key import MultimodalKeySpan
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _identity(content):
    return "sha256:" + hashlib.sha256(content.encode()).hexdigest()


def _external_imports():
    modules = {}
    exports = {
        "lmcache.integration.sglang.multi_process_adapter": {
            "LMCacheMPConnector": object,
        },
        "lmcache.integration.sglang.sglang_adapter": {
            "LMCacheLayerwiseConnector": object,
            "LoadMetadata": SimpleNamespace,
            "StoreMetadata": SimpleNamespace,
        },
        "lmcache.integration.sglang.utils": {
            "lmcache_get_config": lambda path: None,
        },
        "sglang.srt.mem_cache.storage.flexkv.flexkv_connector": {
            "FlexKVConnector": object,
        },
    }
    for name, attributes in exports.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        modules[name] = module
    return modules


class _Stream:
    def synchronize(self):
        pass

    def wait_stream(self, stream):
        pass


class _Allocator:
    device = "cpu"

    def __init__(self):
        self.next_slot = 100

    def available_size(self):
        return 1000

    def alloc(self, count):
        start = self.next_slot
        self.next_slot += count
        return torch.arange(start, start + count, dtype=torch.int64)

    def free(self, slots):
        pass

    def free_segments(self, segments):
        pass


class _ExternalIO:
    """CPU transfer boundary that returns all requested text and records stores."""

    def __init__(self, backend, page_size):
        self.backend = backend
        self.page_size = page_size
        self.lookups = []
        self.stores = []
        self.prefetches = []
        self.sessions_ended = []

    def chunk_size(self):
        return self.page_size

    def lookup_kv(self, token_ids, request_id=None, *, token_mask=None, handle=None):
        self.lookups.append(list(token_ids))
        if self.backend == "lmcache":
            return len(token_ids)
        return 1, int(token_mask.sum())

    def retrieve_kv(self, metadata_or_handle, slots=None):
        if self.backend == "lmcache":
            return len(metadata_or_handle.token_ids) - metadata_or_handle.offset
        return slots.numel()

    def start_load_kv(self, metadata):
        self.lookups.append(list(metadata.token_ids))
        return len(metadata.token_ids) - metadata.offset

    def start_load_kv_layerwise(self, handle, slots):
        return slots.numel(), 0

    def store_kv(self, metadata=None, *, handle=None, token_ids=None, kv_indices=None):
        if self.backend == "lmcache":
            token_ids, kv_indices = metadata.token_ids, metadata.kv_indices
        self.stores.append((list(token_ids), kv_indices.clone()))
        return 1

    def release_pending(self, request_id):
        pass

    def end_session(self, request_id):
        self.sessions_ended.append(request_id)

    def prefetch_async(self, handle, token_ids):
        self.prefetches.append(list(token_ids))


class TestMultimodalExternalCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(sys.modules, _external_imports()):
            cls.lmc_module = importlib.import_module(
                "sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache"
            )
            cls.flex_module = importlib.import_module(
                "sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache"
            )

    def setUp(self):
        stream = _Stream()
        self.enterContext(
            patch.object(
                torch,
                "get_device_module",
                return_value=SimpleNamespace(current_stream=lambda: stream),
            )
        )
        self.enterContext(
            patch.object(torch.cuda, "stream", return_value=nullcontext())
        )
        self.enterContext(
            patch.object(
                self.lmc_module, "device_stream_context", return_value=nullcontext()
            )
        )
        for module in (self.lmc_module, self.flex_module):
            self.enterContext(
                patch.object(
                    module,
                    "get_spec",
                    return_value=SimpleNamespace(speculative_eagle_topk=None),
                )
            )

    def _cache(self, backend, mode, page_size=1, is_eagle=False):
        module = self.lmc_module if backend == "lmcache" else self.flex_module
        cache_type = (
            module.LMCRadixCache if backend == "lmcache" else module.FlexKVRadixCache
        )
        mode_type = module.LMCacheMode if backend == "lmcache" else module.FlexKVMode
        cache = cache_type.__new__(cache_type)
        pool = SimpleNamespace(req_to_token=torch.arange(64).reshape(1, 64))
        RadixCache.__init__(
            cache,
            CacheInitParams(
                disable=False,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=_Allocator(),
                page_size=page_size,
                is_eagle=is_eagle,
            ),
        )
        cache._mode = mode_type[mode]
        cache.load_stream = _Stream()
        cache.store_stream = _Stream()
        cache._node_lock = threading.Lock()
        external = _ExternalIO(backend, page_size)
        if backend == "lmcache":
            cache.lmcache_connector = external
            cache._in_flight_nodes = []
            cache._mp_load_back_markers = {}
        else:
            cache.flexkv_connector = external
            cache._inflight_store_nodes = {}
            cache._load_markers = {}
        return cache, external

    @staticmethod
    def _req(tokens, spans):
        return SimpleNamespace(
            rid="example",
            cache_request_handle=CacheRequestHandle("example", 0),
            origin_input_ids=array("q", tokens),
            output_ids=array("q"),
            mm_cache_spans=spans,
            extra_key=None,
            cache_salt=None,
            priority=0,
            last_node=None,
            kv=SimpleNamespace(
                kv_committed_len=len(tokens), req_pool_idx=0, cache_protected_len=0
            ),
        )

    def _match(self, cache, tokens, spans):
        req = self._req(tokens, spans)
        result = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", tokens), mm_spans=spans),
                req=req,
            )
        )
        if result.host_hit_length:
            loaded, _ = cache.init_load_back(
                InitLoadBackParams(
                    best_match_node=result.best_match_node,
                    host_hit_length=result.host_hit_length,
                    req=req,
                )
            )
            return result.device_indices.numel() + loaded.numel()
        return result.device_indices.numel()

    def test_external_load_keeps_only_page_aligned_text_before_first_media(self):
        for backend in ("lmcache", "flexkv"):
            for mode in ("MP", "IP"):
                for page_size in (1, 4):
                    for start in (0, 1, 3, 5, 8):
                        with self.subTest(
                            backend=backend, mode=mode, page=page_size, start=start
                        ):
                            cache, external = self._cache(backend, mode, page_size)
                            tokens = list(range(12))
                            spans = (
                                MultimodalKeySpan(start, start + 1, _identity("image")),
                            )
                            expected = start // page_size * page_size
                            self.assertEqual(
                                self._match(cache, tokens, spans), expected
                            )
                            self.assertEqual(
                                external.lookups,
                                [tokens[:expected]] if expected else [],
                            )

    def test_same_media_and_earlier_media_prefix_stay_local(self):
        tokens = list(range(12))
        spans = (
            MultimodalKeySpan(2, 3, _identity("first")),
            MultimodalKeySpan(7, 8, _identity("second")),
        )
        changed = (spans[0], MultimodalKeySpan(7, 8, _identity("new second")))
        for backend in ("lmcache", "flexkv"):
            for mode in ("MP", "IP"):
                with self.subTest(backend=backend, mode=mode):
                    cache, external = self._cache(backend, mode)
                    cache.insert(
                        InsertParams(
                            key=RadixKey(array("q", tokens), mm_spans=spans),
                            value=torch.arange(12),
                        )
                    )
                    self.assertEqual(self._match(cache, tokens, spans), 12)
                    self.assertEqual(self._match(cache, tokens, changed), 7)
                    self.assertEqual(external.lookups, [])

    def test_store_exports_earlier_text_and_retains_full_local_media(self):
        tokens = list(range(12))
        for backend in ("lmcache", "flexkv"):
            for mode in ("MP", "IP"):
                for page_size in (1, 4):
                    for start in (0, 1, 5, 8):
                        with self.subTest(
                            backend=backend, mode=mode, page=page_size, start=start
                        ):
                            cache, external = self._cache(backend, mode, page_size)
                            spans = (
                                MultimodalKeySpan(start, start + 1, _identity("image")),
                            )
                            req = self._req(tokens, spans)
                            cache.cache_finished_req(req, kv_len_to_handle=len(tokens))
                            expected = start // page_size * page_size
                            self.assertEqual(len(external.stores), int(expected > 0))
                            if expected:
                                ids, slots = external.stores[0]
                                self.assertEqual(ids, tokens[:expected])
                                self.assertEqual(slots.tolist(), list(range(expected)))
                            self.assertEqual(self._match(cache, tokens, spans), 12)
                            if backend == "lmcache" and mode == "MP":
                                self.assertEqual(external.sessions_ended, [req.rid])

    def test_multimodal_bigram_requests_use_exact_local_cache_only(self):
        tokens = list(range(12))
        spans = (MultimodalKeySpan(4, 5, _identity("image")),)
        for backend in ("lmcache", "flexkv"):
            for mode in ("MP", "IP"):
                with self.subTest(backend=backend, mode=mode):
                    cache, external = self._cache(backend, mode, is_eagle=True)
                    self.assertEqual(self._match(cache, tokens, spans), 0)
                    cache.cache_finished_req(
                        self._req(tokens, spans), kv_len_to_handle=len(tokens)
                    )
                    self.assertEqual(self._match(cache, tokens, spans), 11)
                    self.assertEqual(external.lookups, [])
                    self.assertEqual(external.stores, [])

    def test_text_only_requests_keep_external_load_and_store(self):
        tokens = list(range(12))
        for backend in ("lmcache", "flexkv"):
            for mode in ("MP", "IP"):
                with self.subTest(backend=backend, mode=mode):
                    cache, external = self._cache(backend, mode, page_size=4)
                    self.assertEqual(self._match(cache, tokens, ()), 12)
                    cache.cache_finished_req(
                        self._req(tokens, ()), kv_len_to_handle=len(tokens)
                    )
                    self.assertEqual(external.stores[0][0], tokens)

    def test_prefetch_rejects_text_conditioned_on_prior_media(self):
        handle = CacheRequestHandle("prefetch", 0)
        cache, external = self._cache("flexkv", "MP", page_size=4)
        first = MultimodalKeySpan(1, 2, _identity("first"))
        cache.prefetch_from_storage(
            handle,
            cache.root_node,
            [20, 21, 22, 23],
            None,
            None,
            matched_prefix_tokens=[10, 11, 12, 13],
            matched_prefix_mm_spans=(first,),
        )
        self.assertEqual(external.prefetches, [])
        later = MultimodalKeySpan(2, 3, _identity("later"))
        cache.prefetch_from_storage(
            handle,
            cache.root_node,
            [20, 21, 22, 23],
            None,
            None,
            matched_prefix_tokens=[10, 11],
            mm_spans=(later,),
        )
        self.assertEqual(external.prefetches, [[10, 11, 20, 21]])
        cache.prefetch_from_storage(handle, cache.root_node, [30, 31, 32, 33])
        self.assertEqual(external.prefetches[-1], [30, 31, 32, 33])
        cache.page_size = 1
        cache.prefetch_from_storage(
            handle,
            cache.root_node,
            [20, 21, 22, 23],
            None,
            None,
            matched_prefix_tokens=[10, 11],
            mm_spans=(later,),
            storage_hit_end=3,
        )
        self.assertEqual(external.prefetches[-1], [10, 11, 20])


if __name__ == "__main__":
    unittest.main()
