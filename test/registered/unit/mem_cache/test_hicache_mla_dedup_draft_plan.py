"""Unit tests for MLA host-dedup draft-cache registration (v0.5.17 backport).

Upstream PR #26691 tests the HiCacheDraftPlan machinery (PACKED vs SIDECAR
modes and UnifiedRadixCache sidecars). That machinery does not exist at
v0.5.17: draft KV is always registered as an independent, rank-local legacy
host pool via maybe_register_hicache_draft, which is exactly the behavior the
dedup path requires (only the target MLA KV is deduplicated; draft KV stays
rank-local on every TP rank). These tests pin that guarantee for the
backport: --enable-mla-hicache-host-dedup must not change draft registration,
even when the tree cache is a UnifiedRadixCache.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _run_registration(*, enable_dedup: bool, tree_cache):
    draft_pool = mock.Mock(spec=MLATokenToKVPool)
    draft_pool.size = 256
    set_draft_kv_pool = mock.Mock()
    tree_cache.cache_controller = SimpleNamespace(
        mem_pool_host=SimpleNamespace(size=1024),
        set_draft_kv_pool=set_draft_kv_pool,
    )
    server_args = SimpleNamespace(
        enable_mla_hicache_host_dedup=enable_dedup,
        hicache_mem_layout="page_first",
        hicache_storage_backend=None,
    )
    with (
        mock.patch.object(
            kv_cache_builder, "get_draft_kv_pool", return_value=draft_pool
        ),
        mock.patch(
            "sglang.srt.mem_cache.pool_host.mla.MLATokenToKVPoolHost"
        ) as host_pool_cls,
    ):
        kv_cache_builder.maybe_register_hicache_draft(
            tree_cache=tree_cache,
            draft_worker=object(),
            spec_algorithm=SimpleNamespace(is_ngram=lambda: False),
            server_args=server_args,
            enable_hierarchical_cache=True,
            page_size=64,
        )
    return host_pool_cls, set_draft_kv_pool


class TestHiCacheMLADedupDraftRegistration(unittest.TestCase):
    def test_dedup_keeps_draft_rank_local_legacy_pool(self):
        host_pool_cls, set_draft_kv_pool = _run_registration(
            enable_dedup=True, tree_cache=SimpleNamespace()
        )

        # Draft host pool is a full per-rank MLATokenToKVPoolHost (never an
        # is_dummy dedup pool) and is registered as the independent sidecar.
        host_pool_cls.assert_called_once()
        self.assertNotIn("is_dummy", host_pool_cls.call_args.kwargs)
        set_draft_kv_pool.assert_called_once()

    def test_unified_cache_still_uses_legacy_draft_pool_with_dedup(self):
        tree_cache = UnifiedRadixCache.__new__(UnifiedRadixCache)

        host_pool_cls, set_draft_kv_pool = _run_registration(
            enable_dedup=True, tree_cache=tree_cache
        )

        host_pool_cls.assert_called_once()
        set_draft_kv_pool.assert_called_once()

    def test_registration_unchanged_without_dedup(self):
        host_pool_cls, set_draft_kv_pool = _run_registration(
            enable_dedup=False, tree_cache=SimpleNamespace()
        )

        host_pool_cls.assert_called_once()
        set_draft_kv_pool.assert_called_once()


if __name__ == "__main__":
    unittest.main()
