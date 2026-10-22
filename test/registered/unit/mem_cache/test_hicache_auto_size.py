import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.mem_cache import hicache_auto_size as sizing
from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host import base, mamba, mha
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
from sglang.srt.runtime_context import get_context, get_memory, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDFlashHiCacheStagingBudget(CustomTestCase):
    """Regression: late HiCache scratch must not spend reserved runtime headroom."""

    def setUp(self):
        self.enterContext(
            get_context().override_server_args(
                enable_hierarchical_cache=True,
                hicache_mem_layout="page_first",
                mem_fraction_static=0.75,
                tp_size=2,
                cuda_graph_config=CudaGraphConfig(),
            )
        )
        self.enterContext(
            get_parallel().override(
                world_group=SimpleNamespace(world_size=2, cpu_group=None),
            )
        )
        # CPU tests replace only hardware/JIT availability, not allocation shapes.
        self.enterContext(
            patch.object(
                kcc.current_platform, "capabilities", hicache_device_kernels=True
            )
        )
        for module in (mha, mamba):
            self.enterContext(patch.object(module, "_is_cuda", True))
            self.enterContext(
                patch.object(
                    module,
                    "can_use_write_back_jit_kernel",
                    side_effect=lambda *, element_size: element_size % 16 == 0,
                    create=True,
                )
            )
        cfg = self.cfg = object.__new__(kcc.KVCacheConfigurator)
        cfg.device, cfg.gpu_id = "cuda", 0
        cfg.page_size = 64
        cfg.post_capture_kv_active = False
        cfg.is_hybrid_swa = cfg.use_mla_backend = False
        cfg.kv_cache_dtype = torch.bfloat16
        cfg.layer_info = SimpleNamespace(start_layer=0, end_layer=40)
        cfg.model_config = SimpleNamespace(
            is_multimodal=False,
            get_num_kv_heads=lambda tp, dcp: 2 // tp,
            head_dim=256,
            v_head_dim=256,
        )
        cfg.mambaish_config = SimpleNamespace(
            full_attention_layer_ids=list(range(0, 40, 4)),
            mamba2_cache_params=SimpleNamespace(
                layers=[i for i in range(40) if i % 4],
                shape=SimpleNamespace(temporal=(16, 128, 128), conv=((4096, 3),)),
                dtype=SimpleNamespace(temporal=torch.float32, conv=torch.bfloat16),
            ),
        )
        cfg.dflash_cache_sizing = SimpleNamespace(context_gpu_reserve_bytes=3 << 30)
        self.enterContext(
            patch.object(kcc, "get_available_gpu_memory", return_value=80)
        )
        # The existing recurrent-pool reserve is independent of transfer scratch.
        self.enterContext(
            patch.object(
                kcc.KVCacheConfigurator,
                "_handle_max_mamba_cache",
                side_effect=lambda n: n,
            )
        )

    def test_initial_budget_prices_late_staging_once(self):
        expected = (80 - 25 - 3) * (1 << 30) - 106_332_160
        self.assertEqual(self.cfg._profile_available_bytes(100), expected)
        self.assertEqual(self.cfg._profile_available_bytes(100), expected)

    def test_disabled_cache_and_direct_layout_do_not_reserve_staging(self):
        for fields in (
            dict(enable_hierarchical_cache=False),
            dict(hicache_mem_layout="page_first_direct"),
        ):
            with (
                self.subTest(fields=fields),
                get_memory().override(**fields),
            ):
                self.assertEqual(self.cfg._profile_available_bytes(100), 52 << 30)

    def test_flag_off_profile_keeps_existing_budget(self):
        self.cfg.dflash_cache_sizing = None
        self.assertEqual(self.cfg._profile_available_bytes(100), 55 << 30)

    def test_unavailable_kernels_do_not_reserve_staging(self):
        with (
            patch.object(mha, "can_use_write_back_jit_kernel", return_value=False),
            patch.object(mamba, "can_use_write_back_jit_kernel", return_value=False),
        ):
            self.assertEqual(self.cfg._profile_available_bytes(100), 52 << 30)

    def test_reserve_matches_allocations_and_kernel_fallbacks(self):
        for temporal, conv in (
            ((16, 128, 128), ((4096, 3),)),
            ((0,), ((16, 8), (3,))),  # conv-only; the unaligned row falls back
        ):
            with self.subTest(temporal=temporal, conv=conv):
                self.cfg.mambaish_config.mamba2_cache_params.shape = SimpleNamespace(
                    temporal=temporal, conv=conv
                )
                host = object.__new__(mamba.MambaPoolHost)
                host.layout, host._buffer_device = "page_first", "meta"
                host.temporal_buffer = torch.empty(
                    (257, 30, 1, *temporal), dtype=torch.float32, device="meta"
                )
                host.conv_buffer = [
                    torch.empty((257, 30, 1, *s), dtype=torch.bfloat16, device="meta")
                    for s in conv
                ]
                host._init_write_back_staging_buffers()
                info = SimpleNamespace(
                    head_num=1,
                    head_dim=256,
                    v_head_dim=256,
                    dtype=torch.bfloat16,
                    device="meta",
                )
                kv = mha.prepare_mha_write_back_staging(
                    info, layer_num=10, page_size=64
                )
                allocated = sum(
                    t.nbytes
                    for t in (
                        *kv,
                        host.temporal_staging_buffer,
                        *host.conv_staging_buffers,
                    )
                    if t is not None
                )
                self.assertEqual(self.cfg._dflash_hicache_staging_bytes(), allocated)
                # A host pool smaller than one transfer chunk needs less scratch.
                small = mha.prepare_mha_write_back_staging(
                    info, layer_num=10, page_size=64, page_capacity=4
                )
                self.assertLess(sum(t.nbytes for t in small), sum(t.nbytes for t in kv))
                # Post-capture preparation can pass its existing allocations through.
                reused = mha.prepare_mha_write_back_staging(
                    info, layer_num=10, page_size=64, staging=kv
                )
                self.assertIs(reused, kv)


class TestHiCacheAutoSize(CustomTestCase):
    def test_raw_checkpoint_reservation_leaves_room_for_target_host_pools(self):
        pool = MHATokenToKVPool(
            size=128,
            page_size=2,
            dtype=torch.float16,
            head_num=2,
            head_dim=4,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            page_size=2,
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: pool,
                dflash_cache_sizing=SimpleNamespace(checkpoint_window_bytes=1000),
            ),
        )
        with (
            get_context().override_server_args(
                enable_hierarchical_cache=True,
                enable_dflash_swa_cache=True,
                max_mamba_cache_size=5,
            ),
            patch.object(base, "ranks_per_host", return_value=1),
            patch.object(
                base,
                "available_host_memory_bytes",
                return_value=base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 10_000,
            ),
            sizing.auto_size_hicache(params, None, enabled=True),
        ):
            base.host_memory_budget_bytes(5000)
            host = MHATokenToKVPoolHost(
                pool.get_device_pool_infos()[0].buffer_info,
                get_memory().hicache_ratio,
                0,
                2,
                "layer_first",
                pin_memory=False,
                device="cpu",
                device_capacity_tokens=pool.size,
                start_layer=pool.start_layer,
                end_layer=pool.end_layer,
                write_back_staging=pool.hicache_write_back_staging,
            )
            self.assertLessEqual(5000 + host.size * host.size_per_token, 8000)

    def test_hybrid_target_draft_uses_its_sidecar_slot_capacity(self):
        # EAGLE3 can pair a hybrid target with a plain MHA draft. Full drafts
        # follow full target slots; SWA drafts follow the smaller SWA capacity.
        target = Mock(
            spec=SWAKVPool,
            size=128,
            full_kv_pool=Mock(
                size=128,
                host_capacity_bytes=None,
                get_kv_size_bytes=Mock(return_value=4096),
            ),
            swa_kv_pool=Mock(
                size=32,
                host_capacity_bytes=None,
                get_kv_size_bytes=Mock(return_value=1024),
            ),
        )
        draft_mha = Mock(
            spec=MHATokenToKVPool,
            size=16,
            host_capacity_bytes=None,
            get_kv_size_bytes=Mock(return_value=(128, 128)),
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=target)),
            page_size=2,
        )
        for draft, expected_sidecar_bytes in (
            (draft_mha, 2048),
            (Mock(spec=BaseSWAKVPool, swa_kv_pool=draft_mha), 512),
        ):
            with self.subTest(
                draft_type=type(draft).__name__, bytes=expected_sidecar_bytes
            ):
                plan = Mock(mode="sidecar", device_pools=(draft,))
                self.assertEqual(
                    sizing._estimate_hicache_bytes(params, plan),
                    4096 + 1024 + expected_sidecar_bytes,
                )

    def test_unified_views_are_sized_from_host_capacity(self):
        """Unified sub-pools answer get_kv_size_bytes with zero (UnifiedKVPool
        logs the shared buffer once) and publish host_capacity_bytes instead,
        the weight the explicit host-size split already uses. An estimate taken
        from get_kv_size_bytes is zero device bytes, and the default ratio then
        divides by it."""
        target = Mock(
            spec=SWAKVPool,
            size=128,
            full_kv_pool=Mock(
                size=128,
                host_capacity_bytes=4096,
                get_kv_size_bytes=Mock(return_value=(0, 0)),
            ),
            swa_kv_pool=Mock(
                size=32,
                host_capacity_bytes=1024,
                get_kv_size_bytes=Mock(return_value=(0, 0)),
            ),
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=target)),
            page_size=2,
        )
        self.assertEqual(sizing._estimate_hicache_bytes(params, None), 4096 + 1024)

    def test_default_ratio_fits_host_budget_and_pools_book_it(self):
        """With only --enable-hierarchical-cache the default ratio shrinks to the
        per-rank budget, pools book one snapshot, and an explicit ratio opts out."""
        pool = MHATokenToKVPool(
            size=128,
            page_size=2,
            dtype=torch.float16,
            head_num=2,
            head_dim=4,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=pool)),
            page_size=2,
        )
        rank_budget = 10_000
        # Four ranks per host (e.g. TP8 over two 4-GPU nodes) share what psutil reports.
        host_free = base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 4 * rank_budget
        with (
            get_context().override_server_args(enable_hierarchical_cache=True),
            patch.object(base, "ranks_per_host", return_value=4),
            patch.object(base, "available_host_memory_bytes", return_value=host_free),
            sizing.auto_size_hicache(params, None, enabled=True),
        ):
            ratio = get_memory().hicache_ratio
            self.assertLess(ratio, 2.0)
            host = MHATokenToKVPoolHost(
                pool.get_device_pool_infos()[0].buffer_info,
                ratio,
                0,
                2,
                "layer_first",
                pin_memory=False,
                device="cpu",
                device_capacity_tokens=pool.size
                if pool.host_capacity_tokens is None
                else pool.host_capacity_tokens,
                start_layer=pool.start_layer,
                end_layer=pool.end_layer,
                write_back_staging=pool.hicache_write_back_staging,
            )
            self.assertLessEqual(host.size * host.size_per_token, 0.8 * rank_budget)
            with self.assertRaisesRegex(ValueError, "Not enough host memory"):
                MHATokenToKVPoolHost(
                    pool.get_device_pool_infos()[0].buffer_info,
                    ratio,
                    0,
                    2,
                    "layer_first",
                    pin_memory=False,
                    device="cpu",
                    device_capacity_tokens=pool.size
                    if pool.host_capacity_tokens is None
                    else pool.host_capacity_tokens,
                    start_layer=pool.start_layer,
                    end_layer=pool.end_layer,
                    write_back_staging=pool.hicache_write_back_staging,
                )
        self.assertIsNone(base._host_memory_budget.get())

        explicit = ServerArgs(model_path="dummy", hicache_ratio=2.0)
        explicit.resolve_once()
        self.assertIsNone(
            resolution_result(explicit, "hicache_host_memory_fraction", 0.8)
        )


if __name__ == "__main__":
    unittest.main()
