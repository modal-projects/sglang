"""Host-slot ownership under pending DFlash backups, fan-out, and eviction.

CPU tensors exercise the real HiCache transfer engine and host-pool copies.
Only accelerator streams/events are replaced to control completion precisely;
the CUDA transfer-ordering test lives in registered/hicache.
"""

import gc
import unittest
import weakref
from array import array
from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.mem_cache import l2_transfer
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope
from sglang.srt.mem_cache.pool_host.dflash import DFlashFeatureTailHostPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType, EvictLayer
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context, get_memory, get_spec
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.dflash_feature_tail import (
    DFlashFeatureCapture,
    DFlashFeatureTail,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _Event:
    def __init__(self, backend):
        self.backend = backend
        self.complete = False

    def record(self):
        self.backend.current.events.append(self)

    def wait(self, stream):
        stream.wait_event(self)

    def query(self):
        return self.complete

    def synchronize(self):
        self.complete = True


class _Stream:
    device = torch.device("cpu")

    def __init__(self):
        self.events = []
        self.waits = []

    def wait_event(self, event):
        self.waits.append(event)

    def synchronize(self):
        for event in self.events:
            event.synchronize()


class _CPUDevice:
    Stream = _Stream

    def __init__(self):
        self.current = _Stream()

    def Event(self, **kwargs):
        return _Event(self)

    def current_stream(self):
        return self.current

    @contextmanager
    def stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous


class TestDFlashFeatureTailHostPool(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            device="cpu",
            page_size=1,
            _mamba_cache_chunk_size=4,
            mamba_track_interval=8,
            mamba_radix_cache_strategy="extra_buffer",
            enable_dflash_swa_cache=True,
            enable_hierarchical_cache=True,
        )
        override.install()
        self.addCleanup(override.restore)
        self.backend = _CPUDevice()
        self.enterContext(patch.object(l2_transfer, "device_module", self.backend))
        self.engine = l2_transfer.L2TransferEngine("direct")
        self.history = torch.arange(24 * 8, dtype=torch.float32).view(24, 8)

    def pool(self, capacity=2):
        pool = DFlashFeatureTailHostPool(
            capacity=capacity,
            window_size=4,
            feature_size=8,
            dtype=torch.float32,
            transfer_engine=self.engine,
            pin_memory=False,
        )
        self.addCleanup(pool.destroy)
        return pool

    def tail(self, end):
        return DFlashFeatureTail.capture(
            self.history, start_pos=0, end_pos=end, window_size=4
        )

    def test_checkpoint_boundaries_and_fanout_survive_source_reuse(self):
        pool = self.pool()
        early, late = pool.backup(self.tail(3)), pool.backup(self.tail(20))
        self.assertIsNone(pool.backup(self.tail(24)))
        self.engine.device_to_host_stream.synchronize()
        expected = self.history.clone()
        self.history.fill_(-1)
        for handle, end in ((late, 20), (early, 3), (late, 20)):
            restored = pool.restore(handle)
            self.assertEqual(
                (restored.start_pos, restored.end_pos), (max(0, end - 4), end)
            )
            torch.testing.assert_close(
                restored.as_tensor(), expected[max(0, end - 4) : end]
            )
            for part in restored.parts:
                part.zero_()
        pool.release(early)
        pool.release(late)
        self.assertEqual(pool.available_size(), 2)
        self.engine.host_to_device_stream.synchronize()
        self.assertEqual(pool.available_size(), 2)

    def test_eviction_during_backup_retains_source_and_fences_slot_reuse(self):
        pool = self.pool(capacity=1)
        tail = self.tail(12)
        source = weakref.ref(tail.parts[0])
        handle = pool.backup(tail)
        previous_use = pool._live[handle]
        del tail
        gc.collect()
        pool.release(handle)
        self.assertIsNotNone(source())
        replacement = pool.backup(self.tail(20))
        self.assertIsNotNone(replacement)
        torch.testing.assert_close(replacement._indices, handle._indices)
        self.assertIn(previous_use, self.engine.device_to_host_stream.waits)
        with self.assertRaisesRegex(ValueError, "released"):
            pool.restore(handle)
        self.engine.device_to_host_stream.synchronize()
        pool.poll()
        self.assertIsNone(source())
        with self.assertRaisesRegex(ValueError, "released"):
            pool.restore(handle)
        torch.testing.assert_close(
            pool.restore(replacement).as_tensor(), self.history[16:20]
        )

    def test_pending_copy_limit_bounds_gpu_sources_even_with_free_host_slots(self):
        pool = DFlashFeatureTailHostPool(
            capacity=4,
            window_size=4,
            feature_size=8,
            dtype=torch.float32,
            transfer_engine=self.engine,
            pin_memory=False,
            max_pending=1,
        )
        self.addCleanup(pool.destroy)
        handle = pool.reserve(12)
        self.assertIsNotNone(handle)
        second = pool.reserve(16)
        self.assertIsNotNone(second)
        pool.write_reserved(handle, self.tail(12))
        first_done = pool._live[handle]
        self.assertFalse(first_done.query())
        pool.write_reserved(second, self.tail(16))
        self.assertTrue(first_done.query())
        self.assertEqual(len(pool._pending), 1)
        self.assertEqual(pool.available_size(), 2)

    def test_eviction_waits_for_last_of_multiple_loads(self):
        pool = self.pool(capacity=1)
        handle = pool.backup(self.tail(12))
        self.engine.device_to_host_stream.synchronize()
        first = pool.restore(handle)
        first_done = self.engine.host_to_device_stream.events[-1]
        second = pool.restore(handle)
        second_done = self.engine.host_to_device_stream.events[-1]
        destination = weakref.ref(second.parts[0])
        del second
        pool.release(handle)
        first_done.synchronize()
        self.assertEqual(pool.available_size(), 1)
        self.assertIs(pool._slot_fences[int(handle._indices[0])], second_done)
        self.assertIsNotNone(destination())
        for part in first.parts:
            part.zero_()
        torch.testing.assert_close(destination(), self.history[8:12])
        second_done.synchronize()
        self.assertEqual(pool.available_size(), 1)
        self.assertIsNone(destination())

    def test_restores_and_delayed_donations_share_a_bounded_transfer_queue(self):
        pool = DFlashFeatureTailHostPool(
            capacity=4,
            window_size=4,
            feature_size=8,
            dtype=torch.float32,
            transfer_engine=self.engine,
            pin_memory=False,
            max_pending=2,
        )
        self.addCleanup(pool.destroy)
        pending = pool.reserve(20)
        ready = pool.backup(self.tail(12))
        # Restores can arrive before the reserved checkpoint's forward finishes.
        # Repeated loads must reclaim completed leases rather than exceed the cap.
        for _ in range(5):
            restored = pool.restore(ready)
            torch.testing.assert_close(restored.as_tensor(), self.history[8:12])
            self.assertLessEqual(len(pool._pending), 2)
        pool.write_reserved(pending, self.tail(20))
        self.assertLessEqual(len(pool._pending), 2)
        torch.testing.assert_close(
            pool.restore(pending).as_tensor(), self.history[16:20]
        )
        self.assertLessEqual(len(pool._pending), 2)

    def test_foreign_handles_and_incomplete_windows_do_not_consume_capacity(self):
        pool, other = self.pool(), self.pool()
        foreign = other.backup(self.tail(12))
        for operation in (pool.restore, pool.release):
            with self.assertRaisesRegex(ValueError, "another pool"):
                operation(foreign)
        for tail in (
            DFlashFeatureTail(9, (torch.zeros(3, 8),)),
            DFlashFeatureTail(-1, (torch.zeros(4, 8),)),
            DFlashFeatureTail(0, (torch.zeros(5, 8),)),
            DFlashFeatureTail(0, (torch.zeros(4, 7),)),
            DFlashFeatureTail(0, (torch.zeros(4, 8, dtype=torch.bfloat16),)),
            DFlashFeatureTail(0, (torch.zeros(4, 16)[:, ::2],)),
        ):
            with self.subTest(tail=tail):
                with self.assertRaisesRegex(ValueError, "complete raw window"):
                    pool.backup(tail)
                self.assertEqual(pool.available_size(), 2)

    def test_failed_transfers_leave_capacity_and_checkpoint_recoverable(self):
        pool = self.pool(capacity=1)
        with patch.object(
            pool,
            "backup_from_device_all_layer_physical",
            side_effect=RuntimeError("copy failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                pool.backup(self.tail(12))
        self.assertEqual(pool.available_size(), 1)
        handle = pool.backup(self.tail(12))
        with patch.object(
            pool,
            "load_to_device_per_layer_physical",
            side_effect=RuntimeError("copy failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                pool.restore(handle)
        torch.testing.assert_close(pool.restore(handle).as_tensor(), self.history[8:12])
        pool.release(handle)
        with self.assertRaisesRegex(ValueError, "released"):
            pool.release(handle)

    def test_destroy_drains_copies_before_discarding_host_storage(self):
        pool = self.pool(capacity=1)
        handle = pool.backup(self.tail(12))
        done = self.engine.device_to_host_stream.events[-1]
        pool.destroy()
        self.assertTrue(done.query())
        self.assertEqual(pool.available_size(), 0)
        with self.assertRaisesRegex(ValueError, "released"):
            pool.restore(handle)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pool.backup(self.tail(12))
        pool.destroy()

    def test_tail_pool_shares_the_hicache_host_memory_budget(self):
        with host_memory_budget_scope(2 * 4 * 8 * 4):
            pool = self.pool(capacity=2)
            self.assertEqual(pool.available_size(), 2)
            with self.assertRaisesRegex(ValueError, "Insufficient host memory"):
                self.pool(capacity=1)

    def test_reservation_is_unreadable_until_filled_and_is_write_once(self):
        pool = self.pool(capacity=1)
        handle = pool.reserve(12)
        self.assertFalse(pool.is_ready(handle))
        self.assertEqual(pool.available_size(), 0)
        with self.assertRaisesRegex(RuntimeError, "not completed"):
            pool.restore(handle)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            pool.write_reserved(handle, self.tail(16))
        pool.write_reserved(handle, self.tail(12))
        # Readiness means a copy was submitted, not a rank-local DMA query.
        self.assertFalse(self.engine.device_to_host_stream.events[-1].query())
        self.assertTrue(pool.is_ready(handle))
        with self.assertRaisesRegex(ValueError, "already"):
            pool.write_reserved(handle, self.tail(12))
        torch.testing.assert_close(pool.restore(handle).as_tensor(), self.history[8:12])

    def test_cancelled_reservation_cannot_write_into_reused_slot(self):
        pool = self.pool(capacity=1)
        abandoned = pool.reserve(12)
        pool.clear()
        replacement = pool.reserve(20)
        with self.assertRaisesRegex(ValueError, "released"):
            pool.write_reserved(abandoned, self.tail(12))
        pool.write_reserved(replacement, self.tail(20))
        torch.testing.assert_close(
            pool.restore(replacement).as_tensor(), self.history[16:20]
        )

    def cache(self, capacity=2):
        pool = self.pool(capacity)
        state_params = Mamba2CacheParams(
            shape=Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=8,
                n_groups=1,
                num_heads=1,
                head_dim=8,
                state_size=4,
                conv_kernel=2,
            ),
            layers=[1],
        )
        req_pool = HybridReqToTokenPool(
            size=4,
            mamba_size=32,
            mamba_spec_state_size=4,
            max_context_len=256,
            device="cpu",
            enable_memory_saver=False,
            cache_params=state_params,
            mamba_layer_ids=[1],
            enable_mamba_extra_buffer=True,
            speculative_num_draft_tokens=16,
        )
        kv_pool = MHATokenToKVPool(
            size=256,
            page_size=1,
            dtype=torch.float32,
            head_num=1,
            head_dim=4,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        allocator = TokenToKVPoolAllocator(
            size=256,
            dtype=torch.float32,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                tree_components=(ComponentType.FULL, ComponentType.MAMBA),
                enable_mamba_extra_buffer=True,
                dflash_feature_tail_host_pool=pool,
                tree_core_backend="rust",
            )
        )
        self.assertEqual(cache._tree_core_backend, "python")
        return cache

    def request_capture(self, cache, end=20, checkpoint=12):
        params = SamplingParams(max_new_tokens=128)
        params.normalize(None)
        req = Req("req", "", array("q", range(1, end + 1)), params, vocab_size=128)
        req.refresh_fill_ids()
        req.extend_end = end
        cache.req_to_token_pool.alloc([req])
        indices = cache.token_to_kv_pool_allocator.alloc(end)
        cache.req_to_token_pool.write((req.kv.req_pool_idx, slice(0, end)), indices)
        req.kv.kv_allocated_len = req.kv.kv_committed_len = end
        req.kv.mamba_last_track_idx = 0
        req.kv.mamba_last_track_seqlen = checkpoint
        capture = DFlashFeatureCapture.prefill(
            reqs=[req],
            target_hidden=self.history[:end],
            prefix_lens=[0],
            extend_lens=[end],
            checkpoints=[(0, checkpoint)],
            window_size=4,
        )
        return req, capture

    def match(self, cache, end=20):
        return cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", range(1, end + 1))),
            )
        )

    def test_host_attachment_requires_opt_in_and_hicache(self):
        for scope in (
            get_spec().override(enable_dflash_swa_cache=False),
            get_memory().override(enable_hierarchical_cache=False),
        ):
            with (
                scope,
                self.assertRaisesRegex(ValueError, "raw checkpoint tails require"),
            ):
                self.cache()

    def test_pending_donation_keeps_exact_boundary_after_slot_replacement(self):
        cache = self.cache()
        req, capture = self.request_capture(cache)
        old_slot = req.kv.mamba_ping_pong_track_buffer[0].clone()
        cache.checkpoint(req, up_to=20)
        self.assertNotEqual(req.kv.mamba_ping_pong_track_buffer[0], old_slot)
        self.assertEqual(self.match(cache).device_prefix_len, 0)
        capture.consume([req])
        match = self.match(cache)
        self.assertEqual(match.device_prefix_len, 12)
        component = cache.components[ComponentType.MAMBA]
        self.assertEqual(
            component.tree_core.get_component_device_value(
                match.best_match_node, ComponentType.MAMBA
            ).item(),
            old_slot.item(),
        )
        tail = component.restore_dflash_features(match.best_match_node, 12)
        torch.testing.assert_close(tail.as_tensor(), self.history[8:12])
        self.assertEqual(req.kv.dflash_feature_state.snapshot().end_pos, 20)
        release_kv_cache(req, cache, checkpoint=False)
        self.assertIsNone(req.kv.dflash_feature_state)
        self.assertEqual(self.match(cache).device_prefix_len, 12)
        torch.testing.assert_close(
            component.restore_dflash_features(match.best_match_node, 12).as_tensor(),
            self.history[8:12],
        )

    def test_duplicate_prefix_releases_unused_pending_reservation(self):
        cache = self.cache()
        first, first_capture = self.request_capture(cache)
        cache.checkpoint(first, up_to=20)
        first_capture.consume([first])
        second, second_capture = self.request_capture(cache)
        cache.checkpoint(second, up_to=20)
        pool = cache.dflash_feature_tail_host_pool
        self.assertEqual(pool.available_size(), 1)
        second_capture.consume([second])
        self.assertEqual(pool.available_size(), 1)
        self.assertEqual(self.match(cache).device_prefix_len, 12)
        self.assertEqual(second.kv.dflash_feature_state.snapshot().end_pos, 20)

    def test_evicted_or_reset_pending_capture_never_revives_cache_bundle(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                cache = self.cache(capacity=1)
                req, capture = self.request_capture(cache)
                cache.checkpoint(req, up_to=20)
                if reset:
                    cache.reset()
                else:
                    node = cache.tree_core.node_by_id(req.last_node)
                    cache.components[ComponentType.MAMBA].evict_component(
                        node,
                        defaultdict(list),
                        defaultdict(list),
                        EvictLayer.ALL,
                    )
                self.assertEqual(
                    cache.dflash_feature_tail_host_pool.available_size(), 1
                )
                capture.consume([req])
                self.assertEqual(self.match(cache).device_prefix_len, 0)
                self.assertEqual(
                    cache.dflash_feature_tail_host_pool.available_size(), 1
                )

    def test_capacity_decline_preserves_request_checkpoint(self):
        cache = self.cache(capacity=1)
        pool = cache.dflash_feature_tail_host_pool
        pool.reserve(4)
        req, capture = self.request_capture(cache)
        capture.consume([req])
        slots = req.kv.mamba_ping_pong_track_buffer.clone()
        free_states = cache.req_to_token_pool.mamba_allocator.available_size()
        cache.checkpoint(req, up_to=20)
        torch.testing.assert_close(req.kv.mamba_ping_pong_track_buffer, slots)
        self.assertEqual(
            cache.req_to_token_pool.mamba_allocator.available_size(), free_states
        )
        self.assertIsNotNone(req.kv.dflash_feature_state.checkpoint(0, 12))
        self.assertEqual(self.match(cache).device_prefix_len, 0)
        self.assertEqual(pool.available_size(), 0)

    def test_logical_capacity_agrees_despite_different_dma_completion(self):
        first, second = self.pool(capacity=1), self.pool(capacity=1)
        a, b = first.backup(self.tail(12)), second.backup(self.tail(12))
        first._live[a].synchronize()
        self.assertFalse(second._live[b].query())
        first.release(a)
        second.release(b)
        self.assertEqual(first.available_size(), second.available_size())
        for pool in (first, second):
            handle = pool.reserve(20)
            # Abandoning an unfilled reservation must retain the old DMA fence.
            fence = pool._slot_fences[int(handle._indices[0])]
            pool.release(handle)
            replacement = pool.backup(self.tail(20))
            self.assertIsNotNone(replacement)
            self.assertIn(fence, self.engine.device_to_host_stream.waits)

    def test_aborted_pending_donation_releases_host_reservation(self):
        cache = self.cache(capacity=1)
        req, capture = self.request_capture(cache)
        cache.checkpoint(req, up_to=20)
        req.finished_reason = FINISH_ABORT()
        release_kv_cache(req, cache, checkpoint=False)
        capture.consume([req])
        self.assertEqual(cache.dflash_feature_tail_host_pool.available_size(), 1)
        self.assertEqual(self.match(cache).device_prefix_len, 0)

    def test_finished_donation_uses_older_checkpoint_after_decode_overshoot(self):
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH

        cache = self.cache()
        req, capture = self.request_capture(cache, end=8, checkpoint=8)
        capture.consume([req])
        later = DFlashFeatureCapture.prefill(
            reqs=[req],
            target_hidden=self.history[8:24],
            prefix_lens=[8],
            extend_lens=[16],
            checkpoints=[(1, 16)],
            window_size=4,
        )
        later.consume([req])
        req.kv.mamba_last_track_idx = 1
        req.kv.mamba_last_track_seqlen = 16
        req.kv.mamba_prev_track_seqlen = 8
        req.finished_reason = FINISH_LENGTH(0)
        old_slot = req.kv.mamba_ping_pong_track_buffer[0].clone()
        release_kv_cache(req, cache, checkpoint=True)
        match = self.match(cache)
        self.assertEqual(match.device_prefix_len, 8)
        component = cache.components[ComponentType.MAMBA]
        self.assertEqual(
            component.tree_core.get_component_device_value(
                match.best_match_node, ComponentType.MAMBA
            ).item(),
            old_slot.item(),
        )
        torch.testing.assert_close(
            component.restore_dflash_features(match.best_match_node, 8).as_tensor(),
            self.history[4:8],
        )
        self.assertEqual(cache.req_to_token_pool.mamba_allocator.available_size(), 31)

    def test_worker_restores_current_projection_and_only_extends_target_suffix(self):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models.dflash import DFlashDraftModel
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        # The fixture stores CPU KV even when this test runs on a GPU host.
        for backend in ("_is_cuda", "_is_hip"):
            self.enterContext(
                patch(f"sglang.srt.mem_cache.memory_pool.{backend}", False)
            )
        cache = self.cache()
        donor, capture = self.request_capture(cache)
        capture.consume([donor])
        cache.checkpoint(donor, up_to=20)
        match = self.match(cache)
        release_kv_cache(donor, cache, checkpoint=False)
        kv_pool = MHATokenToKVPool(
            size=256,
            page_size=1,
            dtype=torch.float32,
            head_num=1,
            head_dim=4,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        translator = KVIndexTranslator(
            req_to_token=cache.req_to_token_pool.req_to_token,
            token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
            token_to_kv_pool=kv_pool,
            page_size=1,
            device="cpu",
            is_draft_worker=True,
        )
        # Small deterministic attention geometry keeps projection and physical
        # writes observable while using the real worker and DFlash projection.
        model = SimpleNamespace(
            config=SimpleNamespace(hidden_size=4),
            num_context_features=2,
            quantizable=False,
            fc=torch.nn.Linear(8, 4, bias=False),
            hidden_norm=torch.nn.RMSNorm(4),
        )
        model.project_target_hidden = lambda raw: (
            DFlashDraftModel.project_target_hidden(model, raw)
        )
        model.prepare_context_hidden_for_kv = lambda layer, hidden: hidden
        attn = SimpleNamespace(
            num_kv_heads=1,
            head_dim=4,
            kv_proj_only=lambda hidden: (hidden * 2, hidden * 3),
            apply_k_norm=lambda keys: keys,
            apply_k_rope=lambda positions, keys: keys + positions[:, None],
            attn=SimpleNamespace(layer_id=0, k_scale=None, v_scale=None),
        )
        model.layers = [SimpleNamespace(self_attn=attn)]
        worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
        worker._feature_tail_window = 4
        worker.draft_model = model
        worker.draft_owns_attention = False
        worker.lilicorr = None
        worker._use_fused_kv_materialize = False
        worker.model_runner = SimpleNamespace(
            device=torch.device("cpu"), prefill_attention_backend_str="torch_native"
        )
        worker.draft_model_runner = SimpleNamespace(
            kv_index_translator=translator, token_to_kv_pool=kv_pool
        )
        worker._tp_sync = SimpleNamespace(sync=lambda *a: None)
        with torch.no_grad():
            model.fc.weight.copy_(torch.arange(32).view(4, 8) / 50)

        # The same host bundle supports two independent branches, including an
        # fc change. The target still receives only the two-token continuation.
        outputs = []
        for revision in range(2):
            with torch.no_grad():
                model.fc.weight[0].mul_(-1)
            req, _ = self.request_capture(cache, end=14)
            req.kv.dflash_feature_state = None
            req.best_match_node = match.best_match_node
            row = cache.req_to_token_pool.req_to_token[req.kv.req_pool_idx]
            kv_pool.k_buffer[0].fill_(float("nan"))
            kv_pool.v_buffer[0].fill_(float("nan"))
            batch = ScheduleBatch(
                reqs=[req],
                forward_mode=ForwardMode.EXTEND,
                sampling_info=SimpleNamespace(is_all_greedy=True),
                prefix_lens=[12],
                extend_lens=[2],
                seq_lens=torch.tensor([14]),
                req_pool_indices=torch.tensor([req.kv.req_pool_idx]),
                out_cache_loc=row[12:14],
                mamba_prefill_checkpoints=[(0, 13)],
                tree_cache=cache,
                req_to_token_pool=cache.req_to_token_pool,
            )

            def target_forward(received, **kwargs):
                self.assertEqual(received.prefix_lens, [12])
                self.assertEqual(received.extend_lens, [2])
                return GenerationBatchResult(
                    logits_output=LogitsProcessorOutput(
                        None, hidden_states=self.history[12:14].clone()
                    ),
                    next_token_ids=torch.tensor([42]),
                    kv_loc_plan=translator.plan(
                        req_pool_indices=batch.req_pool_indices,
                        seq_lens=batch.seq_lens,
                        seq_lens_cpu=batch.seq_lens,
                        write_virtual=batch.out_cache_loc,
                    ),
                )

            worker._target_worker = SimpleNamespace(
                forward_batch_generation=target_forward
            )
            output = worker.forward_batch_generation(batch)
            output.dflash_feature_capture.consume([req])
            with torch.no_grad():
                projected = model.project_target_hidden(self.history[8:14])
            torch.testing.assert_close(
                kv_pool.k_buffer[0][row[8:14], 0],
                projected * 2 + torch.arange(8, 14)[:, None],
            )
            torch.testing.assert_close(kv_pool.v_buffer[0][row[8:14], 0], projected * 3)
            self.assertTrue(torch.isnan(kv_pool.k_buffer[0][row[:8]]).all())
            torch.testing.assert_close(
                req.kv.dflash_feature_state.checkpoint(0, 13).as_tensor(),
                self.history[9:13],
            )
            torch.testing.assert_close(
                req.kv.dflash_feature_state.snapshot().as_tensor(),
                self.history[10:14],
            )
            outputs.append(kv_pool.v_buffer[0][row[8:12]].clone())
            release_kv_cache(req, cache, checkpoint=False)
        self.assertFalse(torch.equal(*outputs))

        # A long suffix supplies its own live window. Only an early interior
        # checkpoint can still require old raw rows; neither case needs old KV.
        for checkpoint, needs_raw in ((None, False), ((0, 13), True), ((0, 16), False)):
            req, _ = self.request_capture(cache)
            req.kv.dflash_feature_state = None
            req.best_match_node = match.best_match_node
            kv_pool.k_buffer[0].fill_(float("nan"))
            batch = ScheduleBatch(
                reqs=[req],
                prefix_lens=[12],
                extend_lens=[8],
                req_pool_indices=torch.tensor([req.kv.req_pool_idx]),
                mamba_prefill_checkpoints=[checkpoint],
                tree_cache=cache,
                req_to_token_pool=cache.req_to_token_pool,
            )
            worker._restore_cached_features(batch)
            self.assertEqual(req.kv.dflash_feature_state is not None, needs_raw)
            self.assertTrue(torch.isnan(kv_pool.k_buffer[0]).all())
            capture = DFlashFeatureCapture.prefill(
                reqs=[req],
                target_hidden=self.history[12:20],
                prefix_lens=[12],
                extend_lens=[8],
                checkpoints=[checkpoint],
                window_size=4,
            )
            capture.consume([req])
            torch.testing.assert_close(
                req.kv.dflash_feature_state.snapshot().as_tensor(),
                self.history[16:20],
            )
            if checkpoint is not None:
                end = checkpoint[1]
                torch.testing.assert_close(
                    req.kv.dflash_feature_state.checkpoint(0, end).as_tensor(),
                    self.history[end - 4 : end],
                )
            release_kv_cache(req, cache, checkpoint=False)

    def test_tail_follows_last_recurrent_copy_and_stays_on_split_child(self):
        for first_eviction in (EvictLayer.DEVICE, EvictLayer.HOST):
            with self.subTest(first_eviction=first_eviction):
                cache = self.cache(capacity=1)
                req, capture = self.request_capture(cache)
                capture.consume([req])
                cache.checkpoint(req, up_to=20)
                node = cache.tree_core.node_by_id(req.last_node)
                handle = node.component_data[ComponentType.MAMBA].metadata[
                    "dflash_feature_tail"
                ]
                self.assertEqual(self.match(cache, end=8).device_prefix_len, 0)
                self.assertNotIn(
                    "dflash_feature_tail",
                    node.parent.component_data[ComponentType.MAMBA].metadata,
                )
                self.assertIs(
                    node.component_data[ComponentType.MAMBA].metadata[
                        "dflash_feature_tail"
                    ],
                    handle,
                )
                # Model a completed HiCache recurrent-state backup; its physical
                # bytes use a separate host pool and are not read by raw restore.
                node.component_data[ComponentType.MAMBA].host_value = torch.tensor([1])
                component = cache.components[ComponentType.MAMBA]
                component.evict_component(
                    node, defaultdict(list), defaultdict(list), first_eviction
                )
                self.assertTrue(cache.dflash_feature_tail_host_pool.is_ready(handle))
                torch.testing.assert_close(
                    component.restore_dflash_features(node.id, 12).as_tensor(),
                    self.history[8:12],
                )
                component.evict_component(
                    node, defaultdict(list), defaultdict(list), EvictLayer.ALL
                )
                self.assertFalse(cache.dflash_feature_tail_host_pool.contains(handle))
                self.engine.device_to_host_stream.synchronize()
                self.engine.host_to_device_stream.synchronize()
                self.assertEqual(
                    cache.dflash_feature_tail_host_pool.available_size(), 1
                )


if __name__ == "__main__":
    unittest.main()
