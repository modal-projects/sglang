"""Recycled pages must have one owner and no writes from a previous owner."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
    DSV4NPUTokenToKVPoolAllocator,
)
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    UnquantizedKVCacheMethod,
)
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.paged import (
    PagedTokenToKVPoolAllocator,
    alloc_extend_naive,
)
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedMambaSWATokenToKVPoolAllocator,
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.unified_mamba import (
    UnifiedMambaTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    _set_kv_buffer_impl,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE_SIZE = 4
NUM_PAGES = 4


def _pool(dtype=torch.bfloat16):
    pool = MLATokenToKVPool.__new__(MLATokenToKVPool)
    pool.page_size = PAGE_SIZE
    pool.size = NUM_PAGES * PAGE_SIZE
    pool.kv_buffer = [
        torch.full((pool.size + PAGE_SIZE, 1, 8), 7.0).to(dtype) for _ in range(2)
    ]
    return pool


def _allocator(pool=None, *, need_sort=False, cls=PagedTokenToKVPoolAllocator):
    return cls(
        size=NUM_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=pool,
        need_sort=need_sort,
    )


class _PendingWrite:
    def __init__(self, pool, indices):
        self.pool = pool
        self.indices = indices
        self.complete = False

    def finish(self):
        if not self.complete:
            for buf in self.pool.kv_buffer:
                buf[self.indices] = 13
            self.complete = True

    def wait(self):
        self.finish()


class _AllocKernel:
    """Replace only accelerator launch with the allocator's CPU reference."""

    def __init__(self, decode):
        self.decode = decode

    def __getitem__(self, grid):
        def run(*args):
            if self.decode:
                seq_lens, last_loc, free_pages, out, _, page_size = args
                prefix_lens = seq_lens - 1
            else:
                prefix_lens, seq_lens, last_loc, free_pages, out, _, page_size = args
            alloc_extend_naive(
                prefix_lens, seq_lens, last_loc, free_pages, out, page_size, "cpu"
            )

        return run


def _release(allocator, indices, doorway):
    if doorway == "free":
        allocator.free(indices)
    elif doorway == "segment":
        allocator.free_segment(indices, start_pos=0)
    else:
        allocator.free_page_ids(indices[::PAGE_SIZE] // PAGE_SIZE)


class TestPagedOwnership(CustomTestCase):
    def test_invalid_and_repeated_frees_cannot_create_capacity(self):
        """Invalid page IDs and cross-call frees cannot alias live requests."""
        for need_sort in (False, True):
            for doorway in ("free", "segment", "page_ids"):
                with self.subTest(need_sort=need_sort, doorway=doorway):
                    allocator = _allocator(need_sort=need_sort)
                    allocated = allocator.alloc(PAGE_SIZE * 2)
                    _release(allocator, allocated[:PAGE_SIZE], doorway)
                    _release(allocator, allocated[:PAGE_SIZE], doorway)
                    allocator.free(torch.tensor([-1, 0, PAGE_SIZE - 1, 10000]))
                    allocator.free_page_ids(torch.tensor([-1, 0, NUM_PAGES + 1]))
                    self.assertEqual(allocator.available_size(), PAGE_SIZE * 3)
                    remaining = allocator.alloc(PAGE_SIZE * 3)
                    pages = remaining[::PAGE_SIZE] // PAGE_SIZE
                    self.assertEqual(set(pages.tolist()), {1, 3, 4})
                    self.assertEqual(pages.unique().numel(), 3)
                    self.assertIsNone(allocator.alloc(PAGE_SIZE))

    def test_group_deduplicates_both_release_buckets_and_owns_views(self):
        allocator = _allocator(need_sort=True)
        allocated = allocator.alloc(PAGE_SIZE * 2)
        page_ids = torch.tensor([1, 1, 2, 2])
        allocator.free_group_begin()
        allocator.free(allocated[:PAGE_SIZE])
        allocator.free_page_ids(page_ids)
        allocated.zero_()
        page_ids.zero_()
        self.assertEqual(allocator.available_size(), PAGE_SIZE * 2)
        allocator.free_group_end()
        self.assertEqual(allocator.available_size(), allocator.size)
        self.assertEqual(allocator.num_staged_pages, 2)
        allocator.alloc(allocator.size)
        self.assertIsNone(allocator.alloc(PAGE_SIZE))

    def test_clear_and_resize_reset_ownership_and_deferred_groups(self):
        allocator = _allocator(need_sort=True)
        old = allocator.alloc(PAGE_SIZE)
        allocator.free_group_begin()
        allocator.free(old)
        allocator.free_page_ids(torch.tensor([1]))
        allocator.resize(SimpleNamespace(max_total_num_tokens=PAGE_SIZE * 2))
        self.assertEqual(allocator.available_size(), PAGE_SIZE * 2)
        allocator.free(old)
        allocator.free_page_ids(torch.tensor([3]))
        self.assertEqual(allocator.available_size(), PAGE_SIZE * 2)
        current = allocator.alloc(PAGE_SIZE * 2)
        allocator.free(current)
        self.assertEqual(allocator.available_size(), PAGE_SIZE * 2)
        allocator.clear()
        allocator.free(current)
        self.assertEqual(allocator.available_size(), PAGE_SIZE * 2)
        self.assertEqual(allocator.num_staged_pages, 0)

    def test_all_allocation_doors_mark_pages_without_clearing_partial_pages(self):
        with (
            patch(
                "sglang.srt.mem_cache.allocator.paged.alloc_extend_kernel",
                _AllocKernel(False),
            ),
            patch(
                "sglang.srt.mem_cache.allocator.paged.alloc_decode_kernel",
                _AllocKernel(True),
            ),
        ):
            for doorway in ("alloc", "extend", "decode", "npu_decode"):
                with self.subTest(doorway=doorway):
                    pool = _pool()
                    cls = (
                        NPUPagedTokenToKVPoolAllocator
                        if doorway == "npu_decode"
                        else PagedTokenToKVPoolAllocator
                    )
                    allocator = _allocator(pool, cls=cls)
                    first = allocator.alloc(PAGE_SIZE)
                    for buf in pool.kv_buffer:
                        buf[first] = 11
                    if doorway == "alloc":
                        new = allocator.alloc(PAGE_SIZE)
                    elif doorway == "extend":
                        new = allocator.alloc_extend(
                            torch.tensor([3]),
                            torch.tensor([3]),
                            torch.tensor([5]),
                            torch.tensor([5]),
                            torch.tensor([6]),
                            2,
                        )
                        self.assertEqual(new.tolist(), [7, 8])
                    else:
                        new = allocator.alloc_decode(
                            torch.tensor([5]), torch.tensor([5]), torch.tensor([7])
                        )
                        self.assertEqual(new.tolist(), [8])
                    for buf in pool.kv_buffer:
                        self.assertTrue(torch.all(buf[first] == 11))
                        self.assertTrue(torch.all(buf[8:12] == 0))
                        self.assertTrue(torch.all(buf[12:16] == 7))
                    allocator.free_page_ids(torch.tensor([2]))
                    self.assertEqual(allocator.available_size(), PAGE_SIZE * 3)

    def test_npu_reference_extend_preserves_release_ownership(self):
        """The NPU large-extend branch shares paged release accounting."""
        allocator = NPUPagedTokenToKVPoolAllocator(
            size=200 * PAGE_SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        indices = allocator.alloc_extend(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([800]),
            torch.tensor([800]),
            torch.tensor([-1]),
            800,
        )
        allocator.free(indices)
        allocator.free(indices)
        self.assertEqual(allocator.available_size(), 800)


class TestPageClearingOrder(CustomTestCase):
    def test_reported_completion_can_be_waited_without_a_page_free(self):
        pool = _pool()
        allocator = _allocator(pool)
        indices = allocator.alloc(PAGE_SIZE)
        allocator.note_forward_launch(_PendingWrite(pool, indices))
        allocator.wait_for_forward()
        self.assertTrue(torch.all(pool.kv_buffer[0][indices] == 13))

    def test_late_forward_write_finishes_before_recycled_page_clear(self):
        for need_sort in (False, True):
            for doorway in ("free", "segment", "page_ids"):
                for grouped in (False, True):
                    with self.subTest(
                        need_sort=need_sort, doorway=doorway, grouped=grouped
                    ):
                        pool = _pool()
                        allocator = _allocator(pool, need_sort=need_sort)
                        indices = allocator.alloc(allocator.size)
                        event = _PendingWrite(pool, indices[:PAGE_SIZE])
                        allocator.note_forward_launch(event)
                        if grouped:
                            allocator.free_group_begin()
                        _release(allocator, indices[:PAGE_SIZE], doorway)
                        if grouped:
                            allocator.free_group_end()
                        reused = allocator.alloc(PAGE_SIZE)
                        event.finish()
                        self.assertEqual(reused.tolist(), indices[:PAGE_SIZE].tolist())
                        for buf in pool.kv_buffer:
                            self.assertTrue(torch.all(buf[reused] == 0))

    def test_free_before_scheduled_forward_launch_keeps_the_hazard(self):
        pool = _pool()
        allocator = _allocator(pool)
        indices = allocator.alloc(allocator.size)
        allocator.free(indices[:PAGE_SIZE])
        allocator.carry_frees_into_next_launch()
        event = _PendingWrite(pool, indices[:PAGE_SIZE])
        allocator.note_forward_launch(event)
        reused = allocator.alloc(PAGE_SIZE)
        event.finish()
        self.assertTrue(torch.all(pool.kv_buffer[0][reused] == 0))

    def test_each_handout_stream_waits_for_the_late_writer(self):
        """A wait queued on one stream does not complete writes for another."""
        queues = [[], []]
        current = 0

        class QueuedPool(MLATokenToKVPool):
            def zero_pages(self, pages):
                queues[current].append(lambda: MLATokenToKVPool.zero_pages(self, pages))

        def drain(stream):
            for action in queues[stream]:
                action()
            queues[stream].clear()

        pool = QueuedPool.__new__(QueuedPool)
        pool.page_size = PAGE_SIZE
        pool.kv_buffer = _pool().kv_buffer
        allocator = _allocator(pool)
        indices = allocator.alloc(allocator.size)
        drain(0)
        writer = _PendingWrite(pool, indices[: 2 * PAGE_SIZE])
        event = SimpleNamespace(wait=lambda: queues[current].append(writer.finish))
        allocator.note_forward_launch(event)
        allocator.free(indices[: 2 * PAGE_SIZE])
        first = allocator.alloc(PAGE_SIZE)
        current = 1
        second = allocator.alloc(PAGE_SIZE)
        drain(1)
        writer.finish()
        drain(0)
        for reused in (first, second):
            self.assertTrue(torch.all(pool.kv_buffer[0][reused] == 0))

    def test_composite_children_receive_carry_once_and_keep_compaction_event(self):
        topologies = (
            (SWATokenToKVPoolAllocator, ("full_attn_allocator", "swa_attn_allocator")),
            (
                PureSWATokenToKVPoolAllocator,
                ("full_attn_allocator", "swa_attn_allocator"),
            ),
            (
                HiSparseTokenToKVPoolAllocator,
                ("logical_attn_allocator", "hisparse_attn_allocator"),
            ),
            (
                DeepSeekV4HiSparseTokenToKVPoolAllocator,
                ("logical_attn_allocator", "hisparse_attn_allocator"),
            ),
            (
                UnifiedSWATokenToKVPoolAllocator,
                ("full_attn_allocator", "swa_attn_allocator"),
            ),
            (
                UnifiedMambaTokenToKVPoolAllocator,
                ("full_attn_allocator", "mamba_allocator"),
            ),
            (
                UnifiedMambaSWATokenToKVPoolAllocator,
                ("full_attn_allocator", "swa_attn_allocator", "mamba_allocator"),
            ),
            (
                DSV4NPUTokenToKVPoolAllocator,
                ("full_attn_allocator", "swa_attn_allocator", "c128_attn_allocator"),
            ),
        )
        for cls, attributes in topologies:
            with self.subTest(cls=cls.__name__):
                composite = cls.__new__(cls)
                pools = [_pool() for _ in attributes]
                children = [_allocator(pool) for pool in pools]
                if cls is PureSWATokenToKVPoolAllocator:
                    children[1] = children[0]
                unique_children = list(dict.fromkeys(children))
                allocated = [child.alloc(child.size) for child in unique_children]
                events = [
                    _PendingWrite(child.get_kvcache(), indices[:PAGE_SIZE])
                    for child, indices in zip(unique_children, allocated)
                ]
                completion = SimpleNamespace(
                    wait=lambda: [event.finish() for event in events]
                )
                for name, child in zip(attributes, children):
                    setattr(composite, name, child)
                compaction_event = object()
                composite._latest_forward_done_event = compaction_event
                for child, indices in zip(unique_children, allocated):
                    child.free(indices[:PAGE_SIZE])
                composite.carry_frees_into_next_launch()
                composite.note_forward_launch(completion)
                reused = [child.alloc(PAGE_SIZE) for child in unique_children]
                completion.wait()
                self.assertIs(composite._latest_forward_done_event, compaction_event)
                for child, indices in zip(unique_children, reused):
                    self.assertTrue(
                        torch.all(child.get_kvcache().kv_buffer[0][indices] == 0)
                    )

    def test_draft_pool_registration_clears_both_index_spaces(self):
        target, draft = _pool(torch.float8_e4m3fn), _pool()
        hybrid = HybridLinearKVPool.__new__(HybridLinearKVPool)
        hybrid.page_size = PAGE_SIZE
        hybrid.full_kv_pool = target
        allocator = _allocator(hybrid)
        configurator = SimpleNamespace(is_draft_worker=True, is_hybrid_swa=False)
        with get_context().override_server_args(disaggregation_mode="null"):
            result = KVCacheConfigurator._build_token_to_kv_pool_allocator(
                configurator,
                sizes=None,
                token_to_kv_pool=draft,
                is_dsv4_model=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=allocator,
            )
        indices = result.alloc(PAGE_SIZE)
        for pool in (target, draft):
            self.assertTrue(
                torch.all(pool.kv_buffer[0].view(torch.uint8)[indices] == 0)
            )
            self.assertTrue(torch.all(pool.kv_buffer[0][0].float() == 7))

    def test_partial_page_extension_does_not_clear_the_existing_owner(self):
        pool = _pool()
        allocator = _allocator(pool)
        indices = allocator.alloc(PAGE_SIZE)
        pool.kv_buffer[0][indices] = 11
        allocator.note_forward_launch(_PendingWrite(pool, indices))
        with patch(
            "sglang.srt.mem_cache.allocator.paged.alloc_decode_kernel",
            _AllocKernel(True),
        ):
            new = allocator.alloc_decode(
                torch.tensor([3]), torch.tensor([3]), torch.tensor([5])
            )
        self.assertEqual(new.tolist(), [6])
        self.assertTrue(torch.all(pool.kv_buffer[0][indices] == 11))

    def test_float8_storage_and_pool_layouts_clear_whole_pages(self):
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            pool = _pool(dtype)
            indices = _allocator(pool).alloc(PAGE_SIZE)
            self.assertTrue(
                torch.all(pool.kv_buffer[0].view(torch.uint8)[indices] == 0)
            )
        for layout in ("nhd", "hnd", "vectorized_5d"):
            with self.subTest(layout=layout):
                pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
                pool.page_size = PAGE_SIZE
                pool.use_hnd = layout == "hnd"
                pool.kv_cache_layout = layout
                pool.quant_method = UnquantizedKVCacheMethod()
                shape = (20, 2, 8) if layout == "nhd" else (5, 2, 4, 8)
                if layout == "vectorized_5d":
                    shape = (5, 2, 1, 4, 8)
                pool.k_buffer = [torch.full(shape, 7.0)]
                pool.v_buffer = [torch.full(shape, 7.0)]
                allocator = _allocator(pool)
                allocator.alloc(PAGE_SIZE)
                for buf in (pool.k_buffer[0], pool.v_buffer[0]):
                    self.assertTrue(torch.all(buf[0] == 7))
                    self.assertTrue(
                        torch.all(buf[4:8] == 0)
                        if layout == "nhd"
                        else torch.all(buf[1] == 0)
                    )

    def test_unsupported_layouts_and_virtual_page_sizes_do_not_register(self):
        class OtherLayout(MLATokenToKVPool):
            pass

        pool = _pool()
        unsupported = OtherLayout.__new__(OtherLayout)
        unsupported.page_size = PAGE_SIZE
        unsupported.kv_buffer = pool.kv_buffer
        hybrid = HybridLinearKVPool.__new__(HybridLinearKVPool)
        hybrid.page_size = PAGE_SIZE
        hybrid.full_kv_pool = unsupported
        for cache in (unsupported, hybrid):
            allocator = _allocator(cache)
            allocator.register_zero_pages_pool(pool)
            allocator.alloc(PAGE_SIZE)
            self.assertTrue(torch.all(pool.kv_buffer[0] == 7))
        pool.page_size = PAGE_SIZE // 2
        allocator = _allocator(pool)
        allocator.alloc(PAGE_SIZE)
        self.assertTrue(torch.all(pool.kv_buffer[0] == 7))
        quantized = MHATokenToKVPool.__new__(MHATokenToKVPool)
        quantized.page_size = PAGE_SIZE
        quantized.quant_method = object()
        quantized.use_hnd = False
        quantized.kv_cache_layout = "nhd"
        quantized.k_buffer = [torch.full((20, 2, 8), 7.0)]
        quantized.v_buffer = [torch.full((20, 2, 8), 7.0)]
        _allocator(quantized).alloc(PAGE_SIZE)
        self.assertTrue(torch.all(quantized.k_buffer[0] == 7))


class TestReservedFallbackWrites(CustomTestCase):
    def test_mha_index_writes_restore_the_zero_padding_source(self):
        for capture in (False, True):
            with self.subTest(capture=capture):
                k = torch.tensor([[float("nan")] * 8, [2.0] * 8])
                v = torch.tensor([[float("nan")] * 8, [3.0] * 8])
                k_cache, v_cache = torch.zeros(4, 8), torch.zeros(4, 8)
                stream = SimpleNamespace(wait_stream=lambda other: None)
                device = SimpleNamespace(
                    current_stream=lambda: stream, stream=lambda other: nullcontext()
                )
                with (
                    patch("sglang.srt.mem_cache.memory_pool._is_cuda", False),
                    patch("sglang.srt.mem_cache.memory_pool._is_hip", False),
                    patch(
                        "sglang.srt.mem_cache.memory_pool._cpu_has_amx_support", False
                    ),
                    patch(
                        "sglang.srt.model_executor.runner.get_is_capture_mode",
                        return_value=capture,
                    ),
                ):
                    _set_kv_buffer_impl(
                        k,
                        v,
                        k_cache,
                        v_cache,
                        torch.tensor([0, 2]),
                        8,
                        torch.float32,
                        device,
                        4,
                        alt_stream=stream,
                    )
                self.assertTrue(torch.all(k_cache[0] == 0))
                self.assertTrue(torch.all(v_cache[0] == 0))
                torch.testing.assert_close(k_cache[2], k[1])
                torch.testing.assert_close(v_cache[2], v[1])

    def test_mla_combined_rows_restore_padding_after_quantized_storage_write(self):
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                pool = _pool(dtype)
                pool.dtype = dtype
                pool.store_dtype = (
                    torch.uint8 if dtype is torch.float8_e4m3fn else dtype
                )
                pool.kv_buffer = [pool.kv_buffer[0].view(pool.store_dtype)]
                pool.start_layer = 0
                pool.kernel_page_blocks = 1
                pool.write_loc_is_dcp_resolved = True
                pool.dsa_kv_cache_store_fp8 = False
                values = torch.full((2, 1, 8), 2.0, dtype=torch.bfloat16)
                values[0] = float("nan")
                pool.set_kv_buffer(
                    SimpleNamespace(layer_id=0), torch.tensor([0, 2]), values, values
                )
                self.assertTrue(torch.all(pool.kv_buffer[0].view(torch.uint8)[0] == 0))
                torch.testing.assert_close(
                    pool.kv_buffer[0].view(dtype)[2].float(), values[1].float()
                )


if __name__ == "__main__":
    unittest.main()
