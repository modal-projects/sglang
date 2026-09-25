"""Delayed verification owns exact cache generations through release.

The controller, pool ownership and both TreeCore implementations are real. Only
CPU allocation selection is overridden; no generation operation is stubbed.
"""

import sys
from array import array
from collections import defaultdict
from unittest.mock import patch

import pytest
import test_unified_radix_cache_unittest as cache_fixtures
import torch

from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType, EvictLayer
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _cache(backend, auxiliary=None, page_size=1):
    components = (ComponentType.FULL,)
    if auxiliary is not None:
        components += (auxiliary,)
    cfg = cache_fixtures.CacheConfig(
        page_size=page_size,
        components=components,
        num_layers=2,
        full_attention_layer_ids=(0,),
        sliding_window_size=4 if auxiliary == ComponentType.SWA else None,
        kv_size=64,
        max_num_reqs=8,
        max_context_len=64,
        head_num=1,
        head_dim=8,
        mamba_cache_size=16,
        mamba_intermediate_size=8,
        mamba_num_heads=1,
        mamba_head_dim=8,
        mamba_state_size=2,
        mamba_conv_kernel=2,
    )
    with (
        patch.object(cache_fixtures, "_TREE_CORE_TEST_BACKEND", backend),
        patch.object(cache_fixtures, "get_device", return_value="cpu"),
    ):
        cache, _, _ = cache_fixtures.build_fixture(cfg, mamba_cache_chunk_size=2)
    if page_size > 1 and auxiliary is None:
        old_allocator = cache.token_to_kv_pool_allocator
        cache.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
            size=cfg.kv_size,
            page_size=page_size,
            dtype=cfg.dtype,
            device="cpu",
            kvcache=old_allocator.get_kvcache(),
            need_sort=False,
        )
    return cache


def _request(cache, tokens, rid="req", session=None, admit=True):
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", tokens),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=16),
        session=session,
    )
    if session is not None:
        session._inflight = True
        session._inflight_rid = rid
    req.init_next_round_input(tree_cache=cache, cow_mamba=False)
    req.set_extend_range(0, len(tokens))
    if admit:
        _admit(cache, req)
    return req


def _admit(cache, req):
    pool = cache.req_to_token_pool
    assert pool.alloc([req]) is not None
    prefix = req.prefix_indices
    tail_len = len(req.get_fill_ids()) - len(prefix)
    partial = min(tail_len, (-len(prefix)) % cache.page_size)
    remaining = tail_len - partial
    pages = (remaining + cache.page_size - 1) // cache.page_size
    allocated = cache.token_to_kv_pool_allocator.alloc(pages * cache.page_size)
    assert allocated is not None
    tail = allocated[:remaining]
    if partial:
        tail = torch.cat(
            [
                prefix[-1] + torch.arange(1, partial + 1),
                tail,
            ]
        )
    indices = torch.cat([prefix, tail.to(torch.int64)])
    pool.write((req.kv.req_pool_idx, slice(0, len(indices))), indices.to(torch.int32))
    req.kv.kv_allocated_len = len(indices)
    req.kv.kv_committed_len = len(indices)
    req.lock_receipt = cache.inc_lock_ref(req.last_node).to_dec_params()
    cache.record_prefix_admission(req)
    req.extend_batch_idx += 1


def _finish(cache, req, is_insert=True, is_retract=False):
    if not is_retract and req.finished_reason is None:
        req.finished_reason = FINISH_LENGTH(len(req.output_ids))
        req.finished_len = len(req.output_ids)
    release_kv_cache(req, cache, is_insert=is_insert, is_retract=is_retract)


def _match_len(cache, tokens):
    return len(
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        ).device_indices
    )


def _insert(cache, tokens):
    values = cache.token_to_kv_pool_allocator.alloc(len(tokens))
    assert values is not None
    return cache.insert(
        InsertParams(key=RadixKey(array("q", tokens)), value=values.to(torch.int64))
    ).last_device_node


def _check(cache):
    cache.tree_core.sanity_check(
        list(cache.ongoing_write_through), list(cache.ongoing_load_back)
    )


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_finish_insert_updates_receipt_and_delayed_results_drain(backend):
    cache = _cache(backend)
    req = _request(cache, [1, 2, 3, 4])
    first = cache.capture_verification_attempt(req)
    second = cache.capture_verification_attempt(req)
    assert first is second
    _finish(cache, req)
    assert _match_len(cache, [1, 2, 3, 4]) == 4
    assert cache.tree_core.prefix_ref_counts()[0] == 1
    cache.release_verification_attempt(first)
    assert cache.tree_core.prefix_ref_counts()[0] == 1
    cache.invalidate_verification_attempt(second)
    assert _match_len(cache, [1, 2, 3, 4]) == 0
    _check(cache)
    fresh = _request(cache, [1, 2, 3, 4], rid="fresh")
    _finish(cache, fresh)
    assert _match_len(cache, [1, 2, 3, 4]) == 4
    cache.invalidate_verification_attempt(second)
    assert _match_len(cache, [1, 2, 3, 4]) == 4
    cache.release_verification_attempt(second)
    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    _check(cache)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_skipped_finish_preserves_prefill_receipt(backend):
    cache = _cache(backend)
    req = _request(cache, [1, 2, 3, 4])
    cache.cache_unfinished_req(req)
    ref = cache.capture_verification_attempt(req)
    _finish(cache, req, is_insert=False)
    cache.invalidate_verification_attempt(ref)
    assert _match_len(cache, [1, 2, 3, 4]) == 0
    cache.release_verification_attempt(ref)
    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    _check(cache)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_first_admitted_device_boundary_survives_prefill_reanchor(backend):
    cache = _cache(backend)
    _insert(cache, [1, 2])
    req = _request(cache, [1, 2, 3, 4])
    assert req.cache_validation_state.start == 2
    cache.cache_unfinished_req(req, chunked=True)
    assert req.kv.cache_protected_len == 4
    ref = cache.capture_verification_attempt(req)
    _finish(cache, req, is_insert=False)
    cache.invalidate_verification_attempt(ref)
    assert _match_len(cache, [1, 2, 3, 4]) == 2
    cache.release_verification_attempt(ref)
    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    _check(cache)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_invalidated_old_reader_cannot_reinsert_or_donate(backend):
    cache = _cache(backend)
    source = _request(cache, [1, 2, 3, 4], rid="source")
    ref = cache.capture_verification_attempt(source)
    _finish(cache, source)
    reader = _request(cache, [1, 2, 3, 4, 5], rid="reader")
    old_node = reader.last_node
    cache.invalidate_verification_attempt(ref)
    assert cache.tree_core.is_invalidated(old_node)
    fresh = _request(cache, [1, 2, 3, 4, 5], rid="fresh")
    _finish(cache, fresh)
    new_node = cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4, 5])))
    ).last_device_node
    cache.cache_unfinished_req(reader)
    assert reader.cache_invalid
    assert reader.last_node == old_node
    _finish(cache, reader)
    assert not cache.tree_core.is_invalidated(new_node)
    assert _match_len(cache, [1, 2, 3, 4, 5]) == 5
    cache.release_verification_attempt(ref)
    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    _check(cache)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_invalid_mamba_request_skips_prepare_and_reclaims_owned_state(backend):
    cache = _cache(backend, ComponentType.MAMBA)
    mamba = cache.req_to_token_pool.mamba_allocator
    initial = mamba.available_size()
    req = _request(cache, [1, 2, 3, 4])
    slot = req.kv.mamba_pool_idx.clone()
    ref = cache.capture_verification_attempt(req)
    cache.invalidate_verification_attempt(ref)
    cache.cache_unfinished_req(req)
    assert torch.equal(req.kv.mamba_pool_idx, slot)
    assert cache.total_size() == (0, 0)
    _finish(cache, req)
    cache.release_verification_attempt(ref)
    assert mamba.available_size() == initial
    assert cache.token_to_kv_pool_allocator.available_size() == 64
    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    _check(cache)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_receipt_survives_split_and_old_suffix_eviction(backend):
    cache = _cache(backend)
    req = _request(cache, [1, 2, 3, 4])
    ref = cache.capture_verification_attempt(req)
    _finish(cache, req)
    prefix = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", [1, 2]))))
    lock = cache.inc_lock_ref(prefix.last_device_node).to_dec_params()
    cache.evict(EvictParams(num_tokens=2))
    assert _match_len(cache, [1, 2, 3, 4]) == 2
    cache.invalidate_verification_attempt(ref)
    assert _match_len(cache, [1, 2, 3, 4]) == 0
    cache.dec_lock_ref(prefix.last_device_node, lock)
    cache.release_verification_attempt(ref)
    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    _check(cache)


class _IndexCopyController:
    """CPU transport boundary; the controller/core allocate and own real slots."""

    def __init__(self, cache, succeed=True):
        self.cache = cache
        cache.load_back_threshold = 1
        self.succeed = succeed
        self.calls = []

    def load(self, *, host_indices, node_id, extra_pools):
        self.calls.append((node_id, len(host_indices), extra_pools))
        if not self.succeed:
            return None
        cache = self.cache
        allocator = cache.token_to_kv_pool_allocator
        full = allocator.full_attn_allocator if cache.supports_swa() else allocator
        result = full.alloc(len(host_indices))
        assert result is not None
        for transfer in extra_pools or ():
            if transfer.device_indices is not None:
                continue
            if transfer.name == PoolName.MAMBA:
                transfer.device_indices = cache.req_to_token_pool.mamba_allocator.alloc(
                    len(transfer.host_indices)
                )
            elif transfer.name == PoolName.SWA:
                transfer.device_indices = allocator.swa_attn_allocator.alloc(
                    len(transfer.host_indices)
                )
            assert transfer.device_indices is not None
        return result.to(torch.int64)


def _backup(cache, node, auxiliary=None):
    core = cache.tree_core
    core.set_hicache_enabled()
    core.is_write_back = True
    start, end = core.prefix_node_span(node)
    transfers = {}
    if auxiliary is not None:
        count = 1 if auxiliary == ComponentType.MAMBA else end - start
        transfers[auxiliary] = [
            PoolTransfer(
                name=PoolName.MAMBA
                if auxiliary == ComponentType.MAMBA
                else PoolName.SWA,
                host_indices=torch.arange(200, 200 + count),
                nodes_to_load=[node],
            )
        ]
    core.mark_write_through_pending([node], node)
    core.commit_backup(node, torch.arange(100 + start, 100 + end), transfers)
    core.finish_write_through([node], node)


def _ack_load(cache, node):
    operation = cache.ongoing_load_back.pop(node)
    cache.dec_lock_ref(node, operation.lock_params)
    cache.dec_host_lock_ref(node, operation.host_lock_params)
    cache.tree_core.finish_load_back(node)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_full_host_restore_retains_boundary_after_reanchor_and_ack(backend):
    cache = _cache(backend)
    node = _insert(cache, [1, 2, 3, 4])
    assert _match_len(cache, [1, 2]) == 2
    _backup(cache, node)
    cache._demote(node, defaultdict(int))
    cache.cache_controller = _IndexCopyController(cache)
    req = _request(cache, [1, 2, 3, 4, 5], admit=False)
    assert len(req.prefix_indices) == 2
    loaded, req.last_node = cache.init_load_back(
        InitLoadBackParams(
            best_match_node=req.best_match_node,
            host_hit_length=req.host_hit_length,
            req=req,
        )
    )
    assert len(loaded) == 2
    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    req.prefix_indices = torch.cat([req.prefix_indices, loaded])
    req.kv.cache_protected_len = len(req.prefix_indices)
    _admit(cache, req)
    assert req.cache_validation_state.start == 2
    cache.cache_unfinished_req(req)
    assert req.kv.cache_protected_len == 5
    ref = cache.capture_verification_attempt(req)
    _finish(cache, req, is_insert=False)
    cache.invalidate_verification_attempt(ref)
    assert _match_len(cache, [1, 2, 3, 4, 5]) == 2
    # The old transfer acknowledgment releases only the retired generation.
    _ack_load(cache, node)
    cache.release_verification_attempt(ref)
    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    _check(cache)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_recurrent_only_restore_records_checkpoint_with_zero_full_indices(backend):
    cache = _cache(backend, ComponentType.MAMBA)
    source = _request(cache, [1, 2, 3, 4], rid="source")
    _finish(cache, source)
    node = source.last_node
    _backup(cache, node, ComponentType.MAMBA)
    freed = cache.tree_core.evict_component(
        node, ComponentType.MAMBA, EvictLayer.DEVICE
    )
    cache._free_values(freed.device_frees, freed.host_frees)
    cache.cache_controller = _IndexCopyController(cache)
    req = _request(cache, [1, 2, 3, 4, 5], admit=False)
    # A component restore at an already attached FULL checkpoint has no new
    # FULL indices. Preserve that observed device boundary before calling the
    # same public restore entry point used by admission.
    req.prefix_indices = cache.tree_core.collect_full_device_indices(
        node, cache.root_node_handle()
    )
    req.last_node = node
    req.kv.cache_protected_len = len(req.prefix_indices)
    req.cache_validation_state.matched_device_end = len(req.prefix_indices)
    assert len(req.prefix_indices) == 4
    loaded, req.last_node = cache.init_load_back(
        InitLoadBackParams(
            best_match_node=req.best_match_node,
            host_hit_length=req.host_hit_length,
            req=req,
        )
    )
    assert len(loaded) == 0
    _admit(cache, req)
    assert req.cache_validation_state.start == 3
    ref = cache.capture_verification_attempt(req)
    cache.invalidate_verification_attempt(ref)
    assert cache.tree_core.is_invalidated(node)
    _finish(cache, req)
    _ack_load(cache, node)
    cache.release_verification_attempt(ref)
    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    _check(cache)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_failed_host_load_and_unadmitted_match_own_no_receipts(backend):
    cache = _cache(backend)
    node = _insert(cache, [1, 2, 3, 4])
    assert _match_len(cache, [1, 2]) == 2
    _backup(cache, node)
    cache._demote(node, defaultdict(int))
    cache.cache_controller = _IndexCopyController(cache, succeed=False)
    req = _request(cache, [1, 2, 3, 4, 5], admit=False)
    loaded, _ = cache.init_load_back(
        InitLoadBackParams(
            best_match_node=req.best_match_node,
            host_hit_length=req.host_hit_length,
            req=req,
        )
    )
    assert len(loaded) == 0
    assert req.cache_validation_state.start is None
    assert req.cache_validation_state.host_start is None
    assert not req.cache_validation_state.host_sources
    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    _check(cache)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_admission_then_allocation_abort_closes_receipts(backend):
    cache = _cache(backend, ComponentType.MAMBA)
    source = _request(cache, [1, 2, 3, 4], rid="source")
    _finish(cache, source)
    node = source.last_node
    _backup(cache, node, ComponentType.MAMBA)
    freed = cache.tree_core.evict_component(
        node, ComponentType.MAMBA, EvictLayer.DEVICE
    )
    cache._free_values(freed.device_frees, freed.host_frees)
    cache.cache_controller = _IndexCopyController(cache)
    req = _request(cache, [1, 2, 3, 4, 5], admit=False)
    loaded, req.last_node = cache.init_load_back(
        InitLoadBackParams(
            best_match_node=req.best_match_node,
            host_hit_length=req.host_hit_length,
            req=req,
        )
    )
    req.prefix_indices = torch.cat([req.prefix_indices, loaded])
    req.kv.cache_protected_len = len(req.prefix_indices)
    cache.record_prefix_admission(req)
    assert not req.kv.holds_kv
    assert req.kv.holds_mamba
    assert cache.tree_core.prefix_ref_counts()[0] > 0
    release_kv_cache(req, cache, is_insert=False)
    _ack_load(cache, node)
    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    assert _match_len(cache, [1, 2, 3, 4]) == 4
    _check(cache)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_swa_restore_uses_source_spans_with_resident_holes(backend):
    cache = _cache(backend, ComponentType.SWA)
    tokens = list(range(1, 9))
    node = _insert(cache, tokens)
    # Split one-token spans so the restored SWA pages have live pages between
    # them: [4,5) and [6,7), while [5,6) and [7,8) remain resident.
    boundaries = {}
    for end in range(1, 9):
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens[:end])))
        )
        boundaries[end] = result.last_device_node
    cache.tree_core.has_swa_host_pool = True
    for end in (5, 7):
        _backup(cache, boundaries[end], ComponentType.SWA)
        freed = cache.tree_core.evict_component(
            boundaries[end], ComponentType.SWA, EvictLayer.DEVICE
        )
        cache._free_values(freed.device_frees, freed.host_frees)
    cache.cache_controller = _IndexCopyController(cache)
    req = _request(cache, [*tokens, 9], admit=False)
    req.prefix_indices = cache.tree_core.collect_full_device_indices(
        node, cache.root_node_handle()
    )
    req.last_node = node
    req.kv.cache_protected_len = len(req.prefix_indices)
    req.cache_validation_state.matched_device_end = len(req.prefix_indices)
    loaded, req.last_node = cache.init_load_back(
        InitLoadBackParams(best_match_node=node, host_hit_length=0, req=req)
    )
    assert len(loaded) == 0
    assert sum(len(t.host_indices) for t in cache.cache_controller.calls[0][2]) == 2
    _admit(cache, req)
    assert req.cache_validation_state.start == 4
    ref = cache.capture_verification_attempt(req)
    cache.invalidate_verification_attempt(ref)
    assert _match_len(cache, tokens) == 4
    _finish(cache, req)
    _ack_load(cache, node)
    cache.release_verification_attempt(ref)
    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    _check(cache)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_retracted_receipt_cannot_poison_new_attempt_or_generation(backend):
    cache = _cache(backend)
    req = _request(cache, [1, 2, 3, 4])
    cache.cache_unfinished_req(req)
    old = cache.capture_verification_attempt(req)
    _finish(cache, req, is_insert=False, is_retract=True)
    req.reset_for_retract()
    assert req.cache_validation_state is None
    assert not req.cache_invalid
    cache.invalidate_verification_attempt(old)
    req.init_next_round_input(tree_cache=cache, cow_mamba=False)
    req.set_extend_range(0, len(req.origin_input_ids))
    _admit(cache, req)
    fresh = req.cache_validation_state
    assert fresh is not old and not fresh.invalid
    _finish(cache, req)
    assert _match_len(cache, [1, 2, 3, 4]) == 4
    cache.invalidate_verification_attempt(old)
    assert not req.cache_invalid
    assert _match_len(cache, [1, 2, 3, 4]) == 4
    cache.release_verification_attempt(old)
    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    _check(cache)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_chunked_prefill_receipt_metadata_is_bounded_until_first_verify(backend):
    cache = _cache(backend)
    req = _request(cache, list(range(1, 34)), admit=False)
    req.set_extend_range(0, 2)
    _admit(cache, req)
    for end in range(2, 33, 2):
        if end > 2:
            req.init_next_round_input(tree_cache=cache, cow_mamba=False)
            req.set_extend_range(end - 2, end)
            tail = cache.token_to_kv_pool_allocator.alloc(2)
            cache.req_to_token_pool.write(
                (req.kv.req_pool_idx, slice(end - 2, end)), tail.to(torch.int32)
            )
            req.kv.kv_allocated_len = req.kv.kv_committed_len = end
            cache.record_prefix_admission(req)
        cache.cache_unfinished_req(req, chunked=True)
        assert req.kv.cache_protected_len == end
        assert cache.tree_core.prefix_ref_counts() == (0, 0)
        _check(cache)
    ref = cache.capture_verification_attempt(req)
    assert cache.tree_core.prefix_ref_counts()[0] == 1
    assert cache.tree_core.prefix_ref_counts()[1] <= 32
    # A final token is cached only at finish. Both exact captured paths remain
    # addressable, without one receipt for every earlier prefill chunk.
    tail = cache.token_to_kv_pool_allocator.alloc(1)
    cache.req_to_token_pool.write(
        (req.kv.req_pool_idx, slice(32, 33)), tail.to(torch.int32)
    )
    req.kv.kv_allocated_len = req.kv.kv_committed_len = 33
    _finish(cache, req)
    assert cache.tree_core.prefix_ref_counts()[0] == 2
    assert cache.tree_core.prefix_ref_counts()[1] <= 65
    cache.invalidate_verification_attempt(ref)
    assert _match_len(cache, list(range(1, 34))) == 0
    cache.release_verification_attempt(ref)
    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    _check(cache)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
