"""Delayed verification preserves streaming-session ownership on both cores."""

import sys
from array import array

import pytest
import torch
from test_cache_verification_lifecycle import _admit, _cache, _finish, _request

from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Req,
    ReqKvInfo,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.session.session_controller import Session
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.fixture(params=["python", "rust"])
def cache(request):
    return _cache(request.param)


def _free_indices(cache):
    allocator = cache.token_to_kv_pool_allocator
    pages = allocator.free_pages.tolist()
    if allocator.release_pages is not None:
        pages.extend(allocator.release_pages.tolist())
    return sorted(
        page * cache.page_size + offset
        for page in pages
        for offset in range(cache.page_size)
    )


def _capacity(cache):
    return _free_indices(cache), sorted(cache.req_to_token_pool.free_slots)


def _assert_released(cache, initial):
    cache.evict(EvictParams(num_tokens=len(initial[0])))
    assert _capacity(cache) == initial
    assert cache.token_to_kv_pool_allocator.available_size() == len(initial[0])
    assert cache.req_to_token_pool.available_size() == len(initial[1])
    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    cache.tree_core.sanity_check(
        list(cache.ongoing_write_through), list(cache.ongoing_load_back)
    )


def _append_output(cache, req, token):
    """Commit one real decode allocation beyond the published prompt prefix."""
    tail = cache.token_to_kv_pool_allocator.alloc(1)
    assert tail is not None
    pos = req.kv.kv_allocated_len
    cache.req_to_token_pool.write(
        (req.kv.req_pool_idx, slice(pos, pos + 1)), tail.to(torch.int32)
    )
    req.kv.kv_allocated_len += 1
    req.kv.kv_committed_len += 1
    req.output_ids.append(token)
    req._refresh_fill_ids()


def _saved_turn(cache, *, session=None, rid="first", retain=True):
    if session is None:
        session = Session(128, "stream", streaming=True)
    req = _request(cache, [1, 2, 3, 4], rid=rid, session=session)
    assert isinstance(req, Req)
    assert isinstance(req.kv, ReqKvInfo)
    cache.cache_unfinished_req(req)
    assert req.kv.cache_protected_len == 4
    receipt = cache.capture_verification_attempt(req) if retain else None
    _append_output(cache, req, 5)
    _finish(cache, req)
    slot = cache.session.slots[session.session_id]
    assert isinstance(slot.kv, ReqKvInfo)
    assert slot.kv.kv_allocated_len == 5
    assert not req.kv.holds_kv
    if retain:
        assert receipt.closed and receipt.pending == 1
        assert receipt.session_slot is slot
        assert cache.tree_core.prefix_ref_counts()[0] == 1
    return session, req, receipt, slot


def test_late_result_releases_idle_slot_after_source_loses_session(cache):
    initial = _capacity(cache)
    session, source, receipt, slot = _saved_turn(cache)
    successor = _request(cache, [1, 2, 3, 4, 5, 6], rid="next", session=session)
    _finish(cache, successor)
    assert source.session is None
    assert session.req_nodes[successor.rid].req is successor
    assert cache.session.slots[session.session_id] is slot
    row = slot.kv.req_pool_idx
    private = cache.req_to_token_pool.req_to_token[row, 4:6].tolist()
    before = _free_indices(cache)

    cache.invalidate_verification_attempt(receipt)

    assert session.session_id not in cache.session.slots
    assert not slot.kv.holds_kv
    assert cache.req_to_token_pool.free_slots.count(row) == 1
    assert _free_indices(cache) == sorted(before + private)
    assert session.req_nodes[successor.rid].req is successor
    cache.invalidate_verification_attempt(receipt)
    assert _free_indices(cache) == sorted(before + private)
    cache.release_verification_attempt(receipt)
    _assert_released(cache, initial)


def test_late_result_preserves_admitted_reader_until_its_finish(cache):
    initial = _capacity(cache)
    session, _, receipt, slot = _saved_turn(cache)
    successor = _request(cache, [1, 2, 3, 4, 5, 6], rid="next", session=session)
    assert slot.admitted
    assert successor.kv is slot.kv
    row = successor.kv.req_pool_idx
    indices = cache.req_to_token_pool.req_to_token[row, :6].clone()
    before = _capacity(cache)

    cache.invalidate_verification_attempt(receipt)

    assert successor.cache_invalid
    assert slot.invalidated
    assert successor.kv is slot.kv and successor.kv.req_pool_idx == row
    assert _capacity(cache) == before
    assert torch.equal(cache.req_to_token_pool.req_to_token[row, :6], indices)
    cache.cache_unfinished_req(successor)
    assert successor.prefix_indices.tolist() == indices.tolist()
    assert (
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4])))
        ).device_indices.numel()
        == 0
    )

    _finish(cache, successor)
    assert session.session_id not in cache.session.slots
    assert not successor.kv.holds_kv
    assert session.req_nodes[successor.rid].req is successor
    cache.release_verification_attempt(receipt)
    _assert_released(cache, initial)


def test_late_result_detaches_and_rematches_unadmitted_successor(cache):
    initial = _capacity(cache)
    session, _, receipt, slot = _saved_turn(cache)
    queued = _request(
        cache, [1, 2, 3, 4, 5, 6], rid="queued", session=session, admit=False
    )
    assert queued.kv is slot.kv and not slot.admitted
    assert queued.prefix_indices.numel() == 5
    borrowed_kv = queued.kv
    row = borrowed_kv.req_pool_idx
    private = cache.req_to_token_pool.req_to_token[row, 4:5].tolist()
    before = _free_indices(cache)

    cache.invalidate_verification_attempt(receipt)

    assert queued.kv is not borrowed_kv
    assert not queued.kv.holds_kv
    assert not queued.cache_invalid
    assert queued.prefix_indices.numel() == 0
    assert queued.kv.cache_protected_len == 0
    assert queued.cache_validation_state.start is None
    assert queued.cache_validation_state.matched_device_end == 0
    assert session.session_id not in cache.session.slots
    assert cache.req_to_token_pool.free_slots.count(row) == 1
    assert _free_indices(cache) == sorted(before + private)

    _admit(cache, queued)
    assert queued.cache_validation_state.start == 0
    _finish(cache, queued)
    assert not cache.session.slots[session.session_id].invalidated
    cache.release_verification_attempt(receipt)
    cache.release_session(session.session_id)
    _assert_released(cache, initial)


def test_old_receipt_does_not_release_fresh_slot_with_same_session_id(cache):
    initial = _capacity(cache)
    session, _, receipt, old_slot = _saved_turn(cache)
    cache.release_session(session.session_id)
    cache.evict(EvictParams(num_tokens=len(initial[0])))
    fresh_session = Session(128, session.session_id, streaming=True)
    _, fresh, _, fresh_slot = _saved_turn(
        cache, session=fresh_session, rid="fresh", retain=False
    )
    assert fresh_slot is not old_slot
    before = _capacity(cache)
    row = fresh_slot.kv.req_pool_idx
    indices = cache.req_to_token_pool.req_to_token[row, :5].clone()

    cache.invalidate_verification_attempt(receipt)

    assert cache.session.slots[session.session_id] is fresh_slot
    assert not fresh_slot.invalidated
    assert fresh_slot.kv.req_pool_idx == row
    assert torch.equal(cache.req_to_token_pool.req_to_token[row, :5], indices)
    assert _capacity(cache) == before
    assert fresh_session.req_nodes[fresh.rid].req is fresh
    assert (
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4])))
        ).device_indices.numel()
        == 4
    )
    cache.release_verification_attempt(receipt)
    cache.release_session(session.session_id)
    _assert_released(cache, initial)


def test_queue_abort_detaches_borrower_and_preserves_saved_slot(cache):
    initial = _capacity(cache)
    session, source, receipt, slot = _saved_turn(cache)
    queued = _request(
        cache, [1, 2, 3, 4, 5, 6], rid="queued", session=session, admit=False
    )
    assert queued.kv is slot.kv
    before = _capacity(cache)
    row = slot.kv.req_pool_idx
    indices = cache.req_to_token_pool.req_to_token[row, :5].clone()
    queued.to_finish = FINISH_ABORT("cancelled before admission")

    queued.init_next_round_input(tree_cache=cache, cow_mamba=False)

    assert queued.session is None
    assert not queued.kv.holds_kv
    assert queued.kv is not slot.kv
    assert cache.session.slots[session.session_id] is slot
    assert slot.matched_req is None and not slot.admitted
    assert slot.kv.req_pool_idx == row
    assert torch.equal(cache.req_to_token_pool.req_to_token[row, :5], indices)
    assert _capacity(cache) == before
    assert session.req_nodes[source.rid].req is source
    assert not session._inflight
    cache.release_verification_attempt(receipt)
    cache.release_session(session.session_id)
    _assert_released(cache, initial)


def test_invalid_natural_finish_keeps_response_checkpoint_without_saved_kv(cache):
    initial = _capacity(cache)
    session = Session(128, "stream", streaming=True)
    req = _request(cache, [1, 2, 3, 4], session=session)
    cache.cache_unfinished_req(req)
    receipt = cache.capture_verification_attempt(req)
    _append_output(cache, req, 5)
    _append_output(cache, req, 6)
    req.finished_reason = FINISH_LENGTH(1)
    req.finished_len = 1

    cache.invalidate_verification_attempt(receipt)
    _finish(cache, req)

    assert req.finished() and not isinstance(req.finished_reason, FINISH_ABORT)
    assert req.output_ids.tolist() == [5]
    assert session.req_nodes[req.rid].req is req
    assert session.committed_origin_len == 4
    assert session.committed_fill_len == len(req.full_untruncated_fill_ids)
    assert not session._inflight
    assert session.session_id not in cache.session.slots
    assert not req.kv.holds_kv
    cache.release_verification_attempt(receipt)
    _assert_released(cache, initial)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_unaligned_session_prefix_invalidates_at_page_boundary(backend):
    cache = _cache(backend, page_size=2)
    assert isinstance(cache.token_to_kv_pool_allocator, PagedTokenToKVPoolAllocator)
    initial = _capacity(cache)
    session = Session(128, "stream", streaming=True)
    source = _request(cache, [1, 2, 3, 4, 5], rid="source", session=session)
    cache.cache_unfinished_req(source)
    assert source.kv.cache_protected_len == 4
    _finish(cache, source)
    slot = cache.session.slots[session.session_id]
    assert slot.kv.kv_committed_len == 5
    before_admission = _capacity(cache)

    successor = _request(cache, [1, 2, 3, 4, 5, 6], rid="next", session=session)

    assert successor.prefix_indices.numel() == 5
    assert successor.kv.cache_protected_len == 4
    assert successor.cache_validation_state.matched_device_end == 5
    assert successor.cache_validation_state.start == 4
    # The sixth token occupies the saved partial page, so admission allocates
    # neither a new page nor a new request row.
    assert _capacity(cache) == before_admission
    receipt = cache.capture_verification_attempt(successor)
    assert cache.tree_core.prefix_ref_counts()[0] == 1
    _finish(cache, successor)
    assert receipt.closed and receipt.pending == 1
    row = slot.kv.req_pool_idx
    private_page = cache.req_to_token_pool.req_to_token[row, 4:6].tolist()
    assert private_page[0] // 2 == private_page[1] // 2
    before = _free_indices(cache)

    cache.invalidate_verification_attempt(receipt)

    assert session.session_id not in cache.session.slots
    assert cache.req_to_token_pool.free_slots.count(row) == 1
    assert _free_indices(cache) == sorted(before + private_page)
    assert (
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4, 5, 6])))
        ).device_indices.numel()
        == 4
    )
    cache.release_verification_attempt(receipt)
    _assert_released(cache, initial)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
