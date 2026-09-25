"""Real CPU-tensor KV/SSM backups retain unresolved verification ancestry."""

import sys

import pytest
import torch
from test_cache_verification_lifecycle import (
    _admit,
    _cache,
    _check,
    _finish,
    _match_len,
    _request,
)

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.common import (
    retraction_backup,
    retraction_discard,
    retraction_restore,
)
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.session.session_controller import Session
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

TOKENS = [1, 2, 3, 4]


def _full_pool(cache):
    pool = cache.token_to_kv_pool_allocator.get_kvcache()
    return pool.full_kv_pool if cache.supports_mamba() else pool


def _fill(cache, req, value):
    pool = _full_pool(cache)
    indices = cache.req_to_token_pool.req_to_token[
        req.kv.req_pool_idx, : req.seqlen
    ].long()
    for buffer in (*pool.k_buffer, *pool.v_buffer):
        buffer[indices] = value
    if cache.supports_mamba():
        state = cache.req_to_token_pool.mamba_pool.mamba_cache
        for conv in state.conv:
            conv[:, req.kv.mamba_pool_idx] = value
        state.temporal[:, req.kv.mamba_pool_idx] = value


def _assert_nan_restored(cache, req):
    pool = _full_pool(cache)
    indices = cache.req_to_token_pool.req_to_token[
        req.kv.req_pool_idx, : req.seqlen
    ].long()
    for buffer in (*pool.k_buffer, *pool.v_buffer):
        assert buffer[indices[:-1]].isnan().all()
        assert (buffer[indices[-1:]] == -7).all()
    if cache.supports_mamba():
        state = cache.req_to_token_pool.mamba_pool.mamba_cache
        for conv in state.conv:
            assert conv[:, req.kv.mamba_pool_idx].isnan().all()
        assert state.temporal[:, req.kv.mamba_pool_idx].isnan().all()


def _backup_and_retract(cache, req):
    source = req.cache_validation_state
    assert retraction_backup(
        req,
        cache,
        cache.req_to_token_pool,
        cache.token_to_kv_pool_allocator,
        "cpu_tensor",
    )
    backup = req.kv.retraction_backup
    assert backup.cache_validation_source is source
    _finish(cache, req, is_insert=False, is_retract=True)
    req.reset_for_retract()
    assert req.kv.retraction_backup is backup
    cache.evict(EvictParams(num_tokens=64, mamba_num=64))
    assert _match_len(cache, TOKENS) == 0
    return backup


def _fresh_allocate(cache, req):
    # Decode resume calls _pre_alloc with prefix_len=0. Restored private KV
    # therefore never overwrites a separately matched cache generation.
    req.prefix_indices = torch.empty(0, dtype=torch.int64)
    req.last_node = req.last_host_node = req.best_match_node = cache.root_node_handle()
    req.kv.cache_protected_len = 0
    req._refresh_fill_ids()
    req.set_extend_range(0, len(req.full_untruncated_fill_ids))
    _admit(cache, req)
    req.is_retracted = False
    _fill(cache, req, -7)


def _restore(cache, req):
    _fresh_allocate(cache, req)
    retraction_restore(
        req,
        cache,
        cache.req_to_token_pool,
        cache.token_to_kv_pool_allocator,
        "cpu_tensor",
    )
    assert req.kv.retraction_backup is None
    _assert_nan_restored(cache, req)
    return req.cache_validation_state


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize(
    "auxiliary", [None, ComponentType.MAMBA], ids=["full", "mamba"]
)
@pytest.mark.parametrize(
    "scenario",
    [
        "invalid_before_restore",
        "finished_b",
        "transitive_invalid",
        "transitive_healthy",
        "discard_recompute",
        "finished_session_b",
    ],
)
def test_cpu_backup_contains_restored_generations_without_rewriting_output(
    backend, auxiliary, scenario
):
    cache = _cache(backend, auxiliary)
    initial_mamba = (
        cache.req_to_token_pool.mamba_allocator.available_size()
        if cache.supports_mamba()
        else None
    )
    session = (
        Session(128, "saved", streaming=True)
        if scenario == "finished_session_b"
        else None
    )
    req = _request(cache, TOKENS, session=session)
    _fill(cache, req, float("nan"))
    source = cache.capture_verification_attempt(req)
    backup = _backup_and_retract(cache, req)
    assert source.closed and source.pending == 1
    invalid = scenario != "transitive_healthy"

    if scenario in ("invalid_before_restore", "discard_recompute"):
        cache.invalidate_verification_attempt(source)
        cache.release_verification_attempt(source)
        assert backup.cache_validation_source.invalid
        assert cache.tree_core.prefix_ref_counts() == (0, 0)

    if scenario == "discard_recompute":
        retraction_discard(req, cache, "cpu_tensor")
        _fresh_allocate(cache, req)
        _fill(cache, req, 17)
        assert req.cache_validation_state.upstream_holds == 0
        assert not cache.request_cache_invalid(req)
        _finish(cache, req)
        assert _match_len(cache, TOKENS) == 4
    else:
        child = _restore(cache, req)
        if scenario == "invalid_before_restore":
            assert child.invalid and cache.request_cache_invalid(req)
            assert child.upstream_holds == 0
        else:
            assert child.upstream_holds == 1
            assert not child.verification_started
        if scenario.startswith("transitive_"):
            _backup_and_retract(cache, req)
            assert child.closed and child.pending == 0 and child.upstream_holds == 1
            leaf = _restore(cache, req)
            assert leaf.upstream_holds == 1 and not leaf.verification_started
        else:
            leaf = child
        _finish(cache, req)
        finished = (list(req.output_ids), req.finished_reason, req.finished_len)
        if scenario != "invalid_before_restore":
            assert leaf.closed and leaf.upstream_holds == 1
            if session is None:
                assert _match_len(cache, TOKENS) == 4
                assert cache.tree_core.prefix_ref_counts()[0] > 0
            else:
                assert leaf.session_slot is cache.session.slots[session.session_id]
            if invalid:
                cache.invalidate_verification_attempt(source)
            cache.release_verification_attempt(source)
        assert (list(req.output_ids), req.finished_reason, req.finished_len) == finished
        assert child.upstream_holds == leaf.upstream_holds == 0
        assert not source.dependents and not child.dependents
        if invalid:
            assert leaf.invalid
            assert _match_len(cache, TOKENS) == 0
            if session is not None:
                assert not cache.session.slots and not cache.session.any_holding_kv()
        else:
            assert not leaf.invalid
            assert _match_len(cache, TOKENS) == 4

    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    _check(cache)
    cache.evict(EvictParams(num_tokens=64, mamba_num=64))
    assert cache.token_to_kv_pool_allocator.available_size() == 64
    assert cache.req_to_token_pool.available_size() == 8
    if cache.supports_mamba():
        assert cache.req_to_token_pool.mamba_allocator.available_size() == initial_mamba
    _check(cache)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_unresolved_backup_fault_hides_finished_restored_prefix(backend):
    """End-to-end assertion also serves production omission controls.

    No intermediate receipt-shape assertion can substitute for retiring the
    separately published descendant generation.
    """
    cache = _cache(backend)
    req = _request(cache, TOKENS)
    _fill(cache, req, float("nan"))
    source = cache.capture_verification_attempt(req)
    assert retraction_backup(
        req,
        cache,
        cache.req_to_token_pool,
        cache.token_to_kv_pool_allocator,
        "cpu_tensor",
    )
    _finish(cache, req, is_insert=False, is_retract=True)
    req.reset_for_retract()
    cache.evict(EvictParams(num_tokens=64))
    _restore(cache, req)
    _finish(cache, req)
    assert _match_len(cache, TOKENS) == 4
    finished = (list(req.output_ids), req.finished_reason, req.finished_len)

    cache.invalidate_verification_attempt(source)

    assert _match_len(cache, TOKENS) == 0
    assert (list(req.output_ids), req.finished_reason, req.finished_len) == finished
    cache.release_verification_attempt(source)
    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    cache.evict(EvictParams(num_tokens=64))
    assert cache.token_to_kv_pool_allocator.available_size() == 64
    _check(cache)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
