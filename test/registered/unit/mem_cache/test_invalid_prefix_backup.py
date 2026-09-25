"""Host-backup lineage through the real common retraction wrappers.

Requests, both tree cores, FULL/SWA pools, host allocation and hybrid transfer
resolution are real. The L2 transport executes synchronous CPU tensor copies;
these tests do not validate GPU transfer kernels or discover a NaN producer.
"""

import sys

import pytest
import torch
from test_cache_verification_lifecycle import (
    _admit,
    _cache,
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
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

TOKENS = [1, 2, 3, 4]


class _CpuCopyCompletion:
    def __init__(self):
        self.finish_event = self
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1
        assert self.synchronizations == 1


class _CpuCopyTransport:
    """Only the async device-transfer boundary is replaced with CPU copies."""

    def __init__(self):
        self.completions = []

    def _copy(self, transfers, *, to_host):
        for transfer in transfers:
            host = transfer.host_pool
            device = transfer.device_pool
            host_indices = transfer.host_indices.long()
            device_indices = transfer.device_indices.long()
            for layer in range(device.layer_num):
                for host_buffer, device_buffer in (
                    (host.k_buffer[layer], device.k_buffer[layer]),
                    (host.v_buffer[layer], device.v_buffer[layer]),
                ):
                    if to_host:
                        host_buffer[host_indices] = device_buffer[device_indices]
                    else:
                        device_buffer[device_indices] = host_buffer[host_indices]
        completion = _CpuCopyCompletion()
        self.completions.append(completion)
        return completion

    def submit_device_to_host(self, transfers):
        return self._copy(transfers, to_host=True)

    def submit_host_to_device(self, transfers, *, layer_num):
        assert layer_num > 0
        return self._copy(transfers, to_host=False)


class _CpuTransferController(HybridCacheController):
    """Wire real transfer-resolution methods without CUDA stream bootstrap."""

    def __init__(self, group, layer_num):
        self.mem_pool_host = group
        self.io_backend = "direct"
        self.write_policy = "write_through_selective"
        self.device = "cpu"
        self.layer_num = layer_num
        self.l2_transfer_engine = _CpuCopyTransport()


@pytest.fixture(params=["python", "rust"])
def backend(request):
    return request.param


@pytest.fixture(params=[None, ComponentType.SWA], ids=["full", "swa"])
def cache(request, backend):
    cache = _cache(backend, auxiliary=request.param)
    kv_pool = cache.token_to_kv_pool_allocator.get_kvcache()
    device_pools = (
        [(PoolName.KV, kv_pool.full_kv_pool), (PoolName.SWA, kv_pool.swa_kv_pool)]
        if request.param == ComponentType.SWA
        else [(PoolName.KV, kv_pool)]
    )
    entries = []
    for name, device_pool in device_pools:
        host_pool = MHATokenToKVPoolHost(
            device_pool=device_pool,
            host_to_device_ratio=1.0,
            host_size=0,
            page_size=1,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
        )
        entries.append(
            PoolEntry(
                name=name,
                host_pool=host_pool,
                device_pool=device_pool,
                layer_mapper=lambda layer: layer,
                is_primary_index_anchor=name == PoolName.KV,
            )
        )
    group = HostPoolGroup(entries)
    cache.host_pool_group = group
    cache.cache_controller = _CpuTransferController(group, kv_pool.layer_num)
    assert cache.supports_retraction_backup()
    yield cache
    group.destroy()


def _host_free(cache):
    return {
        entry.name: sorted(
            torch.cat(
                [entry.host_pool.free_slots, *entry.host_pool.release_slots]
            ).tolist()
        )
        for entry in cache.host_pool_group.entries
    }


def _device_free(cache):
    allocator = cache.token_to_kv_pool_allocator
    pools = (
        [allocator.full_attn_allocator, allocator.swa_attn_allocator]
        if cache.supports_swa()
        else [allocator]
    )
    return [
        sorted(torch.cat([pool.free_pages, pool.release_pages]).tolist())
        for pool in pools
    ], sorted(cache.req_to_token_pool.free_slots)


def _live_buffers(cache, req):
    allocator = cache.token_to_kv_pool_allocator
    pool = allocator.get_kvcache()
    indices = cache.req_to_token_pool.req_to_token[
        req.kv.req_pool_idx, : req.seqlen
    ].long()
    if cache.supports_swa():
        yield pool.full_kv_pool, indices
        yield pool.swa_kv_pool, pool.translate_loc_from_full_to_swa(indices)
    else:
        yield pool, indices


def _fill(cache, req, value, *, copied_prefix=False):
    for pool, indices in _live_buffers(cache, req):
        if copied_prefix:
            indices = indices[:-1]
        for buffer in (*pool.k_buffer, *pool.v_buffer):
            buffer[indices] = value


def _assert_nan_restored(cache, req):
    for pool, indices in _live_buffers(cache, req):
        for buffer in (*pool.k_buffer, *pool.v_buffer):
            assert buffer[indices[:-1]].isnan().all()
            assert (buffer[indices[-1:]] == -7).all()


def _backup_and_retract(cache, req):
    host_before = _host_free(cache)
    source = req.cache_validation_state
    assert retraction_backup(
        req,
        cache,
        cache.req_to_token_pool,
        cache.token_to_kv_pool_allocator,
        "host_pool",
    )
    backup = req.kv.retraction_backup
    assert backup.cache_validation_source is source
    leases = {PoolName.KV: backup.host_indices}
    leases.update(
        {item.name: item.host_indices for item in backup.pool_transfers or []}
    )
    assert set(leases) == set(host_before)
    for name, indices in leases.items():
        host = cache.host_pool_group.get_pool(name)
        assert host.slot_used[indices].all()
        assert host.available_size() == len(host_before[name]) - (req.seqlen - 1)
    _finish(cache, req, is_insert=False, is_retract=True)
    req.reset_for_retract()
    assert req.kv.retraction_backup is backup
    cache.evict(EvictParams(num_tokens=64, swa_num_tokens=64))
    assert _match_len(cache, list(req.origin_input_ids)) == 0
    return backup, host_before


def _readmit(cache, req):
    req.init_next_round_input(tree_cache=cache, cow_mamba=False)
    req.set_extend_range(0, len(req.origin_input_ids))
    _admit(cache, req)
    _fill(cache, req, -7)


def _restore(cache, req, host_before):
    _readmit(cache, req)
    retraction_restore(
        req,
        cache,
        cache.req_to_token_pool,
        cache.token_to_kv_pool_allocator,
        "host_pool",
    )
    assert req.kv.retraction_backup is None
    assert _host_free(cache) == host_before
    for entry in cache.host_pool_group.entries:
        assert not entry.host_pool.slot_used.any()
    # A repeated scheduler cleanup after restore must not release the lease twice.
    retraction_discard(req, cache, "host_pool")
    assert _host_free(cache) == host_before
    _assert_nan_restored(cache, req)


def _assert_drained(cache, initial):
    cache.evict(EvictParams(num_tokens=64, swa_num_tokens=64))
    assert _device_free(cache) == initial
    assert cache.tree_core.prefix_ref_counts() == (0, 0)
    for entry in cache.host_pool_group.entries:
        assert not entry.host_pool.slot_used.any()
        assert entry.host_pool.available_size() == entry.host_pool.logical_size
    assert all(
        item.synchronizations == 1
        for item in cache.cache_controller.l2_transfer_engine.completions
    )
    cache.tree_core.sanity_check(
        list(cache.ongoing_write_through), list(cache.ongoing_load_back)
    )


@pytest.mark.parametrize("restores", [1, 2], ids=["a_to_b", "a_to_b_to_c"])
def test_host_restore_lineage_survives_finished_descendants(cache, restores):
    initial = _device_free(cache)
    req = _request(cache, TOKENS)
    cache.cache_unfinished_req(req)
    source = cache.capture_verification_attempt(req)
    _fill(cache, req, float("nan"), copied_prefix=True)
    descendants = []

    for _ in range(restores):
        _, host_before = _backup_and_retract(cache, req)
        _restore(cache, req, host_before)
        descendant = req.cache_validation_state
        assert descendant.upstream_holds == 1
        descendants.append(descendant)

    _finish(cache, req)
    assert _match_len(cache, TOKENS) == len(TOKENS)
    assert all(state.closed for state in descendants)
    assert cache.tree_core.prefix_ref_counts()[0] > 0

    cache.invalidate_verification_attempt(source)

    assert _match_len(cache, TOKENS) == 0
    assert all(state.invalid for state in descendants)
    cache.release_verification_attempt(source)
    assert all(state.upstream_holds == 0 for state in descendants)
    _assert_drained(cache, initial)


def test_host_restore_after_source_invalid_is_tainted_without_sanitizing(cache):
    initial = _device_free(cache)
    req = _request(cache, TOKENS)
    cache.cache_unfinished_req(req)
    source = cache.capture_verification_attempt(req)
    _fill(cache, req, float("nan"), copied_prefix=True)
    backup, host_before = _backup_and_retract(cache, req)
    cache.invalidate_verification_attempt(source)
    cache.release_verification_attempt(source)
    assert backup.cache_validation_source.invalid

    _restore(cache, req, host_before)

    assert req.cache_invalid and cache.request_cache_invalid(req)
    _finish(cache, req)
    assert _match_len(cache, TOKENS) == 0
    _assert_drained(cache, initial)


def test_discarded_host_backup_does_not_taint_fresh_recomputation(cache):
    initial = _device_free(cache)
    req = _request(cache, TOKENS)
    cache.cache_unfinished_req(req)
    source = cache.capture_verification_attempt(req)
    _fill(cache, req, float("nan"), copied_prefix=True)
    _, host_before = _backup_and_retract(cache, req)

    retraction_discard(req, cache, "host_pool")
    retraction_discard(req, cache, "host_pool")

    assert req.kv.retraction_backup is None
    assert _host_free(cache) == host_before
    _readmit(cache, req)
    _fill(cache, req, 17)
    fresh = req.cache_validation_state
    assert fresh.upstream_holds == 0
    _finish(cache, req)
    cache.invalidate_verification_attempt(source)
    assert not cache.request_cache_invalid(req)
    assert _match_len(cache, TOKENS) == len(TOKENS)
    cache.release_verification_attempt(source)
    _assert_drained(cache, initial)


def test_backup_of_invalidated_reader_carries_taint_across_reset(cache):
    initial = _device_free(cache)
    producer = _request(cache, TOKENS, rid="producer")
    source = cache.capture_verification_attempt(producer)
    _finish(cache, producer)
    reader = _request(cache, [*TOKENS, 5], rid="reader")
    _fill(cache, reader, float("nan"), copied_prefix=True)
    assert not reader.cache_invalid
    assert not reader.cache_validation_state.invalid
    cache.invalidate_verification_attempt(source)

    backup, host_before = _backup_and_retract(cache, reader)

    assert backup.cache_validation_source.invalid
    assert not reader.cache_invalid  # reset cleared the per-attempt flag
    _restore(cache, reader, host_before)
    assert reader.cache_invalid and cache.request_cache_invalid(reader)
    _finish(cache, reader)
    assert _match_len(cache, [*TOKENS, 5]) == 0
    cache.release_verification_attempt(source)
    _assert_drained(cache, initial)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
