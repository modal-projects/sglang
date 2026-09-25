"""Causal CPU check of retirement provenance crossing TreeCore and pool release."""

import unittest
from array import array
from collections import defaultdict
from unittest.mock import patch

import test_unified_eviction_causes as metrics_tests
import test_unified_radix_cache_unittest as shared_fixture
import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import MambaEvictExcessPathStates
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

FULL = ComponentType.FULL
MAMBA = ComponentType.MAMBA


def cache_for(backend, components=(FULL,)):
    helper = metrics_tests.TestUnifiedEvictionCauses()
    with patch.object(shared_fixture, "_TREE_CORE_TEST_BACKEND", backend):
        cache, registry = helper.make_cache(components)
    return helper, cache, registry


def _check_retired_and_healthy_frees_have_distinct_causes_in_one_pressure_walk(backend):
    helper, cache, registry = cache_for(backend, (FULL, MAMBA))
    old = helper.insert(cache, [1, 2, 3, 4])
    receipt = cache.tree_core.capture_prefix_ref(old, 4)
    cache.tree_core.invalidate_prefix_ref(receipt, 0)
    replacement = helper.insert(cache, [1, 2, 3, 4])
    lock = cache.inc_lock_ref(replacement)
    healthy = helper.insert(cache, [9, 10, 11, 12])
    available = cache.token_to_kv_pool_allocator.available_size()
    freed = cache.evict(EvictParams(num_tokens=8))
    assert freed.num_tokens_evicted == 8
    assert freed.mamba_num_evicted == 2
    assert cache.token_to_kv_pool_allocator.available_size() == available + 8
    assert not cache.tree_core.contains_node(old)
    assert not cache.tree_core.contains_node(healthy)
    assert cache.tree_core.contains_node(replacement)
    assert helper.causes(registry) == {
        ("full", "device", "other"): 4,
        ("mamba", "device", "other"): 1,
        ("full", "device", "full_pressure"): 4,
        ("mamba", "device", "full_pressure"): 1,
    }
    cache.dec_lock_ref(replacement, lock.to_dec_params())
    cache.tree_core.release_prefix_ref(receipt)


def _check_retired_host_copy_keeps_other_after_device_demotion(backend):
    helper, cache, registry = cache_for(backend)
    core = cache.tree_core
    core.set_hicache_enabled()
    old = helper.insert(cache, [1, 2, 3, 4])
    healthy = helper.insert(cache, [9, 10, 11, 12])
    for node, start in ((old, 100), (healthy, 200)):
        core.mark_write_through_pending([node], node)
        core.commit_backup(node, torch.arange(start, start + 4), {})
        core.finish_write_through([node], node)
    released = helper.attach_host_pool(cache)
    receipt = core.capture_prefix_ref(old, 4)
    core.invalidate_prefix_ref(receipt, 0)
    assert cache.evict(EvictParams(num_tokens=8)).num_tokens_evicted == 8
    assert released == []
    assert cache.evict_host(8) == 8
    assert sorted(released) == [100, 101, 102, 103, 200, 201, 202, 203]
    assert helper.causes(registry) == {
        ("full", "device", "other"): 4,
        ("full", "device", "full_pressure"): 4,
        ("full", "host", "other"): 4,
        ("full", "host", "host_pressure"): 4,
    }
    core.release_prefix_ref(receipt)


def _insert_with_deferred_path_cap(cache, tokens):
    """Hold the emitted path-cap action until the prefix has been retired."""
    value = cache.token_to_kv_pool_allocator.alloc(len(tokens))
    mamba_value = cache.req_to_token_pool.mamba_allocator.alloc(1)
    assert value is not None and mamba_value is not None
    step = cache.tree_core.begin_insert(
        InsertParams(
            key=RadixKey(array("q", tokens)), value=value, mamba_value=mamba_value
        )
    )
    while True:
        for action in step.actions:
            if not isinstance(action, MambaEvictExcessPathStates):
                cache._apply_cache_action(action)
        if step.result is not None:
            break
        step = cache.tree_core.resume_insert()
    for action in cache.tree_core.end_insert():
        if not isinstance(action, MambaEvictExcessPathStates):
            cache._apply_cache_action(action)
    return step.result.last_device_node


def _check_retired_override_survives_native_append_offsets(backend):
    # A result may append to a caller-owned list; the pre-existing healthy free
    # must retain its operation cause when the retired entry receives an offset.
    original_args = shared_fixture.ServerArgs

    def capped_args(**kwargs):
        return original_args(**kwargs, mamba_max_states_per_path=1)

    with patch.object(shared_fixture, "ServerArgs", side_effect=capped_args):
        helper, cache, registry = cache_for(backend, (FULL, MAMBA))
    _insert_with_deferred_path_cap(cache, [1, 2])
    tail = _insert_with_deferred_path_cap(cache, [1, 2, 3, 4])
    receipt = cache.tree_core.capture_prefix_ref(tail, 4)
    cache.tree_core.invalidate_prefix_ref(receipt, 0)
    extra_slot = cache.req_to_token_pool.mamba_allocator.alloc(1)
    assert extra_slot is not None
    device = defaultdict(list, {MAMBA: [extra_slot]})
    host = defaultdict(list)
    causes = []
    cache.tree_core.evict_excess_path_states(tail, device, host, causes)
    assert len(device[MAMBA]) == 2
    assert len(causes) == 1
    assert (causes[0].component_type, causes[0].tier, causes[0].index) == (
        MAMBA,
        "device",
        1,
    )
    cache._free_values(device, host, cause="mamba_path_cap", free_causes=causes)
    assert helper.causes(registry) == {
        ("mamba", "device", "mamba_path_cap"): 1,
        ("mamba", "device", "other"): 1,
    }
    cache.tree_core.release_prefix_ref(receipt)


class TestRetirementEvictionCauses(CustomTestCase):
    def test_retired_and_healthy_frees_have_distinct_causes_in_one_pressure_walk_python(
        self,
    ):
        _check_retired_and_healthy_frees_have_distinct_causes_in_one_pressure_walk(
            "python"
        )

    def test_retired_and_healthy_frees_have_distinct_causes_in_one_pressure_walk_rust(
        self,
    ):
        _check_retired_and_healthy_frees_have_distinct_causes_in_one_pressure_walk(
            "rust"
        )

    def test_retired_host_copy_keeps_other_after_device_demotion_python(self):
        _check_retired_host_copy_keeps_other_after_device_demotion("python")

    def test_retired_host_copy_keeps_other_after_device_demotion_rust(self):
        _check_retired_host_copy_keeps_other_after_device_demotion("rust")

    def test_retired_override_survives_native_append_offsets_python(self):
        _check_retired_override_survives_native_append_offsets("python")

    def test_retired_override_survives_native_append_offsets_rust(self):
        _check_retired_override_survives_native_append_offsets("rust")


if __name__ == "__main__":
    unittest.main()
