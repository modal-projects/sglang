"""Prefix retirement preserves old owners while admitting fresh generations.

Both backends run the same topology and lifetime interleavings on CPU. The Rust
arm loads the compiled extension and must not silently become skipped coverage.
"""

import subprocess
import sys
from array import array
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from unified_tree_core_inspector import UnifiedTreeCoreInspector

import sglang
from sglang.srt.disaggregation.kv_events import BlockRemoved, StorageMedium
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.umbp.umbp_store import KVEventsSubscriber
from sglang.srt.mem_cache.unified_cache.cache_action import (
    ReplaceWriteThroughOnNodeSplit,
    SWARebuild,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import EvictLayer
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

FULL = ComponentType.FULL
MAMBA = ComponentType.MAMBA
SWA = ComponentType.SWA


def _core(backend, *, page_size=1, is_eagle=False, auxiliary=None, enable_events=False):
    req_pool = allocator = None
    if backend == "python" and auxiliary == MAMBA:
        from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=8,
            n_groups=1,
            num_heads=1,
            head_dim=8,
            state_size=2,
            conv_kernel=2,
        )
        req_pool = HybridReqToTokenPool(
            size=2,
            mamba_size=16,
            mamba_spec_state_size=2,
            max_context_len=32,
            device="cpu",
            enable_memory_saver=False,
            cache_params=Mamba2CacheParams(shape=shape, layers=[0]),
            mamba_layer_ids=[0],
            enable_mamba_extra_buffer=False,
        )
    if backend == "python" and auxiliary == SWA:
        from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        pool = SWAKVPool(
            size=64,
            size_swa=64,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=8,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            device="cpu",
        )
        allocator = SWATokenToKVPoolAllocator(
            size=64,
            size_swa=64,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=page_size,
        is_eagle=is_eagle,
        tree_components=(FULL,) if auxiliary is None else (FULL, auxiliary),
        sliding_window_size=4 if auxiliary == SWA else None,
        enable_kv_cache_events=enable_events,
    )
    with get_context().override_server_args(
        _mamba_cache_chunk_size=2,
        mamba_max_states_per_path=-1,
    ):
        if backend == "rust":
            from rust_unified_tree_core_inspector import RustUnifiedTreeCoreInspector

            return RustUnifiedTreeCoreInspector(params)
        cache = SimpleNamespace(
            enable_session_radix_cache=False, token_to_kv_pool_allocator=allocator
        )
        components = {FULL: FullComponent(cache, params)}
        if auxiliary == MAMBA:
            from sglang.srt.mem_cache.unified_cache.components.mamba import (
                MambaComponent,
            )

            components[MAMBA] = MambaComponent(cache, params)
        elif auxiliary == SWA:
            from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent

            components[SWA] = SWAComponent(cache, params)
        return UnifiedTreeCoreInspector(params, components)


@pytest.fixture(params=["python", "rust"])
def backend(request):
    return request.param


def _key(tokens, *, extra_key=None, cache_salt=None):
    return RadixKey(array("q", tokens), extra_key=extra_key, cache_salt=cache_salt)


def _insert(core, tokens, values, *, prev_prefix_len=0, mamba_slot=None, **namespace):
    step = core.begin_insert(
        InsertParams(
            key=_key(tokens, **namespace),
            value=torch.tensor(values, dtype=torch.int64),
            prev_prefix_len=prev_prefix_len,
            mamba_value=None if mamba_slot is None else torch.tensor([mamba_slot]),
        )
    )
    actions = list(step.actions)
    _apply_component_actions(core, step.actions)
    while step.result is None:
        step = core.resume_insert()
        actions.extend(step.actions)
        _apply_component_actions(core, step.actions)
    actions.extend(core.end_insert())
    return step.result.last_device_node, actions


def _apply_component_actions(core, actions):
    for action in actions:
        if isinstance(action, SWARebuild):
            # A recording allocator maps synthetic FULL slots to separate SWA
            # slots; publication still uses the real component/core operation.
            core.set_component_device_value(
                action.node_id, SWA, action.source_value + 100
            )


def _match(core, tokens, **namespace):
    return core.match_prefix(MatchPrefixParams(key=_key(tokens, **namespace)))


def _drain(step):
    device = {
        ct: [t.tolist() for t in values] for ct, values in step.device_frees.items()
    }
    host = {ct: [t.tolist() for t in values] for ct, values in step.host_frees.items()}
    step.device_frees.clear()
    step.host_frees.clear()
    return device, host


def _check(core, *, write=(), load=()):
    core.sanity_check(list(write), list(load))


class _BlockAdvertisements:
    def __init__(self):
        self.blocks = {StorageMedium.GPU: set(), StorageMedium.CPU: set()}
        module = ModuleType("mori.umbp")
        module.UMBPTierType = SimpleNamespace(
            HBM=StorageMedium.GPU, DRAM=StorageMedium.CPU
        )
        package = ModuleType("mori")
        package.umbp = module
        # Only the transport client and its tier enum are replaced; the actual
        # subscriber handles every event without starting a network receiver.
        with patch.dict(sys.modules, {"mori": package, "mori.umbp": module}):
            self.subscriber = KVEventsSubscriber(self)

    def report_external_kv_blocks(self, hashes, tier):
        self.blocks[tier].update(hashes)
        return True

    def revoke_external_kv_blocks(self, hashes, tier):
        self.blocks[tier].difference_update(hashes)
        return True

    def revoke_all_external_kv_blocks_at_tier(self, tier):
        self.blocks[tier].clear()
        return True

    def receive(self, core):
        events = core.take_events()
        for event in events:
            self.subscriber.on_event(event, batch_ts=0.0, attn_dp_rank=0)
        return events


@pytest.mark.parametrize("salt", [None, "tenant-a"])
def test_retired_eviction_events_preserve_replacement_advertisements(backend, salt):
    core = _core(backend, page_size=2, enable_events=True)
    subscriber = _BlockAdvertisements()
    old, _ = _insert(core, [1, 2, 3, 4], [10, 11, 12, 13], cache_salt=salt)
    subscriber.receive(core)
    hashes = subscriber.blocks[StorageMedium.GPU].copy()
    states = [hashes]
    receipt = core.capture_prefix_ref(old, 4)
    lock = core.inc_lock_ref(old)
    core.invalidate_prefix_ref(receipt, 0)
    subscriber.receive(core)
    states.append(subscriber.blocks[StorageMedium.GPU].copy())
    assert core.protected_size() == 4
    _check(core)

    replacement, _ = _insert(core, [1, 2, 3, 4], [20, 21, 22, 23], cache_salt=salt)
    subscriber.receive(core)
    states.append(subscriber.blocks[StorageMedium.GPU].copy())
    core.dec_lock_ref(old, lock.to_dec_params())
    _drain(core.evict_device_leaf(old, False))
    subscriber.receive(core)
    states.append(subscriber.blocks[StorageMedium.GPU].copy())
    assert states[-1] == hashes
    assert _match(core, [1, 2, 3, 4], cache_salt=salt).best_match_node == replacement
    _check(core)

    _drain(core.evict_device_leaf(replacement, False))
    subscriber.receive(core)
    states.append(subscriber.blocks[StorageMedium.GPU].copy())
    assert states == [hashes, set(), hashes, hashes, set()]
    core.release_prefix_ref(receipt)
    _check(core)


def test_split_backup_ack_publishes_only_the_live_prefix(backend):
    core = _core(backend, page_size=2, enable_events=True)
    core.set_hicache_enabled()
    subscriber = _BlockAdvertisements()
    old, _ = _insert(core, [1, 2, 3, 4], [10, 11, 12, 13])
    initial = subscriber.receive(core)
    prefix_hash, suffix_hash = map(str, initial[0].block_hashes)
    receipt = core.capture_prefix_ref(old, 4)
    lock = core.inc_lock_ref(old)
    core.mark_write_through_pending([old], old)
    core.commit_backup(old, torch.tensor([100, 101, 102, 103]), {})
    actions = core.invalidate_prefix_ref(receipt, 2)
    prefix = actions[0].new_node_id
    withdrawn = subscriber.receive(core)
    assert len(withdrawn) == 1 and isinstance(withdrawn[0], BlockRemoved)
    assert withdrawn[0].medium == StorageMedium.GPU
    assert subscriber.blocks[StorageMedium.GPU] == {prefix_hash}
    assert subscriber.blocks[StorageMedium.CPU] == set()
    _check(core, write=[(old, prefix), (old, old)])

    replacement, _ = _insert(core, [1, 2, 3, 4], [10, 11, 20, 21], prev_prefix_len=2)
    subscriber.receive(core)
    core.finish_write_through([prefix, old], old)
    subscriber.receive(core)
    assert subscriber.blocks[StorageMedium.CPU] == {prefix_hash}
    assert subscriber.blocks[StorageMedium.GPU] == {prefix_hash, suffix_hash}
    _check(core)

    replacement_lock = core.inc_lock_ref(replacement)
    core.mark_write_through_pending([replacement], replacement)
    core.commit_backup(replacement, torch.tensor([200, 201]), {})
    core.finish_write_through([replacement], replacement)
    core.dec_lock_ref(replacement, replacement_lock.to_dec_params())
    subscriber.receive(core)
    assert subscriber.blocks[StorageMedium.CPU] == {prefix_hash, suffix_hash}
    core.dec_lock_ref(old, lock.to_dec_params())
    _drain(core.demote(old))
    _drain(core.drive_host_eviction(component_type=FULL, num_tokens=2))
    assert subscriber.receive(core) == []
    assert subscriber.blocks[StorageMedium.CPU] == {prefix_hash, suffix_hash}
    assert _match(core, [1, 2, 3, 4]).best_match_node == replacement
    _check(core)

    _drain(core.demote(replacement))
    _drain(core.drive_host_eviction(component_type=FULL, num_tokens=2))
    subscriber.receive(core)
    assert subscriber.blocks == {
        StorageMedium.GPU: {prefix_hash},
        StorageMedium.CPU: {prefix_hash},
    }
    core.release_prefix_ref(receipt)
    _check(core)


def test_retired_load_back_cannot_publish_over_a_replacement(backend):
    core = _core(backend, page_size=2, enable_events=True)
    core.set_hicache_enabled()
    core.is_write_back = True
    subscriber = _BlockAdvertisements()
    old, _ = _insert(core, [1, 2, 3, 4], [10, 11, 12, 13])
    core.mark_write_through_pending([old], old)
    core.commit_backup(old, torch.tensor([100, 101, 102, 103]), {})
    core.finish_write_through([old], old)
    subscriber.receive(core)
    hashes = subscriber.blocks[StorageMedium.CPU].copy()
    _drain(core.demote(old))
    subscriber.receive(core)
    assert subscriber.blocks[StorageMedium.GPU] == set()
    kv, auxiliary = core.build_load_back_spec(old)
    receipt = core.capture_prefix_ref(old, 4)
    core.invalidate_prefix_ref(receipt, 0)
    subscriber.receive(core)
    assert subscriber.blocks[StorageMedium.CPU] == set()

    replacement, _ = _insert(core, [1, 2, 3, 4], [20, 21, 22, 23])
    subscriber.receive(core)
    core.commit_load_back(old, torch.tensor([30, 31, 32, 33]), kv, auxiliary)
    lock = core.inc_lock_ref(old)
    assert subscriber.receive(core) == []
    _check(core, load=[(old, old)])
    core.finish_load_back(old)
    core.dec_lock_ref(old, lock.to_dec_params())
    _drain(core.demote(old))
    assert subscriber.receive(core) == []
    assert subscriber.blocks[StorageMedium.GPU] == hashes
    assert _match(core, [1, 2, 3, 4]).best_match_node == replacement
    dropped = core.drop_subtree_no_host(replacement)
    assert dropped.is_dropped
    _drain(dropped)
    subscriber.receive(core)
    assert subscriber.blocks[StorageMedium.GPU] == set()
    core.release_prefix_ref(receipt)
    _check(core)


def test_split_survivor_retirement_withdraws_both_tier_advertisements(backend):
    core = _core(backend, page_size=2, enable_events=True)
    core.set_hicache_enabled()
    subscriber = _BlockAdvertisements()
    old, _ = _insert(core, [1, 2, 3, 4], [10, 11, 12, 13])
    core.mark_write_through_pending([old], old)
    core.commit_backup(old, torch.tensor([100, 101, 102, 103]), {})
    core.finish_write_through([old], old)
    subscriber.receive(core)
    hashes = subscriber.blocks[StorageMedium.GPU].copy()
    receipt = core.capture_prefix_ref(old, 4)
    prefix = _match(core, [1, 2]).best_match_node
    _drain(core.demote(old))
    _drain(core.drive_host_eviction(component_type=FULL, num_tokens=2))
    subscriber.receive(core)
    assert len(subscriber.blocks[StorageMedium.GPU]) == 1
    assert subscriber.blocks[StorageMedium.CPU] == subscriber.blocks[StorageMedium.GPU]
    _check(core)

    core.invalidate_prefix_ref(receipt, 0)
    subscriber.receive(core)
    assert subscriber.blocks == {StorageMedium.GPU: set(), StorageMedium.CPU: set()}
    replacement, _ = _insert(core, [1, 2, 3, 4], [20, 21, 22, 23])
    subscriber.receive(core)
    _drain(core.demote(prefix))
    _drain(core.drive_host_eviction(component_type=FULL, num_tokens=2))
    assert subscriber.receive(core) == []
    assert subscriber.blocks[StorageMedium.GPU] == hashes
    assert _match(core, [1, 2, 3, 4]).best_match_node == replacement
    core.release_prefix_ref(receipt)
    assert core.prefix_ref_counts() == (0, 0)
    _check(core)


def test_retirement_keeps_locks_and_frees_only_the_old_suffix(backend):
    core = _core(backend)
    old, _ = _insert(core, range(8), range(10, 18))
    receipt = core.capture_prefix_ref(old, 8)
    old_lock = core.inc_lock_ref(old)
    _check(core)

    assert core.invalidate_prefix_ref(receipt, 4) == []
    assert core.is_invalidated(old)
    assert core.protected_size() == 8
    assert _match(core, range(8)).device_indices.tolist() == list(range(10, 14))
    assert core.match_full_device_prefix(_key(range(8)))[0] == 4
    _check(core)

    replacement, actions = _insert(
        core, range(8), [10, 11, 12, 13, 30, 31, 32, 33], prev_prefix_len=4
    )
    assert actions == []
    replacement_lock = core.inc_lock_ref(replacement)
    sibling, _ = _insert(
        core, [0, 1, 2, 3, 90, 91], [10, 11, 12, 13, 40, 41], prev_prefix_len=4
    )
    assert not core.is_invalidated(replacement)
    assert not core.is_invalidated(sibling)
    _check(core)

    core.dec_lock_ref(old, old_lock.to_dec_params())
    assert core.get_component_device_lock_ref(replacement, FULL) == 1
    assert _drain(core.evict_device_leaf(old, False)) == (
        {FULL: [[14, 15, 16, 17]]},
        {},
    )
    assert _match(core, range(8)).best_match_node == replacement
    assert core.protected_size() == 8
    _check(core)
    core.dec_lock_ref(replacement, replacement_lock.to_dec_params())
    core.release_prefix_ref(receipt)
    assert core.prefix_ref_counts() == (0, 0)
    _check(core)


def test_split_survivor_is_retired_after_original_suffix_eviction(backend):
    core = _core(backend)
    old, _ = _insert(core, range(8), range(10, 18))
    receipt = core.capture_prefix_ref(old, 8)
    prefix = _match(core, range(4)).best_match_node
    assert core.prefix_ref_counts() == (1, 2)
    _check(core)
    _drain(core.evict_device_leaf(old, False))
    assert core.prefix_ref_counts() == (1, 1)
    _check(core)

    assert core.invalidate_prefix_ref(receipt, 0) == []
    assert core.is_invalidated(prefix)
    assert _match(core, range(8)).device_indices.numel() == 0
    _check(core)
    replacement, _ = _insert(core, range(8), range(20, 28))
    core.invalidate_prefix_ref(receipt, 0)
    assert _match(core, range(8)).best_match_node == replacement
    _check(core)
    _drain(core.evict_device_leaf(prefix, False))
    assert core.prefix_ref_counts() == (1, 0)
    core.release_prefix_ref(receipt)
    assert core.prefix_ref_counts() == (0, 0)
    _check(core)


def test_uncaptured_replacement_suffix_survives_until_its_ancestor_is_retired(backend):
    core = _core(backend)
    old, _ = _insert(core, range(8), range(10, 18))
    receipt = core.capture_prefix_ref(old, 8)
    _match(core, range(4))
    _drain(core.evict_device_leaf(old, False))
    replacement, _ = _insert(core, range(8), range(20, 28))
    core.invalidate_prefix_ref(receipt, 4)
    assert not core.is_invalidated(replacement)
    assert _match(core, range(8)).best_match_node == replacement
    _check(core)
    core.invalidate_prefix_ref(receipt, 0)
    assert core.is_invalidated(replacement)
    assert _match(core, range(8)).device_indices.numel() == 0
    core.release_prefix_ref(receipt)
    _check(core)


@pytest.mark.parametrize("reset", [False, True])
def test_deleted_and_reset_receipts_cannot_target_replacements(backend, reset):
    core = _core(backend)
    old, _ = _insert(core, [1, 2], [10, 11])
    receipt = core.capture_prefix_ref(old, 2)
    if reset:
        core.reset()
    else:
        _drain(core.evict_device_leaf(old, False))
    replacement, _ = _insert(core, [1, 2], [20, 21])
    current = core.capture_prefix_ref(replacement, 2)
    core.invalidate_prefix_ref(receipt, 0)
    core.release_prefix_ref(receipt)
    core.release_prefix_ref(receipt)
    assert core.is_invalidated(old)
    assert _match(core, [1, 2]).device_indices.tolist() == [20, 21]
    assert core.prefix_ref_counts() == (1, 1)
    core.release_prefix_ref(current)
    assert core.prefix_ref_counts() == (0, 0)
    _check(core)


def test_owner_and_transaction_boundaries_reject_without_mutation(backend):
    core, other = _core(backend), _core(backend)
    old, _ = _insert(core, [1, 2], [10, 11])
    other_node, _ = _insert(other, [1, 2], [20, 21])
    receipt = core.capture_prefix_ref(old, 2)
    other_ref = other.capture_prefix_ref(other_node, 2)
    with pytest.raises(ValueError, match="another tree"):
        other.invalidate_prefix_ref(receipt, 0)
    with pytest.raises(ValueError, match="another tree"):
        other.release_prefix_ref(receipt)
    assert other.prefix_ref_counts() == (1, 1)
    assert _match(other, [1, 2]).device_indices.tolist() == [20, 21]

    core.set_hicache_enabled()
    core.write_through_threshold = 1
    step = core.begin_insert(
        InsertParams(
            key=_key([1, 2, 3]), value=torch.tensor([10, 11, 30]), prev_prefix_len=2
        )
    )
    assert step.result is None
    with pytest.raises(RuntimeError, match="insert transaction"):
        core.capture_prefix_ref(old, 2)
    with pytest.raises(RuntimeError, match="insert transaction"):
        core.invalidate_prefix_ref(receipt, 0)
    while step.result is None:
        step = core.resume_insert()
    core.end_insert()
    assert _match(core, [1, 2]).device_indices.tolist() == [10, 11]
    core.release_prefix_ref(receipt)
    other.release_prefix_ref(other_ref)
    _check(core)
    _check(other)


@pytest.mark.parametrize("bigram", [False, True])
def test_namespace_and_page_boundaries_remain_independent(backend, bigram):
    core = _core(backend, page_size=2, is_eagle=bigram)
    tokens = list(range(9 if bigram else 8))
    old, _ = _insert(core, tokens, range(10, 18), extra_key="adapter", cache_salt="a")
    other, _ = _insert(core, tokens, range(20, 28), extra_key="adapter", cache_salt="b")
    receipt = core.capture_prefix_ref(old, 8)
    with pytest.raises(ValueError):
        core.capture_prefix_ref(old, 9)
    with pytest.raises(ValueError):
        core.capture_prefix_ref(old, 10)
    with pytest.raises(ValueError):
        core.invalidate_prefix_ref(receipt, 3)
    assert core.prefix_ref_counts() == (1, 1)
    core.invalidate_prefix_ref(receipt, 4)
    assert _match(
        core, tokens, extra_key="adapter", cache_salt="a"
    ).device_indices.tolist() == [10, 11, 12, 13]
    assert (
        _match(core, tokens, extra_key="adapter", cache_salt="b").best_match_node
        == other
    )
    assert (
        core.match_full_device_prefix(
            _key(tokens, extra_key="adapter", cache_salt="a")
        )[0]
        == 4
    )
    core.release_prefix_ref(receipt)
    _check(core)


def test_write_through_ack_keeps_its_split_generation(backend):
    core = _core(backend)
    core.set_hicache_enabled()
    old, _ = _insert(core, range(4), range(10, 14))
    receipt = core.capture_prefix_ref(old, 4)
    lock = core.inc_lock_ref(old)
    core.mark_write_through_pending([old], old)
    core.commit_backup(old, torch.tensor([100, 101, 102, 103]), {})
    _check(core, write=[(old, old)])
    actions = core.invalidate_prefix_ref(receipt, 2)
    assert len(actions) == 1 and isinstance(actions[0], ReplaceWriteThroughOnNodeSplit)
    split = actions[0]
    assert split.ack_id == old and split.new_child_node_id == old
    prefix = split.new_node_id
    _check(core, write=[(old, prefix), (old, old)])
    replacement, _ = _insert(core, range(4), [10, 11, 20, 21], prev_prefix_len=2)
    core.finish_write_through([prefix, old], old)
    core.dec_lock_ref(old, lock.to_dec_params())
    _check(core)
    assert core.get_component_host_value(old, FULL).tolist() == [102, 103]
    assert core.get_component_host_value(replacement, FULL) is None
    assert _match(core, range(4)).best_match_node == replacement
    core.release_prefix_ref(receipt)


def test_host_lookup_and_load_ack_cannot_resurrect_retired_generations(backend):
    core = _core(backend)
    core.set_hicache_enabled()
    core.is_write_back = True
    old, _ = _insert(core, range(4), range(10, 14))
    receipt = core.capture_prefix_ref(old, 4)
    core.commit_backup(old, torch.tensor([100, 101, 102, 103]), {})
    _drain(core.demote(old))
    _check(core)
    assert _match(core, range(4)).host_hit_length == 4
    kv, aux = core.build_load_back_spec(old)
    assert core.commit_load_back(old, torch.tensor([30, 31, 32, 33]), kv, aux) == []
    lock = core.inc_lock_ref(old)
    _check(core, load=[(old, old)])
    core.invalidate_prefix_ref(receipt, 2)
    prefix = core.get_parent_node_id(old)
    _check(core, load=[(old, old)])
    replacement, _ = _insert(core, range(4), [30, 31, 40, 41], prev_prefix_len=2)
    blocked, auxiliary = core.build_load_back_spec(old)
    assert blocked.nodes_to_load == [] and auxiliary == {}
    core.finish_load_back(old)
    core.dec_lock_ref(old, lock.to_dec_params())
    assert not core.is_invalidated(prefix)
    assert _match(core, range(4)).best_match_node == replacement
    _check(core)
    _drain(core.demote(old))
    assert _match(core, range(4)).best_match_node == replacement
    core.release_prefix_ref(receipt)
    _check(core)


def test_host_loaded_suffix_preserves_the_device_ancestor(backend):
    core = _core(backend)
    core.set_hicache_enabled()
    core.is_write_back = True
    prefix, _ = _insert(core, [1, 2], [10, 11])
    old = core.insert_host(
        prefix, _key([3, 4]), torch.tensor([102, 103]), []
    ).inserted_host_node
    match = _match(core, [1, 2, 3, 4])
    assert match.last_device_node == prefix and match.last_host_node == old
    assert match.device_indices.tolist() == [10, 11] and match.host_hit_length == 2
    receipt = core.capture_prefix_ref(old, 4)
    kv, auxiliary = core.build_load_back_spec(old)
    assert kv.nodes_to_load == [old]
    # Planning a load that the caller declines neither materializes nor retires it.
    assert core.get_component_device_value(old, FULL) is None
    assert core.prefix_ref_counts() == (1, 2)
    _check(core)
    assert core.commit_load_back(old, torch.tensor([20, 21]), kv, auxiliary) == []
    lock = core.inc_lock_ref(old)
    core.invalidate_prefix_ref(receipt, 2)
    assert not core.is_invalidated(prefix) and core.is_invalidated(old)
    assert _match(core, [1, 2, 3, 4]).device_indices.tolist() == [10, 11]
    _check(core, load=[(old, old)])
    replacement, _ = _insert(core, [1, 2, 3, 4], [10, 11, 30, 31], prev_prefix_len=2)
    core.finish_load_back(old)
    core.dec_lock_ref(old, lock.to_dec_params())
    assert _drain(core.demote(old))[0] == {FULL: [[20, 21]]}
    _, host = _drain(core.drive_host_eviction(component_type=FULL, num_tokens=2))
    assert host == {FULL: [[102, 103]]}
    assert _match(core, [1, 2, 3, 4]).best_match_node == replacement
    core.release_prefix_ref(receipt)
    _check(core)


def test_receipt_metadata_is_bounded_by_captured_pages_and_drains(backend):
    core = _core(backend, page_size=2)
    old, _ = _insert(core, range(8), range(10, 18))
    receipts = [core.capture_prefix_ref(old, end) for end in (2, 4, 8)]
    for end in (6, 4, 2):
        _match(core, range(end))
        count, spans = core.prefix_ref_counts()
        assert count == 3 and spans <= (2 + 4 + 8) // 2
        _check(core)
    core.invalidate_prefix_ref(receipts[-1], 0)
    current = old
    while current != core.root_node_handle():
        parent = core.get_parent_node_id(current)
        _drain(core.evict_device_leaf(current, False))
        current = parent
        _check(core)
    assert core.prefix_ref_counts() == (3, 0)
    for receipt in receipts:
        core.release_prefix_ref(receipt)
    assert core.prefix_ref_counts() == (0, 0)


@pytest.mark.parametrize("auxiliary", [MAMBA, SWA])
def test_hybrid_retirement_preserves_component_ownership(backend, auxiliary):
    core = _core(backend, auxiliary=auxiliary)
    slot = 7 if auxiliary == MAMBA else None
    old, _ = _insert(core, range(4), range(10, 14), mamba_slot=slot)
    receipt = core.capture_prefix_ref(old, 4)
    old_lock = core.inc_lock_ref(old)
    _check(core)
    assert core.invalidate_prefix_ref(receipt, 2) == []
    prefix = core.get_parent_node_id(old)
    assert core.get_component_device_lock_ref(old, auxiliary) == 1
    _check(core)

    replacement, _ = _insert(
        core,
        range(4),
        [10, 11, 20, 21],
        prev_prefix_len=2,
        mamba_slot=8 if auxiliary == MAMBA else None,
    )
    new_lock = core.inc_lock_ref(replacement)
    core.dec_lock_ref(old, old_lock.to_dec_params())
    _check(core)
    old_device, old_host = _drain(core.evict_device_leaf(old, False))
    assert old_host == {}
    assert old_device[FULL] == [[12, 13]]
    # SWA frees use FULL indices so the allocator can release their mapped peers.
    assert old_device[auxiliary] == ([[7]] if auxiliary == MAMBA else [[12, 13]])
    assert core.get_component_device_lock_ref(replacement, auxiliary) == 1
    assert _match(core, range(4)).best_match_node == replacement
    _check(core)

    core.dec_lock_ref(replacement, new_lock.to_dec_params())
    new_device, _ = _drain(core.evict_device_leaf(replacement, False))
    assert new_device[FULL] == [[20, 21]]
    assert new_device[auxiliary] == ([[8]] if auxiliary == MAMBA else [[20, 21]])
    _check(core)
    prefix_device, _ = _drain(core.evict_device_leaf(prefix, False))
    assert prefix_device[FULL] == [[10, 11]]
    assert prefix_device.get(auxiliary, []) == (
        [] if auxiliary == MAMBA else [[10, 11]]
    )
    assert core.total_size() == (0, 0)
    core.release_prefix_ref(receipt)
    assert core.prefix_ref_counts() == (0, 0)
    _check(core)


@pytest.mark.parametrize("auxiliary", [None, MAMBA, SWA])
def test_empty_effective_key_never_publishes_root_component_state(backend, auxiliary):
    core = _core(backend, auxiliary=auxiliary)
    node, actions = _insert(core, [], [], mamba_slot=7 if auxiliary == MAMBA else None)
    assert node == core.root_node_handle() and actions == []
    receipt = core.capture_prefix_ref(node, 0)
    assert core.invalidate_prefix_ref(receipt, 0) == []
    assert not core.is_invalidated(node)
    assert core.total_size() == (0, 0)
    core.release_prefix_ref(receipt)
    _check(core)


def test_recurrent_only_restore_retains_generation_identity(backend):
    core = _core(backend, auxiliary=MAMBA)
    core.set_hicache_enabled()
    core.is_write_back = True
    old, _ = _insert(core, range(4), range(10, 14), mamba_slot=7)
    core.mark_write_through_pending([old], old)
    core.commit_backup(
        old,
        torch.tensor([100, 101, 102, 103]),
        {MAMBA: [PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([107]))]},
    )
    core.finish_write_through([old], old)
    assert _drain(core.evict_component(old, MAMBA, EvictLayer.DEVICE))[0] == {
        MAMBA: [[7]]
    }
    _check(core)
    kv, auxiliary = core.build_load_back_spec(old)
    assert kv.host_indices.numel() == 0 and kv.nodes_to_load == []
    assert auxiliary[MAMBA][0].nodes_to_load == [old]
    receipt = core.capture_prefix_ref(old, 4)
    lock = core.inc_lock_ref(old)
    auxiliary[MAMBA][0].device_indices = torch.tensor([8])
    assert (
        core.commit_load_back(old, torch.empty(0, dtype=torch.int64), kv, auxiliary)
        == []
    )
    _check(core, load=[(old, old)])
    core.invalidate_prefix_ref(receipt, 0)
    replacement, _ = _insert(core, range(4), range(20, 24), mamba_slot=9)
    core.finish_load_back(old)
    core.dec_lock_ref(old, lock.to_dec_params())
    assert core.get_component_device_value(old, MAMBA).tolist() == [8]
    assert core.get_component_device_value(replacement, MAMBA).tolist() == [9]
    assert _match(core, range(4)).best_match_node == replacement
    core.release_prefix_ref(receipt)
    _check(core)


def test_late_host_child_inherits_its_retired_parent(backend):
    core = _core(backend)
    core.set_hicache_enabled()
    old, _ = _insert(core, [1, 2], [10, 11])
    core.mark_write_through_pending([old], old)
    core.commit_backup(old, torch.tensor([100, 101]), {})
    core.finish_write_through([old], old)
    receipt = core.capture_prefix_ref(old, 2)
    core.invalidate_prefix_ref(receipt, 0)
    descendant = core.insert_host(
        old, _key([3, 4]), torch.tensor([102, 103]), []
    ).inserted_host_node
    assert core.is_invalidated(descendant)
    assert _match(core, [1, 2, 3, 4]).host_hit_length == 0
    replacement, _ = _insert(core, [1, 2, 3, 4], [20, 21, 22, 23])
    assert _match(core, [1, 2, 3, 4]).best_match_node == replacement
    core.release_prefix_ref(receipt)
    _check(core)


def _check_optimized_python_ownership(operation):
    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    require(sys.flags.optimize == 1, "the ownership probe must run under python -O")
    core = _core("python")
    old, _ = _insert(core, [1, 2, 3, 4], [10, 11, 12, 13])
    receipt = core.capture_prefix_ref(old, 4)
    lock = core.inc_lock_ref(old)
    core.invalidate_prefix_ref(receipt, 0)
    prefix = None
    if operation == "retirement":
        require(
            _match(core, [1, 2, 3, 4]).device_indices.numel() == 0,
            "retired generation still matches through its live token edge",
        )
    else:
        node = core.node_by_id(old)
        prefix, action = core._split_node(node.key, node, 2)
        require(action is None, "the device-only split must not produce a DMA action")
        require(
            core.get_child_node_ids(core.root_node_handle()) == [prefix.id],
            "retired split retained the suffix edge under its former parent",
        )
        require(core.get_child_node_ids(prefix.id) == [old], "split lost its suffix")
        require(core.is_invalidated(prefix.id), "split revived a retired prefix")

    replacement, _ = _insert(core, [1, 2, 3, 4], [20, 21, 22, 23])
    require(replacement != old, "replacement reused the retired generation")
    require(core.protected_size() == 4, "retirement or split changed the old lock")
    _check(core)
    core.dec_lock_ref(old, lock.to_dec_params())
    freed, host = _drain(core.evict_device_leaf(old, False))
    expected = [10, 11, 12, 13] if prefix is None else [12, 13]
    require(freed == {FULL: [expected]} and host == {}, "old slots were not freed once")
    if prefix is not None:
        freed, host = _drain(core.evict_device_leaf(prefix.id, False))
        require(
            freed == {FULL: [[10, 11]]} and host == {}, "split prefix lost ownership"
        )
    require(
        _match(core, [1, 2, 3, 4]).device_indices.tolist() == [20, 21, 22, 23],
        "old generation eviction changed its replacement",
    )
    core.release_prefix_ref(receipt)
    require(core.prefix_ref_counts() == (0, 0), "receipt metadata did not drain")
    _check(core)


@pytest.mark.parametrize("operation", ["retirement", "split"])
def test_optimized_python_keeps_generation_ownership(operation):
    """Optimizing out assertions must not remove mandatory ownership mutations."""
    program = (
        "import sys; sys.path[:0] = sys.argv[1:3]; "
        "from test_prefix_generation_retirement import _check_optimized_python_ownership; "
        "_check_optimized_python_ownership(sys.argv[3])"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            program,
            str(Path(sglang.__file__).resolve().parents[1]),
            str(Path(__file__).resolve().parent),
            operation,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(f"optimized {operation} failed:\n{result.stdout}\n{result.stderr}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
