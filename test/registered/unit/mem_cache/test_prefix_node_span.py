"""Exact node spans agree across the Python and compiled native backends."""

from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.fixture(params=["python", "rust"])
def backend(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def page_size(request):
    return request.param


@pytest.fixture(params=[False, True])
def bigram(request):
    return request.param


@pytest.fixture
def core(backend, page_size, bigram):
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=page_size,
        is_eagle=bigram,
        tree_components=(ComponentType.FULL,),
    )
    if backend == "rust":
        from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore

        return RustUnifiedTreeCore(params)
    cache = SimpleNamespace(enable_session_radix_cache=False)
    return UnifiedTreeCore(params, {ComponentType.FULL: FullComponent(cache, params)})


def _key(end, bigram):
    return RadixKey(array("q", range(end + int(bigram))))


def _insert(core, end, bigram, *, prev_prefix_len=0):
    step = core.begin_insert(
        InsertParams(
            key=_key(end, bigram),
            value=torch.arange(10, 10 + end, dtype=torch.int64),
            prev_prefix_len=prev_prefix_len,
        )
    )
    assert not step.actions
    while step.result is None:
        step = core.resume_insert()
        assert not step.actions
    assert not core.end_insert()
    return step.result.last_device_node


def _match(core, end, bigram):
    return core.match_prefix(MatchPrefixParams(key=_key(end, bigram)))


def test_spans_follow_split_and_retired_identities(core, bigram):
    root = core.empty_match_result.best_match_node
    assert core.prefix_node_span(root) == (0, 0)
    old = _insert(core, 8, bigram)
    assert core.prefix_node_span(old) == (0, 8)

    prefix = _match(core, 4, bigram).best_match_node
    assert core.prefix_node_span(prefix) == (0, 4)
    assert core.prefix_node_span(old) == (4, 8)
    child = _insert(core, 12, bigram, prev_prefix_len=8)
    assert core.prefix_node_span(child) == (8, 12)

    receipt = core.capture_prefix_ref(old, 8)
    assert not core.invalidate_prefix_ref(receipt, 4)
    assert core.is_invalidated(old)
    assert core.is_invalidated(child)
    replacement = _insert(core, 8, bigram, prev_prefix_len=4)
    assert replacement != old
    assert core.prefix_node_span(replacement) == (4, 8)
    assert core.prefix_node_span(old) == (4, 8)
    assert core.prefix_node_span(child) == (8, 12)
    assert _match(core, 8, bigram).best_match_node == replacement
    core.release_prefix_ref(receipt)


def test_span_follows_split_ancestor_of_retired_node(core, bigram):
    old = _insert(core, 12, bigram)
    parent = _match(core, 8, bigram).best_match_node
    receipt = core.capture_prefix_ref(old, 12)
    assert not core.invalidate_prefix_ref(receipt, 8)
    ancestor = _match(core, 4, bigram).best_match_node
    assert core.prefix_node_span(ancestor) == (0, 4)
    assert core.prefix_node_span(parent) == (4, 8)
    assert core.prefix_node_span(old) == (8, 12)
    core.release_prefix_ref(receipt)


@pytest.mark.parametrize("reset", [False, True])
def test_deleted_and_reset_ids_do_not_rematch_replacements(core, bigram, reset):
    old = _insert(core, 8, bigram)
    old_root = core.empty_match_result.best_match_node
    if reset:
        core.reset()
    else:
        result = core.evict_device_leaf(old, False)
        assert sum(v.numel() for v in result.device_frees[ComponentType.FULL]) == 8
        result.device_frees.clear()
        assert not result.host_frees
    replacement = _insert(core, 8, bigram)
    assert replacement != old
    assert core.prefix_node_span(replacement) == (0, 8)
    with pytest.raises(KeyError) as span_error:
        core.prefix_node_span(old)
    with pytest.raises(KeyError) as node_error:
        core.is_root(old)
    assert span_error.value.args == node_error.value.args == (old,)
    if reset:
        with pytest.raises(KeyError):
            core.prefix_node_span(old_root)
        assert core.prefix_node_span(core.empty_match_result.best_match_node) == (0, 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
