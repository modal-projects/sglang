"""Full multimodal identities survive native tree operations and page hashing."""

import shutil
from array import array

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

if shutil.which("cargo") is None:
    pytest.skip("the rust backend builds with cargo", allow_module_level=True)

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.multimodal_key import MultimodalKeySpan
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore
from sglang.srt.mem_cache.rust_tree_core.extension import bindings
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.utils import get_hash_str, get_storage_hash_str


def _span(start, end, value, offset=0):
    return MultimodalKeySpan(start, end, "sha256:" + f"{value:02x}" * 32, offset)


def _key(tokens, spans=(), *, bigram=False, limit=None):
    return RadixKey(
        array("q", tokens), is_bigram=bigram, limit=limit, mm_spans=tuple(spans)
    )


def _wire_spans(key):
    return [(span.start, span.end, span.identity, span.offset) for span in key.mm_spans]


def _core(*, bigram=False, page_size=1):
    return RustUnifiedTreeCore(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=page_size,
            is_eagle=bigram,
            enable_kv_cache_events=True,
            tree_components=(ComponentType.FULL,),
        )
    )


def _insert(core, key, offset=10):
    params = InsertParams(
        key=key, value=torch.arange(offset, offset + len(key), dtype=torch.int64)
    )
    step = core.begin_insert(params)
    while step.result is None:
        step = core.resume_insert()
    core.end_insert()
    return step.result


@pytest.mark.parametrize("bigram", [False, True])
@pytest.mark.parametrize("page_size", [1, 2, 4])
def test_native_page_hash_matches_python_across_span_boundaries(bigram, page_size):
    key = _key(
        [1, 2, 9, 9, 9, 3, 9, 4], [_span(2, 5, 7, 3), _span(6, 7, 8)], bigram=bigram
    )
    for sliced in (key, key[1:], key[2:5], key[:1]):
        assert bindings.get_hash_str(
            sliced.raw_token_ids(), None, page_size, bigram, _wire_spans(sliced)
        ) == get_hash_str(sliced, page_size=page_size)


@pytest.mark.parametrize("bigram", [False, True])
def test_native_match_and_split_use_identity_and_keep_earlier_prefix(bigram):
    core = _core(bigram=bigram)
    core.enable_storage = True
    first = _key([1, 9, 9, 2, 9, 3], [_span(1, 3, 7), _span(4, 5, 8)], bigram=bigram)
    routed = _key([1, 99, 99, 2, 99, 3], first.mm_spans, bigram=bigram)
    changed = _key(
        [1, 99, 99, 2, 99, 3], [_span(1, 3, 7), _span(4, 5, 9)], bigram=bigram
    )
    _insert(core, first)
    assert core.match_prefix(
        MatchPrefixParams(key=routed)
    ).device_indices.tolist() == list(range(10, 10 + len(first)))
    prefix_len = 4 - int(bigram)
    assert (
        core.match_prefix(MatchPrefixParams(key=changed)).device_indices.numel()
        == prefix_len
    )
    _insert(core, changed, offset=20)
    result = core.match_prefix(MatchPrefixParams(key=changed))
    assert result.device_indices.tolist() == list(range(10, 10 + prefix_len)) + list(
        range(20 + prefix_len, 20 + len(changed))
    )
    snapshot = core.snapshot_buffer_backup(
        result.best_match_node, pass_prefix_keys=True
    )
    assert snapshot.key.mm_spans == changed[prefix_len:].mm_spans
    assert (
        snapshot.hash_values == get_storage_hash_str(changed, page_size=1)[prefix_len:]
    )


def test_host_insert_and_single_token_span_remain_content_aware():
    core = _core()
    core.set_hicache_enabled()
    original = _key([9], [_span(0, 1, 7, 11)])
    routed = _key([99], original.mm_spans)
    different = _key([9], [_span(0, 1, 8, 11)])
    inserted = core.insert_host(
        core.root_node_handle(),
        original,
        torch.tensor([100]),
        get_storage_hash_str(original, page_size=1),
    )
    assert inserted.inserted_host_node is not None
    assert core.match_prefix(MatchPrefixParams(key=routed)).host_hit_length == 1
    assert core.match_prefix(MatchPrefixParams(key=different)).host_hit_length == 0


def test_limit_clips_native_sidecar_and_keeps_item_offset():
    core = _core()
    original = _key([1, 9, 9, 9, 2], [_span(1, 4, 7)], limit=3)
    _insert(core, original)
    routed = _key([1, 99, 99], [_span(1, 3, 7)])
    assert core.match_prefix(MatchPrefixParams(key=routed)).device_indices.tolist() == [
        10,
        11,
        12,
    ]


@pytest.mark.parametrize(
    "spans",
    [
        [(0, 0, "sha256:" + "07" * 32, 0)],
        [(0, 2, "sha256:" + "07" * 32, 0)],
        [(0, 1, "short", 0)],
    ],
)
def test_native_boundary_rejects_invalid_span_contract(spans):
    with pytest.raises(ValueError):
        bindings.MatchParamsBinding(array("q", [9]), mm_spans=spans)


def test_multimodal_event_fields_survive_removal_and_text_events_stay_small():
    core = _core()
    media = _key([1, 9, 9, 2], [_span(1, 3, 7)])
    inserted = _insert(core, media)
    expected = get_hash_str(media, page_size=1)
    stores = core.take_events()
    assert len(stores) == 1
    assert stores[0].block_hashes_sha256 == expected
    assert stores[0].parent_block_hash_sha256 is None
    dropped = core.drop_subtree_no_host(inserted.last_device_node)
    assert dropped.is_dropped
    assert sum(
        tensor.numel() for values in dropped.device_frees.values() for tensor in values
    ) == len(media)
    dropped.device_frees.clear()
    dropped.host_frees.clear()
    removed = core.take_events()
    assert removed[0].block_hashes_sha256 == expected

    text = _core()
    _insert(text, _key([1, 2, 3]))
    events = text.take_events()
    assert events[0].block_hashes_sha256 is None
    assert events[0].parent_block_hash_sha256 is None


@pytest.mark.parametrize("bigram", [False, True])
@pytest.mark.parametrize("page_size", [1, 2])
def test_mm_generation_retirement_keeps_split_offsets_and_replacement_events(
    bigram, page_size
):
    core = _core(bigram=bigram, page_size=page_size)
    core.enable_storage = True
    tokens = [1, 9, 9, 9, 9, 9, 9, 2] + ([3] if bigram else [])
    rerouted_tokens = [1, 99, 99, 99, 99, 99, 99, 2] + ([3] if bigram else [])
    media = _key(tokens, [_span(1, 7, 7)], bigram=bigram)
    rerouted = _key(rerouted_tokens, media.mm_spans, bigram=bigram)
    old = _insert(core, media).last_device_node
    receipt = core.capture_prefix_ref(old, len(media))
    lock = core.inc_lock_ref(old)
    core.take_events()

    prefix = core.match_prefix(MatchPrefixParams(key=media[:2])).best_match_node
    snapshot = core.snapshot_buffer_backup(old, pass_prefix_keys=True)
    assert snapshot.key.mm_spans == media[2:].mm_spans
    assert core.snapshot_buffer_backup(prefix, False).key.mm_spans == media[:2].mm_spans
    expected = get_hash_str(media, page_size=page_size)[2 // page_size :]
    core.invalidate_prefix_ref(receipt, 2)
    removals = core.take_events()
    assert [
        value for event in removals for value in event.block_hashes_sha256
    ] == expected
    assert core.is_invalidated(old)
    assert not core.is_invalidated(prefix)

    replacement = _insert(core, rerouted, offset=20).last_device_node
    stores = core.take_events()
    assert [
        value for event in stores for value in event.block_hashes_sha256
    ] == expected
    assert replacement != old
    core.invalidate_prefix_ref(receipt, 2)
    assert not core.is_invalidated(replacement)
    core.dec_lock_ref(old, lock.to_dec_params())
    released = core.evict_device_leaf(old, False)
    released.device_frees.clear()
    released.host_frees.clear()
    assert core.take_events() == []
    assert core.match_prefix(
        MatchPrefixParams(key=rerouted)
    ).device_indices.tolist() == [10, 11, 22, 23, 24, 25, 26, 27]
    core.release_prefix_ref(receipt)
    core.sanity_check([], [])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
