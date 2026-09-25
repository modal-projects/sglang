use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::*;
use crate::components::FULL;
use crate::node::NodeArena;

type Key = MultimodalKey<Vec<i64>>;

fn span(start: usize, end: usize, value: u8, offset: u64) -> MultimodalSpan {
    MultimodalSpan {
        start,
        end,
        identity: [value; 32],
        offset,
    }
}
fn key(tokens: &[i64], spans: Vec<MultimodalSpan>) -> Key {
    Key::new(tokens.to_vec(), spans)
}
fn digest<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn routing_placeholders_do_not_change_identity() {
    let left = key(&[1, 1000001, 1000001, 2], vec![span(1, 3, 7, 0)]);
    let right = key(&[1, 1000022, 1000022, 2], vec![span(1, 3, 7, 0)]);
    assert_eq!(left, right);
    assert_eq!(digest(&left), digest(&right));
    assert_eq!(left.match_len(0, &right, 1), 4);
    assert_eq!(
        key_hash_strings(&left, None, 1),
        key_hash_strings(&right, None, 1)
    );
}

#[test]
fn later_media_keeps_earlier_prefix() {
    let left = key(
        &[1, 9, 9, 2, 9, 3],
        vec![span(1, 3, 7, 0), span(4, 5, 8, 0)],
    );
    let right = key(
        &[1, 8, 8, 2, 8, 3],
        vec![span(1, 3, 7, 0), span(4, 5, 9, 0)],
    );
    assert_eq!(left.match_len(0, &right, 1), 4);
    let left_hashes = key_hash_strings(&left, None, 1);
    let right_hashes = key_hash_strings(&right, None, 1);
    assert_eq!(&left_hashes[..4], &right_hashes[..4]);
    assert_ne!(left_hashes[4], right_hashes[4]);
}

#[test]
fn slices_keep_within_item_offsets_and_repeated_spans() {
    let whole = key(
        &[1, 9, 9, 9, 2, 9],
        vec![span(1, 4, 7, 0), span(5, 6, 7, 0)],
    );
    let (head, tail) = whole.split_at(2);
    assert_eq!(head.spans, vec![span(1, 2, 7, 0)]);
    assert_eq!(tail.spans, vec![span(0, 2, 7, 1), span(3, 4, 7, 0)]);
    assert_ne!(tail.child_key(1), whole.slice_key(1, 2));
    let hashes = key_hash_strings(&whole, None, 1);
    assert_eq!(key_hash_strings(&tail, Some(&hashes[1]), 1), hashes[2..]);
}

#[test]
fn bigram_slices_retain_boundary_media() {
    let raw = key(&[1, 9, 9, 2], vec![span(1, 3, 7, 0)]);
    let bigram = MultimodalKey::<Vec<(i64, i64)>>::key_from_multimodal(&raw).into_owned();
    let (head, tail) = bigram.split_at(1);
    assert_eq!(head.spans, vec![span(1, 2, 7, 0)]);
    assert_eq!(tail.spans, vec![span(0, 2, 7, 0)]);
    let hashes = key_hash_strings(&bigram, None, 1);
    assert_eq!(key_hash_strings(&tail, Some(&hashes[0]), 1), hashes[1..]);
}

#[test]
fn text_only_hash_and_child_key_bytes_stay_unchanged() {
    let raw = vec![1, 2, 3, 4];
    let plain = Key::key_from(Cow::Borrowed(&raw)).into_owned();
    assert_eq!(digest(&raw), digest(&plain));
    assert_eq!(
        key_hash_strings(&plain, None, 2),
        crate::node::get_hash_str::<Vec<i64>>(&raw, None, 2)
    );
    let raw_bigram = <Vec<(i64, i64)>>::key_from(Cow::Borrowed(&raw)).into_owned();
    let bigram = MultimodalKey::<Vec<(i64, i64)>>::key_from_multimodal(&plain).into_owned();
    assert_eq!(digest(&raw_bigram), digest(&bigram));
    assert_eq!(
        key_hash_strings(&bigram, None, 1),
        crate::node::get_hash_str::<Vec<(i64, i64)>>(&raw_bigram, None, 1)
    );
}

#[test]
fn child_edges_compare_full_span_identity() {
    let mut arena = NodeArena::<Key>::new(vec![FULL], 1);
    let root = arena.root();
    let first = key(&[9], vec![span(0, 1, 7, 0)]);
    let other = key(&[9], vec![span(0, 1, 8, 0)]);
    let routed = key(&[99], vec![span(0, 1, 7, 0)]);
    let a = arena.alloc_child(root, first, 0, None).unwrap();
    let b = arena.alloc_child(root, other.clone(), 0, None).unwrap();
    assert_ne!(a, b);
    assert_eq!(
        arena.child_on_key_page_in_namespace(root, Default::default(), routed.page_view(0, 1)),
        Some(a)
    );
    assert_eq!(
        arena.child_on_key_page_in_namespace(root, Default::default(), other.page_view(0, 1)),
        Some(b)
    );
}

fn insert_params<K: ChildKeyType>(
    key: &K,
    offset: i64,
) -> crate::unified_tree_core::InsertParams<'_, K> {
    crate::unified_tree_core::InsertParams {
        key,
        namespace: Default::default(),
        value: tch::Tensor::from_slice(
            &(offset..offset + key.atom_len() as i64).collect::<Vec<_>>(),
        ),
        mamba_value: None,
        prev_prefix_len: 0,
        swa_evicted_seqlen: 0,
        swa_branching_seqlen: None,
        chunked: false,
        priority: 0,
        session_id: None,
        track_adopted_ranges: false,
    }
}

#[test]
fn native_tree_splits_share_prefix_and_preserve_backup_spans() {
    use crate::unified_tree_core::{CacheInitParams, MatchPrefixParams, UnifiedTreeCore};
    let mut tree = UnifiedTreeCore::<Key>::new(CacheInitParams::default(), vec![FULL]);
    tree.set_enable_storage(true);
    let original = key(
        &[1, 9, 9, 2, 9, 3],
        vec![span(1, 3, 7, 0), span(4, 5, 8, 0)],
    );
    let changed = key(
        &[1, 8, 8, 2, 8, 3],
        vec![span(1, 3, 7, 0), span(4, 5, 9, 0)],
    );
    tree.insert(&insert_params(&original, 10));
    let first_match = tree.match_prefix(&MatchPrefixParams {
        key: &changed,
        namespace: Default::default(),
    });
    assert_eq!(first_match.device_indices.size()[0], 4);
    tree.insert(&insert_params(&changed, 20));
    let result = tree.match_prefix(&MatchPrefixParams {
        key: &changed,
        namespace: Default::default(),
    });
    assert!(
        result
            .device_indices
            .equal(&tch::Tensor::from_slice(&[10i64, 11, 12, 13, 24, 25]))
    );
    let snapshot = tree
        .snapshot_buffer_backup(result.last_device_node_id, true)
        .unwrap();
    assert_eq!(snapshot.mm_spans, vec![span(0, 1, 9, 0)]);
    let hashes = key_hash_strings(&changed, None, 1);
    assert_eq!(snapshot.hash_values, hashes[4..]);
}

#[test]
fn page_aligned_and_empty_slices_preserve_span_invariants() {
    let original = key(&[1, 9, 9, 9, 2], vec![span(1, 4, 7, 3)]);
    assert!(original.slice_key(2, 2).spans.is_empty());
    assert_eq!(original.page_aligned(2).spans, vec![span(1, 4, 7, 3)]);
    assert!(validate_spans(&[span(0, 2, 7, u64::MAX)], 2).is_err());
    assert!(validate_spans(&[span(0, 2, 7, 0), span(1, 3, 8, 0)], 3).is_err());
}

#[test]
fn media_events_keep_full_hashes_through_splits_and_removal() {
    use crate::unified_tree_core::{
        CacheInitParams, KvCacheEvent, MatchPrefixParams, UnifiedTreeCore,
    };
    let mut tree = UnifiedTreeCore::<Key>::new(
        CacheInitParams {
            enable_kv_cache_events: true,
            ..Default::default()
        },
        vec![FULL],
    );
    let original = key(&[1, 9, 9, 2], vec![span(1, 3, 7, 0)]);
    let inserted = tree.insert(&insert_params(&original, 10));
    let expected = key_hash_strings(&original, None, 1);
    let events = tree.take_events();
    match &events[0] {
        KvCacheEvent::BlockStored {
            block_hashes_sha256,
            parent_block_hash_sha256,
            ..
        } => {
            assert_eq!(block_hashes_sha256.as_ref().unwrap(), &expected);
            assert!(parent_block_hash_sha256.is_none());
        }
        _ => panic!("expected store event"),
    }
    let prefix = key(&[1], vec![]);
    let matched = tree.match_prefix(&MatchPrefixParams {
        key: &prefix,
        namespace: Default::default(),
    });
    assert_eq!(matched.device_indices.size()[0], 1);
    tree.drop_subtree_no_host(inserted.last_device_node_id.unwrap())
        .unwrap();
    let removed: Vec<String> = tree
        .take_events()
        .into_iter()
        .filter_map(|event| match event {
            KvCacheEvent::BlockRemoved {
                block_hashes_sha256,
                ..
            } => block_hashes_sha256,
            _ => None,
        })
        .flatten()
        .collect();
    assert!(!removed.is_empty());
    assert!(removed.iter().all(|hash| expected.contains(hash)));
}

fn check_mm_retired_split_and_replacement<K: ChildKeyType>() {
    use crate::components::ComponentSet;
    use crate::unified_tree_core::{
        CacheInitParams, DecLockRefParams, KvCacheEvent, MatchPrefixParams, UnifiedTreeCore,
    };
    let original_raw = key(&[1, 9, 9, 9, 9, 2], vec![span(1, 5, 7, 0)]);
    let routed_raw = key(&[1, 99, 99, 99, 99, 2], vec![span(1, 5, 7, 0)]);
    let original = K::key_from_multimodal(&original_raw).into_owned();
    let routed = K::key_from_multimodal(&routed_raw).into_owned();
    let mut tree = UnifiedTreeCore::<K>::new(
        CacheInitParams {
            enable_kv_cache_events: true,
            ..Default::default()
        },
        vec![FULL],
    );
    let old = tree
        .insert(&insert_params(&original, 10))
        .last_device_node_id
        .unwrap();
    let expected = key_hash_strings(&original, None, 1);
    tree.take_events();
    let receipt = tree.capture_prefix_ref(old, original.atom_len()).unwrap();
    let lock = tree.inc_lock_ref(old, ComponentSet::EMPTY).unwrap();
    tree.invalidate_prefix_ref(receipt, 0).unwrap();
    let removed: Vec<_> = tree
        .take_events()
        .into_iter()
        .flat_map(|event| match event {
            KvCacheEvent::BlockRemoved {
                block_hashes_sha256,
                ..
            } => block_hashes_sha256.unwrap(),
            _ => panic!("retirement must only remove the old advertisements"),
        })
        .collect();
    assert_eq!(removed, expected);
    let replacement = tree
        .insert(&insert_params(&routed, 20))
        .last_device_node_id
        .unwrap();
    assert_ne!(old, replacement);
    tree.take_events();
    let old_idx = tree.arena.resolve(old).unwrap();
    let (prefix_idx, action) = tree.split_node_(old_idx, 2);
    assert!(action.is_none());
    let prefix = tree.arena.node(prefix_idx);
    let prefix_id = prefix.id;
    assert!(prefix.retired && prefix.emit_full_hashes);
    assert_eq!(prefix.key.mm_spans(), original.slice_key(0, 2).mm_spans());
    assert_eq!(
        tree.arena.node(old_idx).key.mm_spans(),
        original.suffix(2).mm_spans()
    );
    assert_eq!(prefix.hash_value.as_ref().unwrap(), &expected[..2]);
    assert_eq!(
        tree.arena.node(old_idx).hash_value.as_ref().unwrap(),
        &expected[2..]
    );
    assert_eq!(tree.full_protected_size(), original.atom_len());
    tree.invalidate_prefix_ref(receipt, 0).unwrap();
    assert!(!tree.is_invalidated(replacement));
    tree.dec_lock_ref(
        old,
        &DecLockRefParams {
            node_id: lock.node_id,
            skipped_lock_components: lock.skipped_lock_components,
            ..Default::default()
        },
        false,
    )
    .unwrap();
    tree.evict_device_leaf(old, false).unwrap();
    tree.evict_device_leaf(prefix_id, false).unwrap();
    assert!(tree.take_events().is_empty());
    assert!(
        tree.match_prefix(&MatchPrefixParams {
            key: &routed,
            namespace: Default::default()
        })
        .device_indices
        .equal(&tch::Tensor::from_slice(
            &(20..20 + routed.atom_len() as i64).collect::<Vec<_>>()
        ))
    );
    tree.release_prefix_ref(receipt);
    assert_eq!(tree.inspect_prefix_ref_counts(), (0, 0));
    tree.sanity_check(&[], &[]);
}

#[test]
fn multimodal_retired_splits_keep_replacement_generation_and_full_events() {
    check_mm_retired_split_and_replacement::<Key>();
    check_mm_retired_split_and_replacement::<MultimodalKey<Vec<(i64, i64)>>>();
}

#[test]
fn multimodal_receipt_tracks_surviving_split_after_slot_reuse() {
    use crate::unified_tree_core::{CacheInitParams, MatchPrefixParams, UnifiedTreeCore};
    let mut tree = UnifiedTreeCore::<Key>::new(CacheInitParams::default(), vec![FULL]);
    let original = key(&[1, 9, 9, 9, 9, 2], vec![span(1, 5, 7, 0)]);
    let routed = key(&[1, 99, 99, 99, 99, 2], vec![span(1, 5, 7, 0)]);
    let old = tree
        .insert(&insert_params(&original, 10))
        .last_device_node_id
        .unwrap();
    let receipt = tree.capture_prefix_ref(old, original.atom_len()).unwrap();
    let old_idx = tree.arena.resolve(old).unwrap();
    let (prefix_idx, _) = tree.split_node_(old_idx, 2);
    let prefix_id = tree.arena.node(prefix_idx).id;
    assert_eq!(tree.inspect_prefix_ref_counts(), (1, 2));
    tree.evict_device_leaf(old, false).unwrap();
    assert_eq!(tree.inspect_prefix_ref_counts(), (1, 1));
    tree.invalidate_prefix_ref(receipt, 0).unwrap();
    assert!(tree.is_invalidated(prefix_id));
    let replacement = tree
        .insert(&insert_params(&routed, 20))
        .last_device_node_id
        .unwrap();
    assert_eq!(tree.arena.resolve(replacement).unwrap(), old_idx);
    assert_ne!(replacement, old);
    tree.invalidate_prefix_ref(receipt, 0).unwrap();
    assert!(!tree.is_invalidated(replacement));
    assert_eq!(
        tree.match_prefix(&MatchPrefixParams {
            key: &routed,
            namespace: Default::default()
        })
        .device_indices
        .size()[0],
        6
    );
    tree.evict_device_leaf(prefix_id, false).unwrap();
    tree.release_prefix_ref(receipt);
    assert_eq!(tree.inspect_prefix_ref_counts(), (0, 0));
    tree.sanity_check(&[], &[]);
}
