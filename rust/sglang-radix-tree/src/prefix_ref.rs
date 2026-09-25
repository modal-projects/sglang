//! Active receipt metadata; no nodes or historical ancestry are retained.
//!
//! With page size P and aligned captured ends E_r, R receipts use O(R + S)
//! entries, where S <= sum(E_r / P). Splits partition captured spans, deletion
//! prunes them, and release removes both forward and reverse entries.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::node::NodeId;

pub type PrefixRef = usize;
static NEXT_PREFIX_REF: AtomicUsize = AtomicUsize::new(1);

#[derive(Default)]
pub struct PrefixRefRegistry {
    refs: HashMap<PrefixRef, HashMap<NodeId, (usize, usize)>>,
    by_node: HashMap<NodeId, HashSet<PrefixRef>>,
}

impl PrefixRefRegistry {
    pub fn clear(&mut self) {
        self.refs.clear();
        self.by_node.clear();
    }

    pub fn capture(&mut self, spans: HashMap<NodeId, (usize, usize)>) -> PrefixRef {
        // Process-wide IDs also prevent a receipt from another native core
        // from resolving here. The Python boundary additionally checks owner.
        let id = NEXT_PREFIX_REF
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("prefix receipt ID exhausted");
        for &node_id in spans.keys() {
            self.by_node.entry(node_id).or_default().insert(id);
        }
        self.refs.insert(id, spans);
        id
    }

    pub fn overlapping(&self, id: PrefixRef, start: usize) -> Vec<(usize, NodeId)> {
        let mut spans = Vec::new();
        if let Some(captured) = self.refs.get(&id) {
            spans.extend(
                captured
                    .iter()
                    .filter_map(|(&node_id, &(span_start, end))| {
                        (end > start).then_some((span_start, node_id))
                    }),
            );
        }
        spans.sort_unstable();
        spans
    }

    pub fn release(&mut self, id: PrefixRef) {
        if let Some(spans) = self.refs.remove(&id) {
            for node_id in spans.keys() {
                let refs = self.by_node.get_mut(node_id).unwrap();
                refs.remove(&id);
                if refs.is_empty() {
                    self.by_node.remove(node_id);
                }
            }
        }
    }

    pub fn split(&mut self, child_id: NodeId, prefix_id: NodeId, split_len: usize) {
        let refs = self.by_node.get(&child_id).cloned().unwrap_or_default();
        for id in refs {
            let spans = self.refs.get_mut(&id).unwrap();
            let (start, end) = spans[&child_id];
            let split = start + split_len;
            spans.insert(prefix_id, (start, end.min(split)));
            self.by_node.entry(prefix_id).or_default().insert(id);
            if end > split {
                spans.insert(child_id, (split, end));
            } else {
                spans.remove(&child_id);
                self.by_node.get_mut(&child_id).unwrap().remove(&id);
            }
        }
        if self.by_node.get(&child_id).is_some_and(HashSet::is_empty) {
            self.by_node.remove(&child_id);
        }
    }

    pub fn remove(&mut self, node_id: NodeId) {
        if let Some(refs) = self.by_node.remove(&node_id) {
            for id in refs {
                self.refs.get_mut(&id).unwrap().remove(&node_id);
            }
        }
    }

    pub fn counts(&self) -> (usize, usize) {
        assert!(self.by_node.values().all(|refs| !refs.is_empty()));
        let forward: HashSet<_> = self
            .refs
            .iter()
            .flat_map(|(&id, spans)| spans.keys().map(move |&node_id| (node_id, id)))
            .collect();
        let reverse: HashSet<_> = self
            .by_node
            .iter()
            .flat_map(|(&node_id, refs)| refs.iter().map(move |&id| (node_id, id)))
            .collect();
        assert_eq!(forward, reverse);
        (self.refs.len(), self.refs.values().map(HashMap::len).sum())
    }
}
