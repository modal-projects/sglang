//! Sparse full-width media identities beside compact native token buffers.

use std::borrow::Cow;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use sha2::{Digest, Sha256};

use crate::node::{ChildKeyType, HashDigest, digest_to_hex, hash_page, parse_prior_hash};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultimodalSpan {
    pub start: usize,
    pub end: usize,
    pub identity: HashDigest,
    pub offset: u64,
}

impl MultimodalSpan {
    pub fn slice(spans: &[Self], start: usize, end: usize) -> Vec<Self> {
        if end <= start {
            return Vec::new();
        }
        spans
            .iter()
            .filter(|span| span.end > start && span.start < end)
            .map(|span| {
                let clipped_start = span.start.max(start);
                Self {
                    start: clipped_start - start,
                    end: span.end.min(end) - start,
                    identity: span.identity,
                    offset: span.offset + (clipped_start - span.start) as u64,
                }
            })
            .collect()
    }
}

pub fn validate_spans(spans: &[MultimodalSpan], raw_len: usize) -> Result<(), &'static str> {
    let mut previous_end = 0;
    for span in spans {
        if span.start < previous_end || span.end <= span.start || span.end > raw_len {
            return Err(
                "multimodal spans must be ordered, non-overlapping, nonempty, and within the raw token buffer",
            );
        }
        if span
            .offset
            .checked_add((span.end - span.start - 1) as u64)
            .is_none()
        {
            return Err("multimodal within-item offset exceeds uint64");
        }
        previous_end = span.end;
    }
    Ok(())
}

/// The raw token vector stays compact. Text-only keys allocate no span storage.
#[derive(Clone, Debug, Default)]
pub struct MultimodalKey<K: ChildKeyType> {
    pub tokens: K,
    pub spans: Vec<MultimodalSpan>,
}

impl<K: ChildKeyType> MultimodalKey<K> {
    pub fn new(tokens: K, spans: Vec<MultimodalSpan>) -> Self {
        let raw_len = tokens.atom_len() + usize::from(K::IS_BIGRAM && tokens.atom_len() > 0);
        validate_spans(&spans, raw_len).expect("validated multimodal key spans");
        Self { tokens, spans }
    }
}

impl<K: ChildKeyType> From<Vec<K::Atom>> for MultimodalKey<K> {
    fn from(tokens: Vec<K::Atom>) -> Self {
        Self {
            tokens: tokens.into(),
            spans: Vec::new(),
        }
    }
}

impl<K: ChildKeyType> AsRef<[K::Atom]> for MultimodalKey<K> {
    fn as_ref(&self) -> &[K::Atom] {
        self.tokens.as_ref()
    }
}

impl<K: ChildKeyType> PartialEq for MultimodalKey<K> {
    fn eq(&self, other: &Self) -> bool {
        KeyPageRef::<K>::new(self.tokens.as_ref(), &self.spans, 0).equivalent(
            &KeyPageRef::<K>::new(other.tokens.as_ref(), &other.spans, 0),
        )
    }
}
impl<K: ChildKeyType> Eq for MultimodalKey<K> {}
impl<K: ChildKeyType> Hash for MultimodalKey<K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        KeyPageRef::<K>::new(self.tokens.as_ref(), &self.spans, 0).hash(state);
    }
}

macro_rules! multimodal_key_impl {
    ($raw:ty, $atom:ty, $bigram:expr, $convert:expr) => {
        impl ChildKeyType for MultimodalKey<$raw> {
            type Atom = $atom;
            const IS_BIGRAM: bool = $bigram;
            fn key_from(token_ids: Cow<'_, Vec<i64>>) -> Cow<'_, Self> {
                Cow::Owned(Self {
                    tokens: <$raw>::key_from(token_ids).into_owned(),
                    spans: Vec::new(),
                })
            }
            fn key_from_multimodal(key: &MultimodalKey<Vec<i64>>) -> Cow<'_, Self> {
                $convert(key)
            }
            fn hash_words(atom: &Self::Atom) -> impl Iterator<Item = u32> {
                <$raw>::hash_words(atom)
            }
            fn atom_token(atom: &Self::Atom, index: usize) -> i64 {
                <$raw>::atom_token(atom, index)
            }
            fn raw_token_ids(atoms: &[Self::Atom]) -> Cow<'_, [i64]> {
                <$raw>::raw_token_ids(atoms)
            }
            fn mm_spans(&self) -> &[MultimodalSpan] {
                &self.spans
            }
            fn slice_key(&self, start: usize, end: usize) -> Self {
                let raw_end = if end > start {
                    end + usize::from(Self::IS_BIGRAM)
                } else {
                    start
                };
                Self {
                    tokens: self.tokens[start..end].to_vec(),
                    spans: MultimodalSpan::slice(&self.spans, start, raw_end),
                }
            }
        }
    };
}

multimodal_key_impl!(Vec<i64>, i64, false, Cow::Borrowed);
multimodal_key_impl!(Vec<(i64, i64)>, (i64, i64), true, |key: &MultimodalKey<
    Vec<i64>,
>| {
    let tokens = <Vec<(i64, i64)>>::key_from(Cow::Borrowed(&key.tokens)).into_owned();
    let spans = if tokens.is_empty() {
        Vec::new()
    } else {
        key.spans.clone()
    };
    Cow::Owned(MultimodalKey { tokens, spans })
});

/// Borrowed page view; span coordinates still refer to the original key.
pub struct KeyPageRef<'a, K: ChildKeyType> {
    pub atoms: &'a [K::Atom],
    pub spans: &'a [MultimodalSpan],
    pub start: usize,
}

impl<'a, K: ChildKeyType> KeyPageRef<'a, K> {
    pub fn new(atoms: &'a [K::Atom], spans: &'a [MultimodalSpan], start: usize) -> Self {
        Self {
            atoms,
            spans,
            start,
        }
    }
    pub fn has_mm(&self) -> bool {
        if self.atoms.is_empty() {
            return false;
        }
        let end = self.start + self.atoms.len() + usize::from(K::IS_BIGRAM);
        self.spans
            .iter()
            .any(|span| span.end > self.start && span.start < end)
    }
    fn identity_at(&self, position: usize) -> Option<(&HashDigest, u64)> {
        let index = self.spans.partition_point(|span| span.start <= position);
        if index == 0 {
            return None;
        }
        let span = &self.spans[index - 1];
        (position < span.end)
            .then(|| (&span.identity, span.offset + (position - span.start) as u64))
    }
    pub fn equal_atom(&self, index: usize, other: &Self, other_index: usize) -> bool {
        for part in 0..=usize::from(K::IS_BIGRAM) {
            match (
                self.identity_at(self.start + index + part),
                other.identity_at(other.start + other_index + part),
            ) {
                (Some(left), Some(right)) if left == right => {}
                (None, None)
                    if K::atom_token(&self.atoms[index], part)
                        == K::atom_token(&other.atoms[other_index], part) => {}
                _ => return false,
            }
        }
        true
    }
    pub fn equivalent(&self, other: &Self) -> bool {
        if self.atoms.len() != other.atoms.len() {
            return false;
        }
        if !self.has_mm() && !other.has_mm() {
            return self.atoms == other.atoms;
        }
        (0..self.atoms.len()).all(|index| self.equal_atom(index, other, index))
    }
    pub fn digest(&self, prior: Option<&HashDigest>) -> HashDigest {
        if !self.has_mm() {
            return hash_page::<K>(self.atoms, prior);
        }
        let mut hasher = Sha256::new();
        if let Some(prior) = prior {
            hasher.update(prior);
        }
        hasher.update(b"sglang-mm-cache-key-v1\0");
        hasher.update([u8::from(K::IS_BIGRAM)]);
        hasher.update((self.atoms.len() as u64).to_le_bytes());
        for (index, atom) in self.atoms.iter().enumerate() {
            for part in 0..=usize::from(K::IS_BIGRAM) {
                if let Some((identity, offset)) = self.identity_at(self.start + index + part) {
                    hasher.update([1]);
                    hasher.update(identity);
                    hasher.update(offset.to_le_bytes());
                } else {
                    hasher.update([0]);
                    hasher.update(
                        u32::try_from(K::atom_token(atom, part))
                            .expect("text token id does not fit in uint32")
                            .to_le_bytes(),
                    );
                }
            }
        }
        hasher.finalize().into()
    }
}

impl<K: ChildKeyType> Hash for KeyPageRef<'_, K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if !self.has_mm() {
            self.atoms.hash(state);
            return;
        }
        b"sglang-mm-key-v1".hash(state);
        self.atoms.len().hash(state);
        for (index, atom) in self.atoms.iter().enumerate() {
            for part in 0..=usize::from(K::IS_BIGRAM) {
                match self.identity_at(self.start + index + part) {
                    Some((identity, offset)) => {
                        1u8.hash(state);
                        identity.hash(state);
                        offset.hash(state);
                    }
                    None => {
                        0u8.hash(state);
                        K::atom_token(atom, part).hash(state);
                    }
                }
            }
        }
    }
}

pub fn key_hash_digests<K: ChildKeyType>(
    key: &K,
    prior: Option<&HashDigest>,
    page_size: usize,
) -> Vec<HashDigest> {
    if key.mm_spans().is_empty() {
        return crate::node::get_hash_digests::<K>(key.as_ref(), prior, page_size);
    }
    assert!(page_size > 0, "page_size must be positive");
    let mut prior = prior.copied();
    let mut digests = Vec::with_capacity(key.atom_len().div_ceil(page_size));
    for start in (0..key.atom_len()).step_by(page_size) {
        let end = (start + page_size).min(key.atom_len());
        let digest = key.page_view(start, end - start).digest(prior.as_ref());
        digests.push(digest);
        prior = Some(digest);
    }
    digests
}

pub fn key_hash_strings<K: ChildKeyType>(
    key: &K,
    prior_hash: Option<&str>,
    page_size: usize,
) -> Vec<String> {
    if key.mm_spans().is_empty() {
        return crate::node::get_hash_str::<K>(key.as_ref(), prior_hash, page_size);
    }
    let prior = prior_hash
        .filter(|value| !value.is_empty())
        .map(parse_prior_hash);
    key_hash_digests(key, prior.as_ref(), page_size)
        .iter()
        .map(digest_to_hex)
        .collect()
}

#[cfg(test)]
#[path = "tests/multimodal_key.rs"]
mod tests;
