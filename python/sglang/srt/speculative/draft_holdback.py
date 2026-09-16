"""Prefix-match cap for a request-owned draft KV ring.

A draft whose KV lives in a per-request ring (the bounded all-SWA DFLASH
pool) cannot reuse draft KV across requests: on a prefix hit the ring is empty
for the reused part. The default policy caps the radix match so the target
re-prefills at least one draft window, which rewrites the ring (hard
hold-back). On caches that can only resume at sparse checkpoints (a Mamba
radix cache resumes at linear-attention states) the cap can fall back far
behind the uncapped resume point and cost much more than one window. The
soft policy skips the cap when that extra cost exceeds a threshold and lets
the draft attend only to the written part of its ring until generation fills
the window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.runtime_context import get_schedule, get_spec

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.radix_cache import RadixKey


def soft_holdback_threshold() -> Optional[int]:
    """Extra prefill tokens the hard hold-back may cost before going soft.

    None means the hold-back is always hard. Soft hold-back reads only the
    ring positions written since the resume point, which needs page size 1
    (the compact draft row is not page-aligned to the left otherwise).
    """
    threshold = get_spec().speculative_draft_soft_holdback_threshold
    if threshold is None or int(threshold) < 0:
        return None
    if get_schedule().page_size != 1:
        return None
    return int(threshold)


def resolve_reprefill_key_limit(
    tree_cache: BasePrefixCache, key: RadixKey, input_len: int
) -> Optional[int]:
    """Return the RadixKey limit to match `key` with, given the draft tail.

    `key.limit` is the caller's own cap (logprob start, etc.). Without a
    registered re-prefill tail it is returned unchanged. With one, the match
    is capped at `input_len - tail` unless the soft policy is on and the cache
    reports that the cap would cost more than the threshold in extra prefill.
    """
    tail = tree_cache.swa_reprefill_tail_tokens()
    if not tail:
        return key.limit
    capped = max(0, input_len - tail)
    threshold = soft_holdback_threshold()
    if threshold is not None:
        peek = tree_cache.peek_reprefill_resume(key, capped)
        if peek is not None:
            uncapped_len, capped_len = peek
            if uncapped_len - capped_len > threshold:
                return key.limit
    return capped if key.limit is None else min(key.limit, capped)
