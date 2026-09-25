"""Request-attempt ownership of delayed verification cache receipts.

The controller owns all tree operations. This record carries identities and
lifetime only; it never owns KV buffers or resolves a token key again.
"""

from __future__ import annotations

from typing import Any, Optional

import msgspec

from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import PrefixRef


class CacheVerificationAttempt(msgspec.Struct):
    identity: tuple[Any, int]
    # First admitted device prefix, lowered only by a committed host restore.
    start: Optional[int] = None
    matched_device_end: int = 0
    host_start: Optional[int] = None
    host_sources: list[tuple[int, int]] = msgspec.field(default_factory=list)
    prefix_refs: list[PrefixRef] = msgspec.field(default_factory=list)
    prefix_identity: Optional[tuple[int, int]] = None
    host_refs: list[PrefixRef] = msgspec.field(default_factory=list)
    pending: int = 0
    upstream_holds: int = 0
    dependents: list[CacheVerificationAttempt] = msgspec.field(default_factory=list)
    verification_started: bool = False
    closed: bool = False
    invalid: bool = False
    session_id: Optional[str] = None
    session_slot: Any = None


def request_attempt_identity(req) -> tuple[Any, int]:
    return (req.cache_request_handle, req.retraction_count)
