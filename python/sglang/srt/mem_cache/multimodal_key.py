"""Full content identities attached to multimodal positions in a cache key."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import msgspec


class MultimodalKeySpan(msgspec.Struct, frozen=True, array_like=True):
    start: int
    end: int
    identity: str
    offset: int = 0

    def __post_init__(self):
        if self.start < 0 or self.end <= self.start or self.offset < 0:
            raise ValueError("multimodal cache spans must be nonempty and non-negative")
        if self.offset + self.end - self.start > 1 << 64:
            raise ValueError("multimodal cache span offsets must fit in uint64")
        if (
            not self.identity.startswith("sha256:")
            or len(self.identity) != 71
            or self.identity != self.identity.lower()
            or len(bytes.fromhex(self.identity[7:])) != 32
        ):
            raise ValueError("multimodal cache identity must be a full SHA-256 digest")


def validate_mm_spans(spans: Sequence[MultimodalKeySpan]) -> None:
    previous_end = 0
    for span in spans:
        if span.start < previous_end:
            raise ValueError(
                "multimodal cache spans must be ordered and non-overlapping"
            )
        previous_end = span.end


def slice_mm_spans(
    spans: Sequence[MultimodalKeySpan], start: int, end: int
) -> tuple[MultimodalKeySpan, ...]:
    if end <= start:
        return ()
    return tuple(
        MultimodalKeySpan(
            start=max(span.start, start) - start,
            end=min(span.end, end) - start,
            identity=span.identity,
            offset=span.offset + max(start - span.start, 0),
        )
        for span in spans
        if span.end > start and span.start < end
    )


def shift_mm_spans(
    spans: Sequence[MultimodalKeySpan], distance: int
) -> tuple[MultimodalKeySpan, ...]:
    return tuple(
        MultimodalKeySpan(
            span.start + distance, span.end + distance, span.identity, span.offset
        )
        for span in spans
    )


def mm_identity_at(spans: Sequence[MultimodalKeySpan], position: int):
    for span in spans:
        if span.start > position:
            break
        if position < span.end:
            return (span.identity, span.offset + position - span.start)
    return None


def mm_segment_at(spans: Sequence[MultimodalKeySpan], position: int, end: int):
    for span in spans:
        if span.start > position:
            return None, min(end, span.start)
        if position < span.end:
            return (span.identity, span.offset + position - span.start), min(
                end, span.end
            )
    return None, end


def hash_mm_page(key, start: int, end: int, prior_digest: bytes | None) -> bytes:
    digest = hashlib.sha256()
    if prior_digest is not None:
        digest.update(prior_digest)
    digest.update(b"sglang-mm-cache-key-v1\0")
    digest.update(bytes([key.is_bigram]))
    digest.update((end - start).to_bytes(8, "little"))
    for position in range(start, end):
        for raw_position in range(position, position + 1 + int(key.is_bigram)):
            identity = mm_identity_at(key.mm_spans, raw_position)
            if identity is None:
                digest.update(b"\0")
                digest.update(int(key.token_ids[raw_position]).to_bytes(4, "little"))
            else:
                digest.update(b"\1")
                digest.update(bytes.fromhex(identity[0][7:]))
                digest.update(identity[1].to_bytes(8, "little"))
    return digest.digest()
