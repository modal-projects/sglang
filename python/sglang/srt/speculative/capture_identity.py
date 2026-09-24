"""ABI-v2 causal prefix identity; mirrored by adaptive_spec.identity.

SHA256 chains 64-token pages. Partial-page hashes identify every token boundary,
but the producer only hashes once per full page and once per exported boundary.
The cursor retains at most 63 IDs, regardless of prompt/decode length.
"""

from __future__ import annotations

import hashlib
import struct

PAGE_TOKENS = 64
HASH_SCHEME = "sha256-token-pages-v1"
DOMAIN = b"dflash-prefix-page-v1\0"


class PrefixCursor:
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.position = 0
        self.page_hash = bytes.fromhex(namespace)
        self.tail = []

    def advance(self, tokens):
        offset = 0
        while offset < len(tokens):
            count = min(PAGE_TOKENS - len(self.tail), len(tokens) - offset)
            self.tail.extend(tokens[offset : offset + count])
            self.position += count
            offset += count
            if len(self.tail) == PAGE_TOKENS:
                self.page_hash = hashlib.sha256(
                    DOMAIN + self.page_hash + struct.pack("<64q", *self.tail)
                ).digest()
                self.tail.clear()

    def snapshot(self):
        return {
            "namespace": self.namespace,
            "page_hash": self.page_hash.hex(),
            "page_tokens": list(self.tail),
        }
