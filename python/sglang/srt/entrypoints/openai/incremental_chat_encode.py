"""Per-conversation incremental tokenization cache for chat completions.

Motivation (ttft-iceberg attribution, kimi-v2): on ~500-message / ~95k-token
agent conversations, the OpenAI chat path re-renders and re-tokenizes the FULL
prompt every turn. Conversations grow append-only, so ~99% of that encode work
is identical to the previous turn. On the measured workload the full-prompt
tiktoken encode costs ~18 ms (fast host) to ~49 ms (slow host) per request and
scales linearly with context; this cache reduces it to encoding only the new
suffix (< 1 ms typical).

Design
------
Cache entry per conversation (keyed by an exact hash of the first message's
role+content, which is stable across turns of one session and unique per
user-session on salted benchmark traffic):

    rendered   full rendered chat-template string of the last request
    ids        token ids of `rendered` (exact output of tokenizer.encode)
    bounds     [(char_end, tok_end), ...] at every special-token boundary

On the next request from the same conversation:
  1. common prefix length between the new rendered string and the cached one
  2. junction = last special-token boundary at or before the divergence point
  3. reuse ids[:tok_end]; tokenizer.encode() only rendered[char_end:]

Byte-identity argument
----------------------
tiktoken's encode with allowed_special="all" first splits text on the
special-token regex; BPE runs independently on the segments between specials.
Therefore encode(A + B) == encode(A) + encode(B) whenever A ends exactly at a
special-token boundary. We only reuse prefixes ending at special-token
boundaries, so splicing cached prefix ids with freshly-encoded tail ids is
exact — with two structural caveats from the Kimi TikTokenTokenizer wrapper,
both guarded below (fall back to a full encode when violated):

  * 400k-char windows: the wrapper encodes text in absolute windows of
    TIKTOKEN_MAX_ENCODE_CHARS chars. We only serve from cache when both the
    cached and the new rendered strings are single-window (<= 400k chars).
  * 25k same-class run splitting: the wrapper splits runs of > 25k consecutive
    whitespace / non-whitespace chars at run-relative offsets. A run that
    crosses the junction could be split differently by the incremental
    encode, so we reject junctions inside a same-class run longer than the
    limit (bounded backward/forward scan).

Divergence (compaction restarts, salted or edited history, changed tools) just
shortens the reusable prefix — possibly to zero, which is a plain full encode.
The boundary map is rebuilt from the merged result after every request, so a
diverged entry heals immediately.

Enable with SGLANG_ICEBERG_INCR_ENCODE=1. Only activates for tokenizers with a
tiktoken `.model` and a fast `encode()` that takes no kwargs (Kimi-style); all
other tokenizers silently bypass. Single-threaded use only (the OpenAI serving
layer runs message processing synchronously on the event loop; with
tokenizer_worker_num > 1 each worker process holds its own cache).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections import OrderedDict
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("SGLANG_ICEBERG_INCR_ENCODE", "0") == "1"
MAX_SESSIONS = int(os.environ.get("SGLANG_ICEBERG_INCR_ENCODE_MAX_SESSIONS", "64"))

# Mirrors of the Kimi TikTokenTokenizer wrapper constants (guards, see module
# docstring). If the wrapper's constants ever change, single-window and
# run-crossing guards must be revisited — hence sourced defensively at init.
SINGLE_WINDOW_MAX_CHARS = 400_000
MAX_SAME_CLASS_RUN = 25_000


def _common_prefix_len(a: str, b: str, block: int = 65536) -> int:
    """Length of the common prefix of a and b (block compare + bisect tail)."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i : i + block] == b[i : i + block]:
        i += block
    if i >= n:
        return n
    lo, hi = i, min(i + block, n)  # first difference is in [lo, hi)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _run_crosses_junction(text: str, junction: int, limit: int) -> bool:
    """True if the same-class (isspace) char run containing the junction may
    exceed `limit` chars (scan bounded to limit+1 each way)."""
    n = len(text)
    if junction <= 0 or junction >= n:
        return False
    cls = text[junction].isspace()
    back = 0
    i = junction - 1
    while i >= 0 and text[i].isspace() == cls and back <= limit:
        back += 1
        i -= 1
    fwd = 0
    i = junction
    while i < n and text[i].isspace() == cls and fwd <= limit:
        fwd += 1
        i += 1
    return back + fwd > limit


class _Entry:
    __slots__ = ("rendered", "ids", "bounds")

    def __init__(self, rendered: str, ids: List[int], bounds: List[Tuple[int, int]]):
        self.rendered = rendered
        self.ids = ids
        self.bounds = bounds


class IncrementalChatEncoder:
    """See module docstring. One instance per serving process."""

    def __init__(self, tokenizer):
        self.ok = False
        self.tokenizer = tokenizer
        model = getattr(tokenizer, "model", None)
        specials = getattr(tokenizer, "special_tokens", None)
        if model is None or not specials:
            return
        try:
            if len(tokenizer.encode("")) != 0:
                return  # fast no-kwargs path required (no auto BOS)
        except Exception:
            return
        # longest-first alternation; all Kimi specials are distinct literals
        pat = "|".join(
            re.escape(s) for s in sorted(specials.keys(), key=len, reverse=True)
        )
        self._special_re = re.compile(pat)
        self._special_ids = {int(v) for v in specials.values()}
        self._specials_by_str = {k: int(v) for k, v in specials.items()}
        self._cache: "OrderedDict[str, _Entry]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.fallbacks = 0
        self.ok = True

    # ---- keying -------------------------------------------------------------
    @staticmethod
    def session_key(messages) -> Optional[str]:
        """Stable per-conversation key: hash of the first message (role +
        content). Salted benchmark/system prompts make this unique per user
        session; a same-key collision only costs prefix-match failure."""
        if not messages:
            return None
        m0 = messages[0]
        role = str(m0.get("role", ""))
        content = m0.get("content", "")
        if not isinstance(content, str):
            try:
                content = str(content)
            except Exception:
                return None
        h = hashlib.md5((role + "\x00" + content).encode("utf-8", "replace"))
        return h.hexdigest()

    # ---- boundary map -------------------------------------------------------
    def _scan_bounds_tail(
        self,
        rendered: str,
        ids: List[int],
        char_start: int,
        tok_start: int,
    ) -> Optional[List[Tuple[int, int]]]:
        """Bounds for the region rendered[char_start:] / ids[tok_start:],
        cross-checked text-vs-ids; None on inconsistency."""
        char_marks: List[Tuple[int, int]] = []
        for m in self._special_re.finditer(rendered, char_start):
            char_marks.append((m.end(), self._specials_by_str[m.group(0)]))
        bounds: List[Tuple[int, int]] = []
        k = 0
        sid = self._special_ids
        for off, tid in enumerate(ids):
            if tid in sid:
                if k >= len(char_marks) or char_marks[k][1] != tid:
                    return None
                bounds.append((char_marks[k][0], tok_start + off + 1))
                k += 1
        if k != len(char_marks):
            return None
        return bounds

    # ---- main entry ---------------------------------------------------------
    def encode(self, messages, rendered: str) -> List[int]:
        """Drop-in for tokenizer.encode(rendered) on the chat fast path."""
        key = self.session_key(messages)
        if key is None:
            return self.tokenizer.encode(rendered)

        t0 = time.perf_counter()
        entry = self._cache.get(key)
        hit = None
        if entry is not None:
            hit = self._try_incremental(entry, rendered)

        if hit is not None:
            ids, prefix_bounds, char_end, tok_end, tail_ids = hit
            self.hits += 1
            tail_bounds = self._scan_bounds_tail(rendered, tail_ids, char_end, tok_end)
            bounds = prefix_bounds + tail_bounds if tail_bounds is not None else None
        else:
            self.misses += 1
            ids = self.tokenizer.encode(rendered)
            bounds = self._scan_bounds_tail(rendered, ids, 0, 0)

        if bounds is not None:
            self._cache[key] = _Entry(rendered, ids, bounds)
            self._cache.move_to_end(key)
            while len(self._cache) > MAX_SESSIONS:
                self._cache.popitem(last=False)
        elif key in self._cache:
            del self._cache[key]  # unreusable shape; drop rather than risk it

        try:
            from sglang.srt import iceberg_trace

            iceberg_trace.stash(
                incr_hit=int(hit is not None),
                incr_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception:
            pass
        return ids

    def _try_incremental(self, entry: _Entry, rendered: str):
        """On success: (ids, prefix_bounds, char_end, tok_end, tail_ids)."""
        cpl = _common_prefix_len(rendered, entry.rendered)
        if cpl <= 0 or not entry.bounds:
            return None
        # last special boundary at or before the divergence point
        lo, hi = 0, len(entry.bounds)
        while lo < hi:
            mid = (lo + hi) // 2
            if entry.bounds[mid][0] <= cpl:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            return None
        char_end, tok_end = entry.bounds[lo - 1]
        if char_end <= 0:
            return None
        # Window guard: the wrapper encodes absolute windows of W chars
        # ([0:W], [W:2W], ...), so the split layout over [0, char_end) is
        # identical across requests sharing that prefix. The incremental tail
        # encode, however, sees windows relative to char_end — only safe when
        # the reference has NO absolute window mark inside the tail, i.e. the
        # junction and the end of text sit in the same absolute window. (The
        # cached entry's own encode shared the same absolute marks over its
        # prefix, so no cached-side condition is needed beyond the run guard.)
        w = SINGLE_WINDOW_MAX_CHARS
        if (char_end // w) != (max(len(rendered) - 1, 0) // w):
            self.fallbacks += 1
            return None
        # Same-class-run guard around the junction, on BOTH texts: the new one
        # (its encode(prefix)+encode(tail) must factorize at the junction) and
        # the cached one (its stored ids must factorize there too — a long run
        # crossing the junction in the cached text was run-split at offsets
        # that don't respect the junction). Texts agree up to cpl >= char_end,
        # so the backward scans coincide; forward scans can differ.
        if _run_crosses_junction(
            rendered, char_end, MAX_SAME_CLASS_RUN
        ) or _run_crosses_junction(entry.rendered, char_end, MAX_SAME_CLASS_RUN):
            self.fallbacks += 1
            return None
        tail = rendered[char_end:]
        tail_ids = self.tokenizer.encode(tail) if tail else []
        return (
            entry.ids[:tok_end] + tail_ids,
            entry.bounds[:lo],
            char_end,
            tok_end,
            tail_ids,
        )
