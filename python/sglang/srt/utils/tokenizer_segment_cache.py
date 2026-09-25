import threading
import weakref
from array import array
from collections import OrderedDict
from types import MethodType

_CACHE_ATTR = "_sglang_chat_segment_cache"
_BUDGET_ATTR = "_sglang_chat_segment_budget"
_PATCH_ATTR = "_sglang_chat_segment_patch"
_STATE_LOCK = threading.RLock()
_MIN_DIRECT_CHARS = 4096


def _empty_cache():
    return None


def _splitter(tokenizer):
    splitter = tokenizer._split_whitespaces_or_nonwhitespaces
    if isinstance(splitter, MethodType) and splitter.__self__ is tokenizer:
        return splitter.__func__
    return splitter


class _SegmentCache:
    def __init__(self, tokenizer, *, max_chars, generation):
        self.owner = weakref.ref(tokenizer)
        self.generation = generation
        self.max_chars = max_chars
        # Cap metadata for tiny segments and output storage independently of text.
        self.max_entries = max(1, max_chars // 64)
        self.max_tokens = 4 * max_chars
        self.lock = threading.RLock()
        self.entries = OrderedDict()
        self.chars = 0
        self.tokens = 0
        self.hits = 0
        self.misses = 0
        self.model = None
        self.special_tokens = {}
        self.splitter = None

    def __copy__(self):
        return None

    def __deepcopy__(self, memo):
        return None

    def __reduce__(self):
        return (_empty_cache, ())

    def clear(self):
        self.entries.clear()
        self.chars = 0
        self.tokens = 0

    def matches(self, tokenizer):
        return (
            self.model is tokenizer.model
            and self.special_tokens == tokenizer.special_tokens
            and self.splitter is _splitter(tokenizer)
        )

    def refresh(self, tokenizer):
        if not self.matches(tokenizer):
            self.clear()
            self.model = tokenizer.model
            self.special_tokens = tokenizer.special_tokens.copy()
            self.splitter = _splitter(tokenizer)

    def encode(self, tokenizer, text, *, allow_special_tokens, original):
        key = (bool(allow_special_tokens), text)
        cached = self.entries.get(key)
        if cached is not None:
            self.entries.move_to_end(key)
            self.hits += 1
            return cached

        self.misses += 1
        ids = original(tokenizer, text, allow_special_tokens=allow_special_tokens)
        if not self.matches(tokenizer):
            self.refresh(tokenizer)
            return ids
        if not text or len(text) > self.max_chars or len(ids) > self.max_tokens:
            return ids
        try:
            compact_ids = array("I", ids)
        except (OverflowError, TypeError):
            return ids

        while self.entries and (
            self.chars + len(text) > self.max_chars
            or self.tokens + len(ids) > self.max_tokens
            or len(self.entries) >= self.max_entries
        ):
            (_, evicted_text), evicted_ids = self.entries.popitem(last=False)
            self.chars -= len(evicted_text)
            self.tokens -= len(evicted_ids)
        self.entries[key] = compact_ids
        self.chars += len(text)
        self.tokens += len(ids)
        return ids


def _cache(tokenizer, *, generation):
    with _STATE_LOCK:
        max_chars = tokenizer.__dict__.get(_BUDGET_ATTR, 0)
        if max_chars <= 0:
            return None
        state = tokenizer.__dict__.get(_CACHE_ATTR)
        if (
            state is None
            or state.owner() is not tokenizer
            or state.generation is not generation
            or state.max_chars != max_chars
        ):
            state = _SegmentCache(tokenizer, max_chars=max_chars, generation=generation)
            tokenizer.__dict__[_CACHE_ATTR] = state
        return state


def patch_chat_segment_cache(tokenizer, *, max_chars):
    """Cache independently encoded Kimi chat segments within a tokenizer instance."""
    tokenizer_cls = type(tokenizer)
    if max_chars <= 0:
        with _STATE_LOCK:
            tokenizer.__dict__.pop(_CACHE_ATTR, None)
            tokenizer.__dict__.pop(_BUDGET_ATTR, None)
        return tokenizer
    required = (
        "_encode_text_piece",
        "_encode_chat_segments",
        "_split_whitespaces_or_nonwhitespaces",
        "add_tokens",
    )
    if not all(callable(getattr(tokenizer_cls, name, None)) for name in required):
        return tokenizer
    with _STATE_LOCK:
        tokenizer.__dict__[_BUDGET_ATTR] = max_chars
        if getattr(tokenizer_cls, _PATCH_ATTR, None) is not None:
            return tokenizer
        originals = {name: getattr(tokenizer_cls, name) for name in required}
        original_piece = originals["_encode_text_piece"]
        original_segments = originals["_encode_chat_segments"]
        original_add_tokens = originals["add_tokens"]
        generation = object()

        def encode_text_piece(self, text, allow_special_tokens=True):
            # Short direct calls do not amortize cache locking and validation.
            if len(text) < _MIN_DIRECT_CHARS:
                return original_piece(self, text, allow_special_tokens)
            state = _cache(self, generation=generation)
            if state is None:
                return original_piece(self, text, allow_special_tokens)
            with state.lock:
                state.refresh(self)
                return list(
                    state.encode(
                        self,
                        text,
                        allow_special_tokens=allow_special_tokens,
                        original=original_piece,
                    )
                )

        def encode_chat_segments(self, segments):
            encoder = self._encode_text_piece
            # Custom encoders may be stateful; preserve their original dispatch.
            if (
                not isinstance(encoder, MethodType)
                or encoder.__self__ is not self
                or encoder.__func__ is not encode_text_piece
            ):
                return original_segments(self, segments)
            # An iterator can change tokenizer configuration between yields.
            if not isinstance(segments, (list, tuple)):
                return original_segments(self, segments)
            state = _cache(self, generation=generation)
            if state is None:
                return original_segments(self, segments)
            with state.lock:
                state.refresh(self)
                output = []
                for segment in segments:
                    output.extend(
                        state.encode(
                            self,
                            segment.text,
                            allow_special_tokens=segment.allow_special,
                            original=original_piece,
                        )
                    )
                return output

        def add_tokens(self, *args, **kwargs):
            state = _cache(self, generation=generation)
            if state is None:
                return original_add_tokens(self, *args, **kwargs)
            with state.lock:
                try:
                    return original_add_tokens(self, *args, **kwargs)
                finally:
                    state.clear()

        tokenizer_cls._encode_text_piece = encode_text_piece
        tokenizer_cls._encode_chat_segments = encode_chat_segments
        tokenizer_cls.add_tokens = add_tokens
        setattr(tokenizer_cls, _PATCH_ATTR, originals)
    return tokenizer


def unpatch_chat_segment_cache(tokenizer):
    with _STATE_LOCK:
        tokenizer_cls = type(tokenizer)
        originals = tokenizer_cls.__dict__.get(_PATCH_ATTR)
        if originals is not None:
            for name in ("_encode_text_piece", "_encode_chat_segments", "add_tokens"):
                setattr(tokenizer_cls, name, originals[name])
            delattr(tokenizer_cls, _PATCH_ATTR)
        tokenizer.__dict__.pop(_CACHE_ATTR, None)
        tokenizer.__dict__.pop(_BUDGET_ATTR, None)
    return tokenizer
