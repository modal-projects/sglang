import copy
import pickle
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import tiktoken

from sglang.srt.utils.tokenizer_segment_cache import (
    _CACHE_ATTR,
    patch_chat_segment_cache,
    unpatch_chat_segment_cache,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _encoding(*, merged=True):
    ranks = {bytes([value]): value for value in range(256)}
    if merged:
        ranks[b"ab"] = 256
    return tiktoken.Encoding(
        name="segment-test",
        pat_str=r"[\s\S]+",
        mergeable_ranks=ranks,
        special_tokens={"<|marker|>": 300},
    )


def _segments(*texts, allow_special=False):
    return [SimpleNamespace(text=text, allow_special=allow_special) for text in texts]


class _PieceTokenizer:
    def __init__(self):
        self.model = _encoding()
        self.special_tokens = {"<|marker|>": 300}
        self.encode_calls = 0
        self.before_encode = None

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(text, limit):
        return [text]

    def _encode_text_piece(self, text, allow_special_tokens=True):
        self.encode_calls += 1
        if self.before_encode is not None:
            self.before_encode()
        if allow_special_tokens and text in self.special_tokens:
            return [self.special_tokens[text]]
        output = []
        for part in self._split_whitespaces_or_nonwhitespaces(text, 25_000):
            if allow_special_tokens:
                output.extend(self.model.encode(part, allowed_special="all"))
            else:
                output.extend(self.model.encode(part, disallowed_special=()))
        return output

    def _encode_chat_segments(self, segments):
        output = []
        for segment in segments:
            output.extend(self._encode_text_piece(segment.text, segment.allow_special))
        return output

    def add_tokens(self, new_tokens, *, fail=False):
        if isinstance(self.model, _MutableEncoding):
            self.model.offset += 1
        else:
            self.model = _encoding(merged=False)
        if fail:
            raise ValueError("mutation failed")
        return len(new_tokens)


class _WideEncoding:
    def encode(self, text, **kwargs):
        return [2**40, -1]


class _MutableEncoding:
    offset = 1

    def encode(self, text, **kwargs):
        return [self.offset]


class _ExpandingEncoding:
    def encode(self, text, **kwargs):
        return [7] * int(text)


def _chat_piece(tokenizer, text, allow_special_tokens=True):
    return tokenizer._encode_chat_segments(
        _segments(text, allow_special=allow_special_tokens)
    )


class TestTokenizerSegmentCache(CustomTestCase):
    def setUp(self):
        self.tokenizer = _PieceTokenizer()
        self.raw_piece = _PieceTokenizer._encode_text_piece
        self.raw_segments = _PieceTokenizer._encode_chat_segments
        self.addCleanup(unpatch_chat_segment_cache, self.tokenizer)

    def enable(self, budget=4096, tokenizer=None):
        return patch_chat_segment_cache(tokenizer or self.tokenizer, max_chars=budget)

    def test_segment_boundaries_modes_and_return_ownership(self):
        cases = [
            _segments("a", "b"),
            _segments("ab"),
            _segments("<|marker|>", allow_special=True),
            _segments("<|marker|>", allow_special=False),
            _segments("\U0001f600", "\ud800", "", "a\nb"),
        ]
        expected = [self.raw_segments(self.tokenizer, case) for case in cases]
        self.assertNotEqual(expected[0], expected[1])
        self.assertNotEqual(expected[2], expected[3])
        self.enable()
        for case, reference in zip(cases, expected):
            for _ in range(2):
                actual = self.tokenizer._encode_chat_segments(case)
                self.assertEqual(actual, reference)
                actual.append(-100)
        before = self.tokenizer.encode_calls
        self.assertEqual(_chat_piece(self.tokenizer, "ab", False), expected[1])
        self.assertEqual(self.tokenizer.encode_calls, before)
        actual = _chat_piece(self.tokenizer, "ab")
        actual.append(-100)
        self.assertEqual(_chat_piece(self.tokenizer, "ab"), [256])

    def test_character_entry_token_bounds_and_lru(self):
        self.enable(budget=192)
        for text in ("a" * 80, "b" * 80, "a" * 80, "c" * 80):
            _chat_piece(self.tokenizer, text)
        state = self.tokenizer.__dict__[_CACHE_ATTR]
        before = self.tokenizer.encode_calls
        _chat_piece(self.tokenizer, "a" * 80)
        self.assertEqual(self.tokenizer.encode_calls, before)
        _chat_piece(self.tokenizer, "b" * 80)
        self.assertEqual(self.tokenizer.encode_calls, before + 1)
        for index in range(30):
            text = str(index)
            expected = self.raw_piece(self.tokenizer, text)
            self.assertEqual(_chat_piece(self.tokenizer, text), expected)
            self.assertLessEqual(state.chars, state.max_chars)
            self.assertLessEqual(state.tokens, state.max_tokens)
            self.assertLessEqual(len(state.entries), state.max_entries)
        self.assertLessEqual(len(state.entries), 3)
        before_entries = tuple(state.entries)
        _chat_piece(self.tokenizer, "x" * 193)
        _chat_piece(self.tokenizer, "")
        self.assertEqual(tuple(state.entries), before_entries)

    def test_direct_piece_admission_and_return_ownership(self):
        self.enable(budget=8192)
        for length in (4095, 4096, 4097):
            text = "a" * length
            expected = self.raw_piece(self.tokenizer, text, False)
            actual = self.tokenizer._encode_text_piece(text, False)
            self.assertEqual(actual, expected)
            before = self.tokenizer.encode_calls
            actual.append(-100)
            self.assertEqual(self.tokenizer._encode_text_piece(text, False), expected)
            self.assertEqual(self.tokenizer.encode_calls - before, int(length < 4096))

    def test_chat_preserves_instance_encoder_overrides(self):
        """Overridden encoders retain dispatch on misses and after cached history."""
        self.tokenizer._encode_text_piece = lambda text, allow_special_tokens=True: [
            700 + int(allow_special_tokens)
        ]
        self.enable()
        self.assertEqual(_chat_piece(self.tokenizer, "ab"), [701])
        self.assertEqual(_chat_piece(self.tokenizer, "ab", False), [700])
        del self.tokenizer._encode_text_piece
        self.assertEqual(_chat_piece(self.tokenizer, "ab"), [256])

        emitted = iter(([702], [703], [704]))
        self.tokenizer._encode_text_piece = lambda text, allow_special_tokens=True: (
            next(emitted)
        )
        self.assertEqual(_chat_piece(self.tokenizer, "ab"), [702])
        self.assertEqual(_chat_piece(self.tokenizer, "new text"), [703])
        self.assertEqual(_chat_piece(self.tokenizer, "ab"), [704])
        del self.tokenizer._encode_text_piece
        self.assertEqual(_chat_piece(self.tokenizer, "ab"), [256])

    def test_chat_preserves_subclass_and_foreign_bound_encoders(self):
        """An inherited segment loop dispatches to the receiver's actual encoder."""
        self.enable()

        class ChildTokenizer(_PieceTokenizer):
            def _encode_text_piece(self, text, allow_special_tokens=True):
                return [800 + int(allow_special_tokens)]

        child = self.enable(tokenizer=ChildTokenizer())
        self.assertEqual(_chat_piece(child, "ab"), [801])
        self.assertEqual(_chat_piece(child, "ab", False), [800])

        other = self.enable(tokenizer=_PieceTokenizer())
        other.model = _MutableEncoding()
        other.model.offset = 802
        self.tokenizer._encode_text_piece = other._encode_text_piece
        self.assertEqual(_chat_piece(self.tokenizer, "ab"), [802])
        self.assertEqual(_chat_piece(self.tokenizer, "a" * 4096), [802])

    def test_chat_override_can_wrap_cached_direct_encoder(self):
        """A decorator keeps its output without recursive or duplicate admission."""
        text = "a" * 4096
        expected = self.raw_piece(self.tokenizer, text) + [900]
        self.enable(budget=8192)
        previous = self.tokenizer._encode_text_piece
        self.tokenizer._encode_text_piece = lambda text, allow_special_tokens=True: (
            previous(text, allow_special_tokens) + [900]
        )
        for _ in range(2):
            self.assertEqual(_chat_piece(self.tokenizer, text), expected)
        state = self.tokenizer.__dict__[_CACHE_ATTR]
        self.assertEqual(state.chars, len(text))
        self.assertEqual(state.tokens, len(expected) - 1)
        self.assertEqual(len(state.entries), 1)

    def test_token_storage_limit_independent_of_text_and_entry_bounds(self):
        self.tokenizer.model = _ExpandingEncoding()
        self.enable(budget=192)
        for text in ("300", "0300", "00300"):
            self.assertEqual(_chat_piece(self.tokenizer, text), [7] * 300)
        state = self.tokenizer.__dict__[_CACHE_ATTR]
        self.assertEqual(state.tokens, 600)
        self.assertEqual(state.chars, 9)
        self.assertEqual(len(state.entries), 2)
        self.assertNotIn((True, "300"), state.entries)
        before_entries = tuple(state.entries)
        self.assertEqual(_chat_piece(self.tokenizer, "769"), [7] * 769)
        self.assertEqual(tuple(state.entries), before_entries)

    def test_independent_instances_budgets_and_copies(self):
        self.enable(budget=128)
        _chat_piece(self.tokenizer, "ab")
        state = self.tokenizer.__dict__[_CACHE_ATTR]
        for copier in (
            copy.copy,
            copy.deepcopy,
            lambda obj: pickle.loads(pickle.dumps(obj)),
        ):
            clone = copier(self.tokenizer)
            self.assertEqual(_chat_piece(clone, "ab"), [256])
            clone_state = clone.__dict__[_CACHE_ATTR]
            self.assertIsNot(clone_state, state)
            self.assertIs(clone_state.owner(), clone)
            self.assertEqual(clone_state.misses, 1)
        disabled = self.enable(budget=0, tokenizer=_PieceTokenizer())
        _chat_piece(disabled, "ab")
        self.assertNotIn(_CACHE_ATTR, disabled.__dict__)
        smaller = self.enable(budget=1, tokenizer=_PieceTokenizer())
        _chat_piece(smaller, "ab")
        smaller_state = smaller.__dict__[_CACHE_ATTR]
        self.assertEqual(smaller_state.max_chars, 1)
        self.assertEqual(smaller_state.chars, 0)
        self.enable(budget=1)
        _chat_piece(self.tokenizer, "ab")
        self.assertEqual(self.tokenizer.__dict__[_CACHE_ATTR].chars, 0)

    def test_model_splitter_and_failed_mutation_invalidate(self):
        self.enable()
        self.assertEqual(_chat_piece(self.tokenizer, "ab"), [256])
        self.tokenizer._split_whitespaces_or_nonwhitespaces = lambda text, limit: list(
            text
        )
        self.assertEqual(_chat_piece(self.tokenizer, "ab"), [97, 98])
        del self.tokenizer._split_whitespaces_or_nonwhitespaces
        self.assertEqual(_chat_piece(self.tokenizer, "ab"), [256])
        self.tokenizer.model = _MutableEncoding()
        self.assertEqual(_chat_piece(self.tokenizer, "ab"), [1])
        with self.assertRaisesRegex(ValueError, "mutation failed"):
            self.tokenizer.add_tokens(["new"], fail=True)
        self.assertEqual(_chat_piece(self.tokenizer, "ab"), [2])
        self.tokenizer.model = _encoding()
        self.assertEqual(_chat_piece(self.tokenizer, "ab"), [256])
        self.assertEqual(_chat_piece(self.tokenizer, "<|marker|>"), [300])
        self.tokenizer.special_tokens["<|marker|>"] = 301
        self.assertEqual(_chat_piece(self.tokenizer, "<|marker|>"), [301])

    def test_unpatch_repatch_invalidates_other_instances(self):
        self.enable()
        peer = self.enable(tokenizer=_PieceTokenizer())
        _chat_piece(peer, "ab")
        old_state = peer.__dict__[_CACHE_ATTR]
        unpatch_chat_segment_cache(self.tokenizer)
        self.assertIs(_PieceTokenizer._encode_text_piece, self.raw_piece)
        peer.add_tokens(["new"])
        self.enable()
        self.assertEqual(_chat_piece(peer, "ab"), [97, 98])
        self.assertIsNot(peer.__dict__[_CACHE_ATTR], old_state)

    def test_wide_token_ids_bypass_compact_storage(self):
        self.tokenizer.model = _WideEncoding()
        self.enable()
        for _ in range(2):
            self.assertEqual(_chat_piece(self.tokenizer, "ab"), [2**40, -1])
            self.assertEqual(
                self.tokenizer._encode_chat_segments(_segments("ab", "cd")),
                [2**40, -1, 2**40, -1],
            )
        self.assertEqual(self.tokenizer.__dict__[_CACHE_ATTR].chars, 0)

    def test_generator_configuration_changes_preserve_original_semantics(self):
        self.enable()
        _chat_piece(self.tokenizer, "ab")

        def segments():
            yield _segments("ab")[0]
            self.tokenizer.model = _encoding(merged=False)
            yield _segments("ab")[0]

        self.assertEqual(
            self.tokenizer._encode_chat_segments(segments()), [256, 97, 98]
        )

    def test_concurrent_misses_and_mutation_are_serialized(self):
        self.enable()
        entered = threading.Event()
        release = threading.Event()
        mutation_started = threading.Event()

        def pause_encode():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("encode was not released")

        def mutate():
            mutation_started.set()
            return self.tokenizer.add_tokens(["new"])

        self.tokenizer.before_encode = pause_encode
        with ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(_chat_piece, self.tokenizer, "ab")
            self.assertTrue(entered.wait(5))
            second = pool.submit(_chat_piece, self.tokenizer, "ab")
            mutation = pool.submit(mutate)
            self.assertTrue(mutation_started.wait(5))
            with self.assertRaises(TimeoutError):
                mutation.result(timeout=0.05)
            release.set()
            self.assertEqual(first.result(5), [256])
            self.assertIn(second.result(5), ([256], [97, 98]))
            self.assertEqual(mutation.result(5), 1)
        self.tokenizer.before_encode = None
        self.assertEqual(_chat_piece(self.tokenizer, "ab"), [97, 98])
        state = self.tokenizer.__dict__[_CACHE_ATTR]
        self.assertEqual(state.chars, 2)
        self.assertEqual(len(state.entries), 1)


if __name__ == "__main__":
    unittest.main()
