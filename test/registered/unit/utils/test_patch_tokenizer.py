import copy
import random
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

from transformers import AddedToken, AutoTokenizer

import sglang.srt.utils.hf_transformers.processor as processor_utils
from sglang.srt.environ import envs
from sglang.srt.utils.patch_tokenizer import (
    _SpecialTokensCachePatcher,
    decode_without_hf_kwargs,
    patch_mm_processor_tokenizer,
    patch_tokenizer,
    unpatch_tokenizer,
)
from sglang.srt.utils.tokenizer_encode_fast_path import _EncodePieceFastPathPatcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=45, suite="base-a-test-cpu", nightly=True)
register_cpu_ci(est_time=40, suite="stage-b-test-cpu-intel")


class TestPatchTokenizerEndToEndTest(unittest.TestCase):
    def test_patched_produces_same_results_as_raw(self):
        tokenizer = _load_tokenizer()
        test_texts = self._generate_test_texts(tokenizer)
        raw_results = self._run_tokenizer_ops(tokenizer, test_texts)

        _SpecialTokensCachePatcher.patch(tokenizer)
        patched_results = self._run_tokenizer_ops(tokenizer, test_texts)
        unpatch_tokenizer(tokenizer)

        self.assertEqual(raw_results, patched_results)

    @classmethod
    def _generate_test_texts(cls, tokenizer):
        special_tokens = tokenizer.all_special_tokens
        return [
            "Hello, world!",
            "This is a longer sentence with multiple words.",
            "Numbers 12345 and symbols !@#$%",
            "    leading and trailing spaces    ",
            "\n\nMultiple\n\nNewlines\n\n",
            *[f"Text with {tok} inside" for tok in special_tokens],
            " ".join(special_tokens),
            *[
                cls._random_text_from_tokens(tokenizer, num_tokens=100)
                for _ in range(5)
            ],
            *[
                cls._random_text_from_tokens(tokenizer, num_tokens=1000)
                for _ in range(3)
            ],
        ]

    @classmethod
    def _random_text_from_tokens(cls, tokenizer, num_tokens):
        token_ids = [
            random.randint(0, tokenizer.vocab_size - 1) for _ in range(num_tokens)
        ]
        return tokenizer.decode(token_ids)

    @classmethod
    def _run_tokenizer_ops(cls, tokenizer, texts):
        encode_results = [tokenizer.encode(t) for t in texts]
        batch_encode_results = tokenizer(texts)["input_ids"]
        return {
            "encode": encode_results,
            "batch_encode": batch_encode_results,
            "decode": [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in encode_results
            ],
            "batch_decode": tokenizer.batch_decode(
                encode_results, skip_special_tokens=True
            ),
            "special_tokens": tokenizer.all_special_tokens,
            "special_ids": tokenizer.all_special_ids,
        }


class TestPatchTokenizerUnitTest(unittest.TestCase):
    def test_patch_unpatch_restores_original(self):
        tokenizer = _load_tokenizer()
        cls = type(tokenizer)

        original_ids = _get_class_attr_ids(cls)

        _SpecialTokensCachePatcher.patch(tokenizer)
        self.assertTrue(getattr(cls, "_sglang_special_tokens_patched", False))

        patched_ids = _get_class_attr_ids(cls)
        changed_attrs = [
            name
            for name in original_ids
            if name in patched_ids and patched_ids[name] != original_ids[name]
        ]
        self.assertGreater(len(changed_attrs), 0, "Patch should change some attributes")

        unpatch_tokenizer(tokenizer)
        self.assertFalse(getattr(cls, "_sglang_special_tokens_patched", False))

        restored_ids = _get_class_attr_ids(cls)
        for name in original_ids:
            if name.startswith("_sglang") or name.startswith("_original"):
                continue
            self.assertEqual(
                restored_ids.get(name),
                original_ids[name],
                f"Attribute {name} should be restored to original",
            )

    def test_patch_caches_special_tokens(self):
        with _patched_tokenizer() as tokenizer:
            tokens1 = tokenizer.all_special_tokens
            ids1 = tokenizer.all_special_ids
            tokens2 = tokenizer.all_special_tokens
            ids2 = tokenizer.all_special_ids

            self.assertIs(tokens1, tokens2)
            self.assertIs(ids1, ids2)

    def test_patch_blocks_add_special_tokens(self):
        with _patched_tokenizer() as tokenizer:
            with self.assertRaises(AssertionError) as ctx:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.assertIn(
                "Cannot modify special tokens after patch", str(ctx.exception)
            )

    def test_patch_blocks_add_tokens_with_special_flag(self):
        with _patched_tokenizer() as tokenizer:
            with self.assertRaises(AssertionError) as ctx:
                tokenizer.add_tokens(["<new>"], special_tokens=True)
            self.assertIn("Cannot add special tokens after patch", str(ctx.exception))

            tokenizer.add_tokens(["<regular>"], special_tokens=False)

    def test_unpatch_clears_cache(self):
        with _patched_tokenizer() as tokenizer:
            _ = tokenizer.all_special_tokens
            _ = tokenizer.all_special_ids
            self.assertTrue(hasattr(tokenizer, "_sglang_cached_special_tokens"))
            self.assertTrue(hasattr(tokenizer, "_sglang_cached_special_ids"))

        self.assertFalse(hasattr(tokenizer, "_sglang_cached_special_tokens"))
        self.assertFalse(hasattr(tokenizer, "_sglang_cached_special_ids"))

    def test_double_patch_is_idempotent(self):
        tokenizer = _load_tokenizer()
        _SpecialTokensCachePatcher.patch(tokenizer)
        _SpecialTokensCachePatcher.patch(tokenizer)

        self.assertTrue(
            getattr(type(tokenizer), "_sglang_special_tokens_patched", False)
        )

        unpatch_tokenizer(tokenizer)

    def test_decode_without_hf_kwargs_uses_native_decode(self):
        tokenizer = _FakeDecodeTokenizer()

        self.assertEqual(
            decode_without_hf_kwargs(tokenizer, [1, 99, 2], True),
            "ab",
        )
        self.assertEqual(
            decode_without_hf_kwargs(tokenizer, [1, 99, 2], False),
            "a<special>b",
        )
        self.assertEqual(tokenizer.decode_calls, [[1, 2], [1, 99, 2]])


class TestEncodePieceFastPath(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            "moonshotai/Kimi-K3",
            revision="f831ab66814297da540d832a5235f8e904f29d06",
            trust_remote_code=True,
        )

    def tearDown(self):
        unpatch_tokenizer(self.tokenizer)

    def test_segment_ids_and_batch_ids_match(self):
        """Shortcuts preserve literals, malformed Unicode and splitter boundaries."""
        tokenizer = self.tokenizer
        specials = list(tokenizer.special_tokens)
        texts = [
            "",
            "hello world",
            "\u4e2d\u6587 café e\u0301 \U0001f680",
            "lone \ud83d surrogate",
            "\udc00",
            "\ud83d\ude80",
            " \n\t" * 15,
            *specials,
            specials[0] + specials[1],
            "prefix " + specials[0] + " suffix",
            *["x " * (n // 2) + "x" * (n % 2) for n in (24999, 25000, 25001)],
            " " * 25001,
        ]
        original = type(tokenizer)._encode_text_piece
        expected = {
            allowed: [original(tokenizer, text, allowed) for text in texts]
            for allowed in (False, True)
        }
        batches = [[""], texts[:9], texts[9:15]]
        expected_batches = [tokenizer(batch)["input_ids"] for batch in batches]
        _EncodePieceFastPathPatcher.patch(tokenizer)
        for allowed in (False, True):
            self.assertEqual(
                expected[allowed],
                [tokenizer._encode_text_piece(text, allowed) for text in texts],
            )
        self.assertEqual(
            expected_batches, [tokenizer(batch)["input_ids"] for batch in batches]
        )

    def test_eligible_segments_bypass_original_encoder(self):
        """Standalone controls and short ordinary text avoid the splitting loop."""
        tokenizer = self.tokenizer
        original = type(tokenizer)._encode_text_piece
        segments = [(token, True) for token in tokenizer.special_tokens]
        segments += [(text, False) for text in ("", "hello", "\ud83d", "x " * 12500)]
        expected = [original(tokenizer, text, allowed) for text, allowed in segments]
        with mock.patch.object(
            type(tokenizer), "_encode_text_piece", autospec=True, side_effect=original
        ) as slow:
            _EncodePieceFastPathPatcher.patch(tokenizer)
            try:
                self.assertEqual(
                    expected[0], tokenizer._encode_text_piece(segments[0][0], True)
                )
                self.assertEqual(slow.call_count, 1)
                for text, allowed in segments[1:]:
                    tokenizer._encode_text_piece(text, allowed)
                slow.reset_mock()
                actual = [
                    tokenizer._encode_text_piece(text, allowed)
                    for text, allowed in segments
                ]
                self.assertEqual(expected, actual)
                self.assertEqual(slow.call_count, 0)
            finally:
                _EncodePieceFastPathPatcher.unpatch(tokenizer)

    def test_configured_special_tokens_preserve_original_ids(self):
        """Configured controls preserve splitting and overlapping-literal precedence."""
        tokenizer = self.tokenizer
        special_id = next(
            value
            for text, value in tokenizer.special_tokens.items()
            if "reserved_token" in text
        )
        texts = [
            "x" * 25000,
            "x" * 25001,
            tokenizer.bos_token + "suffix",
            "synthetic_control",
        ]
        for text in texts:
            with self.subTest(length=len(text)):
                added_tokens = copy.deepcopy(tokenizer.added_tokens_decoder)
                added_tokens[special_id] = AddedToken(text, special=True)
                configured = type(tokenizer)(
                    vocab_file=tokenizer.vocab_file,
                    added_tokens_decoder=added_tokens,
                    **{
                        name: getattr(tokenizer, name)
                        for name in ("bos_token", "eos_token", "unk_token", "pad_token")
                    },
                )
                original = type(configured)._encode_text_piece
                expected = original(configured, text, True)
                _EncodePieceFastPathPatcher.patch(configured)
                try:
                    for _ in range(2):
                        self.assertEqual(
                            expected, configured._encode_text_piece(text, True)
                        )
                    configured.model = tokenizer.model
                    expected_after_replacement = original(configured, text, True)
                    for _ in range(2):
                        self.assertEqual(
                            expected_after_replacement,
                            configured._encode_text_piece(text, True),
                        )
                finally:
                    _EncodePieceFastPathPatcher.unpatch(configured)

    def test_tool_conversation_ids_match(self):
        tokenizer = self.tokenizer
        messages = [{"role": "user", "content": "Inspect the sample data."}]
        for i in range(12):
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{i}",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"key": "example"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{i}",
                        "content": "café \ud83d <|im_end|> sample",
                    },
                ]
            )
        kwargs = dict(tokenize=True, add_generation_prompt=True)
        expected = tokenizer.apply_chat_template(messages, **kwargs)
        with envs.SGLANG_OPT_KIMI_ENCODE_FAST_PATH.override(True):
            patch_tokenizer(tokenizer)
        self.assertEqual(expected, tokenizer.apply_chat_template(messages, **kwargs))

    def test_opt_in_and_unpatch_restore_method(self):
        tokenizer = self.tokenizer
        original = type(tokenizer)._encode_text_piece
        with envs.SGLANG_OPT_KIMI_ENCODE_FAST_PATH.override(False):
            patch_mm_processor_tokenizer(tokenizer)
            patch_tokenizer(tokenizer)
            self.assertIs(type(tokenizer)._encode_text_piece, original)
        with envs.SGLANG_OPT_KIMI_ENCODE_FAST_PATH.override(True):
            with envs.SGLANG_PATCH_TOKENIZER.override(False):
                patch_mm_processor_tokenizer(tokenizer)
                self.assertIs(type(tokenizer)._encode_text_piece, original)
            patch_mm_processor_tokenizer(tokenizer)
            patched = type(tokenizer)._encode_text_piece
            self.assertIsNot(patched, original)
            patch_tokenizer(tokenizer)
            self.assertIs(type(tokenizer)._encode_text_piece, patched)
        unpatch_tokenizer(tokenizer)
        self.assertIs(type(tokenizer)._encode_text_piece, original)

    def test_processor_loading_installs_opted_in_encoder(self):
        tokenizer = self.tokenizer
        expected = tokenizer.encode("sample <|im_end|> text")
        original = type(tokenizer)._encode_text_piece
        processor = SimpleNamespace(tokenizer=tokenizer)
        with (
            mock.patch.object(
                processor_utils.AutoConfig,
                "from_pretrained",
                return_value=SimpleNamespace(model_type="test_vlm", auto_map={}),
            ),
            mock.patch.object(
                processor_utils.AutoProcessor, "from_pretrained", return_value=processor
            ),
            mock.patch.object(tokenizer, "chat_template", "template"),
            envs.SGLANG_OPT_KIMI_ENCODE_FAST_PATH.override(True),
        ):
            loaded = processor_utils.get_processor("test-model")
        self.assertIs(loaded, processor)
        self.assertIsNot(type(loaded.tokenizer)._encode_text_piece, original)
        self.assertEqual(expected, loaded.tokenizer.encode("sample <|im_end|> text"))

    def test_k2_without_segment_method_keeps_encoding(self):
        tokenizer = _load_tokenizer()
        expected = tokenizer.encode("sample <|im_end|> text")
        self.assertFalse(_EncodePieceFastPathPatcher.applies_to(tokenizer))
        with envs.SGLANG_OPT_KIMI_ENCODE_FAST_PATH.override(True):
            patch_tokenizer(tokenizer)
        try:
            self.assertEqual(expected, tokenizer.encode("sample <|im_end|> text"))
        finally:
            unpatch_tokenizer(tokenizer)


def _get_class_attr_ids(cls):
    return {
        n: id(v.fget if isinstance(v, property) else v) for n, v in vars(cls).items()
    }


def _load_tokenizer():
    # The slowness is mainly observed in Kimi
    return AutoTokenizer.from_pretrained(
        "nvidia/Kimi-K2-Thinking-NVFP4", trust_remote_code=True
    )


@contextmanager
def _patched_tokenizer():
    tokenizer = _load_tokenizer()
    _SpecialTokensCachePatcher.patch(tokenizer)
    try:
        yield tokenizer
    finally:
        unpatch_tokenizer(tokenizer)


class _FakeDecodeTokenizer:
    all_special_ids_set = {99}

    def __init__(self):
        self.decode_calls = []

    def decode(self, token_ids):
        token_ids = list(token_ids)
        self.decode_calls.append(token_ids)
        token_text = {1: "a", 2: "b", 99: "<special>"}
        return "".join(token_text[token_id] for token_id in token_ids)


if __name__ == "__main__":
    unittest.main()
