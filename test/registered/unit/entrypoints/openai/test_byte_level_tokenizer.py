import unittest

from sglang.srt.entrypoints.openai.utils import _is_byte_level_tokenizer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _SentencePieceLikeTokenizer:
    """Pieces outside the byte decoder table; counts vocabulary reads."""

    def __init__(self):
        self.vocab_reads = 0

    def get_vocab(self):
        self.vocab_reads += 1
        return {f"▁piece{i}": i for i in range(8)}

    def convert_ids_to_tokens(self, token_id):
        return f"▁piece{token_id}"


class TestIsByteLevelTokenizer(unittest.TestCase):
    def test_a_negative_answer_is_probed_once(self):
        tokenizer = _SentencePieceLikeTokenizer()
        for _ in range(3):
            self.assertFalse(_is_byte_level_tokenizer(tokenizer))
        self.assertEqual(tokenizer.vocab_reads, 1)


if __name__ == "__main__":
    unittest.main()
