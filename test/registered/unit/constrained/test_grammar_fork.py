"""Independent current-state grammar forks for verification settlement."""

import unittest

from llguidance import LLTokenizer, grammar_from
from xgrammar import GrammarCompiler, GrammarMatcher, TokenizerInfo, VocabType

from sglang.srt.constrained.llguidance_backend import GuidanceGrammar
from sglang.srt.constrained.outlines_backend import OutlinesGrammar
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.constrained.xgrammar_backend import (
    MAX_ROLLBACK_TOKENS,
    XGrammarGrammar,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Guide:
    def get_next_state(self, state, token):
        return state * 10 + token


def _xgrammar():
    info = TokenizerInfo(
        ["a", "b", "c", "<eos>"], vocab_type=VocabType.RAW, stop_token_ids=[3]
    )
    compiled = GrammarCompiler(info).compile_grammar('root ::= "a" "b"* "c"')
    return XGrammarGrammar(
        GrammarMatcher(compiled, max_rollback_tokens=MAX_ROLLBACK_TOKENS),
        4,
        compiled,
        [3],
    )


class TestGrammarFork(CustomTestCase):
    def test_xgrammar_long_probe_does_not_change_original(self):
        grammar = _xgrammar()
        grammar.accept_token(0)
        forked = grammar.fork()
        for _ in range(MAX_ROLLBACK_TOKENS + 1):
            forked.accept_token(1)
        forked.accept_token(2)
        forked.accept_token(3)

        self.assertTrue(forked.is_terminated())
        self.assertFalse(grammar.is_terminated())
        self.assertEqual(grammar.accepted_tokens, [0])
        self.assertEqual(grammar.current_token, 0)
        # The original can take a different valid continuation from its prefix.
        grammar.accept_token(2)
        grammar.accept_token(3)
        self.assertTrue(grammar.is_terminated())
        self.assertEqual(grammar.accepted_tokens, [0, 2, 3])
        self.assertEqual(len(forked.accepted_tokens), MAX_ROLLBACK_TOKENS + 4)

    def test_xgrammar_fork_preserves_finished_state_and_copy_stays_pristine(self):
        grammar = _xgrammar()
        for token in (0, 2, 3):
            grammar.accept_token(token)
        grammar.finished = True

        forked = grammar.fork()
        self.assertTrue(forked.finished)
        self.assertTrue(forked.is_terminated())
        self.assertEqual(forked.current_token, 3)
        copied = grammar.copy()
        self.assertFalse(copied.is_terminated())
        self.assertEqual(copied.accepted_tokens, [])
        self.assertIsNone(copied.current_token)

    def test_llguidance_fork_has_independent_matcher_and_finish_flag(self):
        tokenizer = LLTokenizer("byte")
        grammar = GuidanceGrammar(tokenizer, grammar_from("regex", "ab*c"))
        grammar.accept_token(ord("a"))
        forked = grammar.fork()
        for token in (ord("b"), ord("c"), tokenizer.eos_tokens[0]):
            forked.accept_token(token)

        self.assertTrue(forked.is_terminated())
        self.assertFalse(grammar.is_terminated())
        self.assertFalse(grammar.ll_matcher.is_stopped())
        self.assertTrue(forked.fork().is_terminated())
        for token in (ord("c"), tokenizer.eos_tokens[0]):
            grammar.accept_token(token)
        self.assertTrue(grammar.is_terminated())

    def test_outlines_fork_keeps_current_state_and_copy_stays_pristine(self):
        grammar = OutlinesGrammar(_Guide(), None)
        grammar.accept_token(1)
        grammar.accept_token(2)
        forked = grammar.fork()
        self.assertEqual(forked.state, 12)
        forked.accept_token(3)
        forked.finished = True

        self.assertEqual(forked.state, 123)
        self.assertEqual(grammar.state, 12)
        self.assertFalse(grammar.finished)
        grammar.accept_token(4)
        self.assertEqual(grammar.state, 124)
        self.assertEqual(grammar.copy().state, 0)
        self.assertTrue(forked.fork().finished)

    def test_reasoner_fork_preserves_partial_terminator_and_inner_state(self):
        grammar = ReasonerGrammarObject(_xgrammar(), think_end_ids=[9, 8])
        grammar.maybe_init_reasoning(True)
        grammar.accept_token(9)
        forked = grammar.fork()
        for token in (8, 0, 1, 2, 3):
            forked.accept_token(token)

        self.assertTrue(forked.is_terminated())
        self.assertFalse(grammar.is_terminated())
        self.assertEqual(grammar._matched_think_end_tokens, 1)
        self.assertEqual(grammar._thinking_match_history, [0])
        self.assertEqual(grammar.current_token, 9)
        self.assertEqual(grammar.grammar.accepted_tokens, [])
        for token in (8, 0, 2, 3):
            grammar.accept_token(token)
        self.assertTrue(grammar.is_terminated())
        self.assertEqual(grammar.grammar.accepted_tokens, [0, 2, 3])
        self.assertEqual(forked.grammar.accepted_tokens, [0, 1, 2, 3])

    def test_reasoner_fork_without_inner_grammar_keeps_independent_history(self):
        grammar = ReasonerGrammarObject(None, think_end_ids=[9, 8])
        grammar.maybe_init_reasoning(True)
        grammar.accept_token(9)
        forked = grammar.fork()
        forked.accept_token(8)
        forked.finished = True

        self.assertTrue(forked.finished)
        self.assertFalse(grammar.finished)
        self.assertEqual(grammar._thinking_match_history, [0])
        self.assertEqual(grammar.tokens_after_end, -1)
        self.assertEqual(forked.tokens_after_end, 0)


if __name__ == "__main__":
    unittest.main()
