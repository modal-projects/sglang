"""Tests for DFlash prefill hidden-state alignment."""

import unittest

import torch

from sglang.srt.speculative.dflash_worker_v2 import (
    _trim_dflash_prefill_hidden_states,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDFlashPrefillPadding(CustomTestCase):
    def test_trims_only_trailing_attention_tp_padding(self):
        for real_tokens in (1, 2, 3, 5, 6, 7):
            with self.subTest(real_tokens=real_tokens):
                padded_tokens = ((real_tokens + 3) // 4) * 4
                hidden_states = torch.arange(padded_tokens * 2).reshape(
                    padded_tokens, 2
                )

                actual = _trim_dflash_prefill_hidden_states(
                    hidden_states,
                    cache_loc_tokens=real_tokens,
                    attention_tp_size=4,
                )

                torch.testing.assert_close(actual, hidden_states[:real_tokens])

    def test_preserves_aligned_hidden_states_without_copying(self):
        hidden_states = torch.arange(16).reshape(8, 2)

        actual = _trim_dflash_prefill_hidden_states(
            hidden_states,
            cache_loc_tokens=8,
            attention_tp_size=4,
        )

        self.assertIs(actual, hidden_states)

    def test_rejects_non_padding_shape_mismatches(self):
        cases = (
            (4, 5, 4),
            (6, 5, 4),
            (9, 5, 4),
            (6, 5, 1),
        )
        for hidden_tokens, cache_loc_tokens, attention_tp_size in cases:
            with self.subTest(
                hidden_tokens=hidden_tokens,
                cache_loc_tokens=cache_loc_tokens,
                attention_tp_size=attention_tp_size,
            ):
                hidden_states = torch.zeros((hidden_tokens, 2))

                with self.assertRaisesRegex(
                    ValueError, "DFLASH cache_loc length mismatch"
                ):
                    _trim_dflash_prefill_hidden_states(
                        hidden_states,
                        cache_loc_tokens=cache_loc_tokens,
                        attention_tp_size=attention_tp_size,
                    )


if __name__ == "__main__":
    unittest.main()
