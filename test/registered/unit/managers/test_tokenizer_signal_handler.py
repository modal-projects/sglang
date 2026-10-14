"""Unit tests for SignalHandler in srt/managers/tokenizer_manager.py"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import SignalHandler  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSignalHandlerDrain(CustomTestCase):
    def test_sigterm_starts_drain_without_watchdog_escalation(self):
        watchdog = Mock()
        tokenizer_manager = SimpleNamespace(
            _subprocess_watchdog=watchdog, gracefully_exit=False
        )
        SignalHandler(tokenizer_manager).sigterm_handler()
        watchdog.stop.assert_called_once_with()
        self.assertTrue(tokenizer_manager.gracefully_exit)

    def test_sigterm_without_watchdog_still_drains(self):
        tokenizer_manager = SimpleNamespace(
            _subprocess_watchdog=None, gracefully_exit=False
        )
        SignalHandler(tokenizer_manager).sigterm_handler()
        self.assertTrue(tokenizer_manager.gracefully_exit)


if __name__ == "__main__":
    unittest.main()
