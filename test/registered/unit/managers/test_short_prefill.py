import unittest
from types import SimpleNamespace

from sglang.srt.managers.short_prefill import add_chunk_with_short_prefill_budget
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(uncached):
    return SimpleNamespace(origin_input_ids=[0] * uncached, num_matched_prefix_tokens=0)


class _Adder:
    def __init__(self, budget):
        self.rem_chunk_tokens = budget
        self.dllm_config = None
        self.page_size = 64
        self.chunk_budgets = []

    def add_chunked_req(self, req):
        self.chunk_budgets.append(self.rem_chunk_tokens)
        self.rem_chunk_tokens = 0
        return req


class TestShortPrefillReservation(CustomTestCase):
    def _run(self, waiting, scan_waiting):
        adder = _Adder(8192)
        add_chunk_with_short_prefill_budget(
            adder,
            "long",
            waiting,
            threshold=2048,
            chunk_size=4096,
            batch_size=8192,
            scan_waiting=scan_waiting,
        )
        return adder

    def test_reserves_page_aligned_room_for_short_prefills(self):
        long, short = _req(100_000), _req(1000)
        waiting = [long, short]
        adder = self._run(waiting, scan_waiting=True)
        self.assertEqual(adder.chunk_budgets, [8192 - 1024])
        self.assertEqual(adder.rem_chunk_tokens, 1024)
        self.assertEqual(waiting, [short, long])

    def test_stops_at_first_long_request_without_scan(self):
        waiting = [_req(100_000), _req(1000)]
        adder = self._run(waiting, scan_waiting=False)
        self.assertEqual(adder.chunk_budgets, [8192])

    def test_continuing_chunk_keeps_its_minimum(self):
        waiting = [_req(2048), _req(2048), _req(2048)]
        adder = self._run(waiting, scan_waiting=True)
        self.assertEqual(adder.chunk_budgets, [4096])


if __name__ == "__main__":
    unittest.main()
