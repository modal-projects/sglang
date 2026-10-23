"""A spec mixed batch with grammar decode rows must not overlap the previous batch.

The decode rows of a mixed batch are sampled in the target extend, not in verify(),
so the grammar barrier never runs for them. Their bitmask is built from the FSM as
it stands, which misses the previous decode batch's committed tokens while that
result is still queued; the sampled token is then rejected by accept_token and the
request is aborted mid-stream.
"""

import unittest
from collections import deque
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(grammar):
    return SimpleNamespace(grammar=object() if grammar else None)


def _batch(mode, decoding_reqs=None, has_grammar=True):
    batch = SimpleNamespace(
        forward_mode=mode,
        spec_algorithm=SpeculativeAlgorithm.DFLASH,
        decoding_reqs=decoding_reqs,
        has_grammar=has_grammar,
    )
    batch.grammar_needs_sync = lambda: (
        batch.has_grammar and not batch.spec_algorithm.supports_grammar_overlap()
    )
    return batch


def _scheduler(queued):
    return SimpleNamespace(
        require_mlp_sync=False,
        result_queue=deque([object()] * queued),
        _mixed_batch_has_grammar_decode_rows=(
            Scheduler._mixed_batch_has_grammar_decode_rows
        ),
    )


def _disable(batch, queued=1):
    return Scheduler.is_disable_overlap_for_batch(
        _scheduler(queued), batch, last_batch=None
    )


class TestMixedGrammarOverlapSync(unittest.TestCase):
    def test_mixed_batch_with_grammar_decode_row_syncs(self):
        batch = _batch(ForwardMode.MIXED, [_req(False), _req(True)])
        self.assertTrue(_disable(batch))

    def test_mixed_batch_without_grammar_decode_rows_overlaps(self):
        batch = _batch(ForwardMode.MIXED, [_req(False)])
        self.assertFalse(_disable(batch))

    def test_mixed_batch_with_nothing_queued_overlaps(self):
        batch = _batch(ForwardMode.MIXED, [_req(True)])
        self.assertFalse(_disable(batch, queued=0))

    def test_grammar_overlap_decode_keeps_overlapping(self):
        # DFLASH advances the FSM inside verify() through the grammar barrier.
        self.assertFalse(_disable(_batch(ForwardMode.DECODE)))

    def test_plain_extend_overlaps(self):
        self.assertFalse(_disable(_batch(ForwardMode.EXTEND, None)))


if __name__ == "__main__":
    unittest.main()
