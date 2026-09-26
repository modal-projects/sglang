"""Prefill-time mamba keep-slot donation guard tests.

Under extra_buffer_lazy + speculative decoding with the overlap scheduler, the
first verify after a lazy final prefill is launched before the prefill result
is processed. When the pending ping-pong slot allocation fails at verify
prepare, the verify plan falls back to the keep slot as its scatter
destination. If prefill result processing then donates/inserts that physical
slot to the radix tree under the prefill tracked depth C, the verify commits
the crossing state T > C into the same slot and the tree mislabels newer
state as C.

These tests pin the prefill-side alias predicate
(_mamba_lazy_keep_may_be_written_in_flight) and the two decisions it gates:
the finishing release insert flag and the unfinished donation call.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

# page = chunk = interval = 8 keeps every checkpoint grid at 8; with two draft
# tokens the in-window predicate fires at seq_len 12 but not at seq_len 8.
GRID = 8
DRAFT_TOKENS = 2
IN_WINDOW_LEN = 12
OUT_OF_WINDOW_LEN = 8
KEEP_SLOT = 5
PENDING_SLOT = 9


def _make_req(
    *,
    max_new_tokens: int,
    committed_len: int = IN_WINDOW_LEN,
    ping_pong=(KEEP_SLOT, -1),
) -> Req:
    sampling_params = SamplingParams(max_new_tokens=max_new_tokens)
    sampling_params.normalize(None)
    req = Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2]),
        sampling_params=sampling_params,
        vocab_size=128,
    )
    req.kv.kv_committed_len = committed_len
    req.kv.req_pool_idx = 0
    req.kv.mamba_last_track_seqlen = GRID
    if ping_pong is not None:
        req.kv.mamba_ping_pong_track_buffer = torch.tensor(
            list(ping_pong), dtype=torch.int64
        )
        req.kv.mamba_next_track_idx = 0
    return req


def _make_prefill_batch(req: Req, *, spec_active: bool = True) -> ScheduleBatch:
    batch = ScheduleBatch(reqs=[req])
    batch.tree_cache = SimpleNamespace(page_size=GRID)
    batch.device = "cpu"
    batch.model_config = SimpleNamespace(is_encoder_decoder=False)
    batch.enable_overlap = True
    batch.spec_algorithm = SimpleNamespace(is_none=lambda: not spec_active)
    batch.sampling_info = SimpleNamespace(
        penalizer_orchestrator=SimpleNamespace(is_required=False)
    )
    batch.hisparse_coordinator = None
    batch.decoding_reqs = []
    batch.prefill_stats = None
    return batch


def _make_processor() -> SchedulerBatchResultProcessor:
    metrics_reporter = MagicMock()
    metrics_reporter.num_generated_tokens = 0
    processor = SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=True,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=MagicMock(),
        tree_cache=SimpleNamespace(page_size=GRID),
        hisparse_coordinator=None,
        beam_coordinator=MagicMock(),
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=metrics_reporter,
        draft_worker=None,
        model_worker=MagicMock(),
        logprob_result_processor=None,
        output_streamer=MagicMock(),
        abort_request=lambda *args, **kwargs: None,
    )
    return processor


def _make_prefill_result(next_token: int = 4) -> GenerationBatchResult:
    return GenerationBatchResult(
        logits_output=SimpleNamespace(
            hidden_states=None,
            customized_info=None,
            sampling_mask_output=None,
            next_token_sampling_mask_status=None,
        ),
        next_token_ids=torch.tensor([next_token], dtype=torch.int64),
        speculative_num_draft_tokens=0,
    )


def _publish(*, strategy: str = "extra_buffer_lazy"):
    return get_context().override_server_args(
        mamba_radix_cache_strategy=strategy,
        mamba_track_interval=GRID,
        _mamba_cache_chunk_size=GRID,
        speculative_num_draft_tokens=DRAFT_TOKENS,
    )


class TestKeepMayBeWrittenInFlightPredicate(unittest.TestCase):
    def _predicate(self, req, batch, processor):
        with _publish():
            return processor._mamba_lazy_keep_may_be_written_in_flight(req, batch)

    def test_alias_when_pending_empty_and_in_window(self):
        req = _make_req(max_new_tokens=32)
        batch = _make_prefill_batch(req)
        self.assertTrue(self._predicate(req, batch, _make_processor()))

    def test_no_alias_when_pending_slot_allocated(self):
        req = _make_req(max_new_tokens=32, ping_pong=(KEEP_SLOT, PENDING_SLOT))
        batch = _make_prefill_batch(req)
        self.assertFalse(self._predicate(req, batch, _make_processor()))

    def test_no_alias_outside_crossing_window(self):
        req = _make_req(max_new_tokens=32, committed_len=OUT_OF_WINDOW_LEN)
        batch = _make_prefill_batch(req)
        self.assertFalse(self._predicate(req, batch, _make_processor()))

    def test_no_alias_without_speculation(self):
        req = _make_req(max_new_tokens=32)
        batch = _make_prefill_batch(req, spec_active=False)
        self.assertFalse(self._predicate(req, batch, _make_processor()))

    def test_no_alias_without_lazy_strategy(self):
        req = _make_req(max_new_tokens=32)
        batch = _make_prefill_batch(req)
        processor = _make_processor()
        with _publish(strategy="extra_buffer"):
            self.assertFalse(
                processor._mamba_lazy_keep_may_be_written_in_flight(req, batch)
            )

    def test_no_alias_without_ping_pong_buffer(self):
        req = _make_req(max_new_tokens=32, ping_pong=None)
        batch = _make_prefill_batch(req)
        self.assertFalse(self._predicate(req, batch, _make_processor()))

    def test_no_alias_with_single_slot_buffer(self):
        # Non-overlap scheduling keeps one ping-pong slot: donation substitutes
        # the replacement before any verify plan can capture the keep slot.
        req = _make_req(max_new_tokens=32, ping_pong=(KEEP_SLOT,))
        batch = _make_prefill_batch(req)
        self.assertFalse(self._predicate(req, batch, _make_processor()))


@patch(
    "sglang.srt.managers.scheduler_components.batch_result_processor.maybe_cache_unfinished_req"
)
@patch(
    "sglang.srt.managers.scheduler_components.batch_result_processor.release_kv_cache"
)
class TestPrefillDonationDecision(unittest.TestCase):
    def _run_prefill(self, req, batch):
        processor = _make_processor()
        with _publish():
            processor.process_batch_result_prefill(batch, _make_prefill_result())

    def test_finishing_suppresses_insert_under_alias(
        self, release_kv_cache, maybe_cache_unfinished_req
    ):
        req = _make_req(max_new_tokens=1)
        batch = _make_prefill_batch(req)
        self._run_prefill(req, batch)
        release_kv_cache.assert_called_once()
        self.assertFalse(release_kv_cache.call_args.kwargs["is_insert"])
        maybe_cache_unfinished_req.assert_not_called()

    def test_finishing_inserts_when_pending_allocated(
        self, release_kv_cache, maybe_cache_unfinished_req
    ):
        req = _make_req(max_new_tokens=1, ping_pong=(KEEP_SLOT, PENDING_SLOT))
        batch = _make_prefill_batch(req)
        self._run_prefill(req, batch)
        release_kv_cache.assert_called_once()
        self.assertTrue(release_kv_cache.call_args.kwargs["is_insert"])
        maybe_cache_unfinished_req.assert_not_called()

    def test_unfinished_skips_donation_under_alias(
        self, release_kv_cache, maybe_cache_unfinished_req
    ):
        req = _make_req(max_new_tokens=32)
        batch = _make_prefill_batch(req)
        self._run_prefill(req, batch)
        maybe_cache_unfinished_req.assert_not_called()
        release_kv_cache.assert_not_called()

    def test_unfinished_donates_when_pending_allocated(
        self, release_kv_cache, maybe_cache_unfinished_req
    ):
        req = _make_req(max_new_tokens=32, ping_pong=(KEEP_SLOT, PENDING_SLOT))
        batch = _make_prefill_batch(req)
        self._run_prefill(req, batch)
        maybe_cache_unfinished_req.assert_called_once()
        release_kv_cache.assert_not_called()

    def test_unfinished_donates_outside_crossing_window(
        self, release_kv_cache, maybe_cache_unfinished_req
    ):
        req = _make_req(max_new_tokens=32, committed_len=OUT_OF_WINDOW_LEN)
        batch = _make_prefill_batch(req)
        self._run_prefill(req, batch)
        maybe_cache_unfinished_req.assert_called_once()
        release_kv_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
