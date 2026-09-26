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

The predicate reads the req-side CPU plan (mamba_lazy_spec_scatter_pos)
captured by mamba_lazy_spec_prepare rather than the device ping-pong buffer:
under the overlap loop the prefill result pass runs after the in-flight
verify is launched, so a .item() on the device buffer could stall on the
forward. With no captured plan the predicate falls back to the conservative
in-window rule alone.
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
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
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
    scatter_pos=None,
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
    req.kv.mamba_lazy_spec_scatter_pos = scatter_pos
    if ping_pong is not None:
        req.kv.mamba_ping_pong_track_buffer = torch.tensor(
            list(ping_pong), dtype=torch.int64
        )
        req.kv.mamba_next_track_idx = 0
    return req


class _ItemFreeBuffer:
    """Ping-pong buffer double that fails the test on any device read.

    numel() is CPU metadata; .item() and indexing stand in for the
    synchronizing device reads the predicate must not perform.
    """

    def __init__(self, values):
        self._values = list(values)

    def numel(self):
        return len(self._values)

    def item(self):
        raise AssertionError("device read (.item()) reached")

    def __getitem__(self, idx):
        raise AssertionError("device read (__getitem__) reached")


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
        # The verify prepare allocated the pending slot, so its captured plan
        # scatters into position 1: no keep-slot alias.
        req = _make_req(
            max_new_tokens=32, ping_pong=(KEEP_SLOT, PENDING_SLOT), scatter_pos=1
        )
        batch = _make_prefill_batch(req)
        self.assertFalse(self._predicate(req, batch, _make_processor()))

    def test_alias_with_plan_at_keep_and_in_window(self):
        # The captured plan points at the keep slot (pending slot was empty
        # and its allocation failed): alias, decided without any device read.
        req = _make_req(
            max_new_tokens=32, ping_pong=(KEEP_SLOT, PENDING_SLOT), scatter_pos=0
        )
        req.kv.mamba_ping_pong_track_buffer = _ItemFreeBuffer((KEEP_SLOT, PENDING_SLOT))
        batch = _make_prefill_batch(req)
        self.assertTrue(self._predicate(req, batch, _make_processor()))

    def test_no_alias_when_plan_points_at_pending(self):
        # Buffer fixture has the pending slot empty, but the captured plan
        # scatters into the pending slot: no alias. Proves the CPU plan
        # field, not the device buffer, drives the predicate.
        req = _make_req(max_new_tokens=32, scatter_pos=1)
        req.kv.mamba_ping_pong_track_buffer = _ItemFreeBuffer((KEEP_SLOT, -1))
        batch = _make_prefill_batch(req)
        self.assertFalse(self._predicate(req, batch, _make_processor()))

    def test_no_alias_when_plan_at_keep_outside_window(self):
        req = _make_req(
            max_new_tokens=32, committed_len=OUT_OF_WINDOW_LEN, scatter_pos=0
        )
        batch = _make_prefill_batch(req)
        self.assertFalse(self._predicate(req, batch, _make_processor()))

    def test_fallback_without_plan_performs_no_device_read(self):
        # No plan captured for this request: the conservative in-window rule
        # decides alone, still without touching the device buffer.
        req = _make_req(max_new_tokens=32)
        req.kv.mamba_ping_pong_track_buffer = _ItemFreeBuffer((KEEP_SLOT, -1))
        batch = _make_prefill_batch(req)
        self.assertTrue(self._predicate(req, batch, _make_processor()))

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
        req = _make_req(
            max_new_tokens=1, ping_pong=(KEEP_SLOT, PENDING_SLOT), scatter_pos=1
        )
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
        req = _make_req(
            max_new_tokens=32, ping_pong=(KEEP_SLOT, PENDING_SLOT), scatter_pos=1
        )
        batch = _make_prefill_batch(req)
        self._run_prefill(req, batch)
        maybe_cache_unfinished_req.assert_called_once()
        release_kv_cache.assert_not_called()

    def test_unfinished_decision_performs_no_device_read(
        self, release_kv_cache, maybe_cache_unfinished_req
    ):
        # Plan says the verify scatters into the pending slot, so the
        # donation proceeds; the whole prefill pass performs no device read.
        req = _make_req(max_new_tokens=32, scatter_pos=1)
        req.kv.mamba_ping_pong_track_buffer = _ItemFreeBuffer((KEEP_SLOT, -1))
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


class TestMambaLazySpecPreparePlanMirror(unittest.TestCase):
    """mamba_lazy_spec_prepare mirrors its per-req scatter plan onto the req."""

    @staticmethod
    def _pool(alloc_result):
        pool = MagicMock()
        pool.mamba_allocator.alloc.return_value = alloc_result
        pool.set_mamba_ping_pong_slot.side_effect = lambda req, idx, value: (
            req.kv.mamba_ping_pong_track_buffer.__setitem__(idx, value)
        )
        return pool

    def _prepare(self, req, pool):
        batch = ScheduleBatch(reqs=[req])
        batch.req_to_token_pool = pool
        batch.mamba_lazy_spec_prepare(GRID, DRAFT_TOKENS)
        self.assertEqual(
            req.kv.mamba_lazy_spec_scatter_pos,
            batch.mamba_lazy_spec_track_positions_cpu[0],
        )
        return req.kv.mamba_lazy_spec_scatter_pos

    def test_pending_empty_alloc_success_plans_pending(self):
        req = _make_req(max_new_tokens=32)
        pool = self._pool(torch.tensor([PENDING_SLOT], dtype=torch.int64))
        pos = self._prepare(req, pool)
        self.assertEqual(pos, 1)
        self.assertEqual(req.kv.mamba_ping_pong_track_buffer[1].item(), PENDING_SLOT)

    def test_pending_already_allocated_plans_pending(self):
        req = _make_req(max_new_tokens=32, ping_pong=(KEEP_SLOT, PENDING_SLOT))
        pool = self._pool(None)
        pos = self._prepare(req, pool)
        self.assertEqual(pos, 1)
        pool.mamba_allocator.alloc.assert_not_called()

    def test_pending_empty_alloc_failure_plans_keep(self):
        req = _make_req(max_new_tokens=32)
        pool = self._pool(None)
        pos = self._prepare(req, pool)
        self.assertEqual(pos, 0)
        self.assertEqual(req.kv.mamba_ping_pong_track_buffer[1].item(), -1)

    def test_outside_window_plans_keep_without_alloc(self):
        req = _make_req(max_new_tokens=32, committed_len=OUT_OF_WINDOW_LEN)
        pool = self._pool(None)
        pos = self._prepare(req, pool)
        self.assertEqual(pos, 0)
        pool.mamba_allocator.alloc.assert_not_called()


class TestScatterPosLifecycle(unittest.TestCase):
    """The req-side plan mirror is cleared with the rest of the mamba state."""

    def test_reset_for_retract_clears_plan(self):
        req = _make_req(max_new_tokens=32, scatter_pos=1)
        # reset_for_retract expects the KV row to be released already.
        req.kv.req_pool_idx = None
        req.reset_for_retract()
        self.assertIsNone(req.kv.mamba_lazy_spec_scatter_pos)

    def test_free_mamba_cache_clears_plan(self):
        req = _make_req(max_new_tokens=32, scatter_pos=1)
        req.kv.mamba_pool_idx = torch.tensor([3], dtype=torch.int64)
        pool = MagicMock()
        pool.enable_mamba_extra_buffer = True
        pool.enable_mamba_extra_buffer_lazy = True
        pool.mamba_ping_pong_track_buffer_size = 2
        pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.tensor(
            [[KEEP_SLOT, PENDING_SLOT]], dtype=torch.int64
        )
        HybridReqToTokenPool.free_mamba_cache(
            pool, req, mamba_ping_pong_track_buffer_to_keep=0
        )
        self.assertIsNone(req.kv.mamba_lazy_spec_scatter_pos)
        self.assertIsNone(req.kv.mamba_ping_pong_track_buffer)


if __name__ == "__main__":
    unittest.main()
