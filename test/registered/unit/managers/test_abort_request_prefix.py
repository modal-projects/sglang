"""Unit tests for abort-by-rid-prefix.

Tokenizer side: with prefix=True, the rid is treated as a prefix. Matching
logical request lifecycles are marked aborted before dispatch, and the AbortReq
is forwarded with prefix=True for work already owned by the scheduler.

Scheduler side: matching is prefix-based (``rid.startswith``) regardless of
the flag, because batch requests derive child rids as ``f"{rid}_{i}"``.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_tokenizer_manager(rids=(), tokenizer_worker_num=1) -> TokenizerManager:
    """Create a TokenizerManager with mocked dependencies, bypassing __init__."""
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = MagicMock()
    tm.server_args.tokenizer_worker_num = tokenizer_worker_num
    tm.enable_metrics = False
    tm.rid_to_state = {
        rid: SimpleNamespace(
            abort_requested=False,
            obj=SimpleNamespace(parallel_sample_num=1),
        )
        for rid in rids
    }
    tm.logical_rid_to_child_rids = {}
    tm.child_rid_to_logical_rid = {}
    tm.send_to_scheduler = MagicMock()
    tm.tokenizer_ipc_name = None
    # The IPC boundary: sock_send's wire format varies (pickle/msgpack), so
    # tests observe dispatched objects here instead of on the zmq socket.
    tm._dispatch_to_scheduler = MagicMock()
    return tm


def _sent_req(tm) -> AbortReq:
    tm._dispatch_to_scheduler.assert_called_once()
    return tm._dispatch_to_scheduler.call_args.args[0]


class TestAbortRequestPrefix(CustomTestCase):
    def test_prefix_match_sends_abort(self):
        tm = _make_tokenizer_manager(rids=["job-1-seq-0", "job-1-seq-1", "other"])
        tm.abort_request(rid="job-1", prefix=True)

        req = _sent_req(tm)
        self.assertEqual(req.rid, "job-1")
        self.assertTrue(req.prefix)
        self.assertFalse(req.abort_all)

    def test_prefix_without_match_is_ignored(self):
        tm = _make_tokenizer_manager(rids=["other-1", "other-2"])
        tm.abort_request(rid="job-1", prefix=True)

        tm._dispatch_to_scheduler.assert_not_called()

    def test_prefix_requires_full_prefix_not_substring(self):
        tm = _make_tokenizer_manager(rids=["seq-job-1"])
        tm.abort_request(rid="job-1", prefix=True)

        tm._dispatch_to_scheduler.assert_not_called()

    def test_exact_match_still_works_without_prefix(self):
        tm = _make_tokenizer_manager(rids=["job-1"])
        tm.abort_request(rid="job-1")

        req = _sent_req(tm)
        self.assertEqual(req.rid, "job-1")
        self.assertFalse(req.prefix)

    def test_exact_mode_does_not_prefix_match(self):
        # rid is only a prefix of a tracked request; exact mode must ignore it.
        tm = _make_tokenizer_manager(rids=["job-1-seq-0"])
        tm.abort_request(rid="job-1")

        tm._dispatch_to_scheduler.assert_not_called()

    def test_empty_rid_is_ignored(self):
        # An empty rid would prefix-match every request on the scheduler.
        tm = _make_tokenizer_manager(rids=["job-1"])
        tm.abort_request(rid="", prefix=True)

        tm._dispatch_to_scheduler.assert_not_called()

    def test_multi_tokenizer_worker_skips_local_check(self):
        # With >1 tokenizer workers, rid_to_state is not authoritative; the
        # abort must be forwarded even if this worker tracks no matching rid.
        tm = _make_tokenizer_manager(rids=[], tokenizer_worker_num=2)
        tm.abort_request(rid="job-1", prefix=True)

        req = _sent_req(tm)
        self.assertEqual(req.rid, "job-1")
        self.assertTrue(req.prefix)


class TestAbortLogicalRequestLifecycle(CustomTestCase):
    def test_prefix_abort_marks_matching_logical_states(self):
        tm = _make_tokenizer_manager(rids=["job-1-seq-0", "job-1-seq-1", "other"])
        tm.abort_request(rid="job-1", prefix=True)

        self.assertTrue(tm.rid_to_state["job-1-seq-0"].abort_requested)
        self.assertTrue(tm.rid_to_state["job-1-seq-1"].abort_requested)
        self.assertFalse(tm.rid_to_state["other"].abort_requested)
        self.assertEqual(_sent_req(tm).rid, "job-1")

    def test_exact_abort_marks_only_exact_state(self):
        tm = _make_tokenizer_manager(rids=["job-1", "job-1-seq-0"])
        tm.abort_request(rid="job-1")

        self.assertTrue(tm.rid_to_state["job-1"].abort_requested)
        self.assertFalse(tm.rid_to_state["job-1-seq-0"].abort_requested)

    def test_abort_all_marks_every_logical_state(self):
        tm = _make_tokenizer_manager(rids=["a", "b"])
        tm.abort_request(abort_all=True)

        self.assertTrue(tm.rid_to_state["a"].abort_requested)
        self.assertTrue(tm.rid_to_state["b"].abort_requested)

    def test_prefix_abort_fans_out_non_namespaced_parallel_children(self):
        tm = _make_tokenizer_manager(rids=["job-1", "choice-a", "choice-b"])
        tm.logical_rid_to_child_rids = {"job-1": {"choice-a", "choice-b"}}
        tm.child_rid_to_logical_rid = {
            "choice-a": "job-1",
            "choice-b": "job-1",
        }

        tm.abort_request(rid="job-", prefix=True)

        requests = [call.args[0] for call in tm._dispatch_to_scheduler.call_args_list]
        self.assertEqual(
            {request.rid for request in requests}, {"job-", "choice-a", "choice-b"}
        )
        self.assertTrue(tm.rid_to_state["job-1"].abort_requested)


class FakeReq:
    def __init__(self, rid: str):
        self.rid = rid
        self.mamba_pool_idx = None
        self.to_finish = None

    def finished(self) -> bool:
        return False


def _make_scheduler(waiting_rids=(), running_rids=(), chunked_rid=None):
    sched = SimpleNamespace()
    sched.chunked_req = FakeReq(chunked_rid) if chunked_rid is not None else None
    sched.waiting_queue = [FakeReq(rid) for rid in waiting_rids]
    sched.enable_hicache_storage = False
    sched.disaggregation_mode = DisaggregationMode.NULL
    sched.dllm_config = None
    sched.grammar_manager = MagicMock()
    sched.ps = SimpleNamespace(pp_size=1)
    sched.running_batch = SimpleNamespace(reqs=[FakeReq(rid) for rid in running_rids])
    sched.last_batch = None
    sched.cur_batch = None
    sched.ipc_channels = MagicMock()
    return sched


class TestSchedulerAbortMatching(CustomTestCase):
    """Scheduler-side matching semantics for AbortReq (see io_struct.AbortReq:
    always ``rid.startswith``, so batch children ``f"{rid}_{i}"`` are covered)."""

    def test_prefix_abort_isolates_namespaces(self):
        sched = _make_scheduler(
            waiting_rids=["A::1", "A::2", "B::1"],
            running_rids=["A::3", "B::2"],
        )
        Scheduler.abort_request(sched, AbortReq(rid="A::", prefix=True))

        self.assertEqual([req.rid for req in sched.waiting_queue], ["B::1"])
        running = {req.rid: req for req in sched.running_batch.reqs}
        self.assertIsInstance(running["A::3"].to_finish, FINISH_ABORT)
        self.assertIsNone(running["B::2"].to_finish)
        # Every waiting-queue abort echoes back to the tokenizer for cleanup.
        aborted_rids = {
            call.args[0].rid
            for call in sched.ipc_channels.send_to_tokenizer.send_output.call_args_list
        }
        self.assertEqual(aborted_rids, {"A::1", "A::2"})

    def test_matching_is_prefix_based_even_without_prefix_flag(self):
        # Pre-existing scheduler semantics: batch requests derive child rids as
        # f"{rid}_{i}", so an exact-mode abort for the parent must cover them.
        sched = _make_scheduler(waiting_rids=["job-1_0", "job-1_1", "job-2_0"])
        Scheduler.abort_request(sched, AbortReq(rid="job-1", prefix=False))

        self.assertEqual([req.rid for req in sched.waiting_queue], ["job-2_0"])

    def test_chunked_request_is_prefix_matched(self):
        sched = _make_scheduler(chunked_rid="A::9")
        Scheduler.abort_request(sched, AbortReq(rid="A::", prefix=True))

        self.assertIs(sched._pending_chunked_abort_req, sched.chunked_req)

    def test_abort_all_covers_everything(self):
        sched = _make_scheduler(waiting_rids=["A::1"], running_rids=["B::1"])
        Scheduler.abort_request(sched, AbortReq(abort_all=True))

        self.assertEqual(sched.waiting_queue, [])
        self.assertIsInstance(sched.running_batch.reqs[0].to_finish, FINISH_ABORT)


def _make_disagg_req(rid: str, pending_bootstrap: bool = False) -> FakeReq:
    req = FakeReq(rid)
    req.pending_bootstrap = pending_bootstrap
    req.disagg_kv_sender = Mock()
    return req


def _make_prefill_scheduler(waiting_rids=(), bootstrap_rids=(), inflight_rids=()):
    sched = _make_scheduler()
    sched.disaggregation_mode = DisaggregationMode.PREFILL
    sched.waiting_queue = [
        _make_disagg_req(rid, pending_bootstrap=True) for rid in waiting_rids
    ]
    sched.req_to_metadata_buffer_idx_allocator = MagicMock()
    sched.disagg_prefill_bootstrap_queue = SimpleNamespace(
        queue=[_make_disagg_req(rid) for rid in bootstrap_rids]
    )
    sched.disagg_prefill_inflight_queue = [
        _make_disagg_req(rid) for rid in inflight_rids
    ]
    return sched


def _make_decode_req(rid: str) -> SimpleNamespace:
    return SimpleNamespace(req=FakeReq(rid), kv_receiver=Mock())


def _make_retracted_req(rid: str) -> SimpleNamespace:
    return SimpleNamespace(rid=rid, kv_cache_cpu=object())


def _make_decode_scheduler(
    waiting_rids=(), prealloc_rids=(), transfer_rids=(), retracted_rids=()
):
    sched = _make_scheduler(waiting_rids=waiting_rids)
    sched.disaggregation_mode = DisaggregationMode.DECODE
    sched.tree_cache = MagicMock()
    sched.disagg_decode_prealloc_queue = SimpleNamespace(
        queue=[_make_decode_req(rid) for rid in prealloc_rids],
        retracted_queue=[_make_retracted_req(rid) for rid in retracted_rids],
    )
    sched.disagg_decode_transfer_queue = SimpleNamespace(
        queue=[_make_decode_req(rid) for rid in transfer_rids]
    )
    return sched


def _echoed_rids(sched) -> set:
    return {
        call.args[0].rid
        for call in sched.ipc_channels.send_to_tokenizer.send_output.call_args_list
    }


class TestSchedulerDisaggPrefillAbort(CustomTestCase):
    """PREFILL-side disaggregation abort matching: the bootstrap and in-flight
    queues hold requests the waiting queue no longer tracks, and the waiting
    queue itself must release the metadata buffer slot and abort a
    still-bootstrapping KV sender."""

    def test_bootstrap_and_inflight_queues_prefix_matched(self):
        sched = _make_prefill_scheduler(
            bootstrap_rids=["A::1", "B::1"], inflight_rids=["A::2", "B::2"]
        )
        Scheduler.abort_request(sched, AbortReq(rid="A::", prefix=True))

        bootstrap = {r.rid: r for r in sched.disagg_prefill_bootstrap_queue.queue}
        bootstrap["A::1"].disagg_kv_sender.abort.assert_called_once()
        bootstrap["B::1"].disagg_kv_sender.abort.assert_not_called()
        inflight = {r.rid: r for r in sched.disagg_prefill_inflight_queue}
        inflight["A::2"].disagg_kv_sender.abort.assert_called_once()
        inflight["B::2"].disagg_kv_sender.abort.assert_not_called()

    def test_waiting_queue_releases_metadata_buffer(self):
        sched = _make_prefill_scheduler(waiting_rids=["A::1", "B::1"])
        with patch(
            "sglang.srt.managers.scheduler.maybe_release_metadata_buffer"
        ) as release:
            Scheduler.abort_request(sched, AbortReq(rid="A::", prefix=True))

        self.assertEqual([r.rid for r in sched.waiting_queue], ["B::1"])
        release.assert_called_once()
        released_req, allocator = release.call_args.args
        self.assertEqual(released_req.rid, "A::1")
        self.assertIs(allocator, sched.req_to_metadata_buffer_idx_allocator)
        self.assertEqual(_echoed_rids(sched), {"A::1"})

    def test_waiting_queue_pending_bootstrap_gates_sender_abort(self):
        pending = _make_disagg_req("A::1", pending_bootstrap=True)
        bootstrapped = _make_disagg_req("A::2", pending_bootstrap=False)
        sched = _make_prefill_scheduler()
        sched.waiting_queue = [pending, bootstrapped]
        with patch("sglang.srt.managers.scheduler.maybe_release_metadata_buffer"):
            Scheduler.abort_request(sched, AbortReq(rid="A::", prefix=True))

        pending.disagg_kv_sender.abort.assert_called_once()
        bootstrapped.disagg_kv_sender.abort.assert_not_called()

    def test_abort_all_covers_prefill_queues(self):
        sched = _make_prefill_scheduler(bootstrap_rids=["A::1"], inflight_rids=["B::1"])
        Scheduler.abort_request(sched, AbortReq(abort_all=True))

        sched.disagg_prefill_bootstrap_queue.queue[
            0
        ].disagg_kv_sender.abort.assert_called_once()
        sched.disagg_prefill_inflight_queue[
            0
        ].disagg_kv_sender.abort.assert_called_once()


class TestSchedulerDisaggDecodeAbort(CustomTestCase):
    """DECODE-side disaggregation abort matching: prealloc/transfer queues
    abort their KV receivers, the retracted queue frees CPU KV cache and
    echoes the abort back to the tokenizer, and waiting-queue requests
    release their preallocated KV cache."""

    def test_prealloc_and_transfer_queues_prefix_matched(self):
        sched = _make_decode_scheduler(
            prealloc_rids=["A::1", "B::1"], transfer_rids=["A::2", "B::2"]
        )
        Scheduler.abort_request(sched, AbortReq(rid="A::", prefix=True))

        prealloc = {d.req.rid: d for d in sched.disagg_decode_prealloc_queue.queue}
        prealloc["A::1"].kv_receiver.abort.assert_called_once()
        prealloc["B::1"].kv_receiver.abort.assert_not_called()
        transfer = {d.req.rid: d for d in sched.disagg_decode_transfer_queue.queue}
        transfer["A::2"].kv_receiver.abort.assert_called_once()
        transfer["B::2"].kv_receiver.abort.assert_not_called()

    def test_retracted_queue_frees_cpu_cache_and_echoes(self):
        sched = _make_decode_scheduler(retracted_rids=["A::1", "B::1"])
        aborted = sched.disagg_decode_prealloc_queue.retracted_queue[0]
        Scheduler.abort_request(sched, AbortReq(rid="A::", prefix=True))

        self.assertEqual(
            [d.rid for d in sched.disagg_decode_prealloc_queue.retracted_queue],
            ["B::1"],
        )
        self.assertFalse(hasattr(aborted, "kv_cache_cpu"))
        self.assertEqual(_echoed_rids(sched), {"A::1"})

    def test_waiting_queue_releases_kv_cache(self):
        sched = _make_decode_scheduler(waiting_rids=["A::1", "B::1"])
        with patch("sglang.srt.managers.scheduler.release_kv_cache") as release:
            Scheduler.abort_request(sched, AbortReq(rid="A::", prefix=True))

        self.assertEqual([r.rid for r in sched.waiting_queue], ["B::1"])
        release.assert_called_once()
        self.assertEqual(release.call_args.args[0].rid, "A::1")

    def test_exact_mode_is_still_prefix_matched_in_disagg_queues(self):
        # Same load-bearing semantics as the non-disagg queues: batch children
        # derive rids as f"{rid}_{i}", so exact-mode must cover them here too.
        sched = _make_decode_scheduler(prealloc_rids=["job-1_0"])
        Scheduler.abort_request(sched, AbortReq(rid="job-1", prefix=False))

        sched.disagg_decode_prealloc_queue.queue[
            0
        ].kv_receiver.abort.assert_called_once()

    def test_abort_all_covers_decode_queues(self):
        sched = _make_decode_scheduler(
            prealloc_rids=["A::1"], transfer_rids=["B::1"], retracted_rids=["C::1"]
        )
        Scheduler.abort_request(sched, AbortReq(abort_all=True))

        sched.disagg_decode_prealloc_queue.queue[
            0
        ].kv_receiver.abort.assert_called_once()
        sched.disagg_decode_transfer_queue.queue[
            0
        ].kv_receiver.abort.assert_called_once()
        self.assertEqual(sched.disagg_decode_prealloc_queue.retracted_queue, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
