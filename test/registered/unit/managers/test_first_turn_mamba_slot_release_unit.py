import time
import types
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.session.session_controller import Session
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _waiting_req(rid, mamba_pool_idx, req_pool_idx=None, kv_allocated_len=0):
    session = Session(0, "session-a", streaming=True)
    recv_req = TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", [1, 2]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(temperature=0, max_new_tokens=4),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=True,
        session_params=SessionParams(id=session.session_id),
        priority=5,
    )
    req = session.create_req(recv_req, tokenizer=None, vocab_size=16)
    req.kv = ReqKvInfo(
        req_pool_idx=req_pool_idx,
        kv_allocated_len=kv_allocated_len,
        mamba_pool_idx=mamba_pool_idx,
    )
    req.time_stats.wait_queue_entry_time = time.perf_counter() - 100
    return req


def _scheduler_stub(waiting_queue):
    mamba_allocator = Mock()
    tree_cache = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(mamba_allocator=mamba_allocator),
        supports_mamba=lambda: True,
        finish=Mock(),
    )
    stub = SimpleNamespace(
        chunked_req=None,
        _pending_chunked_abort_req=None,
        mm_receiver=None,
        waiting_queue=waiting_queue,
        enable_hicache_storage=False,
        disaggregation_mode=DisaggregationMode.NULL,
        dllm_config=None,
        grammar_manager=SimpleNamespace(abort_requests=Mock()),
        ps=SimpleNamespace(pp_size=1),
        running_batch=SimpleNamespace(reqs=[]),
        last_batch=SimpleNamespace(reqs=[]),
        ipc_channels=SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=Mock())
        ),
        beam_coordinator=SimpleNamespace(retire_group=Mock()),
        tree_cache=tree_cache,
    )
    stub._release_aborted_request = types.MethodType(
        Scheduler._release_aborted_request, stub
    )
    stub._release_dropped_waiting_req_mm_inputs = types.MethodType(
        Scheduler._release_dropped_waiting_req_mm_inputs, stub
    )
    stub._release_dropped_waiting_req_mamba_slot = types.MethodType(
        Scheduler._release_dropped_waiting_req_mamba_slot, stub
    )
    stub.collect_inflight_reqs = types.MethodType(Scheduler.collect_inflight_reqs, stub)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.__dict__.update(vars(stub))
    return scheduler, mamba_allocator


class TestFirstTurnMambaSlotRelease(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="scheduler")

    def test_waiting_timeout_releases_first_turn_mamba_slot(self):
        req = _waiting_req("turn-1", mamba_pool_idx=torch.tensor([3]))
        stub, mamba_allocator = _scheduler_stub([req])
        send_output = stub.ipc_channels.send_to_tokenizer.send_output

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(50.0):
            aborts = Scheduler._poll_timeout_aborts(stub)

        self.assertEqual(len(aborts), 1)
        self.assertEqual(aborts[0].rid, req.rid)
        Scheduler.abort_request(stub, aborts[0])

        self.assertEqual(stub.waiting_queue, [])
        send_output.assert_called_once()
        mamba_allocator.free.assert_called_once()
        self.assertEqual(mamba_allocator.free.call_args[0][0].flatten().tolist(), [3])
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertFalse(req.session.has_unfinished_request())
        self.assertTrue(req.finished())

    def test_priority_eviction_releases_first_turn_mamba_slot(self):
        candidate = _waiting_req("turn-1", mamba_pool_idx=torch.tensor([3]))
        stub, mamba_allocator = _scheduler_stub([candidate])
        stub.max_queued_requests = 1
        stub.enable_priority_scheduling = True
        stub.schedule_low_priority_values_first = True
        recv_req = SimpleNamespace(rid="new-req", priority=0)

        aborted = Scheduler._abort_on_queued_limit(stub, recv_req)

        self.assertFalse(aborted)
        self.assertEqual(stub.waiting_queue, [])
        mamba_allocator.free.assert_called_once()
        self.assertEqual(mamba_allocator.free.call_args[0][0].flatten().tolist(), [3])
        self.assertIsNone(candidate.kv.mamba_pool_idx)
        self.assertFalse(candidate.session.has_unfinished_request())
        self.assertTrue(candidate.finished())

    def test_priority_disabled_reject_releases_first_turn_mamba_slot(self):
        """A priority request rejected because priority scheduling is
        disabled never enters a queue: it must get the same dropped-request
        cleanup as a queue-full reject, or the session stays busy and the
        req-owned early alloc leaks."""
        req = _waiting_req("turn-1", mamba_pool_idx=torch.tensor([3]))
        stub, mamba_allocator = _scheduler_stub([])
        stub.enable_priority_scheduling = False
        stub.abort_on_priority_when_disabled = True
        send_output = stub.ipc_channels.send_to_tokenizer.send_output

        admitted = Scheduler._set_or_validate_priority(stub, req)

        assert admitted is False
        mamba_allocator.free.assert_called_once()
        self.assertEqual(mamba_allocator.free.call_args[0][0].flatten().tolist(), [3])
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertFalse(req.session.has_unfinished_request())
        self.assertIsNotNone(req.finished_reason)
        self.assertEqual(req.finished_reason.status_code, 503)
        send_output.assert_called_once()

    def test_priority_eviction_keeps_slot_owned_by_session(self):
        candidate = _waiting_req(
            "turn-2",
            mamba_pool_idx=torch.tensor([3]),
            req_pool_idx=7,
            kv_allocated_len=8,
        )
        stub, mamba_allocator = _scheduler_stub([candidate])
        stub.max_queued_requests = 1
        stub.enable_priority_scheduling = True
        stub.schedule_low_priority_values_first = True
        recv_req = SimpleNamespace(rid="new-req", priority=0)

        aborted = Scheduler._abort_on_queued_limit(stub, recv_req)

        self.assertFalse(aborted)
        self.assertEqual(stub.waiting_queue, [])
        mamba_allocator.free.assert_not_called()
        self.assertEqual(candidate.kv.mamba_pool_idx.tolist(), [3])


if __name__ == "__main__":
    unittest.main()
