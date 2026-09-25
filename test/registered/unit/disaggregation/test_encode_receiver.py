"""Unit tests for the encode-disaggregation receiver."""

import asyncio
import threading
import time
import unittest
from array import array
from functools import partial
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from sglang.srt.disaggregation.encoder.receiver import (
    MMReceiverBase,
    WaitingMMRequestStatus,
    WaitingRDMARequest,
    WaitingZmqRequest,
    WaitingZmqRequestGrpc,
    _ReceiveRegistrationRunner,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq, EncoderDispatchErrorReq
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.observability import metrics_collector
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_registration_request(request_cls):
    request = request_cls.__new__(request_cls)
    request.rid = "registration-test"
    request.registration_runner = _ReceiveRegistrationRunner(
        "test-encoder-receive-registration"
    )
    request.registration_future = None
    request.registration_error = None
    request.registration_lock = threading.Lock()
    request.status = WaitingMMRequestStatus.PENDING
    request.error_msg = None
    request.error_code = None
    request.err_type = None
    request.embedding_pool = None
    request.embeddings_buffer = None
    request.recv_embedding_data = None
    request._pool_slot_id = None
    request._mm_finalizer = None
    request.recv_socket = None
    request.recv_req = SimpleNamespace(rid=request.rid)
    request.num_items_assigned = {Modality.IMAGE: [1]}
    request.encoder_urls = ["http://encoder"]
    request.host_name = "127.0.0.1"
    request.receive_count = 1
    request.embedding_port = 12345
    return request


def _cancel_registration(request):
    future = request.registration_future
    request.release_resources()
    deadline = time.monotonic() + 1
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert future.cancelled()


class BlockingResponse:
    def __init__(self, started):
        self.started = started

    async def __aenter__(self):
        self.started.set()
        await asyncio.Event().wait()

    async def __aexit__(self, *args):
        return False


class BlockingSession:
    started = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        return BlockingResponse(self.started)


class FailingResponse:
    async def __aenter__(self):
        raise ConnectionError("encoder unavailable")

    async def __aexit__(self, *args):
        return False


class FailingSession(BlockingSession):
    def post(self, *args, **kwargs):
        return FailingResponse()


class TestReceiveRegistration(CustomTestCase):
    def test_http_registration_does_not_block_scheduler(self):
        started = threading.Event()
        BlockingSession.started = started
        request = _make_registration_request(WaitingZmqRequest)
        with patch(
            "sglang.srt.disaggregation.encoder.receiver.aiohttp.ClientSession",
            BlockingSession,
        ):
            scheduler_call = threading.Thread(
                target=request.send_encode_request, daemon=True
            )
            scheduler_call.start()
            self.assertTrue(started.wait(timeout=1))
            scheduler_call.join(timeout=0.1)

        self.assertFalse(scheduler_call.is_alive())
        self.assertEqual(request.status, WaitingMMRequestStatus.PENDING)
        _cancel_registration(request)

    def test_grpc_registration_does_not_block_scheduler(self):
        started = threading.Event()

        async def blocking_registration(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        request = _make_registration_request(WaitingZmqRequestGrpc)
        with patch(
            "sglang.srt.disaggregation.encoder.receiver._grpc_scheduler_receive_url",
            blocking_registration,
        ):
            scheduler_call = threading.Thread(
                target=request.send_encode_request, daemon=True
            )
            scheduler_call.start()
            self.assertTrue(started.wait(timeout=1))
            scheduler_call.join(timeout=0.1)

        self.assertFalse(scheduler_call.is_alive())
        self.assertEqual(request.status, WaitingMMRequestStatus.PENDING)
        _cancel_registration(request)

    def test_failure_is_request_local(self):
        request = _make_registration_request(WaitingZmqRequest)
        with patch(
            "sglang.srt.disaggregation.encoder.receiver.aiohttp.ClientSession",
            FailingSession,
        ):
            request.send_encode_request()
            deadline = time.monotonic() + 1
            while request.status == WaitingMMRequestStatus.PENDING:
                self.assertLess(time.monotonic(), deadline)
                request._try_recv_mm_data()
                time.sleep(0.01)

        self.assertEqual(request.status, WaitingMMRequestStatus.FAIL)
        self.assertEqual(request.error_code, HTTPStatus.BAD_GATEWAY)
        self.assertIn("encoder unavailable", request.error_msg)


class TestEncodeReceiverRequestConstruction(CustomTestCase):
    def test_early_dispatch_error_waits_for_scheduler_request(self):
        encode_finished = threading.Event()
        scheduler_dispatch_ready = threading.Event()
        reported = []
        failure = EncoderDispatchErrorReq(
            rid="request-1",
            error_msg="encoder unavailable",
            error_code=HTTPStatus.BAD_GATEWAY,
        )

        async def fail_encode(**kwargs):
            encode_finished.set()
            return failure

        receiver = SimpleNamespace(encode=fail_encode)
        worker = threading.Thread(
            target=MMReceiverBase._run_encode_in_thread,
            args=(
                receiver,
                failure.rid,
                [],
                "encode",
                {},
                [],
                None,
                scheduler_dispatch_ready,
                reported.append,
            ),
        )
        worker.start()

        self.assertTrue(encode_finished.wait(timeout=1))
        worker.join(timeout=0.05)
        self.assertTrue(worker.is_alive())
        self.assertEqual(reported, [])

        scheduler_dispatch_ready.set()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(reported, [failure])

    def test_dispatch_error_fails_only_owning_wait(self):
        class WaitingRequest:
            def __init__(self, rid):
                self.rid = rid
                self.recv_req = SimpleNamespace(rid=rid)
                self.status = WaitingMMRequestStatus.PENDING
                self.error_msg = None
                self.error_code = None
                self.err_type = None
                self.start_time = 0

            def _try_recv_mm_data(self):
                pass

            def _fail_and_release(self, error_msg, error_code=None):
                self.error_msg = error_msg
                self.error_code = error_code
                self.status = WaitingMMRequestStatus.FAIL

            def release_resources(self):
                pass

            def close_recv_socket(self):
                pass

        owner = WaitingRequest("request-1")
        other = WaitingRequest("request-2")
        receiver = SimpleNamespace(
            waiting_list=[owner, other],
            waiting_by_rid={owner.rid: owner, other.rid: other},
            scheduler_recv_socket=None,
            wait_timeout=float("inf"),
            tp_group=SimpleNamespace(cpu_group=object()),
            _drain_scheduler_embeddings=lambda: None,
            _sync_fail_info_across_tp=lambda request: None,
            create_req=lambda request: request,
        )
        dispatch_error = EncoderDispatchErrorReq(
            rid=owner.rid,
            error_msg="bad media",
            error_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

        with patch("torch.distributed.all_reduce"):
            _, abort_reqs = MMReceiverBase._process_waiting_requests(
                receiver, [dispatch_error], waiting_cls=None
            )

        self.assertEqual(owner.status, WaitingMMRequestStatus.FAIL)
        self.assertEqual(owner.error_msg, dispatch_error.error_msg)
        self.assertEqual(owner.error_code, dispatch_error.error_code)
        self.assertEqual(other.status, WaitingMMRequestStatus.PENDING)
        self.assertEqual([req.rid for req, _, _, _ in abort_reqs], [owner.rid])

    def test_extra_key_and_cache_salt_are_forwarded(self):
        scheduler = SimpleNamespace(
            model_config=SimpleNamespace(hf_eos_token_id={2}, vocab_size=128),
            disaggregation_mode=DisaggregationMode.NULL,
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            metrics_collector=None,
            dllm_config=None,
            tokenizer=object(),
        )
        receiver = SimpleNamespace(scheduler=scheduler)
        recv_req = SimpleNamespace(
            rid="request-1",
            input_text="hello",
            input_ids=array("q", [1, 2]),
            sampling_params=SamplingParams(max_new_tokens=1),
            return_logprob=False,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            lora_id=None,
            input_embeds=None,
            custom_logit_processor=None,
            require_reasoning=False,
            return_hidden_states=False,
            return_routed_experts=False,
            routed_experts_start_len=0,
            bootstrap_host=None,
            bootstrap_port=None,
            bootstrap_room=None,
            routed_dp_rank=None,
            disagg_prefill_dp_rank=None,
            priority=None,
            extra_key="classification",
            cache_salt="tenant-a",
            http_worker_ipc=None,
        )

        req = MMReceiverBase.create_req(receiver, recv_req)

        self.assertEqual(req.extra_key, "classification")
        self.assertEqual(req.cache_salt, "tenant-a")

    def test_rdma_worker_error_is_released_on_scheduler_thread(self):
        scheduler_thread = threading.get_ident()

        class ThreadCheckedSocket:
            closed_by = None

            def close(self):
                self.closed_by = threading.get_ident()

        recv_socket = ThreadCheckedSocket()
        request = WaitingRDMARequest.__new__(WaitingRDMARequest)
        request.rid = "request-1"
        request.status = WaitingMMRequestStatus.PENDING
        request.error_msg = None
        request.error_code = None
        request.recv_socket = recv_socket
        request._receive_error = None
        request._receive_error_lock = threading.Lock()
        request._buffer_lock = threading.Lock()
        request._terminal = False
        request._receive_running = False
        request.registration_future = None
        request.embeddings_buffer = None
        request._pool_slot_id = None
        request.embedding_pool = None
        request._mm_finalizer = None

        worker = threading.Thread(
            target=lambda: asyncio.run(
                request._check_encoder_responses(
                    [ConnectionError("encoder unavailable")], "/send"
                )
            )
        )
        worker.start()
        worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertEqual(request.status, WaitingMMRequestStatus.PENDING)
        self.assertIsNone(request.recv_socket.closed_by)

        request._try_recv_mm_data()

        self.assertEqual(request.status, WaitingMMRequestStatus.FAIL)
        self.assertIsNone(request.recv_socket)
        self.assertTrue(request._terminal)
        self.assertEqual(recv_socket.closed_by, scheduler_thread)

    def test_tp_peer_failure_closes_local_receive_socket(self):
        class WaitingRequest:
            rid = "request-1"
            recv_req = SimpleNamespace(rid=rid)
            status = WaitingMMRequestStatus.PENDING
            error_msg = "peer failed"
            error_code = None
            err_type = None
            start_time = 0
            released = False
            closed = False

            def _try_recv_mm_data(self):
                pass

            def release_resources(self):
                self.released = True

            def close_recv_socket(self):
                self.closed = True

        waiting_req = WaitingRequest()
        receiver = SimpleNamespace(
            waiting_list=[waiting_req],
            waiting_by_rid={waiting_req.rid: waiting_req},
            scheduler_recv_socket=None,
            wait_timeout=float("inf"),
            tp_group=SimpleNamespace(cpu_group=object()),
            _drain_scheduler_embeddings=lambda: None,
            _sync_fail_info_across_tp=lambda request: None,
            create_req=lambda request: request,
        )

        def force_peer_failure(status, **kwargs):
            status.fill_(WaitingMMRequestStatus.FAIL)

        with patch("torch.distributed.all_reduce", force_peer_failure):
            _, abort_reqs = MMReceiverBase._process_waiting_requests(
                receiver, [], waiting_cls=None
            )

        self.assertTrue(waiting_req.released)
        self.assertTrue(waiting_req.closed)
        self.assertEqual(len(abort_reqs), 1)


class TestEncoderTimeoutOutcome(CustomTestCase):
    def test_internal_timeout_preserves_http_status_and_failure_origin(self):
        """Only a receiver wait timeout is an engine fault; external 408 stays invalid."""
        for internal_timeout in (False, True):
            with self.subTest(internal_timeout=internal_timeout):
                request = SimpleNamespace(rid="timeout", return_logprob=False)
                waiting = SimpleNamespace(
                    rid=request.rid,
                    recv_req=request,
                    status=WaitingMMRequestStatus.PENDING
                    if internal_timeout
                    else WaitingMMRequestStatus.FAIL,
                    error_msg="upstream timeout",
                    error_code=HTTPStatus.REQUEST_TIMEOUT,
                    err_type=None,
                    start_time=0 if internal_timeout else time.time(),
                    _try_recv_mm_data=lambda: None,
                    release_resources=lambda: None,
                    close_recv_socket=lambda: None,
                )
                receiver = SimpleNamespace(
                    waiting_list=[waiting],
                    waiting_by_rid={waiting.rid: waiting},
                    scheduler_recv_socket=None,
                    wait_timeout=10,
                    tp_group=SimpleNamespace(cpu_group=None),
                    _drain_scheduler_embeddings=lambda: None,
                    _sync_fail_info_across_tp=lambda request: None,
                    create_req=lambda request: request,
                )
                receiver.process_waiting_requests = lambda reqs: (
                    MMReceiverBase._process_waiting_requests(
                        receiver, reqs, waiting_cls=None
                    )
                )
                emitted = []
                scheduler = SimpleNamespace(
                    ps=SimpleNamespace(pp_rank=0),
                    mm_receiver=receiver,
                    stream_output=lambda reqs, logprob: emitted.extend(
                        req.finished_reason.to_json() for req in reqs
                    ),
                )
                with (
                    get_context().override_server_args(
                        language_only=True, encoder_transfer_backend="zmq_to_scheduler"
                    ),
                    patch("torch.distributed.all_reduce"),
                ):
                    self.assertEqual(
                        SchedulerRequestReceiver._apply_mm_receiver(scheduler, []), []
                    )
                self.assertEqual(len(emitted), 1)
                self.assertEqual(emitted[0]["status_code"], HTTPStatus.REQUEST_TIMEOUT)
                self.assertEqual(
                    emitted[0]["err_type"],
                    "encoder_timeout" if internal_timeout else None,
                )
                self.assertEqual(
                    metrics_collector.finished_outcome(emitted[0]),
                    "engine_fault" if internal_timeout else "invalid_request",
                )
                self.assertEqual(receiver.waiting_list, [])
                self.assertEqual(receiver.waiting_by_rid, {})


class TestEncoderCancellationOutcome(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            language_only=True, encoder_transfer_backend="zmq_to_scheduler"
        )
        override.install()
        self.addCleanup(override.restore)
        self.registry = CollectorRegistry()
        self.labels = {"model_name": "test-model"}

        class Collector(metrics_collector.TokenizerMetricsCollector):
            _counter_cls = partial(Counter, registry=self.registry)
            _gauge_cls = partial(Gauge, registry=self.registry)
            _histogram_cls = partial(Histogram, registry=self.registry)

        self.collector = Collector(labels=self.labels)

    def make_receiver(self):
        waiting = _make_registration_request(WaitingZmqRequest)
        waiting.start_time = time.time()
        waiting.recv_req.return_logprob = False
        receiver = SimpleNamespace(
            waiting_list=[waiting],
            waiting_by_rid={waiting.rid: waiting},
            scheduler_recv_socket=None,
            wait_timeout=10,
            tp_size=1,
            tp_group=SimpleNamespace(cpu_group=None),
            _drain_scheduler_embeddings=lambda: None,
            create_req=lambda request: request,
        )
        receiver._sync_fail_info_across_tp = lambda request: (
            MMReceiverBase._sync_fail_info_across_tp(receiver, request)
        )
        receiver.process_waiting_requests = lambda reqs: (
            MMReceiverBase._process_waiting_requests(receiver, reqs, waiting_cls=None)
        )
        return receiver, waiting

    def finish(self, receiver):
        emitted = []
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=0),
            mm_receiver=receiver,
            stream_output=lambda reqs, logprob: emitted.extend(
                req.finished_reason.to_json() for req in reqs
            ),
        )
        SchedulerRequestReceiver._apply_mm_receiver(scheduler, [])
        self.assertEqual(len(emitted), 1)
        reason = msgspec.json.decode(msgspec.json.encode(emitted[0]))
        self.collector.observe_finished_outcome(
            self.labels, metrics_collector.finished_outcome(reason), 0, 0
        )
        return reason

    def sample(self, outcome):
        return self.registry.get_sample_value(
            "sglang:finished_requests_by_outcome_total",
            {**self.labels, "outcome": outcome},
        )

    def test_user_abort_is_distinct_from_encoder_bad_request(self):
        """A cancellation's existing HTTP 400 must not become an invalid request."""
        for cancelled in (True, False):
            with self.subTest(cancelled=cancelled):
                receiver, waiting = self.make_receiver()
                before_abort = self.sample("abort")
                before_invalid = self.sample("invalid_request")
                if cancelled:
                    MMReceiverBase.abort_waiting_requests(
                        receiver, AbortReq(rid=waiting.rid)
                    )
                else:
                    waiting._fail_and_release("Aborted by user", error_code=400)
                with patch("torch.distributed.all_reduce"):
                    reason = self.finish(receiver)
                self.assertEqual(self.sample("abort"), before_abort + int(cancelled))
                self.assertEqual(
                    self.sample("invalid_request"), before_invalid + int(not cancelled)
                )
                self.assertEqual(reason["status_code"], 400)
                self.assertEqual(reason["err_type"], "cancelled" if cancelled else None)
                self.assertEqual(reason["message"], "Aborted by user")

    def test_peer_cancellation_retains_marker_at_streaming_rank(self):
        """TP error propagation must carry the cancellation tag with its status."""
        receiver, waiting = self.make_receiver()
        peer_receiver, peer = self.make_receiver()
        MMReceiverBase.abort_waiting_requests(peer_receiver, AbortReq(rid=peer.rid))
        receiver.tp_size = 2
        receiver.tp_group.all_gather_object = lambda local: [
            local,
            (peer.error_msg, peer.error_code, peer.err_type),
        ]

        def peer_failed(status, **kwargs):
            status.fill_(WaitingMMRequestStatus.FAIL)

        with patch("torch.distributed.all_reduce", peer_failed):
            reason = self.finish(receiver)
        self.assertEqual(self.sample("abort"), 1)
        self.assertEqual(self.sample("invalid_request"), 0)
        self.assertEqual(reason["status_code"], 400)
        self.assertEqual(reason["err_type"], "cancelled")

    def test_peer_bad_request_cannot_reuse_another_ranks_cancel_marker(self):
        """A selected peer error must carry its own type, including no tag."""
        receiver, waiting = self.make_receiver()
        MMReceiverBase.abort_waiting_requests(receiver, AbortReq(rid=waiting.rid))
        receiver.tp_size = 2
        receiver.tp_group.all_gather_object = lambda local: [
            local,
            ("bad media", 400, None),
        ]
        with patch("torch.distributed.all_reduce"):
            reason = self.finish(receiver)
        self.assertEqual(self.sample("invalid_request"), 1)
        self.assertEqual(self.sample("abort"), 0)
        self.assertEqual(reason["err_type"], None)
        self.assertEqual(reason["message"], "bad media")


if __name__ == "__main__":
    unittest.main()
