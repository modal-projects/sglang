"""Failed multimodal attachment must retire only the failing request's media."""

import asyncio
import dataclasses
import unittest
from array import array
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    SessionParams,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.managers.scheduler import Scheduler, _MultimodalInputProcessingError
from sglang.srt.multimodal.transport import cuda_ipc, memory_pool
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session, SessionController
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _recv(mm_inputs, *, embedding=False, session=None, parent=None, rid="incoming"):
    common = dict(
        rid=rid,
        input_text=None,
        input_ids=array("q", [2]),
        mm_inputs=mm_inputs,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1),
        time_stats=APIServerReqTimeStats(),
    )
    if embedding:
        return TokenizedEmbeddingReqInput(**common)
    return TokenizedGenerateReqInput(
        **common,
        input_embeds=None,
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        session_params=(
            SessionParams(id=session.session_id, rid=parent) if session else None
        ),
    )


class _RecordingSession(Session):
    def __init__(self, *, streaming, distinct_container=False):
        super().__init__(0, session_id="synthetic-session", streaming=streaming)
        self.distinct_container = distinct_container
        self.created = []

    def create_req(self, *args, **kwargs):
        req = super().create_req(*args, **kwargs)
        if self.distinct_container and req.multimodal_inputs is not None:
            req.multimodal_inputs = dataclasses.replace(req.multimodal_inputs)
        self.created.append(req)
        return req


def _scheduler(session=None):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_session_radix_cache = False
    scheduler.model_config = SimpleNamespace(hf_eos_token_id=[], vocab_size=32)
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
    scheduler.tokenizer = None
    scheduler.dllm_config = None
    scheduler._maybe_namespace_elastic_radix_cache = Mock()
    scheduler.init_req_max_new_tokens = Mock()
    scheduler._add_request_to_queue = Mock()
    scheduler.max_req_input_len = 128
    scheduler.pad_input_ids_func = lambda ids, _: ids
    scheduler._mm_processor = None
    scheduler.grammar_manager = SimpleNamespace(
        process_req_with_grammar=lambda _: False
    )
    scheduler.session_controller = {session.session_id: session} if session else {}
    return scheduler


class TestSchedulerMultimodalAttachment(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(speculative_algorithm=None)
        override.install()
        self.addCleanup(override.restore)
        self.raw = torch.tensor([3, 7], dtype=torch.float32).view(torch.uint8)
        self.writes = []
        self.proxy_number = 0
        stream = Mock()
        real_empty = torch.empty

        def cpu_empty(*args, **kwargs):
            kwargs["device"] = "cpu"
            return real_empty(*args, **kwargs)

        patches = (
            patch.object(cuda_ipc, "_pool_acknowledged_generations", {}),
            patch.object(cuda_ipc, "_pool_imported_generations", {}),
            patch.object(cuda_ipc, "_pool_storage_cache", {}),
            patch.object(
                cuda_ipc,
                "_open_pooled_storage_uncached",
                side_effect=lambda _: torch.zeros(
                    128, dtype=torch.uint8
                ).untyped_storage(),
            ),
            patch.object(cuda_ipc, "_release_ipc_export"),
            patch.object(torch, "empty", side_effect=cpu_empty),
            patch.object(torch.cuda, "device", side_effect=lambda *_: nullcontext()),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(memory_pool, "stream_wait_value32"),
            patch.object(cuda_ipc, "resolve_consumer_rank", return_value=0),
            patch.object(
                cuda_ipc,
                "stream_write_value32",
                side_effect=lambda *args: self.writes.append(args),
            ),
            patch(
                "sglang.srt.managers.schedule_batch.get_parallel",
                return_value=SimpleNamespace(tp_rank=0),
            ),
        )
        for context in patches:
            context.start()
            self.addCleanup(context.stop)

    def _incoming(self):
        self.proxy_number += 1
        pool_id = f"synthetic-pool-{self.proxy_number}"
        handle = (0, pool_id, 128, 0, "counter", self.proxy_number, "event", False)
        proxy = cuda_ipc.CudaIpcTensorTransportProxy(
            data=self.raw,
            info_data=self.raw.view(torch.float32),
            pool_ipc_handle=handle,
            pool_ipc_handles=(handle,),
            pool_id=pool_id,
            pool_byte_offset=64,
            ready_byte_offset=0,
            ack_byte_offset=4,
            generation=1,
            total_consumer_count=1,
            use_pool_handle_cache=True,
        )
        item = MultimodalDataItem(
            modality=Modality.IMAGE, feature=proxy, offsets=[(0, 0)]
        )
        return MultimodalInputs(mm_items=[item]), proxy

    def _committed_parent(self, session):
        parent = session.create_req(
            _recv(None, session=session, rid="parent"), tokenizer=None, vocab_size=32
        )
        shared = MultimodalDataItem(
            modality=Modality.IMAGE, feature=torch.tensor([11]), offsets=[(0, 0)]
        )
        parent.multimodal_inputs = MultimodalInputs(mm_items=[shared])
        parent.finished_reason = FINISH_LENGTH(1)
        parent._refresh_fill_ids()
        if session.streaming:
            session.finish_req(parent)
        session.release_finished_req_mm_inputs(parent)
        return parent, shared

    def _queued_history(self, session, rid, *, parent=None, media=None):
        recv = _recv(None, session=session, parent=parent, rid=rid)
        recv.input_ids = array("q", [17])
        req = session.create_req(recv, tokenizer=None, vocab_size=32)
        if media is not None:
            req.multimodal_inputs = media
        req.finished_reason = FINISH_ABORT("queue full before prefill")
        session.release_finished_req_mm_inputs(req)
        return req

    def _check_early_session_rejection(self, failure, retained_sibling):
        self.writes.clear()
        session = Session(0, session_id="synthetic-session", streaming=False)
        historical, historical_proxy = self._incoming()
        prior = self._queued_history(session, "history", media=historical)
        unrelated, unrelated_proxy = self._incoming()
        untouched = self._queued_history(session, "untouched", media=unrelated)
        sibling = (
            self._queued_history(session, "sibling", parent=prior.rid)
            if retained_sibling
            else None
        )
        incoming, incoming_proxy = self._incoming()
        recv = _recv(incoming, session=session, parent=prior.rid, rid=prior.rid)
        scheduler = _scheduler(session)
        if failure == "conversion":
            scheduler._get_multimodal_inputs = Mock(
                side_effect=_MultimodalInputProcessingError("invalid image conversion")
            )
        elif failure == "dflash":
            scheduler.spec_algorithm = SpeculativeAlgorithm.DFLASH
            scheduler.enable_overlap = True
            recv.return_hidden_states = True
        elif failure == "uno":
            scheduler.spec_algorithm = SpeculativeAlgorithm.UNO
            recv.sampling_params.min_p = 0.1
        scheduler.handle_generate_request(
            recv,
            mm_input_error="invalid image metadata"
            if failure == "input_error"
            else None,
        )
        scheduler._add_request_to_queue.assert_called_once()
        rejected = scheduler._add_request_to_queue.call_args.args[0]
        self.assertIsInstance(rejected.to_finish, FINISH_ABORT)
        expected_message = {
            "input_error": "invalid image metadata",
            "conversion": "invalid image conversion",
            "dflash": "DFLASH speculative decoding does not support return_hidden_states",
            "uno": "UNO speculative decoding does not support min_p",
        }[failure]
        self.assertIn(expected_message, rejected.to_finish.message)
        self.assertIsNone(recv.mm_inputs)
        self.assertIsNone(rejected.multimodal_inputs)
        self.assertIsNone(prior.multimodal_inputs)
        self.assertTrue(incoming_proxy._consumer_acknowledged)
        self.assertEqual(historical_proxy._consumer_acknowledged, not retained_sibling)
        self.assertFalse(unrelated_proxy._consumer_acknowledged)
        self.assertIs(untouched.multimodal_inputs, unrelated)
        self.assertEqual(len(self.writes), 1 if retained_sibling else 2)
        self.assertNotIn(rejected.rid, session.req_nodes)
        self.assertFalse(session.has_unfinished_request())
        if sibling is not None:
            self.assertIs(sibling.multimodal_inputs, historical)
            self.assertIs(historical.mm_items[0].feature, historical_proxy)
        rejected.finished_reason = rejected.to_finish
        rejected.to_finish = None
        scheduler._release_dropped_waiting_req_mm_inputs(rejected)
        scheduler._release_dropped_waiting_req_mm_inputs(rejected)
        self.assertEqual(len(self.writes), 1 if retained_sibling else 2)

        scheduler = _scheduler(session)
        surviving = sibling if sibling is not None else untouched
        scheduler.handle_generate_request(
            _recv(None, session=session, parent=surviving.rid, rid="next")
        )
        accepted = scheduler._add_request_to_queue.call_args.args[0]
        self.assertIsNone(accepted.to_finish)
        self.assertIs(accepted.multimodal_inputs, surviving.multimodal_inputs)
        self.assertTrue(session.has_unfinished_request())
        accepted.finished_reason = FINISH_ABORT("queue full before prefill")
        session.release_finished_req_mm_inputs(accepted)
        controller = SessionController(Mock())
        controller.sessions[session.session_id] = session
        controller._close(session.session_id)
        self.assertEqual(controller.sessions, {})
        for proxy in (historical_proxy, incoming_proxy, unrelated_proxy):
            self.assertTrue(proxy._consumer_acknowledged)
        self.assertEqual(len(self.writes), 3)

    def test_same_rid_early_rejection_releases_only_unretained_reservations(self):
        """Rejecting an overwritten turn must retire media before its pointer is cleared."""
        for failure in ("input_error", "conversion", "dflash", "uno"):
            for retained_sibling in (False, True):
                with self.subTest(failure=failure, retained_sibling=retained_sibling):
                    self._check_early_session_rejection(failure, retained_sibling)

    def test_invalid_colliding_session_rid_keeps_the_existing_owner(self):
        """Rejecting a colliding RID cannot discard the distinct live request with that RID."""
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.writes.clear()
                session = Session(
                    0, session_id="synthetic-session", streaming=streaming
                )
                historical, historical_proxy = self._incoming()
                active = session.create_req(
                    _recv(None, session=session, rid="active"), None, 32
                )
                active.multimodal_inputs = historical
                incoming, incoming_proxy = self._incoming()
                scheduler = _scheduler(session)
                scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
                scheduler.handle_generate_request(
                    _recv(incoming, session=session, parent=active.rid, rid=active.rid)
                )
                scheduler._add_request_to_queue.assert_not_called()
                scheduler.output_streamer.stream_output.assert_called_once()
                self.assertTrue(session.has_unfinished_request())
                self.assertIs(active.multimodal_inputs, historical)
                self.assertFalse(historical_proxy._consumer_acknowledged)
                self.assertTrue(incoming_proxy._consumer_acknowledged)
                self.assertEqual(len(self.writes), 1)
                active.finished_reason = FINISH_LENGTH(0)
                active._refresh_fill_ids()
                if streaming:
                    session.finish_req(active)
                session.release_finished_req_mm_inputs(active)
                controller = SessionController(Mock())
                controller.sessions[session.session_id] = session
                controller._close(session.session_id)
                self.assertTrue(historical_proxy._consumer_acknowledged)
                self.assertEqual(len(self.writes), 2)

    def test_generation_and_embedding_failures_release_the_new_lease(self):
        """Padding and position failures cannot orphan media before queue admission."""
        for embedding in (False, True):
            for stage in ("padding", "positions"):
                with self.subTest(embedding=embedding, stage=stage):
                    self.writes.clear()
                    image_inputs, proxy = self._incoming()
                    scheduler = _scheduler()
                    error = RuntimeError(stage)
                    if stage == "padding":
                        scheduler.pad_input_ids_func = Mock(side_effect=error)
                    else:
                        scheduler._mm_processor = SimpleNamespace(
                            compute_mrope_positions=Mock(side_effect=error)
                        )
                    method = (
                        scheduler.handle_embedding_request
                        if embedding
                        else scheduler.handle_generate_request
                    )
                    with self.assertRaises(RuntimeError) as caught:
                        method(_recv(image_inputs, embedding=embedding))
                    self.assertIs(caught.exception, error)
                    self.assertTrue(proxy._consumer_acknowledged)
                    self.assertIsNone(image_inputs.mm_items[0].feature)
                    self.assertEqual(len(self.writes), 1)
                    self.assertEqual(scheduler._add_request_to_queue.call_count, 0)

    def test_session_first_and_append_failures_preserve_committed_media(self):
        """A failed accepted turn clears its live owner without releasing saved media."""
        for streaming in (False, True):
            for append in (False, True):
                for distinct in (False, True) if append else (False,):
                    for stage in ("offsets", "padding", "positions"):
                        with self.subTest(
                            streaming=streaming,
                            append=append,
                            distinct=distinct,
                            stage=stage,
                        ):
                            self.writes.clear()
                            session = _RecordingSession(
                                streaming=streaming, distinct_container=distinct
                            )
                            parent, shared = (
                                self._committed_parent(session)
                                if append
                                else (None, None)
                            )
                            committed = parent.multimodal_inputs if parent else None
                            incoming, proxy = self._incoming()
                            scheduler = _scheduler(session)
                            error = RuntimeError(stage)
                            with ExitStack() as stack:
                                if stage == "offsets":
                                    stack.enter_context(
                                        patch.object(
                                            SessionController,
                                            "adjust_mm_offsets",
                                            side_effect=error,
                                        )
                                    )
                                elif stage == "padding":
                                    scheduler.pad_input_ids_func = Mock(
                                        side_effect=error
                                    )
                                else:
                                    scheduler._mm_processor = SimpleNamespace(
                                        compute_mrope_positions=Mock(side_effect=error)
                                    )
                                with self.assertRaises(RuntimeError) as caught:
                                    scheduler.handle_generate_request(
                                        _recv(
                                            incoming,
                                            session=session,
                                            parent=parent.rid if parent else None,
                                        )
                                    )
                            self.assertIs(caught.exception, error)
                            failed = session.created[-1]
                            self.assertIsInstance(failed.finished_reason, FINISH_ABORT)
                            self.assertIsNone(failed.multimodal_inputs)
                            self.assertFalse(session._active_reqs)
                            self.assertFalse(session._inflight)
                            self.assertFalse(session.has_unfinished_request())
                            self.assertTrue(proxy._consumer_acknowledged)
                            self.assertEqual(len(self.writes), 1)
                            if parent:
                                self.assertIs(parent.multimodal_inputs, committed)
                                self.assertEqual(shared.feature.tolist(), [11])
                                self.assertEqual(committed.mm_items, [shared])

    def test_partial_session_merge_releases_each_new_lease_once(self):
        """A merge failure after copying item references cannot leak or double-release."""
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.writes.clear()
                session = _RecordingSession(streaming=streaming)
                parent, shared = self._committed_parent(session)
                parent.multimodal_inputs.image_pad_len = []
                incoming, proxy = self._incoming()
                scheduler = _scheduler(session)
                with self.assertRaises(TypeError):
                    scheduler.handle_generate_request(
                        _recv(incoming, session=session, parent=parent.rid)
                    )
                self.assertTrue(proxy._consumer_acknowledged)
                self.assertEqual(len(self.writes), 1)
                self.assertEqual(shared.feature.tolist(), [11])
                self.assertEqual(parent.multimodal_inputs.mm_items, [shared])
                self.assertIsNone(session.created[-1].multimodal_inputs)
                self.assertFalse(session.has_unfinished_request())

    def test_attachment_failure_preserves_an_existing_terminal_reason(self):
        """Resource cleanup does not replace a terminal outcome recorded earlier."""
        session = _RecordingSession(streaming=True)
        incoming, proxy = self._incoming()
        scheduler = _scheduler(session)
        reason = FINISH_ABORT("already terminal")
        scheduler._maybe_namespace_elastic_radix_cache = lambda req: setattr(
            req, "finished_reason", reason
        )
        scheduler.pad_input_ids_func = Mock(side_effect=RuntimeError("padding"))
        with self.assertRaisesRegex(RuntimeError, "padding"):
            scheduler.handle_generate_request(_recv(incoming, session=session))
        self.assertIs(session.created[-1].finished_reason, reason)
        self.assertTrue(proxy._consumer_acknowledged)
        self.assertFalse(session.has_unfinished_request())

    def test_cancelled_attachment_retires_the_accepted_session_turn(self):
        """Cancellation during attachment must clear the turn's ownership and latch."""
        session = _RecordingSession(streaming=True)
        incoming, proxy = self._incoming()
        scheduler = _scheduler(session)
        error = asyncio.CancelledError()
        scheduler.pad_input_ids_func = Mock(side_effect=error)
        with self.assertRaises(asyncio.CancelledError) as caught:
            scheduler.handle_generate_request(_recv(incoming, session=session))
        self.assertIs(caught.exception, error)
        self.assertTrue(proxy._consumer_acknowledged)
        self.assertEqual(len(self.writes), 1)
        self.assertFalse(session.has_unfinished_request())

    def test_successful_attachment_retains_media_for_the_queued_request(self):
        """Cleanup applies only to failures; successful inputs reach the queue intact."""
        for embedding in (False, True):
            with self.subTest(embedding=embedding):
                self.writes.clear()
                incoming, proxy = self._incoming()
                scheduler = _scheduler()
                method = (
                    scheduler.handle_embedding_request
                    if embedding
                    else scheduler.handle_generate_request
                )
                method(_recv(incoming, embedding=embedding))
                queued = scheduler._add_request_to_queue.call_args.args[0]
                self.assertIs(queued.multimodal_inputs, incoming)
                self.assertIs(incoming.mm_items[0].feature, proxy)
                self.assertEqual(self.writes, [])


if __name__ == "__main__":
    unittest.main()
