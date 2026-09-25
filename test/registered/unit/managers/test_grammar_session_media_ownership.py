"""Grammar rejection retires session media without losing shared history."""

import unittest
from array import array
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    InvalidGrammarObject,
)
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.managers.io_struct import (
    AbortReq,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.multimodal.transport import cuda_ipc, memory_pool
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session, SessionController

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _create(session, rid, *, parent=None, replace=False, media=None):
    received = TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", [2]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        session_params=SessionParams(
            id=session.session_id, rid=parent, replace=replace
        ),
    )
    req = session.create_req(received, tokenizer=None, vocab_size=32)
    if media is not None:
        req.extend_image_inputs(media)
    return req


def _finish(session, req):
    req.finished_reason = FINISH_LENGTH(1)
    req._refresh_fill_ids()
    if session.streaming:
        session.finish_req(req)
    session.release_finished_req_mm_inputs(req)


class TestGrammarSessionMediaOwnership(CustomTestCase):
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

    def _media(self):
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
            modality=Modality.IMAGE,
            feature=proxy,
            offsets=[(0, 0)],
            hash=self.proxy_number,
            pad_value=self.proxy_number,
            model_specific_data={
                cuda_ipc.DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY: True
            },
        )
        return MultimodalInputs(mm_items=[item]), proxy

    def _reject(self, req, stage):
        backend = BaseGrammarBackend()
        self.addCleanup(backend.executor.shutdown)
        manager = GrammarManager.__new__(GrammarManager)
        manager.grammar_backend = None if stage == "unsupported" else backend
        manager.grammar_queue = []
        manager.pp_rank = 0
        manager.pp_size = 1
        manager.grammar_sync_size = 1
        manager.SGLANG_GRAMMAR_POLL_INTERVAL = 0 if stage == "timeout" else 1
        manager.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 1
        req.sampling_params.json_schema = '{"type":"object"}'
        key = ("json", req.sampling_params.json_schema)
        if stage == "cached_invalid":
            backend.set_cache(key, InvalidGrammarObject("invalid schema"))
        future = Future()
        with patch.object(backend.executor, "submit", return_value=future):
            queued = manager.process_req_with_grammar(req)
        self.assertEqual(queued, stage in ("exception", "timeout", "abort"))
        if stage == "exception":
            future.set_exception(ValueError("invalid schema"))
        elif stage == "abort":
            manager.abort_requests(AbortReq(rid=req.rid))
        if queued:
            self.assertEqual(manager.get_ready_grammar_requests(), [req])
            self.assertFalse(manager.has_waiting_grammars())
        if stage in ("timeout", "abort"):
            self.assertTrue(future.cancelled())
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        messages = {
            "unsupported": "is not supported",
            "cached_invalid": "Failed to compile json grammar: invalid schema",
            "exception": "Grammar compilation failed: invalid schema",
            "timeout": "Grammar preprocessing timed out",
            "abort": "Aborted by AbortReq.",
        }
        self.assertIn(messages[stage], req.to_finish.message)

    def _check_rejection(self, stage):
        for streaming, history in (
            (False, "first"),
            (False, "append"),
            (False, "replace"),
            (True, "first"),
            (True, "append"),
        ):
            with self.subTest(streaming=streaming, history=history):
                self.writes.clear()
                session = Session(
                    0, session_id="synthetic-session", streaming=streaming
                )
                controller = SessionController(Mock())
                controller.sessions[session.session_id] = session
                parent = historical = historical_proxy = abandoned_proxy = None
                if history != "first":
                    historical, historical_proxy = self._media()
                    parent = _create(session, "parent", media=historical)
                    _finish(session, parent)
                if history == "replace":
                    abandoned, abandoned_proxy = self._media()
                    old = _create(session, "old", parent=parent.rid, media=abandoned)
                    _finish(session, old)
                incoming, incoming_proxy = self._media()
                req = _create(
                    session,
                    "rejected",
                    parent=parent.rid if parent else None,
                    replace=history == "replace",
                    media=incoming,
                )
                if abandoned_proxy is not None:
                    self.assertTrue(abandoned_proxy._consumer_acknowledged)
                    self.assertIsNone(abandoned.mm_items[0].feature)
                    self.assertIsNone(old.multimodal_inputs)
                    self.assertNotIn(old.rid, session.req_nodes)
                self._reject(req, stage)
                self.assertTrue(incoming_proxy._consumer_acknowledged)
                self.assertIsNone(incoming.mm_items[0].feature)
                self.assertIsNone(req.multimodal_inputs)
                self.assertNotIn(req.rid, session.req_nodes)
                self.assertFalse(session._active_reqs)
                self.assertFalse(session._retired_reqs)
                self.assertFalse(session._inflight)
                self.assertFalse(session.has_unfinished_request())
                released = 1 + (abandoned_proxy is not None)
                self.assertEqual(len(self.writes), released)
                if parent is not None:
                    self.assertIs(parent.multimodal_inputs, historical)
                    self.assertIs(historical.mm_items[0].feature, historical_proxy)
                    self.assertFalse(historical_proxy._consumer_acknowledged)
                    self.assertEqual(historical.mm_items[0].offsets, [(0, 0)])

                # Repeated terminal cleanup must not acknowledge retained history.
                req.finished_reason, req.to_finish = req.to_finish, None
                session.release_finished_req_mm_inputs(req)
                session.discard_req(req)
                self.assertEqual(len(self.writes), released)
                next_req = _create(
                    session, "next", parent=parent.rid if parent else None
                )
                self.assertIsNone(next_req.to_finish)
                if historical is not None:
                    self.assertEqual(
                        next_req.multimodal_inputs.mm_items, historical.mm_items
                    )
                _finish(session, next_req)
                controller._close(session.session_id)
                self.assertNotIn(session.session_id, controller.sessions)
                self.assertIsNone(next_req.multimodal_inputs)
                if historical_proxy is not None:
                    self.assertTrue(historical_proxy._consumer_acknowledged)
                    self.assertIsNone(historical.mm_items[0].feature)
                self.assertEqual(len(self.writes), released + (parent is not None))

    def test_unsupported_backend_releases_rejected_session_media(self):
        self._check_rejection("unsupported")

    def test_cached_invalid_grammar_releases_rejected_session_media(self):
        self._check_rejection("cached_invalid")

    def test_compilation_exception_releases_rejected_session_media(self):
        self._check_rejection("exception")

    def test_compilation_timeout_releases_rejected_session_media(self):
        self._check_rejection("timeout")

    def test_abort_request_releases_queued_session_media(self):
        self._check_rejection("abort")


if __name__ == "__main__":
    unittest.main()
