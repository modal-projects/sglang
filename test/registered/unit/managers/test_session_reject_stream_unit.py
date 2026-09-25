import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.session.session_controller import Session
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _recv(rid, input_ids, **session_options):
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", input_ids),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(temperature=0, max_new_tokens=4),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=True,
        session_params=SessionParams(id="session-a", **session_options),
    )


class TestSessionRejectStream(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.server_args = ServerArgs(model_path="dummy")
        publish(self.server_args, role="scheduler")
        parallel = patch(
            "sglang.srt.managers.schedule_batch.get_parallel",
            return_value=SimpleNamespace(tp_rank=0),
        )
        parallel.start()
        self.addCleanup(parallel.stop)

    def test_rejection_emits_terminal_empty_output_and_preserves_owner(self):
        """A rejected turn must never run a dummy decode or displace a live turn."""
        for reason in ("busy", "replace", "drop_previous_output", "offset", "empty"):
            with self.subTest(reason=reason):
                session = Session(0, "session-a", streaming=True)
                if reason == "busy":
                    session.create_req(_recv("active", [1, 2]), None, 16)
                options = (
                    {reason: True}
                    if reason in ("replace", "drop_previous_output")
                    else {}
                )
                if reason == "offset":
                    options["offset"] = 1
                recv_req = _recv(
                    "rejected", [] if reason == "empty" else [3], **options
                )
                sender = SimpleNamespace(send_output=Mock())
                streamer = SchedulerOutputStreamer(
                    send_to_detokenizer=sender,
                    tree_cache=None,
                    ps=SimpleNamespace(dp_rank=0, attn_tp_rank=0),
                    server_args=self.server_args,
                    is_generation=True,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    disaggregation_mode=DisaggregationMode.NULL,
                    enable_hicache_storage=lambda: False,
                )
                scheduler = SimpleNamespace(
                    enable_session_radix_cache=False,
                    session_controller={"session-a": session},
                    tokenizer=None,
                    model_config=SimpleNamespace(vocab_size=16, hf_eos_token_id=[]),
                    metrics_reporter=SimpleNamespace(enable_metrics=False),
                    output_streamer=streamer,
                    disaggregation_mode=DisaggregationMode.NULL,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    _maybe_namespace_elastic_radix_cache=Mock(),
                    init_req_max_new_tokens=Mock(),
                    max_req_input_len=128,
                    grammar_manager=SimpleNamespace(
                        process_req_with_grammar=lambda req: False
                    ),
                    _add_request_to_queue=Mock(
                        side_effect=AssertionError("rejected request was queued")
                    ),
                )
                Scheduler.handle_generate_request(scheduler, recv_req)

                sender.send_output.assert_called_once()
                payload = sender.send_output.call_args.args[0]
                self.assertEqual(payload.rids, ["rejected"])
                self.assertEqual([list(ids) for ids in payload.output_ids], [[]])
                self.assertEqual(payload.completion_tokens, [0])
                self.assertEqual(payload.finished_reasons[0]["type"], "abort")
                self.assertEqual(
                    session._inflight_rid, "active" if reason == "busy" else None
                )
                self.assertEqual(session.req_nodes, {})
                scheduler._add_request_to_queue.assert_not_called()

    def _admission_scheduler(self, route):
        session = Session(0, "session-a", streaming=True)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_session_radix_cache = False
        scheduler.session_controller = {session.session_id: session}
        scheduler.tokenizer = None
        scheduler.model_config = SimpleNamespace(vocab_size=16, hf_eos_token_id=[])
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler.dllm_config = None
        scheduler._maybe_namespace_elastic_radix_cache = Mock()
        scheduler.max_new_tokens_limit = None
        scheduler.page_size = 1
        scheduler.max_req_len = 128
        scheduler.max_total_num_tokens = 128
        scheduler.enable_priority_scheduling = False
        scheduler.abort_on_priority_when_disabled = route == "priority_disabled"
        scheduler.max_queued_requests = 1
        scheduler.waiting_queue = [
            Req("existing", None, array("q", [1]), SamplingParams(max_new_tokens=4))
        ]
        scheduler.beam_coordinator = SimpleNamespace(retire_group=Mock())
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=Mock())
        )
        scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
        scheduler._prefetch_kvcache = Mock()
        return scheduler, session

    def test_media_error_survives_admission_rejection(self):
        """Queue rejection must preserve the earlier media error in local and wire state."""
        for route in ("queue_full", "priority_disabled"):
            for use_session in (False, True):
                with self.subTest(route=route, session=use_session):
                    scheduler, session = self._admission_scheduler(route)
                    recv_req = _recv("rejected", [2, 3])
                    if not use_session:
                        recv_req.session_params = None
                    if route == "priority_disabled":
                        recv_req.priority = 1

                    with patch(
                        "sglang.srt.managers.scheduler.get_parallel",
                        return_value=SimpleNamespace(attn_dcp_size=1, pp_size=1),
                    ):
                        scheduler.handle_generate_request(
                            recv_req, mm_input_error="Media preprocessing failed first."
                        )

                    sender = scheduler.ipc_channels.send_to_tokenizer.send_output
                    sender.assert_called_once()
                    payload, req = sender.call_args.args
                    expected = FINISH_ABORT(
                        "Media preprocessing failed first.", 500, "InternalServerError"
                    ).to_json()
                    self.assertEqual(req.finished_reason.to_json(), expected)
                    self.assertEqual(payload.finished_reason, expected)
                    self.assertIsNone(req.to_finish)
                    self.assertEqual(req.output_ids, array("q"))
                    self.assertEqual(
                        [r.rid for r in scheduler.waiting_queue], ["existing"]
                    )
                    self.assertFalse(session.has_unfinished_request())
                    scheduler._prefetch_kvcache.assert_not_called()

    def test_admission_rejection_preserves_finished_reason(self):
        """An existing terminal reason outranks a later deferred error and queue rejection."""
        for route in ("queue_full", "priority_disabled"):
            with self.subTest(route=route):
                scheduler, session = self._admission_scheduler(route)
                req = session.create_req(_recv("rejected", [2, 3]), None, 16)
                if route == "priority_disabled":
                    req.priority = 1
                terminal = FINISH_ABORT("Already finished.", 409, "ConflictError")
                req.finished_reason = terminal
                req.to_finish = FINISH_ABORT(
                    "Later failure.", 500, "InternalServerError"
                )

                scheduler._add_request_to_queue(req)

                sender = scheduler.ipc_channels.send_to_tokenizer.send_output
                sender.assert_called_once()
                payload, sent_req = sender.call_args.args
                self.assertIs(sent_req, req)
                self.assertIs(req.finished_reason, terminal)
                self.assertEqual(payload.finished_reason, terminal.to_json())
                self.assertEqual([r.rid for r in scheduler.waiting_queue], ["existing"])
                self.assertFalse(session.has_unfinished_request())
                scheduler._prefetch_kvcache.assert_not_called()

    def test_media_validation_rejection_releases_only_new_turn_media(self):
        """Post-media validation must release only the rejected turn's additions."""
        for validation in (
            "prompt_length",
            "logprob_start",
            "routed_experts_start",
            "mlx_prompt_logprob",
        ):
            for streaming in (False, True):
                for appended in (False, True):
                    with self.subTest(
                        validation=validation, streaming=streaming, appended=appended
                    ):
                        scheduler, _ = self._admission_scheduler("queue_full")
                        session = Session(0, "session-a", streaming=streaming)
                        scheduler.session_controller[session.session_id] = session
                        scheduler._mm_processor = None
                        scheduler.pad_input_ids_func = None
                        scheduler.max_req_input_len = (
                            2 if validation == "prompt_length" else 128
                        )

                        inherited_proxy = CudaIpcTensorTransportProxy.__new__(
                            CudaIpcTensorTransportProxy
                        )
                        inherited_proxy.total_consumer_count = 1
                        inherited_proxy.release_without_reconstruction = Mock()
                        inherited = MultimodalDataItem(
                            modality=Modality.IMAGE, feature=torch.tensor([1.0])
                        )
                        inherited.set_pad_value()
                        inherited.feature = inherited_proxy
                        if appended:
                            prior = session.create_req(_recv("prior", [1, 2]), None, 16)
                            prior.multimodal_inputs = MultimodalInputs(
                                mm_items=[inherited]
                            )
                            prior.finished_reason = FINISH_LENGTH(length=0)
                            self.assertTrue(session.has_unfinished_request())
                            self.assertFalse(prior.kv.holds_kv)
                            self.assertFalse(prior.kv.holds_mamba)
                            self.assertIsNone(prior.kv.retraction_backup)
                            if streaming:
                                session.finish_req(prior)
                            session.release_finished_req_mm_inputs(prior)
                            self.assertFalse(session.has_unfinished_request())

                        own_proxy = CudaIpcTensorTransportProxy.__new__(
                            CudaIpcTensorTransportProxy
                        )
                        own_proxy.total_consumer_count = 1
                        own_proxy.release_without_reconstruction = Mock()
                        own = MultimodalDataItem(
                            modality=Modality.IMAGE, feature=torch.tensor([2.0])
                        )
                        own.set_pad_value()
                        own.feature = own_proxy
                        recv_req = _recv("rejected", [3, 4, 5])
                        if appended and not streaming:
                            recv_req.session_params.rid = "prior"
                        recv_req.mm_inputs = MultimodalInputs(mm_items=[own])
                        if validation == "logprob_start":
                            recv_req.return_logprob = True
                            recv_req.logprob_start_len = 1024
                        elif validation == "routed_experts_start":
                            recv_req.return_routed_experts = True
                            recv_req.routed_experts_start_len = 1024
                        elif validation == "mlx_prompt_logprob":
                            recv_req.return_logprob = True
                            recv_req.logprob_start_len = 0

                        with (
                            patch(
                                "sglang.srt.managers.scheduler.get_parallel",
                                return_value=SimpleNamespace(
                                    attn_dcp_size=1, pp_size=1
                                ),
                            ),
                            patch(
                                "sglang.srt.managers.scheduler.get_device",
                                return_value=SimpleNamespace(
                                    mlx_enable_sampling=validation
                                    == "mlx_prompt_logprob"
                                ),
                            ),
                        ):
                            scheduler.handle_generate_request(recv_req)

                        sender = scheduler.ipc_channels.send_to_tokenizer.send_output
                        sender.assert_called_once()
                        payload, req = sender.call_args.args
                        expected_message = {
                            "prompt_length": "Multimodal prompt is too long",
                            "logprob_start": "logprob_start_len",
                            "routed_experts_start": "routed_experts_start_len",
                            "mlx_prompt_logprob": "MLX sampling",
                        }[validation]
                        self.assertIn(expected_message, req.finished_reason.message)
                        self.assertEqual(req.finished_reason.status_code, 400)
                        self.assertEqual(
                            payload.finished_reason, req.finished_reason.to_json()
                        )
                        self.assertIsNone(req.multimodal_inputs)
                        self.assertFalse(req.kv.holds_kv)
                        self.assertFalse(req.kv.holds_mamba)
                        self.assertFalse(session.has_unfinished_request())
                        own_proxy.release_without_reconstruction.assert_called_once_with(
                            1
                        )
                        self.assertIsNone(own.feature)
                        if appended:
                            self.assertIs(inherited.feature, inherited_proxy)
                            self.assertEqual(
                                prior.multimodal_inputs.mm_items, [inherited]
                            )
                        inherited_proxy.release_without_reconstruction.assert_not_called()

                        scheduler._release_dropped_waiting_req_mm_inputs(req)
                        scheduler._release_dropped_waiting_req_mm_inputs(req)
                        own_proxy.release_without_reconstruction.assert_called_once_with(
                            1
                        )
                        inherited_proxy.release_without_reconstruction.assert_not_called()


if __name__ == "__main__":
    unittest.main()
