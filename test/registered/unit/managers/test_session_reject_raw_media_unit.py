"""Raw media ownership when session admission returns a terminal rejection."""

import pickle
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
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

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _recv(rid, input_ids, session_id="session-a", **session_options):
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
        session_params=SessionParams(id=session_id, **session_options),
    )


def _proxy(generation, consumers=1):
    # Accelerator release is the only substituted boundary; native containers,
    # session admission, streaming, and PP's pickle wire format run normally.
    proxy = object.__new__(CudaIpcTensorTransportProxy)
    proxy.generation = generation
    proxy.total_consumer_count = consumers
    proxy._consumer_acknowledged = False
    return proxy


class TestSessionRejectRawMedia(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.args = ServerArgs(model_path="dummy")
        publish(self.args, role="scheduler")
        parallel = patch(
            "sglang.srt.managers.schedule_batch.get_parallel",
            return_value=SimpleNamespace(
                tp_rank=0,
                tp_size=2,
                attn_tp_rank=0,
                attn_tp_size=2,
                attn_cp_rank=0,
                attn_cp_size=1,
                pp_size=1,
            ),
        )
        parallel.start()
        self.addCleanup(parallel.stop)

    def _scheduler(self, sessions):
        sender = SimpleNamespace(send_output=Mock())
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_session_radix_cache = False
        scheduler.session_controller = sessions
        scheduler.tokenizer = None
        scheduler.model_config = SimpleNamespace(vocab_size=32, hf_eos_token_id=[])
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler._maybe_namespace_elastic_radix_cache = Mock()
        scheduler.init_req_max_new_tokens = Mock()
        scheduler.max_req_input_len = 128
        scheduler.grammar_manager = SimpleNamespace(
            process_req_with_grammar=lambda req: False
        )
        scheduler._add_request_to_queue = Mock(
            side_effect=AssertionError("rejected request was queued")
        )
        scheduler.output_streamer = SchedulerOutputStreamer(
            send_to_detokenizer=sender,
            tree_cache=None,
            ps=SimpleNamespace(dp_rank=0, attn_tp_rank=0),
            server_args=self.args,
            is_generation=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            enable_hicache_storage=lambda: False,
        )
        return scheduler, sender

    def _session(self, history, active):
        session = Session(0, "session-a", streaming=True)
        shared = MultimodalDataItem(modality=Modality.IMAGE, feature=_proxy(10))
        committed = None
        if history:
            committed = session.create_req(_recv("committed", [1, 2]), None, 32)
            committed.multimodal_inputs = MultimodalInputs(mm_items=[shared])
            committed.finished_reason = FINISH_LENGTH(0)
            session.finish_req(committed)
        if active:
            owner = session.create_req(_recv("active-owner", [4]), None, 32)
            if not history:
                owner.multimodal_inputs = MultimodalInputs(mm_items=[shared])
        return session, shared, committed

    def _assert_terminal(self, sender, rid, message):
        sender.send_output.assert_called_once()
        payload = sender.send_output.call_args.args[0]
        self.assertEqual(payload.rids, [rid])
        self.assertEqual([list(ids) for ids in payload.output_ids], [[]])
        self.assertEqual(payload.completion_tokens, [0])
        self.assertEqual(payload.finished_reasons[0]["type"], "abort")
        self.assertEqual(payload.finished_reasons[0]["message"], message)

    def test_rejections_release_only_incoming_native_media(self):
        """Early admission rejection must retain history and dispose its new payload."""
        reasons = {
            "busy": "Streaming session already has an active request.",
            "replace": "Streaming sessions do not support replace.",
            "drop_previous_output": "Streaming sessions do not support drop_previous_output.",
            "offset": "Streaming sessions do not support offset.",
            "empty": "A session request must contain input tokens after restoring history.",
            "missing": "Invalid request: session id missing does not exist",
            "closing": "Invalid request: close was requested for session session-a",
        }
        for container_type in (MultimodalInputs, MultimodalProcessorOutput):
            for history in (False, True):
                for reason, message in reasons.items():
                    if history and reason == "empty":
                        continue
                    with self.subTest(
                        container=container_type.__name__,
                        history=history,
                        reason=reason,
                    ):
                        active = reason in ("busy", "missing", "closing")
                        session, shared, committed = self._session(history, active)
                        owner = session._inflight_rid
                        old_nodes = dict(session.req_nodes)
                        old_history = (
                            None
                            if committed is None
                            else list(committed.origin_input_ids)
                        )
                        if reason == "closing":
                            session.close_on_finish = True
                        options = {}
                        if reason in ("replace", "drop_previous_output"):
                            options[reason] = True
                        elif reason == "offset":
                            options["offset"] = 1
                        recv = _recv(
                            "rejected",
                            [] if reason == "empty" else [3],
                            "missing" if reason == "missing" else "session-a",
                            **options,
                        )
                        incoming = [_proxy(20), _proxy(21), _proxy(22)]
                        item = MultimodalDataItem(
                            modality=Modality.IMAGE,
                            feature=incoming[0],
                            precomputed_embeddings=incoming[1],
                            model_specific_data={"extra_feature": incoming[2]},
                        )
                        recv.mm_inputs = container_type(mm_items=[item])
                        scheduler, sender = self._scheduler({"session-a": session})
                        with patch.object(
                            CudaIpcTensorTransportProxy,
                            "release_without_reconstruction",
                            autospec=True,
                        ) as release:
                            scheduler.handle_generate_request(recv)
                            self.assertEqual(release.call_count, len(incoming))
                            self.assertEqual(
                                [call.args[0] for call in release.call_args_list],
                                incoming,
                            )
                            self.assertTrue(
                                all(
                                    call.args[1] == 1 for call in release.call_args_list
                                )
                            )
                            scheduler_module._release_unadmitted_mm_inputs(recv)
                            self.assertEqual(release.call_count, len(incoming))
                        self.assertIsNone(recv.mm_inputs)
                        self.assertEqual(session._inflight_rid, owner)
                        self.assertEqual(session.req_nodes, old_nodes)
                        self.assertIsNotNone(shared.feature)
                        if committed is not None:
                            self.assertEqual(
                                list(committed.origin_input_ids), old_history
                            )
                            self.assertIs(
                                committed.multimodal_inputs.mm_items[0], shared
                            )
                        self._assert_terminal(sender, recv.rid, message)
                        scheduler._add_request_to_queue.assert_not_called()

    def test_accepted_turn_retains_incoming_and_historical_media(self):
        """The new early cleanup must not run on successful session admission."""
        for history in (False, True):
            with self.subTest(history=history):
                session, shared, committed = self._session(history, active=False)
                recv = _recv("accepted", [3])
                incoming = MultimodalDataItem(
                    modality=Modality.IMAGE, feature=torch.ones(2)
                )
                media = MultimodalInputs(mm_items=[incoming])
                recv.mm_inputs = media
                scheduler, sender = self._scheduler({"session-a": session})
                scheduler.pad_input_ids_func = None
                scheduler._mm_processor = None
                scheduler._add_request_to_queue = Mock()
                with patch.object(
                    CudaIpcTensorTransportProxy,
                    "release_without_reconstruction",
                    autospec=True,
                ) as release:
                    scheduler.handle_generate_request(recv)
                    release.assert_not_called()
                scheduler._add_request_to_queue.assert_called_once()
                req = scheduler._add_request_to_queue.call_args.args[0]
                self.assertIsNone(req.to_finish)
                self.assertIsNone(req.finished_reason)
                self.assertIs(recv.mm_inputs, media)
                self.assertIsNotNone(incoming.feature)
                self.assertEqual(session._inflight_rid, "accepted")
                self.assertIs(req.multimodal_inputs.mm_items[-1], incoming)
                if committed is not None:
                    self.assertIs(req.multimodal_inputs.mm_items[0], shared)
                    self.assertIsNotNone(shared.feature)
                sender.send_output.assert_not_called()

    def test_output_failure_does_not_repeat_raw_media_release(self):
        session, shared, _ = self._session(history=True, active=True)
        recv = _recv("rejected", [3])
        proxy = _proxy(40)
        recv.mm_inputs = MultimodalProcessorOutput(
            mm_items=[MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)]
        )
        scheduler, sender = self._scheduler({"session-a": session})
        sender.send_output.side_effect = RuntimeError("output unavailable")
        with patch.object(
            CudaIpcTensorTransportProxy, "release_without_reconstruction", autospec=True
        ) as release:
            with self.assertRaisesRegex(RuntimeError, "output unavailable"):
                scheduler.handle_generate_request(recv)
            release.assert_called_once_with(proxy, 1)
            scheduler_module._release_unadmitted_mm_inputs(recv)
            release.assert_called_once_with(proxy, 1)
        self.assertIsNone(recv.mm_inputs)
        self.assertEqual(session._inflight_rid, "active-owner")
        self.assertIsNotNone(shared.feature)

    def test_terminal_payload_relay_does_not_release_tp_lease_again(self):
        """PP relays after ingest; abandoned media must not regain a cleanup owner."""
        recv = _recv("rejected", [3])
        recv.mm_inputs = MultimodalProcessorOutput(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.IMAGE, feature=_proxy(30, consumers=2)
                )
            ]
        )
        # The producer counts TP consumers, not PP stages. Each first-stage TP
        # rank receives its own deserialized copy before any rank processes it.
        tp_inputs = [pickle.loads(pickle.dumps(recv)) for _ in range(2)]
        with patch.object(
            CudaIpcTensorTransportProxy, "release_without_reconstruction", autospec=True
        ) as release:
            for request in tp_inputs:
                session, _, _ = self._session(history=True, active=True)
                scheduler, sender = self._scheduler({"session-a": session})
                scheduler.handle_generate_request(request)
                self._assert_terminal(
                    sender,
                    request.rid,
                    "Streaming session already has an active request.",
                )
                self.assertEqual(session._inflight_rid, "active-owner")
            self.assertEqual(release.call_count, 2)
            self.assertTrue(all(call.args[1] == 1 for call in release.call_args_list))
            relayed = pickle.loads(pickle.dumps(tp_inputs[0]))
            session, _, _ = self._session(history=True, active=True)
            scheduler, sender = self._scheduler({"session-a": session})
            scheduler.handle_generate_request(relayed)
            self.assertEqual(release.call_count, 2)
            self.assertEqual(session._inflight_rid, "active-owner")
            self._assert_terminal(
                sender, relayed.rid, "Streaming session already has an active request."
            )


if __name__ == "__main__":
    unittest.main()
