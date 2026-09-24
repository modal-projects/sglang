import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
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


if __name__ == "__main__":
    unittest.main()
