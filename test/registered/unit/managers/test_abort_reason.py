import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.io_struct import AbortReq  # noqa: E402
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    FINISH_ABORT,
    client_cancel_finish_reason,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.runtime_context import get_context  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _fake_req(rid: str) -> MagicMock:
    req = MagicMock(rid=rid)
    req.weight_version_events = []
    req.output_ids = []
    req.kv.holds_mamba = False
    req.finished.return_value = False
    return req


class TestAbortReason(CustomTestCase):
    def test_scheduler_preserves_waiting_and_running_abort_reason(self):
        override = get_context().override_server_args(speculative_algorithm=None)
        override.install()
        self.addCleanup(override.restore)

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.chunked_req = None
        scheduler.mm_receiver = None
        scheduler.beam_coordinator = MagicMock()
        scheduler.grammar_manager = MagicMock()
        scheduler.dllm_config = None
        scheduler.enable_hicache_storage = False
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.ps = SimpleNamespace(pp_size=1)
        scheduler.waiting_queue = [_fake_req("waiting")]
        scheduler.running_batch = SimpleNamespace(reqs=[_fake_req("running")])
        scheduler.last_batch = None
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=MagicMock())
        )
        finished_reason = client_cancel_finish_reason()

        Scheduler.abort_request(
            scheduler,
            AbortReq(rid="", abort_all=True, finished_reason=finished_reason),
        )

        waiting_abort = (
            scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        )
        self.assertEqual(waiting_abort.finished_reason, finished_reason)
        running = scheduler.running_batch.reqs[0]
        self.assertIsInstance(running.to_finish, FINISH_ABORT)
        self.assertEqual(running.to_finish.to_json(), finished_reason)


if __name__ == "__main__":
    unittest.main()
