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

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAbortReason(CustomTestCase):
    def test_scheduler_preserves_waiting_and_running_abort_reason(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.chunked_req = None
        scheduler.waiting_queue = [SimpleNamespace(rid="waiting", mamba_pool_idx=None)]
        scheduler.enable_hicache_storage = False
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.dllm_config = None
        scheduler.grammar_manager = MagicMock()
        scheduler.ps = SimpleNamespace(pp_size=1)
        running = MagicMock(rid="running")
        running.finished.return_value = False
        scheduler.running_batch = SimpleNamespace(reqs=[running])
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
        self.assertIsInstance(running.to_finish, FINISH_ABORT)
        self.assertEqual(running.to_finish.to_json(), finished_reason)


if __name__ == "__main__":
    unittest.main()
