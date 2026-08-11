import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

FORBIDDEN_TOKENS = ("self.running_batch", "self.last_batch", "self.cur_batch")

DECISION_METHODS = (
    Scheduler.get_next_batch_to_run,
    Scheduler.get_new_batch_prefill,
    Scheduler._get_new_batch_prefill_raw,
    Scheduler._abort_on_running_timeout,
    Scheduler.is_disable_overlap_for_batch,
    SchedulerDisaggregationPrefillMixin.get_next_disagg_prefill_batch_to_run,
    SchedulerDisaggregationPrefillMixin.process_prefill_chunk,
    SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch,
    SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run,
)


class TestDecisionMethodsHaveNoHiddenBatchChannel(unittest.TestCase):
    def test_decision_methods_take_batches_as_params_not_self(self):
        """The batch decision tree must receive running/last batch as params, never via self.*."""
        for method in DECISION_METHODS:
            source = inspect.getsource(inspect.unwrap(method))
            self.assertIn(
                f"def {method.__name__}",
                source,
                msg=f"failed to read the real source of {method.__qualname__}",
            )
            for token in FORBIDDEN_TOKENS:
                self.assertNotIn(
                    token,
                    source,
                    msg=(
                        f"{method.__qualname__} references {token}; pass the batch "
                        "explicitly and return it via NextBatchPlan instead."
                    ),
                )


class TestFlashinferMtpPhaseSync(unittest.TestCase):
    @staticmethod
    def _batch(*, is_extend: bool, is_speculative: bool = True):
        return SimpleNamespace(
            is_extend_in_batch=is_extend,
            forward_mode=SimpleNamespace(is_extend=lambda: is_extend),
            spec_algorithm=SimpleNamespace(is_none=lambda: not is_speculative),
        )

    def _scheduler(self, *, require_mlp_sync: bool):
        scheduler = object.__new__(Scheduler)
        scheduler.require_mlp_sync = require_mlp_sync
        return scheduler

    @patch("sglang.srt.managers.scheduler.get_moe_a2a_backend")
    def test_syncs_only_at_mtp_phase_crossing(self, get_backend):
        get_backend.return_value.is_flashinfer.return_value = True
        scheduler = self._scheduler(require_mlp_sync=True)
        extend = self._batch(is_extend=True)
        decode = self._batch(is_extend=False)

        self.assertTrue(scheduler._needs_flashinfer_mtp_phase_sync(decode, extend))
        self.assertTrue(scheduler._needs_flashinfer_mtp_phase_sync(extend, decode))
        self.assertFalse(scheduler._needs_flashinfer_mtp_phase_sync(decode, decode))

    @patch("sglang.srt.managers.scheduler.get_moe_a2a_backend")
    def test_skips_non_mtp_and_non_flashinfer(self, get_backend):
        scheduler = self._scheduler(require_mlp_sync=False)
        extend = self._batch(is_extend=True)
        decode = self._batch(is_extend=False)

        get_backend.return_value.is_flashinfer.return_value = False
        self.assertFalse(scheduler._needs_flashinfer_mtp_phase_sync(decode, extend))

        get_backend.return_value.is_flashinfer.return_value = True
        non_mtp_decode = self._batch(is_extend=False, is_speculative=False)
        self.assertFalse(
            scheduler._needs_flashinfer_mtp_phase_sync(non_mtp_decode, extend)
        )


if __name__ == "__main__":
    unittest.main()
