"""Regression tests for DFlash state-capture result propagation."""

import inspect
import unittest

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDFlashStateCapture(CustomTestCase):
    def test_decode_result_propagates_target_state_capture_outputs(self):
        source = inspect.getsource(DFlashWorkerV2.forward_batch_generation)
        final_result = source[source.rfind("return GenerationBatchResult(") :]

        self.assertIn(
            "routed_experts_output=target_out.routed_experts_output", final_result
        )
        self.assertIn(
            "indexer_topk_output=target_out.indexer_topk_output", final_result
        )


if __name__ == "__main__":
    unittest.main()
