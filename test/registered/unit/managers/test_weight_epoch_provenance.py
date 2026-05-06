import asyncio
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestWeightEpochProvenance(unittest.TestCase):
    def _new_manager(self) -> TokenizerManager:
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.server_args = SimpleNamespace(weight_version="v2")
        manager.current_weight_epoch = 2
        manager.weight_version_by_epoch = {1: "v1", 2: "v2"}
        return manager

    def test_compact_provenance_reports_mixed_and_stale_kv(self):
        manager = self._new_manager()
        state = ReqState(
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=SimpleNamespace(),
            time_stats=SimpleNamespace(),
            request_weight_epoch=1,
            cache_epoch=1,
        )
        recv_obj = SimpleNamespace(
            weight_epoch_start=[1],
            weight_epoch_end=[2],
            mixed_weight_epochs=[True],
        )

        manager._apply_output_weight_provenance(state, recv_obj, 0)

        meta_info = {}
        manager._add_weight_provenance_to_meta_info(meta_info, state)

        self.assertEqual(meta_info["weight_epoch_start"], 1)
        self.assertEqual(meta_info["weight_epoch_end"], 2)
        self.assertEqual(meta_info["cache_epoch"], 1)
        self.assertTrue(meta_info["mixed_weight_epochs"])
        self.assertTrue(meta_info["resume_from_stale_kv"])
        self.assertEqual(meta_info["weight_version"], "v2")

    def test_provenance_defaults_to_request_epoch_before_first_token(self):
        manager = self._new_manager()
        state = ReqState(
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=SimpleNamespace(),
            time_stats=SimpleNamespace(),
            request_weight_epoch=1,
            cache_epoch=1,
        )

        meta_info = {}
        manager._add_weight_provenance_to_meta_info(meta_info, state)

        self.assertEqual(meta_info["weight_epoch_start"], 1)
        self.assertEqual(meta_info["weight_epoch_end"], 1)
        self.assertFalse(meta_info["mixed_weight_epochs"])
        self.assertFalse(meta_info["resume_from_stale_kv"])
        self.assertEqual(meta_info["weight_version"], "v1")


if __name__ == "__main__":
    unittest.main()
