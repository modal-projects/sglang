"""DFlash V2 platform hooks: portable defaults and ROCm worker selection."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative import spec_info
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.dflash_worker_v2_rocm import DFlashWorkerV2ROCm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDFlashWorkerPlatformHooks(CustomTestCase):
    def test_create_worker_selects_rocm_worker_on_hip(self):
        for hip, expected in ((True, DFlashWorkerV2ROCm), (False, DFlashWorkerV2)):
            with (
                self.subTest(hip=hip),
                mock.patch("sglang.srt.utils.is_hip", return_value=hip),
                mock.patch.object(spec_info, "resolving_view", lambda args: args),
            ):
                worker_cls = spec_info.SpeculativeAlgorithm.DFLASH.create_worker(
                    SimpleNamespace()
                )
                self.assertIs(worker_cls, expected)

    def test_portable_defaults(self):
        worker = object.__new__(DFlashWorkerV2)
        self.assertFalse(DFlashWorkerV2._packed_draft_argmax)
        self.assertFalse(DFlashWorkerV2._publish_before_mamba_commit)
        logits = torch.randn(6, 33)
        self.assertTrue(
            torch.equal(worker._verify_argmax(logits), torch.argmax(logits, dim=-1))
        )
        self.assertFalse(worker._kv_fuse_store(None, None, None, None, None, None))
        steps, track = worker._mamba_commit_steps(
            batch=SimpleNamespace(mamba_track_indices=None),
            commit_lens=torch.tensor([1, 3, 8], dtype=torch.int32),
            seq_lens_pre_verify=torch.tensor([10, 20, 30]),
            seq_lens_post_verify=torch.tensor([11, 23, 38]),
        )
        self.assertTrue(torch.equal(steps, torch.tensor([0, 2, 7])))
        self.assertEqual(steps.dtype, torch.int64)
        self.assertIsNone(track)

    def test_rocm_prefill_append_skips_tail_graph(self):
        worker = object.__new__(DFlashWorkerV2ROCm)
        worker._tail_graph_on = True
        worker._replay_tail_graph = mock.Mock()
        kw = dict(
            target_hidden=torch.zeros(4, 2),
            cache_loc=torch.zeros(4, dtype=torch.int64),
            positions=torch.zeros(4, dtype=torch.int64),
            extend_lens=torch.tensor([4]),
        )
        with mock.patch.object(
            DFlashWorkerV2, "_append_target_hidden_to_draft_kv_by_loc"
        ) as eager:
            worker._append_target_hidden_to_draft_kv_by_loc(**kw)
        worker._replay_tail_graph.assert_not_called()
        eager.assert_called_once_with(cache_loc_2d=None, commit_lens=None, **kw)


if __name__ == "__main__":
    unittest.main()
