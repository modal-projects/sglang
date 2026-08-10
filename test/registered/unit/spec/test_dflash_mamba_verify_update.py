"""Unit tests for committing Mamba state after DFlash verification."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _RecordingAttentionBackend:
    def __init__(self):
        self.calls = []

    def update_mamba_state_after_mtp_verify(self, **kwargs):
        self.calls.append(kwargs)


class TestDFlashMambaVerifyUpdate(CustomTestCase):
    def test_tracks_state_at_accepted_sequence_boundary(self):
        backend = _RecordingAttentionBackend()
        model = object()
        worker = object.__new__(DFlashWorkerV2)
        worker._need_mamba_verify_commit = True
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=backend, model=model)
        )
        seq_lens = torch.tensor([253, 250, 260], dtype=torch.int64)
        batch = SimpleNamespace(
            seq_lens=seq_lens,
            mamba_track_indices=torch.tensor([11, 12, 13], dtype=torch.int64),
            req_pool_indices=torch.tensor([21, 22, 23], dtype=torch.int64),
        )

        with mock.patch(
            "sglang.srt.speculative.dflash_worker_v2.get_exec",
            return_value=SimpleNamespace(
                mamba=SimpleNamespace(mamba_track_interval=256)
            ),
        ):
            worker._update_target_mamba_state_after_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens,
                commit_lens=torch.tensor([4, 5, 3], dtype=torch.int32),
            )

        self.assertEqual(len(backend.calls), 1)
        call = backend.calls[0]
        torch.testing.assert_close(
            call["last_correct_step_indices"],
            torch.tensor([3, 4, 2], dtype=torch.int64),
        )
        torch.testing.assert_close(
            call["mamba_steps_to_track"],
            torch.tensor([2, -1, -1], dtype=torch.int64),
        )
        self.assertIs(call["mamba_track_indices"], batch.mamba_track_indices)
        self.assertIs(call["model"], model)
        torch.testing.assert_close(call["req_pool_indices"], batch.req_pool_indices)


if __name__ == "__main__":
    unittest.main()
