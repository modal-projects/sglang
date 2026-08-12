"""update_mamba_state_after_mtp_verify dispatch: the DSPARK/DFLASH direct
commit must route KDA to the KDA replay, GDN fold to the GDN fold, and only
the legacy (no-replayssm) pool to the intermediate-state scatter."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_backend(*, replayssm_is_kda: bool, replayssm_spec_fold: bool):
    full_attn_backend = MagicMock()
    full_attn_backend.needs_cpu_seq_lens = False
    linear_attn_backend = MagicMock()
    linear_attn_backend.needs_cpu_seq_lens = False
    backend = HybridLinearAttnBackend(
        full_attn_backend=full_attn_backend,
        linear_attn_backend=linear_attn_backend,
        full_attn_layers=[0],
    )

    mamba_pool = MagicMock(spec=[])  # bare object: getattr defaults must miss
    mamba_pool.replayssm_is_kda = replayssm_is_kda
    mamba_pool.replayssm_spec_fold = replayssm_spec_fold

    req_pool = MagicMock()
    req_pool.mamba_pool = mamba_pool
    linear_attn_backend.req_to_token_pool = req_pool
    linear_attn_backend.forward_metadata.mamba_cache_indices = torch.arange(
        8, dtype=torch.int64
    )
    return backend, req_pool


class TestHybridMtpVerifyCommitDispatch(CustomTestCase):
    def _call(self, backend):
        last_correct_step_indices = torch.tensor([2, 0, 4], dtype=torch.int64)
        backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=None,
            mamba_steps_to_track=None,
            model=MagicMock(),
        )
        return last_correct_step_indices

    def test_gdn_fold_routes_to_gdn_commit(self):
        backend, req_pool = _make_backend(
            replayssm_is_kda=False, replayssm_spec_fold=True
        )
        with patch(
            "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold"
            ".commit_gdn_replayssm_fold_after_verify"
        ) as gdn_commit, patch(
            "sglang.srt.layers.attention.hybrid_linear_attn_backend"
            ".scatter_mamba_states_after_mtp_verify"
        ) as scatter:
            last_correct_step_indices = self._call(backend)

        gdn_commit.assert_called_once()
        scatter.assert_not_called()
        kwargs = gdn_commit.call_args.kwargs
        self.assertTrue(
            torch.equal(kwargs["accept_lens"], last_correct_step_indices + 1)
        )
        self.assertTrue(
            torch.equal(
                kwargs["last_correct_step_indices"], last_correct_step_indices
            )
        )
        self.assertIs(
            kwargs["spec_state"],
            req_pool.get_speculative_mamba2_params_all_layers.return_value,
        )
        # Chain layout: 3 requests -> the first 3 planned slots.
        self.assertTrue(
            torch.equal(
                kwargs["state_batch_indices"],
                torch.arange(3, dtype=torch.int64),
            )
        )

    def test_kda_routes_to_kda_commit(self):
        backend, _ = _make_backend(replayssm_is_kda=True, replayssm_spec_fold=True)
        with patch(
            "sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode"
            ".commit_kda_replayssm_after_verify"
        ) as kda_commit, patch(
            "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold"
            ".commit_gdn_replayssm_fold_after_verify"
        ) as gdn_commit, patch(
            "sglang.srt.layers.attention.hybrid_linear_attn_backend"
            ".scatter_mamba_states_after_mtp_verify"
        ) as scatter:
            self._call(backend)

        kda_commit.assert_called_once()
        gdn_commit.assert_not_called()
        scatter.assert_not_called()

    def test_legacy_pool_routes_to_scatter(self):
        backend, _ = _make_backend(replayssm_is_kda=False, replayssm_spec_fold=False)
        with patch(
            "sglang.srt.layers.attention.hybrid_linear_attn_backend"
            ".scatter_mamba_states_after_mtp_verify"
        ) as scatter:
            self._call(backend)

        scatter.assert_called_once()


if __name__ == "__main__":
    unittest.main()
