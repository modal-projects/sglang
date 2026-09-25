import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative import dflash_utils
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang.srt.speculative.verify_validity import (
    prepare_verify_rows_,
    write_first_invalid_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _sampling_info(bs):
    return SimpleNamespace(
        is_all_greedy=False,
        temperatures=torch.ones((bs, 1)),
        top_ks=torch.zeros(bs, dtype=torch.int32),
        top_ps=torch.ones(bs),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
    )


class TestDFlashVerifyValidity(CustomTestCase):
    def test_greedy_worker_tracks_only_reached_rows_and_allows_masked_logits(self):
        """An invalid bonus row must taint the attempt; unreachable rows must not."""
        logits = torch.tensor(
            [
                [[-float("inf"), 2.0], [float("nan"), 1.0], [0.0, 1.0]],
                [[-float("inf"), 2.0], [float("inf"), 1.0], [0.0, 1.0]],
                [[-float("inf"), 2.0], [-float("inf")] * 2, [0.0, 1.0]],
                [[-float("inf"), 2.0], [float("nan"), 1.0], [0.0, 1.0]],
                [[-float("inf"), 2.0], [-float("inf"), 2.0], [0.0, 1.0]],
            ]
        )
        candidates = torch.tensor([[0, 1, 1]] * 3 + [[0, 0, 1], [0, 1, 1]])
        syncs = []
        worker = SimpleNamespace(
            _selector_sample=None,
            _use_triton_accept_bonus=False,
            _tp_sync=SimpleNamespace(sync=lambda site, values: syncs.append(site)),
            block_size=3,
        )
        result = DFlashWorkerV2._accept_block(
            worker,
            candidates=candidates,
            next_token_logits=logits.flatten(0, 1),
            sampling_info=SimpleNamespace(is_all_greedy=True),
            draft_input=None,
            prefix_lens=torch.zeros(5, dtype=torch.int32),
            bs=5,
        )
        self.assertEqual(result[0].tolist(), [1, 1, 1, 0, 2])
        self.assertEqual(result[-1].tolist(), [1, 1, 1, -1, -1])
        self.assertEqual(result[-1].dtype, torch.int32)
        self.assertEqual(result[2].tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(syncs, [SpecTpSyncSite.DFLASH_ACCEPT_GREEDY] * 2)

    def test_probability_validity_is_preserved_before_repair(self):
        """Zero, negative, nonfinite, and overflowing probability mass must fail."""
        values = torch.tensor(
            [
                [0.0, 0.0],
                [-0.2, 1.2],
                [float("nan"), 1.0],
                [float("inf"), 0.0],
                [-float("inf"), 1.0],
                [3.0e38, 3.0e38],
                [0.0, 1.0],
            ]
        )[:, None, :]
        valid_rows = prepare_verify_rows_(values, is_logits=False)
        out = torch.empty(7, dtype=torch.int32)
        write_first_invalid_rows(
            valid_rows=valid_rows, correct_lens=torch.zeros(7), out=out
        )
        self.assertEqual(out.tolist(), [0] * 6 + [-1])
        self.assertEqual(values[:, 0].tolist(), [[1.0, 0.0]] * 6 + [[0.0, 1.0]])

    def test_target_only_result_survives_probability_borrow_and_next_verify(self):
        """Reusing sampler and borrowed probability storage must not change metadata."""
        outputs = []

        @contextmanager
        def recycled_probabilities(*, user):
            yield
            for values in outputs:
                values.fill_(float("nan"))

        def sampling(**kwargs):
            probabilities = kwargs["target_probs"]
            self.assertTrue(probabilities.isfinite().all())
            kwargs["accept_token_num"].copy_(torch.tensor([1, 0]))
            kwargs["accept_index"].zero_()
            kwargs["predicts"].zero_()
            outputs.append(probabilities)

        candidates = torch.zeros((2, 3), dtype=torch.int64)
        logits = torch.zeros((6, 2))
        logits[1].fill_(float("nan"))
        logits[4].fill_(-float("inf"))
        first_invalid = torch.empty(2, dtype=torch.int32)
        with (
            patch.object(dflash_utils, "_DFLASH_SAMPLING_VERIFY_AVAILABLE", True),
            patch.object(dflash_utils, "borrow_graph_pool", recycled_probabilities),
            patch.object(
                dflash_utils, "tree_speculative_sampling_target_only", sampling
            ),
        ):
            for current_logits, destination in (
                (logits, first_invalid),
                (torch.zeros_like(logits), torch.empty_like(first_invalid)),
            ):
                result = dflash_utils.compute_dflash_sampling_correct_drafts_and_bonus(
                    candidates=candidates,
                    next_token_logits=current_logits,
                    sampling_info=_sampling_info(2),
                    threshold_single=1.0,
                    threshold_acc=1.0,
                    first_invalid_rows=destination,
                )
                self.assertEqual(len(result), 2)
        self.assertEqual(first_invalid.tolist(), [1, -1])

    def test_greedy_metadata_uses_the_authoritative_tp_rank(self):
        """A peer's finite logits cannot erase rank zero's invalid verify row."""
        packets = []
        results = []
        for rank in range(2):
            index = 0

            def sync(site, values):
                nonlocal index
                if rank == 0:
                    packets.append(values.clone())
                else:
                    values.copy_(packets[index])
                index += 1

            worker = SimpleNamespace(
                _selector_sample=None,
                _use_triton_accept_bonus=False,
                _tp_sync=SimpleNamespace(sync=sync),
                block_size=2,
            )
            logits = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
            if rank == 0:
                logits[1].fill_(float("nan"))
            results.append(
                DFlashWorkerV2._accept_block(
                    worker,
                    candidates=torch.tensor([[0, 1]]),
                    next_token_logits=logits,
                    sampling_info=SimpleNamespace(is_all_greedy=True),
                    draft_input=None,
                    prefix_lens=torch.zeros(1, dtype=torch.int32),
                    bs=1,
                )
            )
        for result in results:
            self.assertEqual(result[-1].tolist(), [1])
            self.assertEqual(result[2].tolist(), [0])


if __name__ == "__main__":
    unittest.main()
