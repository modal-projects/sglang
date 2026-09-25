import types
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SAMPLED = types.SimpleNamespace(is_all_greedy=False)


class _FakeTpSync:
    def __init__(self):
        self.calls = []

    def enabled(self, site):
        return True

    def sync(self, site, values):
        self.calls.append((site, values.clone()))
        values.fill_(17)
        return values


def _worker(*, selector, is_domino=False):
    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker._tp_sync = _FakeTpSync()
    worker.selector = selector
    worker._is_domino = is_domino
    worker._selector_sampling_enabled = True
    return worker


def _sampling_info(top_ks):
    batch = ScheduleBatch.__new__(ScheduleBatch)
    batch.device = "cpu"
    batch.reqs = [
        types.SimpleNamespace(
            sampling_params=SamplingParams(top_k=top_k),
            return_sampling_mask=False,
            custom_logit_processor=None,
        )
        for top_k in top_ks
    ]
    execution = types.SimpleNamespace(
        deterministic=types.SimpleNamespace(enable_deterministic_inference=False),
        features=types.SimpleNamespace(enable_custom_logit_processor=False),
    )
    with mock.patch(
        "sglang.srt.sampling.sampling_batch_info.get_exec", return_value=execution
    ):
        return SamplingBatchInfo.from_schedule_batch(batch, vocab_size=32)


def _without_scalar_readback(worker, draft_next, sampling_info):
    with (
        mock.patch.object(
            torch.Tensor,
            "__bool__",
            side_effect=AssertionError("draft synchronization read a device scalar"),
        ),
        mock.patch.object(
            torch.Tensor,
            "item",
            side_effect=AssertionError("draft synchronization read a device scalar"),
        ),
    ):
        return worker._sync_greedy_draft(draft_next, sampling_info)


class TestDflashGreedyDraftTpSync(CustomTestCase):
    def _assert_synced(self, worker, draft_next, synced):
        assert len(worker._tp_sync.calls) == 1
        site, values = worker._tp_sync.calls[0]
        assert site == SpecTpSyncSite.DFLASH_DRAFT_GREEDY
        assert torch.equal(values, torch.tensor([[1, 2, 3]], dtype=torch.int64))
        assert torch.equal(synced, torch.full_like(draft_next, 17))

    def test_greedy_selector_draft_is_synchronized(self):
        worker = _worker(selector=object())
        draft_next = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        synced = worker._sync_greedy_draft(draft_next, None)
        self._assert_synced(worker, draft_next, synced)

    def test_sampled_batch_without_selector_is_synchronized(self):
        worker = _worker(selector=None)
        draft_next = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        synced = worker._sync_greedy_draft(draft_next, _SAMPLED)
        self._assert_synced(worker, draft_next, synced)

    def test_sampled_domino_draft_is_synchronized(self):
        worker = _worker(selector=object(), is_domino=True)
        draft_next = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        synced = worker._sync_greedy_draft(draft_next, _SAMPLED)
        self._assert_synced(worker, draft_next, synced)

    def test_sampled_selector_draft_is_not_resynchronized(self):
        """Fully sampled batches must not wait for a device mask readback."""
        worker = _worker(selector=object())
        sampling_info = _sampling_info([8, 8])
        draft_next = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)

        synced = _without_scalar_readback(worker, draft_next, sampling_info)

        assert worker._tp_sync.calls == []
        assert synced is draft_next

    def test_mixed_selector_batch_syncs_only_greedy_rows(self):
        """Mixed rows retain their device mask without reading it on the host."""
        worker = _worker(selector=object())
        sampling_info = _sampling_info([1, 20])
        draft_next = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)

        synced = _without_scalar_readback(worker, draft_next, sampling_info)

        # The full proposal is broadcast from rank 0 (the fake fills it with
        # 17), but only the greedy row takes rank 0's tokens; the sampled row
        # keeps its rank-local draw.
        assert len(worker._tp_sync.calls) == 1
        site, values = worker._tp_sync.calls[0]
        assert site == SpecTpSyncSite.DFLASH_DRAFT_GREEDY
        assert torch.equal(values, draft_next)
        assert torch.equal(
            synced, torch.tensor([[17, 17, 17], [4, 5, 6]], dtype=torch.int64)
        )
        # The rank-local proposal is not clobbered in place.
        assert torch.equal(
            draft_next, torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
        )

    def test_selector_sampling_disabled_syncs_whole_tensor(self):
        worker = _worker(selector=object())
        worker._selector_sampling_enabled = False
        sampling_info = _sampling_info([1, 20])
        draft_next = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)

        synced = worker._sync_greedy_draft(draft_next, sampling_info)

        # Selector sampling is disabled: every proposal is a rank-local argmax
        # regardless of requested top_k, so the whole tensor is synced from
        # rank 0 (the fake fills it with 17).
        assert len(worker._tp_sync.calls) == 1
        site, values = worker._tp_sync.calls[0]
        assert site == SpecTpSyncSite.DFLASH_DRAFT_GREEDY
        assert torch.equal(values, draft_next)
        assert torch.equal(synced, torch.full_like(draft_next, 17))

    def test_disabled_site_skips_device_mask_and_preserves_storage(self):
        """TP1 and an excluded greedy site do no device work for mixed rows."""
        for world_size, policy in ((1, "all"), (2, "all,-18")):
            with self.subTest(world_size=world_size, policy=policy):
                worker = _worker(selector=object())
                group = types.SimpleNamespace(world_size=world_size, rank_in_group=0)
                with envs.SGLANG_SPEC_TP_SYNC.override(policy):
                    worker._tp_sync = SpecTpSync(group)
                draft_next = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
                synced = _without_scalar_readback(
                    worker, draft_next, _sampling_info([1, 8])
                )
                self.assertIs(synced, draft_next)
                self.assertTrue(
                    torch.equal(synced, torch.tensor([[1, 2, 3], [4, 5, 6]]))
                )

    def test_filtered_and_merged_flags_keep_device_rows_authoritative(self):
        """A conservative host flag cannot turn filtered sampled rows greedy."""
        worker = _worker(selector=object())
        info = _sampling_info([1, 8])
        info.filter_batch([1], torch.tensor([1]))
        self.assertTrue(info.is_any_greedy)
        draft_next = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        synced = _without_scalar_readback(worker, draft_next, info)
        self.assertTrue(torch.equal(synced, draft_next))

        for info in (info, _sampling_info([8])):
            info.merge_batch(_sampling_info([1]))
            self.assertTrue(info.is_any_greedy)
            forwarded = info.copy_for_forward()
            self.assertIsInstance(forwarded, SamplingBatchInfo)
            draft_next = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
            synced = _without_scalar_readback(worker, draft_next, forwarded)
            self.assertTrue(
                torch.equal(synced, torch.tensor([[1, 2, 3], [17, 17, 17]]))
            )


if __name__ == "__main__":
    unittest.main()
