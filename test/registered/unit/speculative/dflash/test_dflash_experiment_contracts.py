import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

from sglang.srt.speculative.dflash.contracts import (
    STATUS_ACTIVE,
    STATUS_EOS_SEEN,
    STATUS_FINISHED,
    STATUS_STOPPED_BY_TOKEN,
    DFlashRequestStateTable,
)
from sglang.srt.speculative.dflash.experiments.fixtures import (
    make_accept_bonus_fixture,
    make_prepare_block_fixture,
    make_publish_state_fixture,
)
from sglang.srt.speculative.dflash.kernels.accept_bonus import accept_bonus_control
from sglang.srt.speculative.dflash.kernels.prepare_block import prepare_block_control
from sglang.srt.speculative.dflash.kernels.publish_state import publish_state_control
from sglang.srt.speculative.dflash.reference.core import (
    accept_bonus_reference,
    prepare_block_reference,
    publish_state_reference,
)


class TestDFlashReferenceContracts(unittest.TestCase):
    def test_prepare_block_reference_filters_stale_and_stopped_rows(self):
        state = DFlashRequestStateTable(
            committed_len=torch.tensor([5, 7, 9], dtype=torch.int32),
            reserved_len=torch.tensor([16, 16, 16], dtype=torch.int32),
            next_verified_id=torch.tensor([101, 202, 303], dtype=torch.int32),
            generation=torch.tensor([11, 22, 33], dtype=torch.int32),
            status_flags=torch.tensor(
                [STATUS_ACTIVE, STATUS_ACTIVE, STATUS_ACTIVE | STATUS_FINISHED],
                dtype=torch.int32,
            ),
        )
        req_to_token = torch.arange(3 * 32, dtype=torch.int64).view(3, 32)
        result = prepare_block_reference(
            state=state,
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
            req_generation=torch.tensor([11, 999, 33], dtype=torch.int32),
            req_to_token=req_to_token,
            block_size=4,
            mask_token_id=151643,
        )

        self.assertTrue(result.active_mask[0].item())
        self.assertFalse(result.active_mask[1].item())
        self.assertFalse(result.active_mask[2].item())
        self.assertTrue(
            torch.equal(
                result.query_input_ids[0],
                torch.tensor([101, 151643, 151643, 151643], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                result.query_positions[0],
                torch.tensor([5, 6, 7, 8], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                result.query_slot_ids[0],
                torch.tensor([5, 6, 7, 8], dtype=torch.int64),
            )
        )
        self.assertEqual(int(result.query_slot_ids[1, 0].item()), -1)
        self.assertEqual(int(result.query_slot_ids[2, 0].item()), -1)

    def test_prepare_block_control_matches_reference(self):
        fixture = make_prepare_block_fixture(bucket_bs=8, block_size=8, seed=3)
        expected = prepare_block_reference(
            state=fixture.state,
            req_pool_indices=fixture.req_pool_indices,
            req_generation=fixture.req_generation,
            req_to_token=fixture.req_to_token,
            block_size=fixture.block_size,
            mask_token_id=fixture.mask_token_id,
        )
        actual = prepare_block_control(
            state=fixture.state,
            req_pool_indices=fixture.req_pool_indices,
            req_generation=fixture.req_generation,
            req_to_token=fixture.req_to_token,
            block_size=fixture.block_size,
            mask_token_id=fixture.mask_token_id,
        )
        self.assertTrue(torch.equal(expected.query_input_ids, actual.query_input_ids))
        self.assertTrue(torch.equal(expected.query_positions, actual.query_positions))
        self.assertTrue(torch.equal(expected.query_slot_ids, actual.query_slot_ids))
        self.assertTrue(torch.equal(expected.emit_ids, actual.emit_ids))
        self.assertTrue(torch.equal(expected.active_mask, actual.active_mask))
        self.assertTrue(torch.equal(expected.sample_indices, actual.sample_indices))

    def test_accept_bonus_reference_honors_committed_prefix_stop_semantics(self):
        result = accept_bonus_reference(
            emit_ids=torch.tensor([[11, 22, 33, 44]], dtype=torch.int32),
            target_top1=torch.tensor([[22, 33, 99, 55]], dtype=torch.int32),
            active_mask=torch.tensor([True]),
            eos_token_ids=[33],
            stop_token_ids=[99],
        )
        self.assertEqual(int(result.accept_lens[0].item()), 2)
        self.assertEqual(int(result.commit_lens[0].item()), 3)
        self.assertEqual(int(result.bonus_ids[0].item()), 99)
        self.assertTrue(bool(result.gpu_stop_flags[0].item() & STATUS_EOS_SEEN))
        self.assertFalse(
            bool(result.gpu_stop_flags[0].item() & STATUS_STOPPED_BY_TOKEN)
        )

    def test_accept_bonus_control_matches_reference(self):
        fixture = make_accept_bonus_fixture(bucket_bs=8, block_size=8, seed=4)
        expected = accept_bonus_reference(
            emit_ids=fixture.emit_ids,
            target_top1=fixture.target_top1,
            active_mask=fixture.active_mask,
            eos_token_ids=fixture.eos_token_ids,
            stop_token_ids=fixture.stop_token_ids,
        )
        actual = accept_bonus_control(
            emit_ids=fixture.emit_ids,
            target_top1=fixture.target_top1,
            active_mask=fixture.active_mask,
            eos_token_ids=fixture.eos_token_ids,
            stop_token_ids=fixture.stop_token_ids,
        )
        self.assertTrue(torch.equal(expected.accept_lens, actual.accept_lens))
        self.assertTrue(torch.equal(expected.commit_lens, actual.commit_lens))
        self.assertTrue(torch.equal(expected.bonus_ids, actual.bonus_ids))
        self.assertTrue(torch.equal(expected.gpu_stop_flags, actual.gpu_stop_flags))

    def test_publish_state_reference_updates_gpu_truth_and_clears_active(self):
        state = DFlashRequestStateTable(
            committed_len=torch.tensor([5, 10], dtype=torch.int32),
            reserved_len=torch.tensor([32, 32], dtype=torch.int32),
            next_verified_id=torch.tensor([101, 202], dtype=torch.int32),
            generation=torch.tensor([7, 9], dtype=torch.int32),
            status_flags=torch.tensor(
                [STATUS_ACTIVE, STATUS_ACTIVE], dtype=torch.int32
            ),
        )
        updated = publish_state_reference(
            state=state,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            req_generation=torch.tensor([7, 9], dtype=torch.int32),
            commit_lens=torch.tensor([3, 2], dtype=torch.int32),
            bonus_ids=torch.tensor([1111, 2222], dtype=torch.int32),
            gpu_stop_flags=torch.tensor([0, STATUS_FINISHED], dtype=torch.int32),
        )
        self.assertEqual(int(updated.committed_len[0].item()), 8)
        self.assertEqual(int(updated.next_verified_id[0].item()), 1111)
        self.assertEqual(int(updated.committed_len[1].item()), 12)
        self.assertEqual(int(updated.next_verified_id[1].item()), 2222)
        self.assertTrue(bool(updated.status_flags[1].item() & STATUS_FINISHED))
        self.assertFalse(bool(updated.status_flags[1].item() & STATUS_ACTIVE))

    def test_publish_state_control_matches_reference(self):
        fixture = make_publish_state_fixture(bucket_bs=8, block_size=8, seed=5)
        expected = publish_state_reference(
            state=fixture.state,
            req_pool_indices=fixture.req_pool_indices,
            req_generation=fixture.req_generation,
            commit_lens=fixture.commit_lens,
            bonus_ids=fixture.bonus_ids,
            gpu_stop_flags=fixture.gpu_stop_flags,
        )
        actual = publish_state_control(
            state=fixture.state,
            req_pool_indices=fixture.req_pool_indices,
            req_generation=fixture.req_generation,
            commit_lens=fixture.commit_lens,
            bonus_ids=fixture.bonus_ids,
            gpu_stop_flags=fixture.gpu_stop_flags,
        )
        self.assertTrue(torch.equal(expected.committed_len, actual.committed_len))
        self.assertTrue(torch.equal(expected.reserved_len, actual.reserved_len))
        self.assertTrue(torch.equal(expected.next_verified_id, actual.next_verified_id))
        self.assertTrue(torch.equal(expected.generation, actual.generation))
        self.assertTrue(torch.equal(expected.status_flags, actual.status_flags))


if __name__ == "__main__":
    unittest.main()
