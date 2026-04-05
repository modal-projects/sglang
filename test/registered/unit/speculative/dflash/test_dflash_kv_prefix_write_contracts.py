import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")

from sglang.srt.speculative.dflash.experiments.fixtures import (
    make_commit_write_fixture,
    make_prompt_write_fixture,
)
from sglang.srt.speculative.dflash.kernels.kv_prefix_write import (
    write_commit_masked_dummy_control,
    write_commit_prefix_flatten_control,
    write_commit_prefix_rowwise_control,
    write_prompt_index_copy_control,
)
from sglang.srt.speculative.dflash.reference.kv_prefix_write import (
    write_commit_prefix_reference,
    write_prompt_reference,
)


def _assert_cache_equal(testcase, expected, actual, *, ignore_slot_id=None):
    if ignore_slot_id is None:
        testcase.assertTrue(torch.equal(expected.k_cache, actual.k_cache))
        testcase.assertTrue(torch.equal(expected.v_cache, actual.v_cache))
        return
    mask = torch.ones((expected.num_slots,), dtype=torch.bool)
    mask[ignore_slot_id] = False
    testcase.assertTrue(torch.equal(expected.k_cache[:, mask], actual.k_cache[:, mask]))
    testcase.assertTrue(torch.equal(expected.v_cache[:, mask], actual.v_cache[:, mask]))


class TestDFlashKVPrefixWriteContracts(unittest.TestCase):
    def test_prompt_index_copy_matches_reference(self):
        fixture = make_prompt_write_fixture(
            num_layers=6,
            num_kv_heads=3,
            head_dim=16,
            num_slots=256,
            num_tokens=64,
            seed=21,
        )
        reference = write_prompt_reference(
            cache=fixture.cache.clone(),
            config=fixture.config,
            slot_ids=fixture.slot_ids,
            cache_k=fixture.cache_k,
            cache_v=fixture.cache_v,
        )
        control = write_prompt_index_copy_control(
            cache=fixture.cache.clone(),
            config=fixture.config,
            slot_ids=fixture.slot_ids,
            cache_k=fixture.cache_k,
            cache_v=fixture.cache_v,
        )
        _assert_cache_equal(self, reference, control)

    def test_commit_rowwise_and_flatten_match_reference(self):
        fixture = make_commit_write_fixture(
            num_layers=6,
            num_kv_heads=3,
            head_dim=16,
            num_slots=512,
            batch_size=8,
            block_size=8,
            seed=23,
        )
        reference = write_commit_prefix_reference(
            cache=fixture.cache.clone(),
            config=fixture.config,
            slot_ids_2d=fixture.slot_ids_2d,
            commit_lens=fixture.commit_lens,
            cache_k=fixture.cache_k,
            cache_v=fixture.cache_v,
        )
        rowwise = write_commit_prefix_rowwise_control(
            cache=fixture.cache.clone(),
            config=fixture.config,
            slot_ids_2d=fixture.slot_ids_2d,
            commit_lens=fixture.commit_lens,
            cache_k=fixture.cache_k,
            cache_v=fixture.cache_v,
        )
        flatten = write_commit_prefix_flatten_control(
            cache=fixture.cache.clone(),
            config=fixture.config,
            slot_ids_2d=fixture.slot_ids_2d,
            commit_lens=fixture.commit_lens,
            cache_k=fixture.cache_k,
            cache_v=fixture.cache_v,
        )
        _assert_cache_equal(self, reference, rowwise)
        _assert_cache_equal(self, reference, flatten)

    def test_commit_masked_dummy_only_mutates_dummy_extra_slot(self):
        fixture = make_commit_write_fixture(
            batch_size=8,
            block_size=8,
            num_slots=512,
            seed=25,
        )
        reference = write_commit_prefix_reference(
            cache=fixture.cache.clone(),
            config=fixture.config,
            slot_ids_2d=fixture.slot_ids_2d,
            commit_lens=fixture.commit_lens,
            cache_k=fixture.cache_k,
            cache_v=fixture.cache_v,
        )
        masked = write_commit_masked_dummy_control(
            cache=fixture.cache.clone(),
            config=fixture.config,
            slot_ids_2d=fixture.slot_ids_2d,
            commit_lens=fixture.commit_lens,
            cache_k=fixture.cache_k,
            cache_v=fixture.cache_v,
            dummy_slot_id=fixture.dummy_slot_id,
        )
        _assert_cache_equal(
            self, reference, masked, ignore_slot_id=fixture.dummy_slot_id
        )

    def test_zero_commit_lens_is_noop_for_write_controls(self):
        fixture = make_commit_write_fixture(
            batch_size=4,
            block_size=8,
            seed=27,
        )
        fixture.commit_lens.zero_()
        original = fixture.cache.clone()
        rowwise = write_commit_prefix_rowwise_control(
            cache=fixture.cache.clone(),
            config=fixture.config,
            slot_ids_2d=fixture.slot_ids_2d,
            commit_lens=fixture.commit_lens,
            cache_k=fixture.cache_k,
            cache_v=fixture.cache_v,
        )
        flatten = write_commit_prefix_flatten_control(
            cache=fixture.cache.clone(),
            config=fixture.config,
            slot_ids_2d=fixture.slot_ids_2d,
            commit_lens=fixture.commit_lens,
            cache_k=fixture.cache_k,
            cache_v=fixture.cache_v,
        )
        self.assertTrue(torch.equal(original.k_cache, rowwise.k_cache))
        self.assertTrue(torch.equal(original.v_cache, rowwise.v_cache))
        self.assertTrue(torch.equal(original.k_cache, flatten.k_cache))
        self.assertTrue(torch.equal(original.v_cache, flatten.v_cache))


if __name__ == "__main__":
    unittest.main()
