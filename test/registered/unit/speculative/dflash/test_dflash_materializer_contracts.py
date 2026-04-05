import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")

from sglang.srt.speculative.dflash.experiments.fixtures import (
    make_commit_materializer_fixture,
    make_prompt_materializer_fixture,
)
from sglang.srt.speculative.dflash.kernels.materializer import (
    materialize_commit_grouped_control,
    materialize_commit_per_layer_control,
    materialize_prompt_grouped_control,
    materialize_prompt_per_layer_control,
)
from sglang.srt.speculative.dflash.reference.materializer import (
    materialize_commit_reference,
    materialize_prompt_reference,
)


def _assert_cache_close(testcase, expected, actual, *, atol=1e-5, rtol=1e-5):
    testcase.assertTrue(
        torch.allclose(expected.k_cache, actual.k_cache, atol=atol, rtol=rtol)
    )
    testcase.assertTrue(
        torch.allclose(expected.v_cache, actual.v_cache, atol=atol, rtol=rtol)
    )


class TestDFlashMaterializerContracts(unittest.TestCase):
    def test_prompt_per_layer_and_grouped_match_reference(self):
        fixture = make_prompt_materializer_fixture(
            num_layers=6,
            hidden_size=48,
            num_kv_heads=3,
            head_dim=16,
            rotary_dim=16,
            num_slots=256,
            num_tokens=64,
            seed=7,
        )
        reference = materialize_prompt_reference(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            hidden=fixture.hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
        )
        per_layer = materialize_prompt_per_layer_control(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            hidden=fixture.hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
            chunk_size=17,
        )
        grouped = materialize_prompt_grouped_control(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            hidden=fixture.hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
            group_size=2,
            chunk_size=17,
        )
        _assert_cache_close(self, reference, per_layer)
        _assert_cache_close(self, reference, grouped)

    def test_prompt_materializer_preserves_unwritten_slots(self):
        fixture = make_prompt_materializer_fixture(
            num_tokens=32,
            num_slots=128,
            seed=9,
        )
        original = fixture.cache.clone()
        updated = materialize_prompt_per_layer_control(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            hidden=fixture.hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
        )
        written_mask = torch.zeros((fixture.cache.num_slots,), dtype=torch.bool)
        written_mask[fixture.slot_ids] = True
        self.assertTrue(
            torch.equal(
                original.k_cache[:, ~written_mask],
                updated.k_cache[:, ~written_mask],
            )
        )
        self.assertTrue(
            torch.equal(
                original.v_cache[:, ~written_mask],
                updated.v_cache[:, ~written_mask],
            )
        )

    def test_commit_per_layer_and_grouped_match_reference(self):
        fixture = make_commit_materializer_fixture(
            num_layers=6,
            hidden_size=48,
            num_kv_heads=3,
            head_dim=16,
            rotary_dim=16,
            num_slots=512,
            batch_size=8,
            block_size=8,
            seed=11,
        )
        reference = materialize_commit_reference(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
            commit_lens=fixture.commit_lens,
        )
        per_layer = materialize_commit_per_layer_control(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
            commit_lens=fixture.commit_lens,
        )
        grouped = materialize_commit_grouped_control(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
            commit_lens=fixture.commit_lens,
            group_size=3,
        )
        _assert_cache_close(self, reference, per_layer)
        _assert_cache_close(self, reference, grouped)

    def test_zero_commit_lens_is_noop(self):
        fixture = make_commit_materializer_fixture(
            batch_size=4,
            block_size=8,
            seed=13,
        )
        fixture.commit_lens.zero_()
        original = fixture.cache.clone()
        updated = materialize_commit_per_layer_control(
            cache=fixture.cache.clone(),
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
            slot_ids=fixture.slot_ids,
            commit_lens=fixture.commit_lens,
        )
        self.assertTrue(torch.equal(original.k_cache, updated.k_cache))
        self.assertTrue(torch.equal(original.v_cache, updated.v_cache))


if __name__ == "__main__":
    unittest.main()
