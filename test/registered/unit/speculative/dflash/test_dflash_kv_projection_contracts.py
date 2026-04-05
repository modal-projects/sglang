import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")

from sglang.srt.speculative.dflash.experiments.common import (
    assert_dataclass_tensors_equal,
)
from sglang.srt.speculative.dflash.experiments.fixtures import (
    make_commit_projection_fixture,
    make_prompt_projection_fixture,
)
from sglang.srt.speculative.dflash.kernels.kv_projection import (
    project_commit_grouped_control,
    project_commit_per_layer_control,
    project_prompt_grouped_control,
    project_prompt_per_layer_control,
)
from sglang.srt.speculative.dflash.reference.kv_projection import (
    project_commit_reference,
    project_prompt_reference,
)


class TestDFlashKVProjectionContracts(unittest.TestCase):
    def test_prompt_projection_controls_match_reference(self):
        fixture = make_prompt_projection_fixture(
            num_layers=6,
            hidden_size=48,
            num_kv_heads=3,
            head_dim=16,
            rotary_dim=16,
            num_tokens=64,
            seed=17,
        )
        reference = project_prompt_reference(
            config=fixture.config,
            weights=fixture.weights,
            hidden=fixture.hidden,
            positions=fixture.positions,
        )
        per_layer = project_prompt_per_layer_control(
            config=fixture.config,
            weights=fixture.weights,
            hidden=fixture.hidden,
            positions=fixture.positions,
            chunk_size=19,
        )
        grouped = project_prompt_grouped_control(
            config=fixture.config,
            weights=fixture.weights,
            hidden=fixture.hidden,
            positions=fixture.positions,
            group_size=2,
            chunk_size=19,
        )
        assert_dataclass_tensors_equal(reference, per_layer, atol=1e-5, rtol=1e-5)
        assert_dataclass_tensors_equal(reference, grouped, atol=1e-5, rtol=1e-5)

    def test_prompt_projection_zero_tokens_preserves_shape(self):
        fixture = make_prompt_projection_fixture(
            num_layers=4,
            hidden_size=32,
            num_kv_heads=2,
            head_dim=16,
            rotary_dim=16,
            num_tokens=0,
            seed=19,
        )
        projected = project_prompt_per_layer_control(
            config=fixture.config,
            weights=fixture.weights,
            hidden=fixture.hidden,
            positions=fixture.positions,
        )
        self.assertEqual(
            tuple(projected.cache_k.shape),
            (
                fixture.config.num_layers,
                0,
                fixture.config.num_kv_heads,
                fixture.config.head_dim,
            ),
        )
        self.assertEqual(projected.cache_v.numel(), 0)

    def test_commit_projection_controls_match_reference(self):
        fixture = make_commit_projection_fixture(
            num_layers=6,
            hidden_size=48,
            num_kv_heads=3,
            head_dim=16,
            rotary_dim=16,
            batch_size=8,
            block_size=8,
            seed=23,
        )
        reference = project_commit_reference(
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
        )
        per_layer = project_commit_per_layer_control(
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
        )
        grouped = project_commit_grouped_control(
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
            group_size=3,
        )
        assert_dataclass_tensors_equal(reference, per_layer, atol=1e-5, rtol=1e-5)
        assert_dataclass_tensors_equal(reference, grouped, atol=1e-5, rtol=1e-5)

    def test_commit_projection_matches_flat_prompt_projection(self):
        fixture = make_commit_projection_fixture(
            num_layers=4,
            hidden_size=32,
            num_kv_heads=2,
            head_dim=16,
            rotary_dim=16,
            batch_size=3,
            block_size=5,
            seed=29,
        )
        projected = project_commit_reference(
            config=fixture.config,
            weights=fixture.weights,
            verify_hidden=fixture.verify_hidden,
            positions=fixture.positions,
        )
        flat_prompt = project_prompt_reference(
            config=fixture.config,
            weights=fixture.weights,
            hidden=fixture.verify_hidden.reshape(-1, fixture.config.hidden_size),
            positions=fixture.positions.reshape(-1),
        )
        self.assertTrue(
            torch.allclose(
                projected.cache_k.reshape_as(flat_prompt.cache_k),
                flat_prompt.cache_k,
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                projected.cache_v.reshape_as(flat_prompt.cache_v),
                flat_prompt.cache_v,
                atol=1e-5,
                rtol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
