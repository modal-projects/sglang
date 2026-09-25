"""Final chain draws must remain on positive in-vocabulary probability support."""

import os
import unittest

import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"


def _endpoint_mass(kind, vocab_size, offset=0):
    p = torch.zeros(vocab_size, dtype=torch.float32, device=DEVICE)
    if kind == "equal":
        # Twelve rounded copies of 1/12 can have distinct reduction/scan sums.
        p[offset : offset + 12] = 1.0 / 12.0
        last_positive = offset + 11
    else:
        # The small terms sum to one ULP of 0.5, but each ties down when added
        # to 0.5 alone. The four binary fractions sum to exactly 1 in float64.
        p[offset] = p[offset + 8] = 2.0**-25
        p[offset + 4] = 0.5
        p[offset + 128] = 0.5 - 2.0**-24
        last_positive = offset + 128
    return p, last_positive


def _chain_inputs(p, uniform, mode):
    num_slots = 1 if mode == "target" else 3
    proposal = p.numel() - 1
    target = p[None, None].repeat(1, num_slots, 1)
    draft = torch.zeros((1, num_slots - 1, p.numel()), device=DEVICE)
    if mode != "target":
        draft[:, :, proposal] = 1.0
    if mode == "accept":
        target[:, :-1] = draft
    return dict(
        predicts=torch.full((num_slots,), -1, dtype=torch.int32, device=DEVICE),
        accept_index=torch.full((1, num_slots), -1, dtype=torch.int64, device=DEVICE),
        accept_token_num=torch.full((1,), -1, dtype=torch.int32, device=DEVICE),
        candidates=torch.full(
            (1, num_slots), proposal, dtype=torch.int64, device=DEVICE
        ),
        retrive_index=torch.arange(num_slots, dtype=torch.int64, device=DEVICE)[None],
        retrive_next_token=None,
        retrive_next_sibling=None,
        uniform_samples=torch.full((1, num_slots - 1), 0.5, device=DEVICE),
        uniform_samples_for_final_sampling=torch.tensor([uniform], device=DEVICE),
        target_probs=target,
        draft_probs=draft,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )


@unittest.skipUnless(INTERPRET or torch.cuda.is_available(), "CUDA is required")
class TestChainSamplingCDF(CustomTestCase):
    def _check_draw(self, kwargs, expected, mode):
        num_correct = 2 if mode == "accept" else 0
        self.assertEqual(kwargs["accept_token_num"].item(), num_correct)
        bonus = kwargs["predicts"][num_correct].item()
        p = kwargs["target_probs"][0, num_correct]
        mass = p if mode != "reject" else (p - kwargs["draft_probs"][0, 0]).clamp_min(0)
        self.assertTrue(0 <= bonus < p.numel(), bonus)
        self.assertGreater(mass[bonus].item(), 0.0, (bonus, mass[bonus].item()))
        self.assertEqual(bonus, expected)
        self.assertEqual(
            kwargs["accept_index"][0, : num_correct + 1].tolist(),
            list(range(num_correct + 1)),
        )
        if num_correct:
            self.assertEqual(
                kwargs["predicts"][:num_correct].tolist(),
                [p.numel() - 1] * num_correct,
            )

    def test_rounding_gap_selects_last_positive(self):
        """Endpoint roundoff must not emit a zero-mass tail or vocabulary end."""
        uniform = float.fromhex("0x1.fffffep-1")
        for kind in ("equal", "binary"):
            for vocab_size, offset in ((4096, 0), (8193, 0), (4226, 4096)):
                for mode in ("target", "accept", "reject"):
                    with self.subTest(
                        kind=kind, vocab=vocab_size, offset=offset, mode=mode
                    ):
                        p, expected = _endpoint_mass(kind, vocab_size, offset)
                        self.assertLess(expected, vocab_size - 1)
                        kwargs = _chain_inputs(p, uniform, mode)
                        chain_speculative_sampling_triton(**kwargs)
                        self._check_draw(kwargs, expected, mode)

    def test_uniform_boundaries_and_partial_block(self):
        """A CDF tie advances to the next positive lane across block boundaries."""
        p = torch.zeros(4101, device=DEVICE)
        p[17], p[4095], p[4096] = 0.25, 0.5, 0.25
        for uniform, expected in (
            (0.0, 17),
            (float.fromhex("0x1.fffffep-3"), 17),
            (0.25, 4095),
            (0.75, 4096),
            (float.fromhex("0x1.fffffep-1"), 4096),
        ):
            for mode in ("target", "accept", "reject"):
                with self.subTest(uniform=uniform, mode=mode):
                    kwargs = _chain_inputs(p, uniform, mode)
                    chain_speculative_sampling_triton(**kwargs)
                    self._check_draw(kwargs, expected, mode)

    def test_tiny_positive_tail(self):
        """Support is strictly positive mass, without an absolute epsilon cutoff."""
        p = torch.zeros(4101, device=DEVICE)
        p[4096] = p[4099] = 2.0**-100
        for uniform, expected in ((0.0, 4096), (0.5, 4099)):
            for mode in ("target", "accept", "reject"):
                with self.subTest(uniform=uniform, mode=mode):
                    kwargs = _chain_inputs(p, uniform, mode)
                    chain_speculative_sampling_triton(**kwargs)
                    self._check_draw(kwargs, expected, mode)

    def test_no_match_malformed_inputs_do_not_use_support_fallback(self):
        """Unsupported no-match inputs must not be turned into a support draw."""
        for case in (
            "zero",
            "zero_residual",
            "nan",
            "inf",
            "negative_total",
            "one",
            "nan_u",
            "inf_u",
        ):
            with self.subTest(case=case):
                p = torch.zeros(4097, device=DEVICE)
                p[17] = 1.0
                uniform = 0.5
                if case == "zero":
                    p.zero_()
                elif case == "nan":
                    p[0] = float("nan")
                elif case == "inf":
                    p[0] = float("inf")
                elif case == "negative_total":
                    p[0], p[17] = -0.2, 0.1
                elif case == "one":
                    uniform = 1.0
                elif case == "nan_u":
                    uniform = float("nan")
                elif case == "inf_u":
                    uniform = float("inf")
                mode = "reject" if case == "zero_residual" else "target"
                kwargs = _chain_inputs(p, uniform, mode)
                if case == "zero_residual":
                    kwargs["draft_probs"].copy_(p[None, None])
                chain_speculative_sampling_triton(**kwargs)
                self.assertEqual(kwargs["predicts"][0].item(), p.numel() - 1)

    def test_fallback_validates_blocks_before_last_support(self):
        """Finding late support must not skip malformed target mass before it."""
        for invalid in (-1.0, float("nan")):
            with self.subTest(invalid=invalid):
                p, _ = _endpoint_mass("binary", 4226, 4096)
                p[7] = invalid
                kwargs = _chain_inputs(p, float.fromhex("0x1.fffffep-1"), "reject")
                chain_speculative_sampling_triton(**kwargs)
                self.assertEqual(kwargs["accept_token_num"].item(), 0)
                self.assertEqual(kwargs["predicts"][0].item(), p.numel() - 1)

    @unittest.skipIf(INTERPRET, "CUDA graph replay requires CUDA execution")
    def test_graph_replay_uses_current_support(self):
        p, expected = _endpoint_mass("binary", 8193)
        kwargs = _chain_inputs(p, float.fromhex("0x1.fffffep-1"), "reject")
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            chain_speculative_sampling_triton(**kwargs)
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            chain_speculative_sampling_triton(**kwargs)
        for offset in (0, 4096):
            p, expected = _endpoint_mass("binary", 8193, offset)
            kwargs["target_probs"].copy_(p[None, None])
            kwargs["predicts"].fill_(-1)
            graph.replay()
            self._check_draw(kwargs, expected, "reject")


if __name__ == "__main__":
    unittest.main()
