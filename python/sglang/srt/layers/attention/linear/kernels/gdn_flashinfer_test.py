"""Tests for FlashInfer GDN checkpoint integration."""

from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_extend_requests_compact_post_chunk_checkpoints():
    kernel = object.__new__(FlashInferGDNKernel)
    kernel.use_state_pool = True
    kernel.supports_state_checkpoints = True
    captured = {}

    def fake_prefill(**kwargs):
        captured.update(kwargs)
        kwargs["output_state"].fill_(5)
        kwargs["state_checkpoints"].fill_(7)
        return torch.zeros_like(kwargs["q"]), kwargs["output_state"]

    kernel._prefill_fn = fake_prefill
    total_tokens = 258
    q = torch.zeros(1, total_tokens, 1, 2)
    ssm_states = torch.zeros(3, 1, 2, 2)

    with patch(
        "sglang.kernels.ops.attention.fla.l2norm.l2norm_fwd",
        side_effect=lambda value: value,
    ):
        output, final_state, checkpoints = kernel.extend(
            q=q,
            k=q,
            v=q,
            g=torch.zeros(1, total_tokens, 1),
            beta=torch.zeros(1, total_tokens, 1),
            ssm_states=ssm_states,
            cache_indices=torch.tensor([0, 1]),
            query_start_loc=torch.tensor([0, 130, 258], dtype=torch.int32),
            return_intermediate_states=True,
            checkpoint_every_n_tokens=64,
        )

    assert output.shape == (1, total_tokens, 1, 2)
    assert final_state is None
    assert checkpoints.shape == (4, 1, 2, 2)
    torch.testing.assert_close(
        captured["checkpoint_cu_starts"], torch.tensor([0, 2, 4])
    )
    assert captured["checkpoint_every_n_tokens"] == 64
    torch.testing.assert_close(checkpoints, torch.full_like(checkpoints, 7))
    torch.testing.assert_close(ssm_states[:2], torch.full_like(ssm_states[:2], 5))


def test_extend_disables_checkpoint_interval_when_capture_is_not_needed():
    kernel = object.__new__(FlashInferGDNKernel)
    kernel.use_state_pool = True
    kernel.supports_state_checkpoints = True
    captured = {}

    def fake_prefill(**kwargs):
        captured.update(kwargs)
        return torch.zeros_like(kwargs["q"]), kwargs["output_state"]

    kernel._prefill_fn = fake_prefill
    q = torch.zeros(1, 128, 1, 2)
    ssm_states = torch.zeros(2, 1, 2, 2)

    with patch(
        "sglang.kernels.ops.attention.fla.l2norm.l2norm_fwd",
        side_effect=lambda value: value,
    ):
        _, _, checkpoints = kernel.extend(
            q=q,
            k=q,
            v=q,
            g=torch.zeros(1, 128, 1),
            beta=torch.zeros(1, 128, 1),
            ssm_states=ssm_states,
            cache_indices=torch.tensor([0]),
            query_start_loc=torch.tensor([0, 128], dtype=torch.int32),
            return_intermediate_states=False,
            checkpoint_every_n_tokens=64,
        )

    assert checkpoints is None
    assert captured["state_checkpoints"] is None
    assert captured["checkpoint_cu_starts"] is None
    assert captured["checkpoint_every_n_tokens"] == 0
