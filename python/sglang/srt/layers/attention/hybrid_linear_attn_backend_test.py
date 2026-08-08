"""Tests for hybrid linear-attention checkpoint tracking."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_aligned_checkpoint_copies_final_state_without_intermediate_states():
    backend = object.__new__(MambaAttnBackendBase)
    states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    expected = states[1].clone()
    metadata = SimpleNamespace(
        has_mamba_track_mask=True,
        track_ssm_h_src=torch.empty(0, dtype=torch.int64),
        track_ssm_h_dst=torch.empty(0, dtype=torch.int64),
        track_ssm_final_src=torch.tensor([1]),
        track_ssm_final_dst=torch.tensor([2]),
    )

    backend._track_mamba_state_extend(
        forward_batch=SimpleNamespace(),
        h=None,
        ssm_states=states,
        forward_metadata=metadata,
    )

    torch.testing.assert_close(states[2], expected)


def test_single_post_chunk_checkpoint_preserves_head_axis():
    backend = object.__new__(MambaAttnBackendBase)
    states = torch.zeros(2, 2, 2, 2)
    checkpoint = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    metadata = SimpleNamespace(
        has_mamba_track_mask=True,
        track_ssm_h_src=torch.tensor([0]),
        track_ssm_h_dst=torch.tensor([1]),
        track_ssm_final_src=torch.empty(0, dtype=torch.int64),
        track_ssm_final_dst=torch.empty(0, dtype=torch.int64),
    )

    backend._track_mamba_state_extend(
        forward_batch=SimpleNamespace(),
        h=checkpoint,
        ssm_states=states,
        forward_metadata=metadata,
    )

    torch.testing.assert_close(states[1], checkpoint[0])


def test_post_chunk_checkpoint_index_selects_last_complete_chunk():
    backend = object.__new__(MambaAttnBackendBase)
    backend.device = torch.device("cpu")
    backend.uses_post_chunk_checkpoints = True
    forward_batch = SimpleNamespace(
        mamba_track_mask=torch.tensor([True, True, False]),
        extend_seq_lens=torch.tensor([270, 128, 63]),
        mamba_track_indices=torch.tensor([7, 8, 9]),
        mamba_track_seqlens=torch.tensor([270, 128, -1]),
        extend_prefix_lens=torch.tensor([0, 0, 0]),
    )
    server_args = SimpleNamespace(mamba_cache_chunk_size=64)

    with patch(
        "sglang.srt.layers.attention.hybrid_linear_attn_backend.get_server_args",
        return_value=server_args,
    ):
        h_src, h_dst, final_src, final_dst = backend._init_track_ssm_indices(
            mamba_cache_indices=torch.tensor([1, 2, 3]),
            forward_batch=forward_batch,
        )

    # 270 tokens has checkpoints after 64/128/192/256; index 3 is the state
    # at 256. The exactly aligned 128-token row uses its final pool state.
    torch.testing.assert_close(h_src, torch.tensor([3]))
    torch.testing.assert_close(h_dst, torch.tensor([7]))
    torch.testing.assert_close(final_src, torch.tensor([2]))
    torch.testing.assert_close(final_dst, torch.tensor([8]))
