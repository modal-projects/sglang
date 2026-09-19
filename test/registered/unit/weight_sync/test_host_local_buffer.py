from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sglang.srt.weight_sync.host_local_buffer as host_memory
from sglang.srt.weight_sync.host_local_buffer import HostLocalSharedBuffer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_host_local_buffer_is_aligned_unlinked_and_writable(tmp_path):
    with patch.object(host_memory, "_SHARED_MEMORY_ROOT", tmp_path):
        buffer = HostLocalSharedBuffer(
            nbytes=17,
            host_group=None,
            name="test",
        )

    view = None
    try:
        assert buffer.nbytes == 4096
        assert not buffer.path.exists()
        view = buffer.view(4, offset=8)
        view.copy_(torch.tensor([1, 2, 3, 4], dtype=torch.uint8))
        assert buffer.tensor[8:12].tolist() == [1, 2, 3, 4]

        with pytest.raises(ValueError, match="exceeds capacity"):
            buffer.view(2, offset=4095)
    finally:
        if view is not None:
            del view
        buffer.close()


def test_host_local_buffer_can_interleave_future_page_faults(tmp_path):
    with (
        patch.object(host_memory, "_SHARED_MEMORY_ROOT", tmp_path),
        patch.object(
            host_memory,
            "numa_interleave_memory",
            return_value=(0, 1),
        ) as interleave,
    ):
        buffer = HostLocalSharedBuffer(
            nbytes=17,
            host_group=None,
            name="test",
            numa_interleave=True,
        )

    try:
        interleave.assert_called_once_with(buffer.tensor.data_ptr(), 4096)
        assert buffer.interleaved_numa_nodes == (0, 1)
    finally:
        buffer.close()


def test_host_local_buffer_checks_available_capacity(tmp_path):
    filesystem = SimpleNamespace(f_bavail=0, f_frsize=4096)
    with (
        patch.object(host_memory, "_SHARED_MEMORY_ROOT", tmp_path),
        patch.object(host_memory.os, "statvfs", return_value=filesystem),
        pytest.raises(RuntimeError, match="insufficient shared-memory capacity"),
    ):
        HostLocalSharedBuffer(
            nbytes=4096,
            host_group=None,
            name="test",
        )
