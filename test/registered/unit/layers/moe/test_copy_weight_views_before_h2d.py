import concurrent.futures
from unittest import mock

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
    _batched_weight_copy_workers,
    _copy_weight_view_before_h2d,
)
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _CopyOnlyMxfp4MoE(FusedMoE):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.quant_method = object.__new__(Mxfp4MoEMethod)

    def weight_loader(self, target, source):
        self._copy_loaded_weight(target, source)


def _assert_independent_copy(source: torch.Tensor, result: torch.Tensor) -> None:
    assert torch.equal(result, source)
    assert result is not source
    assert result.is_contiguous()
    assert result.storage_offset() == 0
    assert result.untyped_storage().nbytes() == result.numel() * result.element_size()


def test_skips_copy_for_exact_storage():
    source = torch.arange(16).reshape(4, 4)
    assert _copy_weight_view_before_h2d(source) is source


def test_copies_zero_offset_storage_view():
    backing = torch.arange(32).reshape(8, 4)
    source = backing.narrow(0, 0, 2)
    assert source.storage_offset() == 0
    assert source.untyped_storage().nbytes() > source.numel() * source.element_size()
    _assert_independent_copy(source, _copy_weight_view_before_h2d(source))


def test_copies_nonzero_offset_storage_view():
    backing = torch.arange(32).reshape(8, 4)
    source = backing.narrow(0, 6, 2)
    assert source.storage_offset() != 0
    _assert_independent_copy(source, _copy_weight_view_before_h2d(source))


def test_copies_noncontiguous_tensor():
    source = torch.arange(16).reshape(4, 4).transpose(0, 1)
    assert not source.is_contiguous()
    _assert_independent_copy(source, _copy_weight_view_before_h2d(source))


def test_batched_loader_preserves_native_copy_views():
    layer = _CopyOnlyMxfp4MoE()
    storage = torch.zeros((6, 5), dtype=torch.float32)
    targets = [storage[1:3], storage[3:6]]
    sources = [
        torch.arange(10, dtype=torch.float32).reshape(2, 5),
        torch.arange(15, dtype=torch.float32).reshape(5, 3).t(),
    ]

    layer.batched_weight_loader(
        [((target, source), {}) for target, source in zip(targets, sources)]
    )

    for target, source in zip(targets, sources):
        torch.testing.assert_close(target, source)


def test_batched_loader_orders_overlapping_destinations():
    layer = _CopyOnlyMxfp4MoE()
    overlap_target = torch.zeros(8)
    first = torch.ones_like(overlap_target)
    second = torch.full_like(overlap_target, 2)
    independent_targets = [torch.zeros(1024), torch.zeros(1024)]
    independent_sources = [
        torch.full_like(independent_targets[0], 3),
        torch.full_like(independent_targets[1], 4),
    ]

    class RecordingExecutor:
        def __init__(self, delegate):
            self.delegate = delegate
            self.submissions = 0

        def submit(self, function, *args, **kwargs):
            self.submissions += 1
            return self.delegate.submit(function, *args, **kwargs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        recording_executor = RecordingExecutor(executor)
        layer.batched_weight_loader(
            [
                ((target, source), {})
                for target, source in zip(independent_targets, independent_sources)
            ]
            + [((overlap_target, first), {}), ((overlap_target, second), {})],
            executor=recording_executor,
        )

    assert recording_executor.submissions > 0
    for target, source in zip(independent_targets, independent_sources):
        torch.testing.assert_close(target, source)
    torch.testing.assert_close(overlap_target, second)


def test_batched_copy_workers_share_cpus_between_local_model_workers():
    parallel = mock.Mock(
        world_group=mock.Mock(local_size=4, world_size=4),
        nnodes=1,
        enable_dp_attention=False,
        dp_size=1,
    )
    with (
        mock.patch(
            "sglang.srt.layers.moe.fused_moe_triton.layer.os.sched_getaffinity",
            return_value=set(range(80)),
        ),
        mock.patch.object(torch.distributed, "is_initialized", return_value=True),
        mock.patch(
            "sglang.srt.layers.moe.fused_moe_triton.layer.get_parallel",
            return_value=parallel,
        ),
        mock.patch(
            "sglang.srt.layers.moe.fused_moe_triton.layer."
            "envs.SGLANG_SET_CPU_AFFINITY.get",
            return_value=False,
        ),
    ):
        assert _batched_weight_copy_workers() == 20


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
