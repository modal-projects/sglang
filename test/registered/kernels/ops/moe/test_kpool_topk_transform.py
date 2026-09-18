import torch

from sglang.kernels.ops.moe.kpool_topk_transform import (
    fast_kpool_topk_transform_fused,
)
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_noncanonical_nan_payload_is_memory_safe():
    rows, width = 3840, 960
    pool_size, topk = 4, 2048
    seq_lens = torch.arange(1, rows + 1, dtype=torch.int32, device="cuda")
    lengths = torch.div(seq_lens, pool_size, rounding_mode="floor")

    score = torch.full((rows, width), float("-inf"), dtype=torch.float32, device="cuda")
    valid = torch.arange(width, device="cuda").view(1, -1) < lengths.view(-1, 1)
    score.view(torch.int32)[valid] = 0x7FFFFFFF

    output = fast_kpool_topk_transform_fused(
        score,
        lengths,
        pool_size,
        topk,
        seq_lens=seq_lens,
    )
    torch.cuda.synchronize()

    selected_groups = output[:, :topk:pool_size] // pool_size
    rank = torch.arange(topk // pool_size, device="cuda").view(1, -1)
    selected = rank < lengths.clamp(max=topk // pool_size).view(-1, 1)
    assert torch.all(selected_groups[selected] >= 0)
    assert torch.all(
        selected_groups[selected] < lengths.view(-1, 1).expand_as(selected)[selected]
    )


if __name__ == "__main__":
    test_noncanonical_nan_payload_is_memory_safe()
