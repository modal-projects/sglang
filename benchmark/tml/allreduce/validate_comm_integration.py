"""Integration check of tml/kernels/comm.py custom-AR dispatch.

Drives get_ar_buffer + symm_mem_all_reduce (the paths the model actually calls)
through a real TorchSymmMemCommunicator with SGLANG_OPT_USE_INKLING_CUSTOM_AR=1:
resource build (flags barrier, msgspec struct), buffer enlargement, the
v4/v3/v2/mm dispatch buckets, v4 rotation, and the producer-writes-into-buffer
(input_is_ar_buffer) fast path.

Run: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \
    benchmark/tml/allreduce/validate_comm_integration.py
"""

import os

os.environ["SGLANG_OPT_USE_INKLING_CUSTOM_AR"] = "1"

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.torch_symm_mem import (
    TorchSymmMemCommunicator,
)
from sglang.tml.kernels import comm as inkling_comm

HIDDEN = 6144


class FakeGroup:
    """Minimal GroupCoordinator stand-in for symm_mem_all_reduce."""

    def __init__(self, comm):
        self.torch_symm_mem_comm = comm
        self.world_size = dist.get_world_size()

    def all_reduce(self, t):
        dist.all_reduce(t)
        return t


def main():
    rank = int(os.environ["LOCAL_RANK"])
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    comm = TorchSymmMemCommunicator(dist.group.WORLD, dev)
    assert not comm.disabled, "symm-mem communicator disabled on this box"
    assert comm.max_size >= 256 * 1024 * 1024, "buffer not enlarged by env flag"
    group = FakeGroup(comm)

    def pattern(tk, salt, r=rank):
        i = torch.arange(tk * HIDDEN, device=dev, dtype=torch.float32)
        return ((((i + salt) % 9) - 4) * (r + 1)).to(torch.bfloat16).view(tk, HIDDEN)

    def expected(tk, salt):
        scale = world * (world + 1) // 2
        i = torch.arange(tk * HIDDEN, device=dev, dtype=torch.float32)
        return ((((i + salt) % 9) - 4) * scale).to(torch.bfloat16).view(tk, HIDDEN)

    used = {}
    # Repeats of the small sizes exercise the v5 A/B staging rotation; the mix
    # exercises every dispatch bucket (v5 / mm / v3b / v3) plus bucket
    # switching in one stream.
    for salt, tk in enumerate(
        [1, 2, 1, 2, 3, 8, 32, 64, 96, 192, 256, 512, 1024, 3072, 4096, 1, 2, 16384]
    ):
        buf = inkling_comm.get_ar_buffer(group, tk, HIDDEN, torch.bfloat16)
        if buf is not None:
            buf.copy_(pattern(tk, salt))
            out = inkling_comm.symm_mem_all_reduce(buf, group, input_is_ar_buffer=True)
        else:
            out = inkling_comm.symm_mem_all_reduce(pattern(tk, salt), group)
        bad = (out != expected(tk, salt)).sum().item()
        assert bad == 0, f"rank{rank} tokens={tk} salt={salt}: {bad} mismatches"
        used[tk] = "ar_buffer" if buf is not None else "staged"
    torch.cuda.synchronize(dev)
    dist.barrier()
    if rank == 0:
        print(f"comm.py integration OK; paths: {used}", flush=True)
        print("ALL_OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
